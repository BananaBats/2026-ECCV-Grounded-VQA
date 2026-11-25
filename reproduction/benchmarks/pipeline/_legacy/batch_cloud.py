"""Small Vertex Batch/GCS transport for registered SVAG sparse requests.

This module does not make requests on import. Callers must journal a submit
intent before submit(): job creation has no automatic retry or idempotency key.
Use list_jobs_by_display_name() to reconcile an uncertain create response.

References:
https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/capabilities/batch-inference/new-job-from-cloud-storage
https://docs.cloud.google.com/storage/docs/json_api/v1/objects/insert
https://docs.cloud.google.com/storage/docs/json_api/v1/buckets/insert
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import quote
import uuid


MODEL = 'gemini-3.7-flash'
STORAGE = 'https://storage.googleapis.com/storage/v1'
UPLOAD = 'https://storage.googleapis.com/upload/storage/v1'


class CloudTransportError(RuntimeError):
    def __init__(self, operation, code, body):
        message = safe_error(body)
        super().__init__(f'{operation}: HTTP {code}: {message}')
        self.operation = operation
        self.code = self.status_code = code
        self.response_body = message


def safe_error(value):
    text = str(value)[:8000]
    text = re.sub(r'(?i)Bearer\s+[^\s\"\']+', 'Bearer [redacted]', text)
    return re.sub(
        r'(?i)(access_token|refresh_token|client_secret)([\"\']?\s*[:=]\s*[\"\']?)[^\s,\"\']+',
        r'\1\2[redacted]', text)


def parse_gs_uri(uri):
    if not isinstance(uri, str) or not uri.startswith('gs://'):
        raise ValueError('Expected a gs://bucket/object URI')
    bucket, slash, key = uri[5:].partition('/')
    if not bucket or not slash or not key:
        raise ValueError('Expected a nonempty bucket and object/prefix')
    return bucket, key


def _crc32c_table():
    table = []
    for byte in range(256):
        value = byte
        for _ in range(8):
            value = (value >> 1) ^ (0x82F63B78 if value & 1 else 0)
        table.append(value)
    return tuple(table)


_CRC32C = _crc32c_table()


def file_digests(path, *, crc32c=False):
    sha256, md5, size, crc = hashlib.sha256(), hashlib.md5(), 0, 0xFFFFFFFF
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            sha256.update(chunk); md5.update(chunk); size += len(chunk)
            if crc32c:
                for byte in chunk:crc = _CRC32C[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    result = {'size': size, 'sha256': sha256.hexdigest(),
              'md5Hash': base64.b64encode(md5.digest()).decode('ascii')}
    if crc32c:
        result['crc32c'] = base64.b64encode((crc ^ 0xFFFFFFFF).to_bytes(4,'big')).decode('ascii')
    return result


def json_value(value):
    if hasattr(value, 'model_dump'):
        return value.model_dump(mode='json', exclude_none=True)
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, 'value'):
        return json_value(value.value)
    raise TypeError(f'Cannot serialize {type(value).__name__}')


class BatchCloud:
    """Keep cloud operations explicit, private, and safe to resume.

    New buckets use US multi-region, uniform access and public access
    prevention. Existing buckets must belong to the selected project. This
    class never broadens IAM permissions or makes objects public.
    """
    def __init__(self, credentials, project, location='global', bucket=None, *,
                 session=None, client=None, timeout=300):
        if not isinstance(project, str) or not re.fullmatch(r'[a-z][a-z0-9-]{4,61}[a-z0-9]', project):
            raise ValueError('Expected a Google Cloud project ID')
        self.project, self.location = project, location
        self.bucket = bucket or f'{project}-svag-sparse-batch'
        if not re.fullmatch(r'[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]', self.bucket):
            raise ValueError('Invalid Cloud Storage bucket name')
        if not isinstance(location, str) or not re.fullmatch(r'[a-z0-9-]+', location):
            raise ValueError('Invalid Vertex location')
        self.timeout = timeout
        self._project_number = None
        self._own_session, self._own_client = session is None, client is None
        if session is None:
            from google.auth.transport.requests import AuthorizedSession
            session = AuthorizedSession(credentials)
        self.session = session
        if client is None:
            from google import genai
            from google.genai import types
            client = genai.Client(vertexai=True, project=project, location=location,
                credentials=credentials,
                http_options=types.HttpOptions(api_version='v1', timeout=int(timeout*1000),
                    retry_options=types.HttpRetryOptions(attempts=1)))
        self.client = client

    def _request(self, method, url, operation, *, accepted=(200,), **kwargs):
        response = self.session.request(method, url, timeout=self.timeout, **kwargs)
        if response.status_code not in accepted:
            try:
                body = response.json()
            except (ValueError, TypeError):
                body = response.text
            raise CloudTransportError(operation, response.status_code, body)
        return response

    @property
    def project_number(self):
        if self._project_number is None:
            # Storage already knows which project owns each bucket. Avoid an
            # unrelated Cloud Resource Manager API dependency: listing with
            # project=... and finding this exact bucket proves membership.
            page = None
            while True:
                params = {'project': self.project,
                          'fields': 'items(name,projectNumber),nextPageToken'}
                if page:params['pageToken'] = page
                response = self._request('GET', STORAGE+'/b',
                    'verify sparse bucket project ownership', params=params)
                payload = response.json()
                for item in payload.get('items', []):
                    if item.get('name') == self.bucket:
                        number = str(item.get('projectNumber', ''))
                        if not number.isdigit():
                            raise ValueError('Storage API did not return a numeric project identifier')
                        self._project_number = number
                        return self._project_number
                page = payload.get('nextPageToken')
                if not page:
                    raise ValueError('Sparse Batch bucket is not listed in the selected project')
        return self._project_number

    def ensure_bucket(self, *, storage_location='US'):
        url = STORAGE+'/b/'+quote(self.bucket, safe='')
        response = self._request('GET', url, 'get sparse batch bucket', accepted=(200, 404))
        if response.status_code == 200:
            metadata = response.json()
            if str(metadata.get('projectNumber')) != self.project_number:
                raise ValueError('Sparse Batch bucket belongs to a different project')
            return metadata
        body = {'name': self.bucket, 'location': storage_location, 'storageClass': 'STANDARD',
                'iamConfiguration': {'uniformBucketLevelAccess': {'enabled': True},
                                     'publicAccessPrevention': 'enforced'}}
        response = self._request('POST', STORAGE+'/b', 'create sparse batch bucket',
            accepted=(200, 201, 409), params={'project': self.project}, json=body)
        if response.status_code == 409:
            # Another preparer may have created the same deterministic bucket.
            response = self._request('GET', url, 'verify concurrently created sparse bucket')
            metadata = response.json()
            if str(metadata.get('projectNumber')) != self.project_number:
                raise ValueError('Sparse Batch bucket name is owned by another project')
            return metadata
        metadata = response.json()
        number = str(metadata.get('projectNumber', ''))
        if number.isdigit():
            self._project_number = number
        return metadata

    def _object_metadata(self, bucket, key, *, allow_missing=False):
        response = self._request('GET', STORAGE+'/b/'+quote(bucket, safe='')+'/o/'+quote(key, safe=''),
                                 'get sparse batch object', accepted=(200,404) if allow_missing else (200,))
        return None if response.status_code == 404 else response.json()

    @staticmethod
    def _check_object(metadata, digests):
        # Composite GCS output objects omit MD5; their CRC32C is authoritative.
        checksum = 'md5Hash' if metadata.get('md5Hash') else 'crc32c'
        if (str(metadata.get('size')) != str(digests['size'])
                or not metadata.get(checksum)
                or metadata.get(checksum) != digests.get(checksum)):
            raise ValueError('Cloud object content differs from the immutable local artifact')

    def upload_file(self, local_path, object_name, content_type='application/octet-stream'):
        path = Path(local_path)
        if not isinstance(object_name, str) or not object_name or object_name.startswith('/'):
            raise ValueError('Expected a nonempty relative Cloud Storage object name')
        digests = file_digests(path)
        metadata = self._object_metadata(self.bucket, object_name, allow_missing=True)
        uploaded = False
        if metadata is None:
            with path.open('rb') as stream:
                response = self._request('POST', UPLOAD+'/b/'+quote(self.bucket, safe='')+'/o',
                    'upload sparse batch object', accepted=(200,201,412),
                    params={'uploadType':'media', 'name': object_name, 'ifGenerationMatch':'0'},
                    headers={'Content-Type': content_type, 'Content-Length': str(digests['size'])},
                    data=stream)
            if response.status_code == 412:
                metadata = self._object_metadata(self.bucket, object_name)
            else:
                metadata = response.json(); uploaded = True
        if not metadata.get('md5Hash'):
            digests = file_digests(path, crc32c=True)
        self._check_object(metadata, digests)
        return {**metadata, 'uri': f'gs://{self.bucket}/{object_name}',
                'sha256': digests['sha256'], 'uploaded': uploaded}

    def submit(self, input_uri, output_prefix, display_name, *, model=MODEL):
        """Create exactly one job. Caller must persist intent before invoking."""
        source_bucket, _ = parse_gs_uri(input_uri)
        output_bucket, _ = parse_gs_uri(output_prefix)
        if source_bucket != self.bucket or output_bucket != self.bucket:
            raise ValueError('Batch source and output must use the registered private bucket')
        if not isinstance(display_name, str) or not display_name or len(display_name) > 128:
            raise ValueError('Batch display name must be nonempty and at most 128 characters')
        from google.genai import types
        job = self.client.batches.create(model=model, src=input_uri,
            config=types.CreateBatchJobConfig(display_name=display_name,
                                             dest=output_prefix.rstrip('/')+'/'))
        return json_value(job)

    def get_job(self, name):
        return json_value(self.client.batches.get(name=name))

    def list_jobs_by_display_name(self, display_name):
        from google.genai import types
        found = []
        # JSON string escaping is appropriate inside the API's filter grammar.
        config = types.ListBatchJobsConfig(page_size=100,
            filter='displayName='+json.dumps(display_name))
        for job in self.client.batches.list(config=config):
            value = json_value(job)
            if value.get('display_name', value.get('displayName')) == display_name:
                found.append(value)
        return found

    def list_output(self, prefix_uri):
        bucket, prefix = parse_gs_uri(prefix_uri)
        if bucket != self.bucket:
            raise ValueError('Output prefix differs from the registered private bucket')
        items, page = [], None
        while True:
            params = {'prefix':prefix}
            if page:params['pageToken'] = page
            response = self._request('GET', STORAGE+'/b/'+quote(bucket, safe='')+'/o',
                                     'list sparse batch output', params=params)
            payload = response.json()
            for item in payload.get('items', []):
                if not item['name'].startswith(prefix):
                    raise ValueError('Storage list returned an object outside the requested prefix')
                items.append({**item, 'uri': f'gs://{bucket}/{item["name"]}'})
            page = payload.get('nextPageToken')
            if not page:return items

    def download_object(self, uri, local_path):
        bucket, key = parse_gs_uri(uri)
        if bucket != self.bucket:
            raise ValueError('Output object differs from the registered private bucket')
        path = Path(local_path); path.parent.mkdir(parents=True, exist_ok=True)
        metadata = self._object_metadata(bucket, key)
        generation = metadata.get('generation')
        temporary = path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
        response = None
        try:
            params = {'alt':'media'}
            if generation:params['generation'] = generation
            response = self._request('GET', STORAGE+'/b/'+quote(bucket, safe='')+'/o/'+quote(key, safe=''),
                'download sparse batch output', params=params, stream=True)
            with temporary.open('wb') as stream:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    if chunk:stream.write(chunk)
            digests = file_digests(temporary, crc32c=not bool(metadata.get('md5Hash')))
            self._check_object(metadata, digests)
            temporary.replace(path)
            return {'uri':uri, 'local_path':str(path.resolve()), 'bytes':digests['size'],
                    'sha256':digests['sha256'], 'generation':generation}
        finally:
            if response is not None:response.close()
            temporary.unlink(missing_ok=True)

    def close(self):
        if self._own_session:self.session.close()
        if self._own_client:
            close = getattr(self.client,'close',None)
            if callable(close):close()
