"""Exact local JSON-schema expansion from the original Batch planner."""

def inline_json_schema(schema):
    """Expand local JSON-schema refs before Batch's BigQuery input import.

    The online API accepts Pydantic's $defs/$ref representation, but Batch
    imports JSON object keys as column names and rejects the dollar signs.
    Keep every constraint and validate responses with the original models.
    """
    def expand(value, stack=()):
        if isinstance(value, list):
            return [expand(item, stack) for item in value]
        if not isinstance(value, dict):
            return value
        result = {}
        if '$ref' in value:
            ref = value['$ref']
            if not isinstance(ref, str) or not ref.startswith('#/'):
                raise ValueError('Only local JSON-schema references are supported')
            if ref in stack:
                raise ValueError('Recursive JSON schema cannot be inlined for Batch')
            target = schema
            for token in ref[2:].split('/'):
                target = target[token.replace('~1', '/').replace('~0', '~')]
            result = expand(target, (*stack, ref))
        for key, item in value.items():
            if key in {'$ref', '$defs', 'definitions', '$schema'}:
                continue
            if key.startswith('$'):
                raise ValueError('Unsupported Batch JSON-schema keyword: ' + key)
            result[key] = expand(item, stack)
        return result

    return expand(schema)
