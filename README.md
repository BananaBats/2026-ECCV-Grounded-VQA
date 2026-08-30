# Gemini-SAM3

Code and reproducibility artifacts for **Gemini-SAM3: Semantic Planning and
Amodal Tracking for Grounded Video Question Answering**.

The released zero-shot path separates semantic planning from dense tracking:

1. Gemini 3.7 Flash creates stable target identities, visibility segments, and
   anchor boxes.
2. Official zero-shot SAM3 replays the saved plan at stride 3 with dense
   boundary windows and segment gating.
3. Gemini 3.7 Flash predicts sparse amodal boxes every 30 source frames.
4. Reliability-gated fusion combines sparse semantic boxes with genuine sampled
   SAM3 support.
5. A fixed validation-derived rule routes eligible multi-target questions.

## Scope

This bundle intentionally excludes all amodal-training code, trained residual
heads, training data, training checkpoints, and alternate-branch inference
implementation. The final routed prediction is preserved as an exact compressed
artifact. The generic router is included and accepts an externally supplied
alternate prediction, but that alternate branch is not part of this repository.

The zero-shot planner, SAM3 replay, sparse Gemini stage, fusion, routing,
validation, saved test plans, successful sparse predictions, reports, and
submission artifacts are included. See [SOURCE_AUDIT.md](SOURCE_AUDIT.md) for
the complete provenance audit and explicit exclusions.

## Results

| Submission | Test score |
|---|---:|
| pipeline-v7 | 0.6263 |
| pipeline-v7 + Gemini sparse fusion | 0.6620 |
| amodal-trained dense branch | 0.6448 |
| final validation-routed fusion | **0.6637** |

The full method and interpretation are in
[docs/technical_report.md](docs/technical_report.md).

## Repository layout

```text
src/pipeline/          Gemini planning and zero-shot SAM3 replay
src/sparse_fusion/     30-frame Gemini inference and v13-style fusion
src/routing/           deterministic router and submission validator
references/            v13 fusion source used by the parity test
scripts/               portable stage launchers
tests/                 unit tests
config/test.tsv        1,859 test question keys
inputs/                packed saved plans (1,859) and sparse outputs (1,857)
artifacts/             checksums, summaries, compressed submissions
docs/                  technical report in Markdown and HTML
```

The two sparse failures, `video_7360_q5` and `video_7567_q5`, used the zero-shot
dense prediction directly. This behavior is implemented by
`finalize_fusion.py` and recorded in `artifacts/fusion_final_summary.json`.

## Environment

Python 3.11 was used. Install the lightweight dependencies:

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Planning and sparse inference require `ffmpeg`, Vertex AI credentials, and a
Google Cloud project. Dense replay additionally requires a compatible CUDA
PyTorch environment, the official SAM3 source checkout, and the official
`sam3.pt` checkpoint. These third-party files are not vendored.

Expected dataset layout:

```text
<VQA_PROJECT_ROOT>/dataset/test/
  annotations/grounded_question_test.json
  videos/*.mp4
```

## Reproduce the zero-shot path

The exact plans and sparse predictions are already included, so the costly
Gemini calls can be skipped.

```bash
scripts/unpack_inputs.sh
export VQA_PROJECT_ROOT=/path/to/project
export SAM3_ROOT=/path/to/official/sam3
export SAM3_CHECKPOINT=/path/to/sam3.pt
CUDA_VISIBLE_DEVICES=0,1,2,3 scripts/run_dense_tracking.sh

SPARSE_ROOT=sparse_predictions/test scripts/run_fusion.sh
```

To regenerate plans or sparse boxes:

```bash
export VQA_PROJECT_ROOT=/path/to/project
export GOOGLE_CLOUD_PROJECT=your-project
export GOOGLE_GENAI_USE_VERTEXAI=true

scripts/run_planning.sh
scripts/run_sparse.sh
```

To apply the fixed router when an excluded alternate prediction is available:

```bash
ALTERNATE_FUSED_PREDICTIONS=/path/to/alternate/predictions_or_shards \
  scripts/run_route.sh
```

## Final artifact

`artifacts/final_predictions.json.gz` is a byte-exact gzip container of the
submitted JSON. Restore and verify it with:

```bash
scripts/unpack_final_predictions.sh
sha256sum -c artifacts/SHA256SUMS
cd artifacts && sha256sum -c UNCOMPRESSED_SHA256SUMS && cd ..
python src/routing/validate_submission.py \
  artifacts/final_predictions.json --expected-questions 1859
```

The uncompressed JSON is intentionally not committed because it exceeds
GitHub's 100 MB per-file limit.

## Tests

```bash
python -m unittest discover -s tests -p 'test_*.py'
bash -n scripts/*.sh
```

Recorded end-to-end verification results are in
[VERIFICATION.md](VERIFICATION.md).
