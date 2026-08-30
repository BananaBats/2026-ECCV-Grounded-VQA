# Verification record

The GitHub bundle was verified against the original final-run workspace.

| Check | Result |
|---|---|
| Unit tests | 11 passed |
| Shell syntax | all 7 launch/restore scripts passed `bash -n` |
| Saved plan archive | 1,859 plans, gzip integrity passed |
| Sparse archive | 1,857 predictions, gzip integrity passed |
| Known sparse fallback | exactly 2: `video_7360_q5`, `video_7567_q5` |
| Final submission structure | 932 videos, 1,859 questions, 3,206 tracks, 1,886,126 boxes |
| Finalizer reproduction | byte-identical to zero-shot fused source |
| Router reproduction | 504 routed, 1,355 base; byte-identical to final submission |
| GitHub file limit | no file exceeds 95 MB; largest file is below 43 MB |
| Credential/path scan | no workspace path, cloud project ID, or credential path in code/scripts/tests |
| Excluded-code scan | no amodal-training implementation in code/scripts/tests |

Canonical uncompressed SHA-256 values:

```text
zero-shot dense:
0134287c22cbbee313770e5efbfcf847acf61cb1382f5b420444f5ca77c36f63

zero-shot fused:
cf7e1dac04663f367e3d1d69f90a132382239d48ee6bbc0100b4fa6166f489f4

final routed submission:
688e84e966947bdf47ceb20f504737b4ee43680784815b732c9b3e0339d6d996
```
