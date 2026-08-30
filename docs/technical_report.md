# Gemini-SAM3: Semantic Planning and Amodal Tracking for Grounded Video Question Answering

## Abstract

Grounded video question answering requires a system to interpret a natural-language question, identify the corresponding object instances, and localize each instance consistently throughout a video. This is particularly difficult when the requested object is small, temporarily occluded, visually similar to nearby objects, or visible only during a short interval. We introduce **Gemini-SAM3**, a modular framework that separates semantic target understanding from pixel-level temporal tracking. Gemini 3.7 Flash first converts the question and video into a structured tracking plan containing stable target identities, visibility intervals, anchor frames, and anchor boxes. The same plan drives two complementary localization paths: sparse amodal box prediction by Gemini and dense mask propagation by SAM3. A confidence-gated fusion procedure combines the semantic robustness of the sparse predictions with the temporal resolution of dense tracking, while a validation-derived selective router chooses between zero-shot and amodal-fine-tuned SAM3 branches for predefined question categories. On the official test set, the zero-shot pipeline scores 0.6263, sparse Gemini fusion improves it to 0.6620, the fine-tuned amodal tracker scores 0.6448, and the final routed train fusion achieves 0.6637.

## 1. Introduction

Grounded video question answering (GVQA) differs from conventional video object tracking because the objects to be tracked are not given directly as category labels or first-frame masks. Instead, a question may refer to an object through its role in an event, its interaction with another object, its order of appearance, or its membership in a set. The system must therefore solve two coupled problems: it must first determine *what* the question refers to and then maintain the identity and location of every referred instance over time.

This coupling creates several failure modes. A tracker may accurately follow the wrong object if the semantic interpretation is incorrect, while a vision-language model may identify the correct object but produce temporally sparse or spatially inconsistent boxes. Visible-mask trackers introduce an additional limitation under occlusion: their predictions describe only the currently visible pixels and may shrink, drift to the occluder, switch to a similar object, or disappear altogether. These observations motivate a division of labor in which a multimodal language model performs global semantic planning and amodal reasoning, while a segmentation tracker supplies high-resolution temporal propagation.

Gemini-SAM3 implements this division through a reusable semantic tracking contract. Gemini 3.7 Flash analyzes the full video and question once, assigns stable object identities, estimates when each target is visible, and selects informative anchor frames and boxes. SAM3 then replays this plan to produce dense tracks, and a second Gemini pass predicts sparse amodal boxes at fixed temporal intervals. Finally, the system fuses the two sources only when their identities and spatial trajectories agree, and uses a deterministic validation-derived routing rule to exploit the complementary behavior of zero-shot and amodal-fine-tuned SAM3.

### Problem formulation

Given a video \(V = \{I_t\}_{t=0}^{T-1}\) and a natural-language question \(q\), the task is to return a set of target tracks

\[
\mathcal{Y} = \{\tau_k\}_{k=1}^{K}, \qquad
\tau_k = \{(t, b_{k,t})\mid t \in \mathcal{T}_k\},
\]

where \(K\) is the number of physical object instances referred to by the question, \(b_{k,t}=[x_1,y_1,x_2,y_2]\) is a normalized bounding box for target \(k\) at frame \(t\), and \(\mathcal{T}_k\) is the interval over which that object should be tracked. The target identity must remain stable through viewpoint changes, interactions, disappearance and reappearance, and partial or full occlusion. Because the desired box is amodal, \(b_{k,t}\) should approximate the full spatial extent of the object rather than only its visible pixels whenever the object is occluded.

We factor the prediction into semantic planning, dense tracking, sparse amodal estimation, fusion, and routing:

\[
P = G_{\mathrm{plan}}(V,q), \quad
D^{(z)},D^{(a)} = S_{\mathrm{SAM3}}(V,P), \quad
A = G_{\mathrm{amodal}}(V,P),
\]

\[
F^{(z)}=\Phi(D^{(z)},A,P), \qquad
F^{(a)}=\Phi(D^{(a)},A,P), \qquad
\hat{\mathcal{Y}}=R(q,P,F^{(z)},F^{(a)}).
\]

Here, \(P\) is the fixed semantic plan, \(D^{(z)}\) and \(D^{(a)}\) are tracks from zero-shot and amodal-fine-tuned SAM3, \(A\) denotes Gemini's sparse amodal boxes, \(\Phi\) is the confidence-gated fusion operator, and \(R\) is the selective router.

### Contributions

Our main contributions are:

1. **Semantic target planning.** We convert an open-ended grounded question into a structured, tracker-independent plan with verified target noun phrases, stable instance IDs, visibility segments, and multi-anchor box prompts.
2. **Complementary sparse and dense tracking.** We combine Gemini's explicit amodal reasoning at sparse frames with SAM3's dense bidirectional mask propagation, preserving the strengths of each model instead of forcing either model to solve the entire task.
3. **Reliability-aware fusion.** We introduce target-level trust gates, residual alignment, and conservative temporal interpolation to prevent unreliable dense masks from overwriting semantically grounded sparse predictions.
4. **Amodal adaptation with minimal trainable capacity.** We adapt SAM3 using a lightweight residual mask-logit head trained on SAIL-VOS while freezing the original visible-tracking backbone, decoder, and memory pathway.
5. **Validation-derived selective routing.** We use a fixed, question-conditioned routing policy to choose the dense branch before test evaluation, improving the final test score without using test annotations or manual per-example selection.

## 2. Method

The complete pipeline has three conceptual stages. First, Gemini creates a semantic plan that fixes the meaning and identity of the requested targets. Second, two trackers operate from this shared plan: Gemini produces sparse amodal boxes, while SAM3 produces dense mask trajectories using either the official zero-shot checkpoint or the amodal-fine-tuned residual head. Third, target-level fusion and question-level routing construct the final dense bounding-box tracks.

### 2.1 Semantic Target Planning with Gemini

The planning stage is designed to prevent semantic ambiguity from propagating into tracking. Rather than asking Gemini to generate the final frame-by-frame track directly, we ask it to produce an invariant tracking contract

\[
P=\{q,\; \text{target IDs},\; \text{identity descriptions},\; \text{visibility segments},\; \text{anchor frames},\; \text{anchor boxes}\}.
\]

We first construct a low-cost video proxy by uniformly sampling at most 192 frames, resizing the longer side to at most 512 pixels, and burning the original frame index into each sampled frame. The proxy video preserves the global temporal context and the mapping back to source-frame indices; its frame rate is capped at 6 fps, and the Gemini video input is sampled at 4 fps. This allows the model to reason about long videos without losing the exact source coordinate system needed by downstream trackers.

Planning proceeds in several structured calls. The first call extracts a concise visual noun phrase describing the object or set of objects requested by the question. The prompt excludes people, hands, and arms unless they are themselves the answer, and asks for attributes that separate the target from distractors. A second, independent critic call solves the question again and marks the proposed noun phrase as correct or incorrect. If it is rejected, the corrected phrase and the expected number of physical instances are passed to the anchor planner.

The anchor planner assigns a stable integer ID \(k\in\{0,\ldots,K-1\}\) to each physical target and records a visual identity description and one or more visibility segments for that target. These segments represent intervals in which the target is visually confirmed; they are not interpreted as proof that the object is physically absent outside the interval. The planner prefers one frame in which all targets are clearly visible, but it may select up to three anchors when a single frame cannot cover every identity. Anchor candidates lie on a stride-10 source-frame grid, and an off-grid response is snapped to the nearest valid frame.

For spatial precision, each selected source frame is extracted at its exact frame index and sent back to Gemini. The model returns a tight box in the 0--1000 coordinate system for every target visible at that anchor. All planning calls use temperature 0, forced structured function calling, and schema validation. The resulting plan is saved once and replayed by every subsequent branch, ensuring that zero-shot SAM3, fine-tuned SAM3, and sparse Gemini tracking use identical target definitions and object IDs.

### 2.2 Sparse and Dense Video Tracking

#### 2.2.1 Sparse Tracking with Gemini

The sparse tracker predicts amodal boxes on the fixed source-frame grid \(t\in\{0,30,60,\ldots\}\). Frames are extracted by decoder frame index rather than timestamp seeking, which keeps them aligned with the plan and the evaluator. Up to eight images, together with a compact representation of the saved plan, are processed in one Gemini 3.7 Flash call. In this stage the plan is not reinterpreted: the prompt requires every prediction to use a known target ID and to preserve the physical identity established during planning.

Gemini is explicitly instructed to predict one object per box and to exclude hands, occluders, and unions of multiple targets. For a partially occluded target, it estimates the complete extent of the object behind the occluder; for a fully occluded target, it uses the surrounding frames to infer the most plausible location and size of the target itself. An ID may be omitted only when the object has actually left the scene, not merely because it is hidden or falls outside a visibility segment. Predictions are accepted only if their frame order, ID set, ID uniqueness, required visible-instance coverage, coordinate range, and positive area pass strict validation; invalid responses and transient API failures are retried up to three times.

The sparse output is deliberately not densified by unconditional endpoint holding. It serves as a semantically grounded amodal trajectory and as a reference against which dense SAM3 tracks are validated. This makes the sparse branch especially useful during occlusion, where a visible-mask tracker may shrink to a fragment or drift toward a visually dominant occluder.

#### 2.2.2 Dense Video Tracking with SAM3

The zero-shot dense branch uses the official SAM3 PVS checkpoint without additional training. It replays the target IDs, anchor frames, and anchor boxes from the Gemini plan and therefore requires no further language-model call. The sampled frame set contains the regular source grid \(0,3,6,\ldots\), all anchor frames, the final video frame, the endpoints of every visibility segment, and stride-1 windows of radius 15 around important visibility boundaries.

All target boxes are registered in a single multi-object SAM3 session with their stable plan IDs. After conditioning, the tracker performs one bidirectional propagation pass. The propagation starts from the non-terminal anchor closest to the temporal center of the sampled video; terminal anchors are avoided when alternatives exist because starting at the last sampled frame can prevent masks from propagating beyond the conditioning frame. Joint propagation helps preserve relative identities when several visually similar objects coexist.

SAM3 outputs are gated independently for each object. Masks outside that object's planned visibility segments, expanded by a 15-frame margin, are removed to reduce persistence onto a similar object after the true target disappears. Non-empty masks are converted to normalized XYXY boxes while still in memory. The standalone pipeline-v7 export fills the gaps between sampled frames to satisfy the dense submission format, but the later fusion stage reconstructs the actual sampled coverage before deciding whether a SAM3 box is trustworthy.

The amodal-trained dense branch receives exactly the same Gemini anchors and boxes but runs each target in an independent pass. Forward and backward propagation maintain separate persistent tracking states, and eight-frame chunks limit feature-memory computation without resetting the tracker or adding new prompts at chunk boundaries. A visible-memory skip guard writes a non-conditioning frame to memory only when the visible mask is non-empty, its presence probability is at least 0.5, and its predicted IoU is at least 0.5. The amodal prediction may still be emitted on an unreliable frame, but it cannot contaminate the visible memory used for later frames.

### 2.3 Selective Routing

We first fuse sparse Gemini boxes with each dense SAM3 candidate using the same sparse-first procedure. Because the compact pipeline-v7 output contains filled boxes at every frame, we recover an approximation of its genuine SAM3 support from the intersection of non-empty sampled frames and the segment-gating intervals. A dense target is trusted only if it satisfies

\[
\operatorname{IoU}(b_{\mathrm{seed}},b_{\mathrm{SAM}})\geq0.25,
\quad N_{\mathrm{shared}}\geq3,
\quad \frac{1}{N_{\mathrm{shared}}}\sum_t
\operatorname{IoU}(b^{A}_t,b^{D}_t)\geq0.30.
\]

If this gate fails, the fused track retains the Gemini sparse trajectory. If it passes, sparse boxes are placed first and genuine sampled SAM3 boxes are overlaid. On shared frames with IoU at least 0.30, we estimate the residual \(r(t)=b^A_t-b^D_t\) and interpolate this residual over the SAM3 trajectory; a one-sided residual anchor is propagated for at most 60 frames. Remaining gaps are linearly interpolated only when their length is at most six frames or their endpoint boxes have IoU of at least 0.30. These restrictions prevent a locally incorrect dense track from being extrapolated through a long disappearance.

This process produces two question-level candidates: \(F^{(z)}\), obtained from zero-shot pipeline-v7 and Gemini sparse boxes, and \(F^{(a)}\), obtained from amodal-trained SAM3 and the same sparse boxes. The final router is a deterministic rule selected using a held-out validation subset:

\[
R(q,P)=
\begin{cases}
F^{(a)}, & |P.\mathrm{targets}|>1,\ q\notin\text{occlusion-game},\ q\neq\text{appear-twice},\\
F^{(z)}, & \text{otherwise}.
\end{cases}
\]

In the final test run, 504 multi-target questions satisfying this rule used the amodal-trained fusion, while the remaining 1,355 questions retained the zero-shot fusion. Importantly, this was not a manual choice made after inspecting individual test predictions or test scores. The rule depends only on the question text and the number of targets in the precomputed plan, was fixed from validation behavior, and was then applied uniformly to the test set without access to test annotations.

### 2.4 Amodal SAM3 Fine-tuning

We train on the official SAIL-VOS RGB frames, native visible instance masks, and object-wise amodal masks, using 10 training sequences and four disjoint validation sequences organized as shot-aware causal clips of eight frames. The SAM3 backbone, visible decoder, proposal decoder, and memory path remain frozen, while a lightweight residual head combines image features, visible logits, and proposal logits to predict a bounded correction that is added to the visible mask logits. We optimize only this residual head for 20 epochs at 1008-pixel resolution with AdamW, a cosine learning-rate schedule from \(10^{-4}\) to \(10^{-5}\), batch size 1, BF16 execution for frozen SAM3, and FP32 computation for the residual branch. The objective is \(5\mathcal{L}_{\mathrm{focal}}+5\mathcal{L}_{\mathrm{dice}}+5\mathcal{L}_{\mathrm{containment}}\), where fully occluded objects remain positive amodal targets, ignored regions are excluded, and the containment term encourages the visible mask to be a subset of the amodal mask. At inference, the epoch-20 checkpoint uses the same Gemini plan as pipeline-v7 and performs continuous bidirectional tracking with the visible-memory skip guard described above.

## 3. Results

The following table reports the official test-set submission results supplied for the four system variants. All scores are shown on the same evaluation scale, and higher is better.

| Submission | Main components | Test score | Absolute gain over pipeline-v7 |
|---|---|---:|---:|
| pipeline-v7 | Gemini semantic plan + zero-shot SAM3 dense tracking | 0.6263 | -- |
| fusion | pipeline-v7 + Gemini 3.7 Flash sparse amodal boxes with v13-style fusion | 0.6620 | +0.0357 |
| amodal_train | Gemini semantic plan + amodal-fine-tuned SAM3 | 0.6448 | +0.0185 |
| **train fusion** | Selective routing between zero-shot fusion and amodal-trained fusion | **0.6637** | **+0.0374** |

Fine-tuning SAM3 for amodal prediction improves the standalone score from 0.6263 to 0.6448, an absolute gain of 0.0185. This confirms that learning from visible/amodal mask pairs helps the dense tracker preserve object extent under occlusion. However, `amodal_train` remains 0.0172 below the 0.6620 fusion score. The comparison suggests that the learned residual head improves dense tracking but does not yet match Gemini's explicit amodal box reasoning at difficult sparse frames; the latter can use broader temporal and semantic context when the visible evidence is weak.

The large gain from 0.6263 to 0.6620 also shows that the sparse and dense branches are complementary. SAM3 supplies smooth, high-frequency localization when the target is visible and its track is reliable, whereas Gemini provides identity-aware boxes during occlusion and other failure intervals. The trust gate is important here because the improvement comes from selectively overlaying reliable dense evidence, not from blindly averaging two potentially inconsistent trajectories.

The final `train fusion` result reaches 0.6637, improving over the zero-shot fusion by a further 0.0017 and producing the best score among the submitted variants. A defensible interpretation is that amodal training is beneficial for a subset of question structures, especially eligible multi-target cases, but is not uniformly better for every question. The router captures this specialization using a rule derived on held-out validation data and based only on observable question/plan attributes. Therefore, the result should be described as **validation-based deterministic routing**, not as manual selection of whichever method happened to perform better on each test question. The relatively small final gain also indicates that the two fused systems agree on most cases and that the amodal-trained branch contributes mainly on a limited but useful subset.

## 4. Conclusion

We presented Gemini-SAM3, a grounded video question answering framework that separates semantic target definition, sparse amodal reasoning, and dense video tracking. Gemini 3.7 Flash converts each question and video into a stable multi-object tracking plan and later supplies identity-preserving sparse amodal boxes, while SAM3 propagates the planned targets densely through time. Reliability-aware fusion resolves the mismatch between Gemini's sparse temporal resolution and SAM3's vulnerability to occlusion and drift. Amodal fine-tuning improves the standalone SAM3 branch, and a fixed validation-derived router allows that branch to complement the stronger zero-shot Gemini fusion on selected question categories. The final train-fusion submission achieves the best test score of 0.6637, demonstrating that semantic planning, amodal reasoning, dense propagation, and selective model routing are complementary components for grounded tracking in challenging videos.
