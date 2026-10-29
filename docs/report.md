# Llama-3-EDDY-Qwen3-Embedding-8B: a linear adapter from Qwen3-Embedding-8B to ARDY's text space

**Built with Meta Llama 3.** EDDY is the project and code name. The published model is named Llama-3-EDDY-Qwen3-Embedding-8B, as the Meta Llama 3 Community License requires. See the [license copy](../LICENSE-LLAMA3.txt) and the [use policy](../USE_POLICY-LLAMA3.md).

Cyril Tsui, 2026-10-29. Technical report. Every number comes from the author's working log; estimates are marked.

## 1. Problem

ARDY (NVIDIA, 2026) generates human motion from text. Its text encoder is LLM2Vec on Llama 3 8B. In bf16 it needs about 14 GB of VRAM, and no provider hosts it. It is the heaviest part of ARDY at run time.

ARDY uses one pooled vector per prompt (mean pooling, shape `[B, 1, 4096]`, no token sequence). A single linear layer, `embed_text`, projects it to 1024 dimensions inside the model. An empty prompt is a zero vector (used for classifier-free guidance, weight 2.0 by default). Because the interface is one vector, another embedding model plus a learned map can take the place of LLM2Vec.

EDDY (Embedding aDapter for ARDY) is that map. The goal is the same motion as LLM2Vec gives, not only the same vector. LLM2Vec stays the teacher, because ARDY's behavior is defined only through LLM2Vec vectors.

**Scope.** ARDY-SOMA-RP-30FPS-Horizon60. Input: Qwen3-Embedding-8B, 4096 dimensions, no instruction, from any host.

## 2. Method

The shipped adapter is ridge regression (λ = 0.01), linear only. The input is L2-normalized. The output is standardized with the per-dimension mean μ and std σ of the LLM2Vec targets (train split only): `y = (W x̂ + b) · σ + μ`. The file is about 67 MB.

An MLP branch was tried (linear map plus a residual MLP, trained with SGD from the ridge solution, with an extra loss on the error after `embed_text`). It lowered the vector error on SEED validation (e_c median 0.315 for ridge, 0.299 for the best MLP). It did not move motion: in the validation motion check its R@3 was 78.0 against ridge's 78.2 (Section 5). The project rule is "motion decides" and "start simple", so ridge was kept. Other loss terms (cosine, similarity structure, rewording-mean target, through-the-denoiser) were not tried.

**Vector error.** e_c is the error after `embed_text`, divided by the distance of the target from the train mean (the mean offset makes plain relative error look small even for a useless prediction). The worse of the root and body `embed_text` weights is reported. It is a proxy: the 8-layer transformer after `embed_text` may be more sensitive in some directions.

## 3. Data

All targets are bf16 LLM2Vec vectors. Inputs are Qwen3-Embedding-8B vectors (bf16, last-token pooling, 4096 dimensions) of the same texts. Splits are by group, never by row: all texts of one motion (or one content group) stay in one split.

| Set | Source | Rows | Train / val / test |
| --- | --- | --- | --- |
| `data/full` | HumanML3D captions, 200 stream-style prompts, 30 fixed ARDY prompts | 58,208 | 44,512 / 5,682 / 5,629 |
| `data/seed` | Every overview and event text in SEED-Timeline-Annotations (BONES-SEED labels), split by the Kimodo benchmark's split files; plus 1,499 rewordings of 300 val labels (used only to set a reference level ε) | 67,667 | 47,410 / 6,069 / 3,615 (plus other held-out rows) |
| `data/join` | One text per SEED train or val motion: 2 or 3 consecutive event labels joined by a template ("A. Then, B.") | 22,000 | 20,000 / 2,000 / none |

The adapter trains on the train splits: 44,512 + 47,410 + 20,000 = 111,922 pairs (sum computed from the table).

- **Why SEED.** ARDY was trained on Bones Rigplay labels, and BONES-SEED is the public part of that data. SEED validation therefore decides every choice. HumanML3D is a different text style, which ARDY never saw.
- **No benchmark text in training.** A text that appears on a held-out motion and on a train motion is kept only in the held-out split. Test and test-repetition motions are never used for the joined texts.
- **Why joined texts.** The first benchmark run showed the largest loss on timeline-multi prompts, which join several events (median 28 words). Only 2.0 % of the earlier training texts had more than one sentence. The joined texts are template joins, while the benchmark's multi texts are LLM rewrites, so the styles differ.
- **Two GPUs.** The vectors of `data/full` and `data/seed` were computed on a 24 GB GPU; those of `data/join` locally on an RTX 4080 SUPER. A check on 100 texts gave e_c 0.011 (median) between the two, against about 0.32 for the adapter, so the difference was accepted.
- **Reading.** About 300 SEED labels were read by hand. Two validation labels with leaked annotation template were dropped.

## 4. Evaluation protocol

1. **Vector checks (safety net).** e_c, minimal-pair accuracy (left/right, slow/fast and similar; the prediction for A must be nearer to target A than to target B), and the worst prompts read by hand.
2. **Validation motion check.** 500 SEED validation event texts with their durations. ARDY-SOMA-RP-30FPS-Horizon60 generates one motion per text, with seeds 0 to 4, for each text encoder. Stored vectors stand in for the encoder, so LLM2Vec and Qwen are not loaded. TMR-SOMA-RP-v1 scores R@1, R@3 and R@10 with the 500 texts as the gallery. The paired R@3 difference (adapter minus LLM2Vec, same seed) is the sharper test, because both runs share the sampling noise. FID is unpaired: it is measured against LLM2Vec seed 0 using seeds 1 to 4 only. Pass: R@3 within the LLM2Vec seed std, FID not clearly above LLM2Vec's.
3. **Kimodo benchmark, run once.** The content split, text-to-motion cases (overview 917, timeline single 917, timeline multi 789), rebuilt for ARDY's motion representation from BONES-SEED. The same cases, per-case seeds and scoring for LLM2Vec bf16 and the adapter, batch size 1. Our ground-truth row matches NVIDIA's within 0.4 R@3. The test split is used once; every later choice was made on validation.
4. **End to end.** ARDY with `EddyTextEncoder` and live Qwen3-Embedding-8B, and a hosted-versus-local comparison with OpenRouter.

Reference level ε (not a pass bar): the median e_c between a label and its five LLM rewordings, ε = 0.279. The motion check showed that ε is stricter than ARDY: ridge misses it (0.313) and still has motion within the seed noise.

## 5. Results

### Vector error

| Model (SEED val) | e_c median | e_c p95 | Minimal pairs |
| --- | --- | --- | --- |
| Predict the mean | 1.000 | 1.000 | 0.000 |
| Procrustes | 0.453 | 0.592 | 0.993 |
| Ridge, λ = 0.01 | 0.315 | 0.458 | 0.997 |
| Linear + MLP, ARDY-space loss | 0.299 | 0.459 | 0.998 |

(This table used the maximum over four `embed_text` weights from two checkpoints; later numbers use only SOMA-RP-30FPS-Horizon60's.) On the test split, the adapter before the joined texts gave e_c 0.320 on SEED (validation 0.321), 0.361 on HumanML3D (validation 0.361) and 0.332 on stream prompts. Minimal pairs: 0.998 and 0.994. The ridge learning curve was still falling slowly at 92k pairs.

### Validation motion check (500 SEED validation prompts, 5 seeds)

| Text encoder | R@1 | R@3 | Paired R@3 vs. LLM2Vec | FID |
| --- | --- | --- | --- | --- |
| LLM2Vec bf16 | 57.6 ± 1.0 | 79.1 ± 2.1 | n/a | 0.047 ± 0.006 |
| Ridge (before joined texts) | 56.0 ± 0.6 | 78.2 ± 1.0 | −0.9 ± 1.6 | 0.056 ± 0.005 |
| MLP + ARDY-space loss (not kept) | 56.2 ± 1.5 | 78.0 ± 1.6 | −1.1 ± 1.0 | 0.054 ± 0.004 |
| Final adapter, local Qwen | 55.7 ± 1.0 | 78.1 ± 1.0 | −1.0 ± 2.2 | 0.055 ± 0.005 |
| Final adapter, OpenRouter Qwen | 55.8 ± 1.1 | 77.2 ± 1.6 | −1.9 ± 1.1 | 0.054 ± 0.005 |

Both adapters pass the rule. The drop looks small but probably real (R@3 lower in 4 of 5 seeds for the MLP). The hosted run is within noise of the local run (paired R@3 −0.9 ± 1.6), but its R@3 was lower than LLM2Vec's in all 5 seeds, so it is at the edge.

### Joined multi-event prompts (500 joined validation prompts, 5 seeds)

| Text encoder | R@1 | R@3 | R@10 | FID |
| --- | --- | --- | --- | --- |
| LLM2Vec bf16 | 69.2 ± 1.2 | 87.2 ± 1.4 | 97.0 ± 0.9 | 0.032 |
| Ridge before joined texts | 64.6 ± 1.6 | 83.2 ± 1.3 | 95.2 ± 0.6 | 0.042 |
| Ridge with joined texts (final) | 66.9 ± 1.4 | 84.9 ± 0.8 | 95.8 ± 0.7 | 0.040 |

The final adapter beats the old one by 1.6 ± 1.4 R@3 (higher in 4 of 5 seeds) and 2.3 ± 1.1 R@1 (5 of 5). On SEED validation prompts it changes nothing (paired R@3 −0.1 ± 0.9). Joined-validation e_c fell from 0.427 to 0.366 while SEED validation stayed at 0.321 to 0.322. It was chosen on validation.

### Test prompts (500 SEED test texts, run once, adapter before joined texts)

| Text encoder | R@1 | R@3 | Paired R@3 | FID |
| --- | --- | --- | --- | --- |
| LLM2Vec bf16 | 52.6 ± 1.4 | 70.6 ± 1.2 | n/a | 0.072 ± 0.010 |
| Ridge | 50.6 ± 1.1 | 68.8 ± 1.3 | −1.8 ± 2.3 | 0.079 ± 0.008 |

### Kimodo benchmark (run once for LLM2Vec and the adapter before joined texts)

| | Overview R@3 | FID | Timeline single R@3 | FID | Timeline multi R@3 | FID |
| --- | --- | --- | --- | --- | --- | --- |
| Ground truth | 88.88 | 0 | 86.04 | 0 | 88.21 | 0 |
| LLM2Vec (ours) | 75.90 | 0.060 | 64.78 | 0.046 | 77.19 | 0.044 |
| Ridge before joined texts | 73.94 | 0.066 | 64.34 | 0.046 | 71.23 | 0.052 |
| Difference | −2.0 | +0.006 | −0.4 | 0 | −6.0 | +0.008 |
| LLM2Vec (NVIDIA, published) | 77.86 | 0.058 | 63.90 | 0.045 | 76.43 | 0.046 |

Per-case counts (cases in the top 3 only with LLM2Vec vs. only with the adapter): overview 109 vs. 78 (z = 2.27), timeline single 75 vs. 66 (z = 0.76, no clear difference), timeline multi 109 vs. 65 (z = 3.34). Our LLM2Vec overview score is 1.9 below NVIDIA's published one (protocol and seeds differ: batch 1 versus 32, bf16 versus fp32).

**Second look (timeline multi only; not used for any choice).** After the final adapter (with joined texts) was chosen on validation, the 789 timeline-multi cases were run once more with it. Overview and timeline single were not run again.

| Timeline multi | R@3 | FID | Skate | Contact |
| --- | --- | --- | --- | --- |
| LLM2Vec (ours) | 77.19 | 0.044 | 2.455 | 0.986 |
| Adapter before joined texts | 71.23 | 0.052 | 2.403 | 0.986 |
| Final adapter | 73.38 | 0.051 | 2.422 | 0.986 |

The gap shrinks from −6.0 to −3.8 R@3 and is still real (per case 97 vs. 63, z = 2.69). Final versus old is 38 vs. 28 cases (z = 1.23), which alone is not a clear difference, but it has the same direction and size as on validation.

### End to end and hosted Qwen

- ARDY ran with `load_model(text_encoder=EddyTextEncoder(...))` and ARDY's own `generate_eval.py` on 45 prompts (30 fixed, 10 stream prompts, 5 joined). The vectors ARDY received equal the adapter's offline output (maximum difference 1.2e-5). R@3 94.2 ± 3.0 (adapter) against 95.1 ± 1.7 (LLM2Vec), 5 seeds. With the same seed, adapter and LLM2Vec motions are closer to each other (TMR cosine median 0.97) than two LLM2Vec motions with different seeds (0.87). The author watched all 45 side-by-side videos: some pairs differ, none clearly in favor of one side. This was one viewer, not blinded.
- ARDY with the adapter peaked at 0.96 GiB of VRAM and generated the 45 motions in 16 s. EDDY loads in 0.3 s. Local Qwen3-Embedding-8B needs 14.1 GiB and does not fit beside ARDY on a 16 GB GPU.
- OpenRouter's `qwen/qwen3-embedding-8b` gives vectors close to local ones: cosine median 0.998 on one day and 0.999 on the next, minimum 0.9958 on both. It also gives slightly different vectors from call to call (minimum cosine 0.9967 between the two days). Adapter e_c median: 0.320 on local, 0.329 on hosted vectors. One prompt took a median of 3.7 s (maximum 38 s over 10 calls); a batch of 50 took about 9 s.
- Local Qwen3-Embedding-8B on the RTX 4080 SUPER matched the training inputs (cosine ≥ 0.99982).

## 6. Limitations

- **Information the input lacks cannot be added.** EDDY cannot recover detail that Qwen's vector does not carry.
- **Multi-event prompts** still lose 3.8 R@3 on the benchmark. The training joins are templates; the benchmark texts are LLM rewrites.
- **Overview prompts** lose 2.0 R@3, with the adapter before joined texts. It was not re-measured with the final adapter.
- **Prompt style.** Imperatives, first and second person, one-word prompts, negation and durations in the text did not occur in SEED. They are out of scope; the answer is to write prompts in the SEED style. The effect on motion was not measured.
- **Coverage.** Vector error rises with distance from the training texts (SEED val e_c median 0.29 for texts with a near-duplicate in train, 0.46 for the farthest group). The estimated gaps are larger hand gestures and mimes, and finger-level signs.
- **Hosted Qwen** is within noise of local, but at the edge of the pass rule.
- **Statistical power.** Five seeds and 500 prompts give differences of about 1 R@3 only as trends. The paired standard errors are 1 to 2 points.
- **One host.** Only OpenRouter was tested as a host.
- **Test set.** Test texts were used once (SEED test, benchmark). The final adapter (with joined texts) was not measured on the benchmark's overview and timeline-single cases or on the SEED test prompts.
- **Joined vectors.** `data/join` vectors come from a different GPU than the other data (difference accepted, e_c 0.011).

## 7. What was not done

- No CPU benchmark. Single runs from the demo (Ryzen 9 7900X, after warm-up): one 6 s motion took 2.8 s on CPU and 0.4 to 0.7 s on an RTX 4080 SUPER. The first ARDY load took 1.2 s on CPU and 10 to 14 s on GPU (weights cached). An embedding request through OpenRouter took 1.4 to 2.2 s.
- No test of other input models, other ARDY checkpoints, or other hosts.
- No loss terms beyond MSE and the ARDY-space loss, no random search, and no 3-seed final configuration for the linear model (ridge is deterministic).
- No comparison of fp32 and bf16 LLM2Vec targets (accepted as noise; NVIDIA reports about ±1 R@3).
- No folding of the adapter into ARDY's `embed_text` weights. The math was checked (maximum error 3.6e-7), but the folded model was not tested.
- No blinded visual study.
- The effect of the prompt style on motion was not measured.

## 8. Reproducing

See the main [README](../README.md) for code and the training command. Data folders and generated motions are not included. The author's run files hold the per-seed numbers behind the tables above.

## 9. License and attribution

EDDY's code, adapter weights and this report are under the Apache License 2.0 ([LICENSE](../LICENSE)); attributions are in [NOTICE](../NOTICE).

This is the author's reading of the licenses, not legal advice. Checked at the source on 2026-10-29. The table with all works is in the main [README](../README.md#third-party-licenses-and-attribution).

**Built with Meta Llama 3.** The training targets were produced by LLM2Vec, which runs Meta Llama 3 weights. The agreement is in [LICENSE-LLAMA3.txt](../LICENSE-LLAMA3.txt). The copies are from the `main` branch of github.com/meta-llama/llama3.

Training data includes Motion Data by Bones Studio. EDDY was not trained on Bones motion data. The motions were used only to run the benchmark evaluation. The training texts come from NVIDIA's SEED-Timeline-Annotations (CC BY 4.0).
