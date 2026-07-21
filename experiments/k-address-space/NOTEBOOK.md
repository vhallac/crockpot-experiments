# K-address-space lab notebook

## 2026-07-21 — K-address-space M1.5 v1.1 NoPE-GPT-Small CPU rerun prep

### Question / Hypothesis

Does the corrected M1.5 v1.1 repeated-segment probe reproduce the NoPE-GPT-Small Family A depth profile while also exercising the formerly empty Family B induction-control family and the corrected length sweep (`L ∈ {4, 7}`)?

### Experiment Design Summary

This rerun implements the v1.1 corrections in `experiments/k-address-space/addendum-M1.5.md` and the compatible local code-fix brief in `experiments/k-address-space/CODEFIX-M1.5.md` without committing that brief. The implementation removes Family B equal-length filtering, computes slot positions from cumulative offsets, derives `R_min = max(120, 2*d_head)`, defaults the token budget to trained context minus 32, emits feasibility/rejection manifest fields, changes the shuffled-null gate to an upper-tail check, and applies the variance floor to every derived position statistic.

Compatibility check: the code-fix D1–D5 items correspond directly to addendum v1.1 C1–C4 / §§2.0, 2.2, 3, 4.1, and 7.5. The implementation keeps CODEFIX as an untracked disposable file and treats the addendum as the durable spec.

### Planned Procedure

Run tests and a CPU smoke check, commit the pre-run state, then run the full corrected NoPE CPU experiment from the committed state:

```bash
PYTHONPATH=experiments/dead-keys:experiments/k-address-space ./scripts/nix-cpu-run -m kaddress.scripts.position_content \
  --model nope-gpt-small \
  --revision 320681e33a029517e27c68a0f9c2b07ea0004155 \
  --families A,B,C \
  --output-dir outputs/k_address_space_m15_v11_nope_gpt_small_cpu_20260721
```

### Expected Signal / Interpretation Plan

The regression bar is that Family A should remain close to the previous NoPE depth profile, while Family B must be non-empty. G1 must pass at layer 0, the corrected one-sided shuffled-null gate should avoid treating negative null R² as leakage, and degenerate rows should carry zero/NaN derived position statistics with `degenerate=true`. If Family A changes materially, the cumulative-offset or length-sweep refactor may have changed the measurement rather than only fixing controls.

### Pre-run Provenance

- Spec: `experiments/k-address-space/addendum-M1.5.md` v1.1
- Code-fix brief: `experiments/k-address-space/CODEFIX-M1.5.md` (untracked disposable file; do not commit)
- Code branch: `main`
- Pre-run commit: _Pending_
- Planned output location: `outputs/k_address_space_m15_v11_nope_gpt_small_cpu_20260721`
- Publication target: GitHub Release `run/k-address-space-m15-v11-nope-gpt-small/20260721`
- Random seed: default script seed `0`
- Environment: local CPU via `scripts/nix-cpu-run`; exact manifest environment to be recorded at run time
- Model: `andrewdalpino/NoPE-GPT-Small-Base` pinned at Hugging Face revision `320681e33a029517e27c68a0f9c2b07ea0004155`
- Preparation checklist: `temp/repro-checklists/20260721-k-address-space-m15-v11-nope-gpt-small-cpu.md`
- Local verification: py_compile passed; unittest suite passed; NoPE A/B one-layer/one-head smoke passed with Family B non-empty and `shuffle_null_ok=True`.

### Results

Run completed locally on CPU from pre-run commit `133f115`.

Command:

```bash
PYTHONPATH=experiments/dead-keys:experiments/k-address-space ./scripts/nix-cpu-run -m kaddress.scripts.position_content \
  --model nope-gpt-small \
  --revision 320681e33a029517e27c68a0f9c2b07ea0004155 \
  --families A,B,C \
  --output-dir outputs/k_address_space_m15_v11_nope_gpt_small_cpu_20260721
```

Run window: `20260721T110903Z` to `20260721T113337Z`; exit code 0. The disposable code-fix brief `experiments/k-address-space/CODEFIX-M1.5.md` was deleted after the run completed, per instruction.

Outputs were published as GitHub Release assets:

- Release: <https://github.com/vhallac/crockpot-experiments/releases/tag/run/k-address-space-m15-v11-nope-gpt-small/20260721>
- `k_address_space_m15_v11_nope_gpt_small_cpu_20260721.tar.gz` — 20,199,687 bytes; SHA256 `5511822789f285483bab43cd28e8a16cffcb1cf575e4becf77dd24fc03256512`
- `SHA256SUMS-m15-v11-nope-20260721.txt` — 938 bytes; checksum file verified by re-download and byte comparison.

Internal output checksums:

- `kaddress_m15_nope-gpt-small.csv` — SHA256 `f00420e4ddfdbddc22b6a3f188cf6327d42e444c8ddf604a7b7f1237d5c6eb1e`
- `kaddress_m15_gates_nope-gpt-small.csv` — SHA256 `927ac35665c81eea3e172ff38e9c208c5739c84a0b36ea509e607096a2d4f69d`
- `kaddress_m15_manifest_nope-gpt-small.json` — SHA256 `6e3e455cb7e06466a639d2b0d9bbd3cb0e15b57a06a40ef188ebfe000ba67688`
- `kaddress_m15_projectors_nope-gpt-small.npz` — SHA256 `262c57585bb209413a94564962131ac2535d2e3b3d09654c84cdeeb70029afc6`
- `run.log` — SHA256 `bb4612f6cef2fd89224791231035d2b754c704d91088c39d376c050971f08320`

Manifest highlights: `stimulus_count=19`, `summary_rows=37632`, `families=[A,B,C]`, `trained_context=1024`, `max_length=992`, `min_repetitions=128`, `segment_lengths=[4,7]`, `rejected_stimuli=[]`, `requested_device=cpu`, `cuda_available=false`, Python `3.11.11`, Torch `2.5.1`, model revision `320681e33a029517e27c68a0f9c2b07ea0004155`. Family A contributed 16 stimuli (8 at L=4 and 8 at L=7), Family B contributed 2 frame/content-varying stimuli, and Family C contributed 1 natural-recurrence stimulus.

Gate and null summary:

- G1 architectural zero passed for all 1,520 layer-0 checked slots/heads.
- Corrected G4 passed: `shuffle_null_ok=true`; shuffled-null mean over all rows was `-0.0357`, and negative null values were no longer counted as leakage.
- Degenerate layer-0 rows were explicitly marked and had derived position statistics zeroed.

Selected Family A slot-level depth means:

| layer | position fraction | ridge R² | PCA k90 | R² after PC projection |
|---:|---:|---:|---:|---:|
| 0 | 0.000001 | 0.000 | 0.00 | 0.000 |
| 1 | 0.00729 | 0.042 | 0.82 | 0.041 |
| 2 | 0.00893 | 0.257 | 1.28 | 0.159 |
| 6 | 0.0280 | 0.582 | 1.66 | 0.242 |
| 12 | 0.0234 | 0.900 | 1.16 | 0.059 |
| 18 | 0.0558 | 0.947 | 2.08 | 0.074 |
| 23 | 0.0656 | 0.972 | 2.15 | 0.034 |

Selected aggregate means show Family B is now non-empty and strongly position-decodable at depth: Family B aggregate ridge R² rises from `0.000` at layer 0 to `0.828` at layer 6, `0.969` at layer 12, and `0.973` at layer 23.

### Analysis

The corrected run preserves the core NoPE result: layer-0 keys are effectively position-free, then position becomes strongly decodable from K with depth. Family A's depth curve is directionally consistent with the first run, though not byte-identical because v1.1 now uses both L=4 and L=7 cells and 16 surviving Family A stimuli instead of the previous accidental subset. The headline still holds: Family A ridge R² reaches about `0.90` by layer 12 and `0.97` by layer 23.

The major correction is successful: Family B is no longer empty. The offset-based frame-token construction yields two vocabulary-disjoint Family B stimuli with actual cumulative offsets, and Family B shows a similar depth-rising position signal. This removes the previous caveat that the induction-control family never ran.

The spec corrections also behave as intended. The feasibility matrix records L=4 and L=7 as feasible at `R_min=128`; no stimuli were silently rejected; the one-sided shuffled-null gate passes despite a negative null mean; and the widened variance floor prevents layer-0 PCA/noise artifacts from appearing as real position components.

Caveat: Family A aggregate projector fidelity is weaker than the earlier single-effective-L run at upper layers (e.g. aggregate Family A layer-23 R² after projection ≈ `0.163` rather than near zero). This is expected to be more stringent because v1.1 combines two segment lengths to break absolute-position vs repetition-index collinearity; it should be tracked in cross-model comparisons rather than treated as a failed reproduction.

### Conclusion / Next Step

The v1.1 NoPE CPU rerun is valid and supersedes the first M1.5 NoPE run for corrected-family and length-sweep claims. It confirms computed positional information in NoPE keys at depth, validates the repaired Family B induction control, and records a stricter multi-L projector-fidelity caveat for follow-up. Next step: use this v1.1 implementation for GPT-2/Pythia/Qwen3 cross-model M1.5 runs.

## 2026-07-21 — K-address-space M1.5 NoPE-GPT-Small position-content run prep

### Question / Hypothesis

Does `andrewdalpino/NoPE-GPT-Small-Base` compute positional information into attention keys at depth when token content is held constant by repeated-segment stimuli? The primary prediction is that layer-0 `k_pre` is an architectural zero, while deeper layers develop measurable position fraction, ridge decodability, and a position subspace that may be difficult to remove without harming token identity.

### Experiment Design Summary

Implement and run Addendum §5-M1.5 for the selected NoPE model. The new `kaddress.scripts.position_content` script generates repeated-token stimuli for Families A/B/C, extracts per-layer/per-head keys with the existing NoPE hook path, and reports M1.5.1–M1.5.6 diagnostics per stimulus/slot/head: position fraction, ridge CV R² with variance-floor guard, PCA capacity, leading-PC geometry, shuffled-y null, and position-PC removability with token-identity retention. It also writes per-slot position-removal PCA bases as the initial Π projector artifact.

### Planned Procedure

Prepare and smoke-test locally, commit the pre-run state, then run the selected NoPE model with the pinned revision:

```bash
PYTHONPATH=experiments/dead-keys:experiments/k-address-space ./scripts/nix-cpu-run -m kaddress.scripts.position_content \
  --model nope-gpt-small \
  --revision 320681e33a029517e27c68a0f9c2b07ea0004155 \
  --families A,B,C \
  --min-repetitions 120 \
  --max-length 950 \
  --output-dir outputs/k_address_space_m15_nope_gpt_small_20260721
```

If local CPU runtime is poor, rerun the same command through `scripts/cuda-run` on RunPod with `--device cuda` and the shared CUDA venv.

### Expected Signal / Interpretation Plan

G1 must pass for NoPE layer 0: within-slot position fraction below `1e-5`, and the deliberate perturbation check must prove the gate can fail. Shuffled-y ridge R² should remain near zero. The headline will be the depth curve for Family A: if position fraction/R² rise with depth, the probe confirms key-level computed position in NoPE; if they stay flat, the result contradicts the pre-registered NoPE expectation. Family B/C are corroborating controls and must be reported as caveats if they disagree in sign with Family A.

### Pre-run Provenance

- Spec: `experiments/k-address-space/addendum-M1.5.md`
- Parent spec: `experiments/k-address-space/spec.md`
- Code branch: `main`
- Pre-run commit: `3fa27d0` (`Add aggregate M1.5 projector fidelity`; initial prep commit `54c11ac`)
- Planned output location: `outputs/k_address_space_m15_nope_gpt_small_20260721`
- Random seed: default script seed `0`
- Environment: local CPU via `scripts/nix-cpu-run` unless escalated to RunPod CUDA; exact manifest environment to be recorded at run time
- Model: `andrewdalpino/NoPE-GPT-Small-Base` pinned at Hugging Face revision `320681e33a029517e27c68a0f9c2b07ea0004155`
- Preparation checklist: `temp/repro-checklists/20260721-k-address-space-m15-nope-gpt-small.md`
- Local verification: `py_compile` passed; GPT-2 1-stimulus/1-layer/1-head smoke passed; NoPE 1-stimulus/1-layer/1-head smoke passed with G1 position fraction ≈ `1.27e-7` and ridge R² `0.0`.

### Results

Run completed locally on CPU from pre-run commit `3fa27d0`.

Command:

```bash
PYTHONPATH=experiments/dead-keys:experiments/k-address-space ./scripts/nix-cpu-run -m kaddress.scripts.position_content \
  --model nope-gpt-small \
  --revision 320681e33a029517e27c68a0f9c2b07ea0004155 \
  --families A,B,C \
  --min-repetitions 120 \
  --max-length 950 \
  --output-dir outputs/k_address_space_m15_nope_gpt_small_20260721
```

Outputs were published as GitHub Release assets:

- Release: <https://github.com/vhallac/crockpot-experiments/releases/tag/run/k-address-space-m15-nope-gpt-small/20260721>
- `k_address_space_m15_nope_gpt_small_20260721.tar.gz` — 8,133,533 bytes; SHA256 `00ab55541353b58f0bc099218cf4dd8494f33036f1ee3e15eaec40f96984eb1f`
- `kaddress_m15_nope-gpt-small.csv` — 3,521,429 bytes; SHA256 `c5b1bfa6894e2b6403c73f85e7d89f36b15597c113a982e8a215ecb94418e136`
- `kaddress_m15_gates_nope-gpt-small.csv` — 55,436 bytes; SHA256 `500abe947b6a23a2d088459edbe0573aef0bb02c8e9f58cb23f3fc768b027ec3`
- `kaddress_m15_manifest_nope-gpt-small.json` — 2,462 bytes; SHA256 `c54a79b0fd45af4b2800e7dce99aa76e580fe472bec5b08e55b00d1e308a3d74`
- `kaddress_m15_projectors_nope-gpt-small.npz` — 9,636,244 bytes; SHA256 `c17e6dcb70aa86f8c1ea0af90de130d4f7e6314478a9588270e672bd0b8a2403`

Manifest highlights: `stimulus_count=6`, `summary_rows=14208`, `families=[A,B,C]`, `requested_device=cpu`, `cuda_available=false`, Python `3.11.11`, Torch `2.5.1`, model revision `320681e33a029517e27c68a0f9c2b07ea0004155`. Family A contributed five repeated-segment stimuli with 135–158 repetitions and 6–7 token slots; Family C contributed one natural-recurrence stimulus. Family B produced no valid stimuli under the tokenizer alignment plus ≥120-repetition constraints.

Gate and null summary:

- G1 architectural zero passed for all 560 layer-0 checked slots/heads; max layer-0 position fraction was `1.52e-6`, below the `1e-5` variance floor, and the deliberate perturbation check could fail.
- The manifest records `shuffle_null_ok=false`: slot-level shuffled-R² absolute quantiles were median `0.0359`, 90% `0.0967`, 95% `0.1222`, 99% `0.1902`. Aggregate Family A projector rows were much cleaner, with selected-layer shuffled R² around `-0.003` to `-0.006`.

Selected Family A slot-level depth means:

| layer | position fraction | ridge R² | PCA k90 | R² after PC projection |
|---:|---:|---:|---:|---:|
| 0 | 8.88e-7 | 0.000 | 1.00 | 0.000 |
| 1 | 0.00797 | 0.029 | 1.01 | 0.020 |
| 2 | 0.0113 | 0.152 | 1.14 | 0.138 |
| 6 | 0.0400 | 0.727 | 1.71 | 0.251 |
| 12 | 0.0481 | 0.951 | 1.10 | 0.055 |
| 18 | 0.0843 | 0.951 | 2.04 | 0.010 |
| 23 | 0.0892 | 0.979 | 2.20 | -0.023 |

### Analysis

The primary NoPE prediction is supported: layer-0 keys are effectively position-free, but repeated-token Family A keys become strongly position-decodable with depth. Mean ridge R² reaches `0.727` by layer 6 and ≈`0.95–0.98` from layer 12 onward, while the position fraction rises from `8.88e-7` at layer 0 to `0.089` at layer 23.

The position signal is low-dimensional in the slot-level analysis: Family A needs about 1–2 PCs to explain 90% of residual variation across most depths. Projecting out those PCs usually removes most position decodability at upper layers. In the aggregate Family A projector rows, layer-23 ridge R² drops from `0.9069` to `0.0049`; token-identity nearest-centroid accuracy on the reservoir sample is retained (`0.9844 → 0.9957`). This suggests the computed position component is removable for this NoPE run, at least under the implemented projector/sample diagnostic.

Caveats: Family B failed to instantiate valid aligned frame-token stimuli for this tokenizer/settings, and the slot-level shuffled-y null is not as clean as pre-registered even though aggregate nulls are near zero. Family C corroborates that recurrence keys contain strong position signal (`position_fraction=0.177`, `ridge_r2=0.848` on average), but it remains confounded and cannot overturn Family A.

### Conclusion / Next Step

This selected NoPE M1.5 run validates the depth-resolved key-level computed-position effect: NoPE layer 0 is an architectural zero, then `k_pre` keys acquire a strong, low-dimensional, mostly removable position component with depth. Next step is to run the same M1.5 implementation on GPT-2/Pythia/Qwen3 for the addendum's cross-model stamped-vs-computed decision tree, and to fix/improve Family B generation before treating B as a passed induction-control condition.

## 2026-07-20 — K-address-space M1 NoPE-GPT-Small full CUDA run prep

### Question / Hypothesis

Does `andrewdalpino/NoPE-GPT-Small-Base`, a true NoPE model with no positional encoding path, show M1 address-purity heads when keys are measured directly from attention `qkv_proj` output?

### Experiment Design Summary

Full Track A extraction for NoPE-GPT-Small-Base: deterministic generator output, all 24 layers, all 16 heads, `k_pre` keys from the remote implementation's `SelfAttention.qkv_proj`, head-mean-centered cosine, and pairwise M1 AUC against same-type/different-referent and position-matched controls.

Preparation changes over the Qwen3 run:
- Added model tag `nope-gpt-small` for `andrewdalpino/NoPE-GPT-Small-Base`, loaded via `AutoModel` with `trust_remote_code=True` after inspecting the remote implementation.
- Verified from remote `model.py` that `NoPEGPT.forward()` starts from `token_embeddings(x)`, feeds decoder blocks directly, and `SelfAttention` uses `qkv_proj` keys without positional embeddings, RoPE, or ALiBi.
- Added `_capture_nope_k()` to hook each decoder block's `attention.qkv_proj` and store per-head keys.
- Added a runtime NoPE sanity check that aborts if any module name suggests a positional/RoPE/ALiBi path.

### Planned Procedure

Run on RunPod CUDA from the pre-run commit:

```bash
cd /workspace/dead-keys-census
git pull --ff-only
./scripts/runpod-persistent-cache-setup
. ~/.dead-keys-census-runpod-env
DEAD_KEYS_CUDA_VENV=/workspace/dead-keys-census-cache/venvs/cuda-system \
DEAD_KEYS_CUDA_SKIP_INSTALL=1 \
PYTHONPATH=experiments/dead-keys:experiments/k-address-space \
./scripts/cuda-run -m kaddress.scripts.address_purity \
  --model nope-gpt-small --device cuda --limit-docs 999 \
  --output-dir outputs/k_address_space_m1_nope_gpt_small_full_cuda_20260720
```

### Expected Signal / Interpretation Plan

A valid run should produce 384 per-head rows (24 layers × 16 heads × 1 key variant), a manifest, compact mention vectors, and NoPE sanity output. Address-head calls require AUC > 0.9 against both controls. This is the true no-positional-encoding endpoint for the positional-encoding gradient in the spec.

### Pre-run Provenance

- Spec: `experiments/k-address-space/spec.md`
- Code branch: `main`
- Pre-run commit: `2e72271` (`Prepare k-address M1 NoPE run`); successful rerun code commits `d407c5f` (`Fix NoPE remote model loading`), `c42adca` (`Load NoPE checkpoint weights explicitly`), and `0555c06` (`Fix NoPE manifest key variant label`)
- Planned output location: `outputs/k_address_space_m1_nope_gpt_small_full_cuda_20260720`
- Random seed: default script seed `0`
- Environment: planned RunPod CUDA via `scripts/cuda-run`; exact pod/GPU/torch/transformers versions to be recorded at run time from the manifest
- Model: `andrewdalpino/NoPE-GPT-Small-Base` pinned at Hugging Face revision `320681e33a029517e27c68a0f9c2b07ea0004155`
- Preparation checklist: `temp/repro-checklists/20260720-k-address-space-m1-nope-gpt-small.md`
- Local verification: `py_compile` passed for loader and address-purity script; remote `model.py` inspection found no positional embedding/RoPE/ALiBi path.

### Results

Run completed on RunPod L4 from commit `0555c06` after retry fixes for the remote-code import shims, explicit checkpoint loading into the wrapper's inner model, and the NoPE manifest key-variant label. Pod `t1cpksbru2b23g` used network volume `sndrrdckku` mounted at `/workspace`.

Command:

```bash
DEAD_KEYS_CUDA_VENV=/workspace/dead-keys-census-cache/venvs/cuda-system \
DEAD_KEYS_CUDA_SKIP_INSTALL=1 \
PYTHONPATH=experiments/dead-keys:experiments/k-address-space \
./scripts/cuda-run -m kaddress.scripts.address_purity \
  --model nope-gpt-small \
  --revision 320681e33a029517e27c68a0f9c2b07ea0004155 \
  --device cuda --limit-docs 999 \
  --output-dir outputs/k_address_space_m1_nope_gpt_small_full_cuda_20260720
```

Run window: `2026-07-20T16:14:28Z` to `2026-07-20T16:17:27Z`; `EXIT_CODE=0`.

Outputs were published as GitHub Release assets:

- Release: <https://github.com/vhallac/crockpot-experiments/releases/tag/output-k-address-space-m1-nope-gpt-small-full-cuda-20260720>
- `kaddress_m1_nope-gpt-small.csv` — 34,333 bytes; SHA256 `7e968a6dc931d39ceb36a1a03b1e768932139d49e51d935c7511262f4885573d`
- `kaddress_manifest_nope-gpt-small.json` — 724 bytes; SHA256 `e3cf40b2c11bc542997b97ab8ff3737d35bdccf74b2750c37f89117026ba7d07`
- `kaddress_mentions_nope-gpt-small.npz` — 129,190,223 bytes; SHA256 `36b5bbaa822a8c9ffb47a3939b35bcd111b2ce2352bc36dd3f4a5a03ce58412e`
- `run.log` — 1,658 bytes; SHA256 `07338007c381f8c7c93069bbde41075400b6b8c1c15b412bf16ad7f48b43dfa7`

Manifest highlights: `doc_count=36`, `mention_token_rows=516096`, `max_doc_tokens=829`, `requested_device=cuda`, `cuda_available=true`, `cuda_device=NVIDIA L4`, `torch=2.8.0+cu128`, `torch_cuda=12.8`, Python `3.12.3`, HF revision `320681e33a029517e27c68a0f9c2b07ea0004155`.

NoPE sanity output:

```text
NoPE sanity: token_embeddings feed decoder body directly; SelfAttention.qkv_proj keys are used without positional embedding, RoPE, or ALiBi
```

**Address heads: 0/384.**

| Variant | Mean AUC vs same-type | Mean AUC vs pos-matched | Mean diff-surface AUC | Max same-type AUC | Max pos-matched AUC |
|---------|----------------------|------------------------|----------------------|-------------------|---------------------|
| `k_pre` | 0.5328               | 0.3549                 | 0.3439               | 0.6278            | 0.5691              |

Top same-type AUC head: layer 0 head 14, same-type AUC `0.6278`, position-matched AUC `0.5186`, diff-surface AUC `0.2629`.

### Analysis

NoPE-GPT-Small-Base shows no M1 address heads under the pre-registered threshold (AUC > 0.9 against both controls). Its best same-type AUC is 0.6278 and best position-matched AUC is 0.5691, far below threshold. Diff-surface same-referent purity is also weak (mean 0.3439, max 0.4956), so this NoPE endpoint does not reveal semantic address clustering in the implemented Track A/M1 slice.

Across the implemented quartet M1 Track A runs (GPT-2, Pythia-410m, Qwen3-0.6B, and NoPE-GPT-Small-Base), the pre-registered small-model sweep finds zero address heads by the strict threshold. Because the NoPE run uses no positional encoding path, the negative result is not explained by RoPE namespace distortion; it points instead toward this M1 signal being absent or too weak at these small model scales and synthetic Track A conditions.

### Conclusion / Next Step

This is a valid CUDA extraction of NoPE-GPT-Small-Base Track A / M1 in direct `k_pre` coordinates. It completes the planned quartet M1 sweep with no address heads at these model scales. The next durable step is a cross-model comparison/report over the four published CSVs, followed by deciding whether to stop the line at small scale or repeat M1 on a larger model where semantic addressing may be more plausible.

## 2026-07-20 — K-address-space M1 Qwen3 full CUDA run prep

### Question / Hypothesis

Does Qwen3-0.6B show M1 address-purity heads when keys are measured in both Qwen3 address coordinates (`k_pre`: post-`k_norm`, pre-RoPE) and cached coordinates (`k_post`: post-RoPE), grouped by the model's 8 KV heads under GQA?

### Experiment Design Summary

Full Track A extraction for Qwen3-0.6B: deterministic generator output, all 28 layers, all 8 KV heads, both `k_pre` and `k_post`, head-mean-centered cosine, and pairwise M1 AUC against same-type/different-referent and position-matched controls.

Preparation changes over the Pythia run:
- `_capture_qwen_k()`: hooks Qwen3 `k_proj` for pilot raw keys, hooks `k_norm` for `k_pre`, and extracts cached `k_post` from `past_key_values`.
- Qwen3 sanity gate confirms hook order `k_proj -> k_norm -> RoPE`, checks `num_attention_heads=16`, `num_key_value_heads=8`, Q-to-KV group size 2, reconstructs full-RoPE `k_post` from `k_pre`, and perturbs RoPE to prove the gate can fail.
- M1 rows are grouped by KV head (`head == kv_head`) rather than Q head.
- Runtime requirement updated to `transformers>=4.51.0` because the local Nix Transformers 4.46.2 package does not recognize `model_type=qwen3`.

### Planned Procedure

Run on RunPod CUDA from the pre-run commit after refreshing/installing the shared CUDA venv if needed:

```bash
cd /workspace/dead-keys-census
./scripts/runpod-persistent-cache-setup
. ~/.dead-keys-census-runpod-env
PYTHONPATH=experiments/dead-keys:experiments/k-address-space ./scripts/cuda-run   -m kaddress.scripts.address_purity   --model qwen3   --device cuda   --limit-docs 999   --sanity-gate-strict   --output-dir outputs/k_address_space_m1_qwen3_full_cuda_20260720
```

If reusing a known-current venv, `DEAD_KEYS_CUDA_SKIP_INSTALL=1` is acceptable only after verifying `python -c "import transformers; print(transformers.__version__)"` reports `>=4.51.0`.

### Expected Signal / Interpretation Plan

A valid run should produce 448 per-head rows (28 layers × 8 KV heads × 2 key variants), a manifest, compact mention vectors, and strict sanity-gate output. Address-head calls require AUC > 0.9 against both controls. Compare Qwen3 address-head count and pre/post AUC deltas to the prior GPT-2 and Pythia M1 runs; Qwen3's QK norm is expected to tighten within-referent key directions if the QK-norm prediction holds.

### Pre-run Provenance

- Spec: `experiments/k-address-space/spec.md`
- Code branch: `main`
- Pre-run commit: `f3bb63a` (`Prepare k-address M1 Qwen3 run`); successful rerun code commit `35c91c2` (`Add RunPod pod bring-up helper`)
- Planned output location: `outputs/k_address_space_m1_qwen3_full_cuda_20260720`
- Random seed: default script seed `0`
- Environment: planned RunPod CUDA via `scripts/cuda-run`; exact pod/GPU/torch/transformers versions to be recorded at run time from the manifest
- Model: `Qwen/Qwen3-0.6B` via Hugging Face default revision
- Preparation checklist: `temp/repro-checklists/20260720-k-address-space-m1-qwen3.md`
- Local verification: `py_compile` passed; GPT-2 1-doc/1-layer/1-head regression smoke passed; Pythia 1-doc/1-layer/1-head strict RoPE regression smoke passed. Local Qwen3 smoke is blocked by Nix Transformers 4.46.2 and must run in the updated CUDA venv.

### Results

Run completed on RunPod L4 from commit `35c91c2` after the retry helper created pod `cf160k2go7rknf` with network volume `sndrrdckku` mounted at `/workspace`.

Command:

```bash
DEAD_KEYS_CUDA_VENV=/workspace/dead-keys-census-cache/venvs/cuda-system \
DEAD_KEYS_CUDA_SKIP_INSTALL=1 \
PYTHONPATH=experiments/dead-keys:experiments/k-address-space \
./scripts/cuda-run -m kaddress.scripts.address_purity \
  --model qwen3 --device cuda --limit-docs 999 --sanity-gate-strict \
  --output-dir outputs/k_address_space_m1_qwen3_full_cuda_20260720
```

Run window: `2026-07-20T15:14:05Z` to `2026-07-20T15:19:18Z`; `EXIT_CODE=0`.

Outputs were published as GitHub Release assets:

- Release: <https://github.com/vhallac/crockpot-experiments/releases/tag/output-k-address-space-m1-qwen3-full-cuda-20260720>
- `kaddress_m1_qwen3.csv` — 40,144 bytes; SHA256 `4732acee42cc31d7af502bf58dd9450f7faf4813029ad0d24acf194156c3be73`
- `kaddress_manifest_qwen3.json` — 669 bytes; SHA256 `27c2a6cb9d446d4352513a74dbd3b81c142ba3a8c383dba54bbf38888a3436fd`
- `kaddress_mentions_qwen3.npz` — 297,261,610 bytes; SHA256 `70181eda3dbde619d6a7f843764b5e717baab7898681c6fc90882fe1526e6e8f`
- `run.log` — 1,624 bytes; SHA256 `27c3a46ef71e7dbea73e5a25133e0ba79bf969419ee4c83e332d5de58280ddab`

Manifest highlights: `doc_count=36`, `mention_token_rows=602112`, `max_doc_tokens=829`, `requested_device=cuda`, `cuda_available=true`, `cuda_device=NVIDIA L4`, `torch=2.8.0+cu128`, `torch_cuda=12.8`, Python `3.12.3`.

Strict Qwen3 sanity gate passed:

```text
Qwen3 sanity: q_heads=16 kv_heads=8 q_to_kv_group=2 hook_order=k_proj→k_norm→RoPE raw_norm_mean=41.4591 pre_norm_mean=50.8979
RoPE sanity gate: rotary_ndims=128/128  rel_l2_err=6.01e-09  max_abs_err=4.77e-06  max_elem_rel_err=1.85e-03  static_match=True  perturb_fails=True  → PASS
```

**Address heads: 0/448 (0/224 `k_pre`, 0/224 `k_post`).**

| Variant | Mean AUC vs same-type | Mean AUC vs pos-matched | Mean diff-surface AUC | Max same-type AUC | Max pos-matched AUC |
|---------|----------------------|------------------------|----------------------|-------------------|---------------------|
| `k_pre` | 0.5476               | 0.3386                 | 0.2424               | 0.6578            | 0.6220              |
| `k_post`| 0.5373               | 0.2403                 | 0.3006               | 0.6795            | 0.7629              |

AUC delta (`pre - post`) on same-type controls: mean `+0.0103`; `157/224` KV heads have `pre > post`.

### Analysis

Qwen3-0.6B shows no M1 address heads under the pre-registered threshold (AUC > 0.9 against both same-type/different-referent and position-matched controls). The best same-type AUC is 0.6795, still far below the address-head cutoff, and the mean diff-surface AUC remains low for both variants.

The namespace-direction signal matches the Pythia trend only weakly: `k_pre` has a small average same-type AUC advantage over `k_post` (+0.0103), and `k_post` reduces position-matched AUC on average. The magnitude is small and does not rescue address purity.

Across the implemented trio M1 Track A runs (GPT-2, Pythia-410m, Qwen3-0.6B), the pre-registered small-model sweep finds zero address heads by the strict threshold.

### Conclusion / Next Step

This is a valid CUDA extraction of Qwen3-0.6B Track A / M1 for both Qwen3 address coordinates (`k_pre`) and cached RoPE coordinates (`k_post`). It completes the implemented trio M1 sweep with no address heads at these model scales. Next analysis should compare the three published CSVs directly and decide whether to stop this line at small scale or repeat M1 on a larger model where semantic addressing is more plausible.

## 2026-07-18 — K-address-space M1 Pythia full CUDA run

### Question / Hypothesis

Does the M1 address-purity measurement (same-referent cosine AUC against controls) show address heads in Pythia-410m's `k_pre` (pre-RoPE) or `k_post` (post-RoPE) keys, and does partial RoPE (25% rotary dims) reduce address purity relative to pre-rotation keys?

### Experiment Design Summary

Full Track A extraction for Pythia-410m: 36 documents, all 24 layers, all 16 heads, both `k_pre` (hooked from `query_key_value` output) and `k_post` (extracted from `past_key_values` via `use_cache=True`).

Key implementation additions over the GPT-2 slice:
- `_capture_pythia_k()`: hook `query_key_value` for `k_pre`, extract `k_post` from cache
- `_rope_sanity_gate()`: reconstruct `k_post` from `k_pre` using from-scratch partial RoPE (config `rotary_pct=0.25`, `rotary_ndims=16/64`), verify against cached `k_post` (rel err ≤ 1e-3), confirm static dims match, perturbation check proves gate can fail
- `_extract_mentions()` dispatches by model tag, stores `key_variant` column
- `_summarize_auc()` groups by `(layer, head, key_variant)`

### Planned Procedure

```bash
PYTHONPATH=experiments/dead-keys:experiments/k-address-space DEAD_KEYS_CUDA_VENV=/workspace/dead-keys-census-cache/venvs/cuda-system DEAD_KEYS_CUDA_SKIP_INSTALL=1 ./scripts/cuda-run \
  -m kaddress.scripts.address_purity \
  --model pythia410 \
  --device cuda \
  --limit-docs 999 \
  --output-dir outputs/k_address_space_m1_pythia410_full_cuda_20260718
```

### Pre-run Provenance

- Spec: `experiments/k-address-space/spec.md`
- Code branch: `main`
- Pre-run commit: `64cb36f` (`Fix RoPE sanity gate: compute cos/sin from config frequencies`)
- Planned output location: `outputs/k_address_space_m1_pythia410_full_cuda_20260718`
- Random seed: default script seed `0`
- Environment: RunPod NVIDIA L4 (EU-RO-1), `deadd-keys-census-cuda` template (`1zpm2v05rn`), volume `dead-weight` (`sndrrdckku`), venv `/workspace/dead-keys-census-cache/venvs/cuda-system`
- Model: `EleutherAI/pythia-410m` via Hugging Face default revision
- Pod: `hlh4s9gxn0luyk`, $0.39/hr, L4, 6 vCPU, 62GB RAM
- Local smoke verification: 1/2 doc CPU smoke with sanity gate PASS before GPU run

### Results

Run completed on RunPod L4. GPU utilization verified at 97% during extraction.

Outputs under `outputs/k_address_space_m1_pythia410_full_cuda_20260718/`:

- `kaddress_m1_pythia410.csv` — 68,889 bytes; SHA256 `3af266fe209bcf34948762f4688646eea311a24ace73c56e77f42ff07f35639c`
- `kaddress_manifest_pythia410.json` — 681 bytes; SHA256 `1bcd5f300d0f9c57fd2671f0dca7e45237008d70987ffdadb29b7449ae061b3f`
- `kaddress_mentions_pythia410.npz` — 272,372,400 bytes; SHA256 `a1c305714744b714304c8915f2dae9712d48d1f231261028d1e9426e9cb9bc81`

Manifest highlights: `doc_count=36`, `mention_token_rows=1087488`, `max_doc_tokens=841`, `requested_device=cuda`, `cuda_available=true`, `cuda_device=NVIDIA L4`, `torch=2.8.0+cu128`.

RoPE sanity gate: `max_rel_err=0.00e+00`, `static_match=True`, `perturb_fails=True` → **PASS**.

**Address heads: 0/768 (0/384 per variant).**

| Variant | Mean AUC vs same-type | Mean AUC vs pos-matched | Mean diff-surface AUC | Max same-type AUC |
|---------|----------------------|------------------------|----------------------|-------------------|
| k_pre   | 0.5404               | 0.3708                 | 0.2833               | 0.6526 (L9 H9)    |
| k_post  | 0.5294               | 0.1742                 | 0.3951               | 0.6558 (L9 H9)    |

AUC delta (pre − post): mean = +0.0111, 272/384 heads (70.8%) have pre > post. RoPE rotation slightly reduces same-referent purity on average, consistent with the namespace hypothesis direction, but the effect is tiny.

Position-matched AUC collapses for k_post (0.3708 → 0.1742), indicating that RoPE does scatter keys by position — but this affects controls as much as same-referent pairs.

Diff-surface AUC is very low for k_pre (0.283) — the discriminating M2 test shows minimal semantic addressing. For k_post, diff-surface AUC is actually higher (0.395) but still far below address-head threshold.

### Analysis

No address heads found by the pre-registered M1 threshold (AUC > 0.9 against both controls) in Pythia-410m for either k_pre or k_post. The best observed same-type AUC is 0.656, substantially below the 0.9 address-head cutoff.

This result aligns with the GPT-2 M1 run (0/144 address heads) and does not contradict the spec's scale caveat: semantic addressing may emerge at larger scales (> 7B) and absence at 410M–0.6B is not absence at scale.

Modest support for the namespace hypothesis direction: k_pre purity > k_post purity in 70.8% of heads, and position-matched AUC drops sharply after RoPE (0.371 → 0.174). However, the mean delta (+0.011) is negligible in magnitude.

### Conclusion / Next Step

This run is a valid CUDA extraction of Pythia-410m Track A / M1 for both k_pre and k_post. It does not show address heads by the pre-registered threshold.

The trio census now has two of three models (GPT-2, Pythia-410m); Qwen3-0.6B remains for the full spec sweep.

## 2026-07-17 — K-address-space M1 GPT-2 smoke

### Question / Hypothesis

Does the first `spec.md` write/address measurement (M1) show any same-referent key-neighborhood signal on a small deterministic Track A slice for GPT-2?

### Experiment Design Summary

This is a first runnable smoke slice, not the full pre-registered weekend run. It generates deterministic synthetic Track A state-update documents with gold mention spans, extracts GPT-2 pre-rotation/pre-cache keys (`k_pre`; GPT-2 has no RoPE), head-mean-centers vectors per layer/head, and computes pairwise cosine AUC for same-referent mention-token pairs against same-type/different-referent and near-position different-referent controls.

### Planned Procedure

```bash
PYTHONPATH=experiments/dead-keys:experiments/k-address-space ./scripts/nix-cpu-run -m kaddress.scripts.address_purity \
  --model gpt2 \
  --limit-docs 2 \
  --limit-layers 1 \
  --limit-heads 1 \
  --output-dir outputs/k_address_space_m1_gpt2_smoke
```

### Expected Signal / Interpretation Plan

A successful smoke run should produce a manifest, a per-head M1 CSV, and compact mention vectors. AUC is not expected to be meaningful at `--limit-docs 2 --limit-layers 1 --limit-heads 1`; this run validates the extraction/analysis path before a broader local or GPU run.

### Pre-run Provenance

- Spec: `experiments/k-address-space/spec.md`
- Code branch: `main`
- Pre-run commit: `3bc1714` (`Add k-address-space M1 smoke experiment`)
- Planned output location: `outputs/k_address_space_m1_gpt2_smoke`
- Random seed: default script seed `0`
- Environment: local CPU via `scripts/nix-cpu-run`
- Model: `gpt2` via Hugging Face default revision

### Results

Run completed locally on CPU.

Outputs under `outputs/k_address_space_m1_gpt2_smoke/`:

- `kaddress_m1_gpt2.csv` — 290 bytes; SHA256 `fbc48a8e570b6dc536bdc912d3860802653a646dc01a982852ce2fd4ab3b16bb`
- `kaddress_manifest_gpt2.json` — 564 bytes; SHA256 `3e5f2052e7ca75eefbf080bc3ce953d0e0d68cc9eaae0aa69ac717ea0a2acf79`
- `kaddress_mentions_gpt2.npz` — 5784 bytes; SHA256 `642fcd300a574d219d94817b18e4bc8579a23019dc9e907850936ad82c312486`

Observed smoke row:

| layer | head | same-ref pairs | same-type diff-ref pairs | position-matched diff-ref pairs | AUC vs same-type | AUC vs position-matched | diff-surface AUC | address head |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 0 | 16 | 78 | 22 | 0.53125 | 0.34659 | 0.21154 | false |

### Analysis

The run validates that the first M1 extraction/analysis path executes end-to-end and produces compact outputs. With only two documents, one layer, and one head, the AUC values are smoke-test diagnostics only and should not be interpreted as evidence for or against the address-space hypothesis.

### Conclusion / Next Step

The first runnable experiment slice is in place and smoke-tested. Next step is a broader GPT-2 M1 run over all heads/layers and more Track A documents, followed by RoPE sanity-gate implementation before extending to Pythia/Qwen3.

## 2026-07-17 — K-address-space M1 GPT-2 full implemented Track A run

### Question / Hypothesis

Does the implemented Track A / M1 address-purity slice show same-referent key-neighborhood signal across all GPT-2 layers and heads on the full deterministic Track A generator output currently implemented in this repository?

### Experiment Design Summary

This run scales the already smoke-tested first slice from 2 documents / 1 layer / 1 head to the full implemented GPT-2 slice: deterministic Track A synthetic documents generated by `generate_track_a`, GPT-2 pre-cache keys (`k_pre`; no RoPE), all 12 layers and 12 heads, head-mean-centered cosine, and pairwise AUC of same-referent mention-token pairs against same-type/different-referent and near-position different-referent controls.

The implementation is still narrower than the full pre-registration: it covers GPT-2 and M1 on Track A only. Pythia/Qwen3 and RoPE sanity gates remain future work before those models should be claimed as run.

### Planned Procedure

Run on RunPod L4 from a committed repository state, using only non-interactive SSH command arguments (no piped `ssh -tt`):

```bash
PYTHONPATH=experiments/dead-keys:experiments/k-address-space DEAD_KEYS_CUDA_VENV=/venv-deadkeys DEAD_KEYS_CUDA_SKIP_INSTALL=1 ./scripts/cuda-run \
  -m kaddress.scripts.address_purity \
  --model gpt2 \
  --device cuda \
  --limit-docs 999 \
  --output-dir outputs/k_address_space_m1_gpt2_full_20260717
```

`--limit-docs 999` is used because the CLI default is a 2-document smoke limit; the current generator exhausts at 36 documents.

### Expected Signal / Interpretation Plan

A full implemented run should produce 144 per-head rows (12 layers × 12 heads), a manifest, and compact mention vectors. Address-head calls require AUC > 0.9 against both control classes. Diff-surface AUC is the discriminating lexical-vs-semantic diagnostic, but results are still limited to the synthetic Track A generator and GPT-2 only.

### Pre-run Provenance

- Spec: `experiments/k-address-space/spec.md`
- Code branch: `main`
- Pre-run commit: `1dbe4e6` for the first attempt; redo fix commit pending after failed attempt.
- Planned output location: `outputs/k_address_space_m1_gpt2_full_20260717`
- Random seed: default script seed `0`
- Environment: RunPod NVIDIA L4 via `scripts/cuda-run`
- Model: `gpt2` via Hugging Face default revision

### Results

First RunPod attempt failed before producing usable outputs. The process was stopped after it stayed CPU-bound and logged a GPT-2 context warning:

- Active process: `python3 -m kaddress.scripts.address_purity ...`, PID 875, about 102% CPU.
- GPU: 0% utilization during inspection.
- Warning: `Token indices sequence length is longer than the specified maximum sequence length for this model (1507 > 1024)`.
- Output directory: no result files produced.

Redo fix: Track A filler is now inserted once per update round instead of after every referent sentence, long filler was shortened, tokenization uses explicit `truncation=True, max_length=...`, an over-budget generator guard fails fast, and M1 pair collection is vectorized per document.

Verification before relaunch:

```text
36 Track A docs; max GPT-2 tokens = 829; docs over 950 = 0
outputs/k_address_space_m1_gpt2_fix_smoke/ written by 2-doc smoke
outputs/k_address_space_m1_gpt2_fix_full_localcheck/ written by 36-doc 1-layer/1-head check
```

Full rerun did not reach successful output analysis. It was user-interrupted after external observation showed the rerun was still not making meaningful use of the GPU.

Second failure record:

- Failure mode: user interrupt due to lack of sustained GPU utilization.
- Root cause carried forward from the interrupted run: the forward pass used CUDA, but `_capture_gpt2_k()` copied every layer's keys to CPU immediately with `k.cpu()`, and the downstream cosine/AUC work used pandas/NumPy on CPU.
- Protocol outcome: this experiment attempt is concluded as failed; the next attempt must keep captured keys and pairwise analysis on CUDA and verify GPU utilization before treating the run as valid.

### Analysis

No scientific result is available from the interrupted full run. The failure is an implementation/execution failure, not evidence for or against K-space address purity.

### Conclusion / Next Step

Conclude this attempt as failed. Patch the M1 implementation so captured key tensors remain on the requested device and the cosine/AUC summarization can run on CUDA, then create a new pre-run entry and rerun only after GPU-use verification.

## 2026-07-17 — K-address-space M1 GPT-2 full CUDA rerun after GPU patch

### Question / Hypothesis

Does the implemented Track A / M1 address-purity slice show same-referent key-neighborhood signal across all GPT-2 layers and heads on the full deterministic Track A generator output, when the extraction and analysis path keeps key vectors and pairwise cosine/AUC computation on CUDA?

### Experiment Design Summary

This is a redo of the interrupted GPT-2 M1 full implemented Track A run. The scientific design is unchanged: deterministic Track A documents, GPT-2 `k_pre`, all 12 layers and 12 heads, head-mean-centered cosine, and pairwise AUC against same-type/different-referent and position-matched controls.

The implementation change is execution-only: `_capture_gpt2_k()` no longer copies keys to CPU, mention vectors are stored as torch tensors on the requested device, cosine matrices are computed with torch, and AUC scoring uses torch tensors before final CSV/NPZ serialization.

### Planned Procedure

Run on RunPod L4 from a committed repository state, using only non-interactive external-IP SSH command arguments:

```bash
PYTHONPATH=experiments/dead-keys:experiments/k-address-space DEAD_KEYS_CUDA_VENV=/workspace/dead-keys-census-cache/venvs/cuda-system DEAD_KEYS_CUDA_SKIP_INSTALL=1 ./scripts/cuda-run \
  -m kaddress.scripts.address_purity \
  --model gpt2 \
  --device cuda \
  --limit-docs 999 \
  --output-dir outputs/k_address_space_m1_gpt2_full_cuda_20260717
```

Before accepting the run as valid, verify from outside the process that the manifest reports `requested_device: cuda`, `cuda_available: true`, and an NVIDIA L4 device, and sample `nvidia-smi` during execution for nonzero GPU memory/utilization.

### Expected Signal / Interpretation Plan

A valid full run should produce 144 per-head rows (12 layers × 12 heads), a manifest, and compact mention vectors. Address-head calls require AUC > 0.9 against both control classes. Diff-surface AUC remains the lexical-vs-semantic diagnostic. If GPU utilization remains absent after this patch, treat the attempt as another setup/implementation failure rather than a scientific result.

### Pre-run Provenance

- Spec: `experiments/k-address-space/spec.md`
- Code branch: `main`
- Pre-run commit: `6797a78` (`Keep k-address M1 analysis on torch device`)
- Planned output location: `outputs/k_address_space_m1_gpt2_full_cuda_20260717`
- Random seed: default script seed `0`
- Environment: RunPod NVIDIA L4 via `scripts/cuda-run`
- Model: `gpt2` via Hugging Face default revision
- Local verification before commit: 2-doc smoke and 36-doc/1-layer local CPU checks passed with the torch execution path.

### Results

Run completed on RunPod `dead-weight-m1-replacement-20260717190712` using the existing `/workspace/dead-keys-census-cache/venvs/cuda-system` CUDA venv with `DEAD_KEYS_CUDA_SKIP_INSTALL=1`. The erroneous `/workspace/dead-keys-census-cache/venvs/cuda-lite` environment was removed before rerun.

Outputs under `outputs/k_address_space_m1_gpt2_full_cuda_20260717/`:

- `kaddress_m1_gpt2.csv` — 12,438 bytes; SHA256 `e3f976f94f7e3b38be37a39477e4ecd6af947c6545324a732cb97ff9d7f3e07d`
- `kaddress_manifest_gpt2.json` — 647 bytes; SHA256 `fc9cb196224c3d64e3bbb1bd4d8fe107384c122e13c8bd050eefe8f0fcbd4588`
- `kaddress_mentions_gpt2.npz` — 48,325,922 bytes; SHA256 `7154ddd904a5878d28895959ee61298e6e1437f69e39efc7882d347f9cb76c7b`

Manifest highlights: `limit_docs=999`, `doc_count=36`, `mention_token_rows=193536`, `max_doc_tokens=829`, `requested_device=cuda`, `cuda_available=true`, `cuda_device=NVIDIA L4`, `torch=2.8.0+cu128`.

During the full rerun, external `nvidia-smi` sampling observed the process on GPU with 91% GPU utilization, 4,170 MiB GPU memory used, and PID 1542 using 4,164 MiB.

### Analysis

The implemented Track A / M1 GPT-2 full CUDA run produced 144 per-head rows and found `address_heads_m1=0/144` under the current threshold requiring AUC > 0.9 against both same-type/different-referent and position-matched controls.

Aggregate AUCs across heads were modest: mean same-referent-vs-same-type-different-referent AUC `0.502206`; mean same-referent-vs-position-matched-different-referent AUC `0.128059`.

Top same-type AUC rows remained far below the address-head threshold. Best observed row was layer 3 head 4 with same-type AUC `0.614722` and position-matched AUC `0.386818`.

### Published Measurements

Published as a GitHub release artifact bundle:

- Release: https://github.com/vhallac/crockpot-experiments/releases/tag/output-k-address-space-m1-gpt2-full-cuda-20260717
- Bundle: `k_address_space_m1_gpt2_full_cuda_20260717.tgz` — 47,079,221 bytes; SHA256 `b60dbd9e0693002cc6fe76baff497f2d6260d2606622f7149670bd218028bce4`
- Per-file checksums are attached as `SHA256SUMS` in the release.

### Conclusion / Next Step

This run is a valid CUDA execution of the current implemented GPT-2 Track A / M1 slice, not an environmental failure. It does not show address heads by the pre-registered M1 threshold in GPT-2 for this implemented synthetic slice. RoPE models are not expected to rescue this specific observation; if anything, RoPE makes a clean fixed K-address-space interpretation harder. Next step is to decide whether to extend the implementation toward the remaining spec items or revise the synthetic Track A slice before moving to RoPE models.
