# K-address-space lab notebook

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
- Pre-run commit: _pending_
- Planned output location: `outputs/k_address_space_m1_qwen3_full_cuda_20260720`
- Random seed: default script seed `0`
- Environment: planned RunPod CUDA via `scripts/cuda-run`; exact pod/GPU/torch/transformers versions to be recorded at run time from the manifest
- Model: `Qwen/Qwen3-0.6B` via Hugging Face default revision
- Preparation checklist: `temp/repro-checklists/20260720-k-address-space-m1-qwen3.md`
- Local verification: `py_compile` passed; GPT-2 1-doc/1-layer/1-head regression smoke passed; Pythia 1-doc/1-layer/1-head strict RoPE regression smoke passed. Local Qwen3 smoke is blocked by Nix Transformers 4.46.2 and must run in the updated CUDA venv.

### Results

_Pending run._

### Analysis

_Pending output analysis._

### Conclusion / Next Step

_Pending._

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
