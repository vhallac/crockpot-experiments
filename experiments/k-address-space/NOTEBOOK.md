# K-address-space lab notebook

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
- Pre-run commit: _pending_
- Planned output location: `outputs/k_address_space_m1_gpt2_smoke`
- Random seed: default script seed `0`
- Environment: local CPU via `scripts/nix-cpu-run`
- Model: `gpt2` via Hugging Face default revision

### Results

_Pending run._

### Analysis

_Pending output analysis._

### Conclusion / Next Step

_Pending._
