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
