# RoPE-as-Scaffold Notebook

Newest entries first.

## 2026-07-24 — RS1.a RunPod validation preparation

### Question / Hypothesis

RS1.a validates the zero-training half of RS1: Qwen3 with runtime RoPE disabled should be a distinct, falsifiable model state, and the frozen FineWeb-Edu perplexity plus M1.5/M1.6 probes should be runnable before any DroPE recalibration spend.

### Experiment Design Summary

Prepared RunPod-only validation for states 1–2:

- State 1: `qwen3` (`Qwen/Qwen3-0.6B`) with RoPE enabled.
- State 2: `qwen3-dropped` using the same weights with centralized identity rotary embeddings.
- Gate G-RS1.1: assert dropped-state `k_pre == k_post`, then restore true RoPE and require that identity check to fail.
- Frozen eval: FineWeb-Edu `HuggingFaceFW/fineweb-edu`, config `sample-10BT`, streaming train split, first `eval_tokens` packed with EOS as held-out slice, `eval_context=2048`, `stride=2048`, token-weighted CE then perplexity.
- Probe gate: run M1.5/M1.6 on `qwen3` and `qwen3-dropped`; only proceed to RS1b training if the dropped-state machinery passes and dropped perplexity is meaningfully worse than RoPE baseline.

### Planned Procedure

Run inside a RunPod pod from `/workspace/crockpot-experiments` after cache setup:

```bash
cd /workspace/crockpot-experiments
./scripts/runpod-persistent-cache-setup
. ~/.crockpot-experiments-runpod-env
export DEAD_KEYS_CUDA_VENV=/workspace/venv
```

1. Run G-RS1.1:

```bash
PYTHONPATH=experiments/dead-keys:experiments/k-address-space \
  ./scripts/cuda-python experiments/rope-as-scaffold/scripts/verify_grs11.py \
  --output outputs/rope_as_scaffold_rs1a_$(date -u +%Y%m%dT%H%M%SZ)/grs11.json
```

2. Run the frozen RS1.a perplexity eval:

```bash
PYTHONPATH=experiments/dead-keys \
  ./scripts/cuda-python experiments/rope-as-scaffold/scripts/eval_perplexity.py \
  --models qwen3 qwen3-dropped \
  --eval-tokens 5000000 \
  --output-dir outputs/rope_as_scaffold_rs1a_eval_$(date -u +%Y%m%dT%H%M%SZ)
```

3. Run M1.5 for states 1–2:

```bash
PYTHONPATH=experiments/dead-keys:experiments/k-address-space \
  ./scripts/cuda-python -m kaddress.scripts.position_content \
  --model qwen3 --device cuda \
  --output-dir outputs/rope_as_scaffold_rs1a_m15_qwen3_$(date -u +%Y%m%dT%H%M%SZ)

PYTHONPATH=experiments/dead-keys:experiments/k-address-space \
  ./scripts/cuda-python -m kaddress.scripts.position_content \
  --model qwen3-dropped --device cuda \
  --output-dir outputs/rope_as_scaffold_rs1a_m15_qwen3_dropped_$(date -u +%Y%m%dT%H%M%SZ)
```

4. Run M1.6 for states 1–2:

```bash
PYTHONPATH=experiments/dead-keys:experiments/k-address-space \
  ./scripts/cuda-python -m kaddress.scripts.m16_discriminator \
  --model qwen3 --device cuda \
  --output-dir outputs/rope_as_scaffold_rs1a_m16_qwen3_$(date -u +%Y%m%dT%H%M%SZ)

PYTHONPATH=experiments/dead-keys:experiments/k-address-space \
  ./scripts/cuda-python -m kaddress.scripts.m16_discriminator \
  --model qwen3-dropped --device cuda \
  --output-dir outputs/rope_as_scaffold_rs1a_m16_qwen3_dropped_$(date -u +%Y%m%dT%H%M%SZ)
```

### Expected Signal / Interpretation Plan

- G-RS1.1 must pass and be perturbable before trusting any dropped-state probe.
- G-RS1.2 half-1 expects `ppl(qwen3-dropped) >> ppl(qwen3)` on the exact frozen eval slice.
- M1.5/M1.6 outputs are the pre-training baseline for deciding whether RS1b training is worth launching.

### Pre-run Provenance

- Spec: `experiments/rope-as-scaffold/RS1-spec.md` §10.
- Code branch: `main`.
- Pre-run commit: `8ae255e584634cb0668eb8968d55d2d06faa18c2`.
- Planned output location: `outputs/rope_as_scaffold_rs1a_*` on RunPod, later packaged externally if promoted to a reproducible run.

### Run Evidence

- Run id: `rope_as_scaffold_rs1a_20260724T0559Z`.
- RunPod pod: `gkpvc4epm4em7r` (`NVIDIA L4`, driver `570.195.03`, 23034 MiB VRAM).
- Start/end: `2026-07-24T05:58:54Z` → `2026-07-24T09:06:32Z`.
- Local manifest: `experiments/rope-as-scaffold/artifacts/RS1a-run-20260724.md`.
- Redo note: first M1.5 attempt OOMed because Qwen's config-derived context exceeded L4 memory; M1.5 was redone with `--max-length 1024` for both states.

### Published Outputs

- Release: <https://github.com/vhallac/crockpot-experiments/releases/tag/run/rope-as-scaffold-rs1a/20260724>
- Bundle: `rope_as_scaffold_rs1a_20260724T0559Z.tar.gz`
- SHA256: `224765f0042b8c185a8bbd74a28914a18729b98ebd580b254b316f2f54b122e2`

### Results

- G-RS1.1 passed: dropped-state `k_pre == k_post` with max abs `0.0`; restoring true RoPE failed identity with max abs `67.4829330444336`.
- Frozen FineWeb-Edu perplexity over 5,000,000 eval tokens:
  - `qwen3`: CE `3.0818854172969434`, PPL `21.799464762112162`.
  - `qwen3-dropped`: CE `10.336862396656565`, PPL `30849.085845460013`.
- M1.5 completed for `qwen3` and `qwen3-dropped`; G1 passed and G2 was not applicable.
- M1.6 completed:
  - `qwen3`: G6 pass, G7 pass count `39`, transitivity confirmed count `448`.
  - `qwen3-dropped`: G6 pass, G7 pass count `0`, transitivity confirmed count `0`.

### Analysis

Runtime RoPE removal is a valid and falsifiable state, but it catastrophically degrades language-modeling quality. The M1.6 comparison is consistent with RoPE carrying or scaffolding transitive induction behavior in this setup: baseline Qwen3 has confirmed transitivity signals, while dropped-RoPE has none under the same discriminator protocol.

### Conclusion / Next Step

RS1.a passes as a zero-training validation. Proceeding to RS1b is justified only as a recalibration/retraining test: the runtime-dropped model is not usable as-is, but it is a valid baseline for asking whether training can recover function without RoPE.
