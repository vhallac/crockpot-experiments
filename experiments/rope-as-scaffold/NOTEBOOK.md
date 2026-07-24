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
- **Model revision (reproducibility gap, low severity).** RS1a loaded `Qwen/Qwen3-0.6B` at the
  library-default `main` reference; all manifests record `revision: null`. Per `AGENTS.md`
  (§ Pin model and dataset revisions) this is a defect — the loader should have pinned an explicit
  SHA. Best-effort recovery: the snapshot is `Qwen/Qwen3-0.6B` `main` HEAD **as of 2026-07-24**;
  Qwen3-0.6B is slow-moving, so this recovers the actual revision with high accuracy but is
  unverified for this run. RS1b **MUST** pin an explicit SHA at load time (spec §10.A).

### Published Outputs

- Release: <https://github.com/vhallac/crockpot-experiments/releases/tag/run/rope-as-scaffold-rs1a/20260724>
- Bundle: `rope_as_scaffold_rs1a_20260724T0559Z.tar.gz`
- SHA256: `224765f0042b8c185a8bbd74a28914a18729b98ebd580b254b316f2f54b122e2`

### Final Provenance

- Run-record commit: `b023eaf` (this notebook's Results, first recorded).
- Analysis-correction commit: `45122f6` (M1.5 emergent-position result added, M1.6 over-read fixed).
- Reproducibility follow-ups commit: this commit (revision caveat, final provenance, RS1b length pin).

### Results

- G-RS1.1 passed: dropped-state `k_pre == k_post` with max abs `0.0`; restoring true RoPE failed identity with max abs `67.4829330444336`.
- Frozen FineWeb-Edu perplexity over 5,000,000 eval tokens:
  - `qwen3`: CE `3.0818854172969434`, PPL `21.799464762112162`.
  - `qwen3-dropped`: CE `10.336862396656565`, PPL `30849.085845460013`.
  - Context: dropped CE `10.34` nats sits just under uniform-random `ln(152k) ≈ 11.93` — runtime
    RoPE removal renders the model **near-random**, not merely degraded. G-RS1.2 half-1 passes
    emphatically (the drop is decisively non-vacuous).
- M1.5 completed for `qwen3` and `qwen3-dropped`; G1 passed and G2 was not applicable.
  Null-corrected ridge R² (`r2_minus_null_mean`, mean over heads/stimuli) by depth — the
  informative State-2 probe:

  | layer | RoPE `k_pre` | RoPE `k_post` | dropped `k_pre==k_post` |
  |---|---|---|---|
  | 0 | 0.000 | 1.043 | 0.000 |
  | 1 | 0.894 | 1.037 | 0.171 |
  | 2 | 0.949 | 1.044 | 0.520 |
  | 3 | 0.990 | 1.037 | 0.763 |
  | 4 | 0.985 | 1.050 | 0.979 |
  | 6+ (plateau) | ~0.99 | ~1.04 | **~1.03** |

  The dropped state retains near-full key-position decodability (plateau ~1.03, ≥ the RoPE model's
  emergent `k_pre` ~0.99) with the native-NoPE depth shape (architectural zero at L0 → rises →
  plateaus by L4–6) — **before any recalibration**. Early layers (L1–L3) lose position relative to
  the RoPE model and converge by L4.
- M1.6 completed:
  - `qwen3`: G6 pass, G7 pass count `39`, transitivity confirmed count `448`.
  - `qwen3-dropped`: G6 pass, G7 pass count `0`, transitivity confirmed count `0`.

### Analysis

Runtime RoPE removal is a valid, falsifiable state that catastrophically degrades language-modeling
quality (near-random perplexity), yet the representations tell a sharply different story.

**Headline — emergent position survives the drop almost intact.** Despite near-random perplexity,
the dropped model's key-position decodability plateaus at ~1.03 (M1.5 table above), matching or
exceeding the RoPE model's emergent `k_pre`, with the characteristic native-NoPE depth profile —
and this is present *before* any recalibration. So the dropped state is a clean **dissociation:
position is fully decodable from K even in a model that cannot use it for LM.** This directly
foreshadows P.RS1.b (position "fills in"): there is little to fill in — it is already there. The
implication for RS1b is that recalibration is testing whether training **reconnects the readout** to
already-present position, not whether position must be rebuilt.

**Mechanistic refinement of P1.5.c.** In the RoPE model, `k_pre` is already highly position-decodable
at L1 (0.89) while the dropped model is only 0.17 there, converging by L4. Read: shallow-layer
position in the RoPE model is **rotation-propagated** (mixed into the residual by early rotated
attention, and lost when the rotation is removed), whereas deep-layer position is **emergently
reconstructed and rotation-independent**. This separates the two sources of position that P1.5.c
lumped together.

**Caveat on M1.6 (correcting an earlier over-read).** The dropped state shows no G7 attention signal
and no confirmed transitivity, but this is **not** clean evidence that RoPE specifically carries
transitive induction: the dropped model is near-random, so it fails *every* behavioral probe
trivially — the null is confounded by global model breakdown. The interpretable State-2 probe is
M1.5 (representations), not M1.6 (behavior). The genuine addressing test (P.RS1.c) is State 1 (RoPE)
vs State 3 (recalibrated, **functional**), which RS1b produces; State-2 M1.6 nulls are expected and
near-uninformative and should not be read as a positive claim about RoPE's causal role.

### Conclusion / Next Step

RS1.a passes as a zero-training validation and green-lights RS1b. It also sharpens RS1b's
hypothesis: because emergent key-position is *already* near-ceiling in the dropped model (M1.5),
RS1b tests whether light recalibration can reconnect the LM readout to that already-present position
— predict P.RS1.a (perplexity recovers) with the M1.5 profile changing little, after which P.RS1.c
(addressing) becomes measurable on a functional model. The runtime-dropped model is not usable
as-is, but it is a valid baseline for that test. The `qwen3-dropped` plumbing (rotary-disable,
frozen eval, probe branches) is proven; only the training loop (spec §10.C) is new for RS1b.
