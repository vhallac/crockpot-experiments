# RoPE-as-Scaffold Notebook

Newest entries first.

## 2026-07-25 — RS1b LR-corrected rerun (pre-run)

### Question / Hypothesis

RS1b v1 (`qwen3-droped-20260724`, peak LR 3e-5) plateaued at held-out PPL≈35 against the ~21.8
RoPE-baseline target, and its M1.6 probe showed `transitivity_confirmed` collapsing from 448/448
(RoPE) to 0/448 — but this is confounded: is the collapse a genuine consequence of removing RoPE,
or an artifact of under-training at too low an LR? DroPE's own paper (arXiv 2512.12167, Appendix
D.3 Table 11) ablates this exact LR on a comparable-scale model (SmolLM-360M) and shows 3e-5 is
measurably worse (final loss 2.7–3.1) than their defaults (3e-4: 2.53–2.56; 1e-3+QK-norm: 2.496,
their best). Qwen3 already has native QK-norm, so 1e-3 should be safe without the instability their
ablation needed QK-norm to solve. This run tests whether a corrected LR (1e-3) achieves full
perplexity recovery, and if so, whether the M1.5/M1.6 mechanistic picture changes once the
training-budget confound is removed.

### Experiment Design Summary

- Identical to RS1b v1 (`RS1-spec.md` §10.C) except **peak LR: 1e-3 (was 3e-5)**. Token budget
  (1B tokens), train context (2048), schedule shape (cosine → 10% of peak, 2% warmup), optimizer,
  seed (0), and all data/eval definitions are unchanged — one variable changed, for a clean
  comparison against v1.
- Fallback: if 1e-3 destabilizes (loss spikes despite QK-norm), drop to 3e-4 and record the
  substitution.
- Output artifact: a new checkpoint directory (distinct from `qwen3-droped-20260724`), followed by
  the same M1.5/M1.6 probe sequence used on v1.

### Planned Procedure

Same training script (`experiments/rope-as-scaffold/scripts/train_qwen3_nope.py`) and probe
commands as the v1 run and the 2026-07-25 M1.5/M1.6 probes entry below, with `--lr 1e-3` (or `3e-4`
on fallback) and a new `--output-dir`/run-id timestamp. Full command bundle to follow the same
smoke-test-first guardrail from `qwen3-nope-training.md` before committing to the full run.

### Expected Signal / Interpretation Plan

- **If PPL recovers close to ~21.8 and M1.6 transitivity/output_above_noise stay near zero:**
  strong evidence the addressing collapse is a genuine consequence of RoPE removal, not a training
  artifact — sharpens C1/P.RS1.c substantially and motivates the RS3/C3 behavioral follow-up
  discussed for this finding.
- **If PPL recovers and transitivity/output_above_noise recover substantially too:** the v1
  collapse was primarily an under-training artifact; P.RS1.c would need re-adjudication in the
  program's favor, and RS1b-ctrl's recipe (§11) should be updated to match this LR.
- Either outcome is reportable; this run exists specifically to remove the LR confound from the
  v1 interpretation, not to presuppose a direction.

### Pre-run Provenance

- Spec: `experiments/rope-as-scaffold/RS1-spec.md` §10.C (LR revision recorded there).
- Related: `experiments/rope-as-scaffold/rs1b-lr-retuning-note.md` (full reasoning for the LR
  change, options considered, and the RS1b-ctrl consequence).
- Code branch: `main`.
- Pre-run commit: `50a4e5a`.
- Prior run for comparison: `qwen3-droped-20260724` (RS1b v1), see the 2026-07-24 training report
  and the 2026-07-25 M1.5/M1.6 probes entry below.

### Results

_Pending run._

### Analysis

_Pending output analysis._

### Conclusion / Next Step

_Pending._

## 2026-07-25 — RS1b DroPE-trained M1.5/M1.6 probes (pre-run)

### Question / Hypothesis

After RS1b recalibration of Qwen3-0.6B with identity RoPE (`qwen3-droped`), do the M1.5 and M1.6 probes show that emergent key-position fills in while query-readable addressing remains absent? This run proceeds despite the training hiccup that held-out perplexity recovered only to about 35 rather than the original Qwen3 baseline.

### Experiment Design Summary

- State 3: `qwen3-droped`, loaded from the trained checkpoint at `/workspace/rs1b-artifacts/qwen3-droped-20260724` via `QWEN3_DROPED_PATH`.
- M1.5: `kaddress.scripts.position_content` on CUDA, full Qwen3 layer/head scope, `--max-length 1024` matching the prior RS1.a L4-safe/Qwen probe adjustment.
- M1.6: `kaddress.scripts.m16_discriminator` on CUDA, full Qwen3 layer/head scope, default v1.1 repetitions/stimuli/gates.
- Static GPU audit: both scripts load through `deadkeys.common.loading.load_model`; `qwen3-droped` points at the trained checkpoint and applies identity rotary embeddings; both scripts print progress. M1.5 uses batched torch GPU extraction/ridge paths with small final CPU serialization/statistics; M1.6 does per-head GPU forwards with scalar readouts, matching prior accepted CUDA runs.

### Planned Procedure

Run on the existing A100 SXM pod from `/workspace/crockpot-experiments` after syncing this pre-run commit:

```bash
cd /workspace/crockpot-experiments
. ~/.crockpot-experiments-runpod-env
export DEAD_KEYS_CUDA_VENV=/workspace/venv
export DEAD_KEYS_CUDA_SKIP_INSTALL=1
export QWEN3_DROPED_PATH=/workspace/rs1b-artifacts/qwen3-droped-20260724
RUN_ID=rope_as_scaffold_rs1b_probes_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p logs
PYTHONPATH=experiments/dead-keys:experiments/k-address-space ./scripts/cuda-python -m kaddress.scripts.position_content \
  --model qwen3-droped --device cuda --max-length 1024 \
  --output-dir outputs/${RUN_ID}_m15_qwen3_droped 2>&1 | tee logs/${RUN_ID}_m15.log
PYTHONPATH=experiments/dead-keys:experiments/k-address-space ./scripts/cuda-python -m kaddress.scripts.m16_discriminator \
  --model qwen3-droped --device cuda \
  --output-dir outputs/${RUN_ID}_m16_qwen3_droped 2>&1 | tee logs/${RUN_ID}_m16.log
```

### Expected Signal / Interpretation Plan

- P.RS1.b: M1.5 should find present/depth-rising emergent key-position in the trained DroPE'd model, ideally at least as strong as the RoPE model's `k_pre` baseline.
- P.RS1.c: M1.6 should remain null for query-readable addressing; appearance of robust output addressing would falsify the scaffold/non-addressability story.
- The suboptimal LR / PPL≈35 training result is treated as a limitation and possible under-recovery confound, not as a reason to skip the mechanistic probes.

### Pre-run Provenance

- Spec: `experiments/rope-as-scaffold/RS1-spec.md` §§2–5, §10.
- Code branch: `main`.
- Pre-run commit: `f200c92c5825b084e265a8b5c6c8d02fc685b671`.
- Fix/rerun commit: `a25f8cf` (M1.5 aggregate analysis on CUDA).
- Planned output location: RunPod `outputs/rope_as_scaffold_rs1b_probes_*`; to be packaged and published as a GitHub Release asset.

### Published Outputs

- Release: <https://github.com/vhallac/crockpot-experiments/releases/tag/run/rope-as-scaffold-rs1b/20260725>
- Bundle: `rope_as_scaffold_rs1b_probes_20260725T0830Z.tar.gz`
- SHA256: `1c5d9b7c48db62475e302c83b6215a8cfb0a0b95dd7fe92f4c66358f061af337`

### Final Provenance

- Run-record commit: `a25f8cf` (CUDA aggregate fix).
- Analysis commit: this commit (notebook Results, Analysis, Conclusion).

### Results

Run id: `rope_as_scaffold_rs1b_probes_20260725T0830Z`. RunPod A100 SXM pod, driver `570.195.03`.

**M1.5 (position-content probe)** — gates: G1 `PASS` (architectural zero at layer 0; qwen3-droped's `rotary_mode=identity` means `k_pre == k_post` so G2 is `NOT_APPLICABLE`). `summary_rows=43904`, `shuffle_null_ok=True`.

Null-corrected ridge R² (`r2_minus_null_mean`, mean over heads/stimuli) by depth — the State-3 trained model:

| layer | RS1a 0-shot dropped | RS1b trained |
|---|---|---|
| 1 | 0.171 | 0.179 |
| 2 | 0.520 | 0.475 |
| 3 | 0.763 | 0.806 |
| 4 | 0.979 | 0.996 |
| 5 | ~1.030 | 1.006 |
| 6–27 (plateau) | ~1.030 | ~1.020–1.033 |

**M1.6 (hypothesis discriminator)** — gates: G6 `PASS` (marker neutrality confirmed). G7 pass count `13` (vs RS1a: qwen3 `39`, qwen3-dropped `0`). `classification_rows=448`, transitivity confirmed `0`.

### Analysis

The headline is that **light DroPE recalibration changes the positional representation
negligibly, and the addressing-confirmation signal — not just the training loss — did not
recover.**

**M1.5 — P.RS1.b holds by the spec's literal criterion, but is uninformative beyond RS1a.**
The trained model's key-position R² (present, plateau ~1.02–1.03) satisfies the pre-registered
condition ("present and ≥ the RoPE model's k_pre emergent position," RoPE plateau ~0.99) — the
falsifier ("absent, yet perplexity recovers") did not trigger. So P.RS1.b **holds**, not
"falsified": position is present and at least as decodable as the RoPE baseline. The reason this
doesn't feel like a positive result is that RS1a already found this same near-ceiling level in the
*zero-shot, untrained* dropped state — RS1b's recalibration left it essentially unchanged rather
than causing it to "fill in." Read correctly: **RS1b confirms RS1a's finding survives training,
it does not add a new increment of evidence for the fill-in dynamic itself.** Shallow layers
(1–3) show minor scatter (±0.05) consistent with noisier estimation at low R², not a systematic
training effect.

**M1.6 — the pre-registered addressing criterion, not G7, is the number that matters, and it
argues against P.RS1.c holding cleanly.** G7 (noise-controlled attention) is a gate on the causal
patching test's validity, not the addressing verdict. The actual pre-registered criteria are
`output_above_noise` and `transitivity_confirmed`:

| state | g7_pass | output_above_noise | transitivity_confirmed |
|---|---|---|---|
| qwen3 (RoPE baseline) | 39/448 | 4/448 | **448/448 (100%)** |
| qwen3-dropped (RS1a, untrained) | 0/448 | 0/448 | 0/448 |
| qwen3-droped (RS1b, trained, PPL≈35) | 13/448 | 0/448 | **0/448** |

RS1a's own notebook already flagged that the untrained state's null is confounded ("the dropped
model is near-random... the null is confounded by global model breakdown... the genuine
addressing test is State 1 vs State 3"). State 3 is now functional (PPL 35, not near-random), so
this confound no longer applies — and the comparison it unblocks shows transitivity collapsing
from **100% of heads to 0%**. Per the spec's own falsifier language ("addressing appears or
disappears across the transition"), this reads as **addressing disappearing**, not "unchanged."
That is the spec's own "most surprising outcome" branch ("reopens the address question at the
training level"), not a clean pass for P.RS1.c.

G7's partial movement (0→13) shows *some* attention-steerability returned with training, while the
addressing-confirmation metrics stayed pinned at exactly zero — a dissociation between
steerability and addressing that echoes the RoPE model's own pattern (G7=39 but
output_above_noise only 4/448: steerable considerably more often than genuinely addressable even
at baseline). So the shape of the gap is consistent with the program's running theme; what's new
here is that in the trained DroPE'd model the addressing side of that gap didn't just stay small,
it went to zero.

**Open question this run cannot resolve: is the collapse a RoPE-removal effect, or an
under-training artifact?** The training plateaued at PPL≈35 (vs. ~21.8 target) at a peak LR
(3e-5) that DroPE's own paper's ablation shows is measurably under-tuned relative to their
default (3e-4) — see `rs1b-lr-retuning-note.md`. Two considerations cut in opposite directions:

- *Against under-training as the full explanation:* the collapse is complete (0/448), not
  graded. Under-training typically produces partial, proportional degradation, not a hard
  binary cutoff — and G7 *did* show partial, graded recovery, so the model wasn't uniformly
  undertrained across every mechanistic marker.
- *For under-training as a live explanation:* induction/transitivity-style circuits are known
  in the broader literature to require substantial training to emerge or reassemble, and our
  recalibration budget (1B tokens, ~2000 steps) is tiny next to that. It's plausible the same
  under-tuned LR that capped perplexity recovery starved a fragile, higher-order circuit
  (transitivity) far more severely than a simpler, linearly-decodable one (M1.5's raw
  key-position signal) — simple linear structure typically stabilizes faster during training
  than multi-hop induction-style mechanisms do.

This is not adjudicated by the current data. It is directly resolvable by the pending
LR-corrected rerun: if a properly-tuned recalibration (full perplexity recovery to near-RoPE
levels) still shows `output_above_noise`/`transitivity_confirmed` near zero, that is much
stronger evidence for a genuine, structural loss of addressing on RoPE removal. If it recovers
substantially, the current collapse was primarily a training-budget artifact.

### Conclusion / Next Step

RS1b's mechanistic probes are complete, but the result is more ambiguous than "the scaffold
hypothesis survives cleanly" — one prediction holds informatively-thinly (P.RS1.b, but adding no
new evidence beyond RS1a), and one is trending toward the spec's *most surprising* branch rather
than confirming C1 (P.RS1.c: addressing collapsed from 100% to 0% transitivity in a now-functional
model, not merely "stayed null").

1. **Position ≠ RoPE remains a stable, confirmed result** (RS1a + RS1b agree): key-position is
   decodable at near-ceiling fidelity with or without RoPE, with or without recalibration.
2. **Addressing did not recover, and by the unconfounded functional-model comparison, it may have
   actively been lost.** This is the more consequential, less comfortable reading of this run and
   should not be undersold as "M1.6 isn't sensitive enough" without first ruling out the LR
   confound.
3. **Next step is not optional now, it's load-bearing for interpretation:** rerun RS1b at a
   corrected LR (`rs1b-lr-retuning-note.md`, option A) and re-probe M1.6 on that checkpoint. The
   two possible outcomes cleanly discriminate between "addressing loss is a training artifact"
   and "addressing loss is a real consequence of removing RoPE" — which is a substantially
   different conclusion for the program's central C1 claim than what a partial-recovery framing
   would suggest.

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
