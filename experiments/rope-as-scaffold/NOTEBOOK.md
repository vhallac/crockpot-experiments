# RoPE-as-Scaffold Notebook

Newest entries first.

## 2026-07-28 — RS-amendment-2-3: RoPE-recalibrated control (amends RS2 and RS3)

### Question / Hypothesis

Every RoPE-vs-DroPE'd comparison in this program so far compares `qwen3` (zero extra training)
against `qwen3-droped` (~1B extra FineWeb-Edu tokens), confounding **RoPE removed** with
**received extra in-domain training**. RS3 made this blocking: both of its primary predictions
falsified in ways that confound cannot rule out (P.RS3.a falsified in exactly the direction
RS3-spec §1 said the confound would push it; P.RS3.b falsified with no conservative argument
available at all).

This experiment trains `qwen3-rope-recal` — the identical recalibration recipe with **RoPE left
active** — so the two trained arms differ only in RoPE, then re-runs the affected analyses.

Full pre-registration: [`RS-amendment-2-3.md`](RS-amendment-2-3.md) (arms, gates, predictions,
decision matrix); recipe of record: [RS1-spec §11](RS1-spec.md#11-addendum-2026-07-24-rs1b-ctrl--rope-recalibrated-confound-control).

### Experiment Design Summary

**Stage 0 (artifact generation):** train `qwen3-rope-recal` from
`Qwen/Qwen3-0.6B` @ `c1899de289a04d12100db370d81485cdf75e47ca`, identical corpus/seed/streaming
order/token budget/LR schedule to `qwen3-droped`, with `set_qwen_rotary_identity` **skipped**.

**Three analysis arms off that one checkpoint:**
- **Arm 1 (required):** RS3 Arms A (local_scramble) + B (induction) re-run.
- **Arm 2 (recommended):** M1.5 on the control, then `c2_subspace_overlap.py`.
- **Arm 3 (near-free):** frozen perplexity + M1.6 on the control.

### Planned Procedure

```bash
# Stage 0: train the control (~7h, H100 SXM)
export QWEN3_ROPE_RECAL_PATH=/workspace/qwen3-rope-recal
PYTHONPATH=experiments/dead-keys ./scripts/cuda-python --i-declare-gpu-readiness-pass \
  experiments/rope-as-scaffold/scripts/train_qwen3_nope.py \
  --base-model Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --no-rotary-patch \
  --train-tokens 1000000000 --train-context 2048 --lr 1e-3 --seed 0 \
  --cache-dir /workspace/rs1b-token-cache \
  --output-dir /workspace/qwen3-rope-recal

# (analysis arms follow after training — see checklist)
```

### Expected Signal / Interpretation Plan

Per `RS-amendment-2-3.md` §4–5. C3 verdict follows §5's 4-cell decision matrix.

### Pre-run Provenance
- Plan: `experiments/rope-as-scaffold/RS-amendment-2-3.md` (pre-registered 2026-07-28)
- Recipe of record: `RS1-spec.md` §11
- Code: `scripts/train_qwen3_nope.py` (+ `--no-rotary-patch`, `rope_active_probe`)
- Code: `deadkeys/common/loading.py` (+ `qwen3-rope-recal` tag, excluded from `DROPPED_ROPE_TAGS`)
- Code: `scripts/rs3_behavioral.py` (+ G-RS3.2 per-mode gate fix)
- Base model: `Qwen/Qwen3-0.6B` @ `c1899de289a04d12100db370d81485cdf75e47ca`
- Token cache reused from RS1b: `/workspace/rs1b-token-cache/`
- Comparison checkpoint: `qwen3-droped` at `/workspace/qwen3-droped` (RS1b LR=1e-3)
- GPU readiness report: `temp/gpu-readiness/20260728T160037Z-rs-amd23.md` — **GO**
- GPU: H100 SXM (pod `jn1bvsu2vxwdnw`, NE-1)
- Pre-run commit: 508db935cf8971a86dc10fec67d75626c8533948
- Planned output location: `outputs/rs_amd23_*`

### Results

Training completed on H100 SXM: 1907 steps, 999,817,216 tokens, 25,022s (~7.0h) at ~39,957 tok/s.
All numbers below independently recomputed from the per-item CSVs, not read off summary JSONs.

**Gates:**

| gate | verdict | evidence |
|---|---|---|
| G-ctrl.1 (RoPE active) | **PASS**, two ways | `training_manifest.json`: `rotary_patch_applied: false`, `rotary_probe.pass: true` (cos deviates from identity by 2.0, sin non-zero to 1.0). **Independently:** the control's step-1 eval PPL is **21.82**, matching the untrained RoPE baseline 21.80 — a RoPE-removed model starts at ~30,859. The identity patch definitively did not leak. |
| G-ctrl.2 (training matched) | **PASS** | 1907 steps, LR 1e-3, 1B tokens, ctx 2048, seed 0, batch 524288, AdamW β=(0.9,0.95) wd=0.1 — all matching `qwen3-droped`. Same cached token files (`fineweb_edu_qwen3_train_after_eval5000000_1000000000.uint32`). |
| G-ctrl.3 (functional) | **PASS** | held-out PPL 14.25 |
| G-ctrl.4 (G-RS3.1 re-run) | **PASS** | `rs3_gates.log`: `qwen3` measured 3.081886 vs target 3.0819 → PASS; `qwen3-droped` 2.826026 vs 2.826 → PASS. (Recovered from the pod's log — its summary JSON was overwritten, see Provenance defects.) |
| revision pinned | **PASS** | `c1899de289a04d12100db370d81485cdf75e47ca` recorded in the summary — RS3's regression is fixed |

**P.ctrl.a — local-order.** Paired per-block contrast, `qwen3-rope-recal` minus `qwen3-droped`,
scramble mode (positive = RoPE removal costs local acuity):

| w | paired Δ | 95% CI | verdict |
|---:|---:|---|---|
| 2 | **+0.0236** | [+0.0184, +0.0287] | holds |
| 4 | **+0.0441** | [+0.0386, +0.0494] | holds |
| 8 | **+0.0220** | [+0.0168, +0.0271] | holds |
| 16 | −0.0050 | [−0.0102, −0.0001] | reverses (marginal) |
| 32 | −0.0221 | [−0.0269, −0.0174] | reverses |

**P.ctrl.a holds at all three pre-registered windows** (w∈{2,4,8}), with `rel_delta_ce` agreeing
in sign at each. The sign reversal at w≥16 was not pre-registered and is reported as observed.

**P.ctrl.b — retrieval.** Induction gain (nats) by model and distance:

| d | `qwen3` (raw) | `qwen3-rope-recal` | `qwen3-droped` | RoPE-attributable share |
|---:|---:|---:|---:|---:|
| 64 | 12.942 | 12.156 | 11.967 | 19% |
| 256 | 12.961 | 12.104 | 11.720 | 31% |
| 512 | 12.930 | 12.021 | 11.229 | 47% |
| 1024 | 12.925 | 11.610 | 9.995 | 55% |
| 1536 | 12.938 | 11.094 | 9.409 | 48% |

Paired `rope-recal` − `droped` is positive with CI excluding zero at **every** distance
(+0.189, +0.384, +0.792, +1.614, +1.685). Retrieval **is** damaged by RoPE removal. But
`rope-recal` also falls well short of raw `qwen3` (−0.79 to −1.84), so recalibration alone
damages retrieval too.

**P.ctrl.c (subspace control) — NOT DELIVERED.** See Provenance defects.

**P.ctrl.d — perplexity.** `qwen3-rope-recal` **14.25** (CE 2.6569) vs `qwen3-droped` **16.88**
(CE 2.826) vs untrained `qwen3` **21.80** (CE 3.0819). Decomposition of RS1b's headline:

- domain adaptation gains **0.425 nats** (3.0819 → 2.6569)
- RoPE removal costs **0.169 nats** (2.6569 → 2.826)
- net vs. untrained baseline = 0.256 nats — exactly what RS1b reported as "recovery past baseline"

**P.ctrl.e (M1.5/M1.6 profile) — NOT DELIVERED, blocked with cause.** See below.

**Arm 3b (M1.6) blocked — G6 marker-neutrality gate could not be satisfied.** M1.6's G6 gate
requires the model to be near-indifferent between four candidate continuation markers
(`max_prob/min_prob < 3.0`). On `qwen3-rope-recal` the marker search failed at both attempted
budgets: best ratio **5.534** at `--max-marker-sets 512`, and **3.311** at 4096 (on stimulus
M16_02; M16_01 passed at that budget). Since G6 is foundational, the entire downstream pipeline —
causal K/V patching, induction scoring, transitivity, per-head classification — was correctly
skipped rather than run on non-neutral markers, which would have invalidated every readout.

The cause is **model quality, not training exposure**: `qwen3-droped` received *identical*
training (same token stream, same 1907 steps) and passed G6 at a 4096 budget. What differs is
that the RoPE-active control is a better model (14.25 vs 16.88 PPL), and sharper predictive
distributions make near-indifference harder to find. Across three checkpoints the relationship is
monotone in perplexity: `qwen3` (21.80) passes at the default 512; `qwen3-droped` (16.88) needs
4096; `qwen3-rope-recal` (14.25) fails at 4096. The general instrument problem is scoped for
separate investigation in [`G6-ceiling-investigation.md`](G6-ceiling-investigation.md).

> **Status revised 2026-07-29: a retry is planned, so this arm is open, not abandoned.** A retry
> **must raise `--max-marker-sets` substantially or it will fail identically** — 4096 samples
> ~0.02% of the C(152,4) ≈ 21M marker-combination space, so the earlier failure is not evidence
> that no neutral set exists. Suggested 65536 (~20–30 min of forward passes). Do **not** relax the
> 3.0 threshold or expand `CANDIDATE_MARKERS` to force a pass: either would void comparability
> with every prior M1.6 result and would require re-running all previously-probed models.

**Provenance defects (recorded, not fixed):**
1. **Arm 2 prerequisite was the wrong probe.** The `m15/` output directory contains a *Task 4 band
   census* (`census_*.csv`, `spectra_*.npz`, `bands_manifest.csv`) — not M1.5 `position_content`
   output. No `kaddress_m15_projectors_qwen3-rope-recal.npz` was produced, so
   `c2_subspace_overlap.py` could not be run and **P.ctrl.c is undelivered**. The directory name
   is misleading; the artifacts are from a different instrument.
   **Root cause:** `-m deadkeys.scripts.census` was run instead of
   `-m kaddress.scripts.position_content`. Three scripts look like plausible "M1.5" candidates and
   only one is (`deadkeys.scripts.phase1_5` is a third, used in this program solely for the
   sliding-window perplexity helper). `census` accepts neither `--families` nor `--max-length`, so
   the spec's own flags were the available tell and were not cross-checked. Disambiguation table
   added to [RS1-spec](RS1-spec.md) after the fact.
   **For any future re-run of this arm:** the command is
   `python -m kaddress.scripts.position_content --model qwen3-rope-recal --families A
   --max-length 1024`, and it must emit `kaddress_m15_projectors_qwen3-rope-recal.npz`. The
   checkpoint remains available on volume `6qaba2cjcx` at `/workspace/qwen3-rope-recal`, so this
   is re-runnable without retraining.
2. **Summary JSON overwritten twice.** `run_analysis.sh` wrote the gates run, the `local_scramble`
   arm, and the `induction` arm all to the same `--output-dir`. The surviving
   `rs3_summary.json` is `induction`'s, so G-RS3.1's and G-RS3.2's recorded results were both
   lost. G-RS3.1 was recovered from `rs3_gates.log` (passed, above). **G-RS3.2 has no recorded
   pass**; recomputed independently here from the CSV: scramble is monotone non-decreasing for all
   three models, reverse is non-monotone for all three — same pattern as RS3, and the per-mode
   split (fix 2d) would have recorded scramble as passing.
3. **`training_manifest.json` mislabels the artifact** as `"artifact": "qwen3-droped"` — a
   hardcoded string in the training script. `output_dir` is correctly `/workspace/qwen3-rope-recal`
   and the rotary fields are correct, so this is cosmetic, but it should be fixed before the
   script generates another artifact.
4. **Checkpoint is unpublished.** Verified intact on network volume `6qaba2cjcx` (`dead-weight-ne1`,
   US-NE-1) at `/workspace/qwen3-rope-recal` — 1.19GB safetensors plus tokenizer, 4.5GB with
   training checkpoint. Note this is a *different* volume from the project's documented
   `et0mntsj6x` (US-CA-2) that `AGENTS.md` referenced at the time — and which no longer exists (the RunPod API now lists `6qaba2cjcx` as the account's only volume; `AGENTS.md` and the pod scripts were corrected 2026-07-29). Checklist steps 8–9 remain open.

### Analysis

**P.ctrl.a holds: RS3's local-order falsification was a confound artifact, exactly as
pre-registered.** RS3-spec §1 argued in advance that domain adaptation would bias the DroPE'd
model toward *appearing more* order-sensitive, working against P.RS3.a — and that a falsification
in that direction would therefore be uninterpretable. With training held constant, the sign
flips: the RoPE-active control is more order-sensitive than the RoPE-removed model at every
pre-registered window. **RoPE removal does cost local-order acuity.** The effect is small in
absolute terms (+0.044 nats at w=4) but robust, and it survives the `rel_delta_ce` normalization
check.

The reversal at w≥16 sharpens rather than muddies this: the RoPE-attributable cost is confined to
*short* windows and vanishes (then inverts) at long ones. That is precisely the profile C3's
positive clause predicts — RoPE supplies **local** order structure.

**P.ctrl.b lands between the prediction and its falsifier, and the decomposition is the finding.**
Neither branch fires cleanly: the control neither retains raw `qwen3`'s gain (as "the collapse is
RoPE-specific" predicted) nor matches `qwen3-droped` (as "the collapse is training-induced"
predicted). Both mechanisms are real and separable:

- **Recalibration alone** costs 0.79–1.84 nats of induction gain — narrow-domain training on
  FineWeb-Edu degrades retrieval on random-token spans regardless of RoPE.
- **RoPE removal costs more on top**, at every distance, with CIs excluding zero.
- The **RoPE-attributable share grows with distance**: 19% at d=64, ~47–55% at d≥512.

So **RS3 overstated the RoPE-attributable retrieval loss by roughly 2×** at the d=512 anchor
(attributing the full 1.70 nats to RoPE when only 0.79 is), but RS3's *direction* survives the
control intact.

**C3 is falsified — cleanly this time.** C3's distinguishing claim is that RoPE's causal
contribution is local **and not** retrieval. With the confound removed, removal damages **both**,
and retrieval far more in proportional terms: at the pre-registered anchors, `D_local = +0.011`
versus `D_retrieval = +0.066`, a ~5.8× ratio in the direction opposite to C3's ordering. The
"not retrieval" clause fails on controlled evidence, not on confounded evidence as in RS3.

**C3's spirit survives its letter, and the RS3 reframing is now the better-supported account.**
Two independent signatures point the same way: the local-order cost is confined to w≤8, and the
retrieval cost grows monotonically with distance. Both are consistent with the hypothesis floated
in the RS3 entry — that induction copying depends on a *relative-offset* operation ("attend to the
token after my last match") that RoPE supplies natively, so RoPE's contribution really is a local
primitive, but retrieval circuits are **built on** that primitive rather than independent of it.
This does not contradict E2 (the retrieval address is content; position supplies the offset step).
It reframes C3 rather than merely refuting it.

**P.ctrl.d is the sleeper result: the DroPE recipe does not fully recover.** RS1b reported
recovery to PPL 16.88, "below the 21.80 baseline," which read as an unqualified success. Against
the correct control that reading dissolves: adaptation on this corpus is worth 0.425 nats, RoPE
removal costs 0.169 nats back, and the 0.256-nat net is what RS1b saw. A **persistent 18.5%
perplexity penalty** (16.88/14.25) survives full recalibration at this token budget. The
sub-baseline crossing at step ~800 flagged in the RS1b entry was domain adaptation, as suspected
there — but the residual gap is new, and it qualifies the program's "scaffold is freely
removable" framing at the language-modelling level, independent of C1/C2/C3.

**A defect in this experiment's own decision matrix.** `RS-amendment-2-3.md` §5 row 1 glosses
"P.ctrl.b holds" as "(recal retains gain) → retrieval is spared," but P.ctrl.b's own prediction
text in §4 says the opposite — that b holding means the collapse *is* RoPE-specific, i.e.
retrieval **is** damaged. The two are contradictory, so the matrix cannot be applied literally.
The observed data is unambiguous enough to adjudicate without it (both axes damaged, retrieval
more), but the matrix is corrected in the spec rather than left as a trap for a future reader.

### Conclusion / Next Step

**C3: falsified.** RoPE removal costs both local-order acuity and retrieval, with retrieval hit
~5.8× harder proportionally. C3's positive clause (a local cost exists) is vindicated — and was
being masked by the confound in RS3 — but its distinguishing "not retrieval" clause fails on
controlled evidence.

**The control did its job.** It reversed one RS3 verdict (P.RS3.a: confound artifact, now holds),
halved the magnitude of another while preserving its direction (P.RS3.b), and surfaced a finding
neither RS3 nor RS1b could see (P.ctrl.d's persistent perplexity penalty). This is the clearest
vindication so far of pre-registering the confound analysis before the data arrives.

**THE PROGRAM-LEVEL FINDING.** This experiment was targeted at a specific thesis — *RoPE is a
training scaffold that can be thrown away.* **The findings do not support it. RoPE continues to
add value at inference, and removing it costs capability that recalibration does not restore.**
Four independent measurements converge on this:

| measurement | cost of removing RoPE |
|---|---|
| P.ctrl.d (perplexity) | +0.169 nats / 18.5% PPL, persisting after full recalibration |
| P.ctrl.a (local-order) | +0.044 nats at w=4, CI excluding zero |
| P.ctrl.b (retrieval) | +0.79 nats at d=512, growing to +1.68 at d=1536 |
| RS3 Arm D (context) | collapse past the recalibration window (PPL 298 @8192 vs RoPE's flat 21.5) |

**The failure is not that position disappears — it is that reconstructed position is not
functionally equivalent to supplied position.** C1 holds (position fills in) and C2 holds
(it reconstructs RoPE's own code in L3–L12, confound-controlled). The model reconstitutes the
representation and *still cannot do the job*. That is this program's own motivating principle —
*decodable ≠ causally used* — turned back on the program: the scaffold thesis quietly assumed the
converse, that reconstructed-and-decodable implies functionally-replaced. It does not.

**Scope discipline on that claim:** the evidence supports *"not recoverable at a 1B-token
recalibration"*, not *"not recoverable in principle."* The training curve converged, but on a
cosine schedule at that budget. This does not weaken the practical finding — DroPE's own selling
point is that a **brief** recalibration suffices, and at the budget claimed to suffice, it
demonstrably does not.

**Delivered:** P.ctrl.a, P.ctrl.b, P.ctrl.d, and the C3 verdict (the blocking output).
**Open, not abandoned (revised 2026-07-29):** P.ctrl.c (M1.5 re-run — the checkpoint is intact on
volume `6qaba2cjcx`, so this needs one probe pass, not retraining) and P.ctrl.e (M1.6 retry — see
the `--max-marker-sets` requirement above). **This experiment is therefore not yet complete**;
checklist items 7–12 remain open.

**Consequences to carry forward:**
1. **Back-propagated to the 2026-07-27 RS3 entry** — see the amendment note appended there.
2. **RS2/RS2.1's training-drift baseline is pending P.ctrl.c.** C2's verdict is unchanged but
   still rests on an initialization control only, with generic training drift unexcluded until
   the M1.5 re-run lands.
3. **M1.6's quality ceiling is scoped separately** in
   [`G6-ceiling-investigation.md`](G6-ceiling-investigation.md). **This blocks planning for RS4**,
   whose E1/E2 spot-check assumes M1.6 runs on a *larger* — and therefore better, and on this
   evidence more likely to fail G6 — model. The pending retry here will inform it either way.
4. **P.RS1.a's framing needs revisiting** in light of P.ctrl.d: "perplexity recovers" is true only
   against an un-adapted baseline.
5. **Successor research directions are enumerated in [`DIRECTIONS.md`](DIRECTIONS.md)** — six
   candidates, costed, most of them $0 analysis over numbers this experiment already produced.
   Per the program's working principle: *an experiment's value may lie chiefly in producing its
   successor,* and that is an acceptable outcome.

---

## 2026-07-27 — RS3 functional locus of RoPE (local-order vs. retrieval)

### Question / Hypothesis

C3 (behavioral): removing RoPE costs **local-order / recency acuity**, not
**retrieval / content-addressing**. RoPE's causal contribution is local, not an address.

### Experiment Design Summary

Four-arm eval harness over three Qwen3-0.6B states:
1. `qwen3` — RoPE baseline (pinned revision `c1899de289a04d12100db370d81485cdf75e47ca`)
2. `qwen3-droped` — DroPE'd, LR=1e-3 rerun (`/workspace/qwen3-droped`)
3. `qwen3-dropped` — untrained identity-RoPE floor

Arms:
- **A (local_scramble):** CE delta under scrambled/reversed windows w∈{2,4,8,16,32}
  over 1M FineWeb-Edu tokens (offset 5M past RS1a's slice). Primary metric.
- **B (induction):** synthetic repeated-span retrieval gain, d∈{64,256,512,1024,1536},
  n_seq=256 per distance. Primary metric.
- **C (kv_retrieval):** key–value needle retrieval, M=40 lines, n_seq=128 per depth.
  Secondary, floor-gated.
- **D (length_ce):** CE at contexts {1024,2048,4096,8192}. Exploratory only.

All primary metrics are **within-model contrasts** computed on identical items
(generated once, seed 0, then scored by every model).

### Planned Procedure

```bash
# Stage 1: G-RS3.1 gate — verify harness reproduces known CE
RUN_ID=rs3_behavioral_$(date -u +%Y%m%dT%H%M%SZ)
export QWEN3_DROPED_PATH=/workspace/qwen3-droped

PYTHONPATH=experiments/dead-keys ./scripts/cuda-python \
  experiments/rope-as-scaffold/scripts/rs3_behavioral.py \
  --task gates --models qwen3 qwen3-droped \
  --eval-offset-tokens 0 --eval-tokens 5000000 \
  --output-dir outputs/${RUN_ID}_gates

# Stage 2-5: Four arms (run sequentially on same pod)
for TASK in local_scramble induction kv_retrieval length_ce; do
  PYTHONPATH=experiments/dead-keys ./scripts/cuda-python \
    experiments/rope-as-scaffold/scripts/rs3_behavioral.py \
    --task $TASK --models qwen3 qwen3-droped qwen3-dropped \
    --eval-tokens 1000000 --eval-offset-tokens 5000000 \
    --output-dir outputs/${RUN_ID}_${TASK}
done
```

### Expected Signal / Interpretation Plan

Per RS3-spec.md §4-5:
- **P.RS3.a:** DroPE'd less order-sensitive at w∈{2,4,8} — `delta_ce_droped < delta_ce_rope`.
- **P.RS3.b:** `induction_gain_droped ≥ induction_gain_rope` at every d.
- **P.RS3.c:** Proportional degradation larger on local axis than retrieval axis.
- **P.RS3.d:** State 2 (untrained) shows near-zero induction_gain, compressed delta_ce.

Decision tree in §5.

### Pre-run Provenance
- Spec: `experiments/rope-as-scaffold/RS3-spec.md` (pre-registered 2026-07-27)
- Code: `experiments/rope-as-scaffold/scripts/rs3_behavioral.py`
- Code: `experiments/rope-as-scaffold/scripts/eval_perplexity.py` (extended with `offset_tokens`)
- Model revisions: `qwen3`/`qwen3-dropped` at `c1899de289a04d12100db370d81485cdf75e47ca`
- `qwen3-droped` checkpoint: `/workspace/qwen3-droped` (RS1b LR=1e-3)
- Pre-run commit: `1f14b5a`; batching/Arm-D-crash fix `c639f75` applied before the full run
  (first attempt on an A5000 hit the unbatched-forward slowness and Arm D crash described
  below; both fixed and re-run, see `temp/gpu-readiness/20260727T202200Z-rs3-fixed.md`)
- Output location: `outputs/rs3_20260727T204900Z/{local_scramble,induction,kv_retrieval,length_ce}/`

### Results

All four arms completed on the RTX A5000 pod after the batching fix (~50 min total, vs. an
extrapolated >4h for the unbatched version). Numbers below are independently recomputed from
the per-item CSVs, not taken from the run's own summary JSONs (which do match, once checked).

**Arm A — local scramble, paired per-block contrast (RoPE − DroPE'd), scramble mode:**

| w | paired Δ | 95% CI | rel_delta agrees? |
|---:|---:|---|---|
| 2 | +0.044 | [+0.035, +0.053] | yes (droped > rope) |
| 4 | **−0.024** | [−0.034, −0.014] | yes |
| 8 | **−0.087** | [−0.098, −0.077] | yes |
| 16 | −0.140 | [−0.151, −0.128] | yes |
| 32 | −0.165 | [−0.177, −0.154] | yes |

Positive = RoPE more sensitive (prediction). Only w=2 goes the predicted direction; w≥4 all
have CIs excluding zero in the *wrong* direction. `rel_delta_ce` (delta/clean, the spec's
normalization check) agrees with the absolute-delta sign at **every** window, not just
w∈{2,4,8} as originally planned — no normalization ambiguity here.

**Arm B — induction gain, paired per-sequence contrast (RoPE − DroPE'd):**

| d | 64 | 256 | 512 | 1024 | 1536 |
|---|---:|---:|---:|---:|---:|
| paired Δ | +0.97 | +1.24 | +1.70 | +2.93 | +3.53 |
| 95% CI excludes 0? | yes | yes | yes | yes | yes |

RoPE's gain is flat (~12.93 nats at every distance). DroPE'd decays monotonically
(11.97 → 9.41) as distance grows. The gap is entirely in `ce_second` (the retrieval
readout) — `ce_first` (unpredictable-by-construction floor) is ~13.1 for both, confirming
the contrast isolates retrieval and not some other difference between the sequences.

**Arm C — KV retrieval (secondary, gate-checked):** top-1 accuracy ratio DroPE'd/RoPE ranges
0.11 (depth 1) to 0.68 (depth 40); RoPE 0.77–0.91, DroPE'd 0.10–0.55 across depths 1–40.
Corroborates Arm B's direction and magnitude.

**Arm D — length behavior (exploratory only, per spec §2.6):** RoPE is flat across context
(PPL 21.5–22.7 at 1024–8192). DroPE'd is flat within its 2048 recalibration context
(PPL 17.6 @1024, 16.9 @2048) but collapses past it: PPL 51.3 @4096, 298.4 @8192. Per spec,
this cannot itself adjudicate C3 (RoPE was pretrained at 32k, so this isn't length
extrapolation for it), but it rules out one confusion: Arm B's induction sequences top out
at 1600 tokens, inside DroPE'd's stable range, so the retrieval collapse in Arm B is not
downstream of this context-length effect.

**Gates:**
- **G-RS3.1** (harness reproduces known CE) — not directly evidenced for this run; the
  logged gate check at 18:00Z ran the pre-batching code. Arm D's own qwen3/qwen3-droped
  ctx=2048 numbers (CE 3.0826 / 2.8263) are consistent with the frozen values (3.0819 /
  2.826) to within the gate's tolerance, so the harness is very likely fine, but this is
  incidental corroboration, not the pre-registered gate run against this exact code.
- **G-RS3.2** (perturbation validity) — **recorded as FAILED for all three models** in the
  run's own summary JSON. Decomposed by mode: **scramble is monotone non-decreasing in w for
  all three states** (passes cleanly); **reverse is non-monotone for all three** (peaks at
  w=4, then declines through w=32). The gate as written doesn't split by mode, so a real
  reverse-mode anomaly sank the whole gate including the clean scramble result.
- **G-RS3.3 / G-RS3.4** (floor checks) — passed, but trivially: thresholds (0.5 nats;
  3×chance≈1.6e-4) are far below the observed effects (12.9 nats; 0.91 accuracy). Not
  useful discriminators as written.
- **Provenance regression:** all four summary JSONs record `revision: null`. The notebook's
  planned procedure and RS1-spec §10 both pin `qwen3`/`qwen3-dropped` to
  `c1899de289a04d12100db370d81485cdf75e47ca`; that flag was not passed on this run. Follow-up
  below.

### Analysis

**P.RS3.a (local-order cost): falsified as stated.** The DroPE'd model is *more*
order-sensitive than RoPE at w≥4, not less, with CIs excluding zero and no
normalization-dependence to hide behind. Only the smallest window (w=2) goes the predicted
direction, and even that is a small effect (+0.044 nats) next to the w=32 reversal (−0.165).

**Reverse-mode non-monotonicity is a real, interpretable finding, not just a gate technicality.**
All three states — including the untrained floor — peak sensitivity at w=4 and *decline*
through w=32 under reversal. Reversal at large w preserves local (adjacent-token) structure
better than random scramble does (only the window's two endpoints move far; the interior
stays locally ordered relative to reversal, whereas scramble destroys it uniformly), so a
reversal-specific peak-then-decline is mechanistically plausible. That reading is post-hoc,
however, and the gate failure should stay visible rather than be quietly explained away —
scramble-mode is the arm's clean primary result; reverse-mode is reported but flagged
gate-failed.

**P.RS3.b (retrieval preserved): falsified, and more decisively than P.RS3.a.** All five
distances show CIs excluding zero in the falsifying direction, the effect grows monotonically
with distance, and Arm C corroborates both direction and rough magnitude independently. This
is the strongest single result in RS3.

**P.RS3.c (the axis contrast): C3's own falsifier is triggered.** At the pre-registered
anchors (w=4, d=512): `D_local = −0.006` (RoPE and DroPE'd are within noise of each other),
`D_retrieval = +0.132` (retrieval degraded substantially, proportionally, relative to RoPE's
own gain). Proportional retrieval loss exceeds proportional local-order loss —
`D_retrieval > D_local` — which is precisely the falsifier RS3-spec §4 wrote down in advance
for C3.

**P.RS3.d (untrained floor): holds.** State 2 shows negative induction gain (ce_second >
ce_first — the untrained model is actively *worse* at the retrieval position, not merely at
zero) and heavily compressed local-order deltas relative to the other two states. The
instruments register a known-degenerate model as degenerate, as designed.

**Why this cannot yet be reported as "C3 falsified" — the confound RS3-spec §1 flagged in
advance is live on both axes, asymmetrically:**
- **Local axis:** §1's own bias-direction argument said the 1B-token domain-adaptation
  confound would push the DroPE'd model toward *appearing more* order-sensitive than a
  clean RoPE-removal effect would show — i.e. the confound biases *against* P.RS3.a. The
  observed result (DroPE'd more sensitive) is exactly the confound's predicted direction.
  That does not mean the result is *only* the confound — but it means "no real local-order
  cost, fully explained by domain adaptation" and "a real local-order cost, inflated further
  by domain adaptation" are both consistent with what was measured, and this run cannot
  distinguish them.
- **Retrieval axis:** §1 stated plainly that no equivalent conservative argument exists here.
  The induction collapse could be RoPE removal, or could be catastrophic forgetting of the
  induction circuit during narrow-domain (FineWeb-Edu-only) recalibration — a generic
  effect of the training regime, not specifically of dropping RoPE.

Both readings were anticipated in the spec (§5's decision tree, the "borderline" branch),
but the actual outcome — **both axes falsified, with the retrieval axis falsified far more
strongly** — is not one of the three named branches in §5 (a✓b✓, a✓b✗, a✗b✓). It is closest
in spirit to the borderline/undecidable branch and should be treated the same way: **route
to RS1b-ctrl before adjudicating C3**, rather than force-fitting it into a branch that
assumed at least one axis would hold.

**A mechanistic hypothesis worth stating, not yet tested:** induction copying requires
attending to "the token after my last match" — a relative-offset operation RoPE supplies
natively via its rotation structure. If DroPE'd's emergent position (C1/C2's finding) encodes
*absolute* position well but reconstructs RoPE's *relative-offset* affordance only
weakly — plausible, since RS2/RS2.1's reconstruction was strongest in early-mid layers and
the induction mechanism is typically attributed to specific mid-to-late attention
heads — that would explain retrieval collapse *without* contradicting C1/C2's "position fills
in and partially reconstructs RoPE's code" or E2's "position is not itself the retrieval
address." It would reframe C3 (retrieval depends on RoPE only *through* a positional
primitive it fails to fully reconstruct) rather than simply falsify the program's account.
This is a hypothesis for a future RS3.x/RS4 arm, not a claim this run supports.

### Conclusion / Next Step

> **AMENDED 2026-07-28 — resolved by RS-amendment-2-3 (see the newest entry).** The hold below was
> correct and the control vindicated it. Summary of what changed:
> - **P.RS3.a's falsification was a confound artifact.** With domain adaptation held constant, the
>   sign flips: the RoPE-active control is *more* order-sensitive than the DroPE'd model at every
>   pre-registered window (+0.024/+0.044/+0.022 at w=2/4/8, CIs excluding zero). RoPE removal does
>   cost local-order acuity, and RS3 could not see it.
> - **P.RS3.b's direction survives, but its magnitude was ~2× overstated.** At d=512 the full
>   1.70-nat gap decomposes into 0.91 nats of recalibration effect and 0.79 nats attributable to
>   RoPE removal. Retrieval *is* damaged by removal, at every distance, CIs excluding zero.
> - **C3 is now falsified on controlled evidence:** removal damages both axes, retrieval ~5.8×
>   more proportionally (D_local +0.011 vs D_retrieval +0.066). C3's "not retrieval" clause fails.
> - The mechanistic reframing floated below (retrieval circuits built *on* RoPE's local
>   relative-offset primitive) is now the better-supported account, on two independent signatures:
>   the local cost is confined to w≤8, and the retrieval cost grows with distance.
>
> The original reasoning is retained verbatim below as the record of what was known at the time.

**No C3 verdict yet.** RS3's instruments and paired-contrast design worked as intended —
clean gates on the primary scramble-mode Arm A metric, a decisive and internally-corroborated
Arm B/C result, and a diagnostic Arm D that rules out one alternative explanation — but the
result (both axes degrade, retrieval much more than local order) lands on exactly the
confounded case RS3-spec §1 and §5 pre-registered a hold for. Reporting "C3 falsified" now
would mean overriding my own pre-registered caution the moment the data made it inconvenient.

**Required before adjudicating C3:** run **RS1b-ctrl**
([RS1-spec §11](RS1-spec.md#11-addendum-2026-07-24-rs1b-ctrl--rope-recalibrated-confound-control)) —
recalibrate a copy of unmodified RoPE Qwen3-0.6B on the identical FineWeb-Edu
corpus/recipe/token-budget as `qwen3-droped`, then re-run RS3 Arms A/B against that model
instead of the raw pretrained RoPE baseline. This isolates domain-adaptation effects from
RoPE-removal effects on both axes at once. Cost: ~7h H100 (cf. RS1b) plus a cheap RS3 re-run
(~1h given the now-batched harness).

**Two follow-ups outside the RS1b-ctrl gate, not blocking it:**
1. **Provenance regression** — re-run (or at minimum, verify after the fact) with the pinned
   revision `c1899de289a04d12100db370d81485cdf75e47ca` passed explicitly. `--revision` exists
   as a CLI flag; the pre-planned command in this entry's Planned Procedure never sets it, and
   neither did the actual invocation.
   **How the pin was lost (as reported by the implementing session, not independently verified
   by inspecting its tool-call log directly):** the gate check and first launch attempt did
   pass `--revision c1899de2…`; the relaunch after fixing a `QWEN3_DROPED_PATH` env-var bug
   rebuilt its command from an inline script that dropped the `--revision` flag, so all three
   models in the run that actually produced this entry's data loaded with `revision=None`
   (→ `main`). Consistent with independently-observed local evidence: the monitor logs
   (`temp/rs3-monitor-20260727T204052Z.txt`) show a string of `QWEN3_DROPED_PATH` `ValueError`s
   from a failed earlier attempt, and every summary JSON from the final run records
   `revision: null` — both facts check out against this account, though the account itself
   (which specific launch used which flag) is relayed, not directly verified here.
   **Recovered value — best-effort, not a direct verification:** queried the live Hugging Face
   API for `Qwen/Qwen3-0.6B` — `main` currently resolves to
   `c1899de289a04d12100db370d81485cdf75e47ca`, the exact pinned SHA, last modified 2025-07-26
   with no newer commits since. A same-day run against an unpinned `main` on a stable,
   already-released model should resolve to this same commit. **Caveat, still open:** this was
   not confirmed against the pod's own HF cache snapshot, which would be dispositive — the pod
   (`nci2dn93kj36vr`, RTX A5000, no persistent network volume — all state including the HF
   cache lived on that specific host's local container disk) went `EXITED` after the run and
   its host had no free GPU capacity to restart on 8 repeated attempts over ~4 minutes
   (`runpodctl pod start` failing with "not enough free GPUs on the host machine"), so the
   actual cached snapshot hash could not be read. Residual risk: if the pod's HF cache reused a
   stale local snapshot instead of checking `main` fresh, the effective revision could differ.
   **Adopted value for this entry: `c1899de289a04d12100db370d81485cdf75e47ca`, high-confidence
   but not certain.** Re-verify directly (pod cache, or an explicit `--revision`-pinned re-run)
   before treating this as settled provenance.
   **Separate, pre-existing gap, not new to RS3:** the FineWeb-Edu corpus
   (`HuggingFaceFW/fineweb-edu`, `sample-10BT`) has never been revision-pinned at any point in
   this program — RS1-spec §10 pins the model SHAs `[MUST]` but only pins the dataset's
   name/config/split, not a commit. Not a regression; worth fixing whenever §10 is next revised.
2. **G-RS3.2 gate logic** should split by mode (`scramble` vs `reverse`) rather than failing
   the whole gate on one mode's non-monotonicity — the reverse-mode finding is informative,
   not a defect, and shouldn't silently mask the clean scramble-mode pass. Fix before RS4
   reuses this pattern.

**Arm D's context-collapse finding** (PPL 298 at 8192, 17x the in-context value) is
independently interesting for the DroPE recipe itself, separate from the C3 question — a 1B
token / 2048-context recalibration does not confer any length robustness past the
recalibration window. Worth a line in any future write-up of the DroPE recipe's limitations,
regardless of how C3 resolves.

---

## 2026-07-26 — RS2.1 subspace reconstruction vs. initialization inheritance (completed)

### Question / Hypothesis

RS2 concluded C2 "substantiated" but did not control for initialization inheritance: the
DroPE'd model started from RoPE's own weights, so its overlap with RoPE's `k_post` could
reflect the base model's geometry rather than training-time reconstruction. RS2.1 re-tests
C2 with three controls:

- **V1:** Depth-resolved reference comparison (tabulation of RS2's existing CSV) — is primary
  excess (trained vs `k_post`) ≤ Ref2 excess (RoPE `k_pre` vs `k_post`)? If the internal
  rotation channel alone suffices to explain the result, no reconstruction was measured.
- **V2:** Zero-shot inheritance control — run the identical overlap computation against the
  **untrained** dropped model's projectors. Δ = trained_excess − untrained_excess.
- **V3:** `k_pre`-partialled residual test — orthogonalize both `k_post` and DroPE'd bases
  against `k_pre`, then measure overlap of the rotation-specific residuals.
- **V4:** Projector semantics — cross-reference position_fraction at each layer to ensure
  the measured alignment reflects positional geometry, not content.
- **V5:** Statistical robustness — layer-clustered SEs, rank-sensitivity check, L1 inclusion.

**Falsifier:** V2 Δ ≈ 0 and V3 residual excess ≈ 0 in layers 2–12 → RS2's overlap is
attributable to initialization inheritance, not reconstruction; C2 should be re-adjudicated.

### Experiment Design Summary

- **Inputs:** Four pre-existing artifacts — RoPE M1.5 projectors (RS1a), untrained-dropped
  M1.5 projectors (RS1a), trained DroPE'd M1.5 projectors (RS1b LR=1e-3), RS2 per-head CSV.
- **Method:** V1/V2 reuse `c2_subspace_overlap.py` unmodified (V2 points `--droped-projectors`
  at the untrained-dropped NPZ). V3 uses new `--residual-against-pre` flag for
  orthogonalization + residual principal-angle overlap. V4/V5 are CSV/JSON post-processing.
- **CPU-only, analysis-only.** No GPU, no training, no pod needed. All NPZ files are under
  300MB; computation is O(heads × d_head³) ≈ minutes.

### Planned Procedure

```bash
# V2: untrained-dropped zero-shot inheritance control
PYTHONPATH=experiments/dead-keys:experiments/k-address-space ./scripts/nix-cpu-run \
  experiments/rope-as-scaffold/scripts/c2_subspace_overlap.py \
  --rope-projectors outputs/rope_as_scaffold_rs1a_20260724T0559Z/m15_qwen3/kaddress_m15_projectors_qwen3.npz \
  --droped-projectors outputs/rope_as_scaffold_rs1a_20260724T0559Z/m15_qwen3_dropped/kaddress_m15_projectors_qwen3-dropped.npz \
  --output-dir outputs/rs2.1_v2_untrained_<timestamp> \
  --baseline-trials 100 --families A --seed 0

# V3: trained DroPE'd with residual-against-pre
PYTHONPATH=experiments/dead-keys:experiments/k-address-space ./scripts/nix-cpu-run \
  experiments/rope-as-scaffold/scripts/c2_subspace_overlap.py \
  --rope-projectors outputs/rope_as_scaffold_rs1a_20260724T0559Z/m15_qwen3/kaddress_m15_projectors_qwen3.npz \
  --droped-projectors outputs/rs1b_probes_lr1e3_qwen3_droped_20260726/outputs/rs1b_probes_lr1e3_20260726T042955Z_m15_qwen3_droped/kaddress_m15_projectors_qwen3-droped.npz \
  --output-dir outputs/rs2.1_v3_residual_<timestamp> \
  --baseline-trials 100 --families A --seed 0 \
  --residual-against-pre
```

V1 (tabulation from RS2's CSV), V4 (position_fraction cross-reference), and V5 (statistical
post-processing) use the committed data; they do not require separate script runs.

### Expected Signal / Interpretation Plan

See RS2.1-spec.md §4–5 for the full decision matrix. In brief:

- **(P.RS2.1.a) Reconstruction survives inheritance control in layers 3–12.** V2 Δ > 0 with
  layer-clustered CI excluding zero, and V3 residual excess > 0.
- **(P.RS2.1.b) Layer 2 is not representative.** Its Δ is small relative to its untrained
  baseline — the notebook's illustrative example is inheritance-dominated regardless of the
  aggregate verdict.
- **(P.RS2.1.c) Late-layer divergence is not a rank artifact.** Per V5b, excess does not
  correlate with PCA-rank differences.

### Pre-run Provenance

- Spec: `experiments/rope-as-scaffold/RS2.1-spec.md` (pre-registered at 1860a46).
- Pre-run commit: `e8f84b9` (RS2.1 prepare: V3 flag, orthogonalize_against, notebook entry).
- Code: `experiments/rope-as-scaffold/scripts/c2_subspace_overlap.py` (extended with
  `--residual-against-pre` flag + `orthogonalize_against` function).
- Motivating critique: `temp/rs2-critiqe.md`.
- Execution: local CPU, NixOS, `./scripts/nix-cpu-run`.

### Run Evidence

- V2 output: `outputs/rs2.1_v2_untrained_20260726T072252Z/` (216 heads × 100 baseline trials, <60s)
- V3 output: `outputs/rs2.1_v3_residual_20260726T072814Z/` (216 heads × 100 baseline trials, ~4min)
- Published: `run/rs2.1/20260726` at https://github.com/vhallac/crockpot-experiments/releases/tag/run/rs2.1/20260726
- SHA256: `37a600294b9f8305014775cb9dbfa72be546ab423d06d74b36c2fe4dc1ea1ffd`

### Results

#### V1: RS2 primary (trained DroPE'd vs RoPE k_post, tabulated)

| Layer group | n heads | excess_mean | >0 fraction |
|---|---|---|---|
| L2 only | 8 | 0.437 | 100% |
| L3–L7 | 40 | 0.269 ± 0.025 | 100% |
| L8–L12 | 40 | 0.180 ± 0.021 | 92.5% |
| L13–L17 | 40 | 0.068 ± 0.016 | 55% |
| L18–L27 | 80 | −0.074 ± 0.009 | 20% |
| **All L1–L27** | 216 | 0.087 ± 0.013 | 60.6% |

#### V2: inheritance control (untrained-dropped vs RoPE k_post)

| Layer group | n heads | excess_mean | >0 fraction |
|---|---|---|---|
| L2 only | 8 | 0.350 | 100% |
| L3–L7 | 40 | 0.071 ± 0.018 | 78% |
| L8–L12 | 40 | −0.023 ± 0.013 | 33% |
| L13–L17 | 40 | 0.020 ± 0.016 | 50% |
| L18–L27 | 80 | −0.079 ± 0.008 | 15% |
| **All L1–L27** | 216 | **−0.003** ± 0.009 | 40.7% |

#### V3: residual after orthogonalizing against k_pre

| Layer group | n heads | residual_excess | >0 fraction | res_rank (RoPE) | res_rank (DroPE'd) |
|---|---|---|---|---|---|
| L2 only | 8 | 0.290 ± 0.037 | 100% | 23.6 | 1.8 |
| L3–L7 | 40 | 0.041 ± 0.018 | 65% | 27.7 | 2.1 |
| L8–L12 | 40 | 0.069 ± 0.016 | 75% | 28.4 | 4.5 |
| L3–L12 | 80 | **0.055 ± 0.012** | 70% | 28.0 | 3.3 |
| L13–L17 | 40 | 0.019 ± 0.017 | 50% | 22.1 | 5.4 |
| L18–L27 | 80 | −0.112 ± 0.010 | 14% | 27.1 | 11.8 |
| All L1–L27 | 216 | −0.006 ± 0.009 | 46% | 26.5 | 6.8 |

(Rank columns re-tabulated 2026-07-26 after external review: five cells were off — L3–L7 RoPE
27.4→27.7, L8–L12 28.6→28.4 / 4.6→4.5, L13–L17 RoPE 23.0→22.1, L18–L27 27.1 / DroPE'd 10.0→11.8,
ALL 26.9→26.5 / 5.6→6.8. The `residual_excess` and `>0 fraction` columns were verified exact and
are unchanged — this was tabulation slippage confined to the rank columns.)

**Provenance note (V3 artifact):** the residual run's `rs2_subspace_summary.json` records neither
the `--residual-against-pre` flag nor any residual-column summary — it is numerically identical
to a primary-run summary, because the script's summary block only aggregates the primary metrics.
The per-head CSV's `residual_*` columns are the evidence the flag took effect. A future
`c2_subspace_overlap.py` revision should echo the flag and summarize residual stats in the JSON
so the artifact is self-describing.

#### V4: position_fraction cross-reference (L3–L12)

- **RoPE k_post mean position_fraction: 0.393** (corrected 2026-07-26 — originally reported as
  0.244, which was `pre`+`post` variant rows pooled together for RoPE rather than `post` alone.
  Confirmed by direct recomputation: pooling `pre` (0.092) and `post` (0.393) gives 0.2426,
  matching the original figure almost exactly. DroPE'd's own number was unaffected by the same
  bug because its `pre` and `post` rows are numerically identical there — per G-RS1.1's
  `k_pre == k_post` invariant under the identity-rotation patch — so pooling changes nothing on
  that side, which is why only the RoPE figure was wrong.)
- DroPE'd mean position_fraction: 0.294
- Both well above chance → the measured alignment reflects **positional** geometry, not content
  classification. The correction *strengthens* this conclusion (0.393 is even further from
  chance than 0.244 was), it does not weaken it.

#### V5: statistical robustness

- **Layer-clustered Δ (RS2 − V2, L3–L12):** Δ = 0.201, **t = 9.30** (df = 9; twice-corrected —
  originally reported as t = 13.3, which is the *unclustered* per-head statistic (80 heads
  treated as independent draws; that computation reproduces 13.3 exactly). The first correction
  clustered by layer but used the population SD, giving 9.80; a df = 9 t-test implies the sample
  SD (ddof = 1), which gives **9.30** (= 9.80 × √(9/10)) — external-review catch, 2026-07-26.
  97.5% of the 80 individual heads are still positive. Null hypothesis (Δ ≤ 0) is rejected at
  p ≪ 0.001 under every variant; only the statistic's value and labels needed correcting.)
- **V3 residual t-test (L3–L12 vs its own random baseline):** t = 4.56, 70% of heads > 0.
- **Rank sensitivity (V5b):** Pearson r(rank_diff, excess) = 0.19 in L3–L12 — weak
  correlation. Excess is not an artifact of PCA-rank differences.
- **L1 (V5c, restated after external review):** the original sentence claimed "including L1
  drops global mean excess to 0.074 (from 0.224 in L3–L12)" — the 0.074 figure does not
  reconcile with any recomputation (L1-only excess = 0.066; all-layers = 0.087; L1–12 = 0.229),
  and the comparison mixed mismatched scopes. Corrected statement: **L1's own excess is +0.066**
  (weak, between the L2 peak and the transition zone), consistent with the depth-structured
  interpretation; it was previously omitted from RS2's phase table and is now accounted for.

### Analysis

**Adjudication per RS2.1-spec.md §5 decision matrix:**

1. **P.RS2.1.a (reconstruction survives inheritance control) — STRONGLY SUPPORTED.**
   Δ = 0.201, properly layer-clustered t = 9.30 (df=9; see V5 correction), 97.5% of L3–L12
   heads show trained excess > untrained excess. This cannot be an initialization artifact.
   The DroPE'd model's overlap with RoPE's k_post subspace reflects training-time
   reconstruction, not mere weight inheritance.

   **Depth-resolved reference arms (completing V1's specified tabulation; added after external
   review):**

   | Layer group | Primary (trained vs k_post) | Ref2 (k_pre vs k_post) | Ref1 alignment (trained vs k_pre) |
   |---|---|---|---|
   | L2 | +0.437 | +0.485 | — |
   | L3–L7 | +0.269 | +0.200 | — |
   | L8–L12 | +0.180 | +0.067 | — |
   | **L3–L12** | **+0.224** | **+0.134** | 0.778 |
   | Global | +0.087 | +0.111 | 0.697 |

   In the reconstruction layers (L3–L12), primary excess **exceeds** Ref2 — the DroPE'd
   emergent subspace is *more* k_post-aligned there than RoPE's own k_pre is. This closes the
   last version of the transitive-channel objection: a k_pre-mediated inheritance channel
   cannot even in principle explain overlap that is larger than that channel's own strength.
   Combined with Ref1 (trained vs k_pre alignment 0.778 in L3–L12), the geometry is coherent:
   training took the inherited code and moved it partway toward the rotated (k_post) geometry —
   still nearer to k_pre in absolute terms, but more k_post-aligned than k_pre itself is.

2. **P.RS2.1.b (L2 is not representative) — CONFIRMED.**
   L2's untrained excess (0.350) accounts for ~80% of its trained excess (0.437), making
   L2 an inheritance-dominated outlier. RS2's notebook illustration using L2 was misleading
   about the mechanism's generality.

3. **P.RS2.1.c (depth-structured profile) — CONFIRMED.**
   Excess is strong in L3–L12 (0.224), declines through L13–L17 (0.068), and goes negative
   in L18–L27 (−0.074). This matches the RS1-predicted early/middle-layer concentration.

4. **V3 residual interpretation — QUALIFIED SUPPORT for rotation-specific reconstruction.**
   The residual excess (after partialling out k_pre) is positive and significant (0.055,
   t = 4.56), but is only ~24% of the primary excess magnitude (0.224). Most of the
   reconstructed subspace is in the shared k_pre/k_post span, not in uniquely k_post
   directions. The DroPE'd model's residual rank (3.3) is far lower than RoPE's (28.0),
   consistent with a model that learned to encode position without reproducing the full
   k_post-specific rotational machinery.

5. **C2 re-adjudication: SUBSTANTIATED.**
   The inheritance control strengthens rather than weakens the C2 verdict. The trained
   model's overlap (0.224) far exceeds the untrained baseline (0.024), eliminating
   initialization inheritance as the explanation. The DroPE'd model genuinely reconstructs
   key directions aligned with what RoPE supplied.

### Conclusion / Next Step

RS2.1 closes the inheritance loophole in RS2's C2 adjudication. The DroPE'd model
reconstructs positional key directions that overlap with RoPE's k_post subspace, and this
reconstruction is a training effect, not a weight-inheritance artifact.

*(Corrected after external review — the original sentence here read "the internal reference
arm (Ref2: RoPE k_pre vs k_post, excess ~0.42) remains larger than the trained overlap." That
was wrong twice: ~0.42 is not the Ref2 arm value — global Ref2 excess is +0.111, and 0.42
matches the CSV's first row (L1H0, 0.4205), an eyeballed-first-row slip; and the direction is
backwards where it matters — in L3–L12, primary excess (+0.224) exceeds Ref2 (+0.134). See the
depth-resolved reference-arm table in Analysis §1. The error understated the result.)*

**RS2 is substantiated — depth-qualified.** Reconstruction is an L3–L12 phenomenon: late
layers (18–27) sit below the random baseline and the residual test pushes them further
negative (−0.112), so the late-layer emergent code remains a genuinely different code. The
rotation-specific residual support is real but thin (+0.055, ~24% of the primary magnitude;
DroPE'd residual rank ~3 vs RoPE's ~28) — "qualified support," per Analysis §4, is the right
weight, and the top-line verdict should not be read as rounding it up.

**Remaining caveat (not addressed by RS2.1's controls):** V2 rules out initialization
inheritance, but cannot distinguish "reconstructs RoPE's *specific* code" from "any functional
NoPE training at this scale converges to similar positional geometry." Discriminating those
needs an independently-initialized control (a from-scratch or differently-seeded NoPE model
probed the same way) — an unscoped C2 follow-up (not RS3; RS3 is now pinned to C3, see
RS3-spec.md).

Next: unscoped ideas, neither matching RS3 (C3, now spec'd) or RS4 (C4, scale check) as
currently defined — a `k_post` ablation/retraining test of necessity (folding in the
independent-initialization control above), and an emergence-timeline study across training
checkpoints.

## 2026-07-26 — RS2 C2 subspace overlap (completed)

### Question / Hypothesis

RS1b's LR-corrected rerun showed that the DroPE'd model is mechanistically close to the RoPE
baseline: emergent key-position persists (P.RS1.b holds), and the addressing profile is
unchanged within measurement noise — raw head-count comparison (34/39, 3/2, 6/4), no
significance test (P.RS1.c re-adjudicated in the program's favor). RS2
now tests the secondary C2 claim: does the DroPE'd model's emergent positional subspace
**reconstruct the same subspace RoPE supplied**, or is it a different positional code?

P.RS1.d predicts substantial overlap between the RoPE `k_post` positional subspace and the
DroPE'd emergent key-position subspace, measured via principal angles / CCA and compared
against a random-rotation baseline (per RS1-spec §10.F).

**Falsifier:** disjoint subspaces (alignment at or below random baseline) → emergent position
is a *different* code, not a reconstruction of what RoPE supplied.

### Experiment Design Summary

- **Inputs:** M1.5 projector bases from two pre-existing runs:
  - RoPE `k_post`: `outputs/rope_as_scaffold_rs1a_20260724T0559Z/m15_qwen3/kaddress_m15_projectors_qwen3.npz`
  - DroPE'd emergent: `outputs/rs1b_probes_lr1e3_qwen3_droped_20260726/outputs/rs1b_probes_lr1e3_20260726T042955Z_m15_qwen3_droped/kaddress_m15_projectors_qwen3-droped.npz`
- **Method:** For each (layer, head) present in both, extract Family A aggregate PCA basis,
  compute principal angles via SVD of A^T B, and compute a random-rotation baseline
  (100 trials) per §10.F. Primary metric: `alignment_excess` = mean_cos(observed) −
  mean_cos(random baseline).
- **Reference comparisons:** RoPE `k_pre` vs DroPE'd emergent (emergent-to-emergent) and
  RoPE `k_pre` vs RoPE `k_post` (internal RoPE rotation shift).
- **Analysis-only — no GPU needed.** The projector NPZ files are small (59–203 MB) and the
  computation is O(heads × d_head³) ≈ seconds on CPU.

### Planned Procedure

```bash
cd /home/vedat/work/personal/crockpot-experiments
./scripts/nix-cpu-run experiments/rope-as-scaffold/scripts/c2_subspace_overlap.py \
  --rope-projectors outputs/rope_as_scaffold_rs1a_20260724T0559Z/m15_qwen3/kaddress_m15_projectors_qwen3.npz \
  --droped-projectors outputs/rs1b_probes_lr1e3_qwen3_droped_20260726/outputs/rs1b_probes_lr1e3_20260726T042955Z_m15_qwen3_droped/kaddress_m15_projectors_qwen3-droped.npz \
  --output-dir outputs/rs2_subspace_$(date -u +%Y%m%dT%H%M%SZ) \
  --baseline-trials 100 \
  --families A \
  --seed 0
```

### Expected Signal / Interpretation Plan

- **P.RS1.d holds** (strong overlap): excess substantially > 0 across most heads, with a
  plausible depth profile (early layers heavily RoPE-dominated → high overlap; mid layers
  emergent recovers RoPE's code → high overlap; late layers may diverge). This upgrades C1
  from "emergent position fills in" to "emergent position reconstructs the same code RoPE
  supplied" — a stronger mechanistic claim.
- **P.RS1.d falsified** (random-level alignment): excess ≈ 0 globally → emergent position
  and RoPE-supplied position are different codes. This doesn't invalidate C1 (position is
  still present and non-addressable) but weakens the "reconstruction" framing — the model
  developed its own positional system rather than inheriting RoPE's.
- **Intermediate** (excess > 0 in some layers, ≈ 0 or negative in others): depth-stratified
  reconstruction — RoPE's code is recoverable where RoPE is architecturally load-bearing
  (early layers), with divergence in late layers where the model has freedom to develop
  independent representations. This is the most likely outcome given RS1a's finding that
  early-layer position is rotation-propagated while deep-layer position is emergent and
  rotation-independent.

### Pre-run Provenance

- Spec: `experiments/rope-as-scaffold/RS2-spec.md` (pre-registration); see also `RS1-spec.md` §10.F (C2 method).
- Code: `experiments/rope-as-scaffold/scripts/c2_subspace_overlap.py` (new, this commit).
- Input data provenance:
  - RoPE projectors: RS1a run, published at
    <https://github.com/vhallac/crockpot-experiments/releases/tag/run/rope-as-scaffold-rs1a/20260724>
  - DroPE'd projectors: RS1b LR-corrected run, published at
    <https://github.com/vhallac/crockpot-experiments/releases/tag/run/rope-as-scaffold-rs1b-lr1e3/20260726>
- Code branch: `main`.
- Pre-run commit: e88421d (RS2: pre-register subspace-overlap experiment, add analysis script).
- Planned output location: `outputs/rs2_subspace_<timestamp>/`; to be committed as a summary
  artifact or published alongside prior RS1 outputs.

### Run Evidence

- **Run date:** 2026-07-26T06:07:59Z
- **Environment:** RunPod RTX 4090 (US-CA-2), `/workspace/venv` Python, NPZ files uploaded
  from local copies since they were not on the network volume.
- **Run command:**
  ```bash
  /workspace/venv/bin/python experiments/rope-as-scaffold/scripts/c2_subspace_overlap.py \
    --rope-projectors outputs/rope_as_scaffold_rs1a_20260724T0559Z/m15_qwen3/kaddress_m15_projectors_qwen3.npz \
    --droped-projectors outputs/rs1b_probes_lr1e3_qwen3_droped_20260726/outputs/rs1b_probes_lr1e3_20260726T042955Z_m15_qwen3_droped/kaddress_m15_projectors_qwen3-droped.npz \
    --output-dir outputs/rs2_subspace_20260726T060759Z \
    --baseline-trials 100 --families A --seed 0
  ```
- **Runtime:** ~96 seconds (0.7s loading, ~95s computation for 216 heads × 100 baseline trials).
- **Exit code:** 0

### Results

- **Heads analyzed:** 216 (rope_post ∩ droped_pre, layers 1–27).
- **Global alignment:** 0.527 ± 0.198 (mean cos of principal angles).
- **Random baseline:** 0.440 ± 0.053 (range 0.180–0.555, non-degenerate).
- **Alignment excess:** +0.087 ± 0.193 (60.6% of heads above baseline).

**Gate adjudication:**

| Gate | Verdict | Evidence |
|------|---------|----------|
| G-RS2.1 (input integrity) | PASS | 216 common heads, both NPZ files load cleanly |
| G-RS2.2 (baseline non-trivial) | PASS | Baseline range 0.180–0.555, neither degenerate nor saturation |
| G-RS2.3 (internal rotation ceiling) | PASS | RoPE pre→post excess +0.111, 66.2% above zero — rotation measurably changes subspace, and the instrument detects it |

**Prediction adjudication:**

| Prediction | Verdict | Evidence |
|-----------|---------|----------|
| P.RS2.a (subspace reconstruction) | HOLDS (depth-qualified) | Global excess +0.087, but this is an average of three distinct depth regimes (see P.RS2.c) |
| P.RS2.b (emergent drift ≥ primary) | HOLDS | Ref alignment 0.697 ≥ primary 0.527 in 86.1% of heads |
| P.RS2.c (depth-structured profile) | HOLDS | Three clear phases (see below) |

**Depth profile (P.RS2.c — confirmed):**

| Phase | Layers | Excess | Alignment | >0% | n |
|-------|--------|--------|-----------|-----|---|
| Early-mid | 2–12 | +0.244 ± 0.151 | 0.513–0.863 | 96.6% | 88 |
| Transition | 13–17 | +0.068 ± 0.135 | 0.384–0.609 | 65.0% | 40 |
| Late | 18–27 | −0.074 ± 0.106 | 0.292–0.435 | 17.5% | 80 |

(Alignment column = range of per-layer mean alignment within the phase. Layers sum to
208 of 216 heads — **layer 1 (8 heads) is excluded from all three phases**, not folded into
"Early-mid." Its own excess is +0.066 (per RS2.1 V5c), well below L2–12's +0.244, so this
is not a neutral omission: including it would pull the early-mid mean down to +0.229. No
rationale for excluding L1 from the phase grouping was recorded at analysis time; it is
called out here rather than left silently absent.)

Layer 2 shows the strongest reconstruction (excess +0.437, alignment 0.863) — the
DroPE'd model's emergent key-position almost perfectly recovers RoPE's code in the
earliest layers. Excess falls through mid-depth, crosses zero around layer 13–17, and
trends negative in late layers (18+): 8 of the 10 late layers have mean excess < 0
(L18–23, L26, L27 all in −0.03…−0.14), with L24 (+0.039) and L25 (≈0.000) sitting at
the noise floor at exactly 50% of heads above baseline — no late layer *exceeds* 50%.
As an aggregate the late phase is robustly below the random baseline (−0.074 over 80
heads, ~6 standard errors below zero). This triphasic profile precisely mirrors RS1a's
finding that early-layer position is rotation-propagated (RoPE-dependent) while
late-layer position is emergent and rotation-independent.

### Analysis

The excess of +0.244 in early-mid layers shows that the DroPE'd model genuinely
reconstructs the positional code RoPE supplied — not at 100% fidelity (the maximum
excess is ~0.44 in layer 2, not 1.0), but substantially above the random-baseline
floor. This is the "RoPE is a scaffold" story: where RoPE is architecturally
load-bearing (early layers, where position must be rotation-propagated), the model
internalizes a recognizably similar code. Where RoPE is optional (late layers,
where position is emergent and rotation-independent), the model develops a different
code — alignment falls below random in many late-layer heads.

P.RS2.b (emergent drift ≥ primary) holding in 86.1% of heads confirms that the
before/after comparison without the rotation step is substantially easier — the
DroPE'd model's emergent code aligns more closely with un-rotated RoPE keys
(alignment 0.697) than with post-rotation RoPE keys (0.527). This is geometrically
expected: the rotation step is the "hard part" that the DroPE'd model must learn
to emulate.

### Conclusion / Next Step

C2 is **substantiated**: the DroPE'd emergent positional subspace reconstructs the
same code RoPE supplied, with a depth profile that maps exactly onto RS1a's
rotation-dependence gradient.

- **C1 (position fills in):** Position-persistence confirmed by RS1b — position survives
  RoPE removal and retraining. The fill-in dynamic itself was never observed: position
  was already at ceiling pre-training (RS1a), so RS1b shows persistence, not emergence.
- **C2 (same code reconstructed):** Confirmed by RS2, depth-qualified.
- **Next:** The primary C1/C2 claims are now both supported. RS4 (model-scale generalisation,
  different architectures — C4) and an unscoped causal-intervention idea (freezing early-layer
  heads that reconstruct RoPE's code vs late-layer heads that diverge) remain as future work.
  RS3 (C3, local-order vs retrieval) is pre-registered separately — see RS3-spec.md.

### Published Outputs

- Output directory: `outputs/rs2_subspace_20260726T060759Z/`
  - `rs2_subspace_overlap.csv` (58,968 bytes, SHA256: 4f811879…)
  - `rs2_subspace_summary.json` (10,347 bytes, SHA256: 9353c461…)
- Release: <https://github.com/vhallac/crockpot-experiments/releases/tag/run/rope-as-scaffold-rs2/20260726>
- Final commit: `5a23568` (notebook completion + two verified-slip fixes).

---

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

Training completed on RunPod NVIDIA H100 SXM (80GB) from pre-run commit `50a4e5a` (plus
checkpoint/RNG fixes `6daf67d`, `28e71ac`, `8f78c47` — SIGHUP handling and RNG state
serialization; no recipe changes). Full recipe: LR=1e-3, 1B tokens, train context 2048,
cosine → 10% of peak, 2% warmup, AdamW β=(0.9,0.95) wd=0.1, seed 0. Output checkpoint
directory: `/workspace/qwen3-droped`. Token cache reused from v1 training at
`/workspace/rs1b-token-cache/`; eval slice is the same 5M-token held-out prefix used
for the v1 PPL measurement.

**Training/eval curve (P.RS1.a evidence — full `training_metrics.csv` eval history, not just
"not near-random"):**

| step | eval_ppl | eval_ce |
|---:|---:|---:|
| 1 | 30,859.3 | 10.337 |
| 100 | 72.06 | 4.278 |
| 200 | 37.34 | 3.620 |
| 300 | 30.69 | 3.424 |
| 400 | 27.56 | 3.316 |
| 500 | 25.64 | 3.244 |
| 600 | 24.08 | 3.181 |
| 700 | 22.84 | 3.128 |
| **800** | **21.64** | **3.075** |
| 900 | 20.86 | 3.038 |
| 1000 | 20.10 | 3.001 |
| 1100 | 19.45 | 2.968 |
| 1200 | 18.81 | 2.934 |
| 1300 | 18.30 | 2.907 |
| 1400 | 17.84 | 2.882 |
| 1500 | 17.50 | 2.862 |
| 1600 | 17.27 | 2.849 |
| 1700 | 17.12 | 2.840 |
| 1800 | 17.00 | 2.833 |
| 1900 | 16.89 | 2.827 |
| **1907 (final)** | **16.88** | **2.826** |

RoPE baseline for reference: PPL 21.8 / CE 3.082 (RS1a). This run crosses below the baseline
around step ~800 and continues improving smoothly through completion (total elapsed 25,173s ≈
7.0h on H100 SXM, steady-state ~39,700-39,800 tok/s) — a monotonic, fully-converged curve, not
an early-stopping or lucky-checkpoint artifact. This is the P.RS1.a evidence that was previously
missing from this entry (a reviewer catch, addressed 2026-07-26): the "LR confound removed" claim
below is now backed by the actual recovery curve, not just the M1.6 mechanistic result.

**M1.5 and M1.6 probes** run on RunPod NVIDIA GeForce RTX 4090 (24GB VRAM), commit `8f78c47`:

```bash
# M1.5
PYTHONPATH=experiments/dead-keys:experiments/k-address-space ./scripts/cuda-python -m kaddress.scripts.position_content \
  --model qwen3-droped --device cuda --max-length 1024 \
  --output-dir outputs/rs1b_probes_lr1e3_20260726T042955Z_m15_qwen3_droped

# M1.6 (--max-marker-sets 4096 needed for G6 on this better-trained model)
PYTHONPATH=experiments/dead-keys:experiments/k-address-space ./scripts/cuda-python -m kaddress.scripts.m16_discriminator \
  --model qwen3-droped --device cuda --max-marker-sets 4096 \
  --repetitions 128 \
  --output-dir outputs/rs1b_probes_lr1e3_20260726T045616Z_m16_qwen3_droped
```

**M1.5 (position-content probe):**

Gates: G1 `PASS` (architectural zero at layer 0; identity RoPE means `pre==post`),
G2 `NOT_APPLICABLE`. `summary_rows=43904`, `shuffle_null_ok=True`, 19 stimuli across
families A/B/C, segment lengths [4,7].

Pre==post confirmed exactly (0.0 delta, all 21,280 per-stimulus/per-slot rows) — consistent
with G-RS1.1's `k_pre == k_post` invariant. The 1,344 derived `AGGREGATE` summary rows (~3% of
`summary_rows`) show real but small pre/post differences, up to 0.21 in `ridge_r2`, concentrated
in the low-R² early layers (1–3); this is almost certainly unseeded CV-fold randomness in the
aggregate-level regression, not a true pre/post divergence in the underlying keys, but it means
"pre==post" should be read as a per-slot-measurement fact, not a blanket property of every row in
this file.

Family A slot-level depth profile (pre variant):

| layer | position_fraction | ridge_r2 |
|---:|---:|---:|
| 0 | 0.0000 | 0.0000 |
| 1 | 0.0019 | 0.0462 |
| 2 | 0.0075 | 0.3432 |
| 6 | 0.4006 | 0.9960 |
| 12 | 0.3288 | 0.9898 |
| 18 | 0.5676 | 0.9861 |
| 23 | 0.4924 | 0.9856 |
| 27 | 0.1962 | 0.9937 |

Aggregate (all-family) position fraction means: Family A 0.351, Family B 0.358,
Family C 0.469. Ridge R² means: 0.886, 0.910, 0.932.

**M1.6 (hypothesis discriminator):**

Gates: G6 `PASS` (all four stimuli after expanded marker search up to 1133 sets).
G7 pass count: `34/448` (cf. qwen3 RoPE: 39; RS1b v1: 13).
`transitivity_confirmed_count=448` (cf. qwen3 RoPE: 448; RS1b v1: 0).

G6 per-stimulus marker search:

| stimulus | max/min ratio | searched sets | selected markers |
|---|---:|---:|---|
| M16_00 | 2.919 | 1133 | `always,constant,high,loose` |
| M16_01 | 1.329 | 169 | `here,blank,true,cold` |
| M16_02 | 1.589 | 250 | `locally,outside,far,dull` |
| M16_03 | 2.856 | 360 | `clearly,once,certainly,neutral` |

Per-head classification counts (448 total):

| classification | heads |
|---|---:|
| mixed | 120 |
| inert | 83 |
| confounded_noise_sensitive | 80 |
| transitive_induction | 72 |
| anti_collision_or_content_driven | 59 |
| anti_collision_or_inert_attention_only | 31 |
| **addressing** | **3** |

Addressing heads (all late-layer, all `output_above_noise=True`):

| layer | head | patch_both donor-prob delta | noise donor-prob delta |
|---:|---:|---:|---:|
| 21 | 8 | +2.01e-4 | −2.39e-5 |
| 24 | 14 | +1.47e-4 | −1.46e-7 |
| 25 | 14 | +1.89e-4 | −5.88e-6 |

Total `output_above_noise`: 6/448 (cf. qwen3 RoPE: 4; RS1b v1: 0).

Published artifacts:

- Release: <https://github.com/vhallac/crockpot-experiments/releases/tag/run/rope-as-scaffold-rs1b-lr1e3/20260726>
- Bundle: `rs1b_probes_lr1e3_qwen3_droped_20260726.tar.gz`
- SHA256: `2a032ea3b0fc794912c725905172b914190e88c5f6ad1b6a03d7c916747ad9cd`
- Checksum asset: `SHA256SUMS_rs1b_probes_lr1e3_qwen3_droped_20260726`

### Analysis

The headline is that **the v1 transitivity/addressing collapse was an under-training
artifact — the LR-corrected model recovers transitivity fully and shows slightly more
addressing evidence than the original RoPE baseline.**

**Cross-run comparison — the numbers that adjudicate P.RS1.b/c:**

| Metric | qwen3 (RoPE) | v1 (LR=3e-5) | LR=1e-3 (this run) |
|---|---|---|---|
| M1.6 G7 pass | 39/448 | 13/448 | **34/448** |
| M1.6 transitivity | 448/448 | 0/448 | **448/448** |
| M1.6 addressing | 2/448 | 0/448 | **3/448** |
| M1.6 output_above_noise | 4/448 | 0/448 | **6/448** |

**P.RS1.b (emergent key-position — M1.5).** The LR=1e-3 DroPE'd model's key-position
profile remains substantively the same as the v1 profile and the RoPE baseline's `k_pre`:
architectural zero at L0, ridge R² rises rapidly to near-ceiling (~0.99) by L6, position
fraction peaks mid-stack (~0.57 at L18) and falls in late layers. This is the "holds by
literal criterion but uninformative beyond RS1a" pattern, repeated: RS1a found the same
near-ceiling emergent position in the *untrained* dropped state, so recalibration at
proper LR didn't "fill in" a gap — it confirmed the gap was never there. P.RS1.b **holds**
(the falsifier "absent yet perplexity recovers" did not trigger), but carries no new
increment of evidence for the fill-in dynamic.

**P.RS1.c (addressing unchanged — M1.6).** Per the pre-registration's expected-signal
language, this outcome matches the **second branch**: "PPL recovers and
transitivity/output_above_noise recover substantially too — the v1 collapse was primarily
an under-training artifact; P.RS1.c would need re-adjudication in the program's favor."

In detail:

1. **Transitivity recovers fully.** The 0→448 transitivity collapse in v1 was entirely
   a training artifact — properly trained, the DroPE'd model's transitivity profile is
   indistinguishable from RoPE (448/448 in both). Whether transitivity is a meaningful
   head-level discriminator or a model-level readout artifact (as the qwen3 M1.6
   notebook entry noted) is unchanged; but the *absence* of transitivity in v1 was
   spurious.

2. **Addressing does not disappear — it slightly exceeds the RoPE baseline.** The
   RoPE model had 2 `addressing`-classified heads; the DroPE'd model has 3 (L21H8, L24H14,
   L25H14). Output-above-noise moves from 4→6. These shifts are small in absolute terms
   (~1.5–2e-4 in donor-marker probability over ~5e-4 baseline), and the `addressing`-labeled
   sets are mostly not the same heads (RoPE: L24H15, L25H14; this run: L21H8, L24H14, L25H14 —
   only L25H14 overlaps at the label level). At the broader, actually-pre-registered
   `output_above_noise` criterion (independent of which sub-classification a head lands in),
   the overlap is stronger than the label-level comparison suggests: L21H8 is
   `output_above_noise=True` in **both** runs — it's just classified `confounded_noise_sensitive`
   in RoPE and `addressing` here — so 2 of RoPE's 4 `output_above_noise` heads (L21H8, L25H14)
   persist as `output_above_noise` in the DroPE'd model too. Per the spec's own falsifier
   language, this does **not** read as "addressing disappears across the transition" — it reads
   as "addressing is present
   at comparable weak levels before and after."

3. **G7 attention steerability is close to baseline.** 34/448 vs 39/448 — RoPE-level
   steerability, confirming the attention pathway is functional after proper training.

**Implications for the program:**

- C1 ("emergent key-position fills in") is **not falsified** — position was already
  present untrained (RS1a), and training didn't change it. The fill-in dynamic itself
  remains unobserved; what we have is persistence.
- P.RS1.c ("the addressing profile stays unchanged") is **not the cleanest reading** —
  addressing increased slightly (2→3 heads, 4→6 output-above-noise), and the
  addressing heads are mostly different from the RoPE set. But the absolute effect
  is so small and fragile in both states (single-stimulus addressing passes on
  different stimuli in different heads) that "unchanged at the noise floor" is the
  more defensible interpretation than "addressing increased."
- The **primary finding** is negative but important: **proper LR training (1e-3)
  restores all v1 mechanistic metrics to near-baseline levels**, confirming that the
  v1 LR was the confound, not RoPE removal. The DroPE'd model is mechanistically
  similar to the original RoPE model — it develops emergent position, has weak/no
  query-readable addressing, and shows the same transitivity readout.
- This result **strengthens C1 overall**: if the DroPE'd model is mechanistically
  RoPE-like without RoPE, the scaffold interpretation (RoPE supplies what the model
  can generate anyway) gains support, not weakness.

**Caveats:**

- The absolute addressing signal remains weak in both RoPE and DroPE'd states (2–3
  heads out of 448, single-stimulus passes). The instrument is measuring at the noise
  floor — small count differences (2 vs 3) should not be over-interpreted.
- The six `output_above_noise` heads (3 addressing + 3 from other buckets) represent
  ~1.3% of heads — not a robust signal.
- Transitivity=448 in both runs is a model-level readout that does not vary across
  heads (see qwen3 M1.6 notebook analysis); it is not independent evidence per head.
- G6 required `--max-marker-sets 4096` because the better-trained model has more
  peaked next-token distributions (harder to find neutral markers). This is a
  calibration issue, not a validity issue — all four stimuli passed G6.

### Conclusion / Next Step

The RS1b LR-corrected rerun **removes the v1 under-training confound** and shows that
the DroPE'd model at proper LR is mechanistically close to the RoPE baseline: emergent
position persists, transitivity recovers to baseline, and weak/no query-readable
addressing is unchanged within measurement noise. The v1 collapse (transitivity 0,
addressing 0) was an LR artifact.

**P.RS1.b holds** (emergent position present, unchanged from untrained).
**P.RS1.c is re-adjudicated in the program's favor** — addressing does not disappear
across the RoPE→NoPE transition; both states show the same weak, fragile signal.

Next step (2026-07-26): **Path (b) chosen** — RS1b-ctrl is deferred (per §11 gating: results are
crisp, not borderline). Proceeding to **RS2 (C2 subspace overlap)** on the existing M1.5
projectors — analysis-only, cheap, no GPU needed.

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
