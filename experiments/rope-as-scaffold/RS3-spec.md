# RS3 — Functional locus of RoPE (local-order acuity vs. retrieval)

**Dated:** 2026-07-27
**Status:** pre-registered, not yet run
**Program:** [`rope-as-scaffold`](README.md) — tests claim **C3** (behavioral).
**Depends on:** RS1a + RS1b (LR-corrected) complete — consumes the `qwen3-droped` checkpoint and
the frozen FineWeb-Edu eval definition. Not gated on RS2/RS2.1 (those adjudicate C2, a different
claim), though RS2.1's depth profile motivates the interpretation in §6.

> **Scope note.** Two `NOTEBOOK.md` next-step lines describe "RS3" differently — the RS2 entry
> calls it model-scale generalisation (that is **RS4/C4**), and the RS2.1 entry proposes a
> `k_post` ablation plus an independently-initialized NoPE control (a **C2** follow-up, not C3).
> This spec follows the README's planned-experiments table, which is the pre-registration of
> record: **RS3 tests C3.** The other two ideas remain unscoped and are not folded in here.

---

## 0. Question

C1/C2 established *what survives* RoPE removal: emergent key-position is present, non-addressable,
and (for L3–L12) genuinely reconstructs the code RoPE supplied. C3 asks the complementary
behavioral question — **what actually gets worse?**

**Primary (C3):** removing RoPE costs **local-order / recency acuity**, not
**retrieval / content-addressing**. I.e. RoPE's causal contribution is local, not an address.

**Falsifier (from README's claim set, verbatim intent):** removal degrades retrieval *more* than
local-order tasks → RoPE's contribution is (partly) a retrieval mechanism, and the program's
central "scaffold, not mechanism" framing fails at the behavioral level even though C1/C2 hold at
the representational level.

This is the claim that turns [RNoPE-SWA](references/literature-survey.md)'s correlational
"RoPE = local, NoPE = retrieval" observation into a causal statement, because we hold the model
fixed and remove RoPE, rather than comparing two independently-trained architectures.

---

## 1. Why this instrument, and the one thing that decides the design

### The confound that shapes everything

`qwen3-droped` received **1B extra FineWeb-Edu tokens**; `qwen3` received none. This is not a
minor asymmetry — the DroPE'd model is a **better** FineWeb-Edu model than the RoPE baseline
(PPL 16.88 vs 21.80, `NOTEBOOK.md` RS1b LR-corrected entry). Any **cross-model absolute**
behavioral comparison therefore measures domain adaptation at least as much as it measures RoPE
removal, and would be worthless as C3 evidence.

**Consequence — pinned design rule:** every primary RS3 metric is a **within-model contrast**
(a perturbed-vs-clean difference, or a repeated-vs-first-occurrence difference) computed on
*identical text* for both models, then compared across models. A within-model difference cancels
the level shift; it does not cancel a change in *sensitivity*, which is why §6 keeps the residual
confound live and §5 gates on it.

**Bias direction (report this alongside the result).** For the local axis the residual confound
runs **against** C3: a model better adapted to the eval domain has sharper, more confident clean
predictions and therefore *more to lose* from perturbation, inflating the DroPE'd model's
perturbation sensitivity. C3 predicts the DroPE'd model is *less* local-order sensitive. So a
confirming result on the local axis is obtained **despite** the confound, which strengthens it.
No such conservative argument is available for the retrieval axis — state that plainly.

**No PPL-matched checkpoint is available.** The training curve passes RoPE-baseline PPL at
step ~800, but `train_qwen3_nope.py` writes a single rolling `ckpt.pt`, so no step-800 checkpoint
survives. PPL-matching would require a retrain; it is not in scope.

### Why these instruments

- **Local-scramble CE** measures exactly what C3 names — sensitivity to token order *within a
  short window* — with no task design, no prompting, and no floor risk on a 0.6B base model.
- **Induction (repeated-span) CE gain** is content-addressed retrieval, is robustly present in
  small base models, and is naturally a within-model paired quantity.
- Both run on frozen, already-defined data. Only GPU inference; no training.

---

## 2. Design

### 2.1 Inputs (all frozen, no new training)

| tag | source | note |
|---|---|---|
| `qwen3` (state 1, RoPE) | `Qwen/Qwen3-0.6B` via `MODEL_IDS` | pinned revision per RS1-spec §10 |
| `qwen3-droped` (state 3, DroPE'd) | `QWEN3_DROPED_PATH=/workspace/qwen3-droped` | LR=1e-3 rerun, published at [run/rope-as-scaffold-rs1b-lr1e3/20260726](https://github.com/vhallac/crockpot-experiments/releases/tag/run/rope-as-scaffold-rs1b-lr1e3/20260726) |
| `qwen3-dropped` (state 2, untrained) | `Qwen/Qwen3-0.6B` + identity rotary | **reference floor only**, not a comparison arm — near-random per RS1a |
| eval text | FineWeb-Edu `sample-10BT`, packed per RS1's frozen rule | **offset by 5,000,000 tokens** (see §2.2) |

State 2 is included so every "cost" number has a *zero-recalibration* lower bound on the same
axis. It is explicitly not part of any prediction: a near-random model's perturbation deltas are
uninterpretable as acuity.

### 2.2 Eval slice

Reuse `eval_perplexity.py`'s frozen packing rule (stream `sample-10BT` train in provider order,
one EOS after each non-empty document), but take tokens `[5_000_000, 5_000_000 + N)` — i.e. the
slice **immediately after** the 5M-token prefix used for RS1a/RS1b perplexity and for
training-time eval monitoring. RS3 conclusions then rest on text no RS1 number ever touched.

Implementation: add an `offset_tokens: int = 0` parameter to `fineweb_edu_eval_ids`. At default 0
its behavior must be byte-identical to today's, so RS1a/RS1b perplexity stays reproducible.

### 2.3 Arm A — local-order axis (primary)

For each model, score CE over `N_A = 1,000,000` tokens packed into 2048-token blocks, under:

- **clean** — unperturbed.
- **scramble(w)** for `w ∈ {2, 4, 8, 16, 32}` — within each non-overlapping window of `w` tokens,
  apply a random permutation (seed 0, **the same permutation for both models**).
- **reverse(w)** for the same `w` — deterministic within-window reversal. Robustness variant
  sharing the code path; guards against the random-permutation result being a sampling artifact.

CE is scored over the whole perturbed block (the perturbation is applied to the sequence the model
both conditions on and predicts — this is a *sequence-level* order perturbation, not a
context-only one; say so in the notebook, it is a common source of confusion).

Metrics, per model:
- `delta_ce(w) = CE_perturbed(w) − CE_clean` — **primary**, absolute order-sensitivity.
- `rel_delta_ce(w) = delta_ce(w) / CE_clean` — **secondary**, normalized for the level shift.

Both are reported. A strong read requires them to agree in direction; if they disagree, the
result is "sensitive to normalization" and must be reported as such, not resolved by picking one.

### 2.4 Arm B — retrieval axis (primary)

Synthetic induction sequences, `n_seq = 256` per distance, seed 0, identical sequences for both
models:

- Sample a span `S` of `k = 32` token ids uniformly from `[1000, 20000)` excluding all special
  ids (mid-frequency band; avoids both special tokens and the unused tail of the vocab).
- Sample filler `F` of length `d` from natural FineWeb-Edu text (the §2.2 slice).
- Sequence = `S · F · S`.
- `d ∈ {64, 256, 512, 1024, 1536}` (total length stays ≤ 2048, the recalibration context).

Metrics, per model and distance:
- `ce_first` — mean CE on the tokens of the **first** occurrence of `S` (unpredictable by
  construction: this is the per-model floor for "random tokens with no retrievable antecedent").
- `ce_second` — mean CE on the **second** occurrence, excluding its first token (which carries no
  match cue yet).
- `induction_gain(d) = ce_first − ce_second` — **primary**. Strictly within-model. A model that
  cannot retrieve the earlier span scores ≈ 0.

### 2.5 Arm C — key–value retrieval (secondary, floor-gated)

Context of `M = 40` lines `"<key>: <value>"` with keys/values sampled as in §2.4 (2 tokens each),
then the query `"<key_j>:"`. Measure (i) top-1 accuracy of the first value token and (ii) CE of
the true value. Vary `j` across 5 evenly-spaced depths, `n_seq = 128` per depth.

This is the closest thing to a "retrieval task" in the RNoPE-SWA sense, but a 0.6B **base** model
may sit at the floor, which is why it is secondary and gated (G-RS3.4). If the gate fails, report
the task as uninformative — **not** as "both models fail retrieval."

### 2.6 Arm D — length behavior (exploratory only, cannot carry a verdict)

CE over `N_D = 500,000` tokens at eval contexts `{1024, 2048, 4096, 8192}`, reported as
`CE(ctx)/CE(2048)` per model.

**Why this cannot adjudicate C3's length-extrapolation clause:** `qwen3` was *pretrained* at 32k
context, so 4096/8192 is in-distribution for it; `qwen3-droped` was *recalibrated* at 2048. Any
DroPE'd degradation past 2048 is therefore consistent with "recalibrated at a short context" and
with "lost length behavior" alike, and the two are not separable with the checkpoints on hand.
Report the curves as a diagnostic; make no C3 claim from them.

### 2.7 Statistics (pinned in advance)

- All comparisons are **paired on identical items** (same 2048-token block; same synthetic
  sequence), so compute per-item deltas first, then aggregate.
- Report mean, and a **95% bootstrap CI over items** (10,000 resamples, seed 0). Bootstrap over
  *blocks* for Arm A and over *sequences* for Arms B/C.
- Where an SD is reported, use the sample SD (`ddof=1`).
- Cross-model contrasts are **differences of paired within-model deltas**; bootstrap those
  directly over the shared item index rather than combining two separate CIs.

---

## 3. Gates (each must be able to fail)

- **G-RS3.1 — the harness is wired to the right checkpoints.** On the original frozen 5M-token
  prefix, the harness reproduces `qwen3` CE = 3.0819 within **±0.005** (fp32-to-fp32, directly
  comparable) and `qwen3-droped` CE = 2.826 within **±0.05** (looser by necessity — see the dtype
  note below). Fails → wrong checkpoint, wrong packing, or wrong rotary patch; stop. (Cheap, and
  it catches the single most likely way this experiment goes silently wrong.)

  > **Dtype asymmetry — flagged here because RS3 is the first run in a position to close it.**
  > `eval_perplexity.py` has only ever been run on states 1–2 (`NOTEBOOK.md` RS1a entry); it loads
  > fp32 via `load_model`. State 3's CE 2.826 / PPL 16.88 comes from the **training script's own
  > eval**, which held the model in **bf16** (`train_qwen3_nope.py` loads
  > `torch_dtype=torch.bfloat16`). The reduction is the same function
  > (`token_weighted_ce_and_ppl`) and the eval slice is the same 5M prefix, but the precision is
  > not. So the program's headline "16.88 vs 21.80" comparison currently mixes a bf16 measurement
  > against an fp32 one, and no frozen-harness fp32 number exists for state 3. RS3 must run
  > `eval_perplexity.py --models qwen3-droped` once (~5 min, needs the GPU it already has) and
  > **report that fp32 CE as a first-class result**, so later work stops propagating the mixed
  > comparison. The ±0.05 tolerance above is a placeholder for the expected bf16→fp32 shift; if the
  > gap exceeds it, that is a finding to investigate, not automatically a wiring error.
- **G-RS3.2 — the local perturbation does what it claims.** For both models and both perturbation
  modes: `CE_perturbed(w) > CE_clean` for every `w`, and `delta_ce` is non-decreasing from
  `w = 2` to `w = 32` (larger windows destroy more order). Fails → the perturbation is not
  measuring order sensitivity; Arm A is void.
- **G-RS3.3 — the retrieval instrument is above floor.** `qwen3` shows `induction_gain > 0.5` nats
  at `d = 64` with a CI excluding zero. Fails → the instrument cannot detect retrieval even in the
  model that is supposed to have it, so it cannot detect *loss* of retrieval; Arm B is void and
  C3's retrieval half is unadjudicated (report as such — do not substitute Arm C).
- **G-RS3.4 — Arm C is above floor.** `qwen3` top-1 accuracy at the shallowest needle depth is
  ≥ 3× the empirical chance rate. Fails → Arm C uninformative; drop it from adjudication.

---

## 4. Pre-registered predictions

- **(P.RS3.a — local-order cost, primary).** The DroPE'd model is **less** order-sensitive than
  RoPE at short windows: `delta_ce_droped(w) < delta_ce_rope(w)` for `w ∈ {2, 4, 8}`, paired CI
  excluding zero, and the same sign under `rel_delta_ce`. *Falsifier:* `≥` at those windows →
  removal did not cost local-order acuity, and C3's positive half fails.
- **(P.RS3.b — retrieval preserved, primary).** `induction_gain_droped(d) ≥ induction_gain_rope(d)`,
  or within CI of it, at **every** `d`. *Falsifier:* DroPE'd gain is below RoPE's with a CI
  excluding zero at any `d` → retrieval degraded on removal.
- **(P.RS3.c — the axis contrast, the actual C3 test).** Proportional degradation is larger on the
  local axis than on the retrieval axis:
  `D_local = [delta_ce_rope(4) − delta_ce_droped(4)] / delta_ce_rope(4)` exceeds
  `D_retrieval = [gain_rope(512) − gain_droped(512)] / gain_rope(512)`.
  *Falsifier:* `D_retrieval > D_local` → removal cost retrieval more than local order, which is
  C3's own falsifier. **The `w=4` and `d=512` anchors are fixed here, in advance, to prevent
  post-hoc selection of the window/distance that produces the desired contrast**; the full curves
  are reported regardless, and any curve-based reading is explicitly secondary.
- **(P.RS3.d — untrained floor, sanity).** State 2 (`qwen3-dropped`) shows near-zero
  `induction_gain` at all `d` and a much-compressed `delta_ce`. Confirms both metrics register the
  known-degenerate model as degenerate. *Falsifier:* state 2 looks healthy on either axis → the
  metrics are not measuring capability; both arms are suspect.

---

## 5. Decision tree

- **P.RS3.a ✓ + P.RS3.b ✓ + P.RS3.c ✓** → **C3 supported.** RoPE's causal contribution is local
  acuity, not retrieval. Combined with C1/C2 this completes the scaffold thesis behaviorally.
  Proceed to RS4 (scale).
- **P.RS3.a ✓, P.RS3.b ✗** → **C3 falsified as stated** — removal cost *both*. Retrieval loss is
  the headline; E2 ("position is not a retrieval address") is in tension with a behavioral
  retrieval loss and must be reconciled before RS4. Stop and rethink, per README's gating.
- **P.RS3.a ✗, P.RS3.b ✓** → removal cost neither axis measurably. C3 is unsupported but not
  falsified in its ordering; the honest read is "at 0.6B with 1B recalibration tokens, RoPE
  removal has no detectable behavioral cost on these instruments" — which is itself a strong,
  publishable DroPE-corroborating result, and a reason to question whether the instruments are
  sensitive enough (check against state 2 via P.RS3.d).
- **Borderline on P.RS3.c** (the two proportional degradations are within overlapping CIs, or the
  sign flips between the `w`/`d` anchors and the curves) → **do not adjudicate.** Run
  **RS1b-ctrl** ([RS1-spec §11](RS1-spec.md#11-addendum-2026-07-24-rs1b-ctrl--rope-recalibrated-confound-control)),
  which is pre-registered and currently deferred, and re-run RS3's arms against the
  RoPE-recalibrated model instead of the raw RoPE baseline. This is exactly the "necessary if
  borderline" branch §11 anticipated. Cost: one recalibration run (~7h H100, cf. RS1b) plus a
  cheap re-run of this harness.
- **Any gate fails** → adjudicate only the arms whose gates passed; record the void arm as void.

---

## 6. Threats to validity

- **The residual training confound.** Within-model contrasts cancel the *level* shift, not a
  change in *sensitivity* induced by 1B extra tokens. Mitigated on the local axis by the
  bias-direction argument (§1: the confound works against P.RS3.a), **not** mitigated on the
  retrieval axis. If the verdict turns on Arm B, RS1b-ctrl is the fix, per §5.
- **P.RS3.c compares two metrics in different units.** `delta_ce` (nats of order sensitivity) and
  `induction_gain` (nats of retrieval advantage) are made comparable only by expressing each as a
  *proportion of the RoPE model's own value*. That is a defensible "which axis moved more,
  proportionally," **not** a formal test of a difference between axes, and must be worded that way
  in the notebook. Do not report a p-value for the cross-axis contrast.
- **Synthetic retrieval ≠ retrieval in the wild.** Induction over a random span is the cleanest
  content-addressed retrieval available at this scale, but it is a narrow operationalization; C3's
  "retrieval" in the RNoPE-SWA sense is a broader capability. Arm C partly widens this, floor
  permitting.
- **Sequence-level perturbation conflates conditioning and prediction.** Scrambling the whole
  block changes both what the model sees and what it must predict. This is intentional (it is the
  cheap, assumption-free version) but it means `delta_ce` is not purely a *readout* of local-order
  representation.
- **Single model, single seed, single recipe** — same standing limitation as RS1/RS2/RS2.1.
- **Arm D cannot support a C3 claim** (§2.6). Included as a diagnostic only.

---

## 7. Schedule & budget

GPU inference only; no training, no new checkpoints.

| arm | forwards | est. wall-clock (single mid-range GPU) |
|---|---|---|
| A (local, 3 models × 11 conditions × 1M tok) | ~33M tok | ~30–45 min |
| B (induction, 3 × 5 × 256 seqs) | ~4M tok | ~5 min |
| C (KV, 3 × 5 × 128 seqs) | ~2M tok | ~3 min |
| D (length, 3 × 4 × 0.5M tok) | ~6M tok | ~10 min |
| G-RS3.1 (2 models × 5M tok) | 10M tok | ~10 min |

Roughly **1.5h on one GPU**, single-digit dollars on a 4090-class pod. The `qwen3-droped`
checkpoint must be on the network volume (`/workspace/qwen3-droped`) — confirm before booting.
Engineering is the real cost: one new script (~350 lines) plus the `offset_tokens` addition.

---

## 8. Deliverable

`rs3_local_scramble.csv`, `rs3_induction.csv`, `rs3_kv_retrieval.csv`, `rs3_length_ce.csv`, and
`rs3_summary.json` (gate outcomes, per-prediction adjudication, bootstrap CIs) — plus a completed
`NOTEBOOK.md` entry adjudicating P.RS3.a–d and placing C3 in the decision tree of §5.

Secondary deliverable, independent of the C3 verdict: the **first fp32 frozen-harness perplexity
for state 3** (G-RS3.1's dtype note), which retires the mixed bf16/fp32 "16.88 vs 21.80"
comparison the program has been quoting.

---

## 9. Implementation notes

- **New script:** `experiments/rope-as-scaffold/scripts/rs3_behavioral.py`, with
  `--task {local_scramble,induction,kv_retrieval,length_ce}` selecting the arm and `--models`
  taking the tags. Load models through `deadkeys.common.loading.load_model` — it already applies
  the identity-rotary patch for `qwen3-dropped`/`qwen3-droped` and enforces `QWEN3_DROPED_PATH`.
  Do not reimplement model loading or the rotary disable.
- **Reuse, don't fork, the eval slice.** Import `fineweb_edu_eval_ids` from `eval_perplexity.py`
  and extend it with `offset_tokens` (§2.2). Keep default behavior identical.
- **Reuse the CE reduction** already frozen in `eval_perplexity.py`
  (`token_weighted_mean_cross_entropy_then_exp`) so RS3's CEs are directly comparable to RS1's.
  Report CE, not perplexity, for all deltas — differences of logs are the meaningful quantity.
- **Per-item outputs are mandatory.** Write per-block and per-sequence rows, not just aggregates.
  RS2.1 needed per-head rows to re-derive statistics under review and they existed; the same will
  be true here. The summary JSON must record every CLI flag that affects results.
- **Determinism:** one seed (0) drives the perturbation permutations, the synthetic-token sampling,
  and the bootstrap. The *same* perturbed/synthetic items must be scored by every model — generate
  items once, then loop models inside.
- Batch by block; `torch.no_grad()`; fp32 to match RS1's loading path.

```bash
# sketch — exact invocation goes in the pre-run notebook entry
export QWEN3_DROPED_PATH=/workspace/qwen3-droped
RUN_ID=rs3_behavioral_$(date -u +%Y%m%dT%H%M%SZ)
PYTHONPATH=experiments/dead-keys ./scripts/cuda-python \
  experiments/rope-as-scaffold/scripts/rs3_behavioral.py \
  --task local_scramble --models qwen3 qwen3-droped qwen3-dropped \
  --eval-tokens 1000000 --eval-offset-tokens 5000000 \
  --windows 2 4 8 16 32 --modes scramble reverse \
  --bootstrap 10000 --seed 0 \
  --output-dir outputs/${RUN_ID}
```
