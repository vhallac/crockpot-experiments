# RS-amendment-2-3 — RoPE-recalibrated control, amending RS2 and RS3

**Dated:** 2026-07-28
**Status:** pre-registered, not yet run
**Program:** [`rope-as-scaffold`](README.md) — unblocks **C3** (amends RS3), strengthens **C2**
(amends RS2/RS2.1), closes the original **C1** confound question.

**What this is.** The execution plan for the control experiment pre-registered as **RS1b-ctrl**
in [RS1-spec §11](RS1-spec.md#11-addendum-2026-07-24-rs1b-ctrl--rope-recalibrated-confound-control).
It is filed as an *amendment* rather than a new RS-number because it produces no new claim of its
own: it re-runs analyses already completed for RS2 and RS3 against a properly-controlled baseline,
and its output is a correction (or confirmation) of those entries' verdicts. §11 remains the
pre-registration of record for the **recipe** and the original P.RS1.ctrl.a/b; this document
scopes the **analysis arms** that consume the checkpoint §11 defines, and pre-registers their
predictions and decision matrix.

**Amends:** `NOTEBOOK.md` 2026-07-27 (RS3 — currently ends "no C3 verdict yet"), and
2026-07-26 (RS2/RS2.1 — C2 substantiated, with a training-drift baseline still missing).

---

## 0. Question

Every RoPE-vs-DroPE'd comparison in this program so far has compared `qwen3` (**zero** extra
training) against `qwen3-droped` (**~1B** extra FineWeb-Edu tokens). That confounds two variables
that were never independently manipulated: **RoPE removed** and **received extra in-domain
training**.

RS3 made this decisive rather than academic. Both of its primary predictions falsified, and
neither falsification is attributable:

- **P.RS3.a** (local-order cost) falsified in *exactly* the direction RS3-spec §1's own
  bias-direction argument said the confound would push it.
- **P.RS3.b** (retrieval preserved) falsified with **no** conservative argument available —
  §1 said so explicitly, in advance.

**The question this experiment answers:** with domain adaptation held constant on both sides,
which of RS3's effects survive? I.e. train a **RoPE-active** copy on the identical data and
compare `qwen3-rope-recal` vs `qwen3-droped`, so the only remaining difference is RoPE.

---

## 1. What the control does and does not close

**Closes:** the training/removal confound for every C1/C2/C3 comparison built on the
state-1-vs-state-3 contrast. After this, "RoPE removed" is the only variable that differs
between the two trained arms.

**Does not close:** RS2.1's *remaining* caveat — "reconstructs RoPE's *specific* code" vs. "any
functional NoPE training at this scale converges to similar positional geometry." That needs an
**independently-initialized** control (from-scratch or differently-seeded NoPE), which this is
not: `qwen3-rope-recal` starts from the same weights as everything else. Keep the two controls
distinct; this plan does not claim the independent-init gap is addressed.

**Also does not fix:** RS3 Arm D (length behavior). Its confound is pretrain-context (32k) vs.
recalibration-context (2048), and `qwen3-rope-recal` is recalibrated at 2048 too — it shares the
problem rather than resolving it. Arm D stays exploratory-only regardless.

---

## 2. Design

### 2.1 Stage 0 — the checkpoint (artifact generation, not a hypothesis test)

Per repo convention (cf. RS1b's training, `qwen3-nope-training.md`), producing the model is
artifact generation and runs **off** the NOTEBOOK.md hypothesis-test flow. It is still pinned and
published.

Recipe: **identical to `qwen3-droped`'s LR-corrected run in every respect except the rotary
patch.** From RS1-spec §10.C as revised, and `NOTEBOOK.md`'s RS1b LR-corrected entry:

| parameter | value |
|---|---|
| base | `Qwen/Qwen3-0.6B` @ `c1899de289a04d12100db370d81485cdf75e47ca` (**pass `--revision` explicitly**) |
| rotary patch | **skipped** — `set_qwen_rotary_identity` NOT applied; RoPE active throughout |
| corpus | `HuggingFaceFW/fineweb-edu`, `sample-10BT`, streamed, provider order |
| token budget | 1B, train context 2048 |
| optimizer | AdamW β=(0.9,0.95), wd=0.1 |
| LR | peak **1e-3**, cosine → 10% of peak, 2% warmup |
| seed | 0 |
| eval slice | same 5M-token held-out prefix |

**[MUST] The token stream must be the identical sequence `qwen3-droped` saw**, not merely another
1B-token draw — same seed, same streaming order, ideally the same cached token file
(`/workspace/rs1b-token-cache/`). A different draw reintroduces a nuisance variable the control
exists to eliminate.

**Engineering:** one flag on `train_qwen3_nope.py` (leave the identity patch disabled), plus a new
model tag. `deadkeys.common.loading` needs `qwen3-rope-recal` added to `MODEL_IDS`, loaded from a
local path via an env var (mirroring `QWEN3_DROPED_PATH`), and **deliberately excluded** from
`DROPPED_ROPE_TAGS` — that exclusion is the entire scientific content of the arm, so it warrants
an explicit test or assertion rather than being left implicit.

### 2.2 Arm 1 — RS3 Arms A + B re-run (**required**; this is why we're doing it)

Re-run `rs3_behavioral.py --task local_scramble` and `--task induction` with
`qwen3-rope-recal` in place of raw `qwen3`. Identical items (seed 0), identical windows/distances,
identical paired-bootstrap statistics as RS3.

Primary contrast becomes **`qwen3-rope-recal` (trained, RoPE-on) vs `qwen3-droped` (trained,
RoPE-off)**. Keep raw `qwen3` in the run as a third arm so all three points are on one axis:
untrained-RoPE / trained-RoPE / trained-noRoPE.

Arm C (kv_retrieval) is a cheap optional corroboration (~2 min); include it if convenient. Arm D
is out of scope per §1.

### 2.3 Arm 2 — RS2 subspace control (**recommended**; near-free once the checkpoint exists)

RS2.1 established the *initialization* baseline (V2: untrained-dropped, zero training). It has no
*training-drift* baseline — "how much does 1B tokens of this corpus move key geometry toward
`k_post` regardless of RoPE?" `qwen3-rope-recal` supplies exactly that third point.

Run M1.5 on `qwen3-rope-recal` (`--families A --max-length 1024`, matching RS1a/RS1b so bases are
comparable), then `c2_subspace_overlap.py` with `--droped-projectors` pointed at the resulting
`k_pre` NPZ and `--rope-projectors` at RS1a's `qwen3` NPZ — same seed, same
`--baseline-trials 100`, same families as RS2.

This brackets the RS2 effect between three conditions rather than two:

| condition | L3–L7 excess | L8–L12 excess | source |
|---|---:|---:|---|
| untrained, RoPE-off (V2) | +0.071 | −0.023 | RS2.1 |
| **trained, RoPE-on** | **?** | **?** | this arm |
| trained, RoPE-off (V1) | +0.269 | +0.180 | RS2 |

### 2.4 Arm 3 — original §11 predictions (**near-free bonus**)

Once the checkpoint exists, run the frozen perplexity eval and M1.6 on it to settle
P.RS1.ctrl.a/b as originally written. M1.5 is already required by Arm 2.

**P.RS1.ctrl.a needs restating in light of what we now know.** §11 wrote it when we expected
state 3 to recover *toward* ~21.8 PPL. It actually reached **16.88 — better than the untrained
RoPE baseline.** The live question is therefore no longer "does the control fail to recover" but
"how much of state 3's *sub-baseline* perplexity is domain adaptation?" See P.ctrl.d below.

---

## 3. Gates

- **G-ctrl.1 — the control is actually RoPE-active.** Assert `uses_dropped_rope("qwen3-rope-recal")
  is False`, and verify positionally: the model's `k_pre` and `k_post` must **differ** (unlike
  `qwen3-droped`, where G-RS1.1 established they are numerically identical). If `k_pre == k_post`,
  the identity patch leaked in and the whole control is void.
- **G-ctrl.2 — training matched.** The control's token count, step count, LR schedule, and seed
  match `qwen3-droped`'s `training_metrics.csv` exactly. Any mismatch → the arms aren't
  comparable; stop and fix before analysis.
- **G-ctrl.3 — the control is a functional model.** Held-out PPL is in a sane range (single/low
  double digits, not near-random) — otherwise the training run itself failed and no comparison is
  meaningful.
- **G-ctrl.4 — RS3 harness reproduces.** Re-run G-RS3.1 against this exact code, which RS3 itself
  never did (`NOTEBOOK.md` 2026-07-27, Gates). Also fix G-RS3.2 to split by perturbation mode
  before re-use, per RS3's follow-up #2.

---

## 4. Pre-registered predictions

**(P.ctrl.a — local-order, the P.RS3.a re-test).** With training held constant, if RoPE removal
genuinely costs local-order acuity, then `delta_ce(rope-recal) > delta_ce(droped)` at w∈{2,4,8}
(paired CI excluding zero). *RS3 observed the opposite sign against the untrained baseline
(−0.024 at w=4).* **Falsifier:** `delta_ce(rope-recal) ≤ delta_ce(droped)` → C3's local half is
falsified for real, not as a confound artifact.

**(P.ctrl.b — retrieval, the P.RS3.b re-test; the decisive one).** If the induction collapse is
RoPE-specific, `qwen3-rope-recal`'s gain stays near raw `qwen3`'s (~12.9 nats, flat across
distance) while `qwen3-droped` stays at ~11.2 (d=512) and decaying. **Falsifier:**
`gain(rope-recal) ≈ gain(droped)` → the collapse is narrow-domain-training-induced, RS3 Arm B is
void as C3 evidence, and C3's retrieval half returns to unadjudicated.

**(P.ctrl.c — subspace, the RS2 control).** `qwen3-rope-recal`'s `k_pre`-vs-`qwen3-k_post` excess
in L3–L12 stays near the untrained V2 level (≈ −0.02 … +0.07), **not** near the trained-RoPE-off
V1 level (+0.18 … +0.27). *Supports:* C2's reconstruction is RoPE-removal-specific. **Falsifier:**
excess comparable to V1 → the "reconstruction" RS2/RS2.1 measured is substantially generic
training drift, and C2 needs re-adjudication a second time.

**(P.ctrl.d — perplexity attribution, restating P.RS1.ctrl.a).** `qwen3-rope-recal`'s held-out PPL
lands **near `qwen3-droped`'s 16.88**, well below the untrained 21.80 baseline. *Read:* state 3's
sub-baseline perplexity is domain adaptation, not evidence about RoPE — and the notebook's
"crosses below the baseline around step ~800" observation is not the meaningful signal it might
appear to be. The genuinely RoPE-attributable part of P.RS1.a is the *repair* (≈30,859 → ~21.8),
not the *overshoot* (~21.8 → 16.88). **Falsifier:** the control lands materially worse than 16.88
(say >19) → some of the overshoot is RoPE-removal-specific after all, which would be a surprising
and interesting result in its own right.

**(P.ctrl.e — M1.5/M1.6 profile, = P.RS1.ctrl.b).** The control's position-decodability and
addressing profile stay close to state 1's. *Falsifier:* a shift comparable to state 3's → C1's
causal attribution weakens correspondingly.

---

## 5. Decision matrix for C3 (the point of the exercise)

Applied to Arm 1's results, using RS3's pre-registered anchors (w=4, d=512):

| P.ctrl.a (local) | P.ctrl.b (retrieval) | C3 verdict |
|---|---|---|
| holds (recal more sensitive) | holds (recal retains gain) | **C3 supported after all** — RS3's double falsification was confound-driven; local cost is real and retrieval is spared |
| falsified | holds | **C3 falsified, cleanly and interestingly** — removal costs retrieval, not local order: the *inverse* of the claim. Forces the §1 reframing (RoPE's local primitive is what retrieval is built on) into the foreground |
| holds | falsified | Both axes degrade, local more; C3 partially supported but its "not retrieval" clause fails |
| falsified | falsified | **Neither effect is RoPE-attributable** — RS3's whole result was domain adaptation. C3 unadjudicated; the instruments are sound but the comparison was never informative. Report as such |

**Do not** collapse the bottom-right cell into "C3 falsified." A null control result means RS3
measured training, not RoPE, and the honest output is a retracted result plus a note on what
would be needed next (independent-init control, or a matched-domain corpus).

---

## 6. Threats to validity

- **Still single-model, single-seed, single-recipe** — standing limitation across RS1–RS3.
- **The control shares state 3's recalibration context (2048).** Fine for Arms 1–2 (all items fit
  well inside it); fatal for any length-extrapolation claim, hence Arm D's exclusion.
- **"Identical token stream" is load-bearing and easy to get subtly wrong** — a re-streamed
  dataset, a changed cache, or a different shuffle silently degrades this from a control to
  another arm. G-ctrl.2 exists for this; treat a mismatch as blocking, not cosmetic.
- **Arm 2's comparison is not perfectly symmetric.** `qwen3-rope-recal`'s `k_pre` sits in a model
  where RoPE still supplies position, so its keys face no pressure to encode position — which is
  precisely the control condition, but it means a *null* result there ("no drift") is partly
  expected by construction and shouldn't be oversold as a strong confirmation of C2.
- **Multiple arms off one checkpoint invites cherry-picking.** Predictions are fixed here, in
  advance, with anchors inherited from RS3; report every arm regardless of which way it lands.

---

## 7. Schedule & budget

| stage | compute | est. |
|---|---|---|
| Stage 0 — training | 1× A100 / H100 / H200 SXM — see GPU note below | ~12.5h, **~$20** |
| Arm 2 prereq — M1.5 probe | same pod | ~0.5–1h |
| Arm 3 — perplexity + M1.6 | same pod | ~1h |
| Arm 1 — RS3 A/B (+C) | same pod, batched harness | ~1h |
| Arm 2 — subspace analysis | CPU-only | minutes |

**Total ≈ $25–30**, one pod session. Do all GPU work in a single booting of the pod — the
checkpoint is the expensive artifact, and every downstream probe is cheap by comparison. Produce a
GPU readiness report before launching (AGENTS.md), and publish the checkpoint + outputs per the
reproducible-research flow.

**GPU note — by card, not by architecture family** (the family names are easy to mis-map):

| verdict | cards | why |
|---|---|---|
| **Safe** | A100 (Ampere), **H100 and H200** (both Hopper, CC 9.0), A5000/A6000, L40S, RTX 4090 | flash-attention supported |
| **Avoid** | B100, B200, GB200, RTX 5090, RTX PRO 6000 Blackwell | Blackwell; flash-attention unsupported (Dao-AILab/flash-attention#1987) |

**H200 is Hopper, not Blackwell** — same GH100 die as H100, just HBM3e and 141GB instead of 80GB.
It is safe, but buys nothing here: a 0.6B model at 2048 context is nowhere near memory-bound, so
prefer whichever of A100/H100/H200 is cheapest and available rather than paying an H200 premium.

---

## 8. Deliverable

- `qwen3-rope-recal` checkpoint, published, with its `training_metrics.csv`.
- RS3 Arms A/B (+C) CSVs re-run against it, adjudicating P.ctrl.a/b and yielding a **C3 verdict
  via §5's matrix** — the blocking output.
- `c2_subspace_overlap` CSV for the RS2 control arm, adjudicating P.ctrl.c.
- Frozen perplexity + M1.5/M1.6 outputs, adjudicating P.ctrl.d/e and closing §11's original
  question.
- A completed `NOTEBOOK.md` entry, plus **an amendment to RS3's 2026-07-27 entry** recording the
  now-attributable verdict (the RS3 entry currently ends at "no C3 verdict yet" and must be
  updated, not left dangling).

---

## 9. Implementation notes

- **Model tag:** add `qwen3-rope-recal` to `MODEL_IDS` → `Qwen/Qwen3-0.6B`, resolved from a local
  path via `QWEN3_ROPE_RECAL_PATH` (mirror the `qwen3-droped` branch in `load_model`). **Do not**
  add it to `DROPPED_ROPE_TAGS`. Add a test asserting `uses_dropped_rope("qwen3-rope-recal") is
  False` — G-ctrl.1 depends on it.
- **Training:** reuse `train_qwen3_nope.py`; the rotary-identity application is the only behavioral
  change. Prefer an explicit `--no-rotary-patch`-style flag over commenting the call out, so the
  run manifest records which mode ran.
- **Reuse the cached token stream** at `/workspace/rs1b-token-cache/` rather than re-streaming.
- **Pass `--revision c1899de289a04d12100db370d81485cdf75e47ca`** on every model-loading
  invocation — training, probes, and RS3 re-run. RS3's relaunch dropped it
  (`NOTEBOOK.md` 2026-07-27, "Provenance regression"); this run should not repeat that.
- **Fix G-RS3.2's mode-splitting** before the Arm 1 re-run, so the reverse-mode non-monotonicity
  stops masking the clean scramble-mode pass.
- **Signal-triggered snapshots** (SIGHUP/SIGUSR1) are already in `train_qwen3_nope.py` from RS1b;
  keep them enabled for a ~12h run.
