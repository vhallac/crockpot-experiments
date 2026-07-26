# RS1b learning-rate retuning — resolved (historical)

**Status:** **resolved, 2026-07-26 (reviewer catch: this note was stale, still said "parked").**
Option **A** (full clean restart at a corrected LR) was chosen and run at **peak LR 1e-3**
(the stretch option below, not the safer 3e-4 first step — Qwen3's native QK-norm made it safe as
predicted, no instability). Held-out PPL recovered from ~30,859 to **16.88** (below the 21.8 RoPE
baseline), crossing the baseline around step ~800 of 1907 and improving smoothly to completion —
see `NOTEBOOK.md`'s "2026-07-25 — RS1b LR-corrected rerun" entry for the full curve and the M1.5/M1.6
results this unblocked. `RS1-spec.md` §10.C now records 1e-3 as the canonical recipe value. The
rest of this note is kept as-is below for historical record of the reasoning that led there —
read it as the pre-decision analysis, not a current open question.

## The finding

RS1-spec.md §10.C picked peak LR **3e-5** for recalibration, flagged at the time as "the #1 recipe
risk (P.RS1.a null-masquerade)." Reading DroPE's actual paper (arXiv 2512.12167) surfaced direct
evidence that this may have been too conservative:

- DroPE's own recalibration LRs are **higher than pretraining peak**, not lower: 1.0×10⁻³ for the
  small-scale 500M-param ablation (vs. 3.0×10⁻⁴ pretraining peak); 1.0×10⁻³ for SmolLM (their
  attempted 3.0×10⁻³ caused instability at small batch).
- Appendix D.3 / Table 11 runs an ablation on SmolLM-360M (close to our 0.6B scale) at **exactly
  our LR (3×10⁻⁵)** alongside their higher settings, final recalibration loss:

  | LR | with QKNorm | without QKNorm |
  |---|---|---|
  | 1×10⁻³ | 2.496 (best) | 6.334 (unstable) |
  | 3×10⁻⁴ (their default) | 2.555 | 2.530 |
  | **3×10⁻⁵ (ours)** | **3.102** | **2.713** |

  At our exact LR, their model lands measurably worse than their default. Our own run has
  plateaued around PPL ≈35 (CE ≈3.55) against a ≈21.8 target — the same shape of shortfall.
- Qwen3 already has QK-norm natively (it's part of the base architecture, not something we'd need
  to add) — the paper's QK-norm ablation shows QK-norm is specifically what makes the *higher* LR
  (1e-3) safe rather than unstable. So we may be *better* positioned than their raw ablation model
  to push LR up, not worse.

**Read:** the current RS1b plateau is plausibly an under-tuned-LR artifact, not evidence that
RoPE-removal recovery has a hard ceiling. Not confirmed — a real alternative explanation (corpus
mismatch: Qwen3's broad pretraining mix vs. our narrow FineWeb-Edu-only recalibration corpus,
already flagged in RS1-spec.md §7) could also produce a plateau. The two aren't mutually exclusive
and this note doesn't adjudicate between them.

## Options once the current run finishes

**A. Full clean restart at a corrected LR (leaning toward this).** Re-run the identical recipe
(§10.C) from the untouched base checkpoint, only changing peak LR — most likely **3e-4** (DroPE's
own broadly-validated "default," a 10x increase, safer first step) with 1e-3 as a stretch option
given Qwen3's native QK-norm should stabilize it per the paper's own mechanism. Per the paper's own
data, higher LR does not need *more* tokens to reach a better optimum within the same budget — so
this should cost roughly the same ~12.5h/~$19 as the current run, not more. Cleanest to interpret:
each run stays a single-phase, spec-matching recalibration; "RS1b v1" (this run) and "RS1b v2"
(corrected LR) are both valid, independently-interpretable data points rather than one run
retroactively patched mid-flight.

**B. Continue the finished checkpoint for additional steps at a bumped-up LR.** Cheaper in the
sense of not discarding the current run's progress, but this is a two-phase schedule (cosine down
to near-zero, then re-warmed back up) that DroPE's paper doesn't validate — it's a novel variant,
not a replication of their recipe. Re-warming LR after Adam's moment estimates have settled into a
near-converged, low-gradient-noise regime carries some risk of a transient loss spike (recoverable,
but not something we have direct evidence about in this exact setting). Also muddies what "state 3"
means relative to the already-pre-registered spec — would need to be framed as an explicit new
variant, not a silent continuation of the current `qwen3-droped` artifact.

**Leaning:** (A) is more rigorous and probably no more expensive; (B) is a legitimate fallback if
compute/cost becomes the binding constraint, not if it doesn't.

## Consequence for RS1b-ctrl (§11 addendum)

If RS1b gets rerun at a corrected LR, **RS1b-ctrl's recipe (RS1-spec.md §11) must match whichever
RS1b recipe becomes canonical**, not stay pinned to the original 3e-5. Running the control at a
since-superseded, under-tuned LR would replicate the same confound-masking problem in the control
arm and defeat its purpose. Revisit §11's recipe reference once this is decided.

## Do not act on this yet (historical — resolved, see Status above)

This section originally said to wait for the run and its M1.5/M1.6 results before deciding between
(A)/(B)/neither. That decision is made: (A) ran, resolved cleanly (PPL 16.88, M1.6 transitivity and
addressing both restored to RoPE-comparable levels). Kept here unedited as the record of what the
decision criteria were before the data existed.
