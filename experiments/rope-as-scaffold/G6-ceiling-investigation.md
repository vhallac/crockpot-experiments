# G6 ceiling — M1.6 becomes inapplicable as models improve (investigation, not yet scoped)

**Dated:** 2026-07-29
**Status:** **reference only — not pre-registered, not scheduled.** To be scoped as its own
run/experiment after the current program work completes.
**Raised by:** RS-amendment-2-3 (`NOTEBOOK.md` 2026-07-28), where M1.6 could not be run on
`qwen3-rope-recal` at all.
**Affects:** the M1.6 instrument (`experiments/k-address-space/kaddress/scripts/m16_discriminator.py`),
claim **E2**, and — prospectively — **RS4**.

---

## The observation

M1.6's gate **G6 (marker neutrality)** requires the model to be near-indifferent between four
candidate continuation markers: `max_prob / min_prob < 3.0` at the readout position
(`m16_discriminator.py` line 563). Markers are found by random search over combinations drawn
from `CANDIDATE_MARKERS` (152 entries, filtered to single-token), 4 at a time
(`PROBED_MARKER_ROLES`), with a budget of `--max-marker-sets` (default 512).

Across three checkpoints of the *same* base model, G6 feasibility degrades monotonically with
model quality:

| model | held-out PPL | G6 outcome |
|---|---:|---|
| `qwen3` (untrained) | 21.80 | passes at default budget (512) |
| `qwen3-droped` (1B tokens, RoPE off) | 16.88 | needs `--max-marker-sets 4096` |
| `qwen3-rope-recal` (1B tokens, RoPE on) | **14.25** | **fails at 4096** — best ratio 3.311 |

**The discriminator is model quality, not training exposure.** `qwen3-droped` and
`qwen3-rope-recal` received *identical* training — same cached token stream, same 1907 steps,
same recipe, differing only in the rotary patch — and one passes while the other fails. What
separates them is that the RoPE-active model is simply better (14.25 vs 16.88 PPL). Sharper
predictive distributions make near-indifference between four adverbs harder to find.

## Why this matters beyond one missing datapoint

1. **It is a ceiling on the instrument, not a calibration nuisance.** If G6 feasibility falls as
   perplexity falls, then M1.6 — and therefore the entire causal-addressing readout behind **E2**
   — becomes progressively inapplicable to better models. The instrument works best exactly where
   the models are least interesting.
2. **It threatens RS4 directly.** RS4's design is an E1/E2 spot-check on a **larger** model. A
   larger model will be a better model, and on this evidence more likely to fail G6 outright. RS4
   should not be launched assuming M1.6 will run.
3. **Cross-model M1.6 comparisons already use different stimuli.** `qwen3` and `qwen3-droped`
   passed G6 with *different* marker sets (found at 512 vs 4096 budget). That is by design —
   neutrality must be established per model — but it means RS1b's M1.6 comparison was never
   literally same-input across states. This wrinkle was not flagged when P.RS1.c was adjudicated
   and should be revisited.

## What the RS-amendment-2-3 failure does *not* establish

The lab assistant's report concluded that the marker vocabulary is fundamentally inadequate —
that "no combination of adverbs from the list will be neutral." **That does not follow from the
evidence.** The search draws 4 markers from ~152 candidates: C(152,4) ≈ **21 million**
combinations. A 4096-set budget samples ~**0.02%** of that space. The trend across budgets is
also *encouraging* rather than terminal — best ratio improved 5.534 → 3.311 when the budget went
512 → 4096, against a 3.0 target — and at 4096 stimulus M16_01 passed while M16_02 did not.

An exhaustion claim needs exhaustion evidence. It may still be true; it is not yet shown.

## Questions this investigation should answer

1. **Is it actually infeasible, or under-searched?** Run the marker search at much larger budgets
   (65536+, ~0.3% of the space) on `qwen3-rope-recal`. Cheap — each check is one forward pass.
2. **What is the shape of the ratio distribution?** Log the full distribution of `max_min_ratio`
   over a large sample rather than just the running minimum. That distinguishes "the tail reaches
   below 3.0 and we under-sampled" from "the distribution is bounded above 3.0 for this model."
3. **Does the ceiling scale as predicted?** If G6 difficulty tracks perplexity, that relationship
   should be measurable and extrapolable — which would let RS4 predict feasibility *before*
   spending on a larger model.
4. **Is there an instrument redesign that removes the dependence?** E.g. per-model marker
   calibration against a larger or model-specific vocabulary, a neutrality criterion that is
   relative rather than absolute, or a readout that does not require near-uniform priors at all.
   **Constraint:** any change to the marker vocabulary or the 3.0 threshold breaks comparability
   with every existing M1.6 result, so a redesign implies re-running all prior models.

## Constraints on any fix

- **Do not relax the 3.0 threshold** to make a run succeed. It would void comparability with all
  existing M1.6 results and defeat the gate's purpose.
- **Do not expand `CANDIDATE_MARKERS` for a single model.** Same problem: it changes the
  instrument. If the vocabulary is expanded, every previously-probed model must be re-run.

## Evidence pointers

- `NOTEBOOK.md` 2026-07-28 (RS-amendment-2-3) — P.ctrl.e recorded blocked, with the failure ratios
- `NOTEBOOK.md` 2026-07-25 / 2026-07-26 — the `qwen3-droped` runs, including the
  `--max-marker-sets 4096` note
- `m16_discriminator.py` — `CANDIDATE_MARKERS` (line 18), G6 threshold (line 563),
  `_single_token_markers` (line 213), search loop and failure path (lines 574–624)
