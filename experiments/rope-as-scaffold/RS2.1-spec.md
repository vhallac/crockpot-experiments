# RS2.1 — Subspace reconstruction vs. initialization inheritance (RS2 correction)

**Dated:** 2026-07-26
**Status:** pre-registered, not yet formally run
**Program:** [`rope-as-scaffold`](README.md) — re-tests claim **C2**, correcting RS2's adjudication.
**Depends on:** RS2 (`RS2-spec.md`, `NOTEBOOK.md` 2026-07-26 entry) — this experiment does not
replace RS2's data, it re-analyzes the same question with a confound RS2 didn't control for.
**Motivating critique:** `temp/rs2-critiqe.md` (external review, treated as reviewer input, not
pre-registration itself).

---

## 0. Question

RS2 concluded C2 "substantiated": the DroPE'd model's emergent key-position subspace
reconstructs the code RoPE supplied, evidenced by alignment excess over a random-rotation
baseline (+0.087 global, depth-structured: strong in early-mid layers, negative in late layers).

**The gap RS2 didn't close:** the DroPE'd model was *initialized from the RoPE model's own
weights* and recalibrated for only 1B tokens. Its pre-rotation key geometry could resemble
RoPE's `k_post` **simply because it started from RoPE's `k_pre`**, not because training
reconstructed anything. RS2 never asked "how much overlap would exist with zero training at
all?" — it has no baseline for that.

**Primary (C2, re-adjudicated):** *after controlling for what the same weights already show at
initialization (zero training)*, does training measurably increase the DroPE'd model's subspace
overlap with RoPE's `k_post`, specifically in the layers where RS2 claimed reconstruction?

**Falsifier:** trained-minus-untrained excess (Δ) is ≈ 0 or negative in the early-mid layers
(2–12) → the RS2 overlap is attributable to initialization inheritance, not reconstruction, and
C2 should be re-adjudicated as "the model retained the base model's emergent code; nothing
RoPE-specific was reconstructed by training."

---

## 1. Why this instrument, and what's already known

**Exploratory finding (2026-07-26, informal, CPU-only, pre-registration-motivating — not a
substitute for the formal run below).** Using the already-published RS1a `qwen3-dropped`
(untrained, zero-shot RoPE-removed) M1.5 projectors against RoPE `k_post`, computed with the
unmodified `c2_subspace_overlap.py`:

| layer | trained excess (RS2) | untrained excess (this check) | Δ (trained − untrained) |
|---:|---:|---:|---:|
| 1 | +0.066 | +0.018 | +0.049 |
| **2** | **+0.437** | **+0.350** | **+0.086** |
| 3 | +0.306 | +0.117 | +0.189 |
| 4 | +0.234 | +0.085 | +0.149 |
| 5 | +0.327 | +0.105 | +0.222 |
| 6 | +0.303 | +0.072 | +0.231 |
| 7 | +0.176 | −0.026 | +0.202 |
| 8 | +0.087 | −0.093 | +0.180 |
| 9 | +0.246 | +0.090 | +0.156 |
| 10 | +0.242 | +0.012 | +0.229 |
| 11 | +0.042 | −0.052 | +0.095 |
| 12 | +0.281 | −0.072 | +0.353 |

**Read carefully — this cuts two ways, not one:**
- **Layer 2** (the notebook's single headline "almost perfectly recovers RoPE's code" example)
  is **mostly inheritance**: the untrained model already shows +0.350 excess before any training;
  training adds only +0.086 more. The notebook's chosen illustration is misleading.
- **Layers 3–12** (the bulk of the "early-mid" phase RS2 reported) show **substantial Δ > 0**
  (+0.15 to +0.35) — training clearly moved the subspace closer to RoPE's `k_post` well beyond
  what initialization alone explains. The *aggregate* C2 claim for this phase looks like it
  survives the inheritance control; the specific headline example used to illustrate it does not.

This is exactly why a formal, gated, published RS2.1 is needed rather than treating the informal
check as sufficient: it changes the notebook's illustrative example and demands the V3–V5 checks
below before any claim is re-published as settled.

---

## 2. Design

### 2.1 Inputs (all pre-existing, no new training)

| tag | NPZ | source |
|---|---|---|
| RoPE `k_pre`/`k_post` | `kaddress_m15_projectors_qwen3.npz` | RS1a, `run/rope-as-scaffold-rs1a/20260724` |
| Untrained-dropped (zero-shot) | `kaddress_m15_projectors_qwen3-dropped.npz` | RS1a, same release |
| DroPE'd (trained, LR=1e-3) | `kaddress_m15_projectors_qwen3-droped.npz` | RS1b LR-corrected rerun, `run/rope-as-scaffold-rs1b-lr1e3/20260726` |
| RS2's per-head results | `rs2_subspace_overlap.csv` | RS2, `run/rope-as-scaffold-rs2/20260726` |

### 2.2 Method — five verification arms (from the motivating critique, formalized)

**V1 — Depth-resolved reference comparison (tabulation only, already done above informally;
formal run re-derives from committed data, no rerun needed).** Per layer, is primary excess
(trained vs `k_post`) ≤ Ref2 excess (RoPE `k_pre` vs `k_post`, already in RS2's CSV)? Informal
result: **no, not uniformly** — primary exceeds Ref2 at 9 of 11 early-mid layers (3–12), meaning
the internal-rotation channel alone does *not* suffice to explain the RS2 result. This is a
genuine argument *against* pure inheritance via that specific channel, and should be formally
re-derived and reported alongside V2 (they can disagree, and the disagreement is itself
informative about *which* inheritance channel — if any — is doing the work).

**V2 — Zero-shot inheritance control (the decisive test; formal rerun of the informal check
above, on the record, with committed provenance).** `c2_subspace_overlap.py`, run with
`--droped-projectors` pointed at the untrained-dropped NPZ instead of the trained one, same
RoPE projectors, same seed/baseline-trials/families as RS2. Compute
`Δ(layer) = excess_trained(layer) − excess_untrained(layer)` per layer, with per-head deltas
retained for the clustered-SE analysis in V5.

**V3 — Partial out the `k_pre`-shared component (new code required).** Per head: orthogonalize
the `k_post` basis against `k_pre` (residual = the rotation-specific component of RoPE's code,
i.e. what `k_pre` alone does *not* already explain); orthogonalize the trained DroPE'd basis
against `k_pre` likewise; compute principal-angle overlap between the two **residual**
subspaces against a dimension-matched random baseline. This directly measures reconstruction of
the component C2 actually claims — the code *rotation added* — with the inherited component
removed. **V2 and V3 should agree**; both positive in layers 2–12 is what "C2 substantiated"
legitimately requires.

**V4 — Projector semantics at the headline layer.** Cross-reference against M1.5's own
`position_fraction` at layer 2: the DroPE'd model's own M1.5 output records
`position_fraction ≈ 0.0075` at layer 2 (i.e., ~0.75% of key variance is position-related there).
If the aggregate PCA basis at layer 2 is overwhelmingly non-positional, a high alignment there
reflects shared **content geometry**, not shared **positional code**, regardless of what V2/V3
show — the layer would need to be excluded or reweighted, not just relabeled. Check this at every
layer in the early-mid phase, not just layer 2 (report `position_fraction` alongside
`alignment_excess` per layer).

**V5 — Statistical robustness.**
(a) Recompute the "~6 SE below zero" late-phase claim with SEs clustered by layer (80 heads
across 10 layers are not independent draws).
(b) Check random-baseline rank-sensitivity: report per-head PCA rank (`n_rope_post_components`,
`n_droped_components`, and — now available per the earlier G-RS2.3 fix — `n_rope_pre_components`)
against `alignment_excess`; if rank varies systematically with depth or between models, the
below-random late-layer result could be a rank-mismatch artifact rather than genuine divergence
(this is a live concern: the exploratory check above already found the untrained-dropped basis
has a *different rank* than RoPE's own `k_pre` at matched layer/head — e.g. layer 6/head 0: RoPE
`k_pre` rank 4 vs. untrained-dropped rank 14 — so rank is not a stable, comparable quantity across
independently-run M1.5 passes and must be checked, not assumed uniform).
(c) Report layer 1's phase statistics explicitly — RS2's phase table silently omitted it
(216 heads analyzed, 208 in the three named phases; layer 1 unaccounted for).

---

## 3. Gates

- **G-RS2.1.1 — inputs exist and load.** All four NPZ/CSV inputs in §2.1 are present and parse
  with the existing `load_projector_bases`. If the untrained-dropped NPZ were missing, regenerate
  via one M1.5 pass on `qwen3-dropped` at `--max-length 1024` (RS1a-documented settings) — not
  needed here, confirmed already present.
- **G-RS2.1.2 — V2 and V1 don't silently contradict without explanation.** V1 (primary vs Ref2)
  and V2 (trained vs untrained) measure related but distinct things (different reference channel).
  If they disagree in direction for the same layer, that must be explained (e.g. by rank
  differences per V5b) before adjudicating, not averaged over.
- **G-RS2.1.3 — V3's residual computation is well-conditioned.** After orthogonalizing against
  `k_pre`, residual rank must be reported; if residual rank collapses to ~0 for most heads (i.e.
  `k_post` is almost entirely explained by `k_pre` with no rotation-specific residual), V3 is
  uninformative by construction and that itself is a finding (RoPE's rotation adds very little
  new subspace structure at that layer), not a null result to discard.

---

## 4. Pre-registered predictions

- **(P.RS2.1.a) Reconstruction survives inheritance control in layers 3–12.** Δ (V2) > 0 with a
  layer-clustered CI excluding zero for layers 3–12, and V3's residual excess is also > 0 in the
  same range. *Falsifier:* either goes to ≈0 or negative → re-adjudicate per the decision matrix
  below.
- **(P.RS2.1.b) Layer 2 is not representative.** Layer 2's Δ is small relative to its untrained
  baseline (already observed informally: +0.086 Δ vs +0.350 untrained-alone) — i.e. layer 2
  specifically is inheritance-dominated even if other early-mid layers are not. If confirmed
  formally, the notebook's choice of layer 2 as the illustrative "strongest reconstruction"
  example must be corrected regardless of how the aggregate phase claim resolves.
- **(P.RS2.1.c) Late-layer divergence is not a rank artifact.** Per V5b, `alignment_excess` in
  layers 18–27 does not correlate with PCA-rank differences between models. *Falsifier:* a clear
  rank–excess correlation → the "model develops an independent code" claim needs to be
  re-examined as a possible dimensionality artifact instead.

---

## 5. Decision matrix (from the motivating critique, adopted verbatim)

| V2 (Δ, L2–12) | V3 (residual excess, L2–12) | Verdict |
|---|---|---|
| > 0, CI excludes 0 | > 0 | C2 reconstruction supported — depth-qualified as RS2 reported, now confound-controlled |
| ≈ 0 | ≈ 0 | C2 not supported → re-adjudicate as inheritance: "retained the base model's emergent code; rotation-specific component not reconstructed" |
| > 0 | ≈ 0 | Training increased similarity only within the `k_pre`-shared component — still not "the code RoPE supplied"; C2 fails as stated |
| ≈ 0 or < 0 | > 0 | Unexpected — investigate before adjudicating (likely rank/method artifact; see V5b) |

Independent of the matrix: if V4 shows layer 2 (and possibly other early layers) carry
negligible positional signal, the early-mid phase boundary itself needs re-drawing (e.g. to
layers 6–12 only, where both models have ridge R² ≈ 0.99 per existing M1.5 data) before the
matrix is applied — don't average a non-positional layer into a "positional subspace overlap"
verdict.

---

## 6. Threats to validity

- **This is still a single-model, single-seed, single-recalibration-recipe result** — same scope
  limitation as RS1/RS2.
- **V3's orthogonalization is a new, uncommitted method** (unlike M1.5/M1.6/RS2's core, which
  reuse validated code) — verify the residual-rank reporting in G-RS2.1.3 before trusting V3's
  numbers as more than exploratory.
- **The untrained-dropped M1.5 run and the trained DroPE'd M1.5 run are two independently-executed
  M1.5 passes**, not literally the same forward pass with/without training — sampling/aggregation
  details (documented already to affect PCA rank, §1) could introduce noise unrelated to the
  scientific question. V5b's rank-sensitivity check exists specifically to bound this.

---

## 7. Schedule & budget

CPU-only, analysis-only, no GPU, no training. V1/V2 are seconds each (already demonstrated
informally). V3 needs a small new script (orthogonalization + residual principal angles) —
same order of runtime as the existing script, minutes at most. V4/V5 are CSV/JSON post-processing
against data already on hand. Total: under an hour including the new V3 code, zero dollar cost.

---

## 8. Deliverable

A `rs2.1_subspace_delta.csv`/`.json` (V1/V2), a `rs2.1_residual_overlap.csv` (V3), and a
completed `NOTEBOOK.md` entry that either confirms RS2's C2 verdict with the inheritance confound
now controlled for, or issues a correction per the decision matrix — plus, independent of the
verdict, a fix to the notebook's layer-2 illustrative example if P.RS2.1.b confirms it's
inheritance-dominated.

---

## 9. Implementation notes

- Reuse `c2_subspace_overlap.py` unmodified for V1/V2 (just point `--droped-projectors` at the
  untrained-dropped NPZ for V2; V1 needs no rerun, just re-tabulate RS2's existing CSV).
- V3 needs a new function: orthogonalize a basis against a reference basis (project out the
  reference's row-space, re-orthonormalize the remainder via the existing `_orthonormalize_rows`
  pattern), then reuse `principal_angles`/`random_rotation_baseline` on the residuals. Suggest
  adding this as `--residual-against-pre` flag on the existing script rather than a new file, to
  keep the gate/baseline machinery shared.
- All input paths are pinned in §2.1; no new pod, no new spend.
