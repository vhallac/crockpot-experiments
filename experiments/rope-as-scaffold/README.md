# RoPE-as-Scaffold

**Status:** program charter (2026-07-23). This directory defines a *directed* research
program with a falsifiable claim set. It is not yet a pre-registered experiment; per-experiment
`spec.md` files and runs follow the repo's `reproducible-research` lifecycle once claims are
scoped.

## What this program tests

Whether **RoPE's positional contribution is a removable training scaffold** — i.e. whether the
position RoPE supplies is, after training, (a) reconstructable as *emergent* key-position and
(b) *not* used as a causal retrieval address, so that removing RoPE removes a training-time
convenience rather than a load-bearing inference mechanism. This is the mechanistic question
sitting underneath [DroPE](references/literature-survey.md) (Sakana, 2025), which showed
*empirically* that RoPE can be dropped post-training (keeping perplexity, improving length
generalization) but justified it only with optimization theory, not interpretability evidence.

## Why we are doing this — the motivating findings

This program exists because two results from
[`experiments/k-address-space`](../k-address-space/) pointed straight at it:

- **Emergent redundancy (M1.5, P1.5.c).** RoPE-*stamped* models (Pythia-410m, Qwen3-0.6B)
  **still compute position into the pre-rotation key stream (`k_pre`) with depth** — an
  architectural zero at layer 0 rising to ridge R² ≈ 0.96 late, at low variance share — even
  though RoPE already hands them position for free. RoPE's stamp is *partly redundant* with
  emergent computation. (`../k-address-space/NOTEBOOK.md`, M1.5 v1.1 Pythia/Qwen3 entries.)
- **Non-addressability (M1.6, causal patching).** Across NoPE-GPT-Small and full-RoPE
  Qwen3-0.6B, K's position/content is decodable and *attention-steerable* but **not a
  query-readable retrieval address**. NoPE is flat (content-specific K-patch attention ≈ +0.05
  max); Qwen3 is more steerable (up to +0.17, 9/448 heads > 0.10) yet still yields **no robust
  output-addressing** (2/448 fragile heads, single-stimulus, stimulus-confounded).
  (`../k-address-space/NOTEBOOK.md`, M1.6 NoPE v1.1 and Qwen3 v1.1 entries; `addendum-M1.6.md`.)

Together: RoPE gives a **stronger positional handle** than NoPE's emergent position, but that
handle is **not load-bearing as a retrieval address**. That is the mechanistic seed of "why
RoPE is a droppable scaffold," and the reason for this program.

## Central thesis

> After training, RoPE's positional contribution decomposes into (i) a positional signal the
> model can and does reconstruct without it (emergent key-position) and (ii) a stronger local
> attention handle that is **not** used as a causal retrieval address. Removing RoPE therefore
> costs local-order/recency acuity and training convenience, **not** a retrieval mechanism —
> which is why post-hoc removal (DroPE) preserves capability and helps length generalization.

## Directed claim set (prove / disprove)

**Established (motivating, from k-address-space — carried in as priors, not re-tested here):**
- **E1 — emergent redundancy.** RoPE models recompute key-position at depth (M1.5 P1.5.c).
- **E2 — non-addressability.** K-position is steerable but not a retrieval address, in both
  NoPE and full-RoPE (M1.6).

**To test here (each falsifiable; the point of the program):**
- **C1 — scaffold-removal invariance (first experiment).** Apply the DroPE recipe (pretrain
  RoPE → drop PE → brief recalibration) to a small model, then run the M1.5/M1.6 probes
  **before vs after**. *Predict:* emergent `k_pre` position fills in to match the pre-drop
  decodability, and the (non)addressing profile is unchanged. *Falsifier:* removal loses a
  retrieval/addressing capability that emergent position does not recover.
- **C2 — subspace equivalence.** The dropped model's emergent position reconstructs the *same*
  positional information/subspace RoPE supplied (measure overlap/alignment). *Falsifier:*
  disjoint subspaces — emergent position ≠ RoPE-supplied position.
- **C3 — functional locus of RoPE.** What actually degrades on removal is **local-order /
  recency acuity and length-extrapolation behavior**, not retrieval/addressing — i.e. RoPE's
  causal contribution is local, not address. *Falsifier:* removal degrades retrieval more than
  local-order tasks. (Sharpens [RNoPE-SWA](references/literature-survey.md)'s correlational
  "RoPE = local, NoPE = retrieval" into a causal claim.)
- **C4 — scale check.** E1/E2 hold beyond 0.6B (spot-check one larger model). *Falsifier:*
  a genuine retrieval-address emerges at scale, or emergent redundancy vanishes.

## Relation to prior work

The method (drop RoPE post-training) is **not novel** — DroPE owns it. An exhaustive novelty
sweep (see [`references/novelty-check.md`](references/novelty-check.md)) further found that the
program's *conceptual pillars are largely pre-existing*: "attention ≈ content-addressable
memory" (Hopfield/Ramsauer 2020), the tape/delta-write framing (fast-weight programmers,
Schlag/Irie 2021), and — critically — the epistemic core "**decodable ≠ causally used**" (already
demonstrated for other latents in [2604.22128](https://arxiv.org/pdf/2604.22128)). So M1.6 is a
*domain instance* of a known principle, not a new insight. **What remains genuinely open is
narrow:** the specific causal measurement (position-in-K is non-addressable across the
NoPE→full-RoPE gradient) and — the only piece that would be a new *result* rather than a new
instance of an old idea — the **DroPE before/after (RS1)**. Consequence: RS1 is load-bearing;
without it, reconsider whether the note is worth writing. Full landscape in
[`references/literature-survey.md`](references/literature-survey.md).

## Planned experiments

| id | claim | sketch | cost |
|---|---|---|---|
| **RS1** ([spec](RS1-spec.md)) | C1 (+C2) | DroPE recipe on Qwen3-0.6B + before/after M1.5/M1.6 probes (the "M1.7" increment) | needs light training — step up from k-address-space's inference-only runs |
| RS2 | C2 | emergent-vs-RoPE positional subspace overlap on the RS1 checkpoints | analysis-only |
| RS3 | C3 | task/perplexity ablations along a local-order axis vs a retrieval axis, RoPE vs dropped | eval harness |
| RS4 | C4 | E1/E2 spot-check on one >0.6B model | GPU |

RS1 is the load-bearing one: it converts the DroPE connection from *citation* to *result*.
Each experiment gets its own `spec.md` (pre-registration) before running.

### Candidate extensions (unscoped)

- **RoPE as a training warmup for NoPE (proposed).** Instead of dropping RoPE from a fully
  RoPE-pretrained model (RS1), use RoPE only as an *early scaffold* for an otherwise-NoPE run:
  add RoPE for the unstable early phase (per DroPE's vanishing-gradient rationale), drop it early,
  continue as NoPE. Question: *how little RoPE, and only when, is needed to get a good NoPE model?*
  **Caveat — overlaps DroPE:** the basic "RoPE → train → de-RoPE beats from-scratch NoPE" loop is
  essentially DroPE's own claim #2, so only the **minimal/early-warmup** refinement is potentially
  novel. Mechanistic hook: P1.5.c suggests RoPE bootstraps the emergent-position circuitry, which —
  once formed — may make RoPE dispensable early. **Requires from-scratch training** (a real step up
  from RS1's recalibration-from-pretrained) and a **novelty check** before scoping. Not started.
  **Condition to pursue:** only if RS1 confirms the clean scaffold thesis (`P.RS1.a`+`b`+`c` hold,
  ideally `b'`). Its hook *requires* RS1 to show emergent position genuinely fills in and
  reconstructs; if RS1 instead shows position is *unnecessary* (b fails), *partially* reconstructed
  / RoPE-residue (b' fails), *causal for retrieval* (c fails), or that DroPE doesn't even replicate
  (a fails), this extension is respectively mis-framed, premature, moot, or blocked — do not spend
  from-scratch-training budget on it.

### Gating — RS1 is the program's decision point

The expensive downstream work is conditioned on which RS1 branch lands:

- **RS1 confirms (a+b+c, ideally b'):** scaffold thesis holds → RS4 (scale-generalize) and the
  RoPE-warmup extension become well-motivated; RS2/RS3 are cheap analysis continuations on the RS1
  checkpoints.
- **a holds, b fails** (perplexity recovers, position *not* reconstructed): reframe to a
  *dispensability* claim; the warmup hook is void → drop/rescope it. RS4 still relevant.
- **b' fails** (partial NoPE / RoPE residue): removal isn't clean → the matched-pair from-scratch
  gold standard is the priority, not the warmup extension.
- **c fails** (addressing changes across the transition): E2 is wrong → stop the extensions and
  rethink the thesis before spending more.
- **a fails** (DroPE doesn't replicate at 0.6B): premise broken at this scale → only recipe-debug
  or a larger-model retry make sense; all extensions premature.

Nothing beyond RS1 (and the cheap RS2/RS3 analyses) should consume real training budget until RS1
picks the branch.

## Result policy

Follows repo conventions (`AGENTS.md`, `.pi/skills/reproducible-research`): committed spec +
pre-run notebook entry, raw outputs published externally (not in git), completed notebook with
provenance. `references/` holds only small, curated survey/novelty documents.

## References

- Motivating findings: [`../k-address-space/`](../k-address-space/) — `NOTEBOOK.md` (M1.5/M1.6
  entries), `addendum-M1.5.md`, `addendum-M1.6.md`.
- [`references/literature-survey.md`](references/literature-survey.md) — DroPE and the RoPE↔NoPE line.
- [`references/novelty-check.md`](references/novelty-check.md) — gap analysis and the 2505.13027 verdict.
