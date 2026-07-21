# ADDENDUM §5-M1.5 — Positional Content of K (Repeated-Segment Probe)

**Dated:** 2026-07-21
**Status:** pre-registered, not yet run
**Parent:** SPEC §5 (K-Space as Address Space); supersedes nothing, unblocks M1/M2
**Models:** `gpt2` (124M, 12L×12H×64) · `EleutherAI/pythia-410m` (24L×16H×64, rotary 16/64, θ=1e4) ·
`Qwen/Qwen3-0.6B` (28L×8KV×128, QK-RMSNorm, θ=1e6) · `andrewdalpino/NoPE-GPT-Small-Base` (24L×16H×64, no PE)
**Budget:** < $5. One forward pass per stimulus; feasible on CPU.

---

## 0. Why this exists

The 2026-07-17..20 runs established two things, neither of them about models:

1. **Track A contains zero valid address-purity trials (F8).** Referent labels for the
   shared-alias mentions are causally unavailable at the token where the key is computed;
   name mentions are lexically confounded. Four models returned four nulls on the
   surface-controlled referent test (FWER p: gpt2 0.417 · pythia 0.967 · qwen3 0.133 ·
   nope 0.717). The instrument, not the hypothesis, produced those numbers.
2. **The label-free position probe is confounded by the same corpus.** Absolute position is
   predictable from `(doc_id, update_idx)` alone at **R² = 0.945**. Any key encoding document
   and round predicts position without any positional encoding being involved. The depth
   profile therefore measured "keys encode context," not "position enters K."

Exactly one measurement survived, because layer 0 has had no attention and so no context to
encode — **fraction of the layer-0 key that is position** (within-token residual scale ÷ raw scale):

| model / variant | position fraction @ L0 |
|---|---|
| gpt2 (learned absolute) | 0.649 |
| pythia `k_post` (16/64 dims rotated, θ=1e4) | 0.385 |
| qwen3 `k_post` (128/128 dims rotated, θ=1e6) | 0.193 |
| pythia `k_pre`, qwen3 `k_pre`, nope | ~1e-7 (zero) |

Two facts fall out of that table and motivate everything below.
**(i)** Nominal rotation fraction is the wrong axis — Qwen3 rotates every dimension and stamps
position *least*, because θ=1e6 makes per-position angles tiny. Empirically recovered from the
release vectors: qwen3 slowest-dim θ = 1.241e-06 = 1e6^(−126/128); pythia 3.162e-04 = 1e4^(−14/16).
Effective fraction of head dims rotated past 90° at a typical Δp≈400: **gpt2 0.00 · pythia 0.16 · qwen3 0.41**.
**(ii)** NoPE's layer-0 keys are *bitwise identical* across all 240 occurrences of a token
(max pairwise L2 = 0.00000). That is a verified architectural zero — the calibration standard
the trio never had.

M1.5 extends the surviving measurement to all depths by removing the confound at the stimulus
level, and returns a transform that M1 needs.

---

## 1. This is not a NoPE test

The probe reads position out of K. What that *means* differs by column:

| | `k_pre` @ L0 | `k_pre` @ depth | `k_post` |
|---|---|---|---|
| **nope** | 0 by architecture | **computed** position | n/a |
| **pythia / qwen3** | 0 (verified ~1e-7) | **leaked/computed** position | stamped + computed |
| **gpt2** | already 0.649 | contaminated from L0 | n/a |

The middle column is the object of study. Pythia and Qwen3 `k_pre` at layer ℓ>0 is not
position-free: the residual stream feeding `k_proj` has passed through ℓ attention layers that
attended with rotated keys, so position has leaked back into content. GPT-2 cannot participate
in the depth analysis — its keys carry absolute position from layer 0, leaving no baseline
against which to measure an increment. It remains in the run as a positive control.

The distinction that matters downstream:

- **Stamped position** (RoPE, absolute) is content-independent, uniform across tokens, and
  **invertible** by a known transform. This is what makes "sieve pre-rotation, re-rotate
  survivors" (§5 Exp 3) viable.
- **Computed position** is content-dependent, built by attention, shares a subspace with
  content, and has **no inverse**. If a model computes position into K, address and namespace
  are the same bits, and the §5 decision tree's branch-1 consequences do not follow.

Calibration on novelty: *that* NoPE encodes position implicitly is established (causal masking
suffices; it is why NoPE works at all). Confirming it is replication. Unestablished, and the
target here: the **key-level, depth-resolved, capacity-quantified** version — how many of d_head
dimensions position consumes, at which layer, and how much room it leaves for content.

---

## 2. Stimulus design

**Principle:** hold content constant so that any position decodable from K is necessarily
*computed by the model* rather than *inferred from varying context*.

### Family A — identical repetition (primary)

One segment S repeated R times, no separator variation:

```
[Alice is a successful engineer.] × R
 slot: 0    1  2      3        4
```

- **R ≥ 120** required (ridge on 64 dims); R ≥ 200 preferred for qwen3 (128 dims).
- **Shared-ceiling run** (all four models, cross-model comparable): R chosen so total ≤ 950
  GPT-2 tokens. With a 7-token segment, R = 135.
- **Extended runs** (within-model depth resolution): pythia → 2048 ctx, R ≈ 290;
  qwen3 → 8k ctx, R ≈ 1100. NoPE-GPT context ceiling to be read from its config, not assumed
  (the architecture permits variable block size; the *trained* block size is the limit that matters).
- **≥ 8 distinct segments** as replicates. A single stimulus yields single-stimulus artifacts.
- **Segment lengths ∈ {4, 7, 12} tokens**, so that repetition index and absolute token position
  can be separated (same token position, different repetition count, and vice versa).
- **Every slot probed separately** (slot 0 … L−1), not just one word. Tells us whether the
  position code is token-dependent or token-invariant.

### Family B — frame-constant, content-varying (induction control)

Same syntactic frame, different content words per repetition
(`{Name} is a {adj} {profession}.`). Probe the *frame* tokens only (`is`, `a`, `.`), which are
token-identical across repetitions. Content varies, so this does not isolate computed position
as cleanly as A — its purpose is solely to test whether Family A's result survives outside the
degenerate induction regime (see §7.1).

### Family C — natural recurrence (external validity)

Natural prose in which a frequent function word (`the`, `,`) recurs ≥ 120 times. Probe those
occurrences. Fully confounded in the Track A sense; included only to check that A's magnitudes
are not an artifact of synthetic text.

**Adjudication order: A is primary. B and C corroborate or caveat; they never overturn A.**

---

## 3. Extraction

Reuses the §5 hook code unchanged. Per (layer, head, variant ∈ {pre, post}), per stimulus,
record the key vector at every probed slot occurrence, plus:

- `repetition_index` r ∈ [0, R), `slot_index` s ∈ [0, L), `token_pos` = r·L + s, `token_id`
- residual-stream norm at that position (massive-activation flag, 5× layer median)
- attention mass received by position 0 from the final query (sink stats — repeated text is a
  known trigger for anomalous sink behaviour; record, do not analyse here)

**Mandatory gates before analysis** (each must be able to fail; verify by perturbation):

- **G1 — architectural zero.** NoPE @ L0, pythia `k_pre` @ L0, qwen3 `k_pre` @ L0: within-slot
  variance across repetitions must be < 1e-5 relative to raw scale. Any nonzero value is an
  extraction bug, not a finding.
- **G2 — architectural one.** gpt2 @ L0 and `k_post` @ L0 must give R² ≥ 0.9. If not, the
  probe itself is broken.
- **G3 — RoPE reconstruction.** As in §5-M1, retained.
- **G4 — shuffled-y null.** Permuted targets must give R² ≈ 0 (observed −0.006 in prior runs).

---

## 4. Measurements

Let X be the (R × d_head) matrix of keys for one (layer, head, variant, slot), y the repetition
index. Let `raw = mean|X|` and `resid = mean|X − X̄|` where X̄ is the across-repetition mean.

- **M1.5.1 — Position fraction.** `resid / raw`. Direct extension of the layer-0 table in §0 to
  all depths. Interpretable as "what fraction of this key is not the token."
- **M1.5.2 — Decodability.** 5-fold CV R² of ridge y ← X, alphas ∈ logspace(−2, 4).
  **Variance floor guard: if `resid/raw` < 1e-5, report R² = 0 and do not fit.** (See §7.2.)
- **M1.5.3 — Position capacity.** PCA on (X − X̄); number of components to reach 90% of residual
  variance, and the fraction of *total* key variance those components carry. **This is the
  measurement that converts "position as namespace" into a number:** if position occupies 3 of
  64 dimensions, 61 remain for content.
- **M1.5.4 — Code geometry.** Is the leading position PC (a) monotone in r → absolute-like;
  (b) periodic with period L → local/relative; (c) log-monotone → compressive? Report Spearman ρ
  vs r, and the dominant Fourier component of PC1 vs slot phase.
- **M1.5.5 — Leakage curve (the headline).** M1.5.1 and M1.5.3 as a function of depth, for
  `k_pre` in pythia/qwen3 (stamped models) against nope (unstamped). All three pinned to zero at
  layer 0 by G1, so the curve shape is the entire result.
- **M1.5.6 — Removability.** Project out the top-k position PCs (k from M1.5.3). Report
  (a) residual position R² after projection; (b) retained token-identity decodability
  (multiclass CV accuracy, token id from key) before vs after. A transform that removes position
  and destroys content is useless; both numbers are required.

---

## 5. Pre-registered predictions

- **(P1.5.a)** G1 holds in all three architectural-zero cases; G2 holds. *If either fails, stop
  and debug — no downstream number is interpretable.*
- **(P1.5.b)** NoPE's position fraction and R² rise monotonically with depth, reaching
  substantial decodability (R² > 0.5) by mid-stack. Replication of known implicit-position
  results, at the key level; primarily a validity check on the probe.
- **(P1.5.c)** *Genuinely uncertain, and the reason to run all four.* Pythia/qwen3 `k_pre`
  position fraction also rises with depth — i.e. models compute position into keys **even when
  it is already stamped for free**. If confirmed, implicit position is not a NoPE quirk but a
  default consequence of causal attention, and every published K-space geometry analysis is more
  position-contaminated than assumed. If instead the stamped models stay flat while NoPE rises,
  the two mechanisms are substitutes and position is built only on demand. **Either outcome is a
  result; the second is the more interesting one and is not a NoPE finding at all.**
- **(P1.5.d)** Position occupies a low-dimensional subspace (M1.5.3 ≤ 0.15·d_head to 90%
  variance) in the stamped models, and a *higher*-dimensional, more entangled one in NoPE —
  because computed position must share capacity with content whereas stamped position does not.
- **(P1.5.e)** Qwen3 `k_post` shows the weakest stamped position fraction of the RoPE models at
  matched relative depth, consistent with θ=1e6 and the measured L0 table (0.193 vs 0.385).
- **(P1.5.f)** M1.5.6 succeeds for stamped position (position removed, token identity retained)
  and *fails* for computed position in NoPE at depth (removing position also degrades token-identity
  decodability). This is the operational form of the stamped/computed distinction in §1.

---

## 6. Decision tree

- **P1.5.c: stamped models rise** → position and content are entangled in `k_pre` at depth in
  *every* architecture. Consequence for §5: the §5-Exp3 plan ("sieve pre-rotation") loses its
  guarantee, because `k_pre` is not the position-free object it was assumed to be; the sieve must
  operate on the position-orthogonal complement from M1.5.6 instead. Also promotes this addendum
  from methodology to a standalone finding.
- **P1.5.c: stamped models flat** → `k_pre` is a clean content space in RoPE models. §5-Exp3
  proceeds as originally specified; NoPE is architecturally distinct and must be analysed with
  its own position-removal transform.
- **P1.5.f succeeds broadly** → ship Π (position-removal projector) into the M1 re-run. This is
  the proper representation-level fix for F3, replacing the pair-sampling workaround: compute
  address purity on ΠK rather than trying to match controls on distance.
- **P1.5.f fails in NoPE only** → NoPE is excluded from address-purity analysis (its keys cannot
  be decontaminated), and the M1 re-run proceeds on the other three.
- **P1.5.d inverted** (stamped models more entangled than NoPE) → the capacity argument is wrong
  and "position as namespace" should be dropped from the tape framing rather than defended.

---

## 7. Known traps, recorded before the fact

### 7.1 Induction regime
Repeated identical text strongly drives induction heads; the model is not in a typical operating
regime. Family A's causal cleanliness is bought at this cost. **Mitigation:** Family B and C must
be reported alongside A in every table; if A and B disagree in *sign*, the finding is an induction
artifact and must be reported as such. Record sink/massive-activation stats (§3) so the regime is
documented rather than assumed benign.

### 7.2 Variance-floor amplification — an actual bug from this weekend
The first pass of this probe reported layer-0 position R² of 0.20 (pythia `k_pre`) and 0.23 (nope).
Both were false. The true within-token residual is ~1e-7 of raw scale; standardizing by a std of
1e-7 amplified pure floating-point rounding to unit scale, after which ridge fit per-document
rounding patterns that happen to correlate with position. **Same failure class as the L0
degenerate-geometry incident: normalizing near-zero variance manufactures signal.** M1.5.2's guard
is mandatory, and the T4 degenerate trap must be extended to cover regression probes, not only AUC.

### 7.3 Counting is not a confound
The model can count preceding copies. Counting via causal attention **is** implicit positional
encoding — it is the mechanism under study, not an alternative explanation. What Family A excludes
is *position inferable from varying content*, which is precisely the Track A defect (R² = 0.945).

### 7.4 Post-selection inference
Do not report "the best head" without a family-wise null. A z ≈ 4 selected as best-of-384 is not
significant; this error was made and retracted on 2026-07-20. Use the max-statistic null across
heads within each shuffle, as in the corrected M1 analysis.

### 7.5 Scale ceiling
0.6B maximum. Absence of an effect here is not absence at 7B. Every claim scale-qualified.

### 7.6 Cross-model comparability
Tokenizers differ, so segment token-length differs. Report against **repetition index** for
cross-model tables; record raw token position for within-model analysis. Report per relative
depth (ℓ / n_layers), never raw layer index, when comparing 12/24/28-layer models.

---

## 8. Deliverable

Beyond the report: **Π, a per-(model, layer, head) position-removal projector**, with its
M1.5.6 fidelity numbers attached. This is the artifact that unblocks the M1 re-run — address
purity measured on ΠK is a representation-level control for position, replacing the
distance-matched-pair workaround (F3) which can only ever approximate it.

Note the sequencing consequence: **M1.5 is not a detour from the address experiment; it produces
the transform that experiment requires.** It is also fully independent of corpus v3 (F8), so it
can run before the corpus is fixed.

---

## 9. Schedule & budget

- **Step 1** (~1 h, no GPU): stimulus generators for Families A/B/C; length sweep; slot indexing;
  gate harness G1–G4 with deliberate perturbation checks.
- **Step 2** (~20 min GPU, or CPU with patience): shared-ceiling extraction, 4 models ×
  8 segments × 3 lengths × 3 families. Single forward per stimulus, ≤ 950 tokens. Trivial cost.
- **Step 3** (~30 min, CPU): M1.5.1–M1.5.6, plots, `REPORT-M1.5.md` with P1.5.a–f adjudicated in
  the §2–§4 verdict style.
- **Step 4** (optional, ~20 min GPU): extended-context runs for pythia (2048) and qwen3 (8k),
  re-running M1.5.5 on the depth axis with better sample counts.

Estimated total: **< $5**, dominated by instance spin-up rather than compute.
