# ADDENDUM §5-M1.6 v1.1 — Hypothesis Discriminator: Addressing vs Anti-Collision vs Transitive Induction

**Dated:** 2026-07-23
**Supersedes:** v1.0 (pre-run commit `6f45f91`, retained unmodified in git history —
`git show 6f45f91:experiments/k-address-space/addendum-M1.6.md`). This is a corrections
revision, not a rewrite of the predictions: all P1.6.* in §5 are unchanged. §10 records the
adjudication after the first (v1.0-design) NoPE run.
**Status:** pre-registered; v1.0 run once on NoPE-GPT-Small (2026-07-23), superseded by the
v1.1 re-run (pending).
**Parent:** builds on M1.5's confirmed result (position is ~1-2 dimensions, ~94% decodable
from K by mid-depth in NoPE-GPT-Small). Does not modify M1.5; M1.5's gates and stimuli are
reused where possible.
**Numbering note:** this is M1.6, not M1.5-extended. §5-M1.7 is reserved, unassigned, for
future extensions.
**Budget:** < $5. Reuses M1.5 extraction hooks; adds one patched forward pass and one
attention-readout pass per stimulus. CPU-feasible.

---

## CHANGELOG v1.0 → v1.1

Three design defects surfaced by the first NoPE run (`nope-gpt-small`, 2026-07-23) and its
review. All are spec-level; v1.0's predictions are unchanged.

| # | Defect | Level | Fix |
|---|---|---|---|
| **C1** | **Marker design caps R (§2.1).** "Each repetition ends in a *distinct* single-token continuation marker" bounds R by the neutral-marker vocabulary size; the run got **R = 4**, versus the M1.5 regime (R = 128–248) whose signal M1.6 is meant to probe. Whether the M1.5 position signal even exists at R=4 is unestablished, so the run could not bear on M1.5. | **SPEC** | §2.1 places distinct markers **only at the probed repetitions** (target r\*, donor r′, altered-interior, readout); all other repetitions are marker-free. R_min ≥ 128 is restored by reusing M1.5 Family A generation. The per-repetition-distinct requirement is dropped. |
| **C2** | **Noise control was applied to output only, not attention (§3/§4.1).** The addressing call keyed on K-patch *attention* redirection, but the norm-matched-noise control was compared only against the output/donor-probability readout. Noise redirects attention as much as a donor K-patch (run: noise +0.179 vs donor +0.192), so the attention criterion was uncontrolled — producing a phantom 33-head "addressing" bucket. | **SPEC** | §3/§4.1 make the norm-matched-noise control mandatory on **both** the attention-redirection and the output readouts; extraction records a noise-attention delta (new gate G7), and the addressing criterion must exceed it. |
| **C3** | **Addressing was callable on attention alone; transitivity was skipped (§4.1/§4.2.3).** The §4.1 key let attention-redirect-alone read as addressing, and §4.2.3 transitivity — the sharp discriminator — was treated as optional and not run, so induction vs anti-collision stayed unadjudicated. | **SPEC** | §4.1 requires addressing = attention redirects **above noise** AND output follows **above noise**; attention-moves/output-null is explicitly *not* addressing. §4.2.3 transitivity becomes **mandatory** and is the deciding measurement whenever patch-K output is null. |

---

## 0. Why this exists

M1.5 established that K contains a compact, highly decodable, monotone-in-repetition signal.
It did **not** establish what that signal is *for*. Three mechanisms predict the identical
K-geometry M1.5 measured, and are therefore indistinguishable by anything M1.5 can measure:

- **Addressing.** Q carries a matching position-like coordinate and dials it to select a
  specific repetition, symmetric with content-based retrieval.
- **Anti-collision.** The position component in K is inert cargo. Its only function is to
  keep otherwise bit-identical keys (as in Family A's exact repetition) from collapsing
  softmax into a uniform blend. Q never reads it as a coordinate; nothing downstream is built
  to interpret its value.
- **Transitive induction.** Selection never touches a position coordinate at all. It runs
  through the standard induction-head circuit: a previous-token head records "the token before
  me was X" at each position, and a later query matches its own preceding token against that
  record, attending to whatever came **after** the most recent match. What M1.5 decodes as
  "position" may be a side effect of accumulated match history, not a signal anything uses as
  an address.

All three predict: low-rank, depth-increasing, monotone-in-r, ~94%-decodable K geometry.
None of M1.5's measurements (position fraction, ridge R², PCA dimensionality, projector
fidelity) can tell them apart, because all three are stated about K alone. M1.6 moves from
characterizing K's geometry to testing what Q does with it, via two complementary
instruments: **causal patching** (does forcing a version change what gets selected/copied?)
and **induction scoring** (does attention follow the specific content-chained rule that
induction predicts, rather than a free-standing coordinate or nothing at all?).

---

## 1. Relationship to prior threads

This is the causal version of two things already in the program:

- **§5-M1 M4 (version dominance).** M4 asked, observationally, whether later |V| in an update
  chain correlates with being read (latest-wins vs averaging). M1.6's patching test asks the
  same question causally: force a specific repetition's K and/or V into the "current" slot and
  read off whether selection follows.
- **§5 tape hypothesis, "accumulation vs hand-over."** If patching K redirects attention
  cleanly, that is direct evidence for addressing-style accumulation (any slot is individually
  retrievable). If attention is patch-insensitive but content-driven, that favours hand-over or
  induction-style local computation over a genuine address space.

---

## 2. Stimulus requirements

Reuses M1.5 Family A generation (§5-M1.5 v1.1 §2.1), with one mandatory addition.

### 2.1 Continuation-divergence marker — REVISED v1.1 (fixes C1)

**v1.1 change:** distinct markers are placed **only at the probed repetitions** — the target
r\*, the donor r′, the altered-interior repetition (§4.2.3), and the final readout — not on
every repetition. All other repetitions are left marker-free (a shared neutral continuation or
none), so the segment can be repeated at the **M1.5 scale (R_min ≥ 128)** using M1.5 Family A
generation directly. v1.0 required a distinct marker on *every* repetition, which capped R at
the neutral-marker vocabulary size (the first run got R = 4) and severed the link to the M1.5
regime this test is meant to probe.

Only the probed repetitions need a checkable behavioral signature: the divergence markers exist
so selection has a next-token signature there, rather than only an attention-weight signature.

```
"Alice is a successful engineer today."
"Alice is a successful engineer again."
"Alice is a successful engineer still."
...
```

- Marker vocabulary for the probed slots: a small set of single-token (verify per tokenizer),
  semantically interchangeable words — enough distinct markers for {r\*, r′, altered-interior,
  readout}, not for all R. Next-token probability across the markers actually used must be
  unbiased absent any selection mechanism (that is G6). Candidates: `{today, again, still, now,
  once, indeed, truly, ...}`, filtered to single-token per model. Because only a handful of
  markers are needed, per-stimulus neutral-marker selection is cheap — search for a G6-passing
  set before patching rather than reusing one fixed set across stimuli.
- Record the continuation-token id per repetition in the stimulus metadata.
- The probed readout is the model's next-token distribution at the final query position
  (after the last full repetition, at the point where the next word would be predicted), or,
  for interior manipulations (§4.2), at a synthetic query inserted immediately after the
  candidate repetition's slot.

### 2.2 Donor/target design

For the patching experiment (§4.1), designate:
- **target repetition r\*** — the one whose cache entries will be overwritten. Use an
  interior repetition, not the last (so "most recent" isn't confounded with "patched").
- **donor repetition r′** — a different repetition, ideally several positions away from r\*,
  whose K/V will be transplanted in.

For the induction-score experiment (§4.2), no patching is needed; all R repetitions are used
as-is, plus one **altered-interior** variant per stimulus in which a single interior
repetition's continuation is swapped to a marker word used nowhere else, enabling the
transitivity test in §4.2.3.

---

## 3. Extraction

Reuses M1.5 hooks. Two additional captures:

- **Query vectors.** At the readout position, capture Q per (layer, head), pre- and
  post-RoPE where applicable (not needed for NoPE but keep the harness general for the
  eventual Pythia/Qwen3 runs).
- **Full attention weight matrix** at the readout position, over all cached positions —
  not just the aggregate statistics M1.5 recorded, but the raw per-key weight, so induction
  score and patched-vs-unpatched attention deltas can be computed exactly.
- **Noise-patch attention delta (REVISED v1.1, fixes C2).** For every head, record the
  target-attention delta induced by the norm-matched-noise control patch, not only its effect
  on the output/donor probability. The addressing criterion (§4.1) is defined against this
  noise-attention baseline; without it, "K-patch redirects attention" is uncontrolled, because
  overwriting a key with *any* vector — donor or noise — perturbs the softmax. In the v1.0 run,
  noise moved target attention +0.179 vs the donor's +0.192 (nearly equal), so the effect was
  generic perturbation, not content-specific selection.

**Gate G6 — marker neutrality (new).** Before any patching, verify that in an **unpatched**
stimulus, next-token probability across the R continuation words is roughly uniform
(max/min ratio < 3, say) when no other manipulation has been applied. If the model already
strongly prefers one continuation word irrespective of position, the marker vocabulary is
biased and must be replaced — this gate must be able to fail, and should be tested against a
deliberately biased vocabulary (e.g. reusing "again" for two different repetitions) to confirm
it can.

**Gate G7 — noise-controlled attention (new, v1.1, fixes C2).** A head may be considered for an
addressing call only if its donor K-patch target-attention delta exceeds its noise-patch
target-attention delta by a pre-set margin. This is the attention-side analogue of the noise
control the addendum already required on the output side. The gate must be able to fail, and
does: in the v1.0 run almost no head clears it (donor +0.192 ≈ noise +0.179 for the bucket that
v1.0 mislabelled "addressing"). Heads that fail G7 are perturbation-sensitive, not address-like,
regardless of how large their raw K-patch attention delta is.

---

## 4. Measurements

### 4.1 Causal patching (tests addressing vs. everything else)

For each stimulus, target r\*, donor r′, at every (layer, head) or at a selected subset
(start with the heads M1.5 flagged as carrying the position signal, e.g. those with
`ridge_r2 > 0.9` at that depth):

**Patch stage for RoPE models (REVISED v1.1, C2/C3 clarification).** For rotary models
(pythia/qwen3) the K-patch is applied to `k_pre` (**pre-rotation**), and RoPE is then applied so
the transplanted donor content is re-addressed to r\*'s position. Patching `k_post` (the literal
cached, already-rotated key) transplants the donor's *own position rotation* along with its
content, confounding content with absolute position and making the addressing readout
uninterpretable — a null could be a position mismatch rather than absent addressing. NoPE has no
rotation, so its single K is patched directly. A separate rotation-only swap (target content,
donor rotation) may be used to test position-as-coordinate, but the primary content-addressing
test uses the pre-rotation patch.

- **Patch-K.** Overwrite the cached K vectors at r\*'s slot(s) with r′'s K vectors (V
  untouched). Re-run the forward from that point; record (a) attention weight at the readout
  position onto r\*'s slot before vs. after, (b) next-token probability shift toward r′'s
  continuation word.
- **Patch-V.** Overwrite V only, K untouched. Record the same two readouts.
- **Patch-both.** Full transplant of r′ into r\*'s slot. Record the same two readouts, plus
  whether behavior matches patch-K, patch-V, or neither alone (superadditivity check).
- **Control patch.** Overwrite r\*'s K/V with **freshly sampled noise matched in norm** to the
  original (not a donor repetition). This distinguishes "patching does something because any
  perturbation disrupts attention" from "patching does something because r′'s specific content
  was selected."

**Interpretation key — REVISED v1.1 (fixes C3).** "Redirects" below means **above the noise-patch
attention baseline** (G7); "output follows" means the donor-marker probability shift exceeds the
noise-patch output baseline. **Attention redirection alone is not addressing** — overwriting K
with any vector perturbs the softmax, so the addressing row requires *both* effects above noise.

| Patch-K attention (vs noise) | Output (vs noise) | Reading |
|---|---|---|
| redirects to r\* above noise | follows toward r′ above noise | **addressing**: K's position-like value is a dialable coordinate Q can be steered by — the only cell that supports the tape/address framing |
| redirects above noise | **null** (≤ noise) | **not addressing**: K content shapes attention weights but is not read into the output — anti-collision / inert-leaning; the position code is decodable and even attention-relevant but causally sterile for retrieval (**this is the v1.0 NoPE outcome**) |
| no redirect above noise | follows above noise | content-driven selection, K position inert — anti-collision or induction (need §4.2, esp. §4.2.3, to split) |
| no redirect above noise | null | neither K nor V position content is causally load-bearing — pure ballast, or selection routes elsewhere in the residual |
| donor patch ≈ noise patch (attention *or* output) | — | generic perturbation sensitivity, not selection of r′ — **discard that head's patch findings**, confounded |

### 4.2 Induction score (tests transitive induction vs. addressing/anti-collision)

#### 4.2.1 Standard induction score
At the readout query (preceding token t), compute attention weight over all earlier positions
whose preceding token also equals t (i.e., every prior occurrence of the same slot across
repetitions). Induction predicts concentration specifically at **match-position + 1** (the
token that followed the most recent occurrence of the bigram ending in t), not spread
uniformly across matches and not weighted by simple recency alone.

Report, per (layer, head): fraction of attention mass on (match+1) positions vs. mass on other
match positions vs. mass elsewhere. A clean induction head shows most mass concentrated at
match+1 for the **most recent** match specifically.

#### 4.2.2 Recency vs. match-recency
Distinguish "attention favours whatever is closest" (a positional recency bias, consistent
with anti-collision-flavoured decay) from "attention favours the most recent *match*, wherever
it is" (induction proper). If repetitions have variable spacing (possible once §5-M1.5 v1.1's
segment-length sweep is in place), these two predictions can be pulled apart: a pure recency
account degrades with absolute distance regardless of match structure; an induction account
tracks the match position exactly regardless of how far back it sits.

#### 4.2.3 Transitivity test (the sharp discriminator) — MANDATORY in v1.1 (fixes C3)

**v1.1 change:** this test is required, not optional, and is the deciding measurement whenever
the patch-K output effect (§4.1) is null — which is the v1.0 NoPE outcome. It is the only
measurement that cleanly separates transitive induction from anti-collision. A v1.1 run that
omits it does not adjudicate the mechanism and is incomplete.

In the altered-interior stimulus (§2.2): swap one interior repetition's continuation to a
unique marker word, and construct the sequence so this altered repetition is the most recent
match at the point where a synthetic query with matching preceding-token context is inserted.

- **Induction predicts:** the query's top prediction is the altered repetition's marker word —
  selection transitively follows "whatever came after the last match," and can be steered by
  changing what an *interior*, non-final repetition was followed by.
- **Addressing predicts:** prediction is insensitive to the interior alteration unless the
  query's position-coordinate is specifically set to address that repetition — i.e., altering
  content elsewhere shouldn't change what an address-based read returns unless the address
  itself changed.
- **Anti-collision predicts:** no systematic tracking either way; whichever continuation gets
  predicted is governed by whatever generic recency/frequency bias exists, indifferent to the
  controlled alteration.

---

## 5. Pre-registered predictions

- **(P1.6.a)** G6 holds: unpatched continuation words are near-uniform in next-token
  probability. *(If not, fix the marker vocabulary before trusting anything below.)*
- **(P1.6.b)** Noise-control patches produce measurably less attention/output disruption than
  donor patches at matched norm — i.e., the patching effect (if any) is content-specific, not
  generic perturbation sensitivity.
- **(P1.6.c)** *Open, and the crux of the whole addendum.* At the heads M1.5 flagged as
  carrying strong position decodability, does patch-K redirect attention? **Yes → addressing**
  is live. **No → anti-collision or induction**, adjudicated by §4.2.
- **(P1.6.d)** Induction concentration (match+1) is present and substantial in at least some
  heads, given that Family A is a maximally repetitive, induction-favouring regime — this is
  the safest prediction in the set, since induction heads are well-established in this exact
  regime.
- **(P1.6.e)** The transitivity test (§4.2.3) is the deciding measurement if P1.6.c comes back
  negative: transitive tracking of the altered interior repetition would confirm induction as
  the operative mechanism, distinct from anti-collision, which predicts no such tracking.

---

## 6. Decision tree

- **Patch-K redirects attention (P1.6.c yes)** → addressing is real at this scale, at least in
  the flagged heads. This reopens the §5 tape/namespace framing on firmer footing: K's position
  component is not just decodable, it is *usable* by Q. Next step: check whether the same heads
  also show clean induction structure (§4.2) — addressing and induction are not mutually
  exclusive; a head could implement induction *by* using a position coordinate rather than
  pure content-matching.
- **Patch-K null, patch-V drives output, induction score strong, transitivity confirmed** →
  **transitive induction**, not addressing. The position signal M1.5 decoded from K is a
  correlate of accumulated induction bookkeeping, not a retrieval coordinate. The "tape as
  address space" framing should be retired in favour of "tape as content-chained pointer
  structure" — a materially different and more standard (and more mechanistically grounded)
  claim.
- **Patch-K null, patch-V null, induction score weak, transitivity fails** → **anti-collision**
  survives as the leading account: the position code exists, is decodable, but is causally
  inert for selection in this regime. This would be the least exciting outcome scientifically
  but is a legitimate and useful negative result — it would mean M1.5's entire depth-profile
  finding is a story about softmax degeneracy avoidance, not about retrieval or addressing at
  all, and the tape metaphor should be scoped down accordingly.
- **Mixed across heads/layers** (most likely outcome) → report a per-head/layer taxonomy, as
  in the accumulation/hand-over split from earlier: some heads address, some induct, some
  carry inert position ballast. This is itself the finding, and the natural next step is
  relating the taxonomy to depth (the M1.5 two-regime split at layer ~15 is a candidate
  boundary to check against this classification).

---

## 7. Known traps

- **Patching confound (§3, G6, §4.1 noise control).** Any overwrite disrupts the forward pass
  somewhat; without the noise-matched control, "patching changed something" is not evidence
  for content-specific selection. This control is mandatory, not optional.
- **Marker vocabulary bias.** If continuation words differ in base rate frequency, next-token
  probability will be biased regardless of any selection mechanism, confounding both the
  patching readout and the transitivity test. G6 exists specifically for this.
- **Induction is expected here, not surprising.** Family A is about as induction-favourable a
  regime as exists (exact repeated bigrams, high repetition count). Finding induction heads is
  not itself a finding; the finding is whether induction alone accounts for the M1.5 K-geometry
  or whether addressing/anti-collision components remain after induction is factored out.
  Don't over-claim P1.6.d as novel — its purpose is to establish the baseline against which
  P1.6.c and P1.6.e are read.
- **Single-model scope.** Like M1.5's first run, this is NoPE-GPT-Small only until re-run
  across the trio. Induction-head strength and prevalence are known to vary with scale and
  training; absence of a clean pattern here doesn't generalize.
- **Layer selection for patching (§4.1).** Restricting the first pass to M1.5-flagged
  high-decodability heads is a reasonable cost-saving choice but risks missing addressing
  behavior in heads M1.5 didn't flag (e.g., low position-fraction but high functional use).
  If P1.6.c comes back uniformly negative, widen the head set before concluding anti-collision
  or induction won.

---

## 8. Deliverable

A per-(layer, head) three-way classification — **addressing / anti-collision / induction /
mixed / inert** — with the patching and induction-score evidence attached per head. This
either upgrades the "tape" framing to a causally-grounded addressing claim, replaces it with
the better-established induction-head account, or scopes it down to a softmax-degeneracy
patch. Any of the three is a real, reportable result.

## 9. Schedule & budget

- **Step 1** (~30 min): extend Family A generation with continuation markers; verify marker
  neutrality (G6) before proceeding.
- **Step 2** (~30 min, CPU): patching passes (K/V/both/noise-control) at the M1.5-flagged
  heads, for a handful of stimuli × target/donor pairs.
- **Step 3** (~30 min, CPU): induction-score and transitivity-test passes across all heads.
- **Step 4** (~30 min): classification table, `REPORT-M1.6.md` with P1.6.a–e adjudicated.

Estimated total: **< $5**, CPU-feasible throughout, no new model downloads.

---

## 10. Adjudication after run 1 (v1.0 design, `nope-gpt-small`, 2026-07-23)

Run under the v1.0 design (R = 4, single G6-valid stimulus, transitivity not run). Full results
and the corrected reading are in the lab notebook (M1.6 NoPE entry, 2026-07-23). Summary:

| prediction | status | evidence |
|---|---|---|
| **P1.6.a** G6 holds | **PASS** | marker max/min ratio 1.21 < 3 on the valid prefix |
| **P1.6.b** noise < donor disruption | **FAILS on attention** (C2) | noise-patch target-attention +0.179 ≈ donor +0.192; output readout was at noise for the flagged heads |
| **P1.6.c** patch-K redirects attention → addressing | **NO — addressing not supported** | output-following null across all 384 heads (max donor-marker shift +0.010 over a ~0.036 baseline). ~25 late-layer heads (L17–22) show content-specific attention redirection (e.g. L19H7 K +0.43 vs noise −0.07), but it does not propagate to output → attention-moves/output-null = anti-collision/inert, not addressing |
| **P1.6.d** induction present | **PRESENT** (expected, not a finding) | match+1 mass high, ~0.58 mean among flagged heads, top near 1.0 |
| **P1.6.e** transitivity decides if P1.6.c negative | **NOT RUN** | §4.2.3 skipped under v1.0; induction vs anti-collision therefore unadjudicated |

**Verdict:** weak evidence **against** causal addressing at this scale/regime; the tape-as-address
framing is weakened. **Superseded by the v1.1 re-run** — probed-only markers at R ≥ 128 (C1),
multi-stimulus with per-stimulus G6 marker search, noise-controlled attention via G7 (C2), and
mandatory §4.2.3 transitivity (C3) — pending.
