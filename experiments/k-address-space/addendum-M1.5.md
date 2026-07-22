# ADDENDUM §5-M1.5 v1.1 — Positional Content of K (Repeated-Segment Probe)

**Dated:** 2026-07-21
**Supersedes:** `addendum-M1.5.md` (v1.0, 2026-07-21) — **v1.0 is retained unmodified as the
pre-registration record.** This document is a corrections revision, not a rewrite of the
predictions. All P1.5.* predictions are unchanged in content; §10 records their adjudication
status after the NoPE run.

---

## CHANGELOG v1.0 → v1.1

Four defects found by the first run (`nope-gpt-small`, 2026-07-21). Two are spec-level, two
are implementation-level but were enabled by spec silence.

| # | Defect | Level | Fix |
|---|---|---|---|
| **C1** | **Budget arithmetic was internally inconsistent.** v1.0 §2 required R ≥ 120 *and* segment lengths up to 12 tokens *and* max_length ≤ 950. 12 × 120 = 1440 > 950. The length sweep was arithmetically impossible as written, so it was never attempted. | **SPEC** | §2.0 now defines R_min as a function of d_head, derives a per-model budget, and requires a build-time feasibility assertion plus a published feasibility matrix. |
| **C2** | **Family B alignment was under-specified.** v1.0 said to "probe the frame tokens, which are token-identical across repetitions," which invited a fixed-slot-index implementation. The implementation therefore required *every repetition to tokenize to the same total length* — impossible with real vocabulary (`Farid`, `Greta` are 2 tokens in this tokenizer; `Alice`…`Hector` are 1). All 8 cyclic offsets failed identically; Family B yielded zero stimuli and the induction control never ran. | **SPEC** | §2.2 now mandates **offset-based slot identity**: a slot is *(frame word, occurrence index within repetition)*, located by decoded piece, with per-repetition absolute positions. Equal token length across repetitions is explicitly **forbidden** as a requirement. |
| **C3** | **Silent stimulus rejection.** Every rejection path was a bare `continue`. A whole family yielding nothing was invisible in the manifest; it took forensics to discover. Five of eight Family A segments were also dropped this way (8-token segments → 950//8 = 118 < 120). | impl (spec silent) | §3 adds **G5 family-yield gate** and a mandatory `rejected_stimuli` manifest field with reasons. |
| **C4** | **Null gate mis-specified and variance floor under-scoped.** (a) The shuffled-null check tested `|R²| > 0.05` two-sided; negative CV R² is the *expected* null behaviour at n ≈ 2·d, so 4,759 of 4,773 "breaches" were negative (min −0.39) against only 14 positive (max +0.072) — a false alarm that flagged a healthy run. (b) The variance floor correctly zeroed `ridge_r2` at layer 0 but not `pca_components_90pct`, which reported 9.125 — a PCA of 1e-6 floating-point noise. | **SPEC** (both under-specified in v1.0 §4/§7.2) | §4 floor now applies to **every** downstream statistic; §7.5 null gate is **one-sided**, and adjudication moves to the pooled estimator. |

---

## 0. Why this exists

Unchanged from v1.0 §0. Briefly: Track A contains zero valid address-purity trials (F8), and
its label-free position probe is confounded because position is predictable from
`(doc_id, update_idx)` at R² = 0.945. Only the layer-0 measurement survived:

| model / variant | position fraction @ L0 |
|---|---|
| gpt2 (learned absolute) | 0.649 |
| pythia `k_post` (16/64 dims rotated, θ=1e4) | 0.385 |
| qwen3 `k_post` (128/128 dims rotated, θ=1e6) | 0.193 |
| pythia `k_pre`, qwen3 `k_pre`, nope | ~1e-7 (zero) |

M1.5 extends this to all depths by removing the confound at the stimulus level, and returns
Π, the position-removal projector that the M1 re-run requires.

## 1. This is not a NoPE test

Unchanged from v1.0 §1. The probe reads position out of K; what that means differs by column
(`k_pre` @ L0 = architectural zero; `k_pre` @ depth = computed/leaked position; `k_post` =
stamped + computed). The stamped/computed distinction is the point: stamped position is
content-independent and invertible; computed position shares a subspace with content and has
no inverse.

---

## 2. Stimulus design

**Principle (unchanged):** hold content constant so that any position decodable from K is
necessarily *computed by the model* rather than *inferred from varying context*.

### 2.0 Budget arithmetic — NEW, fixes C1

Three quantities interact and v1.0 fixed all three independently. They must instead be derived:

- **R_min(d_head) = max(120, 2 · d_head)** — the ridge on d_head features needs n ≥ 2d for a
  stable CV R². The first run used n = 135 against d = 64 (2.1×), and its slot-level nulls
  showed the expected small-sample negative bias (mean −0.043) while pooled nulls were clean
  (−0.003 to −0.006). Hence also §4.0 below.
- **max_length(model) = trained_context − 32** (margin for BOS/edge effects). Read the
  *trained* block size from config; do not inherit another model's ceiling. NoPE-GPT-Small
  permits variable context, so its trained value must be read, not assumed to be 950.
- **Feasibility: R_min × L ≤ max_length** must hold for each (model, segment length L) cell.

**Build-time assertion, mandatory:** if a requested (model, L) cell is infeasible, the run
**fails loudly** with the arithmetic, rather than silently emitting fewer stimuli.

Resulting feasibility matrix (R_min from d_head; publish this table in the manifest):

| model | d_head | R_min | max_length | L=4 | L=7 | L=12 |
|---|---|---|---|---|---|---|
| gpt2 | 64 | 128 | 992 | ✅ 512 | ✅ 896 | ❌ 1536 |
| nope-gpt-small | 64 | 128 | *(read config)* | ✅ | ✅ | *(depends)* |
| pythia-410m | 64 | 128 | 2016 | ✅ 512 | ✅ 896 | ✅ 1536 |
| qwen3-0.6B | 128 | 256 | 4064 | ✅ 1024 | ✅ 1792 | ✅ 3072 |

- **L = 7 is the mandatory cross-model cell** — feasible everywhere, and the basis of all
  cross-model comparison.
- **At least two distinct L values are mandatory per model** (L ∈ {4, 7} at minimum).
  Rationale: with a single L, token position is `L·r + s`, perfectly collinear with repetition
  index, and **M1.5.4's absolute-vs-relative discrimination is unadjudicable.** The first run
  had L ∈ {6,7} only by accident and could not separate "counts tokens" from "counts repetitions."
- L = 12 runs where feasible; its absence on gpt2 is recorded, not worked around.

### 2.1 Family A — identical repetition (primary)

One segment S repeated R times, no separator variation. Unchanged from v1.0 except:

- **≥ 8 segments must *survive*, not merely be listed.** v1.0 said "≥ 8 distinct segments";
  the implementation listed 8 and shipped 5 because three exceeded the length ceiling. Author
  the segment pool against the per-model L targets, verify token length at build time, and
  regenerate or substitute until 8 survive at each L.
- Every slot 0…L−1 probed separately.

### 2.2 Family B — frame-constant, content-varying (induction control) — REWRITTEN, fixes C2

Same syntactic frame, different content words per repetition: `{Name} is a {adj} {profession}.`
Probe **only the frame tokens** (`is`, `a`, `.`), which carry a constant token id while their
surrounding content varies.

**Slot identity is offset-based, not index-based.** This is the load-bearing change:

- A slot is defined by the pair **(frame word, occurrence index within the repetition)** — e.g.
  ("is", 0), ("a", 0), (".", 0).
- Slots are located by encoding each repetition **independently**, decoding each token, and
  matching the stripped piece against the frame vocabulary.
- Absolute positions are computed from **actual cumulative offsets** of the concatenated
  repetitions, never from `r · L + s`.
- **Repetitions may differ in token length. Requiring equal length across repetitions is
  forbidden** — it is unsatisfiable with natural vocabulary and silently voids the family.
- **Assert** (do not filter): the token id at a given slot is constant across repetitions. If
  it is not, that is a stimulus-construction error to report, not a stimulus to drop.
- Record per repetition which content words were used, so the analysis can verify that content
  genuinely varies.

Because content varies, Family B does not isolate computed position as cleanly as A. **Its sole
purpose is to test whether Family A's result survives outside the degenerate induction regime.**

### 2.3 Family C — natural recurrence (external validity)

Unchanged. Natural prose, probe a function word recurring ≥ R_min times. Fully confounded by
design; corroboration only.

**Adjudication order (unchanged): A is primary. B and C corroborate or caveat; they never
overturn A. If A and B disagree in sign, the finding is an induction artifact and must be
reported as such.**

---

## 3. Extraction

Unchanged from v1.0 except the gates.

**Mandatory gates — each must be able to fail; verify by perturbation:**

- **G1 — architectural zero.** NoPE @ L0, pythia/qwen3 `k_pre` @ L0: within-slot variance
  across repetitions < 1e-5 relative to raw scale. *(Passed on the NoPE run at ~1e-6, with a
  perturbation reaching 2.5e-4. Retained as-is.)*
- **G2 — architectural one.** gpt2 @ L0 and `k_post` @ L0 must give R² ≥ 0.9.
- **G3 — RoPE reconstruction.** As in §5-M1.
- **G4 — shuffled-y null.** See §7.5 for the corrected one-sided form.
- **G5 — family yield. NEW, fixes C3.** Every requested family must produce ≥ 1 stimulus, and
  every requested (model, L) cell must produce ≥ 1 stimulus. Violation is a **run failure**,
  not a warning.
- **Manifest must carry `rejected_stimuli`:** one record per discarded candidate with
  `{stimulus_id, family, target_L, reason, token_len, max_reps_possible}`. Reasons are
  enumerated, not free text (`below_min_repetitions`, `slot_token_not_constant`,
  `frame_token_absent`, `exceeds_max_length`).

---

## 4. Measurements

Unchanged from v1.0 (M1.5.1–M1.5.6), with two corrections.

### 4.0 Estimator hierarchy — NEW, fixes C4(a) fallout

- **Pooled (aggregate) estimates are the adjudication basis.** Slot-level estimates at
  n ≈ 2·d_head carry a substantial negative CV bias and are reported as **descriptive only**.
  Evidence: on the NoPE run, slot-level nulls averaged −0.043 while pooled nulls sat at
  −0.003…−0.006 for the same data.
- Slot-level results remain in the CSV for diagnostics and for the M1.5.4 slot-dependence
  question, which needs them.

### 4.1 Variance floor — SCOPE WIDENED, fixes C4(b)

If `resid_mean_abs / raw_mean_abs < 1e-5`, the row is **degenerate** and **every** derived
statistic is set to 0 or NaN — `ridge_r2`, `pca_components_90pct`,
`pca_residual_variance_fraction_90pct`, `pc1_spearman_repetition`, `pc1_dominant_fourier_bin`,
`r2_after_position_pc_projection`. v1.0 gated only the regression, which let layer 0 report
`pca_components_90pct = 9.125` — a PCA of floating-point noise.

---

## 5–6. Predictions and decision tree

**Unchanged from v1.0.** P1.5.a–f stand as pre-registered; see §10 for adjudication so far.

---

## 7. Known traps

§7.1 (induction regime), §7.2 (variance-floor amplification), §7.3 (counting is not a
confound), §7.4 (post-selection inference), §7.6 (cross-model comparability) unchanged.

### 7.5 Null gate is one-sided — REVISED, fixes C4(a)

Cross-validated R² on a null model is **negatively** biased at small n; negative values are
evidence of correct behaviour, not leakage. The gate must be:

```
G4 fails iff  quantile(shuffled_r2, 0.99) > +0.05          # upper tail only
```

Preferred stronger form: adjudicate observed R² against the **shuffled distribution** rather
than against zero — report `r2_minus_null_mean` and a permutation p-value, exactly as the
corrected M1 analysis does for AUC. Record the negative tail as a diagnostic of sample
adequacy: a slot-level null mean below −0.10 means n is too small for that cell, and it should
be reported rather than silently trusted.

### 7.7 Cyclic offsets do not create independent stimuli — NEW

v1.0's Family B design generated "one stimulus per cyclic offset," which the implementation
read as 8 attempts. Because the offset only rotates the starting point of a full cycle, all
offsets contain the same multiset of sentences — 8 correlated stimuli, and 8 identical
failures. **Independent Family B stimuli must differ in vocabulary, not in phase.** Use
disjoint name/adjective/profession pools per stimulus.

---

## 8. Deliverable

Unchanged: **Π, a per-(model, layer, head) position-removal projector**, with M1.5.6 fidelity
attached. Confirmed viable by the NoPE run (§10).

## 9. Schedule & budget

Unchanged (< $5; CPU-feasible). Add: re-run `nope-gpt-small` alongside the other three, since
its Family B and length-sweep cells never executed.

---

## 10. Adjudication status after run 1 (`nope-gpt-small`, 2026-07-21)

| prediction | status | evidence |
|---|---|---|
| **P1.5.a** gates hold | **PASS** | G1 at ~1e-6 vs 1e-5 floor; perturbation can fail |
| **P1.5.b** NoPE position rises with depth | **CONFIRMED** | R² 0 → ~0.35 (L3) → ~0.88 (L8) → ~0.94 plateau; position fraction ~0.008 → ~0.089 |
| **P1.5.c** stamped models also compute position | **PENDING** | needs pythia/qwen3 `k_pre` |
| **P1.5.d** computed position is higher-dimensional | **PARTLY WRONG** | 1–2 PCs of 64 through L8–L14 — far more compact than predicted; expands to ~13 PCs only at L15–23 while R² stays flat. Revise: the prediction may hold only for the upper-third regime. |
| **P1.5.e** qwen3 `k_post` weakest stamped fraction | **PENDING** | needs qwen3 |
| **P1.5.f** projector fails for computed position | **REFUTED (usefully)** | L23 pooled R² 0.907 → 0.005 after projection; token-identity accuracy 0.984 → 0.996. Removing position *improves* content decodability. Π is viable → M1 re-run unblocked. |

**Unadjudicated by construction:** M1.5.4 absolute-vs-relative (single effective L → collinear;
fixed by §2.0), and the entire induction control (Family B yielded nothing; fixed by §2.2).

**Additional finding not predicted:** position is not head-specialised — by layer 10, 16 of 16
heads exceed R² 0.9. Worth carrying into the cross-model runs as a descriptive statistic.

**Source of truth.** The figures in this §10 table are a derived summary of the durable
run record in `NOTEBOOK.md` and the published CSV outputs, which are authoritative. On any
discrepancy, trust the notebook and the CSVs, not this summary. (The original v1.0-era draft
of this cell carried a NoPE Family-A position-fraction endpoint of `0.21`, a session-note slip
corrected here to the measured `~0.089`.)
