# K-address-space — consolidated findings (M1, M1.5, M1.6)

**Dated:** 2026-07-23. Curated synthesis across the whole experiment group. Companion to
[`CLOSING-NOTE.md`](CLOSING-NOTE.md). Primary records with checksums and release URLs are in
[`NOTEBOOK.md`](NOTEBOOK.md); pre-registrations in [`spec.md`](spec.md),
[`addendum-M1.5.md`](addendum-M1.5.md), [`addendum-M1.6.md`](addendum-M1.6.md).

**Central question (spec §0):** do cached key vectors act as **content addresses** — same-referent
mentions clustering in K, position acting as a namespace, ‖V‖ as version dominance — or is
K-geometry dominated by position and surface form? **Answer reached: K is *not* a query-readable
address at the scales tested.** Retrieval is content-addressed (Hopfield-style); the positional
component of K is decodable and attention-relevant but **causally inert for the output**.

Models: `gpt2` (124M, learned absolute), `pythia-410m` (partial RoPE, θ=1e4), `Qwen3-0.6B`
(full RoPE, θ=1e6, QK-norm), `NoPE-GPT-Small` (no positional encoding).

---

## M1 — address purity (write test) — RETRACTED

All four models returned **0 address heads** (GPT-2 0/144, Pythia 0/768, Qwen3 0/448, NoPE
0/384; best AUCs ~0.50–0.68). **These are instrument artifacts, not facts about the models** —
corpus defect **F8**: in the Track A generator the disambiguating detail is emitted *after* the
shared-alias mention and rotates per round, so referent identity is causally unavailable at the
probed token → **zero valid address-purity trials**. The previously reported Pythia "whisper
heads" are withdrawn. Extraction/RoPE-reconstruction machinery was verified correct; only the
corpus was void. M1/M2/M3 are dead until a corpus v3 (disambiguators precede mentions), which
was **not built** — see [`NOTEBOOK.md`](NOTEBOOK.md) "Known corpus defect F8" and `spec.md` §8.

---

## M1.5 — positional content of K (repeated-segment probe) — the real results

Method: identical-segment repetition (R ≥ 128), hold content constant so any decodable position
is *computed*, not inferred. Per (model, layer, head): position fraction, ridge CV R²,
PCA-to-90%, position-removal projector fidelity. Gates G1 (architectural zero), G2 (architectural
one), G4 (one-sided shuffled null), all perturbation-falsifiable.

- **NoPE** — architectural zero at L0 (ridge R² 0, verified < 1e-6), then position becomes
  strongly decodable with depth: **R² 0 → ~0.77 (L8) → ~0.97 (L23)**, position fraction low
  throughout (→ ~0.09). Low-dimensional (~1–2 PCs mid-stack). Computed position, not stamped.
- **GPT-2** (stamped, learned absolute) — position decodable essentially **perfectly from L0**
  (R² ≈ 1.0, flat), position **fraction declines** with depth (~0.72 → ~0.43). ~3 PCs. It
  *dilutes* a front-loaded stamp.
- **Pythia** (partial RoPE) — **P1.5.c confirmed:** `k_pre` is architectural-zero at L0 yet
  develops decodable position at depth (**R² 0 → ~0.93 by L1 → ~0.96 late**, null-corrected
  `r2_minus_null ≈ +0.95–1.0`), at low variance share (peak ~0.083). `k_post` position fraction
  **declines** with depth (0.38 → 0.07), R² ≈ 1.
- **Qwen3** (full RoPE, θ=1e6) — **P1.5.c confirmed** in `k_pre` (R² → ~0.97). **Key divergence:**
  `k_post` position fraction **rises** with depth (0.24 → ~0.5 plateau) while R² ≈ 1 — the
  *opposite* of Pythia/GPT-2, which dilute. Full RoPE **accumulates** position into the cached
  key with depth. **P1.5.e supported at L0** (qwen3 `k_post` 0.24 < pythia 0.38 — weakest stamped
  fraction, as θ=1e6 predicts; the depth-averaged comparison reverses it and is the wrong metric).

**Cross-model:**
- **Decodability ≠ salience dissociation** (both RoPE-free `k_pre` and NoPE): position is highly
  decodable (R² ~0.96) yet occupies a tiny, low-dimensional slice of the key.
- **Aggregate-vs-slot metric caveat:** P1.5.d ("computed position is higher-dimensional") is **not
  supported at the slot level** (~2–3 PCs; the "~13 PCs" figure is aggregate-projector-only). The
  Π position-removal projector (P1.5.f) is only cleanly effective at mid-depth per-slot; the
  aggregate projector is a different, stronger operator. Any downstream use must pick one basis.
- **Headline novel point: P1.5.c** — RoPE-*stamped* models still compute emergent position into
  the **pre-rotation** key stream at depth. This cuts against the textbook view that the unrotated
  representation is pure content; a targeted novelty sweep found it apparently unclaimed (see
  [`../rope-as-scaffold/references/novelty-check.md`](../rope-as-scaffold/references/novelty-check.md)).

---

## M1.6 — hypothesis discriminator (causal patching) — the address verdict

Discriminates **addressing** (K's position is a dialable coordinate Q reads) vs **anti-collision**
(position is inert cargo that only keeps identical keys apart) vs **transitive induction**. Method:
donor→target K/V patch at a probed repetition + norm-matched-noise control; addressing requires
attention redirect **and** output-follow, both **above noise**. v1.1 fixes: R restored to 128
(C1), noise control on attention via gate G7 (C2), mandatory transitivity (C3); RoPE models
patch **`k_pre`** pre-rotation so donor *content* is re-addressed to the target position.

- **NoPE (R=128):** **0 addressing heads.** Output-following null across all 384 heads (max
  donor-marker shift +0.010 over ~0.036 baseline). Only 4/384 pass G7 and all are output-null.
  The v0 (R=4) "attention redirection up to +0.43" **collapsed to +0.05 max at R=128** — it was a
  low-R artifact. Transitivity uninformative (head-independent readout).
- **Qwen3 (full RoPE, `k_pre` patch):** **no robust addressing** — 2/448 heads pass, both fragile
  (fire on a single stimulus, the highest-baseline one → stimulus confound). Output max ~5e-4 on a
  ~5e-4 baseline. **But Qwen3 is more attention-steerable than NoPE** (content-specific K-patch
  redirection up to **+0.17**, 9 heads > 0.10, vs NoPE's ~+0.05) — a real **RoPE > NoPE
  steerability gradient** that does *not* convert to output-addressing. Transitivity uninformative
  (altered-marker rank constant = 4 across all stimuli and heads).

**Verdict:** across the NoPE → full-RoPE gradient, **K is not a query-readable retrieval address.**
Position in K is decodable and (in RoPE) attention-steerable, but **causally sterile for the
output**. The tape/coordinate-address framing is retired; content-addressed retrieval
(Hopfield-style) is the operative mechanism, with the positional component as cargo. Among the
"cargo" accounts, **anti-collision/decorrelator is the leading survivor** (consistent with
steerable-but-inert), **induction unadjudicated** (the transitivity instrument is model-level and
underpowered at n=4 stimuli — an instrument limit, not a mechanism result).

**Methodological caveats (carry into any successor):** the M1.6 negatives are instances of the
established **decodable ≠ causally used** principle (Dyck brackets, arXiv 2604.22128); a *null*
patch effect is confounded between "unused" and "used-but-redundant" — the per-(layer,head) patch
does not rule out **distributed/redundant** addressing (no all-heads-at-slot ablation was run).
Mitigations present: counterfactual donor steering + norm-matched noise control.

---

## Prediction adjudication

| prediction | status |
|---|---|
| **P5.a** address heads exist | **invalid (F8)** — retracted |
| **P5.b** survive diff-surface | **invalid (F8)** — retracted |
| **P5.c** RoPE namespace dose-response | **invalid via M1/M3 (F8)**; partial position-dosage from M1.5 (effective-rotation fraction collapsed to ~0/0.16/0.41, not 0/0.25/1.0) |
| **P5.d** ‖W_O v‖ grows with update index | **not tested** (M4 not run) |
| **P5.e** read is latest-wins | **not tested** (M4 not run) |
| **P5.f** QK-norm tightens addresses | **partial** (Qwen3 QK-norm characterised in M1.5; no clean within-referent test) |
| **P5.g** n_eff is a data property | **not tested** (M7 not run) |
| **P1.5.a** gates hold | **pass** |
| **P1.5.b** NoPE position rises with depth | **confirmed** |
| **P1.5.c** stamped models also compute position | **confirmed** (Pythia, Qwen3) — headline |
| **P1.5.d** computed position higher-dimensional | **not supported at slot level** (aggregate-only) |
| **P1.5.e** Qwen3 weakest stamped fraction | **supported at L0** |
| **P1.5.f** projector removes position | **scope-dependent** (slot mid-depth yes; aggregate stricter) |
| **P1.6.a** G6 marker neutrality | **pass** |
| **P1.6.c** patch-K addressing | **no** (NoPE and Qwen3) — headline |
| **P1.6.d** induction present | present, **not confirmed as the output mechanism** |
| **P1.6.e** transitivity decides | **unadjudicated** (instrument uninformative) |

## Limitations

Scale ceiling 0.6B (absence here ≠ absence at 7B+). Synthetic repeated-segment stimuli. M1.6 is
single-model-per-architecture (NoPE + Qwen3). The transitivity instrument is weak. M1 corpus is
void (F8) and was not rebuilt.

## What carries forward

Two salvageable, forward-looking results seed the successor program
[`../rope-as-scaffold/`](../rope-as-scaffold/): **P1.5.c** (emergent pre-rotation position in
stamped models) and the **M1.6 causal method + verdict** (position decodable/steerable but not an
address; RoPE > NoPE steerability). See [`CLOSING-NOTE.md`](CLOSING-NOTE.md).
