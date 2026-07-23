# RS1 — Scaffold-removal invariance (the DroPE before/after)

**Dated:** 2026-07-23
**Status:** pre-registered, not yet run
**Program:** [`rope-as-scaffold`](README.md) — tests claim **C1** (primary) and **C2** (secondary).
**Relationship to prior work:** this is the one piece the novelty check flags as a genuinely new
*result* rather than a new instance of a known principle — it converts the DroPE connection from
*citation* to *mechanistic evidence*. See [`references/novelty-check.md`](references/novelty-check.md).

---

## 0. Question

DroPE (Sakana, 2025) showed *empirically* that RoPE can be dropped after pretraining and briefly
recalibrated into a NoPE model that keeps perplexity and improves length generalization — but
supplied no interpretability account of *why* removing RoPE is safe. Our k-address-space findings
propose the mechanism: RoPE's position is (E1) reconstructable as emergent key-position and (E2)
not used as a query-readable retrieval address. RS1 tests that mechanism directly by running the
DroPE transition on one model and probing it **before vs after** with the M1.5 and M1.6 instruments.

**Primary (C1):** after the RoPE→drop→recalibrate transition, does the model's **emergent
key-position fill in** (so the model isn't position-blind), while the **addressing profile stays
unchanged** (still non-addressable)? I.e., is what RoPE supplied redundant and non-load-bearing?

**Secondary (C2):** does the recalibrated model's emergent positional subspace **reconstruct the
subspace RoPE supplied** (overlap/alignment with the pre-drop `k_post` positional directions)?

---

## 1. Why this instrument

- We already have **validated before-state baselines**: Qwen3-0.6B M1.5 (position profiles) and
  M1.6 v1.1 (`k_pre` addressing verdict) are measured and in `../k-address-space/`. The DroPE'd
  checkpoint is the only new artifact; the probes are reused unchanged.
- Qwen3-0.6B is the **cleanest DroPE subject** in our set: full RoPE (θ=1e6, all dims), so
  "drop RoPE" = remove all rotation → a pure NoPE(+QK-norm) model, with no partially-rotated
  ambiguity (unlike Pythia's 25% RoPE). The M1.5/M1.6 harness already supports it.
- The transition is *causal by construction*: we change exactly one thing (the positional
  mechanism) and read the mechanistic consequences off the same instruments used pre-drop.

---

## 2. Design

### 2.1 Subject and states

Primary subject: **Qwen3-0.6B** (base). Three probed states on a **single held-out eval set**:

1. **RoPE** — the unmodified checkpoint (before-state; M1.5/M1.6 baselines already exist, re-run
   on the eval set for a matched comparison).
2. **Dropped** — RoPE rotation replaced by identity at inference, **no recalibration** (the
   "scaffold yanked, not yet healed" waypoint). Inference-only, near-free.
3. **DroPE'd** — RoPE removed *and* the model recalibrated per the DroPE recipe (below), then
   re-probed. This is the load-bearing state.

Plus one **native-NoPE reference profile** (not part of the transition — a static comparison
target, M1.5 only):

4. **Native NoPE** — a model that learned NoPE from scratch, to answer *"did the DroPE'd model
   become a proper NoPE, or a partial one leaning on residual RoPE-era circuitry?"* Primary:
   **`andrewdalpino/NoPE-GPT-400M-Base`** (20L, hidden 1280, GQA 20Q/5KV, head-dim 64, ctx 8192,
   `nope_gpt` family — same hook path as the existing `nope-gpt-small`, so harness reuse is a new
   model tag `nope-gpt-400m`, not new probe code). ~400M is the same size *class* as Qwen3-0.6B.
   Optional robustness reference: **`starmpcc/NoPE_1.5B_FW_EDU_15T`** (Llama-3 arch, 15T tokens —
   the "Behind RoPE" checkpoint), better-trained but larger. Neither is a controlled match (see
   §7); they bracket the reference, and the comparison is of **depth-profile shape**, not magnitude.

### 2.2 DroPE recipe (state 3)

- Remove the rotary transform from all attention layers (keep QK-norm; the result is a
  NoPE+QK-norm model).
- **Recalibrate** on a generic corpus (e.g. FineWeb-Edu / C4 / OpenWebText subset) at the model's
  original context length, for a **small token budget** in the DroPE range (target ~1–2B tokens,
  ≤ a few % of pretraining-scale; exact count fixed at build time, bounded and recorded).
- Standard LM objective, cosine LR decay, a single seed. Record every hyperparameter in the
  manifest. No architecture change beyond the rotation removal.

Note: this is the program's **first experiment that requires actual training** (a real, if light,
GPU commitment — see §7), unlike the inference-only k-address-space runs.

### 2.3 Staging

- **RS1a (cheap, first):** states 1 + 2 only — inference-only. De-risks RS1b: confirms the drop
  actually breaks the model and shows how much emergent position already exists pre-recalibration.
- **RS1b (the result):** state 3 — recalibrate, then probe. Commit only after RS1a's gates pass.

---

## 3. Extraction / probes (reused, unchanged)

Run the existing harness on each state; no new probe code, so the before/after comparison is
apples-to-apples.

- **Perplexity** on the fixed held-out eval set (all three states).
- **M1.5** (`kaddress.scripts.position_content`): per-layer key-position decodability (ridge CV
  R², null-corrected), position fraction, PCA-to-90%, position-removal projector fidelity. Reuse
  gates G1 (architectural zero at L0), G2 (architectural one), G4 (one-sided shuffled null).
- **M1.6** (`kaddress.scripts.m16_discriminator`): causal K/V patching, G6 marker neutrality,
  G7 noise-controlled attention, output-above-noise addressing criterion. **RoPE-state patches use
  `k_pre`** (pre-rotation, per addendum-M1.6 v1.1 §4.1); dropped/DroPE'd states have no rotation,
  so the single K is patched directly.
- **C2 analysis (secondary):** subspace overlap (CCA / principal-angle alignment) between the
  RoPE state's `k_post` positional directions and the DroPE'd state's emergent key-position
  directions, per layer.
- **Native-NoPE reference (state 4):** M1.5 only, via a new model tag `nope-gpt-400m` reusing the
  existing `nope-gpt` hook path (`_capture_nope_k` on `qkv_proj`) — no new probe code. Inference-only.

---

## 4. Gates (each must be able to fail; verify by perturbation)

- **G-RS1.1 — rotation actually removed.** In the dropped/DroPE'd forward, the rotary transform is
  identity (verify cos=1/sin=0 or equivalent; perturb once to confirm the check can fail).
- **G-RS1.2 — the drop did something / recalibration did something.** Perplexity(dropped) ≫
  perplexity(RoPE) **and** perplexity(DroPE'd) ≪ perplexity(dropped). If dropping RoPE barely
  moves perplexity, the intervention is vacuous and the whole comparison is void; if recalibration
  doesn't recover it, that is a real (recorded) outcome, not a gate pass. This gate fails loudly on
  a no-op drop.
- Reuse **M1.5 G1/G2/G4** and **M1.6 G6/G7** unchanged; a state that cannot instantiate them
  (e.g. G6 marker search fails) invalidates that state's probe, as in the parent experiments.

---

## 5. Pre-registered predictions

- **(P.RS1.a) DroPE replicates at 0.6B.** Recalibration recovers held-out perplexity to near the
  RoPE baseline (state 3 ≈ state 1 ≫ state 2). *Falsifier:* perplexity does not recover → DroPE
  doesn't hold at this scale / recipe is off (scope result, report it).
- **(P.RS1.b) Emergent position fills in (C1 / E1).** The DroPE'd model's emergent key-position
  (M1.5, null-corrected ridge R² by depth) is **present and ≥ the RoPE model's `k_pre` emergent
  position** — i.e. removing the stamp does not leave the model position-blind; it reconstructs
  position. *Falsifier:* emergent position is absent in the DroPE'd model **yet** perplexity
  recovers → position was even more dispensable than "redundant" (a different, still-interesting
  result).
- **(P.RS1.b′) Became a proper NoPE (native-NoPE contrast).** The DroPE'd model's emergent
  key-position **depth-profile shape** matches the native-NoPE reference (characteristic
  architectural-zero at L0 → rising with depth → plateau), not merely exceeds its own before-state.
  *Falsifier:* the DroPE'd profile sits **well below** the native-NoPE reference (weak/absent
  emergent position) → the recalibration produced a *partial* NoPE **leaning on residual RoPE-era
  circuitry**, not a genuine reconstruction. (Read as *shape*, given the size/training confounds in §7.)
- **(P.RS1.c) Addressing profile unchanged (C1 / E2).** The DroPE'd model shows **no more
  query-readable addressing** than the RoPE model (both null under M1.6). *Falsifier:* addressing
  **appears or disappears** across the transition → the positional component *was* doing something
  causal for retrieval, contradicting E2.
- **(P.RS1.d) Subspace reconstruction (C2).** The DroPE'd emergent positional subspace
  substantially overlaps the RoPE `k_post` positional subspace (principal-angle alignment above a
  random-rotation baseline). *Falsifier:* disjoint subspaces → emergent position is a *different*
  code, not a reconstruction of what RoPE supplied.

---

## 6. Decision tree

- **P.RS1.a + b + c hold** → direct mechanistic support: *RoPE is a removable scaffold — its
  position is redundant with emergent computation and was not a load-bearing retrieval address.*
  This is the tech-note result; C2 (P.RS1.d) upgrades it from "position fills in" to "position
  reconstructs the same code."
- **a holds, b fails (perplexity recovers, no emergent position)** → position is *unnecessary*,
  not merely *redundant*; the model routes around it. Reframe the thesis (stronger claim about
  dispensability), still publishable.
- **b holds but b′ fails (position fills in, but below the native-NoPE shape)** → the DroPE'd model
  is a *partial* NoPE running partly on residual RoPE-era circuitry rather than a full
  reconstruction → the "scaffold cleanly removed" claim is qualified; the matched-pair gold
  standard (§7) becomes the needed follow-up.
- **a fails (perplexity does not recover)** → DroPE does not replicate at 0.6B under our recipe →
  debug recipe (token budget, LR, corpus) or report a scope limit; do not over-interpret b/c on a
  broken model.
- **c fails (addressing changes)** → E2 is wrong: removing/relearning position altered a causal
  retrieval mechanism. This would be the most surprising outcome and reopens the address question
  at the training level.

Any outcome is a reportable result; the point is to make the DroPE mechanism *measurable*.

---

## 7. Threats to validity

- **Recalibration corpus ≠ original pretraining distribution.** Perplexity is compared on a fixed
  *held-out generic* set across all three states, not on Qwen3's original mix — so the claim is
  "recovers on held-out generic data," not "recovers original loss." State this.
- **QK-norm remains.** Dropped Qwen3 is NoPE+QK-norm, not vanilla NoPE. Legitimate NoPE variant;
  record it and avoid over-generalizing to all NoPE.
- **Single model / single seed / 0.6B.** Scale- and architecture-qualified, as in the parent
  program. Pythia-410m (partial RoPE) is an optional robustness subject; a from-scratch small
  RoPE→DroPE model (matched data, full control of the before-state) is the stronger but heavier
  validation, deferred.
- **The native-NoPE reference is not a controlled match.** NoPE-GPT-400M is ~400M vs Qwen3's 600M
  (same class, not exact) and is *lightly trained* (FineWeb/SmolTalk/UltraFeedback) versus Qwen3's
  heavy training, and uses a different tokenizer/data — so a profile *difference* could reflect
  training budget or capacity, not the RoPE→dropped history. Mitigations: (i) compare **shape**,
  not magnitude; (ii) bracket with the better-trained-but-larger `NoPE_1.5B` reference; (iii) the
  only fully-controlled version is the deferred gold standard — a **matched-pair** RoPE+NoPE
  trained from scratch on identical data, DroPE the RoPE one. Treat b′ as suggestive, not decisive.
- **Compute step-up.** This is the first program experiment needing real training; a bad recipe
  can masquerade as a null (P.RS1.a falsifier + G-RS1.2 guard against silently accepting that).
- **Reused-probe assumption.** M1.5/M1.6 are validated on the RoPE/NoPE states already; the only
  novel measurement surface is the recalibrated checkpoint, which keeps the comparison clean.

---

## 8. Schedule & budget

- **RS1a** (states 1–2, inference + M1.5/M1.6 re-run on eval set): hours; reuses harness; ~free
  beyond a GPU session. Gate G-RS1.1/G-RS1.2 checked here before committing to training.
- **RS1b** (recalibration ~1–2B tokens on 0.6B + re-probe): the cost — a **real, light training
  run**, single GPU, order **1–3 days** rented; est. **~$50–150 GPU** depending on hardware. This
  is materially above the k-address-space "< $5" runs and is the honest price of turning the DroPE
  bridge into a result.
- **Native-NoPE reference** (state 4, M1.5 on `nope-gpt-400m`; optional `NoPE_1.5B`): inference-only,
  hours, ~free — one added model tag, no training.
- **Analysis** (M1.5/M1.6/C2 before-vs-after + native-NoPE shape contrast): CPU/GPU, reuses harness.

Follows the repo `reproducible-research` lifecycle: committed spec + pre-run notebook entry (a new
`experiments/rope-as-scaffold/NOTEBOOK.md`), pre-run commit, published recalibrated-checkpoint +
outputs as external artifacts with checksums, completed notebook. **Do not run before RS1a's gates
pass.**

## 9. Deliverable

A three-state (RoPE / dropped / DroPE'd) before-after table on perplexity, M1.5 emergent-position
profiles, M1.6 addressing verdict, and C2 subspace overlap — adjudicating P.RS1.a–d — i.e. the
first *mechanistic* account of why DroPE's scaffold removal is safe, or a falsification of it.
