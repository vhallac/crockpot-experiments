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
- **Recalibrate** on **FineWeb-Edu `sample-10BT`** (`HuggingFaceFW/fineweb-edu`, open ODC-BY,
  streamable) — the corpus DroPE used *and* the data lineage of the NoPE-GPT-400M reference, so
  recipe, subject, and reference share a distribution. Take ~1–2B tokens of its ~10B for training
  (bounded, recorded), tokenize with **Qwen3's tokenizer**, pre-tokenize once to a mmap'd shard
  (**uint32**; Qwen3 vocab ~152k). Standard LM objective, cosine LR decay, single seed; record
  every hyperparameter. **Context length, LR, and the full recipe are pinned in [§10 Implementation
  Contract](#10-implementation-contract-execution-brief) — read it before implementing** (it
  supersedes any "original context length" reading here: the pinned training context is 2048, not
  Qwen3's 40960).
- **Eval slice:** hold out a **disjoint ~5–10M-token** slice of the same corpus for the perplexity
  measurement used across *all three* states (and gate G-RS1.2) — matched train/eval distribution
  so the before/after perplexity delta isolates *position*, not domain. This slice is needed from
  the first (inference-only) stage.
- **Data caveat:** Qwen3 is multilingual + code + math; FineWeb-Edu is English web, so the
  recalibrated model shifts toward English web. Controlled for the position claim by the matched
  held-out eval; optionally add small code/multilingual eval slices to *characterize* the shift,
  but do not treat the recalibrated checkpoint as a general Qwen3.
- No architecture change beyond the rotation removal.

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
- **(OPTIONAL, nice-to-have) Length-extrapolation perplexity.** Evaluate each state (RoPE /
  dropped / DroPE'd) at **1× and ~2× the recalibration context length** on the held-out slice.
  External-validity garnish: confirms the DroPE'd model exhibits DroPE's *signature* benefit
  (better extrapolation than RoPE). Inference-only, near-free. **Not load-bearing for C1/C2** —
  omit freely; see P.RS1.e.

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
- **(P.RS1.e — OPTIONAL, external-validity garnish)** The DroPE'd model extrapolates better than
  the RoPE baseline: perplexity at ~2× context degrades less for DroPE'd than for RoPE — DroPE's
  own signature benefit, reproduced as confirmation. **Not a requirement for C1/C2** and **no
  falsifier gates the experiment on it**; it exists only to show our checkpoint behaves like a real
  DroPE'd model. (Full LongBench/RULER length-gen benchmarking is deliberately *out of scope* —
  DroPE owns that headline; RS1 needs only the perplexity-recovery reproduction in P.RS1.a.)

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
- **RS1b** (recalibration ~1–2B tokens on 0.6B + re-probe): a **single-GPU, few-hour** job, not a
  cluster job. Compute ≈ 6·N·T ≈ 5×10¹⁸ FLOPs → **~3–4 h on one H100, ~10 h on one A100, ~15–20 h
  on one 24 GB 4090**. Full FT of 0.6B needs only ~7 GB (bf16 weights + grads + AdamW states), so a
  **24 GB card suffices** — no 80 GB GPU, no multi-node. Cost **~$6–16** on a single RunPod pod
  (A100-80GB ~$1.39/hr, H100-SXM ~$1.49/hr), **~$5** on Vast.ai (4090 ~$0.3/hr, interruptible →
  checkpoint every ~15 min), or possibly **free** under Modal's $30/mo credit. **Full-parameter,
  not LoRA:** dropping RoPE is a large mechanistic shift where LoRA underperforms, and full FT is
  cheap at this size. Tooling: HF Transformers + rotary-identity monkeypatch (reuse
  `m16_discriminator._apply_rotary_pos_emb`) + streamed FineWeb-Edu, packed to context. (Earlier
  "1–3 days / ~$50–150" estimate was over-conservative and is corrected here.)
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

---

## 10. Implementation Contract (execution brief)

**Read this before writing any RS1 code.** §§0–9 pin the *science* (hypotheses, gates, falsifiers,
decision tree). This section pins the *engineering* decisions an implementer would otherwise have to
invent — and where those decisions silently affect a gate, they are marked **[MUST]**. Defaults
marked **[D]** are recommendations: override with a recorded rationale, but *record the value used*.

**Trust boundary (per `reproducible-research`):** treat this contract as design to *execute and
question*, not gospel. If a stated assumption contradicts what you find in the code — an arch field,
a hook path, a config attribute — surface it as a finding and stop; do not paper over it. Several
"reuse unchanged" claims in §3 are **not** literally true (see §10.G) and are called out here.

**Context — what does not yet exist in the repo.** Everything to date is inference-only probing.
RS1 is the first experiment that introduces: (i) a training loop (there is **no** `optimizer.step`/
`backward`/`AdamW` anywhere), (ii) a local-checkpoint load path, (iii) a frozen perplexity/eval
definition shared across states, and (iv) probe-script branches for a rotation-disabled Qwen3.

### 10.A Model tags and loading paths

Three new loadable states + one reference. `load_model`/`MODEL_IDS` is HF-id-keyed with no
local-checkpoint path today — that must be added.

| tag | source | rotation | notes |
|---|---|---|---|
| `qwen3` (existing) | `Qwen/Qwen3-0.6B` | on | State 1 (RoPE baseline). Unchanged. |
| `qwen3-dropped` | same weights as `qwen3` | **disabled at runtime** | State 2. No new weights; a load flag reusing the `qwen3` checkpoint with rotary forced to identity (§10.B). |
| `qwen3-droped` | **local dir** (fine-tuned) | disabled (baked into recipe) | State 3. Requires local-path loading — new. |
| `nope-gpt-400m` | `andrewdalpino/NoPE-GPT-400M-Base` | native NoPE | State 4 reference, M1.5 only. **[MUST verify]** the `_capture_nope_k` hook on `qkv_proj` matches the 400M arch (20L, GQA 20Q/5KV, head-dim 64) before trusting the profile — the Small model's hook is *assumed* to transfer, not confirmed. |

- **[MUST]** Add a local-checkpoint branch to `load_model` (tag→filesystem path, or a `--model-path`
  override) so State 3 loads. Pin the exact directory in the run manifest.
- **[MUST]** Pin HF revisions (commit SHA) for `qwen3` and `nope-gpt-400m`, as already done for
  `nope-gpt-small` in the M1.6 manifests.

### 10.B Rotary-disable mechanism (states 2 & 3) + G-RS1.1

- **[D]** Disable rotation by **forcing `cos=1, sin=0` at the position-embedding source** (the
  `position_embeddings` tuple Qwen3Attention consumes), rather than monkeypatching
  `apply_rotary_pos_emb` in `transformers`. Rationale: least invasive, survives library updates,
  identical at train and inference time. `m16_discriminator._apply_rotary_pos_emb` is a
  *patching-stage* reimplementation — reuse its *math* for verification, not as the forward hook.
- **[MUST]** State 3 must train with this exact mechanism active, so the trained weights match the
  probed forward. Any train-time/probe-time rotary mismatch voids the before/after.
- **[MUST] G-RS1.1, perturbable:** assert `cos≡1 ∧ sin≡0` (or `k_post==k_pre` elementwise) at every
  layer in a dropped/DroPE'd forward; then flip one layer back to true RoPE and confirm the
  assertion *fails*. Run this inside a training-config forward too, not only inference.

### 10.C Training recipe (RS1b, State 3) — full block

Written from scratch (no training loop exists). Full-parameter recalibration:

| field | **[D]** default | note |
|---|---|---|
| framework | raw PyTorch loop (or HF `Trainer`) | small model; a raw loop is auditable and avoids Trainer rotary/config surprises |
| precision | bf16 mixed | §8 budget assumes it |
| optimizer | AdamW, β=(0.9, 0.95), eps=1e-8, wd=0.1 | standard LM recalibration |
| peak LR | **3e-5** *(decided)* | recalibration, not pretraining — kept low. This is the #1 recipe risk (P.RS1.a null-masquerade). |
| schedule | cosine → 10% of peak; warmup 2% of steps | |
| grad clip | 1.0 | |
| **train context** | **2048** *(decided; supersedes §2.2 "original context length")* | bounds memory, keeps the run cheap, and makes P.RS1.e's ~2× test (→4096) meaningful. |
| global batch | ~0.5M tokens (seq 2048 × grad-accum) | ⇒ ~2–4k steps for 1–2B tokens |
| token budget | 1–2B (record exact) | as §2.2 |
| seed | 0 (record) | single seed per §7 |
| checkpointing | every ~250 steps + final; keep final + best-eval | enables the Vast interruptible resume §8 assumes |

- **[MUST]** Emit a training-loss + periodic held-out-perplexity curve to the run artifacts —
  needed to distinguish "DroPE didn't replicate" (real result) from "recipe under-tuned" (bug), the
  §7 threat and the P.RS1.a falsifier.

### 10.D Data pipeline

- **[MUST]** Corpus `HuggingFaceFW/fineweb-edu`, `sample-10BT`, streamed. Tokenizer Qwen3's.
- **[D]** Pre-tokenize once to a single mmap'd **uint32** shard (vocab ~152k > 2¹⁶). Packing:
  concatenate documents with EOS between, split into contiguous `train_context`-length blocks (no
  cross-doc masking — standard for recalibration).
- **[MUST] Held-out determinism:** carve the disjoint ~5–10M-token eval slice by a **fixed rule
  recorded in the run manifest** (e.g. "first N tokens are eval, training reads from offset N," or a
  fixed-seed document split). The *rule*, not just the size, must be pinned — every gate is a
  cross-state delta on this exact slice.

### 10.E Perplexity / eval definition — frozen across all states **[MUST]**

- Reuse the sliding-window `perplexity()` in `experiments/dead-keys/deadkeys/scripts/phase1_5.py`
  as the *pattern*. **[MUST] wrinkle:** it reads `model.config.n_positions` (a GPT-2 field); Qwen3
  uses `max_position_embeddings`. Parameterize the window explicitly instead of reading either; set
  it to `eval_context`.
- **[MUST]** One definition, identical for States 1/2/3: token-weighted mean CE → exp;
  `eval_context = 2048`; stride **[D]** = `eval_context` (non-overlapping) *or* 512 (overlapping,
  lower-variance) — pick one and freeze it. Dropped/DroPE'd states run through the *same*
  rotation-disabled forward as their probes.
- This single number feeds **G-RS1.2** and **P.RS1.a**; drift between states voids both.

**[MUST] Three length knobs — pin each independently; do not conflate.** RS1 now has three distinct
lengths, and a comparison is only valid if the *same* value is used across all states for a given
knob:

| knob | value | scope |
|---|---|---|
| training context | 2048 | RS1b recalibration (§10.C) |
| perplexity `eval_context` | 2048 | frozen eval, all states (§10.E) |
| **M1.5 position-probe `--max-length`** | **1024** | M1.5 across all states |

The M1.5 probe length is **1024, not 2048**: RS1a ran States 1–2 at `--max-length 1024` (an L4-OOM
redo, recorded in `NOTEBOOK.md`), so **RS1b M1.5 on State 3 MUST also use 1024** — or States 1–2 must
be re-run at the newly chosen length. Otherwise the before/after M1.5 profiles (P.RS1.b/b′) are not
apples-to-apples. This is a probe-stimulus length and is independent of the two 2048 contexts above.

### 10.F C2 subspace analysis (secondary — may ship after C1)

- **[D]** Per layer, define the "positional subspace" as the top-`k` PCA directions of the
  position-decodable component (project keys onto the M1.5 ridge-position prediction, take principal
  directions), `k` = enough to reach the M1.5 PCA-to-90% variance already reported. Compare RoPE
  `k_post` vs DroPE'd emergent via **principal angles / CCA**.
- **[MUST for validity]** baseline = mean alignment under random rotations of one subspace; report
  overlap *above* baseline, not raw.
- **Design-on-implementation, not reuse:** unlike M1.5/M1.6 there is no existing code. C2 is the
  *secondary* claim — it may ship after C1.

### 10.G Probe-script changes (correcting §3's "reuse unchanged")

- **[MUST]** `position_content.py` and `m16_discriminator.py` both branch on `lm.tag == "qwen3"` and
  *always* apply rotary. Add branches for `qwen3-dropped`/`qwen3-droped` where `k_pre == k_post`
  (no rotation) and M1.6 patches the single K directly (§3 states the intent; the code does not do
  it yet).
- **[D]** Simplest: treat these tags as "qwen3 arch, rotation off" — reuse `_capture_qwen_k`, assert
  `pre==post`, use `pre` as the single K.

### 10.H Staged acceptance (what "done" means per stage)

1. **Plumbing smoke (near-zero compute):** load `qwen3-dropped`; G-RS1.1 passes *and is shown
   falsifiable*; perplexity(dropped) computes on the eval slice.
2. **RS1a gate:** States 1+2 M1.5/M1.6 run; **G-RS1.2 half-1** holds (ppl(dropped) ≫ ppl(RoPE)).
   *Only then* commit to training. This stage exercises §§10.B/D-partial/E/G at zero training
   cost — it is the forcing function that proves the machinery before any GPU spend.
3. **RS1b:** train State 3; **G-RS1.2 half-2** (ppl(DroPE'd) ≪ ppl(dropped)); re-probe; publish the
   checkpoint (~1.2 GB bf16) + outputs as a GitHub Release with checksums, per `AGENTS.md`.
