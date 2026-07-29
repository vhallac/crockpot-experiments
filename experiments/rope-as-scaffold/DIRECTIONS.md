# Directions for Future Research — rope-as-scaffold

**Dated:** 2026-07-29
**Parent:** RS-amendment-2-3 (`NOTEBOOK.md` 2026-07-28) and the RS1–RS3 line.
**Status:** candidates only. None is pre-registered. Each becomes a real experiment only when it
gets its own directory, its own single-question spec, and its own notebook.

## How to read this document

Each direction states **one question**, what evidence it **inherits** (so it can be started
without re-running anything), what it **costs**, and what would make it **not worth doing**. A
direction that inherits everything it needs is a writing-and-analysis task, not an experiment
budget item — that distinction is the whole point of costing them here.

Working principle for this program: **an experiment's value may lie chiefly in producing its
successor.** RS-amendment-2-3 settled its own question and spawned six; that is a good outcome,
not a consolation.

**Inheritance hygiene.** Every direction below names the artifact it borrows from. When one is
promoted to a real experiment, its spec must *restate* the inherited numbers rather than linking
to them — parent outputs age, tarballs move, and "re-doable on its own" has to survive that.

---

## D1 — Is post-hoc PE removal evaluated against the wrong baseline?

**Question.** Does the reported success of DroPE-style positional-encoding removal depend on
comparing the modified model against an *un-adapted* baseline, and does a matched-training control
reveal a persistent penalty?

**Inherits (complete — no new runs).** The three-way perplexity triple from P.ctrl.d:
`qwen3` CE 3.0819 / `qwen3-rope-recal` CE 2.6569 / `qwen3-droped` CE 2.8260, all on the frozen RS1
5M-token eval slice, same harness, same cached token stream. Plus RS1b's training curve.

**Why it may be the most externally interesting.** The decomposition — adaptation worth 0.425
nats, removal costing 0.169 back — is a *methodological* claim that generalises past this model
and this recipe. If the literature routinely reports PE-removal results against un-adapted
baselines, this is a reusable critique, not a Qwen3-0.6B curiosity.

**Cost.** **$0.** Analysis and writing, plus a literature check on how DroPE and neighbours report
their baselines (`references/literature-survey.md` is the starting point).

**Kill criteria.** If the literature already routinely uses matched-training controls, this is a
local finding about our own earlier framing and nothing more — record it and stop.

---

## D2 — Reconstructed position is not functionally equivalent to supplied position

**Question.** How can the positional subspace be demonstrably reconstructed (C2, L3–L12,
confound-controlled) while function degrades on every behavioural axis measured (C3)?

**Inherits (complete).** RS2/RS2.1 subspace numbers (V1 excess +0.269 / +0.180 in L3–L7 / L8–L12;
V2 inheritance control; V3 residual +0.055) and the RS3 + amendment behavioural numbers.

**Why it matters conceptually.** This is the program's motivating principle — *decodable ≠ causally
used* — applied to the program's own thesis. The scaffold argument assumed that reconstructed and
decodable implies functionally replaced; it does not. Stating that cleanly may be the most durable
intellectual output of the whole line.

**Cost.** **$0** to state and evidence from existing numbers. Optional extension: targeted probes
asking *which* property of RoPE's code fails to transfer (magnitude? relative structure?
head-specific geometry?) — that would need GPU.

**Kill criteria.** If the reconstruction is weak enough that "reconstructed" was always an
overstatement, this collapses into a restatement of C2's known qualification (residual rank ~3 vs
RoPE's ~28) and is not independently interesting.

---

## D3 — Does retrieval depend specifically on RoPE's relative-offset primitive?

**Question.** Is the induction/retrieval loss attributable to a *relative-offset* operation
("attend to the token after my last match") rather than to positional information generally?

**Inherits (partial).** Induction gain × distance × three models, and the key signature: the
RoPE-attributable share of the loss **grows with distance** (19% at d=64 → 47% at d=512 → ~55% at
d=1024). Plus the complementary Arm A signature — the local-order cost is confined to w≤8.

**Cost.** **$0** to formalise the hypothesis. Testing it needs head-level induction analysis
(which heads carry the match+1 operation, and what happens to them across the transition) →
modest GPU, and it would want M1.5/M1.6-style machinery.

**Kill criteria.** If head-level analysis shows the loss is diffuse rather than concentrated in
identifiable induction heads, the "primitive" framing is wrong and this becomes a generic
capability-degradation result.

**Note.** This is the mechanistic account the RS3 entry floated and the amendment strengthened. It
is currently a *hypothesis with two converging signatures*, not a finding.

---

## D4 — Does PE removal destroy length behaviour beyond the recalibration window?

**Question.** Is the DroPE'd model's context collapse a property of *removal*, or merely of having
been recalibrated at a short context?

**Inherits (partial, and confounded).** RS3 Arm D: `qwen3` flat across 1024–8192 (PPL 21.5–22.7)
versus `qwen3-droped` collapsing (16.9 @2048 → 51.3 @4096 → 298.4 @8192).

**The confound, stated plainly.** `qwen3` was *pretrained* at 32k so 8192 is in-distribution for
it; `qwen3-droped` was recalibrated at 2048. `qwen3-rope-recal` shares the 2048 context, so it
does **not** disentangle this — Arm D was explicitly excluded from the amendment's scope for this
reason. Separating the two requires a longer-context recalibration arm.

**Why it is worth the money anyway.** It bears directly on NoPE's claimed length-generalisation
advantage, and our data currently cuts *against* that claim for post-hoc removal.

**Cost.** GPU — a longer-context recalibration arm. Not costed; depends on context length chosen.

**Kill criteria.** If a length-matched recalibration closes the gap, the collapse is a
recalibration-context artifact and there is no finding.

---

## D5 — The G6 ceiling: M1.6 becomes inapplicable as models improve

**Question.** Is M1.6's marker-neutrality gate genuinely infeasible on better models, or merely
under-searched — and can the relationship to model quality be characterised well enough to predict
feasibility before spending?

**Already scoped** in [`G6-ceiling-investigation.md`](G6-ceiling-investigation.md); not repeated
here.

**Inherits.** The three-point monotone relationship (`qwen3` 21.80 PPL passes at 512 →
`qwen3-droped` 16.88 needs 4096 → `qwen3-rope-recal` 14.25 fails at 4096).

**Cost.** Cheap GPU — the marker search is forward passes.

**Priority note.** This **blocks RS4 planning**, since RS4's E1/E2 spot-check assumes M1.6 runs on
a larger and therefore better model. The M1.6 retry pending inside RS-amendment-2-3 will inform it.

---

## D6 — RoPE as a training warmup for NoPE

**Question.** Does early RoPE produce a better NoPE model than never having had it — i.e. does
RoPE have *training-time* scaffolding value, separate from the inference-time value now
established?

**Inherits nothing numeric.** This is the only direction here that requires new training, which is
precisely why it must clear a higher bar than the others.

**Design crux — read before scoping.** Starting from a fully-pretrained checkpoint does **not**
test warmup: both branches would inherit RoPE-formed positional circuitry, and the "no RoPE" branch
would essentially be `qwen3-droped` with extra steps. A clean test needs a starting point *before*
positional circuits form. Practical vehicle: **Pythia publishes ~143 intermediate checkpoints**,
`pythia410` is already in `MODEL_IDS`, and `load_model` already threads `revision` (the codebase
documents `step1000` as the example). Cost of not-chaining to the Qwen3 line: results would be a
companion on a different model — arguably acceptable for a training-dynamics question.

**Gating tension that needs an explicit ruling.** The README gates this on RS1 confirming the
scaffold thesis, and lists "*c fails (causal for retrieval) → moot*." By the letter the gate is
**open** (P.RS1.a/b/c all held). By substance it has partly closed: RS3 + the control showed
retrieval *is* causally damaged, via a different route than M1.6 addressing. Suggested resolution:
treat the gate as **open but re-motivated** — the original hook ("RoPE bootstraps position, then
becomes dispensable") is dead, since we have shown it does not become dispensable. The surviving
question is narrower and still real.

**Cost.** Real training, two branches. Order-of-magnitude **$30–40** on H100 for a Pythia-410m pair
at ~1B tokens each. The only money item on this list.

**Kill criteria.** If the warmed branch matches the never-RoPE branch, RoPE has no training-time
value and the whole extension folds.

---

## Loose threads (not directions yet)

- **Local-order sensitivity inverts at w≥16.** `qwen3-droped` is *more* scramble-sensitive than the
  control at w=16 (−0.005) and w=32 (−0.022), reversing the w≤8 result. Not pre-registered, small,
  unexplained. Inherits the full Arm A window sweep; $0 to look at. Could be a real signature of
  where RoPE's contribution stops, or could be noise in a metric that saturates at large windows.
- **`ce_first` is flat at ~13.1 across all three models** including the near-random untrained one.
  Reassuring for instrument validity, but worth understanding — it may indicate the induction floor
  is set by something other than model quality.

---

## Not directions — open items inside existing experiments

Listed to prevent them being mistaken for future work:

- **P.ctrl.c** (subspace training-drift baseline) — an open arm of RS-amendment-2-3, pending an
  M1.5 re-run on the existing checkpoint. Not a successor.
- **P.ctrl.e** (M1.6 profile) — an open arm of RS-amendment-2-3, pending a retry at a much larger
  `--max-marker-sets`. Not a successor.
- **RS4** (C4 scale check) — already planned; blocked on D5.
