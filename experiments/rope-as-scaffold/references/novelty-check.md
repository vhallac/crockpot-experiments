# Novelty check

**Compiled:** 2026-07-23. Web survey of ~5 targeted queries plus a full read of the nearest
paper. **Not a systematic review** — "appears unclaimed" means "not surfaced here," not
"is unclaimed." A proper related-work pass is a prerequisite before publishing or pitching.
Supports the [program charter](../README.md); landscape in
[`literature-survey.md`](literature-survey.md).

## Verdict by piece

| piece | novelty | why |
|---|---|---|
| **Emergent position in general** (NoPE reconstructs position) | **crowded — do not claim** | Haviv 2022; Kazemnejad 2023; [2501.00073](https://arxiv.org/abs/2501.00073) |
| **P1.5.c** — RoPE-stamped models *also* compute position into the **pre-rotation** key stream (`k_pre`), depth-resolved, across the PE gradient | **appears unclaimed** (verify with one targeted search on the pre-RoPE-key angle) | not covered by the nearest paper (see below); not surfaced elsewhere |
| **M1.6 causal-addressing negative** — K is decodable/attention-steerable but not a query-readable retrieval address, tested by K/V patching, NoPE vs full-RoPE; RoPE more steerable than NoPE yet neither addresses | **appears unclaimed** | activation-patching literature targets circuits (IOI, induction), not position-as-address across the PE gradient |
| **The DroPE bridge** — causal-mechanistic link from "K isn't an address" to "why removal is safe" | **unclaimed but unmeasured** | the intuition exists (DroPE, RNoPE-SWA); no one has the causal evidence — and neither do we yet (needs RS1 before/after) |

## The nearest paper: *Deconstructing Positional Information* (2505.13027)

Full read (2026-07-23) confirms it does **not** pre-empt P1.5.c:

1. It does **not** distinguish pre- vs post-rotation keys; it treats RoPE as a unified
   attention-logit decomposition. The `k_pre` axis P1.5.c lives on is absent.
2. It does **not** claim emergent/implicit position in RoPE models' learned key content. Its
   thesis is the reverse framing (position explicitly encoded by RoPE) plus a theoretical
   contribution: multiplicative coupling → **spectral contraction → optimization stability**.
3. Depth analysis is "which heads process position" (single-head deposit in early layers), not
   whether position content re-emerges in the pre-rotation representation with depth.
4. Scope: a single **6-layer** decoder on **controlled synthetic tasks**, no scale; NoPE only a
   one-line baseline. Authors caveat it "may not directly translate to real-world datasets."

So the paper that looked closest is a different genre (theoretical/optimization/synthetic) and
leaves P1.5.c and the M1.6 addressing question open.

## A contradiction worth owning

A common framing (e.g. the ["From RoPE to NoPE and Back Again"](https://medium.com/@cenghanbayram35/from-rope-to-nope-and-back-again-is-positional-embedding-the-wrong-question-13654966f8d2)
writeup) claims *"RoPE's signal is so potent it may inhibit the model from forming its own
implicit positional representations."* P1.5.c **complicates this**: Pythia/Qwen3 still compute
position into `k_pre` at depth despite RoPE. A small, specific point to own — if a targeted
search confirms it is unclaimed.

## Consequences for the write-up

- **Foreground M1.6** (addressing negative + RoPE>NoPE steerability) and **P1.5.c** (redundant
  emergent position) as the spine; treat general emergent-position as related work; **make no
  novelty claim** on the NoPE-emergent-position front.
- Together P1.5.c + M1.6 form a two-part mechanistic case ("RoPE's position is redundant *and*
  non-addressable → droppable scaffold") — the part that actually speaks to DroPE.
- **Outstanding before any claim of novelty:** (a) one targeted search on "pre-RoPE key emergent
  position in stamped models"; (b) the RS1 before/after run to make the DroPE link a *result*.

## Exhaustive pass (2026-07-23) — prior art found, verdict refined downward

A broader sweep (~14 queries across associative-memory, fast-weight, tie-breaking,
causal-use, and position-head-interpretability framings) found substantial prior art for the
program's *conceptual pillars*. The initial optimistic read above is refined: **the framing is
largely pre-existing; only the specific causal measurement and the DroPE tie-in remain open.**

Prior art per pillar:

| framing pillar this program leans on | pre-existing work | status |
|---|---|---|
| "K / attention is content-addressable / associative memory" | Modern Hopfield ("Hopfield Networks is All You Need", Ramsauer et al. 2020); linear-attention = Hopfield | **established — do not claim** |
| "tape / write-at-similar-key / delta-commit" memory | Fast-Weight Programmers (Schlag/Irie/Schmidhuber 2021); DeltaNet and its 2025–26 descendants | **established** |
| "**decodable ≠ causally used**" (the epistemic core of M1.5→M1.6) | *Dissociating Decodability and Causal Use in Bracket-Sequence Transformers* ([2604.22128](https://arxiv.org/pdf/2604.22128)) — decodable depth/distance are causally inert, only top-of-stack is used | **established as a principle** — our M1.6 is a *domain instance*, not a new insight |
| "position breaks ties / anti-collision among identical keys" | known property of PE under repeated tokens (softmax degeneracy; APE-keeps-sink vs RoPE/ALiBi-collapse literature) | **established** |
| "RoPE is partly redundant / droppable" | DroPE; Selective RoPE ([2511.17388](https://arxiv.org/html/2511.17388v1)); half-dim RoPE | **established** |
| "NoPE reconstructs position" | Haviv 2022; Kazemnejad 2023; [2501.00073](https://arxiv.org/abs/2501.00073); [2305.13571](https://arxiv.org/pdf/2305.13571) | **established** |
| "RoPE concentrates position in shallow heads / deposit patterns" | [2505.13027](https://arxiv.org/html/2505.13027) (via selective-RoPE ablation) | **established, adjacent to P1.5.c** |
| position/KV decoupled as a namespace for reuse | LazyAttention ([2606.04302](https://arxiv.org/pdf/2606.04302)) | **established (efficiency angle)** |

**What genuinely remains unclaimed (narrowed):**
- The **specific empirical result** — causal K/V patching showing the *positional* component of K
  is decodable + attention-steerable but **not a query-readable retrieval address**, measured
  **across the NoPE→full-RoPE gradient**, with the RoPE>NoPE steerability-without-addressing
  contrast. Nobody ran *this* experiment — but it is an *instance* of decodability≠causal-use
  (2604.22128), so the contribution is "a specific, clean instantiation on positional
  information + the cross-PE comparison," not a new concept.
- **P1.5.c** (pre-rotation emergent position at depth in stamped models) — still not directly
  found; [2505.13027](https://arxiv.org/html/2505.13027)'s deposit-pattern result is the nearest
  and must be cited/contrasted carefully.
- The **DroPE mechanistic bridge** (E1+E2 → why removal is safe) — unclaimed **and unmeasured**;
  RS1 is the only part that would be a genuinely new *result* rather than a new instance of an
  old idea.

**Revised recommendation.** The conceptual novelty is thin: every framing pillar has prior art,
and the epistemic core (decodable ≠ causally used) is already published. A standalone note built
only on M1.5/M1.6 would be a *small empirical instance* that must heavily cite Hopfield/Ramsauer,
2604.22128, DroPE, and 2505.13027 — publishable at most as a short note, not a conceptual
contribution. **RS1 (the DroPE before/after) is therefore not optional if the goal is a real
contribution** — it is the one piece that converts this from "another instance of a known
principle" into "mechanistic evidence about a specific method (DroPE)." Absent RS1, reconsider
whether the tech note is worth writing.

## Open queries still to run

- Read 2604.22128 in full and position M1.6 explicitly as a positional-information instance of it.
- One more targeted search on P1.5.c phrasing before claiming even the narrow pre-rotation result.
