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

## Open queries still to run

- Precise search: pre-rotation-key emergent position in RoPE/stamped models (close out P1.5.c).
- Whether anyone has causally shown position-in-K is non-addressable (close out M1.6 novelty).
