# K-address-space — closing note

**Dated:** 2026-07-23
**Status:** **CLOSED.** No further experiments planned in this group. Records are retained;
the two forward-looking results migrate to a successor program.

## Verdict

The k-address-space program set out to test whether cached key vectors behave like a
**content-addressable "transactional tape"** — same-referent mentions clustering in K, position
acting as a namespace, ‖V‖ as version dominance. Its central question is now **answered, and the
answer is negative at the scales tested**: via causal patching (M1.6), across the full positional-
encoding gradient (NoPE → full-RoPE Qwen3), **K is not a query-readable retrieval address**. The
positional content of K is decodable and — in RoPE models — attention-steerable, but it is
**causally sterile for the output**. Retrieval is content-addressed (Hopfield-style); the
positional component is cargo. Full results and adjudication: [`FINDINGS.md`](FINDINGS.md).

## The two-pronged direction, and why each prong is closed

The spec (§4) had two separable spines. They are closed for different reasons.

**Spine 1 — addressing / namespace (M1, M2, M3; P5.a/b/c/f).** *Is K a content-address with
position as a namespace?*
- **Answered — negatively.** M1.6 rules out query-readable addressing in both NoPE and full-RoPE.
- **And the instrument for M1/M2/M3 is void:** corpus defect **F8** left zero valid address-purity
  trials, and corpus v3 was never built. So the direct measurements are simultaneously *answered
  elsewhere* and *dead in their original form*.
- Continuing here would be re-running a broken instrument to re-ask a settled question. Nothing to
  gain.

**Spine 2 — version dominance / read semantics / ‖V‖ (M4, M7; P5.d/e/g).** *Does later ‖V‖ win;
is the read latest-wins vs averaging; does effective support track version count?*
- **Never run — and deliberately not pursued here.** Three reasons: (i) it is the naive question
  that seeded the whole program, and on inspection it is "obvious to check" with **limited novelty
  headroom**; (ii) the practical form — ‖V‖ / ‖W_O v‖ as a token-importance signal for KV-cache
  eviction — is a **crowded, fast-moving area** (VATP, StreamingLLM/attention-sinks, Massive
  Activations, Active-Dormant heads, Expected Attention, DepthKV, OBCache); (iii) the synthetic
  0.6B repeated-segment harness is the **wrong instrument** for a practical eviction claim anyway.
- If ever revived, it should be a **fresh, practically-framed program with its own novelty check**
  and a real-task instrument — not a k-address-space tack-on. (A scoping survey of the ‖V‖
  landscape, including the counter-intuitive attention-sink result, was done and can seed that if
  chosen.)

## Why close rather than continue

- The load-bearing question (Spine 1) is **resolved**; the remaining Spine-1 measurements are
  **corpus-dead** (F8).
- Spine 2 is **low-novelty, wrong-instrument, and crowded** at this scale.
- Everything here is **capped at 0.6B**; absence of addressing is scale-qualified and does not
  motivate more small-scale poking.
- The genuinely novel and forward-looking findings do **not** need this group to continue — they
  seed a successor (below). Continuing "for completeness" would be the distraction pattern, not
  new science. This is a disciplined close, not an abandonment of the results.

## What migrates forward

Two salvageable results carry into the successor program
[`../rope-as-scaffold/`](../rope-as-scaffold/), which asks — mechanistically — **whether RoPE's
positional contribution is a removable training scaffold** (the question underneath *DroPE*,
which showed empirically that RoPE can be dropped post-training but supplied no interpretability
account):

- **P1.5.c** — RoPE-stamped models still compute emergent position into the **pre-rotation** key
  stream at depth (apparently unclaimed; cuts against the textbook RoPE picture). → "RoPE's stamp
  is partly redundant with emergent computation."
- **M1.6 method + verdict** — position is decodable/steerable but **not an address**, with a
  **RoPE > NoPE steerability gradient**. → "the position RoPE provides is not load-bearing as a
  retrieval mechanism."

Together these are the mechanistic seed for *why* the scaffold is removable. That program's
load-bearing next step is **RS1** (the DroPE before/after), which is the only piece that would be
a genuinely new *result* rather than a new instance of an established principle
([`../rope-as-scaffold/references/novelty-check.md`](../rope-as-scaffold/references/novelty-check.md)).
The RoPE-as-scaffold investigation is **deferred to a later date**, not started here.

## Retained artifacts

`spec.md`, `addendum-M1.5.md`, `addendum-M1.6.md` (pre-registrations, v1.0 in git history),
`NOTEBOOK.md` (per-run records with checksums and published release URLs), `FINDINGS.md`
(this synthesis), and the code under `kaddress/`. Nothing is deleted; the group is frozen.
