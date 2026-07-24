# Qwen3 NoPE (DroPE) Training — Implementation Note

**Status:** artifact-generation step only — deliberately **not** tracked in `NOTEBOOK.md`. This
note is the provenance record for the training step instead. The *probing* phase that follows
(M1.5/M1.6/perplexity/C2 on the trained checkpoint) is still a normal RS1b reproducible-run under
`NOTEBOOK.md` — training just produces the checkpoint that run consumes.

**Deliverable:** final bf16 weights of the recalibrated Qwen3-0.6B (spec tag `qwen3-droped`, local
directory), persisted on the RunPod store. No optimizer state, no periodic/resumable checkpoints —
final weights only.

## Storage budget (~10–15GB persistent store)

- Token budget (1–2B) is a **single pass** over `sample-10BT` (~10B tokens available), plus a small
  disjoint eval slice — no token is trained on more than once. Caching therefore does not introduce
  repeated exposure; it exists purely for **restart resilience** (see "No-checkpoint tradeoff"
  below): without a cache, a crash mid-run means re-streaming *and* re-tokenizing 1–2B tokens from
  the Hub from scratch.
- **Stream once, write a local tokenized cache as you go.** Tokenize FineWeb-Edu on the fly from the
  streaming source (as `eval_perplexity.py` already does) and persist the resulting token-id shard
  to a local mmap'd **uint32** file (vocab ~152k > 2¹⁶) at the same time training consumes it. A
  restart reads the local cache instead of re-streaming + re-tokenizing.
- Budget: train-token cache 1–2B × 4 bytes = **4–8GB**; eval-slice cache (5–10M tokens) ≈
  20–40MB, trivial; final weights ≈ 1.2GB. Total ≈ **5.5–9.5GB** — fits comfortably within the
  10–15GB allowance.
- Confirm whether this store is the *same* volume as the project's existing RunPod network volume
  (`AGENTS.md` → `$HF_HOME`, CUDA venv, etc.) or a separate one. If separate, keep the base-model
  download cache and CUDA venv on the larger existing volume, not this one.

## Recipe (pinned — spec `RS1-spec.md` §10.C; do not deviate without recording the change)

| field | value |
|---|---|
| optimizer | AdamW, β=(0.9, 0.95), eps=1e-8, wd=0.1 |
| peak LR | 3e-5 |
| schedule | cosine → 10% of peak; warmup 2% of steps |
| grad clip | 1.0 |
| precision | bf16 mixed |
| train context | **2048** |
| global batch | ~0.5M tokens |
| token budget | 1–2B (record exact count used) |
| seed | 0 |

## Rotary-disable mechanism (spec §10.B)

- Force `cos=1, sin=0` at the `position_embeddings` source (Qwen3Attention's input tuple), **not**
  a monkeypatch of `apply_rotary_pos_emb`. Reuse `set_qwen_rotary_identity()` /
  `deadkeys.common.loading` (already built and gate-verified for `qwen3-dropped` in RS1a).
- **Must be baked into training**, not just applied at inference: state 3 is trained *through* this
  identity path so the trained weights match the probed forward exactly. Any train/probe rotary
  mismatch voids the whole before/after comparison.
- Verify with the existing G-RS1.1 pattern (`verify_grs11.py`) on the training-config forward before
  a full run: `cos≡1 ∧ sin≡0` at every layer, and flipping one layer back to real RoPE must make the
  check fail.

## Data pipeline (spec §10.D)

- `HuggingFaceFW/fineweb-edu`, config `sample-10BT`, streamed train split, Qwen3 tokenizer.
- **MUST** record the deterministic held-out eval-slice carving rule (not just its size) — reuse the
  exact rule already frozen in `eval_perplexity.py` (first `eval_tokens` packed tokens = eval;
  training reads from beyond that offset) so states 1/2/3 share one held-out set.
- Packing: concatenate docs with EOS between, split into contiguous 2048-token blocks, no
  cross-doc masking.

## Eval definition — frozen, do not redefine (spec §10.E)

- `eval_context=2048`, stride 2048, token-weighted mean CE → `exp()` for perplexity. Same
  definition, same code path (`eval_perplexity.py`), across states 1/2/3.
- G-RS1.2 half-2 (perplexity(DroPE'd) ≪ perplexity(dropped)) and P.RS1.a both depend on this being
  identical to what RS1a already ran for states 1–2.

## M1.5 probe-length gotcha (learned in RS1a, easy to miss)

RS1a's states 1–2 were re-run at **`--max-length 1024`** (an L4-OOM redo), not 2048. State 3's
M1.5 run **must also use 1024**, or the before/after position-decodability profiles (P.RS1.b/b′)
are not comparable. This is independent of the two 2048 contexts above (training context, eval
context).

## Revision pinning (AGENTS.md, MUST)

Resolve the exact Qwen3-0.6B HF revision SHA up front (`HfApi().model_info(...).sha`); do not load
`main`. RS1a recorded `revision: null` — a flagged, already-known defect; do not repeat it here.
Record the SHA in this checkpoint's manifest.

## Local-checkpoint load path (spec §10.A, MUST)

Build the `qwen3-droped` filesystem-path branch in `load_model` (tag → local dir, or a
`--model-path` override) alongside training — the probing phase needs it immediately after
training finishes.

## Observability

Even though this isn't a tracked notebook run, emit a training-loss + periodic held-out-perplexity
curve to a log/artifact file. This is the only way to distinguish "DroPE doesn't replicate" (a
real, reportable P.RS1.a falsifier) from "recipe under-tuned" (a bug) after the fact.

## Cost / smoke-test guardrail

First paid GPU spend in this program (~$6–16, single GPU, few hours). Before committing to the
full run: do a bounded smoke (a few dozen steps) confirming (a) loss actually drops, (b) the
rotary-identity forward holds under the training config, (c) a saved checkpoint reloads through
the same forward and reproduces the same loss. Only then launch the full run.

## No-checkpoint tradeoff

Only final weights will be saved (no periodic/resumable *model* checkpoints). The token cache
(above) makes a restart cheap on the **data** side — no re-streaming/re-tokenizing — but does
**not** give the training run itself a resume point: if the pod dies mid-run, all optimizer/model
progress is still lost and training restarts from step 0 (reading from the now-local cache).
Acceptable given the low cost, but worth confirming the pod type before launching a multi-hour job
with zero training-progress resumability.

## If this checkpoint gets published later

No GH Release / `NOTEBOOK.md` flow applies to this training step by design. If the trained NoPE
Qwen3 is later published (HF or otherwise), the pins recorded here — recipe, exact token count,
data revision + slice rule, base-model SHA — are the model-card provenance; keep them in the
checkpoint's own manifest file even without a notebook entry.
