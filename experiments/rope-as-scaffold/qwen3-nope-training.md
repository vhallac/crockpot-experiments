# Qwen3 NoPE (DroPE) Training — Implementation Note

**Status:** artifact-generation step only — deliberately **not** tracked in `NOTEBOOK.md`. This
note is the provenance record for the training step instead. The *probing* phase that follows
(M1.5/M1.6/perplexity/C2 on the trained checkpoint) is still a normal RS1b reproducible-run under
`NOTEBOOK.md` — training just produces the checkpoint that run consumes.

**Deliverable:** final bf16 weights of the recalibrated Qwen3-0.6B (spec tag `qwen3-droped`, local
directory), persisted on the RunPod store, plus a resumable training checkpoint saved periodically
and on signal (see "Checkpointing and resumability" below).

## Storage budget (~10–15GB persistent store)

- Token budget (1–2B) is a **single pass** over `sample-10BT` (~10B tokens available), plus a small
  disjoint eval slice — no token is trained on more than once. Caching therefore does not introduce
  repeated exposure; it exists purely for **restart resilience** (see "Checkpointing and
  resumability" below): without a cache, a crash mid-run means re-streaming *and* re-tokenizing
  1–2B tokens from the Hub from scratch.
- **Stream once, write a local tokenized cache as you go.** Tokenize FineWeb-Edu on the fly from the
  streaming source (as `eval_perplexity.py` already does) and persist the resulting token-id shard
  to a local mmap'd **uint32** file (vocab ~152k > 2¹⁶) at the same time training consumes it. A
  restart reads the local cache instead of re-streaming + re-tokenizing.
- Budget: train-token cache 1–2B × 4 bytes = **4–8GB**; eval-slice cache (5–10M tokens) ≈
  20–40MB, trivial; final weights ≈ 1.2GB; resumable training checkpoint (bf16 model + bf16 AdamW
  `exp_avg`/`exp_avg_sq` state — no fp32 master copies anywhere in this script) ≈ **3.6GB
  steady-state, ~7.2GB transient peak** during a checkpoint overwrite (old + new coexist briefly
  before the atomic rename). Total steady-state ≈ **8.8–12.8GB**; peak during a mid-run checkpoint
  write ≈ **12.4–16.4GB** — verified against a live pod at 14GB already used / 15GB avail on a 28GB
  volume, with several GB of margin at the worst moment. Recompute against the actual `df` reading
  before launching if the volume's fill level has changed.
- Confirm whether this store is the *same* volume as the project's existing RunPod network volume
  (`AGENTS.md` → `$HF_HOME`, CUDA venv, etc.) or a separate one. If separate, keep the base-model
  download cache and CUDA venv on the larger existing volume, not this one.

## Recipe (pinned — spec `RS1-spec.md` §10.C; do not deviate without recording the change)

| field | value |
|---|---|
| optimizer | AdamW, β=(0.9, 0.95), eps=1e-8, wd=0.1 |
| peak LR | **1e-3** (revised 2026-07-25, was 3e-5 — see `rs1b-lr-retuning-note.md` and `RS1-spec.md` §10.C; fall back to 3e-4 if unstable) |
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

First paid GPU spend in this program (~$6–16 at the original LR; more at the corrected LR's higher
step-for-step cost is not expected since token budget/step count are unchanged — single GPU, few
hours). Before committing to the full run: do a bounded smoke (a few dozen steps, with
`--checkpoint-every` set low enough to trigger at least once) confirming (a) loss actually drops,
(b) the rotary-identity forward holds under the training config, (c) `--resume-from` on the smoke
checkpoint reloads and continues training (loss/eval trajectory picks up consistently, not a
discontinuity), and (d) a `SIGHUP`/`SIGUSR1` sent mid-smoke checkpoints and exits/continues as
expected. Only then launch the full run.

## Checkpointing and resumability

Implemented in `train_qwen3_nope.py` (added after the RS1b v1 run, per the SIGHUP/snapshot lesson
below — this is the "next training script" that lesson pointed at):

- **Periodic:** every `--checkpoint-every` steps (default 500), overwriting a single
  `training_checkpoint/ckpt.pt` — not one file per step, to keep steady-state disk cost bounded
  (see storage budget above).
- **Signal-triggered:** `SIGHUP` → checkpoint then exit; `SIGUSR1` → checkpoint then continue.
  Both are handled as flags checked right after `optimizer.step()`/`scheduler.step()`, never
  mid-gradient-accumulation, so a signal can't land on a torn/inconsistent state.
- **Atomic writes:** each save goes to `ckpt.pt.tmp` then `Path.replace()`s onto `ckpt.pt` — a
  crash mid-`torch.save` (disk full, OOM, network hiccup on the NFS-mounted volume) leaves the
  prior good checkpoint intact instead of corrupting the only copy.
- **Checkpoint payload, for bit-faithful resumption:** model/optimizer/scheduler state dicts, RNG
  state (python/numpy/torch/cuda — without this a resumed run silently diverges from what a
  continuous run would have produced), plus `step`/`block_cursor`/`elapsed_s` so the resumed run's
  data position and `training_metrics.csv` timeline are both continuous across the resume
  boundary, not reset to zero.
- **`--resume-from <dir>`:** restores everything above and asserts `total_steps` matches the
  current config. It does **not** validate that `--lr`/`--micro-batch-size`/etc. match the
  original launch — resuming with different CLI args than the original run will silently produce
  an inconsistent trajectory. Treat "identical CLI args to the original launch" as a hard
  assumption when resuming, not something the script enforces.
- `training_metrics.csv` is opened in append mode (not truncated) when `start_step > 0`, so a
  resumed run's loss/eval history stays complete across the resume boundary.

This replaces the earlier no-checkpoint policy entirely — the tradeoff described in that policy
(cheap but zero resumability) no longer applies to this script.

## If this checkpoint gets published later

No GH Release / `NOTEBOOK.md` flow applies to this training step by design. If the trained NoPE
Qwen3 is later published (HF or otherwise), the pins recorded here — recipe, exact token count,
data revision + slice rule, base-model SHA — are the model-card provenance; keep them in the
checkpoint's own manifest file even without a notebook entry.
