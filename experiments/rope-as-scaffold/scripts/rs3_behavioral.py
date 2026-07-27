"""RS3 — Functional locus of RoPE (local-order acuity vs. retrieval).

Pre-registered in ``experiments/rope-as-scaffold/RS3-spec.md``, 2026-07-27.
Tests claim C3 of the RoPE-as-Scaffold program.

Arms:
  A — local_scramble: CE delta under within-window scramble/reverse
  B — induction:        synthetic repeated-span retrieval gain
  C — kv_retrieval:     key–value lookup (secondary, floor-gated)
  D — length_ce:        CE at varying eval context lengths (exploratory)

Every arm generates items once (seed 0) then scores all models on the same items.
Per-item outputs are always written; aggregates + bootstrap CIs go into a summary JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic
from typing import Sequence

import torch
import torch.nn.functional as F
import numpy as np

# --- Reuse the frozen RS1 infrastructure ------------------------------------

# Ensure the eval_perplexity module is importable.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from eval_perplexity import (
    EVAL_CONTEXT,
    STRIDE,
    fineweb_edu_eval_ids,
    token_weighted_ce_and_ppl,
)

from deadkeys.common.loading import MODEL_IDS, load_model, uses_dropped_rope

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RS3_EVAL_OFFSET_DEFAULT = 5_000_000  # tokens to skip past RS1a's eval slice

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Return mean, SD, and 95 % bootstrap CI for an array of per-item deltas."""
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = float(np.mean(values[idx]))
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if n > 1 else float("nan"),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "n_items": n,
    }


ARM_A_BATCH_SIZE = 4   # 2048-token blocks — memory-bound; keep small
ARM_B_BATCH_SIZE = 16  # up to ~1600-token items — moderate
ARM_C_BATCH_SIZE = 32  # 243-token items — short, can fill


def _ce_batched(
    model: torch.nn.Module,
    batch: torch.Tensor,
    device: torch.device,
) -> list[float]:
    """Token-weighted CE for a batch of equal-length sequences [N, L].

    Returns one float per sequence.  All sequences in the batch MUST
    have the same length (no padding — callers split by length first).
    """
    N, L = batch.shape
    if L < 2:
        return [float("nan")] * N
    inp = batch.to(device)
    with torch.no_grad():
        out = model(inp, use_cache=False)
        logits = out.logits[:, :-1, :]                 # [N, L-1, V]
        labels = inp[:, 1:]                            # [N, L-1]
        # Per-sequence token-weighted mean CE.
        ce_per_seq = F.cross_entropy(
            logits.reshape(N * (L - 1), -1),
            labels.reshape(-1),
            reduction="none",
        ).view(N, L - 1).mean(dim=1)                   # [N]
    return [float(v.item()) for v in ce_per_seq]


def _split_blocks(ids: torch.Tensor, block_size: int) -> list[torch.Tensor]:
    """Split 1-D token tensor into contiguous non-overlapping blocks."""
    blocks: list[torch.Tensor] = []
    for start in range(0, len(ids) - block_size + 1, block_size):
        blocks.append(ids[start : start + block_size])
    return blocks


# ---------------------------------------------------------------------------
# Arm A — Local-scramble CE
# ---------------------------------------------------------------------------


def _scramble_block(ids: torch.Tensor, w: int, seed: int) -> torch.Tensor:
    """Within each non-overlapping window of size `w`, random-permute tokens."""
    rng = random.Random(seed)
    permuted = ids.clone()
    n = len(ids)
    for start in range(0, n, w):
        end = min(start + w, n)
        win = list(range(start, end))
        rng.shuffle(win)
        for j, idx in enumerate(range(start, end)):
            permuted[idx] = ids[win[j]]
    return permuted


def _reverse_block(ids: torch.Tensor, w: int) -> torch.Tensor:
    """Within each non-overlapping window of size `w`, reverse token order."""
    rev = ids.clone()
    n = len(ids)
    for start in range(0, n, w):
        end = min(start + w, n)
        rev[start:end] = ids[start:end].flip(0)
    return rev


def _arm_a_score_model(
    model: torch.nn.Module,
    blocks: list[torch.Tensor],
    windows: list[int],
    modes: list[str],
    device: torch.device,
    seed: int,
) -> list[dict]:
    """Score one model on all blocks × conditions. Returns per-block rows."""
    rows: list[dict] = []
    n_blocks = len(blocks)
    batch_size = ARM_A_BATCH_SIZE

    # Pre-compute clean CE per block in batches (shared across perturbations).
    clean_ces: list[float] = []
    for b_start in range(0, n_blocks, batch_size):
        b_end = min(b_start + batch_size, n_blocks)
        batch = torch.stack(blocks[b_start:b_end])     # [B, L]
        ces = _ce_batched(model, batch, device)
        clean_ces.extend(ces)
        print(f"    arm-A clean block {b_end}/{n_blocks}", flush=True)

    # Perturbation conditions — also batched.
    for mode in modes:
        for w in windows:
            for b_start in range(0, n_blocks, batch_size):
                b_end = min(b_start + batch_size, n_blocks)
                perturbed_blocks: list[torch.Tensor] = []
                for bi in range(b_start, b_end):
                    block = blocks[bi]
                    if mode == "scramble":
                        perturbed_blocks.append(_scramble_block(block, w, seed))
                    elif mode == "reverse":
                        perturbed_blocks.append(_reverse_block(block, w))
                    else:
                        raise ValueError(f"unknown mode: {mode}")
                batch = torch.stack(perturbed_blocks)
                ces = _ce_batched(model, batch, device)
                for i, bi in enumerate(range(b_start, b_end)):
                    rows.append(
                        {
                            "block": bi,
                            "mode": mode,
                            "window": w,
                            "ce_clean": clean_ces[bi],
                            "ce_perturbed": ces[i],
                            "delta_ce": ces[i] - clean_ces[bi],
                        }
                    )
            print(f"    arm-A {mode} w={w} done", flush=True)
    return rows


def run_local_scramble(
    model_tags: list[str],
    eval_ids: torch.Tensor,
    *,
    block_size: int = EVAL_CONTEXT,
    windows: list[int] | None = None,
    modes: list[str] | None = None,
    bootstrap: int = 10_000,
    seed: int = 0,
    device: str = "cuda",
    revision: str | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Arm A: local-order perturbation CE delta."""
    if windows is None:
        windows = [2, 4, 8, 16, 32]
    if modes is None:
        modes = ["scramble", "reverse"]

    blocks = _split_blocks(eval_ids, block_size)
    n_blocks = len(blocks)
    print(f"Arm A: {n_blocks} blocks × {len(windows)} windows × {len(modes)} modes"
          f" × {len(model_tags)} models", flush=True)

    all_rows: list[dict] = []
    summaries: dict[str, list[dict]] = {}

    for tag in model_tags:
        print(f"Arm A — loading {tag} …", flush=True)
        start = monotonic()
        lm = load_model(tag, device=device, revision=revision)
        rows = _arm_a_score_model(lm.model, blocks, windows, modes,
                                  torch.device(device), seed)
        for r in rows:
            r["model"] = tag
        all_rows.extend(rows)
        elapsed = monotonic() - start
        print(f"  {tag} done in {elapsed:.0f}s", flush=True)

        # Per-model summary: mean delta_ce per (mode, window).
        for mode in modes:
            for w in windows:
                subset = [r for r in rows if r["mode"] == mode and r["window"] == w]
                deltas = np.array([r["delta_ce"] for r in subset], dtype=np.float64)
                stats = _bootstrap_ci(deltas, n_resamples=bootstrap, seed=seed)
                summaries.setdefault(tag, []).append(
                    {
                        "mode": mode,
                        "window": w,
                        **stats,
                    }
                )

    # Per-item CSV.
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "rs3_local_scramble.csv").open("w", newline="") as f:
            fieldnames = [
                "model", "block", "mode", "window",
                "ce_clean", "ce_perturbed", "delta_ce",
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_rows)

    return {"arm": "local_scramble", "n_blocks": n_blocks, "summaries": summaries}


# ---------------------------------------------------------------------------
# Arm B — Induction (repeated-span retrieval)
# ---------------------------------------------------------------------------


def _generate_induction_items(
    n_seq: int,
    distances: list[int],
    span_len: int,
    vocab_lo: int,
    vocab_hi: int,
    special_ids: set[int],
    filler_ids: torch.Tensor,
    seed: int,
) -> list[dict]:
    """Generate induction sequences. Returns list of dicts with seq, first_span_pos, etc."""
    rng = random.Random(seed)
    items: list[dict] = []
    vocab = [i for i in range(vocab_lo, vocab_hi + 1) if i not in special_ids]

    for distance in distances:
        for _ in range(n_seq):
            span = [rng.choice(vocab) for _ in range(span_len)]
            # Sample filler from natural text.
            filler_start = rng.randint(0, len(filler_ids) - distance - 1)
            filler = filler_ids[filler_start:filler_start + distance].tolist()
            seq = span + filler + span
            first_span_pos = 0
            second_span_pos = span_len + distance
            items.append(
                {
                    "distance": distance,
                    "span": span,
                    "seq": seq,
                    "first_span_pos": first_span_pos,
                    "second_span_pos": second_span_pos,
                }
            )
    return items


def _arm_b_score_model(
    model: torch.nn.Module,
    items: list[dict],
    device: torch.device,
) -> list[dict]:
    """Score one model on all induction items. Returns per-sequence rows."""
    rows: list[dict] = []
    n_items = len(items)
    batch_size = ARM_B_BATCH_SIZE

    # Group items by distance (equal length within each group — no padding needed).
    by_dist: dict[int, list[dict]] = {}
    for idx, item in enumerate(items):
        by_dist.setdefault(item["distance"], []).append({"idx": idx, **item})

    for distance, group in by_dist.items():
        n = len(group)
        for g_start in range(0, n, batch_size):
            g_end = min(g_start + batch_size, n)
            sub = group[g_start:g_end]
            # Stack sequences into [B, L] batch.
            seqs = [torch.tensor(it["seq"], dtype=torch.long) for it in sub]
            batch = torch.stack(seqs).to(device)                    # [B, L]
            B, L = batch.shape

            with torch.no_grad():
                out = model(batch, use_cache=False)
                logits = out.logits                              # [B, L, V]

            span_len = len(sub[0]["span"])
            p1_start = sub[0]["first_span_pos"]
            p2_start = sub[0]["second_span_pos"]

            # CE on first span: positions [p1_start, p1_start+span_len).
            # Predictions at positions p1_start..p1_start+span_len-1
            # target positions p1_start+1..p1_start+span_len
            first_logits = logits[:, p1_start:p1_start + span_len, :]   # [B, span_len, V]
            first_labels = batch[:, p1_start + 1:p1_start + span_len + 1]  # [B, span_len]
            ce_first_per_seq = F.cross_entropy(
                first_logits.reshape(B * span_len, -1),
                first_labels.reshape(-1),
                reduction="none",
            ).view(B, span_len).mean(dim=1)                              # [B]

            # CE on second span (excl. first token, last token has no next-token label).
            n_second = span_len - 2
            second_logits = logits[:, p2_start + 1:p2_start + span_len - 1, :]  # [B, n_second, V]
            second_labels = batch[:, p2_start + 2:p2_start + span_len]          # [B, n_second]
            ce_second_per_seq = F.cross_entropy(
                second_logits.reshape(B * n_second, -1),
                second_labels.reshape(-1),
                reduction="none",
            ).view(B, n_second).mean(dim=1)                               # [B]

            ce_first_list = [float(v.item()) for v in ce_first_per_seq]
            ce_second_list = [float(v.item()) for v in ce_second_per_seq]

            for i, it in enumerate(sub):
                rows.append(
                    {
                        "seq_idx": it["idx"],
                        "distance": distance,
                        "ce_first": ce_first_list[i],
                        "ce_second": ce_second_list[i],
                        "induction_gain": ce_first_list[i] - ce_second_list[i],
                    }
                )

            print(f"    arm-B d={distance} seq {g_end}/{n}", flush=True)

    return rows


def run_induction(
    model_tags: list[str],
    tokenizer,
    *,
    eval_ids: torch.Tensor | None = None,
    distances: list[int] | None = None,
    n_seq: int = 256,
    span_len: int = 32,
    vocab_lo: int = 1000,
    vocab_hi: int = 20000,
    bootstrap: int = 10_000,
    seed: int = 0,
    device: str = "cuda",
    revision: str | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Arm B: synthetic induction retrieval gain."""
    if distances is None:
        distances = [64, 256, 512, 1024, 1536]

    # Collect special token ids to exclude from random-span sampling.
    special_ids: set[int] = set()
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        v = getattr(tokenizer, attr, None)
        if v is not None:
            special_ids.add(int(v))

    # Use the RS3 eval slice as filler source.
    if eval_ids is None:
        raise ValueError("eval_ids is required for arm B filler sampling")
    filler_ids = eval_ids

    items = _generate_induction_items(
        n_seq, distances, span_len,
        vocab_lo, vocab_hi, special_ids,
        filler_ids, seed,
    )
    print(f"Arm B: {len(items)} sequences × {len(model_tags)} models", flush=True)

    all_rows: list[dict] = []
    summaries: dict[str, list[dict]] = {}

    for tag in model_tags:
        print(f"Arm B — loading {tag} …", flush=True)
        start = monotonic()
        lm = load_model(tag, device=device, revision=revision)
        rows = _arm_b_score_model(lm.model, items, torch.device(device))
        for r in rows:
            r["model"] = tag
        all_rows.extend(rows)
        elapsed = monotonic() - start
        print(f"  {tag} done in {elapsed:.0f}s", flush=True)

        for d in distances:
            subset = [r for r in rows if r["distance"] == d]
            gains = np.array([r["induction_gain"] for r in subset], dtype=np.float64)
            stats = _bootstrap_ci(gains, n_resamples=bootstrap, seed=seed)
            summaries.setdefault(tag, []).append({"distance": d, **stats})

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "rs3_induction.csv").open("w", newline="") as f:
            fieldnames = [
                "model", "seq_idx", "distance",
                "ce_first", "ce_second", "induction_gain",
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_rows)

    return {"arm": "induction", "n_seq_per_distance": n_seq, "summaries": summaries}


# ---------------------------------------------------------------------------
# Arm C — Key–Value retrieval (secondary)
# ---------------------------------------------------------------------------


def _generate_kv_items(
    n_seq: int,
    context_lines: int,
    depths: list[int],
    vocab_lo: int,
    vocab_hi: int,
    special_ids: set[int],
    seed: int,
) -> list[dict]:
    """Generate KV-retrieval sequences.

    Each item: M lines of "<key>: <value>", then query "<target_key>:"
    where target_key is at one of the specified depths (1-indexed, 1=shallowest).
    Keys and values are 2-token random strings.
    """
    rng = random.Random(seed)
    vocab = [i for i in range(vocab_lo, vocab_hi + 1) if i not in special_ids]
    items: list[dict] = []

    for depth in depths:
        for _ in range(n_seq):
            keys: list[list[int]] = []
            values: list[list[int]] = []
            for _ in range(context_lines):
                k = [rng.choice(vocab) for _ in range(2)]
                v = [rng.choice(vocab) for _ in range(2)]
                keys.append(k)
                values.append(v)
            # Target position: depth-1 (0-indexed) from the start.
            target_idx = depth - 1  # 1-indexed → 0-indexed
            target_key = keys[target_idx]
            target_value = values[target_idx]
            items.append(
                {
                    "depth": depth,
                    "target_idx": target_idx,
                    "keys": keys,
                    "values": values,
                    "target_key": target_key,
                    "target_value": target_value,
                }
            )
    return items


def _arm_c_score_model(
    model: torch.nn.Module,
    items: list[dict],
    tokenizer,
    device: torch.device,
) -> list[dict]:
    """Score one model on KV-retrieval items.

    All items have the same length (M lines × 6 tokens + query 3 tokens),
    so we can stack them into a straight [N, L] batch — no padding needed.
    """
    rows: list[dict] = []
    colon_id = tokenizer.encode(":", add_special_tokens=False)[0]
    newline_id = tokenizer.encode("\n", add_special_tokens=False)[0]
    n_items = len(items)
    batch_size = ARM_C_BATCH_SIZE

    # Pre-build all sequences as token lists (CPU, cheap).
    all_seqs: list[list[int]] = []
    for item in items:
        seq: list[int] = []
        for ki in range(len(item["keys"])):
            seq.extend(item["keys"][ki])
            seq.append(colon_id)
            seq.extend(item["values"][ki])
            seq.append(newline_id)
        seq.extend(item["target_key"])
        seq.append(colon_id)
        all_seqs.append(seq)

    L = len(all_seqs[0])  # all items have the same length

    for s_start in range(0, n_items, batch_size):
        s_end = min(s_start + batch_size, n_items)
        sub_seqs = [torch.tensor(s, dtype=torch.long) for s in all_seqs[s_start:s_end]]
        batch = torch.stack(sub_seqs).to(device)              # [B, L]
        B = batch.shape[0]

        with torch.no_grad():
            out = model(batch, use_cache=False)
            logits = out.logits                                # [B, L, V]

        predict_pos = L - 1  # last token is ":"
        pred_logits = logits[:, predict_pos, :]                # [B, V]
        true_val0_list = [item["target_value"][0] for item in items[s_start:s_end]]
        true_val0_t = torch.tensor(true_val0_list, device=device)  # [B]

        # Top-1 accuracy per item.
        top1_per_item = (torch.argmax(pred_logits, dim=1) == true_val0_t).int()  # [B]

        # CE per item.
        ce_per_item = F.cross_entropy(pred_logits, true_val0_t, reduction="none")  # [B]

        top1_list = [int(v.item()) for v in top1_per_item]
        ce_list = [float(v.item()) for v in ce_per_item]

        for i, idx in enumerate(range(s_start, s_end)):
            rows.append(
                {
                    "seq_idx": idx,
                    "depth": items[idx]["depth"],
                    "top1_correct": top1_list[i],
                    "ce_true_value": ce_list[i],
                }
            )

        print(f"    arm-C seq {s_end}/{n_items}", flush=True)

    return rows


def run_kv_retrieval(
    model_tags: list[str],
    tokenizer,
    *,
    context_lines: int = 40,
    depths: list[int] | None = None,
    n_seq: int = 128,
    vocab_lo: int = 1000,
    vocab_hi: int = 20000,
    bootstrap: int = 10_000,
    seed: int = 0,
    device: str = "cuda",
    revision: str | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Arm C: key–value retrieval (secondary)."""
    if depths is None:
        depths = [1, 10, 20, 30, 40]

    special_ids: set[int] = set()
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        v = getattr(tokenizer, attr, None)
        if v is not None:
            special_ids.add(int(v))

    items = _generate_kv_items(
        n_seq, context_lines, depths,
        vocab_lo, vocab_hi, special_ids, seed,
    )
    print(f"Arm C: {len(items)} sequences × {len(model_tags)} models", flush=True)

    all_rows: list[dict] = []
    summaries: dict[str, list[dict]] = {}

    for tag in model_tags:
        print(f"Arm C — loading {tag} …", flush=True)
        start = monotonic()
        lm = load_model(tag, device=device, revision=revision)
        rows = _arm_c_score_model(lm.model, items, tokenizer, torch.device(device))
        for r in rows:
            r["model"] = tag
        all_rows.extend(rows)
        elapsed = monotonic() - start
        print(f"  {tag} done in {elapsed:.0f}s", flush=True)

        for d in depths:
            subset = [r for r in rows if r["depth"] == d]
            accs = np.array([r["top1_correct"] for r in subset], dtype=np.float64)
            ces = np.array([r["ce_true_value"] for r in subset], dtype=np.float64)
            summary = {"depth": d}
            summary.update(
                {f"top1_{k}": v for k, v in _bootstrap_ci(accs, bootstrap, seed).items()}
            )
            # Overwrite mean with accuracy rate.
            summary["top1_mean"] = float(np.mean(accs))
            summary["ce_mean"] = float(np.mean(ces))
            summaries.setdefault(tag, []).append(summary)

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "rs3_kv_retrieval.csv").open("w", newline="") as f:
            fieldnames = ["model", "seq_idx", "depth", "top1_correct", "ce_true_value"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_rows)

    return {"arm": "kv_retrieval", "n_seq_per_depth": n_seq, "summaries": summaries}


# ---------------------------------------------------------------------------
# Arm D — local CE-at-context helper (RS1-frozen function stays locked)
# ---------------------------------------------------------------------------


def _ce_at_context(
    model: torch.nn.Module,
    ids: torch.Tensor,
    eval_context: int,
    stride: int,
    device: torch.device,
) -> tuple[float, float, int]:
    """Token-weighted CE at arbitrary eval_context/stride.

    Unlike the RS1-frozen `token_weighted_ce_and_ppl`, this helper accepts
    any context length and stride.  It is local to RS3 Arm D only.
    """
    if ids.ndim != 1:
        raise ValueError("ids must be a 1D token tensor")

    weighted_loss = 0.0
    prediction_tokens = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, max(1, len(ids) - 1), stride):
            end = min(start + eval_context, len(ids))
            chunk = ids[start:end].unsqueeze(0).to(device)
            if chunk.shape[1] < 2:
                continue
            out = model(chunk, use_cache=False)
            logits = out.logits[:, :-1, :]
            labels = chunk[:, 1:]
            n = int(labels.numel())
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                reduction="mean",
            )
            weighted_loss += float(loss.item()) * n
            prediction_tokens += n
            if end == len(ids):
                break
    if prediction_tokens == 0:
        return float("nan"), float("nan"), 0
    ce = weighted_loss / prediction_tokens
    return ce, float(math.exp(ce)), prediction_tokens


# ---------------------------------------------------------------------------
# Arm D — Length behaviour CE (exploratory)
# ---------------------------------------------------------------------------


def run_length_ce(
    model_tags: list[str],
    eval_ids: torch.Tensor,
    *,
    contexts: list[int] | None = None,
    device: str = "cuda",
    revision: str | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Arm D: CE at varying eval context lengths (exploratory)."""
    if contexts is None:
        contexts = [1024, 2048, 4096, 8192]

    ref_ctx = 2048  # reference context for normalisation

    all_rows: list[dict] = []
    summaries: dict[str, list[dict]] = {}

    for tag in model_tags:
        print(f"Arm D — loading {tag} …", flush=True)
        start = monotonic()
        lm = load_model(tag, device=device, revision=revision)
        model_device = next(lm.model.parameters()).device
        # Use at most 500K tokens total; truncate eval_ids to max context.
        max_len = min(len(eval_ids), 500_000)
        ids = eval_ids[:max_len]

        for ctx in contexts:
            ce, ppl, pred_tokens = _ce_at_context(
                lm.model, ids, eval_context=ctx, stride=ctx, device=model_device,
            )
            all_rows.append(
                {
                    "model": tag,
                    "eval_context": ctx,
                    "token_weighted_ce": ce,
                    "perplexity": ppl,
                    "prediction_tokens": pred_tokens,
                }
            )
            print(f"  {tag} ctx={ctx} ce={ce:.6f} ppl={ppl:.4f} ({pred_tokens} tok)",
                  flush=True)

        elapsed = monotonic() - start
        print(f"  {tag} done in {elapsed:.0f}s", flush=True)

        # Normalise to reference context.
        ref_ce = next(
            r["token_weighted_ce"] for r in all_rows
            if r["model"] == tag and r["eval_context"] == ref_ctx
        )
        for r in all_rows:
            if r["model"] == tag:
                r["ce_normalised"] = r["token_weighted_ce"] / ref_ce

        summaries.setdefault(tag, []).append(
            {ctx: r["ce_normalised"] for r in all_rows if r["model"] == tag
             for ctx_ in [r["eval_context"]] if ctx_ == ctx}
        )
        # Build a proper summary list.
        tag_summaries = []
        for r in all_rows:
            if r["model"] == tag:
                tag_summaries.append(
                    {
                        "eval_context": r["eval_context"],
                        "token_weighted_ce": r["token_weighted_ce"],
                        "perplexity": r["perplexity"],
                        "ce_normalised": r["ce_normalised"],
                    }
                )
        summaries[tag] = tag_summaries

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "rs3_length_ce.csv").open("w", newline="") as f:
            fieldnames = [
                "model", "eval_context", "token_weighted_ce",
                "perplexity", "prediction_tokens", "ce_normalised",
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_rows)

    return {"arm": "length_ce", "reference_context": ref_ctx, "summaries": summaries}


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------


def _check_gate_rs31(
    model_tags: list[str],
    eval_ids: torch.Tensor,
    *,
    device: str,
    revision: str | None,
    target_ce: dict[str, float],
    tolerance: dict[str, float],
) -> dict:
    """G-RS3.1: harness reproduces known CE on frozen 5M-token prefix."""
    results: dict[str, dict] = {}
    for tag in model_tags:
        if tag not in target_ce:
            continue
        lm = load_model(tag, device=device, revision=revision)
        ce, ppl, pred_tokens = token_weighted_ce_and_ppl(lm.model, eval_ids)
        target = target_ce[tag]
        tol = tolerance.get(tag, 0.05)
        passed = abs(ce - target) <= tol
        results[tag] = {
            "target_ce": target,
            "measured_ce": ce,
            "ppl": ppl,
            "tol": tol,
            "passed": passed,
        }
        status = "PASS" if passed else "FAIL"
        print(f"G-RS3.1 {tag}: target={target} measured={ce:.6f} tol={tol} → {status}",
              flush=True)
    return results


def _gates_from_arm_a(summaries: dict) -> dict:
    """G-RS3.2: CE_perturbed > CE_clean, delta non-decreasing in w.

    Returns a dict with per-model gate results.
    """
    gate = {}
    for tag, entries in summaries.items():
        # Collect perturbation deltas by mode and window.
        by_mode: dict[str, dict[int, float]] = {}
        for e in entries:
            by_mode.setdefault(e["mode"], {})[e["window"]] = e["mean"]

        monotonic_pass = True
        all_positive = True
        for mode, wd in by_mode.items():
            sorted_ws = sorted(wd.keys())
            for i in range(1, len(sorted_ws)):
                if wd[sorted_ws[i]] < wd[sorted_ws[i - 1]]:
                    monotonic_pass = False
            for w, d in wd.items():
                if d <= 0:
                    all_positive = False

        gate[tag] = {
            "all_positive": all_positive,
            "monotonic_non_decreasing": monotonic_pass,
            "passed": all_positive and monotonic_pass,
        }
    return gate


def _gates_from_arm_b(summaries: dict) -> dict:
    """G-RS3.3: qwen3 shows induction_gain > 0.5 at d=64, CI excludes zero."""
    gate = {}
    for tag, entries in summaries.items():
        if tag != "qwen3":
            continue
        for e in entries:
            if e["distance"] == 64:
                passed = e["mean"] > 0.5 and e["ci_lower"] > 0
                gate[tag] = {
                    "gain_at_d64": e["mean"],
                    "ci_lower": e["ci_lower"],
                    "passed": passed,
                }
                break
    return gate


def _gates_from_arm_c(summaries: dict, depths: list[int]) -> dict:
    """G-RS3.4: qwen3 top-1 accuracy at shallowest depth ≥ 3× chance.

    Chance rate ≈ 1/vocab for the parameter band; we use 1/19000 as a conservative
    estimate since the sampled vocab range is [1000, 20000) minus specials.
    """
    chance = 1.0 / 19000
    gate = {}
    for tag, entries in summaries.items():
        if tag != "qwen3":
            continue
        shallowest = min(depths)
        for e in entries:
            if e["depth"] == shallowest:
                acc = e["top1_mean"]
                passed = acc >= 3 * chance
                gate[tag] = {
                    "shallowest_depth": shallowest,
                    "top1_accuracy": acc,
                    "chance_rate": chance,
                    "threshold_3x": 3 * chance,
                    "passed": passed,
                }
                break
    return gate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RS3 — Functional locus of RoPE (local-order vs. retrieval)"
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=["local_scramble", "induction", "kv_retrieval", "length_ce", "gates"],
        help="Which arm to run, or 'gates' for G-RS3.1 only.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen3", "qwen3-droped", "qwen3-dropped"],
        choices=sorted(MODEL_IDS),
    )
    parser.add_argument("--eval-tokens", type=int, default=1_000_000)
    parser.add_argument(
        "--eval-offset-tokens",
        type=int,
        default=RS3_EVAL_OFFSET_DEFAULT,
        help="Tokens to skip past RS1a's eval slice (default: 5M).  Set to 0 to "
             "use the RS1a prefix for G-RS3.1.",
    )
    parser.add_argument("--windows", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    parser.add_argument("--modes", nargs="+", default=["scramble", "reverse"])
    parser.add_argument("--distances", type=int, nargs="+",
                        default=[64, 256, 512, 1024, 1536])
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 10, 20, 30, 40])
    parser.add_argument("--contexts", type=int, nargs="+",
                        default=[1024, 2048, 4096, 8192])
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer once from the first model.
    tokenizer_lm = load_model(args.models[0], device="cpu", revision=args.revision)

    run_id = out_dir.name
    summary: dict = {
        "experiment": "RS3",
        "run_id": run_id,
        "task": args.task,
        "models": args.models,
        "revision": args.revision,
        "eval_offset_tokens": args.eval_offset_tokens,
        "seed": args.seed,
        "bootstrap_resamples": args.bootstrap,
        "device": args.device,
        "gates": {},
        "arms": {},
    }

    if args.task == "gates":
        # G-RS3.1 only: reproduce known CE on the original 5M-token prefix.
        ids = fineweb_edu_eval_ids(
            tokenizer_lm.tokenizer,
            eval_tokens=5_000_000,
            offset_tokens=0,
        )
        target_ce = {"qwen3": 3.0819}
        tolerance = {"qwen3": 0.005}
        # For qwen3-droped we use the bf16→fp32 expected shift tolerance.
        if "qwen3-droped" in args.models:
            target_ce["qwen3-droped"] = 2.826
            tolerance["qwen3-droped"] = 0.05
        gate_results = _check_gate_rs31(
            args.models, ids,
            device=args.device,
            revision=args.revision,
            target_ce=target_ce,
            tolerance=tolerance,
        )
        summary["gates"]["G-RS3.1"] = gate_results
    else:
        # Load the RS3 eval slice (offset past RS1a's prefix by default).
        eval_ids = fineweb_edu_eval_ids(
            tokenizer_lm.tokenizer,
            eval_tokens=args.eval_tokens,
            offset_tokens=args.eval_offset_tokens,
        )
        print(f"Eval slice: {eval_ids.numel()} tokens "
              f"(offset={args.eval_offset_tokens}, requested={args.eval_tokens})",
              flush=True)

        if args.task == "local_scramble":
            arm_result = run_local_scramble(
                args.models,
                eval_ids,
                windows=args.windows,
                modes=args.modes,
                bootstrap=args.bootstrap,
                seed=args.seed,
                device=args.device,
                revision=args.revision,
                output_dir=out_dir,
            )
            summary["arms"]["local_scramble"] = arm_result
            # Gate evaluation.
            summary["gates"]["G-RS3.2"] = _gates_from_arm_a(arm_result["summaries"])

        elif args.task == "induction":
            arm_result = run_induction(
                args.models,
                tokenizer_lm.tokenizer,
                eval_ids=eval_ids,
                distances=args.distances,
                n_seq=256,
                bootstrap=args.bootstrap,
                seed=args.seed,
                device=args.device,
                revision=args.revision,
                output_dir=out_dir,
            )
            summary["arms"]["induction"] = arm_result
            summary["gates"]["G-RS3.3"] = _gates_from_arm_b(arm_result["summaries"])

        elif args.task == "kv_retrieval":
            arm_result = run_kv_retrieval(
                args.models,
                tokenizer_lm.tokenizer,
                depths=args.depths,
                n_seq=128,
                bootstrap=args.bootstrap,
                seed=args.seed,
                device=args.device,
                revision=args.revision,
                output_dir=out_dir,
            )
            summary["arms"]["kv_retrieval"] = arm_result
            summary["gates"]["G-RS3.4"] = _gates_from_arm_c(
                arm_result["summaries"], args.depths,
            )

        elif args.task == "length_ce":
            arm_result = run_length_ce(
                args.models,
                eval_ids,
                contexts=args.contexts,
                device=args.device,
                revision=args.revision,
                output_dir=out_dir,
            )
            summary["arms"]["length_ce"] = arm_result

    # Write summary JSON.
    summary["cli_flags"] = vars(args)
    # Convert Path to str for JSON.
    summary["cli_flags"]["output_dir"] = str(summary["cli_flags"]["output_dir"])
    (out_dir / "rs3_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(f"Summary written to {out_dir / 'rs3_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
