from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

from deadkeys.common.loading import set_qwen_rotary_identity

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_perplexity import (  # noqa: E402
    DATASET_NAME,
    DATASET_PATH,
    DATASET_SPLIT,
    EVAL_CONTEXT,
    EVAL_SLICE_RULE,
    token_weighted_ce_and_ppl,
)


@dataclass(frozen=True)
class TrainConfig:
    base_model: str
    base_revision: str
    dataset_path: str
    dataset_name: str
    dataset_split: str
    eval_slice_rule: str
    train_context: int
    eval_tokens: int
    train_tokens: int
    global_batch_tokens: int
    micro_batch_size: int
    learning_rate: float
    min_lr_fraction: float
    warmup_fraction: float
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_eps: float
    grad_clip: float
    seed: int
    dtype: str


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_revision(model_id: str, revision: str | None) -> str:
    info = HfApi().model_info(model_id, revision=revision)
    if not info.sha:
        raise RuntimeError(f"could not resolve revision SHA for {model_id}")
    return str(info.sha)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rotary_identity_probe(model: torch.nn.Module, device: torch.device, *, seq_len: int) -> dict[str, float | bool]:
    input_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    pos = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    emb = model.model.embed_tokens(input_ids)
    cos, sin = model.model.rotary_emb(emb, pos)
    identity_cos_max = float((cos - 1).abs().max().item())
    identity_sin_max = float(sin.abs().max().item())
    set_qwen_rotary_identity(model, enabled=False)
    try:
        cos_true, sin_true = model.model.rotary_emb(emb, pos)
        true_rope_delta = float(max((cos_true - 1).abs().max().item(), sin_true.abs().max().item()))
    finally:
        set_qwen_rotary_identity(model, enabled=True)
    return {
        "identity_cos_max_abs_error": identity_cos_max,
        "identity_sin_max_abs_error": identity_sin_max,
        "true_rope_max_abs_delta_from_identity": true_rope_delta,
        "pass": identity_cos_max == 0.0 and identity_sin_max == 0.0 and true_rope_delta > 0.0,
    }


def build_or_reuse_token_cache(tokenizer, cache_dir: Path, *, eval_tokens: int, train_tokens: int) -> tuple[Path, Path]:
    from datasets import load_dataset

    cache_dir.mkdir(parents=True, exist_ok=True)
    eval_path = cache_dir / f"fineweb_edu_qwen3_eval_{eval_tokens}.uint32"
    train_path = cache_dir / f"fineweb_edu_qwen3_train_after_eval{eval_tokens}_{train_tokens}.uint32"
    if eval_path.exists() and train_path.exists():
        if eval_path.stat().st_size == eval_tokens * 4 and train_path.stat().st_size == train_tokens * 4:
            return eval_path, train_path

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise RuntimeError("Qwen3 tokenizer must have eos_token_id for RS1 packing")
    total_needed = eval_tokens + train_tokens
    eval_mm = np.memmap(eval_path, mode="w+", dtype=np.uint32, shape=(eval_tokens,))
    train_mm = np.memmap(train_path, mode="w+", dtype=np.uint32, shape=(train_tokens,))
    total = 0
    ds = load_dataset(DATASET_PATH, DATASET_NAME, split=DATASET_SPLIT, streaming=True)
    for row in ds:
        text = str(row.get("text") or "")
        if not text.strip():
            continue
        ids = tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"]
        if not ids:
            continue
        ids.append(int(eos_id))
        arr = np.asarray(ids, dtype=np.uint32)
        offset = 0
        while offset < len(arr) and total < total_needed:
            if total < eval_tokens:
                n = min(len(arr) - offset, eval_tokens - total)
                eval_mm[total : total + n] = arr[offset : offset + n]
            else:
                train_pos = total - eval_tokens
                n = min(len(arr) - offset, train_tokens - train_pos)
                train_mm[train_pos : train_pos + n] = arr[offset : offset + n]
            total += n
            offset += n
        if total >= total_needed:
            break
        if total and total % 10_000_000 < len(arr):
            print(f"cache_progress tokens={total}/{total_needed}", flush=True)
    eval_mm.flush(); train_mm.flush()
    if total < total_needed:
        raise RuntimeError(f"dataset stream ended after {total}; need {total_needed}")
    return eval_path, train_path


def batch_from_memmap(train_mm: np.memmap, start_block: int, *, micro_batch_size: int, context: int, device: torch.device) -> torch.Tensor:
    start = start_block * context
    stop = start + micro_batch_size * context
    arr = np.asarray(train_mm[start:stop], dtype=np.int64).reshape(micro_batch_size, context)
    return torch.from_numpy(arr).to(device=device, dtype=torch.long)


def evaluate_cached(model: torch.nn.Module, eval_path: Path, *, eval_tokens: int) -> tuple[float, float, int]:
    eval_mm = np.memmap(eval_path, mode="r", dtype=np.uint32, shape=(eval_tokens,))
    ids = torch.from_numpy(np.asarray(eval_mm, dtype=np.int64))
    return token_weighted_ce_and_ppl(model, ids, eval_context=EVAL_CONTEXT, stride=EVAL_CONTEXT)


def write_training_report(out_dir: Path, manifest: dict, rows: list[dict]) -> None:
    last = rows[-1] if rows else {}
    lines = [
        "# RS1b Qwen3 NoPE training report",
        "",
        f"Generated: {utc_now()}",
        "",
        "This is a training/artifact-generation report, not a reproducible-research notebook entry.",
        "",
        f"- base model: `{manifest['config']['base_model']}`",
        f"- base revision: `{manifest['config']['base_revision']}`",
        f"- output dir: `{manifest['output_dir']}`",
        f"- train tokens requested: {manifest['config']['train_tokens']}",
        f"- eval slice rule: {manifest['config']['eval_slice_rule']}",
        f"- final row: `{last}`",
        "",
        "Artifacts: `training_manifest.json`, `training_metrics.csv`, final bf16 model files, tokenizer files.",
    ]
    (out_dir / "training_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Qwen3-0.6B through identity RoPE for RS1b qwen3-droped")
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("RS1B_TOKEN_CACHE", "/workspace/rs1b-token-cache")))
    parser.add_argument("--eval-tokens", type=int, default=5_000_000)
    parser.add_argument("--train-tokens", type=int, required=True)
    parser.add_argument("--train-context", type=int, default=2048)
    parser.add_argument("--global-batch-tokens", type=int, default=524288)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--warmup-fraction", type=float, default=0.02)
    parser.add_argument("--min-lr-fraction", type=float, default=0.10)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every-steps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.train_context != EVAL_CONTEXT:
        raise ValueError("RS1b training context is pinned at 2048")
    if args.global_batch_tokens % args.train_context != 0:
        raise ValueError("global_batch_tokens must be divisible by train_context")

    seed_everything(args.seed)
    device = torch.device(args.device)
    base_sha = resolve_revision(args.base_model, args.revision)
    rev_kw = {"revision": base_sha}
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, **rev_kw)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, **rev_kw)
    set_qwen_rotary_identity(model, enabled=True)
    model.to(device)
    model.train()

    probe = rotary_identity_probe(model, device, seq_len=min(args.train_context, 32))
    if not probe["pass"]:
        raise RuntimeError(f"rotary identity probe failed: {probe}")

    eval_path, train_path = build_or_reuse_token_cache(tokenizer, args.cache_dir, eval_tokens=args.eval_tokens, train_tokens=args.train_tokens)
    train_blocks = args.train_tokens // args.train_context
    micro_tokens = args.micro_batch_size * args.train_context
    grad_accum = max(1, math.ceil(args.global_batch_tokens / micro_tokens))
    steps_available = train_blocks // (args.micro_batch_size * grad_accum)
    total_steps = min(steps_available, args.max_steps) if args.max_steps else steps_available
    if total_steps < 1:
        raise ValueError("token budget is too small for one optimizer step")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=args.weight_decay)
    warmup_steps = max(1, int(total_steps * args.warmup_fraction))
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return args.min_lr_fraction + (1.0 - args.min_lr_fraction) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    train_mm = np.memmap(train_path, mode="r", dtype=np.uint32, shape=(args.train_tokens,))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = TrainConfig(
        base_model=args.base_model,
        base_revision=base_sha,
        dataset_path=DATASET_PATH,
        dataset_name=DATASET_NAME,
        dataset_split=DATASET_SPLIT,
        eval_slice_rule=EVAL_SLICE_RULE + "; training starts immediately after that prefix",
        train_context=args.train_context,
        eval_tokens=args.eval_tokens,
        train_tokens=args.train_tokens,
        global_batch_tokens=args.global_batch_tokens,
        micro_batch_size=args.micro_batch_size,
        learning_rate=args.lr,
        min_lr_fraction=args.min_lr_fraction,
        warmup_fraction=args.warmup_fraction,
        weight_decay=args.weight_decay,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        grad_clip=args.grad_clip,
        seed=args.seed,
        dtype="bf16",
    )
    manifest = {
        "artifact": "qwen3-droped",
        "created_at": utc_now(),
        "output_dir": str(args.output_dir),
        "token_cache": {"eval_path": str(eval_path), "train_path": str(train_path)},
        "rotary_identity_probe": probe,
        "config": asdict(config),
        "total_steps": total_steps,
        "grad_accumulation_steps": grad_accum,
        "hardware": {"device": str(device), "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
    }
    (args.output_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    rows: list[dict] = []
    metrics_path = args.output_dir / "training_metrics.csv"
    start_time = monotonic()
    block_cursor = 0
    fieldnames = ["step", "tokens", "train_loss", "eval_ce", "eval_ppl", "lr", "elapsed_s", "tokens_per_s"]
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for step in range(1, total_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for _ in range(grad_accum):
                batch = batch_from_memmap(train_mm, block_cursor, micro_batch_size=args.micro_batch_size, context=args.train_context, device=device)
                block_cursor += args.micro_batch_size
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    out = model(input_ids=batch, labels=batch, use_cache=False)
                    loss = out.loss / grad_accum
                loss.backward()
                loss_sum += float(loss.detach().item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            do_eval = step == 1 or step == total_steps or (args.eval_every_steps and step % args.eval_every_steps == 0)
            eval_ce = eval_ppl = float("nan")
            if do_eval:
                model.eval()
                eval_ce, eval_ppl, _ = evaluate_cached(model, eval_path, eval_tokens=args.eval_tokens)
                model.train()
            elapsed = monotonic() - start_time
            tokens = min(block_cursor * args.train_context, args.train_tokens)
            row = {
                "step": step,
                "tokens": tokens,
                "train_loss": loss_sum,
                "eval_ce": eval_ce,
                "eval_ppl": eval_ppl,
                "lr": scheduler.get_last_lr()[0],
                "elapsed_s": elapsed,
                "tokens_per_s": tokens / elapsed if elapsed > 0 else float("nan"),
            }
            writer.writerow(row); f.flush()
            rows.append(row)
            print(
                f"step={step}/{total_steps} tokens={tokens} train_loss={loss_sum:.6f} "
                f"eval_ppl={eval_ppl:.6f} tok_s={row['tokens_per_s']:.1f}",
                flush=True,
            )

    model.eval()
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    write_training_report(args.output_dir, manifest, rows)


if __name__ == "__main__":
    main()
