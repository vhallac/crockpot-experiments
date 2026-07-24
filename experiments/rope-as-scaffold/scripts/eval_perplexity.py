from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic

import torch
import torch.nn.functional as F

from deadkeys.common.loading import MODEL_IDS, load_model, uses_dropped_rope

DATASET_PATH = "HuggingFaceFW/fineweb-edu"
DATASET_NAME = "sample-10BT"
DATASET_SPLIT = "train"
DATASET_STREAMING = True
EVAL_CONTEXT = 2048
STRIDE = EVAL_CONTEXT
DEFAULT_EVAL_TOKENS = 5_000_000
EVAL_SLICE_RULE = (
    "stream HuggingFaceFW/fineweb-edu sample-10BT train in provider order; "
    "concatenate non-empty document text with one EOS token after each document; "
    "use the first eval_tokens tokens as the held-out eval slice"
)


@dataclass(frozen=True)
class EvalConfig:
    dataset_path: str = DATASET_PATH
    dataset_name: str = DATASET_NAME
    dataset_split: str = DATASET_SPLIT
    dataset_streaming: bool = DATASET_STREAMING
    eval_context: int = EVAL_CONTEXT
    stride: int = STRIDE
    eval_slice_rule: str = EVAL_SLICE_RULE
    ce_reduction: str = "token_weighted_mean_cross_entropy_then_exp"


@dataclass(frozen=True)
class PerplexityResult:
    model: str
    hf_id: str
    revision: str | None
    dropped_rope: bool
    eval_tokens: int
    prediction_tokens: int
    token_weighted_ce: float
    perplexity: float
    elapsed_s: float


def fineweb_edu_eval_ids(tokenizer, *, eval_tokens: int) -> torch.Tensor:
    """Return the deterministic RS1 held-out eval prefix as token ids."""
    from datasets import load_dataset

    if eval_tokens < 2:
        raise ValueError("eval_tokens must be at least 2")
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise RuntimeError("RS1 FineWeb-Edu packing requires a tokenizer eos_token_id")

    ds = load_dataset(DATASET_PATH, DATASET_NAME, split=DATASET_SPLIT, streaming=DATASET_STREAMING)
    chunks: list[torch.Tensor] = []
    total = 0
    for row in ds:
        text = str(row.get("text") or "")
        if not text.strip():
            continue
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False, verbose=False)["input_ids"][0]
        if ids.numel() == 0:
            continue
        ids = torch.cat([ids, torch.tensor([eos_id], dtype=ids.dtype)])
        chunks.append(ids)
        total += int(ids.numel())
        if total >= eval_tokens:
            return torch.cat(chunks)[:eval_tokens]
    raise RuntimeError(f"FineWeb-Edu stream ended after {total} tokens; need {eval_tokens}")


def token_weighted_ce_and_ppl(
    model: torch.nn.Module,
    ids: torch.Tensor,
    *,
    eval_context: int = EVAL_CONTEXT,
    stride: int = STRIDE,
) -> tuple[float, float, int]:
    """Frozen RS1 perplexity: one token-weighted CE definition for every state."""
    if eval_context != EVAL_CONTEXT or stride != STRIDE:
        raise ValueError(f"RS1 eval is frozen at eval_context={EVAL_CONTEXT}, stride={STRIDE}")
    if ids.ndim != 1:
        raise ValueError("ids must be a 1D token tensor")

    device = next(model.parameters()).device
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
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="mean")
            weighted_loss += float(loss.item()) * n
            prediction_tokens += n
            if end == len(ids):
                break
    if prediction_tokens == 0:
        return float("nan"), float("nan"), 0
    ce = weighted_loss / prediction_tokens
    return ce, float(math.exp(ce)), prediction_tokens


def evaluate_model(model_tag: str, ids: torch.Tensor, *, device: str, revision: str | None) -> PerplexityResult:
    start = monotonic()
    lm = load_model(model_tag, device=device, revision=revision)
    ce, ppl, prediction_tokens = token_weighted_ce_and_ppl(lm.model, ids)
    return PerplexityResult(
        model=model_tag,
        hf_id=lm.hf_id,
        revision=revision,
        dropped_rope=uses_dropped_rope(model_tag),
        eval_tokens=int(ids.numel()),
        prediction_tokens=prediction_tokens,
        token_weighted_ce=ce,
        perplexity=ppl,
        elapsed_s=monotonic() - start,
    )


def write_outputs(out_dir: Path, results: list[PerplexityResult], *, eval_tokens: int, models: list[str], revision: str | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    config = EvalConfig()
    manifest = {
        "experiment": "RS1.a",
        "task": "frozen_perplexity_eval",
        "models": models,
        "revision": revision,
        "requested_eval_tokens": eval_tokens,
        "eval_config": asdict(config),
        "results": [asdict(r) for r in results],
    }
    (out_dir / "rs1_eval_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with (out_dir / "rs1_perplexity.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for row in results:
            writer.writerow(asdict(row))


def main() -> None:
    parser = argparse.ArgumentParser(description="RS1 frozen FineWeb-Edu perplexity eval")
    parser.add_argument("--models", nargs="+", default=["qwen3", "qwen3-dropped"], choices=sorted(MODEL_IDS))
    parser.add_argument("--eval-tokens", type=int, default=DEFAULT_EVAL_TOKENS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not args.models:
        raise ValueError("at least one model is required")

    tokenizer_lm = load_model(args.models[0], device="cpu", revision=args.revision)
    ids = fineweb_edu_eval_ids(tokenizer_lm.tokenizer, eval_tokens=args.eval_tokens)
    del tokenizer_lm

    results: list[PerplexityResult] = []
    for model_tag in args.models:
        result = evaluate_model(model_tag, ids, device=args.device, revision=args.revision)
        results.append(result)
        print(
            f"model={result.model} eval_tokens={result.eval_tokens} "
            f"prediction_tokens={result.prediction_tokens} ce={result.token_weighted_ce:.6f} "
            f"ppl={result.perplexity:.6f} elapsed_s={result.elapsed_s:.1f}",
            flush=True,
        )
    write_outputs(args.output_dir, results, eval_tokens=args.eval_tokens, models=list(args.models), revision=args.revision)


if __name__ == "__main__":
    main()
