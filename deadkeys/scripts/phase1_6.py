from __future__ import annotations

import argparse
import contextlib
import math
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from deadkeys.common.loading import load_model
from deadkeys.scripts.plane_truncation import capture_qk_norms, get_text, select_planes

# Phase 1.6 is the causal "remove" follow-up to the geometric Phase 1 / 1.5
# evidence.  Measurement question: if Qwen3 has QK directions that look dead or
# low-impact, are they behaviorally dispensable?  The intended ordering is
# remove(dead) << remove(random) << remove(live), where "<<" means smaller
# perplexity delta, KL/logit drift, retrieval degradation, and attention shift.
#
# This script keeps the intervention GPU-resident: instead of CPU-side weight
# surgery or materializing attention maps, it installs forward hooks on q_proj
# and k_proj and masks whole RoPE planes in the projection outputs.  That is the
# runtime equivalent of removing selected 2D rotary planes from QK interaction,
# while preserving value/output paths.


@dataclass(frozen=True)
class PlaneChoice:
    layer: int
    head: int
    kv_head: int
    planes: tuple[int, ...]


def _plane_scores(lm, captures, device: torch.device) -> dict[tuple[int, int], torch.Tensor]:
    group = lm.n_heads // lm.n_kv_heads
    scores: dict[tuple[int, int], torch.Tensor] = {}
    for li in range(lm.n_layers):
        if (li, "q_norms") not in captures:
            continue
        q_norms = captures[(li, "q_norms")].to(device=device, dtype=torch.float32, non_blocking=True)
        k_norms = captures[(li, "k_norms")].to(device=device, dtype=torch.float32, non_blocking=True)
        q_p999 = torch.quantile(q_norms, 0.999, dim=0)
        k_max = k_norms.amax(dim=0)
        for h in range(lm.n_heads):
            kv = h // group
            lo = kv * group
            hi = min(lo + group, lm.n_heads)
            scores[(li, h)] = q_p999[lo:hi].amax(dim=0) * k_max[kv] / math.sqrt(lm.d_head)
    return scores


def choose_planes(lm, captures, *, mode: str, fraction: float, seed: int, device: torch.device) -> list[PlaneChoice]:
    rng = random.Random(seed)
    n_planes = lm.d_head // 2
    count = max(1, min(n_planes, int(round(fraction * n_planes))))
    group = lm.n_heads // lm.n_kv_heads
    scores = _plane_scores(lm, captures, device)
    choices: list[PlaneChoice] = []
    for li in range(lm.n_layers):
        for h in range(lm.n_heads):
            if (li, h) not in scores:
                continue
            if mode == "dead":
                planes = torch.argsort(scores[(li, h)])[:count].tolist()
            elif mode == "live":
                planes = torch.argsort(scores[(li, h)], descending=True)[:count].tolist()
            elif mode == "random":
                planes = sorted(rng.sample(range(n_planes), count))
            else:
                raise ValueError(f"unknown mode {mode!r}")
            choices.append(PlaneChoice(li, h, h // group, tuple(int(p) for p in planes)))
    return choices


@contextlib.contextmanager
def removed_planes(lm, choices: list[PlaneChoice]):
    layers = lm.model.model.layers
    by_layer_q: dict[int, dict[int, tuple[int, ...]]] = {}
    by_layer_k: dict[int, dict[int, set[int]]] = {}
    for c in choices:
        by_layer_q.setdefault(c.layer, {})[c.head] = c.planes
        # For GQA, a K plane is removed only if every Q head sharing that KV head
        # selected it; this avoids over-removing shared keys for one query head.
        by_layer_k.setdefault(c.layer, {}).setdefault(c.kv_head, set(c.planes))
        by_layer_k[c.layer][c.kv_head] &= set(c.planes)
    handles = []
    for li, layer in enumerate(layers):
        q_sel = by_layer_q.get(li, {})
        k_sel = by_layer_k.get(li, {})
        if q_sel:
            def q_hook(_module, _inputs, output, q_sel=q_sel):
                y = output.clone()
                for h, planes in q_sel.items():
                    for p in planes:
                        y[..., h * lm.d_head + 2 * p : h * lm.d_head + 2 * p + 2] = 0
                return y
            handles.append(layer.self_attn.q_proj.register_forward_hook(q_hook))
        if k_sel:
            def k_hook(_module, _inputs, output, k_sel=k_sel):
                y = output.clone()
                for kv, planes in k_sel.items():
                    for p in planes:
                        y[..., kv * lm.d_head + 2 * p : kv * lm.d_head + 2 * p + 2] = 0
                return y
            handles.append(layer.self_attn.k_proj.register_forward_hook(k_hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def eval_metrics(model, ids: torch.Tensor, *, stride: int, forward_tokens: int, base_logits: list[torch.Tensor] | None = None):
    losses, weights, logits_out = [], [], []
    kls, max_abs, top1 = [], [], []
    max_len = min(int(getattr(model.config, "max_position_embeddings", 4096)), forward_tokens)
    device = next(model.parameters()).device
    with torch.inference_mode():
        idx = 0
        for start in range(0, max(1, len(ids) - 1), stride):
            end = min(start + max_len, len(ids))
            chunk = ids[start:end].unsqueeze(0).to(device)
            if chunk.shape[1] < 2:
                continue
            logits = model(chunk, use_cache=False).logits[:, :-1].float()
            labels = chunk[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="mean")
            n = int(labels.numel())
            losses.append(float(loss.item()) * n); weights.append(n)
            if base_logits is None:
                logits_out.append(logits.detach())
            else:
                ref = base_logits[idx].to(device)
                kls.append(float(F.kl_div(F.log_softmax(logits, -1), F.softmax(ref, -1), reduction="batchmean").item()))
                max_abs.append(float((logits - ref).abs().max().item()))
                top1.append(float((logits.argmax(-1) == ref.argmax(-1)).float().mean().item()))
            idx += 1
            if end == len(ids):
                break
    ppl = float(math.exp(sum(losses) / sum(weights))) if weights else float("nan")
    return ppl, logits_out, {"kl": sum(kls) / len(kls) if kls else 0.0, "max_logit_delta": max(max_abs) if max_abs else 0.0, "top1_agreement": sum(top1) / len(top1) if top1 else 1.0}


def run(args: argparse.Namespace) -> None:
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    lm = load_model(args.model, device=device)
    if lm.tag != "qwen3":
        raise NotImplementedError("Phase 1.6 is currently scoped to qwen3")
    if args.eval_tokens < 200_000 and not args.allow_smoke_under_200k:
        raise RuntimeError("Phase 1.6 requires >=200k eval tokens; pass --allow-smoke-under-200k for smoke tests")
    ids = get_text(lm.tokenizer, max(args.eval_tokens, args.calibration_tokens))
    captures = capture_qk_norms(lm, ids[:args.calibration_tokens], args.calibration_tokens, args.limit_layers, args.forward_tokens, args.observed_tokens)
    plan = select_planes(lm, captures, tuple(args.epsilons), args.observed_tokens, device=device)
    plan.to_csv(out / "phase1_6_plane_selection_qwen3.csv", index=False)
    eval_ids = ids[:args.eval_tokens]
    base_ppl, base_logits, _ = eval_metrics(lm.model, eval_ids, stride=args.stride, forward_tokens=args.forward_tokens)
    rows = [{"condition": "base", "fraction": 0.0, "ppl": base_ppl, "ppl_delta": 0.0, "kl": 0.0, "max_logit_delta": 0.0, "top1_agreement": 1.0}]
    for frac in args.fractions:
        for mode in ("dead", "random", "live"):
            choices = choose_planes(lm, captures, mode=mode, fraction=frac, seed=args.seed, device=device)
            with removed_planes(lm, choices):
                ppl, _, drift = eval_metrics(lm.model, eval_ids, stride=args.stride, forward_tokens=args.forward_tokens, base_logits=base_logits)
            rows.append({"condition": f"remove_{mode}", "fraction": frac, "ppl": ppl, "ppl_delta": ppl - base_ppl, **drift})
            pd.DataFrame(rows).to_csv(out / "phase1_6_removal_qwen3.csv", index=False)
    print(f"wrote Phase 1.6 outputs to {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1.6 Qwen3 causal removal of dead/random/live RoPE planes")
    p.add_argument("--model", default="qwen3", choices=["qwen3"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default="outputs/phase1_6_qwen3")
    p.add_argument("--calibration-tokens", type=int, default=100_000)
    p.add_argument("--eval-tokens", type=int, default=200_000)
    p.add_argument("--stride", type=int, default=512)
    p.add_argument("--forward-tokens", type=int, default=1024)
    p.add_argument("--observed-tokens", type=int, default=10_000)
    p.add_argument("--epsilons", type=float, nargs="+", default=[0.05, 0.1, 0.5, 1.0])
    p.add_argument("--fractions", type=float, nargs="+", default=[0.10, 0.25, 0.50])
    p.add_argument("--limit-layers", type=int)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--allow-smoke-under-200k", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
