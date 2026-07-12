from __future__ import annotations

import argparse
import contextlib
from pathlib import Path

import pandas as pd
import torch

from deadkeys.common.loading import load_model
from deadkeys.scripts.phase1_6 import choose_planes, eval_metrics
from deadkeys.scripts.plane_truncation import capture_qk_norms, get_text, select_planes

# Phase 1.7 is the causal "perturb" follow-up.  Removal asks whether Qwen3 can
# live without suspected dead QK directions; perturbation asks whether those
# directions are ignored even when corrupted.  The expected sensitivity curve is
# perturb(dead) << perturb(random) << perturb(live), measured by perplexity
# delta, KL/logit drift, retrieval degradation, and attention redistribution.
#
# Perturbations are installed as GPU-resident q_proj/k_proj forward hooks.  The
# hook edits whole 2D RoPE planes in projection output tensors, so the heavy path
# remains model forward passes on CUDA/ROCm rather than CPU-side activation loops.
# Supported corruptions: additive norm-scaled Gaussian noise, sign flip, and a
# 90-degree within-plane rotation that preserves plane norm but destroys
# orientation.


def _edit_planes(y: torch.Tensor, *, width: int, selected: dict[int, tuple[int, ...]], mode: str, alpha: float, generator: torch.Generator | None) -> torch.Tensor:
    out = y.clone()
    for head, planes in selected.items():
        for p in planes:
            sl = slice(head * width + 2 * p, head * width + 2 * p + 2)
            part = out[..., sl]
            if mode == "noise":
                rms = part.float().pow(2).mean().sqrt().to(dtype=part.dtype).clamp_min(torch.finfo(part.dtype).eps)
                noise = torch.randn(part.shape, device=part.device, dtype=part.dtype, generator=generator)
                out[..., sl] = part + alpha * rms * noise
            elif mode == "sign_flip":
                out[..., sl] = -part
            elif mode == "rotate90":
                out[..., sl] = torch.stack((-part[..., 1], part[..., 0]), dim=-1)
            else:
                raise ValueError(f"unknown perturbation mode {mode!r}")
    return out


@contextlib.contextmanager
def perturbed_planes(lm, choices, *, perturb_mode: str, alpha: float, seed: int):
    layers = lm.model.model.layers
    by_layer_q: dict[int, dict[int, tuple[int, ...]]] = {}
    by_layer_k: dict[int, dict[int, set[int]]] = {}
    for c in choices:
        by_layer_q.setdefault(c.layer, {})[c.head] = c.planes
        by_layer_k.setdefault(c.layer, {}).setdefault(c.kv_head, set(c.planes))
        by_layer_k[c.layer][c.kv_head] &= set(c.planes)
    # A CUDA generator keeps additive noise generated on-device and reproducible.
    gen = None
    device = next(lm.model.parameters()).device
    if perturb_mode == "noise":
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
    handles = []
    for li, layer in enumerate(layers):
        q_sel = by_layer_q.get(li, {})
        k_sel = by_layer_k.get(li, {})
        if q_sel:
            def q_hook(_module, _inputs, output, q_sel=q_sel):
                return _edit_planes(output, width=lm.d_head, selected=q_sel, mode=perturb_mode, alpha=alpha, generator=gen)
            handles.append(layer.self_attn.q_proj.register_forward_hook(q_hook))
        if k_sel:
            def k_hook(_module, _inputs, output, k_sel=k_sel):
                return _edit_planes(output, width=lm.d_head, selected=k_sel, mode=perturb_mode, alpha=alpha, generator=gen)
            handles.append(layer.self_attn.k_proj.register_forward_hook(k_hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def run(args: argparse.Namespace) -> None:
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    lm = load_model(args.model, device=device)
    if lm.tag != "qwen3":
        raise NotImplementedError("Phase 1.7 is currently scoped to qwen3")
    if args.eval_tokens < 200_000 and not args.allow_smoke_under_200k:
        raise RuntimeError("Phase 1.7 requires >=200k eval tokens; pass --allow-smoke-under-200k for smoke tests")
    ids = get_text(lm.tokenizer, max(args.eval_tokens, args.calibration_tokens))
    captures = capture_qk_norms(lm, ids[:args.calibration_tokens], args.calibration_tokens, args.limit_layers, args.forward_tokens, args.observed_tokens)
    plan = select_planes(lm, captures, tuple(args.epsilons), args.observed_tokens, device=device)
    plan.to_csv(out / "phase1_7_plane_selection_qwen3.csv", index=False)
    eval_ids = ids[:args.eval_tokens]
    base_ppl, base_logits, _ = eval_metrics(lm.model, eval_ids, stride=args.stride, forward_tokens=args.forward_tokens)
    rows = [{"condition": "base", "subspace": "base", "perturbation": "none", "fraction": 0.0, "alpha": 0.0, "ppl": base_ppl, "ppl_delta": 0.0, "kl": 0.0, "max_logit_delta": 0.0, "top1_agreement": 1.0}]
    for frac in args.fractions:
        for subspace in ("dead", "random", "live"):
            choices = choose_planes(lm, captures, mode=subspace, fraction=frac, seed=args.seed, device=device)
            for perturb_mode in args.perturbations:
                alphas = args.alphas if perturb_mode == "noise" else [1.0]
                for alpha in alphas:
                    with perturbed_planes(lm, choices, perturb_mode=perturb_mode, alpha=alpha, seed=args.seed):
                        ppl, _, drift = eval_metrics(lm.model, eval_ids, stride=args.stride, forward_tokens=args.forward_tokens, base_logits=base_logits)
                    rows.append({"condition": f"perturb_{subspace}", "subspace": subspace, "perturbation": perturb_mode, "fraction": frac, "alpha": alpha, "ppl": ppl, "ppl_delta": ppl - base_ppl, **drift})
                    pd.DataFrame(rows).to_csv(out / "phase1_7_perturbation_qwen3.csv", index=False)
    print(f"wrote Phase 1.7 outputs to {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1.7 Qwen3 causal perturbation of dead/random/live RoPE planes")
    p.add_argument("--model", default="qwen3", choices=["qwen3"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default="outputs/phase1_7_qwen3")
    p.add_argument("--calibration-tokens", type=int, default=100_000)
    p.add_argument("--eval-tokens", type=int, default=200_000)
    p.add_argument("--stride", type=int, default=512)
    p.add_argument("--forward-tokens", type=int, default=1024)
    p.add_argument("--observed-tokens", type=int, default=10_000)
    p.add_argument("--epsilons", type=float, nargs="+", default=[0.05, 0.1, 0.5, 1.0])
    p.add_argument("--fractions", type=float, nargs="+", default=[0.10, 0.25, 0.50])
    p.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.03, 0.1, 0.3, 1.0])
    p.add_argument("--perturbations", nargs="+", choices=["noise", "sign_flip", "rotate90"], default=["noise", "sign_flip", "rotate90"])
    p.add_argument("--limit-layers", type=int)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--allow-smoke-under-200k", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
