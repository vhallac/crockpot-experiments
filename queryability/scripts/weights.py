from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from deadkeys.common.loading import MODEL_IDS, iter_heads, load_model, sanity_check
from deadkeys.common.rope import rotary_dim
from deadkeys.common.spectra import effective_rank


def _rank_from_singular_values(s: torch.Tensor, eps: float) -> int:
    if s.numel() == 0:
        return 0
    return int((s > eps * s.max().clamp_min(1e-30)).sum().item())


def _stable_rank(s: torch.Tensor) -> float:
    if s.numel() == 0:
        return 0.0
    top = s.max().square().clamp_min(1e-30)
    return float((s.square().sum() / top).item())


def _condition_number(s: torch.Tensor, eps: float) -> float:
    if s.numel() == 0:
        return float("nan")
    live = s[s > eps * s.max().clamp_min(1e-30)]
    if live.numel() == 0:
        return float("inf")
    return float((s.max() / live.min()).item())


def _summarize(prefix: str, s: torch.Tensor, eps: float) -> dict[str, float | int]:
    s = s.detach().float().cpu()
    near_zero = s <= eps * s.max().clamp_min(1e-30) if s.numel() else torch.empty(0, dtype=torch.bool)
    return {
        f"{prefix}_sigma_max": float(s.max().item()) if s.numel() else float("nan"),
        f"{prefix}_sigma_min": float(s.min().item()) if s.numel() else float("nan"),
        f"{prefix}_rank_eps": _rank_from_singular_values(s, eps),
        f"{prefix}_effective_rank": effective_rank(s),
        f"{prefix}_stable_rank": _stable_rank(s),
        f"{prefix}_condition_number_eps": _condition_number(s, eps),
        f"{prefix}_near_zero_fraction_eps": float(near_zero.float().mean().item()) if s.numel() else float("nan"),
        f"{prefix}_energy": float(s.square().sum().item()),
    }


def _environment_summary(device: torch.device) -> dict[str, object]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": getattr(torch.version, "cuda", None),
        "torch_hip": getattr(torch.version, "hip", None),
        "cuda_available": torch.cuda.is_available(),
        "requested_device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _write_manifest(path: Path, *, args: argparse.Namespace, lm, sanity: dict[str, float], d_rotary: int) -> None:
    manifest = {
        "script": "queryability.scripts.weights",
        "model": args.model,
        "hf_id": lm.hf_id,
        "revision": args.revision,
        "device": args.device,
        "limit_layers": args.limit_layers,
        "limit_heads": args.limit_heads,
        "eps": args.eps,
        "atol": args.atol,
        "d_model": lm.d_model,
        "d_head": lm.d_head,
        "n_layers": lm.n_layers,
        "n_heads": lm.n_heads,
        "n_kv_heads": lm.n_kv_heads,
        "rotary_dim": d_rotary,
        "geometry": "raw_pre_rope_projection_weights",
        "qk_norm_note": "qwen3 weights are raw q_proj/k_proj weights; q_norm/k_norm are non-linear and are not folded into this SVD",
        "sanity": sanity,
        "environment": _environment_summary(torch.device(args.device)),
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")

    env = _environment_summary(device)
    print(
        "environment: "
        f"torch={env['torch']} cuda_available={env['cuda_available']} "
        f"requested_device={env['requested_device']} cuda_device={env['cuda_device']}"
    )

    lm = load_model(args.model, device=device, revision=args.revision)
    d_rotary = rotary_dim(lm.config, lm.d_head, lm.tag)
    if d_rotary and not args.allow_rope_raw:
        raise ValueError(
            f"{args.model} uses RoPE over {d_rotary}/{lm.d_head} head dimensions. "
            "This script computes raw pre-RoPE projection-weight SVD only; rerun with "
            "--allow-rope-raw to record that interpretation explicitly."
        )
    if d_rotary:
        print(
            f"raw pre-RoPE projection SVD acknowledged for {args.model}: "
            f"rotary_dim={d_rotary}/{lm.d_head}"
        )

    sanity = sanity_check(lm, atol=args.atol)
    print(f"sanity {args.model}: max_q_error={sanity['max_q_error']:.3g} max_k_error={sanity['max_k_error']:.3g}")

    rows = []
    spectra: dict[str, np.ndarray] = {}
    for hs in iter_heads(lm, limit_layers=args.limit_layers, limit_heads=args.limit_heads):
        Wq = hs.A.detach().to(device=device, dtype=torch.float32)
        Wk = hs.B.detach().to(device=device, dtype=torch.float32)
        paired = Wq.T @ Wk

        s_q = torch.linalg.svdvals(Wq)
        s_k = torch.linalg.svdvals(Wk)
        s_paired = torch.linalg.svdvals(paired)

        prefix = f"l{hs.layer}.h{hs.head}"
        spectra[f"{prefix}.S_Q"] = s_q.cpu().numpy()
        spectra[f"{prefix}.S_K"] = s_k.cpu().numpy()
        spectra[f"{prefix}.S_QTK"] = s_paired.cpu().numpy()

        row = {
            "model": args.model,
            "layer": hs.layer,
            "head": hs.head,
            "d_model": lm.d_model,
            "d_head": lm.d_head,
            "n_heads": lm.n_heads,
            "n_kv_heads": lm.n_kv_heads,
            "kv_head": hs.kv_head,
            "rotary_dim": d_rotary,
            "geometry": "raw_pre_rope_projection_weights",
            "eps": args.eps,
            "sanity_max_q_error": sanity["max_q_error"],
            "sanity_max_k_error": sanity["max_k_error"],
        }
        row.update(_summarize("q", s_q, args.eps))
        row.update(_summarize("k", s_k, args.eps))
        row.update(_summarize("paired", s_paired, args.eps))
        rows.append(row)
        print(
            f"{args.model} layer={hs.layer} head={hs.head} "
            f"paired_rank={row['paired_rank_eps']} paired_erank={row['paired_effective_rank']:.3g}"
        )

    df = pd.DataFrame(rows)
    suffix = f"_{args.revision}" if args.revision else ""
    csv_path = out / f"queryability_{args.model}{suffix}.csv"
    npz_path = out / f"queryability_spectra_{args.model}{suffix}.npz"
    df.to_csv(csv_path, index=False)
    manifest_path = out / f"queryability_manifest_{args.model}{suffix}.json"
    np.savez_compressed(npz_path, **spectra)
    _write_manifest(manifest_path, args=args, lm=lm, sanity=sanity, d_rotary=d_rotary)
    print(f"wrote {csv_path}")
    print(f"wrote {npz_path}")
    print(f"wrote {manifest_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Weights-only paired W_Q^T W_K queryability spectrum")
    p.add_argument("--model", default="gpt2", choices=sorted(MODEL_IDS))
    p.add_argument("--output-dir", default="outputs/queryability")
    p.add_argument("--limit-layers", type=int)
    p.add_argument("--limit-heads", type=int)
    p.add_argument("--eps", type=float, default=1e-6, help="relative singular-value threshold")
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--revision", default=None)
    p.add_argument(
        "--allow-rope-raw",
        action="store_true",
        help="acknowledge that RoPE models are analyzed as raw pre-RoPE projection weights, not RoPE-aware maps",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
