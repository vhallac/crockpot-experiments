from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from deadkeys.common.loading import MODEL_IDS, iter_heads, load_model, sanity_check
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


def run(args: argparse.Namespace) -> None:
    if args.model != "gpt2":
        raise ValueError("this first queryability experiment is intentionally GPT-2 only")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")

    lm = load_model(args.model, device=device, revision=args.revision)
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
    np.savez_compressed(npz_path, **spectra)
    print(f"wrote {csv_path}")
    print(f"wrote {npz_path}")


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
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
