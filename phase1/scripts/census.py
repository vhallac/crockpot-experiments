from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from phase1.common.loading import MODEL_IDS, iter_heads, load_model, sanity_check
from phase1.common.rope import Band, rope_bands
from phase1.common.spectra import group_dead_fraction, group_random_baseline, head_metrics, random_baseline


def _rows_for_bands(config, tag: str, d_head: int) -> list[Band]:
    bands = [Band("all", list(range(d_head)))]
    if tag != "gpt2":
        bands.extend(rope_bands(config, d_head, tag))
    return bands


def _compute_one(A: torch.Tensor, B: torch.Tensor, *, samples: int, seed: int, misalign_rotations: int):
    m = head_metrics(A, B, samples=samples, seed=seed, misalign_rotations=misalign_rotations)
    rb = random_baseline(A, B, samples=samples, seed=seed + 1000)
    m.dead_frac_random_baseline = rb
    return m


def run(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    lm = load_model(args.model)
    sanity = sanity_check(lm, atol=args.atol)
    print(f"sanity {args.model}: max_q_error={sanity['max_q_error']:.3g} max_k_error={sanity['max_k_error']:.3g}")

    rows = []
    spectra: dict[str, np.ndarray] = {}
    group_cache: dict[tuple[int, int, str], dict[str, object]] = {}
    bands = _rows_for_bands(lm.config, lm.tag, lm.d_head)

    for hs in iter_heads(lm, limit_layers=args.limit_layers, limit_heads=args.limit_heads):
        for band in bands:
            if not band.dims:
                continue
            A = hs.A[band.dims, :]
            B = hs.B[band.dims, :]
            if A.shape[0] < 2 or B.shape[0] < 2:
                continue
            seed = args.seed + 100_000 * hs.layer + 1000 * hs.head + len(rows)
            metrics = _compute_one(A, B, samples=args.samples, seed=seed, misalign_rotations=args.misalign_rotations)
            prefix = f"l{hs.layer}.h{hs.head}.{band.name}"
            spectra[f"{prefix}.S_A"] = metrics.S_A
            spectra[f"{prefix}.S_B"] = metrics.S_B
            spectra[f"{prefix}.S_M"] = metrics.S_M
            rows.append(
                {
                    "layer": hs.layer,
                    "head": hs.head,
                    "kv_head": hs.kv_head,
                    "band": band.name,
                    "S_A_sum": float(metrics.S_A.sum()),
                    "S_B_sum": float(metrics.S_B.sum()),
                    "S_M_sum": float(metrics.S_M.sum()),
                    "erank_A": metrics.erank_A,
                    "erank_B": metrics.erank_B,
                    "erank_M": metrics.erank_M,
                    "misalign_index": metrics.misalign_index,
                    "misalign_z": metrics.misalign_z,
                    "dead_frac": metrics.dead_frac,
                    "dead_frac_random_baseline": metrics.dead_frac_random_baseline,
                    "t5_threshold": metrics.t5_threshold,
                    "is_group_level": False,
                    "sanity_max_q_error": sanity["max_q_error"],
                    "sanity_max_k_error": sanity["max_k_error"],
                }
            )
            print(f"{args.model} layer={hs.layer} head={hs.head} band={band.name} dead={metrics.dead_frac:.3g} rand={metrics.dead_frac_random_baseline:.3g}")
            if lm.n_kv_heads != lm.n_heads:
                key = (hs.layer, hs.kv_head, band.name)
                entry = group_cache.setdefault(key, {"A_list": [], "B": B})
                entry["A_list"].append(A)

    for (layer, kv_head, band_name), entry in sorted(group_cache.items()):
        A_list = entry["A_list"]
        B = entry["B"]
        if not isinstance(A_list, list):
            continue
        seed = args.seed + 500_000 + 100_000 * layer + 1000 * kv_head + len(band_name)
        dead, t5 = group_dead_fraction(A_list, B, samples=args.samples, seed=seed)
        rb = group_random_baseline(A_list, B, samples=args.samples, seed=seed + 1000)
        rows.append(
            {
                "layer": layer,
                "head": -1,
                "kv_head": kv_head,
                "band": band_name,
                "S_A_sum": np.nan,
                "S_B_sum": np.nan,
                "S_M_sum": np.nan,
                "erank_A": np.nan,
                "erank_B": np.nan,
                "erank_M": np.nan,
                "misalign_index": np.nan,
                "misalign_z": np.nan,
                "dead_frac": dead,
                "dead_frac_random_baseline": rb,
                "t5_threshold": t5,
                "is_group_level": True,
                "sanity_max_q_error": sanity["max_q_error"],
                "sanity_max_k_error": sanity["max_k_error"],
            }
        )
        print(f"{args.model} layer={layer} kv_head={kv_head} band={band_name} group_dead={dead:.3g} group_rand={rb:.3g}")

    df = pd.DataFrame(rows)
    parquet_path = out / f"census_{args.model}.parquet"
    csv_path = out / f"census_{args.model}.csv"
    npz_path = out / f"spectra_{args.model}.npz"
    try:
        df.to_parquet(parquet_path, index=False)
        print(f"wrote {parquet_path}")
    except Exception as exc:
        print(f"parquet write failed: {exc}")
    df.to_csv(csv_path, index=False)
    np.savez_compressed(npz_path, **spectra)
    print(f"wrote {csv_path}")
    print(f"wrote {npz_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 weights-only dead-key census")
    p.add_argument("--model", required=True, choices=sorted(MODEL_IDS))
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--samples", type=int, default=10_000, help="random unit vectors for t5 baseline")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--limit-layers", type=int)
    p.add_argument("--limit-heads", type=int)
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--misalign-rotations", type=int, default=200, help="random orthogonal rotations for misalignment z-score")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
