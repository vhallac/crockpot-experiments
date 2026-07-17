from __future__ import annotations

import argparse
import filecmp
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from deadkeys.common.loading import MODEL_IDS, iter_heads, load_model, sanity_check
from deadkeys.common.rope import Band, rope_bands, rotary_dim
from deadkeys.common.spectra import group_dead_fraction, group_random_baseline, head_metrics, random_baseline


def _rows_for_bands(config, tag: str, d_head: int) -> list[Band]:
    bands = [Band("all", list(range(d_head)))]
    if tag != "gpt2":
        bands.extend(rope_bands(config, d_head, tag))
    return bands


def _compute_one(A: torch.Tensor, B: torch.Tensor, *, samples: int, seed: int, misalign_rotations: int, device: torch.device):
    m = head_metrics(A, B, samples=samples, seed=seed, misalign_rotations=misalign_rotations, device=device)
    rb = random_baseline(A, B, samples=samples, seed=seed + 1000, device=device)
    m.dead_frac_random_baseline = rb
    return m


def run(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")
    lm = load_model(args.model, device=device, revision=args.revision)
    print(f"device {device}")
    sanity = sanity_check(lm, atol=args.atol)
    print(f"sanity {args.model}: max_q_error={sanity['max_q_error']:.3g} max_k_error={sanity['max_k_error']:.3g}")

    rows = []
    spectra: dict[str, np.ndarray] = {}
    group_cache: dict[tuple[int, int, str], dict[str, object]] = {}
    bands = _rows_for_bands(lm.config, lm.tag, lm.d_head)
    d_rot = rotary_dim(lm.config, lm.d_head, lm.tag)
    half_rot = d_rot // 2 if d_rot > 0 else 0
    manifest_rows = []
    for band in bands:
        manifest_rows.append(
            {
                "model": args.model,
                "band": band.name,
                "dims": " ".join(map(str, band.dims)),
                "dim_count": len(band.dims),
                "rotate_half_pairs": "" if band.planes is None else " ".join(f"({p},{p + half_rot})" for p in band.planes),
                "pair_indices": "" if band.planes is None else " ".join(map(str, band.planes)),
            }
        )

    for hs in iter_heads(lm, limit_layers=args.limit_layers, limit_heads=args.limit_heads):
        for band in bands:
            if not band.dims:
                continue
            A = hs.A[band.dims, :]
            B = hs.B[band.dims, :]
            if A.shape[0] < 2 or B.shape[0] < 2:
                continue
            seed = args.seed + 100_000 * hs.layer + 1000 * hs.head + len(rows)
            metrics = _compute_one(A, B, samples=args.samples, seed=seed, misalign_rotations=args.misalign_rotations, device=device)
            prefix = f"l{hs.layer}.h{hs.head}.{band.name}"
            spectra[f"{prefix}.S_A"] = metrics.S_A
            spectra[f"{prefix}.S_B"] = metrics.S_B
            spectra[f"{prefix}.S_M"] = metrics.S_M
            spectra[f"{prefix}.U_B_top5"] = metrics.U_B_top5
            spectra[f"{prefix}.U_A_top8"] = metrics.U_A_top8
            spectra[f"{prefix}.A_soft_basis"] = metrics.A_soft_basis
            rows.append(
                {
                    "layer": hs.layer,
                    "head": hs.head,
                    "kv_head": hs.kv_head,
                    "band": band.name,
                    "band_dims": " ".join(map(str, band.dims)),
                    "band_planes": "" if band.planes is None else " ".join(map(str, band.planes)),
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
                    "park0": metrics.park[0] if metrics.park else float("nan"),
                    "park1": metrics.park[1] if len(metrics.park) > 1 else float("nan"),
                    "park2": metrics.park[2] if len(metrics.park) > 2 else float("nan"),
                    "park3": metrics.park[3] if len(metrics.park) > 3 else float("nan"),
                    "park4": metrics.park[4] if len(metrics.park) > 4 else float("nan"),
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
        dead, t5 = group_dead_fraction(A_list, B, samples=args.samples, seed=seed, device=device)
        rb = group_random_baseline(A_list, B, samples=args.samples, seed=seed + 1000, device=device)
        rows.append(
            {
                "layer": layer,
                "head": -1,
                "kv_head": kv_head,
                "band": band_name,
                "band_dims": np.nan,
                "band_planes": np.nan,
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
                "park0": np.nan,
                "park1": np.nan,
                "park2": np.nan,
                "park3": np.nan,
                "park4": np.nan,
                "is_group_level": True,
                "sanity_max_q_error": sanity["max_q_error"],
                "sanity_max_k_error": sanity["max_k_error"],
            }
        )
        print(f"{args.model} layer={layer} kv_head={kv_head} band={band_name} group_dead={dead:.3g} group_rand={rb:.3g}")

    df = pd.DataFrame(rows)
    suffix = f"_{args.revision}" if args.revision else ""
    parquet_path = out / f"census_{args.model}{suffix}.parquet"
    csv_path = out / f"census_{args.model}{suffix}.csv"
    npz_path = out / f"spectra_{args.model}{suffix}.npz"
    try:
        df.to_parquet(parquet_path, index=False)
        print(f"wrote {parquet_path}")
    except Exception as exc:
        print(f"parquet write failed: {exc}")
    df.to_csv(csv_path, index=False)
    np.savez_compressed(npz_path, **spectra)
    manifest_path = out / "bands_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    report_path = out / "REPORT.md"
    report_lines = [
        f"# Task 4 band census rerun — {args.model}",
        "",
        "## Scope",
        "Band census only. Rotary bands use HuggingFace rotate_half pairing: partner of dim i is i + d_rot/2.",
        "",
        "## Outputs",
        f"- `{csv_path.name}`",
        f"- `{npz_path.name}`",
        "- `bands_manifest.csv`",
        "",
        "## Anomaly log",
    ]
    anomalies = []
    if args.model.startswith("pythia"):
        dc = next((r for r in manifest_rows if r["band"] == "dc"), None)
        if dc is None or dc["dim_count"] != 48:
            anomalies.append(f"Pythia DC band dim_count is {None if dc is None else dc['dim_count']}, expected 48.")
    if args.prior_output_dir:
        prior = Path(args.prior_output_dir)
        identical = []
        for path in [csv_path, manifest_path, npz_path]:
            other = prior / path.name
            if other.exists() and filecmp.cmp(path, other, shallow=False):
                identical.append(path.name)
        if identical:
            anomalies.append("Bit-identical to prior run for: " + ", ".join(identical) + ".")
        else:
            report_lines.extend(["", f"Prior comparison directory: `{prior}`; no compared output was bit-identical."])
    if anomalies:
        report_lines.extend(f"- {item}" for item in anomalies)
    else:
        report_lines.append("- No anomalies detected by this script.")
    report_lines.extend(
        [
            "",
            "## Run parameters",
            f"- samples: {args.samples}",
            f"- misalign_rotations: {args.misalign_rotations}",
            f"- device: {args.device}",
            f"- sanity_max_q_error: {sanity['max_q_error']:.6g}",
            f"- sanity_max_k_error: {sanity['max_k_error']:.6g}",
        ]
    )
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"wrote {csv_path}")
    print(f"wrote {npz_path}")
    print(f"wrote {manifest_path}")
    print(f"wrote {report_path}")


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
    p.add_argument("--device", default="cpu", help="torch device for model weights and matrix work, e.g. cpu or cuda")
    p.add_argument("--revision", default=None, help="HuggingFace checkpoint revision, e.g. step1000 for Pythia")
    p.add_argument("--prior-output-dir", default=None, help="optional previous output directory to compare for bit-identical artifacts")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
