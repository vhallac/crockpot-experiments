"""RS2 / C2 — Subspace overlap between RoPE k_post and DroPE'd emergent key-position.

Loads M1.5 projector bases from two pre-computed NPZ files, computes principal
angles per layer/head between the positional subspaces, and compares against
a random-rotation baseline per §10.F of RS1-spec.md.

Usage:
    PYTHONPATH=experiments/dead-keys:experiments/k-address-space \
      python experiments/rope-as-scaffold/scripts/c2_subspace_overlap.py \
      --rope-projectors outputs/.../kaddress_m15_projectors_qwen3.npz \
      --droped-projectors outputs/.../kaddress_m15_projectors_qwen3-droped.npz \
      --output-dir outputs/rs2_subspace_$(date -u +%Y%m%dT%H%M%SZ)
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ── projector loading ──────────────────────────────────────────────────────

def _parse_key(key: str) -> tuple[str, int, int, str] | None:
    """Parse a projector key like 'post/layer06/head00/aggregate_familyA'.

    Returns (variant_str, layer, head, family_or_slot) or None."""
    parts = key.split("/")
    if len(parts) != 4:
        return None
    variant_str, layer_str, head_str, detail = parts
    try:
        layer = int(layer_str.replace("layer", ""))
        head = int(head_str.replace("head", ""))
    except ValueError:
        return None
    return variant_str, layer, head, detail


def load_projector_bases(
    npz_path: str | Path,
    variant: str,
    *,
    families: tuple[str, ...] = ("A",),
    aggregate_only: bool = True,
) -> dict[tuple[int, int], np.ndarray]:
    """Load per-head projector bases from an M1.5 output NPZ.

    Args:
        npz_path: Path to kaddress_m15_projectors_*.npz.
        variant: 'pre' or 'post' — which key-variant to load.
        families: Which aggregate families to use (default: A only).
        aggregate_only: If True, only load aggregate_family* keys.

    Returns:
        Mapping (layer, head) -> basis array of shape (k, d_head).
        When multiple families are requested, their basis rows are stacked.
    """
    data = np.load(npz_path, allow_pickle=True)
    per_head: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)

    for key in data.keys():
        parsed = _parse_key(key)
        if parsed is None:
            continue
        v, layer, head, detail = parsed
        if v != variant:
            continue

        if aggregate_only:
            if not detail.startswith("aggregate_family"):
                continue
            fam = detail.replace("aggregate_family", "")
            if fam not in families:
                continue
        else:
            # For per-slot keys, extract family from stim ID
            # detail format: stimA4_00/slot00
            pass  # not used currently

        per_head[(layer, head)].append(data[key])

    # Stack basis rows for each head
    result: dict[tuple[int, int], np.ndarray] = {}
    for (layer, head), bases in per_head.items():
        if not bases:
            continue
        stacked = np.concatenate(bases, axis=0).astype(np.float64)
        # Re-orthonormalize (bases from different families may overlap slightly)
        stacked = _orthonormalize_rows(stacked)
        if stacked.shape[0] > 0:
            result[(layer, head)] = stacked

    return result


def _orthonormalize_rows(basis: np.ndarray) -> np.ndarray:
    """Re-orthonormalize rows via SVD, dropping near-zero directions."""
    if basis.shape[0] == 0:
        return basis.astype(np.float64)
    if basis.shape[0] == 1:
        # Single row: just normalize
        norm = np.linalg.norm(basis[0])
        return basis.astype(np.float64) if norm < 1e-12 else (basis / norm).astype(np.float64)
    # SVD: U @ diag(S) @ Vh
    # We want Vh rows as orthonormal basis for the row-space of `basis`
    _, s, vh = np.linalg.svd(basis, full_matrices=False)
    keep = s > (s[0] * 1e-10)
    return vh[keep].astype(np.float64)


# ── principal angles ────────────────────────────────────────────────────────

def principal_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Principal angles between row-orthonormal subspaces A (k1×d) and B (k2×d).

    Returns angles in radians, sorted ascending (smallest = best alignment).
    An empty array if either subspace is rank-0.
    """
    if A.shape[0] == 0 or B.shape[0] == 0:
        return np.array([], dtype=np.float64)
    # Compute A @ B^T and SVD
    M = A @ B.T  # (k1, k2)
    _, s, _ = np.linalg.svd(M, full_matrices=False)
    s = np.clip(s, 0.0, 1.0)
    return np.sort(np.arccos(s))


def subspace_alignment(A: np.ndarray, B: np.ndarray) -> float:
    """Scalar alignment: mean of cos(principal angles).

    1.0 = identical subspaces (up to rank), 0.0 = fully orthogonal.
    """
    angles = principal_angles(A, B)
    if len(angles) == 0:
        return 0.0
    return float(np.mean(np.cos(angles)))


# ── baseline ────────────────────────────────────────────────────────────────

def random_rotation_baseline(
    A: np.ndarray,
    B: np.ndarray,
    *,
    n_trials: int = 100,
    seed: int = 42,
) -> tuple[float, float]:
    """Empirical baseline: align A against randomly rotated copies of B.

    Returns (mean_alignment, std_alignment).
    """
    if A.shape[0] == 0 or B.shape[0] == 0:
        return 0.0, 0.0
    d = A.shape[1]
    rng = np.random.default_rng(seed)
    alignments = np.empty(n_trials, dtype=np.float64)
    for i in range(n_trials):
        # Random orthonormal matrix in d dimensions
        Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        B_rot = (Q @ B.T).T  # rotate the rows
        alignments[i] = subspace_alignment(A, B_rot)
    return float(alignments.mean()), float(alignments.std())


# ── main ────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rope_path = Path(args.rope_projectors)
    droped_path = Path(args.droped_projectors)
    if not rope_path.exists():
        raise FileNotFoundError(f"rope projectors not found: {rope_path}")
    if not droped_path.exists():
        raise FileNotFoundError(f"droped projectors not found: {droped_path}")

    families = tuple(args.families.split(","))
    print(f"loading RoPE post projectors from {rope_path}")
    t0 = time.monotonic()
    rope_post = load_projector_bases(rope_path, variant="post", families=families)
    rope_pre = load_projector_bases(rope_path, variant="pre", families=families)
    print(f"  rope_post heads: {len(rope_post)}, rope_pre heads: {len(rope_pre)}")

    print(f"loading DroPE'd pre projectors from {droped_path}")
    droped_pre = load_projector_bases(droped_path, variant="pre", families=families)
    print(f"  droped_pre heads: {len(droped_pre)}")
    print(f"loading took {time.monotonic() - t0:.1f}s")

    # Build the set of (layer, head) present in both
    common = sorted(set(rope_post.keys()) & set(droped_pre.keys()))
    print(f"common heads: {len(common)}")

    if not common:
        raise RuntimeError("no common layer/head pairs between projector files")

    rows: list[dict] = []
    n_trials = args.baseline_trials

    print(f"computing principal angles for {len(common)} heads "
          f"(baseline trials={n_trials})...")
    t0 = time.monotonic()

    for idx, (layer, head) in enumerate(common):
        A = rope_post[(layer, head)]   # RoPE k_post subspace
        B = droped_pre[(layer, head)]  # DroPE'd emergent subspace

        # Primary: RoPE post vs DroPE'd emergent
        angles = principal_angles(A, B)
        alignment = subspace_alignment(A, B)
        base_mean, base_std = random_rotation_baseline(A, B, n_trials=n_trials, seed=args.seed + layer * 100 + head)
        excess = alignment - base_mean

        # Reference: RoPE pre vs DroPE'd emergent (emergent-to-emergent)
        n_rope_pre_components = float("nan")
        if (layer, head) in rope_pre:
            A_pre = rope_pre[(layer, head)]
            n_rope_pre_components = A_pre.shape[0]
            ref_angles = principal_angles(A_pre, B)
            ref_alignment = subspace_alignment(A_pre, B)
        else:
            ref_angles = np.array([])
            ref_alignment = float("nan")

        # Reference 2: RoPE pre vs RoPE post (internal RoPE rotation shift).
        # G-RS2.3 requires this alignment to sit above a random baseline, not just below 1.0 —
        # so it needs its own baseline (rotating A_pre, not reusing the primary A-vs-B baseline,
        # since that was computed for B's rank, not necessarily A_pre's).
        if (layer, head) in rope_pre:
            internal_angles = principal_angles(A_pre, A)
            internal_alignment = subspace_alignment(A_pre, A)
            internal_base_mean, internal_base_std = random_rotation_baseline(
                A_pre, A, n_trials=n_trials, seed=args.seed + 500_000 + layer * 100 + head
            )
            internal_excess = internal_alignment - internal_base_mean
        else:
            internal_angles = np.array([])
            internal_alignment = float("nan")
            internal_base_mean = float("nan")
            internal_base_std = float("nan")
            internal_excess = float("nan")

        n_angles = len(angles)
        row = {
            "layer": layer,
            "head": head,
            "n_rope_post_components": A.shape[0],
            "n_rope_pre_components": n_rope_pre_components,
            "n_droped_components": B.shape[0],
            "min_dim": min(A.shape[0], B.shape[0]),
            "alignment_mean_cos": alignment,
            "random_baseline_mean": base_mean,
            "random_baseline_std": base_std,
            "alignment_excess": excess,
            "alignment_excess_sigma": float("nan") if base_std == 0 else excess / base_std,
            "principal_angle_min": float(np.min(angles)) if n_angles else float("nan"),
            "principal_angle_max": float(np.max(angles)) if n_angles else float("nan"),
            "principal_angle_mean": float(np.mean(angles)) if n_angles else float("nan"),
            "n_principal_angles": n_angles,
            "ref_rope_pre_vs_droped_alignment": ref_alignment,
            "ref_rope_pre_vs_post_alignment": internal_alignment,
            "ref_rope_pre_vs_post_baseline_mean": internal_base_mean,
            "ref_rope_pre_vs_post_baseline_std": internal_base_std,
            "ref_rope_pre_vs_post_excess": internal_excess,
        }
        rows.append(row)

        if (idx + 1) % 50 == 0 or idx + 1 == len(common):
            elapsed = max(time.monotonic() - t0, 1e-9)
            print(f"  progress {idx+1}/{len(common)} rate={(idx+1)/elapsed:.1f}/s", flush=True)

    # ── summary statistics ──────────────────────────────────────────────────
    alignments = np.array([r["alignment_mean_cos"] for r in rows])
    excesses = np.array([r["alignment_excess"] for r in rows])
    baselines = np.array([r["random_baseline_mean"] for r in rows])

    # By-layer aggregation
    layer_stats = []
    for layer in sorted(set(r["layer"] for r in rows)):
        layer_rows = [r for r in rows if r["layer"] == layer]
        al = np.array([r["alignment_mean_cos"] for r in layer_rows])
        ex = np.array([r["alignment_excess"] for r in layer_rows])
        bl = np.array([r["random_baseline_mean"] for r in layer_rows])
        n_heads = len(layer_rows)
        layer_stats.append({
            "layer": layer,
            "n_heads": n_heads,
            "alignment_mean": float(np.mean(al)),
            "alignment_std": float(np.std(al)),
            "baseline_mean": float(np.mean(bl)),
            "baseline_std": float(np.std(bl)),
            "excess_mean": float(np.mean(ex)),
            "excess_std": float(np.std(ex)),
            "excess_above_zero_fraction": float(np.mean(ex > 0)),
        })

    summary = {
        "analysis": "RS2 C2 — subspace overlap",
        "rope_projectors": str(rope_path.resolve()),
        "droped_projectors": str(droped_path.resolve()),
        "families": list(families),
        "baseline_trials": n_trials,
        "seed": args.seed,
        "n_heads_total": len(common),
        "n_layers": len(set(r["layer"] for r in rows)),
        "alignment_mean_cos_global_mean": float(np.mean(alignments)),
        "alignment_mean_cos_global_std": float(np.std(alignments)),
        "baseline_global_mean": float(np.mean(baselines)),
        "baseline_global_std": float(np.std(baselines)),
        "excess_global_mean": float(np.mean(excesses)),
        "excess_global_std": float(np.std(excesses)),
        "excess_above_zero_fraction": float(np.mean(excesses > 0)),
        "layer_breakdown": layer_stats,
    }

    # ── write outputs ───────────────────────────────────────────────────────
    import csv as csv_mod

    csv_path = out / "rs2_subspace_overlap.csv"
    json_path = out / "rs2_subspace_summary.json"

    fieldnames = [
        "layer", "head",
        "n_rope_post_components", "n_rope_pre_components", "n_droped_components", "min_dim",
        "alignment_mean_cos", "random_baseline_mean", "random_baseline_std",
        "alignment_excess", "alignment_excess_sigma",
        "principal_angle_min", "principal_angle_max", "principal_angle_mean",
        "n_principal_angles",
        "ref_rope_pre_vs_droped_alignment",
        "ref_rope_pre_vs_post_alignment", "ref_rope_pre_vs_post_baseline_mean",
        "ref_rope_pre_vs_post_baseline_std", "ref_rope_pre_vs_post_excess",
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = csv_mod.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"\nwrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"\nSummary:")
    print(f"  alignment mean ± std:    {summary['alignment_mean_cos_global_mean']:.4f} ± {summary['alignment_mean_cos_global_std']:.4f}")
    print(f"  baseline mean ± std:     {summary['baseline_global_mean']:.4f} ± {summary['baseline_global_std']:.4f}")
    print(f"  excess mean ± std:       {summary['excess_global_mean']:.4f} ± {summary['excess_global_std']:.4f}")
    print(f"  excess > 0 fraction:     {summary['excess_above_zero_fraction']:.3f}")

    # Per-layer summary
    print(f"\nPer-layer excess (alignment above random baseline):")
    print(f"  {'layer':>5s}  {'excess':>8s}  {'align':>8s}  {'base':>8s}  {'>0%':>6s}")
    for ls in layer_stats:
        print(f"  {ls['layer']:5d}  {ls['excess_mean']:8.4f}  {ls['alignment_mean']:8.4f}  {ls['baseline_mean']:8.4f}  {ls['excess_above_zero_fraction']:6.3f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RS2 C2 subspace overlap analysis")
    p.add_argument("--rope-projectors", required=True, help="Path to RoPE-state M1.5 projectors NPZ (qwen3)")
    p.add_argument("--droped-projectors", required=True, help="Path to DroPE'd-state M1.5 projectors NPZ (qwen3-droped)")
    p.add_argument("--output-dir", required=True, help="Directory for output CSV and summary JSON")
    p.add_argument("--families", default="A", help="Comma-separated families to use (default: A)")
    p.add_argument("--baseline-trials", type=int, default=100, help="Random rotation trials per head")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
