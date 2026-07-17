"""Task 3 — Pythia checkpoint biography (the dynamics experiment).

Runs the weights-only census (SPEC §3) across 12 Pythia-410m training
checkpoints, plus the alignment-entailment null model (SPEC §3B.4, k=5 only,
100 synthetics) at 4 sentinel checkpoints. Aggregates a longitudinal
``biography_pythia410.csv`` and produces the plots + REPORT.md that answer the
pre-registered questions P3.a–P3.e (see ``temp/next_steps_plan.md`` §3.3).

Weights-only, CPU-viable. Same RNG seed across checkpoints for the random
baselines so the longitudinal comparison is apples-to-apples.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from deadkeys.common.loading import load_model
from deadkeys.scripts import census as census_mod
from deadkeys.scripts.phase1_5 import decompose_heads, null_model

# (§3.1) Checkpoint revisions to census, in training order.
CHECKPOINTS = [
    "step0", "step512", "step1000", "step2000", "step4000", "step8000",
    "step16000", "step32000", "step64000", "step100000", "step128000", "step143000",
]
# Null model only at the four sentinels (§3.1).
NULL_CHECKPOINTS = ["step0", "step4000", "step32000", "step143000"]
# Monk criterion (§3.2): dead_frac > 0.3 in an early/mid layer.
MONK_DEAD_FRAC = 0.3
MONK_MAX_LAYER = 16
# Head whose spectra overlay traces the monarchy forming (§3.2).
SPECTRA_HEAD = (23, 13)


def run_checkpoint_census(model: str, revision: str, out: Path, args) -> pd.DataFrame:
    """Run one census in-process and return its per-head DataFrame."""
    ns = argparse.Namespace(
        model=model,
        output_dir=str(out),
        samples=args.samples,
        seed=args.seed,
        limit_layers=None,
        limit_heads=None,
        atol=args.atol,
        misalign_rotations=args.misalign_rotations,
        device=args.device,
        revision=revision,
    )
    print(f"=== census {model} revision={revision} ===", flush=True)
    census_mod.run(ns)
    suffix = f"_{revision}"
    csv = out / f"census_{model}{suffix}.csv"
    return pd.read_csv(csv)


def run_null_checkpoint(model: str, revision: str, out: Path, args) -> pd.DataFrame:
    """Run the alignment-entailment null model (k=5, 100 synthetics) at one checkpoint."""
    print(f"=== null model {model} revision={revision} ===", flush=True)
    device = torch.device(args.device)
    lm = load_model(model, device=device, revision=revision)
    decomps = decompose_heads(lm)
    df = null_model(decomps, samples=args.null_samples, dead_samples=args.null_dead_samples,
                    depths=(5,), seed=args.seed)
    df["checkpoint"] = revision
    df.to_csv(out / f"null_model_{model}_{revision}.csv", index=False)
    # Free model memory.
    del lm, decomps
    import gc
    gc.collect()
    return df


def aggregate_biography(per_ckpt: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per (checkpoint, layer) medians of the headline observables + monk count.

    Uses only the ``all`` band, per-head (non-group) rows.
    """
    rows = []
    for ckpt, df in per_ckpt.items():
        d = df[(df["band"] == "all") & (~df["is_group_level"])]
        step = int(ckpt.replace("step", ""))
        monk_count = int(((d["dead_frac"] > MONK_DEAD_FRAC) & (d["layer"] <= MONK_MAX_LAYER)).sum())
        for layer, g in d.groupby("layer"):
            rows.append({
                "checkpoint": ckpt,
                "step": step,
                "layer": int(layer),
                "erank_A": g["erank_A"].median(),
                "erank_M": g["erank_M"].median(),
                "dead_frac": g["dead_frac"].median(),
                "misalign_z": g["misalign_z"].median(),
                "park0": g["park0"].median(),
            })
        rows.append({
            "checkpoint": ckpt, "step": step, "layer": -1,
            "erank_A": np.nan, "erank_M": np.nan, "dead_frac": np.nan,
            "misalign_z": np.nan, "park0": np.nan,
            "monk_count": monk_count,
        })
    return pd.DataFrame(rows)


def _log_step_axis(ax, steps):
    steps = sorted(set(steps))
    labels = [f"{s//1000}k" if s >= 1000 else str(s) for s in steps]
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("checkpoint (training step)")


def plot_biography(bio: pd.DataFrame, per_ckpt: dict[str, pd.DataFrame], out: Path) -> None:
    ckpts = [c for c in CHECKPOINTS if c in per_ckpt]
    steps = [int(c.replace("step", "")) for c in ckpts]
    layers = sorted({int(l) for l in bio["layer"] if l >= 0})

    def heatmap(col, title, fname, cmap="viridis"):
        Z = np.full((len(layers), len(ckpts)), np.nan)
        for ci, ckpt in enumerate(ckpts):
            sub = bio[(bio["checkpoint"] == ckpt) & (bio["layer"] >= 0)]
            mp = dict(zip(sub["layer"], sub[col]))
            for li, lay in enumerate(layers):
                Z[li, ci] = mp.get(lay, np.nan)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(Z, aspect="auto", cmap=cmap, origin="lower")
        ax.set_yticks(range(0, len(layers), 4))
        ax.set_yticklabels([layers[i] for i in range(0, len(layers), 4)])
        _log_step_axis(ax, steps)
        ax.set_ylabel("layer")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label=col)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=150)
        plt.close(fig)

    heatmap("erank_A", "erank_A (layer × checkpoint)", "heatmap_erank_A.png")
    heatmap("erank_M", "erank_M (layer × checkpoint)", "heatmap_erank_M.png")
    heatmap("dead_frac", "dead_frac (layer × checkpoint)", "heatmap_dead_frac.png", cmap="magma")
    heatmap("misalign_z", "misalign_z (layer × checkpoint)", "heatmap_misalign_z.png", cmap="coolwarm")

    # park0 median trajectory (per-checkpoint median over all layers/heads).
    fig, ax = plt.subplots(figsize=(7, 4))
    park = [bio[(bio["checkpoint"] == c) & (bio["layer"] >= 0)]["park0"].median() for c in ckpts]
    ax.plot(steps, park, marker="o")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("training step")
    ax.set_ylabel("median park0")
    ax.set_title("park0 trajectory (lighthouse parking)")
    fig.tight_layout()
    fig.savefig(out / "trajectory_park0.png", dpi=150)
    plt.close(fig)

    # monk-count trajectory.
    fig, ax = plt.subplots(figsize=(7, 4))
    monks = [int(bio[(bio["checkpoint"] == c) & (bio["layer"] == -1)]["monk_count"].iloc[0]) for c in ckpts]
    ax.plot(steps, monks, marker="s", color="darkred")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("training step")
    ax.set_ylabel("monk head count")
    ax.set_title(f"monk count (dead_frac>{MONK_DEAD_FRAC}, layer≤{MONK_MAX_LAYER})")
    fig.tight_layout()
    fig.savefig(out / "trajectory_monks.png", dpi=150)
    plt.close(fig)

    # spectra overlay for the monarchy head L23.H13 across checkpoints.
    fig, ax = plt.subplots(figsize=(8, 5))
    for ckpt in ckpts:
        npz = np.load(out / f"spectra_pythia410_{ckpt}.npz")
        key = f"l{SPECTRA_HEAD[0]}.h{SPECTRA_HEAD[1]}.all.S_M"
        if key in npz.files:
            s = npz[key]
            ax.plot(np.arange(1, len(s) + 1), s, marker=".", markersize=3, linewidth=0.8, label=ckpt)
    ax.set_yscale("log")
    ax.set_xlabel("singular index")
    ax.set_ylabel("S_M (log)")
    ax.set_title(f"spectra overlay L{SPECTRA_HEAD[0]}.H{SPECTRA_HEAD[1]} — monarchy forming")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "spectra_overlay_L23H13.png", dpi=150)
    plt.close(fig)


def write_report(bio: pd.DataFrame, per_ckpt: dict[str, pd.DataFrame],
                 nulls: dict[str, pd.DataFrame], out: Path) -> None:
    """Answer P3.a–P3.e from the aggregated data."""
    ckpts = [c for c in CHECKPOINTS if c in per_ckpt]
    # P3.a: late-layer erank collapse — architectural (step0) or formed?
    late = bio["layer"].max()
    def layer_median(col, ckpt, lay):
        sub = bio[(bio["checkpoint"] == ckpt) & (bio["layer"] == lay)]
        return float(sub[col].iloc[0]) if len(sub) else float("nan")
    erank_late_step0 = layer_median("erank_A", "step0", late)
    erank_late_final = layer_median("erank_A", ckpts[-1], late)
    # earliest checkpoint where late-layer erank_A drops below 0.5 of step0 value
    collapse_step = None
    for c in ckpts:
        v = layer_median("erank_A", c, late)
        if np.isfinite(v) and erank_late_step0 > 0 and v < 0.5 * erank_late_step0:
            collapse_step = int(c.replace("step", ""))
            break

    # P3.b: tail-rescue — compare dead_frac at early vs late for broadband (all band).
    dead_step0 = float(bio[(bio["checkpoint"] == "step0") & (bio["layer"] >= 0)]["dead_frac"].median())
    dead_final = float(bio[(bio["checkpoint"] == ckpts[-1]) & (bio["layer"] >= 0)]["dead_frac"].median())

    # P3.c: monks.
    monk_traj = {c: int(bio[(bio["checkpoint"] == c) & (bio["layer"] == -1)]["monk_count"].iloc[0]) for c in ckpts}
    first_monk = next((int(c.replace("step", "")) for c in ckpts if monk_traj[c] > 0), None)

    # P3.d: anti-alignment — strongly negative misalign_z coupled to dead_frac.
    def anti_align_count(ckpt):
        d = per_ckpt[ckpt][(per_ckpt[ckpt]["band"] == "all") & (~per_ckpt[ckpt]["is_group_level"])]
        return int(((d["misalign_z"] < -2) & (d["dead_frac"] > 0.2)).sum())
    anti_traj = {c: anti_align_count(c) for c in ckpts}

    # P3.e: induction window reorganization — max |Δerank_M|/step around 1k–10k.
    mid_ckpts = [c for c in ckpts if 512 <= int(c.replace("step", "")) <= 16000]
    reorg_notes = []
    prev = None
    for c in mid_ckpts:
        v = float(bio[(bio["checkpoint"] == c) & (bio["layer"] >= 0)]["erank_M"].median())
        if prev is not None and np.isfinite(prev) and np.isfinite(v):
            reorg_notes.append(f"{c}: Δmedian_erank_M={v - prev:+.3f}")
        prev = v

    # Null model summary per sentinel.
    null_lines = []
    for c, ndf in nulls.items():
        z = ndf["z"]
        null_lines.append(f"  {c}: median z={z.median():+.2f}, "
                          f"frac below null (z<0)={(z<0).mean():.2%}")

    report = f"""# Task 3 — Pythia-410m Checkpoint Biography (REPORT.md)

## Setup

Census (SPEC §3, all bands, corrected Pythia band seams: DC = unrotated 48 dims,
4 log-spaced bands over the 8 rotated planes) run at {len(ckpts)} checkpoints:
{', '.join(ckpts)}.

Same RNG seed ({'ARGS_SEED'}) for all random baselines across checkpoints.
Spectra stored with singular vectors (U_A top-8, U_B top-5) so park0 and the null
model are computable per checkpoint. Null model (k=5, 100 synthetics) at 4
sentinels: {', '.join(NULL_CHECKPOINTS)}.

Parameters: samples per head = {'ARGS_SAMPLES'}, misalign_rotations =
{'ARGS_MISALIGN'}, null dead_samples = {'ARGS_NULL_DEAD'}.

Monk criterion: dead_frac > {MONK_DEAD_FRAC} AND layer ≤ {MONK_MAX_LAYER}.

## Pre-registered questions

### P3.a — Late-layer erank collapse: architectural or formed?

Late layer = {late}. erank_A at step0 = {erank_late_step0:.2f}; at final
({ckpts[-1]}) = {erank_late_final:.2f}.
{"Collapse formed during training: earliest step where late-layer erank_A < 50% of step0 value = " + str(collapse_step) if collapse_step else "No ≥50% collapse detected relative to step0 (late-layer erank roughly stable or already low at init)."}
See `heatmap_erank_A.png` — the collapse's birth certificate.

### P3.b — Broadband tail-rescue: early-then-destroyed, or never forms?

Median dead_frac (all band, all layers): step0 = {dead_step0:.3f}, final =
{dead_final:.3f}.
Pythia-final lacks the broadband tail-rescue that GPT-2/Qwen3 show. Whether it
ever existed early is read off the dead_frac heatmap trajectory
(`heatmap_dead_frac.png`): if step0 dead_frac ≈ random baseline (~0.087) and
stays flat, rescue never formed; if it dips below baseline at an intermediate
step then climbs, it formed and was destroyed.

### P3.c — Monks: gradual or transition? Identities persist?

Monk-count trajectory: {monk_traj}.
First checkpoint with a monk: {first_monk}.
A sharp jump between consecutive checkpoints = phase transition; a steady climb
= gradual. See `trajectory_monks.png`. Persistence: cross-reference which
(layer, head) identities are monks at each checkpoint (per-checkpoint census
CSVs) — stable membership implies persistence.

### P3.d — Qwen3-style anti-alignment at any Pythia checkpoint?

Anti-aligned head count (misalign_z < -2 AND dead_frac > 0.2, all band):
{anti_traj}.
Qwen3 shows strongly negative misalign_z coupled to high dead_frac. If the count
is ~0 at every checkpoint, Pythia never develops Qwen3-style anti-alignment.
See `heatmap_misalign_z.png`.

### P3.e — Induction-emergence window (~1k–10k) coincides with reorganization?

Median erank_M deltas across the 1k–10k window:
{chr(10).join('  ' + n for n in reorg_notes)}
Largest |Δ| in this window localizes where training reorganizes QK interaction
rank. Cross-reference with `heatmap_erank_M.png`.

## Null model (alignment-entailment, k=5)

{chr(10).join(null_lines)}

Per-checkpoint null CSVs: `null_model_pythia410_<ckpt>.csv`.
If median z ≪ 0 the tail is rescued relative to lighthouse-entailment (as in
GPT-2 v1); if z ≈ 0 the deadness is bookkeeping-entailed; if z ≫ 0 it is excess
structure (oubliette / gradient shadow).

## Outputs

- `biography_pythia410.csv` — per (checkpoint, layer) medians + monk count.
- `census_pythia410_<ckpt>.csv` / `.parquet` / `spectra_...npz` per checkpoint.
- `null_model_pythia410_<ckpt>.csv` per sentinel.
- Plots: heatmaps (erank_A, erank_M, dead_frac, misalign_z), park0 + monk
  trajectories, spectra overlay L23.H13.
"""
    report = report.replace("ARGS_SEED", str(getattr(write_report, "_seed", 1234)))
    report = report.replace("ARGS_SAMPLES", str(getattr(write_report, "_samples", 4000)))
    report = report.replace("ARGS_MISALIGN", str(getattr(write_report, "_misalign", 100)))
    report = report.replace("ARGS_NULL_DEAD", str(getattr(write_report, "_null_dead", 2000)))
    (out / "REPORT.md").write_text(report)


def run(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Stash params for the report template.
    write_report._seed = args.seed
    write_report._samples = args.samples
    write_report._misalign = args.misalign_rotations
    write_report._null_dead = args.null_dead_samples

    per_ckpt: dict[str, pd.DataFrame] = {}
    for ckpt in CHECKPOINTS:
        try:
            per_ckpt[ckpt] = run_checkpoint_census(args.model, ckpt, out, args)
        except Exception as exc:
            print(f"!! checkpoint {ckpt} failed: {exc}", flush=True)

    nulls: dict[str, pd.DataFrame] = {}
    for ckpt in NULL_CHECKPOINTS:
        if ckpt not in per_ckpt:
            continue
        try:
            nulls[ckpt] = run_null_checkpoint(args.model, ckpt, out, args)
        except Exception as exc:
            print(f"!! null {ckpt} failed: {exc}", flush=True)

    bio = aggregate_biography(per_ckpt)
    bio.to_csv(out / "biography_pythia410.csv", index=False)
    print(f"wrote {out / 'biography_pythia410.csv'}")

    plot_biography(bio, per_ckpt, out)
    print("wrote plots")

    write_report(bio, per_ckpt, nulls, out)
    print(f"wrote {out / 'REPORT.md'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 3 — Pythia checkpoint biography")
    p.add_argument("--model", default="pythia410", choices=sorted({"pythia410", "pythia1.4"}))
    p.add_argument("--output-dir", default="outputs/task3_pythia_biography")
    p.add_argument("--samples", type=int, default=4000, help="random unit vectors for t5 baseline")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--misalign-rotations", type=int, default=100)
    p.add_argument("--null-samples", type=int, default=100, help="synthetic (A',B') pairs per head for the null model")
    p.add_argument("--null-dead-samples", type=int, default=2000, help="random dirs for dead_fraction inside the null model")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())