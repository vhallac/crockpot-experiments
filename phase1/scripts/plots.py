from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def run(args: argparse.Namespace) -> None:
    inp = Path(args.input)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = read_table(inp)
    if df.empty:
        raise SystemExit("input table is empty")
    if "is_group_level" in df.columns:
        df = df[~df["is_group_level"]]

    model = args.model
    for band, bdf in df.groupby("band"):
        pivot = bdf.pivot(index="layer", columns="head", values="dead_frac")
        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(pivot.values, aspect="auto", interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(f"{model} dead_frac heatmap ({band})")
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        ax.set_xticks(range(len(pivot.columns)), pivot.columns)
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        fig.colorbar(im, ax=ax, label="dead_frac")
        fig.tight_layout()
        fig.savefig(out / f"{model}_dead_frac_heatmap_{band}.pdf")
        fig.savefig(out / f"{model}_dead_frac_heatmap_{band}.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    df[df["band"] == "all"]["dead_frac"].hist(ax=ax, bins=10, range=(0, 1))
    ax.set_title(f"{model} dead_frac histogram")
    ax.set_xlabel("dead_frac")
    ax.set_ylabel("heads")
    fig.tight_layout()
    fig.savefig(out / f"{model}_dead_frac_hist.pdf")
    fig.savefig(out / f"{model}_dead_frac_hist.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    all_df = df[df["band"] == "all"]
    ax.scatter(all_df["layer"], all_df["misalign_index"], s=12)
    ax.set_title(f"{model} misalignment vs layer")
    ax.set_xlabel("layer")
    ax.set_ylabel("misalign_index")
    fig.tight_layout()
    fig.savefig(out / f"{model}_misalign_vs_layer.pdf")
    fig.savefig(out / f"{model}_misalign_vs_layer.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(all_df["dead_frac_random_baseline"], all_df["dead_frac"], s=12)
    ax.plot([0, 1], [0, 1], color="black", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{model} real vs random dead_frac")
    ax.set_xlabel("random baseline")
    ax.set_ylabel("real")
    fig.tight_layout()
    fig.savefig(out / f"{model}_real_vs_random_dead_frac.pdf")
    fig.savefig(out / f"{model}_real_vs_random_dead_frac.png", dpi=150)
    plt.close(fig)

    print(f"wrote plots to {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 census plots")
    p.add_argument("--input", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output-dir", default="outputs")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
