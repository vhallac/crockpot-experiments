from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Band:
    name: str
    dims: list[int]
    planes: list[int] | None = None


def rotary_dim(config, d_head: int, model_tag: str) -> int:
    if model_tag == "gpt2":
        return 0
    if model_tag.startswith("pythia"):
        return min(16, d_head)
    if hasattr(config, "rotary_pct"):
        return int(d_head * float(config.rotary_pct))
    if hasattr(config, "partial_rotary_factor"):
        return int(d_head * float(config.partial_rotary_factor))
    return d_head


def rope_bands(config, d_head: int, model_tag: str, n_bands: int = 4) -> list[Band]:
    """Partition RoPE rotate_half pairs into log-spaced frequency bands.

    HuggingFace rotary embeddings pair dimension i with i + d_rot/2, not adjacent
    dimensions. Pair index i has theta_i = base^(-2i / d_rot). Larger i is lower
    frequency. Pythia has a partial rotary seam: with rotary_pct=0.25, dims
    16..63 are exactly unrotated/DC and the 8 rotated pairs are split evenly.
    Qwen-style full-RoPE heads split their 64 pairs into four contiguous 16-pair
    bands; because log(theta_i) is linear in i, these are log-frequency bands.
    """
    d_rot = rotary_dim(config, d_head, model_tag)
    if d_rot <= 0:
        return [Band("all", list(range(d_head)), None)]
    if d_rot % 2:
        d_rot -= 1
    n_pairs = d_rot // 2
    if n_pairs == 0:
        return [Band("dc", list(range(d_head)), None)]

    labels = ["high", "mid_high", "mid_low", "low"]
    edges = [round(i * n_pairs / n_bands) for i in range(n_bands + 1)]
    edges[0] = 0
    edges[-1] = n_pairs

    bands: list[Band] = []
    for bi in range(n_bands):
        lo, hi = edges[bi], edges[bi + 1]
        if lo >= hi:
            continue
        pairs = list(range(lo, hi))
        dims: list[int] = []
        for p in pairs:
            dims.extend([p, p + n_pairs])
        bands.append(Band(labels[min(bi, len(labels) - 1)], dims, pairs))

    if d_rot < d_head:
        bands.append(Band("dc", list(range(d_rot, d_head)), None))
    return bands
