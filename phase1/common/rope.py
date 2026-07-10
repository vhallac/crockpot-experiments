from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Band:
    name: str
    dims: list[int]


def rotary_dim(config, d_head: int, model_tag: str) -> int:
    if model_tag == "gpt2":
        return 0
    if hasattr(config, "rotary_pct"):
        return int(d_head * float(config.rotary_pct))
    if hasattr(config, "partial_rotary_factor"):
        return int(d_head * float(config.partial_rotary_factor))
    return d_head


def rope_bands(config, d_head: int, model_tag: str, n_bands: int = 4) -> list[Band]:
    """Partition RoPE planes into log-spaced frequency-index bands.

    Plane index i has theta_i = base^(-2i / d_rot). Larger i is lower frequency.
    We split plane indices into contiguous bands in log-like index space and add a
    DC band for any unrotated dimensions.
    """
    d_rot = rotary_dim(config, d_head, model_tag)
    if d_rot <= 0:
        return [Band("all", list(range(d_head)))]
    if d_rot % 2:
        d_rot -= 1
    n_planes = d_rot // 2
    if n_planes == 0:
        return [Band("dc", list(range(d_head)))]

    # Use geometric edges over 1..n_planes+1, then convert to 0-based intervals.
    raw_edges = [round(math.exp(x)) for x in [math.log(1) + i * (math.log(n_planes + 1) - math.log(1)) / n_bands for i in range(n_bands + 1)]]
    edges = [0]
    for e in raw_edges[1:]:
        edges.append(max(edges[-1] + 1, min(n_planes, e - 1)))
    edges[-1] = n_planes

    bands: list[Band] = []
    labels = ["high", "mid_high", "mid_low", "low"]
    for bi in range(n_bands):
        lo, hi = edges[bi], edges[bi + 1]
        if lo >= hi:
            continue
        dims: list[int] = []
        for p in range(lo, hi):
            dims.extend([2 * p, 2 * p + 1])
        bands.append(Band(labels[min(bi, len(labels) - 1)], dims))

    if d_rot < d_head:
        bands.append(Band("dc", list(range(d_rot, d_head))))
    return bands
