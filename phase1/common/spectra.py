from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class HeadMetrics:
    S_A: np.ndarray
    S_B: np.ndarray
    S_M: np.ndarray
    U_B_top5: np.ndarray
    A_soft_basis: np.ndarray
    erank_A: float
    erank_B: float
    erank_M: float
    misalign_index: float
    misalign_z: float
    dead_frac: float
    dead_frac_random_baseline: float
    t5_threshold: float


def effective_rank(s: torch.Tensor) -> float:
    e = s.float().square()
    total = e.sum()
    if total <= 0:
        return 0.0
    p = e / total
    entropy = -(p[p > 0] * torch.log(p[p > 0])).sum()
    return float(torch.exp(entropy).item())


def _unit_random(n: int, d: int, device: torch.device, seed: int) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    u = torch.randn(n, d, generator=g, device=device, dtype=torch.float32)
    return u / u.norm(dim=1, keepdim=True).clamp_min(1e-12)


def pullback_scores(A: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Return ||A.T @ u|| for row-batched unit directions u."""
    return (dirs @ A).norm(dim=1)


def dead_fraction(A: torch.Tensor, U_B: torch.Tensor, S_B: torch.Tensor, *, samples: int, seed: int) -> tuple[float, float]:
    d_head = A.shape[0]
    baseline_dirs = _unit_random(samples, d_head, A.device, seed)
    baseline = pullback_scores(A, baseline_dirs)
    t5 = torch.quantile(baseline, 0.05)
    key_scores = pullback_scores(A, U_B.T.contiguous())
    weights = S_B.float().square()
    dead = weights[key_scores < t5].sum() / weights.sum().clamp_min(1e-30)
    return float(dead.item()), float(t5.item())


def misalignment_z_score(S_A: torch.Tensor, S_B: torch.Tensor, raw: torch.Tensor, *, rotations: int = 200, seed: int = 0) -> float:
    """Spec §3.2 random-orthogonal baseline for the raw misalignment index."""
    if rotations <= 1:
        return float("nan")
    d = S_A.numel()
    denom = torch.sum(S_A * S_B).clamp_min(1e-30)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    vals = []
    diag_a = torch.diag(S_A)
    diag_b = torch.diag(S_B)
    for _ in range(rotations):
        Q, R = torch.linalg.qr(torch.randn(d, d, generator=gen, dtype=torch.float32))
        signs = torch.sign(torch.diag(R))
        signs[signs == 0] = 1
        Q = Q * signs.unsqueeze(0)
        vals.append(torch.linalg.svdvals(diag_a @ Q @ diag_b).sum() / denom)
    rand = torch.stack(vals)
    std = rand.std(unbiased=True)
    if std <= 1e-12:
        return float("nan")
    return float(((raw - rand.mean()) / std).item())


def head_metrics(A: torch.Tensor, B: torch.Tensor, *, samples: int = 10_000, seed: int = 0, misalign_rotations: int = 200) -> HeadMetrics:
    A = A.detach().to(dtype=torch.float32, device="cpu")
    B = B.detach().to(dtype=torch.float32, device="cpu")

    U_A, S_A, _ = torch.linalg.svd(A, full_matrices=False)
    U_B, S_B, _ = torch.linalg.svd(B, full_matrices=False)
    C = torch.diag(S_A) @ (U_A.T @ U_B) @ torch.diag(S_B)
    S_M = torch.linalg.svdvals(C)
    denom = torch.sum(S_A * S_B).clamp_min(1e-30)
    raw_misalign = S_M.sum() / denom
    misalign_z = misalignment_z_score(S_A, S_B, raw_misalign, rotations=misalign_rotations, seed=seed + 31)
    dead, t5 = dead_fraction(A, U_B, S_B, samples=samples, seed=seed)
    a_pullbacks = pullback_scores(A, U_A.T.contiguous())
    soft_idx = torch.nonzero(a_pullbacks < t5, as_tuple=False).flatten()[:8]
    a_soft_basis = U_A[:, soft_idx].contiguous()

    return HeadMetrics(
        S_A=S_A.numpy(),
        S_B=S_B.numpy(),
        S_M=S_M.numpy(),
        U_B_top5=U_B[:, :5].contiguous().numpy(),
        A_soft_basis=a_soft_basis.numpy(),
        erank_A=effective_rank(S_A),
        erank_B=effective_rank(S_B),
        erank_M=effective_rank(S_M),
        misalign_index=float(raw_misalign.item()),
        misalign_z=misalign_z,
        dead_frac=dead,
        dead_frac_random_baseline=np.nan,
        t5_threshold=t5,
    )


def random_baseline(A: torch.Tensor, B: torch.Tensor, *, samples: int, seed: int) -> float:
    A = A.detach().to(dtype=torch.float32, device="cpu")
    B = B.detach().to(dtype=torch.float32, device="cpu")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    Ar = torch.randn(A.shape, generator=gen, dtype=torch.float32)
    Br = torch.randn(B.shape, generator=gen, dtype=torch.float32)
    Ar = Ar * (A.norm() / Ar.norm().clamp_min(1e-30))
    Br = Br * (B.norm() / Br.norm().clamp_min(1e-30))
    U_Br, S_Br, _ = torch.linalg.svd(Br, full_matrices=False)
    dead, _ = dead_fraction(Ar, U_Br, S_Br, samples=samples, seed=seed + 17)
    return dead


def group_dead_fraction(A_list: list[torch.Tensor], B: torch.Tensor, *, samples: int, seed: int) -> tuple[float, float]:
    """GQA dead fraction: key direction must be weak for every query head reader."""
    A_list = [A.detach().to(dtype=torch.float32, device="cpu") for A in A_list]
    B = B.detach().to(dtype=torch.float32, device="cpu")
    U_B, S_B, _ = torch.linalg.svd(B, full_matrices=False)
    d_head = B.shape[0]
    baseline_dirs = _unit_random(samples, d_head, B.device, seed)
    baseline = torch.stack([pullback_scores(A, baseline_dirs) for A in A_list], dim=0).max(dim=0).values
    t5 = torch.quantile(baseline, 0.05)
    key_dirs = U_B.T.contiguous()
    key_scores = torch.stack([pullback_scores(A, key_dirs) for A in A_list], dim=0).max(dim=0).values
    weights = S_B.float().square()
    dead = weights[key_scores < t5].sum() / weights.sum().clamp_min(1e-30)
    return float(dead.item()), float(t5.item())


def group_random_baseline(A_list: list[torch.Tensor], B: torch.Tensor, *, samples: int, seed: int) -> float:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    rand_as = []
    for A in A_list:
        A = A.detach().to(dtype=torch.float32, device="cpu")
        Ar = torch.randn(A.shape, generator=gen, dtype=torch.float32)
        rand_as.append(Ar * (A.norm() / Ar.norm().clamp_min(1e-30)))
    B = B.detach().to(dtype=torch.float32, device="cpu")
    Br = torch.randn(B.shape, generator=gen, dtype=torch.float32)
    Br = Br * (B.norm() / Br.norm().clamp_min(1e-30))
    dead, _ = group_dead_fraction(rand_as, Br, samples=samples, seed=seed + 17)
    return dead
