from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from types import MethodType

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from deadkeys.common.loading import MODEL_IDS, iter_heads, load_model, sanity_check
from deadkeys.common.spectra import dead_fraction

EPSILONS = (0.01, 0.05, 0.1, 0.5, 1.0)
FIXED_RANKS = (60, 56, 48, 40, 32, 24, 16)
NULL_DEPTHS = (5, 10, 20, 32)


@dataclass
class HeadDecomp:
    layer: int
    head: int
    A: torch.Tensor
    B: torch.Tensor
    V_A: torch.Tensor
    V_B: torch.Tensor
    P: torch.Tensor
    Sig: torch.Tensor
    Qt: torch.Tensor
    S_A: torch.Tensor
    S_B: torch.Tensor
    U_A: torch.Tensor
    U_B: torch.Tensor
    q_bias: torch.Tensor
    k_bias: torch.Tensor
    uniform_norm_bound: float


def get_text(tokenizer, max_tokens: int) -> torch.Tensor:
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(x["text"] for x in ds if x["text"].strip())
    ids = tokenizer(text, return_tensors="pt", verbose=False)["input_ids"][0]
    if ids.numel() < max_tokens:
        raise RuntimeError(f"WikiText-2 only yielded {ids.numel()}; need {max_tokens}")
    return ids[:max_tokens]


def hidden_inputs(lm, input_ids: torch.Tensor, max_tokens: int) -> dict[int, torch.Tensor]:
    """Capture QKV module inputs for GPT-2/Pythia without forcing CPU compute."""
    model = lm.model
    max_len = int(getattr(model.config, "n_positions", getattr(model.config, "max_position_embeddings", 2048)))
    device = next(model.parameters()).device
    chunks: dict[int, list[torch.Tensor]] = {i: [] for i in range(lm.n_layers)}
    handles = []

    if lm.tag == "gpt2":
        modules = [block.attn.c_attn for block in model.transformer.h]
    elif lm.tag.startswith("pythia"):
        modules = [layer.attention.query_key_value for layer in model.gpt_neox.layers]
    elif lm.tag in {"qwen25", "qwen3", "openllama7"}:
        modules = [layer.self_attn.q_proj for layer in model.model.layers]
    else:
        raise NotImplementedError(f"Phase 1.5 hidden capture not implemented for {lm.tag}")

    for li, module in enumerate(modules):
        def hook(_module, inputs, _output, li=li):
            chunks[li].append(inputs[0][0].detach())
        handles.append(module.register_forward_hook(hook))

    limit = min(max_tokens, input_ids.numel())
    try:
        with torch.no_grad():
            for start in range(0, limit, max_len):
                ids = input_ids[start : min(start + max_len, limit)].unsqueeze(0).to(device)
                model(ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return {li: torch.cat(parts, dim=0) for li, parts in chunks.items() if parts}


def decompose_heads(lm, *, limit_layers: int | None = None, limit_heads: int | None = None) -> list[HeadDecomp]:
    out = []
    device = next(lm.model.parameters()).device
    for hs in iter_heads(lm, limit_layers=limit_layers, limit_heads=limit_heads):
        A = hs.A.detach().to(device=device, dtype=torch.float32)
        B = hs.B.detach().to(device=device, dtype=torch.float32)
        U_A, S_A, Vh_A = torch.linalg.svd(A, full_matrices=False)
        U_B, S_B, Vh_B = torch.linalg.svd(B, full_matrices=False)
        C = torch.diag(S_A) @ (U_A.T @ U_B) @ torch.diag(S_B)
        P, Sig, Vh_C = torch.linalg.svd(C, full_matrices=False)
        if lm.tag == "gpt2":
            ln = lm.model.transformer.h[hs.layer].ln_1
        elif lm.tag.startswith("pythia"):
            ln = lm.model.gpt_neox.layers[hs.layer].input_layernorm
        elif lm.tag in {"qwen25", "qwen3", "openllama7"}:
            ln = lm.model.model.layers[hs.layer].input_layernorm
        else:
            raise NotImplementedError(f"Phase 1.5 decomposition not implemented for {lm.tag}")
        gamma = ln.weight.detach().to(device=device, dtype=torch.float32)
        ln_bias = getattr(ln, "bias", None)
        beta_norm = 0.0 if ln_bias is None else float(ln_bias.detach().to(device=device, dtype=torch.float32).norm().item())
        norm_bound = float(gamma.abs().max().item() * math.sqrt(lm.d_model) + beta_norm)
        out.append(HeadDecomp(
            layer=hs.layer,
            head=hs.head,
            A=A,
            B=B,
            V_A=Vh_A.T.contiguous(),
            V_B=Vh_B.T.contiguous(),
            P=P.contiguous(),
            Sig=Sig.contiguous(),
            Qt=Vh_C.contiguous(),
            S_A=S_A,
            S_B=S_B,
            U_A=U_A,
            U_B=U_B,
            q_bias=torch.zeros_like(S_A) if hs.q_bias is None else hs.q_bias.detach().to(device=device, dtype=torch.float32),
            k_bias=torch.zeros_like(S_B) if hs.k_bias is None else hs.k_bias.detach().to(device=device, dtype=torch.float32),
            uniform_norm_bound=norm_bound,
        ))
    return out


def rank_for_epsilon(sig: torch.Tensor, q_bound: float, k_bound: float, d_head: int, eps: float) -> tuple[int, float]:
    for r in range(sig.numel() + 1):
        tail = 0.0 if r >= sig.numel() else float(sig[r].item())
        err = tail * q_bound * k_bound / math.sqrt(d_head)
        if err <= eps:
            return r, err
    return sig.numel(), 0.0


def factors_for_rank(hd: HeadDecomp, r: int) -> tuple[torch.Tensor, torch.Tensor]:
    if r == 0:
        return torch.empty(0, hd.A.shape[1], device=hd.A.device), torch.empty(0, hd.A.shape[1], device=hd.A.device)
    root = torch.sqrt(hd.Sig[:r]).diag()
    wq = root @ (hd.V_A @ hd.P[:, :r]).T
    wk = root @ (hd.V_B @ hd.Qt.T[:, :r]).T
    return wq.contiguous(), wk.contiguous()


def observed_dlogit(hd: HeadDecomp, x: torch.Tensor, r: int, d_head: int) -> float:
    wq, wk = factors_for_rank(hd, r)
    x = x.float()
    qlin = x @ hd.A.T
    klin = x @ hd.B.T
    q0 = qlin + hd.q_bias
    k0 = klin + hd.k_bias
    orig = q0 @ k0.T / math.sqrt(d_head)
    if r == 0:
        trunc = torch.zeros_like(orig)
    else:
        q = x @ wq.T
        k = x @ wk.T
        trunc = q @ k.T
        trunc = trunc + (qlin @ hd.k_bias).unsqueeze(1)
        trunc = trunc + (klin @ hd.q_bias).unsqueeze(0)
        trunc = (trunc + torch.dot(hd.q_bias, hd.k_bias)) / math.sqrt(d_head)
    return float((orig - trunc).abs().max().item())


def install_truncated_attention(lm, ranks: dict[tuple[int, int], int], decomps: list[HeadDecomp]) -> None:
    by_layer: dict[int, list[HeadDecomp]] = {}
    for hd in decomps:
        by_layer.setdefault(hd.layer, []).append(hd)

    for li, block in enumerate(lm.model.transformer.h):
        hds = sorted(by_layer[li], key=lambda h: h.head)
        layer_ranks = [ranks[(hd.layer, hd.head)] for hd in hds]
        factors = [(hd, *factors_for_rank(hd, ranks[(hd.layer, hd.head)])) for hd in hds]
        orig_attn = block.attn
        n_heads = lm.n_heads
        d_head = lm.d_head
        same_rank = len(set(layer_ranks)) == 1
        if same_rank:
            r = layer_ranks[0]
            WQ = torch.stack([factors_for_rank(hd, r)[0] for hd in hds], dim=0)
            WK = torch.stack([factors_for_rank(hd, r)[1] for hd in hds], dim=0)
            QB = torch.stack([hd.q_bias for hd in hds], dim=0)
            KB = torch.stack([hd.k_bias for hd in hds], dim=0)
        else:
            WQ = WK = QB = KB = None

        def forward(self, hidden_states, past_key_values=None, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, output_attentions=False, _factors=factors, _same_rank=same_rank, _WQ=WQ, _WK=WK, _QB=QB, _KB=KB, **kwargs):
            if past_key_values is not None or encoder_hidden_states is not None:
                raise RuntimeError("Phase 1.5 patched GPT-2 attention supports no-cache self-attention eval only")
            qkv = self.c_attn(hidden_states)
            query_states, key_states, value_states = qkv.split(self.split_size, dim=2)
            bsz, seq_len, _ = hidden_states.shape
            query_states = query_states.view(bsz, seq_len, n_heads, d_head).transpose(1, 2)
            key_states = key_states.view(bsz, seq_len, n_heads, d_head).transpose(1, 2)
            value_states = value_states.view(bsz, seq_len, n_heads, d_head).transpose(1, 2)
            causal = torch.tril(torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool))
            mask_value = torch.finfo(hidden_states.dtype).min
            if _same_rank:
                WQd = _WQ.to(hidden_states.device, hidden_states.dtype)
                WKd = _WK.to(hidden_states.device, hidden_states.dtype)
                QBd = _QB.to(hidden_states.device, hidden_states.dtype)
                KBd = _KB.to(hidden_states.device, hidden_states.dtype)
                qlin = query_states - QBd[None, :, None, :]
                klin = key_states - KBd[None, :, None, :]
                if WQd.shape[1] == 0:
                    scores = torch.zeros(bsz, n_heads, seq_len, seq_len, device=hidden_states.device, dtype=hidden_states.dtype)
                else:
                    q = torch.einsum("btd,hrd->bhtr", hidden_states, WQd)
                    k = torch.einsum("btd,hrd->bhtr", hidden_states, WKd)
                    scores = torch.einsum("bhtr,bhsr->bhts", q, k)
                scores = scores + torch.einsum("bhtd,hd->bht", qlin, KBd).unsqueeze(3)
                scores = scores + torch.einsum("bhsd,hd->bhs", klin, QBd).unsqueeze(2)
                scores = (scores + torch.einsum("hd,hd->h", QBd, KBd)[None, :, None, None]) / math.sqrt(d_head)
                scores = torch.where(causal[None, None, :, :], scores, torch.tensor(mask_value, device=scores.device, dtype=scores.dtype))
                if attention_mask is not None:
                    scores = scores + attention_mask
                probs = self.attn_dropout(torch.softmax(scores, dim=-1))
                attn_output = torch.einsum("bhts,bhsd->bhtd", probs, value_states).transpose(1, 2).reshape(bsz, seq_len, n_heads * d_head).contiguous()
                attn_weights = probs if output_attentions else None
            else:
                attn_outputs = []
                attn_weights_out = []
                for h, (hd, wq, wk) in enumerate(_factors):
                    q_full = query_states[:, h]
                    k_full = key_states[:, h]
                    q_bias = hd.q_bias.to(hidden_states.device, hidden_states.dtype)
                    k_bias = hd.k_bias.to(hidden_states.device, hidden_states.dtype)
                    qlin = q_full - q_bias
                    klin = k_full - k_bias
                    if wq.numel() == 0:
                        scores = torch.zeros(bsz, seq_len, seq_len, device=hidden_states.device, dtype=hidden_states.dtype)
                    else:
                        q = hidden_states @ wq.to(hidden_states.device, hidden_states.dtype).T
                        k = hidden_states @ wk.to(hidden_states.device, hidden_states.dtype).T
                        scores = q @ k.transpose(-1, -2)
                    scores = scores + (qlin @ k_bias).unsqueeze(2)
                    scores = scores + (klin @ q_bias).unsqueeze(1)
                    scores = (scores + torch.dot(q_bias, k_bias)) / math.sqrt(d_head)
                    scores = torch.where(causal, scores, torch.tensor(mask_value, device=scores.device, dtype=scores.dtype))
                    if attention_mask is not None:
                        scores = scores + attention_mask[:, 0]
                    probs = self.attn_dropout(torch.softmax(scores, dim=-1))
                    attn_outputs.append(probs @ value_states[:, h])
                    if output_attentions:
                        attn_weights_out.append(probs)
                attn_output = torch.stack(attn_outputs, dim=2).reshape(bsz, seq_len, n_heads * d_head).contiguous()
                attn_weights = torch.stack(attn_weights_out, dim=1) if output_attentions else None
            attn_output = self.c_proj(attn_output)
            attn_output = self.resid_dropout(attn_output)
            return attn_output, attn_weights

        block.attn.forward = MethodType(forward, orig_attn)


def perplexity(model, ids: torch.Tensor, stride: int) -> float:
    losses = []
    weights = []
    max_len = int(model.config.n_positions)
    device = next(model.parameters()).device
    with torch.no_grad():
        for start in range(0, max(1, len(ids) - 1), stride):
            end = min(start + max_len, len(ids))
            chunk = ids[start:end].unsqueeze(0).to(device)
            if chunk.shape[1] < 2:
                continue
            out = model(chunk, use_cache=False)
            logits = out.logits[:, :-1]
            labels = chunk[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="mean")
            n = int(labels.numel())
            losses.append(float(loss.item()) * n)
            weights.append(n)
            if end == len(ids):
                break
    return float(math.exp(sum(losses) / sum(weights))) if weights else float("nan")


def covariance_certificate(decomps: list[HeadDecomp], hidden: dict[int, torch.Tensor], d_head: int, q: float) -> pd.DataFrame:
    rows = []
    for hd in decomps:
        x = hidden[hd.layer].float()
        for eps in EPSILONS:
            chosen = hd.Sig.numel()
            chosen_E = 0.0
            chosen_q = 0.0
            chosen_k = 0.0
            for r in range(hd.Sig.numel() + 1):
                if r >= hd.Sig.numel():
                    E = qn = kn = 0.0
                else:
                    q_basis = hd.V_A @ hd.P[:, r:]
                    k_basis = hd.V_B @ hd.Qt.T[:, r:]
                    qn = float(torch.quantile((x @ q_basis).norm(dim=1), q).item())
                    kn = float(torch.quantile((x @ k_basis).norm(dim=1), q).item())
                    E = float(hd.Sig[r].item()) * qn * kn / math.sqrt(d_head)
                if E <= eps:
                    chosen, chosen_E, chosen_q, chosen_k = r, E, qn, kn
                    break
            rows.append({
                "layer": hd.layer, "head": hd.head, "epsilon": eps, "r_h_cov": chosen,
                "E_r_cov": chosen_E, "qproj_quantile": chosen_q, "kproj_quantile": chosen_k,
                "quantile": q, "params_kept_cov": chosen / d_head,
            })
    return pd.DataFrame(rows)


def fixed_ranks(decomps: list[HeadDecomp], r: int) -> dict[tuple[int, int], int]:
    return {(hd.layer, hd.head): min(r, hd.Sig.numel()) for hd in decomps}


def census_guided_ranks(decomps: list[HeadDecomp], census_path: Path | None, dead_samples: int, seed: int) -> dict[tuple[int, int], int]:
    dead: dict[tuple[int, int], float] = {}
    if census_path and census_path.exists():
        df = pd.read_parquet(census_path) if census_path.suffix == ".parquet" else pd.read_csv(census_path)
        df = df[(df["band"] == "all") & (~df.get("is_group_level", False))]
        dead = {(int(r.layer), int(r.head)): float(r.dead_frac) for r in df.itertuples()}
    ranks = {}
    for hd in decomps:
        d = dead.get((hd.layer, hd.head))
        if d is None:
            d, _ = dead_fraction(hd.A, hd.U_B, hd.S_B, samples=dead_samples, seed=seed + 1000 * hd.layer + hd.head)
        target = (1.0 - d) * float(hd.S_B.square().sum().item())
        cum = torch.cumsum(hd.S_B.square(), dim=0)
        r = int(torch.searchsorted(cum, torch.tensor(target, device=cum.device), right=False).item()) + 1
        ranks[(hd.layer, hd.head)] = max(1, min(r, hd.Sig.numel()))
    return ranks


def eval_rank_mode(mode: str, label: str, ranks: dict[tuple[int, int], int], base_ppl: float, ids: torch.Tensor, stride: int, decomps: list[HeadDecomp], d_head: int, device: torch.device) -> dict[str, float | str]:
    if all(r == d_head for r in ranks.values()):
        p = base_ppl
    else:
        lm_eval = load_model("gpt2")
        lm_eval.model.to(device)
        install_truncated_attention(lm_eval, ranks, decomps)
        p = perplexity(lm_eval.model, ids, stride)
    mean_rank = float(np.mean(list(ranks.values())))
    return {
        "mode": mode, "label": label, "epsilon": np.nan, "baseline_ppl": base_ppl,
        "truncated_ppl": p, "ppl_delta": p - base_ppl, "mean_rank_kept": mean_rank,
        "params_kept": mean_rank / d_head, "compression_frac": 1 - mean_rank / d_head,
        "eval_tokens": len(ids),
    }


def _batched_random_orthogonal(samples: int, dim: int, gen: torch.Generator, device: torch.device) -> torch.Tensor:
    """Generate Haar-ish orthogonal bases in one QR call on the target device."""
    q, r = torch.linalg.qr(torch.randn(samples, dim, dim, generator=gen, device=device))
    signs = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    signs[signs == 0] = 1
    return q * signs.unsqueeze(-2)


def null_model(decomps: list[HeadDecomp], samples: int, dead_samples: int, depths: tuple[int, ...], seed: int) -> pd.DataFrame:
    rows = []
    device = decomps[0].A.device if decomps else torch.device("cpu")
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    for hi, hd in enumerate(decomps, start=1):
        print(f"null_model head {hi}/{len(decomps)} layer={hd.layer} head={hd.head} device={device}", flush=True)
        measured_dead, _ = dead_fraction(hd.A, hd.U_B, hd.S_B, samples=dead_samples, seed=seed + 1000 * hd.layer + hd.head)
        dim = hd.A.shape[0]
        for depth in depths:
            k = min(depth, dim)
            parks = torch.sum(hd.U_A[:, :k] * hd.U_B[:, :k], dim=0).abs().clamp(max=0.999)
            QA = _batched_random_orthogonal(samples, dim, gen, device)
            QB = torch.empty_like(QA)
            used: list[torch.Tensor] = []
            for i, p in enumerate(parks):
                base = QA[:, :, i]
                noise = torch.randn(samples, dim, generator=gen, device=device)
                for u in used + [base]:
                    noise = noise - (noise * u).sum(dim=1, keepdim=True) * u
                noise = noise / noise.norm(dim=1, keepdim=True).clamp_min(1e-12)
                QB[:, :, i] = p * base + torch.sqrt(1 - p * p) * noise
                used.append(QB[:, :, i])
            if k < dim:
                rest = torch.randn(samples, dim, dim - k, generator=gen, device=device)
                for u in [QA[:, :, i] for i in range(k)] + used:
                    rest = rest - u.unsqueeze(2) * torch.bmm(u.unsqueeze(1), rest)
                Qrest, _ = torch.linalg.qr(rest)
                QB[:, :, k:] = Qrest[:, :, : dim - k]
            # Build synthetic heads directly from their left singular bases.
            # This preserves the intended QA/QB alignment while avoiding an
            # extra per-sample SVD in the null model hot path. QA/QB generation
            # is batched, so CUDA does one larger QR instead of many tiny ones.
            A_syn = QA * hd.S_A.view(1, 1, dim)
            vals = []
            for si in range(samples):
                d, _ = dead_fraction(A_syn[si], QB[si], hd.S_B, samples=dead_samples, seed=seed + 17 * si + depth)
                vals.append(d)
            arr = np.array(vals, dtype=float)
            std = float(arr.std(ddof=1)) if len(arr) > 1 else float("nan")
            mean = float(arr.mean()) if len(arr) else float("nan")
            rows.append({
                "layer": hd.layer, "head": hd.head, "k_align": depth,
                "dead_frac": measured_dead, "null_mean": mean, "null_std": std,
                "z": (measured_dead - mean) / std if std and std > 0 else float("nan"),
            })
    return pd.DataFrame(rows)


def plot_outputs(out: Path, model: str) -> None:
    trunc = pd.read_csv(out / f"truncation_{model}.csv")
    ppl_path = out / f"ppl_curve_{model}.csv"
    null = pd.read_csv(out / f"null_model_{model}.csv")
    cov = pd.read_csv(out / f"covariance_certificate_{model}.csv")

    if not ppl_path.exists():
        return
    ppl = pd.read_csv(ppl_path)

    fig, ax = plt.subplots(figsize=(7, 4))
    for mode, df in ppl.groupby("mode"):
        ax.plot(df["compression_frac"], df["ppl_delta"], marker="o", label=mode)
    ax.set_xlabel("QK rank compression fraction")
    ax.set_ylabel("PPL delta")
    ax.set_title("GPT-2 Phase 1.5 perplexity curve")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "ppl_curve_gpt2.png", dpi=150); fig.savefig(out / "ppl_curve_gpt2.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    trunc.groupby("epsilon")["params_kept"].mean().plot(kind="bar", ax=ax)
    ax.set_ylabel("mean rank fraction kept")
    ax.set_title("Uniform certificate rank kept")
    fig.tight_layout(); fig.savefig(out / "truncation_gpt2.png", dpi=150); fig.savefig(out / "truncation_gpt2.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    cov.groupby("epsilon")["params_kept_cov"].mean().plot(kind="bar", ax=ax)
    ax.set_ylabel("mean rank fraction kept")
    ax.set_title("Covariance-weighted certificate rank kept")
    fig.tight_layout(); fig.savefig(out / "covariance_certificate_gpt2.png", dpi=150); fig.savefig(out / "covariance_certificate_gpt2.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for depth, df in null.groupby("k_align"):
        ax.scatter(df["null_mean"], df["dead_frac"], s=10, label=f"k={depth}", alpha=0.7)
    ax.plot([0, 1], [0, 1], color="black", linewidth=1)
    ax.set_xlabel("null mean dead_frac")
    ax.set_ylabel("measured dead_frac")
    ax.set_title("Alignment-entailment null by depth")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "null_model_gpt2.png", dpi=150); fig.savefig(out / "null_model_gpt2.pdf"); plt.close(fig)


def run(args: argparse.Namespace) -> None:
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")
    lm = load_model(args.model, device=device)
    print("sanity", sanity_check(lm))
    print(f"device {device}; model_device={next(lm.model.parameters()).device}; cuda_available={torch.cuda.is_available()}", flush=True)
    ids = get_text(lm.tokenizer, args.eval_tokens)
    if ids.numel() < 200_000 and not args.allow_smoke_under_200k:
        raise RuntimeError("Phase 1.5 spec requires >=200k eval tokens; pass --allow-smoke-under-200k for bounded smoke tests")
    cal_ids = ids[: min(args.calibration_tokens, len(ids))]
    decomps = decompose_heads(lm, limit_layers=args.limit_layers, limit_heads=args.limit_heads)
    hidden = hidden_inputs(lm, cal_ids, min(args.calibration_tokens, len(cal_ids)))
    observed_hidden = {li: x[: min(args.observed_tokens, x.shape[0])] for li, x in hidden.items()}

    trunc_rows = []
    ranks_by_eps: dict[float, dict[tuple[int, int], int]] = {e: {} for e in EPSILONS}
    for hd in decomps:
        x_obs = observed_hidden[hd.layer]
        x_cal = hidden[hd.layer]
        dist_norm = float(torch.quantile(x_cal.norm(dim=1), 0.999).item()) if x_cal.shape[0] else float("nan")
        for eps in EPSILONS:
            r, E = rank_for_epsilon(hd.Sig, hd.uniform_norm_bound, hd.uniform_norm_bound, lm.d_head, eps)
            r_dist, E_dist = rank_for_epsilon(hd.Sig, dist_norm, dist_norm, lm.d_head, eps)
            obs = observed_dlogit(hd, x_obs, r, lm.d_head)
            ranks_by_eps[eps][(hd.layer, hd.head)] = r
            trunc_rows.append({
                "layer": hd.layer, "head": hd.head, "epsilon": eps, "r_h": r, "E_r": E,
                "r_h_dist": r_dist, "E_r_dist": E_dist, "dist_norm_p999": dist_norm,
                "observed_max_dlogit": obs, "params_kept": r / lm.d_head,
                "observed_le_certified": bool(obs <= E + 1e-3),
            })
    trunc_df = pd.DataFrame(trunc_rows)
    trunc_df.to_csv(out / f"truncation_{args.model}.csv", index=False)

    cov_df = covariance_certificate(decomps, hidden, lm.d_head, args.cert_quantile)
    cov_df.to_csv(out / f"covariance_certificate_{args.model}.csv", index=False)

    if args.skip_ppl or args.model != "gpt2":
        if args.model != "gpt2" and not args.skip_ppl:
            print(f"skipping PPL/truncated-attention eval: currently implemented only for gpt2, not {args.model}")
        null_df = null_model(decomps, samples=args.null_samples, dead_samples=args.null_dead_samples, depths=tuple(args.null_depths), seed=args.seed)
        null_df.to_csv(out / f"null_model_{args.model}.csv", index=False)
        plot_outputs(out, args.model)
        print(f"wrote Phase 1.5 certificate/null outputs to {out}")
        return

    base_ppl = perplexity(lm.model, ids, args.stride)
    ppl_rows = []

    def append_ppl(row: dict) -> None:
        ppl_rows.append(row)
        pd.DataFrame(ppl_rows).to_csv(out / f"ppl_curve_{args.model}.csv", index=False)

    for eps in EPSILONS:
        if np.mean(list(ranks_by_eps[eps].values())) == lm.d_head:
            print(f"ANOMALY: no uniform-certificate truncation occurred at epsilon={eps}")
        append_ppl(eval_rank_mode("uniform_certificate", f"eps={eps}", ranks_by_eps[eps], base_ppl, ids, args.stride, decomps, lm.d_head, device) | {"epsilon": eps})
        print(f"uniform eps={eps} done")

    for r in FIXED_RANKS:
        row = eval_rank_mode("fixed_rank", f"r={r}", fixed_ranks(decomps, r), base_ppl, ids, args.stride, decomps, lm.d_head, device)
        append_ppl(row)
        print(f"fixed rank {r} ppl_delta={row['ppl_delta']:.4g}")

    guided = census_guided_ranks(decomps, Path(args.census), args.null_dead_samples, args.seed)
    guided_mean = int(round(float(np.mean(list(guided.values())))))
    append_ppl(eval_rank_mode("census_guided", "sb_energy_1_minus_dead_frac", guided, base_ppl, ids, args.stride, decomps, lm.d_head, device))
    append_ppl(eval_rank_mode("matched_uniform_control", f"r={guided_mean}", fixed_ranks(decomps, guided_mean), base_ppl, ids, args.stride, decomps, lm.d_head, device))

    null_df = null_model(decomps, samples=args.null_samples, dead_samples=args.null_dead_samples, depths=tuple(args.null_depths), seed=args.seed)
    null_df.to_csv(out / f"null_model_{args.model}.csv", index=False)
    plot_outputs(out, args.model)
    print(f"wrote Phase 1.5 outputs to {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1.5 certified and empirical QK truncation")
    p.add_argument("--model", default="gpt2", choices=sorted(MODEL_IDS))
    p.add_argument("--output-dir", default="outputs/phase1_5")
    p.add_argument("--eval-tokens", type=int, default=200_000)
    p.add_argument("--calibration-tokens", type=int, default=10_000)
    p.add_argument("--observed-tokens", type=int, default=10_000)
    p.add_argument("--stride", type=int, default=512)
    p.add_argument("--null-samples", type=int, default=500)
    p.add_argument("--null-dead-samples", type=int, default=10_000)
    p.add_argument("--null-depths", type=int, nargs="+", default=list(NULL_DEPTHS))
    p.add_argument("--cert-quantile", type=float, default=0.999)
    p.add_argument("--census", default="outputs/census_gpt2.parquet")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cpu", help="torch device for model/eval work, e.g. cpu or cuda")
    p.add_argument("--limit-layers", type=int, help="smoke-test limit on layers to decompose")
    p.add_argument("--limit-heads", type=int, help="smoke-test limit on heads to decompose")
    p.add_argument("--allow-smoke-under-200k", action="store_true", help="allow bounded smoke tests below the spec's 200k eval-token floor")
    p.add_argument("--skip-ppl", action="store_true", help="write certificate/null outputs only; useful for non-GPT-2 smoke tests")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
