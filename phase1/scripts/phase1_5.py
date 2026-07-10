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

from phase1.common.loading import iter_heads, load_model, sanity_check
from phase1.common.spectra import dead_fraction

EPSILONS = (0.01, 0.05, 0.1, 0.5, 1.0)


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
    try:
        from datasets import load_dataset

        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(x["text"] for x in ds if x["text"].strip())
    except Exception as exc:
        print(f"dataset unavailable, using fallback text: {exc}")
        text = ("The quick brown fox jumps over the lazy dog. "
                "This fallback corpus is only for smoke testing Phase 1.5.\n") * 2000
    ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
    return ids[:max_tokens]


def gpt2_hidden_inputs(model, input_ids: torch.Tensor, max_tokens: int) -> dict[int, torch.Tensor]:
    ids = input_ids[:max_tokens].unsqueeze(0)
    pos = torch.arange(ids.shape[1]).unsqueeze(0)
    with torch.no_grad():
        hidden = model.transformer.wte(ids) + model.transformer.wpe(pos)
        out = {}
        for li, block in enumerate(model.transformer.h):
            x = block.ln_1(hidden)
            out[li] = x[0].detach().cpu()
            attn_out, _ = block.attn(x, output_attentions=False)
            hidden = hidden + attn_out
            mlp_in = block.ln_2(hidden)
            hidden = hidden + block.mlp(mlp_in)
    return out


def decompose_heads(lm) -> list[HeadDecomp]:
    by_layer = {hs.layer: [] for hs in iter_heads(lm)}
    heads = list(iter_heads(lm))
    out = []
    for hs in heads:
        A = hs.A.float().cpu()
        B = hs.B.float().cpu()
        U_A, S_A, Vh_A = torch.linalg.svd(A, full_matrices=False)
        U_B, S_B, Vh_B = torch.linalg.svd(B, full_matrices=False)
        C = torch.diag(S_A) @ (U_A.T @ U_B) @ torch.diag(S_B)
        P, Sig, Vh_C = torch.linalg.svd(C, full_matrices=False)
        block = lm.model.transformer.h[hs.layer]
        gamma = block.ln_1.weight.detach().float().cpu()
        beta = block.ln_1.bias.detach().float().cpu()
        norm_bound = float(gamma.abs().max().item() * math.sqrt(lm.d_model) + beta.norm().item())
        out.append(
            HeadDecomp(
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
                q_bias=torch.zeros_like(S_A) if hs.q_bias is None else hs.q_bias.float().cpu(),
                k_bias=torch.zeros_like(S_B) if hs.k_bias is None else hs.k_bias.float().cpu(),
                uniform_norm_bound=norm_bound,
            )
        )
    return out


def rank_for_epsilon(sig: torch.Tensor, norm_bound: float, d_head: int, eps: float) -> tuple[int, float]:
    # Error after keeping r components is Sig[r] (0-index) or 0 at full rank.
    for r in range(sig.numel() + 1):
        tail = 0.0 if r >= sig.numel() else float(sig[r].item())
        err = tail * norm_bound * norm_bound / math.sqrt(d_head)
        if err <= eps:
            return r, err
    return sig.numel(), 0.0


def factors_for_rank(hd: HeadDecomp, r: int) -> tuple[torch.Tensor, torch.Tensor]:
    if r == 0:
        return torch.empty(0, hd.A.shape[1]), torch.empty(0, hd.A.shape[1])
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
        factors = [(hd, *factors_for_rank(hd, ranks[(hd.layer, hd.head)])) for hd in hds]
        orig_attn = block.attn
        n_heads = lm.n_heads
        d_head = lm.d_head

        def forward(self, hidden_states, past_key_values=None, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, output_attentions=False, _factors=factors, **kwargs):
            if past_key_values is not None or encoder_hidden_states is not None:
                raise RuntimeError("Phase 1.5 patched GPT-2 attention supports no-cache self-attention eval only")
            qkv = self.c_attn(hidden_states)
            query_states, key_states, value_states = qkv.split(self.split_size, dim=2)
            bsz, seq_len, _ = hidden_states.shape
            query_states = query_states.view(bsz, seq_len, n_heads, d_head).transpose(1, 2)
            key_states = key_states.view(bsz, seq_len, n_heads, d_head).transpose(1, 2)
            value_states = value_states.view(bsz, seq_len, n_heads, d_head).transpose(1, 2)
            attn_outputs = []
            attn_weights_out = []
            causal = torch.tril(torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool))
            mask_value = torch.finfo(hidden_states.dtype).min
            for h, (hd, wq, wk) in enumerate(_factors):
                wq = wq.to(hidden_states.device, hidden_states.dtype)
                wk = wk.to(hidden_states.device, hidden_states.dtype)
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
                probs = torch.softmax(scores, dim=-1)
                probs = self.attn_dropout(probs)
                out = probs @ value_states[:, h]
                attn_outputs.append(out)
                if output_attentions:
                    attn_weights_out.append(probs)
            attn_output = torch.stack(attn_outputs, dim=2).reshape(bsz, seq_len, n_heads * d_head).contiguous()
            attn_output = self.c_proj(attn_output)
            attn_output = self.resid_dropout(attn_output)
            attn_weights = torch.stack(attn_weights_out, dim=1) if output_attentions else None
            return attn_output, attn_weights

        block.attn.forward = MethodType(forward, orig_attn)


def perplexity(model, ids: torch.Tensor, stride: int) -> float:
    losses = []
    max_len = int(model.config.n_positions)
    with torch.no_grad():
        for start in range(0, max(1, len(ids) - 1), stride):
            end = min(start + max_len, len(ids))
            chunk = ids[start:end].unsqueeze(0)
            if chunk.shape[1] < 2:
                continue
            out = model(chunk, use_cache=False)
            logits = out.logits[:, :-1]
            labels = chunk[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="mean")
            losses.append(float(loss.item()))
            if end == len(ids):
                break
    return float(math.exp(sum(losses) / len(losses))) if losses else float("nan")


def null_model(decomps: list[HeadDecomp], samples: int, dead_samples: int, seed: int) -> pd.DataFrame:
    rows = []
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    for hd in decomps:
        measured_dead, _ = dead_fraction(hd.A, hd.U_B, hd.S_B, samples=dead_samples, seed=seed + 1000 * hd.layer + hd.head)
        vals = []
        # Approximate alignment-entailment null: preserve spectra and park0..4 as diagonal overlaps.
        parks = torch.sum(hd.U_A[:, :5] * hd.U_B[:, :5], dim=0).abs().clamp(max=0.999)
        for _ in range(samples):
            QA, _ = torch.linalg.qr(torch.randn(hd.A.shape[0], hd.A.shape[0], generator=gen))
            QB = torch.empty_like(QA)
            used = []
            for i, p in enumerate(parks):
                base = QA[:, i]
                noise = torch.randn(hd.A.shape[0], generator=gen)
                for u in used + [base]:
                    noise = noise - (noise @ u) * u
                noise = noise / noise.norm().clamp_min(1e-12)
                QB[:, i] = p * base + torch.sqrt(1 - p * p) * noise
                used.append(QB[:, i])
            if len(parks) < hd.A.shape[0]:
                rest = torch.randn(hd.A.shape[0], hd.A.shape[0] - len(parks), generator=gen)
                for u in list(QA[:, : len(parks)].T) + used:
                    rest = rest - u[:, None] @ (u[None, :] @ rest)
                Qrest, _ = torch.linalg.qr(rest)
                QB[:, len(parks):] = Qrest[:, : hd.A.shape[0] - len(parks)]
            A_syn = torch.diag(hd.S_A) @ QA.T
            B_syn = torch.diag(hd.S_B) @ QB.T
            U_Bs, S_Bs, _ = torch.linalg.svd(B_syn, full_matrices=False)
            d, _ = dead_fraction(A_syn, U_Bs, S_Bs, samples=dead_samples, seed=seed + len(vals))
            vals.append(d)
        arr = np.array(vals, dtype=float)
        std = float(arr.std(ddof=1)) if len(arr) > 1 else float("nan")
        rows.append({
            "layer": hd.layer,
            "head": hd.head,
            "dead_frac": measured_dead,
            "null_mean": float(arr.mean()) if len(arr) else float("nan"),
            "null_std": std,
            "z": (measured_dead - float(arr.mean())) / std if std and std > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


def plot_outputs(out: Path) -> None:
    trunc = pd.read_csv(out / "truncation_gpt2.csv")
    ppl = pd.read_csv(out / "ppl_curve_gpt2.csv")
    null = pd.read_csv(out / "null_model_gpt2.csv")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ppl["compression_frac"], ppl["ppl_delta"], marker="o")
    for _, row in ppl.iterrows():
        ax.annotate(str(row["epsilon"]), (row["compression_frac"], row["ppl_delta"]))
    ax.set_xlabel("QK rank compression fraction")
    ax.set_ylabel("PPL delta")
    ax.set_title("GPT-2 Phase 1.5 perplexity curve")
    fig.tight_layout(); fig.savefig(out / "ppl_curve_gpt2.png", dpi=150); fig.savefig(out / "ppl_curve_gpt2.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    trunc.groupby("epsilon")["params_kept"].mean().plot(kind="bar", ax=ax)
    ax.set_ylabel("mean rank fraction kept")
    ax.set_title("GPT-2 truncation rank kept")
    fig.tight_layout(); fig.savefig(out / "truncation_gpt2.png", dpi=150); fig.savefig(out / "truncation_gpt2.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(null["null_mean"], null["dead_frac"], s=12)
    ax.plot([0, 1], [0, 1], color="black", linewidth=1)
    ax.set_xlabel("null mean dead_frac")
    ax.set_ylabel("measured dead_frac")
    ax.set_title("GPT-2 alignment-entailment null")
    fig.tight_layout(); fig.savefig(out / "null_model_gpt2.png", dpi=150); fig.savefig(out / "null_model_gpt2.pdf"); plt.close(fig)


def run(args: argparse.Namespace) -> None:
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    lm = load_model("gpt2")
    print("sanity", sanity_check(lm))
    ids = get_text(lm.tokenizer, args.eval_tokens)
    cal_ids = ids[: min(args.calibration_tokens, len(ids))]
    decomps = decompose_heads(lm)
    hidden = gpt2_hidden_inputs(lm.model, cal_ids, min(args.observed_tokens, len(cal_ids)))

    trunc_rows = []
    ranks_by_eps: dict[float, dict[tuple[int, int], int]] = {e: {} for e in EPSILONS}
    for hd in decomps:
        x = hidden[hd.layer]
        dist_norm = float(torch.quantile(x.norm(dim=1), 0.999).item()) if x.shape[0] else float("nan")
        for eps in EPSILONS:
            r, E = rank_for_epsilon(hd.Sig, hd.uniform_norm_bound, lm.d_head, eps)
            r_dist, E_dist = rank_for_epsilon(hd.Sig, dist_norm, lm.d_head, eps)
            obs = observed_dlogit(hd, x, r, lm.d_head)
            ranks_by_eps[eps][(hd.layer, hd.head)] = r
            trunc_rows.append({
                "layer": hd.layer, "head": hd.head, "epsilon": eps, "r_h": r, "E_r": E,
                "r_h_dist": r_dist, "E_r_dist": E_dist, "dist_norm_p999": dist_norm,
                "observed_max_dlogit": obs, "params_kept": r / lm.d_head,
                "observed_le_certified": bool(obs <= E + 1e-3),
            })
    trunc_df = pd.DataFrame(trunc_rows)
    trunc_df.to_csv(out / "truncation_gpt2.csv", index=False)

    base_ppl = perplexity(lm.model, ids, args.stride)
    ppl_rows = []
    for eps in EPSILONS:
        lm_eps = load_model("gpt2")
        install_truncated_attention(lm_eps, ranks_by_eps[eps], decomps)
        p = perplexity(lm_eps.model, ids, args.stride)
        mean_rank = float(np.mean(list(ranks_by_eps[eps].values())))
        ppl_rows.append({
            "epsilon": eps, "baseline_ppl": base_ppl, "truncated_ppl": p, "ppl_delta": p - base_ppl,
            "mean_rank_kept": mean_rank, "params_kept": mean_rank / lm.d_head,
            "compression_frac": 1 - mean_rank / lm.d_head, "eval_tokens": len(ids),
        })
        print(f"eps={eps} ppl={p:.4g} delta={p-base_ppl:.4g} mean_rank={mean_rank:.2f}")
    pd.DataFrame(ppl_rows).to_csv(out / "ppl_curve_gpt2.csv", index=False)

    null_df = null_model(decomps, samples=args.null_samples, dead_samples=args.null_dead_samples, seed=args.seed)
    null_df.to_csv(out / "null_model_gpt2.csv", index=False)
    plot_outputs(out)
    print(f"wrote Phase 1.5 outputs to {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1.5 certified QK truncation for GPT-2")
    p.add_argument("--output-dir", default="outputs/phase1_5")
    p.add_argument("--eval-tokens", type=int, default=4096)
    p.add_argument("--calibration-tokens", type=int, default=4096)
    p.add_argument("--observed-tokens", type=int, default=256)
    p.add_argument("--stride", type=int, default=512)
    p.add_argument("--null-samples", type=int, default=25)
    p.add_argument("--null-dead-samples", type=int, default=1024)
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
