from __future__ import annotations

import argparse
import copy
import math
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from deadkeys.common.loading import load_model

EPSILONS = (0.05, 0.1, 0.5, 1.0)
FRACTIONS = (0.10, 0.25, 0.50)
NEEDLE = "Qwen3 plane truncation calibration sample. "


def get_text(tokenizer, max_tokens: int, *, mix_repeats: int = 1) -> torch.Tensor:
    """Return a deterministic WikiText + chat/code-ish calibration/eval stream."""
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    wiki = "\n\n".join(x["text"] for x in ds if x["text"].strip())
    synthetic = (
        "\nUser: Please inspect the cache eviction routine.\n"
        "Assistant: The function should preserve keys while compacting values.\n"
        "```python\ndef update_cache(key, value):\n    cache[key] = value\n    return cache.get(key)\n```\n"
    )
    text = (wiki + "\n\n" + synthetic * 64 + NEEDLE) * mix_repeats
    ids = tokenizer(text, return_tensors="pt", verbose=False)["input_ids"][0]
    if ids.numel() < max_tokens:
        raise RuntimeError(f"text mix yielded {ids.numel()} tokens; need {max_tokens}")
    return ids[:max_tokens]


def _module_lists(lm):
    layers = lm.model.model.layers
    return [layer.self_attn.q_proj for layer in layers], [layer.self_attn.k_proj for layer in layers]


def capture_qk_norms(lm, ids: torch.Tensor, max_tokens: int, limit_layers: int | None, forward_tokens: int, observed_tokens: int) -> dict[tuple[int, str], torch.Tensor]:
    """Capture compact post-q_norm/k_norm, pre-RoPE calibration statistics.

    Store per-plane norms for all calibration tokens, plus full q/k activations
    only for the observed-token sanity subset. This keeps the 100k-token Qwen3
    run within L4 host memory.
    """
    if lm.tag != "qwen3":
        raise NotImplementedError("plane truncation task is currently registered for qwen3 only")
    model = lm.model
    device = next(model.parameters()).device
    max_len = min(int(getattr(model.config, "max_position_embeddings", 4096)), forward_tokens)
    qs, ks = _module_lists(lm)
    layers = model.model.layers
    n_layers = min(lm.n_layers, limit_layers or lm.n_layers)
    chunks: dict[tuple[int, str], list[torch.Tensor]] = {(i, "q_norms"): [] for i in range(n_layers)}
    chunks.update({(i, "k_norms"): [] for i in range(n_layers)})
    chunks.update({(i, "q_obs"): [] for i in range(n_layers)})
    chunks.update({(i, "k_obs"): [] for i in range(n_layers)})
    seen = {i: 0 for i in range(n_layers)}
    obs_limit = min(observed_tokens, max_tokens)
    handles = []

    for li in range(n_layers):
        q_norm = layers[li].self_attn.q_norm
        k_norm = layers[li].self_attn.k_norm

        def q_hook(_module, _inputs, output, li=li, q_norm=q_norm):
            q = output[0].detach().view(-1, lm.n_heads, lm.d_head).float()
            q = q_norm(q)
            chunks[(li, "q_norms")].append(torch.linalg.vector_norm(q.view(q.shape[0], lm.n_heads, lm.d_head // 2, 2), dim=-1).cpu())
            remain = max(0, obs_limit - seen[li])
            if remain:
                chunks[(li, "q_obs")].append(q[:remain].cpu())

        def k_hook(_module, _inputs, output, li=li, k_norm=k_norm):
            k = output[0].detach().view(-1, lm.n_kv_heads, lm.d_head).float()
            k = k_norm(k)
            chunks[(li, "k_norms")].append(torch.linalg.vector_norm(k.view(k.shape[0], lm.n_kv_heads, lm.d_head // 2, 2), dim=-1).cpu())
            remain = max(0, obs_limit - seen[li])
            if remain:
                chunks[(li, "k_obs")].append(k[:remain].cpu())
            seen[li] += k.shape[0]

        handles.append(qs[li].register_forward_hook(q_hook))
        handles.append(ks[li].register_forward_hook(k_hook))

    limit = min(max_tokens, ids.numel())
    try:
        with torch.inference_mode():
            for start in range(0, limit, max_len):
                chunk = ids[start : min(start + max_len, limit)].unsqueeze(0).to(device)
                model(chunk, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return {key: torch.cat(parts, dim=0) for key, parts in chunks.items() if parts}

def _observed_plane_dlogit(q: torch.Tensor, k: torch.Tensor, planes: list[int], d_head: int, chunk: int = 256) -> float:
    if not planes:
        return 0.0
    dims = []
    for p in planes:
        dims.extend([2 * p, 2 * p + 1])
    qd = q[:, dims].float()
    kd = k[:, dims].float()
    max_abs = 0.0
    scale = math.sqrt(d_head)
    for start in range(0, qd.shape[0], chunk):
        vals = qd[start : start + chunk] @ kd.T / scale
        max_abs = max(max_abs, float(vals.abs().max().item()))
    return max_abs


def select_planes(lm, captures: dict[tuple[int, str], torch.Tensor], epsilons: tuple[float, ...], observed_tokens: int) -> pd.DataFrame:
    rows = []
    group = lm.n_heads // lm.n_kv_heads
    n_planes = lm.d_head // 2
    for li in range(lm.n_layers):
        if (li, "q_norms") not in captures:
            continue
        q_norms = captures[(li, "q_norms")]
        k_norms = captures[(li, "k_norms")]
        q_obs = captures[(li, "q_obs")]
        k_obs = captures[(li, "k_obs")]
        q_p999 = torch.quantile(q_norms, 0.999, dim=0)
        k_p999 = torch.quantile(k_norms, 0.999, dim=0)
        for h in range(lm.n_heads):
            kv = h // group
            b = q_p999[h] * k_p999[kv] / math.sqrt(lm.d_head)
            order = torch.argsort(b)
            for eps in epsilons:
                total = 0.0
                dropped = []
                for p in order.tolist():
                    cand = total + float(b[p].item())
                    if cand > eps:
                        break
                    total = cand
                    dropped.append(p)
                observed = _observed_plane_dlogit(q_obs[:observed_tokens, h].reshape(-1, lm.d_head), k_obs[:observed_tokens, kv].reshape(-1, lm.d_head), dropped, lm.d_head)
                rows.append({
                    "layer": li,
                    "head": h,
                    "kv_head": kv,
                    "epsilon": eps,
                    "planes_dropped": len(dropped),
                    "dropped_planes": " ".join(map(str, dropped)),
                    "certified_sum": total,
                    "observed_max_dlogit": observed,
                    "cache_width_saved_frac": len(dropped) / n_planes,
                    "ok": bool(observed <= total + 1e-3),
                })
    return pd.DataFrame(rows)


def zero_planes_inplace(lm, plan: pd.DataFrame, *, epsilon: float | None = None, fraction: float | None = None, random_control: bool = False, seed: int = 1234) -> None:
    rng = random.Random(seed)
    n_planes = lm.d_head // 2
    layers = lm.model.model.layers
    group = lm.n_heads // lm.n_kv_heads
    for li, layer in enumerate(layers):
        layer_plan = plan[plan["layer"] == li]
        if epsilon is not None:
            layer_plan = layer_plan[layer_plan["epsilon"] == epsilon]
        if layer_plan.empty:
            continue
        q_mask = torch.ones_like(layer.self_attn.q_proj.weight.data)
        k_mask = torch.ones_like(layer.self_attn.k_proj.weight.data)
        kv_drop: dict[int, set[int]] = {kv: set(range(n_planes)) for kv in range(lm.n_kv_heads)}
        for row in layer_plan.itertuples():
            h = int(row.head)
            if fraction is None:
                dropped = [int(x) for x in str(row.dropped_planes).split() if x]
            else:
                count = int(round(fraction * n_planes))
                if random_control:
                    dropped = sorted(rng.sample(range(n_planes), count))
                else:
                    scored = [int(x) for x in str(row.dropped_planes).split() if x]
                    rest = [p for p in range(n_planes) if p not in scored]
                    dropped = (scored + rest)[:count]
            for p in dropped:
                q_mask[h * lm.d_head + 2 * p : h * lm.d_head + 2 * p + 2, :] = 0
            kv_drop[int(row.kv_head)] &= set(dropped)
        for kv, dropped in kv_drop.items():
            for p in dropped:
                k_mask[kv * lm.d_head + 2 * p : kv * lm.d_head + 2 * p + 2, :] = 0
        layer.self_attn.q_proj.weight.data.mul_(q_mask)
        layer.self_attn.k_proj.weight.data.mul_(k_mask)


def perplexity(model, ids: torch.Tensor, stride: int, forward_tokens: int) -> float:
    losses = []
    weights = []
    max_len = min(int(getattr(model.config, "max_position_embeddings", 4096)), forward_tokens)
    device = next(model.parameters()).device
    with torch.inference_mode():
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


def run(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    lm = load_model(args.model, device=device)
    if lm.tag != "qwen3":
        raise NotImplementedError("Task 1 is scoped to qwen3")
    ids = get_text(lm.tokenizer, max(args.eval_tokens, args.calibration_tokens))
    if args.eval_tokens < 200_000 and not args.allow_smoke_under_200k:
        raise RuntimeError("Task 1 requires >=200k eval tokens; pass --allow-smoke-under-200k for smoke tests")
    captures = capture_qk_norms(lm, ids[: args.calibration_tokens], args.calibration_tokens, args.limit_layers, args.forward_tokens, args.observed_tokens)
    plan = select_planes(lm, captures, tuple(args.epsilons), args.observed_tokens)
    if not bool(plan["ok"].all()):
        bad = plan[~plan["ok"]].head()
        raise RuntimeError(f"observed plane dlogit exceeded certificate; first failures:\n{bad}")
    plan.to_csv(out / "plane_truncation_qwen3.csv", index=False)

    eval_ids = ids[: args.eval_tokens]
    base_ppl = perplexity(lm.model, eval_ids, args.stride, args.forward_tokens)
    ppl_rows = [{"mode": "base", "label": "base", "baseline_ppl": base_ppl, "truncated_ppl": base_ppl, "ppl_delta": 0.0, "eval_tokens": len(eval_ids)}]

    for eps in args.epsilons:
        lm_eval = copy.deepcopy(lm)
        zero_planes_inplace(lm_eval, plan, epsilon=eps, seed=args.seed)
        p = perplexity(lm_eval.model, eval_ids, args.stride, args.forward_tokens)
        ppl_rows.append({"mode": "certified", "label": f"eps={eps}", "epsilon": eps, "baseline_ppl": base_ppl, "truncated_ppl": p, "ppl_delta": p - base_ppl, "eval_tokens": len(eval_ids)})
        pd.DataFrame(ppl_rows).to_csv(out / "ppl_planes_qwen3.csv", index=False)

    eps05 = plan[plan["epsilon"] == 0.5]
    matched = float(eps05["cache_width_saved_frac"].mean()) if not eps05.empty else 0.0
    for frac in list(args.fractions) + ([matched] if matched > 0 else []):
        for random_control in (False, True):
            lm_eval = copy.deepcopy(lm)
            zero_planes_inplace(lm_eval, plan[plan["epsilon"] == 1.0], fraction=frac, random_control=random_control, seed=args.seed)
            p = perplexity(lm_eval.model, eval_ids, args.stride, args.forward_tokens)
            ppl_rows.append({"mode": "random_control" if random_control else "bound_fraction", "label": f"frac={frac:.3f}", "fraction": frac, "baseline_ppl": base_ppl, "truncated_ppl": p, "ppl_delta": p - base_ppl, "eval_tokens": len(eval_ids)})
            pd.DataFrame(ppl_rows).to_csv(out / "ppl_planes_qwen3.csv", index=False)

    report = out / "REPORT.md"
    report.write_text(
        "# Task 1 — Plane-preserving Qwen3 truncation\n\n"
        f"Model: {args.model}\n\nCalibration tokens: {args.calibration_tokens}\n\n"
        f"Eval tokens: {args.eval_tokens}\n\nObserved sanity tokens: {args.observed_tokens}\n\n"
        "Mix: WikiText-2 test plus deterministic chat/code sample.\n\n"
        "Patch: zeroed whole q_proj/k_proj rotary-plane rows; v_proj and o_proj untouched. "
        "Deployment equivalent would slice dropped K-cache plane width.\n\n"
        "Outputs: `plane_truncation_qwen3.csv`, `ppl_planes_qwen3.csv`.\n",
        encoding="utf-8",
    )
    print(f"wrote Task 1 outputs to {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 1 Qwen3 plane-preserving certified truncation + perplexity")
    p.add_argument("--model", default="qwen3", choices=["qwen3"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default="outputs/task1_plane_truncation_qwen3")
    p.add_argument("--calibration-tokens", type=int, default=100_000)
    p.add_argument("--eval-tokens", type=int, default=200_000)
    p.add_argument("--stride", type=int, default=512)
    p.add_argument("--forward-tokens", type=int, default=1024, help="max tokens per forward pass to fit L4 memory")
    p.add_argument("--observed-tokens", type=int, default=10_000)
    p.add_argument("--epsilons", type=float, nargs="+", default=list(EPSILONS))
    p.add_argument("--fractions", type=float, nargs="+", default=list(FRACTIONS))
    p.add_argument("--limit-layers", type=int)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--allow-smoke-under-200k", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
