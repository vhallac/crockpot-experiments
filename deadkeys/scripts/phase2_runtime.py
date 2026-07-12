from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from deadkeys.common.loading import MODEL_IDS, LoadedModel, iter_heads

NEEDLE = "The maintenance code for the auxiliary pump is 7413."
QUESTION = "What is the maintenance code for the auxiliary pump?"


def _device(name: str) -> torch.device:
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")
    return dev


def _load_for_attentions(tag: str, device: torch.device):
    if tag not in MODEL_IDS:
        raise ValueError(f"unknown model tag {tag!r}; choose one of {sorted(MODEL_IDS)}")
    hf_id = MODEL_IDS[tag]
    config = AutoConfig.from_pretrained(hf_id)
    config.attn_implementation = "eager"
    try:
        model = AutoModelForCausalLM.from_pretrained(hf_id, config=config, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    except ImportError as exc:
        if "Accelerate" not in str(exc):
            raise
        model = AutoModelForCausalLM.from_pretrained(hf_id, config=config, torch_dtype=torch.float32)
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, config


def _prompt(tokenizer, prefill_tokens: int) -> tuple[str, int | None]:
    base = (
        "This is a technical maintenance log for a pump station. "
        "The document contains routine inspections, calibration notes, and spare part records. "
    )
    text = base
    while len(tokenizer(text, add_special_tokens=False)["input_ids"]) < max(16, prefill_tokens // 4):
        text += base
    needle_start = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    text += " " + NEEDLE + " "
    while len(tokenizer(text + "\n" + QUESTION, add_special_tokens=False)["input_ids"]) < prefill_tokens:
        text += base
    text += "\n" + QUESTION
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:prefill_tokens]
    needle_ids = tokenizer(NEEDLE, add_special_tokens=False)["input_ids"]
    needle_pos = None
    for i in range(0, max(0, len(ids) - len(needle_ids) + 1)):
        if ids[i : i + len(needle_ids)] == needle_ids:
            needle_pos = i
            break
    return tokenizer.decode(ids), needle_pos


def _capture_prefill(model, tag: str, limit_layers: int | None):
    captured: dict[tuple[int, str], torch.Tensor] = {}
    handles = []

    def save(layer: int, kind: str):
        def hook(_module, _inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            captured[(layer, kind)] = out.detach()
        return hook

    if tag == "gpt2":
        layers = model.transformer.h
        for li, block in enumerate(layers[: limit_layers or len(layers)]):
            handles.append(block.attn.c_attn.register_forward_hook(save(li, "qkv")))
    elif tag.startswith("pythia"):
        layers = model.gpt_neox.layers
        for li, layer in enumerate(layers[: limit_layers or len(layers)]):
            handles.append(layer.attention.query_key_value.register_forward_hook(save(li, "qkv")))
    else:
        layers = model.model.layers
        for li, layer in enumerate(layers[: limit_layers or len(layers)]):
            handles.append(layer.self_attn.k_proj.register_forward_hook(save(li, "k")))
            handles.append(layer.self_attn.v_proj.register_forward_hook(save(li, "v")))
    return captured, handles


def _split_kv(model, tag: str, layer: int, captured: dict[tuple[int, str], torch.Tensor], n_heads: int, n_kv_heads: int, d_model: int, d_head: int):
    if tag == "gpt2":
        qkv = captured[(layer, "qkv")][0].float()
        _q, k, v = qkv.split(d_model, dim=-1)
        return k.view(k.shape[0], n_heads, d_head), v.view(v.shape[0], n_heads, d_head)
    if tag.startswith("pythia"):
        qkv = captured[(layer, "qkv")][0].float().view(-1, n_heads, 3, d_head)
        return qkv[:, :, 1, :], qkv[:, :, 2, :]
    k = captured[(layer, "k")][0].float().view(-1, n_kv_heads, d_head)
    v = captured[(layer, "v")][0].float().view(-1, n_kv_heads, d_head)
    if tag == "qwen3":
        k_norm = model.model.layers[layer].self_attn.k_norm
        k = k_norm(k).float()
    return k, v


def _prefill_tables(lm, model, input_ids: torch.Tensor, limit_layers: int | None, limit_heads: int | None):
    captured, handles = _capture_prefill(model, lm.tag, limit_layers)
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=True)
    finally:
        for h in handles:
            h.remove()

    heads = list(iter_heads(lm, limit_layers=limit_layers, limit_heads=limit_heads))
    rows = []
    by_layer = defaultdict(list)
    for hs in heads:
        by_layer[hs.layer].append(hs)

    for li, layer_heads in by_layer.items():
        k_all, v_all = _split_kv(model, lm.tag, li, captured, lm.n_heads, lm.n_kv_heads, lm.d_model, lm.d_head)
        for hs in layer_heads:
            k = k_all[:, hs.kv_head, :]
            v = v_all[:, hs.kv_head, :]
            A = hs.A.to(k.device, dtype=torch.float32)
            pull = torch.linalg.vector_norm(k @ A, dim=1)
            k_norm = torch.linalg.vector_norm(k, dim=1).clamp_min(1e-12)
            s_dir = pull / (torch.linalg.svdvals(A)[0].to(k.device).clamp_min(1e-12) * k_norm)
            v_norm = torch.linalg.vector_norm(v, dim=1)
            for pos in range(k.shape[0]):
                rows.append({
                    "layer": hs.layer,
                    "head": hs.head,
                    "kv_head": hs.kv_head,
                    "position": pos,
                    "s_low": float(pull[pos].item()),
                    "s_dir": float(s_dir[pos].item()),
                    "s_sigma": float("nan"),
                    "v_norm": float(v_norm[pos].item()),
                    "attn_mass": 0.0,
                })
    return pd.DataFrame(rows)


def _decode_attention_mass(model, input_ids: torch.Tensor, limit_layers: int | None, limit_heads: int | None, decode_tokens: int, recency_window: int):
    masses = defaultdict(float)
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True)
        past = out.past_key_values
        cur = input_ids[:, -1:]
        prefill_len = input_ids.shape[1]
        for step in range(decode_tokens):
            out = model(input_ids=cur, past_key_values=past, use_cache=True, output_attentions=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]
            cur = torch.argmax(logits, dim=-1, keepdim=True)
            q_abs = prefill_len + step
            keep_upto = min(prefill_len, max(0, q_abs - recency_window))
            if keep_upto == 0:
                continue
            for li, attn in enumerate(out.attentions[: limit_layers or len(out.attentions)]):
                if attn is None:
                    raise RuntimeError("attention weights are None; eager attention is required")
                # [batch, query_heads, q_len=1, kv_len]
                layer_attn = attn[0, :, -1, :keep_upto].float()
                n_heads = layer_attn.shape[0]
                max_heads = min(n_heads, limit_heads or n_heads)
                for h in range(max_heads):
                    vals = layer_attn[h]
                    for pos, val in enumerate(vals):
                        masses[(li, h, pos)] += float(val.item())
    return masses


def run(args: argparse.Namespace) -> Path:
    dev = _device(args.device)
    model, tokenizer, config = _load_for_attentions(args.model, dev)
    d_model = int(getattr(config, "hidden_size", getattr(config, "n_embd", 0)))
    n_heads = int(getattr(config, "num_attention_heads", getattr(config, "n_head", 0)))
    n_kv_heads = int(getattr(config, "num_key_value_heads", n_heads))
    n_layers = int(getattr(config, "num_hidden_layers", getattr(config, "n_layer", 0)))
    d_head = int(getattr(config, "head_dim", d_model // n_heads))
    lm = LoadedModel(args.model, MODEL_IDS[args.model], model, tokenizer, config, n_layers, n_heads, n_kv_heads, d_model, d_head)

    text, needle_pos = _prompt(tokenizer, args.prefill_tokens)
    toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.prefill_tokens)
    input_ids = toks["input_ids"].to(dev)

    df = _prefill_tables(lm, model, input_ids, args.limit_layers, args.limit_heads)
    masses = _decode_attention_mass(model, input_ids, args.limit_layers, args.limit_heads, args.decode_tokens, args.recency_window)
    if masses:
        idx = {(int(r["layer"]), int(r["head"]), int(r["position"])): i for i, r in df.iterrows()}
        for key, mass in masses.items():
            if key in idx:
                df.at[idx[key], "attn_mass"] = mass
    if needle_pos is None:
        df["is_needle"] = False
    else:
        needle_len = len(tokenizer(NEEDLE, add_special_tokens=False)["input_ids"])
        df["is_needle"] = (df["position"] >= needle_pos) & (df["position"] < needle_pos + needle_len)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / f"wedge_{args.model}"
    try:
        out = stem.with_suffix(".parquet")
        df.to_parquet(out, index=False)
    except Exception:
        out = stem.with_suffix(".csv")
        df.to_csv(out, index=False)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 2 runtime wedge collection, GPU-first and streaming final tables to CPU.")
    p.add_argument("--model", choices=sorted(MODEL_IDS), required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit-layers", type=int)
    p.add_argument("--limit-heads", type=int)
    p.add_argument("--prefill-tokens", type=int, default=4096)
    p.add_argument("--decode-tokens", type=int, default=256)
    p.add_argument("--recency-window", type=int, default=256)
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = p.parse_args()
    out = run(args)
    print(out)


if __name__ == "__main__":
    main()
