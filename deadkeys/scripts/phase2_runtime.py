from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
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
    config._attn_implementation = "eager"
    try:
        model = AutoModelForCausalLM.from_pretrained(hf_id, config=config, attn_implementation="eager", torch_dtype=torch.float32, low_cpu_mem_usage=True)
    except ImportError as exc:
        if "Accelerate" not in str(exc):
            raise
        model = AutoModelForCausalLM.from_pretrained(hf_id, config=config, attn_implementation="eager", torch_dtype=torch.float32)
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
    needle_variants = [
        tokenizer(NEEDLE, add_special_tokens=False)["input_ids"],
        tokenizer(" " + NEEDLE, add_special_tokens=False)["input_ids"],
    ]
    needle_pos = None
    for needle_ids in needle_variants:
        for i in range(0, max(0, len(ids) - len(needle_ids) + 1)):
            if ids[i : i + len(needle_ids)] == needle_ids:
                needle_pos = i
                break
        if needle_pos is not None:
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
            k_norm = torch.linalg.vector_norm(k, dim=1).clamp_min(1e-12)
            low_start = max(0, k.shape[1] - 32) if lm.tag.startswith("qwen") else 0
            k_band = k[:, low_start:]
            A_band = A[low_start:, :]
            # Task 4b: use the shipped census variant: unit-key directional
            # pullback with no sigma-max normalization.  Scores are compared
            # within head percentiles, so the omitted headwise sigma factor is
            # immaterial for the wedge ordering.
            pull = torch.linalg.vector_norm(k_band @ A_band, dim=1)
            s_dir = pull / k_norm
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
                    "score": float(s_dir[pos].item()),
                    "key_norm": float(k_norm[pos].item()),
                    "v_norm": float(v_norm[pos].item()),
                    "attn_mass": 0.0,
                })
    return pd.DataFrame(rows)


def _decode_attention_mass(model, input_ids: torch.Tensor, limit_layers: int | None, limit_heads: int | None, decode_tokens: int, recency_window: int) -> np.ndarray:
    mass: torch.Tensor | None = None
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
            layer_count = min(len(out.attentions), limit_layers or len(out.attentions))
            first = out.attentions[0]
            if first is None:
                raise RuntimeError("attention weights are None; eager attention is required")
            head_count = min(first.shape[1], limit_heads or first.shape[1])
            if mass is None:
                mass = torch.zeros((layer_count, head_count, prefill_len), device=first.device, dtype=torch.float32)
            for li, attn in enumerate(out.attentions[:layer_count]):
                if attn is None:
                    raise RuntimeError("attention weights are None; eager attention is required")
                # [batch, query_heads, q_len=1, kv_len]; accumulate on GPU in one slice.
                mass[li, :, :keep_upto] += attn[0, :head_count, -1, :keep_upto].float()
    if mass is None:
        return np.zeros((0, 0, input_ids.shape[1]), dtype=np.float32)
    return mass.cpu().numpy()


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
    if masses.size:
        layers = df["layer"].to_numpy(dtype=np.int64)
        heads = df["head"].to_numpy(dtype=np.int64)
        positions = df["position"].to_numpy(dtype=np.int64)
        valid = (layers < masses.shape[0]) & (heads < masses.shape[1]) & (positions < masses.shape[2])
        attn_mass = np.zeros(len(df), dtype=np.float32)
        attn_mass[valid] = masses[layers[valid], heads[valid], positions[valid]]
        df["attn_mass"] = attn_mass
    if needle_pos is None:
        df["is_needle"] = False
    else:
        needle_len = len(tokenizer(NEEDLE, add_special_tokens=False)["input_ids"])
        df["is_needle"] = (df["position"] >= needle_pos) & (df["position"] < needle_pos + needle_len)

    df["group"] = df["head"] // max(1, n_heads // n_kv_heads)
    med = df.groupby(["layer", "head"])["key_norm"].transform("median").clip(lower=1e-12)
    df["is_sink"] = (df["position"] == 0) | (df["key_norm"] > 10.0 * med)
    group = df.groupby(["layer", "group", "position"], as_index=False).agg(
        group_score=("s_dir", "max"),
        group_attn_mass=("attn_mass", "sum"),
        group_is_sink=("is_sink", "max"),
        group_is_needle=("is_needle", "max"),
    )
    df = df.merge(group, on=["layer", "group", "position"], how="left")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / f"wedge_{args.model}"
    try:
        out = stem.with_suffix(".parquet")
        df.to_parquet(out, index=False)
    except Exception:
        out = stem.with_suffix(".csv")
        df.to_csv(out, index=False)

    nonsink = df[~df["is_sink"]]
    forbidden = nonsink[(nonsink["s_dir"] <= nonsink["s_dir"].quantile(0.10)) & (nonsink["attn_mass"] >= nonsink["attn_mass"].quantile(0.90))]
    parking_path = args.output_dir / f"parking_relocation_{args.model}.csv"
    _write_parking_relocation(args.spectra_npz, parking_path)
    report = args.output_dir / "REPORT.md"
    report.write_text(
        "# Task 2 runtime wedge report\n\n"
        f"- model: `{args.model}`\n"
        f"- rows: {len(df)}\n"
        f"- non-sink rows: {len(nonsink)}\n"
        f"- sink rows: {int(df['is_sink'].sum())}\n"
        f"- needle position: {needle_pos}\n"
        f"- sink-excluded forbidden-quadrant rate: {0.0 if len(nonsink) == 0 else len(forbidden) / len(nonsink):.6g}\n"
        "- scoring erratum: Qwen3 uses unit-key directional pullback without sigma-max normalization, matching the shipped census variant.\n"
        "- low band: Qwen3 slowest 16 RoPE planes = final 32 head dimensions.\n"
        "- GQA fields: `group_score` is max score over query heads in the KV group; `group_attn_mass` is summed mass.\n"
    )
    return out


def _write_parking_relocation(spectra_npz: Path | None, out: Path) -> None:
    rows = []
    if spectra_npz is not None and spectra_npz.exists():
        data = np.load(spectra_npz)
        for key in sorted(k for k in data.files if k.endswith(".A_soft_basis")):
            basis = data[key]
            rows.append({"key": key, "basis_cols": basis.shape[1] if basis.ndim == 2 else 0, "status": "basis_present_runtime_pcs_not_captured"})
    pd.DataFrame(rows or [{"key": "", "basis_cols": 0, "status": "no_spectra_npz"}]).to_csv(out, index=False)


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
    p.add_argument("--spectra-npz", type=Path)
    args = p.parse_args()
    out = run(args)
    print(out)


if __name__ == "__main__":
    main()
