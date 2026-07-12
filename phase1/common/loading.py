from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


MODEL_IDS = {
    "gpt2": "gpt2",
    "pythia410": "EleutherAI/pythia-410m",
    "pythia1.4": "EleutherAI/pythia-1.4b",
    "openllama7": "openlm-research/open_llama_7b",
    "qwen25": "Qwen/Qwen2.5-0.5B",
    "qwen3": "Qwen/Qwen3-0.6B",
}


@dataclass(frozen=True)
class HeadSpec:
    layer: int
    head: int
    kv_head: int
    A: torch.Tensor
    B: torch.Tensor
    q_bias: torch.Tensor | None = None
    k_bias: torch.Tensor | None = None


@dataclass(frozen=True)
class LoadedModel:
    tag: str
    hf_id: str
    model: torch.nn.Module
    tokenizer: object
    config: object
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_model: int
    d_head: int


def load_model(tag: str, *, device: str | torch.device | None = None) -> LoadedModel:
    if tag not in MODEL_IDS:
        raise ValueError(f"unknown model tag {tag!r}; choose one of {sorted(MODEL_IDS)}")
    hf_id = MODEL_IDS[tag]
    config = AutoConfig.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    if device is not None:
        model.to(torch.device(device))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(hf_id)

    d_model = int(getattr(config, "hidden_size", getattr(config, "n_embd", 0)))
    n_heads = int(getattr(config, "num_attention_heads", getattr(config, "n_head", 0)))
    n_kv_heads = int(getattr(config, "num_key_value_heads", n_heads))
    n_layers = int(getattr(config, "num_hidden_layers", getattr(config, "n_layer", 0)))
    d_head = int(getattr(config, "head_dim", d_model // n_heads))
    return LoadedModel(tag, hf_id, model, tokenizer, config, n_layers, n_heads, n_kv_heads, d_model, d_head)


def _bias_slice(bias: torch.Tensor | None, start: int, stop: int) -> torch.Tensor | None:
    if bias is None:
        return None
    return bias.detach()[start:stop].float()


def iter_heads(lm: LoadedModel, limit_layers: int | None = None, limit_heads: int | None = None) -> Iterable[HeadSpec]:
    tag = lm.tag
    if tag == "gpt2":
        layers = lm.model.transformer.h
        for li, block in enumerate(layers[: limit_layers or len(layers)]):
            W = block.attn.c_attn.weight.detach().T.float()  # [3*d_model, d_model]
            bias = getattr(block.attn.c_attn, "bias", None)
            b = None if bias is None else bias.detach().float()
            Wq, Wk, _ = W.split(lm.d_model, dim=0)
            bq, bk = (None, None) if b is None else b.split(lm.d_model, dim=0)[:2]
            for h in range(min(lm.n_heads, limit_heads or lm.n_heads)):
                s, e = h * lm.d_head, (h + 1) * lm.d_head
                yield HeadSpec(li, h, h, Wq[s:e], Wk[s:e], _bias_slice(bq, s, e), _bias_slice(bk, s, e))
        return

    if tag.startswith("pythia"):
        layers = lm.model.gpt_neox.layers
        for li, layer in enumerate(layers[: limit_layers or len(layers)]):
            qkv = layer.attention.query_key_value
            W = qkv.weight.detach().float().view(lm.n_heads, 3, lm.d_head, lm.d_model)
            b = None if qkv.bias is None else qkv.bias.detach().float().view(lm.n_heads, 3, lm.d_head)
            for h in range(min(lm.n_heads, limit_heads or lm.n_heads)):
                yield HeadSpec(li, h, h, W[h, 0], W[h, 1], None if b is None else b[h, 0], None if b is None else b[h, 1])
        return

    if tag in {"qwen25", "qwen3", "openllama7"}:
        layers = lm.model.model.layers
        group = lm.n_heads // lm.n_kv_heads
        for li, layer in enumerate(layers[: limit_layers or len(layers)]):
            attn = layer.self_attn
            Wq = attn.q_proj.weight.detach().float().view(lm.n_heads, lm.d_head, lm.d_model)
            Wk = attn.k_proj.weight.detach().float().view(lm.n_kv_heads, lm.d_head, lm.d_model)
            bq = None if attn.q_proj.bias is None else attn.q_proj.bias.detach().float().view(lm.n_heads, lm.d_head)
            bk = None if attn.k_proj.bias is None else attn.k_proj.bias.detach().float().view(lm.n_kv_heads, lm.d_head)
            for h in range(min(lm.n_heads, limit_heads or lm.n_heads)):
                kv = h // group
                yield HeadSpec(li, h, kv, Wq[h], Wk[kv], None if bq is None else bq[h], None if bk is None else bk[kv])
        return

    raise NotImplementedError(tag)


def sanity_check(lm: LoadedModel, *, atol: float = 1e-4) -> dict[str, float]:
    """Tiny §6.4 check for currently implemented hook points.

    Returns max absolute q/k reconstruction error. The census uses weights only,
    but this catches transposed or incorrectly interleaved head slicing.
    """
    device = next(lm.model.parameters()).device
    tok = lm.tokenizer("Sanity check for q and k slicing.", return_tensors="pt")
    input_ids = tok["input_ids"][:, :10].to(device)
    with torch.no_grad():
        if lm.tag == "gpt2":
            block = lm.model.transformer.h[0]
            hidden = lm.model.transformer.wte(input_ids) + lm.model.transformer.wpe(torch.arange(input_ids.shape[1], device=device).unsqueeze(0))
            x = block.ln_1(hidden)
            proj = block.attn.c_attn(x)
            q_ref, k_ref, _ = proj.split(lm.d_model, dim=-1)
        elif lm.tag.startswith("pythia"):
            layer = lm.model.gpt_neox.layers[0]
            hidden = lm.model.gpt_neox.embed_in(input_ids)
            x = layer.input_layernorm(hidden)
            proj = layer.attention.query_key_value(x).view(1, input_ids.shape[1], lm.n_heads, 3, lm.d_head)
            q_ref = proj[:, :, :, 0, :].reshape(1, input_ids.shape[1], lm.d_model)
            k_ref = proj[:, :, :, 1, :].reshape(1, input_ids.shape[1], lm.d_model)
        else:
            layer = lm.model.model.layers[0]
            hidden = lm.model.model.embed_tokens(input_ids)
            x = layer.input_layernorm(hidden)
            q_ref = layer.self_attn.q_proj(x)
            k_ref = layer.self_attn.k_proj(x)

        max_q = 0.0
        max_k = 0.0
        for hs in iter_heads(lm, limit_layers=1):
            x0 = x[0]
            q = x0 @ hs.A.T
            k = x0 @ hs.B.T
            if hs.q_bias is not None:
                q = q + hs.q_bias
            if hs.k_bias is not None:
                k = k + hs.k_bias
            q_r = q_ref[0, :, hs.head * lm.d_head : (hs.head + 1) * lm.d_head]
            if lm.n_kv_heads == lm.n_heads:
                k_r = k_ref[0, :, hs.head * lm.d_head : (hs.head + 1) * lm.d_head]
            else:
                k_r = k_ref[0, :, hs.kv_head * lm.d_head : (hs.kv_head + 1) * lm.d_head]
            max_q = max(max_q, float((q - q_r).abs().max().item()))
            max_k = max(max_k, float((k - k_r).abs().max().item()))
        if max_q > atol or max_k > atol:
            raise RuntimeError(f"sanity check failed for {lm.tag}: max_q={max_q:.3g}, max_k={max_k:.3g}, atol={atol}")
        return {"max_q_error": max_q, "max_k_error": max_k}
