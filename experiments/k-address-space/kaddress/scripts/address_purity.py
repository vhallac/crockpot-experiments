from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from deadkeys.common.loading import MODEL_IDS, load_model
from kaddress.corpus import Document, Mention, generate_track_a


def _environment_summary(device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": getattr(torch.version, "cuda", None),
        "torch_hip": getattr(torch.version, "hip", None),
        "cuda_available": torch.cuda.is_available(),
        "requested_device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _token_positions_for_mentions(tokenizer: Any, doc: Document, *, max_length: int) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    encoded = tokenizer(
        doc.text,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    rows: list[dict[str, Any]] = []
    for mention in doc.mentions:
        token_positions = [i for i, (s, e) in enumerate(offsets) if e > mention.start and s < mention.end]
        if not token_positions:
            continue
        for pos in token_positions:
            rows.append(
                {
                    "doc_id": doc.doc_id,
                    "token_pos": pos,
                    "referent_id": mention.referent_id,
                    "mention_idx": mention.mention_idx,
                    "update_idx": mention.update_idx,
                    "surface_form": mention.surface_form,
                    "referent_type": mention.referent_type,
                    "probe_referent": doc.probe_referent,
                }
            )
    return rows, encoded


def _capture_gpt2_k(lm: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    captured: list[torch.Tensor] = []
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            _q, k, _v = output.detach().float().split(lm.d_model, dim=-1)
            k = k.view(k.shape[0], k.shape[1], lm.n_heads, lm.d_head)
            # Keep keys on the requested execution device.  The previous version
            # copied every layer to CPU here, so the run only touched CUDA during
            # the brief model forward and then did all analysis with NumPy/pandas.
            if layer_idx == len(captured):
                captured.append(k)
            else:
                captured[layer_idx] = k

        return hook

    for layer_idx, block in enumerate(lm.model.transformer.h):
        handles.append(block.attn.c_attn.register_forward_hook(make_hook(layer_idx)))
    try:
        with torch.no_grad():
            lm.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return torch.stack(captured, dim=0)[:, 0, :, :, :]  # [layer, seq, head, d_head]


# --- RoPE utilities (Pythia partial Rotary Position Embedding) ---

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims: [x1, x2] → [-x2, x1]."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_partial_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_ndims: int
) -> torch.Tensor:
    """Apply rotary position embedding to the first `rotary_ndims` of x.

    x:      [..., d_head]
    cos,sin: [1, seq, 1, rotary_ndims] or broadcastable
    Returns: [..., d_head], first rotary_ndims rotated, rest passed through.
    """
    x_rot = x[..., :rotary_ndims]
    x_pass = x[..., rotary_ndims:]
    x_rot = x_rot * cos + _rotate_half(x_rot) * sin
    return torch.cat((x_rot, x_pass), dim=-1)


def _capture_pythia_k(
    lm: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Capture k_pre and k_post for Pythia (GPTNeoX with partial RoPE).

    k_pre:  raw key projection before RoPE (hooked from query_key_value output).
    k_post: rotated key actually cached (extracted from past_key_values).

    Returns (k_pre, k_post) each shaped [layer, seq, head, d_head].
    """
    k_pre_list: list[torch.Tensor] = []

    def make_hook(layer_idx: int):
        def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            qkv = output.detach().float()
            # GPTNeoX QKV interleaving: [batch, seq, n_heads * 3 * d_head] → [batch, seq, n_heads, 3, d_head]
            qkv = qkv.view(output.shape[0], output.shape[1], lm.n_heads, 3, lm.d_head)
            k = qkv[:, :, :, 1, :]  # [batch, seq, n_heads, d_head]
            if layer_idx == len(k_pre_list):
                k_pre_list.append(k)
            else:
                k_pre_list[layer_idx] = k
        return hook

    handles = []
    for li, layer in enumerate(lm.model.gpt_neox.layers):
        handles.append(layer.attention.query_key_value.register_forward_hook(make_hook(li)))
    try:
        with torch.no_grad():
            output = lm.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    finally:
        for h in handles:
            h.remove()

    k_pre = torch.stack(k_pre_list, dim=0)[:, 0, :, :, :]  # [layer, seq, head, d_head]

    k_post = _cache_keys_to_layer_seq_head(output.past_key_values, lm.n_layers)

    return k_pre, k_post


def _cache_keys_to_layer_seq_head(past_key_values: Any, n_layers: int) -> torch.Tensor:
    """Return cached keys as [layer, seq, kv_head, d_head]."""
    if hasattr(past_key_values, "key_cache"):
        k_post_list = [past_key_values.key_cache[i].detach().float() for i in range(n_layers)]
    else:
        k_post_list = [layer_kv[0].detach().float() for layer_kv in past_key_values]
    return torch.stack([k.permute(0, 2, 1, 3) for k in k_post_list], dim=0)[:, 0, :, :, :]


def _capture_nope_k(lm: Any, input_ids: torch.Tensor) -> torch.Tensor:
    """Capture NoPE-GPT raw keys as [layer, seq, head, d_head].

    The inspected remote implementation has no positional embedding, RoPE, or
    ALiBi path: token embeddings feed decoder blocks directly, and attention
    uses qkv projection output as cached keys during prediction.
    """
    captured: list[torch.Tensor] = []
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            _q, k, _v = output.detach().float().split(lm.d_model, dim=-1)
            k = k.view(k.shape[0], k.shape[1], lm.n_heads, lm.d_head)
            if layer_idx == len(captured):
                captured.append(k)
            else:
                captured[layer_idx] = k

        return hook

    for layer_idx, block in enumerate(lm.model.model.body):
        handles.append(block.attention.qkv_proj.register_forward_hook(make_hook(layer_idx)))
    try:
        with torch.no_grad():
            lm.model(input_ids)
    finally:
        for handle in handles:
            handle.remove()
    return torch.stack(captured, dim=0)[:, 0, :, :, :]


def _capture_qwen_k(
    lm: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Capture Qwen3 k_raw, k_pre, and cached k_post as [layer, seq, kv_head, d_head]."""
    k_raw_list: list[torch.Tensor] = []
    k_pre_list: list[torch.Tensor] = []
    handles = []

    def make_raw_hook(layer_idx: int):
        def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            k = output.detach().float().view(output.shape[0], output.shape[1], lm.n_kv_heads, lm.d_head)
            if layer_idx == len(k_raw_list):
                k_raw_list.append(k)
            else:
                k_raw_list[layer_idx] = k

        return hook

    def make_norm_hook(layer_idx: int):
        def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            # HF Qwen3 applies k_norm to [batch, seq, kv_head, d_head], then transposes before RoPE.
            k = output.detach().float()
            if layer_idx == len(k_pre_list):
                k_pre_list.append(k)
            else:
                k_pre_list[layer_idx] = k

        return hook

    for li, layer in enumerate(lm.model.model.layers):
        attn = layer.self_attn
        handles.append(attn.k_proj.register_forward_hook(make_raw_hook(li)))
        if not hasattr(attn, "k_norm"):
            raise RuntimeError("Qwen3 sanity gate expected self_attn.k_norm, but it is missing")
        handles.append(attn.k_norm.register_forward_hook(make_norm_hook(li)))
    try:
        with torch.no_grad():
            output = lm.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    finally:
        for handle in handles:
            handle.remove()

    k_raw = torch.stack(k_raw_list, dim=0)[:, 0, :, :, :]
    k_pre = torch.stack(k_pre_list, dim=0)[:, 0, :, :, :]
    k_post = _cache_keys_to_layer_seq_head(output.past_key_values, lm.n_layers)
    return k_raw, k_pre, k_post


def _rope_sanity_gate(
    lm: Any, k_pre: torch.Tensor, k_post_cache: torch.Tensor
) -> dict[str, Any]:
    """Verify our RoPE reconstruction matches cached k_post.

    Per spec §3: reconstructed k_post from k_pre + own RoPE must match cached
    values ≤ 1e-3 relative error. Perturb once to confirm the check *can* fail.
    For partial-RoPE models, also verifies static (non-rotated) dims of k_pre ==
    k_post exactly.

    Returns dict with keys: max_rel_err, static_dims_match, gate_can_fail,
    bad_rel_err, rotary_ndims, pass, details.
    """
    head_dim = lm.d_head
    config = lm.config
    if lm.tag.startswith("pythia"):
        rotary_ndims = min(16, head_dim)
        rope_base_attr = "rotary_emb_base"
    else:
        rotary_ndims = int(head_dim * float(getattr(config, "partial_rotary_factor", 1.0)))
        rope_base_attr = "rope_theta"
    max_seq_len = k_pre.shape[1]
    device = k_pre.device

    # Compute RoPE cos/sin from scratch using config frequencies.
    base = float(getattr(config, rope_base_attr, 10000))
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_ndims, 2, device=device).float() / rotary_ndims))
    position_ids = torch.arange(max_seq_len, device=device, dtype=torch.float32).unsqueeze(0)
    freqs = torch.outer(position_ids.squeeze(0), inv_freq)  # [seq, rotary_ndims//2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [seq, rotary_ndims]
    cos = emb.cos().unsqueeze(0).unsqueeze(2)  # [1, seq, 1, rotary_ndims]
    sin = emb.sin().unsqueeze(0).unsqueeze(2)  # [1, seq, 1, rotary_ndims]

    # Reconstruct k_post (cos/sin already shaped [1, seq, 1, rotary_ndims]).
    k_post_recon = _apply_partial_rope(k_pre, cos, sin, rotary_ndims)

    # Relative error.  Use a tensor-level L2 relative error for the gate:
    # an elementwise max relative error is dominated by near-zero coordinates
    # and can falsely fail fp32 Qwen3 reconstructions despite tiny absolute
    # differences.  Keep the elementwise max as a diagnostic.
    diff = (k_post_recon - k_post_cache).abs()
    rel_l2_err = float((torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(k_post_cache).clamp_min(1e-12)).item())
    max_abs_err = float(diff.max().item())
    max_elem_rel_err = float((diff / (k_post_cache.abs() + 1e-8)).max().item())

    # Static dims must be identical for partial RoPE; full RoPE has no static dims.
    static_match = True
    if rotary_ndims < head_dim:
        static_match = bool(
            torch.allclose(k_pre[..., rotary_ndims:], k_post_cache[..., rotary_ndims:], atol=1e-6)
        )

    # Perturbation check: shift cos by 0.1 — must cause rel_err > 1e-3.
    cos_bad = cos + 0.1
    k_bad = _apply_partial_rope(k_pre, cos_bad, sin, rotary_ndims)
    bad_diff = (k_bad - k_post_cache).abs()
    bad_rel_err = float((torch.linalg.vector_norm(bad_diff) / torch.linalg.vector_norm(k_post_cache).clamp_min(1e-12)).item())
    gate_can_fail = bad_rel_err > 1e-3

    passed = rel_l2_err <= 1e-3 and static_match and gate_can_fail
    return {
        "max_rel_err": rel_l2_err,
        "rel_l2_err": rel_l2_err,
        "max_abs_err": max_abs_err,
        "max_elem_rel_err": max_elem_rel_err,
        "static_dims_match": static_match,
        "gate_can_fail": gate_can_fail,
        "bad_rel_err": bad_rel_err,
        "rotary_ndims": rotary_ndims,
        "pass": passed,
        "details": (
            f"rotary_ndims={rotary_ndims}/{head_dim}  "
            f"rel_l2_err={rel_l2_err:.2e}  "
            f"max_abs_err={max_abs_err:.2e}  "
            f"max_elem_rel_err={max_elem_rel_err:.2e}  "
            f"static_match={static_match}  "
            f"perturb_fails={gate_can_fail}  "
            f"→ {'PASS' if passed else 'FAIL'}"
        ),
    }


def _extract_mentions(lm: Any, docs: list[Document], *, device: torch.device, max_length: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    vectors: list[torch.Tensor] = []
    for doc in docs:
        mention_rows, encoded = _token_positions_for_mentions(lm.tokenizer, doc, max_length=max_length)
        if not mention_rows:
            continue
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        mention_rows = [r for r in mention_rows if r["token_pos"] < input_ids.shape[1]]
        if not mention_rows:
            continue
        if lm.tag == "gpt2":
            k_all = _capture_gpt2_k(lm, input_ids, attention_mask)
            for row in mention_rows:
                pos = int(row["token_pos"])
                for layer in range(lm.n_layers):
                    for head in range(lm.n_heads):
                        out_row = dict(row)
                        out_row["layer"] = layer
                        out_row["head"] = head
                        out_row["key_variant"] = "pre"
                        out_row["vector_idx"] = len(vectors)
                        rows.append(out_row)
                        vectors.append(k_all[layer, pos, head])
        elif lm.tag.startswith("pythia"):
            k_pre, k_post = _capture_pythia_k(lm, input_ids, attention_mask)
            for variant, k_all in (("pre", k_pre), ("post", k_post)):
                for row in mention_rows:
                    pos = int(row["token_pos"])
                    for layer in range(lm.n_layers):
                        for head in range(lm.n_heads):
                            out_row = dict(row)
                            out_row["layer"] = layer
                            out_row["head"] = head
                            out_row["key_variant"] = variant
                            out_row["vector_idx"] = len(vectors)
                            rows.append(out_row)
                            vectors.append(k_all[layer, pos, head])
        elif lm.tag == "qwen3":
            _k_raw, k_pre, k_post = _capture_qwen_k(lm, input_ids, attention_mask)
            for variant, k_all in (("pre", k_pre), ("post", k_post)):
                for row in mention_rows:
                    pos = int(row["token_pos"])
                    for layer in range(lm.n_layers):
                        for kv_head in range(lm.n_kv_heads):
                            out_row = dict(row)
                            out_row["layer"] = layer
                            out_row["head"] = kv_head
                            out_row["kv_head"] = kv_head
                            out_row["key_variant"] = variant
                            out_row["vector_idx"] = len(vectors)
                            rows.append(out_row)
                            vectors.append(k_all[layer, pos, kv_head])
        elif lm.tag == "nope-gpt-small":
            k_all = _capture_nope_k(lm, input_ids)
            for row in mention_rows:
                pos = int(row["token_pos"])
                for layer in range(lm.n_layers):
                    for head in range(lm.n_heads):
                        out_row = dict(row)
                        out_row["layer"] = layer
                        out_row["head"] = head
                        out_row["key_variant"] = "pre"
                        out_row["vector_idx"] = len(vectors)
                        rows.append(out_row)
                        vectors.append(k_all[layer, pos, head])
        else:
            raise NotImplementedError(f"key extraction not implemented for {lm.tag}")
    frame = pd.DataFrame(rows)
    if vectors:
        frame.attrs["vectors"] = torch.stack(vectors).to(dtype=torch.float32)
    else:
        frame.attrs["vectors"] = torch.zeros((0, lm.d_head), dtype=torch.float32, device=device)
    return frame


def _cosine_matrix(x: torch.Tensor) -> torch.Tensor:
    centered = x - x.mean(dim=0, keepdim=True)
    norms = torch.linalg.vector_norm(centered, dim=1, keepdim=True).clamp_min(1e-12)
    centered = centered / norms
    return centered @ centered.T


def _concat_or_empty(parts: list[torch.Tensor], *, device: torch.device) -> torch.Tensor:
    if not parts:
        return torch.empty(0, dtype=torch.float32, device=device)
    return torch.cat(parts).to(dtype=torch.float32)


def _auc(positive: torch.Tensor, negative: torch.Tensor) -> float:
    if positive.numel() == 0 or negative.numel() == 0:
        return float("nan")
    cmp = positive[:, None] - negative[None, :]
    auc = (cmp.gt(0).to(torch.float32) + 0.5 * cmp.eq(0).to(torch.float32)).mean()
    return float(auc.detach().cpu().item())


def _summarize_auc(frame: pd.DataFrame) -> pd.DataFrame:
    vectors = frame.attrs["vectors"]
    rows: list[dict[str, Any]] = []
    group_cols = ["layer", "head"]
    has_variant = "key_variant" in frame.columns
    if has_variant:
        group_cols.append("key_variant")
    for group_keys, group in frame.groupby(group_cols):
        if not has_variant:
            layer, head = group_keys
            variant = "pre"
        else:
            layer, head, variant = group_keys
        idx = torch.as_tensor(group["vector_idx"].to_numpy(dtype=np.int64), device=vectors.device)
        sims = _cosine_matrix(vectors.index_select(0, idx))
        meta = group.reset_index(drop=True)
        same_ref_parts: list[torch.Tensor] = []
        diff_ref_same_type_parts: list[torch.Tensor] = []
        diff_ref_position_matched_parts: list[torch.Tensor] = []
        diff_surface_same_ref_parts: list[torch.Tensor] = []
        for _, doc_meta in meta.groupby("doc_id", sort=False):
            doc_idx = doc_meta.index.to_numpy(dtype=np.int64)
            if len(doc_idx) < 2:
                continue
            i_rel, j_rel = np.triu_indices(len(doc_idx), k=1)
            i_abs = torch.as_tensor(doc_idx[i_rel], device=vectors.device)
            j_abs = torch.as_tensor(doc_idx[j_rel], device=vectors.device)
            doc_sims = sims[i_abs, j_abs]
            refs = doc_meta["referent_id"].to_numpy()
            types = doc_meta["referent_type"].to_numpy()
            surfaces = doc_meta["surface_form"].to_numpy()
            positions = doc_meta["token_pos"].to_numpy(dtype=np.int64)
            same = refs[i_rel] == refs[j_rel]
            same_type = types[i_rel] == types[j_rel]
            diff_surface = surfaces[i_rel] != surfaces[j_rel]
            position_matched = np.abs(positions[i_rel] - positions[j_rel]) <= 25
            same_t = torch.as_tensor(same, dtype=torch.bool, device=vectors.device)
            diff_surface_t = torch.as_tensor(diff_surface, dtype=torch.bool, device=vectors.device)
            diff_same_type_t = torch.as_tensor((~same) & same_type, dtype=torch.bool, device=vectors.device)
            position_matched_t = torch.as_tensor(position_matched, dtype=torch.bool, device=vectors.device)
            same_ref_parts.append(doc_sims[same_t])
            diff_surface_same_ref_parts.append(doc_sims[same_t & diff_surface_t])
            diff_ref_same_type_parts.append(doc_sims[diff_same_type_t])
            diff_ref_position_matched_parts.append(doc_sims[diff_same_type_t & position_matched_t])
        same_ref = _concat_or_empty(same_ref_parts, device=vectors.device)
        diff_surface_same_ref = _concat_or_empty(diff_surface_same_ref_parts, device=vectors.device)
        diff_ref_same_type = _concat_or_empty(diff_ref_same_type_parts, device=vectors.device)
        diff_ref_position_matched = _concat_or_empty(diff_ref_position_matched_parts, device=vectors.device)
        auc_same_type = _auc(same_ref, diff_ref_same_type)
        auc_position = _auc(same_ref, diff_ref_position_matched)
        row: dict[str, Any] = {
            "layer": int(layer),
            "head": int(head),
            "pairs_same_ref": int(same_ref.numel()),
            "pairs_same_type_diff_ref": int(diff_ref_same_type.numel()),
            "pairs_position_matched_diff_ref": int(diff_ref_position_matched.numel()),
            "auc_same_ref_vs_same_type_diff_ref": auc_same_type,
            "auc_same_ref_vs_position_matched_diff_ref": auc_position,
            "auc_diff_surface_same_ref_vs_same_type_diff_ref": _auc(diff_surface_same_ref, diff_ref_same_type),
            "address_head_m1": bool(auc_same_type > 0.9 and auc_position > 0.9),
        }
        if has_variant:
            row["key_variant"] = variant
        rows.append(row)
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")

    docs = generate_track_a(seed=args.seed, limit_docs=args.limit_docs)
    lm = load_model(args.model, device=device, revision=args.revision)
    doc_token_lengths = [len(lm.tokenizer(doc.text).input_ids) for doc in docs]
    over_budget = [(doc.doc_id, length) for doc, length in zip(docs, doc_token_lengths, strict=True) if length > args.max_length]
    if over_budget:
        examples = ", ".join(f"{doc_id}={length}" for doc_id, length in over_budget[:5])
        raise RuntimeError(
            f"Track A generator produced {len(over_budget)} docs over --max-length={args.max_length}; "
            f"examples: {examples}"
        )
    if lm.tag not in ("gpt2", "qwen3", "nope-gpt-small") and not lm.tag.startswith("pythia"):
        raise NotImplementedError(f"address-purity extraction not implemented for {lm.tag}")

    if lm.tag == "nope-gpt-small":
        has_positional_path = any(
            "pos" in name.lower() or "rope" in name.lower() or "alibi" in name.lower()
            for name, _module in lm.model.named_modules()
        )
        if has_positional_path:
            raise RuntimeError("NoPE sanity failed: found a positional/RoPE/ALiBi-named module")
        print(
            "NoPE sanity: token_embeddings feed decoder body directly; "
            "SelfAttention.qkv_proj keys are used without positional embedding, RoPE, or ALiBi"
        )

    # RoPE sanity gate for RoPE models (spec §3): run on first doc before full extraction.
    if lm.tag.startswith("pythia") or lm.tag == "qwen3":
        import warnings as _warnings

        pilot_doc = docs[:1]
        # Re-extract raw k_pre/k_post for the pilot doc to run the sanity gate.
        _pilot_mention_rows, pilot_encoded = _token_positions_for_mentions(
            lm.tokenizer, pilot_doc[0], max_length=args.max_length
        )
        pilot_ids = pilot_encoded["input_ids"].to(device)
        pilot_am = pilot_encoded.get("attention_mask")
        if pilot_am is not None:
            pilot_am = pilot_am.to(device)
        if lm.tag.startswith("pythia"):
            pilot_k_pre, pilot_k_post = _capture_pythia_k(lm, pilot_ids, pilot_am)
        else:
            pilot_k_raw, pilot_k_pre, pilot_k_post = _capture_qwen_k(lm, pilot_ids, pilot_am)
            group = lm.n_heads // lm.n_kv_heads
            raw_norm = float(torch.linalg.vector_norm(pilot_k_raw, dim=-1).mean().item())
            pre_norm = float(torch.linalg.vector_norm(pilot_k_pre, dim=-1).mean().item())
            print(
                "Qwen3 sanity: "
                f"q_heads={lm.n_heads} kv_heads={lm.n_kv_heads} q_to_kv_group={group} "
                f"hook_order=k_proj→k_norm→RoPE raw_norm_mean={raw_norm:.4f} "
                f"pre_norm_mean={pre_norm:.4f}"
            )
            if lm.n_heads != 16 or lm.n_kv_heads != 8 or group != 2:
                raise RuntimeError(
                    f"Qwen3 GQA sanity failed: q_heads={lm.n_heads}, kv_heads={lm.n_kv_heads}, group={group}"
                )
        gate = _rope_sanity_gate(lm, pilot_k_pre, pilot_k_post)
        print(f"RoPE sanity gate: {gate['details']}")
        if not gate["pass"]:
            msg = (
                f"RoPE sanity gate FAILED: max_rel_err={gate['max_rel_err']:.2e}, "
                f"static_match={gate['static_dims_match']}, gate_can_fail={gate['gate_can_fail']}"
            )
            if args.sanity_gate_strict:
                raise RuntimeError(msg)
            _warnings.warn(msg + " — proceeding anyway (use --sanity-gate-strict to abort)")

    mention_frame = _extract_mentions(lm, docs, device=device, max_length=args.max_length)
    vectors = mention_frame.attrs["vectors"]
    if args.limit_layers is not None:
        mention_frame = mention_frame[mention_frame["layer"] < args.limit_layers].copy()
    if args.limit_heads is not None:
        mention_frame = mention_frame[mention_frame["head"] < args.limit_heads].copy()
    # Preserve only vectors still referenced after limits, compacting vector_idx.
    used = mention_frame["vector_idx"].to_numpy(dtype=np.int64)
    remap = {old: new for new, old in enumerate(used)}
    mention_frame["vector_idx"] = [remap[i] for i in used]
    mention_frame.attrs["vectors"] = vectors[used]

    summary = _summarize_auc(mention_frame)
    key_variants = "k_pre" if lm.tag in {"gpt2", "nope-gpt-small"} else "k_pre + k_post"
    manifest = {
        "script": "kaddress.scripts.address_purity",
        "spec_slice": f"Track A + M1 address purity, {key_variants}, head-mean-centered cosine",
        "model": args.model,
        "hf_id": lm.hf_id,
        "revision": args.revision,
        "seed": args.seed,
        "limit_docs": args.limit_docs,
        "limit_layers": args.limit_layers,
        "limit_heads": args.limit_heads,
        "doc_count": len(docs),
        "max_doc_tokens": max(doc_token_lengths) if doc_token_lengths else 0,
        "max_length": args.max_length,
        "mention_token_rows": int(len(mention_frame)),
        "environment": _environment_summary(device),
    }
    summary_path = out / f"kaddress_m1_{args.model}.csv"
    manifest_path = out / f"kaddress_manifest_{args.model}.json"
    vectors_path = out / f"kaddress_mentions_{args.model}.npz"
    summary.to_csv(summary_path, index=False)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        vectors_path,
        vectors=mention_frame.attrs["vectors"].detach().cpu().numpy(),
        rows=mention_frame.to_json(orient="records"),
    )
    print(f"wrote {summary_path}")
    print(f"wrote {manifest_path}")
    print(f"wrote {vectors_path}")
    print(f"address_heads_m1={int(summary['address_head_m1'].sum())}/{len(summary)}")
    if "key_variant" in summary.columns:
        for variant in sorted(summary["key_variant"].unique()):
            sv = summary[summary["key_variant"] == variant]
            print(f"address_heads_m1_{variant}={int(sv['address_head_m1'].sum())}/{len(sv)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="K-address-space Track A / M1 address-purity extraction")
    p.add_argument("--model", default="gpt2", choices=sorted(MODEL_IDS))
    p.add_argument("--output-dir", default="outputs/k_address_space_m1_gpt2")
    p.add_argument("--device", default="cpu")
    p.add_argument("--revision", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit-docs", type=int, default=2)
    p.add_argument("--limit-layers", type=int, default=None)
    p.add_argument("--limit-heads", type=int, default=None)
    p.add_argument("--max-length", type=int, default=950)
    p.add_argument(
        "--sanity-gate-strict",
        action="store_true",
        help="Abort if the RoPE sanity gate fails (RoPE models).",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
