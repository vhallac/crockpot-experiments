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
            if layer_idx == len(captured):
                captured.append(k.cpu())
            else:
                captured[layer_idx] = k.cpu()

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


def _extract_mentions(lm: Any, docs: list[Document], *, device: torch.device, max_length: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
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
        if lm.tag != "gpt2":
            raise NotImplementedError("first runnable slice currently supports gpt2 k_pre extraction")
        k_by_layer = _capture_gpt2_k(lm, input_ids, attention_mask)
        for row in mention_rows:
            pos = int(row["token_pos"])
            for layer in range(lm.n_layers):
                for head in range(lm.n_heads):
                    out_row = dict(row)
                    out_row["layer"] = layer
                    out_row["head"] = head
                    out_row["vector_idx"] = len(vectors)
                    rows.append(out_row)
                    vectors.append(k_by_layer[layer, pos, head].numpy())
    frame = pd.DataFrame(rows)
    frame.attrs["vectors"] = np.stack(vectors).astype(np.float32) if vectors else np.zeros((0, lm.d_head), dtype=np.float32)
    return frame


def _cosine_matrix(x: np.ndarray) -> np.ndarray:
    centered = x - x.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    centered = centered / np.clip(norms, 1e-12, None)
    return centered @ centered.T


def _auc(positive: list[float], negative: list[float]) -> float:
    if not positive or not negative:
        return float("nan")
    scores = np.asarray(positive + negative, dtype=np.float64)
    labels = np.asarray([1] * len(positive) + [0] * len(negative), dtype=np.int32)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # Average ranks for ties.
    unique_scores, inverse = np.unique(scores, return_inverse=True)
    for idx in range(len(unique_scores)):
        tied = inverse == idx
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    pos_ranks = ranks[labels == 1].sum()
    n_pos = len(positive)
    n_neg = len(negative)
    return float((pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _summarize_auc(frame: pd.DataFrame) -> pd.DataFrame:
    vectors = frame.attrs["vectors"]
    rows: list[dict[str, Any]] = []
    for (layer, head), group in frame.groupby(["layer", "head"]):
        idx = group["vector_idx"].to_numpy(dtype=np.int64)
        sims = _cosine_matrix(vectors[idx])
        meta = group.reset_index(drop=True)
        same_ref: list[float] = []
        diff_ref_same_type: list[float] = []
        diff_ref_position_matched: list[float] = []
        diff_surface_same_ref: list[float] = []
        for _, doc_meta in meta.groupby("doc_id", sort=False):
            doc_idx = doc_meta.index.to_numpy(dtype=np.int64)
            if len(doc_idx) < 2:
                continue
            i_rel, j_rel = np.triu_indices(len(doc_idx), k=1)
            i_abs = doc_idx[i_rel]
            j_abs = doc_idx[j_rel]
            doc_sims = sims[i_abs, j_abs]
            refs = doc_meta["referent_id"].to_numpy()
            types = doc_meta["referent_type"].to_numpy()
            surfaces = doc_meta["surface_form"].to_numpy()
            positions = doc_meta["token_pos"].to_numpy(dtype=np.int64)
            same = refs[i_rel] == refs[j_rel]
            same_type = types[i_rel] == types[j_rel]
            diff_surface = surfaces[i_rel] != surfaces[j_rel]
            position_matched = np.abs(positions[i_rel] - positions[j_rel]) <= 25
            same_ref.extend(doc_sims[same].astype(float).tolist())
            diff_surface_same_ref.extend(doc_sims[same & diff_surface].astype(float).tolist())
            diff_same_type = (~same) & same_type
            diff_ref_same_type.extend(doc_sims[diff_same_type].astype(float).tolist())
            diff_ref_position_matched.extend(doc_sims[diff_same_type & position_matched].astype(float).tolist())
        auc_same_type = _auc(same_ref, diff_ref_same_type)
        auc_position = _auc(same_ref, diff_ref_position_matched)
        rows.append(
            {
                "layer": int(layer),
                "head": int(head),
                "pairs_same_ref": len(same_ref),
                "pairs_same_type_diff_ref": len(diff_ref_same_type),
                "pairs_position_matched_diff_ref": len(diff_ref_position_matched),
                "auc_same_ref_vs_same_type_diff_ref": auc_same_type,
                "auc_same_ref_vs_position_matched_diff_ref": auc_position,
                "auc_diff_surface_same_ref_vs_same_type_diff_ref": _auc(diff_surface_same_ref, diff_ref_same_type),
                "address_head_m1": bool(auc_same_type > 0.9 and auc_position > 0.9),
            }
        )
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
    if lm.tag != "gpt2":
        raise NotImplementedError("the first implemented address-purity slice is gpt2-only")
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
    manifest = {
        "script": "kaddress.scripts.address_purity",
        "spec_slice": "Track A + M1 address purity, k_pre, head-mean-centered cosine",
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
    np.savez_compressed(vectors_path, vectors=mention_frame.attrs["vectors"], rows=mention_frame.to_json(orient="records"))
    print(f"wrote {summary_path}")
    print(f"wrote {manifest_path}")
    print(f"wrote {vectors_path}")
    print(f"address_heads_m1={int(summary['address_head_m1'].sum())}/{len(summary)}")


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
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
