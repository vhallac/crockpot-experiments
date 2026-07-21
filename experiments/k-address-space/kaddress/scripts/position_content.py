from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from deadkeys.common.loading import MODEL_IDS, load_model
from kaddress.scripts.address_purity import (
    _capture_gpt2_k,
    _capture_nope_k,
    _capture_pythia_k,
    _capture_qwen_k,
    _environment_summary,
)


@dataclass(frozen=True)
class Stimulus:
    stimulus_id: str
    family: str
    segment_id: str
    input_ids: list[int]
    # slot index -> positions, one position per repetition/occurrence
    slots: dict[int, list[int]]
    slot_token_ids: dict[int, int]
    repetitions: int
    segment_token_len: int | None
    text_preview: str


def _encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _decode_piece(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([token_id], clean_up_tokenization_spaces=False).strip()


def _family_a_segments() -> list[str]:
    return [
        "Alice is a successful engineer.",
        "Boris found the quiet library.",
        "Clara keeps a small red notebook.",
        "Derek moved the marble today.",
        "Elena wrote a careful status update.",
        "Farid placed the key inside.",
        "Greta solved the puzzle quickly.",
        "Hector opened a blue cabinet.",
    ]


def _family_b_words() -> tuple[list[str], list[str], list[str], list[str]]:
    return (
        ["Alice", "Boris", "Clara", "Derek", "Elena", "Farid", "Greta", "Hector", "Iris", "Jonas"],
        ["calm", "brisk", "careful", "eager", "formal", "gentle", "honest", "jolly", "kind", "lucid"],
        ["engineer", "artist", "doctor", "lawyer", "teacher", "baker", "pilot", "writer", "nurse", "clerk"],
        [".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
    )


def _family_c_text() -> str:
    return (
        "The city kept the old market near the river, and the market kept the habit of opening before dawn. "
        "The carts rolled over the stones, the bakers raised the shutters, and the first train crossed the bridge. "
        "In the square, the clerk counted the crates while the driver checked the list and the guard watched the gate. "
        "The morning was ordinary, but the ordinary rhythm made the place easy to remember. "
    )


def _repeat_to_limit(piece: list[int], *, min_repetitions: int, max_length: int, requested: int | None) -> int:
    if not piece:
        return 0
    cap = max_length // len(piece)
    if requested is not None:
        cap = min(cap, requested)
    return cap if cap >= min_repetitions else cap


def build_stimuli(
    tokenizer: Any,
    *,
    families: set[str],
    max_length: int,
    min_repetitions: int,
    repetitions: int | None,
    limit_stimuli: int | None,
) -> list[Stimulus]:
    stimuli: list[Stimulus] = []

    if "A" in families:
        for seg_no, text in enumerate(_family_a_segments()):
            ids = _encode(tokenizer, text)
            r = _repeat_to_limit(ids, min_repetitions=min_repetitions, max_length=max_length, requested=repetitions)
            if r < min_repetitions:
                continue
            input_ids = ids * r
            slots = {slot: [rep * len(ids) + slot for rep in range(r)] for slot in range(len(ids))}
            token_ids = {slot: ids[slot] for slot in range(len(ids))}
            stimuli.append(Stimulus(f"A{seg_no:02d}", "A", f"seg{seg_no:02d}", input_ids, slots, token_ids, r, len(ids), text))

    if "B" in families:
        names, adjs, jobs, punct = _family_b_words()
        # Build one stimulus per cyclic offset.  Probe token-identical frame slots only.
        for offset in range(8):
            reps: list[list[int]] = []
            for i in range(max(min_repetitions, repetitions or min_repetitions)):
                j = (i + offset) % len(names)
                reps.append(_encode(tokenizer, f"{names[j]} is a {adjs[j]} {jobs[j]}{punct[j]}"))
            common_len = len(reps[0]) if reps else 0
            if common_len == 0 or any(len(x) != common_len for x in reps):
                continue
            r_cap = max_length // common_len
            r = min(len(reps), r_cap)
            if r < min_repetitions:
                continue
            reps = reps[:r]
            probe_slots: dict[int, list[int]] = {}
            token_ids: dict[int, int] = {}
            for slot in range(common_len):
                ids_at_slot = [rep[slot] for rep in reps]
                if len(set(ids_at_slot)) != 1:
                    continue
                piece = _decode_piece(tokenizer, ids_at_slot[0])
                if piece in {"is", "a", "."}:
                    probe_slots[slot] = [rep_i * common_len + slot for rep_i in range(r)]
                    token_ids[slot] = ids_at_slot[0]
            if not probe_slots:
                continue
            flat = [tok for rep in reps for tok in rep]
            stimuli.append(Stimulus(f"B{offset:02d}", "B", f"frame{offset:02d}", flat, probe_slots, token_ids, r, common_len, "{Name} is a {adj} {profession}."))

    if "C" in families:
        base_ids = _encode(tokenizer, _family_c_text())
        r = _repeat_to_limit(base_ids, min_repetitions=1, max_length=max_length, requested=None)
        # Natural-recurrence control: one long repeated paragraph, probe frequent pieces.
        input_ids = base_ids * max(1, r)
        token_to_positions: dict[int, list[int]] = {}
        for pos, tok in enumerate(input_ids):
            piece = _decode_piece(tokenizer, tok)
            if piece in {"the", ",", "."}:
                token_to_positions.setdefault(tok, []).append(pos)
        slot = 0
        probe_slots: dict[int, list[int]] = {}
        token_ids: dict[int, int] = {}
        for tok, positions in sorted(token_to_positions.items(), key=lambda kv: -len(kv[1])):
            if len(positions) >= min_repetitions:
                probe_slots[slot] = positions[: max_length]
                token_ids[slot] = tok
                slot += 1
        if probe_slots:
            stimuli.append(Stimulus("C00", "C", "natural-recurrence", input_ids, probe_slots, token_ids, min(len(v) for v in probe_slots.values()), None, _family_c_text()[:120]))

    if limit_stimuli is not None:
        stimuli = stimuli[:limit_stimuli]
    return stimuli


def _capture_keys(lm: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> list[tuple[str, torch.Tensor]]:
    if lm.tag == "gpt2":
        return [("pre", _capture_gpt2_k(lm, input_ids, attention_mask))]
    if lm.tag.startswith("pythia"):
        pre, post = _capture_pythia_k(lm, input_ids, attention_mask)
        return [("pre", pre), ("post", post)]
    if lm.tag == "qwen3":
        _raw, pre, post = _capture_qwen_k(lm, input_ids, attention_mask)
        return [("pre", pre), ("post", post)]
    if lm.tag == "nope-gpt-small":
        return [("pre", _capture_nope_k(lm, input_ids))]
    raise NotImplementedError(f"M1.5 extraction not implemented for {lm.tag}")


def _ridge_cv_r2(x: np.ndarray, y: np.ndarray, *, folds: int, seed: int, alphas: np.ndarray) -> float:
    n = len(y)
    if n < folds + 2 or float(np.var(y)) <= 0:
        return 0.0
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    best = -np.inf
    for alpha in alphas:
        scores = []
        for fold in range(folds):
            val_mask = np.zeros(n, dtype=bool)
            val_mask[order[fold::folds]] = True
            tr = ~val_mask
            x_tr = x[tr]
            y_tr = y[tr]
            x_val = x[val_mask]
            y_val = y[val_mask]
            xm = x_tr.mean(axis=0, keepdims=True)
            ym = y_tr.mean()
            xc = x_tr - xm
            yc = y_tr - ym
            xtx = xc.T @ xc
            beta = np.linalg.solve(xtx + alpha * np.eye(xtx.shape[0]), xc.T @ yc)
            pred = (x_val - xm) @ beta + ym
            denom = np.sum((y_val - y_val.mean()) ** 2)
            scores.append(0.0 if denom <= 0 else 1.0 - float(np.sum((y_val - pred) ** 2) / denom))
        best = max(best, float(np.mean(scores)))
    return best


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    # Average ties.
    values = a[order]
    start = 0
    while start < len(a):
        end = start + 1
        while end < len(a) and values[end] == values[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    ac = a - a.mean()
    bc = b - b.mean()
    denom = math.sqrt(float((ac @ ac) * (bc @ bc)))
    return float("nan") if denom == 0 else float((ac @ bc) / denom)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return _corr(_rankdata(a), _rankdata(b))


def _pca_stats(resid: np.ndarray) -> tuple[int, float, np.ndarray, np.ndarray]:
    if resid.shape[0] < 2:
        d = resid.shape[1]
        return 0, 0.0, np.zeros((0, d), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    _u, s, vt = np.linalg.svd(resid, full_matrices=False)
    var = s**2
    total = float(var.sum())
    if total <= 0:
        return 0, 0.0, vt[:0].astype(np.float32), var.astype(np.float32)
    cum = np.cumsum(var) / total
    k90 = int(np.searchsorted(cum, 0.90) + 1)
    return k90, float(var[:k90].sum() / total), vt[:k90].astype(np.float32), var.astype(np.float32)


def _nearest_centroid_cv_accuracy(x: np.ndarray, labels: np.ndarray, *, folds: int, seed: int) -> float:
    if len(np.unique(labels)) < 2 or len(labels) < folds + 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(labels))
    accs = []
    for fold in range(folds):
        val_mask = np.zeros(len(labels), dtype=bool)
        val_mask[order[fold::folds]] = True
        tr = ~val_mask
        centroids = {lab: x[tr & (labels == lab)].mean(axis=0) for lab in np.unique(labels[tr])}
        if not centroids:
            continue
        labs = np.array(list(centroids.keys()))
        c = np.stack([centroids[lab] for lab in labs])
        d2 = ((x[val_mask, None, :] - c[None, :, :]) ** 2).sum(axis=2)
        pred = labs[np.argmin(d2, axis=1)]
        accs.append(float(np.mean(pred == labels[val_mask])))
    return float(np.mean(accs)) if accs else float("nan")


def _analyse_matrix(x: np.ndarray, y: np.ndarray, token_ids: np.ndarray, *, seed: int, variance_floor: float) -> dict[str, Any]:
    raw = float(np.mean(np.abs(x)))
    mean = x.mean(axis=0, keepdims=True)
    resid = x - mean
    resid_scale = float(np.mean(np.abs(resid)))
    frac = 0.0 if raw <= 0 else resid_scale / raw
    if frac < variance_floor:
        r2 = 0.0
        shuffled_r2 = 0.0
    else:
        alphas = np.logspace(-2, 4, 13)
        r2 = _ridge_cv_r2(x, y.astype(float), folds=5, seed=seed, alphas=alphas)
        shuffled = np.random.default_rng(seed + 17).permutation(y.astype(float))
        shuffled_r2 = _ridge_cv_r2(x, shuffled, folds=5, seed=seed + 1, alphas=alphas)
    k90, frac_resid_var, basis, _var = _pca_stats(resid)
    pc1 = resid @ basis[0] if len(basis) else np.zeros(len(y), dtype=float)
    dominant_bin = -1
    if len(pc1) >= 4:
        amp = np.abs(np.fft.rfft(pc1 - pc1.mean()))
        if len(amp) > 1:
            dominant_bin = int(np.argmax(amp[1:]) + 1)
    if len(basis):
        proj = resid - (resid @ basis.T) @ basis
        x_proj = mean + proj
    else:
        x_proj = x
    r2_after = 0.0 if frac < variance_floor else _ridge_cv_r2(x_proj, y.astype(float), folds=5, seed=seed + 2, alphas=np.logspace(-2, 4, 13))
    token_acc_before = _nearest_centroid_cv_accuracy(x, token_ids, folds=5, seed=seed + 3)
    token_acc_after = _nearest_centroid_cv_accuracy(x_proj, token_ids, folds=5, seed=seed + 4)
    return {
        "raw_mean_abs": raw,
        "resid_mean_abs": resid_scale,
        "position_fraction": frac,
        "ridge_r2": r2,
        "shuffled_ridge_r2": shuffled_r2,
        "pca_components_90pct": k90,
        "pca_residual_variance_fraction_90pct": frac_resid_var,
        "pc1_spearman_repetition": _spearman(pc1, y.astype(float)) if len(basis) else float("nan"),
        "pc1_dominant_fourier_bin": dominant_bin,
        "r2_after_position_pc_projection": r2_after,
        "token_identity_acc_before": token_acc_before,
        "token_identity_acc_after": token_acc_after,
        "basis": basis,
    }


@dataclass
class AggregateState:
    family: str
    variant: str
    layer: int
    head: int
    d_head: int
    raw_abs_sum: float = 0.0
    resid_abs_sum: float = 0.0
    value_count: int = 0
    cov: np.ndarray | None = None
    sample_x: list[np.ndarray] | None = None
    sample_y: list[float] | None = None
    sample_token: list[int] | None = None
    seen: int = 0

    def add(self, x: np.ndarray, y: np.ndarray, token_ids: np.ndarray, *, sample_cap: int, rng: np.random.Generator) -> None:
        resid = x - x.mean(axis=0, keepdims=True)
        if self.cov is None:
            self.cov = np.zeros((self.d_head, self.d_head), dtype=np.float64)
            self.sample_x = []
            self.sample_y = []
            self.sample_token = []
        self.raw_abs_sum += float(np.abs(x).sum())
        self.resid_abs_sum += float(np.abs(resid).sum())
        self.value_count += int(x.size)
        self.cov += resid.T @ resid
        for row, yy, tok in zip(x, y, token_ids, strict=True):
            self.seen += 1
            assert self.sample_x is not None and self.sample_y is not None and self.sample_token is not None
            if len(self.sample_x) < sample_cap:
                self.sample_x.append(row.astype(np.float32, copy=True))
                self.sample_y.append(float(yy))
                self.sample_token.append(int(tok))
            else:
                j = int(rng.integers(0, self.seen))
                if j < sample_cap:
                    self.sample_x[j] = row.astype(np.float32, copy=True)
                    self.sample_y[j] = float(yy)
                    self.sample_token[j] = int(tok)


def _aggregate_stats(state: AggregateState, *, seed: int, variance_floor: float) -> tuple[dict[str, Any], np.ndarray]:
    assert state.cov is not None and state.sample_x is not None and state.sample_y is not None and state.sample_token is not None
    raw = state.raw_abs_sum / max(1, state.value_count)
    resid_scale = state.resid_abs_sum / max(1, state.value_count)
    frac = 0.0 if raw <= 0 else resid_scale / raw
    eigvals, eigvecs = np.linalg.eigh(state.cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order].T.astype(np.float32)
    total = float(np.maximum(eigvals, 0).sum())
    if total <= 0:
        k90 = 0
        frac_var = 0.0
        basis = eigvecs[:0]
    else:
        cum = np.cumsum(np.maximum(eigvals, 0)) / total
        k90 = int(np.searchsorted(cum, 0.90) + 1)
        frac_var = float(np.maximum(eigvals[:k90], 0).sum() / total)
        basis = eigvecs[:k90]
    x = np.stack(state.sample_x)
    y = np.asarray(state.sample_y, dtype=float)
    token_ids = np.asarray(state.sample_token, dtype=np.int64)
    if frac < variance_floor:
        r2 = 0.0
        shuffled_r2 = 0.0
        r2_after = 0.0
    else:
        alphas = np.logspace(-2, 4, 13)
        r2 = _ridge_cv_r2(x, y, folds=5, seed=seed, alphas=alphas)
        shuffled_r2 = _ridge_cv_r2(x, np.random.default_rng(seed + 17).permutation(y), folds=5, seed=seed + 1, alphas=alphas)
        if len(basis):
            mean = x.mean(axis=0, keepdims=True)
            resid = x - mean
            x_proj = mean + resid - (resid @ basis.T) @ basis
        else:
            x_proj = x
        r2_after = _ridge_cv_r2(x_proj, y, folds=5, seed=seed + 2, alphas=alphas)
    token_acc_before = _nearest_centroid_cv_accuracy(x, token_ids, folds=5, seed=seed + 3)
    if len(basis):
        mean = x.mean(axis=0, keepdims=True)
        resid = x - mean
        x_proj = mean + resid - (resid @ basis.T) @ basis
    else:
        x_proj = x
    token_acc_after = _nearest_centroid_cv_accuracy(x_proj, token_ids, folds=5, seed=seed + 4)
    return {
        "raw_mean_abs": raw,
        "resid_mean_abs": resid_scale,
        "position_fraction": frac,
        "ridge_r2": r2,
        "shuffled_ridge_r2": shuffled_r2,
        "pca_components_90pct": k90,
        "pca_residual_variance_fraction_90pct": frac_var,
        "pc1_spearman_repetition": float("nan"),
        "pc1_dominant_fourier_bin": -1,
        "r2_after_position_pc_projection": r2_after,
        "token_identity_acc_before": token_acc_before,
        "token_identity_acc_after": token_acc_after,
        "sample_rows": len(x),
        "source_rows": state.seen,
    }, basis


def run(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")

    lm = load_model(args.model, device=device, revision=args.revision)
    if lm.tag not in {"gpt2", "qwen3", "nope-gpt-small"} and not lm.tag.startswith("pythia"):
        raise NotImplementedError(f"M1.5 extraction not implemented for {lm.tag}")

    families = {f.strip().upper() for f in args.families.split(",") if f.strip()}
    stimuli = build_stimuli(
        lm.tokenizer,
        families=families,
        max_length=args.max_length,
        min_repetitions=args.min_repetitions,
        repetitions=args.repetitions,
        limit_stimuli=args.limit_stimuli,
    )
    if not stimuli:
        raise RuntimeError("no valid M1.5 stimuli were generated for this tokenizer/settings")

    rows: list[dict[str, Any]] = []
    projector_basis: dict[str, np.ndarray] = {}
    gate_rows: list[dict[str, Any]] = []
    aggregates: dict[tuple[str, str, int, int], AggregateState] = {}
    sample_rng = np.random.default_rng(args.seed + 12345)

    for stim_i, stim in enumerate(stimuli):
        input_ids = torch.tensor([stim.input_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            variants = _capture_keys(lm, input_ids, attention_mask)
        for variant, k_all in variants:
            n_layers, _seq, n_heads, d_head = k_all.shape
            if args.limit_layers is not None:
                n_layers = min(n_layers, args.limit_layers)
            if args.limit_heads is not None:
                n_heads = min(n_heads, args.limit_heads)
            for layer in range(n_layers):
                for head in range(n_heads):
                    for slot, positions in stim.slots.items():
                        if len(positions) < max(10, args.min_repetitions if stim.family != "C" else 10):
                            continue
                        pos_t = torch.as_tensor(positions, dtype=torch.long, device=k_all.device)
                        x = k_all[layer].index_select(0, pos_t)[:, head, :].detach().float().cpu().numpy()
                        y = np.arange(len(positions), dtype=float)
                        token_ids = np.full(len(positions), stim.slot_token_ids[slot], dtype=np.int64)
                        stats = _analyse_matrix(x, y, token_ids, seed=args.seed + stim_i + layer * 1000 + head, variance_floor=args.variance_floor)
                        basis = stats.pop("basis")
                        row = {
                            "model": args.model,
                            "hf_id": lm.hf_id,
                            "family": stim.family,
                            "stimulus_id": stim.stimulus_id,
                            "segment_id": stim.segment_id,
                            "repetitions": len(positions),
                            "segment_token_len": stim.segment_token_len,
                            "slot_index": int(slot),
                            "slot_token_id": int(stim.slot_token_ids[slot]),
                            "slot_token_text": _decode_piece(lm.tokenizer, stim.slot_token_ids[slot]),
                            "layer": layer,
                            "head": head,
                            "key_variant": variant,
                            "d_head": d_head,
                            **{k: v for k, v in stats.items() if k != "basis"},
                        }
                        rows.append(row)
                        if stim.family == "A":
                            key = f"{variant}/layer{layer:02d}/head{head:02d}/stim{stim.stimulus_id}/slot{slot:02d}"
                            projector_basis[key] = basis
                        agg_key = (stim.family, variant, layer, head)
                        if agg_key not in aggregates:
                            aggregates[agg_key] = AggregateState(stim.family, variant, layer, head, d_head)
                        aggregates[agg_key].add(x, y, token_ids, sample_cap=args.aggregate_sample_cap, rng=sample_rng)
                        is_arch_zero_case = (
                            layer == 0
                            and variant == "pre"
                            and (lm.tag in {"nope-gpt-small", "qwen3"} or lm.tag.startswith("pythia"))
                        )
                        if is_arch_zero_case:
                            raw = row["raw_mean_abs"]
                            rel = row["position_fraction"]
                            perturbed = x.copy()
                            perturbed += (np.arange(len(positions))[:, None] / max(1, len(positions) - 1)) * raw * 1e-3
                            p_raw = float(np.mean(np.abs(perturbed)))
                            p_rel = float(np.mean(np.abs(perturbed - perturbed.mean(axis=0, keepdims=True))) / max(p_raw, 1e-12))
                            gate_rows.append(
                                {
                                    "gate": "G1_architectural_zero",
                                    "stimulus_id": stim.stimulus_id,
                                    "family": stim.family,
                                    "layer": layer,
                                    "head": head,
                                    "key_variant": variant,
                                    "slot_index": int(slot),
                                    "position_fraction": rel,
                                    "threshold": args.variance_floor,
                                    "pass": bool(rel < args.variance_floor),
                                    "perturbed_position_fraction": p_rel,
                                    "perturbation_can_fail": bool(p_rel >= args.variance_floor),
                                }
                            )
        print(f"processed {stim.stimulus_id} family={stim.family} seq={len(stim.input_ids)} slots={len(stim.slots)}")

    for agg_i, state in enumerate(aggregates.values()):
        stats, basis = _aggregate_stats(state, seed=args.seed + 900_000 + agg_i, variance_floor=args.variance_floor)
        rows.append(
            {
                "model": args.model,
                "hf_id": lm.hf_id,
                "family": state.family,
                "stimulus_id": "AGGREGATE",
                "segment_id": "all",
                "repetitions": stats.pop("source_rows"),
                "segment_token_len": None,
                "slot_index": -1,
                "slot_token_id": -1,
                "slot_token_text": "<aggregate>",
                "layer": state.layer,
                "head": state.head,
                "key_variant": state.variant,
                "d_head": state.d_head,
                **stats,
            }
        )
        projector_basis[f"{state.variant}/layer{state.layer:02d}/head{state.head:02d}/aggregate_family{state.family}"] = basis

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise RuntimeError("analysis produced no rows")
    gates = pd.DataFrame(gate_rows)
    gate_pass = bool(gates.empty or (gates["pass"].all() and gates["perturbation_can_fail"].all()))
    shuffle_ok = bool((summary["shuffled_ridge_r2"].abs() < args.shuffle_r2_abs_warn).mean() >= 0.95)

    summary_path = out / f"kaddress_m15_{args.model}.csv"
    gates_path = out / f"kaddress_m15_gates_{args.model}.csv"
    manifest_path = out / f"kaddress_m15_manifest_{args.model}.json"
    projectors_path = out / f"kaddress_m15_projectors_{args.model}.npz"
    summary.to_csv(summary_path, index=False)
    gates.to_csv(gates_path, index=False)
    np.savez_compressed(projectors_path, **projector_basis)
    manifest = {
        "script": "kaddress.scripts.position_content",
        "spec_slice": "ADDENDUM §5-M1.5 repeated-segment position-content probe",
        "model": args.model,
        "hf_id": lm.hf_id,
        "revision": args.revision,
        "seed": args.seed,
        "families": sorted(families),
        "stimulus_count": len(stimuli),
        "max_length": args.max_length,
        "min_repetitions": args.min_repetitions,
        "requested_repetitions": args.repetitions,
        "limit_layers": args.limit_layers,
        "limit_heads": args.limit_heads,
        "summary_rows": int(len(summary)),
        "gate_g1_pass": gate_pass,
        "shuffle_null_ok_95pct_abs_threshold": args.shuffle_r2_abs_warn,
        "shuffle_null_ok": shuffle_ok,
        "projector_file": projectors_path.name,
        "environment": _environment_summary(device),
        "stimuli": [
            {
                "stimulus_id": s.stimulus_id,
                "family": s.family,
                "segment_id": s.segment_id,
                "tokens": len(s.input_ids),
                "slots": len(s.slots),
                "repetitions": s.repetitions,
                "segment_token_len": s.segment_token_len,
                "preview": s.text_preview,
            }
            for s in stimuli
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {summary_path}")
    print(f"wrote {gates_path}")
    print(f"wrote {manifest_path}")
    print(f"wrote {projectors_path}")
    print(f"gate_g1_pass={gate_pass} shuffle_null_ok={shuffle_ok}")
    print(
        summary.groupby(["family", "key_variant"])[["position_fraction", "ridge_r2", "pca_components_90pct"]]
        .mean(numeric_only=True)
        .to_string()
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M1.5 repeated-segment position-content probe for K vectors")
    p.add_argument("--model", default="nope-gpt-small", choices=sorted(MODEL_IDS))
    p.add_argument("--output-dir", default="outputs/k_address_space_m15_nope_gpt_small")
    p.add_argument("--device", default="cpu")
    p.add_argument("--revision", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--families", default="A,B,C", help="Comma-separated subset of A,B,C")
    p.add_argument("--max-length", type=int, default=950)
    p.add_argument("--min-repetitions", type=int, default=120)
    p.add_argument("--repetitions", type=int, default=None)
    p.add_argument("--limit-stimuli", type=int, default=None)
    p.add_argument("--limit-layers", type=int, default=None)
    p.add_argument("--limit-heads", type=int, default=None)
    p.add_argument("--variance-floor", type=float, default=1e-5)
    p.add_argument("--shuffle-r2-abs-warn", type=float, default=0.05)
    p.add_argument(
        "--aggregate-sample-cap",
        type=int,
        default=2048,
        help="Reservoir sample size per family/variant/layer/head for aggregate M1.5.6 token-identity fidelity.",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
