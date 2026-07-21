from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from collections import Counter
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
    target_L: int | None = None
    content_words: list[dict[str, str]] | None = None


@dataclass(frozen=True)
class BuildResult:
    stimuli: list[Stimulus]
    rejected_stimuli: list[dict[str, Any]]
    feasibility: list[dict[str, Any]]


def _encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _decode_piece(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([token_id], clean_up_tokenization_spaces=False).strip()


FRAME_TOKENS = {"is", "a", "."}
REJECTION_REASONS = {
    "below_min_repetitions",
    "exceeds_max_length",
    "slot_token_not_constant",
    "frame_token_absent",
    "insufficient_occurrences",
}


def _family_a_candidate_texts() -> list[str]:
    names = ["Alice", "Boris", "Clara", "Derek", "Elena", "Farid", "Greta", "Hector", "Iris", "Jonas", "Kara", "Liam"]
    verbs = ["waved", "smiled", "waited", "nodded", "listened", "arrived", "agreed", "replied", "looked", "paused", "rested", "worked"]
    adjectives = ["calm", "brave", "quiet", "quick", "steady", "kind", "bright", "careful", "formal", "gentle", "honest", "lucid"]
    nouns = ["engineer", "artist", "doctor", "lawyer", "teacher", "baker", "pilot", "writer", "nurse", "clerk", "market", "garden"]
    texts: list[str] = []
    for name in names:
        for verb in verbs:
            texts.append(f"{name} {verb} today.")
            texts.append(f"{name} {verb} again today.")
    for name in names:
        for adj in adjectives:
            for noun in nouns:
                texts.append(f"{name} is a {adj} {noun}.")
                texts.append(f"{name} found the {adj} {noun}.")
    return texts


def _family_a_segments_for_length(tokenizer: Any, target_l: int, *, count: int = 8) -> list[tuple[str, list[int]]]:
    found: list[tuple[str, list[int]]] = []
    seen_ids: set[tuple[int, ...]] = set()
    for text in _family_a_candidate_texts():
        ids = _encode(tokenizer, text)
        key = tuple(ids)
        if len(ids) == target_l and key not in seen_ids:
            found.append((text, ids))
            seen_ids.add(key)
            if len(found) >= count:
                return found
    raise RuntimeError(f"only {len(found)} Family A segments survived for L={target_l}; need {count}")


def _family_b_word_groups() -> list[dict[str, list[str]]]:
    return [
        {
            "names": ["Alice", "Boris", "Derek", "Elena", "Farid", "Hector", "Iris", "Jonas"],
            "adjs": ["calm", "brisk", "careful", "eager", "formal", "gentle", "honest", "jolly"],
            "jobs": ["engineer", "artist", "doctor", "lawyer", "teacher", "baker", "pilot", "writer"],
        },
        {
            "names": ["Liam", "Nolan", "Omar", "Priya", "Noel", "Otto", "Ruth", "Sybil"],
            "adjs": ["kind", "lucid", "merry", "nimble", "open", "patient", "quick", "steady"],
            "jobs": ["nurse", "clerk", "miner", "tailor", "farmer", "guard", "driver", "chef"],
        },
    ]


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
    return cap


def _reject(rejections: list[dict[str, Any]], **record: Any) -> None:
    assert record["reason"] in REJECTION_REASONS, record
    rejections.append(record)


def _parse_segment_lengths(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        vals = [int(x.strip()) for x in value.split(",") if x.strip()]
    else:
        vals = [int(x) for x in value]
    vals = sorted(set(vals))
    if 7 not in vals:
        raise RuntimeError("segment-lengths must include mandatory L=7")
    if len(vals) < 2:
        raise RuntimeError("segment-lengths must include at least two distinct L values")
    return vals


def build_stimuli(
    tokenizer: Any,
    *,
    families: set[str],
    max_length: int,
    min_repetitions: int,
    repetitions: int | None,
    limit_stimuli: int | None,
    segment_lengths: Iterable[int] = (4, 7),
) -> BuildResult:
    stimuli: list[Stimulus] = []
    rejected: list[dict[str, Any]] = []
    feasibility: list[dict[str, Any]] = []
    target_lengths = list(segment_lengths)

    for target_l in target_lengths:
        needed = min_repetitions * target_l
        feasible = needed <= max_length
        feasibility.append({"target_L": target_l, "r_min": min_repetitions, "needed_tokens": needed, "max_length": max_length, "feasible": feasible})
        if not feasible:
            raise RuntimeError(f"cell L={target_l} needs {needed} tokens, budget {max_length}")

    if "A" in families:
        for target_l in target_lengths:
            for seg_no, (text, ids) in enumerate(_family_a_segments_for_length(tokenizer, target_l)):
                r = _repeat_to_limit(ids, min_repetitions=min_repetitions, max_length=max_length, requested=repetitions)
                if r < min_repetitions:
                    _reject(rejected, stimulus_id=f"A{target_l}_{seg_no:02d}", family="A", target_L=target_l, reason="below_min_repetitions", token_len=len(ids), max_reps_possible=r)
                    continue
                input_ids = ids * r
                slots = {slot: [rep * len(ids) + slot for rep in range(r)] for slot in range(len(ids))}
                token_ids = {slot: ids[slot] for slot in range(len(ids))}
                stimuli.append(Stimulus(f"A{target_l}_{seg_no:02d}", "A", f"L{target_l}_seg{seg_no:02d}", input_ids, slots, token_ids, r, len(ids), text, target_l))

    if "B" in families:
        for group_i, group in enumerate(_family_b_word_groups()):
            reps: list[list[int]] = []
            content_words: list[dict[str, str]] = []
            for i in range(max(min_repetitions, repetitions or min_repetitions)):
                j = i % len(group["names"])
                sentence = f"{group['names'][j]} is a {group['adjs'][j]} {group['jobs'][j]}."
                reps.append(_encode(tokenizer, sentence))
                content_words.append({"name": group["names"][j], "adjective": group["adjs"][j], "profession": group["jobs"][j]})
            offsets: list[int] = []
            cursor = 0
            kept_reps: list[list[int]] = []
            kept_content: list[dict[str, str]] = []
            for rep_ids, content in zip(reps, content_words, strict=True):
                if cursor + len(rep_ids) > max_length:
                    break
                offsets.append(cursor)
                kept_reps.append(rep_ids)
                kept_content.append(content)
                cursor += len(rep_ids)
            if len(kept_reps) < min_repetitions:
                _reject(rejected, stimulus_id=f"B{group_i:02d}", family="B", target_L=None, reason="below_min_repetitions", token_len=None, max_reps_possible=len(kept_reps))
                continue
            slot_positions: dict[tuple[str, int], list[int]] = {}
            slot_tokens: dict[tuple[str, int], int] = {}
            constant_failed = False
            for rep_i, ids in enumerate(kept_reps):
                occ: Counter[str] = Counter()
                for j, tid in enumerate(ids):
                    piece = _decode_piece(tokenizer, tid)
                    if piece in FRAME_TOKENS:
                        key = (piece, occ[piece])
                        occ[piece] += 1
                        if key in slot_tokens and slot_tokens[key] != tid:
                            _reject(rejected, stimulus_id=f"B{group_i:02d}", family="B", target_L=None, reason="slot_token_not_constant", token_len=len(ids), max_reps_possible=len(kept_reps))
                            constant_failed = True
                            break
                        slot_tokens[key] = tid
                        slot_positions.setdefault(key, []).append(offsets[rep_i] + j)
                if constant_failed:
                    break
            if constant_failed:
                continue
            full_slots = [(key, pos) for key, pos in slot_positions.items() if len(pos) == len(kept_reps)]
            if not full_slots:
                _reject(rejected, stimulus_id=f"B{group_i:02d}", family="B", target_L=None, reason="frame_token_absent", token_len=None, max_reps_possible=len(kept_reps))
                continue
            slots = {slot_i: pos for slot_i, (_key, pos) in enumerate(sorted(full_slots))}
            token_ids = {slot_i: slot_tokens[key] for slot_i, (key, _pos) in enumerate(sorted(full_slots))}
            flat = [tok for rep in kept_reps for tok in rep]
            stimuli.append(Stimulus(f"B{group_i:02d}", "B", f"frame_group{group_i:02d}", flat, slots, token_ids, len(kept_reps), None, "{Name} is a {adj} {profession}.", None, kept_content))

    if "C" in families:
        base_ids = _encode(tokenizer, _family_c_text())
        r = _repeat_to_limit(base_ids, min_repetitions=1, max_length=max_length, requested=None)
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
        else:
            _reject(rejected, stimulus_id="C00", family="C", target_L=None, reason="insufficient_occurrences", token_len=len(base_ids), max_reps_possible=max((len(v) for v in token_to_positions.values()), default=0))

    if limit_stimuli is not None:
        stimuli = stimuli[:limit_stimuli]

    missing = sorted(f for f in families if not any(s.family == f for s in stimuli))
    if missing:
        raise RuntimeError(f"G5 family yield failed for {missing}; rejected_stimuli={json.dumps(rejected, sort_keys=True)}")
    if "A" in families:
        missing_cells = [L for L in target_lengths if not any(s.family == "A" and s.target_L == L for s in stimuli)]
        if missing_cells:
            raise RuntimeError(f"G5 M1.5 length-cell yield failed for L={missing_cells}; rejected_stimuli={json.dumps(rejected, sort_keys=True)}")
    return BuildResult(stimuli, rejected, feasibility)


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


def _null_stats(x: np.ndarray, y: np.ndarray, *, seed: int, alphas: np.ndarray, permutations: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed + 17)
    vals = np.asarray([
        _ridge_cv_r2(x, rng.permutation(y), folds=5, seed=seed + 1 + i, alphas=alphas)
        for i in range(max(1, permutations))
    ], dtype=float)
    return float(vals.mean()), float(np.quantile(vals, 0.99)), float(vals.max())


def _analyse_matrix(x: np.ndarray, y: np.ndarray, token_ids: np.ndarray, *, seed: int, variance_floor: float, null_permutations: int) -> dict[str, Any]:
    raw = float(np.mean(np.abs(x)))
    mean = x.mean(axis=0, keepdims=True)
    resid = x - mean
    resid_scale = float(np.mean(np.abs(resid)))
    frac = 0.0 if raw <= 0 else resid_scale / raw
    degenerate = frac < variance_floor
    basis = np.zeros((0, x.shape[1]), dtype=np.float32)
    token_acc_before = _nearest_centroid_cv_accuracy(x, token_ids, folds=5, seed=seed + 3)
    if degenerate:
        token_acc_after = token_acc_before
        return {
            "raw_mean_abs": raw,
            "resid_mean_abs": resid_scale,
            "position_fraction": frac,
            "degenerate": True,
            "ridge_r2": 0.0,
            "shuffled_ridge_r2": 0.0,
            "shuffled_ridge_r2_p99": 0.0,
            "r2_minus_null_mean": 0.0,
            "permutation_p_value": float("nan"),
            "pca_components_90pct": 0,
            "pca_residual_variance_fraction_90pct": 0.0,
            "pc1_spearman_repetition": float("nan"),
            "pc1_dominant_fourier_bin": -1,
            "r2_after_position_pc_projection": 0.0,
            "token_identity_acc_before": token_acc_before,
            "token_identity_acc_after": token_acc_after,
            "basis": basis,
        }
    alphas = np.logspace(-2, 4, 13)
    y_float = y.astype(float)
    r2 = _ridge_cv_r2(x, y_float, folds=5, seed=seed, alphas=alphas)
    null_mean, null_p99, null_max = _null_stats(x, y_float, seed=seed, alphas=alphas, permutations=null_permutations)
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
    r2_after = _ridge_cv_r2(x_proj, y_float, folds=5, seed=seed + 2, alphas=alphas)
    token_acc_after = _nearest_centroid_cv_accuracy(x_proj, token_ids, folds=5, seed=seed + 4)
    return {
        "raw_mean_abs": raw,
        "resid_mean_abs": resid_scale,
        "position_fraction": frac,
        "degenerate": False,
        "ridge_r2": r2,
        "shuffled_ridge_r2": null_mean,
        "shuffled_ridge_r2_p99": null_p99,
        "r2_minus_null_mean": r2 - null_mean,
        "permutation_p_value": (1.0 + float(null_max >= r2)) / (1.0 + max(1, null_permutations)),
        "pca_components_90pct": k90,
        "pca_residual_variance_fraction_90pct": frac_resid_var,
        "pc1_spearman_repetition": _spearman(pc1, y_float) if len(basis) else float("nan"),
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
    degenerate = frac < variance_floor
    if degenerate:
        r2 = 0.0
        shuffled_r2 = 0.0
        shuffled_r2_p99 = 0.0
        r2_minus_null_mean = 0.0
        permutation_p_value = float("nan")
        r2_after = 0.0
        k90 = 0
        frac_var = 0.0
        basis = eigvecs[:0]
    else:
        alphas = np.logspace(-2, 4, 13)
        r2 = _ridge_cv_r2(x, y, folds=5, seed=seed, alphas=alphas)
        shuffled_r2, shuffled_r2_p99, null_max = _null_stats(x, y, seed=seed, alphas=alphas, permutations=5)
        r2_minus_null_mean = r2 - shuffled_r2
        permutation_p_value = (1.0 + float(null_max >= r2)) / 6.0
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
        "degenerate": degenerate,
        "ridge_r2": r2,
        "shuffled_ridge_r2": shuffled_r2,
        "shuffled_ridge_r2_p99": shuffled_r2_p99,
        "r2_minus_null_mean": r2_minus_null_mean,
        "permutation_p_value": permutation_p_value,
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


def _trained_context(lm: Any) -> int | None:
    for obj in (getattr(lm, "model", None), getattr(getattr(lm, "model", None), "config", None), getattr(lm, "tokenizer", None)):
        for attr in ("max_position_embeddings", "n_positions", "n_ctx", "block_size", "context_length", "max_sequence_length"):
            value = getattr(obj, attr, None)
            if isinstance(value, int) and 0 < value < 1_000_000:
                return value
    if lm.tag == "nope-gpt-small":
        # The remote config does not expose a context field; the model card/checkpoint is GPT-small scale.
        return 1024
    return None


def _d_head(lm: Any) -> int:
    cfg = getattr(getattr(lm, "model", None), "config", None)
    hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None) or getattr(cfg, "embedding_dimensions", None)
    heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None) or getattr(cfg, "num_heads", None)
    if hidden and heads:
        return int(hidden) // int(heads)
    if lm.tag == "qwen3":
        return 128
    return 64


def run(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")

    lm = load_model(args.model, device=device, revision=args.revision)
    if lm.tag not in {"gpt2", "qwen3", "nope-gpt-small"} and not lm.tag.startswith("pythia"):
        raise NotImplementedError(f"M1.5 extraction not implemented for {lm.tag}")

    context = _trained_context(lm)
    if context is None and args.max_length is None:
        raise RuntimeError("could not derive trained context from model config; pass --max-length explicitly")
    max_length = int(args.max_length if args.max_length is not None else context - 32)
    families = {f.strip().upper() for f in args.families.split(",") if f.strip()}
    segment_lengths = _parse_segment_lengths(args.segment_lengths)
    d_head_for_budget = _d_head(lm)
    effective_min_repetitions = max(args.min_repetitions, 2 * d_head_for_budget)
    build = build_stimuli(
        lm.tokenizer,
        families=families,
        max_length=max_length,
        min_repetitions=effective_min_repetitions,
        repetitions=args.repetitions,
        limit_stimuli=args.limit_stimuli,
        segment_lengths=segment_lengths,
    )
    stimuli = build.stimuli
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
                        stats = _analyse_matrix(x, y, token_ids, seed=args.seed + stim_i + layer * 1000 + head, variance_floor=args.variance_floor, null_permutations=args.null_permutations)
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
    shuffle_ok = bool(np.quantile(summary["shuffled_ridge_r2_p99"], 0.99) <= args.shuffle_r2_abs_warn)

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
        "trained_context": context,
        "max_length": max_length,
        "min_repetitions": effective_min_repetitions,
        "cli_min_repetitions": args.min_repetitions,
        "segment_lengths": segment_lengths,
        "feasibility_matrix": build.feasibility,
        "rejected_stimuli": build.rejected_stimuli,
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
                "target_L": s.target_L,
                "preview": s.text_preview,
                "content_words": s.content_words,
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
    p.add_argument("--max-length", type=int, default=None, help="Token budget; defaults to trained context minus 32.")
    p.add_argument("--min-repetitions", type=int, default=120, help="Lower-bound override; effective R_min is max(this, 2*d_head).")
    p.add_argument("--segment-lengths", default="4,7", help="Comma-separated Family A segment lengths; must include 7 and at least two values.")
    p.add_argument("--repetitions", type=int, default=None)
    p.add_argument("--limit-stimuli", type=int, default=None)
    p.add_argument("--limit-layers", type=int, default=None)
    p.add_argument("--limit-heads", type=int, default=None)
    p.add_argument("--variance-floor", type=float, default=1e-5)
    p.add_argument("--shuffle-r2-abs-warn", type=float, default=0.05)
    p.add_argument("--null-permutations", type=int, default=5)
    p.add_argument(
        "--aggregate-sample-cap",
        type=int,
        default=2048,
        help="Reservoir sample size per family/variant/layer/head for aggregate M1.5.6 token-identity fidelity.",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
