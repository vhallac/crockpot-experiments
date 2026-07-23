from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from deadkeys.common.loading import MODEL_IDS, load_model
from kaddress.scripts.address_purity import _environment_summary


CANDIDATE_MARKERS = [
    # Ordered by empirical base-probability similarity for the NoPE-GPT-Small
    # Family-A prefix, but v1.1 searches per stimulus instead of assuming one
    # globally neutral marker set.
    "soon",
    "early",
    "briefly",
    "now",
    "still",
    "later",
    "today",
    "clearly",
    "indeed",
    "daily",
    "again",
    "quietly",
    "truly",
    "surely",
    "calmly",
    "once",
    "then",
    "always",
    "often",
    "usually",
    "sometimes",
    "perhaps",
    "maybe",
    "likely",
    "certainly",
    "possibly",
    "already",
    "finally",
    "simply",
    "mostly",
    "really",
    "actually",
    "gradually",
    "quickly",
    "slowly",
    "carefully",
    "steadily",
    "gently",
    "softly",
    "plainly",
    "directly",
    "exactly",
    "fully",
    "partly",
    "equally",
    "nearly",
    "slightly",
    "newly",
    "recently",
    "previously",
    "currently",
    "meanwhile",
    "afterward",
    "before",
    "after",
    "here",
    "there",
    "locally",
    "globally",
    "jointly",
    "separately",
    "together",
    "apart",
    "onward",
    "forward",
    "back",
    "away",
    "around",
    "inside",
    "outside",
    "above",
    "below",
    "near",
    "far",
    "first",
    "second",
    "third",
    "next",
    "last",
    "final",
    "initial",
    "fresh",
    "clean",
    "neutral",
    "common",
    "ordinary",
    "regular",
    "constant",
    "stable",
    "simple",
    "plain",
    "blank",
    "quiet",
    "calm",
    "fair",
    "even",
    "mild",
    "spare",
    "extra",
    "other",
    "same",
    "different",
    "small",
    "large",
    "short",
    "long",
    "low",
    "high",
    "real",
    "true",
    "false",
    "good",
    "better",
    "best",
    "worst",
    "weak",
    "strong",
    "loose",
    "tight",
    "open",
    "closed",
    "empty",
    "full",
    "round",
    "square",
    "sharp",
    "dull",
    "bright",
    "dark",
    "warm",
    "cool",
    "hot",
    "cold",
    "dry",
    "wet",
    "thin",
    "thick",
    "light",
    "heavy",
    "basic",
    "advanced",
    "major",
    "minor",
    "prime",
    "solid",
    "rough",
    "smooth",
    "public",
    "private",
    "central",
    "remote",
    "local",
]

PROBED_MARKER_ROLES = ("target", "donor", "altered", "readout")


@dataclass(frozen=True)
class M16Stimulus:
    stimulus_id: str
    input_ids: list[int]
    transitivity_input_ids: list[int]
    prefix_len: int
    period_token_id: int
    markers: list[str]
    marker_token_ids: list[int]
    marker_roles: dict[str, int]
    marker_positions: list[int]
    continuation_positions: list[int]
    transitivity_continuation_positions: list[int]
    final_query_start: int
    readout_pos: int
    transitivity_readout_pos: int
    target_rep: int
    donor_rep: int
    altered_rep: int
    readout_rep: int
    text_preview: str


@dataclass(frozen=True)
class ForwardReadout:
    probs: torch.Tensor
    attention: torch.Tensor
    logits: torch.Tensor


def _encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _single_token_markers(tokenizer: Any, requested: int) -> tuple[list[str], list[int]]:
    markers: list[str] = []
    token_ids: list[int] = []
    for word in CANDIDATE_MARKERS:
        ids = _encode(tokenizer, " " + word)
        if len(ids) == 1 and ids[0] not in token_ids:
            markers.append(word)
            token_ids.append(ids[0])
        if len(markers) >= requested:
            return markers, token_ids
    raise RuntimeError(f"only found {len(markers)} single-token continuation markers; need {requested}")


def _probe_repetitions(repetitions: int) -> dict[str, int]:
    if repetitions < 8:
        raise RuntimeError("M1.6 v1.1 requires at least 8 repetitions for separated probe roles")
    target = max(2, repetitions // 2 - 1)
    donor = min(repetitions - 3, target + max(3, repetitions // 4))
    if donor == target:
        donor = target + 1
    altered = max(1, target - max(2, repetitions // 8))
    while altered in {target, donor}:
        altered -= 1
    readout = repetitions - 1
    return {"target": target, "donor": donor, "altered": altered, "readout": readout}


def _build_one_stimulus(
    tokenizer: Any,
    *,
    stimulus_id: str,
    prefix: str,
    repetitions: int,
    markers: list[str],
    marker_token_ids: list[int],
) -> M16Stimulus:
    period = _encode(tokenizer, ".")
    if len(period) != 1:
        raise RuntimeError("expected '.' to be one token for M1.6 stimulus construction")
    prefix_ids = _encode(tokenizer, prefix)
    roles = _probe_repetitions(repetitions)
    marker_by_rep = {roles[role]: i for i, role in enumerate(PROBED_MARKER_ROLES)}

    reps: list[int] = []
    marker_positions: list[int] = [-1] * len(marker_token_ids)
    continuation_positions: list[int] = []
    for rep_i in range(repetitions):
        start = len(reps)
        reps.extend(prefix_ids)
        cont_pos = start + len(prefix_ids)
        continuation_positions.append(cont_pos)
        marker_i = marker_by_rep.get(rep_i)
        if marker_i is None:
            reps.extend(period)
        else:
            marker_positions[marker_i] = cont_pos
            reps.append(marker_token_ids[marker_i])
            reps.extend(period)
    final_query_start = len(reps)
    reps.extend(prefix_ids)
    readout_pos = len(reps) - 1

    # Mandatory v1.1 transitivity stimulus: stop after the altered interior
    # repetition, then insert a matching prefix query. If the head implements
    # transitive induction, the most recent match+1 is the altered marker.
    trans_reps: list[int] = []
    trans_continuations: list[int] = []
    for rep_i in range(roles["altered"] + 1):
        start = len(trans_reps)
        trans_reps.extend(prefix_ids)
        cont_pos = start + len(prefix_ids)
        trans_continuations.append(cont_pos)
        if rep_i == roles["altered"]:
            trans_reps.append(marker_token_ids[PROBED_MARKER_ROLES.index("altered")])
            trans_reps.extend(period)
        else:
            trans_reps.extend(period)
    trans_reps.extend(prefix_ids)
    transitivity_readout_pos = len(trans_reps) - 1

    role_marker_text = {role: markers[i] for i, role in enumerate(PROBED_MARKER_ROLES)}
    text_preview = f"{prefix}. x{repetitions} probes={role_marker_text}"
    return M16Stimulus(
        stimulus_id=stimulus_id,
        input_ids=reps,
        transitivity_input_ids=trans_reps,
        prefix_len=len(prefix_ids),
        period_token_id=period[0],
        markers=markers,
        marker_token_ids=marker_token_ids,
        marker_roles={role: i for i, role in enumerate(PROBED_MARKER_ROLES)},
        marker_positions=marker_positions,
        continuation_positions=continuation_positions,
        transitivity_continuation_positions=trans_continuations,
        final_query_start=final_query_start,
        readout_pos=readout_pos,
        transitivity_readout_pos=transitivity_readout_pos,
        target_rep=roles["target"],
        donor_rep=roles["donor"],
        altered_rep=roles["altered"],
        readout_rep=roles["readout"],
        text_preview=text_preview,
    )


def build_stimuli(tokenizer: Any, *, repetitions: int, limit_stimuli: int | None) -> list[M16Stimulus]:
    markers, marker_token_ids = _single_token_markers(tokenizer, len(PROBED_MARKER_ROLES))
    prefixes = [
        "Alice is a successful engineer",
        "Boris is a careful artist",
        "Clara is a patient doctor",
        "Derek is a steady teacher",
    ]
    stimuli = [
        _build_one_stimulus(
            tokenizer,
            stimulus_id=f"M16_{stim_i:02d}",
            prefix=prefix,
            repetitions=repetitions,
            markers=markers,
            marker_token_ids=marker_token_ids,
        )
        for stim_i, prefix in enumerate(prefixes)
    ]
    return stimuli[:limit_stimuli] if limit_stimuli is not None else stimuli


def _extract_logits(output: Any) -> torch.Tensor:
    if isinstance(output, dict):
        return output["logits"]
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, torch.Tensor):
        return output
    raise TypeError(f"cannot extract logits from {type(output)!r}")


def _layer_modules(lm: Any) -> list[torch.nn.Module]:
    if lm.tag == "nope-gpt-small":
        return list(lm.model.model.body)
    if lm.tag == "qwen3":
        return list(lm.model.model.layers)
    raise NotImplementedError("M1.6 causal patching currently supports --model nope-gpt-small and --model qwen3")


def _plain_probs(lm: Any, input_ids: torch.Tensor, readout_pos: int) -> torch.Tensor:
    with torch.no_grad():
        logits = _extract_logits(lm.model(input_ids))
    return torch.softmax(logits[0, readout_pos, :], dim=-1).detach().float()


def _run_nope_with_attention_patch(
    lm: Any,
    input_ids: torch.Tensor,
    *,
    layer_idx: int,
    head_idx: int,
    readout_pos: int,
    target_pos: int,
    donor_pos: int,
    mode: str,
    noise_seed: int,
) -> ForwardReadout:
    layer = _layer_modules(lm)[layer_idx]
    attn_mod = layer.attention
    original_forward = attn_mod.forward
    captured: dict[str, torch.Tensor] = {}

    def patched_forward(x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.size()
        q, k, v = attn_mod.qkv_proj(x).split(attn_mod.embedding_dimensions, dim=-1)
        q = q.view(b, t, attn_mod.num_heads, attn_mod.head_dimensions).transpose(1, 2)
        k = k.view(b, t, attn_mod.num_heads, attn_mod.head_dimensions).transpose(1, 2).clone()
        v = v.view(b, t, attn_mod.num_heads, attn_mod.head_dimensions).transpose(1, 2).clone()
        if mode in {"k", "both"}:
            k[:, head_idx, target_pos, :] = k[:, head_idx, donor_pos, :]
        if mode in {"v", "both"}:
            v[:, head_idx, target_pos, :] = v[:, head_idx, donor_pos, :]
        if mode == "noise":
            gen = torch.Generator(device=x.device)
            gen.manual_seed(noise_seed)
            noise_k = torch.randn(k[:, head_idx, target_pos, :].shape, generator=gen, device=x.device, dtype=k.dtype)
            noise_v = torch.randn(v[:, head_idx, target_pos, :].shape, generator=gen, device=x.device, dtype=v.dtype)
            k_norm = torch.linalg.vector_norm(k[:, head_idx, target_pos, :], dim=-1, keepdim=True).clamp_min(1e-12)
            v_norm = torch.linalg.vector_norm(v[:, head_idx, target_pos, :], dim=-1, keepdim=True).clamp_min(1e-12)
            k[:, head_idx, target_pos, :] = noise_k * (k_norm / torch.linalg.vector_norm(noise_k, dim=-1, keepdim=True).clamp_min(1e-12))
            v[:, head_idx, target_pos, :] = noise_v * (v_norm / torch.linalg.vector_norm(noise_v, dim=-1, keepdim=True).clamp_min(1e-12))
        scores = torch.matmul(q, k.transpose(-2, -1)) * attn_mod.scale
        causal = torch.ones((t, t), dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~causal[None, None, :, :], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        captured["attention"] = weights[0, head_idx, readout_pos, :].detach().float()
        z = torch.matmul(weights, v)
        z = z.transpose(1, 2).contiguous().view(b, t, d)
        return attn_mod.out_proj(z)

    attn_mod.forward = patched_forward  # type: ignore[method-assign]
    try:
        with torch.no_grad():
            output = lm.model(input_ids)
    finally:
        attn_mod.forward = original_forward  # type: ignore[method-assign]
    logits = _extract_logits(output)
    probs = torch.softmax(logits[0, readout_pos, :], dim=-1).detach().float()
    return ForwardReadout(probs=probs, attention=captured["attention"], logits=logits.detach().float())


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


def _run_qwen_with_attention_patch(
    lm: Any,
    input_ids: torch.Tensor,
    *,
    layer_idx: int,
    head_idx: int,
    readout_pos: int,
    target_pos: int,
    donor_pos: int,
    mode: str,
    noise_seed: int,
) -> ForwardReadout:
    layer = _layer_modules(lm)[layer_idx]
    attn_mod = layer.self_attn
    original_forward = attn_mod.forward
    captured: dict[str, torch.Tensor] = {}

    def patched_forward(
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if past_key_values is not None:
            raise NotImplementedError("M1.6 qwen3 patching expects a no-cache forward")
        if position_embeddings is None:
            raise RuntimeError("Qwen3 forward did not provide RoPE position embeddings")
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn_mod.head_dim)
        q = attn_mod.q_norm(attn_mod.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        k = attn_mod.k_norm(attn_mod.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        v = attn_mod.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        # v1.1 fix (RoPE addressing test): transplant the donor's PRE-rotation key (k_pre) into
        # the target slot, then rotate — so the donor CONTENT is re-addressed to target_pos by
        # RoPE instead of carrying the donor's own position rotation. Patching post-rotation
        # (k_post) would confound content with the donor's absolute position and make the
        # addressing readout uninterpretable. V is unrotated, so its patch stage is immaterial.
        k = _repeat_kv(k, attn_mod.num_key_value_groups).clone()
        v = _repeat_kv(v, attn_mod.num_key_value_groups).clone()
        if mode in {"k", "both"}:
            k[:, head_idx, target_pos, :] = k[:, head_idx, donor_pos, :]
        if mode in {"v", "both"}:
            v[:, head_idx, target_pos, :] = v[:, head_idx, donor_pos, :]
        if mode == "noise":
            gen = torch.Generator(device=hidden_states.device)
            gen.manual_seed(noise_seed)
            noise_k = torch.randn(k[:, head_idx, target_pos, :].shape, generator=gen, device=hidden_states.device, dtype=k.dtype)
            noise_v = torch.randn(v[:, head_idx, target_pos, :].shape, generator=gen, device=hidden_states.device, dtype=v.dtype)
            k_norm = torch.linalg.vector_norm(k[:, head_idx, target_pos, :], dim=-1, keepdim=True).clamp_min(1e-12)
            v_norm = torch.linalg.vector_norm(v[:, head_idx, target_pos, :], dim=-1, keepdim=True).clamp_min(1e-12)
            k[:, head_idx, target_pos, :] = noise_k * (k_norm / torch.linalg.vector_norm(noise_k, dim=-1, keepdim=True).clamp_min(1e-12))
            v[:, head_idx, target_pos, :] = noise_v * (v_norm / torch.linalg.vector_norm(noise_v, dim=-1, keepdim=True).clamp_min(1e-12))
        # Rotate after patching: each slot (including the patched target) is rotated by its own
        # position, so the transplanted donor content is addressed to target_pos. RoPE preserves
        # norm, so the norm-matched noise control above stays valid across the rotation.
        q, k = _apply_rotary_pos_emb(q, k, *position_embeddings)
        scores = torch.matmul(q, k.transpose(2, 3)) * attn_mod.scaling
        if attention_mask is not None:
            mask = attention_mask[:, :, :, : k.shape[-2]]
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, float("-inf"))
            else:
                scores = scores + mask
        else:
            t = hidden_states.shape[1]
            causal = torch.ones((t, t), dtype=torch.bool, device=hidden_states.device).tril()
            scores = scores.masked_fill(~causal[None, None, :, :], float("-inf"))
        weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        captured["attention"] = weights[0, head_idx, readout_pos, :].detach().float()
        z = torch.matmul(weights, v).transpose(1, 2).contiguous().reshape(*input_shape, -1)
        return attn_mod.o_proj(z), weights

    attn_mod.forward = patched_forward  # type: ignore[method-assign]
    try:
        with torch.no_grad():
            output = lm.model(input_ids, use_cache=False)
    finally:
        attn_mod.forward = original_forward  # type: ignore[method-assign]
    logits = _extract_logits(output)
    probs = torch.softmax(logits[0, readout_pos, :], dim=-1).detach().float()
    return ForwardReadout(probs=probs, attention=captured["attention"], logits=logits.detach().float())


def _run_with_attention_patch(lm: Any, input_ids: torch.Tensor, **kwargs: Any) -> ForwardReadout:
    if lm.tag == "nope-gpt-small":
        return _run_nope_with_attention_patch(lm, input_ids, **kwargs)
    if lm.tag == "qwen3":
        return _run_qwen_with_attention_patch(lm, input_ids, **kwargs)
    raise NotImplementedError("M1.6 causal patching currently supports --model nope-gpt-small and --model qwen3")


def _marker_probs(probs: torch.Tensor, marker_token_ids: list[int]) -> list[float]:
    ids = torch.as_tensor(marker_token_ids, dtype=torch.long, device=probs.device)
    return [float(x) for x in probs.index_select(0, ids).detach().cpu()]


def _neutrality_row(stim: M16Stimulus, probs: torch.Tensor) -> dict[str, Any]:
    vals = _marker_probs(probs, stim.marker_token_ids)
    mn = max(min(vals), 1e-30)
    mx = max(vals)
    return {
        "stimulus_id": stim.stimulus_id,
        "gate": "G6_marker_neutrality",
        "marker_count": len(vals),
        "min_marker_prob": mn,
        "max_marker_prob": mx,
        "max_min_ratio": mx / mn,
        "pass": bool(mx / mn < 3.0),
    }


def _search_g6_stimulus(
    lm: Any,
    *,
    stimulus_id: str,
    prefix: str,
    repetitions: int,
    device: torch.device,
    max_marker_sets: int,
    seed: int,
) -> tuple[M16Stimulus, dict[str, Any]]:
    candidates, candidate_ids = _single_token_markers(lm.tokenizer, len(CANDIDATE_MARKERS))
    checked = 0
    best: tuple[float, M16Stimulus, dict[str, Any]] | None = None
    seen: set[tuple[int, ...]] = set()
    rng = random.Random(seed)

    def candidate_combos() -> Any:
        # Try a small deterministic prefix first for stable historical behavior,
        # then sample the expanded vocabulary. Exact G6 is evaluated after marker
        # insertion, so cheap random search is more reliable than base-frequency
        # sorting for prefixes whose inserted probe words perturb the readout.
        yield tuple(range(len(PROBED_MARKER_ROLES)))
        while True:
            yield tuple(sorted(rng.sample(range(len(candidates)), len(PROBED_MARKER_ROLES))))

    for combo in candidate_combos():
        if combo in seen:
            continue
        seen.add(combo)
        markers = [candidates[i] for i in combo]
        marker_ids = [candidate_ids[i] for i in combo]
        stim = _build_one_stimulus(
            lm.tokenizer,
            stimulus_id=stimulus_id,
            prefix=prefix,
            repetitions=repetitions,
            markers=markers,
            marker_token_ids=marker_ids,
        )
        input_ids = torch.tensor([stim.input_ids], dtype=torch.long, device=device)
        gate = _neutrality_row(stim, _plain_probs(lm, input_ids, stim.readout_pos))
        gate["searched_marker_sets"] = checked + 1
        gate["selected_markers"] = ",".join(markers)
        ratio = float(gate["max_min_ratio"])
        if best is None or ratio < best[0]:
            best = (ratio, stim, gate)
        checked += 1
        if gate["pass"]:
            return stim, gate
        if checked >= max_marker_sets:
            break
    assert best is not None
    raise RuntimeError(
        f"G6 marker search failed for {stimulus_id} after {checked} sets; "
        f"best_ratio={best[0]:.3f} best_markers={best[2]['selected_markers']}"
    )


def _select_g6_stimuli(lm: Any, *, repetitions: int, limit_stimuli: int | None, device: torch.device, max_marker_sets: int) -> tuple[list[M16Stimulus], list[dict[str, Any]]]:
    prefixes = [
        "Alice is a successful engineer",
        "Boris is a careful artist",
        "Clara is a patient doctor",
        "Derek is a steady teacher",
    ]
    if limit_stimuli is not None:
        prefixes = prefixes[:limit_stimuli]
    stimuli: list[M16Stimulus] = []
    gates: list[dict[str, Any]] = []
    for stim_i, prefix in enumerate(prefixes):
        stim, gate = _search_g6_stimulus(
            lm,
            stimulus_id=f"M16_{stim_i:02d}",
            prefix=prefix,
            repetitions=repetitions,
            device=device,
            max_marker_sets=max_marker_sets,
            seed=17_003 + stim_i,
        )
        stimuli.append(stim)
        gates.append(gate)
    return stimuli, gates


def _attention_sum(attention: torch.Tensor, positions: list[int]) -> float:
    if not positions:
        return 0.0
    ids = torch.as_tensor(positions, dtype=torch.long, device=attention.device)
    return float(attention.index_select(0, ids).sum().item())


def _induction_metrics(stim: M16Stimulus, attention: torch.Tensor) -> dict[str, float]:
    continuation_mass = _attention_sum(attention, stim.continuation_positions)
    previous_positions = [p - 1 for p in stim.continuation_positions]
    previous_token_mass = _attention_sum(attention, previous_positions)
    most_recent = stim.continuation_positions[-1]
    most_recent_mass = float(attention[most_recent].item())
    elsewhere = max(0.0, 1.0 - continuation_mass - previous_token_mass)
    return {
        "induction_match_plus_one_mass": continuation_mass,
        "induction_most_recent_match_plus_one_mass": most_recent_mass,
        "induction_other_match_plus_one_mass": continuation_mass - most_recent_mass,
        "induction_previous_token_match_mass": previous_token_mass,
        "induction_elsewhere_mass": elsewhere,
    }


def _transitivity_metrics(stim: M16Stimulus, readout: ForwardReadout) -> dict[str, float | int]:
    altered_i = stim.marker_roles["altered"]
    altered_token_id = stim.marker_token_ids[altered_i]
    marker_vals = _marker_probs(readout.probs, stim.marker_token_ids)
    altered_prob = marker_vals[altered_i]
    continuation_mass = _attention_sum(readout.attention, stim.transitivity_continuation_positions)
    altered_pos = stim.transitivity_continuation_positions[-1]
    altered_attention = float(readout.attention[altered_pos].item())
    rank = int((readout.probs > readout.probs[altered_token_id]).sum().item() + 1)
    return {
        "transitivity_altered_marker_prob": altered_prob,
        "transitivity_altered_marker_rank": rank,
        "transitivity_match_plus_one_mass": continuation_mass,
        "transitivity_altered_match_plus_one_attention": altered_attention,
        "transitivity_other_match_plus_one_mass": continuation_mass - altered_attention,
    }


def _classification(rows: pd.DataFrame, *, attention_margin: float, output_margin: float) -> pd.DataFrame:
    out: list[dict[str, Any]] = []
    for (layer, head), g in rows.groupby(["layer", "head"]):
        base = g[g["patch_mode"] == "baseline"]
        k = g[g["patch_mode"] == "k"]
        v = g[g["patch_mode"] == "v"]
        both = g[g["patch_mode"] == "both"]
        noise = g[g["patch_mode"] == "noise"]

        def mean_col(df: pd.DataFrame, col: str) -> float:
            return float(df[col].mean()) if not df.empty else float("nan")

        k_attn_delta = mean_col(k, "target_attention_delta")
        noise_attn_delta = mean_col(noise, "target_attention_delta")
        v_prob_delta = mean_col(v, "donor_prob_delta")
        both_prob_delta = mean_col(both, "donor_prob_delta")
        noise_prob_delta = mean_col(noise, "donor_prob_delta")
        induction_mass = mean_col(base, "induction_match_plus_one_mass")
        trans_prob = mean_col(base, "transitivity_altered_marker_prob")
        trans_rank = mean_col(base, "transitivity_altered_marker_rank")
        attention_above_noise = k_attn_delta > 0 and k_attn_delta > noise_attn_delta + attention_margin
        output_above_noise = both_prob_delta > abs(noise_prob_delta) + output_margin
        transitivity_confirmed = trans_rank <= 10 and trans_prob > output_margin

        if not attention_above_noise and abs(noise_attn_delta) >= abs(k_attn_delta) and abs(noise_attn_delta) > attention_margin:
            cls = "confounded_noise_sensitive"
        elif attention_above_noise and output_above_noise:
            cls = "addressing"
        elif attention_above_noise and not output_above_noise:
            cls = "anti_collision_or_inert_attention_only"
        elif induction_mass > 0.20 and transitivity_confirmed:
            cls = "transitive_induction"
        elif induction_mass > 0.20:
            cls = "induction_unconfirmed"
        elif max(abs(k_attn_delta), abs(v_prob_delta), abs(both_prob_delta)) <= max(output_margin, 1e-4) and induction_mass <= 0.05:
            cls = "inert"
        elif induction_mass <= 0.05:
            cls = "anti_collision_or_content_driven"
        else:
            cls = "mixed"
        out.append({
            "layer": int(layer),
            "head": int(head),
            "classification": cls,
            "mean_patch_k_target_attention_delta": k_attn_delta,
            "mean_noise_target_attention_delta": noise_attn_delta,
            "g7_noise_controlled_attention_pass": bool(attention_above_noise),
            "mean_patch_v_donor_prob_delta": v_prob_delta,
            "mean_patch_both_donor_prob_delta": both_prob_delta,
            "mean_noise_donor_prob_delta": noise_prob_delta,
            "output_above_noise": bool(output_above_noise),
            "mean_induction_match_plus_one_mass": induction_mass,
            "mean_transitivity_altered_marker_prob": trans_prob,
            "mean_transitivity_altered_marker_rank": trans_rank,
            "transitivity_confirmed": bool(transitivity_confirmed),
        })
    return pd.DataFrame(out)


def run(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    if args.repetitions < 128 and not args.allow_low_repetitions:
        raise RuntimeError("M1.6 v1.1 requires --repetitions >= 128; use --allow-low-repetitions only for smoke tests")
    lm = load_model(args.model, device=device, revision=args.revision)
    if lm.tag not in {"nope-gpt-small", "qwen3"}:
        raise NotImplementedError("M1.6 v1.1 harness supports --model nope-gpt-small and --model qwen3")
    stimuli, stimulus_gates = _select_g6_stimuli(
        lm,
        repetitions=args.repetitions,
        limit_stimuli=args.limit_stimuli,
        device=device,
        max_marker_sets=args.max_marker_sets,
    )
    n_layers = min(lm.n_layers, args.limit_layers or lm.n_layers)
    n_heads = min(lm.n_heads, args.limit_heads or lm.n_heads)
    rows: list[dict[str, Any]] = []
    gates: list[dict[str, Any]] = []
    for gate in stimulus_gates:
        gates.append(gate | {"layer": -1, "head": -1})
    start = time.monotonic()
    unit = 0
    print(f"starting M1.6 discriminator model={args.model} device={device} stimuli={len(stimuli)} layers={n_layers} heads={n_heads} repetitions={args.repetitions}", flush=True)
    for stim_i, stim in enumerate(stimuli):
        input_ids = torch.tensor([stim.input_ids], dtype=torch.long, device=device)
        transitivity_ids = torch.tensor([stim.transitivity_input_ids], dtype=torch.long, device=device)
        target_pos = stim.marker_positions[stim.marker_roles["target"]]
        donor_pos = stim.marker_positions[stim.marker_roles["donor"]]
        for layer in range(n_layers):
            for head in range(n_heads):
                baseline = _run_with_attention_patch(lm, input_ids, layer_idx=layer, head_idx=head, readout_pos=stim.readout_pos, target_pos=target_pos, donor_pos=donor_pos, mode="baseline", noise_seed=args.seed + unit)
                gates.append(_neutrality_row(stim, baseline.probs) | {"layer": layer, "head": head})
                base_probs = _marker_probs(baseline.probs, stim.marker_token_ids)
                induction = _induction_metrics(stim, baseline.attention)
                transitivity = _run_with_attention_patch(lm, transitivity_ids, layer_idx=layer, head_idx=head, readout_pos=stim.transitivity_readout_pos, target_pos=target_pos if target_pos < len(stim.transitivity_input_ids) else 0, donor_pos=donor_pos if donor_pos < len(stim.transitivity_input_ids) else 0, mode="baseline", noise_seed=args.seed + 200_000 + unit)
                trans_metrics = _transitivity_metrics(stim, transitivity)
                base_target_attn = float(baseline.attention[target_pos].item())
                base_donor_attn = float(baseline.attention[donor_pos].item())
                target_i = stim.marker_roles["target"]
                donor_i = stim.marker_roles["donor"]
                base_donor_prob = base_probs[donor_i]
                base_target_prob = base_probs[target_i]
                common = {
                    "model": args.model,
                    "hf_id": lm.hf_id,
                    "stimulus_id": stim.stimulus_id,
                    "layer": layer,
                    "head": head,
                    "readout_pos": stim.readout_pos,
                    "transitivity_readout_pos": stim.transitivity_readout_pos,
                    "repetitions": args.repetitions,
                    "target_rep": stim.target_rep,
                    "donor_rep": stim.donor_rep,
                    "altered_rep": stim.altered_rep,
                    "readout_rep": stim.readout_rep,
                    "target_marker": stim.markers[target_i],
                    "donor_marker": stim.markers[donor_i],
                    "altered_marker": stim.markers[stim.marker_roles["altered"]],
                    "target_marker_token_id": stim.marker_token_ids[target_i],
                    "donor_marker_token_id": stim.marker_token_ids[donor_i],
                    "altered_marker_token_id": stim.marker_token_ids[stim.marker_roles["altered"]],
                    "baseline_target_attention": base_target_attn,
                    "baseline_donor_attention": base_donor_attn,
                    "baseline_target_prob": base_target_prob,
                    "baseline_donor_prob": base_donor_prob,
                    **induction,
                    **trans_metrics,
                }
                rows.append(common | {"patch_mode": "baseline", "target_attention": base_target_attn, "donor_attention": base_donor_attn, "target_prob": base_target_prob, "donor_prob": base_donor_prob, "target_attention_delta": 0.0, "donor_prob_delta": 0.0})
                for mode in ("k", "v", "both", "noise"):
                    patched = _run_with_attention_patch(lm, input_ids, layer_idx=layer, head_idx=head, readout_pos=stim.readout_pos, target_pos=target_pos, donor_pos=donor_pos, mode=mode, noise_seed=args.seed + 100_000 + unit)
                    probs = _marker_probs(patched.probs, stim.marker_token_ids)
                    target_attn = float(patched.attention[target_pos].item())
                    donor_attn = float(patched.attention[donor_pos].item())
                    rows.append(common | {"patch_mode": mode, "target_attention": target_attn, "donor_attention": donor_attn, "target_prob": probs[target_i], "donor_prob": probs[donor_i], "target_attention_delta": target_attn - base_target_attn, "donor_prob_delta": probs[donor_i] - base_donor_prob})
                unit += 1
                if unit % args.progress_every == 0:
                    elapsed = max(time.monotonic() - start, 1e-9)
                    total = len(stimuli) * n_layers * n_heads
                    eta = (total - unit) / max(unit / elapsed, 1e-9)
                    print(f"progress units={unit}/{total} rate={unit/elapsed:.3f}/s eta={eta/60:.1f}m stimulus={stim.stimulus_id} layer={layer} head={head}", flush=True)
        print(f"processed {stim.stimulus_id} seq={len(stim.input_ids)} transitivity_seq={len(stim.transitivity_input_ids)} units={unit}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    row_df = pd.DataFrame(rows)
    gate_df = pd.DataFrame(gates)
    class_df = _classification(row_df, attention_margin=args.attention_margin, output_margin=args.output_margin)
    summary_path = out / f"kaddress_m16_{args.model}.csv"
    gate_path = out / f"kaddress_m16_gates_{args.model}.csv"
    class_path = out / f"kaddress_m16_classification_{args.model}.csv"
    manifest_path = out / f"kaddress_m16_manifest_{args.model}.json"
    row_df.to_csv(summary_path, index=False)
    gate_df.to_csv(gate_path, index=False)
    class_df.to_csv(class_path, index=False)
    manifest = {
        "script": "kaddress.scripts.m16_discriminator",
        "spec_slice": "ADDENDUM §5-M1.6 v1.1 hypothesis discriminator",
        "model": args.model,
        "hf_id": lm.hf_id,
        "revision": args.revision,
        "seed": args.seed,
        "repetitions": args.repetitions,
        "allow_low_repetitions": args.allow_low_repetitions,
        "limit_stimuli": args.limit_stimuli,
        "limit_layers": args.limit_layers,
        "limit_heads": args.limit_heads,
        "stimulus_count": len(stimuli),
        "summary_rows": int(len(row_df)),
        "classification_rows": int(len(class_df)),
        "attention_margin": args.attention_margin,
        "output_margin": args.output_margin,
        "gate_g6_pass": "PASS" if bool(gate_df[gate_df["gate"] == "G6_marker_neutrality"]["pass"].all()) else "FAIL",
        "gate_g7_pass_count": int(class_df["g7_noise_controlled_attention_pass"].sum()),
        "transitivity_confirmed_count": int(class_df["transitivity_confirmed"].sum()),
        "environment": _environment_summary(device),
        "stimuli": [stim.__dict__ for stim in stimuli],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {summary_path}")
    print(f"wrote {gate_path}")
    print(f"wrote {class_path}")
    print(f"wrote {manifest_path}")
    print(f"gate_g6_pass={manifest['gate_g6_pass']}")
    print(f"gate_g7_pass_count={manifest['gate_g7_pass_count']}")
    print(f"transitivity_confirmed_count={manifest['transitivity_confirmed_count']}")
    print(class_df["classification"].value_counts().to_string())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M1.6 addressing-vs-induction discriminator")
    p.add_argument("--model", default="nope-gpt-small", choices=sorted(MODEL_IDS))
    p.add_argument("--output-dir", default="outputs/k_address_space_m16_nope_gpt_small")
    p.add_argument("--device", default="cpu")
    p.add_argument("--revision", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repetitions", type=int, default=128, help="Total repeated clauses. M1.6 v1.1 marks only probed repetitions; full runs require >=128.")
    p.add_argument("--allow-low-repetitions", action="store_true", help="Permit repetitions <128 for local/CUDA smoke tests only.")
    p.add_argument("--limit-stimuli", type=int, default=None)
    p.add_argument("--limit-layers", type=int, default=None)
    p.add_argument("--limit-heads", type=int, default=None)
    p.add_argument("--max-marker-sets", type=int, default=512)
    p.add_argument("--attention-margin", type=float, default=0.02)
    p.add_argument("--output-margin", type=float, default=1e-4)
    p.add_argument("--progress-every", type=int, default=20)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
