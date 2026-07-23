from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from deadkeys.common.loading import MODEL_IDS, load_model
from kaddress.scripts.address_purity import _environment_summary


CANDIDATE_MARKERS = [
    # Ordered by empirical base-probability similarity for the NoPE-GPT-Small
    # Family-A prefix, so the default first four satisfy the M1.6 G6 marker
    # neutrality gate better than a semantically natural but frequency-skewed
    # list such as {today, again, still, now}.
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
]


@dataclass(frozen=True)
class M16Stimulus:
    stimulus_id: str
    input_ids: list[int]
    prefix_len: int
    rep_len: int
    markers: list[str]
    marker_token_ids: list[int]
    marker_positions: list[int]
    final_query_start: int
    readout_pos: int
    target_rep: int
    donor_rep: int
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


def build_stimuli(tokenizer: Any, *, repetitions: int, limit_stimuli: int | None) -> list[M16Stimulus]:
    markers, marker_token_ids = _single_token_markers(tokenizer, repetitions)
    period = _encode(tokenizer, ".")
    if len(period) != 1:
        raise RuntimeError("expected '.' to be one token for M1.6 stimulus construction")
    prefixes = [
        "Alice is a successful engineer",
        "Boris is a careful artist",
        "Clara is a patient doctor",
        "Derek is a steady teacher",
    ]
    stimuli: list[M16Stimulus] = []
    for stim_i, prefix in enumerate(prefixes):
        prefix_ids = _encode(tokenizer, prefix)
        reps: list[int] = []
        marker_positions: list[int] = []
        for rep_i, marker_id in enumerate(marker_token_ids):
            start = len(reps)
            reps.extend(prefix_ids)
            marker_positions.append(start + len(prefix_ids))
            reps.append(marker_id)
            reps.extend(period)
        final_query_start = len(reps)
        reps.extend(prefix_ids)
        readout_pos = len(reps) - 1
        target_rep = max(1, repetitions // 2 - 1)
        donor_rep = min(repetitions - 2, target_rep + max(2, repetitions // 4))
        text_preview = " ".join([f"{prefix} {m}." for m in markers[:4]]) + f" ... {prefix}"
        stimuli.append(
            M16Stimulus(
                stimulus_id=f"M16_{stim_i:02d}",
                input_ids=reps,
                prefix_len=len(prefix_ids),
                rep_len=len(prefix_ids) + 1 + len(period),
                markers=markers,
                marker_token_ids=marker_token_ids,
                marker_positions=marker_positions,
                final_query_start=final_query_start,
                readout_pos=readout_pos,
                target_rep=target_rep,
                donor_rep=donor_rep,
                text_preview=text_preview,
            )
        )
    return stimuli[:limit_stimuli] if limit_stimuli is not None else stimuli


def _extract_logits(output: Any) -> torch.Tensor:
    if isinstance(output, dict):
        return output["logits"]
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, torch.Tensor):
        return output
    raise TypeError(f"cannot extract logits from {type(output)!r}")


def _layer_modules_nope(lm: Any) -> list[torch.nn.Module]:
    if lm.tag != "nope-gpt-small":
        raise NotImplementedError("M1.6 causal patching is currently implemented for NoPE-GPT-Small only")
    return list(lm.model.model.body)


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
    layer = _layer_modules_nope(lm)[layer_idx]
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


def _induction_metrics(stim: M16Stimulus, attention: torch.Tensor) -> dict[str, float]:
    marker_pos = torch.as_tensor(stim.marker_positions, dtype=torch.long, device=attention.device)
    engineer_pos = torch.as_tensor([p - 1 for p in stim.marker_positions], dtype=torch.long, device=attention.device)
    marker_mass = float(attention.index_select(0, marker_pos).sum().item())
    previous_token_mass = float(attention.index_select(0, engineer_pos).sum().item())
    most_recent_marker_mass = float(attention[stim.marker_positions[-1]].item())
    elsewhere = max(0.0, 1.0 - marker_mass - previous_token_mass)
    return {
        "induction_match_plus_one_mass": marker_mass,
        "induction_most_recent_match_plus_one_mass": most_recent_marker_mass,
        "induction_other_match_plus_one_mass": marker_mass - most_recent_marker_mass,
        "induction_previous_token_match_mass": previous_token_mass,
        "induction_elsewhere_mass": elsewhere,
    }


def _classification(rows: pd.DataFrame) -> pd.DataFrame:
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
        v_prob_delta = mean_col(v, "donor_prob_delta")
        both_prob_delta = mean_col(both, "donor_prob_delta")
        noise_delta = abs(mean_col(noise, "donor_prob_delta"))
        induction_mass = mean_col(base, "induction_match_plus_one_mass")
        if noise_delta >= max(abs(k_attn_delta), abs(v_prob_delta), abs(both_prob_delta)) and noise_delta > 1e-4:
            cls = "confounded_noise_sensitive"
        elif k_attn_delta > 0.02 and both_prob_delta > 1e-4:
            cls = "addressing"
        elif induction_mass > 0.20 and both_prob_delta <= 1e-4:
            cls = "induction"
        elif max(abs(k_attn_delta), abs(v_prob_delta), abs(both_prob_delta)) <= 1e-4 and induction_mass <= 0.05:
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
            "mean_patch_v_donor_prob_delta": v_prob_delta,
            "mean_patch_both_donor_prob_delta": both_prob_delta,
            "mean_noise_abs_donor_prob_delta": noise_delta,
            "mean_induction_match_plus_one_mass": induction_mass,
        })
    return pd.DataFrame(out)


def run(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    lm = load_model(args.model, device=device, revision=args.revision)
    if lm.tag != "nope-gpt-small":
        raise NotImplementedError("This first M1.6 harness supports --model nope-gpt-small only")
    stimuli = build_stimuli(lm.tokenizer, repetitions=args.repetitions, limit_stimuli=args.limit_stimuli)
    n_layers = min(lm.n_layers, args.limit_layers or lm.n_layers)
    n_heads = min(lm.n_heads, args.limit_heads or lm.n_heads)
    rows: list[dict[str, Any]] = []
    gates: list[dict[str, Any]] = []
    start = time.monotonic()
    unit = 0
    print(f"starting M1.6 discriminator model={args.model} device={device} stimuli={len(stimuli)} layers={n_layers} heads={n_heads}", flush=True)
    for stim_i, stim in enumerate(stimuli):
        input_ids = torch.tensor([stim.input_ids], dtype=torch.long, device=device)
        target_pos = stim.marker_positions[stim.target_rep]
        donor_pos = stim.marker_positions[stim.donor_rep]
        for layer in range(n_layers):
            for head in range(n_heads):
                baseline = _run_nope_with_attention_patch(lm, input_ids, layer_idx=layer, head_idx=head, readout_pos=stim.readout_pos, target_pos=target_pos, donor_pos=donor_pos, mode="baseline", noise_seed=args.seed + unit)
                gates.append(_neutrality_row(stim, baseline.probs) | {"layer": layer, "head": head})
                base_probs = _marker_probs(baseline.probs, stim.marker_token_ids)
                induction = _induction_metrics(stim, baseline.attention)
                base_target_attn = float(baseline.attention[target_pos].item())
                base_donor_attn = float(baseline.attention[donor_pos].item())
                base_donor_prob = base_probs[stim.donor_rep]
                base_target_prob = base_probs[stim.target_rep]
                common = {
                    "model": args.model,
                    "hf_id": lm.hf_id,
                    "stimulus_id": stim.stimulus_id,
                    "layer": layer,
                    "head": head,
                    "readout_pos": stim.readout_pos,
                    "target_rep": stim.target_rep,
                    "donor_rep": stim.donor_rep,
                    "target_marker": stim.markers[stim.target_rep],
                    "donor_marker": stim.markers[stim.donor_rep],
                    "target_marker_token_id": stim.marker_token_ids[stim.target_rep],
                    "donor_marker_token_id": stim.marker_token_ids[stim.donor_rep],
                    "baseline_target_attention": base_target_attn,
                    "baseline_donor_attention": base_donor_attn,
                    "baseline_target_prob": base_target_prob,
                    "baseline_donor_prob": base_donor_prob,
                    **induction,
                }
                rows.append(common | {"patch_mode": "baseline", "target_attention": base_target_attn, "donor_attention": base_donor_attn, "target_prob": base_target_prob, "donor_prob": base_donor_prob, "target_attention_delta": 0.0, "donor_prob_delta": 0.0})
                for mode in ("k", "v", "both", "noise"):
                    patched = _run_nope_with_attention_patch(lm, input_ids, layer_idx=layer, head_idx=head, readout_pos=stim.readout_pos, target_pos=target_pos, donor_pos=donor_pos, mode=mode, noise_seed=args.seed + 100_000 + unit)
                    probs = _marker_probs(patched.probs, stim.marker_token_ids)
                    target_attn = float(patched.attention[target_pos].item())
                    donor_attn = float(patched.attention[donor_pos].item())
                    rows.append(common | {"patch_mode": mode, "target_attention": target_attn, "donor_attention": donor_attn, "target_prob": probs[stim.target_rep], "donor_prob": probs[stim.donor_rep], "target_attention_delta": target_attn - base_target_attn, "donor_prob_delta": probs[stim.donor_rep] - base_donor_prob})
                unit += 1
                if unit % args.progress_every == 0:
                    elapsed = max(time.monotonic() - start, 1e-9)
                    total = len(stimuli) * n_layers * n_heads
                    eta = (total - unit) / max(unit / elapsed, 1e-9)
                    print(f"progress units={unit}/{total} rate={unit/elapsed:.3f}/s eta={eta/60:.1f}m stimulus={stim.stimulus_id} layer={layer} head={head}", flush=True)
        print(f"processed {stim.stimulus_id} seq={len(stim.input_ids)} units={unit}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    row_df = pd.DataFrame(rows)
    gate_df = pd.DataFrame(gates)
    class_df = _classification(row_df)
    summary_path = out / f"kaddress_m16_{args.model}.csv"
    gate_path = out / f"kaddress_m16_gates_{args.model}.csv"
    class_path = out / f"kaddress_m16_classification_{args.model}.csv"
    manifest_path = out / f"kaddress_m16_manifest_{args.model}.json"
    row_df.to_csv(summary_path, index=False)
    gate_df.to_csv(gate_path, index=False)
    class_df.to_csv(class_path, index=False)
    manifest = {
        "script": "kaddress.scripts.m16_discriminator",
        "spec_slice": "ADDENDUM §5-M1.6 hypothesis discriminator",
        "model": args.model,
        "hf_id": lm.hf_id,
        "revision": args.revision,
        "seed": args.seed,
        "repetitions": args.repetitions,
        "limit_stimuli": args.limit_stimuli,
        "limit_layers": args.limit_layers,
        "limit_heads": args.limit_heads,
        "stimulus_count": len(stimuli),
        "summary_rows": int(len(row_df)),
        "classification_rows": int(len(class_df)),
        "gate_g6_pass": "PASS" if bool(gate_df["pass"].all()) else "FAIL",
        "environment": _environment_summary(device),
        "stimuli": [stim.__dict__ for stim in stimuli],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {summary_path}")
    print(f"wrote {gate_path}")
    print(f"wrote {class_path}")
    print(f"wrote {manifest_path}")
    print(f"gate_g6_pass={manifest['gate_g6_pass']}")
    print(class_df["classification"].value_counts().to_string())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M1.6 addressing-vs-induction discriminator for NoPE-GPT-Small")
    p.add_argument("--model", default="nope-gpt-small", choices=sorted(MODEL_IDS))
    p.add_argument("--output-dir", default="outputs/k_address_space_m16_nope_gpt_small")
    p.add_argument("--device", default="cpu")
    p.add_argument("--revision", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repetitions", type=int, default=4, help="Continuation-marker repetitions per stimulus; also marker vocabulary size.")
    p.add_argument("--limit-stimuli", type=int, default=None)
    p.add_argument("--limit-layers", type=int, default=None)
    p.add_argument("--limit-heads", type=int, default=None)
    p.add_argument("--progress-every", type=int, default=20)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
