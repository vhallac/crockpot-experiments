from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from deadkeys.common.loading import load_model, set_qwen_rotary_identity
from kaddress.scripts.address_purity import _capture_qwen_k


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def verify_grs11(
    *,
    model: str = "qwen3-dropped",
    device: str = "cuda",
    text: str = "G-RS1.1 verifies that dropped Qwen rotary embeddings are identity.",
    atol: float = 1e-5,
) -> dict[str, Any]:
    """Verify RS1 gate G-RS1.1 for a dropped-Qwen model state.

    The positive check runs the normal loaded forward, which should have the
    centralized identity rotary source active.  The falsifiability check flips
    the same model back to true Qwen RoPE and requires pre/post K identity to
    fail on the same input.
    """
    lm = load_model(model, device=device)
    if model not in {"qwen3-dropped", "qwen3-droped"}:
        raise ValueError("G-RS1.1 helper currently expects --model qwen3-dropped or qwen3-droped")
    encoded = lm.tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(next(lm.model.parameters()).device)

    with torch.no_grad():
        _raw, k_pre_identity, k_post_identity = _capture_qwen_k(lm, input_ids, None)
    identity_max_abs = _max_abs(k_pre_identity, k_post_identity)
    identity_pass = bool(torch.allclose(k_pre_identity, k_post_identity, atol=atol, rtol=0.0))

    set_qwen_rotary_identity(lm.model, enabled=False)
    try:
        with torch.no_grad():
            _raw, k_pre_true, k_post_true = _capture_qwen_k(lm, input_ids, None)
    finally:
        set_qwen_rotary_identity(lm.model, enabled=True)
    true_rope_max_abs = _max_abs(k_pre_true, k_post_true)
    perturbation_fails = not bool(torch.allclose(k_pre_true, k_post_true, atol=atol, rtol=0.0))

    passed = identity_pass and perturbation_fails
    return {
        "gate": "G-RS1.1",
        "model": model,
        "device": str(next(lm.model.parameters()).device),
        "tokens": int(input_ids.shape[1]),
        "atol": atol,
        "identity_max_abs": identity_max_abs,
        "identity_pass": identity_pass,
        "true_rope_max_abs": true_rope_max_abs,
        "perturbation_fails": perturbation_fails,
        "pass": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify RS1 G-RS1.1 dropped-Qwen rotary identity gate")
    parser.add_argument("--model", default="qwen3-dropped", choices=["qwen3-dropped", "qwen3-droped"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--text", default="G-RS1.1 verifies that dropped Qwen rotary embeddings are identity.")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = verify_grs11(model=args.model, device=args.device, text=args.text, atol=args.atol)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    print(payload)
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
