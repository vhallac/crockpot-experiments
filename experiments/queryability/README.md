# Queryability experiments

## What this measures

This experiment family studies the paired geometry of a transformer's query and key maps.

For one attention head, ignoring RoPE and attention scaling:

```text
q = W_Q r_q
k = W_K r_k
score(q, k) = q^T k = r_q^T W_Q^T W_K r_k
```

The central object is therefore the paired bilinear map:

```text
M = W_Q^T W_K
```

The question is not only whether `W_Q` or `W_K` have small singular directions individually. The direct queryability question is whether there are key-producing residual directions `r_k` for which:

```text
|| W_Q^T W_K r_k || ≈ 0
```

If so, those directions are weakly visible to all queries that can be produced through `W_Q`, in the weights-only linear sense.

## Expected signal

The expected signal is a per-head paired bilinear map with sharply bounded interaction rank and possible near-null key-producing residual directions. For GPT-2 layer 0 head 0, the smoke result already showed `paired_rank=64` and `paired_erank~57`, consistent with a full-rank head whose paired interaction dimension is bounded by the head width.

## Scope: raw pre-RoPE weights-only paired SVD

GPT-2 is the simplest starting model because it has no RoPE. Pythia and Qwen can also be run through the same raw calculation, but the result must be interpreted as **pre-RoPE projection-weight geometry**:

- Pythia has partial RoPE, so only part of each head is subsequently rotated at runtime.
- Qwen3 has full RoPE and QK norm; this script does not fold the non-linear `q_norm` / `k_norm` into the SVD.
- RoPE-aware or activation-conditioned queryability needs a separate experiment; do not infer runtime long-range behavior directly from this raw SVD.

For RoPE models the CLI intentionally requires `--allow-rope-raw` so a remote run records that interpretation instead of silently producing easy-to-misread data.

The experiment computes, per layer/head:

1. the per-head `W_Q` and `W_K` slices using the existing `deadkeys` loading code;
2. `M = W_Q.T @ W_K`;
3. the singular values of `W_Q`, `W_K`, and `M`;
4. simple summary metrics such as rank, effective rank, stable rank, condition number, and near-zero singular-value fractions.

This is a weights-only experiment. It can say that a direction is geometrically weak or impossible for the paired maps. It cannot yet say whether real activations visit that direction during inference.

## How to execute

### Smoke command

With the project `uv` environment (installs the heavy ROCm/CUDA torch wheel):

```bash
PYTHONPATH=experiments/dead-keys:experiments/queryability uv run python -m queryability.scripts.weights \
  --model gpt2 \
  --limit-layers 1 \
  --limit-heads 1 \
  --output-dir outputs/queryability_smoke
```

On the local NixOS host where the ROCm wheel install is blocked by disk
space, use the CPU-only nix-shell wrapper instead (this experiment is
weights-only and needs no GPU):

```bash
PYTHONPATH=experiments/dead-keys:experiments/queryability ./scripts/nix-cpu-run -m queryability.scripts.weights \
  --model gpt2 \
  --limit-layers 1 \
  --limit-heads 1 \
  --output-dir outputs/queryability_smoke
```

Expected outputs:

```text
outputs/queryability_smoke/queryability_gpt2.csv
outputs/queryability_smoke/queryability_spectra_gpt2.npz
outputs/queryability_smoke/queryability_manifest_gpt2.json
```

The paired spectrum `S_QTK` has `d_model` singular values, but since
`rank(W_Q^T W_K) <= d_head`, only `d_head` of them are nonzero; the rest
are ~0. For GPT-2 layer 0 head 0 this gives `paired_rank=64` (= d_head)
and `paired_erank~57`, consistent with a full-rank head whose paired
interaction dimension is bounded by the head width.

## Result policy

Write generated queryability outputs under `outputs/`. Do not commit full `.npz`, CSV, or manifest result sets by default unless they are intentionally curated as a small paper artifact.

## Later RunPod raw-SVD commands

Do not use these locally unless the weights are already cached; they download larger models. On a RunPod CUDA host, use the project CUDA wrapper so PyTorch/CUDA installs and Hugging Face weights live in the persistent cache configured by `scripts/runpod-persistent-cache-setup`:

```bash
DEAD_KEYS_CUDA_VENV=/venv-deadkeys DEAD_KEYS_CUDA_SKIP_INSTALL=1 \
  PYTHONPATH=experiments/dead-keys:experiments/queryability ./scripts/cuda-run -m queryability.scripts.weights \
    --model pythia410 \
    --limit-layers 1 \
    --limit-heads 1 \
    --device cuda \
    --allow-rope-raw \
    --output-dir outputs/queryability_raw_pythia410_smoke

DEAD_KEYS_CUDA_VENV=/venv-deadkeys DEAD_KEYS_CUDA_SKIP_INSTALL=1 \
  PYTHONPATH=experiments/dead-keys:experiments/queryability ./scripts/cuda-run -m queryability.scripts.weights \
    --model qwen3 \
    --limit-layers 1 \
    --limit-heads 1 \
    --device cuda \
    --allow-rope-raw \
    --output-dir outputs/queryability_raw_qwen3_smoke
```

For full runs, remove the layer/head limits only after the one-head smoke writes CSV, NPZ, and manifest files and the console confirms `requested_device=cuda` with a real CUDA device.
