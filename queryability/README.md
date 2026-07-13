# Queryability experiments

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

## First target: GPT-2 weights-only paired SVD

GPT-2 is the simplest starting model because it has no RoPE. The first experiment computes, per layer/head:

1. the per-head `W_Q` and `W_K` slices using the existing `deadkeys` loading code;
2. `M = W_Q.T @ W_K`;
3. the singular values of `W_Q`, `W_K`, and `M`;
4. simple summary metrics such as rank, effective rank, stable rank, condition number, and near-zero singular-value fractions.

This is a weights-only experiment. It can say that a direction is geometrically weak or impossible for the paired maps. It cannot yet say whether real activations visit that direction during inference.

## Smoke command

With the project `uv` environment (installs the heavy ROCm/CUDA torch wheel):

```bash
uv run python -m queryability.scripts.weights \
  --model gpt2 \
  --limit-layers 1 \
  --limit-heads 1 \
  --output-dir outputs/queryability_smoke
```

On the local NixOS host where the ROCm wheel install is blocked by disk
space, use the CPU-only nix-shell wrapper instead (this experiment is
weights-only and needs no GPU):

```bash
./scripts/nix-cpu-run -m queryability.scripts.weights \
  --model gpt2 \
  --limit-layers 1 \
  --limit-heads 1 \
  --output-dir outputs/queryability_smoke
```

Expected outputs:

```text
outputs/queryability_smoke/queryability_gpt2.csv
outputs/queryability_smoke/queryability_spectra_gpt2.npz
```

The paired spectrum `S_QTK` has `d_model` singular values, but since
`rank(W_Q^T W_K) <= d_head`, only `d_head` of them are nonzero; the rest
are ~0. For GPT-2 layer 0 head 0 this gives `paired_rank=64` (= d_head)
and `paired_erank~57`, consistent with a full-rank head whose paired
interaction dimension is bounded by the head width.
