# Repro checklist — k-address-space M1.5 v1.1 qwen3 CUDA — 2026-07-22

## Pre-run intent

Run `experiments/k-address-space` M1.5 v1.1 for `qwen3` (`Qwen/Qwen3-0.6B`) on RunPod CUDA. This adjudicates the Qwen3 side of the RoPE stamped-vs-computed comparison, especially P1.5.c and P1.5.e.

Reviewer feedback incorporated for this run: do not stop analysis at ridge `R²` and `position_fraction`. Pull dimensionality/projector columns by default from the qwen3 outputs as well:

- `pca_components_90pct`
- `pca_residual_variance_fraction_90pct`
- `r2_after_position_pc_projection`
- `token_identity_acc_before`
- `token_identity_acc_after`
- per-slot versus `AGGREGATE` rows, to avoid the aggregate-vs-slot trap found in the Pythia post-writeup.

## Static GPU/path audit

The executed command is `kaddress.scripts.position_content`. The hot per-slot CUDA path is `_analyse_matrix_torch`, which keeps ridge CV, batched shuffled nulls, SVD/PCA, projection, and nearest-centroid diagnostics on torch CUDA tensors. The known Pythia CPU-bound failure path (`.cpu().numpy()` before NumPy ridge/SVD loops) has been removed for CUDA rows. Remaining CPU transfers are for small diagnostics/final serialization (`basis` arrays and final CSV/NPZ writes), which are outside the dominant compute path.

Progress is emitted with `--progress-every`, and remote log monitoring plus GPU utilization checks are required during the run.

## Planned command

```bash
PYTHONPATH=experiments/dead-keys:experiments/k-address-space python -m kaddress.scripts.position_content \
  --model qwen3 \
  --device cuda \
  --families A,B,C \
  --segment-lengths 4,7,12 \
  --repetitions 256 \
  --max-length 3072 \
  --progress-every 1000 \
  --output-dir outputs/k_address_space_m15_v11_qwen3_cuda_20260722
```

## Validation gates

- CUDA available and environment recorded in manifest.
- Non-empty summary CSV, gates CSV, manifest JSON, projectors NPZ.
- G1 architectural zero passes for qwen3 layer-0 `k_pre`.
- G2 architectural one passes for qwen3 layer-0 `k_post`.
- G2 perturbation check can fail.
- `shuffle_null_ok` reported from upper-tail gate.
- Post-run analysis reports dimensionality/projector columns for slot and aggregate rows separately.

## Post-run status

Completed on RunPod NVIDIA L4. Outputs copied locally and packaged under `outputs/k_address_space_m15_v11_qwen3_cuda_20260722*`. Reviewer-requested dimensionality/projector columns were extracted in the notebook entry.
