# Dead Keys Census — Project Notes

## Layout

- `dead_key_census_spec.md` — source specification.
- `phase1/` — Phase 1 weights-only census implementation.
  - `common/loading.py` — HuggingFace model loading, per-architecture Q/K slicing, sanity checks.
  - `common/spectra.py` — SVD, effective rank, dead-fraction, random baseline math.
  - `common/rope.py` — RoPE band partition helpers.
  - `scripts/census.py` — CLI for Phase 1 census.
  - `scripts/plots.py` — CLI for Phase 1 plots from parquet/CSV output.
- `outputs/` — generated census tables, spectra `.npz`, and plots.

## Environment

Use `uv` / `uvx`; do not install packages as root.

Typical commands:

```bash
uv sync
uv run python -m phase1.scripts.census --model gpt2 --limit-layers 1 --limit-heads 1 --samples 1024
uv run python -m phase1.scripts.plots --input outputs/census_gpt2.parquet --model gpt2
```

### ROCm PyTorch on AMD Radeon 890M

This project is pinned to ROCm PyTorch in `pyproject.toml` / `uv.lock`:

- `torch==2.9.1+rocm6.3`
- `pytorch-triton-rocm==3.5.1`
- Python constrained to `>=3.10,<3.12` because the ROCm companion wheels are not available for newer Python versions here.

On the AMD Radeon 890M, PyTorch detects the GPU but common kernels fail unless this compatibility override is set. PyTorch ROCm attention also needs the experimental AOTriton kernels enabled to avoid slow/fallback attention behavior:

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
```

Prefer the project wrappers, which set both variables automatically:

```bash
./scripts/rocm-run python -m phase1.scripts.census --model gpt2 --limit-layers 1 --limit-heads 1 --samples 1024
./scripts/rocm-python -m phase1.scripts.census --model gpt2 --limit-layers 1 --limit-heads 1 --samples 1024
```

ROCm verification command:

```bash
./scripts/rocm-run python - <<'PY'
import torch
print(torch.__version__, torch.version.hip, torch.cuda.is_available(), torch.cuda.get_device_name(0))
x = torch.ones(16, device='cuda') + 2
torch.cuda.synchronize()
print(x.device, x[:3].cpu().tolist())
PY
```

If parquet support is unavailable, the census script also writes CSV.

## Ways of working

- Treat `dead_key_census_spec.md` as authoritative.
- Keep all Phase 1 code under `phase1/` unless explicitly asked otherwise.
- Run the §6.4 q/k reconstruction sanity check before trusting census outputs.
- Never pool raw scores across heads.
- Use smoke-test limits before full-model runs:
  - `--limit-layers 1`
  - `--limit-heads 1`
  - smaller `--samples`, e.g. `1024`
- Verify outputs externally before reporting success, e.g.:

```bash
test -f outputs/census_gpt2.parquet || test -f outputs/census_gpt2.csv
find outputs -maxdepth 1 -type f | sort
```
