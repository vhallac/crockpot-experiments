# Dead Keys Census — Project Notes

## Layout

- `dead_key_census_spec.md` — source specification.
- `deadkeys/` — Phase 1 weights-only census implementation.
  - `common/loading.py` — HuggingFace model loading, per-architecture Q/K slicing, sanity checks.
  - `common/spectra.py` — SVD, effective rank, dead-fraction, random baseline math.
  - `common/rope.py` — RoPE band partition helpers.
  - `scripts/census.py` — CLI for Phase 1 census.
  - `scripts/plots.py` — CLI for Phase 1 plots from parquet/CSV output.
- `outputs/` — generated census tables, spectra `.npz`, and plots.
- `requirements-runpod-cuda.txt` — CUDA dependency set for NVIDIA RunPod hosts.
- `scripts/cuda-run`, `scripts/cuda-python` — dedicated CUDA virtualenv wrappers for RunPod; they do not alter the local ROCm environment.
- `scripts/runpod-persistent-cache-setup` — in-pod setup script that moves heavyweight model/package caches to the RunPod network volume.

## Environment

Use `uv` / `uvx`; do not install packages as root.

Typical commands:

```bash
uv sync
uv run python -m deadkeys.scripts.census --model gpt2 --limit-layers 1 --limit-heads 1 --samples 1024
uv run python -m deadkeys.scripts.plots --input outputs/census_gpt2.parquet --model gpt2
```

### RunPod NVIDIA/CUDA environment

Use the project skill `.pi/skills/runpod-usage` as the procedural source of truth for RunPod lifecycle, SSH access, pod creation, network-volume handling, persistent-cache setup, CUDA smoke tests, workload stopping, and cost hygiene.

Keep this section limited to project-specific parameters and historical facts.

Non-secret RunPod metadata discovered for this project:

- Original pod name: `dead-weight`
- Original pod id: `6mwc5q4jwwcgw9`
- First migration pod name: `dead-weight-migration`
- First migration pod id: `lszgheen2t7qor`
- First migration pod GPU from RunPod API: `RTX A4500`
- First migration pod machine type: secure cloud GPU pod
- Additional migration pod name: `dead-weight-migration-2`
- Additional migration pod id: `6r332ke14n1lpx`
- Additional migration pod GPU: NVIDIA L4
- Current desired status for idle project pods: keep `EXITED`
- Local RunPod credential env var: `RUNPOD_API_KEY` (do not print or commit value)
- In-pod GitHub credential env var: `RUNPOD_SECRET_GITHUB_TOKEN` (do not print or commit value)
- Local `~/.ssh/config`: no `dead-weight`, `runpod`, or pod-id host entry was present during discovery.
- Reusable private RunPod template: `dead-keys-census-cuda` (id `1zpm2v05rn`).

Ad-hoc replacement pod preferences when the latest pod is unavailable:

- GPU: `RTX A5000` or `L4`
- CPU: at most 6 cores
- Container image: `ghcr.io/vhallac/dead-keys-census-runpod:latest` (public GHCR image for this project)
- RAM: around 60 GB
- Disk: around 30 GB
- Prefer the same datacenter / compatible GPU when possible so the existing network volume is available.
- Attach the network volume at pod creation/deployment time; do not expect to attach one later to an existing pod.

Project paths and cache parameters for RunPod:

- Repository path in pods: `/workspace/dead-keys-census`
- Persistent cache root: `/workspace/dead-keys-census-cache`
- Setup script: `./scripts/runpod-persistent-cache-setup`
- Cache env file: `~/.dead-keys-census-runpod-env`
- CUDA wrappers: `scripts/cuda-run`, `scripts/cuda-python`
- CUDA venv override: `DEAD_KEYS_CUDA_VENV=/path/to/venv`
- Existing compatible CUDA venv observed on `dead-weight-migration-2`: `/venv-deadkeys`
- To reuse that venv without reinstalling heavyweight wheels: `DEAD_KEYS_CUDA_VENV=/venv-deadkeys DEAD_KEYS_CUDA_SKIP_INSTALL=1 ./scripts/cuda-run ...`
- A failed install into `/workspace/dead-keys-census-cache/uv` hit `Quota exceeded`; prefer the existing venv above unless intentionally rebuilding caches.

RunPod project command examples:

```bash
# CUDA sanity check inside a RunPod host, after following the runpod-usage skill.
./scripts/cuda-run - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))
x = torch.ones(16, device='cuda') + 2
torch.cuda.synchronize()
print(x.device, x[:3].cpu().tolist())
PY

# Phase 1 smoke only; do not use full defaults until this shows CUDA use.
./scripts/cuda-run -m deadkeys.scripts.census \
  --model pythia410 --limit-layers 1 --limit-heads 1 \
  --samples 256 --misalign-rotations 2 --device cuda \
  --output-dir outputs/smoke_pythia_gpu

# Phase 1.5 Pythia certificate/null smoke only. PPL/truncated-attention eval is
# currently implemented for GPT-2 only, so Pythia smoke uses --skip-ppl.
./scripts/cuda-run -m deadkeys.scripts.phase1_5 \
  --model pythia410 --limit-layers 1 --limit-heads 1 \
  --eval-tokens 1024 --calibration-tokens 256 --observed-tokens 128 \
  --null-samples 2 --null-dead-samples 128 --null-depths 5 \
  --allow-smoke-under-200k --skip-ppl --device cuda \
  --output-dir outputs/phase1_5_smoke_pythia_gpu
```

Discovery note: on 2026-07-10, `podResume` for the original `dead-weight` pod failed with `There are not enough free GPUs on the host machine to start this pod`; the replacement `dead-weight-migration` pod was created and configured instead.

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
./scripts/rocm-run python -m deadkeys.scripts.census --model gpt2 --limit-layers 1 --limit-heads 1 --samples 1024
./scripts/rocm-python -m deadkeys.scripts.census --model gpt2 --limit-layers 1 --limit-heads 1 --samples 1024
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
- Keep all Phase 1 code under `deadkeys/` unless explicitly asked otherwise.
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
