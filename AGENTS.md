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
- `requirements-runpod-cuda.txt` — CUDA dependency set for NVIDIA RunPod hosts.
- `scripts/cuda-run`, `scripts/cuda-python` — dedicated CUDA virtualenv wrappers for RunPod; they do not alter the local ROCm environment.
- `scripts/runpod-persistent-cache-setup` — in-pod setup script that moves heavyweight model/package caches to the RunPod network volume.

## Environment

Use `uv` / `uvx`; do not install packages as root.

Typical commands:

```bash
uv sync
uv run python -m phase1.scripts.census --model gpt2 --limit-layers 1 --limit-heads 1 --samples 1024
uv run python -m phase1.scripts.plots --input outputs/census_gpt2.parquet --model gpt2
```

### RunPod `dead-weight` / `dead-weight-migration` NVIDIA/CUDA environment

Non-secret RunPod metadata discovered 2026-07-10:

- Original pod name: `dead-weight`
- Original pod id: `6mwc5q4jwwcgw9`
- Migration pod name: `dead-weight-migration`
- Migration pod id: `lszgheen2t7qor`
- Migration pod SSH user discovered from RunPod SSH command: `lszgheen2t7qor-64410ed2`
- GPU display name from RunPod API: `RTX A4500`
- Machine type: secure cloud GPU pod
- Current desired status after investigation: keep `EXITED`
- Local RunPod credential env var: `RUNPOD_API_KEY` (do not print or commit value)
- In-pod GitHub credential env var: `RUNPOD_SECRET_GITHUB_TOKEN` (do not print or commit value)
- Local `~/.ssh/config`: no `dead-weight`, `runpod`, or pod-id host entry was present during discovery.

RunPod migration / startup procedure:

1. Keep `dead-weight-migration` halted when idle. Current verified halted state: `desiredStatus: EXITED`.
2. If the original pod cannot resume because its host has no free GPU capacity, create/deploy a replacement pod in RunPod with the same network volume attached at creation time. Network volumes for pods are effectively chosen at deployment time; do not expect to attach one later to an existing pod.
3. Prefer same datacenter / compatible GPU when possible so the existing network volume is available. Name the replacement clearly, e.g. `dead-weight-migration`.
4. After deployment, query the pod by name via GraphQL and record its new pod id, GPU, image, and runtime SSH ports here. The current migration pod id is `lszgheen2t7qor`.
5. Before installing dependencies or downloading models, run `./scripts/runpod-persistent-cache-setup` inside the pod so Hugging Face, torch, pip, uv, Triton, and CUDA virtualenv data live under `/workspace/dead-keys-census-cache` on the network volume.
6. For cross-datacenter moves, RunPod's documented path is two running pods and `rsync` between their `/workspace` mounts; this is separate from simply deploying a replacement pod against the same existing volume.

RunPod SSH after migration:

1. First query runtime ports from GraphQL; while the pod is `EXITED`, `runtime` is `null`, so IP/ports are unavailable until it is running.
2. Prefer the public TCP SSH endpoint from `runtime.ports`: `ssh -p <publicPort> root@<public-ip> -i ~/.ssh/id_ed25519`. During discovery this was `root@213.173.98.71 -p 19159`, but runtime IP/port can change after migration/restart.
3. If public TCP SSH auth fails or is unavailable, use the RunPod web UI Connect/SSH command. It shows the complete random proxy user, e.g. `ssh <pod-id>-<suffix>@ssh.runpod.io -i ~/.ssh/id_ed25519`.
4. The migration proxy user discovered for `lszgheen2t7qor` was `lszgheen2t7qor-64410ed2`.
5. The pod id alone is available from GraphQL, but the extra random `ssh.runpod.io` suffix was not present in the basic `myself { pods { ... runtime { ports ... } } }` response. Treat the UI Connect command as the reliable source for that suffix.
6. The `ssh.runpod.io` proxy requires a PTY; for scripted commands, pipe commands into `ssh -tt <user>@ssh.runpod.io -i ~/.ssh/id_ed25519` rather than using plain non-interactive `ssh <host> command`.

The local `pyproject.toml` / `uv.lock` remain pinned for ROCm.

Before installing dependencies or downloading models in the pod, make heavyweight downloads persistent on the network volume:

```bash
# Inside the pod. Auto-detects common RunPod network volume mounts such as /workspace.
./scripts/runpod-persistent-cache-setup

# If auto-detection fails, set the network-volume path explicitly:
DEAD_KEYS_PERSISTENT_CACHE_ROOT=/workspace/dead-keys-census-cache ./scripts/runpod-persistent-cache-setup

# Load the generated cache environment in the current shell if needed:
. ~/.dead-keys-census-runpod-env
```

This configures:

- `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE` for model downloads.
- `TORCH_HOME` for PyTorch hub/checkpoint cache.
- `PIP_CACHE_DIR` for Python wheels, including large PyTorch/CUDA wheels.
- `UV_CACHE_DIR` for uv package cache.
- `TRITON_CACHE_DIR` for Triton kernels.
- `DEAD_KEYS_CUDA_VENV` under the persistent cache root, so the CUDA virtualenv itself survives pod restarts/rebuilds.

For NVIDIA RunPod hosts, use the dedicated CUDA environment files instead:

```bash
./scripts/cuda-run - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))
x = torch.ones(16, device='cuda') + 2
torch.cuda.synchronize()
print(x.device, x[:3].cpu().tolist())
PY

./scripts/cuda-run -m phase1.scripts.census --model gpt2 --limit-layers 1 --limit-heads 1 --samples 1024
```

CUDA wrapper details:

- `scripts/cuda-run` sources `~/.dead-keys-census-runpod-env` if present, then creates/uses `DEAD_KEYS_CUDA_VENV` if set; otherwise it falls back to `.venv-cuda`.
- Override the venv path with `DEAD_KEYS_CUDA_VENV=/path/to/venv`.
- Override Python with `DEAD_KEYS_CUDA_PYTHON=python3.11` if the pod has multiple Python versions.
- The wrapper avoids changing the ROCm `uv` environment used on the local AMD machine.

RunPod API helpers:

```bash
# Query non-secret pod state for original and migration pods
curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" https://api.runpod.io/graphql \
  -H 'content-type: application/json' \
  --data-binary '{"query":"query { myself { pods { id name desiredStatus runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } } machine { gpuDisplayName cpuCount memoryTotal secureCloud machineType } } } }"}' \
  | jq '.data.myself.pods[] | select(.name=="dead-weight" or .name=="dead-weight-migration")'

# Resume migration pod
curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" https://api.runpod.io/graphql \
  -H 'content-type: application/json' \
  --data-binary '{"query":"mutation { podResume(input: {podId: \"lszgheen2t7qor\"}) { id name desiredStatus } }"}'

# Stop migration pod when finished; verify desiredStatus is EXITED afterward
curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" https://api.runpod.io/graphql \
  -H 'content-type: application/json' \
  --data-binary '{"query":"mutation { podStop(input: {podId: \"lszgheen2t7qor\"}) { id name desiredStatus } }"}'
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
