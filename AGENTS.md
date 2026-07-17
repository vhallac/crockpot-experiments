# Transformer Mechanistic Experiments — Agent Notes

## Repository purpose

This repository is an experiment collection for transformer-mechanistic probes, not a single-purpose dead-key project. Keep experiment groups independently understandable, but reuse shared loading, execution, and RunPod infrastructure where practical.

Current experiment groups:

- `experiments/dead-keys/` — original dead-key census and follow-on RoPE/intervention phases. See `experiments/dead-keys/README.md` and `spec.md`.
- `experiments/queryability/` — paired `W_Q^T W_K` query/key geometry. See `experiments/queryability/README.md`.
- `experiments/k-address-space/` — pre-registered K-space/content-address experiment. See `experiments/k-address-space/README.md` and `spec.md`.

## Experiment documentation standard

Every experiment group must have a README or equivalent preamble that states:

1. what it measures;
2. what signal or outcome it expects;
3. how to run a local smoke test;
4. how generated outputs are handled.

Reproducible runs record their lab notebook entries in `experiments/<experiment-id>/NOTEBOOK.md`, newest entry first. The notebook is a durable git artifact: commit the pre-run entry before the run and the completed entry after analysis, per the reproducible-research skill.

Experiment implementations live under `experiments/<experiment-id>/`. If an experiment contains an importable package, place that package inside the experiment directory and add the experiment package roots to `PYTHONPATH` for direct module execution.

Each experiment's own `spec.md`/README is authoritative for that experiment. Do not treat `experiments/dead-keys/spec.md` as global guidance outside the dead-key experiment group.

## Reproducible experiment methodology

Use the project skill `.pi/skills/reproducible-research` as the procedural source of truth for full experiment runs: pre-run lab notebook entries, pre-run commits, external output publication, analysis, and completion evidence. Sections below supply the project-specific parameters that skill defers to (notebook path, publication medium, tag convention, scratch area).

Per-run throw-away checklists and other disposable reports belong under `temp/`, which is the repository's scratch convention and is ignored by git.

## Cross-experiment caching via network volume

Python CUDA libraries (PyTorch, transformers, etc.) and model parameters are
large downloads (5+ GB). They MUST live on the RunPod network volume so new
pods and new experiments reuse them without re-downloading.

### What lives on the network volume

| Resource | Network-volume path | Notes |
|----------|-------------------|-------|
| CUDA uv venv | `$DEAD_KEYS_CUDA_VENV` (default: `<cache-root>/venvs/cuda`) | Single shared venv for all experiments; installed once per requirements change |
| HuggingFace models | `$HF_HOME` (default: `<cache-root>/huggingface`) | Downloaded automatically by transformers on first use |
| HuggingFace datasets | `$HF_HOME` | Downloaded automatically by datasets on first use |
| Torch extensions | `$TORCH_HOME` (default: `<cache-root>/torch`) | Compiled Triton kernels etc. |
| Triton cache | `$TRITON_CACHE_DIR` (default: `<cache-root>/triton`) | Triton compiled kernels |
| uv/pip caches | `$UV_CACHE_DIR`, `$PIP_CACHE_DIR` | Package wheels |

### Per-pod initialization

When a new pod starts (even one from a saved template), the network volume is
mounted at `/workspace` but the cache environment variables are not yet set.
Run the setup script once:

```bash
cd /workspace/crockpot-experiments  # or /workspace/dead-keys-census (legacy)
./scripts/runpod-persistent-cache-setup
. ~/.dead-keys-census-runpod-env
```

After this, `./scripts/cuda-run` automatically uses the shared network-volume
venv. The first `cuda-run` invocation in a new pod installs packages into that
venv if needed; subsequent invocations (and invocations from other experiments)
skip the install step because the venv already exists.

### New experiments reuse the shared venv

Each experiment under `experiments/<id>/` uses the same `./scripts/cuda-run`
wrapper, which points to the same `DEAD_KEYS_CUDA_VENV` on the network volume.
If an experiment needs additional Python packages beyond what is in
`requirements-runpod-cuda.txt`, add them there — they are installed once and
shared across all experiments.

### Skipping reinstallation

Set `DEAD_KEYS_CUDA_SKIP_INSTALL=1` to skip the `uv pip install` step if you
know the venv is current:

```bash
DEAD_KEYS_CUDA_SKIP_INSTALL=1 ./scripts/cuda-run -m experiment.script --model gpt2
```

## Environment and package management

Use `uv` / `uvx`; do not create ad-hoc `venv` directories and do not install packages as root.

Typical local setup:

```bash
uv sync
PYTHONPATH=experiments/dead-keys uv run python -m deadkeys.scripts.census --model gpt2 --limit-layers 1 --limit-heads 1 --samples 1024 --output-dir outputs/deadkeys_smoke_gpt2
PYTHONPATH=experiments/dead-keys:experiments/queryability uv run python -m queryability.scripts.weights --model gpt2 --limit-layers 1 --limit-heads 1 --output-dir outputs/queryability_smoke_gpt2
```

The checked-in `pyproject.toml` / `uv.lock` are currently pinned for ROCm PyTorch. On NixOS, generic `uv` Python binaries may not execute; use project wrappers where documented.

## Generated results and paper artifacts

Generated experiment outputs are valuable but can become huge. Default policy:

- Write raw/generated outputs under `outputs/<experiment-id-or-run-id>/`.
- Keep `outputs/` ignored by git.
- Do not commit raw `.npz`, parquet, CSV, logs, model dumps, or full extraction directories unless explicitly curated.
- Commit small, durable artifacts only when useful: README findings, run manifests, compact summary tables, plots selected for a paper, and scripts that reproduce the result.
- For paper-bound results, create a small curated directory such as `paper-artifacts/<experiment-id>/` or an experiment-local `artifacts/` directory, with provenance pointing back to the external/raw run location.
- If raw results need preservation, store them outside git in external storage and commit a manifest with paths, hashes, model revisions, command lines, and dates.
- For reproducible-research publication in this repository, use GitHub Release assets as the default external publication medium for packaged outputs, unless the user chooses another store for a run.

GitHub Release conventions for reproducible runs:

- Tag: `run/<experiment-id>/<YYYYMMDD>`; append `-2`, `-3`, … for additional same-day runs of the same experiment.
- Create releases with the `gh` CLI (for example `gh release create <tag> <assets...> --notes-file <notes>`); mark them as pre-release if the analysis is not yet complete.
- Release notes must reference the pre-run commit SHA, the final commit SHA once available, and the run command.
- Upload a `SHA256SUMS` file alongside the packaged assets. The same hashes go into the committed run manifest, so the release and the git-side manifest can verify each other.
- Post-publication verification: list assets (`gh release view <tag> --json assets`) and re-download at least the checksum file to confirm integrity.

## Local and host-specific wrappers

Prefer wrappers over hand-built environments:

```bash
PYTHONPATH=experiments/dead-keys ./scripts/rocm-run python -m deadkeys.scripts.census --model gpt2 --limit-layers 1 --limit-heads 1 --samples 1024
PYTHONPATH=experiments/dead-keys ./scripts/rocm-python -m deadkeys.scripts.census --model gpt2 --limit-layers 1 --limit-heads 1 --samples 1024
PYTHONPATH=experiments/dead-keys:experiments/queryability ./scripts/nix-cpu-run -m queryability.scripts.weights --model gpt2 --limit-layers 1 --limit-heads 1 --output-dir outputs/queryability_smoke_gpt2
```

Use smoke-test limits before full-model runs:

- `--limit-layers 1`
- `--limit-heads 1`
- small samples/doc limits appropriate to the experiment

Verify outputs externally before reporting success, for example:

```bash
test -d outputs/<run-id>
find outputs/<run-id> -maxdepth 1 -type f | sort
```

## RunPod NVIDIA/CUDA environment

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
- Reusable private RunPod template: `dead-keys-census-cuda` (id `1zpm2v05rn`)

Ad-hoc replacement pod preferences when the latest pod is unavailable:

- GPU: `RTX A5000` or `L4`
- CPU: at most 6 cores
- Container image: `ghcr.io/vhallac/dead-keys-census-runpod:latest` (public GHCR image for this project)
- RAM: around 60 GB
- Disk: around 30 GB
- Prefer the same datacenter / compatible GPU when possible so the existing network volume is available.
- Attach the network volume at pod creation/deployment time; do not expect to attach one later to an existing pod.

Project paths and cache parameters for RunPod remain legacy-named until the repo/image rename is deliberately performed:

- Repository path in pods: `/workspace/dead-keys-census`
- Persistent cache root: `/workspace/dead-keys-census-cache`
- Setup script: `./scripts/runpod-persistent-cache-setup`
- Cache env file: `~/.dead-keys-census-runpod-env`
- CUDA wrappers: `scripts/cuda-run`, `scripts/cuda-python`
- CUDA venv override: `DEAD_KEYS_CUDA_VENV=/path/to/venv`
- Existing compatible CUDA venv observed on `dead-weight-migration-2`: `/venv-deadkeys`
- To reuse that venv without reinstalling heavyweight wheels: `DEAD_KEYS_CUDA_VENV=/venv-deadkeys DEAD_KEYS_CUDA_SKIP_INSTALL=1 ./scripts/cuda-run ...`
- The shared CUDA venv on the network volume is the primary venv for all experiments; avoid creating per-experiment venvs.
- A failed install into `/workspace/dead-keys-census-cache/uv` hit `Quota exceeded`; prefer the existing venv above unless intentionally rebuilding caches.

CUDA sanity check inside a RunPod host after following the RunPod skill:

```bash
./scripts/cuda-run - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))
x = torch.ones(16, device='cuda') + 2
torch.cuda.synchronize()
print(x.device, x[:3].cpu().tolist())
PY
```

Discovery note: on 2026-07-10, `podResume` for the original `dead-weight` pod failed with `There are not enough free GPUs on the host machine to start this pod`; the replacement `dead-weight-migration` pod was created and configured instead.

## ROCm PyTorch on AMD Radeon 890M

This project is pinned to ROCm PyTorch in `pyproject.toml` / `uv.lock`:

- `torch==2.9.1+rocm6.3`
- `pytorch-triton-rocm==3.5.1`
- Python constrained to `>=3.10,<3.12` because the ROCm companion wheels are not available for newer Python versions here.

On the AMD Radeon 890M, PyTorch detects the GPU but common kernels fail unless this compatibility override is set. PyTorch ROCm attention also needs the experimental AOTriton kernels enabled to avoid slow/fallback attention behavior:

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
```

Prefer the project wrappers, which set both variables automatically.

## Development hygiene

- Keep code/test or code/smoke updates together for each experiment change.
- Make surgical changes; do not rename packages, RunPod images, or remote paths unless the task is specifically an infra rename.
- Preserve legacy names in operational docs until the actual image/template/path migration is complete.
- Before full runs, execute the experiment-specific sanity gates from its README/spec.
- Before claiming completion, verify with real commands and report the evidence.
