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

When drafting or completing a reproducible-run entry in `experiments/<experiment-id>/NOTEBOOK.md`, also add or update a `mini_mem` note for the current experiment. Keep it short-lived and operational: experiment id, notebook path/heading, checklist path, current process step, output/publication target, and next required action. Delete the note once the experiment is complete and the final notebook update is committed.

## Cross-experiment caching via network volume

Python CUDA libraries (PyTorch, transformers, etc.) and model parameters are
large downloads (5+ GB). They MUST live on the RunPod network volume so new
pods and new experiments reuse them without re-downloading.

### What lives on the network volume

| Resource | Network-volume path | Notes |
|----------|-------------------|-------|
| CUDA venv | `/workspace/venv` (`$DEAD_KEYS_CUDA_VENV`) | Canonical shared venv for all RunPod CUDA experiments; installed once per requirements change |
| HuggingFace models | `$HF_HOME` (default: `<cache-root>/huggingface`) | Downloaded automatically by transformers on first use |
| HuggingFace datasets | `$HF_HOME` | Downloaded automatically by datasets on first use |
| Torch extensions | `$TORCH_HOME` (default: `<cache-root>/torch`) | Compiled Triton kernels etc. |
| Triton cache | `$TRITON_CACHE_DIR` (default: `<cache-root>/triton`) | Triton compiled kernels |
| pip cache | `$PIP_CACHE_DIR` | Package wheels; may be purged after the shared CUDA venv is known-good if network-volume space is tight |
| uv cache | `$UV_CACHE_DIR` | Scratch install cache only; do not preserve large CUDA wheels here unless intentionally rebuilding dependencies |

### Per-pod initialization

When a new pod starts (even one from a saved template), the network volume is
mounted at `/workspace` but the cache environment variables are not yet set.
Run the setup script once:

```bash
cd /workspace/crockpot-experiments
./scripts/runpod-persistent-cache-setup
. ~/.crockpot-experiments-runpod-env
```

After this, set or verify:

```bash
export DEAD_KEYS_CUDA_VENV=/workspace/venv
```

`/workspace/venv` is the canonical shared network-volume venv. The first
`cuda-run` invocation in a new pod installs packages into that venv if needed;
subsequent invocations (and invocations from other experiments) skip the install
step because the venv already exists. RunPod CUDA dependency state is the shared
venv, not the uv wheel cache; if `$UV_CACHE_DIR` grows by multiple GB after
dependency installation, treat it as purgeable installer scratch unless a
rebuild is actively in progress.

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

Use the repository wrappers; do not create ad-hoc `venv` directories and do not install packages as root. Local development may use `uv` / `uvx`, but RunPod CUDA runs should go through `./scripts/cuda-run` / `./scripts/cuda-python` so they reuse the single shared `$DEAD_KEYS_CUDA_VENV` instead of oscillating between separate environments.

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
- Package reproducible-run output bundles as `.tar.gz` archives, not `.tgz` or `.zip`, so release assets use one consistent extension.
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

Before a CUDA full run, review the code path that the command will execute and record the preflight finding in the checklist/notebook. The dominant computation must actually stay on GPU, not merely model loading or extraction. Audit hot loops for GPU-to-CPU escapes such as `.cpu()`, `.numpy()`, `.tolist()`, `.item()`, `float(tensor)`, `int(tensor)`, `bool(tensor)`, Pandas/DataFrame work, `np.linalg`/`np.fft`/`sklearn`/`scipy`, or Python loops immediately after CUDA tensor extraction. If such escapes are in the dominant path, either move that work to batched `torch` on the CUDA tensor, justify why it is not dominant, or do not start the paid CUDA run. The M1.5 Pythia aborted run on 2026-07-22 is the cautionary example: keys were captured on GPU, then every slot/head/layer matrix was converted with `.cpu().numpy()` and analysed by NumPy linear algebra/permutation loops, making the run CPU-bound.

CUDA tripwire for paid/reproducible runs: after the static audit, run a bounded CUDA smoke whose command shape matches the full run closely enough to exercise the real hot path. Record progress rate, GPU utilization, CPU utilization, and an extrapolated full-run wall-clock estimate in the checklist/notebook. If extrapolated runtime exceeds the planned budget, if one CPU core is pegged while GPU utilization is low or bursty, or if progress is dominated by many tiny GPU kernels and scalar synchronizations, stop and fix/vectorize before launching the full run. Static `.item()`/`.cpu()` calls are allowed only when they are outside the hot path, after coarse batched work, or solely for final reporting/projector serialization.

Long or reproducible runs must publish progress while running. Prefer stdout progress lines with completed units, current stimulus/model slice, rate, and ETA/budget comparison. If stdout is reserved for machine-readable data, write the same information to a log file and record the log path in the notebook/checklist. Do not start an opaque long run where stopping at 10% and 99% would look the same to the operator.

Verify outputs externally before reporting success, for example:

```bash
test -d outputs/<run-id>
find outputs/<run-id> -maxdepth 1 -type f | sort
```

## Project scripts

Scripts under `scripts/` are the preferred operational entry points:

- `scripts/nix-cpu-run` — local NixOS CPU-only Python runner with torch, transformers, datasets, pandas, matplotlib, pyarrow, and accelerate. Use for weights-only/local smokes when the ROCm/uv environment would download large wheels or is unnecessary.
- `scripts/rocm-run` — local ROCm wrapper for the AMD Radeon 890M. It exports `HSA_OVERRIDE_GFX_VERSION=11.0.0` and `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`, then executes `uv run "$@"`.
- `scripts/rocm-python` — convenience wrapper that invokes `scripts/rocm-run python "$@"`.
- `scripts/runpod-persistent-cache-setup` — run inside a RunPod pod after `cd /workspace/crockpot-experiments`. It creates/updates `/workspace/crockpot-experiments-cache`, writes `~/.crockpot-experiments-runpod-env`, points HuggingFace/torch/pip/uv/Triton caches at the network volume, and sets `DEAD_KEYS_CUDA_VENV=/workspace/venv`.
- `scripts/cuda-run` — RunPod CUDA Python runner. It sources `~/.crockpot-experiments-runpod-env`, creates `/workspace/venv` if missing, installs `requirements-runpod-cuda.txt` with `uv pip` unless `DEAD_KEYS_CUDA_SKIP_INSTALL=1`, then execs the venv Python with the supplied arguments.
- `scripts/cuda-python` — convenience wrapper around `scripts/cuda-run` for Python commands/modules.
- `scripts/runpod-bring-up` — local helper to create a RunPod pod from project template id `1zpm2v05rn`, attach network volume `sndrrdckku` at `/workspace`, wait for SSH readiness, and print JSON metadata. Use for GPU experiment pods; for CPU-only cleanup pods, use the RunPod REST CPU `computeType: "CPU"` path documented in the RunPod API docs.
- `scripts/watch-experiment-run` — remote/local process monitor for long runs. It polls a required `--pattern`, prints matching process CPU/memory plus system CPU/memory/GPU utilization every interval, and exits when no matching process remains or `--timeout` is reached.

## RunPod NVIDIA/CUDA environment

Global skills `~/.pi/agent/skills/runpod-usage` (concepts, methodology, GPU selection, storage) and `~/.pi/agent/skills/runpodctl` (CLI for pod create/list/stop/delete, SSH, templates) are the procedural source of truth for all RunPod operations. Do not replicate procedural instructions here.

Keep this section limited to project-specific parameters and historical facts.

For ad-hoc replacement pod creation, use the project helper instead of hand-built
RunPod REST/CLI calls:

```bash
scripts/runpod-bring-up "NVIDIA L4"
```

The helper creates a pod from template `1zpm2v05rn`, attaches network volume
`sndrrdckku` at `/workspace`, waits for direct public-IP SSH readiness, and
prints JSON containing the pod id and SSH command. It times out after 120s by
default (`RUNPOD_BRING_UP_TIMEOUT=<seconds>` to override) and deletes the newly
created pod on timeout unless `RUNPOD_KEEP_ON_TIMEOUT=1` is set. Use this helper
before falling back to manual pod initialization/status checks.

### RunPod initialization timebox and recovered failure modes

RunPod setup/status checks are billable. For pod initialization, runtime-port
discovery, SSH readiness, `runpodctl exec` readiness, and similar manual
status checks, spend at most two minutes wall-clock per pod per phase before
changing strategy. Longer waits are acceptable only after an experiment job has
actually started and the check is monitoring job execution or GPU utilization.

Recovered failure modes to avoid:

- Use `runpodctl get pod --allfields`; this CLI does not support
  `runpodctl pod list`.
- Do not run `runpodctl update` from the Nix-installed `runpodctl`; it attempts
  to replace a binary under `/nix/store` and is not a useful recovery step.
- If a pod is `RUNNING` but `runtime.ports` remains null after the two-minute
  readiness budget, stop/remove it instead of continuing to poll. The
  the old project-specific GHCR image was observed in this state on 2026-07-20;
  for SSH-driven work, fall back to the known-good project template / official
  RunPod PyTorch image with `startSsh: true`.
- If `runpodctl exec` says `Waiting for Pod to come online...` against a pod
  with no runtime ports, stop the attempt within the same two-minute readiness
  budget.


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
- Reusable private RunPod template: id `1zpm2v05rn` (legacy name in RunPod may still be `dead-keys-census-cuda` until manually renamed)
- When creating or refreshing that template (see `~/.pi/agent/skills/runpodctl` for template commands), set template `env.PUBLIC_KEY` to the currently active local SSH public key: `ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIC3+THEulpHy8+xqtiB0GEbsZM9GGrHEKmRTNrcqOAfx vedat@cowork.thinkerer.net` (`~/.ssh/id_ed25519.pub`, fingerprint `SHA256:Ln1JafG0riXUGmk8Dj85yxP7HgqoE49hOEtqYOIBrRI`).

Ad-hoc replacement pod preferences when the latest pod is unavailable:

- GPU: `RTX A5000` or `L4`
- CPU: at most 6 cores
- Container image for future publishes: `ghcr.io/vhallac/crockpot-experiments-runpod:latest`
- RAM: around 60 GB
- Disk: around 30 GB
- Prefer the same datacenter / compatible GPU when possible so the existing network volume is available.
- Attach the network volume at pod creation/deployment time; do not expect to attach one later to an existing pod.

Current project paths and cache parameters for RunPod:

- Repository path in pods: `/workspace/crockpot-experiments`
- Persistent cache root: `/workspace/crockpot-experiments-cache`
- Setup script: `./scripts/runpod-persistent-cache-setup`
- Cache env file: `~/.crockpot-experiments-runpod-env`
- CUDA wrappers: `scripts/cuda-run`, `scripts/cuda-python`
- CUDA venv override: `DEAD_KEYS_CUDA_VENV=/path/to/venv`
- Canonical shared CUDA venv on the network volume: `/workspace/venv`
- Preserved old checkout after 2026-07-22 cleanup: `/workspace/crockpot-experiments-legacy`
- Legacy compatible CUDA venv observed on `dead-weight-migration-2`: `/venv-deadkeys`
- To reuse the network-volume venv without reinstalling heavyweight wheels: `DEAD_KEYS_CUDA_VENV=/workspace/venv DEAD_KEYS_CUDA_SKIP_INSTALL=1 ./scripts/cuda-run ...`
- The shared CUDA venv on the network volume is the primary venv for all experiments; avoid creating per-experiment venvs.
- Empty or incomplete venvs previously found under the old cache root were purged on 2026-07-22.
- Treat large uv/pip cache contents as installer scratch unless intentionally rebuilding caches.

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
