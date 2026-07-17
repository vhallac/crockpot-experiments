# Dead-key census experiment group

## What this measures

This group studies whether transformer attention heads contain low-effect or near-dead key directions in their Q/K geometry, and how those directions interact with RoPE bands and intervention experiments.

## Expected signal

The original expectation is that some heads have measurable low-rank or low-effect key subspaces, with model- and band-specific structure. Later phases test whether those subspaces are merely geometric artifacts or have runtime/perplexity consequences.

## Methodology notes

- Run the q/k reconstruction sanity check before trusting census outputs.
- Never pool raw scores across heads.
- Use smoke limits before full-model runs, then verify expected output files externally.

## How to execute

Use the repository `uv` environment for local setup and the project wrappers for host-specific execution:

```bash
uv sync
PYTHONPATH=experiments/dead-keys uv run python -m deadkeys.scripts.census --model gpt2 --limit-layers 1 --limit-heads 1 --samples 1024 --output-dir outputs/deadkeys_smoke_gpt2
```

On local ROCm hosts, prefer:

```bash
PYTHONPATH=experiments/dead-keys ./scripts/rocm-run python -m deadkeys.scripts.census --model gpt2 --limit-layers 1 --limit-heads 1 --samples 1024 --output-dir outputs/deadkeys_smoke_gpt2
```

On RunPod CUDA hosts, follow `AGENTS.md` and `.pi/skills/runpod-usage`, then use `scripts/cuda-run`.

## Result policy

Write generated outputs under `outputs/`. Do not commit full experiment results by default. Commit only small manifests, summaries, or paper-ready extracts when intentionally curated.
