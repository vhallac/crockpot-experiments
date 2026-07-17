# Experiment groups

Use this directory for all experiment groups, including groups with implementation code. Keep each experiment independently understandable with its docs and any importable Python package inside the experiment directory.

Every experiment directory should include:

- `README.md` — what is measured, expected signal, execution path, result policy;
- `spec.md` — pre-registration or detailed design when available;
- small curated artifacts only, not raw generated outputs.

Importable packages under experiment directories are not installed as top-level packages. For direct module execution, set `PYTHONPATH` to the relevant experiment package roots, for example:

```bash
PYTHONPATH=experiments/dead-keys uv run python -m deadkeys.scripts.census --model gpt2 --limit-layers 1 --limit-heads 1 --samples 1024 --output-dir outputs/deadkeys_smoke_gpt2
PYTHONPATH=experiments/dead-keys:experiments/queryability uv run python -m queryability.scripts.weights --model gpt2 --limit-layers 1 --limit-heads 1 --output-dir outputs/queryability_smoke_gpt2
```

Generated outputs belong under repository-level `outputs/`, which is ignored by git.
