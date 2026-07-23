# Crockpot Experiments

A slow-simmering collection of transformer mechanistic-interpretability experiments. Each experiment group probes a specific hypothesis about how attention machinery organizes information, with pre-registered specs, committed pre-run state, and externally published outputs so results can be traced back to the exact code and conditions that produced them.

## Experiment groups

- [`experiments/dead-keys/`](experiments/dead-keys/) — census of dead keys across models, with follow-on RoPE and intervention phases.
- [`experiments/queryability/`](experiments/queryability/) — paired `W_Q^T W_K` query/key geometry analysis.
- [`experiments/k-address-space/`](experiments/k-address-space/) — pre-registered test of whether cached key vectors behave like content addresses, with RoPE acting as a positional namespace mechanism.
- [`experiments/rope-as-scaffold/`](experiments/rope-as-scaffold/) — directed program testing whether RoPE's positional contribution is a removable training scaffold, motivated by the k-address-space M1.5/M1.6 findings and connecting to DroPE.

Each group carries its own `README.md` (what it measures, expected signal, how to run it, result policy) and, where available, a `spec.md` pre-registration. Reproducible runs additionally keep a `NOTEBOOK.md` lab notebook in the experiment directory.

## Methodology

Experiments here aim to be reproducible from durable artifacts rather than ad-hoc outputs:

- the spec and code are committed before the full run (pre-run commit);
- a lab notebook entry records the hypothesis and interpretation plan before results exist;
- raw outputs stay out of git and are published as checksummed GitHub Release assets, tagged `run/<experiment-id>/<date>`;
- the completed notebook and a run manifest tie results back to commits, commands, and hashes.

Operational conventions live in [`AGENTS.md`](AGENTS.md). Procedural workflows live in [`.pi/skills/`](.pi/skills/): `reproducible-research` for the experiment lifecycle and `runpod-usage` for GPU pod operations. Both are written for agentic execution but are equally readable as human protocol documents.

## Running

The project uses `uv` for environment management. A typical smoke test:

```bash
uv sync
PYTHONPATH=experiments/dead-keys uv run python -m deadkeys.scripts.census \
  --model gpt2 --limit-layers 1 --limit-heads 1 --samples 1024 \
  --output-dir outputs/deadkeys_smoke_gpt2
```

The checked-in lockfile is pinned for ROCm PyTorch; `scripts/` contains wrappers for ROCm, CUDA (RunPod), and NixOS CPU execution. See `AGENTS.md` for hardware-specific notes.

Generated outputs land under `outputs/`, which is ignored by git; published run artifacts are attached to GitHub Releases instead.

## Author

Vedat Hallac ([@vhallac](https://github.com/vhallac))

## License

This repository is licensed under the GNU General Public License v3.0 — see [LICENSE](LICENSE).
