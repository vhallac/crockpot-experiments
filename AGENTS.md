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
