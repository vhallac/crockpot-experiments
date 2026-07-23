# K-space as address space experiment

## What this measures

This pre-registered experiment tests whether cached key vectors behave like content addresses for referents: same-referent mentions clustering in K-space, RoPE splitting addresses by position as a namespace mechanism, and value/output norms acting as version-dominance signals during reads.

> **M1 status — retracted (F8).** All published M1 address-purity results (GPT-2, Pythia,
> Qwen3, NoPE) are instrument artifacts: the Track A corpus contains zero valid trials, so
> the nulls say nothing about the models or scale. The M1 rerun is deferred behind M1.6.
> See "Known corpus defect F8" at the top of [`NOTEBOOK.md`](NOTEBOOK.md). M1.5 and M1.6 are
> unaffected (they use repeated-segment stimuli).

## Expected signal

The primary predictions are:

- Some address heads should separate same-referent pairs from lexical and position controls.
- The address signal should survive at least some different-surface coreference chains.
- RoPE should produce a dose-response namespace effect: GPT-2 < Pythia < Qwen3, with Pythia's static subspace acting as an internal control.
- Version-dominance via `||v||` or `||W_O v||` is uncertain and explicitly outcome-informative either way.

## How to execute

The first implemented slice runs Track A + M1 address purity for GPT-2 `k_pre` keys:

```bash
PYTHONPATH=experiments/dead-keys:experiments/k-address-space ./scripts/nix-cpu-run -m kaddress.scripts.address_purity \
  --model gpt2 \
  --limit-docs 2 \
  --limit-layers 1 \
  --limit-heads 1 \
  --output-dir outputs/k_address_space_m1_gpt2_smoke
```

The M1.5 repeated-segment position/content probe from `addendum-M1.5.md` has its own smoke command:

```bash
PYTHONPATH=experiments/dead-keys:experiments/k-address-space ./scripts/nix-cpu-run -m kaddress.scripts.position_content \
  --model nope-gpt-small \
  --revision 320681e33a029517e27c68a0f9c2b07ea0004155 \
  --families A,B \
  --repetitions 128 \
  --limit-layers 1 \
  --limit-heads 1 \
  --null-permutations 1 \
  --output-dir outputs/k_address_space_m15_nope_smoke
```

The M1.6 v1.1 hypothesis discriminator from `addendum-M1.6.md` has a NoPE-only R=128 smoke command:

```bash
PYTHONPATH=experiments/dead-keys:experiments/k-address-space ./scripts/nix-cpu-run -m kaddress.scripts.m16_discriminator \
  --model nope-gpt-small \
  --revision 320681e33a029517e27c68a0f9c2b07ea0004155 \
  --repetitions 128 \
  --limit-stimuli 1 \
  --limit-layers 1 \
  --limit-heads 1 \
  --output-dir outputs/k_address_space_m16_nope_v11_smoke
```

Before full RoPE-model M1 runs, implement and pass the remaining sanity gates in `spec.md`, especially key reconstruction from `k_pre` + RoPE to `k_post` and a perturbation check that proves the gate can fail for RoPE models.

## Result policy

The spec estimates full extracted data at roughly 50–150 MB per model when storage is disciplined. Keep generated data under `outputs/` and out of git by default. Commit only small manifests, analysis summaries, and paper-ready curated tables/figures.

## Source

See [`spec.md`](spec.md) for the full pre-registration.
