# Phase 1.5 GPT-2 Certified QK Truncation

## Goal memory
- Final artefacts: `truncation_gpt2.csv`, `ppl_curve_gpt2.csv`, `null_model_gpt2.csv`, and plots.
- Source inputs: `dead_key_census_spec.md`, GPT-2 HF model, Phase 1 loading/spectra helpers.
- Quality expectations: GPT-2 only, no training, verify generated outputs externally.
- Must not lose: original GPT-2 attention scaling by `sqrt(d_head)`; V/O paths untouched in conceptual patch/eval.

## Bounds
- Work unit: one complete Phase 1.5 implementation pass with bounded execution.
- Stop rule: stop after implementation/execution report, or if verification fails twice.
- Verification per unit: compile, run CLI, verify expected files/rows.

## Progress
1. [x] Implement Phase 1.5 script
2. [x] Execute bounded GPT-2 Phase 1.5 run
3. [x] Verify outputs

## Notes / uncertainties
- Full spec asks WikiText-2 ≥200k tokens and null samples=500/head; current execution used 4096 eval tokens and 5 null samples/head as a bounded first run.
- Uniform certified ranks kept all 64 dimensions for eps <= 1.0, so the initial certified truncation curve shows 0 compression under the hard bound.
