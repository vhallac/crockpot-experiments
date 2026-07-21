# K-address-space M1.5 — NoPE-GPT-Small position-content probe

**Date:** 2026-07-21  
**Model:** `andrewdalpino/NoPE-GPT-Small-Base` @ `320681e33a029517e27c68a0f9c2b07ea0004155`  
**Run:** `outputs/k_address_space_m15_nope_gpt_small_20260721`  
**Published outputs:** <https://github.com/vhallac/crockpot-experiments/releases/tag/run/k-address-space-m15-nope-gpt-small/20260721>

## Verdict

NoPE-GPT-Small has an architectural-zero key at layer 0, then develops strong decodable position information in `k_pre` with depth under repeated-token Family A stimuli.

Selected Family A layer means:

| layer | position fraction | ridge R² | PCA k90 | R² after PC projection |
|---:|---:|---:|---:|---:|
| 0 | 8.88e-7 | 0.000 | 1.00 | 0.000 |
| 1 | 0.00797 | 0.029 | 1.01 | 0.020 |
| 2 | 0.0113 | 0.152 | 1.14 | 0.138 |
| 6 | 0.0400 | 0.727 | 1.71 | 0.251 |
| 12 | 0.0481 | 0.951 | 1.10 | 0.055 |
| 18 | 0.0843 | 0.951 | 2.04 | 0.010 |
| 23 | 0.0892 | 0.979 | 2.20 | -0.023 |

The aggregate Family A projector rows show token identity is retained after projection in the reservoir sample: layer-23 mean token-identity accuracy `0.9844 → 0.9957`, while ridge R² drops from `0.9069` to `0.0049` after projecting out the aggregate position PCs.

## Gates and controls

- **G1 architectural zero:** passed for all 560 layer-0 checked slots/heads. Max layer-0 position fraction was `1.52e-6`, below the `1e-5` floor, and deliberate perturbation made the gate fail.
- **Variance floor:** layer-0 ridge R² was forced to `0.0` as intended.
- **Shuffled-y null:** not fully clean under the pre-run warning threshold. Slot-level shuffled R² absolute quantiles were median `0.0359`, 90% `0.0967`, 95% `0.1222`, 99% `0.1902`; the manifest therefore records `shuffle_null_ok=false`. The aggregate rows are much cleaner, e.g. selected Family A layers have shuffled R² around `-0.003` to `-0.006`.

## Caveats

- Family B produced no valid stimuli for this tokenizer/settings because the frame-token alignment and ≥120 repetition constraints were not simultaneously satisfied. Family C did run as a natural-recurrence control and was stronger than A on average (`position_fraction=0.177`, `ridge_r2=0.848`), but remains confounded as pre-registered.
- This is the selected NoPE run only; the cross-model M1.5 comparison still needs GPT-2/Pythia/Qwen3 runs for the addendum's full decision tree.
