# K-address-space lab notebook

## 2026-07-23 — K-address-space M1.6 Qwen3 v1.1 RoPE k_pre CUDA re-run prep

### Question / Hypothesis

Does Qwen3-0.6B's RoPE/GQA key space expose a query-readable repetition address when the M1.6 donor K patch is applied to `k_pre` (post Q/K norm, pre-RoPE) and then rotated into the target slot's position, or do the repeated-segment signals remain anti-collision/inert or induction-like rather than causal output-addressing?

### Experiment Design Summary

Run `kaddress.scripts.m16_discriminator` for `qwen3` (`Qwen/Qwen3-0.6B`) on RunPod CUDA with M1.6 v1.1 at `R=128`, four stimuli, per-stimulus G6 marker search, G7 noise-controlled attention, output-above-noise addressing criterion, and mandatory altered-interior transitivity readout. The RoPE-specific correction in commit `4ea8fc4` patches qwen3 K before `_apply_rotary_pos_emb`, so donor content is transplanted while target-slot RoPE supplies the target position; the previous post-RoPE `k_post` transplant is treated as confounded and superseded.

### Planned Procedure

1. Review the M1.6 addendum and qwen3 patch implementation for protocol/code alignment and CUDA suitability.
2. Commit this fresh pre-run notebook state.
3. Bring up a RunPod CUDA pod using `scripts/runpod-bring-up`, initialize `/workspace` cache setup, and run through `scripts/cuda-run` with the shared `/workspace/venv`.
4. Run a qwen3 CUDA tripwire matching the full command shape with `--repetitions 128 --limit-stimuli 1 --limit-layers 1 --limit-heads 1`; record progress rate and GPU/CPU utilization.
5. If the tripwire is healthy, run the full qwen3 M1.6 v1.1 RoPE `k_pre` command across all layers/heads/stimuli with progress lines.
6. Package outputs as `.tar.gz`, generate `SHA256SUMS`, publish through a GitHub Release, verify the release assets, analyse the CSVs, then complete this notebook entry.

Planned full command:

```bash
PYTHONPATH=experiments/dead-keys:experiments/k-address-space DEAD_KEYS_CUDA_SKIP_INSTALL=1 ./scripts/cuda-run -m kaddress.scripts.m16_discriminator \
  --model qwen3 \
  --device cuda \
  --repetitions 128 \
  --output-dir outputs/k_address_space_m16_qwen3_v11_rope_kpre_cuda_20260723 \
  --progress-every 20
```

### Expected Signal / Interpretation Plan

- G6 must pass per stimulus after marker search; failed stimuli invalidate the run.
- G7 requires Patch-K target-attention delta to exceed matched-noise attention by the registered margin.
- Addressing requires both G7 attention redirection and donor-marker output movement above noise.
- Attention movement without output movement is not addressing.
- Match+1 attention and altered-interior marker tracking are used to assess whether induction explains the repeated-segment behavior.
- For RoPE models, a null after `k_pre` patching is interpretable as lack of content-addressing under this instrument; it is not the donor-position-rotation confound of the interrupted `k_post`-patch run.

### Pre-run Provenance

- Spec: `experiments/k-address-space/addendum-M1.6.md` v1.1, including RoPE `k_pre` patch-stage clarification
- Code branch: `main`
- Pre-run commit: `af540ae`
- Planned output location: `outputs/k_address_space_m16_qwen3_v11_rope_kpre_cuda_20260723`
- Checklist: `temp/repro-checklists/20260723-k-address-space-m16-qwen3-v11.md`
- Local preparation evidence: `PYTHONPATH=experiments/dead-keys:experiments/k-address-space ./scripts/nix-cpu-run -m unittest experiments/k-address-space/tests/test_position_content.py` (14 OK); `PYTHONPATH=experiments/dead-keys:experiments/k-address-space ./scripts/nix-cpu-run -m py_compile experiments/k-address-space/kaddress/scripts/m16_discriminator.py` passed. Local Transformers does not support qwen3, so qwen3 execution sanity is deferred to RunPod.

### Results

Run command on RunPod NVIDIA L4 pod `m55vlv45okg93c` from pre-run commit `af540ae`:

```bash
PYTHONPATH=experiments/dead-keys:experiments/k-address-space DEAD_KEYS_CUDA_SKIP_INSTALL=1 ./scripts/cuda-run -m kaddress.scripts.m16_discriminator \
  --model qwen3 \
  --device cuda \
  --repetitions 128 \
  --output-dir outputs/k_address_space_m16_qwen3_v11_rope_kpre_cuda_20260723 \
  --progress-every 20
```

Tripwire evidence: the CUDA tripwire with `--repetitions 128 --limit-stimuli 1 --limit-layers 1 --limit-heads 1` passed G6 and ran at about `1.159` units/s, with GPU samples up to 100%. The full run completed `1792/1792` units at about `1.016` units/s with 95-100% GPU samples and wrote four files under `outputs/k_address_space_m16_qwen3_v11_rope_kpre_cuda_20260723`.

Post-run correction: initial analysis found a derived-classification bug in G7: `k_attn_delta > noise_attn_delta + margin` could pass even when patch-K target-attention movement was non-positive. The raw CUDA measurements were preserved; only `kaddress_m16_classification_qwen3.csv` and `kaddress_m16_manifest_qwen3.json` were recomputed locally from raw rows after patching the rule to require `mean patch-k target_attention_delta > 0` as well as above-noise movement. The recomputed manifest records `derived_recomputed_from_raw_at_utc=2026-07-23T10:43:31Z`.

Corrected manifest highlights:

| field | value |
|---|---:|
| repetitions | 128 |
| stimulus count | 4 |
| raw summary rows | 8960 |
| classification rows | 448 |
| G6 | PASS |
| G7 pass count | 39 |
| transitivity confirmed count | 448 |

G6 per-stimulus marker search:

| stimulus | max/min ratio | searched sets | selected markers |
|---|---:|---:|---|
| M16_00 | 1.624 | 58 | `exactly,together,same,thick` |
| M16_01 | 2.141 | 8 | `maybe,first,best,tight` |
| M16_02 | 2.923 | 18 | `equally,quiet,weak,round` |
| M16_03 | 2.019 | 226 | `softly,currently,last,minor` |

Per-head corrected classification counts:

| classification | heads |
|---|---:|
| confounded_noise_sensitive | 150 |
| mixed | 124 |
| inert | 65 |
| anti_collision_or_content_driven | 40 |
| anti_collision_or_inert_attention_only | 37 |
| transitive_induction | 30 |
| addressing | 2 |

Corrected aggregate addressing heads:

| layer | head | mean K attention delta | mean noise attention delta | mean both donor-prob delta | mean noise donor-prob delta | per-stimulus addressing passes |
|---:|---:|---:|---:|---:|---:|---|
| 24 | 15 | 0.108811 | -0.025904 | 0.000507 | 0.000023 | M16_01 only |
| 25 | 14 | 0.002021 | -0.111828 | 0.000637 | 0.000021 | M16_01 only |

Published artifacts:

- Release: https://github.com/vhallac/crockpot-experiments/releases/tag/run/k-address-space-m16-qwen3/20260723
- Bundle: `k_address_space_m16_qwen3_v11_rope_kpre_cuda_20260723.tar.gz`
- Bundle SHA256: `65af38a577fc67ec727f2abc7c966742e68e7fc78e2c8513c14abeb92299e779`
- Checksum asset: `SHA256SUMS_k_address_space_m16_qwen3_v11_rope_kpre_cuda_20260723`
- Git-side manifest: `experiments/k-address-space/artifacts/m16_qwen3_v11_rope_kpre_20260723_manifest.json`

### Analysis

The raw qwen3 RoPE `k_pre` run is valid for M1.6 v1.1: R is restored to 128, all four stimuli pass G6, the full layer/head/stimulus grid completed on CUDA, and the post-run G7 correction was applied only to derived classification from preserved raw rows.

The corrected result is **not a robust positive address-space finding**. Only 39/448 heads pass the noise-controlled attention gate, and only 2/448 pass the aggregate addressing rule. Those two aggregate positives are fragile: each has a per-stimulus addressing pass on M16_01 only, and no head shows addressing across multiple stimuli. Output shifts are small (`~5e-4` to `~6e-4` in mean donor-marker probability), while many heads are noise-sensitive or mixed.

The transitivity readout is not strong head-specific evidence in this summary. It is confirmed for all 448 heads with rank `4` and nearly constant altered-marker probability (`~0.017764`), so it currently reads more like a model/readout-level behavior than a discriminating per-head mechanism. The 30 `transitive_induction` classifications and high match+1 masses in some heads keep induction-like explanations live, but the all-head transitivity pass should not be over-interpreted as 448 independent confirmations.

Compared with the pre-fix derived artifacts, the corrected rule removes the false-positive `L21H8` aggregate addressing call and reduces G7 passes from 77 to 39. The remaining late-layer heads (`L24H15`, `L25H14`) are best treated as weak/equivocal candidates, not as a confirmed query-readable tape address.

### Conclusion / Next Step

M1.6 Qwen3 v1.1 RoPE `k_pre` does **not** provide robust evidence that Qwen3 exposes a query-readable repetition address under this instrument. The defensible conclusion is a null/equivocal result: raw measurements are valid and two late heads weakly satisfy the aggregate addressing rule after correction, but the effect is one-stimulus-fragile, small in output, and not separated cleanly from induction/readout-level behavior.

Use the corrected published artifacts for any downstream write-up. If Qwen3 is revisited, the next measurement should strengthen the per-head transitivity/addressing split rather than treating the current all-head transitivity flag as decisive.

## 2026-07-23 — K-address-space M1.6 Qwen3 v1.1 CUDA run prep

### Question / Hypothesis

Does Qwen3-0.6B's RoPE/GQA key space expose a query-readable repetition address under the corrected M1.6 v1.1 discriminator, or do the M1.5-style repeated-segment signals remain anti-collision/inert or induction-like rather than causal output-addressing?

### Experiment Design Summary

Run `kaddress.scripts.m16_discriminator` for `qwen3` (`Qwen/Qwen3-0.6B`) on RunPod CUDA with the final M1.6 v1.1 design: `R=128`, four stimuli, per-stimulus G6 marker search, G7 noise-controlled attention, output-above-noise addressing criterion, and mandatory altered-interior transitivity readout. The qwen3 harness patches query-head-local expanded K/V vectors after Qwen3 q/k RMSNorm and RoPE, so GQA heads are measured as per-query-head causal interventions while preserving the model's grouped key/value projections.

### Planned Procedure

1. Commit this pre-run notebook entry and the qwen3 M1.6 harness support.
2. Bring up a RunPod CUDA pod using `scripts/runpod-bring-up`, initialize `/workspace` cache setup, and run through `scripts/cuda-run` with the shared `/workspace/venv`.
3. Run a qwen3 CUDA tripwire matching the full command shape with `--repetitions 128 --limit-stimuli 1 --limit-layers 1 --limit-heads 1`; record progress rate and GPU/CPU utilization.
4. If the tripwire is healthy, run the full qwen3 M1.6 v1.1 command across all layers/heads/stimuli with progress lines.
5. Package outputs as `.tar.gz`, generate `SHA256SUMS`, publish through a GitHub Release, verify the release assets, analyse the CSVs, then complete this notebook entry.

Planned full command:

```bash
PYTHONPATH=experiments/dead-keys:experiments/k-address-space DEAD_KEYS_CUDA_SKIP_INSTALL=1 ./scripts/cuda-run -m kaddress.scripts.m16_discriminator \
  --model qwen3 \
  --device cuda \
  --repetitions 128 \
  --output-dir outputs/k_address_space_m16_qwen3_v11_cuda_20260723 \
  --progress-every 20
```

### Expected Signal / Interpretation Plan

- G6 must pass per stimulus after marker search; failed stimuli invalidate the run.
- G7 requires Patch-K target-attention delta to exceed matched-noise attention by the registered margin.
- Addressing requires both G7 attention redirection and donor-marker output movement above noise.
- Attention movement without output movement is not addressing.
- Match+1 attention and altered-interior marker tracking are used to assess whether induction explains the repeated-segment behavior.

### Pre-run Provenance

- Spec: `experiments/k-address-space/addendum-M1.6.md` v1.1
- Code branch: `main`
- Pre-run commit: `292ea1f`; attention-mask fix commit before full run: `e5b0973`
- Planned output location: `outputs/k_address_space_m16_qwen3_v11_cuda_20260723`
- Checklist: `temp/repro-checklists/20260723-k-address-space-m16-qwen3-v11.md`
- Local preparation evidence: `PYTHONPATH=experiments/dead-keys:experiments/k-address-space ./scripts/nix-cpu-run -m unittest experiments/k-address-space/tests/test_position_content.py` (14 OK); `PYTHONPATH=experiments/dead-keys:experiments/k-address-space ./scripts/nix-cpu-run -m py_compile experiments/k-address-space/kaddress/scripts/m16_discriminator.py` passed. Local Transformers does not support qwen3, so qwen3 smoke is deferred to RunPod.

### Results

Run interrupted by user on 2026-07-23T09:51Z before completion because the M1.6 RoPE addendum needs an additional hypothesis/measurement before this qwen3 run should be interpreted or published.

Interruption evidence: full command was running on RunPod pod `t4q2qswjocnx0x` (`crockpot-debug-20260723092803`, NVIDIA L4) from commit `e5b0973`; log tail before termination showed progress through `760/1792` units at about `1.045` units/s, last reported slice `stimulus=M16_01 layer=19 head=7`, with GPU monitor samples at 96-100% utilization and about 5602 MiB used. The process PID `828` was terminated, and the RunPod pod was stopped with `desiredStatus=EXITED` at `2026-07-23 09:51:34 UTC`.

No completed output package or release was produced for this interrupted run.

### Analysis

Not analysed. The partial run is intentionally superseded by the planned RoPE-specific addendum update and must not be treated as a completed qwen3 M1.6 result.

### Conclusion / Next Step

Revise `experiments/k-address-space/addendum-M1.6.md` with the new RoPE hypothesis/measurement, then rerun qwen3 M1.6 from a fresh pre-run entry/checklist.

## 2026-07-23 — K-address-space M1.6 NoPE-GPT-Small v1.1 CUDA re-run prep

### Question / Hypothesis

Does the compact NoPE-GPT-Small key-position signal from M1.5 act as a query-readable address at the M1.5 scale (`R >= 128`), or is the v1.0 attention-moves/output-null result better explained by anti-collision/inert key cargo or transitive induction? This v1.1 re-run implements the addendum corrections C1-C3 before interpreting any model-level taxonomy.

### Experiment Design Summary

Run `kaddress.scripts.m16_discriminator` for `nope-gpt-small` (`andrewdalpino/NoPE-GPT-Small-Base`, revision `320681e33a029517e27c68a0f9c2b07ea0004155`) on RunPod CUDA with the v1.1 harness. The repaired implementation keeps all non-probed repetitions marker-free, marks only target/donor/altered/readout repetitions, searches marker sets separately per stimulus for G6 neutrality, records noise-patch target-attention deltas for G7, requires addressing to pass both attention-above-noise and output-above-noise, and runs the altered-interior transitivity readout for every layer/head.

### Planned Procedure

1. Commit this pre-run notebook entry, the v1.1 script changes, and smoke tests.
2. Bring up a RunPod CUDA pod with `scripts/runpod-bring-up`, initialize the network-volume cache, and use `/workspace/venv` through `scripts/cuda-run` / `scripts/cuda-python`.
3. Run a CUDA tripwire matching the real command shape with `--repetitions 128 --limit-stimuli 1 --limit-layers 1 --limit-heads 1`; record progress rate, utilization, and extrapolated runtime.
4. If the tripwire is GPU-bound and within budget, run the full NoPE v1.1 command on CUDA with all four stimuli and progress lines enabled.
5. Package outputs as `.tar.gz`, generate `SHA256SUMS`, publish via a GitHub Release, verify release assets, analyse the CSVs, then complete this notebook entry.

Planned full command:

```bash
PYTHONPATH=experiments/dead-keys:experiments/k-address-space DEAD_KEYS_CUDA_SKIP_INSTALL=1 ./scripts/cuda-run -m kaddress.scripts.m16_discriminator \
  --model nope-gpt-small \
  --revision 320681e33a029517e27c68a0f9c2b07ea0004155 \
  --device cuda \
  --repetitions 128 \
  --output-dir outputs/k_address_space_m16_nope_gpt_small_v11_cuda_20260723 \
  --progress-every 20
```

### Expected Signal / Interpretation Plan

- G6 must pass per stimulus after marker search; failed stimuli invalidate patching readouts rather than being averaged in.
- G7 requires Patch-K target-attention delta to exceed norm-matched-noise target-attention delta by the pre-set margin.
- Addressing requires both G7 attention redirection and donor-marker output movement above the noise-output baseline.
- Attention-above-noise with output-null is not addressing; it supports anti-collision/inert attention-only behavior.
- High match+1 mass plus altered-marker tracking in the mandatory transitivity stimulus supports transitive induction.

### Pre-run Provenance

- Spec: `experiments/k-address-space/addendum-M1.6.md` v1.1
- Code branch: `main`
- Pre-run commit: `fcb2bd7`
- Planned output location: `outputs/k_address_space_m16_nope_gpt_small_v11_cuda_20260723`
- Checklist: `temp/repro-checklists/20260723-k-address-space-m16-nope-v11.md`
- Local preparation evidence: `PYTHONPATH=experiments/dead-keys:experiments/k-address-space ./scripts/nix-cpu-run -m unittest experiments/k-address-space/tests/test_position_content.py` (14 OK); R=128 1-stimulus/1-layer/1-head local smoke wrote `outputs/k_address_space_m16_nope_v11_r128_local_smoke` with `gate_g6_pass=PASS`.

### Results

Run command on RunPod NVIDIA L4 from valid pre-run commit `fcb2bd7`:

```bash
PYTHONPATH=experiments/dead-keys:experiments/k-address-space DEAD_KEYS_CUDA_SKIP_INSTALL=1 ./scripts/cuda-run -m kaddress.scripts.m16_discriminator \
  --model nope-gpt-small \
  --revision 320681e33a029517e27c68a0f9c2b07ea0004155 \
  --device cuda \
  --repetitions 128 \
  --output-dir outputs/k_address_space_m16_nope_gpt_small_v11_cuda_20260723 \
  --progress-every 20
```

Redo note: the first full v1.1 run completed but was discarded because G6 failed for one stimulus (`M16_01` ratio `4.25`). The harness was fixed to expand marker search, sample exact candidate sets, and abort if G6 cannot pass; the valid run below is after that redo.

Run evidence: the CUDA tripwire over all four stimuli with `--limit-layers 1 --limit-heads 1` passed G6 and ran at `~1.75` units/s with ~99-100% GPU utilization. The full run log reports `starting M1.6 discriminator model=nope-gpt-small device=cuda stimuli=4 layers=24 heads=16 repetitions=128`, progress through `units=1520/1536` at `~1.76` units/s, then writes all four output files. Manifest highlights: `repetitions=128`, `stimulus_count=4`, `summary_rows=7680`, `classification_rows=384`, `gate_g6_pass=PASS`, `gate_g7_pass_count=4`, `transitivity_confirmed_count=0`.

G6 per-stimulus marker search:

| stimulus | max/min ratio | searched sets | selected markers |
|---|---:|---:|---|
| M16_00 | 2.295 | 46 | `soon,quietly,forward,third` |
| M16_01 | 2.759 | 40 | `previously,final,tight,square` |
| M16_02 | 1.767 | 12 | `currently,there,away,different` |
| M16_03 | 2.734 | 64 | `truly,steadily,cool,solid` |

Published artifacts:

- Release: https://github.com/vhallac/crockpot-experiments/releases/tag/run/k-address-space-m16-nope/20260723-2
- Bundle: `k_address_space_m16_nope_gpt_small_v11_cuda_20260723.tar.gz`
- SHA256: `8f7f6051eb17fcd0dbf06c4f3e1b9aa116827b2f7b93899c92fb57fcee214f00`
- Git-side manifest: `experiments/k-address-space/artifacts/m16_nope_v11_20260723_manifest.json`

Per-head classification counts:

| classification | heads |
|---|---:|
| confounded_noise_sensitive | 201 |
| mixed | 65 |
| induction_unconfirmed | 51 |
| inert | 35 |
| anti_collision_or_content_driven | 28 |
| anti_collision_or_inert_attention_only | 4 |

### Analysis

The v1.1 result is stronger against the address-reading interpretation than the v1.0 run. **No heads classify as addressing.** Only 4/384 heads pass G7, and all four are `anti_collision_or_inert_attention_only`: Patch-K target attention exceeds matched-noise attention, but output does not follow. Their donor-probability shifts under patch-both are tiny (`~3e-08`, `1e-07`, `1.5e-06`, `7e-08`) and `output_above_noise=False` in all cases. Across all rows, the largest patch-both donor-probability shift is only `7.7e-06`, below the largest absolute noise-output shift (`3.5e-05`).

Most late/mid-depth heads are labelled `confounded_noise_sensitive` because the K-patch attention movement is not cleanly above the noise-patch attention movement. This directly confirms why C2 mattered: raw attention perturbation is common, but content-specific/noise-controlled attention redirection is rare.

Induction-style match+1 attention is present in many heads but the altered-interior transitivity output test does **not** confirm transitive induction as the output mechanism. Mean match+1 mass across heads is `0.210`, with top heads near `0.94`, but `transitivity_confirmed_count=0`. The altered marker's output rank is stimulus-dependent and head-independent in this harness (e.g. rank 3 for one stimulus, rank 42/750/1647 for the others), so it reads as marker/base-rate behavior rather than systematic head-level transitive copying.

Layer pattern: layers 9-16 are dominated by `confounded_noise_sensitive`; layers 0-8 contain most `inert` and low-effect anti-collision/content-driven heads; the four G7-passing attention-only heads are late (L17H6, L18H11, L20H7, L22H8). This is compatible with late attention perturbability without a causal output read.

**Added post-review (recomputed from the published CSVs).** Two points strengthen the reading above.

1. **The K-patch attention effect itself collapses at R=128 — the v1.0 attention redirection was a low-R artifact.** The v1.0 run (R=4) had late heads with content-specific K-patch target-attention deltas up to `+0.43` and ~25 heads above `+0.10`. At the M1.5 scale here (R=128), the max K-patch attention delta across all 384 heads is only `+0.050` (min `−0.252`); just 2 heads exceed `|0.10|` and none exceed `+0.05` in the addressing direction. The four G7 passers clear the noise *margin* but with negligible absolute movement (`+0.038`, `−0.007`, `−0.068`, `+0.043`). So the finding is not merely "attention moves but output doesn't" — at scale, **neither attention nor output responds to K-patching meaningfully**. The v1.0 attention redirection did not survive the C1 regime fix, which is exactly what C1 was meant to expose. This makes the anti-collision/inert verdict cleaner and more robust, not weaker.

2. **`transitivity_confirmed=0` carries no head-level information — the instrument, not the mechanism, is the limit.** The transitivity readout is a model-level next-token prediction, and the CSV confirms it is **head-independent**: the altered-marker rank takes exactly one value per stimulus across all 384 head-rows (`42 / 750 / 3 / 1647` for M16_00–03). One stimulus (M16_02) ranks the altered marker 3rd, the other three bury it — noise at n=4, not systematic transitive copying. So `transitivity_confirmed=0` means the (model-level, n=4) instrument did not detect transitive induction, **not** that induction is excluded. The per-head `transitivity_confirmed` column is therefore not meaningful as written, and induction vs anti-collision is left unadjudicated by this run. A proper transitivity test needs per-head isolation (e.g. ablation during the transitivity query) and many more stimuli.

### Conclusion / Next Step

At the M1.5 scale (`R=128`) with probed-only markers, per-stimulus G6, G7 noise-controlled attention, and mandatory transitivity, NoPE-GPT-Small shows **no evidence for query-readable addressing** in M1.6. The cleanest positive effect is limited to four late heads where K-patching moves attention above noise, but none moves the output above noise. Transitivity is not confirmed, so this run does not replace addressing with a clean transitive-induction account; instead it supports a conservative reading: the M1.5 key-position signal is decodable and sometimes attention-relevant, but causally sterile for next-token retrieval in this instrument. The tape-as-address framing should remain retired/down-scoped for NoPE-GPT-Small unless a future design finds output-following under a less marker-sensitive transitivity readout.

## 2026-07-23 — K-address-space M1.6 NoPE-GPT-Small CUDA run prep

### Question / Hypothesis

Does the compact, highly decodable repetition-position signal found in NoPE-GPT-Small keys by M1.5 act as a usable address read by queries, or is it better explained by anti-collision/inert cargo or ordinary transitive induction? M1.6 discriminates these accounts with marker-neutral continuation stimuli, per-head K/V patching, noise controls, and induction-score readouts.

### Experiment Design Summary

Run `kaddress.scripts.m16_discriminator` for `nope-gpt-small` (`andrewdalpino/NoPE-GPT-Small-Base`) on RunPod CUDA. The implementation constructs repeated Family-A-style clauses ending in distinct single-token continuation markers, checks G6 marker neutrality on the unpatched next-token distribution, then for each selected NoPE layer/head performs baseline, K-only, V-only, K+V, and norm-matched-noise patch passes at an interior target marker slot using a separated donor marker slot. The first implementation is NoPE-only because it patches the audited NoPE attention module directly; RoPE pre/post generalization is deferred.

### Planned Procedure

1. Commit this pre-run notebook entry, the M1.6 script, and smoke tests.
2. Bring up a RunPod CUDA pod with `scripts/runpod-bring-up`, initialize the shared network-volume cache, and use `/workspace/venv` through `scripts/cuda-run` / `scripts/cuda-python`.
3. Run a CUDA tripwire matching the real command shape with `--limit-stimuli 1 --limit-layers 1 --limit-heads 1`; record GPU/CPU utilization and extrapolated runtime.
4. If the tripwire is GPU-bound and within budget, run the full NoPE M1.6 command on CUDA with progress lines enabled.
5. Package outputs as `.tar.gz`, generate `SHA256SUMS`, publish via a GitHub Release, verify release assets, then analyse the CSVs and complete this notebook entry.

Planned full command:

```bash
PYTHONPATH=experiments/dead-keys:experiments/k-address-space ./scripts/cuda-run -m kaddress.scripts.m16_discriminator \
  --model nope-gpt-small \
  --revision 320681e33a029517e27c68a0f9c2b07ea0004155 \
  --device cuda \
  --output-dir outputs/k_address_space_m16_nope_gpt_small_cuda_20260723 \
  --progress-every 20
```

### Expected Signal / Interpretation Plan

- G6 must pass before patching results are interpretable; an initial local biased-marker smoke failed, so the default marker order was changed to a NoPE-neutral four-marker set and resmoked.
- Patch-K attention redirection plus donor-specific probability movement supports addressing.
- Patch-V probability movement without K attention redirection supports content-driven use with position not acting as a dial.
- Noise effects comparable to donor patches confound causal patch interpretation.
- High match+1 attention mass supports induction; classifications are reported per layer/head as addressing, induction, anti-collision/content-driven, inert, mixed, or noise-confounded.

### Pre-run Provenance

- Spec: `experiments/k-address-space/addendum-M1.6.md`
- Code branch: `main`
- Pre-run commit: `0b7c9a9`
- Planned output location: `outputs/k_address_space_m16_nope_gpt_small_cuda_20260723`
- Checklist: `temp/repro-checklists/20260723-k-address-space-m16-nope.md`
- Local preparation evidence: `./scripts/nix-cpu-run -m unittest experiments/k-address-space/tests/test_position_content.py` (13 OK); local 1-stimulus/1-layer/1-head smoke wrote M1.6 outputs and passed G6.

### Results

Run command on RunPod NVIDIA L4 from pre-run commit `0b7c9a9`:

```bash
PYTHONPATH=experiments/dead-keys:experiments/k-address-space DEAD_KEYS_CUDA_SKIP_INSTALL=1 ./scripts/cuda-run -m kaddress.scripts.m16_discriminator \
  --model nope-gpt-small \
  --revision 320681e33a029517e27c68a0f9c2b07ea0004155 \
  --device cuda \
  --limit-stimuli 1 \
  --output-dir outputs/k_address_space_m16_nope_gpt_small_cuda_20260723 \
  --progress-every 20
```

The `--limit-stimuli 1` scope is a deliberate G6 recovery: a full four-prefix run completed but failed marker neutrality on some prefixes, so those patching results were discarded as pre-registered-invalid. The rerun keeps full all-layer/all-head NoPE coverage (`24 × 16 = 384` heads) on the first prefix, whose marker set passes G6.

Run evidence: log reports `starting M1.6 discriminator model=nope-gpt-small device=cuda stimuli=1 layers=24 heads=16`, progress through `units=384/384` at about `19.75/s`, and writes all four output files. The manifest records `gate_g6_pass=PASS`, `summary_rows=1920`, `classification_rows=384`, and environment CUDA device `NVIDIA L4`. G6 has 384/384 passing rows, max/min marker-probability ratio `1.2096`.

Published artifacts:

- Release: https://github.com/vhallac/crockpot-experiments/releases/tag/run/k-address-space-m16-nope/20260723
- Bundle: `k_address_space_m16_nope_gpt_small_cuda_20260723.tar.gz`
- SHA256: `5a8d42e6adc6fa2a6d79714ba78a8a0b223ca79f3e76489b122448eb165fc33c`

Per-head classification counts:

| classification | heads |
|---|---:|
| mixed | 130 |
| anti_collision_or_content_driven | 104 |
| confounded_noise_sensitive | 76 |
| addressing | 33 |
| induction | 26 |
| inert | 15 |

Layer pattern: addressing-classified heads cluster mainly in late layers 17–23 (28 of 33), while inert heads are almost entirely layer 8 (13 of 15). Mixed heads are common across the stack.

### Analysis

> **Interpretation corrected 2026-07-23 (post-review).** The raw counts/numbers in Results
> are accurate, but the original reading below over-stated the evidence for addressing. Two
> errors: (1) it treated K-patch *attention* redirection as addressing without noise-controlling
> that criterion; (2) it did not weight the **null output-following**, which is the decisive
> measurement. Corrected analysis follows.

**Output-following is null across all heads — the decisive result.** Addressing (addendum §4.1)
requires attention to redirect *and* the output to follow. It does not follow: patch-both (donor
K+V into the target slot) moves the donor marker's next-token probability by at most `+0.010` in
the single best head (L17H9), against a baseline donor prob of `~0.036` and a noise floor of
`~0.001`; the max content-specific output shift across all 384 heads is `+0.010`. No head
causally steers the readout toward the donor's continuation. By the addendum's own key, this is
**not** the addressing quadrant.

**The 33-head "addressing" bucket is an artifact of an un-noise-controlled attention criterion.**
The classifier keys on K-patch attention redirection, but the noise control was applied only to
the output readout (`mean_noise_abs_donor_prob_delta`), not to attention — there is no
noise-attention column. Noise-controlled, the bucket collapses: addressing-classified heads show
K-patch attention delta `+0.192` versus **noise-patch attention delta `+0.179`** — essentially
equal, i.e. generic perturbation, which the addendum key says to discard. Only ~25 heads (mostly
L17–22) show genuinely content-specific redirection where noise does *not* move attention — e.g.
L19H7 (`K +0.431` vs `noise −0.065`), L22H8 (`+0.411` vs `−0.068`), L19H3 (`+0.447` vs `+0.037`).
For those ~25 heads the attention redirection is real, but it still does not propagate to the
output (per the null above). So the honest reading is: **some late-layer heads redirect attention
content-specifically under a K-patch, but nothing is read into the output** — attention-moves /
output-null, which the decision tree maps to **anti-collision / inert**, not addressing.

Induction is present (P1.6.d), as expected in this maximally repetitive regime: match+1 mass is
high in many heads (mean `~0.58` among the flagged heads, top near `1.0`). Per the addendum this
is a baseline, not a finding, and the sharp discriminator that would confirm induction as *the*
mechanism — the §4.2.3 altered-interior transitivity test — was **not run**. So induction vs
anti-collision is not finally adjudicated here.

Caveats on scope: (1) **R = 4.** The stimulus is four repetitions (`soon/early/briefly/now`)
because the distinct-single-token-marker requirement caps R at the marker-vocabulary size. That
is 20–60× smaller than the M1.5 regime (R = 128–248) whose signal M1.6 is meant to probe;
whether the M1.5 position signal is even present at R=4 is unestablished. This is a design
tension in `addendum-M1.6.md` §2.1, not just this run. (2) **One stimulus** (G6 passed only for
the first prefix), so no prevalence stability. (3) The `anti_collision_or_content_driven` bucket
(104 heads) has near-zero K redirection, near-zero output deltas, and low induction mass
(`0.0196`) — consistent with inert/anti-collision.

### Conclusion / Next Step

M1.6 gives **weak evidence against** causal addressing in NoPE-GPT-Small at this scale and
regime, not qualified support for it. Attention is content-specifically redirectable under a
K-patch in ~25 late-layer heads, but the redirection does not propagate to the output in any
head (max donor-marker shift `+0.010` over a `~0.036` baseline), and the aggregate "addressing"
count was inflated by an attention criterion that was not noise-controlled. Read against the
addendum decision tree, this leans anti-collision/inert with induction present; the
tape-as-address framing is **weakened**, and the corpus-v3 M1 rerun stays low priority.

This is a first-pass, single-stimulus, R=4 result and does not settle the mechanism. A proper
M1.6 pass needs: (a) the §4.2.3 altered-interior **transitivity** test (the deciding measurement);
(b) a **noise-controlled attention** criterion, not just noise-controlled output; (c)
multi-stimulus runs with per-stimulus neutral markers; and (d) a resolution of the R=4-vs-M1.5
regime gap — either raise R (needs a larger neutral-marker vocabulary or markers only at probe
points) or explicitly scope M1.6's claims to the low-R regime. Until then these per-head counts
should not be used as a model-level taxonomy.

## Known corpus defect F8 — all M1 Track A results are retracted

**Discovered:** 2026-07-21 (code review + data forensics), after the M1 runs below were
published. **Scope:** every M1 address-purity result in this notebook — GPT-2, Pythia-410m,
Qwen3-0.6B, and NoPE-GPT-Small — is an **instrument artifact, not a fact about the models.**

**What F8 is.** The M1 address-purity test asks whether same-referent mentions cluster in
K-space beyond lexical/positional controls. The discriminating (`diff-surface`) trials rest
on shared-alias mentions (`"the person"`), but in the Track A generator the referent's
disambiguating detail (place/value) is emitted **after** the alias token and rotates per
round (`kaddress/corpus.py`, the `generate_track_a` update loop). So at the token position
where the alias key is computed, the referent identity is **causally unavailable** — no
correctly-built instrument could recover it. Same-surface trials, meanwhile, have
referent = name and are lexically trivial. The net effect: **Track A contains zero valid
address-purity trials.**

**Consequences.**

- The four M1 nulls below ("0 address heads", best AUCs ~0.5–0.68) measure a corpus with
  nothing to measure. They do **not** support "semantic addressing is absent or too weak at
  ≤0.6B scale", and their Conclusion sections must be read with that caveat.
- The previously reported Pythia "whisper heads" (L2H4, L8H13) are **withdrawn** — they were
  computed on poisoned rows and fall below chance on clean rows.
- Fixing the M1 code defects alone (F2 row expansion, F3 proximity filter vs true
  distance-matched control, F5 missing permutation null, F6 O(n²) AUC) is **not worth doing
  in isolation** — they fix the instrument, not the corpus. Any M1 rerun requires a **corpus
  v3** (disambiguators precede mentions) plus those code fixes as one package.

**Status.** Corpus v3 is not yet designed/built; the M1 rerun is **deferred behind M1.6**
(the hypothesis discriminator, `addendum-M1.6.md`), whose outcome decides whether a
corpus-v3 M1 is worth building. **M1.5 and M1.6 do not depend on F8** — they use
repeated-segment stimuli that need no referent labels.


## 2026-07-22 — K-address-space M1.5 v1.1 Qwen3-0.6B CUDA run

Question: does Qwen3-0.6B, a full-RoPE model with θ=1e6 and `d_head=128`, show computed/leaked pre-RoPE key position at depth, and is its stamped post-RoPE position fraction weaker than Pythia as predicted by P1.5.e?

Run command on RunPod NVIDIA L4 from commit `6380360`:

```bash
PYTHONPATH=experiments/dead-keys:experiments/k-address-space python -m kaddress.scripts.position_content \
  --model qwen3 \
  --device cuda \
  --families A,B,C \
  --segment-lengths 4,7,12 \
  --repetitions 256 \
  --max-length 3072 \
  --progress-every 1000 \
  --output-dir outputs/k_address_space_m15_v11_qwen3_cuda_20260722
```

Two run-shape caveats are deliberate and recorded: `--repetitions 256` uses Qwen3's required `R_min=2*d_head`; `--max-length 3072` keeps the L=12 cell feasible while avoiding the 4064-token Family C CUDA/OOM path on the L4. The run still includes all requested families A/B/C and the length sweep L=4,7,12.

Preflight/fixes before the successful run:

- Added longer Family A templates so Qwen3 has actual 12-token repeated segments; the original short pool had zero L=12 Qwen3 survivors.
- Avoided a Transformers 4.57 Qwen3 CUDA mask path that requested ~100 GiB for all-ones unpadded masks; M1.5 inputs are single unpadded sequences, so omitting the mask is equivalent.
- Released full-layer CUDA key tensors between stimuli to avoid carrying fragmentation into late long stimuli.
- Per reviewer feedback, the analysis below explicitly pulls the dimensionality/projector columns by default and separates slot rows from `AGGREGATE` rows.

Run evidence: log reports `starting M1.5 analysis model=qwen3 device=cuda stimuli=27 families=A,B,C`, progress to `units=86464` at about `45.0/s`, and writes all four outputs. The manifest records `summary_rows=87808`, `gate_g1_pass=PASS`, `gate_g2_pass=PASS`, `shuffle_null_ok=true`, `max_length=3072`, `requested_repetitions=256`, and `gates_evaluated={"G1_architectural_zero": 1544, "G2_architectural_one": 1544}`. The gates CSV has exactly 3,088 rows; all gate rows pass and all perturbation checks can fail.

Reviewer-requested columns present in the CSV: `pca_components_90pct`, `pca_residual_variance_fraction_90pct`, `r2_after_position_pc_projection`, `token_identity_acc_before`, and `token_identity_acc_after`. The projector artifact is `kaddress_m15_projectors_qwen3.npz`.

Selected Family A **slot-level** means:

| variant | layer | position_fraction | ridge_r2 | pca_k90 | r2_after_projection |
|---|---:|---:|---:|---:|---:|
| pre | 0 | 9.55e-08 | 0.000 | 0.00 | 0.000 |
| pre | 1 | 0.018 | 0.909 | 1.16 | 0.208 |
| pre | 7 | 0.065 | 0.953 | 2.32 | 0.135 |
| pre | 14 | 0.138 | 0.971 | 4.26 | 0.024 |
| pre | 21 | 0.099 | 0.969 | 2.37 | 0.092 |
| pre | 27 | 0.054 | 0.970 | 2.51 | 0.069 |
| post | 0 | 0.242 | 1.000 | 17.43 | -0.022 |
| post | 14 | 0.499 | 0.998 | 19.28 | -0.028 |
| post | 27 | 0.507 | 1.000 | 23.76 | -0.032 |

Selected Family A **aggregate** means:

| variant | layer | position_fraction | ridge_r2 | pca_k90 | r2_after_projection | token_acc_before | token_acc_after |
|---|---:|---:|---:|---:|---:|---:|
| pre | 0 | 0.000002 | 0.000 | 0.00 | 0.000 | 1.000 | 1.000 |
| pre | 1 | 0.018 | 0.458 | 4.63 | -0.001 | 0.982 | 1.000 |
| pre | 7 | 0.064 | 0.804 | 11.50 | 0.028 | 0.948 | 0.997 |
| pre | 14 | 0.135 | 0.817 | 23.13 | 0.022 | 0.843 | 0.974 |
| pre | 21 | 0.099 | 0.816 | 11.13 | 0.007 | 0.935 | 0.978 |
| pre | 27 | 0.054 | 0.742 | 24.13 | -0.001 | 0.895 | 0.949 |
| post | 0 | 0.244 | 0.868 | 29.13 | 0.013 | 0.934 | 1.000 |
| post | 14 | 0.498 | 0.902 | 30.63 | 0.003 | 0.559 | 0.896 |
| post | 27 | 0.507 | 0.922 | 32.88 | 0.050 | 0.591 | 0.883 |

Interpretation: Qwen3 passes the architectural contrast cleanly (`k_pre` layer 0 zero, `k_post` layer 0 one), and then develops strong pre-RoPE position decodability at depth. Family A slot-level `k_pre` ridge R² rises from zero to ~0.91 at layer 1 and ~0.95–0.97 through most later layers, while the pre-RoPE `position_fraction` remains much lower than post-RoPE (`~0.086` vs `~0.428` overall for slot Family A). This confirms P1.5.c for a second RoPE model: stamped-position models also compute/leak position internally.

P1.5.e is **supported at its pre-registered comparison**, which is the **layer-0** stamped fraction (addendum §0 references the position-fraction-@-L0 table: qwen3 `k_post` 0.193 < pythia 0.385 < gpt2 0.649). At L0 this run gives Qwen3 Family A slot-level `k_post` position fraction `0.242` vs Pythia `0.381` — the same ordering (qwen3 weakest), as θ=1e6 predicts. Pythia's L0 reproduces the §0 value almost exactly (0.381 vs 0.385); Qwen3's is a little higher than the M1 reanalysis (0.242 vs 0.193) but the ordering holds. The earlier read that P1.5.e "fails" used the **depth-averaged** fraction (qwen3 ~0.428 > pythia ~0.252), which is the wrong statistic here — it is confounded by the depth trend below.

**New, unpredicted finding — post-RoPE position fraction diverges by architecture with depth.** Qwen3 `k_post` position fraction *rises* from L0 (`0.24`) to a `~0.45–0.51` plateau by mid-stack (`0.507` at L27), while ridge R² stays ≈1.0 (position always perfectly decodable). Pythia `k_post` instead *falls* with depth (`0.38 → 0.07`), and GPT-2 also falls (`0.72 → 0.42`). So full RoPE (θ=1e6, all 128 dims) starts with the **weakest** stamp yet **accumulates** position into the cached key with depth, whereas partial-RoPE (pythia, θ=1e4) and learned-absolute (gpt2) **dilute** it. This is why the depth-averaged fraction reverses the L0 ordering — it is not evidence against P1.5.e but a distinct property of how full-RoPE propagates position through depth, and is a headline candidate for the cross-model M1.5 summary. Mechanism (QK-norm + θ=1e6 vs partial RoPE) is untested here and should be probed before a paper claim.

Dimensionality/projector nuance: at the slot level, Qwen3 `k_pre` position is still fairly low-dimensional in most depths (about 1–4 PCs at selected Family A layers), broadly matching the Pythia/NoPE slot-level caveat. The aggregate rows are much higher-dimensional (Family A `k_pre` selected depths reach ~11–24 PCs), which reproduces the aggregate-vs-slot trap: aggregate projectors are a different and stricter operator than per-slot projectors. Projection usually removes most aggregate Family A position (`r2_after_projection` near 0), while slot-level early-layer `k_pre` leaves residual R² (e.g. layer 1 ~0.208), so Π fidelity remains scope-dependent.

Artifacts copied locally:

- `outputs/k_address_space_m15_v11_qwen3_cuda_20260722/`
- `outputs/k_address_space_m15_v11_qwen3_cuda_20260722.tar.gz`
- `outputs/SHA256SUMS_k_address_space_m15_v11_qwen3_cuda_20260722`

## 2026-07-22 — K-address-space M1.5 v1.1 Pythia-410m CUDA run prep

### Question / Hypothesis

Does Pythia-410m expose both stamped RoPE position (`k_post`, including layer 0) and computed/leaked positional information in pre-RoPE keys (`k_pre`, especially at depth) under the corrected M1.5 v1.1 repeated-segment probe? The expected M1.5 signal is that `k_pre` at layer 0 satisfies the architectural-zero gate (G1), `k_post` at layer 0 satisfies the architectural-one gate (G2), and deeper `k_pre` rows adjudicate whether RoPE models compute position internally rather than merely carrying the architectural stamp.

### Experiment Design Summary

Run `kaddress.scripts.position_content` for model tag `pythia410` (`EleutherAI/pythia-410m`) on a RunPod CUDA GPU using all families A/B/C. Because Pythia has `d_head=64`, the effective minimum repetitions are `R_min=max(120, 2*d_head)=128`; its trained context budget supports segment lengths `L=4,7,12`, so this run includes the L=12 cell in addition to the mandatory cross-model L=7 and the second-length L=4 control. The run uses the repaired v1.1 gates and manifest semantics from ADDENDUM §3.

### Planned Procedure

Prepare and commit this pre-run notebook entry, bring up a RunPod GPU using the project template/shared network-volume cache, verify CUDA before the full run, then execute:

```bash
PYTHONPATH=experiments/dead-keys:experiments/k-address-space ./scripts/cuda-run -m kaddress.scripts.position_content \
  --model pythia410 \
  --device cuda \
  --families A,B,C \
  --segment-lengths 4,7,12 \
  --output-dir outputs/k_address_space_m15_v11_pythia410_cuda_20260722
```

Package the output directory as `k_address_space_m15_v11_pythia410_cuda_20260722.tar.gz`, publish it with a `SHA256SUMS` file to GitHub Release `run/k-address-space-m15-v11-pythia410/20260722`, verify the uploaded assets, then complete this entry with run results and analysis.

### Expected Signal / Interpretation Plan

The run is valid only if it used CUDA, emitted non-empty measurement and gate CSVs, `gate_g1_pass=PASS` for applicable layer-0 `k_pre` rows, `gate_g2_pass=PASS` for applicable layer-0 `k_post` rows, `gates_evaluated` matches the gates CSV, and the shuffled-null gate remains acceptable. Interpretation focuses on cross-depth `k_pre` aggregate/slot R² and position fraction: a rise from layer-0 zero would support P1.5.c that RoPE models compute or leak position internally, while `k_post` provides the stamped-position comparison.

### Pre-run Provenance

- Spec: `experiments/k-address-space/addendum-M1.5.md` v1.1
- Code branch: `docs/f8-caveats-m15-report-absorb`
- Pre-run commit: `947ed85` (`Prepare Pythia M1.5 CUDA run`)
- Corrected run commit: `99257e9` (`Speed up M1.5 CUDA ridge analysis`)
- Planned output location: `outputs/k_address_space_m15_v11_pythia410_cuda_20260722`
- Published output: https://github.com/vhallac/crockpot-experiments/releases/tag/run/k-address-space-m15-v11-pythia410/20260722
- Published assets:
  - `k_address_space_m15_v11_pythia410_cuda_20260722.tar.gz`
  - `SHA256SUMS_k_address_space_m15_v11_pythia410_cuda_20260722`
- SHA256: `6d23a8b0cf89dc84638ea80a55d35de87fa40552febd25989a7d295d8658a324  k_address_space_m15_v11_pythia410_cuda_20260722.tar.gz`
- Random seed: default script seed `0`
- Environment: RunPod pod `33cxmm98cdwse2`, NVIDIA L4, Python 3.12.3, Torch 2.8.0+cu128, CUDA 12.8, `cuda_available=true`, requested device `cuda`
- Model: `pythia410` (`EleutherAI/pythia-410m`, default Hugging Face revision)
- Preparation checklist: `temp/repro-checklists/20260722-k-address-space-m15-v11-pythia410-cuda.md`

### Results

First full CUDA attempt was aborted by operator request because it was CPU-bound and opaque.

Attempted command:

```bash
PYTHONPATH=experiments/dead-keys:experiments/k-address-space ./scripts/cuda-run -m kaddress.scripts.position_content \
  --model pythia410 \
  --device cuda \
  --families A,B,C \
  --segment-lengths 4,7,12 \
  --output-dir outputs/k_address_space_m15_v11_pythia410_cuda_20260722
```

Failure evidence: remote PID `1789` on RunPod pod `j2001wsvr7vimt` was stopped at `20260722T084434Z` after roughly 90 minutes without completion/progress visibility beyond coarse stimulus-level output. No completed result is published from this attempt.

Diagnosis: `position_content.py` captured Pythia keys on CUDA, but the dominant M1.5 analysis immediately converted every slot/head/layer matrix with `.cpu().numpy()` and then ran NumPy ridge regression, SVD/PCA, FFT, and permutation null loops on CPU. This invalidated the CUDA-run assumption even though model extraction used GPU.

Redo plan: patch the hot per-slot analysis path to use `torch` operations on CUDA tensors, add frequent progress lines (`--progress-every`), commit the fix, run a CUDA smoke test, then restart the full Pythia run from the new committed state.

Second CUDA attempt after the first GPU fix was also stopped by operator request at `20260722T100224Z`: remote PID `1128` on RunPod pod `33cxmm98cdwse2` had run for `46:39` and reached only `progress units=12000 rate=4.31/s`, extrapolating beyond the intended budget. Diagnosis: the code no longer ran the main regression through NumPy, but the torch path still launched many tiny ridge/null solves and scalar synchronizations per slot/head/layer unit. Corrective action: batch ridge cross-validation across alphas and shuffled null targets on the device, defer CPU copies until after per-unit CUDA analysis, then run a bounded CUDA smoke before restarting the full run.

Corrected CUDA rerun completed on pod `33cxmm98cdwse2` after the batching fix. Command:

```bash
PYTHONPATH=experiments/dead-keys:experiments/k-address-space python -m kaddress.scripts.position_content \
  --model pythia410 \
  --device cuda \
  --families A,B,C \
  --segment-lengths 4,7,12 \
  --progress-every 1000 \
  --output-dir outputs/k_address_space_m15_v11_pythia410_cuda_20260722
```

Run evidence: the log reports `starting M1.5 analysis model=pythia410 device=cuda stimuli=26 families=A,B,C`, progress rising to `progress units=144000 rate=56.22/s`, `processed C00 family=C seq=1955 slots=1 units=144384`, and writes all expected output files. The manifest records `summary_rows=146688`, `gate_g1_pass=PASS`, `gate_g2_pass=PASS`, `shuffle_null_ok=true`, and `gates_evaluated={"G1_architectural_zero": 3008, "G2_architectural_one": 3008}`. The gates CSV has exactly 6,016 rows matching those manifest counts; all G1 and G2 rows pass. G2's perturbation check can fail: perturbed ridge R² ranges from about `-0.1636` to `0.0209`, far below the 0.9 threshold.

Published output: https://github.com/vhallac/crockpot-experiments/releases/tag/run/k-address-space-m15-v11-pythia410/20260722. The uploaded checksum asset was downloaded and compared byte-for-byte with the local `SHA256SUMS` file.

### Analysis

The validity gates behave as intended for a RoPE model. Layer-0 `k_pre` has architectural zero position signal (`ridge_r2=0`, `position_fraction=0`) and all 3,008 G1 rows pass. Layer-0 `k_post` has the expected stamped RoPE position signal: G2 rows have ridge R² essentially 1.0, and the perturbation destroys the signal.

The substantive result is mixed but informative. In `k_post`, position is almost perfectly linearly decodable at every layer (`ridge_r2` mean about `0.9981` overall), as expected from stamped rotary position. The `position_fraction` diagnostic declines with depth: from about `0.3824` at layer 0 to `0.0696` at layer 23, meaning the position-decodable direction remains present but occupies a smaller share of variance in later post-RoPE keys.

For `k_pre`, layer 0 is exactly zero, but deeper layers show high position decodability by ridge R²: layer means jump to `0.9255` at layer 1, range around `0.82–0.99` through the stack, and end around `0.9610` at layer 23. However, the `position_fraction` remains much smaller than `k_post`: overall about `0.0357`, peaking by layer mean near `0.0829` at layer 8 and then falling below `0.02` in the last layers. This supports the hypothesis that Pythia's pre-RoPE key stream contains computed/leaked positional information after layer 0, but it is a low-variance component rather than the dominant geometry of the key vectors.

Family-level aggregates from the run log are consistent with that interpretation: `pre` position fraction is about `0.0329` for Family A, `0.0976` for Family B, and `0.1740` for Family C, while `post` remains much larger (`0.2519`, `0.2663`, `0.3654`) and ridge R² remains near 1.

The pre-RoPE signal is real, not a small-sample fluke: null-corrected `r2_minus_null_mean` for `k_pre` is about `+0.95` to `+1.02` at every depth L1–L23, with the shuffled-null mean sitting at about `−0.03`. (Note: the `permutation_p_value` column is quantised at `0.1667` throughout — too few permutations to ever cross significance — so `r2_minus_null` and the shuffled null, not the p-value, carry the evidence. Worth increasing the permutation count before any paper-facing significance claim.)

**Added post-writeup: dimensionality and projector, computed at slot level.** These bear on the addendum's P1.5.d/P1.5.f and on the Π deliverable for an M1 rerun. **Metric caveat first:** the striking NoPE figures on record — aggregate-projector "~13 PCs to 90% at L15–23" (P1.5.d) and "L23 ridge R² 0.907 → 0.005 after projection" (P1.5.f) — are **aggregate** statistics. Recomputed **like-for-like at the slot level**, Pythia and NoPE behave similarly and those dramatic numbers do not appear.

Pythia `k_pre` slot-level depth profile (all families):

| layer | ridge R² | position fraction | PCA k90 | R² after PC projection |
|---:|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.00 | 0.000 |
| 1 | 0.926 | 0.019 | 1.88 | +0.204 |
| 4 | 0.910 | 0.044 | 1.85 | +0.271 |
| 8 | 0.990 | 0.083 | 2.67 | −0.006 |
| 12 | 0.985 | 0.043 | 3.25 | −0.001 |
| 15 | 0.985 | 0.034 | 3.24 | −0.002 |
| 18 | 0.982 | 0.019 | 2.74 | +0.037 |
| 23 | 0.961 | 0.010 | 2.58 | +0.076 |

- **Dimensionality (P1.5.d).** At slot level Pythia `k_pre` position stays low-dimensional throughout (~2–3 PCs to 90%), peaking mid-stack (L12–15) and *not* expanding — if anything contracting — in the top layers. NoPE `k_pre` slot-level is similarly compact (~1 PC through L12, rising only to ~2.6 by L18–23). Neither shows a ~13-component blow-up; that figure is aggregate-only. So P1.5.d ("computed position is higher-dimensional") is **not supported at the slot level** on either model, and the "two-regime split at ~L15" flagged as an M1.6 target is an aggregate-projector artifact rather than a per-slot geometric fact.
- **Projector Π fidelity (P1.5.f).** The per-slot position-removal projector is only partially effective at early/mid layers for **both** models — Pythia leaves R² `+0.20 / +0.27` at L1/L4; NoPE leaves `+0.33 / +0.21` at L4/L8 — and removes most position by the upper layers (Pythia L23 `0.96 → +0.076`; NoPE L23 `0.97 → +0.028`). Π's early-layer imperfection is therefore general, not Pythia-specific, and it is cleanest in the middle depths where M1 address heads are expected. The clean "0.91 → 0.005" on record is the aggregate projector (pooled across slots/L), a stronger and different operator than the per-slot basis; an M1 rerun that relies on Π should specify which of the two it uses.

### Conclusion / Next Step

The Pythia-410m M1.5 v1.1 CUDA run is complete and published. The result validates the repaired M1.5 gates on a RoPE model and gives positive evidence for nonzero, internally present pre-RoPE positional information beyond layer 0, but with small variance share compared with the explicit post-RoPE positional stamp. This is the first **stamped-position** model to show it: Pythia already carries RoPE position in `k_post`, yet still computes/leaks decodable position into `k_pre` at depth — so P1.5.c is affirmative here, extending the NoPE result beyond the no-positional-encoding case. The decodability-vs-variance dissociation (high `k_pre` R², low position fraction) and the depth-wise dilution of the stamped `k_post` fraction (mirroring GPT-2's learned-absolute pattern) also replicate.

Two adjudications tighten on re-analysis: P1.5.d is **not supported at the slot level** (position stays ~2–3 dimensional; the ~13-PC expansion is aggregate-only), and P1.5.f's projector is only middling per-slot at early/late layers on both models. Next: (a) run qwen3 for P1.5.e (stamped fraction expected weaker than Pythia, θ=1e6 vs 1e4); (b) build the cross-model M1.5 summary over NoPE/GPT-2/Pythia (+qwen3) using slot-level metrics consistently; (c) before any paper-facing significance claim, raise the permutation count (currently p-value floored at 0.167) and decide whether the Π deliverable is the aggregate or per-slot projector.

## 2026-07-21 — K-address-space M1.5 v1.1 GPT-2 gatefix CPU rerun prep

### Question / Hypothesis

Repair D6a/D6b in the M1.5 v1.1 gate implementation and re-run the GPT-2 CPU experiment. The prior GPT-2 output `outputs/k_address_space_m15_v11_gpt2_cpu_20260721` measured the intended statistics but wrote a 1-byte, zero-row `kaddress_m15_gates_gpt2.csv` while reporting `gate_g1_pass=true`; GPT-2 has no applicable G1 architectural-zero gate, and the missing G2 architectural-one gate meant the run had no functioning architectural gate. The expected result after the fix is that GPT-2 layer-0 keys emit `G2_architectural_one` rows with ridge R² ≥ 0.9 and a perturbation check that can fail, while the measurement CSV remains unchanged.

### Experiment Design Summary

Implement ADDENDUM §5-M1.5 v1.1 §3 gate semantics in `kaddress.scripts.position_content`: G1 remains the architectural-zero gate for NoPE and RoPE `k_pre` layer-0 cases; G2 applies to stamped positional encodings detected from model config (GPT-2 learned absolute position and RoPE `k_post` layer-0 cases); gate manifest fields become `PASS` / `FAIL` / `NOT_APPLICABLE`; `gates_evaluated` records row counts; and a run with no applicable architectural gates fails loudly instead of reporting green.

### Planned Procedure

Run local unit tests plus GPT-2/NoPE smoke checks, commit the gatefix pre-run state, then re-run the full GPT-2 CPU experiment:

```bash
PYTHONPATH=experiments/dead-keys:experiments/k-address-space ./scripts/nix-cpu-run -m kaddress.scripts.position_content \
  --model gpt2 \
  --families A,B,C \
  --output-dir outputs/k_address_space_m15_v11_gpt2_cpu_20260721_gatefix
```

After the run, compare `kaddress_m15_gpt2.csv` against the prior GPT-2 run byte-for-byte or by SHA256 to verify the regression bar that only gate reporting changed.

### Expected Signal / Interpretation Plan

The rerun is valid only if `gate_g2_pass=PASS`, `gate_g1_pass=NOT_APPLICABLE`, `gates_evaluated.G2_architectural_one` equals the number of G2 rows in the gates CSV, and every G2 row has `pass=true` and `perturbation_can_fail=true`. The prior scientific interpretation should stand if `kaddress_m15_gpt2.csv` is unchanged.

### Pre-run Provenance

- Spec: `experiments/k-address-space/addendum-M1.5.md` v1.1 §3
- Code branch: `main`
- Pre-run commit: `946830c`
- Planned output location: `outputs/k_address_space_m15_v11_gpt2_cpu_20260721_gatefix`
- Prior defective output: `outputs/k_address_space_m15_v11_gpt2_cpu_20260721`
- Publication target: GitHub Release `run/k-address-space-m15-v11-gpt2/20260721` replacement/additional gatefix tarball
- Random seed: default script seed `0`
- Environment: local CPU via `scripts/nix-cpu-run`; exact manifest environment to be recorded at run time
- Model: `gpt2` (Hugging Face model id `gpt2`)
- Preparation checklist: `temp/repro-checklists/20260721-k-address-space-m15-v11-gpt2-gatefix-cpu.md`

### Results

Gatefix rerun completed locally on CPU from pre-run commit `946830c`.

Command:

```bash
PYTHONPATH=experiments/dead-keys:experiments/k-address-space ./scripts/nix-cpu-run -m kaddress.scripts.position_content \
  --model gpt2 \
  --families A,B,C \
  --output-dir outputs/k_address_space_m15_v11_gpt2_cpu_20260721_gatefix
```

Run window: `20260721T164434Z` to `20260721T165452Z`; exit code 0.

Outputs were published as additional GitHub Release assets on the existing GPT-2 release:

- Release: <https://github.com/vhallac/crockpot-experiments/releases/tag/run/k-address-space-m15-v11-gpt2/20260721>
- Artifact: <https://github.com/vhallac/crockpot-experiments/releases/download/run/k-address-space-m15-v11-gpt2/20260721/k_address_space_m15_v11_gpt2_cpu_20260721_gatefix.tar.gz> — 13,213,743 bytes; SHA256 `0d14c2a0336c1c6ecf8d8cc5a83efa32aec9489f0d1c0dd2eabf7375396573ae`
- Checksums: <https://github.com/vhallac/crockpot-experiments/releases/download/run/k-address-space-m15-v11-gpt2/20260721/SHA256SUMS-m15-v11-gpt2-gatefix-20260721.txt> — 883 bytes; checksum file verified by re-download and byte comparison.

Internal output checksums:

- `kaddress_m15_gates_gpt2.csv` — SHA256 `94daa9f573152c229f0edc70d8137dea543dfbc12d4976cdbb7f733154375afb`
- `kaddress_m15_gpt2.csv` — SHA256 `e76fea1d3ac0aa4d8a4cd57846970645715fe7d99b74458222a6c48ec93f9eec`
- `kaddress_m15_manifest_gpt2.json` — SHA256 `931f752910f76a2ac55552761e64020dfe6102a7404c99875033743653af878f`
- `kaddress_m15_projectors_gpt2.npz` — SHA256 `8c0a346749e969e17d17a21f0eef38a3e6193d16a8685a9609000d9c6734cefb`
- `run.log` — SHA256 `34f60414a896dd61d289b5fd816ccace978bed14799361d7013abe77f44a5672`

Manifest highlights: `stimulus_count=19`, `summary_rows=14112`, `families=[A,B,C]`, `trained_context=1024`, `max_length=992`, `min_repetitions=128`, `segment_lengths=[4,7]`, `requested_device=cpu`, `cuda_available=false`, Python `3.11.11`, Torch `2.5.1`.

Gate results now behave as intended:

- `gate_g1_pass=NOT_APPLICABLE`
- `gate_g2_pass=PASS`
- `gates_evaluated={G1_architectural_zero: 0, G2_architectural_one: 1140}`
- Gates CSV row count: 1,140, all `G2_architectural_one`
- Every G2 row has `pass=true` and `perturbation_can_fail=true`
- G2 ridge R² min/mean: `0.907914 / 0.999073`
- Maximum perturbed ridge R²: `0.039493`, below the `0.9` G2 threshold

Regression check: the rerun `kaddress_m15_gpt2.csv` is byte-identical to the prior GPT-2 run (`e76fea1d3ac0aa4d8a4cd57846970645715fe7d99b74458222a6c48ec93f9eec`, 4,053,422 bytes). This confirms the fix changed gate reporting only, not the measurement table.

### Analysis

The previous GPT-2 run's scientific measurements were stable, but its gate result was invalid: it had no G1 rows because GPT-2 has no architectural-zero layer-0 key case, and the required G2 architectural-one gate was not implemented. Reporting `gate_g1_pass=true` on an empty gates table was therefore a false green.

The gatefix rerun repairs that defect. GPT-2 now validates through G2 exactly where expected for learned absolute positional embeddings: all layer-0 slot/head rows pass the `ridge_r2 >= 0.9` threshold, with the minimum matching the expected reference range (`0.9079`). The perturbation check also demonstrates that the gate can fail: shuffling the position→key correspondence drives every perturbed ridge R² far below the threshold.

Because the measurement CSV is byte-identical to the prior run, the prior interpretation of GPT-2's strong layer-0 positional signal and depth-wise persistence remains unchanged. The durable correction is limited to gate semantics and manifest visibility.

### Conclusion / Next Step

The GPT-2 M1.5 v1.1 result is now valid under ADDENDUM §3 gates. The earlier release asset is retained as the defective-gate historical output; the gatefix tarball should be used for gate-validity claims. Next step remains the RoPE stamped-vs-computed comparison on Pythia/Qwen3 using the same gate semantics.

## 2026-07-21 — K-address-space M1.5 v1.1 GPT-2 CPU run prep

### Question / Hypothesis

Does GPT-2, with learned absolute positional embeddings, expose strong positional information in attention keys under the corrected M1.5 v1.1 repeated-segment probe? The expectation from addendum v1.1 is that GPT-2 layer-0 keys should satisfy the architectural-one gate (G2) with high position decodability, unlike NoPE layer-0 architectural zero, while deeper layers may mix stamped and computed position.

### Experiment Design Summary

Run the already-corrected `kaddress.scripts.position_content` implementation from `experiments/k-address-space/addendum-M1.5.md` v1.1 for Hugging Face `gpt2` on local CPU. The run will use all families A/B/C, default effective `R_min = max(120, 2*d_head) = 128`, default trained context minus 32 token budget, and the mandatory feasible `L ∈ {4, 7}` cells. L=12 is known infeasible for GPT-2 under the v1.1 budget and is not requested.

### Planned Procedure

Run local verification and a constrained GPT-2 smoke check, commit the pre-run notebook state, then run the full GPT-2 CPU experiment from the committed state:

```bash
PYTHONPATH=experiments/dead-keys:experiments/k-address-space ./scripts/nix-cpu-run -m kaddress.scripts.position_content \
  --model gpt2 \
  --families A,B,C \
  --output-dir outputs/k_address_space_m15_v11_gpt2_cpu_20260721
```

### Expected Signal / Interpretation Plan

G2 should pass for GPT-2 layer 0, with layer-0 positional information high enough to contrast with the NoPE G1 architectural-zero result. Family A is primary. Family B must be non-empty and is an induction-control corroboration; Family C remains external-validity corroboration only. The corrected one-sided shuffled-null gate should pass or be reported directly if positive null tails exceed threshold. Aggregate rows are the adjudication basis; slot-level rows are descriptive diagnostics.

### Pre-run Provenance

- Spec: `experiments/k-address-space/addendum-M1.5.md` v1.1
- Parent spec: `experiments/k-address-space/spec.md`
- Code branch: `main`
- Pre-run commit: `5710f96`
- Planned output location: `outputs/k_address_space_m15_v11_gpt2_cpu_20260721`
- Publication target: GitHub Release `run/k-address-space-m15-v11-gpt2/20260721`
- Random seed: default script seed `0`
- Environment: local CPU via `scripts/nix-cpu-run`; exact manifest environment recorded below
- Model: `gpt2` (Hugging Face model id `gpt2`)
- Preparation checklist: `temp/repro-checklists/20260721-k-address-space-m15-v11-gpt2-cpu.md`

### Results

Run completed locally on CPU from pre-run commit `5710f96`.

Command:

```bash
PYTHONPATH=experiments/dead-keys:experiments/k-address-space ./scripts/nix-cpu-run -m kaddress.scripts.position_content \
  --model gpt2 \
  --families A,B,C \
  --output-dir outputs/k_address_space_m15_v11_gpt2_cpu_20260721
```

Run window: `20260721T162109Z` to `20260721T163113Z`; exit code 0.

Outputs were published as GitHub Release assets:

- Release: <https://github.com/vhallac/crockpot-experiments/releases/tag/run/k-address-space-m15-v11-gpt2/20260721>
- Artifact: <https://github.com/vhallac/crockpot-experiments/releases/download/run/k-address-space-m15-v11-gpt2/20260721/k_address_space_m15_v11_gpt2_cpu_20260721.tar.gz> — 13,184,611 bytes; SHA256 `43d35742f6d51ef0f9621b0f5f95a8e975de9cd9802bbed62f6fe2824aca9dc6`
- Checksums: <https://github.com/vhallac/crockpot-experiments/releases/download/run/k-address-space-m15-v11-gpt2/20260721/SHA256SUMS-m15-v11-gpt2-20260721.txt> — 869 bytes; checksum file verified by re-download and byte comparison.

Internal output checksums:

- `kaddress_m15_gates_gpt2.csv` — SHA256 `01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b`
- `kaddress_m15_gpt2.csv` — SHA256 `e76fea1d3ac0aa4d8a4cd57846970645715fe7d99b74458222a6c48ec93f9eec`
- `kaddress_m15_manifest_gpt2.json` — SHA256 `ef7bc82fb39068109d0a777cd8b66fff6b8025c599347c1d6c6b72e5652b47a1`
- `kaddress_m15_projectors_gpt2.npz` — SHA256 `8c0a346749e969e17d17a21f0eef38a3e6193d16a8685a9609000d9c6734cefb`
- `run.log` — SHA256 `3cbfb1508671107d2705d420d7aa25a7efc6bc329f1240546ca8164b6a2629f6`

Manifest highlights: `stimulus_count=19`, `summary_rows=14112`, `families=[A,B,C]`, `trained_context=1024`, `max_length=992`, `min_repetitions=128`, `segment_lengths=[4,7]`, `rejected_stimuli=[]`, `requested_device=cpu`, `cuda_available=false`, Python `3.11.11`, Torch `2.5.1`. Family A contributed 16 stimuli (8 at L=4 and 8 at L=7), Family B contributed 2 frame/content-varying stimuli, and Family C contributed 1 natural-recurrence stimulus.

Layer-0 GPT-2 positional signal is strong, as expected for learned absolute position embeddings. Slot-level layer-0 ridge R² minima/means/maxima were: Family A `0.908 / 0.999 / 1.000`, Family B `0.995 / 1.000 / 1.000`, Family C `0.998 / 1.000 / 1.000`. The script's `gates` CSV is empty for GPT-2 because the implemented perturbation gate is the NoPE G1 architectural-zero gate; the G2 architectural-one condition was therefore adjudicated from the layer-0 ridge statistics above.

Selected Family A slot-level means:

| layer | position fraction | ridge R² | PCA k90 | R² after PC projection |
|---:|---:|---:|---:|---:|
| 0 | 0.718 | 0.999 | 2.91 | -0.037 |
| 1 | 0.672 | 0.999 | 3.08 | -0.016 |
| 2 | 0.819 | 0.998 | 2.92 | -0.026 |
| 6 | 0.559 | 0.997 | 2.87 | -0.055 |
| 11 | 0.431 | 0.996 | 3.37 | -0.030 |

Selected aggregate means show the multi-slot aggregate estimator also remains strongly position-decodable, though Family A layer-0 aggregate ridge R² is lower than the slot-level G2 statistic because aggregate pooling mixes slots/tokens before fitting: Family A aggregate R² is `0.838` at layer 0 and `0.958` at layer 11; Family B aggregate R² is `0.999` at layer 0 and `0.994` at layer 11.

Corrected shuffled-null summary: `shuffle_null_ok=true`; the all-row 99th percentile of per-row shuffled-null positive tails was `0.0467`, below the `+0.05` gate threshold, though the maximum individual row reached `0.0969` and is treated as a descriptive outlier rather than a gate failure.

### Analysis

GPT-2 behaves like the learned-absolute-position endpoint expected by M1.5. Unlike NoPE, it does not have a layer-0 architectural zero: positional information is already nearly perfectly decodable from slot-level keys at layer 0 across Families A/B/C. This directly supports the addendum's premise that learned absolute position is present as a stamped input signal before depth-wise computation.

The position signal remains strong throughout the network. Family A slot-level ridge R² stays around `0.996–0.999` across the sampled layers, while the position fraction declines from about `0.72` at layer 0 to about `0.43` by layer 11. This suggests GPT-2 keeps position highly decodable even as the relative amount of key variance attributable to absolute position decreases with depth.

The position component is low-dimensional and removable under the implemented PCA projector diagnostic: Family A slot-level PCA k90 stays around 3 components, and R² after position-PC projection is near zero or slightly negative at the selected layers. The aggregate projector is stricter and leaves more residual position signal for Family A (`~0.08–0.19` at selected layers), consistent with the NoPE v1.1 caveat that aggregate multi-L/multi-slot removal is harder than slot-level removal.

Family B is non-empty and agrees in sign with Family A, so the GPT-2 finding is not limited to the degenerate identical-segment induction regime. Family C also corroborates strong recurrence-position decodability, with the expected caveat that natural recurrence is confounded and cannot adjudicate against Family A.

### Conclusion / Next Step

The GPT-2 CPU M1.5 v1.1 run is valid. It confirms the architectural-one contrast to NoPE: GPT-2 layer-0 keys already carry strong learned absolute positional signal, and position remains strongly decodable through all layers. Next step: run the same corrected v1.1 protocol on Pythia/Qwen3 RoPE models for the stamped-vs-computed comparison.

## 2026-07-21 — K-address-space M1.5 v1.1 NoPE-GPT-Small CPU rerun prep

### Question / Hypothesis

Does the corrected M1.5 v1.1 repeated-segment probe reproduce the NoPE-GPT-Small Family A depth profile while also exercising the formerly empty Family B induction-control family and the corrected length sweep (`L ∈ {4, 7}`)?

### Experiment Design Summary

This rerun implements the v1.1 corrections in `experiments/k-address-space/addendum-M1.5.md` and the compatible local code-fix brief in `experiments/k-address-space/CODEFIX-M1.5.md` without committing that brief. The implementation removes Family B equal-length filtering, computes slot positions from cumulative offsets, derives `R_min = max(120, 2*d_head)`, defaults the token budget to trained context minus 32, emits feasibility/rejection manifest fields, changes the shuffled-null gate to an upper-tail check, and applies the variance floor to every derived position statistic.

Compatibility check: the code-fix D1–D5 items correspond directly to addendum v1.1 C1–C4 / §§2.0, 2.2, 3, 4.1, and 7.5. The implementation keeps CODEFIX as an untracked disposable file and treats the addendum as the durable spec.

### Planned Procedure

Run tests and a CPU smoke check, commit the pre-run state, then run the full corrected NoPE CPU experiment from the committed state:

```bash
PYTHONPATH=experiments/dead-keys:experiments/k-address-space ./scripts/nix-cpu-run -m kaddress.scripts.position_content \
  --model nope-gpt-small \
  --revision 320681e33a029517e27c68a0f9c2b07ea0004155 \
  --families A,B,C \
  --output-dir outputs/k_address_space_m15_v11_nope_gpt_small_cpu_20260721
```

### Expected Signal / Interpretation Plan

The regression bar is that Family A should remain close to the previous NoPE depth profile, while Family B must be non-empty. G1 must pass at layer 0, the corrected one-sided shuffled-null gate should avoid treating negative null R² as leakage, and degenerate rows should carry zero/NaN derived position statistics with `degenerate=true`. If Family A changes materially, the cumulative-offset or length-sweep refactor may have changed the measurement rather than only fixing controls.

### Pre-run Provenance

- Spec: `experiments/k-address-space/addendum-M1.5.md` v1.1
- Code-fix brief: `experiments/k-address-space/CODEFIX-M1.5.md` (untracked disposable file; do not commit)
- Code branch: `main`
- Pre-run commit: _Pending_
- Planned output location: `outputs/k_address_space_m15_v11_nope_gpt_small_cpu_20260721`
- Publication target: GitHub Release `run/k-address-space-m15-v11-nope-gpt-small/20260721`
- Random seed: default script seed `0`
- Environment: local CPU via `scripts/nix-cpu-run`; exact manifest environment to be recorded at run time
- Model: `andrewdalpino/NoPE-GPT-Small-Base` pinned at Hugging Face revision `320681e33a029517e27c68a0f9c2b07ea0004155`
- Preparation checklist: `temp/repro-checklists/20260721-k-address-space-m15-v11-nope-gpt-small-cpu.md`
- Local verification: py_compile passed; unittest suite passed; NoPE A/B one-layer/one-head smoke passed with Family B non-empty and `shuffle_null_ok=True`.

### Results

Run completed locally on CPU from pre-run commit `133f115`.

Command:

```bash
PYTHONPATH=experiments/dead-keys:experiments/k-address-space ./scripts/nix-cpu-run -m kaddress.scripts.position_content \
  --model nope-gpt-small \
  --revision 320681e33a029517e27c68a0f9c2b07ea0004155 \
  --families A,B,C \
  --output-dir outputs/k_address_space_m15_v11_nope_gpt_small_cpu_20260721
```

Run window: `20260721T110903Z` to `20260721T113337Z`; exit code 0. The disposable code-fix brief `experiments/k-address-space/CODEFIX-M1.5.md` was deleted after the run completed, per instruction.

Outputs were published as GitHub Release assets:

- Release: <https://github.com/vhallac/crockpot-experiments/releases/tag/run/k-address-space-m15-v11-nope-gpt-small/20260721>
- Artifact: <https://github.com/vhallac/crockpot-experiments/releases/download/run/k-address-space-m15-v11-nope-gpt-small/20260721/k_address_space_m15_v11_nope_gpt_small_cpu_20260721.tar.gz> — 20,199,687 bytes; SHA256 `5511822789f285483bab43cd28e8a16cffcb1cf575e4becf77dd24fc03256512`
- Checksums: <https://github.com/vhallac/crockpot-experiments/releases/download/run/k-address-space-m15-v11-nope-gpt-small/20260721/SHA256SUMS-m15-v11-nope-20260721.txt> — 938 bytes; checksum file verified by re-download and byte comparison.

Internal output checksums:

- `kaddress_m15_nope-gpt-small.csv` — SHA256 `f00420e4ddfdbddc22b6a3f188cf6327d42e444c8ddf604a7b7f1237d5c6eb1e`
- `kaddress_m15_gates_nope-gpt-small.csv` — SHA256 `927ac35665c81eea3e172ff38e9c208c5739c84a0b36ea509e607096a2d4f69d`
- `kaddress_m15_manifest_nope-gpt-small.json` — SHA256 `6e3e455cb7e06466a639d2b0d9bbd3cb0e15b57a06a40ef188ebfe000ba67688`
- `kaddress_m15_projectors_nope-gpt-small.npz` — SHA256 `262c57585bb209413a94564962131ac2535d2e3b3d09654c84cdeeb70029afc6`
- `run.log` — SHA256 `bb4612f6cef2fd89224791231035d2b754c704d91088c39d376c050971f08320`

Manifest highlights: `stimulus_count=19`, `summary_rows=37632`, `families=[A,B,C]`, `trained_context=1024`, `max_length=992`, `min_repetitions=128`, `segment_lengths=[4,7]`, `rejected_stimuli=[]`, `requested_device=cpu`, `cuda_available=false`, Python `3.11.11`, Torch `2.5.1`, model revision `320681e33a029517e27c68a0f9c2b07ea0004155`. Family A contributed 16 stimuli (8 at L=4 and 8 at L=7), Family B contributed 2 frame/content-varying stimuli, and Family C contributed 1 natural-recurrence stimulus.

Gate and null summary:

- G1 architectural zero passed for all 1,520 layer-0 checked slots/heads.
- Corrected G4 passed: `shuffle_null_ok=true`; shuffled-null mean over all rows was `-0.0357`, and negative null values were no longer counted as leakage.
- Degenerate layer-0 rows were explicitly marked and had derived position statistics zeroed.

Selected Family A slot-level depth means:

| layer | position fraction | ridge R² | PCA k90 | R² after PC projection |
|---:|---:|---:|---:|---:|
| 0 | 0.000001 | 0.000 | 0.00 | 0.000 |
| 1 | 0.00729 | 0.042 | 0.82 | 0.041 |
| 2 | 0.00893 | 0.257 | 1.28 | 0.159 |
| 6 | 0.0280 | 0.582 | 1.66 | 0.242 |
| 12 | 0.0234 | 0.900 | 1.16 | 0.059 |
| 18 | 0.0558 | 0.947 | 2.08 | 0.074 |
| 23 | 0.0656 | 0.972 | 2.15 | 0.034 |

Selected aggregate means show Family B is now non-empty and strongly position-decodable at depth: Family B aggregate ridge R² rises from `0.000` at layer 0 to `0.828` at layer 6, `0.969` at layer 12, and `0.973` at layer 23.

### Analysis

The corrected run preserves the core NoPE result: layer-0 keys are effectively position-free, then position becomes strongly decodable from K with depth. Family A's depth curve is directionally consistent with the first run, though not byte-identical because v1.1 now uses both L=4 and L=7 cells and 16 surviving Family A stimuli instead of the previous accidental subset. The headline still holds: Family A ridge R² reaches about `0.90` by layer 12 and `0.97` by layer 23.

The major correction is successful: Family B is no longer empty. The offset-based frame-token construction yields two vocabulary-disjoint Family B stimuli with actual cumulative offsets, and Family B shows a similar depth-rising position signal. This removes the previous caveat that the induction-control family never ran.

The spec corrections also behave as intended. The feasibility matrix records L=4 and L=7 as feasible at `R_min=128`; no stimuli were silently rejected; the one-sided shuffled-null gate passes despite a negative null mean; and the widened variance floor prevents layer-0 PCA/noise artifacts from appearing as real position components.

Caveat: Family A aggregate projector fidelity is weaker than the earlier single-effective-L run at upper layers (e.g. aggregate Family A layer-23 R² after projection ≈ `0.163` rather than near zero). This is expected to be more stringent because v1.1 combines two segment lengths to break absolute-position vs repetition-index collinearity; it should be tracked in cross-model comparisons rather than treated as a failed reproduction.

### Conclusion / Next Step

The v1.1 NoPE CPU rerun is valid and supersedes the first M1.5 NoPE run for corrected-family and length-sweep claims. It confirms computed positional information in NoPE keys at depth, validates the repaired Family B induction control, and records a stricter multi-L projector-fidelity caveat for follow-up. Next step: use this v1.1 implementation for GPT-2/Pythia/Qwen3 cross-model M1.5 runs.

## 2026-07-21 — K-address-space M1.5 NoPE-GPT-Small position-content run prep

> **Absorbed report + superseded.** This is the **first (pre-v1.1)** NoPE M1.5 run. It is
> **superseded by the v1.1 rerun** above (2026-07-21, "M1.5 v1.1 NoPE-GPT-Small CPU rerun
> prep"), which adds the L=4/L=7 length sweep and a non-empty Family B. The standalone
> `REPORT-M1.5.md` (formerly at the experiment root) is folded in here — its verdict:
> *NoPE-GPT-Small has an architectural-zero key at layer 0, then develops strong decodable
> position in `k_pre` with depth under repeated-token Family A stimuli.* Its gates/caveats
> (G1 pass at ~1e-6; `shuffle_null_ok=false` at slot level but clean aggregate nulls;
> Family B empty; Family C stronger but confounded) are recorded in the Results/Analysis
> below. For a durable summary, prefer the v1.1 entry.

### Question / Hypothesis

Does `andrewdalpino/NoPE-GPT-Small-Base` compute positional information into attention keys at depth when token content is held constant by repeated-segment stimuli? The primary prediction is that layer-0 `k_pre` is an architectural zero, while deeper layers develop measurable position fraction, ridge decodability, and a position subspace that may be difficult to remove without harming token identity.

### Experiment Design Summary

Implement and run Addendum §5-M1.5 for the selected NoPE model. The new `kaddress.scripts.position_content` script generates repeated-token stimuli for Families A/B/C, extracts per-layer/per-head keys with the existing NoPE hook path, and reports M1.5.1–M1.5.6 diagnostics per stimulus/slot/head: position fraction, ridge CV R² with variance-floor guard, PCA capacity, leading-PC geometry, shuffled-y null, and position-PC removability with token-identity retention. It also writes per-slot position-removal PCA bases as the initial Π projector artifact.

### Planned Procedure

Prepare and smoke-test locally, commit the pre-run state, then run the selected NoPE model with the pinned revision:

```bash
PYTHONPATH=experiments/dead-keys:experiments/k-address-space ./scripts/nix-cpu-run -m kaddress.scripts.position_content \
  --model nope-gpt-small \
  --revision 320681e33a029517e27c68a0f9c2b07ea0004155 \
  --families A,B,C \
  --min-repetitions 120 \
  --max-length 950 \
  --output-dir outputs/k_address_space_m15_nope_gpt_small_20260721
```

If local CPU runtime is poor, rerun the same command through `scripts/cuda-run` on RunPod with `--device cuda` and the shared CUDA venv.

### Expected Signal / Interpretation Plan

G1 must pass for NoPE layer 0: within-slot position fraction below `1e-5`, and the deliberate perturbation check must prove the gate can fail. Shuffled-y ridge R² should remain near zero. The headline will be the depth curve for Family A: if position fraction/R² rise with depth, the probe confirms key-level computed position in NoPE; if they stay flat, the result contradicts the pre-registered NoPE expectation. Family B/C are corroborating controls and must be reported as caveats if they disagree in sign with Family A.

### Pre-run Provenance

- Spec: `experiments/k-address-space/addendum-M1.5.md`
- Parent spec: `experiments/k-address-space/spec.md`
- Code branch: `main`
- Pre-run commit: `3fa27d0` (`Add aggregate M1.5 projector fidelity`; initial prep commit `54c11ac`)
- Planned output location: `outputs/k_address_space_m15_nope_gpt_small_20260721`
- Random seed: default script seed `0`
- Environment: local CPU via `scripts/nix-cpu-run` unless escalated to RunPod CUDA; exact manifest environment to be recorded at run time
- Model: `andrewdalpino/NoPE-GPT-Small-Base` pinned at Hugging Face revision `320681e33a029517e27c68a0f9c2b07ea0004155`
- Preparation checklist: `temp/repro-checklists/20260721-k-address-space-m15-nope-gpt-small.md`
- Local verification: `py_compile` passed; GPT-2 1-stimulus/1-layer/1-head smoke passed; NoPE 1-stimulus/1-layer/1-head smoke passed with G1 position fraction ≈ `1.27e-7` and ridge R² `0.0`.

### Results

Run completed locally on CPU from pre-run commit `3fa27d0`.

Command:

```bash
PYTHONPATH=experiments/dead-keys:experiments/k-address-space ./scripts/nix-cpu-run -m kaddress.scripts.position_content \
  --model nope-gpt-small \
  --revision 320681e33a029517e27c68a0f9c2b07ea0004155 \
  --families A,B,C \
  --min-repetitions 120 \
  --max-length 950 \
  --output-dir outputs/k_address_space_m15_nope_gpt_small_20260721
```

Outputs were published as GitHub Release assets:

- Release: <https://github.com/vhallac/crockpot-experiments/releases/tag/run/k-address-space-m15-nope-gpt-small/20260721>
- Artifact: <https://github.com/vhallac/crockpot-experiments/releases/download/run/k-address-space-m15-nope-gpt-small/20260721/k_address_space_m15_nope_gpt_small_20260721.tar.gz> — 8,133,533 bytes; SHA256 `00ab55541353b58f0bc099218cf4dd8494f33036f1ee3e15eaec40f96984eb1f`
- Checksums: <https://github.com/vhallac/crockpot-experiments/releases/download/run/k-address-space-m15-nope-gpt-small/20260721/k_address_space_m15_nope_gpt_small_20260721.SHA256SUMS> — 536 bytes.

Internal output checksums:

- `kaddress_m15_nope-gpt-small.csv` — SHA256 `c5b1bfa6894e2b6403c73f85e7d89f36b15597c113a982e8a215ecb94418e136`
- `kaddress_m15_gates_nope-gpt-small.csv` — SHA256 `500abe947b6a23a2d088459edbe0573aef0bb02c8e9f58cb23f3fc768b027ec3`
- `kaddress_m15_manifest_nope-gpt-small.json` — SHA256 `c54a79b0fd45af4b2800e7dce99aa76e580fe472bec5b08e55b00d1e308a3d74`
- `kaddress_m15_projectors_nope-gpt-small.npz` — SHA256 `c17e6dcb70aa86f8c1ea0af90de130d4f7e6314478a9588270e672bd0b8a2403`

Manifest highlights: `stimulus_count=6`, `summary_rows=14208`, `families=[A,B,C]`, `requested_device=cpu`, `cuda_available=false`, Python `3.11.11`, Torch `2.5.1`, model revision `320681e33a029517e27c68a0f9c2b07ea0004155`. Family A contributed five repeated-segment stimuli with 135–158 repetitions and 6–7 token slots; Family C contributed one natural-recurrence stimulus. Family B produced no valid stimuli under the tokenizer alignment plus ≥120-repetition constraints.

Gate and null summary:

- G1 architectural zero passed for all 560 layer-0 checked slots/heads; max layer-0 position fraction was `1.52e-6`, below the `1e-5` variance floor, and the deliberate perturbation check could fail.
- The manifest records `shuffle_null_ok=false`: slot-level shuffled-R² absolute quantiles were median `0.0359`, 90% `0.0967`, 95% `0.1222`, 99% `0.1902`. Aggregate Family A projector rows were much cleaner, with selected-layer shuffled R² around `-0.003` to `-0.006`.

Selected Family A slot-level depth means:

| layer | position fraction | ridge R² | PCA k90 | R² after PC projection |
|---:|---:|---:|---:|---:|
| 0 | 8.88e-7 | 0.000 | 1.00 | 0.000 |
| 1 | 0.00797 | 0.029 | 1.01 | 0.020 |
| 2 | 0.0113 | 0.152 | 1.14 | 0.138 |
| 6 | 0.0400 | 0.727 | 1.71 | 0.251 |
| 12 | 0.0481 | 0.951 | 1.10 | 0.055 |
| 18 | 0.0843 | 0.951 | 2.04 | 0.010 |
| 23 | 0.0892 | 0.979 | 2.20 | -0.023 |

### Analysis

The primary NoPE prediction is supported: layer-0 keys are effectively position-free, but repeated-token Family A keys become strongly position-decodable with depth. Mean ridge R² reaches `0.727` by layer 6 and ≈`0.95–0.98` from layer 12 onward, while the position fraction rises from `8.88e-7` at layer 0 to `0.089` at layer 23.

The position signal is low-dimensional in the slot-level analysis: Family A needs about 1–2 PCs to explain 90% of residual variation across most depths. Projecting out those PCs usually removes most position decodability at upper layers. In the aggregate Family A projector rows, layer-23 ridge R² drops from `0.9069` to `0.0049`; token-identity nearest-centroid accuracy on the reservoir sample is retained (`0.9844 → 0.9957`). This suggests the computed position component is removable for this NoPE run, at least under the implemented projector/sample diagnostic.

Caveats: Family B failed to instantiate valid aligned frame-token stimuli for this tokenizer/settings, and the slot-level shuffled-y null is not as clean as pre-registered even though aggregate nulls are near zero. Family C corroborates that recurrence keys contain strong position signal (`position_fraction=0.177`, `ridge_r2=0.848` on average), but it remains confounded and cannot overturn Family A.

### Conclusion / Next Step

This selected NoPE M1.5 run validates the depth-resolved key-level computed-position effect: NoPE layer 0 is an architectural zero, then `k_pre` keys acquire a strong, low-dimensional, mostly removable position component with depth. Next step is to run the same M1.5 implementation on GPT-2/Pythia/Qwen3 for the addendum's cross-model stamped-vs-computed decision tree, and to fix/improve Family B generation before treating B as a passed induction-control condition.

## 2026-07-20 — K-address-space M1 NoPE-GPT-Small full CUDA run prep

> **RETRACTED (F8).** This run's null is a corpus artifact — Track A contains zero valid
> address-purity trials. See "Known corpus defect F8" at the top of this notebook. The
> extraction/execution below is valid; the address-space *conclusion* is not.

### Question / Hypothesis

Does `andrewdalpino/NoPE-GPT-Small-Base`, a true NoPE model with no positional encoding path, show M1 address-purity heads when keys are measured directly from attention `qkv_proj` output?

### Experiment Design Summary

Full Track A extraction for NoPE-GPT-Small-Base: deterministic generator output, all 24 layers, all 16 heads, `k_pre` keys from the remote implementation's `SelfAttention.qkv_proj`, head-mean-centered cosine, and pairwise M1 AUC against same-type/different-referent and position-matched controls.

Preparation changes over the Qwen3 run:
- Added model tag `nope-gpt-small` for `andrewdalpino/NoPE-GPT-Small-Base`, loaded via `AutoModel` with `trust_remote_code=True` after inspecting the remote implementation.
- Verified from remote `model.py` that `NoPEGPT.forward()` starts from `token_embeddings(x)`, feeds decoder blocks directly, and `SelfAttention` uses `qkv_proj` keys without positional embeddings, RoPE, or ALiBi.
- Added `_capture_nope_k()` to hook each decoder block's `attention.qkv_proj` and store per-head keys.
- Added a runtime NoPE sanity check that aborts if any module name suggests a positional/RoPE/ALiBi path.

### Planned Procedure

Run on RunPod CUDA from the pre-run commit:

```bash
cd /workspace/dead-keys-census
git pull --ff-only
./scripts/runpod-persistent-cache-setup
. ~/.dead-keys-census-runpod-env
DEAD_KEYS_CUDA_VENV=/workspace/dead-keys-census-cache/venvs/cuda-system \
DEAD_KEYS_CUDA_SKIP_INSTALL=1 \
PYTHONPATH=experiments/dead-keys:experiments/k-address-space \
./scripts/cuda-run -m kaddress.scripts.address_purity \
  --model nope-gpt-small --device cuda --limit-docs 999 \
  --output-dir outputs/k_address_space_m1_nope_gpt_small_full_cuda_20260720
```

### Expected Signal / Interpretation Plan

A valid run should produce 384 per-head rows (24 layers × 16 heads × 1 key variant), a manifest, compact mention vectors, and NoPE sanity output. Address-head calls require AUC > 0.9 against both controls. This is the true no-positional-encoding endpoint for the positional-encoding gradient in the spec.

### Pre-run Provenance

- Spec: `experiments/k-address-space/spec.md`
- Code branch: `main`
- Pre-run commit: `2e72271` (`Prepare k-address M1 NoPE run`); successful rerun code commits `d407c5f` (`Fix NoPE remote model loading`), `c42adca` (`Load NoPE checkpoint weights explicitly`), and `0555c06` (`Fix NoPE manifest key variant label`)
- Planned output location: `outputs/k_address_space_m1_nope_gpt_small_full_cuda_20260720`
- Random seed: default script seed `0`
- Environment: planned RunPod CUDA via `scripts/cuda-run`; exact pod/GPU/torch/transformers versions to be recorded at run time from the manifest
- Model: `andrewdalpino/NoPE-GPT-Small-Base` pinned at Hugging Face revision `320681e33a029517e27c68a0f9c2b07ea0004155`
- Preparation checklist: `temp/repro-checklists/20260720-k-address-space-m1-nope-gpt-small.md`
- Local verification: `py_compile` passed for loader and address-purity script; remote `model.py` inspection found no positional embedding/RoPE/ALiBi path.

### Results

Run completed on RunPod L4 from commit `0555c06` after retry fixes for the remote-code import shims, explicit checkpoint loading into the wrapper's inner model, and the NoPE manifest key-variant label. Pod `t1cpksbru2b23g` used network volume `sndrrdckku` mounted at `/workspace`.

Command:

```bash
DEAD_KEYS_CUDA_VENV=/workspace/dead-keys-census-cache/venvs/cuda-system \
DEAD_KEYS_CUDA_SKIP_INSTALL=1 \
PYTHONPATH=experiments/dead-keys:experiments/k-address-space \
./scripts/cuda-run -m kaddress.scripts.address_purity \
  --model nope-gpt-small \
  --revision 320681e33a029517e27c68a0f9c2b07ea0004155 \
  --device cuda --limit-docs 999 \
  --output-dir outputs/k_address_space_m1_nope_gpt_small_full_cuda_20260720
```

Run window: `2026-07-20T16:14:28Z` to `2026-07-20T16:17:27Z`; `EXIT_CODE=0`.

Outputs were published as GitHub Release assets:

- Release: <https://github.com/vhallac/crockpot-experiments/releases/tag/output-k-address-space-m1-nope-gpt-small-full-cuda-20260720>
- Artifact: <https://github.com/vhallac/crockpot-experiments/releases/download/output-k-address-space-m1-nope-gpt-small-full-cuda-20260720/k_address_space_m1_nope_gpt_small_full_cuda_20260720.tar.gz> — 125,653,080 bytes; SHA256 `858af31ba7c707897a9bc40de38b619bb41e6626649a9803fbcc49565fa6f664`
- Checksums: <https://github.com/vhallac/crockpot-experiments/releases/download/output-k-address-space-m1-nope-gpt-small-full-cuda-20260720/SHA256SUMS-m1-nope-gpt-small-full-cuda-20260720.txt> — 846 bytes; SHA256 `702e6b34aa73f0549e1a8d45032a5f7b6924334a24b611eff22189b87661dbad`.

Internal output checksums:

- `kaddress_m1_nope-gpt-small.csv` — SHA256 `7e968a6dc931d39ceb36a1a03b1e768932139d49e51d935c7511262f4885573d`
- `kaddress_manifest_nope-gpt-small.json` — SHA256 `e3cf40b2c11bc542997b97ab8ff3737d35bdccf74b2750c37f89117026ba7d07`
- `kaddress_mentions_nope-gpt-small.npz` — SHA256 `36b5bbaa822a8c9ffb47a3939b35bcd111b2ce2352bc36dd3f4a5a03ce58412e`
- `run.log` — SHA256 `07338007c381f8c7c93069bbde41075400b6b8c1c15b412bf16ad7f48b43dfa7`

Manifest highlights: `doc_count=36`, `mention_token_rows=516096`, `max_doc_tokens=829`, `requested_device=cuda`, `cuda_available=true`, `cuda_device=NVIDIA L4`, `torch=2.8.0+cu128`, `torch_cuda=12.8`, Python `3.12.3`, HF revision `320681e33a029517e27c68a0f9c2b07ea0004155`.

NoPE sanity output:

```text
NoPE sanity: token_embeddings feed decoder body directly; SelfAttention.qkv_proj keys are used without positional embedding, RoPE, or ALiBi
```

**Address heads: 0/384.**

| Variant | Mean AUC vs same-type | Mean AUC vs pos-matched | Mean diff-surface AUC | Max same-type AUC | Max pos-matched AUC |
|---------|----------------------|------------------------|----------------------|-------------------|---------------------|
| `k_pre` | 0.5328               | 0.3549                 | 0.3439               | 0.6278            | 0.5691              |

Top same-type AUC head: layer 0 head 14, same-type AUC `0.6278`, position-matched AUC `0.5186`, diff-surface AUC `0.2629`.

### Analysis

NoPE-GPT-Small-Base shows no M1 address heads under the pre-registered threshold (AUC > 0.9 against both controls). Its best same-type AUC is 0.6278 and best position-matched AUC is 0.5691, far below threshold. Diff-surface same-referent purity is also weak (mean 0.3439, max 0.4956), so this NoPE endpoint does not reveal semantic address clustering in the implemented Track A/M1 slice.

Across the implemented quartet M1 Track A runs (GPT-2, Pythia-410m, Qwen3-0.6B, and NoPE-GPT-Small-Base), the pre-registered small-model sweep returns zero address heads by the strict threshold. **This was originally read as a possible small-scale/synthetic-corpus limit; that reading is withdrawn under F8** — the corpus has no valid trials, so the null is uninformative about scale or about the models, in the NoPE run and all three others.

### Conclusion / Next Step

This is a valid CUDA extraction of NoPE-GPT-Small-Base Track A / M1 in direct `k_pre` coordinates. It completes the planned quartet M1 sweep with no address heads. **Caveat (F8): the null reflects an invalid corpus, not the model or scale** — the address-space question is unadjudicated pending corpus v3, which is deferred behind M1.6.

## 2026-07-20 — K-address-space M1 Qwen3 full CUDA run prep

> **RETRACTED (F8).** This run's null is a corpus artifact — Track A contains zero valid
> address-purity trials. See "Known corpus defect F8" at the top of this notebook. The
> extraction/execution below is valid; the address-space *conclusion* is not.

### Question / Hypothesis

Does Qwen3-0.6B show M1 address-purity heads when keys are measured in both Qwen3 address coordinates (`k_pre`: post-`k_norm`, pre-RoPE) and cached coordinates (`k_post`: post-RoPE), grouped by the model's 8 KV heads under GQA?

### Experiment Design Summary

Full Track A extraction for Qwen3-0.6B: deterministic generator output, all 28 layers, all 8 KV heads, both `k_pre` and `k_post`, head-mean-centered cosine, and pairwise M1 AUC against same-type/different-referent and position-matched controls.

Preparation changes over the Pythia run:
- `_capture_qwen_k()`: hooks Qwen3 `k_proj` for pilot raw keys, hooks `k_norm` for `k_pre`, and extracts cached `k_post` from `past_key_values`.
- Qwen3 sanity gate confirms hook order `k_proj -> k_norm -> RoPE`, checks `num_attention_heads=16`, `num_key_value_heads=8`, Q-to-KV group size 2, reconstructs full-RoPE `k_post` from `k_pre`, and perturbs RoPE to prove the gate can fail.
- M1 rows are grouped by KV head (`head == kv_head`) rather than Q head.
- Runtime requirement updated to `transformers>=4.51.0` because the local Nix Transformers 4.46.2 package does not recognize `model_type=qwen3`.

### Planned Procedure

Run on RunPod CUDA from the pre-run commit after refreshing/installing the shared CUDA venv if needed:

```bash
cd /workspace/dead-keys-census
./scripts/runpod-persistent-cache-setup
. ~/.dead-keys-census-runpod-env
PYTHONPATH=experiments/dead-keys:experiments/k-address-space ./scripts/cuda-run   -m kaddress.scripts.address_purity   --model qwen3   --device cuda   --limit-docs 999   --sanity-gate-strict   --output-dir outputs/k_address_space_m1_qwen3_full_cuda_20260720
```

If reusing a known-current venv, `DEAD_KEYS_CUDA_SKIP_INSTALL=1` is acceptable only after verifying `python -c "import transformers; print(transformers.__version__)"` reports `>=4.51.0`.

### Expected Signal / Interpretation Plan

A valid run should produce 448 per-head rows (28 layers × 8 KV heads × 2 key variants), a manifest, compact mention vectors, and strict sanity-gate output. Address-head calls require AUC > 0.9 against both controls. Compare Qwen3 address-head count and pre/post AUC deltas to the prior GPT-2 and Pythia M1 runs; Qwen3's QK norm is expected to tighten within-referent key directions if the QK-norm prediction holds.

### Pre-run Provenance

- Spec: `experiments/k-address-space/spec.md`
- Code branch: `main`
- Pre-run commit: `f3bb63a` (`Prepare k-address M1 Qwen3 run`); successful rerun code commit `35c91c2` (`Add RunPod pod bring-up helper`)
- Planned output location: `outputs/k_address_space_m1_qwen3_full_cuda_20260720`
- Random seed: default script seed `0`
- Environment: planned RunPod CUDA via `scripts/cuda-run`; exact pod/GPU/torch/transformers versions to be recorded at run time from the manifest
- Model: `Qwen/Qwen3-0.6B` via Hugging Face default revision
- Preparation checklist: `temp/repro-checklists/20260720-k-address-space-m1-qwen3.md`
- Local verification: `py_compile` passed; GPT-2 1-doc/1-layer/1-head regression smoke passed; Pythia 1-doc/1-layer/1-head strict RoPE regression smoke passed. Local Qwen3 smoke is blocked by Nix Transformers 4.46.2 and must run in the updated CUDA venv.

### Results

Run completed on RunPod L4 from commit `35c91c2` after the retry helper created pod `cf160k2go7rknf` with network volume `sndrrdckku` mounted at `/workspace`.

Command:

```bash
DEAD_KEYS_CUDA_VENV=/workspace/dead-keys-census-cache/venvs/cuda-system \
DEAD_KEYS_CUDA_SKIP_INSTALL=1 \
PYTHONPATH=experiments/dead-keys:experiments/k-address-space \
./scripts/cuda-run -m kaddress.scripts.address_purity \
  --model qwen3 --device cuda --limit-docs 999 --sanity-gate-strict \
  --output-dir outputs/k_address_space_m1_qwen3_full_cuda_20260720
```

Run window: `2026-07-20T15:14:05Z` to `2026-07-20T15:19:18Z`; `EXIT_CODE=0`.

Outputs were published as GitHub Release assets:

- Release: <https://github.com/vhallac/crockpot-experiments/releases/tag/output-k-address-space-m1-qwen3-full-cuda-20260720>
- Artifact: <https://github.com/vhallac/crockpot-experiments/releases/download/output-k-address-space-m1-qwen3-full-cuda-20260720/k_address_space_m1_qwen3_full_cuda_20260720.tar.gz> — 292,446,104 bytes; SHA256 `6c51833cc4d4848843eebd4a2967ab71e2d5c2e71a907359789cfb3837cd7047`
- Checksums: <https://github.com/vhallac/crockpot-experiments/releases/download/output-k-address-space-m1-qwen3-full-cuda-20260720/SHA256SUMS-m1-qwen3-full-cuda-20260720.txt> — 765 bytes; SHA256 `e24bff523a1c450fb6ed7e9e0a3f6cb4dc9055deeff1736c1ae7e2fa47fe39e5`.

Internal output checksums:

- `kaddress_m1_qwen3.csv` — SHA256 `4732acee42cc31d7af502bf58dd9450f7faf4813029ad0d24acf194156c3be73`
- `kaddress_manifest_qwen3.json` — SHA256 `27c2a6cb9d446d4352513a74dbd3b81c142ba3a8c383dba54bbf38888a3436fd`
- `kaddress_mentions_qwen3.npz` — SHA256 `70181eda3dbde619d6a7f843764b5e717baab7898681c6fc90882fe1526e6e8f`
- `run.log` — SHA256 `27c3a46ef71e7dbea73e5a25133e0ba79bf969419ee4c83e332d5de58280ddab`

Manifest highlights: `doc_count=36`, `mention_token_rows=602112`, `max_doc_tokens=829`, `requested_device=cuda`, `cuda_available=true`, `cuda_device=NVIDIA L4`, `torch=2.8.0+cu128`, `torch_cuda=12.8`, Python `3.12.3`.

Strict Qwen3 sanity gate passed:

```text
Qwen3 sanity: q_heads=16 kv_heads=8 q_to_kv_group=2 hook_order=k_proj→k_norm→RoPE raw_norm_mean=41.4591 pre_norm_mean=50.8979
RoPE sanity gate: rotary_ndims=128/128  rel_l2_err=6.01e-09  max_abs_err=4.77e-06  max_elem_rel_err=1.85e-03  static_match=True  perturb_fails=True  → PASS
```

**Address heads: 0/448 (0/224 `k_pre`, 0/224 `k_post`).**

| Variant | Mean AUC vs same-type | Mean AUC vs pos-matched | Mean diff-surface AUC | Max same-type AUC | Max pos-matched AUC |
|---------|----------------------|------------------------|----------------------|-------------------|---------------------|
| `k_pre` | 0.5476               | 0.3386                 | 0.2424               | 0.6578            | 0.6220              |
| `k_post`| 0.5373               | 0.2403                 | 0.3006               | 0.6795            | 0.7629              |

AUC delta (`pre - post`) on same-type controls: mean `+0.0103`; `157/224` KV heads have `pre > post`.

### Analysis

Qwen3-0.6B shows no M1 address heads under the pre-registered threshold (AUC > 0.9 against both same-type/different-referent and position-matched controls). The best same-type AUC is 0.6795, still far below the address-head cutoff, and the mean diff-surface AUC remains low for both variants.

The namespace-direction signal matches the Pythia trend only weakly: `k_pre` has a small average same-type AUC advantage over `k_post` (+0.0103), and `k_post` reduces position-matched AUC on average. The magnitude is small and does not rescue address purity.

Across the implemented trio M1 Track A runs (GPT-2, Pythia-410m, Qwen3-0.6B), the pre-registered small-model sweep finds zero address heads by the strict threshold.

### Conclusion / Next Step

This is a valid CUDA extraction of Qwen3-0.6B Track A / M1 for both Qwen3 address coordinates (`k_pre`) and cached RoPE coordinates (`k_post`). It completes the implemented trio M1 sweep with no address heads. **Caveat (F8): the null reflects an invalid corpus, not the model or scale** — the address-space question is unadjudicated pending corpus v3, which is deferred behind M1.6.

## 2026-07-18 — K-address-space M1 Pythia full CUDA run

> **RETRACTED (F8).** This run's null is a corpus artifact — Track A contains zero valid
> address-purity trials. See "Known corpus defect F8" at the top of this notebook. The
> extraction/execution below is valid; the address-space *conclusion* is not. The "namespace
> direction" pre/post AUC deltas noted below are likewise not interpretable on this corpus.

### Question / Hypothesis

Does the M1 address-purity measurement (same-referent cosine AUC against controls) show address heads in Pythia-410m's `k_pre` (pre-RoPE) or `k_post` (post-RoPE) keys, and does partial RoPE (25% rotary dims) reduce address purity relative to pre-rotation keys?

### Experiment Design Summary

Full Track A extraction for Pythia-410m: 36 documents, all 24 layers, all 16 heads, both `k_pre` (hooked from `query_key_value` output) and `k_post` (extracted from `past_key_values` via `use_cache=True`).

Key implementation additions over the GPT-2 slice:
- `_capture_pythia_k()`: hook `query_key_value` for `k_pre`, extract `k_post` from cache
- `_rope_sanity_gate()`: reconstruct `k_post` from `k_pre` using from-scratch partial RoPE (config `rotary_pct=0.25`, `rotary_ndims=16/64`), verify against cached `k_post` (rel err ≤ 1e-3), confirm static dims match, perturbation check proves gate can fail
- `_extract_mentions()` dispatches by model tag, stores `key_variant` column
- `_summarize_auc()` groups by `(layer, head, key_variant)`

### Planned Procedure

```bash
PYTHONPATH=experiments/dead-keys:experiments/k-address-space DEAD_KEYS_CUDA_VENV=/workspace/dead-keys-census-cache/venvs/cuda-system DEAD_KEYS_CUDA_SKIP_INSTALL=1 ./scripts/cuda-run \
  -m kaddress.scripts.address_purity \
  --model pythia410 \
  --device cuda \
  --limit-docs 999 \
  --output-dir outputs/k_address_space_m1_pythia410_full_cuda_20260718
```

### Pre-run Provenance

- Spec: `experiments/k-address-space/spec.md`
- Code branch: `main`
- Pre-run commit: `64cb36f` (`Fix RoPE sanity gate: compute cos/sin from config frequencies`)
- Planned output location: `outputs/k_address_space_m1_pythia410_full_cuda_20260718`
- Random seed: default script seed `0`
- Environment: RunPod NVIDIA L4 (EU-RO-1), `deadd-keys-census-cuda` template (`1zpm2v05rn`), volume `dead-weight` (`sndrrdckku`), venv `/workspace/dead-keys-census-cache/venvs/cuda-system`
- Model: `EleutherAI/pythia-410m` via Hugging Face default revision
- Pod: `hlh4s9gxn0luyk`, $0.39/hr, L4, 6 vCPU, 62GB RAM
- Local smoke verification: 1/2 doc CPU smoke with sanity gate PASS before GPU run

### Results

Run completed on RunPod L4. GPU utilization verified at 97% during extraction.

Outputs under `outputs/k_address_space_m1_pythia410_full_cuda_20260718/` were republished as a `.tar.gz` GitHub Release artifact bundle:

- Release: <https://github.com/vhallac/crockpot-experiments/releases/tag/output-k-address-space-m1-pythia410-full-cuda-20260718>
- Artifact: <https://github.com/vhallac/crockpot-experiments/releases/download/output-k-address-space-m1-pythia410-full-cuda-20260718/k_address_space_m1_pythia410_full_cuda_20260718.tar.gz> — 264,976,544 bytes; SHA256 `a49da820a852b2b1e5185871a5377524f37627075080f6b91b20be664024fecc`
- Checksums: <https://github.com/vhallac/crockpot-experiments/releases/download/output-k-address-space-m1-pythia410-full-cuda-20260718/SHA256SUMS-m1-pythia410-full-cuda-20260718.txt> — 554 bytes; SHA256 `27293c4bbeb01488f9fa76622ce79251582504399788d3016a75aa2250d32e7f`.

Internal output checksums:

- `kaddress_m1_pythia410.csv` — SHA256 `3af266fe209bcf34948762f4688646eea311a24ace73c56e77f42ff07f35639c`
- `kaddress_manifest_pythia410.json` — SHA256 `1bcd5f300d0f9c57fd2671f0dca7e45237008d70987ffdadb29b7449ae061b3f`
- `kaddress_mentions_pythia410.npz` — SHA256 `a1c305714744b714304c8915f2dae9712d48d1f231261028d1e9426e9cb9bc81`

Manifest highlights: `doc_count=36`, `mention_token_rows=1087488`, `max_doc_tokens=841`, `requested_device=cuda`, `cuda_available=true`, `cuda_device=NVIDIA L4`, `torch=2.8.0+cu128`.

RoPE sanity gate: `max_rel_err=0.00e+00`, `static_match=True`, `perturb_fails=True` → **PASS**.

**Address heads: 0/768 (0/384 per variant).**

| Variant | Mean AUC vs same-type | Mean AUC vs pos-matched | Mean diff-surface AUC | Max same-type AUC |
|---------|----------------------|------------------------|----------------------|-------------------|
| k_pre   | 0.5404               | 0.3708                 | 0.2833               | 0.6526 (L9 H9)    |
| k_post  | 0.5294               | 0.1742                 | 0.3951               | 0.6558 (L9 H9)    |

AUC delta (pre − post): mean = +0.0111, 272/384 heads (70.8%) have pre > post. RoPE rotation slightly reduces same-referent purity on average, consistent with the namespace hypothesis direction, but the effect is tiny.

Position-matched AUC collapses for k_post (0.3708 → 0.1742), indicating that RoPE does scatter keys by position — but this affects controls as much as same-referent pairs.

Diff-surface AUC is very low for k_pre (0.283) — the discriminating M2 test shows minimal semantic addressing. For k_post, diff-surface AUC is actually higher (0.395) but still far below address-head threshold.

### Analysis

No address heads found by the pre-registered M1 threshold (AUC > 0.9 against both controls) in Pythia-410m for either k_pre or k_post. The best observed same-type AUC is 0.656, substantially below the 0.9 address-head cutoff.

This result aligns with the GPT-2 M1 run (0/144 address heads) and does not contradict the spec's scale caveat: semantic addressing may emerge at larger scales (> 7B) and absence at 410M–0.6B is not absence at scale.

Modest support for the namespace hypothesis direction: k_pre purity > k_post purity in 70.8% of heads, and position-matched AUC drops sharply after RoPE (0.371 → 0.174). However, the mean delta (+0.011) is negligible in magnitude.

### Conclusion / Next Step

This run is a valid CUDA extraction of Pythia-410m Track A / M1 for both k_pre and k_post. It does not show address heads by the pre-registered threshold. **Caveat (F8): the null reflects an invalid corpus, not the model** — the address-space question is unadjudicated pending corpus v3.

The trio census now has two of three models (GPT-2, Pythia-410m); Qwen3-0.6B remains for the full spec sweep.

## 2026-07-17 — K-address-space M1 GPT-2 smoke

### Question / Hypothesis

Does the first `spec.md` write/address measurement (M1) show any same-referent key-neighborhood signal on a small deterministic Track A slice for GPT-2?

### Experiment Design Summary

This is a first runnable smoke slice, not the full pre-registered weekend run. It generates deterministic synthetic Track A state-update documents with gold mention spans, extracts GPT-2 pre-rotation/pre-cache keys (`k_pre`; GPT-2 has no RoPE), head-mean-centers vectors per layer/head, and computes pairwise cosine AUC for same-referent mention-token pairs against same-type/different-referent and near-position different-referent controls.

### Planned Procedure

```bash
PYTHONPATH=experiments/dead-keys:experiments/k-address-space ./scripts/nix-cpu-run -m kaddress.scripts.address_purity \
  --model gpt2 \
  --limit-docs 2 \
  --limit-layers 1 \
  --limit-heads 1 \
  --output-dir outputs/k_address_space_m1_gpt2_smoke
```

### Expected Signal / Interpretation Plan

A successful smoke run should produce a manifest, a per-head M1 CSV, and compact mention vectors. AUC is not expected to be meaningful at `--limit-docs 2 --limit-layers 1 --limit-heads 1`; this run validates the extraction/analysis path before a broader local or GPU run.

### Pre-run Provenance

- Spec: `experiments/k-address-space/spec.md`
- Code branch: `main`
- Pre-run commit: `3bc1714` (`Add k-address-space M1 smoke experiment`)
- Planned output location: `outputs/k_address_space_m1_gpt2_smoke`
- Random seed: default script seed `0`
- Environment: local CPU via `scripts/nix-cpu-run`
- Model: `gpt2` via Hugging Face default revision

### Results

Run completed locally on CPU.

Outputs under `outputs/k_address_space_m1_gpt2_smoke/`:

- `kaddress_m1_gpt2.csv` — 290 bytes; SHA256 `fbc48a8e570b6dc536bdc912d3860802653a646dc01a982852ce2fd4ab3b16bb`
- `kaddress_manifest_gpt2.json` — 564 bytes; SHA256 `3e5f2052e7ca75eefbf080bc3ce953d0e0d68cc9eaae0aa69ac717ea0a2acf79`
- `kaddress_mentions_gpt2.npz` — 5784 bytes; SHA256 `642fcd300a574d219d94817b18e4bc8579a23019dc9e907850936ad82c312486`

Observed smoke row:

| layer | head | same-ref pairs | same-type diff-ref pairs | position-matched diff-ref pairs | AUC vs same-type | AUC vs position-matched | diff-surface AUC | address head |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 0 | 16 | 78 | 22 | 0.53125 | 0.34659 | 0.21154 | false |

### Analysis

The run validates that the first M1 extraction/analysis path executes end-to-end and produces compact outputs. With only two documents, one layer, and one head, the AUC values are smoke-test diagnostics only and should not be interpreted as evidence for or against the address-space hypothesis.

### Conclusion / Next Step

The first runnable experiment slice is in place and smoke-tested. Next step is a broader GPT-2 M1 run over all heads/layers and more Track A documents, followed by RoPE sanity-gate implementation before extending to Pythia/Qwen3.

## 2026-07-17 — K-address-space M1 GPT-2 full implemented Track A run

### Question / Hypothesis

Does the implemented Track A / M1 address-purity slice show same-referent key-neighborhood signal across all GPT-2 layers and heads on the full deterministic Track A generator output currently implemented in this repository?

### Experiment Design Summary

This run scales the already smoke-tested first slice from 2 documents / 1 layer / 1 head to the full implemented GPT-2 slice: deterministic Track A synthetic documents generated by `generate_track_a`, GPT-2 pre-cache keys (`k_pre`; no RoPE), all 12 layers and 12 heads, head-mean-centered cosine, and pairwise AUC of same-referent mention-token pairs against same-type/different-referent and near-position different-referent controls.

The implementation is still narrower than the full pre-registration: it covers GPT-2 and M1 on Track A only. Pythia/Qwen3 and RoPE sanity gates remain future work before those models should be claimed as run.

### Planned Procedure

Run on RunPod L4 from a committed repository state, using only non-interactive SSH command arguments (no piped `ssh -tt`):

```bash
PYTHONPATH=experiments/dead-keys:experiments/k-address-space DEAD_KEYS_CUDA_VENV=/venv-deadkeys DEAD_KEYS_CUDA_SKIP_INSTALL=1 ./scripts/cuda-run \
  -m kaddress.scripts.address_purity \
  --model gpt2 \
  --device cuda \
  --limit-docs 999 \
  --output-dir outputs/k_address_space_m1_gpt2_full_20260717
```

`--limit-docs 999` is used because the CLI default is a 2-document smoke limit; the current generator exhausts at 36 documents.

### Expected Signal / Interpretation Plan

A full implemented run should produce 144 per-head rows (12 layers × 12 heads), a manifest, and compact mention vectors. Address-head calls require AUC > 0.9 against both control classes. Diff-surface AUC is the discriminating lexical-vs-semantic diagnostic, but results are still limited to the synthetic Track A generator and GPT-2 only.

### Pre-run Provenance

- Spec: `experiments/k-address-space/spec.md`
- Code branch: `main`
- Pre-run commit: `1dbe4e6` for the first attempt; redo fix commit pending after failed attempt.
- Planned output location: `outputs/k_address_space_m1_gpt2_full_20260717`
- Random seed: default script seed `0`
- Environment: RunPod NVIDIA L4 via `scripts/cuda-run`
- Model: `gpt2` via Hugging Face default revision

### Results

First RunPod attempt failed before producing usable outputs. The process was stopped after it stayed CPU-bound and logged a GPT-2 context warning:

- Active process: `python3 -m kaddress.scripts.address_purity ...`, PID 875, about 102% CPU.
- GPU: 0% utilization during inspection.
- Warning: `Token indices sequence length is longer than the specified maximum sequence length for this model (1507 > 1024)`.
- Output directory: no result files produced.

Redo fix: Track A filler is now inserted once per update round instead of after every referent sentence, long filler was shortened, tokenization uses explicit `truncation=True, max_length=...`, an over-budget generator guard fails fast, and M1 pair collection is vectorized per document.

Verification before relaunch:

```text
36 Track A docs; max GPT-2 tokens = 829; docs over 950 = 0
outputs/k_address_space_m1_gpt2_fix_smoke/ written by 2-doc smoke
outputs/k_address_space_m1_gpt2_fix_full_localcheck/ written by 36-doc 1-layer/1-head check
```

Full rerun did not reach successful output analysis. It was user-interrupted after external observation showed the rerun was still not making meaningful use of the GPU.

Second failure record:

- Failure mode: user interrupt due to lack of sustained GPU utilization.
- Root cause carried forward from the interrupted run: the forward pass used CUDA, but `_capture_gpt2_k()` copied every layer's keys to CPU immediately with `k.cpu()`, and the downstream cosine/AUC work used pandas/NumPy on CPU.
- Protocol outcome: this experiment attempt is concluded as failed; the next attempt must keep captured keys and pairwise analysis on CUDA and verify GPU utilization before treating the run as valid.

### Analysis

No scientific result is available from the interrupted full run. The failure is an implementation/execution failure, not evidence for or against K-space address purity.

### Conclusion / Next Step

Conclude this attempt as failed. Patch the M1 implementation so captured key tensors remain on the requested device and the cosine/AUC summarization can run on CUDA, then create a new pre-run entry and rerun only after GPU-use verification.

## 2026-07-17 — K-address-space M1 GPT-2 full CUDA rerun after GPU patch

> **RETRACTED (F8).** This run's null is a corpus artifact — Track A contains zero valid
> address-purity trials. See "Known corpus defect F8" at the top of this notebook. The
> extraction/execution below (GPU-verified) is valid; the address-space *conclusion* is not.

### Question / Hypothesis

Does the implemented Track A / M1 address-purity slice show same-referent key-neighborhood signal across all GPT-2 layers and heads on the full deterministic Track A generator output, when the extraction and analysis path keeps key vectors and pairwise cosine/AUC computation on CUDA?

### Experiment Design Summary

This is a redo of the interrupted GPT-2 M1 full implemented Track A run. The scientific design is unchanged: deterministic Track A documents, GPT-2 `k_pre`, all 12 layers and 12 heads, head-mean-centered cosine, and pairwise AUC against same-type/different-referent and position-matched controls.

The implementation change is execution-only: `_capture_gpt2_k()` no longer copies keys to CPU, mention vectors are stored as torch tensors on the requested device, cosine matrices are computed with torch, and AUC scoring uses torch tensors before final CSV/NPZ serialization.

### Planned Procedure

Run on RunPod L4 from a committed repository state, using only non-interactive external-IP SSH command arguments:

```bash
PYTHONPATH=experiments/dead-keys:experiments/k-address-space DEAD_KEYS_CUDA_VENV=/workspace/dead-keys-census-cache/venvs/cuda-system DEAD_KEYS_CUDA_SKIP_INSTALL=1 ./scripts/cuda-run \
  -m kaddress.scripts.address_purity \
  --model gpt2 \
  --device cuda \
  --limit-docs 999 \
  --output-dir outputs/k_address_space_m1_gpt2_full_cuda_20260717
```

Before accepting the run as valid, verify from outside the process that the manifest reports `requested_device: cuda`, `cuda_available: true`, and an NVIDIA L4 device, and sample `nvidia-smi` during execution for nonzero GPU memory/utilization.

### Expected Signal / Interpretation Plan

A valid full run should produce 144 per-head rows (12 layers × 12 heads), a manifest, and compact mention vectors. Address-head calls require AUC > 0.9 against both control classes. Diff-surface AUC remains the lexical-vs-semantic diagnostic. If GPU utilization remains absent after this patch, treat the attempt as another setup/implementation failure rather than a scientific result.

### Pre-run Provenance

- Spec: `experiments/k-address-space/spec.md`
- Code branch: `main`
- Pre-run commit: `6797a78` (`Keep k-address M1 analysis on torch device`)
- Planned output location: `outputs/k_address_space_m1_gpt2_full_cuda_20260717`
- Random seed: default script seed `0`
- Environment: RunPod NVIDIA L4 via `scripts/cuda-run`
- Model: `gpt2` via Hugging Face default revision
- Local verification before commit: 2-doc smoke and 36-doc/1-layer local CPU checks passed with the torch execution path.

### Results

Run completed on RunPod `dead-weight-m1-replacement-20260717190712` using the existing `/workspace/dead-keys-census-cache/venvs/cuda-system` CUDA venv with `DEAD_KEYS_CUDA_SKIP_INSTALL=1`. The erroneous `/workspace/dead-keys-census-cache/venvs/cuda-lite` environment was removed before rerun.

Outputs under `outputs/k_address_space_m1_gpt2_full_cuda_20260717/`:

- `kaddress_m1_gpt2.csv` — 12,438 bytes; SHA256 `e3f976f94f7e3b38be37a39477e4ecd6af947c6545324a732cb97ff9d7f3e07d`
- `kaddress_manifest_gpt2.json` — 647 bytes; SHA256 `fc9cb196224c3d64e3bbb1bd4d8fe107384c122e13c8bd050eefe8f0fcbd4588`
- `kaddress_mentions_gpt2.npz` — 48,325,922 bytes; SHA256 `7154ddd904a5878d28895959ee61298e6e1437f69e39efc7882d347f9cb76c7b`

Manifest highlights: `limit_docs=999`, `doc_count=36`, `mention_token_rows=193536`, `max_doc_tokens=829`, `requested_device=cuda`, `cuda_available=true`, `cuda_device=NVIDIA L4`, `torch=2.8.0+cu128`.

During the full rerun, external `nvidia-smi` sampling observed the process on GPU with 91% GPU utilization, 4,170 MiB GPU memory used, and PID 1542 using 4,164 MiB.

### Analysis

The implemented Track A / M1 GPT-2 full CUDA run produced 144 per-head rows and found `address_heads_m1=0/144` under the current threshold requiring AUC > 0.9 against both same-type/different-referent and position-matched controls.

Aggregate AUCs across heads were modest: mean same-referent-vs-same-type-different-referent AUC `0.502206`; mean same-referent-vs-position-matched-different-referent AUC `0.128059`.

Top same-type AUC rows remained far below the address-head threshold. Best observed row was layer 3 head 4 with same-type AUC `0.614722` and position-matched AUC `0.386818`.

### Published Measurements

Published as a GitHub release artifact bundle:

- Release: <https://github.com/vhallac/crockpot-experiments/releases/tag/output-k-address-space-m1-gpt2-full-cuda-20260717>
- Artifact: <https://github.com/vhallac/crockpot-experiments/releases/download/output-k-address-space-m1-gpt2-full-cuda-20260717/k_address_space_m1_gpt2_full_cuda_20260717.tar.gz> — 47,079,221 bytes; SHA256 `b60dbd9e0693002cc6fe76baff497f2d6260d2606622f7149670bd218028bce4`
- Checksums: <https://github.com/vhallac/crockpot-experiments/releases/download/output-k-address-space-m1-gpt2-full-cuda-20260717/SHA256SUMS-m1-gpt2-full-cuda-20260717.txt> — 519 bytes; SHA256 `8a3282335afa0dec19987410eacfcee71fcd04438c6ebcb260d5278e20bb04b6`.

### Conclusion / Next Step

This run is a valid CUDA execution of the current implemented GPT-2 Track A / M1 slice, not an environmental failure. It does not show address heads by the pre-registered M1 threshold in GPT-2 for this implemented synthetic slice. RoPE models are not expected to rescue this specific observation; if anything, RoPE makes a clean fixed K-address-space interpretation harder. Next step is to decide whether to extend the implementation toward the remaining spec items or revise the synthetic Track A slice before moving to RoPE models.
