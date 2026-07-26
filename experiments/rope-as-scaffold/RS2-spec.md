# RS2 — Subspace equivalence (emergent-vs-RoPE positional subspace overlap)

**Dated:** 2026-07-26
**Status:** pre-registered, not yet run
**Program:** [`rope-as-scaffold`](README.md) — tests claim **C2** (secondary).
**Depends on:** RS1 (RS1a + RS1b) completed — consumes their M1.5 projector outputs.

---

## 0. Question

RS1 showed that a DroPE'd Qwen3-0.6B's emergent key-position **fills in** after RoPE removal
(P.RS1.b holds), and that the addressing profile is statistically indistinguishable from the
RoPE baseline (P.RS1.c holds). RS1 established *that* emergent position is present and
non-addressable after the DroPE transition.

RS2 asks the next question: **is the emergent position the *same* positional code RoPE
supplied, or a different one?** I.e., does the DroPE'd model's emergent key-position subspace
**reconstruct** the RoPE `k_post` positional subspace, or has the model developed an
independent positional code?

**Primary (C2):** the DroPE'd model's emergent positional subspace substantially overlaps
the RoPE `k_post` positional subspace, above a random-rotation baseline — measured via
principal angles between the M1.5-produced PCA bases of each.

**Falsifier:** disjoint subspaces (alignment at or below random baseline) → emergent position
is a *different* code, not a reconstruction of what RoPE supplied. This does not invalidate C1
(position is still present and non-addressable) but qualifies the "reconstruction" framing.

---

## 1. Why this instrument

- **No new data collection.** RS2 consumes only the M1.5 projector NPZ files already produced
  by RS1a (RoPE state `k_post` and `k_pre`) and RS1b (DroPE'd state emergent `k_pre`). Both
  runs are complete, their outputs are published as GitHub Releases, and all artifacts are
  committed to the notebook.
- **Analysis-only, near-free.** The projector files are small (59–203 MB each). The
  computation is O(heads × d_head³) per head, completing in seconds on a single CPU core.
  No GPU, no pod, no training.
- **Single instrument, three comparisons.** The same principal-angle machinery answers all
  the interesting subspace questions from the same two projector files:
  1. **Primary:** RoPE `k_post` vs DroPE'd emergent `k_pre` → does emergent position
     reconstruct RoPE's code? (P.RS1.d)
  2. **Reference 1:** RoPE `k_pre` vs DroPE'd emergent `k_pre` → emergent-to-emergent
     comparison (how much does the emergent code drift across the transition?).
  3. **Reference 2:** RoPE `k_pre` vs RoPE `k_post` → the internal rotation shift (how much
     does rotation *change* the positional subspace, without any DroPE transition? — a
     ceiling on expected alignment).

---

## 2. Design

### 2.1 Inputs

Two M1.5 projector NPZ files, frozen and published:

| tag | NPZ | source run |
|---|---|---|
| RoPE `k_post` + `k_pre` | `outputs/rope_as_scaffold_rs1a_20260724T0559Z/m15_qwen3/kaddress_m15_projectors_qwen3.npz` | RS1a (states 1–2 run), published at [run/rope-as-scaffold-rs1a/20260724](https://github.com/vhallac/crockpot-experiments/releases/tag/run/rope-as-scaffold-rs1a/20260724) |
| DroPE'd emergent `k_pre` | `outputs/rs1b_probes_lr1e3_qwen3_droped_20260726/outputs/rs1b_probes_lr1e3_20260726T042955Z_m15_qwen3_droped/kaddress_m15_projectors_qwen3-droped.npz` | RS1b LR-corrected rerun, published at [run/rope-as-scaffold-rs1b-lr1e3/20260726](https://github.com/vhallac/crockpot-experiments/releases/tag/run/rope-as-scaffold-rs1b-lr1e3/20260726) |

Both were produced with `--families A --max-length 1024`, so the probe stimulus length is matched (§10.E of RS1-spec).

### 2.2 Method

Per (layer, head) present in both NPZ files:

1. Load the Family A aggregate PCA basis (rows = orthonormal basis vectors, dimension d_head = 128 for Qwen3-0.6B).
2. Compute **principal angles** between the RoPE `k_post` basis A and the DroPE'd emergent basis B via SVD of A^T B → arccos of singular values.
3. Compute the **scalar alignment** = mean cos(principal angles). 1.0 = identical subspaces, 0.0 = fully orthogonal.
4. Compute a **random-rotation baseline:** rotate B by 100 random orthonormal matrices and re-measure alignment; report mean ± std.
5. Primary metric: **`alignment_excess`** = observed alignment − random baseline mean. Excess > 0 means the observed overlap is above chance.

The implementation is `experiments/rope-as-scaffold/scripts/c2_subspace_overlap.py` (already committed).

### 2.3 Outputs

- `rs2_subspace_overlap.csv` — one row per (layer, head) with all alignment metrics.
- `rs2_subspace_summary.json` — global and per-layer summary statistics.

---

## 3. Gates (each must be able to fail)

- **G-RS2.1 — input integrity.** Both NPZ files exist, contain `post/` and `pre/` variant keys
  respectively, and share at least one common (layer, head). If the files are missing or
  incompatible → gate fails; stop and locate the artifacts.
- **G-RS2.2 — baseline is non-trivial.** The random-rotation baseline mean alignment is > 0
  and < 1.0 (i.e. not degenerate — confirms the subspace dimensionality is such that random
  overlap is measurable but not ceiling). If baseline ≈ 1.0, the subspaces saturate the
  ambient dimension (d_head too small for the number of PCA components) → report as a
  limitation; alignment_excess is then uninterpretable.
- **G-RS2.3 — internal rotation alignment is a sensible ceiling.** RoPE `k_pre` vs RoPE
  `k_post` alignment should be < 1.0 (rotation *does* change the positional subspace) but
  > random baseline (it's the *same* model's keys, just rotated). If it's at baseline →
  the instrument can't see the difference RoPE makes even internally → void.

---

## 4. Pre-registered predictions

These are P.RS1.d from RS1-spec §5, elaborated for RS2 with the three comparison arms:

- **(P.RS2.a — primary, = P.RS1.d) Subspace reconstruction.** The RoPE `k_post` vs DroPE'd
  emergent `k_pre` alignment excess is **substantially > 0** across most heads. *Falsifier:*
  alignment excess ≈ 0 globally (at or below random baseline) → emergent position is a
  *different* code, not a reconstruction.
- **(P.RS2.b — emergent drift reference).** RoPE `k_pre` vs DroPE'd emergent `k_pre`
  (emergent-to-emergent) alignment is **≥** the RoPE `k_post` vs DroPE'd emergent alignment
  (primary). Rationale: removing the rotation step that separates `k_pre` from `k_post` should
  make the before/after comparison *easier*, not harder. *Falsifier:* the primary alignment
  exceeds the emergent-to-emergent reference → rotation *helps* alignment, an unexpected
  geometry result.
- **(P.RS2.c — depth profile).** Alignment excess has a **depth-structured profile**, not
  uniform: early layers (where RS1a found position is rotation-propagated and ceiling-level
  decodable) show **high overlap** with RoPE's code; late layers (where position is emergent
  and rotation-independent) may diverge. *Falsifier:* uniform profile across depth (either
  uniformly high or uniformly low) suggests the depth structure RS1a reported is not reflected
  in the subspace geometry — either the instrument is insensitive to real differences or the
  depth-structured finding was artifactual.

---

## 5. Decision tree

- **P.RS2.a holds + P.RS2.c holds** → strong C2: emergent position genuinely reconstructs
  RoPE's code, and the depth profile mirrors the known rotation-dependence gradient. Upgrades
  the mechanistic account from "position fills in" to "position reconstructs the *same*
  positional information."
- **P.RS2.a holds, P.RS2.c fails** → C2 holds but without the depth structure RS1a led us to
  expect. Alignment is uniformly high → the instrument is coarse and everything looks similar;
  this downgrades the fine-grained interpretation but C2 still stands.
- **P.RS2.a fails** → C2 falsified: emergent position is a different code. C1 still stands
  (position is present and non-addressable). The narrative changes from "RoPE is a scaffold
  whose code is internalized" to "RoPE is a scaffold that enables the model to develop
  *any* positional code, which then takes over."
- **G-RS2.3 fails** (internal alignment at baseline) → the instrument is broken. Do not
  interpret C2; debug the projector loading / basis extraction.

---

## 6. Threats to validity

- **Subspace rank mismatch.** The RoPE `k_post` and DroPE'd emergent bases may have different
  numbers of PCA components (different PCA-to-90% rank). The principal-angle method handles
  this naturally (min(k1, k2) angles), but if one basis has very few components while the
  other is high-rank, alignment may be artificially high (the low-rank subspace fits inside
  the high-rank one). Report component counts per head.
- **Aggregate basis conflates position sources.** The Family A basis pools across multiple
  stimulus positions; it captures the *pooled* positional subspace, not per-position subspaces.
  If RoPE and emergent encode position in fundamentally different *bases* that both project
  well onto the same pooled subspace, alignment could be high even with different per-position
  codes. This is inherent to the M1.5 instrument; acknowledge it.
- **Single model, single DroPE recipe.** As with RS1, this is a 0.6B / Qwen3-arch /
  single-seed result. Generalization is RS4's job.
- **No new data — trust chain on prior artifacts.** RS2's validity depends on RS1a and RS1b
  having produced valid projector files. If either run had an undetected error, RS2 inherits
  it. Mitigated by the fact that both runs passed their own gates and the RS1b run passed
  P.RS1.a/b/c.

---

## 7. Schedule & budget

- **Compute:** CPU-only, seconds. The NPZ files are ~59–203 MB each; principal-angle
  computation is ~0.01s per head × 448 heads ≈ 4–5 seconds.
- **Cost:** effectively zero. Runs locally or on any CPU pod.
- **Engineering:** script is written and committed (`c2_subspace_overlap.py`). No new code.
- **Analysis:** ~15–30 minutes to interpret the per-layer CSV, produce depth-profile plots,
  and write the notebook entry.

---

## 8. Deliverable

A `rs2_subspace_summary.json` with global and per-layer alignment statistics, adjudicating
P.RS2.a–c — i.e. whether the DroPE'd model's emergent positional subspace reconstructs the
same code RoPE supplied, and at what depth. Plus a completed notebook entry with interpretation.

---

## 9. Implementation notes

RS2 is analysis-only — no new model runs, no training, no GPU. The implementation is already
committed (`c2_subspace_overlap.py`). The run command from the notebook pre-run entry is the
canonical invocation:

```bash
cd /home/vedat/work/personal/crockpot-experiments
./scripts/nix-cpu-run experiments/rope-as-scaffold/scripts/c2_subspace_overlap.py \
  --rope-projectors outputs/rope_as_scaffold_rs1a_20260724T0559Z/m15_qwen3/kaddress_m15_projectors_qwen3.npz \
  --droped-projectors outputs/rs1b_probes_lr1e3_qwen3_droped_20260726/outputs/rs1b_probes_lr1e3_20260726T042955Z_m15_qwen3_droped/kaddress_m15_projectors_qwen3-droped.npz \
  --output-dir outputs/rs2_subspace_$(date -u +%Y%m%dT%H%M%SZ) \
  --baseline-trials 100 \
  --families A \
  --seed 0
```

All parameters are pinned. The output directory timestamp is the only free variable.
