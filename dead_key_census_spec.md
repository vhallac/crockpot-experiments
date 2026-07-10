# Dead-Key Census & Reachability Sieve — Implementation Specification

> **Purpose of this document:** Complete, self-contained specification for writing the
> analysis scripts. Written for an implementer (human or LLM) with no access to the
> originating discussion. Everything needed is defined here. Follow shapes and
> conventions exactly — the most likely failure mode is a silent transpose or
> head-slicing error that produces confident nonsense.

---

## 0. One-paragraph background

In transformer attention, a cached key `k` can only influence generation if some future
query `q` produces a large dot product `q·k`. Queries are always manufactured as
`q = W_Q x` (x = residual stream vector), so the largest logit any bounded input can
produce against `k` is governed by `‖W_Q^T k‖` — the **pullback score**. Keys whose
energy lies in the weak directions of `W_Q` (the "soft kernel") are nearly unreachable
by ANY query, present or future. This is a **write-time, query-agnostic** eviction
signal. The project has two phases:

- **Phase 1 (weights-only census):** measure the soft-kernel structure per head from
  checkpoint weights alone. No forward passes.
- **Phase 2 (runtime wedge validation):** verify on real text that low-pullback keys
  receive near-zero attention during decode.

**Pre-registered predictions** (do not tune toward these; just measure):

1. **Wedge:** In a scatter of (write-time pullback score) vs (attention mass actually
   received), the low-score/high-mass quadrant is empty. High-score keys may receive
   any mass (reachability is necessary, not sufficient).
2. **Head bimodality:** dead-key fraction is bimodal across heads — high for diffuse
   heads, near-zero for sharp retrieval heads.
3. **Outlier parking:** the massive outlier directions that dominate raw key norms lie
   preferentially INSIDE the soft kernel (explains the published finding that low-norm
   keys receive high attention).
4. **Band split (RoPE models):** keys can be short-range-alive but long-range-dead;
   long-range reachability lives in the low-frequency rotary channels.

---

## 1. Conventions (read carefully)

- All math uses the convention: `W_Q`, `W_K` map **model space → head space**.
  Shape: `[d_head, d_model]`. A query is `q = W_Q @ x`, a key is `k = W_K @ y`,
  with `x, y ∈ R^{d_model}`, `q, k ∈ R^{d_head}`.
- The attention logit (ignoring scaling and RoPE) is `q·k = x^T (W_Q^T W_K) y`.
- **Pullback score of a key** `k ∈ R^{d_head}`:
  `s(k) = ‖W_Q^T k‖₂` where `W_Q^T k ∈ R^{d_model}`.
- **Interaction matrix per head:** `M = W_Q^T W_K`, shape `[d_model, d_model]`,
  rank ≤ d_head. NEVER materialize M directly (4096×4096 per head is wasteful and
  its spectrum can be computed in head space — see §3.2).
- All spectra, scores, and fractions are computed **per layer, per head**. Never pool
  across heads (different heads have incommensurable geometries).
- Float32 everywhere for the linear algebra (load fp16/bf16 weights, cast up).

---

## 2. Models

| Tag | HF id | Attention | Positional | Notes |
|---|---|---|---|---|
| gpt2 | `gpt2` | MHA, 12 heads | learned absolute | Control: NO RoPE. Skip band analysis. **Conv1D pitfall — see §6.1** |
| pythia410 | `EleutherAI/pythia-410m` | MHA | partial RoPE (25%) | Primary theory organism. Fused QKV — see §6.2. Rotary applies to first 25% of head dims only → clean literal band boundary |
| pythia1.4 | `EleutherAI/pythia-1.4b` | MHA | partial RoPE (25%) | Scale check of pythia410 |
| openllama7 | `openlm-research/open_llama_7b` | MHA | full RoPE | Ungated MHA at 7B. (Alternative: `meta-llama/Llama-2-7b-hf`, gated) |
| qwen25 | `Qwen/Qwen2.5-0.5B` | **GQA** | full RoPE | No QK-norm. GQA handling — see §3.5 |
| qwen3 | `Qwen/Qwen3-0.6B` | **GQA** | full RoPE | **Has QK-norm** → angular metric variant, see §3.6 |

Phase 1 runs on ALL models (CPU, minutes each).
Phase 2 runs on: gpt2, pythia410, qwen25, qwen3 (CPU/small GPU OK). openllama7 optional.

---

## 3. Phase 1 — Weights-only census

### 3.1 Extract per-head projection matrices

For each layer `l` and head `h`, produce:
- `A = W_Q[l,h]`, shape `[d_head, d_model]`
- `B = W_K[l,h]`, shape `[d_head, d_model]`

Extraction is architecture-specific — see §6 pitfalls before writing this code.
Verify with the sanity check in §6.4 before proceeding.

### 3.2 Spectra and alignment (per head)

Compute (economy SVDs; `A` is `[d_head, d_model]` so U is `[d_head, d_head]`):

```
U_A, S_A, _ = svd(A)          # S_A: [d_head] singular values of W_Q
U_B, S_B, _ = svd(B)          # S_B: [d_head] singular values of W_K
C = diag(S_A) @ (U_A.T @ U_B) @ diag(S_B)     # [d_head, d_head]
S_M = svdvals(C)              # = the nonzero singular values of M = A^T B
```

`U_A.T @ U_B` is the **alignment rotation** between query and key output bases.

Record per head:
- `S_A`, `S_B`, `S_M` (full vectors)
- Effective rank of each: `erank(S) = exp(entropy(S² / sum(S²)))`
- **Misalignment index:** `sum(S_M) / sum(S_A * S_B)` — equals 1 iff bases perfectly
  aligned; smaller = key-generating capacity aimed at query-blind directions.
- **Misalignment z-score (REQUIRED — the raw index is misleading):** for near-flat
  spectra the raw index has a floor just below 1 even under a completely random
  rotation (empirically ~0.976 at erank≈60/64), so raw values like 0.98 do NOT mean
  "98% aligned". Compute a per-head baseline: for 200 random orthogonal `R`
  (`[d_head, d_head]`, via QR of Gaussian), `m_rand = sum(svdvals(diag(S_A) @ R @
  diag(S_B))) / sum(S_A*S_B)`. Report `misalign_z = (raw - mean(m_rand)) /
  std(m_rand)`. All cross-head and cross-layer comparisons use the z-score; the raw
  index is recorded but never interpreted.

### 3.3 Dead-fraction via random-direction baseline

Definition of "soft kernel" is relative to chance:

1. Sample 10,000 random unit vectors `u ∈ R^{d_head}` (Gaussian, normalized).
2. Compute the baseline distribution `s(u) = ‖A^T u‖₂`.
3. Let `t5 =` 5th percentile of that distribution.
4. **Dead fraction of W_K:** generate the key-side energy profile — for each right
   singular direction of B, the unit key direction is `U_B[:, i]` with weight
   `S_B[i]²`. Dead fraction = `Σ_{i: s(U_B[:,i]) < t5} S_B[i]² / Σ_i S_B[i]²`.

Also compute a **matched random baseline**: replace A and B with random Gaussian
matrices scaled to the same Frobenius norms, recompute everything once per head.
Report real minus random. (Random `A^T B` already has a decaying spectrum —
Marchenko–Pastur-like; the claim is always "deader than chance", never "decaying".)

### 3.4 RoPE band analysis (skip for gpt2)

RoPE rotates head dimensions in 2D planes `(2i, 2i+1)` with frequencies
`θ_i = base^(-2i/d_rot)` (base usually 10000; `d_rot = d_head` for full RoPE,
`d_rot = d_head // 4` for Pythia). Higher `i` = slower rotation = longer range.

1. Partition the rotary planes into 4 log-spaced frequency bands; for Pythia, the
   un-rotated 75% of dimensions form a fifth, literal **DC band**.
2. For each band `b`, restrict rows: `A_b = A[dims_b, :]`, `B_b = B[dims_b, :]`
   (a plane contributes both its dimensions).
3. Recompute §3.2 and §3.3 per band.
4. The **low band (+ DC band for Pythia)** score is the *long-range reachability*;
   the high band is short-range only.
5. Optional richer version: `M_b(δ) = A_b^T R_δ,b B_b` for relative offsets
   δ ∈ {1, 8, 64, 512, 4096}, where `R_δ,b` is the block-diagonal 2D rotation by
   angles `δ·θ_i` for planes in band b. Report `sum(svdvals)` vs δ per band.

### 3.5 GQA handling (qwen25, qwen3)

One KV head serves a **group** of query heads. For KV head `g` with query-head group
`H(g)`:
- Compute all per-query-head quantities of §3.2–3.4 for each `h ∈ H(g)` against the
  shared `B_g`.
- The **group score** of any key direction is the **max** over `h ∈ H(g)` of the
  per-head score. Dead fraction uses the group score (a key must be dead to ALL its
  readers).
- Report both per-query-head and group-level numbers.

### 3.6 QK-norm variant (qwen3 only)

Qwen3 applies RMSNorm to q and k per head after projection. Magnitude information is
destroyed; the logit is (learned temperature) × cosine. Consequences:
- Raw pullback norm is NOT the right runtime score. Use the **directional score**:
  `s_dir(k) = ‖A^T k̂‖₂ / S_A[0]` where `k̂ = k/‖k‖` (i.e., normalized by the top
  singular value; ranges (0,1]; measures alignment of the key direction with W_Q's
  strong subspace).
- Run the census with `s_dir` in place of `s`. Everything else is unchanged.
- The gpt2/pythia vs qwen3 comparison of dead structure is itself a result: it tests
  whether amplitude carried gating information that QK-norm confiscated.

### 3.7 Phase 1 outputs

One parquet/CSV per model: `census_{model}.parquet` with columns:

```
layer, head, kv_head, band, S_A_sum, S_B_sum, S_M_sum, erank_A, erank_B, erank_M,
misalign_index, dead_frac, dead_frac_random_baseline, t5_threshold, is_group_level
```

Plus one `.npz` per model with the raw spectra vectors, AND per head: the top-5 right
singular directions of W_K in head space (`U_B[:, :5]`, shape `[d_head, 5]`) and the
soft-kernel basis of W_Q (left singular vectors of A below the 5th-percentile
threshold, up to 8 vectors). These are required for the outlier-parking check
(prediction 3): without stored directions, only spectra can be re-analyzed.

Plots (matplotlib, one PDF per model):
1. Heatmap: dead_frac over (layer × head), one panel per band.
2. Histogram of dead_frac across all heads (bimodality check — prediction 2).
3. Misalignment index vs layer (scatter, one point per head).
4. Real-vs-random dead_frac scatter (diagonal = "no deader than chance").

---

## 3B. Phase 1.5 — Certified QK truncation (run on gpt2 FIRST) + null model

**Goal:** functional validation of the census. Truncate each head's QK interaction to
the directions the census says are live, with a PROVEN bound on logit perturbation,
and show perplexity does not move. This converts the census from descriptive to
causal. No training anywhere.

### 3B.1 Truncation recipe (per head, exact)

Using the SVDs already computed in §3.2 (`A = U_A S_A V_A^T`, `B = U_B S_B V_B^T`,
economy, `V_*` shape `[d_model, d_head]`):

```
C  = diag(S_A) @ (U_A.T @ U_B) @ diag(S_B)     # [d_head, d_head]
P, Sig, Qt = svd(C)                             # M = (V_A P) diag(Sig) (V_B Q)^T
# rank-r factors (r chosen per head in 3B.2):
Wq_new = diag(sqrt(Sig[:r])) @ (V_A @ P[:, :r]).T    # [r, d_model]
Wk_new = diag(sqrt(Sig[:r])) @ (V_B @ Qt.T[:, :r]).T # [r, d_model]
```

Then `Wq_new.T @ Wk_new` equals the best rank-r approximation of the head's true
interaction matrix `M = W_Q^T W_K`. Patch the model so this head computes logits as
`(Wq_new x) · (Wk_new y) / sqrt(d_head)` (keep the ORIGINAL `sqrt(d_head)` scaling;
V and O paths untouched). Biases: fold `k`-bias into a per-key constant, keep as-is.

### 3B.2 The certificate (choose r per head)

Truncation error for any inputs: `|Δlogit| ≤ Sig[r] * ‖x‖ * ‖y‖ / sqrt(d_head)`
where x, y are the post-LayerNorm residual vectors feeding the head.

- **Uniform bound (gpt2):** pre-affine LN output has exact norm `sqrt(d_model)`, so
  `‖x‖ ≤ max|γ| * sqrt(d_model) + ‖β‖` using that layer's LN gain γ and bias β.
  Plug in for both x and y → hard bound `E_r` per head per rank.
- **Distributional bound:** measure the 99.9th percentile of `‖x‖` on ~100k
  calibration tokens; same formula. Report both.
- Choose per-head `r_h(ε) = min r s.t. E_r ≤ ε` for
  `ε ∈ {0.01, 0.05, 0.1, 0.5, 1.0}` (logit units; softmax sensitivity ~ e^ε).

### 3B.3 Evaluation

1. For each ε: truncate ALL heads at their `r_h(ε)`; report WikiText-2 perplexity
   (stride-512 eval, **≥ 200k tokens — 4k is underpowered, do not ship less**) vs
   the untouched model, plus mean rank kept and parameter/cache-width reduction.
2. Sanity: on 10k calibration tokens, record observed max |Δlogit| per head and
   verify observed ≤ certified bound (if violated, the LN bound or the patch is
   wrong — STOP).
3. One curve: perplexity delta vs compression fraction, annotated with ε. Expected
   shape: flat then cliff. Report where the cliff starts vs census dead_frac.
4. **Empirical (uncertified) sweep — REQUIRED after the v1 result:** the uniform
   certificate is vacuous on gpt2 (flat spectra × worst-case quantifier ⇒ rank 64
   chosen everywhere; v1 measured this). The functional question must therefore be
   answered empirically: truncate all heads at fixed global ranks
   r ∈ {60, 56, 48, 40, 32, 24, 16} (same refactor recipe), report the full
   perplexity curve. Separately, per-head ranks proportional to census liveness
   (keep enough directions to cover 1−dead_frac of S_B² energy) vs a matched
   uniform-rank control — this tests whether the census predicts WHERE to cut.
5. **Covariance-weighted certificate (rung 2):** replace `‖x‖·‖y‖` in the bound
   with empirical quantiles of the PROJECTED energies `‖P_cut x‖, ‖P_cut y‖`
   measured on calibration tokens (P_cut = the truncated directions). LayerNormed
   residuals spread energy ~uniformly over d_model, so projections onto a
   (64−r)-dim cut are ~`sqrt((64−r)/768)` of the norm — expect 10–40× tighter
   bounds. Report certified compression under this bound at the same ε grid, with
   the quantile (99.9%) stated as the certificate's probabilistic condition.

**Anomaly-log requirement (repeat offense):** "no truncation occurred at any ε" is
a mandatory anomaly-log entry, not a silent table row. Any run where a phase's
core operation is a no-op MUST say so in prose in REPORT.md.

**RoPE caveat:** this recipe is exact only for non-RoPE models (gpt2). For RoPE
models the interaction varies with offset; truncation must preserve rotary plane
pairing — do NOT run 3B on RoPE models in this pass; gpt2 only.

### 3B.4 Alignment-entailment null model (settles whether the dead tail is a separate finding)

The lighthouse (top directions under-parked, measured park0 ≈ 0.03 vs 0.125) may
GEOMETRICALLY entail excess tail deadness via orthogonality. Test: per head,
generate 500 synthetic (A', B') pairs: random Gaussian matrices with the head's
true spectra (S_A, S_B), where the top-k left singular vectors of B' are rotated to
match the head's measured top-k alignment with A' (k = 5; match the measured
`park0..park4` overlap values), and the remaining directions are random in the
orthogonal complement. Compute dead_frac on each synthetic pair. Report per head:
measured dead_frac vs the null distribution's mean ± std, and the z-score.
- If measured ≈ null: the tail deadness is bookkeeping entailed by the lighthouse —
  ONE finding, not two. Report honestly.
- If measured >> null: the excess is a separate structure (oubliette or gradient
  shadow) — Phase 2's token-type analysis and the checkpoint dynamics discriminate.
- **v1 result: measured << null (median z = −14; 130/144 heads BELOW entailment).**
  The tail is RESCUED relative to lighthouse-entailment. Follow-up REQUIRED:
  rerun the null at k ∈ {5, 10, 20, 32} to produce the alignment-depth profile —
  the depth at which measured dead_frac meets the null is the effective width of
  the aligned band per head. Report that width's distribution and its correlation
  with the census observables. The three excess-deadness exceptions
  (L4.H11 z=+57, L5.H0 z=+28, L7.H2 z=+3.4) get individual panels: spectra,
  alignment profile, and (in Phase 2) their runtime key-cloud occupancy.

### 3B.5 Outputs

`truncation_gpt2.csv` (per head × ε: r_h, E_r, observed_max_dlogit, params_kept),
`ppl_curve_gpt2.csv`, `null_model_gpt2.csv` (per head: dead_frac, null_mean,
null_std, z), plots for each, and the §8 report items extended with a Phase 1.5
table (ppl at each ε) and the null-model verdict.

### 4.1 Data collection

For each Phase-2 model:

1. Prefill a long document (≥ 4k tokens; use a technical article or code file;
   include a distinctive "needle" sentence at ~25% depth, e.g.
   `"The maintenance code for the auxiliary pump is 7413."`).
2. Capture, per layer and head, for every prefill position:
   - the **pre-RoPE key** `k` (hook the k-projection module output BEFORE rotary is
     applied — in HF Llama/Qwen/GPTNeoX code the rotation happens inside the
     attention forward AFTER the linear projection, so hooking the `k_proj` /
     sliced fused output gives pre-RoPE keys; verify per §6.4),
   - `‖v‖₂` of the corresponding value vector.
3. Decode 256 tokens greedily from a prompt that ends with a question requiring the
   needle (`"What is the maintenance code for the auxiliary pump?"`). Capture, per
   decode step, per layer/head: the post-softmax attention weights over all cached
   positions. (Set `attn_implementation="eager"` and `output_attentions=True`, or
   hook manually; SDPA/flash paths do not expose weights.)

### 4.2 Scores and target

Per cached key (layer, head, position):
- **x-axis (write-time score):**
  - Non-QK-norm models: low-band pullback `s_low(k) = ‖A_low^T k_low‖₂` computed
    from the PRE-RoPE key (low band ≈ unrotated ≈ valid at long range). For
    Pythia use DC band + lowest rotary band.
  - Qwen3: directional score `s_dir` from §3.6 on the normalized pre-RoPE key.
  - GQA: group max across the query heads sharing this KV head.
- **y-axis (ground truth):** total attention mass received =
  `Σ_{decode steps} Σ_{query heads in group} attn_weight`, restricted to decode
  queries at relative offset > 256 from the key (exclude the recency window —
  the claim is about long-range reachability).

### 4.3 Analysis

1. **Wedge plot** per head (and a pooled per-model version with per-head rank
   normalization — rank-normalize x within each head first, THEN pool; never pool
   raw scores): scatter x vs y, log y. Compute the "wedge violation rate": fraction
   of keys in the bottom 20% of x that land in the top 5% of y. Prediction: ≈ 0.
2. **Sieve simulation:** sort keys by x ascending; simulate evicting the bottom p%
   for p ∈ {10, 25, 50, 75}; report the attention mass that would have been lost
   (recompute nothing — just sum the y of evicted keys / total y). Per head and
   per model.
3. **Needle check:** report the x-score percentile of the needle sentence's keys in
   every head. Prediction: high percentile in at least the retrieval-relevant heads
   (the needle must never be sieved).
4. **Optional (prediction 3):** compute the top-5 principal components of the whole
   key cloud per head (these capture outlier/shared directions); report the mean
   pullback score of those directions vs the mean over random directions.
5. **Oubliette check (token-type analysis):** label every cached position by coarse
   type: {punctuation, stopword/function word, newline/whitespace-like, number,
   content word (everything else), needle}. Per head, report the type composition of
   the bottom-decile-by-x keys vs the full cache. Hypothesis: the dead zone is
   disproportionately punctuation/function/whitespace tokens (the model files
   don't-care keys to be unfindable). If instead dead-zone occupancy is
   type-uniform, the gradient-shadow account gains and the oubliette loses.

### 4.4 Calibration covariance (optional refinement, cheap)

From the decode-time queries collected in 4.1, estimate per head
`Σ_q = mean(q q^T)` (`[d_head, d_head]`). Alternative x-axis:
`s_Σ(k) = sqrt(k^T Σ_q k)`. Report whether the wedge is cleaner under `s_Σ` than
under the uniform pullback. (This tests: weights-only score vs data-aware score.)

> **HAZARD — analysis only, never an eviction signal.** `Σ_q` estimated from
> observed queries infers unreachability from non-use. A needle's relevant query
> arrives in the FUTURE (built from question tokens that do not exist at scoring
> time), so context-observed `Σ_q` will systematically under-score needle keys and
> bust long-range retrieval if used to evict. The weights-only pullback (§1) is a
> sup over ALL possible residual streams — it covers unwritten future queries and
> is the only score in this spec licensed for eviction decisions. Observed-query
> statistics may be used for the OPPOSITE polarity only: certifying keys as ALIVE
> (pinning/protection), never as dead. If comparing the two scores, report the
> divergence per head — it measures the gap between "cannot be looked at" and
> "has not been looked at".

### 4.5 Phase 2 outputs

`wedge_{model}.parquet`: columns
`layer, head, kv_head, position, s_low, s_dir, s_sigma, v_norm, attn_mass, is_needle`
Plots: wedge scatter grid (per head), sieve-loss curves, needle percentile heatmap.

---

## 5. Suggested repository layout

```
deadkeys/
  common/
    loading.py        # model loading, per-arch weight slicing (§6), verification
    rope.py           # frequency ladder, band partition, R_δ construction
    spectra.py        # §3.2–3.3 computations
  scripts/
    01_census.py      # Phase 1, all models, CPU. args: --model TAG
    02_census_plots.py
    03_collect_runtime.py   # Phase 2 §4.1 capture. args: --model TAG --text FILE
    04_wedge.py             # Phase 2 §4.2–4.4 analysis + plots
  outputs/            # parquet, npz, pdf
  texts/              # input corpus + needle
```

Dependencies: `torch` (CPU fine), `transformers`, `numpy`, `pandas`, `pyarrow`,
`matplotlib`. No training, no backward passes anywhere.

---

## 6. Pitfalls — read before writing loading.py

### 6.1 GPT-2 uses Conv1D, not nn.Linear
`transformer.h[l].attn.c_attn` is a Conv1D whose `.weight` has shape
`[d_model, 3*d_model]` — the TRANSPOSE of nn.Linear convention. To get the §1
convention (`[d_head, d_model]`, mapping model→head):
```python
W = block.attn.c_attn.weight.T          # now [3*d_model, d_model]
Wq, Wk, Wv = W.split(d_model, dim=0)    # each [d_model, d_model]
A = Wq[h*d_head:(h+1)*d_head, :]        # heads are contiguous row blocks
```

### 6.2 Pythia (GPTNeoX) fuses QKV **interleaved per head**
`attention.query_key_value.weight` has shape `[3*d_model, d_model]` but the layout is
**per-head interleaved**, not three stacked blocks:
```python
W = layer.attention.query_key_value.weight        # [3*d_model, d_model]
W = W.view(n_heads, 3, d_head, d_model)           # verify against modeling_gpt_neox
A = W[h, 0]     # W_Q for head h, [d_head, d_model]
B = W[h, 1]     # W_K
```
Check the reshape against the installed `modeling_gpt_neox.py` — if the model's
forward uses `view(..., num_heads, 3*head_size)` then split, the above is correct.
**Do not assume the GPT-2 layout here.**

### 6.3 Pythia partial rotary
`config.rotary_pct = 0.25`: only the FIRST `d_head//4` dimensions of each head are
rotated; the rest are never rotated (the literal DC band). The band partition of
§3.4 must respect this.

### 6.4 Mandatory sanity check before any analysis
For each model, one tiny forward pass on a 10-token input with hooks: reconstruct
`q_manual = A @ x` and `k_manual = B @ x` from the sliced weights and the hooked
residual-stream input (post-LayerNorm input to the attention block — hook the
`ln_1`/`input_layernorm` output), and compare to the hooked q/k projections
(pre-RoPE) with `torch.allclose(..., atol=1e-4)` after matching dtype. **If this
fails, the head slicing is wrong; stop.** Note: models with biases (gpt2, qwen25 has
q/k biases) — include the bias in the reconstruction; the census itself uses the
weight matrix only (record biases separately; their pullback can be added as a
constant offset later).

### 6.5 Attention-weight capture
`output_attentions=True` requires `attn_implementation="eager"` at load time for
Llama/Qwen-family models. Flash/SDPA silently return None.

### 6.6 Never pool raw scores across heads
Different heads = different private geometries. All cross-head aggregation must be
on rank-normalized or fraction-type quantities.

### 6.7 QK-norm location (qwen3)
The q/k RMSNorm is per-head, applied after projection and before RoPE. For §3.6 and
§4.2, hook AFTER the norm, BEFORE the rotation; or apply the norm manually to the
hooked projection output (weights at `self_attn.q_norm` / `k_norm`).

---

## 7. Acceptance criteria (what "done" looks like)

1. §6.4 sanity check passes on all six models.
2. `census_*.parquet` + plot PDFs exist for all six; random baselines included.
3. `wedge_*.parquet` + plots for the four Phase-2 models; wedge violation rate,
   sieve-loss curves, and needle percentiles reported in a summary table.
4. A `RESULTS.md` stating, for each pre-registered prediction: confirmed / refuted /
   ambiguous, with the one plot that decides it.

No interpretation is required from the implementer — the four predictions in §0 are
the only claims being tested, and the plots listed are sufficient to adjudicate them.

---

## 8. Review bundle (for external review — REQUIRED)

Produce a single self-contained `REPORT.md` (plus PNG plots) designed to be read
WITHOUT access to the parquet files or the machine. It must contain, inline as
markdown tables:

1. **Run manifest:** model tags + exact HF revisions, dtype, corpus files with token
   counts, needle text and its token position, decode length, date, wall-clock per
   phase, and the git hash / random seed.
2. **Sanity-check table:** per model, §6.4 max abs reconstruction error (q and k).
3. **Census summary table:** per model × band: median and IQR of dead_frac across
   heads, same for the random baseline, misalignment index median, and the 5 heads
   with highest / 5 with lowest dead_frac (layer.head ids + values).
4. **Bimodality stat:** per model, dip-test p-value or simply the histogram bin
   counts of dead_frac in 10 bins (raw counts, so the shape is readable as text).
5. **Wedge table:** per model: wedge violation rate (overall and worst single head),
   sieve-loss table (attention mass lost at p = 10/25/50/75% eviction, median and
   worst head), and needle percentile (min across layers of the max across heads —
   i.e., the needle's worst best-case).
6. **Prediction adjudication:** the four §0 predictions, each with
   confirmed/refuted/ambiguous, the single deciding number, and the filename of the
   one deciding plot.
7. **Anomaly log:** anything that surprised the implementer or required deviation
   from this spec (hook point changes, dtype issues, layers skipped), with reasons.
   An empty anomaly log on a first run is itself suspicious — say so if truly empty.

Plots referenced in §6 must be exported as PNG (not only PDF), ≤ 1500px wide, so
they can be uploaded alongside the report. Numbers in tables: 3 significant digits.
The report should be readable top-to-bottom as evidence, worst news first.
