# Literature survey — RoPE as a removable scaffold / RoPE↔NoPE

**Compiled:** 2026-07-23 (web survey; not a systematic review). Supports the
[program charter](../README.md). Companion: [`novelty-check.md`](novelty-check.md).

## The direct hit — and a name collision

- **DroPE** — *Extending the Context of Pretrained LLMs by Dropping their Positional
  Embeddings* (Sakana AI, Dec 2025, [arXiv 2512.12167](https://arxiv.org/abs/2512.12167) ·
  [blog](https://sakana.ai/drope/)). This is the method behind this program's motivation:
  pretrain with RoPE → **remove positional embeddings from all layers → short recalibration**
  (reported: RoPE checkpoint ~14B tokens, drop, recalibrate ~2B tokens; **< 1% of the
  pretraining budget**). Reported results: **matches full-RoPE in-context perplexity**, **beats
  NoPE-from-scratch**, and gives **zero-shot context extension** outperforming RoPE-scaling and
  specialized long-context architectures on LongBench / RULER. Framing: *positional encoding is
  a training scaffold, not a permanent necessity* — RoPE gives training stability (NoPE-from-
  scratch suffers vanishing gradients) but "warps semantic attention by compressing low
  frequencies" under extrapolation, so drop it after it has served its training-time purpose.
  **DroPE justifies this with optimization theory, not interpretability evidence — the gap this
  program targets.**
- **DRoPE** (name collision, unrelated) — *Directional Rotary Position Embedding* (Mar 2025,
  [arXiv 2503.15029](https://arxiv.org/abs/2503.15029)), a RoPE variant for autonomous-driving
  agent trajectory modeling. Not about removal.

## The RoPE↔NoPE landscape

- **RNoPE-SWA** — *Rope to Nope and Back Again* (Jan 2025,
  [arXiv 2501.18795](https://arxiv.org/html/2501.18795v1)). Hybrid: ~25% NoPE full-attention
  layers + ~75% RoPE sliding-window. Finding directly relevant to claim C3: *"NoPE layers excel
  at information retrieval, while RoPE layers handle local information (recency bias)."* On
  RULER 8k→256k a RoPE-scaling baseline lost ~41% retrieval / ~44% QA; the hybrid lost only
  ~22% / ~23%. Establishes RoPE = local/recency, NoPE = retrieval/long-range — **correlationally**.
- **Kazemnejad et al. 2023** — *The Impact of Positional Encoding on Length Generalization*
  (NeurIPS, [arXiv 2305.19466](https://arxiv.org/pdf/2305.19466)). NoPE generalizes to longer
  sequences better than RoPE/ALiBi and needs no extra compute; NoPE can represent absolute+
  relative PE and learns T5-relative-like patterns.
- **Haviv et al. 2022** — decoder-only transformers without any PE still learn positional
  information (via the causal mask). Origin of "emergent position."
- **Position Information Emerges … via Similarity of Nearby Embeddings** (Jan 2025,
  [arXiv 2501.00073](https://arxiv.org/abs/2501.00073)) — mechanism for NoPE emergent position.
- **Deconstructing Positional Information: From Attention Logits to Training Biases** (May 2025,
  [arXiv 2505.13027](https://arxiv.org/pdf/2505.13027)) — nearest to our M1.5, but theoretical
  (6-layer synthetic; spectral-contraction/optimization argument) and does **not** analyze
  pre-vs-post-rotation keys. See [`novelty-check.md`](novelty-check.md).

## Assessment of the program's two motivating "why"s

- **"RoPE adds a positional handle but it isn't load-bearing for retrieval; NoPE-with-emergent-
  position may be as good or better" — strongly supported and potentially large.** Mainstream
  frontier (Kazemnejad, RNoPE-SWA, DroPE). The large effect is concentrated in **length
  extrapolation and long-range retrieval**, where RoPE actively hurts and dropping it helps.
- **"RoPE is compute waste" — true but not the compelling axis.** RoPE's per-layer rotation is
  cheap in FLOPs; the documented cost is **length-extrapolation harm / semantic warping**, and
  DroPE's efficiency claim is about avoiding expensive long-context finetuning (< 1%
  recalibration), not inference FLOPs. Frame the case on the extrapolation axis, not FLOPs.

## Bottom line

The **method** (drop RoPE post-training) exists and is validated (DroPE). What is unclaimed is
the **mechanistic explanation** for why the scaffold is removable — which is exactly what the
k-address-space M1.5 (emergent redundancy) + M1.6 (non-addressability) findings supply, and
what this program is designed to establish causally (claims C1–C4).
