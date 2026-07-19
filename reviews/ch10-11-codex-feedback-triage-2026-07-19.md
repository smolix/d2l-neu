# Triage: Codex reviews of ch. 10 Attention and ch. 11 Transformers

*2026-07-19. Inputs: `attention_feedback.md` (≈90 items), `transformer_feedback.md`
(≈85 items). Every factual claim verified by two Opus passes against the source
files, the committed executed outputs, live runs in both venvs, PyPI, and the
official model cards. Verdicts: REAL-DEFECT / PARTIAL (kernel of truth, smaller
fix than proposed or already partly handled) / FALSE-ALERT / STYLE-JUDGMENT /
POLICY. This file records the verdict; fixes are a separate pass.*

## Headline

The review is high quality — most PARTIALs are worth acting on — but **both of
its loudest "Critical" alarms failed verification**, and several other items
criticize hedges the files already contain. Tallies:

| | REAL | PARTIAL | FALSE | STYLE | POLICY |
|---|---|---|---|---|---|
| Attention (~90 items) | 8 | ~30 | 9 | ~30 | ~8 |
| Transformers (~85 items) | 4 | ~27 | 10 | ~18 | ~9 |

## Confirmed real defects (fix pass, in priority order)

**Attention**
1. **TinyCharLM's circuits equations are affine, not linear** — `qkv`/`proj`
   ship with default biases while `what-attention-computes` claims the fixed-
   pattern map "is linear". Cleanest fix: `bias=False` + re-capture (also
   strengthens the analyzability story). (A-#57)
2. **Tied-head "arbitrary bigram lookup table"** — with tying, zero-block
   logits are `E Eᵀ e_x`: symmetric, PSD, rank ≤ d. Call it a constrained
   low-rank bigram model. (A-#3)
3. **"holds to the byte"** contradicted by the cell's own printed output
   (n=2048: 40.6 MiB measured vs 32.0 predicted; exact only for n≥4096 and in
   JAX). Reword prose + slide. (A-#45)
4. **"three orders of magnitude beyond n=2d"** — real ratios are 8–64×
   (d=4096, n=131072 → 16×). (A-#5b)
5. **Flax `.value` deprecation warnings live in committed JAX outputs of
   what-attention-computes** (the earlier sweep covered chapter_transformers
   only). Fix + re-capture. (A-#7b)
6. **BMM described as "elementwise product"** (scoring). (A-#24)
7. **"training longer widens the gap"** asserted with no displayed cell —
   remove or attribute to author-side observation. (A-#43)

**Transformers**
8. **GPT-2 compute comparison wrong for the loaded checkpoint** — 124M is 26×
   our 4.7M (not "sixty times"); 6ND ≈ 7e18 (not "~1e20", which fits the 1.5B
   XL). Also echoed in a slide and scaling-laws. (T-#2)
9. **Qwen3-235B-A22B KV heads: 4, not 8** (64:8 is the dense Qwen3-32B);
   verified against the HF config + Qwen blog. moe.md's expert row is correct.
   (T-#7)
10. **ViT JAX PatchEmbedding uses `padding='SAME'` vs PT VALID** — latent
    mismatch in a `#@save` class for non-divisible image sizes (verified:
    30×30/patch-4 → 64 tokens vs 49 embedding slots). `VALID` + divisibility
    assert both tabs. (T-#5)
11. **"weight for weight … the encoder block of 2017" uses GELU** — 2017 was
    ReLU, and `FeedForward` doesn't offer ReLU. Rename or add the option.
    (T-#19)

**Latent-safety PARTIALs promoted into the fix pass:** `forward_step`'s
`is_causal=(T>1)` would miscompute a multi-token chunk into a non-empty cache
(verified top-left alignment; currently unreachable — add the assert); JAX
cache `t+T ≤ max_len` assert (dynamic_update_slice clamps silently); GQA
divisibility asserts; block config-string validation (typo silently changes
architecture); JAX chunked-attention `V.shape[-1]` + tail handling. (T-#4,
T-#42, T-#39, T-#18, A-#48)

## False alerts (do NOT "fix")

1. **Xiong misattribution (T-#1, review's Critical)** — misread: the file
   attributes rank collapse to Dong et al. 2021 explicitly and cites Xiong
   only for the warmup/init-gradient result.
2. **`nnx.view` "no longer provided / fails on Colab" (A-#7a, T-#10)** — flax
   0.12.7 is both the pinned and the *current PyPI latest* version; `nnx.view`
   exists, runs warning-free, and is used 43× book-wide including d2l/jax.py.
3. **"entropy doesn't support the >0.9-weight claim" (A-#20)** — refuted with
   a bound: max-prob capped at 0.9 over 64 keys forces entropy ≥ 0.325 nats;
   our measured <0.2 therefore implies typical max-prob >0.9.
4. **Windowed path-length sentence "inverted" (A-#4)** — not inverted;
   "keeping O(n/w) instead" pins the correct result. One verb ("gives back" →
   "gives up") is worth changing, nothing more.
5. **Perceiver "flat cost" (T-#51)** — the file already states the O(MN)
   reading cost and attributes flatness to the tested range.
6. **DeepSeek 23B-vs-37B active (T-#61)**, **Mixtral accounting (T-#68)**,
   **cache-size formula (T-#44)**, **anti-diagonal numbers (T-#55)** — all
   verified sound in the committed outputs.
7. **"five-point study isn't a scaling law" / fitted exponents (T-#75, #70)** —
   the file says exactly that itself and declines to fit, per house policy.
8. **GQA "parameter-matched" (T-#37)** — the file never claims it.
9. **Softmax "gradient never vanishes identically" (A-#14)**, **duplicate-key
   symmetry (A-#30)**, **"rule of thumb" already labeled (A-#16)**,
   **parallel-prefix memory disclosure (A-#55)**, **copy-test hedging
   (A-#63)** — all already correct or already hedged in the files.

## Notable PARTIALs (worth doing, smaller than proposed)

Masked-softmax fully-masked-row: shipped behavior (uniform over invalid keys)
is documented + probed by an exercise; add one warning clause, not the
proposed 5-step rewrite (A-#1). Gaussian→dot-product: drop "constant
norms"/BatchNorm, keep the expansion as the equal-norm special case (A-#2).
Single-head tagline: soften index/summary to the restricted-interface
separation the proposition actually proves (A-#6). KV-cache "quadratic →
linear": true for the dominant dense term at our sizes (and the file flags
attention as subdominant); add the cumulative-quadratic-attention caveat
(T-#3). Post-LN diagnostic: print Q/K/V/O grad norms or narrow the claim to
the query/key projections (T-#12). "Every serious LM uses BPE" → subword
tokenization (T-#25). Attention-sink "stores nothing": weight-only evidence;
measure the OV contribution or rephrase (T-#43). MLA oracle labeling: one
clause that "6× smaller" refers to a trained projection (T-#41). Aux-free
router: note the sigmoid/group-routing simplification vs DeepSeek-V3 (T-#62).
Plus the absolutes pass (exact/entire/every/never/free) and the **"honest"
purge** — the latter aligns with docs/writing-avoid.md §4 and is endorsed.

## Policy items (owner-decided)

Approved by Alex 2026-07-19: **mask-composition** treatment (scoring + the
encoder–decoder mask assembly), **two-variable scaling law**
L(N,D)=E+AN^{-α}+BD^{-β} taught as the published form (no fitted exponents),
**normalization-placement taxonomy diagram**. CPU policy: no CPU-fallback
requirement for GPU-measurement notebooks; cells that gain nothing from a GPU
should not require one. Remaining open: GPT-2 reference-logit regression
check (cheap, recommended), vit.svg redraw in house style (recommended —
one-style-per-chapter rule), attention-dropout paragraph, MHA shape ledger,
claims-by-regime box, circuit-limits box, benchmark-provenance annotations.
