# Review of Chapter 11: Transformers

## Scope and method

This review covers the chapter introduction and all seven source notebooks in `chapter_transformers/`: Transformer blocks, GPT, KV caching, encoders and decoders, vision transformers, mixture-of-experts models, and scaling laws. I checked the committed PyTorch and JAX outputs rather than inferring results from source code alone. I also checked the Basics part, the sequence-model chapters, Chapter 10 on attention, the optimization chapter, and the optimization appendix before calling material missing.

Several deliberate boundaries are appropriate and should remain. Tokenization, language-model objectives, recurrent sequence models, decoding algorithms, and basic attention are already introduced earlier. The optimization chapter and appendix cover optimizers, schedules, and hyperparameter transfer. A later systems/distributed chapter will cover FSDP, DeepSpeed, distributed training, serving stacks, and parallelism in detail. Those subjects should be cross-referenced here, not duplicated.

The review does **not** recommend adding multi-seed sweeps to the published notebooks. Where an experimental claim needs robustness evidence, that work belongs in author-side validation; the deployed notebook should either report a deliberately narrow conclusion or include only a cheap invariant/check. Notebook runtime is part of teaching quality.

## Overall assessment

The chapter has an excellent high-level ambition. Instead of presenting only the 2017 encoder--decoder, it gives learners a coherent path from a configurable block to a small language model, cached decoding, bidirectional and encoder--decoder use, ViTs, MoE models, and scaling laws. The direct mapping from equations to compact code is often very good. The small trained models make otherwise abstract distinctions visible, and most displayed numerical claims agree with the committed outputs.

The principal problem is epistemic calibration. A toy ablation is frequently described as establishing a general mechanism; an asymptotic or idealized systems statement is presented without its omitted term; and a family-level trend becomes “every” or “the” modern design. There are also several concrete errors: the GPT-2 compute comparison, the claim that KV caching makes generation linear, the attribution of the post-LN mechanism to Xiong et al., a JAX patch-embedding mismatch, and a wrong Qwen3 KV-head count. Correcting these matters more than adding breadth.

## Highest-priority corrections

| Priority | Location | Problem | Recommended correction |
|---|---|---|---|
| Critical | `transformer-block.md`, post-LN discussion | The text attributes post-LN instability to the mechanism in Xiong et al. (2020), but that paper's mean-field result is that post-LN has large expected gradients near the output layers at initialization and motivates learning-rate warmup. It does not establish the notebook's Q/K rank-collapse or “attention receives no gradient” explanation. | Cite Xiong only for its actual initialization-gradient result. Present the notebook as one constructed failure mode, measure more than `W_q`, and cite separate rank-collapse work only within its assumptions. |
| Critical | `gpt.md`, compute comparison | The loaded GPT-2 small model has about 124M parameters, roughly 26 times the notebook's 4.7M model, not 60 times. With roughly 10B training tokens, `6ND` gives about `7.4e18` FLOPs, not `1e20`. The latter is closer to the largest 1.5B GPT-2 variant. | Separate the four GPT-2 sizes and their training estimates. State exactly which checkpoint was loaded and compare like with like. |
| Critical | `kv-cache.md`, opening and complexity summary | KV caching does **not** turn complete autoregressive generation from quadratic to linear in generated length. It removes repeated projection and FFN work, but each new query still reads and attends over `t` cached keys/values, so total cached-attention work and cache traffic are `Theta(T^2)`. | Give a term-by-term complexity table: uncached dense layers `Theta(T^2)` plus uncached attention `Theta(T^3)` over all decoding steps; cached projections/FFNs `Theta(T)`; cached attention and KV reads `Theta(T^2)`. Explain why cached decoding can nevertheless look nearly flat per token at short contexts. |
| Critical | `kv-cache.md`, PyTorch `forward_step` | `is_causal=(T > 1)` is not sufficient when a multi-token chunk is appended to a nonempty cache. A rectangular query-by-key mask needs an offset; otherwise the causality relation is wrong. | Either restrict the interface explicitly to empty-cache prefill or one-token decode and assert it, or construct an offset causal mask for chunked prefill/decode. |
| Critical | `vision-transformer.md`, JAX patch embedding | JAX uses `padding='SAME'`, while PyTorch uses an unpadded patch convolution and `num_patches` is computed with floor division. For image dimensions not divisible by patch size, JAX produces a ceiling number of patches and can disagree with the positional-embedding length. | Use `VALID` in JAX to match PyTorch, or assert exact divisibility and document the policy. Add a cheap cross-framework shape test. |
| Critical | `scaling-laws.md`, FLOP accounting | The prose defines `N` as non-embedding parameters, but the profiler sums all two-dimensional weights, including the vocabulary embedding/output matrix. A tied output matrix still performs a large dense vocabulary projection. The claim that `6ND` with non-embedding `N` exactly matches the profile is therefore inconsistent. | Define a core parameter count and add the vocabulary-projection term, or define an “active matmul parameter” count that includes the tied head. Use the same convention in equations, code, and captions. |
| Critical | `scaling-laws.md`, modern model table | The Qwen3-235B-A22B entry gives 64 query and 8 KV heads. The official Qwen3 model card/blog specifies 64 query heads and **4 KV heads** for that MoE model; 64/8 applies to a different dense family member. | Give one row per exact model rather than combining family ranges. Verify every architectural field against the corresponding official configuration. |
| High | `encoders-decoders.md`, alignment explanation | A reversed target does not mathematically force raw cross-attention maps to be anti-diagonal. Contextual encoder states can mix source positions and downstream layers can distribute the computation. | Describe the observed anti-diagonal as an interpretable solution learned by this shallow model, not a necessary representation. Keep the reported 100% PyTorch and 92% JAX head-averaged argmax alignment as evidence for this run. |
| High | `vision-transformer.md`, CNN comparison | The experiment uses different architectures, learning rates, compute, and optimization responses. Equal parameter counts, data, and epochs do not isolate “inductive bias” as the cause of the accuracy gap. | Present the result as a descriptive comparison consistent with the known data efficiency of convolutional priors. Move causal attribution to a literature-backed discussion or author-side controlled study; do not lengthen the deployed notebook with seed sweeps. |
| High | JAX notebooks | `vision-transformer.md` still uses `nnx.view`, an API known to fail with the current Flax/NNX environment. | Replace it with the current train/eval state mechanism and keep the static compatibility scan that rejects deprecated NNX calls before publication. |

## 1. Transformer Blocks

### Mathematical and conceptual precision

The residual-stream presentation is a strong organizing idea, but “every modern model has moved to pre-norm” is contradicted by designs discussed later in the same notebook. OLMo 2 places normalization after the attention and MLP sublayers but before the residual addition; other recent models use both pre- and post-sublayer normalization or Peri-LN-style variants. Say that pre-norm became a common stable baseline, then introduce newer alternatives as attempts to control residual scale and optimization.

The post-LN diagnostic is instructive, but its conclusion currently exceeds its measurement. It reports the gradient norm of `W_q` only. Even if query/key logits collapse, the value and output projections may receive gradients, and the residual/FFN paths remain active. “Attention receives no gradient” should become “the measured query projection receives negligible gradient in this constructed configuration.” If the intended lesson concerns the whole sublayer, print the Q, K, V, and O gradient norms; this is a cheap diagnostic, not a repeated training sweep.

The citation should be corrected. [Xiong et al. (2020)](https://proceedings.mlr.press/v119/xiong20b.html) analyze expected gradient magnitudes at initialization and find large gradients near the output in post-LN Transformers; their result explains the need for warmup in their setting. It is not evidence for the exact rank-collapse story in this notebook. Rank-collapse results also have architectural assumptions, and should not be transferred automatically to a residual block with an FFN.

The statement that pre-norm residual RMS grows as `sqrt(L)` is a useful initialization heuristic under approximately independent, similarly scaled branch increments. It is not a general law after training, where branch outputs are correlated and their scale is learned. Make those assumptions visible in the equation or caption.

RMSNorm is correctly described as omitting mean subtraction, but “same quality” is too categorical. Quality and stability depend on architecture and task. The timing result is also implementation- and hardware-specific; a framework may fuse LayerNorm just as effectively. Report the measured device, dtype, shape, synchronization policy, and uncertainty, and conclude only that the displayed implementation was faster in that environment.

QK normalization controls query/key norms but does not fully “pin” attention logits. Angles, learned projection geometry, optional learned scales, temperature, and positional transformations still affect magnitude. Also distinguish LayerNorm-normalized Q/K, RMS-normalized Q/K, and explicit L2/cosine attention; they are not identical.

The feed-forward section should avoid retrospective inevitability. The `4d` hidden width was a historical convention, not a mathematical constant, and has been revisited frequently. GELU did not “end the debate” about ReLU's kink; modern models use GELU, SiLU/SwiGLU, squared ReLU, and other gates. The parameter-matched `8d/3` SwiGLU arithmetic is useful and should remain.

Parameter share does not establish where “knowledge lives.” FFNs often hold most dense-block parameters and can implement key-value-like associations, but representations and factual behavior are distributed across embeddings, attention, normalization, and residual streams. Replace the ontological claim with a precise observation about parameter and activation allocation.

### Code and experiment design

The configurable block is a good teaching spine, but the alleged “weight-for-weight 2017 encoder block” is not such a reproduction: the 2017 Transformer used ReLU, explicit dropout, and particular dimensions and normalization details, whereas the configured cell uses GELU. Call it a compact post-norm encoder-style block or configure the actual historical choices.

Validate configuration strings instead of silently mapping every non-`rms` value to LayerNorm and every non-`gelu` value to SwiGLU. Add `d % num_heads == 0` assertions. These fail-fast checks improve teaching because a typo becomes a meaningful error rather than a different architecture.

The GELU/SwiGLU comparison supports a conclusion about the shown small model and optimization settings. Equal learning rates are not necessarily equally favorable to both activations. The prose says offline runs were checked; that is adequate author-side practice. The published experiment should stay short and use qualified wording rather than adding seed trials.

The chapter would benefit from a compact normalization-position diagram showing the exact algebra of post-norm, pre-norm, post-sublayer norm, and pre+post norm on a shared residual stream. This is a better use of space than more training curves and would prevent terminology from becoming ambiguous.

## 2. GPT: From Block to Language Model

### Correctness and scope

“Every serious language model uses BPE” is false. WordPiece, unigram SentencePiece, byte-level schemes, and tokenization-free/byte models are important counterexamples. The intended teaching point is that subword tokenization remains a common practical compromise. Cross-reference the earlier tokenization treatment rather than restating it.

The modern GPT configuration should either include the QK normalization just advocated in the previous notebook or explain why this implementation intentionally remains GPT-2-like. Likewise, distinguish attention-probability dropout from residual-branch dropout; “dropout” is not a single interchangeable location.

The decoding complexity discussion needs a precise decomposition. In a naive implementation, each step recomputes all `t` token projections and FFNs, producing `Theta(T^2)` cumulative dense-layer work, while the full `t x t` attention at every step produces `Theta(T^3)` cumulative attention work. With a KV cache, dense work becomes linear but attention remains cumulatively quadratic. This prepares the next notebook without making an incorrect one-line claim.

The sampling helper should reject nonpositive temperatures, clamp or validate `top_k`, and support EOS termination. These are small code-quality improvements that expose the mathematical domain of the operations.

The JAX fixed-size generation buffer avoids recompilation, but it still evaluates the final-sized graph at each step. Say explicitly that this is a compilation-stability teaching device, not an efficient decoder.

### Pretrained-weight verification

The weight-layout explanation is good, especially the transposition of GPT-2's Conv1D-style matrices. Readable output and a plausible short-corpus perplexity do not prove every weight was mapped correctly. Add one cheap regression check: compare the logits for a fixed short token sequence against the reference Hugging Face checkpoint within a tolerance, or store a checksum/reference slice. This is much more diagnostic than another training run and adds negligible runtime.

Correct the scale comparison. The notebook model is about 4.7M parameters and processes about 16M sampled tokens, so `6ND` is approximately `4.5e14` FLOPs. GPT-2 small is about 124M parameters—roughly 26 times larger—and, under a rough 10B-token estimate, costs about `7e18` FLOPs. A figure near `1e20` belongs to a much larger GPT-2 variant or a different token/accounting assumption. Label all such numbers as estimates because the original training mixture, reuse, and exact implementation matter.

“Nothing in the code changes across eleven orders of magnitude” is pedagogically attractive but false in practical terms. The block abstraction survives, but data pipelines, numerics, kernels, optimizer state, parallelism, fault tolerance, and often architectural details change substantially. Since systems are reserved for a later chapter, one sentence and a forward reference are sufficient.

### What the toy training run establishes

The aggressive-learning-rate post-LN ablation shows that this configuration reached a loss near the unigram baseline and failed to learn useful next-token structure. It does not show that all post-LN models learn nothing, nor that attention specifically is the sole failure. A gentler learning-rate claim should either point to the displayed result or be framed as author-side observation.

`val_loss` averages batch means. If the last batch is smaller, examples are not exactly token-weighted. Sum losses and token counts or drop the incomplete batch. This has little effect here but is the mathematically correct utility.

If generated text is described as “verbatim,” establish corpus overlap for the quoted span. Otherwise say that the sample appears memorized or closely matches training text. More data is the most direct cure for a tiny corpus, but it is not the only response to overfitting: early stopping and regularization can improve validation behavior, even though they do not create missing linguistic coverage.

## 3. KV Caching and Autoregressive Inference

### Complexity and cache semantics

The chapter should lead with a table rather than “quadratic to linear”:

| Operation over `T` generated tokens | No cache | KV cache |
|---|---:|---:|
| Q/K/V, output projections and FFNs recomputed over prefixes | `Theta(T^2 d^2)` | `Theta(T d^2)` |
| Attention score/value mixing | `Theta(T^3 d)` | `Theta(T^2 d)` |
| Stored KV activations | transient | `Theta(L T H_k d_h)` |

Constants and prompt length matter, but the table makes the asymptotics honest. A cached step is not simply “about `2N`, independent of `t`”: it also reads the cache and performs attention proportional to context length. At small contexts, model-weight reads and launch overhead can dominate, so a timing curve may look nearly flat. Present that as the observed regime rather than the general curve.

The teaching PyTorch cache concatenates tensors at every step. That copies the existing cache and reallocates storage, adding avoidable traffic. It is acceptable for clarity only if labeled explicitly. A short preallocated-cache variant or diagram can then explain why real systems use fixed buffers or paged blocks; detailed serving implementation belongs in the later systems chapter.

The causal-mask contract is currently unsafe for chunked decoding. PyTorch's `is_causal=True` for a rectangular `T`-query by `(t+T)`-key matrix does not by itself express the desired absolute-position offset in every API/kernel. Restrict and assert the two supported calls—empty-cache prefill and single-token decode—or build the explicit offset mask.

The manual cached forward hardcodes the pre-norm order and bypasses generic block behaviors such as post-norm and dropout. It is therefore a cached implementation of the default GPT configuration, not of every `TransformerBlock` the chapter constructed. Encode that restriction in the type/API or implement cache support in the attention/block modules themselves.

For JAX, assert `t + T <= max_len`; dynamic slice/update primitives may clamp out-of-bounds starts rather than report the semantic error a student expects. Avoid assuming every layer has identical head/KV dimensions unless the model constructor enforces it.

The cache-size formulas and the GPT-2 example are otherwise sound: two tensors per layer yield `2 L H_k d_h` elements per token, and the quoted FP32 GPT-2-small values follow from that convention. Comparisons with weights must specify dtype, GQA/MQA, batch size, and whether prefixes are shared.

### Roofline discussion

The decode estimate of roughly half a FLOP per byte for a one-token dense weight multiply is a useful lower-bound intuition, but it omits cache reads, activation traffic, quantization metadata, and fusion. Similarly, “prefill is compute-bound” is not universal: short prompts, small batches, small models, and inefficient kernels can remain memory- or launch-bound. Label the printed arithmetic intensity as an idealized dense-layer estimate.

The chapter correctly notes that decode and prefill have different bottlenecks. Keep this conceptual distinction and forward-reference continuous batching, paged attention, quantized caches, and distributed serving rather than implementing them here.

### GQA and MLA

GQA code should assert both `d % H == 0` and `H % H_kv == 0`. The RoPE helper also needs a cache-position offset before it can actually be dropped into cached decoding as the prose suggests. `H_kv = H` recovers the MHA architecture, but not identical output unless the weights are mapped identically.

The small GQA quality comparison changes the total parameter count as well as KV sharing. It is still a useful demonstration, but should not be called parameter-matched. “Four to eight query heads per KV head” is a common contemporary choice, not a near-universal rule.

The MLA section needs a closer correspondence to the architecture it names. DeepSeek MLA uses a learned low-rank KV latent together with a decoupled RoPE key component, and matrix absorption is central to avoiding full reconstructed K/V materialization during decode; see the [DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437). A per-sequence SVD is an oracle compression diagnostic, not the model's mechanism. If the SVD basis is sequence-specific, its storage must be counted; only learned global projection weights are amortized. One 1,024-token passage can show that one activation matrix is compressible, not establish a universal low-rank premise. Narrow the conclusion and add a diagram of the actual latent/cache/data path.

### Attention sinks

The displayed results do support a large first-token attention mass in the tested GPT-2 passage and show that preserving a sink token can reduce the damage of a narrow post-hoc window in that example. They do not establish that lost sink capacity, rather than lost context, is always the dominant source of degradation. The experiment modifies a pretrained full-attention model without retraining, so it diagnoses abrupt masking for that model and passage.

“The sink stores nothing” is stronger than an attention-weight plot can establish. Attention mass can be high while the value/OV contribution is small, but that contribution should be measured before assigning the mechanism. Rephrase as a normalization or routing hypothesis and distinguish attention weight from causal contribution.

## 4. Encoders, Decoders, and Encoder--Decoder Models

### Modeling distinctions

A causal representation is built from the prefix **including the current token**, not “the left half.” An encoder is not inherently unable to generate text: masked iterative models, non-autoregressive models, and diffusion-style text models are counterexamples. The precise statement is that an unmasked encoder does not directly implement the usual left-to-right autoregressive factorization.

The simplified all-mask MLM objective is a reasonable teaching device, and the notebook later discusses BERT's 80/10/10 corruption. Label the trained model “BERT-like MLM encoder,” not BERT itself, because it omits other historical BERT ingredients. The actual edge-masking results are useful: bidirectional context substantially improves interior-token loss while giving little or no advantage at the boundary in both committed frameworks.

The final position is not “exactly the causal language-model situation.” The MLM was trained with random masks elsewhere, has a different objective, and the first/last positions differ in which side is available. The result estimates the value of two-sided context for this MLM; it is not a direct controlled estimate of causal-LM performance.

Avoid “strongest representation” and “backbone of choice” claims. Encoder, decoder, and encoder--decoder choices depend on objective, latency, data, and output structure. “Random strings never match” should become “collisions are negligible at this length/alphabet” or use a deterministic disjoint split.

### Cross-attention experiment

The reverse-sequence task and visualization are excellent teaching choices. The conclusion must preserve causal alternatives: because encoder states are contextual, the target token need not attend only to its raw aligned source position. The observed anti-diagonal is evidence that this shallow trained model found a direct-alignment solution. It is not forced by the task.

The implementation omits source-padding and target-padding masks and uses fixed-length synthetic data. Add one compact mask-composition example: source padding for encoder self-attention, causal AND target-padding masks for decoder self-attention, and source padding for cross-attention. This is a genuine practical omission not supplied by the earlier attention chapter, which teaches the primitives but not their encoder--decoder composition.

Teacher forcing, exposure bias, greedy/beam sampling, and evaluation are already treated in the sequence-model and decoding material; add precise cross-references rather than duplicating them.

### Latent bottlenecks

Perceiver-style cross-attention is not fixed-cost with respect to input length. It costs roughly `O(M N d)` to read `N` inputs into `M` latents, followed by `O(M^2 d)` latent self-attention, plus projection terms. The latent stack avoids `N^2`, but it still must read the input. Likewise, the shown benchmark compares different architectures and amounts of work; it demonstrates the expected regime but is not a controlled speed comparison of “the same two blocks.”

The statement that most current vision-language models use learned query tokens is too broad. Many use MLP/projector mappings, pooled or patch tokens, or resamplers. Present query tokens as one important interface pattern.

The final taxonomy should leave room for encoder-only systems, diffusion transformers, and non-autoregressive generation. Decoder-only architectures are dominant in many current language-generation settings, not “everything else,” and generation is not computationally “free.”

## 5. Vision Transformers

### Architecture and code

The core patchify--project--prepend--position--transform sequence is clear. Correct the JAX padding mismatch and remove `nnx.view`. Add an explicit divisibility assertion in both frameworks; silent edge-patch policies are a poor first implementation.

The text says a ViT has “no idea” that tokens lie on a grid. Absolute positional embeddings do not directly encode a two-dimensional adjacency rule, but the shared strided patch convolution already imposes local extraction, weight sharing, and a regular rasterization. A more precise contrast is that a plain ViT lacks the strong translation-equivariant locality of a CNN after tokenization.

Translation equivariance itself needs qualifications. Ideal stride-one convolution on an infinite or periodic domain is translation equivariant. Zero padding, boundaries, stride, pooling, and patchification break exact equivariance for some shifts. Call it a strong built-in prior, not an unconditional theorem enjoyed by the entire CNN classifier.

### Experimental inference and figures

The parameter-matched CIFAR comparison is informative but not causal. ViT and CNN use different optimal learning rates in the notebook (`0.1` versus `0.01`), different operations, and different compute. Equal epoch counts and parameter counts do not isolate inductive bias. Report exactly what the run shows: in this small-data, small-model setup, the selected CNN trains to higher accuracy than the selected plain ViT, a result consistent with stronger convolutional priors.

The learned-position result—neighboring embeddings having slightly higher cosine similarity than distant ones—is also descriptive. Without a comparison to initialization or another null distribution, it does not prove that the model discovered image geometry. Say the learned table is consistent with weak local structure. A cheap before/after statistic could support the interpretation without repeated training.

The old `vit.svg` is below the visual standard of the newer chapter diagrams. Replace it with a clean two-panel figure: raster-to-patch-token geometry on the left, and token/position/class flow through the residual stack on the right. Show tensor shapes and use consistent color to track the class token. Avoid decorative 3D blocks and crossing arrows.

Hierarchical/local vision transformers, relative spatial bias, masked image modeling, and self-supervised pretraining deserve only a concise “where modern vision models went next” box with links to the later vision material. They should not turn this notebook into a survey.

## 6. Mixture of Experts

### Economics and routing

The stored-versus-active parameter distinction is central, but “idle experts charge nothing” is false. Inactive experts still consume memory capacity, placement effort, checkpoint/storage bandwidth, and potentially communication or weight-loading cost. Active parameter count is not identical to latency or serving cost because routing, all-to-all communication, load imbalance, and small expert GEMMs matter. Introduce this caveat conceptually and defer mechanisms to the systems chapter.

The Mixtral arithmetic is accurate under its stated convention: about 46.7B total and 12.9B active parameters. The DeepSeek-V3 figures should explicitly distinguish the 23.1B routed-expert active parameters from the roughly 37B total active parameters once shared/dense components are included.

“Training touches every expert” is false per step. Only selected experts receive routed tokens and expert-weight gradients in ordinary sparse MoE training; across a sufficiently varied run, all healthy experts may be used. Experts can also be sharded across devices rather than resident together on one accelerator.

Routing collapse is a risk, not an inevitable stable endpoint. “Dead below 2%” is an arbitrary visualization threshold and should not be defined as mathematical death. A balanced token histogram also does not prove expert specialization or improved capacity; it proves only more even routing.

The pedagogical implementation evaluates every expert and then masks, so its actual notebook compute scales with all experts. The text acknowledges this, but every subsequent “same compute” comparison should repeat that it refers to an ideal sparse kernel/runtime, not the code being timed.

### Faithfulness to modern routers

The auxiliary-loss-free routing cell is inspired by DeepSeek but is not an exact implementation of DeepSeek-V3. The reported design uses sigmoid affinity scores, group-limited routing, per-expert biases for selection, and normalized selected gates. The notebook uses a full softmax and adds a bias only for top-k selection. Label it a simplified controller and link to the exact equations in the [technical report](https://arxiv.org/abs/2412.19437).

With softmax probabilities, selected gates can still send gradient to unselected router logits through the denominator. The exercise usefully asks learners to inspect this; bring the answer into the main text because it prevents the common misconception that top-k makes every unselected logit gradient exactly zero.

Fine-grained experts create many possible expert combinations, but the combinatorial count does not mean training realizes or uses them uniformly. A shared expert is motivated as a place for broadly useful computation; the architecture does not guarantee that “common knowledge lives once.”

Add a short conceptual box covering capacity factors/token dropping versus dropless routing, router z-loss/stability, and load versus importance balancing. Capacity is already an exercise and detailed dispatch is systems material, so equations and a failure-mode diagram are enough.

Avoid calling MoE the universal frontier default. It is a major scaling strategy and appears in important frontier models, while strong dense models remain. Date model examples and distinguish architecture choice from economic inevitability.

## 7. Scaling Laws

### FLOPs and parameter conventions

The derivation of approximately `12 L d^2` parameters per standard dense Transformer layer is useful under the stated MHA and `4d` FFN assumptions. It does not transfer unchanged to GQA, gated FFNs with different ratios, local attention, or MoE. Keep the formula as a baseline and annotate the substitutions.

Resolve the embedding/output-count inconsistency described in the priority table. A transparent per-token forward estimate is:

- core dense matmuls: approximately `2 N_core` FLOPs;
- vocabulary logits: approximately `2 V d` FLOPs when the full head is evaluated; and
- sequence attention: an additional context-dependent term.

Training then multiplies relevant matmuls by roughly three for forward plus backward, yielding the familiar approximate coefficient six only under a declared counting convention. Optimizer updates are `O(N)` per step, not per token; whether they are negligible depends on tokens per optimizer step.

The XLA/JAX cost analysis is useful corroboration, not exact ground truth. The notebook itself observes that it misses a SwiGLU matmul in one path. Call it a compiler estimate, document its version dependence, and do not use “exact authority.” The JAX profiling cell should use distinct PRNG keys for `X` and `Y`; identical random arrays do not affect FLOP counts, but teach poor key discipline.

### What a scaling law is

A raw validation loss generally does not become a perfect line in log--log coordinates because it approaches an irreducible/data-dependent entropy floor. Teach the standard form

`L(N,D) = E + A N^{-alpha} + B D^{-beta}`,

or a clearly labeled approximation to excess loss. This is the missing conceptual center of the notebook. It naturally leads to compute-optimal allocation under `C approximately 6ND`, and to uncertainty about fitted exponents and constants.

The five-model PTB experiment is a useful miniature capacity curve, but it is not a reproduction of a compute-optimal scaling law. Width, depth, head count, learning rate, and total compute all change; there are only five points; and the largest-point bend can reflect optimization or architecture as well as data limitation. The committed losses show diminishing validation improvement and a growing train/validation gap. Describe precisely those observations.

Do not add a large sweep to the deployed notebook. Use a synthetic schematic or a small fit to published measurements to teach the two-variable law, and retain the tiny experiment as a fast illustration of diminishing returns. Any fit validation or multiple trials belongs in author-side testing.

### Chinchilla and data-constrained training

The “about 20 tokens per parameter” Chinchilla result is an empirical compute-optimal finding for the studied model/data/optimizer regime, not a universal constant. The original question concerns allocation at fixed training compute and mostly fresh training tokens; see [Hoffmann et al. (2022)](https://arxiv.org/abs/2203.15556). This notebook samples about 16.4M training tokens from a much smaller corpus, so repeated exposure is central. Dividing unique corpus tokens by 20 is a heuristic warning, not a derived maximum useful parameter count.

The held-out corpus segment is valuable for a fast notebook but does not establish broad “English generalization”; it measures generalization to that corpus split. Mention data-constrained and repeated-token scaling as a modern qualification, without attempting a full experimental treatment.

The inverse-width learning-rate rule used in the sweep is not full maximal-update parametrization, especially while depth and other dimensions change. Call it an empirical stabilization rule for this notebook or connect it carefully to the optimization appendix's more complete treatment.

### Modern architecture table

One row should represent one exact released configuration, with a citation to the official model card/config. Family ranges encourage accidental combinations of incompatible values. At minimum include exact `d_model`, layers, Q heads, KV heads, FFN/expert widths, local/global attention pattern, normalization positions, RoPE settings, and active/total parameter definitions where relevant.

The current “constructor call” display is intentionally compact but too lossy to support the claim that modern models differ only in a few arguments. It omits QK norm, local windows, RoPE base/scaling, normalization placement, tokenizer/vocabulary, bias policy, and MoE routing. The Qwen correction should use the [official Qwen3 release information](https://qwenlm.github.io/blog/qwen3/). OLMo 2 is a useful counterexample to a universal pre-norm narrative; its reordered RMSNorm and QK normalization should be represented accurately.

Rename the subsection from a complete “modern recipe” to “selected block-level design patterns.” A real recipe also includes initialization, optimizer, schedule, batch/tokens, data mixture, precision, and systems choices, most of which belong elsewhere. Window/sink attention is itself an architectural long-context choice, so the statement that architecture appears nowhere in the long-context list is internally inconsistent.

The notebook should end by distinguishing:

1. training-compute-optimal model/data allocation;
2. data-constrained scaling with repeated or limited high-quality data;
3. inference-aware or lifetime-compute-optimal design; and
4. architecture/system changes that alter constants or effective scaling.

Only the first needs a derivation here. The remaining three can be concise forward pointers. Claims that capabilities “emerge abruptly” should be qualified: apparent discontinuities depend on metric resolution, prompting, and thresholding, even when underlying loss improves smoothly.

## Figures and visual design

The strongest visuals are the ones that expose a computation or comparison directly: gradient-flow traces, cross-attention alignment, cache growth, and routing histograms. Several can be improved by making the inferential limit visible in the figure itself.

- Add tensor shapes and an explicit residual stream to the Transformer-block diagram. Use a four-column comparison for normalization placement rather than multiple prose descriptions.
- Replace “flat” cache timing rhetoric with a plot that overlays measured latency and the theoretical context-dependent term. State prompt length, batch, dtype, device, synchronization, and whether compilation/warmup is excluded.
- Draw KV caching as append-only preallocated blocks; annotate the current concatenation code as pedagogical.
- Draw GQA and MLA with distinct query heads, shared KV heads, cached latent, decoupled positional key, and reconstruction/matrix-absorption paths. The present compression story otherwise invites an incorrect “SVD at inference” mental model.
- Preserve the encoder/decoder mask diagrams, but add padding-mask composition and label axes/normalization direction on attention heatmaps.
- Replace `vit.svg` with a restrained, shape-accurate raster-to-token figure. Use no decorative perspective, avoid crossings, and make the class token and positional information traceable by color.
- In scaling figures, show the irreducible-loss floor and distinguish observed points, fitted region, and extrapolation. Never imply a power law by drawing a straight line through five heterogeneous toy models.

## Writing and terminology

The chapter relies too much on confidence-signaling absolutes. Search and revise “every,” “nothing,” “exactly,” “never,” “for free,” “the entire difference,” “charge nothing,” and “inevitable.” Often the repair is just to state the population: “in this implementation,” “for the displayed run,” “in many current decoder-only LMs,” or “under the dense-MHA assumptions above.”

The word “honest” appears repeatedly as a way to certify a comparison or equation. Replace it with the property actually meant: “matched parameter count,” “includes projection FLOPs,” “measured on held-out tokens,” “reports compiler-estimated operations,” or “uses the same training budget.” This removes rhetorical assurance and makes the evidence auditable.

Avoid anthropomorphic and teleological claims such as an architecture “discovering” geometry, a shared expert “holding common knowledge,” or evolution “settling” a debate unless followed by an operational measure. These metaphors are memorable, but they currently carry more causal meaning than the experiments establish.

Standardize notation across notebooks:

- use `d`, `d_h`, `H`, and `H_kv` consistently;
- distinguish prompt length, generated length, and total context;
- distinguish stored, total, routed-active, and matmul-active parameters;
- distinguish attention probabilities from OV contributions;
- distinguish exact complexity from an implementation-dependent timing regime; and
- state whether loss is per token and whether averages are token-weighted.

## Completeness after checking neighboring chapters

The chapter is already broad. The following are genuine additions or clarifications; most should be boxes, diagrams, or cross-references rather than new long notebooks.

1. **Mask composition in encoder--decoders.** Chapter 10 supplies mask primitives, but no current section assembles source padding, target padding, causality, and cross-attention validity into the three masks a practical seq2seq model needs.
2. **A correct two-variable scaling-law model.** Neither the optimization appendix nor the new optimization chapter replaces the missing `E + A N^-alpha + B D^-beta` formulation and compute-optimal allocation argument.
3. **Normalization taxonomy.** The material names many schemes but lacks a single algebraically precise comparison of pre-, post-, post-sublayer, pre+post, and Peri-LN-style placement.
4. **Cache API invariants.** State valid shapes, absolute positions, maximum length, prefill/decode modes, and offset masking. This is model correctness, not distributed-serving detail.
5. **MoE routing failure modes.** A concise treatment of capacity, dropping/dropless routing, router stability, and what balance metrics do and do not prove is needed before the later systems treatment.
6. **Training-optimal versus inference-aware scaling.** One paragraph should explain why equal training loss/compute does not imply equal deployment cost; implementation belongs later.

The following should **not** be added here as substantial duplicate material:

- optimizer algorithms, schedules, and full muP derivations: optimization chapter/appendix;
- beam search, teacher forcing, and sequence metrics: earlier sequence/decoding chapters;
- basic attention, RoPE derivations, FlashAttention, window attention, and induction circuits: Chapter 10;
- detailed FSDP, DeepSpeed, tensor/pipeline/expert parallelism, continuous batching, vLLM/SGLang, and cluster networking: planned systems/distributed chapter;
- full BERT pretraining history and downstream applications: later representation-learning material; and
- comprehensive vision-model taxonomy: later computer-vision chapters.

## Recommended revision order

1. Fix the factual and mathematical errors in the priority table, especially post-LN attribution, GPT-2 compute, KV-cache complexity/masking, patch padding, parameter-count conventions, and Qwen3 configuration.
2. Repair the runnable code: remove `nnx.view`, enforce shape/configuration invariants, correct cache offsets, and add the cheap GPT-2 reference-logit check.
3. Rewrite conclusions around what each displayed experiment actually establishes. Keep the notebooks fast; do not add deployment-time seed sweeps.
4. Rebuild the scaling-law section around the two-variable law and explicit counting conventions.
5. Add the three high-value diagrams: normalization placement, cache semantics/complexity, and faithful GQA/MLA data flow.
6. Add the compact missing-topic boxes and cross-references, then perform an absolute-word and “honest” pass across the entire chapter.

After these changes, the chapter would retain its unusually practical breadth while meeting a top-course standard: equations with declared assumptions, demonstrations with appropriately bounded conclusions, code whose contracts are explicit, and modern examples verified against primary model documentation.
