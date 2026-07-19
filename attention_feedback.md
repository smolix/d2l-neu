# Review of Chapter 10: Attention

## Scope and method

This review covers the six source notebooks in `chapter_attention/`, the chapter introduction, and the committed executed outputs under `outputs/pytorch/chapter_attention/` and `outputs/jax/chapter_attention/`. I also checked the neighboring Transformer chapter, the sequence-model chapters, the optimization chapter, and the mathematics appendices before classifying material as missing. In particular, I do **not** count cross-attention applications, transformer normalization/FFNs, full encoder--decoder models, state-space models, or optimizer details as omissions here: those are covered in the immediately following or explicitly cross-referenced chapters.

The executed artifacts are useful evidence. They show that the main experiments currently run in both PyTorch and JAX and that many numerical statements in the prose match the displayed results. They do not, by themselves, justify claims outside the demonstrated architecture or regime. This review does not recommend adding repeated-seed sweeps to deployed notebooks. Robustness checks can remain part of author-side testing; the published notebooks should stay reasonably fast.

## Overall assessment

This is already an unusually ambitious attention chapter. Its arc—from kernel regression, through masking and multi-head attention, to positional information, efficient attention, and a mechanistic case study—is substantially stronger than the conventional treatment that stops after the Transformer equations. The online-softmax derivation, the restricted single-head proposition, the permutation-equivariance proof, and the induction-circuit notebook are especially good teaching choices.

The main weakness is not lack of material. It is that the prose repeatedly promotes a useful special case, a toy experiment, or a hardware-dependent observation into a universal conclusion. A smaller number of statements are mathematically or computationally wrong. The highest-priority revision should therefore tighten scope and assumptions rather than add more experiments.

## Highest-priority corrections

| Priority | Location | Problem | Recommended correction |
|---|---|---|---|
| Critical | `attention-scoring.md`, masked softmax, lines 204--215 | Replacing every score in a fully masked row by the most negative finite value produces a **uniform distribution over invalid keys**. That is not a safe degradation: the attention output becomes an average of invalid values and can leak padded or otherwise forbidden content. | Either guarantee and assert that every query has at least one valid key, or return zero for fully masked rows after softmax. Teach arbitrary boolean masks as the primary interface and valid lengths as a convenient prefix special case. |
| Critical | `attention-scoring.md`, lines 39--52 | The Gaussian-to-dot-product motivation drops `-||k_i||^2/2` on the claim that LayerNorm or BatchNorm makes key norms nearly constant. LayerNorm is applied before learned key projection in a standard transformer; it does not make `W_k x_i` have constant norm. BatchNorm is not the standard transformer normalizer. Dropping the term can change key rankings substantially. | Present scaled dot product as a learned compatibility function with favorable matrix-multiplication structure. Retain the Gaussian expansion only as the exact special case in which projected key norms are constant, or use cosine attention after explicit Q/K normalization. |
| Critical | `what-attention-computes.md`, lines 160--164 | With a tied embedding/output matrix, the zero-block logits are `E e_x`, i.e. a symmetric positive-semidefinite Gram matrix of rank at most `d`, **not an arbitrary vocabulary-by-vocabulary lookup table**. It cannot represent arbitrary bigram statistics. | Say it is a constrained bigram model. If the intended claim is an arbitrary bigram table, use an untied output head and state the rank condition. |
| Critical | `attention-at-scale.md`, lines 594--597 | The text says windowed attention “gives back the O(1) path length ... keeping O(n/w) instead.” The intended statement is the opposite: windowing **gives up** constant path length; the worst path across the sequence becomes `Theta(n/w)` layers/hops. | Correct the sentence and distinguish graph-theoretic reach from effective information transmission. `1 + L(w-1)` is the receptive-field bound, not a guarantee that distant information is learned or preserved. |
| High | `attention-at-scale.md`, lines 67--88 and 103--125 | The architecture comparison charges CNNs and RNNs for their `d^2` transforms but lists attention as only `O(n^2 d)`, omitting Q/K/V/output projections. The later full-layer count correctly adds `8nd^2`. The table is therefore not comparing like with like. The claim that production contexts are “three orders of magnitude” beyond `n=2d` is also numerically false for common widths/contexts: e.g. `d=4096`, `n=131072` is only 16 times the crossover. | Put full self-attention cost `Theta(nd^2+n^2d)` in the comparison table, or explicitly label the row “attention mixing only.” Replace the orders-of-magnitude claim by worked ratios for representative configurations. |
| High | `multihead-attention.md`, opening and summary | The single-head proposition is correct under its stated construction, but the chapter/index turns it into “one head is provably not enough” without preserving the assumptions: position-only keys, independent random values, a single shared mixture, and no content-dependent representation that can encode both values. | Retitle the result as a separation for a restricted one-layer attention interface. State explicitly that it does not prove every one-head transformer is incapable of the task once values, keys, residual paths, depth, or preceding computation are allowed to change. |
| High | JAX cells in `attention-scoring.md:438`, `multihead-attention.md:367,437,466` | These cells call `nnx.view(..., deterministic=True)`. The current pinned Flax/NNX API no longer provides the `nnx.view` used by these notebooks; this is the same class of Colab failure already observed elsewhere. | Use module `eval()`/`train()` state or the current NNX split/merge/state API consistently. Add a static compatibility check that rejects `nnx.view` and deprecated `.value` access before notebook publication. |
| High | `attention-at-scale.md` | The notebook is not CPU-runnable: it unconditionally calls CUDA allocator APIs, forces PyTorch FlashAttention, and requests JAX `implementation='cudnn'`. A CPU Colab or local CPU environment fails rather than merely running slowly. | Mark the GPU-only cells, guard them with capability checks, and provide a reduced CPU fallback or skip with an explanatory output. The conceptual chunked/windowed cells can remain device-independent. |

## Section-by-section review

### Chapter introduction

The database analogy is an excellent entry point, but “the analogy is exact except on one point” (`index.md:15`) is too strong. A conventional database query operates on externally defined keys and records and normally returns exact or ranked records. Neural attention usually constructs context-dependent keys and values through learned projections, returns a convex mixture, and composes that mixture with output projections and residual paths. Approximate nearest-neighbor lookup is also not the same operation as softmax averaging. Calling the analogy useful rather than exact would preserve the intuition without creating misconceptions.

The opening also says attention “refus[es] to summarize” and “keeps every entry.” That describes full attention during a forward pass, not attention as a mathematical necessity: windowing, pooling, latent bottlenecks, KV compression, and recurrent linear attention later in this same chapter all summarize or discard information. A more durable formulation is that attention provides content-dependent direct access to a collection of representations.

Several universals should be qualified and dated: “essentially every current model” uses RoPE, “the variants that survived are the ones we implement,” and the implication that approximate attention has not survived. ALiBi, learned relative biases, NoPE/interleaved NoPE, state-space/attention hybrids, retrieval augmentation, and task-specific sparse/linear mechanisms are real counterexamples. The chapter can still identify RoPE and fused exact attention as dominant defaults without claiming exclusivity.

### Queries, Keys, and Values

#### Mathematical correctness

- “Every network we have built so far assumes an input of fixed, known size” (`queries-keys-values.md:4`) is contradicted by the RNNs named two lines later and by CNNs with global/adaptive pooling. The actual limitation is a fixed-size communication state or fixed receptive field, not necessarily a fixed input shape.
- The statement that a database needs no compression (`lines 34--35`) is pedagogically risky. Real systems compress, index, shard, and approximate large databases precisely to make lookup effective. Neural attention itself learns compressed representations and later pays a quadratic access bill.
- The softmax map is differentiable, but “its gradient never vanishes identically” (`lines 98--100`) needs care. The softmax Jacobian always has the all-ones vector in its null space because adding a constant to all logits changes nothing. Gradients with respect to upstream parameters can also be exactly zero through symmetry, zero inputs, masked paths, or finite-precision saturation. The useful statement is that softmax is smooth and provides gradients away from saturated/degenerate directions.
- The Nadaraya--Watson statement (`lines 183--188`) should say “consistent for the conditional mean under regularity and bandwidth conditions.” That conditional mean is Bayes-optimal under squared loss. “Converges to the statistically optimal predictor” sounds like an unconditional rate-optimality result, which requires additional assumptions and is not what a weak-consistency citation alone establishes.
- “The kernel's precise shape matters far less than its bandwidth” is a reasonable rule of thumb, not a theorem for all dimensions, boundaries, tails, and smoothness classes. Label it as a heuristic for the illustrated setting.
- A fixed kernel does not treat every query alike in the literal sense: its weights change with the query. It uses the same geometry and bandwidth for every query. That is the precise limitation to contrast with learned representations.

#### Teaching and figures

The Nadaraya--Watson sequence is effective: the learner sees the estimator, its bias--variance behavior, and its attention map before any neural machinery appears. The bandwidth-collapse exercise is particularly valuable. The old `qkv.svg`, however, is much weaker than the newer diagrams: it is a generic box-and-arrow picture and does not visually distinguish score computation, normalization across keys, and value mixing. Replace it with one small worked lookup containing actual vectors/weights and an explicit “weights sum to one” annotation. The kernel figure and regression plots can remain.

### Attention Scoring and Masking

#### Scaled dot product and saturation

The variance calculation is correct under the stated independent, zero-mean, unit-variance model. It should be framed as an initialization-scale heuristic. In self-attention, queries and keys arise from the same residual stream and are neither independent nor unit-variance throughout training. The `1/sqrt(d_h)` factor stabilizes the conventional initialization scale; it does not permanently ensure unit-variance logits.

The entropy experiment supports the qualitative claim that unscaled logits sharpen as dimension grows. It does **not** support the sentence that entropy below 0.2 means “the average query puts a weight of more than 0.9 on a single key”: entropy and average maximum probability are different statistics, and the code never measures the latter. Print `alpha.max(-1).values.mean()` if that number is desired, or remove it.

“Gradients stop flowing to everything upstream of the scores” is too broad. Saturation suppresses the score/QK gradient path. Gradients still flow through the values, the output projection, and residual connections. Likewise, “softmax's gradient never vanishes exactly” should be replaced as discussed above.

The displayed Jacobian statistic is the mean Frobenius norm, but the prose sometimes speaks as if it measured all useful upstream gradient transmission. Explain why this norm is a diagnostic rather than the gradient of the training loss.

#### Masking

The dtype discussion is timely, but the proposed finite-minimum behavior is unsafe for fully masked rows. A robust teaching implementation is:

1. construct an arbitrary boolean validity mask;
2. fill invalid logits with `-inf` or a safe finite sentinel;
3. compute softmax;
4. set invalid probabilities to zero; and
5. renormalize only rows with at least one valid key, leaving fully invalid rows zero.

This also creates the right place to teach mask broadcasting and composition: padding mask AND causal mask AND optional structural mask. Prefix lengths are compact, but production attention frequently uses packed sequences, block masks, and non-prefix validity patterns.

The sentence that the triangular mask is “the entire difference” between reading and generation is false. A causal mask permits an autoregressive factorization; generation also requires a shifted next-token objective, an autoregressive decoding loop, and a probabilistic output head. Encoder-only models can generate through iterative masked-token or diffusion-style procedures, while an unmasked transformer can serve in non-autoregressive generation.

At `lines 328--330`, batch matrix multiplication is called an “elementwise product.” It is a batch of matrix products, not an elementwise product.

#### Code/API

The PyTorch implementation exposes weights by mutating `self.attention_weights`; JAX returns them as an extra result. That difference is understandable, but downstream examples increasingly depend on it. Document the interface deliberately and avoid hidden mutable analysis state in reusable training modules where possible.

### Multi-Head and Cross-Attention

The proposition and its numerical verification are strong. The relative norm error `sqrt(1/2)` and the displayed approximately 0.707 values agree. The problem is scope, not algebra.

The cost discussion should separate ideal arithmetic from realized performance. Holding total width `d` fixed does keep the four main projection parameter count near `4d^2` and the dominant projection/attention matmul FLOPs independent of head count. It does not make heads “free”:

- the softmax, mask, dropout, and launch/layout work scale with the number of heads;
- smaller head dimensions can use kernels less efficiently;
- attention weights/masks have `B x H x n x n` shape; and
- fused kernels have backend-specific head-dimension constraints.

Similarly, “every production system folds heads into the batch” is false. Many fused APIs preserve an explicit head axis and fuse/reorder it internally. Teach reshape/transpose as one implementation strategy, not a production invariant.

Add explicit constructor assertions that `num_hiddens % num_heads == 0`. Later GQA code should similarly assert divisibility between query and KV head counts.

The alignment example is readable and the observed output supports its claims, but the prose should say “in this run” when interpreting the no-match row. A random no-match query can spuriously concentrate on one key; diffuseness is not guaranteed. Duplicate key embeddings split mass by symmetry only if their scores are exactly equal.

### Positional Information

#### Equivariance and absolute positions

The permutation-equivariance proposition and proof are correct for unmasked self-attention. The conclusion “attention computes a bag-of-tokens summary” is not: equivariance means per-token outputs reorder with inputs; it is not permutation invariance and does not reduce the sequence to one bag-level summary. Say that unmasked self-attention without positional signals cannot distinguish two sequences that differ only by a joint permutation except by permuting the corresponding outputs.

The binary-counter analogy for sinusoidal encoding is visually useful but loose. Frequencies are geometrically spaced, not powers of two, sinusoidal components are periodic, and “each row is unique” is not a safe global or finite-precision claim. Present binary counting as an analogy for multiple scales, not as the mathematical structure.

The rotation derivation for `[sin, cos]` is correct. When explaining why additive sinusoids do not make dot-product attention translation invariant, note that the illustrated raw sum is not the complete learned query/key projection. The generic conclusion remains true, but the code is an illustration, not a proof about every learned projection.

#### RoPE, ALiBi, and NoPE

The RoPE relative-score identity is correct in exact arithmetic. Add the assumptions that the rotated dimension is even and that the simple implementation rotates all head dimensions. Many deployed variants rotate only part of each head and use different bases or scaling schedules.

“Low-frequency pairs ... carry content” is inaccurate. Every rotated pair carries content; low-frequency pairs merely change phase slowly with position. RoPE does not allocate separate content-only dimensions unless the architecture explicitly leaves some dimensions unrotated.

The numerical shift test is directionally good. The committed JAX output changes by about `1.5e-3` to `1.8e-3`, versus `3.3` for additive sinusoids; the PyTorch output is around `1e-6` to `1e-5`. Thus “round-off” is fair, but “at least three orders of magnitude below” should be checked against the actual score scale before being stated universally.

The displayed ALiBi slope formula is the simple power-of-two-head case. The original construction has special handling for non-power-of-two head counts. State the restriction. A fixed distance bias imposes a prior, but “cannot fully unlearn” is too absolute: other layers and content logits can counteract it, even though the fixed additive bias itself is not trainable.

The NoPE explanation through the number of softmax competitors is too simple. The causal mask breaks permutation symmetry, but whether a finite-depth network can recover absolute or relative position depends on architecture, content, boundary signals, and depth. Present competitor count as one source of asymmetry, not a full mechanism.

#### Extrapolation experiment

The actual PyTorch and JAX outputs support the stated qualitative ordering in this deliberately minimal model: learned/sinusoidal embeddings degrade beyond 128, RoPE initially improves and then degrades sharply by 512, ALiBi stays flat, and NoPE changes little but starts worse. That is a useful result.

The interpretation needs a firmer boundary. `TinyCharLM` is an attention-only character model without FFNs or normalization, and the only position-sensitive benchmark is perplexity on resegmented held-out prose. It is not evidence that ALiBi is generally superior to RoPE in modern transformers. Long-context quality also includes retrieval, position-sensitive reasoning, and use of information far from the end; perplexity alone can be dominated by local statistics.

The model is described as having “only trainable machinery ... attention,” but it also has a trainable token embedding/tied output head. Say that attention is its only trainable **sequence-mixing** mechanism.

The evaluation helper should accumulate total negative log likelihood and token count rather than average batch means. More importantly, verify and state how much held-out text each context length covers: changing segmentation can change truncation/coverage, so “same held-out text” should be literal, not approximate.

Claims that “training longer widens the gap,” that simple RoPE rescaling hurts immediately, and that fine-tuning recovers performance are not backed by a displayed cell in this notebook. Either add a compact result table produced during author-side validation and committed as data, or narrow/remove those claims. This does not require rerunning repeated trials during deployment.

### The Cost of Attention

#### Arithmetic and memory

The full-layer formula `8nd^2 + 4n^2d` (ignoring lower-order terms and biases) is useful and correct for standard MHA. The earlier comparison table should use it consistently.

The memory arithmetic examples are correct, but the text overstates measurement precision. In committed PyTorch output, `n=2048` measures 40.6 MiB against a 32 MiB two-buffer prediction; larger points match 128/512/2048 MiB. Say that the quadratic term dominates and the prediction matches asymptotically after allocator/workspace overhead, not that it holds “exactly” or “to the byte.” JAX's compiler report happens to match the two-buffer prediction for the displayed shapes, but compiler memory analysis is not an immutable property of the Python expression.

“At inference time arithmetic is the less important half” is regime-dependent. Prefill can be compute-bound; autoregressive decode is commonly bandwidth-bound; short sequences and small models behave differently. This chapter itself later makes the more precise distinction.

#### Online softmax and FlashAttention

The online-softmax recurrence is mathematically correct in real arithmetic. “No approximation” should be followed by the standard numerical qualification: changing summation/tiling order can change floating-point results. The code correctly checks agreement rather than bit identity.

The JAX chunked implementation reshapes values with `d` (`V.reshape(-1, chunk_size, d)`) rather than `V.shape[-1]`. It therefore silently assumes value width equals key/query width, unlike the PyTorch code. It also requires `n` to be divisible by the chunk size, while the PyTorch implementation handles a tail block. Either assert both restrictions or implement padding/tail handling. Accumulators should explicitly use fp32 for half-precision inputs and cast the output back; this is especially important if the notebook presents the code as a miniature of production kernels.

The FlashAttention discussion correctly emphasizes I/O awareness, tiling, and backward recomputation; that agrees with the primary paper's claim that the algorithm reduces HBM traffic. “Moving bytes, not multiplying them, is the scarce resource” is still regime-dependent, and “there is no trade-off” is too broad. Backend availability, masks, dropout, determinism requirements, head dimension, requests for attention weights, and numerical behavior can select a different kernel. Say fused exact attention is the default when the backend supports the required semantics.

The benchmark is a valid demonstration of these particular kernels on the build hardware: PyTorch reports about 19.8 ms/4112 MiB naive versus 1.08 ms/16.5 MiB fused; JAX reports about 9.71 ms/4096 MiB versus 1.53 ms/negligible compiler temporary. Include hardware and software provenance next to the plot or caption so a student does not read these ratios as algorithmic constants.

#### Sparse and linear attention

The blocked window implementation assumes `n % w == 0` in both frameworks, although only the JAX comment admits it. It also assumes `d_v=d_k`, and JAX's zero padding defaults to float32 instead of inheriting `K/V` dtype. Add assertions or generalize it.

The statement that a query block needs only blocks `b-1` and `b` is correct for the chosen causal window of exactly `w`, but make the indexing diagram explicit. This is one of the few places where a small diagram of two adjacent blocks would clarify more than code.

The normalized linear-attention recurrence is correct when the feature map yields nonnegative similarities and the denominator is nonzero. `ELU+1` is mathematically positive but can approach zero numerically; add an epsilon and discuss accumulation precision. State complexity in terms of feature dimension `r` and value dimension `d_v`: the recurrent state is `r x d_v` plus `r`, not necessarily `d x d`.

The parallel-prefix implementation materializes an `n x r x d_v` tensor, so it is linear in sequence length but not necessarily memory-light during training. This distinction is important before comparing it with recurrent inference.

The benchmark compares short teaching implementations, not optimized production kernels. It supports the measured implementations and asymptotic trends; it does not establish that approximate attention loses in deployment “by the wrong margin.” Avoid broad claims that almost no approximation is deployed. The relationship to Mamba-2/structured state-space duality also needs its structural assumptions: a normalized nonlinear feature-map recurrence is not literally every linear state-space model.

### What Attention Computes

This is the chapter's pedagogical high point. It connects behavior, attention patterns, and weight products rather than treating heatmaps as explanations. The repeated-pattern task is well chosen because it distinguishes a positional shortcut from a content-based induction algorithm.

Several equations silently omit biases even though `TinyCharLM` constructs `nn.Linear`/`nnx.Linear` with biases. Therefore the displayed residual update is not exact, and fixing attention patterns does not make the model purely linear without affine terms. The simplest repair is to instantiate QKV/projection layers with `bias=False`; otherwise include affine contributions and call the path expansion affine.

“Whatever [a head] writes stays there” is too literal. Residual additions remain algebraically present, but later heads can cancel, rotate, overwrite in a functional sense, or make a direction irrelevant to the output. Describe the stream as additive bookkeeping, not permanent semantic memory.

RoPE affects direct QK scores, while values are unrotated. The sentence “position can influence where ... but never what it moves” is only a single-layer direct-path statement. Earlier layers can encode position into the residual stream that later value projections read, and the selected source position changes the content moved.

The classification of every one-layer path as a skip-trigram is a useful restricted circuit vocabulary, but “they are all a one-layer model has” is too exhaustive. Softmax normalization couples all candidate keys, scores can depend on content and relative position, and multiple heads plus the residual term yield richer context-dependent functions. The impossibility argument for the canonical induction circuit should retain its assumptions: context-free keys at the first layer and no positional shortcut.

The claim that the first copy is unpredictable “in principle” should acknowledge accidental repeats within a random pattern. Marginally the next fresh symbol is uniform, but the observed prefix may occasionally disambiguate because the sampled pattern contains repeated subsequences.

The weight-level copy test is thoughtfully labeled partial. Preserve that caution in the summary. The PyTorch and JAX committed outputs differ materially—0.98 versus 0.72 of rows have diagonal argmax—yet both support a strong copying tendency. Do not summarize this as “all tokens” without the existing “in some runs” qualification.

The text calls the model ablation a “causal handle,” but the ablation is an exercise, not a displayed performed intervention. Input-period changes are distributional probes, not causal model ablations. Either report a compact ablation result or say that the notebook establishes behavioral and weight-level consistency while proposing causal ablation as the next step.

The bridge from the Olsson et al. induction-head results to a chatbot reusing names and formats is plausible, but “recognizably the one operating” is stronger than the evidence. Induction heads are one contributor to in-context learning; modern model behavior is not reducible to that circuit, and causal attribution remains contested. The surrounding paragraph already admits this and should carry that caution into the final sentence.

## Code quality and maintainability

1. **Remove deprecated Flax calls.** Besides `nnx.view`, `what-attention-computes.md:136--143,641--643` uses deprecated `.value` access; committed JAX output already prints deprecation warnings. Use `variable[...]` consistently.
2. **Make device requirements explicit.** The cost notebook should not fail on CPU during import/execution. Capability-gate CUDA allocator/profiler and cuDNN/Flash kernels.
3. **Assert shape contracts.** Check even RoPE dimension, hidden/head divisibility, `n % block_size`, and key/value dimensions at public function boundaries.
4. **Do not encode unsafe mask semantics.** Fully masked rows must not average invalid values.
5. **Use stable accumulation dtypes.** Online softmax, long recurrent sums, and half-precision kernels should accumulate in fp32.
6. **Avoid hidden mutable analysis state where feasible.** Returning optional attention weights is clearer than relying on a side effect populated by the most recent forward call.
7. **Keep benchmark provenance adjacent to results.** Framework versions exist in output JSON, but readers need device/dtype/backend near timing plots.

## Figures and visual design

The best diagrams are the RoPE rotation, online softmax, residual stream, and induction circuit: each depicts a specific invariant or dataflow that prose alone would be slower to convey. The CNN/RNN/attention figure is useful historically but its associated table must compare consistent costs.

Recommended figure changes:

- Replace the old QKV diagram with a worked soft lookup containing one query, three keys, three scores, normalized weights, and the resulting value mixture.
- Add a small “mask composition” diagram showing padding, causal, and arbitrary structural masks combined before softmax, including the fully masked-row outcome.
- Add a two-block sliding-window diagram next to the `O(nw)` implementation; the current code makes its indexing harder to verify than necessary.
- Annotate benchmark plots with hardware, dtype, and backend.
- For long-context perplexity, show the training length as a vertical/reference marker and distinguish absolute perplexity from degradation relative to each model's value at 128. The current plot can make a worse in-range model look robust merely because it remains uniformly mediocre.
- Improve heatmap labeling in the induction notebook: show block/head IDs, the expected offset/target as overlays, and use a consistent query-row/key-column convention in every caption.

## Writing and terminology

The writing is energetic and often memorable, but confidence-signaling language sometimes substitutes for scope. Words such as “exact,” “entire,” “every,” “never,” “free,” “provably,” and “essentially every” should be reserved for statements whose assumptions are in the same sentence. The recurring word “honest” should also be removed; it appears in this chapter's scale/interpretability prose and does not make a claim more trustworthy. State the limitation directly.

Other edits:

- Use “sequence length,” “head dimension,” and “model width” consistently; several equations call all of them `d` in nearby paragraphs.
- Distinguish exact real-arithmetic equality from floating-point agreement.
- Distinguish graph reach, statistical capacity, parameter count, FLOPs, memory capacity, memory traffic, and wall-clock time. The chapter occasionally slides between them.
- Use “attention mixing” for `softmax(QK^T)V` and “multi-head attention layer” for the full projected operation. This resolves several cost inconsistencies.
- Replace historical verdicts such as “survived,” “dead,” and “swallowed the architecture” where they obscure a still-active design space.

## Completeness: additions worth making

The chapter is already broad. The additions below close conceptual gaps rather than open new surveys.

1. **A short mask-semantics subsection** in “Attention Scoring and Masking”: arbitrary boolean/additive masks, broadcasting, composition, fully masked rows, and packed sequences. This is the most important missing practical topic.
2. **A one-paragraph attention-dropout explanation** next to `DotProductAttention`: dropout is applied to normalized weights during training, surviving weights are rescaled, row sums need not remain one in a realized training pass, and evaluation disables it. Dropout itself is covered earlier, but its placement changes the attention interpretation.
3. **A compact shape ledger** for MHA: `Q/K/V`, heads, score tensor, concatenation, and output projection, with separate `d_k` and `d_v`. This would eliminate several silent equal-dimension assumptions in later code.
4. **A “claims by regime” box** in the scale section: training/prefill/decode; full/fused/windowed/linear; arithmetic/memory/traffic. The material is present, but the boundaries are not consolidated.
5. **A small limitations box for circuit analysis**: bias-free attention-only assumptions, superposition, nonlinear FFNs/norms, and why an attention map is neither a feature attribution nor a causal explanation. The final section already contains almost all of this; collect it into a reusable checklist.

I would not add a broad taxonomy of every efficient-attention paper here. The current handoff to the state-space chapter and later systems material is appropriate. Nor would I duplicate transformer blocks, cross-attention applications, BERT, ViT, or optimizer training recipes; the adjacent chapters already cover them.

## Suggested revision order

1. Fix masked softmax and all stale NNX calls; add device guards and shape assertions.
2. Correct the Gaussian-to-dot derivation, cost table, crossover arithmetic, window path sentence, and tied-head bigram claim.
3. Narrow the theorem/experiment conclusions, especially single-head insufficiency, positional extrapolation, FlashAttention universality, and induction-head attribution.
4. Replace or augment the QKV and mask figures and label benchmark provenance.
5. Add the compact mask, shape-ledger, attention-dropout, and circuit-limit boxes.
6. Perform a final absolute-word pass across source and slides so that slide summaries do not reintroduce claims already qualified in the chapter text.
