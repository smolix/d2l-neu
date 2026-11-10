# Attention Scoring and Masking

Source: `chapter_attention/attention-scoring.md`, about 3,000 words of body prose, 16 cells, 8 exercises (about 1,400 words).

## Verdict

A competent section with two passages of real quality (the dot product derived from the Gaussian kernel, and the numerics of the masked softmax) and a structural problem: the computation the reader must be able to write from memory, $\mathrm{softmax}(\mathbf{Q}\mathbf{K}^\top/\sqrt{d})\,\mathbf{V}$, is shown with a shapes figure and then "tested" on random tensors whose only checked property is their shape. Not one cell in this section runs attention on an input where the reader can say what the right answer is. Masking, a secondary topic, gets five cells and a figure. The section then ends with a retelling of the Bahdanau paper illustrated by a schematic, hand-made "attention map" and a second ending ("We can now ..."). The most important fix: give :eqref:`eq_softmax_QK_V` a small worked case with numbers and a meaningful result (the letter-matching demo that now sits at the end of `multihead-attention.md` is exactly that and uses only this section's class), and replace the history-by-narration with a trained toy alignment, which Exercise 8(a) already specifies.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought and focus | 3 |
| Pedagogy | 3 |
| Language | 4 |
| Ease and enjoyment | 3 |

## What works

- **The dot product from the Gaussian kernel** (:eqref:`eq_gaussian-expansion`). Expanding $-\tfrac12\|\mathbf{q}-\mathbf{k}\|^2$, cancelling the query term in the softmax, and saying that dropping the key norm "is a modeling decision rather than an approximation" is a short derivation that connects the two sections and is not in The Illustrated Transformer.
- **The masked softmax numerics** (under "The Masked Softmax Operation"). $-10^6$ fails in float16, $-\infty$ gives NaN on a fully masked row, the most negative finite value gives a uniform row that must be zeroed. Each choice is rejected for a stated reason. This is the detail that matters in practice and most texts omit. Exercises 3 and 4(c) extend it well.
- **Masks compose by AND under broadcasting**, with `fig_mask-algebra` (rendered: clear, three panels, shapes labelled) and the remark that harmless masks can have an empty intersection.
- **Costs with numbers**: 523,776 of 1,048,576 pairs excluded by the causal mask; 4 MiB and 0.27 billion FLOPs for one GPT-2 head at 1,024 tokens. All three verified.
- "A key that loses at the start has little chance to catch up." and "Nothing in the training objective asked for alignments. They emerged because they help to translate." Keep both.
- `DotProductAttention.forward` is three lines. A reader can memorize it.

## Structural findings

1. **No worked example of the central formula.** The reader gets `fig_attention-shapes` (good for shapes) and a cell that draws random queries, keys and values and checks that the output is `(2, 1, 4)`. Three cells in this section (the batched product, the class, the additive score) end in a `d2l.check_shape` call where a demonstration should be. Against the stated bar ("every step has a picture and a small concrete case") this is the main gap. Two repairs, both cheap: (a) a three-token example with $d = 2$ written out in the text, scores, weights and output as numbers; (b) move the "attention"/"translation" letter demo from `multihead-attention.md` (label `subsec_cross-attention-alignment`) here as the test of the class. It uses `d2l.DotProductAttention` alone, its scores are predictable ($\sqrt{32} \approx 5.7$ for a match), and its heatmap can be read.
2. **Order.** The alternative score (additive attention) and the origin of the idea arrive after the class, 600 lines after the question "which score should the network learn" was posed. The comparison belongs in "Dot-Product Attention", as the foil: two paragraphs, the formula, and the reason it lost (an $n \times m \times r$ array against one matrix product). The history then needs no section of its own.
3. **"Attention as Learned Alignment" narrates a paper instead of doing the experiment.** It tells what Bahdanau et al. built and found, shows a figure whose caption admits it is "schematic", and runs additive scoring on random matrices. Exercise 8 ("A learned alignment") contains the experiment that would replace all of it: one learned query per output position, keys from token and position, values the tokens, a fixed permutation as target. Part (a) trains in seconds and yields a real attention map the reader can compare with $\pi$. Promote part (a) to the body, keep (b) to (d) as the exercise, and drop the schematic figure or keep it only as an illustration of what the 2014 result looked like, said plainly.
4. **Proportion.** Masking takes about 270 source lines, the scored and batched computation about 190. Two of the masking cells duplicate what the next cell or the figure shows (SCORE-9).
5. **Two endings.** The paragraph beginning "We can now compute attention for a whole minibatch" restates the section before the Summary restates it again (SCORE-17).
6. **Where queries and keys come from is deferred.** "Queries and keys of a network are computed from its inputs by learned linear maps" points forward to the next section, which introduces the maps only per head. Single-head attention with $\mathbf{W}_q$, $\mathbf{W}_k$, $\mathbf{W}_v$ is never written down anywhere in the chapter. It should be written here, in two lines, after :eqref:`eq_softmax_QK_V` (see MHA-2 in the multi-head review).

The section does not need to be split. With finding 2 and 3 applied it becomes: score (with the additive foil), scale, the matrix form and class with a worked example, masks, a trained alignment.

## Passage-level findings

**SCORE-1** (moderate, explanation). Under "Dot-Product Attention".
Anchor: `modeling decision rather than an approximation: the dot product becomes`
The reader is told the key-norm term is dropped by decision but not what the decision changes. It changes something concrete: with a dot product a long key receives more weight from every query that points its way, so key norm acts as a prior over keys, while the Gaussian score penalizes long keys. One sentence, with a pointer to Exercise 1(c) of :numref:`sec_queries-keys-values`, which shows a query for two keys collapsing onto the longer one.

**SCORE-2** (moderate, correctness; fairly sure). Under "Score Variance and the $1/\sqrt{d}$ Factor".
Anchor: `the calculation above describes a network at initialization. That is when`
The calculation describes queries and keys with unit-variance coordinates. Whether that holds at initialization depends on the initializer. The layers of this chapter use `nn.Linear`'s default (and the JAX `Linear` class copies it): uniform on $\pm 1/\sqrt{n_\textrm{in}}$, which gives each projected coordinate variance $1/3$ for unit-variance inputs, hence score variance $d/9$ unscaled and $1/9$ scaled. I simulated this case with NumPy: at $d = 64$ with 64 keys the unscaled entropy is about 2.0 nats (not 0.6), and the scaled entropy is 4.1, within 0.06 of uniform. The multi-head section's own Exercise 5(d) says that every head of a fresh layer weights the two positions "close to (0.5, 0.5)". So the later claim "A model with unscaled scores would therefore start training with nearly one-hot attention" is true for the experiment's standard normal vectors and for LeCun-normal projections of normalized inputs, and overstated for the book's own layers at typical head widths. Fix: say that the experiment models unit-variance coordinates, state the robust conclusion (the factor makes the sharpness of the weights independent of the head width, so a model does not become sharper merely by being widened, and the same learning rate serves different widths), and drop "That is when it matters".

**SCORE-3** (minor, language). Same subsection.
Anchor: `chance to catch up. Only this route saturates. Gradients still reach the`
"This route" has no antecedent a reader can find. Write "Only the path through the scores saturates."

**SCORE-4** (moderate, thought). Same subsection.
Anchor: `scores the Jacobian is never exactly zero, since the all-ones vector alone`
Third statement of this fact in two sections (see QKV-2), and it qualifies a point the paragraph has already made. Keep it here once, cut it from `queries-keys-values.md`, and let Exercise 2(a) prove it.

**SCORE-5** (moderate, correctness of prose against output). Same subsection.
Anchor: `norm first *rises*: at small $d$ the unscaled weights concentrate on a few`
The cell prints three values of the unscaled norm, 0.253, 0.219, 0.085 (PyTorch; JAX 0.260, 0.230, 0.083). They fall monotonically. The rise is real (my NumPy rerun gives 0.26, 0.30, 0.30, 0.26, 0.22 at $d = 4, 8, 16, 32, 64$) but no cell shows it, and a reader comparing prose and output sees a contradiction. The same paragraph says the curves cross "near $d = 128$", and the paragraph before quotes the largest weight as "about 80%" at $d = 64$ and "about 90%" at $d = 512$. Both numbers are right (I get 0.78 and 0.92) and neither is printed. Fix: print the unscaled and scaled norm and the mean largest weight for `d in (4, 16, 64, 128, 512)`, or plot the norm against $d$ beside the entropy, and write "the unscaled norm is at first larger than the scaled one".

**SCORE-6** (minor, pedagogy). Same subsection.
Anchor: `With scaling, the entropy stays near $3.7$ nats over two orders of`
The reader will ask why 3.7 and not $\log 64 = 4.16$. Exercise 6 derives $H \approx \log m - s^2/2$, which gives 3.66 at once. Quote the result in a clause here ("half a nat below uniform, as Exercise 6 derives"); it is the kind of small exact fact that makes an experiment feel understood.

**SCORE-7** (minor, correctness; currency). Under "The Masked Softmax Operation".
Anchor: `constant such as $-10^6$. That works in single precision. Float16, a format`
Accurate, but in 2026 the usual reduced-precision training format is bfloat16, which has the exponent range of float32, so $-10^6$ is representable there. The argument for `finfo(dtype).min` still stands (it is correct for every dtype without the author having to know which one is in use). Say so in a clause, or a reader who trains in bf16 will think the paragraph does not apply.

**SCORE-8** (minor, explanation). Same subsection.
Anchor: `any precision, and no NaN can arise.`
The claim relies on the framework's softmax subtracting the row maximum. State that this is what `F.softmax` and `jax.nn.softmax` do (a clause), since the argument is about a property of the implementation.

**SCORE-9** (moderate, pedagogy). Under "The Masked Softmax Operation" and "Causal Masking".
Anchors: `With one valid length per query, each row is cut at its own length:` and `On six random scores, the masked weights form a lower triangle:`
Two cells that add nothing. The per-query-lengths demonstration serves a use the section never names (when does each query have its own valid length?). The random causal heatmap is shown again as the left panel of the composite cell thirty lines later. Cut both, or keep the per-query case only if a sentence says where it arises.

**SCORE-10** (minor, pedagogy). Under "Causal Masking".
Anchor: `arises when new tokens attend to the keys of earlier tokens kept in a`
The rectangular causal mask is explained through a cache the reader meets a chapter later. The forward reference is not load-bearing, but four lines of the first explanation of causality go to it. Move the rectangular case to a code comment and one sentence.

**SCORE-11** (moderate, explanation). Under "Batch Matrix Multiplication".
Anchor: `The *batch matrix multiplication* (BMM) computes one matrix product per`
A reader who finished Basics has used `@` on batches. The subsection spends a display equation, a cell on tensors of ones, and two tab notes on it, and its notation collides with the next subsection: here $n$ is the batch size and $\mathbf{Q}$, $\mathbf{K}$ are arbitrary matrices of shapes $a \times b$ and $b \times c$; forty lines later $n$ is the number of queries and $\mathbf{Q} \in \mathbb{R}^{n \times d}$. Fix: delete the subsection and say inside "The DotProductAttention Class" that `@` multiplies the last two axes and carries all leading axes (batch, later heads) along.

**SCORE-12** (major, pedagogy). Under "The DotProductAttention Class".
Anchor: `attention = DotProductAttention(dropout=0.5)`
The class that "every later model in this book builds on" is tested only for the shape of its output on random inputs, with `dropout=0.5` set and then switched off by `eval()`. See structural finding 1. Replace with an input whose answer is known. Draft lead-in for the letter demo: "Give each letter a random embedding of width 32 and let the letters of one word query those of another. Equal letters share an embedding, so their scaled score is close to $\sqrt{32} \approx 5.7$, while different letters score near zero. We know which keys each query should find before we run the layer."

**SCORE-13** (moderate, language; tab prose). Same subsection.
Anchor: `The weights live in an `nnx.Intermediate` variable rather than in a plain`
Ten lines of Flax state management (views, retracing, memory of compiled steps) in a tab note, against the guide's "at most a sentence or two" (section 9). The PyTorch reader does not see it and the JAX reader meets framework plumbing in the middle of the definition of attention. Fix: one sentence ("In JAX the weights are recorded only in eager calls; call the module outside `nnx.jit` to inspect them") and put the rest in the code comments, where half of it already is.

**SCORE-14** (minor, correctness; currency). Same subsection.
Anchor: `attention weights as a regularizer and keeps the weights of its last call`
Attention dropout is mostly set to zero in current large language models. Say in a clause that the default of 0 is the common setting today, so the reader does not copy `dropout=0.5` from the test cell.

**SCORE-15** (moderate, dependency). Under "Attention as Learned Alignment".
Anchor: `:numref:`sec_seq2seq`. This is the bottleneck of`
"We build such an encoder--decoder in :numref:`sec_seq2seq`" points to `chapter_natural-language-processing-pretraining/seq2seq.md`, a chapter now in the Attic part, twenty-odd chapters after this one. The Basics reader has not built an encoder-decoder. Either describe it in the one sentence already present and drop "we build", or say "(retained in the Attic, :numref:`sec_seq2seq`)".

**SCORE-16** (moderate, correctness). Same subsection.
Anchors: `decoder reads. At every output step, the current decoder state served as a` and `the encoder states had different widths, so instead of a dot product,`
Two statements about the paper. (a) In Bahdanau et al. the alignment score for output step $i$ is computed from the previous decoder state $s_{i-1}$, since the context vector is an input to computing $s_i$. "Current" should be "previous" (certain). (b) "had different widths, so instead of a dot product" attributes a motive. The widths did differ (a bidirectional encoder gives annotations twice as wide as the decoder state), but I do not believe the paper considers or rejects a dot product; it parametrizes the alignment model as a feed-forward network. Unverified as a claim about the authors' reasoning. Write "and they scored with a small network with one hidden layer, which also accommodates the different widths of decoder and encoder states".

**SCORE-17** (moderate, pedagogy). Same subsection.
Anchor: `Each row is a distribution over the six keys. The intermediate array`
The cell's output is not read, and it holds the one interesting fact about additive attention. In both stored outputs the three rows are nearly the same (PyTorch: 0.31, 0.11, 0.10, 0.22, 0.16, 0.11 against 0.30, 0.09, 0.11, 0.25, 0.16, 0.08): the weights barely depend on the query. Exercise 7 explains why (without the $\tanh$ the query term is constant across keys and cancels in the softmax, so all query dependence comes from the nonlinearity). Either read the output and say this in two sentences, which also gives a second reason the dot product won, or cut the cell and keep the formula.

**SCORE-18** (minor, pedagogy). Same subsection.
Anchor: `in the style of a learned attention map (darker cells indicate larger weight; schematic)`
A drawn map standing in for a measured one. Readers of a book whose rule is that claims are tested will take it for a result unless they read the parenthesis. See structural finding 3.

**SCORE-19** (moderate, language; author decision 6). End of the same subsection.
Anchor: `We can now compute attention for a whole minibatch, with scores that`
The "We can now ..." recap before the Summary. Delete the first sentence. Keep the two limits it names (one weighted average per query, no knowledge of position) as the closing sentence of the body, since they are the open questions the next two sections answer.

**SCORE-20** (minor, language). Opening.
Anchor: `This section answers both questions. The score is the dot product of query and`
The third opening paragraph gives away every result of the section in six sentences, in the order of the headings. A roadmap that is a table of contents is skipped. Two sentences suffice: what the score will be and that exclusions are masks.

Recurring small pattern: variation of terms for one object. "Width" and "dimension" for $d$ (the plot axis says "dimension d", the prose "width"), "score", "scoring function" and "compatibility function", "valid length" and "length". Pick one each.

## Code and experiments

- **Cells that show only a shape** (three cells, in both tabs): `attention-scoring-batch-matrix-multiplication`, `attention-scoring-the-dotproductattention-class-2`, `attention-scoring-from-alignment-to-attention`. See SCORE-11, SCORE-12, SCORE-17.
- **Numbers no cell prints**: the unscaled largest weight (80%, 90%), the crossing "near $d = 128$", the rise of the unscaled Jacobian norm (SCORE-5). All correct by my rerun.
- **Output against prose**: `attention-scoring-softmax-saturation-and-the-1-sqrt-d-factor-2` shows a monotone fall where the prose says "first rises" (SCORE-5). "With scaling, the norm is the same at every dimension" holds to two digits (0.177, 0.183, 0.185). "Falls to about half of it at $d = 512$" holds (0.085 against 0.185). Entropy 3.7 and 0.2 nats confirmed.
- **Experiment read for more than it shows**: the saturation experiment is on standard normal vectors, read as a statement about networks at initialization (SCORE-2).
- **Redundant cells**: `attention-scoring-the-masked-softmax-operation-3`, `attention-scoring-causal-masking-1` (SCORE-9).
- `attention-scoring-composing-masks-1` draws a boolean mask as a heatmap. It is a computed object, so it passes the house rule, but `fig_mask-algebra` could carry a fourth panel instead.
- `masked_softmax`, `padding_mask`, `causal_mask` are compact and correct. I traced the fully masked row and the rectangular causal case by hand.
- The two tabs fragment the prose once, at the JAX note on `nnx.Intermediate` (SCORE-13). JAX outputs for the printed cells agree with the prose.

## Exercises

Eight problems, the strongest set of the three sections. Exercise 1 shows that the key bias has zero gradient (the reason `MultiHeadAttention` defaults to `bias=False`, which the next section never explains). Exercise 3(b) (a NaN at a discarded padding position poisons valid positions in the next layer) and 4(c) (adding two `finfo.min` biases overflows to $-\infty$) are sharp and practical. Exercise 6 derives the 3.7 nats; I checked $\log 64 - 1/2 = 3.66$. Exercise 7 is the insight missing from the body (SCORE-17).

- **Promote** Exercise 8(a) to the body (structural finding 3).
- **Trim** Exercise 2: parts (b) and (c) both end in "explain the printed norms"; keep (a) and (c).
- **Add** one easy problem: compute scores, weights and output by hand for two queries and three keys in $\mathbb{R}^2$, then apply a causal mask and recompute. No problem here is a warm-up.
- Exercise 5 (attention dropout) is sound but concerns a feature that is mostly switched off today; it is the first candidate if the set must shrink.

## Still open from October

The October review called the chapter problem-led with experiments that are read out; that holds here for the saturation experiment and not for the class tests. The exercise volume it flagged for the chapter is moderate in this section (about 45% of the body).

## For the chapter review

- Assumes `sec_attention-pooling` (Gaussian kernel, heatmaps) and softmax from Basics. No appendix dependence. One reference into the Attic (`sec_seq2seq`) and two forward references to `sec_kv-cache` and `sec_attention-at-scale`, none load-bearing.
- This section should own: the softmax Jacobian and saturation (cut from the first section), single-head attention with learned projections (missing from the chapter), and the first meaningful run of `DotProductAttention` (move the letter demo here from the multi-head section).
- Additive attention and the Bahdanau history should shrink to a foil inside the scoring discussion plus a trained toy alignment.
- Suspected chapter-wide pattern: classes are validated by `d2l.check_shape` on random or constant tensors instead of on a case with a known answer; the chapter review should count these across all six sections.
- The "We can now ..." second ending occurs here; check the other sections.
