# Queries, Keys, and Values

Source: `chapter_attention/queries-keys-values.md`, about 1,500 words of body prose, 5 cells, 7 exercises (about 1,100 words).

## Verdict

This is the best of the three opening sections of the chapter and its first page meets the bar: the 768-number state of a GPT-2-wide recurrent model, the two-record scalar database, and the soft query that returns 19.5 are a small concrete case of the kind The Illustrated Transformer is loved for, and the reader can redo the arithmetic by hand. After that page the section becomes competent and ordinary. The second half is a kernel-regression demonstration whose result ("the bandwidth decides the fit") is never tied to the attention the reader came for. The most important fix is to say that the bandwidth is the sharpness of the softmax, the same knob as the factor $1/\sqrt{d}$ of the next section, and to carry the scalar database through to a vector example so that the general definition also has numbers.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought and focus | 4 |
| Pedagogy | 3 |
| Language | 4 |
| Ease and enjoyment | 3 |

## What works

- The opening (lines 4 to 30). The problem is felt before the mechanism: "all written before the network could know that the 17th token would matter". The database $\{(2,10),(5,20)\}$ with $q=4$ returning nothing, then 19.5, is worked in full, and I checked it ($e^{-4}/(e^{-4}+e^{-1}) = 0.047$, output 19.53). Protect this passage in revision.
- "What grows instead is memory. A recurrent network keeps one state, while attention keeps a key and a value for every position it may need to read." The cost of the idea is named as soon as the idea is stated.
- `fig_qkv` (rendered and checked: scores 2.0, 1.0, 0.1 give 0.66, 0.24, 0.10) shows the output as a point inside the triangle of values. The convex-hull reading is the right picture.
- The caution about heatmaps: "In a trained network, such a heatmap is a hypothesis about what the layer does, not a proof. A large weight on a key whose value the next layer ignores explains nothing." This is an author judging, and it sets up the last section of the chapter.
- The closing limit of the fixed kernel: "A query cannot ask for the key three positions back, or for the key with the same grammatical role, because neither is a matter of distance in the input space." It ends the section on the question the next one answers, with no recap.
- `nadaraya_watson` is four lines and returns the weights, so one function serves the fit and the heatmap.

## Structural findings

1. **The experiment is about regression, not about attention.** Half of the body (the kernel list, the estimator, the bandwidth comparison) teaches Nadaraya-Watson regression on noisy one-dimensional data. A reader who "wants to understand and build modern systems" learns that small $\sigma$ chases noise and large $\sigma$ flattens, which is a statistics lesson. The attention lesson in it is that one scalar controls how sharply a query selects, and that this scalar divides the score before the exponential. The section never says so, and the next section then introduces $1/\sqrt{d}$ as if sharpness were a new topic. One sentence here and one back-reference there would make the regression earn its space.
2. **The general definition has no numbers.** The scalar example is excellent, but the vector form (:eqref:`eq_attention_pooling`) is illustrated only by `fig_qkv`, where the scores are given rather than computed from a query and keys. Extend the opening database to two-dimensional keys in one short paragraph, or let the figure show where $s_i$ comes from.
3. **A digression sits between the definition and its picture.** The paragraph on the softmax Jacobian and hard attention (lines 89 to 96) separates :eqref:`eq_softmax_attention` from `fig_qkv`. The Jacobian fact is taught again, properly, in the next section.
4. **Exercise weight.** About 1,100 words of exercises against 1,500 of prose, and no easy problem. See Exercises.

The section is one section and has one idea. Do not split it.

## Passage-level findings

**QKV-1** (minor, correctness). Under the section opening.
Anchor: `:cite:`Kalchbrenner.Grefenstette.Blunsom.2014` face the same limit, and on`
"Convolutions over sequences face the same limit" is not explained and not obviously true: a convolution keeps one vector per position, and its limit is the receptive field, not a fixed-size state. The cited model pools to a fixed vector, but the reader cannot know that. Fix: delete the sentence, or say "models that pool a sentence into one vector".

**QKV-2** (moderate, thought). Under "Attention as Differentiable Lookup".
Anchor: `the zero matrix, although its entries become very small when one score`
The statement that the softmax Jacobian is never zero for finite scores appears here, again in `attention-scoring.md` ("the Jacobian is never exactly zero, since the all-ones vector alone spans its null space"), and in exercises of both sections. Here it interrupts the definition before the figure, and the reader has no use for it yet. Fix: keep "The softmax is differentiable, so gradients reach the query, the keys, and the values", move the hard-attention remark to one clause, and leave saturation to the next section, which owns it.

**QKV-3** (moderate, explanation). Under "Similarity Kernels".
Anchor: `A *kernel* $\alpha(\mathbf{q}, \mathbf{k})$ measures the similarity of a`
Thirty lines earlier $\alpha(\mathbf{q},\mathbf{k}_i)$ was defined as a weight that is nonnegative and sums to one over the keys. Here the same symbol is an unnormalized kernel, and :eqref:`eq_nadaraya-watson` then divides $\alpha$ by $\sum_j \alpha$. A careful reader stops. Fix: write the kernel as $K(\mathbf{q},\mathbf{k})$ and keep $\alpha$ for normalized weights throughout the chapter.

**QKV-4** (moderate, thought). Under "Similarity Kernels".
Anchor: `*Parzen windows* :cite:`parzen1957consistent`.`
Four kernels are listed at equal weight with a four-panel figure, and only the Gaussian is used afterwards. The boxcar is defined only where it equals 1. The citation is to Parzen's 1957 paper on spectral estimates of time series (checked in `d2l.bib`); the density-estimation paper is Parzen (1962), "On estimation of a probability density function and mode". Fix: define the Gaussian kernel with bandwidth $\sigma$ in the display, mention in one sentence that boxcar and triangular windows exist, drop `fig_attention_kernels` or reduce it to the Gaussian at three bandwidths (which would prepare the experiment), and correct or drop the Parzen citation. I am certain of the citation mismatch.

**QKV-5** (minor, correctness). Under "Similarity Kernels".
Anchor: `the best possible predictor :cite:`mack1982weak`.`
"The best possible predictor" is the conditional mean $\mathbb{E}[y \mid x]$, best for squared error, and the convergence needs smoothness and density conditions. Say "to the conditional mean $\mathbb{E}[y \mid x]$, the best predictor under squared error".

**QKV-6** (moderate, pedagogy). Under "Nadaraya--Watson Regression".
Anchor: `The bandwidth decides the fit. With $\sigma = 0.1$ the estimate chases the`
The paragraph reads the plot correctly (I rendered the stored PyTorch figure: $\sigma = 0.1$ follows single points, $\sigma = 2$ is nearly flat, $\sigma = 0.5$ is off at both ends). What is missing is the sentence that turns the result into a statement about attention. Draft: "In the language of :eqref:`eq_softmax_attention`, the score is $-(q-k)^2/2\sigma^2$, so $1/\sigma^2$ multiplies the scores before the softmax. A large multiplier makes the weights nearly one-hot, a small one makes them nearly uniform. Every attention layer has this knob, and :numref:`sec_attention-scoring-functions` shows how the Transformer sets it."

**QKV-7** (minor, language). Under "Attention Weights as a Diagnostic".
Anchor: `the chapter with `d2l.show_heatmaps`. The helper takes an array of shape`
The shape convention of a plotting helper is API detail in the middle of an argument about what weights mean. Move it to a code comment.

**QKV-8** (minor, language). Heading and Summary.
Anchor: `## Attention Pooling with Fixed Kernels`
"Attention pooling" is used in the heading, in the label `sec_attention-pooling`, and in the Summary, but the body calls :eqref:`eq_attention_pooling` "the attention output" and never introduces the term. Either define it at the equation ("this weighted sum is also called *attention pooling*") or use "attention" throughout.

**QKV-9** (minor, pedagogy). Under "Attention as Differentiable Lookup".
Anchor: `uses the differentiable form throughout. :numref:`fig_qkv` works one lookup`
The figure "works one lookup through" but its scores 2.0, 1.0, 0.1 come from nowhere. Tie it to the opening example (use the score $-(q-k)^2$ on three scalar keys, or two-dimensional keys with a dot product), so that the picture and the arithmetic are the same case.

## Code and experiments

- `queries-keys-values-queries-keys-and-values` (imports): `math`, `numpy`, `nn`, `F` (PyTorch) and `nnx`, `optax` (JAX) are not used in this section.
- `queries-keys-values-nadaraya-watson-regression-1`: a data plot of a computed fit, acceptable under the house rule. The noise has standard deviation 1 on a function that ranges over about 4 units with 40 points, so a different draw moves the $\sigma = 0.5$ curve visibly. The prose makes no claim finer than the plot supports.
- `queries-keys-values-nadaraya-watson-regression-in-action-3`: the prose matches the stored heatmaps (largest weight near 0.9 for $\sigma = 0.1$, about 0.05 for $\sigma = 2$, "a few hundredths").
- No cell only confirms a derivation. No numbers are quoted that a cell does not show.
- I did not render the JAX outputs.

## Exercises

Seven problems, all substantial, three of them marked hard, five needing code. They are good problems. Exercise 3(c) (the NaN at $q = 15$ in single precision, cured by taking the softmax of the scores) is the kind of problem that teaches something the text did not. Exercise 7(d) (what a shared map guarantees and separate $\mathbf{W}_q$, $\mathbf{W}_k$ lose) is the best preparation for the next section and deserves a sentence in the body of that section.

- **Add** one easy opening problem: redo the scalar database by hand for $q = 3$ and for a third record, and say what happens to the answer as the score is multiplied by 10. Nothing at present is solvable in five minutes.
- **Cut or move** Exercise 4 ("A learned bandwidth"). The October review noted that it repeats what `chapter_recurrent-modern/test-time-regression.md` does. If it stays, keep only parts (a) and (c).
- **Trim** Exercise 6 ("Soft lookup as retrieval") to parts (a), (b) and (d). Part (c) is a measurement whose explanation (Jensen on the random inner products) is a second problem.
- Exercise 5 is a classic and should stay.

## Still open from October

The exercise volume (then 132% of the body, now roughly 70%) is reduced but still heavy for a 1,500-word section. Exercise 4 still overlaps chapter 12. The October "keep" (the scalar database) is intact.

## For the chapter review

- This section is the right first section and the right length. It assumes only :numref:`sec_rnn` and :numref:`sec_softmax`, and nothing from the appendix.
- It sets up the next section through the Gaussian kernel (expanded there into the dot product) and through the heatmap helper. It should also set up the $1/\sqrt{d}$ factor through the bandwidth (QKV-6).
- The chapter introduction and this opening both state the fixed-state problem (the mystery novel there, the 768 numbers here). The two do not repeat each other's example, which is acceptable; keep it that way.
- Suspected chapter-wide pattern: clean, plain, short-sentence prose with little voice after the opening page, and exercise sets that are strong but heavy and without easy problems.
- The softmax Jacobian fact has three homes in the first two sections; the chapter review should assign it to `attention-scoring.md`.
