# Writing pass: `chapter_linear-regression/linear-regression.md` (section 2.1)

Prose only. No code cell, label, figure include, exercise or slide was
changed. Checker: "OK: only prose changed"; `tools/lint_source.py` clean.

## Words

Prose words (checker count): 5756 before, 5820 after (+64, about 1%).
The section did not get shorter. Repetition and cuttable prose removed about
250 words; three additions asked for by the review or the assignment put
about 310 back: the scalar noise-ball calculation (LR-7 in the review), the
numeric instance of likelihood (LR-10), and the single full statement of the
factor $\frac{1}{2}$ at the definition of the loss. If the length matters
more than LR-10, the likelihood paragraph (45 words) is the one to drop.

## Before and after

1. Before: "The factor $\frac{1}{2}$ only makes the derivative tidier:"
   After: "The factor $\frac{1}{2}$ is there for convenience only: it cancels
   the 2 that differentiation produces, so that [...]. It is a convention of
   this book and by no means a universal one. The built-in mean squared
   error of most libraries omits it, so that loss and its gradient are
   exactly twice ours. The minimizer does not notice. Gradient methods do: a
   step is the learning rate times the gradient, so the library loss needs
   half our learning rate to take the same steps. Whenever the two losses
   meet later in the book, this factor of 2 is the whole difference between
   them."
2. Before: "The linear model therefore describes the average label: we
   assume that the conditional expectation
   $\mathbb{E}[Y \mid X = \mathbf{x}]$, the mean price of houses with
   features $\mathbf{x}$, equals [...]"
   After: "What, then, does the model claim? It describes the average label:
   we assume that the mean price of houses with features $\mathbf{x}$, which
   we write as the conditional expectation $\mathbb{E}[y \mid \mathbf{x}]$,
   equals [...]"
3. Before: "Squaring makes large errors count far more than small ones: an
   error of 10 costs a hundred times as much as an error of 1. The squared
   loss therefore penalizes large errors heavily, but it also lets a single
   bad label dominate the fit. To see how much, we place [...]"
   After: "Squaring makes large errors count far more than small ones. An
   error of 10 costs a hundred times as much as an error of 1, so a single
   bad label can dominate the fit. How badly? To find out, we place [...]"
4. Before: "A zero gradient does not in general mark a minimum, but here
   every solution of the normal equations is one, by a short argument."
   After: "A zero gradient does not in general mark a minimum. Here it does,
   and the argument is short."
5. Before: "Its size follows from a balance. Each step adds noise of variance
   proportional to $\eta^2 / |\mathcal{B}|$, while the pull toward
   $\hat{\mathbf{w}}$ removes a fraction proportional to $\eta$ of the
   squared distance."
   After: "How large is the cloud? A single weight is enough to see it. Let
   $u = w - \hat{w}$ be its distance from the minimizer, and scale the
   feature so that the gradient of the full loss is $u$. A step of SGD then
   replaces $u$ by $(1 - \eta)u - \eta\xi$, [...]"
6. Before: "Both the predictions $\mathbf{X}\mathbf{w} + b$ for a minibatch
   and the gradient in [...] are matrix--vector products, [...]. To measure
   the difference, we predict [...]"
   After: "We have claimed twice that one matrix product over many examples
   beats a loop over them. It is time to measure. Both the predictions
   [...]"
7. Before: "A probabilistic view tells us when it is the right choice: we
   model how the labels scatter [...]"
   After: "When is it the right choice? A probabilistic view gives the
   answer. We model how the labels scatter [...]"
8. Before: "We can now write down a linear model, fit it exactly or with
   minibatch SGD, and choose its loss from an assumption about the noise.
   Two questions remain."
   After: "Two things are still missing. We have not trained anything yet:
   [...] The house of 150 square meters has to wait a little longer: this
   chapter works with synthetic data, where the right answer is known, and
   real house prices arrive in :numref:`sec_kaggle_house`."

## Repetition removed

- **Factor $\frac{1}{2}$ (CON-1, WD-7).** Was touched three times in this
  file (at the loss, in the "hats" paragraph, and in full under minibatch
  SGD). Now explained once, in "The Squared Loss"
  (`subsec_linear-regression-loss-function`), including the consequence for
  the learning rate. Under minibatch SGD it is one clause with a reference
  back; the general rule (scaling the loss by $c$) and the sentence "A
  learning rate has a meaning only together with the loss it is used with"
  stay there. Later sections can cite either label: both now lead to the
  fact.
- **Legendre and Gauss (IDX-1).** Reduced to "The method is two centuries
  old" with the citation.
- **Opening roadmap / closing / summary.** Dropped the roadmap sentence about
  the rest of the chapter (the closing says it), the recap sentence of the
  closing ("We can now write down ..."), the preview sentence at the head of
  the Gaussian part ("Normal scatter yields the squared loss ..."), and the
  minibatch clause of the summary.
- "Trains every network/model in this book" said twice within 40 lines; the
  second is now "the iterative method that every other model needs".
- The gradient $2\mathbf{X}^\top\mathbf{r}$ re-derived under projection: cut.
- LR-2 (the triple statement about large errors), LR-11 ("for a fixed
  $\sigma$ ... whatever the value of $\sigma$").

## Review findings

Applied: LR-1, LR-2, LR-4 (scaling remark removed; see deferral for
$\mathbf{w}^*$), LR-5 (the extended $\mathbf{X}$, $\mathbf{w}$ are announced
for "this subsection and the next", and the SGD update is introduced "with
the bias again written separately"), LR-6 (invertibility in one
parenthesis, definition of the inverse cut), LR-7 option (a), LR-8
(parenthesis deleted), LR-10, LR-11, LR-12, LR-13, LR-14. Structural: "the
house disappears" (closing now says where the house is valued).

Partly applied in prose: the misplaced vectorization subsection. The cells
cannot move, so the claim in "The Linear Model" now says outright that the
matrix routine is faster, the SGD subsection points "below", and the
subsection opens by saying that it measures a claim made twice. LR-9: the
prose now says in one clause why `wait` is called.

Deferred:
- LR-3: needs a changed code cell (grid scan instead of gradient descent).
  The subgradient paragraph and "stays near 2" are unchanged.
- LR-9 proper (CPU timing without `wait`, or helper in `d2l`): code.
- Moving "Vectorization for Speed": moves code cells.
- LR-4, moving the $\mathbf{w}^*$ convention to the synthetic data section:
  moves material between files; that section's writer cannot see it.
- "Too much for one section" (split): structural.
- Exercise remarks (4(c), splitting 8): exercises are out of scope.

## Factual claims added or changed

- "The method is two centuries old" (was "Legendre and Gauss at the
  beginning of the 19th century").
- The library mean squared error and its gradient are "exactly twice ours",
  and it "needs half our learning rate to take the same steps" (moved, same
  content as before). "Whenever the two losses meet later in the book, this
  factor of 2 is the whole difference between them": new; true of the
  concise and weight decay sections as they stand.
- Noise ball, scalar case: with $u \leftarrow (1-\eta)u - \eta\xi$,
  $\operatorname{Var}\xi = s^2/|\mathcal{B}|$, $\xi$ treated as independent
  of $u$, the stationary variance is
  $\eta s^2/((2-\eta)|\mathcal{B}|) \approx \eta s^2/(2|\mathcal{B}|)$. The
  text gives the approximation only. Replaces three unsupported
  proportionalities; the conclusion ($\propto \eta/|\mathcal{B}|$, growing
  with label noise) is unchanged.
- "each step then also shrinks the distance less" as the reason for four
  times as many steps (contraction factor $1-\eta$).
- Likelihood instance: joint density about 0.058 for the fitted line and
  about 0.045 for the horizontal line through the mean, with $\sigma = 1$
  (computed: 0.05842 and 0.04550; the review's 0.046 was rounded up). "No
  other line does better."
- "this chapter works with synthetic data ... real house prices arrive in
  :numref:`sec_kaggle_house`".
- Summary: the squared error is the Gaussian negative log-likelihood, "which
  is why it is sensitive to outliers" (was two parallel statements).
- Removed claims: sampling with/without replacement and the reference to
  `sec_sgd`; input-dependent noise $\sigma(\mathbf{x})$; "feature
  dimensionality"; that $(2, -3.4)$ and $4.2$ return as the synthetic
  parameters (true, but the synthetic section says it).

No number describing an output was changed. "Faster by a factor of hundreds
or more" holds in all four stores (181x to 997x).

## Slides against the body

No statement in the slides section now contradicts the body. Two slides are
no longer matched by the same wording: the SGD slide gives the loss-scale
rule without the factor $\frac{1}{2}$ (fine), and the noise-ball slides
state $\propto \eta/|\mathcal{B}|$, which the body now derives for one
weight. The body no longer names Legendre and Gauss; the slides do not
either.
