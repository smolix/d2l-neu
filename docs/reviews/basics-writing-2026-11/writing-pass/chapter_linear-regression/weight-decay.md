# Writing pass: Weight Decay (section 2.7)

Source: `chapter_linear-regression/weight-decay.md`. Prose only; no cell,
label, figure, exercise, or slide was changed.

**Prose words:** 5111 before, 5091 after (checker: "OK: only prose changed";
`lint_source.py` clean).

## Before and after

1. Before: "A small example shows where the excess flexibility of a linear
   model lies."
   After: "Where does the excess flexibility of a linear model lie? A small
   example shows it."
2. Before: "Least squares chooses among such weight vectors by how well they
   fit the noise in the training labels, and since the small differences
   $x_1 - x_2$ are noise as well, it can return large weights of opposite
   sign."
   After: "The data say almost nothing about how to split the weight between
   the two features. Least squares settles the split by fitting the noise in
   the training labels, and since $x_1 - x_2$ is tiny, moving the predictions
   even a little through it takes large weights of opposite sign."
3. Before: "We define the penalty and its effect on gradient descent, watch it
   repair a model with 200 features trained on 20 examples, and then ask which
   directions in weight space it shrinks."
   After: "We define the penalty and its effect on gradient descent, try it on
   a model with 200 features and only 20 training examples, and then ask which
   directions in weight space it shrinks. That question will bring us back to
   the duplicated features, where the penalty does its proper work."
4. Before: "The factor $\frac{1}{2}$ makes the gradient of the penalty simply
   $\lambda\mathbf{w}$. We do not penalize the bias: adding a constant ..."
   After: "The factor $\frac{1}{2}$ is merely convenient: it makes the gradient
   of the penalty $\lambda\mathbf{w}$. Leaving out the bias, by contrast, is
   deliberate. Adding a constant ..."
5. Before: "Now we add the penalty, with $\lambda = 3$."
   After: "Now we add the penalty. Which $\lambda$? The prior has an opinion:
   with $\sigma = 0.01$, $n = 20$, and weights of size $\tau = 0.01$, it asks
   for $\lambda = \sigma^2/(n\tau^2) = 0.05$. We use $\lambda = 3$, sixty times
   as much, and owe the reader the reason. The formula describes the minimizer
   of the penalized loss, not the point that 120 steps of SGD reach. At
   $\lambda = 0.05$ the whole run would keep $0.9995^{120} \approx 94\%$ of the
   weights, and the penalty would leave almost no trace ..."
6. Before: "The second effect is what helped in our experiment; a small example
   below shows the first."
   After: "The first effect is the one that makes weight decay worth having,
   and a small example below shows it. The second, as it turns out, is all
   that helped in our experiment."
7. Before: "Ridge keeps its error at about 0.01."
   After: "Ridge keeps its error at about 0.01. This is the situation the
   penalty is made for: a direction that the data barely determine, and noise
   ready to fill it." (The next subsection then opens "Was the experiment with
   200 features such a situation?")
8. Before: "The penalty cannot recover what the data do not determine. ... The
   regularized model comes close to this limit. Started from
   $\mathbf{w} = \mathbf{0}$, the unpenalized model does about as well ..."
   After: "We should be candid about how little was gained. The penalty cannot
   recover what the data do not determine. ... The regularized model, at about
   0.01, comes close to this limit. So, however, does a model that ignores its
   inputs and predicts $b^* = 0.05$ everywhere, whose loss is
   $\frac{1}{2}(0.02 + \sigma^2) \approx 0.01$. And so does the unpenalized
   model started from $\mathbf{w} = \mathbf{0}$ ..."

## Repetition removed

- **Factor of 2 (WD-7).** The conversion paragraph in "Concise
  Implementation" no longer re-derives why the framework gradient is twice
  ours; one clause with a reference to `sec_linear_concise`, then only what is
  new here (the decay term is not doubled, so `wd` becomes 6). The sentence
  after the output states the factor once ("the framework's mean squared
  error, twice our loss").
- "Infinitely many weight vectors fit 20 labels" was in the opening, the
  scratch setup, and the ridge subsection; the scratch setup now says "the
  extreme case of the opening".
- Gradient of the penalty $\lambda\mathbf{w}$ stated twice in a row; once now.
- $0.97^{120} \approx 2.6\%$ derived once; later cited as "2.6%" and "0.97".
- The Cauchy-Schwarz bound is no longer written out a second time under the
  twins experiment.
- Norm ranges 0.12 to 0.16 and 0.02 to 0.06 no longer repeated after the
  concise run.
- "Keeps / damps / removes" appeared in the subsection opening, the closing
  paragraph, and the summary; the closing paragraph ("We can now control ...
  Two questions remain open") is replaced by a three-sentence ending, and the
  summary is the only recap.
- Statement of the unpenalized result before its cell ("fits exactly, yet its
  validation loss stays high") removed; it is said once, after the output.
- Tab prose: "adds `wd` times the parameter to its gradient" was in the shared
  prose and in three tabs; the tabs now say only what is framework-specific.
- Scratch model paragraph no longer retells what `LinearRegressionScratch`
  stores and how it trains.

## Review findings

Applied (prose): WD-2 (the 0.05 against 3 contrast, at the experiment), WD-3
(monomial formula and symbols $k$, $q$ cut; the three counts kept), WD-4,
WD-5 (gloss added), WD-6 (raw-versus-centered caveat reduced to a
parenthesis), WD-7 (one clause plus reference), WD-10 (summary now ends on
the duplicated features), and the prose half of WD-1: the opening no longer
promises a "repair", the result paragraph ends on a question instead of a
success, the text says the constant predictor does as well, and the
duplicated-features experiment is named as the case the penalty is for.

Deferred:
- WD-1 proper (make the twins experiment the main one, or redesign the
  200-feature one): needs new cells and outputs.
- WD-8 (replace the ledger paragraphs by a table): partly done by trimming
  numbers; a table of per-run values would need the outputs of all four
  frameworks and is better produced by a cell.
- WD-9 ($\mu$ for eigenvalues, `sigma` against $\sigma$): symbols are in code
  and figures; chapter-wide decision.
- Structural findings (four ideas in one section; moving the twins
  experiment forward; shortening the $\ell_1$ subsection): would move cells.

## Factual claims added or changed

1. The Gaussian-prior formula gives $\lambda = 0.01^2/(20 \cdot 0.01^2) = 0.05$
   for the experiment, and the experiment's 3 is sixty times that. (The
   formula is in the book's convention with the factor $\frac{1}{2}$, the same
   as `WeightDecayScratch`, so no factor of 2 enters.) This gives away the
   last part of exercise 3(a) and the number quoted in exercise 5(c).
2. At $\lambda = 0.05$ and $\eta = 0.01$, 120 steps keep
   $0.9995^{120} \approx 94\%$ of the weights, so "the penalty would leave
   almost no trace". Not run; it follows from the decay factor and from
   shrinkage factors $\mu/(\mu + 0.05) \approx 0.99$ in the determined
   directions.
3. A model that predicts $b^* = 0.05$ for every input has expected loss
   $\frac{1}{2}(\|\mathbf{w}^*\|^2 + \sigma^2) = \frac{1}{2}(0.02 + 0.0001)
   \approx 0.01$, the same as the regularized model (stored validation losses
   0.0097 to 0.0104 across the four frameworks).
4. "All 20 of its eigenvalues are 4 or larger" (was "between about 4 and
   19"; stored minima 4.8 to 5.3).
5. Centering "changes the numbers only slightly" replaces "lower df(3) a
   little, to about 14".
6. The concise validation loss is described as "twice our loss" and as
   matching the scratch run (stored: 0.0193 to 0.0207 against 2 x 0.0097 to
   0.0104).
7. Judgments: the damping of poorly determined directions "is the one that
   makes weight decay worth having"; removal of the initialization "is all
   that helped in our experiment" (the second is the text's own earlier
   conclusion).
8. Closing: "In practice, weight decay is this one optimizer argument".

## Slides against the body

No slide contradicts the revised body. Two omissions: the slide "λ = 3
Halves the Validation Loss" presents the halving without the body's
qualification at that point (a later slide does say the gain came from
removing the initialization), and no slide mentions that the prior suggests
0.05 or that the constant predictor reaches the same loss. The slide "Halve
η, Double the Decay, and the Steps Match" still carries the full conversion,
which is fine for a slide.
