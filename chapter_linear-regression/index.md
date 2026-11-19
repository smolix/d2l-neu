# Linear Regression
:label:`chap_regression`

Before we worry about making our neural networks deep, it pays to build shallow
ones in which the inputs connect directly to the outputs. Starting here has two
big advantages. First, without complex architectures to distract us, we can focus
squarely on the universal machinery of neural network training—parameterizing a
model, feeding it data, choosing a loss function, optimizing its parameters, and
judging whether it will generalize to new examples. Second, this family of
single-layer networks turns out to coincide with classical linear models, which
remain workhorses of statistics and machine learning and serve as essential
baselines whenever we want to justify a fancier architecture.

To make things concrete, imagine you are looking at a house for sale and trying
to judge whether the asking price is fair. Intuitively, you recall what similar
homes in the neighborhood recently sold for and adjust up or down for a larger
floor plan or an older roof. *Linear regression* formalizes this intuition with
numbers: it predicts a continuous target as a weighted sum of the input
measurements plus a constant offset, choosing the weights to keep the squared
prediction errors small on past sales. Each weight represents a single
adjustment, such as the price per square meter. The method itself originated in
astronomy at the turn of the nineteenth century: Adrien-Marie Legendre published
the method of least squares in 1805 to compute comet orbits, and Carl Friedrich
Gauss—who famously used it to recover the lost dwarf planet Ceres in 1801—published
his probabilistic derivation in 1809, noting that he had been using the technique
since 1795 :cite:`Legendre.1805,Gauss.1809`.

:numref:`chap_preliminaries` gave us our mathematical and computational tools;
this chapter assembles them into our first complete learning pipeline and asks
what can go wrong along the way. Two silent failure modes appear here that will
recur throughout the rest of the book: a buggy implementation can run cleanly
without raising an error while learning the wrong parameters, and a model can fit
its training data near-perfectly while predicting badly on unseen examples. Both
are far easier to dissect while the model itself is still linear.

We start with the mathematical foundations in :numref:`sec_linear_regression`,
deriving the linear model, squared-error loss, its closed-form solution, its
probabilistic interpretation under Gaussian noise, and minibatch stochastic
gradient descent—the optimization engine behind nearly every model in this book.
The next four sections turn these equations into trustworthy software. First, we
generate synthetic data whose true parameters we know
(:numref:`sec_synthetic-regression-data`) and implement the entire training loop
from scratch so we can verify that our optimizer recovers the ground truth
(:numref:`sec_linear_scratch`). We then organize this boilerplate into a clean
set of reusable object-oriented abstractions—`Module`, `DataModule`, and
`Trainer`—that every subsequent chapter builds upon (:numref:`sec_oo-design`),
before seeing how high-level framework APIs let us rewrite linear regression in
just a few lines of code (:numref:`sec_linear_concise`). Finally, we confront the
central statistical challenge of machine learning: understanding when a good fit
on training data translates to accurate predictions on new data
(:numref:`sec_generalization_basics`), and controlling a model's complexity
smoothly through *weight decay* rather than dropping features outright
(:numref:`sec_weight_decay`).

In :numref:`chap_classification`, we will apply this exact same recipe to
classification problems; in :numref:`chap_perceptrons`, we will stack multiple
layers with nonlinear activations to build our first truly deep networks.

```toc
:maxdepth: 2

linear-regression
synthetic-regression-data
linear-regression-scratch
oo-design
linear-regression-concise
generalization
weight-decay
```

## Resources and Further Reading {.unnumbered}

The resources are grouped into books, courses, and papers, and each names the section it accompanies. All are freely accessible online except the two papers, as noted.

**Books**

- [An Introduction to Statistical Learning (ISL) — James, Witten, Hastie & Tibshirani](https://www.statlearning.com/) — free PDF (R and Python editions); the gentlest rigorous treatment of regression, classification, and regularization, with worked labs; companion to :numref:`sec_linear_regression` and :numref:`sec_generalization_basics`.
- [The Elements of Statistical Learning (ESL) — Hastie, Tibshirani & Friedman](https://hastie.su.domains/ElemStatLearn/) — free PDF; the graduate companion to ISL, with deep coverage of shrinkage, the bias–variance tradeoff, and model selection (:numref:`sec_generalization_basics`, :numref:`sec_weight_decay`).
- [Pattern Recognition and Machine Learning — Christopher Bishop](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) — free PDF from Microsoft Research; a probabilistic, Bayesian-flavoured account of linear models and regularization (:numref:`sec_linear_regression`, :numref:`sec_weight_decay`).
- [Probabilistic Machine Learning: An Introduction — Kevin Murphy](https://probml.github.io/pml-book/book1.html) — free draft PDF; a modern, unifying probabilistic perspective spanning regression through deep learning (:numref:`sec_linear_regression`, :numref:`sec_generalization_basics`).
- [Understanding Deep Learning (UDL) — Simon Prince](https://udlbook.github.io/udlbook/) — free PDF; a figure-rich modern treatment whose loss-function chapter generalizes the "match the loss to the noise model" recipe of :numref:`sec_linear_regression` into a full menu.
- [Deep Learning: Foundations and Concepts — Bishop & Bishop (2024)](https://www.bishopbook.com/) — free to read online; the successor to PRML, connecting the probabilistic view of linear regression and regularization directly to deep networks (:numref:`sec_linear_regression`, :numref:`sec_weight_decay`).
- [Introduction to Applied Linear Algebra: Vectors, Matrices, and Least Squares (VMLS) — Boyd & Vandenberghe](https://web.stanford.edu/~boyd/vmls/) — free PDF, slides, and code; an applications-first text built around least squares, the computation of :numref:`sec_linear_regression`.

**Courses and video lectures**

- [Supervised Machine Learning: Regression and Classification — Andrew Ng (Stanford / DeepLearning.AI)](https://www.coursera.org/learn/machine-learning) — free to audit; the foundational course covering both settings of this part of the book, starting from linear regression trained by gradient descent (:numref:`sec_linear_scratch`).
- [Statistical Learning — Hastie & Tibshirani (Stanford Online, edX)](https://www.edx.org/learn/statistics/stanford-university-statistical-learning) — free to audit; lecture series following ISL, including regression, cross-validation, ridge, and the LASSO (:numref:`sec_generalization_basics`, :numref:`sec_weight_decay`).

**Foundational papers**

- [Ridge Regression: Biased Estimation for Nonorthogonal Problems — Hoerl & Kennard (1970), *Technometrics*](https://www.tandfonline.com/doi/abs/10.1080/00401706.1970.10488634) — the origin of $\ell_2$ regularization, the weight decay of :numref:`sec_weight_decay` (paywalled).
- [Regression Shrinkage and Selection via the Lasso — Tibshirani (1996), *JRSS-B*](https://academic.oup.com/jrsssb/article/58/1/267/7027929) — introduces the $\ell_1$-penalized counterpart, contrasting sparse selection with ridge's shrinkage (paywalled).

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §2]{.kicker}

A complete learning procedure in its simplest form<br>
**model · loss · minibatch SGD · synthetic data · generalization · weight decay**
:::
:::

::: {.slide title="Every Part of Learning Appears in Linear Regression"}
[The problem]{.kicker}

- Predict a price from area and age with
  $\hat{y} = \mathbf{w}^\top \mathbf{x} + b$: a loss measures the misfit,
  minibatch SGD lowers it.
- **A wrong program still runs:** one shape bug, a falling loss, and wrong
  weights.
- **A perfect fit can mislead:** 20 weights fit 20 noisy points exactly and
  miss new ones.

::: {.d2l-note}
Both failures recur in every later chapter. Linear regression is simple
enough to analyze them.
:::
:::

::: {.slide title="By the End of This Chapter You Can"}
[Goals]{.kicker}

- fit a linear model in closed form and by minibatch SGD;
- derive a loss from a model of the noise;
- verify an implementation against known parameters, to within a few
  standard errors $\sigma/\sqrt{n}$;
- diagnose underfitting and overfitting, and select a model on validation
  data;
- limit a model's flexibility continuously with a penalty of strength
  $\lambda$ on large weights.
:::

::: {.slide title="Map of the Chapter"}
[Chapter map]{.kicker}

| section | question | adds |
|---|---|---|
| Linear Regression | how does a linear model predict, and how do we fit it? | model, loss, closed form, minibatch SGD |
| Synthetic Data | how do we test training when we know the answer? | data with known $\mathbf{w}^*, b^*$ |
| Implementation from Scratch | what does each part look like when we write it? | SGD and the training loop by hand |
| Object-Oriented Design | how do we organize model, data, and training once? | `Module`, `DataModule`, `Trainer` |
| Concise Implementation | which framework parts replace ours? | framework layers and losses |
| Generalization | when does a good fit predict new data well? | training vs. test error |
| Weight Decay | how do we limit flexibility continuously? | a penalty on large weights |
:::

::: {.slide title="Every Section Works on a Running Example With Real Numbers"}
[Running examples]{.kicker}

| section | running example |
|---|---|
| Linear Regression | a 150 m² house, predicted at 480,000 dollars; the three-point line (inputs 0, 1, 2; labels 1, 1, 0) |
| Synthetic Data | $\mathbf{w}^* = (2, -3.4)$, $b^* = 4.2$, noise $\sigma = 0.01$; 1,000 training, 1,000 validation examples |
| Implementation from Scratch | minibatches of 32; loss floor $\sigma^2/2 = 5 \times 10^{-5}$ |
| Object-Oriented Design | the same data and model, trained once more through `Module`, `DataModule`, `Trainer` |
| Concise Implementation | the same data and model, from framework layers |
| Generalization | 20 noisy samples of a cubic; polynomials of degree 1 to 19 |
| Weight Decay | 200 features, 300 noisy examples; two near-duplicate features |
:::
