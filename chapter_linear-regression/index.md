# Linear Regression
:label:`chap_regression`

Suppose we want to predict a number from measurements: the price of a house
from its area and age, a patient's stay in hospital from test results, next
quarter's demand for a product. Linear regression predicts the number as a
weighted sum of the measurements plus a constant. It goes back two
centuries, to Legendre and Gauss, and is still the first method to try. It
also contains every part of learning in its simplest form: a model with parameters, a loss that
measures how badly the parameters fit, an algorithm that improves them, and
predictions for examples the model has not seen.

:numref:`chap_preliminaries` supplied the tools; this chapter assembles them
into a complete learning procedure and asks what can go wrong. A model can
fit its training data well and still predict new data badly, and an
implementation can be wrong without raising an error. Both problems stay with us in every later
chapter, so we study them where the model is simple enough to analyze.

:numref:`sec_linear_regression` asks how a linear model predicts, which loss
measures its errors, and how minibatch stochastic gradient descent reduces
that loss. :numref:`sec_oo-design` asks how to organize models, data, and
training so that later chapters change only the part under study.
:numref:`sec_synthetic-regression-data` asks how to test a training procedure
on data whose true parameters we know. :numref:`sec_linear_scratch` asks what
each part of training looks like when we write it ourselves, and how to check
it. :numref:`sec_linear_concise` asks how the layers, losses, and optimizers
of a framework replace those parts. :numref:`sec_generalization_basics` asks
when a model that fits its training data predicts new data well, and how to
choose among models. :numref:`sec_weight_decay` asks how to limit a model's
flexibility continuously, rather than by dropping features.

The next chapter applies the same procedure to classification, and the one
after replaces the linear model by a multilayer network.

```toc
:maxdepth: 2

linear-regression
oo-design
synthetic-regression-data
linear-regression-scratch
linear-regression-concise
generalization
weight-decay
```

## Resources and Further Reading {.unnumbered}

The resources are grouped into books, courses, and papers, and each names the section it accompanies. The next chapter builds on the shared statistical references rather than listing them again. All are freely accessible online except the two papers, as noted.

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
| Object-Oriented Design | how do we organize model, data, and training once? | `Module`, `DataModule`, `Trainer` |
| Synthetic Regression Data | how do we test training when we know the answer? | data with known $\mathbf{w}^*, b^*$ |
| Implementation from Scratch | what does each part look like when we write it? | SGD and the training loop by hand |
| Concise Implementation | which framework parts replace ours? | framework layers and losses |
| Generalization | when does a good fit predict new data well? | training vs. test error |
| Weight Decay | how do we limit flexibility continuously? | a penalty on large weights |
:::

::: {.slide title="Every Section Works on a Running Example With Real Numbers"}
[Running examples]{.kicker}

| section | running example |
|---|---|
| Linear Regression | a 150 m² house, predicted at 480,000 dollars; the three-point line (inputs 0, 1, 2; labels 1, 1, 0) |
| Object-Oriented Design | $\mathbf{w}^* = (2, -3.4)$, $b^* = 4.2$ in `Module`, `DataModule`, `Trainer` |
| Synthetic Regression Data | the same $\mathbf{w}^*, b^*$, noise $\sigma = 0.01$; 1,000 training, 1,000 validation examples |
| Implementation from Scratch | minibatches of 32; loss floor $\sigma^2/2 = 5 \times 10^{-5}$ |
| Concise Implementation | the same data and model, from framework layers |
| Generalization | 20 noisy samples of a cubic; polynomials of degree 1 to 19 |
| Weight Decay | 200 features, 20 examples; two near-duplicate features |
:::
