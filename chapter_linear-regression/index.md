# Linear Regression
:label:`chap_regression`

*Regression* is the workhorse of quantitative prediction whenever the target is a meaningful number rather than a discrete category: estimating a home's sale price from its floor area and age, forecasting a patient's hospital stay from vital signs, or predicting next quarter's electricity demand. It is also the ideal proving ground for deep learning. Every essential ingredient of modern neural network training—parameterized forward passes, loss functions derived from maximum likelihood, minibatch stochastic gradient descent, object-oriented training harnesses, generalization diagnostics, and weight-norm regularization—already appears in its cleanest, most transparent form in *linear regression*.

In this chapter, we build linear regression from first principles all the way to a complete, reusable training pipeline:

- **Theory and Statistical Foundations (:numref:`sec_linear_regression`):** We formulate the affine model and squared error loss, contrast the closed-form normal equations (and their geometric interpretation as orthogonal projection) with minibatch stochastic gradient descent (SGD), show why vectorization is essential on modern hardware, and ground squared loss in maximum-likelihood estimation under Gaussian noise.
- **Controlled Synthetic Data (:numref:`sec_synthetic-regression-data`):** We generate a synthetic regression dataset with known ground-truth parameters $(\mathbf{w}^*, b^*)$ and known noise variance, giving us an exact benchmark for verifying our implementations.
- **From-Scratch and Concise Implementations (:numref:`sec_linear_scratch`):** We first implement the model, squared loss, minibatch iterator, and SGD training loop from raw tensors and automatic differentiation—checking that the training loss lands near the theoretical noise floor—and then re-express the exact same model in a few lines using high-level framework layers, built-in losses, and optimizers.
- **Object-Oriented Design (:numref:`sec_oo-design`):** To avoid rewriting training boilerplate in every chapter, we organize our code around three core abstractions—`Module` (models), `DataModule` (data loaders), and `Trainer` (the training loop)—that power the rest of the book.
- **Generalization and Model Selection (:numref:`sec_generalization_basics`):** Using polynomial curve fitting, we distinguish training error from generalization error, unpack the bias--variance trade-off, and use validation splits and $K$-fold cross-validation to select model complexity.
- **Weight Decay (:numref:`sec_weight_decay`):** Finally, we tame high-dimensional overfitting by adding an $\ell_2$ penalty (ridge regression), showing both how it acts as per-step weight decay in SGD and how it selectively shrinks poorly determined eigendirections of the features.

```toc
:maxdepth: 2

linear-regression
synthetic-regression-data
linear-regression-scratch
oo-design
generalization
weight-decay
```

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
