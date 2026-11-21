# Linear Regression
:label:`chap_regression`

*Regression* is the workhorse of quantitative prediction whenever the target is a meaningful number rather than a discrete category: estimating a home's sale price from its floor area and age, forecasting a patient's hospital stay from vital signs, or predicting next quarter's electricity demand. It is also the ideal proving ground for deep learning. Every essential ingredient of modern neural network training—parameterized forward passes, loss functions derived from maximum likelihood, minibatch stochastic gradient descent, object-oriented training harnesses, generalization diagnostics, and weight-norm regularization—already appears in its cleanest, most transparent form in *linear regression*.

In this chapter, we build linear regression from first principles all the way to a complete, reusable training pipeline:

- **Theory and Statistical Foundations (:numref:`sec_linear_regression`):** We formulate the affine model and squared error loss, contrast the closed-form normal equations (and their geometric interpretation as orthogonal projection) with minibatch stochastic gradient descent (SGD), show why vectorization is essential on modern hardware, and ground squared loss in maximum-likelihood estimation under Gaussian noise.
- **Controlled Synthetic Data (:numref:`sec_synthetic-regression-data`):** We generate a synthetic regression dataset with known ground-truth parameters $(\mathbf{w}^*, b^*)$ and known noise variance, giving us an exact benchmark for verifying our implementations.
- **Implementation from Scratch (:numref:`sec_linear_scratch`):** We implement the model, squared loss, minibatch iterator, and SGD training loop from raw tensors and automatic differentiation—checking that the training loss lands near the theoretical noise floor and that the learned parameters recover $(\mathbf{w}^*, b^*)$.
- **Object-Oriented Design (:numref:`sec_oo-design`):** To avoid rewriting training boilerplate in every chapter, we organize our code around three core abstractions—`Module` (models), `DataModule` (data loaders), and `Trainer` (the training loop)—that power the rest of the book.
- **Concise Implementation (:numref:`sec_linear_concise`):** Once every moving part is transparent, we re-express the exact same model in a few lines using high-level framework layers, built-in losses, and optimizers inside `d2l.Module` and `d2l.Trainer`.
- **Generalization and Model Selection (:numref:`sec_generalization_basics`):** Using polynomial curve fitting, we distinguish training error from generalization error, unpack the bias--variance trade-off, and use validation splits and $K$-fold cross-validation to select model complexity.
- **Weight Decay (:numref:`sec_weight_decay`):** Finally, we tame high-dimensional overfitting by adding an $\ell_2$ penalty (ridge regression), showing both how it acts as per-step weight decay in SGD and how it selectively shrinks poorly determined eigendirections of the features.

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

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §2]{.kicker}

Linear Neural Networks for Regression<br>
**affine models · squared loss · normal equations · minibatch SGD · generalization · weight decay**
:::
:::

::: {.slide title="Linear Regression Is Deep Learning in Miniature"}
[The Proving Ground]{.kicker}

Every essential ingredient of modern neural network training appears in its most transparent, analytically solvable form in linear regression:

- **Parameterized forward pass and loss:** Predict a continuous target via $\hat{y} = \mathbf{w}^\top \mathbf{x} + b$ and score misfit via the squared error $L(\mathbf{w}, b) = \frac{1}{2n}\|\mathbf{X}\mathbf{w} + b - \mathbf{y}\|_2^2$, grounded in Gaussian maximum likelihood.
- **Exact closed form vs. scalable iteration:** The normal equations $\mathbf{X}^\top\mathbf{X}\mathbf{w}^* = \mathbf{X}^\top\mathbf{y}$ project $\mathbf{y}$ orthogonally in $O(nd^2 + d^3)$ time, whereas minibatch SGD updates parameters in $O(|\mathcal{B}|d)$ per step.
- **Two recurring failure modes:** A program with a silent shape bug (such as $(32,) - (32, 1)$ broadcasting to $32 \times 32$) still lowers its loss, and a high-degree model that interpolates 20 noisy points fails catastrophically on new data.

::: {.d2l-note}
In linear regression we can solve the exact optimum, compute the theoretical noise floor $\frac{1}{2}\sigma^2$, and diagnose both code bugs and statistical overfitting from first principles.
:::
:::

::: {.slide title="From First Principles to a Reusable Training Pipeline"}
[Architectural Roadmap]{.kicker}

We develop linear regression across seven stages, moving from mathematical foundations to an object-oriented training harness, concise framework APIs, and statistical regularization:

| Section | Core Question | Key Mathematical & Systems Concepts |
|---|---|---|
| **§2.1 Linear Regression** | How do we formulate and optimize a linear predictor? | Affine model $\mathbf{X}\mathbf{w} + b$, squared loss, normal equations, minibatch SGD, Gaussian MLE |
| **§2.2 Synthetic Data** | How do we verify an optimizer when real data lack an answer key? | Ground-truth $\mathbf{w}^*, b^*$, noise floor $\sigma^2/2$, standard error $\sigma/\sqrt{n}$ |
| **§2.3 From Scratch** | What happens inside each step of training when we write it by hand? | Tensor autograd loop, broadcasting guards, learning-rate stability |
| **§2.4 Object-Oriented Design** | How do we decouple models, datasets, and training loops once for the book? | `Module`, `DataModule`, `Trainer`, size-weighted `Averager`, `ProgressBoard` |
| **§2.5 Concise Implementation** | How do framework layers, losses, and optimizers replace hand-written parts? | `LazyLinear` / `Dense` / `nnx.Linear`, MSE loss scaling, framework `SGD` |
| **§2.6 Generalization** | When does a low training loss translate into accurate predictions on new data? | Empirical vs. population risk, bias–variance decomposition, $K$-fold cross-validation |
| **§2.7 Weight Decay** | How do we control model capacity continuously in high dimensions? | $\ell_2$ ridge penalty vs. $\ell_1$ lasso sparsity, MAP Gaussian prior, spectral shrinkage |
:::

::: {.slide title="Concrete Running Benchmarks Across the Chapter"}
[Quantitative Anchors]{.kicker}

Every section tests its theory and code against a concrete numerical benchmark where exact error floors and parameter recovery tolerances are known in advance:

| Section | Running Dataset & Ground Truth | Diagnostic Target & Verification Criterion |
|---|---|---|
| **§2.1 Theory & Loss** | House pricing ($150\text{ m}^2$, $20\text{ yr} \to \$480\text{k}$); 3-point line $(0,1), (1,1), (2,0)$ | Exact projection $\hat{\mathbf{w}} = (7/6, -1/2)^\top$ with orthogonal residual $\mathbf{X}^\top\mathbf{r} = \mathbf{0}$ |
| **§2.2–§2.5 Synthetic Testbed** | $n = 1{,}000$ train / $1{,}000$ val examples, $\mathbf{w}^* = (2, -3.4)^\top$, $b^* = 4.2$, $\sigma = 0.01$ | Loss drops from $16.6$ to floor $\sigma^2/2 = 5 \times 10^{-5}$ (or $\sigma^2 = 10^{-4}$); weights recovered within $\sim 3\sigma/\sqrt{n} \approx 10^{-3}$ |
| **§2.6 Polynomial Fitting** | $n = 20$ noisy samples of cubic $5 + 1.2x - 3.4x^2 + 5.6x^3$ on $[-1, 1]$, $\sigma = 0.1$ | Degree $1$ underfits (high bias); degree $3$ hits $\sigma^2 = 0.01$; degree $19$ interpolates noise (huge variance) |
| **§2.7 High-Dim Ridge** | $d = 200$ features, $n = 300$ noisy examples, $\mathbf{w}^* = (0.1, \ldots, 0.1)^\top$, $\sigma = 1.0$ | Unregularized SGD overfits (val loss $> 1.2$); prior $\lambda = \sigma^2/(n\tau^2) = 1/3$ shrinks effective $\text{df}$ to $\approx 127$ |
:::

::: {.slide title="The Complete Supervised Learning Loop"}
[End-to-End Perspective]{.kicker}

Every model in the rest of this book—from multilayer perceptrons to Transformers—reuses the exact pipeline we construct in Chapter 2:

::: {.cols .vc}
::: {.col}
- **Forward & loss (`Module`):** Map minibatch inputs $\mathbf{X} \in \mathbb{R}^{|\mathcal{B}| \times d}$ to predictions $\hat{\mathbf{y}}$ and evaluate the penalized empirical loss $L_{\mathcal{B}}(\mathbf{w}, b) + \frac{\lambda}{2}\|\mathbf{w}\|_2^2$.
- **Backward & update (`Trainer`):** Differentiate $L_{\mathcal{B}}$ via reverse-mode autograd and step parameters via $(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta \nabla L_{\mathcal{B}}$.
- **Validation & selection (`DataModule`):** Track held-out risk after every epoch to tune hyperparameters $(\eta, |\mathcal{B}|, \lambda)$ without leaking test data.
:::

::: {.col .fig}
![](../img/singleneuron.svg){.nostretch width=100%}
:::
:::

::: {.d2l-note .rule}
Viewed as a neural network, linear regression is a single fully connected layer with $d$ inputs and $1$ linear output neuron.
:::
:::
