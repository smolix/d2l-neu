# Linear Classification
:label:`chap_classification`

In :numref:`chap_regression`, our goal was to predict a continuous quantity such as the sale price of a house. Many tasks ask for a discrete category instead: an online catalog photo depicts a shirt, a coat, or a sandal, never "5.5" of anything. Simply numbering the categories 1 through 10 and running linear regression fails badly, because it imposes an arbitrary ordering as though a pullover lay halfway between a pair of trousers and a dress, while leaving the model unable to express calibrated uncertainty ("most likely a shirt, possibly a pullover").

Happily, almost all of our regression training machinery—minibatch data loaders, vectorized forward passes, automatic differentiation, and minibatch SGD—carries straight over to *classification*. Only three ingredients change: we represent unordered categories via *one-hot vectors*, compute one linear score (*logit*) per category and normalize them via the *softmax* function into a probability distribution, and replace squared error with the *cross-entropy loss* (derived from maximum likelihood under a categorical label model, just as squared error arose from Gaussian noise). Nearly every neural classifier in this book, right up to large language models, ends in this softmax layer and trains under this loss.

We first formulate softmax regression and the cross-entropy loss, derive its residual gradient $\hat{\mathbf{y}} - \mathbf{y}$, and connect it to Shannon's information theory (:numref:`sec_softmax`). We then build a reusable `FashionMNIST` image pipeline (:numref:`sec_fashion_mnist`) and implement softmax regression both from scratch and concisely (:numref:`sec_softmax_scratch`), diagnosing errors with confusion matrices and learned $28 \times 28$ templates while curing floating-point overflow and underflow with a fused log-sum-exp loss.

Where classification departs sharply from regression is in *evaluation* and *deployment*, which occupy the second half of the chapter. Raw accuracy misleads whenever positive examples are rare or mistakes carry unequal costs, motivating precision, recall, cost-sensitive thresholds, and calibration diagnostics (:numref:`sec_classification`). Even on balanced classes, a finite test set carries statistical error bars that erode under repeated reuse, leading us to Hoeffding bounds, adaptive overfitting, and Vapnik--Chervonenkis (VC) theory (:numref:`chap_classification_generalization`). Finally, :numref:`sec_environment-and-distribution-shift` asks what happens when deployment data drift away from the training distribution, developing importance-weighted covariate-shift correction, confusion-matrix label-shift estimation, and defenses against runaway feedback loops.

```toc
:maxdepth: 2

softmax-regression
image-classification-dataset
softmax-regression-scratch
classification
generalization-classification
environment-and-distribution-shift
```

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §3]{.kicker}

Linear Neural Networks for Classification: From Logits to Trusted Decisions<br>**one-hot labels · softmax · cross-entropy · Fashion-MNIST · calibration · generalization · distribution shift**
:::
:::

::: {.slide title="Why Classification Needs Its Own Loss and Diagnostics"}
[The Central Challenge]{.kicker}

::: {.cols .vc}
::: {.col}
Predicting an unordered discrete category ($y \in \{1, \ldots, q\}$) departs from continuous regression in three fundamental ways:

- **Arbitrary integer codes distort geometry:** Numbering ten garments $1$ through $10$ under squared loss penalizes T-shirt ($1$) $\to$ boot ($10$) $81\times$ more than T-shirt ($1$) $\to$ trouser ($2$).
- **Raw accuracy hides asymmetric failures:** Under $1\%$ disease prevalence, always predicting *healthy* scores $99\%$ accuracy while detecting $0\%$ of sick patients.
- **Test sets vouch only for their own distribution:** A linear model at $85\%$ source accuracy can collapse to $\sim 50\%$ under deployment shift.
:::

::: {.col .fig .big}
@fig:ch03-cls-why-classification
:::
:::
:::

::: {.slide title="From Regression to Classification: What Changes"}
[The Modelling Recipe]{.kicker}

::: {.cols .vc}
::: {.col}
Our regression training pipeline—minibatch loaders, vectorized GEMM, autodiff, and SGD—carries over with three targeted upgrades:

- **One-hot target & multi-logit head:** Replace scalar $y \in \mathbb{R}$ and $\hat{y}$ with one-hot $\mathbf{y} \in \{0, 1\}^q$ and $q$ affine logits $\mathbf{o} = \mathbf{W}^\top\mathbf{x} + \mathbf{b}$ normalized via $\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{o})$.
- **Cross-entropy likelihood loss:** Replace Gaussian squared error with categorical negative log-likelihood $l(\mathbf{y}, \hat{\mathbf{y}}) = -\sum_{j=1}^q y_j \log \hat{y}_j = -\log \hat{y}_y$.
- **Identical residual gradient form:** The logit gradient simplifies to $\nabla_{\mathbf{o}}\, l = \hat{\mathbf{y}} - \mathbf{y}$ (prediction minus target).
:::

::: {.col .fig .big}
@fig:ch03-cls-reg-vs-cls
:::
:::
:::

::: {.slide title="Architecture and Roadmap of Chapter §{num:chap_classification}"}
[Chapter Roadmap]{.kicker}

::: {.cols .vc}
::: {.col}
The six sections progress from mathematical formulation and benchmark data to implementation, decision metrics, statistical guarantees, and deployment robustness:

- **Model & pipeline (§{sec_softmax}–§{sec_softmax_scratch}):** Softmax probabilities, cross-entropy loss, Fashion-MNIST `DataModule` ($60\text{k} + 10\text{k}$ garments), `Classifier` base class, $10\times 10$ confusion matrices, and fused LogSumExp loss.
- **Decisions, bounds & shift (§{sec_classification}–§{sec_environment-and-distribution-shift}):** Precision/recall, Bayes cost thresholds, ECE calibration, test-set error bars ($\le 0.5/\sqrt{n}$), adaptive overfitting, VC dimension, and covariate/label shift correction.
:::

::: {.col .fig .big}
@fig:ch03-cls-roadmap
:::
:::
:::

::: {.slide title="Concrete Running Benchmarks Across Every Section"}
[Empirical Anchors]{.kicker}

::: {.cols .vc}
::: {.col}
Every concept in this chapter is anchored to concrete numbers and reproducible experiments:

- **Core model (§{sec_softmax}–§{sec_softmax_scratch}):** $2\times 2$ toy logits $(1.0, 2.2, 0.3) \to (0.21, 0.69, 0.10)$ ($0.37\text{ nats}$); $7{,}850$-parameter Fashion-MNIST classifier reaching $\sim 83\%$ validation accuracy in $10$ epochs.
- **Auditing & shift (§{sec_classification}–§{sec_environment-and-distribution-shift}):** $1\%$-prevalence screening ($90\%$ recall $\to 15.4\%$ precision, $\tau^* \approx 1\%$ at $100:1$ cost); $\pm 0.7\%$ normal vs. $\pm 1.4\%$ Hoeffding radius at $n=10\text{k}$; importance weighting ($\sim 55\% \to 95\%$) and $\mathbf{C}^{-1}\boldsymbol{\mu}$ label-shift recovery.
:::

::: {.col .fig .big}
@fig:ch03-cls-benchmarks
:::
:::
:::

::: {.slide title="Concept Check & Chapter Study Guide"}
[Self-Assessment]{.kicker}

Before diving into §{sec_softmax}, test your intuition on three foundational questions that motivate the chapter:

1. **Output count:** Why does a 10-class linear classifier on $28 \times 28$ images require $7{,}850$ parameters instead of the $785$ parameters used in linear regression?
2. **Numerical range:** In single-precision floating point (`float32`), why does evaluating $\exp(1000) / (\exp(1000) + \exp(0) + \exp(0))$ return `NaN` even though the mathematical ratio is $1$?
3. **Evaluation precision:** If two models differ by $0.3$ percentage points on a $10{,}000$-image test set around $83\%$ accuracy, is that gap statistically distinguishable from sampling noise?

::: {.d2l-note .fragment .fade-up}
**Answers:** **(1)** Each of the $q = 10$ classes needs its own $784$-pixel weight vector $\mathbf{w}_j$ and bias $b_j$ to score class $j$ against all others ($784 \times 10 + 10 = 7{,}850$). **(2)** `float32` overflows to `inf` around $\exp(88.7) \approx 3.4 \times 10^{38}$, turning the ratio into $\infty / \infty = \texttt{NaN}$; subtracting $\max_k o_k = 1000$ first keeps all exponents $\le 0$. **(3)** No: at $\epsilon = 0.17$ and $n = 10{,}000$, a single model's standard error is $\sqrt{0.17 \times 0.83 / 10{,}000} \approx 0.38$ percentage points ($95\%$ radius $\pm 0.74$ points).
:::
:::
