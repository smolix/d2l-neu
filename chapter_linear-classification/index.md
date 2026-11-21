# Linear Classification
:label:`chap_classification`

In :numref:`chap_regression`, our goal was to predict a continuous quantity such as the sale price of a house. Many real-world tasks, however, ask for a discrete category instead: an online catalog photo depicts a shirt, a coat, or a sandal—never "5.5" of anything. Simply numbering the categories 1 through 10 and running linear regression fails badly, because it imposes an arbitrary ordering as though a pullover lay halfway between a pair of trousers and a dress, while leaving the model unable to express calibrated uncertainty ("most likely a shirt, possibly a pullover").

Happily, almost all of our regression training machinery—minibatch data loaders, vectorized forward passes, automatic differentiation, and minibatch SGD—carries straight over to *classification*. Only three ingredients change: we represent unordered categories via *one-hot vectors*, compute one linear score (*logit*) per category and normalize them via the *softmax* function into a probability distribution, and replace squared error with the *cross-entropy loss* (derived from maximum likelihood under a categorical label model, just as squared error arose from Gaussian noise). Virtually every neural classifier in this book—right up to today's largest language models—terminates in this exact softmax layer and trains under this exact loss.

Where classification departs sharply from regression is in *evaluation* and *deployment*. A predicted probability is not yet an action; raw accuracy can mislead whenever one class is rare or different mistakes carry asymmetric costs; a finite test set carries statistical error bars that erode under repeated reuse; and real deployment environments often shift away from the training distribution. This chapter unfolds in six sections:

- **Softmax Regression and Cross-Entropy (:numref:`sec_softmax`):** We formulate one-hot encodings, linear logits, the softmax transformation, and the cross-entropy loss, deriving its clean residual gradient $\hat{\mathbf{y}} - \mathbf{y}$ and connecting it to Shannon's information theory and temperature scaling.
- **Image Classification Data (:numref:`sec_fashion_mnist`):** We introduce the Fashion-MNIST benchmark ($60{,}000$ training and $10{,}000$ validation images across 10 garment categories) and build a reusable `FashionMNIST` data module.
- **Softmax Regression Implementation (:numref:`sec_softmax_scratch`):** We implement softmax regression both from scratch and concisely (:numref:`sec_softmax_concise`), introducing the `Classifier` base class, diagnosing misclassifications via the confusion matrix and learned $28 \times 28$ weight templates, and resolving floating-point overflow and underflow via the fused log-sum-exp cross-entropy loss.
- **Measuring Classifiers (:numref:`sec_classification`):** We examine class imbalance, precision, recall, cost-sensitive decision thresholds, reliability diagrams, expected calibration error (ECE), and post-hoc temperature scaling.
- **Generalization in Classification (:numref:`chap_classification_generalization`):** We quantify test-set confidence intervals (normal vs. Hoeffding), selection bias and adaptive overfitting from test-set reuse, and uniform convergence via the Vapnik--Chervonenkis (VC) dimension.
- **Environment and Distribution Shift (:numref:`sec_environment-and-distribution-shift`):** We distinguish covariate, label, and concept shift, implement importance-weighted domain adaptation and confusion-matrix label-shift correction, and examine feedback loops in deployed decision systems.

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

Predicting an unordered discrete category ($y \in \{1, \ldots, q\}$) departs from continuous regression in three fundamental ways:

- **Arbitrary integer codes distort geometry:** Numbering ten garments $1$ through $10$ and applying squared loss makes a prediction of $5.5$ fall halfway between coat and sandal, while penalizing T-shirt ($1$) $\to$ ankle boot ($10$) $81\times$ more heavily than T-shirt ($1$) $\to$ trouser ($2$).
- **Raw accuracy hides asymmetric failures:** When $1\%$ of patients carry a disease, a trivial classifier that always predicts *healthy* scores $99\%$ accuracy while detecting $0\%$ of sick patients.
- **Test sets vouch only for their own distribution:** A linear model that achieves $85\%$ accuracy on source data can drop to $\sim 50\%$ on shifted deployment inputs—worse than guessing the majority class.

::: {.d2l-note}
A neural classifier must output calibrated class probabilities $\hat{\mathbf{y}}$, optimize a proper likelihood loss, and be audited against finite-sample error bars and deployment shifts.
:::
:::

::: {.slide title="From Regression to Classification: What Changes"}
[The Modelling Recipe]{.kicker}

Almost all of our regression training pipeline—minibatch data loaders, vectorized matrix products, automatic differentiation, and minibatch SGD—carries straight over with three targeted replacements:

| Component | Linear Regression (§2) | Softmax Regression (§3) |
|:--|:--|:--|
| **Target representation** | Continuous scalar $y \in \mathbb{R}$ | Discrete class $y \in \{1, \ldots, q\}$, one-hot $\mathbf{y} \in \{0, 1\}^q$ |
| **Forward model** | Single affine score $\hat{y} = \mathbf{w}^\top\mathbf{x} + b$ | $q$ logits $\mathbf{o} = \mathbf{W}^\top\mathbf{x} + \mathbf{b}$ normalized via $\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{o})$ |
| **Noise / label model** | Gaussian $P(y \mid \mathbf{x}) = \mathcal{N}(y; \hat{y}, \sigma^2)$ | Categorical $P(y \mid \mathbf{x}) = \hat{y}_y = \prod_{j=1}^q \hat{y}_j^{y_j}$ |
| **Maximum-likelihood loss** | Squared error $\frac{1}{2}(\hat{y} - y)^2$ | Cross-entropy $l(\mathbf{y}, \hat{\mathbf{y}}) = -\sum_{j=1}^q y_j \log \hat{y}_j = -\log \hat{y}_y$ |
| **Logit / output gradient** | Residual $\partial_{\hat{y}} l = \hat{y} - y$ | Residual vector $\nabla_{\mathbf{o}}\, l = \hat{\mathbf{y}} - \mathbf{y}$ |

::: {.d2l-note .rule}
Virtually every neural classifier—from this chapter's $7{,}850$-parameter baseline to frontier language models—terminates in a linear logit layer paired with the fused softmax cross-entropy loss.
:::
:::

::: {.slide title="Architecture and Roadmap of Chapter 3"}
[Chapter Roadmap]{.kicker}

The six sections progress from mathematical formulation and benchmark data to implementation, decision metrics, statistical guarantees, and deployment robustness:

| Section | Core Question | Key Mathematical & Systems Tools |
|:--|:--|:--|
| **§3.1 Softmax Regression** | How does a linear model output valid class probabilities? | One-hot vectors, $\mathrm{softmax}(\mathbf{o})$, cross-entropy, $\nabla_{\mathbf{o}} l = \hat{\mathbf{y}} - \mathbf{y}$, temperature $T$ |
| **§3.2 Image Classification Data** | Which benchmark separates weak and strong models in seconds? | Fashion-MNIST ($60\text{k} + 10\text{k}$ $28\times 28$ garments), `DataModule` throughput |
| **§3.3 Softmax Implementation** | How do we prevent float32 overflow/underflow and inspect errors? | `Classifier` base class, $10\times 10$ confusion matrix, templates, fused LogSumExp loss |
| **§3.4 Measuring Classifiers** | When does higher accuracy yield a worse decision system? | Precision, recall, Bayes cost threshold $\kappa_{\mathrm{FP}}/(\kappa_{\mathrm{FP}}+\kappa_{\mathrm{FN}})$, ECE, temperature scaling |
| **§3.5 Generalization** | How far can we trust a finite test score or training fit? | Standard error $\le 0.5/\sqrt{n}$, Hoeffding & union bounds, adaptive overfitting, VC dimension |
| **§3.6 Distribution Shift** | How do we adapt when deployment data drift from training data? | Covariate vs. label vs. concept shift, domain-classifier $\beta(\mathbf{x})$, $\mathbf{C}\mathbf{p}_T = \boldsymbol{\mu}$, feedback loops |
:::

::: {.slide title="Concrete Running Benchmarks Across Every Section"}
[Empirical Anchors]{.kicker}

Every concept in this chapter is anchored to concrete numbers and reproducible experiments:

| Section | Running Quantitative Anchor |
|:--|:--|
| **§3.1 Softmax Regression** | $2\times 2$ image with cat/chicken/dog logits $(1.0, 2.2, 0.3)$ $\to$ probabilities $(0.21, 0.69, 0.10)$ and chicken loss $-\log 0.69 \approx 0.37\text{ nats}$ |
| **§3.2 Fashion-MNIST Data** | $60{,}000$ train / $10{,}000$ val grayscale $28\times 28$ images across $10$ classes; minibatches of $64$ ($\approx 200\text{ KB}$) or $256$ ($\approx 0.8\text{ MB}$) in `float32` |
| **§3.3 Implementation** | $784 \times 10 + 10 = 7{,}850$ parameters reach $\sim 83\%$ val / $\sim 84\%$ train accuracy in $10$ epochs; fused LogSumExp handles logits $(1000, 0, 0)$ |
| **§3.4 Measuring Classifiers** | $1\%$ disease prevalence: $90\%$-recall / $5\%$-FPR screener has $95\%$ accuracy and $15\%$ precision; cost ratio $100:1$ sets threshold at $1/101 \approx 1\%$ |
| **§3.5 Generalization** | $n = 10{,}000$ test images give $\pm 0.7\%$ ($95\%$ normal) or $\pm 1.4\%$ (Hoeffding); majority vote of lucky coin-flippers jumps from $50\%$ to $\sim 70\%$ |
| **§3.6 Distribution Shift** | 2D Gaussian shift $(0,0) \to (2,0)$: importance weights lift target accuracy from $\sim 55\%$ to $\sim 95\%$; $\mathbf{C}^{-1}\boldsymbol{\mu}$ recovers a $50\%$ target shirt prior |
:::

::: {.slide title="Concept Check & Chapter Study Guide"}
[Self-Assessment]{.kicker}

Before diving into §3.1, test your intuition on three foundational questions that motivate the chapter:

1. **Output count:** Why does a 10-class linear classifier on $28 \times 28$ images require $7{,}850$ parameters instead of the $785$ parameters used in linear regression?
2. **Numerical range:** In single-precision floating point (`float32`), why does evaluating $\exp(1000) / (\exp(1000) + \exp(0) + \exp(0))$ return `NaN` even though the mathematical ratio is $1$?
3. **Evaluation precision:** If two models differ by $0.3$ percentage points on a $10{,}000$-image test set around $83\%$ accuracy, is that gap statistically distinguishable from sampling noise?

::: {.d2l-note .fragment .fade-up}
**Answers:** **(1)** Each of the $q = 10$ classes needs its own $784$-pixel weight vector $\mathbf{w}_j$ and bias $b_j$ to score class $j$ against all others ($784 \times 10 + 10 = 7{,}850$). **(2)** `float32` overflows to `inf` around $\exp(88.7) \approx 3.4 \times 10^{38}$, turning the ratio into $\infty / \infty = \texttt{NaN}$; subtracting $\max_k o_k = 1000$ first keeps all exponents $\le 0$. **(3)** No: at $\epsilon = 0.17$ and $n = 10{,}000$, a single model's standard error is $\sqrt{0.17 \times 0.83 / 10{,}000} \approx 0.38$ percentage points ($95\%$ radius $\pm 0.74$ points).
:::
:::
