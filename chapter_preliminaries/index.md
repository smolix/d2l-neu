# Preliminaries
:label:`chap_preliminaries`

To prepare for your dive into deep learning, you will need a few essential survival skills: (i) techniques for storing and manipulating multidimensional data in memory (:numref:`sec_ndarray`); (ii) tools for ingesting and cleaning tabular data from raw files, complete with missing entries and categorical text fields (:numref:`sec_pandas`); (iii) fluency in the linear-algebraic operations that transform high-dimensional representations and measure their size (:numref:`sec_linear-algebra`); (iv) differential calculus to determine which direction to adjust each parameter in order to decrease a loss function (:numref:`sec_calculus`); (v) automatic differentiation so that the framework handles the chain rule across millions of parameters for you (:numref:`sec_autograd`); and (vi) a working grasp of probability and statistics, our primary language for reasoning under uncertainty and distinguishing genuine signal from sampling noise (:numref:`sec_prob`).

What deep learning adds to classical linear algebra, calculus, and probability is *scale*---applying them to millions or billions of variables where calculation by hand is out of the question. Doing so relies on two software pillars that every subsequent chapter takes for granted: *tensor libraries*, which apply compiled operations to entire arrays of numbers in parallel on CPUs and GPUs, and *automatic differentiation*, which differentiates arbitrary numerical programs mechanically.

This chapter provides a rapid, hands-on crash course in these prerequisites. If you are already comfortable with NumPy, linear algebra, and multivariable calculus, you can skim the familiar parts---though we still recommend reading :numref:`sec_autograd` on automatic differentiation and :numref:`sec_prob` on standard errors and statistical estimation, as later chapters rely on both constantly. For a deeper theoretical foundation with complete proofs and derivations, our *Mathematics for Deep Learning* appendix develops linear algebra (:numref:`chap_mdl-linear-algebra`), calculus (:numref:`chap_mdl-calculus`), and probability and statistics (:numref:`chap_mdl-probability-statistics`) in full detail.

```toc
:maxdepth: 2

ndarray
pandas
linear-algebra
calculus
autograd
probability
```

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §1]{.kicker}

Mathematical Foundations and Software Pillars of Deep Learning<br>**tensors · data preprocessing · linear algebra · calculus · automatic differentiation · probability**
:::
:::

::: {.slide title="Why Deep Learning Needs Two New Software Pillars"}
[Motivation]{.kicker}

Classical machine learning rests on linear algebra, multivariable calculus, and probability—what deep learning adds is **scale**, applying them to millions or billions of parameters where hand derivation and scalar Python loops break down.

::: {.cols}
::: {.col}
- **The Scale of One Minibatch:** One Fashion-MNIST batch of 256 images ($28 \times 28$) holds $200{,}704$ pixels; a 10-class linear layer uses $7{,}850$ weights, while foundation models update $10^9$–$10^{12}$ weights per step.
- **Pillar 1 — Tensor Libraries (§1.1–1.3):** Replace scattered Python objects and loops with contiguous memory buffers and parallel SIMD/GPU kernels that transform entire minibatches at once.
:::

::: {.col}
- **Pillar 2 — Automatic Differentiation (§1.4–1.5):** Tracks computational graphs to apply the chain rule mechanically across arbitrary numerical programs without hand-coded derivatives.
- **Statistical Reasoning (§1.6):** Quantifies uncertainty under finite-sample noise so we can distinguish genuine generalization improvements from random fluctuations and update beliefs from evidence.
:::
:::

::: {.d2l-note}
Every architecture in this book—from linear regression to Transformers—is built by composing these exact primitives inside an iterative training loop.
:::
:::

::: {.slide title="Anatomy of a Complete Machine Learning Iteration"}
[The Learning Loop]{.kicker}

Every supervised learning pipeline executes a closed feedback loop from raw records on disk to statistically validated predictions:

| Stage in the Loop | Mathematical & Computational Operation | Chapter 1 Section |
| :--- | :--- | :--- |
| **1. Ingest & Clean** | Parse raw tables, impute `NaN`s, encode categories, standardize scales | **§1.2 Data Preprocessing** |
| **2. Store & Batch** | Pack examples into contiguous `float32` / `bfloat16` tensor buffers | **§1.1 Data Manipulation** |
| **3. Forward Pass** | Transform minibatch representations via matrix products $\mathbf{X}\mathbf{W} + \mathbf{b}$ | **§1.3 Linear Algebra** |
| **4. Measure & Differentiate** | Evaluate scalar loss $\mathcal{L}$ and relate parameter perturbations to $\nabla \mathcal{L}$ | **§1.4 Calculus** |
| **5. Backpropagate** | Traverse the reverse computational graph to accumulate parameter gradients | **§1.5 Automatic Differentiation** |
| **6. Evaluate Under Noise** | Bound test-set uncertainty via standard errors and Bayesian inference | **§1.6 Probability & Statistics** |
:::

::: {.slide title="Concrete Running Examples Anchoring Each Section"}
[Pedagogical Thread]{.kicker}

Rather than treating mathematics in the abstract, each section develops its tools around a concrete numerical dataset or estimation problem:

| Section | Core Abstraction | Concrete Running Benchmark in Chapter 1 |
| :--- | :--- | :--- |
| **§1.1 Data Manipulation** | Strided storage buffers & views | `images`: Fashion-MNIST minibatch `(256, 1, 28, 28)` in `uint8` $\to$ `float32` |
| **§1.2 Data Preprocessing** | Leakage-free tabular pipelines | `house_tiny.csv`: 10 homes with missing rooms, text roofs, and $500\times$ scale gaps |
| **§1.3 Linear Algebra** | Batched matmuls, norms, spectra | Flattened batch $\mathbf{X} \in \mathbb{R}^{256 \times 784}$ multiplying weight matrix $\mathbf{W} \in \mathbb{R}^{784 \times 10}$ |
| **§1.4–1.5 Calculus & Autograd** | Jacobians, chain rule, reverse-mode AD | Fitting a line to 3 data points; backpropagating through Python control flow |
| **§1.6 Probability & Statistics** | Bayes' rule & confidence intervals | Diagnostic testing under low prevalence ($P(\text{disease} \mid +)$) & coin-bias standard error |
:::

::: {.slide title="How to Navigate Chapter 1 and the Math Appendix"}
[Study Guide]{.kicker}

::: {.cols}
::: {.col}
- **Hands-on systems focus:** Every concept in §1.1–1.6 is paired immediately with executable framework code so you build intuition for shapes, dtypes, memory aliasing, and gradients before training neural networks in Chapter 2.
- **Where not to skip:** Even if you know NumPy and multivariable calculus well, study **§1.5 (Automatic Differentiation)** for computational graph semantics (`detach`, in-place mutation hazards) and **§1.6 (Probability)** for standard errors on finite test sets.
:::

::: {.col}
- **Deeper proofs and derivations:** When you want full theoretical developments—singular value decomposition, matrix calculus identities, measure-theoretic probability, or maximum likelihood asymptotics—consult the *Mathematics for Deep Learning* appendix.
:::
:::

::: {.d2l-note}
**Up next (§1.1 Data Manipulation):** How tensors represent multidimensional arrays in contiguous memory and execute parallel operations without Python loops.
:::
:::
