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

::: {.cols .vc}
::: {.col}
- **Scale breaks scalar code:** One 256-image Fashion-MNIST batch holds $200{,}704$ pixels; foundation models update $10^9\text{--}10^{12}$ weights per step.
- **Pillar 1 — Contiguous Tensor Kernels** Pack data into unboxed RAM/VRAM buffers and transform whole batches on SIMD/GPU lanes.
- **Pillar 2 — Reverse-Mode Autograd** Record the execution DAG to obtain all parameter gradients $\nabla_{\mathbf{w}} L$ in a single backward sweep.
- **Statistical Foundation:** Bound test-set uncertainty ($\sigma/\sqrt{n}$) and update beliefs via Bayes' rule.
:::

::: {.col .fig .big}
@fig:prelim-pillars
:::
:::
:::

::: {.slide title="Anatomy of a Complete Machine Learning Iteration"}
[The Learning Loop]{.kicker}

::: {.cols .vc}
::: {.col}
- **Data to tensors:** Parse raw tables, impute gaps without leakage, and pack batches into contiguous `float32` / `bfloat16` buffers.
- **Forward pass:** Project inputs to class logits via batched matrix products $\mathbf{O} = \mathbf{X}\mathbf{W} + \mathbf{b}$.
- **Loss & backward sweep:** Evaluate scalar loss $L$ and propagate Vector–Jacobian Products backward to compute $\nabla_{\mathbf{W}} L$.
- **Step & statistical evaluation:** Update parameters and quantify finite-sample error bars.
:::

::: {.col .fig .big}
@fig:prelim-ml-loop
:::
:::
:::

::: {.slide title="Concrete Running Examples Anchoring Each Section"}
[Pedagogical Thread]{.kicker}

::: {.cols .vc}
::: {.col}
- **Tensors & linear algebra (`images` batch):** Fashion-MNIST minibatch `(256, 1, 28, 28)` flattened to $\mathbf{X} \in \mathbb{R}^{256 \times 784}$ and multiplied by $\mathbf{W} \in \mathbb{R}^{784 \times 10}$.
- **Data preprocessing (`house_tiny.csv`):** 10 homes with missing rooms, categorical roofs, and $500\times$ scale gaps.
- **Calculus & autograd (line loss & control flow):** Fitting $L(\mathbf{w}) = \|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2$ on 3 points and differentiating through Python loops.
- **Probability (coin tosses & screening):** $1/\sqrt{n}$ confidence bands and two-stage Bayesian medical diagnosis.
:::

::: {.col .fig .big}
@fig:prelim-running-example
:::
:::
:::

::: {.slide title="How to Navigate the Preliminaries and the Math Appendix"}
[Study Guide]{.kicker}

::: {.cols .vc}
::: {.col}
- **Hands-on systems focus:** Every concept pairs directly with executable framework code for shapes, dtypes, memory aliasing, and gradients.
- **Must-read sections:** Even if you know NumPy and calculus, study **the autograd section** for DAG semantics (`detach`, in-place hazards) and **the probability section** for finite-sample standard errors.
- **Deep theoretical proofs:** Jump to **Chapters 33–36 (Math Appendix)** for SVD, matrix calculus identities, and MLE asymptotics.
:::

::: {.col .fig .big}
@fig:prelim-study-roadmap
:::
:::
:::
