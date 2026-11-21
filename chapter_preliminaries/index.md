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

The tools for the smallest complete learning problem<br>
**tensors · tables · linear algebra · calculus · autograd · probability · documentation**
:::
:::

::: {.slide title="Even a Small Learning Problem Involves Many Numbers"}
[The problem]{.kicker}

- Learning takes six steps: load examples, represent them as numbers,
  predict, measure the loss, change the parameters, and check whether the
  result is real.
- A minibatch of 256 grayscale images of 28 × 28 pixels holds 200,704
  numbers; a linear classifier for ten categories has
  784 × 10 + 10 = 7,850 parameters.
- Python loops over single numbers are too slow at this scale, and
  derivatives by hand are too error-prone.

::: {.d2l-note}
Each step needs one tool, and each section of this chapter supplies one.
:::
:::

::: {.slide title="By the End of This Chapter You Can"}
[Goals]{.kicker}

- store a minibatch of images as a tensor and compute on all of it at once;
- turn a table with gaps and words into tensors;
- write a linear classifier for a minibatch as one matrix product, and
  predict its shape and its cost;
- compute a gradient by hand, and let a framework compute it for you;
- decide whether a measured difference is larger than sampling noise, and
  update a probability with evidence;
- find out what an unfamiliar function does.
:::

::: {.slide title="Map of the Chapter"}
[Chapter map]{.kicker}

| step | tool | section |
|---|---|---|
| store data, compute on all of it at once | tensors | Data Manipulation |
| turn table records into tensors | pandas | Data Preprocessing |
| predict for a whole minibatch; stack layers | matrix products, norms, eigenvalues | Linear Algebra |
| find which way lowers the loss | derivatives, gradients, chain rule | Calculus |
| get derivatives without algebra | automatic differentiation | Automatic Differentiation |
| tell a pattern from noise; update a belief with evidence | standard error, conditional probability, Bayes' theorem | Probability and Statistics |
| find out what a function does | documentation, `help` | Documentation |
:::

::: {.slide title="Every Section Works on a Running Example With Real Numbers"}
[Running examples]{.kicker}

| section | running example |
|---|---|
| Data Manipulation | `images`, a minibatch of shape $(256, 1, 28, 28)$, `uint8` |
| Data Preprocessing | `house_tiny.csv`: 10 homes, 3 + 4 missing entries, roof type as text |
| Linear Algebra | the same minibatch as $\mathbf{X} \in \mathbb{R}^{256 \times 784}$; 7,850 parameters |
| Calculus, Automatic Differentiation | the line model: inputs 0, 1, 2 with labels 1, 1, 0 |
| Probability and Statistics | a coin with 56 heads in 100 tosses; an HIV test |
| Documentation | gamma samples with shape $k = 2$ and rate $3$: does the sampler take a rate or a scale? |
:::
