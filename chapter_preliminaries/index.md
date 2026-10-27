# Preliminaries
:label:`chap_preliminaries`

Consider the smallest complete learning problem: load a table of examples,
represent it as numbers, compute a prediction and a loss, find how the loss
changes with each parameter, and decide whether the result is a real pattern or
an accident of the sample. Even a modest problem, such as sorting small
grayscale images of clothing into ten categories, involves many numbers: one
minibatch of 256 images holds 200,704 pixel values, and a linear classifier for
them has almost 8,000 parameters. A Python loop over individual numbers is far
too slow at this scale, and deriving thousands of partial derivatives by hand
is slow and error-prone.

Each step of the problem needs its own tool. :numref:`sec_ndarray` asks how to
store such data and compute with all of it at once.
:numref:`sec_pandas` asks how to turn a table of real records, with missing
entries and text categories, into tensors. :numref:`sec_linear-algebra`
asks how a model turns a minibatch of inputs into predictions, how to
measure the size of vectors and errors, and what repeated multiplication by
a matrix does to a vector. :numref:`sec_calculus` asks how the
loss changes when a parameter changes, and which direction lowers it.
:numref:`sec_autograd` asks how a framework computes these derivatives for
us. :numref:`sec_prob` asks how evidence updates a probability, and when a
measured difference, such as a change in accuracy from one minibatch to the
next, is larger than sampling noise. Finally,
:numref:`sec_lookup_api` asks how to find out what a function does when the
book does not say.

Each section develops only what later chapters need, together with the
intuition behind each result. The Mathematics for Deep Learning appendix treats
linear algebra (:numref:`chap_mdl-linear-algebra`), calculus
(:numref:`chap_mdl-calculus`), and probability
(:numref:`chap_mdl-probability-statistics`) in depth, with proofs; readers who
meet these subjects here for the first time can turn to it at any point.

```toc
:maxdepth: 2

ndarray
pandas
linear-algebra
calculus
autograd
probability
lookup-api
```

## Resources and Further Reading {.unnumbered}

The resources are grouped into books, courses, and documentation, and each
names the section it accompanies. All are freely accessible online.

**Books**

- [Python for Data Analysis, 3rd ed. — Wes McKinney](https://wesmckinney.com/book/) — an open-access guide to NumPy and pandas by the creator of pandas; a companion to :numref:`sec_ndarray` and :numref:`sec_pandas`.
- [Python Data Science Handbook — Jake VanderPlas](https://jakevdp.github.io/PythonDataScienceHandbook/) — an example-driven tour of NumPy, pandas, and the wider scientific-Python stack used throughout the book.
- [Mathematics for Machine Learning — Deisenroth, Faisal & Ong](https://mml-book.github.io/) — a development of linear algebra, calculus, and probability aimed at machine learning, in more depth than :numref:`sec_linear-algebra`, :numref:`sec_calculus`, and :numref:`sec_prob`.
- [Introduction to Probability — Blitzstein & Hwang](http://probabilitybook.net/) — a free PDF of the textbook of Harvard's Stat 110, whose [lecture videos and materials](https://stat110.hsites.harvard.edu/) are also online; a fuller treatment of :numref:`sec_prob`.

**Courses and video lectures**

- [Essence of Linear Algebra — 3Blue1Brown](https://www.3blue1brown.com/topics/linear-algebra) — animated lectures on vectors and matrices as geometric transformations; a visual companion to :numref:`sec_linear-algebra`.
- [Essence of Calculus — 3Blue1Brown](https://www.3blue1brown.com/topics/calculus) — animated lectures on limits, derivatives, and the chain rule; a visual companion to :numref:`sec_calculus` and :numref:`sec_autograd`.
- [NumPy: the absolute basics for beginners — NumPy project](https://numpy.org/doc/stable/user/absolute_beginners.html) — the official introduction to the array operations of :numref:`sec_ndarray`.
- [NumPy Tutorials — NumPy project](https://numpy.org/numpy-tutorials/) — runnable notebooks that apply array programming to concrete data tasks.

**Tutorials, notes, and documentation**

- [Array programming with NumPy — Harris, Millman, van der Walt et al. (Nature, 2020)](https://doi.org/10.1038/s41586-020-2649-2) — the reference paper on the n-dimensional array model that every tensor library in this book follows (preprint at [arXiv:2006.10256](https://arxiv.org/abs/2006.10256)).
- [NumPy user guide — NumPy project](https://numpy.org/doc/stable/user/index.html) — the official reference for array creation, indexing, broadcasting, and data types (:numref:`sec_ndarray`).
- [pandas user guide — pandas project](https://pandas.pydata.org/docs/user_guide/index.html) — the official reference for the loading, indexing, and missing-data handling of :numref:`sec_pandas`.
- [PyTorch documentation — PyTorch project](https://docs.pytorch.org/docs/stable/index.html) — the official API reference for tensor operations and automatic differentiation; :numref:`sec_lookup_api` lists the corresponding references for JAX, TensorFlow, and MXNet and shows how to use them.
- [SciPy documentation — SciPy project](https://docs.scipy.org/doc/scipy/) — routines for linear algebra, optimization, and statistics beyond what the tensor libraries provide.

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
- update a probability with evidence, and decide whether a measured
  difference is larger than sampling noise;
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
| update a belief with evidence; tell a pattern from noise | conditional probability, Bayes' theorem, standard error | Probability and Statistics |
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
| Documentation | what `ones(2, 3)` does, from its signature |
:::
