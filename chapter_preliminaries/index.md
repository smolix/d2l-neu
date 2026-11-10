# Preliminaries
:label:`chap_preliminaries`

The introduction described every learning problem by four components: data,
a model that turns inputs into predictions, a loss that says how badly the
model does, and an algorithm that adjusts the model's parameters to lower the
loss. Now suppose that we had to write the program that sorts photographs of
garments. What would it have to do, and what do we need to know to write it?

It would store the photographs as numbers and compute with all of them at
once. It would combine those numbers into a score for each category, find
out how the loss changes when each parameter changes, and judge whether its
accuracy reflects a real pattern or the luck of the examples it saw.

Probability began in 1654, in letters between Pascal and Fermat about a game
of chance, and calculus is more than three centuries old. A reader who feared
that deep learning needs exotic mathematics may relax. Deep learning borrows
old mathematics and applies it at a scale where calculation by hand is out of
the question, and for that it needs two kinds of software. Array libraries
apply one operation to millions of numbers in compiled code. Automatic
differentiation computes the derivatives of a program. Every later chapter
relies on both.

The chapter takes the program's needs in order. Two sections are about data:
how to store it in tensors and compute with it (:numref:`sec_ndarray`), and
how to get there from a table of real records, with its missing entries and
its text categories (:numref:`sec_pandas`). Three sections follow a
prediction from a weighted sum to a gradient: linear algebra for the
prediction and for the size of its error (:numref:`sec_linear-algebra`),
calculus for the direction in which the loss falls (:numref:`sec_calculus`),
and automatic differentiation, which lets the framework compute the
derivatives for us (:numref:`sec_autograd`). :numref:`sec_prob` is about
doubt: how evidence updates a probability, and when a measured difference is
larger than sampling noise. Finally, :numref:`sec_lookup_api` shows how to
find out what a function does when the book does not say.

This is a long chapter, and a reader at home with NumPy and calculus can skim
much of it. We would still read :numref:`sec_autograd` and the part of
:numref:`sec_prob` on the standard error, since the rest of the book uses
both constantly. Each section develops only what later chapters need. The
Mathematics for Deep Learning appendix treats linear algebra
(:numref:`chap_mdl-linear-algebra`), calculus (:numref:`chap_mdl-calculus`),
and probability (:numref:`chap_mdl-probability-statistics`) in depth, with
proofs; readers new to these subjects can turn to it at any point.

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
