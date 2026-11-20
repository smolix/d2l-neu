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
