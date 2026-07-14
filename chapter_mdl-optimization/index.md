# Optimization
:label:`chap_mdl-optimization`

Gradients tell us which way is downhill; optimization is the study of how to
actually get to the bottom. This chapter explains *why* gradient descent and its
accelerated, stochastic, and second-order variants work, *what* the adaptive
per-coordinate methods that actually train today's networks --- AdaGrad, Adam,
AdamW --- add to them and where their guarantees break, *when* convexity
guarantees they find the global optimum, *how* constraints and their multipliers
reshape a problem through Lagrangian duality, and *which* numerical pitfalls turn
a correct algorithm into a `NaN`. The main book's optimization chapter is the
reader's practical first encounter: it should explain how to choose, configure,
and diagnose optimizers in training. This appendix is its mathematical
companion. It derives representative guarantees and failure modes, but is not
intended to duplicate the main chapter's workflow-oriented treatment.

```toc
:maxdepth: 2

mdl-gradient-based-optimization
mdl-adaptive-stochastic-methods
mdl-convexity
mdl-constrained-optimization-duality
mdl-numerical-stability-conditioning
```

## Resources and Further Reading {.unnumbered}

A short, opinionated reading list for going deeper on optimization as it is used
in machine learning: convexity, gradient and second-order methods, duality, and
numerics.

**Books**

- [Convex Optimization — Boyd & Vandenberghe](https://web.stanford.edu/~boyd/cvxbook/) — the standard reference on convex modeling, duality, and interior-point methods; the full PDF is free and the [direct download](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf) is kept available by the authors.
- [Numerical Optimization — Nocedal & Wright (2nd ed., Springer)](https://doi.org/10.1007/978-0-387-40065-5) — the definitive treatment of the algorithms (line search, trust region, quasi-Newton, conjugate gradient, interior-point) behind continuous optimization.
- [Lectures on Convex Optimization — Nesterov (2nd ed., Springer)](https://doi.org/10.1007/978-3-319-91578-4) — the rigorous source for complexity bounds and accelerated first-order methods, by the inventor of acceleration.
- [Optimization for Data Analysis — Wright & Recht (Cambridge, 2022)](https://www.cambridge.org/core/books/optimization-for-data-analysis/C02C3708905D236AA354D1CE1739A6A2) — a compact, modern account aimed squarely at data science: gradient, accelerated, stochastic, and coordinate methods.
- [Mathematics for Machine Learning — Deisenroth, Faisal & Ong](https://mml-book.github.io/) — the free continuous-optimization chapter is an excellent, self-contained bridge from calculus to gradient descent and constrained optimization.

**Courses and video lectures**

- [Stanford EE364a: Convex Optimization I — Boyd](https://web.stanford.edu/class/ee364a/) — slides, homework, and software companion to the Boyd & Vandenberghe text; the [2023 lecture videos](https://www.youtube.com/playlist?list=PLoROMvodv4rMJqxxviPa4AmDClvcbHi6h) are on YouTube.
- [CMU 10-725: Convex Optimization — Tibshirani](https://www.stat.cmu.edu/~ryantibs/convexopt/) — ML-oriented slides and scribed notes covering subgradients, proximal and stochastic methods, duality, and ADMM; matching [Fall 2016 lecture videos](https://www.youtube.com/playlist?list=PLjbUi5mgii6AVdvImLB9-Hako68p9MpIC).

**Tutorials, notes, and surveys**

- [Optimization Methods for Large-Scale Machine Learning — Bottou, Curtis & Nocedal](https://arxiv.org/abs/1606.04838) — the survey to read on why stochastic gradient methods dominate large-scale training, with a clean theory and practical analysis.
- [An overview of gradient descent optimization algorithms — Ruder](https://www.ruder.io/optimizing-gradient-descent/) — a readable tour of momentum, Nesterov, Adagrad, RMSprop, and Adam and how they relate.
- [Why Momentum Really Works — Goh (Distill)](https://distill.pub/2017/momentum/) — an interactive deep dive that explains momentum and acceleration through the convex-quadratic model.
- [Optimization (Chapter 5), Patterns, Predictions, and Actions — Hardt & Recht](https://mlstory.org/optimization.html) — a concise, free chapter framing optimization specifically as the engine of supervised learning.
