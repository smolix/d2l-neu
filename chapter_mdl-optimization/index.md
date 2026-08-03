# Optimization
:label:`chap_mdl-optimization`

The main optimization chapter introduces the update rules used to train models
and discusses how to choose and diagnose them. This appendix supplies the
mathematical analysis behind those rules. It asks when a local descent direction
produces a finite decrease, how stochastic and coordinatewise scaling alter the
guarantee, when convexity upgrades stationarity to global optimality, how
constraints change first-order conditions, and when finite-precision arithmetic
invalidates an otherwise sound algorithm.

The first three sections form the main theoretical path: gradient methods,
stochastic and adaptive variants, and convexity. The constrained-optimization
section then develops multipliers, projections, and duality. The final section
is largely independent and can be read whenever numerical stability becomes
relevant. Throughout, results are stated with the assumptions that make them
valid and are separated from empirical guidance for large neural networks.

```toc
:maxdepth: 2

mdl-gradient-based-optimization
mdl-adaptive-stochastic-methods
mdl-convexity
mdl-constrained-optimization-duality
mdl-numerical-stability-conditioning
```

## Resources and Further Reading {.unnumbered}

The following references cover convexity, first- and second-order methods,
duality, and numerical optimization for machine learning.

**Books**

- [Convex Optimization — Boyd & Vandenberghe](https://web.stanford.edu/~boyd/cvxbook/) — covers convex modeling, duality, and interior-point methods; the authors provide a free [PDF](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf).
- [Numerical Optimization — Nocedal & Wright (2nd ed., Springer)](https://doi.org/10.1007/978-0-387-40065-5) — treats line-search, trust-region, quasi-Newton, conjugate-gradient, and interior-point methods.
- [Lectures on Convex Optimization — Nesterov (2nd ed., Springer)](https://doi.org/10.1007/978-3-319-91578-4) — develops complexity bounds and accelerated first-order methods.
- [Optimization for Data Analysis — Wright & Recht (Cambridge, 2022)](https://www.cambridge.org/core/books/optimization-for-data-analysis/C02C3708905D236AA354D1CE1739A6A2) — presents gradient, accelerated, stochastic, and coordinate methods for data analysis.
- [Mathematics for Machine Learning — Deisenroth, Faisal & Ong](https://mml-book.github.io/) — includes a free, self-contained chapter connecting calculus with gradient descent and constrained optimization.

**Courses and video lectures**

- [Stanford EE364a: Convex Optimization I — Boyd](https://web.stanford.edu/class/ee364a/) — slides, homework, and software companion to the Boyd & Vandenberghe text; the [2023 lecture videos](https://www.youtube.com/playlist?list=PLoROMvodv4rMJqxxviPa4AmDClvcbHi6h) are on YouTube.
- [CMU 10-725: Convex Optimization — Tibshirani](https://www.stat.cmu.edu/~ryantibs/convexopt/) — ML-oriented slides and scribed notes covering subgradients, proximal and stochastic methods, duality, and ADMM; matching [Fall 2016 lecture videos](https://www.youtube.com/playlist?list=PLjbUi5mgii6AVdvImLB9-Hako68p9MpIC).

**Tutorials, notes, and surveys**

- [Optimization Methods for Large-Scale Machine Learning — Bottou, Curtis & Nocedal](https://arxiv.org/abs/1606.04838) — surveys the theory and practice of stochastic-gradient methods for large-scale learning.
- [An overview of gradient descent optimization algorithms — Ruder](https://www.ruder.io/optimizing-gradient-descent/) — reviews momentum, Nesterov acceleration, Adagrad, RMSprop, and Adam.
- [Why Momentum Really Works — Goh (Distill)](https://distill.pub/2017/momentum/) — interactively explains momentum and acceleration through convex quadratic objectives.
- [Optimization (Chapter 5), Patterns, Predictions, and Actions — Hardt & Recht](https://mlstory.org/optimization.html) — gives a concise account of optimization in supervised learning.
