# Calculus and Automatic Differentiation
:label:`chap_mdl-calculus`

Almost every model in this book was trained by the same loop: compute a
loss, ask the framework for its gradient, and move every weight a small step
against it. Two parts of that loop went unexplained. Suppose a run diverges
after we double the learning rate. The gradient is still correct, but the
step has left the region where the slope predicts the loss. How far can that
prediction be trusted? The gradient also comes cheaply. Nudging a hundred
million weights one at a time would take a hundred million network runs, yet
one call returns every slope for the cost of a few. How?

Calculus answers these questions, although it was created for others. In the
seventeenth century Newton and Leibniz asked how to describe a quantity that
changes continuously, and how to add up infinitely many small pieces. Their
answers, the derivative and the integral, undo each other. The derivative is
the best linear model of a function near a point. Besides the loss value, it
is all an optimizer knows. The integral enters once data or weights are
random: a probability or an expected loss is an area under a curve, and a
policy-gradient method (:numref:`sec_policygradient`) differentiates one with
respect to the distribution it samples from.

The Basics part stated the rules of differentiation (:numref:`sec_calculus`)
and used automatic differentiation as a tool (:numref:`sec_autograd`); the
main text needs no more. This chapter is additional reading, for the reader
who wants the proofs. Three results matter most: the descent lemma, which
turns a bound on the curvature into the safe step size that :numref:`sec_gd`
quotes; the chain rule as a product of Jacobians, cheapest when multiplied
from the loss backward, which is backpropagation (:numref:`sec_backprop`);
and the two estimators for the gradient of an expectation, neither of which
always wins.

The first part builds the linear and the quadratic model of a function, in
one variable (:numref:`sec_mdl-single_variable_calculus`) and in many
(:numref:`sec_mdl-multivariable_calculus`), which bound a safe step and tell
a minimum from a saddle. The second follows the chain rule from a sum over
the paths of a graph to a product of layer Jacobians
(:numref:`sec_mdl-matrix-calculus`) and to an algorithm
(:numref:`sec_mdl-matrix-calculus-autodiff`). The integral and expectations
come last (:numref:`sec_mdl-integral_calculus`).

The school calculus that opens several sections can be skimmed. Vector
calculus beyond the gradient (divergence, Stokes' theorem) and measure theory
are left out; for differential equations see :numref:`chap_mdl-dynamics`.

```toc
:maxdepth: 2

mdl-single-variable-calculus
mdl-multivariable-calculus
mdl-matrix-calculus
mdl-matrix-calculus-autodiff
mdl-integral-calculus
further-reading
```

## Resources and Further Reading {.unnumbered}

The resources below accompany the sections of this chapter; each names the
section it goes with. Topics that the chapter does not teach, such as
automatic differentiation at nonsmooth points, recursive checkpointing, and
the Lebesgue integral, are collected in
:numref:`sec_mdl-calculus-further-reading`.

**Books**

- [Mathematics for Machine Learning — Deisenroth, Faisal & Ong](https://mml-book.github.io/) — free Cambridge text; its Chapter 5, "Vector Calculus," covers gradients, Jacobians, and the chain rule for machine learning, the material of :numref:`sec_mdl-single_variable_calculus` and :numref:`sec_mdl-multivariable_calculus`.
- [Calculus — Gilbert Strang, MIT OpenCourseWare](https://ocw.mit.edu/courses/res-18-001-calculus-fall-2023/) — a free textbook (3rd edition) whose chapters on integrals, the fundamental theorem, and multiple integrals accompany :numref:`sec_mdl-integral_calculus`.
- [The Matrix Cookbook — Petersen & Pedersen](https://archive.org/details/imm3274) — a reference table of matrix-derivative identities, useful for checking the results that :numref:`sec_mdl-matrix-calculus` derives by the differential method.
- [Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation — Griewank & Walther](https://epubs.siam.org/doi/book/10.1137/1.9780898717761) — the standard monograph (SIAM, 2nd ed.) on forward and reverse mode and checkpointing, the subject of :numref:`sec_mdl-matrix-calculus-autodiff`.
- [The Art of Differentiating Computer Programs — Naumann](https://epubs.siam.org/doi/book/10.1137/1.9781611972078) — an introduction to algorithmic differentiation from the compiler's point of view, with tangent and adjoint code written out by hand; it accompanies :numref:`sec_mdl-matrix-calculus-autodiff`.

**Courses and video lectures**

- [Essence of Calculus — 3Blue1Brown](https://www.3blue1brown.com/topics/calculus) — a twelve-part visual introduction to derivatives, integrals, and the chain rule, for :numref:`sec_mdl-single_variable_calculus` and :numref:`sec_mdl-integral_calculus`.
- [Single Variable Calculus (18.01SC) — MIT OpenCourseWare](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/) — a self-study course with lecture videos and graded problem sets for :numref:`sec_mdl-single_variable_calculus`.
- [Multivariable Calculus (18.02SC) — MIT OpenCourseWare](https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/) — partial derivatives, gradients, Lagrange multipliers, and double integrals, for :numref:`sec_mdl-multivariable_calculus`.
- [Matrix Calculus for Machine Learning and Beyond (18.S096) — Edelman & Johnson, MIT OCW](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/) — derivatives of matrix-valued maps treated as linear maps rather than entry by entry, the view of :numref:`sec_mdl-matrix-calculus`; lecture notes and videos included.

**Tutorials and notes**

- [The Matrix Calculus You Need For Deep Learning — Parr & Howard](https://explained.ai/matrix-calculus/) — a free primer on the index method for the Jacobians of :numref:`sec_mdl-matrix-calculus`, assuming only Calculus 1.
- [Calculus on Computational Graphs: Backpropagation — Christopher Olah](https://colah.github.io/posts/2015-08-Backprop/) — a short, diagram-driven post on the chain rule as a sum over paths, the backward sweep of :numref:`sec_mdl-multivariable_calculus`.
- [matrixcalculus.org — Laue, Mitterreiter & Giesen](https://www.matrixcalculus.org/) — an online symbolic calculator for vector and matrix derivatives, for checking the hand derivations of :numref:`sec_mdl-matrix-calculus`.

**Automatic differentiation and gradient estimation**

- [Automatic Differentiation in Machine Learning: a Survey — Baydin, Pearlmutter, Radul & Siskind](https://www.jmlr.org/papers/volume18/17-468/17-468.pdf) — the standard survey (JMLR, 2018) of the modes and their implementations, and of why automatic differentiation is neither symbolic nor numerical differentiation; it accompanies :numref:`sec_mdl-matrix-calculus-autodiff`.
- [The Autodiff Cookbook — JAX documentation](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html) — `grad`, JVPs, VJPs, Jacobians, and Hessian–vector products with runnable examples, the compositions of :numref:`sec_mdl-matrix-calculus-autodiff`.
- [Autodidax: JAX core from scratch — JAX documentation](https://docs.jax.dev/en/latest/autodidax.html) — builds forward- and reverse-mode autodiff (and `jit`/`vmap`) from basic components, a larger version of the engines built in :numref:`sec_mdl-matrix-calculus-autodiff`.
- [A Gentle Introduction to torch.autograd — PyTorch documentation](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) — how PyTorch records a dynamic graph and replays it in reverse, the tape of :numref:`sec_mdl-matrix-calculus-autodiff`.
- [Monte Carlo Gradient Estimation in Machine Learning — Mohamed, Rosca, Figurnov & Mnih](https://jmlr.org/papers/v21/19-346.html) — a survey (JMLR, 2020) of the score-function and pathwise estimators of :numref:`sec_mdl-integral_calculus`, with their variances and the conditions under which each applies.
