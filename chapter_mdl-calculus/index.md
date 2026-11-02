# Calculus and Automatic Differentiation
:label:`chap_mdl-calculus`

A network with a hundred million weights computes one number, its loss.
Every training step needs to know how that number changes when all the
weights move a little at once. Moving one weight at a time and running the
network again would take a hundred million runs per step. The derivative
answers the question for every weight at once, and backpropagation computes
it for the cost of a few runs. A second question appears as soon as data or
weights are random. A probability, or an expected loss over a continuous
variable, is an area under a curve, and training through random sampling
needs the derivative of such an area.

The Basics part stated the rules of differentiation and used automatic
differentiation as a tool. It did not show how far the local linear model
can be trusted, what a slope means at the corner of a ReLU, why the backward
order is cheap, or how to compute and differentiate an integral. This
chapter proves these facts and checks them in code. One function recurs
throughout, the log-sum-exp $\log\sum_j e^{x_j}$, whose gradient is the
softmax.

:numref:`sec_mdl-single_variable_calculus` asks how a function of one
variable responds to a small change, how large a step its local model
allows, and what a slope means at a corner.
:numref:`sec_mdl-multivariable_calculus` asks in which direction many weights
should move, how one backward sweep produces that direction, and how second
derivatives tell a minimum from a saddle. :numref:`sec_mdl-matrix-calculus`
asks how to differentiate a layer whose inputs and parameters are vectors and
matrices. :numref:`sec_mdl-matrix-calculus-autodiff` asks how a program
computes these derivatives, and why multiplying from the loss end costs only
a few runs of the network. :numref:`sec_mdl-integral_calculus` asks how to
add up infinitely many thin slices, and how to differentiate an expectation
with respect to a parameter.

We assume school calculus and the Basics part. Vector calculus beyond the
gradient (divergence, Stokes' theorem) and measure theory are left out;
differential equations are the subject of :numref:`chap_mdl-dynamics`.

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
