# Calculus and Automatic Differentiation
:label:`chap_mdl-calculus`

Throughout this book, training a neural network follows the same rhythm:
evaluate a loss, ask the framework for its gradient, and nudge every weight a
small step downhill. We have relied on that loop without pausing to ask two
natural questions about how it works. First, suppose a training run suddenly
diverges after we double the learning rate. The gradient itself is still
correct, yet the step has carried us past the small neighborhood where a local
slope reliably predicts the loss. How far can that linear approximation be
trusted? Second, notice how remarkably cheap the gradient is. Estimating every
slope by perturbing a hundred million weights one at a time would require a
hundred million forward passes, yet a single backward call returns all of them
for roughly the cost of a few forward passes. How is that possible?

Both questions belong to calculus, a subject born in the seventeenth century
when Newton and Leibniz asked how to measure continuous change and how to add
up infinitely many tiny pieces. Their two tools, the derivative and the
integral, turn out to be inverses of each other. In deep learning, the
derivative supplies the best linear model of a loss surface near our current
weights, and aside from the loss value itself, it is the only information an
optimizer ever sees. The integral enters as soon as data or weights are
random: a probability or an expected loss is an area under a curve, and
methods such as policy gradients (:numref:`sec_policygradient`) must
differentiate that integral with respect to the very distribution they sample
from.

Earlier in the book, we stated the basic rules of differentiation
(:numref:`sec_calculus`) and treated automatic differentiation as a ready-made
tool (:numref:`sec_autograd`). The main chapters require nothing more, so this
appendix chapter is optional reading for readers who want to see why the
machinery works and where its guarantees come from. Three results do most of
the heavy lifting: the descent lemma, which converts a bound on curvature into
the safe step-size rule quoted in :numref:`sec_gd`; the chain rule written as
a product of Jacobians, which becomes far cheaper when multiplied from the
loss backward to give backpropagation (:numref:`sec_backprop`); and the two
complementary estimators for the gradient of an expectation, which trade
generality against variance.

The chapter unfolds in three parts. We first build linear and quadratic
approximations of a function, in one variable
(:numref:`sec_mdl-single_variable_calculus`) and then in many
(:numref:`sec_mdl-multivariable_calculus`), so we can bound safe step sizes
and tell a minimum from a saddle point. Next, we follow the chain rule from a
sum over paths in a computation graph (:numref:`sec_mdl-chain-rule-graphs`) to
products of layer Jacobians (:numref:`sec_mdl-matrix-calculus`), and finally
to the algorithms that multiply them automatically
(:numref:`sec_mdl-matrix-calculus-autodiff`). The closing part introduces
integration (:numref:`sec_mdl-integral_calculus`) and brings sampling and
differentiation together to estimate gradients of expectations
(:numref:`sec_mdl-expectation-gradients`).

If you are already comfortable with introductory calculus, feel free to skim
the warm-up pages that open several sections. We omit vector calculus beyond
the gradient (such as divergence and Stokes' theorem) as well as measure
theory; for differential equations, see :numref:`chap_mdl-dynamics`.

```toc
:maxdepth: 2

mdl-single-variable-calculus
mdl-multivariable-calculus
mdl-chain-rule-graphs
mdl-matrix-calculus
mdl-matrix-calculus-autodiff
mdl-integral-calculus
mdl-expectation-gradients
further-reading
```

## Resources and Further Reading {.unnumbered}

The resources below accompany the main sections of this chapter, with each
entry noting the section it supports. Topics beyond what the chapter teaches,
such as automatic differentiation at nonsmooth points, recursive
checkpointing, and the Lebesgue integral, are gathered in
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
- [Calculus on Computational Graphs: Backpropagation — Christopher Olah](https://colah.github.io/posts/2015-08-Backprop/) — a short, diagram-driven post on the chain rule as a sum over paths, the backward sweep of :numref:`sec_mdl-chain-rule-graphs`.
- [matrixcalculus.org — Laue, Mitterreiter & Giesen](https://www.matrixcalculus.org/) — an online symbolic calculator for vector and matrix derivatives, for checking the hand derivations of :numref:`sec_mdl-matrix-calculus`.

**Automatic differentiation and gradient estimation**

- [Automatic Differentiation in Machine Learning: a Survey — Baydin, Pearlmutter, Radul & Siskind](https://www.jmlr.org/papers/volume18/17-468/17-468.pdf) — the standard survey (JMLR, 2018) of the modes and their implementations, and of why automatic differentiation is neither symbolic nor numerical differentiation; it accompanies :numref:`sec_mdl-matrix-calculus-autodiff`.
- [The Autodiff Cookbook — JAX documentation](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html) — `grad`, JVPs, VJPs, Jacobians, and Hessian–vector products with runnable examples, the compositions of :numref:`sec_mdl-matrix-calculus-autodiff`.
- [Autodidax: JAX core from scratch — JAX documentation](https://docs.jax.dev/en/latest/autodidax.html) — builds forward- and reverse-mode autodiff (and `jit`/`vmap`) from basic components, a larger version of the engines built in :numref:`sec_mdl-matrix-calculus-autodiff`.
- [A Gentle Introduction to torch.autograd — PyTorch documentation](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) — how PyTorch records a dynamic graph and replays it in reverse, the tape of :numref:`sec_mdl-matrix-calculus-autodiff`.
- [Monte Carlo Gradient Estimation in Machine Learning — Mohamed, Rosca, Figurnov & Mnih](https://jmlr.org/papers/v21/19-346.html) — a survey (JMLR, 2020) of the score-function and pathwise estimators of :numref:`sec_mdl-expectation-gradients`, with their variances and the conditions under which each applies.
