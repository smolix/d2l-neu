# Calculus and Automatic Differentiation
:label:`chap_mdl-calculus`

Training a network means asking, repeatedly, how a scalar loss changes when we
nudge a parameter. This chapter builds the calculus that answers it: derivatives
and the local-linear view in one variable, gradients and the chain rule in many,
the matrix-calculus and automatic-differentiation machinery that makes
backpropagation cheap, and finally integration---the operation we will need for
probability (:numref:`chap_mdl-probability-statistics`) and for the differential
equations of :numref:`chap_mdl-dynamics`.

```toc
:maxdepth: 2

mdl-single-variable-calculus
mdl-multivariable-calculus
mdl-matrix-calculus-autodiff
mdl-integral-calculus
```

## Resources and Further Reading

If you want to go deeper, the following—from refreshers on single-variable calculus to the matrix calculus and automatic-differentiation machinery behind backpropagation—are the canonical references we recommend.

**Books**

- [Mathematics for Machine Learning — Deisenroth, Faisal & Ong](https://mml-book.github.io/) — free Cambridge text; Chapter 5, "Vector Calculus," is the cleanest treatment of gradients, Jacobians, and the chain rule aimed squarely at ML.
- [The Matrix Cookbook — Petersen & Pedersen](https://archive.org/details/imm3274) — the standard quick-reference for matrix-derivative identities; reach for it when you need $\partial(\mathbf{x}^\top \mathbf{A}\mathbf{x})$ and friends without re-deriving them.
- [Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation — Griewank & Walther](https://epubs.siam.org/doi/book/10.1137/1.9780898717761) — the definitive monograph (SIAM, 2nd ed.) on automatic differentiation, including forward/reverse modes and checkpointing.

**Courses and video lectures**

- [Essence of Calculus — 3Blue1Brown](https://www.3blue1brown.com/topics/calculus) — a twelve-part visual series that builds derivatives, integrals, and the chain rule from intuition; the best on-ramp if calculus feels rusty.
- [Single Variable Calculus (18.01SC) — MIT OpenCourseWare](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/) — full self-study course with lecture videos and graded problem sets for the one-variable foundations.
- [Multivariable Calculus (18.02SC) — MIT OpenCourseWare](https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/) — partial derivatives, gradients, and vector calculus, the multivariable companion to 18.01.
- [Matrix Calculus for Machine Learning and Beyond (18.S096) — Edelman & Johnson, MIT OCW](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/) — treats derivatives of matrix-valued maps holistically rather than entry-by-entry; lecture notes and videos included.

**Tutorials, notes, and visual introductions**

- [The Matrix Calculus You Need For Deep Learning — Parr & Howard](https://explained.ai/matrix-calculus/) — a free, self-contained primer that develops exactly the matrix calculus needed to read a backprop derivation, assuming only Calculus 1.
- [Calculus on Computational Graphs: Backpropagation — Christopher Olah](https://colah.github.io/posts/2015-08-Backprop/) — a short, diagram-driven post showing how the chain rule on a computational graph *is* backpropagation.
- [matrixcalculus.org — Laue, Mitterreiter & Giesen](https://www.matrixcalculus.org/) — an online symbolic calculator for vector and matrix derivatives; useful for checking hand-derived gradients and exporting LaTeX or Python.

**Automatic differentiation**

- [Automatic Differentiation in Machine Learning: a Survey — Baydin, Pearlmutter, Radul & Siskind](https://www.jmlr.org/papers/volume18/17-468/17-468.pdf) — the standard survey (JMLR, 2018); read it to understand why AD is neither symbolic nor numerical differentiation.
- [The Art of Differentiating Computer Programs — Naumann](https://epubs.siam.org/doi/book/10.1137/1.9781611972078) — SIAM's introduction to algorithmic differentiation from the compiler's point of view, by the author of the NP-completeness result for optimal Jacobian accumulation cited in this chapter.
- [The Autodiff Cookbook — JAX documentation](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html) — `grad`, JVPs/VJPs, Jacobians, and Hessian-vector products with runnable examples.
- [Autodidax: JAX core from scratch — JAX documentation](https://docs.jax.dev/en/latest/autodidax.html) — builds forward- and reverse-mode autodiff (and `jit`/`vmap`) from the ground up, demystifying what a modern AD system actually does.
- [A Gentle Introduction to torch.autograd — PyTorch documentation](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) — how PyTorch records a dynamic computational graph and replays it in reverse to compute gradients via `.backward()`.
