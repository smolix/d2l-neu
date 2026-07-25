# Mathematics for Deep Learning

This appendix gathers the mathematical results the main text relies on, developed
far enough to be useful and no further. It is organized as six chapters, each
intuition-first and paired with runnable examples. *Linear algebra* covers
vectors, matrices, norms, and the decompositions behind every layer. *Calculus*
develops derivatives, the chain rule, and the multivariable and matrix calculus
that backpropagation automates. *Optimization* treats convexity, gradient
methods, and the convergence and conditioning results the training loop depends
on.

The remaining chapters supply the probabilistic and dynamical foundations:
*probability and statistical learning* (distributions, estimation, and
generalization), *information theory* (entropy, cross-entropy, and the KL
divergence that most losses descend from), and *dynamics* (differential
equations and the continuous-time view that connects to modern normalizing-flow
and diffusion models). Read it start to finish, or dip in when a main-text
chapter cites a result.
