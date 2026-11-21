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
