# Optimization
:label:`chap_mdl-optimization`

Gradients tell us which way is downhill; optimization is the study of how to
actually get to the bottom. This chapter explains *why* gradient descent and its
accelerated, stochastic, and second-order variants work, *when* convexity
guarantees they find the global optimum, *how* constraints and their multipliers
reshape a problem through Lagrangian duality, and *which* numerical pitfalls turn
a correct algorithm into a `NaN`. The main book's optimization chapter covers the
practical optimizer zoo; here we develop the foundations that explain it.

```toc
:maxdepth: 2

mdl-gradient-based-optimization
mdl-convexity
mdl-constrained-optimization-duality
mdl-numerical-stability-conditioning
```
