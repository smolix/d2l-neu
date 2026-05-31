# Dynamics: Differential Equations and Generative Flows
:label:`chap_mdl-dynamics`

Many of the most powerful generative models are, at heart, differential
equations. This capstone chapter develops the dynamics needed to read them:
ordinary differential equations and their solvers (and the neural-ODE view of a
residual network), stochastic differential equations and the Itô calculus, the
Fokker--Planck equation and the probability-flow ODE that make diffusion
reversible, and finally score matching, diffusion, and flow matching---unified
as different choices of probability path, training objective, and sampler. It
draws on calculus (:numref:`chap_mdl-calculus`), probability
(:numref:`chap_mdl-probability-statistics`), optimization
(:numref:`chap_mdl-optimization`), and divergences
(:numref:`chap_mdl-information-theory`).

```toc
:maxdepth: 2

mdl-odes-solvers
mdl-sdes
mdl-fokker-planck-probability-flow
mdl-score-matching-diffusion-flow
```
