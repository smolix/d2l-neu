# Wave-2 report: chapter_mdl-dynamics/mdl-odes-solvers.md

## Structure (1399 lines; was a 383-line stub)

5 top-level `##` + Summary + Exercises + Discussions tabs (×4) + 7 slides; 13
code fences (imports ×4 tabs, 5 untagged NumPy demos, neural-ODE training ×4
tabs), 11 `:eqlabel:`s. All 10 stub labels survive on matching headings:
`sec_mdl-odes-solvers` (anchor), `sec_mdl-vector-fields-trajectories` (##),
`sec_mdl-ode-existence-uniqueness` (###), `sec_mdl-linear-odes-stability` (##),
`sec_mdl-euler-runge-kutta` (##), `sec_mdl-stiffness-implicit` (###),
`sec_mdl-neural-odes` (##), `sec_mdl-adjoint-method` (###),
`sec_mdl-continuous-normalizing-flows` (##), `fig_mdl-dyn-ode-field` (kept
include, caption verbatim). Propositions w/ proofs: Picard–Lindelöf
(contraction sketch + Picard-iteration-builds-e^t example), matrix exp solves
linear ODEs (termwise differentiation), **e^{At} = V e^{Λt} V⁻¹ (3-line
telescoping proof — the eigendecomposition section's pointer now lands on a
real definition; stub's wrong prereq claim fixed)**, Euler global order 1
(full error-accumulation proof, (1+hL)-recursion → (Mh/2L)(e^{LT}−1)), RK4
order 4 (statement, proof cited to Hairer–Nørsett–Wanner), Euler stable iff
h<2/λ + backward Euler A-stable (2 lines each), ResNet block = Euler step
(1 line), adjoint equations (variational proof via d/dt(aᵀδx) telescoping),
det(I+hJ) = 1+h tr J+O(h²) lemma (Leibniz, 2 lines), instantaneous change of
variables (5 lines), Hutchinson unbiased (2 lines). Counterexamples: √|x|
fan (closed-form blades x_c), ẋ=x² blow-up at t=1/x₀. §6.x pseudo-refs all
purged → 14 distinct `:numref:` targets, all verified to exist; plain-text
citations → `:cite:` keys.

## Figure SPECS (report-only; NOT created — text has no dangling refs)

- `mdl-dyn-phase-portraits`: 2×3 gallery (stable node, unstable node, saddle /
  stable spiral, unstable spiral, center): arrow grid + 2–3 trajectories per
  panel, eigenvalue signature annotated (λ₁,λ₂<0; opposite signs; a±ib …).
  Insert after the stability-dictionary table (§ The Stability Dictionary).
- `mdl-dyn-uniqueness-fan`: solutions of ẋ=√|x| through 0 — x≡0 plus
  x_c(t)=(t−c)²/4 for t>c, c ∈ {0, 0.5, 1, 1.5, 2}; annotate "Lipschitz fails
  at 0". Insert beside the counterexample paragraph (§ Existence and Uniqueness).
- `mdl-dyn-resnet-as-euler`: left, stack of residual blocks with "+f_θ" skip
  annotations; right, smooth trajectory through a vector field with the h=1
  Euler polyline overlaid; bridge label x_{l+1}=x_l+f_θ(x_l) ⇔ Euler, h=1.
  Insert in § Residual Networks Are Euler Steps.
- Error-vs-h log–log plot is a computed `d2l.plot` cell
  (`#odes-solvers-euler-rk4-order`), deliberately not a pre-generated figure.

## BibTeX (NEW keys used; d2l.bib NOT edited — for reconciliation pass)

```bibtex
@inproceedings{Chen.Rubanova.Bettencourt.ea.2018, title={Neural Ordinary Differential Equations},
  author={Chen, Ricky T. Q. and Rubanova, Yulia and Bettencourt, Jesse and Duvenaud, David},
  booktitle={Advances in Neural Information Processing Systems}, volume={31}, year={2018}}
@inproceedings{Grathwohl.Chen.Bettencourt.ea.2018, title={{FFJORD}: Free-Form Continuous Dynamics for Scalable Reversible Generative Models},
  author={Grathwohl, Will and Chen, Ricky T. Q. and Bettencourt, Jesse and Sutskever, Ilya and Duvenaud, David},
  booktitle={International Conference on Learning Representations}, year={2019}}
@article{Hutchinson.1989, title={A Stochastic Estimator of the Trace of the Influence Matrix for {L}aplacian Smoothing Splines},
  author={Hutchinson, Michael F.}, journal={Communications in Statistics---Simulation and Computation},
  volume={18}, number={3}, pages={1059--1076}, year={1989}}
@book{Hairer.Norsett.Wanner.1993, title={Solving Ordinary Differential Equations {I}: Nonstiff Problems},
  author={Hairer, Ernst and N{\o}rsett, Syvert P. and Wanner, Gerhard}, publisher={Springer}, edition={2}, year={1993}}
@book{Pontryagin.Boltyanskii.Gamkrelidze.ea.1962, title={The Mathematical Theory of Optimal Processes},
  author={Pontryagin, Lev S. and Boltyanskii, Vladimir G. and Gamkrelidze, Revaz V. and Mishchenko, Evgenii F.},
  publisher={Interscience}, year={1962}}
```
(Key `Grathwohl…2018` per mandate; entry is the ICLR 2019 paper, arXiv 1810.01367.)

## Verification log (cumulative, in document order, per venv; MPLBACKEND=Agg)

- All 7 cell groups pass under .venv-pytorch (2.8 s total), -mxnet (3.6 s),
  -tensorflow (3.3 s), -jax (4.0 s). Every cell ≪ 60 s CPU.
- e^{At}: series vs eigen 5.6e-17; (I+tA/n)^n (n=2²⁴) vs eigen 5.4e-08;
  ‖e^{At}x₀‖ = e^{−t/2}‖x₀‖ to 6 digits. Order plot: measured slopes 1.040 /
  4.022. Stiffness sweep brackets h=2/λ exactly; diag(−50,−1) explodes in the
  dead mode. Adjoint: discrete vs FD 1.2e-10; discrete→continuous gap
  2.9e-2 / 2.8e-3 / 2.8e-4 at n=100/1000/10000 (clean O(h)). CNF: the two
  log-density rows agree to all 6 printed digits; Hutchinson −1.7554±0.0173
  vs tr = −1.7535. Neural-ODE training (401 full-batch Adam iters, 2-32-2
  tanh field, 10 unrolled Euler steps): final loss <1e-5 in all frameworks;
  mean endpoint error pytorch 0.0011, mxnet 0.0013, tensorflow 0.0017, jax
  0.0021. `tools/lint_source.py` passes (0 errors, 0 warnings).

## Deviations

1. **Length 1399 vs ~850–1000 target.** Structural overhead: ~290 lines of
   code fences (two ×4-tab groups), 88 slide lines, 11 display-equation blocks,
   16 Discussions lines; prose density matches the constrained-optimization
   exemplar (1145 lines with only 3 single-variant cells).
2. **Code B** verifies the adjoint with *hand-written* reverse mode through
   the unrolled solver (the discrete adjoint — framed as "what autograd
   builds, written out") against finite differences and the continuous-adjoint
   quadrature, untagged NumPy — instead of framework autograd, per the
   "untagged numpy for solver demos" rule. The O(h) discrete→continuous table
   is a bonus the autograd version couldn't show.
3. **MXNet did not need to simplify**: full Gluon loop works. Gotcha for
   re-tuners: `init.Normal(0.5)` plateaus at loss 0.04; `init.Xavier()` used.
   JAX: `lax.fori_loop` + jit'd hand-rolled Adam (no optax).
4. **No scipy** anywhere: the e^{At} cell cross-checks three independent
   computations instead, keeping the one-imports-cell rule clean.
5. `:numref:` to `sec_mdl-euler-maruyama` / `sec_mdl-ornstein-uhlenbeck`
   assumes the 27.2 writer preserves those stub labels (their mandate mirrors
   mine; their report confirms both survive).
