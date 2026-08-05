# chapter_mdl-dynamics — Chapter Overview

Best external match by far: Särkkä & Solin, *Applied Stochastic Differential
Equations* (free Aalto PDF, verified directly — 324 pp., end-of-chapter
problem sets throughout). Its Ch. 2 (ODE background), Ch. 3 (pragmatic SDE
intro), Ch. 4 (Itô calculus), Ch. 5 (Fokker–Planck–Kolmogorov), and Ch. 8
(numerical SDE simulation) map almost one-to-one onto our four sections and
repeatedly pose near-identical problems (OU solve-and-simulate, GBM via log
substitution, stationary-FPK derivation, finite-difference PDE solves). Evans,
*An Introduction to SDEs* (Berkeley lecture notes, verified PDF) contributes a
compact, well-targeted Itô-calculus/martingale exercise set. For the
score-matching/diffusion/flow-matching section, the standout find is MIT
6.S184/6.S975 *Generative AI with SDEs* (Holderrieth, Erives, Shaul; IAP
2025/2026) — its three graded lab notebooks (github.com/eje24/iap-diffusion-labs,
content verified by direct notebook extraction) are a genuine hands-on
problem-set tradition covering exactly our material: Euler/Euler–Maruyama
simulators, Langevin dynamics, conditional-path/score/velocity implementation,
and classifier-free guidance. By contrast, MIT 18.03 psets are gated behind
MITx/Learning-Modules logins except for a handful of public 18.03SC PDFs
(verified); Stanford CS236's homeworks are not publicly posted; Berkeley
CS294-158's public `deepul` homework repo stops at HW4 (GANs), before its
diffusion unit; and Yang Song's score-matching blog post is pure narrative
with zero embedded exercises. Coverage gap worth flagging: Anderson's
time-reversal theorem and the probability-flow-ODE/reverse-SDE λ-family have
no classical course-problem-set tradition at all — even MIT 6.S184 treats them
only in lecture notes, never as a discrete lab exercise. All four sections'
existing exercise sets were already rated excellent by the prior style review
(zero clarity defects across 32 exercises); external material is used here
almost entirely as *additions* folded into existing strong problems, not
replacements. One structural finding across all four sections: despite heavy
code content, every existing exercise is effectively [conceptual]
(pencil-and-paper); none directly touches the sections' own code cells. The
rewrites below fix this by attaching short-code/extended parts sourced from
the external material.

---

## chapter_mdl-dynamics/mdl-odes-solvers.md — Ordinary Differential Equations and Numerical Solvers

**Topic:** Vector fields and flows; Picard–Lindelöf existence/uniqueness; the matrix exponential and eigenvalue-based stability; Euler/Runge–Kutta convergence and stiffness; gradient descent as Euler discretization; Neural ODEs and the (continuous/discrete) adjoint method; continuous normalizing flows and the Hutchinson trace estimator.

**Current exercises:** 8; disposition: keep 7, rewrite 1, drop 0 — the set is
excellent (style review: "clarity: none found... every item ends with an
explicit derivation, numeric check, or named cell to adapt") and is kept
almost entirely intact. The one gap: all 8 are pure pencil-and-paper tasks
even though the section runs 7 code cells (matrix-exponential 3-ways,
Euler/RK4 order sweep, stiffness sweep, GD-as-Euler, Neural-ODE training,
adjoint check, CNF trace/Hutchinson check) — none of the 8 exercises actually
touches that code. The rewrite folds in a short-code part to close this gap;
titles are added throughout to match the naming convention already used in
the other three files of this chapter (fokker-planck and sdes are 8/8
bold-named).

**External sources found:**
- Särkkä & Solin, *Applied SDEs* (2019, free PDF), Ch. 2 Exercise 2.9 — solve
  $\dot x=-x$, $x(0)=1$ analytically, then run Euler at $h=1,10^{-1},\dots,10^{-4}$
  and compare to the exact value — the same order-1 convergence check our
  section already runs on the spiral field, but on the scalar test equation.
  users.aalto.fi/~asolin/sde-book/sde-book.pdf (verified, p. 21).
- Särkkä & Solin, Ch. 2 Exercise 2.10 — implement RK4 and Heun for the
  nonlinear second-order ODE $\ddot x+\dot x-(\alpha-x^2)x=0$ and ask how the
  choice of $\Delta t$ affects the two methods' answers. Same page.
- Särkkä & Solin, Ch. 2 Exercise 2.7 — compute $\exp(Ft)$ for a nilpotent
  $3\times3$ Jordan block two ways (series, Laplace transform). Same page.
- MIT 18.03SC Differential Equations (Fall 2011), OCW, Unit IV "First-order
  Systems," Problem Set 9 Part I, Problem 1 — find a fundamental matrix of
  $\dot{\mathbf x}=\left(\begin{smallmatrix}2&1\\1&2\end{smallmatrix}\right)\mathbf x$
  and use it to solve the IVP $\mathbf x(0)=(3,-2)$ — ocw.mit.edu, PDF
  directly fetched and verified (`MIT18_03SCF11_ps9_s35q.pdf`).
- MIT 6.S184/6.S975 (Holderrieth, Erives, Shaul; IAP 2026), Lab 1 "Simulating
  ODEs and SDEs," Question 1.1 — implement `EulerSimulator`'s `step` method
  for $X_{t+h}=X_t+hu_t(X_t)$ — the same construction our section already
  builds; useful mainly as confirmation that this is the standard first
  exercise of any solver-focused course. github.com/eje24/iap-diffusion-labs
  (notebook `lab_one.ipynb`, verified by direct download and JSON extraction).

**Proposed problem set** (8 problems):
1. [conceptual] **Flows, integral curves, and one explicit integral.** Verify
   by differentiation that $\mathbf{x}(t)=e^{-t}\mathbf{x}_0$ solves
   $\dot{\mathbf{x}}=-\mathbf{x}$ and that the rotational field preserves
   $\|\mathbf{x}(t)\|$; write and evaluate the integral form
   :eqref:`eq_mdl-ode-integral-form` for $\dot x=t$.
   *Provenance:* original.
2. [conceptual] **Lipschitz continuity and its two failure modes.** Show a
   linear field is Lipschitz with constant $\|A\|$; construct a
   non-unique solution family for $\dot x=\sqrt{|x|}$, $x(0)=0$ and locate
   exactly where Lipschitz fails; compute the blow-up time of $\dot
   x=x^2$ and explain why Picard–Lindelöf only promises a local solution.
   *Provenance:* original.
3. **Matrix-exponential algebra and phase portraits.** From the series
   :eqref:`eq_mdl-ode-matrix-exp-series`, show $e^{At}$ commutes with $A$ and
   $e^{A(t+s)}=e^{At}e^{As}$; classify the phase portraits of three given
   $2\times2$ matrices using the stability dictionary.
   *Provenance:* original; the compute-and-classify style parallels MIT
   18.03SC PS9 Problem 1 (overlap low — that problem is a bare
   fundamental-matrix IVP solve, no stability classification).
   [conceptual]
4. [short-code] **Euler and Heun, tested off the linear case.** Derive
   Euler's $O(h^2)$ local / $O(h)$ global error from Taylor's theorem and
   rehearse the accumulation argument; show the midpoint method's update
   matches the true Taylor expansion through $h^2$ in the vector case. Then:
   1. Implement a `heun` function alongside the section's existing `euler`
      and `rk4` (`#odes-solvers-euler-rk4-order`), and run all three on the
      damped pendulum $\ddot\theta=-\sin\theta-\gamma\dot\theta$ (already
      introduced in "Linearization at Fixed Points"), rewritten as a
      first-order system, using a fine-step RK4 solution as ground truth.
   2. Measure convergence slopes on log–log axes as in the text's own cell
      and report whether the theoretical orders 1 (Euler), 2 (Heun), 4 (RK4)
      still hold for this genuinely nonlinear field, not just the book's
      linear spiral test case.

   *Provenance:* parts (a)–(b) original; part (c) adapted from Särkkä &
   Solin Ex. 2.10 (overlap medium — same RK4-vs-Heun comparison idea, but a
   different system, and our success criterion is a measured convergence
   slope rather than a qualitative $\Delta t$-sensitivity comment).
5. [conceptual] **Stability thresholds, stiffness, and implicit steps.**
   Re-derive the forward-Euler stability bound $h<2/\lambda$ and backward
   Euler's unconditional stability; for $A=\mathrm{diag}(-100,-1)$ compute
   the forward-Euler stability step, the steps needed to reach $T=5$, and how
   both change if the fast eigenvalue moves to $-10^4$; write the equation
   one backward-Euler step must solve for nonlinear $\mathbf f$ and the
   Newton iteration used to solve it.
   *Provenance:* original.
6. [conceptual] **Residual blocks as Euler steps.** Identify the implied
   step size of a residual block; explain what halving the solver step while
   doubling the block count means architecturally; use Picard–Lindelöf to
   argue the time-$T$ flow of a Lipschitz $\mathbf f_\theta$ is invertible;
   explain why a plain (non-residual) layer has no such guarantee.
   *Provenance:* original.
7. [conceptual] **The adjoint method, unassisted.** Re-derive the adjoint
   equations :eqref:`eq_mdl-ode-adjoint`–:eqref:`eq_mdl-ode-adjoint-grad`
   from the variational argument without looking; show one Euler step of
   the adjoint ODE is a vector–Jacobian product that reproduces the
   `#odes-solvers-adjoint-check` backprop recursion; explain what breaks
   about reconstructing $\mathbf x(t)$ backward when the forward dynamics
   contract strongly.
   *Provenance:* original.
8. [conceptual] **Instantaneous change of variables and Hutchinson's
   estimator.** Derive the instantaneous change-of-variables formula as
   $h\to0$ and check it against $\Phi_t=e^{At}$; prove the Hutchinson
   estimator is unbiased, compute its Rademacher-probe variance in terms of
   $M+M^\top$'s off-diagonal entries, and explain why trace estimation costs
   one VJP while determinant estimation has no comparable trick.
   *Provenance:* original.

---

## chapter_mdl-dynamics/mdl-sdes.md — Stochastic Differential Equations

**Topic:** Brownian motion and quadratic variation; the Itô integral, isometry, and Itô's lemma; the SDE/Euler–Maruyama definition and strong/weak convergence orders; the Ornstein–Uhlenbeck process, its stationary law, and the variance-preserving normalization.

**Current exercises:** 8; disposition: keep 7, rewrite 1, drop 0 — this is the
strongest-named set in the group (8/8 bold titles, style review found no
clarity issues at all); kept almost verbatim. Two of the kept items turn out
to already be, almost verbatim, textbook problems from two different books —
worth citing on adoption even with zero rewrite. The rewrite adds a Milstein
extension the section's own text sets up but never asks the reader to build.

**External sources found:**
- Evans, *An Introduction to SDEs* (Berkeley Math 195 notes, PDF verified via
  direct extraction, cmor-faculty.rice.edu/~cox/stoch/SDE.course.pdf),
  Exercises 1–2 — show $Y(t)=e^{W(t)-t/2}$ solves $dY=Y\,dW$, and that
  $P(t)=p_0e^{\sigma W(t)+(\mu-\sigma^2/2)t}$ solves $dP=\mu P\,dt+\sigma P\,dW$
  — the identical GBM-via-Itô computation our Exercise 5 already poses.
- Evans, Exercise 38 — for the Langevin/OU equation with
  $X_0\sim\mathcal N(0,\sigma^2/2b)$, show
  $\mathbb E[X(s)X(t)]=(\sigma^2/2b)e^{-b|t-s|}$ — exactly the stationary
  covariance-decay result our Exercise 7 already asks for, under different
  notation ($b\leftrightarrow\theta$).
- Särkkä & Solin, Ch. 3 Exercises 3.2–3.3 — solve the scalar OU SDE for
  $m(t)$, $P(t)$; take the $t\to\infty$ limit two ways; then simulate 1,000
  Euler–Maruyama trajectories and check the mean/covariance match. Same
  structure as our Exercise 7's "re-derive... then check numerically"
  pattern. Verified PDF, pp. 40–41.
- Särkkä & Solin, Ch. 4 Exercises 4.1(b), 4.2, 4.3 — Itô-differential of
  $x^2$; verify $x(t)=\exp(\beta(t))$ solves a specific SDE; solve
  Black–Scholes via $y=\log x$ and compare mean/variance to Euler–Maruyama.
  Verified PDF, pp. 57–58.
- Särkkä & Solin, Ch. 8 Exercise 8.1 — verify
  $x(t)=x_0\exp((c-g^2/2)t+g\beta(t))$ solves $dx=-cx\,dt+gx\,d\beta$, then
  simulate with the **Milstein** method and compare histograms to the exact
  solution at $t=1$. Verified PDF, p. 161.
- MIT 6.S184/6.S975, Lab 1, Question 2.1–2.2 — implement `BrownianMotion`
  and `OUProcess` drift/diffusion methods, then answer, in two qualitative
  sentences, what happens to the OU trajectory as $\theta$ or $\sigma$ go up
  or down (hint: watch $D=\sigma^2/2\theta$) — a lighter-weight,
  intuition-building companion to our Exercise 7. Verified via notebook
  extraction, `lab_one.ipynb`.

**Proposed problem set** (8 problems):
1. **The square-root scaling is forced.** For the random walk with step
   duration $\Delta t$ and step size $c\,(\Delta t)^\gamma$, compute
   $\operatorname{Var}(W_t^{(\Delta t)})$ and show the $\Delta t\to0$ limit
   is $0$, $\infty$, or $c^2t$ depending on $\gamma\gtrless\tfrac12$;
   conclude Brownian motion is the only nontrivial scaling limit.
   *Provenance:* original.
   [conceptual]
2. **No velocity, and why the integral form is the fix.** Show
   $\mathbb P(|\Delta W/\Delta t|>M)\to1$ as $\Delta t\to0$ for every fixed
   $M$; explain why this rules out a pathwise $\dot X=f+g\dot W$ reading and
   how the integral formulation sidesteps it.
   *Provenance:* original.
   [conceptual]
3. [short-code] **Quadratic variation, by hand and on a smooth path.**
   Re-derive $\mathbb E[Q_n]=t$ and $\operatorname{Var}(Q_n)\le2\delta t$;
   explain why the same computation gives $Q_n\to0$ for a $C^1$ path; verify
   both numerically by adapting `#sdes-quadratic-variation` to
   $x(t)=W_1\cdot t$ (random slope, smooth in $t$).
   *Provenance:* original.
4. **Itô's lemma, by hand.** Apply Itô's lemma to $\phi(x)=x^2$ for general
   $dX=f\,dt+g\,dW$ and name the term ordinary calculus misses; specialize to
   $f=0,g=1$ to confirm $\int_0^t W\,dW=\tfrac12(W_t^2-t)$ and check this
   integral has zero mean.
   *Provenance:* original; the specialization is the same computation as
   Evans Ex. 30 ($I(t)=W^2(t)-t$ is a martingale) under a different framing
   (overlap medium — martingale property vs. direct mean check).
   [conceptual]
5. **Geometric Brownian motion and the $-\sigma^2/2$ correction.** Apply
   Itô's lemma to $\phi(x)=\log x$ for $dX=\mu X\,dt+\sigma X\,dW$ to derive
   $X_t=X_0\exp((\mu-\sigma^2/2)t+\sigma W_t)$; locate the $-\sigma^2/2$ term
   in the Taylor expansion; show $\mathbb E[X_t]=X_0e^{\mu t}$ nonetheless,
   and explain how the typical path can grow slower than the mean.
   *Provenance:* adapted from Evans Exercises 1–2 and Särkkä & Solin Ex. 4.3
   (overlap high — both pose exactly this GBM-via-log-substitution problem;
   cite both on adoption).
   [conceptual]
6. [short-code] **Euler–Maruyama, its orders, and one upgrade.**
   1. Show the EM update reduces to forward Euler as $g\to0$; explain what
      breaks if the noise scaled by $\Delta t$ instead of $\sqrt{\Delta t}$.
   2. Identify the Milstein term $\tfrac12g\,\partial_xg\,((\Delta
      W)^2-\Delta t)$ for additive noise and explain why it vanishes,
      giving OU strong order 1.
   3. Predict and verify (by adapting `#sdes-em-strong-order`) the
      strong-order slope of the VP-SDE with $\beta(t)=1+t$.
   4. Implement the Milstein correction for geometric Brownian motion
      (multiplicative noise, where the correction does *not* vanish) and
      confirm it restores strong order 1 against EM's measured order
      $\approx0.5$ on the same $\Delta t$ grid used in part (c).

   *Provenance:* parts (a)–(c) original; part (d) adapted from Särkkä &
   Solin Ex. 8.1 (overlap low-medium — same Milstein-vs-EM idea, but our
   check is a measured convergence slope against the section's own
   log-log framework rather than a histogram match at one $t$).
7. **The OU process, end to end.** Re-derive the solution and transition
   kernel from Itô's lemma and the isometry without looking; show that in
   the stationary regime $\operatorname{Cov}(X_s,X_t)=(\sigma^2/2\theta)
   e^{-\theta|t-s|}$; check numerically with the `#sdes-ou-cloud` ensemble.
   *Provenance:* the solve-and-simulate half is adapted from Särkkä & Solin
   Ex. 3.2–3.3, and the stationary-covariance half from Evans Ex. 38
   (overlap high for both — cite on adoption).
   [conceptual]
8. **Variance preservation, exactly.** From the VP marginal, show
   $\operatorname{Var}(X_t)=\bar\alpha_t\operatorname{Var}(X_0)+(1-\bar\alpha_t)$
   is identically 1 iff $\operatorname{Var}(X_0)=1$; for Rademacher data
   derive $\mathbb E[X_t^4]=3-2\bar\alpha_t^2$ as printed by
   `#sdes-vp-normalization`.
   *Provenance:* original.
   [conceptual]

---

## chapter_mdl-dynamics/mdl-fokker-planck-probability-flow.md — The Fokker–Planck Equation and Probability Flow

**Topic:** From SDE paths to the deterministic Fokker–Planck PDE for
$p_t$; rewriting diffusion as transport (the continuity equation) to get the
probability-flow ODE; the score function; Anderson's time-reversal theorem
and the $\lambda$-family of reverse processes.

**Current exercises:** 8; disposition: keep 6, rewrite 2, drop 0 — the
strongest-reviewed set in the whole MDL group (style review: "clarity: none
found" — every item closes with an explicit derivation or check). Rewrites
add a Langevin/score corollary to Ex. 1 and a numerical-PDE cross-check to
Ex. 2, both sourced externally; both were previously pure closed-book
derivations.

**External sources found:**
- Särkkä & Solin, Ch. 5 Exercise 5.5 — "Stationary FPK equation: show
  Equation (5.26) solves the corresponding stationary FPK" — the same
  set-$\partial_tp=0$-and-solve idea our Exercise 1 already runs on the OU
  case, stated there for a general gradient-drift SDE. Verified PDF, p. 74.
- Särkkä & Solin, Ch. 5 Exercise 5.2 — for the (nonlinear-drift) Beneš SDE
  $dx=\tanh(x)\,dt+d\beta$, verify a given non-Gaussian closed-form density
  solves its FPK, plot its time evolution, then simulate 1,000
  Euler–Maruyama trajectories and check the histogram matches at $t=5$ — a
  genuinely nonlinear analogue of our section's OU-only FPK check. Same PDF,
  p. 73.
- Särkkä & Solin, Ch. 5 Exercise 5.3 — discretize the (Beneš) FPK equation
  in $x$ into a matrix ODE $d\mathbf p/dt=F\mathbf p$ via centered finite
  differences, then solve it three ways: backward Euler, numerical
  $\exp(Ft)$, forward Euler — directly reusing the matrix-exponential and
  implicit/explicit Euler machinery of `sec_mdl-odes-solvers`. Same PDF, pp.
  73–74.
- Särkkä & Solin, Ch. 5 Exercise 5.6 — construct an SDE of Langevin form
  $dx=\tfrac12\nabla\log\pi(x)\,dt+d\beta$ whose stationary law is a
  prescribed target $\pi$ (a Gamma density), then simulate and check
  convergence. Same PDF, p. 74.
- MIT 6.S184/6.S975, Lab 1, Question 3.2 "Ornstein–Uhlenbeck as Langevin
  Dynamics" — show that for $p(x)=\mathcal N(0,\sigma^2/2\theta)$, the score
  is $-2\theta x/\sigma^2$, hence the general Langevin SDE
  $dX=\tfrac12\sigma^2\nabla\log p(X)\,dt+\sigma\,dW$ is *exactly* the OU
  SDE — the precise special case of Särkkä & Solin Ex. 5.6, framed around
  the same OU process our section already treats as its running example.
  Verified via notebook extraction, `lab_one.ipynb`.
- Evans, *Intro to SDEs*, Exercise 36 — for $u$ solving the backward
  diffusion equation $u_t+\tfrac12u_{xx}=0$, show $\mathbb
  E[u(W(t),t)]=u(0,0)$ — a genuine finding of *no* overlap: our section
  builds only the forward FPK/continuity-equation route, never the backward
  (Kolmogorov) equation, so this is inspired-by-only, overlap low, and not
  adopted (would require introducing a tool the section doesn't build).

**Proposed problem set** (8 problems):
1. **Stationary distribution from scratch.** Set $\partial_tp=0$ in the OU
   FPK equation, integrate once in $x$, argue the constant of integration
   (the probability current) must vanish by decay at infinity, solve the
   resulting first-order ODE, and confirm
   $p_\infty=\mathcal N(0,\sigma^2/2\theta)$ with no Gaussian ansatz. Then:
   1. Using the score $\nabla\log p_\infty(x)=-2\theta x/\sigma^2$ you just
      found, show the Langevin SDE
      $dX=\tfrac12\sigma^2\nabla\log p_\infty(X)\,dt+\sigma\,dW$ is
      identical to the original OU SDE — i.e., the OU process is its own
      Langevin sampler.

   *Provenance:* part (a) original; part (b) adapted from MIT 6.S184 Lab 1
   Question 3.2 (overlap high — same identity, same OU process; cite on
   adoption) and generalized by Särkkä & Solin Ex. 5.6's Langevin-SDE
   recipe (overlap low, general case not required here).
   [conceptual]
2. [extended] **Heat kernel, twice over.** Use the moment-ODE proposition
   with $\theta=0$ to show $\mathcal N(x_0,v_0+\sigma^2t)$ solves the heat
   equation, and conclude the marginal law of Brownian motion from $x_0$;
   discuss the $v_0\to0$ limit. Then:
   1. Discretize the heat equation $\partial_tp=\tfrac12\sigma^2\partial_{xx}p$
      on a grid via centered differences into $d\mathbf p/dt=F\mathbf p$
      (Dirichlet boundary $p(\pm L,t)=0$).
   2. Solve this matrix ODE three ways — forward Euler, backward Euler, and
      numerical $\exp(Ft)$ (reusing `sec_mdl-euler-runge-kutta`'s tools) —
      and compare all three against the closed-form Gaussian from part (a)
      at a few times, reporting where forward Euler needs a step-size
      restriction that the other two do not.

   *Provenance:* part (a) original; parts (b)–(c) adapted from Särkkä &
   Solin Ex. 5.3 (overlap high — same three-solver FPK finite-difference
   recipe, specialized here to the heat equation instead of the Beneš SDE,
   which keeps the problem within this book's own prerequisite tools).
3. **Continuity from conservation.** Re-derive the continuity equation from
   the divergence theorem without looking; derive the along-trajectory rule
   $\tfrac{d}{dt}\log q_t(\mathbf x(t))=-\nabla\cdot\mathbf v$ and reconcile
   it with the instantaneous change-of-variables formula of
   `sec_mdl-continuous-normalizing-flows`.
   *Provenance:* original.
   [conceptual]
4. **The sign, in $d$ dimensions.** Prove
   $\tfrac12\nabla\cdot(g^2\nabla p)=\nabla\cdot(p\cdot\tfrac12g^2\nabla\log
   p)$ for $\mathbf x\in\mathbb R^d$, stating exactly where $p>0$ is used;
   explain in one sentence each why the right side carries a plus sign while
   the probability-flow velocity carries a minus.
   *Provenance:* original.
   [conceptual]
5. **The Gaussian flow is affine.** For OU from a point mass, show the
   probability-flow velocity is affine in $x$, solve the PF-ODE in closed
   form, verify it maps $\mathcal N(m_s,v_s)$ to $\mathcal N(m_t,v_t)$, and
   explain why two of its trajectories can never cross while two SDE paths
   can.
   *Provenance:* original.
   [conceptual]
6. **The factor of two.** Re-derive the reverse-family drift
   :eqref:`eq_mdl-dyn-reverse-family` from Fokker–Planck bookkeeping;
   explain in words why the reverse SDE needs $g^2\nabla\log p_t$ where the
   PF-ODE needs only half that, and what the dial $\lambda$ trades off.
   *Provenance:* original.
   [conceptual]
7. **Mixture scores.** Derive the responsibility-weighted score formula for
   $K$ components, specialize to the symmetric two-component case to get the
   $\tanh$ closed form, find all its zeros for $\mu\gg s_0$, and classify
   each as a mode or a repeller of $\dot x=s(x)$.
   *Provenance:* original.
   [conceptual]
8. [short-code] **Break the sampler.** Rerun the reverse-SDE experiment with
   two modifications, one at a time: an asymmetric prior
   $\pi=(0.25,0.75)$ (predict the recovered mass split before running); and
   the *wrong* drift correction $\tfrac12g^2\nabla\log p_t$ with the noise
   left unchanged — describe and explain the recovered density using the
   $\lambda$-family with mismatched drift and noise.
   *Provenance:* original.

---

## chapter_mdl-dynamics/mdl-score-matching-diffusion-flow.md — Score Matching, Diffusion, and Flow Matching

**Topic:** Estimating scores (explicit/implicit/denoising score matching,
Tweedie's formula) and velocities (conditional flow matching) by regression
to a conditional mean; DDPM as a discretized VP-SDE; Langevin dynamics, DDIM,
classifier(-free) guidance; rectified flow, reflow, and the
Benamou–Brenier optimal-transport connection; numerical sampling of the
learned dynamics.

**Current exercises:** 8; disposition: keep 6, rewrite 2, drop 0 — this is
the group's other flawlessly-reviewed set. All 8 are, however, pure
derivations; despite the section running ~10 code cells (DSM training, DDPM
marginal check, Langevin sampler, DDIM strides, CFG sampler, flow-matching
training, reflow, energy-distance sweep, Euler-vs-Heun), none of the 8
touches any of it. This is the strongest hands-on external tradition found in
the whole chapter (MIT 6.S184/6.S975's Labs 1–3 implement almost exactly this
section's pipeline), so two rewrites fold in implementation/verification
parts sourced directly from it.

**External sources found:**
- MIT 6.S184/6.S975 (Holderrieth, Erives, Shaul; IAP 2025/2026), Lab 2 "Flow
  Matching and Score Matching," Problems 2.2–2.4 — implement
  `sample_conditional_path`, `conditional_vector_field`, and
  `conditional_score` for a Gaussian conditional probability path, checked
  against a reference figure — the same conditional-score derivation as our
  Exercise 1, but with a coding/verification step attached.
  github.com/eje24/iap-diffusion-labs, `lab_two.ipynb` (verified by direct
  notebook download and JSON extraction).
- MIT 6.S184/6.S975, Lab 2, Question 3.3 "Deriving the Marginal Score from
  the Marginal Flow" — for the concrete schedule $\alpha_t=t$,
  $\beta_t=\sqrt{1-t}$, derive
  $\tilde s_t^\theta(x)=(u_t^\theta(x)-a_tx)/b_t$ and check it numerically
  against an independently-trained score network — the identical
  score–velocity dictionary derivation our text states as
  :eqref:`eq_mdl-score-velocity` but never turns into an exercise.
  Verified, `lab_two.ipynb`.
- MIT 6.S184/6.S975, Lab 1, Questions 3.1–3.2 — implement Langevin dynamics
  and prove the OU-as-Langevin-dynamics identity — medium overlap with our
  Exercise 6 (Langevin stationarity), which instead proves the general
  $p\propto e^{-E}$ case and its EM discretization bias.
- MIT 6.S184/6.S975, Lab 3 "A Conditional Generative Model for Images," Part
  2 — derive $\tilde u_t(x|y)=u_t(x)+w\,b_t\nabla\log p_t(y|x)$ from Bayes'
  rule and implement the $\eta$-dropout CFG training objective — high
  overlap with our Exercise 7 (CFG as a score tilt), same Bayes-rule
  derivation, framed for velocities instead of scores.
- **No external tradition found:** Yang Song's blog post "Generative
  Modeling by Estimating Gradients of the Data Distribution"
  (yang-song.net/blog/2021/score/, fetched and checked directly) contains
  zero embedded exercises or reader prompts of any kind — pure narrative
  exposition, confirmed by direct reading of the post's structure.
- **No accessible tradition found:** Stanford CS236's homeworks (Week 7/10
  lectures cover exactly this material) are not posted publicly outside the
  enrolled-student LMS; no public PDF or repo located after search.
- **No accessible tradition found:** Berkeley CS294-158's public homework
  repo (github.com/rll/deepul) contains only HW1–HW4 (autoregressive models
  through GANs) — its diffusion/score-matching lecture content has no
  matching public homework directory in any located course-year mirror.

**Proposed problem set** (8 problems):
1. [short-code] **Conditional scores, derived and implemented.** Derive
   $\nabla_{\tilde{\mathbf x}}\log p_\sigma(\tilde{\mathbf
   x}\mid\mathbf x)=(\mathbf x-\tilde{\mathbf x})/\sigma^2$ from the Gaussian
   density and verify it equals $-\boldsymbol\epsilon/\sigma$; derive
   Hyvärinen's identity in one dimension, stating exactly where the boundary
   term vanishes. Then implement the conditional score as a standalone
   function of $(\tilde x,x,\sigma)$ and check it agrees with the section's
   own `mixture_score(q, var, means)` in the one-component limit (a single
   Gaussian, `means=(m,)`), to machine precision.
   *Provenance:* parts (a)–(b) original; the implementation-and-check part
   is adapted from MIT 6.S184 Lab 2 Problems 2.2–2.4 (overlap medium — same
   "implement the conditional score, check numerically" structure, using
   this book's own `mixture_score` as the reference rather than a plotted
   figure).
2. **The regression lemma, twice used.** Prove the regression lemma and use
   it to show marginal and conditional flow matching share minimizers;
   identify exactly where the proof needs $p_t(\mathbf x)>0$.
   *Provenance:* original.
   [conceptual]
3. **Straight paths and one exact Euler step.** From the linear path, derive
   the constant conditional velocity; show that if every trajectory of the
   *learned* field is a straight line at constant speed, one Euler step
   integrates it exactly, and say what the local truncation error reduces to
   along such a trajectory.
   *Provenance:* original.
   [conceptual]
4. **DDPM is reweighted DSM.** Show the DDPM loss equals the
   noise-conditional DSM loss with $\lambda(t)=1-\bar\alpha_t$ and
   $\mathbf s_\theta=-\boldsymbol\epsilon_\theta/\sqrt{1-\bar\alpha_t}$;
   say which noise levels the simple loss emphasizes relative to
   $\lambda(t)=1$ and why that might help perceptual quality.
   *Provenance:* original.
   [conceptual]
5. **Placing variance-exploding SMLD in the table.** Add variance-exploding
   SMLD ($\mathbf x_t=\mathbf x_0+\sigma(t)\boldsymbol\epsilon$) as a new row
   in the unifying table (object learned, training loss, sampler,
   stochastic?), and predict its step-count behavior relative to the VP row.
   *Provenance:* original.
   [conceptual]
6. **Langevin stationarity, exactly and discretely.** Verify $p\propto
   e^{-E}$ is stationary for the Langevin SDE by direct substitution into
   Fokker–Planck; for $E(x)=x^2/(2v)$, compute the discrete chain's
   stationary variance exactly and show its $O(h)$ bias; name the classical
   acceptance step that removes it.
   *Provenance:* original; the SDE-stationarity half overlaps at low level
   with MIT 6.S184 Lab 1 Question 3.1's Langevin implementation (different
   task: our problem proves stationarity and bias, theirs only simulates).
   [conceptual]
7. **CFG as a score tilt.** Substitute the Bayes identity into the CFG field
   to show $\tilde{\mathbf s}=\nabla\log[p_t(\mathbf
   x)p_t(y\mid\mathbf x)^\gamma]$; for a two-component Gaussian mixture with
   equally likely classes, describe what $\gamma>1$ does to the effective
   density, and exhibit a case where the tilted object is not proportional
   to any noised-data marginal.
   *Provenance:* the Bayes-identity derivation overlaps at high level with
   MIT 6.S184 Lab 3 Part 2's CFG derivation for velocities (overlap high for
   the algebra, though that lab works with velocities/vector fields while
   this problem works with scores/densities directly — cite on adoption).
   [conceptual]
8. [short-code] **Stochastic interpolants, generalized and specialized.**
   For $\mathbf x_t=\alpha_t\mathbf x_0+\beta_t\mathbf x_1+\gamma_t\mathbf w$,
   derive the conditional velocity as the CFM target; identify schedule
   choices recovering (a) rectified flow and (b) a variance-preserving
   diffusion path; say what $\gamma_t>0$ does in the interior. Then, for
   schedule (b), verify the score–velocity identity
   :eqref:`eq_mdl-score-velocity` by direct substitution of $\alpha_t,\beta_t$
   and their derivatives, and confirm numerically — by adapting
   `#mdl-score-matching-diffusion-flow-score-noise-and-velocity-are-one-function`
   to your schedule — that the two routes to $u_t$ agree to machine
   precision, as the text's cosine-schedule check did.
   *Provenance:* parts (a)–(b) original; the numerical verification is
   adapted from MIT 6.S184 Lab 2 Question 3.3 (overlap medium — same
   score-from-flow verification technique, different concrete schedule).
