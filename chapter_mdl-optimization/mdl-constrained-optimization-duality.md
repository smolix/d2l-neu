# Constrained Optimization and Duality
:label:`sec_mdl-constrained-optimization-duality`

Constraints and their multipliers are everywhere in deep learning, even when they
are not written down as such: projected and clipped updates, trust regions, the
support-vector machine and attention recast as optimization, and the
regularization-as-constraint equivalence behind weight decay. This is the
Boyd-style section of the chapter. It teaches you to *read* a constrained problem
and its dual — to recognize that a multiplier is a shadow price, that complementary
slackness pins down which constraints are active, and that a hard problem sometimes
becomes easy when viewed from the dual side. We build from Lagrange multipliers for
equalities, to the full Lagrangian and KKT conditions for inequalities, to
Lagrangian duality and the strong-duality guarantee, then ground it all in two
worked examples (the SVM dual and water-filling) and a map of the standard problem
classes.

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
This file is a teaching outline / detailed table of contents, not finished prose.
It fixes the subsection flow, key results, diagrams, worked examples, and draft
exercises. Body framing: foundations — teach the *machinery* of constraints and
duality so the main book's SVM, regularization, and constrained-update material can
reference it. Boyd-style depth is explicitly wanted here.
:::

**Prerequisites:** `sec_mdl-multivariable_calculus` (gradients, tangency, level
sets), `sec_mdl-gradient-based-optimization` (stationarity), `sec_mdl-convexity`
(convex sets/functions — required for strong duality and KKT sufficiency).
Standard reference: :cite:`Boyd.Vandenberghe.2004` (ch. 5); Nocedal & Wright
(*Numerical Optimization*, 2006) for the numerical view — cited descriptively as
no bib key exists.

::: {.callout-note title="⟢ 3.3.1 Equality Constraints and Lagrange Multipliers"}
**Outline:** 1. The geometric heart: at a constrained optimum the objective's
gradient is *parallel* to the constraint's gradient — you cannot move along the
constraint surface and still decrease $f$. · 2. Introduce the multiplier $\nu$ as
the proportionality constant. · 3. The Lagrangian $\mathcal{L}=f+\nu g$ packages
this: its stationarity recovers both the parallel-gradient condition and the
constraint itself. · 4. Multiple equality constraints $\Rightarrow$ a multiplier per
constraint.
**Key results to state:** at the optimum $\nabla f(\mathbf{x}^\star)=-\nu\,\nabla g(\mathbf{x}^\star)$ (gradients parallel); $\mathcal{L}(\mathbf{x},\nu)=f(\mathbf{x})+\nu\,g(\mathbf{x})$; $\nabla_{\mathbf{x}}\mathcal{L}=0,\ \nabla_\nu\mathcal{L}=g(\mathbf{x})=0$.
**Diagrams:** `fig_mdl-lagrange-tangency` — objective level curves tangent to a single constraint curve $g(\mathbf{x})=0$, with $\nabla f \parallel \nabla g$ drawn at the tangency point.
**Worked example(s):** minimize $\mathbf{x}^\top\mathbf{x}$ subject to $\mathbf{a}^\top\mathbf{x}=b$ (closest point on a hyperplane) — recover the projection formula from `sec_mdl-geometry-linear-algebraic-ops`.
**Exercises (draft):** (1) Derive $\nabla f=\nu\nabla g$ from "no feasible descent direction." (2) Maximize entropy $-\sum p_i\log p_i$ subject to $\sum p_i=1$ and recover the uniform distribution. (3) Find the rectangle of maximum area with fixed perimeter via a multiplier.
:::

::: {.callout-note title="⟢ 3.3.2 Inequality Constraints and the Lagrangian"}
**Outline:** 1. Inequalities $g_i(\mathbf{x})\le0$ partition into *active* (tight) and
*inactive* (slack) at the optimum. · 2. The full Lagrangian with multipliers
$\lambda_i\ge0$ for inequalities and $\nu_j$ for equalities. · 3. The sign
constraint $\lambda_i\ge0$ encodes "the constraint can only push one way." · 4.
Multipliers as **shadow prices**: $\lambda_i^\star=-\partial p^\star/\partial b_i$
measures how much the optimum improves if constraint $i$ is relaxed.
**Key results to state:** $\mathcal{L}(\mathbf{x},\boldsymbol{\lambda},\boldsymbol{\nu})=f(\mathbf{x})+\sum_i\lambda_i g_i(\mathbf{x})+\sum_j\nu_j h_j(\mathbf{x})$, $\lambda_i\ge0$; inactive constraint $\Rightarrow$ behaves as unconstrained locally; shadow-price/sensitivity interpretation $\lambda_i^\star = -\partial p^\star/\partial b_i$.
**Diagrams:** reuse `fig_mdl-lagrange-tangency` with an inequality region shaded; introduce `fig_mdl-kkt-active-set` here (optimum on the boundary of active constraints, interior for inactive ones).
**Worked example(s):** minimize $\tfrac12\|\mathbf{x}-\mathbf{x}_0\|_2^2$ subject to $\|\mathbf{x}\|_2\le r$ — projection onto a ball; show the constraint is active iff $\mathbf{x}_0$ is outside the ball, and read $\lambda$ as the shadow price of the radius.
**Exercises (draft):** (1) Show an inactive constraint has $\lambda_i=0$ at the optimum. (2) Interpret $\lambda$ as a shadow price for the ball-projection example. (3) Why must inequality multipliers be nonnegative but equality multipliers free?
:::

::: {.callout-note title="⟢ 3.3.3 The KKT Conditions"}
**Outline:** 1. Assemble the four Karush–Kuhn–Tucker conditions. · 2.
*Complementary slackness* is the workhorse: $\lambda_i g_i=0$ forces, for each
constraint, either $\lambda_i=0$ (inactive) or $g_i=0$ (active) — this is what
"finds the active set." · 3. Status of KKT: **necessary** at any optimum satisfying
a constraint qualification (in general), and **necessary + sufficient** for a global
optimum under convexity + Slater. · 4. KKT as the bridge from "set the gradient to
zero" (unconstrained, §3.1.1) to constrained problems.
**Key results to state:** stationarity $\nabla f+\sum_i\lambda_i\nabla g_i+\sum_j\nu_j\nabla h_j=0$; primal feasibility $g_i\le0,\ h_j=0$; dual feasibility $\lambda_i\ge0$; complementary slackness $\lambda_i g_i(\mathbf{x}^\star)=0$; under convexity + Slater, KKT points are exactly the global optima.
**Diagrams:** `fig_mdl-kkt-active-set` — a feasible polygon with the optimum sitting on two active edges; arrows show $-\nabla f$ as a nonnegative combination of the active-constraint normals (the geometric meaning of stationarity + $\lambda\ge0$), and an inactive edge with $\lambda=0$.
**Worked example(s):** solve a 2-variable QP with one active and one inactive inequality, exhibiting complementary slackness explicitly.
**Exercises (draft):** (1) Write out all four KKT conditions for a given small QP and solve. (2) Show complementary slackness implies the optimum's active set determines the solution. (3) Give a non-convex problem where a KKT point is *not* optimal (necessity without sufficiency).
:::

::: {.callout-note title="⟢ 3.3.4 Lagrangian Duality"}
**Outline:** 1. Define the dual function $g(\boldsymbol{\lambda},\boldsymbol{\nu})=\inf_{\mathbf{x}}\mathcal{L}$. · 2. The
universal fact: the dual function is **always concave** (a pointwise infimum of
affine functions in $(\boldsymbol{\lambda},\boldsymbol{\nu})$), *regardless of whether the primal is
convex* — so the dual problem is always a convex (concave-maximization) problem. ·
3. **Weak duality** always holds: $d^\star\le p^\star$, with gap = "best lower
bound" the dual can certify. · 4. **Strong duality** ($d^\star=p^\star$) holds for
convex problems under **Slater's condition** (a strictly feasible point). · 5.
Body hook: the SVM, the regularization↔constraint equivalence, and many "attention
as optimization" results all live on the dual side — solving the dual *is* solving
the primal when strong duality holds.
**Key results to state:** $g(\boldsymbol{\lambda},\boldsymbol{\nu})=\inf_{\mathbf{x}}\mathcal{L}(\mathbf{x},\boldsymbol{\lambda},\boldsymbol{\nu})$ concave always; weak duality $d^\star=\max_{\boldsymbol{\lambda}\succeq0,\boldsymbol{\nu}} g \le p^\star$; duality gap $p^\star-d^\star\ge0$; Slater $\Rightarrow$ strong duality $d^\star=p^\star$ for convex problems; at strong duality, KKT holds and optimal $\boldsymbol{\lambda}^\star$ are the shadow prices of §3.3.2.
**Diagrams:** `fig_mdl-primal-dual-gap` — the dual function as the lower envelope of $\mathcal{L}$ over $\mathbf{x}$; a number line marking $d^\star \le p^\star$ with the gap shaded (zero gap under convexity + Slater), and a non-convex case with a strict gap.
**Worked example(s):** dual of an equality-constrained quadratic (closed-form, zero gap); a small non-convex example exhibiting a strictly positive duality gap.
**Exercises (draft):** (1) Prove the dual function is concave for *any* primal. (2) Prove weak duality. (3) Exhibit a problem with a nonzero duality gap (Slater fails). (4) Show that under strong duality, $\lambda_i^\star$ equals the constraint sensitivity $-\partial p^\star/\partial b_i$.
:::

::: {.callout-note title="⟢ 3.3.5 Worked Examples: SVM Dual and Water-Filling"}
**Outline:** 1. **Simplex projection warm-up** (tiny QP): minimize
$\tfrac12\|\mathbf{x}-\mathbf{y}\|_2^2$ over the probability simplex; KKT gives the
soft-threshold-and-normalize solution — directly reusable for attention/sparsemax.
· 2. **SVM dual**: start from hard-margin max-margin primal, form the Lagrangian,
eliminate $\mathbf{w},b$ via stationarity, arrive at the dual QP in the multipliers
$\alpha_i$; complementary slackness shows **support vectors = active constraints**
($\alpha_i>0$ only on the margin). · 3. **Water-filling**: maximize
$\sum_i\log(1+P_i/\sigma_i^2)$ under a power budget $\sum_i P_i\le P$; KKT yields the
classic "pour water until levels equalize" closed form — a clean shadow-price story.
· 4. Optional CVXPY check confirming the closed forms.
**Key results to state:** SVM dual $\max_{\boldsymbol{\alpha}\succeq0}\ \sum_i\alpha_i-\tfrac12\sum_{i,j}\alpha_i\alpha_j y_iy_j\,\mathbf{x}_i^\top\mathbf{x}_j$ s.t. $\sum_i\alpha_iy_i=0$, with $\mathbf{w}^\star=\sum_i\alpha_iy_i\mathbf{x}_i$ and support vectors where $\alpha_i>0$; water-filling $P_i^\star=(\mu-\sigma_i^2)_+$ with $\mu$ set by $\sum_i P_i^\star=P$; simplex projection $x_i^\star=(y_i-\tau)_+$ with $\tau$ from $\sum_i x_i^\star=1$.
**Diagrams:** reuse `fig_mdl-kkt-active-set` for the SVM support-vector picture (margin boundary = active constraints); a small water-level bar chart inset (no new SVG).
**Worked example(s):** the three computations above, each verified against `cvxpy`; the SVM dual demonstrated on a 2-D toy with two support vectors.
**Exercises (draft):** (1) Derive the simplex projection via KKT and solve numerically. (2) Show SVM complementary slackness pins support vectors to the margin. (3) Derive water-filling and explain the shadow price $\mu$. (4) Recover the dual from the primal SVM and identify $\mathbf{w}^\star$. Attribute the SVM to Cortes & Vapnik (1995) descriptively (no bib key).
:::

::: {.callout-note title="⟢ 3.3.6 Problem Classes: LP, QP, SOCP, SDP"}
**Outline:** 1. The standard convex hierarchy LP ⊂ QP ⊂ SOCP ⊂ SDP, by objective and
constraint type. · 2. Where DL sub-problems land: $\ell_1$/LASSO and simplex
projection (QP/LP), trust-region steps (QP/SOCP), spectral-norm and PSD constraints
(SDP). · 3. Why the taxonomy matters: it tells you which off-the-shelf solver
applies and what worst-case complexity to expect (all polynomial-time via interior
point). · 4. Note: deep training itself is *outside* this hierarchy (non-convex),
but many of its *sub*-problems and analyses are inside.
**Key results to state:** LP $\min \mathbf{c}^\top\mathbf{x}$ s.t. $A\mathbf{x}\preceq\mathbf{b}$; QP convex iff Hessian $\succeq0$; SOCP cone $\|A\mathbf{x}+\mathbf{b}\|_2\le \mathbf{c}^\top\mathbf{x}+d$; SDP variable $X\succeq0$; inclusion chain LP ⊆ QP ⊆ SOCP ⊆ SDP; all solvable in polynomial time by interior-point methods.
**Diagrams:** none new — a nested-boxes schematic can be rendered inline (described, not a fig).
**Worked example(s):** classify five DL sub-problems (ridge, LASSO, simplex projection, trust-region step, spectral-norm constraint) into the hierarchy and name the solver.
**Exercises (draft):** (1) Show ridge regression is a QP and LASSO is a QP/LP. (2) Express a trust-region subproblem as an SOCP. (3) Show that nuclear-norm minimization is an SDP. (4) Argue why the GD analysis of §3.1 is *not* one of these classes.
:::

## Summary

*Planned.* A constrained optimum has the objective gradient balanced against the
active constraints' gradients; the Lagrangian packages this, with multipliers
$\lambda_i\ge0$ acting as shadow prices. The KKT conditions (stationarity, primal/dual
feasibility, complementary slackness) characterize optima — necessary in general,
sufficient under convexity + Slater. The dual function is always concave, weak
duality $d^\star\le p^\star$ always holds, and strong duality (zero gap) holds for
convex problems satisfying Slater. The SVM dual and water-filling show the machinery
end-to-end; the LP/QP/SOCP/SDP hierarchy tells you which solver fits.

## Exercises

*Planned — consolidated.* (1) Derive $\nabla f=\nu\nabla g$ from no-feasible-descent.
(2) KKT for simplex projection + solve. (3) Dual concave regardless of primal;
exhibit a gap. (4) SVM KKT pins the support vectors.

## Discussions

*Planned placeholder.* Builds on `sec_mdl-convexity` and
`sec_mdl-gradient-based-optimization`; feeds the regularization-as-constraint view
in `sec_mdl-numerical-stability-conditioning` (§3.4.5) and the main book's SVM,
weight-decay, and constrained-update material, which should back-reference §3.3.4 for
the penalty↔constraint equivalence.

<!-- slides -->

*Planned.* Slide deck to be authored once body cells exist, with `@<id>`
placeholders for the Lagrange-tangency derivation, the SVM-dual reduction, and the
water-filling closed form.
