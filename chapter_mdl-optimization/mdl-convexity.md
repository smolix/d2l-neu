# Convex Sets and Convex Functions
:label:`sec_mdl-convexity`

Convexity is the line between "gradient descent provably finds the answer" and "we
hope." For a convex problem, *every local minimum is global*, there are no saddle
points or spurious basins to get stuck in, and the rates of
`sec_mdl-gradient-based-optimization` come with global guarantees. Convexity tells
you which deep-learning sub-problems are *easy* — the last-layer softmax/logistic
loss in the logits, projections onto constraint sets, $\ell_1/\ell_2$ regularizers,
the SVM dual — and it tells you precisely what you forfeit by stacking nonlinear
layers. This section develops the three equivalent lenses for recognizing a convex
function, the inequality (Jensen's) that powers half of information theory, and an
honest reality check on why deep networks are non-convex and what we keep anyway.

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
This file is a teaching outline / detailed table of contents, not finished prose.
It fixes the subsection flow, key results, diagrams, worked examples, and draft
exercises. Body framing: foundations — *why* convexity gives global guarantees, so
the main book's convex sub-problems (`sec_convexity`, SVM, softmax regression) and
the divergence bounds in `sec_mdl-information_theory` can lean on it. Do not
duplicate the main book's convexity-for-optimization how-to.
:::

**Prerequisites:** `sec_mdl-geometry-linear-algebraic-ops` (inner products,
half-spaces, hyperplanes), `sec_mdl-single_variable_calculus` and
`sec_mdl-multivariable_calculus` (first/second derivatives, Hessian, PSD),
`sec_mdl-gradient-based-optimization` (descent, condition number). Standard
reference: :cite:`Boyd.Vandenberghe.2004` (ch. 2–3),
:cite:`Goodfellow.Bengio.Courville.2016` (ch. 4).

::: {.callout-note title="⟢ 3.2.1 Convex Sets"}
**Outline:** 1. Definition: a set is convex if the segment between any two of its
points stays inside. · 2. The catalog DL actually uses: half-spaces, hyperplanes,
norm balls, the probability simplex, the PSD cone. · 3. Operations that *preserve*
convexity (intersection, affine image/preimage, scaling) — the practical way to
*prove* a set convex without first principles. · 4. Convex hull as the smallest
convex set containing a point cloud.
**Key results to state:** convex iff $\theta\mathbf{x}+(1-\theta)\mathbf{y}\in C$ for all $\mathbf{x},\mathbf{y}\in C,\ \theta\in[0,1]$; half-space $\{\mathbf{x}:\mathbf{a}^\top\mathbf{x}\le b\}$; simplex $\{\mathbf{x}\succeq 0,\ \mathbf{1}^\top\mathbf{x}=1\}$; PSD cone $\mathbb{S}^n_+=\{A=A^\top:\mathbf{z}^\top A\mathbf{z}\ge0\ \forall\mathbf{z}\}$; intersection of convex sets is convex (union generally is not).
**Diagrams:** :numref:`fig_mdl-opt-convex-vs-nonconvex-set` (below) — a convex blob with an interior chord vs. a non-convex crescent whose chord exits, plus the simplex triangle and a half-space.
**Worked example(s):** verify the probability simplex is the intersection of a hyperplane and the nonnegative orthant (hence convex); show an annulus is not convex.
**Exercises (draft):** (1) Prove the intersection of convex sets is convex; give a two-set union counterexample. (2) Show every norm ball is convex from the triangle inequality. (3) Show the PSD cone is convex.
:::

The definition is visual: a set is convex when the segment between any two of its points stays inside it. :numref:`fig_mdl-opt-convex-vs-nonconvex-set` contrasts a convex set, where every chord lies within, with a non-convex crescent, where a chord between two of its points slips outside; it also shows the two convex sets deep learning uses most, the probability simplex (a hyperplane cut of the nonnegative orthant) and a half-space.

![Convex versus non-convex sets. Left: a convex set, where the segment between any two points stays inside. Middle: a non-convex crescent, where a chord between two of its points passes outside the set. Right: the probability simplex $\{\mathbf{p}\succeq0,\ \mathbf{1}^\top\mathbf{p}=1\}$ and a half-space $\{\mathbf{x}:\mathbf{a}^\top\mathbf{x}\leq b\}$, both convex.](../img/mdl-opt-convex-vs-nonconvex-set.svg)
:label:`fig_mdl-opt-convex-vs-nonconvex-set`

::: {.callout-note title="⟢ 3.2.2 Convex Functions: Three Lenses"}
**Outline:** 1. **Chord lens** (definition, no smoothness needed): the graph lies
below every chord. · 2. **First-order lens** (differentiable): the graph lies above
every tangent — the tangent is a global *under-estimator*, the property gradient
methods exploit. · 3. **Second-order lens** (twice-differentiable): the Hessian is
PSD everywhere. · 4. Strict and *strongly* convex variants (ties the $\mu$ of
`sec_mdl-gradient-based-optimization` §3.1.3). · 5. Emphasize the three are
equivalent under their smoothness assumptions — pick whichever is easiest to check.
**Key results to state:** chord: $f(\theta\mathbf{x}+(1-\theta)\mathbf{y})\le\theta f(\mathbf{x})+(1-\theta)f(\mathbf{y})$; first-order: $f(\mathbf{y})\ge f(\mathbf{x})+\nabla f(\mathbf{x})^\top(\mathbf{y}-\mathbf{x})$; second-order: $\nabla^2 f(\mathbf{x})\succeq 0$; strongly convex: $\nabla^2 f\succeq \mu I$.
**Diagrams:** :numref:`fig_mdl-opt-chord-above-graph` (below) — a convex curve with a chord lying above it (chord lens) beside the same curve with a tangent lying below it (first-order lens), annotating both inequalities.
**Worked example(s):** verify the squared loss $\tfrac12\|\mathbf{X}\mathbf{w}-\mathbf{y}\|_2^2$ is convex (Hessian $X^\top X\succeq0$) by all three lenses; show $\|\mathbf{x}\|_1$ is convex by the chord lens (non-smooth, so the second-order lens does not apply).
**Exercises (draft):** (1) Prove the three lenses equivalent (chord $\Leftrightarrow$ first-order $\Leftrightarrow$ second-order). (2) Show $x\log x$ is convex on $x>0$. (3) Show a sum $\sum_i\max(0,1-y_i\,\mathbf{w}^\top\mathbf{x}_i)$ (the hinge loss) is convex.
:::

The first two lenses are two ways of looking at the same picture, shown in :numref:`fig_mdl-opt-chord-above-graph`. The chord lens reads off the definition directly: the chord joining any two points on the graph lies *above* the graph. The first-order lens reads the slope: the tangent at any point lies *below* the graph, so it is a global under-estimator — the property gradient methods rely on.

![Two equivalent lenses on convexity. Left (chord lens): the chord joining two points lies above the graph, so $f(\theta\mathbf{x}+(1-\theta)\mathbf{y})\le\theta f(\mathbf{x})+(1-\theta)f(\mathbf{y})$. Right (first-order lens): the tangent at a point lies below the graph, so $f(\mathbf{y})\ge f(\mathbf{x})+\nabla f(\mathbf{x})^\top(\mathbf{y}-\mathbf{x})$.](../img/mdl-opt-chord-above-graph.svg)
:label:`fig_mdl-opt-chord-above-graph`

::: {.callout-note title="⟢ 3.2.3 Jensen's Inequality"}
**Outline:** 1. Generalize the chord inequality from two points to any distribution.
· 2. State Jensen: for convex $f$, $\mathbb{E}[f(X)]\ge f(\mathbb{E}[X])$ (reversed
for concave). · 3. The bridge to expectations — this is the workhorse behind the
nonnegativity of KL divergence and the entropy bounds of
`sec_mdl-information_theory`. · 4. AM–GM as a one-line sanity check (concavity of
$\log$).
**Key results to state:** $\mathbb{E}[f(X)]\ge f(\mathbb{E}[X])$; equality iff $X$ is a.s. constant (or $f$ affine on $X$'s support); AM–GM $\tfrac1n\sum x_i\ge(\prod x_i)^{1/n}$ from concavity of $\log$; $D_{\mathrm{KL}}(p\|q)\ge 0$ via Jensen on $-\log$.
**Diagrams:** reuse :numref:`fig_mdl-opt-chord-above-graph`; annotate $\mathbb{E}[f(X)]$ above $f(\mathbb{E}[X])$ for a two-mass distribution.
**Worked example(s):** derive AM–GM from Jensen; derive $D_{\mathrm{KL}}\ge0$ from Jensen on $-\log$ (forward-link: `sec_mdl-information_theory` Gibbs' inequality should back-reference here).
**Exercises (draft):** (1) Prove Jensen for a two-point distribution directly from the chord definition. (2) Use Jensen to show $D_{\mathrm{KL}}(p\|q)\ge0$ with equality iff $p=q$. (3) Derive AM–GM and the harmonic–geometric–arithmetic mean chain.
:::

::: {.callout-note title="⟢ 3.2.4 Why Convexity Matters"}
**Outline:** 1. The headline theorem: for a convex function over a convex set,
*every local minimum is a global minimum*. · 2. Proof sketch from the first-order
lens (a stationary point under-estimates everything, so it is the global min). · 3.
No saddles, no spurious local basins — the failure modes that plague deep nets are
*structurally absent*. · 4. Sublevel sets $\{f\le\alpha\}$ are convex (so the
optimal set is convex); strong convexity $\Rightarrow$ a *unique* minimizer. · 5.
Tie back to `sec_mdl-gradient-based-optimization`: this is what upgrades the local
rates into *global* guarantees.
**Key results to state:** local min $\Rightarrow$ global min for convex $f$; $\nabla f(\mathbf{x}^\star)=0\Rightarrow \mathbf{x}^\star$ global (unconstrained convex); sublevel sets convex; strongly convex $\Rightarrow$ unique minimizer.
**Diagrams:** :numref:`fig_mdl-opt-local-equals-global` (below) — a convex bowl (single global min) beside a non-convex landscape with multiple local minima and a saddle, gradient descent reaching the one global minimum on the left but landing in either basin on the right.
**Worked example(s):** prove local=global from the first-order condition; exhibit a double-well non-convex function where GD's limit depends on initialization.
**Exercises (draft):** (1) Prove every local min of a convex function is global. (2) Show a strongly convex function has a unique minimizer. (3) Give a non-convex function with a continuum of global minima and one with a strict saddle.
:::

:numref:`fig_mdl-opt-local-equals-global` shows why this is the dividing line. On a convex objective gradient descent reaches the single global minimum no matter where it starts; on a non-convex landscape the same algorithm slides into whichever basin it happens to start in, separated by a saddle — the local minimum on one side is *not* the global one.

![Why convexity matters. Left: a convex objective has one global minimum, and gradient descent from any start reaches it. Right: a non-convex landscape with two local minima separated by a saddle; gradient descent lands in different minima depending on its starting point, so a local minimum need not be global.](../img/mdl-opt-local-equals-global.svg)
:label:`fig_mdl-opt-local-equals-global`

::: {.callout-note title="⟢ 3.2.5 Recognizing Convexity (a Checklist)"}
**Outline:** 1. The composition calculus that lets you certify convexity *without*
computing a Hessian. · 2. Convexity-preserving operations: nonnegative weighted
sums, pointwise maximum/supremum, affine pre-composition $f(A\mathbf{x}+\mathbf{b})$,
and composition rules with monotone outer functions. · 3. log-sum-exp as the
canonical worked case (ties directly to `sec_mdl-numerical-stability-conditioning`).
· 4. A practical checklist students can run on any candidate loss.
**Key results to state:** $\sum_i w_i f_i$ convex for $w_i\ge0$; $\max_i f_i$ convex; $f(A\mathbf{x}+\mathbf{b})$ convex if $f$ convex; $\mathrm{lse}(\mathbf{x})=\log\sum_i e^{x_i}$ is convex (Hessian $=\mathrm{diag}(\mathbf{s})-\mathbf{s}\mathbf{s}^\top\succeq0$ where $\mathbf{s}=\mathrm{softmax}(\mathbf{x})$).
**Diagrams:** none new; reference :numref:`fig_mdl-opt-chord-above-graph` for the pointwise-max picture (upper envelope of lines is convex).
**Worked example(s):** prove $\mathrm{lse}$ convex by showing its Hessian is PSD (it is a covariance matrix of the softmax distribution); conclude softmax/multinomial cross-entropy is convex in the logits.
**Exercises (draft):** (1) Show the pointwise max of convex functions is convex; the pointwise min need not be. (2) Show $\mathrm{lse}$ is convex and identify its Hessian as a softmax covariance. (3) Use the checklist to certify logistic-regression NLL convex in $\mathbf{w}$.
:::

::: {.callout-note title="⟢ 3.2.6 Reality Check: Deep Nets Are Non-Convex"}
**Outline:** 1. Composition destroys convexity: stack a convex loss on a nonlinear
network and the parameter-space landscape is generally non-convex (permutation
symmetries alone create many equivalent minima). · 2. What we still keep:
over-parameterization tends to smooth the landscape; many local minima are
near-equivalent in loss; bad strict saddles are escapable by SGD noise. · 3.
Intuition-only pointer to the **Polyak–Łojasiewicz (PL) condition** — a relaxation
giving *linear* convergence to a global value *without* convexity, which partly
explains why over-parameterized nets train well. · 4. Honest framing: the convex
theory is the *idealization* the practical optimizers approximate, not a literal
description of deep training.
**Key results to state:** convexity not preserved under composition with nonlinear maps; PL inequality $\tfrac12\|\nabla f(\mathbf{x})\|_2^2\ge \mu\,(f(\mathbf{x})-f^\star)$ $\Rightarrow$ linear convergence of GD (state, do not prove); loss-landscape symmetry $\Rightarrow$ multiple global minima.
**Diagrams:** reuse :numref:`fig_mdl-opt-local-equals-global` (the non-convex panel) with a PL annotation; no new SVG.
**Worked example(s):** show a tiny 1-hidden-unit network's loss is non-convex in its weights (two symmetric global minima); verify a PL-satisfying non-convex function (e.g. $x^2 + 3\sin^2 x$ style) still converges linearly under GD.
**Exercises (draft):** (1) Exhibit non-convexity of a 2-layer net's loss via weight-permutation symmetry. (2) Show PL is implied by strong convexity but not conversely. (3) Discuss why convexity of the *last-layer* problem (fixed features) still matters in practice.
:::

## Summary

*Planned.* A set is convex if segments stay inside; a function is convex by any of
three equivalent lenses (chord / tangent-below / PSD Hessian). Jensen's inequality
lifts the chord property to expectations and underwrites $D_{\mathrm{KL}}\ge0$. The
payoff of convexity is that local equals global and the descent rates become global
guarantees. A convexity checklist (sums, max, affine composition, log-sum-exp) lets
you certify most last-layer losses. Deep networks are non-convex, but
over-parameterization and PL-type conditions explain why first-order methods still
succeed.

## Exercises

*Planned — consolidated.* (1) Intersection of convex sets convex; union
counterexample. (2) First/second-order equivalence. (3) Jensen $\Rightarrow$ AM–GM
and $D_{\mathrm{KL}}\ge0$. (4) Convexity of $x\log x$, hinge loss, and the
non-convexity of a 2-layer net's loss.

## Discussions

*Planned placeholder.* Foundation for `sec_mdl-constrained-optimization-duality`
(strong duality / KKT sufficiency need convexity) and for the global guarantees in
`sec_mdl-gradient-based-optimization`; forward to `sec_mdl-information_theory`
(Jensen/KL) and the main book's `sec_convexity`, SVM, and softmax-regression
sections.

<!-- slides -->

*Planned.* Slide deck to be authored once body cells exist, with `@<id>`
placeholders for the three-lens convexity check, the Jensen/AM–GM derivation, and
the convex-vs-double-well landscape plot.
