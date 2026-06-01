# Gradient-Based Optimization
:label:`sec_mdl-gradient-based-optimization`

Every neural network is moved by some descendant of gradient descent. Before the
main book hands you the optimizer zoo — momentum, RMSProp, Adam, learning-rate
schedules — this section explains the foundations underneath all of them: *why* a
negative-gradient step makes progress, *how fast* it converges, and *what breaks
it*. The recurring character is the **condition number** $\kappa$: a single number,
read off the Hessian's spectrum, that predicts whether gradient descent glides to
the minimum or grinds along an ill-conditioned valley. We develop the deterministic
theory on quadratics (where everything is computable in closed form), then add the
two ingredients that make it practical at scale — momentum and stochasticity — and
close with a short look at second-order methods to explain why we *don't* use them.

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
This file is a teaching outline / detailed table of contents, not finished prose.
It fixes the subsection flow, the key results to state, the diagrams to build, the
worked examples to compute, and draft exercises. Body framing: foundations-level —
explain *why* the main book's optimizers work, do **not** duplicate the main book's
training how-to (`sec_optimization-intro`, `sec_gd`, `sec_sgd`, `sec_momentum`,
`sec_minibatch-sgd`, `sec_adam`).
:::

**Prerequisites:** `sec_mdl-multivariable_calculus` (gradient, directional
derivative, Hessian, second-order Taylor), `sec_mdl-eigendecompositions` and
`sec_mdl-svd-low-rank` (eigenvalues, condition number $\kappa = \lambda_{\max}/\lambda_{\min}$),
with `sec_mdl-convexity` forward-referenced for the global-optimum guarantees. Standard
references: :cite:`Boyd.Vandenberghe.2004`, :cite:`Nesterov.2018`,
:cite:`Goodfellow.Bengio.Courville.2016` (ch. 4, 8).

::: {.callout-note title="⟢ 3.1.1 Descent Directions"}
**Outline:** 1. Start from the first-order model $f(\mathbf{x} + \eta\mathbf{d}) \approx f(\mathbf{x}) + \eta\,\nabla f(\mathbf{x})^\top \mathbf{d}$. · 2. Any direction with negative directional derivative is a *descent direction*; for small enough $\eta$ it strictly decreases $f$. · 3. Among unit directions, $-\nabla f$ is steepest (Cauchy–Schwarz equality case) — this is *why* we follow the negative gradient, not a definition. · 4. Note other valid descent directions (Newton, preconditioned) preview §3.1.7.
**Key results to state:** $\mathbf{d}$ is a descent direction iff $\nabla f(\mathbf{x})^\top \mathbf{d} < 0$; $\arg\min_{\|\mathbf{d}\|_2 = 1} \nabla f^\top \mathbf{d} = -\nabla f / \|\nabla f\|_2$; directional derivative $D_{\mathbf{d}} f = \nabla f^\top \mathbf{d}$.
**Diagrams:** reuse `fig_mdl-gradient-contour` idea (from `sec_mdl-multivariable_calculus`) only by reference — the new fig lives in §3.1.3.
**Worked example(s):** on $f(x,y)=x^2+10y^2$, show $-\nabla f$ vs. an arbitrary descent direction both lower $f$, but steepest is locally fastest.
**Exercises (draft):** (1) Show $\mathbf{d}=-B\nabla f$ is a descent direction for any positive-definite $B$. (2) Find the steepest-descent direction under the norm $\|\mathbf{d}\|_A=\sqrt{\mathbf{d}^\top A\mathbf{d}}$ and recognize it as a preconditioned gradient.
:::

::: {.callout-note title="⟢ 3.1.2 Gradient Descent and L-Smoothness"}
**Outline:** 1. The iteration $\mathbf{x}_{k+1} = \mathbf{x}_k - \eta\,\nabla f(\mathbf{x}_k)$. · 2. Define $L$-smoothness ($\nabla f$ is $L$-Lipschitz; equivalently $\nabla^2 f \preceq L I$) and derive the quadratic *upper* bound. · 3. Plug the GD step into the upper bound to get the **descent lemma**: progress is guaranteed for $\eta \le 1/L$, with per-step decrease $\ge \tfrac{\eta}{2}\|\nabla f\|_2^2$. · 4. Intuition hook: $L$ is the worst-case curvature; step shorter than $1/L$ so you never overshoot the parabola that bounds $f$ from above.
**Key results to state:** $f(\mathbf{y}) \le f(\mathbf{x}) + \nabla f(\mathbf{x})^\top(\mathbf{y}-\mathbf{x}) + \tfrac{L}{2}\|\mathbf{y}-\mathbf{x}\|_2^2$; descent lemma $f(\mathbf{x}_{k+1}) \le f(\mathbf{x}_k) - \eta(1-\tfrac{L\eta}{2})\|\nabla f(\mathbf{x}_k)\|_2^2$; monotone for $0<\eta\le 1/L$.
**Diagrams:** :numref:`fig_mdl-opt-gd-bowl-vs-valley` (below) — GD trajectory on an isotropic bowl ($\kappa\approx1$, straight to minimum) beside an anisotropic valley ($\kappa\gg1$, zig-zag), the headline contrast for the whole section.
**Worked example(s):** for the quadratic $f(\mathbf{x})=\tfrac12\mathbf{x}^\top A\mathbf{x}$, identify $L=\lambda_{\max}(A)$ and verify the descent lemma numerically.
**Exercises (draft):** (1) Prove the descent lemma from $L$-smoothness. (2) Show $\eta=1/L$ maximizes the guaranteed one-step decrease.
:::

The headline contrast for the whole section is in :numref:`fig_mdl-opt-gd-bowl-vs-valley`. On a well-conditioned bowl the contours are near-circular and gradient descent heads almost straight to the minimum; in an ill-conditioned valley the step size is throttled by the steep direction (to stay stable) while the flat $\lambda_{\min}$ direction needs many such small steps, so the iterates zig-zag across the narrow valley. The zig-zag is the visible cost of a large condition number $\kappa$.

![Gradient descent on a quadratic $f(\mathbf{x})=\tfrac12\mathbf{x}^\top A\mathbf{x}$. Left: a well-conditioned bowl ($\kappa\approx1$, near-circular contours, an almost straight path). Right: an ill-conditioned valley ($\kappa\gg1$, elongated contours); a step size near the stability ceiling makes the steep mode oscillate while the slow $\lambda_{\min}$ axis barely moves, producing a zig-zag.](../img/mdl-opt-gd-bowl-vs-valley.svg)
:label:`fig_mdl-opt-gd-bowl-vs-valley`

::: {.callout-note title="⟢ 3.1.3 Convergence Rates and the Condition Number"}
**Outline:** 1. The closed-form GD analysis on the quadratic: in the eigenbasis of $A$, each coordinate contracts independently by $(1-\eta\lambda_i)$. · 2. Smooth-convex (no strong convexity): sublinear $O(1/k)$ in function value. · 3. Strongly convex: *linear* (geometric) convergence with rate governed by $\kappa$ — slow along the small-curvature axis. · 4. The eigenvalue-bowl picture: the slowest mode sets the rate; this is the math behind the §3.1.2 valley diagram.
**Key results to state:** per-mode factor $1-\eta\lambda_i$; optimal fixed step $\eta^\star = \tfrac{2}{\lambda_{\min}+\lambda_{\max}}$ giving contraction $\tfrac{\kappa-1}{\kappa+1}$ per step; smooth-convex $f(\mathbf{x}_k)-f^\star \le \tfrac{L\|\mathbf{x}_0-\mathbf{x}^\star\|_2^2}{2k} = O(1/k)$; strongly convex (parameter $\mu$, $\kappa=L/\mu$) $f(\mathbf{x}_k)-f^\star \le \big(1-\tfrac{1}{\kappa}\big)^k \big(f(\mathbf{x}_0)-f^\star\big)$, i.e. $O\!\big(\kappa\log\tfrac1\varepsilon\big)$ iterations.
**Diagrams:** reuse :numref:`fig_mdl-opt-gd-bowl-vs-valley`; its right panel annotates the zig-zag with the slow ($\lambda_{\min}$) axis.
**Worked example(s):** closed-form GD on $\tfrac12\mathbf{x}^\top A\mathbf{x}$ with $A=\mathrm{diag}(1,10)$: derive the per-mode factors, the optimal $\eta^\star$, and verify the $(\kappa-1)/(\kappa+1)$ contraction against an iteration count.
**Exercises (draft):** (1) Derive $\eta^\star=2/(\lambda_{\min}+\lambda_{\max})$ by minimizing $\max_i|1-\eta\lambda_i|$. (2) Show the optimal contraction is exactly $(\kappa-1)/(\kappa+1)$ and explain why $\kappa\to1$ is one-step convergence. (3) Connect $\kappa$ here to $\kappa=\sigma_{\max}/\sigma_{\min}$ from `sec_mdl-svd-low-rank` (forward-link to `sec_mdl-numerical-stability-conditioning`).
:::

::: {.callout-note title="⟢ 3.1.4 Step Size and Line Search"}
**Outline:** 1. Fixed step vs. adaptive step. · 2. Stability ceiling: on a quadratic, GD's iteration map has spectral radius $\max_i|1-\eta\lambda_i|$; $\eta > 2/L$ makes it $>1$ and the iterates *diverge*. · 3. Backtracking line search with the Armijo (sufficient-decrease) condition — guarantees progress without knowing $L$. · 4. Hook to the main book's learning-rate schedules: warmup/decay are practical surrogates for line search at scale.
**Key results to state:** convergence requires $0<\eta<2/L$; divergence for $\eta>2/L$; Armijo condition $f(\mathbf{x}-\eta\nabla f) \le f(\mathbf{x}) - c\,\eta\,\|\nabla f\|_2^2$ with $c\in(0,1)$; backtrack $\eta \leftarrow \beta\eta$ until satisfied.
**Diagrams:** small inset (no dedicated fig) — three GD trajectories at $\eta<1/L$ (slow), $\eta\approx 2/(\lambda_{\min}+\lambda_{\max})$ (fast), $\eta>2/L$ (spiraling out). Specified as part of the :numref:`fig_mdl-opt-gd-bowl-vs-valley` family; no new SVG.
**Worked example(s):** $\eta$-sweep on the 1-D quadratic $f(x)=\tfrac{L}{2}x^2$ showing convergence → oscillation at $\eta=2/L$ → divergence; tabulate $|1-\eta L|$.
**Exercises (draft):** (1) Find the exact $\eta$ at which $f(x)=\tfrac{L}{2}x^2$ oscillates without converging or diverging. (2) Implement backtracking line search and show it never picks $\eta>2/L$ on a quadratic.
:::

::: {.callout-note title="⟢ 3.1.5 Momentum and Nesterov Acceleration"}
**Outline:** 1. Heavy-ball as a *damped oscillator*: the update $\mathbf{v}_{k+1}=\beta\mathbf{v}_k - \eta\nabla f$, $\mathbf{x}_{k+1}=\mathbf{x}_k+\mathbf{v}_{k+1}$ adds inertia, averaging out the zig-zag in the slow valley. · 2. Why it helps: on a quadratic it damps the high-curvature oscillation while accelerating the low-curvature drift. · 3. Nesterov's "look-ahead" gradient and its provably *optimal* first-order rate. · 4. The headline payoff: the dependence on the condition number improves from $\kappa$ to $\sqrt{\kappa}$.
**Key results to state:** heavy-ball $(\beta,\eta)$ optimal on quadratics gives contraction $\approx \tfrac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}$ with $\beta^\star=\big(\tfrac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\big)^2$; Nesterov smooth-convex rate $f(\mathbf{x}_k)-f^\star = O(1/k^2)$; strongly convex $O\!\big((1-1/\sqrt{\kappa})^k\big)$, i.e. $O(\sqrt{\kappa}\log\tfrac1\varepsilon)$ iterations vs. GD's $O(\kappa\log\tfrac1\varepsilon)$.
**Diagrams:** `fig_mdl-momentum-damping` — same anisotropic valley as §3.1.2 with plain-GD zig-zag overlaid against the damped momentum path that cuts diagonally to the minimum; inset of the 1-D mass-spring-damper analogy.
**Worked example(s):** momentum on $A=\mathrm{diag}(1,10)$ recovering the $\sqrt{\kappa}$ speedup; plot iteration count to fixed tolerance for GD vs. momentum vs. Nesterov.
**Exercises (draft):** (1) Write momentum on a 1-D quadratic as a 2-D linear map and find the eigenvalues of its iteration matrix. (2) Show the critically-damped choice gives the $\sqrt{\kappa}$ rate. (3) Explain why the $\kappa\to\sqrt{\kappa}$ improvement matters most for ill-conditioned problems. Cite :cite:`Polyak.1964`, :cite:`Nesterov.2018`, :cite:`Sutskever.Martens.Dahl.ea.2013`.
:::

::: {.callout-note title="⟢ 3.1.6 Stochastic Gradient Descent"}
**Outline:** 1. The scale problem: full-batch GD costs $O(N)$ per step; replace $\nabla f$ with a mini-batch estimate. · 2. The mini-batch gradient is an *unbiased* estimate of the full gradient; its variance scales as $1/b$ (batch size $b$). · 3. Consequence: per-step cost decouples from $N$, but noise floors the accuracy at a fixed step size — hence *decaying* step sizes. · 4. Rate: convex SGD attains $O(1/\sqrt{k})$ (and $O(1/k)$ strongly convex with $\eta_k\propto 1/k$); contrast with the deterministic linear rate. · 5. Hook: this is *why* the main book's SGD/Adam are not full-batch and why learning rate and batch size are coupled.
**Key results to state:** $\mathbb{E}[\hat{\mathbf{g}}_b]=\nabla f$; $\mathrm{Var}(\hat{\mathbf{g}}_b)=\tfrac1b\,\mathrm{Var}(\hat{\mathbf{g}}_1)$ (i.i.d. sampling); Robbins–Monro step conditions $\sum_k\eta_k=\infty,\ \sum_k\eta_k^2<\infty$; convex rate $\mathbb{E}[f(\bar{\mathbf{x}}_k)]-f^\star = O(1/\sqrt{k})$; noise ball radius $\propto \eta\,\sigma^2$ at fixed $\eta$.
**Diagrams:** `fig_mdl-sgd-noisy-path` — noisy SGD trajectory rattling toward the minimum and settling into a noise ball, beside the smooth full-batch path; inset showing the ball shrinking as $\eta$ decays.
**Worked example(s):** estimate gradient variance vs. $b$ on a logistic-regression toy and confirm the $1/b$ law; show fixed-$\eta$ SGD plateaus in a noise ball while a $1/k$ schedule converges.
**Exercises (draft):** (1) Prove the mini-batch gradient is unbiased and its variance is $\propto 1/b$. (2) Batch size vs. compute: derive the variance-per-FLOP tradeoff and explain diminishing returns of large batches. (3) Show a constant step size cannot reach $f^\star$ when gradient noise is nonzero. Reference :cite:`Goodfellow.Bengio.Courville.2016` (ch. 8); attribute the step-size conditions to Robbins and Monro descriptively.
:::

::: {.callout-note title="⟢ 3.1.7 Second-Order Coda: Why Not Newton?"}
**Outline:** 1. Newton minimizes the *local quadratic model* exactly: $\mathbf{x}_{k+1}=\mathbf{x}_k - (\nabla^2 f)^{-1}\nabla f$. · 2. It is **affine-invariant** — no condition-number dependence, no learning rate to tune — and converges quadratically near a minimum. · 3. Why we don't use it at scale: forming/inverting the $d\times d$ Hessian is $O(d^3)$ with $d$ in the billions; the Hessian may be indefinite away from a minimum. · 4. Pointer: quasi-Newton (BFGS/L-BFGS) approximates curvature from gradients; Adam-style methods are *diagonal* preconditioners, the cheap shadow of Newton.
**Key results to state:** Newton step $-(\nabla^2 f)^{-1}\nabla f$; on a quadratic it reaches the minimum in *one* step; local quadratic convergence $\|\mathbf{x}_{k+1}-\mathbf{x}^\star\| \le C\|\mathbf{x}_k-\mathbf{x}^\star\|^2$; cost $O(d^3)$ per step.
**Diagrams:** none new — reference the §3.1.3 contour, noting Newton transforms the elongated valley into a circular bowl.
**Worked example(s):** Newton vs. GD on $A=\mathrm{diag}(1,100)$: Newton in one step, GD's contraction $(\kappa-1)/(\kappa+1)\approx 0.98$.
**Exercises (draft):** (1) Show Newton solves a quadratic in one step regardless of $\kappa$. (2) Explain why Newton is affine-invariant while GD is not. (3) Estimate the FLOPs of a Newton step for $d=10^9$ and conclude. Quasi-Newton pointer: :cite:`Liu.Nocedal.1989`; adaptive diagonal preconditioning: :cite:`Kingma.Ba.2014`.
:::

## Summary

*Planned.* GD follows $-\nabla f$ because it is the steepest descent direction;
$L$-smoothness bounds the safe step size $\eta\le 1/L$; the condition number $\kappa$
controls the speed — sublinear $O(1/k)$ for smooth-convex, linear with rate
$(\kappa-1)/(\kappa+1)$ for strongly convex. Momentum/Nesterov improve $\kappa\to\sqrt{\kappa}$;
SGD trades exact gradients for unbiased $1/b$-variance estimates and an $O(1/\sqrt{k})$
rate, decoupling per-step cost from dataset size. Newton removes $\kappa$ entirely but
costs $O(d^3)$, which is why deep learning uses first-order methods with cheap
curvature surrogates.

## Exercises

*Planned — consolidated from the per-subsection drafts above.* (1) $\eta\le1/L$
descent guarantee. (2) Optimal-step contraction $(\kappa-1)/(\kappa+1)$. (3) 2-D
quadratic converging on one axis while diverging on another (choose $\eta$ between
$2/\lambda_{\min}$ and $2/\lambda_{\max}$). (4) Batch size: variance vs. compute.

## Discussions

*Planned placeholder.* Link to the main book's optimization chapter
(`sec_optimization-intro`, `sec_gd`, `sec_sgd`, `sec_momentum`) for the how-to; this
section is the rate/condition-number justification those sections should
back-reference.

<!-- slides -->

*Planned.* Slide deck `::: {.slide ...}` divs to be authored once the body cells
exist, with `@<id>` placeholders pointing to the worked-example code fences
(`fig_mdl-opt-gd-bowl-vs-valley`, the $\eta$-sweep, the momentum speedup).
