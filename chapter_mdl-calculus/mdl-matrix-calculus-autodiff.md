# Matrix Calculus and Automatic Differentiation
:label:`sec_mdl-matrix-calculus-autodiff`

The previous two sections built up differentiation from a single weight (:numref:`sec_mdl-single_variable_calculus`) to the gradient of a scalar loss over many weights (:numref:`sec_mdl-multivariable_calculus`). Real network layers, however, map *vectors to vectors* and their parameters are *matrices*, so the natural object of study is the derivative of a vector-valued map---the *Jacobian*---and the natural question is why `loss.backward()` is cheap. This section answers both. Its punchline is that **backpropagation is reverse-mode automatic differentiation: a sequence of vector--Jacobian products**, and that the choice between forward- and reverse-mode AD is dictated entirely by the shape of the Jacobian you are after.

This section **absorbs the "A Little Matrix Calculus" material currently at the end of :numref:`sec_mdl-multivariable_calculus`** (the $\boldsymbol\beta^\top\mathbf x$, $\mathbf x^\top A\mathbf x$, and $\|\mathbf X-\mathbf U\mathbf V\|^2$ derivations, plus the "guess from the $1\times1$ case and fix the shapes by transpose" heuristic). That material moves into §2.3.4 below; the wrong-variable typo in its $\boldsymbol\beta^\top\mathbf x$ derivation is fixed in the migration.

::: {.callout-note title="⟢ Planned — section under construction"}
This is the §2.3 plan. Subsections 2.3.1--2.3.6 below are ToC stubs; §2.3.4 is seeded by existing prose migrated from §2.2.
:::

## The Jacobian of a Vector-Valued Map
:label:`subsec_mdl-jacobian`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** A layer $\mathbf f:\mathbb R^n\to\mathbb R^m$ does not have "a derivative" but a whole matrix of them; the *Jacobian* $J_{\mathbf f}$ is the single best local linear approximation, generalizing the slope of :numref:`sec_mdl-single_variable_calculus` and the gradient of :numref:`sec_mdl-multivariable_calculus` in one stroke.
**Outline:** 1. Define $J_{\mathbf f}(\mathbf x)\in\mathbb R^{m\times n}$ with $[J_{\mathbf f}]_{ij} = \partial f_i/\partial x_j$. · 2. The linearization $\mathbf f(\mathbf x+\boldsymbol\epsilon)\approx \mathbf f(\mathbf x) + J_{\mathbf f}(\mathbf x)\,\boldsymbol\epsilon$ (the vector form of :eqref:`eq_mdl-small_change`). · 3. Special cases: $m=1$ recovers the gradient (one *row*), and the Jacobian of $\nabla f$ is the Hessian $\mathbf H$ (tie to :eqref:`eq_mdl-hess_def`). · 4. Worked $2\to2$ map, with a finite-difference cross-check.
**Key results to state:** $[J_{\mathbf f}]_{ij}=\partial f_i/\partial x_j$; $\mathbf f(\mathbf x+\boldsymbol\epsilon)\approx \mathbf f(\mathbf x)+J_{\mathbf f}\boldsymbol\epsilon$; $\nabla f = J_f^\top$ (a column) and $\mathbf H = J_{\nabla f}$.
**Diagrams:** `fig_mdl-jacobian-stack` — a deep net drawn as a stack of layers with the end-to-end gradient shown as the matrix product $J_L\cdots J_2 J_1$ of per-layer Jacobians (introduced here, paid off in 2.3.2).
**Worked example(s):** Jacobian of $\mathbf f(x,y)=(x^2y,\ \sin(x+y))$ by hand vs. finite differences; identify the gradient and Hessian special cases on a scalar $f$.
**Exercises (draft):** 1. Compute $J$ of $(x^2y,\sin(x+y))$ and verify the linearization at a point. 2. Show $\nabla f = J_f^\top$ when $m=1$. 3. Give the Jacobian of an affine map $\mathbf x\mapsto A\mathbf x+\mathbf b$. 4. Confirm the Hessian is the Jacobian of the gradient for $f(x,y)=x^2+xy$.
**Prereqs / cross-refs:** :numref:`sec_mdl-multivariable_calculus` (gradient, Hessian); :numref:`sec_mdl-geometry-linear-algebraic-ops` (matrix-vector products, transpose); feeds 2.3.2.
:::

## Chain Rule as Jacobian Composition
:label:`subsec_mdl-jacobian-chain`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** The multivariate chain rule of :numref:`sec_mdl-multivariable_calculus` is, in matrix form, just *multiplication of Jacobians*---which means a deep network's end-to-end derivative is one long matrix product, and the order in which you multiply that product is exactly the difference between forward- and reverse-mode AD (2.3.5--2.3.6).
**Outline:** 1. State $J_{\mathbf g\circ\mathbf f}(\mathbf x) = J_{\mathbf g}(\mathbf f(\mathbf x))\,J_{\mathbf f}(\mathbf x)$. · 2. Iterate to a depth-$L$ composition: $J = J_L J_{L-1}\cdots J_1$. · 3. Read the path-sum chain rule of §2.2 as the entries of this product. · 4. Foreshadow associativity: the *same* product, parenthesized right-to-left (VJPs) or left-to-right (JVPs), gives reverse vs. forward mode.
**Key results to state:** $J_{\mathbf g\circ\mathbf f} = J_{\mathbf g}\,J_{\mathbf f}$; depth-$L$ net $\Rightarrow J = \prod_{\ell=L}^{1} J_\ell$.
**Diagrams:** reuse `fig_mdl-jacobian-stack`; reuse the §2.2 DAGs (`fig_mdl-chain-1`, `fig_mdl-chain-2`) to show edges carrying local Jacobians.
**Worked example(s):** two-layer composition $\mathbf h = \mathbf g(\mathbf f(\mathbf x))$; multiply the $2\times2$ Jacobians and check against differentiating the composite directly.
**Exercises (draft):** 1. Compose two affine-then-elementwise layers and write $J$ as a product. 2. Show matrix-multiplication associativity is what lets us choose an evaluation order. 3. Where does a $\mathrm{diag}(\sigma')$ factor come from in an elementwise activation? 4. Relate the product to the §2.2 path-sum formula.
**Prereqs / cross-refs:** 2.3.1; :numref:`sec_mdl-multivariable_calculus` (multivariate chain rule, backprop); feeds 2.3.5, 2.3.6.
:::

## Layout Conventions: Numerator vs Denominator
:label:`subsec_mdl-layout`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** The single most common source of stray transposes and sign errors in matrix calculus is mixing up *layout conventions*; stating the convention explicitly up front---and flagging which one the identities below use---saves the reader from the usual confusion.
**Outline:** 1. *Numerator (Jacobian) layout:* the derivative has the shape of the numerator, so $\partial\mathbf y/\partial\mathbf x$ is $m\times n$; the de-facto ML/Jacobian default. · 2. *Denominator (gradient) layout:* the derivative has the shape of the denominator, so $\partial f/\partial\mathbf x$ is a *column* matching $\mathbf x$; common in statistics/pattern recognition and used by the migrated §2.2 identities. · 3. The conversion rule: switching layouts *transposes* the result---numerator and denominator forms are transposes of each other. · 4. Pin it down: "$\nabla f$ is a column (denominator layout); the Jacobian $J_f$ of a scalar $f$ is its transpose (a row, numerator layout)." · 5. **Note:** the identities in 2.3.4 are written in denominator layout, matching the migrated §2.2 material.
**Key results to state:** numerator-layout $\partial\mathbf y/\partial\mathbf x \in\mathbb R^{m\times n}$ vs. denominator-layout $\partial\mathbf y/\partial\mathbf x\in\mathbb R^{n\times m}$; the two differ by a transpose; $\nabla f = (J_f)^\top$.
**Diagrams:** a small two-column "same derivative, two shapes" table/figure showing a $1\times k$ row (numerator) beside a $k\times1$ column (denominator) for $\partial f/\partial\mathbf x$.
**Worked example(s):** write $\partial(\boldsymbol\beta^\top\mathbf x)/\partial\mathbf x$ in both layouts and verify they are transposes; show why a chain-rule product "type-checks" only when layouts are consistent.
**Exercises (draft):** 1. Convert $\partial(A\mathbf x)/\partial\mathbf x$ between layouts. 2. Which layout makes $J_{\mathbf g\circ\mathbf f}=J_{\mathbf g}J_{\mathbf f}$ hold without transposes? 3. Re-derive a §2.2 identity in numerator layout. 4. Spot the layout bug in a given (wrong) gradient expression.
**Prereqs / cross-refs:** 2.3.1 (Jacobian); :numref:`sec_mdl-geometry-linear-algebraic-ops` (transpose); used throughout 2.3.4; cite the Matrix Cookbook :cite:`Petersen.Pedersen.ea.2008`.
:::

## Useful Matrix-Derivative Identities
:label:`subsec_mdl-matrix-identities`

::: {.callout-note title="⟢ Planned — absorbs §2.2 'A Little Matrix Calculus' (migrated, typo fixed)"}
**Body framing:** A small table of matrix-derivative identities covers most of what shows up in deep learning, and the "guess the answer from the scalar $1\times1$ case, then fix the matrix shapes by inserting transposes" heuristic lets you reconstruct them without memorizing a reference.
**Outline (migrate + modernize the existing §2.2 derivations):** 1. $\dfrac{\partial}{\partial\mathbf x}(\boldsymbol\beta^\top\mathbf x) = \boldsymbol\beta$ — *migrated from §2.2, with the wrong-variable typo fixed* ($\partial f/\partial x_i = \beta_i$, differentiating w.r.t. $x_i$ not $\beta_i$). · 2. $\dfrac{\partial}{\partial\mathbf x}(\mathbf x^\top A\mathbf x) = (A+A^\top)\mathbf x$ — migrated Einstein-notation derivation. · 3. $\dfrac{\partial}{\partial\mathbf V}\|\mathbf X-\mathbf U\mathbf V\|_2^2 = -2\mathbf U^\top(\mathbf X-\mathbf U\mathbf V)$ — migrated matrix-factorization example. · 4. *New rows:* $\nabla_{\mathbf A}\operatorname{tr}(\mathbf A\mathbf B) = \mathbf B^\top$; least squares $\nabla_{\mathbf x}\|\mathbf A\mathbf x-\mathbf b\|_2^2 = 2\mathbf A^\top(\mathbf A\mathbf x-\mathbf b)$. · 5. Restate the "$1\times1$ guess, transpose to fix shapes" heuristic explicitly as a method.
**Key results to state:** $\partial(\boldsymbol\beta^\top\mathbf x)/\partial\mathbf x=\boldsymbol\beta$; $\partial(\mathbf x^\top A\mathbf x)/\partial\mathbf x=(A+A^\top)\mathbf x$; $\nabla_{\mathbf x}\|\mathbf A\mathbf x-\mathbf b\|^2=2\mathbf A^\top(\mathbf A\mathbf x-\mathbf b)$; $\nabla_{\mathbf A}\operatorname{tr}(\mathbf A\mathbf B)=\mathbf B^\top$; $\partial_{\mathbf V}\|\mathbf X-\mathbf U\mathbf V\|^2=-2\mathbf U^\top(\mathbf X-\mathbf U\mathbf V)$. *(All in denominator layout, per 2.3.3.)*
**Diagrams:** none required (this is a table); optional small "shape-matching" sketch for the heuristic.
**Worked example(s):** the migrated $\mathbf x^\top A\mathbf x$ index derivation; the migrated $\|\mathbf X-\mathbf U\mathbf V\|^2$ derivation; least-squares gradient two ways (heuristic guess vs. index proof).
**Exercises (draft):** 1. $\nabla\|\mathbf A\mathbf x-\mathbf b\|^2$ via the heuristic and via indices. 2. $\nabla_{\mathbf x}(\boldsymbol\beta^\top\mathbf x)$ and $\nabla_{\mathbf x}(\mathbf x^\top\boldsymbol\beta)$ — why equal (migrated §2.2 Ex.1). 3. $\partial\|\mathbf v\|_2/\partial\mathbf v$ (migrated §2.2 Ex.2). 4. Derive $\nabla_{\mathbf A}\operatorname{tr}(\mathbf A\mathbf B)$.
**Prereqs / cross-refs:** 2.3.3 (layout); :numref:`sec_mdl-geometry-linear-algebraic-ops`; the Matrix Cookbook :cite:`Petersen.Pedersen.ea.2008`; least squares feeds Ch3.
:::

## Forward-Mode AD and Dual Numbers
:label:`subsec_mdl-forward-mode`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** Forward-mode AD computes a *Jacobian--vector product* (JVP)---one *column* of the Jacobian per pass---by carrying a derivative alongside every value as the computation runs forward; the cleanest way to see it is *dual numbers*, where a tiny "$\epsilon$ with $\epsilon^2=0$" tracks the derivative automatically.
**Outline:** 1. JVP: forward mode evaluates $J\mathbf v$ for a chosen seed $\mathbf v$ in one forward pass, never forming $J$. · 2. Dual numbers $a + b\,\epsilon$ with $\epsilon^2=0$: arithmetic rules reproduce the sum/product/chain rules (echo the $\epsilon^2$-is-negligible argument of :numref:`sec_mdl-single_variable_calculus`). · 3. Cost: one pass per *input* direction, so forward mode is cheap for **tall** Jacobians ($m\gg n$, few inputs); $O(n)$ passes for a full $\mathbb R^n\to\mathbb R^m$ map. · 4. A ~15-line dual-number class; sanity-check against `jax.jvp`.
**Key results to state:** forward mode computes $J\mathbf v$ (a column when $\mathbf v=\mathbf e_j$); dual-number rule $(a+b\epsilon)(c+d\epsilon)=ac+(ad+bc)\epsilon$ with $\epsilon^2=0$; full Jacobian in $n$ forward passes.
**Diagrams:** `fig_mdl-fwd-vs-rev` (shared with 2.3.6) — left panel: JVP builds the Jacobian one *column* per pass; with the tall-vs-wide cost rule annotated.
**Worked example(s):** a ~15-line dual-number `Dual` class evaluating $f$ and $f'$ together; cross-check on $f(x,y)$ vs. `jax.jvp`/`torch.func.jvp`.
**Exercises (draft):** 1. Extend the dual-number class to support the product and chain rules. 2. Use it to differentiate $\sin(x^2)$ at a point. 3. How many forward passes for the full Jacobian of $\mathbb R^2\to\mathbb R^{100}$? 4. Show $\epsilon^2=0$ encodes "drop higher-order terms."
**Prereqs / cross-refs:** 2.3.1--2.3.2; :numref:`sec_mdl-single_variable_calculus` (higher-order-term argument); contrast with 2.3.6.
:::

## Reverse-Mode AD, the Tape, and Backprop
:label:`subsec_mdl-reverse-mode`

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
**Body framing:** Reverse-mode AD computes a *vector--Jacobian product* (VJP)---one *row* of the Jacobian per pass---by recording operations on a *tape* during the forward pass and replaying it backward; because a deep-learning loss is *scalar* ($m=1$), a single reverse pass yields the entire gradient, which is precisely **backpropagation = reverse-mode AD = a sequence of VJPs**.
**Outline:** 1. VJP: reverse mode evaluates $\mathbf u^\top J$ for a chosen $\mathbf u$ in one backward pass; seeding $\mathbf u=1$ on a scalar output gives $\nabla f$. · 2. The *tape* (Wengert list): record each primitive and its local Jacobian on the forward pass, then accumulate adjoints backward (this *is* the §2.2 "keep $\partial f$ in the numerator" reordering, now named). · 3. Cost: one pass per *output*, so reverse mode is cheap for **wide** Jacobians ($m\ll n$, few outputs); scalar loss $\Rightarrow$ one pass for all parameters---the whole reason training is feasible. · 4. A ~25-line tape; cross-check against framework autograd. · 5. Memory trade-off: reverse mode must store the forward intermediates (motivates checkpointing).
**Key results to state:** reverse mode computes $\mathbf u^\top J$ (a row when $\mathbf u=\mathbf e_i$); scalar loss ($m=1$) $\Rightarrow$ full gradient in **one** reverse pass; the cost rule "forward for tall, reverse for wide"; backprop = reverse-mode AD.
**Diagrams:** `fig_mdl-fwd-vs-rev` (right panel) — VJP builds the Jacobian one *row* per pass, with the tall/wide decision rule; `fig_mdl-dual-vs-tape` — side-by-side of a dual-number forward sweep vs. a tape-based forward-record / backward-replay, highlighting where each stores information.
**Worked example(s):** a ~25-line reverse-mode `Tape` reproducing the §2.2 backprop example (the $(u+v)^2$ network); cross-check gradients against `torch.autograd`/`jax.grad`/`jax.vjp`.
**Exercises (draft):** 1. For a scalar loss over $n=1000$ params, count forward-mode vs. reverse-mode passes (1000 vs. 1). 2. Implement a VJP for matrix multiply. 3. Why does reverse mode store intermediates but forward mode need not? 4. Extend the toy tape with one new primitive (e.g. $\exp$).
**Prereqs / cross-refs:** 2.3.1--2.3.5; :numref:`sec_mdl-multivariable_calculus` (the backprop example, "keep $\partial f$ in the numerator"); :numref:`sec_autograd` (framework usage---*theory here, usage there*); feeds Ch3.1 (optimization), Ch4.3, and the change-of-variables Jacobian in :numref:`sec_mdl-integral_calculus`.
:::

## Summary

* The **Jacobian** $J_{\mathbf f}\in\mathbb R^{m\times n}$ is the best local linear map of a vector-valued $\mathbf f$; the gradient (one row, $m=1$) and the Hessian (Jacobian of $\nabla f$) are special cases.
* The chain rule in matrix form is **Jacobian composition** $J_{\mathbf g\circ\mathbf f}=J_{\mathbf g}J_{\mathbf f}$, so a deep network's derivative is one long matrix product.
* **Layout conventions** (numerator vs denominator) differ by a transpose; stating which one you use eliminates most stray-transpose bugs.
* **Forward-mode AD** (JVPs, one Jacobian column per pass) is cheap for tall Jacobians; **reverse-mode AD** (VJPs, one row per pass) is cheap for wide ones.
* Because a loss is scalar, **backpropagation is reverse-mode AD**: a single backward sweep over the tape computes the gradient w.r.t. every parameter at the cost of one extra forward pass.

## Exercises

1. Compute the Jacobian of $\mathbf f(x,y)=(x^2y,\ \sin(x+y))$ and verify the linear approximation $\mathbf f(\mathbf x+\boldsymbol\epsilon)\approx\mathbf f(\mathbf x)+J\boldsymbol\epsilon$ at $(1,0)$.
2. Compute $\nabla_{\mathbf x}\|\mathbf A\mathbf x-\mathbf b\|_2^2$ two ways: by the "$1\times1$ guess + transpose" heuristic, and by an index/Einstein-notation derivation.
3. For a scalar loss with $n=1000$ parameters, how many passes does forward mode need to assemble the full gradient, and how many does reverse mode need? Explain the difference via JVPs vs. VJPs.
4. Extend the dual-number construction so that it correctly differentiates a product $g(x)h(x)$ and a composition $g(h(x))$, and connect each rule to §2.1.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/)
:end_tab:
