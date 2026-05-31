# Numerical Stability and Conditioning
:label:`sec_mdl-numerical-stability-conditioning`

The math can be right and your loss can still go to `NaN`. This short section
explains the floating-point failure modes that bite real training runs — overflow,
underflow, and catastrophic cancellation — and the handful of two-line fixes that
keep softmax, cross-entropy, and ill-conditioned least squares alive:
max-subtraction, log-space arithmetic, and ridge regularization. It also closes a
loop opened earlier in the chapter: the *condition number* $\kappa$ that set
gradient descent's convergence rate in `sec_mdl-gradient-based-optimization` is the
*same* $\kappa = \sigma_{\max}/\sigma_{\min}$ that governs numerical error
amplification — one number, two consequences. The payoffs land downstream: the
log-space trick rescues naive Bayes in `sec_mdl-naive_bayes` from underflow, and the
stable cross-entropy here is the same computation analyzed in
`sec_mdl-information_theory`.

::: {.callout-note title="⟢ Planned — outline only (not yet written)"}
This file is a teaching outline / detailed table of contents, not finished prose.
Intentionally **short** (§3.4). It fixes the subsection flow, key results, diagrams,
worked examples, and draft exercises. Body framing: explain the failure modes and
the fixes so later chapters can simply *use* them and point back here.
:::

**Prerequisites:** `sec_mdl-gradient-based-optimization` (condition number ↔ GD
speed), `sec_mdl-svd-low-rank` ($\kappa=\sigma_{\max}/\sigma_{\min}$),
`sec_mdl-convexity` (log-sum-exp convexity). Standard reference: Goldberg's *What
Every Computer Scientist Should Know About Floating-Point Arithmetic* (1991, cited
descriptively); :cite:`Goodfellow.Bengio.Courville.2016` (ch. 4) for the ML framing.

::: {.callout-note title="⟢ 3.4.1 Floating Point and Machine Epsilon"}
**Outline:** 1. Floating point = sign · mantissa · $2^{\text{exponent}}$ with a
*finite* mantissa, so most reals are stored only approximately. · 2. Machine epsilon
$\varepsilon_{\text{mach}}$ as the gap between $1$ and the next representable number —
the relative-error floor of every operation. · 3. Format table: fp32, fp16, bfloat16
— and why fp16's tiny mantissa *and* narrow exponent range make it amplify
everything (motivates loss scaling / mixed precision). · 4. Overflow and underflow
thresholds: where $e^x$ becomes $\infty$ or $0$.
**Key results to state:** representable as $(-1)^s\cdot(1.m)\cdot2^{e}$; fp32
$\varepsilon_{\text{mach}}=2^{-23}\approx1.2\times10^{-7}$ (round-to-nearest unit
roundoff $2^{-24}$), fp16 $\varepsilon_{\text{mach}}=2^{-10}\approx10^{-3}$, bfloat16
$2^{-8}$ (same exponent range as fp32, fewer mantissa bits); fp32 overflow $\approx
3.4\times10^{38}$, smallest normal $\approx1.2\times10^{-38}$; fp16 overflows around
$x\approx 11$ in $e^x$, fp32 around $x\approx 88$.
**Diagrams:** `fig_mdl-over-under-flow-line` — a real number line with representable
fp values (denser near $0$, sparser far out) and the overflow/underflow cliffs at
the extremes, annotated with fp16 vs. fp32 thresholds.
**Worked example(s):** print `finfo(float32).eps`, `.max`, `.tiny` for fp32/fp16/bf16;
show $1 + \varepsilon/2 = 1$ in fp32; find the smallest $x$ with $e^x=\infty$ in each
format.
**Exercises (draft):** (1) Compute $\varepsilon_{\text{mach}}$ for fp32 by a doubling
loop. (2) Find logits where $e^x$ overflows in fp16 but not fp32. (3) Explain why
bfloat16 trades mantissa bits for exponent range and when that helps training.
:::

::: {.callout-note title="⟢ 3.4.2 Log-Sum-Exp and Stable Softmax / Cross-Entropy"}
**Outline:** 1. The problem: softmax exponentiates logits, so a logit of $100$
overflows fp32 long before normalization. · 2. The fix: subtracting any constant $c$
from every logit leaves softmax *unchanged*; choosing $c=\max_i x_i$ makes the
largest exponent exactly $1$, so nothing overflows and the denominator never
underflows to $0$. · 3. Log-sum-exp identity and the *fused* `log_softmax` /
cross-entropy that never materializes the raw probabilities (so $\log 0$ never
happens). · 4. This is exactly why frameworks expose `logsumexp`, `log_softmax`, and
"from-logits" loss functions — and why you should pass logits, not probabilities.
**Key results to state:** $\mathrm{softmax}(\mathbf{x})_i$ invariant under
$\mathbf{x}\mapsto\mathbf{x}-c\mathbf{1}$; stable form
$\mathrm{lse}(\mathbf{x})=c+\log\sum_i e^{x_i-c}$ with $c=\max_i x_i$;
$\log\mathrm{softmax}(\mathbf{x})_i = x_i-\mathrm{lse}(\mathbf{x})$; cross-entropy
$-\log\mathrm{softmax}(\mathbf{x})_y = \mathrm{lse}(\mathbf{x})-x_y$ computed directly
from logits.
**Diagrams:** `fig_mdl-softmax-max-subtraction` — two side-by-side computation paths
on the same large logits: the naive path producing `inf`/`NaN`, the max-subtracted
path producing the correct probabilities, with the subtracted $\max$ highlighted.
**Worked example(s):** naive `exp(logits)` $\to$ `inf` vs. max-subtracted on logits
like $[1000, 1001, 1002]$; verify `lse` and `log_softmax` against the framework's
fused ops; show the from-logits CE matches but the from-probabilities path NaNs.
**Exercises (draft):** (1) Prove max-subtraction leaves softmax unchanged. (2)
Construct logits where fp32 softmax overflows and confirm the fix. (3) Show
$\mathrm{lse}$ is convex (ties to `sec_mdl-convexity` §3.2.5) and that its gradient is
the softmax. Forward-links: underflow payoff in `sec_mdl-naive_bayes`; cross-entropy
framing in `sec_mdl-information_theory`.
:::

::: {.callout-note title="⟢ 3.4.3 Catastrophic Cancellation"}
**Outline:** 1. The failure mode: subtracting two nearly-equal floating-point numbers
annihilates the leading significant digits, leaving relative error blown up by orders
of magnitude. · 2. Canonical examples: the naive "$\mathbb{E}[X^2]-\mathbb{E}[X]^2$"
variance formula; $\log(1-\mathrm{softmax})$; the quadratic formula near a double
root. · 3. The fix is *reformulation*, not higher precision: Welford's one-pass mean
correction for variance; `log1p`/`expm1` for $\log(1+x)$ and $e^x-1$ near $0$. · 4.
General principle: keep subtractions away from the result of catastrophic round-off.
**Key results to state:** relative error of $a-b$ with $a\approx b$ amplified by
$\approx |a|/|a-b|$; naive variance $\tfrac1n\sum x_i^2-\bar{x}^2$ can go *negative*;
Welford update $M_k = M_{k-1}+\tfrac{(x_k-M_{k-1})(x_k-M_k)}{1}$ accumulating
$\sum(x_i-\bar x)^2$ stably; $\log(1+x)\to$ `log1p(x)` for $|x|\ll1$.
**Diagrams:** none new — an inset table of "naive vs. reformulated, error vs. true"
suffices (described, not a fig).
**Worked example(s):** sample variance of $\{10^8, 10^8+1, 10^8+2\}$ — naive formula
loses all precision (or goes negative), Welford is exact; $\log(1+10^{-10})$ via
naive `log(1+x)` vs. `log1p`.
**Exercises (draft):** (1) Construct data where the naive variance is negative; fix
with Welford. (2) Show $|a|/|a-b|$ is the cancellation amplification factor. (3)
Rewrite $\sqrt{x+1}-\sqrt{x}$ for large $x$ to avoid cancellation.
:::

::: {.callout-note title="⟢ 3.4.4 Conditioning Revisited"}
**Outline:** 1. Define the condition number of a linear system /matrix
$\kappa(A)=\sigma_{\max}/\sigma_{\min}$ (tie to `sec_mdl-svd-low-rank`). · 2. The
forward-error bound: solving $A\mathbf{x}=\mathbf{b}$ amplifies relative input error
by up to $\kappa$ — so $\kappa\approx10^{k}$ costs you $k$ significant digits. · 3.
The punchline of the chapter: this is the **same $\kappa$** that set GD's contraction
$(\kappa-1)/(\kappa+1)$ in `sec_mdl-gradient-based-optimization` §3.1.3 — one number
controls both the *speed* of optimization and the *accuracy* of the linear algebra.
· 4. Why forming $A^\top A$ (normal equations) is dangerous: it *squares* the
condition number; prefer QR/SVD solves.
**Key results to state:** $\kappa(A)=\sigma_{\max}/\sigma_{\min}=\|A\|\,\|A^{-1}\|$;
relative error bound $\tfrac{\|\delta\mathbf{x}\|}{\|\mathbf{x}\|}\le\kappa(A)\tfrac{\|\delta\mathbf{b}\|}{\|\mathbf{b}\|}$;
$\kappa(A^\top A)=\kappa(A)^2$; loses $\approx\log_{10}\kappa$ digits.
**Diagrams:** reuse the ill-conditioned-contours picture from `sec_mdl-svd-low-rank`
/ `sec_mdl-gradient-based-optimization` (the elongated valley) by reference; no new
SVG.
**Worked example(s):** build a near-singular $A$, solve $A\mathbf{x}=\mathbf{b}$ via
`solve`, normal equations, and `lstsq`/SVD; report `cond` and the lost digits;
connect the same $\kappa$ to the GD iteration count from §3.1.3.
**Exercises (draft):** (1) Show $\kappa(A^\top A)=\kappa(A)^2$ and conclude why normal
equations are risky. (2) Relate digits lost to $\log_{10}\kappa$ on a Hilbert
matrix. (3) Verify the same $\kappa$ predicts the GD rate and the solve error.
:::

::: {.callout-note title="⟢ 3.4.5 Regularization as Conditioning"}
**Outline:** 1. Ridge adds $\lambda I$ to $A^\top A$ (or to the Hessian). · 2.
Spectral effect: every eigenvalue shifts up by $\lambda$, so $\sigma_{\min}$ moves
away from $0$ and $\kappa$ *drops* — a singular or near-singular system becomes
solvable. · 3. Double payoff: lower $\kappa$ means both a *better-conditioned* solve
(this section) and *faster* GD (§3.1.3) — regularization is conditioning. · 4. The
connection back to `sec_mdl-constrained-optimization-duality` §3.3.4: the ridge
penalty $\lambda\|\mathbf{w}\|^2$ is the dual/Lagrangian form of a norm constraint
$\|\mathbf{w}\|\le t$ — penalty and constraint are two views of the same problem.
**Key results to state:** ridge solution $\mathbf{w}=(A^\top A+\lambda I)^{-1}A^\top\mathbf{b}$;
eigenvalues $\lambda_i\mapsto\lambda_i+\lambda$, so $\kappa\mapsto\tfrac{\sigma_{\max}^2+\lambda}{\sigma_{\min}^2+\lambda}<\kappa(A)^2$;
makes a rank-deficient $A^\top A$ invertible; GD contraction improves correspondingly.
**Diagrams:** none new — an eigenvalue-shift bar chart (before/after $+\lambda$) can
be rendered inline (described, not a fig).
**Worked example(s):** singular $A^\top A$ — plain solve fails, ridge solve succeeds;
plot $\kappa$ and the GD iteration count as functions of $\lambda$, showing both drop
together.
**Exercises (draft):** (1) Show $A^\top A+\lambda I$ is always invertible for
$\lambda>0$. (2) Compute $\kappa$ before/after ridge and the resulting GD-rate
change. (3) Show the ridge penalty is the Lagrangian of an $\ell_2$-ball constraint
(ties to §3.3.4).
:::

## Summary

*Planned.* Floating point has finite precision ($\varepsilon_{\text{mach}}$),
overflow/underflow cliffs, and catastrophic cancellation when subtracting near-equal
numbers. The fixes are reformulations: max-subtraction and log-space for
softmax/cross-entropy, Welford and `log1p` against cancellation. The condition number
$\kappa=\sigma_{\max}/\sigma_{\min}$ is one number with two consequences — error
amplification in linear solves and GD's convergence rate — and ridge regularization
lowers $\kappa$, improving both at once, while being the Lagrangian dual of a norm
constraint.

## Exercises

*Planned — consolidated.* (1) Max-subtraction leaves softmax unchanged. (2) Logits
where fp32 softmax overflows. (3) $\kappa$ before/after ridge $\to$ GD rate. (4)
One-pass variance cancellation and the Welford fix.

## Discussions

*Planned placeholder.* Ties to `sec_mdl-svd-low-rank` and
`sec_mdl-gradient-based-optimization` (both via $\kappa$) and to
`sec_mdl-constrained-optimization-duality` (penalty↔constraint); forward payoffs in
`sec_mdl-naive_bayes` (log-space underflow fix) and `sec_mdl-information_theory`
(stable cross-entropy). The main book applies these inside loss layers and mixed
precision.

<!-- slides -->

*Planned.* Slide deck to be authored once body cells exist, with `@<id>`
placeholders for the max-subtraction NaN-vs-correct demo, the Welford cancellation
example, and the ridge $\kappa$-before/after plot.
