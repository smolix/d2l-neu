# Stochastic Gradient Methods
:label:`sec_mdl-stochastic-gradient-methods`

<!-- Placeholder opening; the section writer replaces it. -->
A minibatch gradient is right on average but noisy. This section measures
that noise, shows where a fixed step stalls and how a decaying step removes
the floor, states what can be proved without convexity, and shows how
stored gradients cancel the noise on a finite sum.

```{.python .input #mdl-stochastic-gradient-methods-stochastic-gradient-methods}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import numpy as np
```

```{.python .input #mdl-stochastic-gradient-methods-stochastic-gradient-methods}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
```

```{.python .input #mdl-stochastic-gradient-methods-stochastic-gradient-methods}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
```

```{.python .input #mdl-stochastic-gradient-methods-stochastic-gradient-methods}
#@tab jax
%matplotlib inline
from d2l import jax as d2l
import numpy as np
```

## Minibatch Gradients
:label:`subsec_mdl-stochastic-gradients`

### The Cost of Full Gradients

A training loss is an average over data,

$$
f(\mathbf{w}) = \frac{1}{N} \sum_{i=1}^N f_i(\mathbf{w}),
$$

where $f_i$ is the loss on example $i$. The exact gradient
$\nabla f = \tfrac1N \sum_i \nabla f_i$ touches all $N$ examples; at
$N \sim 10^9$, one gradient-descent step costs a full pass over the dataset.
**Stochastic gradient descent** replaces the exact gradient with an estimate
from a random **minibatch** $B$ of $b$ indices, drawn uniformly (for the
analysis, with replacement):

$$
\hat{\mathbf{g}}_b = \frac{1}{b} \sum_{i \in B} \nabla f_i(\mathbf{w}),
\qquad
\mathbf{w}_{k+1} = \mathbf{w}_k - \eta_k\, \hat{\mathbf{g}}_b.
$$
:eqlabel:`eq_mdl-opt-minibatch`

The estimate has two basic properties that recur in minibatch analyses.

**Proposition (minibatch gradients: unbiased, variance $\propto 1/b$).** *Let
$\Sigma$ denote the covariance of a single uniformly drawn example gradient,
$\Sigma = \tfrac1N \sum_i \big(\nabla f_i - \nabla f\big)\big(\nabla f_i - \nabla f\big)^\top$
(all gradients at the current $\mathbf{w}$). Then*

$$
\mathbb{E}\big[\hat{\mathbf{g}}_b\big] = \nabla f(\mathbf{w}),
\qquad
\mathbb{E}\big[\|\hat{\mathbf{g}}_b - \nabla f(\mathbf{w})\|^2\big] = \frac{\mathrm{tr}\, \Sigma}{b}.
$$
:eqlabel:`eq_mdl-opt-variance`

**Proof.** Each index is uniform on $\{1, \ldots, N\}$, so each term has
expectation $\tfrac1N \sum_i \nabla f_i = \nabla f$; by linearity so does the
average. This equality depends on uniform sampling and on conditioning on the
current iterate; biased sampling requires a correction. For the variance, the
$b$ draws are
independent and identically distributed, so the covariance of their average is
$\Sigma / b$ (covariances of independent terms add, and the $1/b$ outside
enters squared). Taking the trace turns covariance into expected squared
error. $\blacksquare$

The expected squared estimation error falls like $1/b$, while its root-mean-square
magnitude falls like $1/\sqrt{b}$: increasing the batch computation by
$100\times$ reduces this magnitude by only $10\times$. This scaling contributes
to diminishing returns from very large batches (exercise :exref:`ex_mdl-gradient-based-optimization-use-eqref-eq-mdl`); the practical
operating point also depends on parallel hardware and optimization dynamics. The cell
verifies both claims on a hand-rolled logistic-regression problem, the toy
we will keep for the rest of the section: $n = 200$ points in five dimensions,
labels flipped with logistic noise, plus a small $\ell_2$ regularizer that
makes the loss strongly convex.

```{.python .input #gradient-based-optimization-sgd-variance}
rng = np.random.default_rng(42)
n, d = 200, 5
w_true = np.array([1.0, -2.0, 1.5, 0.0, 2.0])
X = rng.normal(size=(n, d))
y = np.where(rng.random(n) < 1 / (1 + np.exp(-X @ w_true)), 1.0, -1.0)
lam_reg = 0.1

def loss(w):
    return np.mean(np.log1p(np.exp(-y * (X @ w)))) + 0.5 * lam_reg * w @ w

def example_grads(w):                        # row i: gradient of example i's loss
    s = 1 / (1 + np.exp(y * (X @ w)))        # sigma(-y_i x_i^T w)
    return -(y * s)[:, None] * X + lam_reg * w

G = example_grads(np.zeros(d))
g_full = G.mean(axis=0)
tr_sigma = ((G - g_full)**2).sum(axis=1).mean()   # tr Sigma over the dataset
print(f'single-example gradient noise  tr Sigma = {tr_sigma:.4f}')
for b in [1, 4, 16, 64]:
    batches = rng.integers(0, n, size=(2000, b))  # 2000 minibatches per size
    err = ((G[batches].mean(axis=1) - g_full)**2).sum(axis=1).mean()
    print(f'b = {b:2d}:  measured E|g_hat - g|^2 = {err:.5f}   '
          f'predicted tr Sigma / b = {tr_sigma / b:.5f}')
```

At the initialization the single-example noise level is
$\mathrm{tr}\,\Sigma = 1.1447$, and the measured squared error of the
minibatch gradient tracks the $1/b$ prediction within Monte Carlo error at
every batch size: $1.155$ vs. $1.145$ at $b = 1$, down to $0.0182$ vs.
$0.0179$ at $b = 64$.

## The Noise Floor and Step-Size Decay
:label:`subsec_mdl-step-size-decay`

### Stationary Error under Fixed-Step SGD

What does persistent gradient noise do to convergence? A one-dimensional strongly convex quadratic gives an exact recurrence. Model the update as
$x_{k+1} = x_k - \eta\,(\lambda x_k + \xi_k)$ with i.i.d. noise
$\mathbb{E}[\xi_k] = 0$, $\mathbb{E}[\xi_k^2] = \sigma^2$. Squaring and taking
expectations (the cross term vanishes by unbiasedness),

$$
\mathbb{E}\big[x_{k+1}^2\big] = (1 - \eta\lambda)^2\, \mathbb{E}\big[x_k^2\big] + \eta^2 \sigma^2,
$$

a contraction plus a constant noise injection, whose fixed point is

$$
\mathbb{E}\big[x_\infty^2\big]
= \frac{\eta^2 \sigma^2}{1 - (1 - \eta\lambda)^2}
= \frac{\eta\, \sigma^2}{\lambda\, (2 - \eta\lambda)}
\;\approx\; \frac{\eta\, \sigma^2}{2\lambda}
\qquad (\eta \ll 1/\lambda).
$$
:eqlabel:`eq_mdl-opt-noise-ball`

In this scalar quadratic model, fixed-step SGD does *not* converge to the
minimizer; it converges to a stationary *distribution* whose iterates fluctuate
inside a **noise
ball** of squared radius proportional to $\eta\,\sigma^2$, as in
:numref:`fig_mdl-opt-sgd-noise-ball`. Far from the optimum the contraction
term dominates and SGD makes GD-like linear progress; once the error reaches
the ball, signal and noise balance, and subsequent iterates continue to fluctuate within the stationary distribution. The radius formula is a design guide: halving $\eta$ halves the
squared radius (and so does doubling $b$, since $\sigma^2 \mapsto \sigma^2/b$
by :eqref:`eq_mdl-opt-variance`; step size and batch size are coupled parameters, the coupling behind the learning-rate rules of
:numref:`sec_batch_size`).

![Gradient descent versus SGD on the same strongly convex bowl. The full-batch path converges to the minimizer; fixed-step SGD first descends and then fluctuates inside a noise ball whose squared radius scales like $\eta\sigma^2/(2\lambda)$. Halving the step size shrinks the ball in this model, which motivates decaying schedules under persistent gradient noise.](../img/mdl-opt-sgd-noise-ball.svg)
:label:`fig_mdl-opt-sgd-noise-ball`

Under the persistent-noise assumptions of this model, convergence to the
minimizer requires a decaying step: fast enough to control the noise, slowly
enough to retain cumulative movement. The classical
conditions of :citet:`Robbins.Monro.1951` make both demands precise:

$$
\sum_{k} \eta_k = \infty,
\qquad
\sum_{k} \eta_k^2 < \infty.
$$
:eqlabel:`eq_mdl-opt-robbins-monro`

These step-size conditions are only one part of a convergence theorem. With
unbiased stochastic gradients, controlled second moments, a smooth strongly
convex objective, and a suitable constant in $\eta_k \propto 1/k$, SGD attains
$\mathbb{E}[f(\mathbf{w}_k)] - f^\star = O(1/k)$. For a convex but not strongly
convex objective, appropriately scheduled SGD or averaged iterates attain the
usual $O(1/\sqrt{k})$ stochastic rate; Polyak--Ruppert averaging has sharper
asymptotic conclusions under its own regularity conditions
:cite:`Polyak.Juditsky.1992,Bottou.2010,Goodfellow.Bengio.Courville.2016`.
A condition on the constant is easy to miss: with $\eta_k = c/k$, the
$O(1/k)$ guarantee requires $c$
large enough relative to the curvature, $c > 1/(2\mu)$; choose $c$ too
small and the rate degrades to $O(k^{-2\mu c})$
(exercise :exref:`ex_mdl-gradient-based-optimization-one-over-k-constant`), arbitrarily slower
than advertised, one reason practical schedules decay more gently than $1/k$
:cite:`Bottou.Curtis.Nocedal.2018`. Compare the
deterministic linear rate :eqref:`eq_mdl-opt-gd-rate-strongly-convex`: noise
changes the *rate class*, from geometric to
polynomial. The cell shows both regimes on the logistic toy: fixed steps
approach an $\eta$-proportional floor, whereas the decaying schedule continues
toward zero.

```{.python .input #gradient-based-optimization-sgd-schedule}
w_star = np.zeros(d)
for _ in range(4000):                         # full-batch reference optimum
    w_star = w_star - 0.5 * example_grads(w_star).mean(axis=0)
f_star = loss(w_star)

def sgd(schedule, steps=4000, b=8, seed=1):
    rg = np.random.default_rng(seed)
    w, gaps = np.zeros(d), []
    for k in range(steps):
        idx = rg.integers(0, n, size=b)
        s = 1 / (1 + np.exp(y[idx] * (X[idx] @ w)))
        g = (-(y[idx] * s)[:, None] * X[idx]).mean(axis=0) + lam_reg * w
        w = w - schedule(k) * g
        gaps.append(loss(w) - f_star)
    return np.array(gaps)

gap_fix = sgd(lambda k: 0.8)
gap_half = sgd(lambda k: 0.4)
gap_dec = sgd(lambda k: 0.8 / (1 + k / 100))
print(f'fixed eta = 0.8:     mean gap, last 1000 steps = {gap_fix[-1000:].mean():.2e}')
print(f'fixed eta = 0.4:     mean gap, last 1000 steps = {gap_half[-1000:].mean():.2e}')
print(f'decay 0.8/(1+k/100): mean gap, last 1000 steps = {gap_dec[-1000:].mean():.2e}')
d2l.plot(np.arange(1, 4001), [gap_fix, gap_dec], 'step k', 'optimality gap',
         xscale='log', yscale='log',
         legend=['fixed eta = 0.8', 'decay 0.8/(1+k/100)'])
```

The measurements agree with the theoretical predictions. Fixed $\eta = 0.8$ stalls at
an average gap of $2.3 \times 10^{-2}$; halving the step to $0.4$ halves the
floor to $1.1 \times 10^{-2}$: the linear-in-$\eta$ noise ball of
:eqref:`eq_mdl-opt-noise-ball`, measured. The $1/k$ schedule is at
$5.7 \times 10^{-4}$ after the same $4000$ steps and still descending as a
power law, the straight line on the log-log plot: its early large steps cross
the valley, and its late small steps shrink the ball. Modern schedules (step
decay, cosine, and warmup; :numref:`sec_scheduler`) manage this tradeoff
empirically on losses that are neither convex nor stationary. Warmup starts with a small $\eta$ and ramps
it upward. It can reduce early instability while gradient statistics and local
curvature are changing rapidly, but the quadratic analysis does not establish a
universal warmup rule for neural networks. Schedules and warmup are discussed in
:numref:`sec_mdl-adaptive-stochastic-methods`.

### What Decay Does, and Which Shape

A learning-rate schedule is a sequence $\eta_t$ (the main book surveys the
common ones in :numref:`sec_scheduler`), and the relevant mathematical tradeoff was already proved in
:numref:`sec_mdl-gradient-based-optimization`: on the persistent-noise
quadratic analyzed there, a constant step leaves SGD at a noise floor
proportional to $\eta$ :eqref:`eq_mdl-opt-noise-ball`. Under the assumptions of
the Robbins--Monro theorem, decay satisfying :eqref:`eq_mdl-opt-robbins-monro`
can converge to the optimum, whereas an undersized constant in
$\eta_k = c/k$ can degrade the rate class.

Add to this the Ghadimi--Lan prescription of this section: for a fixed
nonconvex budget $K$, a *constant* $\eta \propto 1/\sqrt{K}$ is already
**minimax-optimal**, meaning no method that sees only the same
unbiased, bounded-variance gradient oracle can guarantee a better worst-case
rate than the $K^{-1/2}$ it achieves; the matching lower bound is due to
:citet:`Arjevani.Carmon.Duchi.ea.2023`. In this analysis, decay has a specific role: it lowers the noise floor, at a cost in transient progress, and
beyond convex problems no theorem ranks one decay *shape* against another.
The following discussion distinguishes proved statements from empirical design
choices. Three common schedules, drawn in
:numref:`fig_mdl-opt-schedule-zoo`, are these:

![Four schedules at equal budget on the noisy quadratic used below. A constant step retains a nonzero noise floor; $c/k$ decay is sensitive to $c$; cosine allocates a long tail to small steps; warmup--stable--decay (WSD) keeps a constant plateau before a final decay. The plotted schedules include linear warmup, a common empirical choice while an adaptive preconditioner is estimated from few gradients.](../img/mdl-opt-schedule-zoo.svg)
:label:`fig_mdl-opt-schedule-zoo`

**Cosine decay** :cite:`Loshchilov.Hutter.2016` sets
$\eta_t = \tfrac12 \eta_0 (1 + \cos(\pi t / K))$: a smooth descent from
$\eta_0$ to $0$ with most of its time spent at moderate steps and a long,
gentle tail. It satisfies no optimality theorem; it is popular because it has
one parameter ($\eta_0$), no kinks to tune, and is widely used across
training budgets. **Warmup--stable--decay (WSD)** :cite:`Hu.Tu.Han.ea.2024` holds
$\eta_0$ constant for most of the run and decays only in a final fraction
(often the last $10$--$20\%$). Its practical appeal is operational: the
long constant plateau means checkpoints from one run can be re-decayed to any
budget, rather than committing to $K$ at launch. On the quadratic model, the
constant plateau permits rapid transient progress but retains a noise floor;
the final decay reduces that floor. This interpretation does not prove that the
plateau is beneficial on a neural-network loss. Its practical advantages and
typical final-decay fraction are empirical choices. The cell
compares the shapes at equal budget on the noisy quadratic, where
the noise floor is exact, printing the loss at $80\%$ of the budget
and at the end:

```{.python .input #adaptive-stochastic-methods-schedules}
lam_s, sigma, K, eta0 = 1.0, 1.0, 2000, 0.3

def run_schedule(eta_fn, seeds=200):
    at80, final = [], []
    for s in range(seeds):
        rg, x = np.random.default_rng(s), 3.0
        for k in range(K):
            x = x - eta_fn(k) * (lam_s * x + sigma * rg.standard_normal())
            if k == int(0.8 * K) - 1:
                at80.append(0.5 * lam_s * x * x)
        final.append(0.5 * lam_s * x * x)
    return np.mean(at80), np.mean(final)

schedules = [
    ('constant', lambda k: eta0),
    ('c/k decay', lambda k: eta0 / (1 + k / 50)),
    ('cosine', lambda k: 0.5 * eta0 * (1 + np.cos(np.pi * k / K))),
    ('WSD 80/20', lambda k: eta0 if k < 0.8 * K else eta0 * (K - k) / (0.2 * K)),
]
print(f'         E f at 80% budget   E f at end     (floor ~ '
      f'{eta0 * sigma**2 / (2 * (2 - eta0 * lam_s)):.3f})')
for name, fn in schedules:
    m80, mK = run_schedule(fn)
    print(f'{name:10s}    {m80:9.2e}      {mK:9.2e}')
```

The constant run sits on its floor at both checkpoints ($9.5$ and
$9.2 \times 10^{-2}$, against the printed prediction
$\eta\sigma^2/(2(2 - \eta\lambda)) \approx 0.088$), while every decaying
schedule ends one to two orders lower. The instructive row is WSD: at $80\%$
of the budget it is *still on the constant's floor* ($9.5 \times 10^{-2}$,
equal to the constant row to every printed digit, since the two runs are the
same trajectory until the decay begins), then the final $400$ steps drop it
by a factor of $17$. All of WSD's visible improvement arrives in the decay
phase. On this quadratic, cosine's longer tail of small steps gives the lower
final error
($1.6 \times 10^{-3}$); this quadratic does not evaluate WSD's operational benefits, such as
re-decayable checkpoints or allocating more steps to a large-learning-rate
phase.

## Nonconvex Losses
:label:`subsec_mdl-nonconvex-sgd`

### Combining Nonconvexity and Gradient Noise

Recall two results from :numref:`sec_mdl-gradient-based-optimization`.
For $L$-smooth $f$ bounded below, *deterministic* gradient descent satisfies
$\min_{k < K} \|\nabla f(\mathbf{x}_k)\|^2 \le 2L(f(\mathbf{x}_0) - f^\star)/K$:
stationarity at rate $O(1/K)$, no convexity needed. And for *stochastic*
gradients on a strongly convex quadratic, a fixed step converges to a noise
ball of squared radius $\approx \eta\sigma^2/(2\lambda)$ around the minimizer
:eqref:`eq_mdl-opt-noise-ball`. Neural-network training combines the two settings: the objective is nonconvex
and the gradients are noisy. Neither preceding result covers both properties. The theorem of :citet:`Ghadimi.Lan.2013` applies
the descent lemma conditionally on the current iterate. Conditional
unbiasedness removes one cross term, while a conditional variance bound controls
the quadratic term. With a biased estimator, an additional inner product
remains and this proof no longer gives the stated rate.

Model one SGD step as $\mathbf{x}_{k+1} = \mathbf{x}_k - \eta\,\mathbf{g}_k$,
where the stochastic gradient is unbiased with bounded variance:

$$
\mathbb{E}\left[\mathbf{g}_k \mid \mathbf{x}_k\right] = \nabla f(\mathbf{x}_k),
\qquad
\mathbb{E}\left[\|\mathbf{g}_k - \nabla f(\mathbf{x}_k)\|^2 \mid \mathbf{x}_k\right] \;\le\; \sigma^2.
$$
:eqlabel:`eq_mdl-opt-sgd-oracle`

The minibatch proposition of :numref:`sec_mdl-gradient-based-optimization`
supplies exactly these hypotheses, with $\sigma^2 = \mathrm{tr}\,\Sigma/b$.

### The Ghadimi--Lan Rate

**Proposition (SGD finds approximate stationary points).** *Let $f$ be
$L$-smooth and bounded below by $f^\star$, write
$\Delta = f(\mathbf{x}_0) - f^\star$, and run SGD with a constant step
$\eta \le 1/L$. Then*

$$
\frac{1}{K} \sum_{k=0}^{K-1} \mathbb{E}\left[\|\nabla f(\mathbf{x}_k)\|^2\right]
\;\le\; \frac{2\Delta}{\eta K} + L\eta\sigma^2,
$$
:eqlabel:`eq_mdl-opt-ghadimi-lan`

*and choosing $\eta = \min\bigl(1/L,\ \sqrt{2\Delta/(L\sigma^2 K)}\bigr)$
balances the two terms:*

$$
\mathbb{E}\left[\|\nabla f(\mathbf{x}_R)\|^2\right]
\;\le\; \frac{2L\Delta}{K} + 2\sigma\,\sqrt{\frac{2L\Delta}{K}},
\qquad R \sim \mathrm{Uniform}\{0, \ldots, K-1\}.
$$
:eqlabel:`eq_mdl-opt-ghadimi-lan-balanced`

**Proof.** Apply the quadratic upper bound
:eqref:`eq_mdl-opt-quadratic-ceiling` to the step actually taken,
$\mathbf{x}_{k+1} - \mathbf{x}_k = -\eta\,\mathbf{g}_k$:

$$
f(\mathbf{x}_{k+1}) \;\le\; f(\mathbf{x}_k) - \eta\, \nabla f(\mathbf{x}_k)^\top \mathbf{g}_k + \tfrac{L\eta^2}{2}\, \|\mathbf{g}_k\|^2.
$$

Take the expectation conditioned on $\mathbf{x}_k$. Unbiasedness turns the
middle term into $-\eta\,\|\nabla f(\mathbf{x}_k)\|^2$, and the variance
decomposition
$\mathbb{E}\|\mathbf{g}_k\|^2 = \|\nabla f(\mathbf{x}_k)\|^2 + \mathbb{E}\|\mathbf{g}_k - \nabla f(\mathbf{x}_k)\|^2$
bounds the last by
$\tfrac{L\eta^2}{2}(\|\nabla f(\mathbf{x}_k)\|^2 + \sigma^2)$. Collecting,

$$
\mathbb{E}\left[f(\mathbf{x}_{k+1}) \mid \mathbf{x}_k\right]
\;\le\; f(\mathbf{x}_k) - \eta\left(1 - \tfrac{L\eta}{2}\right) \|\nabla f(\mathbf{x}_k)\|^2 + \tfrac{L\eta^2 \sigma^2}{2},
$$

the descent lemma with an additional variance term
$\tfrac{L}{2}\eta^2\sigma^2$ per step. For $\eta \le 1/L$ the bracket is at
least $\tfrac12$. Take total
expectations, telescope over $k = 0, \ldots, K-1$ (the left side collapses to
$\mathbb{E}[f(\mathbf{x}_K)] - f(\mathbf{x}_0) \ge -\Delta$), and divide by
$\eta K/2$ to get :eqref:`eq_mdl-opt-ghadimi-lan`. The first term of the bound
falls in $\eta$, the second grows; the stated $\eta$ equalizes them, and the
average over $k$ *is* the expectation at a uniformly random iterate $R$.
$\blacksquare$

Compare the optimized bound with its deterministic counterpart. Without noise
($\sigma = 0$) the second term vanishes and the $2L\Delta/K$ term reproduces
the $O(1/K)$ rate exactly. With noise, the second term dominates for large
$K$, and it decays like $K^{-1/2}$: the stochastic term changes the asymptotic rate from $1/K$
to square-root order, the same square root that governed the minibatch
variance ($1/b$ energy, $1/\sqrt{b}$ amplitude). Be careful about what is
bounded: the theorem controls the **expected squared gradient norm**. Running
$100\times$ longer reduces that bound by $10\times$; interpreting it as a bound
on the gradient norm itself gives only a factor $\sqrt{10}$. A tenfold reduction
of that norm would require $10{,}000\times$ the budget under this worst-case
rate. The prescribed step obeys the same
logic as the noise-ball analysis: a longer budget affords a smaller $\eta$,
reducing the stationary error associated with persistent noise.

One subtlety is a feature of the theorem rather than a defect. The deterministic result bounded $\min_k \|\nabla f(\mathbf{x}_k)\|^2$; when
exact gradients are available, the best iterate can be identified. With
stochastic gradients, $\|\nabla f(\mathbf{x}_k)\|$ is not observed directly,
so selecting the best iterate requires additional estimation. The theorem's output is therefore a *randomly selected* iterate,
whose expected squared gradient norm is the average that
:eqref:`eq_mdl-opt-ghadimi-lan` controls. The following experiment measures this
distinction on a nonconvex toy (a one-neuron $\tanh$ regression with noisy
labels, single-example gradients), sweeping the budget $K$ with the
theory-prescribed $\eta \propto 1/\sqrt{K}$ and measuring both the
random-iterate average and the (non-executable) trajectory minimum of
$\|\nabla f\|^2$, each averaged over $20$ seeds.

```{.python .input #adaptive-stochastic-methods-ghadimi-lan}
n, d = 64, 5
rng = np.random.default_rng(0)
Xd = rng.normal(size=(n, d))
yd = np.tanh(Xd @ rng.normal(size=d)) + 0.5 * rng.normal(size=n)

def full_grad(w):                              # exact gradient, for measurement
    t = np.tanh(Xd @ w)
    return 2 * (Xd * ((t - yd) * (1 - t**2))[:, None]).mean(axis=0)

def stoch_grad(w, i):                          # single-example gradient (b = 1)
    t = np.tanh(Xd[i] @ w)
    return 2 * (t - yd[i]) * (1 - t**2) * Xd[i]

Ks, avg_res, min_res = [125, 500, 2000, 8000], [], []
print('    K    eta_K     E|grad(x_R)|^2   min-so-far')
for K in Ks:
    eta = 0.5 / np.sqrt(K)                     # the theorem's eta ~ 1/sqrt(K)
    avgs, mins = [], []
    for seed in range(20):
        rg, w, tot, m = np.random.default_rng(seed), np.zeros(d), 0.0, np.inf
        for k in range(K):
            g2 = (full_grad(w)**2).sum()
            tot, m = tot + g2, min(m, g2)
            w = w - eta * stoch_grad(w, rg.integers(0, n))
        avgs.append(tot / K), mins.append(m)
    avg_res.append(np.mean(avgs)), min_res.append(np.mean(mins))
    print(f'{K:5d}   {eta:.4f}      {avg_res[-1]:.3e}      {min_res[-1]:.3e}')
print(f'log-log slope, random iterate: '
      f'{np.polyfit(np.log(Ks), np.log(avg_res), 1)[0]:.2f}  (theory: -1/2)')
print(f'log-log slope, min-so-far:     '
      f'{np.polyfit(np.log(Ks), np.log(min_res), 1)[0]:.2f}')
```

The random-iterate column scales as promised: a $64\times$ increase in budget
yields a $12\times$ reduction, log-log slope $-0.60$, close to the
theoretical $-1/2$ for this finite experiment. (The residual steepness comes from the average itself:
the trajectory average still contains the early high-gradient iterates, whose
contribution decays like $1/K$. The transient term $2\Delta/(\eta K)$ does not explain this difference: with the cell's $\eta \propto 1/\sqrt{K}$ it is of the same
$K^{-1/2}$ order as the noise term, so it cannot steepen the slope.) The
min-so-far column falls much faster,
slope $-1.23$: on this example the best iterate improves faster than the worst-case guarantee,
but noisy gradient observations do not directly identify that iterate. The worst-case guarantee here is
$K^{-1/2}$, and the schedule designs of this section
use the random-iterate guarantee rather than the trajectory minimum.

## Variance Reduction
:label:`sec_mdl-variance-reduction`

Minibatching reduces variance by averaging more examples, but a fixed-size
minibatch still has noise at the optimum. A finite training set offers another
possibility: reuse information from earlier gradients as a **control variate**.
Write the empirical objective as

$$
F(\mathbf x)=\frac1n\sum_{i=1}^n f_i(\mathbf x).
$$

At a reference point $\widetilde{\mathbf x}$, compute the full gradient
$\widetilde{\boldsymbol\mu}=\nabla F(\widetilde{\mathbf x})$. The **SVRG**
estimator at the current point is

$$
\mathbf v_i(\mathbf x)
=\nabla f_i(\mathbf x)-\nabla f_i(\widetilde{\mathbf x})
 +\widetilde{\boldsymbol\mu}.
$$
:eqlabel:`eq_mdl-opt-svrg`

For a uniformly sampled index,

$$
\mathbb E_i[\mathbf v_i(\mathbf x)]
=\nabla F(\mathbf x)-\nabla F(\widetilde{\mathbf x})
 +\nabla F(\widetilde{\mathbf x})
=\nabla F(\mathbf x),
$$

so the estimator is unbiased. Its advantage is not the mean but the variance:
when $\mathbf x$ is close to the snapshot, the two component gradients are
highly correlated and their difference is much less variable than either one.
After an inner loop moves too far from the snapshot, refresh the full gradient
and repeat. Under smooth strong convexity, this mechanism yields geometric
convergence with a fixed step size, rather than SGD's fixed-step noise floor
:cite:`Bottou.Curtis.Nocedal.2018`.

The cell compares SVRG with SGD on ridge regression. One SVRG epoch pays $n$
component-gradient equivalents for the full gradient and two per inner step;
SGD receives the same total budget of $3n$ component gradients per reported
epoch. Both use a fixed step size, so the comparison isolates the control
variate rather than a schedule.

```{.python .input #mdl-adaptive-svrg}
rng = np.random.default_rng(4)
n_vr, d_vr = 300, 12
X_vr = rng.standard_normal((n_vr, d_vr))
w_true_vr = rng.standard_normal(d_vr)
y_vr = X_vr @ w_true_vr + 0.5 * rng.standard_normal(n_vr)
lam_vr = 0.1
H_vr = X_vr.T @ X_vr / n_vr + lam_vr * np.eye(d_vr)
w_star_vr = np.linalg.solve(H_vr, X_vr.T @ y_vr / n_vr)

def component_grad(w, i):
    return X_vr[i] * (X_vr[i] @ w - y_vr[i]) + lam_vr * w

def full_grad(w):
    return X_vr.T @ (X_vr @ w - y_vr) / n_vr + lam_vr * w

def objective_vr(w):
    return (0.5 * np.mean((X_vr @ w - y_vr)**2)
            + 0.5 * lam_vr * (w @ w))

eta_vr, epochs_vr = 0.05, 20
snapshot, gaps_svrg = np.zeros(d_vr), []
for _ in range(epochs_vr):
    mean_grad = full_grad(snapshot)
    w = snapshot.copy()
    for _ in range(n_vr):
        i = rng.integers(n_vr)
        v = component_grad(w, i) - component_grad(snapshot, i) + mean_grad
        w -= eta_vr * v
    snapshot = w
    gaps_svrg.append(objective_vr(snapshot) - objective_vr(w_star_vr))

rng_sgd = np.random.default_rng(5)
w_sgd, gaps_sgd = np.zeros(d_vr), []
for _ in range(epochs_vr):
    for _ in range(3 * n_vr):               # same component-gradient budget
        i = rng_sgd.integers(n_vr)
        w_sgd -= eta_vr * component_grad(w_sgd, i)
    gaps_sgd.append(objective_vr(w_sgd) - objective_vr(w_star_vr))

d2l.plot(np.arange(1, epochs_vr + 1), [gaps_sgd, gaps_svrg],
         'equal-budget epoch', 'optimality gap',
         legend=['SGD, fixed step', 'SVRG, fixed step'], yscale='log')
```

SGD reduces the objective quickly at first, but its constant-step iterates
continue to fluctuate around the optimum. SVRG incurs the cost of its snapshots and then drives the gap
down geometrically on this finite, strongly convex problem. The result does not establish that SVRG universally outperforms SGD: full passes are expensive on enormous or
streaming datasets, the linear-rate theorem uses finite-sum smoothness and
convexity, and stale control variates are less effective when the objective
changes during training.

**SAGA** removes the periodic full-gradient pass by storing the most recently
seen gradient for every example and using their table average as the control
variate. It spends one new component gradient per step but stores $O(nd)$
numbers in the literal form (structure can reduce this for generalized linear
models). SARAH-style estimators update the control variate recursively. The methods share one control-variate principle: correlate a noisy new
estimate with a stored estimate whose mean is known, subtract the shared noise,
and add the known mean back.

Variance reduction is distinct from momentum. Momentum averages gradients
across recent iterates and can reduce high-frequency noise, but its moving
average is generally biased for the gradient at the current point. SVRG and
SAGA engineer an estimator that is unbiased at the current point and whose
variance shrinks as the optimization converges.

## Summary

* Minibatch gradients are **unbiased** with variance
  $\mathrm{tr}\,\Sigma / b$. On the scalar quadratic model, fixed-step SGD
  has a **noise ball** of squared radius
  $\approx \eta\sigma^2/(2\lambda)$. Under the usual stochastic
  approximation assumptions, Robbins--Monro decay
  ($\sum \eta_k = \infty$, $\sum \eta_k^2 < \infty$) permits convergence
  to the optimum.
* The **Ghadimi--Lan theorem** extends the no-convexity guarantee to
  SGD: with $\eta \le 1/L$ the descent lemma survives in expectation with a
  variance term $\tfrac{L}{2}\eta^2\sigma^2$ per step, and at the balanced
  $\eta \propto 1/\sqrt{K}$ a random iterate satisfies
  $\mathbb{E}\|\nabla f\|^2 = O(\sigma\sqrt{L\Delta/K})$. Noise turns the
  deterministic $1/K$ rate into $K^{-1/2}$, and the
  guarantee is for a *random* iterate (the best one cannot be identified
  from noisy evaluations).
* **Schedules** balance the noise floor ($\propto \eta$) against transient
  progress; beyond convexity no theorem ranks decay shapes. Cosine spends a
  long small-step tail; **WSD** stays on the constant-step floor until a
  final decay phase reduces it sharply, as illustrated on a noisy
  quadratic.
* On finite-sum objectives, **SVRG** subtracts a component gradient at a
  snapshot and adds the snapshot's full gradient, producing an unbiased
  control-variate estimator whose variance shrinks near the snapshot. **SAGA**
  replaces full-gradient refreshes by a table of stored component gradients.
  These methods can reach linear fixed-step convergence under smooth strong
  convexity, but require repeated access to a finite dataset.

## Exercises

1. [code] **Noise scale and critical batch size.** ● How much does one
   minibatch step lower the loss, and how does the gain depend on $b$? Write
   $\mathbf{g} = \nabla f(\mathbf{w})$ and $H = \nabla^2 f(\mathbf{w})$, and
   approximate the loss after the step
   $\mathbf{w} - \eta\,\hat{\mathbf{g}}_b$ of :eqref:`eq_mdl-opt-minibatch`
   by its second-order Taylor expansion at $\mathbf{w}$. The analysis follows
   :citet:`McCandlish.Kaplan.Amodei.ea.2018`.
    1. Use the covariance $\Sigma/b$ of $\hat{\mathbf{g}}_b$, established in
       the proof of :eqref:`eq_mdl-opt-variance`, to show that the expected
       loss after the step is minimized by
       $\eta_{\textrm{opt}}(b) = \eta_{\max} / (1 + B_{\textrm{noise}}/b)$,
       with expected improvement $\Delta_{\max} / (1 + B_{\textrm{noise}}/b)$,
       where

       $$
       \eta_{\max} = \frac{\|\mathbf{g}\|^2}{\mathbf{g}^\top H \mathbf{g}},
       \qquad
       \Delta_{\max} = \frac{\|\mathbf{g}\|^4}{2\,\mathbf{g}^\top H \mathbf{g}},
       \qquad
       B_{\textrm{noise}} = \frac{\mathrm{tr}(H \Sigma)}{\mathbf{g}^\top H \mathbf{g}}.
       $$

    1. The linear scaling rule of :citet:`Goyal.Dollar.Girshick.ea.2017`
       multiplies $\eta$ by $s$ when it multiplies $b$ by $s$. For which
       batch sizes does this rule follow from $\eta_{\textrm{opt}}(b)$? What
       does a larger batch buy per step once $b \gg B_{\textrm{noise}}$?
    1. For $H$ proportional to the identity, $B_{\textrm{noise}}$ reduces to
       $B_{\textrm{simple}} = \mathrm{tr}\,\Sigma / \|\mathbf{g}\|^2$. Compute
       $B_{\textrm{simple}}$ and $B_{\textrm{noise}}$ for the logistic loss
       of the `#gradient-based-optimization-sgd-variance` cell at
       $\mathbf{w} = \mathbf{0}$ and after $5$, $10$, and $20$ full-batch
       gradient steps with $\eta = 0.5$, using `example_grads` and the
       Hessian formula of the `#gradient-based-optimization-newton` cell.
       How does the noise scale change as the iterates approach the optimum?
       Treat $B_{\textrm{noise}}$ as the critical batch size that separates
       the two regimes of (b). How should the batch size change over a
       training run?
   :exlabel:`ex_mdl-gradient-based-optimization-use-eqref-eq-mdl`

1. [code] **The constant in a $1/k$ schedule.** ● With
   $\eta_k \propto 1/k$, SGD attains the $O(1/k)$ rate only when the
   constant is large enough relative to
   $\mu$ (:numref:`subsec_mdl-stochastic-gradients`). The scalar model of
   :eqref:`eq_mdl-opt-noise-ball` with curvature $\lambda = \mu$ and steps
   $\eta_k = c/(k + k_0)$ makes this condition exact: the second moment
   $e_k = \mathbb{E}[x_k^2]$ obeys

   $$
   e_{k+1} = (1 - \eta_k \mu)^2\, e_k + \eta_k^2 \sigma^2.
   $$

    1. Let $2\mu c > 1$. Substitute $e_k \approx A/k$ for large $k$, keep
       the terms of order $1/k^2$, and solve for $A$. Which $c$ minimizes
       $A$?
    1. Let $2\mu c < 1$. Show that $e_k = \Theta(k^{-2\mu c})$.
    1. Iterate the recursion with $\mu = \sigma = 1$, $k_0 = 10$, and
       $e_0 = 1$ up to $k = 10^6$ for $c \in \{0.1, 0.25, 0.5, 1, 2\}$, and
       estimate the slope of $\log e_k$ against $\log k$ between $k = 10^5$
       and $k = 10^6$. Which costs more, a constant twice as large as the
       minimizer of $A$ or one half as large?
   :exlabel:`ex_mdl-gradient-based-optimization-one-over-k-constant`

1. [code] **SVRG inner loop and SAGA.** Use the data and functions of the
   `#mdl-adaptive-svrg` cell (`component_grad`, `full_grad`,
   `objective_vr`) with a budget of $60n = 18000$ component-gradient
   evaluations, the budget behind the cell's $20$ reported epochs. Measure
   each run by the number of evaluations it needs to bring the optimality
   gap below $10^{-10}$, checking once per epoch.
    1. Let the inner loop run $m$ steps, so that one epoch costs $n + 2m$
       evaluations, and vary $m \in \{10, 30, 75, 300, 1200\}$ at
       $\eta \in \{0.01, 0.05\}$. Which $m$ is best at each step size?
       Explain the trade-off between amortizing the snapshot gradient and
       letting the control variate go stale, and why the best $m$ moves
       with $\eta$.
    1. Implement SAGA: keep a table with the most recent gradient of every
       component, filled at $\mathbf{w}_0$ for $n$ evaluations, and use
       $\nabla f_i(\mathbf{w}) - \mathbf{t}_i + \tfrac1n \sum_j \mathbf{t}_j$
       as the estimator, where $\mathbf{t}_i$ is the table entry of example
       $i$. Compare SAGA with the best SVRG run at both step sizes, and state
       the memory that its table needs for this problem.
   :exlabel:`ex_mdl-adaptive-stochastic-methods-svrg-inner-loop`

1. **Biased gradient estimators.** ● Keep the variance bound of
   :eqref:`eq_mdl-opt-sgd-oracle`, now measured around the conditional mean,
   but replace unbiasedness by
   $\mathbb{E}[\mathbf{g}_k \mid \mathbf{x}_k] = \nabla f(\mathbf{x}_k) + \mathbf{b}_k$
   with $\|\mathbf{b}_k\| \le \delta$.
    1. Redo the proof of :eqref:`eq_mdl-opt-ghadimi-lan` for
       $\eta \le 1/L$ and show

       $$\frac{1}{K} \sum_{k=0}^{K-1} \mathbb{E}\left[\|\nabla f(\mathbf{x}_k)\|^2\right] \;\le\; \frac{2\Delta}{\eta K} + L\eta\sigma^2 + \delta^2.$$

       Use $L\eta \le 1$ on the term
       $\tfrac{L\eta^2}{2}\|\nabla f(\mathbf{x}_k) + \mathbf{b}_k\|^2$
       before expanding it.
    1. Show that the term $\delta^2$ cannot be removed: give a
       one-dimensional $f$ and a constant bias for which SGD with $\sigma = 0$
       converges to a point where $|f'|^2 = \delta^2$.
    1. A single minibatch drawn uniformly without replacement is still an
       unbiased estimator of the full gradient. Which of the following
       estimators are biased in this sense: the example gradient
       at a step of random reshuffling (:numref:`sec_sgd`), which visits
       every example once per epoch in a fresh random order; a stochastic
       gradient clipped to norm at most $c$; the gradient evaluated at the
       iterate of $\tau$ steps earlier, as in asynchronous training?

## Discussions

[Discussions](https://d2l.discourse.group/)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §26.3]{.kicker}

Stochastic gradient methods<br>**minibatch noise · noise floor · step-size decay · nonconvex rate · variance reduction**.
:::
:::

::: {.slide title="Minibatch noise: unbiased, variance $\propto 1/b$"}
[Stochastic gradients]{.kicker}

A training loss averages over data, so the exact gradient costs a full
pass. A random minibatch estimate is unbiased with variance that falls
like $1/b$:

$$\mathbb{E}[\hat{\mathbf{g}}_b] = \nabla f, \qquad
\mathbb{E}\big[\|\hat{\mathbf{g}}_b - \nabla f\|^2\big] = \frac{\mathrm{tr}\,\Sigma}{b}.$$

::: {.d2l-note}
Noise *energy* falls like $1/b$, so *amplitude* falls like $1/\sqrt{b}$:
$100\times$ the compute reduces the noise only $10\times$, which is why
huge batches show diminishing returns.
:::
:::

::: {.slide title="Fixed-step SGD has a nonzero noise floor"}
[Stochastic gradients]{.kicker}

::: {.cols .vc}
::: {.col .narrow}
A contraction plus constant noise injection has a nonzero fixed point:
In the scalar quadratic model, SGD first descends and then fluctuates with
squared radius

$$\mathbb{E}[x_\infty^2] \approx \frac{\eta\,\sigma^2}{2\lambda}.$$

Halving $\eta$ halves the squared radius in this model; decaying schedules use
this tradeoff under persistent noise.
:::

::: {.col .fig .big}
@fig:mdl-opt-sgd-noise-ball
:::
:::
:::

::: {.slide title="Step-Size Decay under Persistent Noise"}
[Stochastic gradients]{.kicker}

Robbins--Monro decay ($\sum\eta_k=\infty,\ \sum\eta_k^2<\infty$) quenches
the noise while still travelling far, on the logistic toy:

@!gradient-based-optimization-sgd-schedule

Fixed $\eta$ approaches a nonzero floor; $1/k$ continues toward zero: noise moves the *rate
class* from geometric to polynomial.
:::

::: {.slide title="Schedules Balance Transient Progress and Stationary Error"}
[Schedules]{.kicker}

::: {.cols .vc}
::: {.col}
On the persistent-noise quadratic, a constant step has a nonzero stationary error
$\propto \eta$. Under Robbins--Monro assumptions, a suitable decaying step can
converge to the optimum; beyond convexity *no theorem ranks decay shapes*.

- **cosine**: smooth decay with a long tail of small steps
- **WSD**: hold the constant phase, then reduce the step in a final decay;
  checkpoints stay re-decayable to any budget
:::

::: {.col .fig .big}
@fig:mdl-opt-schedule-zoo
:::
:::
:::

::: {.slide title="WSD improves during its decay phase"}
[Schedules]{.kicker}

On the noisy quadratic, where the stationary error is known exactly,
at $80\%$ of budget and at the end:

@!adaptive-stochastic-methods-schedules

WSD matches the constant schedule at $80\%$ (the trajectories agree to every
printed digit), then $400$ decay steps reduce it by $17\times$. Cosine's longer tail
gives the lower final error on this quadratic; WSD remains useful for
re-decayable checkpoints and time spent at large steps.
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[SGD without convexity]{.dtitle}

[the descent lemma with stochastic gradients]{.dsub}
:::
:::

::: {.slide title="Conditional assumptions add a variance term"}
[Ghadimi--Lan]{.kicker}

Condition on $\mathbf{x}_k$: unbiasedness removes the linear cross term, and
the conditional variance bound contributes $L\eta^2\sigma^2/2$. A biased
gradient estimator leaves an additional term and falls outside this result.

. . .

Telescoping the resulting inequality and balancing the two terms with
$\eta \propto 1/\sqrt{K}$:

$$\mathbb{E}\bigl[\|\nabla f(\mathbf{x}_R)\|^2\bigr]
\;\le\; \frac{2L\Delta}{K} + 2\sigma\sqrt{\frac{2L\Delta}{K}},
\qquad R \sim \mathrm{Uniform}\{0, \ldots, K-1\}.$$
:::

::: {.slide title="Stochastic Noise Gives a Square-Root Rate"}
[Ghadimi--Lan]{.kicker}

At $\sigma = 0$ the deterministic $O(1/K)$ returns; with noise the
$K^{-1/2}$ term rules: $10\times$ smaller gradients cost $100\times$
the budget. Measured on a nonconvex toy, $20$ seeds per budget:

@!adaptive-stochastic-methods-ghadimi-lan

::: {.d2l-note}
The guarantee is for a **randomly selected** iterate: with noisy
evaluations you can never *identify* the best one, and the min-so-far
column (slope $-1.23$) is not directly selectable from noisy observations.
:::
:::
