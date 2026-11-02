# Uniform Convergence and Generalization
:label:`sec_mdl-generalization-bounds`

This opening is a placeholder from the restructuring pass; the section writer replaces it.

```{.python .input #mdl-generalization-bounds-uniform-convergence-and-generalization}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import numpy as onp
```

```{.python .input #mdl-generalization-bounds-uniform-convergence-and-generalization}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as onp
```

```{.python .input #mdl-generalization-bounds-uniform-convergence-and-generalization}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as onp
```

```{.python .input #mdl-generalization-bounds-uniform-convergence-and-generalization}
#@tab jax
%matplotlib inline
from d2l import jax as d2l
import numpy as onp
```

## From One Estimate to Uniform Convergence

### Data-Dependent Function Selection

We now apply the result to learning. Hoeffding's inequality certifies the
empirical risk of **one fixed function**:
choose $f$, *then* draw the sample $S=\{(\mathbf x_i,y_i)\}_{i=1}^n$, and the
empirical risk $\hat R(f)=\frac1n\sum_i \ell(f(\mathbf x_i),y_i)$ is within
$t$ of the true risk $R(f)=E[\ell(f(\mathbf x),y)]$ except with probability
$2e^{-2nt^2}$. A test set works exactly because it respects this order: the
model was frozen before the test data were drawn. But a *learner* violates the
order by construction: it searches a class $\mathcal F$ and returns the
$\hat f$ that makes $\hat R$ small *on the sample it saw*. The empirical risk
of $\hat f$ is a minimum of fluctuating quantities, selected precisely where
the fluctuation is most favorable, and its optimism is not covered by any
single-function bound. The quantity that must concentrate is the *worst case
over the class*,

$$
\sup_{f\in\mathcal F}\ \bigl|\hat R(f) - R(f)\bigr|,
$$

because if this supremum is at most $t$ then *every* function's empirical risk
is trustworthy to $t$, including whichever one the learner happens to pick.
Bounds on this supremum are called **uniform convergence** bounds, and they
are the classical guarantees that :numref:`sec_generalization_deep` finds
vacuous for a trained MLP.
The *test-set reuse* discussed in
:numref:`chap_classification_generalization` is the same trap: an analyst who
evaluates many models on one test set and reports the best is a learner whose
"class" is the set of models tried. The bound we prove next quantifies this
when the pool of models is fixed *before* the test data are drawn; what
happens when it is not, we take up right after the proof.

### Finite Classes: the Union Bound

For a finite class, the union bound supplies a simultaneous guarantee.

**Proposition (finite-class uniform convergence).** *Let $\mathcal F$ be
finite, let the loss take values in $[0,1]$, and let $S$ be an i.i.d. sample
of size $n$. Then with probability at least $1-\delta$,*

$$
\bigl|\hat R(f) - R(f)\bigr|
 \le \sqrt{\frac{\log(2|\mathcal F|/\delta)}{2n}}
\qquad\textrm{simultaneously for every } f\in\mathcal F .
$$
:eqlabel:`eq_mdl-finite-class`

**Proof.** For each fixed $f$, $\hat R(f)$ is an average of $n$ i.i.d. terms
in $[0,1]$, so Hoeffding :eqref:`eq_mdl-hoeffding` gives
$P(|\hat R(f)-R(f)|\ge t)\le 2e^{-2nt^2}$. The probability that *any* of the
$|\mathcal F|$ bad events occurs is at most the sum of their probabilities
(the union bound again), i.e. $2|\mathcal F|e^{-2nt^2}$. Setting this to
$\delta$ and solving for $t$ gives the claim. $\blacksquare$

The effect of selecting from a class appears as $\log|\mathcal F|$.
Guaranteeing one function puts $\log(2/\delta)$ in the numerator; allowing a
choice among $|\mathcal F|$ adds
$\log|\mathcal F|$: one extra bit of sample information per bit of selection
freedom. Since the deviation shrinks like the square root, the
sample size needed grows only *logarithmically* in the class size, which is
why choosing among even thousands of hyperparameter configurations on a
$10{,}000$-point validation set is tolerable: $t$ grows only from
$\sqrt{\log(2/\delta)/2n}\approx 0.014$ to
$\sqrt{\log(2\cdot 2000/\delta)/2n}\approx 0.024$ at $\delta=0.05$. Even a
*million* models widen the guarantee only to $\approx 0.030$, a factor of
$2.18$ over the single-function interval. But mind the hypothesis doing the
work: the union bound treats $\mathcal F$ as fixed *before* the sample is
drawn, so this arithmetic covers a pre-registered pool of candidates. A
leaderboard is different. When each submission is chosen after seeing earlier
scores on the same test set, the models are data-dependent functions, the
proposition's "for each fixed $f$" fails, and :eqref:`eq_mdl-finite-class`
does not apply; guarantees then degrade much faster, and analyzing how is the
subject of *adaptive data analysis* :cite:`Dwork.Feldman.Hardt.ea.2015`. This
is :numref:`chap_classification_generalization`'s test-set-reuse warning, now
with both its constant and its boundary.

### Rademacher Complexity

Real hypothesis classes are infinite (every $\mathbf w\in\mathbb{R}^d$ is a
different linear classifier), so the union bound stalls. The classical escapes
either *discretize* the class (covering numbers: replace $\mathcal F$ by the
finitely many centers of small balls that cover it, and union-bound over
those) or count its effective
behaviors on $n$ points (the VC dimension of :cite:`Vapnik.Chervonenkis.1971`,
sketched in
:numref:`chap_classification_generalization`). The modern tool is more
direct, and it starts from a question a machine learner can act on: **how well
can the class correlate with pure noise?** Let
$\varepsilon_1,\ldots,\varepsilon_n$ be i.i.d. Rademacher variables (fair
random signs) and define the **empirical Rademacher complexity** of a class
$\mathcal F$ on the sample $S=(\mathbf x_1,\ldots,\mathbf x_n)$ as

$$
\widehat{\mathfrak R}_S(\mathcal F)
 = E_{\boldsymbol\varepsilon}\!\left[\,
   \sup_{f\in\mathcal F}\ \frac1n \sum_{i=1}^n \varepsilon_i\, f(\mathbf x_i)
 \right],
$$
:eqlabel:`eq_mdl-rademacher`

with $\mathfrak R_n(\mathcal F)=E_S[\widehat{\mathfrak R}_S(\mathcal F)]$ its
average over samples. The random signs are fictitious labels containing no
signal. The supremum measures how strongly functions in the class can
correlate with them. If $\widehat{\mathfrak R}$ is near $1$, the class can fit
arbitrary labels, and its empirical risks need not estimate population risks
well. If $\widehat{\mathfrak R}$ tends to zero, empirical and population risks
converge uniformly. The following theorem makes this relation precise:

**Uniform convergence via Rademacher complexity**
:cite:`Bartlett.Mendelson.2002`. *For $\mathcal F$ taking values in $[0,1]$,
with probability at least $1-\delta$ over the sample,*

$$
R(f) \le \hat R(f) + 2\,\mathfrak R_n(\mathcal F)
 + \sqrt{\frac{\log(1/\delta)}{2n}}
\qquad\textrm{simultaneously for every } f\in\mathcal F .
$$
:eqlabel:`eq_mdl-rademacher-bound`

In this statement $\mathcal F$ is the *loss* class: its members are functions
$(\mathbf x,y)\mapsto\ell(f(\mathbf x),y)$ taking values in $[0,1]$. We relate
this class to the predictor class at the end of the subsection.
We do not give the full proof, but its central move, *symmetrization* (the
step that conjures coin flips out of a statement that contains none), is short
enough to sketch, and it explains the factor $2$. Introduce a *ghost sample*
$S'$: a second, fictitious dataset of size $n$, independent of $S$, that
exists only inside the analysis. Since $R(f)=E_{S'}[\hat R_{S'}(f)]$, Jensen's
inequality (:numref:`subsec_mdl-jensen`) bounds the expected uniform deviation
by a comparison of two concrete samples,

$$
E_S\Bigl[\sup_{f} \bigl(R(f) - \hat R_S(f)\bigr)\Bigr]
 \le E_{S,S'}\Bigl[\sup_{f} \frac1n \sum_{i=1}^n
 \bigl(\ell'_i(f) - \ell_i(f)\bigr)\Bigr],
$$

with $\ell_i,\ell'_i$ the losses on the $i$-th real and ghost points. Now the
trick: $S$ and $S'$ are i.i.d., so swapping the $i$-th real and ghost points
leaves the joint distribution unchanged, which means each difference
$\ell'_i-\ell_i$ can be multiplied by an independent random sign
$\varepsilon_i$ *for free*. Splitting the signed sum into its two halves
leaves $2\,E[\sup_f \frac1n\sum_i \varepsilon_i \ell_i(f)] =
2\,\mathfrak R_n(\ell\circ\mathcal F)$: the unknown truth has vanished, and a
correlation with coin flips stands in its place. The final step from this
statement *in expectation* to the high-probability form
:eqref:`eq_mdl-rademacher-bound` is *McDiarmid's inequality*
:cite:`McDiarmid.1989`, the extension of
Hoeffding from sums to any function that no single coordinate can move much
(here, changing one sample point moves the supremum by at most $1/n$); it is
stated in
exercise :exref:`ex_mdl-concentration-generalization-mcdiarmid-s-inequality`, which
carries out this step.
Sanity-check the theorem at its ends. A singleton class has $\widehat{\mathfrak R}=0$ (the expectation of
$\frac1n\sum\varepsilon_i f(\mathbf x_i)$ is $0$ with nothing to optimize) and
:eqref:`eq_mdl-rademacher-bound` collapses to one-sided Hoeffding. A finite
class obeys $\mathfrak R_n \le \sqrt{2\log|\mathcal F|/n}$ (Massart's lemma
:cite:`Boucheron.Lugosi.Massart.2013`),
recovering :eqref:`eq_mdl-finite-class` up to constants. For binary
classes, the growth function and VC dimension bound the number of labelings
that the class can realize on $n$ points, giving
$\mathfrak R_n\lesssim\sqrt{d_{\mathrm{VC}}\log n / n}$
:cite:`boucheron2005theory`. At the other end, the class of *all* functions
into $[-1,1]$ matches every sign pattern perfectly,
$\widehat{\mathfrak R}=1$, and the bound is vacuous, as it must be, since
that class can memorize anything.

### The Linear Class in Closed Form

Rademacher complexity is useful when it can be bounded or computed. For
norm-constrained linear functions, the calculation is short and leads directly
to a dimension-independent bound.

**Proposition (Rademacher complexity of a norm-bounded linear class).** *Let
$\mathcal F=\{\mathbf x\mapsto\langle\mathbf w,\mathbf x\rangle :
\|\mathbf w\|_2\le B\}$ and suppose the sample satisfies
$\|\mathbf x_i\|_2\le r$ for all $i$. Then*

$$
\widehat{\mathfrak R}_S(\mathcal F) \le \frac{B\,r}{\sqrt n}.
$$
:eqlabel:`eq_mdl-linear-rademacher`

**Proof.** The supremum is explicit: by linearity and Cauchy--Schwarz (with
equality at $\mathbf w$ aligned to the sum),

$$
\sup_{\|\mathbf w\|\le B}\ \frac1n\sum_{i=1}^n
 \varepsilon_i \langle\mathbf w,\mathbf x_i\rangle
 = \sup_{\|\mathbf w\|\le B}\ \frac1n
   \Bigl\langle\mathbf w,\ \sum_i \varepsilon_i\mathbf x_i\Bigr\rangle
 = \frac Bn\,\Bigl\|\sum_{i=1}^n \varepsilon_i\mathbf x_i\Bigr\| .
$$

Jensen's inequality (:numref:`subsec_mdl-jensen`; the square root is concave)
moves the expectation inside the square:

$$
E_{\boldsymbol\varepsilon}\Bigl\|\sum_i\varepsilon_i\mathbf x_i\Bigr\|
 \le \sqrt{E_{\boldsymbol\varepsilon}\Bigl\|\sum_i\varepsilon_i\mathbf x_i\Bigr\|^2}
 = \sqrt{\sum_{i,j} E[\varepsilon_i\varepsilon_j]\,
   \langle\mathbf x_i,\mathbf x_j\rangle}
 = \sqrt{\sum_i \|\mathbf x_i\|^2}
 \le r\sqrt n,
$$

the cross terms vanishing because independent signs have
$E[\varepsilon_i\varepsilon_j]=0$ for $i\ne j$. Multiplying by $B/n$ gives
$Br/\sqrt n$. $\blacksquare$

Now look at what is *absent* from :eqref:`eq_mdl-linear-rademacher`: the
dimension $d$. A linear class over a million features and over ten features
has the *same* bound only when both its weight-radius $B$ and input-radius $r$
match. Thus parameter count does not appear explicitly; capacity in this bound
is controlled by the **product** $Br$. Changing or adding features can change
$r$, so weight norm alone is not an invariant measure of capacity. The bound
nevertheless gives one reason norm control can help: shrinking
$\|\mathbf w\|$ permits a smaller radius $B$ when the class is specified
independently of the sample. This connects to weight decay
(:numref:`sec_weight_decay`) and motivates tracking solution norm below. It
does not by itself explain the double-descent experiment, because feature
radii and data-dependent selection also matter.

One step connects the two propositions.
:eqref:`eq_mdl-rademacher-bound` wants the complexity of the *loss* class
$\ell\circ\mathcal F$ (the functions whose empirical means we actually
compare), while :eqref:`eq_mdl-linear-rademacher` computes that of the
*predictor* class $\mathcal F$. The bridge is the *contraction principle*
:cite:`Boucheron.Lugosi.Massart.2013`: composing every $f\in\mathcal F$ with
one fixed $L$-Lipschitz function $\phi$ multiplies the Rademacher complexity
by at most $L$ after centering $\phi(0)=0$; subtracting the constant
$\phi(0)$ does not affect the signed complexity. Thus
$\widehat{\mathfrak R}_S(\phi\circ\mathcal F)\le
L\,\widehat{\mathfrak R}_S(\mathcal F)$. Margin losses (the hinge, the
clipped square, the logistic) are Lipschitz in the prediction, so the loss
class of the norm-bounded linear model inherits the bound
$L\,Br/\sqrt n$ and the norm-controls-capacity conclusion survives the
composition intact.

### Vacuous Bounds and Useful Complexity Measures

For modern neural networks, these bounds are often vacuous.
:cite:`zhang2021understanding` trained standard
architectures to zero training error on CIFAR-10 with *randomly shuffled
labels* (on ImageNet, random-label training still reaches near-perfect
*training* accuracy): the class realized by "this architecture, trained by
SGD" can correlate essentially perfectly with coin flips, so its Rademacher
complexity on such
samples is essentially $1$ and :eqref:`eq_mdl-rademacher-bound` certifies
nothing; norm-based refinements, evaluated at the sizes practitioners use,
yield bounds orders of magnitude above the trivial bound of $1$. This is a
fact about *these bounds*, the ones that take
uniform convergence over the entire representable class.
The same architecture can generalize on real labels, showing that the data,
optimizer, and reached solution contain information ignored by a bound on the
entire representable class. Subsequent work therefore studies smaller,
algorithm- or data-dependent classes. Uniform convergence provides one precise
form of a generalization guarantee, but it is not the only framework for
analyzing learning algorithms.
Here the crude choice of $\mathcal F$ makes the displayed bound vacuous. The
last section studies a smaller random-features model in which solution norm and
conditioning can be computed directly.

### Coin Flips in Code
:label:`subsec_mdl-coin-flips-code`

Both phenomena can be measured. First we estimate
$\widehat{\mathfrak R}_S$ for the norm-bounded linear class by Monte Carlo:
the proof gave the supremum in closed form,
$\frac Bn\|\sum_i\varepsilon_i\mathbf x_i\|$, so we average that norm
over random sign draws and compare against $Br/\sqrt n$. Then we exhibit the
Zhang phenomenon in miniature: a class rich enough to *interpolate* (here,
minimum-norm least squares on $p=2n$ random features, via the pseudoinverse of
:numref:`subsec_mdl-pseudoinverse`) achieves correlation $1$ with every coin
flip it is shown.

```{.python .input #mdl-concentration-generalization-coin-flips-in-code}
rng = onp.random.default_rng(0)
n, d, B, num_sigma = 50, 20, 1.0, 2000
X = rng.standard_normal((n, d))
X /= onp.linalg.norm(X, axis=1, keepdims=True)     # ||x_i|| = r = 1
signs = rng.choice([-1.0, 1.0], size=(num_sigma, n))
sup_corr = B * onp.linalg.norm(signs @ X, axis=1) / n  # closed-form supremum
print(f'Monte-Carlo Rademacher complexity  = {sup_corr.mean():.4f}')
print(f'bound B*r/sqrt(n)                  = {B / onp.sqrt(n):.4f}')

# The Zhang phenomenon in miniature: an interpolating class hits corr = 1
Phi = rng.standard_normal((n, 2 * n))        # p = 2n features: interpolation
W = onp.linalg.pinv(Phi) @ signs.T           # min-norm fit to every flip set
corr = onp.mean(signs.T * (Phi @ W), axis=0)     # (1/n) sum_i eps_i f(x_i)
w_norm = onp.linalg.norm(W, axis=0).mean()
r_phi = onp.linalg.norm(Phi, axis=1).mean()
print(f'correlation of min-norm fits with coin flips = {corr.mean():.4f}')
print(f'norm it costs: mean ||w|| = {w_norm:.2f};  implied capacity bound '
      f'B*r/sqrt(n) = {w_norm * r_phi / onp.sqrt(n):.2f}')
```

The Monte-Carlo estimate is $0.1390$ against the bound $0.1414$: the
Cauchy--Schwarz/Jensen argument is nearly tight, losing under $2\%$ (the only
slack is Jensen's step, and $\|\sum\varepsilon_i\mathbf x_i\|$ has small
relative fluctuation: concentration again). So a unit-norm linear class on
fifty unit-norm points can fake a correlation of about $0.14$ with pure noise,
and no more: its empirical risks are meaningful. The interpolating class, by
contrast, scores a correlation of exactly $1.0000$ on every one of the two
thousand sign patterns; it is the all-functions endpoint of the theorem, in
a hundred features and fifty points. The norm it needed is printed next to it:
the
sign-fitting solutions need norm about $1.0$ on features of norm about $10$
(both averaged over fits and rows in the print), so the smallest linear class
containing them has capacity bound $Br/\sqrt n$ of at least $1.41 > 1$,
vacuous, exactly as the previous subsection said it must be. The two prints summarize the section: the capacity that
matters is the norm the fit actually uses on the data.

## Interpolation and Double Descent

### Classical U-Shaped Risk Curves

The familiar U-curve is a useful heuristic: as a model class grows, bias may
fall while variance rises. Neither the bias--variance identity nor uniform
convergence requires this shape, however. In some interpolating regimes, test
error peaks near the point where training error first reaches zero and then
falls as the model grows. This behavior is called **double descent**
:cite:`Belkin.Hsu.Ma.ea.2019,nakkiran2021deep`.
:numref:`subsec_double_descent` measures it for random ReLU features on
Fashion-MNIST. Here we analyze the same construction on synthetic data, where
its conditioning and minimum-norm mechanisms can be computed exactly; they
should not be read as a general explanation of deep-network generalization.

### The Minimum-Norm Mechanism

The smallest such model is **random-features regression**
:cite:`Rahimi.Recht.2007`. Fix a feature map
built from randomness (ours will be $\phi(\mathbf x)=\mathrm{ReLU}
(\mathbf V\mathbf x)$ with a random frozen $\mathbf V$, a one-hidden-layer
network whose first layer is never trained) and fit only the linear head
$\mathbf w\in\mathbb{R}^p$ by least squares on $n$ training points.
We vary the feature count $p$ across the interpolation threshold $p=n$.
The fitted model behaves differently below, at, and above this threshold.
Write $\boldsymbol\Phi\in\mathbb{R}^{n\times p}$ for the feature matrix
and fit $\mathbf w=\boldsymbol\Phi^{+}\mathbf y$ with the pseudoinverse of
:numref:`subsec_mdl-pseudoinverse`, which returns the least-squares solution
for $p<n$ and the **minimum-norm interpolant** for $p\ge n$.

*Below the threshold* ($p<n$) the system is overdetermined: the fit cannot
match all $n$ labels, residual noise is averaged rather than reproduced, and
the classical U-curve logic applies verbatim: bias falls and variance rises
with $p$.

*At the threshold* ($p=n$) the feature matrix is square, and interpolation
requires inverting it: $\mathbf w=\boldsymbol\Phi^{-1}\mathbf y$. A random
square matrix is almost surely invertible but almost never *well*
invertible: its smallest singular value $\sigma_{\min}$ is typically tiny,
and the condition-number lens of :numref:`subsec_mdl-condition-number` says
what happens next: the solution acquires a component of size
$\langle\mathbf u_{\min},\mathbf y\rangle/\sigma_{\min}$ along the worst
direction. The labels' noise, which a regression is supposed to average away,
is instead *divided by a near-zero number*. The solution norm explodes and the
test error spikes. This is the peak of double descent, and it is a
conditioning event.

*Beyond the threshold* ($p>n$) interpolation is easy: infinitely many
$\mathbf w$ fit exactly, and the pseudoinverse picks the smallest. With
*nested* features (each model uses the first $p$
columns of one master matrix) this yields an exact monotonicity: any
interpolant
available at $p$ is still available at $p+1$ by padding with a zero, so the
feasible set only grows and

$$
\bigl\|\mathbf w^{(p+1)}_{\min}\bigr\| \le \bigl\|\mathbf w^{(p)}_{\min}\bigr\| :
$$

**more nested features let the minimum-norm interpolant get no larger.** This is
a useful part of the mechanism, but it is not by itself a Rademacher-complexity
proof of the second descent. The bound :eqref:`eq_mdl-linear-rademacher` depends
on $Br$: as $p$ grows here, $\|\mathbf w\|$ falls while the random-feature
norm $\|\boldsymbol\phi(\mathbf x)\|$ grows. Moreover, selecting the radius
from the fitted, data-dependent solution requires a localized or
algorithm-dependent argument. The experiment below therefore tracks solution
norm as a diagnostic correlated with test error, not as a complete capacity
certificate.

### A Numerical Double-Descent Example
:label:`subsec_mdl-double-descent-example`

We now compute the complete double-descent curve. We use forty training points
from a noisy linear teacher in fifteen dimensions; ReLU random features with
nested columns; the head fit by `pinv`: least squares below the threshold,
minimum-norm interpolation above it. We sweep $p$ from $2$ to $400$ straight
through $p=n=40$, and we track the two quantities the mechanism says to watch:
the test error and $\|\mathbf w\|$. Medians over $20$ independent runs keep
the curve readable.

```{.python .input #mdl-concentration-generalization-double-descent-in-twenty-five-lines}
rng = onp.random.default_rng(0)
n, n_test, d_in, p_max, trials, noise = 40, 500, 15, 400, 20, 0.1
p_grid = onp.array([2, 5, 10, 15, 20, 25, 30, 35, 38, 40, 42, 45, 50, 60,
                    80, 120, 200, 400])
test_mse, train_mse, w_norm = (onp.zeros((trials, len(p_grid)))
                               for _ in range(3))
for tr in range(trials):
    beta = rng.standard_normal(d_in)
    beta /= onp.linalg.norm(beta)                    # teacher: y = <beta, x>
    X, X_te = (rng.standard_normal((m, d_in)) for m in (n, n_test))
    y = X @ beta + noise * rng.standard_normal(n)    # noisy training labels
    y_te = X_te @ beta
    V = rng.standard_normal((p_max, d_in)) / onp.sqrt(d_in)
    Phi, Phi_te = onp.maximum(X @ V.T, 0), onp.maximum(X_te @ V.T, 0)
    for j, p in enumerate(p_grid):                   # nested feature prefixes
        w = onp.linalg.pinv(Phi[:, :p]) @ y  # least-squares/min-norm solution
        test_mse[tr, j] = onp.mean((Phi_te[:, :p] @ w - y_te) ** 2)
        train_mse[tr, j] = onp.mean((Phi[:, :p] @ w - y) ** 2)
        w_norm[tr, j] = onp.linalg.norm(w)
med = [onp.median(a, axis=0) for a in (test_mse, train_mse, w_norm)]
for j, p in enumerate(p_grid):
    print(f'p={p:3d}  test MSE={med[0][j]:9.4f}  train MSE={med[1][j]:9.2e}'
          f'  ||w||={med[2][j]:7.2f}')
d2l.plot(p_grid, [med[0], med[2]], 'number of random features p',
         'median over 20 runs', legend=['test MSE', '||w|| of the fit'],
         xscale='log', yscale='log')
```

In the underparameterized regime, test error falls from $1.04$ at $p=2$ to
about $0.68$ around $p=15$, then rises as variance increases. At the
interpolation threshold $p=40=n$, the training error drops from
$5\times10^{-3}$ to $10^{-28}$ (exact interpolation, up to floating
point). At the same point, the test error rises to $33.6$, about fifty times its
underparameterized minimum, while $\|\mathbf w\|$ increases to $18.5$. This is
the effect of dividing the noise component by the small singular value
$\sigma_{\min}$. Beyond the threshold, from $p=42$ onward, the norm decreases
monotonically ($4.96$,
$3.64$, $2.42$, $1.70$, $1.18$, down to $0.39$ at $p=400$; the nesting
guarantees this). The test error falls to $0.65$ at
$p=50$, $0.18$ at $p=80$, and $0.060$ at $p=400$, ten times below the
best underparameterized value. By contrast, the training error remains zero
up to floating point everywhere past the threshold (the table
prints $10^{-28}$), so no
empirical-risk-based criterion can tell these models apart; the solution norm
distinguishes them in this experiment. The Rademacher calculation motivates
looking beyond parameter count, but its relevant product also includes feature
norms and does not by itself predict this test curve. The best model in this
entire experiment is the most overparameterized one, fitting noisy data
*exactly*, with ten times more parameters than data points.

### Benign Overfitting
:label:`subsec_mdl-benign-overfitting`

One puzzle remains inside the mechanism. The $p=400$ model interpolates its
noisy labels (it reproduces every corrupted $y_i$ exactly), yet its test
error is the best on the table. When does fitting noise *not* hurt? The
minimum-norm solution splits across the spectrum of the feature matrix: the
*signal* is captured along the few directions with large singular values,
while interpolating the residual *noise* is distributed (because minimizing
the norm spreads it) across the many directions with small singular values,
where each contaminates predictions at new points only faintly. Overfitting is
**benign** when the spectrum has this shape: a few strong directions to carry
the signal, plus a large reservoir of weak directions that absorb and
*average out* the noise, in effect performing implicit regularization without
a regularizer. :cite:`Bartlett.Long.Lugosi.Tsigler.2020` make this exact for
linear regression, characterizing benignity via two effective ranks of the
covariance, and :cite:`Belkin.Hsu.Ma.ea.2019` frame the modern regime it
explains; :cite:`Bartlett.Montanari.Rakhlin.2021` survey the fast-growing
theory. For deep networks the picture is instructive but not settled: features
there are *learned*, reshaping the spectrum during training, and a theory that
predicts a given network's test error from first principles remains open.

## Summary

* Hoeffding certifies one *pre-chosen* function; a learner chooses *after*
  seeing data, so guarantees must hold *uniformly* over the class. For finite
  classes the union bound gives deviation
  $\sqrt{\log(2|\mathcal F|/\delta)/2n}$, valid for a pool of models fixed
  before the data are drawn; genuinely adaptive test-set reuse voids the
  union bound, and its guarantees degrade faster (adaptive data analysis).
* *Rademacher complexity* (how well a class can correlate with random
  signs) bounds uniform deviations:
  $R(f)\le\hat R(f)+2\mathfrak R_n(\mathcal F)+\sqrt{\log(1/\delta)/2n}$. For
  the linear class $\{\|\mathbf w\|\le B\}$ on data with $\|\mathbf x\|\le r$
  it computes to $Br/\sqrt n$: parameter count is absent when the weight and
  data radii are held fixed. A class rich enough to fit arbitrary random signs
  has complexity near its maximum and yields a vacuous bound; the failure belongs to the crude class choice, and uniform
  convergence survives as the language of guarantees.
* *Double descent*: sweeping random features through the interpolation
  threshold $p=n$, test error follows the classical U, spikes at $p=n$ (a
  near-singular system divides noise by $\sigma_{\min}$), then descends again
  while the minimum-norm interpolant shrinks with added nested features. This
  norm trend is a mechanism and diagnostic; a formal capacity explanation must
  also track feature norms and the data-dependent selection of the solution. Interpolating noise is *benign* when a few strong
  spectral directions carry the signal and many weak ones absorb the noise.

## Exercises

1. **McDiarmid's inequality.** ● The inequality of :citet:`McDiarmid.1989`
   extends Hoeffding's from sums to functions with *bounded differences*.
   Suppose that changing only the $i$-th argument of $g(x_1,\ldots,x_n)$
   moves its value by at most $c_i$, and that $X_1,\ldots,X_n$ are
   independent. Then

   $$P\bigl(g(X_1,\ldots,X_n)-E[g(X_1,\ldots,X_n)]\ge t\bigr)\le\exp\left(-\frac{2t^2}{\sum_{i=1}^n c_i^2}\right),$$

   and the same bound holds for the lower tail. Take this statement on
   faith; its proof uses martingales.
    1. Recover Hoeffding's inequality :eqref:`eq_mdl-hoeffding` from it.
    1. Let $g(S)=\sup_{f\in\mathcal F}\bigl(R(f)-\hat R_S(f)\bigr)$ for a loss
       class $\mathcal F$ with values in $[0,1]$. Show that $g$ has bounded
       differences $c_i=1/n$. Combine McDiarmid's inequality with the
       symmetrization bound $E_S[g(S)]\le 2\mathfrak R_n(\mathcal F)$ and
       derive :eqref:`eq_mdl-rademacher-bound`.
    1. For data in $[0,1]$, the sample mean has bounded differences $1/n$.
       Construct a sample of odd size $n$ in $[0,1]$ in which changing one
       point moves the sample median by $1$. What does this imply for using
       McDiarmid's inequality to put an error bar on the bootstrap estimate
       of the median in :numref:`sec_mdl-statistics`?
   :exlabel:`ex_mdl-concentration-generalization-mcdiarmid-s-inequality`

1. **Massart's lemma and $\ell_1$ balls.** ● Massart's lemma for finite
   classes and the Rademacher complexity of $\ell_1$-bounded linear classes
   rest on one bound for the expected maximum of sub-Gaussian variables.
    1. Let $Z_1,\ldots,Z_N$ be centered and sub-Gaussian with variance proxy
       $\sigma^2$ in the sense of :eqref:`eq_mdl-subgaussian`, not
       necessarily independent. Show that
       $E[\max_j Z_j]\le\sigma\sqrt{2\log N}$ by bounding
       $e^{\lambda E[\max_j Z_j]}$ by $\sum_j E[e^{\lambda Z_j}]$ and
       optimizing over $\lambda>0$. Deduce Massart's lemma,
       $\widehat{\mathfrak R}_S(\mathcal F)\le\sqrt{2\log|\mathcal F|/n}$,
       for a finite class with values in $[-1,1]$.
    1. Let $\mathcal F=\{\mathbf x\mapsto\langle\mathbf w,\mathbf x\rangle:
       \|\mathbf w\|_1\le B\}$ and $\|\mathbf x_i\|_\infty\le r$ for all $i$.
       Follow the proof of :eqref:`eq_mdl-linear-rademacher`, with the
       duality $\sup_{\|\mathbf w\|_1\le B}\langle\mathbf w,\mathbf
       v\rangle=B\|\mathbf v\|_\infty$ in place of Cauchy--Schwarz, and show
       that $\widehat{\mathfrak R}_S(\mathcal F)\le Br\sqrt{2\log(2d)/n}$.
       Why does the dimension $d$ now enter, and why only logarithmically?
    1. Let the inputs lie in $\{-1,+1\}^d$ and let the target weight vector
       have $k$ nonzero entries, each equal to $\pm1$. Compare the bound
       of (b) with :eqref:`eq_mdl-linear-rademacher`, each with the smallest
       radius $B$ that contains the target. For which $k$ is the $\ell_1$
       bound smaller when $d=10^4$?
   :exlabel:`ex_mdl-concentration-generalization-massart-l1`

1. **Margin bound.** ● Let $\mathcal F=\{\mathbf x\mapsto\langle\mathbf
   w,\mathbf x\rangle:\|\mathbf w\|_2\le B\}$ with $\|\mathbf x\|_2\le r$,
   let the labels be $y\in\{-1,+1\}$, and let the ramp loss
   $\phi_\gamma(u)=\min\{1,\max\{0,1-u/\gamma\}\}$ with a fixed $\gamma>0$
   act on the margin $u=yf(\mathbf x)$.
    1. Show that $\phi_\gamma(yf(\mathbf x))$ is at least the 0-1 loss
       $\mathbf 1\{yf(\mathbf x)\le 0\}$ and that $\phi_\gamma$ is
       $(1/\gamma)$-Lipschitz. Show that multiplying each prediction
       $f(\mathbf x_i)$ by a fixed label $y_i$ leaves
       $\widehat{\mathfrak R}_S(\mathcal F)$ unchanged.
    1. Combine (a), the contraction principle,
       :eqref:`eq_mdl-rademacher-bound`, and
       :eqref:`eq_mdl-linear-rademacher` into a bound on the 0-1 risk of
       every $f\in\mathcal F$ in terms of its empirical ramp loss. For
       $n=50{,}000$ and $\delta=0.05$, find the largest ratio $Br/\gamma$ for
       which the complexity and confidence terms together stay below $0.1$.
    1. The bound holds for a radius $B$ fixed before the data are drawn,
       whereas the norm of a trained weight vector is known only afterwards.
       Apply the bound with radius $B_j=2^j$ and confidence
       $\delta_j=\delta/(j(j+1))$ for every $j\ge1$, and take a union bound
       over $j$. Which bound holds for a weight vector with
       $\|\mathbf w\|>1$ chosen after seeing the data, and what does the
       freedom to choose the radius cost?
   :exlabel:`ex_mdl-concentration-generalization-margin-bound`

1. [code] **Fitting random signs.** ● The cell of
   :numref:`subsec_mdl-coin-flips-code` fits $2{,}000$ sign vectors
   $\boldsymbol\varepsilon\in\{-1,+1\}^n$ with
   $\mathbf w=\boldsymbol\Phi^+\boldsymbol\varepsilon$ on $p=2n$ standard
   Gaussian features and finds the correlation
   $\frac1n\sum_i\varepsilon_i(\boldsymbol\Phi\mathbf w)_i=1$.
    1. Let $p<n$ and let $\boldsymbol\Phi$ have full column rank. Show that
       the correlation has expectation $p/n$ over the signs. The matrix
       $\boldsymbol\Phi\boldsymbol\Phi^+$ is a projection.
    1. Let $p>n$, let $r$ be the largest row norm of $\boldsymbol\Phi$, and
       let $B$ be large enough that the class $\{\|\mathbf w\|\le B\}$
       contains an interpolant of every sign vector. Explain why
       $Br/\sqrt n$ cannot then be smaller than $1$. For a standard Gaussian
       $\boldsymbol\Phi$ with $p>n+1$,
       $E[(\boldsymbol\Phi\boldsymbol\Phi^\top)^{-1}]=\mathbf I_n/(p-n-1)$.
       Use this identity to predict the typical $\|\mathbf w\|$, and the
       typical value of $\|\mathbf w\|$ times a row norm of
       $\boldsymbol\Phi$ divided by $\sqrt n$, as functions of $p$ and
       $n$.
    1. Rerun the cell with $n=50$ for $p\in\{10,25,45,55,100,400,1000\}$
       and compare the measured correlation, the mean $\|\mathbf w\|$, and
       the printed capacity bound with (a) and (b). What happens to the
       capacity bound as $p\to\infty$?
   :exlabel:`ex_mdl-concentration-generalization-fitting-random-signs`

1. [code] **Leaderboard attack.** ●● A leaderboard reports the 0-1 loss of
   each submitted label vector on a holdout set of $n=4{,}000$ examples with
   labels $\mathbf y\in\{0,1\}^n$. The *boosting attack* of
   :citet:`Blum.Hardt.2015` uses no knowledge of the task: submit $k$
   uniformly random label vectors, keep those with reported loss at most
   $\tfrac12$, and submit their coordinatewise majority vote. On fresh data,
   the majority vote has loss $\tfrac12$.
    1. Implement the attack for uniformly random $\mathbf y$ and
       $k\in\{10,30,100,300,1000,3000\}$, and record the reported loss of
       the majority vote. Compare its distance from $\tfrac12$ with the
       deviation that :eqref:`eq_mdl-finite-class` allows for a
       pre-registered pool of $k+1$ models at $\delta=0.05$. From which $k$
       on does the attack exceed this deviation, and which hypothesis of the
       proposition does it violate?
    1. :citet:`Blum.Hardt.2015` prove that the reported loss is
       $\tfrac12-\Omega(\sqrt{k/n})$ but do not give the constant. Derive it
       heuristically with Gaussian approximations: estimate the expected
       excess agreement with $\mathbf y$ of a kept vector, then the
       probability that the majority of about $k/2$ such vectors is correct
       on one coordinate. Compare the constant with the measurements
       of (a).
    1. The *Ladder* mechanism of :citet:`Blum.Hardt.2015` releases a new
       score only when a submission beats the best released score by more
       than a step $\eta$, rounds it to a multiple of $\eta$, and otherwise
       repeats the previous score. Against the Ladder, the attacker can only
       keep the vectors that lowered the released score. Implement both for
       $\eta\in\{0.005,0.01\}$ and $k\in\{100,1000,3000\}$. How far below
       $\tfrac12$ does the majority vote now get, and how does this compare
       with the deviation of (a)?
   :exlabel:`ex_mdl-concentration-generalization-leaderboard-attack`

1. [extended] **Benign spectra.** ●●
   :citet:`Bartlett.Long.Lugosi.Tsigler.2020` characterize when the minimum-norm interpolant in linear regression has
   small excess risk, the mechanism that
   :numref:`subsec_mdl-benign-overfitting` describes in words. Let
   $\mathbf x\sim\mathcal N(\mathbf 0,\boldsymbol\Sigma)$ with
   $\boldsymbol\Sigma=\mathrm{diag}(\lambda_1,\ldots,\lambda_p)$ and
   $\lambda_1\ge\lambda_2\ge\cdots$, let
   $y=\langle\boldsymbol\theta^*,\mathbf x\rangle+\xi$ with
   $\xi\sim\mathcal N(0,\sigma^2)$, stack $n<p$ training inputs into
   $\mathbf X\in\mathbb R^{n\times p}$, and let
   $\hat{\boldsymbol\theta}=\mathbf X^+\mathbf y$. The excess risk is
   $(\hat{\boldsymbol\theta}-\boldsymbol\theta^*)^\top\boldsymbol\Sigma
   (\hat{\boldsymbol\theta}-\boldsymbol\theta^*)$.
    1. Show that its expectation over the noise is the sum of a bias term
       $\boldsymbol\theta^{*\top}(\mathbf I-\mathbf P)\boldsymbol\Sigma
       (\mathbf I-\mathbf P)\boldsymbol\theta^*$, where
       $\mathbf P=\mathbf X^+\mathbf X$, and a variance term
       $\sigma^2\operatorname{tr}\bigl((\mathbf X^+)^\top\boldsymbol\Sigma
       \mathbf X^+\bigr)$. Implement both terms; no test set is needed.
    1. The paper's effective ranks are
       $r_k=\sum_{i>k}\lambda_i/\lambda_{k+1}$ and
       $R_k=\bigl(\sum_{i>k}\lambda_i\bigr)^2/\sum_{i>k}\lambda_i^2$. With
       $k^*$ the smallest $k$ such that $r_k\ge bn$, the variance term is of
       order $\sigma^2(k^*/n+n/R_{k^*})$ up to constants; take $b=1$. Use
       $n=100$, $\sigma=1$, five strong directions with $\lambda_i=1$ that
       contain a unit-norm $\boldsymbol\theta^*$, and $q$ weak directions
       with $\lambda_i=1/q$ each, for $q\in\{100,300,10^3,3\cdot10^3,10^4\}$.
       Compare the measured variance with $\sigma^2(k^*/n+n/R_{k^*})$. What
       happens at $q=100$, and why?
    1. Repeat with power-law spectra $\lambda_i=i^{-\alpha}$ truncated at
       $p=40n$, for $\alpha\in\{0.5,1,2\}$, $\boldsymbol\theta^*=\mathbf e_1$,
       and $n\in\{50,100,200,400\}$. For which spectra does the excess risk
       decrease with $n$? For the others, does the bias or the variance
       fail, and how do $r_0/n$, $k^*/n$, and $n/R_{k^*}$ show it?
   :exlabel:`ex_mdl-concentration-generalization-benign-spectra`

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §27.8]{.kicker}

When learners can be trusted<br>**uniform convergence · Rademacher complexity · double descent**.
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[Uniform convergence]{.dtitle}

[the function chosen after the data]{.dsub}
:::
:::

::: {.slide title="The function chosen after the data"}
[The pivot]{.kicker}

Hoeffding certifies **one fixed function**: choose $f$, *then* draw the
sample. A test set works precisely because it respects this order.

. . .

A *learner* violates it by construction: it searches $\mathcal F$ and
returns the $\hat f$ that looks best *on the sample it saw*, a minimum
of fluctuating quantities, selected exactly where the fluctuation
flatters. The quantity that must concentrate is the worst case,

$$\sup_{f\in\mathcal F}\;\bigl|\hat R(f) - R(f)\bigr|.$$

::: {.d2l-note .warn}
Test-set **reuse** is the same trap in different clothes: an analyst who
tries many models on one test set *is* a learner over the class of models
tried.
:::
:::

::: {.slide title="Finite classes: the union bound"}
[Finite classes]{.kicker}

For finite $\mathcal F$, Hoeffding + a union bound over $|\mathcal F|$
bad events: with probability $1-\delta$, simultaneously for every $f$,

$$\bigl|\hat R(f) - R(f)\bigr|
\;\le\; \sqrt{\frac{\log(2|\mathcal F|/\delta)}{2n}}.$$

. . .

Selection freedom enters only through $\log|\mathcal F|$, under the
square root: $2{,}000$ hyperparameter configs on a $10{,}000$-point
validation set move the guarantee only from $0.014$ to $0.024$; even a
*pre-registered* million, only to $0.030$. Adaptive submissions, each
chosen after seeing earlier scores, void the union bound entirely:
guarantees then degrade much faster (adaptive data analysis).
:::

::: {.slide title="Rademacher complexity: correlate with coin flips"}
[Infinite classes]{.kicker}

Real classes are infinite, so ask a question a learner can act on: how
well can $\mathcal F$ correlate with **pure noise**?

$$\widehat{\mathfrak R}_S(\mathcal F)
= E_{\boldsymbol\varepsilon}\Bigl[\sup_{f\in\mathcal F}
\tfrac1n\textstyle\sum_i \varepsilon_i f(\mathbf x_i)\Bigr],
\qquad
R(f) \le \hat R(f) + 2\,\mathfrak R_n(\mathcal F)
+ \sqrt{\tfrac{\log(1/\delta)}{2n}}.$$

A class that can chase coin flips can chase the noise in real labels; one
that cannot has trustworthy empirical risks.

::: {.d2l-note}
The factor $2$ is **symmetrization**: a ghost sample plus the swap trick
conjures the random signs out of a statement that contains none;
McDiarmid lifts it to high probability.
:::
:::

::: {.slide title="The linear class: capacity is Br/√n"}
[The linear class]{.kicker}

For $\mathcal F = \{\mathbf x \mapsto \langle\mathbf w,\mathbf x\rangle :
\|\mathbf w\| \le B\}$ on data with $\|\mathbf x_i\| \le r$, the supremum
is Cauchy--Schwarz in closed form, and independent signs kill the cross
terms:

$$\widehat{\mathfrak R}_S
= \frac Bn\,E\Bigl\|\sum_i \varepsilon_i \mathbf x_i\Bigr\|
\;\le\; \frac{B\,r}{\sqrt n}.$$

. . .

::: {.d2l-note .rule}
The dimension $d$ is absent from this bound: linear classes with different
feature counts have the same bound when the input and weight radii match.
Feature changes can alter those radii, and a radius chosen after fitting needs
a data-dependent analysis.
:::
:::

::: {.slide title="Coin flips in code"}
[Measured]{.kicker}

Monte-Carlo Rademacher complexity of the unit-norm linear class, plus
the Zhang phenomenon in miniature, an interpolating class fed
$2{,}000$ sets of coin flips:

@!mdl-concentration-generalization-coin-flips-in-code

The bound $0.1414$ is within $2\%$ of the truth. The interpolating class
scores correlation $1.0000$ on *every* flip set, and its own printout
shows the norm it needed: the smallest linear class containing those fits
has $Br/\sqrt n \approx 1.41 > 1$. Vacuous, as it must be.
:::

::: {.slide title="When uniform-convergence bounds are vacuous"}
[Limits]{.kicker}

Zhang et al. trained standard architectures to zero training error on
CIFAR-10 with **randomly shuffled labels**: the class "this architecture,
trained by SGD" correlates perfectly with coin flips, so its Rademacher
complexity is $\approx 1$ and the bound certifies nothing.

. . .

This is a fact about *these bounds*, which take uniform convergence over
the entire representable class. The same network generalizes on real labels,
so a whole-class bound misses properties of the data, algorithm, and reached
solution.

::: {.d2l-note}
Solution norm is one useful diagnostic. The final section studies it in a
controlled random-features model; it is not a complete theory of deep networks.
:::
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Interpolation and double descent]{.dtitle}

[a controlled random-features example]{.dsub}
:::
:::

::: {.slide title="Beyond the U-curve"}
[Double descent]{.kicker}

The U-curve is a heuristic, not a consequence required by bias--variance or
uniform convergence. In some interpolating problems, test error peaks near
the **interpolation threshold** and then falls again as the model grows.

. . .

The smallest model that shows it: **random-features regression**.
$\phi(\mathbf x) = \mathrm{ReLU}(\mathbf V\mathbf x)$ with $\mathbf V$
random and frozen, the head fit by the pseudoinverse: least squares below
the threshold, the **minimum-norm interpolant** above it.
:::

::: {.slide title="The peak is a conditioning event"}
[Double descent]{.kicker}

At $p = n$ the feature matrix is square and almost never *well*
invertible: interpolation divides the labels' noise by a tiny
$\sigma_{\min}$, the norm explodes, the test error spikes.

. . .

Beyond it, nested features give an exact monotonicity: any interpolant
at $p$ pads with a zero to one at $p+1$, so

$$\bigl\|\mathbf w^{(p+1)}_{\min}\bigr\| \;\le\; \bigl\|\mathbf w^{(p)}_{\min}\bigr\| :$$

**more nested features let the minimum-norm interpolant get smaller.** This
helps explain the experiment below, but $Br/\sqrt n$ also contains the feature
radius $r$, and a data-dependent fitted radius needs localized analysis.
:::

::: {.slide title="Double descent in twenty-six lines" layout="tight"}
[Measured: forty noisy points, ReLU random features, swept through p = n]{.kicker}

@!mdl-concentration-generalization-double-descent-in-twenty-five-lines
:::

::: {.slide title="Norm tracks the second descent here"}
[Double descent]{.kicker}

Read the sweep's numbers against the mechanism: at $p = n = 40$ the
train error hits $10^{-28}$ (exact interpolation) and the test error
erupts to **33.6**, fifty times the classical minimum, with
$\|\mathbf w\| = 18.5$. Then the norm falls monotonically to $0.39$ and
the test error follows it down to **0.060** at $p=400$: *ten times
better than the best underparameterized model*, from a model that fits
noisy data exactly.

. . .

Past the threshold the train error is zero up to floating point
everywhere, so training error does not distinguish the $p=42$ and $p=400$
fits. Their solution norms differ and correlate with test error in this
experiment; the Rademacher bound alone does not predict the curve because
feature norms and data-dependent selection also matter.
:::

::: {.slide title="Benign overfitting in linear models"}
[Double descent]{.kicker}

The $p=400$ model reproduces every *corrupted* label exactly, yet tests
best. The min-norm solution splits across the spectrum: a few strong
singular directions carry the *signal*, and minimizing the norm spreads
the interpolated *noise* across the many weak directions, where it
barely contaminates new predictions.

. . .

Under suitable covariance spectra, interpolation can be **benign**
when the spectrum offers a few strong directions for the signal and a
large reservoir of weak ones to absorb the noise.

::: {.d2l-note}
Made exact for linear regression (two effective ranks of the covariance);
for deep networks, where the features are *learned* and reshape the
spectrum, the theory is instructive but open.
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

- **Uniform convergence:** $\log|\mathcal F|$ for a pre-registered pool,
  then Rademacher; for a linear class with fixed input and weight radii the
  bound is $Br/\sqrt n$, with no explicit parameter count.
- Whole-class bounds can be **vacuous** for models able to interpolate random
  labels (Zhang et al.).
- **Double descent:** a $\sigma_{\min}$ spike, then the min-norm
  interpolant's norm falls, and overfitting can be benign.

::: {.d2l-note}
Hoeffding certifies a fixed test-set estimate, and the union bound extends the
result to a fixed finite collection of models. For linear classes, the
resulting capacity measure depends on the norm.
:::
:::
