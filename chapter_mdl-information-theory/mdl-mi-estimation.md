# Estimating Mutual Information
:label:`sec_mdl-mi-estimation`

Mutual information is defined through densities that are usually unknown. This
section asks what can be learned about it from samples alone.

Most of this section is plain-NumPy probability; the deep-learning library
appears when we train a critic at the end of the variational-bounds
section.

```{.python .input #mdl-mi-estimation-estimating-mutual-information-1}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
npx.set_np()
import numpy as onp  # plain NumPy for the probability demos
from scipy.special import digamma
```

```{.python .input #mdl-mi-estimation-estimating-mutual-information-1}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import numpy as onp  # plain NumPy for the probability demos
from scipy.special import digamma
```

```{.python .input #mdl-mi-estimation-estimating-mutual-information-1}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import numpy as onp  # plain NumPy for the probability demos
from scipy.special import digamma
```

```{.python .input #mdl-mi-estimation-estimating-mutual-information-1}
#@tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import optax
import numpy as onp  # plain NumPy for the probability demos
from scipy.special import digamma
```

Every experiment below samples the standard bivariate Gaussian pair with
correlation $\rho$ of :numref:`subsec_mdl-mi-gaussian`, whose mutual
information is known in closed form.

```{.python .input #mdl-mi-estimation-estimating-mutual-information-2}
def gauss_pair(rng, rho, n):
    """n samples of a standard bivariate Gaussian with correlation rho."""
    x = rng.standard_normal(n)
    y = rho * x + onp.sqrt(1 - rho**2) * rng.standard_normal(n)
    return x, y
```

## Why Measuring Mutual Information Is Hard
:label:`sec_mdl-mi-hard`

The next problem is to estimate mutual information from samples, such as
images and their augmentations or sentences and their continuations. This is
substantially harder than evaluating the closed-form examples above.

### Estimation in High Dimensions
:label:`subsec_mdl-mi-high-dim`

The histogram plug-in estimator used above scales poorly in high dimensions.
With $b$ bins per axis, a $d$-dimensional
histogram has $b^d$ cells; at $d = 10$ and a modest $b = 10$ that is ten
billion cells, almost all empty at any realistic sample size, and the
plug-in estimate is dominated by sampling noise in the occupied few. Already
at $d = 2$ and $\rho = 0.99$ the histogram underestimated MI by $0.18$ nats.

Nonparametric estimators can improve sample efficiency but retain strong
dimension-dependent limitations. The KSG estimator
:cite:`Kraskov.Stogbauer.Grassberger.2004` replaces bins with
$k$-nearest-neighbor distances: around each
sample find the Chebyshev (max-coordinate) distance $\epsilon_i$ to its $k$-th
nearest neighbor in
the *joint* space, count the marginal neighbors $n_x^{(i)}$ and $n_y^{(i)}$
lying strictly closer than $\epsilon_i$, and average

$$\hat{I}_{\textrm{KSG}} = \psi(k) + \psi(n) - \frac{1}{n} \sum_{i=1}^{n} \left(\psi\left(n_x^{(i)} + 1\right) + \psi\left(n_y^{(i)} + 1\right)\right),$$
:eqlabel:`eq_mdl-ksg`

where $\psi = (\log \Gamma)'$ is the digamma function. The local radii
$\epsilon_i$ adapt the resolution to wherever the data actually sit: no bins
to choose, no bandwidth to tune.

```{.python .input #mdl-mutual-information-the-curse-of-estimation}
def ksg_mi(x, y, k=5):
    """Kraskov et al. (2004) k-NN estimate of I(X;Y), in nats."""
    n = len(x)
    dx = onp.abs(x[:, None, :] - x[None, :, :]).max(axis=-1)
    dy = onp.abs(y[:, None, :] - y[None, :, :]).max(axis=-1)
    d = onp.maximum(dx, dy)               # Chebyshev distance in joint space
    onp.fill_diagonal(d, onp.inf)
    eps = onp.sort(d, axis=1)[:, k - 1]   # distance to the k-th neighbor
    n_x = (dx < eps[:, None]).sum(axis=1) - 1   # marginal counts, minus self
    n_y = (dy < eps[:, None]).sum(axis=1) - 1
    return float(digamma(k) + digamma(n)
                 - (digamma(n_x + 1) + digamma(n_y + 1)).mean())

rng = onp.random.default_rng(1)
n = 2000
for dim in (1, 2):
    for rho in (0.5, 0.7, 0.9):
        pairs = [gauss_pair(rng, rho, n) for _ in range(dim)]
        x = onp.stack([u for u, _ in pairs], axis=1)
        y = onp.stack([v for _, v in pairs], axis=1)
        print(f'd = {dim}, rho = {rho}: KSG = {ksg_mi(x, y):.3f}, '
              f'closed form = {-dim / 2 * onp.log(1 - rho**2):.3f} nats')
```

With $2{,}000$ samples, a hundredth of what the histogram consumed, KSG
lands within a few hundredths of a nat of the closed form across
correlations in this one- and two-dimensional experiment, using a fixed
neighborhood parameter $k$. This is a considerable improvement over the
histogram estimate under the tested conditions. However, KSG's guarantees rest
on smoothness assumptions that become more restrictive as $d$ rises;
nearest-neighbor distances also concentrate in high dimension. These effects
can make the estimator unreliable for image-scale variables.

The reparameterization invariance we celebrated is part of the problem.
For continuous variables with continuous marginal distributions, mutual
information depends only on the copula, the joint law of the marginal ranks.
More generally, any measurable invertible reparameterization of either variable
leaves MI unchanged. An estimator therefore cannot rely on a preferred marginal
coordinate system. The following theorem describes one consequence for
distribution-free lower confidence bounds.

### Distribution-Free Lower Bounds Grow Only Logarithmically
:label:`subsec_mdl-mi-distribution-free`

Suppose we are conservative and ask only for a *lower bound*: an estimator
$\hat{I}_N$, computed from $N$ samples of the joint, such that
$\hat{I}_N \leq I(X; Y)$ with high probability *for every distribution*.
(Lower bounds are the useful direction for representation learning, where we
want to certify that two views share a lot of information.)
:citet:`McAllester.Stratos.2020` proved that any such estimator has the
following order-level limitation:

> **Theorem (McAllester--Stratos, informal).** Any distribution-free,
> high-confidence lower bound on $I(X;Y)$ computed from $N$ samples cannot
> exceed $O(\log N)$.

The theorem is an order-level impossibility statement; its constants depend on
the confidence formulation. One intuition comes from rare likelihood-ratio
events. When MI is large, much of the distinguishing evidence can lie in matched
pairs whose probability is on the scale $e^{-I}$. Unless the sample size is
exponential in $I$, a distribution-free confidence procedure cannot safely
infer the contribution of events it has not observed. This motivates the
$O(\log N)$ rate.

It does not impose an exact $\log N$ range on every estimator or point estimate,
nor does it imply the deterministic formula $\min(I,\log N)$. The exact numerical
ceiling below belongs to InfoNCE, whose classification form gives a separate
finite-$N$ inequality.

### The Exact InfoNCE Ceiling

InfoNCE has the exact range $\log N-\mathcal L_{\mathrm{NCE}}\leq\log N$ because
its classification loss is nonnegative. To illustrate this estimator-specific
limit without critic error, draw one *positive* pair
$(x, y_1) \sim P_{X,Y}$ and $N - 1$ *negatives* $y_2, \ldots, y_N \sim P_Y$,
and score how well the positive can be identified. The Bayes-optimal rule
scores candidates by the exact likelihood ratio, the
exponentiated pointwise mutual information :eqref:`eq_mdl-pmi_def`, and the
average log-probability it assigns to the truth, plus $\log N$, is the InfoNCE
lower bound. We derive it in
:numref:`sec_mdl-infonce`. For a Gaussian pair the density ratio is available
in closed form, so the simulation isolates the estimator's limitation from
critic approximation error.

```{.python .input #mutual-information-log-n-ceiling}
def pmi_gauss(x, y, rho):
    """Exact pmi(x, y) of the standard bivariate Gaussian, in nats."""
    c = 1 - rho**2
    return (-0.5 * onp.log(c) - (x**2 - 2 * rho * x * y + y**2) / (2 * c)
            + (x**2 + y**2) / 2)

def infonce_oracle(rng, rho, N, batches):
    """InfoNCE estimate with the *true* density-ratio critic."""
    vals = []
    for _ in range(batches):
        x, y = gauss_pair(rng, rho, N)
        f = pmi_gauss(x[:, None], y[None, :], rho)     # all N x N scores
        f -= f.max(axis=1, keepdims=True)              # stable log-softmax
        ll = onp.diag(f) - onp.log(onp.exp(f).sum(axis=1))
        vals.append(ll.mean() + onp.log(N))
    return onp.array(vals)

rng = onp.random.default_rng(7)
true_mi = onp.array([0.25, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7])
rhos = onp.sqrt(1 - onp.exp(-2 * true_mi))             # invert the closed form
ests = [onp.array([infonce_oracle(rng, r, N, 200 if N <= 128 else 50).mean()
                   for r in rhos]) for N in (16, 128, 1024)]
d2l.plot(true_mi, ests + [true_mi], 'true I(X;Y) (nats)', 'estimate (nats)',
         legend=['N=16', 'N=128', 'N=1024', 'truth'], figsize=(5, 3))
```

Each curve follows the diagonal while the true mutual information is small, then
bends flat as it approaches its ceiling $\ln N$ ($\approx 2.77$, $4.85$, and
$6.93$ nats for the three batch sizes), *even though the critic is exact*.
Changing the architecture or optimizing the critic more accurately cannot
raise this estimator's range; increasing $N$ raises it only logarithmically.
This is InfoNCE's estimator-specific saturation, distinct from the more general
order-level result above.

The distinction matters in practice. A reported variational lower bound need
not be an accurate measurement of MI. It may nevertheless be a useful training
objective: improving a valid but loose bound can shape representations even
when the bound remains far below the true information.


## Variational Bounds and InfoNCE
:label:`sec_mdl-mi-variational-bounds`

Mutual information involves the unknown densities of
:eqref:`eq_mdl-mut_ent_def`, which usually cannot be evaluated directly from
samples. Variational methods instead *bound it from below* using samples and an
auxiliary model (a "decoder" or a "critic"), then tighten the bound by
optimizing that model. This converts one estimation problem into an
optimization problem, while introducing approximation and finite-sample error.
Contrastive methods use the resulting lower bound as a
training objective. We derive three classical bounds and then InfoNCE;
:citet:`Poole.Oord.Alemi.ea.2019` provide a unified treatment.

### The Barber--Agakov Bound

The first bound replaces the unknown posterior with a model of it.

**Proposition (Barber--Agakov bound).** *For any conditional density
$q(x \mid y)$,*

$$I(X; Y) \geq H(X) + E_{(x,y) \sim P_{X,Y}}\left[\log q(x \mid y)\right],$$
:eqlabel:`eq_mdl-ba_bound`

*with equality if and only if $q(x \mid y) = p(x \mid y)$.*

**Proof.** Write $I(X; Y) = H(X) - H(X \mid Y)$ and insert the model:

$$
I(X; Y) - H(X) - E\left[\log q(x \mid y)\right]
= E\left[\log \frac{p(x \mid y)}{q(x \mid y)}\right]
= E_{y}\left[D_{\textrm{KL}}\!\left(p(\cdot \mid y) \,\|\, q(\cdot \mid y)\right)\right]
\geq 0
$$

by Gibbs' inequality, applied for each value of $y$ and averaged. Equality
holds iff the per-$y$ divergences all vanish, i.e., $q$ *is* the posterior.
$\blacksquare$

The bound :cite:`Barber.Agakov.2003` is the information-theoretic reading of
an encoder--decoder: if a decoder can reconstruct $X$ from $Y$ with low
cross-entropy, the pair must share a lot of information. Its weakness is the
$H(X)$ term, which is unknown in exactly the situations where we need the
bound (though it is a constant: *maximizing* the bound never needs it). The
same proof with $q(x \mid y)/p(x)$ in place of $q(x \mid y)$ gives the
equivalent ratio form $I \geq E[\log\left(q(x \mid y)/p(x)\right)]$, which we
will reuse for InfoNCE.

### Donsker--Varadhan and MINE

The second bound dispenses with normalized models altogether: it
characterizes KL divergence, hence mutual information, as a supremum over
arbitrary *critic* functions $T(x, y)$, with no requirement that anything
integrate to one. This is the same convex-duality pattern that drives the
f-GAN construction in :numref:`sec_mdl-divergences-distances`.

**Proposition (Donsker--Varadhan representation).** *For any distributions
$P, Q$ and any function $T$ with $E_Q[e^T] < \infty$,*

$$D_{\textrm{KL}}(P \| Q) \geq E_P[T] - \log E_Q\left[e^{T}\right],$$
:eqlabel:`eq_mdl-dv_bound`

*with equality if and only if $T = \log \frac{dP}{dQ} + c$ for a constant
$c$; taking the supremum over $T$ attains $D_{\textrm{KL}}(P \| Q)$ (both
statements for $P \ll Q$ with finite divergence)
:cite:`Donsker.Varadhan.1983`.*

**Proof.** Given $T$, define the *Gibbs distribution*
$g(x) = q(x)\, e^{T(x)} / Z$ with normalizer $Z = E_Q[e^T]$, the
distribution $Q$ "tilted" by the critic. Then

$$
D_{\textrm{KL}}(P \| Q) - \left(E_P[T] - \log Z\right)
= E_P\left[\log \frac{p}{q} - T + \log Z\right]
= E_P\left[\log \frac{p}{g}\right]
= D_{\textrm{KL}}(P \| G) \geq 0,
$$

by Gibbs' inequality (the eponymous coincidence is no accident). Equality
holds iff $P = G$, i.e., $p/q = e^{T}/Z$, i.e., $T = \log(p/q) + \log Z$.
$\blacksquare$

Specializing $P = P_{X,Y}$ and $Q = P_X \otimes P_Y$ bounds mutual
information: for any critic $T(x,y)$,

$$
I(X; Y) \geq E_{P_{X,Y}}[T] - \log E_{P_X \otimes P_Y}\left[e^{T}\right],
$$

where the second expectation is estimated by *shuffling*: pairing each $x$
with a $y$ drawn from a different example destroys the dependence and
samples the product of marginals. **MINE** (mutual information neural
estimation, :citet:`Belghazi.Baratin.Rajeshwar.ea.2018`) is exactly this
bound with $T$ a neural network trained by gradient ascent on
:eqref:`eq_mdl-dv_bound`, plus a fix for the subtlety that the
$\log E[e^T]$ term makes naive minibatch gradients biased.

### The NWJ Bound and the Bias--Variance Spectrum

The logarithm in front of $E_Q[e^T]$ causes the difficulty: it couples all
samples in the batch and gives the DV bound its bias and, at high MI, its
exploding variance (the expectation of $e^T$ under $Q$ is carried by the
same rare events that the McAllester--Stratos argument warned about). The
tangent-line inequality $\log z \leq z/e$ (equality at $z = e$) trades
tightness for tractability: applying it to $Z = E_Q[e^T]$ in
:eqref:`eq_mdl-dv_bound` gives, for every critic $T$,

$$I(X; Y) \geq E_{P_{X,Y}}[T] - e^{-1}\, E_{P_X \otimes P_Y}\left[e^{T}\right],$$
:eqlabel:`eq_mdl-nwj_bound`

the **NWJ bound** :cite:`Nguyen.Wainwright.Jordan.2010`, tight at
$T^* = 1 + \log\frac{dP}{dQ}$. Since $e^{-1} E_Q[e^T] = E_Q[e^{T-1}]$, the
right-hand side is exactly the f-GAN bound for the KL divergence,
$E_P[T] - E_Q[e^{T-1}]$, that Fenchel duality produced in
:numref:`sec_mdl-divergences-distances`: this is where we meet that bound
again. Because no log-of-average appears, plain
minibatch estimates of :eqref:`eq_mdl-nwj_bound` are unbiased for the bound
itself, but the $e^{T}$ term still has heavy tails, so the variance problem
survives. :numref:`fig_mdl-mi-variational-bounds` sketches the resulting
family: estimators stacked under the true value, trading bias
for variance, with the contrastive bound we meet next sitting lowest but
steadiest.

![Variational lower bounds on mutual information, after :citet:`Poole.Oord.Alemi.ea.2019` (schematic). Every bound sits at or below the true-MI diagonal, and each fails in its own way as $I$ grows: InfoNCE is low-variance but capped at $\log N$; the decoder-based Barber--Agakov bound is low-variance but biased downward by its decoder's gap (it flattens); and the critic-based NWJ and Donsker--Varadhan / MINE bounds, drawn as one curve, lose control of their batch-to-batch spread (shaded band), which explodes as $I$ grows.](../img/mdl-it-mi-variational-bounds.svg)
:label:`fig_mdl-mi-variational-bounds`

### InfoNCE: Estimation as Classification
:label:`sec_mdl-infonce`

InfoNCE takes a different route to taming the partition
function: it embeds the problem in a *classification* task, the guessing
game we simulated in :numref:`sec_mdl-mi-hard`. Draw one positive pair and
$N-1$ negatives, score every candidate with a critic $f(x, y)$, and ask a
softmax to point at the positive. The **InfoNCE loss**
:cite:`Oord.Li.Vinyals.2018` is the resulting cross-entropy,

$$\mathcal{L}_{\textrm{NCE}} = -E\left[\log \frac{e^{f(x, y_1)}}{\sum_{j=1}^{N} e^{f(x, y_j)}}\right],$$
:eqlabel:`eq_mdl-infonce_def`

where the expectation is over $(x, y_1) \sim P_{X,Y}$ and
$y_2, \ldots, y_N \sim P_Y$ i.i.d., and the associated mutual-information
*estimator* is the batch ceiling minus the loss,

$$\hat{I}_{\textrm{NCE}} = \log N - \mathcal{L}_{\textrm{NCE}}.$$
:eqlabel:`eq_mdl-infonce_est`

(Many papers, following :citet:`Oord.Li.Vinyals.2018`, write the loss with a
$\frac{1}{N}$ inside the logarithm and a compensating $+\log N$ outside; the
two cancel, so that convention and :eqref:`eq_mdl-infonce_def` are the same
number.) Note what the loss *is*: a $k$-class categorical cross-entropy
with $k = N$, the same loss dissected in
:numref:`subsec_mdl-nll-crossentropy` and
:numref:`sec_mdl-information_theory`, with classes that are *other examples
in the batch* rather than fixed labels. That is the entire engineering
appeal (this loss is exactly what deep-learning libraries optimize best),
and it is the loss
of CPC, SimCLR, and CLIP-style dual encoders, where $x$ and $y$ are two
views or two modalities of the same datum and the batch supplies the
negatives :cite:`Oord.Li.Vinyals.2018,radford2021learning`.

![The InfoNCE game. Left: an anchor $x$ is scored by the critic $f$ against the positive $y^+$ drawn jointly with it and $N-1$ negatives drawn independently from the marginal. Right: a softmax over the $N$ scores must point at the positive; the resulting cross-entropy certifies $I(X;Y) \geq \log N - \mathcal{L}_{\textrm{NCE}}$, a bound that can never exceed $\log N$.](../img/mdl-it-infonce-pos-neg.svg)
:label:`fig_mdl-infonce-pos-neg`

:numref:`fig_mdl-infonce-pos-neg` illustrates the classification problem. The
corresponding result is
:cite:`Poole.Oord.Alemi.ea.2019`: small InfoNCE loss *certifies* mutual
information.

**Proposition (InfoNCE bound).** *For any critic $f$ and any $N \geq 1$,*

$$I(X; Y) \geq \hat{I}_{\textrm{NCE}} = \log N - \mathcal{L}_{\textrm{NCE}}, \qquad \textrm{and } \hat{I}_{\textrm{NCE}} \textrm{ never exceeds } \log N.$$
:eqlabel:`eq_mdl-infonce_bound`

**Proof.** Write $V = (Y_1, \ldots, Y_N)$ for the whole bag of candidates,
where $(X, Y_1) \sim P_{X,Y}$ and the negatives $Y_{2:N} \sim P_Y$ are
independent of $(X, Y_1)$.

*Step 1: independent negatives add no information.* By the corollary to the chain rule,
independent side information adds nothing: $I(X; V) = I(X; Y_1)$.

*Step 2: define a normalized variational model.* Define

$$
g(x, v) = \frac{e^{f(x, y_1)}}{\frac{1}{N}\sum_{j=1}^N e^{f(x, y_j)}}.
$$

Under the product distribution $p(x)\,p(v)$, where all $N$ candidates,
including $y_1$, are i.i.d. draws from $P_Y$, exchangeability of the $y_j$
makes each of the $N$ terms $E\big[e^{f(x,y_i)} / \frac{1}{N}\sum_j
e^{f(x,y_j)}\big]$ equal, so for every fixed $x$,

$$
E_{p(v)}\left[g(x, V)\right]
= \frac{1}{N}\sum_{i=1}^{N} E\left[\frac{e^{f(x, y_i)}}{\frac{1}{N}\sum_{j} e^{f(x, y_j)}}\right]
= E\left[\frac{\frac{1}{N}\sum_{i} e^{f(x, y_i)}}{\frac{1}{N}\sum_{j} e^{f(x, y_j)}}\right] = 1.
$$

Hence $q(v \mid x) = p(v)\, g(x, v)$ is a *bona fide* normalized conditional
distribution over bags.

*Step 3: Barber--Agakov on the bag.* The ratio form of
:eqref:`eq_mdl-ba_bound` applied to $(X, V)$ with the model $q(v \mid x)$
gives

$$
I(X; V) \geq E_{p(x, v)}\left[\log \frac{q(v \mid x)}{p(v)}\right]
= E\left[\log g(X, V)\right]
= \log N - \mathcal{L}_{\textrm{NCE}}.
$$

Combining with Step 1 proves the bound. For the ceiling, the softmax assigns
the positive a probability at most $1$, so
$\mathcal{L}_{\textrm{NCE}} \geq 0$ and the estimator
:eqref:`eq_mdl-infonce_est` is at most $\log N$. $\blacksquare$

The two halves of the proposition are the two faces of contrastive learning.
The bound explains *why it works*: driving the classification loss down
provably drives shared information up, and the optimal critic recovers the
log density ratio up to a function of $x$,
$f^*(x,y) = \textrm{pmi}(x,y) + c(x)$ (Exercise :exref:`ex_mdl-mutual-information-optimal-critic`;
:cite:`Poole.Oord.Alemi.ea.2019`). With $N$ candidates, the bound has the exact
upper limit $\log N$. This fact partly motivates the large candidate sets used
by CPC, SimCLR, and CLIP, although optimization and negative diversity also
matter. One practical hyperparameter deserves attention: in
SimCLR-style systems the critic is a scaled cosine similarity between the
two embeddings, $f(x, y) = \textrm{sim}(z_x, z_y)/\tau$, and the
*temperature* $\tau$
:cite:`Chen.Kornblith.Norouzi.ea.2020` is the same knob as the distillation
temperature of :numref:`sec_mdl-information_theory`: lowering $\tau$
sharpens the softmax over candidates and concentrates the loss on the
hardest negatives.

The following experiment compares all three bounds with critics set to their
optima using the closed-form Gaussian
ratio. The remaining behavior therefore belongs to the *bounds*, rather than
critic optimization. We use a batch of $N=128$ and report the mean and standard
deviation over $200$ batches.

```{.python .input #mutual-information-perfect-critic-bounds}
rng = onp.random.default_rng(3)
N, B = 128, 200
print('true I    DV (MINE)         NWJ              InfoNCE')
for tm in [0.5, 1.0, 2.0, 4.0, 6.0]:
    rho = onp.sqrt(1 - onp.exp(-2 * tm))
    dv, nwj, nce = [], [], []
    for _ in range(B):
        x, y = gauss_pair(rng, rho, N)
        joint = pmi_gauss(x, y, rho)               # T* on positive pairs
        # A random permutation leaves ~1 fixed point per batch of N=128, so
        # one "negative" is secretly a true pair; the O(1/N) contamination
        # is negligible next to the heavy tails probed here.
        prod = pmi_gauss(x, rng.permutation(y), rho)   # T* on shuffled pairs
        dv.append(joint.mean() - onp.log(onp.mean(onp.exp(prod))))
        nwj.append(joint.mean() + 1 - onp.mean(onp.exp(prod)))
        f = pmi_gauss(x[:, None], y[None, :], rho)
        f -= f.max(axis=1, keepdims=True)
        ll = onp.diag(f) - onp.log(onp.exp(f).sum(axis=1))
        nce.append(ll.mean() + onp.log(N))
    print(f'{tm:5.1f}   {onp.mean(dv):6.2f} +-{onp.std(dv):6.2f} '
          f'  {onp.mean(nwj):6.2f} +-{onp.std(nwj):6.2f} '
          f'  {onp.mean(nce):6.2f} +-{onp.std(nce):6.2f}')
```

The table puts numbers on the critic-based bounds of
:numref:`fig_mdl-mi-variational-bounds`, here with exact critics. At low
mutual information all three estimators are accurate and well-behaved. As
the truth climbs past $\ln N \approx 4.85$ nats, they fail in their
characteristic ways: InfoNCE stays steady (standard deviation a few
hundredths of a nat) but saturates at its ceiling; NWJ's batch-to-batch
spread grows first, exploding to many nats while its mean falls far below
the truth; and DV's log-of-average adds upward bias and the largest blow-up,
until single batches return highly unstable values: at a true MI of $6$
nats its batch estimates average far *above* the truth with a spread of tens
of nats, despite using a perfect critic. Estimator choice is therefore a
bias--variance decision, not a ranking.

### Experiment: Learning the Critic
:label:`subsec_mdl-infonce-learned-critic`

Finally, let the critic be learned, as in practice. We train a small MLP
critic $f(x, y)$ with the InfoNCE loss on a correlated Gaussian pair with
$\rho = 0.99$, whose true mutual information we know exactly:
$-\tfrac{1}{2}\log(1 - 0.99^2) \approx 1.958$ nats. The
data are generated once in NumPy with a fixed
seed; training uses batches of $N = 128$, and we then evaluate the trained
critic's InfoNCE bound at several evaluation batch sizes to examine the
$\log N$ ceiling and its approach to the true value.

```{.python .input #mutual-information-infonce-train}
#@tab mxnet
rho, n_pool = 0.99, 4096
data_rng = onp.random.default_rng(0)
x_tr, y_tr = (a.astype('float32') for a in gauss_pair(data_rng, rho, n_pool))
x_ev, y_ev = (a.astype('float32') for a in gauss_pair(data_rng, rho, n_pool))

np.random.seed(0)
critic = gluon.nn.Sequential()
critic.add(gluon.nn.Dense(64, activation='relu'),
           gluon.nn.Dense(64, activation='relu'), gluon.nn.Dense(1))
critic.initialize(init.Xavier())
trainer = gluon.Trainer(critic.collect_params(), 'adam',
                        {'learning_rate': 1e-3})

def scores(x, y):
    """Matrix of critic scores f(x_i, y_j) for all pairs in a batch."""
    N = x.shape[0]
    xy = np.stack([np.tile(x.reshape(-1, 1), (1, N)),
                   np.tile(y.reshape(1, -1), (N, 1))], axis=-1)
    return critic(xy.reshape(-1, 2)).reshape(N, N)

def diag_minus_logsumexp(f):
    """Per-row log-softmax of the diagonal (positive-pair) scores."""
    m = f.max(axis=1, keepdims=True)
    lse = (m + np.log(np.exp(f - m).sum(axis=1, keepdims=True)))[:, 0]
    return (f * np.eye(f.shape[0])).sum(axis=1) - lse

N = 128
for step in range(400):
    idx = data_rng.choice(n_pool, N, replace=False)
    x, y = np.array(x_tr[idx]), np.array(y_tr[idx])
    with autograd.record():
        loss = -diag_minus_logsumexp(scores(x, y)).mean()
    loss.backward()
    trainer.step(1)

for N_ev in [2, 8, 32, 128]:
    vals = [float(diag_minus_logsumexp(
                scores(np.array(x_ev[i:i + N_ev]),
                       np.array(y_ev[i:i + N_ev]))).mean())
            for i in range(0, n_pool, N_ev)]
    print(f'N = {N_ev:4d}: ln N = {onp.log(N_ev):.3f}, '
          f'InfoNCE bound = {onp.log(N_ev) + onp.mean(vals):.3f} nats')
print(f'true I(X;Y) = {-0.5 * onp.log(1 - rho**2):.3f} nats')
```

```{.python .input #mutual-information-infonce-train}
#@tab pytorch
rho, n_pool = 0.99, 4096
data_rng = onp.random.default_rng(0)
x_tr, y_tr = (torch.tensor(a, dtype=torch.float32)
              for a in gauss_pair(data_rng, rho, n_pool))
x_ev, y_ev = (torch.tensor(a, dtype=torch.float32)
              for a in gauss_pair(data_rng, rho, n_pool))

torch.manual_seed(0)
critic = torch.nn.Sequential(
    torch.nn.Linear(2, 64), torch.nn.ReLU(),
    torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
opt = torch.optim.Adam(critic.parameters(), lr=1e-3)

def scores(x, y):
    """Matrix of critic scores f(x_i, y_j) for all pairs in a batch."""
    N = len(x)
    xy = torch.stack([x.reshape(N, 1).expand(N, N),
                      y.reshape(1, N).expand(N, N)], dim=-1)
    return critic(xy.reshape(-1, 2)).reshape(N, N)

N = 128
for step in range(400):
    idx = torch.tensor(data_rng.choice(n_pool, N, replace=False))
    f = scores(x_tr[idx], y_tr[idx])
    loss = (torch.logsumexp(f, dim=1) - f.diag()).mean()
    opt.zero_grad(); loss.backward(); opt.step()

with torch.no_grad():
    for N_ev in [2, 8, 32, 128]:
        vals = [(scores(x_ev[i:i + N_ev], y_ev[i:i + N_ev]).diag()
                 - torch.logsumexp(scores(x_ev[i:i + N_ev],
                                          y_ev[i:i + N_ev]), dim=1)
                 ).mean().item()
                for i in range(0, n_pool, N_ev)]
        print(f'N = {N_ev:4d}: ln N = {onp.log(N_ev):.3f}, '
              f'InfoNCE bound = {onp.log(N_ev) + onp.mean(vals):.3f} nats')
print(f'true I(X;Y) = {-0.5 * onp.log(1 - rho**2):.3f} nats')
```

```{.python .input #mutual-information-infonce-train}
#@tab tensorflow
rho, n_pool = 0.99, 4096
data_rng = onp.random.default_rng(0)
x_tr, y_tr = (a.astype('float32') for a in gauss_pair(data_rng, rho, n_pool))
x_ev, y_ev = (a.astype('float32') for a in gauss_pair(data_rng, rho, n_pool))

tf.random.set_seed(0)
critic = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'), tf.keras.layers.Dense(1)])
opt = tf.keras.optimizers.Adam(1e-3)

def scores(x, y):
    """Matrix of critic scores f(x_i, y_j) for all pairs in a batch."""
    N = tf.shape(x)[0]
    xy = tf.stack([tf.tile(tf.reshape(x, (-1, 1)), (1, N)),
                   tf.tile(tf.reshape(y, (1, -1)), (N, 1))], axis=-1)
    return tf.reshape(critic(tf.reshape(xy, (-1, 2))), (N, N))

@tf.function
def step(x, y):
    with tf.GradientTape() as tape:
        f = scores(x, y)
        loss = tf.reduce_mean(tf.reduce_logsumexp(f, axis=1)
                              - tf.linalg.diag_part(f))
    grads = tape.gradient(loss, critic.trainable_variables)
    opt.apply_gradients(zip(grads, critic.trainable_variables))

N = 128
for s in range(400):
    idx = data_rng.choice(n_pool, N, replace=False)
    step(tf.constant(x_tr[idx]), tf.constant(y_tr[idx]))

for N_ev in [2, 8, 32, 128]:
    vals = []
    for i in range(0, n_pool, N_ev):
        f = scores(tf.constant(x_ev[i:i + N_ev]),
                   tf.constant(y_ev[i:i + N_ev]))
        vals.append(float(tf.reduce_mean(
            tf.linalg.diag_part(f) - tf.reduce_logsumexp(f, axis=1))))
    print(f'N = {N_ev:4d}: ln N = {onp.log(N_ev):.3f}, '
          f'InfoNCE bound = {onp.log(N_ev) + onp.mean(vals):.3f} nats')
print(f'true I(X;Y) = {-0.5 * onp.log(1 - rho**2):.3f} nats')
```

```{.python .input #mutual-information-infonce-train}
#@tab jax
rho, n_pool = 0.99, 4096
data_rng = onp.random.default_rng(0)
x_tr, y_tr = (jnp.array(a, dtype='float32')
              for a in gauss_pair(data_rng, rho, n_pool))
x_ev, y_ev = (jnp.array(a, dtype='float32')
              for a in gauss_pair(data_rng, rho, n_pool))

def init_mlp(key, sizes=(2, 64, 64, 1)):
    params = []
    for d_in, d_out in zip(sizes[:-1], sizes[1:]):
        key, sub = jax.random.split(key)
        params.append((jax.random.normal(sub, (d_in, d_out))
                       * jnp.sqrt(2 / d_in), jnp.zeros(d_out)))
    return params

def scores(params, x, y):
    """Matrix of critic scores f(x_i, y_j) for all pairs in a batch."""
    h = jnp.stack(jnp.meshgrid(x, y, indexing='ij'), axis=-1)
    for W, b in params[:-1]:
        h = jax.nn.relu(h @ W + b)
    return (h @ params[-1][0] + params[-1][1])[..., 0]

def loss_fn(params, x, y):
    f = scores(params, x, y)
    return (jax.scipy.special.logsumexp(f, axis=1) - jnp.diag(f)).mean()

params = init_mlp(jax.random.PRNGKey(0))
opt = optax.adam(1e-3)
state = opt.init(params)

@jax.jit
def step(params, state, x, y):
    grads = jax.grad(loss_fn)(params, x, y)
    updates, state = opt.update(grads, state)
    return optax.apply_updates(params, updates), state

N = 128
for s in range(400):
    idx = data_rng.choice(n_pool, N, replace=False)
    params, state = step(params, state, x_tr[idx], y_tr[idx])

for N_ev in [2, 8, 32, 128]:
    vals = []
    for i in range(0, n_pool, N_ev):
        f = scores(params, x_ev[i:i + N_ev], y_ev[i:i + N_ev])
        vals.append(float((jnp.diag(f)
                    - jax.scipy.special.logsumexp(f, axis=1)).mean()))
    print(f'N = {N_ev:4d}: ln N = {onp.log(N_ev):.3f}, '
          f'InfoNCE bound = {onp.log(N_ev) + onp.mean(vals):.3f} nats')
print(f'true I(X;Y) = {-0.5 * onp.log(1 - rho**2):.3f} nats')
```

In this controlled evaluation, the trained critic and data pool are fixed while
the number of negatives changes. At $N=2$, InfoNCE cannot exceed $\ln2$; at
$N=8$, the ceiling exceeds the true $1.958$ nats but the learned bound remains
about $1.4$; by $N=128$, it reaches about $1.89$. The negative count therefore
sets InfoNCE's exact upper range here, but it is not the sole determinant of the
result: critic approximation, optimization, in-batch dependence, and finite
evaluation data also contribute.

## The Limits of Mutual Information

### The Compression-Phase Debate
:label:`subsec_mdl-compression-debate`

The information plane has also been central to a debate about deep learning.
:citet:`Shwartz-Ziv.Tishby.2017` plotted estimated $(I(X;Z), I(Y;Z))$
trajectories of ordinary classifiers *during training* and reported a
two-phase dynamic, a fast *fitting* phase (both informations rise) followed
by a long *compression* phase ($I(X;Z)$ falls while $I(Y;Z)$ holds),
proposed as the mechanism by which deep networks generalize.
:citet:`Saxe.Bansal.Dapello.ea.2018` re-examined the claim and found the
compression phase to be largely an artifact: it appears with saturating
activations (tanh squashes activations into the bins' edges, which the
binned MI estimator reads as compression) and disappears with ReLU networks,
which can generalize without any measured compression. Moreover, for
a *deterministic* network, $I(X; Z)$ is infinite for continuous inputs (the
singular case flagged at the definition :eqref:`eq_mdl-mut_ent_def`) and
equals the constant $H(X)$ for discrete inputs that the network maps to
distinct activations: whatever the plotted trajectories track, it is
a property of the binning, not a mutual information. Where the debate
stands: the
IB *objective* :eqref:`eq_mdl-ib_lagrangian` is well-posed for explicitly
stochastic encoders and trains well (VIB improves robustness and
calibration in practice), but the claim that standard training *implicitly*
performs IB compression remains unsettled. The debate illustrates the
difficulty of interpreting MI estimates.

### What Mutual Information Estimates Establish
:label:`sec_mdl-mi-limits`

The preceding results suggest several guidelines for interpreting mutual
information estimates.

* **Separate estimates, bounds, and confidence guarantees.** InfoNCE, NWJ,
  DV/MINE, and decoder objectives are variational bounds evaluated with finite
  samples and often imperfect critics. Histogram, kernel, and nearest-neighbor
  methods are point estimators with different bias and tuning problems. The
  general $O(\log N)$ theorem concerns distribution-free high-confidence lower
  guarantees; it does not give every point estimator an exact range. InfoNCE has
  its own exact $\log N$ cap, whereas empirical DV and NWJ values are unbounded
  and can become extremely variable.
* **A useful objective need not provide an accurate estimate.**
  :citet:`Tschannen.Djolonga.Rubenstein.ea.2020` show that downstream
  representation quality correlates poorly with the tightness of the MI
  bound being maximized: looser bounds sometimes yield *better*
  representations, and InfoNCE's success owes as much to the inductive
  biases of the critic architecture and the choice of views as to its
  information content. Maximizing an MI bound can therefore be a useful
  training signal without providing a reliable numerical estimate of mutual
  information.
* **Invariances complicate estimation.** MI's reparameterization invariance
  provides a broad notion of dependence but makes estimation difficult; any
  pipeline that needs a trustworthy dependence number
  on high-dimensional data should confront that tension directly (e.g., by
  validating its estimator on a known-MI synthetic pair, as we did here).
* **The formal guarantees remain useful.** The definitions
  and their calculus, the data-processing inequality, Fano's floor, the
  bounds' validity as *bounds*, and the closed-form anchors used to test
  estimators provide a sound basis for analysis, even when high-dimensional
  numerical estimates are difficult to interpret.

## Summary

* Estimating MI is statistically hard: any distribution-free,
  high-confidence lower guarantee from $N$ samples grows at most on the order of
  $\log N$ (McAllester--Stratos). Separately, InfoNCE has an exact $\log N$ cap
  even with an ideal critic.

* Practical estimators are variational lower bounds: Barber--Agakov
  (decoder), Donsker--Varadhan/MINE and NWJ (critics), and InfoNCE
  (contrastive classification, a categorical cross-entropy). InfoNCE
  certifies $I \geq \log N - \mathcal{L}$, is low-variance but saturates at
  $\log N$; DV/NWJ are tighter in expectation but their variance explodes
  at high MI.

## Exercises

1. [code] **Bias of the histogram estimator.** The plug-in estimate of
   `binned_mi` depends on the sample size $n$, on the number $b$ of bins per
   axis, and on the coordinates of the data. Mutual information depends on
   none of them.
    1. For an independent pair, $2n$ times the plug-in estimate is the
       $G$-statistic of the test for independence. When every cell's
       expected count is large, this statistic is approximately
       $\chi^2$-distributed with $(b-1)^2$ degrees of freedom. Predict the
       mean estimate at $\rho = 0$ for $n = 200{,}000$ and $b = 40$, and
       compare it with the value $0.0027$ that the cell of
       :numref:`subsec_mdl-mi-gaussian` prints for $\rho = 0$. Why is the
       printed value smaller?
    1. Replace each coordinate by its rank divided by $n$ before binning, so
       that every bin of each axis holds the same number of samples. Average
       the estimate over $20$ independent pairs with $\rho = 0$ for
       $n \in \{2000, 20{,}000\}$ and $b \in \{10, 40\}$, and compare with the
       prediction of (a). For which setting does the prediction fail, and
       why?
    1. Apply $x \mapsto x^3$ to the first coordinate of a pair with
       $\rho = 0.9$ and $n = 200{,}000$, and estimate $I(X;Y)$ with and
       without the rank transform. Why is the rank-based estimate exactly
       unchanged?
    1. Repeat (c) at $\rho = 0.99$ with $b \in \{40, 100\}$, using ranks.
       Together with (b), what does this suggest about how $b$ should grow
       with $n$?
   :exlabel:`ex_mdl-mutual-information-histogram-bias`

1. **Barber--Agakov with a Gaussian decoder.** Take the standard bivariate
   Gaussian pair with correlation $\rho$ and the decoders
   $q(x \mid y) = \mathcal{N}(x; a y, s^2)$ with parameters $a \in \mathbb{R}$
   and $s^2 > 0$. In the bound :eqref:`eq_mdl-ba_bound`, use the
   differential entropy $h(X) = \tfrac{1}{2}\log(2\pi e)$ in place of
   $H(X)$.
    1. Evaluate the right-hand side of :eqref:`eq_mdl-ba_bound` as a
       function of $a$ and $s^2$. Maximize it over both parameters and show
       that the maximum equals :eqref:`eq_mdl-gaussian_mi`.
    1. A decoder trained with the squared error alone corresponds to the
       fixed variance $s^2 = 1$. Show that the best bound in this family is
       $\rho^2/2$, and compare it with the true value at $\rho = 0.9$ and
       $\rho = 0.99$. How does the gap behave as $\rho \to 1$?
    1. Let $\hat x(y)$ be any predictor of $X$ with mean squared error $m$.
       Use the decoder $\mathcal{N}(x; \hat x(y), m)$ to show that
       $I(X;Y) \geq -\tfrac{1}{2}\log m$. What is the corresponding bound
       when $X \sim \mathcal{N}(0, v)$? How does it relate to the Gaussian
       rate--distortion function :eqref:`eq_mdl-gaussian-rd` of
       :numref:`subsec_mdl-rate-distortion` at distortion $m$?
   :exlabel:`ex_mdl-mutual-information-ba-gaussian-decoder`

1. **Temperature of a cosine critic.** SimCLR-style systems score
   candidates with $f(x, y) = s(x, y)/\tau$, where
   $s(x, y) = \textrm{sim}(z_x, z_y) \in [-1, 1]$ is a cosine similarity and
   $\tau > 0$ is the temperature (:numref:`sec_mdl-infonce`). For one anchor
   $x$, write $s_j = s(x, y_j)$ and let $\ell$ be the loss inside the
   expectation :eqref:`eq_mdl-infonce_def`.
    1. Show that
       $\mathcal{L}_{\textrm{NCE}} \geq \log\big(1 + (N-1)e^{-2/\tau}\big)$,
       and hence $\hat{I}_{\textrm{NCE}} < 2/\tau$ for every $N$. Evaluate
       the cap $\log N - \log\big(1 + (N-1)e^{-2/\tau}\big)$ for $N = 4096$
       and $\tau \in \{1, 0.5, 0.1\}$. When does the temperature rather than
       the batch size limit the bound?
    1. Show that $\partial \ell / \partial s_j = p_j/\tau$ for every negative
       $j \geq 2$, where $p_j$ is the softmax probability of candidate $j$.
       Express $\partial \ell / \partial s_1$ through the $p_j$.
    1. Show that $\tau \ell \to \max\{0, \max_{j \geq 2} s_j - s_1\}$ as
       $\tau \to 0$, and that

       $$\ell = \log N - \frac{1}{\tau}\Big(s_1 - \frac{1}{N}\sum_{j=1}^N s_j\Big) + O(\tau^{-2}) \quad \textrm{as } \tau \to \infty.$$

       Which negatives drive the gradient in each limit
       :cite:`Wang.Liu.2021`?
   :exlabel:`ex_mdl-mutual-information-temperature`

1. **Entropy bounds from samples.** ●
   :numref:`subsec_mdl-mi-distribution-free` states the theorem of
   :citet:`McAllester.Stratos.2020` informally. This problem proves a
   version of it for entropy. Draw a sample $S$ of $N$
   i.i.d. values from a discrete distribution $p$. The *type* of $S$ records,
   for each $i \geq 1$, how many distinct values occur exactly $i$ times in
   $S$; it forgets which values these are. Let $B$ be a function of the type
   that is a lower confidence bound for every discrete distribution:
   $P_{S \sim p^N}\big(B \leq H(p)\big) \geq 1 - \delta$ for every $p$,
   with $\delta < 1/2$.
    1. Show that $N$ draws from the uniform distribution on $M$ values
       contain a repeated value with probability at most $N(N-1)/(2M)$.
    1. Let $T_0$ be the type of a sample of $N$ distinct values. Apply the
       validity of $B$ to the uniform distribution on $N^2$ values and
       conclude that $B(T_0) \leq 2 \log N$.
    1. Let $p$ be uniform on $M = 2^{256}$ values, so that
       $H(p) \approx 177$ nats. For $N = 10^6$, bound the probability that
       $B$ reports more than $2 \log N \approx 27.6$ nats.
    1. For $X = Y$ uniform on $M$ values, $I(X;Y) = H(X)$. Explain why a
       distribution-free lower bound on $I(X;Y)$ that depends on the sample
       of pairs only through its type obeys the same limit. Which assumption
       must an estimator add to certify more than $2 \log N$ nats from $N$
       samples?
   :exlabel:`ex_mdl-mutual-information-entropy-bounds-samples`

1. **Optimal InfoNCE critic.** ● For one anchor $x$ and one bag
   $v = (y_1, \ldots, y_N)$, write
   $\ell(x, v) = -\log\big(e^{f(x, y_1)} / \sum_{j=1}^N e^{f(x, y_j)}\big)$,
   so that $\mathcal{L}_{\textrm{NCE}}$ of :eqref:`eq_mdl-infonce_def` is
   its expectation.
    1. Show that replacing $f(x, y)$ by $f(x, y) + c(x)$, for any function
       $c$, leaves $\mathcal{L}_{\textrm{NCE}}$ unchanged.
    1. Shuffle the bag so that the positive sits at a uniformly random
       position $J$ and the other $N - 1$ positions hold the negatives. Show
       that

       $$P(J = i \mid x, y_1, \ldots, y_N) = \frac{e^{\textrm{pmi}(x, y_i)}}{\sum_{j=1}^N e^{\textrm{pmi}(x, y_j)}}.$$

       Use Gibbs' inequality to conclude that every critic
       $f(x, y) = \textrm{pmi}(x, y) + c(x)$ minimizes
       $\mathcal{L}_{\textrm{NCE}}$.
    1. For this critic, show that

       $$\hat{I}_{\textrm{NCE}} = I(X;Y) - E\left[\log \frac{1}{N}\sum_{j=1}^N e^{\textrm{pmi}(x, y_j)}\right].$$

       Use the law of large numbers to argue that the correction tends to
       zero as $N \to \infty$ when $I(X;Y)$ is finite. How does this explain
       the curves that `infonce_oracle` produces in :numref:`sec_mdl-mi-hard`?
   :exlabel:`ex_mdl-mutual-information-optimal-critic`

1. **Negative sampling as shifted PMI.** ● Skip-gram with negative
   sampling of :numref:`subsec_negative-sampling` scores a center word $w$
   and a context word $o$ by $s(w, o) = \mathbf{u}_o^\top \mathbf{v}_w$ and
   draws $K$ noise words from a distribution $P$. Let $p(w, o)$ be the
   frequency of the pair among all (center, context) pairs of the corpus,
   with marginals
   $p(w)$ and $p(o)$. Averaged over the corpus, the loss of
   :numref:`subsec_negative-sampling` is

   $$\ell = -\sum_{w, o} p(w, o) \log \sigma\big(s(w, o)\big) - K \sum_{w} p(w) \sum_{o} P(o) \log \sigma\big(-s(w, o)\big),$$

   where $\sigma$ is the sigmoid :eqref:`eq_sigma-f`. Assume that the
   embeddings have enough dimensions for every score $s(w, o)$ to be chosen
   independently.
    1. Show that $\ell$ is minimized at
       $s^*(w, o) = \log \frac{p(w, o)}{p(w)\, P(o)} - \log K$, which equals
       $\textrm{pmi}(w, o) - \log K$ when $P$ is the unigram distribution
       $p(o)$ :cite:`Levy.Goldberg.2014`.
    1. word2vec draws noise words with probability proportional to their
       count raised to the power $0.75$ (:numref:`sec_word2vec_data`). Show
       that $s^*$ then differs from $\textrm{pmi}(w, o) - \log K$ by a term
       that depends on $o$ alone. For which context words is this term
       negative? How does it act on the rare pairs of problem
       :exref:`ex_mdl-mutual-information-rare-pairs`?
    1. Compare $s^*$ with the optimal InfoNCE critic of problem
       :exref:`ex_mdl-mutual-information-optimal-critic`b. Why does the
       binary classification of negative sampling fix the additive term that
       InfoNCE leaves free?
   :exlabel:`ex_mdl-mutual-information-negative-sampling-pmi`

1. [extended] **Compression phase at small scale.** ●●
   :citet:`Shwartz-Ziv.Tishby.2017` trained fully connected tanh networks of
   widths 12-10-7-5-4-3-2 on all $4096$ binary input patterns of $12$ bits.
   For each hidden layer $T$ they binned every unit's activation into $30$
   equal intervals of $[-1, 1]$ and computed $I(X;T)$ and $I(Y;T)$ from the
   binned values. :numref:`subsec_mdl-compression-debate` describes the
   debate about the compression phase they reported.
    1. Let $X$ be uniform on the $4096$ patterns and let $T_{\textrm{bin}}$
       be the binned layer of a deterministic network. Show that the binned
       estimate of $I(X;T)$ equals $H(T_{\textrm{bin}})$ and never exceeds
       $\ln 4096 \approx 8.32$ nats. When does it decrease during training?
    1. Fix a random vector $\mathbf{w} \in \mathbb{R}^{12}$ and let $Y = 1$
       when $\mathbf{w}^\top \mathbf{x}$ exceeds its median over the $4096$
       patterns. Train the tanh network, with a softmax over its two output
       units, by SGD with the cross-entropy loss on a random $85\%$ of the
       patterns. Plot the trajectory of every hidden layer in the
       information plane $\big(I(X;T), I(Y;T)\big)$ over training.
    1. Repeat (b) with ReLU hidden units, binning each unit into $30$ equal
       intervals between $0$ and its largest activation over the run.
    1. Record the fraction of tanh units with $|t| > 0.99$ during training.
       Does the decrease of the binned $I(X;T)$ in (b) track this fraction?
       Which conclusions about compression survive the change of activation
       in (c) :cite:`Saxe.Bansal.Dapello.ea.2018`?
   :exlabel:`ex_mdl-mutual-information-compression-phase`

[Discussions](https://d2l.discourse.group/)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §26.0]{.kicker}

**Estimating mutual information**
:::
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Why measuring MI is hard]{.dtitle}

[high-dimensional estimation and the $\log N$ ceiling]{.dsub}
:::
:::

::: {.slide title="Estimation in high dimensions"}
[Estimation limits]{.kicker}

A histogram needs $b^d$ cells, almost all empty in high dimensions. The very
reparameterization-invariance that makes MI the *right* notion of dependence
is what makes it hard to estimate: a distribution-free estimator must survive
every coordinate warping at once.

::: {.d2l-note .rule}
Distribution-free lower confidence guarantees from a finite sample cannot
certify arbitrarily large dependence.
:::
:::

::: {.slide title="Nearest-Neighbor Estimation Remains Dimension-Limited"}
[KSG]{.kicker}

Kraskov's $k$-NN estimator replaces bins with adaptive neighborhood radii,
and with a fixed $k$, $2{,}000$ samples land within hundredths of a nat of the
closed form in the experiment below:

@!mdl-mutual-information-the-curse-of-estimation

The low-dimensional result does not extend automatically to image-scale
variables, where neighbor distances concentrate.
:::

::: {.slide title="InfoNCE cannot exceed log N"}
[Estimator-specific range]{.kicker}

Because the classification loss is nonnegative, InfoNCE is at most $\log N$.
Even an exact likelihood-ratio critic approaches this ceiling when the true MI
is larger:

@!mutual-information-log-n-ceiling

With 256 candidates, the InfoNCE lower bound is at most $\ln256\approx5.5$ nats.
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[Variational bounds and InfoNCE]{.dtitle}

[bound it below, then maximize the bound]{.dsub}
:::
:::

::: {.slide title="Bound MI from below"}
[Variational bounds]{.kicker}

::: {.cols .vc}
::: {.col}
- **Barber–Agakov**: a decoder, $I \ge H(X)+\mathbb{E}[\log q(x\mid y)]$,
  low variance but biased down by the decoder's gap.
- **Donsker–Varadhan / MINE**: $I \ge \mathbb{E}_{P}[T] - \log\mathbb{E}_{Q}[e^T]$.
- **NWJ**: the same with $e^{-1}\mathbb{E}_Q[e^T]$, unbiased for the bound
  but with heavy $e^T$ tails.

All bound below; all are maximized, not measured.
:::

::: {.col .fig}
@fig:mdl-it-mi-variational-bounds
:::
:::
:::

::: {.slide title="Bias against variance"}
[Estimator behavior]{.kicker}

With *exact* critics on a known-MI pair: past $\log N$, NWJ's batch-to-batch
spread explodes (unbiased for its bound, heavy tails); DV's log-of-average adds
bias on top of the blow-up; InfoNCE stays steady but saturates at the
ceiling:

@!mutual-information-perfect-critic-bounds

The estimators trade bias against variance.
:::

::: {.slide title="InfoNCE: estimation as classification"}
[Contrastive]{.kicker}

::: {.cols .vc}
::: {.col}
Pick the positive among $N-1$ negatives: $\mathcal L_{\mathrm{NCE}}$ is a plain
softmax cross-entropy, and the estimator is the ceiling minus the loss. For any
critic,

$$I(X;Y) \ge \hat I_{\mathrm{NCE}} = \log N - \mathcal L_{\mathrm{NCE}} \le \log N,$$

with optimal critic $f^\star = \mathrm{pmi}+c$. This is the loss of CPC,
SimCLR, and CLIP.
:::

::: {.col .fig}
@fig:mdl-it-infonce-pos-neg
:::
:::
:::

::: {.slide title="Negative Count Caps the InfoNCE Bound"}
[In practice]{.kicker}

A small MLP critic at $\rho=0.99$ ($I\approx1.96$ nats): at $N=2$ the bound
cannot clear $\ln 2$; widen the batch and it climbs toward the truth:

@!mutual-information-infonce-train

. . .

In SimCLR and CLIP the critic is a scaled cosine,
$f = \mathrm{sim}(z_x,z_y)/\tau$; the *temperature* is distillation's knob
again, concentrating the loss on the hardest negatives.
:::

::: {.slide title="What MI estimates establish"}
[The caveats]{.kicker}

::: {.d2l-note .rule}
Read the MI *number* as a capped bound; judge the *objective* by its
representations. The calculus, the DPI, Fano's lower bound, and the variational
bounds remain valid when interpreted with their stated assumptions.
:::

The "compression phase" of training is contested (Saxe et al.: an estimator
artifact for $\tanh$ nets). Read MI as a training signal, not a readout.
:::
