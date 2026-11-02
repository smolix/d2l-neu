# Mutual Information and Representation Learning
:label:`sec_mdl-mutual-information`

The entropies and divergences of :numref:`sec_mdl-information_theory` compare
individual distributions. Mutual information instead measures the statistical
dependence between two random variables. This quantity motivates several
contrastive and self-supervised methods, including SimCLR, CPC, and CLIP-style
dual encoders
:cite:`Chen.Kornblith.Norouzi.ea.2020,Oord.Li.Vinyals.2018,radford2021learning`.
Their InfoNCE objective provides a lower bound on mutual information. Such
bounds are useful training objectives, but they are difficult to interpret as
measurements: a distribution-free estimate based on $N$ samples cannot in
general certify much more than $\log N$ nats. We first develop mutual
information and the data-processing inequality, then study this estimation
barrier, variational bounds, InfoNCE, and the information bottleneck.

As in :numref:`sec_mdl-information_theory`, **we work in nats** (natural
logarithm) throughout, flagging the occasional conversion to bits explicitly;
recall $1 \textrm{ bit} = \ln 2 \approx 0.693$ nats. We also lean on two
facts from that section without re-proving them: Gibbs' inequality
($D_{\textrm{KL}} \geq 0$ with equality iff the distributions agree) and the
decomposition $\textrm{CE} = H + D_{\textrm{KL}}$.

```{.python .input #mutual-information-imports}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import numpy as onp  # plain NumPy for the probability demos
```

```{.python .input #mutual-information-imports}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as onp  # plain NumPy for the probability demos
```

```{.python .input #mutual-information-imports}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as onp  # plain NumPy for the probability demos
```

```{.python .input #mutual-information-imports}
#@tab jax
%matplotlib inline
from d2l import jax as d2l
import numpy as onp  # plain NumPy for the probability demos
```

## Mutual Information: Definitions and Properties
:label:`sec_mdl-mi-recap`

Throughout, $(X,Y)$ is a pair of random variables with joint distribution
$P_{X,Y}$ and marginals $P_X,P_Y$. We begin the entropy bookkeeping in the
discrete case. For continuous variables the same symbols denote differential
entropies only when the relevant integrals are finite; the KL definition of
mutual information below remains primary and avoids undefined differences such
as $\infty-\infty$. The basic question is how observing $Y$ changes our
uncertainty about $X$.

### From One Variable to Two: Joint and Conditional Entropy

For discrete variables, the *joint entropy* averages the self-information of
the pair:

$$H(X, Y) = -E_{(x, y) \sim P_{X,Y}} [\log p_{X, Y}(x, y)],$$
:eqlabel:`eq_mdl-joint_ent_def`

a sum over outcomes. The two extremes orient the discrete definition: if
$Y=X$, the pair carries no more information than either variable alone,
$H(X,Y)=H(X)$; if $X$ and $Y$ are independent, surprises add,
$H(X,Y)=H(X)+H(Y)$. Every discrete dependent pair sits between these bounds.
For a continuous $Y=X$, by contrast, the joint law lies on a lower-dimensional
diagonal, has no ordinary joint density with respect to area, and has infinite
mutual information; the discrete identity must not be transferred verbatim.

Often we care about a directed version of the question. Take $X$ to be the
pixels of an image and $Y$ its class label. The image is a complicated,
information-rich object, but once we have *seen* it, the label holds few
surprises: a legible digit already reveals its class. The
*conditional entropy*

$$H(Y \mid X) = - E_{(x, y) \sim P_{X,Y}} [\log p(y \mid x)],$$
:eqlabel:`eq_mdl-cond_ent_def`

with $p(y \mid x) = p_{X,Y}(x, y)/p_X(x)$, measures exactly this residual:
the average surprise in $Y$ *after* $X$ is known. The three quantities are
linked by a bookkeeping identity.

**Proposition (chain rule for entropy).** *For discrete variables, or for
continuous variables when the displayed entropies are finite,
$H(X,Y)=H(X)+H(Y\mid X)$.*

**Proof.** Inside the expectation defining $H(X,Y)$, factor the joint density:
$\log p_{X,Y}(x,y) = \log p_X(x) + \log p(y \mid x)$. Taking
$-E_{(x,y) \sim P_{X,Y}}$ of both sides turns the left side into $H(X,Y)$ and
the two right-hand terms into $H(X)$ and $H(Y \mid X)$.
$\blacksquare$

Read it as an accounting statement: the total information in the pair is the
information in $X$ plus whatever $Y$ adds once $X$ is known.

Consider the $2 \times 2$ joint distribution

$$
p_{X,Y} = \begin{pmatrix} 0.1 & 0.4 \\ 0.2 & 0.3 \end{pmatrix},
$$

with rows indexing $x \in \{0, 1\}$ and columns $y \in \{0, 1\}$; the entries
sum to $1$ as a joint p.m.f. must. The marginals come *from the joint* by
summing rows and columns: $p_X = (0.5, 0.5)$ and $p_Y = (0.3, 0.7)$. From the
definitions, $H(X,Y) \approx 1.2799$ nats, $H(X) = \ln 2 \approx 0.6931$
nats, $H(Y) \approx 0.6109$ nats, and the chain rule predicts
$H(Y \mid X) = H(X,Y) - H(X) \approx 0.5867$ nats, which the direct
definition :eqref:`eq_mdl-cond_ent_def` confirms.

```{.python .input #mutual-information-joint-conditional}
p_xy = onp.array([[0.1, 0.4],
                  [0.2, 0.3]])           # a valid joint: entries sum to 1
p_x, p_y = p_xy.sum(axis=1), p_xy.sum(axis=0)  # marginals FROM the joint

entropy = lambda p: float(-(p * onp.log(p)).sum())
H_xy, H_x, H_y = entropy(p_xy), entropy(p_x), entropy(p_y)
# Conditional entropy directly: p(y|x) = p(x,y) / p(x), weighted by p(x,y)
H_y_given_x = float(-(p_xy * onp.log(p_xy / p_x[:, None])).sum())

print(f'marginals: p_x = {p_x}, p_y = {p_y}')
print(f'H(X,Y) = {H_xy:.4f}, H(X) = {H_x:.4f}, H(Y) = {H_y:.4f} nats')
print(f'H(Y|X) = {H_y_given_x:.4f} = H(X,Y) - H(X) = {H_xy - H_x:.4f} nats')
```

### Mutual Information as a Divergence from Independence

The conditional entropy measured what $Y$ keeps *to itself*; we want what
$X$ and $Y$ *share*. The definition needs no entropy at all. If $X$ and $Y$
were independent, their joint
distribution would be the product of marginals $P_X \otimes P_Y$, with density
$p_X(x)\,p_Y(y)$. The *mutual information* is the KL divergence from that
hypothetical independent world to the actual joint:

$$I(X; Y) = D_{\textrm{KL}}\!\left(P_{X,Y} \,\|\, P_X \otimes P_Y\right) = E_{(x, y) \sim P_{X,Y}} \left[ \log\frac{p_{X, Y}(x, y)}{p_X(x)\, p_Y(y)} \right].$$
:eqlabel:`eq_mdl-mut_ent_def`

In words: mutual information is *how far the pair is from being independent*,
measured in the same units as every other quantity in this chapter. (The
semicolon in $I(X; Y)$ is the standard notation, distinguishing the two
arguments from a single joint object.) The definition presumes the joint has
a density with respect to the product of the marginals; when it does not,
for instance when $Y = f(X)$ is a deterministic function of a continuous
$X$, the divergence is infinite, and so is $I(X; Y)$. Symmetry and
nonnegativity are immediate from the divergence form, and the entropy
identities follow by expanding the logarithm.

**Proposition (properties of mutual information).** *For any pair $(X,Y)$,
$I(X;Y)$ is symmetric and non-negative, with equality if and only if $X$ and
$Y$ are independent. When the relevant discrete or differential entropies are
finite, it also satisfies*

$$
I(X;Y)=H(X)+H(Y)-H(X,Y)=H(X)-H(X\mid Y)=H(Y)-H(Y\mid X).
$$

**Proof.** When the entropy terms are finite, expand the logarithm in
:eqref:`eq_mdl-mut_ent_def`:

$$
\begin{aligned}
I(X; Y)
&= E\left[\log p_{X,Y}(x,y)\right] - E\left[\log p_X(x)\right] - E\left[\log p_Y(y)\right] \\
&= -H(X,Y) + H(X) + H(Y).
\end{aligned}
$$

Substituting the chain rule $H(X,Y) = H(Y) + H(X \mid Y)$ gives
$I(X;Y) = H(X) - H(X \mid Y)$, and the symmetric substitution gives
$H(Y) - H(Y \mid X)$. Symmetry is immediate from
:eqref:`eq_mdl-mut_ent_def`, whose density ratio is unchanged when $X$ and $Y$
are swapped. Nonnegativity is Gibbs' inequality
(:numref:`sec_mdl-information_theory`) applied to the two distributions
$P_{X,Y}$ and $P_X \otimes P_Y$: the divergence is non-negative, and zero
exactly when they coincide, i.e., when
$p_{X,Y}(x,y) = p_X(x)\,p_Y(y)$ everywhere, which is the definition of
independence. $\blacksquare$

The identities have a picture, :numref:`fig_mdl-mi-overlap`: draw $H(X)$ and
$H(Y)$ as overlapping disks. The overlap is $I(X;Y)$, the crescents are the
conditional entropies, the union is the joint entropy, and each identity in
the properties proposition is one way of reading off the overlap's area.
The picture also
explains the directed reading: $I(X;Y) = H(X) - H(X \mid Y)$ is the
*reduction in surprise* about $X$ obtained by observing $Y$.

![Mutual information $I(X;Y)$ as the overlap of the entropies $H(X)$ and $H(Y)$: the crescents are the conditional entropies $H(X\mid Y)$ and $H(Y\mid X)$, the union is the joint entropy $H(X,Y)$, and the three equivalent expressions for $I(X;Y)$ follow by reading off the areas.](../img/mdl-it-mi-overlap.svg)
:label:`fig_mdl-mi-overlap`

Back to the worked example. With marginals computed from the joint, the
divergence form :eqref:`eq_mdl-mut_ent_def` and the entropy identity must
return the same number, and they do: $I(X;Y) \approx 0.0242$ nats
($\approx 0.035$ bits after dividing by $\ln 2$). The pair is *barely*
dependent: the joint $\begin{smallmatrix}0.1 & 0.4\\0.2 & 0.3\end{smallmatrix}$
is close to the product of its marginals
$\begin{smallmatrix}0.15 & 0.35\\0.15 & 0.35\end{smallmatrix}$, and mutual
information quantifies the gap.

```{.python .input #mutual-information-discrete-joint}
def mutual_information(p_xy):
    """I(X;Y) of a discrete joint p.m.f., in nats."""
    p_x = p_xy.sum(axis=1, keepdims=True)   # marginals from the joint
    p_y = p_xy.sum(axis=0, keepdims=True)
    return float((p_xy * onp.log(p_xy / (p_x * p_y))).sum())

mi_kl = mutual_information(p_xy)            # divergence form
mi_ent = H_x + H_y - H_xy                    # entropy identity
print(f'I(X;Y) = {mi_kl:.4f} nats (KL form) '
      f'= {mi_ent:.4f} nats (entropy identity)')
print(f'       = {mi_kl / onp.log(2):.4f} bits')
```

### A Gaussian Example
:label:`subsec_mdl-mi-gaussian`

For continuous variables, the bivariate Gaussian provides a useful worked
example with a closed form. We use this expression as a known reference value
for the estimators below.

**Proposition (Gaussian mutual information).** *If $(X, Y)$ is bivariate
Gaussian with correlation coefficient $\rho$, then*

$$I(X; Y) = -\tfrac{1}{2} \log\left(1 - \rho^2\right).$$
:eqlabel:`eq_mdl-gaussian_mi`

**Proof.** Mutual information is invariant under separate invertible maps of
$X$ and of $Y$ (proved in the next subsection), so we may standardize both
variables and take

$$
\Sigma = \begin{pmatrix} 1 & \rho \\ \rho & 1 \end{pmatrix},
\qquad \det \Sigma = 1 - \rho^2.
$$

For a standard Gaussian the differential entropy is
$h(X) = h(Y) = \tfrac{1}{2}\log(2\pi e)$, a value computed explicitly inside
the maximum-entropy proof of :numref:`subsec_mdl-gaussian-max-entropy`; the
same integration against the bivariate density gives
$h(X, Y) = \tfrac{1}{2}\log\left((2\pi e)^2 \det\Sigma\right)$. The entropy
identity of the properties proposition carries over: joint and conditional
differential
entropies are the same formulas with integrals in place of sums, and the
chain-rule algebra never used discreteness. It gives

$$
I(X; Y) = h(X) + h(Y) - h(X, Y)
= \log(2\pi e) - \tfrac{1}{2}\log\left((2\pi e)^2 (1 - \rho^2)\right)
= -\tfrac{1}{2}\log(1 - \rho^2).
$$

Each differential entropy individually depends on the coordinates (recall the
caveat of :numref:`sec_mdl-information_theory`: differential entropy can even
be negative), but the offending terms cancel in the combination: mutual
information is a *relative* quantity. $\blacksquare$

Read the limits: at $\rho = 0$ the variables are independent (for jointly
Gaussian variables, uncorrelated implies independent) and $I = 0$; as
$\rho \to \pm 1$ the joint
density collapses onto a line, $Y$ determines $X$, and $I \to \infty$. In
between, the formula gives concrete targets: $\rho = 0.5$ carries
$\approx 0.144$ nats, $\rho = 0.9$ carries $\approx 0.830$ nats, and
$\rho = 0.99$ carries $\approx 1.96$ nats.

The following example estimates :eqref:`eq_mdl-gaussian_mi` by binning the
samples into a 2-D histogram and applying the
discrete formula to the bin frequencies.

```{.python .input #mutual-information-gaussian-binned}
def gauss_pair(rng, rho, n):
    """n samples of a standard bivariate Gaussian with correlation rho."""
    x = rng.standard_normal(n)
    y = rho * x + onp.sqrt(1 - rho**2) * rng.standard_normal(n)
    return x, y

def binned_mi(x, y, bins=40):
    """Plug-in MI estimate from a 2-D histogram, in nats."""
    counts, _, _ = onp.histogram2d(x, y, bins=bins)
    p = counts / counts.sum()
    p_x, p_y = p.sum(axis=1, keepdims=True), p.sum(axis=0, keepdims=True)
    mask = p > 0
    return float((p[mask] * onp.log(p[mask] / (p_x * p_y)[mask])).sum())

rng = onp.random.default_rng(42)
rhos = onp.array([0.0, 0.3, 0.6, 0.9, 0.99])
est = [binned_mi(*gauss_pair(rng, r, 200_000)) for r in rhos]
true = [-0.5 * onp.log(1 - r**2) for r in rhos]
for r, e, t in zip(rhos, est, true):
    print(f'rho = {r:4.2f}: binned = {e:.4f}, closed form = {t:.4f} nats')
d2l.plot(rhos, [onp.array(est), onp.array(true)], 'correlation rho',
         'I(X;Y) (nats)', legend=['binned estimate', 'closed form'],
         figsize=(5, 3))
```

With $200{,}000$ samples and a $40 \times 40$ grid, the histogram tracks the
closed form well at moderate correlation, but look closely at
$\rho = 0.99$: the estimate comes in at roughly $1.78$ nats against a truth
of $1.96$, missing about $0.18$ nats because the probability mass hides in a
ridge thinner than the bins. Even in *two* dimensions with abundant data,
the plug-in histogram estimator is already biased once the dependence is
strong. The high-dimensional estimators developed below face still sharper limits.

### Pointwise Mutual Information
:label:`subsec_mdl-pmi`

Mutual information is an *average*. Its summand in
:eqref:`eq_mdl-mut_ent_def` is the *pointwise mutual information*

$$\textrm{pmi}(x, y) = \log\frac{p_{X, Y}(x, y)}{p_X(x)\, p_Y(y)}$$
:eqlabel:`eq_mdl-pmi_def`

compares how often the specific pair $(x, y)$ occurs to how often it *would*
occur if $X$ and $Y$ were independent. Positive pmi means the outcomes attract
(they co-occur more than chance predicts); negative pmi means they repel; and
$I(X;Y) = E_{(x,y) \sim P_{X,Y}}[\textrm{pmi}(x,y)]$ is the average
attraction.

Pointwise mutual information is widely used in natural language
processing :cite:`Church.Hanks.1990`, where it separates genuine
*collocations* from words that merely co-occur because both are common. Here
is the effect in miniature. Suppose we count consecutive word pairs (bigrams)
in a small corpus and tally how often each of the first words *new*,
*machine*, *the* is followed by each of *york*, *learning*, *day*:

| count | york | learning | day |
| :-- | --: | --: | --: |
| **new** | 38 | 3 | 9 |
| **machine** | 1 | 46 | 3 |
| **the** | 11 | 21 | 168 |

The raw counts mislead: "the day" is by far the most frequent pair ($168$
occurrences against $38$ for "new york"), because *the* and *day* are
both frequent words. Pointwise mutual information corrects for the marginals.

```{.python .input #mutual-information-pmi-corpus}
counts = onp.array([[38., 3., 9.],     # new york / new learning / new day
                    [1., 46., 3.],     # machine ...
                    [11., 21., 168.]]) # the ...
first, second = ['new', 'machine', 'the'], ['york', 'learning', 'day']

p = counts / counts.sum()
p_w, p_c = p.sum(axis=1, keepdims=True), p.sum(axis=0, keepdims=True)
pmi = onp.log(p / (p_w * p_c))
for (i, j) in [(0, 0), (1, 1), (2, 2), (2, 0)]:
    print(f'pmi({first[i]}, {second[j]}) = {pmi[i, j]:6.3f} nats '
          f'(count {counts[i, j]:.0f})')
print(f'I(first; second) = {float((p * pmi).sum()):.4f} nats')
```

After adjustment for the marginals, the collocations receive the higher scores:
"new york" scores $\textrm{pmi} \approx 1.52$ nats and "machine learning"
$\approx 1.37$, while the frequent-but-unremarkable "the day" manages only
$\approx 0.34$; the pair "the york" comes out *negative*
($\approx -1.11$ nats), occurring three times less often than independence
would predict. The table's overall mutual information is $\approx 0.41$ nats,
summarizing how strongly first words constrain second words on average. This
count-divide-and-log pattern (and its smoothed variants) underpins
classical word-association mining and survives inside modern embedding
methods, where factorizing a pmi matrix recovers word vectors
:cite:`Levy.Goldberg.2014`.

### Mutual Information as Nonlinear Correlation
:label:`sec_mdl-mi-nonlinear-correlation`

How does mutual information relate to the correlation coefficient of
:numref:`sec_mdl-random_variables`? Correlation detects *linear* dependence:
$\rho(X, Y) = 0$ does **not** imply independence, only the absence of a
linear trend. Mutual information detects *any* dependence: by the equality
condition of the
properties proposition, $I(X; Y) = 0$ holds *if and only if* $X \perp Y$.
Two further properties sharpen the contrast.

**Proposition (reparameterization invariance).** *If $g$ and $h$ are
invertible maps (in the continuous case, smooth with smooth inverses), then
$I(g(X); h(Y)) = I(X; Y)$.*

**Proof.** For discrete variables, invertibility means $g$ and $h$ merely
relabel outcomes: the joint p.m.f. of $(g(X), h(Y))$ takes the same values as
that of $(X, Y)$ at different labels, and the sum in
:eqref:`eq_mdl-mut_ent_def` is unchanged. For continuous variables, write
mutual information as the KL divergence
$D_{\textrm{KL}}(P_{X,Y} \| P_X \otimes P_Y)$ and apply the smooth change of
variables $(x, y) \mapsto (g(x), h(y))$: *both* densities in the ratio pick
up the same Jacobian factor $|g'(x)|^{-1} |h'(y)|^{-1}$, which cancels.
(Contrast differential entropy, where the uncancelled Jacobian is exactly why
it is coordinate-dependent; see :numref:`sec_mdl-information_theory`.)
$\blacksquare$

Correlation enjoys no such invariance: it is preserved only by *affine* maps
(up to sign). Mutual information therefore captures a broader notion of
dependence: it asks whether two variables are informative about each other
in *any* coordinates, not only whether they align linearly. A standard
counterexample makes both points at once: let $X \sim \mathcal{N}(0, 1)$ and
$Y = X^2 + 0.3\,\epsilon$ with independent noise
$\epsilon \sim \mathcal{N}(0,1)$. Then $\textrm{Cov}(X, Y) = E[X^3] = 0$ by
symmetry, so the correlation vanishes, yet $Y$ is nearly a function of $X$,
and the dependence is strong.

```{.python .input #mutual-information-nonlinear-correlation}
rng = onp.random.default_rng(0)
n = 200_000
x = rng.standard_normal(n)
y = x**2 + 0.3 * rng.standard_normal(n)    # zero correlation by symmetry

print(f'sample correlation rho = {onp.corrcoef(x, y)[0, 1]:+.4f}')
print(f'binned MI estimate     = {binned_mi(x, y):.4f} nats')
print(f'after x -> 2x + 3      = {binned_mi(2 * x + 3, y):.4f} nats')
```

The correlation is numerically zero while the mutual information estimate is
about one nat, *more* shared information than the $\rho = 0.9$ Gaussian pair
above ($0.830$ nats). The third line checks invariance: an affine
reparameterization of $X$ leaves the estimate exactly unchanged (the
histogram bins stretch with the data, so even the bin counts are identical).
One caution for later: this invariance is also why mutual information is hard
to *estimate*, since an estimator must implicitly cope with every
reparameterization of the data at once; :numref:`sec_mdl-mi-hard` describes
the resulting estimation cost.

### Conditional Mutual Information and the Chain Rule

Like entropy, mutual information has a conditional version. The *conditional mutual
information*

$$I(X; Y \mid Z) = H(X \mid Z) - H(X \mid Y, Z)$$
:eqlabel:`eq_mdl-cond_mi_def`

is the information $X$ and $Y$ share *once $Z$ is already known*.
Equivalently, $I(X; Y \mid Z) = E_{z}\left[D_{\textrm{KL}}(P_{X,Y \mid z}
\,\|\, P_{X \mid z} \otimes P_{Y \mid z})\right]$: expanding each
conditional entropy as an expectation and collecting the logarithms into the
ratio $p_{X,Y \mid z}/(p_{X \mid z}\, p_{Y \mid z})$ repeats the algebra of
the properties proposition under each conditional law. As an average of
per-$z$ divergences it
inherits nonnegativity from Gibbs, with $I(X; Y \mid Z) = 0$ if and only if
$X$ and $Y$ are conditionally independent given $Z$. The chain rule then
splits joint information into sequential contributions.

**Proposition (chain rule for mutual information).** *For any $X, Y, Z$,*

$$
I(X; Y, Z) = I(X; Z) + I(X; Y \mid Z).
$$

**Proof.** Telescope with the entropy identities:

$$
\begin{aligned}
I(X; Y, Z) &= H(X) - H(X \mid Y, Z) \\
&= \underbrace{H(X) - H(X \mid Z)}_{I(X;\, Z)}
+ \underbrace{H(X \mid Z) - H(X \mid Y, Z)}_{I(X;\, Y \mid Z)}.
\quad\blacksquare
\end{aligned}
$$

**Corollary (independent side information is free).** *If $W$ is independent
of the pair $(X, Y)$, then $I(X; (Y, W)) = I(X; Y)$.*

**Proof.** By the chain rule, $I(X; Y, W) = I(X; W) + I(X; Y \mid W)$. The
first term is zero since $W \perp X$. For the second, independence of $W$
from the pair means conditioning on $W$ changes neither $H(X)$ nor
$H(X \mid Y)$, so $I(X; Y \mid W) = I(X; Y)$. $\blacksquare$

The corollary is exactly the accounting we will need for InfoNCE in
:numref:`sec_mdl-infonce`, where a batch of independent "negative" samples
is appended to $Y$ and must add no information about $X$. The chain rule's
main consequence deserves its own subsection.

### The Data-Processing Inequality
:label:`subsec_mdl-dpi`

Random variables $X, Y, Z$ form a *Markov chain*, written
$X \to Y \to Z$, when $Z$ depends on $(X, Y)$ only through $Y$:
$p(z \mid x, y) = p(z \mid y)$. This is the structure of *processing*: $Y$ is
computed from $X$ (a measurement, a feature map, a hidden layer), then $Z$ is
computed from $Y$ alone. Equivalently, $X$ and $Z$ are conditionally
independent given $Y$, i.e., $I(X; Z \mid Y) = 0$. The following theorem
:cite:`Cover.Thomas.1999` governs this structure.

**Proposition (data-processing inequality).** *If $X \to Y \to Z$ is a
Markov chain, then*

$$
I(X; Z) \leq I(X; Y),
$$

*with equality if and only if $I(X; Y \mid Z) = 0$ (i.e., $X \to Z \to Y$ is
also a Markov chain).*

**Proof.** Expand $I(X; Y, Z)$ by the chain rule in both orders:

$$
I(X; Z) + I(X; Y \mid Z) = I(X; Y, Z) = I(X; Y) + I(X; Z \mid Y).
$$

The Markov property makes the last term zero, $I(X; Z \mid Y) = 0$, leaving
$I(X; Z) = I(X; Y) - I(X; Y \mid Z) \leq I(X; Y)$, since conditional mutual
information is non-negative. Equality holds exactly when
$I(X; Y \mid Z) = 0$. $\blacksquare$

No deterministic or random processing of $Y$ can *increase* the information
it carries about $X$. Three consequences follow:

* **Representations only lose.** A network computes its representation
  layer-by-layer, $X \to Z_1 \to Z_2 \to \cdots$, so
  $I(X; Z_1) \geq I(X; Z_2) \geq \cdots$ and likewise
  $I(Y; Z_1) \geq I(Y; Z_2) \geq \cdots$ for a label $Y$ at the *input* end
  of the chain (since $Y \to X \to Z_\ell$ is Markov). Whatever label
  information the input lacks, no architecture can create; whatever a layer
  discards, no later layer recovers. This is the theorem that makes the
  information bottleneck of :numref:`sec_mdl-information-bottleneck` a
  well-defined objective.
* **Equality is achievable.** If $Z = g(Y)$ for an *invertible* $g$, then the
  chain runs both ways and $I(X; Z) = I(X; Y)$, consistent with the
  reparameterization invariance proved earlier. A many-to-one or noisy map may lose information, but it need not: a
  sufficient statistic can be noninvertible or noisy while preserving all the
  information about $X$. Equality in data processing is characterized by such
  sufficiency, not by invertibility alone.
* **A sufficient statistic is a lossless compression.** When equality holds
  with $Z$ much "smaller" than $Y$, the map has discarded only what was
  irrelevant to $X$: this is the information-theoretic reading of a
  sufficient statistic (:numref:`sec_mdl-exponential-family`).

One caution about scope. The data-processing inequality is a statement about
Markov chains, not about conditioning in general. Conditioning can *create*
dependence, and $I(X; Y \mid Z) > I(X; Y)$ is perfectly possible: in the
extreme case of two independent fair bits with $Z$ their XOR, conditioning
turns zero dependence into a full $\ln 2$ nats (Exercise :exref:`ex_mdl-mutual-information-synergy-conditioning-can-create` works this out).
"Processing destroys information" is a theorem about the chain
$X \to Y \to Z$ alone; outside that structure, conditioning can raise or
lower dependence, as the XOR example shows.

The following example verifies the bookkeeping on a simple nontrivial chain:
a fair bit $X$, flipped with probability $0.1$ to give $Y$, flipped again
with probability $0.2$ to give $Z$.

```{.python .input #mutual-information-dpi-check}
p_x = onp.array([0.5, 0.5])
K1 = onp.array([[0.9, 0.1], [0.1, 0.9]])   # p(y|x): flip w.p. 0.1
K2 = onp.array([[0.8, 0.2], [0.2, 0.8]])   # p(z|y): flip w.p. 0.2
p_xyz = p_x[:, None, None] * K1[:, :, None] * K2[None, :, :]

def cond_mi(p_abc):
    """I(A;B|C) for a joint array indexed [a, b, c], in nats."""
    p_ac, p_bc, p_c = p_abc.sum(1), p_abc.sum(0), p_abc.sum((0, 1))
    ratio = p_abc * p_c[None, None, :] / (p_ac[:, None, :] * p_bc[None, :, :])
    return float((p_abc * onp.log(ratio)).sum())

I_xy = mutual_information(p_xyz.sum(axis=2))           # I(X;Y)
I_xz = mutual_information(p_xyz.sum(axis=1))           # I(X;Z)
I_xz_y = cond_mi(onp.transpose(p_xyz, (0, 2, 1)))      # I(X;Z|Y)
I_xy_z = cond_mi(p_xyz)                                # I(X;Y|Z)
print(f'I(X;Y) = {I_xy:.4f},  I(X;Z) = {I_xz:.4f} nats  (DPI: smaller)')
print(f'I(X;Z|Y) = {I_xz_y:.4f}  (Markov: zero)')
print(f'chain rule, both orders: {I_xz + I_xy_z:.4f} = {I_xy + I_xz_y:.4f}')
```

Every claim checks out: $I(X; Z \mid Y) = 0$ certifies the Markov structure,
both expansions of $I(X; Y, Z)$ agree at $0.3681$ nats, and processing
through the second noisy channel costs two thirds of the information:
$I(X;Y) \approx 0.368$ nats falls to $I(X;Z) \approx 0.120$ nats. (For
calibration, a noiseless bit would carry $\ln 2 \approx 0.693$ nats.)

### What Mutual Information Guarantees: Fano's Inequality

One guarantee explains why anyone wants high mutual information in the
first place: enough shared information is a *prerequisite* for accurate
prediction.

**Proposition (Fano's inequality).** *Let $X$ take values in a set of size
$k$, and let $\hat{X} = g(Y)$ be any estimate of $X$ computed from $Y$, with
error probability $P_e = P(\hat{X} \neq X)$. Then*

$$H_b(P_e) + P_e \log(k - 1) \geq H(X \mid Y) = H(X) - I(X; Y),$$
:eqlabel:`eq_mdl-fano`

*where $H_b$ is the binary entropy in nats* :cite:`Cover.Thomas.1999`.

**Proof.** Let $E = \mathbf{1}_{\hat{X} \neq X}$ be the error indicator and
expand $H(E, X \mid \hat{X})$ by the chain rule in both orders. One order:
$H(E, X \mid \hat{X}) = H(X \mid \hat{X})$, since $E$ is a function of
$(X, \hat{X})$. The other:
$H(E, X \mid \hat{X}) = H(E \mid \hat{X}) + H(X \mid E, \hat{X})
\leq H_b(P_e) + P_e \log(k-1)$, because conditioning cannot raise entropy
($H(E \mid \hat{X}) \leq H(E) = H_b(P_e)$, since $I(E; \hat{X}) \geq 0$),
and given $E$: when $E = 0$, $X = \hat{X}$ is known (zero entropy), and
when $E = 1$ (probability $P_e$), $X$ ranges over at most $k - 1$ values.
Finally $X \to Y \to \hat{X}$ is a Markov chain, so the data-processing
inequality gives $I(X; \hat{X}) \leq I(X; Y)$, i.e.,
$H(X \mid \hat{X}) \geq H(X \mid Y)$. Chaining the three displays proves the
claim. $\blacksquare$

Fano converts information into a floor on achievable error: no
representation, classifier, or amount of compute can beat it. It quantifies
the "prerequisite" reading of contrastive learning: features that support
accurate downstream classification *must* carry high mutual information with
the label. Combining this requirement with the estimation ceiling gives:

```{.python .input #mutual-information-fano}
k, p_e = 1000, 0.05
H_b = lambda p: -p * onp.log(p) - (1 - p) * onp.log(1 - p)
need = onp.log(k) - H_b(p_e) - p_e * onp.log(k - 1)
print(f'{p_e:.0%} error on {k} balanced classes needs '
      f'I(X;Y) >= {need:.3f} nats')
print(f'InfoNCE can reach that value only if N >= e^I = {onp.exp(need):.0f}')
```

Five percent error on a thousand balanced classes requires at least
$6.36$ nats of label information. Since InfoNCE cannot exceed $\log N$, it can
reach that numerical value only when $N\geq e^{6.36}\approx581$. This is a
necessary range condition for InfoNCE, not a universal sample-size threshold
for every estimator or confidence procedure.

## Communication and Compression

### Lossy Compression and Rate--Distortion
:label:`subsec_mdl-rate-distortion`

Lossless recovery is often unnecessary. An image codec may change pixels
slightly, and a learned representation may deliberately discard nuisance
detail. Choose a distortion measure $d(x,\hat x)$ and allow average distortion
at most $D$. The **rate--distortion function** is

$$
R(D)=\inf_{p(\hat x\mid x):\;E[d(X,\hat X)]\le D} I(X;\hat X).
$$
:eqlabel:`eq_mdl-rate-distortion`

It is the smallest information rate a representation can retain while meeting
the distortion budget. A stricter budget costs more bits; relaxing $D$ lowers
the rate. For a Gaussian source $X\sim\mathcal N(0,\sigma^2)$ under squared
error,

$$
R(D)=\frac12\log\frac{\sigma^2}{D}
\quad (0<D<\sigma^2),
\qquad R(D)=0\quad(D\ge\sigma^2),
$$
:eqlabel:`eq_mdl-gaussian-rd`

in nats per sample. The left panel of :numref:`fig_mdl-it-capacity-rd` plots
it: bits buy reconstruction quality at a rate that explodes as $D\to0$ and
drops to zero once the allowed distortion exceeds the source variance —
at that point transmitting nothing and reconstructing the mean is already
good enough. This curve is a useful benchmark for learned compression:
reconstruction losses measure distortion, while quantized latents and entropy
models determine rate. Objectives of the form
$D+\beta R$ select one supporting point of the trade-off — geometrically, the
point where the curve has slope $-1/\beta$. Autoencoders and
variational bottlenecks use approximations to these two terms; they do not evade
the trade-off by learning the codec.

### Noisy Channels and Capacity
:label:`subsec_mdl-channel-capacity`

Compression asks how few bits describe a source; communication asks how many
bits survive a noisy channel. A memoryless channel is a conditional
distribution $p(y\mid x)$, whose **capacity** is

$$
C=\max_{p(x)} I(X;Y),
$$
:eqlabel:`eq_mdl-channel-capacity`

where the maximization chooses the input distribution best matched to the
channel. Shannon's noisy-channel coding theorem says that long block codes can
make decoding error arbitrarily small at every rate below $C$, while rates
above $C$ cannot be made reliable :cite:`Shannon.1948,Cover.Thomas.1999`.
For a binary symmetric channel that flips each bit independently with
probability $\varepsilon$, the uniform input is optimal and

$$
C=1-h_2(\varepsilon)\quad\text{bits per channel use},
$$

where $h_2(\varepsilon) = -\varepsilon \log_2 \varepsilon -
(1 - \varepsilon) \log_2 (1 - \varepsilon)$ is the binary entropy in bits.
The right panel of
:numref:`fig_mdl-it-capacity-rd` plots this curve, and it rewards a moment's
reading. At $\varepsilon=0$ one bit survives per use. At $\varepsilon=1/2$
the output is independent of the input and capacity is zero — no code, however
clever, gets anything through. And at $\varepsilon=1$ capacity is back to one
bit: a channel that *always* lies is as good as one that always tells the
truth, because the receiver can flip every bit. What destroys
information is not corruption but *unpredictability*.

![Shannon's two operational curves. Left: the Gaussian rate–distortion function — the rate, in nats per sample, needed to reconstruct within squared error $D$; fidelity gets expensive fast, and beyond the source variance nothing needs to be sent at all. Right: the capacity of a binary symmetric channel with flip probability $\varepsilon$; a coin-flip channel carries nothing, while a channel that always flips is again perfect.](../img/mdl-it-capacity-rd.svg)
:label:`fig_mdl-it-capacity-rd`

The same mutual-information lens applies when a representation passes through
noise, quantization, dropout, or a bandwidth bottleneck. Data processing limits
what later layers can recover, while coding or redundancy determines how close
a designed system comes to the limit.

## The Information Bottleneck
:label:`sec_mdl-information-bottleneck`

We close with the principle that joins this section's two themes (what
representations preserve and what we can measure) into a single objective,
and with an assessment of where its empirical support stands.

### The Information-Bottleneck Lagrangian

What makes a representation $Z$ of an input $X$ *good* for predicting a
label $Y$? The *information bottleneck* (IB) principle
:cite:`Tishby.Pereira.Bialek.1999` answers with two mutual informations: a
good $Z$ keeps little of $X$ (it *compresses*) while keeping much of $Y$ (it
*predicts*). Over stochastic encoders $p(z \mid x)$, minimize the Lagrangian

$$\min_{p(z \mid x)}\; I(X; Z) - \beta\, I(Y; Z).$$
:eqlabel:`eq_mdl-ib_lagrangian`

The data-processing inequality is what makes this well-posed: since
$Y \to X \to Z$ is a Markov chain (the encoder sees only $X$), we always
have $I(Y; Z) \leq \min\{I(X; Y),\, I(X; Z)\}$: the representation can
never know more about the label than the input does, and predicting well
forces $Z$ to retain input information. The multiplier $\beta$ sets the tradeoff.
In finite/discrete settings, or in stochastic encoder families where optima
exist, the $\beta\to0$ limit favors a constant representation. Large $\beta$
prioritizes predictive information; recovering a minimal sufficient statistic
additionally requires that one lie in the encoder family and that the relevant
optimum be attained. For deterministic continuous encoders, $I(X;Z)$ can be
infinite, which is one reason the stochastic formulation matters. Sweeping
$\beta$ traces an attainable frontier in the *information plane* with coordinates
$(I(X;Z),I(Y;Z))$ for the chosen family. Rate--distortion theory
:cite:`Shannon.1959` is the special case where
"relevance" is a hand-chosen distortion measure; IB lets the label define
relevance instead. The *deep variational information bottleneck* (VIB)
:cite:`Alemi.Fischer.Dillon.ea.2017` makes :eqref:`eq_mdl-ib_lagrangian`
trainable by applying the bounds derived above to both terms: a
Barber--Agakov-style decoder bound for $I(Y;Z)$ from below and a variational
upper bound for $I(X;Z)$. That is the same replace-the-intractable-posterior
pattern as the ELBO of :numref:`sec_mdl-latent-em-elbo`, and another reminder
that in deep learning, mutual informations are optimized through bounds, not
computed. The frontier this
tradeoff traces is shown in :numref:`fig_mdl-ib-tradeoff`.

![The information plane of the Gaussian bottleneck with $\rho = 0.9$: the concave frontier traces the best attainable $I(Y;Z)$ at each compression level $I(X;Z)$, the shaded region above it is infeasible, and the dashed ceiling is the data-processing limit $I(X;Y) = -\tfrac{1}{2}\log(1-\rho^2) \approx 0.830$ nats. Dots mark the operating points the multiplier $\beta$ selects; for $\beta \leq 1/\rho^2$ the optimum collapses to the origin.](../img/mdl-it-ib-tradeoff.svg)
:label:`fig_mdl-ib-tradeoff`

### The Information Plane: a Gaussian Bottleneck in Closed Form
:label:`subsec_mdl-gaussian-bottleneck`

Rather than train a VIB (two stacked variational bounds would put us two
bounds away from anything checkable), we exhibit the IB tradeoff in a model
where every quantity is exact :cite:`Chechik.Globerson.Tishby.ea.2005`. Let
$(X, Y)$ be our standard Gaussian pair with $\rho = 0.9$, so
$I(X; Y) \approx 0.830$ nats, and let the encoder be a noisy channel

$$
Z = X + \sigma\, \epsilon, \qquad \epsilon \sim \mathcal{N}(0, 1),
$$

with the noise scale $\sigma$ as the compression knob. Both informations
follow from the Gaussian anchor :eqref:`eq_mdl-gaussian_mi`: the correlations
are $\rho_{XZ}^2 = 1/(1 + \sigma^2)$ and
$\rho_{YZ}^2 = \rho^2/(1 + \sigma^2)$, so

$$
I(X; Z) = \tfrac{1}{2}\log\left(1 + \sigma^{-2}\right), \qquad
I(Y; Z) = -\tfrac{1}{2}\log\left(1 - \frac{\rho^2}{1 + \sigma^2}\right).
$$
:eqlabel:`eq_mdl-gaussian-ib-info`

In this scalar Gaussian family the noise level is the *only* degree of
freedom, so sweeping $\sigma$ traces the family's entire curve; the Gaussian
information bottleneck theorem of :cite:`Chechik.Globerson.Tishby.ea.2005`
adds that this curve is the *optimal* frontier over all encoders
$p(z \mid x)$. Minimizing the Lagrangian
:eqref:`eq_mdl-ib_lagrangian` over $\sigma$ for
each $\beta$ picks out one operating point on the frontier.

```{.python .input #mutual-information-ib-plane}
rho = 0.9
sigma = onp.logspace(-2, 2, 401)            # noise scale: the beta knob
I_xz = 0.5 * onp.log(1 + sigma**-2)         # compression cost
I_yz = -0.5 * onp.log(1 - rho**2 / (1 + sigma**2))  # predictive value
I_xy = -0.5 * onp.log(1 - rho**2)

d2l.plot(I_xz, [I_yz, onp.full_like(I_xz, I_xy)], 'I(X;Z) (nats)',
         'I(Y;Z) (nats)', legend=['IB frontier', 'I(X;Y) ceiling'],
         xlim=[0, 3], figsize=(5, 3))
for beta in [1.0, 1.5, 2.0, 4.0, 16.0]:
    i = onp.argmin(I_xz - beta * I_yz)      # IB Lagrangian on the grid
    d2l.plt.plot(I_xz[i], I_yz[i], 'ro')
    d2l.plt.annotate(f'{beta:g}', (I_xz[i], I_yz[i]),
                     textcoords='offset points', xytext=(6, -2))
    print(f'beta = {beta:4.1f}: I(X;Z) = {I_xz[i]:.3f}, '
          f'I(Y;Z) = {I_yz[i]:.3f} nats')
```

The frontier rises steeply from the origin (the first fraction of a nat of
retained input information increases label information at the best exchange rate,
$\rho^2$ nats per nat), then flattens as $I(Y;Z)$ approaches its DPI ceiling
$I(X;Y)$: retaining more of $X$ yields progressively less additional
information about $Y$. The marked
points show the $\beta$ knob at work, including the degenerate regime: for
$\beta \leq 1/\rho^2 \approx 1.23$ the exchange rate never beats the price
and the optimum is the origin, $I(X;Z) = I(Y;Z) = 0$; beyond it, rising
$\beta$ moves the operating point up the frontier. A deep VIB approximates
this construction on real data using encoder noise instead of $\sigma$ and
variational bounds instead of closed forms.
Its $\beta$-sweep traces a noisy version of the same curve.

## Summary

* Mutual information $I(X;Y) = D_{\textrm{KL}}(P_{X,Y} \| P_X \otimes P_Y)$
  measures how far a pair of variables is from independence. It is
  symmetric, non-negative, zero exactly at independence, equals
  $H(X) - H(X \mid Y) = H(X) + H(Y) - H(X,Y)$, and is invariant under
  invertible reparameterization of either variable; unlike correlation, it
  detects *any* dependence.

* The Gaussian pair gives the closed-form anchor
  $I = -\tfrac{1}{2}\log(1 - \rho^2)$; pointwise mutual information scores
  individual co-occurrences and powers collocation mining in NLP.

* The chain rule $I(X; Y, Z) = I(X;Z) + I(X; Y \mid Z)$ yields the
  data-processing inequality: along a Markov chain $X \to Y \to Z$,
  $I(X;Z) \leq I(X;Y)$. Processing never creates information;
  representations only lose it.

* **Rate--distortion** gives the smallest retained information compatible with
  a distortion budget, while **channel capacity** $C=\max_{p(x)}I(X;Y)$ gives
  the largest reliable communication rate through noise.

* The information bottleneck $\min I(X;Z) - \beta\, I(Y;Z)$ frames good
  representation as a compression--prediction tradeoff, made trainable by
  the same variational bounds (VIB); the claim that ordinary training
  implicitly compresses is contested (Saxe et al.). Fano's inequality is
  the guarantee that survives: low prediction error *requires* high mutual
  information.

## Exercises

1. **Gaussian pointwise mutual information.** Let $(X, Y)$ be the standard
   bivariate Gaussian pair with correlation $\rho$ that `gauss_pair`
   samples.
    1. Derive the expression for $\textrm{pmi}(x, y)$ that the function
       `pmi_gauss` of :numref:`sec_mdl-mi-hard` evaluates.
    1. Average it over $P_{X,Y}$ and recover :eqref:`eq_mdl-gaussian_mi`
       without the entropy identities.
    1. Average it over $P_X \otimes P_Y$ instead. Show that the result is
       $-D_{\textrm{KL}}(P_X \otimes P_Y \,\|\, P_{X,Y})$, evaluate it in
       closed form, and compute it for $\rho = 0.99$.
    1. Show that $E_{P_X \otimes P_Y}\big[e^{\textrm{pmi}(x, y)}\big] = 1$
       for every $\rho$, although by (c) a typical shuffled pair has a tiny
       $e^{\textrm{pmi}}$ at $\rho = 0.99$. Which shuffled pairs carry this
       expectation? Explain why the average of $e^{\textrm{pmi}}$ over $128$
       shuffled pairs, which the DV and NWJ estimates of
       :numref:`sec_mdl-infonce` compute, varies strongly from batch to
       batch.
   :exlabel:`ex_mdl-mutual-information-gaussian-pmi`

1. [code] **Rare pairs and PMI.** Pointwise mutual information
   :eqref:`eq_mdl-pmi_def` corrects co-occurrence counts for the frequencies
   of the two words (:numref:`subsec_mdl-pmi`).
    1. Show that
       $\textrm{pmi}(x, y) \leq \min\{-\log p_X(x), -\log p_Y(y)\}$. When is
       each of the two bounds attained?
    1. Extend `counts` by a fourth first word *sri* and a fourth second word
       *lanka* that occur only in the bigram "sri lanka", with count $2$.
       Recompute the pmi of "new york", "machine learning", "the day", and
       "sri lanka", and repeat with count $1$. Zero counts give
       $\textrm{pmi} = -\infty$ and make `(p * pmi).sum()` return `nan`, so
       restrict sums to the nonzero cells, as `binned_mi` does.
    1. Compare the contributions $p_{X,Y}(x, y)\,\textrm{pmi}(x, y)$ of "new
       york" and "sri lanka" to $I(X;Y)$. Then add $1$ to every count and
       recompute both pmi values. Which of these scores would you use to rank
       collocations, and why?
   :exlabel:`ex_mdl-mutual-information-rare-pairs`

1. **Information versus usable information.** A label $Y$ is a fair bit.
   Given $Y = 0$ the input $X \in \mathbb{R}^2$ is uniform on the circle of
   radius $1$, and given $Y = 1$ it is uniform on the circle of radius $2$.
   A representation computes $Z = \|X\|^2$.
    1. Show that $I(Y;Z) = I(Y;X) = \ln 2$, although $x \mapsto \|x\|^2$ is
       not invertible. Which condition for equality in the data-processing
       inequality of :numref:`subsec_mdl-dpi` holds?
    1. Show that no linear classifier
       $\hat y = \mathbf{1}[\mathbf{w}^\top \mathbf{x} + b > 0]$ classifies
       every input correctly, whereas a threshold on $Z$ does.
    1. The layers of a network form a Markov chain
       $X \to Z_1 \to Z_2 \to \cdots$. Suppose that the accuracy of a linear
       classifier fit to $Z_\ell$ rises with $\ell$. Why does this not
       contradict $I(Y;Z_1) \geq I(Y;Z_2) \geq \cdots$?
    1. :citet:`Xu.Zhao.Song.ea.2020` measure the information that a family
       $\mathcal{V}$ of predictors $q(y \mid x)$ can extract. Let
       $H_{\mathcal{V}}(Y \mid X) = \inf_{q \in \mathcal{V}} E[-\log q(Y \mid X)]$,
       and let $H_{\mathcal{V}}(Y)$ be the same infimum over the predictors
       in $\mathcal{V}$ that ignore their input. The *usable information* is

       $$I_{\mathcal{V}}(X \to Y) = H_{\mathcal{V}}(Y) - H_{\mathcal{V}}(Y \mid X).$$

       Let $\mathcal{V}$ be logistic regression on the given input. Show that
       $I_{\mathcal{V}}(X \to Y) = 0$ and $I_{\mathcal{V}}(Z \to Y) = \ln 2$.
       For $X$, use that the expected logistic loss is convex in
       $(\mathbf{w}, b)$ and unchanged when $\mathbf{w}$ is rotated. Which
       inequality of :numref:`subsec_mdl-dpi` does $I_{\mathcal{V}}$ violate?
   :exlabel:`ex_mdl-mutual-information-usable-information`

1. **Synergy and the Venn diagram.** Let $X$ and $Y$ be independent fair
   bits and $Z = X \oplus Y$ their exclusive or.
    1. Show that $I(X;Y) = 0$ but $I(X;Y \mid Z) = \ln 2$. Which of the
       Markov chains $X \to Z \to Y$, $X \to Y \to Z$, and $Y \to X \to Z$
       hold? What does the data-processing inequality therefore say about
       this triple?
    1. In a three-variable version of :numref:`fig_mdl-mi-overlap`, the
       area common to all three disks would be the *interaction information*
       $I(X;Y;Z) = I(X;Y) - I(X;Y \mid Z)$. Prove that
       $I(X;Y;Z) = H(X) + H(Y) + H(Z) - H(X,Y) - H(X,Z) - H(Y,Z) + H(X,Y,Z)$,
       and conclude that $I(X;Y;Z)$ is symmetric in its three arguments.
    1. Evaluate $I(X;Y;Z)$ for the triple of (a) and for three copies
       $X = Y = Z$ of one fair bit. What do the two values imply for a
       three-disk diagram of entropies?

   *Adapted from Thomas M. Cover and Joy A. Thomas,
   [Elements of Information Theory, 2nd ed.](https://www.wiley.com/en-us/Elements+of+Information+Theory%2C+2nd+Edition-p-9780471241959),
   Exercise 2.25, and David J. C. MacKay,
   [Information Theory, Inference, and Learning Algorithms](https://www.inference.org.uk/itprnn/book.pdf),
   Exercise 8.8.*
   :exlabel:`ex_mdl-mutual-information-synergy-conditioning-can-create`

1. **Fano and a converse.** ● Fano's inequality :eqref:`eq_mdl-fano` bounds
   from below the error of any estimate $\hat{X} = g(Y)$ of a label $X$ with
   $k$ values.
    1. Let $\hat{X}(y) = \arg\max_x p(x \mid y)$. Show that its error
       satisfies $P_e \leq 1 - e^{-H(X \mid Y)}$. Bound
       $\max_x p(x \mid y)$ from below by $\sum_x p(x \mid y)^2$ and apply
       Jensen's inequality twice.
    1. An observation $Y$ carries $I(X;Y) = 9$ nats about a balanced
       $10{,}000$-class label $X$. Use :eqref:`eq_mdl-fano` and (a) to
       bracket the error of the best classifier. The inequality
       :eqref:`eq_mdl-fano` has no closed-form solution for $P_e$, so solve
       it numerically. What does the width of this interval say about
       predicting a downstream error rate from a mutual information?
   :exlabel:`ex_mdl-mutual-information-fano-converse`

1. [code] **Scalar quantization and rate--distortion.** ● A uniform scalar
   quantizer with step $\Delta$ replaces $X$ by the nearest multiple of
   $\Delta$, the center of its bin, and an entropy coder then spends about
   $H(X_\Delta)$ nats per sample on the index $X_\Delta$ of the bin. The
   differential entropy :eqref:`eq_mdl-diff-ent-def` depends on the units in
   which $x$ is measured, and the quantizer shows how.
    1. Let $X$ have a continuous density $p$. Show that
       $H(X_\Delta) + \log \Delta \to h(X)$ as $\Delta \to 0$, so that the
       discrete entropy grows without bound. Check the limit numerically for
       $X \sim \mathcal{N}(0, 1)$ with $\Delta = 0.1$ and $\Delta = 0.01$.
    1. Let $X \sim \mathcal{N}(0, \sigma^2)$. For small $\Delta$ the mean
       squared error is approximately $\Delta^2/12$. Combine this
       approximation with (a) to show that the rate exceeds the Gaussian
       rate--distortion function :eqref:`eq_mdl-gaussian-rd` at the same
       distortion by $\tfrac{1}{2}\log(2\pi e/12)$ nats ($\approx 0.255$
       bits) per sample, independently of $\sigma$ and $\Delta$.
    1. Check the gap numerically for $\sigma = 1$ and
       $\Delta = 2, 1, 0.5, 0.25, 0.1$, computing the bin probabilities and
       the distortion by numerical integration. Part (b) uses two
       approximations, $D \approx \Delta^2/12$ and
       $H(X_\Delta) \approx h(X) - \log \Delta$. Which of them accounts for
       the deviation of the gap from its limit at $\Delta = 1$ and
       $\Delta = 0.5$?
    1. A learned codec rounds each coordinate of its latent vector to the
       nearest integer and entropy-codes the result. What does the gap of
       (b) imply for such a codec at high rate? Which kind of quantizer can
       close the gap?
   :exlabel:`ex_mdl-information-theory-scalar-quantization`

1. **Binary channels.** ● For a channel $p(y \mid x)$, let
   $H(Y \mid X) = \sum_x p(x)\, H(Y \mid X = x)$ be the average entropy of
   the output given the input. The mutual information in
   :eqref:`eq_mdl-channel-capacity` is then
   $I(X; Y) = H(Y) - H(Y \mid X)$; :numref:`sec_mdl-mi-recap`
   develops both quantities.
    1. Show that the binary symmetric channel of
       :numref:`subsec_mdl-channel-capacity` has capacity
       $1 - h_2(\varepsilon)$ bits and that the uniform input attains it.
    1. A feature takes the values $\pm 1$ and passes through dropout with
       rate $\alpha$, which replaces it by $0$ with probability $\alpha$
       and rescales it otherwise. Show that this binary erasure channel has
       capacity $1 - \alpha$ bits.
    1. Find the flip probability $\varepsilon$ at which the binary symmetric
       channel has the same capacity as dropout with $\alpha = 0.5$. Why
       does a flipped bit cost more capacity than an erased one?
   :exlabel:`ex_mdl-information-theory-binary-channels`

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/)
:end_tab:

[Discussions](https://d2l.discourse.group/t/36836)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §26.3]{.kicker}

How far is a pair from independent?<br>**mutual information and the objectives of representation learning**.
:::
:::

::: {.slide title="Mutual information: distance from independence"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
$I(X;Y) = D_{\mathrm{KL}}(P_{X,Y}\,\|\,P_X\otimes P_Y)$: how far the joint sits
from the product of its marginals. Symmetric, $\ge 0$, zero iff independent,
and

$$I = H(X) - H(X\mid Y) = H(X)+H(Y)-H(X,Y).$$
:::

::: {.col .fig}
@fig:mdl-it-mi-overlap
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[The calculus of dependence]{.dtitle}

[entropy bookkeeping, invariance, and the data-processing inequality]{.dsub}
:::
:::

::: {.slide title="Joint and conditional entropy"}
[Bookkeeping]{.kicker}

The chain rule $H(X,Y) = H(X) + H(Y\mid X)$ is an accounting identity: the surprise
of the pair is the surprise of $X$ plus the residual in $Y$:

@!mutual-information-joint-conditional

Every entropy quantity, one concrete number.
:::

::: {.slide title="Equivalent definitions of mutual information"}
[Definition]{.kicker}

The shared area of the entropy diagram, computed two ways (the KL-from-
independence form and the entropy identity), gives the same numerical value:

@mutual-information-discrete-joint

Symmetry and $I \ge 0$ are immediate from Gibbs.
:::

::: {.slide title="The Gaussian anchor"}
[Ground truth]{.kicker}

Ground truth is $I = -\tfrac12\log(1-\rho^2)$, yet a histogram is already
biased at $\rho=0.99$:

@!mutual-information-gaussian-binned
:::

::: {.slide title="MI sees what correlation cannot"}
[Nonlinear dependence]{.kicker}

Correlation measures only *linear* association. With $Y = X^2 + \text{noise}$,
$\rho \approx 0$ yet $I \approx 1$ nat, and MI is invariant to any invertible
reparameterization:

@mutual-information-nonlinear-correlation
:::

::: {.slide title="Pointwise MI ranks co-occurrences"}
[PMI]{.kicker}

$\mathrm{pmi}(x,y) = \log\frac{p(x,y)}{p(x)\,p(y)}$ scores a single pair, and
$I = \mathbb{E}[\mathrm{pmi}]$. It corrects for frequency: "new york" beats
"the day" despite far fewer counts:

@!mutual-information-pmi-corpus
:::

::: {.slide title="Processing Cannot Increase Mutual Information"}
[The DPI]{.kicker}

For a Markov chain $X\to Y\to Z$, $I(X;Z) \le I(X;Y)$:

*Proof.* Expand $I(X;Y,Z)$ both ways:
$I(X;Z)+I(X;Y\mid Z) = I(X;Y)+I(X;Z\mid Y)$; the Markov property kills the last
term. $\blacksquare$

@!mutual-information-dpi-check

::: {.d2l-note}
No layer can *create* label information; invertible maps lose nothing. But
conditioning is not processing: $Z = X \oplus Y$ turns two independent bits
into $I(X;Y\mid Z) = \ln 2$.
:::
:::

::: {.slide title="Fano: MI floors the error rate"}
[The guarantee]{.kicker}

$H_b(P_e) + P_e\log(k-1) \ge H(X\mid Y) = H(X) - I(X;Y)$. Too little MI makes
low error *impossible*:

@mutual-information-fano

5% error on 1000 classes demands $6.36$ nats; taking the ceiling at face
value, *certifying* that much MI needs batches of $N \gtrsim e^{6.36} \approx 581$.
The ceiling and the floor meet.
:::

::: {.slide}
::: {.divider}
[04]{.dnum}

[The bottleneck and the limits of MI]{.dtitle}

[compression with a purpose, Fano, and the scope of MI estimates]{.dsub}
:::
:::

::: {.slide title="The information bottleneck"}
[Compression with a purpose]{.kicker}

::: {.cols .vc}
::: {.col}
$\min\; I(X;Z) - \beta\,I(Y;Z)$: squeeze the input, keep the label. The DPI
makes it well-posed ($I(Y;Z)\le I(X;Y)$), and $\beta$ sweeps the frontier;
below $\beta = 1/\rho^2$ the code collapses to nothing.
:::

::: {.col .fig}
@fig:mdl-it-ib-tradeoff
:::
:::
:::

::: {.slide title="Mutual Information Measures Dependence, Not Usefulness"}
[Wrap-up]{.kicker}

::: {.cols}
::: {.col}
- MI $=$ KL from independence; three entropy forms; symmetric, $\ge 0$.
- Gaussian anchor $-\tfrac12\log(1-\rho^2)$; MI catches nonlinear dependence.
- DPI: processing only loses information; invertible maps lose none.
:::

::: {.col}
- Histograms and KSG degrade in high dimension; distribution-free lower
  confidence guarantees grow at most on the order of $\log N$.
- InfoNCE $=$ classification: $\hat I_{\mathrm{NCE}} = \log N - \mathcal L$,
  the loss of CPC, SimCLR, CLIP.
- IB compresses with a purpose; Fano floors the error; read MI as a signal.
:::
:::

::: {.d2l-note}
Every loss in this chapter is a code length, and every divergence a choice of
how to compare distributions.
:::
:::
