# The Exponential Family
:label:`sec_mdl-exponential-family`

This opening is a placeholder from the restructuring pass; the section writer replaces it.

```{.python .input #mdl-exponential-family-the-exponential-family}
#@tab mxnet
from d2l import mxnet as d2l
import numpy as np
```

```{.python .input #mdl-exponential-family-the-exponential-family}
#@tab pytorch
from d2l import torch as d2l
import numpy as np
```

```{.python .input #mdl-exponential-family-the-exponential-family}
#@tab tensorflow
from d2l import tensorflow as d2l
import numpy as np
```

```{.python .input #mdl-exponential-family-the-exponential-family}
#@tab jax
from d2l import jax as d2l
import numpy as np
```

## Definition and Properties

Nearly every distribution above (Bernoulli, categorical, binomial, Poisson,
exponential, Gaussian, Laplace with fixed location) is a special case of one
form, the *exponential family*. A distribution
belongs to it if its density (or mass) can be written

$$
p(\mathbf x\mid\boldsymbol\eta) = h(\mathbf x)\,\exp\!\bigl(\boldsymbol\eta^\top T(\mathbf x) - A(\boldsymbol\eta)\bigr).
$$
:eqlabel:`eq_mdl-exp_family`

The four ingredients each play a clear role. The *base measure* $h(\mathbf x)$ is a
fixed background that does not depend on the parameters. The *natural parameters*
$\boldsymbol\eta$ are the knobs. The *sufficient statistics* $T(\mathbf x)$ are the
only features of the data that matter: given $T(\mathbf x)$, no other detail of
$\mathbf x$ affects the probability, which is exactly what "sufficient" means. The
*log-partition function* (or cumulant function)

$$
A(\boldsymbol\eta) = \log\!\int h(\mathbf x)\,\exp\!\bigl(\boldsymbol\eta^\top T(\mathbf x)\bigr)\,d\mathbf x
$$
:eqlabel:`eq_mdl-log_partition`

is whatever it takes to make :eqref:`eq_mdl-exp_family` integrate to one.

### Identifying Standard Distributions

The form looks abstract until we match the pieces. The **Bernoulli**
$p^x(1-p)^{1-x}$ rewrites as

$$
\exp\!\Bigl(x\log\tfrac{p}{1-p} + \log(1-p)\Bigr),
$$

so $T(x)=x$, the natural parameter $\eta=\log\tfrac{p}{1-p}$ is the *logit*, and
$A(\eta)=\log(1+e^\eta)$ is the *softplus*; the link between probabilities and
logits is built into the family. The **Poisson** $\tfrac{\lambda^k e^{-\lambda}}{k!}$
has base measure $h(k)=1/k!$, statistic $T(k)=k$, parameter $\eta=\log\lambda$, and
$A(\eta)=e^\eta=\lambda$. The **Gaussian** with unknown mean and variance is a
two-parameter member: completing the exponent,

$$
\frac{1}{\sqrt{2\pi}}\exp\!\Bigl(\underbrace{\tfrac{\mu}{\sigma^2}}_{\eta_1}\,x
+ \underbrace{\bigl(-\tfrac{1}{2\sigma^2}\bigr)}_{\eta_2}\,x^2
- \underbrace{\bigl(\tfrac{\mu^2}{2\sigma^2}+\log\sigma\bigr)}_{A(\boldsymbol\eta)}\Bigr),
$$

so $T(x)=(x,x^2)^\top$ with base measure $h(x)=1/\sqrt{2\pi}$. The second natural
parameter is negative, $\eta_2=-1/(2\sigma^2)<0$, which is what forces the
tail-suppressing $e^{-x^2}$ decay. The exact split between $h$, $\eta$, and $T$ is
not unique; what matters is that the distribution *fits the form at all*.

Membership has edges. The conspicuous absentees from the list above
are the two *uniforms*: when the endpoints are unknown parameters, the support of
$U(a,b)$ (and likewise of $U(n)$) moves with those parameters, whereas every member
of :eqref:`eq_mdl-exp_family` has the fixed, parameter-free support of its base
measure $h$. This is the classic counterexample. Heavy-tailed laws live outside
the family for a different reason. The *Student-$t$* takes two lines to define:
draw a random precision $1/\sigma^2$ from a Gamma distribution (a two-parameter
density on the positive half-line, defined in full among the conjugate priors
below), then draw a Gaussian with that precision. Averaging the Gaussian over its
random scale fattens the $e^{-x^2}$ tails into polynomial ones, and the Cauchy of
:numref:`sec_mdl-random_variables` is its one-degree-of-freedom extreme. The
support here is all of $\mathbb R$, fixed, yet the mixing destroys the exponential
form: no fixed finite-dimensional sufficient statistic exists, the situation
described by the *Pitman--Koopman--Darmois* theorem :cite:`Bishop.2006`. The
exclusion costs two conveniences: these laws admit no finite-dimensional
conjugate prior, and their negative log-likelihoods are not convex in the
parameters. Both conveniences are exponential-family privileges, as the rest of
this section shows.

### Maximum-Entropy Derivation
:label:`subsec_mdl-max-entropy-derivation`

Maximum entropy produces an exponential-family form only relative to a chosen
support, base measure, and feasible set of constraints. Fix a base measure
$h$, require $p$ to be absolutely continuous with respect to it, and impose
normalization together with $\mathbb E_p[T(\mathbf x)]=\boldsymbol\tau$. If
the constraints are feasible, the log-partition function is finite, and a
maximizer exists in the regular interior case, the optimizer has the
exponential form below. Existence can fail for incompatible constraints or
heavy-tailed settings.

The two maximum-entropy facts cited earlier
:cite:`Cover.Thomas.1999` are the same
statement seen twice: the discrete uniform is the maximum-entropy distribution on a
finite set when we fix *nothing* but the support, and the Gaussian is the
maximum-entropy distribution on the line when we fix the *mean and variance*. In
general, maximizing the entropy *relative to the base measure* $h$,
$H_h[p]=-\int p\log\tfrac{p}{h}$, subject to fixing the expected
sufficient statistics $\mathbb E[T(\mathbf x)]=\boldsymbol\tau$ has, by a Lagrange
multiplier argument (:numref:`subsec_mdl-lagrange-multipliers`), the solution

$$
p(\mathbf x) \propto h(\mathbf x)\,\exp\!\bigl(\boldsymbol\eta^\top T(\mathbf x)\bigr),
$$

which is precisely :eqref:`eq_mdl-exp_family`: the multipliers $\boldsymbol\eta$
are the natural parameters and $A(\boldsymbol\eta)$ is again the normalizer
:cite:`Murphy.2022,Wainwright.Jordan.2008`. One subtlety: the base measure does
real work here. With plain Shannon entropy, i.e. $h\equiv1$, fixing the mean on
$\{0,1,2,\dots\}$ yields the *geometric* distribution; only entropy relative to
$h(k)=1/k!$ yields the Poisson. For the uniform and the Gaussian above, $h$ is
constant, so plain entropy was the relevant objective in those cases. Thus,
when the optimization problem has a solution, the solution is least committal
only relative to the selected support, base measure, entropy, and moment
constraints. Changing those choices can change the optimizer.

### The Moment Property

The single most useful fact about :eqref:`eq_mdl-exp_family` is that $A$ is a moment
generator: *differentiating the log-partition function returns the mean of the
sufficient statistics.*

**Proposition (mean of the sufficient statistics).**
*For an exponential-family distribution :eqref:`eq_mdl-exp_family`,*

$$
\nabla_{\boldsymbol\eta} A(\boldsymbol\eta) = \mathbb E[T(\mathbf x)].
$$
:eqlabel:`eq_mdl-exp_family_moment`

**Proof.** Differentiate :eqref:`eq_mdl-log_partition` under the integral sign;
we grant the interchange of derivative and integral, which is valid on the
interior of the natural-parameter domain :cite:`Folland.1999`.
Writing $Z(\boldsymbol\eta)=e^{A(\boldsymbol\eta)}=\int h(\mathbf x)e^{\boldsymbol\eta^\top T(\mathbf x)}d\mathbf x$,
the derivative of the log is $\nabla A = (\nabla Z)/Z$, and

$$
\nabla_{\boldsymbol\eta} Z
= \int h(\mathbf x)\,T(\mathbf x)\,e^{\boldsymbol\eta^\top T(\mathbf x)}\,d\mathbf x .
$$

Dividing by $Z=e^{A}$ pulls the $e^{-A}$ back inside, reconstituting the density
:eqref:`eq_mdl-exp_family`:

$$
\nabla_{\boldsymbol\eta} A
= \frac{1}{Z}\int h(\mathbf x)\,T(\mathbf x)\,e^{\boldsymbol\eta^\top T(\mathbf x)}\,d\mathbf x
= \int T(\mathbf x)\,\underbrace{h(\mathbf x)\,e^{\boldsymbol\eta^\top T(\mathbf x) - A(\boldsymbol\eta)}}_{p(\mathbf x\mid\boldsymbol\eta)}\,d\mathbf x
= \mathbb E[T(\mathbf x)] . \quad\blacksquare
$$

Differentiating once more (the same interchange) turns $\nabla A=(\nabla Z)/Z$
into

$$
\nabla^2 A
= \frac{\nabla^2 Z}{Z} - \frac{(\nabla Z)(\nabla Z)^\top}{Z^2}
= \mathbb E[T\,T^\top] - \mathbb E[T]\,\mathbb E[T]^\top
= \operatorname{Cov}(T),
$$

so $A$ is *convex* (a covariance matrix is PSD, :numref:`subsec_mdl-psd`).
Consequently, for a sample $x_1,\dots,x_n$ the negative
log-likelihood is

$$
-\sum_{i=1}^{n} \log p(x_i\mid\boldsymbol\eta)
= n\,A(\boldsymbol\eta) - \boldsymbol\eta^\top \sum_{i=1}^{n} T(x_i)
- \sum_{i=1}^{n} \log h(x_i),
$$
:eqlabel:`eq_mdl-exp_family_nll`

convex in $\boldsymbol\eta$ because $A$ is. Setting its gradient to zero gives
$n\nabla A(\boldsymbol\eta)=\sum_i T(x_i)$: by the moment property, maximum
likelihood *matches the model's expected sufficient statistics to their empirical
averages*, $\mathbb E[T]=\bar T$. Uniqueness of the optimum needs one more
hypothesis, strict convexity: $\operatorname{Cov}(T)\succ0$, which holds when the
representation is *minimal* (no linear combination of the sufficient statistics
is constant). The section's own softmax parameterization :eqref:`eq_mdl-softmax`
is *not* minimal: adding the same constant to every logit leaves every $p_k$
unchanged, so the categorical negative log-likelihood is flat along the direction
$\mathbf 1$ and its optima form a line rather than a point. Multinomial logistic
regression in statistics software removes this direction by fixing one logit at
zero. Deep-learning implementations keep all $K$ logits and use the invariance
only to subtract the maximum logit before exponentiating, which prevents
overflow (:numref:`sec_softmax_concise`). We verify
:eqref:`eq_mdl-exp_family_moment` for the Bernoulli, where it should read
$A'(\eta)=\sigma(\eta)=\mathbb E[x]=p$.

```{.python .input #distributions-exp-family}
import numpy as onp
eta = 0.7                                        # natural parameter (logit)
A = lambda e: onp.log1p(onp.exp(e))                # Bernoulli log-partition (softplus)
eps = 1e-6
dA = (A(eta + eps) - A(eta - eps)) / (2 * eps)   # numerical dA/deta
print('dA/deta      =', round(float(dA), 6))
print('sigmoid(eta) = E[x] = p =', round(float(1 / (1 + onp.exp(-eta))), 6))
```

The derivative of the softplus is the sigmoid, and it lands exactly on the
Bernoulli mean $p$. This is no coincidence: the sigmoid and softmax links that
deep-learning classifiers use are *precisely* the maps from natural parameters to
means in the Bernoulli and categorical families, and the convexity of $A$ is why
their negative log-likelihoods are convex in the natural parameters. If those
parameters are affine functions of fixed features, this gives a convex
optimization problem. A neural network makes the natural parameters nonlinear
functions of its weights, so the same loss is not generally convex in the
weights. The exponential family still supplies the likelihood and the link
functions used by standard deep-learning losses
:cite:`Bishop.2006,Koller.Friedman.2009`.

### Generalized Linear Models
:label:`subsec_mdl-glm`

Regression and classification models predict a distribution of a label $y$
for each input $\mathbf x$. A *generalized linear model* takes that
distribution from an exponential family and makes its natural parameter a
linear function of the input, $\boldsymbol\eta = \mathbf W^\top\mathbf x$ (a
bias is a constant feature). Write $\boldsymbol\mu(\mathbf x) =
\nabla A(\boldsymbol\eta) = \mathbb E[T(y)\mid\mathbf x]$ for the model's mean
of the sufficient statistic. The negative log-likelihood of one example is
$A(\boldsymbol\eta) - \boldsymbol\eta^\top T(y) - \log h(y)$, so by the moment
property its gradient with respect to $\boldsymbol\eta$ is
$\boldsymbol\mu(\mathbf x) - T(y)$, and the chain rule through
$\boldsymbol\eta = \mathbf W^\top\mathbf x$ gives

$$
\nabla_{\mathbf W}\bigl(-\log p(y\mid\mathbf x)\bigr)
= \mathbf x\,\bigl(\boldsymbol\mu(\mathbf x) - T(y)\bigr)^\top .
$$
:eqlabel:`eq_mdl-glm-gradient`

The gradient is the input times the error of the predicted mean: prediction
minus observation, whatever the family. The loss is convex in $\mathbf W$,
since it is the convex function $A$ composed with a linear map, plus a linear
term. Four families give four familiar models:

| label $y$ | family | $T(y)$ | mean $\boldsymbol\mu(\mathbf x)$ | model |
|:--|:--|:--|:--|:--|
| real number | Gaussian, variance 1 | $y$ | $\eta$ | linear regression |
| yes or no | Bernoulli | $y \in \{0, 1\}$ | $\sigma(\eta)$ | logistic regression |
| one of $K$ classes | categorical | one-hot $\mathbf y$ | $\mathrm{softmax}(\boldsymbol\eta)$ | softmax regression |
| count | Poisson | $y$ | $e^{\eta}$ | Poisson regression |

With variance $\sigma^2$ instead of 1, the Gaussian negative log-likelihood
is $(\hat y - y)^2/(2\sigma^2)$ plus a constant, where $\hat y =
\mathbf w^\top\mathbf x$: the squared loss of :numref:`sec_linear_regression`
divided by $\sigma^2$, with the gradient $\mathbf x(\hat y - y)/\sigma^2$. Softmax regression in
:numref:`sec_softmax` has the gradient $\mathbf x(\hat{\mathbf y} -
\mathbf y)^\top$ for the same reason. A neural network keeps the last step,
a linear map into the natural parameters followed by the family's
negative log-likelihood, and replaces the fixed input $\mathbf x$ by learned
features. The loss is then no longer convex in the weights, but its
gradient with respect to the natural parameters is still prediction minus
observation.

### Origins of the Softmax
:label:`subsec_mdl-softmax-origins`

The softmax :eqref:`eq_mdl-softmax` turns $K$ logits into class
probabilities, and the table above lists it as the mean of the categorical
family. Other maps would also do. Normalizing the softplus values
$\log(1+e^{z_k})$ instead of $e^{z_k}$ gives positive numbers that sum to
one and keep the order of the logits. The choice fixes the loss we train
with and the meaning of a logit: under the softmax,
$z_i-z_j=\log(p_i/p_j)$ is a log-ratio of probabilities. Three arguments
from different fields lead to the exponential: a model of noisy choices
from psychology and economics, the Boltzmann distribution of statistical
physics, and the exponential family. Each explains a different property of
the softmax.

**Noisy choices.** Suppose that a decision maker, such as a subject in a
perception experiment, a shopper, or whoever labels our data, gives each
option $k$ a score $z_k$ but does not always pick the best one. A simple
model of this behavior perturbs every score by independent noise
$\epsilon_k$ and picks the largest perturbed score,

$$
y = \operatorname*{argmax}_{k}\,(z_k+\epsilon_k).
$$

The probability of class $k$ is the chance that option $k$ wins this noisy
contest, so the noise distribution decides its form.
:citet:`Fechner.1860` analyzed judgments of which of two lifted weights is
heavier in this way, with Gaussian errors, and :citet:`Thurstone.1927` made
the model a general law of comparative judgment. With two options and
$\epsilon_k\sim\mathcal N(0,s^2)$, option 1 wins when
$\epsilon_2-\epsilon_1<z_1-z_2$. The difference $\epsilon_2-\epsilon_1$ is
Gaussian with variance $2s^2$, so

$$
P(y=1) = \Phi\Bigl(\frac{z_1-z_2}{\sqrt2\,s}\Bigr),
$$

where $\Phi$ is the standard Gaussian CDF. This is the *probit* model: the
probability depends only on the score difference, measured in units of the
noise.

With more than two options the probit model has no closed form. Let $\phi$
be the standard Gaussian density. Option $k$ wins when every other
perturbed score falls below its own, and averaging this chance over the
standardized noise $u=\epsilon_k/s$ of option $k$ gives

$$
P(y=k) = \int_{-\infty}^{\infty}\phi(u)\prod_{j\ne k}\Phi\Bigl(u+\frac{z_k-z_j}{s}\Bigr)\,du .
$$

For $K>2$ this integral and its gradient must be computed numerically, for
every example at every training step. If the noise is correlated across
options, the integral becomes $(K-1)$-dimensional.

Gumbel noise removes the integral. The standard *Gumbel* distribution has
the CDF $G(x)=\exp(-e^{-x})$. It is the limiting distribution of the
suitably shifted and scaled maximum of many independent draws from
distributions such as the Gaussian or the exponential :cite:`Gumbel.1954`,
and :citet:`McFadden.1974` used it as the noise in models of discrete
economic choices.

**Proposition (Gumbel noise gives the softmax).** *If
$\epsilon_1,\dots,\epsilon_K$ are independent standard Gumbel variables,
the noisy argmax follows the softmax probabilities,*

$$
P\Bigl(\operatorname*{argmax}_j\,(z_j+\epsilon_j)=k\Bigr) = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}} .
$$

**Proof.** Write $S=\sum_j e^{z_j}$. The perturbed score $z_j+\epsilon_j$
has the CDF $G(t-z_j)=\exp(-e^{z_j}e^{-t})$. If option $k$ scores
$t=z_k+\epsilon_k$, it wins with probability
$\prod_{j\ne k}\exp(-e^{z_j}e^{-t})=\exp\bigl(-(S-e^{z_k})e^{-t}\bigr)$, by
independence. We multiply by the density
$e^{z_k}e^{-t}\exp(-e^{z_k}e^{-t})$ of $t$, integrate, and substitute
$v=e^{-t}$:

$$
P(y=k) = \int_{-\infty}^{\infty} e^{z_k}e^{-t}\exp\bigl(-Se^{-t}\bigr)\,dt
= e^{z_k}\int_0^{\infty} e^{-Sv}\,dv = \frac{e^{z_k}}{S} . \quad\blacksquare
$$

The proof works because a product of Gumbel CDFs is again a Gumbel CDF.
Over all $K$ options,
$P\bigl(\max_j(z_j+\epsilon_j)\le t\bigr)=\exp(-Se^{-t})=G(t-\log S)$, so
the winning score is a standard Gumbel variable shifted by the log-sum-exp
$\log\sum_j e^{z_j}$. The log-sum-exp is thus a smoothed maximum of the
logits, and its gradient, the softmax, gives the chance that each option
attains the maximum. Exercise
:exref:`ex_mdl-distributions-exponential-races` proves the proposition a
second way, through races between exponential waiting times.

Read from right to left, the proposition is the Gumbel-max sampler of the
categorical paragraph above. Because
$\operatorname{argmax}_k(z_k+T\epsilon_k)=\operatorname{argmax}_k(z_k/T+\epsilon_k)$
for $T>0$, Gumbel noise of scale $T$ samples from
$\mathrm{softmax}(\mathbf z/T)$: the amount of noise in the choice acts as
a temperature.

With two options, the difference of two independent standard Gumbel
variables has the logistic distribution, whose CDF is the sigmoid
$\sigma(x)=1/(1+e^{-x})$, so this *logit* model gives
$P(y=1)=\sigma(z_1-z_2)$. The sigmoid is close to a rescaled Gaussian CDF,
$|\sigma(1.7x)-\Phi(x)|<0.01$ for all $x$, so fitted probit and logit
models give nearly the same probabilities and differ mainly in the scale of
their scores. They part in the tails, which decay like $e^{-x}$ for the
logistic and like $e^{-x^2/2}$ for the Gaussian. A confidently
misclassified example therefore costs a negative log-likelihood linear in
its score difference under the logit model and quadratic under the probit
model, so mislabeled examples pull a probit fit harder
(exercise :exref:`ex_mdl-distributions-probit-logit`).

The softmax has one property that is unrealistic for similar options: the
ratio $p_i/p_j=e^{z_i-z_j}$ does not depend on the other options. If a
commuter takes the car or a red bus with probability $\tfrac12$ each,
adding an identical blue bus should leave the car at $\tfrac12$, but a
softmax with equal scores for the two buses gives all three options
$\tfrac13$. A probit model with strongly correlated noise for the two buses
keeps the car near $\tfrac12$, which is why economists accept its
integrals for such choices.

**The Boltzmann distribution.** A physical system in thermal equilibrium at
temperature $T$, such as a molecule in a gas, occupies a state $k$ of
energy $E_k$ with probability

$$
p_k = \frac{\exp\bigl(-E_k/(k_{\mathrm B}T)\bigr)}{\sum_j \exp\bigl(-E_j/(k_{\mathrm B}T)\bigr)},
$$

where $k_{\mathrm B}$ is Boltzmann's constant. Low-energy states are the
most probable, and the temperature sets how strongly they are preferred.
Boltzmann found this law for the molecules of a gas, and
:citet:`Gibbs.1902` made it the basis of statistical mechanics. If we
measure temperature in units of energy, so that $k_{\mathrm B}=1$, and set
$E_k=-z_k$, the law becomes $\mathrm{softmax}(\mathbf z/T)$. A logit is a
negative energy, and the most probable class is the state of lowest energy.
As $T\to0$ the distribution concentrates on the largest logit, the argmax;
as $T\to\infty$ it becomes uniform.

Maximum entropy explains the exponential without appeal to physics
:cite:`Jaynes.1957`. By :numref:`subsec_mdl-max-entropy-derivation`, the
distribution of largest entropy with prescribed expected values of some
statistics is proportional to the exponential of a linear combination of
those statistics, with the Lagrange multipliers as coefficients. On $K$
states with the single constraint $\sum_k p_kE_k=\bar E$, the maximizer is
therefore $p_k\propto\exp(-E_k/T)$, with $-1/T$ as the multiplier. The
softmax is thus the least committal distribution over the classes among
all those with a given average logit $\sum_k p_kz_k$. The higher the
required average, the lower the temperature. As $T$ falls from $\infty$ to
$0$, the average logit rises from the mean of the logits to their maximum.

*Energy-based models* take the physical reading literally
:cite:`LeCun.Chopra.Hadsell.ea.2006,Ranzato.Boureau.Chopra.ea.2007`. They
learn an energy $E(\mathbf x,y)$ for every pair of input and output and set
$p(y\mid\mathbf x)\propto\exp(-E(\mathbf x,y))$. A classifier is the case
$E(\mathbf x,k)=-z_k(\mathbf x)$, in which the normalizing sum has only $K$
terms. When $y$ ranges over images or sentences, the sum has far too many
terms to compute, and avoiding it is the central problem of energy-based
learning.

**The exponential family.** The third derivation is the one this section
has already made. With the class coded as a one-hot vector $\mathbf y$, the
categorical distribution with probabilities $\mathrm{softmax}(\mathbf z)$
reads

$$
P(\mathbf y) = \exp\bigl(\mathbf z^\top\mathbf y - A(\mathbf z)\bigr),
\qquad
A(\mathbf z) = \log\sum_{k=1}^K e^{z_k}.
$$

This is an exponential family :eqref:`eq_mdl-exp_family` with the logits as
natural parameters, the one-hot vector as sufficient statistic, and the
log-sum-exp as log-partition function. By the moment property
:eqref:`eq_mdl-exp_family_moment`, the softmax is its mean map,
$\nabla A(\mathbf z)=\mathrm{softmax}(\mathbf z)=\mathbb E[\mathbf y]$
(exercise :exref:`ex_mdl-distributions-minimal-categorical` works out the
details). This view explains how the softmax trains: as
:numref:`subsec_mdl-glm` showed, its negative log-likelihood is convex in
the logits and has the gradient $\mathrm{softmax}(\mathbf z)-\mathbf y$.
The three derivations agree on the function. Noisy choices give the labels
a generative model and an exact sampler, the Boltzmann distribution gives
the temperature and the maximum-entropy reading, and the exponential family
gives the convex loss and its gradient.

**Geometry of the argmax.** Under every derivation the most probable class
has the largest logit, so a linear model
$z_k(\mathbf x)=\mathbf w_k^\top\mathbf x+b_k$ assigns class $j$ to the
region $\{\mathbf x : z_j(\mathbf x)\ge z_k(\mathbf x)\text{ for all }k\}$.
This region is an intersection of $K-1$ halfspaces, hence a convex
polyhedron, possibly empty. Two classes tie on the hyperplane
$z_j(\mathbf x)=z_k(\mathbf x)$. For three classes the tie hyperplanes are
linearly dependent, since $(z_1-z_2)+(z_2-z_3)=z_1-z_3$, so a point on two
of them lies on the third. In the plane, two tie lines that are not
parallel meet in a point, which then lies on the third line as well
(:numref:`fig_mdl-clf-decision-regions`); if two of the lines are parallel,
so is the third. In general, all $K$ scores tie on the solutions of $K-1$
affine equations in $\mathbb R^d$, a set of dimension $d-K+1$ for generic
weights and empty when $K>d+1$, so four classes in the plane generically
share no point. In the noisy-choice model, ties have probability zero under
any continuous noise, so the argmax is unique and the class probabilities
sum to one.

## Conjugate Priors

The distributions above model observations, but Bayesian inference also
requires distributions over their parameters: the Bernoulli probability $p$,
the Poisson rate $\lambda$, or the categorical probability vector $\mathbf p$.
Such a distribution is a *prior*. Three continuous distributions provide the
standard priors for the discrete likelihoods introduced above: the **Beta**,
**Gamma**, and **Dirichlet** distributions.

A prior is **conjugate** to a likelihood if the posterior belongs to the same
family as the prior. Observations then update the parameters of the
distribution without changing its functional form. The dashed links in
:numref:`fig_mdl-prob-family-tree` show these conjugate pairs.

### Beta--Bernoulli Conjugacy and Pseudo-Counts

The **Beta** distribution is a density on the unit interval $p\in[0,1]$ (exactly
the range of a probability) with two shape parameters $\alpha,\beta>0$,

$$
\mathrm{Beta}(p\mid\alpha,\beta) = \frac{p^{\alpha-1}(1-p)^{\beta-1}}{B(\alpha,\beta)},
$$
:eqlabel:`eq_mdl-beta_pdf`

where the normalizing constant is
$B(\alpha,\beta)=\Gamma(\alpha)\Gamma(\beta)/\Gamma(\alpha+\beta)$, with $\Gamma$
the Gamma function that the exercises of :numref:`sec_mdl-integral_calculus`
introduced (recall $\Gamma(z+1)=z\,\Gamma(z)$ and $\Gamma(n+1)=n!$). The mean
follows from this normalizer:

$$
\mathbb E[p]
= \frac{1}{B(\alpha,\beta)}\int_0^1 p^{\alpha}(1-p)^{\beta-1}\,dp
= \frac{B(\alpha+1,\beta)}{B(\alpha,\beta)}
= \frac{\alpha}{\alpha+\beta},
$$

using $\Gamma(\alpha+1)=\alpha\,\Gamma(\alpha)$ to simplify the ratio; the same
move one step further gives the variance
$\operatorname{Var}(p)=\alpha\beta/\bigl((\alpha+\beta)^2(\alpha+\beta+1)\bigr)$.
The uniform prior on $[0,1]$ that :numref:`sec_mdl-maximum_likelihood` will place
on the coin's bias is the special case $\alpha=\beta=1$.

Now take a Beta prior on a coin's bias $p$ and observe a Bernoulli/binomial sample
with $x$ heads in $n$ flips. Multiplying the prior by the likelihood
:eqref:`eq_mdl-binomial_pmf`,

$$
\underbrace{p^{\alpha-1}(1-p)^{\beta-1}}_{\text{prior}}\cdot
\underbrace{p^{x}(1-p)^{n-x}}_{\text{likelihood}}
= p^{(\alpha+x)-1}(1-p)^{(\beta+n-x)-1},
$$

which, up to normalization, is again a Beta, now with parameters shifted by the
data:

$$
p \mid x \;\sim\; \mathrm{Beta}\bigl(\alpha+x,\ \beta+(n-x)\bigr).
$$
:eqlabel:`eq_mdl-beta_posterior`

This update has a **pseudo-count** interpretation. The parameters
$\alpha$ and $\beta$ act as prior counts for heads and tails. Observing $x$
heads and $n-x$ tails adds these observations to the corresponding prior
counts. The posterior mean

$$
\mathbb E[p\mid x] = \frac{\alpha+x}{\alpha+\beta+n}
$$

is the empirical frequency after including the pseudo-counts. It lies between
the prior mean $\alpha/(\alpha+\beta)$ and the maximum-likelihood estimate
$x/n$, and approaches the latter as $n$ grows. With the uniform prior
$\alpha=\beta=1$ it becomes *Laplace's rule of succession* $(x+1)/(n+2)$
:cite:`Laplace.1814`: estimate a
probability by adding one phantom observation of each outcome. The cell verifies
the update on a small sample.

```{.python .input #distributions-conjugate}
alpha, beta = 2.0, 2.0                           # Beta prior pseudo-counts
x, n = 7, 10                                     # 7 heads in 10 flips
post_alpha, post_beta = alpha + x, beta + (n - x)  # Beta posterior update
print('prior mean      =', round(alpha / (alpha + beta), 4))
print('MLE  x/n        =', round(x / n, 4))
print('posterior Beta  =', (post_alpha, post_beta))
print('posterior mean  =', round(post_alpha / (post_alpha + post_beta), 4))
```

The posterior mean lies between the prior mean $0.5$ and the empirical
frequency $0.7$, as the pseudo-count calculation predicts.
:numref:`fig_mdl-prob-beta-posterior` shows the same update with more data: the
flat $\mathrm{Beta}(1,1)$ prior becomes concentrated around the true bias, and
its width decreases as $1/\sqrt n$.

![Bayesian updating as sharpening. Starting from the flat $\mathrm{Beta}(1,1)$ prior (left), observing $9$ heads in $13$ flips of a coin with true bias $p^\ast=0.7$ (dashed line) gives the $\mathrm{Beta}(10,5)$ posterior (middle); after $130$ flips the $\mathrm{Beta}(91,41)$ posterior has piled up tightly around $p^\ast$ (right). Each update just adds the observed heads and tails to the pseudo-counts, and the posterior standard deviation shrinks like $1/\sqrt{n}$.](../img/mdl-prob-beta-posterior.svg)
:label:`fig_mdl-prob-beta-posterior`

### Other Conjugate Pairs
:label:`subsec_mdl-other-conjugate-pairs`

The other discrete laws have their own conjugate partners, built the same way.

* **Gamma--Poisson.** The **Gamma** distribution
  $\mathrm{Gamma}(\lambda\mid\alpha,\beta)=\frac{\beta^\alpha}{\Gamma(\alpha)}\,\lambda^{\alpha-1}e^{-\beta\lambda}$
  is a density on the rate $\lambda>0$ with mean $\alpha/\beta$ and variance
  $\alpha/\beta^2$. Its conjugacy with the Poisson follows by the same
  multiplication used for the Beta: $n$ observed counts $x_1,\dots,x_n$
  contribute the likelihood
  $\propto\lambda^{\sum_i x_i}e^{-n\lambda}$, and multiplying by the prior
  collects powers into $\lambda^{\alpha+\sum_i x_i-1}e^{-(\beta+n)\lambda}$, the
  posterior $\mathrm{Gamma}\bigl(\alpha+\sum_i x_i,\ \beta+n\bigr)$: pseudo-events
  over a pseudo-window. Averaging the Poisson over this Gamma uncertainty in its
  rate yields the *negative binomial*, the over-dispersed fix promised in the
  Poisson section.
* **Gamma as a waiting time.** The Gamma is more than a prior: with integer shape
  $\alpha=k$, $\mathrm{Gamma}(k,\beta)$ is the waiting time to the *$k$-th* event
  of the Poisson process we named at the bus stop, the sum of $k$ independent
  $\mathrm{Exp}(\beta)$ waits, extending the exponential's arrow in the family
  tree just as the binomial extends the Bernoulli's.
* **Dirichlet--Multinomial.** The **Dirichlet** distribution

  $$
  \mathrm{Dir}(\mathbf p\mid\boldsymbol\alpha)=\frac{\Gamma(\sum_k\alpha_k)}{\prod_k\Gamma(\alpha_k)}\prod_k p_k^{\alpha_k-1}
  $$

  is the multivariate Beta, a density over probability vectors on the simplex,
  with mean $\mathbb E[p_k]=\alpha_k/\sum_j\alpha_j$. It is conjugate to the
  categorical/multinomial by the same multiplication: the likelihood contributes
  $\prod_k p_k^{x_k}$, so observing class counts $\mathbf x$ updates the prior to
  $\mathrm{Dir}(\boldsymbol\alpha+\mathbf x)$, one pseudo-count per class.

These are three instances of one theorem. **Every exponential-family likelihood
has a conjugate prior**, provided the hyperparameters are chosen so that the
prior below is normalizable, and that prior is itself an exponential family in
the natural parameters $\boldsymbol\eta$:
$p(\boldsymbol\eta)\propto\exp\!\bigl(\boldsymbol\nu^\top\boldsymbol\eta - \kappa\,A(\boldsymbol\eta)\bigr)$.
Multiplying this prior by the likelihood :eqref:`eq_mdl-exp_family` adds the data's
sufficient statistics to $\boldsymbol\nu$ and increments $\kappa$ by the sample
size. The pseudo-count update of :eqref:`eq_mdl-beta_posterior` is this general
mechanism specialized to the Bernoulli, and the hyperparameters
$(\boldsymbol\nu,\kappa)$ are *pseudo-data*: a prior sufficient statistic and a prior
sample size :cite:`Bishop.2006,Murphy.2022`. Conjugacy makes the exponential
family the natural home of tractable Bayesian inference as well as of the
standard losses.

## Summary

* The **exponential family** $p(\mathbf x)=h(\mathbf x)\exp(\boldsymbol\eta^\top
  T(\mathbf x)-A(\boldsymbol\eta))$ unifies almost all of them (the uniforms, whose
  support moves with their parameters, stay outside). Under feasible moment
  constraints and regularity conditions, the form arises by maximizing
  entropy relative to a selected base measure $h$. Its
  log-partition $A$ generates moments: $\nabla A(\boldsymbol\eta)=\mathbb E[T(\mathbf x)]$,
  and $A$ is convex. Thus its negative log-likelihood is convex in the natural
  parameters; with an affine parameter map this yields a convex generalized
  linear model, whereas a neural-network parameterization remains nonconvex.
  In such a model the gradient is the input times prediction minus
  observation, which is why linear, logistic, softmax, and Poisson regression
  share one training rule.
* The **softmax** arises three ways: as the choice under Gumbel-perturbed scores
  (Gaussian noise gives the probit model instead), as the Boltzmann distribution
  at temperature $T$, and as the mean map of the categorical family.
* Every exponential-family likelihood has a **conjugate prior**: the **Beta** for the
  Bernoulli/binomial, the **Gamma** for the Poisson, the **Dirichlet** for the
  categorical/multinomial. Bayesian updating then just adds the data's sufficient
  statistics to the prior's *pseudo-counts*: e.g. a $\mathrm{Beta}(\alpha,\beta)$
  prior and $x$ heads in $n$ flips give a $\mathrm{Beta}(\alpha+x,\beta+n-x)$
  posterior.

## Exercises

1. **Exponential races.** Let $X_1,\dots,X_m$ be independent with
   $X_i\sim\mathrm{Exp}(\lambda_i)$, for example the waiting times for the next
   event in $m$ independent event streams.
    1. Show that $\min_i X_i\sim\mathrm{Exp}\bigl(\sum_i\lambda_i\bigr)$.
    1. Show that the $j$-th stream fires first with probability
       $P\bigl(X_j=\min_i X_i\bigr)=\lambda_j/\sum_i\lambda_i$.
    1. Let $E_1,\dots,E_K$ be independent $\mathrm{Exp}(1)$ variables and let
       $\mathbf z\in\mathbb R^K$ be a vector of logits. Use (b) to show that
       $\operatorname{argmax}_k\,(z_k-\log E_k)$ is distributed according to
       the softmax probabilities :eqref:`eq_mdl-softmax`. Because $-\log E_k$
       has the standard Gumbel distribution, this result is the Gumbel-max
       trick.
   :exlabel:`ex_mdl-distributions-exponential-races`

1. **Maximum entropy on the integers.** Consider distributions
   $(p_0,p_1,p_2,\dots)$ on $\{0,1,2,\dots\}$ with a fixed mean $m>0$.
   :numref:`subsec_mdl-max-entropy-derivation` asserts that the base measure
   decides which of them maximizes entropy.
    1. Maximize the Shannon entropy $H[p]=-\sum_k p_k\log p_k$ subject to
       $\sum_k p_k=1$ and $\sum_k k\,p_k=m$, using Lagrange
       multipliers (:numref:`subsec_mdl-lagrange-multipliers`). Show that the stationary
       point is a geometric distribution $p_k=(1-q)\,q^k$, and express $q$
       through $m$.
    1. Repeat the derivation for the entropy relative to $h(k)=1/k!$,
       $H_h[p]=-\sum_k p_k\log\bigl(p_k/h(k)\bigr)$, and show that the
       maximizer is $\mathrm{Poisson}(m)$.
    1. At $m=4$, compute the Shannon entropy of both distributions
       numerically. Which one is larger, and why must it be? Compare
       $P(X\ge15)$ under the two distributions. Which assumption about large
       counts does the choice of base measure encode?
   :exlabel:`ex_mdl-distributions-max-entropy-integers`

1. **Moments from the log-partition.** The moment property
   :eqref:`eq_mdl-exp_family_moment` and its second-order form
   $\nabla^2A=\operatorname{Cov}(T)$ turn moment calculations into
   differentiation.
    1. For the Poisson, $A(\eta)=e^\eta$. Derive the mean and the variance
       from $A$. Which property of the function $A$ makes them equal?
    1. For the Gaussian, the natural parameters are $\eta_1=\mu/\sigma^2$
       and $\eta_2=-1/(2\sigma^2)$. Express $A$ as a function of
       $(\eta_1,\eta_2)$. Differentiate it to recover $\mathbb E[x]$,
       $\mathbb E[x^2]$, $\operatorname{Var}(x)$ and
       $\operatorname{Cov}(x,x^2)$.
    1. The Laplace distribution with a fixed location $\mu$ has the
       sufficient statistic $T(x)=|x-\mu|$. Write it in the form
       :eqref:`eq_mdl-exp_family`, and use $A$ to compute
       $\mathbb E\,|x-\mu|$ and $\operatorname{Var}|x-\mu|$. Check the result
       against the Laplace variance $2b^2$.
   :exlabel:`ex_mdl-distributions-log-partition-moments`

1. [code] **Probit and logit.** For two classes with the score difference
   $m=z_1-z_2$, the logit model gives $P(y=1)=\sigma(m)$ and the probit model
   with noise scale $s$ gives $P(y=1)=\Phi\bigl(m/(\sqrt2\,s)\bigr)$
   (:numref:`subsec_mdl-softmax-origins`).
    1. Show that the difference of two independent standard Gumbel variables
       has the CDF $\sigma(x)$. The standard Gumbel distribution has variance
       $\pi^2/6$. Find the scale $c$ for which the distribution with CDF
       $\sigma(cx)$ has variance 1, like the standard Gaussian.
    1. Find numerically the scale $c$ that minimizes
       $\max_x|\sigma(cx)-\Phi(x)|$, and the value of this maximum. Why is
       this scale smaller than the one of (a)?
    1. An example of class 2 lies on the wrong side of the decision
       boundary, with $m>0$. Show that its negative log-likelihood
       $-\log\sigma(-m)$ under the logit model lies between $m$ and
       $m+e^{-m}$, and that its derivative with respect to $m$ is less than
       1. Use the bounds
       $\frac{x}{1+x^2}\,\phi(x)\le1-\Phi(x)\le\frac{\phi(x)}{x}$ for $x>0$ to
       show that its negative log-likelihood
       $-\log\Phi\bigl(-m/(\sqrt2\,s)\bigr)$ under the probit model grows like
       $m^2/(4s^2)$, and that its derivative grows linearly in $m$.
    1. Draw 100 inputs $x\sim\mathcal N(1,1)$ of class 1 and 100 inputs
       $x\sim\mathcal N(-1,1)$ of class 2, and let $m=wx+b$. Fit $(w,b)$ by
       maximum likelihood under both models. Then add three examples at
       $x=6$ labeled as class 2, and fit again. For each model, compute the
       ratio of the slope $w$ with these mislabeled examples to the slope
       without them, and explain the difference between the models with (c).
       Compute $\log\Phi$ with a numerically stable routine such as
       `scipy.special.log_ndtr`.
   :exlabel:`ex_mdl-distributions-probit-logit`

1. **Minimal categorical family.** ● Code a categorical outcome as a one-hot
   vector $\mathbf x\in\{0,1\}^K$, with class probabilities
   $\mathbf p=\mathrm{softmax}(\mathbf z)$ given by :eqref:`eq_mdl-softmax`.
    1. Write $P(\mathbf x)$ in the form :eqref:`eq_mdl-exp_family` with
       $T(\mathbf x)=\mathbf x$ and natural parameters
       $\boldsymbol\eta=\mathbf z$. Find $A(\mathbf z)$ and verify that
       $\nabla A(\mathbf z)=\mathrm{softmax}(\mathbf z)$. Deduce the gradient
       of the cross-entropy loss $-\log P(\mathbf y)$ of a one-hot label
       $\mathbf y$ with respect to the logits.
    1. Compute $\nabla^2 A(\mathbf z)$ and show that it equals the
       covariance $\operatorname{diag}(\mathbf p)-\mathbf p\mathbf p^\top$
       of the one-hot vector $\mathbf x$. Every one-hot vector satisfies
       $\mathbf x\mathbf x^\top=\operatorname{diag}(\mathbf x)$. Show that
       $\nabla^2A(\mathbf z)\,\mathbf 1=\mathbf 0$, and that the gradient of
       the negative log-likelihood of a dataset is orthogonal to $\mathbf 1$
       at every $\mathbf z$. Gradient descent with a small step size runs on
       the logits from a starting point $\mathbf z_0$, and every class
       occurs in the data. Which function of $\mathbf z$ does gradient
       descent never change? Which of the minimizers does it therefore
       approach?
    1. Fix $z_K=0$. This gives a minimal representation with
       $\boldsymbol\eta=(z_1,\dots,z_{K-1})$ and
       $T(\mathbf x)=(x_1,\dots,x_{K-1})$. Show that $\nabla^2A(\boldsymbol\eta)$
       is positive definite whenever every $p_k>0$. Given class counts
       $n_1,\dots,n_K$, solve the moment-matching condition
       $\nabla A(\boldsymbol\eta)=\bar T$ for the maximum-likelihood
       $\hat{\boldsymbol\eta}$. What happens when some class has $n_k=0$?
   :exlabel:`ex_mdl-distributions-minimal-categorical`

1. [code] **Poisson regression.** ● A generalized linear model for counts
   makes the natural parameter of a Poisson an affine function of the
   features: $y_i\mid\mathbf x_i\sim\mathrm{Poisson}(e^{\eta_i})$ with
   $\eta_i=\mathbf w^\top\mathbf x_i+b$.
    1. Use the negative log-likelihood :eqref:`eq_mdl-exp_family_nll` to
       write the loss as a function of $(\mathbf w,b)$, and compute its
       gradient and Hessian. Show that the loss is convex. Which equations
       does a zero gradient impose on the fitted means
       $\hat m_i=e^{\hat\eta_i}$?
    1. Generate $n=1000$ examples with $\mathbf x_i\sim\mathcal N(\mathbf 0,\mathbf I_2)$,
       $\mathbf w=(-0.3,1.0)$ and $b=0.5$. Starting from zero, fit
       $(\mathbf w,b)$ with the Newton step :eqref:`eq_mdl-opt-newton-step`,
       and check the equations of (a) at the solution.
    1. Multiply each rate by an independent $g_i\sim\mathrm{Gamma}(1,1)$,
       which has mean $1$, so that the counts become overdispersed while their
       conditional means stay $e^{\eta_i}$. Fit the Poisson model to $200$
       such datasets and to $200$ datasets without the multiplier. Compare
       the mean and the standard deviation of the estimates across datasets.
       Explain the behavior of the mean with the equations of (a).
   :exlabel:`ex_mdl-distributions-poisson-regression`

1. **Conjugate priors from natural parameters.** ● For a scalar natural
   parameter $\eta$, the general conjugate prior of an exponential-family
   likelihood is

   $$p(\eta)\propto\exp\bigl(\nu\,\eta-\kappa\,A(\eta)\bigr),$$

   and $n$ observations update it by $\nu\to\nu+\sum_i T(x_i)$ and
   $\kappa\to\kappa+n$.
    1. For the Poisson, $\eta=\log\lambda$ and $A(\eta)=e^\eta$. Change
       variables from $\eta$ to $\lambda$, including the Jacobian, and show
       that the prior on $\lambda$ is a Gamma distribution. Identify its shape
       and rate, and check that the update reproduces the Gamma posterior
       of :numref:`subsec_mdl-other-conjugate-pairs`.
    1. For the Bernoulli, $\eta=\log\frac{p}{1-p}$ and
       $A(\eta)=\log(1+e^\eta)$. Show that the prior on $p$ is a Beta
       distribution, identify its parameters, and find the values of
       $(\nu,\kappa)$ for which it is normalizable.
    1. Take the flat prior on the logit, $\nu=\kappa=0$. Which density on $p$
       does it induce, and is that density normalizable? After $x$ heads in
       $n$ flips, when is the posterior normalizable, and what is its mean?
       Compare it with the mean of the posterior :eqref:`eq_mdl-beta_posterior`
       under the uniform prior $\mathrm{Beta}(1,1)$ on $p$.
   :exlabel:`ex_mdl-distributions-conjugate-recipe`

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §27.3]{.kicker}

The exponential family<br>**one form, maximum entropy, the moment property, conjugate priors**.
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[The exponential family]{.dtitle}

[one form, maximum entropy, the moment property]{.dsub}
:::
:::

::: {.slide title="One shared form"}
[Unification]{.kicker}

$$p(\mathbf x\mid\boldsymbol\eta) = h(\mathbf x)\,
\exp\!\bigl(\boldsymbol\eta^\top T(\mathbf x) - A(\boldsymbol\eta)\bigr).$$

Base measure $h$, **natural parameters** $\boldsymbol\eta$, sufficient
statistics $T$, log-partition $A$. Bernoulli ($\eta=\operatorname{logit}p$,
$A=\operatorname{softplus}$), Poisson ($\eta=\log\lambda$), and Gaussian
all fit.

::: {.d2l-note}
Two exclusions, two reasons: the **uniforms** stay outside because their
support moves with the parameters; **Cauchy and Student-$t$** do not admit a
fixed finite-dimensional sufficient statistic for their usual unknown
location-and-scale families. They therefore lack the standard
finite-dimensional conjugate update, and their negative log-likelihoods are
not generally convex.
:::
:::

::: {.slide title="Where the form comes from"}
[Unification]{.kicker}

Maximize entropy $H_h[p]$ subject to fixed averages
$\mathbb E[T(\mathbf x)] = \boldsymbol\tau$. The Lagrange multipliers
*are* the natural parameters, and the maximizer is exactly
$p\propto h\,e^{\boldsymbol\eta^\top T}$.

::: {.d2l-note .rule}
The exponential family is the **least-committal** family consistent with a
chosen set of expected statistics.
:::
:::

::: {.slide title="The moment property"}
[Unification]{.kicker}

Differentiating the log-partition recovers the moments:

$$\nabla A(\boldsymbol\eta) = \mathbb E[T(\mathbf x)], \qquad
\nabla^2 A(\boldsymbol\eta) = \operatorname{Cov}(T) \succeq 0.$$

So $A$ is **convex** and the MLE equation is moment matching,
$\mathbb E[T] = \bar T$. Autograd confirms $dA/d\eta=\sigma(\eta)$ for the
Bernoulli:

~~~python
dA = (A(eta + eps) - A(eta - eps)) / (2 * eps)   # numerical dA/deta
~~~

@!distributions-exp-family
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Conjugate priors]{.dtitle}

[Beta · Gamma · Dirichlet: updating is counting]{.dsub}
:::
:::

::: {.slide title="Beta–Bernoulli: pseudo-counts"}
[Priors]{.kicker}

::: {.cols .vc}
::: {.col}
A $\text{Beta}(\alpha,\beta)$ prior times a Bernoulli likelihood with $x$
heads in $n$ flips gives a $\text{Beta}(\alpha+x,\beta+n-x)$ posterior;
$\alpha,\beta$ act as **phantom** heads and tails:

$$\mathbb E[\theta\mid X] = \frac{\alpha+x}{\alpha+\beta+n}.$$

@!distributions-conjugate

::: {.d2l-note}
$\alpha=\beta=1$ recovers Laplace's rule of succession $(x+1)/(n+2)$.
:::
:::

::: {.col .fig}
@fig:mdl-prob-beta-posterior
:::
:::
:::

::: {.slide title="The rest of the tier"}
[Priors]{.kicker}

::: {.cols}
::: {.col}
**Gamma → Poisson.** Pseudo-events over a pseudo-window; marginalizing
the rate gives the over-dispersed negative binomial.
:::

::: {.col}
**Dirichlet → Categorical.** A multivariate Beta: one pseudo-count per
class on the simplex.
:::
:::

::: {.d2l-note .rule}
**General fact.** Every exponential-family likelihood has a conjugate
prior of the same form; its hyperparameters are pseudo-data
$(\boldsymbol\nu,\kappa)$.
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

- The exponential family unifies them; $\nabla A=\mathbb E[T]$, and its NLL is convex in natural parameters.
- The usual conjugate families (Beta / Gamma / Dirichlet) update by adding sufficient statistics.
- The named shapes are the vocabulary; maximum likelihood is the grammar (next).
:::
