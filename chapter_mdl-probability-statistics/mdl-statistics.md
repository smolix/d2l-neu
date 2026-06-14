# Statistics
:label:`sec_mdl-statistics`

A trained model is only ever fit to a *finite* sample, so every quantity we read off it---an accuracy, a learned weight, an estimated mean---is a guess computed from random data and would come out differently on a fresh draw. Statistics is the discipline that quantifies that randomness: it tells us how far a guess typically sits from the truth, when an apparent improvement is real rather than noise, and how confident we are entitled to be. This section develops the three ideas a deep-learning practitioner reaches for most often. We define an *estimator* and the two ways it can be wrong---*bias* and *variance*---and prove the decomposition that ties them together; this single identity is the same U-curve that governs under- and over-fitting in :numref:`sec_generalization_basics`, so it is worth deriving carefully. We then turn to *hypothesis testing*, the framework behind A/B tests and benchmark comparisons, and close with *confidence intervals*, which attach a notion of uncertainty to a point estimate. Throughout we take the true parameter $\theta$ to be a scalar; the vector case is identical with sums of squares replaced by squared norms.

We first load the per-framework library so the computations below have `d2l` and the tensor library in scope, plus plain NumPy as `onp` for the label-shuffling in the permutation test and the resampling in the bootstrap, both framework-agnostic. The estimator simulations are likewise framework-agnostic apart from the random-number call, so the worked cells branch only where they must.

```{.python .input #statistics-imports}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
import numpy as onp  # plain NumPy for framework-agnostic resampling below
```

```{.python .input #statistics-imports}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import numpy as onp  # plain NumPy for framework-agnostic resampling below
```

```{.python .input #statistics-imports}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import numpy as onp  # plain NumPy for framework-agnostic resampling below
```

```{.python .input #statistics-imports}
#@tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import numpy as onp  # plain NumPy for framework-agnostic resampling below
```

## Estimators and Their Quality

### Estimators

An *estimator* is a recipe that turns data into a guess for an unknown parameter. Formally, given samples $x_1,\ldots,x_n$ drawn from a distribution governed by a parameter $\theta$, an estimator is a function

$$
\hat\theta_n = \hat f(x_1,\ldots,x_n)
$$

that we hope lands near $\theta$. We have met estimators already: in :numref:`sec_mdl-maximum_likelihood` the maximum-likelihood estimate of a Bernoulli probability was the fraction of observed ones, and the maximum-likelihood estimate of a Gaussian mean was the sample average. The key fact is that $\hat\theta_n$ is itself a *random variable*: it depends on the random sample, so it would come out differently on a fresh dataset. Asking whether an estimator is *good* is therefore asking about the distribution of $\hat\theta_n$ over repeated datasets---its *sampling distribution*---and that distribution has two features that matter, its center and its spread.

### Bias and Variance

The first feature is the *center*. The **bias** of $\hat\theta_n$ measures the systematic gap between where the estimator centers and the truth,

$$
\operatorname{Bias}(\hat\theta_n) = \mathbb{E}[\hat\theta_n] - \theta ,
$$
:eqlabel:`eq_mdl-bias`

the expectation taken over the random sample. When $\operatorname{Bias}(\hat\theta_n)=0$ for every $\theta$ we call $\hat\theta_n$ *unbiased*: it is right *on average*, even though any single estimate misses. Bias is the error that does not wash out by collecting more data of the same kind---it is baked into the recipe.

The second feature is the *spread*. The **variance** measures how much the estimator fluctuates around its own center, with the **standard error** its square root,

$$
\operatorname{Var}(\hat\theta_n) = \mathbb{E}\!\left[(\hat\theta_n - \mathbb{E}[\hat\theta_n])^2\right],
\qquad
\operatorname{se}(\hat\theta_n) = \sqrt{\operatorname{Var}(\hat\theta_n)} .
$$
:eqlabel:`eq_mdl-var_est`

Note carefully that variance is measured against $\mathbb{E}[\hat\theta_n]$, *not* against the true $\theta$: it captures the noise in the estimator, not its accuracy. :numref:`fig_mdl-sampling-distribution` makes the two features visible by drawing the sampling distribution for two estimators of the same $\theta$. Bias is the offset of the distribution's center from $\theta$; variance is its width.

![The sampling distribution of an estimator, drawn over many datasets. Bias is the offset of the center $\mathbb{E}\left(\hat\theta\right)$ from the true $\theta$; variance is the spread. Left: low bias, high variance. Right: high bias, low variance. A good estimator wants both small, but usually one is traded for the other.](../img/mdl-prob-sampling-distribution.svg)
:label:`fig_mdl-sampling-distribution`

### Consistency and Efficiency

Bias and variance describe an estimator at a *fixed* sample size. Two further notions describe how it behaves as data accumulates. An estimator is *asymptotically unbiased* if its bias vanishes in the limit, $\lim_{n\to\infty}\operatorname{Bias}(\hat\theta_n)=0$; many estimators used in practice are biased at finite $n$ but asymptotically unbiased, which is usually good enough. A stronger and more useful guarantee is *consistency*: $\hat\theta_n$ is consistent if it *converges in probability* to $\theta$,

$$
\hat\theta_n \xrightarrow{P} \theta,
\qquad\textrm{i.e.}\qquad
P\bigl(|\hat\theta_n-\theta|>\varepsilon\bigr)\to 0 \quad\textrm{for every } \varepsilon>0 .
$$

Consistency is the formal content of the slogan "more data gets us arbitrarily close to the truth." Its prototype is the *law of large numbers* (LLN), which says exactly this for the sample mean: as $n\to\infty$ the average $\bar x$ converges in probability to the population mean. A clean sufficient condition for consistency in general is that *both* the bias and the variance tend to zero, since then the whole sampling distribution collapses onto $\theta$; we will see this happen explicitly for the sample mean below. (The two limits are independent: an estimator can be asymptotically unbiased yet inconsistent if its variance does not vanish, and vice versa.)

Finally, among *unbiased* estimators we prefer the one that fluctuates least, and we call it *efficient*: efficiency ranks unbiased estimators by their variance, the smaller the better. There is a hard floor here---the Cramér--Rao bound puts a lower limit on the variance of any unbiased estimator---and an estimator that attains it is as good as unbiased estimation can be. We will not need the bound itself, only the idea it formalizes: once unbiasedness is secured, the remaining game is to minimize variance, which is exactly the second half of the decomposition we turn to next.

## The Bias-Variance Decomposition

We now have two distinct ways an estimator can be wrong---a systematic offset (bias) and random fluctuation (variance)---and a single number that ought to combine them: the *mean squared error*. It is worth pausing on the remarkable fact that the MSE is *exactly* the sum of these two contributions, with no cross term. This decomposition is the centerpiece of the section.

### Mean Squared Error and the Decomposition

The simplest summary of how far an estimator lands from the truth is the **mean squared error**,

$$
\operatorname{MSE}(\hat\theta_n) = \mathbb{E}\!\left[(\hat\theta_n-\theta)^2\right] .
$$
:eqlabel:`eq_mdl-mse_est`

It is always non-negative, and the smaller it is the closer $\hat\theta_n$ sits to $\theta$ on average. If you have read :numref:`sec_linear_regression` you will recognize it as the squared-error loss, now applied to an estimator rather than a prediction.

**Proposition (bias-variance decomposition).** *For any estimator $\hat\theta_n$ of a fixed parameter $\theta$,*

$$
\operatorname{MSE}(\hat\theta_n) = \operatorname{Bias}(\hat\theta_n)^2 + \operatorname{Var}(\hat\theta_n) .
$$
:eqlabel:`eq_mdl-bias-variance`

**Proof.** Abbreviate $\mu = \mathbb{E}[\hat\theta_n]$, the center of the estimator, and add and subtract it inside the square:

$$
\operatorname{MSE}(\hat\theta_n)
 = \mathbb{E}\!\left[(\hat\theta_n - \theta)^2\right]
 = \mathbb{E}\!\left[\bigl((\hat\theta_n - \mu) + (\mu - \theta)\bigr)^2\right].
$$

Expanding the square gives three terms. The first is $\mathbb{E}[(\hat\theta_n-\mu)^2]=\operatorname{Var}(\hat\theta_n)$; the last is $(\mu-\theta)^2=\operatorname{Bias}(\hat\theta_n)^2$, a constant. The middle, cross term *vanishes*, because $\mu-\theta$ is a constant and $\hat\theta_n-\mu$ has mean zero by the definition of $\mu$:

$$
2\,(\mu-\theta)\,\mathbb{E}[\hat\theta_n - \mu]
 = 2\,(\mu-\theta)\,(\mu - \mu) = 0 .
$$

What remains is $\operatorname{Var}(\hat\theta_n)+\operatorname{Bias}(\hat\theta_n)^2$. $\blacksquare$

The vanishing cross term is the whole story: because the deviation from the center has mean zero, the systematic part and the fluctuating part of the error never interfere, and the squared error splits cleanly into the two pieces of :numref:`fig_mdl-sampling-distribution`. One immediate payoff is the consistency criterion promised above: if both $\operatorname{Bias}(\hat\theta_n)\to0$ and $\operatorname{Var}(\hat\theta_n)\to0$, then :eqref:`eq_mdl-bias-variance` forces $\operatorname{MSE}(\hat\theta_n)\to0$, which implies $\hat\theta_n\xrightarrow{P}\theta$: applying Markov's inequality :eqref:`eq_mdl-markov` from :numref:`sec_mdl-random_variables` to the nonnegative random variable $(\hat\theta_n-\theta)^2$ gives $P(|\hat\theta_n-\theta|>\varepsilon) \le \operatorname{MSE}(\hat\theta_n)/\varepsilon^2 \to 0$. (Chebyshev's inequality is the same bound centered at the mean; since $\hat\theta_n$ may be biased, we aim it at $\theta$ directly. The bound is also quantitative---the miss probability decays at least as fast as the MSE---and sharper, exponentially decaying rates come from concentration inequalities when the data are bounded or sub-Gaussian.)

### The Trade-off and Generalization

Identity :eqref:`eq_mdl-bias-variance` is more than bookkeeping; it explains the central tension of model fitting. Read $\hat\theta_n$ as a *fitted model* and $\theta$ as the function we wish it had learned. A model too simple to capture the signal---a straight line for a curved relationship---has large bias: it misses systematically no matter how much data we feed it. A model too flexible chases the noise in the particular training set, so it has large variance: a fresh dataset would fit it to a wildly different shape. These are the familiar failures of *underfitting* (high bias) and *overfitting* (high variance) from :numref:`sec_generalization_basics`.

As we dial up model complexity, the squared bias falls while the variance rises, and their sum traces a U with a minimum at the sweet spot, shown in :numref:`fig_mdl-bias-variance-u-curve`. For prediction the accounting acquires one more term: the expected test error is $\operatorname{Bias}^2 + \operatorname{Var} + \sigma^2$, where $\sigma^2$ is the *irreducible noise* in the labels that no model, however good, can remove. Since that floor is a constant, it shifts the U upward without moving its sweet spot, and the decomposition and the generalization U-curve are essentially the same picture. This also explains *why regularization helps*: techniques like weight decay deliberately add a little bias in exchange for a large reduction in variance, sliding leftward on the curve to a lower total error. One caveat is in order for deep learning: heavily overparameterized models need not obey the textbook U---past the point where a model can interpolate its training data, test error often *descends again* as capacity grows, the *double descent* phenomenon :cite:`Belkin.Hsu.Ma.ea.2019`---which is why :numref:`sec_generalization_basics` cautions that larger architectures with more parameters can generalize *better*. The decomposition itself remains exactly true; what breaks down is the intuition that variance must keep rising with parameter count.

![As model complexity grows, squared bias falls and variance rises; their sum, the MSE (test error), is a U-curve with a minimum at the sweet spot.](../img/mdl-prob-bias-variance-u-curve.svg)
:label:`fig_mdl-bias-variance-u-curve`

### The Decomposition in Code

The decomposition is an exact algebraic identity, so it should hold to numerical precision on a concrete example. We first define bias and MSE as the formulas :eqref:`eq_mdl-bias` and :eqref:`eq_mdl-mse_est` say---averages over a collection of estimates, against the true parameter.

```{.python .input #statistics-estimator-metrics}
#@tab mxnet
def stat_bias(true_theta, est_theta):  # E[theta_hat] - theta
    return np.mean(est_theta) - true_theta

def mse(est_theta, true_theta):        # E[(theta_hat - theta)^2]
    return np.mean(np.square(est_theta - true_theta))
```

```{.python .input #statistics-estimator-metrics}
#@tab pytorch
def stat_bias(true_theta, est_theta):  # E[theta_hat] - theta
    return torch.mean(est_theta) - true_theta

def mse(est_theta, true_theta):        # E[(theta_hat - theta)^2]
    return torch.mean(torch.square(est_theta - true_theta))
```

```{.python .input #statistics-estimator-metrics}
#@tab tensorflow
def stat_bias(true_theta, est_theta):  # E[theta_hat] - theta
    return tf.reduce_mean(est_theta) - true_theta

def mse(est_theta, true_theta):        # E[(theta_hat - theta)^2]
    return tf.reduce_mean(tf.square(est_theta - true_theta))
```

```{.python .input #statistics-estimator-metrics}
#@tab jax
def stat_bias(true_theta, est_theta):  # E[theta_hat] - theta
    return jnp.mean(est_theta) - true_theta

def mse(est_theta, true_theta):        # E[(theta_hat - theta)^2]
    return jnp.mean(jnp.square(est_theta - true_theta))
```

To exercise these we need the *sampling distribution* itself, not a single dataset: we draw many independent datasets from $\mathcal{N}(\theta,\sigma^2)$, compute the sample mean on each, and collect the resulting estimates. Their spread is the variance and their center the bias.

```{.python .input #statistics-sampling-distribution}
#@tab mxnet
theta_true, sigma = 1.0, 4.0
num_datasets, n = 10000, 30  # 10k datasets, each of n=30 points
samples = np.random.normal(theta_true, sigma, (num_datasets, n))
theta_hats = samples.mean(axis=1)  # one sample-mean estimate per dataset
```

```{.python .input #statistics-sampling-distribution}
#@tab pytorch
theta_true, sigma = 1.0, 4.0
num_datasets, n = 10000, 30  # 10k datasets, each of n=30 points
samples = torch.normal(theta_true, sigma, size=(num_datasets, n))
theta_hats = samples.mean(axis=1)  # one sample-mean estimate per dataset
```

```{.python .input #statistics-sampling-distribution}
#@tab tensorflow
theta_true, sigma = 1.0, 4.0
num_datasets, n = 10000, 30  # 10k datasets, each of n=30 points
samples = tf.random.normal((num_datasets, n), theta_true, sigma)
theta_hats = tf.reduce_mean(samples, axis=1)  # one estimate per dataset
```

```{.python .input #statistics-sampling-distribution}
#@tab jax
theta_true, sigma = 1.0, 4.0
num_datasets, n = 10000, 30  # 10k datasets, each of n=30 points
key = jax.random.PRNGKey(0)
samples = jax.random.normal(key, (num_datasets, n)) * sigma + theta_true
theta_hats = samples.mean(axis=1)  # one sample-mean estimate per dataset
```

Now we read the decomposition off the empirical sampling distribution. The MSE of the estimates around the true $\theta$ should match the squared bias plus the variance of the estimates around their own mean---the two sides of :eqref:`eq_mdl-bias-variance`. One detail matters for exactness: the identity is a statement about expectations under a *single* distribution, so every term must be computed under the same one---here the empirical distribution of our $10{,}000$ estimates, whose expectations are plain averages. That dictates the *plug-in* variance that divides by the number of estimates (`ddof=0`, the default in most libraries), not the unbiased $n-1$ variant we meet in the next subsection: with the plug-in choice, the proof's algebra goes through verbatim for the empirical averages, and the two sides agree to floating-point round-off.

```{.python .input #statistics-verify-decomposition}
#@tab mxnet
bias = stat_bias(theta_true, theta_hats)
# Default ddof=0: the plug-in variance of the estimates, which makes the
# identity exact for empirical averages (the n-1 variant is the next subsection)
var = np.var(theta_hats)
mse(theta_hats, theta_true), var + np.square(bias)
```

```{.python .input #statistics-verify-decomposition}
#@tab pytorch
bias = stat_bias(theta_true, theta_hats)
var = theta_hats.var(unbiased=False)  # plug-in variance (ddof=0): exact identity
mse(theta_hats, theta_true), var + torch.square(bias)
```

```{.python .input #statistics-verify-decomposition}
#@tab tensorflow
bias = stat_bias(theta_true, theta_hats)
var = tf.math.reduce_variance(theta_hats)  # plug-in variance: divides by n
mse(theta_hats, theta_true), var + tf.square(bias)
```

```{.python .input #statistics-verify-decomposition}
#@tab jax
bias = stat_bias(theta_true, theta_hats)
var = jnp.var(theta_hats)  # default ddof=0: the plug-in variance, exact identity
mse(theta_hats, theta_true), var + jnp.square(bias)
```

The two numbers agree to floating-point round-off---the identity is exact, not merely approximate---and both are close to the theoretical value. For the sample mean of $\mathcal{N}(\theta,\sigma^2)$ the bias is exactly zero (the average of unbiased draws is unbiased) and the variance is $\sigma^2/n$ (the variance-of-a-sum result from :numref:`sec_mdl-random_variables`), so $\operatorname{MSE}=\sigma^2/n = 16/30 \approx 0.53$. Because *both* the bias ($0$) and the variance ($\sigma^2/n\to0$) vanish as $n\to\infty$, the sample mean is consistent---this is the law of large numbers, and exactly the criterion from the decomposition above.

### Why the Unbiased Variance Divides by $n-1$

The sample mean was unbiased for free. The sample *variance* is more delicate, and it exposes a subtlety that every framework's `std` function encodes in a `ddof` flag. Given samples $x_1,\ldots,x_n$ with sample mean $\bar x=\frac1n\sum_i x_i$, the natural estimator of the population variance $\sigma^2$ would average the squared deviations,

$$
s_0^2 = \frac1n\sum_{i=1}^n (x_i-\bar x)^2 .
$$

This is *biased*: it systematically underestimates $\sigma^2$, because the deviations are measured from $\bar x$---the point that *minimizes* the sum of squared deviations for this particular sample---rather than from the unknown true mean $\mu$. The fix is to divide by $n-1$ instead of $n$, and the factor is exactly what unbiasedness requires.

**Proposition (unbiased sample variance).** *For i.i.d. samples with variance $\sigma^2$,*

$$
s^2 = \frac{1}{n-1}\sum_{i=1}^n (x_i-\bar x)^2
\qquad\textrm{satisfies}\qquad
\mathbb{E}[s^2] = \sigma^2 .
$$
:eqlabel:`eq_mdl-unbiased-var`

**Proof.** Center the data at the true mean $\mu$ by writing $x_i-\bar x = (x_i-\mu)-(\bar x-\mu)$, and expand the sum of squared deviations:

$$
\sum_{i=1}^n (x_i-\bar x)^2
 = \sum_{i=1}^n (x_i-\mu)^2 - n\,(\bar x-\mu)^2 ,
$$

where the cross term collapsed because $\sum_i (x_i-\mu) = n(\bar x-\mu)$. Now take expectations. Each $\mathbb{E}[(x_i-\mu)^2]=\sigma^2$, so the first sum has expectation $n\sigma^2$. The second uses the variance of the sample mean, $\mathbb{E}[(\bar x-\mu)^2]=\operatorname{Var}(\bar x)=\sigma^2/n$, so that term has expectation $n\cdot\sigma^2/n=\sigma^2$. Hence

$$
\mathbb{E}\!\left[\sum_{i=1}^n (x_i-\bar x)^2\right] = n\sigma^2 - \sigma^2 = (n-1)\,\sigma^2 .
$$

Dividing by $n-1$ gives $\mathbb{E}[s^2]=\sigma^2$. $\blacksquare$

The intuition is *degrees of freedom*: estimating $\bar x$ from the same data consumes one degree of freedom, so only $n-1$ of the deviations are free to vary, and dividing by $n-1$ rather than $n$ corrects for it exactly. (As $n\to\infty$ the two estimators agree, so $s_0^2$ is biased but asymptotically unbiased and consistent.) We can watch the bias appear and the correction remove it by estimating both variances over many datasets and averaging.

```{.python .input #statistics-unbiased-variance}
#@tab mxnet
n = 3  # small n makes the 1/n bias glaring; the gap shrinks like 1/n
data = np.random.normal(0, 2, (100000, n))  # sigma^2 = 4
dev2 = np.square(data - data.mean(axis=1, keepdims=True)).sum(axis=1)
print('true variance    = 4')
print(f'E[divide by n]   = {float((dev2 / n).mean()):.3f}  (biased)')
print(f'E[divide by n-1] = {float((dev2 / (n - 1)).mean()):.3f}  (unbiased)')
```

```{.python .input #statistics-unbiased-variance}
#@tab pytorch
n = 3  # small n makes the 1/n bias glaring; the gap shrinks like 1/n
data = torch.normal(0, 2, size=(100000, n))  # sigma^2 = 4
dev2 = torch.square(data - data.mean(axis=1, keepdim=True)).sum(axis=1)
print('true variance    = 4')
print(f'E[divide by n]   = {(dev2 / n).mean():.3f}  (biased)')
print(f'E[divide by n-1] = {(dev2 / (n - 1)).mean():.3f}  (unbiased)')
```

```{.python .input #statistics-unbiased-variance}
#@tab tensorflow
n = 3  # small n makes the 1/n bias glaring; the gap shrinks like 1/n
data = tf.random.normal((100000, n), 0, 2)  # sigma^2 = 4
dev2 = tf.reduce_sum(tf.square(
    data - tf.reduce_mean(data, axis=1, keepdims=True)), axis=1)
print('true variance    = 4')
print(f'E[divide by n]   = {tf.reduce_mean(dev2 / n):.3f}  (biased)')
print(f'E[divide by n-1] = {tf.reduce_mean(dev2 / (n - 1)):.3f}  (unbiased)')
```

```{.python .input #statistics-unbiased-variance}
#@tab jax
n = 3  # small n makes the 1/n bias glaring; the gap shrinks like 1/n
data = jax.random.normal(jax.random.PRNGKey(1), (100000, n)) * 2  # sigma^2 = 4
dev2 = jnp.square(data - data.mean(axis=1, keepdims=True)).sum(axis=1)
print('true variance    = 4')
print(f'E[divide by n]   = {(dev2 / n).mean():.3f}  (biased)')
print(f'E[divide by n-1] = {(dev2 / (n - 1)).mean():.3f}  (unbiased)')
```

With $n=3$ the biased estimator averages near $\tfrac{n-1}{n}\sigma^2 = \tfrac23\cdot 4 \approx 2.67$, while dividing by $n-1$ recovers $4$, confirming :eqref:`eq_mdl-unbiased-var`. This is precisely why `numpy` and friends expose `ddof` ("delta degrees of freedom"): `ddof=1` divides by $n-1$ for the unbiased estimate, `ddof=0` divides by $n$.

## Hypothesis Testing

The bias-variance picture asks *how good* a single estimate is. Hypothesis testing asks a different question that dominates experimental practice: given two estimates---a baseline and a new model, a control group and a treatment---is the observed difference *real*, or could it be a fluke of the particular sample? This is the framework behind A/B testing and behind claims that one architecture beats another on a benchmark.

### The Setup: Null, Alternative, and Two Kinds of Error

A *hypothesis test* weighs evidence against a default claim. The **null hypothesis** $H_0$ is that default---typically "there is no effect," e.g. the new model is no better than the baseline---and the **alternative** $H_A$ is the effect we hope to detect: sometimes the null's outright negation, but often one-sided or otherwise composite, e.g. "the new model is *better*." The asymmetry is deliberate: we never *prove* $H_0$; we either gather enough evidence to *reject* it in favor of $H_A$, or we fail to, much as a court returns "guilty" or "not guilty" rather than "innocent."

Because the data are random, the decision can go wrong in two ways. A **type I error** (false positive) is rejecting $H_0$ when it is in fact true---declaring an effect that is not there. A **type II error** (false negative) is failing to reject $H_0$ when it is in fact false---missing a real effect. Their rates have standard names, the significance level $\alpha$ and $\beta$,

$$
\alpha = P(\textrm{reject } H_0 \mid H_0 \textrm{ true}),
\qquad
\beta = P(\textrm{fail to reject } H_0 \mid H_0 \textrm{ false}),
$$

and the four possible outcomes arrange into the $2\times2$ decision matrix of :numref:`fig_mdl-type-i-ii-matrix`: rows are whether $H_0$ is true or false, columns are our decision. The diagonal cells are correct; the off-diagonal cells are the two errors. This picture is the surest way to keep $\alpha$ and $\beta$ from getting swapped.

![The $2\times 2$ hypothesis-test decision matrix: correct decisions on the diagonal, the type I error (rate $\alpha$) and type II error (rate $\beta$) off the diagonal, and the power $1-\beta$ in the correct-rejection cell.](../img/mdl-prob-type-i-ii-matrix.svg)
:label:`fig_mdl-type-i-ii-matrix`

### Significance and Power

We *choose* the type I error rate up front: the **significance level** $\alpha$ is the risk of a false positive we are willing to tolerate, conventionally $\alpha=0.05$. The complement $1-\alpha$ is the *confidence level*; we reserve that name for the confidence intervals below and are careful not to call $1-\alpha$ the significance. The bottom-right cell of the matrix is the quantity we want to be large: the **statistical power**

$$
1 - \beta = P(\textrm{reject } H_0 \mid H_0 \textrm{ false})
$$

is the probability the test *detects* a real effect. A test with $\alpha=0.05$ but power $0.2$ rejects a true null only $5\%$ of the time yet still misses $80\%$ of genuine effects---underpowered, and worthless for confirming improvements. A common target is $1-\beta=0.8$.

Power is what determines how much data we need. The probability of detecting an effect grows with both the *effect size* (how false $H_0$ really is) and the sample size, and for the standard tests the required $n$ scales like $1/(\textrm{effect size})^2$. As an indicative one-sample two-sided $z$-test at $\alpha=0.05$ and power $0.8$: testing $H_0\!:\mu=0$ on unit-variance Gaussian data whose true mean is $1$ (a large effect) needs only about $8$ samples, whereas the same test against a true mean of $0.01$ (a tiny effect) needs on the order of $80{,}000$. This is why marginal benchmark gains demand enormous test sets to confirm. :numref:`fig_mdl-power` traces the whole family of *power curves*: for each effect size $\delta$, the probability of detection climbs from $\alpha$ (a false-positive rate is all a test delivers at $\delta=0$) toward $1$ as $n$ grows, crossing the conventional target $0.8$ at a sample size proportional to $1/\delta^2$.

![Power of the one-sample two-sided $z$-test at $\alpha=0.05$ as a function of the sample size $n$, one curve per effect size $\delta$. Every curve starts near $\alpha$ and climbs toward $1$; the dashed line marks the conventional target $0.8$, reached at a sample size that scales like $1/\delta^2$---about $8$ samples for $\delta=1$ but nearly $80{,}000$ for $\delta=0.01$.](../img/mdl-prob-power.svg)
:label:`fig_mdl-power`

### Test Statistics, $p$-values, and Significance

To run a test we compress the data into a single **test statistic** $T(x)$---a scalar chosen so that extreme values are unlikely under $H_0$. The mean difference between two groups is a natural choice. Crucially, under $H_0$ the statistic has a known (often approximately Gaussian) *null distribution*, and that is what lets us judge whether an observed value is surprising.

The verdict is delivered by the **$p$-value**: the probability, *computed under $H_0$*, of seeing a statistic at least as extreme as the one we observed. For a two-sided test (the common case, where a deviation in either direction counts as evidence),

$$
p\textrm{-value} = P_{H_0}\bigl(|T(X)| \ge |T(x)|\bigr),
$$

valid when the null distribution is symmetric about $0$; in general the two-sided $p$-value is $2\,\min\{P_{H_0}(T\ge t),\, P_{H_0}(T\le t)\}$, which doubles the smaller tail. The one-sided version uses a single tail. We reject $H_0$ when $p \le \alpha$. Geometrically, the rejection region is the set of statistic values whose $p$-value falls below $\alpha$; :numref:`fig_mdl-statistical_significance` shows it for a Gaussian null at $\alpha=0.05$ as the two tails beyond the critical values $\pm 1.96$---the two-sided $z$-test's rejection region $|T|\ge1.96$---together holding $5\%$ of the probability. A statistic landing in those tails would be very unlikely if $H_0$ held, so we reject.

![Statistical significance: under the null distribution, the central region holds probability $1-\alpha$ and the two tails together hold $\alpha$. A test statistic in the tails is unlikely under $H_0$, so we reject it.](../img/mdl-prob-significance.svg)
:label:`fig_mdl-statistical_significance`

A persistent warning is in order, because the $p$-value is among the most misread numbers in science :cite:`Wasserstein.Lazar.2016`. It is $P(\textrm{data this extreme}\mid H_0)$, a statement about the data *given* the null---*not* $P(H_0\mid\textrm{data})$, the probability the null is true, which would require a prior and Bayes' rule. A large $p$-value does *not* confirm $H_0$; it means only that we failed to detect an effect, possibly because the test was underpowered.

A subtler trap is *multiple testing*. The $\alpha=0.05$ guarantee holds for a *single* pre-specified test; run $m$ of them under a true null---sweeping hyperparameters, comparing across benchmarks, retrying until something "works"---and the chance of at least one spurious win is $1-(1-\alpha)^m$, which already exceeds $0.4$ at $m=10$. Reporting only the test that cleared $p\le\alpha$ is *$p$-hacking*, and it is how noise gets published as a result. The simplest guard is the *Bonferroni correction*: to hold the family-wide false-positive rate at $\alpha$, test each of the $m$ hypotheses at the stricter level $\alpha/m$. When $m$ runs into the hundreds or thousands---a hyperparameter sweep, a screen of model variants---Bonferroni grows hopelessly conservative, and large-scale practice instead controls the *false discovery rate*, the expected fraction of rejections that are false, via the Benjamini--Hochberg procedure :cite:`Benjamini.Hochberg.1995`.

To summarize, a hypothesis test proceeds in five steps:

1. State $H_0$ and $H_A$.
2. Fix the significance level $\alpha$ and a target power $1-\beta$ (which, with the expected effect size, sets the sample size).
3. Collect the data.
4. Compute the test statistic and its $p$-value under $H_0$.
5. Reject $H_0$ if $p \le \alpha$; otherwise fail to reject.

### A Worked Test: Comparing Two Models

Let us run the recipe once, on the comparison practitioners face most often: is model B really better than model A, or did it just draw lucky seeds? We simulate per-seed test accuracies for the two models---twenty training runs each, with a true gap of $0.8\%$ buried in seed-to-seed noise of comparable size. Step 1: $H_0$ is that the two models are equally good, i.e. the two accuracy samples come from the same distribution, and $H_A$ is that they differ; step 2: $\alpha=0.05$. The test statistic is the gap between the mean accuracies. Rather than assume a Gaussian null distribution, we use a **permutation test**, which manufactures the null distribution from the data itself: if $H_0$ holds, the labels "A" and "B" carry no information---the $40$ numbers are *exchangeable*---so shuffling the labels and recomputing the gap, many times over, shows exactly how large a gap arises by pure chance. The two-sided $p$-value is the fraction of shuffles producing a gap at least as extreme as the observed one (counting the observed labeling itself among them, which keeps the estimate valid and never exactly zero).

```{.python .input #statistics-permutation-test}
rng = onp.random.default_rng(1)
num_seeds = 20                               # 20 training runs per model
acc_a = rng.normal(0.850, 0.010, num_seeds)  # per-seed accuracy, model A
acc_b = rng.normal(0.858, 0.010, num_seeds)  # model B: a real +0.008 gap
observed = acc_b.mean() - acc_a.mean()       # test statistic: gap in means

pooled = onp.concatenate([acc_a, acc_b])     # under H_0 the labels are arbitrary
B = 10000
gaps = onp.empty(B)
for b in range(B):
    perm = rng.permutation(pooled)           # shuffle the model labels
    gaps[b] = perm[num_seeds:].mean() - perm[:num_seeds].mean()
p_value = (1 + (onp.abs(gaps) >= abs(observed)).sum()) / (B + 1)  # two-sided
print(f'observed gap        = {observed:.4f}')
print(f'permutation p-value = {p_value:.4f}')
```

The observed gap is $0.0073$---model B looks better by about three quarters of an accuracy point---and only about $2\%$ of label shuffles produce a gap that large, so $p \approx 0.02 \le \alpha = 0.05$ and we reject $H_0$: the improvement is unlikely to be a fluke. Note how close the call is, though. A *real* $0.8\%$ improvement, measured over twenty seeds, only just clears the bar---the power discussion above in action; with five seeds (a common budget) the same gap would usually go undetected. The permutation test assumes no Gaussian shape and works for any statistic we care to compute on the two groups; the same resample-and-recompute idea returns in the bootstrap below.

## Confidence Intervals

A point estimate $\hat\theta$ carries no notion of uncertainty---it is a single number that hides how much it would wobble on fresh data. A **confidence interval** repairs this by reporting an *interval* engineered to contain the true $\theta$ with high probability. The idea is due to Jerzy Neyman :cite:`Neyman.1937`.

### Definition and Interpretation

A confidence interval for $\theta$ is an interval $C_n$ computed from the data such that

$$
P_\theta(C_n \ni \theta) \ge 1 - \alpha \quad \textrm{for all } \theta,
$$
:eqlabel:`eq_mdl-confidence`

where $1-\alpha$ is the *confidence level* or *coverage*. We write $C_n \ni \theta$ rather than $\theta \in C_n$ to stress where the randomness lives: $\theta$ is a *fixed* unknown, and it is the *interval* $C_n$ that is random, redrawn with every dataset.

This makes the correct interpretation subtle, and it is worth getting right. A $95\%$ confidence interval does *not* mean "the true $\theta$ lies in this particular interval with probability $95\%$"---that particular interval is already drawn, and $\theta$ either is or is not inside it. The right reading is *about the procedure*: if we generated many intervals this way, $95\%$ of them would contain $\theta$. The guarantee is on the long-run hit rate of the recipe, not on any single interval. This frequentist guarantee should not be confused with the Bayesian *credible interval*, which treats $\theta$ itself as random with a prior---the machinery behind MAP estimation in :numref:`sec_mdl-maximum_likelihood`---and therefore *can* assert "$\theta$ lies in this particular interval with probability $95\%$," at the price of that probability depending on the chosen prior.

### A Gaussian Example

The classic case is the mean of a Gaussian $\mathcal{N}(\mu,\sigma^2)$ with both parameters unknown. From $n$ samples we form the usual estimators $\hat\mu_n=\frac1n\sum_i x_i$ and the unbiased $\hat\sigma_n^2=\frac1{n-1}\sum_i (x_i-\hat\mu_n)^2$ from :eqref:`eq_mdl-unbiased-var`. The studentized statistic

$$
T = \frac{\hat\mu_n - \mu}{\hat\sigma_n/\sqrt n}
$$

follows *Student's $t$-distribution* on $n-1$ degrees of freedom, which approaches a standard Gaussian as $n\to\infty$. That Gaussian limit is the *central limit theorem* of :numref:`sec_mdl-distributions`: it is what makes the sampling distribution of a mean asymptotically Gaussian and so licenses the $z$-quantile $1.96$. For large $n$, then, $T$ lands in $[-1.96, 1.96]$ with probability $\approx95\%$ (the Gaussian's central $95\%$)---exactly $95\%$ in the Gaussian limit, and slightly *less* at finite $n$, where the exact $t$-distribution has heavier tails and one should use the wider $t$-quantile in its place. Rearranging $-1.96 \le T \le 1.96$ for $\mu$ yields the interval

$$
\left[\hat\mu_n - 1.96\,\frac{\hat\sigma_n}{\sqrt n},\; \hat\mu_n + 1.96\,\frac{\hat\sigma_n}{\sqrt n}\right].
$$
:eqlabel:`eq_mdl-gauss_confidence`

This is one of the most-used formulas in statistics. The half-width $1.96\,\hat\sigma_n/\sqrt n$ shrinks like $1/\sqrt n$---to halve the interval we need *four times* the data. Let us construct one for a standard-normal sample, taking the asymptotic $t_\star=1.96$.

```{.python .input #statistics-confidence-interval}
#@tab mxnet
N = 1000
samples = np.random.normal(loc=0, scale=1, size=(N,))
t_star = 1.96  # asymptotic value; small N would look this up in a t-table
mu_hat = np.mean(samples)
se = samples.std(ddof=1) / np.sqrt(N)  # ddof=1: unbiased sigma_hat
(mu_hat - t_star * se, mu_hat + t_star * se)
```

```{.python .input #statistics-confidence-interval}
#@tab pytorch
N = 1000
samples = torch.normal(0, 1, size=(N,))
t_star = 1.96  # asymptotic value; small N would look this up in a t-table
mu_hat = torch.mean(samples)
se = samples.std(unbiased=True) / N**0.5  # unbiased=True: ddof=1
(mu_hat - t_star * se, mu_hat + t_star * se)
```

```{.python .input #statistics-confidence-interval}
#@tab tensorflow
N = 1000
samples = tf.random.normal((N,), 0, 1)
t_star = 1.96  # asymptotic value; small N would look this up in a t-table
mu_hat = tf.reduce_mean(samples)
n_d = tf.cast(tf.size(samples), samples.dtype)
sigma_hat = tf.sqrt(tf.reduce_sum(tf.square(samples - mu_hat)) / (n_d - 1))
se = sigma_hat / tf.sqrt(n_d)  # ddof=1 done by hand: divide by n-1
(mu_hat - t_star * se, mu_hat + t_star * se)
```

```{.python .input #statistics-confidence-interval}
#@tab jax
N = 1000
# split a fresh subkey rather than reusing the PRNGKey(0) stream from above
key = jax.random.split(jax.random.PRNGKey(0))[1]
samples = jax.random.normal(key, (N,))
t_star = 1.96  # asymptotic value; small N would look this up in a t-table
mu_hat = jnp.mean(samples)
se = jnp.std(samples, ddof=1) / jnp.sqrt(N)  # ddof=1: unbiased sigma_hat
(mu_hat - t_star * se, mu_hat + t_star * se)
```

The interval is narrow and brackets the true mean $0$, as it should roughly $95\%$ of the time. The same $1/\sqrt n$ scaling shows up everywhere uncertainty is reported---error bars on a learning curve, the spread of accuracies across random seeds---and :eqref:`eq_mdl-gauss_confidence` is the formula behind them.

### The Bootstrap

The Gaussian interval :eqref:`eq_mdl-gauss_confidence` rests on a lucky accident: the sample mean has a known sampling distribution, so its standard error has a closed form, $\hat\sigma_n/\sqrt n$. Most quantities a practitioner actually cares about enjoy no such luck. What is the standard error of a *median*, a *correlation*, a model's *test accuracy*, its *AUC* or *BLEU* score? These are complicated functions of the data with no textbook sampling distribution, and writing down their standard error analytically ranges from painful to impossible.

The **bootstrap**, introduced by Bradley Efron :cite:`Efron.1979`, is the strikingly simple idea that escapes this. We never had access to the true distribution $F$ that generated our $n$ data points; if we did, we could simulate the sampling distribution of *any* statistic by drawing fresh datasets from $F$ and recomputing it. The bootstrap's move---the **plug-in principle**---is to substitute the *empirical* distribution $\hat F_n$, which puts mass $1/n$ on each observed point, for the unknown $F$. Drawing $n$ points from $\hat F_n$ is exactly *resampling our own data $n$ times with replacement*. Concretely:

1. From the original sample of size $n$, draw $n$ points **with replacement** to form a *bootstrap resample*; some points appear several times, others not at all.
2. Compute the statistic $\hat\theta^*$ on the resample.
3. Repeat $B$ times to obtain $\hat\theta^*_1,\ldots,\hat\theta^*_B$.

The spread of these $B$ replicates approximates the sampling distribution of $\hat\theta$, so their standard deviation estimates its standard error, and the $\alpha/2$ and $1-\alpha/2$ empirical percentiles of $\{\hat\theta^*_b\}$ form a *percentile* confidence interval. How many replicates? A few hundred ($B\approx200$) suffice for a standard error, but the percentile interval rests on estimated tail quantiles, so use $B$ of at least $1{,}000$--$2{,}000$; resampling is cheap, and below we simply take $B=10{,}000$. :numref:`fig_mdl-bootstrap` shows the construction: one original sample fans out into many resamples, whose statistics pile up into a histogram standing in for the true (unknowable) sampling distribution, with the central band cut off at those percentiles. The crucial caveat is that this resampling distribution is *centered at $\hat\theta$, not at $\theta$*: the bootstrap estimates the *shape and width* of the sampling distribution from the one sample we have, and is only as representative as that sample.

![The bootstrap. From a single observed sample (top) we draw many resamples of the same size *with replacement* (middle); recomputing the statistic $\hat\theta$ on each gives the replicates $\hat\theta^\ast_b$, whose histogram (bottom) approximates the sampling distribution. Its spread estimates the standard error, and the central $1-\alpha$ percentile band is a confidence interval. The resampling distribution is centered at $\hat\theta$, the dashed estimate, rather than at the unknown true $\theta$.](../img/mdl-prob-bootstrap.svg)
:label:`fig_mdl-bootstrap`

Let us bootstrap a statistic with no closed-form standard error---the **median**---from a skewed sample, and contrast it with the Gaussian machinery. The code is pure NumPy (`onp`) because resampling is framework-agnostic: we index the data with a matrix of random positions to draw $B$ resamples at once.

```{.python .input #statistics-bootstrap}
rng = onp.random.default_rng(0)
data = rng.exponential(scale=1.0, size=200)  # skewed: median != mean, no SE formula
n = len(data)
theta_hat = onp.median(data)                 # statistic of interest

B = 10000                                     # number of bootstrap resamples
idx = rng.integers(0, n, size=(B, n))         # n positions per resample, WITH replacement
boot = onp.median(data[idx], axis=1)          # one median per resample

se_boot = boot.std(ddof=1)                    # bootstrap standard error of the median
ci_pct = onp.percentile(boot, [2.5, 97.5])    # percentile 95% CI -- no formula needed
print(f'sample median        = {theta_hat:.3f}')
print(f'bootstrap SE         = {se_boot:.3f}')
print(f'percentile 95% CI    = ({ci_pct[0]:.3f}, {ci_pct[1]:.3f})')
```

The bootstrap hands us a standard error and an interval for the median directly, with no distribution theory at all. For contrast, the Gaussian formula :eqref:`eq_mdl-gauss_confidence` only knows how to handle the *mean*---a different target, which on this skewed data sits well above the median.

```{.python .input #statistics-bootstrap-contrast}
mu_hat = data.mean()
se_mean = data.std(ddof=1) / n**0.5           # closed-form SE, but only for the mean
ci_gauss = (mu_hat - 1.96 * se_mean, mu_hat + 1.96 * se_mean)
print(f'Gaussian 95% CI (mean) = ({ci_gauss[0]:.3f}, {ci_gauss[1]:.3f})')
```

The two intervals answer different questions and do not overlap, a direct consequence of the skew. The percentile interval for the median is also slightly *asymmetric* about $\hat\theta$---it inherits the shape of the resampling distribution rather than forcing the symmetric $\pm 1.96\,\widehat{\operatorname{se}}$ of a Gaussian. This is exactly why the bootstrap is indispensable in machine learning: error bars on a held-out accuracy, an AUC, or a BLEU score have no closed-form standard error, but resampling the test set delivers one in a few lines :cite:`Efron.Hastie.2016`.

## Summary

* An *estimator* $\hat\theta_n$ is a function of the data; being random, it has a *sampling distribution* whose center and spread are summarized by *bias* $\mathbb{E}[\hat\theta_n]-\theta$ and *variance*. *Consistency* ($\hat\theta_n\xrightarrow{P}\theta$) follows when both shrink with $n$; *efficiency* ranks unbiased estimators by their variance.
* The *bias-variance decomposition* $\operatorname{MSE}(\hat\theta_n)=\operatorname{Bias}(\hat\theta_n)^2+\operatorname{Var}(\hat\theta_n)$ splits the error cleanly because, after centering at $\mathbb{E}[\hat\theta_n]$, the cross term vanishes. This is the same U-curve as the under/overfitting trade-off (expected test error adds an irreducible noise floor $\sigma^2$), and it explains why regularization trades bias for variance---though past the interpolation threshold, overparameterized models can break the U (*double descent*).
* The unbiased sample variance divides by $n-1$, not $n$: estimating the mean from the same data costs one degree of freedom, and the $1/(n-1)$ factor corrects the resulting bias exactly.
* *Hypothesis testing* weighs evidence against a null $H_0$ via a test statistic and its $p$-value $P_{H_0}(\textrm{data this extreme})$; we control the type I error rate $\alpha$ and want high power $1-\beta$. A $p$-value is not $P(H_0\mid\textrm{data})$. A *permutation test* manufactures the null distribution by shuffling group labels, with no Gaussian assumptions; under many tests, control the family-wise error rate (Bonferroni) or the false discovery rate (Benjamini--Hochberg).
* A *confidence interval* contains $\theta$ with probability $\ge 1-\alpha$ over repeated datasets; the Gaussian interval $\hat\mu_n \pm 1.96\,\hat\sigma_n/\sqrt n$, licensed by the *central limit theorem*, has half-width shrinking like $1/\sqrt n$.
* The *bootstrap* estimates the sampling distribution of *any* statistic with no closed-form standard error---a median, an accuracy, an AUC---by resampling the data with replacement: the spread of the replicates is the standard error and their central percentiles form a confidence interval.

## Exercises

1. Let $X_1, \ldots, X_n \overset{\textrm{iid}}{\sim} \textrm{Unif}(0,\theta)$ and consider the estimators $\hat\theta = \max\{X_1,\ldots,X_n\}$ and $\tilde\theta = \frac2n\sum_i X_i$. Find the bias, variance, and MSE of each, and decide which is better. Is $\hat\theta$ biased? Is it consistent? Finally, try to bootstrap a confidence interval for $\theta$ from $\hat\theta=\max_i X_i$ and explain why it is poor: with what probability does a bootstrap resample contain the largest observation, so that $\hat\theta^* = \hat\theta$ exactly? (This is the classic example where the bootstrap fails: the true sampling distribution of $\hat\theta$ lives entirely below $\theta$, while the resampling distribution puts a large point mass exactly at $\hat\theta$.)
2. Prove the bias-variance decomposition :eqref:`eq_mdl-bias-variance` directly by expanding $\mathbb{E}[(\hat\theta_n-\theta)^2]$ into $\mathbb{E}[\hat\theta_n^2]-2\theta\,\mathbb{E}[\hat\theta_n]+\theta^2$ and substituting $\mathbb{E}[\hat\theta_n^2]=\operatorname{Var}(\hat\theta_n)+\mathbb{E}[\hat\theta_n]^2$. Confirm it agrees with the add-and-subtract proof in the text.
3. The decomposition check computes the variance of the $10{,}000$ estimates with the plug-in estimator (`ddof=0`). Rerun it with the unbiased estimator (`ddof=1`) and compare the two sides of :eqref:`eq_mdl-bias-variance` again. Which variance estimator makes the identity exact to floating-point precision, and why? (Hint: the proof manipulates expectations under a single distribution. Treat the empirical distribution of the estimates as that distribution: its expectations are plain averages that divide by the number of estimates---which denominator does that force? How large is the resulting mismatch with the other choice, and how does it shrink as the number of datasets grows?)
4. Shrink the per-dataset size $n$ in the sampling-distribution simulation and confirm the spread of $\hat\theta$ widens like $\sigma/\sqrt n$. Repeat with the biased estimator $\hat\theta=\max_i X_i$ for $\textrm{Unif}(0,\theta)$ and watch the center shift away from $\theta$.
5. A test reports $p = 0.5$. Is this evidence that $H_0$ is true? Explain in terms of $P(\textrm{data}\mid H_0)$ versus $P(H_0\mid\textrm{data})$, and describe a situation where a large $p$-value reflects only low power.
6. Using the $1/(\textrm{effect size})^2$ scaling, estimate how many times more samples are needed to detect an effect of size $0.1$ than one of size $0.5$ at the same $\alpha$ and power.
7. Run the confidence-interval code with $N=2$ and $\alpha=0.5$ (so $t_\star=1.0$) for $100$ independently generated datasets, and look at the resulting intervals. Some are extremely short and far from the true mean $0$. Does this contradict the $1-\alpha$ coverage guarantee? Would you trust a short interval as a sign of a precise estimate?
8. Modify the bootstrap code to target the *mean* instead of the median. Compare the bootstrap standard error and percentile interval against the closed-form Gaussian results $\hat\sigma_n/\sqrt n$ and :eqref:`eq_mdl-gauss_confidence` on the same sample. Do they roughly agree? Why should they, given the central limit theorem?
9. Suppose you compare $m=20$ models against a baseline, all in truth no better, each with an independent test at $\alpha=0.05$. What is the probability that at least one clears $p\le 0.05$ by chance? Recompute the per-test threshold the Bonferroni correction prescribes to hold the family-wide false-positive rate at $0.05$, and verify it brings the spurious-win probability back near $0.05$.

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/419)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/1102)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/1103)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/1103)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §25.5]{.kicker}

Estimators, hypothesis tests, and confidence intervals<br>**statistics for practitioners**.
:::
:::

::: {.slide title="Why statistics?"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
A trained model is a guess made from random data. Statistics says how far
that guess sits from the truth, whether an improvement is real, and how
much to trust a number.

- Bias–variance = the under/overfit U-curve.
- p-values behind every A/B test and benchmark claim.
- Confidence intervals behind the error bars on a loss curve.
:::

::: {.col .fig}
@fig:mdl-prob-sampling-distribution
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[Estimators and their quality]{.dtitle}

[bias, variance, MSE, consistency, efficiency]{.dsub}
:::
:::

::: {.slide title="An estimator is a random variable"}
[Estimators]{.kicker}

$\hat\theta_n = \hat f(x_1,\dots,x_n)$ is computed from a random sample, so
it has a **sampling distribution** — it would land somewhere else on a
fresh dataset.

::: {.d2l-note}
That randomness is the whole subject: how much would the answer move, and
is it centered on the truth?
:::
:::

::: {.slide title="Bias and variance"}
[Estimators]{.kicker}

$$\operatorname{Bias}(\hat\theta_n) = \mathbb E[\hat\theta_n] - \theta,
\qquad
\operatorname{Var}(\hat\theta_n) = \mathbb E\bigl[(\hat\theta_n - \mathbb E[\hat\theta_n])^2\bigr].$$

Bias is a systematic offset (it does **not** wash out with more data);
variance is measured against the estimator's own center, not the truth.

@fig:mdl-prob-sampling-distribution
:::

::: {.slide title="Consistency and efficiency"}
[Estimators]{.kicker}

**Consistent**: $\hat\theta_n\xrightarrow{P}\theta$ — guaranteed if both
bias and variance vanish (the law of large numbers is the prototype).

. . .

**Efficient**: smallest variance among unbiased estimators. The
Cramér–Rao bound is the floor; the MLE reaches it asymptotically.
:::

::: {.slide title="MSE: the honest scorecard"}
[Estimators]{.kicker}

$\operatorname{MSE}(\hat\theta_n) = \mathbb E[(\hat\theta_n-\theta)^2]$
folds both errors into one number.

::: {.d2l-note .rule}
**Proposition.**
$\operatorname{MSE}(\hat\theta_n) =
\operatorname{Bias}(\hat\theta_n)^2 + \operatorname{Var}(\hat\theta_n).$
:::
:::

::: {.slide title="Proof: the cross term vanishes"}
[Estimators]{.kicker}

Let $\mu=\mathbb E[\hat\theta_n]$ and split
$\hat\theta_n-\theta = (\hat\theta_n-\mu) + (\mu-\theta)$.

. . .

Squaring and taking expectations, the cross term is
$2(\mu-\theta)\,\mathbb E[\hat\theta_n-\mu] = 0$, leaving
$\operatorname{Var} + \operatorname{Bias}^2$. $\blacksquare$

::: {.d2l-note}
Markov then gives $P(|\hat\theta_n-\theta|>\varepsilon)\le
\operatorname{MSE}/\varepsilon^2$ — small MSE forces consistency.
:::
:::

::: {.slide title="Verify it in code"}
[Estimators]{.kicker}

Ten thousand sample means of $n=30$ Gaussian draws; the two sides of the
decomposition agree to floating point:

@statistics-sampling-distribution

. . .

@statistics-verify-decomposition
:::

::: {.slide title="The U-curve = generalization"}
[Estimators]{.kicker}

::: {.cols .vc}
::: {.col}
As model complexity grows, bias falls and variance rises; expected test
error is their sum plus irreducible noise:

$$\text{err} = \operatorname{Bias}^2 + \operatorname{Var} + \sigma^2.$$

Regularization deliberately adds bias to cut variance more.
:::

::: {.col .fig}
@fig:mdl-prob-bias-variance-u-curve
:::
:::
:::

::: {.slide title="Why divide by n − 1?"}
[Estimators]{.kicker}

Deviations from $\bar x$ are too small (it minimizes them), so dividing by
$n$ is biased low.

::: {.d2l-note .rule}
**Proposition.** $s^2 = \tfrac{1}{n-1}\sum_i (x_i-\bar x)^2$ has
$\mathbb E[s^2]=\sigma^2$ — one degree of freedom is spent estimating
$\bar x$.
:::

@statistics-unbiased-variance
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Hypothesis testing]{.dtitle}

[null & alternative, test statistic, p-value, power]{.dsub}
:::
:::

::: {.slide title="Two kinds of error"}
[Testing]{.kicker}

::: {.cols .vc}
::: {.col}
We never prove $H_0$ — only reject it or fail to:

$$\alpha = P(\text{reject}\mid H_0), \qquad
\beta = P(\text{fail}\mid H_A).$$

$\alpha$ = significance (false positive); $1-\beta$ = power.
:::

::: {.col .fig}
@fig:mdl-prob-type-i-ii-matrix
:::
:::
:::

::: {.slide title="Significance and power"}
[Testing]{.kicker}

Fix $\alpha$ (often $0.05$) and target power (often $0.8$). The sample size
to detect an effect scales as $n\propto 1/\delta^2$:

@fig:mdl-prob-power

::: {.d2l-note}
A $0.01$ improvement needs tens of thousands of test examples to confirm —
why marginal benchmark gains are fragile.
:::
:::

::: {.slide title="The five-step recipe"}
[Testing]{.kicker}

1. State $H_0$ and $H_A$.
2. Fix $\alpha$ and a target power → required $n$.
3. Collect data.
4. Compute the statistic $T(x)$ and its p-value.
5. Reject $H_0$ iff $p\le\alpha$.

::: {.d2l-note .rule}
A p-value is $P(\text{data this extreme}\mid H_0)$ — **not**
$P(H_0\mid\text{data})$.
:::
:::

::: {.slide title="The rejection region"}
[Testing]{.kicker}

$p = P_{H_0}\bigl(|T| \ge |t_{\text{obs}}|\bigr)$; land in the tails of
total mass $\alpha$ and reject:

@fig:mdl-prob-significance

::: {.d2l-note}
Run $m$ tests and $P(\ge 1\text{ false win}) = 1-(1-\alpha)^m$. Bonferroni
tests at $\alpha/m$; at ML scale, control the false-discovery rate.
:::
:::

::: {.slide title="A worked test: two models"}
[Testing]{.kicker}

Under $H_0$ the labels are exchangeable, so **shuffle** them, recompute the
accuracy gap, and repeat. The p-value is the fraction of shuffles at least
as extreme:

@statistics-permutation-test

The $0.73\%$ gap is real here — but only just, with $20$ seeds.
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[Quantifying uncertainty]{.dtitle}

[confidence intervals and the bootstrap]{.dsub}
:::
:::

::: {.slide title="What a confidence interval is"}
[Intervals]{.kicker}

A random interval $C_n$ with $P_\theta(C_n \ni \theta)\ge 1-\alpha$ for all
$\theta$. The **interval** is random; $\theta$ is fixed.

. . .

Correct reading: across many repetitions, $95\%$ of the constructed
intervals trap $\theta$ — *not* that this one does with probability $0.95$
(that is a Bayesian credible interval, which needs a prior).
:::

::: {.slide title="The Gaussian interval"}
[Intervals]{.kicker}

By the CLT, $T=(\hat\mu_n-\mu)/(\hat\sigma_n/\sqrt n)$ is approximately
standard normal, giving

$$\Bigl[\hat\mu_n - 1.96\,\tfrac{\hat\sigma_n}{\sqrt n},\;
\hat\mu_n + 1.96\,\tfrac{\hat\sigma_n}{\sqrt n}\Bigr].$$

@statistics-confidence-interval

::: {.d2l-note}
Half-width shrinks like $1/\sqrt n$ — four times the data to halve the
error bar.
:::
:::

::: {.slide title="The bootstrap: any statistic, no formula"}
[Intervals]{.kicker}

Substitute the empirical distribution for the unknown one: resample $n$
points **with replacement**, recompute the statistic, repeat:

@fig:mdl-prob-bootstrap

::: {.d2l-note .rule}
Works for the median, AUC, accuracy, BLEU — anywhere no closed-form
standard error exists.
:::
:::

::: {.slide title="Bootstrap in code"}
[Intervals]{.kicker}

The bootstrap SE and a percentile interval for a skewed sample's median:

@statistics-bootstrap

. . .

The mean's Gaussian interval lands elsewhere — same data, different
statistic, different answer:

@statistics-bootstrap-contrast
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

::: {.cols}
::: {.col}
- Estimators are random; quality = bias, variance, and their sum, MSE = $\operatorname{Bias}^2+\operatorname{Var}$.
- Consistent when both vanish; the U-curve is under/overfitting as one identity; $n-1$ pays for one degree of freedom.
:::

::: {.col}
- Testing: control $\alpha$, want power $\ge 0.8$; the p-value is data-given-null, not null-given-data; correct for multiplicity.
- Intervals shrink like $1/\sqrt n$; the bootstrap gives error bars for any statistic.
:::
:::

::: {.d2l-note}
The MSE split, the U-curve, and the CLT are the same few ideas in
different clothes throughout the book.
:::
:::
