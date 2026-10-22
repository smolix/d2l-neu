# Naive Bayes
:label:`sec_mdl-naive_bayes`

The **naive Bayes** classifier estimates a probabilistic model from counts and
predicts with Bayes' rule. Its conditional-independence assumption reduces a
high-dimensional density estimate to separate estimates for each feature. This
section derives the classifier, applies it to handwritten digits, and evaluates
its errors and calibration. A bootstrap interval quantifies uncertainty in its
accuracy.

```{.python .input #naive-bayes-imports}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, npx
from mxnet import np as mnp
import numpy as np
npx.set_np()
d2l.use_svg_display()
```

```{.python .input #naive-bayes-imports}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
import torch
import torchvision
d2l.use_svg_display()
```

```{.python .input #naive-bayes-imports}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
d2l.use_svg_display()
```

```{.python .input #naive-bayes-imports}
#@tab jax
%matplotlib inline
from d2l import jax as d2l
import numpy as np
import tensorflow as tf
d2l.use_svg_display()
```

## Bayes' Rule for Classification

A classifier maps an example $\mathbf{x}\in\mathbb{R}^d$ to a label
$y\in\{1,\ldots,K\}$. A probabilistic classifier estimates the posterior
$p(y\mid\mathbf{x})$ and predicts its most probable label:

$$\hat{y} = \mathop{\mathrm{argmax}}_y \, p(y\mid\mathbf{x}).$$

An unconstrained table for $p(y\mid\mathbf{x})$ is impractical in high
dimensions. With $d$ binary features there are $2^d$ possible inputs, so the
table grows exponentially and most entries receive little or no data. For an
MNIST image, $d=784$.

Bayes' rule :eqref:`eq_mdl-bayes_density` expresses this posterior in terms
of a class-conditional distribution, the *generative* direction:

$$\hat{y} = \mathop{\mathrm{argmax}}_y \, p(y\mid\mathbf{x}) = \mathop{\mathrm{argmax}}_y \, \frac{p(\mathbf{x}\mid y)\,p(y)}{p(\mathbf{x})} = \mathop{\mathrm{argmax}}_y \, p(\mathbf{x}\mid y)\,p(y).$$

The denominator $p(\mathbf{x})$ is independent of $y$ and therefore does not
affect the maximizing label. Hence
$p(y\mid\mathbf{x})\propto p(\mathbf{x}\mid y)p(y)$. If normalized posterior
probabilities are needed, divide these class scores by their sum.

A **generative classifier** models the class-conditional distribution
$p(\mathbf{x}\mid y)$ and the prior $p(y)$, then applies Bayes' rule.
A **discriminative classifier**, such as logistic regression or the softmax
classifier in :numref:`sec_softmax`, models $p(y\mid\mathbf{x})$ directly and
need not specify the input distribution.

For the models studied by :citet:`Ng.Jordan.2002`, a generative classifier can
approach its asymptotic error with fewer examples, while a discriminative model
can attain a lower asymptotic error. This comparison depends on the model and
data distribution; neither ordering is universal. Naive Bayes is a simple
generative classifier. :numref:`fig_mdl-naive-genvdisc` compares the two
approaches.

![Two routes to a classifier. Left: a generative model learns how each class generates inputs (the class-conditional densities $p(\mathbf{x}\mid y)$ weighted by the priors $p(y)$) and flips them through Bayes' rule, predicting whichever class makes the observation most plausible; the tie point of the weighted densities is where the decision flips. Right: a discriminative model need not specify how inputs are generated; it devotes its capacity directly to the posterior $p(y\mid\mathbf{x})$, that is, to the decision boundary between the classes.](../img/mdl-prob-naive-genvdisc.svg)
:label:`fig_mdl-naive-genvdisc`

This reformulation alone does not reduce the number of parameters:
$p(\mathbf{x}\mid y)$ is still a distribution over $2^d$ binary patterns.
The probability chain rule (:numref:`sec_mdl-random_variables`) gives

$$p(\mathbf{x}\mid y) = p(x_1\mid y)\,p(x_2\mid x_1, y)\cdots p(x_d\mid x_1,\ldots,x_{d-1}, y),$$

The later factors depend on increasingly long feature histories, so the
unrestricted model still requires exponentially many parameters.

### The Naive Assumption

Naive Bayes assumes that the features are jointly conditionally independent
given the label. Pairwise conditional independence is insufficient when $d>2$;
the required assumption factorizes the complete conditional distribution into
one-feature marginals:

$$p(\mathbf{x}\mid y) = \prod_{i=1}^d p(x_i\mid y).$$
:eqlabel:`eq_mdl-naive_assumption`

For binary features, an unrestricted class-conditional distribution has
$2^d-1$ free probabilities per class. The factorization uses only $d$
Bernoulli parameters per class, plus $K-1$ class-prior parameters. This is the
computational and statistical benefit of the assumption.

As a graphical model the factorization is a star: the label $y$ is a parent of
each feature, with no edges among features
(:numref:`fig_mdl-naive-independence`, left). Pixel data retain substantial
conditional dependence after the digit label is known, as illustrated by the
additional feature edges in the right panel.

![Conditional-independence structures. Left: in naive Bayes, the label $y$ is a parent of each feature and there are no feature-to-feature edges, representing $p(\mathbf x\mid y)=\prod_i p(x_i\mid y)$. Right: a model with additional conditional dependence includes feature edges such as $x_1$–$x_2$ and $x_3$–$x_d$.](../img/mdl-prob-naive-independence.svg)
:label:`fig_mdl-naive-independence`

For MNIST this assumption is inaccurate: neighboring pixels remain correlated
within a digit class. Nevertheless, classification depends on the ordering of
class scores, not on every estimated probability being accurate. Naive Bayes
can therefore classify well even when its posterior probabilities are poorly
calibrated :cite:`Domingos.Pazzani.1997`.

Substituting :eqref:`eq_mdl-naive_assumption` into the Bayes classifier gives
the **naive Bayes rule** :cite:`Maron.1961`:

$$\hat{y} = \mathop{\mathrm{argmax}}_y \; p(y) \prod_{i=1}^d p(x_i\mid y).$$
:eqlabel:`eq_mdl-naive_bayes`

### Computation in Log Space

Equation :eqref:`eq_mdl-naive_bayes` multiplies $d$ numbers in $[0,1]$.
For $d=784$, these products can be extremely small. In the MNIST experiment
below, 99.1% of the single-precision class scores underflow to zero. Even in
float64, the nonzero products range from about $10^{-26}$ to $10^{-323}$.

The logarithm is strictly increasing, so it preserves the maximizing class and
turns products into sums. Applying it to :eqref:`eq_mdl-naive_bayes` yields the
stable log-space rule discussed in
:numref:`sec_mdl-numerical-stability-conditioning`:

$$\hat{y} = \mathop{\mathrm{argmax}}_y \; \log p(y) + \sum_{i=1}^d \log p(x_i\mid y).$$
:eqlabel:`eq_mdl-naive_bayes_log`

Each class score is now a sum of 785 log probabilities rather than a
product prone to underflow, and each term is finite as long as no estimated
probability is zero. Prediction requires a table of precomputed log
probabilities and a sequence of additions.

For a binary feature,
$\log p(x_i\mid y)=x_i\log p(x_i=1\mid y)+(1-x_i)\log p(x_i=0\mid y)$.
Therefore :eqref:`eq_mdl-naive_bayes_log` becomes

$$\log p(y) + \sum_{i=1}^d \Bigl[ x_i \log p(x_i{=}1\mid y) + (1-x_i)\log p(x_i{=}0\mid y) \Bigr],$$

which is **affine in $\mathbf{x}$**, a constant plus a weighted sum of the
pixels :cite:`Bishop.2006`. Thus **Bernoulli naive Bayes** has class scores
$\mathbf{w}_y\cdot\mathbf{x}+b_y$, and every pairwise decision boundary is a
hyperplane. Multinomial naive Bayes is similarly affine in feature counts.
These models have the same boundary form as softmax regression, although their
parameters are estimated from class-conditional counts rather than by directly
optimizing conditional likelihood.

This linear-boundary conclusion does not hold for every naive Bayes likelihood.
With Gaussian features, the class score contains
$-(x_i-\mu_{iy})^2/(2\sigma_{iy}^2)$. If the variance of feature $i$ is shared
across classes, its $x_i^2$ term cancels when two class scores are compared and
the boundary is linear. With class-specific variances, the quadratic terms do
not cancel, and the boundary is generally quadratic.

## Parameter Estimation from Counts

Naive Bayes estimates the class prior $p(y)$ and the feature likelihoods
$p(x_i\mid y)$. For the categorical and Bernoulli models considered here,
their maximum-likelihood estimates are empirical frequencies.

If class $y$ occurs $n_y$ times among $n=\sum_y n_y$ examples, then
$\hat p(y)=n_y/n$. For a binary feature, $\hat p(x_i=1\mid y)$ is the
fraction of class-$y$ examples in which feature $i$ is one. We store these
values in an array $P_{xy}$, represented below as one $28\times28$ grid per
class. The complementary probability is
$\hat p(x_i=0\mid y)=1-\hat p(x_i=1\mid y)$.

One hazard remains. If feature $i$ is *never* on for class $y$ in the training set, the MLE is $\hat p(x_i=1\mid y)=0$, and a single such feature that is on at test time annihilates the whole product in :eqref:`eq_mdl-naive_bayes` (and sends :eqref:`eq_mdl-naive_bayes_log` to $-\infty$). The cure is **Laplace smoothing** :cite:`Laplace.1814`: add a pseudocount, estimating $p(x_i=1\mid y)$ as $(n_{iy}+1)/(n_y+2)$ rather than $n_{iy}/n_y$, where $n_{iy}$ counts the class-$y$ examples with feature $i$ on and the $+2$ covers the two outcomes a binary pixel can take; a categorical feature with $v$ possible values gets $+1$ on each value's count and $+v$ in the denominator, so the smoothed probabilities still sum to one (the multinomial text model of exercise :exref:`ex_mdl-naive-bayes-two-event-models` smooths the same way, with $+|V|$ for a vocabulary $V$). The pseudocount has a Bayesian justification: $(n_{iy}+1)/(n_y+2)$ is the posterior *mean* of the Bernoulli parameter under a uniform $\mathrm{Beta}(1,1)$ prior, Laplace's *rule of succession*, as :numref:`subsec_mdl-beta-map` derives (along with the numerically coincident, but conceptually distinct, $\mathrm{Beta}(2,2)$ MAP estimate it is often conflated with).

For continuous features, **Gaussian naive Bayes** models each feature within
each class by a univariate Gaussian (:numref:`sec_mdl-distributions`).
Training estimates a mean and variance for each class--feature pair, and the
log-space score sums Gaussian log densities. In the experiment below we instead
threshold pixel intensities into binary features. This produces a simpler model
but discards grayscale information.

## A Worked Example: MNIST Digits

MNIST contains $28\times28$ grayscale images of handwritten digits
:cite:`LeCun.Bottou.Bengio.ea.1998`. We threshold each pixel at half intensity,
so $x_i\in\{0,1\}$ records whether it is active. The training and test splits
are converted to NumPy arrays: `X` has shape `(n, 28, 28)` and `Y` contains
integer labels. Subsequent estimation uses only NumPy counts.

```{.python .input #naive-bayes-load}
#@tab mxnet
def binarize(data, label):
    return mnp.floor(data.astype('float32') / 128).squeeze(axis=-1), label

train = gluon.data.vision.MNIST(train=True).transform(binarize)
test = gluon.data.vision.MNIST(train=False).transform(binarize)
X, Y = (a.asnumpy() for a in train[:])
X_test, Y_test = (a.asnumpy() for a in test[:])
X.shape, Y.shape
```

```{.python .input #naive-bayes-load}
#@tab pytorch
def load(train):
    ds = torchvision.datasets.MNIST(root='../data', train=train, download=True)
    X = np.floor(ds.data.numpy() / 128).astype('float32')
    return X, ds.targets.numpy()

X, Y = load(train=True)
X_test, Y_test = load(train=False)
X.shape, Y.shape
```

```{.python .input #naive-bayes-load}
#@tab tensorflow
(X, Y), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
X = np.floor(X / 128).astype('float32')
X_test = np.floor(X_test / 128).astype('float32')
X.shape, Y.shape
```

```{.python .input #naive-bayes-load}
#@tab jax
(X, Y), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
X = np.floor(X / 128).astype('float32')
X_test = np.floor(X_test / 128).astype('float32')
X.shape, Y.shape
```

### Estimating the Model

Training is two counts, done in NumPy. The prior $\hat p(y)$ is the class frequency; the likelihood array $P_{xy}$ holds, for each class, the Laplace-smoothed fraction of its images in which each pixel is inked. Prediction will then work in log space. The per-class `for y in range(10)` loop below is written for clarity, one class at a time; an idiomatic vectorized form would replace it with `np.bincount(Y)` for the class counts and a one-hot matrix multiply `onehot(Y).T @ X` for the per-class pixel sums.

```{.python .input #naive-bayes-train}
n_y = np.array([(Y == y).sum() for y in range(10)])
P_y = n_y / n_y.sum()                                  # class prior, p(y)

n_x = np.stack([X[Y == y].sum(axis=0) for y in range(10)])
P_xy = (n_x + 1) / (n_y + 2).reshape(10, 1, 1)         # Laplace-smoothed p(x_i=1|y)
P_y
```

Each class's slice of $P_{xy}$ is a $28\times 28$ image of per-pixel "on" probabilities for one digit: a picture of what the model believes a class looks like. Plotting all ten shows ghostly averaged digits: this is the entire learned model.

```{.python .input #naive-bayes-templates}
d2l.show_images([P_xy[y] for y in range(10)], 2, 5,
                titles=[str(y) for y in range(10)]);
```

The blur is the naive assumption made visible: the model knows each pixel's marginal firing rate but nothing about how pixels co-occur, so every sharp stroke smears into an average.

### Classifying and Evaluating

Prediction sums log-likelihoods per :eqref:`eq_mdl-naive_bayes_log`. We precompute $\log P_{xy}$, $\log(1-P_{xy})$, and $\log P_y$, then score a batch of images at once: a pixel that is on contributes $\log P_{xy}$, a pixel that is off contributes $\log(1-P_{xy})$ (the code computes both terms for every pixel and weights them by $x_i$ and $1-x_i$), and the prior is added in. The winning class is the $\mathrm{argmax}$. Before reporting the accuracy, the cell also puts numbers to the underflow story that motivated log space, by checking what the *raw* products of :eqref:`eq_mdl-naive_bayes` would have done in single and double precision.

```{.python .input #naive-bayes-predict}
log_P_xy, log_P_xy_neg = np.log(P_xy), np.log(1 - P_xy)
log_P_y = np.log(P_y)

def scores(X):                       # log p(y) + sum_i log p(x_i|y), per class
    X = X.reshape(-1, 1, 28, 28)                       # (m, 1, 28, 28)
    return (X * log_P_xy + (1 - X) * log_P_xy_neg).reshape(
        len(X), 10, -1).sum(axis=2) + log_P_y

def predict(X):
    return scores(X).argmax(axis=1)

s = scores(X_test)                   # measure what the raw products would do
print(f'float32 underflow: {(np.exp(s.astype(np.float32)) == 0).mean():.1%}'
      f' of class scores; smallest float64 survivor'
      f' = 1e{int(s[np.exp(s) > 0].min() / np.log(10))}')
float((predict(X_test) == Y_test).mean())              # Test accuracy
```

Naive Bayes reaches about $84\%$ accuracy, compared with $10\%$ for random
guessing. Its ten averaged templates cannot represent dependence among pixels,
which limits performance on images. Modern neural classifiers model these
dependencies and obtain substantially lower error rates.

Images are in fact the *hard* case for the naive assumption, because adjacent pixels are so tightly coupled. Its canonical home is the opposite regime: **bag-of-words text classification**, where each feature records whether a given word appears in a document (the *Bernoulli* event model; the *multinomial* variant counts occurrences instead) :cite:`Manning.Raghavan.Schutze.2008`. There the independence story comes far closer to holding, though it is still not literally true: the presence of any one word out of a vocabulary of tens of thousands says comparatively little about the presence of most others, whereas neighboring pixels almost always agree. The same counting-and-argmax recipe therefore makes naive Bayes a strong, famously cheap baseline for topic labelling and spam filtering, which it dominated for decades :cite:`Sahami.Dumais.Heckerman.ea.1998`.

### Calibration

Three questions remain about the accuracy number, and a generative classifier with a knowingly false assumption is exactly the model to ask them of: how *precise* is the $84.27\%$; *which* mistakes make up the missing $16\%$; and can the model's own confidence be *trusted*? Each takes a few lines.

First, precision. The test accuracy is an estimate computed from $10{,}000$ random test examples, so it carries a standard error, and the bootstrap of :numref:`sec_mdl-statistics` delivers it exactly as promised there: resample the test indices with replacement, recompute the accuracy on each resample, and read off the spread.

```{.python .input #mdl-naive-bayes-calibration-1}
rng = np.random.default_rng(0)
correct = (predict(X_test) == Y_test)                  # per-example 0/1 outcomes
idx = rng.integers(0, len(correct), (1000, len(correct)))  # resampled test sets
boot = correct[idx].mean(axis=1)                       # one accuracy per resample
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f'test accuracy = {correct.mean():.4f}, '
      f'bootstrap 95% CI = ({lo:.4f}, {hi:.4f})')
```

The error bar spans about $\pm0.7$ accuracy points: the "$84.27\%$" is really "$84.3\%\pm0.7$", and the third decimal is noise. One caution about comparisons: this interval quantifies the uncertainty in *our model's* accuracy, not the gap to a competitor. Deciding whether another model on the same test set is genuinely better calls for a *paired* comparison (bootstrap the accuracy *difference* example by example, or use McNemar's test); because the two models' errors are correlated, a competitor whose accuracy falls inside our interval can still be significantly better. Second, the mistakes. A *confusion matrix* tallies predictions against truth, one row per true digit and one column per predicted digit, and turns the flat error rate into structure.

```{.python .input #mdl-naive-bayes-calibration-2}
conf = np.zeros((10, 10), dtype=int)
np.add.at(conf, (Y_test, predict(X_test)), 1)          # tally (true, predicted)
off = conf - np.diag(np.diag(conf))                    # errors only
worst = np.dstack(np.unravel_index(
    np.argsort(off, axis=None)[::-1], (10, 10)))[0][:4]
print('most confused true->predicted:',
      ', '.join(f'{t}->{p} ({off[t, p]})' for t, p in worst))
d2l.plt.imshow(conf, cmap='Blues')
d2l.plt.xlabel('predicted digit')
d2l.plt.ylabel('true digit')
d2l.plt.colorbar();
```

The errors are anything but uniform: the model's worst failures are $4\to9$, $5\to3$, $8\to3$, and $7\to9$ (the reverse confusion $9\to4$ sits just below the printed four). These are largely the digit pairs whose learned *templates*, plotted above, overlap the most: the most-overlapping pair is indeed the top confusion $4/9$, though the correspondence is loose rather than exact, as Exercise :exref:`ex_mdl-naive-bayes-check-template-overlap-explanation` lets you check. A $4$ and a $9$ differ mainly in whether the top strokes close into a loop, a fact carried by the *joint* behavior of a handful of neighboring pixels; a model that sees only per-pixel marginals is structurally blind to it. The confusion matrix is the independence assumption's failure map.

Third, the confidence. The class scores determine a genuine posterior: since each score $s_y(\mathbf x)$ is the logarithm of the Bayes-rule numerator $p(\mathbf x\mid y)\,p(y)$, exponentiating and normalizing restores the dropped denominator, so $p(y\mid\mathbf x) = \exp(s_y(\mathbf x))/\sum_{y'}\exp(s_{y'}(\mathbf x))$, a softmax of the scores (exercise :exref:`ex_mdl-naive-bayes-discussion-after-eqref-eq` derives the two-class case). The model therefore announces a probability along with each predicted digit. A model is **calibrated** if its announced probabilities match empirical
frequencies: among predictions made with confidence $c$, a fraction $c$ should
be correct. We predicted early in this section that multiplying $784$ falsely-independent factors would leave the model badly miscalibrated; the cell below checks, binning the test examples by their maximum posterior and comparing the claimed confidence with the achieved accuracy in each bin, the tabular form of a *reliability diagram*, the standard name for this diagnostic. (The softmax is evaluated in its subtract-the-max stable form from :numref:`subsec_mdl-stable-softmax`.)

```{.python .input #mdl-naive-bayes-calibration-3}
s = scores(X_test)
post = np.exp(s - s.max(axis=1, keepdims=True))        # softmax, stably
post /= post.sum(axis=1, keepdims=True)
conf_max = post.max(axis=1)                            # claimed confidence
print(f'mean claimed confidence = {conf_max.mean():.3f},   '
      f'actual accuracy = {correct.mean():.3f}')
for a, b in zip([0.0, 0.9, 0.99, 0.999], [0.9, 0.99, 0.999, 1.0]):
    m = (conf_max > a) & (conf_max <= b)
    print(f'confidence in ({a}, {b}]: {m.sum():5d} examples,  '
          f'claimed {conf_max[m].mean():.4f},  achieved {correct[m].mean():.4f}')
```

On average the model claims $98.6\%$ confidence while delivering $84.3\%$ accuracy, and the miscalibration is worst where the claims are strongest: $87\%$ of all test examples land in the top bin, where the model asserts essentially $100\%$ certainty yet is right only $89.9\%$ of the time, and every other bin overstates itself too, with examples announced at $72\%$ confidence being correct barely $37\%$ of the time. (The *ordering* still carries signal, since accuracy does rise with confidence bin, but the probabilities themselves are grossly inflated.) This is exactly what the log-space picture predicts. Each of the $784$ pixels contributes its log-likelihood ratio to the score *as if* it were independent evidence, so correlated pixels get counted many times over and the score gaps between classes grow far beyond what the evidence supports: the median gap between the best and second-best class score on the test set is about $30$ nats (natural-log units; :numref:`sec_mdl-divergences-distances` makes the unit precise), and since $e^{-30}\approx 10^{-13}$ that already pushes the softmax to $0$ or $1$; gaps of hundreds of nats are typical only between the best and *worst* classes. The same failure extends well past naive Bayes: modern neural classifiers trained to low loss are overconfident in the same direction, though to a smaller degree :cite:`Guo.Pleiss.Sun.Weinberger.2017`, which is why the reliability check you just ran, confidence binned against accuracy, is a standard diagnostic for any classifier whose probabilities you intend to consume.

## Summary

* Bayes' rule recasts classification generatively: $p(y\mid\mathbf{x}) \propto p(\mathbf{x}\mid y)\,p(y)$, predicting the label that maximizes the numerator.
* As a **generative** model it estimates $p(\mathbf{x}\mid y)\,p(y)$, the mirror image of a **discriminative** model like softmax regression, which estimates $p(y\mid\mathbf{x})$ directly. Sample-efficiency and asymptotic-error comparisons depend on the model families and which assumptions match the data.
* For binary features, the conditional-independence factorization
  $p(\mathbf{x}\mid y)=\prod_i p(x_i\mid y)$ replaces $2^d-1$
  class-conditional parameters per class by $d$. Classification can remain
  useful even when the resulting probabilities are inaccurate.
* Working in log space avoids the underflow of multiplying hundreds of probabilities. Bernoulli and multinomial naive Bayes have scores affine in their chosen features and therefore linear decision boundaries; Gaussian naive Bayes with class-dependent variances generally has quadratic boundaries.
* Training is maximum likelihood by counting: class priors and per-feature frequencies. Laplace smoothing, $(n_{iy}+1)/(n_y+2)$ for a binary feature, is the posterior mean under a uniform prior (:numref:`subsec_mdl-beta-map`) and keeps every log-probability finite.
* On MNIST it learns ten averaged digit templates. Conditional dependence and
  binarization limit this model; sparse bag-of-words text is a common domain in
  which naive Bayes remains a useful baseline.
* The $84.27\%$ carries a bootstrap error bar of about $\pm 0.7$ points; the confusion matrix localizes the failures in template-overlapping pairs like $4/9$ and $5/3$; and the model is severely **miscalibrated**, claiming $98.6\%$ mean confidence while delivering $84.3\%$ accuracy, because falsely independent factors double-count correlated evidence. Bin confidence against accuracy (a reliability diagram) before trusting any classifier's probabilities.

## Exercises

1. **XOR and the naive assumption.** The four points $(0,0)$, $(0,1)$,
   $(1,0)$ and $(1,1)$ carry the labels $y=x_1\oplus x_2$, that is $0$, $1$,
   $1$ and $0$. Each point occurs once in the training set.
    1. Compute the maximum-likelihood estimates of $p(y)$ and
       $p(x_i=1\mid y)$, and the class scores of all four points. Which
       predictions does the naive Bayes rule :eqref:`eq_mdl-naive_bayes`
       make?
    1. Show that no choice of Bernoulli naive Bayes parameters classifies all
       four points correctly. Use the affine form of the class scores derived
       after :eqref:`eq_mdl-naive_bayes_log`.
    1. Replace the two factors $p(x_1\mid y)\,p(x_2\mid y)$ by a joint table
       $p(x_1,x_2\mid y)$. In the language of Bayesian networks
       :cite:`Koller.Friedman.2009`, this adds the edge between $x_1$ and
       $x_2$ that the right panel of :numref:`fig_mdl-naive-independence`
       shows. Estimate this model from the four points and show that it
       classifies them correctly. Why is its class score no longer affine in
       $\mathbf x$?
    1. Within one class, let $x_1$ and $x_2$ be independent fair bits and let
       $x_3=x_1\oplus x_2$. Show that every pair of the three features is
       independent. Compare the probability of each of the eight patterns
       with the product of the three marginals. Why can a check of all
       pairwise correlations within a class not confirm the assumption
       :eqref:`eq_mdl-naive_assumption`?
   :exlabel:`ex_mdl-naive-bayes-xor`

1. **Naive Bayes as logistic regression.** Consider two classes
   $y\in\{0,1\}$ with the prior $\pi=p(y=1)$ and the Bernoulli parameters
   $\theta_{iy}=p(x_i=1\mid y)$.
    1. Show that the naive Bayes posterior is
       $p(y=1\mid\mathbf x)=\sigma(\mathbf w^\top\mathbf x+b)$ with the
       sigmoid $\sigma(z)=1/(1+e^{-z})$, and express each $w_i$ and $b$
       through the $\theta_{iy}$ and $\pi$. Which sign does $w_i$ have for a
       pixel that is inked more often in class $1$ than in class $0$?
    1. Show that, conversely, every pair $(\mathbf w,b)$ arises from some
       choice of $\theta_{iy}\in(0,1)$ and $\pi\in(0,1)$. Naive Bayes and
       logistic regression therefore define the same set of posteriors and
       differ in how they fit it.
    1. Suppose the two classes of the training set are linearly separable.
       How does the conditional likelihood of logistic regression change when
       a separating $(\mathbf w,b)$ is multiplied by a factor $t\to\infty$?
       What does this imply for the maximum-likelihood estimate of
       $(\mathbf w,b)$? Which $(\mathbf w,b)$ does naive Bayes with Laplace
       smoothing return on the same data?
   :exlabel:`ex_mdl-naive-bayes-discussion-after-eqref-eq`

1. **Gaussian class-conditionals.** A single feature has
   $x\mid y=0\sim\mathcal N(0,1)$ and $x\mid y=1\sim\mathcal N(0,4)$, and both
   classes have prior $1/2$.
    1. Find the set of $x$ that the Bayes classifier assigns to class $1$.
       Why can no threshold on $x$ produce this set? Which transformed
       feature turns the boundary into a threshold?
    1. In $d$ dimensions, Gaussian naive Bayes with class-specific variances
       has the class score
       $$\log p(y)-\sum_{i=1}^d\left[\frac{(x_i-\mu_{iy})^2}{2\sigma_{iy}^2}+\frac12\log\left(2\pi\sigma_{iy}^2\right)\right].$$
       Show that the boundary between two classes is a quadratic surface
       without cross terms $x_ix_j$.
    1. Two classes in two dimensions have equal priors and the
       class-conditionals $\mathcal N(\mathbf 0,\boldsymbol\Sigma_+)$ and
       $\mathcal N(\mathbf 0,\boldsymbol\Sigma_-)$ with
       $\boldsymbol\Sigma_\pm=\left(\begin{smallmatrix}1&\pm\rho\\\pm\rho&1\end{smallmatrix}\right)$
       for some $0<\rho<1$. Find the Bayes classifier, and show that its
       class score depends on $\mathbf x$ only through the cross term
       $x_1x_2$ that (b) excludes. As for the XOR data of exercise
       :exref:`ex_mdl-naive-bayes-xor`, the two classes have the same
       marginals, so Gaussian naive Bayes is at chance even with unlimited
       data. Under $\mathcal N(\mathbf 0,\boldsymbol\Sigma_+)$,
       $P(x_1x_2>0)=\frac12+\frac1\pi\arcsin\rho$. Find the Bayes error as
       a function of $\rho$, and evaluate it for $\rho=0.1$ and $\rho=0.9$.

1. [code] **Smoothing and unseen pixels.** The `#naive-bayes-train` cell
   estimates $p(x_i{=}1\mid y)$ by $(n_{iy}+1)/(n_y+2)$.
    1. Without the smoothing, a pixel that is off in every training image of
       class $y$ sends the log-score :eqref:`eq_mdl-naive_bayes_log` of class
       $y$ to $-\infty$ whenever it is on in a test image. Use the counts
       `n_x` to find how many pixels of each class are never on in training.
       How many test images have a pixel on that is never on in the training
       images of their true class?
    1. Generalize the cell to a pseudocount $\alpha$, estimating
       $p(x_i{=}1\mid y)$ by $(n_{iy}+\alpha)/(n_y+2\alpha)$, and report the
       test accuracy for $\alpha\in\{0,0.01,0.1,1,10,100,1000\}$. In floating
       point, $0\cdot(-\infty)$ is NaN. Which values does `scores` return at
       $\alpha=0$, and which class does `predict` then return?
    1. Rewrite `scores` so that each pixel selects one of its two log
       probabilities, for example with `np.where`, instead of multiplying both
       by $x_i$ and $1-x_i$. Rerun $\alpha=0$. Which fraction of the class
       scores is $-\infty$, and for how many test images is every class score
       $-\infty$? Was the failure in (b) a property of the unsmoothed model or
       of its implementation?
    1. The estimate with pseudocount $\alpha$ is the posterior mean of
       $p(x_i{=}1\mid y)$ under a Beta prior (:numref:`subsec_mdl-beta-map`).
       Which prior is it? Why does the accuracy fall as $\alpha$ grows large?
   :exlabel:`ex_mdl-naive-bayes-suppose-we-omitted-laplace`

1. [code] **Missing pixels.** A generative model can classify an input of
   which some features are unobserved. Let $O\subseteq\{1,\ldots,d\}$ be the
   set of observed features and $\mathbf x_O$ their values.
    1. Show that under :eqref:`eq_mdl-naive_assumption` the posterior
       $p(y\mid\mathbf x_O)$ follows from :eqref:`eq_mdl-naive_bayes` by
       dropping the factors of the unobserved features. Why does a
       discriminative model of $p(y\mid\mathbf x)$ offer no such operation?
    1. Hide the top $14$ rows of every test image. Compare the test accuracy
       of the rule from (a) with that of setting the hidden pixels to $0$ and
       applying `predict`. Which assumption about the hidden pixels does the
       second rule make?

1. **Two event models for text.** A vocabulary $V$ has the four words
   *goal*, *match*, *vote* and *poll*. The training set contains three
   sports documents, "goal goal match", "goal match match" and "match vote",
   and three politics documents, "vote poll poll", "vote vote match" and
   "poll vote". The multinomial model treats a document as a sequence of
   independent word draws from a class-specific distribution over $V$. The
   Bernoulli model records for each word of $V$ only whether it occurs in the
   document.
    1. Estimate both models with Laplace smoothing. For the multinomial
       model, add $1$ to each word count of a class and $|V|$ to the class's
       total count. For the Bernoulli model, estimate the probability that a
       word occurs by $(n_{iy}+1)/(n_y+2)$, where $n_{iy}$ counts the
       class-$y$ documents that contain word $i$.
    1. Classify the document "goal match poll poll" with both models. Which
       words drive each decision?
    1. Concatenate a document with itself $k$ times. How do the log posterior
       odds of the two models change? In real text, a word that has occurred
       once in a document tends to occur again :cite:`Church.Gale.1995`, a
       finding that exercise :exref:`ex_mdl-distributions-word-burstiness`
       of :numref:`sec_mdl-distributions` reproduces. Which assumption of the
       multinomial model does this violate?
   :exlabel:`ex_mdl-naive-bayes-two-event-models`

1. [code] **Paired accuracy comparison.** Fit the model with the
   pseudocounts $\alpha=1$ and $\alpha=10$ of exercise
   :exref:`ex_mdl-naive-bayes-suppose-we-omitted-laplace`b, and record for
   each model which test images it classifies correctly.
    1. Does the test accuracy of the $\alpha=10$ model lie inside the
       bootstrap interval that the `#mdl-naive-bayes-calibration-1` cell
       computes for the $\alpha=1$ model?
    1. Compute a bootstrap interval of the accuracy difference, using the
       same resampled test indices for both models. Does it contain $0$?
       Count the test images on which the two models disagree. From this
       count and the accuracy difference, predict the width of the interval
       with the variance of the per-example differences $d_i$ of exercise
       :exref:`ex_mdl-statistics-paired-comparisons`a of
       :numref:`sec_mdl-statistics`, and compare the prediction with the
       bootstrap interval. Why is this interval much narrower than the
       interval of one accuracy?

1. [code] **Template overlap and confusions.** Flatten the ten smoothed
   templates in `P_xy` into vectors and compute the cosine similarity of every
   pair of classes. Rank the pairs by similarity, and rank them by the
   confusion count of the `#mdl-naive-bayes-calibration-2` cell, adding the
   errors in both directions of a pair. How well do the two rankings agree?
   Which highly similar pair is rarely confused?
   :exlabel:`ex_mdl-naive-bayes-check-template-overlap-explanation`

1. **Duplicated features.** Two classes have equal priors, and a single
   binary feature has $p(x=1\mid y=1)=0.8$ and $p(x=1\mid y=0)=0.2$. A data
   pipeline stores this feature $k$ times, so naive Bayes sees $k$ identical
   features and treats them as conditionally independent.
    1. Show that the log posterior odds of naive Bayes are $k$ times those of
       the correct model, while its predictions are those of the correct
       model.
    1. The correct model's predictions are right with probability $0.8$.
       Which confidence does naive Bayes claim for each of its predictions?
       Find the smallest $k$ for which this confidence is at least $99.9\%$.
    1. Show that dividing every class score by $T=k$ before the softmax
       restores the posterior of the correct model.
   :exlabel:`ex_mdl-naive-bayes-duplicated-features`

1. [code] **Label shift.** Under label shift
   (:numref:`sec_environment-and-distribution-shift`), the test prior $q(y)$
   differs from the training prior $p(y)$, while $p(\mathbf x\mid y)$ stays
   the same.
    1. Show that the posterior under the test prior is
       $q(y\mid\mathbf x)\propto p(y\mid\mathbf x)\,q(y)/p(y)$. Why does naive
       Bayes need no retraining to use it?
    1. Build a test set in which digit $4$ makes up $90\%$ of the images:
       keep all test images of digit $4$ and draw the others at random from
       the remaining digits. Compare the accuracy of `predict` with that of
       the corrected rule of (a), with $q(4)=0.9$ and $q(y)=1/90$ for every
       other digit.
   :exlabel:`ex_mdl-naive-bayes-label-shift`

1. [code] **Grayscale Gaussian naive Bayes.** ● The
   `#naive-bayes-load` cell discards grayscale information by thresholding.
   Reload the images without the threshold, scale the intensities to
   $[0,1]$, and fit Gaussian naive Bayes with a mean $\mu_{iy}$ and a
   variance $\sigma_{iy}^2$ for each class and pixel.
    1. How many class--pixel pairs have zero training variance? What does the
       log density of such a pixel do at test time? Add a floor $\epsilon$ to
       every variance and report the test accuracy for
       $\epsilon\in\{10^{-9},10^{-4},10^{-3},10^{-2},10^{-1}\}$. Which role
       of the pseudocount in the Bernoulli model does $\epsilon$ play?
    1. Replace the class-specific variances by one variance per pixel, the
       average of the ten class variances, so that the decision boundaries
       become linear. Repeat the sweep over $\epsilon$.
    1. Compare the best accuracy of (a) and (b) with the $84\%$ of the
       Bernoulli model. Within one class, choose a pixel whose mean intensity
       in that class lies near $1/2$, and plot the histogram of its
       intensities over the class's training images. How well does a
       Gaussian describe this histogram? What does that imply for the
       comparison?

1. [code] **Score gaps under the model.** ● Exercise
   :exref:`ex_mdl-naive-bayes-check-template-overlap-explanation` compares
   templates, which describe only the per-pixel means. For two classes $a$
   and $b$, let $g(\mathbf x)=s_a(\mathbf x)-s_b(\mathbf x)$ be the
   difference of their scores as computed by `scores`, so that the rule
   ranks $b$ above $a$ exactly when $g(\mathbf x)<0$. Write $\theta_{iy}$ for
   the entries of `P_xy`.
    1. By exercise :exref:`ex_mdl-naive-bayes-discussion-after-eqref-eq`a,
       the gap is affine, $g(\mathbf x)=\mathbf w^\top\mathbf x+c$, with the
       weights and offset of that exercise for the classes $a$ and $b$.
       Compute $\mathbf w$ and $c$ from `P_xy` and `log_P_y`, and display
       $\mathbf w$ for $a=4$ and $b=9$ as a $28\times28$ image. Which pixels
       decide between the two digits?
    1. Let $\mathbf x$ follow the model's own class-$a$ distribution, with
       independent pixels $x_i\sim\mathrm{Bernoulli}(\theta_{ia})$. Show that
       $E[g]$ equals the Kullback--Leibler divergence
       $\sum_{\mathbf x}p(\mathbf x\mid a)\log\bigl(p(\mathbf x\mid a)/p(\mathbf x\mid b)\bigr)$
       of the two class-conditional distributions of the model plus
       $\log\bigl(p(a)/p(b)\bigr)$, and that
       $\operatorname{Var}[g]=\sum_i w_i^2\,\theta_{ia}(1-\theta_{ia})$.
    1. Let $a=4$ and $b=9$. Compute $g$ on the test images of digit $4$, and
       compare its empirical mean and standard deviation with the two values
       of (b). From the model's mean and variance, use a Gaussian
       approximation to predict the fraction of these images with $g<0$, and
       compare it with the observed fraction.
    1. On real images, $\operatorname{Var}[g]=\mathbf w^\top\boldsymbol\Sigma_a\mathbf w$,
       where $\boldsymbol\Sigma_a$ is the covariance matrix of the pixels of
       class $a$. Which entries of $\boldsymbol\Sigma_a$ does the variance
       of (b) omit? Which property of handwritten strokes makes their
       contribution positive?
   :exlabel:`ex_mdl-naive-bayes-score-gaps`

1. [code] **Temperature scaling.** ● Exercise
   :exref:`ex_mdl-naive-bayes-duplicated-features` shows that dividing the
   class scores by $T=k$ undoes a $k$-fold duplication of features. For
   MNIST the effective duplication is unknown, and temperature scaling
   :cite:`Guo.Pleiss.Sun.Weinberger.2017` fits $T$ on held-out data instead:
   the posterior becomes the softmax of $s_y(\mathbf x)/T$.
    1. Refit the model on the first $50{,}000$ training images, and choose $T$
       by minimizing the average negative log-likelihood of the posterior on
       the remaining $10{,}000$. Why does $T$ not change the test accuracy?
    1. Repeat the reliability table of the `#mdl-naive-bayes-calibration-3`
       cell for the full-data model with the fitted $T$, using the bin edges
       $0$, $0.5$, $0.7$, $0.9$, $0.99$ and $1$. Compare the mean claimed
       confidence with the accuracy.
    1. In the setting of exercise
       :exref:`ex_mdl-naive-bayes-duplicated-features`, show that the
       standard deviation of the score gap on real data is $\sqrt k$ times
       the standard deviation that the model predicts. Compute the square of
       this ratio for the pair $(4,9)$ of exercise
       :exref:`ex_mdl-naive-bayes-score-gaps`c and for a few other confused
       pairs, and compare it with the fitted $T$.
    1. Apply the prior correction of exercise
       :exref:`ex_mdl-naive-bayes-label-shift`a to the temperature-scaled
       posterior, on the test set of exercise
       :exref:`ex_mdl-naive-bayes-label-shift`b. Why does the corrected prior
       change more predictions after temperature scaling?
   :exlabel:`ex_mdl-naive-bayes-temperature-scaling`

1. [extended] **Generative versus discriminative learning.** ●●
   :citet:`Ng.Jordan.2002` compare naive Bayes with logistic regression,
   which by exercise :exref:`ex_mdl-naive-bayes-discussion-after-eqref-eq`
   fits the same family of posteriors by maximizing the conditional
   likelihood. They prove that the naive Bayes parameters approach their
   population values after a number of examples logarithmic in $d$. For
   logistic regression, the number of examples needed to approach its
   asymptotic error is at most linear in $d$, and in the worst case it is
   also at least linear in $d$. On many of their
   $15$ datasets, naive Bayes has the lower test error for small training
   sets and logistic regression for large ones.
    1. Let each class have at least $m$ training examples. Use Hoeffding's
       inequality :eqref:`eq_mdl-hoeffding` and a union bound to show that
       all $2d$ unsmoothed estimates $\hat\theta_{iy}$ of a two-class model
       lie within $\varepsilon$ of their population values with probability
       at least $1-\delta$ when
       $$m\ge\frac{\log(4d/\delta)}{2\varepsilon^2}.$$
       Evaluate the bound for $d=784$, $\varepsilon=0.05$ and $\delta=0.05$.
    1. A uniform bound on $|\hat\theta_{iy}-\theta_{iy}|$ does not by itself
       bound the change of the class scores. Why not, for the many MNIST
       pixels with $\theta_{iy}$ close to $0$? What must be assumed about the
       $\theta_{iy}$ for the scores to converge at the rate of the
       parameters?
    1. Restrict the binarized MNIST data to the digits $4$ and $9$. For the
       total training-set sizes
       $n\in\{10,20,50,100,200,500,1000,2000,5000\}$ and for the full set,
       draw random training subsets that contain both digits, $20$ for each
       size up to $n=200$ and $5$ for each larger size. Fit naive Bayes with
       Laplace smoothing, and logistic regression without weight decay. For
       small $n$ the training data are linearly separable, and by exercise
       :exref:`ex_mdl-naive-bayes-discussion-after-eqref-eq`c the logistic
       loss has no minimizer, so run a fixed number of full-batch optimizer
       steps from zero initialization. Plot the average test errors of both
       classifiers against $n$ on a logarithmic axis. In which range of $n$,
       if any, does naive Bayes have the lower error?
    1. Repeat (c) with synthetic data for which the naive assumption holds:
       draw balanced labels and sample each pixel independently from
       $\mathrm{Bernoulli}(\theta_{iy})$, with the $\theta_{iy}$ estimated
       from all training images of the two digits. How do the two learning
       curves and their asymptotic errors change?
    1. Theorem 4 of :citet:`Ng.Jordan.2002` bounds the excess error of naive
       Bayes by the probability that the score gap of the population model
       falls near zero. Use the score-gap variances of exercise
       :exref:`ex_mdl-naive-bayes-score-gaps` to compare this probability for
       real pixels and for the synthetic data of (d). How does this
       comparison, together with the asymptotic errors of (c) and (d),
       account for the two pairs of learning curves?

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/418)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/1100)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/1101)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/1101)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §25.6]{.kicker}

The chapter's ideas at work in one system<br>**count, smooth, argmax, then check the result**.
:::
:::

::: {.slide title="Why not fit p(y | x) directly?"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
A classifier wants $\hat y = \operatorname{argmax}_y p(y\mid\mathbf x)$.
An unrestricted conditional table has $2^d$ binary feature patterns and is
impractical at MNIST scale ($d=784$).

::: {.d2l-note}
The plan: **maximum likelihood** (the maximum-likelihood section) fits a
generative model by counting; Bayes flips it into a classifier; the
**statistics** of the statistics section then judge the result: error bar,
failure map, calibration.
:::
:::

::: {.col .fig}
@fig:mdl-prob-naive-genvdisc
:::
:::
:::

::: {.slide title="Bayes' rule → a generative classifier"}
[The route]{.kicker}

Flip to the generative direction; the label-independent denominator drops:

$$\hat y = \operatorname*{argmax}_y\, p(y\mid\mathbf x)
= \operatorname*{argmax}_y\, p(\mathbf x\mid y)\,p(y).$$

. . .

We still owe a model for $p(\mathbf x\mid y)$, a distribution over all
$2^d$ patterns. No savings *yet*.
:::

::: {.slide title="Generative vs. discriminative"}
[Two routes]{.kicker}

::: {.cols .vc}
::: {.col}
- **Generative** (here): learn class-conditionals $p(\mathbf x\mid y)$ + prior, flip through Bayes.
- **Discriminative** (softmax / logistic): fit the boundary $p(y\mid\mathbf x)$ directly.

::: {.d2l-note}
In some well-specified comparisons, generative fitting approaches its
asymptotic error with fewer samples while discriminative fitting has lower
asymptotic error. This ordering is model dependent.
:::
:::

::: {.col .fig}
@fig:mdl-prob-naive-genvdisc
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[The naive assumption]{.dtitle}

[conditional independence, and why it helps]{.dsub}
:::
:::

::: {.slide title="The factorization"}
[Independence]{.kicker}

Assume features are **conditionally independent given the label**:

$$p(\mathbf x\mid y) = \prod_{i=1}^d p(x_i\mid y).$$

. . .

::: {.d2l-note .rule}
For binary features, this replaces $2^d-1$ class-conditional parameters per
class by $d$ Bernoulli parameters.
:::
:::

::: {.slide title="The graphical model"}
[Independence]{.kicker}

::: {.cols .vc}
::: {.col}
The label is a parent of every feature; no feature-to-feature edges appear in
the factorized model. The right panel illustrates conditional dependence that
the model does not represent.

For MNIST the assumption is inaccurate because pixels remain correlated within
a class. Classification can still be useful even when probabilities are
miscalibrated.
:::

::: {.col .fig .big}
@fig:mdl-prob-naive-independence
:::
:::
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Log space and linearity]{.dtitle}

[underflow, the log-sum, and Bernoulli decision boundaries]{.dsub}
:::
:::

::: {.slide title="Predict in log space"}
[Numerical]{.kicker}

Multiplying $784$ probabilities underflows: in single precision nearly
every class score becomes an exact zero. Since $\log$ is increasing it
preserves the argmax and turns the product into a sum:

$$\hat y = \operatorname*{argmax}_y\, \log p(y) +
\sum_{i=1}^d \log p(x_i\mid y).$$
:::

::: {.slide title="Bernoulli scores are affine"}
[Linearity]{.kicker}

For binary pixels, $\log p(x_i\mid y) = x_i\log p_{iy} + (1-x_i)\log(1-p_{iy})$,
so each class score is

$$s_y(\mathbf x) = \mathbf w_y^\top\mathbf x + b_y.$$

::: {.d2l-note .rule}
Naive Bayes draws the **same kind of decision hyperplanes** as softmax
regression; only the way it fits the weights differs.

Gaussian naive Bayes is linear only when each feature variance is shared across
classes; class-specific variances generally give quadratic boundaries.
:::
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[Training is counting]{.dtitle}

[priors, likelihoods, and Laplace smoothing]{.dsub}
:::
:::

::: {.slide title="Maximum likelihood = frequencies"}
[Counting]{.kicker}

Both ingredients are counts: the class prior $\hat p(y)=n_y/n$ and the
per-pixel firing rate, Laplace-smoothed as
$\hat p(x_i{=}1\mid y) = \tfrac{n_{iy}+1}{n_y+2}$ so a never-seen pixel
cannot send a log-score to $-\infty$:

@naive-bayes-train

::: {.d2l-note}
The $+1/+2$ is the posterior mean under a $\text{Beta}(1,1)$ prior:
pseudo-observations as regularization.
:::
:::

::: {.slide}
::: {.divider}
[04]{.dnum}

[MNIST, end to end]{.dtitle}

[binarize, learn templates, 84% accuracy]{.dsub}
:::
:::

::: {.slide title="Load and binarize"}
[MNIST]{.kicker}

Threshold each $28\times28$ grayscale image at $128$ to get binary pixels
$x_i\in\{0,1\}$:

@naive-bayes-load

A Gaussian naive Bayes would instead keep a per-class mean and variance
for each continuous pixel.
:::

::: {.slide title="What the model learns"}
[MNIST]{.kicker}

Each class is just an averaged template. The blur
**is** the naive assumption: per-pixel marginals, nothing about
co-occurrence.

@naive-bayes-templates
:::

::: {.slide title="Classify and evaluate"}
[MNIST]{.kicker}

Sum the log-likelihoods, take the argmax, and measure what the *raw*
products would have done:

@!naive-bayes-predict

::: {.d2l-note .rule}
**84.27%** on this implementation: far above $10\%$ chance and far below
modern image classifiers. Conditional independence and binarization are both
important model limitations.
:::
:::

::: {.slide title="A common domain: text"}
[Domain]{.kicker}

Pixels are tightly coupled. In a bag-of-words representation, the factorized
model is often a useful and inexpensive baseline, including for spam filtering.

::: {.d2l-note}
The multinomial event model counts word occurrences (with a $+|V|$
denominator) instead of presence/absence.
:::
:::

::: {.slide}
::: {.divider}
[05]{.dnum}

[Evaluation and uncertainty]{.dtitle}

[error bar · failure map · calibration]{.dsub}
:::
:::

::: {.slide title="How precise is 84.27%?"}
[Error bar]{.kicker}

Accuracy on 10,000 test examples is an estimate of population accuracy.
A bootstrap interval quantifies its sampling uncertainty by resampling the test
set and recomputing the statistic:

@!mdl-naive-bayes-calibration-1

The 95% bootstrap interval is summarized as $84.3\%\pm0.7$ percentage points,
so reporting a third decimal is not meaningful. Comparing two models on the
same test set requires a paired analysis; overlapping marginal intervals do not
by themselves imply that their difference is insignificant.
:::

::: {.slide title="Classwise errors" layout="tight"}
[A confusion matrix identifies systematic confusions]{.kicker}

@!mdl-naive-bayes-calibration-2

The largest confusions ($4\to9$, $5\to3$, and $8\to3$) occur among classes
with similar learned templates. Distinguishing them depends partly on joint
patterns among neighboring pixels, which the marginal feature model omits.
:::

::: {.slide title="Calibration of posterior scores"}
[Calibration]{.kicker}

After normalization, the class scores define posterior probabilities under the
fitted model. Calibration compares these probabilities with empirical
frequencies by binning examples according to predicted confidence:

@!mdl-naive-bayes-calibration-3

The mean predicted confidence is $98.6\%$, while accuracy is $84.3\%$.
In the highest-confidence bin, predictions are correct $89.9\%$ of the time.

::: {.d2l-note .warn}
The model treats 784 correlated pixels as conditionally independent and
therefore overcounts redundant evidence. The median gap between the largest
and second-largest class scores is about 30 nats, enough to make the normalized
posterior nearly degenerate. Probability estimates intended for downstream use
should be checked with a calibration analysis.
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

::: {.cols}
::: {.col}
- Naive Bayes combines Bayes' rule with conditional feature independence.
- For binary features, the factorization reduces an unrestricted
  $2^d-1$-parameter class-conditional distribution to $d$ Bernoulli parameters
  per class.
- Parameter estimates are smoothed empirical counts, and prediction is
  performed in log space.
- Bernoulli and multinomial scores are affine; Gaussian boundaries are linear
  only under shared per-feature variances and otherwise quadratic.
:::

::: {.col}
- The method is often useful for sparse text features but is limited on images
  by conditional dependence and the chosen pixel likelihood.
- This experiment obtains $84.3\%\pm0.7$ accuracy. The confusion matrix
  identifies similar templates, and the calibration curve shows substantial
  overconfidence.
:::
:::

::: {.d2l-note}
An accuracy report is more informative when accompanied by uncertainty,
classwise errors, and a calibration check.
:::
:::
