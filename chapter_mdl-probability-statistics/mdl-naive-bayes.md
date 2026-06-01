# Naive Bayes
:label:`sec_mdl-naive_bayes`

The maximum-likelihood principle of :numref:`sec_mdl-maximum_likelihood` tells us how to fit a probabilistic model; here we put it to work on a *classifier*. The **naive Bayes** classifier is the simplest thing that deserves the name "learning": it estimates a probability model by counting, predicts with Bayes' rule, and survives the curse of dimensionality through one bold --- and visibly wrong --- assumption. It is the cleanest place to watch probability turn into a working algorithm, so we build it end to end and run it on handwritten digits.

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

A classifier maps an example $\mathbf{x}\in\mathbb{R}^d$ to a label $y\in\{1,\ldots,n\}$. The probabilistically honest target is the *posterior* $p(y\mid\mathbf{x})$ --- how plausible each label is given what we observed --- and the natural prediction is its most likely value,

$$\hat{y} = \mathop{\mathrm{argmax}}_y \, p(y\mid\mathbf{x}).$$

Estimating $p(y\mid\mathbf{x})$ directly is hopeless. With $d$ binary features there are $2^d$ distinct inputs $\mathbf{x}$; storing a label distribution for each would need on the order of $2^d$ numbers, and we would have to *see* most of those inputs to estimate them. With the $784$ pixels of an MNIST image that is $2^{784}$ patterns --- vastly more than there are atoms in the universe. Memorizing one answer per input is not learning; it is a lookup table we can never fill.

Bayes' rule turns the problem around. Instead of modelling "which label, given this image," we model "which images, given this label" --- the *generative* direction:

$$\hat{y} = \mathop{\mathrm{argmax}}_y \, p(y\mid\mathbf{x}) = \mathop{\mathrm{argmax}}_y \, \frac{p(\mathbf{x}\mid y)\,p(y)}{p(\mathbf{x})} = \mathop{\mathrm{argmax}}_y \, p(\mathbf{x}\mid y)\,p(y).$$

The denominator $p(\mathbf{x})$ is the same for every $y$, so it cannot change which label wins the $\mathrm{argmax}$ and we drop it; in shorthand, $p(y\mid\mathbf{x}) \propto p(\mathbf{x}\mid y)\,p(y)$. (Should we ever want the actual posterior probabilities, normalizing the numerators so they sum to one recovers $p(\mathbf{x})$ for free.) This has not yet bought us anything: the *class-conditional* $p(\mathbf{x}\mid y)$ is a distribution over the same $2^d$ patterns. The chain rule lays the difficulty bare,

$$p(\mathbf{x}\mid y) = p(x_1\mid y)\,p(x_2\mid x_1, y)\cdots p(x_d\mid x_1,\ldots,x_{d-1}, y),$$

a product whose later factors condition on ever-longer histories and still hide $2^d$ parameters.

### The Naive Assumption

Here is the leap. *Assume the features are conditionally independent given the label.* Then every history drops out and the product collapses to one factor per feature:

$$p(\mathbf{x}\mid y) = \prod_{i=1}^d p(x_i\mid y).$$
:eqlabel:`eq_mdl-naive_assumption`

This is the assumption that makes Bayes "naive," and it is *false*: in a real digit the pixels are strongly correlated --- an inked pixel makes its neighbors far more likely to be inked too. The model pretends each pixel is painted by an independent coin flip whose bias depends only on the digit. Yet the assumption is wonderfully cheap. We no longer estimate one giant joint distribution; we estimate $d$ tiny one-feature distributions $p(x_i\mid y)$ per class --- only $\mathcal{O}(dn)$ numbers instead of $\mathcal{O}(2^d n)$. The curse of dimensionality is broken by fiat.

Why tolerate a false assumption? Because the classifier does not need the probabilities to be *right* --- it only needs the largest one to land on the correct label. A model can be badly miscalibrated (massively over- or under-confident, as multiplying $784$ near-independent factors will be) and still pick the right winner. Naive Bayes routinely classifies well even where its independence story is plainly wrong, and the savings let it learn from a modest dataset rather than an impossible one.

**Proposition (Naive Bayes classifier).** Under the conditional-independence assumption :eqref:`eq_mdl-naive_assumption`, the predicted label is

$$\hat{y} = \mathop{\mathrm{argmax}}_y \; p(y) \prod_{i=1}^d p(x_i\mid y).$$
:eqlabel:`eq_mdl-naive_bayes`

*Proof.* Substitute the factorized class-conditional :eqref:`eq_mdl-naive_assumption` into the Bayes-rule predictor $\hat{y} = \mathop{\mathrm{argmax}}_y p(\mathbf{x}\mid y)\,p(y)$, having already discarded the label-independent denominator $p(\mathbf{x})$. $\blacksquare$

### Doing It in Log Space

Equation :eqref:`eq_mdl-naive_bayes` multiplies $d$ probabilities, each in $[0,1]$. For $d=784$ this product underflows to a hard zero in floating point long before the $\mathrm{argmax}$ can compare anything --- the practical face of the numerical issue we met in :numref:`sec_mdl-maximum_likelihood`. The fix is the same: the $\mathrm{argmax}$ is unchanged by the increasing map $\log$, and $\log$ turns the product into a sum.

**Proposition (log-space form).** The naive Bayes prediction :eqref:`eq_mdl-naive_bayes` equals

$$\hat{y} = \mathop{\mathrm{argmax}}_y \; \log p(y) + \sum_{i=1}^d \log p(x_i\mid y).$$
:eqlabel:`eq_mdl-naive_bayes_log`

*Proof.* Apply $\log$ to the objective of :eqref:`eq_mdl-naive_bayes`; since $\log$ is strictly increasing it preserves the maximizer, and $\log(ab)=\log a + \log b$ converts the product to the sum. $\blacksquare$

A score is now a sum of $785$ well-behaved logarithms instead of a product that rounds to zero --- the classifier becomes one matrix of pre-computed log-probabilities and a few additions.

## Training Is Counting

Naive Bayes needs two ingredients: the class prior $p(y)$ and, for each class, the per-feature likelihoods $p(x_i\mid y)$. Both are estimated by **maximum likelihood**, and for the categorical and Bernoulli models here the MLE is just an empirical frequency --- *training is counting*.

For the prior, the maximum-likelihood estimate of $p(y)$ is the fraction of training examples carrying label $y$: if class $y$ appears $n_y$ times in $n = \sum_y n_y$ examples, then $\hat p(y) = n_y / n$. For binary features, $p(x_i = 1\mid y)$ is the probability that feature $i$ fires for class $y$ --- a Bernoulli parameter whose MLE is again a frequency: of the $n_y$ examples in class $y$, the fraction in which feature $i$ is on. Storing $\hat p(x_i = 1 \mid y)$ in a matrix $P_{xy}$ fixes both Bernoulli outcomes, since $\hat p(x_i = 0\mid y) = 1 - P_{xy}[i,y]$.

One hazard remains. If feature $i$ is *never* on for class $y$ in the training set, the MLE is $\hat p(x_i=1\mid y)=0$, and a single such feature at test time annihilates the whole product in :eqref:`eq_mdl-naive_bayes` (and sends :eqref:`eq_mdl-naive_bayes_log` to $-\infty$). The cure is **Laplace smoothing**: add a pseudocount, estimating $p(x_i=1\mid y)$ as $(n_{iy}+1)/(n_y+2)$ rather than $n_{iy}/n_y$. The $+2$ in the denominator covers the two outcomes a binary pixel can take. This is no ad-hoc patch: it is the MAP estimate under a uniform Beta prior --- a single phantom "on" and "off" observation per feature, exactly the prior-as-regularizer story of MAP estimation in :numref:`sec_mdl-maximum_likelihood`.

## A Worked Example: MNIST Digits

We classify handwritten digits from MNIST :cite:`LeCun.Bottou.Bengio.ea.1998` --- $28\times 28$ grayscale images of the digits $0$ through $9$. To make each pixel a binary feature we threshold it at half-intensity, so $x_i\in\{0,1\}$ records whether pixel $i$ is inked. We load the two splits and binarize, ending with plain NumPy arrays so the estimation below is identical across frameworks: `X` of shape `(n, 28, 28)` and integer labels `Y`.

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
    ds = torchvision.datasets.MNIST(root='./temp', train=train, download=True)
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

Training is two counts, done in NumPy. The prior $\hat p(y)$ is the class frequency; the likelihood matrix $P_{xy}$ holds, for each class, the Laplace-smoothed fraction of its images in which each pixel is inked. From here on the code is framework-agnostic --- the model is pure counting --- and prediction works in log space.

```{.python .input #naive-bayes-train}
n_y = np.array([(Y == y).sum() for y in range(10)])
P_y = n_y / n_y.sum()                                  # class prior, p(y)

n_x = np.stack([X[Y == y].sum(axis=0) for y in range(10)])
P_xy = (n_x + 1) / (n_y + 2).reshape(10, 1, 1)         # Laplace-smoothed p(x_i=1|y)
P_y
```

Each row of $P_{xy}$ is a $28\times 28$ image of per-pixel "on" probabilities for one digit --- a picture of *exactly* what the model believes a class looks like. Plotting all ten shows ghostly averaged digits: this is the entire learned model.

```{.python .input #naive-bayes-templates}
d2l.show_images([P_xy[y] for y in range(10)], 2, 5,
                titles=[str(y) for y in range(10)]);
```

The blur is the naive assumption made visible: the model knows each pixel's marginal firing rate but nothing about how pixels co-occur, so every sharp stroke smears into an average.

### Classifying and Evaluating

Prediction sums log-likelihoods per :eqref:`eq_mdl-naive_bayes_log`. We precompute $\log P_{xy}$, $\log(1-P_{xy})$, and $\log P_y$, then score a batch of images at once: a pixel that is on contributes $\log P_{xy}$, a pixel that is off contributes $\log(1-P_{xy})$, and the prior is added in. The winning class is the $\mathrm{argmax}$.

```{.python .input #naive-bayes-predict}
log_P_xy, log_P_xy_neg = np.log(P_xy), np.log(1 - P_xy)
log_P_y = np.log(P_y)

def predict(X):
    X = X.reshape(-1, 1, 28, 28)                       # (m, 1, 28, 28)
    scores = (X * log_P_xy + (1 - X) * log_P_xy_neg).reshape(
        len(X), 10, -1).sum(axis=2) + log_P_y
    return scores.argmax(axis=1)

float((predict(X_test) == Y_test).mean())              # Test accuracy
```

Naive Bayes lands around $84\%$ accuracy --- far above the $10\%$ of random guessing, from a model that is nothing but ten averaged templates and a counting pass over the data. Yet modern networks reach error rates below $1\%$, and the gap is precisely the price of the naive assumption: pixels in a real digit are emphatically *not* independent given the class, and pretending otherwise leaves a great deal on the table. That honest failure is the lesson. Naive Bayes shows how far a clean probabilistic idea and a single counting pass can take you --- and exactly where a wrong independence assumption stops you. (On problems where features genuinely are close to conditionally independent, such as bag-of-words text classification, the same classifier is a strong baseline, which is why it ruled spam filtering for decades.)

## Summary

* Bayes' rule recasts classification generatively: $p(y\mid\mathbf{x}) \propto p(\mathbf{x}\mid y)\,p(y)$, predicting the label that maximizes the numerator.
* The **naive** conditional-independence assumption $p(\mathbf{x}\mid y)=\prod_i p(x_i\mid y)$ is false but cheap --- it slays the curse of dimensionality, replacing $\mathcal{O}(2^d)$ parameters with $\mathcal{O}(d)$ --- and the classifier needs only the $\mathrm{argmax}$, not the probabilities, to be right.
* Working in log space (sums of log-likelihoods) avoids the underflow of multiplying hundreds of probabilities.
* Training is maximum likelihood by counting: class priors and per-feature frequencies, with Laplace smoothing as a tiny uniform (MAP) prior.
* On MNIST it learns ten averaged digit templates and classifies respectably, but its independence assumption caps accuracy --- a clean illustration of a generative classifier and of the cost of a wrong model.

## Exercises
1. Consider the dataset $\{(0,0),(0,1),(1,0),(1,1)\}$ with labels given by the XOR of the two coordinates, $\{0,1,1,0\}$. Compute the naive Bayes estimates $p(y)$ and $p(x_i\mid y)$. Does the classifier separate the points? If not, which assumption is violated?
2. Suppose we omitted Laplace smoothing and, at test time, an example contained a feature value never observed for some class in training. What would the model's log-score for that class be, and why?
3. The naive Bayes classifier is a special case of a Bayesian network, in which dependencies among random variables are encoded by a graph (see :citet:`Koller.Friedman.2009`). Explain why adding an explicit edge between the two inputs of the XOR model would let it classify the points correctly.

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

::: {.slide title="Naive Bayes: Bayes' rule for classification"}
Predict the most likely label given the features. Estimating
$p(y \mid \mathbf{x})$ directly is hopeless --- $2^d$ inputs --- so go
*generative* with Bayes' rule:

$$p(y \mid \mathbf{x}) \;\propto\; p(\mathbf{x} \mid y)\, p(y).$$

We still owe a model for $p(\mathbf{x} \mid y)$, a distribution over all
$2^d$ feature patterns.
:::

::: {.slide title="The naive assumption"}
Assume features are **conditionally independent given the label**:

$$p(\mathbf{x} \mid y) = \prod_{i=1}^d p(x_i \mid y).$$

Wrong --- image pixels are obviously correlated --- but cheap:
$\mathcal{O}(d)$ parameters, not $\mathcal{O}(2^d)$. And the classifier
only needs the *argmax* to be right, not the probabilities. Hence

$$\hat{y} = \mathop{\mathrm{argmax}}_y \; p(y) \prod_{i=1}^d p(x_i \mid y).$$
:::

::: {.slide title="Predict in log space"}
A product of $784$ probabilities underflows to zero. $\log$ preserves
the argmax and turns the product into a sum:

$$\hat{y} = \mathop{\mathrm{argmax}}_y \; \log p(y)
   + \sum_{i=1}^d \log p(x_i \mid y).$$
:::

::: {.slide title="Training is counting"}
Both ingredients are maximum-likelihood frequencies: the class prior
$p(y)$ and the per-pixel firing rate $p(x_i = 1 \mid y)$. Laplace
smoothing $(n_{iy}+1)/(n_y+2)$ is a tiny uniform prior (MAP) that keeps
log-probabilities finite.

@naive-bayes-train
:::

::: {.slide title="What the model learns"}
Each class is just an averaged template --- ten blurry digits. The blur
*is* the naive assumption: per-pixel rates, no pixel co-occurrence.

@naive-bayes-templates
:::

::: {.slide title="Classify and evaluate"}
Score by summing log-likelihoods, take the argmax. About $84\%$ on
MNIST --- well above chance, but capped by the independence assumption
(modern nets: $<1\%$ error). A great teaching classifier, honest about
its own failure.

@naive-bayes-predict
:::

::: {.slide title="Recap"}
- Bayes' rule + conditional independence $=$ naive Bayes.
- Breaks the curse of dimensionality: $\mathcal{O}(d)$ not
  $\mathcal{O}(2^d)$ parameters.
- Training is one counting pass (MLE); smooth, then predict in logs.
- Strong baseline where features really are near-independent (text);
  bad on images, where they are not.
:::
