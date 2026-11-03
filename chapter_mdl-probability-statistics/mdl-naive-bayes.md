# Naive Bayes
:label:`sec_mdl-naive_bayes`

In :numref:`sec_softmax_concise` we trained softmax regression on
Fashion-MNIST. It found 7,850 numbers, one weight for each pixel and class
plus ten offsets, by gradient descent on the probability of the label given
the image. It reached about 83% accuracy, and its probabilities passed a
reliability check. Gradient descent is not the only way to obtain such
numbers. We could instead describe how each class produces its images, what
a typical coat or sandal looks like, and let Bayes' rule turn that
description into a classifier. Can this be done, and what does it cost?

The obstacle is size. A description of how coats look must give a
probability to every possible black-and-white picture. There are
far more such pictures than atoms in the universe, and we have 60,000
training images. The *naive Bayes* classifier removes the obstacle with one
assumption: once the class is known, the features tell us nothing about each
other.

The idea goes back to Maron's work on sorting documents by the words they
contain :cite:`Maron.1961`, and the spam filters of the 1990s were built on
it :cite:`Sahami.Dumais.Heckerman.ea.1998`. Under the assumption, a filter
needs only how often each word appears in spam and in ordinary mail, and a
garment classifier only how often each pixel is inked in each class.
Training is counting, and prediction is multiplying.

Naive Bayes is the simplest *generative* classifier, one built from a model
of how each class produces its inputs. It is fitted in one pass without an
optimizer, and it can ignore a missing feature by dropping its factor. It
also shows more clearly than any other model what a false independence
assumption costs. The classes are
still ranked usefully, but the model's confidence is inflated, because
correlated evidence is counted many times. This section puts the chapter's
tools to work on one model: maximum likelihood fits it, the Beta posterior
smooths it, the bootstrap puts an error bar on its accuracy, and a
reliability check exposes its overconfidence.

After the images are loaded, the code is plain NumPy, the same in every
framework tab; the framework supplies only the data loader.

```{.python .input #naive-bayes-imports}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import npx
import numpy as np
npx.set_np()
d2l.use_svg_display()
```

```{.python .input #naive-bayes-imports}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
d2l.use_svg_display()
```

```{.python .input #naive-bayes-imports}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
d2l.use_svg_display()
```

```{.python .input #naive-bayes-imports}
#@tab jax
%matplotlib inline
from d2l import jax as d2l
import numpy as np
d2l.use_svg_display()
```

## Bayes' Rule for Classification

A classifier needs a rule that maps an image to a label. This part derives
such a rule from a model of how each class produces its images, and works it
by hand on a two-word spam filter.

### The Generative Route

A classifier assigns a label $y\in\{1,\ldots,K\}$ to an input $\mathbf x$.
If we knew the posterior $p(y\mid\mathbf x)$, the probability of each label
given the input, the guess that is right most often would be the most
probable label,

$$\hat y=\operatorname*{argmax}_y\,p(y\mid\mathbf x).$$

For Fashion-MNIST $K=10$. If each of the 784 pixels is recorded as inked or
blank, $\mathbf x$ is one of $2^{784}$ patterns, and since
$784\log_{10}2=236.0$ that is about $10^{236}$. A table of
$p(y\mid\mathbf x)$ would need a row for every pattern, and 60,000 training
images fill almost none of them.

Bayes' rule (:numref:`sec_mdl-random_variables`) writes the posterior as
$p(y\mid\mathbf x)=p(\mathbf x\mid y)\,p(y)/p(\mathbf x)$. The denominator
is the same for every class, so it cannot change which class is largest:

$$\hat y=\operatorname*{argmax}_y\,p(\mathbf x\mid y)\,p(y).$$

The classifier compares, for each class, how probable the image is under
that class times how common the class is. We call the product
$p(\mathbf x\mid y)\,p(y)$ the *Bayes numerator*. Dividing the $K$
numerators by their sum restores the denominator and gives back the
posterior.

A *generative classifier* models how the inputs of each class arise, the
class-conditional distribution $p(\mathbf x\mid y)$, together with the prior
$p(y)$, and then applies Bayes' rule. A *discriminative classifier* models
$p(y\mid\mathbf x)$ directly and never says how the inputs arise; softmax
regression (:numref:`sec_softmax`) is one.

A reader may object that Bayes' rule has saved nothing: $p(\mathbf x\mid y)$
is still a distribution over $2^{784}$ patterns. The objection is correct,
and the chain rule of probability shows where the size comes from. Written
out one feature at a time,

$$p(\mathbf x\mid y)=p(x_1\mid y)\,p(x_2\mid x_1,y)\cdots p(x_d\mid x_1,\ldots,x_{d-1},y).$$

Each factor conditions on the whole history of earlier features, and the
last one alone is a table over $2^{d-1}$ histories. The next subsection
removes the histories.

### The Naive Assumption

We need a model of $p(\mathbf x\mid y)$ with few enough numbers to estimate.
The naive assumption says, in words: once we know the garment, learning that
one pixel is inked tells us nothing more about any other pixel. In symbols,
the features are *conditionally independent given the label*, so that every
factor of the chain rule forgets its history:

$$p(\mathbf x\mid y) = \prod_{i=1}^d p(x_i\mid y).$$
:eqlabel:`eq_mdl-naive_assumption`

The distribution of a whole image is the product of $d$ one-pixel
distributions. For binary pixels each factor is fixed by one number, the
probability that the pixel is inked in that class. An unrestricted
$p(\mathbf x\mid y)$ has $2^{784}-1$ free probabilities per class; the
factorized one has $784$. Each of these 784 numbers is a frequency that
6,000 training images per class can estimate well. Both models need $K-1=9$
further numbers for the prior.

The assumption is about all $d$ features at once, and checking pairs does not
establish it. Within one class, let $x_1$ and $x_2$ be independent fair bits
and let $x_3=x_1\oplus x_2$ be $1$ when exactly one of them is $1$. Any two
of the three features are independent, yet the third is determined by the
other two. The pattern $(1,1,1)$ therefore has probability $0$, while the
product of the three marginals assigns it $\tfrac18$
(exercise :exref:`ex_mdl-naive-bayes-xor`).

Drawn as a graph, with an arrow from each variable to the variables it
directly influences, the factorization is a star: the label is a parent of
every feature, and no edges join the features
(:numref:`fig_mdl-naive-independence`, left). A model that keeps some
dependence between features adds edges among them (right).

![Conditional-independence structures. Left: in naive Bayes the label $y$ is a parent of each feature and no edges join the features, which represents $p(\mathbf x\mid y)=\prod_i p(x_i\mid y)$. Right: a model that keeps conditional dependence between features adds edges among them.](../img/mdl-prob-naive-independence.svg)
:label:`fig_mdl-naive-independence`

Substituting the factorization :eqref:`eq_mdl-naive_assumption` into Bayes'
rule gives the *naive Bayes rule* :cite:`Maron.1961`:

$$\hat{y} = \operatorname*{argmax}_y \; p(y) \prod_{i=1}^d p(x_i\mid y).$$
:eqlabel:`eq_mdl-naive_bayes`

The classifier multiplies the prior by one factor per feature and picks the
class with the largest product.

The assumption is false for images. Neighbouring pixels of a garment are
inked together or blank together, because they belong to the same sleeve or
the same hem. Treating them as independent evidence counts the same evidence
several times. We will see that this pushes the model's posterior toward $0$
and $1$, far beyond what its accuracy justifies, while the ranking of the
classes stays useful :cite:`Domingos.Pazzani.1997`.

### A Two-Word Spam Filter

Before turning to 784 pixels, we work the smallest case to the end. A mail filter
looks at two features: whether a mail contains the word "free" and whether
it contains "meeting". Its training set has ten mails, 4 spam and 6
ordinary, with these counts:

| | spam (4 mails) | ordinary (6 mails) |
|---|---|---|
| contains "free" | 3 | 1 |
| contains "meeting" | 0 | 4 |

Counting gives every number the model needs. The prior is
$\hat p(\text{spam})=4/10=0.4$. "Free" appears in $3/4$ of the spam and in
$1/6$ of the ordinary mails; "meeting" appears in $0/4$ of the spam and in
$4/6$ of the ordinary mails. A new mail contains "free" and not "meeting".
Its two Bayes numerators are

$$\text{spam: } 0.4\cdot\tfrac34\cdot\bigl(1-\tfrac04\bigr)=0.30,\qquad
\text{ordinary: } 0.6\cdot\tfrac16\cdot\bigl(1-\tfrac46\bigr)=\tfrac1{30}\approx0.033,$$

so the posterior probability of spam is $0.30/(0.30+0.033)=0.90$. The same
computation reads more clearly as odds. Each feature multiplies the prior
odds by its likelihood ratio, the probability of what we saw under spam
divided by its probability under ordinary mail:

$$\frac{p(\text{spam}\mid\mathbf x)}{p(\text{ordinary}\mid\mathbf x)}
=\underbrace{\frac{0.4}{0.6}}_{\text{prior}}\cdot
\underbrace{\frac{3/4}{1/6}}_{\text{"free"}}\cdot
\underbrace{\frac{1}{2/6}}_{\text{no "meeting"}}
=\frac23\cdot4.5\cdot3=9.$$

Odds of $9$ to $1$ are a probability of $9/10$. The 784-pixel classifier
below is this computation with 784 likelihood ratios instead of two. The
filter will return three times: when a count is zero, when we take
logarithms, and when a feature is counted twice.

## Learning by Counting

The naive Bayes rule needs the prior and one probability per class and
feature. This part estimates them, protects them against zero counts, and
rewrites the rule as a sum that a computer can evaluate and that is linear
in the pixels.

### Maximum Likelihood and Smoothing

The numbers of the spam filter were frequencies. Maximum likelihood
justifies this choice and also shows where it fails. Write
$\theta_{iy}=p(x_i=1\mid y)$ for the probability that feature $i$ is on in
class $y$. Suppose class $y$ occurs $n_y$ times among $n$ training examples,
and feature $i$ is on in $n_{iy}$ of them. Under the naive model the
log-likelihood of the training set splits into one term for the prior and
one term per class and feature:

$$\sum_y n_y\log p(y)+\sum_{y}\sum_{i=1}^d\Bigl[n_{iy}\log\theta_{iy}+(n_y-n_{iy})\log(1-\theta_{iy})\Bigr].$$

Each bracket is the log-likelihood of $n_{iy}$ heads in $n_y$ flips of a coin
with bias $\theta_{iy}$, so each parameter is fitted on its own. The
maximum-likelihood estimate of a coin's bias is its observed frequency
(:numref:`sec_mdl-maximum_likelihood`), which gives
$\hat\theta_{iy}=n_{iy}/n_y$ and $\hat p(y)=n_y/n$.

Frequencies have one hazard. If feature $i$ is never on in the training
examples of class $y$, then $\hat\theta_{iy}=0$, and every new example with
feature $i$ on gets a Bayes numerator of $0$ for class $y$, whatever its
other features say. One feature vetoes the class. The spam filter has such a
count: "meeting" never occurred in spam, so a mail containing "free" and
"meeting" has spam numerator $0.4\cdot\frac34\cdot0=0$ and is declared
ordinary with certainty. The training cell below counts how often a pixel
of Fashion-MNIST is never inked in a class.

The usual fix is *Laplace smoothing* :cite:`Laplace.1814`: add one imagined
example with the feature on and one with it off,

$$\hat\theta_{iy}=\frac{n_{iy}+1}{n_y+2}.$$

The estimate moves a little toward $\tfrac12$ and is never $0$ or $1$. For
the coin of :numref:`sec_mdl-exponential-family`, 9 heads in 13 flips give $9/13\approx0.69$ by counting
and $10/15\approx0.67$ smoothed. A coin that showed 0 heads in 13 flips is
"never heads" by counting and has probability $1/15$ smoothed.

The smoothed estimate is a Bayesian one. Under a uniform prior on $\theta$,
$n_{iy}$ successes in $n_y$ trials give the posterior
$\mathrm{Beta}(n_{iy}+1,\,n_y-n_{iy}+1)$, whose mean is
$(n_{iy}+1)/(n_y+2)$ (:numref:`sec_mdl-exponential-family`); Laplace called
this the rule of succession.

Smoothing the spam filter changes its estimates to $4/6$ and $1/6$ for
"free" and "meeting" in spam, and to $2/8$ and $5/8$ in ordinary mail; we
leave the prior at $0.4$. The mail with both words now has the numerators
$0.4\cdot\frac46\cdot\frac16=0.044$ and $0.6\cdot\frac28\cdot\frac58=0.094$,
a spam probability of $0.32$ instead of $0$. The first mail, "free" without
"meeting", drops from $0.90$ to $0.80$, because four spam mails are weak
evidence that spam never mentions a meeting.

### Scores in Log Space

The naive Bayes rule multiplies $d+1$ numbers between $0$ and $1$, and for
an image the product is tiny. If every factor were $\tfrac12$,
the product of 784 of them would be $2^{-784}\approx10^{-236}$. The smallest
positive single-precision number is about $1.4\cdot10^{-45}$ and the smallest
positive double about $4.9\cdot10^{-324}$, so the product underflows to $0$
in single precision and comes close to the floor of double precision.

The logarithm turns the product into a sum. Because $\log$ is increasing, it
does not change which class is largest
(:numref:`sec_mdl-numerical-stability-conditioning` discusses this
precaution in general). Define the *class score* $s_y(\mathbf x)$ as the
logarithm of the Bayes numerator. The rule becomes

$$\hat{y} = \operatorname*{argmax}_y \; s_y(\mathbf x),\qquad s_y(\mathbf x)=\log p(y) + \sum_{i=1}^d \log p(x_i\mid y).$$
:eqlabel:`eq_mdl-naive_bayes_log`

A sum of 785 logarithms replaces the product of 785 probabilities, and every
term is finite as long as no estimate is $0$, which smoothing guarantees.
For the unsmoothed spam filter the scores of the first mail are
$\log0.30=-1.20$ and $\log\frac1{30}=-3.40$; their gap,
$2.20=\log 9$, is the logarithm of the odds we computed by hand.

### Linear Decision Boundaries

With binary features the class score has a simple form, and that form tells
us which classifiers naive Bayes can represent. A binary feature contributes
$\log\theta_{iy}$ when it is on and $\log(1-\theta_{iy})$ when it is off, so
$\log p(x_i\mid y)=x_i\log\theta_{iy}+(1-x_i)\log(1-\theta_{iy})$. Collecting
the terms that multiply $x_i$ and the terms that do not,

$$
s_y(\mathbf x)=\mathbf w_y^\top\mathbf x+b_y,\qquad
w_{iy}=\log\frac{\theta_{iy}}{1-\theta_{iy}},\qquad
b_y=\log p(y)+\sum_{i=1}^d\log(1-\theta_{iy}).
$$
:eqlabel:`eq_mdl-naive_bayes_affine`

Each inked pixel adds $w_{iy}$, the log-odds that the pixel is inked in
class $y$, and the offset $b_y$ is the score of a blank image. Every class score is affine in
$\mathbf x$, so the boundary between any two classes, where their scores are
equal, is a hyperplane :cite:`Bishop.2006`. This is the same family of
classifiers as softmax regression, with the same 7,850 parameters for
Fashion-MNIST: $784\times10$ weights and $10$ offsets. The two methods differ
only in how they choose the numbers, by counting or by gradient descent on
$p(y\mid\mathbf x)$. With two classes the posterior is a sigmoid of the
score difference,
$p(y=1\mid\mathbf x)=\operatorname{sigmoid}\bigl((\mathbf w_1-\mathbf w_0)^\top\mathbf x+b_1-b_0\bigr)$,
which is the form of logistic regression
(exercise :exref:`ex_mdl-naive-bayes-discussion-after-eqref-eq`).

A two-dimensional example shows how much the way of choosing matters.
:numref:`fig_mdl-naive-genvdisc` shows two classes whose two features are
correlated within each class, so that each cloud of points is tilted. The
naive generative fit models each class by a Gaussian with independent
features, drawn as axis-aligned ellipses. The two classes share the
per-feature variances, so the $x_i^2$ terms cancel between the class scores
and Bayes' rule again gives a line. The ellipses ignore the tilt, and the
line cuts across both clouds: a point from either class falls on the wrong
side with probability 19%. The line fitted directly to $p(y\mid\mathbf x)$
runs along the tilt and errs with probability 7%.

![Two routes to a linear classifier on the same points; the two features are correlated within each class. Left: the naive generative fit models each class by a Gaussian with independent features (ellipses at one and two standard deviations) and takes the boundary from Bayes' rule. Right: the line fitted to $p(y\mid\mathbf x)$ directly. Errors are population values for equal priors.](../img/mdl-prob-naive-genvdisc.svg)
:label:`fig_mdl-naive-genvdisc`

The linear form depends on the kind of feature. Gaussian naive Bayes models
a continuous feature $x_i$ in class $y$ by a Gaussian with mean $\mu_{iy}$
and variance $\sigma_{iy}^2$, so the class score contains
$-(x_i-\mu_{iy})^2/(2\sigma_{iy}^2)$. If each feature has the same variance
in every class, as in the figure, the $x_i^2$ terms are equal in all scores,
cancel in every comparison, and the boundary is linear. With class-specific
variances they do not cancel, and the boundary is quadratic in each $x_i$,
though still without cross terms $x_ix_j$
(exercise :exref:`ex_mdl-naive-bayes-gaussian-boundaries`).

## Fashion-MNIST by Counting

We now fit naive Bayes to the images on which we trained softmax
regression, and look at what the model has learned before measuring how
well it classifies.

### Binary Pixels

Fashion-MNIST :cite:`Xiao.Rasul.Vollgraf.2017` has 60,000 training and 10,000
test images of ten kinds of garments, $28\times28$ grayscale pixels each
(:numref:`sec_fashion_mnist`). Its background pixels are exactly $0$, so we
set $x_i=1$ where pixel $i$ has any value above $0$, that is, where it
belongs to the garment, and $x_i=0$ on the background. The loader returns the raw images of both splits;
we flatten each image into a vector of 784 binary features.

```{.python .input #naive-bayes-load}
#@tab mxnet
data = d2l.FashionMNIST()
def binarize(images):  # 1 where the pixel belongs to the garment
    return (images.reshape(-1, 784) > 0).astype(np.float32)

X, Y = (a.asnumpy() for a in data.read(train=True))
X_test, Y_test = (a.asnumpy() for a in data.read(train=False))
X, X_test = binarize(X), binarize(X_test)
print(X.shape, X_test.shape, f'ink fraction {X.mean():.2f}')
```

```{.python .input #naive-bayes-load}
#@tab pytorch
data = d2l.FashionMNIST()
def binarize(images):  # 1 where the pixel belongs to the garment
    return (images.reshape(-1, 784) > 0).astype(np.float32)

X, Y = binarize(data.train.data.numpy()), data.train.targets.numpy()
X_test, Y_test = binarize(data.val.data.numpy()), data.val.targets.numpy()
print(X.shape, X_test.shape, f'ink fraction {X.mean():.2f}')
```

```{.python .input #naive-bayes-load}
#@tab tensorflow
data = d2l.FashionMNIST()
def binarize(images):  # 1 where the pixel belongs to the garment
    return (images.reshape(-1, 784) > 0).astype(np.float32)

(X, Y), (X_test, Y_test) = data.train, data.val
X, X_test = binarize(X), binarize(X_test)
print(X.shape, X_test.shape, f'ink fraction {X.mean():.2f}')
```

```{.python .input #naive-bayes-load}
#@tab jax
data = d2l.FashionMNIST()
def binarize(images):  # 1 where the pixel belongs to the garment
    return (images.reshape(-1, 784) > 0).astype(np.float32)

(X, Y), (X_test, Y_test) = data.train, data.val
X, X_test = binarize(X), binarize(X_test)
print(X.shape, X_test.shape, f'ink fraction {X.mean():.2f}')
```

Half of all pixels are inked. Each class has 6,000 training and 1,000 test
images, so the estimated prior is $0.1$ for every class and the prior term
is the same in every score; the code keeps it anyway.

### Templates and Samples

Training is two counts: the images per class, $n_y$, and for each class and
pixel the images in which the pixel is inked, $n_{iy}$. The code stores the
smoothed estimates $\hat\theta_{iy}$ in the array `P_xy`, one row per class.
The cell also answers the question left open above: how many pixels are
never inked in the training images of a class, and how many test images ink
one of them?

```{.python .input #naive-bayes-train}
n_y = np.bincount(Y, minlength=10)              # images per class
n_x = np.stack([X[Y == y].sum(axis=0) for y in range(10)])
P_y = n_y / n_y.sum()                           # prior p(y)
P_xy = (n_x + 1) / (n_y + 2)[:, None]           # smoothed theta_iy
print('prior:', P_y)
print('never-inked pixels per class:', (n_x == 0).sum(axis=1))
unseen = (X_test * (n_x[Y_test] == 0)).sum(axis=1) > 0
print('test images inking one of their class:', unseen.sum())
```

The zero counts are rare here: a handful of pixels for most classes and 57
for sneakers, which are flat and leave the top of the image blank. Only 5
test images ink a pixel that their own class never inked in training.
Without smoothing, each of these five would get a score of $-\infty$ for its
true class and be misclassified whatever its other 783 pixels say.

Each row of `P_xy` is an image: the fraction of the class's training images
in which each pixel is inked. We show the ten rows as $28\times28$ pictures,
one per class.

```{.python .input #naive-bayes-templates}
d2l.show_images(P_xy.reshape(10, 28, 28), 2, 5,
                titles=data.text_labels(range(10)), cmap='gray');
```

The templates are blurred outlines of the garments. Each template is the
per-pixel mean of its class, so the blur shows how much the garments vary in
placement and shape. Any model that matches the per-pixel frequencies of
these images, with or without independence, has the same templates; the blur
is a property of the data. The naive assumption adds something else: these
means, together with the prior, are all the model keeps. To see what that
implies, we draw images from the model, inking every pixel independently
with probability $\hat\theta_{iy}$.

```{.python .input #mdl-naive-bayes-templates-and-samples-1}
rng = np.random.default_rng()
samples = (rng.random((1000, 10, 784)) < P_xy).astype(np.float32)
d2l.show_images(samples[0].reshape(10, 28, 28), 2, 5,
                titles=data.text_labels(range(10)), cmap='gray');
```

A sample has the right silhouette, but a ragged, speckled outline, because
every pixel near an edge is decided by its own coin. We can put a number on
the difference. Along a row of a real image, ink and background alternate a
few times; in a sample they alternate more often. The next cell compares the
amount of ink and the number of ink/background switches per row in the
training images and in 1,000 samples per class.

```{.python .input #mdl-naive-bayes-templates-and-samples-2}
def switches(images):  # ink/background changes along a row, mean per row
    return np.abs(np.diff(images.reshape(-1, 28, 28), axis=2)).sum(axis=2).mean()

print(f'ink fraction: data {X.mean():.3f}, samples {samples.mean():.3f}')
print('switches per row, data:   ',
      np.round([switches(X[Y == y]) for y in range(10)], 1))
print('switches per row, samples:',
      np.round([switches(samples[:, y]) for y in range(10)], 1))
```

The samples carry the same amount of ink as the data, since each pixel is
inked with the right frequency. But they switch between ink and background
far more often: for pullovers about 5 times per row instead of 3, for
sandals and bags about 8 times instead of 3. Real garments have smooth edges
because neighbouring pixels agree, and that agreement is what the model
discards.

### Checking the Assumption

The samples show the assumption failing over the whole image; one pair of
pixels shows it with numbers. We take two neighbouring pixels on the bottom
hem of pullovers, in row 26 and columns 14 and 15. If they were independent
given the class, the fraction of pullovers with both inked would equal the
product of the two fractions. The cell compares this product with the joint
frequency, in the training pullovers and in the 1,000 sampled ones.

```{.python .input #mdl-naive-bayes-checking-the-assumption}
i, j = 26 * 28 + 14, 26 * 28 + 15     # two neighbouring hem pixels
for name, Z in (('data', X[Y == 2]), ('model', samples[:, 2])):
    p_i, p_j, both = Z[:, i].mean(), Z[:, j].mean(), (Z[:, i] * Z[:, j]).mean()
    print(f'{name:5s}: P(i)={p_i:.2f}  P(j)={p_j:.2f}  '
          f'P(i)P(j)={p_i * p_j:.2f}  P(i,j)={both:.2f}')
```

Each pixel is inked in about 55% of the pullovers. Independence predicts
that both are inked in $0.55\times0.55\approx30\%$ of them; in the data both
are inked in 54%, almost as often as either one alone. The hem either
reaches row 26 or it does not, so the two pixels are nearly one feature
recorded twice. The model's samples, by construction, follow the product.
This duplication is the reason for the overconfidence we will measure in the
calibration subsection.

### Test Accuracy

Prediction uses the affine form :eqref:`eq_mdl-naive_bayes_affine`. We
compute the weights $w_{iy}$ and offsets $b_y$ from `P_xy` and `P_y`, score
all test images and classes with one matrix product, and keep the
predictions for the rest of the section. What is the test accuracy of the
counted model?

```{.python .input #naive-bayes-predict}
W = np.log(P_xy) - np.log1p(-P_xy)              # w_iy, log-odds per pixel
b = np.log1p(-P_xy).sum(axis=1) + np.log(P_y)   # b_y, score of a blank image
s = X_test @ W.T + b                            # s_y(x), shape (10000, 10)
pred = s.argmax(axis=1)
print(f'test accuracy {(pred == Y_test).mean():.4f}')
```

Naive Bayes classifies about 71% of the test images correctly, against 10%
for guessing. The number does not change from run to run, because
counting involves no randomness; the next part asks how precise it is.

The scores also show what the raw product of :eqref:`eq_mdl-naive_bayes`
would have done. Each score is the natural logarithm of a Bayes numerator, so
dividing by $\log 10$ gives its decimal exponent. We compare the numerators
with the smallest positive single-precision number and with the smallest
*normal* double, below which a double loses precision.

```{.python .input #mdl-naive-bayes-test-accuracy-1}
log10 = s / np.log(10)                     # decimal exponent of each numerator
print(f'log10 of the numerators: largest {log10.max():.0f}, smallest '
      f'{log10.min():.0f}, median best {np.median(log10.max(axis=1)):.0f}')
f32 = -149 * np.log(2)     # log of 2**-149, the smallest positive float32
f64 = -1022 * np.log(2)    # log of 2**-1022, the smallest normal float64
print(f'images whose best numerator underflows float32: '
      f'{(s.max(axis=1) < f32).mean():.2f}')
print(f'numerators below the smallest normal float64: {(s < f64).mean():.2f}')
```

The Bayes numerators lie between about $10^{-32}$ and $10^{-1000}$, and the
best one of a typical image is about $10^{-81}$. For 96% of the test images
even the largest of the ten numerators is below the smallest positive
single-precision number, so the raw rule would see ten zeros and could not
classify them at all. In double precision 44% of the numerators fall below
the smallest normal number, where digits are lost, or underflow to zero. The log scores never leave the range of ordinary
numbers, so the computation in log space loses nothing to underflow.

Text was the first application of naive Bayes and is still its standard
one. Each feature records whether a word occurs in a document, so the
features are many and sparse. A spam filter is the two-word example above
with tens of thousands of words :cite:`Manning.Raghavan.Schutze.2008`. Exercise
:exref:`ex_mdl-naive-bayes-two-event-models` compares the two standard ways
of modeling a document.

## Evaluating the Classifier

An accuracy of 71% raises three questions. How precise is it? Which
mistakes make up the missing 29%? And can the probabilities the model
reports be trusted?

### Error Bars

The test accuracy is computed from 10,000 random test images, so another
test set would give a slightly different number. Each image is a Bernoulli
outcome, correct or not, so the accuracy $\hat p$ is the mean of $n$ coin
flips and has the standard error $\sqrt{\hat p(1-\hat p)/n}$
(:numref:`sec_mdl-statistics`). For $\hat p=0.706$ and $n=10{,}000$ this is

$$\sqrt{\frac{0.706\cdot0.294}{10^4}}\approx0.0046,\qquad 1.96\times0.0046\approx0.009,$$

so a 95% confidence interval reaches about $0.9$ points to either side. The
bootstrap of :numref:`sec_mdl-statistics` gives a second, formula-free
answer: resample the test images with replacement, recompute the accuracy on
each resample, and read off the middle 95% of the results.

```{.python .input #mdl-naive-bayes-calibration-1}
rng = np.random.default_rng()
correct = pred == Y_test                           # per-image 0/1 outcomes
n, p = len(correct), correct.mean()
boot = correct[rng.integers(0, n, (1000, n))].mean(axis=1)  # 1,000 resamples
print(f'accuracy {p:.4f}, bootstrap 95% interval '
      f'({np.percentile(boot, 2.5):.3f}, {np.percentile(boot, 97.5):.3f})')
print(f'binomial half-width 1.96 se = {1.96 * np.sqrt(p * (1 - p) / n):.4f}')
```

The bootstrap interval runs from about 0.70 to 0.715, the binomial
half-width is $0.009$, and the two methods agree. The accuracy is therefore
$71\%\pm0.9$ points; the digits after the decimal point of "70.59%" are
noise. Two classifiers evaluated on the same test set make correlated
errors, so a paired analysis of their per-image differences compares them
more sensitively than two separate intervals
(exercise :exref:`ex_mdl-naive-bayes-paired-comparison`).

### Confusions and Weights

The error rate says how often the model is wrong; a *confusion matrix* says
how. It counts the test images by true class (row) and predicted class
(column), so the diagonal holds the correct predictions.

```{.python .input #mdl-naive-bayes-calibration-2}
conf = np.zeros((10, 10), dtype=int)
np.add.at(conf, (Y_test, pred), 1)       # count (true, predicted) pairs
print(conf)
```

The next cell lists the six largest off-diagonal counts and the accuracy
of each class.

```{.python .input #mdl-naive-bayes-confusions-and-weights-1}
names = data.text_labels(range(10))
off = conf - np.diag(np.diag(conf))      # errors only
order = np.argsort(off, axis=None, kind='stable')[::-1][:6]  # largest first
pairs = [f'{names[t]}->{names[q]} {off[t, q]}'
         for t, q in zip(*np.unravel_index(order, off.shape))]
print(', '.join(pairs[:3]))
print(', '.join(pairs[3:]))
print('per-class accuracy:', np.round(np.diag(conf) / conf.sum(axis=1), 2))
```

The errors are concentrated. Coats are taken for pullovers 372 times and
pullovers for coats 242 times; shirts are taken for pullovers, coats and
t-shirts more than 200 times each; sandals for sneakers 232 times. Trousers
(91%) and sneakers (89%) are easy, coats (48%) and shirts (17%) are not.
The errors sit among the four upper-body garments and among the three kinds
of shoes.

Is this the best a linear classifier can do on these pixels? Softmax
regression draws the same kind of hyperplanes with the same 7,850
parameters, but chooses them by minimizing the cross-entropy of
$p(y\mid\mathbf x)$. We train it on the same binary pixels with minibatch
stochastic gradient descent. The code uses the fact from
:numref:`sec_softmax` that the gradient of the cross-entropy with respect to
the logits is the vector of predicted probabilities minus the one-hot label.

```{.python .input #mdl-naive-bayes-confusions-and-weights-2}
rng = np.random.default_rng()
W_sr, b_sr = np.zeros((10, 784)), np.zeros(10)
for epoch in range(20):                  # minibatches of 256 images
    for idx in np.array_split(rng.permutation(len(X)), len(X) // 256):
        O = X[idx] @ W_sr.T + b_sr                   # logits
        E = np.exp(O - O.max(axis=1, keepdims=True))
        G = E / E.sum(axis=1, keepdims=True) - np.eye(10)[Y[idx]]
        W_sr -= 0.05 * G.T @ X[idx] / len(idx)       # mean-loss gradient
        b_sr -= 0.05 * G.mean(axis=0)
pred_sr = (X_test @ W_sr.T + b_sr).argmax(axis=1)
conf_sr = np.zeros((10, 10), dtype=int)
np.add.at(conf_sr, (Y_test, pred_sr), 1)
print(f'accuracy: softmax regression {(pred_sr == Y_test).mean():.3f}, '
      f'naive Bayes {(pred == Y_test).mean():.3f}')
print('pullover<->coat errors: softmax regression',
      conf_sr[2, 4] + conf_sr[4, 2], 'naive Bayes', conf[2, 4] + conf[4, 2])
```

Softmax regression reaches about 83%, twelve points above naive Bayes, and
roughly halves the confusions between pullovers and coats. The family of
hyperplanes is therefore not the limit. The naive Bayes errors come from how it sets
the weights: each pixel is weighted as if it were independent evidence,
whereas the direct fit learns to discount pixels that only repeat what
their neighbours say.

The weights themselves show what naive Bayes uses to tell two classes
apart. The score difference between pullover and coat is affine with the
weight vector $\mathbf w_{\text{pullover}}-\mathbf w_{\text{coat}}$, and an
inked pixel moves the decision toward pullover by its entry. We display this
vector as an image, red for pullover and blue for coat, and print its
largest entry together with the counts behind it.

```{.python .input #mdl-naive-bayes-confusions-and-weights-3}
w = (W[2] - W[4]).reshape(28, 28)        # pullover minus coat, in nats
m = np.abs(w).max()
d2l.plt.imshow(w, cmap='RdBu_r', vmin=-m, vmax=m)
d2l.plt.colorbar()
k = np.abs(w).argmax()
print(f'largest |weight| {m:.1f} nats at row {k // 28}, column {k % 28}: '
      f'inked in {n_x[2, k]:.0f} pullovers and {n_x[4, k]:.0f} coats')
```

Red pixels lie along both sides of the image and in the top rows, where
pullovers are inked more often than coats. Blue pixels fill the torso. They
are inked in almost every pullover and almost every coat, a little more
reliably in coats, and the log-odds scale turns that small difference into
weights near $-1$. Since nearly every image inks them, they shift the score
difference by almost the same amount for every pullover and every coat, and
the offsets $b_y$ absorb that shift. The decision is made along the sides.
The largest weight, about 3 nats, sits in a corner that is inked in about
twenty pullovers and in no coat, so its size is set by the pseudo-count of
smoothing rather than by much evidence.

A generative model has one advantage that the direct fit lacks. If some
pixels are missing at test time, say hidden by an occlusion, naive Bayes
drops their factors from the product. Under the model this is exact, since
summing $p(x_i\mid y)$ over the two values of a hidden $x_i$ gives $1$.
Softmax regression has no such operation and must fill in a value
(exercise :exref:`ex_mdl-naive-bayes-missing-pixels`).

### Calibration

The scores do more than rank the classes. Exponentiating them restores the
Bayes numerators, and normalizing restores the denominator that we dropped,
so the model's posterior is the softmax of its scores:

$$p(y\mid\mathbf x)=\frac{\exp s_y(\mathbf x)}{\sum_{y'}\exp s_{y'}(\mathbf x)}.$$

Every prediction therefore comes with a confidence, the largest of these
probabilities. As in :numref:`sec_classification`, a classifier is
*calibrated* if its confidences match frequencies: among the predictions
made with confidence $c$, a fraction $c$ is correct. To check, we sort the
test images into bins by confidence and compare, in each bin, the mean
claimed confidence with the achieved accuracy.

The *expected calibration error* summarizes the check in one number. It
averages the gap between claimed confidence and achieved accuracy over the
bins, weighted by the fraction of images in each. The softmax is computed in its stable form,
with the largest score subtracted first (:numref:`subsec_mdl-stable-softmax`).
The cell also prints the median gap between the best and the second-best
score.

```{.python .input #mdl-naive-bayes-calibration-3}
post = np.exp(s - s.max(axis=1, keepdims=True))      # softmax, stably
post /= post.sum(axis=1, keepdims=True)
c = post.max(axis=1)                                 # claimed confidence
print(f'mean confidence {c.mean():.3f}, accuracy {correct.mean():.3f}')
edges, ece = [0, 0.9, 0.99, 0.999, 1], 0
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (c > lo) & (c <= hi)
    ece += m.mean() * abs(c[m].mean() - correct[m].mean())
    print(f'({lo}, {hi}]: {m.sum():5d} images, claimed {c[m].mean():.3f}, '
          f'achieved {correct[m].mean():.3f}')
top2 = np.sort(s, axis=1)[:, -2:]
print(f'ECE {ece:.2f}, median top-two score gap '
      f'{np.median(top2[:, 1] - top2[:, 0]):.0f} nats')
```

The model claims 99% confidence on average and is right 71% of the time.
Most test images, about 92%, land in the top bin, where the model claims
certainty to three decimals and is right on about 73% of them. In the three
lower bins it claims between 73% and 99.6% and achieves only 35% to 42%.
Every bin is overconfident, so the expected calibration error equals the
mean confidence minus the accuracy, about 0.29; softmax regression in
:numref:`sec_softmax_concise` stayed below 0.03. The ordering still carries
some information, since accuracy rises from bin to bin, but the
probabilities cannot be used as stated.

The reason is the duplicated evidence of the hem pixels. The simplest case
of the effect is a feature that is stored several times.

**Proposition (duplicated evidence multiplies the log-odds).** Consider two
classes with priors $\pi_0$ and $\pi_1$ and a binary feature $x$ with
likelihood ratio $r(x)=p(x\mid y=1)/p(x\mid y=0)$. Suppose the data contain
$k$ identical copies of $x$, and naive Bayes models each copy by its true
class-conditional distribution $p(x\mid y)$ but treats the copies as
conditionally independent. Then naive Bayes reports the log posterior odds

$$\log\frac{\pi_1}{\pi_0}+k\log r(x),$$

while the correct log posterior odds are $\log(\pi_1/\pi_0)+\log r(x)$. If
$\pi_0=\pi_1$, both predict the same class for every $k$, and naive Bayes
claims the confidence $\bar r^k/(1+\bar r^k)$ with $\bar r=\max(r,1/r)$.

**Proof.** The copies always agree, so the true distribution of the $k$
copies given $y$ is that of $x$ alone, $p(x\mid y)$. The naive model
multiplies $k$ equal factors and uses $p(x\mid y)^k$. The log ratio of the
two class likelihoods is therefore $\log r(x)$ under the truth and
$k\log r(x)$ under the model; adding the log prior odds gives both claims.
With equal priors the predicted class is decided by the sign of the log
odds, and $\log r(x)$ and $k\log r(x)$ have the same sign. The claimed
posterior of the predicted class is $\bar r^k/(1+\bar r^k)$, since its odds
are $\bar r^k$. $\blacksquare$

Take a clue that is right 80% of the time, $p(x=1\mid y=1)=0.8$ and
$p(x=1\mid y=0)=0.2$, so that $r=4$ when the clue is on. One copy gives the
correct confidence $0.80$; two, three, four and five copies make naive Bayes
claim $0.94$, $0.985$, $0.996$ and $0.999$, while its predictions stay right
80% of the time. Duplicating the word "free" in the spam filter has the same
effect: the odds of the first mail rise from $9$ to
$\frac23\cdot4.5^2\cdot3=40.5$, a spam probability of $0.976$.

A reader may object that the proposition protects the ranking only with
equal priors. That is right. With prior odds $\pi_1/\pi_0=1/5$ and $r=4$,
one copy gives posterior odds $4/5$ and the correct prediction, class $0$;
two copies give $16/5$ and predict class $1$. Repeated evidence can overturn
a prior that the evidence alone should not overturn. The ranking is safe
when the evidence points the same way as the prior, or when a single copy
already outweighs it.

On Fashion-MNIST the 784 pixels are not identical copies of one another,
but many of them are strongly correlated. The two hem pixels are nearly one
pixel counted twice, and hundreds of such groups add up to the median score
gap of about 60 nats between the best and second-best class. Since
$e^{-60}\approx10^{-26}$, the softmax turns such a gap into a confidence of
$1$. Correlated groups also differ in size, so a large region of redundant
pixels can outvote a small region that carries independent information.
This distorts the ranking as well; the direct fit, which can learn to
discount redundant pixels, is less exposed to it.

Overconfidence is not peculiar to naive Bayes. Deep networks are often
overconfident in the same direction, though usually by less
:cite:`Guo.Pleiss.Sun.Weinberger.2017`, and the reliability check above
applies to any classifier whose probabilities will be used.

We can now fit a classifier by counting, score it stably in log space as a
linear classifier, and report its accuracy with an error bar, its confusion
matrix and its calibration. Two problems remain open. The counted
hyperplanes rank the classes well enough for 71% accuracy, yet hyperplanes
of the same family fitted directly reach 83%, and the counted probabilities
are far too confident. Dividing all scores by one fitted number repairs much
of the second problem without refitting
(exercise :exref:`ex_mdl-naive-bayes-temperature-scaling`). When a
generative fit beats a discriminative one is the subject of
exercise :exref:`ex_mdl-naive-bayes-generative-vs-discriminative`.

## Summary

Bayes' rule builds a classifier from a model of how each class produces
its inputs, but a table of $p(\mathbf x\mid y)$ over $10^{236}$ binary
images cannot be estimated. The naive assumption, that the features are
conditionally independent given the class, reduces the model to one
frequency per class and feature, smoothed by one pseudo-count. For a spam
filter with two words or for 784 pixels, training is counting. The log of
the naive Bayes numerator is affine in binary features, so Bernoulli naive
Bayes is a linear classifier with the same 7,850 parameters as softmax
regression.

On Fashion-MNIST it reaches about 71% $\pm$ 0.9 points, twelve points below
the discriminative fit of the same family. It claims 99% confidence,
because correlated pixels are counted as independent evidence and
duplicated evidence multiplies the log-odds. The model uses three of the
chapter's tools: maximum likelihood, the Beta posterior and the bootstrap.
Its overconfidence is the reason to check the calibration of any
classifier.

## Exercises

1. **XOR and the naive assumption.** The four points $(0,0)$, $(0,1)$,
   $(1,0)$ and $(1,1)$ carry the labels $y=x_1\oplus x_2$, that is $0$, $1$,
   $1$ and $0$. Each point occurs once in the training set.
    1. Compute the maximum-likelihood estimates of $p(y)$ and
       $\theta_{iy}=p(x_i=1\mid y)$, and the class scores of all four
       points. Which predictions does the naive Bayes rule
       :eqref:`eq_mdl-naive_bayes` make?
    1. Show that no choice of Bernoulli naive Bayes parameters classifies all
       four points correctly, using the affine form
       :eqref:`eq_mdl-naive_bayes_affine` of the scores. Why can degenerate parameters
       $\theta_{iy}\in\{0,1\}$ not help? (Each class must give positive
       probability to its own two points.)
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
   $y\in\{0,1\}$ with the prior $\pi_1=p(y=1)$ and the Bernoulli parameters
   $\theta_{iy}=p(x_i=1\mid y)$.
    1. Show that the naive Bayes posterior is
       $p(y=1\mid\mathbf x)=\operatorname{sigmoid}(\mathbf w^\top\mathbf x+b)$
       with $\operatorname{sigmoid}(z)=1/(1+e^{-z})$ and
       $w_i=\log\frac{\theta_{i1}(1-\theta_{i0})}{\theta_{i0}(1-\theta_{i1})}$.
       Express $b$ through the $\theta_{iy}$ and $\pi_1$. Which sign does $w_i$
       have for a pixel that is inked more often in class $1$ than in class
       $0$?
    1. Show that, conversely, every pair $(\mathbf w,b)$ arises from some
       choice of $\theta_{iy}\in(0,1)$ and $\pi_1\in(0,1)$. Naive Bayes and
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
       for some $0<\rho<1$. Show that the difference of the two class scores
       is $2\rho x_1x_2/(1-\rho^2)$, so that the Bayes classifier assigns
       class $+$ exactly when $x_1x_2>0$, and that each single score also
       depends on $x_1^2+x_2^2$. The two classes have the same marginals, so
       Gaussian naive Bayes is at chance even with unlimited data. Under
       $\mathcal N(\mathbf 0,\boldsymbol\Sigma_+)$,
       $P(x_1x_2>0)=\frac12+\frac1\pi\arcsin\rho$ (Sheppard's formula). Find
       the Bayes error as a function of $\rho$, and evaluate it for
       $\rho=0.1$ and $\rho=0.9$.
   :exlabel:`ex_mdl-naive-bayes-gaussian-boundaries`

1. [code] **Smoothing and unseen pixels.** The `#naive-bayes-train` cell
   estimates $\theta_{iy}$ by $(n_{iy}+1)/(n_y+2)$.
    1. Generalize the cell to a pseudocount $\alpha$, estimating
       $\theta_{iy}$ by $(n_{iy}+\alpha)/(n_y+2\alpha)$, and report the test
       accuracy for $\alpha\in\{0,0.01,0.1,1,10,100,1000\}$. In floating
       point, $0\cdot(-\infty)$ is NaN. Which values does the affine score
       of the `#naive-bayes-predict` cell take at $\alpha=0$, and which class
       does `argmax` then return for every image?
    1. Rewrite the score so that each pixel selects one of its two log
       probabilities, for example with `np.where` one class at a time,
       instead of multiplying both by $x_i$ and $1-x_i$. Rerun $\alpha=0$. Which
       fraction of the class scores is $-\infty$, for how many test images
       is every class score $-\infty$, and what is the accuracy? Was the
       failure in (a) a property of the unsmoothed model or of its
       implementation?
    1. The estimate with pseudocount $\alpha$ is the posterior mean of
       $\theta_{iy}$ under a Beta prior. Which prior is it? Why does the
       accuracy fall as $\alpha$ grows large?
   :exlabel:`ex_mdl-naive-bayes-suppose-we-omitted-laplace`

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

1. [code] **Paired comparison.** The text compared naive Bayes with softmax
   regression by their accuracies, `pred` against `pred_sr`.
    1. Bootstrap the per-image difference in correctness, using the same
       resampled indices for both models, and report a 95% interval for the
       difference of the accuracies.
    1. Compare the width of this interval with that of the two marginal
       intervals and with the unpaired standard error
       $\sqrt{p_1(1-p_1)/n+p_2(1-p_2)/n}$. Compute the correlation of the two
       vectors of correctness, and use it to explain why pairing helps.
    1. Count the test images that both models get wrong, that only naive
       Bayes gets right, and that only softmax regression gets right. Which of
       these counts determine the difference of the accuracies? What would the
       table look like for a model that differs from naive Bayes only by
       correcting some of its errors?
   :exlabel:`ex_mdl-naive-bayes-paired-comparison`

1. **Duplicated and correlated evidence.** Two classes have equal priors,
   and a binary feature has $p(x=1\mid y=1)=0.8$ and $p(x=1\mid y=0)=0.2$.
    1. A pipeline stores the feature $k$ times. Find the smallest $k$ for
       which naive Bayes claims at least 99.9% confidence for predictions that
       are right 80% of the time.
    1. Now store two copies that agree only partly: given $y$, the second
       copy equals the first with probability $q$ and is otherwise a fresh
       draw from the same class-conditional distribution. Show that both
       copies have the same marginal as $x$, so naive Bayes still claims
       $16/17\approx0.94$ when both copies are $1$. Show that the correct
       posterior is $r/(1+r)$ with $r=4(0.8+0.2q)/(0.2+0.8q)$, and evaluate it
       for $q\in\{1,0.75,0.5,0.25,0\}$. How does the over-count depend on the
       correlation of the copies?
    1. Show that dividing every class score by $T=k$ before the softmax
       restores the correct posterior in (a). Why are equal priors needed
       for this? (The prior term is divided by $k$ too.)
   :exlabel:`ex_mdl-naive-bayes-duplicated-features`

1. [code] **Label shift.** Under label shift
   (:numref:`sec_environment-and-distribution-shift`), the target prior
   $p_T(y)$ differs from the source prior $p_S(y)$, while
   $p(\mathbf x\mid y)$ stays the same.
    1. Show that the posterior under the target prior is
       $p_T(y\mid\mathbf x)\propto p_S(y\mid\mathbf x)\,p_T(y)/p_S(y)$. Why
       does naive Bayes need no retraining to use it?
    1. Build a test set in which pullovers make up 90% of the images: all
       1,000 test pullovers and 111 images drawn at random from the other
       classes. Compare the accuracy of `pred` on it with that of the
       corrected rule of (a), with $p_T(\text{pullover})=0.9$ and
       $p_T(y)=1/90$ for every other class.
    1. The correction adds $\log 9\approx2.2$ nats to the pullover score and
       subtracts $2.2$ from the others. Compare this with the score gaps of
       the `#mdl-naive-bayes-calibration-3` cell and explain the size of the
       change you measured in (b).
   :exlabel:`ex_mdl-naive-bayes-label-shift`

1. [code] **Score gaps under the model.** ● For two classes $u$ and $v$, let
   $g(\mathbf x)=s_u(\mathbf x)-s_v(\mathbf x)$ be the difference of their
   scores, so that the rule ranks $v$ above $u$ exactly when
   $g(\mathbf x)<0$. By :eqref:`eq_mdl-naive_bayes_affine`,
   $g(\mathbf x)=\mathbf w^\top\mathbf x+b_u-b_v$ with
   $\mathbf w=\mathbf w_u-\mathbf w_v$.
    1. Let $\mathbf x$ follow the model's own class-$u$ distribution, with
       independent pixels $x_i\sim\mathrm{Bernoulli}(\theta_{iu})$. Show that
       $\mathbb{E}[g]$ equals the Kullback--Leibler divergence
       $\sum_{\mathbf x}p(\mathbf x\mid u)\log\bigl(p(\mathbf x\mid u)/p(\mathbf x\mid v)\bigr)$
       of the two class-conditional distributions of the model plus
       $\log\bigl(p(u)/p(v)\bigr)$, and that
       $\textrm{Var}[g]=\sum_i w_i^2\,\theta_{iu}(1-\theta_{iu})$.
    1. Let $u$ be pullover and $v$ coat. Compute $g$ on the test pullovers,
       and compare its empirical mean and standard deviation with the two
       values of (a). From the model's mean and variance, use a Gaussian
       approximation to predict the fraction of these images with $g<0$, and
       compare it with the observed fraction.
    1. On real images,
       $\textrm{Var}[g]=\mathbf w^\top\boldsymbol\Sigma_u\mathbf w$,
       where $\boldsymbol\Sigma_u$ is the covariance matrix of the pixels of
       class $u$. Which entries of $\boldsymbol\Sigma_u$ does the variance of
       (a) omit? Show that their contribution
       $\sum_{i\ne j}w_iw_j\Sigma_{u,ij}$ is positive when positively
       correlated neighbours carry weights of the same sign, and check both
       the correlation and the sign agreement for the hem pixels of the
       text.
   :exlabel:`ex_mdl-naive-bayes-score-gaps`

1. [code] **Temperature scaling.** ● Exercise
   :exref:`ex_mdl-naive-bayes-duplicated-features` shows that dividing the
   class scores by $T=k$ undoes a $k$-fold duplication of features. For
   Fashion-MNIST the effective duplication is unknown, and temperature
   scaling :cite:`Guo.Pleiss.Sun.Weinberger.2017` fits $T$ on held-out data
   instead: the posterior becomes the softmax of $s_y(\mathbf x)/T$.
    1. Refit the model on the first $50{,}000$ training images, and choose
       $T$ by minimizing the average negative log-likelihood of the
       posterior on the remaining $10{,}000$. Why does $T$ not change the
       test accuracy?
    1. Apply the fitted $T$ to the scores of the same $50{,}000$-image model
       on the test set, and repeat the reliability table of the
       `#mdl-naive-bayes-calibration-3` cell with the bin edges $0$, $0.5$,
       $0.7$, $0.9$, $0.99$ and $1$. Compare the mean claimed confidence with
       the accuracy.
    1. In the setting of exercise
       :exref:`ex_mdl-naive-bayes-duplicated-features`, show that the
       standard deviation of the score gap on real data is $\sqrt k$ times
       the standard deviation that the model predicts. Compute the square of
       this ratio for the pair pullover/coat of exercise
       :exref:`ex_mdl-naive-bayes-score-gaps` and for t-shirt/shirt and
       pullover/shirt, and compare it with the fitted $T$.
    1. Apply the prior correction of exercise
       :exref:`ex_mdl-naive-bayes-label-shift` to the temperature-scaled
       posterior, on the test set built there. Why does the corrected prior
       change many more predictions after temperature scaling?
   :exlabel:`ex_mdl-naive-bayes-temperature-scaling`

1. [code] **Missing pixels.** Hide a random half of the pixels of every test
   image with a Boolean mask `M` of the shape of `X_test` that is `True` for
   the visible pixels.
    1. Show that under the model, marginalizing a hidden pixel out of
       $p(\mathbf x\mid y)$ removes its factor, so that the score of the
       visible pixels is $\log p(y)+\sum_{i\text{ visible}}\log p(x_i\mid y)$.
       Implement it with two matrix products,
       `(X_test * M) @ np.log(P_xy).T + ((1 - X_test) * M) @ np.log(1 - P_xy).T`
       plus the log prior, and compare the accuracy with the 71% of the full
       images.
    1. Instead fill the hidden pixels with $0$, with $1$, or with the mean of
       the training images, and score with the affine form. Explain each
       accuracy from :eqref:`eq_mdl-naive_bayes_affine`; what does filling
       with $0$ add to the score of every class?
    1. Score softmax regression from the text on the images with hidden
       pixels set to $0$. Which operation does a discriminative model lack
       that the generative model used in (a)?
   :exlabel:`ex_mdl-naive-bayes-missing-pixels`

1. [extended] **Generative versus discriminative learning.** ●●
   :citet:`Ng.Jordan.2002` compare naive Bayes with logistic regression,
   which by exercise :exref:`ex_mdl-naive-bayes-discussion-after-eqref-eq`
   fits the same family of posteriors by maximizing the conditional
   likelihood. They prove that the naive Bayes parameters approach their
   population values after a number of examples logarithmic in $d$. For
   logistic regression, the number of examples needed to approach its
   asymptotic error is at most linear in $d$, and in the worst case it is
   also at least linear in $d$. On many of their $15$ datasets, naive Bayes
   has the lower test error for small training sets and logistic regression
   for large ones.
    1. Let each class have at least $m$ training examples. Use Hoeffding's
       inequality (:numref:`sec_mdl-concentration-generalization`) and a union bound to show that
       all $2d$ unsmoothed estimates $\hat\theta_{iy}$ of a two-class model
       lie within $\varepsilon$ of their population values with probability
       at least $1-\delta$ when
       $$m\ge\frac{\log(4d/\delta)}{2\varepsilon^2}.$$
       Evaluate the bound for $d=784$, $\varepsilon=0.05$ and $\delta=0.05$.
    1. A uniform bound on $|\hat\theta_{iy}-\theta_{iy}|$ does not by itself
       bound the change of the class scores. Why not, for the pixels with
       $\theta_{iy}$ close to $0$ or $1$, such as the corner pixel that
       carries the largest pullover-coat weight in the text? What must be
       assumed about the $\theta_{iy}$ for the scores to converge at the rate
       of the parameters?
    1. Restrict the data to pullovers and coats. For the total training-set
       sizes $n\in\{10,20,50,100,200,500,1000,2000,5000\}$ and for all
       $12{,}000$ images, draw random training subsets that contain both
       classes, $20$ for each size up to $n=200$ and $5$ for each larger
       size. Fit naive Bayes with Laplace smoothing, and logistic regression
       without weight decay. For small $n$ the training data are linearly
       separable, and by exercise
       :exref:`ex_mdl-naive-bayes-discussion-after-eqref-eq`c the logistic
       loss has no minimizer, so run a fixed number of full-batch gradient
       steps from zero initialization. Plot the average test errors of both
       classifiers against $n$ on a logarithmic axis with `d2l.plot`. In
       which range of $n$, if any, does naive Bayes have the lower error?
    1. Repeat (c) with synthetic data for which the naive assumption holds:
       draw balanced labels and sample each pixel independently from
       $\mathrm{Bernoulli}(\theta_{iy})$, with the $\theta_{iy}$ estimated
       from all training images of the two classes. How do the two learning
       curves and their asymptotic errors change?
    1. Theorem 4 of :citet:`Ng.Jordan.2002` bounds the excess error of naive
       Bayes by the probability that the score gap of the population model
       falls near zero. Use the score-gap variances of exercise
       :exref:`ex_mdl-naive-bayes-score-gaps` to compare this probability for
       real pixels and for the synthetic data of (d). How does this
       comparison, together with the asymptotic errors of (c) and (d),
       account for the two pairs of learning curves?
   :exlabel:`ex_mdl-naive-bayes-generative-vs-discriminative`

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
[Dive into Deep Learning · §27.9]{.kicker}

A classifier from a model of each class<br>**count, smooth, add logs, then check the probabilities**
:::
:::

::: {.slide title="$10^{236}$ Black-and-White Images, 60,000 Training Images"}
[The problem]{.kicker}

- Softmax regression on Fashion-MNIST: 7,850 numbers by gradient descent, about 83%.
- Another route: model how each class produces its images, then apply Bayes' rule.
- A model of "how coats look" gives a probability to every binary image.
- 784 pixels: $2^{784}\approx10^{236}$ images, against 60,000 for training.

::: {.d2l-note}
Can a classifier be learned by counting instead of by gradient descent?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- derive the naive Bayes rule from Bayes' rule and one assumption;
- fit it by counting, with one pseudo-count against zero counts;
- score it in log space and read it as a linear classifier;
- report its accuracy with an error bar, its confusions and its calibration;
- prove why it is overconfident.

**Running example:** Fashion-MNIST with "garment pixels". **By hand:** a spam
filter with two words and ten mails.

**Idea:** if features are independent given the class, counting each one
separately is enough; the ranking survives, the stated probabilities do not.
:::

::: {.slide title="One Classifier Built from the Chapter's Tools"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Random Variables | densities, moments, covariance |
| Distributions | the named laws and their losses |
| The Exponential Family | one form, convex fits, conjugacy |
| Statistics | error bars, tests, intervals |
| Maximum Likelihood | losses, Fisher information, MAP, EM |
| Bayesian Computation | averaging over a posterior |
| Concentration Inequalities | exponential tails of averages |
| Uniform Convergence | guarantees after selection |
| **Naive Bayes** | **one classifier, end to end** |
:::

::: {.col .narrow}
- **Uses:** Bayes' rule, MLE, the Beta posterior, the bootstrap.
- **Skips:** mixtures and EM, text pipelines, deep generative models.
:::
:::
:::

::: {.slide title="Bayes' Rule Turns a Model of Each Class into a Classifier"}
[Bayes' rule]{.kicker}

$$\hat y=\operatorname*{argmax}_y p(y\mid\mathbf x)
=\operatorname*{argmax}_y \frac{p(\mathbf x\mid y)\,p(y)}{p(\mathbf x)}
=\operatorname*{argmax}_y p(\mathbf x\mid y)\,p(y).$$

- $p(\mathbf x)$ is the same for every class, so it drops.
- **Generative:** model $p(\mathbf x\mid y)$ and $p(y)$. **Discriminative:** model $p(y\mid\mathbf x)$.

. . .

Still $2^{784}$ entries per class: by the chain rule,
$p(\mathbf x\mid y)=\prod_i p(x_i\mid x_1,\ldots,x_{i-1},y)$, and each factor
remembers the whole history.
:::

::: {.slide title="Independence Turns $2^{784}-1$ Probabilities per Class into 784"}
[The naive assumption]{.kicker}

::: {.cols .vc}
::: {.col}
Once the class is known, one pixel says nothing about another:

$$p(\mathbf x\mid y)=\prod_{i=1}^{784}p(x_i\mid y).$$

- One number per class and pixel: $\theta_{iy}=p(x_i=1\mid y)$.
- Pairwise independence is not enough: $x_3=x_1\oplus x_2$.
:::

::: {.col .fig}
@fig:mdl-prob-naive-independence
:::
:::
:::

::: {.slide title="Two Words Classify a Mail: Posterior 0.90"}
[Worked example]{.kicker}

| ten mails | spam (4) | ordinary (6) |
|---|---|---|
| contains "free" | 3 | 1 |
| contains "meeting" | 0 | 4 |

New mail: "free", no "meeting".

- Numerators: $0.4\cdot\tfrac34\cdot1=0.30$ against $0.6\cdot\tfrac16\cdot\tfrac26=0.033$.
- Odds: $\underbrace{\tfrac23}_{\text{prior}}\times\underbrace{4.5}_{\text{"free"}}\times\underbrace{3}_{\text{no "meeting"}}=9$, so $p(\text{spam}\mid\mathbf x)=0.90$.

::: {.d2l-note}
Each feature multiplies the odds by its likelihood ratio.
:::
:::

::: {.slide title="How Many Numbers per Class?"}
[Check]{.kicker}

An image has 784 binary pixels and there are ten classes. How many free
probabilities does $p(\mathbf x\mid y)$ need per class, with and without the
naive assumption?

. . .

- Without: $2^{784}-1\approx10^{236}$.
- With: $784$, each a frequency estimated from 6,000 images.
- The prior adds $9$ in both cases.
:::

::: {.slide title="Counting Is Maximum Likelihood; +1 Keeps Every Factor Positive"}
[Smoothing]{.kicker}

- MLE: $\hat\theta_{iy}=n_{iy}/n_y$, the frequency, and $\hat p(y)=n_y/n$.
- A zero count vetoes the class: "meeting" never in spam, so any mail with "meeting" is ordinary.
- Laplace: $\hat\theta_{iy}=(n_{iy}+1)/(n_y+2)$, the posterior mean under $\mathrm{Beta}(1,1)$.
- A coin with 9 heads in 13 flips: $9/13=0.69\to10/15=0.67$; with 0 heads: $0\to1/15$.
- Mail with both words: spam probability $0\to0.32$.
:::

::: {.slide title="The Raw Product Underflows: $2^{-784}\approx10^{-236}$"}
[Log space]{.kicker}

- 785 factors below 1; smallest positive float32 $\approx1.4\cdot10^{-45}$.
- $\log$ is increasing, so it keeps the argmax. The **class score**:

$$s_y(\mathbf x)=\log p(y)+\sum_{i=1}^{784}\log p(x_i\mid y).$$

- Spam filter: $s=\log0.30=-1.20$ against $\log\tfrac1{30}=-3.40$; gap $\log 9=2.20$.
:::

::: {.slide title="Bernoulli Naive Bayes Is a Linear Classifier"}
[Linear boundaries]{.kicker}

$\log p(x_i\mid y)=x_i\log\theta_{iy}+(1-x_i)\log(1-\theta_{iy})$, so

$$s_y(\mathbf x)=\mathbf w_y^\top\mathbf x+b_y,\quad
w_{iy}=\log\frac{\theta_{iy}}{1-\theta_{iy}},\quad
b_y=\log p(y)+\sum_i\log(1-\theta_{iy}).$$

- Each inked pixel adds its log-odds; $b_y$ scores the blank image.

::: {.d2l-note}
Same family and same 7,850 numbers as softmax regression, chosen by counting
instead of gradient descent.
:::
:::

::: {.slide title="Two Ways to Choose the Line: 19% Error Against 7%"}
[Generative and discriminative]{.kicker}

@fig:mdl-prob-naive-genvdisc

- Left: axis-aligned Gaussians ignore the tilt of the classes; Bayes' rule gives a poor line.
- Right: the line fitted to $p(y\mid\mathbf x)$ directly follows the tilt.
:::

::: {.slide title="Training Is Two Counts"}
[Fashion-MNIST]{.kicker}

$x_i=1$ where the pixel belongs to the garment; ink fraction $0.50$.

@naive-bayes-train

::: {.d2l-note}
Zero counts are rare here: only 5 test images ink a pixel their class never inked.
:::
:::

::: {.slide title="Templates Are Means; Samples Show What the Model Forgets"}
[What the model learned]{.kicker}

@!mdl-naive-bayes-templates-and-samples-1

- Template $\hat\theta_{iy}$ = per-pixel mean, a property of the data, not of the assumption.
- Sample: each pixel its own coin. Same ink (0.50), ragged edges.
- Ink/background switches per row: pullovers 5 against 3, sandals 8 against 3.
:::

::: {.slide title="Independence Predicts 30% for Two Hem Pixels; the Data Say 54%"}
[The assumption, measured]{.kicker}

Two neighbouring pixels on the bottom hem of pullovers:

@!mdl-naive-bayes-checking-the-assumption

- The hem reaches the row or it does not: nearly one feature recorded twice.
- The model's samples follow the product by construction.
:::

::: {.slide title="About 71% ± 0.9 Points on 10,000 Test Images"}
[Accuracy and its error bar]{.kicker}

$s=\mathbf X\mathbf W^\top+\mathbf b$, one matrix product for all images:

@!naive-bayes-predict

@!mdl-naive-bayes-calibration-1

- Binomial: $\sqrt{0.706\cdot0.294/10^4}=0.0046$, times $1.96$ is $0.009$.
- The digits after the decimal point of 70.59% are noise.
:::

::: {.slide title="Same 7,850 Numbers, Fitted Directly: Higher or Lower?"}
[Check]{.kicker}

Softmax regression uses the same hyperplanes on the same binary pixels, but
fits them by gradient descent on $p(y\mid\mathbf x)$. Will it beat 71%,
and by how much?

. . .

@!mdl-naive-bayes-confusions-and-weights-2

- About twelve points higher: the family of lines is not the limit.
- Naive Bayes weights each pixel as independent evidence.
:::

::: {.slide title="Errors Sit Among Upper-Body Garments and Among Shoes"}
[Confusions]{.kicker}

@!mdl-naive-bayes-confusions-and-weights-1

- Coat and pullover: 614 errors between them; shirts are right only 17% of the time.
- Trousers (91%) and sneakers (89%) are easy.
:::

::: {.slide title="The Torso Weights Cancel; the Sides Decide Pullover Against Coat"}
[Weights]{.kicker}

::: {.cols .vc}
::: {.col}
@!mdl-naive-bayes-confusions-and-weights-3
:::

::: {.col .narrow}
- $\mathbf w_{\text{pullover}}-\mathbf w_{\text{coat}}$, in nats.
- Blue torso: inked in nearly every image of both classes.
- Red sides: inked more often in pullovers.
- The largest weight rests on a pseudo-count.
:::
:::
:::

::: {.slide title="Claimed 99%, Achieved 71%"}
[Calibration]{.kicker}

Confidence = largest softmax of the scores. Bin by confidence, compare with accuracy:

@!mdl-naive-bayes-calibration-3

::: {.d2l-note .warn}
92% of the images claim 1.000 and are right 73% of the time; every bin is overconfident.
:::
:::

::: {.slide title="Duplicated Evidence Multiplies the Log-Odds"}
[Proposition]{.kicker}

Two classes, priors $\pi_0,\pi_1$; one feature $x$ with likelihood ratio
$r(x)=p(x\mid1)/p(x\mid0)$, stored $k$ times.

$$\log\text{odds}_{\text{naive}}=\log\frac{\pi_1}{\pi_0}+k\log r(x)
\quad\text{against}\quad\log\frac{\pi_1}{\pi_0}+\log r(x).$$

- **Proof:** the copies agree, so the truth uses $p(x\mid y)$; the model uses $p(x\mid y)^k$.
- Equal priors: same prediction for every $k$, claimed confidence $\bar r^k/(1+\bar r^k)$, $\bar r=\max(r,1/r)$.
- Unequal priors: prior odds $1/5$, $r=4$: one copy says class 0, two say class 1.
:::

::: {.slide title="An 80% Clue Copied Five Times"}
[Check]{.kicker}

A clue is right 80% of the time ($r=4$), the priors are equal, and the
feature is stored five times. What confidence does naive Bayes claim? And
what do 784 correlated pixels do?

. . .

- $4^5/(1+4^5)=1024/1025=0.999$, for predictions right 80% of the time.
- Fashion-MNIST: median gap between the two best scores about 60 nats.
- $e^{-60}\approx10^{-26}$: the softmax returns 0 or 1.
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

- Naive assumption: $2^{784}-1$ probabilities per class become 784 counts.
- Spam filter by hand: odds $\tfrac23\times4.5\times3=9$, posterior 0.90.
- Log scores are affine: a linear classifier with 7,850 parameters.
- Fashion-MNIST: about $71\%\pm0.9$; the same lines fitted directly, about 83%.
- Claimed 99% against 71% achieved: $k$ copies multiply the log-odds by $k$.

::: {.d2l-note}
Next: dividing every score by one fitted temperature repairs much of the
overconfidence without refitting.
:::
:::
