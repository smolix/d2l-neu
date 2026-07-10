```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Environment and Distribution Shift
:label:`sec_environment-and-distribution-shift`

In the previous sections, we worked through
a number of hands-on applications of machine learning,
fitting models to a variety of datasets.
And yet, we never stopped to contemplate
either where data came from in the first place
or what we ultimately plan to do
with the outputs from our models.
Too often, machine learning developers
in possession of data rush to develop models
without pausing to consider these fundamental issues.

Many failed machine learning deployments
can be traced back to this failure.
Sometimes models appear to perform marvelously
as measured by test set accuracy
but fail catastrophically in deployment
when the distribution of data suddenly shifts.
More insidiously, sometimes the very deployment of a model
can be the catalyst that perturbs the data distribution.
Say, for example, that we trained a model
to predict who will repay rather than default on a loan,
finding that an applicant's choice of footwear
was associated with the risk of default
(Oxfords indicate repayment, sneakers indicate default).
We might be inclined 
thereafter to grant a loan
to any applicant wearing Oxfords
and to deny all applicants wearing sneakers.

In this case, our ill-considered leap from
pattern recognition to decision-making
and our failure to critically consider the environment
might have disastrous consequences.
For starters, as soon as we began
making decisions based on footwear,
customers would catch on and change their behavior.
Before long, all applicants would be wearing Oxfords,
without any coincident improvement in credit-worthiness.
Take a minute to digest this because similar issues abound
in many applications of machine learning:
by introducing our model-based decisions to the environment,
we might break the model. This is a machine-learning incarnation of *Goodhart's law*:
when a measure becomes a target, it ceases to be a good measure.

While we cannot possibly give these topics
a complete treatment in one section,
we aim here to expose some common concerns,
and to stimulate the critical thinking
required to detect such situations early,
mitigate damage, and use machine learning responsibly.
Some of the solutions are simple
(ask for the "right" data),
some are technically difficult
(implement a reinforcement learning system),
and others require that we step outside
statistical prediction altogether and
grapple with difficult philosophical questions
concerning the ethical application of algorithms.

## Types of Distribution Shift

To begin, we stick with the passive prediction setting
considering the various ways that data distributions might shift
and what might be done to salvage model performance.
In one classic setup, we assume that our training data
was sampled from some distribution $p_S(\mathbf{x},y)$
but that our test data will consist
of unlabeled examples drawn from
some different distribution $p_T(\mathbf{x},y)$.
Already, we must confront a sobering reality.
Absent any assumptions on how $p_S$
and $p_T$ relate to each other,
learning a classifier that works at test time is impossible.

Consider a binary classification problem,
where we wish to distinguish between dogs and cats.
If the distribution can shift in arbitrary ways,
then our setup permits the pathological case
in which the distribution over inputs remains
constant: $p_S(\mathbf{x}) = p_T(\mathbf{x})$,
but the labels are all flipped:
$p_S(y \mid \mathbf{x}) = 1 - p_T(y \mid \mathbf{x})$.
In other words, if God can suddenly decide
that in the future all "cats" are now dogs
and what we previously called "dogs" are now cats, without
any change in the distribution of inputs $p(\mathbf{x})$,
then we cannot possibly distinguish this setting
from one in which the distribution did not change at all.

Fortunately, under some restricted assumptions
on the ways our data might change in the future,
principled algorithms can detect shift
and sometimes even adapt on the fly,
improving on the accuracy of the original classifier.

### Covariate Shift

Among categories of distribution shift,
covariate shift may be the most widely studied.
Here, we assume that while the distribution of inputs
may change over time, the labeling function,
i.e., the conditional distribution
$P(y \mid \mathbf{x})$ does not change.
Statisticians call this *covariate shift*
because the problem arises due to a
shift in the distribution of the covariates (features).
While we can sometimes reason about distribution shift
without invoking causality, we note that covariate shift
is the natural assumption to invoke in settings
where we believe that $\mathbf{x}$ causes $y$.

Consider the challenge of distinguishing cats and dogs.
Our training data might consist of images of the kind in :numref:`fig_cat-dog-train`.

![Training data for distinguishing cats and dogs (illustrations: Lafeez Hossain / 500px / Getty Images; ilkermetinkursova / iStock / Getty Images Plus; GlobalP / iStock / Getty Images Plus; Musthafa Aboobakuru / 500px / Getty Images).](../img/cat-dog-train.png)
:label:`fig_cat-dog-train`


At test time we are asked to classify the images in :numref:`fig_cat-dog-test`.

![Test data for distinguishing cats and dogs (illustrations: SIBAS_minich / iStock / Getty Images Plus; Ghrzuzudu / iStock / Getty Images Plus; id-work / DigitalVision Vectors / Getty Images; Yime / iStock / Getty Images Plus).](../img/cat-dog-test.png)
:label:`fig_cat-dog-test`

The training set consists of photos,
while the test set contains only cartoons.
Training on a dataset with substantially different
characteristics from the test set
can spell trouble absent a coherent plan
for how to adapt to the new domain.

### Label Shift

*Label shift* describes the converse problem.
Here, we assume that the label marginal $P(y)$
can change
but the class-conditional distribution
$P(\mathbf{x} \mid y)$ remains fixed across domains.
Label shift is a reasonable assumption to make
when we believe that $y$ causes $\mathbf{x}$.
For example, we may want to predict diagnoses
given their symptoms (or other manifestations),
even as the relative prevalence of diagnoses
is changing over time.
Label shift is the appropriate assumption here
because diseases cause symptoms.
In some degenerate cases the label shift
and covariate shift assumptions can hold simultaneously.
For example, when the label is deterministic,
the covariate shift assumption will be satisfied,
even when $y$ causes $\mathbf{x}$.
Interestingly, in these cases,
it is often advantageous to work with methods
that flow from the label shift assumption.
That is because these methods tend
to involve manipulating objects that look like labels (often low-dimensional),
as opposed to objects that look like inputs,
which tend to be high-dimensional in deep learning.

### Concept Shift

We may also encounter the related problem of *concept shift*,
which arises when the very definitions of labels can change.
This sounds weird (a *cat* is a *cat*, no?).
However, other categories are subject to changes in usage over time.
Diagnostic criteria for mental illness,
what passes for fashionable, and job titles,
are all subject to considerable
amounts of concept shift.
It turns out that if we navigate around the United States,
shifting the source of our data by geography,
we will find considerable concept shift regarding
the distribution of names for *soft drinks*
as shown in :numref:`fig_popvssoda`.

![Concept shift for soft drink names in the United States (CC-BY: Alan McConchie, PopVsSoda.com).](../img/popvssoda.png)
:width:`400px`
:label:`fig_popvssoda`

If we were to build a machine translation system,
the distribution $P(y \mid \mathbf{x})$ might be different
depending on our location.
This problem can be tricky to spot.
We might hope to exploit knowledge
that shift only takes place gradually
either in a temporal or geographic sense.

## Examples of Distribution Shift

Before turning to formalism and algorithms,
we can discuss some concrete situations
where covariate or concept shift might not be obvious.


### Medical Diagnostics

Imagine that you want to design an algorithm to detect cancer.
You collect data from healthy and sick people
and you train your algorithm.
It works fine, giving you high accuracy
and you conclude that you are ready
for a successful career in medical diagnostics.
*Not so fast.*

The distributions that gave rise to the training data
and those you will encounter in the wild might differ considerably.
This happened to an unfortunate startup
that some of us authors worked with years ago.
They were developing a blood test for a disease
that predominantly affects older men
and hoped to study it using blood samples
that they had collected from patients.
However, it is considerably more difficult
to obtain blood samples from healthy men
than from sick patients already in the system.
To compensate, the startup solicited
blood donations from students on a university campus
to serve as healthy controls in developing their test.
Then they asked whether we could help them
to build a classifier for detecting the disease.

As we explained to them,
it would indeed be easy to distinguish
between the healthy and sick cohorts
with near-perfect accuracy.
However, that is because the test subjects
differed in age, hormone levels,
physical activity, diet, alcohol consumption,
and many more factors unrelated to the disease.
This was unlikely to be the case with real patients.
Due to their sampling procedure,
we could expect to encounter extreme covariate shift.
Moreover, this case was unlikely to be
correctable via conventional methods.
In short, they wasted a significant sum of money.



### Self-Driving Cars

Say a company wanted to use machine learning
for developing self-driving cars.
One key component here is a roadside detector.
Since real annotated data is expensive to get,
they had the (smart and questionable) idea
to use synthetic data from a game rendering engine
as additional training data.
This worked really well on "test data"
drawn from the rendering engine.
Alas, inside a real car it was a disaster.
As it turned out, the roadside had been rendered
with a very simplistic texture.
More importantly, *all* the roadside had been rendered
with the *same* texture and the roadside detector
learned about this "feature" very quickly.

A famous (and possibly apocryphal) cautionary tale makes the same point.
As the story goes, the US Army once tried to train a neural network
to detect tanks hidden among trees.
They photographed a forest with no tanks,
then drove tanks in and photographed it again,
and the classifier appeared to work *perfectly* on held-out images,
until it failed in the field.
It had supposedly learned not to find tanks
but to tell the tank-free photos from the rest:
the two image sets differed in lighting and shadow
(one set was taken in the early morning, the other at noon), not in their tanks.
Whether or not it happened exactly this way, the lesson is exact.
A spurious feature correlated with the label in your sample,
but absent in deployment,
is enough to fool a model that never saw the distinction you actually care about.

### Nonstationary Distributions

A much more subtle situation arises
when the distribution changes slowly
(also known as *nonstationary distribution*)
and the model is not updated adequately.
Below are some typical cases.

* We train a computational advertising model and then fail to update it frequently (e.g., we forget to incorporate that an obscure new device called an iPad was just launched).
* We build a spam filter. It works well at detecting all spam that we have seen so far. But then the spammers wise up and craft new messages that look unlike anything we have seen before.
* We build a product recommendation system. It works throughout the winter but then continues to recommend Santa hats long after Christmas.

### Further Failure Modes

* We build a face detector. It works well on all benchmarks. Unfortunately it fails on test data: the offending examples are close-ups where the face fills the entire image (no such data was in the training set).
* We build a web search engine for the US market and want to deploy it in the UK.
* We train an image classifier by compiling a large dataset where each among a large set of classes is equally represented in the dataset, say 1000 categories, represented by 1000 images each. Then we deploy the system in the real world, where the actual label distribution of photographs is decidedly non-uniform.






## Correction of Distribution Shift

As we have discussed, there are many cases
where training and test distributions
$P(\mathbf{x}, y)$ are different.
In some cases, we get lucky and the models work
despite covariate, label, or concept shift.
In other cases, we can do better by employing
principled strategies to cope with the shift.
The remainder of this section grows considerably more technical.
The impatient reader could continue on to the next section
as this material is not prerequisite to subsequent concepts.

Recall from :numref:`subsec_empirical-risk-and-risk` the distinction
between the *empirical risk* :eqref:`eq_empirical-risk-min` (the average
loss on the training data) and the *risk*
:eqref:`eq_true-risk` (the expected loss under the true data
distribution $p(\mathbf{x}, y)$). In practice we cannot evaluate the risk
directly and so we turn to *empirical risk minimization*, hoping that
minimizing the empirical risk on the training set will approximately
minimize the risk.



### Covariate Shift Correction
:label:`subsec_covariate-shift-correction`

Assume that we want to estimate
some dependency $P(y \mid \mathbf{x})$
for which we have labeled data $(\mathbf{x}_i, y_i)$.
Unfortunately, the observations $\mathbf{x}_i$ are drawn
from some *source distribution* $q(\mathbf{x})$
rather than the *target distribution* $p(\mathbf{x})$.
Fortunately,
the dependency assumption means
that the conditional distribution does not change: $p(y \mid \mathbf{x}) = q(y \mid \mathbf{x})$.
If the source distribution $q(\mathbf{x})$ is "wrong",
we can correct for that by reweighting the risk with the following simple
identity :cite:`Shimodaira.2000`:

$$
\begin{aligned}
\int\int l(f(\mathbf{x}), y) p(y \mid \mathbf{x})p(\mathbf{x}) \;d\mathbf{x}dy =
\int\int l(f(\mathbf{x}), y) q(y \mid \mathbf{x})q(\mathbf{x})\frac{p(\mathbf{x})}{q(\mathbf{x})} \;d\mathbf{x}dy.
\end{aligned}
$$
:eqlabel:`eq_covariate-shift-identity`

In other words, we need to reweigh each data example
by the ratio of the
probability
that it would have been drawn from the correct distribution to that from the wrong one:

$$\beta_i \stackrel{\textrm{def}}{=} \frac{p(\mathbf{x}_i)}{q(\mathbf{x}_i)}.$$

Plugging in the weight $\beta_i$ for
each data example $(\mathbf{x}_i, y_i)$
we can train our model using
*weighted empirical risk minimization*:

$$\mathop{\mathrm{minimize}}_f \frac{1}{n} \sum_{i=1}^n \beta_i l(f(\mathbf{x}_i), y_i).$$
:eqlabel:`eq_weighted-empirical-risk-min`



Alas, we do not know that ratio,
so before we can do anything useful we need to estimate it.
Many methods estimate this ratio directly, matching moments of the reweighted
source to the target without ever estimating $p$ and $q$ separately
:cite:`Gretton.Borgwardt.Rasch.ea.2012`.
Note that for any such approach, we need samples
drawn from both distributions: the "true" $p$, e.g.,
by access to test data, and the one used
for generating the training set $q$ (the latter is trivially available).
Note however, that we only need features $\mathbf{x} \sim p(\mathbf{x})$;
we do not need to access labels $y \sim p(y)$.

In this case, there exists a very effective approach
that will give almost as good results as the original: namely, logistic regression,
which is a special case of softmax regression (see :numref:`sec_softmax`)
for binary classification.
This is all that is needed to compute estimated probability ratios.
We learn a classifier to distinguish
between data drawn from $p(\mathbf{x})$
and data drawn from $q(\mathbf{x})$.
If it is impossible to distinguish
between the two distributions
then it means that the associated instances
are equally likely to come from
either one of those two distributions.
On the other hand, any instances
that can be well discriminated
should be significantly overweighted
or underweighted accordingly.

For simplicity's sake assume that we have
an equal number of instances from both distributions
$p(\mathbf{x})$
and $q(\mathbf{x})$, respectively.
Now denote by $z$ labels that are $1$
for data drawn from $p$ and $-1$ for data drawn from $q$.
Then the probability in a mixed dataset is given by

$$P(z=1 \mid \mathbf{x}) = \frac{p(\mathbf{x})}{p(\mathbf{x})+q(\mathbf{x})} \textrm{ and hence } \frac{P(z=1 \mid \mathbf{x})}{P(z=-1 \mid \mathbf{x})} = \frac{p(\mathbf{x})}{q(\mathbf{x})}.$$

Thus, if we use a logistic regression approach,
where $P(z=1 \mid \mathbf{x})=\frac{1}{1+\exp(-h(\mathbf{x}))}$ ($h$ is a parametrized function),
it follows that

$$
\beta_i = \frac{1/(1 + \exp(-h(\mathbf{x}_i)))}{\exp(-h(\mathbf{x}_i))/(1 + \exp(-h(\mathbf{x}_i)))} = \exp(h(\mathbf{x}_i)).
$$

(With unequal sample sizes $\exp(h)$ estimates $p/q$ only up to the constant
$m/n$, which does not affect the weighted minimizer.)

As a result, we need to solve two problems:
the first, to distinguish between
data drawn from both distributions,
and then a weighted empirical risk minimization problem
in :eqref:`eq_weighted-empirical-risk-min`
where we weigh terms by $\beta_i$.

Now we are ready to describe a correction algorithm.
Suppose that we have a training set $\{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$ and an unlabeled test set $\{\mathbf{u}_1, \ldots, \mathbf{u}_m\}$.
For covariate shift,
we assume that $\mathbf{x}_i$ for all $1 \leq i \leq n$ are drawn from some source distribution
and $\mathbf{u}_i$ for all $1 \leq i \leq m$
are drawn from the target distribution.
Here is a prototypical algorithm
for correcting covariate shift:

1. Create a binary-classification training set: $\{(\mathbf{x}_1, -1), \ldots, (\mathbf{x}_n, -1), (\mathbf{u}_1, 1), \ldots, (\mathbf{u}_m, 1)\}$.
1. Train a binary classifier using logistic regression to get the function $h$.
1. Weigh training data using $\beta_i = \exp(h(\mathbf{x}_i))$ or better $\beta_i = \min(\exp(h(\mathbf{x}_i)), c)$ for some constant $c$.
1. Use weights $\beta_i$ for training on $\{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$ in :eqref:`eq_weighted-empirical-risk-min`.

Clipping the weights at a ceiling $c$ trades a little bias for much lower variance:
when source and target barely overlap, a handful of examples acquire enormous weights
$\beta_i$ that would otherwise dominate, and destabilize, the weighted objective.
:numref:`fig_mdl-clf-density-ratio` shows the geometry of the whole construction:
where the target density $p$ exceeds the source density $q$,
the ratio $\beta = p/q$ grows, and it grows *exponentially* fast
out in the tail where the source has almost no mass,
which is exactly where the clip takes over.

![Importance weights for covariate shift. Training data comes from the source density $q$ (left curve) but the risk we care about weights points by the target density $p$ (right curve). The correction weight $\beta(x) = p(x)/q(x)$ is near zero where only the source has mass, crosses $1$ where the densities agree, and explodes where the target outweighs a vanishing source; clipping $\beta$ at a ceiling $c$ (dashed) caps the variance contributed by those rare, enormously weighted examples.](../img/mdl-clf-density-ratio.svg)
:label:`fig_mdl-clf-density-ratio`

Note that the above algorithm relies on one assumption.
For this scheme to work, we need that each data example
in the target (e.g., test time) distribution
had nonzero probability of occurring at training time.
If we find a point where $p(\mathbf{x}) > 0$ but $q(\mathbf{x}) = 0$,
then the corresponding importance weight should be infinity.

#### Covariate Shift Correction in Code

The entire pipeline, from discriminator to reweighted training, fits in a
few lines, so let us watch it work. We make the shift two-dimensional so that
it is drastic but visible: source inputs are Gaussian around the origin,
target inputs are the same Gaussian shifted to be centered at $(2, 0)$, and both share
one labeling rule (covariate shift by construction). The label depends on
$\mathbf{x}$ through a *curved* boundary, so a linear classifier is
misspecified and it matters *where* it spends its capacity. The only trainer
we need is logistic regression by gradient descent, with an optional
per-example weight:

```{.python .input #environment-and-distribution-shift-covariate-shift-correction-1}
import numpy as np

rng = np.random.default_rng(0)
n = 1000
X_src = rng.normal(0.0, 1.0, (n, 2))            # source q: centered at (0, 0)
X_tgt = rng.normal(0.0, 1.0, (n, 2)) + [2, 0]   # target p: centered at (2, 0)
label = lambda X: (X[:, 1] > 0.5 * X[:, 0]**2 - 1).astype(float)
y_src, y_tgt = label(X_src), label(X_tgt)

def fit_logreg(X, y, weights=None, lr=0.1, steps=2000):
    w, b = np.zeros(X.shape[1]), 0.0
    v = np.ones(len(y)) if weights is None else weights / weights.mean()
    for _ in range(steps):
        g = v * (1 / (1 + np.exp(-(X @ w + b))) - y)   # weighted residual
        w -= lr * X.T @ g / len(y)
        b -= lr * g.mean()
    return w, b
```

Step one of the algorithm: pool the source inputs (labeled $z=0$) with the
*unlabeled* target inputs ($z=1$) and train the domain discriminator $h$. For
two unit Gaussians the true log-density-ratio is exactly linear,
$\log (p(\mathbf{x})/q(\mathbf{x})) = 2x_1 - 2$, so we can check the learned $h$
against the truth, and the weights are $\beta_i = \exp(h(\mathbf{x}_i))$:

```{.python .input #environment-and-distribution-shift-covariate-shift-correction-2}
w_h, b_h = fit_logreg(np.concatenate([X_src, X_tgt]),
                      np.concatenate([np.zeros(n), np.ones(n)]))
beta = np.exp(X_src @ w_h + b_h)
print(f'learned h(x) = {w_h[0]:.2f} x1 {w_h[1]:+.2f} x2 {b_h:+.2f} '
      f'(true log-ratio: 2 x1 - 2)')
print(f'beta on source data: mean {beta.mean():.2f}, max {beta.max():.1f}')
```

Step two: train the actual classifier three ways, on the same source data
with the same labels, and evaluate each on the *target* domain, which is the
one we care about:

```{.python .input #environment-and-distribution-shift-covariate-shift-correction-3}
acc = lambda wb, X, y: ((X @ wb[0] + wb[1] > 0) == (y > 0.5)).mean()
for name, wts in [('unweighted', None), ('weighted', beta),
                  ('clipped, c=5', np.minimum(beta, 5))]:
    wb = fit_logreg(X_src, y_src, wts)
    print(f'{name:12s}  target accuracy: {acc(wb, X_tgt, y_tgt):.3f}'
          f'   (source accuracy: {acc(wb, X_src, y_src):.3f})')
```

The unweighted model is fitted where the *source* data lives, so on the
target domain it is no better than a coin flip; reweighting by $\beta$ lifts
target accuracy above 90%, at the price of a worse fit on the now-discounted
source region, exactly the trade the identity
:eqref:`eq_covariate-shift-identity` prescribes. The clipped run bears out the
earlier warning: the largest raw weight here is over 50, so a handful of the
thousand source points would otherwise dominate the objective, and capping
$\beta$ at $c=5$ here even helps a little.
Exercises 3 and 4 let you probe when this pipeline fails, most instructively
when the supports stop overlapping.






### Label Shift Correction

Assume that we are dealing with a
classification task with $k$ categories.
Using the same notation in :numref:`subsec_covariate-shift-correction`,
$q$ and $p$ are the source distribution (e.g., training time) and target distribution (e.g., test time), respectively.
Assume that the distribution of labels shifts over time:
$q(y) \neq p(y)$, but the class-conditional distribution
stays the same: $q(\mathbf{x} \mid y)=p(\mathbf{x} \mid y)$.
If the source distribution $q(y)$ is "wrong",
we can correct for that
according to
the following identity in the risk
as defined in
:eqref:`eq_true-risk`:

$$
\begin{aligned}
\int\int l(f(\mathbf{x}), y) p(\mathbf{x} \mid y)p(y) \;d\mathbf{x}dy =
\int\int l(f(\mathbf{x}), y) q(\mathbf{x} \mid y)q(y)\frac{p(y)}{q(y)} \;d\mathbf{x}dy.
\end{aligned}
$$



Here, our importance weights will correspond to the
label likelihood ratios:

$$\beta_i \stackrel{\textrm{def}}{=} \frac{p(y_i)}{q(y_i)}.$$

One nice thing about label shift is that
if we have a reasonably good model
on the source distribution,
then we can get consistent estimates of these weights
without ever having to deal with the ambient dimension.
In deep learning, the inputs tend
to be high-dimensional objects like images,
while the labels are often simpler objects like categories.

To estimate the target label distribution,
we first take our reasonably good off-the-shelf classifier
(typically trained on the training data)
and compute its confusion matrix $\mathbf{C}$ on the validation set
(also from the training distribution).
Recall the $k \times k$ confusion matrix of :numref:`sec_classification`,
column-normalized exactly as we computed it in :numref:`sec_softmax_scratch`:
entry $c_{ij}$ is the fraction of validation examples of true class $j$
that the model predicted as class $i$, so each column sums to $1$
and estimates $P(\hat{y}=i \mid y=j)$.

Now, we cannot calculate the confusion matrix
on the target data directly
because we do not get to see the labels for the examples
that we see in the wild,
unless we invest in a complex real-time annotation pipeline.
What we can do, however, is average all of our model's predictions
at test time together, yielding the mean model outputs $\mu(\hat{\mathbf{y}}) \in \mathbb{R}^k$,
where the $i^\textrm{th}$ element $\mu(\hat{y}_i)$
is the fraction of the total predictions on the test set
where our model predicted $i$.

It turns out that under some mild conditions, namely that
our classifier was reasonably accurate in the first place,
that the target data contains only categories
that we have seen before,
and that the label shift assumption holds in the first place
(the strongest assumption here), we can estimate the test set label distribution
by solving a simple linear system

$$\mathbf{C} p(\mathbf{y}) = \mu(\hat{\mathbf{y}}),$$

because as an estimate $\sum_{j=1}^k c_{ij} p(y_j) = \mu(\hat{y}_i)$ holds for all $1 \leq i \leq k$,
where $p(y_j)$ is the $j^\textrm{th}$ element of the $k$-dimensional label distribution vector $p(\mathbf{y})$.
If our classifier is accurate enough that $\mathbf{C}$
is diagonally dominant (each class is predicted correctly
more often than it is mistaken for any collection of others),
then $\mathbf{C}$ will be invertible,
and we get a solution $p(\mathbf{y}) = \mathbf{C}^{-1} \mu(\hat{\mathbf{y}})$.
This confusion-matrix estimator goes back to :citet:`Saerens.Latinne.Decaestecker.2002`;
:citet:`Lipton.Wang.Smola.2018` showed that, treating the trained classifier as a
black box, it yields *consistent* estimates of the target label distribution under
the label-shift assumption (an approach they call black-box shift estimation).

Because we observe the labels on the source data,
it is easy to estimate the distribution $q(y)$.
Then, for any training example $i$ with label $y_i$,
we can take the ratio of our estimated $p(y_i)/q(y_i)$
to calculate the weight $\beta_i$,
and plug this into weighted empirical risk minimization
in :eqref:`eq_weighted-empirical-risk-min`.


### Concept Shift Correction

Concept shift is much harder to fix in a principled manner.
For instance, in a situation where suddenly the problem changes
from distinguishing cats from dogs to one of
distinguishing white from black animals,
it will be unreasonable to assume
that we can do much better than just collecting new labels
and training from scratch.
Fortunately, in practice, such extreme shifts are rare.
Instead, what usually happens is that the task keeps on changing slowly.
To make things more concrete, here are some examples:

* In computational advertising, new products are launched,
old products become less popular. This means that the distribution over ads and their popularity changes gradually and any click-through rate predictor needs to change gradually with it.
* Traffic camera lenses degrade gradually due to environmental wear, affecting image quality progressively.
* News content changes gradually (i.e., most of the news remains unchanged but new stories appear).

In such cases, we can use the same approach that we used for training networks to make them adapt to the change in the data. In other words, we use the existing network weights and simply perform a few update steps with the new data rather than training from scratch.


## A Taxonomy of Learning Problems

Armed with knowledge about how to deal with changes in distributions, we can now consider some other aspects of machine learning problem formulation.


### Batch Learning

In *batch learning*, we have access to training features and labels $\{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$, which we use to train a model $f(\mathbf{x})$. Later on, we deploy this model to score new data $(\mathbf{x}, y)$ drawn from the same distribution. This is the default assumption for any of the problems that we discuss here. For instance, we might train a cat detector based on lots of pictures of cats and dogs. Once we have trained it, we ship it as part of a smart catdoor computer vision system that lets only cats in. This is then installed in a customer's home and is never updated again (barring extreme circumstances).


### Online Learning

Now imagine that the data $(\mathbf{x}_i, y_i)$ arrives one sample at a time. More specifically, assume that we first observe $\mathbf{x}_i$, then we need to come up with an estimate $f(\mathbf{x}_i)$. Only once we have done this do we observe $y_i$ and so receive a reward or incur a loss, given our decision.
Many real problems fall into this category. For example, we need to predict tomorrow's stock price, which allows us to trade based on that estimate and at the end of the day we find out whether our estimate made us a profit. In other words, in *online learning*, we have the following cycle where we are continuously improving our model given new observations:

$$\begin{aligned}&\textrm{model } f_t \longrightarrow \textrm{data }  \mathbf{x}_t \longrightarrow \textrm{estimate } f_t(\mathbf{x}_t) \longrightarrow\\ \textrm{obs}&\textrm{ervation } y_t \longrightarrow \textrm{loss } l(y_t, f_t(\mathbf{x}_t)) \longrightarrow \textrm{model } f_{t+1}\end{aligned}$$

### Bandits

A *bandit* problem supplies only partial feedback: after choosing an action, the
learner observes the reward for that action rather than labels or rewards for
all alternatives. The classical multi-armed bandit has finitely many actions;
contextual and continuous-action variants are also common. This feedback
structure distinguishes bandits from ordinary supervised online learning.


### Control

In control problems, an action changes the system state and therefore affects
later observations. A coffee-boiler controller, for example, sees a temperature
that depends on its earlier heating decisions; a PID
(proportional--integral--derivative) controller is one standard approach.
Likewise, recommendations shown to a user can change what the user reads next.
Sequential dynamics distinguish control from ordinary online prediction.




### Reinforcement Learning

*Reinforcement learning* studies sequential decisions in which actions affect
future states and rewards may be delayed. It includes single-agent control as
well as cooperative and competitive multiagent problems. Chess, Go, and
StarCraft are multiagent examples; autonomous driving is a control problem in
which other road users also respond to the vehicle's actions. Partial
observability, delayed credit, and exploration are separate difficulties that
may occur in these settings.

### Considering the Environment

One key distinction between the different situations above is that a strategy that might have worked throughout in the case of a stationary environment, might not work throughout in an environment that can adapt. For instance, an arbitrage opportunity discovered by a trader is likely to disappear once it is exploited. The speed and manner at which the environment changes determines to a large extent the type of algorithms that we can bring to bear. For instance, if we know that things may only change slowly, we can force any estimate to change only slowly, too. If we know that the environment might change instantaneously, but only very infrequently, we can make allowances for that. These types of knowledge are what let the aspiring data scientist deal with concept shift, i.e., when the problem that is being solved can change over time.




## Fairness, Accountability, and Transparency in Machine Learning

Deploying a machine learning system often turns predictions into decisions
that affect people.
These technical systems can impact the lives
of individuals who are subject to the resulting decisions.
The leap from considering predictions to making decisions
raises new ethical questions alongside the technical ones,
and these must be carefully considered.
If we are deploying a medical diagnostic system,
we need to know for which populations
it may work and for which it may not.
Overlooking foreseeable risks to the welfare of
a subpopulation could cause us to administer inferior care.
Moreover, once we contemplate decision-making systems,
we must step back and reconsider how we evaluate our technology.
Among other consequences of this change of scope,
we will find that *accuracy* is seldom the right measure.
For instance, when translating predictions into actions,
we will often want to take into account
the potential cost sensitivity of erring in various ways.
The cost of an error can differ across decisions and populations. Thresholds
should therefore be chosen from an explicit loss model and evaluated separately
for affected groups. Threshold adjustment alone does not establish fairness:
different fairness criteria can conflict, and the labels, data-collection
process, and decision policy may themselves create harm. This section only
identifies the problem; a later treatment develops the definitions and their
limitations.
We also want to be careful about
how prediction systems can lead to feedback loops.
For example, consider predictive policing systems,
which allocate patrol officers
to areas with high forecasted crime.
It is easy to see how a worrying pattern can emerge:

 1. Neighborhoods with more crime get more patrols.
 1. Consequently, more crimes are discovered in these neighborhoods, entering the training data available for future iterations.
 1. Exposed to more positives, the model predicts yet more crime in these neighborhoods.
 1. In the next iteration, the updated model targets the same neighborhood even more heavily leading to yet more crimes discovered, etc.

Often, the various mechanisms by which
a model's predictions become coupled to its training data
are unaccounted for in the modeling process.
This can lead to what researchers call *runaway feedback loops*.
Additionally, we want to be careful about
whether we are addressing the right problem in the first place.
Predictive algorithms now play an outsize role
in mediating the dissemination of information.
Should the news that an individual encounters
be determined by the set of Facebook pages they have *Liked*?
These are just a few among the many pressing ethical dilemmas
that you might encounter in a career in machine learning.


## Summary

In many cases training and test sets do not come from the same distribution. This is called distribution shift.
The risk is the expectation of the loss over the entire population of data drawn from their true distribution. However, this entire population is usually unavailable. Empirical risk is an average loss over the training data to approximate the risk. In practice, we perform empirical risk minimization.

Under covariate- or label-shift assumptions, unlabeled target data can support
specific reweighting corrections. A change in the input marginal may be
detectable without labels, but the claim that $P(y\mid\mathbf{x})$ or
$P(\mathbf{x}\mid y)$ stayed fixed is not generally identifiable from those
data alone. Corrections therefore depend on an assumption that must be defended
from domain knowledge and checked when target labels become available.
In some cases, the environment may remember automated actions and respond in surprising ways. We must account for this possibility when building models and continue to monitor live systems, open to the possibility that our models and the environment will become entangled in unanticipated ways.

These ideas predate the current era of large pretrained models, but
distribution shift has only become more central since, as a foundation model is
routinely deployed on domains, users, and time periods unlike its training
corpus. Curated benchmarks such as WILDS :cite:`Koh.Sagawa.Marklund.ea.2021`
show that models with strong in-distribution accuracy can still degrade sharply
out of distribution, and that a correction which helps on one shift often fails
on another, so it pays to measure on the shift you actually face.

## Exercises

1. If you change the behavior of a search engine, how might users respond? How might advertisers respond? Explain why this is an instance of the feedback loop described for the loan/footwear example at the start of the section.
1. Starting from the risk under the target distribution $p(\mathbf{x}, y)$, derive the covariate-shift reweighting identity :eqref:`eq_covariate-shift-identity` (whose sample version is the weighted objective :eqref:`eq_weighted-empirical-risk-min`), and state precisely the assumption on the supports of $p(\mathbf{x})$ and $q(\mathbf{x})$ under which the importance weights $\beta_i=p(\mathbf{x}_i)/q(\mathbf{x}_i)$ are finite.
1. Implement a covariate shift detector. Take any labeled dataset and create a shifted copy of the features (e.g., add Gaussian noise, or subsample by thresholding one feature). Train a logistic-regression classifier to distinguish "original" from "shifted" inputs and report its accuracy. Relate the accuracy to how detectable the shift is, and to the classifier-as-shift-detector idea in :numref:`subsec_covariate-shift-correction`. *Hint: if the classifier cannot beat chance, the two distributions are indistinguishable from these features.*
1. Implement a covariate shift corrector. Using the classifier from the previous exercise, compute weights $\beta_i=\exp(h(\mathbf{x}_i))$, retrain your downstream model with weighted empirical risk minimization :eqref:`eq_weighted-empirical-risk-min`, and compare its target-domain accuracy with and without reweighting. What happens to the variance of the $\beta_i$ as the shift grows, and how does clipping $\beta_i\leftarrow\min(\beta_i,c)$ help?
1. You have a $k$-class classifier and its validation confusion matrix $\mathbf{C}$. Show that the linear system $\mathbf{C}\, p(\mathbf{y})=\mu(\hat{\mathbf{y}})$ follows from the law of total probability under the label-shift assumption, and explain why $\mathbf{C}$ must be invertible for the estimate $p(\mathbf{y})=\mathbf{C}^{-1}\mu(\hat{\mathbf{y}})$ to be usable.
1. Besides distribution shift, what else could make the empirical risk a poor approximation of the risk? *Hint: think about dependence between examples, and about the loss not matching the deployment objective.*


[Discussions](https://d2l.discourse.group/t/105)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §4.7]{.kicker}

When the world stops matching the training set<br>**Distribution shift**: how it breaks models, and what we can do about it.
:::
:::

::: {.slide title="The question we skipped"}
[Why this matters]{.kicker}

We fit models to data and measure test accuracy. But we rarely ask **where the data came from** or **what the prediction will be used for**.

. . .

::: {.d2l-note .warn}
A loan model finds that **Oxfords repay, sneakers default**. Approve everyone in Oxfords, and soon *everyone* wears Oxfords, with no change in who actually repays. The decision **broke the signal**.
:::

. . .

This is **Goodhart's law**: *when a measure becomes a target, it ceases to be a good measure.* Deploying a model can perturb the very distribution it was trained on.
:::

::: {.slide title="Train here, deploy there"}
[The setup]{.kicker}

Training data is drawn from a **source** distribution $p_S(\mathbf{x}, y)$; at test time we meet a **target** $p_T(\mathbf{x}, y)$ that may differ.

. . .

::: {.d2l-note}
**With no link between $p_S$ and $p_T$, learning cannot transfer.** Suppose the inputs are unchanged, $p_S(\mathbf{x})=p_T(\mathbf{x})$, but every label flips, $p_S(y\mid\mathbf{x})=1-p_T(y\mid\mathbf{x})$: "cats" become "dogs" overnight. No algorithm can tell this apart from no shift at all.
:::

The way out is **structure**: assume *how* the world may change, and that assumption buys us detection, sometimes correction.
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[Three Kinds of Shift]{.dtitle}

[what stays fixed tells you what to do]{.dsub}
:::
:::

::: {.slide title="Covariate shift: the inputs move"}
[Three kinds of shift]{.kicker}

::: {.cols .vc}
::: {.col}
The input distribution $P(\mathbf{x})$ changes, but the **labeling rule** $P(y\mid\mathbf{x})$ holds.

The natural assumption when **$\mathbf{x}$ causes $y$**: a cat is a cat whether photographed or drawn.

::: {.d2l-note}
Train on **photos**, test on **cartoons** of the same animals. Same labels, very different pixels, and trouble without a plan to adapt.
:::
:::

::: {.col .fig}
![Source: photographs. Target: cartoons. $P(\mathbf{x})$ shifts; the cat-vs-dog rule does not.](../img/cat-dog-train.png){width=100%}
:::
:::
:::

::: {.slide title="Label shift: the mix moves"}
[Three kinds of shift]{.kicker}

The label frequencies $P(y)$ change, but each class still **looks the same**: $P(\mathbf{x}\mid y)$ is fixed.

. . .

The natural assumption when **$y$ causes $\mathbf{x}$**: diseases cause symptoms, so as an outbreak shifts how common a diagnosis is, the symptom pattern per disease is unchanged.

. . .

::: {.d2l-note .rule}
**Why prefer it when both could apply?** Its corrections live in **label space** (low-dimensional categories), not in high-dimensional input space, exactly the cheap side in deep learning.
:::
:::

::: {.slide title="Concept shift: the labels themselves move"}
[Three kinds of shift]{.kicker}

::: {.cols .vc}
::: {.col}
Now the **definition** of a label drifts: $P(y\mid\mathbf{x})$ changes because what counts as the answer changed.

What people call a *soft drink* depends on **where you ask** ("soda", "pop", "coke").

::: {.d2l-note}
Diagnostic criteria, fashion, and job titles can drift this way across time or
geography. Gradual drift can sometimes be tracked with fresh labeled data;
abrupt changes require faster detection and may require a new model.
:::
:::

::: {.col .fig}
![Concept shift: the name for the same drink across the US (PopVsSoda.com, CC-BY: Alan McConchie).](../img/popvssoda.png){width=100%}
:::
:::
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[When It Bites]{.dtitle}

[spurious features and slow drift]{.dsub}
:::
:::

::: {.slide title="The model learned the wrong thing"}
[When it bites]{.kicker}

A blood-test startup drew **healthy controls from students**, sick patients from the clinic. The classifier hit near-perfect accuracy, on age, hormones, and diet, **not the disease**.

. . .

::: {.d2l-note .warn}
**The tank fable.** A net "detects tanks" perfectly on held-out images, then fails in the field. The tank photos were shot at noon, the empty ones at dawn: it learned the **lighting**, never the tank.
:::

A spurious feature, present in your sample but gone at deployment, fools a model that never saw the distinction you care about.
:::

::: {.slide title="Slow drift, stale model"}
[When it bites]{.kicker}

The subtler failure: the distribution moves **gradually** (a *nonstationary* world) and the model is never refreshed.

. . .

::: {.cols}
::: {.col}
::: {.d2l-note}
A **spam filter** stops working once spammers craft messages unlike any seen before.
:::
:::

::: {.col}
::: {.d2l-note}
A **recommender** keeps pushing Santa hats long after Christmas.
:::
:::
:::

The signal did not break all at once, it eroded while nobody was watching.
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[Correcting Shift]{.dtitle}

[reweighting the risk we cannot see]{.dsub}
:::
:::

::: {.slide title="Risk vs. empirical risk"}
[The frame]{.kicker}

What we *want* to minimize is the **risk**: expected loss under the true distribution $p(\mathbf{x}, y)$.

$$R(f) = \mathbb{E}_{(\mathbf{x}, y)\sim p}\,[\,l(f(\mathbf{x}), y)\,].$$

. . .

We cannot evaluate it, so we minimize the **empirical risk**, the average loss on the training sample, and *hope* the two agree.

$$\hat{R}(f) = \frac{1}{n}\sum_{i=1}^{n} l(f(\mathbf{x}_i), y_i).$$

. . .

Under shift, the training sample comes from the **wrong** distribution, so this hope fails, unless we correct the average.
:::

::: {.slide title="Reweight to the right distribution"}
[Covariate shift correction]{.kicker}

Labeled data comes from source $q(\mathbf{x})$, but we care about target $p(\mathbf{x})$. Because $p(y\mid\mathbf{x})=q(y\mid\mathbf{x})$, one identity fixes the risk:

$$\mathbb{E}_{p}[\,l\,] = \mathbb{E}_{q}\!\left[\,\frac{p(\mathbf{x})}{q(\mathbf{x})}\, l\,\right].$$

. . .

So **reweight each example** by how much more likely it is under the target than the source, and minimize a *weighted* empirical risk:

$$\beta_i = \frac{p(\mathbf{x}_i)}{q(\mathbf{x}_i)}, \qquad \min_f\ \frac{1}{n}\sum_{i=1}^{n}\beta_i\, l(f(\mathbf{x}_i), y_i).$$
:::

::: {.slide title="A classifier estimates the weights"}
[Covariate shift correction]{.kicker}

We do not know $p/q$. But **train a classifier to tell source from target** ($z=+1$ for target, $-1$ for source), and the odds *are* the ratio:

$$\frac{P(z{=}1\mid\mathbf{x})}{P(z{=}{-}1\mid\mathbf{x})} = \frac{p(\mathbf{x})}{q(\mathbf{x})}.$$

. . .

With a logistic model $P(z{=}1\mid\mathbf{x})=\sigma(h(\mathbf{x}))$ this collapses to $\beta_i = \exp(h(\mathbf{x}_i))$. We need only **unlabeled** target features $\mathbf{x}\sim p$.
:::

::: {.slide title="Where the weights explode, and why we clip"}
[Covariate shift correction · geometry]{.kicker}

![Training data comes from the source $q$ (left curve); the risk we care about weights points by the target $p$ (right curve). The weight $\beta = p/q$ is near zero where only the source has mass, crosses $1$ where the densities agree, and explodes out in the tail where the source has almost nothing; the dashed line clips it at a ceiling $c$.](../img/mdl-clf-density-ratio.svg){width=88%}

::: {.d2l-note .rule}
**Clip** $\beta_i \leftarrow \min(\exp(h(\mathbf{x}_i)), c)$: where the
domains barely overlap, a few examples grab enormous weights and dominate
the objective, so a little bias buys much less variance. If $p > 0$ where
$q = 0$, the true weight is *infinite*: no reweighting can conjure data
that was never sampled.
:::
:::

::: {.slide title="The discriminator recovers the truth" only="pytorch"}
[Covariate shift correction · watch it work]{.kicker}

A 2-D rig: source Gaussian at the origin, target the same Gaussian shifted
to $(2, 0)$, one shared *curved* labeling rule (covariate shift by
construction), with a known answer: the true log-ratio is $2x_1 - 2$. Pool
the inputs, train the domain classifier $h$:

@!environment-and-distribution-shift-covariate-shift-correction-2

::: {.d2l-note}
Learned: $2.06\,x_1 + 0.09\,x_2 - 2.03$. The discriminator *is* the density
ratio, and note the $\beta$ tail: one source point already carries weight
$56$.
:::
:::

::: {.slide title="Reweighting turns a coin flip into 0.93" only="pytorch"}
[Covariate shift correction · the verdict]{.kicker}

Train the actual classifier three ways on the *same* labeled source data;
evaluate on the **target**, the domain we care about:

@!environment-and-distribution-shift-covariate-shift-correction-3

::: {.d2l-note .rule}
Unweighted fits where the *source* lives: **0.502** on the target, a coin
flip. Reweighting: **0.933**, bought by a worse fit on the discounted
source region, exactly the trade the identity prescribes. Clipping at
$c=5$ tames the $\beta > 50$ outliers and even helps: **0.945**.
:::
:::

::: {.slide title="Watch it work: 0.502 → 0.933 → 0.945" except="pytorch"}
[Covariate shift correction · the verdict]{.kicker}

A 2-D rig: source Gaussian at the origin, target shifted to $(2, 0)$, one
shared curved labeling rule, so the true log-ratio is known: $2x_1 - 2$.

. . .

- The logistic discriminator recovers $2.06\,x_1 + 0.09\,x_2 - 2.03$: the
  density ratio, learned from unlabeled inputs alone.
- Target accuracy, three ways: **unweighted 0.502** (a coin flip, since the
  model fit where the *source* lives), **weighted 0.933**, **clipped at
  $c{=}5$: 0.945**.

::: {.d2l-note .rule}
Reweighting pays on the target by discounting the source region, exactly
the trade the identity prescribes; the clip tames raw weights that reach
$\beta > 50$ and even helps.
:::
:::

::: {.slide title="Label shift: invert a confusion matrix"}
[Label shift correction]{.kicker}

Here $P(y)$ shifts while $P(\mathbf{x}\mid y)$ is fixed, so the weights are label ratios $\beta_i=p(y_i)/q(y_i)$, and we never touch the high-dimensional inputs.

. . .

Take an off-the-shelf classifier, measure its $k\times k$ **confusion matrix** $\mathbf{C}$ on a source validation set (the very matrix we computed for Fashion-MNIST in §4.4, column-normalized), and the **average prediction** $\mu(\hat{\mathbf{y}})$ on the (unlabeled) target. They are linked by total probability:

$$\mathbf{C}\, p(\mathbf{y}) = \mu(\hat{\mathbf{y}}) \quad\Longrightarrow\quad p(\mathbf{y}) = \mathbf{C}^{-1}\mu(\hat{\mathbf{y}}).$$

. . .

The system requires a **nonsingular** $\mathbf{C}$. Strict diagonal dominance
(each diagonal entry exceeds the sum of the other entries in its row) is one
sufficient condition; then form $\beta_i$ and reweight.
:::

::: {.slide title="Concept shift: just keep up"}
[Concept shift correction]{.kicker}

When the labels are redefined, there is no clever reweighting, the old answers are simply wrong.

. . .

When concept shift is gradual, as in changing ads or news, fresh labeled data
can reveal the moving target. One practical response is:

::: {.d2l-note}
Keep the current weights and **take a few update steps on fresh data**, rather than retraining from scratch. Let the model track the moving target.
:::
:::

::: {.slide}
::: {.divider}
[04]{.dnum}

[Beyond Passive Prediction]{.dtitle}

[when the environment reacts to you]{.dsub}
:::
:::

::: {.slide title="A taxonomy of learning problems"}
[The bigger picture]{.kicker}

Everything above assumed we *passively predict*. The environment can also **react**:

. . .

- **Batch:** train once, deploy, never update (the smart catdoor).
- **Online:** data arrives one point at a time; predict, then learn from the outcome.
- **Bandits:** online, but a finite set of actions, so stronger guarantees.
- **Control & RL:** the environment **remembers** and responds, possibly adversarially (a thermostat, a chess opponent, other cars).

. . .

A strategy that is safe in a stationary world can fail once the world adapts to it, an arbitrage trade vanishes the moment it is exploited.
:::

::: {.slide title="Predictions become decisions"}
[Fairness & feedback]{.kicker}

Deploying a model is rarely *just* prediction, it **automates decisions** about people, where **accuracy is seldom the right measure** (the costs of different errors differ).

. . .

::: {.d2l-note .warn}
**Predictive policing runaway loop.** More patrols → more crime *recorded* in that area → the model predicts even more crime there → still more patrols. The data feeds back into the model, and the loop runs away.
:::

Watch for feedback loops, cost-sensitive errors, and whether you are solving the right problem at all.
:::

::: {.slide title="Shift in the foundation-model era"}
[The modern picture]{.kicker}

Benchmarks like **WILDS** collect *real* shifts (hospitals, cameras,
countries, time) along an axis **orthogonal** to our mechanism taxonomy:

::: {.cols}
::: {.col}
::: {.d2l-note}
**Domain generalization:** test domains never seen in training. *Camelyon17*:
a tumor classifier trained on a few hospitals' slides must survive a **new
hospital's** staining quirks.
:::
:::

::: {.col}
::: {.d2l-note}
**Subpopulation shift:** same domains, new proportions, so what matters is
**worst-group** accuracy. *CivilComments*: average toxicity accuracy conceals
much larger errors on some demographic groups.
:::
:::
:::

. . .

::: {.d2l-note .warn}
**OOD detection ≠ shift correction.** Detection *rejects* inputs the model
cannot handle; correction *reweights* for a target that is here to stay. A
deployed system needs both, and fixes that shine on one shift routinely
fail on another, so measure on the shift you actually face.
:::
:::

::: {.slide title="Summary"}
[Wrap-up]{.kicker}

::: {.cols}
::: {.col}
- **Shift** = train and test distributions differ; failing to notice is a top cause of deployment disasters.
- **Three kinds:** *covariate* ($P(\mathbf{x})$ moves, $\mathbf{x}\!\to\!y$), *label* ($P(y)$ moves, $y\!\to\!\mathbf{x}$), *concept* (the labels themselves move).
:::

::: {.col}
- **Correct** covariate shift by reweighting with $\beta_i=p(\mathbf{x}_i)/q(\mathbf{x}_i)$, estimated by a source-vs-target classifier (demo: target accuracy $0.502 \to 0.933$, clipped $0.945$); label shift by inverting the confusion matrix.
- **Beware the environment:** it may remember your actions and feed them back. Measure on the shift you actually face (WILDS), and keep monitoring live systems.
:::
:::
:::
