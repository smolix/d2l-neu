```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Measuring Classifiers
:label:`sec_classification`

The softmax regression of :numref:`sec_softmax_scratch` and
:numref:`sec_softmax_concise` labels about 83% of the Fashion-MNIST
validation images correctly. Is a classifier with higher accuracy the
better one? Only if every example matters equally and every mistake is
equally bad, since accuracy counts every example once and treats every
mistake alike. Three kinds of decision problem break these assumptions, and
they are the common ones: one class is much rarer than the others, some
mistakes cost more than others, and the decision is made on the predicted
probability itself rather than on the predicted class. Each calls for a
measurement of its own. We study the first two on a screening test whose
outcomes we can count by hand, and the third on the trained softmax
regression, whose probabilities we have so far taken on trust.

```{.python .input #classification-the-base-classification-model}
%%tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #classification-the-base-classification-model}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #classification-the-base-classification-model}
%%tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #classification-the-base-classification-model}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## The Limits of Accuracy
:label:`subsec_beyond-accuracy`

### Class Imbalance

Consider screening for a disease that affects 1% of the population. A
classifier that ignores its input and always predicts *healthy* is right
for 99% of the people, an accuracy of 0.99, and finds no sick patient. Now
take a screening test that detects 90% of the sick and wrongly flags 5% of
the healthy. Among 100,000 people, 1,000 of them sick, it finds 900 and
misses 100, and it flags 4,950 of the 99,000 healthy people. We count the
four possible outcomes in a table whose rows are the predictions and whose
columns are the true classes:

| | sick | healthy |
|:--|--:|--:|
| **predicted sick** | 900 | 4,950 |
| **predicted healthy** | 100 | 94,050 |

Such a table of counts is a *confusion matrix*. With the sick as the *positive*
class, the four cells are the *true positives* (TP, 900), *false positives*
(FP, 4,950), *false negatives* (FN, 100), and *true negatives* (TN, 94,050).
The test is right for the 94,950 people on the diagonal, an accuracy of
0.95. Accuracy ranks the useless classifier above the useful one. Every
person counts once, so the 1,000 sick people make up only 1% of the
accuracy, however well or badly a classifier treats them.

Two ratios focus on the sick. The *recall* is the fraction of the
sick that the classifier finds, and the *precision* is the fraction of the
flagged people who are sick:

$$\textrm{recall} = \frac{\textrm{TP}}{\textrm{TP} + \textrm{FN}}, \qquad \textrm{precision} = \frac{\textrm{TP}}{\textrm{TP} + \textrm{FP}}.$$

Recall reads the first column of the table, and precision reads its first
row. For the screening test the recall is $900 / 1{,}000 = 0.9$ and the
precision $900 / 5{,}850 \approx 0.15$.

Recall separates the two classifiers: 0 for the one that never flags
anybody, 0.9 for the test. The precision of the always-healthy classifier is
undefined, since it flags nobody. The precision of the test is a less
pleasant surprise. It is the probability that a flagged person is sick, and
it is only 0.15. This is the effect of Bayes' theorem that
:numref:`subsec_probability_hiv_app` showed for an HIV test: when a disease
is rare, the false positives from the many healthy people outnumber the
true positives, even for an accurate test. Is a test with high recall and
low precision any good? For screening it is, since every flagged person
gets a second, more specific test.

How much does the precision owe to the rarity of the disease? We keep the
test, which finds 90% of the sick and flags 5% of the healthy, and vary the
fraction of sick people in the population.

```{.python .input #classification-beyond-accuracy}
recall, false_alarms = 0.9, 0.05  # Properties of the test
for prevalence in [0.001, 0.01, 0.1, 0.5]:
    tp, fp = recall * prevalence, false_alarms * (1 - prevalence)
    accuracy = tp + (1 - false_alarms) * (1 - prevalence)
    print(f'{prevalence:5.1%} sick: accuracy {accuracy:.3f}, '
          f'recall {recall:.2f}, precision {tp / (tp + fp):.3f}')
```

The same test has a precision of 2% when one person in a thousand is sick
and of 95% when half of them are, while its accuracy hardly moves. Recall
and the rate of false alarms describe the test; precision describes the test
in a population.

### Asymmetric Costs

Missing a sick patient is worse than a false alarm. Suppose that a false
negative costs 100 times as much as a false positive, $\kappa_\textrm{FN} = 100$
and $\kappa_\textrm{FP} = 1$. The cost of a classifier weights the cells of its
confusion matrix by these costs. Per person, the always-healthy classifier
costs $100 \cdot 1{,}000 / 100{,}000 = 1$ and the screening test
$(100 \cdot 100 + 1 \cdot 4{,}950) / 100{,}000 \approx 0.15$, almost seven
times less. With costs, the useful classifier wins.

Costs also change the decision itself. Suppose a model estimates the
probability $p$ that a patient is sick. Predicting *healthy* costs
$\kappa_\textrm{FN}$ with probability $p$, and predicting *sick* costs
$\kappa_\textrm{FP}$ with probability $1 - p$, so the expected cost is smaller
when we predict *sick* exactly if

$$p \, \kappa_\textrm{FN} > (1 - p)\, \kappa_\textrm{FP}, \quad \textrm{that is,} \quad p > \frac{\kappa_\textrm{FP}}{\kappa_\textrm{FP} + \kappa_\textrm{FN}}.$$
:eqlabel:`eq_cost-threshold`

With equal costs the threshold is $\frac{1}{2}$, and for two classes the
rule picks the class with the larger probability, as accuracy assumes. With
$\kappa_\textrm{FN} = 100$ and $\kappa_\textrm{FP} = 1$, the threshold drops to
$1/101 \approx 0.01$: a patient with a 2% chance of being sick should be
flagged. Lowering the threshold raises the recall and lowers the
precision, since more people are flagged. When the costs are unknown,
practitioners often report precision and recall together, or their harmonic
mean, the *F1 score*, which is high only when both are.

For $q$ classes, costs $\kappa_{ij}$ for predicting $i$ when the class is
$j$ weight the entries of the confusion matrix $\mathbf{N}$ of
:eqref:`eq_confusion-matrix` as in the binary case.

## Calibration

### Reliability Diagrams

The rule :eqref:`eq_cost-threshold` acts on the probability $p$ itself, so
its decisions are only as good as the probabilities. A classifier is
*calibrated* if its probabilities match frequencies: of all patients to
whom it assigns $p = 0.02$, 2% are sick. If only 0.2% of them are, the rule
flags all of them, although at the threshold of $0.01$ none of them should
be flagged. Checking this on validation data is simple. We sort the
patients into bins by $p$ and compare the mean of $p$ in each bin with the
fraction of its patients who are sick.

With many classes there is one probability per class to check, and the
usual shortcut is to check only the largest. The *confidence* of a
prediction is the probability of the predicted class. We sort the examples
into bins by their confidence, for instance $[0.5, 0.7)$, $[0.7, 0.9)$, and
$[0.9, 1]$, and compute in each bin the mean confidence and the accuracy. A
*reliability diagram* plots the accuracy of each bin against its mean
confidence. A calibrated classifier lies on the diagonal; points below it
mean that the classifier is more confident than it is right, or
*overconfident*, a common failing of deep networks trained on
cross-entropy. Note, however, that this check says nothing about the small
probabilities: a rule that thresholds $p(\textrm{sick})$ near $0.01$ needs
the check on $p(\textrm{sick})$ itself.

We check the softmax regression of :numref:`sec_softmax_concise`. We train
it as there, with the progress plot switched off since it would repeat the
plot of that section, and collect the logits and the labels of all
validation images as NumPy arrays.

```{.python .input #classification-calibration-1}
data = d2l.FashionMNIST(batch_size=256)
model = d2l.SoftmaxRegression(num_outputs=10, lr=0.1)
trainer = d2l.Trainer(max_epochs=10)
trainer.board.display = False
trainer.fit(model, data)
O, y = [], []
for X, Y in data.val_dataloader():
    O.append(d2l.numpy(model(X)))
    y.append(d2l.numpy(Y))
O, y = np.concatenate(O), np.concatenate(y)
print(f'validation accuracy {(O.argmax(axis=1) == y).mean():.3f}')
```

The confidence of each prediction is the largest entry of the softmax of
its logits. We compute the softmax in NumPy, shifted by the largest logit
as in :numref:`sec_softmax_scratch`.

```{.python .input #classification-calibration-2}
def softmax(O):
    """Row-wise softmax of a NumPy array of logits."""
    E = np.exp(O - O.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)

conf = softmax(O).max(axis=1)  # Confidence of each prediction
correct = (O.argmax(axis=1) == y).astype(float)
```

We sort the predictions into ten bins by their confidence, $[0, 0.1)$,
$[0.1, 0.2)$, and so on up to $[0.9, 1]$, and compute in each bin $b$ the
number of predictions $n_b$, their mean confidence $\bar{c}_b$, and their
accuracy $a_b$. The first bin stays empty, since with $q = 10$ classes the
largest probability is at least $1/10$. The expected calibration error
(ECE) is the mean gap between confidence and accuracy, weighted by the
fraction of the $n$ predictions that fall into each bin,

$$\textrm{ECE} = \sum_b \frac{n_b}{n} \left| a_b - \bar{c}_b \right|.$$

```{.python .input #softmax-regression-concise-calibration-2}
def reliability(conf, correct, num_bins=10):
    """Count, mean confidence, and accuracy per confidence bin; the ECE."""
    bins = np.minimum((conf * num_bins).astype(int), num_bins - 1)
    n = np.bincount(bins, minlength=num_bins)
    mean_conf = np.bincount(bins, conf, num_bins) / np.maximum(n, 1)
    acc = np.bincount(bins, correct, num_bins) / np.maximum(n, 1)
    return n, mean_conf, acc, np.sum(n * np.abs(acc - mean_conf)) / len(conf)

n, mean_conf, acc, ece = reliability(conf, correct)
for b in np.flatnonzero(n):
    print(f'confidence {b / 10:.1f} to {(b + 1) / 10:.1f}: {n[b]:4d} '
          f'predictions, mean {mean_conf[b]:.3f}, accuracy {acc[b]:.3f}')
print(f'ECE: {ece:.3f}')
```

About half of the predictions fall into the top bin, with a mean
confidence near 0.97, and 97% to 98% of them are correct. The two lowest
bins hold between a handful and about 70 predictions, too few to say
anything about their accuracy. From confidence 0.3 up, every bin holds
hundreds of predictions, and its accuracy lies near its mean confidence, in
most bins slightly above it. The reliability diagram shows the bins with at
least 100 predictions, whose accuracy has a standard error of at most 5
points.

```{.python .input #softmax-regression-concise-calibration-3}
used = n >= 100  # Fewer predictions give too noisy an accuracy
d2l.plot([mean_conf[used], [0, 1]], [acc[used], [0, 1]], 'mean confidence',
         'accuracy', legend=['softmax regression', 'calibrated'],
         fmts=('o-', 'k:'), figsize=(3.5, 3.5))
```

The curve stays close to the diagonal, and mostly just above it. Over all
validation images the mean confidence is about 81% and the accuracy about
83%, and the expected calibration error stays below 0.03 in repeated runs.
The model is slightly *under*confident, and its probabilities can be taken
nearly at face value on images like its validation images. This is a
property of the trained model, not of the softmax, and the same check
applies unchanged to every classifier that follows. Accuracy, which uses
only the predicted class, could not have told us any of it.

### Temperature Scaling

When the check finds a gap, the standard remedy is as simple as the
diagnosis. *Temperature scaling* divides all logits by one number $T > 0$,
the temperature of :numref:`sec_softmax`, before the softmax, and fits $T$
by minimizing the cross-entropy on held-out data
:cite:`Guo.Pleiss.Sun.Weinberger.2017`. Dividing by $T$ keeps the order of
the logits, so every prediction and the accuracy stay the same. Only the
confidences move, and with $T > 1$ they move down, the direction an
overconfident network needs. Our model needs the opposite correction, if
any. Its confidences are slightly too low, so the
fitted temperature should come out below 1. We fit $T$ on the first half of
the validation images, by a grid search that minimizes the cross-entropy
written in the stable form of :numref:`sec_softmax_concise`, and measure
the calibration error on the second half before and after scaling.

```{.python .input #classification-temperature-scaling}
def nll(O, y, T):
    """Mean cross-entropy of softmax(O / T), computed from the logits."""
    Z = O / T
    m = Z.max(axis=1)
    lse = m + np.log(np.exp(Z - m[:, None]).sum(axis=1))
    return np.mean(lse - Z[np.arange(len(y)), y])

half = len(y) // 2
Ts = np.linspace(0.5, 2, 151)
T = Ts[np.argmin([nll(O[:half], y[:half], t) for t in Ts])]
ece = lambda Z: reliability(softmax(Z).max(axis=1), correct[half:])[3]
print(f'T = {T:.2f}; ECE on the second half: {ece(O[half:]):.3f} before, '
      f'{ece(O[half:] / T):.3f} after scaling')
```

The fitted temperature lies a little below 1, around 0.9, and scaling moves
the calibration error of the second half by less than a hundredth, in some
runs up rather than down. Softmax regression needed little repair, and at
errors this small the binned estimate is as much noise as signal. For an
overconfident network the same three lines return a $T$ above 1 and a
visibly straighter reliability diagram; the procedure does not care which
way the gap points.

All of these numbers, from the accuracy to the calibration error, were
measured on 10,000 images and would come out a little different on another
10,000. How different, and how far they can be trusted for images the model
has never seen, is the subject of :numref:`chap_classification_generalization`.

## Summary

Accuracy counts every example once and every mistake alike. When one class
is rare, it can rank a useless classifier above a useful one; recall and
precision separate them, and precision depends on how common the class is.
Costs of mistakes turn the predicted probabilities into a decision
threshold, which is only as good as the probabilities are calibrated. A
reliability diagram compares confidence with accuracy, and the expected
calibration error summarizes it. Softmax regression on Fashion-MNIST is
nearly calibrated, slightly underconfident; temperature scaling, which
divides the logits by one fitted number, leaves its accuracy unchanged and
changes its calibration little.

## Exercises

1. **A small confusion matrix.** A classifier for three classes is
   evaluated on 300 validation examples, 100 of each class. With
   predictions in the rows and true classes in the columns, as in
   :eqref:`eq_confusion-matrix`, its confusion matrix is
   $$\mathbf{N} = \begin{pmatrix} 90 & 20 & 0 \\ 10 & 70 & 10 \\ 0 & 10 & 90 \end{pmatrix}.$$
    1. Compute the accuracy, and the recall and the precision of each
       class.
    1. Divide each column of $\mathbf{N}$ by its sum, then each row by its
       sum. State what the diagonal of each result holds.
    1. Suppose the second class becomes ten times as common as each of the
       others while the classifier stays the same. Compute the new accuracy
       and the new precision of the second class, and say which of the
       numbers of the first part are properties of the classifier alone.

1. **Optimal decisions and rejection.** In a multiclass problem, let
   $\kappa_{ij}$ be the cost of predicting class $i$ when the true class is
   $j$, and let $p(j \mid x)$ be the probability of class $j$ given the
   input $x$.
    1. Write the expected cost of predicting $i$ in terms of the costs and
       $p(j \mid x)$, and state the rule that selects the optimal $i$.
       Check that it reduces to :eqref:`eq_cost-threshold` for two classes.
    1. Let a correct guess cost $0$ and a wrong guess cost $\lambda_s$,
       and allow the classifier to *abstain* instead of guessing, at a
       fixed cost $\lambda_r < \lambda_s$. Derive the threshold on the
       top-class probability above which guessing beats abstaining.
    1. In medical diagnosis, deferring to a specialist is cheap relative to
       a misdiagnosis, say $\lambda_r/\lambda_s = 0.05$. Compute the
       threshold and state what the classifier should do with a prediction
       at $0.9$ confidence. Then describe a case in which reporting the
       full distribution $p(y \mid x)$ serves the physician better than
       either guessing or abstaining.

    *Adapted from Kevin Murphy,
    [Probabilistic Machine Learning: An Introduction](https://probml.github.io/pml-book/book1.html),
    Exercise 5.1.*
1. **Loss at equal accuracy.** Two classifiers $A$ and $B$ both achieve
   90% accuracy on a ten-class test set. On every example that $A$
   classifies correctly, it assigns probability $0.91$ to the correct
   class. On every example that $B$ classifies correctly, it assigns
   $0.51$.
    1. Compute the average cross-entropy loss of each classifier on its
       correctly classified examples. Why would knowing only that $B$'s
       *average* correct-class probability is $0.51$ not have sufficed to
       compute $B$'s loss?
    1. Explain why these numbers do not decide which classifier is safer
       to deploy, and list the further information about the two
       classifiers you would need.
1. **Class-weighted loss.** In the screening example of
   :numref:`subsec_beyond-accuracy`, missing a sick patient costs much
   more than a false alarm. Give each class $j$ a weight $w_j > 0$ and
   train $f$ on the class-weighted average loss

   $$\frac{1}{n} \sum_{i=1}^n w_{y^{(i)}}\, l\big(f(\mathbf{x}^{(i)}), y^{(i)}\big),$$

   where $l$ is the cross-entropy loss and $y^{(i)}$ is the class of
   example $i$, so that every example of class $j$ counts $w_j$ times.
    1. Let the training examples be drawn with class prior $p(y)$ and
       class-conditional densities $p(\mathbf{x} \mid y)$. Show that the
       expected class-weighted loss is proportional to the expected
       unweighted loss under the distribution with the same
       class-conditionals and class prior proportional to $w_y\, p(y)$.
       Name the change to the training data, rather than to the loss,
       that has the same effect.
    1. Fix an input $\mathbf{x}$ with class probabilities
       $p(j \mid \mathbf{x})$, and suppose the model can output any
       distribution $\hat{\mathbf{y}}$. Show that the expected weighted loss
       $$-\sum_j w_j\, p(j \mid \mathbf{x}) \log \hat{y}_j$$
       is minimized by $\hat{y}_j \propto w_j\, p(j \mid \mathbf{x})$.
    1. For the two classes healthy and sick, find the threshold on
       $p(\textrm{sick} \mid \mathbf{x})$ above which the $\arg\max$ of
       this $\hat{\mathbf{y}}$ is sick, and choose the weights so that the
       threshold equals that of :eqref:`eq_cost-threshold`. Explain why
       choosing $w_\textrm{sick} > w_\textrm{healthy}$ raises recall, and
       what it does to the number of false positives.
1. **Calibration and usefulness.** A ten-class problem has the same number
   of examples in every class.
    1. A classifier ignores its input and outputs the probability $0.1$ for
       every class, breaking ties at random. Show that it is calibrated,
       and compute its accuracy.
    1. Another classifier is right on 95% of the examples and always
       reports confidence $1$. Draw its reliability diagram and compute its
       expected calibration error.
    1. Explain why calibration and accuracy measure different things, and
       why neither can replace the other when a decision uses the
       threshold :eqref:`eq_cost-threshold`.
1. [code] **ROC curves.** A binary classifier that outputs a score
   $s \in (0, 1)$ for the positive class can be thresholded at any
   $\tau \in [0, 1]$, not just $\frac{1}{2}$: predict positive whenever
   $s > \tau$. The *receiver operating characteristic* (ROC) curve plots
   the true-positive rate (recall) against the false-positive rate
   $\textrm{FP}/(\textrm{FP}+\textrm{TN})$ as $\tau$ sweeps from $1$ down
   to $0$. Simulate $1000$ positive and $1000$ negative examples with
   scores $s = \sigma(z)$, where $\sigma(z) = 1/(1 + \exp(-z))$ is the
   sigmoid, $z \sim \mathcal{N}(\mu, 1)$ for positives, and
   $z \sim \mathcal{N}(0, 1)$ for negatives.
    1. Compute and plot the ROC curve for $\mu \in \{0, 1, 2\}$, and state
       which points of the plot the endpoints $\tau = 1$ and $\tau = 0$
       give.
    1. Show that a classifier whose scores are independent of the labels
       traces the diagonal in expectation.
    1. For $\mu = 2$, plot the *precision-recall curve*, precision against
       recall as $\tau$ sweeps, once with equal numbers of positives and
       negatives and once with positives making up 1% of the examples
       (keep 1000 negatives and draw 10 positives, or reweight). Explain why
       the ROC curve does not change with the prevalence while the
       precision-recall curve drops, and which of the two describes the
       screening test of this section better.
   :exlabel:`ex_classification-roc-curves`
1. [code] **AUC as ranking probability.** ● This problem uses the ROC
   curves and simulated scores of problem :exref:`ex_classification-roc-curves`.
    1. Show that, when no two scores tie, the area under the ROC curve
       equals the probability that a randomly chosen positive example
       receives a higher score than a randomly chosen negative one, which
       makes it a threshold-free summary of ranking quality.
    1. Compute this probability in closed form for the simulated scores
       and compare it with the trapezoidal area under each of the curves.
    1. Apply the ROC curve and the AUC to the trained softmax regression
       on the validation images of two classes $a$ and $b$, with $a$ as the
       positive class and the score $s = \sigma(o_a - o_b)$, the sigmoid of
       the difference of the two logits: sneaker versus sandal, and shirt
       versus T-shirt. Compare the ordering of the two AUCs with the
       entries of the confusion matrix $\mathbf{C}$ of
       :numref:`sec_softmax_scratch` for the same pairs, and explain why
       the two diagnostics need not agree.
   :exlabel:`ex_classification-auc-ranking-probability`

1. [code] **Temperature scaling.** The text fits $T$ on the first half of
   the validation images.
    1. Draw the reliability diagrams of the second half before and after
       scaling, and show that the accuracy does not change.
    1. Fit $T$ on the training images instead and compare it with the $T$
       of the text. Explain why the fit should use data that the model was
       not trained on.
    1. Compute the expected calibration error of the unscaled model with 5,
       10, and 20 bins. How much does the number depend on the binning?

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/6808)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/6809)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/6810)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/17981)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §3.5]{.kicker}

Measuring Classifiers<br>**imbalance · costs · calibration**
:::
:::

::: {.slide title="A Higher Accuracy Is Not Always the Better Classifier"}
[The problem]{.kicker}

Softmax regression labels about **83%** of the Fashion-MNIST validation
images correctly. Would 85% be better? Accuracy counts every example once
and every mistake alike, and three common situations break that:

. . .

- **A rare class.** 1% of people have a disease: "always healthy" scores
  **99%** and finds nobody; a test that finds 90% of the sick scores
  **95%**.
- **Unequal costs.** A missed patient may be worth 100 false alarms.
- **Decisions on probabilities.** A threshold on $p(\textrm{sick})$ is only
  as good as $p(\textrm{sick})$.

::: {.d2l-note}
Each needs a measurement of its own: recall and precision, costs, and
calibration.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- read a confusion matrix: recall, precision, and the cost of mistakes;
- derive the decision threshold that costs imply;
- check calibration with a reliability diagram and fit a temperature.

**Running examples:** screening 100,000 people for a disease that 1% have;
the softmax regression of the previous decks on the 10,000 Fashion-MNIST
validation images.
:::

::: {.slide title="How to Measure the Classifiers of This Chapter"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Softmax Regression | probabilities over classes, cross-entropy |
| Image Classification Data | Fashion-MNIST in minibatches |
| Softmax Regression from Scratch | softmax regression by hand, the `Classifier` class |
| Concise Implementation | framework layers, a stable loss |
| **Measuring Classifiers** | **precision, recall, costs, calibration** |
| Generalization in Classification | error bars, test-set reuse, VC dimension |
| Environment and Distribution Shift | correcting for shifted data |
:::

::: {.col .narrow}
- **Uses:** the trained softmax regression; the confusion matrix; Bayes'
  theorem.
- **Skips:** ROC curves (exercises), several labels per example, ranking
  metrics.
:::
:::
:::

::: {.slide title="A Useless Screener Scores 99%, a Useful One 95%"}
[Class imbalance]{.kicker}

1% of 100,000 people are sick. A test finds 90% of them and flags 5% of the
healthy:

| | sick | healthy |
|:--|--:|--:|
| **predicted sick** | 900 (TP) | 4,950 (FP) |
| **predicted healthy** | 100 (FN) | 94,050 (TN) |

- Test: accuracy $94{,}950 / 100{,}000 = 0.95$.
- Always *healthy*: accuracy $0.99$, and it finds nobody.

::: {.d2l-note .warn}
Each person counts once: the 1,000 sick are 1% of the accuracy.
:::
:::

::: {.slide title="Recall and Precision Look at the Sick Separately"}
[Class imbalance]{.kicker}

$$\textrm{recall} = \frac{\textrm{TP}}{\textrm{TP} + \textrm{FN}} \;\;(\textrm{sick found}), \qquad \textrm{precision} = \frac{\textrm{TP}}{\textrm{TP} + \textrm{FP}} \;\;(\textrm{flags right})$$

@!classification-beyond-accuracy

- Precision is $P(\textrm{sick} \mid \textrm{flagged})$: Bayes' theorem
  again. The rare disease makes false positives outnumber true ones.
:::

::: {.slide title="Check: A Test With Fewer False Alarms"}
[Check]{.kicker}

The test now flags only **1%** of the healthy and still finds 90% of the
sick. Compute its accuracy and precision. Does it beat *always healthy* on
accuracy?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answers.** FP $= 990$, TN $= 98{,}010$. Accuracy
$(900 + 98{,}010)/100{,}000 = 0.989$: still below $0.99$. Precision
$900 / 1{,}890 \approx 0.48$.
:::
:::
:::

::: {.slide title="Costs Move the Decision Threshold From ½ to 1%"}
[Asymmetric costs]{.kicker}

A missed patient costs 100 false alarms: $\kappa_\textrm{FN} = 100$,
$\kappa_\textrm{FP} = 1$.

- Cost per person: always *healthy* $1.0$; the test $0.15$.
- With $p = P(\textrm{sick} \mid \mathbf{x})$, flag the patient if

$$p\, \kappa_\textrm{FN} > (1 - p)\, \kappa_\textrm{FP} \quad\Longleftrightarrow\quad p > \frac{\kappa_\textrm{FP}}{\kappa_\textrm{FP} + \kappa_\textrm{FN}} = \frac{1}{101}.$$

::: {.d2l-note .rule}
Equal costs give $\frac{1}{2}$, the argmax. A lower threshold raises recall
and lowers precision.
:::
:::

::: {.slide title="Confidence Is the Largest Predicted Probability"}
[Calibration]{.kicker}

Softmax regression of the previous deck, trained again; confidence: the
largest entry of $\mathrm{softmax}(\mathbf{o})$.

Ten bins by confidence; in each, $n_b$ predictions, mean confidence
$\bar{c}_b$, accuracy $a_b$:

$$\textrm{ECE} = \sum_b \frac{n_b}{n}\, \bigl|a_b - \bar{c}_b\bigr|$$

::: {.d2l-note}
The first bin, $[0, 0.1)$, stays empty, since 10 probabilities cannot all
be below $1/10$.
:::
:::

::: {.slide title="In Every Well-Filled Bin, Accuracy Tracks Confidence"}
[Calibration]{.kicker}

@!softmax-regression-concise-calibration-2

- The two lowest bins hold at most about 70 predictions: their accuracy is
  noise.
- From 0.3 up, hundreds of predictions per bin; accuracy near the
  confidence, in most bins slightly above it.
:::

::: {.slide title="Softmax Regression Is Nearly Calibrated: ECE Below 0.03"}
[Calibration]{.kicker}

::: {.cols .vc}
::: {.col .narrow}
- About half of the predictions have confidence above 0.9; 97% to 98% of
  them are right.
- Overall: mean confidence about 81%, accuracy about 83%, slightly
  underconfident.
- Bins with at least 100 predictions shown.
:::

::: {.col .fig .big}
@!softmax-regression-concise-calibration-3
:::
:::
:::

::: {.slide title="Check: Dividing the Logits by T = 2"}
[Check]{.kicker}

We divide all logits of the trained model by $T = 2$ before the softmax.
What happens to the accuracy, to the confidences, and to the expected
calibration error?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** The order of the logits is unchanged, so every prediction and
the accuracy stay the same. All confidences move toward $1/10$. The model
was already slightly underconfident, so the gap to the accuracy grows, and
so does the ECE.
:::
:::
:::

::: {.slide title="Temperature Scaling Changes Confidence, Not Decisions"}
[Calibration]{.kicker}

- **Temperature scaling:** softmax of $\mathbf{o} / T$, with one $T > 0$
  fitted by minimizing cross-entropy on held-out data; $T > 1$ lowers
  every confidence, the repair most deep networks need.
- Fitted on half of the validation images, checked on the other half:

@!classification-temperature-scaling

::: {.d2l-note .rule}
Dividing by $T$ keeps the order of the logits: the same predictions, the
same accuracy. Our model gets $T \approx 0.9$ and an ECE that barely moves:
little to repair.
:::
:::

::: {.slide title="Check: Calibrated but Useless?"}
[Check]{.kicker}

Ten balanced classes. A classifier ignores its input and outputs $0.1$ for
every class, breaking ties at random. Is it calibrated? What is its
accuracy?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Confidence $0.1$ on every example, and right on 10% of them:
perfectly calibrated, accuracy $0.1$. Calibration and accuracy measure
different things.
:::
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

| idea | running example |
|---|---|
| imbalance: recall and precision | always *healthy* $0.99$, test $0.95$; test recall $0.9$, precision $0.15$ |
| costs set the threshold | $\kappa_\textrm{FN} = 100$, $\kappa_\textrm{FP} = 1$: flag if $p > 1/101$ |
| calibration | mean confidence about 81%, accuracy about 83%, ECE below 0.03; $T \approx 0.9$ changes little |

**Next:** how far a number measured on 10,000 images can be trusted.
:::
