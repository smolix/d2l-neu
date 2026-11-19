```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Measuring Classifiers
:label:`sec_classification`

Our softmax regression classifier from :numref:`sec_softmax_scratch` and
:numref:`sec_softmax_concise` correctly labels about 83% of the
Fashion-MNIST validation images. Does higher accuracy automatically make
one classifier better than another? Only when every example carries equal
weight and every mistake inflicts the same harm—because accuracy counts
each example once and treats all errors identically. Yet in practice, three
everyday situations shatter those assumptions: one class may be far rarer
than the others, different kinds of mistakes may carry wildly asymmetric
costs, and downstream decisions may act directly on the predicted
*probabilities* rather than on the top predicted class. Each scenario
demands its own diagnostic. We unpack the first two using a medical
screening example whose outcomes we can tally by hand, and then put the
third to the test on our trained Fashion-MNIST classifier, whose
probabilities we have so far accepted on faith.

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

Imagine screening a population for a disease that affects 1% of people. A
trivial classifier that ignores every test result and always predicts
*healthy* achieves 99% accuracy—yet catches not a single sick patient. Now
compare it against a real screening test that detects 90% of sick patients
while falsely flagging 5% of healthy ones. Out of 100,000 people (1,000
sick and 99,000 healthy), the test catches 900 sick patients, misses 100,
and falsely flags 4,950 healthy people. Arranging these four counts in a
table—predicted classes in the rows, true classes in the columns—gives:

| | sick | healthy |
|:--|--:|--:|
| **predicted sick** | 900 | 4,950 |
| **predicted healthy** | 100 | 94,050 |

This $2 \times 2$ table is a binary *confusion matrix*. Taking *sick* as
the *positive* class, its four entries are the *true positives* (TP = 900),
*false positives* (FP = 4,950), *false negatives* (FN = 100), and *true
negatives* (TN = 94,050). Summing the diagonal gives 94,950 correct calls,
or an accuracy of 95%—so raw accuracy actually ranks the useless
"always-healthy" rule *ahead* of a genuinely lifesaving screening test!
Because every individual counts equally, the 1,000 sick patients contribute
at most 1% to overall accuracy regardless of how well or poorly we detect
them.

To zoom in on the rare positive class, we pair two complementary ratios.
*Recall* measures the fraction of sick patients the classifier catches,
while *precision* measures the fraction of flagged patients who are
actually sick:

$$\textrm{recall} = \frac{\textrm{TP}}{\textrm{TP} + \textrm{FN}}, \qquad \textrm{precision} = \frac{\textrm{TP}}{\textrm{TP} + \textrm{FP}}.$$

In our table, recall reads down the first column ($900 / 1{,}000 = 0.90$)
and precision reads across the first row ($900 / 5{,}850 \approx 0.15$).

Recall immediately separates the two rules: 0 for "always healthy" versus
0.90 for the screening test (whose trivial rival has undefined precision
since it never raises a flag). Precision, however, delivers a sobering
surprise: only 15% of people who test positive are actually sick! This is
the exact same Bayesian effect we encountered with HIV testing in
:numref:`subsec_probability_hiv_app`: whenever a condition is rare, false
alarms from the vast healthy majority easily outnumber true positives from
the tiny sick minority. Is a test with 90% recall and 15% precision still
valuable? Absolutely—in two-stage clinical screening, its job is to catch
nearly every sick patient cheaply so that flagged individuals can be sent
on to a slower, highly specific follow-up test.

To see just how heavily precision depends on disease prevalence, let us
hold the test's own sensitivity (90% recall) and false-alarm rate (5%)
fixed while sweeping the share of sick people in the population:

```{.python .input #classification-beyond-accuracy}
recall, false_alarms = 0.9, 0.05  # Properties of the test
for prevalence in [0.001, 0.01, 0.1, 0.5]:
    tp, fp = recall * prevalence, false_alarms * (1 - prevalence)
    accuracy = tp + (1 - false_alarms) * (1 - prevalence)
    print(f'{prevalence:5.1%} sick: accuracy {accuracy:.3f}, '
          f'recall {recall:.2f}, precision {tp / (tp + fp):.3f}')
```

The exact same diagnostic test swings from 2% precision at 0.1% prevalence
all the way to 95% precision at 50% prevalence, even while its accuracy
scarcely budges. Keep the distinction straight: recall and the false-alarm
rate are intrinsic properties of the test itself, whereas precision is a
property of the test *combined with the population* to which it is applied.

### Asymmetric Costs

In clinical screening, missing a sick patient is vastly worse than causing
a false alarm. Suppose a false negative costs 100 times as much as a false
positive: $\kappa_\textrm{FN} = 100$ and $\kappa_\textrm{FP} = 1$. Weighting
the confusion matrix by these penalties, the "always-healthy" rule incurs
an expected cost of $100 \cdot 1{,}000 / 100{,}000 = 1.0$ per person,
whereas our screening test costs only
$(100 \cdot 100 + 1 \cdot 4{,}950) / 100{,}000 \approx 0.15$ per person—nearly
seven times lower. Once costs reflect reality, the useful classifier wins
decisively.

Asymmetric costs also alter the optimal *decision rule*. Suppose our model
estimates a probability $p$ that a patient is sick. Calling the patient
*healthy* carries an expected cost of $p\,\kappa_\textrm{FN}$, whereas
flagging them *sick* carries an expected cost of $(1 - p)\,\kappa_\textrm{FP}$.
Flagging the patient minimizes expected cost precisely when

$$p \, \kappa_\textrm{FN} > (1 - p)\, \kappa_\textrm{FP}, \quad \textrm{that is,} \quad p > \frac{\kappa_\textrm{FP}}{\kappa_\textrm{FP} + \kappa_\textrm{FN}}.$$
:eqlabel:`eq_cost-threshold`

When both mistakes cost the same, the threshold is $\frac{1}{2}$—recovering
ordinary $\operatorname{argmax}$ classification. With
$\kappa_\textrm{FN} = 100$ and $\kappa_\textrm{FP} = 1$, however, the
threshold plunges to $1/101 \approx 0.01$: even a 2% chance of illness
warrants flagging the patient. Lowering the threshold trades precision for
higher recall as more patients get flagged. When exact error costs are hard
to pin down, practitioners commonly report precision and recall side by
side, or combine them into their harmonic mean—the *F1 score*—which stays
high only when both precision and recall are strong.

The exact same framework extends naturally to $q$ classes: a cost matrix
with entries $\kappa_{ij}$ (the penalty for predicting class $i$ when the
true class is $j$) weights the entries of the multiclass confusion matrix
$\mathbf{N}$ from :eqref:`eq_confusion-matrix`.

## Calibration

### Reliability Diagrams

Notice that the cost-sensitive decision rule :eqref:`eq_cost-threshold`
operates directly on the numerical probability $p$, which means its
decisions are only as trustworthy as those probabilities themselves. A
classifier is *calibrated* when its predicted probabilities match empirical
long-run frequencies: among all patients assigned $p = 0.02$, roughly 2%
should actually be sick. If only 0.2% are sick, our threshold at $0.01$
would mistakenly flag the entire group even though none of them truly
crosses the 1% risk bar. Checking calibration on binary validation data is
straightforward: bucket patients by predicted $p$ and compare each bucket's
average $p$ against the actual fraction of sick patients in that bucket.

With $q$ classes there are $q$ probabilities per example, so practitioners
usually start by checking the *confidence*—the highest predicted
probability (that of the predicted class). We partition validation examples
into confidence bins such as $[0.5, 0.7)$, $[0.7, 0.9)$, and $[0.9, 1]$,
compute each bin's mean confidence and empirical accuracy, and plot accuracy
against mean confidence in a *reliability diagram*. A perfectly calibrated
model tracks the diagonal; points sagging *below* the diagonal signal an
*overconfident* model—one whose predictions are more cocksure than its
actual accuracy warrants, a notorious tendency of deep neural networks
trained with cross-entropy. Keep in mind, though, that top-class
calibration says nothing about tiny probabilities: if your decision rule
thresholds $p(\textrm{sick})$ around $0.01$, you must check calibration
directly on $p(\textrm{sick})$ itself.

Let us run this diagnostic on our Fashion-MNIST softmax regression model
from :numref:`sec_softmax_concise`. We retrain it cleanly (with live
plotting silenced so we do not duplicate the previous section's chart) and
gather the validation logits and labels into NumPy arrays.

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

To compute each prediction's confidence, we apply the numerically shifted
softmax (:numref:`sec_softmax_scratch`) to the logits in NumPy and take the
maximum probability per row.

```{.python .input #classification-calibration-2}
def softmax(O):
    """Row-wise softmax of a NumPy array of logits."""
    E = np.exp(O - O.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)

conf = softmax(O).max(axis=1)  # Confidence of each prediction
correct = (O.argmax(axis=1) == y).astype(float)
```

Next, we bucket the predictions into ten uniform bins—$[0, 0.1)$,
$[0.1, 0.2)$, through $[0.9, 1]$—and tally in each bin $b$ its count $n_b$,
mean confidence $\bar{c}_b$, and empirical accuracy $a_b$ (the first bin is
necessarily empty because ten probabilities summing to 1 must have a
maximum of at least $0.1$). To summarize overall miscalibration in a single
number, we compute the *expected calibration error* (ECE), the
sample-weighted average absolute gap between accuracy and confidence across
bins:

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

Roughly half of all validation predictions land in the top bin with mean
confidence near 0.97—and sure enough, 97% to 98% of them are right. The two
lowest populated bins contain only a handful to about 70 images, too sparse
for reliable accuracy estimates. From confidence 0.3 upward, however, every
bin contains hundreds of examples, with empirical accuracy hugging the mean
confidence closely (usually sitting slightly above it). Let us plot the
reliability diagram for all bins with at least 100 predictions (keeping the
bin-wise standard error at or below 5 percentage points).

```{.python .input #softmax-regression-concise-calibration-3}
used = n >= 100  # Fewer predictions give too noisy an accuracy
d2l.plot([mean_conf[used], [0, 1]], [acc[used], [0, 1]], 'mean confidence',
         'accuracy', legend=['softmax regression', 'calibrated'],
         fmts=('o-', 'k:'), figsize=(3.5, 3.5))
```

The curve tracks the diagonal remarkably well, hovering just slightly above
it. Across all 10,000 validation images, mean confidence is around 81%
against ~83% accuracy, keeping ECE below 0.03 across runs. In other words,
this shallow linear model is actually slightly *under*confident—so on
in-distribution garments, its probabilities can be taken very nearly at
face value. Remember that calibration is a property of a *trained model*,
not an automatic guarantee of the softmax function; we can run this exact
same check on any deeper network we train later, something raw accuracy
alone could never reveal.

### Temperature Scaling

When a reliability diagram does uncover miscalibration, the standard fix is
surprisingly simple. *Temperature scaling* divides every logit by a single
positive scalar $T > 0$ (the temperature of :numref:`sec_softmax`) before
applying softmax, fitting $T$ to minimize cross-entropy on held-out
validation data :cite:`Guo.Pleiss.Sun.Weinberger.2017`. Because dividing by
a positive constant preserves the ordering of logits, temperature scaling
never alters a single predicted class or changes accuracy by even a hair—it
only recalibrates confidence, pushing overconfident networks down when
$T > 1$. Since our linear model is slightly underconfident, we expect the
fitted $T$ to land slightly below 1. Let us fit $T$ by grid search on the
first half of the validation set (using the numerically stable log-sum-exp
cross-entropy from :numref:`sec_softmax_concise`) and evaluate ECE on the
second half before and after scaling.

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

As expected, the fitted temperature sits just below 1 (around 0.9), and
scaling shifts the held-out ECE by less than a hundredth—sometimes nudging
it slightly up rather than down, since at this tiny level of miscalibration
binning noise rivals the remaining gap. On deep, overconfident networks,
however, these exact same lines routinely find $T > 1$ and pull a sagging
reliability curve sharply back toward the diagonal.

Every metric we just computed—accuracy, confusion counts, ECE—came from a
single sample of 10,000 validation images, and would wiggle slightly on a
fresh sample of 10,000. How large is that sampling uncertainty, and how
trustworthy do our scores remain once we start tuning hyperparameters? We
turn to those statistical questions next in
:numref:`chap_classification_generalization`.

## Summary

Classification accuracy treats every example and every mistake equally,
which can mislead badly in practice. Under severe class imbalance, a
trivial majority-class predictor can beat a useful detector on accuracy;
*recall* and *precision* expose the difference, though precision depends
heavily on class prevalence via Bayes' theorem. When false negatives and
false positives carry unequal costs, expected-cost minimization shifts the
decision threshold away from $\frac{1}{2}$—making *calibration* essential
so that predicted probabilities reflect true empirical frequencies.
Reliability diagrams and expected calibration error (ECE) diagnose
miscalibration, and post-hoc *temperature scaling* adjusts confidence via a
single scalar $T > 0$ without altering accuracy. On Fashion-MNIST, softmax
regression turns out to be well calibrated (slightly underconfident, ECE
$< 0.03$), leaving little for temperature scaling to fix.

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
