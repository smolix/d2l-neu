```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Measuring Classifiers
:label:`sec_classification`

Our Fashion-MNIST softmax regression model classifies roughly $83\%$ of validation images correctly. Does higher accuracy always mean a better classifier? Only when every class is equally important and every type of error carries the exact same cost. In real deployments, three everyday conditions break that symmetry:

1. **Class imbalance:** When the positive class of interest is rare (say, $1\%$ of patients have a disease or $0.1\%$ of transactions are fraudulent), a useless classifier that always predicts the majority class scores $99\%$ or $99.9\%$ accuracy while catching zero positives.
2. **Asymmetric error costs:** Missing a life-threatening illness (a false negative) is vastly costlier than ordering a follow-up test (a false positive).
3. **Probability-driven decisions:** Downstream decision rules act directly on the numerical predicted probabilities $\hat{\mathbf{y}}$, which are trustworthy only if the classifier is *calibrated*.

In this section, we develop the metrics and diagnostics for all three: *precision* and *recall*, *cost-sensitive decision thresholds*, *reliability diagrams* and *expected calibration error* (ECE), and post-hoc *temperature scaling*.

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

Suppose we screen $100{,}000$ people for a disease with $1\%$ prevalence ($1{,}000$ sick and $99{,}000$ healthy). A trivial rule that ignores every test result and always predicts *healthy* achieves $99\%$ accuracy while detecting not a single sick patient. Now compare it against a real screening test that detects $90\%$ of sick patients while falsely flagging $5\%$ of healthy ones:

| | sick (positive) | healthy (negative) |
|:--|--:|--:|
| **predicted sick** | $900$ (TP) | $4{,}950$ (FP) |
| **predicted healthy** | $100$ (FN) | $94{,}050$ (TN) |

In this $2 \times 2$ binary confusion matrix, the four cells are *true positives* ($\textrm{TP} = 900$), *false positives* ($\textrm{FP} = 4{,}950$), *false negatives* ($\textrm{FN} = 100$), and *true negatives* ($\textrm{TN} = 94{,}050$). Summing the diagonal gives $94{,}950 / 100{,}000 = 95\%$ accuracy—so raw accuracy ranks the useless "always-healthy" rule ($99\%$) ahead of a lifesaving test ($95\%$)!

To evaluate performance on the positive class specifically, we pair two ratios: *recall* (or *sensitivity* / *true-positive rate*), the fraction of true positives caught, and *precision* (or *positive predictive value*), the fraction of positive flags that are genuinely positive:

$$\textrm{recall} = \frac{\textrm{TP}}{\textrm{TP} + \textrm{FN}}, \qquad \textrm{precision} = \frac{\textrm{TP}}{\textrm{TP} + \textrm{FP}}.$$

Reading down the first column gives $\textrm{recall} = 900 / 1{,}000 = 0.90$ (versus $0$ for "always healthy"). Reading across the first row gives $\textrm{precision} = 900 / 5{,}850 \approx 0.15$: only $15\%$ of flagged people are actually sick, because $5\%$ false alarms from the huge healthy majority ($4{,}950$) outnumber the $900$ true positives (the exact Bayes' theorem effect from :numref:`subsec_probability_hiv_app`). In two-stage clinical screening, $90\%$ recall with $15\%$ precision is ideal: the cheap first test catches almost every sick patient and sends only the $5{,}850$ flagged patients to a definitive follow-up test.

Let us hold our test's intrinsic recall ($0.90$) and false-alarm rate ($0.05$) fixed while sweeping the disease prevalence across $\{0.1\%, 1\%, 10\%, 50\%\}$:

```{.python .input #classification-beyond-accuracy}
recall, false_alarms = 0.9, 0.05  # Properties of the test
for prevalence in [0.001, 0.01, 0.1, 0.5]:
    tp, fp = recall * prevalence, false_alarms * (1 - prevalence)
    accuracy = tp + (1 - false_alarms) * (1 - prevalence)
    print(f'{prevalence:5.1%} sick: accuracy {accuracy:.3f}, '
          f'recall {recall:.2f}, precision {tp / (tp + fp):.3f}')
```

Notice the lesson: recall and false-alarm rate are properties of the *classifier alone*, whereas precision depends heavily on the *population prevalence*.

### Asymmetric Costs

Suppose missing a sick patient costs $\kappa_{\textrm{FN}} = 100$ whereas a false alarm costs $\kappa_{\textrm{FP}} = 1$ (and correct calls cost $0$). Weighted by these penalties, the "always-healthy" rule incurs an expected cost of $100 \times 1{,}000 / 100{,}000 = 1.00$ per person, whereas our screening test costs $(100 \times 100 + 1 \times 4{,}950) / 100{,}000 \approx 0.15$ per person—nearly seven times lower!

More generally, if a classifier estimates probability $p = P(\textrm{sick} \mid \mathbf{x})$ for an individual, predicting *healthy* has expected cost $p\,\kappa_{\textrm{FN}}$ and predicting *sick* has expected cost $(1 - p)\,\kappa_{\textrm{FP}}$. Predicting *sick* minimizes expected cost whenever

$$p \, \kappa_\textrm{FN} > (1 - p)\, \kappa_\textrm{FP}, \quad \textrm{that is,} \quad p > \frac{\kappa_\textrm{FP}}{\kappa_\textrm{FP} + \kappa_\textrm{FN}}.$$
:eqlabel:`eq_cost-threshold`

Equal costs ($\kappa_{\textrm{FN}} = \kappa_{\textrm{FP}}$) recover the familiar $1/2$ threshold (argmax classification). With $\kappa_{\textrm{FN}} = 100$ and $\kappa_{\textrm{FP}} = 1$, however, the optimal threshold drops to $1/101 \approx 0.01$: even a $2\%$ predicted risk warrants flagging the patient! Lowering the threshold raises recall at the expense of precision (when costs are unspecified, practitioners often summarize the trade-off via the harmonic mean of precision and recall, the *F1 score* $= 2\,\textrm{precision}\cdot\textrm{recall}/(\textrm{precision} + \textrm{recall})$, or sweep all thresholds via ROC and precision--recall curves; see Exercises :exref:`ex_classification-roc-curves` and :exref:`ex_classification-auc-ranking-probability`).

## Calibration

### Reliability Diagrams

Cost-sensitive decision rules like :eqref:`eq_cost-threshold` act directly on the numerical value of $p$, which means they rely on the classifier being *calibrated*: among all examples where the classifier predicts confidence $\hat{p} \approx 0.80$, roughly $80\%$ should actually be correct.

In a $q$-class problem, we check top-class calibration by recording each validation prediction's *confidence* $\hat{p}^{(i)} = \max_j \hat{y}^{(i)}_j$ and correctness indicator $c^{(i)} = \mathbf{1}(\operatorname{argmax}_j \hat{y}^{(i)}_j = y^{(i)})$, partitioning the examples into confidence bins $[0, 0.1), [0.1, 0.2), \ldots, [0.9, 1.0]$, and comparing each bin's mean confidence $\bar{c}_b$ against its empirical accuracy $a_b$. Plotting $a_b$ against $\bar{c}_b$ gives a *reliability diagram* (where a perfectly calibrated model lies along the diagonal $a_b = \bar{c}_b$), and their sample-weighted average gap defines the *expected calibration error* (ECE):

$$\textrm{ECE} = \sum_b \frac{n_b}{n} \left| a_b - \bar{c}_b \right|.$$

Let us train our concise `SoftmaxRegression` model on Fashion-MNIST, collect its validation logits `O` and labels `y`, and compute its per-bin reliability statistics, ECE, and reliability diagram:

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

Next, `reliability(conf, correct, num_bins=10)` groups the validation predictions into ten equal-width confidence bins via `np.bincount`, returning the per-bin sample counts `n`, mean confidences `mean_conf`, empirical accuracies `acc`, and sample-weighted `ece`:

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

Filtering to well-populated bins with at least $100$ predictions (`used = n >= 100`) so noisy small-sample ratios do not distort the curve, we plot `acc[used]` against `mean_conf[used]` alongside the ideal $y = x$ diagonal:

```{.python .input #softmax-regression-concise-calibration-3}
used = n >= 100  # Fewer predictions give too noisy an accuracy
d2l.plot([mean_conf[used], [0, 1]], [acc[used], [0, 1]], 'mean confidence',
         'accuracy', legend=['softmax regression', 'calibrated'],
         fmts=('o-', 'k:'), figsize=(3.5, 3.5))
```

Across all well-populated bins ($n_b \geq 100$), empirical accuracy tracks mean confidence within two or three percentage points (sitting slightly *above* the diagonal: average confidence $\sim 81\%$ vs. accuracy $\sim 83\%$, for an ECE below $0.03$). Unlike deep overparameterized networks—which frequently become severely overconfident—shallow softmax regression on Fashion-MNIST is already well calibrated (slightly underconfident).

### Temperature Scaling

When a reliability diagram reveals miscalibration, *temperature scaling* :cite:`Guo.Pleiss.Sun.Weinberger.2017` offers a one-parameter post-hoc fix: hold the trained weights fixed, divide the logits by a scalar temperature $T > 0$ (:numref:`sec_softmax`), and fit $T$ to minimize cross-entropy on a held-out validation split. Because dividing by $T > 0$ preserves logit ordering, temperature scaling never alters a single class prediction or changes accuracy—it only rescales confidence ($T > 1$ damps overconfidence; $T < 1$ sharpens underconfidence). Let us fit $T$ on the first half of our Fashion-MNIST validation set and check the ECE on the second half:

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

Consistent with our slightly underconfident reliability curve, the fitted temperature sits just below $1$ ($T \approx 0.9$), leaving the already tiny ECE essentially unchanged.

## Summary

- **Beyond accuracy under class imbalance:** When the positive class is rare, overall accuracy is dominated by true negatives; *recall* ($\textrm{TP}/(\textrm{TP}+\textrm{FN})$) and *precision* ($\textrm{TP}/(\textrm{TP}+\textrm{FP})$) isolate positive-class detection, with precision scaling via Bayes' theorem with class prevalence.
- **Cost-sensitive thresholding:** When false negatives and false positives carry costs $\kappa_{\textrm{FN}}$ and $\kappa_{\textrm{FP}}$, expected cost is minimized by predicting positive whenever $p > \kappa_{\textrm{FP}} / (\kappa_{\textrm{FP}} + \kappa_{\textrm{FN}})$.
- **Calibration and temperature scaling:** Reliability diagrams and $\textrm{ECE} = \sum_b (n_b/n)|a_b - \bar{c}_b|$ measure whether predicted confidences match empirical accuracy; fitting a single post-hoc temperature $T > 0$ on held-out data recalibrates confidence without changing accuracy.

## Exercises

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
[Dive into Deep Learning · §3.4]{.kicker}

Measuring Classifiers: Precision, Recall, Costs & Calibration<br>**class imbalance · recall & precision · Bayes prevalence effect · cost-sensitive thresholds · reliability diagrams · ECE · temperature scaling**
:::
:::

::: {.slide title="When Higher Accuracy Means a Worse Classifier"}
[The Limits of Raw Accuracy]{.kicker}

Our Fashion-MNIST classifier achieved $\sim 83\%$ accuracy because all $10$ garment classes are balanced and every mistake carries equal weight. In real deployments, three everyday conditions break that symmetry:

- **Severe class imbalance:** When only $1\%$ of patients carry a disease (or $0.1\%$ of credit-card transactions are fraudulent), a trivial rule that always predicts the majority class scores $99\%$ or $99.9\%$ accuracy while detecting **zero** positive cases.
- **Asymmetric error costs:** Missing a life-threatening illness (a false negative) is vastly costlier than ordering a confirmatory lab test (a false positive).
- **Probability-driven decisions:** Downstream decision rules act directly on the numerical value of $\hat{p} = P(y \mid \mathbf{x})$, which is valid only if predicted confidences match empirical accuracy (*calibration*).
:::

::: {.slide title="The Binary Confusion Matrix Under Class Imbalance"}
[1% Disease Prevalence Benchmark]{.kicker}

Suppose we screen $100{,}000$ people for a disease with $1\%$ prevalence ($1{,}000$ sick, $99{,}000$ healthy) using a test that detects $90\%$ of sick patients while falsely flagging $5\%$ of healthy people:

| Prediction \ True Status | Sick (Positive, $1{,}000$) | Healthy (Negative, $99{,}000$) | Row Total |
|:--|--:|--:|--:|
| **Predicted Sick** | $\mathrm{TP} = 900$ | $\mathrm{FP} = 4{,}950$ | $5{,}850$ flagged |
| **Predicted Healthy** | $\mathrm{FN} = 100$ | $\mathrm{TN} = 94{,}050$ | $94{,}150$ cleared |

- **The accuracy paradox:** Summing the diagonal gives $(900 + 94{,}050) / 100{,}000 = 94.95\%$ accuracy—so raw accuracy ranks the useless *"always-healthy"* rule ($99.0\%$) ahead of a test that catches $90\%$ of sick patients!

::: {.d2l-note .warn}
Because $99\%$ of the population is healthy, true negatives ($\mathrm{TN}$) swamp the accuracy denominator while the $1{,}000$ sick patients contribute at most $1\%$ to overall accuracy.
:::
:::

::: {.slide title="Isolating Positive Detection: Recall and Precision"}
[Column vs. Row Ratios]{.kicker}

To evaluate performance on the positive class without being swamped by true negatives ($\mathrm{TN}$), we pair two complementary ratios from the binary confusion matrix:

$$\mathrm{recall} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FN}}, \qquad \mathrm{precision} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FP}}$$

- **Recall (sensitivity / true-positive rate):** Reads down the *Sick* column—the fraction of truly sick patients caught ($900 / 1{,}000 = 0.90$, versus $0.00$ for *"always healthy"*).
- **Precision (positive predictive value):** Reads across the *Predicted Sick* row—the fraction of positive flags that are genuinely sick ($900 / 5{,}850 \approx 0.154$).
- **Why $15\%$ precision is still clinically valuable:** In two-stage screening, the cheap first test catches $90\%$ of cases and sends only the $5{,}850$ flagged patients ($5.85\%$ of the population) to a definitive follow-up test.
:::

::: {.slide title="How Precision Scales with Population Prevalence"}
[Bayes' Theorem in Classifier Evaluation]{.kicker}

Holding our test's intrinsic recall ($0.90$) and false-alarm rate ($\mathrm{FPR} = \mathrm{FP}/(\mathrm{FP}+\mathrm{TN}) = 0.05$) fixed while sweeping the disease prevalence $\pi = P(\text{sick})$ shows what depends on the population:

@classification-beyond-accuracy

::: {.d2l-note .rule}
**Intrinsic vs. prevalence-dependent metrics:** Recall $P(\text{flagged} \mid \text{sick})$ and false-alarm rate $P(\text{flagged} \mid \text{healthy})$ are properties of the *classifier alone*, whereas precision $P(\text{sick} \mid \text{flagged}) = \frac{\text{recall}\cdot\pi}{\text{recall}\cdot\pi + \text{FPR}\cdot(1-\pi)}$ depends strongly on population prevalence $\pi$ via Bayes' theorem.
:::
:::

::: {.slide title="Asymmetric Error Costs and Expected Risk"}
[Cost-Sensitive Evaluation]{.kicker}

When false negatives and false positives carry unequal penalties $\kappa_{\mathrm{FN}}$ and $\kappa_{\mathrm{FP}}$ (with correct classifications costing $0$), we evaluate classifiers by their **expected cost per person**:

$$\text{Expected Cost} = \frac{\kappa_{\mathrm{FN}}\cdot\mathrm{FN} + \kappa_{\mathrm{FP}}\cdot\mathrm{FP}}{n}$$

- **Numerical comparison ($\kappa_{\mathrm{FN}} = 100,\; \kappa_{\mathrm{FP}} = 1$):**
  - *"Always healthy"* ($\mathrm{FN} = 1{,}000,\; \mathrm{FP} = 0$): costs $(100 \times 1{,}000) / 100{,}000 = 1.00$ per person.
  - *Screening test* ($\mathrm{FN} = 100,\; \mathrm{FP} = 4{,}950$): costs $(100 \times 100 + 1 \times 4{,}950) / 100{,}000 = 0.1495$ per person—nearly **$7\times$ lower cost** despite $4$ points lower accuracy!
:::

::: {.slide title="Deriving the Bayes-Optimal Decision Threshold"}
[From Probabilities to Optimal Actions]{.kicker}

Given a calibrated probability estimate $p = P(\text{sick} \mid \mathbf{x})$ for an individual, what decision minimizes expected cost?

- **Expected cost of each action:** Predicting *healthy* incurs expected cost $p\,\kappa_{\mathrm{FN}}$, while predicting *sick* incurs expected cost $(1 - p)\,\kappa_{\mathrm{FP}}$.
- **Optimal threshold rule:** Flag the patient as *sick* whenever predicting *sick* is cheaper than predicting *healthy*:
  $$p\,\kappa_{\mathrm{FN}} > (1 - p)\,\kappa_{\mathrm{FP}} \quad \Longleftrightarrow \quad p > \tau^* = \frac{\kappa_{\mathrm{FP}}}{\kappa_{\mathrm{FP}} + \kappa_{\mathrm{FN}}}$$
- **Concrete impact:** Equal costs ($\kappa_{\mathrm{FN}} = \kappa_{\mathrm{FP}}$) recover the standard $\tau^* = 1/2$ ($\operatorname{argmax}$) rule. When $\kappa_{\mathrm{FN}} = 100$ and $\kappa_{\mathrm{FP}} = 1$, the threshold drops to $\tau^* = 1/101 \approx 0.01$: even a $2\%$ predicted risk warrants flagging the patient!
:::

::: {.slide title="Threshold Trade-Offs: Precision, Recall, F1 & ROC"}
[Sweeping the Decision Threshold]{.kicker}

Varying the decision threshold $\tau \in [0, 1]$ trades recall directly against precision and false alarms:

- **Lowering $\tau$ ($\tau \downarrow$):** Flags more examples, which monotonically **increases recall** ($\mathrm{TP} \uparrow$) while **decreasing precision** ($\mathrm{FP} \uparrow$).
- **F1 score (when explicit costs are unavailable):** Summarizes precision and recall at a chosen threshold via their harmonic mean, $\mathrm{F1} = \frac{2\,\mathrm{precision}\cdot\mathrm{recall}}{\mathrm{precision} + \mathrm{recall}}$, which is small whenever either precision or recall is small.
- **Threshold-free curves (Exercises 4–5):**
  - **ROC curve:** Plots recall ($\mathrm{TPR}$) against false-positive rate ($\mathrm{FPR}$) as $\tau$ sweeps from $1$ to $0$; its area (**AUC**) equals the probability that a random positive ranks above a random negative and is invariant to class prevalence.
  - **Precision–Recall curve:** Plots precision against recall; far more informative than ROC under severe class imbalance ($1\%$ positives).
:::

::: {.slide title="What Does It Mean for a Classifier to Be Calibrated?"}
[Trustworthy Probability Estimates]{.kicker}

Cost-sensitive decision rules ($p > \kappa_{\mathrm{FP}}/(\kappa_{\mathrm{FP}}+\kappa_{\mathrm{FN}})$) take the model's output probability literally. For a $q$-class classifier, define each prediction's **confidence** and **correctness**:

$$\hat{p}^{(i)} = \max_{j \in \{1,\ldots,q\}} \hat{y}^{(i)}_j, \qquad c^{(i)} = \mathbf{1}\!\Big(\operatorname*{argmax}_j \hat{y}^{(i)}_j = y^{(i)}\Big)$$

- **Top-class calibration condition:** Among all examples where the classifier reports confidence $\hat{p} \approx \alpha$, the empirical fraction of correct predictions should equal $\alpha$:
  $$P\big(c = 1 \;\big|\; \hat{p} = \alpha\big) = \alpha \quad \text{for all } \alpha \in [1/q, 1]$$
- **Accuracy vs. calibration are orthogonal:** A trivial classifier on $10$ balanced classes that always predicts uniform probabilities $(0.1, \ldots, 0.1)$ has $10\%$ accuracy at $0.10$ confidence—**perfectly calibrated, yet useless**! Conversely, a $95\%$-accurate model that always reports confidence $1.00$ is sharp but miscalibrated.
:::

::: {.slide title="Reliability Bins and Expected Calibration Error (ECE)"}
[Quantifying Miscalibration]{.kicker}

To measure calibration on $n$ validation examples, we partition $[0, 1]$ into $B = 10$ equal-width confidence bins $I_b = [\frac{b-1}{B}, \frac{b}{B})$ and compare each bin's mean confidence $\bar{c}_b$ against its empirical accuracy $a_b$:

$$\mathrm{ECE} = \sum_{b=1}^B \frac{n_b}{n}\,\big|a_b - \bar{c}_b\big|$$

@-classification-calibration-1

@classification-calibration-2

- **Empty low-confidence bins:** Because the maximum of $q = 10$ probabilities summing to $1$ is always at least $1/10 = 0.10$ (by the pigeonhole principle), bin $[0, 0.1)$ is mathematically guaranteed to have $n_1 = 0$.
:::

::: {.slide title="Auditing Calibration of Softmax Regression"}
[Per-Bin Reliability Table on Fashion-MNIST]{.kicker}

Computing `reliability(conf, correct)` across the $10{,}000$ Fashion-MNIST validation predictions reveals how closely empirical accuracy tracks predicted confidence:

@softmax-regression-concise-calibration-2

- **Interpreting the bin counts:** Across all well-populated bins ($[0.4, 0.5)$ through $[0.9, 1.0]$, holding $99\%$ of predictions), empirical accuracy matches mean confidence within $1\text{--}4$ percentage points, yielding an overall $\mathrm{ECE} < 0.03$.
:::

::: {.slide title="Visualizing the Reliability Diagram"}
[Overconfidence vs. Underconfidence]{.kicker}

Plotting bin accuracy $a_b$ against mean confidence $\bar{c}_b$ for all bins with $n_b \ge 100$ produces the **reliability diagram**:

::: {.cols .vc}
::: {.col}
- **Diagonal reference ($a_b = \bar{c}_b$):** A perfectly calibrated classifier lies directly along the dotted diagonal line.
- **Above vs. below the diagonal:** Points *above* the diagonal indicate an **underconfident** model ($a_b > \bar{c}_b$); points *below* indicate an **overconfident** model ($a_b < \bar{c}_b$).
- **Shallow vs. deep networks:** Our linear Fashion-MNIST model sits slightly above the diagonal (mean confidence $\approx 81\%$ vs. accuracy $\approx 83\%$), whereas deep modern neural networks typically sag well below the diagonal ($T > 1$ needed).
:::

::: {.col .fig}
@!softmax-regression-concise-calibration-3
:::
:::
:::

::: {.slide title="Post-Hoc Recalibration via Temperature Scaling"}
[One-Parameter Convex Calibration Fix]{.kicker}

When a classifier is miscalibrated, **temperature scaling** (Guo et al., 2017) freezes the trained weights and fits a single scalar temperature $T > 0$ on a held-out validation split to minimize the negative log-likelihood of $\mathrm{softmax}(\mathbf{o} / T)$:

@classification-temperature-scaling

::: {.d2l-note .rule}
**Why fit $T$ on held-out data?** Fitting $T$ on the training set would reflect the model's overfitted training margins rather than its generalization margins. Because $T > 0$ preserves logit order, temperature scaling recalibrates confidence while leaving every class decision and overall accuracy **100% unchanged**.
:::
:::

::: {.slide title="Concept Check: Screening Precision, Reject Option & ECE"}
[Concept Check]{.kicker}

Work out these three diagnostic and decision-making problems:

1. **Cleaner screening test:** In our $1\%$-prevalence disease example ($1{,}000$ sick, $99{,}000$ healthy), suppose an upgraded test keeps $90\%$ recall ($\mathrm{TP} = 900$) while cutting false alarms on the healthy from $5\%$ down to $1\%$. What are its new accuracy and precision, and does its accuracy beat *"always healthy"* ($99.0\%$)?
2. **Abstention / specialist referral (Exercise 1):** Suppose a correct diagnosis costs $0$, a misdiagnosis costs $\lambda_s = 100$, and referring an uncertain case to a human specialist costs $\lambda_r = 5$. Above what top-class confidence $\hat{p}$ should the automated classifier make a diagnosis instead of abstaining?
3. **Temperature shift:** If we divide all logits of our slightly underconfident Fashion-MNIST model by $T = 2$, what happens to its validation accuracy and to its ECE?

::: {.d2l-note .fragment .fade-up}
**Answers:** **(1)** $\mathrm{FP} = 0.01 \times 99{,}000 = 990$ and $\mathrm{TN} = 98{,}010$, giving accuracy $(900 + 98{,}010)/100{,}000 = 98.91\%$ (still below $99.0\%$!) while precision jumps from $15.4\%$ to $900 / (900 + 990) \approx 47.6\%$. **(2)** Expected error cost of guessing is $(1 - \hat{p})\lambda_s$; guessing beats abstaining when $(1 - \hat{p})\lambda_s < \lambda_r$ $\iff$ $\hat{p} > 1 - \lambda_r/\lambda_s = 1 - 5/100 = 0.95$. **(3)** Accuracy is unchanged (order of logits is preserved), while $T = 2$ pushes confidences even lower toward $0.10$, worsening underconfidence and increasing ECE.
:::
:::

::: {.slide title="Summary: Metrics, Decisions, and Calibration"}
[Section Summary]{.kicker}

| Diagnostic / Tool | Mathematical Definition | What It Solves |
|:--|:--|:--|
| **Recall (Sensitivity / TPR)** | $\mathrm{TP} / (\mathrm{TP} + \mathrm{FN}) = P(\hat{y}=1 \mid y=1)$ | Fraction of true positives caught; invariant to class prevalence |
| **Precision (PPV)** | $\mathrm{TP} / (\mathrm{TP} + \mathrm{FP}) = P(y=1 \mid \hat{y}=1)$ | Fraction of positive flags that are true; scales with prevalence via Bayes' rule |
| **Bayes cost threshold** | Predict positive iff $p > \frac{\kappa_{\mathrm{FP}}}{\kappa_{\mathrm{FP}} + \kappa_{\mathrm{FN}}}$ | Minimizes expected asymmetric cost ($\kappa_{\mathrm{FN}}=100, \kappa_{\mathrm{FP}}=1$ $\implies$ $\tau^* = 1/101$) |
| **Reliability diagram & ECE** | $\mathrm{ECE} = \sum_{b=1}^B \frac{n_b}{n}\big|a_b - \bar{c}_b\big|$ | Audits whether predicted confidence $\bar{c}_b$ matches empirical bin accuracy $a_b$ |
| **Temperature scaling** | $\hat{\mathbf{y}}(T) = \mathrm{softmax}(\mathbf{o}/T),\; T^* = \operatorname{argmin}_T \mathrm{NLL}_{\mathrm{val}}(T)$ | 1D convex post-hoc recalibration on held-out data; never alters accuracy |
:::
