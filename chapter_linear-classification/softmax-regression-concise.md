```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Concise Implementation
:label:`sec_softmax_concise`

In :numref:`sec_softmax_scratch` we wrote every part of softmax regression by
hand. In practice nobody does. We use the building blocks of a deep learning
framework instead, as :numref:`sec_linear_concise` did for linear regression:
a fully connected layer that holds the weights and the bias, and a
cross-entropy loss. If saving lines of code were all there is to it, this
section could be very short. The framework's loss, however, differs from ours
in one respect that matters: it takes the logits rather than the
probabilities. Recall that the softmax we wrote by hand returns `NaN` once a
logit grows large, and that the floor under its probabilities kept the loss
finite only by cutting off the gradient. How can the loss be computed so that
it stays finite, and keeps its gradient, for every value of the logits?

The answer, it turns out, is never to form the probabilities at all. We
write the model with the framework's layer, derive the form of the loss that
makes this possible, and install the framework's loss on `Classifier`, so
that every classifier in the book inherits it. After training we ask a
question that accuracy cannot answer: can the predicted probabilities be
taken at face value? The reliability diagram of :numref:`sec_classification`
answers it.

```{.python .input #softmax-regression-concise-concise-implementation-of-softmax-regression}
%%tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #softmax-regression-concise-concise-implementation-of-softmax-regression}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #softmax-regression-concise-concise-implementation-of-softmax-regression}
%%tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #softmax-regression-concise-concise-implementation-of-softmax-regression}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## The Model

As in :numref:`sec_linear_concise`, a fully connected layer of the framework
holds the weights $\mathbf{W}$ and the bias $\mathbf{b}$ and computes
$\mathbf{O} = \mathbf{X}\mathbf{W} + \mathbf{b}$. It has 10 outputs, one per
class. The model flattens each image into a vector of 784 pixel values
before the layer, as the scratch implementation did, and its `forward`
returns the logits. Note that we set the initial values ourselves: the
weights start from a normal distribution with mean 0 and standard deviation
0.01 and the biases at 0, again as in the scratch implementation. Each
framework's layer would otherwise draw its weights from a default
distribution of its own, and the four tabs would train different models.

:begin_tab:`mxnet`
The `Dense` layer flattens its input itself, so the $(n, 1, 28, 28)$ images
become $n \times 784$ inputs. Gluon creates the weights at the first forward
pass, once the number of inputs is known, and draws them as `initialize`
prescribes.
:end_tab:

:begin_tab:`pytorch`
`nn.LazyLinear` creates its weights only at the first forward pass, once the
number of inputs is known, so the constructor cannot initialize them. The
`Trainer` calls the method `weight_init` after that pass instead.
:end_tab:

:begin_tab:`tensorflow`
Keras creates the weights of `Dense` at the first forward pass, once the
number of inputs is known, and draws them from the `kernel_initializer`.
:end_tab:

:begin_tab:`jax`
NNX layers create their parameters when they are constructed, so
`nnx.Linear` needs the number of inputs up front: $28 \times 28 = 784$ by
default, and `num_inputs` for images of another size.
:end_tab:

```{.python .input #softmax-regression-concise-defining-the-model}
%%tab pytorch
class SoftmaxRegression(d2l.Classifier):  #@save
    """The softmax regression model."""
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(),
                                 nn.LazyLinear(num_outputs))

    def weight_init(self, module):
        # The Trainer calls this after the lazy layer has learned its shape
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.01)
            nn.init.zeros_(module.bias)

    def forward(self, X):
        return self.net(X)
```

```{.python .input #softmax-regression-concise-defining-the-model}
%%tab mxnet
class SoftmaxRegression(d2l.Classifier):  #@save
    """The softmax regression model."""
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Dense(num_outputs)
        self.net.initialize(init.Normal(sigma=0.01))

    def forward(self, X):
        return self.net(X)
```

```{.python .input #softmax-regression-concise-defining-the-model}
%%tab tensorflow
class SoftmaxRegression(d2l.Classifier):  #@save
    """The softmax regression model."""
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = tf.keras.models.Sequential()
        self.net.add(tf.keras.layers.Flatten())
        self.net.add(tf.keras.layers.Dense(
            num_outputs,
            kernel_initializer=tf.initializers.RandomNormal(stddev=0.01)))

    def forward(self, X):
        return self.net(X)
```

```{.python .input #softmax-regression-concise-defining-the-model}
%%tab jax
class SoftmaxRegression(d2l.Classifier):  #@save
    """The softmax regression model."""
    def __init__(self, num_outputs, lr, num_inputs=784, rngs=None):
        super().__init__()
        self.save_hyperparameters(ignore=['rngs'])
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        self.net = nnx.Linear(num_inputs, num_outputs,
                              kernel_init=nnx.initializers.normal(0.01),
                              rngs=rngs)

    def forward(self, X):
        X = X.reshape((X.shape[0], -1))  # Flatten
        return self.net(X)
```

## A Stable Cross-Entropy Loss
:label:`subsec_softmax-implementation-revisited`

### Overflow and Underflow

The scratch implementation computes the probabilities first and takes the
logarithm afterwards. Both steps can fail in floating point. At the upper
end, the exponential of a logit above about 88 overflows in single precision,
which is what turned the softmax of $(1000, 0, 0)$ into `NaN` in
:numref:`sec_softmax_scratch`. This failure is the easy one. A common shift
of all logits leaves the softmax unchanged
(:numref:`subsec_softmax_operation`), so we may subtract the largest logit,
$\bar{o} = \max_k o_k$, from each of them, after which no exponent is
positive and nothing overflows.

Unfortunately, the shift does nothing for the lower end. Take the logits
$(0, -200, 0)$. The probability of the second class is about $e^{-200}/2$,
far smaller than any positive single-precision number, and so it is stored
as 0. (This happens to every logit that lies more than about 104 below the
largest one.) The logarithm of this 0 is $-\infty$, and backpropagation
through it produces `NaN`.

### The Log-Sum-Exp Form

What does the loss need? Only the logarithm of one probability, and the
logarithm of a quotient is a difference. Taking the logarithm of the softmax
on paper, before anything is computed, removes the division:

$$
\log \hat{y}_j = o_j - \log \sum_k \exp (o_k).
$$

This is the key step. For an example of true class $y$, the loss
$l = -\log \hat y_y$ becomes a function of the logits alone, with no
probability in between. As it stands the sum can still overflow, so we pull
the factor $\exp(\bar{o})$ out of it:

$$
l(y, \mathbf{o}) = \log \sum_{k} \exp(o_k) - o_y =
\bar{o} + \log \sum_{k} \exp(o_k - \bar{o}) - o_y.
$$
:eqlabel:`eq_softmax-concise-lse-loss`

The two sides are equal, but only the right side should be handed to a
computer. All its exponents are at most 0, and one of them is equal to 0, so
the sum lies between 1 and $q$, and its logarithm between 0 and $\log q$.
Every term is an ordinary number, whatever the logits. For $(0, -200, 0)$
with the second class as the true one, the loss is
$\log 2 + 200 \approx 200.7$: large, as it should be for so wrong a
prediction, but finite.

The first term, $\log \sum_k \exp(o_k)$, is the *log-sum-exp* of the logits
(:numref:`subsec_softmax_and_derivatives`). We have just seen that it
exceeds the largest logit by at most $\log q$, which is why it is called a
smooth version of the maximum. For two classes with logits $(x, 0)$ it is
$\log(1 + e^x)$, which we compare with $\max(x, 0)$:

```{.python .input #softmax-regression-concise-lse-vs-max}
x = d2l.arange(-4.0, 4.0, 0.01)
lse, mx = d2l.log(1 + d2l.exp(x)), (x + d2l.abs(x)) / 2
d2l.plot(x, [lse, mx, lse - mx], 'x', legend=['lse(x, 0)', 'max(x, 0)', 'gap'])
```

The gap is largest at the tie $x = 0$, where it equals
$\log 2 \approx 0.69$, the bound $\log q$ for $q = 2$, and it vanishes as
$|x|$ grows.

### The Framework Loss

The cross-entropy losses of the frameworks compute
:eqref:`eq_softmax-concise-lse-loss` when they are given logits. They never
form the probabilities, so neither the exponential of a large number nor the
logarithm of 0 occurs. Their gradient is the gradient of this expression,
$\partial l / \partial o_j = \mathrm{softmax}(\mathbf{o})_j - y_j$
(:numref:`subsec_softmax_and_derivatives`), with no floor to cut it off. We
do not put this claim to the test here. The exercises do, on the logits
$(1000, 0, 0)$ that broke the softmax of the previous section.

We install the framework's loss as the method `loss` of `Classifier`, so that
every classification model in the book inherits it. The method takes
logits and integer labels. It first flattens all leading axes of both, since
later models predict one class for each position of a sequence, and returns
the mean loss, or with `averaged=False` the loss of every example.

:begin_tab:`pytorch`
In PyTorch the loss is `F.cross_entropy`.
:end_tab:

:begin_tab:`mxnet`
In Gluon the loss is `SoftmaxCrossEntropyLoss`. With its default setting
`from_logits=False` it computes the logarithms of the probabilities from the
logits itself.
:end_tab:

:begin_tab:`tensorflow`
In Keras the loss is `SparseCategoricalCrossentropy`. It assumes
probabilities unless we pass `from_logits=True`.
:end_tab:

:begin_tab:`jax`
In Optax the loss is `softmax_cross_entropy_with_integer_labels`.
:end_tab:

```{.python .input #softmax-regression-concise-softmax-revisited}
%%tab pytorch
@d2l.add_to_class(d2l.Classifier)  #@save
def loss(self, Y_hat, Y, averaged=True):
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    return F.cross_entropy(
        Y_hat, Y, reduction='mean' if averaged else 'none')
```

```{.python .input #softmax-regression-concise-softmax-revisited}
%%tab mxnet
@d2l.add_to_class(d2l.Classifier)  #@save
def loss(self, Y_hat, Y, averaged=True):
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    fn = gluon.loss.SoftmaxCrossEntropyLoss()
    l = fn(Y_hat, Y)
    return l.mean() if averaged else l
```

```{.python .input #softmax-regression-concise-softmax-revisited}
%%tab tensorflow
@d2l.add_to_class(d2l.Classifier)  #@save
def loss(self, Y_hat, Y, averaged=True):
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    reduction = (tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
                 if averaged else tf.keras.losses.Reduction.NONE)
    fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=reduction)
    return fn(Y, Y_hat)
```

```{.python .input #softmax-regression-concise-softmax-revisited}
%%tab jax
@d2l.add_to_class(d2l.Classifier)  #@save
def loss(self, Y_hat, Y, averaged=True):
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    fn = optax.softmax_cross_entropy_with_integer_labels
    return fn(Y_hat, Y).mean() if averaged else fn(Y_hat, Y)
```

The loss takes logits, not probabilities. A model whose `forward` ended
with a softmax would hand it probabilities, and the loss would apply a second
softmax to them. No error message appears, and the model still trains, but
its loss can no longer come near 0. This is why the `forward` of
`SoftmaxRegression` returns the logits.

## Training and Calibration

### The Training Run

We train with the same data, minibatch size, learning rate, and number of
epochs as in :numref:`sec_softmax_scratch`.

```{.python .input #softmax-regression-concise-training}
data = d2l.FashionMNIST(batch_size=256)
model = SoftmaxRegression(num_outputs=10, lr=0.1)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

After ten epochs the validation accuracy is about 83%, as for the
implementation from scratch, and the loss curves look alike. This is as it
should be. The framework's components change how the model is computed, not
the model.

### Calibration

The accuracy uses only the predicted class, the one with the largest logit.
The model says more than that. It assigns a probability to every class, and
the largest of these, the confidence of the prediction, says how sure the
model is. Should we believe it? We may if the classifier is calibrated
(:numref:`sec_classification`), that is, if of all predictions made with
confidence 0.8, 80% are correct. To check, we collect the confidence and the
correctness of every validation prediction. Conveniently, the confidence
needs no softmax. If we label an example with its own predicted class, the
loss :eqref:`eq_softmax-concise-lse-loss` is
$\log \sum_k \exp(o_k) - \max_k o_k = -\log \max_j \hat{y}_j$, so the
exponential of minus this loss is the confidence.

:begin_tab:`pytorch`
`torch.inference_mode` tells PyTorch not to record the operations for
differentiation, since we compute no gradients here.
:end_tab:

```{.python .input #softmax-regression-concise-calibration-1}
%%tab pytorch
conf, correct = [], []
with torch.inference_mode():
    for X, y in data.val_dataloader():
        O = model(X)
        preds = d2l.astype(d2l.argmax(O, axis=1), y.dtype)
        conf.append(d2l.exp(-model.loss(O, preds, averaged=False)))
        correct.append(model.accuracy(O, y, averaged=False))
conf = d2l.numpy(d2l.concat(conf, 0))
correct = d2l.numpy(d2l.concat(correct, 0))
```

```{.python .input #softmax-regression-concise-calibration-1}
%%tab mxnet, tensorflow, jax
conf, correct = [], []
for X, y in data.val_dataloader():
    O = model(X)
    preds = d2l.astype(d2l.argmax(O, axis=1), y.dtype)
    conf.append(d2l.exp(-model.loss(O, preds, averaged=False)))
    correct.append(model.accuracy(O, y, averaged=False))
conf = d2l.numpy(d2l.concat(conf, 0))
correct = d2l.numpy(d2l.concat(correct, 0))
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
The model is slightly *under*confident. This is not the defect we were
warned about: deep networks trained on the same loss are often
overconfident, and temperature scaling (:numref:`sec_classification`) lowers
their confidences. Lowering ours would only widen the gap. As they are, the
probabilities of softmax regression can be taken nearly at face value on
images like its validation images. This is a property of the trained model,
not of the softmax, and the same check applies unchanged to every classifier
that follows.

One reservation remains. The accuracy of about 83% and the calibration were
measured on the same 10,000 test images. How far such numbers can be trusted
for new images, and what reusing the test set does to them, is the subject
of the next section.

## Summary

Softmax regression with framework components is a fully connected layer on
flattened images whose forward pass returns logits, trained with the
framework's cross-entropy loss. The loss takes logits because computing it
from probabilities fails in floating point, by overflow of the exponential
or by the logarithm of a probability that has underflowed to 0. Written as
$l = \log \sum_k \exp(o_k) - o_y$ and evaluated after subtracting the
largest logit, the loss stays finite for any logits and keeps its gradient,
$\mathrm{softmax}(\mathbf{o}) - \mathbf{y}$. `Classifier.loss` installs this
loss for every classifier in the book. Trained for ten epochs, the model
reaches about 83% validation accuracy, like the version from scratch, and
its confidences match its accuracy to within about three points.

## Exercises

1. [code] **Double softmax.** The `loss` of this section expects logits.
   Suppose that a model's `forward` mistakenly ends with a softmax. The
   loss then receives a probability vector $\mathbf{p}$ in place of the
   logits and evaluates $-\log \mathrm{softmax}(\mathbf{p})_y$.
    1. Find the smallest value this loss can take over all probability
       vectors $\mathbf{p}$ and labels $y$, as a function of the number of
       classes $q$. Evaluate it for $q = 10$.
    1. Copy the from-scratch `softmax` of :numref:`sec_softmax_scratch`,
       which is not part of the `d2l` package. Subclass
       `SoftmaxRegression` so that `forward` applies this `softmax` to the
       logits. Train it with `lr=0.1`, `d2l.FashionMNIST(batch_size=256)`,
       and `d2l.Trainer(max_epochs=10)`. Compare its training loss and
       validation accuracy with those of `SoftmaxRegression`, and relate
       the final training loss to the value from the previous step.

1. [code] **Fused loss on extreme logits.** :numref:`sec_softmax_scratch`
   showed that the from-scratch `softmax` returns `NaN` on the logits
   $\mathbf{o} = (1000, 0, 0)$. That `softmax` is not part of the `d2l`
   package, so copy it, and store $\mathbf{o}$ as a $1 \times 3$ matrix.
    1. Pass $\mathbf{o}$ to the `loss` of this section, first with label
       $y = 0$ and then with $y = 1$. Check both values against
       :eqref:`eq_softmax-concise-lse-loss` evaluated by hand.
    1. Compute the same two losses by the from-scratch route, `softmax`
       followed by `d2l.cross_entropy`. Explain each value it returns, and
       determine the largest finite loss this route can return.
    1. Verify on benign logits such as $\mathbf{o} = (2, 1, 0)$ that the
       two routes agree to floating-point precision.

1. [code] **Stable softplus.** For two classes with logits $(x, 0)$, the
   first term of the loss is $\log(1 + e^x)$, the *softplus* of $x$.
    1. Show that $\log(1 + e^x) = \max(x, 0) + \log(1 + e^{-|x|})$.
    1. In single precision, evaluate the naive expression
       `log(1 + exp(x))`, the rewritten expression, and the rewritten
       expression with `log1p` in place of `log(1 + ...)`, for
       $x \in \{-100, -20, -5, 0, 5, 20, 100\}$. Compare each value with a
       double-precision reference, and report where each version
       overflows and where it loses all relative accuracy.
    1. Compute the derivative of the naive expression at $x = -20$ by
       automatic differentiation. Explain why it is accurate although the
       value of the expression is not, and what this implies for training
       and for reporting the loss of confidently classified examples.

1. **Number formats.** ● Deep learning uses several floating-point formats.
   Each consists of one sign bit, an exponent field and a mantissa field.
   The exponent and mantissa fields have 11 and 52 bits in FP64, 8 and 23
   bits in FP32, 8 and 10 bits in TF32, 8 and 7 bits in BFLOAT16, and 5
   and 10 bits in FP16. Assume that every format represents subnormal
   numbers as FP32 does. For each format, determine the smallest and
   largest argument of the exponential function for which the result
   neither underflows to $0$ nor overflows. Present the result as a table
   with one row per format, and explain how each limit depends on the
   numbers of exponent and mantissa bits.

1. [code] **Temperature scaling.** As :numref:`sec_softmax` explains,
   temperature scaling divides all logits by one number $T > 0$ before the
   softmax and fits $T$ on held-out data.
    1. Split the validation images into two halves. On the first half,
       find the $T$ that minimizes the cross-entropy of
       $\mathrm{softmax}(\mathbf{o} / T)$, by a grid search over
       $T \in [0.5, 2]$. Is $T$ above or below 1, and what does that say
       about the trained model?
    1. Compute the expected calibration error on the second half before
       and after scaling, and draw both reliability diagrams.
    1. Show that the accuracy does not change, and explain why the fit
       uses data that the model was not trained on.

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/52)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/53)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/260)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/17983)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §3.5]{.kicker}

Concise Implementation<br>**a framework layer · the log-sum-exp loss · calibration**
:::
:::

::: {.slide title="The Scratch Loss Fails on Large Logits and Stops the Gradient"}
[The problem]{.kicker}

Softmax first, then $-\log \hat{y}_y$:

- logits $(1000, 0, 0)$: $\exp(1000) = \infty$ in float32, and the softmax
  is $(\texttt{NaN}, 0, 0)$;
- a probability that underflows to 0 gives $-\log 0 = \infty$; the floor
  $10^{-12}$ caps the loss at 27.6 but zeroes the gradient.

. . .

The loss needs only one logarithm, and it needs no probabilities:

$$\log \hat{y}_y = o_y - \log \sum_k \exp(o_k)$$

::: {.d2l-note}
Framework losses take the logits and evaluate this expression stably.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- write softmax regression with a framework layer in a few lines;
- derive the log-sum-exp form of the loss and say why it cannot overflow;
- bound log-sum-exp between $\max_k o_k$ and $\max_k o_k + \log q$;
- name each framework's loss on logits, installed once on `Classifier`;
- check calibration with a reliability diagram and the expected
  calibration error.

**Running example:** Fashion-MNIST in minibatches of 256, ten epochs at
learning rate 0.1, about 83% validation accuracy.
:::

::: {.slide title="The Same Model, With the Framework's Layer and Loss"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Softmax Regression | probabilities over classes, cross-entropy |
| Image Classification Data | Fashion-MNIST in minibatches |
| Measuring Classifiers | accuracy, costs, calibration, the `Classifier` class |
| Softmax Regression from Scratch | softmax regression by hand |
| **Concise Implementation** | **framework layers, a stable loss** |
| Generalization in Classification | error bars, test-set reuse, VC dimension |
| Environment and Distribution Shift | correcting for shifted data |
:::

::: {.col .narrow}
- **Uses:** the scratch model and its failure on large logits; the
  `Classifier`; the reliability diagram.
- **Skips:** temperature scaling of the trained model (an exercise).
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[The model]{.dtitle}

[one fully connected layer that returns logits]{.dsub}
:::
:::

::: {.slide title="One Fully Connected Layer Is the Whole Model"}
[The model]{.kicker}

Flatten each image to 784 values, then one layer with 10 outputs holds
$\mathbf{W}$ and $\mathbf{b}$. Weights from $\mathcal{N}(0, 0.01^2)$, biases
0, in every framework:

| framework | layers | initialization |
|---|---|---|
| PyTorch | `Flatten`, `LazyLinear(10)` | `weight_init`, after the first pass |
| TensorFlow | `Flatten`, `Dense(10)` | `kernel_initializer=RandomNormal(stddev=0.01)` |
| MXNet | `Dense(10)`, flattens itself | `initialize(init.Normal(sigma=0.01))` |
| JAX | `nnx.Linear(784, 10)` | `kernel_init=nnx.initializers.normal(0.01)` |

`forward` returns `self.net(X)`; loss and accuracy come from `Classifier`.
:::

::: {.slide title="The Forward Pass Returns Logits, Not Probabilities"}
[The model]{.kicker}

`forward` returns $\mathbf{o} = \mathbf{W}^\top\mathbf{x} + \mathbf{b}$,
with no softmax.

. . .

::: {.d2l-note .rule}
The loss applies the softmax itself. A softmax in `forward` as well would
be applied twice, and the loss could never fall below a positive floor.
:::

- Predictions need no softmax either: the largest logit is the largest
  probability.
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[A stable loss]{.dtitle}

[overflow · underflow · log-sum-exp]{.dsub}
:::
:::

::: {.slide title="float32 exp Overflows Above 88 and Reaches 0 Below −104"}
[A stable loss]{.kicker}

Single-precision numbers span roughly $10^{-38}$ to $10^{38}$:

| argument $o$ | $\exp(o)$ in float32 |
|---|---|
| above about 88 | $\infty$: overflow |
| about $-87$ to 88 | full precision |
| about $-104$ to $-87$ | subnormal, losing digits |
| below about $-104$ | exactly 0: underflow |

::: {.d2l-note .warn}
One large logit overflows the softmax; one very negative logit gives a
probability of 0, and its logarithm is $-\infty$.
:::
:::

::: {.slide title="Shifting by the Maximum Removes Overflow, Not Underflow"}
[A stable loss]{.kicker}

With $\bar{o} = \max_k o_k$, the factor $\exp(\bar{o})$ cancels:

$$\hat y_j = \frac{\exp(o_j - \bar{o})}{\sum_k \exp(o_k - \bar{o})}$$

- Every exponent is $\le 0$: numerators at most 1, denominator in
  $[1, q]$. No overflow.
- $o_j - \bar{o} < -104$ still gives $\hat{y}_j = 0$ and
  $\log \hat{y}_j = -\infty$.
:::

::: {.slide title="The Logarithm Removes the Division"}
[A stable loss]{.kicker}

$$\log \hat{y}_j = o_j - \bar{o} - \log \sum_k \exp(o_k - \bar{o})$$

Every term is finite. For true class $y$:

$$l(y, \mathbf{o}) = \log \sum_k \exp(o_k) - o_y = \bar{o} + \log \sum_k \exp(o_k - \bar{o}) - o_y$$

::: {.d2l-note .rule}
The loss is a function of the logits alone. Its gradient is
$\mathrm{softmax}(\mathbf{o}) - \mathbf{y}$, with no floor to cut it off.
:::
:::

::: {.slide title="Log-Sum-Exp Lies Between max and max + log q"}
[A stable loss]{.kicker}

::: {.cols .vc}
::: {.col .narrow}
The sum holds $\exp(\bar{o})$ and $q$ terms no larger:

$$\bar{o} \le \log \sum_k \exp(o_k) \le \bar{o} + \log q$$

Two classes, logits $(x, 0)$: the gap peaks at the tie, $\log 2 \approx 0.69$.
:::

::: {.col .fig .big}
@!softmax-regression-concise-lse-vs-max
:::
:::
:::

::: {.slide title="Check: The Loss on Logits (1000, 0, 0)"}
[Check]{.kicker}

Compute the loss for the logits $(1000, 0, 0)$, first for the label of
the first class, then for the label of the second. What does the
scratch route (softmax, then the floored logarithm) return?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $\log \sum_k \exp(o_k) = 1000 + \log(1 + 2e^{-1000}) \approx 1000$,
so the losses are $\approx 0$ and $\approx 1000$. The scratch route
computes the softmax $(\texttt{NaN}, 0, 0)$: `NaN` for the first label,
and the floor's $27.6$ for the second.
:::
:::
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[The loss in code]{.dtitle}

[one fused call per framework]{.dsub}
:::
:::

::: {.slide title="Four Frameworks, One Expression"}
[The loss in code]{.kicker}

| framework | loss on logits and integer labels |
|---|---|
| PyTorch | `F.cross_entropy` |
| TensorFlow | `SparseCategoricalCrossentropy(from_logits=True)` |
| MXNet | `gluon.loss.SoftmaxCrossEntropyLoss()` |
| JAX | `optax.softmax_cross_entropy_with_integer_labels` |

All four compute $\bar{o} + \log \sum_k \exp(o_k - \bar{o}) - o_y$ and
never form a probability.
:::

::: {.slide title="Installed Once on Classifier, Inherited by Every Classifier"}
[The loss in code]{.kicker}

@softmax-regression-concise-softmax-revisited

- Leading axes are flattened: later models predict a class per position.
- `averaged=False` returns the loss of every example.
:::

::: {.slide}
::: {.divider}
[04]{.dnum}

[Training and calibration]{.dtitle}

[accuracy · confidence · the reliability diagram]{.dsub}
:::
:::

::: {.slide title="Ten Epochs Reach About 83%, as From Scratch"}
[Training]{.kicker}

::: {.cols .vc}
::: {.col .narrow}
Same data, minibatch size, learning rate, and epochs as the scratch model.

- About 83% validation accuracy.
- The framework's components change how the model is computed, not the
  model.
:::

::: {.col .fig .big}
@!softmax-regression-concise-training
:::
:::
:::

::: {.slide title="Confidence Is exp(−Loss) at the Predicted Class"}
[Calibration]{.kicker}

Confidence: the largest probability. Labeling an example with its own
prediction $\hat{y}$ gives
$l(\hat{y}, \mathbf{o}) = \log\sum_k \exp(o_k) - \max_k o_k = -\log \max_j \hat{y}_j$.

Ten bins by confidence; in each, $n_b$ predictions, mean confidence
$\bar{c}_b$, accuracy $a_b$:

$$\textrm{ECE} = \sum_b \frac{n_b}{n}\, \bigl|a_b - \bar{c}_b\bigr|$$

::: {.d2l-note}
No softmax needed: the fused loss already holds $-\log$ of the confidence.
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

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

- **Model:** flatten, one layer with 10 outputs; `forward` returns logits.
- **Failure:** $\exp$ overflows above 88; probabilities underflow below
  about $-104$, and $\log 0 = -\infty$.
- **Fix:** $l = \bar{o} + \log \sum_k \exp(o_k - \bar{o}) - o_y$, finite for
  any logits; $\bar{o} \le \textrm{lse} \le \bar{o} + \log q$.
- **Code:** each framework's loss on logits, installed once on
  `Classifier`.
- **Result:** about 83% validation accuracy, as from scratch; ECE 0.01 to
  0.03, slightly underconfident.

**Next:** how far an accuracy measured on 10,000 test images can be
trusted.
:::
