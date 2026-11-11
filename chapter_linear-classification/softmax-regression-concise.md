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
it stays finite, and keeps its gradient, for every value of the logits? The
answer, it turns out, is never to form the probabilities at all.

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
As in :numref:`sec_linear_concise`, the lazy layer learns its input size at
the first forward pass, and the `Trainer` calls `weight_init` after it.
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
logarithm afterwards, and each step can fail in floating point. Single
precision stores numbers up to about $3.4 \times 10^{38}$, so $\exp(o)$
overflows to infinity once $o$ exceeds about 88, which is what turned the
softmax of $(1000, 0, 0)$ into `NaN` in :numref:`sec_softmax_scratch`. This
failure is the easy one. A common shift of all logits leaves the softmax
unchanged, so subtracting the largest logit, $o_{\max} = \max_k o_k$, from
each of them removes it, as we saw there. Unfortunately, the shift does
nothing for the lower end. For the logits $(0, -200, 0)$ the probability of
the second class is about $e^{-200}/2$, smaller than any positive
single-precision number, and so it is stored as 0; this happens to every
logit that lies more than about 104 below the largest one. The logarithm of
this 0 is $-\infty$, and backpropagation through it produces `NaN`. The
floor of the scratch loss hides the infinity and discards the gradient with
it. Both failures have the same cause: we formed a probability that the
loss never needed.

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
the factor $\exp(o_{\max})$ out of it:

$$
l(y, \mathbf{o}) = \log \sum_{k} \exp(o_k) - o_y =
o_{\max} + \log \sum_{k} \exp(o_k - o_{\max}) - o_y.
$$
:eqlabel:`eq_softmax-concise-lse-loss`

The two sides are equal, but only the right side should be handed to a
computer. All its exponents are at most 0, and one of them is equal to 0, so
the sum lies between 1 and $q$ and its logarithm between 0 and $\log q$.
Every term is an ordinary number, whatever the logits. For $(0, -200, 0)$
with the second class as the true one, the loss is
$\log 2 + 200 \approx 200.7$: large, as it should be for so wrong a
prediction, but finite. The first term, $\log \sum_k \exp(o_k)$, is the
*log-sum-exp* of the logits that :numref:`subsec_softmax_and_derivatives`
differentiated.

### The Framework Loss

The cross-entropy losses of the frameworks compute
:eqref:`eq_softmax-concise-lse-loss` when they are given logits. They never
form the probabilities, so neither the exponential of a large number nor the
logarithm of 0 occurs. Their gradient is the gradient of this expression,
$\partial l / \partial o_j = \mathrm{softmax}(\mathbf{o})_j - y_j$
(:numref:`subsec_softmax_and_derivatives`), with no floor to cut it off.

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

We put the loss to the test on the logits $(1000, 0, 0)$ that broke the
softmax of the previous section, once with label 0 and once with label 1.
By :eqref:`eq_softmax-concise-lse-loss` the losses are
$\log(1 + 2e^{-1000}) \approx 0$ and $1000 + \log(1 + 2e^{-1000}) \approx 1000$.

```{.python .input #softmax-regression-concise-the-framework-loss}
O = d2l.tensor([[1000.0, 0.0, 0.0], [1000.0, 0.0, 0.0]])
y = d2l.tensor([0, 1])
SoftmaxRegression(num_outputs=3, lr=0.1).loss(O, y, averaged=False)
```

Both losses are finite and exact: the first example is classified correctly
with certainty, and the second is wrong by a margin of 1000.

:begin_tab:`pytorch`
The first loss prints as `-0.`, a zero with its sign bit set, which equals 0.
:end_tab:

## Training

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
the model. We now have a classifier, a loss that cannot blow up, and a
number, 83%. The rest of the chapter asks what the number is worth,
beginning with the probabilities behind it: can they be taken at face
value?

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
reaches about 83% validation accuracy, like the version from scratch.

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

1. [code] **Fused loss on extreme logits.** The text passes the logits
   $\mathbf{o} = (1000, 0, 0)$ to the `loss` of this section with labels 0
   and 1. Copy the from-scratch `softmax` of :numref:`sec_softmax_scratch`,
   which is not part of the `d2l` package.
    1. Compute the same two losses by the from-scratch route, `softmax`
       followed by `d2l.cross_entropy`. Explain each value it returns, and
       determine the largest finite loss this route can return.
    1. Verify on benign logits such as $\mathbf{o} = (2, 1, 0)$ that the
       two routes agree to floating-point precision.

1. [code] **Stable softplus.** For two classes with logits $(x, 0)$, the
   first term of the loss is $\log(1 + e^x)$, the *softplus* of $x$.
    1. Plot $\log(1 + e^x)$ and $\max(x, 0)$ on $[-5, 5]$. Show that the
       first exceeds the second by at most $\log 2$, at $x = 0$, and
       explain in one sentence why log-sum-exp is called a smooth maximum.
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

1. [code] **The gradient survives.** In :numref:`sec_softmax_scratch`,
   exercise 1, the floored loss of the logits $(0, -200, 0)$ with label 1
   had a zero gradient. Pass the same logits and label to the `loss` of
   this section and compute the gradient with respect to the logits by
   automatic differentiation. Compare it with $\hat{\mathbf{y}} - \mathbf{y}$
   from :numref:`sec_softmax`, and explain why this gradient would move the
   logit of class 1 while the floored one did not.

1. **Number formats.** Half-precision formats store a number in 16 bits:
   FP16 has 5 exponent bits and 10 mantissa bits, BFLOAT16 has 8 and 7, and
   FP32 has 8 and 23. Determine for each format the largest logit $o$ for
   which $\exp(o)$ does not overflow. Then explain why a network whose
   logits reach 20 can be trained in BFLOAT16 but not in FP16 unless its
   softmax is computed in FP32. Finally, consider the shifted sum of
   :eqref:`eq_softmax-concise-lse-loss` for $q = 10^5$ classes, each term
   at most 1, accumulated in FP16: what goes wrong once the running sum
   exceeds 2,048, and what does the 10-bit mantissa have to do with it?

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
[Dive into Deep Learning · §3.4]{.kicker}

Concise Implementation<br>**a framework layer · the log-sum-exp loss**
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
- derive the log-sum-exp form of the loss and say why it cannot overflow
  or underflow;
- predict what the loss returns on the logits $(1000, 0, 0)$, and check it;
- name each framework's loss on logits, installed once on `Classifier`.

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
| Softmax Regression from Scratch | softmax regression by hand, the `Classifier` class |
| **Concise Implementation** | **framework layers, a stable loss** |
| Measuring Classifiers | precision, recall, costs, calibration |
| Generalization in Classification | error bars, test-set reuse, VC dimension |
| Environment and Distribution Shift | correcting for shifted data |
:::

::: {.col .narrow}
- **Uses:** the scratch model and its failure on large logits; the
  `Classifier`.
- **Skips:** calibration (next deck).
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

With $o_{\max} = \max_k o_k$, the factor $\exp(o_{\max})$ cancels:

$$\hat y_j = \frac{\exp(o_j - o_{\max})}{\sum_k \exp(o_k - o_{\max})}$$

- Every exponent is $\le 0$: numerators at most 1, denominator in
  $[1, q]$. No overflow.
- $o_j - o_{\max} < -104$ still gives $\hat{y}_j = 0$ and
  $\log \hat{y}_j = -\infty$.
:::

::: {.slide title="The Logarithm Removes the Division"}
[A stable loss]{.kicker}

$$\log \hat{y}_j = o_j - o_{\max} - \log \sum_k \exp(o_k - o_{\max})$$

Every term is finite. For true class $y$:

$$l(y, \mathbf{o}) = \log \sum_k \exp(o_k) - o_y = o_{\max} + \log \sum_k \exp(o_k - o_{\max}) - o_y$$

::: {.d2l-note .rule}
The loss is a function of the logits alone. Its gradient is
$\mathrm{softmax}(\mathbf{o}) - \mathbf{y}$, with no floor to cut it off.
:::
:::

::: {.slide title="The Shifted Sum Lies Between 1 and q"}
[A stable loss]{.kicker}

One exponent is $0$ and the other $q - 1$ are $\le 0$:

$$1 \le \sum_k \exp(o_k - o_{\max}) \le q, \qquad 0 \le \log \sum_k \exp(o_k - o_{\max}) \le \log q$$

Logits $(0, -200, 0)$, true class 2: the loss is $\log 2 + 200 \approx 200.7$.
Large, as it should be, and finite.
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

@!softmax-regression-concise-the-framework-loss
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

All four compute $o_{\max} + \log \sum_k \exp(o_k - o_{\max}) - o_y$ and
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

[Training]{.dtitle}

[the same model, the same accuracy]{.dsub}
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

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

- **Model:** flatten, one layer with 10 outputs; `forward` returns logits.
- **Failure:** $\exp$ overflows above 88; probabilities underflow below
  about $-104$, and $\log 0 = -\infty$.
- **Fix:** $l = o_{\max} + \log \sum_k \exp(o_k - o_{\max}) - o_y$, finite
  for any logits: $0$ and $1000$ on $(1000, 0, 0)$ with labels 0 and 1.
- **Code:** each framework's loss on logits, installed once on
  `Classifier`.
- **Result:** about 83% validation accuracy, as from scratch.

**Next:** what accuracy hides, and whether the predicted probabilities can
be trusted.
:::
