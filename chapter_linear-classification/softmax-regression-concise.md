```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Concise Implementation
:label:`sec_softmax_concise`

Just as high-level deep learning APIs simplified linear regression in
:numref:`sec_linear_concise`, they make softmax regression dramatically
cleaner and safer to implement. In :numref:`sec_softmax_scratch` we built
every piece by hand; here we delegate the parameters to a built-in fully
connected layer and the objective to the framework's cross-entropy loss.

If trimming boilerplate were the whole story, this section would take five
lines. Yet the framework's loss introduces one crucial design difference:
*it accepts raw logits rather than probabilities*. Recall from
:numref:`sec_softmax_scratch` that our naive softmax explodes to `NaN` on
large logits, while clamping tiny probabilities in the loss prevents
$-\infty$ only by killing the gradient. How can we compute cross-entropy so
that both the loss and its gradient stay finite for *any* logit vector? As
we are about to see, the secret is never to materialize the probabilities
at all.

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

As in :numref:`sec_linear_concise`, a single built-in fully connected layer
stores $\mathbf{W}$ and $\mathbf{b}$ and evaluates
$\mathbf{O} = \mathbf{X}\mathbf{W} + \mathbf{b}$ with 10 outputs—one per
garment class. Just as before, we flatten each four-dimensional image batch
into rows of 784 pixels before the linear layer, and `forward` returns the
unnormalized logits directly. To keep all four frameworks aligned with our
from-scratch baseline, we explicitly initialize weights from
$\mathcal{N}(0, 0.01^2)$ and biases to 0; otherwise, each framework would
apply its own default initializer and train slightly different models.

:begin_tab:`mxnet`
Gluon's `Dense` layer automatically flattens higher-order inputs while
preserving the leading batch dimension, converting $(n, 1, 28, 28)$ tensors
directly into $n \times 784$ matrices. It allocates its weights on the
first forward pass once the input width is known and populates them
according to `initialize`.
:end_tab:

:begin_tab:`pytorch`
We precede `nn.LazyLinear` with `nn.Flatten()` to collapse the channel,
height, and width axes into a single feature dimension. As in
:numref:`sec_linear_concise`, `LazyLinear` infers its input dimension on
the first forward pass, after which `Trainer` invokes `weight_init`.
:end_tab:

:begin_tab:`tensorflow`
A `Flatten` layer converts the four-dimensional batch into two dimensions
while preserving the batch axis. Keras materializes the `Dense` weights on
the first forward pass using `kernel_initializer`.
:end_tab:

:begin_tab:`jax`
Because Flax NNX layers allocate parameters upon construction, `nnx.Linear`
takes the flattened input size up front—784 by default for $28 \times 28$
images, or a custom `num_inputs` when resizing is enabled.
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

Our scratch implementation computes probabilities first and takes their
logarithm afterward—a two-step pipeline that looks harmless on paper yet
invites floating-point disaster at both extremes. Single-precision floats
top out around $3.4 \times 10^{38}$, so $\exp(o)$ overflows to $\infty$
whenever $o > 88$; that is precisely what turned `softmax([1000, 0, 0])`
into `NaN` in :numref:`sec_softmax_scratch`. Overflow is the easy bug to
patch: because softmax is shift-invariant, subtracting the row maximum
$o_{\max} = \max_k o_k$ ensures every exponent is $\leq 0$.

Unfortunately, shifting does nothing to rescue the lower end. Given logits
$(0, -200, 0)$, the second class has probability roughly $e^{-200}/2$—far
smaller than the smallest positive float32 value—so it *underflows* cleanly
to exact `0.0` (this happens to any logit more than about 104 below
$o_{\max}$). Taking the logarithm of `0.0` yields $-\infty$, and
backpropagating through it unleashes the dreaded screenful of `NaN`s. The
scratch loss's `1e-12` clamp merely swept that infinity under the rug at
the cost of zeroing out the gradient. Both hazards stem from the exact same
mistake: computing intermediate probabilities that the loss never actually
needs.

### The Log-Sum-Exp Form

What does cross-entropy genuinely need? Only the *logarithm* of one
probability—and by combining softmax and the logarithm algebraically *before*
evaluating any exponentials, the division turns into a simple subtraction:

$$
\log \hat{y}_j = o_j - \log \sum_k \exp (o_k).
$$

For an example with true class $y$, the loss $l = -\log \hat y_y$ becomes a
direct function of the logits, bypassing probabilities altogether. To keep
the sum itself from overflowing, we factor $\exp(o_{\max})$ out of the
logarithm (the classic [LogSumExp trick](https://en.wikipedia.org/wiki/LogSumExp)):

$$
l(y, \mathbf{o}) = \log \sum_{k} \exp(o_k) - o_y =
o_{\max} + \log \sum_{k} \exp(o_k - o_{\max}) - o_y.
$$
:eqlabel:`eq_softmax-concise-lse-loss`

Mathematically both sides are identical, yet numerically they are worlds
apart. On the right-hand side, every exponent $o_k - o_{\max}$ is at most
$0$ and at least one equals $0$, pinning the sum strictly inside $[1, q]$
and its logarithm inside $[0, \log q]$. No overflow, no underflow, no
$\log 0$: every term stays finite for arbitrary logits. On $(0, -200, 0)$
with true class 1 (the second class), the loss evaluates smoothly to
$\log 2 + 200 \approx 200.7$—appropriately steep for a confident blunder,
yet completely well-behaved. Notice that the leading term
$\log \sum_k \exp(o_k)$ is the exact *log-sum-exp* we differentiated in
:numref:`subsec_softmax_and_derivatives`.

### The Framework Loss

Every deep learning framework fuses softmax and cross-entropy via
:eqref:`eq_softmax-concise-lse-loss` whenever you pass it raw logits.
Because probabilities are never materialized in between, neither overflow
nor $\log 0$ can occur, and automatic differentiation yields the exact
gradient $\partial l / \partial o_j = \mathrm{softmax}(\mathbf{o})_j - y_j$
(:numref:`subsec_softmax_and_derivatives`) with no artificial clamp cutting
it off.

Let us install this fused loss once on `d2l.Classifier` so every classifier
in the book inherits it automatically. The method accepts logits and
integer class labels, flattens any extra leading axes (preparing for later
sequence models that classify every token position), and returns either the
batch mean or—when `averaged=False`—per-example losses.

:begin_tab:`pytorch`
PyTorch exposes the fused loss as `F.cross_entropy`.
:end_tab:

:begin_tab:`mxnet`
Gluon provides `SoftmaxCrossEntropyLoss`. Despite its counterintuitive
parameter name, its default `from_logits=False` expects raw logits and
computes log-softmax internally.
:end_tab:

:begin_tab:`tensorflow`
Keras provides `SparseCategoricalCrossentropy`; we must pass
`from_logits=True` so it expects raw logits rather than pre-normalized
probabilities.
:end_tab:

:begin_tab:`jax`
Optax provides the fused loss as `softmax_cross_entropy_with_integer_labels`.
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

A subtle bug to watch out for: because `loss` expects *raw logits*, adding
a softmax at the end of `forward` would apply softmax *twice*. Nothing
crashes and training even appears to run, yet the loss gets trapped above a
strictly positive floor and can never approach zero. That is why
`SoftmaxRegression.forward` returns logits directly (we keep a separate
softmax handy only when we explicitly want to display predicted
probabilities).

Let us stress-test our fused loss on the exact extreme logits
$(1000, 0, 0)$ that broke our scratch implementation, paired once with true
label 0 and once with true label 1. By :eqref:`eq_softmax-concise-lse-loss`,
the exact answers are $\log(1 + 2e^{-1000}) \approx 0$ and
$1000 + \log(1 + 2e^{-1000}) \approx 1000$.

```{.python .input #softmax-regression-concise-the-framework-loss}
O = d2l.tensor([[1000.0, 0.0, 0.0], [1000.0, 0.0, 0.0]])
y = d2l.tensor([0, 1])
SoftmaxRegression(num_outputs=3, lr=0.1).loss(O, y, averaged=False)
```

Both values come back completely finite and exact: zero loss for the
certain correct prediction, and 1000 for the confident mistake.

:begin_tab:`pytorch`
PyTorch prints the first entry as `-0.`, IEEE-754 negative zero, which is
numerically identical to 0.
:end_tab:

## Training

Now we train `SoftmaxRegression` using the exact same dataset, minibatch
size, learning rate, and epoch count as in :numref:`sec_softmax_scratch`.

```{.python .input #softmax-regression-concise-training}
data = d2l.FashionMNIST(batch_size=256)
model = SoftmaxRegression(num_outputs=10, lr=0.1)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

Ten epochs bring us right back to roughly 83% validation accuracy with
virtually identical loss curves—only with far fewer lines of code and rock-solid
numerical stability. That is exactly what we expect: switching to framework
primitives changes *how* the computation runs, not the mathematical model.
We now have a clean classifier, a loss that cannot blow up, and a headline
number of 83%. The rest of this chapter interrogates what that number is
truly worth, starting with the probabilities behind it: can we take them at
face value?

## Summary

High-level deep learning APIs let us define softmax regression in a handful
of lines—a flattening step followed by a single fully connected layer
returning logits—while quietly protecting us from numerical hazards like
overflow and underflow. That convenience is both a blessing and a mild
curse: it makes deep learning immediately accessible, yet hiding the sharp
edges can make bugs harder to diagnose when you eventually build custom
losses or components outside the framework's safety rails. That is why we
work through both from-scratch and concise implementations side by side.
Here, fusing softmax and cross-entropy via the shifted log-sum-exp identity
$l = o_{\max} + \log \sum_k \exp(o_k - o_{\max}) - o_y$ guarantees finite
losses and intact gradients $\mathrm{softmax}(\mathbf{o}) - \mathbf{y}$ for
arbitrary logits, and installing it on `Classifier.loss` equips every
classifier in the rest of the book with the same stability.

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
