```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Concise Implementation of Softmax Regression
:label:`sec_softmax_concise`



Just as high-level deep learning frameworks
made it easier to implement linear regression
(see :numref:`sec_linear_concise`),
they are similarly convenient here.

```{.python .input #softmax-regression-concise-concise-implementation-of-softmax-regression}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, init, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input #softmax-regression-concise-concise-implementation-of-softmax-regression}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input #softmax-regression-concise-concise-implementation-of-softmax-regression}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input #softmax-regression-concise-concise-implementation-of-softmax-regression}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
from functools import partial
import jax
from jax import numpy as jnp
import optax
```

## Defining the Model

As in :numref:`sec_linear_concise`, 
we construct our fully connected layer 
using the built-in layer. 
The built-in `__call__` method then invokes `forward` 
whenever we need to apply the network to some input.

:begin_tab:`mxnet`
Even though the input `X` is a fourth-order tensor, 
the built-in `Dense` layer 
will automatically convert `X` into a second-order tensor 
by keeping the dimensionality along the first axis unchanged.
:end_tab:

:begin_tab:`pytorch`
We use a `Flatten` layer to convert the fourth-order tensor `X` to second order 
by keeping the dimensionality along the first axis unchanged.

:end_tab:

:begin_tab:`tensorflow`
We use a `Flatten` layer to convert the fourth-order tensor `X` 
by keeping the dimension along the first axis unchanged.
:end_tab:

:begin_tab:`jax`
Flax allows users to write the network class in a more compact way
using `@nn.compact` decorator. With `@nn.compact`, one
can simply write all network logic inside a single “forward pass”
method, without needing to define the standard `setup` method in
the dataclass.
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
        self.net.initialize()
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
        self.net.add(tf.keras.layers.Dense(num_outputs))

    def forward(self, X):
        return self.net(X)
```

```{.python .input #softmax-regression-concise-defining-the-model}
%%tab jax
class SoftmaxRegression(d2l.Classifier):  #@save
    num_outputs: int
    lr: float

    @nn.compact
    def __call__(self, X):
        X = X.reshape((X.shape[0], -1))  # Flatten
        X = nn.Dense(self.num_outputs)(X)
        return X
```

## Softmax Revisited
:label:`subsec_softmax-implementation-revisited`

In :numref:`sec_softmax_scratch` we computed the softmax explicitly
and then took its logarithm inside the cross-entropy loss. To keep that
version usable we had to *clamp* the probabilities away from zero, a
band-aid that prevents $\log 0$ but still forms the overflow-prone
softmax first and silently kills the gradient on any clamped entry.
Here we remove the problem at its source rather than patch its symptom.

Recall that the softmax function computes probabilities via
$\hat y_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}$.
If some of the $o_k$ are very large, i.e., very positive,
then $\exp(o_k)$ might be larger than the largest number
we can have for certain data types. This is called *overflow*.
Conversely, a strongly negative $o_k$ makes
$\exp(o_k)$ *underflow* to $0$. Single-precision floats span roughly
$10^{-38}$ to $10^{38}$, so $\exp$ overflows once an argument exceeds about
$88$ and underflows to $0$ once it drops below about $-88$. A single large
positive logit therefore overflows the numerator, while strongly negative
logits underflow individual terms to $0$: harmless in the sum, but fatal
once we take a logarithm.
A way round this problem is to subtract $\bar{o} \stackrel{\textrm{def}}{=} \max_k o_k$ from
all entries:

$$
\hat y_j = \frac{\exp o_j}{\sum_k \exp o_k} =
\frac{\exp(o_j - \bar{o}) \exp \bar{o}}{\sum_k \exp (o_k - \bar{o}) \exp \bar{o}} =
\frac{\exp(o_j - \bar{o})}{\sum_k \exp (o_k - \bar{o})}.
$$

By construction we know that $o_j - \bar{o} \leq 0$ for all $j$. As such, for a $q$-class
classification problem, the denominator is contained in the interval $[1, q]$. Moreover, the
numerator never exceeds $1$, thus preventing numerical overflow. Numerical underflow only
occurs when $\exp(o_j - \bar{o})$ numerically evaluates as $0$. Nonetheless, a few steps down
the road we might find ourselves in trouble when we want to compute $\log \hat{y}_j$ as $\log 0$.
In particular, in backpropagation,
we might find ourselves faced with a screenful
of the dreaded `NaN` (Not a Number) results.

Fortunately, we are saved by the fact that
even though we are computing exponential functions,
we ultimately intend to take their log
(when calculating the cross-entropy loss).
By combining softmax and cross-entropy,
we can escape the numerical stability issues altogether. We have:

$$
\log \hat{y}_j =
\log \frac{\exp(o_j - \bar{o})}{\sum_k \exp (o_k - \bar{o})} =
o_j - \bar{o} - \log \sum_k \exp (o_k - \bar{o}).
$$

This avoids both overflow and underflow. We are not quite done, though,
because the object we actually need is not $\log \hat y_j$ but the loss.
For an example with true class $y$, the cross-entropy loss is
$\ell = -\log \hat y_y$. Substituting the stabilized expression above turns
the loss into a function of the **logits alone**:

$$
\ell(y, \mathbf{o}) = \log \sum_{k} \exp(o_k) - o_y =
\underbrace{\bar{o} + \log \sum_{k} \exp(o_k - \bar{o})}_{\textrm{numerically stable}} - o_y,
\qquad \bar{o} = \max_k o_k.
$$

The first term, $\log \sum_k \exp(o_k)$, is the *log-sum-exp* function, a
smooth upper bound on $\max_k o_k$; the second equality is the only safe way
to evaluate it, since every exponent $o_k - \bar{o} \leq 0$. This is precisely
what the built-in cross-entropy loss computes when handed raw logits: it never
forms the softmax probabilities at all, so neither $\exp$ of a large number nor
$\log$ of a zero ever occurs. Because the fused loss differentiates this exact
expression, its gradient is the clean
$\partial_{o_j}\ell = \mathrm{softmax}(\mathbf{o})_j - y_j$ derived in
:numref:`subsec_softmax_and_derivatives`, with no clamp to perturb it. We keep
the explicit softmax of :numref:`sec_softmax_scratch` only for *reading off*
predicted probabilities at inference time; for the loss we pass logits and let
the fused operation do the rest.

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
@partial(jax.jit, static_argnums=(0, 5))
def loss(self, params, X, Y, state, averaged=True):
    # To be used later (e.g., for batch norm)
    Y_hat = state.apply_fn({'params': params}, *X,
                           mutable=False, rngs=None)
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    fn = optax.softmax_cross_entropy_with_integer_labels
    # The returned empty dictionary is a placeholder for auxiliary data,
    # which will be used later (e.g., for batch norm)
    return (fn(Y_hat, Y).mean(), {}) if averaged else (fn(Y_hat, Y), {})
```

Each framework exposes this fused loss under a slightly different name, but
all four take **logits**, not probabilities: passing softmax outputs would
apply the softmax twice. PyTorch's `F.cross_entropy` consumes logits by
definition; TensorFlow uses `SparseCategoricalCrossentropy(from_logits=True)`;
JAX/Optax provides `softmax_cross_entropy_with_integer_labels`; and MXNet's
`SoftmaxCrossEntropyLoss` (with its default `from_logits=False`) applies the
stable softmax internally. Correspondingly, the model's `forward` returns raw
logits and contains **no** softmax: the loss owns that step. We defined this
`loss` on the base `Classifier` (note the `#@save`), so every classifier in
the rest of the book inherits the numerically stable version.

## Training

Next we train our model. We use Fashion-MNIST images, flattened to 784-dimensional feature vectors.

```{.python .input #softmax-regression-concise-training}
data = d2l.FashionMNIST(batch_size=256)
model = SoftmaxRegression(num_outputs=10, lr=0.1)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

As before, training converges to about 83--84% validation accuracy, the
same solution as the from-scratch version of :numref:`sec_softmax_scratch`
(read it off the validation curve), now in far fewer lines of code.


## Summary

High-level APIs are very convenient at hiding from their user potentially dangerous aspects, such as numerical stability. Moreover, they allow users to design models concisely with very few lines of code. This is both a blessing and a curse. The obvious benefit is that it makes things highly accessible, even to engineers who never took a single class of statistics in their life (in fact, they are part of the target audience of the book). But hiding the sharp edges also comes with a price: a disincentive to add new and different components on your own, since there is little muscle memory for doing it. Moreover, it makes it more difficult to *fix* things whenever the protective padding of
a framework fails to cover all the corner cases entirely. Again, this is due to lack of familiarity.

As such, we strongly urge you to review *both* the bare bones and the elegant versions of many of the implementations that follow. While we emphasize ease of understanding, the implementations are nonetheless usually quite performant (convolutions are the big exception here). It is our intention to allow you to build on these when you invent something new that no framework can give you.

The fused cross-entropy loss in this section is a case in point: it is not merely fewer lines than the from-scratch version, it is the *correct* implementation, the one you should reach for rather than hand-roll, because it computes the log-sum-exp directly and never materializes an unstable softmax.


## Exercises

1. Deep learning uses many different number formats, including FP64 double precision (used extremely rarely),
FP32 single precision, BFLOAT16 (good for compressed representations), FP16 (very unstable), TF32 (a new format from NVIDIA), and INT8. Compute the smallest and largest argument of the exponential function for which the result does not lead to numerical underflow or overflow.
1. INT8 is a very limited format consisting of integers in $[-128, 127]$ (or $[0, 255]$ for the unsigned variant). How could you extend its dynamic range without using more bits? Do standard multiplication and addition still work?
1. Take the from-scratch `softmax` of :numref:`sec_softmax_scratch` and feed it the logits $\mathbf{o} = (1000, 0, 0)$. What do you get, and why? Now compute the loss for the same logits with the framework's `cross_entropy`, passing the logits directly. Why is it finite? Verify that on *benign* logits, e.g., $\mathbf{o} = (2, 1, 0)$, the two routes agree to floating-point precision.
1. Show, using the identity $\ell = \log\sum_k \exp(o_k) - o_y$, that adding the same constant $c$ to every logit leaves the loss unchanged. Why does this make subtracting $\bar{o} = \max_k o_k$ a free and safe choice?

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

::: {.slide title="Concise softmax regression"}
Same model, same data — using the framework's built-in primitives:

- One linear layer instead of hand-rolled `W` and `b`.
- Built-in `cross_entropy` that fuses softmax + log + NLL with
  numerical-stability tricks (the LogSumExp trick).
- Same `Trainer`, same convergence, much less code.
:::

::: {.slide title="The model"}
Imports + a one-line linear layer wrapped in our `Classifier`
scaffold:

@softmax-regression-concise-concise-implementation-of-softmax-regression

@softmax-regression-concise-defining-the-model
:::

::: {.slide title="Softmax + cross-entropy, fused"}
Computing softmax then `log` then NLL separately blows up
numerically when logits are large (`exp(100)` overflows in
common float32 arithmetic). The
framework's `cross_entropy` takes raw **logits** and computes the
loss directly via the LogSumExp trick — equivalent math, stable
arithmetic:

$$\log \sum_j e^{o_j}
  = m + \log \sum_j e^{o_j-m}, \quad m=\max_j o_j.$$

@softmax-regression-concise-softmax-revisited

The model output skips the explicit softmax — the loss handles
both pieces.
:::

::: {.slide title="Train"}
Same Fashion-MNIST data, same 10 epochs, same `Trainer`:

@softmax-regression-concise-training

Identical accuracy curve to the from-scratch version. Built-in
loss = cleaner code + better numerics.
:::

::: {.slide title="Recap"}
- **From-scratch** taught us softmax and cross-entropy; **concise**
  is what we actually use.
- Built-in `cross_entropy(logits, y)` ≡ `softmax → log → NLL`
  with the LogSumExp stability trick baked in.
- The forward pass should output **logits**, not softmax
  probabilities — the loss does the rest.
:::
