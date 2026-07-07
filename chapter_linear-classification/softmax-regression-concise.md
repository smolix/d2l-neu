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
$88$ and gradually underflows past about $-88$
(entering the *subnormal* range),
reaching exactly $0$ only near $-104$. A single large
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
smooth upper bound on $\max_k o_k$ (you proved this, including the fact that
the gap never exceeds $\log q$, in :numref:`sec_softmax`, exercise 6); the
second equality is the only safe way
to evaluate it, since every exponent $o_k - \bar{o} \leq 0$. This is precisely
what the built-in cross-entropy loss computes when handed raw logits: it never
forms the softmax probabilities at all, so neither $\exp$ of a large number nor
$\log$ of a zero ever occurs. Because the fused loss differentiates this exact
expression, its gradient is exactly
$\partial_{o_j}\ell = \mathrm{softmax}(\mathbf{o})_j - y_j$ derived in
:numref:`subsec_softmax_and_derivatives`, with no clamp to perturb it. We keep
the explicit softmax of :numref:`sec_softmax_scratch` only for *reading off*
predicted probabilities at inference time; for the loss we pass logits and let
the fused operation do the rest.

For two classes with
logits $(x, 0)$ the loss's first term is $\mathrm{lse}(x, 0) = \log(1 + e^x)$,
which hugs $\max(x, 0)$ from above, with the gap largest at the tie $x = 0$,
where it equals $\log 2 \approx 0.69$, our bound $\log q$ for $q = 2$:

```{.python .input #softmax-regression-concise-lse-vs-max}
x = d2l.arange(-4.0, 4.0, 0.01)
lse, mx = d2l.log(1 + d2l.exp(x)), (x + d2l.abs(x)) / 2
d2l.plot(x, [lse, mx, lse - mx], 'x', legend=['lse(x, 0)', 'max(x, 0)', 'gap'])
```

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

The built-in fused loss (named differently in each library) takes **logits**,
not probabilities: passing softmax outputs would apply the softmax twice.
Correspondingly, the model's `forward` returns raw logits and contains **no**
softmax: the loss owns that step. We defined this `loss` on the base
`Classifier` (note the `#@save`), so every classifier in the rest of the book
inherits the numerically stable version.

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

As such, we strongly urge you to review *both* the bare bones and the concise versions of many of the implementations that follow. While we emphasize ease of understanding, the implementations are nonetheless usually quite performant (convolutions are the big exception here). It is our intention to allow you to build on these when you invent something new that no framework can give you.

The fused cross-entropy loss in this section is the implementation to reach for: it computes the log-sum-exp directly and never materializes an unstable softmax.


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

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §4.5]{.kicker}

Concise softmax regression<br>One linear layer, and the *numerically stable* loss the framework hands you for free.
:::
:::

::: {.slide title="Same model, far fewer lines"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
The from-scratch version built softmax, then `log`, then the
negative-log-likelihood by hand. The concise version replaces all
of it with two built-ins:

- one **linear layer** in place of `W` and `b`;
- one **cross-entropy** call that takes raw scores.

::: {.d2l-note}
The convenience hides one thing: the loss is
not the naive `softmax → log → NLL`. It is the *stable* rewrite.
:::
:::

::: {.col .fig .big}
![](../img/mdl-clf-loss-accuracy.svg){width=100%}
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[The concise model]{.dtitle}

[a linear layer in the Classifier scaffold]{.dsub}
:::
:::

::: {.slide title="One linear layer, wrapped"}
[The model]{.kicker}

Flatten each image to a 784-vector, then a single linear layer to
the 10 class scores. Everything else is inherited from
`Classifier`:

@softmax-regression-concise-defining-the-model
:::

::: {.slide title="The forward pass returns logits"}
[The model]{.kicker}

Notice what `forward` does **not** do: there is no softmax. It
returns raw scores $\mathbf{o}\in\mathbb{R}^{10}$ (the *logits*).

. . .

::: {.d2l-note .rule}
**Logits, not probabilities.** The loss will own the softmax step.
Applying softmax here *and* in the loss would apply it twice.
:::
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Why the loss is rewritten]{.dtitle}

[overflow, underflow, and the log-sum-exp trick]{.dsub}
:::
:::

::: {.slide title="The danger hiding in softmax"}
[Numerical stability]{.kicker}

Softmax exponentiates the logits: $\hat y_j = \exp(o_j) / \sum_k \exp(o_k)$.

. . .

Float32 spans roughly $10^{-38}$ to $10^{38}$: $\exp$ **overflows to
$\infty$** once its argument passes $\approx +88$, and past $\approx -88$ it
**gradually underflows** through the subnormals, hitting exactly $0$ near
$-104$.

. . .

::: {.d2l-note .warn}
Feed the from-scratch softmax the logits $\mathbf{o}=(1000, 0, 0)$:
$\exp(1000)=\infty$, the ratio is $\infty/\infty=$ `NaN`, and one
`NaN` poisons the entire backward pass. We watched this happen in §4.4;
the fused loss below never forms that ratio.
:::
:::

::: {.slide title="Fix, step 1: shift by the max"}
[Numerical stability]{.kicker}

Softmax is **unchanged** if we subtract the same constant from every
logit (the $\exp\bar{o}$ factors cancel). Choose $\bar{o}=\max_k o_k$:

$$\hat y_j =
\frac{\exp(o_j - \bar{o})}{\sum_k \exp(o_k - \bar{o})},
\qquad \bar{o} = \max_k o_k.$$

. . .

Now every exponent $o_j - \bar{o} \le 0$, so each $\exp$ lands in
$(0, 1]$: **no overflow**. The denominator sits in $[1, q]$.
:::

::: {.slide title="Fix, step 2: never form the softmax"}
[Numerical stability]{.kicker}

Underflow could still bite if we then took $\log$ of a near-zero
probability. But we only ever want $\log \hat y_j$ for the loss, so
fold the $\log$ in and the division disappears:

$$\log \hat y_j = (o_j - \bar{o}) - \log \sum_k \exp(o_k - \bar{o}).$$

. . .

No probability is ever materialized: no $\exp$ of a large number, no
$\log$ of a zero.
:::

::: {.slide title="The log-sum-exp loss"}
[Numerical stability]{.kicker}

For true class $y$ the loss $\ell = -\log \hat y_y$ becomes a function
of the **logits alone**:

$$\ell(y, \mathbf{o}) =
\underbrace{\bar{o} + \log \textstyle\sum_k \exp(o_k - \bar{o})}_{\text{log-sum-exp, evaluated stably}} - o_y.$$

. . .

::: {.cols}
::: {.col}
$\log\sum_k\exp(o_k)$ is a **smooth upper bound on $\max_k o_k$**, the
"soft max" the function is named for.
:::

::: {.col}
Its gradient is
$\partial_{o_j}\ell = \mathrm{softmax}(\mathbf{o})_j - y_j$, with no
clamp to perturb it.
:::
:::
:::

::: {.slide title="The soft max hugs the hard max: gap at most log 2" only="pytorch"}
[Numerical stability · measured]{.kicker}

::: {.cols .vc}
::: {.col .narrow}
For two classes with logits $(x, 0)$ the loss's first term is
$\mathrm{lse}(x, 0) = \log(1 + e^x)$. Plot it against $\max(x, 0)$:

::: {.d2l-note .rule}
The gap peaks at the **tie** $x = 0$, where it equals
$\log 2 \approx 0.69$, the bound $\log q$ you proved in §4.1
(exercise 6), here at $q = 2$. Away from the tie, soft and hard max are
indistinguishable.
:::
:::

::: {.col .fig .big}
@!softmax-regression-concise-lse-vs-max
:::
:::
:::

::: {.slide title="The soft max hugs the hard max: gap at most log 2" except="pytorch"}
[Numerical stability · the bound]{.kicker}

For two classes with logits $(x, 0)$ the loss's first term is
$\mathrm{lse}(x, 0) = \log(1 + e^x)$, a smooth curve hugging
$\max(x, 0)$ from above:

$$\max_k o_k \;\le\; \mathrm{lse}(\mathbf{o}) \;\le\; \max_k o_k + \log q.$$

::: {.d2l-note .rule}
The gap peaks at the **tie** $x = 0$, where it equals
$\log 2 \approx 0.69$, the bound $\log q$ you proved in §4.1
(exercise 6), here at $q = 2$. Away from the tie, soft and hard max are
indistinguishable.
:::
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[In code]{.dtitle}

[one fused call, four frameworks]{.dsub}
:::
:::

::: {.slide title="Hand the loss the logits" only="pytorch"}
[The fused loss]{.kicker}

PyTorch's `F.cross_entropy` consumes **logits** by definition: it
computes the stable log-sum-exp internally. We attach it to the base
`Classifier`, so every classifier in the book inherits it:

@softmax-regression-concise-softmax-revisited
:::

::: {.slide title="Hand the loss the logits" only="tensorflow"}
[The fused loss]{.kicker}

`SparseCategoricalCrossentropy(from_logits=True)` is the switch that
says "these are scores, not probabilities", so Keras does the stable
log-sum-exp instead of assuming a softmax already ran:

@softmax-regression-concise-softmax-revisited
:::

::: {.slide title="Hand the loss the logits" only="jax"}
[The fused loss]{.kicker}

Optax names it for exactly what it does:
`softmax_cross_entropy_with_integer_labels` takes logits plus integer
labels and fuses the stable softmax with the cross-entropy:

@softmax-regression-concise-softmax-revisited
:::

::: {.slide title="Hand the loss the logits" only="mxnet"}
[The fused loss]{.kicker}

MXNet's `SoftmaxCrossEntropyLoss` (default `from_logits=False`) applies
the *stable* softmax internally, then the cross-entropy, so we still
pass raw logits, never probabilities:

@softmax-regression-concise-softmax-revisited
:::

::: {.slide title="One rule for the fused loss"}
[The fused loss]{.kicker}

The name differs by library; the contract does not. **The built-in fused
loss takes logits, not probabilities**: passing softmax outputs would
softmax twice.

Defined once on `Classifier` (note the `#@save`): the whole book
inherits the stable loss.
:::

::: {.slide}
::: {.divider}
[04]{.dnum}

[Train]{.dtitle}

[same data, same curve, less code]{.dsub}
:::
:::

::: {.slide title="Train"}
[Results]{.kicker}

::: {.cols .vc}
::: {.col}
Same Fashion-MNIST, same 10 epochs, same `Trainer`:

@softmax-regression-concise-training
:::

::: {.col .narrow}
Converges to the **same ~83–84%** validation accuracy as the
from-scratch model of §4.4, now in a handful of lines, and with
the *correct* loss instead of a clamped one.
:::
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

::: {.cols}
::: {.col}
- **From scratch** taught *what* softmax and cross-entropy are;
  **concise** is what we reach for.
- The forward pass outputs **logits**; the built-in loss owns the
  softmax.
:::

::: {.col}
- That built-in is the **log-sum-exp** rewrite
  $\ell = \bar{o} + \log\sum_k e^{o_k-\bar{o}} - o_y$, not a naive
  `softmax → log → NLL`.
- lse is a **smooth max**: within $\log q$ of $\max_k o_k$, gap largest
  ($\log 2$ for $q{=}2$) exactly at the tie.
- Fewer lines **and** numerically correct: float32's $\pm 88$ (and $-104$)
  cliffs never come into play.
:::
:::
:::
