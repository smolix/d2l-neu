```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Concise Implementation of Linear Regression
:label:`sec_linear_concise`

In :numref:`sec_linear_scratch` we implemented every piece of linear regression
by hand: we initialized the weights, coded the forward pass, wrote out the squared
error, and ran the parameter update ourselves.
You *should* know how to do this, and doing it once is instructive.
But because data iterators, loss functions, optimizers, and neural network layers
are so common, modern deep learning frameworks package all of them as reusable,
heavily optimized, well-tested components, freeing us to focus on the model
rather than on low-level bookkeeping.
In this section we rebuild the very same model from :numref:`sec_linear_scratch`
using these high-level APIs, showing exactly which hand-rolled piece each
framework primitive replaces.

```{.python .input #linear-regression-concise-concise-implementation-of-linear-regression}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input #linear-regression-concise-concise-implementation-of-linear-regression}
%%tab pytorch
from d2l import torch as d2l
import numpy as np
import torch
from torch import nn
```

```{.python .input #linear-regression-concise-concise-implementation-of-linear-regression}
%%tab tensorflow
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input #linear-regression-concise-concise-implementation-of-linear-regression}
%%tab jax
from d2l import jax as d2l
from flax import nnx
import jax
from jax import numpy as jnp
import optax
```

## Defining the Model

Each component from :numref:`sec_linear_scratch` has a direct counterpart here.
The hand-rolled weight vector $\mathbf{w}$ and bias $b$ are replaced by a single
*layer*; our manual squared-error computation is replaced by a built-in *loss*;
and our explicit parameter-update loop is replaced by an *optimizer* object.
The situation is similar to coding up your own blog from scratch.
Doing it once or twice is rewarding and instructive,
but you would be a lousy web developer
if you spent a month reinventing the wheel.

For standard operations,
we can use a framework's predefined layers,
which allow us to focus
on the layers used to construct the model
rather than worrying about their implementation.
Recall the architecture of a single-layer network
as described in :numref:`fig_single_neuron`.
The layer is called *fully connected*,
since each of its inputs is connected
to each of its outputs
by means of a matrix--vector multiplication.

:begin_tab:`mxnet`
In Gluon, the fully connected layer is defined in the `Dense` class.
Since we only want to generate a single scalar output,
we set that number to 1.
For convenience,
Gluon does not require us to specify
the input shape for each layer.
Hence we do not need to tell Gluon
how many inputs go into this linear layer.
When we first pass data through our model,
e.g., when we execute `net(X)` later,
Gluon will automatically infer the number of inputs to each layer and
thus instantiate the correct model.
We will describe how this works in more detail later.
:end_tab:

:begin_tab:`pytorch`
In PyTorch, the fully connected layer is defined in `Linear` and `LazyLinear` classes (available since version 1.8.0). 
The latter
allows users to specify *merely*
the output dimension,
while the former
additionally asks for
how many inputs go into this layer.
Specifying input shapes is inconvenient and may require nontrivial calculations
(such as in convolutional layers).
Thus, for simplicity, we will use such "lazy" layers
whenever we can. 
:end_tab:

:begin_tab:`tensorflow`
In Keras, the fully connected layer is defined in the `Dense` class.
Since we only want to generate a single scalar output,
we set that number to 1.
For convenience,
Keras does not require us to specify
the input shape for each layer.
We do not need to tell Keras
how many inputs go into this linear layer.
When we first try to pass data through our model,
e.g., when we execute `net(X)` later,
Keras will automatically infer
the number of inputs to each layer.
We will describe how this works in more detail later.
:end_tab:

```{.python .input #linear-regression-concise-defining-the-model-1}
%%tab pytorch
class LinearRegression(d2l.Module):  #@save
    """The linear regression model implemented with high-level APIs."""
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(1)
```

```{.python .input #linear-regression-concise-defining-the-model-1}
%%tab mxnet
class LinearRegression(d2l.Module):  #@save
    """The linear regression model implemented with high-level APIs."""
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Dense(1)
        self.net.initialize(init.Normal(sigma=0.01))
```

```{.python .input #linear-regression-concise-defining-the-model-1}
%%tab tensorflow
class LinearRegression(d2l.Module):  #@save
    """The linear regression model implemented with high-level APIs."""
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        initializer = tf.initializers.RandomNormal(stddev=0.01)
        self.net = tf.keras.layers.Dense(1, kernel_initializer=initializer)
```

```{.python .input #linear-regression-concise-defining-the-model-1}
%%tab jax
class LinearRegression(d2l.Module):  #@save
    """The linear regression model implemented with high-level APIs."""
    def __init__(self, num_inputs, lr, rngs=None):
        super().__init__()
        self.save_hyperparameters(ignore=['rngs'])
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        self.net = nnx.Linear(
            num_inputs, 1, kernel_init=nnx.initializers.normal(0.01),
            rngs=rngs)
```

In the `forward` method we just invoke the built-in `__call__` method of the predefined layers to compute the outputs.

```{.python .input #linear-regression-concise-defining-the-model-2}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(LinearRegression)  #@save
def forward(self, X):
    return self.net(X)
```

```{.python .input #linear-regression-concise-defining-the-model-2}
%%tab jax
@d2l.add_to_class(LinearRegression)  #@save
def forward(self, X):
    return self.net(X)
```

## Defining the Loss Function

:begin_tab:`mxnet`
The `loss` module defines many useful loss functions.
We choose the built-in `loss.L2Loss` rather than maintaining another
implementation.
Because the `loss` that it returns is
the squared error for each example,
we use `mean` to average the loss over the minibatch.
:end_tab:

:begin_tab:`pytorch`
The `MSELoss` class computes the mean squared error (without the $1/2$ factor in :eqref:`eq_mse`).
By default, `MSELoss` returns the average loss over examples.
Using the library primitive also gives us its tested reduction and
automatic-differentiation behavior; performance should be measured rather
than assumed.
:end_tab:

:begin_tab:`tensorflow`
The `MeanSquaredError` class computes the mean squared error (without the $1/2$ factor in :eqref:`eq_mse`).
By default, it returns the average loss over examples.
:end_tab:

```{.python .input #linear-regression-concise-defining-the-loss-function}
%%tab pytorch
@d2l.add_to_class(LinearRegression)  #@save
def loss(self, y_hat, y):
    fn = nn.MSELoss()
    return fn(y_hat, y)
```

```{.python .input #linear-regression-concise-defining-the-loss-function}
%%tab mxnet
@d2l.add_to_class(LinearRegression)  #@save
def loss(self, y_hat, y):
    fn = gluon.loss.L2Loss()
    return 2 * fn(y_hat, y).mean()  # Gluon's L2Loss includes 1/2; multiply by 2 to get plain MSE
```

```{.python .input #linear-regression-concise-defining-the-loss-function}
%%tab tensorflow
@d2l.add_to_class(LinearRegression)  #@save
def loss(self, y_hat, y):
    fn = tf.keras.losses.MeanSquaredError()
    return fn(y, y_hat)
```

```{.python .input #linear-regression-concise-defining-the-loss-function}
%%tab jax
@d2l.add_to_class(LinearRegression)  #@save
def loss(self, y_hat, y):
    return d2l.reduce_mean(jnp.square(y_hat - y))
```

## Defining the Optimization Algorithm

:begin_tab:`mxnet`
Minibatch SGD is a standard tool
for optimizing neural networks
and thus Gluon supports it alongside a number of
variations on this algorithm through its `Trainer` class.
Note that Gluon's `Trainer` class stands
for the optimization algorithm,
while the `Trainer` class we created in :numref:`sec_oo-design`
contains the training method,
i.e., repeatedly call the optimizer
to update the model parameters.
When we instantiate `Trainer`,
we specify the parameters to optimize over,
obtainable from our model `net` via `net.collect_params()`,
the optimization algorithm we wish to use (`sgd`),
and a dictionary of hyperparameters
required by our optimization algorithm.
:end_tab:

:begin_tab:`pytorch`
Minibatch SGD is a standard tool
for optimizing neural networks
and thus PyTorch supports it alongside a number of
variations on this algorithm in the `optim` module.
When we instantiate an `SGD` instance,
we specify the parameters to optimize over,
obtainable from our model via `self.parameters()`,
and the learning rate (`self.lr`)
required by our optimization algorithm.
:end_tab:

:begin_tab:`tensorflow`
Minibatch SGD is a standard tool
for optimizing neural networks
and thus Keras supports it alongside a number of
variations on this algorithm in the `optimizers` module.
:end_tab:

```{.python .input #linear-regression-concise-defining-the-optimization-algorithm}
%%tab pytorch
@d2l.add_to_class(LinearRegression)  #@save
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), self.lr)
```

```{.python .input #linear-regression-concise-defining-the-optimization-algorithm}
%%tab tensorflow
@d2l.add_to_class(LinearRegression)  #@save
def configure_optimizers(self):
    return tf.keras.optimizers.SGD(self.lr)
```

```{.python .input #linear-regression-concise-defining-the-optimization-algorithm}
%%tab jax
@d2l.add_to_class(LinearRegression)  #@save
def configure_optimizers(self):
    return optax.sgd(self.lr)
```

```{.python .input #linear-regression-concise-defining-the-optimization-algorithm}
%%tab mxnet
@d2l.add_to_class(LinearRegression)  #@save
def configure_optimizers(self):
    return gluon.Trainer(self.collect_params(),
                         'sgd', {'learning_rate': self.lr})
```

## Training

You might have noticed that expressing our model through
high-level APIs of a deep learning framework
requires fewer lines of code.
We did not have to allocate parameters individually,
define our loss function, or implement minibatch SGD.
Once we start working with much more complex models,
the advantages of the high-level API will grow considerably.

Now that we have all the basic pieces in place,
the training loop itself is the same
as the one we implemented from scratch.
So we just call the `fit` method (introduced in :numref:`oo-design-training`),
which relies on the implementation of the `fit_epoch` method
in :numref:`sec_linear_scratch`,
to train our model.

```{.python .input #linear-regression-concise-training-1}
%%tab pytorch
model = LinearRegression(lr=0.03)
data = d2l.SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
# Materialize lazy parameters before replacing their default initialization.
model(data.X[:1])
with torch.no_grad():
    model.net.weight.normal_(0, 0.01)
    model.net.bias.fill_(0)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

```{.python .input #linear-regression-concise-training-1}
%%tab mxnet, tensorflow
model = LinearRegression(lr=0.03)
data = d2l.SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

```{.python .input #linear-regression-concise-training-1}
%%tab jax
model = LinearRegression(2, lr=0.03)
data = d2l.SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

Below, we
compare the model parameters learned
by training on finite data
and the actual parameters
that generated our dataset.
This is where the concise version differs conceptually from the scratch one:
the parameters no longer hang off our class as `self.w` and `self.b` but live
*inside* the layer object, so `get_w_b` reaches through `net` to find them.
As in our implementation from scratch,
note that our estimated parameters
are close to their true counterparts.

```{.python .input #linear-regression-concise-training-2}
%%tab pytorch
@d2l.add_to_class(LinearRegression)  #@save
def get_w_b(self):
    return (self.net.weight.detach(), self.net.bias.detach())
w, b = model.get_w_b()
```

```{.python .input #linear-regression-concise-training-2}
%%tab mxnet
@d2l.add_to_class(LinearRegression)  #@save
def get_w_b(self):
    return (self.net.weight.data(), self.net.bias.data())
w, b = model.get_w_b()
```

```{.python .input #linear-regression-concise-training-2}
%%tab tensorflow
@d2l.add_to_class(LinearRegression)  #@save
def get_w_b(self):
    return (self.get_weights()[0], self.get_weights()[1])

w, b = model.get_w_b()
```

```{.python .input #linear-regression-concise-training-2}
%%tab jax
@d2l.add_to_class(LinearRegression)  #@save
def get_w_b(self):
    return self.net.kernel[...], self.net.bias[...]

w, b = model.get_w_b()
```

```{.python .input #linear-regression-concise-training-3}
print(f'error in estimating w: {data.w - d2l.reshape(w, data.w.shape)}')
print(f'error in estimating b: {data.b - b}')
```

## Summary

This section contains the first
implementation of a deep network (in this book)
to tap into the conveniences afforded
by modern deep learning frameworks,
such as MXNet :cite:`Chen.Li.Li.ea.2015`, 
JAX :cite:`Frostig.Johnson.Leary.2018`, 
PyTorch :cite:`Paszke.Gross.Massa.ea.2019`, 
and Tensorflow :cite:`Abadi.Barham.Chen.ea.2016`.
We used framework defaults for loading data, defining a layer,
a loss function, an optimizer and a training loop.
Whenever the framework provides all necessary features,
it is generally a good idea to use them,
since the library implementations of these components
tend to be heavily optimized for performance
and properly tested for reliability.
At the same time, try not to forget
that these modules *can* be implemented directly.
This is especially important for aspiring researchers
who wish to live on the leading edge of model development,
where you will be inventing new components
that cannot possibly exist in any current library.

:begin_tab:`mxnet`
In Gluon, the `data` module provides tools for data processing,
the `nn` module defines a large number of neural network layers,
and the `loss` module defines many common loss functions.
Moreover, the `initializer` gives access
to many choices for parameter initialization.
Conveniently for the user,
dimensionality and storage are automatically inferred.
A consequence of this lazy initialization is that
you must not attempt to access parameters
before they have been instantiated (and initialized).
:end_tab:

:begin_tab:`pytorch`
In PyTorch, the `data` module provides tools for data processing,
the `nn` module defines a large number of neural network layers and common loss functions.
Because we used `nn.LazyLinear`, the input dimensions are inferred automatically
on the first forward pass. Before that pass, the layer contains
`UninitializedParameter` placeholders: writing to them does not initialize the
eventual weights. The training cell therefore makes a one-example dry run and
then initializes the materialized parameters inside `torch.no_grad()`.
This lazy shape inference pays off in deeper networks (convolutional layers,
variable-length sequences), where computing the input size of each layer by hand
would be tedious and error-prone.
:end_tab:

:begin_tab:`tensorflow`
In TensorFlow, the `data` module provides tools for data processing,
the `keras` module defines a large number of neural network layers and common loss functions.
Moreover, the `initializers` module provides various methods for model parameter initialization.
Dimensionality and storage for networks are automatically inferred
(but be careful not to attempt to access parameters before they have been initialized).
:end_tab:

## Exercises

1. The framework loss functions used above (e.g., `nn.MSELoss`) return the mean loss over the minibatch by default. How would you need to change the learning rate if instead you replaced this average with the *sum* of the losses over the minibatch (e.g., by passing `reduction='sum'`)?
1. Review the framework documentation to see which loss functions are provided. In particular,
   replace the squared loss with Huber's robust loss function. That is, use the loss function
   $$l(y,y') = \begin{cases}|y-y'| -\frac{\sigma}{2} & \textrm{ if } |y-y'| > \sigma \\ \frac{1}{2 \sigma} (y-y')^2 & \textrm{ otherwise}\end{cases}$$
   Rerun the outlier demonstration of :numref:`subsec_linear-regression-loss-function` (one corrupted label) with it: does Huber's loss recover the robust estimate, the least-squares one, or something in between? (Compare the penalty curves in :numref:`fig_linreg-loss-menu`.)
1. How do you access the gradient of the weights of the model?
1. What is the effect on the solution if you change the learning rate and the number of epochs? Does it keep on improving?
1. How does the solution change as you vary the amount of data generated?
    1. Plot the estimation error for $\hat{\mathbf{w}} - \mathbf{w}$ and $\hat{b} - b$ as a function of the amount of data. Hint: increase the amount of data logarithmically rather than linearly, i.e., 5, 10, 20, 50, ..., 10,000 rather than 1000, 2000, ..., 10,000.
    2. Why is the suggestion in the hint appropriate?
1. Time the from-scratch implementation of :numref:`sec_linear_scratch` against the concise one here, training each for 10, 100, and 1,000 epochs on the same synthetic dataset. Which is faster, and does the gap grow with the number of epochs? What does this tell you about the overhead of Python-level parameter bookkeeping versus framework-optimized operations?


:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/44)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/45)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/204)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/17977)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §3.5]{.kicker}

The same model, the concise way<br>**batteries-included layers, losses, and optimizers replace the hand-rolled parts**.
:::
:::

::: {.slide title="From hand-rolled to high-level"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
Last section we wrote *every* piece by hand: the weight vector, the
forward pass, the squared error, the update step.

Those pieces are so universal that frameworks ship them, tuned and
tested. We swap each one for its built-in counterpart:

::: {.d2l-note}
**Layer** replaces `w`, `b` · **loss** replaces our squared error ·
**optimizer** replaces the update loop.
:::
:::

::: {.col .narrow}
| By hand | Built-in |
|---|---|
| `w`, `b` | a **layer** |
| MSE math | a **loss** |
| update step | an **optimizer** |
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[The Model]{.dtitle}

[a single linear layer]{.dsub}
:::
:::

::: {.slide title="The layer already is the model"}
[The Model]{.kicker}

::: {.cols .vc}
::: {.col}
What we hand-rolled as `w`, `b`, and a matrix--vector product, every
framework ships as a **fully connected layer**: each input wired to the
one output: exactly the picture of linear regression.

::: {.d2l-note}
The layer owns its parameters. We no longer allocate them, initialize
them, or even know their shapes ahead of time.
:::
:::

::: {.col .fig .big}
![One fully connected layer with a single output is linear regression.](../img/singleneuron.svg)
:::
:::
:::

::: {.slide title="One layer, not a weight vector" except="jax,tensorflow,mxnet"}
[The Model]{.kicker}

`LazyLinear(1)` is the whole model. The **lazy** variant defers the
input dimension until the first forward pass. Initialize its parameters only
after that first pass:

@linear-regression-concise-defining-the-model-1

::: {.d2l-note .rule}
Lazy shape inference pays off in deep nets (conv layers, variable-length
sequences) where the input size is tedious to work out.
:::
:::

::: {.slide title="One layer, not a weight vector" only="mxnet"}
[The Model]{.kicker}

`Dense(1)` is the whole model. Gluon **infers** the input dimension on
the first forward pass, so we specify only the single output:

@linear-regression-concise-defining-the-model-1

::: {.d2l-note}
Initialize the weights now; storage is allocated lazily at first use.
:::
:::

::: {.slide title="One layer, not a weight vector" only="tensorflow"}
[The Model]{.kicker}

`Dense(1)` is the whole model. Keras **infers** the input dimension on
the first forward pass, so we specify only the single output:

@linear-regression-concise-defining-the-model-1

::: {.d2l-note}
The `RandomNormal` initializer breaks symmetry; storage is allocated at
first use.
:::
:::

::: {.slide title="One layer, with explicit sizes and randomness" only="jax"}
[The Model]{.kicker}

`nnx.Linear(num_inputs, 1, rngs=rngs)` is the whole model. NNX has no
lazy mode, so we state the input width and hand the layer an explicit
RNG stream — and its parameters exist, on the module, as soon as the
constructor returns:

@linear-regression-concise-defining-the-model-1

::: {.d2l-note .rule}
Two JAX signatures — explicit shapes, explicit randomness. The weights
then live on the module, just as in the other frameworks.
:::
:::

::: {.slide title="The forward pass is a one-liner"}
[The Model]{.kicker}

`forward` just calls the layer. All the matrix--vector arithmetic we
wrote by hand now lives inside it:

@linear-regression-concise-defining-the-model-2
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Loss & Optimizer]{.dtitle}

[two more pieces, off the shelf]{.dsub}
:::
:::

::: {.slide title="Loss: built-in mean squared error" except="mxnet"}
[Loss & Optimizer]{.kicker}

The framework's MSE replaces our hand-written squared error:

@linear-regression-concise-defining-the-loss-function

::: {.d2l-note}
It omits the $\tfrac{1}{2}$ factor we used by hand, and averages over the
minibatch by default.
:::
:::

::: {.slide title="Loss: built-in mean squared error" only="mxnet"}
[Loss & Optimizer]{.kicker}

Gluon's `L2Loss` replaces our hand-written squared error, with one
quirk:

@linear-regression-concise-defining-the-loss-function

::: {.d2l-note .warn}
`L2Loss` *includes* the $\tfrac{1}{2}$ factor, so we multiply by 2 to
recover plain MSE before averaging.
:::
:::

::: {.slide title="Optimizer: minibatch SGD in one call"}
[Loss & Optimizer]{.kicker}

The update loop becomes a single optimizer object, handed the
parameters and the learning rate:

@linear-regression-concise-defining-the-optimization-algorithm

::: {.d2l-note}
The same `optim`/`Trainer` family also gives momentum, Adam, and more
by swapping one line.
:::
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[Training]{.dtitle}

[the scaffold never changed]{.dsub}
:::
:::

::: {.slide title="The same Trainer drives it all"}
[Training]{.kicker}

::: {.cols .vc}
::: {.col}
Our `Trainer`, `Module`, and `DataModule` from the object-oriented-design
section don't care that the model is now a built-in layer.

The training loop is **identical** to the from-scratch version.
:::

::: {.col .fig .big}
![](../img/mdl-linreg-oo-classes.svg)
:::
:::
:::

::: {.slide title="Fit: same data, same curve, a fraction of the code"}
[Training]{.kicker}

::: {.cols .vc}
::: {.col}
Same synthetic data, same ten epochs, same `fit` call as the
linear-regression-from-scratch section:

@-linear-regression-concise-training-1

Nothing about the *training run* can tell the two implementations apart:
only the amount of code we wrote changed.
:::

::: {.col .fig}
@!linear-regression-concise-training-1
:::
:::
:::

::: {.slide title="Where the parameters live now"}
[Training · payoff]{.kicker}

They no longer hang off our class as `self.w`, `self.b`; they live
**inside** the layer, so `get_w_b` reaches through `net`:

@linear-regression-concise-training-2

. . .

@linear-regression-concise-training-3

::: {.d2l-note}
Same verdict as the linear-regression-from-scratch section: the true
$\mathbf{w}^* = [2,-3.4]$, $b^* = 4.2$
recovered to a few $10^{-4}$. The built-in pieces really do compute the
same thing our hand-rolled ones did.
:::
:::

::: {.slide title="Summary"}
[Wrap-up]{.kicker}

::: {.cols}
::: {.col}
- **From scratch** showed *what* happens; **concise** is what we
  actually use day to day.
- A single **layer** stands in for `w`, `b`; a built-in **loss** and
  **optimizer** replace the rest.
:::

::: {.col}
- The `Module` / `Trainer` / `DataModule` scaffold is **unchanged**;
  only the model's internals got shorter.
- Same minibatch loop, same convergence: ~5 lines of model code, error
  order $10^{-4}$.
:::
:::
:::
