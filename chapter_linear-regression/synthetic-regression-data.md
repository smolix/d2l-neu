```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Synthetic Regression Data
:label:`sec_synthetic-regression-data`


Before we can train a model we need data.
Real datasets are what we ultimately care about,
but they conflate three separate sources of failure:
a misspecified model, a flawed optimization algorithm,
and pathological data.
When a method performs poorly on real data, these explanations can be
difficult to distinguish.
*Synthetic data* removes this ambiguity by construction.
If we know the data-generating process exactly
(the true weights $\mathbf{w}^*$, the true bias $b^*$,
and the noise distribution),
then any *systematic* failure to recover them
(beyond the irreducible noise) points to the algorithm
or implementation.
This is why a compatible synthetic dataset is a useful early implementation
test for a new learning method.
We first confirm that the implementation solves a compatible problem with known
parameters before evaluating it on real data.

```{.python .input #synthetic-regression-data}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx, gluon
import random
npx.set_np()
```

```{.python .input #synthetic-regression-data}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import random
```

```{.python .input #synthetic-regression-data}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import random
```

```{.python .input #synthetic-regression-data}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import numpy as np
import random
import tensorflow as tf
```

## Generating the Dataset

For clarity, this example uses two-dimensional inputs.
The following code snippet generates 1000 examples
with 2-dimensional features drawn 
from a standard normal distribution.
The resulting design matrix $\mathbf{X}$
belongs to $\mathbb{R}^{1000 \times 2}$. 
We generate each label by applying 
a *ground truth* linear function, 
corrupting them via additive noise $\boldsymbol{\epsilon}$, 
drawn independently and identically for each example:

$$\mathbf{y}= \mathbf{X} \mathbf{w}^* + b^* + \boldsymbol{\epsilon}.$$

For convenience we assume that $\boldsymbol{\epsilon}$ is drawn 
from a normal distribution with mean $\mu= 0$ 
and standard deviation $\sigma = 0.01$.
We put the generation code in the `__init__` method of a subclass
of `d2l.DataModule` (introduced in :numref:`oo-design-data`),
calling `save_hyperparameters()` so that every constructor argument
(the parameters `w` and `b`, the noise level, the split sizes, and
`batch_size`) is stored as an attribute and the dataset stays
introspectable.

```{.python .input #synthetic-regression-data-generating-the-dataset-1}
%%tab pytorch
class SyntheticRegressionData(d2l.DataModule):  #@save
    """Synthetic data for linear regression."""
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000, 
                 batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val
        self.X = d2l.randn(n, len(w))
        eps = d2l.randn(n, 1) * noise
        self.y = d2l.matmul(self.X, d2l.reshape(w, (-1, 1))) + b + eps
```

```{.python .input #synthetic-regression-data-generating-the-dataset-1}
%%tab tensorflow
class SyntheticRegressionData(d2l.DataModule):  #@save
    """Synthetic data for linear regression."""
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000, 
                 batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val
        self.X = tf.random.normal((n, w.shape[0]))
        eps = tf.random.normal((n, 1)) * noise
        self.y = d2l.matmul(self.X, d2l.reshape(w, (-1, 1))) + b + eps
```

```{.python .input #synthetic-regression-data-generating-the-dataset-1}
%%tab jax
class SyntheticRegressionData(d2l.DataModule):  #@save
    """Synthetic data for linear regression."""
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000,
                 batch_size=32, key=None):
        super().__init__()
        self.save_hyperparameters()
        # Resolve the key at call time rather than reusing a key in the signature.
        key = jax.random.key(0) if key is None else key
        n = num_train + num_val
        key1, key2 = jax.random.split(key)
        self.X = jax.random.normal(key1, (n, w.shape[0]))
        eps = jax.random.normal(key2, (n, 1)) * noise
        self.y = d2l.matmul(self.X, d2l.reshape(w, (-1, 1))) + b + eps
```

```{.python .input #synthetic-regression-data-generating-the-dataset-1}
%%tab mxnet
class SyntheticRegressionData(d2l.DataModule):  #@save
    """Synthetic data for linear regression."""
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000, 
                 batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val
        self.X = d2l.randn(n, len(w))
        eps = d2l.randn(n, 1) * noise
        self.y = d2l.matmul(self.X, d2l.reshape(w, (-1, 1))) + b + eps
```

Below, we set the true parameters to $\mathbf{w}^* = [2, -3.4]^\top$ and $b^* = 4.2$.
Later, we can check our estimated parameters against these *ground truth* values.

```{.python .input #synthetic-regression-data-generating-the-dataset-2}
data = SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
```

Each row of `data.X` is a feature vector in $\mathbb{R}^2$, and each row of `data.y` is a scalar label. We inspect the first entry.

```{.python .input #synthetic-regression-data-generating-the-dataset-3}
print('features:', data.X[0],'\nlabel:', data.y[0])
```

## Reading the Dataset

Training machine learning models often requires multiple passes over a dataset, 
grabbing one minibatch of examples at a time. 
This data is then used to update the model. 
To illustrate how this works, we 
implement the `get_dataloader` method, 
registering it in the `SyntheticRegressionData` class via `add_to_class` (introduced in :numref:`oo-design-utilities`).
It takes a batch size, a matrix of features,
and a vector of labels, and generates minibatches of size `batch_size`.
As such, each minibatch consists of a tuple of features and labels. 
Note that we need to be mindful of whether we're in training or validation mode: 
in the former, we will want to read the data in random order, 
whereas for the latter, being able to read data in a pre-defined order 
may be important for debugging purposes.

```{.python .input #synthetic-regression-data-reading-the-dataset-1}
%%tab pytorch
@d2l.add_to_class(SyntheticRegressionData)
def get_dataloader(self, train):
    if train:
        indices = list(range(0, self.num_train))
        # The examples are read in random order
        random.shuffle(indices)
    else:
        indices = list(range(self.num_train, self.num_train+self.num_val))
    for i in range(0, len(indices), self.batch_size):
        batch_indices = d2l.tensor(indices[i: i+self.batch_size])
        yield self.X[batch_indices], self.y[batch_indices]
```

```{.python .input #synthetic-regression-data-reading-the-dataset-1}
%%tab tensorflow
@d2l.add_to_class(SyntheticRegressionData)
def get_dataloader(self, train):
    if train:
        indices = list(range(0, self.num_train))
        # The examples are read in random order
        random.shuffle(indices)
    else:
        indices = list(range(self.num_train, self.num_train+self.num_val))
    for i in range(0, len(indices), self.batch_size):
        j = tf.constant(indices[i : i+self.batch_size])
        yield tf.gather(self.X, j), tf.gather(self.y, j)
```

```{.python .input #synthetic-regression-data-reading-the-dataset-1}
%%tab jax
@d2l.add_to_class(SyntheticRegressionData)
def get_dataloader(self, train):
    if train:
        indices = list(range(0, self.num_train))
        # The examples are read in random order
        random.shuffle(indices)
    else:
        indices = list(range(self.num_train, self.num_train+self.num_val))
    for i in range(0, len(indices), self.batch_size):
        batch_indices = d2l.tensor(indices[i: i+self.batch_size])
        yield self.X[batch_indices], self.y[batch_indices]
```

```{.python .input #synthetic-regression-data-reading-the-dataset-1}
%%tab mxnet
@d2l.add_to_class(SyntheticRegressionData)
def get_dataloader(self, train):
    if train:
        indices = list(range(0, self.num_train))
        # The examples are read in random order
        random.shuffle(indices)
    else:
        indices = list(range(self.num_train, self.num_train+self.num_val))
    for i in range(0, len(indices), self.batch_size):
        batch_indices = d2l.tensor(indices[i: i+self.batch_size])
        yield self.X[batch_indices], self.y[batch_indices]
```

We inspect the first minibatch. The feature shape records the minibatch size
and input dimensionality, while the label shape has the same leading dimension.

```{.python .input #synthetic-regression-data-reading-the-dataset-2}
X, y = next(iter(data.train_dataloader()))
print('X shape:', X.shape, '\ny shape:', y.shape)
```

Iterating over `data.train_dataloader()` yields distinct minibatches
until the dataset is exhausted (try it).
Writing the loader by hand makes every step explicit,
but it costs us in three ways:
all of the data must fit in memory, the iteration is single-threaded
Python looping over indices, and there is no prefetching to overlap
data loading with computation on the previous batch.
The data loaders built into a deep learning framework fix all three.
They run several worker processes in parallel, prefetch the next batch
while the current one trains, and stream from sources such as files,
network streams, or generators that produce data on the fly.
We now switch to the framework's built-in loader,
which presents an identical interface to the caller.

## Concise Implementation of the Data Loader

Rather than writing our own iterator,
we can call the existing API in a framework to load data.
As before, we need a dataset with features `X` and labels `y`. 
Beyond that, we set `batch_size` in the built-in data loader 
and let it take care of shuffling examples  efficiently.

:begin_tab:`jax`
JAX offers a NumPy-like API with device acceleration and functional
transformations, and at least the current version ships no data loading
methods of its own. Other libraries already provide great data loaders,
and JAX suggests using them instead. Here we will grab TensorFlow's data loader
and modify it slightly to make it work with JAX.
:end_tab:

```{.python .input #synthetic-regression-data-concise-implementation-of-the-data-loader-1}
%%tab pytorch
@d2l.add_to_class(d2l.DataModule)  #@save
def get_tensorloader(self, tensors, train, indices=slice(0, None)):
    tensors = tuple(a[indices] for a in tensors)
    dataset = torch.utils.data.TensorDataset(*tensors)
    return torch.utils.data.DataLoader(dataset, self.batch_size,
                                       shuffle=train)
```

```{.python .input #synthetic-regression-data-concise-implementation-of-the-data-loader-1}
%%tab tensorflow
@d2l.add_to_class(d2l.DataModule)  #@save
def get_tensorloader(self, tensors, train, indices=slice(0, None)):
    tensors = tuple(a[indices] for a in tensors)
    shuffle_buffer = tensors[0].shape[0] if train else 1
    return tf.data.Dataset.from_tensor_slices(tensors).shuffle(
        buffer_size=shuffle_buffer).batch(self.batch_size)
```

```{.python .input #synthetic-regression-data-concise-implementation-of-the-data-loader-1}
%%tab jax
class TensorFlowDataLoader:  #@save
    """Expose a tf.data.Dataset as re-iterable NumPy batches."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        return self.dataset.as_numpy_iterator()

    def __len__(self):
        return len(self.dataset)

@d2l.add_to_class(d2l.DataModule)  #@save
def get_tensorloader(self, tensors, train, indices=slice(0, None)):
    tensors = tuple(a[indices] for a in tensors)
    # Use TensorFlow's data loader. JAX and Flax do not provide data-loading
    # functionality. `drop_remainder=train` keeps every
    # *training* minibatch the same shape, so a `@jax.jit`'d step
    # function compiles once per epoch instead of recompiling for the
    # smaller last batch.
    shuffle_buffer = tensors[0].shape[0] if train else 1
    dataset = tf.data.Dataset.from_tensor_slices(tensors).shuffle(
        buffer_size=shuffle_buffer).batch(
            self.batch_size, drop_remainder=train)
    return TensorFlowDataLoader(dataset)
```

```{.python .input #synthetic-regression-data-concise-implementation-of-the-data-loader-1}
%%tab mxnet
@d2l.add_to_class(d2l.DataModule)  #@save
def get_tensorloader(self, tensors, train, indices=slice(0, None)):
    tensors = tuple(a[indices] for a in tensors)
    dataset = gluon.data.ArrayDataset(*tensors)
    return gluon.data.DataLoader(dataset, self.batch_size,
                                 shuffle=train)
```

```{.python .input #synthetic-regression-data-concise-implementation-of-the-data-loader-2}
@d2l.add_to_class(SyntheticRegressionData)  #@save
def get_dataloader(self, train):
    i = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader((self.X, self.y), train, i)
```

The new data loader behaves just like the previous one, except that it is more efficient and has some added functionality.

```{.python .input #synthetic-regression-data-concise-implementation-of-the-data-loader-3  n=4}
X, y = next(iter(data.train_dataloader()))
print('X shape:', X.shape, '\ny shape:', y.shape)
```

For instance, the data loader provided by the framework API 
supports the built-in `__len__` method, 
so we can query its length, 
i.e., the number of batches.

```{.python .input #synthetic-regression-data-concise-implementation-of-the-data-loader-4}
len(data.train_dataloader())
```

With 1000 training examples and a batch size of 32, we expect
$\lceil 1000 / 32 \rceil = 32$ batches: 31 full ones and a final
partial batch of 8 examples.
Note also that the built-in training loader *reshuffles* the examples at
the start of every epoch, just as our hand-rolled loader drew a fresh
random order on each call; exercise 8 of :numref:`sec_linear_scratch`
asks why this reshuffling matters.

:begin_tab:`jax`
The JAX loader reports 31 batches rather than 32.
This is because `get_tensorloader` passes `drop_remainder=True` when
training: the final partial batch of 8 examples is discarded.
We do this so that every training minibatch has an identical shape,
which keeps a `@jax.jit`-compiled training step from being recompiled
for the differently sized last batch. The price is that we drop a
handful of examples each epoch, which is negligible here. A loader that
keeps the partial batch would report 32.
:end_tab:

## Summary

Synthetic data lets us check the recovered parameters against the truth
we fixed: because we chose $\mathbf{w}^*$ and $b^*$ ourselves, we can see
after training whether the estimates agree, which makes such datasets
an early place to validate an algorithm against a compatible known-answer
problem. Passing this check does not establish robustness to misspecification
or real deployment data.
The `SyntheticRegressionData` class introduced here packages this
data-generating process as a `DataModule` subclass, separating *where
the batches come from* from *how a model consumes them*.
Along the way we implemented the same `get_dataloader` protocol twice:
a transparent hand-rolled iterator that is easy to read but loads
everything in memory and loops in Python, and a framework-native loader
that shuffles, prefetches, and parallelizes for us.
The hand-rolled version is there to teach; the framework version is what
we use from here on.


## Exercises

:begin_tab:`mxnet`
1. **Partial batches.** When the number of examples is not divisible by the
   batch size, `get_tensorloader` keeps the final partial batch. Find the
   argument of Gluon's `DataLoader` that discards it instead, and give one
   training scenario where you would enable it and one where you would not.
:end_tab:

:begin_tab:`pytorch`
1. **Partial batches.** When the number of examples is not divisible by the
   batch size, `get_tensorloader` keeps the final partial batch. Find the
   argument of `DataLoader` that drops it instead, and give one training
   scenario where you would enable it and one where you would not.
:end_tab:

:begin_tab:`tensorflow`
1. **Partial batches.** When the number of examples is not divisible by the
   batch size, `get_tensorloader` keeps the final partial batch. Find the
   argument of `Dataset.batch` that drops it instead, and give one training
   scenario where you would enable it and one where you would not.
:end_tab:

:begin_tab:`jax`
1. **Partial batches.** `get_tensorloader` passes `drop_remainder=train`,
   so the training loader drops its final partial batch while the
   validation loader keeps its partial batch of 8 examples. Predict what
   `len(data.val_dataloader())` returns. State what keeping that batch
   costs under `@jax.jit`, and whether you would drop it for validation as
   well.
:end_tab:

2. **Data beyond memory.** ● Suppose that we want to generate a huge
   dataset, where both the dimension of `w` and the number of examples
   `num_train + num_val` are large.
    1. Explain which statements in `SyntheticRegressionData.__init__` and in
       `get_dataloader` fail once $\mathbf{X}$ no longer fits in memory, and
       how each would have to change.
    1. Design an algorithm that visits the examples on disk in a fresh
       random order every epoch while (i) reading mostly long contiguous
       blocks rather than individual examples and (ii) never storing a
       permutation table with one entry per example. Pseudorandom
       permutations :cite:`Naor.Reingold.1999` address requirement (ii);
       state what addresses requirement (i).
1. [code] **On-the-fly generation.** Write a `DataModule` subclass whose
   `get_dataloader` draws a fresh minibatch of synthetic data on every
   `next` call instead of storing $\mathbf{X}$ and $\mathbf{y}$ in
   `__init__`. Confirm that two successive minibatches differ.

:begin_tab:`mxnet`
4. **Reproducibility.** Two instances of `SyntheticRegressionData` built
   with the same arguments hold different `X` and `y`. Make them identical
   by fixing the seed with `npx.random.seed`. State where the call must be
   placed relative to the draws of `X` and `eps`, and explain why seeding
   once at the top of a notebook does not make the second instance equal to
   the first.
:end_tab:

:begin_tab:`pytorch`
4. **Reproducibility.** Two instances of `SyntheticRegressionData` built
   with the same arguments hold different `X` and `y`. Make them identical
   by fixing the seed with `torch.manual_seed`. State where the call must be
   placed relative to the draws of `X` and `eps`, and explain why seeding
   once at the top of a notebook does not make the second instance equal to
   the first.
:end_tab:

:begin_tab:`tensorflow`
4. **Reproducibility.** Two instances of `SyntheticRegressionData` built
   with the same arguments hold different `X` and `y`. Make them identical
   by fixing the seed with `tf.random.set_seed`. State where the call must
   be placed relative to the draws of `X` and `eps`, and explain why seeding
   once at the top of a notebook does not make the second instance equal to
   the first.
:end_tab:

:begin_tab:`jax`
4. **Reproducibility.** `SyntheticRegressionData` splits its `key` into
   `key1` for `X` and `key2` for `eps`. Explain why two instances built with
   the same `key` hold identical data without any global seed. Then predict
   what `eps` would be, relative to the entries of `X`, if both draws used
   `key` itself instead of `key1` and `key2`, and check your prediction
   numerically.
:end_tab:

5. [code] **Recovery under noise.** Vary the noise standard deviation
   `noise` over $\{0.001, 0.01, 0.1, 0.5, 1.0\}$ with `num_train=1000` and
   fit $\hat{\mathbf{w}}$ on each dataset by least squares, using the
   normal equations of :numref:`sec_linear_regression` or `lstsq`.
    1. Before running, state how you expect
       $\|\hat{\mathbf{w}} - \mathbf{w}^*\|_2$ to scale with the noise
       level.
    1. Plot the error against the noise level on logarithmic axes.


:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/6662)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/6663)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/6664)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/17975)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §3.3]{.kicker}

Build a dataset with known generating parameters<br>**to isolate implementation and optimization errors**.
:::
:::

::: {.slide title="Why use synthetic data?"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
On **real** data, a poor result may reflect model misspecification, an
optimization or implementation error, or properties of the data.

**Synthetic** data specifies the generative law, so we can test whether a
compatible method recovers known parameters:

$$\mathbf{y} = \mathbf{X}\mathbf{w}^* + b^* + \boldsymbol{\epsilon},
  \qquad \boldsymbol{\epsilon}\sim\mathcal{N}(0,\sigma^2 I).$$

::: {.d2l-note}
Agreement with $\mathbf{w}^*,b^*$ supports the implementation on this controlled
problem. Systematic disagreement indicates an optimization or implementation
problem, provided the fitted model matches the generator.
:::
:::

::: {.col .narrow}
The dataset lives in a `DataModule` (the object-oriented-design section):
*where the batches come from*, kept separate from the model.

![](../img/mdl-linreg-oo-classes.svg)
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[Generating the data]{.dtitle}

[a DataModule that knows the ground truth]{.dsub}
:::
:::

::: {.slide title="Generate data in a DataModule" except="jax"}
[Generating the data]{.kicker}

Draw $\mathbf{X}\sim\mathcal{N}(0,1)$, apply the true line, add tiny noise,
all inside `__init__` ($n=2000$ examples, two features):

@synthetic-regression-data-generating-the-dataset-1

::: {.d2l-note}
`save_hyperparameters()` stores every argument as an attribute.
:::
:::

::: {.slide title="Generate data with explicit JAX keys" only="jax"}
[Generating the data]{.kicker}

JAX randomness is **functional**: thread a `key` in, `split` it for
independent $\mathbf{X}$ and $\boldsymbol{\epsilon}$ draws (same `key` in
→ same dataset out):

@synthetic-regression-data-generating-the-dataset-1
:::

::: {.slide title="Set and inspect the generating parameters"}
[Generating the data]{.kicker}

Instantiate with the true $\mathbf{w}^*=[2,-3.4]^\top$, $b^*=4.2$:

@synthetic-regression-data-generating-the-dataset-2

. . .

Each feature row is a vector in $\mathbb{R}^2$; each label is a scalar:

@synthetic-regression-data-generating-the-dataset-3

::: {.d2l-note .rule}
The next two sections compare the fitted parameters with $[2, -3.4]$ and
$4.2$, allowing for estimation error from the added noise.
:::
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Reading the data]{.dtitle}

[minibatches, by hand and by framework]{.dsub}
:::
:::

::: {.slide title="A minibatch sampler, by hand"}
[Reading the data]{.kicker}

The hand-written minibatch loader shuffles the indices (afresh on
every training pass), then `yield` `batch_size` rows at a time (one
batch is $32\times2$ features, $32\times1$ labels).

@synthetic-regression-data-reading-the-dataset-1

. . .

::: {.d2l-note .warn}
This implementation is transparent, but it keeps all data in memory, iterates
in single-threaded Python, and does not prefetch batches.
:::
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[The built-in loader]{.dtitle}

[the same interface with framework data-loading features]{.dsub}
:::
:::

::: {.slide title="Hand the work to the framework" except="jax"}
[The built-in loader]{.kicker}

The framework loader can shuffle, prefetch, and parallelize loading. First,
wrap the tensors:

@synthetic-regression-data-concise-implementation-of-the-data-loader-1

. . .

Then make `get_dataloader` use it for the training or validation split:

@synthetic-regression-data-concise-implementation-of-the-data-loader-2
:::

::: {.slide title="Hand the work to the framework" only="jax"}
[The built-in loader]{.kicker}

JAX ships no loader, so borrow TensorFlow's and unwrap it to NumPy. The relevant option is `drop_remainder=train`; `get_dataloader` then slices the train/val range and calls this.

@synthetic-regression-data-concise-implementation-of-the-data-loader-1
:::

::: {.slide title="Same interface, drop-in" except="jax"}
[The built-in loader]{.kicker}

The caller sees an identical protocol, one minibatch at a time:

@synthetic-regression-data-concise-implementation-of-the-data-loader-3

. . .

And it knows its own length, so `len(dl)` is the batches per epoch
($\lceil 1000/32\rceil = 32$: 31 full, one of 8):

@synthetic-regression-data-concise-implementation-of-the-data-loader-4
:::

::: {.slide title="Same interface, drop-in" only="jax"}
[The built-in loader]{.kicker}

The caller sees an identical protocol, one minibatch at a time:

@synthetic-regression-data-concise-implementation-of-the-data-loader-3

. . .

JAX reports **31**, not 32: `drop_remainder=True` discards the partial
last batch, so every `@jax.jit` step sees one shape.

@synthetic-regression-data-concise-implementation-of-the-data-loader-4

::: {.d2l-note .rule}
This setting omits 8 examples in each epoch for this dataset.
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

::: {.cols}
::: {.col}
- **Synthetic data** fixes the answer up front ($\mathbf{w}^*=[2,-3.4]$,
  $b^*=4.2$), enabling a controlled check of a compatible training method.
- A `DataModule` packages *where batches come from*, reusable across
  models.
:::

::: {.col}
- **Hand-rolled vs. built-in** loader: one protocol; the framework
  version shuffles, prefetches, parallelizes.
- **Watch the last batch:** a loader either keeps the partial final
  minibatch or drops it ($32$ vs. $31$ batches here).
:::
:::
:::
