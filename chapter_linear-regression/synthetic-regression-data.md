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
When a method performs poorly on real data,
all three explanations remain on the table at once.
*Synthetic data* removes this ambiguity by construction.
If we know the data-generating process exactly
(the true weights $\mathbf{w}^*$, the true bias $b^*$,
and the noise distribution),
then any failure to recover those parameters
is an algorithm or implementation failure, full stop:
the data is provably learnable, because we built it that way.
This is what makes synthetic datasets the indispensable first test
for any new learning method.
We confirm that it solves a problem with a known answer
before we ever hand it a real one.

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
import tensorflow_datasets as tfds
```

## Generating the Dataset

For this example, we will work in low dimension
for succinctness.
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
        # Resolve the key at call time (the None idiom) rather than baking a
        # fixed PRNGKey into the signature; default stays deterministic.
        key = jax.random.PRNGKey(0) if key is None else key
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

Each row of `data.X` is a feature vector in $\mathbb{R}^2$ and each row of `data.y` is a scalar label. Let's have a look at the first entry.

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

To build some intuition, let's inspect the first minibatch of
data. Each minibatch of features provides us with both its size and the dimensionality of input features.
Likewise, our minibatch of labels will have a matching shape given by `batch_size`.

```{.python .input #synthetic-regression-data-reading-the-dataset-2}
X, y = next(iter(data.train_dataloader()))
print('X shape:', X.shape, '\ny shape:', y.shape)
```

Iterating over `data.train_dataloader()` yields distinct minibatches
until the dataset is exhausted (try it).
This hand-rolled loader is worth writing once,
because it shows exactly what happens under the hood,
but it pays for that transparency in three ways:
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
JAX is all about NumPy like API with device acceleration and the functional
transformations, so at least the current version doesn’t include data loading
methods. With other  libraries we already have great data loaders out there,
and JAX suggests using them instead. Here we will grab TensorFlow’s data loader,
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
@d2l.add_to_class(d2l.DataModule)  #@save
def get_tensorloader(self, tensors, train, indices=slice(0, None)):
    tensors = tuple(a[indices] for a in tensors)
    # Use Tensorflow Datasets & Dataloader. JAX or Flax do not provide
    # any dataloading functionality. `drop_remainder=train` keeps every
    # *training* minibatch the same shape, so a `@jax.jit`'d step
    # function compiles once per epoch instead of recompiling for the
    # smaller last batch (a common source of multi-minute slowdowns on
    # NLP datasets where the last batch is a different shape every time).
    shuffle_buffer = tensors[0].shape[0] if train else 1
    return tfds.as_numpy(
        tf.data.Dataset.from_tensor_slices(tensors).shuffle(
            buffer_size=shuffle_buffer
        ).batch(self.batch_size, drop_remainder=train))
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
You may notice that the JAX loader reports 31 batches rather than 32.
This is because `get_tensorloader` passes `drop_remainder=True` when
training: the final partial batch of 8 examples is discarded.
We do this so that every training minibatch has an identical shape,
which keeps a `@jax.jit`-compiled training step from being recompiled
for the differently sized last batch (a recompilation that can cost
minutes per epoch on larger datasets). The price is that we drop a
handful of examples each epoch, which is negligible here. The other
three frameworks keep the partial batch and so report 32.
:end_tab:

## Summary

Synthetic data closes the loop on learning: because we fixed
$\mathbf{w}^*$ and $b^*$ ourselves, we can check after training whether
the recovered parameters agree with the truth, which makes such datasets
the first place to validate any new algorithm.
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

1. When the number of examples is not divisible by the batch size, the loaders above keep the final partial batch (except in JAX, whose `drop_remainder=train` behavior is explained in the callout above). What does PyTorch's `drop_last` argument (and its TensorFlow counterpart, `batch(..., drop_remainder=...)`) do, and when would you want to enable it even outside JAX?
1. Suppose that we want to generate a huge dataset, where both the size of the parameter vector `w` and the number of examples `num_examples` are large.
    1. What happens if we cannot hold all data in memory?
    1. How would you shuffle the data if it is held on disk? Your task is to design an *efficient* algorithm that does not require too many random reads or writes. Hint: [pseudorandom permutation generators](https://en.wikipedia.org/wiki/Pseudorandom_permutation) allow you to design a reshuffle without the need to store the permutation table explicitly :cite:`Naor.Reingold.1999`. 
1. Implement a data generator that produces new data on the fly, every time the iterator is called. 
1. **(Reproducibility across frameworks.)** How would you design a random data
   generator that produces the *same* dataset every time it is called? In PyTorch
   and TensorFlow a single global call (`torch.manual_seed` or
   `tf.random.set_seed`) suffices. JAX takes a different stance: its PRNG is
   *functional*, with no global state, so randomness is threaded through an
   explicit `key`. Explain why passing `key=jax.random.PRNGKey(42)` to
   `SyntheticRegressionData` already makes the JAX version reproducible, and why
   re-using the *same* key for both $\mathbf{X}$ and $\boldsymbol{\epsilon}$
   (instead of splitting it) would be a bug.
1. **(Signal-to-noise and recovery.)** Vary the noise standard deviation `noise`
   over $\{0.001, 0.01, 0.1, 0.5, 1.0\}$. After fitting a linear model on each
   dataset (using the code from :numref:`sec_linear_scratch` or
   :numref:`sec_linear_concise`), how closely does the estimate
   $\hat{\mathbf{w}}$ match the true $\mathbf{w}^* = [2, -3.4]^\top$? Plot the
   error $\|\hat{\mathbf{w}} - \mathbf{w}^*\|_2$ as a function of $\sigma$. How do
   you expect it to scale with $\sigma$ and with the number of training examples,
   and does the experiment agree?


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

Synthetic regression **data**<br>Build a dataset whose answer you already know, so a failed fit can only be the *algorithm's* fault.
:::
:::

::: {.slide title="Why fabricate the data?"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
On **real** data, a poor result has three suspects at once: a wrong
model, a broken optimizer, or pathological data.

**Synthetic** data removes the third. We *choose* the generative law,
so the data is provably learnable:

$$\mathbf{y} = \mathbf{X}\mathbf{w}^* + b^* + \boldsymbol{\epsilon},
  \qquad \boldsymbol{\epsilon}\sim\mathcal{N}(0,\sigma^2 I).$$

::: {.d2l-note}
Recover $\mathbf{w}^*,b^*$ → the method works. Miss them → the bug is
yours, full stop.
:::
:::

::: {.col .narrow}
The dataset lives in a `DataModule` (§3.2): *where the batches come from*,
kept separate from the model.

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

::: {.slide title="A DataModule that builds itself" except="jax"}
[Generating the data]{.kicker}

Draw $\mathbf{X}\sim\mathcal{N}(0,1)$, apply the true line, add tiny noise,
all inside `__init__` ($n=2000$ examples, two features):

@synthetic-regression-data-generating-the-dataset-1

::: {.d2l-note}
`save_hyperparameters()` stores every argument as an attribute.
:::
:::

::: {.slide title="A DataModule that builds itself" only="jax"}
[Generating the data]{.kicker}

JAX randomness is **functional**: thread a `key` in, `split` it for
independent $\mathbf{X}$ and $\boldsymbol{\epsilon}$ draws (same `key` in
→ same dataset out):

@synthetic-regression-data-generating-the-dataset-1
:::

::: {.slide title="Fix the ground truth, then peek"}
[Generating the data]{.kicker}

Instantiate with the true $\mathbf{w}^*=[2,-3.4]^\top$, $b^*=4.2$, the
numbers we will try to recover later:

@synthetic-regression-data-generating-the-dataset-2

. . .

Each feature row is a vector in $\mathbb{R}^2$; each label is a scalar:

@synthetic-regression-data-generating-the-dataset-3
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

Roll the minibatch loader ourselves: permute the indices, then `yield` `batch_size` rows at a time (one batch is $32\times2$ features, $32\times1$ labels).

@synthetic-regression-data-reading-the-dataset-1

. . .

::: {.d2l-note .warn}
Transparent, but it loads everything in memory, loops in Python, and never prefetches.
:::
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[The built-in loader]{.dtitle}

[same interface, production speed]{.dsub}
:::
:::

::: {.slide title="Hand the work to the framework" except="jax"}
[The built-in loader]{.kicker}

The framework's loader shuffles, prefetches, and parallelizes for us.
Wrap the tensors once...

@synthetic-regression-data-concise-implementation-of-the-data-loader-1

. . .

...then rewire `get_dataloader` to use it (training vs. validation split):

@synthetic-regression-data-concise-implementation-of-the-data-loader-2
:::

::: {.slide title="Hand the work to the framework" only="jax"}
[The built-in loader]{.kicker}

JAX ships no loader, so borrow TensorFlow's and unwrap it to NumPy. The one twist is `drop_remainder=train`; `get_dataloader` then slices the train/val range and calls this.

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
last batch so every minibatch has one shape...

@synthetic-regression-data-concise-implementation-of-the-data-loader-4

::: {.d2l-note .rule}
...which keeps a `@jax.jit` step from recompiling for a smaller last
batch. We lose 8 examples per epoch, here negligible.
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

::: {.cols}
::: {.col}
- **Synthetic data** fixes $\mathbf{w}^*,b^*$ up front, so you can check
  recovery later, the first test for any new method.
- A `DataModule` packages *where batches come from*, reusable across
  models.
:::

::: {.col}
- **Hand-rolled vs. built-in** loader: one protocol; the framework
  version shuffles, prefetches, parallelizes.
- **Watch the framing:** JAX threads a PRNG `key` and drops the partial
  batch ($31$ vs. $32$).
:::
:::
:::
