```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# The Base Classification Model
:label:`sec_classification`

You may have noticed that the implementations from scratch and the concise implementation using framework functionality were quite similar in the case of regression. The same is true for classification. Since many models in this book deal with classification, it is worth adding functionalities to support this setting specifically. This section provides a base class for classification models to simplify future code.

```{.python .input #classification-the-base-classification-model}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon
npx.set_np()
```

```{.python .input #classification-the-base-classification-model}
%%tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input #classification-the-base-classification-model}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input #classification-the-base-classification-model}
%%tab jax
from d2l import jax as d2l
from functools import partial
from jax import numpy as jnp
import jax
import optax
```

## The `Classifier` Class

:begin_tab:`pytorch, mxnet, tensorflow`
We define the `Classifier` class below. In the `validation_step` we report both the loss value and the classification accuracy on a validation batch. We draw an update for every `num_val_batches` batches. This has the benefit of generating the averaged loss and accuracy on the whole validation data. These average numbers are not exactly correct if the last batch contains fewer examples, but we ignore this minor difference to keep the code simple.
:end_tab:

:begin_tab:`tensorflow`
The `_report_val` method mirrors `validation_step`
but accepts a precomputed `y_hat`,
so the compiled validation graph
(see :numref:`sec_linear_scratch`)
only needs to run the forward pass once.
:end_tab:

:begin_tab:`jax`
We define the `Classifier` class below. In the `validation_step` we report both the loss value and the classification accuracy on a validation batch. We draw an update for every `num_val_batches` batches. This has the benefit of generating the averaged loss and accuracy on the whole validation data. These average numbers are not exactly correct if the last batch contains fewer examples, but we ignore this minor difference to keep the code simple.

We also redefine the `training_step` method for JAX since all models that will
subclass `Classifier` later will have a loss that returns auxiliary data.
This auxiliary data can be used for models with batch normalization
(to be explained in :numref:`sec_batch_norm`), while in all other cases
we will make the loss also return a placeholder (empty dictionary) to
represent the auxiliary data.
:end_tab:

```{.python .input #classification-the-classifier-class-1}
%%tab pytorch, mxnet, tensorflow
class Classifier(d2l.Module):  #@save
    """The base class of classification models."""
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
        self.plot('acc', self.accuracy(Y_hat, batch[-1]), train=False)

    def _report_val(self, y_hat, batch):
        self.plot('loss', self.loss(y_hat, batch[-1]), train=False)
        self.plot('acc', self.accuracy(y_hat, batch[-1]), train=False)
```

```{.python .input #classification-the-classifier-class-1}
%%tab jax
class Classifier(d2l.Module):  #@save
    """The base class of classification models."""
    def training_step(self, params, batch, state):
        # Here value is a tuple since models with BatchNorm layers require
        # the loss to return auxiliary data
        value, grads = jax.value_and_grad(
            self.loss, has_aux=True)(params, batch[:-1], batch[-1], state)
        l, _ = value
        self.plot("loss", l, train=True)
        return value, grads

    def validation_step(self, params, batch, state):
        # Discard the second returned value. It is used for training models
        # with BatchNorm layers since loss also returns auxiliary data
        l, _ = self.loss(params, batch[:-1], batch[-1], state)
        self.plot('loss', l, train=False)
        self.plot('acc', self.accuracy(params, batch[:-1], batch[-1], state),
                  train=False)
```

By default we use a stochastic gradient descent optimizer, operating on minibatches, just as we did in the context of linear regression.

```{.python .input #classification-the-classifier-class-2}
%%tab mxnet
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    params = self.parameters()
    if isinstance(params, list):
        return d2l.SGD(params, self.lr)
    return gluon.Trainer(params, 'sgd', {'learning_rate': self.lr})
```

```{.python .input #classification-the-classifier-class-2}
%%tab pytorch
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), lr=self.lr)
```

```{.python .input #classification-the-classifier-class-2}
%%tab tensorflow
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return tf.keras.optimizers.SGD(float(self.lr))
```

```{.python .input #classification-the-classifier-class-2}
%%tab jax
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return optax.sgd(self.lr)
```

## Accuracy

Given the predicted probability distribution `y_hat`,
we typically choose the class with the highest predicted probability
whenever we must output a hard prediction.
Indeed, many applications require that we make a choice.
For instance, Gmail must categorize an email into "Primary", "Social", "Updates", "Forums", or "Spam".
It might estimate probabilities internally,
but at the end of the day it has to choose one among the classes.

When predictions are consistent with the label class `y`, they are correct.
The classification accuracy is the fraction of all predictions that are correct.
Although it can be difficult to optimize accuracy directly (it is not differentiable),
it is often the performance measure that we care about the most. It is often *the*
relevant quantity in benchmarks. As such, we will nearly always report it when training classifiers.

Accuracy is computed as follows.
First, if `y_hat` is a matrix,
we assume that the second dimension stores prediction scores for each class.
We use `argmax` to obtain the predicted class by the index for the largest entry in each row.
Then we compare the predicted class with the ground truth `y` elementwise.
Since the equality operator `==` is sensitive to data types,
we convert `y_hat`'s data type to match that of `y`.
The result is a tensor containing entries of 0 (false) and 1 (true).
Taking the sum yields the number of correct predictions.

```{.python .input #classification-accuracy-1  n=9}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(Classifier)  #@save
def accuracy(self, Y_hat, Y, averaged=True):
    """Compute the number of correct predictions."""
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    preds = d2l.astype(d2l.argmax(Y_hat, axis=1), Y.dtype)
    compare = d2l.astype(preds == d2l.reshape(Y, (-1,)), d2l.float32)
    return d2l.reduce_mean(compare) if averaged else compare
```

:begin_tab:`jax`
The JAX `accuracy` differs from the imperative version in a
few places. It takes `params` and `state` instead of a
precomputed `Y_hat` (Flax modules are stateless, so the
forward pass needs both), reaches into `state.batch_stats` to
support models with BatchNorm (a no-op for models without it),
and is decorated with `@jax.jit` for compiled execution. The
arithmetic that follows the forward pass is identical to the
other frameworks.
:end_tab:

```{.python .input #classification-accuracy-1  n=9}
%%tab jax
@d2l.add_to_class(Classifier)  #@save
@partial(jax.jit, static_argnums=(0, 5))
def accuracy(self, params, X, Y, state, averaged=True):
    """Compute the number of correct predictions."""
    Y_hat = state.apply_fn({'params': params,
                            'batch_stats': state.batch_stats},  # BatchNorm Only
                           *X)
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    preds = d2l.astype(d2l.argmax(Y_hat, axis=1), Y.dtype)
    compare = d2l.astype(preds == d2l.reshape(Y, (-1,)), d2l.float32)
    return d2l.reduce_mean(compare) if averaged else compare
```

:begin_tab:`mxnet`
MXNet's `gluon.Block.collect_params` only finds parameters declared
through Gluon's `Parameter` machinery — it misses bare `np.ndarray`
attributes that the from-scratch implementations in this book use.
We extend `d2l.Module` with a fallback `get_scratch_params` that
walks attributes recursively, and a `parameters` method that returns
Gluon's params when present and the scratch params otherwise. The
other frameworks don't need this — PyTorch's `nn.Module`, TensorFlow
Keras, and JAX/Flax all expose parameters uniformly.
:end_tab:

```{.python .input #classification-accuracy-2  n=10}
%%tab mxnet

@d2l.add_to_class(d2l.Module)  #@save
def get_scratch_params(self):
    params = []
    for attr in dir(self):
        a = getattr(self, attr)
        if isinstance(a, np.ndarray):
            params.append(a)
        if isinstance(a, d2l.Module):
            params.extend(a.get_scratch_params())
    return params

@d2l.add_to_class(d2l.Module)  #@save
def parameters(self):
    params = self.collect_params()
    return params if isinstance(params, dict) and len(
        params.keys()) else self.get_scratch_params()
```

## Summary

Classification is a sufficiently common problem that it warrants its own convenience functions. Of central importance in classification is the *accuracy* of the classifier. Note that while we often care primarily about accuracy, we train classifiers to optimize a variety of other objectives for statistical and computational reasons. However, regardless of which loss function was minimized during training, it is useful to have a convenience method for assessing the accuracy of our classifier empirically. 


## Exercises

1. Denote by $L_\textrm{v}$ the validation loss, and let $L_\textrm{v}^\textrm{q}$ be its quick and dirty estimate computed by the loss function averaging in this section. Lastly, denote by $l_\textrm{v}^\textrm{b}$ the loss on the last minibatch. Express $L_\textrm{v}$ in terms of $L_\textrm{v}^\textrm{q}$, $l_\textrm{v}^\textrm{b}$, and the sample and minibatch sizes.
1. Show that the quick and dirty estimate $L_\textrm{v}^\textrm{q}$ is unbiased. That is, show that $E[L_\textrm{v}] = E[L_\textrm{v}^\textrm{q}]$. Why would you still want to use $L_\textrm{v}$ instead?
1. Given a multiclass classification loss, denoting by $l(y,y')$ the penalty of estimating $y'$ when we see $y$ and given a probability $p(y \mid x)$, formulate the rule for an optimal selection of $y'$. Hint: express the expected loss, using $l$ and $p(y \mid x)$.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/6808)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/6809)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/6810)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17981)
:end_tab:

<!-- slides -->

::: {.slide title="The shared classifier base"}
A small `Classifier` base class that every classification model
in the book inherits from. Same role as `d2l.Module` for
regression — but with classification-specific defaults:

- A **validation step** that reports loss **and** accuracy.
- An **accuracy** helper that compares the argmax of the predicted
  scores to the true labels.

Subclasses just supply `forward` (and a custom `loss` if not
plain cross-entropy).
:::

::: {.slide title="Scores, probabilities, decisions"}
Classifiers usually produce a vector of scores
$\mathbf{o}\in\mathbb{R}^q$. The training loss may turn
scores into probabilities, but the deployed decision is often
just

$$\hat{y}=\arg\max_j o_j.$$

Keep the roles separate:

- **scores/logits:** differentiable quantities the model outputs;
- **loss:** smooth training signal, e.g. cross-entropy;
- **accuracy:** discrete evaluation metric after taking argmax.

Accuracy is what many benchmarks report, but it is not a useful
gradient: one tiny score change usually leaves argmax unchanged.
:::

::: {.slide title="Base classifier imports"}
@classification-the-base-classification-model
:::

::: {.slide title="The `Classifier` class"}
@classification-the-classifier-class-1
:::

::: {.slide title="Default optimizer hook"}
A default `configure_optimizers` on `Module` so subclasses don't
have to write it:

@classification-the-classifier-class-2
:::

::: {.slide title="Accuracy"}
Take the **argmax** along the class axis, compare with the true
label element-wise, and average. The result is the fraction of
correctly-classified examples in the batch:

@classification-accuracy-1

The validation step then reports both the loss (lower is better)
and accuracy (higher is better) every epoch.
:::

::: {.slide title="Why report both loss and accuracy?"}
Two models can have the same accuracy but different confidence.
Cross-entropy still notices whether the correct class received
probability 0.51 or 0.99.

Use both during training:

- **loss** detects calibration and optimization progress;
- **accuracy** tracks the hard decision quality students and
  benchmarks usually care about;
- disagreement between them is diagnostic, not a bug.
:::

::: {.slide title="Recap"}
- `Classifier(d2l.Module)` adds **accuracy reporting** to the
  base scaffold from the regression chapter.
- One line for accuracy: `argmax → ==y → mean`.
- The same training loop now drives every classification model
  we'll build through the rest of the book.
:::
