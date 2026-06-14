```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# The Base Classification Model
:label:`sec_classification`

Every classification model in this book, from the linear softmax regressor we build next to the deep convolutional networks of later chapters, shares two common needs: a *validation step* that reports both loss and accuracy, and a default optimizer. Rather than re-implementing these in every subclass, we collect them once in a `Classifier` base class that extends the `d2l.Module` scaffold introduced in :numref:`sec_oo-design`. The payoff is the same one that motivated `Module` itself: a new classifier supplies only what is genuinely model-specific (its `forward` pass, and a `loss` if it is not plain cross-entropy), and inherits the training and evaluation machinery for free.

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

By default we use a stochastic gradient descent optimizer operating on minibatches, just as we did in the context of linear regression. `configure_optimizers` is a hook: `Trainer` calls it once at the start of training (see :numref:`sec_oo-design`), and it returns the optimizer object that `Trainer` then uses to update the parameters after each backward pass. We install the default here, on `d2l.Module` itself, so that no individual subclass has to repeat it. A subclass is free to override the method to switch optimizers (later chapters do exactly that), but plain SGD is the right default for the models in this chapter.

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

Before we implement the accuracy metric, it is worth asking why a classifier needs *two* numbers at all. A single forward pass produces a vector of scores $\mathbf{o}\in\mathbb{R}^q$, one per class, and from there the picture forks into two branches that read the *same* scores for very different purposes (:numref:`fig_mdl-clf-loss-accuracy`). On the training branch we turn the scores into probabilities with the softmax and read off the cross-entropy loss. This loss is a smooth function of the parameters, so gradient descent can minimize it; and it keeps rewarding the model for putting more probability on the correct class even after the decision is already right, nudging a confidence of $0.51$ toward $0.99$. On the evaluation branch we take the $\arg\max$ of the scores to a single hard decision $\hat{y}$, compare it with the label, and count the hit. This is the accuracy: the fraction of correct decisions, the number practitioners and benchmarks ultimately care about, but a *discrete* quantity whose gradient is zero almost everywhere, since a tiny change to the scores almost never flips which entry is largest.

So we report both, and for complementary reasons. Two models can reach identical accuracy while one is confidently right and the other barely so, and only the loss can tell them apart, which is why it, not accuracy, is what we optimize. Accuracy in turn measures the hard-decision quality that the loss only stands in for. When the two disagree (accuracy flat while the loss still drops, say) that is diagnostic information about optimization and calibration, not a bug.

![From model scores to a training loss and an evaluation accuracy. One forward pass produces the logits $\mathbf{o}$; the top branch softmaxes them into probabilities $\hat{\mathbf{y}}$ and reads off the differentiable cross-entropy loss that drives gradient descent, while the bottom branch takes the $\arg\max$ to a hard decision $\hat{y}$, compares it with the label $y$, and counts it for accuracy. The numbers shown are the exact softmax and cross-entropy of the logits $(1.0, 2.2, 0.3)$ for true class $y=1$.](../img/mdl-clf-loss-accuracy.svg)
:label:`fig_mdl-clf-loss-accuracy`

Taking the hard decision is what many applications require. Given the predicted probability distribution `y_hat`, we choose the class with the highest predicted probability whenever we must commit to one. Gmail, for instance, must file an email under "Primary", "Social", "Updates", "Forums", or "Spam": it might estimate probabilities internally, but at the end of the day it has to pick a single folder. A prediction that matches the label class `y` is correct, and accuracy is simply the fraction of predictions that are.

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
through Gluon's `Parameter` machinery, so it misses the bare `np.ndarray`
attributes that the from-scratch implementations in this book use.
We extend `d2l.Module` with a fallback `get_scratch_params` that
walks attributes recursively, and a `parameters` method that returns
Gluon's params when present and the scratch params otherwise. The
other frameworks do not need this, since PyTorch's `nn.Module`, TensorFlow
Keras, and JAX/Flax all expose parameters uniformly.
:end_tab:

```{.python .input #classification-accuracy-2  n=10}
%%tab mxnet

@d2l.add_to_class(d2l.Module)  #@save
def get_scratch_params(self):
    # collect_params() only finds Parameters declared via Gluon's Parameter
    # API. For from-scratch models that store weights as bare np.ndarrays, we
    # walk the object's attributes recursively and gather those instead.
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
    # Return the Gluon ParameterDict when the model uses Gluon layers; fall
    # back to the bare-array scan for from-scratch implementations.
    params = self.collect_params()
    return params if isinstance(params, dict) and len(
        params.keys()) else self.get_scratch_params()
```

## Summary

The `Classifier` class adds two things to `d2l.Module`: an overridden `validation_step` that logs *both* the loss and the accuracy, and a default `configure_optimizers` that returns a minibatch SGD optimizer. Because of this, every classification model in the rest of the book can subclass `Classifier` and supply only its `forward` pass (and a custom `loss`, where the default cross-entropy will not do), inheriting the whole training and evaluation loop. Accuracy itself is the fraction of examples whose predicted class, the $\arg\max$ of the score vector, matches the true label. It is a discrete metric and so cannot serve as a training objective, but it is almost always the number reported in benchmarks and the one the reader should watch alongside the loss.


## Exercises

1. Denote by $L_\textrm{v}$ the validation loss, and let $L_\textrm{v}^\textrm{q}$ be its quick and dirty estimate computed by the loss function averaging in this section. Lastly, denote by $l_\textrm{v}^\textrm{b}$ the loss on the last minibatch. Express $L_\textrm{v}$ in terms of $L_\textrm{v}^\textrm{q}$, $l_\textrm{v}^\textrm{b}$, and the sample and minibatch sizes.
1. Show that the quick and dirty estimate $L_\textrm{v}^\textrm{q}$ is unbiased. That is, show that $E[L_\textrm{v}] = E[L_\textrm{v}^\textrm{q}]$. Why would you still want to use $L_\textrm{v}$ instead?
1. Given a multiclass classification loss, denoting by $l(y,y')$ the penalty of estimating $y'$ when we see $y$ and given a probability $p(y \mid x)$, formulate the rule for an optimal selection of $y'$. Hint: express the expected loss, using $l$ and $p(y \mid x)$.
1. Suppose two classifiers $A$ and $B$ both achieve 90% accuracy on a ten-class test set, but on the examples they get right, $A$ assigns probability $0.91$ on average to the correct class while $B$ assigns only $0.51$. (i) Compute the average cross-entropy loss each incurs on those examples. (ii) Which classifier would you trust more in a safety-critical setting, and why does accuracy alone fail to separate them? (iii) Construct a simple monotone rescaling of the scores (a temperature) that sharpens $B$'s probabilities without changing any of its $\arg\max$ decisions, and argue why its accuracy is therefore unchanged.
1. Generalize `accuracy` to *top-$k$ accuracy*, which counts a prediction as correct when the true class is among the $k$ highest-scoring classes. (i) Modify the four-line implementation to take a `k` argument (hint: replace the single `argmax` with the indices of the $k$ largest scores). (ii) On a $q$-class problem, what is top-$q$ accuracy always equal to, and why? (iii) Why is top-5 accuracy a standard companion to top-1 on benchmarks with many fine-grained classes?

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
[Dive into Deep Learning · §4.3]{.kicker}

The base **classification** model<br>One forward pass, read two ways: a *loss* to train on and an *accuracy* to report.
:::
:::

::: {.slide title="One forward pass, two readings"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
A classifier scores the classes, then the picture **forks**:

- **train** on a smooth **loss** that gradient descent can minimize;
- **report** a hard **accuracy**, the number benchmarks care about.

::: {.d2l-note}
We collect both, once, in a `Classifier` base class so every model
in the book inherits them for free.
:::
:::

::: {.col .fig .big}
![One forward pass produces the logits $\mathbf{o}$; the top branch softmaxes them and reads the differentiable loss, the bottom branch takes the $\arg\max$ to a decision and counts it.](../img/mdl-clf-loss-accuracy.svg)
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[The `Classifier` base class]{.dtitle}

[what every model inherits, what each supplies]{.dsub}
:::
:::

::: {.slide title="Inherit the loop, supply the model"}
[The base class]{.kicker}

::: {.cols .vc}
::: {.col}
`Classifier` extends the `d2l.Module` scaffold from the regression
chapter, adding classification defaults.

- **Inherited:** a validation step (loss + accuracy) and a default
  optimizer.
- **Supplied by a subclass:** its `forward` pass, and a `loss` only
  if plain cross-entropy will not do.
:::

::: {.col .narrow}
::: {.d2l-note .rule}
Same payoff as `Module` itself: write the model-specific part once,
get the training and evaluation machinery for free.
:::
:::
:::
:::

::: {.slide title="Validation reports loss *and* accuracy" except="jax"}
[The base class]{.kicker}

The override logs two curves per validation batch, where regression
logged one:

@classification-the-classifier-class-1

::: {.d2l-note}
Averaging over `num_val_batches` is slightly off on a short last
batch; we ignore that to keep the code simple.
:::
:::

::: {.slide title="Validation under JAX: stateless, functional" only="jax"}
Flax modules carry no state, so the step threads `params`/`state` explicitly:

@classification-the-classifier-class-1
:::

::: {.slide title="A default optimizer, installed once"}
[The base class]{.kicker}

`configure_optimizers` is a hook the `Trainer` calls at startup. We
put plain minibatch **SGD** on `Module` itself, so no subclass repeats
it (later chapters override to switch optimizers):

@classification-the-classifier-class-2
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Accuracy]{.dtitle}

[the hard-decision metric, in four lines]{.dsub}
:::
:::

::: {.slide title="Why a classifier needs *two* numbers"}
[Scores, loss, decision]{.kicker}

::: {.cols .vc}
::: {.col}
The same logits $\mathbf{o}$ feed two branches with different jobs.

. . .

**Loss** (top) softmaxes to probabilities and is *differentiable*, so
it trains the model, and keeps rewarding confidence past the point the
decision is right.

. . .

**Accuracy** (bottom) is $\arg\max$ then compare: a *discrete* count
whose gradient is zero almost everywhere, so it cannot be optimized
directly.
:::

::: {.col .fig}
![Logits $(1.0, 2.2, 0.3)$: softmax $\to (0.21, 0.69, 0.10)$ and cross-entropy $\ell=0.37$ on top; $\arg\max=1$ matches $y=1$, one correct, on the bottom.](../img/mdl-clf-loss-accuracy.svg)
:::
:::
:::

::: {.slide title="Accuracy in four lines"}
[Scores, loss, decision]{.kicker}

`argmax` along the class axis, compare with the label element-wise,
average the 0/1 hits:

@classification-accuracy-1

::: {.d2l-note}
The `astype` matches dtypes before `==`, since the comparison is
type-sensitive. JAX adds `@jax.jit` and runs the forward pass from
`params`; the arithmetic is identical.
:::
:::

::: {.slide title="Finding the weights to score" only="mxnet"}
Gluon's `collect_params` misses bare-array weights, so MXNet adds a recursive fallback:

@classification-accuracy-2
:::

::: {.slide title="Report both, for complementary reasons"}
[Scores, loss, decision]{.kicker}

Two classifiers can hit the **same accuracy** while one is confidently
right and the other barely so.

. . .

Only the **loss** separates a correct-class probability of $0.51$ from
$0.99$, which is why it, not accuracy, is what we optimize.

. . .

::: {.d2l-note .rule}
When the two disagree (accuracy flat while loss still drops) that is a
diagnostic about optimization and calibration, **not a bug**.
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

::: {.cols}
::: {.col}
- `Classifier(d2l.Module)` adds a **loss + accuracy** validation step
  and a default **SGD** optimizer.
- A new model supplies only `forward` (and a custom `loss`), inheriting
  the whole loop.
:::

::: {.col}
- **Accuracy** = fraction whose $\arg\max$ matches the label:
  `argmax → == y → mean`.
- Discrete, so we **train on the loss** and **watch accuracy** beside
  it.
:::
:::
:::
