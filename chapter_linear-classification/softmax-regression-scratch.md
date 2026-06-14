```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Softmax Regression Implementation from Scratch
:label:`sec_softmax_scratch`

Because softmax regression is so fundamental,
we believe that you ought to know
how to implement it yourself.
Here, we limit ourselves to defining the
softmax-specific aspects of the model
and reuse the other components
from our linear regression section,
including the training loop.

```{.python .input #softmax-regression-scratch-softmax-regression-implementation-from-scratch}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon
npx.set_np()
```

```{.python .input #softmax-regression-scratch-softmax-regression-implementation-from-scratch}
%%tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input #softmax-regression-scratch-softmax-regression-implementation-from-scratch}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input #softmax-regression-scratch-softmax-regression-implementation-from-scratch}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
from functools import partial
```

## The Softmax

Let's begin with the most important part:
the mapping from scalars to probabilities.
For a refresher, recall the operation of the sum operator
along specific dimensions in a tensor,
as discussed in :numref:`subsec_lin-alg-reduction`
and :numref:`subsec_lin-alg-non-reduction`.
Given a matrix `X` we can sum over all elements (by default) or only
over elements in the same axis.
The `axis` variable lets us compute row and column sums:

```{.python .input #softmax-regression-scratch-the-softmax-1}
X = d2l.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
d2l.reduce_sum(X, 0, keepdims=True), d2l.reduce_sum(X, 1, keepdims=True)
```

Computing the softmax requires three steps:
(i) exponentiation of each term;
(ii) a sum over each row to compute the normalization constant for each example;
(iii) division of each row by its normalization constant,
ensuring that the result sums to 1:


$$\mathrm{softmax}(\mathbf{X})_{ij} = \frac{\exp(\mathbf{X}_{ij})}{\sum_k \exp(\mathbf{X}_{ik})}.$$


The (logarithm of the) denominator
is called the (log) *partition function*.
It was introduced in [statistical physics](https://en.wikipedia.org/wiki/Partition_function_(statistical_mechanics))
to sum over all possible states in a thermodynamic ensemble.
The implementation is straightforward:

```{.python .input #softmax-regression-scratch-the-softmax-2}
def softmax(X):
    X_exp = d2l.exp(X)
    partition = d2l.reduce_sum(X_exp, 1, keepdims=True)
    return X_exp / partition  # The broadcasting mechanism is applied here
```

For any input `X`, we turn each element
into a nonnegative number.
Each row sums up to 1,
as is required for a probability. Caution: the code above is *not* robust against very large or very small arguments. While it is sufficient to illustrate what is happening, you should *not* use this code verbatim for any serious purpose. Deep learning frameworks have such protections built in and we will be using the built-in softmax going forward.

To see the failure rather than just assert it, feed in a logit that is large
on the scale of $\exp$. A score of $1000$ overflows `exp` to infinity in
float32, so the naive ratio becomes $\infty/\infty$, which evaluates to `NaN`.
The framework's `softmax` subtracts the per-row maximum before exponentiating
(the log-sum-exp trick of :numref:`subsec_softmax-implementation-revisited`) and
returns a finite distribution on exactly the same input:

```{.python .input #softmax-regression-scratch-the-softmax-overflow}
%%tab pytorch
z = torch.tensor([1000., 0., 0.])
naive = torch.exp(z) / torch.exp(z).sum()  # exp(1000) overflows -> nan
stable = torch.softmax(z, dim=0)           # built-in uses the log-sum-exp trick
naive, stable
```

```{.python .input #softmax-regression-scratch-the-softmax-overflow}
%%tab mxnet
z = np.array([1000., 0., 0.])
naive = np.exp(z) / np.exp(z).sum()        # exp(1000) overflows -> nan
stable = npx.softmax(z, axis=0)            # built-in uses the log-sum-exp trick
naive, stable
```

```{.python .input #softmax-regression-scratch-the-softmax-overflow}
%%tab tensorflow
z = tf.constant([1000., 0., 0.])
naive = tf.exp(z) / tf.reduce_sum(tf.exp(z))  # exp(1000) overflows -> nan
stable = tf.nn.softmax(z, axis=0)             # built-in uses the log-sum-exp trick
naive, stable
```

```{.python .input #softmax-regression-scratch-the-softmax-overflow}
%%tab jax
z = jnp.array([1000., 0., 0.])
naive = jnp.exp(z) / jnp.exp(z).sum()      # exp(1000) overflows -> nan
stable = jax.nn.softmax(z, axis=0)         # built-in uses the log-sum-exp trick
naive, stable
```

```{.python .input #softmax-regression-scratch-the-softmax-3}
%%tab mxnet
X = d2l.rand(2, 5)
X_prob = softmax(X)
X_prob, d2l.reduce_sum(X_prob, 1)
```

```{.python .input #softmax-regression-scratch-the-softmax-3}
%%tab tensorflow, pytorch
X = d2l.rand((2, 5))
X_prob = softmax(X)
X_prob, d2l.reduce_sum(X_prob, 1)
```

```{.python .input #softmax-regression-scratch-the-softmax-3}
%%tab jax
X = jax.random.uniform(d2l.get_key(), (2, 5))
X_prob = softmax(X)
X_prob, d2l.reduce_sum(X_prob, 1)
```

## The Model

We now have everything that we need
to implement the softmax regression model.
As in our linear regression example,
each instance will be represented
by a fixed-length vector.
Since the raw data here consists
of $28 \times 28$ pixel images,
we flatten each image,
treating them as vectors of length 784.
In later chapters, we will introduce
convolutional neural networks,
which exploit the spatial structure
in a more satisfying way.


In softmax regression,
the number of outputs from our network
should be equal to the number of classes.
Since our dataset has 10 classes,
our network has an output dimension of 10.
Consequently, our weights constitute a $784 \times 10$ matrix
plus a $1 \times 10$ row vector for the biases.
As with linear regression,
we initialize the weights `W`
with Gaussian noise.
The biases are initialized as zeros.

```{.python .input #softmax-regression-scratch-the-model-1}
%%tab mxnet
class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = np.random.normal(0, sigma, (num_inputs, num_outputs))
        self.b = np.zeros(num_outputs)
        self.W.attach_grad()
        self.b.attach_grad()

    def collect_params(self):
        return [self.W, self.b]
```

```{.python .input #softmax-regression-scratch-the-model-1}
%%tab pytorch
class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = torch.normal(0, sigma, size=(num_inputs, num_outputs),
                              requires_grad=True)
        self.b = torch.zeros(num_outputs, requires_grad=True)

    def parameters(self):
        return [self.W, self.b]
```

```{.python .input #softmax-regression-scratch-the-model-1}
%%tab tensorflow
class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = tf.random.normal((num_inputs, num_outputs), 0, sigma)
        self.b = tf.zeros(num_outputs)
        self.W = tf.Variable(self.W)
        self.b = tf.Variable(self.b)
```

```{.python .input #softmax-regression-scratch-the-model-1}
%%tab jax
class SoftmaxRegressionScratch(d2l.Classifier):
    num_inputs: int
    num_outputs: int
    lr: float
    sigma: float = 0.01

    def setup(self):
        self.W = self.param('W', nn.initializers.normal(self.sigma),
                            (self.num_inputs, self.num_outputs))
        self.b = self.param('b', nn.initializers.zeros, self.num_outputs)
```

The code below defines how the network
maps each input to an output.
Note that we flatten each $28 \times 28$ pixel image in the batch
into a vector using `reshape`
before passing the data through our model.

```{.python .input #softmax-regression-scratch-the-model-2}
@d2l.add_to_class(SoftmaxRegressionScratch)
def forward(self, X):
    X = d2l.reshape(X, (-1, self.W.shape[0]))
    return softmax(d2l.matmul(X, self.W) + self.b)
```

## The Cross-Entropy Loss

Next we need to implement the cross-entropy loss function
(introduced in :numref:`subsec_softmax-regression-loss-func`).
This may be the most common loss function
in all of deep learning.
Recall from :numref:`subsec_softmax-regression-loss-func` that minimizing cross-entropy is equivalent to maximizing the log-likelihood of the correct labels under our categorical model. It is the natural loss for classification.
At the moment, applications of deep learning
easily cast as classification problems
far outnumber those better treated as regression problems.

Recall that cross-entropy takes the negative log-likelihood
of the predicted probability assigned to the true label.
For efficiency we avoid Python for-loops and use indexing instead.
In particular, the one-hot encoding in $\mathbf{y}$
allows us to select the matching terms in $\hat{\mathbf{y}}$.

To see this in action we create sample data `y_hat`
with 2 examples of predicted probabilities over 3 classes and their corresponding labels `y`.
The correct labels are $0$ and $2$ respectively (i.e., the first and third class).
Using `y` as the indices of the probabilities in `y_hat`,
we can pick out terms efficiently.

```{.python .input #softmax-regression-scratch-the-cross-entropy-loss-1}
%%tab mxnet, pytorch, jax
y = d2l.tensor([0, 2])
y_hat = d2l.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]
```

```{.python .input #softmax-regression-scratch-the-cross-entropy-loss-1}
%%tab tensorflow
y_hat = tf.constant([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = tf.constant([0, 2])
tf.gather(y_hat, y, batch_dims=1)
```

:begin_tab:`pytorch, mxnet, tensorflow`
Now we can implement the cross-entropy loss function by averaging over the logarithms of the selected probabilities.
:end_tab:

:begin_tab:`jax`
Now we can implement the cross-entropy loss function by averaging over the logarithms of the selected probabilities.

Note that to make use of `jax.jit` to speed up JAX implementations, and
to make sure `loss` is a pure function, the `cross_entropy` function is re-defined
inside the `loss` to avoid usage of any global variables or functions
which may render the `loss` function impure.
We refer interested readers to the [JAX documentation](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions) on `jax.jit` and pure functions.
:end_tab:

```{.python .input #softmax-regression-scratch-the-cross-entropy-loss-2}
%%tab pytorch
def cross_entropy(y_hat, y):  #@save
    # Tiny clip to keep log finite when softmax outputs underflow to 0.
    p = y_hat[list(range(len(y_hat))), y].clamp(min=1e-12)
    return -d2l.reduce_mean(d2l.log(p))

cross_entropy(y_hat, y)
```

```{.python .input #softmax-regression-scratch-the-cross-entropy-loss-2}
%%tab mxnet
def cross_entropy(y_hat, y):  #@save
    # Tiny clip to keep log finite when softmax outputs underflow to 0.
    p = y_hat[list(range(len(y_hat))), y].clip(min=1e-12)
    return -d2l.reduce_mean(d2l.log(p))

cross_entropy(y_hat, y)
```

```{.python .input #softmax-regression-scratch-the-cross-entropy-loss-2}
%%tab jax
def cross_entropy(y_hat, y):  #@save
    # Tiny clip to keep log finite when softmax outputs underflow to 0.
    p = jnp.clip(jnp.take_along_axis(y_hat, jnp.expand_dims(y, -1),
                                     axis=1).squeeze(-1), min=1e-12)
    return -d2l.reduce_mean(d2l.log(p))

cross_entropy(y_hat, y)
```

```{.python .input #softmax-regression-scratch-the-cross-entropy-loss-2}
%%tab tensorflow
def cross_entropy(y_hat, y):  #@save
    p = tf.gather(y_hat, y, batch_dims=1)
    # Tiny clip to keep log finite when softmax outputs underflow to 0.
    return -tf.reduce_mean(tf.math.log(tf.maximum(p, 1e-12)))

cross_entropy(y_hat, y)
```

Note that we clip $\hat{y}$ away from zero before taking $\log$. Without the clip, $\log(\hat{y})$ produces $-\infty$ (and downstream NaNs) whenever the softmax assigns probability exactly zero to the correct class. Production code typically uses a log-softmax layer that fuses the softmax and log into a single numerically stable operation; the explicit clamp here is the minimal change that keeps the scratch implementation usable as a teaching example without changing its mathematical form. The proper fix, fusing softmax and cross-entropy via the log-sum-exp trick, is derived in :numref:`subsec_softmax-implementation-revisited`.

```{.python .input #softmax-regression-scratch-the-cross-entropy-loss-3}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(SoftmaxRegressionScratch)
def loss(self, y_hat, y):
    return cross_entropy(y_hat, y)
```

```{.python .input #softmax-regression-scratch-the-cross-entropy-loss-3}
%%tab jax
@d2l.add_to_class(SoftmaxRegressionScratch)
@partial(jax.jit, static_argnums=(0))
def loss(self, params, X, y, state):
    def cross_entropy(y_hat, y):
        # Tiny clip to keep log finite when softmax outputs underflow to 0.
        p = jnp.clip(jnp.take_along_axis(y_hat, jnp.expand_dims(y, -1),
                                         axis=1).squeeze(-1), min=1e-12)
        return -d2l.reduce_mean(d2l.log(p))
    y_hat = state.apply_fn({'params': params}, *X)
    # The returned empty dictionary is a placeholder for auxiliary data,
    # which will be used later (e.g., for batch norm)
    return cross_entropy(y_hat, y), {}
```

## Training

We reuse the `fit` method defined in :numref:`sec_linear_scratch` to train the model with 10 epochs.
Note that the number of epochs (`max_epochs`),
the minibatch size (`batch_size`),
and learning rate (`lr`)
are adjustable hyperparameters.
That means that while these values are not
learned during our primary training loop,
they still influence the performance
of our model, both vis-à-vis training
and generalization performance.
In practice you will want to choose these values
based on the *validation* split of the data
and then, ultimately, to evaluate your final model
on the *test* split.
As discussed in :numref:`subsec_generalization-model-selection`,
we will regard the test data of Fashion-MNIST
as the validation set, thus
reporting validation loss and validation accuracy
on this split.

```{.python .input #softmax-regression-scratch-training}
data = d2l.FashionMNIST(batch_size=256)
model = SoftmaxRegressionScratch(num_inputs=784, num_outputs=10, lr=0.1)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

## Prediction

Now that training is complete,
our model is ready to classify some images.

```{.python .input #softmax-regression-scratch-prediction-1}
%%tab pytorch
X, y = next(iter(data.val_dataloader()))
with torch.no_grad():
    preds = d2l.argmax(model(X), axis=1)
preds.shape
```

```{.python .input #softmax-regression-scratch-prediction-1}
%%tab tensorflow
X, y = next(iter(data.val_dataloader()))
preds = d2l.argmax(model(X), axis=1)
preds.shape
```

```{.python .input #softmax-regression-scratch-prediction-1}
%%tab jax
X, y = next(iter(data.val_dataloader()))
preds = d2l.argmax(model.apply({'params': trainer.state.params}, X), axis=1)
preds.shape
```

```{.python .input #softmax-regression-scratch-prediction-1}
%%tab mxnet
X, y = next(iter(data.val_dataloader()))
preds = d2l.argmax(model(X), axis=1)
preds.shape
```

How well do we do overall? We sweep the whole validation set and average the
per-example correct/incorrect flags returned by `accuracy`:

```{.python .input #softmax-regression-scratch-prediction-accuracy}
%%tab pytorch
correct = []
for X_i, y_i in data.val_dataloader():
    with torch.no_grad():
        correct.append(model.accuracy(model(X_i), y_i, averaged=False))
print(f'Test accuracy: {torch.cat(correct).mean():.3f}')
```

The overall test accuracy is approximately 83%, consistent with the training
curve: the ceiling of a linear model on Fashion-MNIST. We are more interested in
the images we label *incorrectly*. We visualize them by
comparing their actual labels
(first line of text output)
with the predictions from the model
(second line of text output).

```{.python .input #softmax-regression-scratch-prediction-2}
wrong = d2l.astype(preds, y.dtype) != y
X, y, preds = X[wrong], y[wrong], preds[wrong]
labels = [a+'\n'+b for a, b in zip(
    data.text_labels(y), data.text_labels(preds))]
data.visualize([X, y], labels=labels)
```

## Summary and Discussion

In this section we built softmax regression entirely from scratch: the softmax
operation, the cross-entropy loss, parameter initialization, the forward pass, and
training on Fashion-MNIST. Breaking each piece open by hand is the purpose. Once
you have seen these five moving parts separately, the one-liner in
:numref:`sec_softmax_concise` is not magic but notation.

**What the training curve tells you.** After 10 epochs with minibatch SGD the
model converges to roughly 83% validation accuracy. That ceiling is not a
hyperparameter problem; it is the limit of linear separability on Fashion-MNIST.
The ten classes are not linearly separable in pixel space (shirts and pullovers
look nearly identical to a linear model). The misclassification visualization at
the end of the section makes this concrete. Replacing the flat linear layer with
even a single hidden layer (Chapter 5) pushes past it.

**Why the clip is only a band-aid.** Our `cross_entropy` clips the softmax output
away from zero before taking the log. This prevents the worst NaN failures, but
the naive `softmax` function itself can overflow for large logits (`exp(100)` is
infinity in float32). The right fix, subtracting $\max_k o_k$ before
exponentiating and then fusing softmax and log into a single numerically stable
operation, is derived in :numref:`subsec_softmax-implementation-revisited`. The
concise implementation applies that fix automatically; always use the framework's
built-in cross-entropy when you are not explicitly studying the internals.

## Exercises

1. In this section, we directly implemented the softmax function based on the mathematical definition of the softmax operation. As discussed in :numref:`sec_softmax` this can cause numerical instabilities.
    1. Test whether `softmax` still works correctly if an input has a value of $100$.
    1. Test whether `softmax` still works correctly if the largest of all inputs is smaller than $-100$.
    1. Implement a fix by looking at the value relative to the largest entry in the argument.
1. Implement a `cross_entropy` function that follows the definition of the cross-entropy loss function $-\sum_i y_i \log \hat{y}_i$.
    1. Try it out in the code example of this section.
    1. Why do you think it runs more slowly?
    1. Should you use it? When would it make sense to?
    1. What do you need to be careful of? Hint: consider the domain of the logarithm.
1. Is it always a good idea to return the most likely label? For example, would you do this for medical diagnosis? How would you try to address this?
1. Assume that we want to use softmax regression to predict the next word based on some features. What are some problems that might arise from a large vocabulary?
1. Experiment with the hyperparameters of the code in this section. In particular:
    1. Plot how the validation loss changes as you change the learning rate.
    1. Do the validation and training loss change as you change the minibatch size? How large or small do you need to go before you see an effect?


:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/50)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/51)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/225)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/17982)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §4.4]{.kicker}

Softmax regression **from scratch**<br>The whole classifier, opened up: the softmax, the cross-entropy loss, and the training loop, each built by hand.
:::
:::

::: {.slide title="The same recipe, two new pieces"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
Linear regression mapped inputs to **one** number. A classifier maps
them to a **distribution over classes**. Two new parts do that:

1. **Softmax** turns raw scores (logits) into probabilities.
2. **Cross-entropy** is the loss that scores a distribution.

::: {.d2l-note}
Everything else, the `Module` / `Trainer` scaffold, is **reused** from
the regression chapter; `Classifier` just adds accuracy reporting.
:::
:::

::: {.col .fig .big}
![](../img/mdl-clf-loss-accuracy.svg)
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[The Softmax]{.dtitle}

[from scores to a probability distribution]{.dsub}
:::
:::

::: {.slide title="First, a reminder: sums along an axis"}
[The Softmax]{.kicker}

::: {.cols .vc}
::: {.col}
Softmax normalizes each row, so we need a **per-row** sum. `axis=1`
collapses the columns; `keepdims` holds the shape for broadcasting:

@softmax-regression-scratch-the-softmax-1
:::

::: {.col .narrow}
::: {.d2l-note}
`axis=0` sums **down** columns, `axis=1` sums **across** rows.
`keepdims=True` keeps a length-1 axis so the result still broadcasts
against `X`.
:::
:::
:::
:::

::: {.slide title="Softmax: exponentiate, sum, divide"}
[The Softmax]{.kicker}

$$\mathrm{softmax}(\mathbf{X})_{ij}
  = \frac{\exp(\mathbf{X}_{ij})}{\sum_k \exp(\mathbf{X}_{ik})}.$$

Three steps: exponentiate every score, sum across the class axis, divide
each row by its total:

@softmax-regression-scratch-the-softmax-2

::: {.d2l-note .warn}
Naive `exp` **overflows** for large logits. Fine for teaching; never use
it in production. The stable fix arrives in §4.5.
:::
:::

::: {.slide title="The output is a real distribution"}
[The Softmax]{.kicker}

Feed in any matrix: every entry becomes non-negative and **each row sums
to 1**, exactly what a probability distribution over classes requires:

@softmax-regression-scratch-the-softmax-3
:::

::: {.slide title="Why it is only a teaching version"}
[The Softmax]{.kicker}

Watch the warning bite. A single logit of $1000$ sends $\exp$ to infinity
in float32, the ratio is $\infty/\infty$, and the whole row turns to `NaN`.
The framework's `softmax` shifts by the row maximum first and stays finite
on the identical input:

@softmax-regression-scratch-the-softmax-overflow

::: {.d2l-note .warn}
One `NaN` poisons every downstream gradient. The stable fix, fusing softmax
and log via log-sum-exp, arrives in §4.5.
:::
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[The Model]{.dtitle}

[one linear layer, ten logits]{.dsub}
:::
:::

::: {.slide title="Parameters: a 784×10 weight matrix"}
[The Model]{.kicker}

Each $28\times28$ image flattens to a length-784 vector; with 10 classes
the weights are a $784\times10$ matrix `W` plus a length-10 bias `b`.
Initialize `W` with Gaussian noise, `b` with zeros:

@softmax-regression-scratch-the-model-1
:::

::: {.slide title="Forward pass: flatten → linear → softmax"}
[The Model]{.kicker}

The model is one expression: reshape the batch to rows of 784, apply the
affine map $\mathbf{X}\mathbf{W}+\mathbf{b}$, then softmax into
per-class probabilities:

@softmax-regression-scratch-the-model-2
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[Cross-Entropy Loss]{.dtitle}

[the loss for a predicted distribution]{.dsub}
:::
:::

::: {.slide title="The loss for distributions"}
[Cross-Entropy Loss]{.kicker}

::: {.cols .vc}
::: {.col}
For an integer label `y`, the loss on one example is just the negative
log-probability the model assigned to the **correct** class:

$$\ell = -\log \hat{y}_{y}.$$

We pick out $\hat{y}_y$ for every row with fancy indexing, no Python
loop. True labels `0` and `2` select the highlighted probabilities:

@softmax-regression-scratch-the-cross-entropy-loss-1
:::

::: {.col .narrow}
::: {.d2l-note}
Minimizing cross-entropy maximizes the **log-likelihood** of the correct
labels. It keeps rewarding higher confidence, nudging $0.51\to0.99$ even
after the decision is already right.
:::
:::
:::
:::

::: {.slide title="Average over the batch"}
[Cross-Entropy Loss]{.kicker}

Take the negative log of each selected probability, then average. A tiny
clip keeps the log finite when a probability underflows to 0:

@softmax-regression-scratch-the-cross-entropy-loss-2
:::

::: {.slide title="Register it as the loss"}
[Cross-Entropy Loss]{.kicker}

Attach `cross_entropy` as the model's `loss`, and every reused training
utility now knows how to optimize this classifier:

@softmax-regression-scratch-the-cross-entropy-loss-3
:::

::: {.slide}
::: {.divider}
[04]{.dnum}

[Train & Predict]{.dtitle}

[fit on Fashion-MNIST, then inspect mistakes]{.dsub}
:::
:::

::: {.slide title="Train on Fashion-MNIST"}
[Training]{.kicker}

Ten epochs of minibatch SGD on Fashion-MNIST. The inherited `Classifier`
runs the validation loop and plots train/validation loss alongside
validation accuracy, no extra code:

@softmax-regression-scratch-training
:::

::: {.slide title="Predict on a fresh batch"}
[Prediction]{.kicker}

Take the argmax of the model's outputs over a fresh validation batch,
one predicted class per image:

@softmax-regression-scratch-prediction-1
:::

::: {.slide title="Look at the mistakes"}
[Prediction]{.kicker}

The interesting cases are the **errors**. Tile the misclassified images,
each captioned `true / predicted`:

@softmax-regression-scratch-prediction-2
:::

::: {.slide title="How accurate, overall?" only="pytorch"}
[Prediction]{.kicker}

Sweep the whole validation set and average the per-example correct flags:

@softmax-regression-scratch-prediction-accuracy

::: {.d2l-note}
About **83%**, and that is the *ceiling* of a linear model on
Fashion-MNIST, not a tuning problem. The next slide shows why.
:::
:::

::: {.slide title="Why a linear model caps out"}
[The ceiling]{.kicker}

::: {.cols .vc}
::: {.col}
A linear classifier draws **straight** decision boundaries. In pixel
space shirts and pullovers overlap, and no hyperplane separates them.

The capacity of lines is finite: in the plane a line shatters any 3
points but **never** the 4-point XOR pattern. A single hidden layer
(Chapter 5) bends the boundary and pushes past 83%.
:::

::: {.col .fig}
![](../img/mdl-clf-shattering.svg)
:::
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

::: {.cols}
::: {.col}
- **Softmax** = exp, row-sum, divide → a probability distribution over
  classes.
- **Cross-entropy** = $-\log \hat{y}_{\text{true}}$, averaged over the
  batch: the natural classification loss.
- **Model** = flatten → one linear layer ($784\times10$) → softmax.
:::

::: {.col}
- **Training** reuses the regression `Trainer`; `Classifier` adds
  accuracy reporting for free.
- **~83%** is the linear ceiling on Fashion-MNIST; richer models just
  replace the forward pass.
- The naive softmax is numerically fragile; production code fuses
  softmax and log (§4.5).
:::
:::
:::
