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
tf.boolean_mask(y_hat, tf.one_hot(y, depth=y_hat.shape[-1]))
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
    p = jnp.clip(y_hat[list(range(len(y_hat))), y], min=1e-12)
    return -d2l.reduce_mean(d2l.log(p))

cross_entropy(y_hat, y)
```

```{.python .input #softmax-regression-scratch-the-cross-entropy-loss-2}
%%tab tensorflow
def cross_entropy(y_hat, y):  #@save
    p = tf.boolean_mask(y_hat, tf.one_hot(y, depth=y_hat.shape[-1]))
    # Tiny clip to keep log finite when softmax outputs underflow to 0.
    return -tf.reduce_mean(tf.math.log(tf.maximum(p, 1e-12)))

cross_entropy(y_hat, y)
```

Note that we clip $\hat{y}$ away from zero before taking $\log$. Without the clip, $\log(\hat{y})$ produces $-\infty$ (and downstream NaNs) whenever the softmax assigns probability exactly zero to the correct class. Production code typically uses a log-softmax layer that fuses the softmax and log into a single numerically stable operation; the explicit clamp here is the minimal change that keeps the scratch implementation usable as a teaching example without changing its mathematical form.

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
        p = jnp.clip(y_hat[list(range(len(y_hat))), y], min=1e-12)
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

We are more interested in the images we label *incorrectly*. We visualize them by
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

## Summary

By now we are starting to get some experience
with solving linear regression
and classification problems.
With it, we have reached what would arguably be
the state of the art of 1960--1970s of statistical modeling.
In the next section, we will show you how to leverage
deep learning frameworks to implement this model
much more efficiently.

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

::: {.slide title="Softmax regression from scratch"}
The same recipe as linear regression, with two new pieces:

1. **Softmax** turns logits into a probability distribution.
2. **Cross-entropy** is the loss for distributions.

Wired into the same `Module` / `Trainer` scaffold from the
regression chapter — `Classifier` adds accuracy reporting and we
inherit the rest.
:::

::: {.slide title="Sums along an axis"}
Quick reminder before defining softmax — sum along chosen axes:

@softmax-regression-scratch-softmax-regression-implementation-from-scratch

@softmax-regression-scratch-the-softmax-1
:::

::: {.slide title="Softmax"}
$$\mathrm{softmax}(\mathbf{X})_{ij}
  = \frac{\exp(\mathbf{X}_{ij})}{\sum_k \exp(\mathbf{X}_{ik})}.$$

Three steps: exponentiate, sum across the class axis, divide.

@softmax-regression-scratch-the-softmax-2

. . .

Result: every row is non-negative and **sums to 1** — a valid
probability distribution over classes:

@softmax-regression-scratch-the-softmax-3
:::

::: {.slide title="The model"}
Flatten each 32×32 image into a 1024-vector, hit one linear layer
that outputs 10 logits — one per class:

@softmax-regression-scratch-the-model-1

. . .

The forward pass = flatten → linear → softmax:

@softmax-regression-scratch-the-model-2
:::

::: {.slide title="Cross-entropy loss"}
For label `y` (an integer class), the loss on one example is just

$$\ell = -\log \hat{y}_{y}$$

— the negative log of the *predicted probability of the correct
class*. Here are two examples with 3 classes:

@softmax-regression-scratch-the-cross-entropy-loss-1
:::

::: {.slide title="Implementing it"}
One line — fancy indexing pulls out `y_hat[i, y[i]]` for each
example, then negative log:

@softmax-regression-scratch-the-cross-entropy-loss-2

. . .

@softmax-regression-scratch-the-cross-entropy-loss-3
:::

::: {.slide title="Train"}
10 epochs on Fashion-MNIST. The base `Classifier` already handles
the validation loop and accuracy reporting:

@softmax-regression-scratch-training
:::

::: {.slide title="Predicting"}
Pull a fresh validation batch and look at predicted vs. true
classes:

@softmax-regression-scratch-prediction-1

. . .

Tile the misclassified images, captioned with `predicted / true`:

@softmax-regression-scratch-prediction-2

Linear models cap out around ~83% on Fashion-MNIST — easy classes
right, ambiguous shirt-vs-pullover wrong.
:::

::: {.slide title="Recap"}
- **Softmax** = exp + row-sum normalization → probabilities.
- **Cross-entropy** = $-\log p_\text{correct}$, the standard
  classification loss.
- A 10-output linear layer + softmax + CE loss is *the* baseline
  classifier — anything fancier (MLPs, CNNs) just replaces the
  forward pass.
:::
