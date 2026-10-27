```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Generalization in Deep Learning
:label:`sec_generalization_deep`

The MLP of :numref:`sec_mlp-implementation` has a hidden layer of 256 units
and $784 \times 256 + 256 + 256 \times 10 + 10 = 203{,}530$ parameters, about
3.4 for each of the 60,000 Fashion-MNIST training images. In
:numref:`sec_generalization_basics`, a polynomial with one weight per
training example passed through all of its 20 training points and missed
new points by orders of magnitude. With more parameters than that, the MLP
could memorize its training set as well. It does not: its accuracy on test
images, about 87%, is only a few points below its accuracy of about 89% on
the training images. Something other than the number of parameters keeps it from
memorizing.

We need to know what that is, because the width of a network, the length of
its training, and its regularization are all chosen for the sake of
generalization. The answer is not completely known, so this section measures.
We first show that the MLP can fit labels that carry no information, so its
capacity alone cannot explain why it generalizes. Then we follow the test
error of a simpler network as its width grows past the number of training
examples, and see it fall a second time. Both observations point to the
training algorithm: among the many parameter settings that fit the data,
gradient descent finds particular ones, and we work out which in two cases
that can be solved exactly. Finally, the order in which gradient descent fits
the examples turns stopping early into a regularizer. We stay with experiments
and with exact results about simple models; theories that bound the test error
of deep networks directly are still being developed.

## Memorization and Capacity

### Random Labels
:label:`subsec_random_labels`

The direct test of capacity is to train the network on labels drawn at
random, independently of the images, as :citet:`zhang2021understanding` did
for image classifiers. There is no pattern to learn from such labels, so a
network that fits them has memorized them. To keep the experiment fast, we
train on 1,000 Fashion-MNIST images, which gives the MLP about 200
parameters per image.

```{.python .input #generalization-deep-fitting-random-labels-1}
%%tab mxnet
%matplotlib inline
import numpy as np
from mxnet import init, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #generalization-deep-fitting-random-labels-1}
%%tab pytorch
%matplotlib inline
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l
```

```{.python .input #generalization-deep-fitting-random-labels-1}
%%tab tensorflow
%matplotlib inline
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #generalization-deep-fitting-random-labels-1}
%%tab jax
%matplotlib inline
import numpy as np
from flax import nnx
from d2l import jax as d2l
```

We draw 2,000 training images at random, 1,000 to train on and 1,000 to hold
out for validation later, and take the first 2,000 test images. All of them
are flattened into NumPy arrays of 784 pixels.

```{.python .input #generalization-deep-fitting-random-labels-2}
data = d2l.FashionMNIST(batch_size=2000)
X, y = next(iter(data.train_dataloader()))          # 2,000 random images
X_test, y_test = next(iter(data.val_dataloader()))  # the first 2,000
X, y, X_test, y_test = (d2l.numpy(a) for a in (X, y, X_test, y_test))
X, X_test = X.reshape(2000, -1), X_test.reshape(2000, -1)
X_train, y_train, X_val, y_val = X[:1000], y[:1000], X[1000:], y[1000:]
```

Two small helpers run the experiments. `ArrayData` serves arrays as
minibatches. `Tracker` is a `Trainer` that trains on a GPU if there is one
and, after every epoch, measures the accuracy of the model on each of a set
of named datasets, the training images among them, and appends it to the
list `accuracy[name]`.

```{.python .input #generalization-deep-fitting-random-labels-3}
class ArrayData(d2l.DataModule):
    """Minibatches of the arrays X and y; training batches are shuffled."""
    def __init__(self, X, y, batch_size=64):
        super().__init__()
        self.save_hyperparameters()

    def get_dataloader(self, train):
        return self.get_tensorloader(
            (d2l.tensor(self.X), d2l.tensor(self.y)), train)

class Tracker(d2l.Trainer):
    """A Trainer that records the accuracy on named datasets every epoch."""
    def __init__(self, max_epochs, datasets):
        super().__init__(max_epochs, num_gpus=1)
        self.datasets = datasets
        self.accuracy = {name: [] for name in datasets}
        self.board.display = False  # We plot the accuracies ourselves

    def validate(self):
        for name, dataset in self.datasets.items():
            self.val_dataloader = dataset.val_dataloader()
            self.accuracy[name].append(super().validate()['acc'])
        return {}
```

The model has the architecture of the MLP of
:numref:`sec_mlp-implementation`, a hidden layer of 256 ReLUs, but draws its
weights by the He initialization of :numref:`subsec_he_init` instead of with
a fixed standard deviation of 0.01, with which the hidden layer learns slowly
at first (:numref:`sec_mlp-implementation`); its biases start at zero. Its
inputs are already flattened.

```{.python .input #generalization-deep-fitting-random-labels-4}
%%tab mxnet
class MLP(d2l.Classifier):
    def __init__(self, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential()
        self.net.add(nn.Dense(num_hiddens, activation='relu'), nn.Dense(10))
        # Normal weights with variance 2 / fan_in; biases stay zero
        self.net.initialize(init.Xavier(rnd_type='gaussian', factor_type='in',
                                        magnitude=2))
```

```{.python .input #generalization-deep-fitting-random-labels-4}
%%tab pytorch
class MLP(d2l.Classifier):
    def __init__(self, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.LazyLinear(num_hiddens), nn.ReLU(),
                                 nn.LazyLinear(10))

    def weight_init(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            nn.init.zeros_(module.bias)
```

```{.python .input #generalization-deep-fitting-random-labels-4}
%%tab tensorflow
class MLP(d2l.Classifier):
    def __init__(self, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        # One initializer per layer: an unseeded Keras initializer returns
        # the same values on every call
        he = lambda: tf.keras.initializers.VarianceScaling(
            2.0, 'fan_in', 'untruncated_normal')
        self.net = tf.keras.models.Sequential([
            tf.keras.layers.Dense(num_hiddens, activation='relu',
                                  kernel_initializer=he()),
            tf.keras.layers.Dense(10, kernel_initializer=he())])
```

```{.python .input #generalization-deep-fitting-random-labels-4}
%%tab jax
class MLP(d2l.Classifier):
    def __init__(self, num_hiddens, lr, num_inputs=784, rngs=None):
        super().__init__()
        self.save_hyperparameters(ignore=['rngs'])
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        he = nnx.initializers.variance_scaling(2.0, 'fan_in', 'normal')
        self.net = nnx.Sequential(
            nnx.Linear(num_inputs, num_hiddens, kernel_init=he, rngs=rngs),
            nnx.relu,
            nnx.Linear(num_hiddens, 10, kernel_init=he, rngs=rngs))
```

`corrupt` replaces each label, with probability `fraction`, by a class drawn
uniformly at random; with `fraction=1` no label depends on its image. Does the
network fit true and random labels alike? We train it on both for 200 epochs
of minibatch SGD, with learning rate 0.1 and minibatches of 64 images, and
record the accuracy on the training images and on the test images.

```{.python .input #generalization-deep-fitting-random-labels-5}
def corrupt(y, fraction, rng):
    """Replace each label, with probability fraction, by a random class."""
    redraw = rng.random(len(y)) < fraction
    return np.where(redraw, rng.integers(0, 10, len(y)), y)

rng = np.random.default_rng()
curves = []
for fraction in (0, 1):
    labels = corrupt(y_train, fraction, rng)
    trainer = Tracker(max_epochs=200, datasets={
        'train': ArrayData(X_train, labels, 1000),
        'test': ArrayData(X_test, y_test, 1000)})
    trainer.fit(MLP(num_hiddens=256, lr=0.1), ArrayData(X_train, labels))
    curves += [trainer.accuracy['train'], trainer.accuracy['test']]
d2l.plot(list(range(1, 201)), curves, 'epoch', 'accuracy',
         fmts=('C0-', 'C1-', 'C0--', 'C1--'), figsize=(5, 3))
d2l.plt.legend(['train, true labels', 'test, true labels',
                'train, random labels', 'test, random labels'],
               loc='center right', bbox_to_anchor=(1, 0.4));
```

With true labels, training accuracy passes 99% after 60 to 90 epochs, and
test accuracy settles at 80% to 83%. With random labels, the network needs
longer, 115 to 145 epochs to pass 99%, but it fits them too: training
accuracy reaches 100%, while test accuracy stays near the 10% of guessing. The network, the optimizer, and the images are the same in both
runs; only the labels differ.

In :numref:`sec_generalization_basics` we called a class of models that can
fit any labels of its training inputs too flexible for a perfect fit to mean
anything. MLPs of this size come close to such a class for these 1,000
images: they fit a labeling that has nothing to do with the images. Whatever
lets the network generalize from the true labels, it is not a lack of
capacity to memorize.

### Capacity Bounds

Learning theory turns capacity into guarantees. In
:numref:`chap_classification_generalization` we met the VC bound: with
probability at least $1 - \delta$ over the draw of $n$ training examples,
every classifier $f$ in a class of VC dimension $v$ satisfies

$$
\epsilon(f) \leq \epsilon_\mathcal{S}(f) + c\sqrt{\frac{v + \log(1/\delta)}{n}},
$$

where $\epsilon(f)$ is the population error, $\epsilon_\mathcal{S}(f)$ the
training error, and $c$ a constant. The bound guarantees that the training
error carries over to new data once $n$ is large compared with $v$. For a
linear classifier, $v$ is one more than the number of features, the number
of its parameters. If we estimate $v$ the same way for the MLP trained on all
60,000 images, the square root alone is
$\sqrt{203{,}530/60{,}000} \approx 1.8$. An error never exceeds 1, so
unless the constant $c$ is below about 0.55, the bound guarantees nothing.
Neither a smaller constant nor a sharper estimate of $v$ can rescue it: as
exercise :exref:`ex_generalization-deep-complexity` shows, random labels
rule out every bound that depends only on the class, $n$, and $\delta$.

### Inductive Bias

If the class does not decide, the choice among the functions that fit the
training data does. A learner that fits $n$ labeled examples must still
decide how to label every other input, and the training data cannot settle
this. Averaged over all ways in which the unseen inputs could be labeled,
every learner has the same expected error, a result known as the *no free
lunch theorem* :cite:`Wolpert.1996`. A learner generalizes on a particular
problem when its preferences among the functions that fit match the way
the problem labels new inputs. These preferences are its *inductive bias*.
The learners we have met make different bets:

| Learner | Inductive bias |
|---|---|
| Linear classifier | classes are separated by hyperplanes |
| Polynomial fit with weight decay | small weights, hence slowly varying curves |
| 1-nearest neighbor | an input has the label of the closest training input |
| MLP trained by gradient descent | the subject of the rest of this section |

The nearest-neighbor rule shows that fitting every training example need not
mean overfitting. It labels each training input correctly by construction,
yet under mild conditions, as the training set grows, its error tends to at
most twice the smallest error any classifier can achieve
:cite:`Cover.Hart.1967`. Fitting
the data exactly is compatible with generalizing; the error is decided by
the labels the learner assigns between the training points.

## Double Descent
:label:`subsec_double_descent`

### The Interpolation Threshold

In :numref:`sec_generalization_basics`, the test error of polynomial fits
fell as the degree rose to 3 and then grew by orders of magnitude up to
degree 19, where the 20 weights fit the 20 training points exactly. The
study stopped there, because beyond that degree the least-squares problem
has infinitely many exact solutions. We now continue past this point, with
a model whose solutions we can compute.

Freeze the hidden layer of the MLP at its random initialization and train
only the output layer. We drop the biases and, instead of cross-entropy,
use the squared loss on one-hot labels, which turns training into linear
least squares. Let $\mathbf{H} = \textrm{ReLU}(\mathbf{X}\mathbf{W}^{(1)})
\in \mathbb{R}^{n \times h}$ be the hidden representation of the $n$
training images with $h$ hidden units, and $\mathbf{Y} \in \mathbb{R}^{n
\times 10}$ their one-hot labels. Fitting chooses the output weights
$\mathbf{W}^{(2)} \in \mathbb{R}^{h \times 10}$ to minimize the squared
error summed over all entries, $\|\mathbf{H}\mathbf{W}^{(2)} -
\mathbf{Y}\|_F^2$.

Each output is a separate least-squares problem with $n$ equations and $h$
unknowns. The MLP of :numref:`subsec_random_labels` fitted 1,000 random
labels with 256 hidden units because all of its 203,530 parameters were
trained. With the hidden layer frozen, each output keeps only its $h$
weights, so an exact fit of $n$ labels needs $h \geq n$. For $h < n$ the
least-squares problem has a unique solution, which usually misses some
labels. At $h = n$ the matrix $\mathbf{H}$ is square. Once we remove the
hidden units that are active on almost no images, as described below, it is
invertible like almost every random square matrix, so exactly one solution
reproduces every label. For $h > n$ there are infinitely many solutions that
reproduce every label, and we take the one of smallest norm.
`np.linalg.lstsq` returns this solution in all three cases. The width
$h = n$, at which an exact fit first becomes possible, is the
*interpolation threshold*.

A network whose hidden layer stays at its random initialization is called a
*random-features* model :cite:`Rahimi.Recht.2007`. Very wide networks
trained with small steps stay close to a linear model of this kind, built
from the gradients at their initialization; this is the *lazy* or *neural
tangent* regime :cite:`Jacot.Gabriel.Hongler.2018`. Networks used in
practice also change their features during training. :numref:`sec_scaling`
shows how width-dependent learning rates keep this feature learning from
vanishing as networks grow wider.

We compute the hidden representation once for 10,000 hidden units; the model
with $h$ units uses its first $h$ columns. Since all pixels are nonnegative, a
unit whose weights are negative on most of the bright pixels is active on only
a few images, or on none. Such a unit contributes a column with almost all
entries zero, and two units that are active only on the same image contribute
proportional columns, so that $\mathbf{H}$ loses rank. About one or two units
in a hundred are active on fewer than ten of the 1,000 training images, so we
draw 11,000 units and keep the first 10,000 of those that are active on at
least ten.

```{.python .input #generalization-deep-beyond-the-interpolation-threshold-1}
W1 = rng.normal(0, np.sqrt(2 / 784), (784, 11000))  # He-initialized, frozen
H, H_test = np.maximum(X_train @ W1, 0), np.maximum(X_test @ W1, 0)
images = (H > 0).sum(axis=0)  # Images on which each unit is active
print(f'units active on fewer than 10 images: {(images < 10).mean():.1%}')
active = np.flatnonzero(images >= 10)[:10000]
H, H_test = H[:, active], H_test[:, active]
Y = np.eye(10)[y_train]                             # One-hot labels

def error(H, W2, y):
    """Fraction of examples whose largest output is not their label."""
    return (np.argmax(H @ W2, axis=1) != y).mean()
```

How does the error depend on the width $h$? We fit the output layer for
widths from 50 to 10,000 and record the smallest singular value of each
feature matrix, which `lstsq` returns, and the norm of the solution.

```{.python .input #generalization-deep-beyond-the-interpolation-threshold-2}
widths = [50, 100, 200, 400, 600, 800, 900, 1000, 1100, 1200, 1500, 2000,
          3000, 5000, 10000]
train_err, test_err, s_min, norm = [], [], {}, {}
for h in widths:
    W2, _, _, s = np.linalg.lstsq(H[:, :h], Y, rcond=None)
    train_err.append(error(H[:, :h], W2, y_train))
    test_err.append(error(H_test[:, :h], W2, y_test))
    s_min[h], norm[h] = s.min(), np.linalg.norm(W2)
d2l.plot([widths, widths, [1000, 1000]], [train_err, test_err, [0, 1]],
         'hidden units h', 'error', legend=['train', 'test', 'h = n'],
         xscale='log', fmts=('C0-', 'C1-', 'k:'), figsize=(5, 3))
```

The left part of the curve is the U of :numref:`sec_generalization_basics`.
Test error falls from about 30% at 50 hidden units to between 21% and 25%
at a few hundred units, and then rises. At the interpolation threshold
$h = n = 1{,}000$, it peaks at about 85% to 90%, barely better than guessing.
Beyond the threshold every model fits all of its training labels, yet test
error falls again, and at 10,000 hidden units it reaches 18% to 23%, below
the best width before the threshold. This shape is called *double descent*
:cite:`Belkin.Hsu.Ma.ea.2019`. The plotted training error counts
misclassified images, so it reaches zero already at 800 to 900 hidden units,
where the largest output of each training image belongs to its label
although the squared error is not yet zero; the squared error vanishes only from
$h = n$ on. Past the threshold, the training error cannot tell the models
apart; the test error can.

### Conditioning at the Threshold

Why does the error peak where the number of hidden units equals the number of
examples? At $h = n$ the feature matrix $\mathbf{H}$ is square, and fitting
must solve $\mathbf{H}\mathbf{W}^{(2)} = \mathbf{Y}$ exactly. A random square
matrix is almost always invertible, but in some direction it is nearly
singular: its smallest singular value $s_\textrm{min}$ is close to zero,
and its inverse stretches that direction by $1/s_\textrm{min}$. To
reproduce the labels, the solution needs a large component along it, and
predictions on new images inherit that component. With fewer hidden units,
least squares need not fit every label; with more, many solutions fit, and
the one of smallest norm avoids the nearly singular direction.

```{.python .input #generalization-deep-conditioning-at-the-threshold-1}
for h in (200, 900, 1000, 1100, 10000):
    print(f'h = {h:5d}: smallest singular value {s_min[h]:.1e}, '
          f'norm of W2 {norm[h]:6.1f}')
```

At the threshold the smallest singular value is tens to hundreds of times
smaller than at 1,100 hidden units, depending on the draw of the features.
The norm of the solution jumps from about 50 at 900 units to several hundred
or more, often thousands, against about 40 at 1,100 units and 3 at 10,000.
Beyond the threshold, the norm can only shrink as units are added: an exact fit with
$h$ hidden units remains an exact fit with $h + 1$ (give the new unit zero
weight), so the smallest norm among the exact fits cannot grow. The
mathematics appendix analyzes this mechanism in a smaller example
(:numref:`subsec_mdl-double-descent-example`).

The threshold moves with the number of examples, which has a surprising
consequence. With 1,000 hidden units, how does the test error change if we
train on half as many images?

```{.python .input #generalization-deep-conditioning-at-the-threshold-2}
for n in (500, 1000):
    W2 = np.linalg.lstsq(H[:n, :1000], Y[:n], rcond=None)[0]
    test_error = error(H_test[:, :1000], W2, y_test)
    print(f'{n} training images: test error {test_error:.2f}')
```

Doubling the training set from 500 to 1,000 images raises the test error
from about a third to 85% to 90%, because the larger set puts the model onto
its interpolation threshold. :citet:`nakkiran2021deep` observed the same
peaks in deep networks, as the width, the number of training epochs, or the
number of examples varies. The peak sits where training can just barely fit
the training set.

## Implicit Bias of Gradient Descent
:label:`subsec_implicit_regularization`

The second descent depended on taking the smallest solution among the many
that fit. Nothing in the loss asks for this: every exact fit has zero training
loss. Which fit does gradient descent find? For two models the answer is
known, and in both the choice is made by the training algorithm. Such a
preference of a training algorithm is called its *implicit bias*.

### Minimum-Norm Interpolation

Consider one output of the random-features model: weights
$\mathbf{w} \in \mathbb{R}^h$, the loss
$L(\mathbf{w}) = \frac{1}{2}\|\mathbf{H}\mathbf{w} - \mathbf{y}\|^2$ with
$h > n$, and a matrix $\mathbf{H}$ whose $n$ rows are linearly independent.
The exact fit of smallest norm is

$$
\mathbf{w}^\star = \mathbf{H}^\top(\mathbf{H}\mathbf{H}^\top)^{-1}\mathbf{y}.
$$

It is a combination of the rows of $\mathbf{H}$, and it reproduces the labels,
since $\mathbf{H}\mathbf{w}^\star = \mathbf{y}$.

**Proposition.** Gradient descent on $L$ started at $\mathbf{w}_0 = \mathbf{0}$,
with a learning rate small enough for it to converge, converges to
$\mathbf{w}^\star$.

**Proof.** The gradient $\mathbf{H}^\top(\mathbf{H}\mathbf{w} - \mathbf{y})$
is a combination of the rows of $\mathbf{H}$. Starting from zero, every
iterate is therefore a combination of the rows, $\mathbf{w}_t =
\mathbf{H}^\top\mathbf{a}_t$ for some $\mathbf{a}_t \in \mathbb{R}^n$, and so
is the limit. At the limit the gradient vanishes, so for this convex loss
the limit is a minimizer. Since the rows of $\mathbf{H}$ are independent,
exact fits exist, so the limit fits exactly:
$\mathbf{H}\mathbf{H}^\top\mathbf{a} = \mathbf{y}$, which gives $\mathbf{a} =
(\mathbf{H}\mathbf{H}^\top)^{-1}\mathbf{y}$ and the limit $\mathbf{w}^\star$.
It remains to see that no exact fit is shorter. Every exact fit is
$\mathbf{w}^\star + \mathbf{v}$ with $\mathbf{H}\mathbf{v} = \mathbf{0}$. Such
a $\mathbf{v}$ is orthogonal to every row of $\mathbf{H}$, hence to
$\mathbf{w}^\star$, so $\|\mathbf{w}^\star + \mathbf{v}\|^2 =
\|\mathbf{w}^\star\|^2 + \|\mathbf{v}\|^2$. $\blacksquare$

:numref:`fig_mdl-mlp-min-norm` shows the argument for one equation in two
unknowns. Every step moves perpendicular to the line of exact fits, so a
start at zero ends at the point of the line closest to the origin. The
component of a start $\mathbf{w}_0$ along the line never changes, so a start
with such a component ends at a different, longer fit.

![Gradient descent on least squares with one equation $\mathbf{h}^\top\mathbf{w} = y$ in two unknowns. The exact fits form a line (green). Each gradient is a multiple of $\mathbf{h}$, so each step moves perpendicular to that line. Started at $\mathbf{0}$ (blue), gradient descent stops at the point of the line closest to the origin, where the dashed circle touches it. Started at $\mathbf{w}_0$ (orange), it stops at a longer exact fit.](../img/mdl-mlp-min-norm.svg)
:label:`fig_mdl-mlp-min-norm`

We check the proposition on a random problem with 20 equations and 100
unknowns, starting once at zero and once at a random point.

```{.python .input #generalization-deep-minimum-norm-interpolation}
H_small = rng.normal(size=(20, 100))
y_small = rng.normal(size=20)
w_star = np.linalg.lstsq(H_small, y_small, rcond=None)[0]
lr = 1 / np.linalg.norm(H_small, 2) ** 2  # Below 2 / largest curvature
for name, w in (('zero', np.zeros(100)), ('random', rng.normal(size=100))):
    for _ in range(2000):
        w = w - lr * H_small.T @ (H_small @ w - y_small)
    residual = np.linalg.norm(H_small @ w - y_small)
    print(f'start {name:6s}: residual {residual:.0e}, norm '
          f'{np.linalg.norm(w):5.2f}, distance to w* '
          f'{np.linalg.norm(w - w_star):.1f}')
```

Both runs fit every label. The run from zero ends at the minimum-norm
solution of `lstsq` up to rounding; the run from a random start ends at
another exact fit, more than ten times longer. The minimum-norm solutions
of the double-descent experiment are therefore the ones gradient descent from
zero would find.

### Maximum-Margin Classification

For classification with cross-entropy the situation differs. Take two classes
with labels $y \in \{-1, +1\}$ and a linear score $\mathbf{w}^\top\mathbf{x}$,
so that an example has the margin $m = y \mathbf{w}^\top\mathbf{x}$ and the
loss $\log(1 + \exp(-m))$ of :numref:`sec_softmax`. If some $\mathbf{w}$
classifies every training example correctly, no finite $\mathbf{w}$ minimizes
the loss: scaling up a separating $\mathbf{w}$ raises every margin and lowers
every loss. Gradient descent lets $\|\mathbf{w}\|$ grow without bound, and the
question becomes which direction $\mathbf{w}/\|\mathbf{w}\|$ it approaches.
:citet:`Soudry.Hoffer.Nacson.ea.2018` proved that it approaches the
*maximum-margin* direction: of all separators, the one whose smallest distance
to a training point is largest. The intuition is that once $\|\mathbf{w}\|$ is
large, the loss of an example is about $\exp(-m)$, so the examples with the
smallest margins dominate the gradient, and gradient descent works mostly on
raising them. :numref:`fig_mdl-mlp-max-margin` shows that the approach is
slow: the training error is zero from the first step, yet the separator is
still turning after $10^5$ steps.

![Gradient descent on the logistic loss for two separable classes, with separators through the origin. (a) The separators after $t = 1$, $10^2$, and $10^4$ steps (gray to blue) turn toward the maximum-margin separator (black), whose margin is dashed; the two examples closest to it are ringed. (b) The angle between $\mathbf{w}_t$ and the maximum-margin direction keeps shrinking long after every training example is classified correctly.](../img/mdl-mlp-max-margin.svg)
:label:`fig_mdl-mlp-max-margin`

Both results concern linear models. For deep networks, statements this exact
are rare: the parameter norm that gradient descent keeps small is not even a
property of the function the network computes, because different parameters
can compute the same function (exercise
:exref:`ex_generalization-deep-implicit-bias`). The next experiment shows a
preference of gradient descent in our MLP that needs no such statement.

## Early Stopping
:label:`subsec_early_stopping`

### Label Noise

Gradient descent also has a preference in time.
:citet:`Arpit.Jastrzebski.Ballas.ea.2017` observed that networks trained on
partly mislabeled data fit the correctly labeled examples first and the
mislabeled ones later. We repeat their experiment on our 1,000 images,
redrawing 20% of the labels at random. A redrawn label is correct one time
in ten, so about 18% of the labels end up wrong. After every epoch we measure
the accuracy on the correctly labeled training images, on the mislabeled
training images against their wrong labels, and on the 1,000 held-out
training images, our validation set.

```{.python .input #generalization-deep-clean-labels-first}
labels = corrupt(y_train, 0.2, rng)
wrong = labels != y_train
trainer = Tracker(max_epochs=200, datasets={
    'clean': ArrayData(X_train[~wrong], labels[~wrong], 1000),
    'mislabeled': ArrayData(X_train[wrong], labels[wrong], 1000),
    'validation': ArrayData(X_val, y_val, 1000),
    'test': ArrayData(X_test, y_test, 1000)})
trainer.fit(MLP(num_hiddens=256, lr=0.1), ArrayData(X_train, labels))
acc = trainer.accuracy
d2l.plot(list(range(1, 201)), [acc['clean'], acc['mislabeled'],
                               acc['validation']], 'epoch', 'accuracy',
         legend=['clean training labels', 'wrong training labels',
                 'validation'], figsize=(5, 3))
```

The two kinds of training images are fit at very different speeds. After 20
epochs the network reproduces the labels of 80% to 90% of the correctly
labeled images, but those of fewer than one in five of the mislabeled ones.
Validation accuracy peaks in this first phase, within the first 40 epochs. Then
the network memorizes the wrong labels, nearly all of them after 100 to 160
epochs, and validation accuracy declines by several points. Memorizing the
mislabeled examples costs accuracy on new images.

### The Stopping Rule

*Early stopping* exploits this order. It monitors the validation accuracy
after every epoch and keeps the parameters of the best epoch. In practice,
training ends once the validation accuracy has not improved for a number of
epochs called the *patience* (exercise
:exref:`ex_generalization-deep-patience`). Here we pick the best epoch from
the recorded history and judge the choice on the test images, which played no
part in it.

```{.python .input #generalization-deep-stopping-rule}
best = int(np.argmax(acc['validation']))
print(f'best validation accuracy after epoch {best + 1}: '
      f'test accuracy {acc["test"][best]:.3f}')
print(f'after all 200 epochs: test accuracy {acc["test"][-1]:.3f}')
```

Stopping at the best validation epoch gains about three to seven points of
test accuracy over training to the end. Early stopping limits how far the
parameters move from their initialization, much as weight decay limits
their size; for least squares the two are nearly equivalent (exercise
:exref:`ex_generalization-deep-early-stopping-ridge`). It also saves the
compute of the remaining epochs.

The rule assumes that validation accuracy does not rise again after a long
plateau. :citet:`Power.Burda.Edwards.ea.2022` found a counterexample on
small algorithmic tasks such as addition modulo a prime. Networks trained
with weight decay fit their training set within about a thousand steps, stay
near chance on validation data for many times as long, and generalize only
after a hundred to a thousand times as many steps (:numref:`fig_grokking`). This is called *grokking*. A patience
rule would have stopped long before.

![Grokking, schematically, after :citet:`Power.Burda.Edwards.ea.2022`. On a small algorithmic task, training accuracy (gray) saturates within about a thousand steps. Validation accuracy (blue) stays near chance for more than ten times as many steps and reaches the training accuracy two to three orders of magnitude after it.](../img/mdl-mlp-grokking.svg)
:label:`fig_grokking`

Gradient descent, not the parameter count, decides which of the many exact
fits a network finds. From zero it finds the minimum-norm fit, on separable
data it approaches the maximum-margin separator, and on noisy labels it fits
the clean examples first. Left to run, it also memorizes the noise. The next
section studies dropout, a perturbation applied during training that may
change what the network prefers to fit.

## Summary

An MLP with more parameters than training examples generalizes, although it
can also fit random labels. Counting parameters, or any
capacity measure of the class alone, therefore cannot explain its
generalization, and the VC bound is vacuous for it. A random-features model
shows what happens past the interpolation threshold: test error peaks where
the number of hidden units equals the number of examples, because the feature
matrix is nearly singular there, and falls again as the width grows. Which of
the many exact fits a model finds is decided by the training algorithm:
gradient descent from zero finds the minimum-norm fit for least squares and
approaches the maximum-margin separator for the logistic loss. With noisy
labels it fits the correctly labeled examples first, so stopping at the best
validation epoch keeps the clean fit and discards the memorized noise.

## Exercises

1. **Limits of complexity measures.** Consider a generalization bound that
   holds for every data distribution: with probability at least
   $1 - \delta$ over the draw of $n$ training examples, every classifier
   $f$ in a class $\mathcal{F}$ satisfies

    $$\epsilon(f) \leq \epsilon_\mathcal{S}(f) + B(\mathcal{F}, n, \delta).$$

    Here $\epsilon(f)$ is the population error, $\epsilon_\mathcal{S}(f)$ is
    the training error, and the term $B$ depends only on $\mathcal{F}$, $n$,
    and $\delta$, as the term $c\sqrt{(v + \log(1/\delta))/n}$ of the VC
    bound in :numref:`chap_classification_generalization` does. Suppose that training
    finds a member of an MLP class $\mathcal{F}$ with zero training error on
    the 60,000 Fashion-MNIST training images, both with the true labels and
    with labels drawn uniformly at random from the 10 classes, independently
    of the images, as :citet:`zhang2021understanding` observed for image
    classifiers.
    1. What population error does every classifier have on the random-label
       data? What lower bound on $B(\mathcal{F}, n, \delta)$ follows?
    1. What does the bound then guarantee about the population error of the
       network trained on the true labels?
   :exlabel:`ex_generalization-deep-complexity`
1. [code] **Patience-based stopping.** `FashionMNIST` uses the Fashion-MNIST
   test set as its validation data, so hold out 10,000 training images as a
   validation set instead. Train an MLP with one hidden layer of 256 units,
   as in :numref:`sec_mlp-implementation`, on the remaining images for a
   fixed number of epochs large enough to overfit, and record the validation loss
   and the test accuracy after every epoch. A patience rule with tolerance
   $\tau$ and patience $k$ stops once the validation loss has failed to
   improve on its best value by more than $\tau$ for $k$ consecutive
   epochs. It then retains the parameters of the epoch with the lowest
   validation loss so far.
    1. Fix $\tau$ and apply the rule to the recorded curves for
       $k \in \{1, 5, 20\}$. For each $k$, report the stopping epoch, the
       retained epoch, and the test accuracy at both.
    1. How does the test accuracy at the retained epoch depend on $k$?
       Relate the dependence to the epoch-to-epoch fluctuation of the
       validation loss.

    *Adapted from Michael Nielsen,
    [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap3.html),
    Chapter 3.*
   :exlabel:`ex_generalization-deep-patience`
1. [code] **More data can hurt.** Fix the random-features model at 1,000
   hidden units and fit it, as in :numref:`subsec_double_descent`, on the
   first $n$ of 2,000 training images for $n$ from 100 to 2,000.
    1. Plot the test error against $n$. Where does it peak, and how does the
       peak compare with the error at $n = 2{,}000$?
    1. Plot the smallest singular value of the feature matrix against $n$ and
       explain the shape of the error curve with it.
    1. Repeat with 500 and 2,000 hidden units. Where does the peak move?
1. [code] **Weight decay at the threshold.** Replace the minimum-norm fit of
   the double-descent experiment by the ridge solution
   $\mathbf{W}^{(2)} = (\frac{1}{n}\mathbf{H}^\top\mathbf{H} + \lambda
   \mathbf{I})^{-1}\frac{1}{n}\mathbf{H}^\top\mathbf{Y}$ of
   :numref:`sec_weight_decay`, which, as there, averages the squared error
   over the $n$ examples.
    1. Plot the test error against the width for $\lambda \in \{10^{-5},
       10^{-3}, 10^{-1}\}$. What happens to the peak at $h = n$?
    1. Show that the ridge solution approaches the minimum-norm solution as
       $\lambda \to 0$ for $h > n$. Hint: write
       $(\frac{1}{n}\mathbf{H}^\top\mathbf{H} + \lambda
       \mathbf{I})^{-1}\frac{1}{n}\mathbf{H}^\top =
       \mathbf{H}^\top(\mathbf{H}\mathbf{H}^\top + n\lambda \mathbf{I})^{-1}$.
    1. For each width, choose $\lambda$ on the 1,000 validation images. Does
       the curve of the tuned model still descend twice?
1. **Early stopping and weight decay.** ● Early stopping limits the number
   of training epochs where weight decay limits the weights. Linear least
   squares makes this precise. Let $\mathbf{X}^\top\mathbf{X}/n$ have
   eigenvalues $\mu_i > 0$, let $\mathbf{w}^*$ minimize
   $L(\mathbf{w}) = \frac{1}{2n}\|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2$, and
   run gradient descent on $L$ from $\mathbf{w}_0 = \mathbf{0}$ with learning
   rate $\eta < 1/\max_i \mu_i$.
    1. Show that after $t$ steps the coordinate of $\mathbf{w}_t$ along the
       $i$-th eigenvector equals $1 - (1 - \eta\mu_i)^t$ times that of
       $\mathbf{w}^*$.
    1. The minimizer of $L(\mathbf{w}) + \frac{\lambda}{2}\|\mathbf{w}\|^2$
       scales the same coordinate by
       $\mu_i/(\mu_i + \lambda)$ (:numref:`subsec_wd-shrinkage`). For which
       $\lambda$, as a function of $\eta$ and $t$, do the two factors
       approximately agree both in the directions with $\eta\mu_i t \ll 1$
       and in those with $\eta\mu_i t \gg 1$?

    *Adapted from Ian Goodfellow, Yoshua Bengio, and Aaron Courville,
    [Deep Learning](https://www.deeplearningbook.org/contents/regularization.html),
    Section 7.8.*
   :exlabel:`ex_generalization-deep-early-stopping-ridge`
1. **Implicit bias of the logistic loss.** ● For linearly separable data,
   gradient descent on the logistic loss converges in direction to the
   maximum-margin separator :cite:`Soudry.Hoffer.Nacson.ea.2018`. Take
   labels $y_i \in \{-1, +1\}$.
    1. For a linear predictor $f(\mathbf{x}) = \mathbf{w}^\top\mathbf{x}$,
       show that on separable data the loss
       $\sum_i \log\left(1 + \exp(-y_i \mathbf{w}^\top\mathbf{x}_i)\right)$
       is positive for every $\mathbf{w}$ but has infimum zero. Conclude that
       $\|\mathbf{w}_t\|$ grows without bound when gradient descent drives
       the loss to zero, so that only the direction
       $\mathbf{w}_t/\|\mathbf{w}_t\|$ can converge.
    1. For a two-layer ReLU network
       $f(\mathbf{x}) = \sum_j v_j \max(0, \mathbf{w}_j^\top\mathbf{x})$,
       show that replacing $(\mathbf{w}_j, v_j)$ by
       $(c_j\mathbf{w}_j, v_j/c_j)$ with $c_j > 0$ leaves $f$ unchanged. Find
       the smallest value of $\sum_j (\|\mathbf{w}_j\|^2 + v_j^2)$ over these
       rescalings. Explain why a margin measured with the parameter norm of
       the network, unlike one measured with $\|\mathbf{w}\|$ for the linear
       predictor, is not a property of the function $f$ alone.
   :exlabel:`ex_generalization-deep-implicit-bias`
1. [extended] **Epoch-wise double descent.** ● Take an MLP with one hidden
   layer of 256 units, as in :numref:`sec_mlp-implementation`, on
   Fashion-MNIST and randomly relabel
   15% of the training examples. Train far past convergence (several hundred
   epochs). After every epoch, record the test error and the training
   accuracy on the relabeled examples, measured against their new labels.
   Plot both against the epoch count on a log axis.
    1. Does the test error descend a second time after the initial rise?
    1. At which epoch does the test error peak, and what fraction of the
       relabeled examples has the network fitted by then?
    1. Repeat with 5% and 30% of the examples relabeled. How do the height
       and the epoch of the peak change?
    1. Suppose a patience rule with $k = 20$ and $\tau = 0$ monitored the
       test-error curve of the 15% run as if it were a validation curve. At
       which epoch would the rule stop, and how does the test error there
       compare with the test error at the end of training?
   :exlabel:`ex_generalization-deep-epoch-wise-double-descent`
1. [extended] **Grokking.** ●● Reproduce the setup of
   :citet:`Power.Burda.Edwards.ea.2022`: train a small network (they use a
   two-layer transformer, but a wide MLP on one-hot pairs also works) to
   predict $c = (a + b) \bmod 97$ from the pair $(a, b)$, using a random 50%
   of all pairs for training, with weight decay, for $10^5$ or more steps.
   Plot training and validation accuracy against the logarithm of the step
   count, and compare with :numref:`fig_grokking`. How does the delay before
   generalization change with the training fraction and with the
   weight-decay strength?
   :exlabel:`ex_generalization-deep-grokking`

[Discussions](https://d2l.discourse.group/t/7473)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §4.5]{.kicker}

Generalization in Deep Learning<br>**random labels · double descent · implicit bias · early stopping**
:::
:::

::: {.slide title="Our MLP Has 3.4 Parameters per Training Image and Still Generalizes"}
[The problem]{.kicker}

| model | parameters | per training example | on test data |
|---|---:|---:|---|
| polynomial, degree 19, 20 points | 20 | 1 | error over 1000× the cubic's |
| MLP, 256 hidden units, Fashion-MNIST | 203,530 | 3.4 | about 87%, against about 89% on training images |

- Counting parameters predicts that the MLP memorizes. It does not.
- To choose width, training time, and regularization we need to know what
  controls generalization instead.

::: {.d2l-note}
Measure: what the network *can* fit, what happens past one parameter per
example, and what gradient descent *does* fit.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- show by experiment that a network's capacity cannot explain its
  generalization, and compute why the VC bound is vacuous for it;
- measure double descent and explain its peak by the conditioning of the
  feature matrix;
- prove that gradient descent from zero finds the minimum-norm fit, and
  state its logistic-loss counterpart, the maximum-margin separator;
- use early stopping against label noise.

**Running example:** Fashion-MNIST after 30 epochs. Softmax regression:
7,850 parameters, about 84%. The MLP: 203,530 parameters, about 87%.
Experiments use 1,000 training images.
:::

::: {.slide title="Generalization Asks Why Large Networks Do Not Memorize"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Multilayer Perceptrons | hidden layers, activation functions, universal approximation |
| Implementation of Multilayer Perceptrons | an MLP trained on Fashion-MNIST |
| Backpropagation | how the gradients are computed, and at what memory cost |
| Numerical Stability and Initialization | vanishing gradients, Xavier and He initialization |
| **Generalization in Deep Learning** | **memorization, double descent, implicit bias, early stopping** |
| Dropout | random masks, robustness to missing units |
| Predicting House Prices on Kaggle | the workflow on real data |
:::

::: {.col .narrow}
- **Uses:** the polynomial U-curve, the VC bound, least squares,
  the MLP with He initialization.
- **Skips:** neural tangent kernels, PAC-Bayesian bounds, the proof of the
  maximum-margin theorem.
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[Memorization and Capacity]{.dtitle}

[random labels, the VC bound, inductive bias]{.dsub}
:::
:::

::: {.slide title="The Same MLP Fits Random Labels as Well as True Ones"}
[Capacity · experiment]{.kicker}

::: {.cols .vc}
::: {.col .narrow}
1,000 images, 203,530 parameters, SGD for 200 epochs. Random labels:
each label redrawn uniformly, independent of the image.

- True labels: over 99% train after 60–90 epochs, 80–83% test.
- Random labels: over 99% train after 115–145 epochs, about 10% test.

::: {.d2l-note}
Same network, optimizer, and images; only the labels differ.
:::
:::

::: {.col .fig .big}
@!generalization-deep-fitting-random-labels-5
:::
:::
:::

::: {.slide title="The VC Bound Is Vacuous for the MLP: √(203,530 / 60,000) ≈ 1.8"}
[Capacity · theory]{.kicker}

With probability $1 - \delta$, every $f$ in a class of VC dimension $v$:

$$\epsilon(f) \leq \epsilon_\mathcal{S}(f) + c\sqrt{\frac{v + \log(1/\delta)}{n}}$$

- $\epsilon$: population error; $\epsilon_\mathcal{S}$: training error.
- Linear classifier: $v$ = number of parameters. Same estimate for the
  MLP on all 60,000 images: $\sqrt{203{,}530/60{,}000} \approx 1.8$.

::: {.d2l-note .warn}
An error never exceeds 1: the bound guarantees nothing. Random labels rule
out **every** bound that depends only on the class and $n$.
:::
:::

::: {.slide title="Every Learner Bets on an Inductive Bias"}
[Capacity · inductive bias]{.kicker}

No free lunch: averaged over all labelings of the unseen inputs, every
learner has the same expected error (Wolpert, 1996).

| learner | prefers |
|---|---|
| linear classifier | classes separated by hyperplanes |
| polynomial + weight decay | small weights, slowly varying curves |
| 1-nearest neighbor | the label of the closest training input |
| MLP + gradient descent | **this deck** |

::: {.d2l-note}
1-NN fits every training point, yet its error tends to at most twice the
best possible (Cover and Hart, 1967). Exact fit ≠ overfitting.
:::
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Double Descent]{.dtitle}

[past one parameter per example]{.dsub}
:::
:::

::: {.slide title="The Classical U: Polynomial Fits Get Worse Up to One Weight per Example"}
[Double descent · recap]{.kicker}

::: {.cols .vc}
::: {.col .narrow}
20 noisy points from a cubic:

- degree 3: test error at the noise level;
- degree 19: 20 weights fit all 20 points, test error more than 1000×
  larger.

::: {.d2l-note}
The study stopped at degree 19: beyond it, infinitely many exact fits.
Which one should we take?
:::
:::

::: {.col .fig .big}
![Training error falls with complexity; test error falls, then rises.](../img/mdl-gen-capacity-error.svg){width=100%}
:::
:::
:::

::: {.slide title="Random Features: Freeze the Hidden Layer, Fit the Output by Least Squares"}
[Double descent · model]{.kicker}

$\mathbf{H} = \textrm{ReLU}(\mathbf{X}\mathbf{W}^{(1)}) \in \mathbb{R}^{n\times h}$,
$\mathbf{W}^{(1)}$ frozen at its He initialization; one-hot labels
$\mathbf{Y} \in \mathbb{R}^{n \times 10}$.

$$\min_{\mathbf{W}^{(2)}} \|\mathbf{H}\mathbf{W}^{(2)} - \mathbf{Y}\|^2$$

- $h < n$: unique least-squares fit, misses some labels.
- $h = n$: square, invertible $\mathbf{H}$: exactly one exact fit, the
  **interpolation threshold**.
- $h > n$: infinitely many exact fits; take the **minimum-norm** one.
  `np.linalg.lstsq` returns it in all three cases.

Here $n = 1{,}000$ images and $h$ from 50 to 10,000.
:::

::: {.slide title="Past Interpolation, Test Error Falls Again"}
[Double descent · experiment]{.kicker}

::: {.cols .vc}
::: {.col .narrow}
- $h < n$: the classical U, best 21–25% at a few hundred units.
- $h = n$: about 85–90% error, near guessing.
- $h = 10{,}000$: 18–23%, **below** the best $h < n$.

::: {.d2l-note}
From $h = n$ on, every model fits every label exactly: training error cannot
tell these models apart.
:::
:::

::: {.col .fig .big}
@!generalization-deep-beyond-the-interpolation-threshold-2
:::
:::
:::

::: {.slide title="The Peak Sits Where Hidden Units Equal Examples: H Is Nearly Singular"}
[Double descent · conditioning]{.kicker}

At $h = n$, $\mathbf{H}$ is square: the fit must invert it. Its smallest
singular value $s_\textrm{min} \approx 0$, and the inverse stretches that
direction by $1/s_\textrm{min}$.

@!generalization-deep-conditioning-at-the-threshold-1

::: {.d2l-note}
Past $h = n$, an exact fit with $h$ units is one with $h + 1$ (zero new
weight): the minimum norm can only shrink.
:::
:::

::: {.slide title="Check: How Can More Data Hurt?"}
[Check]{.kicker}

The random-features model with 1,000 hidden units is trained on 500 images,
then on 1,000. Which test error is lower?

::: {.fragment .fade-up}
@!generalization-deep-conditioning-at-the-threshold-2

::: {.d2l-note}
**Answer.** 500 images. With 1,000 images, $h = n$: the model sits on its
interpolation threshold. Deep networks show the same peaks in width,
epochs, and data (Nakkiran et al., 2021).
:::
:::
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[Implicit Bias]{.dtitle}

[which exact fit gradient descent finds]{.dsub}
:::
:::

::: {.slide title="Gradient Descent from Zero Finds the Minimum-Norm Fit"}
[Implicit bias · least squares]{.kicker}

::: {.cols .vc}
::: {.col}
$L(\mathbf{w}) = \tfrac12\|\mathbf{H}\mathbf{w} - \mathbf{y}\|^2$,
$\mathbf{H} \in \mathbb{R}^{n \times h}$, $h > n$.

I. Gradient $\mathbf{H}^\top(\mathbf{H}\mathbf{w} - \mathbf{y})$: a
combination of the rows.

II. From $\mathbf{w}_0 = \mathbf{0}$: $\mathbf{w}_t = \mathbf{H}^\top
\mathbf{a}_t$, and the limit fits:
$\mathbf{w}^\star = \mathbf{H}^\top(\mathbf{H}\mathbf{H}^\top)^{-1}\mathbf{y}$.

III. Other fits: $\mathbf{w}^\star + \mathbf{v}$, $\mathbf{H}\mathbf{v} = \mathbf{0}$,
$\|\mathbf{w}^\star + \mathbf{v}\|^2 = \|\mathbf{w}^\star\|^2 + \|\mathbf{v}\|^2$.
:::

::: {.col .fig}
![](../img/mdl-mlp-min-norm.svg){width=100%}
:::
:::
:::

::: {.slide title="From Zero, Gradient Descent Lands on the Solution of lstsq"}
[Implicit bias · check]{.kicker}

20 equations, 100 unknowns; 2,000 steps of gradient descent from
$\mathbf{w}_0 = \mathbf{0}$ and from a random $\mathbf{w}_0$:

@!generalization-deep-minimum-norm-interpolation

- Both runs fit every label (residual near 0).
- From zero: distance 0 to the minimum-norm fit $\mathbf{w}^\star$.
- From a random start: an exact fit more than ten times longer.
:::

::: {.slide title="Logistic Gradient Descent Turns Toward the Max-Margin Separator"}
[Implicit bias · classification]{.kicker}

Separable data, labels $\pm 1$, loss $\log(1 + e^{-m})$ with margin
$m = y\,\mathbf{w}^\top\mathbf{x}$: no finite minimizer, $\|\mathbf{w}\| \to \infty$.

![](../img/mdl-mlp-max-margin.svg){width=70%}

::: {.d2l-note}
The direction converges to the maximum-margin separator (Soudry et al.,
2018), slowly: the training error is 0 from step 1.
:::
:::

::: {.slide title="Check: Where Does Gradient Descent from w₀ Stop?"}
[Check]{.kicker}

One equation $w_1 + 2w_2 = 4$, so $\mathbf{h} = (1, 2)$. Gradient descent
starts at $\mathbf{w}_0 = (1, 1)$. Where does it stop, and how long is the
result compared with the minimum-norm fit?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Steps move along $\mathbf{h}$:
$\mathbf{w}_0 + \frac{4 - 3}{5}\mathbf{h} = (1.2, 1.4)$, norm $1.84$.
The minimum-norm fit is $\frac{4}{5}\mathbf{h} = (0.8, 1.6)$, norm $1.79$.
:::
:::
:::

::: {.slide}
::: {.divider}
[04]{.dnum}

[Early Stopping]{.dtitle}

[clean labels first]{.dsub}
:::
:::

::: {.slide title="Clean Labels Are Fit First, Wrong Labels Later"}
[Early stopping · experiment]{.kicker}

::: {.cols .vc}
::: {.col .narrow}
1,000 images, 20% of labels redrawn: about 18% wrong.

- After 20 epochs: 80–90% of the clean labels fit, under 20% of the
  wrong ones. Validation accuracy peaks in this phase.
- Then the wrong labels are memorized, nearly all after 100–160 epochs;
  validation accuracy drops several points.
:::

::: {.col .fig .big}
@!generalization-deep-clean-labels-first
:::
:::
:::

::: {.slide title="Early Stopping Keeps the Clean Fit"}
[Early stopping · rule]{.kicker}

Monitor validation accuracy every epoch; keep the parameters of the best
epoch; stop after *patience* epochs without improvement.

@!generalization-deep-stopping-rule

- About three to seven points of test accuracy, and fewer epochs.
- For least squares, stopping after $t$ steps of size $\eta$ acts like
  weight decay with $\lambda \approx 1/(\eta t)$.
:::

::: {.slide title="Grokking: Generalization Can Arrive Long After Memorization"}
[Early stopping · limits]{.kicker}

![](../img/mdl-mlp-grokking.svg){width=70%}

::: {.d2l-note .warn}
Addition modulo 97 with weight decay: 100% training accuracy within about
a thousand steps, validation near chance ten times as long, full accuracy
two to three orders of magnitude later (Power et al., 2022). A patience rule
stops far too early.
:::
:::

::: {.slide title="Check: How Many Training Labels Does the Memorizing Network Get Wrong?"}
[Check]{.kicker}

20% of 1,000 labels are redrawn uniformly from 10 classes. After 200 epochs
the network reproduces nearly every training label. Against the **true** labels,
what is its training accuracy?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** About 82%. A redrawn label is correct with probability 1/10, so
$0.2 \times 0.9 = 18\%$ of the labels are wrong, and the network reproduces
them.
:::
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

::: {.cols}
::: {.col}
- **Capacity:** the MLP fits random labels (100% train, 10% test); the VC
  term is $\sqrt{203{,}530/60{,}000} \approx 1.8$.
- **Double descent:** random features peak at $h = n$ (85–90% error) and
  fall to 18–23% at $h = 10n$; $s_\textrm{min} \approx 0$ at the peak.
:::

::: {.col}
- **Implicit bias:** gradient descent from zero finds the minimum-norm
  fit; logistic loss approaches the maximum margin.
- **Early stopping:** clean labels first; stopping at the best validation
  epoch gains about three to seven points.
- **Open:** a theory that predicts a deep network's test error.
:::
:::

**Next:** dropout, a perturbation during training that may change what the
network prefers to fit.
:::
