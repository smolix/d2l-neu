```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Generalization in Deep Learning
:label:`sec_generalization_deep`

In :numref:`sec_generalization_basics`, a degree-19 polynomial with $20$ parameters interpolated $20$ noisy data points and blew up on test data. By that classical intuition, our Fashion-MNIST MLP in :numref:`sec_mlp-implementation`—with $203{,}530$ parameters for $60{,}000$ training images ($3.4$ parameters per image)—ought to overfit severely, especially if trained until it interpolates every single training image. Yet even when driven to $100\%$ training accuracy, it still generalizes to $\sim 87\%$ on unseen images!

Understanding why heavily overparameterized networks generalize is one of the central open problems of modern machine learning. In this section, we combine controlled experiments with exact results on tractable models to uncover four key pieces of the puzzle:
1. **Random-label memorization:** Standard MLPs easily reach $100\%$ training accuracy on completely random labels :cite:`zhang2021understanding`, proving that classical distribution-free capacity bounds (like VC dimension) cannot explain their generalization on real labels.
2. **Double descent:** Sweeping model width past the interpolation threshold $h = n$ reveals a spike in test error at $h = n$ (where the feature matrix is nearly singular) followed by a *second descent* as extra parameters allow smaller-norm interpolators :cite:`Belkin.Hsu.Ma.ea.2019`.
3. **Implicit bias of gradient descent:** Among infinitely many zero-loss solutions, gradient descent automatically selects the minimum-norm interpolator under squared loss and the maximum-margin separator under logistic loss :cite:`Soudry.Hoffer.Nacson.ea.2018`.
4. **Clean-first learning and early stopping:** On noisy datasets, gradient descent fits clean structured patterns long before memorizing corrupted labels :cite:`Arpit.Jastrzebski.Ballas.ea.2017`, making *early stopping* a potent regularizer.

```{.python .input #generalization-deep-fitting-random-labels-1}
%%tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #generalization-deep-fitting-random-labels-1}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #generalization-deep-fitting-random-labels-1}
%%tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #generalization-deep-fitting-random-labels-1}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## Memorization and Capacity

### Random Labels
:label:`subsec_random_labels`

To test raw memorization capacity following :citet:`zhang2021understanding`, we draw $2{,}000$ Fashion-MNIST training images ($1{,}000$ for training, $1{,}000$ for validation) and $2{,}000$ test images, define `ArrayData` and `Tracker` helpers along with a He-initialized 256-unit `MLP`, and train for 200 epochs once on the true labels (`fraction=0`) and once on completely random labels (`fraction=1`):

```{.python .input #generalization-deep-fitting-random-labels-2}
data = d2l.FashionMNIST(batch_size=2000)
X, y = next(iter(data.train_dataloader()))          # 2,000 random images
X_test, y_test = next(iter(data.val_dataloader()))  # the first 2,000
X, y, X_test, y_test = (d2l.numpy(a) for a in (X, y, X_test, y_test))
X, X_test = X.reshape(2000, -1), X_test.reshape(2000, -1)
X_train, y_train, X_val, y_val = X[:1000], y[:1000], X[1000:], y[1000:]
```

Two lightweight helpers streamline the experiments ahead: `ArrayData` wraps
NumPy arrays into shuffled minibatches, and `Tracker` subclasses `Trainer`
to log epoch-by-epoch accuracy across an arbitrary dictionary of named
datasets.

```{.python .input #generalization-deep-fitting-random-labels-3}
class ArrayData(d2l.DataModule):  #@save
    """Minibatches of the arrays X and y; training batches are shuffled."""
    def __init__(self, X, y, batch_size=64):
        super().__init__()
        self.save_hyperparameters()

    def get_dataloader(self, train):
        return self.get_tensorloader(
            (d2l.tensor(self.X), d2l.tensor(self.y)), train)

class Tracker(d2l.Trainer):  #@save
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

Our model is the same 256-unit ReLU MLP from
:numref:`sec_mlp-implementation`, upgraded with He initialization
(:numref:`subsec_he_init`) in place of `std=0.01` so training takes off
immediately from epoch 1 (with zero biases and pre-flattened inputs).

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


```{.python .input #generalization-deep-fitting-random-labels-5}
def corrupt(y, fraction, rng):  #@save
    """Replace each label, with probability fraction, by a random class."""
    redraw = rng.random(len(y)) < fraction
    y = y.copy()
    y[redraw] = rng.integers(0, 10, redraw.sum())
    return y

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


Look at what the two runs prove:
- On **true labels**, training accuracy hits $100\%$ within 60–90 epochs while test accuracy plateaus above $80\%$ (even on just $1{,}000$ training images).
- On **completely random labels**, the exact same architecture and optimizer still reach $100\%$ training accuracy (taking slightly longer, $\sim 120$–$150$ epochs) while test accuracy remains at $10\%$ random chance.

### Why Distribution-Free Capacity Bounds Fail, and Inductive Bias

Because our MLP achieves zero training error ($\epsilon_{\mathcal{S}}(f) = 0$) on random labels where the true population error is $\epsilon(f) = 0.90$, *any* distribution-free bound of the form $\epsilon(f) \leq \epsilon_{\mathcal{S}}(f) + B(\mathcal{F}, n, \delta)$ that depends only on the hypothesis class $\mathcal{F}$ must have $B(\mathcal{F}, n, \delta) \geq 0.90$—making it completely vacuous on real labels as well (Exercise :exref:`ex_generalization-deep-complexity`).

By the *no free lunch theorem* :cite:`Wolpert.1996`, no learner can generalize across all distributions; generalization depends on whether the learner's *inductive bias*—which interpolating function it prefers among infinitely many—matches the structure of real data (just as a 1-nearest-neighbor classifier interpolates $100\%$ of its training data yet generalizes well whenever nearby inputs share labels :cite:`Cover.Hart.1967`).

## Double Descent
:label:`subsec_double_descent`

### The Interpolation Threshold and Conditioning

What happens when we increase model width $h$ right past the sample size $n$? Let us test a tractable *random-features* model :cite:`Rahimi.Recht.2007`: freeze a He-initialized first layer $\mathbf{W}^{(1)} \in \mathbb{R}^{784 \times 10000}$ on our $n = 1{,}000$ centered training images to form random ReLU features $\mathbf{H} = \mathrm{ReLU}((\mathbf{X} - \bar{\mathbf{x}})\mathbf{W}^{(1)})$, and fit the output weights $\mathbf{W}^{(2)} \in \mathbb{R}^{h \times 10}$ on the first $h$ features by minimum-norm least squares (`np.linalg.lstsq`) across widths $h \in [50, 10{,}000]$:

```{.python .input #generalization-deep-beyond-the-interpolation-threshold-1}
W1 = rng.normal(0, np.sqrt(2 / 784), (784, 10000))  # He-initialized, frozen
mean = X_train.mean(axis=0)                         # The mean training image
H = np.maximum((X_train - mean) @ W1, 0)
H_test = np.maximum((X_test - mean) @ W1, 0)
Y = np.eye(10)[y_train]                             # One-hot labels

def error(H, W2, y):
    """Fraction of examples whose largest output is not their label."""
    return (np.argmax(H @ W2, axis=1) != y).mean()
```

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


This is the celebrated *double-descent* curve :cite:`Belkin.Hsu.Ma.ea.2019`:
1. **Classical U-curve ($h < n$):** Test error drops from $\sim 30\%$ at $h = 50$ to $\sim 23\%$ around $h = 200$, then shoots up to $\sim 90\%$ right at the *interpolation threshold* $h = n = 1{,}000$.
2. **Second descent ($h > n$):** As $h$ grows past $n = 1{,}000$ up to $10{,}000$, every model interpolates the training set perfectly, yet test error plunges back down to $\sim 20\%$—beating the best underparameterized model!

Why does test error spike so violently at $h = n$ and heal for $h \gg n$? Let us inspect the smallest singular value $s_{\min}$ of $\mathbf{H}_{:, :h}$ and the Frobenius norm $\|\mathbf{W}^{(2)}\|_{\textrm{F}}$:

```{.python .input #generalization-deep-conditioning-at-the-threshold-1}
for h in (200, 900, 1000, 1100, 10000):
    print(f'h = {h:5d}: smallest singular value {s_min[h]:.1e}, '
          f'norm of W2 {norm[h]:6.1f}')
```


At $h = n = 1{,}000$, $\mathbf{H}$ is a square $1000 \times 1000$ matrix with a tiny smallest singular value $s_{\min}$, so the unique exact inverse $\mathbf{H}^{-1}\mathbf{Y}$ divides label noise by $s_{\min}$ and explodes $\|\mathbf{W}^{(2)}\|_{\textrm{F}}$ into the hundreds! For $h \gg n$, the extra columns open up an entire $(h - n)$-dimensional subspace of exact interpolators, allowing the *minimum-norm* solution to avoid poorly conditioned directions and shrink $\|\mathbf{W}^{(2)}\|_{\textrm{F}}$ all the way down to $\sim 2$. In fact, holding $h = 1{,}000$ fixed, doubling the training set from $n = 500$ to $n = 1{,}000$ lands the unregularized fit squarely on the $h = n$ cliff and drives test error *up* from $<30\%$ to $\sim 90\%$:

```{.python .input #generalization-deep-conditioning-at-the-threshold-2}
for n in (500, 1000):
    W2 = np.linalg.lstsq(H[:n, :1000], Y[:n], rcond=None)[0]
    test_error = error(H_test[:, :1000], W2, y_test)
    print(f'{n} training images: test error {test_error:.2f}')
```


## Implicit Bias of Gradient Descent
:label:`subsec_implicit_regularization`

The second descent above relied on `lstsq` picking the *minimum-norm* interpolator among infinitely many zero-loss solutions. Remarkably, gradient descent has that exact preference built in—an *implicit bias* (or *implicit regularization*) that operates even without any explicit penalty term.

### Minimum-Norm Interpolation under Squared Loss

For an overparameterized linear system $\mathbf{H}\mathbf{w} = \mathbf{y}$ ($h > n$, full row rank), the unique minimum-norm interpolator is $\mathbf{w}^\star = \mathbf{H}^\top(\mathbf{H}\mathbf{H}^\top)^{-1}\mathbf{y}$.

**Proposition.** Gradient descent on $L(\mathbf{w}) = \frac{1}{2}\|\mathbf{H}\mathbf{w} - \mathbf{y}\|_2^2$ initialized at $\mathbf{w}_0 = \mathbf{0}$ (with a converging step size) converges to the minimum-norm interpolator $\mathbf{w}^\star$.

**Proof.** Every gradient $\nabla L(\mathbf{w}) = \mathbf{H}^\top(\mathbf{H}\mathbf{w} - \mathbf{y})$ lies in the row space of $\mathbf{H}$. Starting at $\mathbf{w}_0 = \mathbf{0}$, every iterate $\mathbf{w}_t$—and hence the limit $\mathbf{w}_{\infty} = \mathbf{H}^\top\mathbf{a}$—remains in the row space of $\mathbf{H}$ (:numref:`fig_mdl-mlp-min-norm`). Since $\mathbf{H}\mathbf{w}_{\infty} = \mathbf{H}\mathbf{H}^\top\mathbf{a} = \mathbf{y}$, we have $\mathbf{a} = (\mathbf{H}\mathbf{H}^\top)^{-1}\mathbf{y}$ and $\mathbf{w}_{\infty} = \mathbf{w}^\star$. Any other interpolator $\mathbf{w}^\star + \mathbf{v}$ has $\mathbf{H}\mathbf{v} = \mathbf{0}$ (so $\mathbf{v} \perp \mathbf{w}^\star$) and strictly larger norm $\|\mathbf{w}^\star\|_2^2 + \|\mathbf{v}\|_2^2$. $\blacksquare$

![Gradient descent on an underdetermined linear equation $\mathbf{h}^\top\mathbf{w} = y$ moves strictly perpendicular to the solution hyperplane, so starting from $\mathbf{0}$ (blue) lands at the minimum-norm solution $\mathbf{w}^\star$, whereas starting from $\mathbf{w}_0 \neq \mathbf{0}$ (orange) preserves the null-space component of $\mathbf{w}_0$.](../img/mdl-mlp-min-norm.svg)
:label:`fig_mdl-mlp-min-norm`

Let us verify this proposition numerically on $20$ equations in $100$ unknowns, comparing $\mathbf{w}_0 = \mathbf{0}$ against a random initialization:

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


### Maximum-Margin Classification under Logistic Loss

On linearly separable binary classification data ($y_i \in \{-1, +1\}$) under the logistic (cross-entropy) loss $\sum_i \log(1 + \exp(-y_i \mathbf{w}^\top\mathbf{x}_i))$, $\|\mathbf{w}_t\|_2 \to \infty$ as $t \to \infty$, and :citet:`Soudry.Hoffer.Nacson.ea.2018` proved that the *direction* $\mathbf{w}_t / \|\mathbf{w}_t\|_2$ converges (at a $1/\log t$ rate) to the hard-margin SVM *maximum-margin separator* (:numref:`fig_mdl-mlp-max-margin`), because support vectors with the smallest margins dominate the exponentially decaying tail of $\log(1 + e^{-m})$ (see Exercise :exref:`ex_generalization-deep-implicit-bias` for how layer rescaling complicates parameter-norm margins in multi-layer ReLU networks).

![Gradient descent on linearly separable logistic regression rotates $\mathbf{w}_t / \|\mathbf{w}_t\|_2$ toward the maximum-margin hyperplane (left) at a logarithmic rate that continues long after training accuracy reaches $100\%$ (right).](../img/mdl-mlp-max-margin.svg)
:label:`fig_mdl-mlp-max-margin`

## Early Stopping
:label:`subsec_early_stopping`

Even when an MLP has enough capacity to memorize corrupted labels, gradient descent learns *clean structured patterns first* and memorizes *label noise much later* :cite:`Arpit.Jastrzebski.Ballas.ea.2017`. Let us corrupt $20\%$ of the labels in our $1{,}000$-image training set and track accuracy separately on the clean training examples, the mislabeled training examples (scored against their corrupted labels), and the $1{,}000$ clean validation images over 200 epochs:

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


By epoch 20, the MLP gets over $80\%$ of the clean training images right while memorizing fewer than $20\%$ of the corrupted labels—and validation accuracy reaches its peak right in this early window before late-epoch noise memorization drags it down! *Early stopping*—halting training once validation performance fails to improve for a specified *patience* window (Exercise :exref:`ex_generalization-deep-patience`) and restoring the best validation checkpoint—recovers several percentage points of test accuracy while saving compute (and in linear least squares acts analogously to $\ell_2$ weight decay; see Exercise :exref:`ex_generalization-deep-early-stopping-ridge`):

```{.python .input #generalization-deep-stopping-rule}
best = int(np.argmax(acc['validation']))
print(f'best validation accuracy after epoch {best + 1}: '
      f'test accuracy {acc["test"][best]:.3f}')
print(f'after all 200 epochs: test accuracy {acc["test"][-1]:.3f}')
```


One famous exception to early stopping is *grokking* :cite:`Power.Burda.Edwards.ea.2022` (:numref:`fig_grokking`, Exercise :exref:`ex_generalization-deep-grokking`), where networks trained with weight decay on small algorithmic tasks (such as modular addition) memorize the training table rapidly while validation accuracy remains near chance for $100\times$ longer before suddenly jumping to $100\%$.

![Grokking (:citet:`Power.Burda.Edwards.ea.2022`): on modular arithmetic, training accuracy saturates within $\sim 10^3$ steps while validation accuracy jumps to generalization only after $\sim 10^5$ steps.](../img/mdl-mlp-grokking.svg)
:label:`fig_grokking`

## Summary

- **Memorization vs. capacity bounds:** Overparameterized MLPs can memorize $100\%$ of random labels while generalizing on true labels, rendering distribution-free capacity bounds vacuous and placing the burden of explanation on inductive bias.
- **Double descent:** In random-features least squares, test error spikes at the interpolation threshold $h = n$ (where $s_{\min}(\mathbf{H}) \approx 0$ inflates $\|\mathbf{W}^{(2)}\|$) and descends a second time for $h \gg n$ as minimum-norm interpolation spreads weights across redundant features.
- **Implicit bias and early stopping:** Gradient descent from zero converges to the minimum-norm interpolator under squared loss and the maximum-margin direction under logistic loss, and fits clean structure before label noise—making early stopping a simple, effective regularizer.

## Exercises

1. **Limits of complexity measures.** Consider a generalization bound that
   holds for every data distribution: with probability at least
   $1 - \delta$ over the draw of $n$ training examples, every classifier
   $f$ in a class $\mathcal{F}$ satisfies

    $$\epsilon(f) \leq \epsilon_\mathcal{S}(f) + B(\mathcal{F}, n, \delta).$$

    Here $\epsilon(f)$ is the population error, $\epsilon_\mathcal{S}(f)$ is
    the training error, and the term $B$ depends only on $\mathcal{F}$, $n$,
    and $\delta$, as the term $c\sqrt{(v + \log(1/\delta))/n}$ of the VC
    bound does. The text ran the random-label argument against such a bound
    for 1,000 images.
    1. Suppose that training also finds a member of $\mathcal{F}$ with zero
       training error on all 60,000 Fashion-MNIST training images when their
       labels are drawn uniformly at random, as
       :citet:`zhang2021understanding` observed for image classifiers. What
       lower bound on $B(\mathcal{F}, 60{,}000, \delta)$ follows, and what
       does the bound then guarantee for the network trained on the true
       labels?
    1. Many later bounds let the second term depend on the classifier that
       training returns, for example through the norm of its weights or
       through its margins on the training examples. Which step of the
       argument no longer goes through? What must such a quantity do on the
       random-label run if the bound is to say something on the true labels?
   :exlabel:`ex_generalization-deep-complexity`
1. [code] **Patience-based stopping.** `FashionMNIST` uses the Fashion-MNIST
   test set as its validation data, so hold out 10,000 training images as a
   validation set instead. Train an MLP with one hidden layer of 256 units,
   as in :numref:`sec_mlp-implementation`, on the remaining images for a
   fixed number of epochs large enough to overfit, and record the validation loss
   and the test accuracy after every epoch. A patience rule with tolerance
   $\theta$ and patience $k$ stops once the validation loss has failed to
   improve on its best value by more than $\theta$ for $k$ consecutive
   epochs. It then retains the parameters of the epoch with the lowest
   validation loss so far.
    1. Fix $\theta$ and apply the rule to the recorded curves for
       $k \in \{1, 5, 20\}$. For each $k$, report the stopping epoch, the
       retained epoch, and the test accuracy at both.
    1. How does the test accuracy at the retained epoch depend on $k$?
       Relate the dependence to the epoch-to-epoch fluctuation of the
       validation loss.

    *Adapted from Michael Nielsen,
    [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap3.html),
    Chapter 3.*
   :exlabel:`ex_generalization-deep-patience`

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

| model | parameters | per training example | on new data |
|---|---:|---:|---|
| polynomial, degree 19, 20 points | 20 | 1 | error over 1000× the cubic's |
| MLP, 256 hidden units, Fashion-MNIST | 203,530 | 3.4 | about 87%, against about 89% on training images |

- Counting parameters predicts that the MLP memorizes. After 30 epochs it
  had not, but that run was short. What if we train until every training
  label is fit?
- To choose width, training time, and regularization we need to know what
  controls generalization.

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
| Multilayer Perceptrons | hidden layers, universal approximation, activation functions |
| Implementation | an MLP trained on Fashion-MNIST |
| Backpropagation | how the gradients are computed, and at what memory cost |
| Numerical Stability and Initialization | vanishing gradients, Xavier and He initialization |
| **Generalization in Deep Learning** | **memorization, double descent, implicit bias, early stopping** |
| Dropout | random masks, slower memorization of noisy labels |
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

- True labels: over 99% train after 60–90 epochs, about 80–83% test.
- Random labels: over 99% train after 110–150 epochs, about 10% test.

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
out **every** bound that depends only on the class and $n$: on 1,000
random labels the training error is 0 and the population error 0.9, so the
second term is at least 0.9, for true labels too.
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
| 1-nearest neighbor | nearby inputs have the same label |
| MLP + gradient descent | not fully known: **this deck** |

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
- $h < n$: the classical U, best about 23% at a few hundred units.
- $h = n$: about 90% error, that of guessing.
- $h = 10{,}000$: about 20%, a few points **below** the best $h < n$.

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

- After 20 epochs: over 80% of the clean labels fit, under 20% of the
  wrong ones. Validation accuracy peaks in this phase.
- Then the wrong labels are memorized, nearly all after about 100–160 epochs;
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

- Several points of test accuracy as a rule (one to nine in our runs), and
  fewer epochs.
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
- **Double descent:** random features peak at $h = n$ (about 90% error) and
  fall to about 20% at $h = 10n$; $s_\textrm{min} \approx 0$ at the peak.
:::

::: {.col}
- **Implicit bias, for linear models:** gradient descent from zero finds
  the minimum-norm fit; with the logistic loss it approaches the maximum
  margin. Neither result is proved for the MLP.
- **Early stopping:** clean labels first; stopping at the best validation
  epoch gains several points (one to nine in our runs).
- **Open:** a theory that predicts a deep network's test error.
:::
:::

**Next:** if memorizing wrong labels costs accuracy, make memorizing
harder: dropout.
:::
