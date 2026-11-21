```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Concise Implementation of Linear Regression
:label:`sec_linear_concise`

In :numref:`sec_linear_scratch`, we built every piece of linear regression from raw tensors—parameter allocation, the affine forward pass, the squared loss, reverse-mode differentiation, the minibatch generator, and the SGD update step. In :numref:`sec_oo-design`, we organized those components into reusable object-oriented base classes (`d2l.Module`, `d2l.DataModule`, and `d2l.Trainer`) and verified that `LinearRegressionScratch` trained through `trainer.fit(model, data)` reproduced our exact benchmark results.

Writing and testing raw parameter tensors and update rules by hand was straightforward for a 3-parameter linear model, but modern architectures like ResNet-50 (:numref:`sec_resnet`) or GPT (:numref:`sec_gpt`) manage tens of millions to billions of parameters across hundreds of layers and train with stateful adaptive optimizers. Deep learning frameworks—PyTorch :cite:`Paszke.Gross.Massa.ea.2019`, JAX :cite:`Frostig.Johnson.Leary.2018` (with Flax and Optax), TensorFlow :cite:`Abadi.Barham.Chen.ea.2016`, and MXNet :cite:`Chen.Li.Li.ea.2015`—therefore provide pre-built, hardware-tuned layers, numerically tested loss primitives, and standard optimizers out of the box. In this section, we see how concisely we can express linear regression by plugging those high-level framework building blocks directly into `d2l.Module` and `d2l.Trainer`.

```{.python .input #linear-regression-concise-concise-implementation-of-linear-regression}
%%tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #linear-regression-concise-concise-implementation-of-linear-regression}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #linear-regression-concise-concise-implementation-of-linear-regression}
%%tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #linear-regression-concise-concise-implementation-of-linear-regression}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## The Fully Connected Layer

In our concise implementation, a subclass `LinearRegression(d2l.Module)` wraps a single built-in fully connected layer (`Dense` or `Linear` / `LazyLinear`) with one output unit, initializing its weights from $\mathcal{N}(0, 0.01^2)$ and its bias to zero.

:begin_tab:`pytorch`
PyTorch's `nn.LazyLinear(1)` requires only the output dimension at construction time and infers the number of input features $d$ automatically on the very first forward pass!
:end_tab:

:begin_tab:`mxnet`
Gluon's `nn.Dense(1)` likewise infers its input dimension lazily when the first minibatch passes through it.
:end_tab:

:begin_tab:`tensorflow`
Keras's `layers.Dense(1)` also infers its input width lazily on the first call.
:end_tab:

:begin_tab:`jax`
Flax NNX layers allocate parameters eagerly upon construction, so `nnx.Linear` takes both the input dimension (`num_inputs`, which we set to `2`) and the output dimension (`1`).
:end_tab:

```{.python .input #linear-regression-concise-defining-the-model-1}
%%tab pytorch
class LinearRegression(d2l.Module):  #@save
    """Linear regression built from framework components."""
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(1)

    def weight_init(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.01)
            nn.init.zeros_(module.bias)
```

```{.python .input #linear-regression-concise-defining-the-model-1}
%%tab mxnet
class LinearRegression(d2l.Module):  #@save
    """Linear regression built from framework components."""
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Dense(1)
        self.net.initialize(init.Normal(sigma=0.01))
```

```{.python .input #linear-regression-concise-defining-the-model-1}
%%tab tensorflow
class LinearRegression(d2l.Module):  #@save
    """Linear regression built from framework components."""
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        init = tf.initializers.RandomNormal(stddev=0.01)
        self.net = tf.keras.layers.Dense(1, kernel_initializer=init)
```

```{.python .input #linear-regression-concise-defining-the-model-1}
%%tab jax
class LinearRegression(d2l.Module):  #@save
    """Linear regression built from framework components."""
    def __init__(self, num_inputs, lr, rngs=None):
        super().__init__()
        self.save_hyperparameters(ignore=['rngs'])
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        self.net = nnx.Linear(
            num_inputs, 1, kernel_init=nnx.initializers.normal(0.01),
            rngs=rngs)
```

Notice that `LinearRegression` does not even need to define `forward`: it
inherits `Module.forward` from :numref:`sec_oo-design`, which delegates directly
to `self.net(X)`.

## The Loss Function

Instead of writing $(\hat{y} - y)^2 / 2$ by hand, we delegate `loss` to each framework's built-in mean squared error primitive (`nn.MSELoss` in PyTorch, `gluon.loss.L2Loss` in MXNet, `MeanSquaredError` in Keras, or `optax.squared_error` in JAX). Note that PyTorch, TensorFlow, and Optax omit the cosmetic $\frac{1}{2}$ factor in their squared error definitions (and we multiply Gluon's `L2Loss`, which includes it, by $2$), so these losses are twice our scratch values and level off at $\sigma^2 = 10^{-4}$ rather than $\sigma^2 / 2$:

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
    return 2 * fn(y_hat, y).mean()  # L2Loss includes the 1/2
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
    return optax.squared_error(y_hat, y).mean()
```

## The Optimizer

Similarly, `configure_optimizers` returns each framework's built-in minibatch SGD instance (`torch.optim.SGD`, `gluon.Trainer`, `tf.keras.optimizers.SGD`, or `optax.sgd`), automatically wired to the module's registered parameters and learning rate `self.lr`:

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

## Training with Framework Components

With `LinearRegression` defined, training reduces to instantiating the data module, the model (with learning rate $\eta = 0.015$, halved from our scratch rate $\eta = 0.03$ to compensate for the doubled MSE gradient), and `d2l.Trainer(max_epochs=10)`, then calling `trainer.fit(model, data)`:

```{.python .input #linear-regression-concise-training-1}
%%tab pytorch
data = d2l.SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
model = LinearRegression(lr=0.015)
model(data.X[:1])  # Creates the parameters of the lazy layer
model.net.apply(model.weight_init)
trainer = d2l.Trainer(max_epochs=10)
trainer.board.yscale = 'log'
trainer.fit(model, data)
```

```{.python .input #linear-regression-concise-training-1}
%%tab mxnet, tensorflow
data = d2l.SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
model = LinearRegression(lr=0.015)
trainer = d2l.Trainer(max_epochs=10)
trainer.board.yscale = 'log'
trainer.fit(model, data)
```

```{.python .input #linear-regression-concise-training-1}
%%tab jax
data = d2l.SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
model = LinearRegression(2, lr=0.015)
trainer = d2l.Trainer(max_epochs=10)
trainer.board.yscale = 'log'
trainer.fit(model, data)
```

```{.python .input #linear-regression-concise-training-4}
print(f"validation loss: {trainer.history['val_loss'][-1]:.2e}")
```

The final validation loss lands within roughly $15\%$ of the theoretical noise
floor $\sigma^2 = 10^{-4}$.

The validation and training loss curves drop right down to the $10^{-4}$ noise floor, mirroring our from-scratch implementation. Let us add a small helper method `get_w_b` to extract the learned weight tensor and bias from the framework layer and compare them against $(\mathbf{w}^*, b^*)$:

```{.python .input #linear-regression-concise-training-2}
%%tab pytorch
@d2l.add_to_class(LinearRegression)  #@save
def get_w_b(self):
    return self.net.weight.detach(), self.net.bias.detach()
```

```{.python .input #linear-regression-concise-training-2}
%%tab mxnet
@d2l.add_to_class(LinearRegression)  #@save
def get_w_b(self):
    return self.net.weight.data(), self.net.bias.data()
```

```{.python .input #linear-regression-concise-training-2}
%%tab tensorflow
@d2l.add_to_class(LinearRegression)  #@save
def get_w_b(self):
    return self.net.kernel, self.net.bias
```

```{.python .input #linear-regression-concise-training-2}
%%tab jax
@d2l.add_to_class(LinearRegression)  #@save
def get_w_b(self):
    return self.net.kernel[...], self.net.bias[...]
```

```{.python .input #linear-regression-concise-training-3}
w, b = model.get_w_b()
w_error = d2l.numpy(d2l.reshape(w, data.w.shape) - data.w)
b_error = d2l.numpy(b - data.b)
print('error in w:', ', '.join(f'{e:.1e}' for e in w_error))
print('error in b:', ', '.join(f'{e:.1e}' for e in b_error))
```

Once again, the errors in $\hat{\mathbf{w}}$ and $\hat{b}$ stay within about a
thousandth, matching our scratch run because the dominant error sources (label
noise across $1{,}000$ examples and the remaining distance after $10$ epochs) are
identical regardless of who wrote the layer.

Everything we checked in the past three sections leaned on knowing the
ground-truth parameters, a luxury we never have on real datasets; in
:numref:`sec_generalization_basics`, we tackle how to judge models strictly from
held-out data.

## Summary

- **High-level framework layers:** A single built-in dense layer (`nn.LazyLinear(1)`, `nn.Dense(1)`, `layers.Dense(1)`, or `nnx.Linear(num_inputs, 1)`) allocates parameters, registers them with the module, and implements the affine forward pass $\mathbf{X}\mathbf{w} + b$.
- **Loss scaling and learning-rate coupling:** Built-in mean squared error primitives omit the $\frac{1}{2}$ factor used in our scratch loss, doubling both the loss value (noise floor $\sigma^2 = 10^{-4}$) and its gradient; halving the learning rate ($\eta = 0.015$) reproduces the exact same parameter updates.
- **Drop-in compatibility with `Trainer`:** Because `LinearRegression` implements the `d2l.Module` interface (`loss` and `configure_optimizers`, while inheriting `forward`), `d2l.Trainer` fits it without a single line of change to the training loop.

## Exercises

1. [code] **A layer without a bias.** Build `LinearRegression` with a
   layer that has no bias (every framework's dense layer has an argument
   that turns the bias off), and train it as in this section.
    1. Before training, predict the value at which the validation loss
       levels off, from the expected loss
       :eqref:`eq_linreg-scratch-expected-loss` doubled for the mean
       squared error. Then check your prediction.
    1. Predict what the weights converge to and how far they end from
       $\mathbf{w}^*$. Since the features have mean zero, the missing bias
       acts on the weights like label noise of standard deviation
       $b^* = 4.2$; it enters both the estimation error of
       :numref:`subsec_synthetic-regression-data-recovery` and the noise
       ball of minibatch SGD (:numref:`sec_linear_regression`). Compare
       your prediction with the errors of the trained weights.
   :exlabel:`ex_linear-regression-concise-no-bias`

1. [code] **Several outputs.** Change the output width of the layer in
   `LinearRegression` from 1 to 2. After constructing `data`, append to
   `data.y` a second label column
   $y_2 = \mathbf{x}^\top \mathbf{w}_2^* + b_2^* + \epsilon_2$, computed
   from `data.X` with generating parameters of your choice and noise of
   the same size.
    1. Show that the squared loss summed over the two outputs splits into
       two independent linear regressions, one per output. Conclude that
       the parameters learned for one output do not depend on the labels of
       the other.
    1. The framework losses average over every entry of the
       $|\mathcal{B}| \times 2$ matrix of errors of a minibatch
       $\mathcal{B}$, not only over its examples. State by which factor
       the gradient of each output's parameters shrinks relative to the
       single-output model, and which learning rate reproduces the
       single-output training run. Confirm this by training both models
       from the same initial parameters on the same minibatches.

1. [code] **Feature scale and step size.** Multiply the first column of
   `data.X` by 100 after constructing `data`, leaving `data.y` unchanged,
   so that the generating weight of that feature becomes $0.02$. Recall
   from exercise :exref:`ex_linear-regression-scratch-stability` that
   gradient descent diverges once $\eta$ exceeds $2/\mu_{\max}$, where
   $\mu_{\max}$ is the largest eigenvalue of the Hessian $\mathbf{H}$ of
   the loss.
    1. Compute $\mathbf{H}$ for `LinearRegression.loss` and estimate its
       eigenvalues before and after the rescaling.
    1. Predict the largest learning rate for which training does not
       diverge, before and after the rescaling, and check both predictions
       by training at learning rates around them.
    1. At a learning rate just below the new limit, predict how many
       updates the second weight needs to reduce its error by a factor of
       100, and compare the prediction with a measurement. Then standardize
       the rescaled feature as in :numref:`sec_pandas`, and show that
       `lr=0.015` works again.

1. [code] **Optimization versus estimation error.** ● Train
   `LinearRegression` with `lr=0.015` and `max_epochs=10` on
   `SyntheticRegressionData` with `num_train` set to each of
   $5, 50, 500, 5{,}000$ and `batch_size=min(32, num_train)`. For each run,
   compare the distance $\|\hat{\mathbf{w}} - \mathbf{w}^*\|_2$ of the
   trained weights from the generating weights $\mathbf{w}^*$, stored in
   `data.w`, with the same distance for the least-squares solution on the
   same training data, computed as in the recovery subsection
   (:numref:`subsec_synthetic-regression-data-recovery`).
    1. Count the parameter updates in each run, and explain the ratio of
       the two errors at each `num_train`.
    1. Find the number of epochs after which the run with `num_train=5`
       comes within a factor of two of its least-squares error.

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
[Dive into Deep Learning · §2.5]{.kicker}

Concise Implementation of Linear Regression<br>
**framework layers · lazy shape inference · MSE loss scaling · built-in optimizers · Trainer reuse**
:::
:::

::: {.slide title="Why Move From Scratch Code to Concise Framework APIs?"}
[Scaling Up to Deep Networks]{.kicker}

Our from-scratch loop allocated and updated $3$ scalar parameters (`w` and `b`). Modern architectures like ResNet-50 or Llama manage millions to billions of parameters across hundreds of layers:

| Component | From-Scratch Implementation (§2.3–§2.4) | Concise Framework Abstraction (§2.5) |
|---|---|---|
| **Parameters & Forward Pass** | `init_params()` + `d2l.matmul(X, w) + b` | Single fully connected layer (`LazyLinear(1)` / `Dense(1)` / `nnx.Linear(2, 1)`) |
| **Loss Function** | `squared_loss`: $\frac{1}{2|\mathcal{B}|}\sum_i (\hat{y}^{(i)} - y^{(i)})^2$ | Built-in MSE (`nn.MSELoss` / `L2Loss` / `MeanSquaredError` / `optax.squared_error`) |
| **Parameter Updater** | Manual loop `param -= lr * param.grad` | Built-in optimizer (`torch.optim.SGD` / `gluon.Trainer` / `keras.optimizers.SGD` / `optax.sgd`) |
| **Data & Epoch Loop** | `d2l.SyntheticRegressionData` + `d2l.Trainer` | Reuses the exact same `d2l.SyntheticRegressionData` and `d2l.Trainer` |
:::

::: {.slide title="Concise Model: A Single Fully Connected Layer"}
[Framework Layer API]{.kicker}

In `LinearRegression(d2l.Module)`, a single built-in dense layer allocates weights and bias and implements the affine forward pass `self.net(X)`:

@linear-regression-concise-defining-the-model-1

- **Inherited `forward`:** Because `d2l.Module.forward(X)` delegates to `self.net(X)` by default (§2.4), `LinearRegression` needs no custom `forward` method.
- **Lazy input dimension inference (PyTorch / MXNet / Keras):** `nn.LazyLinear(1)` and `Dense(1)` require only the output width `1` at construction time, inferring input width $d = 2$ automatically when the first minibatch passes through!
- **Eager shape specification (JAX / Flax NNX):** `nnx.Linear(num_inputs, 1, ...)` allocates parameter arrays immediately upon construction using explicit PRNG `rngs`.
:::

::: {.slide title="Framework MSE Loss and Halving the Learning Rate"}
[Loss Scaling & Hyperparameter Coupling]{.kicker}

Standard framework MSE primitives (`nn.MSELoss`, Keras `MeanSquaredError`, `optax.squared_error`, and $2 \times$ Gluon `L2Loss`) compute the plain mean squared error *without* our factor of $\frac{1}{2}$:

@linear-regression-concise-defining-the-loss-function

| Objective Convention | Formula | Minibatch Gradient | Noise Floor ($\sigma = 0.01$) | Equivalent Learning Rate |
|---|---|---|---|---|
| **Scratch loss (with $1/2$)** | $\frac{1}{2|\mathcal{B}|}\|\hat{\mathbf{y}} - \mathbf{y}\|_2^2$ | $\mathbf{g}$ | $\frac{1}{2}\sigma^2 = 5 \times 10^{-5}$ | $\eta = 0.03$ |
| **Framework MSE (no $1/2$)** | $\frac{1}{|\mathcal{B}|}\|\hat{\mathbf{y}} - \mathbf{y}\|_2^2$ | $2\mathbf{g}$ | $\sigma^2 = 10^{-4}$ | $\eta = 0.03 / 2 = 0.015$ |

::: {.d2l-note .rule}
Multiplying a loss function by any constant $c > 0$ multiplies all its gradients by $c$. Halving the learning rate to $\eta = 0.015$ makes every framework SGD step identical to our scratch step at $\eta = 0.03$.
:::
:::

::: {.slide title="Configuring the Built-In Framework Optimizer"}
[Optimizer Abstraction]{.kicker}

In `configure_optimizers`, we return the framework's built-in SGD optimizer wired to the module's registered parameters and learning rate `self.lr`:

@linear-regression-concise-defining-the-optimization-algorithm

- **Automatic parameter discovery:** `self.parameters()` (PyTorch), `self.collect_params()` (MXNet), `model.trainable_variables` (Keras), and `nnx.Param` (Flax NNX) traverse the module tree automatically—scaling effortlessly from 1 layer to 1,000 layers.
- **One-line optimizer swap:** Switching from minibatch SGD to Adam or AdamW in later chapters requires changing only this single return statement.
:::

::: {.slide title="Training and Parameter Recovery With Framework Parts"}
[End-to-End Concise Execution]{.kicker}

Fitting `LinearRegression(lr=0.015)` with `d2l.Trainer(max_epochs=10)` reproduces the exact $7\times$-per-epoch convergence and $O(10^{-4})$ parameter recovery:

::: {.cols .vc}
::: {.col}
@-linear-regression-concise-training-1

@!linear-regression-concise-training-4

- **Floor at $\sigma^2 = 10^{-4}$:** Because framework MSE omits the $\frac{1}{2}$ factor, the loss settles at $\sigma^2 = 10^{-4}$ (whose square root $\sqrt{10^{-4}} = 0.01 = \sigma$ gives the root-mean-square prediction error in label units).
:::

::: {.col .fig}
@!linear-regression-concise-training-1
:::
:::
:::

::: {.slide title="Extracting and Verifying the Learned Layer Parameters"}
[Inspecting Framework Layer Weights]{.kicker}

To inspect the learned weights and bias inside `self.net`, we attach a `get_w_b` helper and compare $(\hat{\mathbf{w}}, \hat{b})$ against $\mathbf{w}^* = (2, -3.4)^\top$ and $b^* = 4.2$:

@linear-regression-concise-training-2

@!linear-regression-concise-training-3

::: {.d2l-note}
Both weights and bias match $(\mathbf{w}^*, b^*)$ to within a few times $10^{-4}$ ($\approx \sigma / \sqrt{n}$), confirming that the high-level framework abstractions execute the exact same optimization trajectory as our hand-written loop.
:::
:::

::: {.slide title="Concept Check: Loss Scaling, Missing Bias & Multi-Output"}
[Check Your Understanding]{.kicker}

Test your understanding of framework layer and loss conventions on these three questions:

1. **Why the concise loss plateaus twice as high:** Why does our concise run plateau near $10^{-4}$ while our scratch run in §2.3–§2.4 plateaued near $5 \times 10^{-5}$ on the exact same synthetic generator?
2. **Omitting the bias (`bias=False`):** If you disable the bias in `LinearRegression` on our dataset ($b^* = 4.2$, zero-mean features $\mathbf{x} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_2)$), what value does the validation MSE plateau at?
3. **Two-output MSE scaling:** If `LinearRegression` predicts $2$ outputs instead of $1$, framework MSE averages over all $2|\mathcal{B}|$ entries of the residual matrix. How does that affect each output's gradient and equivalent learning rate?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. Framework MSE computes $\frac{1}{|\mathcal{B}|}\|\hat{\mathbf{y}} - \mathbf{y}\|_2^2$ without our $\frac{1}{2}$ factor, so at the true parameters its expected value is $\mathbb{E}[\epsilon^2] = \sigma^2 = (0.01)^2 = 10^{-4}$ rather than $\frac{1}{2}\sigma^2 = 5 \times 10^{-5}$.
2. Because $\mathbb{E}[\mathbf{x}] = \mathbf{0}$, the weights still converge near $\mathbf{w}^* = (2, -3.4)^\top$, while the unmodeled intercept $b^* = 4.2$ remains in every residual, raising the MSE floor to $(b^*)^2 + \sigma^2 = 4.2^2 + 10^{-4} \approx 17.64$.
3. Dividing by $2|\mathcal{B}|$ instead of $|\mathcal{B}|$ halves the gradient with respect to each output's parameters, so doubling the learning rate to $\eta = 0.03$ restores the single-output step size.
:::
:::

::: {.slide title="Section 2.5 Summary & Key Takeaways"}
[Summary]{.kicker}

| Component | From-Scratch (§2.3–§2.4) | Concise Framework (§2.5) | Key Invariant / Takeaway |
|---|---|---|---|
| **Model & Parameters** | `LinearRegressionScratch` (`self.w`, `self.b`) | `LinearRegression` (`LazyLinear(1)` / `Dense(1)` / `nnx.Linear`) | Inherits `Module.forward(X) -> self.net(X)` |
| **Loss Function** | $\frac{1}{2|\mathcal{B}|}\|\hat{\mathbf{y}} - \mathbf{y}\|_2^2$ | `MSELoss` / $2\times\text{L2Loss}$ / `MeanSquaredError` / `squared_error` | Omits $1/2 \implies$ loss & gradient are $2\times$ larger |
| **Optimizer & Step Size** | Custom `SGD` at $\eta = 0.03$ | Built-in `SGD` at $\eta = 0.015$ | Halving $\eta$ preserves the exact parameter update $-\eta\nabla L$ |
| **Noise Floor & Recovery** | Floor $\sigma^2/2 = 5 \times 10^{-5}$ | Floor $\sigma^2 = 10^{-4}$ ($\sqrt{\text{MSE}} = \sigma = 0.01$) | Both recover $(\mathbf{w}^*, b^*)$ within $\sim 3\sigma/\sqrt{n} \approx 10^{-3}$ |
:::
