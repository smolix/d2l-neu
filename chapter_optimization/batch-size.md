# Batch Size
:label:`sec_batch_size`

A batch size is rarely chosen. It is inherited from an earlier experiment
or set by what fits in memory, and then it stays. Ours is a case in point.
`TinyLM` has read 64 sequences per step ever since
:numref:`sec_test_models`, while we swept its learning rate, exchanged its
optimizer and gave it a schedule. In practice the number is first
questioned on the day a second GPU arrives. Two devices process twice the
batch in the time of one step, so the natural move is to double it. Is
that a good move?

Doubling the batch from 64 to 128 sequences doubles the work in every
step. If the run then needs half as many steps to reach the same loss,
nothing is lost: the total work is unchanged, and on twice as many
accelerators the run finishes in half the time. If it needs as many steps
as before, half of the work was spent for nothing. Which of the two happens
decides how fast a model can be trained, because a run can be shared among
only as many accelerators as its batch can feed.

It turns out that both happen, the first at small batches and the second
at large ones, and that the noise in the gradient decides between them. A
minibatch gradient is the gradient of the whole training set plus noise,
and averaging $b$ examples divides the variance of that noise by $b$
(:numref:`sec_sgd`). While the noise is larger than the gradient, a bigger
batch gives a much better direction, the optimizer can take a longer step
along it, and steps are saved. Once the noise is the smaller of the two,
little is left to average away. :citet:`McCandlish.Kaplan.Amodei.ea.2018`
therefore proposed to measure the batch size at which the two are equally
large, and to read from it how large a batch a run can use efficiently.

We measure that batch size on our two test models, derive from it how the
step count and the learning rate should change with the batch, and test
both predictions. The section ends with a procedure for choosing. One case
needs no procedure: as long as a device is not full, a larger batch costs
almost no extra time (:numref:`sec_sgd`). Our question starts where the
work of a step grows in proportion to its batch.

```{.python .input #batch-size}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #batch-size}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #batch-size-1}
%%tab pytorch
import random
```

## The Gradient-Noise Scale

### Signal and Noise

We need one number for the gradient and one for the noise. Write
$\nabla f$ for the gradient of the training loss over the whole dataset at
the current parameters, and $\nabla f_i$ for the gradient of the loss on
example $i$. The per-example gradients scatter around their mean
$\nabla f$, and we measure the scatter by their expected squared distance
from it,

$$\operatorname{tr} \boldsymbol{\Sigma} = \mathbb{E}_i \|\nabla f_i - \nabla f\|^2,$$

the trace of the covariance matrix $\boldsymbol{\Sigma}$ of the per-example
gradients. We call it the *noise power* of one example. A minibatch
gradient of $b$ independent examples, which we write
$\hat{\mathbf{g}}_b$ in this section to keep the batch size in view, has
the noise power $\operatorname{tr} \boldsymbol{\Sigma} / b$
(:numref:`subsec_sgd-variance`). Every minibatch gradient is thus the sum
of a *signal* $\nabla f$, whose squared length $\|\nabla f\|^2$ is the same
at every batch size, and a *noise* whose squared length averages
$\operatorname{tr} \boldsymbol{\Sigma} / b$ and shrinks as the batch grows.
The two have equal power at the batch size

$$b_{\textrm{noise}} = \frac{\operatorname{tr} \boldsymbol{\Sigma}}{\|\nabla f\|^2},$$
:eqlabel:`eq_noise-scale`

called the *gradient-noise scale*. If, say,
$\operatorname{tr} \boldsymbol{\Sigma} = 100$ and $\|\nabla f\|^2 = 1$, the
gradient of a single example carries a hundred times more noise power than
signal, and a batch of $b_{\textrm{noise}} = 100$ examples brings the two
to parity. Note that the noise scale is no constant of the model or of the
data. Both of its ingredients are evaluated at the current parameters, and
both change as the model trains.

Below $b_{\textrm{noise}}$ the noise dominates, and doubling the batch
gives a markedly better direction. Above it the direction is already
nearly right, and a larger batch computes almost the same gradient again,
at a multiple of the work.

A better direction does not save steps by itself, however, and this point
is easy to miss. At an unchanged learning rate a larger batch mostly
reduces the jitter of the path. What saves steps is the larger learning
rate that a less noisy gradient tolerates. The learning rate is thus part
of the question, and we turn to it as soon as we have measured the noise
scale.

### A Two-Batch Estimator
:label:`subsec_batch-size-estimator`

Neither $\operatorname{tr} \boldsymbol{\Sigma}$ nor $\|\nabla f\|^2$ can be
read off a single minibatch, but both follow from a quantity that can: the
squared norm of a minibatch gradient. We expand
$\|\hat{\mathbf{g}}_b\|^2 = \|\nabla f + (\hat{\mathbf{g}}_b - \nabla f)\|^2$
and take expectations. The cross term vanishes because the deviation has
mean zero, which leaves

$$\mathbb{E}\big[\|\hat{\mathbf{g}}_b\|^2\big] = \|\nabla f\|^2 + \frac{\operatorname{tr} \boldsymbol{\Sigma}}{b}.$$
:eqlabel:`eq_sqnorm-bias`

On average a minibatch gradient is longer than the true gradient, and the
excess squared length is its noise power. Let $N_b$ denote the mean squared
norm of minibatch gradients at batch size $b$. Measuring it at a small
batch size $b_{\textrm{s}}$ and a large one $b_{\textrm{l}}$ gives two
linear equations in the two unknowns, with the solution

$$
\operatorname{tr} \boldsymbol{\Sigma} \approx
\frac{N_{b_{\textrm{s}}} - N_{b_{\textrm{l}}}}{1/b_{\textrm{s}} - 1/b_{\textrm{l}}},
\qquad
\|\nabla f\|^2 \approx
\frac{b_{\textrm{l}} N_{b_{\textrm{l}}} - b_{\textrm{s}} N_{b_{\textrm{s}}}}{b_{\textrm{l}} - b_{\textrm{s}}}.
$$

The difference of the two mean norms isolates the noise, and subtracting
that noise from the large-batch norm leaves the signal. No pass over the
dataset is needed, which makes the estimator cheap enough to log throughout
a large training run. There, each of the devices that share a batch
computes a small-batch gradient anyway, and their average is the
large-batch gradient.

We measure the two test models of :numref:`sec_test_models`: `TinyLM`, the
byte-level transformer of :numref:`subsec_tinylm`, and the image classifier
`FashionCNN`. For each, a sampling function draws $b$ examples
independently, as the estimator assumes. For the language model one
example is a window of 64 bytes at a random offset in the training text, so
its noise scale and its batch sizes count sequences.

```{.python .input #batch-size-a-two-batch-estimator-1}
%%tab pytorch
device = d2l.try_gpu()
data = d2l.WikiText103(batch_size=64, num_steps=64)
fashion = d2l.FashionMNIST(batch_size=256)
X_f = (fashion.train.data.float() / 255).unsqueeze(1).to(device)
y_f = fashion.train.targets.to(device)

def sample_fashion(b):  # b random training images
    i = torch.randint(len(y_f), (b,), device=device)
    return X_f[i], y_f[i]

def sample_lm(b):  # b windows at random offsets in the training text
    starts = torch.randint(len(data.train) - data.num_steps, (b,))
    X, Y = data.windows(data.train, starts.numpy())
    return X.to(device), Y.to(device)
```

```{.python .input #batch-size-a-two-batch-estimator-1}
%%tab jax
data = d2l.WikiText103(batch_size=64, num_steps=64)
fashion = d2l.FashionMNIST(batch_size=256)
X_f = jnp.asarray(fashion.train[0], jnp.float32)[..., None] / 255
y_f = jnp.asarray(fashion.train[1], jnp.int32)
rng = np.random.default_rng(d2l.get_seed())

def sample_fashion(b):  # b random training images
    i = rng.integers(len(y_f), size=b)
    return X_f[i], y_f[i]

def sample_lm(b):  # b windows at random offsets in the training text
    return data.windows(data.train,
                        rng.integers(len(data.train) - data.num_steps, size=b))
```

`noise_scale` follows the two formulas line by line. It averages the
squared gradient norm over 400 small minibatches and 50 large ones, solves
for the noise power and the signal power, and returns their ratio.

```{.python .input #batch-size-a-two-batch-estimator-2}
%%tab pytorch
def ce_loss(model, X, Y):  #@save
    """Mean cross-entropy of the logits model(X) for the labels Y."""
    logits = model(X)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                           Y.reshape(-1))

def grad_sq_norm(model, X, Y):  #@save
    """Squared norm of the gradient of ce_loss over all parameters."""
    grads = torch.autograd.grad(ce_loss(model, X, Y),
                                list(model.parameters()))
    return sum((g ** 2).sum() for g in grads)

def noise_scale(model, sample, b_small=16, b_big=2048, m=400):  #@save
    """Gradient-noise scale from the mean squared gradient norms of m
    minibatches of size b_small and m // 8 of size b_big from sample(b)."""
    def mean_sq_norm(b, m):  # N_b, averaged over m random minibatches
        return torch.stack([grad_sq_norm(model, *sample(b))
                            for _ in range(m)]).mean()
    n_small, n_big = mean_sq_norm(b_small, m), mean_sq_norm(b_big, m // 8)
    tr_sigma = (n_small - n_big) / (1 / b_small - 1 / b_big)
    sq_norm = (b_big * n_big - b_small * n_small) / (b_big - b_small)
    return (tr_sigma / sq_norm).item()
```

```{.python .input #batch-size-a-two-batch-estimator-2}
%%tab jax
def ce_loss(model, X, Y):  #@save
    """Mean cross-entropy of the logits model(X) for the labels Y."""
    logits = model(X)
    return optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(-1, logits.shape[-1]), Y.reshape(-1)).mean()

@nnx.jit  #@save
def grad_sq_norm(model, X, Y):
    """Squared norm of the gradient of ce_loss over all parameters."""
    grads = nnx.grad(ce_loss)(model, X, Y)
    return sum((g ** 2).sum() for g in jax.tree.leaves(grads))

def noise_scale(model, sample, b_small=16, b_big=2048, m=400):  #@save
    """Gradient-noise scale from the mean squared gradient norms of m
    minibatches of size b_small and m // 8 of size b_big from sample(b)."""
    def mean_sq_norm(b, m):  # N_b, averaged over m random minibatches
        return jnp.stack([grad_sq_norm(model, *sample(b))
                          for _ in range(m)]).mean()
    n_small, n_big = mean_sq_norm(b_small, m), mean_sq_norm(b_big, m // 8)
    tr_sigma = (n_small - n_big) / (1 / b_small - 1 / b_big)
    sq_norm = (b_big * n_big - b_small * n_small) / (b_big - b_small)
    return float(tr_sigma / sq_norm)
```

The noise scale depends on the parameters, so we measure it twice: at
initialization, and after 500 steps of Adam with `d2l.train_steps`
(:numref:`sec_test_models`). First the CNN, where the two batch sizes of
the estimator are 16 and 2,048 images. After training we call the
estimator twice on the same network, to see how far two measurements of
one model differ.

```{.python .input #batch-size-a-two-batch-estimator-3}
%%tab pytorch
cnn = d2l.FashionCNN().to(device)
print(f'noise scale at initialization: '
      f'{noise_scale(cnn, sample_fashion):.0f}')
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.004)
d2l.train_steps(cnn, fashion, optimizer, num_steps=500)
print(f'after 500 steps of Adam: {noise_scale(cnn, sample_fashion):.0f}, '
      f'measured again: {noise_scale(cnn, sample_fashion):.0f}')
```

```{.python .input #batch-size-a-two-batch-estimator-3}
%%tab jax
cnn = d2l.FashionCNN(rngs=nnx.Rngs(d2l.get_key()))
print(f'noise scale at initialization: '
      f'{noise_scale(cnn, sample_fashion):.0f}')
optimizer = nnx.Optimizer(cnn, optax.adam(0.004), wrt=nnx.Param)
d2l.train_steps(cnn, fashion, optimizer, num_steps=500)
print(f'after 500 steps of Adam: {noise_scale(cnn, sample_fashion):.0f}, '
      f'measured again: {noise_scale(cnn, sample_fashion):.0f}')
```

Then the language model, with batches of 8 and 256 sequences. Smaller
ones suffice here, because one example is a sequence of 64 bytes and its
gradient is already an average over 64 predictions:

```{.python .input #batch-size-a-two-batch-estimator-4}
%%tab pytorch
lm = d2l.TinyLM(len(data.vocab)).to(device)
print(f'noise scale at initialization: '
      f'{noise_scale(lm, sample_lm, b_small=8, b_big=256):.1f}')
optimizer = torch.optim.Adam(lm.parameters(), lr=0.003)
d2l.train_steps(lm, data, optimizer, num_steps=500)
print(f'after 500 steps of Adam: '
      f'{noise_scale(lm, sample_lm, b_small=8, b_big=256):.0f}')
```

```{.python .input #batch-size-a-two-batch-estimator-4}
%%tab jax
lm = d2l.TinyLM(len(data.vocab), rngs=nnx.Rngs(d2l.get_key()))
print(f'noise scale at initialization: '
      f'{noise_scale(lm, sample_lm, b_small=8, b_big=256):.1f}')
optimizer = nnx.Optimizer(lm, optax.adam(0.003), wrt=nnx.Param)
d2l.train_steps(lm, data, optimizer, num_steps=500)
print(f'after 500 steps of Adam: '
      f'{noise_scale(lm, sample_lm, b_small=8, b_big=256):.0f}')
```

At initialization the noise scale is of the order of a hundred images for
the CNN and about two sequences for the language model. These are small
numbers: the batches of 256 images and 64 sequences on which
:numref:`sec_test_models` trained the two models lie above them.

They do not stay small. After 500 steps the noise scale of the language
model has grown more than tenfold, to the order of a hundred sequences.
Asked for its two parts separately, the estimator shows why (Exercise
:exref:`ex_batch-size-noise-scale-over-training`): the disagreement
between the sequences has changed little, and the gradient that they share
has collapsed. At initialization every sequence asks for the same first
correction, more probability for a space and less for a `z`. Once that is
learned, what remains to be learned differs from one sequence to the next.

The CNN is less orderly. Where 500 steps of Adam take it depends on the
seed, and a second execution of the cell can print a value that differs
from this one by an order of magnitude. Is the estimator to blame? No, its
two measurements of the trained network agree to within a quarter, so the
estimator is repeatable. The network it measures is not. This is less of a
nuisance than it sounds. Whoever chooses a batch size measures the network
that they are training, and for that network the estimator is precise
enough. The spread between seeds troubles only those who copy our number.

They should not copy it in any case, because the magnitudes do not carry
over. For the largest language models that :citet:`kaplan2020scaling`
trained, the batch size that plays this role reached one to two million
tokens, against about a hundred sequences of 64 bytes here. The concept
carries over, and so does the growth during training.

## Batch Size and Learning Rate

A less noisy gradient permits a larger step. How much larger, and how many
steps does it save? No formula answers this for a network, so we start
from the simplest statement about a step that we have, the descent
guarantee of :numref:`sec_gd`, and ask what noise does to it. The result
is a crude model, and the experiments that follow have to say how far it
can be trusted.

### The Best Step under Noise

The guarantee rested on one fact. A function whose curvature is at most
$L$ in every direction lies below a parabola of curvature $L$, so a step
$\boldsymbol{\epsilon}$ changes it by at most
$\boldsymbol{\epsilon}^\top \nabla f + \frac{L}{2} \|\boldsymbol{\epsilon}\|^2$.
For the step $-\eta \nabla f$ of gradient descent this bound is lowest at
$\eta = 1/L$, where it guarantees a decrease of $\|\nabla f\|^2 / (2L)$.
Now take the step with a minibatch gradient,
$\boldsymbol{\epsilon} = -\eta\, \hat{\mathbf{g}}_b$, and average over the
draw of the minibatch. The first term averages to $-\eta \|\nabla f\|^2$,
because $\hat{\mathbf{g}}_b$ is unbiased. The second contains the mean
squared norm of a minibatch gradient, which :eqref:`eq_sqnorm-bias` gives:

$$
\mathbb{E}\big[f(\mathbf{x} - \eta\, \hat{\mathbf{g}}_b)\big] - f(\mathbf{x})
\leq -\eta\, \|\nabla f\|^2
+ \frac{L}{2}\, \eta^2 \Big(\|\nabla f\|^2 + \frac{\operatorname{tr} \boldsymbol{\Sigma}}{b}\Big).
$$
:eqlabel:`eq_noisy-step`

Noise leaves the first term alone: on average a noisy step makes the same
first-order progress as an exact one. It adds to the second term, the
penalty for stepping too far, because a noisy gradient is a longer vector.
The right-hand side is a parabola in $\eta$. Setting its derivative to
zero, and dividing numerator and denominator by $\|\nabla f\|^2$, gives the
best learning rate and the decrease that it guarantees:

$$
\eta_{\textrm{opt}}(b) = \frac{1}{L} \cdot \frac{1}{1 + b_{\textrm{noise}} / b},
\qquad
f(\mathbf{x}) - \mathbb{E}\big[f(\mathbf{x} - \eta_{\textrm{opt}}\, \hat{\mathbf{g}}_b)\big]
\geq \frac{\|\nabla f\|^2}{2L} \cdot \frac{1}{1 + b_{\textrm{noise}} / b}.
$$
:eqlabel:`eq_best-step`

Noise divides the best learning rate and the progress of a step by the same
factor, $1 + b_{\textrm{noise}} / b$. If you keep one formula from this
section, keep this one. At $b = b_{\textrm{noise}}$ a step makes half the
progress of a step with the exact gradient, and at a tenth of
$b_{\textrm{noise}}$ an eleventh. The learning rate matters as much as the
batch. Suppose that we keep the rate that is best for a small batch. Then
no larger batch more than doubles the progress of a step, not even the
whole dataset (Exercise :exref:`ex_batch-size-fixed-rate`).

Now comes the bold step of the argument: we take the guaranteed progress
for the actual one. If every step makes $1 / (1 + b_{\textrm{noise}} / b)$
of the progress of a noise-free step, a run needs
$1 + b_{\textrm{noise}} / b$ times as many steps. Let $S(b)$ be the number
of *steps* that a run at batch size $b$ takes to reach a given loss,
$S_{\min}$ the number it would take with exact gradients, and
$E(b) = b\, S(b)$ the number of *examples* it processes on the way. Then

$$
\frac{S(b)}{S_{\min}} = 1 + \frac{b_{\textrm{noise}}}{b},
\qquad
\frac{E(b)}{E_{\min}} = 1 + \frac{b}{b_{\textrm{noise}}}.
$$
:eqlabel:`eq_steps-examples`

The second equation is the first multiplied by $b$, with
$E_{\min} = b_{\textrm{noise}} S_{\min}$, the number of examples that a run
with very small batches consumes. The step count falls like $1/b$ while $b$
is small compared with $b_{\textrm{noise}}$ and approaches $S_{\min}$
beyond it. The example count does the reverse. With
$b_{\textrm{noise}} = 100$, a batch of 10 needs $11\, S_{\min}$ steps but
only $1.1\, E_{\min}$ examples, and a batch of 1,000 needs
$1.1\, S_{\min}$ steps but $11\, E_{\min}$ examples.

Two names go with these formulas. Far below the noise scale, doubling $b$
halves $S$ and leaves the example count unchanged, which is called
*perfect scaling*. At $b = b_{\textrm{noise}}$ both counts stand at twice
their minimum, and the batch size at which they do so is called the
*critical batch size*. Below it, most of the extra work per step turns
into fewer steps. Above it, most of the extra work saves nothing.

This is the model of :citet:`McCandlish.Kaplan.Amodei.ea.2018` in its
simplest form, and a good deal went into it. A bound was read as a
prediction. The run was assumed to use the best learning rate at every
batch size. The noise scale was treated as one number, although we saw it
grow within 500 steps, so $b_{\textrm{noise}}$ in
:eqref:`eq_steps-examples` stands for an average over the run. And the
curvature was bounded by $L$ in every direction. With the Hessian
$\mathbf{H}$ in place of the bound, the same calculation gives the same
formulas with the noise scale
$\operatorname{tr}(\mathbf{H} \boldsymbol{\Sigma}) / \nabla f^\top \mathbf{H} \nabla f$,
which weights each direction by its curvature. The ratio
:eqref:`eq_noise-scale` that we measure equals it when the curvature is the
same in every direction, and is a cheap substitute otherwise.

### Linear and Square-Root Scaling

The first equation of :eqref:`eq_best-step` says how the learning rate
should move with the batch. Far below the noise scale,
$1 + b_{\textrm{noise}} / b \approx b_{\textrm{noise}} / b$, and the best
learning rate is proportional to the batch size:
$\eta_{\textrm{opt}}(b) \approx b / (L\, b_{\textrm{noise}})$. This is the
*linear scaling rule* for SGD. :citet:`Goyal.Dollar.Girshick.ea.2017`
established it for large-batch training on ImageNet, with a simpler
argument. If the gradient changes little from one step to the next, $k$
steps at rate $\eta$ on batches of $b$ examples move the weights as far as
one step at rate $k\eta$ on all $kb$ examples. The rule keeps $\eta / b$
fixed, and with it the noise ball in which SGD comes to rest
(:numref:`subsec_sgd-variance`), which is a third route to the same rule
:cite:`Smith.Kindermans.Ying.ea.2018`. Momentum changes nothing as long as
$\beta$ stays the same, since it multiplies the effective step by
$1 / (1 - \beta)$ at every batch size (:numref:`sec_momentum`).

The rule has its end built in. As $b$ passes the noise scale,
$\eta_{\textrm{opt}}(b)$ levels off at $1/L$, the best rate for the exact
gradient, which the curvature sets and no batch size can raise. With
$b_{\textrm{noise}} = 100$, going from a batch of 10 to one of 100 raises
the best rate 5.5-fold, from $1/(11L)$ to $1/(2L)$, and going on to 1,000
raises it by a further factor of 1.8 only.

Adam needs a rule of its own, because it rescales the gradient before the
learning rate is applied. This is the step of the section that is easiest
to get wrong, so we write it out for a single coordinate. Adam divides the
average gradient of each coordinate by its scale, the root of the average
squared gradient (:numref:`sec_adam`). Over the examples, let the gradient
of our coordinate have the mean $\bar{g}$ and the standard deviation
$\sigma$, so that the noise of a minibatch gradient has the standard
deviation $\sigma / \sqrt{b}$, and let the noise dominate. The scale is
then about $\sigma / \sqrt{b}$, and leaving momentum aside, one step
changes the coordinate by about

$$-\eta\, \frac{\bar{g} + z\, \sigma / \sqrt{b}}{\sigma / \sqrt{b}} = -\eta\, \frac{\sqrt{b}}{\sigma}\, \bar{g} - \eta\, z,$$

where $z$ is a random number of mean zero and variance one. The first term
is the progress of the step and the second its noise. A batch $k$ times
larger makes the progress of each step $\sqrt{k}$ times larger at the same
$\eta$, because the scale has shrunk. To cover in one step the distance of
$k$ small ones, $\eta$ must supply the other factor of $\sqrt{k}$. The
noise then agrees as well: $k$ small steps accumulate a random
displacement of variance $k \eta^2$, and so does one step at rate
$\sqrt{k}\, \eta$. This is the *square-root scaling rule*.

The analysis of :citet:`Malladi.Lyu.Panigrahi.ea.2022`, which treats Adam
as a stochastic differential equation, arrives at the same rule and moves
$\beta_1$, $\beta_2$ and $\epsilon$ with the batch as well. Common
practice, and this section, move $\eta$ alone. In summary, a learning rate
$\eta_0$ tuned at batch size $b_0$ moves to batch size $b$ by

$$
\eta(b) = \frac{b}{b_0}\, \eta_0 \;\; \textrm{(SGD)},
\qquad
\eta(b) = \sqrt{\frac{b}{b_0}}\, \eta_0 \;\; \textrm{(Adam)},
$$
:eqlabel:`eq_lr-rules`

as long as both batch sizes lie below the noise scale.

One gap remains. The runs below use Adam, and the step counts of
:eqref:`eq_steps-examples` were derived for a gradient step. Do they
apply? At the two ends they do. Far below the noise scale, Adam's division
contributes a factor of $\sqrt{k}$ and the rule another, so a batch $k$
times larger makes $k$ times the progress per step, which is perfect
scaling again. Far above it neither factor is available, and the step
count stops falling. That the same factor $1 + b_{\textrm{noise}} / b$
joins the two ends is an assumption, and the experiments have to bear it
out.

## Steps to a Target
:label:`subsec_batch-size-steps`

The model makes a prediction about training speed that could fail, so we
test it. The test, standard in the study of large-batch training
:cite:`Shallue.Lee.Antognini.ea.2019`, fixes a target loss, trains at a
range of batch sizes, and counts the steps $S(b)$ that each run needs to
reach the target. On logarithmic axes perfect scaling is a straight line of
slope $-1$, and the model predicts that the measured counts follow it at
small batches and bend away near the noise scale.

An experiment of this kind is easily spoiled by its details, so we state
ours. The target is a fixed loss on a fixed set of examples: the first
8,192 training images for the CNN, and for the language model 2,048
windows spread over the validation split, text that no run trains on. The
loss of the current minibatch would not do, because the small-batch runs,
whose loss estimates fluctuate most, would cross the target early by
chance. We evaluate every ten steps, which is also the resolution of every
step count below. Adam's learning rate is set at one batch size, to 0.003
at 64 sequences (the best rate of :numref:`sec_adam`) and to 0.004 at 256
images, and moved to the other batch sizes by the square-root rule. All
batch sizes of a sweep start from the same initialization, so that the
differences along a curve come from the batch size and not from the draw.

```{.python .input #batch-size-steps-to-a-target}
%%tab pytorch
def eval_loss(model, X, Y, chunk=4096):
    with torch.inference_mode():
        return torch.stack([ce_loss(model, X[i:i + chunk], Y[i:i + chunk])
                            for i in range(0, len(Y), chunk)]).mean().item()
```

```{.python .input #batch-size-steps-to-a-target}
%%tab jax
eval_chunk = nnx.jit(ce_loss)

def eval_loss(model, X, Y, chunk=4096):
    return float(jnp.stack([eval_chunk(model, X[i:i + chunk], Y[i:i + chunk])
                            for i in range(0, len(Y), chunk)]).mean())
```

The training loop is `d2l.train_steps`, which the function `reached` stops
once the target is met:

```{.python .input #batch-size-steps-to-a-target-1}
%%tab pytorch, jax
def train_to_target(model, data, optimizer, X_eval, Y_eval, target,
                    max_steps, eval_every=10):
    """Return the number of steps until the eval loss reaches target."""
    def reached(step):
        return (step % eval_every == 0
                and eval_loss(model, X_eval, Y_eval) <= target)
    steps = len(d2l.train_steps(model, data, optimizer, max_steps,
                                stop=reached))
    return steps if reached(steps) else float('inf')
```

### The Language Model

For `TinyLM` the target is a held-out loss of 1.9 nats per byte, a
perplexity of $e^{1.9} \approx 6.7$ against 256 for uniform guessing: well
into learning, and far from converged. The batch sizes run from 4 to 256
sequences. The noise scale was about 2 sequences at initialization and of
the order of a hundred after 500 steps, so the bend should fall inside
this range. We can say more. Take $b_{\textrm{noise}} = 100$ as a round
figure for the whole run. Then :eqref:`eq_steps-examples` predicts that the
first quadrupling of the batch, from 4 to 16 sequences, divides the step
count by $26 / 7.25 \approx 3.6$, and the last, from 64 to 256, by only
$2.56 / 1.39 \approx 1.8$. The plot shows the measured step counts and the
line of perfect scaling through the smallest batch.

```{.python .input #batch-size-the-language-model}
%%tab pytorch
starts = np.arange(0, len(data.val) - data.num_steps, data.num_steps)
X_val, Y_val = (t.to(device) for t in data.windows(data.val,
                                                   starts[::8][:2048]))
lm_bs, lm_steps = [4, 16, 64, 256], []
seed = random.getrandbits(32)  # One initialization for all batch sizes
for b in lm_bs:
    torch.manual_seed(seed)
    data.batch_size = b
    model = d2l.TinyLM(len(data.vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.003 * math.sqrt(b / 64))
    lm_steps.append(train_to_target(model, data, optimizer, X_val, Y_val,
                                    target=1.9, max_steps=8000))
    print(f'b={b:4d}  steps={lm_steps[-1]:6.0f}  '
          f'examples={b * lm_steps[-1]:8.0f}')
d2l.plot(lm_bs, [lm_steps, [lm_steps[0] * lm_bs[0] / b for b in lm_bs]],
         'batch size (sequences)', 'steps to loss 1.9', xscale='log',
         yscale='log', legend=['measured', 'perfect scaling'])
```

```{.python .input #batch-size-the-language-model}
%%tab jax
starts = np.arange(0, len(data.val) - data.num_steps, data.num_steps)
X_val, Y_val = data.windows(data.val, starts[::8][:2048])
lm_bs, lm_steps = [4, 16, 64, 256], []
seed = d2l.get_seed()  # One initialization for all batch sizes
for b in lm_bs:
    data.batch_size = b
    model = d2l.TinyLM(len(data.vocab), rngs=nnx.Rngs(seed))
    optimizer = nnx.Optimizer(model,
                              optax.adam(0.003 * math.sqrt(b / 64)),
                              wrt=nnx.Param)
    lm_steps.append(train_to_target(model, data, optimizer, X_val, Y_val,
                                    target=1.9, max_steps=8000))
    print(f'b={b:4d}  steps={lm_steps[-1]:6.0f}  '
          f'examples={b * lm_steps[-1]:8.0f}')
d2l.plot(lm_bs, [lm_steps, [lm_steps[0] * lm_bs[0] / b for b in lm_bs]],
         'batch size (sequences)', 'steps to loss 1.9', xscale='log',
         yscale='log', legend=['measured', 'perfect scaling'])
```

The first quadrupling runs nearly parallel to the perfect-scaling line. It
cuts the step count a little more than threefold, where the round figure
said 3.6, and the example count grows by only about a quarter. Then the
curve bends away. The last quadrupling roughly halves the step count, as
the predicted 1.8 would have it, and the run at 256 sequences consumes
three to five times as many examples as the run at 4. For a model with a
single round number in it, that is a fair forecast.

### The CNN

On the CNN the target is a loss of 0.35 and the batch sizes run from 8 to
2,048 images. This range extends far beyond the noise scale, which was of
the order of a hundred images at initialization, and that calls for one
change. Beyond the noise scale the best rate no longer grows with the
batch, while the square-root rule keeps raising it. We therefore apply the
rule up to 512 images, a few times that noise scale, and hold the learning
rate at its value there for the batch of 2,048. The language-model sweep
ended at 256 sequences, where the rule had merely doubled the rate, and we
let it run.

```{.python .input #batch-size-the-cnn}
%%tab pytorch
cnn_bs, cnn_steps = [8, 32, 128, 512, 2048], []
seed = random.getrandbits(32)  # One initialization for all batch sizes
for b in cnn_bs:
    torch.manual_seed(seed)
    fashion_b = d2l.FashionMNIST(batch_size=b)
    model = d2l.FashionCNN().to(device)
    lr = 0.004 * math.sqrt(min(b, 512) / 256)  # Held fixed above 512
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    cnn_steps.append(train_to_target(model, fashion_b, optimizer,
                                     X_f[:8192], y_f[:8192], target=0.35,
                                     max_steps=8000))
    print(f'b={b:4d}  steps={cnn_steps[-1]:6.0f}  '
          f'examples={b * cnn_steps[-1]:8.0f}')
d2l.plot(cnn_bs, [cnn_steps,
                  [cnn_steps[0] * cnn_bs[0] / b for b in cnn_bs]],
         'batch size', 'steps to loss 0.35', xscale='log', yscale='log',
         legend=['measured', 'perfect scaling'])
```

```{.python .input #batch-size-the-cnn}
%%tab jax
cnn_bs, cnn_steps = [8, 32, 128, 512, 2048], []
seed = d2l.get_seed()  # One initialization for all batch sizes
for b in cnn_bs:
    fashion_b = d2l.FashionMNIST(batch_size=b)
    model = d2l.FashionCNN(rngs=nnx.Rngs(seed))
    lr = 0.004 * math.sqrt(min(b, 512) / 256)  # Held fixed above 512
    optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)
    cnn_steps.append(train_to_target(model, fashion_b, optimizer,
                                     X_f[:8192], y_f[:8192], target=0.35,
                                     max_steps=8000))
    print(f'b={b:4d}  steps={cnn_steps[-1]:6.0f}  '
          f'examples={b * cnn_steps[-1]:8.0f}')
d2l.plot(cnn_bs, [cnn_steps,
                  [cnn_steps[0] * cnn_bs[0] / b for b in cnn_bs]],
         'batch size', 'steps to loss 0.35', xscale='log', yscale='log',
         legend=['measured', 'perfect scaling'])
```

The CNN shows the same pattern. The first quadrupling, from 8 to 32 images,
stays close to the perfect-scaling line and cuts the step count about
threefold. Then the gap widens. At the top of the range, quadrupling the
batch from 512 to 2,048 images saves less than half of the steps. The run
at 2,048 images thus consumes two to four times the examples of the run
at 512, and seven to ten times those of the run at 8. Past the bend, extra
work per step saves few steps and multiplies the data consumed.

### The Critical Batch Size

Both curves have the shape that :eqref:`eq_steps-examples` predicts, so
let us fit the model to them. Its step count is linear in $1/b$. A
straight-line fit of the measured counts against $1/b$ thus returns
$S_{\min}$ as the intercept and the critical batch size as the ratio of
slope to intercept. We write $b_{\textrm{crit}}$ for this fitted value and
keep it apart from the noise scale on purpose. In the model the two are
equal. In measurement they are different estimates of one quantity,
$b_{\textrm{noise}}$ from the statistics of gradients at one moment and
$b_{\textrm{crit}}$ from the step counts of whole runs. The fit weights
each run by the inverse of its step count, so that the small-batch runs
with their large counts do not dominate it.

```{.python .input #batch-size-the-critical-batch-size-1}
%%tab pytorch, jax
def fit_steps(bs, steps):
    """Fit S(b) = S_min (1 + b_crit / b) by least squares in 1/b."""
    b, S = np.array(bs, dtype=float), np.array(steps, dtype=float)
    ok = np.isfinite(S)  # Leave out runs that never reached the target
    slope, S_min = np.polyfit(1 / b[ok], S[ok], 1, w=1 / S[ok])
    assert S_min > 0, 'step counts never level off: no floor to fit'
    return S_min, slope / S_min

fits = {'TinyLM': fit_steps(lm_bs, lm_steps),
        'FashionCNN': fit_steps(cnn_bs, cnn_steps)}
for name, (S_min, b_crit) in fits.items():
    print(f'{name:10s}  S_min = {S_min:4.0f} steps  b_crit = {b_crit:4.0f}')
```

The fitted critical batch sizes are of the order of a hundred sequences
for `TinyLM` and two to four hundred images for the CNN. That is the same
order of magnitude as the noise scales that each model reached during its
first 500 steps, and closer agreement is more than the model promises. It
speaks of a curvature-weighted noise scale averaged over the whole run,
and of the best learning rate at every batch size. We measured the
unweighted ratio at two moments and moved the learning rate by a rule. For
the CNN the rule was moreover switched off above 512 images, and its
$b_{\textrm{crit}}$ reflects that choice in part.

Dividing each run's steps by $S_{\min}$ and its examples by
$E_{\min} = b_{\textrm{crit}} S_{\min}$ removes the two fitted constants.
Multiplying the two equations of :eqref:`eq_steps-examples` after
subtracting 1 from each shows that, if the model holds, every run of both
models lies on the single curve

$$\Big(\frac{E}{E_{\min}} - 1\Big)\Big(\frac{S}{S_{\min}} - 1\Big) = 1.$$

```{.python .input #batch-size-the-bill-in-examples}
%%tab pytorch, jax
x = np.logspace(0.005, 1.3, 100)  # E / E_min along the model curve
xs, ys = [x], [1 + 1 / (x - 1)]
for (S_min, b_crit), bs, steps in zip(fits.values(), (lm_bs, cnn_bs),
                                      (lm_steps, cnn_steps)):
    xs.append([b * S / (b_crit * S_min) for b, S in zip(bs, steps)])
    ys.append([S / S_min for S in steps])
d2l.plot(xs, ys, 'examples / minimum', 'steps / minimum', xscale='log',
         yscale='log', fmts=('-', 'o', 's'),
         legend=['model', 'TinyLM', 'FashionCNN'])
```

Part of this agreement is built in, because each model's two constants were
fitted to its own four or five runs, and a two-parameter curve can pass
through any two of them. The plot tests the shape: whether a step count
linear in $1/b$ fits all the runs, with two or three runs per model to
spare. On that count the model passes, for the runs of both models stay
close to the one curve.

Moving along the curve trades one resource for the other. Toward the upper
left, below the critical batch size, a run needs many steps and few
examples, and toward the lower right few steps and many examples. More
total work can thus reduce the number of sequential steps and, with enough
hardware, the time. Which point is best depends on what is scarce,
accelerators or days. Large runs choose batch sizes near the bend, where
neither count is more than about twice its minimum.

## Tests of the Learning-Rate Rules

The sweeps above relied on the square-root rule without testing it. A
direct test takes learning-rate grids at two batch sizes. If a rule holds,
the best learning rate at the larger batch is the best rate at the smaller
one, multiplied by the factor that the rule predicts.

### Sweeps at Two Batch Sizes

We run the test on the CNN and let every run process the same 12,800
examples, a fifth of the training set, so that a batch $k$ times larger
takes $k$ times fewer steps. The runs at one batch size share their
initialization and the order of their images, and differ in the learning
rate alone. `final_loss_at` trains once per learning rate with
`d2l.lr_sweep` (:numref:`sec_test_models`) and returns the loss of each
run, averaged over its last quarter. Since no image is seen twice, this is
a loss on images that are new to the model.

```{.python .input #batch-size-learning-rate-rules-1}
%%tab pytorch
def final_loss_at(make_optimizer, lrs, b, seed, num_examples=12800):
    """Train the CNN once per learning rate; return the final losses."""
    curves, _ = d2l.lr_sweep(d2l.FashionCNN, make_optimizer,
                             d2l.FashionMNIST(batch_size=b), lrs, seed,
                             num_examples // b, verbose=False)
    return [d2l.final_loss(c, k=len(c) // 4) for c in curves.values()]
```

```{.python .input #batch-size-learning-rate-rules-1}
%%tab jax
def final_loss_at(make_tx, lrs, b, seed, num_examples=12800):
    """Train the CNN once per learning rate; return the final losses."""
    init_key, order_key = jax.random.split(jax.random.key(seed))
    data = d2l.DataModule()  # Every run sees the images in one order
    data.train_dataloader = lambda: d2l.ArrayLoader(
        (X_f, y_f), b, shuffle=True, key=order_key)
    curves, _ = d2l.lr_sweep(d2l.FashionCNN, make_tx, data, lrs, init_key,
                             num_examples // b, verbose=False)
    return [d2l.final_loss(c, k=len(c) // 4) for c in curves.values()]
```

First SGD with momentum 0.9, at batches of 8 and 64 images, on learning
rates a factor of two apart. Both batches lie below the noise scale of the
untrained CNN, where the linear rule should hold. It predicts that the
best rate moves by the batch ratio of 8, three steps of the grid. Without
any scaling it would stay where it is.

```{.python .input #batch-size-learning-rate-rules-2}
%%tab pytorch
sgd_lrs = [0.003125, 0.00625, 0.0125, 0.025, 0.05, 0.1, 0.2]
make_sgd = lambda model, lr: torch.optim.SGD(model.parameters(), lr,
                                             momentum=0.9)
seed = random.getrandbits(32)
sgd_small = final_loss_at(make_sgd, sgd_lrs, b=8, seed=seed)
sgd_big = final_loss_at(make_sgd, sgd_lrs, b=64, seed=seed)
for b, loss in ((8, sgd_small), (64, sgd_big)):
    print(f'b={b:2d}: best lr {sgd_lrs[np.nanargmin(loss)]:g}')
d2l.plot(sgd_lrs, [sgd_small, sgd_big], 'learning rate', 'final loss',
         xscale='log', legend=['b=8', 'b=64'])
```

```{.python .input #batch-size-learning-rate-rules-2}
%%tab jax
sgd_lrs = [0.003125, 0.00625, 0.0125, 0.025, 0.05, 0.1, 0.2]
make_sgd = lambda lr: optax.sgd(lr, momentum=0.9)
seed = d2l.get_seed()
sgd_small = final_loss_at(make_sgd, sgd_lrs, b=8, seed=seed)
sgd_big = final_loss_at(make_sgd, sgd_lrs, b=64, seed=seed)
for b, loss in ((8, sgd_small), (64, sgd_big)):
    print(f'b={b:2d}: best lr {sgd_lrs[np.nanargmin(loss)]:g}')
d2l.plot(sgd_lrs, [sgd_small, sgd_big], 'learning rate', 'final loss',
         xscale='log', legend=['b=8', 'b=64'])
```

The best rate moves by the factor of 8 that the linear rule predicts, or
in some executions by 4, one step of the grid short of it. The agreement
is less sharp than it may look, because the minimum at the larger batch is
shallow and a grid of factors of two cannot place it more finely. Leaving
the rate alone, however, is ruled out. At the rate that was best for 8
images, the curve of the larger batch lies more than a tenth of a nat
above its own minimum.

For Adam we make the test harder to pass. At a batch ratio of 8 the
square-root rule predicts a shift by $\sqrt{8} \approx 2.8$ and the linear
rule one by 8, less than two grid steps apart, and a shallow minimum would
blur the two. We therefore compare batches of 2 and 128 images, a ratio of
64. Now the three candidates are far apart: no scaling leaves the best rate
in place, the square-root rule moves it by $\sqrt{64} = 8$, three grid
steps, and the linear rule by 64, six grid steps. Note that 128 images
are about the noise scale of the untrained CNN, which is the edge of the
range for which the rule was derived. The cell prints the best rate at
each batch size, and the loss that the rate of each candidate reaches at
the larger batch.

```{.python .input #batch-size-learning-rate-rules-3}
%%tab pytorch
adam_lrs = [0.0001 * 2 ** k for k in range(13)]  # 0.0001 to 0.4096
make_adam = lambda model, lr: torch.optim.Adam(model.parameters(), lr)
seed = random.getrandbits(32)
adam_small = final_loss_at(make_adam, adam_lrs, b=2, seed=seed)
adam_big = final_loss_at(make_adam, adam_lrs, b=128, seed=seed)
i, j = int(np.nanargmin(adam_small)), int(np.nanargmin(adam_big))
print(f'best lr: {adam_lrs[i]:g} at b=2; {adam_lrs[j]:g} at b=128, '
      f'final loss {adam_big[j]:.3f}')
for rule, k in (('no scaling', 0), ('square root', 3), ('linear', 6)):
    if i + k < len(adam_lrs):
        print(f'{rule:11s}: lr {adam_lrs[i + k]:g} at b=128, '
              f'final loss {adam_big[i + k]:.3f}')
    else:
        print(f'{rule:11s}: lr {adam_lrs[i] * 2 ** k:g}, beyond the grid')
d2l.plot(adam_lrs, [adam_small, adam_big], 'learning rate', 'final loss',
         xscale='log', legend=['b=2', 'b=128'])
```

```{.python .input #batch-size-learning-rate-rules-3}
%%tab jax
adam_lrs = [0.0001 * 2 ** k for k in range(13)]  # 0.0001 to 0.4096
make_adam = optax.adam
seed = d2l.get_seed()
adam_small = final_loss_at(make_adam, adam_lrs, b=2, seed=seed)
adam_big = final_loss_at(make_adam, adam_lrs, b=128, seed=seed)
i, j = int(np.nanargmin(adam_small)), int(np.nanargmin(adam_big))
print(f'best lr: {adam_lrs[i]:g} at b=2; {adam_lrs[j]:g} at b=128, '
      f'final loss {adam_big[j]:.3f}')
for rule, k in (('no scaling', 0), ('square root', 3), ('linear', 6)):
    if i + k < len(adam_lrs):
        print(f'{rule:11s}: lr {adam_lrs[i + k]:g} at b=128, '
              f'final loss {adam_big[i + k]:.3f}')
    else:
        print(f'{rule:11s}: lr {adam_lrs[i] * 2 ** k:g}, beyond the grid')
d2l.plot(adam_lrs, [adam_small, adam_big], 'learning rate', 'final loss',
         xscale='log', legend=['b=2', 'b=128'])
```

The best rate moves by about the factor of 8 that the square-root rule
predicts. In most executions it lands on 8 itself or one grid step beside
it, at 4 or 16, and once in a while two steps short, at 2. It never comes
near 64, and the edge of the rule's range has done no visible harm. The
printed losses say what the other two candidates cost. Keeping the rate of
the small batch ends about a tenth of a nat or more above the best run,
because at so small a rate the 100 steps of the large batch are too few.
The rate of the linear rule is never the best, and what it costs is a
matter of luck: a few hundredths of a nat in one execution, a quarter of a
nat in another, and in some the run does not train at all and ends at
$\ln 10 \approx 2.30$, the loss of a uniform guess over the ten classes.
Sixty-four times the batch called for about eight times the learning rate,
not sixty-four.

Neither minimum is sharp, and near it a factor of two in $\eta$ changes the
loss little. The rules bridge large ratios of batch sizes, and a short
sweep around their prediction does the fine tuning.

### Limits of the Rules
:label:`subsec_batch-size-limits`

Both rules were derived for batches below the noise scale, and both end
there. For SGD we saw the best rate level off at $1/L$. That of Adam stops
moving as well, because the scale of a coordinate is then the size of its
gradient itself, which the batch does not change. A rate scaled further
overshoots, and Exercise :exref:`ex_batch-size-past-elbow` looks for the
overshoot on `TinyLM`.

A scaled rate is most at risk at the start of training. The noise scale is
then at its smallest, about 2 sequences for `TinyLM`, so a batch chosen
for the later run starts far above it, where neither rule applies, and at
a rate scaled for a state that the model has yet to reach. Large batches
with scaled-up rates are where warmup entered practice
:cite:`Goyal.Dollar.Girshick.ea.2017`: a rate above what a cold start
survives is raised gradually over the first steps
(:numref:`subsec_scheduler-warmup`). Our own sweeps did without it, and
could afford to, since no rule raised a rate beyond twice its starting
value.

Will a large batch hurt the test accuracy? Not by itself. Early reports of
such a gap :cite:`Keskar.Mudigere.Nocedal.ea.2017` compared runs whose
learning rate and length of training had not been retuned for the larger
batch, and with retuning the gap largely closed
:cite:`Shallue.Lee.Antognini.ea.2019`. The lasting cost of a large batch is
data, which is what this section measured.

## Choosing a Batch Size

### The Noise Scale during Training

So far we measured the noise scale at two moments. A batch size has to
serve a whole run, so we follow the noise scale through one: a new
`TinyLM`, trained with Adam at batch size 64 for 3,000 steps, with its
held-out loss and its noise scale printed every 250 steps.

```{.python .input #batch-size-growing-the-batch}
%%tab pytorch
data.batch_size = 64
lm = d2l.TinyLM(len(data.vocab)).to(device)
optimizer = torch.optim.Adam(lm.parameters(), lr=0.003)
steps, noise = list(range(0, 3001, 250)), []
for step in steps:
    if step > 0:
        d2l.train_steps(lm, data, optimizer, num_steps=250)
    noise.append(noise_scale(lm, sample_lm, b_small=8, b_big=256))
    print(f'step {step:4d}  held-out loss {eval_loss(lm, X_val, Y_val):.3f}  '
          f'noise scale {noise[-1]:5.1f}')
d2l.plot(steps, noise, 'step', 'noise scale (sequences)', yscale='log')
```

```{.python .input #batch-size-growing-the-batch}
%%tab jax
data.batch_size = 64
lm = d2l.TinyLM(len(data.vocab), rngs=nnx.Rngs(d2l.get_key()))
optimizer = nnx.Optimizer(lm, optax.adam(0.003), wrt=nnx.Param)
steps, noise = list(range(0, 3001, 250)), []
for step in steps:
    if step > 0:
        d2l.train_steps(lm, data, optimizer, num_steps=250)
    noise.append(noise_scale(lm, sample_lm, b_small=8, b_big=256))
    print(f'step {step:4d}  held-out loss {eval_loss(lm, X_val, Y_val):.3f}  '
          f'noise scale {noise[-1]:5.1f}')
d2l.plot(steps, noise, 'step', 'noise scale (sequences)', yscale='log')
```

The noise scale grows by about two orders of magnitude. It starts at about
2 sequences, reaches several dozen within 250 steps, and stands at a few
hundred by step 3,000. It grows fastest where the loss falls fastest. The
growth is not smooth: from one checkpoint to the next the value can halve
or double. The estimator alone does not explain jumps of this size, since
its two readings of one CNN agreed to within a quarter. The weights move,
and the noise scale moves with them. Only the trend over hundreds of steps
can be trusted.

The 64 sequences per step that we took for granted are thus right only
for a while. They are some thirty times the noise scale at the first step,
comparable to it a few hundred steps later, and a few times smaller than
it at the end. A fixed batch size resembles a stopped clock, which is
right for a moment and wrong before and after. Chosen for the end of
training it wastes examples at the beginning, and chosen for the beginning
it leaves parallelism unused later.

Large language-model runs therefore raise the batch during training. GPT-3
grew its batch from 32 thousand to 3.2 million tokens over the first
billions of tokens :cite:`brown2020language`. Such a ramp is planned
together with the schedule of :numref:`sec_scheduler`, because it does
part of the work of a decay: for SGD, raising $b$ shrinks the noise ball
as lowering $\eta$ does :cite:`Smith.Kindermans.Ying.ea.2018`.

### A Procedure for a New Model

The measurements of this section combine into a procedure, and it is
short. Its second step is the one on which the others depend.

1. Start from the device. Up to the batch at which the time of a step
   begins to grow with the batch, a larger batch costs almost no time. Use
   at least that many examples, and no more than fit in memory.
1. Measure the noise scale with `noise_scale`, at initialization and
   again after a few hundred steps. The two values answer different
   questions. The later one governs the choice, because the run spends
   nearly all of its steps at that noise scale or above it. The first one
   says how many examples the opening steps waste, which is the argument
   for starting with a smaller batch and raising it.
1. Choose by what is scarce. If it is compute, stay below the later
   value: at a third of it a run consumes a third more examples than the
   least possible. If it is time, and devices are at hand to share the
   batch (:numref:`chap_performance`), go up to about the noise scale and
   not far beyond. There a run needs twice the minimal number of steps and
   twice the minimal number of examples. At four times that batch it saves
   a further three eighths of the steps for two and a half times the
   examples.
1. Move the learning rate by the rule of the optimizer whenever the batch
   changes, and sweep a factor of two around the result.
1. Measure again later in the run, because the answer grows.

And the second GPU of the opening? With the round figure of 100 sequences
for the noise scale, the model says that doubling the batch from 64 to 128
divides the step count by $2.56 / 1.78 \approx 1.4$, not by 2. The run
finishes in about 70% of the time and reads about 40% more data. At that
point the second GPU does the work of less than half a GPU. It does more
later, as the noise scale grows.

The two rules of this section carry a learning rate from one batch size to
another on the same model. They are silent on the transfer that large-scale
training needs most, from a small model on which we can afford to tune to
the large one on which we cannot.

## Summary

A minibatch gradient is the full gradient plus noise, and the two have
equal power at the gradient-noise scale
$b_{\textrm{noise}} = \operatorname{tr} \boldsymbol{\Sigma} / \|\nabla f\|^2$,
which squared gradient norms at two batch sizes suffice to estimate. Noise
divides the best learning rate and the progress of a step by
$1 + b_{\textrm{noise}} / b$. Below the noise scale a larger batch
therefore saves steps at nearly constant total work. Above it the step
count nears a floor and the examples consumed keep growing, so the choice
depends on whether compute or time is scarce. Both test models followed
this curve, with fitted critical batch sizes of the same order as their
measured noise scales. When the batch changes, the learning rate moves
with it, in proportion to the batch for SGD and to its square root for
Adam, up to the noise scale. On the CNN, 64 times the batch moved the best
rate of Adam about eightfold. The noise scale of `TinyLM` grew by about
two orders of magnitude in 3,000 steps, so the batch size that suits a run
grows while it trains.

## Exercises

1. **A larger batch at an unchanged learning rate.** Let
   $\eta_0 = \eta_{\textrm{opt}}(b_0)$ be the best learning rate of
   :eqref:`eq_best-step` at batch size $b_0$, and write
   $c(b) = 1 + b_{\textrm{noise}} / b$.
   :exlabel:`ex_batch-size-fixed-rate`
    1. Insert $\eta_0$ into the bound :eqref:`eq_noisy-step` and show that
       at batch size $b$ the guaranteed decrease of a step is
       $2 - c(b) / c(b_0)$ times its value at $b_0$.
    1. Take $b_{\textrm{noise}} = 100$ and $b_0 = 10$. By what factor does
       the guaranteed decrease grow at $b = 100$, and as $b \to \infty$, if
       the rate stays at $\eta_0$? By what factor if the rate is retuned to
       $\eta_{\textrm{opt}}(b)$?
    1. What would a sweep of steps to a target show if it held the learning
       rate fixed across batch sizes?

1. **The precision of the estimator.** Model a minibatch gradient as
   $\hat{\mathbf{g}}_b = \nabla f + \boldsymbol{\xi}$ with Gaussian noise
   $\boldsymbol{\xi} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma} / b)$.
    1. Show that the variance of $\|\hat{\mathbf{g}}_b\|^2$ is
       $4\, \nabla f^\top \boldsymbol{\Sigma} \nabla f / b + 2 \operatorname{tr}(\boldsymbol{\Sigma}^2) / b^2$.
       You may use that the third moments of $\boldsymbol{\xi}$ vanish and
       that $\|\boldsymbol{\xi}\|^2$ has variance
       $2 \operatorname{tr}(\boldsymbol{\Sigma}^2) / b^2$.
    1. Use
       $\nabla f^\top \boldsymbol{\Sigma} \nabla f \leq \|\nabla f\|^2 \operatorname{tr} \boldsymbol{\Sigma}$
       and
       $\operatorname{tr}(\boldsymbol{\Sigma}^2) \leq (\operatorname{tr} \boldsymbol{\Sigma})^2$
       to show that the standard deviation of one squared norm is at most
       $\sqrt{2}$ times its mean, at every batch size. How precisely does
       `noise_scale` then know $N_b$ from its 400 small and 50 large
       minibatches, in the worst case?
    1. The worst case has all noise in a single direction. Repeat the
       bound for noise that is spread evenly over $d$ directions,
       $\boldsymbol{\Sigma} = (\operatorname{tr} \boldsymbol{\Sigma} / d)\, \mathbf{I}$,
       and explain why two measurements of one network can agree better
       than the worst case suggests.
    1. The signal estimate is $N_{b_{\textrm{l}}}$ minus the noise
       $\operatorname{tr} \boldsymbol{\Sigma} / b_{\textrm{l}}$ that the
       large batch still carries. For a noise scale of 100 and
       $b_{\textrm{l}} = 256$, what fraction of $N_{b_{\textrm{l}}}$ is
       noise? Why does the estimate lose precision once the noise scale
       exceeds $b_{\textrm{l}}$, and which two batch sizes would you give
       the estimator for a model whose noise scale is about 2,000?

1. **Time-optimal and compute-optimal batch size.** Suppose one
   optimizer step at batch size $b$ takes
   $t(b) = t_0 (1 + b / b_{\textrm{sat}})$ seconds, where $b_{\textrm{sat}}$
   is the batch size that saturates the device. Time-to-target is then
   $T(b) = t(b)\, S(b)$, and compute-to-target is the example count
   $E(b) = b\, S(b)$, which by :eqref:`eq_steps-examples` is smallest at
   the smallest batch. Using :eqref:`eq_steps-examples`, find the batch
   size that minimizes $T(b)$, and express its $E(b)$ as a multiple of
   $E_{\min}$.
   :exlabel:`ex_batch-size-time-optimal-compute-optimal`

1. [code] **The two parts of the noise scale.** The last cell of this
   section measured the noise scale of `TinyLM` every 250 steps, too
   coarsely for the first 500 steps, in which it grows fastest. Write a
   variant of `noise_scale` that returns the noise power
   $\operatorname{tr} \boldsymbol{\Sigma}$ and the signal power
   $\|\nabla f\|^2$ as well as their ratio. Repeat the run with
   measurements at steps 0, 25, 50, 100, 200, 300, 400, 500, 1,000, 2,000,
   and 3,000, and record the held-out loss that `eval_loss` reports on
   `X_val` and `Y_val` at each of them.
   :exlabel:`ex_batch-size-noise-scale-over-training`
    1. Which of the two parts accounts for the growth of the noise scale of
       `TinyLM`? Repeat the measurement for `FashionCNN` at initialization
       and after 500 steps of Adam. Is the answer the same?
    1. Plot the noise scale of `TinyLM`, on a logarithmic axis, against the
       step and against the held-out loss. Which of the two plots would you
       use to predict the noise scale of a run at batch size 256, and why?
    1. Suppose a run starts at $b = 4$ and doubles its batch whenever
       $b_{\textrm{noise}}$ overtakes it. At which losses would the
       doublings land? Interpolate $\log b_{\textrm{noise}}$ linearly in
       the loss between checkpoints. Explain why the step counts of this
       run at $b = 64$ would not tell a ramped run when to double.

1. [code] **Past the bend.** Extend the `TinyLM` sweep of
   :numref:`subsec_batch-size-steps` to $b = 1024$ under the square-root
   rule of :eqref:`eq_lr-rules`, then rerun $b = 1024$ with $\eta$ held at
   its $b = 256$ value.
   :exlabel:`ex_batch-size-past-elbow`
    1. How do the step counts of the two $b = 1024$ runs compare with each
       other and with the $b = 256$ run?
    1. Explain both observations with :eqref:`eq_steps-examples` and the
       best learning rate $\eta_{\textrm{opt}}(b)$ of
       :eqref:`eq_best-step`, which levels off past the noise scale.

1. [code] **Retuning instead of scaling.** The CNN sweep recorded in
   `cnn_bs` and `cnn_steps` moved $\eta$ by the square-root rule, held fixed
   above 512 images, rather than retuning it. For each batch size, run a
   three-point learning-rate grid spaced by factors of two around the
   rule's value and keep the best steps-to-target. The middle point is
   already in `cnn_steps`, so this takes ten more runs. Which points of the
   curve move? Refit with `fit_steps`: does $b_{\textrm{crit}}$ move, and
   by more than the difference between two repetitions of the original
   sweep?

1. [code] **The square-root rule across the critical batch size.** ● The
   test of the square-root rule on the CNN stayed below the fitted critical
   batch size. Repeat it on `TinyLM` with batches of 16 and 256 sequences, which
   lie on either side of the fitted $b_{\textrm{crit}}$. Train with Adam
   from one initialization at constant learning rates a factor of two
   apart, from $0.25 \times 10^{-3}$ to $32 \times 10^{-3}$, and give both
   batch sizes the same 128,000 training sequences: 8,000 steps at batch
   size 16 and 500 steps at batch size 256.
    1. By what factor does the best learning rate move? Compare it with the
       factors 1, 4 and 16 of no scaling, the square-root rule and the
       linear rule.
    1. Does the shift reach the factor of the square-root rule? Which
       statement of :numref:`subsec_batch-size-limits` applies to a batch
       above the noise scale?
    1. What would you have concluded about training at batch size 256 had
       you reused the best learning rate of batch size 16?

1. [code] **Noise scale per parameter group.** ● The parameter groups of
   :numref:`subsec_tinylm` split `TinyLM`'s parameters into three groups:
   embeddings, matrices, and vectors (the biases and the LayerNorm scales
   and shifts). Restrict `grad_sq_norm` to one group at a time and measure
   its noise scale with `noise_scale`, at initialization and after 500
   steps of Adam.
    1. Which group has the largest noise scale, and does the ranking change
       during training?
    1. An embedding row receives a gradient only when its token occurs.
       Count the rows of `token_emb` that receive no gradient in a
       minibatch of 8 sequences, and state whether this sparsity can account
       for the embeddings' noise scale. Why does the same count for
       `pos_emb` always come out zero?
    1. What does the ranking imply for one global batch size serving all
       three groups?

1. [extended] **Measured time-compute tradeoff.** ● The step-time model
   $t(b)$ of Exercise :exref:`ex_batch-size-time-optimal-compute-optimal`
   has two constants, $t_0$ and $b_{\textrm{sat}}$, that depend on the
   hardware.
    1. Measure $t_0$ and $b_{\textrm{sat}}$ for `TinyLM` on your hardware
       by timing a gradient computation at several batch sizes with
       `d2l.Benchmark` (:numref:`subsec_async_dispatch`).
    1. Combine $t(b)$ with `lm_bs` and `lm_steps` and plot $T(b)$ against
       $E(b)$. Compare the time-optimal batch size on this curve with the
       prediction of Exercise
       :exref:`ex_batch-size-time-optimal-compute-optimal`, evaluated at
       the $b_{\textrm{crit}}$ that `fit_steps` returned for `TinyLM`, and
       say by what factor the time-optimal batch cuts $T(b)$ relative to
       $b = 4$ and by what factor it raises $E(b)$.

[Discussions](https://d2l.discourse.group/t/36765)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §9.10]{.kicker}

Batch Size<br>
**the gradient-noise scale · steps against examples · learning-rate rules · choosing a batch size**
:::
:::

::: {.slide title="Four Times the Batch: 3× Fewer Steps at 4 Sequences, Only 2× Fewer at 64"}
[The problem]{.kicker}

`TinyLM`, a transformer with 0.47 million parameters that predicts the
next byte of Wikipedia text, trained with Adam to a held-out loss of 1.9
nats per byte at several batch sizes $b$:

| batch (sequences) | steps to the target | examples processed |
|:--|:--|:--|
| $4 \to 16$ | a little more than 3× fewer | about a quarter more |
| $64 \to 256$ | about 2× fewer | about twice as many |

Four times the batch is four times the work per step.

::: {.d2l-note .warn}
Where do the savings stop, and can we tell before training?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- estimate the gradient-noise scale from gradient norms at two batch
  sizes;
- predict steps and examples to a target loss at any batch size;
- move a learning rate to a new batch size, for SGD and for Adam;
- say where these rules end;
- choose a batch size for a new model.

**Idea:** a larger batch saves steps until the noise in its gradient is as
small as the gradient itself.

**Running example:** `TinyLM`, so far at 64 sequences per step, and a CNN
on Fashion-MNIST (0.4 million parameters).
:::

::: {.slide title="Batch Size Sets the Noise That Each Step Carries"}
[Where this fits]{.kicker}

| part | decks |
|---|---|
| **1 · Two problems:** uneven curvature, noisy gradients | Landscapes · Gradient Descent · Stochastic and Minibatch GD |
| **2 · Direction:** which way to step | Two Test Models · Momentum · Adam · AdamW · Muon |
| **3 · The run:** step size over time, noise level, model width | Schedules · **Batch Size** · Scaling Up · Practice |

- **Uses:** gradient noise falls as $1/b$; the descent bound with
  curvature $L$; Adam's update.
- **Skips:** dividing a batch among devices; when to raise the batch
  within a run.
:::

::: {.slide title="A Minibatch Gradient Is Signal Plus Noise"}
[The gradient-noise scale]{.kicker}

- **Signal:** $\nabla f$, the gradient over the whole dataset.
- **Noise power** of one example, with $\nabla f_i$ its gradient:
  $\operatorname{tr}\boldsymbol{\Sigma} = \mathbb{E}_i\|\nabla f_i - \nabla f\|^2$.
- A minibatch gradient $\hat{\mathbf{g}}_b$ of $b$ examples has noise power
  $\operatorname{tr}\boldsymbol{\Sigma}/b$.

Signal and noise are equal at the **gradient-noise scale**

$$b_{\textrm{noise}} = \frac{\operatorname{tr} \boldsymbol{\Sigma}}{\|\nabla f\|^2}$$

::: {.d2l-note}
$\operatorname{tr}\boldsymbol{\Sigma} = 100$, $\|\nabla f\|^2 = 1$: one
example is 100× more noise than signal; $b = 100$ brings them level.
:::

[McCandlish et al., 2018]{.small}
:::

::: {.slide title="Gradient Norms at Two Batch Sizes Give Both Unknowns"}
[The gradient-noise scale]{.kicker}

A minibatch gradient is longer than the true one by its noise power:

$$\mathbb{E}\big[\|\hat{\mathbf{g}}_b\|^2\big] = \|\nabla f\|^2 +
\frac{\operatorname{tr} \boldsymbol{\Sigma}}{b}$$

Mean squared norms $N_s$, $N_l$ at batch sizes $b_s < b_l$:

$$\operatorname{tr} \boldsymbol{\Sigma} \approx
\frac{N_s - N_l}{1/b_s - 1/b_l},
\qquad
\|\nabla f\|^2 \approx \frac{b_l N_l - b_s N_s}{b_l - b_s}$$

::: {.d2l-note}
$N = 0.50$ at $b = 16$, $0.103$ at $b = 2{,}048$:
$\operatorname{tr}\boldsymbol{\Sigma} \approx 6.4$,
$\|\nabla f\|^2 \approx 0.10$, $b_{\textrm{noise}} \approx 64$. No
full-dataset gradient needed.
:::
:::

::: {.slide title="TinyLM's Noise Scale Grows More Than Tenfold within 500 Steps"}
[The gradient-noise scale]{.kicker}

CNN (images), then `TinyLM` (sequences of 64 bytes), before and after 500
steps of Adam:

@!batch-size-a-two-batch-estimator-3

@!batch-size-a-two-batch-estimator-4

- Two measurements of one network agree within a quarter; two seeds can
  differ tenfold.
- The noise scale is no constant of a model: it changes as the model
  trains.
:::

::: {.slide title="Noise Divides the Best Step and Its Progress by 1 + b_noise/b"}
[Batch size and learning rate]{.kicker}

Curvature at most $L$; step $-\eta\,\hat{\mathbf{g}}_b$ with learning rate
$\eta$:

**I.** $\;\mathbb{E}[\textrm{change of } f] \leq -\eta\,\|\nabla f\|^2 +
\frac{L}{2}\,\eta^2\big(\|\nabla f\|^2 + \operatorname{tr}\boldsymbol{\Sigma}/b\big)$

**II.** Minimize the parabola in $\eta$:

$$\boxed{\eta_{\textrm{opt}}(b) = \frac{1}{L}\cdot\frac{1}{1 + b_{\textrm{noise}}/b},
\qquad
\textrm{decrease} = \frac{\|\nabla f\|^2}{2L}\cdot\frac{1}{1 + b_{\textrm{noise}}/b}}$$

**III.** At $b = b_{\textrm{noise}}$: half the step and half the progress
of the exact gradient.
:::

::: {.slide title="Steps and Examples Trade along One Hyperbola"}
[Batch size and learning rate]{.kicker}

**The bold assumption:** steps needed $\propto$ 1 / progress per step,
at the best rate for each $b$.

$S(b)$: steps to a target loss; $E(b) = b\,S(b)$: examples; $S_{\min}$:
steps with the exact gradient; $E_{\min} = b_{\textrm{noise}}\,S_{\min}$.

$$\frac{S(b)}{S_{\min}} = 1 + \frac{b_{\textrm{noise}}}{b},
\qquad
\frac{E(b)}{E_{\min}} = 1 + \frac{b}{b_{\textrm{noise}}}$$

| $b_{\textrm{noise}} = 100$ | steps | examples |
|:--|:--|:--|
| $b = 10$ | $11\,S_{\min}$ | $1.1\,E_{\min}$ |
| $b = 100$: the **critical batch size** | $2\,S_{\min}$ | $2\,E_{\min}$ |
| $b = 1{,}000$ | $1.1\,S_{\min}$ | $11\,E_{\min}$ |
:::

::: {.slide title="Check: Steps and Examples on Either Side of the Critical Batch Size"}
[Check]{.kicker}

$b_{\textrm{noise}} = 100$. Compared with their minima, how many steps and
examples does a run need at $b = 25$? At $b = 400$?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $b = 25$: $S = 5\,S_{\min}$, $E = 1.25\,E_{\min}$.
$b = 400$: $S = 1.25\,S_{\min}$, $E = 5\,E_{\min}$. Sixteen times the
batch cuts the steps fourfold and quadruples the examples.
:::
:::
:::

::: {.slide title="Below the Noise Scale, SGD's Best Rate Grows in Proportion to the Batch"}
[Batch size and learning rate]{.kicker}

$$b \ll b_{\textrm{noise}}:\quad \eta_{\textrm{opt}}(b) \approx \frac{b}{L\, b_{\textrm{noise}}}
\qquad\Longrightarrow\qquad \eta(b) = \frac{b}{b_0}\,\eta_0$$

- $\eta_0$: the rate tuned at batch size $b_0$.
- $k$ steps at rate $\eta$ on $b$ examples ≈ one step at $k\eta$ on $kb$.

| $b_{\textrm{noise}} = 100$ | $b = 10$ | $b = 100$ | $b = 1{,}000$ |
|:--|:--|:--|:--|
| $\eta_{\textrm{opt}}$ | $1/(11L)$ | $1/(2L)$ | $1/(1.1L)$ |

::: {.d2l-note .warn}
Past the noise scale the best rate levels off at $1/L$.
:::

[Goyal et al., 2017; Smith et al., 2018]{.small}
:::

::: {.slide title="Adam's Division Supplies √k, the Learning Rate the Other √k"}
[Batch size and learning rate]{.kicker}

One coordinate: mean gradient $\bar{g}$, noise $\sigma/\sqrt{b}$ in a
batch of $b$. Adam divides by the root of its average squared gradient,
about $\sigma/\sqrt{b}$:

$$\textrm{step} \approx -\eta\,\frac{\bar{g} + z\,\sigma/\sqrt{b}}{\sigma/\sqrt{b}}
= -\eta\,\frac{\sqrt{b}}{\sigma}\,\bar{g} \;-\; \eta\,z,
\qquad z:\ \textrm{mean 0, variance 1}$$

- $k$ times the batch: $\sqrt{k}$ times the progress per step at the same
  $\eta$.
- One step for $k$: $\eta$ supplies the other $\sqrt{k}$.

$$\boxed{\eta(b) = \sqrt{b/b_0}\;\eta_0}$$

[Assumes noise larger than the mean gradient. Malladi et al., 2022]{.small}
:::

::: {.slide title="Prediction: Steps Fall with Slope −1 at Small Batches and Bend near the Noise Scale"}
[Steps to a target]{.kicker}

The test: train to one target loss at many batch sizes and count the steps.

| | `TinyLM` | CNN |
|:--|:--|:--|
| target | held-out loss 1.9 nats per byte | loss 0.35 on 8,192 fixed training images |
| batch sizes | 4 to 256 sequences | 8 to 2,048 images |
| Adam, rate set at | 0.003 at $b = 64$ | 0.004 at $b = 256$ |

- Rate moved by $\sqrt{b/b_0}$; for the CNN held fixed above 512.
- Target checked every 10 steps on fixed examples; one initialization per
  sweep.

[After Shallue et al., 2019, who retune at every batch size]{.small}
:::

::: {.slide title="TinyLM Follows Perfect Scaling at First, Then Bends"}
[Steps to a target]{.kicker}

Predicted with $b_{\textrm{noise}} = 100$: steps ÷ 3.6 from 4 to 16
sequences, ÷ 1.8 from 64 to 256.

@!batch-size-the-language-model

Measured: a little more than threefold, then about half.
:::

::: {.slide title="The CNN Gains Few Steps above a Few Hundred Images"}
[Steps to a target]{.kicker}

@!batch-size-the-cnn

$512 \to 2{,}048$ images: less than half of the steps saved, 2× to 4× the
examples.
:::

::: {.slide title="A Straight-Line Fit in 1/b Gives the Critical Batch Size"}
[Steps to a target]{.kicker}

$S(b) = S_{\min} + S_{\min}\, b_{\textrm{crit}} / b$ is linear in $1/b$:
intercept $S_{\min}$, slope over intercept $b_{\textrm{crit}}$.

@!batch-size-the-critical-batch-size-1

::: {.d2l-note}
Of the order of a hundred sequences and two to four hundred images: the
same order as the noise scales measured in the first 500 steps.
:::
:::

::: {.slide title="Both Models Fall on One Normalized Curve"}
[Steps to a target]{.kicker}

$\big(E/E_{\min} - 1\big)\big(S/S_{\min} - 1\big) = 1$, with
$E_{\min} = b_{\textrm{crit}} S_{\min}$:

@!batch-size-the-bill-in-examples

Upper left: many steps, few examples. Lower right: few steps, many
examples. Large runs sit near the bend.

[Two constants per model are fitted; the plot tests the shape.]{.small}
:::

::: {.slide title="SGD: Eight Times the Batch Moves the Best Rate about Eightfold"}
[Tests of the rules]{.kicker}

CNN, 12,800 examples per run, SGD with momentum 0.9, rates a factor of 2
apart:

@!batch-size-learning-rate-rules-2

The grid places a shallow minimum to within one step: a shift of 4 to 8.
Leaving the rate unchanged is ruled out.
:::

::: {.slide title="Check: Predict the Adam Sweep"}
[Check]{.kicker}

Adam on the CNN at $b = 2$ and at $b = 128$, on learning rates a factor of
2 apart. By how many grid steps does the best rate move under no scaling,
under the square-root rule, and under the linear rule?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** The batch ratio is 64. No scaling: 0 steps. Square root:
$\sqrt{64} = 8 = 2^3$, 3 steps. Linear: $64 = 2^6$, 6 steps. The three
predictions cannot be confused.
:::
:::
:::

::: {.slide title="Adam: 64 Times the Batch Moves the Best Rate about Eightfold"}
[Tests of the rules]{.kicker}

@!batch-size-learning-rate-rules-3

No scaling: 0.1 nats or more above the best run. Linear rule: never the
best.
:::

::: {.slide title="The Rules End at the Noise Scale, and a Scaled-Up Rate Needs Warmup"}
[Limits]{.kicker}

- **Past the noise scale:** the best rate levels off (SGD: at $1/L$); a
  rate scaled further overshoots.
- **At the start:** the noise scale is smallest (`TinyLM`: about 2
  sequences), so every batch is a large batch. A rate scaled far up is
  raised gradually over the first steps: warmup.
- **Test accuracy:** a large batch does not hurt by itself once rate and
  length of training are retuned.

::: {.d2l-note .rule}
The lasting cost of a large batch is data.
:::

[Goyal et al., 2017; Keskar et al., 2017; Shallue et al., 2019]{.small}
:::

::: {.slide title="TinyLM's Noise Scale Grows Two Orders of Magnitude in 3,000 Steps" layout="tight"}
[Choosing a batch size]{.kicker}

Adam at $b = 64$: about 2 sequences at step 0, a few hundred at step 3,000.

@!batch-size-growing-the-batch
:::

::: {.slide title="A Fixed Batch Size Is Right at One Moment of the Run"}
[Choosing a batch size]{.kicker}

| chosen for | early in training | late in training |
|:--|:--|:--|
| the start (small $b$) | efficient | parallelism unused |
| the end (large $b$) | examples wasted | efficient |

- `TinyLM` at $b = 64$: thirty times the noise scale at step 0, a few
  times below it at step 3,000.
- Large runs raise the batch during training. GPT-3: 32 thousand to 3.2
  million tokens.
- SGD: raising $b$ at fixed $\eta$ lowers the noise in the weights, as
  lowering $\eta$ does. Plan the ramp with the schedule.

[Brown et al., 2020; Smith et al., 2018]{.small}
:::

::: {.slide title="Fill the Device, Measure the Noise Scale, Stay Near It"}
[Choosing a batch size]{.kicker}

| step | rule |
|:--|:--|
| 1. device | at least the batch at which step time starts to grow; at most what fits in memory |
| 2. measure | the noise scale at initialization and after a few hundred steps; the later value governs |
| 3a. compute is scarce | stay below the noise scale: at a third of it, $1.33\,E_{\min}$ |
| 3b. time is scarce | go to about the noise scale: $2\,S_{\min}$ steps, $2\,E_{\min}$ examples |
| 4. learning rate | linear rule (SGD) or square-root rule (Adam), then sweep a factor of 2 |
| 5. later | measure again: the answer grows during training |

::: {.d2l-note .warn}
At four times the critical batch size: $1.25\,S_{\min}$ steps for
$5\,E_{\min}$ examples.
:::
:::

::: {.slide title="Check: The Second GPU"}
[Check]{.kicker}

`TinyLM` trains with Adam at $b = 64$ and $\eta = 0.003$; take
$b_{\textrm{noise}} = 100$. A second GPU arrives and you double the batch.
Which learning rate does the rule suggest, and by what factor does the
step count fall?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $\eta = \sqrt{2} \times 0.003 \approx 0.0042$. Steps:
$(1 + 100/64)/(1 + 100/128) = 2.56/1.78 \approx 1.4$ times fewer, not 2.
The run takes about 70% of the time and reads about 40% more data.
:::
:::
:::

::: {.slide title="A Larger Batch Saves Steps up to the Noise Scale"}
[Wrap-up]{.kicker}

- **Noise scale** $b_{\textrm{noise}} = \operatorname{tr}\boldsymbol{\Sigma}
  / \|\nabla f\|^2$, from gradient norms at two batch sizes: of the order
  of 100 images (CNN) and about 2 sequences (`TinyLM`) at initialization.
- **Steps to a target:** $S = S_{\min}(1 + b_{\textrm{noise}}/b)$; fitted
  critical batch sizes of the order of 100 sequences and 200 to 400
  images. From 64 to 128 sequences: 1.4 times fewer steps, not 2.
- **Learning rate:** $\propto b$ for SGD, $\propto \sqrt{b}$ for Adam (64×
  the batch, about 8× the rate), up to the noise scale.
- **During training** the noise scale of `TinyLM` grew by two orders of
  magnitude, so the right batch size grows.

**Next:** carrying a learning rate from a narrow network to a wide one.
:::
