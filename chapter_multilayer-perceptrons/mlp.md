```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Multilayer Perceptrons
:label:`sec_mlp`

In :numref:`sec_softmax`, we introduced
softmax regression,
implementing the algorithm from scratch
(:numref:`sec_softmax_scratch`) and using high-level APIs
(:numref:`sec_softmax_concise`). This allowed us to
train classifiers capable of recognizing
10 categories of clothing from low-resolution images.
Along the way, we learned how to wrangle data,
coerce our outputs into a valid probability distribution,
apply an appropriate loss function,
and minimize it with respect to our model's parameters.
Now that we have mastered these mechanics
in the context of simple linear models,
we can launch our exploration of deep neural networks,
the comparatively rich class of models
with which this book is primarily concerned.

```{.python .input #mlp-multilayer-perceptrons}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input #mlp-multilayer-perceptrons}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input #mlp-multilayer-perceptrons}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input #mlp-multilayer-perceptrons}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
from jax import grad, vmap
```

## Hidden Layers

We described affine transformations in
:numref:`subsec_linear_model` as
linear transformations with added bias.
To begin, recall the model architecture
corresponding to our softmax regression example,
illustrated in :numref:`fig_softmaxreg`.
This model maps inputs directly to outputs
via a single affine transformation,
followed by a softmax operation.
If our labels truly were related
to the input data by a simple affine transformation,
then this approach would be sufficient.
However, linearity (in affine transformations) is a *strong* assumption.

### Limitations of Linear Models

For example, linearity implies the *weaker*
assumption of *monotonicity*, i.e.,
that any increase in our feature must
either always cause an increase in our model's output
(if the corresponding weight is positive),
or always cause a decrease in our model's output
(if the corresponding weight is negative).
Sometimes that makes sense.
For example, if we were trying to predict
whether an individual will repay a loan,
we might reasonably assume that all other things being equal,
an applicant with a higher income
would always be more likely to repay
than one with a lower income.
While monotonic, this relationship likely
is not linearly associated with the probability of
repayment. An increase in income from \$0 to \$50,000
likely corresponds to a bigger increase
in likelihood of repayment
than an increase from \$1 million to \$1.05 million.
One way to handle this might be to postprocess our outcome
such that linearity becomes more plausible,
by using the logistic map (and thus the logarithm of the probability of outcome).

Note that we can easily come up with examples
that violate monotonicity.
Say for example that we want to predict health as a function
of body temperature.
For individuals with a normal body temperature
above 37°C (98.6°F),
higher temperatures indicate greater risk.
However, if the body temperatures drops
below 37°C, lower temperatures indicate greater risk!
Again, we might resolve the problem
with some clever preprocessing, such as using the distance from 37°C
as a feature.


But what about classifying images of cats and dogs?
Should increasing the intensity
of the pixel at location (13, 17)
always increase (or always decrease)
the likelihood that the image depicts a dog?
Reliance on a linear model corresponds to the implicit
assumption that the only requirement
for differentiating cats and dogs is to assess
the brightness of individual pixels.
This approach is doomed to fail in a world
where inverting an image preserves the category.

And yet despite the apparent absurdity of linearity here,
as compared with our previous examples,
it is less obvious that we could address the problem
with a simple preprocessing fix.
That is, because the significance of any pixel
depends in complex ways on its context
(the values of the surrounding pixels).
While there might exist a representation of our data
that would take into account
the relevant interactions among our features,
on top of which a linear model would be suitable,
we simply do not know how to calculate it by hand.
With deep neural networks, we use observational data
to jointly learn both a representation via hidden layers
and a linear predictor that acts upon that representation.

This problem of nonlinearity has been studied for at least a
century :cite:`Fisher.1928`. For instance, decision trees
in their most basic form use a sequence of binary decisions to
decide upon class membership :cite:`quinlan2014c4`. Likewise, kernel
methods have been used for many decades to model nonlinear dependencies
:cite:`Aronszajn.1950`. This has found its way into
nonparametric spline models :cite:`Wahba.1990` and kernel methods
:cite:`Scholkopf.Smola.2002`. It is also something that the brain solves
quite naturally. After all, neurons feed into other neurons which,
in turn, feed into other neurons again :cite:`Cajal.Azoulay.1894`.
Consequently we have a sequence of relatively simple transformations.

### Incorporating Hidden Layers

We can overcome the limitations of linear models
by incorporating one or more hidden layers.
The easiest way to do this is to stack
many fully connected layers on top of one another.
Each layer feeds into the layer above it,
until we generate outputs.
We can think of the first $L-1$ layers
as our representation and the final layer
as our linear predictor.
This architecture is commonly called
a *multilayer perceptron*,
often abbreviated as *MLP* (:numref:`fig_mlp`).

![An MLP with a hidden layer of five hidden units.](../img/mlp.svg)
:label:`fig_mlp`

This MLP has four inputs, three outputs,
and its hidden layer contains five hidden units.
Since the input layer does not involve any calculations,
producing outputs with this network
requires implementing the computations
for both the hidden and output layers;
thus, the number of layers in this MLP is two.
Note that both layers are fully connected.
Every input influences every neuron in the hidden layer,
and each of these in turn influences
every neuron in the output layer. Alas, we are not quite
done yet.

### From Linear to Nonlinear

As before, we denote by the matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$
a minibatch of $n$ examples where each example has $d$ inputs (features).
For a one-hidden-layer MLP whose hidden layer has $h$ hidden units,
we denote by $\mathbf{H} \in \mathbb{R}^{n \times h}$
the outputs of the hidden layer, which are
*hidden representations*.
Since the hidden and output layers are both fully connected,
we have hidden-layer weights $\mathbf{W}^{(1)} \in \mathbb{R}^{d \times h}$ and biases $\mathbf{b}^{(1)} \in \mathbb{R}^{1 \times h}$
and output-layer weights $\mathbf{W}^{(2)} \in \mathbb{R}^{h \times q}$ and biases $\mathbf{b}^{(2)} \in \mathbb{R}^{1 \times q}$.
This allows us to calculate the outputs $\mathbf{O} \in \mathbb{R}^{n \times q}$
of the one-hidden-layer MLP as follows:

$$
\begin{aligned}
    \mathbf{H} & = \mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}, \\
    \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.
\end{aligned}
$$

Note that after adding the hidden layer,
our model now requires us to track and update
additional sets of parameters.
So what have we gained in exchange?
You might be surprised to find out
that (in the model defined above) *we
gain nothing for our troubles*!
The reason is plain.
The hidden units above are given by
an affine function of the inputs,
and the outputs (pre-softmax) are just
an affine function of the hidden units.
An affine function of an affine function
is itself an affine function.
Moreover, our linear model was already
capable of representing any affine function.

To see this formally we can just collapse out the hidden layer in the above definition,
yielding an equivalent single-layer model with parameters
$\mathbf{W} = \mathbf{W}^{(1)}\mathbf{W}^{(2)}$ and $\mathbf{b} = \mathbf{b}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)}$:

$$
\mathbf{O} = (\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})\mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \mathbf{X} \mathbf{W}^{(1)}\mathbf{W}^{(2)} + \mathbf{b}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \mathbf{X} \mathbf{W} + \mathbf{b}.
$$

In order to realize the potential of multilayer architectures,
we need one more key ingredient: a
nonlinear *activation function* $\sigma$
to be applied to each hidden unit
following the affine transformation. For instance, a popular
choice is the ReLU (rectified linear unit) activation function :cite:`Nair.Hinton.2010`
$\sigma(x) = \mathrm{max}(0, x)$ operating on its arguments elementwise.
The outputs of activation functions $\sigma(\cdot)$
are called *activations*.
In general, with activation functions in place,
it is no longer possible to collapse our MLP into a linear model:

$$
\begin{aligned}
    \mathbf{H} & = \sigma(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}), \\
    \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.\\
\end{aligned}
$$

Since each row in $\mathbf{X}$ corresponds to an example in the minibatch,
with some abuse of notation, we define the nonlinearity
$\sigma$ to apply to its inputs in a rowwise fashion,
i.e., one example at a time.
Note that we used the same notation for softmax
when we denoted a rowwise operation in :numref:`subsec_softmax_vectorization`.
Quite frequently the activation functions we use apply not merely rowwise but
elementwise. That means that after computing the linear portion of the layer,
we can calculate each activation
without looking at the values taken by the other hidden units.

To build more general MLPs, we can continue stacking
such hidden layers,
e.g., $\mathbf{H}^{(1)} = \sigma_1(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})$
and $\mathbf{H}^{(2)} = \sigma_2(\mathbf{H}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)})$,
one atop another, yielding ever more expressive models.

### A Concrete Win: XOR

The collapse argument above told us what a hidden layer *cannot* do
without a nonlinearity. Let's now see what one *can* do once the
nonlinearity is in place, using the smallest problem that defeats every
linear model: the *exclusive-or* (XOR) function. Place four points at the
corners of the unit square and label each by whether its two coordinates
*differ*: $(0,0)$ and $(1,1)$ get label $0$, while $(0,1)$ and $(1,0)$ get
label $1$. As :numref:`fig_mdl-mlp-xor` shows on the left, the two classes
sit on opposite diagonals, so no straight line can put one class on each
side. A linear classifier is provably helpless here, no matter how we
choose its weights.

![XOR is not linearly separable, but one ReLU hidden layer makes it so. Left: the four corners of the unit square, coloured by the XOR label (the digit on each marker); the two classes lie on opposite diagonals, so any line misclassifies a corner. Right: the same four points after the hidden map $\mathbf{h} = \operatorname{ReLU}(\mathbf{x}\mathbf{W}^{(1)} + \mathbf{b}^{(1)})$ with $\mathbf{W}^{(1)} = \left(\begin{smallmatrix}1 & 1\\ 1 & 1\end{smallmatrix}\right)$ and $\mathbf{b}^{(1)} = (0, -1)$. The two class-1 corners are folded onto the *same* point $(1,0)$, and the cloud becomes linearly separable: the output neuron $h_1 - 2h_2$ now realizes XOR.](../img/mdl-mlp-xor.svg)
:label:`fig_mdl-mlp-xor`

A single hidden layer with just two ReLU units solves it. The trick is
that the hidden layer is free to *re-represent* the inputs, and a clever
representation can fold the two awkward corners together. The classic
choice (see :citet:`Goodfellow.Bengio.Courville.2016`, Chapter 6) uses

$$\mathbf{W}^{(1)} = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix},
  \quad \mathbf{b}^{(1)} = \begin{pmatrix} 0 & -1 \end{pmatrix},
  \quad \mathbf{w}^{(2)} = \begin{pmatrix} 1 \\ -2 \end{pmatrix},
  \quad b^{(2)} = 0,$$

with a ReLU on the hidden layer. The first hidden unit fires for any
"active" input; the second only fires when *both* coordinates are on,
and subtracting twice the second unit cancels the lone case the first
unit gets wrong. The right panel of :numref:`fig_mdl-mlp-xor` plots the
hidden representation: the two label-1 corners land on top of each other
at $(1,0)$, after which a single line separates the classes. Let's verify
that this hand-built network computes XOR exactly on all four inputs.

```{.python .input #mlp-xor}
%%tab pytorch
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
W1 = torch.tensor([[1., 1.], [1., 1.]])
b1 = torch.tensor([0., -1.])
w2 = torch.tensor([[1.], [-2.]])
H = torch.relu(X @ W1 + b1)          # hidden features, ReLU applied elementwise
O = (H @ w2).squeeze()               # output neuron (pre-threshold)
torch.stack([X[:, 0], X[:, 1], (O > 0.5).float()], dim=1)  # x1, x2, prediction
```

The third column is exactly the XOR of the first two. We *constructed* the
weights here, but the whole point of the rest of this book is that
optimization can *discover* such representations from data. The XOR fix
generalizes: stack nonlinear hidden layers and the network can carve the
input space into arbitrarily intricate regions.

### Universal Approximators

We know that the brain is capable of very sophisticated statistical analysis. As such,
it is worth asking, just *how powerful* a deep network could be. The reassuring answer is
the *universal approximation theorem*. It says that even a single-hidden-layer
network, given enough hidden units and the right weights, can approximate any
continuous function on a bounded domain to arbitrary accuracy. This was proven
in several settings: :citet:`Cybenko.1989` did it for sigmoid activations,
:citet:`micchelli1984interpolation` for radial basis function networks (single
hidden layer, in the context of reproducing kernel Hilbert spaces), and the
result was soon generalized to essentially *any* sensible activation, that is,
any bounded, non-constant function (Hornik, 1991). The conclusion does not hinge
on which of ReLU, sigmoid, or tanh we pick.

It is tempting to read this as "one hidden layer is all you ever need," but the
theorem is more modest than it sounds, and three caveats matter
(:citet:`Goodfellow.Bengio.Courville.2016`, Chapter 6). First, it guarantees
that a good approximation *exists*; it says nothing about whether gradient
descent will *find* it. Second, even a network that fits the training data
perfectly may fail to *generalize* to new examples. Third, the promised single
layer can be impractically wide: matching a target may require *exponentially*
many hidden units. You might think of your neural network as being a bit like
the C programming language. The language, like any other modern language, is
capable of expressing any computable program, but actually coming up with a
program that meets your specifications is the hard part.

So the theorem tells us deep networks are expressive enough; it does not tell us
they are the right tool, nor how to build them. For some problems other methods
fit better, for instance kernel methods can solve regression problems *exactly*,
even in infinite-dimensional spaces :cite:`Kimeldorf.Wahba.1971,Scholkopf.Herbrich.Smola.2001`.
And crucially, where a shallow network would need exponential width, a *deep* one
can often represent the same function far more compactly, trading width for depth
:cite:`Simonyan.Zisserman.2014`. This is one reason practitioners reach for depth
rather than sheer width. We will touch upon more rigorous arguments in subsequent
chapters.


## Activation Functions
:label:`subsec_activation-functions`

Activation functions are differentiable operators for transforming
pre-activation signals to outputs, introducing nonlinearity into the network.
Because activation functions are fundamental to deep learning,
let's briefly survey some common ones.

### ReLU Function

The most popular choice,
due to both simplicity of implementation and
its good performance on a variety of predictive tasks,
is the *rectified linear unit* (*ReLU*) :cite:`Nair.Hinton.2010`.
ReLU provides a very simple nonlinear transformation.
Given an element $x$, the function is defined
as the maximum of that element and $0$:

$$\operatorname{ReLU}(x) = \max(x, 0).$$

Informally, the ReLU function retains only positive
elements and discards all negative elements
by setting the corresponding activations to 0.
To gain some intuition, we can plot the function.
As you can see, the activation function is piecewise linear.

```{.python .input #mlp-relu-function-1}
%%tab mxnet
x = np.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = npx.relu(x)
d2l.plot(x, y, 'x', 'relu(x)', figsize=(5, 2.5))
```

```{.python .input #mlp-relu-function-1}
%%tab pytorch
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
```

```{.python .input #mlp-relu-function-1}
%%tab tensorflow
x = tf.Variable(tf.range(-8.0, 8.0, 0.1), dtype=tf.float32)
y = tf.nn.relu(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'relu(x)', figsize=(5, 2.5))
```

```{.python .input #mlp-relu-function-1}
%%tab jax
x = jnp.arange(-8.0, 8.0, 0.1)
y = jax.nn.relu(x)
d2l.plot(x, y, 'x', 'relu(x)', figsize=(5, 2.5))
```

When the input is negative,
the derivative of the ReLU function is 0,
and when the input is positive,
the derivative of the ReLU function is 1.
Note that the ReLU function is not differentiable
when the input takes value precisely equal to 0.
In these cases, we default to the left-hand-side
derivative and say that the derivative is 0 when the input is 0.
We can get away with this because
the input may never actually be zero (mathematicians would
say that it is nondifferentiable on a set of measure zero).
There is an old adage that if subtle boundary conditions matter,
we are probably doing (*real*) mathematics, not engineering.
That conventional wisdom may apply here, or at least, the fact that
we are not performing constrained optimization :cite:`Mangasarian.1965,Rockafellar.1970`.
We plot the derivative of the ReLU function below.

```{.python .input #mlp-relu-function-2}
%%tab mxnet
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
```

```{.python .input #mlp-relu-function-2}
%%tab pytorch
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
```

```{.python .input #mlp-relu-function-2}
%%tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.relu(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of relu',
         figsize=(5, 2.5))
```

```{.python .input #mlp-relu-function-2}
%%tab jax
grad_relu = vmap(grad(jax.nn.relu))
d2l.plot(x, grad_relu(x), 'x', 'grad of relu', figsize=(5, 2.5))
```

The reason for using ReLU is that
its derivatives are particularly well behaved:
either they vanish or they just let the argument through.
This makes optimization better behaved
and it mitigated the well-documented problem
of vanishing gradients that plagued
previous versions of neural networks (more on this later).

This same flatness has a downside, however. Because the gradient is exactly
zero for negative inputs, a unit whose pre-activation is pushed negative for
every training example receives no gradient and stops updating: it becomes a
permanently silent *dead ReLU*. To keep gradient flowing in that regime, a
number of variants let a little signal through on the left. The best known is
the *parametrized ReLU* (*pReLU*) :cite:`He.Zhang.Ren.ea.2015`, which adds a
linear term so some information still gets through, even when the argument is
negative:

$$\operatorname{pReLU}(x) = \max(0, x) + \alpha \min(0, x).$$

Here $\alpha$ is a small slope (fixed for *leaky* ReLU, learned for pReLU).

### Sigmoid Function

The *sigmoid function* transforms those inputs
whose values lie in the domain $\mathbb{R}$,
to outputs that lie on the interval (0, 1).
For that reason, the sigmoid is
often called a *squashing function*:
it squashes any input in the range (-inf, inf)
to some value in the range (0, 1):

$$\operatorname{sigmoid}(x) = \frac{1}{1 + \exp(-x)}.$$

In the earliest neural networks, scientists
were interested in modeling biological neurons
that either *fire* or *do not fire*.
Thus the pioneers of this field,
going all the way back to McCulloch and Pitts,
the inventors of the artificial neuron,
focused on thresholding units :cite:`McCulloch.Pitts.1943`.
A thresholding activation takes value 0
when its input is below some threshold
and value 1 when the input exceeds the threshold.

When attention shifted to gradient-based learning,
the sigmoid function was a natural choice
because it is a smooth, differentiable
approximation to a thresholding unit.
Sigmoids are still widely used as
activation functions on the output units
when we want to interpret the outputs as probabilities
for binary classification problems: you can think of the sigmoid as a special case of the softmax, namely the softmax over the two logits $\{x, 0\}$.
However, the sigmoid has largely been replaced
by the simpler and more easily trainable ReLU
for most use in hidden layers. Much of this has to do
with the fact that the sigmoid poses challenges for optimization
:cite:`LeCun.Bottou.Orr.ea.1998` since its gradient vanishes for large positive *and* negative arguments.
This can lead to plateaus that are difficult to escape from.
Nonetheless sigmoids are important. In later chapters (e.g., :numref:`sec_lstm`) on recurrent neural networks,
we will describe architectures that leverage sigmoid units
to control the flow of information across time.

Below, we plot the sigmoid function.
Note that when the input is close to 0,
the sigmoid function approaches
a linear transformation.

```{.python .input #mlp-sigmoid-function-1}
%%tab mxnet
with autograd.record():
    y = npx.sigmoid(x)
d2l.plot(x, y, 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

```{.python .input #mlp-sigmoid-function-1}
%%tab pytorch
y = torch.sigmoid(x)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

```{.python .input #mlp-sigmoid-function-1}
%%tab tensorflow
y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

```{.python .input #mlp-sigmoid-function-1}
%%tab jax
y = jax.nn.sigmoid(x)
d2l.plot(x, y, 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

The derivative of the sigmoid function is given by the following equation:

$$\frac{d}{dx} \operatorname{sigmoid}(x) = \frac{\exp(-x)}{(1 + \exp(-x))^2} = \operatorname{sigmoid}(x)\left(1-\operatorname{sigmoid}(x)\right).$$


The derivative of the sigmoid function is plotted below.
Note that when the input is 0,
the derivative of the sigmoid function
reaches a maximum of 0.25.
As the input diverges from 0 in either direction,
the derivative approaches 0.

```{.python .input #mlp-sigmoid-function-2}
%%tab mxnet
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
```

```{.python .input #mlp-sigmoid-function-2}
%%tab pytorch
# Clear out previous gradients
x.grad.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
```

```{.python .input #mlp-sigmoid-function-2}
%%tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of sigmoid',
         figsize=(5, 2.5))
```

```{.python .input #mlp-sigmoid-function-2}
%%tab jax
grad_sigmoid = vmap(grad(jax.nn.sigmoid))
d2l.plot(x, grad_sigmoid(x), 'x', 'grad of sigmoid', figsize=(5, 2.5))
```

### Tanh Function
:label:`subsec_tanh`

Like the sigmoid function, the tanh (hyperbolic tangent)
function also squashes its inputs,
transforming them into elements on the interval between $-1$ and $1$:

$$\operatorname{tanh}(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}.$$

We plot the tanh function below. Note that as input nears 0, the tanh function approaches a linear transformation. Although the shape of the function is similar to that of the sigmoid function, the tanh function exhibits point symmetry about the origin of the coordinate system :cite:`Kalman.Kwasny.1992`.

```{.python .input #mlp-tanh-function-1}
%%tab mxnet
with autograd.record():
    y = np.tanh(x)
d2l.plot(x, y, 'x', 'tanh(x)', figsize=(5, 2.5))
```

```{.python .input #mlp-tanh-function-1}
%%tab pytorch
y = torch.tanh(x)
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
```

```{.python .input #mlp-tanh-function-1}
%%tab tensorflow
y = tf.nn.tanh(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'tanh(x)', figsize=(5, 2.5))
```

```{.python .input #mlp-tanh-function-1}
%%tab jax
y = jax.nn.tanh(x)
d2l.plot(x, y, 'x', 'tanh(x)', figsize=(5, 2.5))
```

The derivative of the tanh function is:

$$\frac{d}{dx} \operatorname{tanh}(x) = 1 - \operatorname{tanh}^2(x).$$

It is plotted below.
As the input nears 0,
the derivative of the tanh function approaches a maximum of 1.
And as we saw with the sigmoid function,
as input moves away from 0 in either direction,
the derivative of the tanh function approaches 0.

```{.python .input #mlp-tanh-function-2}
%%tab mxnet
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
```

```{.python .input #mlp-tanh-function-2}
%%tab pytorch
# Clear out previous gradients
x.grad.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
```

```{.python .input #mlp-tanh-function-2}
%%tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.tanh(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of tanh',
         figsize=(5, 2.5))
```

```{.python .input #mlp-tanh-function-2}
%%tab jax
grad_tanh = vmap(grad(jax.nn.tanh))
d2l.plot(x, grad_tanh(x), 'x', 'grad of tanh', figsize=(5, 2.5))
```

## Summary and Discussion

We now know how to incorporate nonlinearities
to build expressive multilayer neural network architectures.
Your knowledge already puts you in command of a toolkit
much like that of a practitioner circa 1990, except that you can lean on
powerful open-source frameworks to build models in a few lines of code,
rather than coding up layers and their derivatives by hand in C or Fortran.

A key reason ReLU displaced sigmoid and tanh in hidden layers is that it
is so much more amenable to optimization. One could argue that this was one
of the innovations that helped the resurgence of deep learning in the early
2010s. Research on activation functions has not stopped, though, and you will
meet newer ones once we reach the Transformer architectures later in the book.
The most common are *GELU* (Gaussian error linear unit), $x \Phi(x)$, where
$\Phi$ is the standard Gaussian cumulative distribution function
:cite:`Hendrycks.Gimpel.2016`, used in BERT and the GPT family; *Swish*,
$x \operatorname{sigmoid}(\beta x)$ :cite:`Ramachandran.Zoph.Le.2017`; and
*SwiGLU*, a gated variant that is the default feedforward nonlinearity in
recent large language models such as PaLM, LLaMA, and Mistral. For now, ReLU
remains the sensible default for the models we build next.

## Exercises

1. Show that adding layers to a *linear* deep network, i.e., a network without
   nonlinearity $\sigma$ can never increase the expressive power of the network.
   Give an example where it actively reduces it.
1. Find weights for a two-hidden-unit ReLU network that computes XOR, and verify
   them on the four inputs. (You may reuse the construction in
   :numref:`fig_mdl-mlp-xor`, but try to derive your own first.) Can a *single*
   ReLU unit compute XOR? Why or why not?
1. Compute the derivative of the pReLU activation function.
1. Compute the derivative of the Swish activation function $x \operatorname{sigmoid}(\beta x)$.
1. Show that an MLP using only ReLU (or pReLU) constructs a
   continuous piecewise linear function.
1. Explain intuitively why composing ReLU layers can roughly *double* the number
   of linear pieces the network represents with each added layer, so that depth
   buys exponentially many pieces while width buys only linearly many. (This is
   the depth-versus-width gap behind the universal-approximation caveat above.)
1. Sigmoid and tanh are very similar.
    1. Show that $\operatorname{tanh}(x) + 1 = 2 \operatorname{sigmoid}(2x)$.
    1. Prove that the function classes parametrized by both nonlinearities are identical. Hint: affine layers have bias terms, too.
1. Assume that we have a nonlinearity that applies to one minibatch at a time, such as the batch normalization :cite:`Ioffe.Szegedy.2015` (covered in :numref:`sec_batch_norm`). What kinds of problems do you expect this to cause?
1. Provide an example where the gradients vanish for the sigmoid activation function.

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/90)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/91)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/226)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/17984)
:end_tab:

<!-- slides -->

::: {.slide title="MLPs add nonlinear hidden layers"}
A **multilayer perceptron** (MLP) is a stack of
fully-connected layers separated by elementwise
nonlinearities. The simplest deep network — and the
foundation everything else in this book builds on.

A linear classifier draws one hyperplane per class.
That's not enough for most things we want to model:

- **Body temperature → health risk** — U-shaped, not
  even monotonic.
- **Cat vs dog from pixels** — the meaning of pixel
  $(13, 17)$ depends on its neighbors.
- **XOR** — the canonical small problem a linear model
  *provably* cannot solve.
:::

::: {.slide title="The fix: alternate linear and nonlinear"}
Stack linear layers with a **nonlinearity** between them.
The linear layers mix features; the nonlinearity lets the
composition curve, fold, and twist the decision surface.

That's it. Two ingredients, deep architectures from there.
:::

::: {.slide title="Architecture"}
An MLP is a stack of fully-connected layers. The middle
layers are *hidden* — neither input nor output:

![One hidden layer with five units, four inputs, three outputs.](../img/mlp.svg){width=58%}

Math for the one-hidden-layer case (minibatch
$\mathbf{X} \in \mathbb{R}^{n \times d}$, hidden width $h$,
$q$ outputs):

$$\mathbf{H} = \mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}, \qquad
  \mathbf{O} = \mathbf{H} \mathbf{W}^{(2)} + \mathbf{b}^{(2)}.$$

Two layers. Two weight matrices. Two biases. So far it
looks like genuine progress.
:::

::: {.slide title="Why naïve stacking doesn't help"}
Plug $\mathbf{H}$ from the first equation into the second:

$$\mathbf{O} = (\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})\,\mathbf{W}^{(2)} + \mathbf{b}^{(2)}
            = \mathbf{X}\,\underbrace{\mathbf{W}^{(1)}\mathbf{W}^{(2)}}_{=\mathbf{W}} + \underbrace{\mathbf{b}^{(1)}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}}_{=\mathbf{b}}.$$

A composition of affine maps is just another affine map.
The hidden layer adds *zero* expressive power — same model
class as plain softmax regression.

You need a **nonlinearity** between the layers, or stacking
is wasted.
:::

::: {.slide title="Activation functions: the missing ingredient"}
Insert an elementwise nonlinearity $\sigma$ after every
hidden layer:

$$\mathbf{H} = \sigma(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}),\qquad
  \mathbf{O} = \mathbf{H} \mathbf{W}^{(2)} + \mathbf{b}^{(2)}.$$

Now the network represents a **piecewise nonlinear**
function — and stacking actually buys us something.

**Universal approximation theorem** (Cybenko 1989):
a single hidden layer with enough units, plus a sane
$\sigma$, can approximate any continuous function
arbitrarily well.

Caveat: "enough units" can be exponentially many. Depth
trades width for parameter efficiency — the modern reason
deep nets work.
:::

::: {.slide title="Setup"}
@mlp-multilayer-perceptrons
:::

::: {.slide title="ReLU — the modern default"}
$$\mathrm{ReLU}(x) = \max(0, x).$$

@mlp-relu-function-1

Three reasons it dominates:

- **Doesn't saturate on the right** — gradient is exactly
  1 for any $x > 0$. No vanishing gradient.
- **Cheap** — one comparison, one max. No exponential.
- **Sparse activations** — half the units output zero on
  average; acts as implicit regularization.
:::

::: {.slide title="ReLU's derivative"}
The derivative is just the step function — 0 for negative
inputs, 1 for positive:

$$\mathrm{ReLU}'(x) = \mathbb{1}[x > 0].$$

@mlp-relu-function-2
:::

::: {.slide title="Dead ReLU"}
A unit whose pre-activation is always negative gets zero
gradient and never updates again — a permanently silent
neuron.

The fix: **LeakyReLU / PReLU** —
$\max(0, x) + \alpha\min(0, x)$, with a small slope on the
left to keep gradient flowing.
:::

::: {.slide title="Sigmoid — squashes to (0, 1)"}
$$\sigma(x) = \frac{1}{1 + e^{-x}}.$$

@mlp-sigmoid-function-1

The original neural net activation (1960s–2000s). Today
mostly used for:

- **Output layers** in binary classification
  (probability ∈ (0, 1)).
- **Gates** in LSTM/GRU and attention (still ∈ (0, 1)).

For *hidden* layers it's been replaced by ReLU: see why on
the next slide.
:::

::: {.slide title="Why sigmoid hurts deep nets"}
$$\sigma'(x) = \sigma(x)(1 - \sigma(x)).$$

@mlp-sigmoid-function-2

Maximum gradient is $\sigma'(0) = 0.25$. Worse, $\sigma'$
**vanishes** for $|x| \gtrsim 5$.

In a 10-layer net with sigmoid activations, the backward
pass multiplies $\le 0.25$ at every layer — gradients shrink
by $\le 4^{-10} \approx 10^{-6}$ before reaching the input
layer. That's the **vanishing gradient** problem ReLU
solved.
:::

::: {.slide title="Tanh — sigmoid's symmetric cousin"}
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\sigma(2x) - 1.$$

@mlp-tanh-function-1

Range $(-1, 1)$ — **zero-centered**, which mildly helps
optimization. Default in RNNs (LSTM cell update, GRU
candidate hidden state) where bounded activations are
useful.
:::

::: {.slide title="Tanh's derivative"}
Still saturates at both tails — same vanishing-gradient
issue as sigmoid:

@mlp-tanh-function-2
:::

::: {.slide title="Cheat sheet"}
| | Range | Saturates? | Use case |
|---|---|---|---|
| **ReLU** | $[0, \infty)$ | only at $x{<}0$ (dead) | default for hidden |
| **LeakyReLU / PReLU** | $\mathbb{R}$ | no | when ReLU dies |
| **GELU** ($x\Phi(x)$) | $\approx \mathbb{R}$ | barely | Transformers, modern LLMs |
| **Sigmoid** | $(0, 1)$ | both ends | gates, binary output |
| **Tanh** | $(-1, 1)$ | both ends | RNN cells |
| **Softmax** | simplex | one end | multiclass output |

Default: ReLU for hidden layers, GELU if you're imitating
modern Transformer models, sigmoid/softmax at outputs to
turn logits into probabilities.
:::

::: {.slide title="Recap"}
- An MLP = several affine layers, with an *elementwise*
  nonlinearity between them.
- The nonlinearity is essential — without it the stack
  collapses to a single affine map.
- One sufficiently wide hidden layer is a universal
  approximator. Depth makes the same expressiveness
  *parameter-efficient*.
- **ReLU** is the modern default. Sigmoid and tanh persist
  in specific roles (output, gates, RNNs) where their
  bounded ranges are useful.
- The whole rest of this chapter is about *training* MLPs:
  forward pass, backprop, init, regularization.
:::
