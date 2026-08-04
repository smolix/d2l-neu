```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Automatic Differentiation
:label:`sec_autograd`

Recall from :numref:`sec_calculus` 
that derivatives drive all the optimization algorithms
that we will use to train deep networks.
Even when the individual rules are elementary, applying them by hand becomes
tedious and error-prone as models grow more complex.

Modern deep learning frameworks perform this calculation using
*automatic differentiation*
(often shortened to *autograd*). 
As we pass data through each successive function,
the framework builds a *computational graph* 
that tracks how each value depends on others.
To calculate derivatives, 
automatic differentiation 
works backwards through this graph
applying the chain rule. 
The computational algorithm for applying the chain rule
in this fashion is called *backpropagation*.

Autograd has a long history:
the earliest references date back
over half a century :cite:`Wengert.1964`,
and reverse mode, the variant that powers
modern backpropagation, was developed
by :citet:`Linnainmaa.1970`.
:numref:`sec_mdl-matrix-calculus-autodiff`
recounts this history in full.
We begin with the basic automatic-differentiation interface and then examine
how computational graphs are controlled.

```{.python .input #autograd-automatic-differentiation}
%%tab mxnet
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input #autograd-automatic-differentiation}
%%tab pytorch
import torch
```

```{.python .input #autograd-automatic-differentiation}
%%tab tensorflow
import tensorflow as tf
```

```{.python .input #autograd-automatic-differentiation}
%%tab jax
from jax import numpy as jnp
```

## Mechanics

We begin with the basic workflow (attach a gradient,
record a computation, run the backward pass), first
on a scalar-valued function, then on vector-valued ones.

### A Simple Function

Consider differentiating the function
$y = 2\mathbf{x}^{\top}\mathbf{x}$
with respect to the column vector $\mathbf{x}$.
To start, we assign `x` an initial value.

```{.python .input #autograd-a-simple-function-1}
%%tab mxnet
x = np.arange(4.0)
x
```

```{.python .input #autograd-a-simple-function-1}
%%tab pytorch
x = torch.arange(4.0)
x
```

```{.python .input #autograd-a-simple-function-1}
%%tab tensorflow
x = tf.range(4, dtype=tf.float32)
x
```

```{.python .input #autograd-a-simple-function-1}
%%tab jax
x = jnp.arange(4.0)
x
```

:begin_tab:`mxnet, pytorch, tensorflow`
Before we calculate the gradient
of $y$ with respect to $\mathbf{x}$,
we need a place to store it.
In general, we avoid allocating new memory
every time we take a derivative 
because deep learning requires 
successively computing derivatives
with respect to the same parameters
a great many times,
and repeated allocation would add unnecessary overhead. The gradient of a
scalar-valued function
with respect to a vector $\mathbf{x}$
is vector-valued with 
the same shape as $\mathbf{x}$.
:end_tab:

```{.python .input #autograd-a-simple-function-2}
%%tab mxnet
# We allocate memory for a tensor's gradient by invoking `attach_grad`
x.attach_grad()
# After we calculate a gradient taken with respect to `x`, we will be able to
# access it via the `grad` attribute, whose values are initialized with 0s
x.grad
```

```{.python .input #autograd-a-simple-function-2}
%%tab pytorch
# Can also create x = torch.arange(4.0, requires_grad=True)
x.requires_grad_(True)
x.grad  # The gradient is None by default
```

```{.python .input #autograd-a-simple-function-2}
%%tab tensorflow
x = tf.Variable(x)
```

We now calculate our function of `x` and assign the result to `y`.

```{.python .input #autograd-a-simple-function-3}
%%tab mxnet
# Our code is inside an `autograd.record` scope to build the computational
# graph
with autograd.record():
    y = 2 * np.dot(x, x)
y
```

```{.python .input #autograd-a-simple-function-3}
%%tab pytorch
y = 2 * torch.dot(x, x)
y
```

```{.python .input #autograd-a-simple-function-3}
%%tab tensorflow
# Record all computations onto a tape
with tf.GradientTape() as t:
    y = 2 * tf.tensordot(x, x, axes=1)
y
```

```{.python .input #autograd-a-simple-function-3}
%%tab jax
y = lambda x: 2 * jnp.dot(x, x)
y(x)
```

Recording the operations gives the framework a *computational graph*,
shown in :numref:`fig_autograd_graph`. Its nodes are operations and its
edges carry intermediate values.

![The computational graph for $y = 2\mathbf{x}^\top\mathbf{x}$. The forward pass (black) flows from $\mathbf{x}$ to $y$; reverse-mode automatic differentiation walks the same graph backward (blue), multiplying the local derivative at each node via the chain rule to accumulate $\partial y / \partial \mathbf{x} = 4\mathbf{x}$.](../img/autograd-comp-graph.svg)
:label:`fig_autograd_graph`

The forward pass evaluates the graph from $\mathbf{x}$ to $y$; to obtain
the gradient, automatic differentiation then traverses it in reverse,
multiplying the local derivatives along the way. We unpack computational
graphs and backpropagation in full in :numref:`sec_backprop`,
and the underlying mathematics (both modes of automatic differentiation
and their costs) is developed in
:numref:`sec_mdl-matrix-calculus-autodiff`; for now we
use the resulting gradients.

:begin_tab:`mxnet`
We can now take the gradient of `y`
with respect to `x` by calling 
its `backward` method.
Next, we can access the gradient 
via `x`'s `grad` attribute.
:end_tab:

:begin_tab:`pytorch`
We can now take the gradient of `y`
with respect to `x` by calling 
its `backward` method.
Next, we can access the gradient 
via `x`'s `grad` attribute.
:end_tab:

:begin_tab:`tensorflow`
We can now calculate the gradient of `y`
with respect to `x` by calling 
the `gradient` method.
:end_tab:

:begin_tab:`jax`
We can now take the gradient of `y`
with respect to `x` by passing through the
`grad` transform.
:end_tab:

```{.python .input #autograd-a-simple-function-4}
%%tab mxnet
y.backward()
x.grad
```

```{.python .input #autograd-a-simple-function-4}
%%tab pytorch
y.backward()
x.grad
```

```{.python .input #autograd-a-simple-function-4}
%%tab tensorflow
x_grad = t.gradient(y, x)
x_grad
```

```{.python .input #autograd-a-simple-function-4}
%%tab jax
from jax import grad
# The `grad` transform returns a Python function that
# computes the gradient of the original function
x_grad = grad(y)(x)
x_grad
```

We already know that the gradient of the function $y = 2\mathbf{x}^{\top}\mathbf{x}$
with respect to $\mathbf{x}$ should be $4\mathbf{x}$.
We can now verify that the automatic gradient computation
and the expected result are identical.

```{.python .input #autograd-a-simple-function-5}
%%tab mxnet
x.grad == 4 * x
```

```{.python .input #autograd-a-simple-function-5}
%%tab pytorch
x.grad == 4 * x
```

```{.python .input #autograd-a-simple-function-5}
%%tab tensorflow
x_grad == 4 * x
```

```{.python .input #autograd-a-simple-function-5}
%%tab jax
x_grad == 4 * x
```

:begin_tab:`mxnet`
For the new objective $y=\sum_i x_i$, MXNet replaces the recorded gradient
with the new value.
:end_tab:

:begin_tab:`pytorch`
For the new objective $y=\sum_i x_i$, PyTorch adds the new derivative to the
existing gradient buffer unless it is cleared first.
This behavior supports optimizing the sum
of multiple objective functions.
To reset the gradient buffer,
we can call `x.grad.zero_()` as follows:
:end_tab:

:begin_tab:`tensorflow`
For the new objective $y=\sum_i x_i$, `GradientTape.gradient` returns a new
gradient value rather than accumulating into a tensor-owned buffer.
:end_tab:

```{.python .input #autograd-a-simple-function-6}
%%tab mxnet
with autograd.record():
    y = x.sum()
y.backward()
x.grad  # Overwritten by the newly calculated gradient
```

```{.python .input #autograd-a-simple-function-6}
%%tab pytorch
x.grad.zero_()  # Reset the gradient
y = x.sum()
y.backward()
x.grad
```

```{.python .input #autograd-a-simple-function-6}
%%tab tensorflow
with tf.GradientTape() as t:
    y = tf.reduce_sum(x)
t.gradient(y, x)  # Overwritten by the newly calculated gradient
```

```{.python .input #autograd-a-simple-function-6}
%%tab jax
y = lambda x: x.sum()
grad(y)(x)
```

### Backward for Non-Scalar Variables

When `y` is a vector, 
the most natural representation 
of the derivative of  `y`
with respect to a vector `x` 
is a matrix called the *Jacobian*
that contains the partial derivatives
of each component of `y` 
with respect to each component of `x`.
Likewise, for higher-order `y` and `x`,
the result of differentiation could be an even higher-order tensor.

While Jacobians do show up in some
advanced machine learning techniques,
more commonly we sum
the gradients of each component of `y`
with respect to the full vector `x`,
yielding a vector of the same shape as `x`.
For example, we often have a vector 
representing the value of our loss function
calculated separately for each example among
a batch of training examples.
Here, we sum the gradients
computed individually for each example.

:begin_tab:`mxnet`
MXNet handles this problem by reducing all tensors to scalars 
by summing before computing a gradient. 
In other words, rather than returning the Jacobian 
$\partial_{\mathbf{x}} \mathbf{y}$,
it returns the gradient of the sum
$\partial_{\mathbf{x}} \sum_i y_i$. 
:end_tab:

:begin_tab:`pytorch`
Because deep learning frameworks vary 
in how they interpret gradients of
non-scalar tensors,
PyTorch takes some steps to avoid confusion.
Invoking `backward` on a non-scalar elicits an error 
unless we tell PyTorch how to reduce the object to a scalar. 
More formally, we need to provide some vector $\mathbf{v}$ 
such that `backward` will compute 
$\mathbf{v}^\top \partial_{\mathbf{x}} \mathbf{y}$ 
rather than $\partial_{\mathbf{x}} \mathbf{y}$. 
This argument is named `gradient`
because the vector $\mathbf{v}$ is the gradient arriving
from the rest of a larger computation,
as will become clear when we study
backpropagation in :numref:`sec_backprop`. 
For a more detailed description, see the PyTorch documentation on the
`gradient` argument to
[`Tensor.backward`](https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html). 
:end_tab:

:begin_tab:`tensorflow`
By default, TensorFlow returns the gradient of the sum.
In other words, rather than returning 
the Jacobian $\partial_{\mathbf{x}} \mathbf{y}$,
it returns the gradient of the sum
$\partial_{\mathbf{x}} \sum_i y_i$. 
:end_tab:

```{.python .input #autograd-backward-for-non-scalar-variables}
%%tab mxnet
with autograd.record():
    y = x * x  
y.backward()
x.grad  # Equals the gradient of y = sum(x * x)
```

```{.python .input #autograd-backward-for-non-scalar-variables}
%%tab pytorch
x.grad.zero_()
y = x * x
y.backward(gradient=torch.ones(len(y)))  # Equivalently: y.sum().backward()
x.grad
```

```{.python .input #autograd-backward-for-non-scalar-variables}
%%tab tensorflow
with tf.GradientTape() as t:
    y = x * x
t.gradient(y, x)  # Same as y = tf.reduce_sum(x * x)
```

```{.python .input #autograd-backward-for-non-scalar-variables}
%%tab jax
y = lambda x: x * x
# grad is only defined for scalar output functions
grad(lambda x: y(x).sum())(x)
```

## Controlling the Graph

Sometimes the graph the framework records
is not the graph we want to differentiate.
The next two subsections show how to prune it
by detaching individual intermediate results, and how to
switch recording off altogether.

### Detaching Computation

Sometimes, we wish to move some calculations
outside of the recorded computational graph.
For example, say that we use the input 
to create some auxiliary intermediate terms 
for which we do not want to compute a gradient. 
In this case, we need to *detach* 
the respective computational graph
from the final result. 
For example, suppose that `z = x * y` and `y = x * x`
but we want to focus on the *direct* influence of `x` on `z` 
rather than the influence conveyed via `y`. 
In this case, we can create a new variable `u`
that takes the same value as `y` 
but whose computational history is not connected to the new graph.
Thus `u` has no ancestors in the graph
and gradients do not flow through `u` to `x`.
Now consider `z = x * u`.
Because `u` is treated as a constant equal to $x^2$,
the gradient is $\partial z / \partial x = u = x^2$.
Had we *not* detached, so that `z = x * (x * x)` $= x^3$, we
would instead have obtained $\partial z / \partial x = 3x^2$.

```{.python .input #autograd-detaching-computation-1}
%%tab mxnet
with autograd.record():
    y = x * x
    u = y.detach()
    z = u * x
z.backward()
x.grad == u
```

```{.python .input #autograd-detaching-computation-1}
%%tab pytorch
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u
```

```{.python .input #autograd-detaching-computation-1}
%%tab tensorflow
# Set persistent=True to preserve the compute graph. 
# This lets us run t.gradient more than once
with tf.GradientTape(persistent=True) as t:
    y = x * x
    u = tf.stop_gradient(y)
    z = u * x

x_grad = t.gradient(z, x)
x_grad == u
```

```{.python .input #autograd-detaching-computation-1}
%%tab jax
import jax

y = lambda x: x * x
# jax.lax primitives are Python wrappers around XLA operations
u = jax.lax.stop_gradient(y(x))
z = lambda x: u * x

grad(lambda x: z(x).sum())(x) == u
```

Although this procedure
detaches `y`'s ancestors
from the graph leading to `z`, 
the computational graph leading to `y` 
persists and thus we can calculate
the gradient of `y` with respect to `x`.

```{.python .input #autograd-detaching-computation-2}
%%tab mxnet
y.backward()
x.grad == 2 * x
```

```{.python .input #autograd-detaching-computation-2}
%%tab pytorch
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x
```

```{.python .input #autograd-detaching-computation-2}
%%tab tensorflow
t.gradient(y, x) == 2 * x
```

```{.python .input #autograd-detaching-computation-2}
%%tab jax
grad(lambda x: y(x).sum())(x) == 2 * x
```

### Turning Off Gradient Tracking

Recording operations for a backward pass costs time and memory. When we
only need a value (at prediction time, or while updating parameters by
hand), we can skip the bookkeeping entirely.

:begin_tab:`mxnet`
MXNet only builds a graph inside an `autograd.record()` block, so ordinary
computation already carries no gradient bookkeeping. To suspend tracking
*within* a recording scope, wrap the code in `autograd.pause()`.
:end_tab:

:begin_tab:`pytorch`
Wrap the computation in a `torch.no_grad()` block (or decorate a function
with `@torch.no_grad()`). The result still shares data with `x`, but it is
not attached to the graph, so no gradient can flow through it.
:end_tab:

:begin_tab:`tensorflow`
TensorFlow only records operations executed inside a `tf.GradientTape`
(a *tape* is the recorded list of executed operations), so
any computation outside a tape is already untracked. To pause recording
*within* a tape, use `tape.stop_recording()`.
:end_tab:

:begin_tab:`jax`
JAX never records gradients implicitly: differentiation occurs only after a
transform such as `grad` is applied. Thus, users opt in to differentiation.
:end_tab:

```{.python .input #autograd-turning-off-gradient-tracking-1}
%%tab mxnet
with autograd.record():
    with autograd.pause():
        y = 2 * np.dot(x, x)  # not recorded: no gradient will flow through y
y
```

```{.python .input #autograd-turning-off-gradient-tracking-1}
%%tab pytorch
with torch.no_grad():
    y = 2 * torch.dot(x, x)
y.requires_grad  # False: y is detached from the graph
```

```{.python .input #autograd-turning-off-gradient-tracking-1}
%%tab tensorflow
# Outside any GradientTape, nothing is recorded
y = 2 * tf.tensordot(x, x, axes=1)
y
```

```{.python .input #autograd-turning-off-gradient-tracking-1}
%%tab jax
# No graph is built unless we ask for it via a transform like `grad`
y = 2 * jnp.dot(x, x)
y
```

This untracked mode is the default for inference and evaluation
throughout the rest of the book.

## Beyond the Basics

Automatic differentiation extends beyond the fixed formulas considered so far:
it handles arbitrary control flow, derivatives of derivatives,
and even lets us choose the *direction*
in which the graph is traversed.

### Gradients and Python Control Flow

The preceding examples used a fixed sequence of operations such as
`z = x * x * x`. Programs can instead use auxiliary variables and choose
branches from intermediate results.
One benefit of using automatic differentiation
is that even if building the computational graph of 
a function required Python control flow
(e.g., conditionals, loops, and arbitrary function calls),
we can still calculate the gradient of the resulting variable.
To illustrate this, consider the following code snippet where 
the number of iterations of the `while` loop
and the evaluation of the `if` statement
both depend on the value of the input `a`.

```{.python .input #autograd-gradients-and-python-control-flow-1}
%%tab mxnet
def f(a):
    b = a * 2
    while np.linalg.norm(b) < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input #autograd-gradients-and-python-control-flow-1}
%%tab pytorch
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input #autograd-gradients-and-python-control-flow-1}
%%tab tensorflow
def f(a):
    b = a * 2
    while tf.norm(b) < 1000:
        b = b * 2
    if tf.reduce_sum(b) > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input #autograd-gradients-and-python-control-flow-1}
%%tab jax
def f(a):
    b = a * 2
    while jnp.linalg.norm(b) < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

Below, we call this function, passing in a random value, as input.
Since the input is a random variable, 
we do not know what form 
the computational graph will take.
However, whenever we execute `f(a)` 
on a specific input, we realize 
a specific computational graph
and can subsequently run `backward`.

```{.python .input #autograd-gradients-and-python-control-flow-2}
%%tab mxnet
a = np.random.normal()
a.attach_grad()
with autograd.record():
    d = f(a)
d.backward()
```

```{.python .input #autograd-gradients-and-python-control-flow-2}
%%tab pytorch
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
```

```{.python .input #autograd-gradients-and-python-control-flow-2}
%%tab tensorflow
a = tf.Variable(tf.random.normal(shape=()))
with tf.GradientTape() as t:
    d = f(a)
d_grad = t.gradient(d, a)
d_grad
```

```{.python .input #autograd-gradients-and-python-control-flow-2}
%%tab jax
from jax import random
a = random.normal(random.key(1), ())
d = f(a)
d_grad = grad(f)(a)
```

Even though our function `f` is, for demonstration purposes, a bit contrived,
its dependence on the input is quite simple: 
it is a *linear* function of the scalar `a` 
with piecewise defined scale. 
As such, `f(a) / a` is a constant 
and, moreover, it needs to match 
the gradient of `f(a)` with respect to `a`.

```{.python .input #autograd-gradients-and-python-control-flow-3}
%%tab mxnet
a.grad == d / a
```

```{.python .input #autograd-gradients-and-python-control-flow-3}
%%tab pytorch
a.grad == d / a
```

```{.python .input #autograd-gradients-and-python-control-flow-3}
%%tab tensorflow
d_grad == d / a
```

```{.python .input #autograd-gradients-and-python-control-flow-3}
%%tab jax
d_grad == d / a
```

Dynamic control flow occurs frequently in deep learning.
For instance, when processing text, the computational graph
depends on the length of the input. 
Automatic differentiation records or traces the operations for the realized
computation rather than requiring a separate derivative for every possible
execution path.

### Higher-Order Derivatives

Occasionally we need the derivative of a derivative: the curvature of a
function, or the Hessian--vector products (products of the matrix of
second derivatives, the *Hessian*, with a vector) used by some optimizers.
Autograd can differentiate *through* a gradient computation. Take
$f(x) = x^3$, for which $f'(x) = 3x^2$ and $f''(x) = 6x$.

:begin_tab:`mxnet`
Higher-order gradients in MXNet require explicitly retaining the graph of
the first derivative.
The mathematics is framework-agnostic and developed in
:numref:`sec_mdl-matrix-calculus-autodiff`.
:end_tab:

:begin_tab:`pytorch`
Pass `create_graph=True` so the first gradient is itself a differentiable
function of `x`, then differentiate again.
:end_tab:

:begin_tab:`tensorflow`
Nest two `GradientTape`s: the outer tape differentiates the gradient
computed under the inner tape.
:end_tab:

:begin_tab:`jax`
`grad` returns a function, so we apply it twice.
:end_tab:

```{.python .input #autograd-higher-order-derivatives-1}
%%tab pytorch
x3 = torch.tensor(2.0, requires_grad=True)
dy = torch.autograd.grad(x3 ** 3, x3, create_graph=True)[0]  # 3x^2 = 12
d2y = torch.autograd.grad(dy, x3)[0]                          # 6x  = 12
dy, d2y
```

```{.python .input #autograd-higher-order-derivatives-1}
%%tab tensorflow
x3 = tf.Variable(2.0)
with tf.GradientTape() as outer:
    with tf.GradientTape() as inner:
        y = x3 ** 3
    dy = inner.gradient(y, x3)   # 3x^2 = 12
d2y = outer.gradient(dy, x3)     # 6x  = 12
dy, d2y
```

```{.python .input #autograd-higher-order-derivatives-1}
%%tab jax
f = lambda x: x ** 3
dy = grad(f)(2.0)            # 3x^2 = 12
d2y = grad(grad(f))(2.0)    # 6x  = 12
dy, d2y
```

### Forward versus Reverse Mode

Automatic differentiation can traverse the computational graph in either
direction. *Reverse mode*, the variant we have used so far and the engine
behind backpropagation, sweeps from the output back to the inputs,
yielding the gradient of a single scalar with respect to *all* inputs in
one pass. *Forward mode* sweeps the other way, propagating derivatives
from one input outward to every output.

The choice is about cost, and a counting argument settles it.
For a function with $n$ inputs and $m$ outputs,
filling the full matrix of derivatives takes
one reverse sweep per *output* ($m$ sweeps)
or one forward sweep per *input* ($n$ sweeps),
each sweep costing about as much as one evaluation of the function.
A training loss is a single scalar ($m = 1$)
depending on millions of parameters ($n$ huge),
so reverse mode delivers the entire gradient
for the price of roughly one extra forward pass.
Forward mode wins in the opposite regime
(few inputs, many outputs), is useful for per-input sensitivities, and can
participate in efficient Hessian--vector-product constructions,
as in the Julia package ForwardDiff.jl
:cite:`Revels.Lubin.Papamarkou.2016`.
The exercises explore this trade-off further, and
:numref:`sec_mdl-matrix-calculus-autodiff` derives both modes
and their costs in full.

## Discussion

Automatic differentiation frees practitioners
from deriving gradients by hand,
and it makes it practical to train models
for which pen and paper gradient computations 
would be prohibitively time consuming.
While we use autograd to *optimize* models
(in a statistical sense),
the *optimization* of autograd libraries themselves
(in a computational sense)
is a rich subject that matters to framework designers.
Here, tools from compilers and graph manipulation 
are used to compute results 
quickly and with modest memory. 

The basic workflow is:
(i) attach gradients to those variables with respect to which we desire derivatives;
(ii) record the computation of the target value;
(iii) execute the backpropagation function; and
(iv) access the resulting gradient.


## Exercises

1. Why is the second derivative much more expensive to compute than the first derivative?
1. After running the backpropagation function, run it again. What happens, and
   how does the behavior differ across frameworks?
1. In the control flow example where we calculate the derivative of `d` with respect to `a`, what would happen if we changed the variable `a` to a random vector or a matrix? At this point, the result of the calculation `f(a)` is no longer a scalar. What happens to the result? How do we analyze this?
1. Let $f(x) = \sin(x)$. Plot the graph of $f$ and of its derivative $f'$. Do not exploit the fact that $f'(x) = \cos(x)$ but rather use automatic differentiation to get the result. 
1. Let $f(x) = ((\log x^2) \cdot \sin x) + x^{-1}$. Write out a dependency graph tracing results from $x$ to $f(x)$. 
1. Use the chain rule to compute the derivative $\frac{df}{dx}$ of the aforementioned function, placing each term on the dependency graph that you constructed previously. 
1. Given the graph and the intermediate derivative results, you have a number of options when computing the gradient. Evaluate the result once sweeping from $x$ to $f$ (forward mode) and once from $f$ tracing back to $x$ (reverse mode). 
1. For the graph of exercise 5, count the operations that forward mode and reverse mode each perform, and the intermediate values each must store. How would the comparison change for a function with many inputs, or with many outputs? 

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/34)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/35)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/200)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/17970)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §2.5]{.kicker}

From the chain rule to **backpropagation**<br>automatic differentiation for a complete network.
:::
:::

::: {.slide title="Record the forward pass; replay it in reverse"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
Deriving a million-parameter network's gradient by hand is impractical.
Instead, the framework **records** each operation during the forward
pass, then **replays it in reverse**, systematically applying the chain rule
from the calculus section to obtain the gradient with respect to *every* input
at once.

::: {.d2l-note}
Training steps in this book generally include a forward pass and a backward
pass over this graph.
:::
:::

::: {.col .fig .big}
@fig:autograd-workflow
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[The mechanics]{.dtitle}

[record forward · sweep backward]{.dsub}
:::
:::

::: {.slide title="A function with a known answer: ∇y = 4x"}
[Mechanics]{.kicker}

::: {.cols .vc}
::: {.col}
Differentiate $y = 2\,\mathbf{x}^\top\mathbf{x}$ w.r.t. the vector
$\mathbf{x}$. The analytic answer, $\nabla_\mathbf{x} y = 4\mathbf{x}$,
provides a check for the automatic-differentiation result.

@autograd-a-simple-function-1
:::

::: {.col .fig .big}
@fig:autograd-comp-graph
:::
:::
:::

::: {.slide title="Track x, and the graph builds itself" except="jax"}
[Mechanics]{.kicker}

First tell the framework to **track** `x` (reserve a slot for its
gradient), then run the forward pass; `y` is now the root of a
recorded graph:

@autograd-a-simple-function-2

@autograd-a-simple-function-3
:::

::: {.slide title="No setup: grad transforms the function" only="jax"}
[Mechanics]{.kicker}

JAX is **functional**: there is nothing to attach. You write the
function, and `grad` transforms it into its derivative. The forward
pass is an ordinary call:

@autograd-a-simple-function-3
:::

::: {.slide title="One backward call returns the whole gradient"}
[Mechanics]{.kicker}

One call sweeps the graph in reverse, and the result equals the
promised $4\mathbf{x}$, at every coordinate:

@autograd-a-simple-function-4

@autograd-a-simple-function-5

::: {.d2l-note}
That reverse sweep **is** the calculus section's chain rule, run from
output to input.
:::
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Working with gradients]{.dtitle}

[accumulation · non-scalar outputs · detaching · inference]{.dsub}
:::
:::

::: {.slide title="Gradients accumulate: reset first" only="pytorch"}
[Gradients]{.kicker}

PyTorch **adds** each new gradient into `x.grad` rather than replacing
it (handy for summing losses). So zero it before a fresh computation:

@autograd-a-simple-function-6

::: {.d2l-note .warn}
Forgetting `.zero_()` between iterations causes unintended gradient
accumulation.
:::
:::

::: {.slide title="Each gradient starts fresh" except="pytorch"}
[Gradients]{.kicker}

Recording a new computation overwrites the previous gradient; there is
no buffer to reset:

@autograd-a-simple-function-6
:::

::: {.slide title="Vector outputs: differentiate their sum"}
[Gradients]{.kicker}

Gradients are defined for a **scalar** loss. For a vector `y`, the
engine differentiates the **sum** of its components (a vector–Jacobian
product), exactly what a per-example batch loss needs:

@autograd-backward-for-non-scalar-variables
:::

::: {.slide title="detach freezes a value: ∂z/∂x = u, not 3x²"}
[Gradients]{.kicker}

::: {.cols .vc}
::: {.col}
Sometimes a value should count as a **constant**: gradients must not
flow through it. `detach` (or `stop_gradient`) severs the graph above
it, so $z = u \cdot x$ differentiates to $u$, **not** to $3x^2$:

@autograd-detaching-computation-1
:::

::: {.col .fig}
@fig:autograd-detach
:::
:::
:::

::: {.slide title="Inference skips the bookkeeping"}
[Gradients]{.kicker}

When we only need the value (prediction, evaluation, manual updates),
we turn recording off and avoid its bookkeeping cost. This is the default mode
for inference throughout the book:

@autograd-turning-off-gradient-tracking-1
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[Dynamic graphs]{.dtitle}

[the graph is whatever actually ran]{.dsub}
:::
:::

::: {.slide title="The graph records what actually ran"}
[Dynamic graphs]{.kicker}

::: {.cols .vc}
::: {.col}
In eager execution, autograd records the operations executed by `if` and
`while` statements. This function's loop count and branch both depend on its
input (compiled transformations may impose additional constraints):

@autograd-gradients-and-python-control-flow-1
:::

::: {.col .fig}
@fig:autograd-dynamic
:::
:::
:::

::: {.slide title="Branch or loop, the derivative is f(a)/a"}
[Dynamic graphs]{.kicker}

Each call realizes a concrete graph that `backward` can walk. Whichever
branch ran, `f` scaled its input by some constant, so $f(a) = k\,a$
and the gradient must equal $f(a)/a$. It does:

@autograd-gradients-and-python-control-flow-2

@autograd-gradients-and-python-control-flow-3
:::

::: {.slide title="Reverse mode: the whole gradient for one extra pass"}
[Beyond]{.kicker}

::: {.cols .vc}
::: {.col}
A counting argument settles which way to sweep. With $n$ inputs and $m$
outputs, the full derivative matrix costs $m$ **reverse** sweeps or $n$
**forward** sweeps, each sweep priced at roughly one function
evaluation.

A training loss has $m = 1$ and $n$ in the millions: **one** reverse
sweep delivers every parameter's gradient, for the cost of about one
extra forward pass. Forward mode wins the opposite regime (few inputs,
many outputs) and Hessian–vector products.
:::

::: {.col .fig .big}
@fig:autograd-fwd-vs-rev
:::
:::
:::

::: {.slide title="Differentiate the derivative: f″(2) = 12" except="mxnet"}
[Beyond]{.kicker}

The gradient is itself a function on the graph, so we can differentiate
*it*. For $f(x) = x^3$ at $x = 2$: $f'(2) = 3x^2 = 12$ and
$f''(2) = 6x = 12$: the values coincide at this point, and autograd computes
both:

@autograd-higher-order-derivatives-1
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

::: {.cols}
::: {.col}
- **Record** forward, **sweep** backward: the chain rule, automated
  and verified against $4\mathbf{x}$.
- One reverse sweep = the whole gradient ($m{=}1$, $n$ in millions).
- `detach` / no-grad keep values out of the graph.
:::

::: {.col}
- Mind per-framework gradient handling (PyTorch accumulates).
- The graph is built **at runtime**; control flow needs no special
  handling.
- Higher-order derivatives: differentiate the gradient again.
:::
:::

::: {.d2l-note}
Backpropagation through real networks gets its full treatment in the
backpropagation section; forward vs. reverse mode is derived in the
matrix-calculus-and-automatic-differentiation section.
:::
:::
