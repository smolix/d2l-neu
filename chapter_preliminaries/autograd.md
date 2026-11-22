```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Automatic Differentiation
:label:`sec_autograd`

In :numref:`sec_calculus`, deriving the gradient $\nabla_{\mathbf{w}} L = \mathbf{X}^\top (2\mathbf{r}) = (2, 4)$ of our two-parameter line model by hand was straightforward, but doing so across dozens of layers and millions of parameters would be tedious and error-prone. Modern deep learning frameworks automate this via *automatic differentiation* (*autograd*): as the forward pass evaluates the loss, the framework records the sequence of elementary operations and their inputs in a *computational graph*, and then traverses that graph backward (*backpropagation*) to apply the chain rule automatically at a small constant multiple of the forward cost.

```{.python .input #autograd-automatic-differentiation}
%%tab mxnet
from mxnet import autograd, np as mnp, npx
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
import jax
from jax import numpy as jnp
```

## Computing Gradients

### The Forward Pass

We first set up the line model $L(\mathbf{w}) = \|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2$ at $\mathbf{w} = (1, 0)$ from :numref:`sec_calculus` using framework tensors, writing only the forward pass:

```{.python .input #autograd-a-simple-function-1}
%%tab mxnet
X = mnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])  # [1, input]
y = mnp.array([1.0, 1.0, 0.0])
w = mnp.array([1.0, 0.0])

def loss(w):
    return ((X @ w - y) ** 2).sum()
```

```{.python .input #autograd-a-simple-function-1}
%%tab pytorch
X = torch.tensor([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])  # [1, input]
y = torch.tensor([1.0, 1.0, 0.0])
w = torch.tensor([1.0, 0.0])

def loss(w):
    return ((X @ w - y) ** 2).sum()
```

```{.python .input #autograd-a-simple-function-1}
%%tab tensorflow
X = tf.constant([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])  # [1, input]
y = tf.constant([1.0, 1.0, 0.0])
w = tf.constant([1.0, 0.0])

def loss(w):
    return tf.reduce_sum((tf.linalg.matvec(X, w) - y) ** 2)
```

```{.python .input #autograd-a-simple-function-1}
%%tab jax
X = jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])  # [1, input]
y = jnp.array([1.0, 1.0, 0.0])
w = jnp.array([1.0, 0.0])

def loss(w):
    return ((X @ w - y) ** 2).sum()
```

:begin_tab:`mxnet, pytorch, tensorflow`
Because recording operations and saving intermediate activations consumes memory, frameworks track only operations depending on tensors explicitly marked for gradients.
:end_tab:

:begin_tab:`mxnet`
`w.attach_grad()` marks `w` and allocates `w.grad` (initialized to zeros); operations are recorded inside `with autograd.record():`.
:end_tab:

:begin_tab:`pytorch`
`w.requires_grad_(True)` marks `w` in place so PyTorch records operations involving `w` and stores the resulting gradient in `w.grad`.
:end_tab:

:begin_tab:`tensorflow`
A `tf.GradientTape` records operations executed inside its `with` block; wrapping `w` in `tf.Variable` causes the tape to track it automatically.
:end_tab:

:begin_tab:`jax`
JAX is purely functional: tensors carry no mutable gradient buffers, and `jax.grad(loss)` transforms `loss` into a function that computes its gradient.
:end_tab:

```{.python .input #autograd-a-simple-function-2}
%%tab mxnet
w.attach_grad()
w.grad
```

```{.python .input #autograd-a-simple-function-2}
%%tab pytorch
w.requires_grad_(True)
```

```{.python .input #autograd-a-simple-function-2}
%%tab tensorflow
w = tf.Variable(w)
```

We now evaluate the forward pass `L = loss(w)`, letting the framework record the sequence of operations connecting `w` to `L`:

:begin_tab:`mxnet`
MXNet records the operations inside `with autograd.record():`.
:end_tab:

:begin_tab:`tensorflow`
TensorFlow records the operations inside `with tf.GradientTape() as t:`.
:end_tab:

:begin_tab:`jax`
JAX records nothing yet: `jax.grad` traces `loss` when it is called below.
:end_tab:

```{.python .input #autograd-a-simple-function-3}
%%tab mxnet
with autograd.record():  # record the operations that compute the loss
    L = loss(w)
L
```

```{.python .input #autograd-a-simple-function-3}
%%tab pytorch
L = loss(w)
L
```

```{.python .input #autograd-a-simple-function-3}
%%tab tensorflow
with tf.GradientTape() as t:  # record the operations on the tape t
    L = loss(w)
L
```

```{.python .input #autograd-a-simple-function-3}
%%tab jax
loss(w)
```

:begin_tab:`pytorch`
Notice `grad_fn=<SumBackward0>` on `L`, recording the final operation in the computational graph.
:end_tab:

The recorded trace forms the *computational graph* shown in :numref:`fig_autograd_graph`, breaking the loss into four primitive operations: $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w}$, $\mathbf{r} = \hat{\mathbf{y}} - \mathbf{y}$, $\mathbf{s} = \mathbf{r} \odot \mathbf{r}$, and $L = \sum_i s_i$.

![The computational graph of $L = \|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2$ at $\mathbf{w} = (1, 0)$, with one node for $\mathbf{w}$ and for each value computed from it; the constants $\mathbf{X}$ and $\mathbf{y}$ are not drawn, and $\mathbf{r} \odot \mathbf{r}$ is the elementwise square. Each node shows its value (black) and the gradient of $L$ with respect to it (red). The forward pass (blue) runs from left to right; the backward pass (red) starts from $1$ at $L$ and applies the local rule of each operation, from right to left.](../img/autograd-comp-graph.svg)
:label:`fig_autograd_graph`

### The Backward Pass

Triggering the backward pass propagates sensitivities from $\partial L/\partial L = 1$ at the output back to $\mathbf{w}$ through each node's local derivative rule,

$$\nabla_{\mathbf{s}} L = \mathbf{1} \to \nabla_{\mathbf{r}} L = 2\mathbf{r} \to \nabla_{\hat{\mathbf{y}}} L = 2\mathbf{r} \to \nabla_{\mathbf{w}} L = \mathbf{X}^\top(2\mathbf{r}),$$

reproducing $(2, 4)$:

:begin_tab:`mxnet, pytorch`
`L.backward()` runs the backward pass and stores the gradient in `w.grad`.
:end_tab:

:begin_tab:`tensorflow`
`t.gradient(L, w)` computes the gradient of `L` with respect to `w` from the tape `t`.
:end_tab:

:begin_tab:`jax`
`jax.grad(loss)(w)` evaluates the gradient function at `w`.
:end_tab:

```{.python .input #autograd-a-simple-function-4}
%%tab mxnet
L.backward()
w.grad
```

```{.python .input #autograd-a-simple-function-4}
%%tab pytorch
L.backward()
w.grad
```

```{.python .input #autograd-a-simple-function-4}
%%tab tensorflow
t.gradient(L, w)
```

```{.python .input #autograd-a-simple-function-4}
%%tab jax
jax.grad(loss)(w)
```

Because local backward rules require intermediate activations from the forward pass (such as $\mathbf{r}$ and $\mathbf{X}$), training consumes substantially more memory than inference.

### Gradient Accumulation

What happens when we run a second backward pass at the same parameter vector $\mathbf{w}$?

:begin_tab:`jax`
Because `jax.grad` returns a fresh array on every call and stores no state on `w`, gradients never accumulate implicitly.
:end_tab:

```{.python .input #autograd-a-simple-function-6}
%%tab mxnet
with autograd.record():
    L = loss(w)
L.backward()
w.grad  # overwritten by the new gradient
```

```{.python .input #autograd-a-simple-function-6}
%%tab pytorch
L = loss(w)
L.backward()
w.grad  # the new gradient was added to the old one
```

```{.python .input #autograd-a-simple-function-6}
%%tab tensorflow
with tf.GradientTape() as t:
    L = loss(w)
t.gradient(L, w)  # a new tensor; nothing is stored in w
```

:begin_tab:`mxnet`
MXNet overwrites `w.grad` on each new backward pass by default, returning $(2, 4)$.
:end_tab:

:begin_tab:`pytorch`
In PyTorch, `w.grad` becomes $(4, 8)$: PyTorch *adds* each new gradient into `w.grad` (useful when accumulating gradients over micro-batches or multiple losses). In a standard training loop, always clear the buffer first via `w.grad.zero_()`.
:end_tab:

:begin_tab:`tensorflow`
`t.gradient(L, w)` returns a fresh tensor without mutating `w`, giving $(2, 4)$ again.
:end_tab:

### Vector-Valued Functions

When an operation produces a vector of $m$ outputs $\mathbf{u} \in \mathbb{R}^m$ from $n$ inputs $\mathbf{x} \in \mathbb{R}^n$ with $m \times n$ Jacobian $\mathbf{J}$, a backward pass seeded with a weight vector $\mathbf{v} \in \mathbb{R}^m$ computes the transposed *vector--Jacobian product* $\mathbf{J}^\top \mathbf{v} = \nabla_{\mathbf{x}}(\mathbf{v}^\top \mathbf{u})$ without materializing $\mathbf{J}$. Passing $\mathbf{v} = \mathbf{1}$ on the per-example squared errors $(r_1^2, r_2^2, r_3^2)$ differentiates their sum:

:begin_tab:`mxnet`
`losses.backward(v)` accepts $\mathbf{v}$ (defaulting to $\mathbf{1}$ if omitted).
:end_tab:

:begin_tab:`pytorch`
Calling `backward()` on a non-scalar tensor requires passing $\mathbf{v}$ explicitly via `gradient=torch.ones(3)` (equivalent to `losses.sum().backward()`).
:end_tab:

:begin_tab:`tensorflow`
`t.gradient` accepts $\mathbf{v}$ via `output_gradients` (defaulting to $\mathbf{1}$ if omitted).
:end_tab:

:begin_tab:`jax`
`jax.vjp(losses, w)` returns the forward value and a function mapping $\mathbf{v}$ to $(\mathbf{J}^\top\mathbf{v},)$.
:end_tab:

```{.python .input #autograd-backward-for-non-scalar-variables}
%%tab mxnet
with autograd.record():
    losses = (X @ w - y) ** 2  # one loss per example
losses.backward(mnp.ones(3))  # v = 1: differentiates losses.sum()
w.grad
```

```{.python .input #autograd-backward-for-non-scalar-variables}
%%tab pytorch
w.grad.zero_()
losses = (X @ w - y) ** 2  # one loss per example
losses.backward(gradient=torch.ones(3))  # = losses.sum().backward()
w.grad
```

```{.python .input #autograd-backward-for-non-scalar-variables}
%%tab tensorflow
with tf.GradientTape() as t:
    losses = (tf.linalg.matvec(X, w) - y) ** 2  # per-example losses
t.gradient(losses, w, output_gradients=tf.ones(3))  # v = 1
```

```{.python .input #autograd-backward-for-non-scalar-variables}
%%tab jax
def losses(w):
    return (X @ w - y) ** 2  # one loss per example

_, vjp = jax.vjp(losses, w)  # vjp maps v to J^T v
vjp(jnp.ones(3))[0]  # v = 1
```

## Controlling the Graph

### Detaching Values

*Detaching* a tensor creates a view or copy with the same numerical values but no link to its ancestors in the computational graph, so backpropagation treats it as a constant. Consider $u = x^2$ and $z = u x$ on $\mathbf{x} = (0, 1, 2, 3)$ (:numref:`fig_autograd_detach`): when $u$ is attached, gradient flows along both paths to give $\partial z/\partial x = 3x^2$; when $u$ is detached to $c$, differentiating $z = c x$ treats $c$ as a constant and gives $\partial z/\partial x = c = x^2$ (essential whenever a network computes its own bootstrap target, as in Q-learning, :numref:`sec_qlearning`).

![The product $z = ux$ with $u = x^2$, (a) attached and (b) detached. Blue arrows are operations of the forward pass; red arrows show the backward pass, labeled with the local derivatives it multiplies by. Detaching $u$ creates a constant $c$ with the same value and no connection to $x$.](../img/autograd-detach.svg)
:label:`fig_autograd_detach`

:begin_tab:`mxnet, pytorch`
`u.detach()` returns a tensor sharing `u`'s values but stripped of its history.
:end_tab:

:begin_tab:`tensorflow`
`tf.stop_gradient(u)` passes `u` forward while blocking backward gradients.
:end_tab:

:begin_tab:`jax`
`jax.lax.stop_gradient(u)` blocks gradients inside a differentiated function.
:end_tab:

```{.python .input #autograd-detaching-computation-1}
%%tab mxnet
x = mnp.arange(4.0)
x.attach_grad()
with autograd.record():
    u = x * x
    c = u.detach()  # values of u, no history
    z = c * x
z.backward()
x.grad
```

```{.python .input #autograd-detaching-computation-1}
%%tab pytorch
x = torch.arange(4.0, requires_grad=True)
u = x * x
c = u.detach()  # values of u, no history
z = c * x
z.sum().backward()
x.grad
```

```{.python .input #autograd-detaching-computation-1}
%%tab tensorflow
x = tf.Variable(tf.range(4.0))
with tf.GradientTape() as t:
    u = x * x
    c = tf.stop_gradient(u)  # values of u, no history
    z = c * x
t.gradient(z, x)
```

```{.python .input #autograd-detaching-computation-1}
%%tab jax
x = jnp.arange(4.0)

def z(x):
    u = x * x
    c = jax.lax.stop_gradient(u)  # values of u, no history
    return c * x

jax.grad(lambda x: z(x).sum())(x)
```

By contrast, if we compute `z = u * x` directly without detaching `u`, backpropagation flows through both factors of $z = x^3$ and returns $\partial z / \partial x = 3x^2 = (0, 3, 12, 27)$:

```{.python .input #autograd-detaching-computation-2}
%%tab mxnet
with autograd.record():
    u = x * x
    z = u * x
z.backward()
x.grad
```

```{.python .input #autograd-detaching-computation-2}
%%tab pytorch
x.grad.zero_()
z = u * x
z.sum().backward()
x.grad
```

```{.python .input #autograd-detaching-computation-2}
%%tab tensorflow
with tf.GradientTape() as t:
    u = x * x
    z = u * x
t.gradient(z, x)
```

```{.python .input #autograd-detaching-computation-2}
%%tab jax
jax.grad(lambda x: (x * x * x).sum())(x)
```

### Disabling Gradient Tracking

During validation, inference, or parameter updates $\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla L(\mathbf{w})$, we disable gradient tracking so those operations are not recorded on the loss graph:

:begin_tab:`mxnet`
Code outside `autograd.record()` runs unrecorded; `w[:] = ...` updates `w` in place while keeping `w.grad`.
:end_tab:

:begin_tab:`pytorch`
`with torch.no_grad():` disables tracking (required when modifying a leaf tensor with `requires_grad=True` in place).
:end_tab:

:begin_tab:`tensorflow`
Code outside `tf.GradientTape` runs unrecorded; `w.assign_sub(...)` updates `w` in place.
:end_tab:

:begin_tab:`jax`
Outside `jax.grad`, array arithmetic runs untraced.
:end_tab:

```{.python .input #autograd-turning-off-gradient-tracking-1}
%%tab mxnet
w[:] = w - 0.1 * w.grad  # outside autograd.record(): not recorded
w, loss(w)
```

```{.python .input #autograd-turning-off-gradient-tracking-1}
%%tab pytorch
with torch.no_grad():  # nothing in this block is recorded
    w -= 0.1 * w.grad
    L = loss(w)
w, L
```

```{.python .input #autograd-turning-off-gradient-tracking-1}
%%tab tensorflow
with tf.GradientTape() as t:
    L = loss(w)
w.assign_sub(0.1 * t.gradient(L, w))  # outside the tape
w.numpy(), round(float(loss(w)), 4)
```

```{.python .input #autograd-turning-off-gradient-tracking-1}
%%tab jax
w = w - 0.1 * jax.grad(loss)(w)
w, loss(w)
```

As in :numref:`sec_calculus`, one step with $\eta = 0.1$ moves $\mathbf{w}$ from $(1, 0)$ to $(0.8, -0.4)$ and lowers the loss from $1$ to $0.4$.

## Differentiating Programs

### Control Flow
:label:`subsec_autograd_control_flow`

Because autograd records the sequence of elementary operations that *actually executed* on a given input, it differentiates Python functions containing data-dependent `while` loops and `if`/`else` branches directly:

```{.python .input #autograd-gradients-and-python-control-flow-1}
%%tab mxnet
def f(a):
    b = a * 2
    while mnp.linalg.norm(b) < 1000:
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
    while torch.linalg.vector_norm(b) < 1000:
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

For scalar $a$, `f(a) = k * a` is piecewise linear with slope $k = f(a)/a$ ($k = 2^{12} = 4096$ at $a = 0.3$; $k = 100 \cdot 2^{11} = 204{,}800$ at $a = -0.75$, :numref:`fig_autograd_dynamic`), which autograd recovers exactly:

```{.python .input #autograd-gradients-and-python-control-flow-2}
%%tab mxnet
for value in (0.3, -0.75):
    a = mnp.array(value)
    a.attach_grad()
    with autograd.record():
        fa = f(a)
    fa.backward()
    print(f'a = {value:5.2f}   f(a) / a = {float(fa) / value:6.0f}   '
          f'gradient = {float(a.grad):6.0f}')
```

```{.python .input #autograd-gradients-and-python-control-flow-2}
%%tab pytorch
for value in (0.3, -0.75):
    a = torch.tensor(value, requires_grad=True)
    fa = f(a)
    fa.backward()
    print(f'a = {value:5.2f}   f(a) / a = {fa.item() / value:6.0f}   '
          f'gradient = {a.grad.item():6.0f}')
```

```{.python .input #autograd-gradients-and-python-control-flow-2}
%%tab tensorflow
for value in (0.3, -0.75):
    a = tf.Variable(value)
    with tf.GradientTape() as t:
        fa = f(a)
    print(f'a = {value:5.2f}   f(a) / a = {float(fa) / value:6.0f}   '
          f'gradient = {float(t.gradient(fa, a)):6.0f}')
```

```{.python .input #autograd-gradients-and-python-control-flow-2}
%%tab jax
for value in (0.3, -0.75):
    a = jnp.array(value)
    print(f'a = {value:5.2f}   f(a) / a = {f(a) / a:6.0f}   '
          f'gradient = {jax.grad(f)(a):6.0f}')
```

![The graphs recorded by `f` for two inputs. For $a = 0.3$, `f` doubles $a$ $12$ times and the first branch runs; for $a = -0.75$, it doubles $a$ $11$ times and the second branch runs. Each graph is linear in $a$, with its own slope.](../img/autograd-dynamic.svg)
:label:`fig_autograd_dynamic`

Keep in mind that autograd differentiates the *executed trace* rather than the abstract mathematical function: at a threshold point where the loop count jumps, $f(a)$ is discontinuous, yet autograd returns the slope of whichever branch ran.

## Forward and Reverse Mode
:label:`subsec_autograd_fwd_rev`

For a mapping $\mathbf{f} : \mathbb{R}^n \to \mathbb{R}^m$ with $m \times n$ Jacobian $\mathbf{J}$, the chain rule can be traversed in either direction (:numref:`fig_autograd_fwd_rev`; see :citet:`Wengert.1964,Linnainmaa.1970` and :numref:`sec_mdl-matrix-calculus-autodiff`):

* **Reverse mode (vector--Jacobian product, VJP):** A reverse sweep seeded with $\mathbf{v} \in \mathbb{R}^m$ at the outputs computes $\mathbf{J}^\top \mathbf{v}$, extracting one *row* of $\mathbf{J}$ per sweep ($m$ sweeps for the full Jacobian).
* **Forward mode (Jacobian--vector product, JVP):** A forward sweep seeded with a tangent $\mathbf{d} \in \mathbb{R}^n$ at the inputs propagates derivatives forward alongside values to compute $\mathbf{J}\mathbf{d}$, extracting one *column* of $\mathbf{J}$ per sweep ($n$ sweeps for the full Jacobian, with constant $2\times$ memory overhead).

![Two ways to fill the $m \times n$ Jacobian $\mathbf{J}$ of a function with $n$ inputs and $m$ outputs: (a) forward mode, $n$ sweeps in all; (b) reverse mode, $m$ sweeps in all.](../img/autograd-fwd-vs-rev.svg)
:label:`fig_autograd_fwd_rev`

:begin_tab:`mxnet`
MXNet implements reverse-mode automatic differentiation only.
:end_tab:

:begin_tab:`pytorch`
PyTorch provides `torch.func.vjp`, `torch.func.jvp`, `torch.func.jacrev`, and `torch.func.jacfwd`.
:end_tab:

:begin_tab:`tensorflow`
TensorFlow provides `GradientTape.jacobian` (reverse mode) and `tf.autodiff.ForwardAccumulator` (forward mode).
:end_tab:

:begin_tab:`jax`
JAX provides `jax.vjp`, `jax.jvp`, `jax.jacrev`, and `jax.jacfwd`.
:end_tab:

A scalar loss function has $m = 1$ output and $n$ parameters, so its Jacobian is a single row $\nabla_{\mathbf{w}} L^\top$. Reverse mode (*backpropagation*) computes all $n$ components of the gradient in *one sweep*, whereas forward mode would require $n$ sweeps—explaining why reverse mode is universal for training deep networks.

## Summary

* Automatic differentiation records the forward pass on a computational graph and applies the chain rule backward to compute exact gradients (reproducing $\nabla_{\mathbf{w}} L = (2, 4)$ on our line model).
* Detaching an intermediate tensor severs its graph history so backpropagation treats it as a constant, while disabling gradient tracking keeps parameter updates and inference evaluations off the graph.
* Because the graph records the operations that actually ran, autograd differentiates programs with dynamic loops and branches.
* One forward sweep computes one column of an $m \times n$ Jacobian ($\mathbf{J}\mathbf{d}$), whereas one reverse sweep computes one row ($\mathbf{J}^\top\mathbf{v}$); for a scalar loss ($m = 1$), a single reverse sweep yields the entire gradient.

## Exercises

:begin_tab:`mxnet`
1. [code] **Detached intermediates.** Let `u = x * x` and `c = u.detach()`,
   and consider `z = c * c`, `z = u * c`, and `z = u * u`. For each `z`,
   compute `u`, `c`, and `z` afresh inside an `autograd.record()` scope and
   call `z.backward()`. Before running any code, predict the outcome of each
   call, including the resulting `x.grad`. Check your predictions, and
   explain the ratio between the gradients for `z = u * c` and `z = u * u`.

2. [code] **Manual parameter updates.** Perform the gradient step of this
   section as `w = w - 0.1 * w.grad` instead of `w[:] = w - 0.1 * w.grad`,
   and inspect `w.grad`. Explain what the rebinding has done to the buffer
   allocated by `attach_grad`, predict what a second recorded forward pass
   followed by `backward` does, and check.
:end_tab:

:begin_tab:`pytorch`
1. [code] **Detached intermediates.** Let `u = x * x` and `c = u.detach()`,
   and consider `z = c * c`, `z = u * c`, and `z = u * u`. For each `z`,
   recompute `u` and `c`, reset `x.grad`, and call `backward` on the sum of
   the entries of `z`. Before running any code, predict the outcome of each
   call, including the resulting `x.grad`. Check your predictions, and
   explain the ratio between the gradients for `z = u * c` and `z = u * u`.

2. [code] **Manual parameter updates.** Perform the gradient step of this
   section as `w = w - 0.1 * w.grad`, outside `torch.no_grad()`, and inspect
   `w.requires_grad`, `w.grad_fn`, `w.is_leaf`, and `w.grad`. Explain what
   the update has done to the graph, predict what a second forward and
   backward pass leaves in `w.grad`, and check.
:end_tab:

:begin_tab:`tensorflow`
1. [code] **Detached intermediates.** Let `u = x * x` and
   `c = tf.stop_gradient(u)` inside a persistent `tf.GradientTape` `t`.
   Before running any code, predict `t.gradient(z, x)` for each of
   `z = c * c`, `z = u * c`, and `z = u * u`, all computed inside `t`. Check
   your predictions, and explain the ratio between the results for
   `z = u * c` and `z = u * u`.

2. [code] **Manual parameter updates.** Perform the gradient step of this
   section as `w = w - 0.1 * g`, where `g` is the gradient returned by the
   tape, instead of with `assign_sub`, and inspect `type(w)`. Predict what
   `gradient` returns for a second forward pass under a new tape, check, and
   explain why a training loop updates variables with `assign_sub`.
:end_tab:

:begin_tab:`jax`
1. [code] **Detached intermediates.** Write three functions of `x` that
   compute `u = x * x` and `c = jax.lax.stop_gradient(u)` and return
   `c * c`, `u * c`, and `u * u`, respectively. Before running any code,
   predict the gradient of the sum of the entries of each result with
   respect to `x`. Check your predictions with `jax.grad`, and explain the
   ratio between the results for `u * c` and `u * u`.

2. [code] **Manual parameter updates.** Let `w = jnp.arange(4.0)` and
   `g = lambda x: (w * x).sum()`, which reads `w` from the enclosing scope.
   Predict `jax.grad(g)(x)`, check, and state with respect to which array
   `jax.grad` differentiates. Then rewrite `g` so that `jax.grad` returns
   the gradient of `(w * x).sum()` with respect to `w`, use it for one
   update of `w` with learning rate 0.1, and explain what this implies for
   how a JAX training loop must pass the parameters to the loss function.
:end_tab:

3. [code] **From scalar to vector input.** Replace the scalar input `a` of
   the function `f` defined in :numref:`subsec_autograd_control_flow` by a
   random vector of length 5, and differentiate the sum of the entries of
   `fa = f(a)` with respect to `a`.
    1. Express `f(a)` in terms of `a`, and give the Jacobian of `f` at `a`
       and the gradient of the sum of the entries of `fa`.
    1. Compute this gradient with automatic differentiation and check
       whether it still equals `fa / a` elementwise.
    1. Describe the set of inputs `a` at which `f` jumps. Which part of
       this set has no counterpart for a scalar input?

1. [code] **Programs versus functions.** Automatic differentiation
   differentiates the operations that a program executes, not the function
   that the program computes.
    1. The functions $|x|$ and $\sqrt{x^2}$ are equal. Compute the
       derivative of each at $x = 0$ by automatic differentiation, using the
       framework's absolute value and square root. Explain both results from
       the local derivatives of the operations involved. Neither function
       has a derivative at $0$; which of the two results does more harm in
       training, and why?
    1. Write a program that returns $x^2$ for every input $x$ but whose
       automatic derivative at $x = 0$ is $1$ instead of $0$. Explain why
       automatic differentiation cannot detect the difference.

1. **Cost of second derivatives.** ● The counting argument in
   :numref:`subsec_autograd_fwd_rev` puts the cost of the gradient of a
   scalar loss with $n$ parameters at one reverse sweep. Use the same argument to
   estimate, in units of function evaluations, the cost of the full Hessian
   (the $n \times n$ matrix of second derivatives) and the cost of a single
   Hessian–vector product. Explain the difference between the two costs.

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
[Dive into Deep Learning · §1.5]{.kicker}

Automatic Differentiation: Computational Graphs, Reverse-Mode AD, and Dynamic Programs<br>**forward & reverse mode · computational graphs · VJPs · detaching & memory control · dynamic control flow**
:::
:::

::: {.slide title="Three Ways to Differentiate Code: Why Deep Learning Needs Autograd"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
- **Symbolic algebra:** Exponential expression swell; fails on Python loops and branches.
- **Numerical finite differences:** Costs $O(n)$ forward evaluations ($10^9$ per step!) and loses half of `float32` precision to subtraction cancellation.
- **Reverse-mode autograd:** Applies exact local VJP rules along the execution trace to compute $\nabla_{\mathbf{w}} L \in \mathbb{R}^n$ in **one backward sweep**.
:::

::: {.col .fig .big}
@fig:autograd-three-methods
:::
:::
:::

::: {.slide title="Forward-Mode vs. Reverse-Mode AD: The n ≫ m Asymmetry"}
[Jacobians & Sweeps]{.kicker}

::: {.cols .vc}
::: {.col}
For $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$ with $m \times n$ Jacobian $\mathbf{J}$ ($J_{ji} = \partial f_j / \partial x_i$):

- **Forward mode (JVP, $\mathbf{d} \mapsto \mathbf{J}\mathbf{d}$):** Seeds input tangent $\dot{\mathbf{x}} = \mathbf{e}_i$ and propagates forward, computing **one column of $\mathbf{J}$** per sweep ($n$ sweeps).
- **Reverse mode (VJP, $\mathbf{v} \mapsto \mathbf{J}^\top \mathbf{v}$):** Seeds output adjoint $\bar{\mathbf{y}} = \mathbf{e}_j$ and propagates backward, computing **one row of $\mathbf{J}$** per sweep ($m$ sweeps).
- **Why backprop scales:** A scalar loss has $m = 1 \ll n$, so reverse mode computes all $n$ components of $\nabla_{\mathbf{w}} L$ in **one backward pass**!
:::

::: {.col .fig}
@fig:autograd-fwd-vs-rev
:::
:::
:::

::: {.slide title="Defining the Forward Pass: From Mathematical Loss to Code"}
[Forward Trace]{.kicker}

::: {.cols .vc}
::: {.col}
- In deep learning frameworks, we write **only the forward pass**; no hand-written derivative formula ever appears in model code.
- **Running example (§{sec_calculus}):** Our linear model $L(\mathbf{w}) = \|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2$ with design matrix $\mathbf{X} \in \mathbb{R}^{3 \times 2}$ and targets $\mathbf{y} = [1, 1, 0]^\top$, evaluated at $\mathbf{w} = [1, 0]^\top$.
- Likewise, for a quadratic form $y = 2\mathbf{x}^\top \mathbf{x}$ on $\mathbf{x} = [0, 1, 2, 3]^\top$, each primitive operator (`@`, `-`, `** 2`, `.sum()`) carries its own local backward rule.
:::

::: {.col}
@autograd-a-simple-function-1
:::
:::
:::

::: {.slide title="Opting In to Gradient Tracking and Recording the Forward Trace" except="jax"}
[Graph Construction]{.kicker}

::: {.cols .vc}
::: {.col}
- **Explicit opt-in saves VRAM:** Frameworks only record DAG nodes for parameters explicitly marked for gradient tracking.

@autograd-a-simple-function-2

@autograd-a-simple-function-3
:::

::: {.col .fig}
@fig:autograd-imperative-vs-functional
:::
:::
:::

::: {.slide title="Functional Differentiation with jax.grad: Pure Functions, No Mutable State" only="jax"}
[Graph Construction]{.kicker}

::: {.cols .vc}
::: {.col}
- **Pure functional transform:** JAX arrays carry no `.grad` state; `jax.grad(loss)` transforms `loss(w)` into a pure function returning $\nabla_{\mathbf{w}} L(\mathbf{w}) = [2, 4]^\top$.

@autograd-a-simple-function-3

@autograd-a-simple-function-4
:::

::: {.col .fig}
@fig:autograd-imperative-vs-functional@jax
:::
:::
:::

::: {.slide title="Inside the Computational Graph: Forward Values and Backward Adjoints"}
[How Backpropagation Works]{.kicker}

@fig:autograd-comp-graph

::: {.cols}
::: {.col}
- **Forward pass (blue, left $\to$ right):** Evaluates and caches intermediate activations at $\mathbf{w} = (1, 0)$:
  - $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} = (1,1,1)$ $\to$ $\mathbf{r} = \hat{\mathbf{y}} - \mathbf{y} = (0,0,1)$
  - $\mathbf{s} = \mathbf{r}^{\odot 2} = (0,0,1)$ $\to$ $L = \sum_i s_i = 1$
:::

::: {.col}
- **Backward pass (red, right $\to$ left):** Seeds $\bar{L} = 1$ and applies each node's local VJP rule:
  - $\bar{L}=1$ $\to$ $\bar{\mathbf{s}}=(1,1,1)$ $\to$ $\bar{\mathbf{r}} = 2\mathbf{r} \odot \bar{\mathbf{s}} = (0,0,2)$
  - $\bar{\hat{\mathbf{y}}} = \bar{\mathbf{r}} = (0,0,2)$ $\to$ $\bar{\mathbf{w}} = \mathbf{X}^\top \bar{\hat{\mathbf{y}}} = (2,4)$
:::
:::
:::

::: {.slide title="Executing the Backward Pass and Verifying Analytical Gradients"}
[Backward Pass]{.kicker}

::: {.cols .vc}
::: {.col}
- Triggering the backward pass traverses the graph in reverse topological order and recovers the exact analytical gradient $\nabla_{\mathbf{w}} L = \mathbf{X}^\top(2\mathbf{r}) = [2, 4]^\top$.
- **Second check on $y = 2\mathbf{x}^\top \mathbf{x}$ at $\mathbf{x} = [0, 1, 2, 3]^\top$:**
  - **Forward trace:** $\mathbf{u} = \mathbf{x} \odot \mathbf{x} = [0, 1, 4, 9]^\top$ $\to$ $s = \sum_i u_i = 14$ $\to$ $y = 2s = 28$.
  - **Reverse adjoints:** $\bar{y} = 1$ $\to$ $\bar{s} = 2$ $\to$ $\bar{\mathbf{u}} = [2, 2, 2, 2]^\top$ $\to$ $\bar{\mathbf{x}} = \bar{\mathbf{u}} \odot 2\mathbf{x} = 4\mathbf{x} = [0, 4, 8, 12]^\top$.
:::

::: {.col}
@autograd-a-simple-function-4
:::
:::
:::

::: {.slide title="Gradient Accumulation: Why PyTorch Adds into .grad by Default" only="pytorch"}
[Stateful Buffers]{.kicker}

::: {.cols .vc}
::: {.col}
- **Accumulation by default (`w.grad += g`):** Enables summing gradients across micro-batches when a full batch exceeds GPU memory.
- **Classic bug:** Forgetting `w.grad.zero_()` before the next step adds onto stale gradients (`(2, 4) + (2, 4) = (4, 8)`).

@autograd-a-simple-function-6
:::

::: {.col .fig}
@fig:autograd-grad-accumulation
:::
:::
:::

::: {.slide title="A Second Backward Pass Returns a Fresh Gradient" only="tensorflow,mxnet"}
[Stateless Gradients]{.kicker}

- **No accumulation by default:** Each backward pass yields the gradient of the current loss alone, so a second pass gives $(2, 4)$ again rather than $(4, 8)$.
- **Accumulating over micro-batches** therefore means summing the returned gradients yourself.

@autograd-a-simple-function-6
:::

::: {.slide title="Gradient Accumulation vs. Pure Functional Gradients in JAX" only="jax"}
[Stateless Gradients]{.kicker}

::: {.cols .vc}
::: {.col}
- **No stored gradient state:** `jax.grad(loss)(w)` returns a fresh array every call, so there is no buffer to reset and no stale gradient to corrupt the next step.

@autograd-a-simple-function-4
:::
:::
:::

::: {.slide title="Backward for Non-Scalar Outputs: Vector–Jacobian Products (VJPs)"}
[Vector Outputs]{.kicker}

::: {.cols .vc}
::: {.col}
- Differentiating $\mathbf{u}(\mathbf{w}) \in \mathbb{R}^m$ never forms an $m \times n$ Jacobian; a backward pass seeded with $\mathbf{v} \in \mathbb{R}^m$ computes $\mathbf{J}^\top \mathbf{v} = \nabla_{\mathbf{w}}(\mathbf{v}^\top \mathbf{u})$ ($\mathbf{v} = \mathbf{1}$ sums per-example gradients).

@autograd-backward-for-non-scalar-variables
:::

::: {.col .fig}
@fig:autograd-vjp-seed
:::
:::
:::

::: {.slide title="Detaching Subgraphs with detach and stop_gradient"}
[Graph Surgery]{.kicker}

::: {.cols .vc}
::: {.col}
- For $u = x^2$ and $z = u \cdot x = x^3$:
  - **Attached (a):** Gradients flow through both paths $\implies \frac{\partial z}{\partial x} = 2x^2 + x^2 = 3x^2$.
  - **Detached (b):** `c = u.detach()` (`stop_gradient(u)`) freezes $u$ as constant $x^2 \implies \frac{\partial z}{\partial x}\big|_c = c = x^2$.
- **ML use cases:** DQN target networks, BYOL/SimSiam teacher branches, and GAN discriminator updates.
:::

::: {.col .fig .big}
@fig:autograd-detach
:::
:::
:::

::: {.slide title="Detaching in Code: Comparing Attached (3x²) vs. Detached (x²) Gradients"}
[Detaching in Practice]{.kicker}

::: {.cols}
::: {.col}
- **With `detach()` / `stop_gradient()` (`z = c * x`):**
  At $\mathbf{x} = [0, 1, 2, 3]^\top$, $c = \mathbf{x}^2 = [0, 1, 4, 9]^\top$ acts as a constant, yielding $\nabla_{\mathbf{x}} \sum_i z_i = \mathbf{x}^2 = [0, 1, 4, 9]^\top$:

@autograd-detaching-computation-1
:::

::: {.col}
- **Without `detach()` (`z = u * x`):**
  Both paths from $\mathbf{x}$ to $\mathbf{z}$ carry gradients, yielding the full cubic derivative $\nabla_{\mathbf{x}} \sum_i z_i = 3\mathbf{x}^2 = [0, 3, 12, 27]^\top$:

@autograd-detaching-computation-2
:::
:::
:::

::: {.slide title="Activation Memory and Disabling Gradient Tracking"}
[Memory & Inference]{.kicker}

::: {.cols .vc}
::: {.col}
- **Training caches $O(L)$ activations:** Backward VJPs need forward intermediates ($\mathbf{r}$ for $2\mathbf{r}$, $\mathbf{X}$ for $\mathbf{X}^\top$), pinning every layer's activations in HBM until the backward pass.
- **Inference with tracking off:** Frees each layer buffer immediately ($O(1)$ depth memory) and keeps SGD steps off the tape.

@autograd-turning-off-gradient-tracking-1
:::

::: {.col .fig}
@fig:autograd-activation-memory
:::
:::
:::

::: {.slide title="Dynamic Control Flow: Differentiating Through Python Loops and Branches"}
[Differentiable Programs]{.kicker}

::: {.cols .vc}
::: {.col}
- **Define-by-run tracing:** Every forward evaluation records the exact sequence of primitive tensor operations that *actually ran* on that concrete input.
- Consider `f(a)`:
  1. Repeatedly doubles `b` inside a data-dependent `while` loop until $\|\mathbf{b}\|_2 \ge 1000$ ($k$ doublings in total).
  2. Branches on `b.sum() > 0` to return either `b` or `100 * b`.
- For any scalar input $a \neq 0$, the executed trace is a chain of multiplications $f(a) = m \cdot 2^k \cdot a$ with $m \in \{1, 100\}$.
:::

::: {.col}
@autograd-gradients-and-python-control-flow-1
:::
:::
:::

::: {.slide title="Each Execution Trace Builds Its Own Graph and Recovers Its Exact Slope"}
[Trace vs. Function]{.kicker}

::: {.cols .vc}
::: {.col}
- Since $f(a) = k \cdot a$ is piecewise linear along each execution path, its derivative equals the exact slope $k = f(a) / a$:
  - $a = 0.3 > 0$: $12$ doublings (`if` branch) $\implies k = 2^{12} = 4096$.
  - $a = -0.75 < 0$: $11$ doublings (`else` branch) $\implies k = 100 \cdot 2^{11} = 204{,}800$.
- **Caveat:** Autograd differentiates the *executed trace*—at jump thresholds where loop iterations change, it returns the slope of the branch that ran.
:::

::: {.col .fig .big}
@fig:autograd-dynamic

@autograd-gradients-and-python-control-flow-2
:::
:::
:::

::: {.slide title="Concept Check: Activation Memory, Detach, and Accumulation Bugs" only="pytorch"}
[Check Your Understanding]{.kicker}

Predict the behavior and gradient outcome in each scenario:

1. **Missing reset:** You evaluate 3 micro-batches and call `L.backward()` on each without calling `w.grad.zero_()`. What does `w.grad` hold?
2. **Detached branches:** Let `u = x * x` and `c = u.detach()` (or `stop_gradient(u)`). What is $\nabla_{\mathbf{x}} \sum_i z_i$ for (a) `z = u * u`, (b) `z = u * c`, and (c) `z = c * c`?
3. **Forward vs. reverse sweeps:** A physics simulator maps $n = 3$ wing-shape parameters to $m = 10^6$ mesh pressures. Which AD mode computes the full $10^6 \times 3$ Jacobian faster, and in how many sweeps?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. **The sum of all 3 micro-batch gradients** ($\sum_{b=1}^3 \nabla_{\mathbf{w}} L_b$). Intentional for micro-batch gradient accumulation; a bug if you already stepped the optimizer after micro-batches 1 and 2!
2. (a) `z = u * u = x^4` $\implies$ **$4\mathbf{x}^3$**; (b) `z = u * c = c * x^2` $\implies$ **$2c\mathbf{x} = 2\mathbf{x}^3$** (half of $4\mathbf{x}^3$, since one factor is frozen); (c) `z = c * c` has no active path to `x` $\implies$ **$0$** (or no `grad_fn`).
3. **Forward mode (JVP)** needs only **$n = 3$ sweeps** (one per column), whereas reverse mode needs $m = 10^6$ sweeps (one per row)!
:::
:::

::: {.slide title="Concept Check: Activation Memory, Detach, and Accumulation Bugs" except="pytorch"}
[Check Your Understanding]{.kicker}

Predict the behavior and gradient outcome in each scenario:

1. **Micro-batches:** You run the backward pass on 3 micro-batches in a row. What does the last gradient hold, and how do you accumulate all three?
2. **Detached branches:** Let `u = x * x` and `c = u.detach()` (or `stop_gradient(u)`). What is $\nabla_{\mathbf{x}} \sum_i z_i$ for (a) `z = u * u`, (b) `z = u * c`, and (c) `z = c * c`?
3. **Forward vs. reverse sweeps:** A physics simulator maps $n = 3$ wing-shape parameters to $m = 10^6$ mesh pressures. Which AD mode computes the full $10^6 \times 3$ Jacobian faster, and in how many sweeps?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. **Only the third micro-batch's gradient** $\nabla_{\mathbf{w}} L_3$: nothing is added to earlier results, so you sum the three returned gradients yourself.
2. (a) `z = u * u = x^4` $\implies$ **$4\mathbf{x}^3$**; (b) `z = u * c = c * x^2` $\implies$ **$2c\mathbf{x} = 2\mathbf{x}^3$** (half of $4\mathbf{x}^3$, since one factor is frozen); (c) `z = c * c` has no active path to `x` $\implies$ **$0$** (or no `grad_fn`).
3. **Forward mode (JVP)** needs only **$n = 3$ sweeps** (one per column), whereas reverse mode needs $m = 10^6$ sweeps (one per row)!
:::
:::

::: {.slide title="Summary: Autograd Primitives, Memory Effects, and ML Use Cases" only="pytorch"}
[Key Takeaways]{.kicker}

| Operation / Concept | API | Graph & Memory Behavior | Primary Deep Learning Role |
| :--- | :--- | :--- | :--- |
| **Opt-in / Trace** | `w.requires_grad_(True)` | Caches forward activations in VRAM for backward VJPs | Standard neural network training ($O(1)$ reverse sweep) |
| **Reverse Sweep (VJP)** | `L.backward()` or `u.backward(v)` | Frees activation buffers after computing $\mathbf{J}^\top \mathbf{v}$ | Backpropagation of scalar loss ($m = 1$) or upstream vector $\mathbf{v}$ |
| **Reset / Accumulate** | `w.grad.zero_()` (adds `+=` by default) | Decides whether a new gradient adds to or replaces the last one | Micro-batch gradient accumulation vs. fresh optimizer step |
| **Detach Subgraph** | `u.detach()` | Zero-copy constant view; cuts backward edge | RL target networks, BYOL/SimSiam teacher branch, GAN steps |
| **Inference / No-Grad** | `with torch.no_grad():` | Skips graph construction & activation caching ($3\text{--}10\times$ less VRAM) | Validation loops, production inference, parameter updates |
:::

::: {.slide title="Summary: Autograd Primitives, Memory Effects, and ML Use Cases" only="tensorflow"}
[Key Takeaways]{.kicker}

| Operation / Concept | API | Graph & Memory Behavior | Primary Deep Learning Role |
| :--- | :--- | :--- | :--- |
| **Opt-in / Trace** | `tf.Variable` + `tf.GradientTape()` | Caches forward activations in VRAM for backward VJPs | Standard neural network training ($O(1)$ reverse sweep) |
| **Reverse Sweep (VJP)** | `t.gradient(L, w)` or `t.gradient(u, w, output_gradients=v)` | Frees activation buffers after computing $\mathbf{J}^\top \mathbf{v}$ | Backpropagation of scalar loss ($m = 1$) or upstream vector $\mathbf{v}$ |
| **Reset / Accumulate** | None: returns a fresh tensor per call | Decides whether a new gradient adds to or replaces the last one | Micro-batch gradient accumulation vs. fresh optimizer step |
| **Detach Subgraph** | `tf.stop_gradient(u)` | Zero-copy constant view; cuts backward edge | RL target networks, BYOL/SimSiam teacher branch, GAN steps |
| **Inference / No-Grad** | Outside `GradientTape` | Skips graph construction & activation caching ($3\text{--}10\times$ less VRAM) | Validation loops, production inference, parameter updates |
:::

::: {.slide title="Summary: Autograd Primitives, Memory Effects, and ML Use Cases" only="jax"}
[Key Takeaways]{.kicker}

| Operation / Concept | API | Graph & Memory Behavior | Primary Deep Learning Role |
| :--- | :--- | :--- | :--- |
| **Opt-in / Trace** | `jax.grad(f)` | Caches forward activations in VRAM for backward VJPs | Standard neural network training ($O(1)$ reverse sweep) |
| **Reverse Sweep (VJP)** | `jax.grad(loss)(w)` or `jax.vjp(f, w)` | Frees activation buffers after computing $\mathbf{J}^\top \mathbf{v}$ | Backpropagation of scalar loss ($m = 1$) or upstream vector $\mathbf{v}$ |
| **Reset / Accumulate** | None: returns a fresh array per call | Decides whether a new gradient adds to or replaces the last one | Micro-batch gradient accumulation vs. fresh optimizer step |
| **Detach Subgraph** | `jax.lax.stop_gradient(u)` | Zero-copy constant view; cuts backward edge | RL target networks, BYOL/SimSiam teacher branch, GAN steps |
| **Inference / No-Grad** | Outside `jax.grad` | Skips graph construction & activation caching ($3\text{--}10\times$ less VRAM) | Validation loops, production inference, parameter updates |
:::

::: {.slide title="Summary: Autograd Primitives, Memory Effects, and ML Use Cases" only="mxnet"}
[Key Takeaways]{.kicker}

| Operation / Concept | API | Graph & Memory Behavior | Primary Deep Learning Role |
| :--- | :--- | :--- | :--- |
| **Opt-in / Trace** | `w.attach_grad()` + `autograd.record()` | Caches forward activations in VRAM for backward VJPs | Standard neural network training ($O(1)$ reverse sweep) |
| **Reverse Sweep (VJP)** | `L.backward()` or `u.backward(v)` | Frees activation buffers after computing $\mathbf{J}^\top \mathbf{v}$ | Backpropagation of scalar loss ($m = 1$) or upstream vector $\mathbf{v}$ |
| **Reset / Accumulate** | None: overwrites `w.grad` by default | Decides whether a new gradient adds to or replaces the last one | Micro-batch gradient accumulation vs. fresh optimizer step |
| **Detach Subgraph** | `u.detach()` | Zero-copy constant view; cuts backward edge | RL target networks, BYOL/SimSiam teacher branch, GAN steps |
| **Inference / No-Grad** | Outside `autograd.record()` | Skips graph construction & activation caching ($3\text{--}10\times$ less VRAM) | Validation loops, production inference, parameter updates |
:::
