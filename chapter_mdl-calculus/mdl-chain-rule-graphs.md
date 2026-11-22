# The Chain Rule on Computational Graphs
:label:`sec_mdl-chain-rule-graphs`

Every training step in this book begins with a single call, `backward()` or
`grad`, that returns the slope of the loss with respect to every weight in the
network. In :numref:`sec_mdl-multivariable_calculus` we saw what those slopes
mean: together they form the gradient, the vector that predicts how the loss
changes under any small joint move of the parameters. What we have not yet
explained is how a program computes millions of these slopes in roughly the
time it takes to evaluate the network a few times. The naive approach, nudging
one weight at a time and rerunning the network for each perturbation, requires
a separate pass per weight and quickly becomes infeasible.

The chain rule does vastly better, and the reason is easy to see before writing
down any algebra. A neural network is a deep composition of simple functions,
so the derivatives with respect to its many weights share almost all of their
work: every weight's influence has to travel through the later layers to reach
the loss. A single nested formula hides this shared structure, whereas a
*computational graph* lays it bare: one node for each intermediate value and one
directed edge for each direct dependence. Viewed on a graph, the chain rule
becomes a sum over paths, and exploiting the shared work amounts to choosing
how to group that sum. Grouping the paths from the output backward yields the
derivative with respect to every input in a single pass. That grouping is
backpropagation.

To make every step concrete and easy to verify by hand, we work on small graphs
of scalar variables with integer values throughout. The entire development
rests on one fact from :numref:`sec_mdl-multivariable_calculus`: near any point
where a function is differentiable, its change is approximated by the dot
product of the step with its gradient, the linear model
:eqref:`eq_mdl-nabla_use`. Extending the idea to layers that map vectors to
vectors, and measuring its computational cost on real networks, comes in the
two sections that follow.

```{.python .input #mdl-chain-rule-graphs-the-chain-rule-on-computational-graphs}
#@tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #mdl-chain-rule-graphs-the-chain-rule-on-computational-graphs}
#@tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #mdl-chain-rule-graphs-the-chain-rule-on-computational-graphs}
#@tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #mdl-chain-rule-graphs-the-chain-rule-on-computational-graphs}
#@tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## The Chain Rule as a Sum over Paths
:label:`subsec_mdl-chain-rule-paths`

### A Graph of Scalars

Let us start with a concrete example small enough to trace completely by hand.
Four scalar inputs $w, x, y, z$ feed through two layers of intermediate
variables to produce a single scalar output:

$$
\begin{aligned}
f(u, v) & = (u+v)^{2}, \\
u(a, b) & = (a+b)^{2}, \qquad v(a, b) = (a-b)^{2}, \\
a(w, x, y, z) & = (w+x+y+z)^{2}, \qquad b(w, x, y, z) = (w+x-y-z)^2.
\end{aligned}
$$
:eqlabel:`eq_mdl-multi_func_def`

These equations define a directed acyclic graph (:numref:`fig_mdl-chain-1`),
where each node holds an intermediate value and each arrow indicates a direct
functional dependence. There is nothing special about the squaring operations;
we choose them simply because they keep all intermediate values and derivatives
exact integers. Throughout this section, $f$ denotes the scalar output of the
graph under study.

![The dependency graph of :eqref:`eq_mdl-multi_func_def`: each node represents a variable and each directed edge represents a direct functional dependence.](../img/mdl-cal-chain-net1.svg)
:label:`fig_mdl-chain-1`

### Two Paths through One Step

Before tackling the whole graph, consider just its final composite step,
$f(u(a,b), v(a,b))$, and ask how $f$ responds when we nudge $a$ by a small
amount $\epsilon$. Both intermediate nodes feel the perturbation according to
their partial derivatives: $u$ shifts to
$u + \epsilon\frac{\partial u}{\partial a}$, and $v$ shifts to
$v + \epsilon\frac{\partial v}{\partial a}$. Substituting both shifts into the
first-order linear approximation :eqref:`eq_mdl-nabla_use` of $f$ yields

$$
f\!\left(u + \epsilon\tfrac{\partial u}{\partial a},\, v + \epsilon\tfrac{\partial v}{\partial a}\right)
\approx f(u, v) + \epsilon\left[\frac{\partial f}{\partial u}\frac{\partial u}{\partial a} + \frac{\partial f}{\partial v}\frac{\partial v}{\partial a}\right].
$$

Reading off the coefficient of $\epsilon$ gives the derivative of the composite
function with respect to $a$, which is the multivariable chain rule:

$$
\frac{\mathrm{d} f}{\mathrm{d} a} = \frac{\partial f}{\partial u}\frac{\partial u}{\partial a} + \frac{\partial f}{\partial v}\frac{\partial v}{\partial a}.
$$

Notice the two different symbols for differentiation; we will keep them strictly
separate from here on. The curly $\partial$ denotes the derivative of a single
node's defining function with respect to one of its direct arguments, holding
its other arguments fixed. The upright $\mathrm{d}$ denotes the *total*
derivative $\mathrm{d}f/\mathrm{d}a$, which allows every variable downstream of
$a$ to respond. (For an input node, the total derivative coincides with the
ordinary partial derivative of $f$ viewed as a function of the inputs alone.)
Geometrically, there are two directed *paths* by which $a$ influences $f$,
namely $a \to u \to f$ and $a \to v \to f$. Each path contributes the product
of the derivatives along its edges, and the total derivative sums those
contributions across all paths.

This derivation mirrors the one-variable proof of the chain rule
(:numref:`sec_mdl-derivative_table`): we feed the linear change of the inner
functions into the linear model of the outer one. Because two variables move
simultaneously, however, $f$ must be *differentiable* at $(u, v)$, not merely
possess partial derivatives. Recall the counterexample $2xy/(x^2+y^2)$ from
:numref:`subsec_mdl-gradient`: along the diagonal $x = y = t$, both partial
derivatives vanish at the origin, so the chain rule would predict a derivative
of $0$ at $t = 0$, yet the function jumps discontinuously from $0$ to $1$
there.

### Edge Derivatives and Total Derivatives

The same reasoning extends by induction to any directed acyclic graph. Define
the *edge derivative* of a node with respect to one of its parents as the
partial derivative of that node's function with respect to that parent, with
all other parents held fixed. Then the total derivative of an output $f$ with
respect to an input $x$ is

$$
\frac{\mathrm{d} f}{\mathrm{d} x} = \sum_{\text{directed paths } x \to \cdots \to f} \;\; \prod_{\text{edges on the path}} (\text{edge derivative}).
$$
:eqlabel:`eq_mdl-chain-rule-paths`

In words, each directed path from $x$ to $f$ multiplies the local rates of
change along its edges, and the total derivative adds those path products
together.

A graph with a skip connection makes it clear why edge derivatives and total
derivatives must never be confused. Consider the five-node system

$$
\begin{aligned}
p(x, y) & = x + y, \qquad q(x, y) = x - y, \\
r(p) & = p^2, \qquad s(q) = q^2, \\
f(r, p, s) & = r + p + s.
\end{aligned}
$$
:eqlabel:`eq_mdl-multi_func_def_2`

Here $f$ depends on $r$ and $s$, and also directly on $p$ through a skip edge
(:numref:`fig_mdl-chain-2`). Consequently, the input $y$ reaches $f$ along
three distinct paths: $y \to p \to r \to f$, the shortcut $y \to p \to f$, and
$y \to q \to s \to f$. The local edge derivatives are
$\partial f/\partial r = \partial f/\partial p = \partial f/\partial s = 1$,
$\partial r/\partial p = 2p$, $\partial s/\partial q = 2q$,
$\partial p/\partial y = 1$, and $\partial q/\partial y = -1$. Summing the
three path products gives

$$
\begin{aligned}
\frac{\mathrm{d} f}{\mathrm{d} y} &= \underbrace{1 \cdot 2p \cdot 1}_{y \to p \to r \to f} + \underbrace{1 \cdot 1}_{y \to p \to f} + \underbrace{1 \cdot 2q \cdot (-1)}_{y \to q \to s \to f} \\
&= 2p + 1 - 2q = 1 + 4y.
\end{aligned}
$$

We can check this directly by substituting everything into $f$:
$f = (x+y)^2 + (x+y) + (x-y)^2$, whose derivative with respect to $y$ is indeed
$1 + 4y$. At $(x, y) = (1, 2)$, we have $p = 3$ and $q = -1$, so the three
paths contribute $6 + 1 + 2 = 9 = 1 + 4 \cdot 2$. Notice the sharp distinction
at node $p$: the *edge* derivative $\partial f/\partial p$ is just $1$ (holding
$r$ fixed), whereas the *total* derivative $\mathrm{d}f/\mathrm{d}p$ accounts
for both the skip edge and the indirect route through $r$, giving
$2p + 1 = 7$. Confusing these two quantities is the single most common mistake
when applying the multivariable chain rule.

![The dependency graph of :eqref:`eq_mdl-multi_func_def_2`: the input $y$ reaches $f$ by three paths, namely $y\to p\to r\to f$, the skip edge $y\to p\to f$, and $y\to q\to s\to f$.](../img/mdl-cal-chain-net2.svg)
:label:`fig_mdl-chain-2`

This path-sum viewpoint also explains how gradients travel through deep
architectures. Mechanisms that open or close paths, such as the gates in an LSTM
(:numref:`sec_lstm`) or the skip connections in a residual network
(:numref:`sec_resnet`), shape learning directly by controlling which paths carry
gradient signal back to earlier layers.

## Backpropagation
:label:`subsec_mdl-backpropagation`

As a mathematical identity, the sum over paths :eqref:`eq_mdl-chain-rule-paths`
is completely general: it gives every derivative of every computational graph.
As a literal algorithm, however, it is hopeless, because the number of paths
can grow exponentially with depth. Even a simple chain of $m$ stages with two
parallel nodes per stage already contains $2^m$ paths from start to finish.
Fortunately, those paths overlap heavily, and any sub-product shared by many
paths needs to be computed only once. The order in which we factor out those
shared sub-products determines the computational cost.

### Forward Mode

Let us compare two ways of factoring the sum on our four-input function
:eqref:`eq_mdl-multi_func_def` at the point $(w, x, y, z) = (1, 1, -2, 1)$. A
*forward pass* evaluates the nodes in topological order: $a = 1^2 = 1$,
$b = 3^2 = 9$, $u = 10^2 = 100$, $v = (-8)^2 = 64$, and $f = 164^2 = 26896$.
From these node values we read off all the local edge derivatives:

$$
\begin{aligned}
\frac{\partial f}{\partial u} = \frac{\partial f}{\partial v} &= 2(u+v) = 328, \\
\frac{\partial u}{\partial a} = \frac{\partial u}{\partial b} &= 2(a+b) = 20, \\
\frac{\partial v}{\partial a} = 2(a-b) &= -16, \qquad \frac{\partial v}{\partial b} = -2(a-b) = 16, \\
\frac{\partial a}{\partial w} = \frac{\partial a}{\partial x} = \frac{\partial a}{\partial y} = \frac{\partial a}{\partial z} &= 2(w+x+y+z) = 2, \\
\frac{\partial b}{\partial w} = \frac{\partial b}{\partial x} = 2(w+x-y-z) &= 6, \qquad \frac{\partial b}{\partial y} = \frac{\partial b}{\partial z} = -6.
\end{aligned}
$$

Each edge derivative is a local number computed solely from a node and its
immediate parents.

To find $\mathrm{d}f/\mathrm{d}w$, notice that all four paths from $w$ to $f$
pass first through $a$ or $b$ and then through $u$ or $v$. Factoring the paths
from the input $w$ forward through the graph gives

$$
\begin{aligned}
\frac{\mathrm{d} f}{\mathrm{d} w} &= \frac{\partial f}{\partial u}\Bigl(\frac{\partial u}{\partial a}\frac{\partial a}{\partial w} + \frac{\partial u}{\partial b}\frac{\partial b}{\partial w}\Bigr) + \frac{\partial f}{\partial v}\Bigl(\frac{\partial v}{\partial a}\frac{\partial a}{\partial w} + \frac{\partial v}{\partial b}\frac{\partial b}{\partial w}\Bigr) \\
&= 328\,(20\cdot 2 + 20 \cdot 6) + 328\,(-16 \cdot 2 + 16 \cdot 6) \\
&= 328 \cdot 224 = 73472.
\end{aligned}
$$

The two parenthesized sums are simply $\mathrm{d}u/\mathrm{d}w = 160$ and
$\mathrm{d}v/\mathrm{d}w = 64$, the total derivatives of the intermediate nodes
with respect to $w$. Propagating the sensitivity to one input forward through
the graph in this way is called *forward mode*. We can verify this calculation
in plain Python:

```{.python .input #multivariable-calculus-the-backpropagation-algorithm-1}
w, x, y, z = 1, 1, -2, 1
# Forward pass: node values
a, b = (w + x + y + z)**2, (w + x - y - z)**2
u, v = (a + b)**2, (a - b)**2
f = (u + v)**2
print('a, b, u, v, f =', a, b, u, v, f)

# Edge derivatives
df_du, df_dv = 2*(u + v), 2*(u + v)
du_da, du_db, dv_da, dv_db = 2*(a + b), 2*(a + b), 2*(a - b), -2*(a - b)
da_dw, db_dw = 2*(w + x + y + z), 2*(w + x - y - z)

# Forward mode: push the dependence on w through the graph
du_dw, dv_dw = du_da*da_dw + du_db*db_dw, dv_da*da_dw + dv_db*db_dw
df_dw = df_du*du_dw + df_dv*dv_dw
print('df/dw =', df_dw)
```

Forward mode returns $\mathrm{d}f/\mathrm{d}w = 73472$, but none of its
intermediate derivatives can be reused for $x$, $y$, or $z$: both
$\mathrm{d}u/\mathrm{d}w$ and $\mathrm{d}v/\mathrm{d}w$ are tied to $w$ alone.
Because training a neural network requires the derivative of a single scalar
loss with respect to every parameter, computing them one input at a time leaves
most of the shared structure unused.

### Adjoints and the Backward Pass

The remedy is to reverse the perspective: keep the output fixed rather than the
input. Forward mode fixed one input and asked how every downstream node
responds to it. Now we fix the single output $f$ and ask how it responds to
every upstream node. Define the *adjoint* $\bar{n}$ of a node $n$ as the total
derivative $\mathrm{d}f/\mathrm{d}n$ of the output with respect to $n$:
the sum of the edge-derivative products over all directed paths from $n$ to
$f$. Every such path must leave $n$ through one of its immediate children $c$,
and the remaining paths from $c$ to $f$ sum to $\bar{c}$. Grouping the path sum
:eqref:`eq_mdl-chain-rule-paths` by this first edge turns it into a simple
recursion:

$$
\bar{f} = 1, \qquad
\bar{n} = \sum_{\text{children } c \text{ of } n} \bar{c}\,\frac{\partial c}{\partial n}.
$$
:eqlabel:`eq_mdl-adjoint`

Only local edge derivatives and the adjoints of downstream nodes appear on the
right-hand side. Starting from $\bar f = 1$ and visiting the nodes in reverse
topological order, every child's adjoint is already known by the time we need
it. On the skip-edge graph :eqref:`eq_mdl-multi_func_def_2`, for instance, the
recursion gives $\bar r = 1$ and
$\bar p = \bar r \cdot 2p + \bar f \cdot 1 = 7$, matching the total derivative
we found earlier. Applying the same sweep to our four-input graph gives:

* $\bar{u} = \bar{v} = 328$;
* $\bar{a} = \bar{u}\cdot 20 + \bar{v}\cdot(-16) = 1312$ and
  $\bar{b} = \bar{u}\cdot 20 + \bar{v}\cdot 16 = 11808$;
* $\bar{w} = \bar{x} = \bar{a}\cdot 2 + \bar{b}\cdot 6 = 73472$ and
  $\bar{y} = \bar{z} = \bar{a}\cdot 2 - \bar{b}\cdot 6 = -68224$.

A single *backward pass* thus produces the partial derivatives with respect to
all four inputs at once (:numref:`fig_mdl-cal-backward-sweep`). Let us run this
backward sweep in Python, reusing the edge derivatives from the previous cell.

![(a) The forward pass records every node value and local edge derivative. (b) The backward pass accumulates adjoints starting from $\bar f = 1$, giving $\bar a = 328\cdot 20 + 328\cdot(-16) = 1312$ and $\bar w = 1312\cdot 2 + 11808\cdot 6 = 73472$.](../img/mdl-cal-backward-sweep.svg)
:label:`fig_mdl-cal-backward-sweep`

```{.python .input #multivariable-calculus-the-backpropagation-algorithm-2}
da_dy, db_dy = 2*(w + x + y + z), -2*(w + x - y - z)  # also for x, z
# Backward pass: adjoints from the output toward the inputs
u_bar, v_bar = df_du, df_dv
a_bar = u_bar*du_da + v_bar*dv_da
b_bar = u_bar*du_db + v_bar*dv_db
print(f'adjoints: u {u_bar}, v {v_bar}, a {a_bar}, b {b_bar}')
w_bar = a_bar*da_dw + b_bar*db_dw  # x enters a, b like w
y_bar = a_bar*da_dy + b_bar*db_dy  # z enters a, b like y
print('gradient:', {'w': w_bar, 'x': w_bar, 'y': y_bar, 'z': y_bar})
```

The derivatives with respect to $w$ and $x$ match, as do those with respect to
$y$ and $z$, because $f$ depends on its inputs only through the combinations
$w + x$ and $y + z$.

A quick operation count makes the advantage concrete. Forward mode needs six
multiplications per input (two each for $\mathrm{d}u/\mathrm{d}w$,
$\mathrm{d}v/\mathrm{d}w$, and $\mathrm{d}f/\mathrm{d}w$), or $24$ in total for
all four inputs. The backward pass needs four multiplications to compute
$\bar a$ and $\bar b$ and then two per input, or $12$ in total, because it
traverses the shared upper part of the graph only once. A factor of two on a
toy graph is modest, but the gap widens rapidly with the number of inputs:
every additional input costs forward mode an entire pass through the graph,
whereas the backward pass needs only two more multiplications.

### Backpropagation in a Framework

Computing derivatives by propagating adjoints from the output back to the
inputs is *backpropagation* :cite:`Rumelhart.Hinton.Williams.ea.1988`, which
:numref:`sec_backprop` applies to multilayer perceptrons. Every modern
automatic differentiation engine executes the same two passes: it records
intermediate values (or the operations needed to form edge derivatives) during
the forward pass, and then accumulates adjoints during the backward pass. Let
us check our hand computation against the framework's automatic differentiation
on the same four inputs.

```{.python .input #multivariable-calculus-the-backpropagation-algorithm-3}
#@tab mxnet
wxyz = mnp.array([1., 1., -2., 1.])
wxyz.attach_grad()
with autograd.record():
    w, x, y, z = wxyz[0], wxyz[1], wxyz[2], wxyz[3]
    a, b = (w + x + y + z)**2, (w + x - y - z)**2
    u, v = (a + b)**2, (a - b)**2
    f = (u + v)**2
f.backward()  # one backward pass
print(wxyz.grad)
```

```{.python .input #multivariable-calculus-the-backpropagation-algorithm-3}
#@tab pytorch
wxyz = torch.tensor([1., 1., -2., 1.], requires_grad=True)
w, x, y, z = wxyz
a, b = (w + x + y + z)**2, (w + x - y - z)**2
u, v = (a + b)**2, (a - b)**2
f = (u + v)**2
f.backward()  # one backward pass
print(wxyz.grad)
```

```{.python .input #multivariable-calculus-the-backpropagation-algorithm-3}
#@tab tensorflow
wxyz = tf.Variable([1., 1., -2., 1.])
with tf.GradientTape() as t:
    w, x, y, z = tf.unstack(wxyz)
    a, b = (w + x + y + z)**2, (w + x - y - z)**2
    u, v = (a + b)**2, (a - b)**2
    f = (u + v)**2
print(t.gradient(f, wxyz).numpy())  # one backward pass
```

```{.python .input #multivariable-calculus-the-backpropagation-algorithm-3}
#@tab jax
def f_graph(wxyz):
    w, x, y, z = wxyz
    a, b = (w + x + y + z)**2, (w + x - y - z)**2
    u, v = (a + b)**2, (a - b)**2
    return (u + v)**2

print(jax.grad(f_graph)(jnp.array([1., 1., -2., 1.])))  # one backward pass
```

The framework returns $(73472, 73472, -68224, -68224)$, matching the adjoints
from our hand calculation. The agreement is exact rather than approximate
because every intermediate number is an integer below $2^{24}$, which single
precision represents without rounding error.

So far every node in our graphs has been a scalar, and we could count
multiplications on the back of an envelope. In a real neural network, each
layer maps a vector to a vector, so the edge derivatives become matrices and
the order in which those matrices are multiplied matters far more than the
order of scalar multiplications. We take that step in
:numref:`sec_mdl-matrix-calculus`.

## Summary

On a computational graph, the total derivative of an output with respect to an
input equals the sum, over all directed paths between them, of the products of
the local edge derivatives. The adjoint of a node, the total derivative of the
output with respect to that node, collects every path leaving it. Factoring
those paths by their first outgoing edge yields a recursion that, run backward
from the output, computes every adjoint in a single pass. This reverse
accumulation is backpropagation. Propagating one input forward instead computes
only that input's derivative, which is why differentiating a scalar loss with
respect to many parameters is always done backward.

## Exercises

1. [code] **Backward pass for two tanh units.** A network with one input
   $x$, two hidden units, and one output computes, for a target $t$,

   $$h_i = \tanh(w_i x + b_i) \quad (i = 1, 2), \qquad f = v_1 h_1 + v_2 h_2, \qquad \ell = \tfrac12 (f - t)^2.$$

    1. Draw the dependency graph of $\ell$. In plain Python, write a forward
       pass that stores every intermediate value and a backward pass that
       returns the partial derivatives of $\ell$ with respect to all six
       parameters $w_1, b_1, w_2, b_2, v_1, v_2$. Check them against
       automatic differentiation at $x = 1.5$, $t = 0.5$, and
       $(w_1, b_1, w_2, b_2, v_1, v_2) = (0.5, 0, -1, 0.2, 1, -0.5)$.
       Which intermediate derivative does every one of the six partials
       reuse?
    1. Set $w_1 = 5$ and $x = 2$, keeping the other values. Compute
       $\partial \ell / \partial w_1$ with the plain-Python backward pass
       and explain its size from the product of edge derivatives along its
       path. Compare it with the value that automatic differentiation returns
       in single precision, and explain the difference. What does this imply
       for training a unit whose pre-activation is large?
    1. Set $w_1 = w_2$, $b_1 = b_2$, and $v_1 = v_2$. Show from the backward
       pass that both units receive identical gradients, so that gradient
       descent keeps them identical at every step. What does this imply for
       the initialization of hidden units?

1. **Paths through a residual stack.** ● In a one-dimensional model of the
   residual networks of :numref:`sec_resnet`, scalar blocks compute
   $x_{l+1} = x_l + g_l(x_l)$ for $l = 0, \ldots, m-1$ with differentiable
   functions $g_l$.
    1. Use the chain rule to write $\partial x_m / \partial x_0$ as a product
       of $m$ factors.
    1. Expand the product and match each term to a directed path from $x_0$
       to $x_m$ in the dependency graph. How many paths are there, and how
       many of them pass through exactly $k$ of the functions $g_l$? Compare
       the number of paths with the number of multiplications that a
       backward pass needs to compute $\partial x_m / \partial x_0$.
    1. For a trained network of $m = 54$ residual blocks,
       :citet:`Veit.Wilber.Belongie.2016` found that almost all of the
       gradient flows along the paths through $5$ to $17$ blocks. What
       fraction of all paths are these? Now assume that
       $g_l'(x_l) = c > 0$ for every block. Which value of $c$ makes the
       paths through about $11$ blocks contribute the most to
       $\partial x_m / \partial x_0$, and what share of the derivative do the
       paths through $5$ to $17$ blocks then carry?

1. [code] **Gradient checks in single precision.** ● A gradient check
   compares a backward pass with the central difference
   $\bigl(f(w + h, x, y, z) - f(w - h, x, y, z)\bigr) / (2h)$. Let $f$ be the
   function of :eqref:`eq_mdl-multi_func_def`, whose exact partial derivative
   at $(w, x, y, z) = (1, 1, -2, 1)$ is
   $\partial f / \partial w = 73472$.
    1. Implement $f$ in NumPy and compute the central difference at this point
       for $h = 10^{-1}, 10^{-2}, \ldots, 10^{-12}$, once in `float64` and
       once in `float32`. Plot the relative error against $h$ on log-log axes
       for both precisions, and read off the best $h$ and the smallest
       relative error of each.
    1. The error of the central difference is a truncation part of order
       $h^2$ plus a rounding part of order
       $\varepsilon_{\textrm{mach}} |f| / h$, where the machine epsilon
       $\varepsilon_{\textrm{mach}}$ is about $2.2 \times 10^{-16}$ for
       `float64` and $1.2 \times 10^{-7}$ for `float32`. By which factors
       should the best $h$ and the smallest relative error change from
       `float64` to `float32`? Compare the predicted factors with the plot.
    1. Which tolerance on the relative error can a `float32` gradient check
       use without rejecting a correct backward pass? Why are gradient
       checks run in double precision?
    1. Why does the `float32` estimate become exactly zero for the smallest
       values of $h$?
   :exlabel:`ex_mdl-multivariable-calculus-gradient-check`

[Discussions](https://d2l.discourse.group/t/36877)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §34.3]{.kicker}

The Chain Rule on Computational Graphs<br>**dependency DAGs · edge vs. total derivatives · sum over paths · forward mode · adjoints & backpropagation**
:::
:::

::: {.slide title="Why One Backward Pass Replaces a Million Perturbations"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
- **Perturbation Bottleneck:** Nudging $N = 10^8$ parameters one by one requires $10^8$ separate forward passes.
- **Shared Downstream Sub-Paths:** Every early parameter's effect travels through the exact same upper layers to reach the scalar loss $f$.
- **Backpropagation as Path Factoring:** Grouping the path-sum chain rule from $f$ backward reuses every downstream derivative across all inputs.
:::

::: {.col .fig .big}
@fig:ch34-calc-graph-sharing
:::
:::
:::

::: {.slide title="Representing a Function as a Computation Graph"}
[Directed Acyclic Graphs]{.kicker}

::: {.cols .vc}
::: {.col}
- **Concrete 4-input integer benchmark:** Four inputs $(w, x, y, z)$ pass through Layer 1 nodes $a = (w + x + y + z)^2$ and $b = (w + x - y - z)^2$, then Layer 2 nodes $u = (a + b)^2$ and $v = (a - b)^2$, to produce scalar output:
  $$f(u, v) = (u + v)^2$$
- **Nodes and directed edges:** Each node stores one intermediate variable, and a directed edge $n \to c$ indicates that node $c$'s defining formula directly takes $n$ as an argument.
- **Exact integer arithmetic:** Squaring keeps all node values and derivatives exact integers so we can verify every step without floating-point rounding.
:::

::: {.col .fig}
![](../img/mdl-cal-chain-net1.svg){width=100%}
:::
:::
:::

::: {.slide title="Propagating a Perturbation Across Parallel Branches"}
[Multivariable Chain Rule]{.kicker}

::: {.cols .vc}
::: {.col}
- **1. Perturb Intermediate Children:** Nudging $a \mapsto a + \epsilon$ shifts $u$ by $\Delta u \approx \epsilon\frac{\partial u}{\partial a}$ and $v$ by $\Delta v \approx \epsilon\frac{\partial v}{\partial a}$.
- **2. Substitute into Linear Model of $f(u, v)$:** $\Delta f \approx \frac{\partial f}{\partial u}\Delta u + \frac{\partial f}{\partial v}\Delta v$.
- **3. Sum Parallel Branches:** $\frac{\mathrm{d}f}{\mathrm{d}a} = \frac{\partial f}{\partial u}\frac{\partial u}{\partial a} + \frac{\partial f}{\partial v}\frac{\partial v}{\partial a}$.
:::

::: {.col .fig .big}
@fig:ch34-calc-parallel-branches
:::
:::
:::

::: {.slide title="Edge Derivatives (∂) vs. Total Derivatives (d)"}
[Notation & Semantics]{.kicker}

::: {.cols .vc}
::: {.col}
- **Edge Derivative $\frac{\partial c}{\partial n}$:** Local partial of child $c$'s formula with respect to parent $n$, holding **all other direct parents of $c$ fixed**.
- **Total Derivative (Adjoint) $\bar{n} = \frac{\mathrm{d}f}{\mathrm{d}n}$:** Total rate of change of final output $f$ when node $n$ is nudged and **all downstream nodes respond** along every directed path $n \to \cdots \to f$.
:::

::: {.col .fig .big}
@fig:ch34-calc-edge-vs-total
:::
:::
:::

::: {.slide title="The General Chain Rule as a Sum Over Paths"}
[Path-Sum Formula]{.kicker}

::: {.cols .vc}
::: {.col}
- **Path-Sum Chain Rule:** $\frac{\mathrm{d}f}{\mathrm{d}x} = \sum_{\text{paths } x \to \cdots \to f} \prod_{(u \to v)} \frac{\partial v}{\partial u}$ (multiply along edges of each path, add across parallel paths).
- **Exponential Blow-Up of Naive Enumeration:** An $m$-stage graph with 2 parallel nodes per stage has $2^m$ paths; dynamic programming (AD) factors shared nodes in $O(|\text{Edges}|)$ time.
:::

::: {.col .fig .big}
@fig:ch34-calc-path-sum-rule
:::
:::
:::

::: {.slide title="Skip Connections and Multi-Path Interference"}
[Worked Example]{.kicker}

::: {.cols .vc}
::: {.col}
- **A 5-node DAG with a skip edge:** Let $p = x + y$, $q = x - y$, $r = p^2$, $s = q^2$, and $f(r, p, s) = r + p + s$.
- **Three paths from $y$ to $f$:** Path $y \to p \to r \to f$ contributes $1 \cdot 2p \cdot 1 = 2p$, the skip edge $y \to p \to f$ contributes $1 \cdot 1 = 1$, and path $y \to q \to s \to f$ contributes $1 \cdot 2q \cdot (-1) = -2q$:
  $$\frac{\mathrm{d}f}{\mathrm{d}y} = 2p + 1 - 2q = 1 + 4y$$
- **At $(x, y) = (1, 2)$ ($p = 3, q = -1$):** The three paths contribute $6 + 1 + 2 = 9 = 1 + 4(2)$.
- **Edge vs. total at node $p$:** The direct skip edge has $\frac{\partial f}{\partial p} = 1$ (holding $r$ fixed), whereas the total derivative is $\frac{\mathrm{d}f}{\mathrm{d}p} = 2p + 1 = 7$.
:::

::: {.col .fig}
![](../img/mdl-cal-chain-net2.svg){width=100%}
:::
:::
:::

::: {.slide title="Architectural Lesson: ResNets and Gates Shape Paths"}
[Deep Learning Connection]{.kicker}

::: {.cols .vc}
::: {.col}
- **Residual Stack ($x_{l+1} = x_l + g_l(x_l)$):** Each block branches into an identity skip edge (slope $1$) and a residual branch (slope $g_l'$), yielding $\frac{\mathrm{d}x_m}{\mathrm{d}x_0} = \prod_{l=0}^{m-1}(1 + g_l'(x_l))$.
- **Path Ensemble ($2^m$ Paths):** Expanding the product gives $\binom{m}{k}$ paths passing through $k$ nonlinear blocks and skipping $m - k$ blocks with unit slope (Veit et al., 2016).
:::

::: {.col .fig .big}
@fig:ch34-calc-resnet-paths
:::
:::
:::

::: {.slide title="Forward Pass and Local Edge Derivatives"}
[Hand Trace: Step 1]{.kicker}

::: {.cols .vc}
::: {.col}
- **Forward Values at $(1, 1, -2, 1)$:** Layer 1 gives $a = 1^2 = 1$ and $b = 3^2 = 9$; Layer 2 gives $u = 10^2 = 100$ and $v = (-8)^2 = 64$; output is $f = 164^2 = 26{,}896$.
- **All 14 Edge Slopes Known Immediately:** Local partials ($\partial a/\partial w = 2$, $\partial b/\partial w = 6$, $\partial u/\partial a = 20$, $\partial v/\partial a = -16$, $\partial f/\partial u = 328$) depend only on each node and its immediate parents.
:::

::: {.col .fig .big}
@fig:ch34-calc-forward-edge-table
:::
:::
:::

::: {.slide title="Forward Mode: Factoring Paths from One Input Outward"}
[Forward Accumulation]{.kicker}

::: {.cols .vc}
::: {.col}
- **Push One Input Sensitivity Forward:** Fixing $w$, we compute $\mathrm{d}a/\mathrm{d}w = 2, \mathrm{d}b/\mathrm{d}w = 6 \implies \mathrm{d}u/\mathrm{d}w = 160, \mathrm{d}v/\mathrm{d}w = 64 \implies \mathrm{d}f/\mathrm{d}w = 328(160 + 64) = 73{,}472$.
- **Why Forward Mode Scales Poorly with $N$:** Intermediate tangents $\mathrm{d}u/\mathrm{d}w$ and $\mathrm{d}v/\mathrm{d}w$ are specific to $w$, requiring $N$ separate sweeps ($4 \times 6 = 24$ mults) for $N$ inputs.
:::

::: {.col .fig .big}
@fig:ch34-calc-forward-mode-trace
:::
:::
:::

::: {.slide title="Executing Forward Mode in Plain Python"}
[Code Verification]{.kicker}

Running the forward pass and pushing the sensitivity to $w$ forward in Python confirms $\mathrm{d}f/\mathrm{d}w = 73472$ at a cost of $6$ multiplications for that single input:

@-multivariable-calculus-the-backpropagation-algorithm-1
:::

::: {.slide title="Executing Forward Mode in Plain Python (cont.)"}
[Code Verification]{.kicker}

@!multivariable-calculus-the-backpropagation-algorithm-1

::: {.d2l-note .warn}
Computing the full gradient over all $4$ inputs by forward mode requires $4 \times 6 = 24$ multiplications ($N$ separate forward sweeps for $N$ inputs).
:::
:::

::: {.slide title="Adjoints and Reverse-Mode Path Factoring"}
[Backward Accumulation]{.kicker}

::: {.cols .vc}
::: {.col}
- **Flip the perspective—fix the output $f$:** Define the **adjoint** $\bar{n} = \frac{\mathrm{d}f}{\mathrm{d}n}$ of node $n$ as the total derivative of the final output $f$ with respect to $n$, seeded at $\bar{f} = 1$.
- **Group paths by their first outgoing edge:** Every path from $n$ to $f$ must step first to some immediate child $c$ of $n$, and the remaining paths from $c$ to $f$ sum to $\bar{c}$:
  $$\bar{n} = \sum_{c \in \mathrm{children}(n)} \bar{c}\,\frac{\partial c}{\partial n}$$
- **One reverse sweep for all inputs:** Visiting nodes in reverse topological order ensures every child adjoint $\bar{c}$ is ready before $\bar{n}$ is computed.
:::

::: {.col .fig}
![](../img/mdl-cal-backward-sweep.svg){width=100%}
:::
:::
:::

::: {.slide title="Tracing the Backward Sweep Across All Nodes"}
[Hand Trace: Step 2]{.kicker}

::: {.cols .vc}
::: {.col}
- **Top-Down Adjoint Recursion ($\bar{n} = \sum_c \bar{c}\frac{\partial c}{\partial n}$):** Seed $\bar{f} = 1 \implies \bar{u} = \bar{v} = 328 \implies \bar{a} = 1{,}312, \bar{b} = 11{,}808$.
- **All 4 Inputs in 12 Multiplications:** $\bar{w} = \bar{x} = 1312(2) + 11808(6) = 73{,}472$ and $\bar{y} = \bar{z} = 1312(2) - 11808(6) = -68{,}224$.
:::

::: {.col .fig .big}
@fig:ch34-calc-backward-mode-trace
:::
:::
:::

::: {.slide title="Executing the Backward Sweep in Plain Python"}
[Code Verification]{.kicker}

Because $\bar{a} = 1312$ and $\bar{b} = 11808$ already summarize the entire upper graph, each additional input costs only $2$ multiplications at the bottom layer:

@multivariable-calculus-the-backpropagation-algorithm-2

::: {.d2l-note}
Symmetry check: $w$ and $x$ enter $a$ and $b$ only through $w + x$, so $\bar{w} = \bar{x} = 73472$; similarly $y$ and $z$ enter only through $y + z$ with a minus sign in $b$, so $\bar{y} = \bar{z} = -68224$.
:::
:::

::: {.slide title="Verifying Hand Adjoints Against Framework Autograd"}
[Framework Execution]{.kicker}

Every modern automatic differentiation engine executes this exact two-phase algorithm—recording values during the forward pass and accumulating adjoints during `backward()`:

@multivariable-calculus-the-backpropagation-algorithm-3

::: {.d2l-note}
The framework returns `[73472., 73472., -68224., -68224.]`, matching our hand calculation bit-for-bit because every integer intermediate is below $2^{24} \approx 1.68 \times 10^7$, the exact integer limit of IEEE 754 `float32`.
:::
:::

::: {.slide title="Practical Pitfall: Gradient Checks in Float32"}
[Numerical Verification]{.kicker}

::: {.cols .vc}
::: {.col}
- **Competing Errors in Central Differences:** Truncation error scales as $O(h^2)$ for large $h$, while subtractive cancellation scales as $O(\varepsilon_{\text{mach}}|f|/h)$ for small $h$, giving optimal $h^\star \propto \varepsilon_{\text{mach}}^{1/3}$ and min error $O(\varepsilon_{\text{mach}}^{2/3})$.
- **Why `float64` Is Mandatory:** `float64` reaches $\sim 10^{-11}$ relative error at $h^\star \approx 10^{-5}$, whereas `float32` stalls at $\sim 10^{-4}$ and underflows $w + h == w$ to zero for $h \le 10^{-8}$.
:::

::: {.col .fig .big}
@fig:ch34-calc-grad-check-precision
:::
:::
:::

::: {.slide title="Concept Check: Chain Rule on Computation Graphs"}
[Review & Practice]{.kicker}

Work out these three questions on computation graphs and adjoints:

1. **Edge vs. total derivative:** In the graph $p = x + y$, $r = p^2$, $f = r + 3p$, what are the edge derivative $\frac{\partial f}{\partial p}$ and the total derivative (adjoint) $\bar{p} = \frac{\mathrm{d}f}{\mathrm{d}p}$ at $p = 4$?
2. **Shared intermediate adjoint:** In a network with two hidden units $h_1, h_2$, prediction $f = v_1 h_1 + v_2 h_2$, and loss $\ell = \tfrac{1}{2}(f - t)^2$, which adjoint is computed once and reused by every weight in the network?
3. **Symmetry Pitfall (Weight Initialization):** If two hidden units $h_1 = \tanh(w_1 x + b_1)$ and $h_2 = \tanh(w_2 x + b_2)$ are initialized identically ($w_1 = w_2$, $b_1 = b_2$, $v_1 = v_2$), what happens to their adjoints $\bar{w}_1, \bar{w}_2$ during backpropagation?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. The direct edge derivative holding $r$ fixed is $\frac{\partial f}{\partial p} = 3$. The total derivative sums both paths out of $p$: $\bar{p} = \frac{\partial f}{\partial r}\frac{\partial r}{\partial p} + \frac{\partial f}{\partial p} = 1(2p) + 3 = 2(4) + 3 = 11$.
2. Every parameter path to $\ell$ passes through $f$, so the prediction adjoint $\bar{f} = \frac{\mathrm{d}\ell}{\mathrm{d}f} = f - t$ is computed once and shared by all parameters.
3. Because the two paths are symmetric, $\bar{h}_1 = \bar{h}_2$, $\bar{w}_1 = \bar{w}_2$, and $\bar{b}_1 = \bar{b}_2$ at every step—so gradient descent updates both units identically forever! Hidden units must be initialized with random asymmetry.
:::
:::

::: {.slide title="Summary: Computation Graphs & Backpropagation"}
[Key Takeaways]{.kicker}

| Concept / Method | Mathematical Formula | Computational & Architectural Role |
| :--- | :--- | :--- |
| **Edge Derivative ($\partial$)** | $\frac{\partial c}{\partial n}$ (other direct parents of $c$ fixed) | Local sensitivity of a single elementary operation |
| **Total Derivative ($\mathrm{d}$ / $\bar{n}$)** | $\bar{n} = \frac{\mathrm{d}f}{\mathrm{d}n}$ (all downstream nodes respond) | True sensitivity of final output $f$ to intermediate node $n$ |
| **Path-Sum Chain Rule** | $\frac{\mathrm{d}f}{\mathrm{d}x} = \sum_{\text{paths } x \to f} \prod_{(u \to v)} \frac{\partial v}{\partial u}$ | Explains skip-connection gradient highways ($2^m$ paths in ResNets) |
| **Forward-Mode Sweep** | $\dot{n} = \frac{\mathrm{d}n}{\mathrm{d}x_j} = \sum_{p \in \mathrm{parents}(n)} \frac{\partial n}{\partial p}\dot{p}$ | Pushes one input seed forward; requires $N$ sweeps for $N$ parameters |
| **Reverse-Mode (Backprop)** | $\bar{f} = 1$, $\bar{n} = \sum_{c \in \mathrm{children}(n)} \bar{c}\frac{\partial c}{\partial n}$ | Pulls one output seed backward; computes all $N$ gradients in $1$ sweep |
| **Central-Difference Check** | $\frac{f(w+h)-f(w-h)}{2h} = f'(w) + O(h^2) + O(\frac{\varepsilon_{\text{mach}}}{h})$ | Must run in `float64` ($h \approx 10^{-5}$) to avoid `float32` roundoff floor |
:::

