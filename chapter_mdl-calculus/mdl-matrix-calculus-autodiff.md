# Automatic Differentiation
:label:`sec_mdl-matrix-calculus-autodiff`

In the Basics part, every gradient came from one call to the framework's
automatic differentiation. The previous section showed what
that call computes: the derivative of a network is a product of Jacobians,
one per layer, and the order of multiplication decides the cost. A program,
however, is a sequence of additions, multiplications and function calls, not
a list of matrices. How does the framework find the factors from the code,
and how does it multiply them in the cheap order?

How much the cheap order saves is easiest to see on a large network. A
network with $10^8$ parameters computes one number, its loss $\ell$, and every training
step needs the derivative of $\ell$ with respect to each of the $10^8$
parameters. The direct method nudges one parameter, reruns the network, and
divides the change in $\ell$ by the nudge. That is $10^8$ runs per step. At
10 ms per run, one gradient takes $10^6$ s, about 12 days, and the size of
the nudge adds an error of its own. Backpropagation returns all $10^8$
derivatives, correct up to rounding, for the cost of about three runs,
roughly 30 ms. This section explains where the factor of $10^8$ goes.

The short answer is that the product of Jacobians can be evaluated from
either end. Starting from the input end, one pass carries one input
direction, so $10^8$ inputs need $10^8$ passes. Starting from the loss end,
one pass carries one output direction, and a loss has a single output.

:numref:`sec_autograd` showed how to call both products in a framework. Here
we build them ourselves in a few lines of Python, prove that they return
derivatives, trace the backward pass by hand, and count its cost. Then we
combine the two products to compute curvature along one direction and to
differentiate the solution of an equation without differentiating the solver.
Two results are worth remembering above the rest: arithmetic on dual numbers
returns derivatives (the one proposition of the section), and a sweep costs
a small multiple of one run of the program, forward mode needing one sweep
per input direction and reverse mode one per output direction. The last
part, :numref:`subsec_mdl-composing-products`, can wait for a second reading.
Three small examples recur: the chapter's log-sum-exp
$\ell(\mathbf x)=\log\sum_j e^{x_j}$, whose gradient is the softmax; the
plane map $\mathbf f(x,y)=(x+\sin y,\ y+x^2/2)$ from the previous section;
and the expression $y=(uv+u)^2$, which is small enough to trace by hand.

We use the following imports throughout this section.

```{.python .input #matrix-calculus-autodiff-imports}
#@tab mxnet
from mxnet import autograd, npx
from mxnet import np as mnp
import numpy as np
npx.set_np()
```

```{.python .input #matrix-calculus-autodiff-imports}
#@tab pytorch
import numpy as np
import torch
```

```{.python .input #matrix-calculus-autodiff-imports}
#@tab tensorflow
import numpy as np
import tensorflow as tf
```

```{.python .input #matrix-calculus-autodiff-imports}
#@tab jax
import numpy as np
import jax
from jax import numpy as jnp
```

## Forward Mode and Dual Numbers
:label:`subsec_mdl-forward-mode`

### Dual Numbers

We want the value of a program and its derivative from a single run, with no
step size to choose and no formula for $f'$ written down. The device is a small
extension of the real numbers. Adjoin a symbol $\varepsilon$ that is not zero
but whose square is,

$$
\varepsilon^2 = 0, \qquad \varepsilon\neq0,
$$
:eqlabel:`eq_mdl-dual-rule`

and compute with expressions $a+b\varepsilon$, called *dual numbers*, by the
usual rules of algebra. Every differentiation rule of
:numref:`sec_mdl-single_variable_calculus` discards second-order terms, and the
rule $\varepsilon^2=0$ makes that step part of the arithmetic.

Let us see what the rule does to $3+\varepsilon$. Squaring and cubing it
gives

$$
(3+\varepsilon)^2=9+6\varepsilon+\varepsilon^2=9+6\varepsilon,
\qquad
(3+\varepsilon)^3=(9+6\varepsilon)(3+\varepsilon)=27+27\varepsilon .
$$

The $\varepsilon$-parts $6$ and $27$ are the derivatives of $x^2$ and $x^3$ at
$x=3$. No differentiation rule was applied; the derivatives came out of
multiplication. The same happens for any product. Multiplying out and dropping the
$\varepsilon^2$ term gives

$$
(a+b\varepsilon)(c+d\varepsilon) = ac + (ad+bc)\,\varepsilon.
$$
:eqlabel:`eq_mdl-dual-mul`

If $b$ and $d$ are the derivatives of two quantities whose values are $a$ and
$c$, then $ad+bc$ is the derivative of their product. Multiplication of dual
numbers performs the product rule.

A reader may object that no number is nonzero with a zero square. No real
number is, but a matrix can be. Take
$\varepsilon=\left(\begin{smallmatrix}0&1\\0&0\end{smallmatrix}\right)$. Then
$\varepsilon^2=\mathbf 0$, the dual number $a+b\varepsilon$ is the matrix
$\left(\begin{smallmatrix}a&b\\0&a\end{smallmatrix}\right)$, and the matrix
product
$\left(\begin{smallmatrix}a&b\\0&a\end{smallmatrix}\right)\left(\begin{smallmatrix}c&d\\0&c\end{smallmatrix}\right)=\left(\begin{smallmatrix}ac&ad+bc\\0&ac\end{smallmatrix}\right)$
is :eqref:`eq_mdl-dual-mul`. Dual numbers are as concrete as $2\times2$
matrices, just as complex numbers are as concrete as the rotations and
scalings of the plane.

Programs also call functions such as $\sin$, $\exp$, and $\log$. For a function
$\varphi$ given by its Taylor series around $a$, substituting $a+b\varepsilon$
into the series removes every term from $(b\varepsilon)^2$ on, which leaves

$$
\varphi(a+b\varepsilon)=\varphi(a)+\varphi'(a)\,b\,\varepsilon .
$$

So $e^{a+b\varepsilon}=e^a+e^a b\,\varepsilon$ and
$\sin(a+b\varepsilon)=\sin a+(\cos a)\,b\,\varepsilon$ are forced by the
algebra. For a primitive that is only differentiable at $a$, we take the
display as the definition of its action on dual numbers. In both cases the
$\varepsilon$-part follows the chain rule: the derivative $b$ of the argument
is multiplied by $\varphi'(a)$.

**Proposition (dual numbers compute derivatives).** *Let $f$ be an expression
built from constants, the identity, sums, products, and primitives $\varphi$
(such as $\sin$, $\exp$, $\log$). Let each primitive be differentiable at the
argument it receives when $f$ is evaluated at $x$ (for $\log$, a positive
argument). Then evaluating the same expression on the dual number
$x+\varepsilon$ gives*

$$
f(x + \varepsilon) = f(x) + f'(x)\,\varepsilon.
$$
:eqlabel:`eq_mdl-dual-eval`

**Proof.** We use induction on the expression. A constant $c$ becomes
$c+0\varepsilon$ and the identity becomes $x+1\varepsilon$; their
$\varepsilon$-parts are their derivatives $0$ and $1$. Suppose two
subexpressions satisfy $g(x+\varepsilon)=g(x)+g'(x)\varepsilon$ and
$h(x+\varepsilon)=h(x)+h'(x)\varepsilon$. Their sum has $\varepsilon$-part
$g'+h'=(g+h)'$. By :eqref:`eq_mdl-dual-mul`, their product has
$\varepsilon$-part $g'h+gh'=(gh)'$. A primitive applied to $g$ has, by the
rule above with $a=g(x)$ and $b=g'(x)$, the $\varepsilon$-part
$\varphi'(g(x))\,g'(x)=(\varphi\circ g)'(x)$. Each way of building a larger
expression from smaller ones preserves the property, so the whole expression
has it. $\blacksquare$

To differentiate a program, we therefore run it on $x+\varepsilon$ and read the
$\varepsilon$-part of the output. A program with loops and branches is covered
as well: for a given input it executes one particular sequence of operations,
which is an expression of this kind, and the dual run differentiates that
sequence (:numref:`subsec_autograd_control_flow` shows this for the
frameworks). The input's $\varepsilon$-part, here $1$, is called the *seed*.
No formula for $f'$ appears and no step size is chosen; the only error is the
rounding of the arithmetic itself.

### A Forward-Mode Engine

Operator overloading turns the algebra into code. The class below stores the
value $a$ and the derivative part $b$, which we call the *tangent* and write
$\dot v$ for a quantity $v$. It implements :eqref:`eq_mdl-dual-mul` for `*`
and gives $\sin$, $\exp$, and $\log$ the primitive rule. A class counter `ops`
records how many primitive operations have run; we use it to measure cost
in :numref:`subsec_mdl-cost-asymmetry`.

```{.python .input #matrix-calculus-autodiff-dual-numbers}
class Dual:
    """A dual number a + b*eps with eps^2 = 0; b carries the derivative."""
    ops = 0                                         # primitive operations run
    def __init__(self, a, b=0.0):
        self.a, self.b = a, b                       # value, tangent
    def __add__(self, o):
        o = o if isinstance(o, Dual) else Dual(o)
        Dual.ops += 1
        return Dual(self.a + o.a, self.b + o.b)
    __radd__ = __add__
    def __mul__(self, o):
        o = o if isinstance(o, Dual) else Dual(o)
        Dual.ops += 1
        return Dual(self.a * o.a, self.a * o.b + self.b * o.a)   # product rule
    __rmul__ = __mul__
    def sin(self):
        Dual.ops += 1
        return Dual(np.sin(self.a), np.cos(self.a) * self.b)     # chain rule
    def exp(self):
        Dual.ops += 1
        return Dual(np.exp(self.a), np.exp(self.a) * self.b)
    def log(self):
        Dual.ops += 1
        return Dual(np.log(self.a), self.b / self.a)
    def __repr__(self):
        return f'{np.round(self.a, 6)} + {np.round(self.b, 6)} eps'
```

A first run, on $f(x)=\sin(x^2)+e^x$ at $1.3+\varepsilon$, with the hand
derivative $f'(x)=2x\cos(x^2)+e^x$ printed beside the $\varepsilon$-part:

```{.python .input #matrix-calculus-autodiff-dual-eval}
def f(x):
    return (x * x).sin() + x.exp()                  # sin(x^2) + e^x, in dual algebra

x0 = 1.3
out = f(Dual(x0, 1.0))                              # seed the derivative with 1
exact = 2 * x0 * np.cos(x0**2) + np.exp(x0)         # f'(x) by hand
print('dual result :', out)
print('value f(x)  :', np.sin(x0**2) + np.exp(x0))
print("derivative  : dual %.6f  vs  exact %.6f" % (out.b, exact))
```

One run returns the value $4.6622$ and the derivative $3.360101$.

A function of one variable hides the weakness of the method. Log-sum-exp has
three inputs at $\mathbf x=(1,-0.5,2)$, and its gradient is the softmax,
$\partial\ell/\partial x_j=e^{x_j}/\sum_k e^{x_k}=p_j$. How many runs does
forward mode need for this gradient? Seeding input $j$ with $1$ and the other
inputs with $0$ makes the $\varepsilon$-part of the output the partial
derivative $\partial\ell/\partial x_j$, and nothing more.

```{.python .input #mdl-matrix-calculus-autodiff-a-forward-mode-engine-1}
def lse_dual(xs):                                   # log-sum-exp, in dual algebra
    total = 0.0
    for x in xs:
        total = total + x.exp()
    return total.log()

x_lse = [1.0, -0.5, 2.0]
p = np.exp(x_lse) / np.exp(x_lse).sum()             # softmax, the known gradient
for j in range(3):                                  # one pass per input direction
    out = lse_dual([Dual(x, 1.0 if k == j else 0.0) for k, x in enumerate(x_lse)])
    print(f'seed e{j + 1}: value {out.a:.6f}, partial {out.b:.6f}, '
          f'softmax {p[j]:.6f}')
```

It needs three. Each pass returns the value $2.371539$ again and a single
softmax entry, $0.253716$, $0.056612$, or $0.689672$. A gradient over $n$
inputs therefore costs $n$ passes. For the network of the opening, with
$n=10^8$, that is again $10^8$ runs of the program. Forward mode removes the
step size but not the number of runs.

### Vector Tangents and Jacobian Columns

A map $\mathbf f:\mathbb R^n\to\mathbb R^m$ has an $m\times n$ Jacobian
$\mathbf J$. What does one forward pass compute for it? Seed the inputs with a
direction $\mathbf v$, so that input $j$ is $x_j+v_j\varepsilon$. Applying the
proposition to the function $t\mapsto f_i(\mathbf x+t\mathbf v)$ shows that
output $i$ comes out as $f_i(\mathbf x)+(\nabla f_i\cdot\mathbf v)\,\varepsilon$,
so the $\varepsilon$-parts of the outputs form the vector $\mathbf J\mathbf v$.
This product is the *Jacobian--vector product* (JVP). With
$\mathbf v=\mathbf e_j$ it is column $j$ of $\mathbf J$. One pass carries one
input direction.

Take the plane map $\mathbf f(x,y)=(x+\sin y,\ y+x^2/2)$ at
$(0.5,0.5)$. Its Jacobian, row $i$ holding the partial derivatives of output
$i$, is

$$
\mathbf J=\begin{pmatrix}1&\cos y\\ x&1\end{pmatrix}
=\begin{pmatrix}1&0.877583\\0.5&1\end{pmatrix}.
$$

Two passes, seeded with $\mathbf e_1$ and then $\mathbf e_2$, return its two
columns.

```{.python .input #matrix-calculus-autodiff-forward-jvp}
def plane_dual(x, y):
    return [x + y.sin(), y + 0.5 * (x * x)]         # (x + sin y, y + x^2/2)

z0 = (0.5, 0.5)
cols = []
for seed in ((1.0, 0.0), (0.0, 1.0)):               # e1, then e2
    out = plane_dual(Dual(z0[0], seed[0]), Dual(z0[1], seed[1]))
    cols.append([out[0].b, out[1].b])               # tangent = J @ seed, a column
J_fwd = np.array(cols).T                            # columns side by side
J_exact = np.array([[1.0, np.cos(0.5)], [0.5, 1.0]])
print('forward-mode J:\n', J_fwd.round(6))
print('matches the analytic Jacobian:', np.allclose(J_fwd, J_exact))
```

Nothing in `Dual` requires the tangent to be a number. If the tangent of $x$ is
the array $(1,0)$ and the tangent of $y$ is $(0,1)$, every operation carries
both directions at once, and the tangent of output $i$ becomes row $i$ of
$\mathbf J$. Can one pass return the whole matrix?

```{.python .input #mdl-matrix-calculus-autodiff-vector-tangents-and-jacobian-columns-1}
out = plane_dual(Dual(z0[0], np.array([1.0, 0.0])),     # identity seed:
                 Dual(z0[1], np.array([0.0, 1.0])))     # both directions at once
J_one = np.array([out[0].b, out[1].b])                  # row i = tangent of f_i
print('one-pass J:\n', J_one.round(6))
print('matches the analytic Jacobian:', np.allclose(J_one, J_exact))
```

It can, but the saving is smaller than it looks. Every tangent operation now
works on arrays of length $n$, so the single pass does about $n$ times the
arithmetic of a scalar pass. Passes and work are different measures, and
:numref:`subsec_mdl-cost-asymmetry` counts work. For a scalar loss with $10^8$ inputs,
forward mode needs $10^8$ passes, or one pass with tangents of $10^8$ entries.
Either way its cost is proportional to $n$ runs of the program.

## Reverse Mode and the Tape
:label:`subsec_mdl-reverse-mode`

### Two Orders of One Product

To remove the factor $n$ we need a different order of computation.
:numref:`sec_mdl-matrix-calculus` showed that the Jacobian of a program of
$K$ steps, $\mathbf f=\mathbf f_K\circ\cdots\circ\mathbf f_1$, is the product
:eqref:`eq_mdl-jacobian-product` of the Jacobians of its steps,

$$
\mathbf J = \mathbf J_K\,\mathbf J_{K-1}\cdots\mathbf J_1,
$$
:eqlabel:`eq_mdl-ad-jacobian-product`

and that the order of multiplication decides the cost: for three layers of
width $1000$ with a row on the left, multiplying from the left takes
$3\times10^6$ multiplications where the other order takes $2\times10^9$
(:numref:`fig_mdl-cal-jacobian-chain-order`). A program that differentiates
never forms the factors. It holds a seed vector and passes it through one
step after another. Forward mode multiplies a direction $\mathbf v$ in from
the right; the alternative multiplies a row $\mathbf u^\top$ in from the left:

$$
\mathbf J\mathbf v=\mathbf J_K\Bigl(\cdots\bigl(\mathbf J_2(\mathbf J_1\mathbf v)\bigr)\Bigr),
\qquad
\mathbf u^\top\mathbf J=\Bigl(\bigl((\mathbf u^\top\mathbf J_K)\mathbf J_{K-1}\bigr)\cdots\Bigr)\mathbf J_1.
$$
:eqlabel:`eq_mdl-ad-two-orders`

With a seed, every step multiplies a matrix by a vector and never a matrix by
a matrix. The left formula carries an input direction forward; it is the JVP of
forward mode. The right formula carries an output direction backward. Its
result $\mathbf u^\top\mathbf J$ is the *vector--Jacobian product* (VJP), a
combination of the rows of $\mathbf J$ with weights $\mathbf u$; for
$\mathbf u=\mathbf e_i$ it is row $i$.

The cost rule of automatic differentiation follows. Forward mode costs one
sweep per input direction, and reverse mode one sweep per output direction.
Each sweep costs a small constant multiple of one evaluation of the program.
For programs built from $+,-,\times,\div$, Baur and Strassen showed that all
partial derivatives together take at most a fixed multiple of the operations
of the program itself, between three and five depending on which operations
are counted :cite:`Baur.Strassen.1983`. A scalar loss has a $1\times n$
Jacobian, a single row, so one reverse sweep seeded with $1$ returns the
whole gradient.
:numref:`fig_mdl-cal-jacobian-shapes` draws the rule as shapes.

![The shape of the Jacobian decides the cheaper mode. A scalar loss has a $1\times n$ Jacobian: one row, one VJP, one backward pass. A map with one input has an $m\times 1$ Jacobian: one column, one JVP, one forward pass. A full $m\times n$ Jacobian costs $\min(m,n)$ passes with the cheaper mode.](../img/mdl-cal-jacobian-shapes.svg)
:label:`fig_mdl-cal-jacobian-shapes`

The two modes are the same product read from opposite ends. Which end is
cheaper depends only on the shape of $\mathbf J$: the cheap mode is the one
whose number of seeds matches the short side. A loss, with one output and many
inputs, has the most lopsided shape possible.

### The Tape

Reverse mode does not carry the quantity that forward mode carries. In a
program with input $x$ and output $y$, the tangent $\dot v$ of an intermediate
value $v$ is the rate at which $v$ changes when the input moves in the
direction $\dot x$. The adjoint $\bar v$ of :eqref:`eq_mdl-adjoint` is the
total derivative of the output $y$ with respect to $v$, summed over all paths
from $v$ to $y$. A tangent is determined by the steps before $v$, an adjoint
by the steps after it. For $y=v^2$ with $v=3x$, for instance,
$\dot v=3\dot x$ and $\bar v=2v$.

This difference decides how each mode runs
(:numref:`fig_mdl-cal-fwd-vs-rev`). Tangents travel with the values, so forward
mode finishes in the same sweep as the evaluation; `Dual` needs no memory of
the past. Adjoints travel against the order of execution: the adjoint of a node
needs the adjoints of the nodes that use it, and those exist only once the
program has run to the end. Reverse mode therefore cannot finish in one forward
sweep. It records the computation as it runs and then replays the record
backward. The record is called the *tape*, or Wengert list, after the list of
elementary operations that :citet:`Wengert.1964` introduced for forward mode;
running it backward is reverse mode, introduced by :citet:`Linnainmaa.1970`.

![Forward and reverse mode on a chain $x\to a\to b\to\ell$. Both evaluate the chain from left to right (gray). Forward mode (top) carries the tangents $\dot v$ along with the values, one input direction per sweep. Reverse mode (bottom) runs the chain first and then carries the adjoints $\bar v$, the total derivatives of $\ell$ with respect to $v$, back from the seed $\bar\ell=1$, one output direction per sweep, which for a scalar loss is the whole gradient.](../img/mdl-cal-fwd-vs-rev.svg)
:label:`fig_mdl-cal-fwd-vs-rev`

A tape node stores its value, its parents, and a backward closure. Given the
adjoint $g$ of the node, the closure adds the node's contribution to the
adjoints of its parents: for $c=a+b$ it adds $g$ to both, and for $c=ab$ it adds
$b\,g$ to $\bar a$ and $a\,g$ to $\bar b$. The function `backprop` sorts the
nodes so that every node comes after its parents, seeds the output adjoint with
$1$, and runs the closures in reverse order. Like `Dual`, the class counts its
operations, forward and backward.

```{.python .input #matrix-calculus-autodiff-tape}
class Var:
    """A scalar node on the autodiff tape; .grad accumulates the adjoint."""
    ops = 0                                         # recorded ops + backward steps
    def __init__(self, value, parents=(), backward=lambda g: None):
        self.value = value
        self.grad = 0.0
        self._parents = parents                     # nodes this one depends on
        self._backward = backward                   # pushes adjoint to parents
    def __add__(self, o):
        o = o if isinstance(o, Var) else Var(o)
        Var.ops += 1
        out = Var(self.value + o.value, (self, o))
        def back(g):                                # d(a+b) = 1*g to each parent
            Var.ops += 1
            self.grad += g; o.grad += g
        out._backward = back
        return out
    __radd__ = __add__
    def __mul__(self, o):
        o = o if isinstance(o, Var) else Var(o)
        Var.ops += 1
        out = Var(self.value * o.value, (self, o))
        def back(g):                                # product rule, reversed
            Var.ops += 1
            self.grad += o.value * g; o.grad += self.value * g
        out._backward = back
        return out
    __rmul__ = __mul__

def backprop(node):
    """Topologically sort the tape and replay it backward from `node`."""
    order, seen = [], set()
    def visit(v):
        if v not in seen:
            seen.add(v)
            for p in v._parents:
                visit(p)
            order.append(v)
    visit(node)
    node.grad = 1.0                                 # seed the output adjoint
    for v in reversed(order):
        v._backward(v.grad)
```

`Var` implements only `+` and `*`, enough for the expression $y=(uv+u)^2$,
which we first trace by hand. At $u=2$, $v=-3$ it is recorded as three
operations: $t=uv=-6$, then $r=t+u=-4$, then $y=r\cdot r=16$. The backward
sweep visits the nodes in an order such as $y,r,t,u,v$, in which every node
comes after all the nodes that use it (`backprop` happens to take $v$ before
$u$, which is just as good). Each adjoint is the sum of the
contributions along the node's outgoing edges, and each contribution is the
adjoint of the consumer times the local derivative:

| node | outgoing edges and their contributions | adjoint |
|----|----|----|
| $y$ | none: the seed | $\bar y=1$ |
| $r$ | both arguments of $y=r\cdot r$: $\bar y\,r+\bar y\,r$ | $\bar r=-8$ |
| $t$ | $r=t+u$: $\bar r\cdot1$ | $\bar t=-8$ |
| $u$ | $r=t+u$ and $t=uv$: $\bar r\cdot1+\bar t\cdot v$ | $\bar u=-8+24=16$ |
| $v$ | $t=uv$: $\bar t\cdot u$ | $\bar v=-16$ |

Differentiating $y=(uv+u)^2$ as a function of $u$ and $v$ confirms both
results:
$\partial y/\partial u=2(uv+u)(v+1)=2(-4)(-2)=16$ and
$\partial y/\partial v=2(uv+u)\,u=2(-4)(2)=-16$. The sums matter at two nodes:
$r$ feeds both arguments of the last product, and $u$ feeds both $t$ and $r$
(:numref:`fig_mdl-cal-tape-dag`). Overwriting instead of adding would give
$\bar r=-4$, and even with $\bar r$ correct it would give $\bar u=-8$ or $24$,
depending on the order of the closures.
An adjoint is the sum over outgoing edges, which is the chain rule's sum over
paths, and this is why the closures use `+=`.

![The tape of $y=(uv+u)^2$ at $u=2$, $v=-3$. Each box shows the forward value of its node; the orange label shows its adjoint, the derivative of $y$ with respect to that node. The graph is a diamond, not a chain: $u$ feeds both $t=uv$ and $r=t+u$, and $r$ feeds both arguments of $y=r\cdot r$, so the adjoints of $r$ and $u$ are sums of two contributions.](../img/mdl-cal-tape-dag.svg)
:label:`fig_mdl-cal-tape-dag`

Recorded with `Var` and swept once, the same expression gives:

```{.python .input #mdl-matrix-calculus-autodiff-the-tape-1}
u, v = Var(2.0), Var(-3.0)
r = u * v + u                                       # t = uv, then r = t + u
y = r * r                                           # r feeds both arguments
backprop(y)
print('our tape  : dy/du = %.4f  dy/dv = %.4f' % (u.grad, v.grad))
```

These are $\bar u$ and $\bar v$ of the table. The framework's own reverse
mode is the same procedure on a larger scale, and returns the same two numbers:

```{.python .input #matrix-calculus-autodiff-tape-check}
#@tab pytorch
ut = torch.tensor(2.0, requires_grad=True)
vt = torch.tensor(-3.0, requires_grad=True)
rt = ut * vt + ut
(rt * rt).backward()                                # framework autograd
print('torch     : dy/du = %.4f  dy/dv = %.4f' % (ut.grad, vt.grad))
```

```{.python .input #matrix-calculus-autodiff-tape-check}
#@tab mxnet
um, vm = mnp.array(2.0), mnp.array(-3.0)
um.attach_grad(); vm.attach_grad()
with autograd.record():                             # framework autograd
    rm = um * vm + um
    ym = rm * rm
ym.backward()
print('mxnet     : dy/du = %.4f  dy/dv = %.4f' % (float(um.grad), float(vm.grad)))
```

```{.python .input #matrix-calculus-autodiff-tape-check}
#@tab tensorflow
ut, vt = tf.Variable(2.0), tf.Variable(-3.0)
with tf.GradientTape() as tape:                     # framework autograd
    rt = ut * vt + ut
    yt = rt * rt
du, dv = tape.gradient(yt, [ut, vt])
print('tensorflow: dy/du = %.4f  dy/dv = %.4f' % (float(du), float(dv)))
```

```{.python .input #matrix-calculus-autodiff-tape-check}
#@tab jax
def g(u, v):
    r = u * v + u
    return r * r
du, dv = jax.grad(g, argnums=(0, 1))(2.0, -3.0)     # framework autograd
print('jax       : dy/du = %.4f  dy/dv = %.4f' % (float(du), float(dv)))
```

A framework's tape differs from ours in three ways that leave the procedure
unchanged. Its nodes hold tensors, so each backward closure is the VJP rule of
a whole operation, such as :eqref:`eq_mdl-matmul-vjp` for a matrix product.
Its set of primitives is large, each with its own VJP rule. And it builds the
graph in one of two ways. An eager tape records operations as the program
runs, as `Var` does. A tracer runs the program once on placeholder inputs to
obtain its sequence of operations, forms the forward-mode rule of each (a
linear map of tangents), and reverses the sequence, replacing each linear map
by its transpose. The result is the same backward sweep. Backpropagation is
this procedure applied to a neural network
:cite:`Rumelhart.Hinton.Williams.ea.1988`.

:begin_tab:`pytorch`
PyTorch records eagerly: every operation on a tensor with
`requires_grad=True` adds a node, and `backward()` replays the nodes in reverse.
:end_tab:

:begin_tab:`mxnet`
MXNet records eagerly, inside `autograd.record()`; `backward()` replays the
record in reverse.
:end_tab:

:begin_tab:`tensorflow`
TensorFlow records eagerly, inside a `GradientTape` context; `tape.gradient`
replays the record in reverse.
:end_tab:

:begin_tab:`jax`
JAX is a tracer: `jax.grad` returns a new function built in this way.
:end_tab:

### Parameter Gradients
:label:`subsec_mdl-param-gradients`

The product :eqref:`eq_mdl-jacobian-product` is the derivative with respect
to the input, and training needs the derivative with respect to the weights.
They are read off during the same sweep by the backward rule of a layer
(:numref:`subsec_mdl-layer-vjp`). For a layer
$\mathbf h_k=\mathbf W_k\mathbf h_{k-1}$ with $\mathbf W_k$ of shape
$d_k\times d_{k-1}$, the rule reads

$$
\nabla_{\mathbf W_k}\ell=\bar{\mathbf h}_k\,\mathbf h_{k-1}^\top,
$$
:eqlabel:`eq_mdl-ad-param-grad`

the outer product of the adjoint arriving from above and the activation
arriving from below. A weight matters in proportion to how much its output
matters to the loss and how large its input is. What concerns us here is the
cost of the rule.

The outer product costs $d_kd_{k-1}$ multiplications, as many as the forward
product $\mathbf W_k\mathbf h_{k-1}$. Passing the adjoint down,
$\bar{\mathbf h}_{k-1}^\top=\bar{\mathbf h}_k^\top\mathbf W_k$, costs the same
again. Counted in multiplications, the backward sweep therefore costs about
two forward passes, and forward pass plus backward sweep about three, which
is the 30 ms of the opening.
Forward mode, in contrast, treats each weight as an input direction, so a single
layer would need $d_kd_{k-1}$ passes. This is why no framework trains in forward
mode.

### Cost of the Two Modes
:label:`subsec_mdl-cost-asymmetry`

The two engines let us count operations instead of arguing about them. Take
$f(\mathbf x)=(\sum_j x_j^2)^2$ with $n=200$ inputs $x_j=0.01\,j$ for
$j=1,\ldots,200$, built from $+$ and $\times$ alone so that `Var` can record
it (log-sum-exp would need `exp` and `log` closures, the mirror of the `Dual`
methods). Its gradient is
$\nabla f=4\bigl(\sum_jx_j^2\bigr)\mathbf x$, and one evaluation runs $2n+1=401$
primitive operations: $n$ squares, $n$ additions, and the final square.
Forward mode will need $200$ passes of $401$ operations. It is worth guessing
the count for reverse mode before reading the output.

```{.python .input #mdl-matrix-calculus-autodiff-the-cost-asymmetry-measured}
n = 200
def f(xs):                       # f: R^n -> R, from + and * alone
    total = 0.0
    for x in xs:
        total = total + x * x
    return total * total         # (sum of squares)^2

xs0 = [0.01 * (j + 1) for j in range(n)]
Dual.ops = 0
grad_fwd = [f([Dual(x, 1.0 if k == j else 0.0) for k, x in enumerate(xs0)]).b
            for j in range(n)]   # forward mode: one pass per input direction
Var.ops = 0
xs = [Var(x) for x in xs0]       # reverse mode: one recorded pass...
backprop(f(xs))                  # ...and one backward sweep for all n partials
grad_rev = [x.grad for x in xs]

gap = max(abs(a - b) for a, b in zip(grad_fwd, grad_rev))
print(f'forward mode: {Dual.ops:,} operations ({n} passes of {2 * n + 1})')
print(f'reverse mode: {Var.ops:,} operations (one recorded pass, one sweep)')
print(f'ratio {Dual.ops / Var.ops:.1f}, largest gradient difference {gap:.1e}')
```

Forward mode ran $80{,}200=n(2n+1)$ operations and reverse mode
$802=2(2n+1)$: $401$ recorded on the forward pass and $401$ backward closures.
The ratio is $100=n/2$, and the two gradients agree to every printed digit. The
pass counts differ by a factor of $n$ and the operation counts by $n/2$ for
these engines; at $n=10^8$ the ratio is the factor of the opening, up to the
same constant.

The measured factor of two for a gradient seems to contradict the three of
the opening. It does not, because the counter treats a backward closure as
one operation. The closure of a product performs two multiplications, so the
sweep does $402$ multiplications against $201$ on the forward pass, and
gradient and value together cost three times the value, as for the dense
layer. Counting the additions that accumulate the adjoints as well gives
$1204$ operations for the sweep and a factor of four, still within the bound
of Baur and Strassen.

Reverse mode needs more memory instead. The tape keeps all $401$
recorded values until the sweep consumes them, while forward mode, like the
program itself, keeps only the current values, each with its tangent.

### Gradient Checkpointing
:label:`subsec_mdl-checkpointing`

A computation of $T$ recorded steps keeps $T$ values alive until the backward
sweep reaches them. In a deep network these are the activations of every layer
for every example in the batch, and this memory, not the arithmetic, often
limits the batch size and the depth.

The simplest remedy cuts the tape into segments of $s$ steps. The forward pass
keeps only every $s$-th value, the $T/s$ *checkpoints*, and discards the rest.
When the backward sweep reaches a segment, it recomputes that segment's $s$
values from its checkpoint onto a fresh tape, sweeps it, and frees it. At any
moment the memory holds the checkpoints and one segment, and the best segment
length balances the two:

$$
M(s)=\frac{T}{s}+s,
\qquad
\min_s M(s)=M(\sqrt T)=2\sqrt T.
$$
:eqlabel:`eq_mdl-ad-checkpoint-memory`

The minimum follows from $M'(s)=1-T/s^2=0$, or from $T/s+s\ge2\sqrt T$ by the
inequality of arithmetic and geometric means. Memory grows like the square root
of the length instead of the length itself. Each step is recomputed once, so
the extra time is one forward pass. For $T=400$, the segment lengths
$s=1,5,10,20,100,400$ give $M=401,85,50,40,104,401$, smallest at
$s=20=\sqrt{400}$. For $T=10^4$, segments of $s=100$ keep $200$ values instead
of $10{,}000$ :cite:`Chen.Xu.Zhang.ea.2016`. :numref:`fig_mdl-cal-checkpointing`
draws the schedule.

![Gradient checkpointing on a tape of $T$ steps. Plain reverse mode (top) keeps all $T$ values until the backward sweep uses them. Checkpointing (bottom) keeps every $s$-th value; when the backward sweep reaches a segment, the segment is recomputed from its checkpoint and then swept. The memory is $T/s+s$, which is $2\sqrt T$ at $s=\sqrt T$, for one extra forward pass.](../img/mdl-cal-checkpointing.svg)
:label:`fig_mdl-cal-checkpointing`

Checkpointing changes the time and the memory but not the gradient. The
recomputed values are the same numbers, so the result equals that of plain
reverse mode. Frameworks offer the same trade for the blocks of a deep
network: a checkpointed block keeps only its input and is run a second time
when the backward sweep reaches it. Recursive schedules, which checkpoint inside segments as well,
reach $O(\log T)$ memory with $O(T\log T)$ recomputation.

## Composing Jacobian Products
:label:`subsec_mdl-composing-products`

A framework exposes the two products our engines compute, $\mathbf J\mathbf v$
and $\mathbf u^\top\mathbf J$. Curvature along a direction, derivatives of
solutions, and full Jacobians when they are needed are all built by composing
them.

### Jacobian--Vector and Vector--Jacobian Products
:label:`subsec_mdl-jvp-vjp`

The framework calls for the two products are those of
:numref:`subsec_autograd_fwd_rev`. For the plane map at $(0.5,0.5)$ and
$\mathbf v=\mathbf u=(1,2)$, the Jacobian found by `Dual` gives
$\mathbf J\mathbf v=(1+2\cdot0.8776,\ 0.5+2)=(2.7552,\ 2.5)$ and
$\mathbf u^\top\mathbf J=(1+2\cdot0.5,\ 0.8776+2)=(2,\ 2.8776)$, and the
calls return these numbers.

```{.python .input #mdl-matrix-calculus-autodiff-jacobian-vector-and-vector-jacobian-products}
#@tab pytorch
def plane(z):                                       # (x + sin y, y + x^2/2)
    return torch.stack([z[0] + torch.sin(z[1]), z[1] + 0.5 * z[0]**2])

z0 = torch.tensor([0.5, 0.5])
v, u = torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0])
_, Jv = torch.func.jvp(plane, (z0,), (v,))          # forward mode: J v
_, vjp_fn = torch.func.vjp(plane, z0)               # reverse mode: records once,
uJ, = vjp_fn(u)                                     # then u^T J per seed
J = torch.func.jacrev(plane)(z0)                    # one VJP per row
print('J v   =', Jv.numpy().round(4))
print('u^T J =', uJ.numpy().round(4))
print('J     =', J.numpy().astype(float).round(4).tolist())
```

```{.python .input #mdl-matrix-calculus-autodiff-jacobian-vector-and-vector-jacobian-products}
#@tab mxnet
def plane(z):                                       # (x + sin y, y + x^2/2)
    return mnp.stack([z[0] + mnp.sin(z[1]), z[1] + 0.5 * z[0]**2])

def vjp(u):                                         # one reverse sweep seeded with u
    z = mnp.array([0.5, 0.5])
    z.attach_grad()
    with autograd.record():
        y = plane(z)
    y.backward(u)
    return z.grad

v, u = mnp.array([1.0, 2.0]), mnp.array([1.0, 2.0])
uJ = vjp(u)                                         # reverse mode: u^T J
J = mnp.stack([vjp(mnp.array([1.0, 0.0])),          # one VJP per row
               vjp(mnp.array([0.0, 1.0]))])
Jv = mnp.dot(J, v)                                  # no forward mode: J v from J
print('J v   =', Jv.asnumpy().round(4))
print('u^T J =', uJ.asnumpy().round(4))
print('J     =', J.asnumpy().astype(float).round(4).tolist())
```

```{.python .input #mdl-matrix-calculus-autodiff-jacobian-vector-and-vector-jacobian-products}
#@tab tensorflow
def plane(z):                                       # (x + sin y, y + x^2/2)
    return tf.stack([z[0] + tf.sin(z[1]), z[1] + 0.5 * z[0]**2])

z0 = tf.constant([0.5, 0.5])
v, u = tf.constant([1.0, 2.0]), tf.constant([1.0, 2.0])
with tf.autodiff.ForwardAccumulator(z0, v) as acc:  # forward mode: J v
    y = plane(z0)
Jv = acc.jvp(y)
with tf.GradientTape(persistent=True) as tape:      # reverse mode: one tape
    tape.watch(z0)
    y = plane(z0)
uJ = tape.gradient(y, z0, output_gradients=u)       # u^T J
J = tape.jacobian(y, z0)                            # one VJP per row
print('J v   =', Jv.numpy().round(4))
print('u^T J =', uJ.numpy().round(4))
print('J     =', J.numpy().astype(float).round(4).tolist())
```

```{.python .input #mdl-matrix-calculus-autodiff-jacobian-vector-and-vector-jacobian-products}
#@tab jax
def plane(z):                                       # (x + sin y, y + x^2/2)
    return jnp.stack([z[0] + jnp.sin(z[1]), z[1] + 0.5 * z[0]**2])

z0 = jnp.array([0.5, 0.5])
v, u = jnp.array([1.0, 2.0]), jnp.array([1.0, 2.0])
_, Jv = jax.jvp(plane, (z0,), (v,))                 # forward mode: J v
_, vjp_fn = jax.vjp(plane, z0)                      # reverse mode: records once,
uJ, = vjp_fn(u)                                     # then u^T J per seed
J = jax.jacfwd(plane)(z0)                           # one JVP per column
print('J v   =', np.asarray(Jv).round(4))
print('u^T J =', np.asarray(uJ).round(4))
print('J     =', np.asarray(J, dtype=float).round(4).tolist())
```

:begin_tab:`mxnet`
MXNet's autograd has reverse mode only. The tab forms $\mathbf J$ from two VJPs,
one per row, and then multiplies it by $\mathbf v$; for this $2\times2$ map that
costs as much as two JVPs would.
:end_tab:

A JVP and a VJP each cost one sweep, and the derivative information they
carry along is a vector. The VJP also stores the forward intermediates, the tape. A dense
$m\times n$ Jacobian costs $\min(m,n)$ sweeps with the cheaper mode, plus $mn$
numbers of storage. Forming it is the right choice when $n$ is small, as for
this $2\times2$ map, or when its determinant is needed, as in the normalizing
flows that change probability densities by $|\det\mathbf J|$.

### Hessian--Vector Products
:label:`subsec_mdl-hvp`

A model with $n=10^6$ parameters has a Hessian of $10^{12}$ entries, 4 TB in
float32. Newton's method needs the Hessian only to solve
$\mathbf H\mathbf d=-\nabla\ell$ for its step $\mathbf d$, and Newton--CG, or
truncated Newton, solves that system by conjugate gradients, an iterative method
that touches $\mathbf H$ only through products $\mathbf H\mathbf v$
:cite:`Nocedal.Wright.2006,Martens.2010`. Can we compute $\mathbf H\mathbf v$
without $\mathbf H$?

The Hessian is the Jacobian of the gradient map $\nabla\ell$, so
$\mathbf H\mathbf v$ is a JVP of the gradient, the rate of change of the
gradient along $\mathbf v$:

$$
\mathbf H\mathbf v
   = \mathbf J_{\nabla \ell}\,\mathbf v
   = \left.\frac{d}{dt}\right|_{t=0}\nabla \ell(\mathbf x + t\mathbf v).
$$
:eqlabel:`eq_mdl-hvp`

A reverse sweep computes $\nabla\ell$, and a forward-mode pass through that
computation, with tangent $\mathbf v$, differentiates it along $\mathbf v$. This
composition is called *forward over reverse* :cite:`Pearlmutter.1994`. Its cost
is a small constant multiple of the cost of one gradient evaluation, whatever
$n$ is.

Log-sum-exp gives a test with a known answer. Its gradient is the softmax
$\mathbf p$, so its Hessian is the softmax Jacobian
:eqref:`eq_mdl-softmax-jacobian` (it is also the covariance matrix of a
one-hot vector $\mathbf e_J$ with $J$ drawn from $\mathbf p$), and

$$
\mathbf H=\operatorname{diag}(\mathbf p)-\mathbf p\mathbf p^\top,
\qquad
\mathbf H\mathbf v=\mathbf p\odot\mathbf v-\mathbf p\,(\mathbf p^\top\mathbf v),
$$
:eqlabel:`eq_mdl-ad-lse-hessian`

where $\odot$ multiplies entrywise. The product on the right needs only $3n$
multiplications and no $n\times n$ matrix. At $\mathbf x=(1,-0.5,2)$ and
$\mathbf v=(1,0.5,-1)$ we have $\mathbf p=(0.2537,0.0566,0.6897)$ and
$\mathbf p^\top\mathbf v=-0.4077$, so

$$
\mathbf H\mathbf v=(0.3571,\ 0.0514,\ -0.4085).
$$

The test is not vacuous. For the quadratic
$\tfrac12\mathbf x^\top\mathbf A\mathbf x$ the gradient at $\mathbf v$
equals $\mathbf H\mathbf v$, so a program that returned the wrong quantity would
still pass; here $\mathbf H$ depends on $\mathbf x$, and the gradient at
$\mathbf v$, $\operatorname{softmax}(\mathbf v)=(0.574,0.348,0.078)$, is far from
$\mathbf H\mathbf v$.

Where a framework has both modes, their composition is one line.

```{.python .input #mdl-matrix-calculus-autodiff-hessian-vector-products-one-order-up}
#@tab pytorch
x = torch.tensor([1.0, -0.5, 2.0])
v = torch.tensor([1.0, 0.5, -1.0])
lse = lambda x: torch.logsumexp(x, 0)
_, Hv = torch.func.jvp(torch.func.grad(lse), (x,), (v,))  # forward over reverse
p = torch.softmax(x, 0)
print('forward over reverse :', Hv.numpy().round(4))
print('p*v - p (p.v)        :', (p * v - p * (p @ v)).numpy().round(4))
```

```{.python .input #mdl-matrix-calculus-autodiff-hessian-vector-products-one-order-up}
#@tab mxnet
def grad_lse(x):                                    # reverse-mode gradient of lse
    x = x.copy()
    x.attach_grad()
    with autograd.record():
        ell = mnp.log(mnp.exp(x).sum())
    ell.backward()
    return x.grad

x = mnp.array([1.0, -0.5, 2.0], dtype=np.float64)
v = mnp.array([1.0, 0.5, -1.0], dtype=np.float64)
h = 1e-4                                            # directional derivative of
Hv = (grad_lse(x + h * v) - grad_lse(x - h * v)) / (2 * h)   # the gradient
p = mnp.exp(x) / mnp.exp(x).sum()
print('derivative of gradient:', Hv.asnumpy().round(4))
print('p*v - p (p.v)         :', (p * v - p * mnp.dot(p, v)).asnumpy().round(4))
```

```{.python .input #mdl-matrix-calculus-autodiff-hessian-vector-products-one-order-up}
#@tab tensorflow
x = tf.Variable([1.0, -0.5, 2.0])
v = tf.constant([1.0, 0.5, -1.0])
with tf.autodiff.ForwardAccumulator(x, v) as acc:   # forward over ...
    with tf.GradientTape() as tape:                 # ... reverse
        ell = tf.reduce_logsumexp(x)
    g = tape.gradient(ell, x)
Hv = acc.jvp(g)
p = tf.nn.softmax(x)
print('forward over reverse :', Hv.numpy().round(4))
print('p*v - p (p.v)        :', (p * v - p * tf.tensordot(p, v, 1)).numpy().round(4))
```

```{.python .input #mdl-matrix-calculus-autodiff-hessian-vector-products-one-order-up}
#@tab jax
x = jnp.array([1.0, -0.5, 2.0])
v = jnp.array([1.0, 0.5, -1.0])
_, Hv = jax.jvp(jax.grad(jax.nn.logsumexp), (x,), (v,))  # forward over reverse
p = jax.nn.softmax(x)
print('forward over reverse :', np.asarray(Hv).round(4))
print('p*v - p (p.v)        :', np.asarray(p * v - p * (p @ v)).round(4))
```

:begin_tab:`mxnet`
MXNet has no forward mode, and its second-order autograd does not cover `exp`.
The tab therefore evaluates :eqref:`eq_mdl-hvp` directly, as a central
difference of the reverse-mode gradient with step $h=10^{-4}$ in float64; it
agrees with the formula to about eight digits.
:end_tab:

The printed products agree with :eqref:`eq_mdl-ad-lse-hessian` to the printed
precision, and no $3\times3$ matrix of second derivatives was formed.
Exercise :exref:`ex_mdl-matrix-calculus-autodiff-hessian-vector-product`
builds the same product from `Dual` and `Var`, and compares forward over reverse
with reverse over reverse.

### Implicit Differentiation
:label:`subsec_mdl-implicit-diff`

Some outputs are defined by an equation and found by a solver: the fixed point
of an iteration, the equilibrium of a physical model, the minimizer of an inner
optimization problem. Taping twenty Newton iterations and running reverse mode
through them works, but it stores every iteration and differentiates the path
the solver happened to take. We want the derivative from the equation alone.

Start with one equation in one unknown, $g(x,\theta)=x^2-\theta=0$, solved by
$x^\star=\sqrt\theta$. Differentiate the identity
$x^\star(\theta)^2-\theta\equiv0$ in $\theta$:
$2x^\star\,dx^\star/d\theta-1=0$, so

$$
\frac{dx^\star}{d\theta}=\frac{1}{2x^\star}=\frac{1}{2\sqrt\theta},
$$

which is the derivative of $\sqrt\theta$. At $\theta=4$ it is $0.25$. The
formula for $x^\star$ was never used, only the equation and the value of the
solution.

The general statement is the *implicit function theorem* :cite:`Rudin.1976`,
which we use without proof. Let $\mathbf g(\mathbf x,\boldsymbol\theta)$ map
$\mathbb R^d\times\mathbb R^p$ to $\mathbb R^d$, be continuously differentiable
near a solution $(\mathbf x^\star,\boldsymbol\theta)$ of
$\mathbf g=\mathbf 0$, and have an invertible Jacobian
$\partial\mathbf g/\partial\mathbf x$ there. Then, near
$\boldsymbol\theta$, the solution is a differentiable function
$\mathbf x^\star(\boldsymbol\theta)$. Differentiating the identity
$\mathbf g(\mathbf x^\star(\boldsymbol\theta),\boldsymbol\theta)\equiv\mathbf 0$
with the chain rule gives
$\frac{\partial\mathbf g}{\partial\mathbf x}\frac{\partial\mathbf x^\star}{\partial\boldsymbol\theta}
+\frac{\partial\mathbf g}{\partial\boldsymbol\theta}=\mathbf 0$, hence

$$
\frac{\partial\mathbf x^\star}{\partial\boldsymbol\theta}
   = -\Bigl(\frac{\partial\mathbf g}{\partial\mathbf x}\Bigr)^{\!-1}
     \frac{\partial\mathbf g}{\partial\boldsymbol\theta}.
$$
:eqlabel:`eq_mdl-implicit-grad`

Both Jacobians on the right are evaluated at the solution only. The derivative
costs one linear solve with $\partial\mathbf g/\partial\mathbf x$ per
parameter, however many iterations the solver ran. In VJP form it costs one
solve for all parameters together:
$\mathbf u^\top\partial\mathbf x^\star/\partial\boldsymbol\theta
=-\mathbf a^\top\partial\mathbf g/\partial\boldsymbol\theta$, where $\mathbf a$
solves $(\partial\mathbf g/\partial\mathbf x)^\top\mathbf a=\mathbf u$.

The adjoint method for differentiating through ODE solvers rests on the same idea
(:numref:`sec_mdl-adjoint-method`). Deep equilibrium models
:cite:`Bai.Kolter.Koltun.2019` define a whole layer as a fixed point
$\mathbf z^\star=\mathbf f(\mathbf z^\star,\boldsymbol\theta)$ and train it
through :eqref:`eq_mdl-implicit-grad`. Tuning a hyperparameter through an inner
minimization is another instance.

We try :eqref:`eq_mdl-implicit-grad` on a nonlinear $2\times2$ system,
$g_1=x_1^2+\theta x_2-2=0$ and $g_2=x_1x_2+\theta-1=0$, solved by twenty Newton
steps, and compare it with a central difference of the solver's output.

```{.python .input #mdl-matrix-calculus-autodiff-differentiating-through-equations}
# Implicit gradient dx*/dtheta = -(dg/dx)^{-1} dg/dtheta on a 2x2 system,
# checked against a finite difference of the solver output
def g(x, th):                        # g(x, theta) = 0 defines x*(theta)
    return np.array([x[0]**2 + th * x[1] - 2, x[0] * x[1] + th - 1])

def dg_dx(x, th):                    # Jacobian in x, evaluated at (x, th)
    return np.array([[2 * x[0], th], [x[1], x[0]]])

def solve(th):                       # Newton iteration, from a fixed start
    x = np.array([1.0, 1.0])
    for _ in range(20):
        x = x - np.linalg.solve(dg_dx(x, th), g(x, th))
    return x

th, h = 0.3, 1e-6
xs = solve(th)
dg_dth = np.array([xs[1], 1.0])      # dg/dtheta at the solution
implicit = -np.linalg.solve(dg_dx(xs, th), dg_dth)
fd = (solve(th + h) - solve(th - h)) / (2 * h)
print('implicit-function gradient:', implicit.round(8))
print('finite-difference check   :', fd.round(8))
```

The two vectors agree to the precision of the finite difference, and the twenty
Newton iterations never entered the implicit formula. The derivative depends
only on $\mathbf g$ and the solution, so any converged solver gives the same
gradient.

The theorem has two limits, and both can be seen. First, it needs
$\partial\mathbf g/\partial\mathbf x$ to be invertible. For $x^2=\theta$ at
$\theta=0$, $\partial g/\partial x=2x^\star=0$ and the formula divides by zero,
matching the vertical tangent of $\sqrt\theta$ at $0$. Second, the formula
assumes that the solver has converged. What do the formula and the unrolled
solver return when Newton's method on $x^2=4$, started at $x=1$, stops after
$k=1,2,3$ steps? We evaluate the formula $1/(2\tilde x_k)$ at the unconverged
$\tilde x_k$, and differentiate the $k$-step solver itself by a central
difference in $\theta$.

```{.python .input #mdl-matrix-calculus-autodiff-implicit-differentiation-1}
def newton(theta, k, x=1.0):                        # k Newton steps on x^2 = theta
    for _ in range(k):
        x = x - (x * x - theta) / (2 * x)
    return x

theta, h = 4.0, 1e-6
for k in (1, 2, 3):
    x_k = newton(theta, k)
    unrolled = (newton(theta + h, k) - newton(theta - h, k)) / (2 * h)
    print(f'k={k}: x = {x_k:.5f}, formula {1 / (2 * x_k):.4f}, '
          f'unrolled {unrolled:.4f}, exact 0.25')
```

After one, two, and three steps the solver is at $2.5$, $2.05$, and $2.00061$.
The formula returns $0.2000$, $0.2439$, and $0.2499$, and the unrolled
derivative $0.5000$, $0.2900$, and $0.2509$, against the true $0.25$. Both are
wrong before convergence. The formula inherits the error of the solver, and the
unrolled derivative, which differentiates the path, is further off here. The
remedy is to converge first.

One problem stays open. Automatic differentiation evaluates the product
$\mathbf J_K\cdots\mathbf J_1$ to rounding error, but it cannot change what
the product is, and over many steps the product can shrink or grow
geometrically (:numref:`sec_mdl-matrix-calculus`). An exact gradient that is
$10^{-20}$ is of little use. The remedies for vanishing and exploding
gradients are choices of initialization and architecture, which begin in
:numref:`sec_numerical_stability`.

## Summary

The derivative of a program is the product of the Jacobians of its steps, and
automatic differentiation evaluates this product against a seed vector, so that
every step multiplies a matrix by a vector. Forward mode runs the program on
dual numbers and returns one Jacobian--vector product, one input direction, per
pass; reverse mode records a tape, replays it backward, and returns one
vector--Jacobian product, one output direction, per pass. The shape of the
Jacobian decides the mode. A loss has a single output, so one backward sweep
gives every parameter gradient at about three times the cost of the loss. On
$(\sum_jx_j^2)^2$ with $n=200$ inputs, the gradient took $802$ operations in
reverse mode against $80{,}200$ in forward mode. The tape's memory grows with
the length $T$ of the computation, and checkpointing reduces it to $2\sqrt T$
for one extra forward pass. Composing the two modes gives the
Hessian--vector product at a small constant multiple of the cost of one
gradient evaluation, and the implicit function theorem gives the derivative of
a solver's output from one linear solve at the solution, provided the solver
has converged.

## Exercises

1. [code] **Dual and complex steps.** Take
   $\mathbf f(x,y)=(x+\sin y,\ y+x^2/2)$ at $\mathbf x_0=(0.5,0.5)$ and the
   second column $\mathbf J\mathbf e_2=(\cos0.5,\ 1)$ of its Jacobian. The
   central difference
   $(\mathbf f(\mathbf x_0+t\mathbf e_2)-\mathbf f(\mathbf x_0-t\mathbf e_2))/(2t)$
   has an error of order $t^2$ from truncation plus an error of order $u/t$
   from rounding, where $u\approx10^{-16}$ is the float64 unit roundoff.
    1. Predict the best step and the smallest error, then check the prediction
       for $t=10^{-1},10^{-2},\ldots,10^{-12}$ (you should find about
       $2\times10^{-11}$ near $t=10^{-5}$). Compare with one pass of `Dual`
       seeded with $\mathbf e_2$. Why does no step size enter the dual
       computation?
    1. Repeat (a) for the first column $\mathbf J\mathbf e_1$, and explain why
       the central difference is exact there for every $t$, up to rounding.
       (Hint: $\mathbf f$ is a polynomial of degree two in $x$.)
    1. Replacing $\varepsilon$ by the imaginary unit $i$ gives the complex-step
       estimate $\operatorname{Im}\mathbf f(\mathbf x_0+it\mathbf e_2)/t$
       :cite:`Squire.Trapp.1998`, valid for an $\mathbf f$ that extends
       analytically to complex arguments, as this one does. Expand
       $\mathbf f(\mathbf x_0+it\mathbf e_2)$ to third order in $t$ and show that
       the error of the estimate is $O(t^2)$. Why does the estimate subtract no
       nearly equal numbers?
    1. Evaluate the complex-step estimate with NumPy's complex arithmetic for
       $t=10^{-1},10^{-2},\ldots,10^{-20}$ and compare its errors with those of
       (a). Which difference between $\varepsilon^2=0$ and $i^2=-1$ causes the
       $O(t^2)$ error of the complex step?

1. [code] **Compressed Jacobians.** A dense $m\times n$ Jacobian costs
   $\min(m,n)$ passes, and a sparse one can cost fewer. A JVP with the seed
   $\mathbf e_j+\mathbf e_k$ returns the sum of columns $j$ and $k$ of the
   Jacobian. If no row has nonzero entries in both columns, each nonzero entry
   of the sum belongs to exactly one of them.
    1. Show that one JVP recovers the whole Jacobian of an elementwise map
       $\boldsymbol\varphi:\mathbb R^n\to\mathbb R^n$. Which seed does it use?
       How many VJPs does reverse mode need for the same matrix?
    1. Let $f_i(\mathbf x)=x_{i-1}x_i+\sin(x_{i+1})$ for $i=1,\ldots,n$, where
       every term with an index outside $1,\ldots,n$ is dropped. Find three
       seeds from which all nonzero entries of $\mathbf J_{\mathbf f}$ can be
       read off. Implement $\mathbf f$ for `Dual` inputs with $n=10$, recover
       $\mathbf J_{\mathbf f}$ from the three JVPs, and compare it with the
       analytic Jacobian. Can the three JVPs share a single pass, with tangents
       that are arrays of length three?
    1. How many seeds does a banded Jacobian need whose nonzero entries satisfy
       $|i-j|\le b$, and how does this number depend on $n$? A one-dimensional
       convolution with kernel size $k$, without padding, has nonzero entries
       at $0\le j-i\le k-1$; how many seeds does it need?

1. [code] **Hyperparameter gradient.** Ridge regression with one penalty
   $\lambda_j>0$ per feature fits

   $$\mathbf w^\star(\boldsymbol\lambda)=\operatorname*{argmin}_{\mathbf w}\ \|\mathbf X\mathbf w-\mathbf y\|^2+\sum_{j=1}^d\lambda_jw_j^2$$

   on training data $(\mathbf X,\mathbf y)$, and its validation loss on
   data $(\mathbf X_{\text{v}},\mathbf y_{\text{v}})$ is
   $V(\boldsymbol\lambda)=\|\mathbf X_{\text{v}}\mathbf w^\star(\boldsymbol\lambda)-\mathbf y_{\text{v}}\|^2$.
    1. Show that $\mathbf w^\star$ solves
       $\mathbf g(\mathbf w,\boldsymbol\lambda)=(\mathbf X^\top\mathbf X+\operatorname{diag}(\boldsymbol\lambda))\,\mathbf w-\mathbf X^\top\mathbf y=\mathbf 0$,
       and use :eqref:`eq_mdl-implicit-grad` to derive
       $\partial\mathbf w^\star/\partial\boldsymbol\lambda$.
    1. Show that $\nabla_{\boldsymbol\lambda}V=-\mathbf a\odot\mathbf w^\star$,
       where $\mathbf a$ solves one linear system with the matrix
       $\mathbf X^\top\mathbf X+\operatorname{diag}(\boldsymbol\lambda)$,
       and give its right-hand side. How many linear solves does the
       forward route through $\partial\mathbf w^\star/\partial\boldsymbol\lambda$
       need for $d$ penalties, and how many does this adjoint route need?
    1. Implement $V$ and the adjoint gradient in NumPy for random data
       with $50$ training examples, $50$ validation examples, and $d=10$
       features. Check the gradient against central differences of $V$.

    *Adapted from Alan Edelman and Steven G. Johnson,
    [Matrix Calculus for Machine Learning and Beyond, problem set 2](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_pset2.pdf),
    MIT 18.S096, IAP 2023, Problem 1.*

1. [code] **Hessian--vector product.** ● The function
   $f(\mathbf x)=\bigl(\sum_j x_j^2\bigr)^2$ of
   :numref:`subsec_mdl-cost-asymmetry`, evaluated at the inputs `xs0` with
   $n=200$, is built from addition and multiplication alone, so both
   `Dual` and `Var` can evaluate it.
    1. Derive $\nabla f$ and the Hessian $\mathbf H$ of $f$, and write
       $\mathbf H\mathbf v$ in a form that never builds $\mathbf H$.
    1. Draw a random direction $\mathbf v$, wrap each input $x_j$ as
       `Var(Dual(x_j, v_j))`, and run `backprop`. Show that the value
       parts of the resulting adjoints form $\nabla f$ and that their
       $\varepsilon$-parts form $\mathbf H\mathbf v$, and compare both with
       (a) and with your framework's forward-over-reverse product applied to
       the same $f$. Which operations of the backward closures run in dual
       arithmetic, and why does this make the computation
       forward over reverse in the sense of :eqref:`eq_mdl-hvp`?
    1. Nest the tape in itself instead. Wrap each input as
       `Var(Var(x_j))` and run `backprop`; the adjoints are now `Var`
       objects. Form $\sum_j(\nabla f)_jv_j$ from them and run `backprop`
       on it to obtain $\mathbf H\mathbf v$ by reverse over reverse.
       Count the `Var` objects that each method creates, for example with
       a counter in `Var.__init__`, and compare both counts with that of
       one plain gradient. Which method records the backward pass, and
       what does that cost in memory?
   :exlabel:`ex_mdl-matrix-calculus-autodiff-hessian-vector-product`

1. [code] **Checkpointing on the tape.** ● Let
   $x_k=x_{k-1}-0.01\,x_{k-1}^2$ for $k=1,\ldots,T$ with $T=400$ and
   $x_0=1$, and differentiate $x_T$ with respect to $x_0$. `Var` has no
   subtraction, so record each step as `x + (-0.01) * x * x`.
    1. Record the chain on the tape with plain `backprop` and count the
       nodes that the tape holds when the backward sweep starts.
    1. `backprop` seeds the output adjoint with $1$. Generalize it to
       accept any seed. Show that recording a segment
       $x_i\mapsto x_j$ from a fresh leaf for $x_i$ and running the
       generalized `backprop` from $x_j$ with the seed
       $\partial x_T/\partial x_j$ returns $\partial x_T/\partial x_i$ in
       the adjoint of the leaf.
    1. Implement the schedule of :numref:`fig_mdl-cal-checkpointing` with
       segments of $s$ steps: a forward pass in plain floats that stores
       $x_0,x_s,x_{2s},\ldots$, followed by a backward pass that re-records
       one segment at a time, from the last to the first, and applies (b).
       Check that the result equals the gradient of (a).
    1. For $s\in\{1,5,10,20,100,400\}$, report the number of stored
       checkpoints, the largest tape in steps, and the number of evaluations of the
       step $x\mapsto x-0.01x^2$, the first forward pass included. Confirm
       the memory $T/s+s$ of :eqref:`eq_mdl-ad-checkpoint-memory`, that is
       $401,85,50,40,104,401$, and its minimizer $s=\sqrt T=20$. Repeat the
       sweep for $T=100$ and $T=1600$ over the divisors $s\le800$ of $T$ (a
       single segment of $1600$ steps exceeds Python's default recursion
       limit in `backprop`). How does the minimizing $s$ scale with $T$, and
       how many step evaluations does the best schedule need?

1. [extended] **Equilibrium layer.** ●● A deep equilibrium layer
   :cite:`Bai.Kolter.Koltun.2019` outputs the fixed point

   $$\mathbf z^\star=\tanh(\mathbf W\mathbf z^\star+\mathbf c),$$

   where $\mathbf c$ depends on the layer's input. Let the loss be
   $\ell=\tfrac12\|\mathbf z^\star-\mathbf y\|^2$, and write
   $\mathbf D=\operatorname{diag}\bigl(1-(z^\star_1)^2,\ldots,1-(z^\star_d)^2\bigr)$
   for the derivative of $\tanh$ at the fixed point.
    1. Show that the map $\mathbf z\mapsto\tanh(\mathbf W\mathbf z+\mathbf c)$
       is a contraction when $\|\mathbf W\|_2<1$, so that the fixed point
       exists and is unique, and that $\mathbf I-\mathbf D\mathbf W$ is
       then invertible. Use :eqref:`eq_mdl-implicit-grad` to derive
       $\nabla_{\mathbf W}\ell$, and show that it equals
       $(\mathbf D\mathbf a)(\mathbf z^\star)^\top$ for the adjoint
       $\mathbf a$ that solves
       $(\mathbf I-\mathbf D\mathbf W)^\top\mathbf a=\mathbf z^\star-\mathbf y$.
    1. Take $d=20$, a standard normal $\mathbf G\in\mathbb R^{d\times d}$,
       $\mathbf S=\mathbf G+\mathbf G^\top$, and
       $\mathbf W=\rho\,\mathbf S/\|\mathbf S\|_2$ for
       $\rho\in\{0.5,0.9,0.99\}$, and draw the entries of $\mathbf c$ and
       $\mathbf y$ independently from $\mathcal N(0,0.1^2)$. In NumPy,
       compute $\mathbf z^\star$ by fixed-point iteration and the gradient
       of (a) with one linear solve. Check a few of its entries against
       central differences of $\ell$.
    1. Unrolled differentiation runs $K$ iterations from
       $\mathbf z_0=\mathbf 0$ and backpropagates through all of them.
       Implement this backward sweep by hand, one VJP per iteration. Plot
       against $K$ the error of the unrolled gradient relative to the
       gradient of (b), together with the forward error
       $\|\mathbf z_K-\mathbf z^\star\|$. Compare the decay rate of both
       errors with the spectral radius of $\mathbf D\mathbf W$. How does
       the ratio of the two errors behave as $K$ grows, and why? How does
       the memory of each gradient scale with $K$?
    1. Replace the linear solve by $m$ steps of the iteration
       $\mathbf a\leftarrow(\mathbf z^\star-\mathbf y)+(\mathbf D\mathbf W)^\top\mathbf a$
       from $\mathbf a=\mathbf 0$, a truncated Neumann series that
       :citet:`Geng.Zhang.Bai.ea.2021` use as a *phantom gradient*. Show
       that this truncated adjoint gives the same gradient as
       backpropagation through $m$ iterations started at
       $\mathbf z^\star$. Plot the cosine between the truncated and the
       exact gradient against $m$ for each $\rho$. The gradient for $m=1$
       needs no linear solve and is the Jacobian-free backpropagation of
       :citet:`Fung.Heaton.Li.ea.2022`. Is it still a descent direction
       for $\ell$?

1. [code] **A tensor tape.** Extend `Var` from scalars to NumPy arrays, so
   that each node holds an array and each backward closure applies a VJP rule.
    1. State the backward rule of each primitive as a VJP. For
       $\mathbf Y=\mathbf X\mathbf W$, show $\bar{\mathbf X}=\bar{\mathbf Y}\mathbf W^\top$
       and $\bar{\mathbf W}=\mathbf X^\top\bar{\mathbf Y}$. For a broadcast sum
       $\mathbf Y=\mathbf X+\mathbf b$ with a row vector $\mathbf b$, show
       that the adjoint of $\mathbf b$ sums $\bar{\mathbf Y}$ over the rows.
       For $s=\sum_{ij}X_{ij}$, show that the adjoint of $\mathbf X$ is
       $\bar s\,\mathbf 1\mathbf 1^\top$, and for $\mathrm{relu}$ that it is
       $\bar{\mathbf Y}$ times a mask.
    1. Implement a class `TVar` with `__matmul__`, `__add__`, `sum`, and
       `relu`, and reuse `backprop` unchanged. Why does `backprop` not need to
       know that the values are arrays?
    1. Build the loss
       $\tfrac12\sum\bigl(\mathrm{relu}(\mathbf X\mathbf W_1)\mathbf W_2-\mathbf Y\bigr)^2$
       for random $\mathbf X\in\mathbb R^{32\times8}$,
       $\mathbf W_1\in\mathbb R^{8\times16}$,
       $\mathbf W_2\in\mathbb R^{16\times1}$, and $\mathbf Y\in\mathbb R^{32\times1}$
       (you will need a square or a product node), and check
       $\nabla_{\mathbf W_1}\ell$ and $\nabla_{\mathbf W_2}\ell$ against your
       framework's autograd to $10^{-6}$.
    1. Count the multiplications of the forward product $\mathbf X\mathbf W$
       and of its two backward products, and conclude that the backward pass
       of a linear layer costs twice its forward pass.

1. [code] **Mixed mode through a bottleneck.** The text multiplies
   :eqref:`eq_mdl-jacobian-product` from one end or the other. Splitting it
   in the middle can beat both. Let $\mathbf h(\mathbf x)=\tanh(\mathbf A\mathbf x)$
   with $\mathbf A\in\mathbb R^{2\times1000}$, let
   $\mathbf g(\mathbf z)=\tanh(\mathbf B\mathbf z)$ with
   $\mathbf B\in\mathbb R^{1000\times2}$, and let
   $\mathbf f=\mathbf g\circ\mathbf h:\mathbb R^{1000}\to\mathbb R^{1000}$.
    1. Show that $\operatorname{rank}\mathbf J_{\mathbf f}\le2$.
    1. Count the passes needed to form the full $1000\times1000$ Jacobian in
       pure forward mode and in pure reverse mode. In mixed mode, two VJPs
       through $\mathbf h$ give $\mathbf J_{\mathbf h}$ ($2\times1000$), two
       JVPs through $\mathbf g$ give $\mathbf J_{\mathbf g}$ ($1000\times2$),
       and one product forms $\mathbf J_{\mathbf g}\mathbf J_{\mathbf h}$.
    1. Implement all three with your framework's `jvp` and `vjp` (in MXNet,
       with VJPs and the analytic diagonal factors of $\tanh$) and compare
       with its full-Jacobian function. Count multiplications: about
       $5\times10^6$ for either pure mode (each pass multiplies by $\mathbf A$,
       by $\mathbf B$, and by the $\tanh'$ factors) against about
       $2\times10^6$ for mixed mode, and $1000$ runs of the program against four runs of half
       of it.
    1. For a longer chain with one layer of width $k$ in the middle, how many
       half-program passes does mixed mode need? How would you choose where to
       split the product $\mathbf J_K\cdots\mathbf J_1$ in general?

1. [code] **Forward gradients.** Can forward mode train at all? Let
   $\ell:\mathbb R^n\to\mathbb R$ and $\mathbf v\sim\mathcal N(\mathbf 0,\mathbf I_n)$.
   One forward pass seeded with $\mathbf v$ returns
   $\nabla\ell^\top\mathbf v$; define the estimate
   $\hat{\mathbf g}=(\nabla\ell^\top\mathbf v)\,\mathbf v$, which
   :citet:`Baydin.Pearlmutter.Syme.ea.2022` call a forward gradient.
    1. Show that $\mathbb{E}[\hat{\mathbf g}]=\nabla\ell$.
    1. Show that $\mathbb{E}\|\hat{\mathbf g}-\nabla\ell\|^2=(n+1)\|\nabla\ell\|^2$.
       (Hint: $\mathbb{E}[v_iv_jv_k^2]$ is $3$ for $i=j=k$, $1$ for
       $i=j\neq k$, and $0$ when $i\neq j$.)
    1. With `Dual`, and the $f$ and `xs0` of the cost cell ($n=200$), average
       $m$ independent estimates for $m=1,10,10^2,10^3,10^4$ and plot the
       relative error against $m$ on log--log axes. Show that
       the slope is $-\tfrac12$ and that about $2\times10^4$ passes reach a
       relative error of $10\%$.
    1. Compare that pass count with the $200$ coordinate passes and the one
       reverse sweep of the text. What does the random seed gain, and what
       does it cost?

1. [code] **Second derivatives in one forward pass.** Adjoin $\varepsilon$
   with $\varepsilon^3=0$ (but $\varepsilon^2\neq0$), and write
   $a+b\varepsilon+c\varepsilon^2$ as a triple $(a,b,c)$.
    1. Derive the product
       $(a,b,c)(a',b',c')=(aa',\ ab'+ba',\ ac'+bb'+ca')$, and show that for an
       analytic $\varphi$,
       $\varphi(a+b\varepsilon+c\varepsilon^2)=\varphi(a)+\varphi'(a)\,b\,\varepsilon+\bigl(\varphi'(a)\,c+\tfrac12\varphi''(a)\,b^2\bigr)\varepsilon^2$.
    1. Implement `Dual2` with `__add__`, `__mul__`, `sin`, and `exp`, and
       evaluate $f(x)=\sin(x^2)+e^x$ on $(1.3,1,0)$. Show that the result is
       $(f(1.3),\ f'(1.3),\ \tfrac12f''(1.3))$, and check
       $f''(1.3)\approx-3.2806$ against the hand formula
       $f''(x)=2\cos x^2-4x^2\sin x^2+e^x$.
    1. For $\ell:\mathbb R^n\to\mathbb R$ with inputs seeded by a direction
       $\mathbf v$, what does the $\varepsilon^2$-part return? (Answer:
       $\tfrac12\mathbf v^\top\mathbf H\mathbf v$, a number.) How many such
       passes give the vector $\mathbf H\mathbf v$ by polarization, against one
       forward-over-reverse pass?
    1. Write $\varepsilon$ as a $3\times3$ matrix with $\varepsilon^3=\mathbf 0$
       and verify the product of (a) by matrix multiplication.

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/)
:end_tab:

[Discussions](https://d2l.discourse.group/t/36827)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §23.3]{.kicker}

Automatic Differentiation<br>**forward mode · reverse mode · Jacobian products**
:::
:::

::: {.slide title="10⁸ Parameters, One Loss: One at a Time Takes 12 Days per Step"}
[The problem]{.kicker}

- A training step needs $\partial\ell/\partial\theta_i$ for all $10^8$
  parameters $\theta_i$ of a network.
- Nudge one parameter and rerun: $10^8$ runs $\times$ 10 ms $=10^6$ s,
  about 12 days, plus a step-size error.
- Backpropagation: all $10^8$ derivatives, exact up to rounding, for about
  three runs, 30 ms.

::: {.d2l-note .warn}
The derivative of a program is a product of Jacobians. The end we multiply it
from decides the cost.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- build forward mode (dual numbers) and reverse mode (a tape) in a few lines;
- trace a backward sweep by hand and derive a weight gradient from an adjoint;
- pick the mode from the shape of the Jacobian;
- compute $\mathbf H\mathbf v$ without $\mathbf H$, and differentiate through a solver.

**Running example:** log-sum-exp $\ell(\mathbf x)=\log\sum_je^{x_j}$ at
$\mathbf x=(1,-0.5,2)$, gradient $\mathbf p=\operatorname{softmax}(\mathbf x)=(0.254,0.057,0.690)$.
:::

::: {.slide title="Automatic Differentiation, after Matrix Calculus"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Single Variable Calculus | derivative, Taylor, corners |
| Multivariable Calculus | gradient, chain rule, Hessian |
| Matrix Calculus | Jacobians, a layer's backward rule |
| **Automatic Differentiation** | **forward and reverse mode** |
| Integral Calculus | areas, expectations, their gradients |
:::

::: {.col .narrow}
- **Uses:** Jacobians and their products, a layer's backward rule.
- **Skips:** nonsmooth functions, optimal accumulation orders, recursive checkpointing.
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[Forward Mode]{.dtitle}

[dual numbers carry the derivative with the value]{.dsub}
:::
:::

::: {.slide title="ε² = 0 Turns the Product Rule into Multiplication"}
[Forward mode]{.kicker}

Adjoin $\varepsilon\neq0$ with $\varepsilon^2=0$:

$$(3+\varepsilon)^2=9+6\varepsilon,\qquad (3+\varepsilon)^3=27+27\varepsilon .$$

$$(a+b\varepsilon)(c+d\varepsilon)=ac+(ad+bc)\,\varepsilon .$$

- $\varepsilon$-parts $6$, $27$: the derivatives of $x^2$, $x^3$ at $3$.
- $ad+bc$ is the product rule. Concretely,
  $\varepsilon=\left(\begin{smallmatrix}0&1\\0&0\end{smallmatrix}\right)$.
:::

::: {.slide title="Running a Program on x + ε Returns f(x) and f′(x)"}
[Forward mode]{.kicker}

$$f(x+\varepsilon)=f(x)+f'(x)\,\varepsilon$$

for any $f$ built from $+$, $\times$, and differentiable primitives, with
$\varphi(a+b\varepsilon)=\varphi(a)+\varphi'(a)\,b\,\varepsilon$. Sums, products,
and compositions preserve it. For $f(x)=\sin(x^2)+e^x$ at $1.3$:

@!matrix-calculus-autodiff-dual-eval
:::

::: {.slide title="One Forward Pass Delivers One Partial: Three Passes for Three Softmax Entries"}
[Forward mode]{.kicker}

Seed input $j$ with $1$, the rest with $0$; the $\varepsilon$-part is
$\partial\ell/\partial x_j=p_j$:

@!mdl-matrix-calculus-autodiff-a-forward-mode-engine-1

::: {.d2l-note}
A gradient over $n$ inputs costs $n$ passes: for $10^8$ parameters, $10^8$ runs.
:::
:::

::: {.slide title="An Identity Seed Returns the Whole Jacobian in One Pass, at n Times the Arithmetic"}
[Forward mode]{.kicker}

Plane map $\mathbf f(x,y)=(x+\sin y,\ y+x^2/2)$ at $(0.5,0.5)$. A seed
$\mathbf v$ returns the JVP $\mathbf J\mathbf v$; seeds $(1,0)$, $(0,1)$ as
arrays give all of $\mathbf J$:

@!mdl-matrix-calculus-autodiff-vector-tangents-and-jacobian-columns-1

One pass, but every tangent now has $n$ entries: passes and work differ.
:::

::: {.slide title="Check: Forward Mode on a Loss with 10⁸ Parameters"}
[Check]{.kicker}

How many forward passes does the gradient need? Or, in one pass, how much
arithmetic?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $10^8$ passes with scalar tangents, or one pass whose tangents have
$10^8$ entries: about $10^8$ times the work of one run either way.
:::
:::
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Reverse Mode]{.dtitle}

[record forward, replay backward]{.dsub}
:::
:::

::: {.slide title="One Product, Two Orders: Matrix Times Vector, Never Matrix Times Matrix"}
[Reverse mode]{.kicker}

$$\mathbf J\mathbf v=\mathbf J_K(\cdots(\mathbf J_1\mathbf v)),\qquad
\mathbf u^\top\mathbf J=((\mathbf u^\top\mathbf J_K)\cdots)\mathbf J_1$$

![](../img/mdl-cal-jacobian-shapes.svg){width=80% fig-align="center"}

A loss is one row: one VJP sweep seeded with $1$ gives the whole gradient.
:::

::: {.slide title="The Tape Records Forward and Replays Backward"}
[Reverse mode]{.kicker}

::: {.cols .vc}
::: {.col}
Adjoint $\bar v=\partial y/\partial v$. Each node keeps a closure:

```{.python #mdl-matrix-calculus-autodiff-exercises}
def back(g):        # c = a * b
    a.grad += b.value * g
    b.grad += a.value * g
```

Adjoints flow against execution: they exist only after the forward pass ends.
:::

::: {.col .fig}
![](../img/mdl-cal-fwd-vs-rev.svg){width=100%}
:::
:::
:::

::: {.slide title="Check: Fill in the Adjoints of y = (uv + u)² at u = 2, v = −3"}
[Check]{.kicker}

Forward: $t=uv=-6$, $r=t+u=-4$, $y=r\cdot r=16$. Sweep $y,r,t,u,v$.

::: {.fragment .fade-up}
![](../img/mdl-cal-tape-dag.svg){width=78% fig-align="center"}

@!mdl-matrix-calculus-autodiff-the-tape-1
:::
:::

::: {.slide title="One Outer Product per Layer Gives Every Weight Gradient"}
[Reverse mode]{.kicker}

Layer $\mathbf h_k=\mathbf W_k\mathbf h_{k-1}$; the sweep brings
$\bar{\mathbf h}_k=\partial\ell/\partial\mathbf h_k$. Perturb $\mathbf W_k$ alone:
$d\ell=\bar{\mathbf h}_k^\top d\mathbf W_k\,\mathbf h_{k-1}$, so

$$\nabla_{\mathbf W_k}\ell=\bar{\mathbf h}_k\mathbf h_{k-1}^\top,\qquad
\begin{pmatrix}1\\2\end{pmatrix}\begin{pmatrix}3&-1\end{pmatrix}=\begin{pmatrix}3&-1\\6&-2\end{pmatrix}.$$

- Costs one forward product; passing $\bar{\mathbf h}$ down costs another.
- Forward mode would need one pass per weight.
:::

::: {.slide title="Reverse Mode Does n/2 Times Less Work on (Σx²)²: 802 against 80,200 Operations"}
[Reverse mode]{.kicker}

$f(\mathbf x)=(\sum_jx_j^2)^2$, $n=200$: one evaluation is $2n+1=401$
operations. Both engines count:

@!mdl-matrix-calculus-autodiff-the-cost-asymmetry-measured

::: {.d2l-note}
Forward: $n(2n+1)$. Reverse: $2(2n+1)$, plus a tape of $401$ values in memory.
:::
:::

::: {.slide title="Checkpointing Cuts Tape Memory from T to 2√T"}
[Reverse mode]{.kicker}

![](../img/mdl-cal-checkpointing.svg){width=88% fig-align="center"}

- Segments of $s$ steps: memory $T/s+s$, smallest at $s=\sqrt T$.
- $T=400$: $40$ values instead of $401$, for one extra forward pass.
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[Composing Products]{.dtitle}

[curvature and solvers from JVPs and VJPs]{.dsub}
:::
:::

::: {.slide title="A Framework Exposes Two Products: Jv and uᵀJ"}
[Composing products]{.kicker}

Plane map at $(0.5,0.5)$, $\mathbf v=\mathbf u=(1,2)$,
$\mathbf J=\left(\begin{smallmatrix}1&0.878\\0.5&1\end{smallmatrix}\right)$:

@!mdl-matrix-calculus-autodiff-jacobian-vector-and-vector-jacobian-products

::: {.d2l-note}
Each costs one sweep. Form $\mathbf J$ only when it is small or its determinant
is needed.
:::
:::

::: {.slide title="A 10⁶-Parameter Hessian Has 10¹² Entries; Hv Costs a Few Gradients"}
[Composing products]{.kicker}

$\mathbf H\mathbf v=\frac{d}{dt}\nabla\ell(\mathbf x+t\mathbf v)\big|_{t=0}$: a JVP
of the gradient, forward over reverse. For log-sum-exp,
$\mathbf H=\operatorname{diag}(\mathbf p)-\mathbf p\mathbf p^\top$, $\mathbf v=(1,0.5,-1)$:

@!mdl-matrix-calculus-autodiff-hessian-vector-products-one-order-up

$\mathbf H\mathbf 1=\mathbf 0$: shifting all inputs leaves the gradient unchanged.
:::

::: {.slide title="The Solution of an Equation Has a Derivative without a Tape through the Solver"}
[Composing products]{.kicker}

$x^2=\theta$: $2x^\star\,dx^\star/d\theta=1$, so $dx^\star/d\theta=1/(2\sqrt\theta)$.

::: {.d2l-note .rule}
$\mathbf g(\mathbf x^\star,\boldsymbol\theta)=\mathbf 0$ $\Rightarrow$
$\dfrac{\partial\mathbf x^\star}{\partial\boldsymbol\theta}=-\Bigl(\dfrac{\partial\mathbf g}{\partial\mathbf x}\Bigr)^{-1}\dfrac{\partial\mathbf g}{\partial\boldsymbol\theta}$,
both at the solution.
:::

@!mdl-matrix-calculus-autodiff-differentiating-through-equations
:::

::: {.slide title="Check: Newton Stopped Early at x = 2.05 for x² = 4"}
[Check]{.kicker}

What do the formula $1/(2\tilde x)$ and the derivative of the two-step solver
return? The true value is $0.25$.

::: {.fragment .fade-up}
@!mdl-matrix-calculus-autodiff-implicit-differentiation-1

::: {.d2l-note}
**Answer.** $0.244$ and $0.290$. Both are wrong until the solver converges;
converge first.
:::
:::
:::

::: {.slide title="Two Products Compute Every Derivative a Training Loop Needs"}
[Wrap-up]{.kicker}

- Forward mode: one column per pass (dual numbers).
- Reverse mode: one row per pass (the tape); a loss costs one sweep, $802$
  against $80{,}200$ operations.
- Weights: $\nabla_{\mathbf W_k}\ell=\bar{\mathbf h}_k\mathbf h_{k-1}^\top$;
  tape memory $2\sqrt T$ with checkpoints.
- $\mathbf H\mathbf v$ by forward over reverse; a solver's output by one linear
  solve.

**Next:** integration, the chapter's second track.
:::
