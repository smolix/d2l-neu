# Matrix Calculus
:label:`sec_mdl-matrix-calculus`

<!-- Placeholder opening: the section writer replaces this paragraph. -->
This section derives the Jacobian of vector-valued maps, the chain rule as a
product of Jacobians, and the matrix-derivative identities that recur in deep
learning.

We use the following imports throughout this section.

```{.python .input #mdl-matrix-calculus-matrix-calculus}
#@tab mxnet
from mxnet import autograd, npx
from mxnet import np as mnp
import numpy as np
npx.set_np()
```

```{.python .input #mdl-matrix-calculus-matrix-calculus}
#@tab pytorch
import numpy as np
import torch
```

```{.python .input #mdl-matrix-calculus-matrix-calculus}
#@tab tensorflow
import numpy as np
import tensorflow as tf
```

```{.python .input #mdl-matrix-calculus-matrix-calculus}
#@tab jax
import numpy as np
import jax
from jax import numpy as jnp
```

## Derivatives of Vector- and Matrix-Valued Maps
:label:`subsec_mdl-jacobian`

A layer $\mathbf f:\mathbb R^n\to\mathbb R^m$ has one partial derivative for
each input--output pair. The *Jacobian* organizes these derivatives and
generalizes both the slope of
:numref:`sec_mdl-single_variable_calculus` and the gradient of
:numref:`sec_mdl-multivariable_calculus`.

We use numerator layout for a vector-valued Jacobian, so $\mathbf J_{\mathbf f}\in\mathbb R^{m\times n}$, and write the gradient of a scalar field as a column. A later reference box compares this convention with the transposed alternative.


### The Jacobian as the Best Linear Approximation

In one variable, the derivative is the slope of the best straight-line fit:
$f(x+\delta)\approx f(x)+f'(x)\,\delta$. The vector-valued generalization uses
the same local approximation. We say $\mathbf f$ is *differentiable* at $\mathbf x$ if there
is a matrix $\mathbf J\in\mathbb R^{m\times n}$ such that

$$
\mathbf f(\mathbf x+\boldsymbol\delta)
   = \mathbf f(\mathbf x) + \mathbf J\,\boldsymbol\delta + o(\|\boldsymbol\delta\|),
$$
:eqlabel:`eq_mdl-jacobian-linearization`

i.e. the error of the linear model $\mathbf f(\mathbf x)+\mathbf J\boldsymbol\delta$
shrinks *faster* than $\boldsymbol\delta$ itself as $\boldsymbol\delta\to\mathbf 0$.
That matrix, if it exists, is unique, is called the *Jacobian*
$\mathbf J_{\mathbf f}(\mathbf x)$, and its entries are exactly the partial
derivatives:

$$
[\mathbf J_{\mathbf f}]_{ij} = \frac{\partial f_i}{\partial x_j}.
$$
:eqlabel:`eq_mdl-jacobian-entries`

To see why, feed :eqref:`eq_mdl-jacobian-linearization` the perturbation
$\boldsymbol\delta=t\,\mathbf e_j$ (a small step along axis $j$) and read off
component $i$: dividing by $t$ and letting $t\to0$ leaves precisely
$\partial f_i/\partial x_j$ in the $(i,j)$ slot. So row $i$ of $\mathbf J$ collects
the partials of the $i$-th output, and column $j$ records how *all* outputs respond
to nudging input $j$. The Jacobian *is* the best local linear approximation; the
partial-derivative formula is a consequence, not the definition.

:numref:`fig_mdl-cal-jacobian-ellipse` makes the definition visible. Up close, a
differentiable map is *approximated by* a linear map: a small circle of inputs around
$\mathbf x_0$ maps nearly to the ellipse obtained from that circle under
$\mathbf J(\mathbf x_0)$, exactly the picture from
:numref:`sec_mdl-linear-maps` of what a matrix does to the
plane. The leftover bend is the $o(\|\boldsymbol\delta\|)$ remainder, which
vanishes faster than the circle shrinks.

![A differentiable map is locally approximated by its Jacobian. The nonlinear $\mathbf f(x,y)=(x+\sin y,\ y+x^2/2)$ carries a small circle and grid around $\mathbf x_0=(0.5,0.5)$ (left) into the output plane (right). The true image (blue) nearly matches the affine ellipse $\mathbf f(\mathbf x_0)+\mathbf J(\mathbf x_0)\boldsymbol\delta$ (orange); their residual is $o(\|\boldsymbol\delta\|)$ and therefore shrinks faster than the input radius.](../img/mdl-cal-jacobian-ellipse.svg)
:label:`fig_mdl-cal-jacobian-ellipse`

Two special cases recover everything we have already met. When $m=1$ (a scalar
field $f:\mathbb R^n\to\mathbb R$) the Jacobian is a *single row*, the *row*
gradient $\partial f/\partial\mathbf x = [\partial f/\partial x_1,\ldots,\partial
f/\partial x_n]$, and :eqref:`eq_mdl-jacobian-linearization` is the first-order
expansion $f(\mathbf x+\boldsymbol\delta)\approx f(\mathbf x)+\mathbf J\boldsymbol\delta$
of :numref:`sec_mdl-multivariable_calculus`. And the Jacobian of the *gradient field*
$\nabla f:\mathbb R^n\to\mathbb R^n$ is the Hessian
$\mathbf H=\mathbf J_{\nabla f}$ of :eqref:`eq_mdl-hess_def`: differentiating once
more turns the row gradient into the matrix of second partials, where the entry
$[\mathbf J_{\nabla f}]_{ij}=\partial^2 f/\partial x_j\partial x_i$ matches the
$(i,j)$ Hessian entry because mixed partials commute (Clairaut's theorem, proved
in :numref:`sec_mdl-multivariable_calculus`). One construction, three familiar
objects.

Let us pin this down on a concrete $\mathbb R^2\to\mathbb R^2$ map and check the
linear approximation against a finite difference, the numerical signature of a
correct derivative.

```{.python .input #matrix-calculus-autodiff-jacobian-finite-diff}
def f(v):                                 # f: R^2 -> R^2
    x, y = v
    return np.array([x**2 * y, np.sin(x + y)])

def J(v):                                 # the exact Jacobian, entry by entry
    x, y = v
    return np.array([[2 * x * y, x**2],
                     [np.cos(x + y), np.cos(x + y)]])

x0 = np.array([1.0, 0.5])
delta = np.array([1e-3, -2e-3])
linear = f(x0) + J(x0) @ delta            # first-order prediction
exact = f(x0 + delta)                     # true value
print('linear approx:', linear.round(6))
print('exact        :', exact.round(6))
print('error / |delta|:', np.linalg.norm(exact - linear) / np.linalg.norm(delta))
```

The error is tiny *relative to* $\|\boldsymbol\delta\|$, and shrinking it by a
further factor of ten in $\boldsymbol\delta$ shrinks the error by a factor of a
hundred, the $o(\|\boldsymbol\delta\|)$ signature of :eqref:`eq_mdl-jacobian-linearization`.

### The Chain Rule Is Jacobian Composition

The multivariate chain rule of :numref:`sec_mdl-multivariable_calculus` was a
sum over paths; in matrix form it is the *multiplication of Jacobians*.

**Proposition (chain rule).** *If $\mathbf f:\mathbb R^n\to\mathbb R^p$ is
differentiable at $\mathbf x$ and $\mathbf g:\mathbb R^p\to\mathbb R^m$ is
differentiable at $\mathbf f(\mathbf x)$, then $\mathbf g\circ\mathbf f$ is
differentiable at $\mathbf x$ and*

$$
\mathbf J_{\mathbf g\circ\mathbf f}(\mathbf x)
   = \mathbf J_{\mathbf g}\!\bigl(\mathbf f(\mathbf x)\bigr)\,\mathbf J_{\mathbf f}(\mathbf x).
$$
:eqlabel:`eq_mdl-jacobian-chain`

**Proof.** Write $\mathbf y=\mathbf f(\mathbf x)$ and let $\mathbf A=\mathbf
J_{\mathbf f}(\mathbf x)$, $\mathbf B=\mathbf J_{\mathbf g}(\mathbf y)$. Linearity
is the entire argument: substitute the linear model for $\mathbf f$ into the linear
model for $\mathbf g$. By :eqref:`eq_mdl-jacobian-linearization`,
$\mathbf f(\mathbf x+\boldsymbol\delta)=\mathbf y+\mathbf A\boldsymbol\delta
+o(\|\boldsymbol\delta\|)$. Calling that displacement
$\boldsymbol\eta=\mathbf A\boldsymbol\delta+o(\|\boldsymbol\delta\|)$ and feeding it
to $\mathbf g$,

$$
\mathbf g\bigl(\mathbf f(\mathbf x+\boldsymbol\delta)\bigr)
   = \mathbf g(\mathbf y) + \mathbf B\,\boldsymbol\eta + o(\|\boldsymbol\eta\|)
   = \mathbf g(\mathbf y) + \mathbf B\mathbf A\,\boldsymbol\delta + o(\|\boldsymbol\delta\|),
$$

because $\mathbf B$ applied to the $o(\|\boldsymbol\delta\|)$ remainder is again
$o(\|\boldsymbol\delta\|)$, and $\|\boldsymbol\eta\|=O(\|\boldsymbol\delta\|)$
(bounded by a constant multiple of $\|\boldsymbol\delta\|$) folds
$\mathbf g$'s own remainder into $o(\|\boldsymbol\delta\|)$ as well. The
coefficient of $\boldsymbol\delta$ is the best linear map, so by uniqueness it *is*
the Jacobian of the composite: $\mathbf J_{\mathbf g\circ\mathbf f}=\mathbf B\mathbf A$.
$\blacksquare$

The entry-wise chain rule follows from one entry of $\mathbf B\mathbf A$:
$[\mathbf B\mathbf A]_{ik}=\sum_j \partial g_i/\partial y_j\cdot\partial f_j/\partial x_k$
is exactly the "sum over intermediate variables" of
:numref:`sec_mdl-multivariable_calculus`. Matrix multiplication *is* the
bookkeeping of paths.

Iterating :eqref:`eq_mdl-jacobian-chain` is where deep learning enters. A network
of depth $L$ is a composition
$\mathbf f=\mathbf f_L\circ\cdots\circ\mathbf f_1$, so its end-to-end Jacobian is
one long matrix product,

$$
\mathbf J = \mathbf J_L\,\mathbf J_{L-1}\cdots\mathbf J_1,
$$
:eqlabel:`eq_mdl-jacobian-product`

each factor the local Jacobian of one layer. This is the same product-of-Jacobians
that governs backpropagation through time in recurrent networks
(:numref:`sec_mdl-eigendecompositions`).

Two shapes dominate those factors. A dense layer $\mathbf x\mapsto\mathbf W\mathbf x$ is its own best
linear approximation, so its Jacobian is just $\mathbf W$. An *elementwise*
activation $\boldsymbol\varphi(\mathbf x)=(\varphi(x_1),\ldots,\varphi(x_n))$
couples no pair of coordinates (output $i$ depends only on input $i$), so every
off-diagonal partial vanishes and its Jacobian is *diagonal*,
$\mathbf J_{\boldsymbol\varphi}(\mathbf x)
=\operatorname{diag}\bigl(\varphi'(x_1),\ldots,\varphi'(x_n)\bigr)$.
Multiplying by it is mere elementwise scaling; for ReLU, $\varphi'$ is a mask of
zeros and ones marking the active units. A standard deep network alternates the
two, so :eqref:`eq_mdl-jacobian-product` specializes to
$\mathbf J=\operatorname{diag}(\boldsymbol\varphi_L')\,\mathbf W_L\cdots
\operatorname{diag}(\boldsymbol\varphi_1')\,\mathbf W_1$: dense weight matrices
interleaved with cheap diagonal masks. That diagonal factor is the single most
common Jacobian in a real network, and it is why an activation's backward pass
costs only $O(n)$.

Matrix multiplication is
*associative*, so we may evaluate :eqref:`eq_mdl-jacobian-product` in any order.
Multiplying right-to-left propagates *inputs forward*; multiplying left-to-right
propagates *sensitivities backward*. Those two parenthesizations are precisely
forward-mode and reverse-mode automatic differentiation, and choosing between them
is just choosing the cheaper way to multiply a chain of matrices. (They are the
two *extreme* orderings; finding the cheapest ordering for a general computation
graph (*optimal Jacobian accumulation*) is NP-complete :cite:`Naumann.2008`,
so frameworks ship only the two ends; see :cite:`Baydin.Pearlmutter.Radul.ea.2018`
for a survey.)

### Layout Conventions: Numerator vs Denominator

One bookkeeping decision needs settling before we derive anything. For a scalar
field $f:\mathbb R^n\to\mathbb R$, do we write the derivative as a *row* (matching the Jacobian
:eqref:`eq_mdl-jacobian-entries`) or as a *column* (matching the input vector
$\mathbf x$)? Both are in use, and they are *transposes of each other*:

* **Numerator (Jacobian) layout.** The derivative has the shape of the
  *numerator*: $\partial\mathbf y/\partial\mathbf x$ is $m\times n$, so the scalar
  case is a row. This makes the chain rule :eqref:`eq_mdl-jacobian-chain` read
  left-to-right with no transposes, which is why it is the convention of choice for
  the Jacobian theory above and for MIT's *Matrix Calculus for Machine Learning
  and Beyond* course (18.S096, Edelman and Johnson).
* **Denominator (gradient) layout.** The derivative has the shape of the
  *denominator*: $\partial f/\partial\mathbf x$ is a *column* matching $\mathbf x$.
  This is the statistician's $\nabla_{\mathbf x} f$, the thing you subtract in
  gradient descent, and the convention of the derivations below.

Switching layouts simply transposes the result, so the gradient column and the
Jacobian row of a scalar field are related by

$$
\nabla_{\mathbf x} f = \Bigl(\frac{\partial f}{\partial\mathbf x}\Bigr)^{\!\top}.
$$
:eqlabel:`eq_mdl-grad-is-jacobian-transpose`

**This book's convention.** We use **numerator layout for Jacobians** of genuine
vector-valued maps (the chain rule then composes without transposes) and
**denominator layout for gradients**, writing $\nabla_{\mathbf x} f$ for the column you feed an optimizer.
Whenever the two could be confused we make the shape explicit. With that settled,
the identities in the next section are gradients, hence columns; if you ever land
on an expression whose shape does not match its denominator, you have a missing
transpose; use :eqref:`eq_mdl-grad-is-jacobian-transpose` to fix it. The Matrix
Cookbook :cite:`Petersen.Pedersen.ea.2008` tabulates hundreds of such identities;
the point of the next section is that you need to *memorize* almost none of them.

## Key Matrix-Derivative Identities
:label:`subsec_mdl-matrix-identities`

A reference table of matrix-derivative identities is long and forgettable. In
practice only a handful recur in deep learning, and each one yields to the same
two-step recipe of Parr and Howard's primer :cite:`Parr.Howard.2018`:
*differentiate one component with the ordinary scalar rules, then reassemble the
components into a matrix expression.* The same source offers a faster cross-check,
the **scalar-collapse heuristic**: a correct matrix identity must reduce to the
familiar single-variable result when every matrix is $1\times1$ (where products are
scalar products, sums are sums, and transposes do nothing), so you can often *guess*
the matrix form from the scalar one and fix the shapes by inserting transposes.
We derive four identities this way; all are gradients, hence columns (denominator
layout, :eqref:`eq_mdl-grad-is-jacobian-transpose`).

**The linear form $\nabla_{\mathbf x}\,\mathbf a^\top\mathbf x = \mathbf a$.** With
$f(\mathbf x)=\mathbf a^\top\mathbf x=\sum_i a_i x_i$, the $k$-th partial keeps only
the $i=k$ term, so $\partial f/\partial x_k = a_k$; stacking these gives
$\nabla_{\mathbf x} f=\mathbf a$. The scalar shadow is $\frac{d}{dx}(ax)=a$, exactly
the matrix answer with the transpose doing nothing. The one thing to
keep straight is *which* variable we differentiate by: it is $x_k$, not $a_k$.

**The quadratic form $\nabla_{\mathbf x}\,\mathbf x^\top\mathbf A\mathbf x =
(\mathbf A+\mathbf A^\top)\mathbf x$.** Write the form in Einstein notation
(:numref:`sec_mdl-linear-maps`) as
$\mathbf x^\top\mathbf A\mathbf x = x_i a_{ij} x_j$. The product rule on
$\partial/\partial x_k$ hits both $x$'s,

$$
\frac{\partial}{\partial x_k}\bigl(x_i a_{ij} x_j\bigr)
   = \delta_{ik}\,a_{ij}x_j + x_i a_{ij}\,\delta_{jk}
   = a_{kj}x_j + x_i a_{ik}
   = (a_{ki}+a_{ik})\,x_i,
$$

where the last step uses the fact that a summation (dummy) index can be renamed
freely: the first term is a sum over $j$, the second a sum over $i$, so renaming
$j\to i$ in the first term ($a_{kj}x_j = a_{ki}x_i$) puts both sums over the same
index $i$ and lets them combine. The bracket $a_{ki}+a_{ik}$ is the
$(k,i)$ entry of $\mathbf A+\mathbf A^\top$, so the $k$-th partial is the $k$-th
entry of $(\mathbf A+\mathbf A^\top)\mathbf x$, giving
$\nabla_{\mathbf x}\,\mathbf x^\top\mathbf A\mathbf x=(\mathbf A+\mathbf A^\top)\mathbf x$.
The scalar collapse is $\frac{d}{dx}(a x^2)=2ax=(a+a)x$; the matrix answer is the
same with $\mathbf A^\top$ standing in for the "other copy" of $a$. For *symmetric*
$\mathbf A$ this simplifies to the much-used $2\mathbf A\mathbf x$.

Derive, then verify: both identities should survive the same autograd audit we
will shortly apply to softmax, at a randomly drawn point.

```{.python .input #mdl-matrix-calculus-autodiff-a-few-key-identities-derived-not-tabulated-1}
#@tab pytorch
# Autograd audit of the two identities at a random point
torch.manual_seed(0)
a, A = torch.randn(3), torch.randn(3, 3)
x = torch.randn(3, requires_grad=True)
(a @ x).backward()
g = torch.autograd.grad(x @ A @ x, x)[0]
print('grad a^T x   equals a        :', torch.allclose(x.grad, a))
print('grad x^T A x equals (A+A^T)x :', torch.allclose(g, (A + A.T) @ x))
```

```{.python .input #mdl-matrix-calculus-autodiff-a-few-key-identities-derived-not-tabulated-1}
#@tab mxnet
# Autograd audit of the two identities at a random point
a, A = mnp.random.normal(size=3), mnp.random.normal(size=(3, 3))
x = mnp.random.normal(size=3)
x.attach_grad()
with autograd.record():
    lin = mnp.dot(a, x)
lin.backward()
ok1 = np.allclose(x.grad.asnumpy(), a.asnumpy())
with autograd.record():
    quad = mnp.dot(x, mnp.dot(A, x))
quad.backward()
ok2 = np.allclose(x.grad.asnumpy(), mnp.dot(A + A.T, x).asnumpy())
print('grad a^T x   equals a        :', ok1)
print('grad x^T A x equals (A+A^T)x :', ok2)
```

```{.python .input #mdl-matrix-calculus-autodiff-a-few-key-identities-derived-not-tabulated-1}
#@tab tensorflow
# Autograd audit of the two identities at a random point
tf.random.set_seed(0)
a, A = tf.random.normal([3]), tf.random.normal([3, 3])
x = tf.Variable(tf.random.normal([3]))
with tf.GradientTape(persistent=True) as tape:
    lin = tf.tensordot(a, x, 1)
    quad = tf.tensordot(x, tf.linalg.matvec(A, x), 1)
sym = tf.linalg.matvec(A + tf.transpose(A), x)
print('grad a^T x   equals a        :',
      bool(np.allclose(tape.gradient(lin, x), a)))
print('grad x^T A x equals (A+A^T)x :',
      bool(np.allclose(tape.gradient(quad, x), sym)))
del tape
```

```{.python .input #mdl-matrix-calculus-autodiff-a-few-key-identities-derived-not-tabulated-1}
#@tab jax
# Autograd audit of the two identities at a random point
k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
a, A = jax.random.normal(k1, (3,)), jax.random.normal(k2, (3, 3))
x = jax.random.normal(k3, (3,))
print('grad a^T x   equals a        :',
      bool(jnp.allclose(jax.grad(lambda x: a @ x)(x), a)))
print('grad x^T A x equals (A+A^T)x :',
      bool(jnp.allclose(jax.grad(lambda x: x @ A @ x)(x), (A + A.T) @ x)))
```

**The least-squares gradient $\nabla_{\mathbf W}\|\mathbf W\mathbf x-\mathbf y\|^2$.**
Let $\mathbf r=\mathbf W\mathbf x-\mathbf y$ be the residual, so the loss is
$\mathbf r^\top\mathbf r=\sum_i r_i^2$ with $r_i=\sum_j W_{ij}x_j-y_i$. Then

$$
\frac{\partial}{\partial W_{ab}}\sum_i r_i^2
   = 2\sum_i r_i\,\frac{\partial r_i}{\partial W_{ab}}
   = 2 r_a x_b = 2\,[\mathbf r\mathbf x^\top]_{ab},
$$

since $\partial r_i/\partial W_{ab}=\delta_{ia}x_b$. Assembling the $(a,b)$ entries
into a matrix the shape of $\mathbf W$,

$$
\nabla_{\mathbf W}\|\mathbf W\mathbf x-\mathbf y\|^2 = 2\,(\mathbf W\mathbf x-\mathbf y)\,\mathbf x^\top,
$$
:eqlabel:`eq_mdl-lstsq-grad`

a rank-one *outer product* of the residual with the input; this is the gradient a
single linear layer hands back during backpropagation. The companion gradient with
respect to the *input*, $\nabla_{\mathbf x}\|\mathbf W\mathbf x-\mathbf y\|^2
=2\mathbf W^\top(\mathbf W\mathbf x-\mathbf y)$, follows identically; note the
$\mathbf W^\top$. The same pattern, a residual hit by the transpose of whatever
multiplies the variable, persists when everything becomes a matrix. For the
factorization loss $\|\mathbf X-\mathbf U\mathbf V\|_F^2=\sum_{ij}R_{ij}^2$ with
residual $\mathbf R=\mathbf X-\mathbf U\mathbf V$, one entry of $\mathbf V$
appears in $R_{ij}$ through $\partial R_{ij}/\partial V_{ab}
=-U_{ia}\,\delta_{jb}$, so

$$
\frac{\partial}{\partial V_{ab}}\sum_{ij}R_{ij}^2
   = -2\sum_{i} U_{ia}R_{ib}
   = -2\,[\mathbf U^\top\mathbf R]_{ab},
\qquad\text{i.e.}\qquad
\nabla_{\mathbf V}\|\mathbf X-\mathbf U\mathbf V\|_F^2
   = -2\,\mathbf U^\top(\mathbf X-\mathbf U\mathbf V),
$$

exactly the $\mathbf W^\top$-shaped answer the scalar-collapse heuristic predicts
from $\frac{d}{dv}(x-uv)^2=-2u(x-uv)$.

**The softmax--cross-entropy Jacobian.** The derivative a classifier evaluates
most often also has the simplest form. Let $\mathbf p=\operatorname{softmax}(\mathbf z)$,
$p_i=e^{z_i}/\sum_k e^{z_k}$. Quotient rule on $\partial p_i/\partial z_j$ gives the
two cases $p_i(1-p_i)$ when $i=j$ and $-p_i p_j$ when $i\neq j$, which combine into

$$
\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij}-p_j),
\qquad
\mathbf J_{\operatorname{softmax}} = \operatorname{diag}(\mathbf p) - \mathbf p\mathbf p^\top.
$$
:eqlabel:`eq_mdl-softmax-jacobian`

Now compose with the cross-entropy loss $\ell=-\sum_i y_i\log p_i$ against a
one-hot target $\mathbf y$. The chain rule :eqref:`eq_mdl-jacobian-chain` multiplies
the loss's row gradient $\partial\ell/\partial\mathbf p$ by
:eqref:`eq_mdl-softmax-jacobian`, and the algebra collapses:

$$
\frac{\partial\ell}{\partial z_j}
   = -\sum_i \frac{y_i}{p_i}\,p_i(\delta_{ij}-p_j)
   = -y_j + p_j\sum_i y_i
   = p_j - y_j,
$$

using $\sum_i y_i=1$. The gradient of softmax--cross-entropy with respect to the
*logits* (the raw scores $\mathbf z$ that softmax turns into probabilities) is
just $\mathbf p-\mathbf y$: predicted probability minus truth. The softmax
Jacobian cancels against the cross-entropy gradient. Libraries also fuse the two
into a single primitive for a numerical reason: computing $\log p_i$ from an
already-evaluated softmax takes the logarithm of a value that may have underflowed
to zero, whereas the fused primitive computes
$\log p_i = z_i - \log\sum_k e^{z_k}$ directly through a stable log-sum-exp. Let
us confirm both :eqref:`eq_mdl-softmax-jacobian` and the cancellation against a
framework's autograd.

```{.python .input #matrix-calculus-autodiff-softmax-jacobian}
#@tab pytorch
z = torch.tensor([1.0, -0.5, 2.0], requires_grad=True)
p = torch.softmax(z, dim=0)
J = torch.diag(p) - torch.outer(p, p)          # our formula
J_ad = torch.autograd.functional.jacobian(lambda z: torch.softmax(z, 0), z)
y = torch.tensor([0.0, 1.0, 0.0])              # one-hot target (class 1)
loss = -(y * torch.log(torch.softmax(z, 0))).sum()
loss.backward()
print('softmax Jacobian matches autograd:', torch.allclose(J, J_ad, atol=1e-6))
print('d loss / d z :', z.grad.detach().numpy().round(6))
print('p - y        :', (p.detach() - y).numpy().round(6))
```

```{.python .input #matrix-calculus-autodiff-softmax-jacobian}
#@tab mxnet
z = mnp.array([1.0, -0.5, 2.0])
z.attach_grad()
y = mnp.array([0.0, 1.0, 0.0])                 # one-hot target (class 1)
rows = []
for i in range(3):                             # one VJP per output: row i of J
    with autograd.record():
        p = npx.softmax(z)
    p.backward(mnp.eye(3)[i])                  # seed e_i; z.grad is row i
    rows.append(z.grad.asnumpy().copy())
J_ad = np.stack(rows)                          # framework-autograd Jacobian
p = npx.softmax(z).asnumpy()
J = np.diag(p) - np.outer(p, p)                # our formula
with autograd.record():
    loss = -(y * mnp.log(npx.softmax(z))).sum()
loss.backward()
print('softmax Jacobian matches autograd:', np.allclose(J, J_ad, atol=1e-6))
print('d loss / d z :', z.grad.asnumpy().round(6))
print('p - y        :', (p - y.asnumpy()).round(6))
```

```{.python .input #matrix-calculus-autodiff-softmax-jacobian}
#@tab tensorflow
z = tf.Variable([1.0, -0.5, 2.0])
y = tf.constant([0.0, 1.0, 0.0])               # one-hot target (class 1)
with tf.GradientTape(persistent=True) as tape:
    p = tf.nn.softmax(z)
    loss = -tf.reduce_sum(y * tf.math.log(p))
J_ad = tape.jacobian(p, z)                     # framework-autograd Jacobian
grad = tape.gradient(loss, z)
del tape
J = tf.linalg.diag(p) - tf.tensordot(p, p, axes=0)   # our formula
print('softmax Jacobian matches autograd:', bool(np.allclose(J, J_ad, atol=1e-6)))
print('d loss / d z :', grad.numpy().round(6))
print('p - y        :', (p - y).numpy().round(6))
```

```{.python .input #matrix-calculus-autodiff-softmax-jacobian}
#@tab jax
z = jnp.array([1.0, -0.5, 2.0])
p = jax.nn.softmax(z)
J = jnp.diag(p) - jnp.outer(p, p)              # our formula
J_ad = jax.jacobian(jax.nn.softmax)(z)
y = jnp.array([0.0, 1.0, 0.0])                 # one-hot target (class 1)
loss = lambda z: -(y * jnp.log(jax.nn.softmax(z))).sum()
print('softmax Jacobian matches autograd:', bool(jnp.allclose(J, J_ad, atol=1e-6)))
print('d loss / d z :', np.asarray(jax.grad(loss)(z)).round(6))
print('p - y        :', np.asarray(p - y).round(6))
```

The logit gradient equals $\mathbf p-\mathbf y$, no matter how the
intermediate Jacobian looks. Four identities, one method: differentiate a single
component, then read the matrix back off the indices, sanity-checking against the
scalar collapse.

## Summary

* The **Jacobian** $\mathbf J_{\mathbf f}\in\mathbb R^{m\times n}$ is the best local
  linear approximation of a vector-valued $\mathbf f$:
  $\mathbf f(\mathbf x+\boldsymbol\delta)\approx\mathbf f(\mathbf x)+\mathbf J\boldsymbol\delta$,
  with $[\mathbf J]_{ij}=\partial f_i/\partial x_j$. The row gradient ($m=1$) and the
  Hessian (Jacobian of $\nabla f$) are special cases.
* The chain rule in matrix form is **Jacobian composition**
  $\mathbf J_{\mathbf g\circ\mathbf f}=\mathbf J_{\mathbf g}\mathbf J_{\mathbf f}$, so
  a depth-$L$ network's derivative is the matrix product
  $\mathbf J_L\cdots\mathbf J_1$. Different associations of this product give
  forward- and reverse-mode AD.
* **Layout conventions** (numerator vs denominator) differ by a transpose; we use
  numerator layout for Jacobians and denominator (column) layout for gradients.
* The recurring DL identities all follow from differentiating one component and
  reassembling, not from a table: $\nabla\,\mathbf a^\top\mathbf x=\mathbf a$,
  $\nabla\,\mathbf x^\top\mathbf A\mathbf x=(\mathbf A+\mathbf A^\top)\mathbf x$,
  the rank-one $\nabla_{\mathbf W}\|\mathbf W\mathbf x-\mathbf y\|^2=2(\mathbf W\mathbf x-\mathbf y)\mathbf x^\top$,
  and the softmax--cross-entropy logit gradient $\mathbf p-\mathbf y$.

## Exercises

1. **Jacobian of a ReLU network.** Let
   $\mathbf g(\mathbf x)=\mathbf W_2\,\boldsymbol\varphi(\mathbf W_1\mathbf x+\mathbf b)$
   with $\mathbf W_1\in\mathbb R^{h\times n}$, $\mathbf b\in\mathbb R^h$,
   $\mathbf W_2\in\mathbb R^{m\times h}$, and the elementwise ReLU
   $\boldsymbol\varphi$. Assume that no entry of the pre-activation
   $\mathbf W_1\mathbf x+\mathbf b$ is zero. By
   :eqref:`eq_mdl-jacobian-chain`,
   $\mathbf J_{\mathbf g}(\mathbf x)=\mathbf W_2\mathbf M(\mathbf x)\mathbf W_1$,
   where the diagonal matrix $\mathbf M(\mathbf x)$ has the entry $1$ for
   each active hidden unit and $0$ for each inactive one.
    1. Show that
       $\mathbf g(\mathbf x+\boldsymbol\delta)=\mathbf g(\mathbf x)+\mathbf J_{\mathbf g}(\mathbf x)\,\boldsymbol\delta$
       holds exactly for every sufficiently small $\boldsymbol\delta$, so
       that the remainder in :eqref:`eq_mdl-jacobian-linearization`
       vanishes identically. For $n=m=2$, how does the image of a small
       circle under $\mathbf g$ differ from the one in
       :numref:`fig_mdl-cal-jacobian-ellipse`?
    1. Let $k$ hidden units be active at $\mathbf x$. Show that
       $\operatorname{rank}\mathbf J_{\mathbf g}(\mathbf x)\le k$. For
       $n=m=2$ and $k=1$, what is the image of a small circle around
       $\mathbf x$?
    1. Describe the set of inputs at which some pre-activation is zero.
       What happens to $\mathbf J_{\mathbf g}$ when $\mathbf x$ crosses
       this set?

1. **Softmax vector--Jacobian product.** Reverse mode never forms the
   softmax Jacobian :eqref:`eq_mdl-softmax-jacobian`. It maps the adjoint
   $\mathbf u=\nabla_{\mathbf p}\ell$ of the probabilities to the adjoint
   $\mathbf J_{\operatorname{softmax}}^\top\mathbf u$ of the logits.
    1. Show that
       $\mathbf J_{\operatorname{softmax}}^\top\mathbf u=\mathbf p\odot\bigl(\mathbf u-(\mathbf p^\top\mathbf u)\,\mathbf 1\bigr)$,
       where $\odot$ is the elementwise product. For $n$ classes, count
       the operations of this formula and those of forming
       $\mathbf J_{\operatorname{softmax}}$ and multiplying it by
       $\mathbf u$.
    1. Self-attention applies softmax to each row of an $n\times n$ score
       matrix $\mathbf S$ and produces the weights $\mathbf P$. How many
       entries does the Jacobian of $\mathbf P$ with respect to
       $\mathbf S$ have, and how many of them can be nonzero? For
       $n=4096$ tokens in float32, compare the memory of the full
       Jacobian, of its nonzero entries, and of the adjoint of
       $\mathbf S$ that the formula of (a), applied row by row, returns.

1. [code] **Cheapest chain order.** ● By :eqref:`eq_mdl-jacobian-product`
   the Jacobian of a chain of $L$ maps is $\mathbf J_L\cdots\mathbf J_1$
   with $\mathbf J_k\in\mathbb R^{d_k\times d_{k-1}}$. Multiplying a
   $p\times q$ matrix by a $q\times r$ matrix costs $pqr$ multiplications.
    1. Give the costs of the forward order, which multiplies from the
       right, and of the reverse order, which multiplies from the left, as
       functions of $d_0,\ldots,d_L$. Evaluate both for
       $(d_0,\ldots,d_4)=(1000,10,1000,10,1000)$, a map whose layers
       alternate between wide and narrow.
    1. Write a dynamic program over the sub-chains
       $\mathbf J_j\cdots\mathbf J_i$ that finds the cheapest
       parenthesization, and run it on the dimensions of (a). Which order
       does it choose, and how much cheaper is it than the better of the
       two orders of (a)?
    1. For a scalar output, $d_L=1$, compare the optimum with the reverse
       order on a few thousand random dimension vectors, with $L$ between
       $2$ and $6$ and each of $d_0,\ldots,d_{L-1}$ drawn uniformly from
       $\{1,\ldots,10\}$. Characterize the vectors for which some other
       order is cheaper.

1. **Log-determinant gradient.** ● A normalizing flow
   :cite:`rezende2015variational` passes a random vector $\mathbf X$
   through invertible layers. For one layer $\mathbf Y=\mathbf g(\mathbf X)$,
   the log-density of $\mathbf Y$ at $\mathbf g(\mathbf x)$ is
   $\log p_{\mathbf X}(\mathbf x)-\log|\det\mathbf J_{\mathbf g}(\mathbf x)|$,
   a consequence of the change of variables :eqref:`eq_mdl-change_var_nd`
   of :numref:`sec_mdl-integral_calculus`. For a linear
   layer $\mathbf x\mapsto\mathbf A\mathbf x$ the correction
   $-\log|\det\mathbf J_{\mathbf g}|$ is $-\log|\det\mathbf A|$, so maximum-likelihood training differentiates
   $\log|\det\mathbf A|$ with respect to $\mathbf A$. For a real
   invertible matrix $\mathbf A$ the gradient is
   $\nabla_{\mathbf A}\log|\det\mathbf A|=\mathbf A^{-\top}$.
    1. Derive the identity entry by entry. With
       $\mathbf E=\mathbf e_a\mathbf e_b^\top$, write
       $\det(\mathbf A+t\mathbf E)=\det(\mathbf A)\det(\mathbf I+t\mathbf A^{-1}\mathbf E)$,
       expand $\det(\mathbf I+t\mathbf M)$ to first order in $t$, and read
       off the $(a,b)$ partial derivative. Why does the sign of
       $\det\mathbf A$ not matter?
    1. Check the identity on a random invertible $3\times3$ matrix by
       central differences of `np.linalg.slogdet`.
    1. How does the norm of the gradient behave as $\mathbf A$ approaches
       a singular matrix? What does this imply for a flow layer whose
       weight matrix is nearly singular?

1. **Gradient of the spectral norm.** ● Spectral normalization
   :cite:`Miyato.Kataoka.Koyama.ea.2018` divides each weight matrix
   $\mathbf W$ of a network by its largest singular value
   $\sigma_1(\mathbf W)$, so training differentiates $\sigma_1$.
    1. Let $\mathbf S$ be a real symmetric matrix with a simple eigenvalue
       $\lambda$ and unit eigenvector $\mathbf q$, and perturb it to
       $\mathbf S+t\mathbf E$ with $\mathbf E$ symmetric. The implicit
       function theorem of :numref:`subsec_mdl-implicit-diff`, applied to
       $(\mathbf S+t\mathbf E)\mathbf q=\lambda\mathbf q$ and
       $\mathbf q^\top\mathbf q=1$, makes $\lambda(t)$ and $\mathbf q(t)$
       differentiable. Differentiate both equations at $t=0$ to show the
       Hellmann--Feynman theorem
       $\lambda'(0)=\mathbf q^\top\mathbf E\,\mathbf q$, and give
       $\nabla_{\mathbf S}\lambda$ for the inner product
       $\operatorname{tr}(\mathbf A^\top\mathbf B)$.
    1. Apply (a) to
       $\sigma_1(\mathbf W)^2=\lambda_{\max}(\mathbf W^\top\mathbf W)$ to
       show that $\nabla_{\mathbf W}\sigma_1=\mathbf u_1\mathbf v_1^\top$
       when $\sigma_1$ is simple, where $\mathbf u_1$ and $\mathbf v_1$
       are the top left and right singular vectors. Check the formula by
       central differences on a random $5\times3$ matrix in NumPy.
    1. Compute $\sigma_1(\mathbf I+t\operatorname{diag}(1,-1))$ for small
       $t$ of both signs. Is $\sigma_1$ differentiable at the $2\times2$
       identity, and which hypothesis of (a) fails there?

    *Adapted from Alan Edelman and Steven G. Johnson,
    [Matrix Calculus for Machine Learning and Beyond, problem set 2](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_pset2.pdf),
    MIT 18.S096, IAP 2023, Problem 3.*

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

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §23.3]{.kicker}

Derivative computation for neural networks<br>**Jacobians · the chain rule · matrix-derivative identities**.
:::
:::

::: {.slide title="From Scalar Derivatives to Jacobians"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
A network layer maps **vectors to vectors**, $\mathbf f:\mathbb R^n\to\mathbb R^m$, so its derivative is a *matrix*: the **Jacobian**.

- It is the best local **linear** map: up close, $\mathbf f$ *is* a matrix.
- The chain rule becomes **Jacobian multiplication**, one long product per network.
- That product is **associative**, and the cheap order to evaluate it *is* backpropagation.
- We then build both flavours of autodiff in a few dozen lines of Python.
:::

::: {.col .fig}
![](../img/mdl-cal-jacobian-ellipse.svg){width=100%}
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[The Jacobian]{.dtitle}

[the best local linear map]{.dsub}
:::
:::

::: {.slide title="Vectors in, vectors out"}
[The Jacobian]{.kicker}

::: {.cols .vc}
::: {.col}
$\mathbf f$ is differentiable at $\mathbf x$ if a matrix $\mathbf J$ tracks every first-order change:

$$\mathbf f(\mathbf x+\boldsymbol\delta)=\mathbf f(\mathbf x)+\mathbf J\boldsymbol\delta+o(\|\boldsymbol\delta\|),
\qquad [\mathbf J]_{ij}=\frac{\partial f_i}{\partial x_j}.$$

Row $i$ collects the partials of output $i$; column $j$ says how every output reacts to nudging input $j$.
:::

::: {.col .fig}
![](../img/mdl-cal-jacobian-ellipse.svg){width=100%}
:::
:::

A small circle of inputs maps to an approximate **ellipse**, its image under $\mathbf J$. The leftover bend is the $o(\|\boldsymbol\delta\|)$ remainder.
:::

::: {.slide title="Jacobians, Gradients, and Hessians"}
[The Jacobian]{.kicker}

The Jacobian unifies the derivative objects introduced so far:

::: {.d2l-note .rule}
**Scalar map** ($m=1$): $\mathbf J$ is one *row*, the gradient $\partial f/\partial\mathbf x$.

**Vector map:** the full $m\times n$ matrix of partials.

**Hessian:** the Jacobian of the gradient field, $\mathbf H=\mathbf J_{\nabla f}$.
:::

One construction recovers the slope, the gradient, and the matrix of second derivatives.
:::

::: {.slide title="The numerical signature of a derivative"}
[The Jacobian]{.kicker}

The linear model must beat its own error: halve $\boldsymbol\delta$ and the error should fall *fourfold*. Check $\mathbf f(x,y)=(x^2y,\ \sin(x+y))$ against a finite difference, prediction vs truth:

@!matrix-calculus-autodiff-jacobian-finite-diff

The error relative to $\|\boldsymbol\delta\|$ is tiny, and shrinking $\boldsymbol\delta$ tenfold shrinks it a hundredfold: the $o(\|\boldsymbol\delta\|)$ fingerprint.
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[The chain rule is Jacobian composition]{.dtitle}

[multiplication of matrices, in any order]{.dsub}
:::
:::

::: {.slide title="Composition multiplies Jacobians"}
[Chain rule]{.kicker}

The multivariate chain rule (a sum over paths) is, in matrix form, just a product of Jacobians:

::: {.d2l-note .rule}
$$\mathbf J_{\mathbf g\circ\mathbf f}=\mathbf J_{\mathbf g}\,\mathbf J_{\mathbf f},
\qquad
\mathbf J = \mathbf J_L\,\mathbf J_{L-1}\cdots\mathbf J_1.$$
:::

A depth-$L$ network is a composition $\mathbf f_L\circ\cdots\circ\mathbf f_1$, so its end-to-end derivative is **one long matrix product**, one local Jacobian per layer.
:::

::: {.slide title="Two shapes dominate the product"}
[Chain rule]{.kicker}

Real networks alternate just two kinds of factor:

::: {.cols}
::: {.col}
::: {.d2l-note .rule}
**Dense** $\mathbf x\mapsto\mathbf W\mathbf x$

Its own best linear map, so $\mathbf J=\mathbf W$.
:::
:::

::: {.col}
::: {.d2l-note .rule}
**Elementwise** $\boldsymbol\varphi$

Couples no coordinates, so $\mathbf J=\operatorname{diag}(\varphi')$.
:::
:::
:::

So $\mathbf J=\operatorname{diag}(\boldsymbol\varphi_L')\,\mathbf W_L\cdots\operatorname{diag}(\boldsymbol\varphi_1')\,\mathbf W_1$: weights interleaved with cheap diagonal masks. For ReLU that mask is just zeros and ones, which is why a backward pass through an activation costs $O(n)$.
:::

::: {.slide title="Associativity: two brackets, two modes"}
[Chain rule]{.kicker}

Matrix multiplication is **associative**, so the chain $\mathbf J_L\cdots\mathbf J_1$ may be bracketed in any order:

. . .

- Multiply **right-to-left** $\Rightarrow$ propagate *inputs forward*. **Forward mode.**
- Multiply **left-to-right** $\Rightarrow$ propagate *sensitivities backward*. **Reverse mode.**

::: {.d2l-note}
For a dense derivative product, the two modes are alternative parenthesizations; graph reuse, sparsity, batching, and memory can change the practical cost.
:::
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[Identities, derived not tabulated]{.dtitle}

[differentiate one component, reassemble]{.dsub}
:::
:::

::: {.slide title="Four identities, one method"}
[Identities]{.kicker}

Differentiate one component, reassemble, then sanity-check the $1\times1$ **scalar collapse**:

::: {.d2l-note .rule}
$\nabla_{\mathbf x}\,\mathbf a^\top\mathbf x = \mathbf a$

$\nabla_{\mathbf x}\,\mathbf x^\top\mathbf A\mathbf x = (\mathbf A+\mathbf A^\top)\mathbf x$

$\nabla_{\mathbf W}\|\mathbf W\mathbf x-\mathbf y\|^2 = 2(\mathbf W\mathbf x-\mathbf y)\mathbf x^\top$

softmax + cross-entropy, w.r.t. the logits $=\mathbf p-\mathbf y$
:::

No table needed: the least-squares gradient is the rank-one outer product a linear layer returns in backprop.

. . .

Derive, then verify: autograd audits the first two at a random point:

@!mdl-matrix-calculus-autodiff-a-few-key-identities-derived-not-tabulated-1
:::

::: {.slide title="The cancellation that fuses softmax with cross-entropy"}
[Identities]{.kicker}

The softmax Jacobian $\operatorname{diag}(\mathbf p)-\mathbf p\mathbf p^\top$, composed with cross-entropy, collapses to $\mathbf p-\mathbf y$ whatever the Jacobian; the fused softmax--cross-entropy primitive also avoids $\log$ of an underflowed probability via log-sum-exp. Verify against autograd:

@matrix-calculus-autodiff-softmax-jacobian@pytorch
:::

::: {.slide title="The Jacobian and the chain rule"}
[Wrap-up]{.kicker}

- The **Jacobian** is the best local linear map; the gradient ($m=1$) and the Hessian are special cases.
- The chain rule is **Jacobian composition**; a deep net is one long matrix product.
- Its **associativity** is what yields the two autodiff modes.
:::
