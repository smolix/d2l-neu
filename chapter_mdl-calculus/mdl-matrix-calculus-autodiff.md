# Matrix Calculus and Automatic Differentiation
:label:`sec_mdl-matrix-calculus-autodiff`

The previous two sections built differentiation up from a single weight
(:numref:`sec_mdl-single_variable_calculus`) to the gradient of a scalar loss over
many weights (:numref:`sec_mdl-multivariable_calculus`). Real network layers,
however, map *vectors to vectors*, and their parameters are *matrices*, so the
natural object of study is the derivative of a vector-valued map---the
*Jacobian*---and the natural question is why `loss.backward()` is cheap. This
section answers both. Its punchline is that **backpropagation is reverse-mode
automatic differentiation: a sequence of vector--Jacobian products**, and that the
choice between forward- and reverse-mode AD is dictated entirely by the *shape* of
the Jacobian you are after. Along the way we re-derive the handful of matrix
identities that actually recur in deep learning, and we build---in a few dozen
lines of Python---both flavours of automatic differentiation from scratch, so that
the framework's autograd stops being magic.

We load the per-framework library so the verification cells have `d2l` and `np` in
scope. The two automatic-differentiation engines we build below are pure
Python/NumPy and run under every framework; only the cells that *check* them
against a framework's autograd branch per framework.

```{.python .input #matrix-calculus-autodiff-imports}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import numpy as np
```

```{.python .input #matrix-calculus-autodiff-imports}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
import torch
```

```{.python .input #matrix-calculus-autodiff-imports}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input #matrix-calculus-autodiff-imports}
#@tab jax
%matplotlib inline
from d2l import jax as d2l
import numpy as np
import jax
from jax import numpy as jnp
```

## Derivatives of Vector- and Matrix-Valued Maps
:label:`subsec_mdl-jacobian`

A layer $\mathbf f:\mathbb R^n\to\mathbb R^m$ does not have "a derivative" but a
whole matrix of them. The single object that organizes them---and generalizes both
the slope of :numref:`sec_mdl-single_variable_calculus` and the gradient of
:numref:`sec_mdl-multivariable_calculus` in one stroke---is the *Jacobian*.

### The Jacobian as the Best Linear Approximation

In one variable, the derivative is the slope of the best straight-line fit:
$f(x+\delta)\approx f(x)+f'(x)\,\delta$. The vector-valued generalization keeps
this idea verbatim. We say $\mathbf f$ is *differentiable* at $\mathbf x$ if there
is a matrix $\mathbf J\in\mathbb R^{m\times n}$ such that

$$
\mathbf f(\mathbf x+\boldsymbol\delta)
   = \mathbf f(\mathbf x) + \mathbf J\,\boldsymbol\delta + o(\|\boldsymbol\delta\|),
$$
:eqlabel:`eq_mdl-jacobian-linearization`

i.e. the error of the linear model $\mathbf f(\mathbf x)+\mathbf J\boldsymbol\delta$
shrinks *faster* than $\boldsymbol\delta$ itself as $\boldsymbol\delta\to\mathbf 0$.
That matrix---if it exists---is unique, is called the *Jacobian*
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

Two special cases recover everything we have already met. When $m=1$---a scalar
field $f:\mathbb R^n\to\mathbb R$---the Jacobian is a *single row*, the *row*
gradient $\partial f/\partial\mathbf x = [\partial f/\partial x_1,\ldots,\partial
f/\partial x_n]$, and :eqref:`eq_mdl-jacobian-linearization` is the first-order
expansion $f(\mathbf x+\boldsymbol\delta)\approx f(\mathbf x)+\mathbf J\boldsymbol\delta$
of :numref:`sec_mdl-multivariable_calculus`. And the Jacobian of the *gradient field*
$\nabla f:\mathbb R^n\to\mathbb R^n$ is the Hessian
$\mathbf H=\mathbf J_{\nabla f}$ of :eqref:`eq_mdl-hess_def`: differentiating once
more turns the row gradient into the matrix of second partials. One construction,
three familiar objects.

Let us pin this down on a concrete $\mathbb R^2\to\mathbb R^2$ map and check the
linear approximation against a finite difference---the numerical signature of a
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

The error is tiny *relative to* $\|\boldsymbol\delta\|$---and shrinking it by a
further factor of ten in $\boldsymbol\delta$ shrinks the error by a factor of a
hundred, the $o(\|\boldsymbol\delta\|)$ signature of :eqref:`eq_mdl-jacobian-linearization`.

### The Chain Rule Is Jacobian Composition

Here is the conceptual spine of the whole section. The multivariate chain rule of
:numref:`sec_mdl-multivariable_calculus`---a sum over paths---is, in matrix form,
nothing but *multiplication of Jacobians*.

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
$o(\|\boldsymbol\delta\|)$, and $\|\boldsymbol\eta\|=O(\|\boldsymbol\delta\|)$ folds
$\mathbf g$'s own remainder into $o(\|\boldsymbol\delta\|)$ as well. The
coefficient of $\boldsymbol\delta$ is the best linear map, so by uniqueness it *is*
the Jacobian of the composite: $\mathbf J_{\mathbf g\circ\mathbf f}=\mathbf B\mathbf A$.
$\blacksquare$

The entry-wise chain rule falls out by reading off one entry of $\mathbf B\mathbf A$:
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
(:numref:`sec_mdl-eigendecompositions`). And here is the observation that drives
everything in the second half of this section: matrix multiplication is
*associative*, so we may evaluate :eqref:`eq_mdl-jacobian-product` in any order.
Multiplying right-to-left propagates *inputs forward*; multiplying left-to-right
propagates *sensitivities backward*. Those two parenthesizations are precisely
forward-mode and reverse-mode automatic differentiation, and choosing between them
is just choosing the cheaper way to multiply a chain of matrices.

### Layout Conventions: Numerator vs Denominator

One bookkeeping decision causes more stray transposes than anything else in matrix
calculus, so we state it once, plainly. For a scalar field $f:\mathbb R^n\to\mathbb
R$, do we write the derivative as a *row* (matching the Jacobian
:eqref:`eq_mdl-jacobian-entries`) or as a *column* (matching the input vector
$\mathbf x$)? Both are in use, and they are *transposes of each other*:

* **Numerator (Jacobian) layout.** The derivative has the shape of the
  *numerator*: $\partial\mathbf y/\partial\mathbf x$ is $m\times n$, so the scalar
  case is a row. This makes the chain rule :eqref:`eq_mdl-jacobian-chain` read
  left-to-right with no transposes, which is why it is the convention of choice for
  the Jacobian theory above and for the MIT matrix-calculus course.
* **Denominator (gradient) layout.** The derivative has the shape of the
  *denominator*: $\partial f/\partial\mathbf x$ is a *column* matching $\mathbf x$.
  This is the statistician's $\nabla_{\mathbf x} f$, the thing you subtract in
  gradient descent, and the convention of the migrated derivations below.

Switching layouts simply transposes the result, so the gradient column and the
Jacobian row of a scalar field are related by

$$
\nabla_{\mathbf x} f = \Bigl(\frac{\partial f}{\partial\mathbf x}\Bigr)^{\!\top}.
$$
:eqlabel:`eq_mdl-grad-is-jacobian-transpose`

**This book's convention.** We use **numerator layout for Jacobians** of genuine
vector-valued maps (the theory is cleaner) and **denominator layout for
gradients**, writing $\nabla_{\mathbf x} f$ for the column you feed an optimizer.
Whenever the two could be confused we make the shape explicit. With that settled,
the identities in the next section are gradients, hence columns; if you ever land
on an expression whose shape does not match its denominator, you have a missing
transpose---use :eqref:`eq_mdl-grad-is-jacobian-transpose` to fix it. The Matrix
Cookbook :cite:`Petersen.Pedersen.ea.2008` tabulates hundreds of such identities;
the point of the next section is that you need to *memorize* almost none of them.

## A Few Key Identities, Derived Not Tabulated
:label:`subsec_mdl-matrix-identities`

A reference table of matrix-derivative identities is long and forgettable. In
practice only a handful recur in deep learning, and each one yields to the same
two-step recipe of Parr and Howard's primer
([explained.ai/matrix-calculus](https://explained.ai/matrix-calculus/)):
*differentiate one component with the ordinary scalar rules, then reassemble the
components into a matrix expression.* A faster cross-check, due to the same source,
is the **scalar-collapse heuristic**: a correct matrix identity must reduce to the
familiar single-variable result when every matrix is $1\times1$ (where products are
scalar products, sums are sums, and transposes do nothing), so you can often *guess*
the matrix form from the scalar one and fix the shapes by inserting transposes.
We derive four identities this way; all are gradients, hence columns (denominator
layout, :eqref:`eq_mdl-grad-is-jacobian-transpose`).

**The linear form $\nabla_{\mathbf x}\,\mathbf a^\top\mathbf x = \mathbf a$.** With
$f(\mathbf x)=\mathbf a^\top\mathbf x=\sum_i a_i x_i$, the $k$-th partial keeps only
the $i=k$ term, so $\partial f/\partial x_k = a_k$; stacking these gives
$\nabla_{\mathbf x} f=\mathbf a$. The scalar shadow is $\frac{d}{dx}(ax)=a$, exactly
the matrix answer with the transpose doing nothing---reassuring. (This also repairs
the slip in the version of this derivation that lived in
:numref:`sec_mdl-multivariable_calculus`: we differentiate with respect to $x_k$,
not $a_k$.)

**The quadratic form $\nabla_{\mathbf x}\,\mathbf x^\top\mathbf A\mathbf x =
(\mathbf A+\mathbf A^\top)\mathbf x$.** Write the form in Einstein notation as
$\mathbf x^\top\mathbf A\mathbf x = x_i a_{ij} x_j$. The product rule on
$\partial/\partial x_k$ hits both $x$'s,

$$
\frac{\partial}{\partial x_k}\bigl(x_i a_{ij} x_j\bigr)
   = \delta_{ik}\,a_{ij}x_j + x_i a_{ij}\,\delta_{jk}
   = a_{kj}x_j + x_i a_{ik}
   = (a_{ki}+a_{ik})\,x_i,
$$

where the last step renames the dummy index. The bracket $a_{ki}+a_{ik}$ is the
$(k,i)$ entry of $\mathbf A+\mathbf A^\top$, so the $k$-th partial is the $k$-th
entry of $(\mathbf A+\mathbf A^\top)\mathbf x$, giving
$\nabla_{\mathbf x}\,\mathbf x^\top\mathbf A\mathbf x=(\mathbf A+\mathbf A^\top)\mathbf x$.
The scalar collapse is $\frac{d}{dx}(a x^2)=2ax=(a+a)x$; the matrix answer is the
same with $\mathbf A^\top$ standing in for the "other copy" of $a$. For *symmetric*
$\mathbf A$ this simplifies to the much-used $2\mathbf A\mathbf x$.

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

a rank-one *outer product* of the residual with the input---this is the gradient a
single linear layer hands back during backpropagation. The companion gradient with
respect to the *input*, $\nabla_{\mathbf x}\|\mathbf W\mathbf x-\mathbf y\|^2
=2\mathbf W^\top(\mathbf W\mathbf x-\mathbf y)$, follows identically; note the
$\mathbf W^\top$, the same transpose that the scalar-collapse heuristic predicts for
the factorization gradient $\partial_{\mathbf V}\|\mathbf X-\mathbf U\mathbf
V\|^2=-2\mathbf U^\top(\mathbf X-\mathbf U\mathbf V)$ derived in
:numref:`sec_mdl-multivariable_calculus`.

**The softmax--cross-entropy Jacobian.** The single most important derivative in a
classifier is also the cleanest. Let $\mathbf p=\operatorname{softmax}(\mathbf z)$,
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
:eqref:`eq_mdl-softmax-jacobian`, and the algebra collapses spectacularly:

$$
\frac{\partial\ell}{\partial z_j}
   = -\sum_i \frac{y_i}{p_i}\,p_i(\delta_{ij}-p_j)
   = -y_j + p_j\sum_i y_i
   = p_j - y_j,
$$

using $\sum_i y_i=1$. The gradient of softmax--cross-entropy with respect to the
*logits* is just $\mathbf p-\mathbf y$: predicted probability minus truth. The
notorious softmax Jacobian cancels against the cross-entropy gradient, which is
exactly why the two are always fused into one numerically stable primitive. Let us
confirm both :eqref:`eq_mdl-softmax-jacobian` and the cancellation against a
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
#@tab mxnet, tensorflow
z = np.array([1.0, -0.5, 2.0])
p = np.exp(z) / np.exp(z).sum()
J = np.diag(p) - np.outer(p, p)                # our formula
y = np.array([0.0, 1.0, 0.0])                  # one-hot target (class 1)
print('softmax Jacobian rows sum to zero:', np.allclose(J.sum(axis=1), 0))
print('d loss / d z = p - y :', (p - y).round(6))
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

The logit gradient lands exactly on $\mathbf p-\mathbf y$, no matter how the
intermediate Jacobian looks. Four identities, one method: differentiate a single
component, then read the matrix back off the indices, sanity-checking against the
scalar collapse.

## Forward-Mode AD and Dual Numbers
:label:`subsec_mdl-forward-mode`

We now stop differentiating by hand. Automatic differentiation evaluates a function
*and* its derivative in one sweep, to full numerical precision, by carrying
derivative information alongside every value. It is neither symbolic differentiation
(which manipulates formulas and explodes in size, as the swamp of repeated terms in
:numref:`sec_mdl-multivariable_calculus` showed) nor numerical differentiation
(finite differences, which trade off truncation against round-off error). It is a
third thing: the chain rule, applied mechanically to the *program* that computes the
function :cite:`Baydin.Pearlmutter.Radul.ea.2018`. *Forward mode* is the simplest
incarnation, and the cleanest way to see it is through *dual numbers*.

### Dual Numbers: an Algebra That Carries Derivatives

Define a *dual number* $a+b\,\varepsilon$ by analogy with the complex numbers, but
with a new symbol $\varepsilon$ satisfying

$$
\varepsilon^2 = 0, \qquad \varepsilon\neq0.
$$
:eqlabel:`eq_mdl-dual-rule`

Think of $a$ as a value and $b$ as "the derivative riding along". The rule
$\varepsilon^2=0$ is the *exact* algebraic encoding of "discard second-order terms",
the very move we used to derive every differentiation rule in
:numref:`sec_mdl-single_variable_calculus`. Addition is componentwise, and
multiplication, dropping the $\varepsilon^2$ term, is

$$
(a+b\varepsilon)(c+d\varepsilon) = ac + (ad+bc)\,\varepsilon.
$$
:eqlabel:`eq_mdl-dual-mul`

Stare at the $\varepsilon$ coefficient: $ad+bc$ is the *product rule*. This is no
coincidence.

**Proposition (dual numbers compute derivatives).** *For a differentiable $f$,
evaluating its arithmetic on the dual number $x+1\cdot\varepsilon$ yields*

$$
f(x + \varepsilon) = f(x) + f'(x)\,\varepsilon.
$$
:eqlabel:`eq_mdl-dual-eval`

**Proof.** The claim holds for the constant maps and the identity ($x+\varepsilon$
itself), and it is closed under the operations a program is built from. For a sum,
$\varepsilon$-coefficients add, matching $(f+g)'=f'+g'$. For a product,
:eqref:`eq_mdl-dual-mul` gives coefficient $f'(x)g(x)+f(x)g'(x)$, the product rule.
For a composition, substitute $g(x)+g'(x)\varepsilon$ into $f$: a Taylor expansion
of $f$ about $g(x)$ truncates after the linear term because $\varepsilon^2=0$,
leaving $f(g(x))+f'(g(x))g'(x)\,\varepsilon$---the chain rule. Since every
elementary function is assembled from these, induction on the expression gives
:eqref:`eq_mdl-dual-eval`. $\blacksquare$

So differentiation is *free*: run the ordinary computation in the dual-number
algebra, set the input's $\varepsilon$-part to $1$ (the "seed"), and the output's
$\varepsilon$-part is the derivative. No formula for $f'$ is ever written down. The
implementation is a few lines of operator overloading.

```{.python .input #matrix-calculus-autodiff-dual-numbers}
class Dual:
    """A dual number a + b*eps with eps^2 = 0; b carries the derivative."""
    def __init__(self, a, b=0.0):
        self.a, self.b = a, b                       # value, derivative
    def __add__(self, o):
        o = o if isinstance(o, Dual) else Dual(o)
        return Dual(self.a + o.a, self.b + o.b)
    __radd__ = __add__
    def __mul__(self, o):
        o = o if isinstance(o, Dual) else Dual(o)
        return Dual(self.a * o.a, self.a * o.b + self.b * o.a)   # product rule
    __rmul__ = __mul__
    def sin(self):
        return Dual(np.sin(self.a), np.cos(self.a) * self.b)     # chain rule
    def exp(self):
        e = np.exp(self.a)
        return Dual(e, e * self.b)
    def __repr__(self):
        return f'{self.a:.6f} + {self.b:.6f} eps'
```

To differentiate $f(x)=\sin(x^2)+e^x$ at a point, we evaluate it on
$x+1\cdot\varepsilon$ and read the two parts of the result. The value should match
$f$ and the $\varepsilon$-part should match $f'(x)=2x\cos(x^2)+e^x$.

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

Value and derivative emerge from a single evaluation. To recover a full Jacobian of
$\mathbf f:\mathbb R^n\to\mathbb R^m$ we would seed one input direction at a time:
seeding $\boldsymbol\delta=\mathbf e_j$ propagates the $j$-th *column* of the
Jacobian, so $n$ forward passes assemble the whole matrix. More generally a single
forward pass with seed $\mathbf v$ computes the **Jacobian--vector product**
$\mathbf J\mathbf v$ (a JVP)---it never forms $\mathbf J$ explicitly. This is why
forward mode is cheap for *tall* Jacobians (many outputs, few inputs, $m\gg n$) and
expensive when $n$ is large---exactly the wrong regime for a deep network, whose
loss has a single scalar output and millions of inputs. For that we need the other
parenthesization.

## Reverse-Mode AD, the Tape, and Backprop
:label:`subsec_mdl-reverse-mode`

Recall the associativity observation from :eqref:`eq_mdl-jacobian-product`: the
chain of Jacobians may be multiplied in any order. Forward mode multiplies
*right-to-left*, building up $\mathbf J\mathbf v$ one input direction at a time.
*Reverse mode* multiplies *left-to-right*: it computes the **vector--Jacobian
product** $\mathbf u^\top\mathbf J$ (a VJP), one *output* direction at a time.

### Why Reverse Mode Is the Right Cost Model

The asymmetry is decisive for deep learning. Consider a scalar loss $L:\mathbb
R^n\to\mathbb R$ with $n$ parameters. Its Jacobian is a single row ($m=1$), so
seeding the reverse pass with $\mathbf u=1$ at the output computes the *entire*
gradient $\nabla L$ in **one** backward sweep. Forward mode, by contrast, would need
$n$ passes---one per parameter direction---to assemble the same row. The cost rule
is worth stating crisply:

* **Reverse mode** costs one pass per *output*: cheap for *wide* Jacobians
  ($m\ll n$). A scalar loss is the extreme wide case, so its full gradient costs
  about the same as *one* extra forward evaluation, independent of $n$.
* **Forward mode** costs one pass per *input*: cheap for *tall* Jacobians
  ($m\gg n$).

A network has millions of parameters and one loss, so reverse mode wins by a factor
of millions. That single fact---gradient of a scalar at the price of one function
evaluation, regardless of parameter count---is the entire reason training deep
networks is computationally feasible. **This is backpropagation**: the algorithm of
:numref:`sec_mdl-multivariable_calculus`, where we insisted on "keeping
$\partial f$ in the numerator", is reverse-mode AD, and the rule is now rigorous.

### The Tape: Record Forward, Replay Backward

Reverse mode cannot be done by a value-carrying algebra like dual numbers, because
the multiplications run *opposite* to the program's execution: we need each
operation's local Jacobian *after* the forward pass has finished. So we *record* the
computation as it runs---the *Wengert list* or *tape* :cite:`Wengert.1964`---and
then walk it backward. The forward pass builds a directed acyclic graph of
elementary operations and stashes the values needed for each local derivative; the
backward pass seeds the output *adjoint* (the running $\partial L/\partial\cdot$) and
pushes it through each recorded operation's VJP, accumulating contributions where a
value feeds several consumers. :numref:`fig_mdl-cal-fwd-vs-rev` contrasts the two
sweeps.

![Forward-mode versus reverse-mode automatic differentiation on a computation graph. Forward mode (top) propagates a Jacobian--vector product left to right, in lock-step with evaluation, computing one Jacobian column per pass. Reverse mode (bottom) records a tape on a forward pass, then propagates a vector--Jacobian product right to left, computing one Jacobian row---the full gradient of a scalar output---per pass.](../img/mdl-cal-fwd-vs-rev.svg)
:label:`fig_mdl-cal-fwd-vs-rev`

A tiny tape makes this concrete. We record each operation as a node holding its
value and a *backward* closure that, given the adjoint of the node's output,
increments the adjoints of its inputs by the appropriate VJP. Replaying the tape in
reverse and seeding the final adjoint with $1$ leaves $\partial L/\partial\cdot$ in
every input's adjoint slot.

```{.python .input #matrix-calculus-autodiff-tape}
class Var:
    """A scalar node on the autodiff tape; .grad accumulates the adjoint."""
    def __init__(self, value, parents=(), backward=lambda g: None):
        self.value = value
        self.grad = 0.0
        self._parents = parents                     # nodes this one depends on
        self._backward = backward                   # pushes adjoint to parents
    def __add__(self, o):
        o = o if isinstance(o, Var) else Var(o)
        out = Var(self.value + o.value, (self, o))
        def back(g):                                # d(a+b) = 1*g to each parent
            self.grad += g; o.grad += g
        out._backward = back
        return out
    __radd__ = __add__
    def __mul__(self, o):
        o = o if isinstance(o, Var) else Var(o)
        out = Var(self.value * o.value, (self, o))
        def back(g):                                # product rule, reversed
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

We differentiate the same $(u+v)^2$-style expression that
:numref:`sec_mdl-multivariable_calculus` worked by hand, and check the tape's
gradients against a framework's autograd. Take $g(u,v)=(u\,v + u)^2$; one forward
pass records the tape, one backward pass yields both partials.

```{.python .input #matrix-calculus-autodiff-tape-check}
#@tab pytorch
u, v = Var(2.0), Var(-3.0)
y = (u * v + u) * (u * v + u)                       # (u v + u)^2 on our tape
backprop(y)
ut = torch.tensor(2.0, requires_grad=True)
vt = torch.tensor(-3.0, requires_grad=True)
(ut * vt + ut).pow(2).backward()                    # framework autograd
print('our tape   : dy/du = %.4f  dy/dv = %.4f' % (u.grad, v.grad))
print('torch      : dy/du = %.4f  dy/dv = %.4f' % (ut.grad, vt.grad))
```

```{.python .input #matrix-calculus-autodiff-tape-check}
#@tab mxnet, tensorflow
u, v = Var(2.0), Var(-3.0)
y = (u * v + u) * (u * v + u)                       # (u v + u)^2 on our tape
backprop(y)
# Closed-form check: y = (uv+u)^2, dy/du = 2(uv+u)(v+1), dy/dv = 2(uv+u)u
r = u.value * v.value + u.value
print('our tape   : dy/du = %.4f  dy/dv = %.4f' % (u.grad, v.grad))
print('closed form: dy/du = %.4f  dy/dv = %.4f'
      % (2 * r * (v.value + 1), 2 * r * u.value))
```

```{.python .input #matrix-calculus-autodiff-tape-check}
#@tab jax
u, v = Var(2.0), Var(-3.0)
y = (u * v + u) * (u * v + u)                       # (u v + u)^2 on our tape
backprop(y)
g = lambda u, v: (u * v + u) ** 2
du, dv = jax.grad(g, argnums=(0, 1))(2.0, -3.0)     # framework autograd
print('our tape   : dy/du = %.4f  dy/dv = %.4f' % (u.grad, v.grad))
print('jax        : dy/du = %.4f  dy/dv = %.4f' % (float(du), float(dv)))
```

Our thirty-line tape reproduces the framework's gradients exactly. The real systems
generalize it in three ways that do not change the principle: the nodes hold
*tensors* rather than scalars, so each `_backward` is a VJP (a matrix operation, e.g.
the $\mathbf x^\top$ in :eqref:`eq_mdl-lstsq-grad`) rather than a scalar multiply;
the primitive set is large (every differentiable op ships a VJP rule); and the graph
is built either eagerly (PyTorch's dynamic tape) or by tracing (JAX, TensorFlow).
But the skeleton---record forward, seed the output adjoint, replay backward
accumulating VJPs---is exactly what you wrote above and exactly what
`loss.backward()` does.

One trade-off is now visible. Reverse mode must *store the forward intermediates*
(the `value`s on the tape) until the backward pass consumes them, so its memory
grows with the length of the computation---whereas forward mode keeps only the
current value and tangent. This is the memory cost that *gradient checkpointing*
trades back for recomputation, and it is why training memory scales with network
depth. The full story of forward/reverse modes, mixed modes, and checkpointing is
the subject of Griewank and Walther's monograph :cite:`Griewank.1989` and the survey
:cite:`Baydin.Pearlmutter.Radul.ea.2018`; the same reverse-mode tape returns as the
*adjoint method* for differentiating through ODE solvers in
:numref:`chap_mdl-dynamics`.

## Summary

* The **Jacobian** $\mathbf J_{\mathbf f}\in\mathbb R^{m\times n}$ is the best local
  linear approximation of a vector-valued $\mathbf f$:
  $\mathbf f(\mathbf x+\boldsymbol\delta)\approx\mathbf f(\mathbf x)+\mathbf J\boldsymbol\delta$,
  with $[\mathbf J]_{ij}=\partial f_i/\partial x_j$. The row gradient ($m=1$) and the
  Hessian (Jacobian of $\nabla f$) are special cases.
* The chain rule in matrix form is **Jacobian composition**
  $\mathbf J_{\mathbf g\circ\mathbf f}=\mathbf J_{\mathbf g}\mathbf J_{\mathbf f}$, so
  a depth-$L$ network's derivative is one long matrix product
  $\mathbf J_L\cdots\mathbf J_1$. Its *associativity* is the whole story of AD.
* **Layout conventions** (numerator vs denominator) differ by a transpose; we use
  numerator layout for Jacobians and denominator (column) layout for gradients.
* The recurring DL identities---$\nabla\,\mathbf a^\top\mathbf x=\mathbf a$,
  $\nabla\,\mathbf x^\top\mathbf A\mathbf x=(\mathbf A+\mathbf A^\top)\mathbf x$,
  the rank-one $\nabla_{\mathbf W}\|\mathbf W\mathbf x-\mathbf y\|^2=2(\mathbf W\mathbf x-\mathbf y)\mathbf x^\top$,
  and the softmax--cross-entropy logit gradient $\mathbf p-\mathbf y$---all follow
  from differentiating one component and reassembling, not from a table.
* **Forward-mode AD** carries a derivative via *dual numbers* ($\varepsilon^2=0$);
  one pass computes a Jacobian--vector product (a Jacobian *column*), cheap for tall
  Jacobians.
* **Reverse-mode AD** records a *tape* and replays it backward, computing a
  vector--Jacobian product (a Jacobian *row*) per pass. Because a loss is scalar,
  **backpropagation is reverse-mode AD**: one backward sweep yields the gradient w.r.t.
  every parameter at the cost of one extra forward pass---at the price of storing the
  forward intermediates.

## Exercises

1. Compute the Jacobian of $\mathbf f(x,y)=(x^2y,\ \sin(x+y))$ and verify the linear
   approximation $\mathbf f(\mathbf x+\boldsymbol\delta)\approx\mathbf f(\mathbf x)+\mathbf J\boldsymbol\delta$
   at $(1,0)$ against a finite difference. By how much does the error shrink when you
   halve $\boldsymbol\delta$?
2. Show $\nabla_{\mathbf x}\|\mathbf A\mathbf x-\mathbf b\|_2^2=2\mathbf A^\top(\mathbf A\mathbf x-\mathbf b)$
   two ways: by an index/Einstein-notation derivation, and by the scalar-collapse
   heuristic (guess from $\frac{d}{dx}(ax-b)^2$, then fix the shapes with transposes).
3. For a scalar loss with $n=10^6$ parameters, how many passes does forward mode need
   to assemble the full gradient, and how many does reverse mode need? Explain via JVPs
   versus VJPs.
4. Extend the `Dual` class with a `__pow__` (or a `log`) method and use it to
   differentiate $f(x)=\log(1+x^2)$ at $x=2$; check against the analytic derivative.
   Which rule from the proof of :eqref:`eq_mdl-dual-eval` does each new method encode?
5. Add a new primitive (e.g. $\exp$ or $\sin$) to the reverse-mode `Var`/`backprop`
   tape, give it the correct `_backward`, and verify a gradient against a framework's
   autograd.
6. Derive the softmax Jacobian :eqref:`eq_mdl-softmax-jacobian` from the quotient
   rule, then re-derive the logit gradient $\mathbf p-\mathbf y$ and explain why
   fusing softmax with cross-entropy is numerically preferable to composing them.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/)
:end_tab:

<!-- slides -->

::: {.slide title="Vectors in, vectors out: the Jacobian"}
A layer $\mathbf f:\mathbb R^n\to\mathbb R^m$ has a whole
matrix of derivatives---the **Jacobian**, the best local
linear map:

$$\mathbf f(\mathbf x+\boldsymbol\delta)\approx\mathbf f(\mathbf x)+\mathbf J\boldsymbol\delta,
\qquad [\mathbf J]_{ij}=\frac{\partial f_i}{\partial x_j}.$$

A scalar field's gradient is one *row*; the Hessian is the
Jacobian of the gradient. One object, three familiar faces.
:::

::: {.slide title="Chain rule = Jacobian product"}
Composition multiplies Jacobians:

$$\mathbf J_{\mathbf g\circ\mathbf f}=\mathbf J_{\mathbf g}\,\mathbf J_{\mathbf f},
\qquad \mathbf J = \mathbf J_L\,\mathbf J_{L-1}\cdots\mathbf J_1.$$

A deep net's derivative is one long matrix product. Matrix
multiplication is **associative** -- the order you multiply
it in *is* the choice between forward- and reverse-mode AD.
:::

::: {.slide title="Identities you derive, not memorize"}
Differentiate one component, reassemble, sanity-check the
$1\times1$ collapse:

- $\nabla_{\mathbf x}\,\mathbf a^\top\mathbf x = \mathbf a$
- $\nabla_{\mathbf x}\,\mathbf x^\top\mathbf A\mathbf x = (\mathbf A+\mathbf A^\top)\mathbf x$
- $\nabla_{\mathbf W}\|\mathbf W\mathbf x-\mathbf y\|^2 = 2(\mathbf W\mathbf x-\mathbf y)\mathbf x^\top$
- softmax + cross-entropy logit gradient $=\mathbf p-\mathbf y$
:::

::: {.slide title="Forward mode = dual numbers"}
Carry a derivative alongside every value with $\varepsilon^2=0$:

$$(a+b\varepsilon)(c+d\varepsilon)=ac+(ad+bc)\varepsilon,
\qquad f(x+\varepsilon)=f(x)+f'(x)\varepsilon.$$

One pass = a **Jacobian--vector product** (one column).
Cheap for *tall* Jacobians. A 15-line `Dual` class
differentiates $\sin(x^2)+e^x$ for free:

@matrix-calculus-autodiff-dual-eval
:::

::: {.slide title="Reverse mode = the tape = backprop"}
Record the ops forward, replay backward, accumulating
**vector--Jacobian products** (one row per pass). A scalar
loss ($m=1$) $\Rightarrow$ the whole gradient in **one**
backward sweep, regardless of parameter count -- *this is why
training is feasible*.

@matrix-calculus-autodiff-tape-check
:::

::: {.slide title="Recap"}
- Jacobian: best local linear map; gradient and Hessian are
  special cases.
- Chain rule = Jacobian product; associativity = forward vs.
  reverse mode.
- Forward (JVP, dual numbers): one column/pass, good for tall.
- Reverse (VJP, tape): one row/pass, good for wide -- and a
  loss is maximally wide.
- **Backprop = reverse-mode AD**: full gradient at the cost of
  one extra forward pass, paid for by storing intermediates.
:::
