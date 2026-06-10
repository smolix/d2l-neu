# Numerical Stability and Conditioning
:label:`sec_mdl-numerical-stability-conditioning`

The math can be right and your loss can still go to `NaN`. Every result in this
chapter so far was proved over the real numbers; your GPU computes over a
finite, gappy imitation of them. This section explains the floating-point
failure modes that bite real training runs --- overflow, underflow, and
catastrophic cancellation --- and the handful of two-line fixes that keep
softmax, cross-entropy, and ill-conditioned least squares alive:
max-subtraction, log-space arithmetic, and ridge regularization. Two ideas
organize everything. The first, due to numerical analysis and crystallized in
:citet:`Higham.2002`, is to split the blame for a wrong answer between the
*algorithm* (did it solve a nearby problem?) and the *problem* (do nearby
problems have wildly different answers?). The second is that the dividing line
is a single number we have already met: the **condition number**
$\kappa = \sigma_{\max}/\sigma_{\min}$ of :numref:`subsec_mdl-condition-number`,
the same $\kappa$ that sets gradient descent's convergence rate in
:numref:`sec_mdl-gradient-based-optimization` --- one number, two consequences.
The payoffs land downstream: the log-space trick rescues naive Bayes in
:numref:`sec_mdl-naive_bayes` from underflow, and the stable cross-entropy here
is the same computation analyzed in :numref:`sec_mdl-information_theory`.

We proceed in four steps: what floating-point numbers are and where their
cliffs lie; how to compute softmax, log-sum-exp, and cross-entropy without
falling off those cliffs; why subtracting nearly equal numbers destroys digits
and how reformulation (not higher precision) repairs it; and finally
conditioning --- backward versus forward error, the Hilbert-matrix horror
story, why normal equations square the pain, and why ridge regularization is
preconditioning in disguise. The standard references are
:citet:`Goldberg.1991` for floating point and :citet:`Higham.2002` for
everything else; :citet:`Goodfellow.Bengio.Courville.2016` (chapter 4) gives
the deep-learning framing. Most code in this section is deliberately plain
NumPy --- these phenomena belong to the arithmetic, not to any framework ---
with one pointed exception: the cross-entropy cell, where the four frameworks
genuinely behave differently.

```{.python .input #numerical-stability-conditioning-imports}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np as mxnp, npx
npx.set_np()
import numpy as np
```

```{.python .input #numerical-stability-conditioning-imports}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch.nn import functional as F
import numpy as np
```

```{.python .input #numerical-stability-conditioning-imports}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import ml_dtypes
import numpy as np
```

```{.python .input #numerical-stability-conditioning-imports}
#@tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import optax
import numpy as np
```

## Floating-Point Arithmetic
:label:`subsec_mdl-floating-point`

### A Number System with Gaps

A floating-point number is scientific notation in base $2$ with a fixed budget
of digits:

$$
x = (-1)^s \cdot (1.m_1 m_2 \ldots m_p)_2 \cdot 2^{e},
$$
:eqlabel:`eq_mdl-opt-float-format`

a sign bit $s$, a *mantissa* (significand) with $p$ stored bits, and an
integer exponent $e$ from a fixed range. The format is a brilliant compromise:
the exponent gives enormous *range*, the mantissa gives fixed *relative*
precision, and between consecutive powers of two the representable values are
evenly spaced --- so the spacing *doubles* every time the magnitude does.
:numref:`fig_mdl-opt-fp-number-line` shows the resulting number line:
representable values crowd near zero and thin out toward the overflow cliff,
while the *relative* gap between neighbors stays essentially constant.

![Floating-point numbers on the real line. Representable values are dense near zero and sparse far out: the absolute gap between neighbors doubles at every power of two while the relative gap stays near $\varepsilon_{\text{mach}}$. Each format ends in an overflow cliff --- fp16 at $65504$, long before fp32 at about $3.4 \times 10^{38}$ --- and in an underflow region below its smallest normal number.](../img/mdl-opt-fp-number-line.svg)
:label:`fig_mdl-opt-fp-number-line`

That constant relative gap has a name. **Machine epsilon**
$\varepsilon_{\text{mach}}$ is the distance from $1$ to the next representable
number, $\varepsilon_{\text{mach}} = 2^{-p}$ for a $p$-bit mantissa, and it is
the relative-error floor of the entire arithmetic: rounding any real $x$ to
the nearest float $\mathrm{fl}(x)$ obeys

$$
\mathrm{fl}(x) = x\,(1 + \delta), \qquad |\delta| \le u = \tfrac12\,\varepsilon_{\text{mach}},
$$
:eqlabel:`eq_mdl-opt-rounding-model`

and IEEE arithmetic guarantees the same for every single operation: the
computed $x \oplus y$ equals $(x + y)(1 + \delta)$ with $|\delta| \le u$
(:cite:`Goldberg.1991`). The quantity $u$ is the *unit roundoff*. Everything
in this section is bookkeeping on these $(1+\delta)$ factors: one of them is
harmless, and the games begin when millions of them interact --- or when a
subtraction promotes one from relative to *catastrophic* (we get there in
:numref:`subsec_mdl-catastrophic-cancellation`).

Deep learning juggles three formats, and the table below --- printed by your
framework, not transcribed from a spec --- is worth memorizing:

```{.python .input #numerical-stability-conditioning-finfo}
#@tab mxnet
header = f'{"dtype":>10} {"eps":>12} {"smallest normal":>17} {"max":>12}'
print(header)
for dt in [np.float16, np.float32]:
    fi = np.finfo(dt)
    print(f'{np.dtype(dt).name:>10} {fi.eps:12.3e} '
          f'{fi.smallest_normal:17.3e} {fi.max:12.3e}')

def to_bf16(x):
    """Round float32 to the nearest bfloat16 (round half to even)."""
    bits = np.atleast_1d(np.asarray(x, np.float32)).view(np.uint32)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return bits.astype(np.uint32).view(np.float32)

eps_bf16 = float(to_bf16(1.0 + 2.0**-7) - 1.0)   # emulated: mxnet has no bf16
print(f'{"bfloat16":>10} {eps_bf16:12.3e}   (exponent range = float32)')
print('bfloat16 eps equals 2^-7:', eps_bf16 == 2.0**-7,
      ' and 1 + 2^-8 rounds back to 1:', float(to_bf16(1.0 + 2.0**-8)) == 1.0)
```

```{.python .input #numerical-stability-conditioning-finfo}
#@tab pytorch
print(f'{"dtype":>10} {"eps":>12} {"smallest normal":>17} {"max":>12}')
for dt in [torch.float16, torch.bfloat16, torch.float32]:
    fi = torch.finfo(dt)
    print(f'{str(dt)[6:]:>10} {fi.eps:12.3e} '
          f'{fi.smallest_normal:17.3e} {fi.max:12.3e}')
print('bfloat16 eps equals 2^-7:',
      torch.finfo(torch.bfloat16).eps == 2.0**-7)
```

```{.python .input #numerical-stability-conditioning-finfo}
#@tab tensorflow
print(f'{"dtype":>10} {"eps":>12} {"smallest normal":>17} {"max":>12}')
for dt in [tf.float16, tf.bfloat16, tf.float32]:
    fi = ml_dtypes.finfo(dt.as_numpy_dtype)
    print(f'{dt.name:>10} {float(fi.eps):12.3e} '
          f'{float(fi.smallest_normal):17.3e} {float(fi.max):12.3e}')
print('bfloat16 eps equals 2^-7:',
      float(ml_dtypes.finfo(tf.bfloat16.as_numpy_dtype).eps) == 2.0**-7)
```

```{.python .input #numerical-stability-conditioning-finfo}
#@tab jax
print(f'{"dtype":>10} {"eps":>12} {"smallest normal":>17} {"max":>12}')
for dt in [jnp.float16, jnp.bfloat16, jnp.float32]:
    fi = jnp.finfo(dt)
    print(f'{np.dtype(dt).name:>10} {float(fi.eps):12.3e} '
          f'{float(fi.smallest_normal):17.3e} {float(fi.max):12.3e}')
print('bfloat16 eps equals 2^-7:',
      float(jnp.finfo(jnp.bfloat16).eps) == 2.0**-7)
```

Read the three rows as three different bargains. **fp32** ($p = 23$ mantissa
bits) has $\varepsilon_{\text{mach}} = 2^{-23} \approx 1.19 \times 10^{-7}$
--- about seven decimal digits --- with range up to
$3.4 \times 10^{38}$. **fp16** ($p = 10$) keeps a respectable
$\varepsilon_{\text{mach}} = 2^{-10} \approx 9.8 \times 10^{-4}$ but pays for
it with a *tiny* exponent range: it overflows at $65504$ and its smallest
normal number is about $6.1 \times 10^{-5}$, so both big activations and small
gradients fall off its cliffs. **bfloat16** ($p = 7$) makes the opposite
trade: it keeps fp32's full exponent range and sacrifices the mantissa,
leaving $\varepsilon_{\text{mach}} = 2^{-7} = 0.0078$ --- between two and
three decimal digits. The printout confirms the value that is easy to misquote:
bfloat16's epsilon is $2^{-7}$, not $2^{-8}$; the eighth mantissa bit people
sometimes count is the *implicit* leading $1$ in
:eqref:`eq_mdl-opt-float-format`, which contributes precision to no gap.

Two quick experiments make $\varepsilon_{\text{mach}}$ tangible: adding half
an epsilon to $1$ vanishes without a trace, and the absolute gap between
neighbors is a million times larger at $2^{20}$ than at $1$:

```{.python .input #numerical-stability-conditioning-spacing}
eps = np.finfo(np.float32).eps
one = np.float32(1.0)
print('1 + eps   != 1 :', one + eps != one)
print('1 + eps/2 == 1 :', one + np.float32(eps / 2) == one)
print('gap between adjacent float32 values near 1    :',
      np.spacing(np.float32(1.0)))
print('gap between adjacent float32 values near 2^20 :',
      np.spacing(np.float32(2.0**20)))
for dt in [np.float16, np.float32]:
    print(f'{np.dtype(dt).name}: exp(x) overflows for x >',
          f'{np.log(np.finfo(dt).max):.2f}')
```

### Overflow, Underflow, and Mixed Precision

The last two printed lines locate the cliffs that matter most in practice.
Because $e^x$ turns additive scale into multiplicative scale, the overflow
threshold of each format translates into a surprisingly small *logit*:
$e^x = \infty$ in fp32 once $x > \ln(3.4 \times 10^{38}) \approx 88.72$, and
in fp16 once $x > \ln(65504) \approx 11.09$. A logit of $89$ --- nothing
unusual for an unnormalized score late in training --- overflows fp32. At the
other end, $e^{-x}$ *underflows*: below the smallest normal number the format
degrades gracefully through *subnormal* numbers with fewer and fewer
significant bits, and then hits exactly $0$, at which point a subsequent
$\log$ returns $-\infty$ and the backward pass turns to `NaN`.

This is the arithmetic behind **mixed-precision training**
:cite:`Micikevicius.Narang.Alben.ea.2018`. In fp16, gradients routinely fall
below $6 \times 10^{-5}$ and vanish, so the loss is multiplied by a scale
factor before the backward pass (and the gradients divided after) purely to
shift them into representable territory --- *loss scaling* is underflow
management, nothing more. bfloat16 was designed to make that bookkeeping
unnecessary: with fp32's exponent range nothing reasonable overflows or
underflows, and the price --- relative precision of only $2^{-7}$ --- is paid
where deep learning is most tolerant, in the noise-dominated mantissa of each
weight update. The reason a master copy of the weights is kept in fp32 is
:eqref:`eq_mdl-opt-rounding-model` again: a weight update of relative size
below $\varepsilon_{\text{mach}}/2$ rounds to *no update at all*, and at
$\varepsilon_{\text{mach}} = 2^{-7}$ that threshold is hit by perfectly
healthy learning rates.

## Making Softmax and Cross-Entropy Safe
:label:`subsec_mdl-stable-softmax`

### Softmax Overflows --- and the Shift That Fixes It

The most common stability bug in machine learning is one line long. The
softmax

$$
\mathrm{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^n e^{z_j}}
$$

exponentiates its logits, so by the previous subsection it overflows fp32 the
moment any logit exceeds $88.72$ --- the numerator becomes `inf`, the ratio
becomes `inf/inf = NaN`, and the model dies even though the *probabilities*
it was computing are perfectly tame numbers in $[0, 1]$. The failure is
entirely an artifact of the route, and the fix exploits a symmetry of the
destination.

**Proposition (softmax is shift-invariant).** *For every
$\mathbf{z} \in \mathbb{R}^n$ and every $c \in \mathbb{R}$,*

$$
\mathrm{softmax}(\mathbf{z} - c\mathbf{1}) = \mathrm{softmax}(\mathbf{z}).
$$
:eqlabel:`eq_mdl-opt-softmax-shift`

**Proof.** Componentwise,

$$
\frac{e^{z_i - c}}{\sum_j e^{z_j - c}}
= \frac{e^{-c}\, e^{z_i}}{e^{-c} \sum_j e^{z_j}}
= \frac{e^{z_i}}{\sum_j e^{z_j}},
$$

since the common factor $e^{-c} > 0$ cancels. $\blacksquare$

So we may shift the logits by *any* constant before exponentiating, and one
choice is perfect: $c = \max_i z_i$. After the shift every exponent is at most
$0$, so every $e^{z_i - c}$ lies in $(0, 1]$ --- no overflow, ever --- and the
largest term equals exactly $1$, so the denominator lies in $[1, n]$ and can
neither overflow nor underflow to $0$. The cell below watches the naive route
produce `NaN` on logits that the shifted route handles without breaking a
sweat --- and checks that where both routes work, they agree to the last bit,
exactly as :eqref:`eq_mdl-opt-softmax-shift` promises:

```{.python .input #numerical-stability-conditioning-stable-softmax}
def softmax_naive(z):
    e = np.exp(z)
    return e / e.sum()

def softmax_stable(z):
    e = np.exp(z - z.max())        # shift by the max: largest exponent is 0
    return e / e.sum()

z = np.array([1.0, 2.0, 3.0], dtype=np.float32)
with np.errstate(over='ignore', invalid='ignore'):
    print('naive,  logits z      :', softmax_naive(z))
    print('naive,  logits z + 100:', softmax_naive(z + 100.0))
print('stable, logits z + 100:', softmax_stable(z + 100.0))
print('naive and stable agree where both work:',
      bool((softmax_naive(z) == softmax_stable(z + 100.0)).all()))
```

The shifted logits give back $(0.090, 0.245, 0.665)$ --- the same
probabilities as the small logits, bit for bit --- while the naive route
returns three `NaN`s. Every framework's `softmax` does this max-subtraction
internally; the trap is re-implementing it yourself, which is why the rule of
thumb is *never exponentiate a raw logit*.

### The Log-Sum-Exp Trick

The denominator of softmax has a logarithm important enough to earn its own
operator: the **log-sum-exp**

$$
\mathrm{lse}(\mathbf{z}) = \log \sum_{j=1}^n e^{z_j},
$$

which shows up as the normalizer of every exponential-family model, the
partition function of energy-based models, and (as we prove in a moment) the
backbone of cross-entropy. It inherits softmax's overflow problem and the same
shift repairs it --- this time as an *identity* rather than an invariance.

**Proposition (the shifted log-sum-exp is exact and safe).** *For every
$\mathbf{z} \in \mathbb{R}^n$ and every $c \in \mathbb{R}$,*

$$
\mathrm{lse}(\mathbf{z}) = c + \log \sum_{j=1}^n e^{z_j - c} .
$$
:eqlabel:`eq_mdl-opt-stable-lse`

*With the choice $c = \max_j z_j$ the sum lies in $[1, n]$, so the right-hand
side can neither overflow nor take $\log 0$; moreover*

$$
\max_j z_j \;\le\; \mathrm{lse}(\mathbf{z}) \;\le\; \max_j z_j + \log n .
$$

**Proof.** Factoring, $\sum_j e^{z_j} = e^{c} \sum_j e^{z_j - c}$; take
logarithms of both sides to get :eqref:`eq_mdl-opt-stable-lse` --- an exact
rewriting, valid for any $c$. With $c = \max_j z_j$, every term
$e^{z_j - c} \le 1$ and the maximizing term equals $1$, so the sum lies in
$[1, n]$ and its logarithm in $[0, \log n]$; adding $c$ gives the sandwich.
$\blacksquare$

The sandwich says lse is a *soft maximum* --- within $\log n$ of the true max
--- which is also the intuition for why it is convex
(:numref:`sec_mdl-convexity`; its gradient is exactly the softmax, a fact you
will prove in the exercises). Numerically, the identity gives us log-space
arithmetic for free: the **log-softmax** is

$$
\log \mathrm{softmax}(\mathbf{z})_i = z_i - \mathrm{lse}(\mathbf{z}),
$$
:eqlabel:`eq_mdl-opt-log-softmax`

a *subtraction of two safe quantities* that never materializes a probability
--- so probabilities that would underflow to $0$ (and then explode under
$\log$) simply never exist. Logits around $1000$ would overflow even float64;
in log space they are effortless:

```{.python .input #numerical-stability-conditioning-logsumexp}
def log_sum_exp(z):
    c = z.max()
    return c + np.log(np.exp(z - c).sum())

z = np.array([1000.0, 1001.0, 1002.0], dtype=np.float32)
with np.errstate(over='ignore'):
    print('naive  log(sum(exp(z))) :', np.log(np.exp(z).sum()))
print('stable log_sum_exp(z)   :', log_sum_exp(z))
log_p = z - log_sum_exp(z)             # log softmax, eq. above
print('log softmax             :', log_p)
print('probabilities sum to 1  :', f'{np.exp(log_p).sum():.6f}')
```

The naive route says `inf`; the stable route reports
$\mathrm{lse} = 1002.4076$ --- snugly inside the sandwich
$[1002, 1002 + \log 3]$ --- and the log-probabilities
$(-2.408, -1.408, -0.408)$ exponentiate back to a distribution summing to
$1.000013$: equal to $1$ up to the float32 spacing at magnitude $1000$,
which is all one can ask of subtractions performed there. This identity is precisely how :numref:`sec_mdl-naive_bayes`
multiplies hundreds of per-pixel probabilities without underflowing to zero:
sums of logs, never products of probabilities.

### Pass Logits, Not Probabilities

Cross-entropy is where the stakes are highest, because it is the loss
*gradient* that dies. For a label $y$,
:eqref:`eq_mdl-opt-log-softmax` gives

$$
-\log \mathrm{softmax}(\mathbf{z})_y = \mathrm{lse}(\mathbf{z}) - z_y,
$$
:eqlabel:`eq_mdl-opt-ce-from-logits`

computable *directly from the logits* with one stable lse and one
subtraction. This is what every framework's "from logits" loss does, and it is
why those APIs exist. The alternative --- compute probabilities first, then
take the log --- forces the loss through the representable range of
probabilities: a true-class probability below about $10^{-45}$ underflows
fp32 to exactly $0$, and the loss becomes $\infty$. The cell below pits the
two routes against each other on a two-class problem where the label is the
*unlikely* class, with logit gap $t$, so the true loss is
$\log(1 + e^{t}) \approx t$. This is the one computation in this section
where the four frameworks behave genuinely differently, so it is worth
running your tab and reading the matching paragraph below:

```{.python .input #numerical-stability-conditioning-cross-entropy}
#@tab mxnet
print('gap    CE from logits    CE via probabilities')
for t in [20.0, 60.0, 103.0, 104.0]:
    logits = mxnp.array([[0.0, t]])              # label = class 0, the
    from_logits = -npx.log_softmax(logits, axis=1)[0, 0]   # unlikely one
    from_probs = -mxnp.log(npx.softmax(logits, axis=1))[0, 0]
    print(f'{t:5.0f}  {float(from_logits):15.4f}  {float(from_probs):15.4f}')
```

```{.python .input #numerical-stability-conditioning-cross-entropy}
#@tab pytorch
print('gap    CE from logits    CE via probabilities')
for t in [20.0, 60.0, 103.0, 104.0]:
    logits = torch.tensor([[0.0, t]])
    y = torch.tensor([0])                        # label = the unlikely class
    from_logits = F.cross_entropy(logits, y)
    probs = F.softmax(logits, dim=1)             # stable softmax, then...
    from_probs = -torch.log(probs[0, y])         # ...take the log yourself
    print(f'{t:5.0f}  {from_logits.item():15.4f}  {from_probs.item():15.4f}')
```

```{.python .input #numerical-stability-conditioning-cross-entropy}
#@tab tensorflow
print('gap    CE from logits    CE via probabilities')
for t in [20.0, 60.0, 103.0, 104.0]:
    logits = tf.constant([[0.0, t]])
    y = tf.constant([0])                         # label = the unlikely class
    from_logits = tf.keras.losses.sparse_categorical_crossentropy(
        y, logits, from_logits=True)[0]
    from_probs = tf.keras.losses.sparse_categorical_crossentropy(
        y, tf.nn.softmax(logits), from_logits=False)[0]
    print(f'{t:5.0f}  {float(from_logits):15.4f}  {float(from_probs):15.4f}')
```

```{.python .input #numerical-stability-conditioning-cross-entropy}
#@tab jax
print('gap    CE from logits    CE via probabilities')
for t in [20.0, 60.0, 103.0, 104.0]:
    logits = jnp.array([[0.0, t]])
    y = jnp.array([0])                           # label = the unlikely class
    from_logits = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    from_probs = -jnp.log(jax.nn.softmax(logits)[0, 0])
    print(f'{t:5.0f}  {float(from_logits[0]):15.4f}  {float(from_probs):15.4f}')
```

:begin_tab:`mxnet`
The from-logits column reads $20$, $60$, $103$, $104$ --- exact at every gap.
The from-probabilities column matches until the probability $e^{-t}$ leaves
float32's normal range: at gap $103$ it has fallen among the *subnormals*,
where only a couple of significant bits survive, and the loss reads
$103.2789$ --- wrong in the first decimal place with no warning --- and at
gap $104$ the probability underflows to exactly $0$ and the loss is `inf`.
:end_tab:

:begin_tab:`pytorch`
The from-logits column reads $20$, $60$, $103$, $104$ --- exact at every gap.
The from-probabilities column matches until the probability $e^{-t}$ leaves
float32's normal range: at gap $103$ it has fallen among the *subnormals*,
where only a couple of significant bits survive, and the loss reads
$103.2789$ --- wrong in the first decimal place with no warning --- and at
gap $104$ the probability underflows to exactly $0$ and the loss is `inf`.
:end_tab:

:begin_tab:`tensorflow`
The from-logits column reads $20$, $60$, $103$, $104$ --- exact at every gap.
The from-probabilities column is more insidious here than in any other
framework: Keras clips probabilities to $[10^{-7},\, 1 - 10^{-7}]$ before
taking the log, so every row reads $16.1181 = -\log 10^{-7}$. There is no
`inf` and no `NaN` to alert you --- just a loss (and therefore a gradient)
that silently stopped depending on the model the moment the true loss
exceeded about $16$.
:end_tab:

:begin_tab:`jax`
The from-logits column reads $20$, $60$, $103$, $104$ --- exact at every gap.
The from-probabilities column matches at gaps $20$ and $60$, but by gap
$103$ the probability $e^{-t}$ has already underflowed to exactly $0$ under
XLA (which, unlike NumPy, does not linger in the subnormal range here), and
the loss is `inf` --- one gap *earlier* than the same experiment in PyTorch.
:end_tab:

The lesson generalizes far beyond this toy: losses, likelihoods, and
posteriors should live in log space from birth, and the conversion to
probabilities --- if it ever happens --- should be the *last* step, for human
eyes only. :numref:`sec_mdl-information_theory` analyzes what cross-entropy
*means*; :eqref:`eq_mdl-opt-ce-from-logits` is how it is *computed*, in every
framework, every time.

## Catastrophic Cancellation
:label:`subsec_mdl-catastrophic-cancellation`

### Subtraction Annihilates Digits

Overflow announces itself with `inf`; the subtler killer is silent.
Subtracting two nearly equal numbers is *exact* --- no new rounding error is
committed --- but it strips away the leading digits on which both numbers
agreed, leaving only their trailing digits, which is exactly where each
number's *previous* rounding errors live. If $a$ and $b$ are correct to
relative error $u$, their difference is correct only to relative error about

$$
\frac{|a| + |b|}{|a - b|}\; u,
$$
:eqlabel:`eq_mdl-opt-cancellation-factor`

an amplification factor that blows up precisely when $a \approx b$. The
phenomenon is called **catastrophic cancellation**, and a two-line experiment
shows both the disease and a cure. In float32, $1 + 10^{-8}$ rounds to
exactly $1$ (the increment is below $\varepsilon_{\text{mach}}/2$), so the
textbook expression $\log(1 + x)$ returns $0$ --- a $100\%$ relative error
--- while the library function `log1p`, which evaluates the *reformulated*
series around $0$, is exact. Likewise two floats agreeing to seven digits
leave a difference made of pure noise:

```{.python .input #numerical-stability-conditioning-log1p}
x = np.float32(1e-8)
print('float32 rounds 1 + x to     :', np.float32(1.0) + x)
print('log(1 + x) =', np.log(np.float32(1.0) + x),
      '   log1p(x) =', np.log1p(x))
a, b = np.float32(1.0002344), np.float32(1.0002341)
print('a - b in float32            :', a - b, '  (true value 3.0e-07)')
print('amplification (|a|+|b|)/|a-b| ~', f'{(a + b) / abs(a - b):.1e}')
```

The computed difference $2.384 \times 10^{-7}$ misses the true
$3.0 \times 10^{-7}$ by twenty percent: with an amplification factor near
$10^{7}$, float32's seven digits are gone in one subtraction. The catalogue
of standard victims is short and worth knowing --- $\log(1+x)$ and $e^x - 1$
near $0$ (use `log1p` and `expm1`), $1 - \cos x$ near $0$ (use
$2\sin^2(x/2)$), the quadratic formula near a double root, and finite
differences with too small a step, which is exactly the trade-off we
quantified in :numref:`sec_mdl-single_variable_calculus`. In every case the
cure is the same: *reformulate so that the subtraction happens analytically*,
where it costs nothing, rather than numerically, where it costs everything.
Higher precision merely postpones the cliff; reformulation removes it
:cite:`Higham.2002`.

### Case Study: Variance in One Pass

The classic cancellation bug in data science is the "computational formula"
for variance,

$$
\mathrm{Var}(x) = \mathbb{E}[x^2] - \mathbb{E}[x]^2,
$$

beloved because it needs one pass over the data. As algebra it is flawless; as
arithmetic it is a trap. For data with mean $\mu$ and standard deviation
$\sigma \ll |\mu|$, both terms are about $\mu^2$ while their difference is
$\sigma^2$, so :eqref:`eq_mdl-opt-cancellation-factor` predicts an error
amplification of about $\mu^2/\sigma^2$ --- and with $\mu = 10^9$ and
$\sigma = 1$ that is $10^{18}$: more than every digit float64 has. The naive
formula can even return a *negative* variance.

The repair is not float128 --- it is a reformulation due to
:citet:`Welford.1962` that keeps a running mean $m_k$ and a running sum of
*centered* squares $M_k = \sum_{i \le k} (x_i - m_k)^2$, so no large numbers
are ever subtracted:

$$
m_k = m_{k-1} + \frac{x_k - m_{k-1}}{k},
\qquad
M_k = M_{k-1} + (x_k - m_{k-1})(x_k - m_k).
$$
:eqlabel:`eq_mdl-opt-welford`

Note the two different factors in the $M_k$ update --- the deviation from the
*old* mean times the deviation from the *new* mean. That asymmetry is not a
typo; it is exactly what makes the recursion exact:

**Proposition (Welford's recursion is exact).** *With $m_0 = M_0 = 0$, the
recursions :eqref:`eq_mdl-opt-welford` satisfy, for every $k \ge 1$ and in
exact arithmetic,*

$$
m_k = \frac{1}{k} \sum_{i=1}^k x_i,
\qquad
M_k = \sum_{i=1}^k (x_i - m_k)^2 .
$$

**Proof.** The mean claim is the identity $k\, m_k = (k-1)\, m_{k-1} + x_k$,
immediate from the first recursion. For the second claim, induct on $k$ and
write $\delta = x_k - m_{k-1}$, so that $m_k - m_{k-1} = \delta/k$ and
$x_k - m_k = \delta\,(k-1)/k$. Splitting the new sum of squares at its last
term and re-centering the first $k-1$ terms around $m_{k-1}$,

$$
\sum_{i=1}^{k} (x_i - m_k)^2
= \sum_{i=1}^{k-1} \left( (x_i - m_{k-1}) - \tfrac{\delta}{k} \right)^2 + (x_k - m_k)^2
= M_{k-1} + (k-1)\tfrac{\delta^2}{k^2} + \tfrac{(k-1)^2}{k^2}\delta^2,
$$

where the cross term vanished because $\sum_{i \le k-1} (x_i - m_{k-1}) = 0$
and the inductive hypothesis named the first sum $M_{k-1}$. The two correction
terms combine to $\frac{k-1}{k}\,\delta^2 = \delta \cdot \delta \frac{k-1}{k}
= (x_k - m_{k-1})(x_k - m_k)$, which is precisely what the recursion adds.
$\blacksquare$

Numerically, every quantity Welford touches is of size $\sigma$, not $\mu$,
so the $\mu^2/\sigma^2$ amplification never materializes. The showdown ---
$10^5$ samples with mean $10^9$ and true variance $1$, all in float64:

```{.python .input #numerical-stability-conditioning-welford}
rng = np.random.default_rng(0)
x = 1e9 + rng.normal(0.0, 1.0, size=100_000)    # huge mean, unit variance

naive = (x**2).mean() - x.mean()**2             # one pass, cancels
two_pass = ((x - x.mean())**2).mean()           # subtract the mean first

mean, m2 = 0.0, 0.0                             # Welford: one pass, stable
for k, xk in enumerate(x, start=1):
    delta = xk - mean
    mean += delta / k                           # m_k
    m2 += delta * (xk - mean)                   # M_k
welford = m2 / len(x)

print(f'naive E[x^2] - E[x]^2 : {naive:12.6f}')
print(f'Welford, one pass     : {welford:12.6f}')
print(f'two-pass reference    : {two_pass:12.6f}')
```

The naive formula reports a variance in the *hundreds* --- off by a factor of
several hundred, in *double* precision, on a statistic every analyst computes
daily. The answer is pure amplified rounding noise, so even its sign depends
on the summation order of your NumPy build: the same cell printed $384$ under
one build and $-256$ --- a negative variance --- under another. Welford's
one-pass answer $1.000257$ agrees with the two-pass reference to eight
significant digits on every build. This recursion (and its batch-merging
generalization, which you will derive in the exercises) is how framework
`BatchNorm` layers and streaming-statistics utilities track running moments:
one pass, bounded memory, no cancellation.

## Conditioning: One Number, Two Consequences
:label:`subsec_mdl-conditioning-revisited`

### Backward and Forward Error

So far the *algorithms* were at fault, and rewriting them fixed everything.
The deeper half of numerical analysis begins with a different question: when a
computed answer is wrong, is the algorithm to blame --- or the problem?
:citet:`Higham.2002` makes the split precise with two definitions. The
**forward error** is what you care about: the distance between the computed
answer $\hat{\mathbf{x}}$ and the true answer $\mathbf{x}$. The **backward
error** is what the algorithm should be judged by: the size of the smallest
perturbation of the *inputs* for which $\hat{\mathbf{x}}$ would be exactly
correct --- "you got the right answer to a nearby question; how nearby?" An
algorithm with backward error at the rounding floor $u$ is called
**backward stable**: it did everything that can be asked of finite-precision
arithmetic, since merely *storing* the inputs already perturbs them by $u$.
Gaussian elimination with pivoting (`np.linalg.solve`) and the SVD are
backward stable; the naive variance formula is not.

What converts a small backward error into a possibly-large forward error is a
property of the *problem*, and for linear systems it is exactly the condition
number $\kappa(\mathbf{A}) = \sigma_1/\sigma_n$ of
:numref:`subsec_mdl-condition-number`.

**Proposition (forward error $\le$ condition number $\times$ backward
error).** *Let $\mathbf{A}$ be invertible, let
$\mathbf{A}\mathbf{x} = \mathbf{b}$, and suppose the computed
$\hat{\mathbf{x}}$ exactly solves a nearby system,
$(\mathbf{A} + \delta\mathbf{A})\,\hat{\mathbf{x}} = \mathbf{b}$ with
$\|\delta\mathbf{A}\| \le \varepsilon \|\mathbf{A}\|$. Then*

$$
\frac{\|\hat{\mathbf{x}} - \mathbf{x}\|}{\|\hat{\mathbf{x}}\|}
\;\le\; \kappa(\mathbf{A})\, \varepsilon .
$$
:eqlabel:`eq_mdl-opt-backward-forward`

**Proof.** Subtracting $\mathbf{A}\mathbf{x} = \mathbf{b}$ from
$(\mathbf{A} + \delta\mathbf{A})\hat{\mathbf{x}} = \mathbf{b}$ gives
$\mathbf{A}(\hat{\mathbf{x}} - \mathbf{x}) = -\delta\mathbf{A}\,\hat{\mathbf{x}}$,
hence $\hat{\mathbf{x}} - \mathbf{x} = -\mathbf{A}^{-1}\delta\mathbf{A}\,\hat{\mathbf{x}}$
and

$$
\|\hat{\mathbf{x}} - \mathbf{x}\|
\le \|\mathbf{A}^{-1}\|\, \|\delta\mathbf{A}\|\, \|\hat{\mathbf{x}}\|
\le \|\mathbf{A}^{-1}\|\, \|\mathbf{A}\|\, \varepsilon\, \|\hat{\mathbf{x}}\|
= \kappa(\mathbf{A})\,\varepsilon\, \|\hat{\mathbf{x}}\|,
$$

using the operator-norm identities $\|\mathbf{A}\| = \sigma_1$ and
$\|\mathbf{A}^{-1}\| = 1/\sigma_n$ from :numref:`sec_mdl-svd-low-rank`.
$\blacksquare$

(The error here is measured relative to $\hat{\mathbf{x}}$; for small
$\varepsilon$ this matches the error relative to $\mathbf{x}$ to first
order.) This one inequality is the *division of labor* of numerical
computing: the algorithm's job is to make $\varepsilon$ small (backward
stability gives $\varepsilon \approx u$), the problem's conditioning decides
how much of that smallness survives, and the user's rule of thumb falls out
by taking $\log_{10}$ of both sides:

$$
\textrm{correct digits in } \hat{\mathbf{x}}
\;\approx\; \textrm{digits carried by the format} \;-\; \log_{10} \kappa(\mathbf{A}).
$$

A backward-stable solve in float64 carries about $16$ digits, so
$\kappa = 10^{k}$ costs you $k$ of them --- and at $\kappa \approx 10^{16}$
the answer is pure noise even though the algorithm was flawless.

### The Condition Number of a Linear System

The same $\kappa$ also governs sensitivity to errors in the right-hand side
--- the data, in a least-squares problem. The bound is proved (with the
matching worst-case construction showing it is *tight*) in
:numref:`subsec_mdl-condition-number`; since it is two lines, we restate it
here in full.

**Proposition (relative perturbation bound).** *Let $\mathbf{A}$ be
invertible, $\mathbf{A}\mathbf{x} = \mathbf{b} \neq \mathbf{0}$, and
$\mathbf{A}(\mathbf{x} + \delta\mathbf{x}) = \mathbf{b} + \delta\mathbf{b}$.
Then*

$$
\frac{\|\delta\mathbf{x}\|}{\|\mathbf{x}\|}
\;\le\; \kappa(\mathbf{A})\, \frac{\|\delta\mathbf{b}\|}{\|\mathbf{b}\|} .
$$
:eqlabel:`eq_mdl-opt-perturbation-bound`

**Proof.** From $\mathbf{A}\,\delta\mathbf{x} = \delta\mathbf{b}$ we get
$\|\delta\mathbf{x}\| \le \|\delta\mathbf{b}\|/\sigma_n$, and from
$\mathbf{b} = \mathbf{A}\mathbf{x}$ we get
$\|\mathbf{b}\| \le \sigma_1 \|\mathbf{x}\|$, i.e.
$1/\|\mathbf{x}\| \le \sigma_1/\|\mathbf{b}\|$. Multiply the two
inequalities. $\blacksquare$

Time to watch $\kappa$ eat digits on the most famously ill-conditioned matrix
in the business: the **Hilbert matrix** $H_{ij} = 1/(i + j - 1)$, whose
condition number grows *exponentially* with $n$. We solve
$\mathbf{H}\mathbf{x} = \mathbf{b}$ with the answer rigged to be
$\mathbf{x} = \mathbf{1}$, and tabulate the forward error, the digits that
survive, and --- the punchline --- the *backward* error
$\|\mathbf{H}\hat{\mathbf{x}} - \mathbf{b}\| / (\|\mathbf{H}\|\,\|\hat{\mathbf{x}}\|)$:

```{.python .input #numerical-stability-conditioning-hilbert}
print(' n      kappa   log10 kappa   forward error  correct digits  backward error')
for n in [4, 6, 8, 10, 12]:
    i = np.arange(n)
    H = 1.0 / (1.0 + i[:, None] + i[None, :])   # Hilbert matrix
    x_true = np.ones(n)
    b = H @ x_true
    x_hat = np.linalg.solve(H, b)
    kappa = np.linalg.cond(H)
    fwd = np.linalg.norm(x_hat - x_true) / np.linalg.norm(x_true)
    bwd = (np.linalg.norm(H @ x_hat - b)
           / (np.linalg.norm(H, 2) * np.linalg.norm(x_hat)))
    print(f'{n:2d}  {kappa:9.1e}  {np.log10(kappa):8.1f}  {fwd:14.1e}  '
          f'{-np.log10(fwd):11.1f}     {bwd:11.1e}')
```

Read the table row by row against the rule of thumb. At $n = 4$,
$\log_{10}\kappa \approx 4.2$ and about $13$ digits survive of float64's
$16$; by $n = 8$, $\log_{10}\kappa \approx 10.2$ and about $7$ survive; at
$n = 12$, $\log_{10}\kappa \approx 16.2$ and barely one correct digit remains
--- the answer is essentially noise. (The trailing decimals of the error
column vary with your LAPACK build; the staircase does not.) Meanwhile the backward-error column never
leaves the $10^{-16}$ floor: *the solver is blameless at every row*. Each
$\hat{\mathbf{x}}$ exactly solves a system one part in $10^{16}$ away from
the one we posed; the Hilbert matrix simply maps that invisible perturbation
to a visible one, exactly as :eqref:`eq_mdl-opt-backward-forward` licenses it
to. Geometrically, large $\kappa$ means the level sets of
$\|\mathbf{A}\mathbf{x} - \mathbf{b}\|^2$ are extremely elongated ellipsoids
--- the same narrow valley, pictured in :numref:`fig_mdl-la-condition`, down
which gradient descent zig-zags. Sensitivity of the solve and slowness of the
descent are *one geometric fact* viewed from two sides.

### Why Normal Equations Square the Pain

Least squares offers a vivid demonstration that the *route* to a solution can
ruin conditioning even when the destination is fine. The textbook route to
$\min_{\mathbf{w}} \|\mathbf{A}\mathbf{w} - \mathbf{b}\|^2$ is the normal
equations $\mathbf{A}^\top\mathbf{A}\,\mathbf{w} = \mathbf{A}^\top\mathbf{b}$
--- which replace a solve governed by $\kappa(\mathbf{A})$ with one governed
by $\kappa(\mathbf{A}^\top\mathbf{A})$. That substitution is quadratically
bad:

**Proposition (normal equations square the condition number).** *For any
matrix $\mathbf{A}$ with full column rank,*

$$
\kappa(\mathbf{A}^\top\mathbf{A}) = \kappa(\mathbf{A})^2 .
$$
:eqlabel:`eq_mdl-opt-kappa-squared`

**Proof.** From the SVD $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$
we get $\mathbf{A}^\top\mathbf{A} = \mathbf{V}\boldsymbol{\Sigma}^2\mathbf{V}^\top$
(:numref:`subsec_mdl-svd-via-ata`): a symmetric positive-definite matrix
whose singular values are its eigenvalues $\sigma_i^2$. Hence
$\kappa(\mathbf{A}^\top\mathbf{A}) = \sigma_1^2/\sigma_n^2 = \kappa(\mathbf{A})^2$.
$\blacksquare$

By the digits rule of thumb, the normal equations lose $2\log_{10}\kappa$
digits where an SVD- or QR-based solve, which works on $\mathbf{A}$ directly,
loses $\log_{10}\kappa$. With $\kappa(\mathbf{A}) = 10^5$ the predicted gap
is five digits wide --- large enough to measure:

```{.python .input #numerical-stability-conditioning-normal-equations}
rng = np.random.default_rng(1)
m, n = 100, 10
U, _ = np.linalg.qr(rng.normal(size=(m, n)))    # random orthonormal columns
V, _ = np.linalg.qr(rng.normal(size=(n, n)))
sigma = np.logspace(0, -5, n)                   # kappa(A) = 10^5 by design
A = U * sigma @ V.T                             # A = U diag(sigma) V^T
w_true = rng.normal(size=n)
b = A @ w_true
print(f'kappa(A) = {np.linalg.cond(A):.1e}   '
      f'kappa(A^T A) = {np.linalg.cond(A.T @ A):.1e}')
w_ne = np.linalg.solve(A.T @ A, A.T @ b)        # normal equations
w_svd = np.linalg.lstsq(A, b, rcond=None)[0]    # SVD-based solve
for name, w in [('normal equations', w_ne), ('SVD (lstsq)     ', w_svd)]:
    err = np.linalg.norm(w - w_true) / np.linalg.norm(w_true)
    print(f'{name}: relative error {err:.1e}  '
          f'({-np.log10(err):.1f} correct digits)')
```

The printout confirms both the proposition ($\kappa$ of
$\mathbf{A}^\top\mathbf{A}$ is $10^{10}$, the square of $10^{5}$) and its
consequence: the normal equations recover about seven correct digits, the SVD
route about thirteen. Same problem, same data, same float64 --- five to six
digits of accuracy, right at the predicted $\log_{10}\kappa = 5$, forfeited
to a bad route. This is why `lstsq` exists, why frameworks solve
least-squares subproblems by QR or SVD, and why
:numref:`subsec_mdl-pseudoinverse` built the pseudoinverse from the SVD
rather than from $(\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top$.

### Ridge Regularization Is Preconditioning

When $\kappa(\mathbf{A})$ itself is the problem --- nearly collinear
features, a rank-deficient design --- no choice of route saves the original
problem. Ridge regularization changes the problem, and it changes it in
exactly the right direction. Minimizing
$\|\mathbf{A}\mathbf{w} - \mathbf{b}\|^2 + \lambda \|\mathbf{w}\|^2$ yields

$$
\mathbf{w}_\lambda = (\mathbf{A}^\top\mathbf{A} + \lambda \mathbf{I})^{-1} \mathbf{A}^\top \mathbf{b},
$$
:eqlabel:`eq_mdl-opt-ridge-solution`

and the added $\lambda\mathbf{I}$ acts directly on the spectrum.

**Proposition (ridge improves conditioning monotonically).** *Let $\mathbf{A}$
have singular values $\sigma_1 \ge \cdots \ge \sigma_n \ge 0$. For every
$\lambda > 0$ the matrix $\mathbf{A}^\top\mathbf{A} + \lambda\mathbf{I}$ is
symmetric positive definite (hence invertible, even when $\mathbf{A}$ is rank
deficient), with*

$$
\kappa(\mathbf{A}^\top\mathbf{A} + \lambda\mathbf{I})
= \frac{\sigma_1^2 + \lambda}{\sigma_n^2 + \lambda},
$$
:eqlabel:`eq_mdl-opt-ridge-kappa`

*which is strictly decreasing in $\lambda$ whenever $\sigma_1 > \sigma_n$ and
tends to $1$ as $\lambda \to \infty$.*

**Proof.** Writing $\mathbf{A}^\top\mathbf{A} = \mathbf{V}\boldsymbol{\Sigma}^2\mathbf{V}^\top$
as above, $\mathbf{A}^\top\mathbf{A} + \lambda\mathbf{I} =
\mathbf{V}(\boldsymbol{\Sigma}^2 + \lambda\mathbf{I})\mathbf{V}^\top$: the
same eigenvectors, every eigenvalue shifted up to $\sigma_i^2 + \lambda \ge
\lambda > 0$. Positive definiteness and :eqref:`eq_mdl-opt-ridge-kappa`
follow. For monotonicity, with $a = \sigma_1^2 > b = \sigma_n^2$,

$$
\frac{d}{d\lambda} \frac{a + \lambda}{b + \lambda}
= \frac{b - a}{(b + \lambda)^2} < 0,
$$

and as $\lambda \to \infty$ the ratio tends to $1$. $\blacksquare$

Adding $\lambda\mathbf{I}$ lifts the floor of the spectrum while barely moving
its ceiling: the elongated valley of the least-squares objective rounds out
into a bowl, as :numref:`fig_mdl-opt-conditioning-ellipse` shows. And because
$\kappa$ is *one number with two consequences*, this single shift pays twice.
The solve in :eqref:`eq_mdl-opt-ridge-solution` becomes more accurate ---
fewer digits lost, by the rule of thumb --- and gradient descent on the ridge
objective, whose contraction factor $(\kappa - 1)/(\kappa + 1)$ we derived in
:numref:`sec_mdl-gradient-based-optimization`, becomes faster. The cell below
measures both at once: for each $\lambda$ it computes
:eqref:`eq_mdl-opt-ridge-kappa` and *runs* gradient descent to a fixed
relative tolerance of $10^{-6}$, counting iterations.

![Level sets of the least-squares objective before and after adding the ridge term $\lambda \|\mathbf{w}\|^2$. The penalty lifts every eigenvalue of $\mathbf{A}^\top\mathbf{A}$ by $\lambda$, rounding the elongated valley into a bowl: the condition number drops from $\sigma_1^2/\sigma_n^2$ toward $1$, so linear solves lose fewer digits and gradient descent takes fewer steps --- the same $\lambda$ buys both.](../img/mdl-opt-conditioning-ellipse.svg)
:label:`fig_mdl-opt-conditioning-ellipse`

```{.python .input #numerical-stability-conditioning-ridge}
rng = np.random.default_rng(2)
m, n = 200, 20
U, _ = np.linalg.qr(rng.normal(size=(m, n)))
V, _ = np.linalg.qr(rng.normal(size=(n, n)))
sigma = np.logspace(0, -2, n)                   # kappa(A) = 100
A = U * sigma @ V.T
b = A @ rng.normal(size=n)

def gd_iterations(lam, tol=1e-6):
    """Iterations for GD on the ridge objective to reach tol."""
    M, g = A.T @ A + lam * np.eye(n), A.T @ b
    w_star = np.linalg.solve(M, g)
    mu, L = sigma.min()**2 + lam, sigma.max()**2 + lam
    eta, w = 2.0 / (L + mu), np.zeros(n)        # optimal fixed step size
    for k in range(1, 200_000):
        w -= eta * (M @ w - g)
        if np.linalg.norm(w - w_star) <= tol * np.linalg.norm(w_star):
            return k

lams = np.logspace(-4, 0, 9)
kappas = (sigma.max()**2 + lams) / (sigma.min()**2 + lams)
iters = np.array([gd_iterations(lam) for lam in lams])
print('iterations / kappa:', (iters / kappas).round(2))
d2l.plot(lams, [kappas, iters.astype(float)], 'lambda', None,
         legend=['kappa(A^T A + lambda I)', 'GD iterations to 1e-6'],
         xscale='log', yscale='log')
```

The two curves fall together across four orders of magnitude of $\lambda$ ---
from $\kappa \approx 5000$ and roughly $29{,}000$ iterations at
$\lambda = 10^{-4}$ down to $\kappa = 2$ and a dozen iterations at
$\lambda = 1$ --- and the printed ratio sits between $5.9$ and $6.3$
throughout: iteration count is a constant multiple of $\kappa$, the constant
being about $\tfrac12 \ln(1/\textrm{tol}) \approx 6.9$ predicted by the
$(\kappa - 1)/(\kappa + 1)$ contraction (slightly less here because a random
initial error is not perfectly aligned with the slowest eigendirection).
Regularization *is* preconditioning. Of course $\lambda$ is not free: it
biases the solution, shrinking $\mathbf{w}_\lambda$ toward $\mathbf{0}$ ---
and :numref:`sec_mdl-constrained-optimization-duality` showed the precise
sense in which the penalty $\lambda\|\mathbf{w}\|^2$ is the Lagrangian price
of a norm constraint. The practical reading of this section's last experiment
is the cheerful one: the $\lambda$ you were going to add anyway, for
statistical reasons, has been quietly stabilizing your arithmetic and
accelerating your optimizer the whole time.

## Summary

* Floating point is scientific notation with a finite mantissa: relative
  precision $\varepsilon_{\text{mach}}$ ($2^{-23}$ for fp32, $2^{-10}$ for
  fp16, $2^{-7}$ for bfloat16), absolute gaps that double at every power of
  two, and overflow/underflow cliffs ($e^x$ dies at $x \approx 88.7$ in fp32,
  $x \approx 11.1$ in fp16). Mixed precision is engineering around these
  cliffs: loss scaling fights fp16 underflow; bfloat16 trades mantissa for
  fp32's exponent range.
* Softmax is shift-invariant, so subtract the max before exponentiating;
  log-sum-exp with the same shift is an exact identity that never overflows;
  cross-entropy should be computed from logits as
  $\mathrm{lse}(\mathbf{z}) - z_y$. Pass logits to your loss --- the
  from-probabilities route ends in `inf`, `NaN`, or (worse) a silently
  clipped gradient.
* Catastrophic cancellation: subtracting nearly equal numbers amplifies
  existing relative error by $(|a|+|b|)/|a-b|$. The cure is reformulation,
  not precision: `log1p`/`expm1` near zero, and Welford's recursion
  $m_k = m_{k-1} + (x_k - m_{k-1})/k$,
  $M_k = M_{k-1} + (x_k - m_{k-1})(x_k - m_k)$ in place of
  $\mathbb{E}[x^2] - \mathbb{E}[x]^2$, which lost a factor of $400$ in
  float64.
* Backward error judges the algorithm, the condition number judges the
  problem, and forward error $\le \kappa \times$ backward error connects
  them: a backward-stable solve loses about $\log_{10}\kappa$ digits, as the
  Hilbert-matrix table showed digit for digit.
* Normal equations square the condition number
  ($\kappa(\mathbf{A}^\top\mathbf{A}) = \kappa(\mathbf{A})^2$ --- six digits
  lost vs. the SVD route in our experiment); ridge regularization moves it
  the other way, $\kappa = (\sigma_1^2 + \lambda)/(\sigma_n^2 + \lambda)$,
  monotonically improving both solve accuracy and the gradient-descent rate
  --- one number, two consequences, and one knob that helps both.

## Exercises

1. Compute $\varepsilon_{\text{mach}}$ for float32 by a halving loop: start
   from $e = 1$ and halve until $1 + e == 1$ in float32. Why does the loop
   exit at $e = 2^{-24}$ rather than at $2^{-23}$, and how does that relate
   to the unit roundoff $u$ in :eqref:`eq_mdl-opt-rounding-model`? Repeat the
   loop with bfloat16 arithmetic (emulated or native) and confirm $2^{-7}$.
2. Find all integer logits $x$ for which $e^x$ overflows in fp16 but not in
   fp32. A network's final layer emits activations of size $\approx 30$:
   explain, with the numbers from this section, why training it in fp16
   without loss scaling fails even though the softmax probabilities are
   perfectly representable.
3. Prove that $\nabla\, \mathrm{lse}(\mathbf{z}) = \mathrm{softmax}(\mathbf{z})$
   and use :numref:`sec_mdl-convexity` to conclude that lse is convex.
   Then show that the cross-entropy :eqref:`eq_mdl-opt-ce-from-logits` has
   gradient $\mathrm{softmax}(\mathbf{z}) - \mathbf{e}_y$ --- the cleanest
   gradient in deep learning, and another reason to compute the loss from
   logits.
4. Construct a small dataset (three numbers suffice) for which the naive
   variance formula returns a strictly *negative* number in float64, and
   verify that Welford's recursion :eqref:`eq_mdl-opt-welford` gets it right.
   Then derive the *pairwise merge* rule: given $(m, M, k)$ for two disjoint
   batches, express the combined statistics exactly --- this is how the
   computation parallelizes across devices.
5. Show that if $a$ and $b$ carry relative errors of size $u$, the relative
   error of the computed $a - b$ can be as large as
   $(|a| + |b|)\,u / |a - b|$, and that the subtraction itself adds no
   rounding error when $a/2 \le b \le 2a$ (Sterbenz's lemma --- prove it for
   floats with the same exponent).
6. Rewrite each of the following to avoid cancellation, and check one of them
   numerically in float32: $\sqrt{x + 1} - \sqrt{x}$ for large $x$;
   $1 - \cos x$ for small $x$; the smaller root of $ax^2 + bx + c = 0$ when
   $b^2 \gg 4ac$.
7. For the Hilbert experiment, compute the backward error of each solve with
   respect to the right-hand side, $\|\mathbf{H}\hat{\mathbf{x}} - \mathbf{b}\|/\|\mathbf{b}\|$,
   and verify that the forward error is bounded by
   :eqref:`eq_mdl-opt-perturbation-bound` applied to that perturbation. Where
   in the table is the bound tightest?
8. Let $\mathbf{A}$ have $\sigma_1 = 1$ and $\sigma_n = 10^{-3}$. Using
   :eqref:`eq_mdl-opt-ridge-kappa` and the GD contraction factor
   $(\kappa - 1)/(\kappa + 1)$ from
   :numref:`sec_mdl-gradient-based-optimization`, compute the condition
   number and the predicted iteration count (to relative error $10^{-6}$)
   for $\lambda \in \{0, 10^{-4}, 10^{-2}\}$. Then explain, via
   :numref:`sec_mdl-constrained-optimization-duality`, which constrained
   problem each $\lambda$ implicitly solves.

## Discussions

This section closes the loop the chapter opened: the condition number that set
gradient descent's speed in :numref:`sec_mdl-gradient-based-optimization` is
the same number that sets a linear solve's accuracy, and ridge regularization
--- the Lagrangian twin of a norm constraint, per
:numref:`sec_mdl-constrained-optimization-duality` --- improves both at once.
The stability toolkit travels well beyond this chapter: max-subtraction and
log-sum-exp power every softmax and attention layer; log-space arithmetic is
what makes naive Bayes (:numref:`sec_mdl-naive_bayes`) and probabilistic
inference generally feasible; the from-logits cross-entropy is the computation
:numref:`sec_mdl-information_theory` analyzes; and Welford-style running
moments live inside batch normalization. For the full theory --- error
analysis of every algorithm in this book's linear-algebra substrate --- the
reference remains :citet:`Higham.2002`.

[Discussions](https://d2l.discourse.group/t/numerical-stability-and-conditioning)

<!-- slides -->

::: {.slide title="Numerical Stability and Conditioning"}
The math can be right and the loss still goes to `NaN`. Two questions
assign the blame:

- Did the **algorithm** solve a nearby problem? (backward error)
- Do nearby problems have wildly different answers? (conditioning)

Fixes are *reformulations*, not more bits: max-subtraction, log-space,
Welford, ridge.
:::

::: {.slide title="Floating point: a number system with gaps"}
$x = (-1)^s (1.m)_2\, 2^e$: huge range, fixed *relative* precision
$\varepsilon_{\text{mach}}$ --- fp32: $2^{-23}$, fp16: $2^{-10}$,
bfloat16: $2^{-7}$ (fp32's exponent range, 7 mantissa bits).

@fig:mdl-opt-fp-number-line

. . .

@numerical-stability-conditioning-finfo
:::

::: {.slide title="Stable softmax: subtract the max"}
$e^x$ overflows fp32 at $x \approx 88.7$ --- but softmax is
shift-invariant, so shift by $c = \max_i z_i$: every exponent
$\le 0$, denominator in $[1, n]$.

@numerical-stability-conditioning-stable-softmax

. . .

Same shift makes log-sum-exp exact and safe:
$\mathrm{lse}(\mathbf{z}) = c + \log \sum_j e^{z_j - c}$.
:::

::: {.slide title="Pass logits, not probabilities"}
Cross-entropy from logits is one stable lse:
$-\log \mathrm{softmax}(\mathbf{z})_y = \mathrm{lse}(\mathbf{z}) - z_y$.
Via probabilities, the loss underflows, clips, or explodes ---
differently in every framework:

@numerical-stability-conditioning-cross-entropy
:::

::: {.slide title="Catastrophic cancellation and Welford"}
Subtracting near-equal numbers amplifies old rounding error by
$(|a|+|b|)/|a-b|$. Cure = reformulate:
$m_k = m_{k-1} + \frac{x_k - m_{k-1}}{k}$,
$M_k = M_{k-1} + (x_k - m_{k-1})(x_k - m_k)$:

@numerical-stability-conditioning-welford
:::

::: {.slide title="Conditioning: digits lost = log10 kappa"}
Forward error $\le \kappa \times$ backward error. A backward-stable
solver is blameless --- the *problem* amplifies. Hilbert matrices,
digit by digit:

@numerical-stability-conditioning-hilbert
:::

::: {.slide title="Ridge is preconditioning"}
$\kappa(\mathbf{A}^\top\mathbf{A} + \lambda\mathbf{I}) =
\frac{\sigma_1^2 + \lambda}{\sigma_n^2 + \lambda}$: monotone down in
$\lambda$. One knob, two payoffs --- accurate solves *and* fast GD:

@numerical-stability-conditioning-ridge

. . .

- Never exponentiate a raw logit; live in log space.
- Reformulate subtractions away; Welford over
  $\mathbb{E}[x^2] - \mathbb{E}[x]^2$.
- $\kappa$: one number, two consequences; ridge lowers it.
:::
