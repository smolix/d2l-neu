# Discrete Diffusion
:label:`sec_diffusion-discrete`

Every model in this chapter has added Gaussian noise to real numbers:
coordinates of points, pixel intensities, latents of an autoencoder. The
language models of :numref:`sec_gpt` work with tokens, write text one
token at a time from left to right, and never revise a token once it is
written. Adding a small real number to the index of the word "cat" does
not produce a slightly noisy cat. Can text, code or a quantized image still be generated
the way this chapter generates images, many positions at once, each
conditioned on both sides of a gap, with a likelihood bound to train
against? :citet:`Austin.Johnson.Ho.ea.2021` made
the noise a Markov chain on the vocabulary and found that the best chain
erases tokens, replacing them by a mask symbol.
:citet:`Sahoo.Arriola.Schiff.ea.2024` and :citet:`Shi.Han.Wang.ea.2024`
showed that the bound of this chain is a weighted form of the
masked-language-model loss, the objective of today's diffusion language
models :cite:`Nie.Zhu.You.ea.2025`.

The variational bound of :numref:`sec_diffusion-design-space` carries over
unchanged; with erasure as the noise it becomes a cross-entropy on the
erased positions. The sampler reveals many tokens per network evaluation,
at a cost that two tokens make exact, and we test both on a quantized
version of the running example and on binary images.

```{.python .input #discrete-diffusion}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #discrete-diffusion}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #discrete-diffusion-1}
%%tab pytorch, jax
from scipy.special import ndtr, xlogy
```

## Corrupting Tokens

Tokens need a forward process with closed-form marginals and posteriors
that ends in a state carrying no information.

### Transition Matrices

Let a token take one of $K$ values, written as a one-hot row vector
$\mathbf{x} \in \{0, 1\}^K$. Noise for a token is a random change of its
value, given by *transition matrices* $\mathbf{Q}_t \in [0, 1]^{K \times K}$
whose rows sum to one: $[\mathbf{Q}_t]_{ij}$ is the probability of moving
from value $i$ to value $j$ at step $t$. Here $t = 1, \ldots, T$ counts
steps, as $k$ did for the Gaussian chain, and the network receives $t/T$,
the time on the chapter's clock. These are the row-stochastic
matrices of Markov chain Monte Carlo (:numref:`sec_mdl-bayes-mcmc`), with
the same convention that a distribution is a row vector multiplied by
$\mathbf{Q}_t$ from the right. Hence

$$
q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathrm{Cat}(\mathbf{x}_t;\ \mathbf{x}_{t-1} \mathbf{Q}_t),
$$
:eqlabel:`eq_diffusion-discrete-forward`

where $\mathrm{Cat}(\mathbf{x}; \mathbf{p})$ is the categorical
distribution with probability vector $\mathbf{p}$, here the row of
$\mathbf{Q}_t$ selected by the current value. A sequence of $L$ tokens is
corrupted by running the chain at every position independently. Composing
two transitions multiplies their matrices, so by induction on $t$ the
marginal after $t$ steps is

$$
q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathrm{Cat}(\mathbf{x}_t;\ \mathbf{x}_0 \bar{\mathbf{Q}}_t),
\qquad \bar{\mathbf{Q}}_t = \mathbf{Q}_1 \mathbf{Q}_2 \cdots \mathbf{Q}_t.
$$
:eqlabel:`eq_diffusion-discrete-marginal`

The posterior of one step given both of its ends follows from Bayes' rule
and the Markov property,

$$
q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)
= \frac{q(\mathbf{x}_t \mid \mathbf{x}_{t-1})\, q(\mathbf{x}_{t-1} \mid \mathbf{x}_0)}{q(\mathbf{x}_t \mid \mathbf{x}_0)}
= \mathrm{Cat}\!\left(\mathbf{x}_{t-1};\ \frac{\mathbf{x}_t \mathbf{Q}_t^\top \odot \mathbf{x}_0 \bar{\mathbf{Q}}_{t-1}}{\mathbf{x}_0 \bar{\mathbf{Q}}_t \mathbf{x}_t^\top}\right),
$$
:eqlabel:`eq_diffusion-discrete-posterior`

where $\odot$ is the elementwise product: each possible predecessor is
weighted by its probability given the start and by its probability of
leading to the observed $\mathbf{x}_t$.

A binary token that flips with probability $0.1$ at every step shows the
posterior at work. Here
$\mathbf{Q}_1 = \mathbf{Q}_2 = \left(\begin{smallmatrix} 0.9 & 0.1 \\ 0.1 & 0.9 \end{smallmatrix}\right)$.
Start from the first value, $\mathbf{x}_0 = (1, 0)$, and suppose that
after two steps the token shows the second value, $\mathbf{x}_2 = (0, 1)$.
This happens with probability $0.9 \cdot 0.1 + 0.1 \cdot 0.9 = 0.18$, the
denominator. For the middle state,
$\mathbf{x}_2 \mathbf{Q}_2^\top = (0.1, 0.9)$ and
$\mathbf{x}_0 \bar{\mathbf{Q}}_1 = (0.9, 0.1)$. Their elementwise product
is $(0.09, 0.09)$, and dividing by $0.18$ gives $(1/2, 1/2)$. The token
flipped once, and given both ends the flip is equally likely to have
happened at the first step or at the second.

### The Absorbing Kernel

The bound needs a closed form for $\bar{\mathbf{Q}}_t$, and the sampler a
known starting state. The *absorbing kernel* of
:citet:`Austin.Johnson.Ho.ea.2021` provides both. It enlarges the vocabulary by
one value, the *mask* $[\textrm{M}]$, and moves each token there with
probability $\beta_t$ at step $t$; a masked token stays masked:

$$
\mathbf{Q}_t = (1 - \beta_t)\, \mathbf{I} + \beta_t\, \mathbf{1} \mathbf{e}_{\textrm{M}}^\top,
$$
:eqlabel:`eq_diffusion-absorbing-kernel`

with $\mathbf{1}$ the all-ones column vector and $\mathbf{e}_{\textrm{M}}$
the one-hot column vector of the mask, so that every row of
$\mathbf{1} \mathbf{e}_{\textrm{M}}^\top$ puts all its mass on the mask.

Products of such matrices collapse. Write
$\mathbf{P} = \mathbf{1} \mathbf{e}_{\textrm{M}}^\top$. Since
$\mathbf{e}_{\textrm{M}}^\top \mathbf{1} = 1$, we have
$\mathbf{P}^2 = \mathbf{P}$, and therefore
$(a\mathbf{I} + b\mathbf{P})(c\mathbf{I} + d\mathbf{P}) = ac\, \mathbf{I} + (ad + bc + bd)\, \mathbf{P}$.
When $a + b = c + d = 1$ the second coefficient equals $1 - ac$, so the
coefficient of $\mathbf{I}$ multiplies along the chain. Let
$\bar{\alpha}_t = \prod_{s \leq t} (1 - \beta_s)$, the same product as for
the Gaussian chain of :numref:`sec_diffusion-gaussian-paths`. Here it has a
direct meaning, the probability that a token survives $t$ steps without
being masked. Then

$$
\bar{\mathbf{Q}}_t = \bar{\alpha}_t\, \mathbf{I} + (1 - \bar{\alpha}_t)\, \mathbf{1}\mathbf{e}_{\textrm{M}}^\top.
$$
:eqlabel:`eq_diffusion-discrete-qbar`

The chain never confuses one token with another; it only erases. A
schedule with $\bar{\alpha}_T = 0$ ends at the fully masked sequence, which
carries no information about the data, as the Gaussian chain ends at pure
noise (:numref:`fig_diffusion-masking`). The other common choice, the
*uniform kernel*, resamples a token uniformly from the vocabulary
:cite:`Hoogeboom.Nielsen.Jaini.ea.2021` (Exercise
:exref:`ex_discrete-diffusion-uniform-kernel`).

![The absorbing chain on nine tokens. Forward, each token is independently replaced by the mask with probability $\beta_t$ per step and stays masked; after $T$ steps every token is masked. Reverse, the model predicts masked tokens from the partially masked sequence, and a revealed token is never changed.](../img/mdl-diffusion-masking.svg)
:label:`fig_diffusion-masking`

We use the absorbing kernel from now on: it gave the best likelihoods on
text in the comparisons of Austin et al., and its posterior is the
simplest.

## The Masked Bound

### The Same Bound, a Different Target

The reverse model is a Markov chain
$p_{\boldsymbol{\theta}}(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$, started from
the all-mask sequence. :numref:`sec_diffusion-design-space` bounded the
negative log-likelihood of such a chain by

$$
-\log p_{\boldsymbol{\theta}}(\mathbf{x}_0) \leq L_{\textrm{VLB}}
= D_{\mathrm{KL}}\big(q(\mathbf{x}_T \mid \mathbf{x}_0)\, \|\, p(\mathbf{x}_T)\big)
+ \sum_{t=2}^{T} \mathbb{E}_q\Big[D_{\mathrm{KL}}\big(q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)\, \|\, p_{\boldsymbol{\theta}}(\mathbf{x}_{t-1} \mid \mathbf{x}_t)\big)\Big]
+ \mathbb{E}_q\big[-\log p_{\boldsymbol{\theta}}(\mathbf{x}_0 \mid \mathbf{x}_1)\big]:
$$

the evidence lower bound of :numref:`sec_mdl-latent-em-elbo` with the
forward chain as the variational distribution. Its derivation used only
the Markov property and Bayes' rule, so it holds for tokens as it stands.

What changes is the target. For Gaussian noise the network predicts the
noise or the clean point; for tokens the natural target is the clean token
itself. Following :citet:`Austin.Johnson.Ho.ea.2021`, the network outputs,
for each position $i$, a distribution
$p_{\boldsymbol{\theta}}(\mathbf{x}_0^i \mid \mathbf{x}_t)$ over the clean
values given the whole corrupted sequence, and the reverse transition
averages the true posterior over this prediction,

$$
p_{\boldsymbol{\theta}}(\mathbf{x}_{t-1} \mid \mathbf{x}_t)
= \sum_{\hat{\mathbf{x}}_0} q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \hat{\mathbf{x}}_0)\, p_{\boldsymbol{\theta}}(\hat{\mathbf{x}}_0 \mid \mathbf{x}_t),
$$
:eqlabel:`eq_diffusion-x0-parameterization`

position by position, with the prediction at an unmasked position taken
to be the observed token. A perfect prediction makes this the true reverse
transition. Austin et al. renormalize the average differently; for the
absorbing kernel the two forms agree (Exercise
:exref:`ex_discrete-diffusion-two-forms`).

### The Posterior of the Absorbing Chain

**Proposition (posterior of the absorbing chain).** *Under the absorbing
kernel with $\beta_1 > 0$, so that $\bar{\alpha}_t < 1$ for every
$t \geq 1$, consider one position with clean value $\mathbf{x}_0 \neq [\textrm{M}]$.
If $\mathbf{x}_t = \mathbf{x}_0$, then $\mathbf{x}_{t-1} = \mathbf{x}_0$
with probability one. If $\mathbf{x}_t = [\textrm{M}]$, then*

$$
q(\mathbf{x}_{t-1} = \mathbf{x}_0 \mid \mathbf{x}_t = [\textrm{M}], \mathbf{x}_0) = \frac{\bar{\alpha}_{t-1} - \bar{\alpha}_t}{1 - \bar{\alpha}_t},
\qquad
q(\mathbf{x}_{t-1} = [\textrm{M}] \mid \mathbf{x}_t = [\textrm{M}], \mathbf{x}_0) = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t},
$$
:eqlabel:`eq_diffusion-masked-posterior`

*and no other value has positive probability.*

**Proof.** Evaluate :eqref:`eq_diffusion-discrete-posterior` with
:eqref:`eq_diffusion-absorbing-kernel` and
:eqref:`eq_diffusion-discrete-qbar`. The vector
$\mathbf{x}_0 \bar{\mathbf{Q}}_{t-1}$ puts mass $\bar{\alpha}_{t-1}$ on
$\mathbf{x}_0$ and $1 - \bar{\alpha}_{t-1}$ on the mask, so only these two
predecessors are possible. If $\mathbf{x}_t = \mathbf{x}_0$, the vector
$\mathbf{x}_t \mathbf{Q}_t^\top$ has the entry $1 - \beta_t$ at
$\mathbf{x}_0$ and the entry $0$ at the mask, because a mask never returns
to a token, and the posterior is a point mass. If
$\mathbf{x}_t = [\textrm{M}]$, the entries are $\beta_t$ at
$\mathbf{x}_0$ and $1$ at the mask, so the unnormalized posterior is
$\bar{\alpha}_{t-1}\beta_t$ on $\mathbf{x}_0$ and $1 - \bar{\alpha}_{t-1}$
on the mask. Their sum, $1 - \bar{\alpha}_{t-1}(1 - \beta_t) = 1 - \bar{\alpha}_t$,
is the denominator $q(\mathbf{x}_t = [\textrm{M}] \mid \mathbf{x}_0)$, and
$\bar{\alpha}_{t-1}\beta_t = \bar{\alpha}_{t-1} - \bar{\alpha}_t$.
$\blacksquare$

One step back, a masked token either stays masked or reverts to its clean
value. The reverse model :eqref:`eq_diffusion-x0-parameterization`
therefore copies the unmasked positions of $\mathbf{x}_t$ and replaces each
masked one, independently, by a draw from
$p_{\boldsymbol{\theta}}(\mathbf{x}_0^i \mid \mathbf{x}_t)$ with the
probability of reverting.

### A Weighted Cross-Entropy
:label:`subsec_diffusion-masked-ce`

**Proposition (masked diffusion bound).** *For the absorbing kernel with
$\bar{\alpha}_0 = 1$, $\bar{\alpha}_t < 1$ for $t \geq 1$, and
$\bar{\alpha}_T = 0$, and the reverse model just described,*

$$
L_{\textrm{VLB}}
= \sum_{t=1}^{T} \frac{\bar{\alpha}_{t-1} - \bar{\alpha}_t}{1 - \bar{\alpha}_t}\;
\mathbb{E}_{q(\mathbf{x}_t \mid \mathbf{x}_0)}\Big[ \sum_{i \,:\, \mathbf{x}_t^i = [\textrm{M}]} -\log p_{\boldsymbol{\theta}}(\mathbf{x}_0^i \mid \mathbf{x}_t) \Big].
$$
:eqlabel:`eq_diffusion-masked-loss`

**Proof.** The prior term vanishes: $\bar{\alpha}_T = 0$ makes
$q(\mathbf{x}_T \mid \mathbf{x}_0)$ the point mass on the all-mask
sequence, which is also $p(\mathbf{x}_T)$. Both
$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$ and
$p_{\boldsymbol{\theta}}(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ factorize
over positions, so each divergence is a sum of per-position divergences.
At an unmasked position both distributions are the same point mass and
the divergence is zero. At a masked position write
$w_t = (\bar{\alpha}_{t-1} - \bar{\alpha}_t) / (1 - \bar{\alpha}_t)$. The
true posterior puts $1 - w_t$ on the mask and $w_t$ on $\mathbf{x}_0^i$;
the model puts $1 - w_t$ on the mask and
$w_t\, p_{\boldsymbol{\theta}}(\cdot \mid \mathbf{x}_t)$ on the tokens;
hence

$$
D_{\mathrm{KL}}
= (1 - w_t) \log \frac{1 - w_t}{1 - w_t}
+ w_t \log \frac{w_t}{w_t\, p_{\boldsymbol{\theta}}(\mathbf{x}_0^i \mid \mathbf{x}_t)}
= -w_t \log p_{\boldsymbol{\theta}}(\mathbf{x}_0^i \mid \mathbf{x}_t).
$$

This accounts for $t = 2, \ldots, T$. The reconstruction term is
$\mathbb{E}[-\log p_{\boldsymbol{\theta}}(\mathbf{x}_0 \mid \mathbf{x}_1)]$.
Since $w_1 = (1 - \bar{\alpha}_1)/(1 - \bar{\alpha}_1) = 1$, the model
replaces every position masked at step $1$ by a draw from
$p_{\boldsymbol{\theta}}(\cdot \mid \mathbf{x}_1)$ and copies the rest, so
this term equals the sum of
$-\log p_{\boldsymbol{\theta}}(\mathbf{x}_0^i \mid \mathbf{x}_1)$ over the
masked positions, which is the $t = 1$ term of the sum. $\blacksquare$

Divided by $\log 2$ and the number of tokens, the bound is a code
length in bits per token (:numref:`sec_mdl-coding-compression`), and its
exponential in nats per token bounds the perplexity
(:numref:`subsec_mdl-perplexity`).

Under the *linear schedule* $\bar{\alpha}_t = 1 - t/T$, a fraction $t/T$
of the tokens is masked at step $t$, and $w_t = (1/T)/(t/T) = 1/t$. An
error on a heavily masked sequence is charged less than the same error on
a lightly masked one. Since the number of masked positions grows in
proportion to $t$, every step carries the same expected weighted count of
masked positions, $L/T$ for $L$ tokens; the steps differ only in how hard
their masked tokens are to predict.

The sum is also a Riemann sum. Write
$\ell(\mathbf{x}_0, \mathbf{x})$ for the cross-entropy on the masked
positions of $\mathbf{x}$, the inner sum of
:eqref:`eq_diffusion-masked-loss`, and let $T \to \infty$ with
$\bar{\alpha}_t = \bar{\alpha}(t/T)$ for a decreasing function on the
chapter's clock, data at time $0$ and the all-mask sequence at time $1$.
Then $w_t \approx -\bar{\alpha}'(\tau)/(T(1 - \bar{\alpha}(\tau)))$ at
$\tau = t/T$, and the substitution $u = 1 - \bar{\alpha}(\tau)$, the
masking probability, turns the limit of the bound into

$$
\int_0^1 \frac{-\bar{\alpha}'(\tau)}{1 - \bar{\alpha}(\tau)}\, \mathbb{E}\big[\ell(\mathbf{x}_0, \mathbf{x}_\tau)\big]\, d\tau
= \int_0^1 \frac{1}{u}\, \mathbb{E}_u\big[\ell(\mathbf{x}_0, \mathbf{x})\big]\, du,
$$

where $\mathbb{E}_u$ masks each token independently with probability $u$.
For a network told the masking probability rather than the time, the
limit does not depend on the schedule, the discrete counterpart of the
schedule invariance of the Gaussian bound
(:numref:`sec_diffusion-design-space`)
:cite:`Kingma.Salimans.Poole.ea.2021`.

The function `masked_loss` estimates the bound with one step $t$ per
sequence, given the schedule `alpha_bar` (the values
$\bar{\alpha}_0, \ldots, \bar{\alpha}_T$) and the index `mask_id` of the mask; the
factor $T$ makes a uniformly drawn step unbiased for the sum over steps.
The network receives $t/T$, the time on the chapter's clock, and the sum
runs over every axis after the first, so pairs of tokens and images share
the function.

```{.python .input #discrete-diffusion-a-weighted-cross-entropy}
%%tab pytorch
def masked_loss(model, x0, t, alpha_bar, mask_id):
    """One-draw estimate of the masked bound per sequence, in nats."""
    T = len(alpha_bar) - 1
    shape = (-1,) + (1,) * (x0.dim() - 1)      # broadcast over positions
    masked = (torch.rand(x0.shape, device=x0.device)
              < (1 - alpha_bar[t]).view(shape))  # with prob. 1 - abar_t
    logits = model(torch.where(masked, mask_id, x0), t / T)
    nll = -torch.log_softmax(logits, -1).gather(-1, x0[..., None])[..., 0]
    w = (alpha_bar[t - 1] - alpha_bar[t]) / (1 - alpha_bar[t])  # 1/t if linear
    return (T * w.view(shape) * nll * masked).flatten(1).sum(1)
```

```{.python .input #discrete-diffusion-a-weighted-cross-entropy}
%%tab jax
def masked_loss(model, x0, t, alpha_bar, mask_id, key):
    """One-draw estimate of the masked bound per sequence, in nats."""
    T = len(alpha_bar) - 1
    shape = (-1,) + (1,) * (x0.ndim - 1)       # broadcast over positions
    masked = (jax.random.uniform(key, x0.shape)
              < (1 - alpha_bar[t]).reshape(shape))  # with prob. 1 - abar_t
    logits = model(jnp.where(masked, mask_id, x0), t / T)
    nll = -jnp.take_along_axis(jax.nn.log_softmax(logits), x0[..., None],
                               -1)[..., 0]
    w = (alpha_bar[t - 1] - alpha_bar[t]) / (1 - alpha_bar[t])  # 1/t if linear
    return (T * w.reshape(shape) * nll * masked).reshape(len(x0), -1).sum(1)
```

## Sampling by Unmasking

### Parallel Reveals and the Independence Error

At step $t$, one forward pass yields
$p_{\boldsymbol{\theta}}(\cdot \mid \mathbf{x}_t)$ at every masked
position. Each masked position draws a candidate token and accepts it with
probability $w_t$. As for the Gaussian
samplers of :numref:`sec_diffusion-samplers`, fewer steps are possible:
choose levels
$T = t_0 > t_1 > \cdots > t_S = 0$. The coarser chain has the same
marginals :eqref:`eq_diffusion-discrete-qbar`, so the proposition applies
to it unchanged, and a jump from $t_k$ to $t_{k+1}$ reveals each masked
token with probability
$(\bar{\alpha}_{t_{k+1}} - \bar{\alpha}_{t_k}) / (1 - \bar{\alpha}_{t_k})$.
The function `unmask` returns the list of states, the last of which is
the sample.

```{.python .input #discrete-diffusion-parallel-reveals-and-the-independence-error}
%%tab pytorch
@torch.inference_mode()
def unmask(model, shape, steps, alpha_bar, mask_id):
    """Reveal tokens from the all-mask state in `steps` jumps; all states."""
    T = len(alpha_bar) - 1
    levels = torch.linspace(T, 0, steps + 1).round().long().tolist()
    states = [torch.full(shape, mask_id, device=alpha_bar.device)]
    for t, s in zip(levels[:-1], levels[1:]):
        z = states[-1]
        logits = model(z, torch.full(shape[:1], t / T, device=z.device))
        draw = torch.distributions.Categorical(logits=logits).sample()
        p = (alpha_bar[s] - alpha_bar[t]) / (1 - alpha_bar[t])  # reveal prob.
        reveal = (z == mask_id) & (torch.rand(shape, device=z.device) < p)
        states.append(torch.where(reveal, draw, z))
    return states
```

```{.python .input #discrete-diffusion-parallel-reveals-and-the-independence-error}
%%tab jax
@nnx.jit
def reveal(model, z, t, s, alpha_bar, mask_id, key):
    """One jump of the sampler, from level t down to level s."""
    T, (k1, k2) = len(alpha_bar) - 1, jax.random.split(key)
    draw = jax.random.categorical(k1, model(z, jnp.full(z.shape[:1], t / T)))
    p = (alpha_bar[s] - alpha_bar[t]) / (1 - alpha_bar[t])  # reveal prob.
    accept = (z == mask_id) & (jax.random.uniform(k2, z.shape) < p)
    return jnp.where(accept, draw, z)

def unmask(model, shape, steps, alpha_bar, mask_id, key):
    """Reveal tokens from the all-mask state in `steps` jumps; all states."""
    T = len(alpha_bar) - 1
    levels = [round(l) for l in np.linspace(T, 0, steps + 1)]
    states = [jnp.full(shape, mask_id)]
    for t, s, k in zip(levels[:-1], levels[1:], jax.random.split(key, steps)):
        states.append(reveal(model, states[-1], t, s, alpha_bar, mask_id, k))
    return states
```

Tokens revealed in the same jump are drawn independently, each from its
own predicted distribution, because the network's prediction is a product
over positions; tokens revealed in different jumps are drawn one after
another, each conditioned on those already revealed. Two tokens show the
cost exactly. With $S$ equally spaced levels under the linear schedule, a
token is revealed in any given jump with probability $1/S$, so both tokens
are revealed in the same jump with probability $S \cdot (1/S)^2 = 1/S$.
Take an *exact* network, whose prediction at each masked position is the
true conditional of that token given the unmasked ones. Its sampler draws
the pair from the mixture

$$
\left(1 - \tfrac{1}{S}\right) p + \tfrac{1}{S}\, p^1 \otimes p^2
$$

of the joint distribution $p$ of the two tokens and the product of its
marginals $p^1$ and $p^2$. In total variation, half the summed absolute
difference of two probability tables (:numref:`sec_mdl-tv-pinsker`), the
mixture lies $\mathrm{TV}(p^1 \otimes p^2, p) / S$ from $p$: the error falls
as one over the number of steps and grows with the dependence between the
tokens.

The chapter's running example makes this concrete: the mixture of three
Gaussians in the plane with weights $0.5$, $0.3$ and $0.2$ and standard
deviation $0.5$ (:numref:`sec_diffusion-denoising`). Quantizing the plane
onto a $32 \times 32$ grid over $[-5, 5]^2$ turns each sample into a pair of tokens, the
indices of its cell along the two axes. Cell probabilities factorize
into one-dimensional Gaussian integrals, so the cell below computes the
true table $p$ exactly, with its entropy,
the mutual information of the two tokens, and the error of an exact
network's $S$-step sampler.

```{.python .input #discrete-diffusion-parallel-reveals-and-the-independence-error-2}
%%tab pytorch, jax
mix = d2l.GaussianMixture.three_modes()
means, weights = np.float64(mix.means), np.float64(mix.weights)
K, T = 32, 32                 # vocabulary of each token; number of steps
edges = np.linspace(-5, 5, K + 1)
cdf = ndtr((edges - means[:, :, None]) / mix.std)  # (modes, axes, K + 1)
cdf[:, :, 0], cdf[:, :, -1] = 0, 1      # edge cells extend to infinity
cell = np.diff(cdf, axis=2)             # mass of each cell, per mode, axis
table = np.einsum('m,mi,mj->ij', weights, cell[:, 0], cell[:, 1])
p1, p2 = table.sum(1), table.sum(0)     # marginals of the two tokens
product = np.outer(p1, p2)
H = lambda p: -xlogy(p, p).sum()        # entropy in nats
tv = lambda p, q: np.abs(p - q).sum() / 2
mutual = H(p1) + H(p2) - H(table)
print(f'entropy {H(table):.3f} nats, mutual information {mutual:.3f} nats')
for S in (1, 2, 8, 32):
    print(f'{S:>2} steps, exact network: TV {tv(product, table) / S:.3f}')
```

### Diffusion Language Models in Practice

Practical diffusion language models change two parts of this recipe.
The first is the network's input. Which positions are masked is decided by
coin flips that never look at the tokens, so the step only says how many
masks to expect, which the input already shows. The best prediction at a
masked position therefore does not depend on $t$
:cite:`Ou.Nie.Xue.ea.2025`, and the denoiser is a bidirectional
transformer, a GPT without the causal mask, with one extra
vocabulary entry for $[\textrm{M}]$ :cite:`Nie.Zhu.You.ea.2025`.

The second is the order of reveals. MaskGIT
:cite:`Chang.Zhang.Jiang.ea.2022` and LLaDA draw a candidate at every
masked position, keep the candidates that are most probable under their
own predicted distributions, and mask the others again. The two-token
table shows the effect without training. For two tokens, *confidence
remasking* draws both from their
marginals, keeps the draw with the higher probability, and redraws the
other from its conditional given the kept one. The cell compares its exact
output table with those of one step, two random jumps, and a fixed order
(first token from its marginal, second from its conditional).

```{.python .input #discrete-diffusion-diffusion-language-models-in-practice}
%%tab pytorch, jax
c2 = table / p1[:, None]                # p(x2 | x1); each row sums to one
c1 = table / p2[None, :]                # p(x1 | x2); each column sums to one
keep1 = p1[:, None] >= p2[None, :]      # the draw of x1 is the likelier one
confidence = ((product * keep1).sum(1)[:, None] * c2
              + (product * ~keep1).sum(0)[None, :] * c1)
centers = (edges[:-1] + edges[1:]) / 2
grid = np.stack(np.meshgrid(centers, centers, indexing='ij'), -1)
mode = ((grid[:, :, None] - means) ** 2).sum(-1).argmin(-1)  # nearest mode
orders = {'one step': product, 'two random jumps': (product + table) / 2,
          'fixed order': p1[:, None] * c2, 'confidence': confidence}
print(f'{"order":>16} {"TV":>6} {"entropy":>8}  mass near the modes')
for name, q in orders.items():
    masses = ' '.join(f'{q[mode == k].sum():.3f}' for k in range(3))
    print(f'{name:>16} {tv(q, table):6.3f} {H(q):8.3f}  {masses}')
```

With exact conditionals, any order that reveals one token per step is
exact, as the fixed order shows. Two random jumps reveal both tokens
together half of the time and halve the one-step error. Confidence
remasking also reveals one token per step, yet its error is larger than
that of random jumps. Which draw it keeps depends on the drawn values, and
it favors values that were likely to begin with, so it seeks modes: the
smallest mode falls from $0.2$ to about $0.07$, and the entropy drops from
$4.84$ to $4.41$ nats. On text, lower entropy often reads as fluency, and
with an imperfect network it can help to reveal the easy positions first.
But the bound does not describe this sampler and cannot detect its bias.
In all of these samplers a revealed token is never revised, so an early
mistake stays.

At equal compute on fresh data, autoregressive models reach a lower loss
:cite:`Prabhudesai.Wu.Zadeh.ea.2025`, so the case for diffusion is not a better likelihood but parallel
decoding, many tokens per network evaluation, and infilling, generation
conditioned on text on both sides of a gap. For images, masked diffusion
works on the discrete codes of a learned tokenizer
:cite:`Chang.Zhang.Jiang.ea.2022`; the binary pixels below stand in for
such codes because they need no tokenizer.

## Experiments

### The Quantized Mixture

The network for the pair takes the one-hot encoding of both tokens, with
the mask as a thirty-third value at index $K$, together with the time
$t/T$, and outputs logits over the $32$ clean values at each position. The
schedule is linear with $T = 32$ steps.

```{.python .input #discrete-diffusion-the-quantized-mixture}
%%tab pytorch
class MaskedMLP(nn.Module):
    """Logits of both clean tokens of a pair, given the masked pair and t."""
    def __init__(self, K, hidden=256):
        super().__init__()
        self.K = K
        self.net = d2l.make_mlp([2 * (K + 1) + 1, hidden, hidden, 2 * K],
                                nn.SiLU())

    def forward(self, z, t):  # z: (n, 2) in {0, ..., K}, where K is the mask
        h = F.one_hot(z, self.K + 1).float().flatten(1)
        return self.net(torch.cat([h, t[:, None]], 1)).view(-1, 2, self.K)

def quantize(x):  # points in the plane to pairs of tokens in {0, ..., K - 1}
    return ((x + 5) / 10 * K).floor().long().clamp(0, K - 1)

alpha_bar = 1 - torch.arange(T + 1) / T     # the linear schedule
```

```{.python .input #discrete-diffusion-the-quantized-mixture}
%%tab jax
class MaskedMLP(nnx.Module):
    """Logits of both clean tokens of a pair, given the masked pair and t."""
    def __init__(self, K, hidden=256, *, rngs):
        self.K = K
        self.net = d2l.make_mlp([2 * (K + 1) + 1, hidden, hidden, 2 * K],
                                nnx.silu, rngs=rngs)

    def __call__(self, z, t):  # z: (n, 2) in {0, ..., K}, where K is the mask
        h = jax.nn.one_hot(z, self.K + 1).reshape(len(z), -1)
        return self.net(jnp.concatenate([h, t[:, None]], 1)).reshape(
            -1, 2, self.K)

def quantize(x):  # points in the plane to pairs of tokens in {0, ..., K - 1}
    return jnp.clip(jnp.floor((x + 5) / 10 * K), 0, K - 1).astype(jnp.int32)

alpha_bar = 1 - jnp.arange(T + 1) / T       # the linear schedule
```

We train it with `d2l.train_regression`
(:numref:`sec_diffusion-gaussian-paths`) on fresh draws from the mixture,
512 pairs per batch, each with a uniformly drawn step, with a learning rate
that decays along a cosine over 6,000 updates.

```{.python .input #discrete-diffusion-the-quantized-mixture-2}
%%tab pytorch
torch.manual_seed(0)
batch_fn = lambda: (quantize(mix.sample(512)), torch.randint(1, T + 1, (512,)))
loss_fn = lambda model, batch: masked_loss(model, *batch, alpha_bar,
                                           mask_id=K).mean()
_, pair_model, _ = d2l.train_regression(MaskedMLP(K), loss_fn, batch_fn,
                                        6000, cosine=True)
```

```{.python .input #discrete-diffusion-the-quantized-mixture-2}
%%tab jax
def batch_fn(key):
    k1, k2 = jax.random.split(key)
    return (quantize(mix.sample(k1, 512)),
            jax.random.randint(k2, (512,), 1, T + 1))

loss_fn = lambda model, batch, key: masked_loss(model, *batch, alpha_bar,
                                                mask_id=K, key=key).mean()
_, pair_model, _ = d2l.train_regression(
    MaskedMLP(K, rngs=nnx.Rngs(0)), loss_fn, batch_fn, 6000,
    jax.random.key(0), cosine=True)
```

With two positions, the expectation in the bound can be computed exactly.
At each step there are three mask patterns with at least one mask, a
pattern with $m$ masks has probability $u^m (1 - u)^{2 - m}$ with
$u = 1 - \bar{\alpha}_t$, and the 1,024 pairs are weighted by the table.
This average cannot fall below the entropy of the table, the smallest
average negative log-likelihood any model attains
(:numref:`sec_mdl-information_theory`). Even an exact network stays above
it, because it predicts two masked tokens independently and the bound
charges the mutual information (:numref:`sec_mdl-mutual-information`) for
this. An exact network attains

$$
H(X^1, X^2) + \frac{I(X^1; X^2)}{T},
$$

where $X^1$ and $X^2$ are the two tokens of a pair drawn from the table
(Exercise :exref:`ex_discrete-diffusion-revealed-together`). This floor,
$4.840 + 0.479/32 \approx 4.855$ nats, binds every network that predicts
the positions independently.

```{.python .input #discrete-diffusion-the-quantized-mixture-3}
%%tab pytorch
@torch.inference_mode()
def exact_bound(model, table, alpha_bar, mask_id):
    """The bound averaged over the table, summed exactly over t and masks."""
    T, K = len(alpha_bar) - 1, len(table)
    pairs = torch.cartesian_prod(torch.arange(K), torch.arange(K))
    p, total = torch.tensor(table.flatten(), dtype=torch.float32), 0.0
    for t in range(1, T + 1):
        u = 1 - alpha_bar[t]
        w = (alpha_bar[t - 1] - alpha_bar[t]) / u
        for masked in torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.bool):
            z, m = torch.where(masked, mask_id, pairs), masked.sum()
            logp = torch.log_softmax(model(z, torch.full((K * K,), t / T)), -1)
            nll = -(logp.gather(-1, pairs[..., None])[..., 0] * masked).sum(1)
            total += u ** m * (1 - u) ** (2 - m) * w * (p @ nll)
    return float(total)

bound = exact_bound(pair_model, table, alpha_bar, mask_id=K)
print(f'exact bound {bound:.3f} nats; '
      f'entropy {H(table):.3f}, floor {H(table) + mutual / T:.3f}')
```

```{.python .input #discrete-diffusion-the-quantized-mixture-3}
%%tab jax
def exact_bound(model, table, alpha_bar, mask_id):
    """The bound averaged over the table, summed exactly over t and masks."""
    T, K = len(alpha_bar) - 1, len(table)
    pairs = jnp.stack(jnp.meshgrid(jnp.arange(K), jnp.arange(K),
                                   indexing='ij'), -1).reshape(-1, 2)
    p, total = jnp.asarray(table.flatten(), jnp.float32), 0.0
    for t in range(1, T + 1):
        u = 1 - alpha_bar[t]
        w = (alpha_bar[t - 1] - alpha_bar[t]) / u
        for masked in jnp.array([[1, 0], [0, 1], [1, 1]], dtype=bool):
            z, m = jnp.where(masked, mask_id, pairs), masked.sum()
            logp = jax.nn.log_softmax(model(z, jnp.full(K * K, t / T)))
            nll = -(jnp.take_along_axis(logp, pairs[..., None], -1)[..., 0]
                    * masked).sum(1)
            total += u ** m * (1 - u) ** (2 - m) * w * (p @ nll)
    return float(total)

bound = exact_bound(pair_model, table, alpha_bar, mask_id=K)
print(f'exact bound {bound:.3f} nats; '
      f'entropy {H(table):.3f}, floor {H(table) + mutual / T:.3f}')
```

The bound comes within a few hundredths of a nat of the entropy: the
model has learned the marginal of each token and its conditional given the
other. Almost all of the gap is the floor; the network's own excess over
the floor is of the order of a thousandth of a nat. The floor is not just
looseness of the bound: the thirty-two-step sampler of an exact network
models $(1 - 1/32)\, p + (1/32)\, p^1 \otimes p^2$ rather than $p$. The
next cell compares histograms of 20,000 pairs drawn with
$S \in \{32, 8, 2, 1\}$ steps with the table, after printing the floor
that 20,000 true samples impose.

```{.python .input #discrete-diffusion-the-quantized-mixture-4}
%%tab pytorch
def histogram(z):  # pairs of tokens to a K x K table of frequencies
    counts = torch.bincount(z[:, 0] * K + z[:, 1], minlength=K * K)
    return np.float64(counts.view(K, K)) / len(z)

torch.manual_seed(1)
data = quantize(mix.sample(20000))
print(f'TV(histogram of 20000 true samples, table): '
      f'{tv(histogram(data), table):.3f}')
samples = {S: unmask(pair_model, (20000, 2), S, alpha_bar, mask_id=K)[-1]
           for S in (32, 8, 2, 1)}
for S, z in samples.items():
    print(f'{S:>2} steps: TV {tv(histogram(z), table):.3f}, '
          f'exact network {tv(product, table) / S:.3f}')
```

```{.python .input #discrete-diffusion-the-quantized-mixture-4}
%%tab jax
def histogram(z):  # pairs of tokens to a K x K table of frequencies
    counts = jnp.bincount(z[:, 0] * K + z[:, 1], length=K * K)
    return np.float64(counts.reshape(K, K)) / len(z)

key, k1, *keys = jax.random.split(jax.random.key(1), 6)
data = quantize(mix.sample(k1, 20000))
print(f'TV(histogram of 20000 true samples, table): '
      f'{tv(histogram(data), table):.3f}')
samples = {S: unmask(pair_model, (20000, 2), S, alpha_bar, mask_id=K,
                     key=k)[-1]
           for S, k in zip((32, 8, 2, 1), keys)}
for S, z in samples.items():
    print(f'{S:>2} steps: TV {tv(histogram(z), table):.3f}, '
          f'exact network {tv(product, table) / S:.3f}')
```

The distance tracks $\mathrm{TV}(p^1 \otimes p^2, p)/S$. At thirty-two
and eight steps the sampling floor and the model error add a few
hundredths; at two steps and one step the systematic error dominates, and
the distance lies within a few hundredths of the prediction. In the
panels (2,000 pairs, placed at random inside their cells), the one-step
sampler puts mass where the column of one mode crosses the row of another,
where the data has none.

```{.python .input #discrete-diffusion-the-quantized-mixture-5}
%%tab pytorch
decode = lambda z: (z[:2000] + torch.rand(2000, 2)) / K * 10 - 5  # in cell
d2l.scatter_panels([decode(data), decode(samples[32]), decode(samples[1])],
                   ['data', '32 steps', '1 step']);
```

```{.python .input #discrete-diffusion-the-quantized-mixture-5}
%%tab jax
decode = lambda z: (z[:2000] + jax.random.uniform(key, (2000, 2))) / K * 10 - 5
d2l.scatter_panels([decode(data), decode(samples[32]), decode(samples[1])],
                   ['data', '32 steps', '1 step']);
```

### Images as Binary Tokens
:label:`subsec_diffusion-binary-images`

Thresholding Fashion-MNIST at half intensity makes every pixel a token
with two values, and a masked pixel has a third. An image is then a
sequence of 784 tokens on a grid, so the network can be the chapter's
U-Net (:numref:`sec_diffusion-images`); it takes the one-hot encoding as three input
channels and returns the logits of the clean pixel as two output channels.
The loss and the sampler are those of the pair, with a linear schedule of
$T = 100$ steps and the mask at index 2.

```{.python .input #discrete-diffusion-images-as-binary-tokens}
%%tab pytorch
X, _, X_test, _ = d2l.load_fashion_mnist_pm1(binary=True)  # (n, 28, 28)
T_img = 100                                   # steps of the image chain
alpha_bar_img = 1 - torch.arange(T_img + 1, device=X.device) / T_img

class MaskedUNet(nn.Module):
    """d2l.UNet on one-hot pixels (black, white, mask); clean-pixel logits."""
    def __init__(self):
        super().__init__()
        self.unet = d2l.UNet(in_channels=3, out_channels=2)

    def forward(self, z, t):  # z: (n, 28, 28) in {0, 1, 2}
        x = F.one_hot(z, 3).float().permute(0, 3, 1, 2)
        return self.unet(x, t).permute(0, 2, 3, 1)  # (n, 28, 28, 2)
```

```{.python .input #discrete-diffusion-images-as-binary-tokens}
%%tab jax
X, _, X_test, _ = d2l.load_fashion_mnist_pm1(binary=True)  # (n, 28, 28)
T_img = 100                                   # steps of the image chain
alpha_bar_img = 1 - jnp.arange(T_img + 1) / T_img

class MaskedUNet(nnx.Module):
    """d2l.UNet on one-hot pixels (black, white, mask); clean-pixel logits."""
    def __init__(self, *, rngs):
        self.unet = d2l.UNet(in_channels=3, out_channels=2, rngs=rngs)

    def __call__(self, z, t):  # z: (n, 28, 28) in {0, 1, 2}
        return self.unet(jax.nn.one_hot(z, 3), t)  # (n, 28, 28, 2)
```

Training takes 6,000 updates on batches of 128 images, with a constant
learning rate and a moving average of the weights with a half-life of
1,000 updates.

```{.python .input #discrete-diffusion-images-as-binary-tokens-2}
%%tab pytorch
torch.manual_seed(2)
def batch_fn():
    i = torch.randint(0, len(X), (128,), device=X.device)
    return X[i], torch.randint(1, T_img + 1, (128,), device=X.device)

loss_fn = lambda model, batch: masked_loss(  # per pixel
    model, *batch, alpha_bar_img, mask_id=2).mean() / (28 * 28)
_, unet, _ = d2l.train_regression(MaskedUNet().to(X.device), loss_fn,
                                  batch_fn, 6000)
```

```{.python .input #discrete-diffusion-images-as-binary-tokens-2}
%%tab jax
def batch_fn(key):
    k1, k2 = jax.random.split(key)
    return (X[jax.random.randint(k1, (128,), 0, len(X))],
            jax.random.randint(k2, (128,), 1, T_img + 1))

loss_fn = lambda model, batch, key: masked_loss(  # per pixel
    model, *batch, alpha_bar_img, mask_id=2, key=key).mean() / (28 * 28)
_, unet, _ = d2l.train_regression(MaskedUNet(rngs=nnx.Rngs(2)), loss_fn,
                                  batch_fn, 6000, jax.random.key(2))
```

We estimate the bound on 1,000 held-out images, with one random mask per
image and level, and compare it with the cross-entropy of a model that
treats pixels as independent, using the frequency of white at each
position in the training set.

```{.python .input #discrete-diffusion-images-as-binary-tokens-3}
%%tab pytorch
X_eval = X_test[:1000]
with torch.inference_mode():
    bound = sum(masked_loss(unet, X_eval, torch.full_like(X_eval[:, 0, 0], t),
                            alpha_bar_img, mask_id=2).mean()
                for t in range(1, T_img + 1)) / T_img
white = X.float().mean(0).clamp(1e-6, 1 - 1e-6)  # frequency of white
indep = -torch.where(X_eval == 1, white.log(), (1 - white).log()).sum((1, 2))
bits = lambda nats: f'{nats / math.log(2) / (28 * 28):.3f} bits per pixel'
print(f'bound {bits(bound)}, independent pixels {bits(indep.mean())}')
```

```{.python .input #discrete-diffusion-images-as-binary-tokens-3}
%%tab jax
X_eval, keys = X_test[:1000], jax.random.split(jax.random.key(3), T_img)
term = nnx.jit(lambda model, x0, t, key: masked_loss(
    model, x0, jnp.full(len(x0), t), alpha_bar_img, mask_id=2,
    key=key).mean())
bound = sum(term(unet, X_eval, t, k)
            for t, k in zip(range(1, T_img + 1), keys)) / T_img
white = jnp.clip(X.mean(0), 1e-6, 1 - 1e-6)  # frequency of white
indep = -jnp.where(X_eval == 1, jnp.log(white), jnp.log(1 - white)).sum((1, 2))
bits = lambda nats: f'{nats / math.log(2) / (28 * 28):.3f} bits per pixel'
print(f'bound {bits(bound)}, independent pixels {bits(indep.mean())}')
```

The bound is about 0.17 bits per pixel, against about 0.70 for
independent pixels. As code lengths (:numref:`sec_mdl-coding-compression`),
the model could encode a held-out garment in about 135 bits, where coding
each pixel on its own takes about 550; the model has captured much of the
dependence between pixels. The first grid shows 48 samples drawn in a
hundred steps.

```{.python .input #discrete-diffusion-images-as-binary-tokens-4}
%%tab pytorch
show_tokens = lambda z, rows: d2l.show_gray(  # masked pixels in gray
    torch.where(z == 2, 0.0, 2.0 * z - 1)[:, None], rows, 8, scale=0.8)
torch.manual_seed(3)
states = unmask(unet, (48, 28, 28), T_img, alpha_bar_img, mask_id=2)
show_tokens(states[-1], 6);
```

```{.python .input #discrete-diffusion-images-as-binary-tokens-4}
%%tab jax
show_tokens = lambda z, rows: d2l.show_gray(  # masked pixels in gray
    jnp.where(z == 2, 0.0, 2.0 * z - 1)[..., None], rows, 8, scale=0.8)
states = unmask(unet, (48, 28, 28), T_img, alpha_bar_img, mask_id=2,
                key=jax.random.key(4))
show_tokens(states[-1], 6);
```

Many samples are recognizable binary garments, rougher than the
gray-level samples of the earlier sections; others are fragments, at 6,000
updates of training. The second grid follows eight
of them after 20, 40, 60, 80 and 100 steps, with masked pixels in gray.

```{.python .input #discrete-diffusion-images-as-binary-tokens-5}
%%tab pytorch
show_tokens(torch.cat([states[k][:8] for k in (20, 40, 60, 80, 100)]), 5);
```

```{.python .input #discrete-diffusion-images-as-binary-tokens-5}
%%tab jax
show_tokens(jnp.concatenate([states[k][:8] for k in (20, 40, 60, 80, 100)]),
            5);
```

The outline of a garment appears early, and later reveals fill it in
consistently, because every prediction is conditioned on everything
revealed so far. The third grid uses four steps, each revealing about a
quarter of the pixels from predictions that are independent across
positions.

```{.python .input #discrete-diffusion-images-as-binary-tokens-6}
%%tab pytorch
few = unmask(unet, (48, 28, 28), 4, alpha_bar_img, mask_id=2)[-1]
show_tokens(few, 6);
```

```{.python .input #discrete-diffusion-images-as-binary-tokens-6}
%%tab jax
few = unmask(unet, (48, 28, 28), 4, alpha_bar_img, mask_id=2,
             key=jax.random.key(5))[-1]
show_tokens(few, 6);
```

No garment survives: each step reveals about 200 pixels drawn independently
of one another, and the result is texture.

To measure what four steps lose, we count *speckles*, pixels that differ
from all four of their neighbors.

```{.python .input #discrete-diffusion-images-as-binary-tokens-7}
%%tab pytorch
def speckles(z):  # fraction of interior pixels unlike all four neighbors
    c = z[:, 1:-1, 1:-1]
    return ((c != z[:, :-2, 1:-1]) & (c != z[:, 2:, 1:-1])
            & (c != z[:, 1:-1, :-2]) & (c != z[:, 1:-1, 2:])).float().mean()

print(f'speckles: data {speckles(X[:1000]):.4f}, 100 steps '
      f'{speckles(states[-1]):.4f}, 4 steps {speckles(few):.4f}')
```

```{.python .input #discrete-diffusion-images-as-binary-tokens-7}
%%tab jax
def speckles(z):  # fraction of interior pixels unlike all four neighbors
    c = z[:, 1:-1, 1:-1]
    return ((c != z[:, :-2, 1:-1]) & (c != z[:, 2:, 1:-1])
            & (c != z[:, 1:-1, :-2]) & (c != z[:, 1:-1, 2:])).mean()

print(f'speckles: data {speckles(X[:1000]):.4f}, 100 steps '
      f'{speckles(states[-1]):.4f}, 4 steps {speckles(few):.4f}')
```

The data has about four speckles per thousand interior pixels, the
hundred-step samples about twice as many, and the four-step samples about
nine times as many. The count is not a measure of sample quality; it
measures the independence error, the image counterpart of the crossed
modes in the one-step panel of the pair.

## Masked Language Models as One Step

Trained at a single masking level,
:eqref:`eq_diffusion-masked-loss` is the objective of BERT
:cite:`Devlin.Chang.Lee.ea.2018`, which hides 15% of the tokens of a text
and predicts them from both sides (:numref:`sec_bert`), up to BERT's
replacement of some masks by random or unchanged tokens. A masked language
model is thus a discrete diffusion model with one step
:cite:`Austin.Johnson.Ho.ea.2021`, though one that has seen
only one masking level, so a sampler starting from the all-mask sequence
would show it inputs far from its training data. Reveal exactly one token
per step in a random order, and the sampler is an order-agnostic
autoregressive model :cite:`Uria.Murray.Larochelle.2014`. Fix the order to
left to right, and it is the language model of :numref:`sec_gpt`, whose
conditionals are the predictions at inputs with an unmasked prefix and a
masked rest. Diffusion language models operate between these two
extremes, trading sequential network evaluations for the independence
error; the next part of this book studies masked and autoregressive
language models at scale. Over the chapter the four limits of the game of
:numref:`chap_gans` found their answers: a density and a bound
(:numref:`sec_diffusion-design-space`), one regression in place of two
networks (:numref:`sec_diffusion-gaussian-paths`), a likelihood under which
a dropped mode costs on every one of its points, and a transformer that
improves steadily as it grows (:numref:`sec_diffusion-latent`).

## Summary

Diffusion does not need Gaussian noise. A Markov chain on a vocabulary has
closed-form marginals and posteriors, and the variational bound applies to
it as it stands; with the absorbing kernel it becomes a cross-entropy on
the masked positions, weighted by $1/t$ under the linear schedule.
Sampling reveals masked tokens and never revises them; tokens revealed in
the same step are drawn independently, at an error that falls as one over
the number of steps. On the quantized mixture the bound came within a few
hundredths of a nat of the entropy, and a U-Net on binary Fashion-MNIST
reached about 0.17 bits per pixel. BERT is the one-step case of this
model, and a left-to-right language model the case that reveals one token
per step.

## Exercises

1. **The uniform kernel.** Let the forward chain use the uniform kernel
   $\mathbf{Q}_t = (1 - \beta_t)\, \mathbf{I} + \beta_t\, \tfrac{1}{K}\mathbf{1}\mathbf{1}^\top$
   with $\beta_t > 0$ at every step.
    1. Show that
       $\bar{\mathbf{Q}}_t = \bar{\alpha}_t\, \mathbf{I} + (1 - \bar{\alpha}_t)\, \tfrac{1}{K}\mathbf{1}\mathbf{1}^\top$
       and that $\mathbf{x}_t = \mathbf{x}_0$ with probability
       $\bar{\alpha}_t + (1 - \bar{\alpha}_t)/K$. A token that shows a
       value at step $t$ may therefore have been resampled. Which question
       about every position does a network have to answer under this
       kernel but not under the absorbing kernel?
    1. Derive the posterior
       $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$ from
       :eqref:`eq_diffusion-discrete-posterior`, treating the cases
       $\mathbf{x}_t = \mathbf{x}_0$ and $\mathbf{x}_t \neq \mathbf{x}_0$
       separately. Show that for $t \geq 2$ every one of the $K$ values
       has positive probability, unlike under the two-point
       posterior :eqref:`eq_diffusion-masked-posterior`.
    1. The divergence terms of the bound compare this posterior with the
       reverse model at every position. Why are they more expensive to
       evaluate than the summand of :eqref:`eq_diffusion-masked-loss`?
   :exlabel:`ex_discrete-diffusion-uniform-kernel`

1. **Weights of other schedules.** The weight of step $t$ in
   :eqref:`eq_diffusion-masked-loss` is
   $w_t = (\bar{\alpha}_{t-1} - \bar{\alpha}_t)/(1 - \bar{\alpha}_t)$.
    1. Compute $w_t$ for the schedule $\bar{\alpha}_t = 1 - (t/T)^2$ and
       compare it with the weight $1/t$ of the linear schedule.
    1. Show that a $T$-step sampling run from the all-mask sequence
       reveals a given token at step $t$ with probability
       $\bar{\alpha}_{t-1} - \bar{\alpha}_t$, independently of the other
       positions, and that this probability equals $w_t$ times the
       probability that the token is masked at step $t$ of the bound. In
       which part of a sampling run does the schedule of (a) reveal most
       of the tokens?
    1. For a network that receives the masking probability $u$, the
       continuous-time limit in :numref:`subsec_diffusion-masked-ce` is
       $\int_0^1 \frac{1}{u}\, \mathbb{E}_u[\ell]\, du$ for every
       schedule. Show that the integrand stays bounded as $u \to 0$,
       although the weight $1/u$ diverges.
    1. Training draws the time $\tau$ uniformly from $[0, 1]$. Which
       density does the masking probability $u = 1 - \bar{\alpha}(\tau)$
       then have? Why does the schedule change the variance of the
       one-sample estimate of the bound, although it leaves the limiting
       bound unchanged :cite:`Kingma.Salimans.Poole.ea.2021`?
   :exlabel:`ex_discrete-diffusion-schedule-weights`

1. **Pairs revealed together.** Let the sampler use the levels
   $T = t_0 > t_1 > \cdots > t_S = 0$ under a schedule with
   $\bar{\alpha}_0 = 1$ and $\bar{\alpha}_T = 0$, and write
   $\Delta_k = \bar{\alpha}_{t_{k+1}} - \bar{\alpha}_{t_k}$ for the
   increase of $\bar{\alpha}$ in jump $k$, where $k = 0, \ldots, S - 1$.
    1. By Exercise :exref:`ex_discrete-diffusion-schedule-weights`b,
       applied to the coarser chain of the levels $t_k$, a token is
       revealed in jump $k$ with probability $\Delta_k$, independently of
       the other positions. Show that two tokens are revealed in the same
       jump with probability $\rho = \sum_k \Delta_k^2$, and that, among
       all $S$ increments that sum to one, equal increments
       $\Delta_k = 1/S$ minimize $\rho$.
    1. Show that with an exact network the sampler draws the two tokens
       of the quantized mixture from
       $(1 - \rho)\, p + \rho\, p^1 \otimes p^2$.
    1. For a sequence of $L$ tokens and equal increments, compute the
       expected number of pairs of tokens revealed in the same jump. How
       many steps keep it below one for $L = 1024$? Why does this count
       overstate the error of the sampler on text?
   :exlabel:`ex_discrete-diffusion-pairs-revealed`

1. [code] **One token at a time.** An order-agnostic autoregressive
   sampler reveals exactly one token per step. Use `pair_model` with the
   linear `alpha_bar` of the quantized mixture and the mask index `K`.
    1. Write a sampler with two steps. The first reveals one of the two
       positions, chosen uniformly at random, from the prediction at the
       all-mask input with time input $1$. The second reveals the other
       position from the prediction with time input $1/2$. Sample 20,000
       pairs and compare their total variation distance to `table` with
       that of the thirty-two-step sampler.
    1. The same predictions define, for the order that reveals position 1
       first, a table over all 1,024 pairs,

       $$q_{12}(a, b) = p_{\boldsymbol{\theta}}(\mathbf{x}_0^1 = a \mid [\textrm{M}], [\textrm{M}])\, p_{\boldsymbol{\theta}}(\mathbf{x}_0^2 = b \mid a, [\textrm{M}]),$$

       and the reverse order defines a table $q_{21}$. Compute both from
       `pair_model` with the time inputs of (a), and their total variation
       distances to `table` and to each other. An exact network makes the
       two tables equal. What does their difference say about the trained
       network?
   :exlabel:`ex_discrete-diffusion-one-token`

1. **Two reverse parameterizations.** ● The reverse model
   :eqref:`eq_diffusion-x0-parameterization` averages the posterior over
   the network's prediction. :citet:`Austin.Johnson.Ho.ea.2021` instead
   set

   $$p_{\boldsymbol{\theta}}(\mathbf{x}_{t-1} \mid \mathbf{x}_t) \propto \sum_{\hat{\mathbf{x}}_0} q(\mathbf{x}_{t-1}, \mathbf{x}_t \mid \hat{\mathbf{x}}_0)\, \tilde{p}_{\boldsymbol{\theta}}(\hat{\mathbf{x}}_0 \mid \mathbf{x}_t)$$

   for a network output $\tilde{p}_{\boldsymbol{\theta}}$, which
   reweights each candidate by $q(\mathbf{x}_t \mid \hat{\mathbf{x}}_0)$.
   Let $q(\mathbf{x}_0 \mid \mathbf{x}_t)$ and
   $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ denote the conditionals of the
   forward process started from the data.
    1. Take a single token. Show that if the prediction in
       :eqref:`eq_diffusion-x0-parameterization` equals
       $q(\mathbf{x}_0 \mid \mathbf{x}_t)$, the reverse model is the true
       reverse transition $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$, for
       every kernel.
    1. For the same token, show that the renormalized form yields
       $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ when
       $\tilde{p}_{\boldsymbol{\theta}}(\cdot \mid \mathbf{x}_t)$ is the
       data distribution itself, and in general not when it is
       $q(\mathbf{x}_0 \mid \mathbf{x}_t)$. Under the uniform kernel, what
       happens to the evidence that $\mathbf{x}_t$ carries about
       $\mathbf{x}_0$ in the second case?
    1. Show that under the absorbing kernel the weight
       $q(\mathbf{x}_t \mid \hat{\mathbf{x}}_0)$ is the same for every
       candidate at a masked position, so that the two forms agree.
    1. Now take a sequence of $L$ tokens and a prediction that factorizes
       over positions. Show that at position $i$ the renormalized form
       agrees with the average form with prediction
       $q(\mathbf{x}_0^i \mid \mathbf{x}_t)$ when
       $\tilde{p}_{\boldsymbol{\theta}}(\mathbf{x}_0^i \mid \mathbf{x}_t)$
       is the conditional of $\mathbf{x}_0^i$ given the other positions of
       $\mathbf{x}_t$ alone. Which inputs does the network of each form
       have to use at position $i$?
   :exlabel:`ex_discrete-diffusion-two-forms`

1. [code] **Time-agnostic predictions.** ● This section argued that under
   the absorbing kernel the best prediction at a masked position does not
   depend on $t$ :cite:`Ou.Nie.Xue.ea.2025`.
    1. Write the argument as a short proof. Which property of the
       absorbing forward process does it use? Does the same hold for the
       uniform kernel?
    1. A step that reveals no token leaves the input of the next step
       unchanged. A sampler can therefore draw the reveal decisions first
       and evaluate a network without time input only in the jumps that
       reveal at least one token. Show that with $S$ equally spaced levels
       under the linear schedule the expected number of evaluations is
       $S\, (1 - (1 - 1/S)^L)$ for $L$ tokens. Evaluate it for the pair
       with $S = 32$, for the binary images with $S = 100$, and for
       $L = S = 1024$.
    1. BERT :cite:`Devlin.Chang.Lee.ea.2018` is trained to predict masked
       tokens at a masking rate of 15%. By (a), its exact predictions are
       those that the sampler needs at mask patterns of that density.
       Which inputs of a sampling run from the all-mask sequence lie far
       outside its training distribution?
    1. `MaskedMLP` receives the time as an input. Fix the pattern in which
       the first token is masked and the second is observed, and compute
       the predictions of `pair_model` for all 32 values of the second
       token at $t \in \{1, 16, 31\}$, that is, at time inputs $t/32$. Then
       compute two kinds of Kullback–Leibler divergence, each averaged over
       the marginal of the second token: that of the predictions at
       $t = 16$ and at $t = 31$ from those at $t = 1$, and that of the
       predictions at each $t$ from the exact conditionals of `table`
       (clamp probabilities at $10^{-12}$). What do the sizes of the two
       kinds say about the use that the trained network makes of its time
       input?
   :exlabel:`ex_discrete-diffusion-time-agnostic`

1. [code] **Floor of an exact network.** ● Let a schedule with
   $\bar{\alpha}_0 = 1$ and $\bar{\alpha}_T = 0$ have the increments
   $\Delta_t = \bar{\alpha}_{t-1} - \bar{\alpha}_t$, and consider the two
   tokens of the quantized mixture.
    1. Show that the bound :eqref:`eq_diffusion-masked-loss` of an exact
       network, averaged over the true table, is

       $$H(X^1, X^2) + I(X^1; X^2) \sum_{t=1}^{T} \Delta_t^2.$$

       Write the term of step $t$ with the three mask patterns and their
       probabilities; a telescoping sum of $(1 - \bar{\alpha}_t)^2$
       evaluates the sum over $t$. Check that the linear schedule gives
       $H(X^1, X^2) + I(X^1; X^2)/T$.
    1. By Exercise :exref:`ex_discrete-diffusion-pairs-revealed`a,
       $\sum_t \Delta_t^2$ is the probability that the $T$-step sampler
       reveals both tokens in the same step. Explain why the excess of the
       bound over the entropy is the mutual information times this
       probability.
    1. Evaluate the sum for the schedule $\bar{\alpha}_t = 1 - (t/T)^2$ in
       closed form and at $T = 32$, and compare the floor with that of the
       linear schedule. Why does the choice of schedule cease to matter as
       $T \to \infty$?
    1. Write a function with the call signature of `MaskedMLP` that
       returns the logarithms of the exact conditionals computed from
       `table`, and of the marginals where both tokens are masked (clamp
       probabilities at $10^{-12}$). Pass it to `exact_bound` with each of
       the two schedules as `alpha_bar`.
    1. Train `MaskedMLP` with `masked_loss` under the schedule of (c),
       with the settings used for `pair_model`, and compare its exact bound
       with its floor and with the exact bound of `pair_model`.
   :exlabel:`ex_discrete-diffusion-revealed-together`

1. **The order-agnostic bound.** ● Let a network ignore the time, as the
   exact network does by Exercise
   :exref:`ex_discrete-diffusion-time-agnostic`a, and write
   $\ell(i, U) = -\log p_{\boldsymbol{\theta}}(\mathbf{x}_0^i \mid \mathbf{x}_0^U)$
   for its loss at position $i$ when exactly the positions in the set $U$
   are unmasked. The continuous-time limit of the bound in
   :numref:`subsec_diffusion-masked-ce`, written as an integral over the
   masking probability $u$, is then a weighted sum of the losses
   $\ell(i, U)$ over $i \notin U$. :citet:`Ou.Nie.Xue.ea.2025` show that
   it equals the expected negative log-likelihood of an order-agnostic
   autoregressive model :cite:`Uria.Murray.Larochelle.2014`,

   $$\mathbb{E}_{\sigma} \Big[ \sum_{k=1}^{L} -\log p_{\boldsymbol{\theta}}\big(\mathbf{x}_0^{\sigma(k)} \mid \mathbf{x}_0^{\sigma(1)}, \ldots, \mathbf{x}_0^{\sigma(k-1)}\big) \Big],$$

   where $\sigma$ is a uniformly random permutation of the $L$ positions.
    1. Find the weight of $\ell(i, U)$ in the integral when
       $|U| = L - m$. The Beta integral
       $\int_0^1 u^{m-1} (1 - u)^{L-m}\, du = (m-1)!\, (L-m)! / L!$
       evaluates it.
    1. Find the probability that a random order reveals the positions of
       $U$ first and position $i$ next, and conclude that the two
       expressions agree.
    1. Every order gives the same value for an exact network. Which value
       is it? Compare with Exercise
       :exref:`ex_discrete-diffusion-revealed-together`a as
       $T \to \infty$.
    1. Use Jensen's inequality to show that the expression also bounds the
       negative log-likelihood of the model that draws an order uniformly
       at random and then samples the tokens in that order. Which sampler
       of Exercise :exref:`ex_discrete-diffusion-one-token` implements this
       model for two tokens?
   :exlabel:`ex_discrete-diffusion-order-agnostic`

1. [code] **Confidence-ordered unmasking.** ● This section computed the
   output table of confidence remasking on the quantized mixture exactly.
    1. Under the rule of :citet:`Nie.Zhu.You.ea.2025`, both steps take the
       most probable value instead of a draw: in the first step the
       candidates are the modes of $p^1$ and $p^2$, the one with the higher
       probability is kept, and in the second step the other token is the
       mode of its conditional given the kept one. Which pair does this
       sampler output, and with which probability? Compute its total
       variation distance to `table`.
    1. Why can the bound :eqref:`eq_diffusion-masked-loss` not detect the
       error of confidence remasking?
    1. Repeat the comparison of the four orders on a coarser grid,
       $K = 8$, on which each mode covers only a few cells. Which of the
       two errors, that of the one-step draw and that of confidence
       remasking, depends on the grid, and why?
   :exlabel:`ex_discrete-diffusion-confidence-unmasking`

1. [extended] **A masked diffusion language model.** ●●
   :citet:`Prabhudesai.Wu.Zadeh.ea.2025` report that masked diffusion
   models make better use of repeated data than autoregressive models.
   This problem repeats the data on purpose. Use the character-level
   windows `d2l.TimeMachine(batch_size=64, num_steps=128,
   tokenization='char', num_train=100000, num_val=3000)` of
   :numref:`sec_text-sequence`, whose 100,000 training windows start at
   the first 100,000 positions of *The Time Machine*; 2,000 updates on
   minibatches of 64 windows of 128 characters then pass over these
   positions about 160 times.
    1. Train the `GPT` of :numref:`sec_gpt` on these windows for 2,000
       updates with its recipe, `LMOptimizer`, and record its training and
       validation losses every 200 updates. At which update does the
       validation loss reach its minimum, and what do the two losses do
       after it?
    1. Build the denoiser as a subclass of `d2l.GPT` whose attention is
       not causal. `d2l.GPT` builds every attention layer from its nested
       class `CausalAttention`, so a subclass that defines its own nested
       class of that name, derived from `d2l.GPT.CausalAttention`, changes
       every layer and leaves the parameters and their initialization as
       they were. The forward pass of the new attention class computes
       `q, k, v = self.qkv(X)` and calls the framework's fused attention
       without the causal mask. Pass `vocab_size + 1` to make room for the
       mask, and give the model a forward pass that accepts the time and
       ignores it, as the argument of this section allows, so that
       `masked_loss` can call it. Train it with `masked_loss`, with the
       linear `alpha_bar` of $T = 128$ steps and the new vocabulary entry
       as the mask, for 2,000 updates.
    1. Every 200 updates, estimate its bound in bits per character on the
       validation windows, with several masks per window. Plot it next to
       the validation loss of `GPT` in bits per character. Compare the
       best value of each curve and the update at which it is reached.
       Expect the denoiser to stay well behind: in a scoping run on the
       WikiText-103 bytes of :numref:`sec_gpt`, an untuned denoiser of
       this size reached about 2.3 nats per byte after 1,000 updates,
       where `GPT` reaches about 1.3. The gap is the cost that the scaling
       studies measure, not a bug.
    1. Continue training the denoiser to 8,000 updates if its bound still
       falls at update 2,000. Which model begins to overfit later? How
       does training on randomly masked windows change what the denoiser
       can memorize from a window it has seen many times?
    1. Sample windows with $S \in \{128, 32, 8\}$ steps and compare them
       with samples of `GPT`. Where do the few-step samples fail? Relate
       the failures to Exercise
       :exref:`ex_discrete-diffusion-pairs-revealed`c.
   :exlabel:`ex_discrete-diffusion-language-model`

[Discussions](https://d2l.discourse.group/t/36825)
