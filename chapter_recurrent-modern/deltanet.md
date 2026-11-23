# The Delta Rule
:label:`sec_deltanet`

The additive matrix memories of :numref:`sec_matrix-state` store each new pair by adding $\mathbf{v}_t \mathbf{k}_t^\top$ to whatever is already in $\mathbf{S}_{t-1}$. That works as long as every key is bound only once. Real text and code frequently *rebind* keys: a program assigns `rate = 0.05` and later updates `rate = 0.07`; a story moves a character from Paris to Rome. Under additive writes, querying a key that has been written twice returns the sum of its old and new values, and repeated updates bury the latest value under stale ones. How can a fixed-size matrix memory overwrite the value stored under a key?

The answer is to *read before writing*: compute what the memory currently predicts for $\mathbf{k}_t$ ($\hat{\mathbf{v}}_t = \mathbf{S}_{t-1}\mathbf{k}_t$) and write only the prediction error $\beta_t(\mathbf{v}_t - \hat{\mathbf{v}}_t)\mathbf{k}_t^\top$. This *delta rule* :cite:`Widrow.Hoff.1960,Schlag.Irie.Schmidhuber.2021` performs one online step of gradient descent on the squared recall loss $\frac{1}{2}\|\mathbf{S}\mathbf{k}_t - \mathbf{v}_t\|^2$ during the forward pass. Below we derive the delta rule's projection geometry, measure both its rebinding benefit and its signal-erosion cost on random vs. correlated keys, combine it with scalar decay in `GatedDeltaNet` :cite:`Yang.Kautz.Hatamizadeh.2024` evaluated via a chunked triangular solve :cite:`Yang.Wang.Zhang.ea.2024`, and conclude with a unified view of sequence memories as online regression solvers.

```{.python .input #deltanet-the-delta-rule}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #deltanet-the-delta-rule}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## Additive Writes

### A Key Written Twice

First let us quantify how additive writes fail when a key is rebound. If a unit key $\mathbf{k}$ is written $R$ times with values $\mathbf{v}^{(1)}, \ldots, \mathbf{v}^{(R)}$, an additive memory returns their unweighted sum:

$$
\mathbf{S}\mathbf{k} = \mathbf{v}^{(1)} + \mathbf{v}^{(2)} + \cdots +
\mathbf{v}^{(R)},
$$
:eqlabel:`eq_dn-superposition`

plus cross-key interference $\sum_i (\mathbf{k}_i^\top \mathbf{k})\mathbf{v}_i$. A scalar decay $a_t \in (0, 1)$ attenuates older values relative to newer ones, but it decays *all* keys of the same age indiscriminately—fading facts that were written once just as fast as stale values of rebound keys (Exercise :exref:`ex_dn-decay`).

To isolate rebinding from cross-key interference, `overwrite_recall` tests 8 mutually orthonormal keys in $\mathbb{R}^{64}$, writing each key $R \in \{1, 2, 4, 8\}$ times with values drawn from a 32-vector unit codebook and measuring how often the nearest codebook neighbor to $\mathbf{S}\mathbf{k}$ matches the latest value $\mathbf{v}^{(R)}$.

```{.python .input #deltanet-a-key-written-twice-1}
%%tab pytorch, jax
def overwrite_recall(write, num_writes, d=64, num_keys=8, num_values=32,
                     trials=200, seed=0):
    """Recall of the latest value after every key is written num_writes
    times."""
    rng = np.random.default_rng(seed)
    codebook = rng.standard_normal((num_values, d))
    codebook /= np.linalg.norm(codebook, axis=1, keepdims=True)
    recall = 0
    for _ in range(trials):
        keys = np.linalg.qr(rng.standard_normal((d, num_keys)))[0].T  # Rows
        S = np.zeros((d, d))
        for _ in range(num_writes):
            ids = rng.integers(0, num_values, num_keys)
            for k, v in zip(keys, codebook[ids]):
                S = write(S, k, v)
        reads = keys @ S.T  # Row j is S k_j
        recall += ((reads @ codebook.T).argmax(1) == ids).mean() / trials
    return recall

additive = lambda S, k, v: S + np.outer(v, k)
writes = [1, 2, 4, 8]
print('additive:', ' '.join(f'{overwrite_recall(additive, R):.2f}'
                            for R in writes))
```

Even with zero interference between the 8 orthonormal keys in $\mathbb{R}^{64}$, additive recall drops from $1.00$ at $R = 1$ to $0.51$ at $R = 2$ (a coin toss between the two summed values) and $0.21$ at $R = 8$ (Exercise :exref:`ex_dn-order`).

## Writes as Gradient Steps

### The Delta Rule

To replace the value bound to $\mathbf{k}_t$ while disturbing other directions as little as possible, we measure the memory's squared prediction error on the incoming pair $(\mathbf{k}_t, \mathbf{v}_t)$:

$$
\ell_t(\mathbf{S}) = \tfrac{1}{2}\, \|\mathbf{S}\mathbf{k}_t -
\mathbf{v}_t\|^2,
$$
:eqlabel:`eq_dn-recall-loss`

and take one gradient descent step $\mathbf{S}_t = \mathbf{S}_{t-1} - \beta_t \nabla_{\mathbf{S}} \ell_t(\mathbf{S}_{t-1})$ with step size (*write strength*) $\beta_t$:

$$
\mathbf{S}_t = \mathbf{S}_{t-1} - \beta_t (\mathbf{S}_{t-1}\mathbf{k}_t -
\mathbf{v}_t)\mathbf{k}_t^\top.
$$
:eqlabel:`eq_dn-delta`

This is the *delta rule* :cite:`Widrow.Hoff.1960,Schlag.Irie.Schmidhuber.2021`: read the current prediction $\hat{\mathbf{v}}_t = \mathbf{S}_{t-1}\mathbf{k}_t$, and add the rank-one correction $\beta_t(\mathbf{v}_t - \hat{\mathbf{v}}_t)\mathbf{k}_t^\top$. Notice that this gradient step occurs *inside the forward pass* on every token at both training and inference time. For a unit key $\|\mathbf{k}_t\| = 1$, querying $\mathbf{k}_t$ immediately after :eqref:`eq_dn-delta` returns $\mathbf{S}_t \mathbf{k}_t = (1 - \beta_t)\hat{\mathbf{v}}_t + \beta_t \mathbf{v}_t$—an exact replacement when $\beta_t = 1$ and a convex blend when $\beta_t \in (0, 1)$. (By contrast, the additive write $\mathbf{v}_t \mathbf{k}_t^\top$ is a gradient ascent step on the linear alignment $\mathbf{v}_t^\top \mathbf{S}\mathbf{k}_t$, which has no saturation point.)

### The Transition

Factoring $\mathbf{S}_{t-1}$ out of :eqref:`eq_dn-delta` exposes the matrix transition applied to the existing memory:

$$
\mathbf{S}_t = \mathbf{S}_{t-1}(\mathbf{I} - \beta_t \mathbf{k}_t
\mathbf{k}_t^\top) + \beta_t \mathbf{v}_t \mathbf{k}_t^\top.
$$
:eqlabel:`eq_dn-transition`

For a unit key $\mathbf{k}_t$, the symmetric matrix $\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top$ fixes every subspace orthogonal to $\mathbf{k}_t$ (eigenvalue $1$) and scales the direction $\mathbf{k}_t$ by eigenvalue $1 - \beta_t$ (:numref:`fig_dn-geometry`):

- $\beta_t = 0$: identity (no write);
- $\beta_t \in (0, 1)$: partial contraction along $\mathbf{k}_t$;
- $\beta_t = 1$: orthogonal projection $\mathbf{I} - \mathbf{k}_t \mathbf{k}_t^\top$, completely erasing the $\mathbf{k}_t$ component before writing $\mathbf{v}_t \mathbf{k}_t^\top$;
- $\beta_t = 2$: Householder reflection $\mathbf{I} - 2\mathbf{k}_t \mathbf{k}_t^\top$ (eigenvalue $-1$), reversing the sign along $\mathbf{k}_t$.

![The transition $\mathbf{I} - \beta\mathbf{k}\mathbf{k}^\top$ acting on a vector $\mathbf{s}$ in the plane spanned by a unit key $\mathbf{k}$ and an orthogonal direction. The component along $\mathbf{k}$ is multiplied by $1 - \beta$: kept at $\beta = 0$, halved at $\beta = 0.5$, removed at $\beta = 1$, and reversed at $\beta = 2$. The orthogonal component is unchanged.](../img/mdl-modernrnn-delta-geometry.svg)
:label:`fig_dn-geometry`

Two consequences follow from the eigenvalue $1 - \beta_t\|\mathbf{k}_t\|^2$:

1. **Unit-norm keys prevent instability:** With $\|\mathbf{k}_t\| = 1$ and $\beta_t \in [0, 2]$, all eigenvalues of the transition lie in $[-1, 1]$, so homogeneous memory states never blow up geometrically.
2. **Negative eigenvalues enable modulo-2 counting (parity):** When $\beta_t \in [0, 1]$, all transition eigenvalues are non-negative, which in finite precision cannot track parity over arbitrary sequence lengths :cite:`Grazzi.Siems.Franke.ea.2025`. Extending $\beta_t$ to $2$ allows reflections (eigenvalue $-1$) that toggle a stored state on each counted token, computing exact parity at any length. :numref:`subsec_dn-state-tracking` tests whether a trained layer finds this solution.

### The Cost of Correction

On orthonormal keys, $\beta_t = 1$ erases stale values on rebound keys without touching any other key. What happens when $n > d_k$ or keys are random unit vectors so that distinct keys $\mathbf{k}_i$ and $\mathbf{k}_t$ have non-zero overlap $\mathbf{k}_t^\top \mathbf{k}_i$? Each subsequent delta step along $\mathbf{k}_t$ projects out the $\mathbf{k}_t$-component of $\mathbf{v}_i \mathbf{k}_i^\top$, shrinking key $i$'s stored signal by $1 - (\mathbf{k}_t^\top \mathbf{k}_i)^2 \approx 1 - 1/d_k$ per step—so after $m$ later writes, only $(1 - 1/d_k)^m \approx e^{-m/d_k}$ of an old pair's signal survives.

To see when targeted erasure helps and when it hurts, `delta_rule(beta)` implements online delta writes and `correction_cost` measures both squared read error and nearest-neighbor recall at width $d = 64$ across multiple write rounds and key correlations:

```{.python .input #deltanet-what-correction-costs-1}
%%tab pytorch, jax
def delta_rule(beta):
    """The memory after writing the pairs in order with the delta rule."""
    def fit(K, V):
        S = np.zeros((V.shape[1], K.shape[1]))
        for k, v in zip(K, V):
            S = S + beta * np.outer(v - S @ k, k)
        return S
    return fit

def correction_cost(memories, num_keys, writes=1, orthonormal=False, rho=0,
                    d=64, num_values=256, trials=20, seed=0):
    """Squared read error and recall of the latest value when each of
    num_keys keys is written `writes` times, for each memory."""
    rng = np.random.default_rng(seed)
    unit = lambda X: X / np.linalg.norm(X, axis=-1, keepdims=True)
    codebook = unit(rng.standard_normal((num_values, d)))
    error, recall = dict.fromkeys(memories, 0), dict.fromkeys(memories, 0)
    for _ in range(trials):
        if orthonormal:
            keys = np.linalg.qr(rng.standard_normal((d, num_keys)))[0].T
        else:  # Random unit keys that share one direction with weight rho
            keys = unit(np.sqrt(1 - rho**2) * unit(rng.standard_normal(
                (num_keys, d))) + rho * unit(rng.standard_normal(d)))
        ids = rng.integers(0, num_values, (writes, num_keys))  # Row = round
        K, V = np.tile(keys, (writes, 1)), codebook[ids.ravel()]
        for name, fit in memories.items():
            reads = keys @ fit(K, V).T  # Row j is S k_j
            error[name] += ((reads - codebook[ids[-1]]) ** 2).sum(1).mean()
            recall[name] += ((reads @ codebook.T).argmax(1) == ids[-1]).mean()
    return ({name: e / trials for name, e in error.items()},
            {name: r / trials for name, r in recall.items()})
```

We compare four memory rules—`additive`, `delta, β = 1`, `delta, β = 0.5`, and offline `least squares` (`np.linalg.lstsq`, which minimizes $\sum_{s} \|\mathbf{S}\mathbf{k}_s - \mathbf{v}_s\|^2$ over all pairs simultaneously)—across three regimes: (1) 8 orthonormal keys written 4 times each, (2) $n \in \{32, 64, 128, 256\}$ independent random unit keys written once, and (3) 64 correlated keys ($\rho = 0.5$) written once:

```{.python .input #deltanet-what-correction-costs-2}
%%tab pytorch, jax
memories = {'additive': lambda K, V: V.T @ K,  # Sum of the v k^T
            'delta, β = 1': delta_rule(1), 'delta, β = 0.5': delta_rule(0.5),
            'least squares': lambda K, V: np.linalg.lstsq(K, V)[0].T}
cases = {'8 orthonormal keys, 4 writes': dict(num_keys=8, writes=4,
                                              orthonormal=True),
         **{f'{n} keys': dict(num_keys=n) for n in (32, 64, 128, 256)},
         '64 keys, ρ = 0.5': dict(num_keys=64, rho=0.5)}
results = {case: correction_cost(memories, **args)
           for case, args in cases.items()}
for i, measure in enumerate(['squared read error', 'recall']):
    print(f'{measure:28s}', ''.join(f'{name:>15s}' for name in memories))
    for case, result in results.items():
        print(f'{case:28s}', ''.join(f'{result[i][name]:15.2f}'
                                     for name in memories))
```

The table highlights the trade-off of online projection:

- **Rebinding (`8 orthonormal keys, 4 writes`):** `delta` achieves $1.00$ recall at both $\beta = 1$ and $\beta = 0.5$ (squared error $0.00$ and $0.33$), whereas `additive` and unweighted `least squares` average all 4 values bound to each key and score just $0.28$.
- **Correlated keys (`64 keys, ρ = 0.5`):** Because every key shares a common direction, `additive` piles every value onto that shared component (squared error $5.82$, recall $0.54$), whereas `delta` subtracts the prediction along the shared direction before writing (recall $0.77$ at $\beta = 1$ and $0.89$ at $\beta = 0.5$).
- **Crowded independent keys (`128 keys`, `256 keys`):** Here `delta, β = 1` has lower squared error than `additive` ($1.06$ vs. $1.99$ at $n = 128$), yet *lower* nearest-neighbor recall ($0.53$ vs. $0.96$). Why? Because shrinking old reads toward $\mathbf{0}$ lowers their squared error toward $\|\mathbf{0} - \mathbf{v}_j\|^2 = 1$ while eroding the signal of early pairs relative to later ones. Using a gentler step $\beta = 0.5$ recovers both lower squared error ($0.67$) and much higher recall ($0.82$), motivating making $\beta_t$ an input-dependent gate so the network can use $\beta_t \approx 1$ to overwrite a rebound key and smaller $\beta_t$ when accumulating facts.

## Gated DeltaNet

### Decay and Correction

Because the delta rule modifies $\mathbf{S}_{t-1}$ only along the current key $\mathbf{k}_t$, directions orthogonal to recent keys never decay on their own. *Gated DeltaNet* :cite:`Yang.Kautz.Hatamizadeh.2024` combines an input-dependent scalar decay $a_t \in (0, 1)$ (for global forgetting) with the delta rule's rank-one directional update (for targeted key overwriting):

$$
\mathbf{S}_t = a_t\, \mathbf{S}_{t-1}(\mathbf{I} - \beta_t
\mathbf{k}_t\mathbf{k}_t^\top) + \beta_t \mathbf{v}_t \mathbf{k}_t^\top.
$$
:eqlabel:`eq_dn-gated`

The two gates do different jobs, and their joint settings give the layer three ways to treat a new pair (:numref:`fig_dn-gate-modes`). With $a_t$ near zero the layer flushes everything it has stored and keeps only the new pair, which is the right response to a change of topic. With $a_t \approx 1$ and $\beta_t = 1$ it replaces the value under the current key and leaves orthogonal keys untouched, as in the rebinding test above. With $a_t \approx 1$ and a small $\beta_t$ it moves the read only part of the way toward the new value, so that several writes under similar keys are averaged rather than the last one winning.

![One step of Gated DeltaNet on a $2 \times 2$ memory under three settings of its gates. The arrows are the reads along the current key $\mathbf{k}_t$ (blue) and along an orthogonal key (purple), before the step (dashed) and after it (solid). A decay $a_t$ near zero flushes the memory, and only the new pair remains. With $a_t \approx 1$, a write strength $\beta_t = 1$ replaces the value under $\mathbf{k}_t$ by $\mathbf{v}_t$ and leaves the other key alone, while a small $\beta_t$ moves the read only part of the way toward $\mathbf{v}_t$.](../img/mdl-modernrnn-gated-delta-modes.svg)
:label:`fig_dn-gate-modes`

### Chunked Computation

Because the transition $a_t(\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top)$ is a $d_k \times d_k$ matrix rather than a scalar, a naive token-level scan would multiply $d_k \times d_k$ matrices at $O(T d_k^3)$ cost. :citet:`Yang.Wang.Zhang.ea.2024` showed how to evaluate :eqref:`eq_dn-gated` in chunks of size $m$ using the *WY representation*. Rewrite :eqref:`eq_dn-gated` as an additive decayed update $\mathbf{S}_t = a_t \mathbf{S}_{t-1} + \mathbf{u}_t \mathbf{k}_t^\top$ with *effective corrected value* $\mathbf{u}_t = \beta_t(\mathbf{v}_t - a_t \mathbf{S}_{t-1}\mathbf{k}_t)$. Inside a chunk of length $m$ starting from initial state $\mathbf{S}_0$, unrolling $a_t \mathbf{S}_{t-1}\mathbf{k}_t = \gamma_t \mathbf{S}_0 \mathbf{k}_t + \sum_{s < t} L_{ts}(\mathbf{k}_s^\top \mathbf{k}_t)\mathbf{u}_s$ (where $L_{ts} = \prod_{r=s+1}^t a_r$ and $\gamma_t = \prod_{r=1}^t a_r$) yields a strictly lower-triangular $m \times m$ linear system for the chunk's corrected values $\mathbf{u}_1, \ldots, \mathbf{u}_m$:

$$
\mathbf{u}_t + \beta_t \sum_{s < t} L_{ts}\, (\mathbf{k}_s^\top
\mathbf{k}_t)\, \mathbf{u}_s = \beta_t\big(\mathbf{v}_t - \gamma_t
\mathbf{S}_0\mathbf{k}_t\big).
$$

Because the right-hand side is affine in $\mathbf{S}_0$, `delta_rule_writes` solves the unit-lower-triangular $m \times m$ system $\mathbf{M} = \mathbf{I} + \textrm{diag}(\boldsymbol{\beta})\,\textrm{tril}_{-1}(\mathbf{L} \odot \mathbf{K}\mathbf{K}^\top)$ in parallel across all chunks and heads for two right-hand sides—$\mathbf{W} = \mathbf{M}^{-1}(\boldsymbol{\beta} \odot \mathbf{V})$ and $\mathbf{Z} = \mathbf{M}^{-1}((\boldsymbol{\beta} \odot \boldsymbol{\gamma}) \odot \mathbf{K})$—so that $\mathbf{U} = \mathbf{W} - \mathbf{Z}\mathbf{S}_0^\top$ once $\mathbf{S}_0$ is known.

```{.python .input #deltanet-chunked-computation-1}
%%tab pytorch
def delta_rule_writes(K, V, beta, log_a):  #@save
    """W and Z with corrected values U = W - Z S_0^T, for every chunk.

    K: (..., m, d_k); V: (..., m, d_v); beta, log_a: (..., m)
    """
    L = d2l.decay_mask(log_a)  # Products of decays within the chunk
    gamma = torch.exp(log_a.cumsum(-1)).unsqueeze(-1)  # Decay since start
    beta = beta.unsqueeze(-1)
    M = torch.eye(K.shape[-2], device=K.device) + (
        beta * L.tril(-1) * (K @ K.mT))  # Lower triangular
    W = torch.linalg.solve_triangular(M, beta * V, upper=False)
    Z = torch.linalg.solve_triangular(M, beta * gamma * K, upper=False)
    return W, Z
```

```{.python .input #deltanet-chunked-computation-1}
%%tab jax
def delta_rule_writes(K, V, beta, log_a):  #@save
    """W and Z with corrected values U = W - Z S_0^T, for every chunk.

    K: (..., m, d_k); V: (..., m, d_v); beta, log_a: (..., m)
    """
    L = d2l.decay_mask(log_a)  # Products of decays within the chunk
    gamma = jnp.exp(log_a.cumsum(-1))[..., None]  # Decay since start
    beta = beta[..., None]
    M = jnp.eye(K.shape[-2]) + (
        beta * jnp.tril(L, -1) * (K @ K.mT))  # Lower triangular
    W = jax.scipy.linalg.solve_triangular(M, beta * V, lower=True)
    Z = jax.scipy.linalg.solve_triangular(M, beta * gamma * K, lower=True)
    return W, Z
```

Substituting $\mathbf{U} = \mathbf{W} - \mathbf{Z}\mathbf{S}_0^\top$ into the chunk-end state $\mathbf{S}_m = \gamma_m \mathbf{S}_0 + \mathbf{U}^\top \mathbf{D}\mathbf{K}$ (where $\mathbf{D}$ is the diagonal matrix of decays to the end of the chunk) expresses the chunk transition as an affine matrix recurrence across the $T / m$ chunks:

$$
\mathbf{S}_m = \mathbf{S}_0 \mathbf{A} + \mathbf{W}^\top \mathbf{D}
\mathbf{K}, \qquad \mathbf{A} = \gamma_m \mathbf{I} - \mathbf{Z}^\top
\mathbf{D} \mathbf{K}.
$$

Scanning $(\mathbf{A}, \mathbf{W}^\top \mathbf{D}\mathbf{K})$ across the $T / m$ chunks costs only $(T/m) d_k^3$ operations (matching the intra-chunk $O(T d_k^2)$ matrix products when $m \approx d_k$), after which $\mathbf{U} = \mathbf{W} - \mathbf{Z}\mathbf{S}_0^\top$ and all outputs $\mathbf{Y}$ follow in one batched matrix product.

```{.python .input #deltanet-chunked-computation-2}
%%tab pytorch
def chunked_delta_rule(Q, K, V, beta, log_a, chunk_size):  #@save
    """Outputs y_t = S_t q_t of the gated delta rule, computed in chunks.

    Q, K: (batch_size, T, d_k); V: (batch_size, T, d_v);
    beta, log_a: (batch_size, T)
    """
    B, T, m = Q.shape[0], Q.shape[1], chunk_size
    Q, K, V = (X.reshape(B, T // m, m, -1) for X in (Q, K, V))
    beta, log_a = beta.reshape(B, T // m, m), log_a.reshape(B, T // m, m)
    W, Z = delta_rule_writes(K, V, beta, log_a)
    L, cum = d2l.decay_mask(log_a), log_a.cumsum(-1)
    # Each chunk maps the memory S at its start to S A + W^T D K at its end
    to_end = torch.exp(cum[..., -1:] - cum).unsqueeze(-1)
    I = torch.eye(K.shape[-1], device=K.device)
    A = torch.exp(cum[..., -1, None, None]) * I - (Z * to_end).mT @ K
    def combine(earlier, later):  # A scan over chunks, with matrix products
        (A1, W1), (A2, W2) = earlier, later
        return A1 @ A2, W1 @ A2 + W2
    W_end = d2l.associative_scan(combine, (A, (W * to_end).mT @ K))[1]
    S = torch.cat([torch.zeros_like(W_end[:, :1]), W_end[:, :-1]], 1)
    U = W - Z @ S.mT  # Corrected values, with the memory at each start
    Y = torch.exp(cum).unsqueeze(-1) * (Q @ S.mT) + (L * (Q @ K.mT)) @ U
    return Y.reshape(B, T, -1)
```

```{.python .input #deltanet-chunked-computation-2}
%%tab jax
def chunked_delta_rule(Q, K, V, beta, log_a, chunk_size):  #@save
    """Outputs y_t = S_t q_t of the gated delta rule, computed in chunks.

    Q, K: (batch_size, T, d_k); V: (batch_size, T, d_v);
    beta, log_a: (batch_size, T)
    """
    B, T, m = Q.shape[0], Q.shape[1], chunk_size
    Q, K, V = (X.reshape(B, T // m, m, -1) for X in (Q, K, V))
    beta, log_a = beta.reshape(B, T // m, m), log_a.reshape(B, T // m, m)
    W, Z = delta_rule_writes(K, V, beta, log_a)
    L, cum = d2l.decay_mask(log_a), log_a.cumsum(-1)
    # Each chunk maps the memory S at its start to S A + W^T D K at its end
    to_end = jnp.exp(cum[..., -1:] - cum)[..., None]
    I = jnp.eye(K.shape[-1])
    A = jnp.exp(cum[..., -1, None, None]) * I - (Z * to_end).mT @ K
    def combine(earlier, later):  # A scan over chunks, with matrix products
        (A1, W1), (A2, W2) = earlier, later
        return A1 @ A2, W1 @ A2 + W2
    W_end = jax.lax.associative_scan(combine, (A, (W * to_end).mT @ K),
                                     axis=1)[1]
    S = jnp.concatenate([jnp.zeros_like(W_end[:, :1]), W_end[:, :-1]], 1)
    U = W - Z @ S.mT  # Corrected values, with the memory at each start
    Y = jnp.exp(cum)[..., None] * (Q @ S.mT) + (L * (Q @ K.mT)) @ U
    return Y.reshape(B, T, -1)
```

To verify `chunked_delta_rule`, we first implement the sequential token-by-token reference `delta_rule_recurrent`:

```{.python .input #deltanet-chunked-computation-4}
%%tab pytorch
def delta_rule_recurrent(Q, K, V, beta, a):
    """The same outputs, one step at a time."""
    S, outputs = Q.new_zeros(V.shape[-1], K.shape[-1]), []
    for q, k, v, b, a_t in zip(Q, K, V, beta, a):
        S = a_t * S
        S = S + b * torch.outer(v - S @ k, k)
        outputs.append(S @ q)
    return torch.stack(outputs)
```

```{.python .input #deltanet-chunked-computation-4}
%%tab jax
def delta_rule_recurrent(Q, K, V, beta, a):
    """The same outputs, one step at a time."""
    def step(S, inputs):
        q, k, v, b, a_t = inputs
        S = a_t * S
        S = S + b * jnp.outer(v - S @ k, k)
        return S, S @ q
    S = jnp.zeros((V.shape[-1], K.shape[-1]))
    return jax.lax.scan(step, S, (Q, K, V, beta, a))[1]
```

We now compare `chunked_delta_rule` at chunk sizes $m \in \{16, 64\}$ against `delta_rule_recurrent` on a random sequence of length $T = 256$ with unit-normalized keys:

```{.python .input #deltanet-chunked-computation-3}
%%tab pytorch
gen = torch.Generator().manual_seed(0)  # Local: later training stays unseeded
T, d_k, d_v = 256, 32, 32
Q, V = torch.randn(T, d_k, generator=gen), torch.randn(T, d_v, generator=gen)
K = F.normalize(torch.randn(T, d_k, generator=gen), dim=-1)
beta = torch.rand(T, generator=gen)
a = 0.8 + 0.19 * torch.rand(T, generator=gen)
Y = delta_rule_recurrent(Q, K, V, beta, a)
for m in [16, 64]:
    Y_chunked = chunked_delta_rule(Q[None], K[None], V[None], beta[None],
                                   torch.log(a)[None], m)[0]
    print(f'chunk size {m}: relative deviation '
          f'{(Y_chunked - Y).abs().max() / Y.abs().max():.1e}')
```

```{.python .input #deltanet-chunked-computation-3}
%%tab jax
T, d_k, d_v = 256, 32, 32
keys = jax.random.split(jax.random.key(0), 5)
Q = jax.random.normal(keys[0], (T, d_k))
V = jax.random.normal(keys[1], (T, d_v))
K = jax.random.normal(keys[2], (T, d_k))
K = K / jnp.linalg.norm(K, axis=-1, keepdims=True)
beta = jax.random.uniform(keys[3], (T,))
a = 0.8 + 0.19 * jax.random.uniform(keys[4], (T,))
with jax.default_matmul_precision('highest'):  # Full float32, not TF32
    Y = delta_rule_recurrent(Q, K, V, beta, a)
    for m in [16, 64]:
        Y_chunked = chunked_delta_rule(Q[None], K[None], V[None], beta[None],
                                       jnp.log(a)[None], m)[0]
        print(f'chunk size {m}: relative deviation '
              f'{jnp.abs(Y_chunked - Y).max() / jnp.abs(Y).max():.1e}')
```

The chunked algorithm matches the token-level recurrence to $\sim 10^{-7}$ relative error.

### The Layer

`GatedDeltaNet` subclasses `d2l.GatedLinearAttention` (:numref:`sec_matrix-state`), adding only the write-strength projection `W_beta` ($\beta_{t,h} = \sigma(\mathbf{w}_{\beta,h}^\top \mathbf{x}_t + b_{\beta,h}) \in (0, 1)$ per head) and overriding `memory` to call `chunked_delta_rule`. Every other component—the width-4 causal convolution, $\ell_2$-normalized queries and keys, scalar per-head decay, RMS head normalization, and SiLU output gate—is identical.

```{.python .input #deltanet-the-layer}
%%tab pytorch
class GatedDeltaNet(d2l.GatedLinearAttention):  #@save
    """Gated linear attention whose memory is written by the delta rule."""
    def __init__(self, num_hiddens, num_heads, chunk_size=32, conv_width=4,
                 timescales=(4, 256)):
        super().__init__(num_hiddens, num_heads, chunk_size, conv_width,
                         timescales)
        self.W_beta = nn.Linear(num_hiddens, num_heads)  # Write strength
        d2l.init_lecun(self.W_beta)

    def memory(self, X, Q, K, V, log_a):
        beta = d2l.heads_to_batch(torch.sigmoid(self.W_beta(X)),
                                  self.num_heads).squeeze(-1)
        return chunked_delta_rule(Q, K, V, beta, log_a, self.chunk_size)
```

```{.python .input #deltanet-the-layer}
%%tab jax
class GatedDeltaNet(d2l.GatedLinearAttention):  #@save
    """Gated linear attention whose memory is written by the delta rule."""
    def __init__(self, num_hiddens, num_heads, chunk_size=32, conv_width=4,
                 timescales=(4, 256), *, rngs):
        super().__init__(num_hiddens, num_heads, chunk_size, conv_width,
                         timescales, rngs=rngs)
        self.W_beta = nnx.Linear(num_hiddens, num_heads, rngs=rngs)

    def memory(self, X, Q, K, V, log_a):
        beta = d2l.heads_to_batch(jax.nn.sigmoid(self.W_beta(X)),
                                  self.num_heads)[..., 0]
        return chunked_delta_rule(Q, K, V, beta, log_a, self.chunk_size)
```

### Rebinding

Can a trained `GatedDeltaNet` layer overwrite rebound keys where `GatedLinearAttention` struggles? We test a single-block `SeqLM` ($d = 64$, 4 heads of dimension 16, initial decay timescales $[256, 2048]$) on `d2l.associative_recall` with 16 keys drawn from 128 and values from 16, where 8 of the 16 keys are rebound to a new value just 2 writes after their initial binding (`num_rebound=8, gap=2`). Keeping the gap short ensures the two bindings have nearly the same age, so global age-based decay cannot easily separate the new value from the stale one without also forgetting keys written only once. (We use a single block because a two-block additive model can approximate one delta step across depth; see Exercise :exref:`ex_dn-rebinding`.)

```{.python .input #deltanet-rebinding-1}
%%tab pytorch
def rebinding_model(layer, num_hiddens=64, num_heads=4,
                    vocab_size=16 + 128 + 1):
    """One block of `layer`; tokens: 16 values, 128 keys, the filler."""
    return d2l.SeqLM(vocab_size, num_hiddens, [layer(
        num_hiddens, num_heads, chunk_size=16, timescales=(256, 2048))])
```

```{.python .input #deltanet-rebinding-1}
%%tab jax
def rebinding_model(layer, num_hiddens=64, num_heads=4,
                    vocab_size=16 + 128 + 1):
    """One block of `layer`; tokens: 16 values, 128 keys, the filler."""
    rngs = nnx.Rngs(d2l.get_key())  # A new initialization each call
    return d2l.SeqLM(vocab_size, num_hiddens, [layer(
        num_hiddens, num_heads, chunk_size=16, timescales=(256, 2048),
        rngs=rngs)], rngs=rngs)
```

Using `recall_by_kind` to separate accuracy on rebound keys (`rebound`) from keys written only once (`~rebound`), we train two seeds each of `GatedLinearAttention` and `GatedDeltaNet` on the 16-pair rebinding task:

```{.python .input #deltanet-rebinding-2}
%%tab pytorch, jax
def recall_by_kind(model, X, Y, rebound):
    """Recall on the queries for keys written twice and for keys written
    once; a label of -1 matches no prediction."""
    accuracy = lambda mask: float(d2l.query_accuracy(
        model, X, np.where(mask, Y, -1))) / mask.mean()
    return accuracy(rebound), accuracy(~rebound)

task = dict(num_keys=128, num_values=16, num_rebound=8, gap=2)
test = d2l.associative_recall(np.random.default_rng(1), 2048, 16,
                              num_steps=64, **task)
print(f'{"":24s}{"parameters":>12s}{"written twice":>16s}'
      f'{"written once":>16s}')
for name, layer in (('gated linear attention', d2l.GatedLinearAttention),
                    ('Gated DeltaNet', d2l.GatedDeltaNet)):
    models = [rebinding_model(layer) for _ in range(2)]  # Two runs
    for model in models:
        d2l.train_recall(model, pairs=(16,), **task)
    twice, once = zip(*[recall_by_kind(model, *test) for model in models])
    print(f'{name:24s}{d2l.num_params(models[0]):12,d}'
          f'{twice[0]:11.3f} {twice[1]:.3f}{once[0]:11.3f} {once[1]:.3f}')
```

Adding the 260-parameter `W_beta` head raises recall on rebound keys from roughly $0.85$–$0.9$ (`gated linear attention`) to $0.96$ or more (`Gated DeltaNet`), bringing rebound keys within about $2$ percentage points of keys written only once.

### State Tracking
:label:`subsec_dn-state-tracking`

Every experiment so far asked a memory to store something and to return it later. A different demand is to *track* a state that every token may change: which cup hides the ball after a series of swaps, whether a quotation is open, or, in the simplest case, whether the number of ones read so far is odd or even. Parity needs a single bit of state. A 1 flips it and a 0 leaves it alone. Nothing has to be kept for long, but nothing may be skipped either, since one missed bit makes every later answer wrong.

A GRU solves this task the way a program would. Its gates see the previous state $\mathbf{h}_{t-1}$, so one step can compute "flip if the input is 1". The linear recurrences of this chapter gave that up for parallel training: their decays and writes depend on the input alone (:numref:`sec_linear-recurrence`). To flip a stored bit, such a recurrence has to multiply a coordinate of its state by a negative number, and the decay $\mathbf{z}_t \in (0, 1)$ of a minGRU can shrink a coordinate but never change its sign. The delta rule is under the same restriction as long as its write strength is a sigmoid, because the transition :eqref:`eq_dn-transition` then has an eigenvalue $1 - \beta_t \in (0, 1)$ along the key. With $\beta_t \in (0, 2)$ the eigenvalue can reach $-1$. A layer that writes a value $\mathbf{v}$ with $\beta_t = 2$ under one fixed key $\mathbf{k}$ on every 1, and writes nothing on a 0, changes its read by $\mathbf{S}_t\mathbf{k} = 2\mathbf{v} - \mathbf{S}_{t-1}\mathbf{k}$ on every 1. Starting from an empty memory, the read alternates between $2\mathbf{v}$ and $\mathbf{0}$, which is the parity.

The construction exists, then, but does training find it? We train a one-block `d2l.SeqLM` of width 64 on random strings of 64 bits to output, at every step, the parity of the bits so far. We then test it on strings of 256 bits, which separates a model that has learned the rule from one that has only fit the lengths it was trained on. Four mixers compete: a GRU, a minGRU, `GatedDeltaNet`, and `GatedDeltaNetReflect`, which doubles the sigmoid of the write strength and is otherwise the same layer. If the argument above is right, the GRU and the layer with $\beta_t \in (0, 2)$ should be correct at every position, including those beyond the training length, while the minGRU and the sigmoid delta rule should fail, and fall to the chance level of $0.5$ once the strings are longer than those they were trained on.

```{.python .input #deltanet-state-tracking-1}
%%tab pytorch
class GatedDeltaNetReflect(d2l.GatedDeltaNet):
    """Gated DeltaNet with write strengths in (0, 2), so that the transition
    can reflect the memory along the key."""
    def memory(self, X, Q, K, V, log_a):
        beta = 2 * d2l.heads_to_batch(torch.sigmoid(self.W_beta(X)),
                                      self.num_heads).squeeze(-1)
        return chunked_delta_rule(Q, K, V, beta, log_a, self.chunk_size)

def running_parity(rng, batch_size, num_steps):
    """Random bits and, at every step, the parity of the bits so far."""
    X = rng.integers(0, 2, (batch_size, num_steps))
    return X, np.cumsum(X, 1) % 2

def train_parity(model, num_updates=1000):
    """Train on strings of 64 bits; return the accuracy on positions 1-64
    and on positions 65-256 of new strings of 256 bits."""
    rng = np.random.default_rng()
    model.to(d2l.try_gpu())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2,
                                  weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda t: (
        d2l.cosine_warmup(t, num_updates, num_updates // 10)))
    d2l.train_queries(model, lambda step: running_parity(rng, 128, 64),
                      num_updates, optimizer, scheduler)
    X, Y = running_parity(np.random.default_rng(1), 1024, 256)
    return (float(d2l.query_accuracy(model, X[:, :64], Y[:, :64])),
            float(d2l.query_accuracy(model, X, Y[:, 64:])))

d = 64  # Width; the delta rules have 4 heads of size 16
parity_mixers = {
    'GRU': lambda: d2l.Recurrent(d2l.init_lecun(
        nn.GRU(d, d, batch_first=True))),
    'minGRU': lambda: d2l.MinGRU(d, d),
    'Gated DeltaNet': lambda: d2l.GatedDeltaNet(
        d, 4, chunk_size=32, timescales=(256, 2048)),
    'Gated DeltaNet, 2σ': lambda: GatedDeltaNetReflect(
        d, 4, chunk_size=32, timescales=(256, 2048))}

def parity_model(mixer):  # Tokens and outputs: the bits 0 and 1
    return d2l.SeqLM(2, d, [mixer(), mixer()])  # Two blocks
```

```{.python .input #deltanet-state-tracking-1}
%%tab jax
class GatedDeltaNetReflect(d2l.GatedDeltaNet):
    """Gated DeltaNet with write strengths in (0, 2), so that the transition
    can reflect the memory along the key."""
    def memory(self, X, Q, K, V, log_a):
        beta = 2 * d2l.heads_to_batch(jax.nn.sigmoid(self.W_beta(X)),
                                      self.num_heads)[..., 0]
        return chunked_delta_rule(Q, K, V, beta, log_a, self.chunk_size)

def running_parity(rng, batch_size, num_steps):
    """Random bits and, at every step, the parity of the bits so far."""
    X = rng.integers(0, 2, (batch_size, num_steps))
    return X, np.cumsum(X, 1) % 2

def train_parity(model, num_updates=1000):
    """Train on strings of 64 bits; return the accuracy on positions 1-64
    and on positions 65-256 of new strings of 256 bits."""
    rng = np.random.default_rng()
    schedule = lambda t: 1e-2 * d2l.cosine_warmup(t, num_updates,
                                                  num_updates // 10)
    optimizer = nnx.Optimizer(model, optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, weight_decay=0.01)), wrt=nnx.Param)
    X, Y = running_parity(np.random.default_rng(1), 1024, 256)
    with jax.default_matmul_precision('highest'):  # Full float32, not TF32
        d2l.train_queries(model, lambda step: running_parity(rng, 128, 64),
                          num_updates, optimizer)
        return (float(d2l.query_accuracy(model, X[:, :64], Y[:, :64])),
                float(d2l.query_accuracy(model, X, Y[:, 64:])))

d = 64  # Width; the delta rules have 4 heads of size 16
lecun = nnx.initializers.lecun_normal()  # For the recurrent matrix
parity_mixers = {
    'GRU': lambda rngs: nnx.RNN(nnx.GRUCell(
        d, d, recurrent_kernel_init=lecun, rngs=rngs), unroll=8),
    'minGRU': lambda rngs: d2l.MinGRU(d, d, rngs=rngs),
    'Gated DeltaNet': lambda rngs: d2l.GatedDeltaNet(
        d, 4, chunk_size=32, timescales=(256, 2048), rngs=rngs),
    'Gated DeltaNet, 2σ': lambda rngs: GatedDeltaNetReflect(
        d, 4, chunk_size=32, timescales=(256, 2048), rngs=rngs)}

def parity_model(mixer):  # Tokens and outputs: the bits 0 and 1
    rngs = nnx.Rngs(d2l.get_key())  # A new initialization each call
    return d2l.SeqLM(2, d, [mixer(rngs), mixer(rngs)],  # Two blocks
                     rngs=rngs)
```

We train two runs of each mixer and report the accuracy on positions 1–64, the lengths seen in training, and on positions 65–256:

```{.python .input #deltanet-state-tracking-2}
%%tab pytorch, jax
print(f'{"":24s}{"positions 1-64":>16s}{"65-256":>12s}')
for name, mixer in parity_mixers.items():
    (a, b), (c, e) = [train_parity(parity_model(mixer)) for _ in range(2)]
    print(f'{name:24s}{a:11.2f}/{c:.2f}{b:7.2f}/{e:.2f}')
```

The GRU is correct at every position in both runs, at the training lengths and beyond them. The minGRU and the sigmoid delta rule are not. On the first 64 positions they sit above chance, the minGRU at about $0.6$ and Gated DeltaNet between $0.7$ and $0.8$, but neither has learned the rule: beyond position 64 both answer at $0.50$, a coin toss. Doubling the write strength changes this. `GatedDeltaNetReflect` reaches $1.00$ on the training lengths in every run, and in most runs it keeps that accuracy out to 256 bits. Linearity, then, is not the obstacle. What the linear recurrences gave up is an update that depends on the state, and whether a matrix memory can track a state anyway is decided by the range of its transition's eigenvalues. One block of minGRU has neither a negative eigenvalue nor a round of state-dependent gating, and it fails.

The reflection is delicate, though. A doubled sigmoid only approaches $\beta_t = 2$, so every flip also shrinks the stored bit a little, and a run that stops short of 2 loses accuracy on long strings (Exercise :exref:`ex_dn-parity`). Harder state tracking, such as following several cups through their swaps, needs more than one reflection per token.

## Memory as Regression

Stepping back, every matrix memory in this chapter—as well as softmax attention—can be understood as fitting a regression predictor from keys $\mathbf{K}_t \in \mathbb{R}^{t \times d_k}$ to values $\mathbf{V}_t \in \mathbb{R}^{t \times d_v}$ on the fly as a sequence is read :cite:`Wang.Shi.Fox.2025`. The exact linear least-squares minimizer of $\sum_{s \le t} \|\mathbf{S}\mathbf{k}_s - \mathbf{v}_s\|^2$ is:

$$
\mathbf{S}_t^\star = \mathbf{V}_t^\top \mathbf{K}_t\,
(\mathbf{K}_t^\top \mathbf{K}_t)^{-1},
$$
:eqlabel:`eq_dn-least-squares`

where $(\mathbf{K}_t^\top \mathbf{K}_t)^{-1}$ is replaced by the pseudoinverse when $t < d_k$. Look at the two factors of :eqref:`eq_dn-least-squares` (:numref:`tab_dn-regression`):

- **Linear attention** keeps only the first factor, the unnormalized cross-moment $\mathbf{V}_t^\top \mathbf{K}_t = \sum_{s \le t} \mathbf{v}_s \mathbf{k}_s^\top$, omitting the Gram inverse $(\mathbf{K}_t^\top \mathbf{K}_t)^{-1}$ that decorrelates overlapping keys.
- **The delta rule** takes one online stochastic gradient step per token toward :eqref:`eq_dn-least-squares`, while **Gated DeltaNet** adds weight decay ($a_t < 1$) before each step. Replacing the explicit gradient step by the proximal implicit step $\arg\min_{\mathbf{S}} \frac{1}{2}\|\mathbf{S} - \mathbf{S}_{t-1}\|^2 + \frac{\beta_t}{2}\|\mathbf{S}\mathbf{k}_t - \mathbf{v}_t\|^2$ yields the normalized step size $\beta_t / (1 + \beta_t\|\mathbf{k}_t\|^2)$ of Longhorn :cite:`Liu.Wang.Wu.ea.2024` (Exercise :exref:`ex_dn-implicit`), while carrying $(\mathbf{K}_t^\top \mathbf{K}_t + \lambda\mathbf{I})^{-1}$ explicitly yields recursive least squares (Exercise :exref:`ex_dn-rls`).
- **Softmax attention** performs *nonparametric* Nadaraya–Watson kernel regression (:numref:`sec_attention-pooling`) at query time over all stored pairs, avoiding parametric compression at the cost of an $O(t)$ KV cache.

:Memories as regressions on the pairs read so far.
:label:`tab_dn-regression`

| Memory | What it fits | How | State |
| :-- | :-- | :-- | :-- |
| Softmax attention | kernel-weighted average of all values | at query time | all pairs |
| Linear attention | $\mathbf{V}^\top\mathbf{K}$, least squares without the key correction | one addition per pair | $d_v \times d_k$ |
| Gated linear attention | the same, weighted toward recent pairs | one addition per pair | $d_v \times d_k$ |
| Delta rule | least squares, approximately | one gradient step per pair | $d_v \times d_k$ |
| Gated DeltaNet | least squares with weight decay, approximately | one gradient step per pair | $d_v \times d_k$ |

In all these architectures, learning operates at two timescales: the *fast weights* of the state $\mathbf{S}_t$ fit the key–value associations of the current context during the forward pass, while the *slow weights* of the network ($\mathbf{W}_{qkv}, \mathbf{W}_a, \mathbf{W}_\beta, \ldots$) are trained across sequences by backpropagation to produce keys, values, queries, and step sizes that make that online fit effective. Even with an optimal write rule, however, a $d_v \times d_k$ matrix has finite rank and finite bit capacity; :numref:`sec_hybrids` measures that hard limit and shows how interleaving a small fraction of full attention layers lifts it.

## Summary

Because additive matrix memories superpose every write to the same key, querying a rebound key returns the sum of its old and new values. The delta rule reads the current prediction $\mathbf{S}_{t-1}\mathbf{k}_t$ before writing and adds only the error correction $\beta_t(\mathbf{v}_t - \mathbf{S}_{t-1}\mathbf{k}_t)\mathbf{k}_t^\top$—one online gradient descent step on $\frac{1}{2}\|\mathbf{S}\mathbf{k}_t - \mathbf{v}_t\|^2$ whose transition $\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top$ selectively contracts the memory along $\mathbf{k}_t$. While targeted erasure outperforms additive writes whenever keys are rebound or correlated, aggressive erasure ($\beta_t = 1$) can erode earlier signals when many random keys overlap. `GatedDeltaNet` pairs an input-dependent write strength $\beta_t$ with a scalar head decay $a_t$ and evaluates the recurrence in parallel chunks via a lower-triangular solve (`delta_rule_writes`) and a chunk-level scan (`chunked_delta_rule`), reliably recovering rebound keys on associative recall.

## Exercises

1. **Order does not matter.** Show that the additive memory after a
   sequence of writes does not depend on the order of the writes. Conclude
   that no read rule, however clever, can recover the latest of $R$ values
   written under one key from an additive memory alone, and compute the
   recall that the best rule achieves when the $R$ values are distinct.
   How does the measured recall after eight writes change when the codebook
   grows from 32 to 1,024 values, and why?
   :exlabel:`ex_dn-order`

1. [code] **Decay instead of correction.** Give the additive write of
   `overwrite_recall` a constant decay, $\mathbf{S} \leftarrow a\mathbf{S} +
   \mathbf{v}\mathbf{k}^\top$.
    1. For $R = 2, 4, 8$, find the largest decay $a$ at which the latest
       value is recalled at least 95% of the time. Why does the answer
       depend on how many writes separate two writes of the same key, and
       why does a smaller decay never hurt in this test?
    1. Now replace the orthonormal keys by 16 random unit keys in width 64.
       Rewrite 8 of them $R$ times as before, write the other 8 once at the
       start, and measure the recall of both kinds of key at the end. Which
       write rule serves both kinds of key?
   :exlabel:`ex_dn-decay`

1. **The implicit step.** Show that the minimizer of
   $\tfrac{1}{2}\|\mathbf{S} - \mathbf{S}_{t-1}\|^2 +
   \tfrac{\beta}{2}\|\mathbf{S}\mathbf{k} - \mathbf{v}\|^2$ is
   $\mathbf{S}_{t-1} - \frac{\beta}{1 + \beta\|\mathbf{k}\|^2}
   (\mathbf{S}_{t-1}\mathbf{k} - \mathbf{v})\mathbf{k}^\top$. (Hint: set
   the gradient to zero and multiply the result by $\mathbf{k}$ to find
   $\mathbf{S}\mathbf{k}$ first.) Give the eigenvalues of its transition,
   and show that the step recalls the new pair exactly only in the limit
   $\beta \to \infty$.
   :exlabel:`ex_dn-implicit`

1. [code] **Recursive least squares.** A regularized least-squares fit,
   which approaches :eqref:`eq_dn-least-squares` as $\lambda \to 0$, can be
   updated one pair at a time if the memory also keeps $\mathbf{P}_t =
   (\mathbf{K}_t^\top \mathbf{K}_t + \lambda\mathbf{I})^{-1}$.
    1. Use the Sherman–Morrison formula to derive the updates of
       $\mathbf{P}_t$ and $\mathbf{S}_t$ after one new pair, and show that
       the update of $\mathbf{S}_t$ is a delta rule that writes along
       $\mathbf{P}_t\mathbf{k}_t$ instead of $\mathbf{k}_t$. How many
       numbers does the state now hold, against $d_v d_k$ for the delta
       rule?
    1. Implement it and compare it with the delta rule on the overwrite
       test. Why does recursive least squares, the best fit of all pairs,
       recall the latest value poorly? What would fix it?
   :exlabel:`ex_dn-rls`

1. [code] **Parity.** ● The write strength of `GatedDeltaNet` is a sigmoid,
   so the eigenvalue $1 - \beta_t$ of its transition along the key lies in
   $(0, 1)$. With $2\sigma(\cdot)$ in place of $\sigma(\cdot)$ it lies in
   $(-1, 1)$.
    1. Even then, a trained layer can only approach $\beta_t = 2$. In the
       parity construction of :numref:`subsec_dn-state-tracking`, write
       $\beta = 2 - \epsilon$ on every counted token, and show that after
       $n$ counted tokens the read is
       $\big(1 - (\epsilon - 1)^n\big)\mathbf{v}$. How fast does the
       difference between the reads after an odd and after an even count
       shrink? For $\epsilon = 0.01$, after how many counted tokens is it
       below 1% of $\|\mathbf{v}\|$, and what does that mean for a readout
       with finite precision?
    1. :numref:`subsec_dn-state-tracking` tested its models at four times
       the training length. Train `Gated DeltaNet, 2σ` again and test it on
       strings of 1,024 and of 4,096 bits. At what length does the accuracy
       begin to fall, and is that length consistent with your answer to the
       first part?
   :exlabel:`ex_dn-parity`

1. [code] **Other routes to rebinding.** The rebinding experiment fixed
   three choices so that only the write rule could correct a value. Relax
   them one at a time and explain each result.
    1. Switch off the decay of gated linear attention by overriding its
       method `decays` (it returns $\log a_t$) to return zeros of the same
       shape, so that every $a_t = 1$, and train again. How does its recall
       change on rewritten keys and on keys written once?
    1. Restore the decay and let each rewrite come at any later position
       (`gap=None` in `task`). Does the gap between the two layers shrink?
       Explain with the ages of the two values.
    1. Train both layers with two blocks instead of one. How can two
       additive layers together correct a value?
   :exlabel:`ex_dn-rebinding`

[Discussions](https://d2l.discourse.group/t/36784)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §12.7]{.kicker}

The Delta Rule and Gated DeltaNet<br>**read-before-write error correction · Householder projection geometry · chunked WY triangular solve · memory as online regression**
:::
:::

::: {.slide title="Why Additive Matrix Memories Fail on Rebound Keys"}
[Motivation]{.kicker}

Programs and narratives constantly update state (`rate = 0.05` $\to$ `rate = 0.07`, or a character travelling from Paris to Rome). When a unit key $\mathbf{k}$ is written $R$ times with values $\mathbf{v}^{(1)}, \ldots, \mathbf{v}^{(R)}$, an additive memory returns their unweighted superposition:

$$\mathbf{S}\mathbf{k} = \mathbf{v}^{(1)} + \mathbf{v}^{(2)} + \cdots + \mathbf{v}^{(R)}$$

Testing $8$ mutually orthonormal keys in $\mathbb{R}^{64}$ (zero cross-key interference) written $R \in \{1, 2, 4, 8\}$ times each:

@!deltanet-a-key-written-twice-1

::: {.d2l-note .warn}
Even with $8\times$ more dimensions than keys, additive recall of the latest value collapses to a coin toss ($0.51$) at $R = 2$ and $0.21$ at $R = 8$. Global decay $a_t < 1$ fades old values only by aging *all* keys indiscriminately.
:::
:::

::: {.slide title="The Delta Rule: One Online Gradient Step on Recall Loss"}
[Read Before Writing]{.kicker}

::: {.cols .vc}
::: {.col}
$$\ell_t(\mathbf{S}) = \tfrac{1}{2}\,\|\mathbf{S}\mathbf{k}_t - \mathbf{v}_t\|^2 \implies \mathbf{S}_t = \mathbf{S}_{t-1} + \beta_t(\mathbf{v}_t - \mathbf{S}_{t-1}\mathbf{k}_t)\mathbf{k}_t^\top$$

- **Read Before Writing:** Reads prediction $\hat{\mathbf{v}}_t = \mathbf{S}_{t-1}\mathbf{k}_t$ first, then writes the scaled error residual $\beta_t(\mathbf{v}_t - \hat{\mathbf{v}}_t)\mathbf{k}_t^\top$.
- **Exact Replacement at $\beta_t = 1$:** For $\|\mathbf{k}_t\| = 1$, $\mathbf{S}_t \mathbf{k}_t = \mathbf{v}_t$ in one step.
:::

::: {.col .fig}
@fig:ch12-mrnn-delta-read-before-write
:::
:::
:::

::: {.slide title="Projection Geometry of the Delta Transition Matrix"}
[Directional Erasure]{.kicker}

::: {.cols .vc}
::: {.col}
Factoring $\mathbf{S}_{t-1}$ out of the delta update reveals the right-multiplied state transition matrix:

$$\mathbf{S}_t = \mathbf{S}_{t-1}(\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top) + \beta_t \mathbf{v}_t \mathbf{k}_t^\top$$

For a unit key $\|\mathbf{k}_t\| = 1$, $\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top$ has eigenvalue **$1$** on the $(d_k - 1)$-dimensional subspace $\mathbf{k}_t^\perp$ and eigenvalue **$1 - \beta_t$** along $\mathbf{k}_t$:

- **$\beta_t = 0$:** Identity $\mathbf{I}$ (memory untouched).
- **$\beta_t \in (0, 1)$:** Partial contraction along $\mathbf{k}_t$.
- **$\beta_t = 1$:** Orthogonal projection $\mathbf{I} - \mathbf{k}_t\mathbf{k}_t^\top$ (erases $\mathbf{k}_t$ completely before writing $\mathbf{v}_t \mathbf{k}_t^\top$).
- **$\beta_t = 2$:** Householder reflection $\mathbf{I} - 2\mathbf{k}_t\mathbf{k}_t^\top$ (eigenvalue $-1$, reversing sign along $\mathbf{k}_t$).
:::

::: {.col .fig}
![](../img/mdl-modernrnn-delta-geometry.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="Unit-Key Stability and Householder Parity Reflections"}
[Spectral Properties]{.kicker}

::: {.cols .vc}
::: {.col}
- **Mandatory Key $\ell_2$-Normalization:** Along $\mathbf{k}_t$, eigenvalue $\lambda_{\mathbf{k}} = 1 - \beta_t\|\mathbf{k}_t\|^2$ stays in $[-1, 1]$ for $\beta_t \in [0, 2]$ only when $\|\mathbf{k}_t\| = 1$.
- **Negative Eigenvalues Unlock Parity:** Positive decays $a_t > 0$ cannot count modulo $2$; at $\beta_t = 2$, Householder reflection ($\lambda_{\mathbf{k}} = -1$) toggles readout $\mathbf{0} \leftrightarrow 2\mathbf{v}$.
:::

::: {.col .fig .big}
@fig:ch12-mrnn-eigenvalue-parity
:::
:::
:::

::: {.slide title="The Hidden Cost of Correction: Signal Erosion"}
[Crowded Random Keys]{.kicker}

::: {.cols .vc}
::: {.col}
$$\mathbb{E}[\mathbf{S}_t \mathbf{k}_i] = \Big(1 - \frac{1}{d_k}\Big)^m \mathbf{v}_i \approx e^{-m / d_k}\mathbf{v}_i$$

- **Collateral Contraction:** Every later random unit key $\mathbf{k}_s$ has expected squared overlap $1/d_k$ with $\mathbf{k}_i$, shrinking $\mathbf{v}_i$ by $1 - 1/d_k$ per step.
- **Crowded Decay:** After $m = 2d_k$ random writes with $\beta = 1$, early signals erode to $e^{-2} \approx 13.5\%$.
:::

::: {.col .fig}
@fig:ch12-mrnn-signal-erosion
:::
:::
:::

::: {.slide title="Benchmarking Four Write Rules Across Three Regimes" layout="code"}
[Additive vs. Delta vs. Least Squares]{.kicker}

Comparing `additive`, `delta, β = 1`, `delta, β = 0.5`, and offline `least squares` ($\mathbf{S}^\star = \mathbf{V}^\top \mathbf{K}(\mathbf{K}^\top \mathbf{K})^+$, fitting all pairs simultaneously) at width $d = 64$:

@!deltanet-what-correction-costs-2
:::

::: {.slide title="When Targeted Erasure Helps—and Why βt Must Be Learned"}
[Interpreting the Write-Rule Table]{.kicker}

The benchmark exposes three distinct regimes that motivate learning an input-dependent gate $\beta_t \in (0, 1)$:

1. **Key rebinding (`8 orthonormal keys, 4 writes`):** `delta` ($\beta = 1$ or $0.5$) achieves **$1.00$ recall** of the latest value, whereas `additive` and unweighted `least squares` average all $4$ values bound to each key ($0.28$).
2. **Correlated keys (`64 keys, ρ = 0.5`):** `additive` piles every value onto the shared key direction (error $5.82$, recall $0.54$), whereas `delta` subtracts the shared-direction prediction before writing (recall **$0.89$** at $\beta = 0.5$).
3. **Crowded independent keys (`128 keys`, `256 keys`):** At $n = 128$, `delta, β = 1` has *lower* squared error than `additive` ($1.06$ vs. $1.99$) yet *worse* recall ($0.53$ vs. $0.96$) because shrinking old reads toward $\mathbf{0}$ caps squared error near $\|\mathbf{0} - \mathbf{v}_j\|^2 = 1$ while eroding early signals. Halving the step to $\beta = 0.5$ restores **$0.82$ recall** with **$0.67$ squared error**.
:::

::: {.slide title="Gated DeltaNet: Global Decay Plus Targeted Overwriting"}
[The Gated DeltaNet Recurrence]{.kicker}

::: {.cols .vc}
::: {.col}
$$\mathbf{S}_t = a_t\,\mathbf{S}_{t-1}(\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top) + \beta_t \mathbf{v}_t \mathbf{k}_t^\top$$

- **Three Operating Modes (Yang et al., 2024):**
  1. **Global Flush ($a_t \to 0$):** Wipes the entire matrix at topic boundaries.
  2. **Surgical Key Update ($a_t \approx 1, \beta_t \to 1$):** Overwrites $\mathbf{k}_t$ while preserving $\mathbf{k}_t^\perp$.
  3. **Gentle Accumulation ($a_t \approx 1, \beta_t \approx 0.2$):** Low-erosion associative write.
:::

::: {.col .fig}
@fig:ch12-mrnn-gated-deltanet-modes
:::
:::
:::

::: {.slide title="Chunked Evaluation via Lower-Triangular Solve (WY Form)"}
[Intra-Chunk Triangular System]{.kicker}

::: {.cols .vc}
::: {.col}
$$\mathbf{u}_t + \beta_t \sum_{s < t} L_{ts}\,(\mathbf{k}_s^\top \mathbf{k}_t)\,\mathbf{u}_s = \beta_t\big(\mathbf{v}_t - \gamma_t \mathbf{S}_0 \mathbf{k}_t\big)$$

- **WY Representation:** Rewrites the delta recurrence as $\mathbf{S}_t = a_t \mathbf{S}_{t-1} + \mathbf{u}_t \mathbf{k}_t^\top$ with corrected values $\mathbf{u}_t$.
- **Parallel $m \times m$ Triangular Solve:** Solves $\mathbf{M}\mathbf{W} = \boldsymbol{\beta} \odot \mathbf{V}$ and $\mathbf{M}\mathbf{Z} = (\boldsymbol{\beta} \odot \boldsymbol{\gamma}) \odot \mathbf{K}$ across all chunks in parallel to obtain $\mathbf{U} = \mathbf{W} - \mathbf{Z}\mathbf{S}_0^\top$.
:::

::: {.col .fig}
@fig:ch12-mrnn-wy-triangular-solve
:::
:::
:::

::: {.slide title="Inter-Chunk Matrix Scan and Numerical Verification"}
[Chunked Algorithm Verified]{.kicker}

Substituting $\mathbf{U} = \mathbf{W} - \mathbf{Z}\mathbf{S}_0^\top$ into the chunk-end state yields an affine matrix recurrence across the $T/m$ chunks:

$$\mathbf{S}_m = \mathbf{S}_0\,\underbrace{(\gamma_m \mathbf{I} - \mathbf{Z}^\top \mathbf{D}\mathbf{K})}_{\mathbf{A}^{(c)} \in \mathbb{R}^{d_k \times d_k}} + \mathbf{W}^\top \mathbf{D}\mathbf{K}$$

- **Balanced complexity:** Composing $\mathbf{A}^{(c)}$ across $T/m$ chunks costs $(T/m)d_k^3$ FLOPs—matching the $O(T d_k^2)$ intra-chunk GEMMs whenever chunk size $m \approx d_k$.

@!deltanet-chunked-computation-3
:::

::: {.slide title="Implementing GatedDeltaNet as a Drop-In GLA Subclass"}
[Clean Object-Oriented Design]{.kicker}

Because `GatedLinearAttention` (§{sec_matrix-state}) isolated its state recurrence inside `memory(X, Q, K, V, log_a)`, `GatedDeltaNet` adds just one linear projection `W_beta` ($d \to h$) and overrides `memory`:

@-deltanet-the-layer

::: {.d2l-note}
Every other component—width-4 causal convolution, $\ell_2$-normalized queries and keys, log-uniform head decay initialization, per-head RMSNorm, and SiLU output gating—is inherited unchanged.
:::
:::

::: {.slide title="Key Rebinding Experiment: Short-Gap Overwrites"}
[Rebinding Benchmark Results]{.kicker}

We compare a 1-block `GatedLinearAttention` model against a 1-block `GatedDeltaNet` model ($d = 64$, $4$ heads of $16 \times 16$) on $16$-pair associative recall where $8$ of the $16$ keys are rebound to a new value just **`gap=2` writes after their initial binding**:

@!deltanet-rebinding-2

- **Why `gap=2` isolates the delta rule:** When a key is overwritten just $2$ pairs after it was first bound, both bindings have nearly the same age—so global decay $a_t$ cannot fade the old value without also erasing keys written once prior to that step.
- **Reliable overwriting:** Adding the $260$-parameter `W_beta` projection lifts rebound-key accuracy to **$0.96$ or more** across runs, ahead of `GatedLinearAttention` (roughly $0.85$–$0.9$) and within about $2$ points of single-write accuracy.
:::

::: {.slide title="Why We Tested 1 Block: Depth Approximates a Delta Step"}
[Multi-Layer Interaction]{.kicker}

::: {.cols .vc}
::: {.col}
- **Two Additive Layers Emulate One Delta Step:** Block 1 (`GLA`) can query $\hat{\mathbf{v}}_t = \mathbf{S}_{t-1}^{(1)}\mathbf{k}_t$ into the residual stream so Block 2 (`GLA`) writes $(\mathbf{v}_t - \hat{\mathbf{v}}_t)\mathbf{k}_t^\top$.
- **Why In-Layer Delta Wins:** `GatedDeltaNet` performs read-before-write inside *every single layer* for arbitrary rebindings $R$, freeing deeper layers for reasoning.
:::

::: {.col .fig .big}
@fig:ch12-mrnn-two-block-emulation
:::
:::
:::

::: {.slide title="The Implicit Proximal Delta Step (Longhorn)"}
[Unconditional Stability Without Key Normalization]{.kicker}

::: {.cols .vc}
::: {.col}
$$\mathbf{S}_t = \mathbf{S}_{t-1} + \frac{\beta_t}{1 + \beta_t \|\mathbf{k}_t\|^2}\,(\mathbf{v}_t - \mathbf{S}_{t-1}\mathbf{k}_t)\mathbf{k}_t^\top$$

- **Implicit Proximal Objective (Liu et al., 2024):** Minimizes $\frac{1}{2}\|\mathbf{S} - \mathbf{S}_{t-1}\|_F^2 + \frac{\beta_t}{2}\|\mathbf{S}\mathbf{k}_t - \mathbf{v}_t\|^2$ at the *new* state $\mathbf{S}_t$.
- **Unconditional Stability:** Directional eigenvalue $\frac{1}{1 + \beta_t\|\mathbf{k}_t\|^2} \in (0, 1)$ for any $\|\mathbf{k}_t\|$ and $\beta_t > 0$.
:::

::: {.col .fig}
@fig:ch12-mrnn-explicit-vs-implicit-step
:::
:::
:::

::: {.slide title="Unifying Sequence Mixers as Online Regression Solvers"}
[Test-Time Regression View]{.kicker}

::: {.cols .vc}
::: {.col}
$$\min_{\mathbf{S} \in \mathbb{R}^{d_v \times d_k}} \sum_{s=1}^{t} \tfrac{1}{2}\,\|\mathbf{S}\mathbf{k}_s - \mathbf{v}_s\|^2$$

- **Fast Weights ($\mathbf{S}_t$):** Solve an in-context linear regression $\mathbf{S}_t \mathbf{k}_s \approx \mathbf{v}_s$ across tokens $1\ldots t$ during the forward pass.
- **Slow Weights ($\theta$):** Trained by outer-loop AdamW to shape $(\mathbf{q}_t, \mathbf{k}_t, \mathbf{v}_t, a_t, \beta_t)$.
:::

::: {.col .fig}
@fig:ch12-mrnn-fast-slow-weights
:::
:::
:::

::: {.slide title="Taxonomy of Sequence Memories as Regression Algorithms"}
[The Regression Table]{.kicker}

| Sequence Memory | In-Context Regression Objective / Estimator | Online Update per Token Pair $(\mathbf{k}_t, \mathbf{v}_t)$ | Carried State |
| :--- | :--- | :--- | :---: |
| **Linear Attention** | Unnormalized cross-moment $\mathbf{V}_t^\top \mathbf{K}_t$ (omits $(\mathbf{K}_t^\top\mathbf{K}_t)^{-1}$) | $\mathbf{S}_t = \mathbf{S}_{t-1} + \mathbf{v}_t \mathbf{k}_t^\top$ | $d_v \times d_k$ |
| **Gated Linear Attention** | Exponentially weighted cross-moment | $\mathbf{S}_t = a_t \mathbf{S}_{t-1} + \mathbf{v}_t \mathbf{k}_t^\top$ | $d_v \times d_k$ |
| **DeltaNet** | Online SGD on $\frac{1}{2}\|\mathbf{S}\mathbf{k}_t - \mathbf{v}_t\|^2$ | $\mathbf{S}_t = \mathbf{S}_{t-1} + \beta_t(\mathbf{v}_t - \mathbf{S}_{t-1}\mathbf{k}_t)\mathbf{k}_t^\top$ | $d_v \times d_k$ |
| **Gated DeltaNet** | Online SGD with weight decay $a_t \in (0, 1)$ | $\mathbf{S}_t = a_t\mathbf{S}_{t-1}(\mathbf{I} - \beta_t\mathbf{k}_t\mathbf{k}_t^\top) + \beta_t\mathbf{v}_t\mathbf{k}_t^\top$ | $d_v \times d_k$ |
| **Longhorn (Implicit Step)** | Proximal step $\min_{\mathbf{S}} \frac{1}{2}\|\mathbf{S} - \mathbf{S}_{t-1}\|^2 + \frac{\beta_t}{2}\|\mathbf{S}\mathbf{k}_t - \mathbf{v}_t\|^2$ | Delta rule with normalized step $\frac{\beta_t}{1 + \beta_t\|\mathbf{k}_t\|^2}$ | $d_v \times d_k$ |
| **Recursive Least Squares** | Exact ridge regression $\mathbf{V}_t^\top\mathbf{K}_t(\mathbf{K}_t^\top\mathbf{K}_t + \lambda\mathbf{I})^{-1}$ | Sherman–Morrison update along $\mathbf{P}_t\mathbf{k}_t$ | $d_v d_k + d_k^2$ |
| **Softmax Attention** | Nonparametric Nadaraya–Watson kernel regression | Appends $(\mathbf{k}_t, \mathbf{v}_t)$ to cache; evaluates at query time | $t \times (d_k + d_v)$ |
:::

::: {.slide title="Concept Check: Delta Transitions and Chunked Solves"}
[Check Your Understanding]{.kicker}

Test your mastery of the delta rule with these three questions:

1. Suppose $\mathbf{S}_{t-1}\mathbf{k}_t = \hat{\mathbf{v}}_t$. If we apply the delta rule with a unit key $\|\mathbf{k}_t\| = 1$ and write strength $\beta_t = 0.5$, what does an immediate readout $\mathbf{S}_t \mathbf{k}_t$ return, and what happens to $\mathbf{S}_t \mathbf{q}$ for any query $\mathbf{q} \perp \mathbf{k}_t$?
2. Why does the implicit proximal update (Longhorn) replace $\beta_t$ with $\tilde{\beta}_t = \beta_t / (1 + \beta_t\|\mathbf{k}_t\|^2)$, and why can't an implicit step compute exact parity?
3. In `chunked_delta_rule`, why is the intra-chunk system matrix $\mathbf{M} = \mathbf{I} + \textrm{diag}(\boldsymbol{\beta})\,\textrm{tril}_{-1}(\mathbf{L} \odot \mathbf{K}\mathbf{K}^\top)$ unit lower-triangular?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. $\mathbf{S}_t \mathbf{k}_t = (1 - 0.5)\hat{\mathbf{v}}_t + 0.5\mathbf{v}_t = \mathbf{\tfrac{1}{2}(\hat{\mathbf{v}}_t + \mathbf{v}_t)}$ (halfway blend); for $\mathbf{q} \perp \mathbf{k}_t$, $\mathbf{k}_t^\top \mathbf{q} = 0$, so $\mathbf{S}_t \mathbf{q} = \mathbf{\mathbf{S}_{t-1}\mathbf{q}}$ (**completely unchanged**).
2. Solving $\mathbf{S} - \mathbf{S}_{t-1} + \beta_t(\mathbf{S}\mathbf{k}_t - \mathbf{v}_t)\mathbf{k}_t^\top = \mathbf{0}$ uses the *post-update* prediction $\mathbf{S}\mathbf{k}_t$, yielding directional eigenvalue $1 - \tilde{\beta}_t\|\mathbf{k}_t\|^2 = \frac{1}{1 + \beta_t\|\mathbf{k}_t\|^2} \in (0, 1)$ for all $\beta_t > 0$—preventing overshoot without key normalization, but excluding negative eigenvalues ($\lambda = -1$) needed for parity reflections.
3. Because causality ensures the corrected value $\mathbf{u}_t$ at step $t$ depends only on earlier writes $s < t$ within the chunk ($\textrm{tril}_{-1}$ has zeros on and above the diagonal, so $\mathbf{M}$ has $1$s on the diagonal).
:::
:::

::: {.slide title="Summary: The Delta Rule and Gated DeltaNet"}
[Key Takeaways]{.kicker}

| Concept / Component | Mathematical Form | Key Capability & Trade-Off |
| :--- | :--- | :--- |
| **The Delta Rule** | $\mathbf{S}_t = \mathbf{S}_{t-1} + \beta_t(\mathbf{v}_t - \mathbf{S}_{t-1}\mathbf{k}_t)\mathbf{k}_t^\top$ | Online gradient step on $\frac{1}{2}\|\mathbf{S}\mathbf{k}_t - \mathbf{v}_t\|^2$; reads before writing |
| **Transition Geometry** | $\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top$ ($\|\mathbf{k}_t\| = 1$) | Eigenvalue $1$ on $\mathbf{k}_t^\perp$, $1 - \beta_t$ along $\mathbf{k}_t$ (projection at $\beta=1$, reflection at $\beta=2$) |
| **Signal Erosion Cost** | $(1 - 1/d_k)^m \approx e^{-m/d_k}$ after $m$ random writes | Aggressive $\beta_t = 1$ erodes early overlapping keys; learned $\beta_t \in (0, 1)$ balances both |
| **Gated DeltaNet** | $\mathbf{S}_t = a_t\mathbf{S}_{t-1}(\mathbf{I} - \beta_t\mathbf{k}_t\mathbf{k}_t^\top) + \beta_t\mathbf{v}_t\mathbf{k}_t^\top$ | Combines global head decay $a_t$ with targeted key overwriting $\beta_t$ ($0.96+$ on rebinding) |
| **Chunked WY Algorithm** | Triangular solve $\mathbf{M}\mathbf{W}, \mathbf{M}\mathbf{Z}$ + chunk scan | Evaluates $d_k \times d_k$ matrix transitions in $O(T d_k^2)$ work via parallel Tensor Core GEMMs |
:::

