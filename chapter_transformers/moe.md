# Mixture of Experts
:label:`sec_moe`

The previous section counted the cost of training at about $6ND$
operations for $N$ parameters and $D$ tokens, and the cost of generating a
token at about $2N$ (:numref:`sec_scaling-laws`). Both counts rest on a
property of every model in this chapter: it is *dense*, and each token
passes through every parameter. A model with twice the parameters, which
can store more of what it reads, therefore costs twice as much to train and
to run. Yet
most of what a large model knows is of no use for any one token.
Continuing "the integral of" draws on different knowledge from continuing
"Act I, Scene". Can a model hold more parameters without spending more
computation on each token?

A *mixture of experts* (MoE) does this by running only part of the model
for each token. It keeps $E$ copies of each block's feed-forward network,
the *experts*, and a small learned *router* sends each token to only $k$
of them. The stored parameters then grow with $E$, while the computation
per token grows only with $k$. DeepSeek-V3, for example, stores 671 billion
parameters and uses 37 billion of them for each token
:cite:`Liu.Feng.Xue.ea.2024`.

The idea is old. :citet:`Jacobs.Jordan.Nowlan.ea.1991` trained a committee
of specialist networks under a gating network, and
:citet:`Shazeer.Mirhoseini.Maziarz.ea.2017` made the gate sparse, so that
only a few out of thousands of experts run for each input. It is harder to get
right than it sounds, because nothing in the layer asks the router to share
the work fairly. This section counts stored and active parameters, builds
the layer, asks why the tokens crowd onto a few experts and what can be
done about it, and finally trains a small GPT with eight experts per block
against its dense twin, on enough text for the extra parameters to have
something to learn.

```{.python .input #moe-mixture-of-experts}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #moe-mixture-of-experts}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## Conditional Computation
:label:`subsec_moe-conditional`

The feed-forward network is the natural place for experts. A block of width
$d$ spends $4d^2$ parameters on the four attention projections and, with
the SwiGLU network of :numref:`sec_transformer-block` (three matrices of
size $d \times 8d/3$), $8d^2$ on the feed-forward network: two thirds of
the block. And while attention mixes information between positions, the
feed-forward network transforms each position on its own, so one token can
be handed to a different network than its neighbor without affecting it.

Replacing the network by $E$ experts, of which each token uses $k$,
multiplies the stored feed-forward parameters by $E$ and the active ones by
$k$, and adds a router, a $d \times E$ matrix per layer. At about two
operations per active parameter and token, the cost of a token follows the
active count. The function below counts both for a stack of expert layers.
We apply it to the model that we train later in this section and to two
released ones. Mixtral 8x7B has 8 full-width experts per layer, of which
each token uses 2 :cite:`Jiang.Sablayrolles.Roux.ea.2024`. DeepSeek-V3 has
256 narrow experts per layer, of which each token uses 8, plus one *shared*
expert that every token uses :cite:`Liu.Feng.Xue.ea.2024`.

```{.python .input #moe-conditional-computation}
%%tab pytorch, jax
def moe_accounting(name, d, width, num_experts, num_active, num_layers,
                   shared=0, unit=1e9):
    """Expert parameters stored, and expert and router parameters active
    per token, in a stack of SwiGLU expert layers."""
    per_expert = 3 * d * width          # SwiGLU: three d x width matrices
    store = num_layers * (num_experts + shared) * per_expert
    active = num_layers * ((num_active + shared) * per_expert
                           + d * num_experts)      # + the router
    tag = 'B' if unit == 1e9 else 'M'
    print(f'{name:>12}: {store/unit:5.3g}{tag} in experts, '
          f'{active/unit:5.3g}{tag} active per token, '
          f'ratio {store/active:4.1f}x')
    return store, active

moe_accounting('ours', 128, 341, 8, 1, 2, unit=1e6)
moe_accounting('DeepSeek-V3', 7168, 2048, 256, 8, 58, shared=1)
store, active = moe_accounting('Mixtral 8x7B', 4096, 14336, 8, 2, 32)
attn = 32 * (2 * 4096 * 4096 + 2 * 4096 * 1024)  # GQA: 32 Q, 8 KV heads
emb = 2 * 32000 * 4096                           # embedding + output head
print(f'{"":>12}  with attention and embeddings: '
      f'{(store + attn + emb)/1e9:.1f}B stored, '
      f'{(active + attn + emb)/1e9:.1f}B active')
```

DeepSeek-V3's 58 expert layers store 656B of its 671B parameters, and a
token uses 23B of them: the layers store 28 times what a token uses. The
37B of its model card also count attention, the embeddings, and three dense
layers at the bottom of the stack, which our count of the expert layers
leaves out. Mixtral's name invites a different misreading. Eight 7B models
would hold 56B, but the experts share attention and the embeddings: adding
the attention layers, which share each key and value head among four query
heads (:numref:`sec_kv-cache-reduction`), and the separate embedding and
output matrices to the experts gives 46.7B stored and 12.9B active
parameters. The name "8x7B" thus describes neither what the model stores
nor the work per token: a token uses 12.9B of the 46.7B, a little over a
quarter. Our own model has Mixtral's eight experts per layer but sends each
token to only one of them, so its expert layers store about eight times
what a token uses. The printed ratio is 7.9 because every token also uses
the router.

The count rests on two assumptions. First, it counts only computation.
All $E$ experts must sit in accelerator memory, and during training each
of them receives gradients from the tokens routed to it. Large models
therefore spread their experts over many devices and send every token to
its experts and back, a systems problem that
:numref:`sec_lm_expert_parallel` takes up. Second, the count assumes that
the router spreads tokens evenly over the experts. If it sends most tokens
to a few of them, the other experts hold parameters that hardly ever run,
and the model behaves like a smaller one.

## A Mixture-of-Experts Layer

The layer replaces the feed-forward network of a block, and nothing else:
attention, normalization, and the residual stream stay as they are. We
first define what it computes and then how to compute it without throwing
the savings away.

### Top-$k$ Routing

A linear router scores the $E$ experts for a token
$\mathbf{x} \in \mathbb{R}^d$, the $k$ largest scores select experts, and
the outputs of the selected experts are added, each weighted by the
router's probability:

$$
\mathrm{MoE}(\mathbf{x}) = \sum_{i \in \mathcal{E}_k(\mathbf{x})} p_i(\mathbf{x})\, \mathrm{FFN}_i(\mathbf{x}),
\qquad
\mathbf{p}(\mathbf{x}) = \mathrm{softmax}(\mathbf{W}_r \mathbf{x}),
$$
:eqlabel:`eq_moe-layer`

where $\mathbf{W}_r \in \mathbb{R}^{E \times d}$ is the router and
$\mathcal{E}_k(\mathbf{x})$ holds the indices of the $k$ largest entries of
$\mathbf{p}(\mathbf{x})$. The layer is thus a feed-forward network chosen
per token: two tokens of the same sentence may use entirely different
parameters. Because each token chooses its experts without regard to the
others, this is called *token-choice* routing, and nothing in it keeps the
choices from piling onto a few experts. :numref:`fig_moe-layer` follows
one token through a layer with $E = 8$ and $k = 2$.

![A mixture-of-experts layer with eight experts, two of which are active per token. The router scores every expert. The two most probable experts run, and their outputs are weighted by their probabilities and added. The other six experts are stored but do no work for this token.](../img/mdl-transformers-moe.svg)
:label:`fig_moe-layer`

Selecting the top $k$ is a discrete decision and has no gradient. How,
then, does the router learn? Through the weights. The weight of a selected
expert is its probability $p_i$ itself, as in the Switch transformer
:cite:`fedus2022switch`, not a probability renormalized over the $k$
selected experts: in :numref:`fig_moe-layer` the two weights sum to 0.55,
not to one. This weight has a gradient. If a larger contribution of
expert $i$ would lower the loss, the gradient raises $p_i$, and with it
the chance that the router selects expert $i$ for similar tokens. Through the softmax, every logit
receives a gradient, including those of the experts that were not
selected. Mixtral instead renormalizes the weights of the selected experts
to sum to one :cite:`Jiang.Sablayrolles.Roux.ea.2024`. The router then
learns only how to divide the weight among the selected experts, and for
top-1 routing, where the single weight is always one, it learns nothing
from the loss at all. With raw probabilities it still learns, so what
looks like a detail of normalization decides whether a top-1 router learns
anything. The first exercise works out these gradients.

### Dispatching Tokens to Experts

Evaluating :eqref:`eq_moe-layer` literally, by running all $E$ experts on
every token and multiplying the outputs of the unselected ones by zero,
gives the right answer at $E/k$ times the necessary cost and saves nothing.
An efficient layer brings each token to its experts instead. A minibatch of
$n$ tokens makes $nk$ *assignments*, pairs of a token and one of its
experts, and how many of them each expert receives changes from batch to
batch. To run all experts as one batched product, however, every expert
needs the same number of rows. The layer therefore gives each expert a
buffer of $S = \lceil c\, nk/E \rceil$ slots, where the *capacity factor* $c$ leaves room for an uneven spread.
The Switch transformer trains with $c = 1.25$ :cite:`fedus2022switch`, a
quarter more room than a perfectly even spread would need.

Which assignment goes into which slot, and which one is left out, is
easiest to see on a case small enough to write down. Let us take six
assignments to $E = 3$ experts with $S = 3$ slots each, routed to the
experts $(2, 0, 2, 2, 1, 2)$. Expert 2 receives four assignments and has
room for three, so one of them must go. The rule is first come, first
served. Each assignment gets a place in its expert's queue, in the order
of the tokens and counting from 0, and the assignment at place $r$ goes
into slot $r$ of its expert's buffer. The six places are
$(0, 0, 1, 2, 0, 3)$, and the last assignment, at place 3 of expert 2,
finds no slot: it is *dropped*.

The cell computes the places the way the layer will, with a cumulative
sum over a one-hot encoding of the experts. Row $j$ of the cumulative sum
counts each expert's assignments among the first $j + 1$, so its entry at
the row's own expert, less one, is the place.

The buffers lie end to end in one array, so the assignment at place $r$ of
expert $e$ goes to slot $eS + r$. Every dropped assignment goes to slot
$ES = 9$, one past the end:

```{.python .input #moe-dispatching-tokens-to-experts-2}
%%tab pytorch, jax
E, S = 3, 3                         # three experts, three slots each
e = np.array([2, 0, 2, 2, 1, 2])    # the expert of each assignment
onehot = np.eye(E, dtype=int)[e]    # (6, E)
counts = onehot.cumsum(0)           # assignments to each expert so far
rank = (counts * onehot).sum(1) - 1   # place in its expert's queue
slot = np.where(rank < S, e * S + rank, E * S)  # E S: dropped
print('expert  one-hot  cumulative  place  slot')
for ei, oh, cs, r, sl in zip(e, onehot, counts, rank, slot):
    note = '  dropped' if sl == E * S else ''
    print(f'{ei:6}  {oh}  {cs!s:10}  {r:5}  {sl:4}{note}')
```

First come, first served has a consequence that is easy to miss. When a
buffer overflows, the assignments that lose are those of the tokens late in
the flattened minibatch, so the last sequences of a batch lose their
experts first. A token whose assignment is dropped keeps the outputs of its
other experts and, through the residual connection, its input.

Dropping is a choice of this implementation, and of the early large
mixtures GShard and Switch :cite:`Lepikhin.Lee.Xu.ea.2021,fedus2022switch`,
not a property of mixtures of experts. An implementation that sorts the
tokens by expert and runs each expert on a segment of whatever length it
receives drops nothing, and DeepSeek-V3 drops no tokens in training
:cite:`Liu.Feng.Xue.ea.2024`. We use fixed buffers because every shape is
then known before the routing is, which keeps the code short and lets it be
compiled once.

With the tokens in their buffers, all $E$ experts run at once, as one
batched matrix multiplication over buffers of shape $(E, S, d)$, and each
token collects its $k$ outputs, weighted by their probabilities. The work
is that of $ES \approx cnk$ rows, a fraction $ck/E$ of running every
expert on every token. The layer below does all of this:

```{.python .input #moe-a-mixture-of-experts-layer-1}
%%tab pytorch
class MoELayer(nn.Module):
    """Token-choice top-k mixture of SwiGLU d2l.FeedForward experts, each
    with a buffer of S = capacity_factor * n k / E slots for n tokens."""
    def __init__(self, num_hiddens, num_experts, num_active,
                 capacity_factor=1.25):
        super().__init__()
        self.num_experts, self.num_active = num_experts, num_active
        self.capacity_factor = capacity_factor
        self.router = nn.Linear(num_hiddens, num_experts, bias=False)
        self.experts = nn.ModuleList([d2l.FeedForward(num_hiddens)
                                      for _ in range(num_experts)])
        # Added to the probabilities for the selection only; zero until a
        # balancing method moves it
        self.register_buffer('expert_bias', torch.zeros(num_experts))
        # Assignments per expert, dropped or not, and the dropped ones,
        # counted by record() until reset_stats()
        self.register_buffer('load', torch.zeros(num_experts))
        self.register_buffer('dropped', torch.zeros(()))

    def forward(self, X):
        x = X.reshape(-1, X.shape[-1])                 # (n, d): the tokens
        (n, d), E, k = x.shape, self.num_experts, self.num_active
        S = math.ceil(self.capacity_factor * n * k / E)  # slots per expert
        P = F.softmax(self.router(x), -1)              # (n, E)
        idx = (P + self.expert_bias).topk(k).indices   # (n, k)
        e = idx.flatten()                              # expert of each pair
        onehot = F.one_hot(e, E)                       # (n k, E)
        rank = (onehot.cumsum(0) * onehot).sum(1) - 1  # place in its queue
        slot = torch.where(rank < S, e * S + rank, E * S)  # E S: dropped
        buf = x.new_zeros(E * S + 1, d)                # + an overflow row
        buf[slot] = x.repeat_interleave(k, 0)
        Y = self.run_experts(buf[:-1].reshape(E, S, d)).reshape(E * S, d)
        Y = torch.cat([Y, Y.new_zeros(1, d)])[slot]    # dropped: zeros
        weights = P.gather(-1, idx)                    # (n, k): the p_i
        out = (weights[..., None] * Y.reshape(n, k, d)).sum(1)
        self.record(onehot.sum(0), (rank >= S).sum(), P)
        return out.reshape(X.shape)

    def run_experts(self, H):
        """Each expert's SwiGLU on its own buffer: (E, S, d) -> (E, S, d)."""
        W = lambda name: torch.stack(  # (E, d_in, d_out): one per expert
            [getattr(e, name).weight for e in self.experts]).mT
        return (F.silu(H @ W('W_g')) * (H @ W('W_1'))) @ W('W_2')

    def record(self, counts, num_dropped, P):
        """Add a batch's assignments per expert and drops to the totals,
        and keep its balancing loss, a measure of imbalance."""
        self.load += counts
        self.dropped += num_dropped
        f = counts / counts.sum()                      # this batch's shares
        self.aux_loss = self.num_experts * (f * P.mean(0)).sum()

    def reset_stats(self):
        self.load.zero_()
        self.dropped.zero_()
```

```{.python .input #moe-a-mixture-of-experts-layer-1}
%%tab jax
class MoELayer(nnx.Module):
    """Token-choice top-k mixture of SwiGLU d2l.FeedForward experts, each
    with a buffer of S = capacity_factor * n k / E slots for n tokens."""
    def __init__(self, num_hiddens, num_experts, num_active,
                 capacity_factor=1.25, *, rngs):
        self.num_experts, self.num_active = num_experts, num_active
        self.capacity_factor = capacity_factor
        self.router = d2l.Linear(num_hiddens, num_experts, use_bias=False,
                                 rngs=rngs)
        self.experts = nnx.List([d2l.FeedForward(num_hiddens, rngs=rngs)
                                 for _ in range(num_experts)])
        # Added to the probabilities for the selection only; zero until a
        # balancing method moves it
        self.expert_bias = nnx.Variable(jnp.zeros(num_experts))
        # Assignments per expert, dropped or not, and the dropped ones,
        # counted by record() until reset_stats(); the last batch's
        # balancing loss
        self.load = nnx.Variable(jnp.zeros(num_experts))
        self.dropped = nnx.Variable(jnp.zeros(()))
        self.aux_loss = nnx.Variable(jnp.zeros(()))

    def __call__(self, X):
        x = X.reshape(-1, X.shape[-1])                 # (n, d): the tokens
        (n, d), E, k = x.shape, self.num_experts, self.num_active
        S = math.ceil(self.capacity_factor * n * k / E)  # slots per expert
        P = jax.nn.softmax(self.router(x), -1)         # (n, E)
        _, idx = jax.lax.top_k(P + self.expert_bias[...], k)  # (n, k)
        e = idx.ravel()                                # expert of each pair
        onehot = jax.nn.one_hot(e, E, dtype=jnp.int32)  # (n k, E)
        rank = (onehot.cumsum(0) * onehot).sum(1) - 1  # place in its queue
        slot = jnp.where(rank < S, e * S + rank, E * S)  # E S: dropped
        buf = jnp.zeros((E * S, d), x.dtype).at[slot].set(
            jnp.repeat(x, k, 0), mode='drop')
        Y = self.run_experts(buf.reshape(E, S, d)).reshape(E * S, d)
        Y = Y.at[slot].get(mode='fill', fill_value=0)  # dropped: zeros
        weights = jnp.take_along_axis(P, idx, -1)      # (n, k): the p_i
        out = (weights[..., None] * Y.reshape(n, k, d)).sum(1)
        self.record(onehot.sum(0), (rank >= S).sum(), P)
        return out.reshape(X.shape)

    def run_experts(self, H):
        """Each expert's SwiGLU on its own buffer: (E, S, d) -> (E, S, d)."""
        W = lambda name: jnp.stack(  # (E, d_in, d_out): one per expert
            [getattr(e, name).kernel[...] for e in self.experts])
        return (jax.nn.silu(H @ W('W_g')) * (H @ W('W_1'))) @ W('W_2')

    def record(self, counts, num_dropped, P):
        """Add a batch's assignments per expert and drops to the totals,
        and keep its balancing loss, a measure of imbalance."""
        self.load[...] += counts
        self.dropped[...] += num_dropped
        f = counts / counts.sum()                      # this batch's shares
        self.aux_loss[...] = self.num_experts * (f * P.mean(0)).sum()

    def reset_stats(self):
        self.load[...] = jnp.zeros_like(self.load[...])
        self.dropped[...] = jnp.zeros(())
```

`run_experts` stacks each of the three SwiGLU matrices across the experts
of width $w$, $W_g$ and $W_1$ into arrays of shape $(E, d, w)$ and $W_2$
into one of shape $(E, w, d)$, and applies the SwiGLU of
:numref:`sec_transformer-block` as three batched products, one matrix per
buffer. It reads the experts' `W_g`, `W_1`, and `W_2` by name, so it
assumes their default SwiGLU without biases. The layer also counts the
assignments each expert receives and keeps a selection bias,
`expert_bias`, which stays zero until the balancing methods of
:numref:`subsec_moe-balance-comparison` put both to use; they can be
ignored for now.

:begin_tab:`pytorch`
The buffer has one row more than the $ES$ slots. Every dropped assignment
is written into that last row, which the experts never see, and reading the
outputs back through `slot` returns the appended row of zeros for it.
:end_tab:

:begin_tab:`jax`
Under `jit`, every array needs a shape that is known when the function is
compiled. Here every shape follows from the shape of `X`, since $S$
depends on $n$ only; the routing decides which rows go where, not how many
there are. Indexing with `mode='drop'` skips the updates whose index lies
past the end of the array, the dropped assignments, and `mode='fill'` reads
zeros for them.
:end_tab:

Two helpers find the expert layers of a model and count the parameters that
one token uses, which are all parameters except the unselected experts:

```{.python .input #moe-dispatching-tokens-to-experts-1}
%%tab pytorch
def moe_layers(model):
    """The MoELayer modules of a model."""
    return [m for m in model.modules() if isinstance(m, MoELayer)]

def active_params(model):
    """Parameters used per token: all but the unselected experts."""
    return d2l.num_params(model) - sum(
        d2l.num_params(l.experts) * (l.num_experts - l.num_active)
        // l.num_experts for l in moe_layers(model))
```

```{.python .input #moe-dispatching-tokens-to-experts-1}
%%tab jax
def moe_layers(model):
    """The MoELayer modules of a model."""
    return [m for _, m in nnx.iter_graph(model) if isinstance(m, MoELayer)]

def active_params(model):
    """Parameters used per token: all but the unselected experts."""
    return d2l.num_params(model) - sum(
        d2l.num_params(l.experts) * (l.num_experts - l.num_active)
        // l.num_experts for l in moe_layers(model))
```

Since the layer maps $(n, d)$ to $(n, d)$ like any feed-forward network, it
fits into `d2l.TransformerBlock` through the block's `ffn_factory`
argument, which builds the feed-forward module of the block. We check the
output shape of such a block on 400 random tokens and count the parameters
of a layer with eight experts, two of them active:

```{.python .input #moe-a-mixture-of-experts-layer-2}
%%tab pytorch
blk = d2l.TransformerBlock(256, num_heads=8,
                           ffn_factory=lambda: MoELayer(256, 8, 2))
X = torch.randn(4, 100, 256)
d2l.check_shape(blk(X), X.shape)  # The expert layer fits into the block
moe = MoELayer(256, num_experts=8, num_active=2)
print(f'{d2l.num_params(moe)} parameters stored, '
      f'{active_params(moe)} active per token')
```

```{.python .input #moe-a-mixture-of-experts-layer-2}
%%tab jax
blk = d2l.TransformerBlock(
    256, num_heads=8, rngs=nnx.Rngs(d2l.get_key()),
    ffn_factory=lambda rngs: MoELayer(256, 8, 2, rngs=rngs))
X = jax.random.normal(d2l.get_key(), (4, 100, 256))
d2l.check_shape(blk(X), X.shape)  # The expert layer fits into the block
moe = MoELayer(256, num_experts=8, num_active=2,
               rngs=nnx.Rngs(d2l.get_key()))
print(f'{d2l.num_params(moe)} parameters stored, '
      f'{active_params(moe)} active per token')
```

The layer stores four times what a token uses: two experts out of eight,
plus a router too small to move the ratio.

## Routing Collapse and Load Balancing

How evenly does a real router spread real text? The answer decides how many
assignments the buffers drop and how many experts do any work.

### Uneven Load at Initialization
:label:`subsec_moe-init-load`

Let us begin with a router that has learned nothing. Its weights are small
random numbers, and a random linear map spreads random vectors over the
experts about evenly. Text, however, does not arrive as random vectors.
Random vectors share no common direction; the router inputs of a text do.
Whatever part $\mathbf{m}$ of the input is the same at every position adds
the same scores $\mathbf{W}_r \mathbf{m}$ to the logits of every token, and
the expert with the largest of them gets a head start everywhere.

The byte-level GPT of :numref:`sec_gpt`, which our experiments use, has
such shared parts. In its first block, the router sees the embedding of
the current byte plus what attention has gathered from the earlier bytes,
which in a fresh model is much the same blend at every position. Most
occurrences of a byte therefore look alike to the router, and they go to
the same expert. Bytes are far from equally frequent: nearly one byte in
five is a space (18.9% of the validation text, as the experiment below
prints), more than the $1.25/8 \approx 16\%$ of the assignments that a
buffer holds at $c = 1.25$. One byte value alone can overflow a buffer.
In the second block, the common blend can tilt many tokens toward the
same expert.

We should therefore expect a fresh model to be unbalanced, and its busiest
buffers to overflow, before it has seen a single gradient. The first row of
the experiment in :numref:`subsec_moe-balance-comparison` tests this
prediction.

### Positive Feedback in Routing
:label:`subsec_moe-feedback`

Training can make matters worse. Follow the feedback loop. Suppose some
expert is, by chance, slightly better than its peers on the tokens it
receives. The loss gradient then rewards it twice: its parameters improve
on those tokens, *and* the router's gradient raises its probability,
because weighting a helpful expert more heavily lowers the loss. More
tokens bring more gradient signal, which makes a better expert, which
attracts more tokens. The experts that lose the early rounds receive ever
fewer tokens, learn ever more slowly, and lose probability. This loop can
end in *routing collapse*: a few experts receive most of the assignments,
and much of the stored capacity goes unused.
:citet:`Shazeer.Mirhoseini.Maziarz.ea.2017` observed this imbalance and
added balancing terms to the training loss. Whether the loop runs in a
given model can be tested by freezing the router, so that the router
itself cannot learn, and the experiment below does so.

### An Auxiliary Balancing Loss

The classic remedy adds a differentiable penalty for imbalance to the
training loss. For a minibatch, let $f_i$ be the fraction of the
assignments that expert $i$ received. It is a count, so no gradient flows
through it. Let $\bar{p}_i$ be the mean of the router's probability
$p_i(\mathbf{x})$ over the tokens of the minibatch. The loss of the Switch
transformer :cite:`fedus2022switch`, which descends from GShard's
:cite:`Lepikhin.Lee.Xu.ea.2021`, multiplies the two:

$$
\mathcal{L}_{\mathrm{balance}} = E \sum_{i=1}^{E} f_i\, \bar{p}_i.
$$
:eqlabel:`eq_moe-aux`

The loss is large when the experts that receive many tokens also receive
high probability. For uniform routing, $f_i = \bar{p}_i = 1/E$ and the
loss equals $1$. If one expert receives every token with probability
close to one, it approaches $E$.

The balancing loss is added to the language-modeling loss with a small
weight $\lambda_\textrm{bal}$, 0.01 in the Switch transformer. Its
gradient flows through $\bar{p}$ only and lowers, for every token, the
probability of the busiest experts, whatever the token needs. This is its
drawback. The two gradients are added at every step, and they pull in
opposite directions whenever some tokens do need a busy expert. With
$\lambda_\textrm{bal}$ too large, routing serves balance rather than
prediction; with $\lambda_\textrm{bal}$ too small, the imbalance persists
:cite:`Wang.Gao.Zeng.ea.2024`.

### Balancing Without a Loss

The conflict disappears if balance acts on the selection of experts alone
and adds nothing to the loss. :citet:`Wang.Gao.Zeng.ea.2024` balance the
load this way. A bias $b_i$ per expert is added to the probabilities for
the selection, and only for the selection. After each optimizer step, a
controller moves each bias by a fixed amount $u$ toward balance:

$$
\mathcal{E}_k(\mathbf{x}) = \operatorname{argtop}_k \big(p_i(\mathbf{x}) + b_i\big),
\qquad
b_i \leftarrow b_i + u\, \mathrm{sign}(\bar{f} - f_i),
$$
:eqlabel:`eq_moe-bias`

where $\bar{f} = 1/E$ is the even share. The bias of an overloaded expert
falls, and that of an underloaded expert rises, which makes the
underloaded expert slightly easier to select. Since $b_i$ never enters the
weight $p_i$ in :eqref:`eq_moe-layer`, the controller adds no term to the
loss, and no gradient competes with the language-modeling one. The
controller steers where tokens go; the gradients serve prediction alone.

How large should $u$ be? The bias competes with probabilities near
$1/E$, here $1/8$. At $u = 0.01$, an expert that stays overloaded for 100
steps in a row has lost a bias of 1, more than any difference of
probabilities can make up, so a lasting imbalance cannot survive the
controller. The step $u$ sets how soon it gives way, and also how far the
biases overshoot once it has: with a step comparable to the differences
between probabilities, the load swings from expert to expert instead of
settling. Exercise :exref:`ex_moe-bias-controller` watches both happen.
DeepSeek-V3 used $u = 0.001$ for the first 14.3 trillion of its training
tokens :cite:`Liu.Feng.Xue.ea.2024`, and kept a balancing loss over each
sequence beside the controller, with a weight of only $10^{-4}$.

### Balancing Methods Compared
:label:`subsec_moe-balance-comparison`

Which of these treatments spreads the load, and which model predicts
better? Both remedies fit into an ordinary training loop. The loop below is
`d2l.train_steps` of :numref:`sec_test_models`, which draws minibatches,
takes optimizer steps, and returns the training losses, with two
additions: `aux_weight` adds $\lambda_\textrm{bal}$ times the sum of
$\mathcal{L}_{\mathrm{balance}}$ over the expert layers to the loss, and
`bias_rate` is the rate $u$ of the controller of :eqref:`eq_moe-bias`,
which the layer's `balance` method applies after each optimizer step.

```{.python .input #moe-three-runs-one-budget-1}
%%tab pytorch
@d2l.add_to_class(MoELayer)
@torch.no_grad()  # Not a gradient step
def balance(self, rate):
    """The controller: move each bias one step toward the even load, then
    count the next batch's assignments afresh."""
    self.expert_bias += rate * torch.sign(self.load.mean() - self.load)
    self.load.zero_()

def train_moe_lm(model, data, optimizer, num_steps, aux_weight=0.0,
                 bias_rate=0.0):
    """d2l.train_steps with a balancing loss and the bias controller."""
    device = d2l.try_gpu()
    model.to(device)
    layers, losses = moe_layers(model), []
    for l in layers:
        l.reset_stats()
    while len(losses) < num_steps:
        for X, Y in data.train_dataloader():
            ce = F.cross_entropy(model(X.to(device)).flatten(0, 1),
                                 Y.to(device).flatten())
            optimizer.zero_grad()
            (ce + aux_weight * sum(l.aux_loss for l in layers)).backward()
            optimizer.step()
            for l in layers:
                l.balance(bias_rate)
            losses.append(ce.detach())
            if len(losses) == num_steps:
                return torch.stack(losses).tolist()
```

```{.python .input #moe-three-runs-one-budget-1}
%%tab jax
@d2l.add_to_class(MoELayer)
def balance(self, rate):
    """The controller: move each bias one step toward the even load, then
    count the next batch's assignments afresh."""
    load = self.load[...]
    self.expert_bias[...] += rate * jnp.sign(load.mean() - load)
    self.load[...] = jnp.zeros_like(load)

@nnx.jit  # aux_weight and bias_rate are traced: one compilation for all
def moe_train_step(model, optimizer, X, Y, aux_weight, bias_rate):
    def loss_fn(model):
        ce = optax.softmax_cross_entropy_with_integer_labels(
            model(X), Y).mean()
        aux = sum(l.aux_loss[...] for l in moe_layers(model))
        return ce + aux_weight * aux, ce
    (_, ce), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    for l in moe_layers(model):  # The controller: not a gradient step
        l.balance(bias_rate)
    return ce

def train_moe_lm(model, data, optimizer, num_steps, aux_weight=0.0,
                 bias_rate=0.0):
    """d2l.train_steps with a balancing loss and the bias controller."""
    for l in moe_layers(model):
        l.reset_stats()
    losses = []
    while len(losses) < num_steps:
        for X, Y in data.train_dataloader():
            losses.append(moe_train_step(model, optimizer, jnp.asarray(X),
                                         jnp.asarray(Y), aux_weight,
                                         bias_rate))
            if len(losses) == num_steps:
                return [float(l) for l in jax.device_get(losses)]
```

To read a trained model, `evaluate_moe` computes its validation loss with
`d2l.lm_perplexity` of :numref:`sec_test_models`, whose logarithm is the
mean cross-entropy in nats per byte. During that pass it counts how many
validation tokens each expert receives and how many assignments the
buffers drop. `measure` adds a second pass with buffers large enough for
every assignment (capacity factor $E$, so $S = nk$). Comparing the two
losses tells apart two reasons why an unbalanced model predicts badly:
tokens that skip their expert, and experts that never learned much.

```{.python .input #moe-three-runs-one-budget-2}
%%tab pytorch, jax
def evaluate_moe(model, data):
    """Validation loss; each expert layer's shares of the validation
    tokens, busiest expert first; and the fraction of assignments dropped."""
    layers = moe_layers(model)
    for l in layers:
        l.reset_stats()
    loss = math.log(d2l.lm_perplexity(model, data.val_dataloader()))
    shares = [sorted((l.load / l.load.sum()).tolist(), reverse=True)
              for l in layers]
    dropped = sum(float(l.dropped) for l in layers) / sum(
        float(l.load.sum()) for l in layers)
    return loss, shares, dropped

def measure(model, data):
    """The validation loss, the same with buffers that drop nothing, the
    busiest expert's load in units of the even share, and the fraction
    dropped; and the shares."""
    layers = moe_layers(model)
    val, shares, dropped = evaluate_moe(model, data)
    peak = max(l.num_experts * s[0] for l, s in zip(layers, shares))
    capacity = [l.capacity_factor for l in layers]
    for l in layers:  # S = n k slots: room for every assignment
        l.capacity_factor = l.num_experts
    val_all = evaluate_moe(model, data)[0]
    for l, c in zip(layers, capacity):
        l.capacity_factor = c
    return (val, val_all, peak, dropped), shares
```

The model is the GPT of :numref:`sec_gpt` reduced to two blocks of width
128 and trained with the recipe of that section on the same bytes of
`d2l.WikiText103`. Each feed-forward network becomes eight experts, one
active per token, with buffers at $c = 1.25$. `small_gpt` builds this
model and, without a capacity factor, the dense model of the same size
that :numref:`subsec_moe-gpt` uses. `freeze_router` takes the routers'
weights out of training: they keep their initial values, while everything
else trains, the embeddings and attention that compute the routers' input
included. `balancing_run` builds a fresh model, trains it for 800 steps
with one balancing setting, and measures it:

```{.python .input #moe-balancing-methods-compared-1}
%%tab pytorch
data = d2l.WikiText103(batch_size=64, num_steps=128)

def small_gpt(vocab_size, capacity_factor=None):
    """Two blocks of width 128; with a capacity factor, each feed-forward
    network becomes eight experts with one active per token."""
    experts = lambda: MoELayer(128, 8, 1, capacity_factor=capacity_factor)
    model = d2l.GPT(vocab_size, num_hiddens=128, num_heads=4, num_blks=2,
                    ffn_factory=None if capacity_factor is None else experts)
    return model.to(d2l.try_gpu())

def freeze_router(model):
    """Keep every router at its initial weights: no gradient, no update."""
    for l in moe_layers(model):
        l.router.weight.requires_grad_(False)
```

```{.python .input #moe-balancing-methods-compared-1}
%%tab jax
data = d2l.WikiText103(batch_size=64, num_steps=128)

def small_gpt(vocab_size, capacity_factor=None):
    """Two blocks of width 128; with a capacity factor, each feed-forward
    network becomes eight experts with one active per token."""
    experts = lambda rngs: MoELayer(128, 8, 1,
                                    capacity_factor=capacity_factor, rngs=rngs)
    return d2l.GPT(vocab_size, num_hiddens=128, num_heads=4, num_blks=2,
                   ffn_factory=None if capacity_factor is None else experts,
                   rngs=nnx.Rngs(d2l.get_key()))

def freeze_router(model):
    """Keep every router at its initial weights: a Variable that is not a
    Param gets no gradient and no update."""
    for l in moe_layers(model):
        l.router.kernel = nnx.Variable(l.router.kernel[...])
```

```{.python .input #moe-balancing-methods-compared-2}
%%tab pytorch, jax
def balancing_run(data, num_steps=800, frozen=False, aux_weight=0.0,
                  bias_rate=0.0):
    """Measure a fresh small MoE GPT after num_steps steps of training with
    one balancing setting."""
    model = small_gpt(len(data.vocab), capacity_factor=1.25)
    if frozen:
        freeze_router(model)
    if num_steps:
        optimizer = d2l.LMOptimizer(model, lr=2e-3, num_steps=num_steps,
                                    warmup=100)
        train_moe_lm(model, data, optimizer, num_steps, aux_weight,
                     bias_rate)
    return measure(model, data)
```

Two questions from above meet here: is a fresh model already skewed, and
does a router that learns make it worse? The first row measures fresh
models without any training. The trained models differ only in how the
router is kept in check: not at all, frozen at its initial weights, by the
auxiliary loss at $\lambda_\textrm{bal} = 0.01$, or by the bias controller
at $u = 0.01$. Each row holds two runs from fresh random weights, and its
columns give the validation loss, the same loss with nothing dropped, the
busiest expert's load in units of the even share, and the fraction of
assignments dropped:

```{.python .input #moe-three-runs-one-budget-3}
%%tab pytorch, jax
arms = {'initialization': dict(num_steps=0), 'no balancing': {},
        'frozen router': dict(frozen=True),
        'auxiliary loss': dict(aux_weight=0.01),
        'bias controller': dict(bias_rate=0.01)}
print(f'spaces: {(data.val == ord(" ")).mean():.1%} of the validation bytes')
print(f'{"two runs each":>15}  {"loss":^9}  {"no drops":^9}  '
      f'{"busiest":^9}  {"dropped":^7}')
shares = {}  # The first run's shares
for name, kwargs in arms.items():
    (a, shares[name]), (b, _) = [balancing_run(data, **kwargs)
                                 for _ in range(2)]
    print(f'{name:>15}  {a[0]:.2f} {b[0]:.2f}  {a[1]:.2f} {b[1]:.2f}  '
          f'{a[2]:3.1f}x {b[2]:3.1f}x  {a[3]:3.0%} {b[3]:3.0%}')
```

The plot shows, for the first run of each row, the shares of the eight
experts, sorted from the busiest to the idlest and averaged over the two
layers, against the even share of $1/8$:

```{.python .input #moe-three-runs-one-budget-4}
%%tab pytorch, jax
ranks = list(range(1, 9))
mean_shares = [[sum(rank) / len(rank) for rank in zip(*s)]
               for s in shares.values()]  # Mean over the layers
d2l.plot(ranks, mean_shares + [[1 / 8] * 8], 'expert, busiest first',
         'share of tokens', legend=list(shares) + ['even share'],
         fmts=('C7x:', 'C0o-', 'C4v:', 'C1s--', 'C2^-.', 'k-'),
         legend_loc='outside')
```

The prediction of :numref:`subsec_moe-init-load` holds. Before any
training, the busiest expert of a layer receives about three to four times
its share (2.9 to 3.8 in the table), and the buffers drop a quarter to
three tenths of the assignments. The plot shows how lopsided that is: in
the fresh models, as in every unbalanced one, the three busiest experts
take about three quarters of the tokens and the idlest receive a small
fraction of their share. This is the end state that the feedback loop was
supposed to produce, and a fresh byte-level model starts there.

A router that cannot learn ends about as skewed after 800 steps: its
busiest expert receives 2.4 to 4.1 times its share, and 18% to 28% of the
assignments are dropped. So the skew does not need a learning router.
The frozen arm does not switch the feedback off entirely, though:
the router's input still trains and can drift toward whichever experts
help it. With the router trained and no balancing, the busiest expert ends at
2.8 to 3.9 times its share. Whether the feedback loop adds to the skew
that the inputs produce, two runs per row are too few to tell; exercise
:exref:`ex_moe-feedback` asks for more.

Both remedies spread the tokens much more evenly. In the plot every expert
receives close to its even share; in the table the busiest receives 1.2 to
1.4 times it, and at most 1% of the assignments are dropped.
Every balanced run ends with a lower validation loss than every unbalanced
one, frozen router or not.

Why do the unbalanced models lose? They end 0.05 to 0.25 nats above the
balanced ones. With room for every assignment the unbalanced models win
back 0.02 to 0.08 nats, so tokens that skip their expert at evaluation
explain part of the gap; even then, none of them comes down to the best
balanced runs. The rest is
capacity that training left unused, the idle tail of the plot: experts that
received few tokens learned little, and the assignments dropped during
training never trained the experts they were meant for.

At this scale the runs do not rank the two remedies: the two runs of one
remedy differ by about as much as the two remedies differ from each other.
Their difference shows in larger models. :citet:`Wang.Gao.Zeng.ea.2024`
trained models of 1 and 3 billion parameters both ways, and the controller
balanced the load much more evenly than the auxiliary loss while reaching
a slightly lower validation perplexity. Which should you use? Our table
cannot say, so the answer rests on that study and on the mechanism: the
controller adds no gradient that competes with prediction, and DeepSeek-V3
balances with it. We would start there. The auxiliary loss of GShard and
Switch is the older and more widely tested alternative, and at our scale
it serves just as well.

## Mixture-of-Experts Language Models

The accounting showed that a mixture stores more parameters at the same
cost per token, and balancing makes the experts usable. What no count can
tell us is whether the extra parameters make a better language model.

### A Mixture-of-Experts GPT
:label:`subsec_moe-gpt`

To find out, we train the small GPT of the balancing experiments with and
without experts. :numref:`sec_scaling-laws` showed that more parameters lower the loss only while there is text for
them to learn from, and a mixture needs that text all the more, since each
expert learns only from the tokens routed to it. We therefore compare
where text is plentiful: 6,000 steps on about 50 million bytes, more than
a hundred per parameter of the dense model. The dense `d2l.GPT` competes with the same
model with each feed-forward network replaced by eight experts, one active
per token. Each expert is a full-width `d2l.FeedForward`, so a token passes
through as many parameters in the mixture as in the dense model, apart from
the router's $128 \times 8$ weights per block.

Two settings follow the Switch transformer :cite:`fedus2022switch`. The
balancing is its auxiliary loss at $\lambda_\textrm{bal} = 0.01$, which
our runs could not tell apart from the controller. And the buffers get
more room than in the balancing experiments, a capacity factor of 2,
another setting with which Switch was trained. The reason is the spaces.
Early in training, the first block tends to send them all to one expert
(:numref:`subsec_moe-init-load`), and no balancing method can split them
while the router sees little but the byte: whatever moves one space to
another expert moves the others with it. At $c = 1.25$ a buffer holds
about 16% of the assignments, less than the spaces' 18.9%; at $c = 2$ it
holds a quarter.

```{.python .input #moe-a-mixture-of-experts-gpt-1}
%%tab pytorch, jax
capacity = {'dense': None, 'MoE': 2}  # The dense model has no buffers
models = {name: small_gpt(len(data.vocab), capacity_factor=c)
          for name, c in capacity.items()}
for name, model in models.items():
    print(f'{name:>5}: {d2l.num_params(model) / 1e6:5.2f}M parameters, '
          f'{active_params(model) / 1e6:.2f}M active per token')
```

The mixture stores 5.3 times the parameters of the dense model and uses
the same number per token. What should we expect from training? Early on,
the mixture should trail. Each fresh expert's output is scaled by its
probability, the largest of eight that all start near $1/8$, so with
$k = 1$ the whole feed-forward sublayer starts at a small fraction of the
strength of the dense one. This is the other side of the raw-probability
weight that lets a top-1 router learn. And each expert learns from only an
eighth of the tokens. Both models train
twice for 6,000 steps with the recipe of :numref:`sec_gpt`, and we measure
the validation loss every 250 steps. The cell prints each run's final
losses, the first measurement at which the mixture is ahead, and the
fraction of assignments dropped at the end, and it plots the curves of the
first run:

```{.python .input #moe-a-mixture-of-experts-gpt}
%%tab pytorch, jax
num_steps, every = 6000, 250
curves = {name: [] for name in capacity}  # Validation loss every 250 steps
for run in range(2):
    for name, c in capacity.items():  # Dense first, then MoE
        model = small_gpt(len(data.vocab), capacity_factor=c)
        optimizer = d2l.LMOptimizer(model, lr=2e-3, num_steps=num_steps,
                                    warmup=100)
        curve = []
        for _ in range(num_steps // every):  # Dense: no aux loss to add
            train_moe_lm(model, data, optimizer, every, aux_weight=0.01)
            curve.append(
                math.log(d2l.lm_perplexity(model, data.val_dataloader())))
        curves[name].append(curve)
    dense, moe = curves['dense'][-1], curves['MoE'][-1]
    ahead = next((every * (i + 1) for i, (m, d) in enumerate(zip(moe, dense))
                  if m < d), None)
    print(f'run {run + 1}: dense {dense[-1]:.3f}, MoE {moe[-1]:.3f}; '
          + (f'MoE ahead from step {ahead}, ' if ahead else 'MoE never ahead, ')
          + f'{evaluate_moe(model, data)[2]:.1%} dropped at the end')
d2l.plot(list(range(every, num_steps + 1, every)),
         [curves['dense'][0], curves['MoE'][0]], 'step',
         'validation loss (nats)', legend=['dense', 'MoE'], fmts=('-', '--'))
```

The mixture starts behind, as predicted. The dense curve then flattens
sooner, and the first measurement at which the mixture is ahead falls
between steps 1,000 and 2,000 in every run. The mixture ends with a lower
validation loss than the dense model in every run, by about five
hundredths of a nat (0.04 to 0.06), with nothing dropped.

Is that more than luck? The two runs of the dense model end within a
hundredth of each other, as the run-to-run spread measured in
:numref:`subsec_gpt-run-to-run` leads us to expect, and the two runs of the
mixture within a hundredth and a half. The gain is at least three times the
larger of these ranges. Two runs per model are few, but they leave no doubt
about the sign of the gain, and its size is known to about a hundredth.

The grid of :numref:`sec_scaling-laws` gives a rough yardstick. At 2,000
steps, a third of this budget, tripling a dense model from 0.59 to 1.77
million parameters lowered the loss by about 0.05 nats. The mixture gains
about as much while storing 2.26 million parameters, more than the
tripled model, and using the 0.43 million per token of our dense model.
Roughly, then, a stored expert parameter is worth less than a dense one,
while an operation spent on experts is worth more, and operations are what
every token costs. Much larger studies report the same effect: the Switch transformer reached the quality of a dense T5
model several times faster at the same number of operations per token, and
it already gained from two experts :cite:`fedus2022switch`. The gain needs
data, though. Exercise :exref:`ex_moe-data-regime` repeats the comparison
with the six-block GPT of :numref:`sec_gpt`, which at 1,000 steps reads
fewer than two bytes per parameter instead of more than a hundred.

The gain also has costs that the operation count leaves out. The mixture
keeps 5.3 times the parameters in memory, together with AdamW's two state
values per parameter. Its step takes longer than its operation count
suggests. At $c = 2$ the buffers hold twice as many rows as there are
tokens, half of them empty when the load is even. Copying tokens into the
buffers and outputs back moves data without computing, and the experts'
weights are stacked anew in every step. And wherever the load is uneven,
the buffers drop assignments, which a dense model never does.

### Fine-Grained and Shared Experts
:label:`subsec_moe-fine-grained`

Released mixtures differ in how they divide a budget of expert parameters.
Mixtral's experts are full copies of the feed-forward network, eight per
layer and two per token. DeepSeekMoE :cite:`Dai.Deng.Zhao.ea.2024` spends
such a budget differently, in the two steps shown in
:numref:`fig_moe-fine-grained`. The paper's example starts from 16
full-width experts with 2 active per token. The first step cuts each expert into $m$
narrower ones and lets each token select $m$ times as many: with $m = 4$, 8
of 64 experts of a quarter width instead of 2 of 16 full-width ones.
Neither the stored nor the active parameter count changes. The number of
combinations a token can select does, from $\binom{16}{2} = 120$ to
$\binom{64}{8} \approx 4.4 \times 10^{9}$.

In the ideal case, a token then assembles its network from many narrow
specialists, say one for chemical vocabulary and another for nested
clauses, instead of selecting two broad experts that must each cover a wide
range of inputs. Real routers are less tidy. The routing analysis of
Mixtral found experts that follow syntax and token identity rather than
topic :cite:`Jiang.Sablayrolles.Roux.ea.2024`, and the first block of our
own byte-level model starts out routing by little more than the current
byte.

![Three ways to spend the same expert parameters. Each box is an expert, its width proportional to its size, and the highlighted experts are the ones selected for one token. Top: 16 experts, of which a token uses 2. Middle: the same parameters as 64 experts of a quarter width, of which a token uses 8. Bottom: one of the narrow experts becomes a shared expert that every token uses, and each token selects 7 of the other 63. All three rows store and use the same number of parameters per token.](../img/mdl-transformers-moe-fine.svg)
:label:`fig_moe-fine-grained`

The second step makes one of the narrow experts *shared*: every token
passes through it, in addition to its routed experts. Some processing is
needed by every token, whatever its topic. With routed experts only, each
expert must learn that processing for itself, so the same function is
stored many times over. A shared expert stores it once and leaves the
routed experts free to differ. In experiments at 2 billion stored
parameters, a model with both refinements reached a lower loss on the Pile
than a GShard model with the same stored and active parameters (1.81
against 1.87). It about matched a GShard model with 1.5 times the expert
parameters and 1.5 times the computation :cite:`Dai.Deng.Zhao.ea.2024`.
Neither refinement changes the logic of our layer: narrower experts need
only a configurable width, and a shared expert is one more feed-forward
network whose output is added to the routed mixture.

What a trained model does with its billions of combinations is less clear
than the figure suggests. Exercise :exref:`ex_moe-granularity` counts how
many of them one of our own models actually uses.

## Summary

A mixture of experts replaces each block's feed-forward network by $E$
experts and a router that sends each token to $k$ of them, weighted by the
router's probabilities, so the stored parameters grow with $E$ and the
computation per token only with $k$. An efficient layer copies each token
into a fixed-size buffer of each of its experts and runs all experts at
once; the assignments that do not fit are dropped, by choice of this
implementation. The load is uneven before any training, because the router
of a byte-level model sees little more than the current byte and bytes are
far from equally frequent, and training can reinforce whichever experts
lead. An auxiliary loss on the product of load and probability, or a
controller that biases only the selection, evens out the load. Every
balanced model then ends below every unbalanced one, which loses both to
dropped assignments and to experts that learned little. On plentiful
text, a GPT with eight experts per block and one active per token
overtakes the dense model with the same parameters per token after 1,000 to
2,000 steps and ends about 0.05 nats lower in every run, while storing 5.3
times the parameters. Released models cut the experts finer and add a
shared one.

## Exercises

1. **Router gradient.** The top-$k$ selection in :eqref:`eq_moe-layer` has
   no gradient, yet the router trains. Consider one token with router logits
   $\mathbf{z} = \mathbf{W}_r \mathbf{x}$, so that
   $\mathbf{p} = \mathrm{softmax}(\mathbf{z})$, and treat the selected set
   $\mathcal{E}_k$ as a constant. Write the layer output as
   $\mathbf{y} = \sum_{i \in \mathcal{E}_k} p_i\, \mathrm{FFN}_i(\mathbf{x})$
   and let
   $s_i = (\partial \mathcal{L} / \partial \mathbf{y})^\top \mathrm{FFN}_i(\mathbf{x})$
   be the sensitivity of the loss to expert $i$'s output.
    1. Derive $\partial \mathcal{L} / \partial z_j$ for every expert $j$ in
       terms of $\mathbf{p}$ and the $s_i$. Which logits receive a nonzero
       gradient, and on which experts' outputs does the gradient of an
       unselected logit depend?
    1. Mixtral weights the selected experts by renormalized probabilities
       :cite:`Jiang.Sablayrolles.Roux.ea.2024`,

       $$\tilde{p}_i = \frac{p_i}{\sum_{j \in \mathcal{E}_k} p_j}, \qquad i \in \mathcal{E}_k.$$

       Show that for $k = 1$ the router receives no gradient from the
       language-modeling loss.
    1. Show that for $k = 2$ with renormalization the output depends on
       the logits only through the difference $z_a - z_b$ of the two
       selected experts $a$ and $b$. Which routing decision can the router
       learn from the loss in this case, and which can it not?
   :exlabel:`ex_moe-router-gradient`

1. **Serving arithmetic.** Qwen3-235B-A22B :cite:`Yang.Li.Yang.ea.2025`
   has $d = 4096$ and 94 layers. Every layer holds 128 SwiGLU experts of
   width 1536, of which each token activates 8, and no shared expert. Its
   attention has 64 query heads and 4 key--value heads, each of dimension
   128, and its untied embedding and output matrices cover a vocabulary of
   151,936 tokens.
    1. Use `moe_accounting` for the experts and the router, and add the
       attention and embedding parameters as
       :numref:`subsec_moe-conditional` does for Mixtral. Reproduce the
       235B stored and 22B active parameters of the model's name, ignoring
       normalization weights.
    1. Decoding a batch of $B$ sequences sends one token per sequence
       through every layer, and its speed is set by the bytes of weights
       read per step (:numref:`sec_kv-cache`). Assume that each token
       selects a uniformly random set of 8 experts, independently of the
       other tokens. Derive the expected number of distinct experts that a
       layer reads, and evaluate it for $B = 1$, 8, and 64. Above which
       batch size does a step read 90% of the expert weights, and which
       dense model then reads the same number of bytes per step?

1. **Balancing-loss gradient.** The gradient of :eqref:`eq_moe-aux` flows
   through $\bar{p}_i$ only, because $f_i$ is a count. Consider a batch of
   $n$ tokens, and write $\mathbf{z}(\mathbf{x})$ for the router logits of
   token $\mathbf{x}$.
    1. Derive
       $\partial \mathcal{L}_{\mathrm{balance}} / \partial z_j(\mathbf{x})$.
       For which experts $j$ does the gradient step lower the logit of
       token $\mathbf{x}$, and for which does it raise it?
    1. Show that the gradient vanishes for every token whenever the batch
       load is uniform, $f_i = 1/E$, whatever the individual distributions
       $\mathbf{p}(\mathbf{x})$ are. Which property of the router does the
       loss therefore leave unconstrained?
    1. Show that the gradient for a token tends to zero as its router
       distribution approaches a one-hot vector. What does this imply for
       the ability of the loss to repair a collapse in which the router
       has become confident?

1. [code] **Feedback or inputs.** The two runs per row of
   :numref:`subsec_moe-balance-comparison` cannot tell whether training
   the router adds to the skew that the inputs produce at initialization.
    1. Train the arms without balancing and with a frozen router five times
       each. Compare the busiest expert's load and the fraction dropped
       between the two arms, against the spread within each arm. Does the
       feedback loop of :numref:`subsec_moe-feedback` measurably
       concentrate the routing at this scale?
    1. A frozen router still sees an input that trains, and the gradient
       through $p_i(\mathbf{x})$ can move that input toward the experts
       that help it. Add a third arm that also cuts this path: freeze the
       router and feed it a detached copy of its input
       (`x.detach()` in PyTorch, `jax.lax.stop_gradient(x)` in JAX), so
       that no gradient from the routing reaches the rest of the model.
       Which half of the loop, the router's or the representation's,
       carries the effect you found?
    1. Rerun the arm without balancing and the two remedies with
       `num_active=2`. Does the unbalanced run still concentrate, and as
       strongly? Use the router gradient of
       problem :exref:`ex_moe-router-gradient` to explain how a second
       active expert changes the signal that the runner-up receives.
   :exlabel:`ex_moe-feedback`

1. [code] **Bias controller in isolation.** The controller of
   :eqref:`eq_moe-bias` can be studied without training. Build
   `MoELayer(256, num_experts=8, num_active=1)`, draw a fixed batch of
   shape $(4, 256, 256)$, that is, 1024 standard-normal tokens in four
   sequences, and add one fixed standard-normal vector
   $\boldsymbol{\mu} \in \mathbb{R}^{256}$ to every token, which skews the
   untrained router toward a few experts. Then call `reset_stats` and
   repeat 500 times: run the layer on the batch, and call its `balance`
   method, which updates the biases and resets `load`.
    1. For $u \in \{0.001, 0.01, 0.1\}$, plot the largest load
       $\max_i f_i$ against the iteration. How does the number of
       iterations needed to bring it below $1.5/E$ scale with $u$, and how
       does the load behave afterwards?
    1. After the last iteration, measure the fraction of tokens whose
       selected expert differs from the expert with the largest
       $p_i(\mathbf{x})$. What determines this fraction, and how does it
       depend on $u$?
    1. Replace the sign in :eqref:`eq_moe-bias` by the raw difference,
       $b_i \leftarrow b_i + u' (\bar{f} - f_i)$, with $u'$ chosen so that
       the largest first step equals that of $u = 0.01$. Which behavior
       from the first sub-problem changes?
   :exlabel:`ex_moe-bias-controller`

1. **Capacity factor.** ● Our layer gives each expert a buffer of
   $S = c\, nk/E$ slots for a batch of $n$ tokens, as GShard and the Switch
   transformer do :cite:`Lepikhin.Lee.Xu.ea.2021,fedus2022switch`, and
   drops the assignments that do not fit.
    1. Without compilation, a layer could size its buffers by each
       batch's routing. Now let the experts live on different devices, so
       that every device sends its tokens to the devices of their experts.
       What does a device need to know before it can receive the tokens of
       the others, and what does a buffer size fixed in advance save? What
       must an implementation that drops nothing do instead?
    1. Take $k = 1$, $n = 4096$, and a router that sends each token to a
       uniformly random expert, independently of the other tokens. The
       load of one expert is then binomial with mean $n/E$. Use a normal
       approximation, together with $\mathbb{E}[\max(Z, 0)] = 1/\sqrt{2\pi}$
       for a standard normal $Z$, to estimate the fraction of tokens
       dropped at $c = 1$ for $E = 8$ and for $E = 64$. How does the
       fraction scale with $E$ and $n$, and what does $c = 1.25$ change?
    1. Our layer fills the buffers in the order of the tokens. In a
       causal language model, can whether a token is dropped depend on
       later tokens of its own sequence? On other sequences of the
       minibatch? What does this imply for a model that is evaluated
       with batches of a different size or composition than in training,
       for example while decoding one token per sequence?

1. [code] **When the mixture loses.** :numref:`subsec_moe-gpt` compares
   the two models where the dense model has more text than it can absorb.
    1. Repeat the comparison with the six-block GPT of width 256 from
       :numref:`sec_gpt` and its budget of 1,000 steps, with eight
       full-width experts per block. Train each model three times. Which
       one ends lower?
    1. Compute the training bytes per dense parameter in both comparisons,
       and explain the outcome with the scaling laws of
       :numref:`sec_scaling-laws` and the data each expert sees.
    1. Train the six-block pair for longer. Does the mixture overtake the
       dense model, and if so, after how many steps?
   :exlabel:`ex_moe-data-regime`

1. [code] **Shared expert.** ● Write `SharedMoELayer`, a subclass of
   `MoELayer` that adds the output of one extra `d2l.FeedForward`, applied
   to every token with weight one, to the routed
   mixture (:numref:`subsec_moe-fine-grained`). Compare two models in the
   setup of :numref:`subsec_moe-balance-comparison`, both balanced by the
   bias controller: `MoELayer` with $E = 8$ and $k = 2$, and
   `SharedMoELayer` with 7 routed experts and $k = 1$. Both store eight
   experts per layer and activate two per token.
    1. Train each model three times and compare their validation losses.
       Is the difference larger than the spread between the runs of one
       model?
    1. :citet:`Dai.Deng.Zhao.ea.2024` test whether a shared expert holds
       processing that the routed experts lack: they disable it and
       activate one more routed expert, which keeps the per-token
       computation fixed. Repeat this test on a trained `SharedMoELayer`
       model. Remove the shared path, set `num_active` to 2 in every layer,
       and compare the validation loss with that of the intact model. What
       does the result say about the division of labor? Why does removing
       the shared path without the replacement say little about it?

1. [extended] **Expert granularity.** ● :numref:`subsec_moe-fine-grained`
   cites large-scale evidence that narrower, more numerous experts help at
   a fixed budget, and this section does not test it. In the setup of
   :numref:`subsec_moe-balance-comparison`, an expert at $d = 128$ has
   SwiGLU width 341. Give `MoELayer` experts of configurable width, and
   compare $(E, k) \in \{(8, 1), (16, 2), (32, 4), (64, 8)\}$ with expert
   widths 341, 171, 85, and 43, all balanced by the bias controller. These
   configurations store and activate the same expert parameters as
   $(8, 1)$ to within 1%, while the number of expert combinations
   available to a token grows from 8 to about $4.4 \times 10^9$.
    1. Train each configuration three times and compare the validation
       losses. Does finer granularity help at this scale by more than the
       spread between runs? Relate the outcome to the ablations of
       :citet:`Dai.Deng.Zhao.ea.2024`.
    1. The argument of :numref:`subsec_moe-fine-grained` assumes that
       tokens use many different combinations. In one layer of each trained
       model, record the selected expert set of every token in a fixed set
       of validation batches, and count the distinct sets that occur.
       Compare the count with the number of tokens and with the number of
       available combinations. How much of the added flexibility does the
       model use?
   :exlabel:`ex_moe-granularity`

[Discussions](https://d2l.discourse.group/t/36779)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §12.6]{.kicker}

Mixture of experts<br>
**stored and active parameters · top-k routing · dispatch · load balancing · an MoE GPT · fine-grained and shared experts**
:::
:::

::: {.slide title="A Dense Model Runs Every Parameter for Every Token"}
[The problem]{.kicker}

- GPT-2 small: 124M parameters, about $2 \times 124\textrm{M}$ operations
  per token. Twice the parameters, twice the cost.
- The feed-forward networks hold two thirds of each block: $8d^2$ of
  $12d^2$.
- Yet "the integral of" and "Act I, Scene" need different knowledge.
- DeepSeek-V3 stores 671B parameters and uses **37B** per token.

::: {.d2l-note .warn}
Can a model store many parameters but run only the few each token needs?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- count the stored and active parameters of an expert layer;
- route tokens to fixed buffers and say which assignments are dropped;
- predict an uneven load before training, and balance it with a loss or a
  bias controller;
- read an MoE GPT against a dense GPT with the same parameters per token;
- explain fine-grained and shared experts.

**Running example:** GPT-2 small with eight experts per block, then our
small byte-level GPT.
:::

::: {.slide title="More Parameters per Operation"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| The Transformer Block | attention + FFN on a residual stream |
| A GPT from Scratch | a trained language model; GPT-2 loaded |
| The KV Cache | reuse keys and values while sampling |
| Shrinking the KV Cache | fewer heads, fewer dimensions, fewer tokens |
| Encoders, Decoders, and Cross-Attention | the other two wirings |
| Vision Transformer | images as token sequences |
| Scaling Laws | loss against parameters, data, compute |
| **Mixture of Experts** | **more parameters per FLOP** |
:::

::: {.col .narrow}
- **Uses:** `d2l.FeedForward`, `d2l.GPT`, its data and training recipe;
  tokens per parameter from Scaling Laws.
- **Skips:** experts spread over many devices.
:::
:::
:::

::: {.slide title="A Router Sends Each Token to k of E Experts"}
::: {.cols .vc}
::: {.col .fig .big}
@fig:mdl-transformers-moe
:::

::: {.col}
$$\mathrm{MoE}(\mathbf{x}) = \sum_{i \in \mathcal{E}_k(\mathbf{x})} p_i\, \mathrm{FFN}_i(\mathbf{x})$$

- $\mathbf{p} = \mathrm{softmax}(\mathbf{W}_r \mathbf{x})$;
  $\mathcal{E}_k$: the $k$ largest $p_i$.
- Stored: $E$ experts. Run: $k$.
:::
:::
:::

::: {.slide title="GPT-2 Small With Eight Experts Stores 521M and Uses 124M"}
[Running example]{.kicker}

| | dense | 8 experts, top-1 |
|---|--:|--:|
| FFN per block | 4.72M | 8 × 4.72M |
| FFN, 12 blocks | 56.7M | 453M |
| everything else | 67.8M | 67.8M |
| **stored** | **124M** | **521M** |
| **active per token** | **124M** | **124M** |

Same operations per token; 4.2 times the parameters in memory.
:::

::: {.slide title="Mixtral Uses 12.9B of Its 46.7B Parameters per Token"}
@!moe-conditional-computation

- "8x7B" is not 56B: the experts share attention and embeddings. A token
  uses a little over a quarter of the 46.7B.
- DeepSeek-V3: 656B of its 671B parameters sit in experts, 28 times
  what a token uses there.
- ours: the two-block byte GPT trained later in this deck.
- The count assumes an even spread of tokens over the experts.
:::

::: {.slide title="The Probability Weight Carries the Router's Gradient"}
- Selecting the top $k$ is discrete: no gradient.
- The weight $p_i$ of a selected expert is differentiable.
- If more of expert $i$ would lower the loss, the gradient raises $p_i$,
  and the router selects $i$ more often for similar tokens.
- Switch Transformer (Fedus et al., 2022): weight $= p_i$; in the figure,
  $0.24 + 0.31 = 0.55$. Mixtral renormalizes over the selected $k$; with
  $k = 1$ such a router learns nothing from the loss.
:::

::: {.slide title="Check: What Fraction of a Layer Does One Token Use?"}
[Check]{.kicker}

A layer with $E = 64$ experts of width $w$, $k = 6$ routed, plus 2 shared
experts of the same width. What fraction of the expert parameters does one
token use?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $(6 + 2)/(64 + 2) = 8/66 \approx 12\%$. The router adds
$d \times 64$ weights, which every token uses.
:::
:::
:::

::: {.slide title="Each Expert Gets a Fixed Buffer of S Slots"}
- $n$ tokens make $nk$ assignments (token, expert); each expert's count
  varies by batch.
- Fixed shapes: $S = \lceil c\, nk/E \rceil$ slots per expert, capacity
  factor $c = 1.25$ in Switch.
- All experts run at once on $(E, S, d)$: about $c\,nk$ rows, a fraction
  $ck/E$ of running every expert on every token.

::: {.d2l-note}
A balanced router fills the buffers; an unbalanced one overflows them.
:::
:::

::: {.slide title="Six Assignments, Three Experts, Three Slots: The Last One Is Dropped"}
@!moe-dispatching-tokens-to-experts-2

- First come, first served: place $=$ cumulative count $- 1$; slot
  $= eS + \text{place}$; place $\geq S$ goes to slot $ES = 9$.
- Late tokens of a minibatch are dropped first.
- Dropping is a choice: sorting tokens into variable-length segments drops
  nothing (DeepSeek-V3 drops none).
:::

::: {.slide title="Dispatch: Number, Copy Into Buffers, Run, Gather Back" only="pytorch"}
```{.python #moe-exercises-1}
e = idx.flatten()                              # expert of each pair
onehot = F.one_hot(e, E)                       # (n k, E)
rank = (onehot.cumsum(0) * onehot).sum(1) - 1  # place in its queue
slot = torch.where(rank < S, e * S + rank, E * S)  # E S: dropped
buf = x.new_zeros(E * S + 1, d)                # + an overflow row
buf[slot] = x.repeat_interleave(k, 0)
Y = self.run_experts(buf[:-1].reshape(E, S, d)).reshape(E * S, d)
Y = torch.cat([Y, Y.new_zeros(1, d)])[slot]    # dropped: zeros
weights = P.gather(-1, idx)                    # (n, k): the p_i
out = (weights[..., None] * Y.reshape(n, k, d)).sum(1)
```

`x`: $(n, d)$ tokens; `P`: $(n, E)$ router probabilities; `idx`:
$(n, k)$ selected experts. `run_experts`: SwiGLU as three batched products
on the stacked weights.
:::

::: {.slide title="Dispatch: Number, Copy Into Buffers, Run, Gather Back" only="jax"}
```{.python #moe-exercises-2}
e = idx.ravel()                                # expert of each pair
onehot = jax.nn.one_hot(e, E, dtype=jnp.int32)  # (n k, E)
rank = (onehot.cumsum(0) * onehot).sum(1) - 1  # place in its queue
slot = jnp.where(rank < S, e * S + rank, E * S)  # E S: dropped
buf = jnp.zeros((E * S, d), x.dtype).at[slot].set(
    jnp.repeat(x, k, 0), mode='drop')
Y = self.run_experts(buf.reshape(E, S, d)).reshape(E * S, d)
Y = Y.at[slot].get(mode='fill', fill_value=0)  # dropped: zeros
weights = jnp.take_along_axis(P, idx, -1)      # (n, k): the p_i
out = (weights[..., None] * Y.reshape(n, k, d)).sum(1)
```

`x`: $(n, d)$ tokens; `P`: $(n, E)$ router probabilities; `idx`:
$(n, k)$ selected experts. Every shape follows from $n$: one compilation,
whatever the routing.
:::

::: {.slide title="Prediction: Bytes Are Unbalanced Before Any Training"}
- Random vectors spread evenly; text does not. A part $\mathbf{m}$ shared
  by every position adds the same scores $\mathbf{W}_r \mathbf{m}$ to
  every token's logits.
- Block 1 sees the current byte plus a blend of earlier bytes that is
  much the same everywhere: equal bytes go to one expert.
- Nearly one byte in five is a space (18.9%); one buffer holds
  $1.25/8 \approx 16\%$ of the assignments.

::: {.d2l-note .warn}
A fresh model should overflow its busiest buffers before any gradient.
:::
:::

::: {.slide title="Training Can Make It Worse: Routing Feeds on Itself"}
- An expert slightly better on its tokens is rewarded twice: its weights
  improve, **and** the router raises its probability.
- More tokens, more gradient, a better expert, more tokens.
- End point, *routing collapse*: a few experts take most tokens.
- Test: freeze the router, so that it cannot learn.
:::

::: {.slide title="Remedy 1: Penalize Load Times Probability"}
$$\mathcal{L}_{\mathrm{balance}} = E \sum_{i=1}^{E} f_i\, \bar{p}_i$$

- $f_i$: fraction of the minibatch's assignments that expert $i$
  received (a count, no gradient); $\bar{p}_i$: mean probability of $i$.
- Even routing: $\mathcal{L}_{\mathrm{balance}} = 1$. One expert takes
  everything with $p \approx 1$: about $E$.
- Added with weight $\lambda_\textrm{bal} = 0.01$ (Switch). Its gradient
  lowers the busiest experts' probability for every token, whatever the
  token needs.
:::

::: {.slide title="Remedy 2: Bias the Selection, Not the Weights"}
$$\mathcal{E}_k = \operatorname{argtop}_k(p_i + b_i), \qquad b_i \leftarrow b_i + u\,\mathrm{sign}(1/E - f_i)$$

- After each step, overloaded experts lose bias, underloaded ones gain.
- $b_i$ never enters the weight $p_i$: no competing gradient.
- Scale: $u = 0.01$ against $p_i \approx 1/8$; 100 steps of overload cost
  a bias of 1, so a lasting imbalance always loses. Too large a $u$, and
  the load swings from expert to expert.
- DeepSeek-V3: $u = 0.001$, plus a per-sequence loss at $10^{-4}$.
:::

::: {.slide title="Check: What Is the Balancing Loss Here?"}
[Check]{.kicker}

$E = 4$. A minibatch sends half of its tokens to expert 1 and a sixth to
each of the others; the mean probabilities follow the same split. What is
$\mathcal{L}_{\mathrm{balance}}$?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $4\,(\tfrac12 \cdot \tfrac12 + 3 \cdot \tfrac16 \cdot \tfrac16)
= 4 \cdot \tfrac13 \approx 1.33$, against 1 for even routing.
:::
:::
:::

::: {.slide title="A Fresh Model Is Already Skewed, and a Frozen Router Stays So"}
Two blocks of width 128, 8 experts, top-1, $c = 1.25$, 800 steps on
WikiText-103 bytes. "no drops": the same model with room for every
assignment.

@!moe-three-runs-one-budget-3

::: {.d2l-note}
Fresh: busiest expert 2.9–3.8× its share. Frozen router: 2.4–4.1×.
:::
:::

::: {.slide title="Balanced Runs Spread the Load and End Lower"}
@!moe-three-runs-one-budget-4

- Unbalanced: the three busiest experts take about three quarters of the
  tokens, the idlest little. Balanced: busiest 1.2–1.4×, at most 1% dropped.
- Without dropping, the unbalanced models gain 0.02–0.08 nats and still end
  above the best balanced runs: capacity left untrained costs the rest.
- Not ranked at this scale. Start with the controller: no competing
  gradient, better at 1B+ (Wang et al., 2024), used by DeepSeek-V3.
:::

::: {.slide title="Same Parameters per Token, 5.3 Times the Stored Ones"}
The small GPT of the balancing runs, dense and with each FFN replaced by 8
experts, top-1, auxiliary loss $\lambda_\textrm{bal} = 0.01$:

@!moe-a-mixture-of-experts-gpt-1

- 6,000 steps, about 50M bytes: over 100 per dense parameter.
- Capacity factor 2: a buffer holds a quarter of the assignments, room
  for the spaces that block 1 sends to one expert early on. The experts
  compute $2n$ rows for $n$ tokens.
:::

::: {.slide title="The Mixture Overtakes the Dense Model by Step 2,000"}
@!moe-a-mixture-of-experts-gpt

- Behind until step 1,000 to 2,000, then ahead: about 0.05 nats lower
  (0.04–0.06) at the end in every run, nothing dropped.
:::

::: {.slide title="An Expert Parameter Is Worth Less Stored, More per Operation"}
- Slow start: each fresh expert's output is scaled by $p_i$, the largest of
  eight probabilities near $1/8$, and each expert sees an eighth of the
  tokens.
- Runs of each model agree within 0.015; the gain, 0.04–0.06, is about
  three times that.
- Yardstick (scaling grid, 2,000 steps): tripling a dense model, 0.59M to
  1.77M, gained about 0.05 nats.
- The mixture gains about as much while storing 2.26M and using 0.43M
  per token.

::: {.d2l-note}
Per stored parameter, experts buy less than a dense model; per operation,
they buy more, and operations are what each token costs.
:::
:::

::: {.slide title="Narrow and Shared Experts: Same Budget, Far More Combinations"}
::: {.cols .vc}
::: {.col .fig .big}
@fig:mdl-transformers-moe-fine
:::

::: {.col}
- Cut 16 experts into 64 of a quarter width: $\binom{16}{2} = 120$
  combinations become $\binom{64}{8} \approx 4.4 \times 10^9$.
- A shared expert stores common processing once.
- DeepSeekMoE, 2B stored: loss 1.81 against 1.87 for GShard at equal
  size.
- Real routers are less tidy: Mixtral's follow syntax and token identity
  more than topic.
:::
:::
:::

::: {.slide title="Check: How Much Text Does One Expert See?"}
[Check]{.kicker}

Our MoE GPT trains for 6,000 steps of 64 windows of 128 bytes, with
8 experts per layer and top-1 routing, evenly balanced. How many
bytes does one expert of a layer see, and how many does the dense
FFN see?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $6000 \cdot 64 \cdot 128 \approx 49$M bytes pass through
each layer. The dense FFN sees all of them; each expert sees about
$49\textrm{M}/8 \approx 6$M. Every expert must learn from an eighth of the
data, which is why mixtures need plenty of distinct text.
:::
:::
:::

::: {.slide title="Wrap-up"}
- MoE: $E$ FFNs and a router; parameters grow with $E$, operations per
  token with $k$. Mixtral 46.7B/12.9B; GPT-2 small with 8 experts
  521M/124M.
- Fixed buffers of $\lceil c\,nk/E \rceil$ slots; first come, first
  served; overflow is dropped.
- Byte routing is skewed before training (about 3–4×); a loss or a
  selection bias evens it (1.2–1.4×), and every balanced run ends lower.
- Our MoE GPT overtakes the dense GPT by step 2,000 and ends about 0.05
  nats lower in every run, storing 5.3 times the parameters.

**Next:** the modern recipe: released models as settings of this
chapter's block.
:::
