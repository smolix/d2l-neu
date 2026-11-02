# Mixture of Experts
:label:`sec_moe`

In a dense transformer every token passes through every parameter. The
forward pass costs about two floating-point operations per parameter and
token (:numref:`sec_gpt`), so a model with twice the parameters costs twice
as much to train and to run. Yet most of what a large model knows is of no
use for any one token: continuing "the integral of" draws on different
knowledge from continuing "Act I, Scene". A *mixture of experts* (MoE)
exploits this. It keeps $E$ copies of each block's feed-forward network,
the *experts*, and a small learned *router* sends each token to only $k$
of them. The stored parameters then grow with $E$, while the computation
per token grows only with $k$. DeepSeek-V3, for example, stores 671 billion
parameters and uses 37 billion of them for each token
:cite:`Liu.Feng.Xue.ea.2024`.

The idea is old. :citet:`Jacobs.Jordan.Nowlan.ea.1991` trained a committee
of specialist networks under a gating network, and
:citet:`Shazeer.Mirhoseini.Maziarz.ea.2017` made the gate sparse, so that
only a few of thousands of experts run for each input. This section counts
stored and active parameters, implements the layer, and studies how it
fails: a router left to itself sends most tokens to a few experts. We
compare two remedies, an auxiliary loss and a controller that adds no
gradient. Finally, we give the GPT of :numref:`sec_gpt` eight experts per
block and train it against the dense model at equal computation per token.

```{.python .input #moe-mixture-of-experts}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input #moe-mixture-of-experts}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import nnx
import jax
from jax import numpy as jnp
import math
import optax
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
We apply it to two deployed models and to our own configuration, the one
trained at the end of this section. Mixtral 8x7B has 8 full-width experts
per layer, of which each token uses 2 :cite:`Jiang.Sablayrolles.Roux.ea.2024`.
DeepSeek-V3 has 256 narrow experts per layer, of which each token uses 8,
plus one *shared* expert that every token uses :cite:`Liu.Feng.Xue.ea.2024`.

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
    print(f'{name:>12}: {store/unit:6.1f}{tag} in experts, '
          f'{active/unit:5.1f}{tag} active per token, '
          f'ratio {store/active:4.1f}x')
    return store, active

moe_accounting('ours', 256, 683, 8, 1, 6, unit=1e6)
moe_accounting('DeepSeek-V3', 7168, 2048, 256, 8, 58, shared=1)
store, active = moe_accounting('Mixtral 8x7B', 4096, 14336, 8, 2, 32)
attn = 32 * (2 * 4096 * 4096 + 2 * 4096 * 1024)  # GQA: 32 Q, 8 KV heads
emb = 2 * 32000 * 4096                           # embedding + output head
print(f'{"":>12}  with attention and embeddings: '
      f'{(store + attn + emb)/1e9:.1f}B stored, '
      f'{(active + attn + emb)/1e9:.1f}B active')
```

DeepSeek-V3's 58 expert layers store 656B of its 671B parameters, and a
token uses 23B of them: the layers store 28 times what a token uses. (The
37B on its model card add
attention, the embeddings, and three dense layers, which our count of the
expert layers leaves out.) For Mixtral, adding the attention layers, which
share each key and value head among four query heads (:numref:`sec_kv-cache`),
and the separate embedding and output matrices gives 46.7B stored and 12.9B
active parameters. The name "8x7B" thus describes what the model stores,
not the work per token: a token uses less than two sevenths of the
parameters. Our configuration has Mixtral's eight experts per layer but
sends each token to one of them, so it stores eight times what it uses.

The count rests on two assumptions. First, it counts only computation.
All $E$ experts must sit in accelerator memory, and during training each
of them receives gradients from the tokens routed to it. Large models
therefore spread their experts over many devices and send every token to
its experts and back, a systems problem that the Language Models part
takes up. Second, the count assumes that the router spreads tokens evenly
over the experts. If it sends most tokens to a few of them, the other
experts hold parameters that hardly ever run, and the model behaves like a
smaller one.

## A Mixture-of-Experts Layer

The layer replaces the feed-forward network of a block, and nothing else:
attention, normalization, and the residual stream stay as they are. We
first define what it computes and then how to compute it efficiently.

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
others, this is called *token-choice* routing.
:numref:`fig_moe-layer` follows one token through a layer with $E = 8$ and
$k = 2$.

![A mixture-of-experts layer with eight experts, two of which are active per token. The router scores every expert. The two most probable experts run, and their outputs are weighted by their probabilities and added. The other six experts are stored but do no work for this token.](../img/mdl-transformers-moe.svg)
:label:`fig_moe-layer`

The weight of a selected expert is its probability $p_i$ itself, not a
probability renormalized over the $k$ selected experts. This choice
follows the Switch transformer :cite:`fedus2022switch`. Selecting the top
$k$ is a discrete decision and has no gradient. The weight of a selected
expert has one: if a larger contribution of expert $i$ would lower the
loss, the gradient raises $p_i$, and with it the chance that the router
selects expert $i$ for similar tokens. Through the softmax, every logit
receives a gradient, including those of the experts that were not
selected. Mixtral instead renormalizes the weights of the selected experts
to sum to one :cite:`Jiang.Sablayrolles.Roux.ea.2024`. The router then
learns only how to divide the weight among the selected experts, and for
top-1 routing, where the single weight is always one, it learns nothing
from the loss at all. With raw probabilities, a top-1 router still
learns. The first exercise works out these gradients.

### Dispatching Tokens to Experts

Evaluating :eqref:`eq_moe-layer` literally, by running all $E$ experts on
every token and multiplying the outputs of the unselected ones by zero,
gives the right answer at $E$ times the cost and saves nothing. An
efficient layer brings each token to its experts instead. A minibatch of
$n$ tokens makes $nk$ assignments, pairs of a token and one of its
experts, and how many of them each expert receives changes from batch to
batch. Matrix
multiplications need fixed shapes, so each expert gets a buffer of
$C = \lceil c\, nk/E \rceil$ slots, where the *capacity factor* $c$ leaves
room for an uneven spread; the Switch transformer trains with $c = 1.25$
:cite:`fedus2022switch`.

We number the assignments of each expert in the
order of the tokens and copy the token of an expert's $r^\textrm{th}$
assignment into slot $r$ of that expert's buffer. An assignment with
$r \geq C$ is *dropped*: that expert skips the token, which keeps the
outputs of its other experts and, through the residual connection, its
input. All $E$ experts then run at once, as one batched matrix
multiplication over buffers of shape $(E, C, d)$, and each token collects
its $k$ outputs, weighted by their probabilities. The work is that of
$EC \approx cnk$ rows, a fraction $ck/E$ of running every expert on every
token. Fixed shapes come with dropping: a balanced router loses few
assignments, an unbalanced one many.

Besides its output, the layer keeps four things for the experiments
below: `expert_bias`, a vector added to the probabilities for the top-$k$
selection only, which stays zero until we need it; `load`, the number of
assignments each expert has received, dropped or not; `dropped`, the
number of dropped assignments; and `aux_loss`, a measure of imbalance that
we derive when we use it.

```{.python .input #moe-a-mixture-of-experts-layer-1}
%%tab pytorch
class MoELayer(nn.Module):
    """Token-choice top-k mixture of SwiGLU d2l.FeedForward experts, each
    with a buffer of C = capacity_factor * n k / E slots for n tokens."""
    def __init__(self, num_hiddens, num_experts, num_active,
                 capacity_factor=1.25):
        super().__init__()
        self.num_experts, self.num_active = num_experts, num_active
        self.capacity_factor = capacity_factor
        self.router = nn.Linear(num_hiddens, num_experts, bias=False)
        self.experts = nn.ModuleList([d2l.FeedForward(num_hiddens)
                                      for _ in range(num_experts)])
        self.register_buffer('expert_bias', torch.zeros(num_experts))
        self.register_buffer('load', torch.zeros(num_experts))
        self.register_buffer('dropped', torch.zeros(()))

    def forward(self, X):
        x = X.reshape(-1, X.shape[-1])                 # (n, d): the tokens
        (n, d), E, k = x.shape, self.num_experts, self.num_active
        C = math.ceil(self.capacity_factor * n * k / E)  # slots per expert
        P = F.softmax(self.router(x), -1)              # (n, E)
        idx = (P + self.expert_bias).topk(k).indices   # (n, k)
        e = idx.flatten()                              # expert of each pair
        onehot = F.one_hot(e, E)                       # (n k, E)
        rank = (onehot.cumsum(0) * onehot).sum(1) - 1  # place in its queue
        slot = torch.where(rank < C, e * C + rank, E * C)  # E C: dropped
        buf = x.new_zeros(E * C + 1, d)                # + an overflow row
        buf[slot] = x.repeat_interleave(k, 0)
        Y = self.run_experts(buf[:-1].reshape(E, C, d)).reshape(E * C, d)
        Y = torch.cat([Y, Y.new_zeros(1, d)])[slot]    # dropped: zeros
        weights = P.gather(-1, idx)                    # (n, k): the p_i
        out = (weights[..., None] * Y.reshape(n, k, d)).sum(1)
        counts = onehot.sum(0)
        self.load += counts
        self.dropped += (rank >= C).sum()
        f = counts / counts.sum()                      # this batch's shares
        self.aux_loss = E * (f * P.mean(0)).sum()
        return out.reshape(X.shape)

    def run_experts(self, H):
        """Each expert's SwiGLU on its own buffer: (E, C, d) -> (E, C, d)."""
        W = lambda name: torch.stack(  # (E, d_in, d_out): one per expert
            [getattr(e, name).weight for e in self.experts]).mT
        return (F.silu(H @ W('W_g')) * (H @ W('W_1'))) @ W('W_2')
```

```{.python .input #moe-a-mixture-of-experts-layer-1}
%%tab jax
class MoELayer(nnx.Module):
    """Token-choice top-k mixture of SwiGLU d2l.FeedForward experts, each
    with a buffer of C = capacity_factor * n k / E slots for n tokens."""
    def __init__(self, num_hiddens, num_experts, num_active,
                 capacity_factor=1.25, *, rngs):
        self.num_experts, self.num_active = num_experts, num_active
        self.capacity_factor = capacity_factor
        self.router = d2l.Linear(num_hiddens, num_experts, use_bias=False,
                                 rngs=rngs)
        self.experts = nnx.List([d2l.FeedForward(num_hiddens, rngs=rngs)
                                 for _ in range(num_experts)])
        self.expert_bias = nnx.Variable(jnp.zeros(num_experts))
        self.load = nnx.Variable(jnp.zeros(num_experts))
        self.dropped = nnx.Variable(jnp.zeros(()))
        self.aux_loss = nnx.Variable(jnp.zeros(()))

    def __call__(self, X):
        x = X.reshape(-1, X.shape[-1])                 # (n, d): the tokens
        (n, d), E, k = x.shape, self.num_experts, self.num_active
        C = math.ceil(self.capacity_factor * n * k / E)  # slots per expert
        P = jax.nn.softmax(self.router(x), -1)         # (n, E)
        _, idx = jax.lax.top_k(P + self.expert_bias[...], k)  # (n, k)
        e = idx.ravel()                                # expert of each pair
        onehot = jax.nn.one_hot(e, E, dtype=jnp.int32)  # (n k, E)
        rank = (onehot.cumsum(0) * onehot).sum(1) - 1  # place in its queue
        slot = jnp.where(rank < C, e * C + rank, E * C)  # E C: dropped
        buf = jnp.zeros((E * C, d), x.dtype).at[slot].set(
            jnp.repeat(x, k, 0), mode='drop')
        Y = self.run_experts(buf.reshape(E, C, d)).reshape(E * C, d)
        Y = Y.at[slot].get(mode='fill', fill_value=0)  # dropped: zeros
        weights = jnp.take_along_axis(P, idx, -1)      # (n, k): the p_i
        out = (weights[..., None] * Y.reshape(n, k, d)).sum(1)
        counts = onehot.sum(0)
        self.load[...] += counts
        self.dropped[...] += (rank >= C).sum()
        f = counts / counts.sum()                      # this batch's shares
        self.aux_loss[...] = E * (f * P.mean(0)).sum()
        return out.reshape(X.shape)

    def run_experts(self, H):
        """Each expert's SwiGLU on its own buffer: (E, C, d) -> (E, C, d)."""
        W = lambda name: jnp.stack(  # (E, d_in, d_out): one per expert
            [getattr(e, name).kernel[...] for e in self.experts])
        return (jax.nn.silu(H @ W('W_g')) * (H @ W('W_1'))) @ W('W_2')
```

The cumulative sum over the one-hot encoding numbers the assignments of
each expert: row $j$ counts, for every expert, the assignments to it among
the first $j$, so its entry at the expert of assignment $j$ is that
assignment's place in its expert's queue, counted from one. Slot $EC$ lies
past the end of the buffers and takes every dropped assignment.
`run_experts` stacks one of the three matrices of every `d2l.FeedForward`
into an array of shape
$(E, d, w)$ or $(E, w, d)$, for experts of width $w$, and applies the
SwiGLU of :numref:`sec_transformer-block` as three batched products, one
matrix per buffer. It reads the experts' `W_g`, `W_1`, and `W_2` by name,
so it assumes their default SwiGLU without biases.

:begin_tab:`pytorch`
The buffer has one row more than the $EC$ slots. Every dropped assignment
is written into that last row, which the experts never see, and reading the
outputs back through `slot` returns the appended row of zeros for it.
No group size leaves the device, so the layer never waits for the host.
:end_tab:

:begin_tab:`jax`
Under `jit`, every array needs a shape that is known when the function is
compiled. Here every shape follows from the shape of `X`, since $C$
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
argument, which builds the feed-forward module of the block. We check
shapes on 400 random tokens, count parameters for eight experts with two
active, and look at how the router of the fresh layer spreads the 800
assignments over buffers of $\lceil 1.25 \cdot 800/8 \rceil = 125$ slots:

```{.python .input #moe-a-mixture-of-experts-layer-2}
%%tab pytorch
moe = MoELayer(256, num_experts=8, num_active=2)
X = torch.randn(4, 100, 256)
d2l.check_shape(moe(X), X.shape)
blk = d2l.TransformerBlock(256, num_heads=8,
                           ffn_factory=lambda: MoELayer(256, 8, 2))
d2l.check_shape(blk(X), X.shape)
print(f'{d2l.num_params(moe)} parameters stored, '
      f'{active_params(moe)} active per token')
print('shares of the assignments:',
      ' '.join(f'{s:.2f}' for s in (moe.load / moe.load.sum()).tolist()))
print(f'{float(moe.dropped) / float(moe.load.sum()):.0%} of them dropped')
```

```{.python .input #moe-a-mixture-of-experts-layer-2}
%%tab jax
moe = MoELayer(256, num_experts=8, num_active=2,
               rngs=nnx.Rngs(d2l.get_key()))
X = jax.random.normal(d2l.get_key(), (4, 100, 256))
d2l.check_shape(moe(X), X.shape)
blk = d2l.TransformerBlock(
    256, num_heads=8, rngs=nnx.Rngs(d2l.get_key()),
    ffn_factory=lambda rngs: MoELayer(256, 8, 2, rngs=rngs))
d2l.check_shape(blk(X), X.shape)
print(f'{d2l.num_params(moe)} parameters stored, '
      f'{active_params(moe)} active per token')
print('shares of the assignments:',
      ' '.join(f'{s:.2f}' for s in (moe.load / moe.load.sum()).tolist()))
print(f'{float(moe.dropped) / float(moe.load.sum()):.0%} of them dropped')
```

Each expert holds $3 \cdot 256 \cdot 683 = 524{,}544$ parameters, so the
layer stores about 4.2 million and a token uses about 1.05 million, a
quarter. A fresh router spreads random tokens roughly evenly: no expert's
share is far from the even share of $1/8$, and the buffers, a quarter
larger than an even share, drop almost no assignments. Random vectors are
not text, however, and the router has yet to be trained. The next part
follows the shares inside a language model.

## Routing Collapse and Load Balancing

The router and the experts learn together, and the way they learn tends to
concentrate the tokens on a few experts. This part explains why and
compares two remedies.

### Positive Feedback in Routing
:label:`subsec_moe-feedback`

Follow the feedback loop. At initialization some expert is, by chance,
slightly better than its peers on the tokens it receives. The loss
gradient then rewards it twice: its parameters improve on those tokens,
*and* the router's gradient raises its probability, because weighting a
helpful expert more heavily lowers the loss. More tokens bring more
gradient signal, which makes a better expert, which attracts more tokens.
The experts that lose the early rounds receive ever fewer tokens, learn
ever more slowly, and lose probability. This loop can end in *routing
collapse*: a few experts receive most of the assignments, and much of the
stored capacity goes unused. :citet:`Shazeer.Mirhoseini.Maziarz.ea.2017`
observed this imbalance and added balancing terms to the training loss. We
describe two remedies in use today.

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
weight $\alpha$, 0.01 in the Switch transformer. Its gradient,
which flows through $\bar{p}$ only, lowers the probability of overloaded
experts for every token, whatever the token needs. This is its drawback:
the balancing gradient is added to the language-modeling gradient at every
step, and the two conflict when some tokens do need a busy expert. With
$\alpha$ too large, routing serves balance rather than prediction; with
$\alpha$ too small, the imbalance persists :cite:`Wang.Gao.Zeng.ea.2024`.

### Balancing Without a Loss

:citet:`Wang.Gao.Zeng.ea.2024` balance the load without touching the loss,
and DeepSeek-V3 trains this way :cite:`Liu.Feng.Xue.ea.2024`. A bias $b_i$
per expert is added to the probabilities for the selection, and only for
the selection. After each optimizer step, a controller moves each bias by a
fixed amount $u$ toward balance:

$$
\mathcal{E}_k(\mathbf{x}) = \operatorname{argtop}_k \big(p_i(\mathbf{x}) + b_i\big),
\qquad
b_i \leftarrow b_i + u\, \mathrm{sign}(\bar{f} - f_i),
$$
:eqlabel:`eq_moe-bias`

where $\bar{f} = 1/E$ is the even share. The bias of an overloaded expert
falls, and that of an underloaded expert rises, which makes the
underloaded expert slightly easier to select. Since $b_i$ never enters the
weight $p_i$ in :eqref:`eq_moe-layer`, the model's loss contains no
balancing term, and no gradient competes with the language-modeling one.
The controller steers where tokens go; the gradients serve prediction
alone. The update rate sets how quickly the controller reacts. DeepSeek-V3
used $u = 0.001$ for the first 14.3 trillion of its training tokens; our
runs take 800 steps, and we use $u = 0.01$ so that the controller can
follow the router within a few dozen steps. DeepSeek-V3 also scores
experts with a sigmoid per expert instead of a softmax. Our layer keeps
the softmax and changes only the selection bias, the mechanism that
:citet:`Wang.Gao.Zeng.ea.2024` isolate.

### Balancing Methods Compared
:label:`subsec_moe-balance-comparison`

Both remedies fit into an ordinary training loop. The loop below is
`d2l.train_lm` of :numref:`sec_adam`, which draws minibatches, takes
optimizer steps, and returns the training losses, with the two remedies
added: `aux_weight` adds $\alpha$ times the sum of
$\mathcal{L}_{\mathrm{balance}}$ over the expert layers to the loss, and
`bias_rate` applies the controller of :eqref:`eq_moe-bias` after each
optimizer step, using the load of that step's minibatch.

```{.python .input #moe-three-runs-one-budget-1}
%%tab pytorch
def train_moe_lm(model, data, optimizer, num_steps, aux_weight=0.0,
                 bias_rate=0.0):
    """d2l.train_lm with a balancing loss and the bias controller."""
    device = d2l.try_gpu()
    model.to(device)
    layers, losses, step = moe_layers(model), [], 0
    while step < num_steps:
        for X, Y in data.train_dataloader():
            for l in layers:
                l.load.zero_()
            logits = model(X.to(device))
            ce = F.cross_entropy(logits.flatten(0, 1),
                                 Y.to(device).flatten())
            optimizer.zero_grad()
            (ce + aux_weight * sum(l.aux_loss for l in layers)).backward()
            optimizer.step()
            with torch.no_grad():  # The controller: not a gradient step
                for l in layers:
                    l.expert_bias += bias_rate * torch.sign(
                        l.load.mean() - l.load)
            losses.append(ce.detach())
            step += 1
            if step >= num_steps:
                return torch.stack(losses).tolist()
```

```{.python .input #moe-three-runs-one-budget-1}
%%tab jax
@nnx.jit(static_argnames=('aux_weight', 'bias_rate'))
def moe_train_step(model, optimizer, X, Y, aux_weight, bias_rate):
    for l in moe_layers(model):
        l.load[...] = jnp.zeros_like(l.load[...])
    def loss_fn(model):
        ce = optax.softmax_cross_entropy_with_integer_labels(
            model(X), Y).mean()
        aux = sum(l.aux_loss[...] for l in moe_layers(model))
        return ce + aux_weight * aux, ce
    (_, ce), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    for l in moe_layers(model):  # The controller: not a gradient step
        load = l.load[...]
        l.expert_bias[...] += bias_rate * jnp.sign(load.mean() - load)
    return ce

def train_moe_lm(model, data, optimizer, num_steps, aux_weight=0.0,
                 bias_rate=0.0):
    """d2l.train_lm with a balancing loss and the bias controller."""
    losses, step = [], 0
    while step < num_steps:
        for X, Y in data.train_dataloader():
            losses.append(moe_train_step(model, optimizer, jnp.asarray(X),
                                         jnp.asarray(Y), aux_weight,
                                         bias_rate))
            step += 1
            if step >= num_steps:
                return [float(l) for l in jax.device_get(losses)]
```

To read a trained model, we compute its validation loss with
`d2l.lm_perplexity` of :numref:`sec_lstm`, whose logarithm is the mean
cross-entropy in nats per byte, and count during that pass how many
validation tokens each expert receives and how many assignments the
buffers drop. The shares of each layer are returned busiest first:

```{.python .input #moe-three-runs-one-budget-2}
%%tab pytorch
def evaluate_moe(model, data):
    """Validation loss; each expert layer's shares of the validation
    tokens, busiest expert first; and the fraction of assignments dropped."""
    layers = moe_layers(model)
    for l in layers:
        l.load.zero_()
        l.dropped.zero_()
    loss = math.log(d2l.lm_perplexity(model, data))
    shares = [sorted((l.load / l.load.sum()).tolist(), reverse=True)
              for l in layers]
    dropped = sum(float(l.dropped) for l in layers) / sum(
        float(l.load.sum()) for l in layers)
    return loss, shares, dropped
```

```{.python .input #moe-three-runs-one-budget-2}
%%tab jax
def evaluate_moe(model, data):
    """Validation loss; each expert layer's shares of the validation
    tokens, busiest expert first; and the fraction of assignments dropped."""
    layers = moe_layers(model)
    for l in layers:
        l.load[...] = jnp.zeros_like(l.load[...])
        l.dropped[...] = jnp.zeros(())
    loss = math.log(d2l.lm_perplexity(model, data))
    shares = [sorted((l.load / l.load.sum()).tolist(), reverse=True)
              for l in layers]
    dropped = sum(float(l.dropped) for l in layers) / sum(
        float(l.load.sum()) for l in layers)
    return loss, shares, dropped
```

The experiments use the data and the training recipe of
:numref:`sec_gpt`. The data are the bytes of the Wikipedia articles of
`d2l.WikiText103`, about 270 million for training, in windows of 128 bytes
at random offsets, and the official validation split of about 1.1 million
bytes. A run almost never reads the same passage twice, so the validation
loss measures what a model has learned about English rather than how much
of the text it has memorized. The recipe, bundled as `d2l.LMOptimizer`, is
AdamW with weight decay on the weight matrices, a learning rate that rises
linearly over the first 100 steps and then follows a cosine to zero, and
clipping of the gradient norm at 1. Our model is a small GPT, two blocks
of width 128 with eight experts per block and one active per token, with the capacity factor
of 1.25. We first read a fresh model, then train three models for 800
steps each. The runs differ only in balancing: none, the auxiliary loss at
$\alpha = 0.01$, or the bias controller at $u = 0.01$. Each model draws
its own initialization.

```{.python .input #moe-three-runs-one-budget-3}
%%tab pytorch
data = d2l.WikiText103(batch_size=64, num_steps=128)
small_moe_gpt = lambda: d2l.GPT(len(data.vocab), num_hiddens=128,
                                num_heads=4, num_blks=2,
                                ffn_factory=lambda: MoELayer(128, 8, 1))
shares = {}

def report(name, model):
    val, shares[name], dropped = evaluate_moe(model, data)
    print(f'{name:>15}: validation loss {val:.2f}, busiest expert at '
          f'{8 * max(s[0] for s in shares[name]):.1f} times its share, '
          f'{dropped:.0%} dropped')

report('initialization', small_moe_gpt().to(d2l.try_gpu()))
for name, aux_weight, bias_rate in (('no balancing', 0.0, 0.0),
                                    ('auxiliary loss', 0.01, 0.0),
                                    ('bias controller', 0.0, 0.01)):
    model = small_moe_gpt()
    optimizer = d2l.LMOptimizer(model, lr=2e-3, num_steps=800, warmup=100)
    train_moe_lm(model, data, optimizer, 800, aux_weight, bias_rate)
    report(name, model)
```

```{.python .input #moe-three-runs-one-budget-3}
%%tab jax
data = d2l.WikiText103(batch_size=64, num_steps=128)
small_moe_gpt = lambda: d2l.GPT(
    len(data.vocab), num_hiddens=128, num_heads=4, num_blks=2,
    rngs=nnx.Rngs(d2l.get_key()),
    ffn_factory=lambda rngs: MoELayer(128, 8, 1, rngs=rngs))
shares = {}

def report(name, model):
    val, shares[name], dropped = evaluate_moe(model, data)
    print(f'{name:>15}: validation loss {val:.2f}, busiest expert at '
          f'{8 * max(s[0] for s in shares[name]):.1f} times its share, '
          f'{dropped:.0%} dropped')

report('initialization', small_moe_gpt())
for name, aux_weight, bias_rate in (('no balancing', 0.0, 0.0),
                                    ('auxiliary loss', 0.01, 0.0),
                                    ('bias controller', 0.0, 0.01)):
    model = small_moe_gpt()
    optimizer = d2l.LMOptimizer(model, lr=2e-3, num_steps=800, warmup=100)
    train_moe_lm(model, data, optimizer, 800, aux_weight, bias_rate)
    report(name, model)
```

The plot shows the shares of the eight experts, sorted from the busiest to
the idlest and averaged over the two layers, against the even share of
$1/8$:

```{.python .input #moe-three-runs-one-budget-4}
%%tab pytorch, jax
ranks = list(range(1, 9))
mean_shares = [[sum(rank) / len(rank) for rank in zip(*s)]
               for s in shares.values()]  # Mean over the layers
d2l.plot(ranks, mean_shares + [[1 / 8] * 8], 'expert, busiest first',
         'share of tokens', legend=list(shares) + ['even share'],
         fmts=('C7x:', 'C0o-', 'C1s--', 'C2^-.', 'k-'))
d2l.plt.xticks(ranks);
```

The fresh model is unbalanced before it has seen any data, unlike the
layer that we fed random vectors above, and its buffers drop many
assignments. At initialization the residual stream of the first block
carries little more than the embedding of the current byte. Most
occurrences of a byte therefore look alike to the router and go to the
same expert, and bytes are far from equally frequent: nearly one byte in
five is a space. In the second block, attention has added to every
position nearly the same average of the earlier bytes, and this
common offset tilts most tokens toward the same expert. Training without
balancing does not undo the imbalance. The routing stays concentrated: the
busiest expert of a layer receives several times its even share, some
experts receive almost none, and the buffers keep dropping the tokens that
the busy experts have no room for.

Both remedies spread the tokens much more evenly and drop few assignments,
and both runs end with a lower validation loss than the unbalanced one. In
the unbalanced model, experts that receive almost no tokens add parameters
that the model hardly uses, and many tokens skip their expert altogether.
At this scale neither remedy ends consistently ahead of the other. Their
difference shows in larger models: :citet:`Wang.Gao.Zeng.ea.2024` trained models of 1 and 3
billion parameters both ways, and the controller balanced the load much
more evenly than the auxiliary loss while reaching a slightly lower
validation perplexity.

## Mixture-of-Experts Language Models

Balancing makes the experts usable. Whether they make a better language
model, and how released models arrange them, is the subject of this last
part.

### A Mixture-of-Experts GPT
:label:`subsec_moe-gpt`

The expert layer was built to answer one question: at the same
computation per token, does a model with eight times the feed-forward
parameters predict text better? Extra parameters help only a model that
has more text than it can absorb, so we compare in that regime: the small
GPT of the balancing experiments, two blocks of width 128, trained on
about 50 million bytes, more than a hundred per parameter. The dense
`d2l.GPT` competes with the same model with each feed-forward network
replaced by eight experts, one active per token. The expert layers are
balanced by the auxiliary loss at $\alpha = 0.01$ and use the capacity
factor of 1.25, the settings of the Switch transformer, whose experiments
we compare with below. Each expert is a full-width `d2l.FeedForward`, so a
token passes through as many parameters in the mixture as in the dense
model, apart from the router's $128 \times 8$ weights per block.

```{.python .input #moe-a-mixture-of-experts-gpt-1}
%%tab pytorch
models = {'dense': d2l.GPT(len(data.vocab), num_hiddens=128, num_heads=4,
                           num_blks=2),
          'MoE': small_moe_gpt()}
for name, model in models.items():
    print(f'{name:>5}: {d2l.num_params(model) / 1e6:5.2f}M parameters, '
          f'{active_params(model) / 1e6:.2f}M active per token')
```

```{.python .input #moe-a-mixture-of-experts-gpt-1}
%%tab jax
models = {'dense': d2l.GPT(len(data.vocab), num_hiddens=128, num_heads=4,
                           num_blks=2, rngs=nnx.Rngs(d2l.get_key())),
          'MoE': small_moe_gpt()}
for name, model in models.items():
    print(f'{name:>5}: {d2l.num_params(model) / 1e6:5.2f}M parameters, '
          f'{active_params(model) / 1e6:.2f}M active per token')
```

The mixture stores 5.3 times the parameters of the dense model and uses
the same number per token. Both models train with the recipe of
:numref:`sec_gpt` for 6,000 steps, and we measure the validation loss
every 250 steps:

```{.python .input #moe-a-mixture-of-experts-gpt}
%%tab pytorch, jax
num_steps, every = 6000, 250
curves = {}
for name, model in models.items():
    optimizer = d2l.LMOptimizer(model, lr=2e-3, num_steps=num_steps,
                                warmup=100)
    curves[name] = []
    for _ in range(num_steps // every):
        train_moe_lm(model, data, optimizer, every, aux_weight=0.01)
        curves[name].append(math.log(d2l.lm_perplexity(model, data)))
    print(f'{name:>5}: validation loss {curves[name][-1]:.2f} nats')
d2l.plot(list(range(every, num_steps + 1, every)), list(curves.values()),
         'step', 'validation loss (nats)', legend=list(curves),
         fmts=('-', '--'))
```

Early in training the mixture trails the dense model: each fresh
expert's output is scaled by a probability near $1/8$, so the expert
layers start out much weaker than the dense feed-forward networks, and
each expert learns from only an eighth of the tokens. As the dense model
runs out of capacity, its curve flattens, while the mixture keeps
improving. Its validation loss falls below the dense model's after about
2,000 steps and stays below it to the end, by a few hundredths of a nat.
At the same computation per token, the stored experts make a better
model. Large studies report the same effect. The Switch transformer reached the quality of a dense T5
model several times faster at the same number of operations per token, and
it already gained from two experts :cite:`fedus2022switch`.
:citet:`Clark.Casas.Guy.ea.2022` found that routing improved models of
every size they trained, from 15 million to 1.3 billion parameters, with
the largest gains for the smallest models.

The gain needs data. Each expert sees about $1/E$ of the tokens, and the
extra parameters help only if the text holds more than the dense model can
learn from it. A corpus read many times over offers less, and ours is
read less than once. Exercise :exref:`ex_moe-data-regime` repeats the
comparison with the six-block GPT of :numref:`sec_gpt`, which at 1,000
steps has more parameters than its text can fill.
:citet:`Krajewski.Ludziejewski.Adamczewski.ea.2024` found that when the
training tokens grow with the model, the advantage of mixtures over dense
models grows with the training budget.

The gain also has costs that the operation count leaves out. The mixture
keeps 5.3 times the parameters in memory, together with AdamW's two state
values per parameter. Its step takes longer than its operation count
suggests: the buffers hold a quarter more rows than there are tokens,
copying tokens into the buffers and outputs back moves data without
computing, and the experts' weights are stacked anew in every step. And
wherever the load is uneven, the buffers drop assignments, which a dense
model never does.

### Fine-Grained and Shared Experts
:label:`subsec_moe-fine-grained`

Mixtral's experts are full copies of the feed-forward network, eight per
layer and two per token. DeepSeekMoE :cite:`Dai.Deng.Zhao.ea.2024` spends
such a budget differently, in the two steps shown in
:numref:`fig_moe-fine-grained`. Their example starts from 16 full-width
experts with 2 active per token. The first step cuts each expert into $m$
narrower ones and lets each token select $m$ times as many: with $m = 4$,
8 of 64 experts of a quarter width instead of 2 of 16 full-width ones.
Neither the stored nor the active parameter count changes. The number
of combinations a token can select does, from $\binom{16}{2} = 120$ to
$\binom{64}{8} \approx 4.4 \times 10^{9}$. A token can then assemble its
network from many narrow specialists, say one for chemical vocabulary and
another for nested clauses, instead of selecting two broad experts that
must each cover a wide range of inputs.

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
against 1.87), and about matched a GShard model with 1.5 times the expert
parameters and 1.5 times the computation :cite:`Dai.Deng.Zhao.ea.2024`.
Neither refinement changes the logic of our layer: narrower experts need
only a configurable width, and a shared expert is one more feed-forward
network whose output is added to the routed mixture. The exercises build
both.

Released models make different choices along these axes, as
:numref:`tab_moe-experts` shows.

:Routed-expert configurations of released MoE language models.
:label:`tab_moe-experts`

| Model | Experts per layer | Active $k$ | Expert width / $d$ | Shared expert |
|---|---|---|---|---|
| Mixtral 8x7B :cite:`Jiang.Sablayrolles.Roux.ea.2024` | 8 | 2 | 3.5 | none |
| DeepSeek-V3 :cite:`Liu.Feng.Xue.ea.2024` | 256 | 8 | 0.29 | 1 |
| Qwen3-235B :cite:`Yang.Li.Yang.ea.2025` | 128 | 8 | 0.38 | none |
| gpt-oss-120b :cite:`OpenAI.2025` | 128 | 4 | 1.0 | none |

The three later models use many more experts than Mixtral and between 4
and 8 active per token, and most of them make each expert narrower than
the model itself. They disagree on the shared expert: DeepSeek's models
have used shared experts since DeepSeekMoE, two in DeepSeek-V2
:cite:`DeepSeek-AI.2024` and one in DeepSeek-V3, whereas Qwen3 uses none.
How many experts a model should have, how wide, and how many tokens it
needs before its stored parameters help are questions of scaling, of the
kind that :numref:`sec_scaling-laws` asks for dense models.

## Summary

A dense transformer spends computation on every parameter for every token.
A mixture of experts replaces each block's feed-forward network by $E$
experts and a linear router that sends each token to $k$ of them, weighted
by the router's probabilities: the stored parameters grow with $E$, the
computation per token with $k$. An efficient layer copies each token into
a fixed-size buffer of each of its experts and runs all experts at once on
their buffers, at the cost of dropping the assignments that do not fit.
Left alone, routing concentrates on a few experts, both because similar
tokens go to the same expert and because routing reinforces itself. An auxiliary loss
on the product of load and probability, or a controller that biases only
the selection, evens out the load. With balanced experts, a GPT with eight
experts per block and one active per token reaches a lower validation loss
than the dense model at the same computation per token, once the dense
model has more text than it can absorb, while storing 5.3 times the
parameters; released models refine the design with many narrow
experts and a shared expert.

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
   $N$ tokens, and write $\mathbf{z}(\mathbf{x})$ for the router logits of
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

1. [code] **Feedback-loop ablations.** The argument of
   :numref:`subsec_moe-feedback` names two reinforcing effects: a
   frequently selected expert improves on its tokens, and the router
   gradient raises its selection probability.
    1. Stop the gradient at the router logits inside `MoELayer`, so that
       $\mathbf{W}_r$ keeps its initial values while every other parameter
       trains. Predict whether the unbalanced run of
       :numref:`subsec_moe-balance-comparison` still concentrates its
       routing, then rerun it and compare the busiest expert's share. What
       still changes the routing when $\mathbf{W}_r$ is fixed?
    1. Restore the router gradient and rerun the three runs with
       `num_active=2`. Does the unbalanced run still concentrate, and as
       strongly? Use the router gradient of
       problem :exref:`ex_moe-router-gradient` to explain how a second
       active expert changes the signal that the runner-up receives.

1. [code] **Bias controller in isolation.** The controller of
   :eqref:`eq_moe-bias` can be studied without training. Build
   `MoELayer(256, num_experts=8, num_active=1)`, draw a fixed batch of
   shape $(4, 256, 256)$, that is, 1024 standard-normal tokens in four
   sequences, and add one fixed standard-normal vector
   $\boldsymbol{\mu} \in \mathbb{R}^{256}$ to every token, which skews the
   untrained router toward a few experts. Then repeat 500 times: reset
   `load`, run the layer on the batch, and apply the bias update of
   `train_moe_lm`.
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

1. **Capacity factor.** ● Our layer gives each expert a buffer of
   $C = c\, Nk/E$ slots for a batch of $N$ tokens, as GShard and the Switch
   transformer do :cite:`Lepikhin.Lee.Xu.ea.2021,fedus2022switch`, and
   drops the assignments that do not fit.
    1. Without compilation, a layer could size its buffers by each
       batch's routing. Explain why the buffer size must still be fixed in
       advance when the experts live on different devices.
    1. Take $k = 1$, $N = 4096$, and a router that sends each token to a
       uniformly random expert, independently of the other tokens. The
       load of one expert is then binomial with mean $N/E$. Use a normal
       approximation, together with $\mathbb{E}[\max(Z, 0)] = 1/\sqrt{2\pi}$
       for a standard normal $Z$, to estimate the fraction of tokens
       dropped at $c = 1$ for $E = 8$ and for $E = 64$. How does the
       fraction scale with $E$ and $N$, and what does $c = 1.25$ change?
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

- count stored and active parameters of an expert layer;
- implement top-$k$ routing that runs each expert once on its tokens;
- explain why routing collapses, and balance it with a loss or a bias
  controller;
- read an MoE GPT against a dense GPT at equal computation per token;
- explain fine-grained and shared experts.

**Running example:** GPT-2 small with eight experts per block, then our
4.8M-parameter byte-level GPT.
:::

::: {.slide title="More Parameters per Operation"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| The Transformer Block | attention + FFN on a residual stream |
| A GPT from Scratch | a trained language model; GPT-2 loaded |
| Generation and the KV Cache | reuse keys and values while sampling |
| Encoders, Decoders, and Cross-Attention | the other two wirings |
| Vision Transformer | images as token sequences |
| **Mixture of Experts** | **more parameters per FLOP** |
| Scaling Laws | loss against parameters, data, compute |
:::

::: {.col .narrow}
- **Uses:** `d2l.FeedForward`, `d2l.GPT`, its data and training recipe.
- **Skips:** expert parallelism across devices (Language Models part).
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

::: {.slide title="Mixtral Stores 46.7B and Uses 12.9B; DeepSeek-V3 a 28× Gap"}
@!moe-conditional-computation

- "8x7B" names what is stored, not the work per token.
- DeepSeek-V3: 656B of 671B parameters sit in experts.
- The count assumes an even spread of tokens over the experts.
:::

::: {.slide title="The Probability Weight Carries the Router's Gradient"}
- Selecting the top $k$ is discrete: no gradient.
- The weight $p_i$ of a selected expert is differentiable.
- If more of expert $i$ would lower the loss, the gradient raises $p_i$,
  and the router selects $i$ more often for similar tokens.
- Switch: weight $= p_i$, and every logit gets a gradient through the
  softmax. Mixtral renormalizes over the selected $k$; with $k = 1$ its
  router would then learn nothing from the loss.
:::

::: {.slide title="Each Expert Gets a Fixed Buffer of C Slots"}
- The number of tokens an expert receives changes from batch to batch;
  matrix products need fixed shapes.
- Buffer of $C = \lceil c\, nk/E \rceil$ slots per expert; Switch trains
  with capacity factor $c = 1.25$.
- Number each expert's assignments in token order; number $r \geq C$ is
  *dropped* (the token keeps its residual stream).
- All experts run at once on $(E, C, d)$: about $c\,nk$ rows, a fraction
  $ck/E$ of running every expert on every token.
- Balanced router: few drops. Unbalanced: many.
:::

::: {.slide title="Dispatch: Number, Copy Into Buffers, Run, Gather Back" only="pytorch"}
```{.python #moe-exercises-1}
e = idx.flatten()                              # expert of each pair
onehot = F.one_hot(e, E)                       # (n k, E)
rank = (onehot.cumsum(0) * onehot).sum(1) - 1  # place in its queue
slot = torch.where(rank < C, e * C + rank, E * C)  # E C: dropped
buf = x.new_zeros(E * C + 1, d)                # + an overflow row
buf[slot] = x.repeat_interleave(k, 0)
Y = self.run_experts(buf[:-1].reshape(E, C, d)).reshape(E * C, d)
Y = torch.cat([Y, Y.new_zeros(1, d)])[slot]    # dropped: zeros
weights = P.gather(-1, idx)                    # (n, k): the p_i
out = (weights[..., None] * Y.reshape(n, k, d)).sum(1)
```

`run_experts`: SwiGLU as three batched products on the stacked weights.
:::

::: {.slide title="Dispatch: Number, Copy Into Buffers, Run, Gather Back" only="jax"}
```{.python #moe-exercises-2}
e = idx.ravel()                                # expert of each pair
onehot = jax.nn.one_hot(e, E, dtype=jnp.int32)  # (n k, E)
rank = (onehot.cumsum(0) * onehot).sum(1) - 1  # place in its queue
slot = jnp.where(rank < C, e * C + rank, E * C)  # E C: dropped
buf = jnp.zeros((E * C, d), x.dtype).at[slot].set(
    jnp.repeat(x, k, 0), mode='drop')
Y = self.run_experts(buf.reshape(E, C, d)).reshape(E * C, d)
Y = Y.at[slot].get(mode='fill', fill_value=0)  # dropped: zeros
weights = jnp.take_along_axis(P, idx, -1)      # (n, k): the p_i
out = (weights[..., None] * Y.reshape(n, k, d)).sum(1)
```

Every shape follows from $n$: one compilation, whatever the routing.
:::

::: {.slide title="Eight Experts, Two Active: A Quarter of the Parameters per Token"}
@!moe-a-mixture-of-experts-layer-2

- One expert: $3 \cdot 256 \cdot 683 = 524{,}544$ parameters.
- A fresh router spreads *random* tokens roughly evenly; buffers of 125
  slots for 100 assignments on average drop almost none.
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

::: {.slide title="Routing Feeds on Itself"}
- An expert that is, by chance, slightly better on its tokens is rewarded
  twice: its weights improve, **and** the router raises its probability.
- More tokens bring more gradient, a better expert, more tokens.
- The others see fewer tokens and learn more slowly.
- End point, *routing collapse*: a few experts take most tokens; the rest
  of the stored capacity goes unused.
:::

::: {.slide title="Remedy 1: Penalize Load Times Probability"}
$$\mathcal{L}_{\mathrm{balance}} = E \sum_{i=1}^{E} f_i\, \bar{p}_i$$

- $f_i$: fraction of the minibatch's assignments that expert $i$
  received (a count, no gradient); $\bar{p}_i$: mean probability of $i$.
- Even routing: $\mathcal{L}_{\mathrm{balance}} = 1$. One expert takes
  everything with $p \approx 1$: about $E$.
- Added with weight $\alpha = 0.01$ (Switch). Its gradient pushes every
  token away from busy experts, whatever the token needs.
:::

::: {.slide title="Remedy 2: Bias the Selection, Not the Weights"}
$$\mathcal{E}_k = \operatorname{argtop}_k(p_i + b_i), \qquad b_i \leftarrow b_i + u\,\mathrm{sign}(1/E - f_i)$$

- After each step, overloaded experts lose bias, underloaded ones gain.
- $b_i$ never enters the weight $p_i$: no balancing term in the loss, no
  competing gradient.
- DeepSeek-V3: $u = 0.001$ over trillions of tokens; our 800 steps:
  $u = 0.01$.
:::

::: {.slide title="Three Runs That Differ Only in Balancing"}
Two blocks of width 128, 8 experts, top-1, 800 steps on the bytes of
Wikipedia articles (WikiText-103); first a fresh model:

@!moe-three-runs-one-budget-3
:::

::: {.slide title="Why Text Is Unbalanced Before Any Training"}
- Block 1: the residual stream is little more than the byte's
  embedding, so a byte's occurrences go to one expert.
- Bytes are far from equally frequent: nearly one in five is a space.
- Block 2: attention adds to every position nearly the same average of
  the earlier bytes; the common offset tilts most tokens toward one
  expert.
- Busy experts overflow their buffers: the fresh model drops many
  assignments.
:::

::: {.slide title="Both Remedies Even Out the Load; No Balancing Stays Skewed"}
@!moe-three-runs-one-budget-4

- No balancing: busy experts keep several times their share.
- Both remedies: shares near $1/8$, few drops, lower validation loss.
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

::: {.slide title="Same Computation per Token, 5.3 Times the Parameters"}
The small GPT of the balancing runs, two blocks of width 128, dense and
with each FFN replaced by 8 experts, top-1, auxiliary loss
$\alpha = 0.01$, capacity factor 1.25:

@!moe-a-mixture-of-experts-gpt-1
:::

::: {.slide title="The MoE GPT Reaches a Lower Validation Loss Than the Dense GPT"}
@!moe-a-mixture-of-experts-gpt

- 6,000 steps, about 50M bytes, more than 100 per dense parameter.
- The mixture starts behind, crosses after about 2,000 steps, and stays
  below.
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

::: {.slide title="Narrow and Shared Experts: Same Budget, Far More Combinations"}
@fig:mdl-transformers-moe-fine

Same stored and active parameters in every row (DeepSeekMoE).
:::

::: {.slide title="Released Models Use Many Narrow Experts"}
| Model | Experts | $k$ | width$/d$ | shared |
|---|--:|--:|--:|---|
| Mixtral 8x7B | 8 | 2 | 3.5 | none |
| DeepSeek-V3 | 256 | 8 | 0.29 | 1 |
| Qwen3-235B | 128 | 8 | 0.38 | none |
| gpt-oss-120b | 128 | 4 | 1.0 | none |

- DeepSeekMoE at 2B stored parameters: loss 1.81 against 1.87 for GShard
  at equal stored and active size.
:::

::: {.slide title="Wrap-up"}
- MoE: $E$ FFNs and a router; parameters grow with $E$, operations per
  token with $k$. Mixtral 46.7B/12.9B; GPT-2 small with 8 experts
  521M/124M.
- Dispatch copies tokens into fixed buffers of $\lceil c\,nk/E \rceil$
  slots per expert and runs all experts at once; overflow is dropped.
- Routing feeds on itself; an auxiliary loss or a selection bias keeps
  the load even.
- Balanced, our MoE GPT beats the dense GPT at equal computation per
  token, given enough distinct text.
- Narrow experts and a shared expert: more combinations, common
  processing stored once.

**Next:** how loss scales with parameters, data, and compute.
:::
