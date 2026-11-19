# Scaling Laws
:label:`sec_scaling-laws`

Every model so far in this chapter was sized by hand. We chose a width, a depth
and a number of training steps, and if the result had disappointed us, we
could have changed them and trained again. A large model offers no second
try. The GPT of :numref:`sec_gpt` trained on about $2 \times 10^{14}$
floating-point operations. The largest Llama 3 model used $3.8 \times
10^{25}$ :cite:`Grattafiori.Dubey.Jauhri.ea.2024`, about $10^{11}$ times
as many, in a single run. How should such a budget be divided, before the
run starts, between the parameters of the model and the tokens it reads?

It turns out that the question can be answered from runs far smaller than
the one being planned, because two things about a run can be known before
it starts. Its cost follows from the shape of the model by counting, with
no need to run anything. Its loss can be predicted too, with less
certainty: the loss falls so smoothly as the model and its data grow that a
curve fitted to small runs can be extended to large ones. Such a fitted
curve is called a *scaling law*. The count is arithmetic. The law is an
empirical observation, and extending it over several orders of magnitude
is a bold step. It has held up well enough that the largest runs are now
planned this way.

We start with the count of operations and check it against the
framework's own. We then train fifteen small GPTs and ask the question in
miniature: for a fixed amount of computation, is a larger model trained
briefly better than a smaller one trained longer? Finally we take a law
fitted to hundreds of larger runs, derive the best division of a budget,
and see why deployed models depart from it.

```{.python .input #scaling-laws-scaling-laws-and-the-modern-recipe}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #scaling-laws-scaling-laws-and-the-modern-recipe}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #scaling-laws-1}
%%tab pytorch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.flop_counter import FlopCounterMode
```

## Counting Parameters and FLOPs

### Blocks and Embeddings

The cost of a run depends on how many parameters the model has and where
they sit, so we begin by counting them. All models in this section are
`d2l.GPT`s of :numref:`sec_gpt`, built through one shorthand:

```{.python .input #scaling-laws-counting-parameters-1}
%%tab pytorch
gpt = d2l.GPT
```

```{.python .input #scaling-laws-counting-parameters-1}
%%tab jax
def gpt(*args, **kwargs):  # Fresh random weights for every model
    return d2l.GPT(*args, **kwargs, rngs=nnx.Rngs(d2l.get_key()))
```

A GPT holds two kinds of parameters, and they grow differently. Each block
holds about $12d^2$ for width $d$: $4d^2$ in the attention projections
$\mathbf{W}_q$, $\mathbf{W}_k$, $\mathbf{W}_v$, $\mathbf{W}_o$, and $8d^2$
in the feed-forward network, as two matrices of width $4d$ with GELU or
three of width $8d/3$ with SwiGLU (:numref:`sec_transformer-block`). A stack
of $L$ blocks thus holds $12Ld^2$. The token embedding holds $Vd$ for a
vocabulary of $V$ tokens; weight tying uses it a second time as the output
head without storing a copy. A learned position table adds
$\textrm{max\_len} \times d$. The cell below counts both parts for our
byte-level model and for GPT-2 small:

```{.python .input #scaling-laws-a-parameter-census}
%%tab pytorch, jax
def census(model):
    L, d = len(model.blks), model.num_hiddens
    emb = d2l.num_params(model.token_emb) + (
        d2l.num_params(model.pos_emb) if model.pos == 'learned' else 0)
    total = d2l.num_params(model)
    print(f'embedding {emb / 1e6:6.2f}M ({emb / total:5.1%}),  '
          f'blocks {(total - emb) / 1e6:6.2f}M '
          f'= {(total - emb) / (L * d ** 2):.2f} L d^2,  '
          f'total {total / 1e6:.2f}M')

census(gpt(256))
census(gpt(50257, num_hiddens=768, num_heads=12, num_blks=12,
           pos='learned', norm='layer', act='gelu', bias=True))
```

Both rows confirm the $12Ld^2$ rule to a fraction of a percent. The residue is
normalization gains, the rounding of SwiGLU's width $8d/3$ to a whole number
in our model, and the biases of GPT-2. The two models differ in the share of
the embedding. It is 1.4% of our model, with its vocabulary of 256 bytes, but
nearly a third of GPT-2, whose 124 million parameters are an
85-million-parameter transformer attached to lookup tables of 39 million
entries. The token table holds 38.6 million of them and the position table 0.8
million. Comparing $Vd$ with $12Ld^2$ shows that the embedding dominates when
$V \gtrsim 12Ld$, that is, for small models with large vocabularies.

Which parameters should a scaling study count? The size of the table
follows the tokenizer, not the computation each token receives, so
:citet:`kaplan2020scaling` counted only the *non-embedding* parameters, and
our miniature study below does the same. The law of
:citet:`hoffmann2022training` counts all of them. For our byte-level models
the choice hardly matters: the table of 256 entries adds 7% to the smallest
model of the study and less than 1% to the largest. We therefore write $N$
for both and say which count is meant where it matters.

### Six FLOPs per Parameter and Token

Multiplying a vector by a matrix $\mathbf{W} \in \mathbb{R}^{m \times k}$
takes $mk$ multiplications and $mk$ additions, 2 floating-point operations
(FLOPs) per weight. In the forward pass every weight of a linear layer
meets every token once, so the forward pass costs about $2N$ FLOPs per
token. The backward pass costs twice as much. A linear layer $\mathbf{y} =
\mathbf{W}\mathbf{x}$ receives the gradient $\boldsymbol{\delta}$ with
respect to its output and computes two products from it: the gradient
$\mathbf{W}^\top \boldsymbol{\delta}$ with respect to its input, which
continues the chain rule into the layer below, and the gradient
$\boldsymbol{\delta} \mathbf{x}^\top$ with respect to its weights, which
the optimizer uses. Each costs as much as the forward product. Training on
$D$ tokens therefore takes

$$
C \approx \underbrace{2ND}_{\textrm{forward}} +
\underbrace{4ND}_{\textrm{backward}} = 6ND
$$
:eqlabel:`eq_six_nd`

FLOPs :cite:`kaplan2020scaling`, the estimate that :numref:`sec_gpt` used to
compare its run with larger ones. Generating a token without training needs
only the forward pass, about $2N$ FLOPs (:numref:`sec_kv-cache`). If one
formula from this section is worth memorizing, it is this one: it gives the
cost of a training run from two numbers, before any code is written.

The count leaves out all work that is not a product with a weight matrix.
The largest such term is attention itself. For a context of $n$ tokens,
each query takes a dot product with $n$ keys and averages $n$ values, about
$4nd$ FLOPs per token and block in the forward pass, against $24d^2$ for
the block's $12d^2$ weights. The ratio of the omitted to the counted work
is $n/(6d)$: 8% for our models, with $n = 128$ and $d = 256$, and 100% for
a model of width 4,096 that reads 24,576 tokens at once. Normalization,
softmax, activation functions, and the optimizer update add a few
operations per element, far fewer than the matrix products at these
widths. For dense transformers whose context is short compared with $6d$,
$6ND$ is the leading term of the cost.

### Checking the Arithmetic
:label:`subsec_scaling-laws-checking`

A derivation that drops terms deserves a check. We count the operations
of one training step of the default GPT, with a GELU feed-forward network
(the same $12d^2$ per block as SwiGLU, but exactly, since SwiGLU's width
$8d/3$ is not a whole number and must be rounded), on a minibatch of 64
windows of 128 tokens. Here $N = 4{,}784{,}128$ counts every weight matrix,
the token embedding included, because as the output layer it multiplies
every token like any other weight.

We can predict the answer before the cell prints it. Attention adds $4nd$
FLOPs per token in each of the $L = 6$ blocks, and the output layer adds
weights to $N$ but no attention work, so a counter that sees only matrix
products should report a forward cost of $1 + 4nLd/(2N) = 1.082$ times $2ND$,
a shade below $1 + n/(6d) = 1.083$. A counter that also counts softmax,
normalization and activations should report a little more. The cell prints the
forward cost against $2ND$, the cost of the whole step against $6ND$, and the
ratio of backward to forward work.

:begin_tab:`pytorch`
`FlopCounterMode` adds up the FLOPs of matrix products and counts nothing
for elementwise operations. The cell selects the math backend of attention
with `sdpa_kernel`, which writes attention as two batched matrix products
over the full $n \times n$ score matrix, masked half included, so that the
counter sees them.
:end_tab:

:begin_tab:`jax`
`cost_analysis()` of a compiled function counts the arithmetic XLA will
execute, without running it: the attention scores over the full
$n \times n$ matrix and the elementwise work of softmax, activations and
normalization. Transcendental functions such as the exponentials of the
softmax are tallied under a separate key and left out of `'flops'`. The
cell counts the loss function for the forward pass and its gradient for
the whole step.
:end_tab:

```{.python .input #scaling-laws-checking-the-arithmetic}
%%tab pytorch
model = gpt(256, act='gelu')
X = torch.randint(0, 256, (64, 128))
Y = torch.randint(0, 256, (64, 128))
N = sum(p.numel() for p in model.parameters() if p.ndim == 2)
with sdpa_kernel(SDPBackend.MATH):  # Attention as visible matrix products
    with FlopCounterMode(display=False) as fwd:
        loss = F.cross_entropy(model(X).flatten(0, 1), Y.flatten())
    with FlopCounterMode(display=False) as bwd:
        loss.backward()
f, b = fwd.get_total_flops(), bwd.get_total_flops()
print(f'forward {f:.3e} FLOPs, {f / (2 * N * X.numel()):.3f} x 2ND')
print(f'step    {f + b:.3e} FLOPs, {(f + b) / (6 * N * X.numel()):.3f} x 6ND')
print(f'backward / forward = {b / f:.2f}')
```

```{.python .input #scaling-laws-checking-the-arithmetic}
%%tab jax
model = gpt(256, act='gelu')
key_x, key_y = jax.random.split(d2l.get_key())
X = jax.random.randint(key_x, (64, 128), 0, 256)
Y = jax.random.randint(key_y, (64, 128), 0, 256)
graphdef, params, rest = nnx.split(model, nnx.Param, ...)
N = sum(p.size for p in jax.tree.leaves(params) if p.ndim == 2)
# Count on the CPU: on a GPU the count depends on the kernels XLA picks
params, rest, X, Y = jax.device_put((params, rest, X, Y), d2l.cpu())

def loss_fn(params, X, Y):
    logits = nnx.merge(graphdef, params, rest)(X)
    return optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(-1, logits.shape[-1]), Y.reshape(-1)).mean()

f = jax.jit(loss_fn).lower(params, X, Y).compile().cost_analysis()['flops']
s = jax.jit(jax.value_and_grad(loss_fn)).lower(
    params, X, Y).compile().cost_analysis()['flops']
print(f'forward {f:.3e} FLOPs, {f / (2 * N * X.size):.3f} x 2ND')
print(f'step    {s:.3e} FLOPs, {s / (6 * N * X.size):.3f} x 6ND')
print(f'backward / forward = {(s - f) / f:.2f}')
```

:begin_tab:`pytorch`
The counter lands on the prediction, 1.082, to three digits, and the
backward pass costs 2.00 times the forward pass.
:end_tab:

:begin_tab:`jax`
XLA's count lies 1 to 3% above the prediction of 1.082, at 1.094 for the
forward pass and 1.116 for the step: that excess is the elementwise work
that $6ND$ ignores. The backward pass costs 2.06 times the forward pass.
:end_tab:

The prediction holds, and the backward pass costs about twice the forward
pass, as the two products per layer said it would. A count of FLOPs also
converts into time: a run of $C$ FLOPs on hardware that sustains $R$ FLOPs
per second takes $C/R$ seconds, and the ratio of $R$ to the hardware's
peak, its *utilization*, is the subject of :numref:`chap_performance`.

## A Miniature Scaling Study

Counting gives the cost of a run but not the loss it reaches. For the loss
we have no derivation, only measurements, and the measurements are
surprisingly orderly. :citet:`kaplan2020scaling` trained language models
over many orders of magnitude and found that the loss falls smoothly as
the number of parameters, the amount of data, or the compute grows, as
long as neither of the others holds it back. Not all of the loss can fall,
since text has an entropy that no model removes. What falls is the excess
above that constant, the *reducible* loss, and it falls as a power law:
multiplying the model size by ten removes the same fraction of it, whatever
the size. We now measure this behavior on runs small enough for a reader to
rerun on one GPU.

### Five Sizes and Three Data Budgets
:label:`subsec_scaling-laws-five-sizes`

To separate the effect of model size from that of data, we train a grid of
models that varies the two independently. The corpus is the one of
:numref:`sec_gpt`, the bytes of the Wikipedia articles of
`d2l.WikiText103`, read in windows of 128 at random offsets:

```{.python .input #scaling-laws-a-corpus-bigger-than-a-novella}
%%tab pytorch, jax
data = d2l.WikiText103(batch_size=64, num_steps=128)
print(f'{len(data.train) / 1e6:.0f}M training bytes, '
      f'{len(data.val) / 1e6:.2f}M validation bytes, '
      f'vocabulary of {len(data.vocab)}')
```

The grid has two axes. Five sizes of `d2l.GPT` grow width and depth
together, as model families do: widths from 96 to 384 and depths from 3 to
8 blocks, or 0.33 to 14.2 million non-embedding parameters, a factor
of 43. Each size trains at three data budgets: 500, 1,000, and 2,000 steps of
64 windows, that is $D$ = 4.1, 8.2, and 16.4 million tokens. Even the
largest budget reads only about 6% of the training text, so the grid
varies the amount of data while repeating almost none of it.

Every run uses the recipe of :numref:`sec_gpt` (`d2l.LMOptimizer`: AdamW,
a linear warmup, a cosine decay of the learning rate to zero, and gradient
clipping). Three of its settings must depend on the run, because a scaling
study is only as good as the tuning of its worst point. First, every run
has its own schedule, which decays to zero at the end of that run's
budget. Reading the loss after 500 steps off a run scheduled for 2,000
would measure a model whose learning rate is still high;
:citet:`hoffmann2022training` named this mismatch as one reason why earlier
fits overestimated the loss of short runs.

The other two settings concern the learning rate. :numref:`sec_scaling`
showed that the best Adam learning rate falls as the width grows, in
proportion to $1/d$ by its argument from the first step (its sweep
measured a slope of about $-0.7$). A rate tuned at one width therefore
undertrains or destabilizes the others (Exercise
:exref:`ex_scaling-laws-frozen-learning-rate` measures the effect). We
anchor the peak rate at $2 \times 10^{-3}$ for width 256, the rate of
:numref:`sec_gpt`, and scale it as $256/d$. The warmup lasts a tenth of the
steps but at least 100 steps. The smallest model peaks at a rate of
$5.3 \times 10^{-3}$, and when it reaches that rate after only 50 steps, a
500-step run can stall and end far above its neighbors; with 100 steps of
warmup it ends in line with them. Since almost every passage is read once,
there is no dropout.

Each line of the output is one size, with its validation losses at the
three budgets and its training loss over the last 50 steps of the longest
run:

```{.python .input #scaling-laws-five-sizes-one-diet-1}
%%tab pytorch, jax
sizes = ((96, 3, 4), (128, 3, 4), (192, 4, 6), (256, 6, 8), (384, 8, 8))
budgets = (500, 1000, 2000)  # Steps of 64 windows of 128 bytes
Ns, train, val = [], [], {num_steps: [] for num_steps in budgets}
for d, num_blks, num_heads in sizes:
    for num_steps in budgets:
        model = gpt(len(data.vocab), num_hiddens=d, num_heads=num_heads,
                    num_blks=num_blks)
        optimizer = d2l.LMOptimizer(model, lr=2e-3 * 256 / d,
                                    num_steps=num_steps,
                                    warmup=max(100, num_steps // 10))
        losses = d2l.train_steps(model, data, optimizer, num_steps)
        val[num_steps].append(
            math.log(d2l.lm_perplexity(model, data.val_dataloader())))
    # The size of the models just trained, and the training loss of the
    # longest of their runs
    Ns.append(d2l.num_params(model) - d2l.num_params(model.token_emb))
    train.append(sum(losses[-50:]) / 50)
    print(f'd = {d:3d}, {num_blks} blocks, N = {Ns[-1] / 1e6:5.2f}M: '
          'validation ' + ' / '.join(f'{val[s][-1]:.2f}' for s in budgets)
          + f', training {train[-1]:.2f}')
```

### Loss against Size and Data

Before reading fifteen numbers we need to know how much any one of them can
be trusted. Each point is a single run from fresh random weights.
:numref:`subsec_gpt-run-to-run` trained the grid's 4.7M-parameter model on
8.2M tokens three times with this recipe, and the three runs agreed within
about 0.01 nats. Differences of several hundredths between points are
therefore real; differences of a few thousandths are not. The plot shows
the validation loss, in nats per byte, against the non-embedding count $N$
on log--log axes, one curve per data budget.

```{.python .input #scaling-laws-five-sizes-one-diet-2}
%%tab pytorch, jax
tokens = {s: s * 64 * 128 for s in budgets}  # Tokens read in s steps
d2l.plot(Ns, [val[s] for s in budgets], 'non-embedding parameters N',
         'validation loss (nats)',
         legend=[f'D = {tokens[s] / 1e6:.1f}M' for s in budgets],
         xscale='log', yscale='log', fmts=('C0o-', 'C1s-', 'C2^-'),
         legend_loc='outside')
```

Three regularities stand out. The loss falls as the model grows, at every
budget. It falls as the data grows, at every size: quadrupling the tokens
lowers it by 0.20 to 0.27 nats. And the gains from size shrink as the
model grows: tripling the model from 0.59M to 1.77M parameters lowers the
loss by 0.05 to 0.09 nats, while tripling it again, from 4.72M to 14.2M,
lowers it by only 0.02 to 0.04. Nothing here is overfitting, either: the
training loss of the longest runs lies one to three hundredths above the
validation loss, never below it.

The printout holds a comparison that is easy to miss on the plot. Start from
the smallest model on the smallest budget. Four times the data lowers its loss
by 0.25 nats or more; forty-three times the parameters, the largest model of
the grid on the same data, lowers it by 0.21 to 0.23. At this scale, four
times the data is worth more than forty-three times the parameters, and it
costs a tenth of the computation.

### Equal Compute

Size and data both lower the loss, and both cost computation, so the
question of the section is which of the two a fixed budget should go to. The
grid answers it if we read it along lines of equal compute instead of
along its axes. A set of runs of different sizes trained on one compute
budget is called an *IsoFLOP profile*; our grid offers a rough one. The
cell plots each size's three runs against their training compute
$C = 6ND$ and prints three runs whose costs lie within a factor of two,
each a larger model on fewer tokens than the one before:

```{.python .input #scaling-laws-equal-compute}
%%tab pytorch, jax
d2l.plot([[6 * N * tokens[s] for s in budgets] for N in Ns],
         [[val[s][i] for s in budgets] for i in range(len(Ns))],
         'training compute C = 6ND (FLOPs)', 'validation loss (nats)',
         legend=[f'N = {N / 1e6:.2f}M' for N in Ns], xscale='log',
         fmts=('C0o-', 'C1s-', 'C2^-', 'C3v-', 'C4D-'), legend_loc='outside')
for i, s in ((2, 2000), (3, 1000), (4, 500)):  # Larger N, fewer tokens
    print(f'N = {Ns[i] / 1e6:5.2f}M, D = {tokens[s] / 1e6:4.1f}M tokens: '
          f'C = {6 * Ns[i] * tokens[s]:.1e} FLOPs, '
          f'validation {val[s][i]:.2f}')
```

The smallest of the three models wins. With 1.77M parameters on 16.4M
tokens it ends 0.05 nats ahead of the 4.72M model on 8.2M tokens and 0.13
nats ahead of the 14.2M model on 4.1M tokens, five to thirteen times the
run-to-run spread. And it wins with the least computation of the three,
half that of the largest. The plot shows that this is no accident of the
three runs chosen: wherever two curves overlap in compute, the curve of
the smaller model lies lower. Within this grid, computation spent on more
tokens always beat computation spent on more parameters.

This is the trade that decides how a budget is spent, seen on fifteen
small runs. The grid cannot tell us where it turns, because it contains no
pair of runs in which the smaller model loses at equal compute. Larger
models must win eventually, since a small model trained forever can do no
better than its size allows. To find the point where they do, we need a
law for the loss as a function of both $N$ and $D$.

## Compute-Optimal Allocation

### The Chinchilla Law

Our grid spans a factor of 43 in size and 4 in data, too small a range to
plan a large run from. That needs a law fitted to runs at scale.
:citet:`hoffmann2022training` trained over 400 models, from 70 million to
over 16 billion parameters on 5 to 500 billion tokens, and fitted their
final losses with

$$
\mathcal{L}(N, D) = \mathcal{L}_\infty + \frac{A}{N^{\alpha}} +
\frac{B}{D^{\beta}}.
$$
:eqlabel:`eq_chinchilla_law`

The three terms separate what limits the loss. $\mathcal{L}_\infty$
(written $E$ in the paper) is the loss of an ideal model on the training
distribution, the entropy of the text that no budget removes.
$A/N^{\alpha}$ is the excess loss of a model with $N$ parameters trained
on unlimited data, and $B/D^{\beta}$ the excess loss of an unlimited model
trained on $D$ tokens; $N$ here counts all parameters, embeddings
included. The two power-law terms together are the reducible loss.
Because of $\mathcal{L}_\infty$, the loss itself bends on log--log axes
even when the law holds exactly: only the reducible part falls on a
straight line.

The form is a guess with three parts: a floor that no budget removes, one
power law for each resource, and a sum, as if a shortage of parameters and
a shortage of data did not interact. The last assumption is the bold one,
and our grid already strains it. The sum predicts that quadrupling the
data lowers the loss by the same amount for every model size, while the
grid shows 0.25 to 0.27 nats for the smallest model and 0.20 for the
largest (Exercise :exref:`ex_scaling-laws-fit-grid` fits both the sum and
a form in which the two interact). The rest of the section takes the sum
at face value.

The law also counts $D$ repeated tokens as $D$ fresh ones, an assumption
our grid never tests. For models of up to 9 billion parameters it holds
for about four passes over the data, after which each further pass is
worth less and less :cite:`Muennighoff.Rush.Barak.ea.2023`.

### The Optimal Split

A fixed budget $C \approx 6ND$ allows a large model trained briefly or a
small model trained long. For a given budget, the *compute-optimal* model
is the split of $C$ between $N$ and $D$ with the lowest loss. To find it,
substitute $D = C/(6N)$ into :eqref:`eq_chinchilla_law` and set the
derivative with respect to $N$ to zero. With $\partial D / \partial N =
-D/N$, the optimum satisfies

$$
\alpha \frac{A}{N^{\alpha}} = \beta \frac{B}{D^{\beta}}.
$$
:eqlabel:`eq_chinchilla_optimum`

The condition has a plain reading. On a fixed budget, a model 1% larger
reads 1% fewer tokens. The left side is how much the parameter term falls
for that step, the right side how much the data term rises, and at the
optimum the two cancel: moving FLOPs from parameters to data, or back,
gains nothing. Substituting $D = C/(6N)$ into
:eqref:`eq_chinchilla_optimum` gives $N^{\alpha+\beta} = \frac{\alpha
A}{\beta B} (C/6)^{\beta}$, and hence

$$
N^* = G \left(\frac{C}{6}\right)^{\beta/(\alpha+\beta)}, \quad
D^* = \frac{1}{G} \left(\frac{C}{6}\right)^{\alpha/(\alpha+\beta)}, \quad
\textrm{where } G = \left(\frac{\alpha A}{\beta B}\right)^{1/(\alpha+\beta)}.
$$
:eqlabel:`eq_chinchilla_allocation`

When $\alpha$ and $\beta$ are about equal, both $N^*$ and $D^*$ grow like
$C^{1/2}$: a budget a hundred times larger gives a model ten times larger,
trained on ten times as many tokens, and the ratio $D^*/N^*$ hardly changes
with the budget. :numref:`fig_mdl-transformers-chinchilla` draws the
geometry. A budget of equal compute is a straight line of slope $-1$ on
log--log axes, and the compute-optimal model is the point where that line
touches the lowest loss contour it reaches. Moving along the line in either
direction crosses to higher loss.

![Loss contours of the Chinchilla law with the refit constants of Besiroglu et al. (2024), introduced below, over parameters and training tokens. Dashed: budgets of equal compute $C = 6ND$; dots: the compute-optimal model of each budget; orange: the compute-optimal path. Chinchilla's square sits on the optimum of its budget, covering its dot; Llama 3 8B, on a budget close to Chinchilla's, lies far from the path at a tenth of the optimal size.](../img/mdl-transformers-chinchilla.svg)
:label:`fig_mdl-transformers-chinchilla`

The constants come from fits, and the fits disagree.
:citet:`hoffmann2022training` report $\mathcal{L}_\infty = 1.69$,
$A = 406.4$, $B = 410.7$, $\alpha = 0.34$, $\beta = 0.28$.
:citet:`Besiroglu.Erdil.Barnett.ea.2024` reconstructed the data from the
paper's figures, fitted the same form again, and obtained
$\mathcal{L}_\infty = 1.82$, $A = 482.0$, $B = 2085.4$, $\alpha = 0.348$,
$\beta = 0.366$. They found the original constants inconsistent with the
paper's two other estimates of the optimum (Hoffmann et al. estimated it
in three ways), and the reported confidence intervals far too narrow for
the number of runs. The cell evaluates
:eqref:`eq_chinchilla_allocation` for both fits at Chinchilla's own
budget, $5.76 \times 10^{23}$ FLOPs, at which Chinchilla trained 70
billion parameters on 1.4 trillion tokens, 20 tokens per parameter:

```{.python .input #scaling-laws-the-optimal-split}
%%tab pytorch, jax
fits = {'Hoffmann et al.': (1.69, 406.4, 410.7, 0.34, 0.28),
        'Besiroglu et al.': (1.82, 482.0, 2085.4, 0.348, 0.366)}

def optimum(C, L_inf, A, B, alpha, beta):
    G = (alpha * A / (beta * B)) ** (1 / (alpha + beta))
    N = G * (C / 6) ** (beta / (alpha + beta))
    return N, C / (6 * N)

for name, fit in fits.items():
    N, D = optimum(5.76e23, *fit)
    print(f'{name:17s} N* = {N / 1e9:3.0f}B, D* = {D / 1e12:.2f}T, '
          f'{D / N:3.0f} tokens per parameter')
```

Only the refit reproduces the allocation Chinchilla was trained with. The
published constants put the optimum at less than half of Chinchilla's 70
billion parameters and at twice its 1.4 trillion tokens, about 90 tokens per
parameter. Under the refit $\alpha$ and $\beta$ are nearly equal, so the
exponents of :eqref:`eq_chinchilla_allocation` are 0.51 for $N^*$ and 0.49 for
$D^*$; both grow almost exactly like $C^{1/2}$, and the optimum stays near 20
tokens per parameter over many orders of magnitude of compute. This ratio
belongs to that study's data, tokenizer, and model family; a different corpus
or architecture moves it. Our byte-level grid cannot be checked against 20
directly, since a byte carries much less text than a subword token and tokens
per parameter do not compare across tokenizers. Its three runs of similar cost
read 9, 1.7 and 0.3 tokens per parameter, far on the side where more data is
the better use of a budget, and the run that read the most won.

The exponent was once in dispute. :citet:`kaplan2020scaling`, fitting
language models counted by their non-embedding parameters, had found
$N^* \propto C^{0.73}$. Under that exponent a budget ten times larger goes
into a model 5.4 times larger trained on 1.9 times the tokens; under the
refit it goes into a model and a dataset each a little over three times
larger. Two analyses explain the gap. :citet:`Pearce.Song.2024` trace most
of it to the non-embedding count at small sizes, where the embedding is a
large share of the parameters (Exercise
:exref:`ex_scaling-laws-embedding-count` measures how the count moves a
slope). :citet:`Porian.Wortsman.Jitsev.ea.2024` trace it to three details
of the earlier runs: the output layer left out of the compute, a warmup
too long for the small models, and optimizer settings not tuned for each
size. Both conclude that the earlier answer came from how the small runs
were counted and, in Porian et al.'s account, tuned, not from a different
law. The second account is the rule of our miniature study at full scale:
a scaling study is only as good as the tuning of its worst point.

### Training Past the Optimum

Training is not the only cost. Training compute is spent once; a deployed
model spends about $2N$ FLOPs on every token it generates, for as long as
anyone uses it. A smaller model trained on more tokens can therefore be
cheaper over its life although its training is less efficient
:cite:`Sardana.Portes.Doubov.ea.2024`. How far do real models go in this
direction? The cell applies the refit to dense models with published sizes
and token counts (Chinchilla :cite:`hoffmann2022training`, Llama 3
:cite:`Grattafiori.Dubey.Jauhri.ea.2024`, OLMo 2 :cite:`OLMo.2025`, Gemma 3
:cite:`Gemma.Team.2025`), as if they had been trained on Chinchilla's data.
For each it prints the tokens per parameter, the model's size relative to
the compute-optimal model of the same training budget, and the extra loss
that the law predicts for the difference:

```{.python .input #scaling-laws-training-past-the-optimum}
%%tab pytorch, jax
L_inf, A, B, alpha, beta = fits['Besiroglu et al.']
loss = lambda N, D: L_inf + A / N ** alpha + B / D ** beta
reported = {'Chinchilla': (70e9, 1.4e12), 'Llama 3 8B': (8e9, 15e12),
            'Llama 3 405B': (405e9, 15.6e12), 'OLMo 2 7B': (7e9, 4.05e12),
            'Gemma 3 27B': (27e9, 14e12)}
for name, (N, D) in reported.items():
    N_opt, D_opt = optimum(6 * N * D, L_inf, A, B, alpha, beta)
    print(f'{name:12s} {D / N:5.0f} tokens per parameter, '
          f'{N / N_opt:4.2f} x optimal size, '
          f'loss +{loss(N, D) - loss(N_opt, D_opt):.3f}')
```

Llama 3 405B is 0.65 of the size the refit would choose for its budget,
and the law predicts only 0.001 nats of extra loss for the difference.
Near the optimum the loss is flat: a model a third smaller than optimal
loses almost nothing. Meta's own fit on its own data put the optimum for
that budget at 402 billion parameters and 16.55 trillion tokens
:cite:`Grattafiori.Dubey.Jauhri.ea.2024`, almost exactly what it trained.
That is 41 tokens per parameter against Chinchilla's 20, a real case of a
different setup, with its own corpus, tokenizer and model family, moving
the ratio. Chinchilla itself comes out at 0.96 of its optimum with no
extra loss, a check that the cell reproduces the fit it was built from.

The smaller models are trained far past the optimum, by design. Llama 3 8B
reads almost 1,900 tokens per parameter, and the law puts it at a tenth of
the compute-optimal size for its budget, with about 0.05 nats of extra
loss; OLMo 2 7B and Gemma 3 27B sit between the two Llamas, at about a
fifth of the optimal size. Is 0.05 nats a lot? These are nats per token of
Chinchilla's tokenizer, not the nats per byte of our grid, so the right
yardstick is the reducible loss under the same law: 0.15 nats for the
compute-optimal model of this budget and 0.20 for Llama 3 8B, a third
more.

A fairer comparison holds the loss fixed instead of the budget. The
compute-optimal model with the same predicted loss as Llama 3 8B has
about 34 billion parameters, reads about 0.64 trillion tokens, and needs
18% of the training computation. Llama 3 8B thus costs five and a half
times as much to train as its compute-optimal equal, and in return each
token it generates costs about a quarter as much, 8 billion parameters
against 34. The savings in inference exceed the extra training after
about 11 trillion generated tokens (Exercise
:exref:`ex_scaling-laws-training-for-inference` derives these numbers).
That is fewer tokens than the model read in training. A model with many
users passes that point, and from then on the small model is the cheap
one.

A scaling law predicts one number, the loss on text like the training
data. It says nothing directly about what a model can do with that loss:
whether it answers a question, writes a program, or follows an
instruction is measured separately, and small differences in loss need
not show up there in proportion. Nor does the law cover every
architecture. A mixture of experts (:numref:`sec_moe`) stores many more
parameters than it uses for each token, so its training cost follows the
parameters it uses while its capacity follows all of them, and the law
above was fitted to dense models only. The next section builds such a
model and measures what it gains from parameters that each token skips.

## Summary

Training a dense transformer with $N$ parameters on $D$ tokens costs about
$6ND$ FLOPs, two per parameter and token in the forward pass and four in
the backward pass; attention adds a fraction $n/(6d)$ for contexts of $n$
tokens, and the framework's own count confirms both. The loss is not
derived but measured, and it falls smoothly with size and with data. In a
grid of fifteen small GPTs, the smaller model trained on more tokens beat
the larger model trained briefly at equal compute, by several times the
run-to-run spread. The Chinchilla law $\mathcal{L} = \mathcal{L}_\infty +
A/N^{\alpha} + B/D^{\beta}$ turns this trade into a rule: under a budget
$C \approx 6ND$, parameters and tokens both grow about like $C^{1/2}$, at
about 20 tokens per parameter in the refit of the Chinchilla data. Deployed
models that will generate many tokens are trained far past this point,
because a smaller model is cheaper to run.

## Exercises

1. **Attention-score correction.** :eqref:`eq_six_nd` omits the
   attention-score and value-mixing work, about $4nd$ FLOPs per token and
   block in the forward pass for a context of $n$ tokens, against about
   $24d^2$ for the block's linear maps.
    1. For the model of :numref:`subsec_scaling-laws-checking`
       ($n = 128$, $d = 256$, six blocks, eight heads), estimate the
       elementwise work of one block per token (the RoPE rotation of
       queries and keys, the attention softmax, the GELU of the
       $4d$-wide hidden layer, two normalizations, and two residual
       additions) at a few FLOPs per element. Express it as a fraction of
       $2N$ per token. Is it large enough to change the forward cost by
       more than a few percent? If your framework's counter counts
       elementwise work, compare your estimate with its excess over the
       prediction $1.082 \times 2ND$.
    1. A fused causal kernel skips the masked half of the score matrix.
       Llama 3 8B has width $d = 4096$, was pretrained at context 8,192,
       and was later extended to 131,072
       :cite:`Grattafiori.Dubey.Jauhri.ea.2024`. Treat each of its blocks
       as $12d^2$ parameters. By what factor does $6ND$ underestimate the
       training cost per token at each context length, with and without
       causal skipping?

1. **Variations on six.** The factor $6 = 2 + 4$ in :eqref:`eq_six_nd`
   assumes that every linear layer computes its forward product, the
   gradient with respect to its input, and the gradient with respect to
   its weights, each once. Ignore the embedding and the output head, and
   give the training cost per token in multiples of $N$ for each of the
   following variants.
    1. Activation checkpointing (:numref:`subsec_activation_checkpointing`)
       stores only each block's input and recomputes the block's forward
       pass during the backward pass.
    1. The bottom half of the blocks is frozen, and only the top half is
       trained.
    1. The top half of the blocks is frozen, and only the bottom half is
       trained. Explain why this answer differs from the previous one.

1. **Total versus non-embedding parameters.** Suppose that each model in
   `sizes` had used GPT-2's vocabulary of $V = 50{,}257$ tokens with tied
   embeddings and had reached the validation losses it reached at the
   largest budget with the 256-byte vocabulary.
    1. Compute the total parameter count of each model and the ratio
       between the largest and the smallest. Compare this ratio with the
       one for the non-embedding counts in `Ns`.
    1. Take two models with non-embedding counts $N_1 < N_2$ and total
       counts $T_1 < T_2$. Show that the log--log slope of loss against
       total count equals the slope against non-embedding count multiplied
       by $\log(N_2/N_1) / \log(T_2/T_1)$, and evaluate this factor for the
       smallest and the third-smallest model.
    1. Repeat the evaluation for Llama 3's vocabulary of 128,256 tokens.
       Which of the three exponents would you use to predict the loss of a
       much larger model, one whose embedding is a negligible share of its
       parameters, and why do the other two mislead?
   :exlabel:`ex_scaling-laws-embedding-count`

1. **Pricing a mixture-of-experts run.** A mixture-of-experts model passes
   each token through only some of its parameters, its *active* ones.
   DeepSeek-V3 :cite:`Liu.Feng.Xue.ea.2024` has 671B parameters, of which
   37B are active for each token, and was pretrained on 14.8T tokens in 2.664M
   H800 GPU hours. Take the H800's dense peak as about 990 TFLOP/s in
   BF16 and twice that in FP8, the precision in which the run executes
   the matrix multiplications of its linear layers.
    1. Which of the two parameter counts belongs in :eqref:`eq_six_nd`,
       and why? Estimate the pretraining FLOPs, and compare them with those
       of a dense 671B model trained on the same tokens.
    1. Compute the utilization implied by your estimate against each peak.
       Name two kinds of work in this run that the estimate omits, so that
       the true utilization is higher.

1. [code] **Frozen learning rate.** The sweep of
   :numref:`subsec_scaling-laws-five-sizes` scales the peak learning rate
   as $2 \times 10^{-3} \cdot 256/d$.
    1. Using the width dependence of the tuned learning rate in
       :numref:`sec_scaling`, predict which points of the largest budget
       move, and in which direction, when every model uses $2 \times
       10^{-3}$. Then rerun that column of the sweep with this rate.
    1. :numref:`subsec_gpt-run-to-run` measured the run-to-run spread of
       the 4.7M-parameter model of the sweep. By how much does the frozen
       rate change the validation loss of each model, compared with this
       spread? Which part of the shape of the frozen-rate curve is a
       tuning artifact?
   :exlabel:`ex_scaling-laws-frozen-learning-rate`

1. [code] **A sixth size.** ● A sixth model for the sweep of
   :numref:`subsec_scaling-laws-five-sizes` has width 512, ten blocks, and
   eight heads, about 31.5M non-embedding parameters.
    1. Before training it, predict its validation loss at the largest
       budget twice: once by extending the straight line through the two
       largest models of the largest-budget curve on the log--log plot,
       and once by assuming no improvement over the largest model. Which
       of the two guesses is optimistic, given how the grid's steps in
       size shrank? Estimate the run's cost with :eqref:`eq_six_nd`.
    1. Train it with the same recipe at all three budgets. Which
       prediction was closer, and what do the three results say about the
       headroom that a budget of 16.4M tokens leaves?
    1. Its 500-step run costs about as much as the 1,000-step run of the
       14.2M model. Predict which of the two ends lower, and check.

1. [code] **Fitting the law to the grid.** ● Fit
   :eqref:`eq_chinchilla_law` to the fifteen validation losses of the
   sweep, with $D$ in tokens, by minimizing the squared error of
   $\log \mathcal{L}$ over $\mathcal{L}_\infty$, $\log A$, $\log B$,
   $\alpha$, and $\beta$ with a
   gradient-based optimizer from several starting points.
    1. Report the fitted constants and the largest residual. Is the fitted
       $\mathcal{L}_\infty$ below every measured loss?
    1. Refit fifteen times, each time leaving one point out, and report the
       range of $\alpha$, $\beta$, and $\mathcal{L}_\infty$. Which
       constant is determined worst, and why?
    1. The grid's data gains shrink with model size, which the sum cannot
       describe. The law of :citet:`kaplan2020scaling`,
       $\mathcal{L}(N, D) = \left[(N_c/N)^{\alpha_N/\alpha_D} +
       D_c/D\right]^{\alpha_D}$, lets size and data interact. Show that for
       $\alpha_D < 1$ it predicts the opposite of the grid: the larger the
       model, the more a fixed multiple of the data lowers its loss. Fit it
       anyway and compare its residuals with those of part 2. What would a
       form that matches the grid have to do?
   :exlabel:`ex_scaling-laws-fit-grid`

1. **How the ratio moves with the budget.** ● The compute-optimal
   allocation is given by :eqref:`eq_chinchilla_allocation`.
    1. Show that $D^*/N^* = G^{-2} (C/6)^{(\alpha - \beta)/(\alpha +
       \beta)}$.
    1. Evaluate $D^*/N^*$ at $C = 10^{21}$ and $C = 10^{25}$ FLOPs for both
       fits in `fits`. In which direction does the ratio move with the
       budget under each fit, and by how much?
    1. Both fits come from the same experiments. What do the results say
       about how precisely a fit of :eqref:`eq_chinchilla_law` determines
       the token--parameter ratio at budgets far beyond the largest run
       that was fitted? ($10^{25}$ FLOPs is over a thousand times
       Chinchilla's largest IsoFLOP budget of $3 \times 10^{21}$.)

1. **Training for inference.** ● A model that serves $D_{\textrm{inf}}$
   tokens after training costs about

   $$C_{\textrm{life}} = 6ND + 2N D_{\textrm{inf}}$$

   FLOPs over its lifetime, since generating a token costs
   about $2N$ (:numref:`sec_kv-cache`). :citet:`Sardana.Portes.Doubov.ea.2024`
   minimize this cost at a fixed target loss instead of minimizing the
   loss at a fixed training budget.
    1. Minimize $C_{\textrm{life}}$ subject to $\mathcal{L}(N, D) = \ell$
       under :eqref:`eq_chinchilla_law`. Derive the optimality condition
       that replaces :eqref:`eq_chinchilla_optimum`, and show that the
       optimum moves to smaller $N$ and larger $D$ as $D_{\textrm{inf}}$
       grows.
    1. Llama 3 8B was pretrained on about 15T tokens. Take the refit of
       :citet:`Besiroglu.Erdil.Barnett.ea.2024` at face value and find the
       compute-optimal model with the same predicted loss as $N = 8 \times
       10^9$, $D = 1.5 \times 10^{13}$. Compute the ratio of the two
       training costs, the ratio of their costs per generated token, and
       the number of inference tokens at which the two lifetime costs are
       equal.
   :exlabel:`ex_scaling-laws-training-for-inference`

1. [extended] **IsoFLOP profiles.** ● The sweep of
   :numref:`subsec_scaling-laws-five-sizes` fixes three data budgets, and
   the body could only read it along lines of roughly equal compute.
   :citet:`hoffmann2022training` instead fix the budget $C$ exactly and
   trade $N$ against $D = C/(6N)$.
    1. For $C \in \{3 \times 10^{13}, 10^{14}, 3 \times 10^{14}\}$ FLOPs,
       train each model in `sizes` for $C / (6N \cdot 64 \cdot 128)$ steps,
       rounded (skip runs shorter than 200 steps), with the recipe of the
       sweep. At the smallest budget the lowest loss falls on the smallest
       model, as the body's equal-compute reading predicts; add smaller
       models, such as width 64 with two blocks, until the lowest-loss
       model of each budget has a trained neighbor on either side. Refine
       each minimum by fitting a parabola in $\log N$ to the points around
       it,
       fit $N_{\textrm{opt}} \propto C^{a}$, and compare $a$ with
       $\beta / (\alpha + \beta)$ for both fits in `fits`.
    1. Count the share of the training text that each run reads. Suppose
       the corpus had only 5 million bytes, so that the largest runs read
       it several times. In which direction would repetition bias
       $N_{\textrm{opt}}$ at the largest budget?

    *Adapted from Stanford CS336,
    [Assignment 3](https://github.com/stanford-cs336/assignment3-scaling/blob/main/cs336_assignment3_scaling.pdf),
    Problem (chinchilla_isoflops).*

[Discussions](https://d2l.discourse.group/t/36780)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §11.7]{.kicker}

Scaling Laws<br>
**the cost of a run · a miniature scaling study · the compute-optimal split · training past the optimum**
:::
:::

::: {.slide title="A 3.8×10²⁵-FLOP Run Must Be Planned Before It Starts"}
[The problem]{.kicker}

| run | training FLOPs |
|:--|--:|
| our 4.8M-parameter GPT, 8.2M tokens | $2 \times 10^{14}$ |
| GPT-2 small, 124M parameters, ~10B tokens (estimate) | $7 \times 10^{18}$ |
| Llama 3, 405B parameters, 15.6T tokens | $3.8 \times 10^{25}$ |

::: {.d2l-note .warn}
The largest run is done once. How large a model, and how many tokens,
must be decided before it starts.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- estimate the training cost of a transformer as $6ND$ and say what it
  leaves out;
- read a grid of small runs along lines of equal compute;
- derive the compute-optimal split of a budget between parameters and
  tokens;
- explain why smaller deployed models train far past 20 tokens per
  parameter.

**Running example:** GPT-2 small (124M parameters), and five small GPTs
on the bytes of WikiText-103.
:::

::: {.slide title="Cost Follows from the Shape, Loss from Size and Data"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| The Transformer Block | attention + FFN on a residual stream |
| A GPT from Scratch | a trained language model; GPT-2 loaded |
| The KV Cache | reuse keys and values while sampling |
| Shrinking the KV Cache | fewer heads, narrower or shorter caches |
| Encoders, Decoders, and Cross-Attention | the other two wirings |
| Vision Transformer | images as token sequences |
| **Scaling Laws** | **loss against parameters, data, compute** |
| Mixture of Experts | more parameters per FLOP |
| The Modern Recipe | released models as settings |
:::

::: {.col .narrow}
- **Uses:** `d2l.GPT` and its recipe, learning rate
  $\propto 1/\textrm{width}$.
- **Skips:** data curation; repetition beyond four passes (Language Models
  part).
:::
:::
:::

::: {.slide title="GPT-2's 124M Parameters: an 85M Transformer, 39M of Tables"}
Blocks hold $12Ld^2$ ($4d^2$ attention, $8d^2$ FFN); the embedding $Vd$:

@!scaling-laws-a-parameter-census

- The embedding dominates when $V \gtrsim 12Ld$: small models, large
  vocabularies.
- Kaplan et al. (2020), and our grid, count **non-embedding** $N$;
  Chinchilla counts all parameters.
:::

::: {.slide title="Training Costs Six FLOPs per Parameter and Token"}
$$C \approx \underbrace{2ND}_{\textrm{forward}} + \underbrace{4ND}_{\textrm{backward}} = 6ND$$

- Forward: each weight meets each token once, one multiply and one add.
- Backward: given $\boldsymbol{\delta}$, the gradient with respect to the
  layer's output, two products of the same size:
  $\mathbf{W}^\top\boldsymbol{\delta}$ (for the layer below) and
  $\boldsymbol{\delta}\mathbf{x}^\top$ (for the optimizer).
- Generating a token needs only the forward pass: about $2N$.
:::

::: {.slide title="Attention Adds a Fraction n/(6d) of the Counted Work"}
Per token and block, forward:

| work | FLOPs |
|:--|--:|
| products with the $12d^2$ weights | $24d^2$ |
| scores $\mathbf{q}^\top\mathbf{k}$ over $n$ keys, average of $n$ values | $4nd$ |

- Our models: $n = 128$, $d = 256$ → **8%**.
- Width 4,096 at 24,576 tokens → 100%.
- Norms, softmax, activations: a few FLOPs per element, much less.
:::

::: {.slide title="The Framework's Count Confirms 6ND Plus the Attention Term"}
Prediction: $1 + 4nLd/(2N) = 1.082$ times $2ND$
($n = 128$, $d = 256$, $L = 6$, $N = 4.78$M, output layer included).

@!scaling-laws-checking-the-arithmetic

- A counter of matrix products alone should land on **1.082**.
- A counter that also sees elementwise work should land a few percent
  above it.
- Backward / forward $\approx 2$, the 4-to-2 of the derivation.
:::

::: {.slide title="Check: What Did GPT-2 Small Cost?"}
[Check]{.kicker}

GPT-2 small: 124M parameters, roughly 10B training tokens (an estimate;
the paper does not report it), context 1,024, width 768. Estimate the
training FLOPs and the attention correction.

. . .

- $6ND = 6 \cdot 1.24 \times 10^{8} \cdot 10^{10} \approx 7 \times 10^{18}$.
- $n/(6d) = 1024 / 4608 \approx 22\%$: the correction is no longer small.
:::

::: {.slide title="A Grid of Five Sizes and Three Data Budgets"}
Corpus: WikiText-103 bytes, **270M** for training, read at random offsets.

| axis | values |
|:--|:--|
| width, blocks | 96/3, 128/3, 192/4, 256/6, 384/8 |
| $N$ (non-embedding) | 0.33M → 14.2M, a factor of 43 |
| $D$ (tokens) | 4.1M, 8.2M, 16.4M: at most 6% of the text, almost none repeated |

All 15 runs: AdamW, decay 0.1 on matrices, warmup, cosine, clipping;
no dropout.
:::

::: {.slide title="Every Run Gets Its Own Schedule and Its Own Learning Rate"}
- **Schedule:** the cosine decays to zero at the end of *each* run.
  A 500-step loss read off a 2,000-step run is taken at a high learning
  rate and overstates the loss (Hoffmann et al., 2022).
- **Learning rate:** the best Adam rate falls like $1/d$, so the peak
  rate is $2 \times 10^{-3} \cdot 256/d$.
- **Warmup:** a tenth of the steps, at least 100. With 50, the smallest
  model's 500-step run (peak $5.3 \times 10^{-3}$) can stall far above
  its neighbors; with 100 it ends in line.

::: {.d2l-note}
A study is only as good as the tuning of its worst point.
:::
:::

::: {.slide title="Loss Falls with Size and Data; Each Tripling of the Model Gains Less"}
@!scaling-laws-five-sizes-one-diet-1

- Down a column, $N$ grows: 0.59M → 1.77M gains 0.05–0.09 nats,
  4.72M → 14.2M only 0.02–0.04.
- Along a line, $D$ grows: 4× the tokens gains 0.20–0.27 nats.
- Run-to-run spread of one point: about 0.01 nats.
:::

::: {.slide title="Check: Four Times the Data or 43 Times the Parameters?"}
[Check]{.kicker}

Start from the smallest model (0.33M) on the smallest budget (4.1M
tokens). Which lowers the loss more: 4× the tokens, or the 14.2M model on
the same tokens? Which costs more compute?

. . .

- 4× the data: **0.25 nats or more**, at 4× the compute.
- 43× the parameters: 0.21–0.23 nats, at 43× the compute.
- At this scale, data lowers the loss more, at a tenth of the compute.
:::

::: {.slide title="At Equal Compute, the Smaller Model Trained Longer Wins"}
@!scaling-laws-equal-compute

- Gaps of 0.05 and 0.13 nats, 5–13× the run-to-run spread; the winner
  used half the FLOPs of the 14.2M model.
- Wherever two curves overlap in compute, the smaller model lies lower.
:::

::: {.slide title="The Chinchilla Law Separates Three Limits"}
$$\mathcal{L}(N, D) = \mathcal{L}_\infty + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}$$

- $\mathcal{L}_\infty$ ($E$ in the paper): the entropy of the text, which
  no budget removes.
- $A/N^{\alpha}$: the excess loss of finite parameters.
- $B/D^{\beta}$: the excess loss of finite data.
- The sum assumes that size and data do not interact: the bold
  assumption, which our grid already strains.
- A fixed budget trades the second term against the third.

[Fitted to 400+ runs, 70M–16B parameters, 5–500B tokens (Hoffmann et al., 2022); $N$ counts all parameters.]{.small}
:::

::: {.slide title="At the Optimum, Both Shrinkable Terms Fall at Equal Rates"}
Under $C \approx 6ND$, substitute $D = C/(6N)$ and set $\partial \mathcal{L}/\partial N = 0$:

$$\alpha \frac{A}{N^{\alpha}} = \beta \frac{B}{D^{\beta}}
\quad\Rightarrow\quad
N^* \propto C^{\beta/(\alpha+\beta)}, \;\; D^* \propto C^{\alpha/(\alpha+\beta)}$$

- $\alpha \approx \beta$: both grow like $C^{1/2}$.
- $D^*/N^*$ hardly depends on the budget.
:::

::: {.slide title="The Optimum Is Where a Budget Line Touches a Loss Contour"}
::: {.cols .vc}
::: {.col .fig}
@fig:mdl-transformers-chinchilla
:::

::: {.col .narrow}
- Dashed: equal compute $C = 6ND$, slope $-1$.
- Orange: the compute-optimal path, slope $\approx 1$.
- Llama 3 8B: a similar budget at a tenth of the optimal size.
:::
:::
:::

::: {.slide title="Only the Refit Reproduces Chinchilla's 20 Tokens per Parameter"}
@!scaling-laws-the-optimal-split

- Hoffmann et al.: $\alpha = 0.34$, $\beta = 0.28$; refit (Besiroglu et
  al., 2024): $\alpha = 0.348$, $\beta = 0.366$.
- Published constants: **93** tokens per parameter; refit: **18**;
  Chinchilla trained 70B on 1.4T, **20**.

::: {.d2l-note}
The constants come from fits, and the fits disagree.
:::
:::

::: {.slide title="Counting and Tuning Small Runs Moved the Exponent from 0.73 to 0.5"}
| fit | $N^* \propto$ | 10× the budget gives |
|:--|:--|:--|
| Kaplan et al. (2020) | $C^{0.73}$ | 5.4× the model, 1.9× the tokens |
| refit of Chinchilla | $C^{0.51}$ | a little over 3× each |

- Pearce and Song (2024): non-embedding counts at small sizes, where the
  embedding is a large share.
- Porian et al. (2024): output layer left out of the compute, warmup too
  long for small models, optimizer not tuned per size.
- The earlier answer came from how small runs were counted and set up,
  not from a different law.
:::

::: {.slide title="Check: A Budget One Hundred Times Larger"}
[Check]{.kicker}

Under the refit, the budget grows from $10^{23}$ to $10^{25}$ FLOPs. How
much larger is the compute-optimal model, and how many more tokens does it
read?

. . .

- $N^*$ and $D^*$ both grow about like $C^{1/2}$: **about 10× each**
  (exponents 0.51 and 0.49).
- The ratio stays near 20 tokens per parameter.
:::

::: {.slide title="Smaller Models Train Far Past the Optimum to Make Generation Cheap"}
@!scaling-laws-training-past-the-optimum

- Generating a token costs $2N$, for the model's whole life.
- Llama 3 405B: 0.65 of the optimal size costs 0.001 nats; the loss is
  flat near the optimum.
- Llama 3 8B: a tenth of the optimal size, +0.05 nats: reducible loss
  0.20 against 0.15 (refit, nats per token).
:::

::: {.slide title="Check: The Cost of a Generated Token"}
[Check]{.kicker}

Under the refit, the compute-optimal model with Llama 3 8B's predicted
loss has about 34B parameters and reads 0.64T tokens. Compare the training
costs and the cost of a generated token.

. . .

- Training: $6 \cdot 8\textrm{B} \cdot 15\textrm{T}$ against
  $6 \cdot 34\textrm{B} \cdot 0.64\textrm{T}$, **about 5.5×** more for the 8B
  model.
- Generation: $2N$ per token, **about 4×** cheaper with the 8B model.
- The savings exceed the extra training after about 11T generated tokens,
  fewer than the 15T the model read in training.
:::

::: {.slide title="A Scaling Law Predicts the Loss of Dense Models, Nothing More"}
- It predicts one number: the loss on text like the training data.
- Whether a model answers a question, writes a program or follows an
  instruction is measured separately; small differences in loss need
  not show up there in proportion.
- It was fitted to dense models. A mixture of experts trains on the
  parameters each token uses, while its capacity follows all of them.
:::

::: {.slide title="6ND Prices a Run, and About 20 Tokens per Parameter Spends It Best"}
[Wrap-up]{.kicker}

- Training costs $\approx 6ND$; attention adds $n/(6d)$, 8% for our
  models, confirmed by the framework counts.
- At equal compute, the smaller model trained on more tokens reached the
  lower loss, by 0.05 to 0.13 nats.
- Compute-optimal: $N^*, D^* \propto C^{1/2}$, about 20 tokens per
  parameter (refit).
- Deployed models train up to 100× longer per parameter to make
  generation cheaper.

**Next:** a mixture of experts, whose training cost follows only the
parameters each token uses.
:::
