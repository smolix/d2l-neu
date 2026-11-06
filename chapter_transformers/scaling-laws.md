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

The answer rests on two regularities that can be measured on runs a million
times smaller. The cost of a run follows from the shape of the model by
counting. And the loss falls smoothly with the number of parameters and the
amount of data, along a power law that can be fitted on small models and
extended to large ones. Such a fit is called a *scaling law*.

This section derives the cost, about $6ND$ operations for $N$ parameters
and $D$ training tokens, and checks it against the framework's own count.
It then trains fifteen small GPTs, five sizes at three data budgets, to see
how the loss depends on each. Finally it takes the law fitted by
:citet:`hoffmann2022training` to hundreds of larger runs, derives how a
fixed budget should be split between parameters and tokens, and compares
that split with what deployed models do. :numref:`sec_scaling` transferred
the learning rate from one model size to another; here the model size
itself is the variable.

```{.python .input #scaling-laws-scaling-laws-and-the-modern-recipe}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.flop_counter import FlopCounterMode
```

```{.python .input #scaling-laws-scaling-laws-and-the-modern-recipe}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import nnx
import jax
import math
import numpy as np
import optax
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
    print(f'embedding {emb / 1e6:6.2f}M ({emb / total:4.0%}),  '
          f'blocks {(total - emb) / 1e6:6.2f}M '
          f'= {(total - emb) / (L * d ** 2):.2f} L d^2,  '
          f'total {total / 1e6:.2f}M')

census(gpt(256))
census(gpt(50257, num_hiddens=768, num_heads=12, num_blks=12,
           pos='learned', norm='layer', act='gelu', bias=True))
```

Both rows confirm the $12Ld^2$ rule to a fraction of a percent; the residue
is normalization gains and, for GPT-2, biases. The two models differ in the
share of the embedding. It is 1.4% of our model with its 256 bytes, but
nearly a third of GPT-2, whose 124 million parameters are an
85-million-parameter transformer attached to lookup tables of 39 million entries, 38.6
million for the tokens and 0.8 million for the positions.
Comparing $Vd$ with $12Ld^2$ shows that the embedding dominates when $V
\gtrsim 12Ld$, that is, for small models with large vocabularies. The size
of the table follows the tokenizer, not the computation each token
receives, so :citet:`kaplan2020scaling` counted only *non-embedding*
parameters. The miniature study below does the same, so there $N$ means
the number of non-embedding parameters. The law of
:citet:`hoffmann2022training` that follows it counts all parameters.

### Six FLOPs per Parameter and Token

Multiplying a vector by a matrix $\mathbf{W} \in \mathbb{R}^{m \times n}$
takes $mn$ multiplications and $mn$ additions, 2 floating-point operations
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

FLOPs :cite:`kaplan2020scaling`, the estimate that :numref:`sec_gpt` used
to place its run. Generating a token without training needs only the
forward pass, about $2N$ FLOPs (:numref:`sec_kv-cache`).

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

We compare the estimate with the framework's own count of the operations in
one training step on a minibatch of 64 windows of 128 tokens. The model is
the default GPT with a GELU feed-forward network, which holds the same
$12d^2$ per block as SwiGLU in matrices of round sizes. The cell prints the
counted forward cost against $2ND$, the counted cost of the whole step
against $6ND$, and the ratio of backward to forward work. Here $N$ counts
all two-dimensional parameters, so that it includes the tied embedding in
its role as the output head.

:begin_tab:`pytorch`
`FlopCounterMode` intercepts every operator call and adds up the FLOPs of
matrix products; it counts nothing for elementwise operations. The fused
attention kernel that PyTorch picks on a CPU has no counting formula, so
the cell selects the math backend with `sdpa_kernel`. That backend writes
attention as two batched matrix products, which the counter sees, and it
counts the full $n \times n$ score matrix, the masked half included.
:end_tab:

:begin_tab:`jax`
XLA counts the operations of a compiled program without running it:
`cost_analysis()` of the lowered training step reports every operation of
the optimized computation, the attention scores over the full $n \times n$
matrix, softmax, and normalization included. We compile the step for the
CPU. On a GPU, XLA hands some operations to library kernels whose work it
may leave out of the count, and which kernels it picks can change from one
run to the next.
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

The counted cost lies 8 to 10% above the estimate, the size of the
$n/(6d)$ correction for the attention scores, and the backward pass costs
twice the forward pass, as the derivation says. A count of FLOPs also
converts into time. A run of $C$ FLOPs on hardware that sustains $R$ FLOPs
per second takes $C/R$ seconds, and the ratio of $R$ to the hardware's peak
is its *utilization*. Models as small as ours reach only a small fraction of
the peak of a GPU, because their matrices are too small to keep it busy;
:numref:`chap_performance` measures this fraction and raises it.

## A Miniature Scaling Study

Counting gives the cost of a run but not the loss it reaches. Across many
orders of magnitude, the loss of a language model falls as a power law in
the number of parameters, in the amount of data, and in compute, as long as
neither of the other two limits it :cite:`kaplan2020scaling`. On log--log
axes a power law $L = a N^{-\alpha}$ is a straight line with slope
$-\alpha$, so a line fitted to small models predicts the loss of larger
ones. We now measure this structure in miniature, and in particular the
point where the line stops: when does a larger model stop helping?

### Five Sizes and Three Data Budgets
:label:`subsec_scaling-laws-five-sizes`

To separate the effect of model size from that of data, we train a grid of
models that varies the two independently. The corpus is the one of
:numref:`sec_gpt`, the bytes of the Wikipedia articles of
`d2l.WikiText103`: about 270 million training bytes, read in windows of 128
at random offsets, and the official validation split of about 1.1 million
bytes.

```{.python .input #scaling-laws-a-corpus-bigger-than-a-novella}
%%tab pytorch, jax
data = d2l.WikiText103(batch_size=64, num_steps=128)
print(f'{len(data.train) / 1e6:.0f}M training bytes, '
      f'{len(data.val) / 1e6:.2f}M validation bytes, '
      f'vocabulary of {len(data.vocab)}')
```

The grid has two axes. Five sizes of `d2l.GPT` grow width and depth
together, as model families do: widths from 96 to 384 and depths from 3 to
8 blocks, or 0.33 to 14.2 million non-embedding parameters, a factor of
43. Each size trains at three data budgets: 500, 1,000, and 2,000 steps of
64 windows, that is $D$ = 4.1, 8.2, and 16.4 million tokens. Even the
largest budget reads about 6% of the training text, so every budget
consists of fresh text, and the grid varies the amount of data without
repeating any of it.

The training recipe is that of :numref:`sec_gpt` (`d2l.LMOptimizer`):
AdamW with weight decay 0.1 on the weight matrices, a linear warmup over
the first tenth of the steps, a cosine decay of the learning rate to zero,
and gradient clipping. Two of its settings depend on the run, and both
matter for a scaling study. First, every run has its own schedule, which
decays to zero at the end of that run's budget. Reading the loss after 500
steps off a run scheduled for 2,000 would measure a model whose learning
rate is still high, and :citet:`hoffmann2022training` found that this
choice made earlier fits overestimate the loss of short runs. Second, the
learning rate follows the width. :numref:`sec_scaling` showed that the
largest stable Adam learning rate shrinks in proportion to $1/d$, so a rate
tuned at one width undertrains or destabilizes the others (exercise
:exref:`ex_scaling-laws-frozen-learning-rate` measures the effect). We
anchor the peak rate at $2 \times 10^{-3}$ for width 256, the rate of
:numref:`sec_gpt`, and scale it as $256/d$. There is no dropout: no run
reads any passage twice. Each line of the output is one size,
with its validation losses at the three budgets and its training loss over
the last 50 steps of the longest run:

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
                                    warmup=num_steps // 10)
        losses = d2l.train_lm(model, data, optimizer, num_steps)
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

### Reading the Curves

The plot shows the validation loss against $N$ on log--log axes, one curve
per data budget. The dashed line is a power law fitted to the three
smallest models at the largest budget and extended over the whole range:
the prediction that a pure power law would make for the two larger models.

```{.python .input #scaling-laws-five-sizes-one-diet-2}
%%tab pytorch, jax
x = np.log(np.array(Ns))
slope, intercept = np.polyfit(x[:3], np.log(val[budgets[-1]][:3]), 1)
line = np.exp(intercept + slope * x)
d2l.plot(Ns, [val[s] for s in budgets] + [line],
         'non-embedding parameters N', 'validation loss (nats)',
         legend=[f'D = {s * 64 * 128 / 1e6:.1f}M' for s in budgets]
         + ['power law'], xscale='log', yscale='log',
         ylim=(1.1, 2.3), figsize=(4.5, 3.2),  # Room for the legend
         fmts=('C0o-', 'C1s-', 'C2^-', 'k--'))
print(f'slope {slope:.3f}; the largest model lies '
      f'{val[budgets[-1]][-1] - line[-1]:.3f} nats above the line')
```

At every budget the validation loss falls as the model grows, and more
data lowers every curve. The three smallest models at the largest budget
lie close to a straight line, but the two larger models fall short of
its extension: the curve flattens. At every budget, the largest model gains
only a few hundredths of a nat over the next one, although it has three
times the parameters. More parameters lower the loss only as long as there is data for them to
learn from. With the data fixed, the loss approaches a floor that more
parameters do not lower, and more data lowers the floor.

Two checks support this reading. Each point is a single run with fresh
random weights, and rerunning moves most points by a hundredth or two of
a nat, less than the gaps between the curves. And no run reads the same text
twice, so a larger budget means more distinct text, not more passes over
the same text. The training loss of the longest runs stays within a few
hundredths of the validation loss, so the models do not memorize.
Repetition would change these results only slowly:
:citet:`Muennighoff.Rush.Barak.ea.2023` trained models of up to 9 billion
parameters on repeated data and found that up to about four passes give
almost the same loss as fresh data, with each further pass worth less and
less.

## Compute-Optimal Allocation

### The Chinchilla Law

Our miniature study shows the shape of the loss, but over too small a range
to plan a large run. That needs a law fitted to runs at scale.
:citet:`hoffmann2022training` trained over 400 models, from 70 million to
over 16 billion parameters on 5 to 500 billion tokens, and fitted their
final losses with

$$
L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}.
$$
:eqlabel:`eq_chinchilla_law`

The three terms separate what limits the loss. $E$ is the loss of an ideal
model on the training distribution, the entropy of the text that no budget
removes. $A/N^{\alpha}$ is the excess loss of a model with $N$ parameters
trained on unlimited data, and $B/D^{\beta}$ the excess loss of an
unlimited model trained on $D$ tokens. Here $N$ counts all parameters,
embeddings included. The choice matters at small scale, where the
embedding is a large share of a model, because it changes the fitted
exponent: :citet:`Pearce.Song.2024` trace to it most of the reason why the
earlier fit of :citet:`kaplan2020scaling` favored larger models (exercise
:exref:`ex_scaling-laws-embedding-count` measures the effect on a slope).

In our grid the two counts differ little: the table of 256 bytes adds 7%
to the smallest model and under 1% to the largest, which changes the slope
on log--log axes by about 2%, so the law can be compared with the grid
almost directly.
With $D$ fixed, $E + B/D^{\beta}$ is a constant, and only $A/N^{\alpha}$
falls as the model grows. A constant plus a power law is straight on
log--log axes only while the power law dominates; once $A/N^{\alpha}$ has
shrunk to the size of the constant, the curve flattens toward
$E + B/D^{\beta}$. A smaller $D$ raises that constant, so the curve
flattens earlier. The law counts $D$ repeated tokens as $D$ fresh ones.
Our grid never tests this assumption, which holds for a few passes but not
for many more.

### The Optimal Split

A fixed budget $C \approx 6ND$ allows a large model trained briefly or a
small model trained long. To find the best split, substitute $D = C/(6N)$
into :eqref:`eq_chinchilla_law` and set the derivative with respect to $N$
to zero. With $\partial D / \partial N = -D/N$, the optimum satisfies

$$
\alpha \frac{A}{N^{\alpha}} = \beta \frac{B}{D^{\beta}}.
$$
:eqlabel:`eq_chinchilla_optimum`

At the optimum the two terms that the budget can shrink fall at equal
marginal rates, so moving FLOPs from parameters to data, or back, gains
nothing. Solving :eqref:`eq_chinchilla_optimum` together with $ND = C/6$
gives

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

![Loss contours of the Chinchilla law with the refit constants of Besiroglu et al. (2024), over parameters and training tokens. Dashed: budgets of equal compute $C = 6ND$; dots: the compute-optimal model of each budget; orange: the compute-optimal path. Chinchilla lies on the path; Llama 3 8B, on a budget about a quarter larger, lies far above it.](../img/mdl-transformers-chinchilla.svg)
:label:`fig_mdl-transformers-chinchilla`

The constants come from fits, and the fits disagree.
:citet:`hoffmann2022training` report $E = 1.69$, $A = 406.4$, $B = 410.7$,
$\alpha = 0.34$, $\beta = 0.28$. :citet:`Besiroglu.Erdil.Barnett.ea.2024`
reconstructed the data from the paper's figures, fitted the same form
again, and obtained $E = 1.82$, $A = 482.0$, $B = 2085.4$, $\alpha =
0.348$, $\beta = 0.366$. They found the original constants inconsistent
with the paper's two other estimates of the optimum and their reported
confidence intervals far too narrow for the number of runs. The cell
evaluates :eqref:`eq_chinchilla_allocation` for both fits at Chinchilla's
own budget, $5.76 \times 10^{23}$ FLOPs, at which Chinchilla trained 70
billion parameters on 1.4 trillion tokens, 20 tokens per parameter:

```{.python .input #scaling-laws-the-optimal-split}
%%tab pytorch, jax
fits = {'Hoffmann et al.': (1.69, 406.4, 410.7, 0.34, 0.28),
        'Besiroglu et al.': (1.82, 482.0, 2085.4, 0.348, 0.366)}

def optimum(C, E, A, B, alpha, beta):
    G = (alpha * A / (beta * B)) ** (1 / (alpha + beta))
    N = G * (C / 6) ** (beta / (alpha + beta))
    return N, C / (6 * N)

for name, fit in fits.items():
    N, D = optimum(5.76e23, *fit)
    print(f'{name:17s} N* = {N / 1e9:3.0f}B, D* = {D / 1e12:.2f}T, '
          f'{D / N:3.0f} tokens per parameter')
```

Only the refit reproduces the allocation Chinchilla was trained with. The
published constants put the optimum at less than half the model size and
twice the data, about 90 tokens per parameter. Under the refit, the two
exponents are nearly equal, so $N^*$ and $D^*$ both grow almost exactly
like $C^{1/2}$ and the optimum stays near 20 tokens per parameter over many
orders of magnitude of compute. This ratio belongs to that study's data,
tokenizer, and model family; a different corpus or architecture moves it.

### Training Past the Optimum

Compute-optimal means the lowest loss for a given training budget. A
deployed model also costs about $2N$ FLOPs for every token it generates,
so a smaller model trained on more tokens can be cheaper over its lifetime,
although its training reaches the same loss less efficiently
:cite:`Sardana.Portes.Doubov.ea.2024`. The cell applies the refit to dense
models with published sizes and token counts, as if they had been trained
on Chinchilla's data. For each it prints the tokens per parameter, the
model's size relative to the compute-optimal model of the same training
budget, and the extra loss that the law predicts for the difference:

```{.python .input #scaling-laws-training-past-the-optimum}
%%tab pytorch, jax
E, A, B, alpha, beta = fits['Besiroglu et al.']
loss = lambda N, D: E + A / N ** alpha + B / D ** beta
reported = {'Chinchilla': (70e9, 1.4e12), 'Llama 3 8B': (8e9, 15e12),
            'Llama 3 405B': (405e9, 15.6e12), 'OLMo 2 7B': (7e9, 4.05e12),
            'Gemma 3 27B': (27e9, 14e12)}
for name, (N, D) in reported.items():
    N_opt, D_opt = optimum(6 * N * D, E, A, B, alpha, beta)
    print(f'{name:12s} {D / N:5.0f} tokens per parameter, '
          f'{N / N_opt:4.2f} x optimal size, '
          f'loss +{loss(N, D) - loss(N_opt, D_opt):.3f}')
```

The largest Llama 3 model sits close to the optimum; Meta's own fit on its
data put the optimum for that budget at 402 billion parameters and 16.55
trillion tokens :cite:`Grattafiori.Dubey.Jauhri.ea.2024`. The smaller
models are trained far past it, by design. Llama 3 8B reads almost 1,900
tokens per parameter. The law puts it at a tenth of the compute-optimal
size and predicts about 0.05 nats of extra loss. Each token it generates,
however, takes a tenth of the computation that the optimal model would
need. For a model that will serve trillions of tokens, the smaller model is
the better choice. Choosing such
budgets, and the further cost of running out of fresh data, are topics of
the Language Models part (:numref:`chap_nlp_pretrain`). A mixture of
experts (:numref:`sec_moe`) does not fit this accounting: its training
cost follows its active parameters, its capacity all of them, and the law
above was fitted to dense models only.

## Configurations of Deployed Models

The models trained on such budgets are built from the components of this
chapter. :numref:`tab_modern-recipe` compiles the architecture sections of
seven open-weights families: Mistral 7B
:cite:`Jiang.Sablayrolles.Mensch.ea.2023`, Llama 3
:cite:`Grattafiori.Dubey.Jauhri.ea.2024`, Qwen3
:cite:`Yang.Li.Yang.ea.2025`, OLMo 2 and 3 :cite:`OLMo.2025,OLMo3.2025`,
DeepSeek-V3 :cite:`Liu.Feng.Xue.ea.2024`, Gemma 3
:cite:`Gemma.Team.2025`, and GPT-OSS :cite:`OpenAI.2025`.

:Architectures of seven open-weights model families, as reported in their technical reports (2023–2025). Attention and cache: :numref:`sec_kv-cache`. Normalization: :numref:`sec_transformer-block`. Positions: :numref:`sec_positional-information`. Mixture of experts: :numref:`sec_moe`.
:label:`tab_modern-recipe`

| model | attention and cache | normalization | positions | FFN | dense or MoE |
|:--|:--|:--|:--|:--|:--|
| Mistral 7B (2023) | GQA $32{:}8$; sliding window 4096 | RMSNorm, pre | RoPE, $\theta = 10^4$ | SwiGLU | dense |
| Llama 3 (2024) | GQA $32{:}8$ (8B), $64{:}8$ (70B), $128{:}8$ (405B) | RMSNorm, pre | RoPE, $\theta = 5 \times 10^5$ | SwiGLU | dense |
| Qwen3 (2025) | GQA: $64{:}8$ dense (32B), $64{:}4$ MoE (235B-A22B) | RMSNorm, pre; QK-norm | RoPE, $\theta = 10^6$; YaRN for long context | SwiGLU | dense, and MoE: 128 experts, 8 active |
| OLMo 2/3 (2024/25) | MHA (7B), GQA (32B); OLMo 3: window 4096 on 3 of 4 layers | RMSNorm on sublayer outputs; QK-norm | RoPE | SwiGLU | dense |
| DeepSeek-V3 (2024) | MLA: KV compressed to a 512-dim latent | RMSNorm, pre | RoPE on a decoupled 64-dim slice per head | SwiGLU | MoE: 256 experts plus 1 shared, 8 active |
| Gemma 3 (2025) | GQA; local:global $5{:}1$, window 1024 | RMSNorm, pre and post; QK-norm | RoPE, $\theta = 10^6$ global, $10^4$ local | GeGLU | dense |
| GPT-OSS (2025) | GQA $64{:}8$; window 128 on alternate layers; learned sinks | RMSNorm, pre | RoPE; YaRN | SwiGLU, clamped | MoE: 128 or 32 experts, 4 active |

Every row is the block of :numref:`sec_transformer-block` in the causal
wiring of :numref:`sec_gpt`, and the rows agree on most of its settings: a
gated feed-forward network, RoPE, RMSNorm with an unnormalized residual
stream, no biases in most rows, and no dropout in any. (OLMo normalizes each
sublayer's output before it joins the stream rather than its input, which
also leaves the stream itself unnormalized.) The absence of dropout is a
consequence of scale: a model that reads trillions of tokens about once has nothing to
memorize. The rows differ on two axes. The first is the size of the KV
cache: grouped-query attention in most rows, sliding windows on some
layers, and a compressed latent in DeepSeek-V3, all from
:numref:`sec_kv-cache`.
The second is capacity per FLOP: three of the seven families use a mixture
of experts (:numref:`sec_moe`). The newest designs depart further. Hybrid
models replace most attention layers with the linear recurrences of
:numref:`chap_modern_rnn` :cite:`Kimi.Team.2025b`, and Llama 4 removes
positional encoding from some of its attention layers :cite:`Meta.2025`.

At width 256 and six blocks, three rows of the table become calls of our
constructor. The modern rows need only the number of key--value heads,
because the defaults of `d2l.GPT` are the recipe the table shares. Llama 3
8B shares each key--value head among 4 query heads, Qwen3-32B among 8. The
cell prints the parameter count and the bytes of KV cache per token in
32-bit floats:

```{.python .input #scaling-laws-recipe-rows-as-constructor-calls}
%%tab pytorch, jax
recipes = {'GPT-2': dict(pos='learned', norm='layer', act='gelu',
                         bias=True),
           'Llama 3 8B': dict(num_kv_heads=2),
           'Qwen3-32B': dict(num_kv_heads=1)}
for name, cfg in recipes.items():
    model = gpt(1024, num_hiddens=256, num_heads=8, num_blks=6, **cfg)
    d2l.check_shape(model(d2l.zeros((2, 16), dtype=d2l.int32)),
                    (2, 16, 1024))
    print(f'{name:10s} {d2l.num_params(model) / 1e6:.2f}M parameters, '
          f'{d2l.kv_cache_bytes(model, 1) / 1024:4.1f} KiB of cache '
          'per token')
```

Sharing key--value heads removes part of the key and value projections, a
tenth of the parameters or so at this width, but it shrinks the cache by
the full factor of the sharing, 4 and 8. The settings the flags do not
reach would enter as new attention modules. QK-norm
(:numref:`subsec_qk-norm`) enters through the `attn_factory` hook of
`TransformerBlock`, as in an exercise of :numref:`sec_transformer-block`.
A sliding window is a mask (:numref:`sec_attention-at-scale`,
:numref:`sec_kv-cache`). A learned latent cache replaces the stored
keys and values, which :numref:`sec_kv-cache` sketches with a fixed basis.
The expert layers of the MoE rows enter through `ffn_factory`
(:numref:`sec_moe`). Deployed models are thus
configurations of the chapter's one block, trained at budgets that the law
of this section helps to plan.

## Summary

Training a dense transformer with $N$ parameters on $D$ tokens costs
about $6ND$ FLOPs, two per parameter and token in the forward pass and
four in the backward pass; attention scores add a fraction
$n/(6d)$ for contexts of $n$ tokens, which the framework counts confirm.
In a grid of small GPTs on a fixed corpus, the loss falls as a power law in
$N$ until it approaches a floor that only more data lowers. The law
$L = E + A/N^{\alpha} + B/D^{\beta}$ explains this floor and, under a
budget $C \approx 6ND$, gives the compute-optimal split: parameters and
tokens both grow about like $C^{1/2}$, at about 20 tokens per parameter in
the refit of the Chinchilla data. Smaller deployed models are trained far
past this point to make generation cheaper, and all of them share one
block whose settings differ mainly in the KV cache and in the use of
experts.

## Exercises

1. **Attention-score correction.** :eqref:`eq_six_nd` omits the
   attention-score and value-mixing work, about $4nd$ FLOPs per token and
   block in the forward pass for a context of $n$ tokens, against about
   $24d^2$ for the block's linear maps.
    1. For the minibatch of :numref:`subsec_scaling-laws-checking`
       ($n = 128$, $d = 256$, six blocks, eight heads, $64 \times 128$
       tokens), compute the attention term and add it to $2ND$. Compare
       the sum with the forward count that your framework printed. If a
       remainder is left, express it in FLOPs per token and block. Then
       estimate the elementwise work of one block (the RoPE rotation of
       queries and keys, the attention softmax, the GELU of the $4d$-wide
       hidden layer, two normalizations, and two residual additions) at a
       few FLOPs per element, and check whether it accounts for the
       remainder to within a factor of two.
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

1. **Pricing a mixture-of-experts run.** DeepSeek-V3
   :cite:`Liu.Feng.Xue.ea.2024` has 671B parameters, of which 37B are
   active for each token, and was pretrained on 14.8T tokens in 2.664M
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
    1. Rerun one model of the original sweep twice to measure the
       run-to-run variation. By how much does the frozen rate change the
       validation loss of each model, compared with this variation? Which
       part of the shape of the frozen-rate curve is a tuning artifact?
   :exlabel:`ex_scaling-laws-frozen-learning-rate`

1. [code] **A sixth size.** ● A sixth model for the sweep of
   :numref:`subsec_scaling-laws-five-sizes` has width 512, ten blocks, and
   eight heads, about 31.5M non-embedding parameters.
    1. Before training it, predict its validation loss at the largest
       budget twice: by extending the dashed power law, and by assuming
       that the corpus allows no improvement over the largest measured
       model. Estimate the run's cost with :eqref:`eq_six_nd`.
    1. Train it with the same recipe at all three budgets. Which
       prediction was closer, and what do the three results say about the
       headroom that a budget of 16.4M tokens leaves?

1. [code] **Fitting the law to the grid.** ● Fit
   :eqref:`eq_chinchilla_law` to the fifteen validation losses of the
   sweep, with $D$ in tokens, by minimizing the squared error of
   $\log L$ over $E$, $\log A$, $\log B$, $\alpha$, and $\beta$ with a
   gradient-based optimizer from several starting points.
    1. Report the fitted constants and the largest residual. Is the fitted
       $E$ below every measured loss?
    1. Refit fifteen times, each time leaving one point out, and report the
       range of $\alpha$, $\beta$, and $E$. Which constant is determined
       worst, and why?
    1. The law predicts that the loss difference between two budgets is
       the same for every model size. Check this against the grid. Which
       term of :eqref:`eq_chinchilla_law` would have to depend on $N$ to
       describe what you see?

1. **How the ratio moves with the budget.** ● The compute-optimal
   allocation is given by :eqref:`eq_chinchilla_allocation`.
    1. Show that $D^*/N^* = G^{-2} (C/6)^{(\alpha - \beta)/(\alpha +
       \beta)}$.
    1. Evaluate $D^*/N^*$ at $C = 10^{21}$ and $C = 10^{25}$ FLOPs for both
       fits in `fits`. In which direction does the ratio move with the
       budget under each fit, and by how much?
    1. Both fits come from the same experiments. What do the results say
       about how precisely a fit of :eqref:`eq_chinchilla_law` determines
       the token--parameter ratio at budgets a hundred times beyond the
       largest run that was fitted?

1. **Training for inference.** ● A model that serves $D_{\textrm{inf}}$
   tokens after training costs about

   $$C_{\textrm{life}} = 6ND + 2N D_{\textrm{inf}}$$

   FLOPs over its lifetime, since generating a token costs
   about $2N$ (:numref:`sec_kv-cache`). :citet:`Sardana.Portes.Doubov.ea.2024`
   minimize this cost at a fixed target loss instead of minimizing the
   loss at a fixed training budget.
    1. Minimize $C_{\textrm{life}}$ subject to $L(N, D) = \ell$
       under :eqref:`eq_chinchilla_law`. Derive the optimality condition
       that replaces :eqref:`eq_chinchilla_optimum`, and show that the
       optimum moves to smaller $N$ and larger $D$ as $D_{\textrm{inf}}$
       grows.
    1. Llama 3 8B was pretrained on about 15T tokens. Take the refit of
       :citet:`Besiroglu.Erdil.Barnett.ea.2024` at face value and find the
       compute-optimal model with the same predicted loss as $N = 8 \times
       10^9$, $D = 1.5 \times 10^{13}$. Compute the ratio of the two
       training costs and the number of inference tokens at which the two
       lifetime costs are equal.

1. [extended] **IsoFLOP profiles.** ● The sweep of
   :numref:`subsec_scaling-laws-five-sizes` fixes three data budgets, so
   each model receives a different compute budget.
   :citet:`hoffmann2022training` instead fix the budget $C$ and trade $N$
   against $D = C/(6N)$.
    1. For each budget
       $C \in \{3 \times 10^{13}, 10^{14}, 3 \times 10^{14}\}$ FLOPs and
       each model in `sizes`, train for
       $C / (6N \cdot 64 \cdot 128)$ steps, rounded, with the recipe of
       the sweep, and record the validation loss. Plot validation loss
       against $N$, one curve per budget.
    1. Take the lowest-loss model of each budget as $N_{\textrm{opt}}(C)$,
       and refine it by fitting a parabola in $\log N$ to the points
       around the minimum. Fit $N_{\textrm{opt}} \propto C^{a}$ and
       compare $a$ with $\beta / (\alpha + \beta)$ for both fits in
       `fits`.
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
[Dive into Deep Learning · §12.7]{.kicker}

Scaling laws<br>
**the cost of a run · a miniature scaling study · the compute-optimal split · deployed configurations**
:::
:::

::: {.slide title="A 3.8×10²⁵-FLOP Run Must Be Planned from Runs a Million Times Smaller"}
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
- read a grid of small runs: where a power law holds, and where it bends;
- derive the compute-optimal split of a budget between parameters and
  tokens;
- explain why smaller deployed models train far past 20 tokens per
  parameter.

**Running example:** GPT-2 small (124M parameters), and five small GPTs
on the bytes of WikiText-103.
:::

::: {.slide title="Cost and Loss Are Predictable from the Model's Shape"}
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
| Mixture of Experts | more parameters per FLOP |
| **Scaling Laws** | **loss against parameters, data, compute** |
:::

::: {.col .narrow}
- **Uses:** `d2l.GPT`, the training recipe of the GPT section, learning
  rate $\propto 1/\textrm{width}$.
- **Skips:** data curation, budgets beyond the compute optimum
  (Language Models part).
:::
:::
:::

::: {.slide title="GPT-2's 124M Parameters Are an 85M Transformer and a 38.6M Table"}
Blocks hold $12Ld^2$ ($4d^2$ attention, $8d^2$ FFN); the embedding $Vd$:

@!scaling-laws-a-parameter-census

- The embedding dominates when $V \gtrsim 12Ld$: small models, large
  vocabularies.
- Kaplan-style studies, and our grid, count **non-embedding** $N$;
  Chinchilla counts all parameters.
:::

::: {.slide title="Training Costs Six FLOPs per Parameter and Token"}
$$C \approx \underbrace{2ND}_{\textrm{forward}} + \underbrace{4ND}_{\textrm{backward}} = 6ND$$

- Forward: each weight meets each token once, one multiply and one add.
- Backward: two products per linear layer of the same size,
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

::: {.slide title="The Framework's Count Lands 8–10% Above 6ND"}
@!scaling-laws-checking-the-arithmetic

- The excess is the attention term $n/(6d) \approx 8\%$.
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
| $D$ (tokens) | 4.1M, 8.2M, 16.4M: at most 6% of the text, none repeated |

All 15 runs: AdamW, decay 0.1 on matrices, warmup, cosine, clipping;
no dropout.
:::

::: {.slide title="Every Run Gets Its Own Schedule and Its Own Learning Rate"}
- **Schedule:** the cosine decays to zero at the end of *each* run.
  A 500-step loss read off a 2,000-step run is taken at a high learning
  rate and overstates the loss (Hoffmann et al., 2022).
- **Learning rate:** the stable Adam step shrinks like $1/d$, so the peak
  rate is $2 \times 10^{-3} \cdot 256/d$.

::: {.d2l-note}
A study is only as good as the tuning of its worst point.
:::
:::

::: {.slide title="Fifteen Runs, One Line per Size"}
@!scaling-laws-five-sizes-one-diet-1

- Read down a column: $N$ grows. Read along a line: $D$ grows.
:::

::: {.slide title="With the Data Fixed, Loss Flattens Toward a Floor"}
@!scaling-laws-five-sizes-one-diet-2

- Small models follow a power law (dashed); larger ones fall short.
- At every budget the largest model gains only a few hundredths over
  the next. More data lowers the floor.
:::

::: {.slide title="The Chinchilla Law Separates Three Limits"}
$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}$$

- $E$: the entropy of the text, which no budget removes.
- $A/N^{\alpha}$: the excess loss of finite parameters.
- $B/D^{\beta}$: the excess loss of finite data.
- $D$ fixed: $E + B/D^{\beta}$ is the floor of the grid's curves.

[Fitted to 400+ runs, 70M–16B parameters, 5–500B tokens (Hoffmann et al., 2022); $N$ counts all parameters.]{.small}
:::

::: {.slide title="At the Optimum, Both Shrinkable Terms Fall at Equal Rates"}
Under $C \approx 6ND$, substitute $D = C/(6N)$ and set $\partial L/\partial N = 0$:

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
- Chinchilla itself: 70B parameters on 1.4T tokens.
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

- Generating a token costs $2N$: a 10× smaller model is 10× cheaper to
  serve, for life.
- The refit predicts about 0.05 nats of extra loss for Llama 3 8B
  (Sardana et al., 2024).
:::

::: {.slide title="Open Models Share One Block and Differ in the Cache and the FFN"}
| model | attention + cache | norm | FFN / experts |
|:--|:--|:--|:--|
| Mistral 7B | GQA 32:8, window 4096 | RMS pre | SwiGLU, dense |
| Llama 3 8B | GQA 32:8 | RMS pre | SwiGLU, dense |
| Qwen3 | GQA 64:8 dense, 64:4 MoE | RMS pre + QK | SwiGLU, dense & MoE 128/8 |
| DeepSeek-V3 | MLA: 512-d latent | RMS pre | SwiGLU, MoE 256+1/8 |
| Gemma 3 | GQA, local:global 5:1 | RMS pre+post + QK | GeGLU, dense |
| GPT-OSS | GQA 64:8, window 128 alt., sinks | RMS pre | SwiGLU, MoE 128/4 |

All: RoPE, **no dropout**: a model that reads its tokens about once has
nothing to memorize.
:::

::: {.slide title="At Equal Width, the Rows Differ by Their Cache"}
@!scaling-laws-recipe-rows-as-constructor-calls

- The modern rows need one flag: the defaults of `d2l.GPT` are the shared
  recipe.
- Key--value sharing shrinks the cache 4× and 8×, the parameters by about
  a tenth.
:::

::: {.slide title="Check: The Cost of a Generated Token"}
[Check]{.kicker}

For its budget, the compute-optimal model of Llama 3 8B has about 81B
parameters. How much more does each generated token cost with it?

. . .

- $2N$ per token: $81/8 \approx$ **10×** more.
- The 8B model is 10× cheaper on every served token, for 0.05 nats of
  loss.
:::

::: {.slide title="Wrap-up"}
[Wrap-up]{.kicker}

- Training costs $\approx 6ND$; attention adds $n/(6d)$, 8% for our
  models, confirmed by the framework counts.
- With $D$ fixed, loss follows a power law in $N$ and then flattens toward
  $E + B/D^{\beta}$.
- Compute-optimal: $N^*, D^* \propto C^{1/2}$, about 20 tokens per
  parameter (refit).
- Deployed models train up to 100× longer per parameter to make
  generation cheaper; they share one block and differ in the cache and the FFN.

**Next:** budgets beyond the compute optimum, data, and pretraining at
scale, in the Language Models part.
:::
