# Scaling Laws and the Modern Recipe
:label:`sec_scaling-laws`

The models of this chapter differ from the deployed ones mainly by a
number. Same block, same causal wiring, same training loop; what separates
our minute of GPU time from a frontier run is parameters, tokens, and the
floating-point operations that couple them. This closing section is about
that number. One boundary first: :numref:`sec_scaling` in the optimization
chapter asked how to *tune* across scale — muP transfers a learning rate
found on a small model to a large one. Here we ask what scale itself buys,
and how a training budget should be split between model size and data. We
learn to count parameters and FLOPs, deriving the $6ND$ rule and checking
it against a profiler; we run a scaling study small enough for one GPU
that shows both the celebrated straight line and its less celebrated end;
and we read the 2023–2025 open-weights reports as a table
whose rows are, almost verbatim, constructor calls of the `GPT` class from
:numref:`sec_gpt`.

```{.python .input #scaling-laws-scaling-laws-and-the-modern-recipe}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import time
import torch
from torch.nn import functional as F
from torch.profiler import ProfilerActivity, profile
```

```{.python .input #scaling-laws-scaling-laws-and-the-modern-recipe}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import nnx
import jax
from jax import numpy as jnp
import optax
```

## Counting Parameters and FLOPs

### A Parameter Census

A GPT holds two kinds of parameters, and they scale differently. The
blocks hold $12d^2$ each — $4d^2$ of attention, $8d^2$ of feed-forward
network, however the activation slices it
(:numref:`sec_transformer-block`) — for about $12Ld^2$ across $L$ blocks.
The embedding holds $Vd$ for a vocabulary of size $V$, used twice thanks
to weight tying but stored once; a learned position table adds another
$\textrm{max\_len} \times d$. The census below runs the accounting on our
default configuration and on the GPT-2 configuration from
:numref:`sec_gpt`:

```{.python .input #scaling-laws-a-parameter-census}
%%tab pytorch
count = lambda m: sum(p.numel() for p in m.parameters())

def census(model, d, num_blks):
    emb = count(model.token_emb) + (
        count(model.pos_emb) if model.pos == 'learned' else 0)
    total = count(model)
    print(f'embedding {emb / 1e6:6.2f}M ({emb / total:4.0%}),  '
          f'blocks {(total - emb) / 1e6:6.2f}M '
          f'= {(total - emb) / (num_blks * d ** 2):.2f} L d^2,  '
          f'total {total / 1e6:.2f}M')

census(d2l.GPT(vocab_size=28), d=256, num_blks=6)
census(d2l.GPT(vocab_size=50257, num_hiddens=768, num_heads=12,
               num_blks=12, pos='learned', norm='layer', act='gelu',
               bias=True), d=768, num_blks=12)
```

```{.python .input #scaling-laws-a-parameter-census}
%%tab jax
count = lambda m: sum(
    p.size for p in jax.tree.leaves(nnx.state(m, nnx.Param)))

def census(model, d, num_blks):
    emb = count(model.token_emb) + (
        count(model.pos_emb) if model.pos == 'learned' else 0)
    total = count(model)
    print(f'embedding {emb / 1e6:6.2f}M ({emb / total:4.0%}),  '
          f'blocks {(total - emb) / 1e6:6.2f}M '
          f'= {(total - emb) / (num_blks * d ** 2):.2f} L d^2,  '
          f'total {total / 1e6:.2f}M')

census(d2l.GPT(vocab_size=28), d=256, num_blks=6)
census(d2l.GPT(vocab_size=50257, num_hiddens=768, num_heads=12,
               num_blks=12, pos='learned', norm='layer', act='gelu',
               bias=True), d=768, num_blks=12)
```

Both rows confirm the $12Ld^2$ rule to a fraction of a percent (the
residue is normalization weights and, for GPT-2, biases). The interesting
column is the embedding share: negligible for our character model with its
28 symbols, but nearly a third of GPT-2. Comparing $Vd$ against $12Ld^2$
says the embedding dominates when $V \gtrsim 12Ld$ — small models with
real vocabularies. GPT-2's "124M parameters" is in this sense a generous
description of an 85M-parameter transformer attached to a 39M-parameter
lookup table. Scaling-law papers accordingly count *non-embedding*
parameters :cite:`kaplan2020scaling`: the embedding's size tracks the
tokenizer's vocabulary, not the computation each token receives, and
including it bends the low-end of every curve for accounting reasons
rather than scientific ones. We adopt the same convention; $N$ below means
non-embedding parameters.

### Six FLOPs per Parameter and Token

Training cost has an equally compact rule. Multiplying one token's
activation vector by a weight matrix $\mathbf{W} \in \mathbb{R}^{m \times
n}$ takes $mn$ multiply–add pairs: 2 FLOPs per parameter, so a forward
pass costs about $2N$ per token. The backward pass costs twice the
forward: each linear layer must produce *two* gradients, one with respect
to its input ($\mathbf{W}^\top \boldsymbol{\delta}$, to keep the chain
rule moving) and one with respect to its weights ($\boldsymbol{\delta}
\mathbf{x}^\top$, to learn), and each is a matrix multiply of the same
size as the forward one. Training on $D$ tokens therefore costs

$$
C \approx \underbrace{2ND}_{\textrm{forward}} +
\underbrace{4ND}_{\textrm{backward}} = 6ND
$$
:eqlabel:`eq_six_nd`

floating-point operations :cite:`kaplan2020scaling`. This is training
arithmetic; the corresponding decode-time accounting — why *generating* a
token costs about $2N$ and why the KV cache turns generation into a
memory problem rather than a compute one — was the business of
:numref:`sec_kv-cache`.

What the rule rounds away is worth knowing. The attention scores
$\mathbf{q}^\top \mathbf{k}$ and the mixture $\mathbf{A}\mathbf{V}$ cost
about $4nd$ per token per layer against the linear layers' $24d^2$, a
correction of $n/(6d)$ — about 8% at our context 128 and width 256,
growing to parity only at contexts around $6d$. Normalization, softmax,
and the optimizer update are all vector work, $\mathcal{O}(d)$ or
$\mathcal{O}(N)$ per step rather than per token: noise. A useful habit
follows: the cost of any dense transformer run is two numbers multiplied,
which is how our one-minute run in :numref:`sec_gpt` could be placed at
$5 \times 10^{14}$ FLOPs, the 124M GPT-2's near $7 \times 10^{18}$, and
frontier runs near $10^{25}$, all without consulting a profiler.

### Checking the Arithmetic

A formula this convenient deserves a check against an authority that has
not read our derivation.

:begin_tab:`pytorch`
PyTorch's profiler attributes FLOPs to the operations it recognizes —
above all the `aten::mm` matrix multiplies behind every linear layer. The
fused attention kernel keeps its score arithmetic to itself and reports
none, so the profiler and :eqref:`eq_six_nd` ignore the *same* subleading
term. For the count, $N$ plus the tied output head is exactly the 2-D
parameters of the RoPE configuration (the embedding earns its FLOPs as
the head; a gather costs nothing).
:end_tab:

:begin_tab:`jax`
XLA can price a program without running it: lowering the jitted training
step and asking the compiled artifact for its `cost_analysis()` returns
an exact operation count of the optimized computation, attention scores
and softmax included. One quirk dictates our flags: with `act='swiglu'`
the matched-budget width $\mathrm{round}(8d/3) = 683$ sends one FFN
matrix multiply down a code path whose work the analyzer does not count,
underreporting the FFN by a third. The GELU configuration keeps every
matrix a round size and is counted exactly, at the same parameter budget,
so that is the one we lower. For the count, $N$ plus the tied output head
is exactly the 2-D parameters of the model.
:end_tab:

```{.python .input #scaling-laws-checking-the-arithmetic}
%%tab pytorch
device = d2l.try_gpu()
model = d2l.GPT(vocab_size=28).to(device)
X = torch.randint(0, 28, (64, 128), device=device)
Y = torch.randint(0, 28, (64, 128), device=device)
N = sum(p.numel() for p in model.parameters() if p.ndim == 2)

def train_step():
    loss = F.cross_entropy(model(X).flatten(0, 1), Y.flatten())
    model.zero_grad()
    loss.backward()

for _ in range(3):
    train_step()                    # warm up
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             with_flops=True) as prof:
    train_step()
measured = sum(e.flops for e in prof.key_averages() if e.flops)
print(f'6ND analytic: {6 * N * X.numel():.3e} FLOPs')
print(f'profiler:     {measured:.3e} FLOPs '
      f'(ratio {measured / (6 * N * X.numel()):.3f})')
if device.type == 'cuda':
    torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    train_step()
if device.type == 'cuda':
    torch.cuda.synchronize()
dt = (time.time() - t0) / 20
print(f'{dt * 1e3:.0f} ms per step: '
      f'{6 * N * X.numel() / dt / 1e12:.1f} TFLOP/s achieved')
```

```{.python .input #scaling-laws-checking-the-arithmetic}
%%tab jax
model = d2l.GPT(vocab_size=28, act='gelu', rngs=nnx.Rngs(0))
key_x, key_y = jax.random.split(jax.random.key(0))
X = jax.random.randint(key_x, (64, 128), 0, 28)
Y = jax.random.randint(key_y, (64, 128), 0, 28)
graphdef, params, rest = nnx.split(model, nnx.Param, ...)
N = sum(p.size for p in jax.tree.leaves(params) if p.ndim == 2)

def loss_fn(params, X, Y):
    logits = nnx.merge(graphdef, params, rest)(X)
    return optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(-1, logits.shape[-1]), Y.reshape(-1)).mean()

fwd = jax.jit(loss_fn).lower(params, X, Y).compile().cost_analysis()
both = jax.jit(jax.value_and_grad(loss_fn)).lower(
    params, X, Y).compile().cost_analysis()
print(f'2ND analytic: {2 * N * X.size:.3e} FLOPs, '
      f'XLA forward: {fwd["flops"]:.3e} '
      f'(ratio {fwd["flops"] / (2 * N * X.size):.2f})')
print(f'6ND analytic: {6 * N * X.size:.3e} FLOPs, '
      f'XLA step:    {both["flops"]:.3e} '
      f'(ratio {both["flops"] / (6 * N * X.size):.2f})')
print(f'backward / forward = '
      f'{(both["flops"] - fwd["flops"]) / fwd["flops"]:.2f}')
```

Both authorities side with the formula. The PyTorch profiler's count
agrees with $6ND$ to a fraction of a percent — matching so tightly only
because both books omit the fused attention kernel's score work. XLA's
static analysis, which counts everything, lands about ten percent above
$6ND$, right where the $n/(6d)$ correction plus softmax and normalization
put it, and splits forward to backward at almost exactly one to two: the
derivation's 2-plus-4, read off a compiler's ledger. The timing line adds
a sobering coda: our achieved throughput is roughly an order of magnitude
below what the GPU can deliver, because a model this small cannot keep
the machine fed. Production runs choose widths and batch sizes with the
hardware's appetite in mind, which is one reason real models are as large
as they are.

## A Miniature Scaling Study

Counting tells us what a run costs; it says nothing about what the cost
buys. The empirical regularity that changed how budgets are planned is
the *scaling law*: across many orders of magnitude, language-model loss
falls as a power law in each of model size, data, and compute, provided
none of the three is the bottleneck :cite:`kaplan2020scaling`. On log-log
axes a power law is a straight line, and straight lines extrapolate: a
lab can fit the line on cheap runs and read off what a thousandfold more
compute should deliver. The follow-up that named an era asked how to
*split* a fixed compute budget between $N$ and $D$ and found the answer
roughly balanced: parameters and tokens should grow together, about
twenty tokens per parameter at the optimum
:cite:`hoffmann2022training` — the *Chinchilla* ratio. Its 70B-parameter
model, trained on 1.4 trillion tokens, beat a 280B-parameter model
trained on 300 billion at the same compute: the era's flagships had been
oversized and underfed.

We can watch both halves of that story — the line and the reason it
breaks — on one GPU, by training one family of models on one fixed
corpus.

### A Corpus Bigger than a Novella

:numref:`sec_gpt` already showed what a too-small corpus does: 4.7M
parameters against 180 KB of Wells bottomed out their validation loss
within a few hundred steps and spent the rest of the run memorizing. Our
own preliminary check confirms the diagnosis for the whole size range: on
*The Time Machine* alone, every configuration beyond about $10^5$
parameters is saturated more or less immediately. A scaling study needs a
corpus that can keep its larger sizes honest for at least a while, and we
can assemble one without leaving the book's pantry: the Penn Treebank
training text (5 MB of Wall Street Journal prose that
:numref:`sec_word2vec_data` uses for word vectors) concatenated onto the
novel. The subclass below reuses the entire character pipeline of
`d2l.TimeMachine` and changes two things: `_download` returns the
concatenated text, and the windows stride by `num_steps` instead of
overlapping (five million overlapping windows would be a five-gigabyte
tensor for no statistical gain). One cosmetic note: PTB ships
pre-tokenized, with rare words replaced by `<unk>` and numbers by `N`,
which the character pipeline renders as the literal words *unk* and *n* —
harmless static for a character model.

```{.python .input #scaling-laws-a-corpus-bigger-than-a-novella}
%%tab pytorch
class ScalingCorpus(d2l.TimeMachine):
    """Character corpus: The Time Machine plus the PTB training text."""
    def _download(self):
        data_dir = d2l.download_extract('ptb')
        with open(f'{data_dir}/ptb.train.txt') as f:
            ptb = f.read()
        return super()._download() + ' ' + ptb

    def __init__(self, batch_size, num_steps, num_val=2000):
        d2l.DataModule.__init__(self)
        self.save_hyperparameters()
        corpus, self.vocab = self.build(self._download())
        array = d2l.tensor([corpus[i:i + num_steps + 1] for i in
                            range(0, len(corpus) - num_steps, num_steps)])
        self.X, self.Y = array[:, :-1], array[:, 1:]
        self.num_train = len(array) - num_val

data = ScalingCorpus(batch_size=64, num_steps=128)
print(f'{len(data.X) * 128 / 1e6:.1f}M characters in {len(data.X)} '
      f'windows, vocabulary size {len(data.vocab)}')
```

```{.python .input #scaling-laws-a-corpus-bigger-than-a-novella}
%%tab jax
class ScalingCorpus(d2l.TimeMachine):
    """Character corpus: The Time Machine plus the PTB training text."""
    def _download(self):
        data_dir = d2l.download_extract('ptb')
        with open(f'{data_dir}/ptb.train.txt') as f:
            ptb = f.read()
        return super()._download() + ' ' + ptb

    def __init__(self, batch_size, num_steps, num_val=2000):
        d2l.DataModule.__init__(self)
        self.save_hyperparameters()
        corpus, self.vocab = self.build(self._download())
        array = d2l.tensor([corpus[i:i + num_steps + 1] for i in
                            range(0, len(corpus) - num_steps, num_steps)])
        self.X, self.Y = array[:, :-1], array[:, 1:]
        self.num_train = len(array) - num_val

data = ScalingCorpus(batch_size=64, num_steps=128)
print(f'{len(data.X) * 128 / 1e6:.1f}M characters in {len(data.X)} '
      f'windows, vocabulary size {len(data.vocab)}')
```

Five million characters is still comically small — it is to a frontier
corpus what our models are to a frontier model — but the ratio is what
matters. Our sizes will run from roughly the largest model a corpus like
this can feed at the published twenty-tokens-per-parameter optimum to
fifty times past it, which is exactly the position a study of saturation
wants to be in.

### Five Sizes, One Diet

The design: hold the data fixed and move only the model. Five sizes of
`d2l.GPT`, widths 96 to 384 and depths 3 to 8 grown together the way real
model families grow, spanning 0.33M to 14.2M non-embedding parameters —
about a factor of forty. Every size consumes an identical diet of 16.4M
tokens (2,000 steps of 64 sequences of 128 characters), roughly three
passes over the corpus. Dropout is zero: it was a small-corpus crutch in
:numref:`sec_gpt`, this experiment is *about* the corpus running out, and
the recipe table below will show that models whose corpora outweigh their
parameters do not use it either. The one hyperparameter that must not be
held fixed is the learning rate: :numref:`sec_scaling` showed that the
tuned optimum drifts with width, and a rate frozen at any single value
would handicap one end of the family (our pilot grid confirmed it at
both ends). We apply that section's transfer rule instead of re-tuning:
anchor at $10^{-3}$ for width 256 and scale inversely with width. One
seeded run per size; rerunning a point with a different seed moves it by
less than a hundredth of a nat, well below the structure we are about to
read.

```{.python .input #scaling-laws-five-sizes-one-diet-1}
%%tab pytorch
def val_loss(model, data):
    model.to(device).eval()
    with torch.no_grad():
        losses = [F.cross_entropy(
            model(X.to(device)).flatten(0, 1), Y.to(device).flatten())
            for X, Y in data.val_dataloader()]
    model.train()
    return sum(l.item() for l in losses) / len(losses)

sizes = ((96, 3, 4), (128, 3, 4), (192, 4, 6), (256, 6, 8), (384, 8, 8))
Ns, curves = [], {'train': [], 'validation': []}
for d, num_blks, num_heads in sizes:
    torch.manual_seed(0)
    model = d2l.GPT(len(data.vocab), num_hiddens=d, num_heads=num_heads,
                    num_blks=num_blks, dropout=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3 * 256 / d,
                                  weight_decay=0.0)
    losses = d2l.train_lm(model, data, optimizer, 2000)
    Ns.append(count(model) - count(model.token_emb))
    curves['train'].append(sum(losses[-50:]) / 50)
    curves['validation'].append(val_loss(model, data))
    print(f'd={d:3d}, {num_blks} blocks: {Ns[-1] / 1e6:5.2f}M parameters, '
          f'train {curves["train"][-1]:.2f}, '
          f'validation {curves["validation"][-1]:.2f}')
```

```{.python .input #scaling-laws-five-sizes-one-diet-1}
%%tab jax
@nnx.jit
def batch_loss(model, X, Y):
    logits = model(X)
    return optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(-1, logits.shape[-1]), Y.reshape(-1)).mean()

def val_loss(model, data):
    model.eval()
    losses = [float(batch_loss(model, jnp.asarray(X), jnp.asarray(Y)))
              for X, Y in data.val_dataloader()]
    model.train()
    return sum(losses) / len(losses)

sizes = ((96, 3, 4), (128, 3, 4), (192, 4, 6), (256, 6, 8), (384, 8, 8))
Ns, curves = [], {'train': [], 'validation': []}
for d, num_blks, num_heads in sizes:
    model = d2l.GPT(len(data.vocab), num_hiddens=d, num_heads=num_heads,
                    num_blks=num_blks, dropout=0, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(
        model, optax.adamw(1e-3 * 256 / d, weight_decay=0.0),
        wrt=nnx.Param)
    losses = d2l.train_lm(model, data, optimizer, 2000)
    Ns.append(count(model) - count(model.token_emb))
    curves['train'].append(sum(losses[-50:]) / 50)
    curves['validation'].append(val_loss(model, data))
    print(f'd={d:3d}, {num_blks} blocks: {Ns[-1] / 1e6:5.2f}M parameters, '
          f'train {curves["train"][-1]:.2f}, '
          f'validation {curves["validation"][-1]:.2f}')
```

```{.python .input #scaling-laws-five-sizes-one-diet-2}
%%tab pytorch
d2l.plot(Ns, [curves['train'], curves['validation']],
         'parameters $N$ (non-embedding)', 'loss',
         legend=['train', 'validation'], xscale='log', yscale='log',
         fmts=('o-', 's--'))
```

```{.python .input #scaling-laws-five-sizes-one-diet-2}
%%tab jax
d2l.plot(Ns, [curves['train'], curves['validation']],
         'parameters $N$ (non-embedding)', 'loss',
         legend=['train', 'validation'], xscale='log', yscale='log',
         fmts=('o-', 's--'))
```

### Reading the Bend

The plot has two curves and each tells half the story. The smaller sizes
line up roughly on a straight line: on this diet, multiplying parameters
by three buys a similar-looking improvement each time, which is the
power-law regime that makes scaling predictable. The largest model
departs from it. Its validation loss improves on the next-largest by
visibly less than the trend predicts, while its *training* loss keeps
falling right on schedule — the gap between the two curves grows from
roughly nothing at the small end to several hundredths of a nat at the
largest. Nothing is wrong with the model; something is wrong with its diet.
Fourteen million parameters reading five million characters three times
over have started spending capacity on the corpus itself rather than on
English, the same failure mode :numref:`sec_gpt` produced on purpose,
caught here at its onset.

The Chinchilla ratio says this bend is exactly where it should be.
Twenty tokens per parameter puts the largest model a corpus this size can
feed at roughly $5.1\textrm{M}/20 \approx 0.26$M parameters — our
*smallest* size. Everything to its right is data-starved by
compute-optimal standards and survives on repetition, which is known to
be a serviceable substitute at small multiplicities and a rapidly
decaying one beyond a handful of passes
:cite:`Muennighoff.Rush.Barak.ea.2023`; three passes is mild, which is
why our mid-sized models still track the line and only the largest has
visibly hit the wall. That is the Chinchilla lesson in miniature: *data
must scale with parameters*, and a budget spent on width and depth alone
buys memorization, not language. We decline to fit an exponent to our
five points: one seed and a forty-fold range would dress noise in
decimals; the published fits span six orders of magnitude and entire
model families :cite:`kaplan2020scaling,hoffmann2022training`. What our
miniature shows is the *shape* those fits live on: the straight stretch
where scaling laws are trustworthy, and the departure that marks the edge
of their jurisdiction.

Two footnotes before trusting the plot. First, a scaling study is only as
good as the tuning of each point: had we frozen the learning rate across
widths, part of our "bend" would have been a tuning artifact
(:numref:`sec_scaling`; exercise 5 has you produce this artifact
deliberately). Second, loss is the quantity that scales smoothly;
downstream abilities can surface abruptly as loss creeps down, which is
why small differences on this axis are worth more than they look
:cite:`wei2022emergent`.

### The Published Form of the Law

Our miniature moved $N$ with $D$ pinned; the published fits treat both as
free variables. Across more than four hundred training runs,
:citet:`hoffmann2022training` fit the loss surface

$$
L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}},
$$
:eqlabel:`eq_chinchilla_law`

three terms with three different jobs. $E$ is the floor: the intrinsic
entropy of text, the loss that would remain with unlimited parameters and
unlimited data. Its presence is why raw loss cannot fall along a straight
line forever — a log–log plot only looks straight far above the floor, or
after a fitted $E$ has been subtracted; every raw-loss line must
eventually flatten into it. The term $A/N^{\alpha}$ is the capacity price of
approximating the true distribution with only $N$ parameters; it is the
term our five-point sweep traversed. The term $B/D^{\beta}$ is the
estimation price of seeing only $D$ tokens, and it is what bent our
largest model away from the line: with $D$ fixed, $E + B/D^{\beta}$ acts
as an effective floor that no added capacity can pierce.

Compute-optimal allocation drops out of :eqref:`eq_chinchilla_law` with
one constraint. Fix a budget $C \approx 6ND$, substitute $D = C/6N$, and
set the derivative in $N$ to zero: the optimum satisfies

$$
\alpha \frac{A}{N^{\alpha}} = \beta \frac{B}{D^{\beta}},
$$

i.e., spend until the two shrinkable terms are shrinking at equal
marginal rates. Solving gives $N^* \propto C^{\beta/(\alpha+\beta)}$ and
$D^* \propto C^{\alpha/(\alpha+\beta)}$. The fitted exponents come out
nearly equal ($\alpha \approx 0.34$, $\beta \approx 0.28$), so both
optima scale close to $C^{1/2}$: parameters and tokens should grow in
near-lockstep, and their ratio at the optimum is roughly constant — the
twenty tokens per parameter quoted above is that constant, evaluated at
the fit. We do not fit $E$, $\alpha$, $\beta$ to our own five points (the
previous subsection said why); what the miniature contributes is the
shape of both terms — the straight stretch is $A/N^{\alpha}$ falling
while the data term is negligible, and the bend is the crossover where
the fixed corpus's $B/D^{\beta}$ takes over.

## The Modern Recipe

If scale is what matters, what exactly do the trillion-token runs build?
The reports read like a single answer arrived at independently.
:numref:`tab_modern-recipe` compiles the architecture sections of seven
open-weights families — Mistral 7B
:cite:`Jiang.Sablayrolles.Mensch.ea.2023`, Llama 3
:cite:`Grattafiori.Dubey.Jauhri.ea.2024`, Qwen3
:cite:`Yang.Li.Yang.ea.2025`, OLMo 2 and 3 :cite:`OLMo.2025,OLMo3.2025`,
DeepSeek-V3 :cite:`Liu.Feng.Xue.ea.2024`, Gemma 3
:cite:`Gemma.Team.2025`, and GPT-OSS :cite:`OpenAI.2025` — onto the axes
this chapter built.

### Convergent Evolution

:The 2023–2025 open-weights recipe, one row per model family. Attention and cache column: :numref:`sec_kv-cache`. Normalization: :numref:`sec_transformer-block`. Positions: :numref:`sec_positional-information`. Mixture of experts: :numref:`sec_moe`.
:label:`tab_modern-recipe`

| model | attention and cache | normalization | positions | FFN | dense or MoE | dropout |
|:--|:--|:--|:--|:--|:--|:--|
| Mistral 7B (2023) | GQA $32{:}8$; sliding window 4096 | RMSNorm, pre | RoPE, $\theta = 10^4$ | SwiGLU | dense | none |
| Llama 3 (2024) | GQA $32{:}8$ | RMSNorm, pre | RoPE, $\theta = 5 \times 10^5$ | SwiGLU | dense | none |
| Qwen3 (2025) | GQA: $64{:}8$ dense (32B), $64{:}4$ MoE (235B-A22B) | RMSNorm, pre; QK-norm | RoPE, $\theta = 10^6$; YaRN for long context | SwiGLU | dense, and MoE: 128 experts, 8 active | none |
| OLMo 2/3 (2024/25) | MHA (7B), GQA (32B); OLMo 3: window 4096 on 3 of 4 layers | RMSNorm, post but off-stream; QK-norm | RoPE | SwiGLU | dense | none |
| DeepSeek-V3 (2024) | MLA: KV compressed to a 512-dim latent | RMSNorm, pre | RoPE on a decoupled 64-dim slice per head | SwiGLU | MoE: 256 experts plus 1 shared, 8 active | none |
| Gemma 3 (2025) | GQA; local:global $5{:}1$, window 1024 | RMSNorm, pre and post; QK-norm | RoPE, $\theta = 10^6$ global, $10^4$ local | GeGLU | dense | none |
| GPT-OSS (2025) | GQA $64{:}8$; window 128 on alternate layers; learned sinks | RMSNorm, pre | RoPE; YaRN | SwiGLU, clamped | MoE: 128 or 32 experts, 4 active | none |

Read down the columns and the convergence is hard to miss. Every row is
the block of :numref:`sec_transformer-block` in the causal wiring of
:numref:`sec_gpt`; nothing here is a new architecture in the sense that
the transformer was new against the LSTM. The differences cluster on
exactly three axes. *Stability*: where the norms sit and what they
normalize, with pre-norm almost everywhere, RMSNorm everywhere, QK-norm
spreading through the 2025 column (OLMo's post-but-off-stream placement
is the one live dissent, and it concedes the residual stream's identity
path, the actual lesson of our signal-propagation experiment). *The
cache*: GQA as the default, with the window-plus-sink and latent
compressions of :numref:`sec_kv-cache` where long contexts make the
cache the binding cost. *Capacity per FLOP*: gated FFNs in every row, and
mixture of experts (:numref:`sec_moe`) where the budget wants more
parameters than FLOPs. The dropout column is the quiet punchline of our
scaling study: at trillion-token scale the corpus outweighs the
parameters, overfitting is not the failure mode, and the regularizer we
still needed in :numref:`sec_gpt` has simply left the recipe. Even the
positions column is one idea at different dial settings: RoPE with an
inflated base for longer contexts, stretched further by interpolation
schemes (:numref:`sec_positional-information`); the newest twist —
dropping positions entirely on some layers, as Llama 4's NoPE-interleaved
long-context layers do — is still a setting of the same dial.

### Recipe Rows as Constructor Calls

:numref:`sec_gpt` promised that these rows would come back as argument
lists, and the debt is now due. Scaled to our size — width 256, six
blocks — here are four rows of the table as configurations of `d2l.GPT`,
with grouped-query attention dropped in through the same seam
:numref:`sec_kv-cache` built:

```{.python .input #scaling-laws-recipe-rows-as-constructor-calls}
%%tab pytorch
recipes = {
    'GPT-2 (2019)': dict(pos='learned', norm='layer', act='gelu',
                         pre_norm=True, bias=True, kv_heads=8),
    'Mistral-7B':   dict(pos='rope', norm='rms', act='swiglu',
                         pre_norm=True, bias=False, kv_heads=2),
    'Llama-3':      dict(pos='rope', norm='rms', act='swiglu',
                         pre_norm=True, bias=False, kv_heads=2),
    'Qwen3':        dict(pos='rope', norm='rms', act='swiglu',
                         pre_norm=True, bias=False, kv_heads=2),
}
for name, cfg in recipes.items():
    kv_heads = cfg.pop('kv_heads')
    model = d2l.GPT(vocab_size=1024, num_hiddens=256, num_heads=8,
                    num_blks=6, **cfg)
    if kv_heads < 8:            # grouped-query attention, as deployed
        for blk in model.blks:
            blk.attention = d2l.GQAAttention(
                256, 8, kv_heads, rope=(cfg['pos'] == 'rope'))
    d2l.check_shape(model(torch.zeros(2, 16, dtype=torch.long)),
                    (2, 16, 1024))
    print(f'{name:12s} {count(model) / 1e6:.2f}M  '
          + ', '.join(f'{k}={v!r}' for k, v in cfg.items())
          + f', kv_heads={kv_heads}')
```

```{.python .input #scaling-laws-recipe-rows-as-constructor-calls}
%%tab jax
recipes = {
    'GPT-2 (2019)': dict(pos='learned', norm='layer', act='gelu',
                         pre_norm=True, bias=True, kv_heads=8),
    'Mistral-7B':   dict(pos='rope', norm='rms', act='swiglu',
                         pre_norm=True, bias=False, kv_heads=2),
    'Llama-3':      dict(pos='rope', norm='rms', act='swiglu',
                         pre_norm=True, bias=False, kv_heads=2),
    'Qwen3':        dict(pos='rope', norm='rms', act='swiglu',
                         pre_norm=True, bias=False, kv_heads=2),
}
for name, cfg in recipes.items():
    kv_heads = cfg.pop('kv_heads')
    model = d2l.GPT(vocab_size=1024, num_hiddens=256, num_heads=8,
                    num_blks=6, rngs=nnx.Rngs(0), **cfg)
    if kv_heads < 8:            # grouped-query attention, as deployed
        for blk in model.blks:
            blk.attention = d2l.GQAAttention(
                256, 8, kv_heads, rope=(cfg['pos'] == 'rope'),
                rngs=nnx.Rngs(0))
    d2l.check_shape(model(jnp.zeros((2, 16), dtype=jnp.int32)),
                    (2, 16, 1024))
    print(f'{name:12s} {count(model) / 1e6:.2f}M  '
          + ', '.join(f'{k}={v!r}' for k, v in cfg.items())
          + f', kv_heads={kv_heads}')
```

The printout is the point: the three modern rows are the *same argument
list*. At the resolution of our constructor, the field has one recipe,
and what distinguishes deployed families lives in the columns the flags
do not reach: window widths, RoPE bases, expert counts, and the
normalization refinements we left as exercises (QK-norm slots in through
`attn_factory`, as in :numref:`sec_transformer-block`'s exercises). The
two rows we did not instantiate are the two seams working as designed:
DeepSeek's MLA is the low-rank cache compression whose miniature
:numref:`sec_kv-cache` trained, and every MoE cell in the table swaps the
FFN through `ffn_factory` for the expert layer of :numref:`sec_moe`.
Convergent evolution is the biologist's name for this: independent
lineages under the same selection pressure arriving at the same body
plan. The selection pressures here are three bills — training stability,
decode-time memory, capacity per FLOP — and every 2026 deployment is
recognizably the 2017 transformer with engineering applied to exactly
those three lines of the budget.

## Where the Field Is Moving

Three directions are worth pointers as this chapter closes. First, the
attention layer's quadratic cost (:numref:`sec_attention-at-scale`) is
increasingly rationed rather than paid: the sliding-window rows of
:numref:`tab_modern-recipe` are the mild form, and the stronger form
replaces most attention layers outright with the linear-time recurrent
mixers of :numref:`chap_modern_rnn`, keeping full attention in only a
fraction of layers. These hybrids bet that exact global lookup is worth
its price only a few times per stack; where that bet pays, and where it
fails, is that chapter's story. Second, long context has become an
engineering discipline of its own: the number in a model card is a
compound of RoPE base inflation and interpolation
(:numref:`sec_positional-information`), window-and-sink cache policies
(:numref:`sec_kv-cache`), and the systems work of
:numref:`chap_performance` — none of it a new body plan; the
architectural moves are settings of the attention layer this chapter
already built, and the rest is positional bookkeeping and systems work.
Third, everything this chapter held fixed — what to train *on*, and what
happens after the loss stops falling — is where the gains have moved:
data curation, instruction tuning, and learning from feedback are the
subject of the Language Models part, beginning in
:numref:`chap_nlp_pretrain`.

## Summary

Transformer training cost is two multiplications: non-embedding
parameters ($N \approx 12Ld^2$; embeddings are tokenizer bookkeeping and
are counted separately) times tokens, times six — two FLOPs per parameter
per token forward, four backward, :eqref:`eq_six_nd` — with corrections
(attention scores, $n/6d$) that stay subleading until contexts reach
thousands. A profiler and XLA's cost analysis both confirm the rule on
our GPT to within a few percent. On a fixed diet of tokens, loss falls
with model size along a rough power law until the corpus can no longer
feed the model, then bends away: our five-size study on 5.1M characters
shows the straight stretch, the departure of the largest model, and the
widening train–validation gap that explains it. The published account of
that shape is $L(N,D) = E + A N^{-\alpha} + B D^{-\beta}$ — an entropy
floor, a capacity term, a data term — whose compute-optimal allocation
under $C \approx 6ND$ grows parameters and tokens together, at about
twenty tokens per parameter (the Chinchilla ratio); its lesson is that
data must scale with parameters.
What the scaled-up runs build has converged: GQA or latent-compressed
attention over a pre-norm RMSNorm block with a gated FFN, RoPE positions,
no dropout, and mixture of experts where parameters should outnumber
FLOPs; our recipe table's modern rows collapse to a single argument list
of the `GPT` class, differing in dial settings rather than design. The
2017 block, plus stability, cache, and capacity engineering, is the 2026
frontier.

## Exercises

1. Derive the six in :eqref:`eq_six_nd` for our default configuration
   exactly. Write the per-block matmul parameter count for width 256
   (fused QKV, output projection, and the three SwiGLU matrices), add the
   tied head, and multiply out $6ND$ for one batch. Compare against the
   profiler cell's analytic line. Then add the attention-score term: at
   what context length $n$ would the score work equal the linear-layer
   work for this width?
2. Run the Chinchilla sanity check on our study. At twenty tokens per
   parameter, how many tokens would each of our five sizes want, and how
   many *unique* characters does our corpus provide? Which sizes exceed
   the ratio, and how does that align with where the measured curve
   bends? Our budget shows each character about three times; using the
   findings of :citet:`Muennighoff.Rush.Barak.ea.2023`, argue whether
   doubling the number of passes would rescue the largest model.
3. Add a row to :numref:`tab_modern-recipe`. Pick an open-weights model
   this section does not cover — Kimi K2 :cite:`Kimi.Team.2025` is a good
   subject — read the architecture section of its report, and fill in
   every column. Which cells map onto `d2l.GPT` constructor flags, which
   need the `attn_factory`/`ffn_factory` seams, and which would need
   machinery this chapter has not built?
4. The census showed GPT-2's embedding at roughly a third of its
   parameters. Derive the condition on $V$, $d$, and $L$ under which the
   embedding holds at least half of all parameters, and check it against
   GPT-2's numbers. Why do scaling-law fits improve when embedding
   parameters are excluded from $N$?
5. Rerun the sweep with the learning rate frozen at $10^{-3}$ for every
   size. Which points move, and in which direction? Explain each
   movement using the width-dependence of the tuned optimum from
   :numref:`sec_scaling`, and state what a naive reader of the resulting
   plot would conclude that the properly tuned plot does not support.
6. Extend the sweep one size upward: width 512, ten blocks, about 31M
   non-embedding parameters. Before running it, predict its validation
   loss twice — once by extending the straight line through the small
   sizes, once by assuming the corpus is fully saturated (no improvement
   over the largest measured size) — and estimate the run's cost in
   FLOPs via :eqref:`eq_six_nd`. Then run it. Which prediction was
   closer, and what does the answer say about how much headroom three
   passes over 5.1M characters leave?

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §12.7]{.kicker}

Scaling laws and the modern recipe<br>
**count parameters and FLOPs · a scaling study, bend included · seven model families, one argument list**
:::
:::

::: {.slide title="What scale buys"}
[Boundary: §9's *Scaling Up* transfers hyperparameters across size (muP); this section asks what size itself buys]{.kicker}

- Our GPT and a frontier model differ by a **number**: parameters,
  tokens, FLOPs.
- Learn to count → verify against the machine → measure what the count
  buys → read the 2023–2025 reports as configurations of our class.
:::

::: {.slide title="The parameter census"}
Blocks scale as $12Ld^2$ ($4d^2$ attention + $8d^2$ FFN); embeddings as
$Vd$ — the embedding dominates when $V \gtrsim 12Ld$:

@!scaling-laws-a-parameter-census

- GPT-2's "124M" = an 85M transformer + a 39M lookup table.
- Scaling laws count **non-embedding** parameters $N$: the embedding
  tracks the tokenizer, not the per-token computation.
:::

::: {.slide title="Six FLOPs per parameter and token"}
$$C \approx \underbrace{2ND}_{\textrm{forward}} + \underbrace{4ND}_{\textrm{backward}} = 6ND$$

- Forward: each matmul parameter = one multiply–add per token.
- Backward: **two** matmuls per layer — grad w.r.t. input (chain rule)
  and w.r.t. weights (learning).
- Rounded away: attention scores, $n/(6d)$ — ~8% at $n=128$, $d=256$;
  parity only near $n = 6d$.

::: {.d2l-note}
Decode-time arithmetic (why a generated token costs $2N$) was §12.3's —
the KV cache turns it into a memory bill.
:::
:::

::: {.slide title="Check it against the machine"}
@!scaling-laws-checking-the-arithmetic

- PyTorch profiler: agrees to a fraction of a percent — it and the
  formula ignore the *same* term (the fused kernel's score work).
- XLA static analysis: ~10% above $6ND$ (it counts scores + softmax);
  backward:forward = **2.00**, the 4-to-2 of the derivation.
- Achieved TFLOP/s: an order of magnitude under peak — small models
  underfeed big GPUs.
:::

::: {.slide title="A scaling study on one GPU"}
Design: hold the diet fixed, move only the model.

- Corpus: *The Time Machine* + PTB text = **5.1M characters** (the novel
  alone saturates everything past $10^5$ parameters).
- Five sizes, widths 96→384 with depths 3→8: **0.33M → 14.2M** params.
- Identical diet: 16.4M tokens ≈ 3 passes; dropout 0; single seed
  (seed noise < 0.01 nat, checked).
- Learning rate ∝ 1/width (§9's transfer rule) — a frozen rate would
  handicap one end of the family.
:::

::: {.slide title="The sweep"}
@!scaling-laws-five-sizes-one-diet-1
:::

::: {.slide title="Reading the bend"}
@!scaling-laws-five-sizes-one-diet-2

- Small sizes: roughly a straight line — the power-law regime that makes
  scaling predictable (Kaplan et al., 2020).
- The largest **departs**: train keeps falling on trend, validation
  gains a fraction of the trend; the gap widens to several hundredths.
- Chinchilla (Hoffmann et al., 2022): ~**20 tokens/param** at optimum →
  this corpus feeds ~0.26M params. **Data must scale with parameters.**

::: {.d2l-note}
No fitted exponents: five points and one seed would dress noise in
decimals. The shape — line, then bend — is the finding.
:::
:::

::: {.slide title="The published form of the law"}
$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}} \qquad \textrm{(Hoffmann et al., 2022)}$$

- $E$: the entropy floor — why raw loss cannot stay on a straight line
  forever.
- $A/N^{\alpha}$: capacity — the term our five sizes traversed.
- $B/D^{\beta}$: data — with $D$ fixed, an effective floor: our bend.
- Under $C \approx 6ND$, the optimum sets
  $\alpha A N^{-\alpha} = \beta B D^{-\beta}$ →
  $N^* \propto C^{\beta/(\alpha+\beta)}$,
  $D^* \propto C^{\alpha/(\alpha+\beta)}$; fitted $\alpha \approx 0.34$,
  $\beta \approx 0.28$ put both near $C^{1/2}$: **grow them together**,
  ~20 tokens per parameter.
:::

::: {.slide title="The modern recipe (2023–2025)"}
| model | attention + cache | norm | FFN / experts |
|:--|:--|:--|:--|
| Mistral 7B | GQA 32:8, window 4096 | RMS pre | SwiGLU, dense |
| Llama 3 | GQA 32:8 | RMS pre | SwiGLU, dense |
| Qwen3 | GQA 64:8 dense, 64:4 MoE | RMS pre + QK | SwiGLU, dense & MoE 128/8 |
| DeepSeek-V3 | MLA: 512-d latent | RMS pre | SwiGLU, MoE 256+1/8 |
| Gemma 3 | GQA, local:global 5:1 | RMS pre+post + QK | GeGLU, dense |
| GPT-OSS | GQA 64:8, window 128 alt., sinks | RMS pre | SwiGLU, MoE 128/4 |

All: RoPE positions, **no dropout** — at trillion-token scale the corpus
outweighs the parameters.
:::

::: {.slide title="Recipes as constructor calls"}
@!scaling-laws-recipe-rows-as-constructor-calls

- The three modern rows print the **same argument list**.
- What the flags don't reach: window widths, RoPE bases, expert counts;
  MLA = §12.3's low-rank cache, MoE swaps in via `ffn_factory` (§12.6).
- Convergent evolution: three selection pressures — stability, cache,
  capacity per FLOP — one body plan: the 2017 block.
:::

::: {.slide title="Recap"}
- Cost = $6ND$; a profiler and a compiler both endorse the formula.
- On a fixed diet, loss falls with size along a rough line **until the
  corpus saturates the model** — the bend is the Chinchilla lesson.
- The 2026 recipe: GQA/MLA on a pre-norm RMSNorm block, gated FFN, RoPE,
  no dropout, MoE for capacity — our `GPT` class with different flags.
- Next frontiers: linear-attention hybrids (ch. 13), long context as
  engineering, and the data/post-training story (Language Models part).
:::
