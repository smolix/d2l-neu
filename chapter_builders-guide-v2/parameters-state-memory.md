# Parameters, State, and Memory
:label:`sec_parameters_v2`

Almost everything we do to a model other than calling it operates on its
*state*: the optimizer updates it, a checkpoint serializes it, `.to(device)`
moves it, fine-tuning trains part of it, and the answer to "will this model
fit on my GPU?" is a few lines of arithmetic over it. So far that state has
been handled for us; `nn.Linear` created the tensors and the training loop
updated them. This section opens the box: how to reach any tensor in a model,
which tensors are trained and which merely travel with the model, what they
all cost in bytes, and how to share or freeze them.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input #parameters-state-memory-parameters-state-and-memory}
%%tab pytorch
import torch
from torch import nn
```

## Accessing Parameters
:label:`subsec_param-access`

Our specimen is the residual MLP of :numref:`sec_model_construction_v2`,
redefined here so this section stands on its own: an input layer, a stack of
residual blocks, and an output head.

```{.python .input #parameters-state-memory-accessing-parameters-1}
%%tab pytorch
class ResidualBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

    def forward(self, X):
        return X + self.body(X)

torch.manual_seed(42)
net = nn.Sequential(nn.Linear(20, 64), ResidualBlock(64),
                    ResidualBlock(64), nn.Linear(64, 10))
X = torch.randn(2, 20)
net(X).shape
```

A model built from modules is a tree, and its parameters are the leaves. To
reach one leaf, walk the tree: indexing into a `Sequential` selects a child,
attribute access selects within it. `net[3]` is the output layer; its bias is
an `nn.Parameter`, a tensor that announces itself to the module system as
trainable.

```{.python .input #parameters-state-memory-accessing-parameters-2}
%%tab pytorch
type(net[3].bias), net[3].bias.shape
```

The same path syntax reaches arbitrarily deep. The first linear layer inside
the first residual block sits three levels down:

```{.python .input #parameters-state-memory-accessing-parameters-3}
%%tab pytorch
net[1].body[0].weight.shape
```

Each parameter also carries its gradient. We have not run backpropagation on
this model yet, so there is nothing to see:

```{.python .input #parameters-state-memory-accessing-parameters-4}
%%tab pytorch
net[3].weight.grad is None
```

Reaching parameters one path at a time is right for debugging a single layer.
The optimizer, weight decay, and checkpointing instead need *every* leaf, and
`named_parameters()` provides exactly that: a traversal of the whole tree that
yields each parameter with its path as the name.

```{.python .input #parameters-state-memory-accessing-parameters-5}
%%tab pytorch
[(name, p.shape) for name, p in net.named_parameters()]
```

Read one of the names closely. `1.body.0.weight` means: child
`1` of `net` (the first residual block), its submodule `body`, that module's
child `0`, and finally the leaf `weight`. Names are paths, so they survive any
amount of nesting, and they are exactly the keys of the model's `state_dict`,
the name-to-tensor mapping used for saving and loading
(:numref:`sec_read_write_v2`):

```{.python .input #parameters-state-memory-accessing-parameters-6}
%%tab pytorch
list(net.state_dict()) == [name for name, _ in net.named_parameters()]
```

One tree, one naming scheme, and every consumer, whether optimizer,
checkpoint, or debugger, walks it. The equality above holds for this model
because all of its state happens to be trainable. That is not always so.

## Parameters and Buffers

Some tensors must persist inside a model and follow it from device to device,
yet should never receive a gradient. The canonical example is batch
normalization :cite:`Ioffe.Szegedy.2015`: each layer maintains a running mean
and variance of its inputs, updated during the forward pass and used at
prediction time. Those statistics must be saved with the model and must move
to the GPU with it, but the optimizer has no business touching them. Later
chapters add more examples: causal attention masks, precomputed positional
tables, and the key--value cache of a language model at generation time.

PyTorch calls such tensors *buffers*, registered with `register_buffer`. The
rule of thumb: make it a *parameter* if the optimizer should update it, a
*buffer* if it must persist and travel with the model, and a plain Python
attribute otherwise. Here is a module that standardizes its inputs with
statistics computed once, ahead of time, from a reference sample:

```{.python .input #parameters-state-memory-parameters-and-buffers-1}
%%tab pytorch
class Whitener(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.out = nn.Linear(4, 2)

    def forward(self, X):
        return self.out((X - self.mean) / self.std)

sample = torch.randn(100, 4) * torch.arange(1., 5.)
whiten = Whitener(sample.mean(0), sample.std(0))
list(whiten.state_dict())
```

The buffers appear in the state dict, so they are checkpointed alongside the
weights. They do not appear among the parameters, so the optimizer never sees
them:

```{.python .input #parameters-state-memory-parameters-and-buffers-2}
%%tab pytorch
[name for name, _ in whiten.named_parameters()]
```

Registration is what makes the module system aware of a tensor. A tensor
stored as a plain attribute is invisible to it: not saved, and not converted
when the model moves. We can see this on the CPU by moving the module across
*dtypes*, which uses the same machinery as moving across devices. The
registered buffer converts; the plain attribute is left behind:

```{.python .input #parameters-state-memory-parameters-and-buffers-3}
%%tab pytorch
whiten.note = torch.zeros(4)   # plain attribute: invisible to the module
whiten.to(torch.float64)
whiten.mean.dtype, whiten.note.dtype
```

The device version of the same fact is the classic bug: a model works on the
CPU, then crashes after `.to('cuda')` because an unregistered tensor stayed
behind. On a machine with a GPU the following confirms that buffers move with
the module:

```{.python .input #parameters-state-memory-parameters-and-buffers-4}
%%tab pytorch
if torch.cuda.is_available():
    print('buffer lives on', whiten.to('cuda').mean.device)
else:
    print('no GPU here; on a CUDA machine, whiten.to("cuda") '
          'moves whiten.mean along with the parameters')
```

## Counting Parameters, Counting Bytes

Before any training job comes the question of whether the model fits in
memory, and the answer is arithmetic you can do on a napkin. Counting
parameters is one line. Counting *bytes* requires remembering everything that
training keeps per parameter: the weight itself, its gradient, and the
optimizer's state. Adam maintains two running moments per parameter, so in
fp32 the ledger reads:

| Training state       | Precision | Bytes per parameter |
|----------------------|-----------|---------------------|
| Weights              | fp32      | 4                   |
| Gradients            | fp32      | 4                   |
| Adam first moment    | fp32      | 4                   |
| Adam second moment   | fp32      | 4                   |
| Total                |           | 16                  |

```{.python .input #parameters-state-memory-counting-parameters-counting-bytes-1}
%%tab pytorch
n = sum(p.numel() for p in net.parameters())
weights, grads, adam_state = 4 * n, 4 * n, 8 * n   # bytes, all fp32
print(f'{n} parameters: {(weights + grads + adam_state) / 2**20:.2f} MiB '
      'for weights + gradients + Adam state')
```

The optimizer state is real memory, allocated tensor by tensor. After a
single training step, Adam's two moments together hold exactly two extra
copies of the model:

```{.python .input #parameters-state-memory-counting-parameters-counting-bytes-2}
%%tab pytorch
adam = torch.optim.Adam(net.parameters())
net(X).sum().backward()
adam.step()
moments = sum(t.numel() for s in adam.state.values()
              for t in s.values() if torch.is_tensor(t) and t.ndim > 0)
moments == 2 * n
```

For our little network the total is a third of a megabyte, which is why none
of this mattered until now. Scale the same arithmetic to a 1-billion-parameter
model and it dominates everything: 4 GB for the weights alone and 16 GB for
weights, gradients, and Adam state, before storing a single activation. The
memory that constrains model design is mostly this bookkeeping, and the
remaining term, the activations saved for the backward pass, depends on batch
size and is treated in :numref:`sec_use_gpu_v2`.

Large models train in mixed precision :cite:`Micikevicius.Narang.Alben.ea.2018`,
computing in fp16 or bf16 while Adam keeps fp32 master weights, and here
published accountings disagree. One common convention counts 18 bytes per
parameter (fp32 master weights, an fp16 working copy, fp32 gradients, and the
two moments); the ZeRO paper counts 16, keeping gradients in fp16
:cite:`Rajbhandari.Rasley.Ruwase.ea.2020`. The disagreement is bookkeeping,
not physics: both include the fp32 master copy and both include the moments.
The invariant to remember is that Adam's state alone is 8 bytes per parameter
in fp32, two full extra copies of your model, under every convention.

![Bytes per parameter under fp32 Adam training (top) versus a mixed-precision, ZeRO-style convention (bottom), drawn to the same width per byte. The two ledgers disagree on weights and gradients but agree, byte for byte, on the Adam moments m and v.](../img/bg-memory-ledger.svg)
:label:`fig_bg_memory-ledger`

:numref:`fig_bg_memory-ledger` lays the two conventions side by side, byte for
byte: the moments (right half of each bar) line up exactly, and the only
disagreement is how the weights and gradients themselves are stored.

## Tied Parameters

One tensor can serve several roles in a model. The standard example is the
two ends of a language model. The input embedding is a $|V| \times d$ table
mapping each of $|V|$ tokens to a $d$-dimensional vector; the output
projection maps a $d$-dimensional hidden state to $|V|$ logits, and its weight
matrix has the same shape and an analogous meaning: one vector per token.
*Tying* the two, using a single tensor for both roles, saves $|V| \times d$
parameters and trains better than keeping them separate
:cite:`Press.Wolf.2017,Inan.Khosravi.Socher.2017`. The savings are large: in
GPT-2 :cite:`Radford.Wu.Child.ea.2019` the shared $50257 \times 768$ matrix is
about 38.6 million of the model's 124 million parameters, roughly 31%.

![One weight matrix W, two call sites: the embedding looks up a row by token id, the output head multiplies by its transpose to produce logits. Tying means both point at the same tensor, so gradients from both uses sum into one place.](../img/bg-weight-tying.svg)
:label:`fig_bg_weight-tying`

:numref:`fig_bg_weight-tying` sketches the picture behind the two call sites.
A miniature version shows the mechanics. Tying is one assignment, and it is
aliasing, not copying:

```{.python .input #parameters-state-memory-tied-parameters-1}
%%tab pytorch
class TinyLM(nn.Module):
    def __init__(self, vocab_size, d, tied=True):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d)
        self.body = nn.Linear(d, d)
        self.head = nn.Linear(d, vocab_size, bias=False)
        if tied:
            self.head.weight = self.emb.weight   # one tensor, two roles

    def forward(self, tokens):
        return self.head(torch.relu(self.body(self.emb(tokens))))

lm = TinyLM(vocab_size=100, d=16)
lm.head.weight is lm.emb.weight
```

The module system understands the aliasing. Parameter traversal reports the
shared tensor once, so the count below reflects the $|V| \times d$ saving and
the optimizer updates the tensor once per step:

```{.python .input #parameters-state-memory-tied-parameters-2}
%%tab pytorch
untied = TinyLM(vocab_size=100, d=16, tied=False)
(sum(p.numel() for p in lm.parameters()),
 sum(p.numel() for p in untied.parameters()))
```

The state dict, by contrast, keeps *both* names, `emb.weight` and
`head.weight`, pointing at the same storage. That is deliberate: a checkpoint
saved from a tied model then loads into either a tied or an untied one.

```{.python .input #parameters-state-memory-tied-parameters-3}
%%tab pytorch
sd = lm.state_dict()
(sd['emb.weight'].data_ptr() == sd['head.weight'].data_ptr(),
 [name for name, _ in lm.named_parameters()])
```

What about gradients? During backpropagation each use of the tensor
contributes a gradient, and the contributions accumulate into the single
shared `.grad`. We can verify this against the untied twin: load the tied
model's values into it, run the same backward pass through both, and the tied
gradient equals the *sum* of the untied model's two gradients.

```{.python .input #parameters-state-memory-tied-parameters-4}
%%tab pytorch
untied.load_state_dict(lm.state_dict())   # same values, separate tensors
tokens = torch.randint(0, 100, (2, 8))
lm(tokens).sum().backward()
untied(tokens).sum().backward()
torch.allclose(lm.emb.weight.grad,
               untied.emb.weight.grad + untied.head.weight.grad)
```

## Freezing Parameters

Every fine-tuning recipe (:numref:`sec_fine_tuning`) rests on one primitive:
setting `requires_grad = False` on a parameter excludes it from
backpropagation, so the optimizer has nothing to apply and the weight stays
put. The typical pattern freezes a pretrained backbone and trains only a new
head. On a fresh copy of our residual MLP, freezing everything but the output
layer leaves 650 of 18,634 parameters trainable:

```{.python .input #parameters-state-memory-freezing-parameters-1}
%%tab pytorch
finetune = nn.Sequential(nn.Linear(20, 64), ResidualBlock(64),
                         ResidualBlock(64), nn.Linear(64, 10))
for p in finetune[:-1].parameters():
    p.requires_grad = False

(sum(p.numel() for p in finetune.parameters() if p.requires_grad),
 sum(p.numel() for p in finetune.parameters()))
```

The optimizer should receive only the trainable parameters. One gradient step
then moves the head and nothing else:

```{.python .input #parameters-state-memory-freezing-parameters-2}
%%tab pytorch
head_opt = torch.optim.SGD(
    (p for p in finetune.parameters() if p.requires_grad), lr=0.1)
before = [p.clone() for p in finetune.parameters()]
finetune(X).sum().backward()
head_opt.step()
[not torch.equal(b, p) for b, p in zip(before, finetune.parameters())]
```

Only the last two entries, the head's weight and bias, changed. Two pitfalls
deserve a warning, because both fail silently.

First, freezing does not reclaim optimizer memory. Passing only the trainable
parameters, as above, matters: an optimizer built over *all* parameters keeps
its state for every parameter that ever received a gradient. The Adam
instance from the previous section already stepped once on `net`, so freezing
`net`'s backbone now leaves its moments, 8 bytes per frozen parameter, fully
allocated:

```{.python .input #parameters-state-memory-freezing-parameters-3}
%%tab pytorch
for p in net[:-1].parameters():
    p.requires_grad = False   # frozen, but Adam's moments remain allocated
moments = sum(t.numel() for s in adam.state.values()
              for t in s.values() if torch.is_tensor(t) and t.ndim > 0)
moments == 2 * n
```

Second, freezing governs parameters only, and batch normalization carries
state that is not a parameter. Its running statistics are buffers, updated by
the *forward pass* whenever the layer is in training mode. Freeze a BatchNorm
layer's weight and bias and its running mean keeps drifting anyway:

```{.python .input #parameters-state-memory-freezing-parameters-4}
%%tab pytorch
bn = nn.BatchNorm1d(4)
for p in bn.parameters():
    p.requires_grad = False
before = bn.running_mean.clone()
bn(torch.randn(8, 4) + 3)          # train mode: stats update regardless
torch.allclose(bn.running_mean, before)
```

`requires_grad` and `.eval()` are orthogonal switches: the first stops
gradients, the second stops the behaviors tied to training mode, such as
running-statistics updates and dropout. To pin a BatchNorm layer during
fine-tuning you need both.

Freezing whole tensors is the bluntest form of partial training.
Parameter-efficient methods instead add small trainable low-rank corrections
next to frozen weights; the linear algebra behind them is developed in
:numref:`sec_mdl-svd-low-rank`. A related idea maintains derived,
non-trained state: an *exponential moving average* (EMA) of the weights kept
alongside the trained ones and used for evaluation, which often outperforms
the raw final iterate. Like BatchNorm statistics, the average is state the
optimizer never touches, updated outside backpropagation; we will use it when
we train generative models.

## Summary

A model's state is one named tree of tensors. `named_parameters()` walks the
trainable leaves; the `state_dict` adds buffers, tensors that persist and
move with the model but receive no gradients, such as BatchNorm running
statistics. Training with Adam in fp32 costs 16 bytes per parameter (4 weights,
4 gradients, 8 optimizer state) before activations, and the 8 bytes of Adam
state per parameter survive every mixed-precision accounting convention.
Assigning one parameter to two modules ties them: one entry in
`named_parameters()`, gradients summed over its uses. Setting
`requires_grad = False` freezes a parameter, but reclaims no optimizer state
already allocated and does not stop buffer updates in training mode.

## Exercises

1. Write a helper that reports the byte cost of fp32 Adam training separately
   for each top-level child of `net` (use `net.named_children()`). Which block
   dominates, and would that still hold if the residual blocks were 10 times
   wider?
1. BatchNorm's running mean and variance are buffers. Suppose you re-registered
   them as parameters so that the optimizer updates them. What goes wrong
   during training, and why does gradient descent on a running average not
   compute the same thing as the forward-pass update rule it replaces?
1. Tie two layers as in `TinyLM`, then copy the model with
   `copy.deepcopy`. Is the copy still tied? Check with `is`, and explain how a
   copy operation can preserve aliasing.
1. Freeze `lm.emb.weight` in the tied `TinyLM` but leave `lm.head` alone. How
   many trainable parameters remain? Explain what this teaches about the
   interaction between tying and freezing.
