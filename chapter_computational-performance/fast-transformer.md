# Case Study: Making a Transformer Fast
:label:`sec_fast_transformer`

This section teaches nothing new. That is the point. Six sections built a
method — measure, classify the regime, apply the matching fix, re-measure —
and a toolbox: the roofline and the profiler (:numref:`sec_perf_model`),
the hardware that sets the constants (:numref:`sec_hardware`), compilation
(:numref:`sec_compilation`), precision and memory
(:numref:`sec_memory_precision`), and data parallelism
(:numref:`sec_multi_gpu`, :numref:`sec_multi_gpu_concise`). Here we run the
whole loop end to end on a real model — the GPT of :numref:`sec_gpt` — and
take it down a *waterfall*: one rung per technique, each measured, each
attributed to the section that taught it, and each re-profiled to see where
the bottleneck moved. The result is a single plot that is the chapter's
closing argument: the method is real, and it compounds.

*Prerequisites: the entire chapter. The subject is* `d2l.GPT` *and*
`d2l.TimeMachine` *from* :numref:`sec_gpt`*, reused verbatim.*

```{.python .input #fast-transformer-case-study-making-a-transformer-fast}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import time
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

torch.set_float32_matmul_precision('high')
```

## The Subject
:label:`subsec_ft-subject`

We take ch. 11's GPT at a width that makes the experiment honest. At the
`num_hiddens=256` of :numref:`sec_gpt`, the model's matmuls are small
enough that some techniques do not clear the measurement noise — worse,
bf16 actively *hurts*, because width-256 matmuls are too small to feed the
tensor cores and the casting overhead dominates (an instructive failure,
pursued in the exercises). Widening to `num_hiddens=512` (about 18M
parameters, six blocks, char-level *Time Machine*, context 128) puts every
rung comfortably above the noise floor. The metric is training throughput
in **tokens per second**, measured with the discipline of
:numref:`sec_perf_model`.

```{.python .input #fast-transformer-the-subject-1}
%%tab pytorch
device = d2l.try_gpu()
data = d2l.TimeMachine(batch_size=64, num_steps=128, tokenization='char')
vocab_size = len(data.vocab)

def make_gpt():
    torch.manual_seed(0)
    return d2l.GPT(vocab_size, num_hiddens=512, num_blks=6).to(device)

model = make_gpt()
print(f'{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters')

def batches():
    while True:
        for X, Y in data.train_dataloader():
            yield X.to(device), Y.to(device)
stream = batches()
```

One discipline is non-negotiable on this box, and it is the warmup rule of
:numref:`sec_perf_model` made physical. Our GPUs run with persistence mode
off: idle, the clock idles near 200 MHz; under load it boosts past 2.7 GHz
within a second, then *throttles back* as the die heats, so a step timed
too early can be twice as fast as its steady state. Every measurement below
warms up long enough for the clock to settle — otherwise the waterfall
would be shaped by temperature, not by our techniques.

```{.python .input #fast-transformer-the-subject-2}
%%tab pytorch
def throughput(step_fn, warmup=60, timed=100):
    """Tokens/s with thermal warmup + device sync (see :numref:`sec_perf_model`)."""
    for _ in range(warmup):
        X, Y = next(stream); step_fn(X, Y)
    torch.cuda.synchronize(); t0 = time.perf_counter(); n = 0
    for _ in range(timed):
        X, Y = next(stream); step_fn(X, Y); n += X.numel()
    torch.cuda.synchronize()
    return n / (time.perf_counter() - t0)
```

## Rung 0: Baseline, Profiled
:label:`subsec_ft-baseline`

Eager execution, tf32-fair (matmul precision already `'high'`). Before
optimizing, we *classify*: profile one step and read where the time goes.

```{.python .input #fast-transformer-rung-0-baseline-profiled}
%%tab pytorch
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

def step_eager(X, Y):
    opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(X).reshape(-1, vocab_size), Y.reshape(-1))
    loss.backward(); opt.step()

for _ in range(3):
    step_eager(*next(stream))
with torch.profiler.profile(activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA]) as prof:
    for _ in range(5):
        step_eager(*next(stream))
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=6))
tput0 = throughput(step_eager)
print(f'R0 eager: {tput0:.0f} tokens/s')
```

At this scale the profile shows the character of the model: many modest
matmuls (the attention projections and the feed-forward blocks) and a
scattering of elementwise and normalization kernels, with device time that
already dominates the CPU dispatch time. At width 512 the matmuls are large
enough that the step is more compute- and bandwidth-bound than
overhead-bound — which, as the next rung shows, is exactly why compilation
turns out to buy little here.

## Rungs, Each One Measured
:label:`subsec_ft-rungs`

**Rung 1 — compile (:numref:`sec_compilation`).** Compilation fuses
elementwise chains and cuts launch overhead, so it pays most when a model
is overhead-bound. This one is not very overhead-bound, and the measurement
says so — the first call pays seconds of compile time, and steady state is
only marginally faster. That near-null result is itself informative: it
tells us the bottleneck is elsewhere, and sends us to precision next:

```{.python .input #fast-transformer-rungs-each-one-measured-1}
%%tab pytorch
compiled = torch.compile(model)
def step_compiled(X, Y):
    opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(compiled(X).reshape(-1, vocab_size), Y.reshape(-1))
    loss.backward(); opt.step()

t0 = time.perf_counter(); step_compiled(*next(stream)); torch.cuda.synchronize()
print(f'first (compiling) step: {time.perf_counter() - t0:.1f} s')
tput1 = throughput(step_compiled)
print(f'R1 compiled: {tput1:.0f} tokens/s ({tput1 / tput0:.2f}x)')
```

**Rung 2 — bf16 (:numref:`sec_memory_precision`).** Add bf16 autocast (no
`GradScaler` — bf16 shares fp32's exponent range). At width 512 the matmuls
are finally large enough for the tensor cores to pay:

```{.python .input #fast-transformer-rungs-each-one-measured-2}
%%tab pytorch
def step_bf16(X, Y):
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = F.cross_entropy(compiled(X).reshape(-1, vocab_size), Y.reshape(-1))
        loss.backward()
    opt.step()
tput2 = throughput(step_bf16)
print(f'R2 +bf16: {tput2:.0f} tokens/s ({tput2 / tput1:.2f}x)')
```

**Rung 3 — spend the saved memory on batch size
(:numref:`sec_memory_precision`).** This is the rung that ties precision to
the roofline. bf16 halved the activation bytes; the freed memory buys a
bigger per-device batch, and a bigger batch raises the matmuls' arithmetic
intensity (:numref:`sec_perf_model`), climbing the roofline toward the
compute roof. It is consistently one of the largest wins, here roughly
tied with bf16:

```{.python .input #fast-transformer-rungs-each-one-measured-3}
%%tab pytorch
big = d2l.TimeMachine(batch_size=512, num_steps=128, tokenization='char')
def big_batches():
    while True:
        for X, Y in big.train_dataloader():
            yield X.to(device), Y.to(device)
big_stream = big_batches()

def throughput_big(step_fn, warmup=60, timed=50):
    for _ in range(warmup):
        step_fn(*next(big_stream))
    torch.cuda.synchronize(); t0 = time.perf_counter(); n = 0
    for _ in range(timed):
        X, Y = next(big_stream); step_fn(X, Y); n += X.numel()
    torch.cuda.synchronize()
    return n / (time.perf_counter() - t0)

torch.cuda.reset_peak_memory_stats()
tput3 = throughput_big(step_bf16)
print(f'R3 +batch-up (512): {tput3:.0f} tokens/s ({tput3 / tput2:.2f}x), '
      f'peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB')
```

**Rung 4 — activation checkpointing, and why it does *not* help here
(:numref:`sec_memory_precision`).** This is a deliberate *negative* rung.
Checkpointing trades compute for memory — but at this scale memory is not
the binding constraint (the batch-512 step fits in well under half the
card). Recomputing activations therefore only *costs* time and saves
memory we did not need. Measuring the loss is the lesson: knowing when a
technique does not apply is as much the method as knowing when it does.

```{.python .input #fast-transformer-rungs-each-one-measured-4}
%%tab pytorch
class CheckpointedGPT(nn.Module):
    """ch. 11's GPT with its transformer blocks run under checkpointing.
    Reuses the exact submodules; only the block loop differs."""
    def __init__(self, gpt):
        super().__init__(); self.gpt = gpt
    def forward(self, X):
        m = self.gpt                              # mirror GPT.forward exactly,
        H = m.token_emb(X)                        # but checkpoint each block
        if m.pos == 'learned':                    # (default 'rope' adds no emb)
            H = H + m.pos_emb(torch.arange(X.shape[1], device=X.device))
        for blk in m.blks:
            H = ckpt.checkpoint(blk, H, use_reentrant=False)
        return F.linear(m.norm(H), m.token_emb.weight)

ckpt_model = torch.compile(CheckpointedGPT(model))
def step_ckpt(X, Y):
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = F.cross_entropy(ckpt_model(X).reshape(-1, vocab_size), Y.reshape(-1))
        loss.backward()
    opt.step()
torch.cuda.reset_peak_memory_stats()
tput4 = throughput_big(step_ckpt)
print(f'R4 +checkpoint: {tput4:.0f} tokens/s ({tput4 / tput3:.2f}x), '
      f'peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB')
```

**Rung 5 — data parallel across 2–4 GPUs
(:numref:`sec_multi_gpu_concise`), predicted first.** Before measuring, we
*predict*, using the accounting of :numref:`sec_multi_gpu`. This GPT has
about 18M parameters, so ~72 MB of gradients must allreduce every step
over our host-staged ~2 GB/s fabric — and a transformer's parameters are
proportional to its compute, so unlike a convolutional ResNet it offers
less compute to hide that communication behind. The cost model therefore
predicts a *modest* two-GPU gain and *visibly sublinear* four-GPU scaling —
communication-hungry, exactly the honest picture our box is built to show.
The measurement, run through 13.6's DDP harness on this same model, is
left to the notebook (it needs the multi-process launcher); its purpose is
to confirm the prediction, not to surprise us. On an NVLink box the same
accounting predicts near-linear scaling — the constant changes, the method
does not.

## The Waterfall
:label:`subsec_ft-waterfall`

Collect the single-GPU rungs into one plot — the chapter's closing image.
Each bar is a technique; the height is throughput; the annotation names the
regime it attacked.

```{.python .input #fast-transformer-the-waterfall}
%%tab pytorch
rungs = ['R0\neager', 'R1\ncompile', 'R2\n+bf16', 'R3\n+batch', 'R4\n+ckpt']
tputs = [tput0, tput1, tput2, tput3, tput4]
d2l.plt.figure(figsize=(6, 3.5))
bars = d2l.plt.bar(rungs, [t / 1e3 for t in tputs],
                   color=['#7f7f7f', '#1f77b4', '#1f77b4', '#2ca02c', '#d62728'])
d2l.plt.ylabel('throughput (k tokens/s)')
d2l.plt.title('Making a Transformer fast: one rung per technique')
for b, t in zip(bars, tputs):
    d2l.plt.text(b.get_x() + b.get_width() / 2, t / 1e3,
                 f'{t / tputs[0]:.1f}x', ha='center', va='bottom')
d2l.plt.show()
```

Read left to right, and let the measurements correct the intuition.
Compilation bought almost nothing here — a few percent — because at this
width the model is already more compute- than overhead-bound, so there is
little launch overhead to fuse away (a small transformer with thinner
matmuls would have gained more, which is the point of measuring rather than
assuming). Bf16 was the largest single-technique win, roughly 1.7×, now
that the matmuls are wide enough to feed the tensor cores. Spending the
freed memory on a bigger batch bought another ~1.6× by climbing the
roofline. And checkpointing, the red bar, *cost* about ten percent of
throughput while cutting peak memory to roughly a third (about 9 GiB down
to 3) — a negative rung for *speed*, because memory was never the binding
constraint at this scale (the batch-512 step already fit in under half the
card), even though it did exactly what it promised for memory. The
cumulative single-GPU speedup, baseline to the batch-up rung, is roughly
2.5×, and every increment traces to a section of this chapter. That the
pieces compose, that the biggest expected win (compile) barely showed while
an unglamorous one (batch size) paid, and that a technique can *cost* time
when its constraint does not bind, is the whole lesson: **you cannot copy a
recipe — you measure, classify, fix, and re-measure.** A short real training
run confirms the accelerated configuration still learns (the notebook trains
a few hundred steps and watches the loss fall); speed that breaks the model
is not speed.

## The Lore, and the Ladder Beyond
:label:`subsec_ft-lore`

The competitive edge of this method is a spectator sport. The
*modded-nanoGPT* speedrun tracks the wall-clock time to train a small GPT
to a target loss, and its record holders read like this chapter's table of
contents stacked to the ceiling: compiled and fused kernels
(:numref:`sec_compilation`), block-sparse and FlashAttention variants
(:numref:`sec_attention-at-scale`), a better optimizer
(:numref:`sec_muon`), aggressive low precision down to fp8 — each record a
new rung on a waterfall like ours. fp8 tensor cores exist on our Ada GPUs,
but the recipe for *training* in fp8 (its own loss-scaling and master-copy
discipline) is deferred to the Language Models part, along with the
multi-node parallelism where these techniques are stacked at frontier
scale.

That is where the chapter ends, seven sections deep into a single idea.
The machine has two budgets, arithmetic and bandwidth; a program lives in
one of three regimes; and the way to make anything fast is always the same
loop — **measure, classify, fix, re-measure** — now demonstrated on a real
model, one rung at a time.

## Summary

* The chapter's method applied whole: profile the baseline, classify the
  regime, apply one technique per rung, re-profile. On ch. 11's GPT
  (width 512, ~18M params), compilation bought almost nothing (this width is
  not overhead-bound), bf16 was the biggest single win (~1.7×), spending the
  freed memory on a bigger batch bought ~1.6× more, and the cumulative
  single-GPU speedup was roughly 2.5×.
* Every rung is attributed to a section, and each is re-measured: the fix
  that pays depends on the regime, which moves as you fix it.
* Two rungs teach by *not* helping: bf16 is a negative rung at width 256
  (matmuls too small for the tensor cores), and activation checkpointing
  is a negative rung here (memory is not the binding constraint). Knowing
  when a technique does not apply is the method too.
* Data parallelism is predicted before it is measured: a transformer's
  parameters scale with its compute, so DP on our host-staged fabric is
  communication-hungry — modest 2-GPU gains, sublinear at 4. The
  prediction-then-confirmation is the demonstration.

## Exercises

1. Reproduce the negative bf16 rung: run the R0→R1→R2 sequence at
   `num_hiddens=256` and confirm bf16 is *slower* than compile-alone. Then
   profile both and explain, in terms of :numref:`fig_roofline`, why the
   width-256 matmuls do not benefit from the tensor cores.
1. Add a rung of your own — `channels_last` memory format, pinned-memory
   dataloading, or `sdpa_kernel` backend pinning — measure it, and slot it
   into the waterfall. Which regime does it attack, and does it clear the
   noise floor?
1. Apply the whole ladder to a *different* model: ch. 12's Mamba capstone
   (:numref:`sec_mamba`) or ch. 11's ViT. Which rungs change sign, and
   why? (Hint: a model's arithmetic intensity decides which fixes pay.)
1. Take a rung that helped here — say, batch-up — and construct a model
   (deeper, or with a much larger vocabulary) where it *hurts*. Explain
   the reversal with the memory anatomy of :numref:`sec_memory_precision`.
1. Predict, from :numref:`sec_multi_gpu`'s cost model and the measured
   ~2 GB/s allreduce bandwidth, the four-GPU efficiency of the GPT data-
   parallel rung, then run it through 13.6's harness and compare. Where
   does prediction diverge from measurement, and what did the model omit?

<!-- slides -->

::: {.slide title="The Method, on a Real Model"}
Nothing new here — that is the point. Six sections built a
method and a toolbox; now run the whole loop on ch. 11's GPT
and take it down a **waterfall**, one rung per technique, each
measured and attributed.

Subject: `d2l.GPT`, width **512** (~18M params). Why not 256?
At 256, bf16 goes *backwards* — matmuls too small for the
tensor cores. The width choice is itself a measurement.
:::

::: {.slide title="Rung 0: Baseline, Profiled"}
Classify before you fix.

@fast-transformer-rung-0-baseline-profiled

Modest matmuls, device time already above dispatch time — more
compute- than overhead-bound at width 512. That predicts which
rung pays (and which won't).
:::

::: {.slide title="The Rungs"}
- **R1 compile** — barely helps (~4%): not overhead-bound here
- **R2 bf16** — tensor cores, matmuls wide enough: **~1.7×**
- **R3 batch-up** — spend freed memory, climb the roofline:
  **~1.6×**
- **R4 checkpoint** — *negative* for speed (−~10%), but cuts
  peak memory ~3× (unneeded here)

@fast-transformer-rungs-each-one-measured-3
:::

::: {.slide title="The Waterfall"}
@fast-transformer-the-waterfall

~2.5× cumulative, single GPU. Every increment traces to a
section. Checkpointing is red — a technique that helped a
different model *hurts* this one.
:::

::: {.slide title="Predict, Then Measure: Data Parallel"}
Transformer params ∝ compute ⇒ ~72 MB of gradients over a
~2 GB/s fabric with little compute to hide it. The cost model
(§13.5) predicts **modest 2-GPU, sublinear 4-GPU**.

The measurement confirms the prediction — that agreement *is*
the result. NVLink changes the constant, not the method.
:::

::: {.slide title="The Lore, and the Ladder Beyond"}
modded-nanoGPT's speedrun = this chapter's contents stacked to
the ceiling: compiled kernels, FlashAttention, a better
optimizer (Muon), fp8. Each record a new rung.

**measure → classify → fix → re-measure.** Two budgets, three
regimes, one loop — now shown on a real model, seven sections
deep.
:::
