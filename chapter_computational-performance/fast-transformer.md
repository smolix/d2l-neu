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
attributed to the section that taught it, re-profiling as the bottleneck
moves. The result is a single plot that is the chapter's closing argument:
the method is real, and it compounds.

*Prerequisites: the entire chapter. The subject is* `d2l.GPT` *and*
`d2l.TimeMachine` *from* :numref:`sec_gpt`*, reused verbatim.*

```{.python .input #fast-transformer-case-study-making-a-transformer-fast}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import json
import os
import pathlib
import subprocess
import sys
import time
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

torch.set_float32_matmul_precision('high')
# The profiler's CUDA instrumentation must not outlive its cell — without
# this, every timing after the first profile pays a per-launch tax.
os.environ['TEARDOWN_CUPTI'] = '1'
```

## The Subject
:label:`subsec_ft-subject`

We take ch. 11's GPT at a width where the experiment can be read cleanly.
At the `num_hiddens=256` of :numref:`sec_gpt`, the model's matmuls are
small enough that some techniques do not clear the measurement noise —
worse, bf16 actively *hurts*, because width-256 matmuls are too small to
feed the tensor cores and the casting overhead dominates (an instructive
failure, pursued in the exercises). Widening to `num_hiddens=512` (about
19M parameters, six blocks, char-level *Time Machine*, context 128) puts
every rung comfortably above the noise floor. The metric is training
throughput in **tokens per second**, measured with the discipline of
:numref:`sec_perf_model`. The case study is PyTorch-only by design — it
exercises `torch.compile`, autocast, checkpointing, and DDP as one stack;
JAX readers have met every rung in the JAX tabs of
:numref:`sec_compilation`–:numref:`sec_multi_gpu_concise`, and the
exercises reprise the ladder there.

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

def batches(dm):
    while True:
        for X, Y in dm.train_dataloader():
            if X.shape[0] == dm.batch_size:   # constant shape: no retrace
                yield X.to(device), Y.to(device)
stream = batches(data)
```

Before any rung, the measurement itself must be trustworthy, and two traps
had to be defused here — both worth naming, because each corrupted an
early draft of this section by tens of percent. First, the dataset's
ragged final batch (10,000 windows do not divide by 64) hands
`torch.compile` a new shape once per epoch, and the retrace it triggers
(:numref:`sec_compilation`) lands *inside* a timing window; the stream
above drops the ragged tail so the compiler sees one shape. Second, the
profiler's instrumentation is not free and, by default, does not fully
detach when its `with` block ends — every measurement taken afterwards in
the same process inherits a per-launch tax that hits small-batch
configurations hardest; the `TEARDOWN_CUPTI` line in the imports cell
makes profiling stop costing once it stops running. Both are the
chapter's opening lesson in miniature: *the first thing to profile is
your own experiment.* Beyond that, every measurement warms up long enough
for one-time costs — allocator growth, autotuning, compilation caches,
and whatever clock state the driver is in — to settle, and timed windows
close with a device sync:

```{.python .input #fast-transformer-the-subject-2}
%%tab pytorch
def throughput(step_fn, warmup=60, timed=100):
    """Tokens/s with warmup + device sync (see :numref:`sec_perf_model`)."""
    for _ in range(warmup):
        X, Y = next(stream); step_fn(X, Y)
    torch.cuda.synchronize(); t0 = time.perf_counter(); n = 0
    for _ in range(timed):
        X, Y = next(stream); step_fn(X, Y); n += X.numel()
    torch.cuda.synchronize()
    return n / (time.perf_counter() - t0)
```

One note on the metric: `throughput` times the *whole* pipeline —
`next(stream)` includes the `DataLoader` and the host-to-device copy — so
these are end-to-end tokens per second, the number a training run actually
delivers, not a kernels-only figure. (At this scale the input pipeline is
under two percent of a step; we measured that too.)

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
        torch.profiler.ProfilerActivity.CUDA], acc_events=True) as prof:
    for _ in range(5):
        step_eager(*next(stream))
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=6))
tput0 = throughput(step_eager)
print(f'R0 eager: {tput0:.0f} tokens/s')
```

Read the profile for the model's character. Matmuls (the attention
projections and the feed-forward blocks) carry about half of the device
time — but only about half: the rest is a long tail of small elementwise
and normalization kernels (note the *hundreds* of `aten::mul` calls per
five steps), each launched separately, none individually visible. And the
CPU-total column is nearly as large as the CUDA total — remembering
:numref:`sec_perf_model`'s caveat that these totals overlap across
threads and streams, that still says the dispatch thread is busy most of
the step. The diagnosis, then: partly compute-bound, but with a fusible
elementwise tail and non-trivial launch traffic — which predicts that
compilation should pay something here, and tensor-core precision after
it. The rungs will check both.

## Rungs, Each One Measured
:label:`subsec_ft-rungs`

**Rung 1 — compile (:numref:`sec_compilation`).** The profile promised a
fusible tail and busy dispatch; compilation is the matching fix. Before
the timing, the correctness gate — the compiled model must produce the
same logits as the eager one (they agree to about $10^{-2}$; the residue
is tf32 matmuls re-associated by fusion, not a wrong answer). *Same
answer first, then faster:*

```{.python .input #fast-transformer-rungs-each-one-measured-1}
%%tab pytorch
compiled = torch.compile(model)
def step_compiled(X, Y):
    opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(compiled(X).reshape(-1, vocab_size), Y.reshape(-1))
    loss.backward(); opt.step()

t0 = time.perf_counter(); step_compiled(*next(stream)); torch.cuda.synchronize()
print(f'first (compiling) step: {time.perf_counter() - t0:.1f} s')
X, _ = next(stream)
assert torch.allclose(compiled(X), model(X), atol=1e-2, rtol=1e-3)
tput1 = throughput(step_compiled)
print(f'R1 compiled: {tput1:.0f} tokens/s ({tput1 / tput0:.2f}x)')
```

The first call pays about two seconds of compile time; steady state pays
that back at roughly 1.3× — real money, from a model that is *not*
overhead-bound in the :numref:`sec_perf_model` sense. The gain is the
profile's tail, cashed in: the elementwise chains between matmuls fuse
into a handful of generated kernels, and the launch traffic drops with
them. A re-profile confirms the bottleneck moved:

```{.python .input #fast-transformer-rungs-each-one-measured-5}
%%tab pytorch
with torch.profiler.profile(activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA], acc_events=True) as prof:
    for _ in range(5):
        step_compiled(*next(stream))
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=5))
```

The elementwise tail has folded into `CompiledFunction` regions, and the
matmuls' share of device time has *risen* — the step is now more
tensor-core-bound than before, which is precisely the regime that
precision attacks. So, precision next.

**Rung 2 — bf16 (:numref:`sec_memory_precision`).** Add bf16 autocast on
the forward and loss, with backward outside the context, as
:numref:`sec_memory_precision` prescribed (no `GradScaler` — bf16 shares
fp32's exponent range). At width 512 the matmuls are large enough for the
tensor cores to pay:

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

**Rung 3 — raise the batch, raise the intensity
(:numref:`sec_memory_precision`, :numref:`sec_perf_model`).** A bigger
per-device batch raises the matmuls' arithmetic intensity, climbing the
roofline toward the compute roof — a rung in its own right, whatever the
precision. Memory is its usual price, so we also run a control: a few
fp32 steps at batch 512, to see what the batch *would* cost without bf16.
The control shows fp32 at 512 still fits this 24 GB card — snugly — so
bf16 did not unlock this batch; what it bought is *headroom*, roughly
half the footprint, which becomes the difference between fitting and not
the moment the model, context, or batch grows again:

```{.python .input #fast-transformer-rungs-each-one-measured-3}
%%tab pytorch
big = d2l.TimeMachine(batch_size=512, num_steps=128, tokenization='char')
big_stream = batches(big)

def throughput_big(step_fn, warmup=60, timed=50):
    for _ in range(warmup):
        step_fn(*next(big_stream))
    torch.cuda.synchronize(); t0 = time.perf_counter(); n = 0
    for _ in range(timed):
        X, Y = next(big_stream); step_fn(X, Y); n += X.numel()
    torch.cuda.synchronize()
    return n / (time.perf_counter() - t0)

torch.cuda.reset_peak_memory_stats()      # control: batch 512 in fp32
for _ in range(3):
    step_eager(*next(big_stream))
mem_fp32 = torch.cuda.max_memory_allocated() / 2**30
torch.cuda.reset_peak_memory_stats()
tput3 = throughput_big(step_bf16)
print(f'R3 +batch-up (512): {tput3:.0f} tokens/s ({tput3 / tput2:.2f}x), '
      f'peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB '
      f'(fp32 control: {mem_fp32:.1f} GiB)')
```

Another rung, another re-profile. At batch 512 the average matmul runs
several times longer per call — bigger tiles, better tensor-core
utilization — and the launch-and-dispatch side has receded to a footnote:

```{.python .input #fast-transformer-rungs-each-one-measured-6}
%%tab pytorch
with torch.profiler.profile(activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA], acc_events=True) as prof:
    for _ in range(5):
        step_bf16(*next(big_stream))
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=5))
```

**Rung 4 — activation checkpointing, and why it does *not* help here
(:numref:`sec_memory_precision`).** This is a deliberate *negative* rung.
Checkpointing trades compute for memory — but at this scale memory is not
the binding constraint (the bf16 batch-512 step fits in about a third of
the card). Recomputing activations therefore only *costs* time and saves
memory we did not need. Measuring the loss is the lesson: knowing when a
technique does not apply is as much the method as knowing when it does.
The wrapper below mirrors `GPT.forward` exactly, checkpointing each
block — and because a mirror can drift, we assert it is one before timing
anything:

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

ckpt_gpt = CheckpointedGPT(model)
X, _ = next(big_stream)
assert torch.allclose(ckpt_gpt(X), model(X), atol=1e-6)   # an exact mirror

ckpt_model = torch.compile(ckpt_gpt)
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
about 19M parameters, so roughly 76 MB of gradients must allreduce every
step; the NCCL collective on our host-staged box sustains around five
GB/s per device in :numref:`sec_multi_gpu`'s bytes-per-device convention
(the cell prices with 4.5, toward the conservative end of its run-to-run
range; NCCL's own "busbw" reads ~2 GB/s — the default-fallback figure:
:numref:`sec_multi_gpu_concise` measures a five-fold-faster configured
mode, and shows why these runs nonetheless keep the library's defaults), which
puts the allreduce at
roughly the same tens of milliseconds as the compute itself — a
transformer's parameters are
proportional to its compute, so unlike a convolutional ResNet it offers
little extra compute to hide communication behind. Summing the two terms,
:eqref:`eq_dp_cost` predicts a *weak* two-GPU gain: scarcely more than one
single-GPU throughput at $k=2$, under half of linear at $k=4$. One
refinement before trusting the measurement: the serial sum is a *floor* —
DDP overlaps its bucketed allreduce under backward
(:numref:`fig_ddp_overlap`), so the measurement can land above the
serial prediction, and how far above measures the overlap. We reuse
13.6's sidecar idiom on this very model (eager, per-rank batch 64 — weak
scaling, the convention of :numref:`subsec_mg-accounting`; each rank
streams its own batches, so this is a throughput measurement, not a
convergence claim):

```{.python .input #fast-transformer-rungs-each-one-measured-7}
%%tab pytorch
GPT_DDP = r'''
import json, os, time, torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from d2l import torch as d2l

rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank); dist.init_process_group("nccl")
torch.set_float32_matmul_precision("high")

data = d2l.TimeMachine(batch_size=64, num_steps=128, tokenization='char')
torch.manual_seed(0)
model = d2l.GPT(len(data.vocab), num_hiddens=512, num_blks=6).to(rank)
model = DDP(model, device_ids=[rank])
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
torch.manual_seed(rank)                 # each rank streams its own batches

def batches():
    while True:
        for X, Y in data.train_dataloader():
            yield X.to(rank), Y.to(rank)
stream = batches()

def step(X, Y):
    opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(X).reshape(-1, len(data.vocab)),
                           Y.reshape(-1))
    loss.backward(); opt.step()

for _ in range(30):                     # warmup: clocks, caches, NCCL
    step(*next(stream))
torch.cuda.synchronize(); t0 = time.time(); n = 0
for _ in range(50):
    X, Y = next(stream); step(X, Y); n += X.numel()
torch.cuda.synchronize()
here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, f"rank{rank}.json"), "w") as f:
    json.dump({"tokens_per_s": n / (time.time() - t0)}, f)
dist.destroy_process_group()
'''
GPT_DIR = pathlib.Path('ddp_gpt')
GPT_DIR.mkdir(exist_ok=True)
_ = (GPT_DIR / 'train_gpt_ddp.py').write_text(GPT_DDP)

def ddp_gpt_tokens(k):
    for stale in GPT_DIR.glob('rank*.json'):
        stale.unlink()
    torchrun = str(pathlib.Path(sys.executable).parent / 'torchrun')
    subprocess.run([torchrun, '--standalone', f'--nproc-per-node={k}',
                    str(GPT_DIR / 'train_gpt_ddp.py')], check=True)
    total = 0.0
    for r in range(k):
        with open(GPT_DIR / f'rank{r}.json') as f:
            total += json.load(f)['tokens_per_s']
    return total
```

```{.python .input #fast-transformer-rungs-each-one-measured-8}
%%tab pytorch
n_bytes = 4 * sum(p.numel() for p in model.parameters())
beta = 4.5e9   # NCCL allreduce, effective bytes/device/s on this box (13.5)
t_comm, t_cmp = 2 * n_bytes / beta, 64 * 128 / tput0
print(f'per step: t_comm ~ {1e3 * t_comm:.0f} ms vs '
      f't_compute ~ {1e3 * t_cmp:.0f} ms')
torch.cuda.empty_cache()               # hand the ranks the parent's cache
for k in (k for k in (2, 4) if k <= d2l.num_gpus()):
    floor = k * 64 * 128 / (t_cmp + t_comm)
    meas = ddp_gpt_tokens(k)
    print(f'R5 k={k}: no-overlap floor {floor / 1e3:.0f}k tokens/s, '
          f'measured {meas / 1e3:.0f}k ({meas / tput0:.2f}x of one GPU)')
```

The measurement lands where the model said it would — pinned to the
floor, nowhere near the linear ceiling. At $k=2$ it sits somewhat above
the floor: DDP's overlapped buckets buy back part of the communication
time. At $k=4$ it slips a little *below* it: the floor was priced with
the $\beta$ we measured at two GPUs, and four ranks staging through the
same host path do not quite sustain that constant. Either way, data
parallelism on this fabric turns a second GPU into a modest fraction of
one, and four GPUs into well under half of linear. Communication-hungry
is the diagnosis, and it was written down *before* the launch. On an NVLink
box the same accounting predicts near-linear scaling — the constant
changes, the method does not. Note what this rung is *not*: it uses the
eager baseline, matching the prediction's inputs; stacking DDP on the
compiled-bf16-batch-512 configuration is the exercise, and thinking
through why its efficiency comes out *higher* is the point of doing it.

## The Waterfall
:label:`subsec_ft-waterfall`

Collect the single-GPU rungs into one plot — the chapter's closing image.
The waterfall is *cumulative*: each bar inherits every choice to its
left, so a bar's ratio to its neighbor isolates one technique while its
ratio to the first bar prices the whole stack.

```{.python .input #fast-transformer-the-waterfall}
%%tab pytorch
rungs = ['R0\neager', 'R1\ncompile', 'R2\n+bf16', 'R3\n+batch', 'R4\n+ckpt']
tputs = [tput0, tput1, tput2, tput3, tput4]
print(f'cumulative, R0 -> R3: {tputs[3] / tputs[0]:.2f}x')
d2l.plt.figure(figsize=(6, 3.5))
bars = d2l.plt.bar(rungs, [t / 1e3 for t in tputs],
                   color=['#7f7f7f', '#1f77b4', '#1f77b4', '#2ca02c', '#d62728'])
d2l.plt.ylabel('throughput (k tokens/s)')
d2l.plt.title('Making a Transformer fast: cumulative rungs')
for b, t in zip(bars, tputs):
    d2l.plt.text(b.get_x() + b.get_width() / 2, t / 1e3,
                 f'{t / tputs[0]:.1f}x', ha='center', va='bottom')
d2l.plt.show()
```

Read left to right, and let the measurements calibrate the intuition.
Compilation paid about 1.3× — on a model that was *not* overhead-bound —
by fusing the elementwise tail the profile exposed. Bf16 paid about 1.4×
on top, the tensor cores earning their silicon now that the matmuls are
wide enough to feed them. Raising the batch bought roughly another 1.3×
by climbing the roofline. And checkpointing, the red bar, *cost* about
a tenth of the throughput while cutting peak memory to around a third
(about 9 GiB down to 3) — a negative rung for *speed*, because memory was
never the binding constraint at this scale, even though it did exactly
what it promised for memory. The cumulative single-GPU speedup, baseline
to the batch-up rung, is the number the cell prints — about 2.4× in our
runs — and every increment traces to a section of this chapter. That the
pieces compose; that no single rung dominated but three modest ones
multiplied; that a technique can *cost* time when its constraint does not
bind; and that the largest errors we found while building this section
were in the *measurements*, not the model — that is the whole lesson:
**you cannot copy a recipe — you measure, classify, fix, and
re-measure.**

One closing check, because a waterfall of throughputs proves nothing
about learning: train the fast configuration — compiled, bf16, batch
512 — for a few hundred steps from a fresh initialization and watch the
loss, smoothed over twenty steps, actually fall. (On this tiny corpus a
few hundred big-batch steps revisit the text over a hundred times, so the
loss falls very far — the assertion, not the final value, is the point.)
Speed that breaks the model is not speed:

```{.python .input #fast-transformer-the-waterfall-1}
%%tab pytorch
learner = make_gpt()
learner_c = torch.compile(learner)
opt_l = torch.optim.AdamW(learner.parameters(), lr=1e-3)
losses = []
for _ in range(300):
    X, Y = next(big_stream)
    opt_l.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = F.cross_entropy(learner_c(X).reshape(-1, vocab_size),
                               Y.reshape(-1))
    loss.backward(); opt_l.step()
    losses.append(loss.detach())       # no host read in the hot loop (13.1)
losses = torch.stack(losses).cpu()
first, last = losses[:20].mean(), losses[-20:].mean()
print(f'smoothed loss: first 20 steps {first:.2f} -> last 20 steps {last:.2f}')
assert last < first, 'speed that breaks the model is not speed'
```

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
  (width 512, ~19M params), compile paid about 1.3×, bf16 about 1.4× on
  top, a bigger batch roughly another 1.3× — no single dominant win, but
  three modest ones multiplying to about 2.4×, with the waterfall cell
  printing the exact cumulative figure.
* Every rung is attributed to a section, and the bottleneck is
  re-profiled as it moves: after compilation the elementwise tail is
  fused away and the matmuls' share rises, which is what makes precision
  the next fix.
* Measurement is a technique too, and it can fail: a ragged final batch
  put a `torch.compile` retrace inside our timing window, and the
  profiler's instrumentation outlived its cell until told otherwise.
  Both had to be fixed before the waterfall meant anything.
* Two rungs teach by *not* helping: bf16 is a negative rung at width 256
  (matmuls too small for the tensor cores), and activation checkpointing
  is a negative rung here (memory is not the binding constraint). Knowing
  when a technique does not apply is the method too.
* Data parallelism is predicted before it is measured: a transformer's
  parameters scale with its compute, so DP on our host-staged fabric is
  communication-hungry — the measured throughput lands at the cost
  model's no-overlap floor (a shade above it at two GPUs, a shade below
  at four, where the host-staged fabric no longer sustains its two-GPU
  bandwidth), nowhere near the linear ceiling. The
  prediction-then-measurement is the demonstration.

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
1. Rung 5 parallelized the *eager* model. Stack DDP on the full ladder
   instead — compiled, bf16, per-rank batch 512 — and measure tokens/s at
   $k = 2$ and $k = 4$. Efficiency comes out markedly higher than the
   eager rung's. Explain why with :numref:`sec_multi_gpu`'s cost model:
   which term grew, and which stayed fixed?
1. Our R1 measurement was once corrupted by a retrace inside the timing
   window. Remove the ragged-batch filter from `batches`, re-run R1, and
   find the retrace two ways: in the throughput number, and via
   `torch._dynamo` logging. How many steps per epoch does the offending
   batch appear in, and why does a longer timed window dilute but not
   remove the damage?

<!-- slides -->

::: {.slide title="The Method, on a Real Model"}
Nothing new here — that is the point. Six sections built a
method and a toolbox; now run the whole loop on ch. 11's GPT
and take it down a **waterfall**, one rung per technique, each
measured and attributed.

Subject: `d2l.GPT`, width **512** (~19M params). Why not 256?
At 256, bf16 goes *backwards* — matmuls too small for the
tensor cores. The width choice is itself a measurement.
:::

::: {.slide title="Profile Your Experiment First"}
Two traps corrupted an early draft by tens of percent:

- a **ragged final batch** put a `torch.compile` retrace
  *inside* the timing window → keep shapes constant
- the profiler's instrumentation **outlives its cell** →
  `TEARDOWN_CUPTI=1`, or every later timing pays a launch tax

The metric is end-to-end tokens/s — `DataLoader` and H2D
included.
:::

::: {.slide title="Rung 0: Baseline, Profiled"}
Classify before you fix.

@fast-transformer-rung-0-baseline-profiled

Matmuls ≈ half the device time; the rest is a fusible
elementwise tail, with dispatch busy most of the step. That
predicts compile pays, then precision.
:::

::: {.slide title="The Rungs"}
- **R1 compile** — fuses the tail: **~1.3×** (and asserts
  compiled ≡ eager first)
- **R2 bf16** — tensor cores, matmuls wide enough: **~1.4×**
- **R3 batch-up** — climb the roofline: **~1.3×** (fp32-512
  control: bf16 bought headroom, not admission)
- **R4 checkpoint** — *negative* for speed (−~10%), but cuts
  peak memory ~3× (unneeded here)

@fast-transformer-rungs-each-one-measured-3
:::

::: {.slide title="The Waterfall"}
@fast-transformer-the-waterfall

Cumulative — each bar inherits every choice to its left. No
dominant rung; three modest wins multiply to **~2.4×**.
Checkpointing is red: a technique that helped a different
model *hurts* this one. A 300-step run confirms the fast
configuration still learns.
:::

::: {.slide title="Predict, Then Measure: Data Parallel"}
Transformer params ∝ compute ⇒ ~76 MB of gradients per step
with little compute to hide them. The cost model (§13.5) gives
a **no-overlap floor**; DDP's bucketing buys back some overlap.

@fast-transformer-rungs-each-one-measured-8

Measured lands at the floor — just above it at $k=2$ (overlap),
just below at $k=4$ (the staged fabric sags) — priced *before*
the launch. NVLink changes the constant, not the method.
:::

::: {.slide title="The Lore, and the Ladder Beyond"}
modded-nanoGPT's speedrun = this chapter's contents stacked to
the ceiling: compiled kernels, FlashAttention, a better
optimizer (Muon), fp8. Each record a new rung.

**measure → classify → fix → re-measure.** Two budgets, three
regimes, one loop — now shown on a real model, seven sections
deep.
:::
