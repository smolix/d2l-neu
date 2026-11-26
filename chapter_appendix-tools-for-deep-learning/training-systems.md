# Distributed Model Training
:label:`sec_training_systems`

The multi-GPU sections of :numref:`chap_performance` trained a ResNet and a
GPT on the four GPUs of one machine by giving every GPU a copy of the
model, and worked out how splitting the training state among the GPUs
makes room for a model that no single GPU holds. Models that people
fine-tune for real work are often larger still. A language model with 7
billion parameters, trained with Adam, needs about 112 GB for its weights,
gradients, and optimizer state before it stores a single activation. That
is more than the 80 GB of an H100, and training such a model from scratch
on one GPU would take years.

How do we spread the state and the work of one training run over many
accelerators, possibly in many machines, so that it computes the same
result in less time? Every answer partitions something: the batch, the
training state, the layers, or the sequence. Each partition adds
communication, and every added worker is one more place where the run can
fail. We therefore start from a measured single-device baseline and add
the simplest technique that removes an observed memory or throughput
limit. We order the techniques by the limit each removes, map them to the
libraries that implement them as of November 2026, and end with the
practices that keep a run of several weeks alive.

## Forms of Parallelism

### The Scaling Ladder

Each technique below removes a limit of the one before it and adds
communication, so we adopt them in order, the rungs of a *scaling ladder*
(:numref:`fig_tools_training_ladder`).

![A practical scaling ladder progresses from one accelerator to replication, sharding, and composed parallelism.](../img/tools-training-ladder.svg)
:label:`fig_tools_training_ladder`

1. **One accelerator** establishes the reference: loss curve, tokens or
   examples per second, peak memory, and a checkpoint you can trust.
1. **Data parallelism (DDP)** replicates the model on every GPU, feeds each
   a different slice of the batch, and averages gradients after every
   backward pass. It is the right choice whenever model, gradients, and
   optimizer state fit on one device.
1. **Fully sharded data parallelism (FSDP)**, stage 3 of the ZeRO family,
   shards parameters, gradients, and optimizer state across GPUs and
   gathers each layer only while it is being used
   (:numref:`subsec_mgp-fsdp`). It divides the per-device state memory by
   the number of GPUs and communicates about 1.5 times as much as DDP.
1. **Tensor, pipeline, context, and expert parallelism** split individual
   layers, groups of consecutive layers, the sequence dimension, and the
   experts of a mixture of experts (:numref:`sec_moe`), respectively.
   Large systems compose several of these, because no single axis scales
   far enough on its own.

In PyTorch, DDP and FSDP need a launcher and a wrapper:

```bash
torchrun --standalone --nproc-per-node=4 train.py
```

Inside `train.py`, `DistributedDataParallel(model)` applies DDP and
`fully_shard(model)` from `torch.distributed.fsdp` applies FSDP. This
API, known as FSDP2, stores each sharded parameter as a `DTensor`, and
PyTorch recommends it over the older `FullyShardedDataParallel` wrapper.
In JAX, arrays
carry a `NamedSharding` over a mesh of devices, and `jax.jit` compiles the
collectives that these placements imply
(:numref:`sec_multi_gpu_concise`).

Tensor parallelism exchanges activations inside every layer, while
pipeline parallelism sends activations only at stage boundaries. In the
Hugging Face *Ultra-Scale Playbook*, based on more than 4,000 runs on up
to 512 GPUs, stretching tensor parallelism from one 8-GPU node to two
typically cost about 43% of throughput, while stretching pipeline
parallelism the same way cost 14% when each step had few microbatches. Keep tensor parallelism inside a node,
where NVLink carries its traffic, and map the highest-volume collectives
to the fastest links.

One identity holds on every rung. The batch size that the optimizer sees is

$$
B_{\textrm{global}} = B_{\textrm{device}} \cdot N_{\textrm{ranks}} \cdot N_{\textrm{accumulation}},
$$
:eqlabel:`eq_global_batch`

the per-device batch times the number of data-parallel ranks times the
number of microbatches accumulated per optimizer step. Changing the number
of GPUs without adjusting the per-device batch size or the learning rate
therefore changes the optimization problem. Two further rules concern the
data: distributed samplers must not duplicate examples, and evaluation must
aggregate metrics with the right denominators. Violating either is the
usual reason a multi-GPU run reports a different accuracy.

### Training Memory

Most jobs outgrow one GPU because the training state does not fit, so the
first calculation is a memory budget (:numref:`fig_tools_training_memory`).

![Sharding, checkpointing, accumulation, and offload reduce different memory terms and introduce different costs.](../img/tools-training-memory.svg)
:label:`fig_tools_training_memory`

Peak training memory is the sum of parameters, gradients, optimizer state,
activations, communication buffers, and allocator overhead. For a model
with 7 billion parameters trained in BF16 with Adam, the parameters and
gradients take 14 GB each and Adam's two FP32 moments take 56 GB. The FP32
master copy of the parameters that mixed-precision training keeps adds
28 GB, for a total of 112 GB, or 16 bytes per parameter. The toy model
below omits that 28 GB master copy (leaving $14 + 14 + 56 = 84\text{ GiB}$ of model state when reusing the GB figures as GiB) and adds
18 GiB of activations and 5 GiB of temporary buffers ($107\text{ GiB}$ in total). It is crude, but it
ranks the strategies correctly, and it needs no GPU to run:

```{.python .input #training-systems-memory-model}
terms_gib = {"parameters": 14.0, "gradients": 14.0, "optimizer": 56.0,
             "activations": 18.0, "temporary": 5.0}

def per_device_gib(world_size=1, shard_state=False, ckpt_activations=False):
    divisor = world_size if shard_state else 1
    state = sum(terms_gib[k] / divisor
                for k in ("parameters", "gradients", "optimizer"))
    activations = terms_gib["activations"] * (0.35 if ckpt_activations else 1)
    return state + activations + terms_gib["temporary"]

for label, cfg in [("1 GPU, plain", (1, False, False)),
                   ("8 GPU DDP + ckpt", (8, False, True)),
                   ("8 GPU FSDP + ckpt", (8, True, True))]:
    print(f"{label:>20s}: {per_device_gib(*cfg):6.1f} GiB/device")
```

Plain training needs 107 GiB, more than an 80 GB H100 holds. DDP with
activation checkpointing still needs 95 GiB on every GPU: it replicates
the 84 GiB of state, and only the activations shrink. FSDP with
checkpointing on eight GPUs divides the state by eight and needs 22 GiB
per device.

Each standard memory technique changes a different term of the sum at a
different cost. **Mixed precision** reduces the bytes used by
selected tensors and operations. **Gradient accumulation** runs a smaller
microbatch per step while keeping the global batch of
:eqref:`eq_global_batch`, which reduces activation memory but not model
state. **Activation checkpointing** recomputes selected activations during
backpropagation; its memory savings and compute overhead depend on which
activations it keeps. **CPU or NVMe offload**, such as DeepSpeed's
ZeRO-Offload, moves state out of device memory and adds host-device
transfers to every step.
**LoRA and QLoRA** train small low-rank adapters instead of all weights,
which removes most gradients and optimizer state; QLoRA also stores the
frozen weights in 4 bits, so a 7B model can be fine-tuned on one 16 GB
card. Combine techniques deliberately: individually sensible optimizations
can interfere with compilation, overlap, or numerics.

## Software Libraries

Beyond one node, few people write these techniques by hand. Libraries
implement them, and the choice among them follows mostly from the scale of
the job. The projects below were maintained and widely used as of November
2026; this layer changes quickly, so check a project's recent releases
before adopting it.

**PyTorch built-ins.** Besides DDP and FSDP2, PyTorch has tensor- and
context-parallel APIs built on `DTensor` and a pipeline-parallel API,
`torch.distributed.pipelining`. Its documentation still marks them as
experimental or alpha. Most higher-level
tools below generate calls to these primitives.

**Hugging Face stack.** `accelerate` launches the same script on DDP,
FSDP2, or DeepSpeed, selected by a configuration file rather than by code
changes; `transformers.Trainer` sits on top of it; `peft` implements LoRA
and QLoRA; and `trl` provides trainers for supervised fine-tuning, DPO, and
GRPO-style reinforcement learning. This stack covers most fine-tuning jobs
on one node or a small cluster.

**DeepSpeed.** ZeRO stages 1–3 (sharding the optimizer state, then the
gradients, then the parameters) plus CPU and NVMe offload. ZeRO introduced
the sharding that FSDP2 now provides natively, and FSDP2 can also offload
state to CPU memory. DeepSpeed remains the usual choice when a job offloads
to NVMe or when another framework builds on it.

**Megatron Core and TorchTitan.** Libraries for large-scale pretraining.
NVIDIA's Megatron Core implements tensor, pipeline, context, and expert
parallelism with FP8 and FP4 training; Megatron-LM is its reference
training stack. TorchTitan is the PyTorch-native counterpart: a compact
reference stack that composes FSDP2 with tensor, pipeline, context, and
expert parallelism, with model definitions that include Llama 3 405B and
DeepSeek-V3 and published measurements on up to 512 GPUs. Both suit
models that need several forms of parallelism at once.

**Fine-tuning front ends.** Unsloth began with fast single-GPU LoRA and
QLoRA through fused kernels and now also covers full fine-tuning, GRPO,
and multiple GPUs; its gains depend on model, sequence length, and
hardware. Axolotl drives full and parameter-efficient fine-tunes on one or
several nodes from one YAML file, with FSDP2 and DeepSpeed backends.
LLaMA-Factory covers a similar range with a web interface. The PyTorch
team's `torchtune` stopped active development in 2025.

**JAX.** MaxText is the reference stack for pretraining and post-training
on TPUs and GPUs. Marin, which trains with Levanter, records its training
runs from raw data to the final model, failed experiments included, and
publishes the code, data pipelines, and logs so that others can reproduce
them.

**Reinforcement-learning post-training.** veRL combines FSDP, FSDP2, or
Megatron training with vLLM or SGLang for generating rollouts. OpenRLHF
builds on Ray, DeepSpeed, and vLLM.

### Tools by Scale

:Training tools by scale (as of November 2026)
:label:`tab_training_tools`

| Scale | Typical job | Reach for |
|---|---|---|
| 1 GPU, ≥ 8 GB | LoRA/QLoRA fine-tune ≤ 8B | Unsloth, or PEFT + Trainer |
| 1 node, 2–8 GPUs | full fine-tune up to ~30B, DDP/FSDP2 | Accelerate or Axolotl |
| few nodes | large fine-tune, small pretrain | FSDP2 + torchrun, DeepSpeed |
| many nodes | large-scale pretraining, MoE | Megatron Core, TorchTitan, MaxText |
| RL post-training | GRPO/DPO pipelines | TRL (small), veRL/OpenRLHF (large) |

The model sizes follow from the 16 bytes per parameter of a full
fine-tune: eight 80 GB GPUs hold the state of about 30B parameters with
room for activations. Use the simplest row that meets the measured
constraints, since rows lower in the table need more configuration.

## Fault Tolerance in Long Runs
:label:`subsec_training_long_run`

A run that crashes on day ten and cannot resume has produced nothing. As
the number of workers and the duration of a job grow, failures of
hardware, network, and software become routine, so recovery has to be
designed and tested like the rest of training.

* **Checkpoint completely and atomically.** A resumable checkpoint holds
  model and optimizer state, the learning-rate schedule and step count,
  data-loader position, and RNG state. Write to a temporary path and
  rename on completion, so that a crash of the training process mid-write
  cannot destroy the previous checkpoint; keep more than one until
  restore has been tested. Sharded (per-rank) checkpoints avoid
  gathering a model larger than any single host's memory.
* **Drill the recovery.** Kill the job deliberately, restore on fresh
  workers, and check that loss, step count, and data position continue as
  if nothing happened. Test recovery before relying on spot-priced
  instances (:numref:`sec_cloud_instances`), whose savings depend on
  resuming after every interruption.
* **Keep the accelerators supplied with data.** Profile data loading
  separately from compute: storage reads, decoding and augmentation,
  tokenization, and host-to-device transfer. A GPU that waits for input
  delivers low throughput, and adding GPUs to a data-bound job increases
  contention without increasing speed.
* **Monitor convergence as well as speed.** A high token rate says nothing
  about the loss. Log both, plus the time spent in collectives and in
  checkpointing. Debug distributed failures by
  reproducing them on one process, then on one node, then on many, with
  rank and host on every log line.

These practices make a run restartable, but every checkpoint and restart
still costs time; the exercises work out how that cost sets the best
checkpoint interval for a given failure rate.

## Summary

Distributed training partitions the batch, the training state, the layers, or the sequence across workers. Starting from a measured single-GPU baseline, use DDP while model state fits in memory, move to FSDP or ZeRO when it does not, and add tensor, pipeline, context, or expert parallelism only when model scale requires them, keeping communication-heavy tensor parallelism inside an NVLink node. Because peak memory is a sum of distinct terms, sharding, activation checkpointing, gradient accumulation, host offload, and LoRA each shrink a specific term at a predictable cost in communication, recomputation, or transfer time. As of November 2026, common choices are Unsloth or PEFT on a single GPU, Accelerate or Axolotl with FSDP2 on a single node, Megatron Core, TorchTitan, or MaxText for multi-node pretraining, and TRL or veRL for reinforcement-learning post-training. Multi-week runs finish reliably when checkpoints are complete and written atomically, recovery is drilled ahead of time, data pipelines keep pace with compute, and convergence is logged alongside throughput.

## Exercises

1. **Global batch size.** A pretraining job uses sequences of 4,096
   tokens and a global batch of $2^{22} \approx 4.2$ million tokens, that
   is, 1,024 sequences per optimizer step. Each GPU holds at most 4
   sequences per microbatch.
    1. For $N_{\textrm{ranks}} \in \{32, 64, 128, 256\}$, choose
       $B_{\textrm{device}}$ and $N_{\textrm{accumulation}}$ in
       :eqref:`eq_global_batch` so that the global batch stays at 1,024
       sequences with as few accumulation steps as possible. What is the
       largest world size that can train with this global batch under data
       parallelism alone?
    1. Let a microbatch of 4 sequences take time $t_{\textrm{mb}}$, and let
       the gradient all-reduce at the end of each optimizer step take time
       $t_{\textrm{ar}}$, without overlap. Write the time per optimizer step
       and the tokens per second per GPU as functions of
       $N_{\textrm{ranks}}$. For $t_{\textrm{ar}} = 2t_{\textrm{mb}}$, by
       what factor does the time to train on a fixed number of tokens
       shrink when the world size doubles from 32 to 64, and from 128 to
       256? What limits the speedup as the world size grows?
    1. A colleague moves the job from 64 to 128 GPUs and keeps
       $B_{\textrm{device}}$ and $N_{\textrm{accumulation}}$. How do the
       global batch and the number of optimizer steps for a fixed number of
       tokens change? For adaptive methods such as Adam,
       :numref:`sec_batch_size` scales the learning rate with the square
       root of the batch, $\eta(b) = \sqrt{b/b_0}\,\eta_0$, where $\eta_0$
       is tuned at batch size $b_0$. Which learning rate does this rule
       suggest if the larger batch is kept, and what happens to a warmup
       that is specified in steps?
   :exlabel:`ex_training-systems-global-batch`

1. [code] **Padded evaluation shards.** An evaluation set of 10,001
   examples is evaluated on 8 ranks, and rank $r$ iterates over
   `DistributedSampler(dataset, num_replicas=8, rank=r, shuffle=False)`.
   Because `num_replicas` and `rank` are passed explicitly, one process can
   construct the samplers of all ranks without starting a process group.
    1. Collect the indices of all 8 samplers for
       `dataset = list(range(10001))`. How many indices are there, which
       examples occur twice, and by how many percentage points can the
       duplicates shift the measured accuracy at most?
    1. Repeat with `drop_last=True`. Which examples are never evaluated?
    1. Write an exact evaluation in which the ranks partition the 10,001
       examples without padding, each rank counts its correct predictions
       and its examples, and the ranks combine the counts with
       `torch.distributed.all_reduce`. Test the reduction in one process by
       simulating the 8 ranks with random predictions and replacing the
       all-reduce by a sum over the ranks. Why is the mean of the 8
       per-rank accuracies not exact for these shards?
   :exlabel:`ex_training-systems-padded-evaluation`

1. [code] **Resume parity.** A resumed run is correct when it computes the
   same parameters as an uninterrupted one. Build a small experiment on the
   CPU: a two-layer MLP with dropout, trained on 1,024 random examples with
   20 features by Adam with a `LambdaLR` warmup schedule. The examples come
   from a `DataLoader` with batch size 32 and `shuffle=True`, whose
   shuffling is seeded per epoch by its own `torch.Generator`.
    1. Train for 64 steps without interruption. In a second run, train for
       40 steps, save a checkpoint, rebuild the model, optimizer,
       scheduler, and data loader from a different seed, restore the
       checkpoint, and train for the remaining 24 steps. Make the final
       parameters of both runs bitwise identical. Which object of this
       experiment holds each piece of state that
       :numref:`subsec_training_long_run` lists, and how does the restore
       resume the data loader in the middle of an epoch?
    1. Omit each piece of state from the restore in turn, and report the
       largest parameter difference after the 64 steps. Which omissions
       would show in a loss curve, and which would go unnoticed?
    1. Simulate a crash during `torch.save` by writing only the first half
       of a checkpoint's bytes to its final path, and load the file. Then
       write each checkpoint to a temporary file in the same directory and
       move it into place with `os.replace`. Why does the final path hold a
       complete checkpoint, the previous one or the new one, whenever the
       training process crashes during the second procedure? A power loss
       can leave the rename on disk before the file's contents. Which calls
       does the procedure need so that it also survives a power loss?
   :exlabel:`ex_training-systems-resume-parity`

1. **Optimizer offload.** ● ZeRO-Offload
   :cite:`Ren.Rajbhandari.Aminabadi.ea.2021` keeps 16-bit parameters on
   the GPU for the forward and backward passes, and keeps FP32 master
   parameters and Adam's two moments in host memory, where the CPU performs
   the update. Each step sends the 16-bit gradients to the host and the
   updated 16-bit parameters back. The paper uses FP16, which has the same
   size as BF16. Consider a 7B model on one GPU that
   sustains 400 TFLOP/s, connected over PCIe at 50 GB/s in each direction
   to a host whose memory delivers 200 GB/s.
    1. Compute the time per step of the two transfers, run one after the
       other.
    1. The update on the CPU is limited by memory traffic. Count the bytes
       per parameter that an Adam update reads and writes in this layout,
       and estimate the time of the update.
    1. A forward and backward pass costs about $6N$ FLOPs per token for a
       model with $N$ parameters (:numref:`sec_scaling-laws`). How many
       tokens must a step contain so that the transfers and the update
       take less than 20% of the step time when nothing overlaps?
    1. The one-step delayed parameter update of ZeRO-Offload computes the
       update from the gradients of step $k$ on the CPU while the GPU runs
       the forward and backward passes of step $k + 1$ with the parameters
       that the gradients of step $k - 1$ produced. Under which condition
       does the update then cost no time? What does the delay change about
       the optimization?
   :exlabel:`ex_training-systems-optimizer-offload`

1. **Resumption on fewer workers.** ● A job trains with FSDP on 64 ranks with
   $B_{\textrm{device}} = 4$ sequences and $N_{\textrm{accumulation}} = 4$,
   writes sharded checkpoints, and must resume on 48 ranks after losing two
   nodes.
    1. Sort the contents of a resumable checkpoint, as listed in
       :numref:`subsec_training_long_run`, into state that is the same on
       every rank and state that differs between ranks. For each per-rank
       item, state what the restore must do when the number of ranks
       changes.
    1. Show that no choice of $B_{\textrm{device}}$ and
       $N_{\textrm{accumulation}}$ on 48 ranks reproduces the global batch
       of 1,024 sequences. Compare two remedies, a different global batch
       and unequal microbatches across the ranks, by what each changes
       about the optimization and what each requires of the loss
       normalization.
    1. Design a data order for which the resumed run sees every remaining
       example of the epoch exactly once, for any number of ranks. Why is it
       insufficient to store each rank's position in its own shard?
   :exlabel:`ex_training-systems-elastic-resume`

1. [extended] **Pipeline schedules.** ●● A pipeline of $p$ stages trains on
   $m$ microbatches per optimizer step. The forward pass of one microbatch
   through one stage takes time $t_f$, and its backward pass takes
   $t_b = 2t_f$. A stage can run the forward pass of microbatch $j$ once the
   previous stage has finished it, and the backward pass of $j$ once the
   next stage has finished its backward pass of $j$. A stage holds the
   activations of $j$ from its forward pass until its backward pass. GPipe
   :cite:`Huang.Cheng.Bapna.ea.2019` runs all forward passes before all
   backward passes, while the 1F1B schedule of
   :citet:`Narayanan.Shoeybi.Casper.ea.2021` runs a few warm-up forward
   passes and then alternates one forward and one backward pass. The
   bubble fraction of a schedule is its step time minus the ideal time
   $m(t_f + t_b)$, divided by the ideal time.
    1. Write a simulator that takes the order of the operations on each
       stage, runs each operation as soon as its stage is free and its
       inputs are ready, and returns the step time and, for each stage, the
       largest number of microbatches whose activations it holds at once.
    1. Generate GPipe and 1F1B orders for any $p$ and $m$, deriving the
       number of warm-up forward passes that each stage of 1F1B needs. Run
       them for $p \in \{2, 4, 8\}$ and $m \in \{1, 4, 8, 32\}$. From the
       results, derive the bubble fraction and the activation memory of
       each schedule as functions of $p$ and $m$.
    1. In the interleaved assignment of
       :citet:`Narayanan.Shoeybi.Casper.ea.2021`, each device holds $v$
       nonconsecutive chunks of layers. A microbatch then passes through
       $pv$ stages and visits every device $v$ times, each at $1/v$ of the
       cost. Simulate it for $v \in \{1, 2, 4\}$ with all forward passes
       first. How do the bubble fraction and the number of point-to-point
       transfers per step depend on $v$?
    1. With data parallelism of degree $d$ around the pipeline and
       microbatches of $b$ sequences, the global batch of 1,024 sequences of
       exercise :exref:`ex_training-systems-global-batch` gives
       $m = 1024/(db)$. For $p = 8$ and $b = 1$, compute the bubble
       fraction of 1F1B at $d = 8$, 16, and 64. What limits the
       data-parallel degree of a pipelined job at a fixed global batch?
   :exlabel:`ex_training-systems-pipeline-schedules`

[Discussions](https://d2l.discourse.group/t/36846)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §39.6]{.kicker}

Distributed Model Training: DDP, FSDP/ZeRO, 3D Parallelism, and Fault Tolerance<br>**scaling ladder · DDP vs. FSDP2 · tensor & pipeline parallelism · global batch invariant · memory sharding · ZeRO-Offload · atomic checkpoints**
:::
:::

::: {.slide title="Why Single-Accelerator Training Hits a Wall"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
- **The 7B single-GPU memory wall ($112\text{ GB}$):** Mixed-precision Adam holds BF16 weights ($14\text{ GB}$), BF16 grads ($14\text{ GB}$), FP32 master weights ($28\text{ GB}$), and FP32 Adam $m, v$ ($56\text{ GB}$)—exceeding an $80\text{ GB}$ H100 before any activations.
- **Four partitioning axes:** Split the batch (DDP), training state (FSDP/ZeRO), layer matrices/stages (TP/PP), or sequence/experts (CP/EP).
- **Golden rule:** Start from a verified 1-GPU baseline and climb one rung at a time.
:::

::: {.col .fig .big}
@fig:ch39-tools-7b-wall
:::
:::
:::

::: {.slide title="The Distributed Scaling Ladder"}
[Progressive Scaling]{.kicker}

Climb the scaling ladder in strict order of increasing communication volume and engineering complexity:

::: {.cols .vc}
::: {.col}
1. **Single Accelerator Baseline:** Verify loss convergence, tok/s throughput, peak memory, and checkpoint restore.
2. **Data Parallelism (DDP):** Replicate the model on every GPU, split the batch, and `all_reduce` gradients once per step. Always use DDP while state fits on one GPU.
3. **Fully Sharded Data Parallelism (FSDP / ZeRO-3):** Shard parameters, gradients, and optimizer states across $W$ GPUs; `all_gather` layers on the fly ($\sim 1.5\times$ DDP traffic).
4. **Composed 3D/4D Parallelism (TP / PP / CP / EP):** Split individual layers, layer stages, sequences, or experts when a single layer or activation exceeds device capacity.
:::

::: {.col .fig}
![](../img/tools-training-ladder.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="Data Parallelism (DDP) vs. FSDP2 / NamedSharding"}
[Rungs 2 and 3]{.kicker}

::: {.cols .vc}
::: {.col}
- **Rung 2: DDP (`DistributedDataParallel`):** Replicates all $16N$ state on every GPU and averages bucketed gradients via `all_reduce` ($\approx 2S$ traffic); best choice whenever state fits.
- **Rung 3: FSDP2 (`fully_shard` / `DTensor` & JAX `NamedSharding`):** Divides parameters, gradients, and Adam moments by $W$ ($16N/W$), overlapping per-layer `all_gather` and `reduce_scatter` ($\approx 3S = 1.5\times$ DDP traffic).
:::

::: {.col .fig .big}
@fig:ch39-tools-ddp-vs-fsdp2
:::
:::
:::

::: {.slide title="Tensor, Pipeline, Context, and Expert Parallelism"}
[Rung 4: Model & Sequence Axes]{.kicker}

::: {.cols .vc}
::: {.col}
- **Intra-layer vs. inter-layer splits:** Tensor Parallelism (TP) splits matrices inside every block (2 collectives/layer on the critical path), whereas Pipeline Parallelism (PP) passes activations point-to-point at stage boundaries.
- **Hardware topology rule (*Ultra-Scale Playbook*):** Keep high-frequency **TP inside an 8-GPU NVLink node** ($450\text{--}900\text{ GB/s}$; stretching TP across nodes costs $-43\%$), and scale **PP + FSDP/DDP across nodes**.
:::

::: {.col .fig .big}
@fig:ch39-tools-node-topology-mapping
:::
:::
:::

::: {.slide title="Pipeline Schedules: GPipe vs. 1F1B Bubble and Memory"}
[Pipeline Mechanics]{.kicker}

::: {.cols .vc}
::: {.col}
$$\text{Bubble Fraction} = \frac{p - 1}{m}$$

- **GPipe (all $m$ forwards, then all $m$ backwards):** Needs $m \gg p$ to shrink the bubble, forcing stage 1 to hold **all $m$ microbatch activations at once** ($O(m)$ memory).
- **1F1B Schedule:** Alternates 1 Forward and 1 Backward after warmup, achieving the same $(p-1)/m$ bubble while capping live activations at **$\le p$ microbatches**.
:::

::: {.col .fig}
@fig:ch39-tools-gpipe-vs-1f1b
:::
:::
:::

::: {.slide title="The Global Batch Size Invariant Across Every Rung"}
[Optimization Semantics]{.kicker}

::: {.cols .vc}
::: {.col}
$$B_{\text{global}} = B_{\text{device}} \cdot N_{\text{ranks}} \cdot N_{\text{accumulation}}$$

- **Preserving optimization dynamics:** Doubling $N_{\text{ranks}}$ from $64 \to 128$ without halving $N_{\text{accumulation}}$ silently **doubles $B_{\text{global}}$** and halves optimizer steps.
- **When $B_{\text{global}}$ increases:** Scale Adam learning rate by $\sqrt{B/B_0}$ and define warmup in **tokens** rather than steps.
:::

::: {.col .fig}
@fig:ch39-tools-global-batch-invariant
:::
:::
:::

::: {.slide title="Distributed Samplers and Exact Metric Aggregation"}
[Correctness Pitfalls]{.kicker}

::: {.cols .vc}
::: {.col}
$$\text{Acc}_{\text{exact}} = \frac{\sum_{r=0}^{W-1} c_r}{\sum_{r=0}^{W-1} n_r} \neq \frac{1}{W} \sum_{r=0}^{W-1} \frac{c_r}{n_r}$$

- **Padded `DistributedSampler` bug:** Default padding repeats leading examples so every rank gets $\lceil N/W \rceil$ items, skewing validation metrics.
- **Exact distributed evaluation:** Split examples without padding, tally local `(correct_r, count_r)` integers, and `all_reduce(SUM)` numerator and denominator once.
:::

::: {.col .fig}
@fig:ch39-tools-eval-sampler-pitfalls
:::
:::
:::

::: {.slide title="Anatomy of Peak Training Memory"}
[Memory Budget]{.kicker}

Peak per-GPU memory is the sum of model state, saved activations, temporary communication buffers, and allocator fragmentation:

::: {.cols .vc}
::: {.col}
- **Mixed-precision Adam state for a 7B model ($16\text{ B/param} = 112\text{ GB}$):**
  - BF16 parameters: $2N = 14\text{ GB}$
  - BF16 gradients: $2N = 14\text{ GB}$
  - FP32 master parameters: $4N = 28\text{ GB}$
  - FP32 Adam first & second moments ($m, v$): $8N = 56\text{ GB}$
- **Dynamic terms:** Activations grow linearly with $B_{\text{device}} \times T$, while collective staging buffers add $4\text{--}8\text{ GiB}$.
:::

::: {.col .fig}
![](../img/tools-training-memory.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="Modeling Per-Device Memory: DDP vs. FSDP" layout="tight"}
[Executable Memory Model]{.kicker}

Even leaving out the $28\text{ GB}$ master-weight copy ($84\text{ GiB}$ state $+ 18\text{ GiB}$ activations $+ 5\text{ GiB}$ temporary buffers), our sizing script shows why DDP cannot fit a 7B full fine-tune on 80 GB GPUs:

::: {.cols .vc}
::: {.col}
- **1 GPU plain ($107.0\text{ GiB}$):** Exceeds an $80\text{ GB}$ H100.
- **8-GPU DDP + activation checkpointing ($95.3\text{ GiB/GPU}$):** Checkpointing shrinks activations from $18\text{ GiB}$ to $6.3\text{ GiB}$, but DDP **replicates all $84\text{ GiB}$ of model state** on every GPU—still OOM.
- **8-GPU FSDP + activation checkpointing ($21.8\text{ GiB/GPU}$):** Sharding state by $8$ cuts per-GPU state to $10.5\text{ GiB}$, leaving nearly $60\text{ GiB}$ of headroom for larger microbatches.
:::

::: {.col .fig}
@!training-systems-memory-model
:::
:::
:::

::: {.slide title="Modeling Per-Device Memory: DDP vs. FSDP (Code)" layout="tight"}
[Executable Memory Model]{.kicker}

@-training-systems-memory-model

:::

::: {.slide title="Five Orthogonal Memory-Reduction Techniques"}
[Memory Optimization Trade-Offs]{.kicker}

::: {.cols .vc}
::: {.col}
- **Sharding & Checkpointing:** FSDP2/ZeRO divides $16N$ state by $W$ ($\sim 1.5\times$ comm), while activation checkpointing trades $+25\text{--}33\%$ FLOPs to keep only each block's input (still $O(L)$, but only $d$ values per token per block).
- **Accumulation, Offload & Adapters:** Gradient accumulation shrinks microbatch $B_{\text{device}}$, CPU offload moves master weights/Adam across PCIe, and LoRA/QLoRA freezes base weights to drop $>98\%$ of gradient/optimizer state.
:::

::: {.col .fig .big}
@fig:ch39-tools-five-memory-knobs
:::
:::
:::

::: {.slide title="Arithmetic of CPU Optimizer Offload (ZeRO-Offload)"}
[Host-Device Roofline]{.kicker}

::: {.cols .vc}
::: {.col}
- **Per-step offload cost ($1.54\text{ s}$ for 7B):** Sending $14\text{ GB}$ BF16 grads and receiving $14\text{ GB}$ BF16 weights over $50\text{ GB/s}$ PCIe takes $0.56\text{ s}$, plus $0.98\text{ s}$ to stream $196\text{ GB}$ ($28\text{ B/param}$) through $200\text{ GB/s}$ CPU DDR.
- **Amortization & delayed update:** Keeping offload $< 20\%$ of step time at $400\text{ TFLOP/s}$ requires $\ge 58{,}700\text{ tok/step}$ (or a 1-step delayed CPU update).
:::

::: {.col .fig .big}
@fig:ch39-tools-zero-offload-pipeline
:::
:::
:::

::: {.slide title="Selecting the Simplest Tool for Your Scale (Nov 2026)"}
[Software Ecosystem]{.kicker}

::: {.cols .vc}
::: {.col}
- **1 GPU to 1 Node (2–8 GPUs):** Use **Unsloth** or **HF PEFT** for single-GPU LoRA/QLoRA, and **HF Accelerate**, **Axolotl**, or **LLaMA-Factory** for single-node DDP/FSDP2.
- **Multi-Node & RL Post-Training:** Use **PyTorch FSDP2 (`fully_shard`)** or **DeepSpeed** across a few nodes, **TorchTitan / Megatron Core / MaxText** at 64–512+ GPUs, and **TRL / veRL / OpenRLHF** for DPO/GRPO.
:::

::: {.col .fig .big}
@fig:ch39-tools-scale-tool-selector
:::
:::
:::

::: {.slide title="Fault Tolerance: Complete and Atomic Checkpointing"}
[Surviving Hardware Failures]{.kicker}

::: {.cols .vc}
::: {.col}
- **Bitwise-complete checkpoint bundle:** Save sharded model weights, Adam moments, **LR scheduler state**, step/token counters, deterministic **data cursor**, and per-rank **PRNG states**.
- **Crash-safe atomic rename:** Never overwrite `ckpt` in place; write to `ckpt.tmp`, call `f.flush()` + `os.fsync()`, swap via **`os.replace("ckpt.tmp", "ckpt")`**, and `fsync` the directory.
:::

::: {.col .fig .big}
@fig:ch39-tools-atomic-checkpoint
:::
:::
:::

::: {.slide title="Operational Discipline for Long Distributed Runs"}
[Production Operations]{.kicker}

Four operational habits separate multi-week training runs that finish from runs that stall or silently diverge:

::: {.cols}
::: {.col}
- **1. Drill the recovery before day one:**
  - Deliberately kill the training job at step 40, restore on fresh workers, and verify that step 41's loss, gradient norm, and data batch match an uninterrupted run bitwise.
  - Design data ordering globally by example index so **elastic resumption** works even if you must resume on 48 GPUs after losing two 8-GPU nodes.
:::

::: {.col}
- **2. Profile input pipelines separately from GPU compute:**
  - Measure storage reads, image/text decoding, and host-to-device copy rates independently; adding GPUs to an I/O-bound job only increases storage contention.
- **3. Monitor convergence alongside systems throughput:**
  - Log `(step, loss, grad_norm, tok/s, collective_ms, ckpt_ms)` with **rank and hostname** on every log line; isolate bugs on $1\text{ process}$ $\to$ $1\text{ node}$ $\to$ $N\text{ nodes}$.
:::
:::
:::

::: {.slide title="Concept Check: Sharding, Batch Invariants, and Recovery"}
[Check Your Understanding]{.kicker}

Work through these three distributed training scenarios:

1. A pretraining job maintains a fixed global batch of $B_{\text{global}} = 1{,}024$ sequences (4,096 tokens each $\approx 4.2\text{M}$ tokens), and each GPU holds at most $B_{\text{device}} = 4$ sequences per microbatch. What is the minimum gradient accumulation $N_{\text{accumulation}}$ on $N_{\text{ranks}} = 64$ GPUs, and what is the largest world size that can train this global batch with data parallelism alone?
2. During checkpoint restore, a colleague restores the model weights, Adam moments, and data-loader position, but forgets to restore the `LambdaLR` scheduler state. What happens to the training run, and will it raise an exception?
3. Why does FSDP2 communicate roughly $1.5\times$ as much data per step as DDP?
:::

::: {.slide title="Concept Check: Sharding, Batch Invariants, and Recovery (cont.)"}
[Check Your Understanding]{.kicker}

::: {.d2l-note}
**Answers:**

1. On 64 GPUs with $B_{\text{device}} = 4$, $N_{\text{accumulation}} = 1024 / (64 \times 4) = \mathbf{4}$. The largest pure data-parallel world size (at $B_{\text{device}} = 1, N_{\text{accumulation}} = 1$) is **$N_{\text{ranks}} = 1{,}024$ GPUs** (or $256$ GPUs if keeping $B_{\text{device}} = 4$).
2. **It never raises an exception**, but the learning rate resets to step 0 (re-running warmup or jumping back to peak $\eta_{\max}$), **causing a sudden loss spike or silent accuracy degradation**.
3. For $N$ parameters of size $S = 2N$ bytes, DDP runs one `all_reduce` of gradients ($\approx 2S$ bytes sent/received per GPU). FSDP runs an `all_gather` of weights in forward ($S$), another `all_gather` of weights in backward ($S$), and a `reduce_scatter` of gradients ($S$)—totaling **$3S$ bytes ($1.5\times$ DDP)**.
:::
:::

::: {.slide title="Summary: Multi-GPU and Distributed Training Systems"}
[Key Takeaways]{.kicker}

| System Dimension | Core Rule / Equation | Practical Engineering Guideline |
| :--- | :--- | :--- |
| **Scaling ladder** | 1 GPU $\to$ DDP $\to$ FSDP2 $\to$ TP + PP + CP/EP | Use DDP while state fits; shard state with FSDP2 next; keep TP inside an NVLink node |
| **Global batch size** | $B_{\text{global}} = B_{\text{device}} \cdot N_{\text{ranks}} \cdot N_{\text{accumulation}}$ | Adjust $N_{\text{accumulation}}$ or $\eta(B)$ whenever changing GPU count so optimization dynamics stay invariant |
| **Memory accounting** | $M_{\text{state}} = 16N\text{ B}$ (BF16 weights/grads + FP32 master/Adam) | Combine FSDP ($\div W$), activation checkpointing, and LoRA/QLoRA deliberately to fit target VRAM |
| **Pipeline scheduling** | Bubble fraction $= (p - 1)/m$ | Use **1F1B** instead of GPipe to bound in-flight activation memory to $p$ microbatches |
| **Distributed evaluation** | $\text{Acc} = (\sum_r c_r) / (\sum_r n_r)$ | Avoid padded sampler duplicates on eval sets; `all_reduce` numerator and denominator counts separately |
| **Fault tolerance** | Atomic write: `tmp` $\to$ `fsync` $\to$ `os.replace` | Save model, optimizer, scheduler, data cursor, and RNG state; drill recovery before long or spot runs |
:::
