# Writing pass: GPUs, Devices, and Memory (5.7)

Source: `chapter_builders-guide/gpus-devices-memory.md`. Prose only; no code
cell, label, figure, or slide changed (`check_prose_only.py`: OK;
`lint_source.py`: clean).

Prose words (all four tabs, checker count): 7286 before, 7042 after (-244).
Most of the reduction is tab prose in "Devices and Placement", "Transfers
from the Host", and "Device-Aware Trainer". The twelve-cell placement tour
itself is unchanged, because shortening it means removing cells.

## Before and after

1. Opening.
   - Before: "The models of the earlier chapters were small enough that it
     hardly mattered where they ran. A GPU computes much faster than a CPU: a
     current desktop GPU multiplies two $4096 \times 4096$ matrices in a few
     milliseconds. But it has its own memory, 24 GB on such a card, and every
     array it computes with must be there."
   - After: "Nobody was harmed: the models of the earlier chapters were small
     enough that it hardly mattered where they ran. For larger models it
     matters a great deal. A current desktop GPU multiplies two
     $4096 \times 4096$ matrices in a few milliseconds, far faster than a CPU.
     Unfortunately the speed comes with a constraint: the GPU has its own
     memory, 24 GB on such a card, and every array it computes with must be
     there."
2. The $16N$ count in the opening.
   - Before: "Training it with Adam in fp32 needs 128 GB: 16 bytes per
     parameter for the weights, their gradients, and the two moment estimates
     of Adam, before a single activation is stored."
   - After: "Training it with Adam in fp32 needs 128 GB by the count of
     :numref:`sec_parameters`, before a single activation is stored."
3. Device objects.
   - Before: "To see which GPUs a machine has, run `nvidia-smi` in a shell; it
     lists every card, its memory, and the processes using it."
   - After: the same, followed by "Each framework has its own names for
     devices. The names are soon forgotten; what matters in this part of the
     section is a single fact, which we reach in a moment: an operation runs
     on one device, and all its operands must be there."
4. Synchronization points.
   - Before: "Python waits only when it needs a value: converting a result to
     a Python number or a NumPy array, printing it, or branching on it in an
     `if`."
   - After: "So when does Python wait at all? Only when it needs a value:
     converting a result to a Python number or a NumPy array, printing it, or
     branching on it in an `if`."
5. Timing rules.
   - Before: "The cell above follows the first three rules; the memory
     measurements below follow all of them."
   - After: "The cell above warms up, waits, and reads `perf_counter`, but it
     reports the total for its 32 products without dividing; the timings later
     in this section follow all four rules."
6. Memory that grows from step to step.
   - Before: "a common cause is appending the loss array itself, rather than
     its value, to a Python list."
   - After: "The usual suspect is a Python list of losses: a loss that still
     carries its computation graph keeps the activations of its step alive
     with it, so detach the loss before storing it."
7. Cost of checkpointing.
   - Before: "The price is a second forward pass through every segment,
     :numref:`fig_bg_activation_checkpoint`. Since a backward pass costs about
     twice a forward pass, a step then costs about four forward passes instead
     of three, a third more, although the exact overhead depends on the layers
     and the hardware."
   - After: "The saving costs computation, namely a second forward pass
     through every segment, :numref:`fig_bg_activation_checkpoint`. How much
     longer does a step take? A backward pass costs about twice a forward
     pass: for a linear layer it computes two matrix products, one for the
     gradient of the weights and one for the gradient of the input, where the
     forward pass computes one. A step then costs about four forward passes
     instead of three, a third more. This is a rather crude count, and the
     exact overhead depends on the layers and the hardware."
8. Closing.
   - Before: "What the `Trainer` cannot do is tell whether a model fits before
     it runs out of memory. That estimate is the one we made at the start of
     this section, from the bytes per parameter and the size of the
     activations."
   - After: "It is a small model, and it fits with room to spare. And the
     8-billion-parameter model of the opening? A smaller minibatch and
     activation checkpointing shrink its activations, but its 128 GB of
     weights, gradients, and optimizer state are needed before the first
     activation is stored, and no setting of this section touches them. Note
     that the `Trainer` would not have warned us: it cannot tell whether a
     model fits before it runs out of memory. That estimate we still make by
     hand, from the bytes per parameter and the size of the activations.
     Training on more than one device is taken up in
     :numref:`chap_performance`."

## Repetition removed

- **$16N$ accounting** (owned by 5.2, `sec_parameters`): the opening now
  cites the count in one clause; "What Fills Memory During Training" names
  the split ($4N$, $4N$, $8N$) in one sentence with the reference, because
  the readings that follow are read against those terms; the Summary no
  longer repeats "$16N$ bytes for $N$ parameters with Adam in fp32".
- **"The `Trainer` accepted `num_gpus` and did not use it"**: was the first
  sentence of the section and again the first sentence of each of the four
  tabs under "Device-Aware Trainer". Now one shared sentence there; the tabs
  hold only what differs.
- **Identical sentences in every tab** moved to shared prose, once: "The
  property `device` tells where an array lives"; "Creating data directly on
  the device ... never occupies main memory or crosses the bus"; "Whether a
  computation fits depends on its *peak*"; the no-op move ("What if `Z`
  already lives on the target device?", four times) is one shared sentence.
- **"stands for all CPU cores and all of main memory"** (four tabs): the
  shared definition of a device already says it.
- **Late-binding closure explanation** in the PyTorch and TensorFlow tabs:
  one clause in PyTorch; removed in TensorFlow, whose code comment says it.
- **Pointer to `chap_performance`** for transfers appeared as a detached
  closing line; folded into the subsection's opening paragraph.

## Review findings

Applied: GPU-2 (one clause and a reference); GPU-3 (tab prose only; naming
paragraphs cut to a sentence or two, MXNet context/device history shortened);
GPU-4 (two sentences); GPU-5 (reduced to one shared sentence; the cell
stays); GPU-8 (says which rules the cell follows); GPU-9 (prose only: the
PyTorch `devices` tab removed, since the docstring says it; JAX tab cut to
one sentence; the text now admits that this section does not use the class);
GPU-10 (prose only: no longer claims a measured overlap benefit, says what
the cell measures, drops the undefined "streams", three no-code tabs cut to
one sentence each); GPU-11 (PyTorch, TensorFlow, and JAX tabs say that the
remaining 128 to 136 MiB are `A` and `B`); GPU-12 (detach); GPU-13; GPU-14;
GPU-15 (prose shortened; a code comment would be a code change); GPU-16
(change log cut; cuDNN reduced to one sentence; shared opening sentence);
GPU-18 (the MXNet "measures nothing" tab shortened; the pattern itself
remains). Structural finding 3 (the opening question is now answered in the
closing paragraph).

Already addressed in the file: GPU-7 (figure and caption say op1 to op4;
kept).

Deferred:
- GPU-1: needs a measured CPU time for the same product.
- GPU-6: needs a measured transfer time.
- GPU-17 and structural 5: need a CPU against GPU timing run.
- Structural 1 (split into two sections) and 2 (cut cells of the placement
  tour): move or remove code cells.
- Structural 4 and the rest of GPU-9 (use `Benchmark` in the checkpointing
  cell, or define it later): code change.
- GPU-15, GPU-16 remainder (move detail into code comments): code change.
- Exercises: not touched.

## Factual claims added or changed

- "far faster than a CPU" for the $4096 \times 4096$ product (was "much
  faster"); still without a CPU number.
- "an operation runs on one device, and all its operands must be there"
  (worded to hold for TensorFlow, which copies operands silently).
- The queue cell "reports the total for its 32 products without dividing"
  and uses `perf_counter` (was: "follows the first three rules").
- "This section does not use `Benchmark`" (its timing cells write the loop).
- Pinned memory: "The cell waits for the copies before it stops the clock,
  so it measures the copies themselves and not what a non-blocking copy is
  for"; "somewhat faster" (stored: 0.55 against 0.37 ms). "This is a
  refinement for input pipelines."
- PyTorch: "Most of what remains allocated is the two matrices `A` and `B`
  of the queue experiment, 64 MiB each" (stored 136.1 MiB; I did not account
  for the other 8 MiB). TensorFlow and JAX: "to the 128 MiB of the matrices
  `A` and `B`" (stored 128.0 in both).
- Leak diagnosis: a stored loss that still carries its computation graph
  keeps its step's activations alive; detach before storing.
- Backward pass about twice a forward pass because a linear layer computes
  two matrix products on the way back and one on the way forward.
- "The name has nothing to do with the checkpoint files of
  :numref:`sec_read_write`."
- Closing: a smaller minibatch and checkpointing shrink only the activations
  of the 8B model; the 128 GB are untouched by anything in this section. The
  reference to `chap_performance` says only that it takes up training on
  more than one device; I did not claim that it makes the 8B model fit
  (data-parallel training does not).
- Run-time numbers unchanged: 1.1 GiB to 0.4 to 0.5 GiB, about a quarter
  longer (stored: PyTorch 25%, TensorFlow 23%, JAX 20%).

## Slides against the body

No statement in the slides contradicts the revised body. The slides still
spell out "16 bytes per parameter" several times and "about a third more
work"; both agree with the body. The body now calls the count of forward
passes "rather crude"; the slide gives it without qualification.
