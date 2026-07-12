# GPUs, Devices, and Memory
:label:`sec_use_gpu`

> **Role.** Modernization of the current use-gpu section, which is the most
> structurally load-bearing file in the chapter: it `#@save`s
> `d2l.cpu/gpu/num_gpus/try_gpu/try_all_gpus` **and** monkey-patches
> `d2l.Trainer` with its real device-placement logic — every
> `Trainer(num_gpus=1)` call from Chapter 7 onward depends on this file.
> Both are preserved verbatim in role (same names, same `#@save`s, same
> `add_to_class` patch). What changes: dated prose and claims go, and the
> section gains the memory-management material a single-GPU builder needs.
> Anything multi-device stays in the computational-performance chapter.

## Devices **[MOD]**

*Topics.* The device as a tensor property; `torch.device`,
`cuda:0`/`cuda:1`, and (new, one paragraph) the wider 2026 device zoo —
`mps` on Apple silicon, `xpu`, TPUs via JAX — with the book standardizing
on CPU + CUDA. The `try_gpu`/`try_all_gpus` helpers kept exactly (same
`#@save`). **Cut:** `mxnet-cu100` install instructions, "GPU performance
×1000/decade", the AWS-centric framing; replaced by a pointer to the
appendix for procurement and to `nvidia-smi` for inventory.

*Code (PyTorch).* As current: `d2l.cpu()`, `d2l.gpu()`, `d2l.num_gpus()`,
`try_gpu`, `try_all_gpus` — all `#@save`d here, **byte-identical** blocks
(shards fingerprint content; any edit recaptures the 28/18/25 downstream
files that use them, ×4 frameworks). New *prose only*: note that PyTorch
has since shipped this exact idea natively as `torch.accelerator`
(`device_count()`, `current_accelerator()`, verified in 2.11) — the book's
helpers predate it and stay because they give all four tabs one vocabulary
plus graceful CPU fallback, but the reader should know the native name.
Drop the mxnet-only `Module.set_scratch_params_device` `#@save` (zero
downstream users — dead weight); keep its logic local to the cell that
needs it if any.

## Tensors, Models, and Devices **[KEPT]**

*Topics.* Storage on a device; computation requires operands co-located;
the device-mismatch error read out loud (it is the most common error
message in a beginner's GPU life); explicit copies (`.to`, `cuda(i)`),
no-op copies, and the cost model — cross-device copies are slow relative
to compute, transfer is the thing to minimize. `net.to(device)` for
models. The side figure (`fig_copyto`) survives. **New, one cell:** pinned
memory + `non_blocking=True` as the standard async host→device transfer
pattern (what `DataLoader(pin_memory=True)` is for) — two sentences of
why, one cell of how.

*Code (PyTorch).* Current progression (X on gpu0, Y on gpu1, copy, add)
trimmed to gpu0-only where two GPUs aren't detected; pinned-transfer
timing cell.

## GPU Memory **[NEW]**

*Topics.* The single-GPU builder's real daily problem: fitting. The
allocator model in one paragraph (reserved vs allocated; why `nvidia-smi`
disagrees with `memory_allocated()`); measuring —
`torch.cuda.memory_allocated/reserved/max_memory_allocated`,
`empty_cache()` and what it does/does not do; the lifecycle of an OOM —
what actually occupies memory during training: weights + grads + optimizer
state (the arithmetic of :numref:`sec_parameters`, now measured
empirically) + **activations**, and why activations scale with batch size
while the rest does not.

*Code (PyTorch).* Instrumented mini-loop printing `memory_allocated()`
after model creation, after forward, after backward, after `opt.step()` —
the four plateaus map one-to-one onto the accounting table, closing the
loop between predicted and measured bytes.

## Trading Compute for Memory: Activation Checkpointing **[NEW]**

*Topics.* `torch.utils.checkpoint`: don't store intermediate activations,
recompute them during backward — ~30–40% more compute for a large cut in
activation memory; when it pays (deep stacks of identical blocks — exactly
the config-built residual stack of this chapter, and exactly how large
transformer training always runs). One measured comparison, not a survey.

*Code (PyTorch).* Wrap the residual stack's blocks in
`checkpoint(block, x)`; report max memory and wall-clock with/without at a
batch size where the difference is unambiguous.

## Don't Break the Pipeline **[KEPT]**

*Topics.* The async-execution/sync-point lesson from the current section —
kept nearly as-is because it is correct and permanently useful: kernels are
queued asynchronously; `.item()`, `.numpy()`, printing, and logging force
synchronization; move logging out of the hot loop. Framed slightly more
generally as "the GPU runs ahead of Python; do not make it wait."
Forward pointer to the computational-performance chapter for the full
async/parallelism treatment.

## The Trainer, Now with Devices **[KEPT — structural]**

*Topics.* Redeem the promise from :numref:`sec_oo-design`: the
`@d2l.add_to_class(d2l.Trainer)` patch giving `__init__`/`prepare_batch`/
`prepare_model` real GPU placement. Preserved with the same semantics and
`#@save`s as today (this is the book-wide dependency); prose refreshed to
connect with the memory material above (the Trainer places the model
*once*; batches stream per step — now the reader knows what each costs).

*Code (PyTorch).* The existing patch cells, essentially verbatim; train
the chapter's MLP with `Trainer(max_epochs=1, num_gpus=1)` as the
capstone.

## Summary and Exercises

*Exercises (sketch).* (1) Predict max memory for batch sizes 64/256/1024
using the accounting model, then measure — where does the model break down?
(2) Find the batch size at which checkpointing changes an OOM into a run.
(3) Time the training loop with a `print(loss.item())` every step vs every
epoch; explain the difference in sync terms. (4) Kept from current: measure
compute vs communication for large matrix ops across devices (guarded to
run only when 2 GPUs are present).

> **Downstream constraints (hard).** Must keep, with identical names and
> `#@save` provenance: `cpu()`, `gpu()`, `num_gpus()`, `try_gpu()`,
> `try_all_gpus()`, and the `add_to_class(d2l.Trainer)` device patch —
> `multiple-gpus*.md` calls the helpers directly and every downstream
> `Trainer(num_gpus=1)` cell needs the patch. Label `sec_use_gpu` is cited
> by `ndarray.md` and `multiple-gpus.md` — keep on promotion. Two-GPU cells
> must degrade gracefully on the 1-GPU/CPU render path (the store/gate
> already handles capability tiers).

## Framework Coverage

*All four tabs keep their existing `#@save` device helpers and `Trainer`
patches verbatim.*

- **JAX** — full coverage: `jax.checkpoint`/`remat` is the first-class
  activation-checkpointing equivalent (verified through `jax.grad`);
  the sync-point lesson maps to `block_until_ready()` (verified with a
  dispatch-vs-completion timing probe); pinned memory reframes as
  "dispatch is async by default" (ALT, no per-tensor flag);
  `device.memory_stats()` reports live and peak bytes on GPU and returns
  `None` on CPU (verified on the GPU build).
- **TensorFlow** — full coverage: `tf.recompute_grad` verified;
  `tf.config.experimental.get_memory_info('GPU:0')` exists (correctly
  errors on CPU); pinned-memory analogue is `tf.data`'s
  `prefetch(AUTOTUNE)` (ALT framing); sync lesson unchanged
  (`.numpy()` forces sync).
- **MXNet** — two documented reductions and one strong lesson. **SKIP:
  activation checkpointing** (no recompute utility anywhere in the wheel —
  grepped the tree); **PARTIAL: memory introspection** —
  `gpu_memory_info()` is device-wide (`cudaMemGetInfo`), no per-process
  allocator counters, so the four-plateau demo becomes a coarser
  before/after measurement. Pinned memory DIRECT (`mx.cpu_pinned()`,
  no `non_blocking` keyword). The strongest lesson is the
  sync-point lesson via `npx.waitall()` is *mxnet's strongest material*,
  already written and green in `chapter_computational-performance/` —
  reuse near-verbatim. Note the 2.0 rename `context`→`device` (Context is
  a deprecated shim).
