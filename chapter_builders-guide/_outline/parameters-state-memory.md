# Parameters, State, and Memory
:label:`sec_parameters_v2`

> **Role.** Extends the current "parameters" section from *access mechanics*
> to the full 2026 question: what state does a model carry, and what does it
> cost? Adds buffers, memory accounting, freezing, and realistic weight
> tying. This is the section the planned transformer/generative chapters
> will lean on hardest (runtime KV caches, VRAM budgets, embedding tying,
> partial fine-tuning).

## Accessing Parameters **[KEPT]**

*Topics.* Targeted access (`net[2].weight`, `.bias.grad`), bulk traversal
(`named_parameters()`), nested traversal on a module of modules; the
`state_dict()` view of the same object graph (names are paths). Optimizers,
checkpoints, and debuggers traverse this graph, including aliases.

*Code (PyTorch).* Same progression as the current section (inspect one
layer, then all, then a nested net), tightened; `state_dict().keys()` shown
next to `named_parameters()` so the naming scheme is learned once.

## Parameters vs Buffers **[NEW]**

*Topics.* Not all state is trainable. `register_buffer`: tensors that move
with `.to(device)`, appear in `state_dict()`, but receive no gradient.
Canonical persistent examples include BatchNorm running statistics (Chapter
8) and precomputed tables. Distinguish them from per-request state such as a
KV cache, which follows computation but does not belong in a model checkpoint;
causal masks are often recomputed or registered non-persistently. Rule of
thumb: parameter if optimized, persistent buffer if checkpointed, explicit
runtime cache if request-scoped, plain attribute for reconstructible data.

*Code (PyTorch).* A module with a precomputed constant table as a buffer;
show it in `state_dict()`, show `.parameters()` skips it, move the module
to GPU and note the buffer moved too.

## Counting Parameters, Counting Bytes **[NEW]**

*Topics.* The "will it fit?" arithmetic every builder does before training:
parameter count → bytes at a given dtype → gradient copy → optimizer state
(Adam: two extra fp32 moments) → a stated-dtype multiplier over raw weights;
activations as the remaining (batch-dependent) term, with the full
treatment deferred to :numref:`sec_use_gpu_v2`. Worked numbers on the
book's own MLP, then scaled to a 1B-parameter model to establish the
batch-independent floor; activations may still dominate.

*Code (PyTorch).*

```python
n = sum(p.numel() for p in net.parameters())
bytes_fp32 = 4 * n            # weights
adam_state = 8 * n            # exp_avg + exp_avg_sq, fp32
grads = 4 * n
print(f'{n/1e6:.1f}M params, {(bytes_fp32+adam_state+grads)/2**30:.2f} GiB '
      'for weights+grads+Adam state (before activations)')
```

## Tied Parameters **[MOD]**

*Topics.* Sharing one parameter at several call sites. Replaces the current
toy (one `Linear` reused twice, motivated by nothing) with the real case:
**tying the input embedding and the output projection of a language model**
— saves $|V| \times d$ parameters, and is standard from word2vec-era models
through modern LLMs. What tying means for gradients (contributions sum) and
for traversal and checkpoints (one entry in `named_parameters`, two
compatible paths in PyTorch's `state_dict`).

*Code (PyTorch).* A miniature LM head: `nn.Embedding(V, d)` and an output
`nn.Linear(d, V, bias=False)` with `out.weight = emb.weight`; verify
`id(...)` equality, one entry in `named_parameters()`, both `state_dict()`
paths, and summed gradients via a
small backward pass.

## Freezing Parameters **[NEW]**

*Topics.* `requires_grad=False` as the primitive under every fine-tuning
recipe: freeze a backbone, train a head; passing only trainable params keeps
optimizer membership explicit (`filter(lambda p: p.requires_grad, ...)`),
while already allocated state survives later freezing; interaction
with `.eval()` vs freezing (orthogonal concepts, commonly confused). Brief
forward pointers: fine-tuning in the computer-vision chapter;
parameter-efficient methods (LoRA) with the low-rank math in the linear
algebra appendix (:numref:`sec_mdl-svd-low-rank`). One-paragraph note: **EMA /
weight averaging** as another form of derived, non-trained state, with a
pointer to the controlled recipe experiments.

*Code (PyTorch).* Freeze all but the last layer of the config-built MLP
from :numref:`sec_model_construction_v2`; print trainable vs total
parameter counts; one gradient step showing frozen weights unchanged.

## Summary and Exercises

*Exercises (sketch).* (1) Extend the byte-accounting helper to report per
top-level submodule; which block of the model dominates? (2) Make BatchNorm
running stats trainable by re-registering them as parameters — what goes
wrong, and why? (3) Tie two layers, then `deepcopy` the model — still tied?
Check and explain. (4) Freeze the embedding of the tied LM head but not the
output projection. What actually happens, and what does this teach about
tying + freezing interactions?

> **Downstream constraints.** None hard: current tied-params subsection has
> zero downstream users, so the example swap is free. Parameter-access
> idioms taught here are used ambiently everywhere. Label `sec_parameters`
> / `subsec_param-access` cited only from earlier chapters — keep on
> promotion.

## Framework Coverage

- **JAX** — full coverage with NNX variable types. Buffers use
  `nnx.Variable` subclasses (`nnx.Param`, `nnx.BatchStat`, or a custom type),
  and filters select optimizer/checkpoint views. Tying → `nnx.Embed.attend()`
  reuses the embedding kernel as the output head and yields *one* pytree
  leaf (verified), no `id()`-aliasing bookkeeping. Freezing → optimizer-side
  `optax.multi_transform` (+`set_to_zero`, verified) — the model never
  knows it's frozen. EMA → first-class `optax.ema` (verified): implement,
  don't just mention.
- **TensorFlow** — full coverage. Buffers → `add_weight(trainable=False)`
  (verified); placement-at-creation nuance noted. Tying → no built-in flag:
  small `TiedLMHead` layer doing `matmul(x, emb.embeddings,
  transpose_b=True)` (verified, one shared variable). Freezing →
  `layer.trainable = False` (verified through a train step). EMA →
  *built-in and cleaner than PyTorch*: `Adam(use_ema=True)` +
  `finalize_variable_values()` (verified end-to-end) — give it a real cell.
- **MXNet** — full coverage, one mechanism fewer: `gluon.Constant` *is* a
  `Parameter` with `grad_req='null'` hardwired (verified in source), so
  buffers and freezing share one concept; freezing via `grad_req='null'`
  is the cleanest of the four tabs. Tying via the corpus's existing
  `share_parameters` idiom (shapes match, verified). EMA hand-rolled
  (ALT). Byte accounting gains a third term under `multi_precision=True`
  (fp32 shadow weights).
