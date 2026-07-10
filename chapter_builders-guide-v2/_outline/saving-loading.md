# Saving, Loading, and Pretrained Weights
:label:`sec_read_write_v2`

> **Role.** Rewrite of the current read-write section around the workflow
> that actually dominates 2026 practice: checkpoints are *structured state*
> (weights + optimizer + RNG + step), the interchange format is
> **safetensors**, and the most common reason to load weights is that
> *someone else trained them*. The current section's timeless core — you
> serialize state, not the class — is kept as the organizing principle.
> Almost nothing downstream depends on the current section (one file), so
> the rewrite is unconstrained.

## State, Not Code **[KEPT]**

*Topics.* The architecture/parameters split: the class is code, the
checkpoint is the `state_dict` tree from :numref:`sec_parameters_v2`
(names → tensors, buffers included). To reload you need both the code and
the state — which is why the config object of
:numref:`sec_model_construction_v2` belongs *inside* the checkpoint.
Loading tensors alone (`torch.save`/`torch.load` on a tensor/dict) kept as
the warm-up.

*Code (PyTorch).* Save/load a tensor, a list, a dict (current section's
warm-up, compressed to one cell); then `net.state_dict()` printed as a
tree.

## safetensors: the Interchange Format **[NEW]**

*Topics.* Why pickle-based `torch.save` is a code-execution risk for
*sharing* (and why `weights_only=True` became the default), and how
**safetensors** fixes this by storing only tensors + a JSON header: safe,
zero-copy/mmap-able, framework-neutral — the reason it is the de facto hub
format. `torch.save` demoted to "fine for your own disk, plus the format
you will still encounter in older code."

*Code (PyTorch).* `safetensors.torch.save_file(net.state_dict(), ...)` /
`load_file`; reload into a freshly built net, verify equality (current
section's verification cell, kept). Peek at the JSON header to demystify
the format — it is *just* names, dtypes, shapes, offsets.

## Checkpointing a Training Run **[MOD]**

*Topics.* A crash-safe checkpoint is more than weights: model state,
optimizer state (those Adam moments from :numref:`sec_parameters_v2` —
without them resumption restarts momentum from zero), RNG state, step/epoch
counter, and the config. Resume semantics: rebuild from config, load state,
continue — demonstrated honestly by killing and resuming a short training
run mid-way and showing the loss curve continues rather than restarts.
Note on atomic writes (save to temp, rename) as the difference between a
checkpoint and a corrupted file. Forward pointer: sharded/multi-file
checkpoints and `mmap=True` loading for models larger than RAM — two
sentences, not a demo.

*Code (PyTorch).* `save_checkpoint(path, net, opt, step, cfg)` /
`load_checkpoint(path)` helper pair — **`#@save` in the pytorch (and
mxnet) tabs only**: torch has no single-call native for a full training
checkpoint (the dict idiom *is* the native), whereas TF and JAX do —
their tabs use `tf.train.Checkpoint` and orbax directly, no d2l wrapper.
Then the interrupt-and-resume demonstration on the Fashion-MNIST MLP.

## Loading Weights You Did Not Train **[NEW]**

*Topics.* The most common serialization workflow in 2026: initialize from
pretrained weights. (i) `strict=False` partial loading — load a backbone,
leave the new head randomly initialized; read the
`missing_keys`/`unexpected_keys` report instead of ignoring it.
(ii) Surgery on the state dict as a plain Python dict — rename keys, drop a
head, adapt shapes — it is just a dict, and knowing that removes the fear.
(iii) Where weights come from: torchvision's `weights=` enums as the
worked example (small, no account needed), the Hugging Face Hub as the
ecosystem answer in prose (with safetensors as the connecting thread).
Forward pointer to the fine-tuning section of the computer-vision chapter,
which owns *when and why*; this section owns *how*.

*Code (PyTorch).* Download a small pretrained torchvision model
(`resnet18(weights=...)`); replace its final layer for a 10-class problem;
`load_state_dict(..., strict=False)` on a mutated dict, printing the
missing/unexpected keys report and freezing the backbone
(:numref:`sec_parameters_v2` closes the loop). *(Dataset note: uses no new
dataset — the model itself is the artifact under study.)*

## Summary and Exercises

*Exercises (sketch).* (1) Kept from current: why checkpoint even if
deployment is elsewhere? — extended with the atomic-write failure mode.
(2) Open a safetensors file with a hex viewer / read the header length
manually; how large is the header of your MLP checkpoint? (3) Save at bf16
and reload into an fp32 model (:numref:`sec_numerics_v2`) — what is lost,
and is it acceptable for inference vs resuming training? (4) Take two
checkpoints of the same run 1000 steps apart and average the weights —
evaluate; the result previews weight averaging/EMA.

> **Downstream constraints.** Only `natural-language-inference-bert.md`
> uses the current section's `torch.save`/`load` idiom; it survives
> unchanged since `torch.save` remains taught. New helpers
> (`save_checkpoint`/`load_checkpoint`) are candidate `#@save`s that future
> transformer/generative chapters can standardize on. Adds one small
> external download (torchvision resnet18 weights, ~45 MB) — flag for the
> build's data-cache policy.

## Framework Coverage

*`safetensors` is a **new dependency for all four venvs** (verified working
for torch-style flat dicts, flax, tensorflow; mxnet via the numpy module).*

- **JAX** — `safetensors.flax` verified (expects a *flat* dict → one small
  flatten/unflatten helper around the pytree). Full checkpoints: **cleaner
  than PyTorch** — `orbax.checkpoint.StandardCheckpointer` round-trips the
  whole `TrainState` (params + optax state + step) in one call (verified;
  orbax already ships with flax, no new dep; atomic writes are its default).
  Partial loading: pytree surgery + a hand-written missing/unexpected-keys
  diff (no `strict=False` analogue). Pretrained zoo: none — the JAX cell
  reloads a checkpoint saved earlier in the notebook; HF Hub is the prose
  answer.
- **TensorFlow** — `safetensors.tensorflow` verified (two footguns for the
  prose: numpy round-trip, and `save_file` mutates its input dict). Full
  checkpoints: **cleanest of the four** — `tf.train.Checkpoint(model=,
  optimizer=, step=)` verified incl. Adam slots; one Keras-3 tripwire:
  pre-build the optimizer (`opt.build(vars)`) before `restore()`.
  Pretrained: `keras.applications` verified end-to-end (actual ImageNet
  download) — the cleanest torchvision analogue. **Keras-3 API drift:**
  `by_name=True` now *raises* on native `.weights.h5` — use
  `load_weights(path, skip_mismatch=True)` (verified: skips mismatched
  head, loads backbone, prints what it skipped).
- **MXNet** — safetensors via the numpy bridge (`.asnumpy()` ↔
  `safetensors.numpy`, verified module) — instructive ALT: it *demonstrates*
  the format is framework-neutral. Checkpoints: `save_parameters`/
  `load_parameters` DIRECT; `Trainer.save_states/load_states` exist in
  source [UNVERIFIED; docstring admits lr_mult/wd_mult not saved]; **no RNG
  snapshot API** — reseed-only. Pretrained: `model_zoo resnet18_v2
  (pretrained=True)` *worked in the 2026-06-06 green run* (committed
  store) but rides on archived-project S3 — re-verify at rewrite time;
  `allow_missing`/`ignore_extra` skip *silently* → manual key-set diff in
  the cell.
