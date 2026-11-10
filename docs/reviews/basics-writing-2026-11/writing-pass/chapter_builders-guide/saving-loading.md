# Writing pass: `chapter_builders-guide/saving-loading.md` (5.6)

Section order in chapter 5: model-construction, parameters-state-memory, init,
custom-layers, numerics, **saving-loading**, gpus-devices-memory,
reproducibility-inspection. This section is sixth, so it owns little: the
state tree, buffers, the 16-bytes accounting, Adam's two averages, and
freezing all belong to 5.2; bf16 belongs to 5.5.

Checker: `prose words: 5257 -> 5191 (-66)`, "OK: only prose changed";
`tools/lint_source.py` clean. (The count covers all four tabs. The net change
is small because some findings asked for added sentences; cuts were about
330 words, additions about 265.)

## Before/after pairs

1. Before: "For the small models of earlier chapters, that means repeating a
   few minutes of training. For a large model it can mean repeating weeks of
   computation, and at scale such failures are routine."
   After: "For the small models of earlier chapters this is a nuisance: a few
   minutes of training to repeat. For a large model it can mean weeks of
   computation, and at scale interruptions are the rule."
2. Before: "This section first saves and loads the state of a small network.
   It then asks which file formats are safe to load from a stranger, writes a
   checkpoint from which training resumes exactly, and loads pretrained
   weights into a network with a new output layer."
   After: "This is a practical section, about files: what goes into them,
   which of them can be trusted, and what a load must check. We first save
   and load the state of a small network. We then ask which file formats are
   safe to load from a stranger, write a checkpoint from which training can
   resume, and load pretrained weights into a network with a new output
   layer."
3. Before: "The *safetensors* format is one: a file holds tensors and a
   description of them, and nothing else."
   After: "The *safetensors* format contains no program: a file holds tensors
   and a description of them, and nothing else."
4. Before: "With a fresh Adam, the loss jumps from a few thousandths or less
   to between about 0.5 and 6 after a single step, and after four steps it is
   still hundreds of times higher than at the checkpoint, or more.
   :eqref:`eq_saving_adam` explains the jump."
   After: "With a fresh Adam, a single step raises the loss by about three
   orders of magnitude, and four steps later it has not recovered: it is
   still hundreds of times higher than at the checkpoint. How can one step of
   an optimizer do so much damage to a model that was nearly done?
   :eqref:`eq_saving_adam` explains it."
5. Before: "To check that the checkpoint holds everything the model needs, we
   corrupt every parameter, load the checkpoint, and compare the losses."
   After: "A load must overwrite whatever the model holds at the time. To
   check that it does, we corrupt every parameter, load the checkpoint, and
   compare the losses."
6. Before: "How it computes does not matter here: for loading weights it is a
   model state of a few hundred named tensors."
   After: "We have not met convolutional networks yet, and luckily we do not
   need to: for loading weights, ResNet-50 is simply a model state of a few
   hundred named tensors."
7. Before: "The entries of the last stage are not: the file omitted a whole
   stage, which would now compute features with random weights."
   After: "The entries of the last stage are not. Had we loaded without
   looking, a large part of the network would now compute its features with
   random weights, and no error would have told us."
8. Before: "In fp32, the weights and the two averages take 12 bytes per
   parameter, so the checkpoint of a model with 8 billion parameters takes
   96 GB, six times the 16 GB of its weights in bf16."
   After: "This leaves 12 of the 16 bytes per parameter that training with
   Adam in fp32 stores (:numref:`sec_parameters`): the weights and the two
   averages. For a model with 8 billion parameters the checkpoint therefore
   takes 96 GB, three times the 32 GB of its fp32 weights and six times the
   16 GB file that is enough for inference in bf16 (:numref:`sec_numerics`)."

## Repetition removed

- **16 GB / 96 GB for the 8B model**: was in the opening and again under
  "Training Checkpoints". The opening now says only "several times larger";
  the numbers are derived once, in the body, from 5.2's 16 bytes.
- **Adam's update**: 5.2 owns "two running averages per parameter". The
  four-equation display is reduced to the one update line (label
  `eq_saving_adam` kept), with references to `sec_parameters` and `sec_adam`.
- **"Many models start from weights someone else trained"**: stated in the
  opening; the near-verbatim repeat that opened "Pretrained Weights" is gone.
- **Freezing**: owned by `subsec_freezing` in 5.2. Shared prose is now two
  sentences with that reference; the four tab notes are cut to what the cell
  needs. The batch-normalization caveat is one clause pointing to 5.2.
- **Closing paragraph** ("We can now save ..., check ..., resume ..., and
  load ..."): removed as a second recap; the two hand-off sentences to
  `sec_use_gpu` remain, and the Summary is the only recap.
- **Xavier variance formula** in the experiment setup: removed (belongs to
  `subsec_xavier`, irrelevant here).

## Review findings

Applied: SAV-1, SAV-3, SAV-4, SAV-5, SAV-8, SAV-10, SAV-11, SAV-12, SAV-13,
SAV-14 (prose only), SAV-15; structural finding 4 (prose only).

Applied in part:
- SAV-7 / structural 1: "resumes exactly" changed to "can resume". The
  bit-for-bit comparison with an uninterrupted run needs a new cell.
- Structural 2: one sentence now says the experiment tests only the model
  state and the optimizer state. Making the other compartments do work needs
  a changed experiment.
- SAV-9: the cell stays (no code changes); its lead-in now claims only what
  it shows (a load overwrites), and the next paragraph says the optimizer is
  not yet tested.
- SAV-2: the `x`/`y` cell stays; its lead-in now says it is a first try on
  tensors that belong to no model.

Deferred:
- SAV-6 (one name for Adam's state across the chapter): chapter-wide
  decision. This section keeps "running averages", which is the term 5.2
  introduces; exercise 4 says "second moment", also defined in 5.2.
- SAV-16 (show a checksum): needs a line of code; exercise 6 covers it.
- SAV-14, code part: the freezing cell and its output remain.
- Exercise suggestions (drop exercise 5, add the 105-step exercise):
  exercises were not to be touched.

## Factual claims added or changed

- "a single step raises the loss by about three orders of magnitude ...
  still hundreds of times higher" (was "to between about 0.5 and 6"). Stored
  outputs, loss at checkpoint to loss after one fresh step / after four:
  PyTorch 0.002 to 4.31 / 2.05; MXNet 0.001 to 2.57 / 1.02; TensorFlow
  0.0016 to 3.21 / 1.41; JAX 0.0016 to 2.58 / 1.05. Ratios 1,600 to 2,600
  after one step, 650 to 1,000 after four.
- "96 GB, three times the 32 GB of its fp32 weights and six times the 16 GB
  file ... in bf16" (was "six times the 16 GB of its weights in bf16").
- "12 of the 16 bytes per parameter that training with Adam in fp32 stores".
- ResNet-50: "(23.5 million without its 1,000-class head)". Stored totals
  with a 10-class head: 23,528,522 (PyTorch, JAX), 23,547,402 (MXNet),
  23,555,082 (TensorFlow); all round to 23.5 million without the head.
- Fresh Adam, in words instead of formulas: "After one update each holds a
  small fraction of g and of g⊙g, and the correction divides exactly that
  fraction out". (The fractions are 1 − β₁ and 1 − β₂; β is no longer named
  in the body.)
- "the step counter, which Adam's correction uses and which also sets the
  position in any learning-rate schedule".
- "With the full batch, no dropout, and no learning-rate schedule, the
  random-number state has no work to do, so what follows tests only the
  model state and the optimizer state."
- "such checks have had holes, in PyTorch's `weights_only` loader among
  others" (the version number of the fix is dropped; 2.6 remains once, in
  the PyTorch tab, as the release that changed the default).
- "the last of the four stages into which ResNet-50 groups its layers"
  (was "the deepest of ResNet-50's four stages").
- "We have not met convolutional networks yet" (true at chapter 5).
- Forward reference for Adam is `sec_adam` (inside `chap_optimization`),
  not the chapter label.
- Removed: the MXNet and TensorFlow notes explaining why their parameter
  totals differ (18,880 and 26,560 biases); the per-framework
  batch-normalization details under freezing; the JAX note that this
  ResNet-50 uses `use_running_average=True`.

## Slides that now differ from the body (not edited)

- "Fresh Adam: about 0.5 to 6 after one step" (resume slide and recap): the
  body now gives the factor instead of the range. Not contradictory, but the
  range is the hedge SAV-10 objected to.
- The Adam slide still shows the full four-part update with β₁ = 0.9,
  β₂ = 0.999; the body shows only the parameter update. More detail than
  the body, not a contradiction.
- The pickle slide still dates the fix of the `weights_only` hole to PyTorch 2.6: the body no
  longer gives this version.
- "An 8B Model: 16 GB of Weights, 96 GB of Checkpoint" and the recap line:
  the slide keeps the bf16-versus-fp32 comparison that SAV-1 calls
  misleading; the body now also gives the 32 GB fp32 figure.
