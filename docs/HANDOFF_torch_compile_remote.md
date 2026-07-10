# Handoff: replace TorchScript with `torch.compile` and move Computational Performance

This document hands the work to Codex on the Linux GPU host. It records the
decisions made on 2026-07-10, the repository state inspected on the Mac, the
required edit order, and the verification bar. Read the files named in
Section 1 before editing.

The remote host has four RTX 4090 GPUs and enough CPU and system memory for the
repository scheduler. Let `gmake detect` choose concurrency. Do not wrap a
notebook target in an outer parallel shell loop.

## 1. Read before editing

Read these files in full:

1. `docs/writing-avoid.md`
2. `docs/fable-fix.md`
3. `docs/build-system.md`, especially Sections 3, 5, 6.6, 6.7, 11, and 13
4. `docs/convnet-rewrite/HANDOVER.md`
5. `docs/convnet-rewrite/spec-ch7.md`
6. `docs/convnet-rewrite/spec-ch8.md`
7. This handoff

`docs/writing-avoid.md` governs the prose written during this project. Apply it
to book text, slide text, comments, commit messages, and any audit or plan.
`docs/fable-fix.md` adds the book-specific audit method and source invariants.

The short version is: state the fact, give the evidence, and stop. Do not add
significance language, ceremonial summaries, repeated framing, or claims of
speed without a valid measurement. Comments explain a constraint or failure
mode. They do not narrate the next line of code.

Only source `.md` files may be edited. `.qmd` files are generated.

## 2. Settled decisions

The following decisions are final for this pass.

### 2.1 Chapter order

Computational Performance moves after both recurrent-network chapters and
before Attention Mechanisms and Transformers. The resulting order is:

| Number | Chapter |
|---:|---|
| 9 | Recurrent Neural Networks |
| 10 | Modern Recurrent Neural Networks |
| 11 | Computational Performance |
| 12 | Attention Mechanisms and Transformers |
| 13 | Optimization |
| 14 | Computer Vision |

Chapter 14 and every later chapter keep their current numbers. Only the old
Chapters 11, 12, and 13 change number.

The semantic labels `chap_performance`, `chap_attention-and-transformers`, and
the section labels within those chapters must survive. Use semantic references
instead of writing chapter numbers in prose or docstrings when possible.

### 2.2 PyTorch execution model

Remove TorchScript from active book content. Do not retain a compatibility tab,
historical aside, deprecated exercise, or alias in the teaching code.

Use three separate concepts:

- `state_dict` stores training state and remains the checkpoint format taught
  in Chapter 6.
- `torch.compile` accelerates a Python program at run time.
- `torch.export` captures a full program for serialization and deployment.

Do not imply that `torch.compile` produces a portable artifact. Do not imply
that compilation always gives a speedup or bitwise-identical results.

### 2.3 Where compiled execution begins

Chapters 2 through 10 remain eager in their published examples. Chapter 11
introduces compilation. Sustained PyTorch training after Chapter 11 should use
the compiled path unless the compile cost, unsupported operations, or the
lesson itself gives a reason to remain eager.

This policy prevents an API from appearing before it is explained. It also
keeps small tensor demonstrations and scratch derivations easy to inspect.

Compilation is explicit in reader-facing code. Do not silently turn it on for
every `Trainer` constructed before Chapter 11.

### 2.4 Framework scope

MXNet remains a co-equal tab. Keep `HybridBlock`, `HybridSequential`, and
`hybridize()` where they belong. Rewrite shared prose so that MXNet
hybridization is one framework's execution model rather than the definition of
modern compilation for every framework.

Keep JAX `jit` and TensorFlow `tf.function` material, but check every shared
claim against what those APIs do. Avoid prose about "all frameworks". A reader
of a rendered notebook sees one framework tab.

### 2.5 Chapters 7 and 8 are under active revision

Another editing process currently owns:

- `chapter_convolutional-neural-networks/**`
- `chapter_convolutional-modern/**`
- `docs/convnet-rewrite/**`
- the related chapter entries in `_quarto.yml` and
  `tools/d2l_preprocess.py`
- related bibliography and figure work

Do not perform a compile-focused content audit of Chapters 7 or 8 until that
process is complete and its changes are on the branch used for this work.

Before touching either chapter:

1. Pull the latest branch.
2. Read the completion note appended to `docs/convnet-rewrite/HANDOVER.md`.
3. Confirm with the user that the Chapter 7 and 8 edit is finished.
4. Audit the resulting files, not the pre-rewrite files described in this
   handoff.

Until then, Chapters 7 and 8 are read-only. Compile smoke tests may be run
against them, but do not edit their source, outputs, slides, figures, config
entries, or bibliography on their behalf.

## 3. Repository state inspected on the Mac

The inspection was performed on:

```text
branch: main
commit: 84807eef
date:   2026-07-10
```

The first remote action must be `git pull --ff-only`, followed by a fresh
inventory. The Chapter 7 and 8 work will change paths and configuration, so
line numbers below are anchors to the inspected commit, not instructions to
edit stale lines.

### 3.1 Dependency versions

`pyproject.toml` pins:

```text
torch==2.11.0
torchvision==0.26.0
Linux PyTorch index: CUDA 12.8
```

`chapter_installation/index.md` still tells readers to install PyTorch 2.0.0
and torchvision 0.15.1. That instruction must be updated as part of this pass.
Use the repository's tested versions or the official version selector. Do not
invent a CUDA command from memory.

### 3.2 Deprecated PyTorch content

All active TorchScript references found in chapter source are in
`chapter_computational-performance/hybridize.md`:

- prose introducing TorchScript;
- two `torch.jit.script` conversions;
- the eager versus TorchScript benchmark;
- scripted-module serialization;
- the summary and exercises;
- the slide deck.

There were no `torch.compile` calls in active chapter source at the inspected
commit.

Run this inventory again after pulling:

```bash
rg -n --glob '*.md' --glob '*.py' \
  'torch\.jit|TorchScript|torchscript|torch\.compile|torch\.export' \
  chapter_* d2l tools
```

### 3.3 Shared training abstractions

The PyTorch `Module` and `Trainer` originate in
`chapter_linear-regression/oo-design.md`. Device-aware methods are patched in
`chapter_builders-guide/gpus-devices-memory.md`. In the generated
`d2l/torch.py` at the inspected commit:

- `Trainer.fit` calls `prepare_model` before creating the optimizer;
- `prepare_model` moves the model to the selected device;
- `training_step` computes the model output and loss, then calls `plot`;
- plotting and data loading are Python-side operations.

This order supports in-place module compilation after device placement and
before optimizer creation. Compile the module call, not the progress board or
data loader.

### 3.4 Chapter-numbered helpers

The repository contains 41 source occurrences of `train_ch13` or
`train_batch_ch13` across 11 Markdown files. Their definitions are in
`chapter_computer-vision/image-augmentation.md`, and generated copies appear in
all four `d2l/*.py` modules.

These names are already detached from the current chapter numbering. Do not
rename them to `train_ch11`, `train_ch13`, or `train_ch14`. Replace them with
semantic names after auditing every call site. Candidate names are
`train_classifier` and `train_batch_classifier`; choose names that remain true
for FCN, NLI, sentiment analysis, and recommender examples. Do not leave a
deprecated alias in book code unless an external compatibility requirement is
found and documented.

Inventory command:

```bash
rg -n --glob '*.md' --glob '*.py' 'train(_batch)?_ch[0-9]+' chapter_* d2l
```

Do not broaden this pass into a mechanical rename of every old `train_ch6` or
`train_ch11` helper. Record those in the audit. Rename a helper now when it is
part of the compiled execution path or when chapter movement makes its prose
false.

### 3.5 Direct numerical references

The inspected tree contains direct prose and docstring references to Chapter
13 in:

- `chapter_preface/index.md`;
- `chapter_computer-vision/image-augmentation.md`;
- generated `d2l/*.py` files.

Replace them with the new number only where the number helps the reader.
Prefer `:numref:` or a semantic phrase such as "the computational-performance
chapter". Generated library files are rebuilt from `#@save` source and must
not be edited by hand.

### 3.6 Known pre-Chapter-11 touchpoints

The checked tree needs little compilation-related work before Chapter 11:

- `chapter_preliminaries/ndarray.md` points to `sec_hybridize`. Preserve that
  label or update the reference.
- Chapter 6 needs short forward references about numerical differences,
  device placement, hooks, and export. Section 8 gives the limits.
- `chapter_convolutional-modern/batch-norm.md` claims that the scratch layer
  "must be interpreted by Python." That claim is false once compilation is
  available. Re-locate and revise it only after the Chapter 8 rewrite is done.

Chapters 2 through 5 contain no TorchScript material and need no compile calls.

## 4. Source-of-truth references

Use primary PyTorch sources for software claims. These pages were current when
the handoff was written; verify them again on the remote host because compiler
behavior changes between releases.

- TorchScript deprecation:
  <https://docs.pytorch.org/docs/stable/jit_language_reference_v2>
- `torch.compile` API and `Module.compile()`:
  <https://docs.pytorch.org/docs/stable/generated/torch.compile>
- Compiler programming model:
  <https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.html>
- Common graph breaks:
  <https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.common_graph_breaks.html>
- `fullgraph=True` guidance:
  <https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.fullgraph_true.html>
- Troubleshooting, compile time, and placement:
  <https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_troubleshooting.html>
- `torch.export` and serialization:
  <https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export.html>
- Module hooks under compilation:
  <https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_nn_module.html>
- Numerical accuracy:
  <https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html>

Search `d2l.bib` for an existing PyTorch 2 compiler paper before adding one.
If absent, verify and add the SOSP 2024 PyTorch 2 paper rather than citing a
blog post for the compiler architecture. Add bibliography entries only after
the Chapter 7 and 8 process has released `d2l.bib`.

## 5. Editorial model for Chapter 11

The current file name `hybridize.md` may remain to avoid path churn. The
displayed title and chapter vocabulary should change.

### 5.1 Concepts the section must distinguish

Teach these terms once, in this order:

1. Eager execution: Python dispatches tensor operations as the program runs.
2. Graph capture: the framework recovers tensor computation from a Python
   function or module.
3. Compilation: captured graphs are transformed and lowered for a target.
4. Guards: assumptions attached to compiled code determine whether it can be
   reused.
5. Recompilation: a failed guard may produce another compiled variant.
6. Graph break: unsupported Python ends one captured region and resumes eager
   execution under permissive compilation.
7. Export: ahead-of-time full-graph capture produces a serializable program and
   rejects unsupported behavior.

Do not describe `torch.compile` as conversion from imperative code to one
symbolic program. Do not use "symbolic programming" as the chapter-wide model.
The term may remain inside the MXNet explanation where it is accurate.

### 5.2 Recommended section structure

Rewrite `chapter_computational-performance/hybridize.md` around this sequence:

1. **Eager execution and captured graphs.** Replace the current toy compiler
   story with a short example that separates Python orchestration from tensor
   computation.
2. **Compile a model.** Show eager PyTorch first, then `net.compile()` or
   `torch.compile(net)`. Explain why the in-place module method suits the
   book's `Trainer` and preserves module identity.
3. **Compilation has a cost.** Measure the first invocation separately from
   steady-state execution.
4. **Guards, graph breaks, and dynamic shapes.** Give one small example of each
   failure mode. Keep logging and verbose compiler traces out of committed
   output; show the diagnostic command in prose.
5. **Training under compilation.** Show that forward computation participates
   in compiled autograd while the epoch loop and plotting remain eager.
6. **Export is a different operation.** Use `torch.export.export`,
   `torch.export.save`, `torch.export.load`, and the loaded program's module.
7. **Framework comparison.** Keep MXNet hybridization, JAX `jit`, and
   TensorFlow `tf.function` in self-contained tabs.
8. **Summary and exercises.** Test the student's ability to identify a useful
   compile boundary and diagnose recompilation. Do not ask students to use a
   deprecated API.

The exact subsection count may change when the prose is written. The conceptual
order may not.

### 5.3 Labels and cell IDs

Keep the existing `:label:` value `sec_hybridize` because earlier material
refers to it. A second label such as `sec_graph_compilation` may be added if it
has a distinct heading and a real cross-reference use.

Keep existing code-cell IDs when an existing cell retains the same teaching
role. New cells need IDs in the file's convention. Do not renumber existing
IDs for cosmetic consistency.

The `Benchmark` class in this file is marked `#@save` and is used by later
sections. Preserve the public name and context-manager behavior unless a
repo-wide use audit proves that an API change is needed.

After editing:

```bash
python3 tools/add_cell_ids.py --check
python3 tools/lint_source.py chapter_computational-performance/hybridize.md
```

### 5.4 Benchmark requirements

The current batch-one MLP benchmark cannot support a general speedup claim.
Replace it with a workload large enough to amortize launch overhead on a 4090,
then report what was measured.

For PyTorch:

- create eager and compiled views of the same initialized parameters;
- call `torch.testing.assert_close` before timing;
- use inference mode for an inference benchmark;
- synchronize CUDA immediately before and after each timed region;
- separate first invocation from repeated execution;
- use several repeats and report a stable statistic;
- record the device name and PyTorch public version;
- test a second shape so the text can discuss guards and recompilation;
- compare gradients as well as outputs in the training example.

Do not call the first invocation a "cold compile" unless the Inductor disk
cache was isolated or cleared for that measurement. The first invocation in a
new Python process can still reuse a persistent compiler cache. "First
invocation in this process" is accurate without cache isolation.

Do not make a test fail because speedup is below a threshold. Performance is a
measurement, not a correctness invariant. A regression test may fail on wrong
timing methodology, missing synchronization, compiler failure, or output
disagreement.

Remove the slide claim of "10–100× less Python overhead." No evidence in the
current notebook establishes it.

### 5.5 Graph-break example requirements

The example should teach one cause, not dump compiler internals. Good choices
are data-dependent Python control flow, `.item()`, or printing inside a compiled
region. Show the eager behavior first.

Use `fullgraph=True` in a diagnostic example so an unintended graph break
raises instead of silently falling back. Catch only the expected exception and
print a stable description, not the full version-dependent compiler message.

Mention:

```bash
TORCH_LOGS=graph_breaks,recompiles
```

Do not execute verbose logging in the rendered notebook.

### 5.6 Export example requirements

The PyTorch serialization cell should follow this shape:

```python
program = torch.export.export(net, (x,))
torch.export.save(program, 'my_mlp.pt2')
restored = torch.export.load('my_mlp.pt2')
torch.testing.assert_close(restored.module()(x), net(x))
```

Verify the exact PyTorch 2.11 API on the remote host. The prose must say that
export requires a full graph and may need explicit dynamic-shape constraints.

Keep Chapter 6 checkpointing unchanged. The exported program is not an
optimizer checkpoint and does not replace `state_dict`.

### 5.7 Slides

Rewrite the entire slide block in `hybridize.md`. Existing slides repeat the
outdated imperative-versus-symbolic framing and contain TorchScript.

The new deck should include:

- eager versus captured execution;
- the compile boundary;
- first invocation versus steady state;
- guards and graph breaks;
- compiled training;
- compile versus export;
- one measured benchmark with its hardware stated.

Every `@cell-id` reference must resolve. Run the slide audit and inspect the
render at 720 pixels.

## 6. Trainer integration

The compile option belongs in Chapter 11, not in the Chapters 3 through 6
exposition.

### 6.1 Public behavior

After the Chapter 11 `#@save` patch, PyTorch should support:

```python
trainer = d2l.Trainer(max_epochs=10, num_gpus=1, compile=True)
```

Required behavior:

- `compile=False` is the default;
- the model moves to its device before compilation;
- the model is compiled before the optimizer is constructed;
- only the module call is compiled;
- plotting, data transfer, metric aggregation, and the epoch loop remain eager;
- checkpoint keys and the public module object remain stable;
- eager execution remains available for debugging.

Use `model.compile()` in place unless testing exposes a reason not to. PyTorch
documents it as compiling the module call in place without replacing the
module's structure.

### 6.2 Where to define it

Use the book's incremental class-extension pattern in Chapter 11. A saved
PyTorch cell may redefine `Trainer.__init__` and `Trainer.prepare_model` with
`@d2l.add_to_class` after the Chapter 6 device patch has been introduced.

The patched `__init__` must retain the existing arguments and behavior:

```text
max_epochs
num_gpus=0
gradient_clip_val=0
```

Add `compile=False`. Add compiler option plumbing only if a later notebook
uses it. Avoid an options surface that the book never explains.

The patched `prepare_model` must preserve:

- `model.trainer = self`;
- board limits;
- device movement;
- the final `self.model` assignment.

Then compile after device movement. Do not edit generated `d2l/torch.py` by
hand. `gmake lib` must create the final module from the saved source.

### 6.3 Expected output-store impact

A new saved `Trainer` patch changes the PyTorch library fingerprint. Earlier
notebooks that use `Trainer` may become stale even though `compile` defaults to
false. This is expected.

After `gmake lib` and notebook generation, let the audit name the actual blast
radius:

```bash
gmake lib
gmake notebooks-pytorch
python3 tools/audit_outputs.py --frameworks pytorch --stale
```

Do not guess the re-execution list. Do not suppress it merely because the
default behavior is unchanged.

## 7. Chapter movement and configuration

Do this only after the Chapter 7 and 8 process has finished, since that process
also edits `_quarto.yml` and `tools/d2l_preprocess.py`.

### 7.1 `index.md`

Move `chapter_computational-performance/index` immediately after
`chapter_recurrent-modern/index` and before
`chapter_attention-mechanisms-and-transformers/index`.

### 7.2 `_quarto.yml`

Move the complete `Computational Performance` part after the complete
`Recurrent Neural Networks` part and before `Attention Mechanisms and
Transformers`. Do not split the chapter or move individual sections.

### 7.3 `tools/d2l_preprocess.py`

Update `CHAPTER_NUMBERING` as follows:

- every `chapter_computational-performance/*` entry: 13 to 11;
- every `chapter_attention-mechanisms-and-transformers/*` entry: 11 to 12;
- every `chapter_optimization/*` entry: 12 to 13;
- Chapter 14 and later: unchanged.

Preserve all subsection numbers within each chapter unless the Chapter 11
rewrite explicitly changes its internal order. If `hybridize.md` remains the
first section, it becomes 11.1.

### 7.4 Preface

Rewrite the affected roadmap in `chapter_preface/index.md` so the narrative
matches the new order:

- RNNs;
- computational performance in Chapter 11;
- attention and transformers in Chapter 12;
- optimization in Chapter 13;
- computer vision in Chapter 14.

The current Part 2 and Part 3 paragraphs will both need adjustment. Avoid a
list of bare chapter numbers when semantic cross-references read better.

### 7.5 Cross-reference checks

Run:

```bash
rg -n --glob '*.md' 'Chapter 1[1234]|chapter 1[1234]' chapter_* index.md
rg -n --glob '*.md' ':numref:`chap_performance`|:numref:`sec_hybridize`' chapter_*
python3 tools/lint_source.py --corpus
```

Judge every numeric hit. A reference to an external book's chapter is not part
of this renumbering.

## 8. Changes before Chapter 11

Keep this integration pass short. Chapter 11 owns the compiler explanation.

### 8.1 Chapters 2 through 5

Do not add compile calls or new compiler subsections.

Preserve `sec_hybridize` so the forward reference in
`chapter_preliminaries/ndarray.md` remains valid. If the displayed title makes
the old wording inaccurate, change that one sentence to "graph capture and
compilation" and retain the reference.

Do not modify the original `Trainer` lesson in Chapter 3. The compile option is
added by the Chapter 11 saved patch.

### 8.2 Chapter 6

After checking the final Chapter 6 text, make at most these additions:

1. `chapter_builders-guide/numerics.md`: one paragraph explaining that fused
   or reordered floating-point operations may differ within tolerance, with a
   forward reference to Chapter 11.
2. `chapter_builders-guide/saving-loading.md`: one sentence distinguishing
   checkpoints from exported programs, with a forward reference.
3. `chapter_builders-guide/gpus-devices-memory.md`: state that device placement
   precedes compilation if the current text does not already establish it.
4. `chapter_builders-guide/reproducibility-inspection.md`: add a PyTorch-tab
   qualification about hooks under compilation. Recommend eager execution for
   debugging and point forward to Chapter 11.

Do not add compiler implementation detail to Chapter 6. Do not rewrite the
checkpoint chapter around `torch.export`.

`chapter_builders-guide/index.md` is also touched by the active Chapter 7 and 8
process for its reading list. Wait for that change before editing or rendering
the leading page.

### 8.3 Chapters 7 and 8 after their release

Once the active edit is complete, perform a narrow compatibility audit against
the final files.

The audit should answer:

- Do representative final models compile with PyTorch 2.11?
- Do eager and compiled outputs and gradients agree within dtype-appropriate
  tolerances?
- Does train/eval state, especially normalization state, behave correctly?
- Do fixed Python construction loops remain outside the run-time graph as
  intended?
- Does any final prose claim that scratch tensor code must be interpreted by
  Python?

Run at least these models if they still exist after the rewrite:

- LeNet;
- the scratch normalization model;
- ResNet;
- one densely connected or ConvNeXt-style model.

Do not add `compile=True` to their published training cells. They precede the
compiler chapter. Edit prose only when the final text makes a false claim about
compiled execution.

Record compatibility failures for Chapter 11 discussion or a later fix. Do not
reshape Chapters 7 and 8 around compiler constraints without user approval.

## 9. Changes after Chapter 11

Compilation should be visible at sustained training boundaries. Do not add it
to every PyTorch cell.

### 9.1 Chapter 12: attention and transformers

This is the first chapter that should exercise the compiled `Trainer` in
published code.

Audit and update the PyTorch training cells in:

- `chapter_attention-mechanisms-and-transformers/bahdanau-attention.md`;
- `chapter_attention-mechanisms-and-transformers/transformer.md`;
- `chapter_attention-mechanisms-and-transformers/vision-transformer.md`.

Use `compile=True` for the sustained training runs. Keep shape checks,
attention visualizations, autoregressive decoding, and one-off demonstrations
eager.

Test train and validation separately. A module that switches between training
and evaluation may compile distinct variants. That is expected, but repeated
recompilation per batch is not.

Check variable sequence lengths and masks. Use the default dynamic-shape
behavior first. Set `dynamic=True` only when measured recompilation shows that
it is needed and the resulting performance is acceptable.

### 9.2 Chapter 13: optimization

Most scratch optimization cells are tiny mathematical demonstrations. Keep
them eager.

For `lr-scheduler.md`, compilation may cover the model forward while scheduler
updates remain eager. Do not capture plotting or the epoch scheduler. If a
changing learning rate causes recompilation in a compiled training step, reduce
the compile boundary to the module call.

The opening shared prose currently says the network is "hybridized" even
though that action is specific to the MXNet tab. Rewrite the shared sentence in
framework-neutral terms.

### 9.3 Chapter 14 and later

Audit sustained PyTorch training in these categories:

- computer vision;
- language-model pretraining and applications;
- GANs;
- recommender systems;
- hyperparameter optimization;
- later model-based math chapters.

Use these boundaries:

| Workload | Default |
|---|---|
| Long fixed-shape model training | Compile the model call |
| Tokenization, augmentation, file I/O | Eager |
| Greedy or beam decoding with Python control | Eager unless rewritten and measured |
| Detection postprocessing and visualization | Eager |
| GAN alternating optimizer orchestration | Eager; compile each model separately |
| Short hyperparameter trials | Eager unless the trial amortizes compile time |
| Sparse recommender components | Test first; allow an explicit eager path |
| Neural style transfer | Measure before adding compilation |

An opt-out should have a specific reason in the audit. Do not add repeated
boilerplate to every notebook.

## 10. Semantic training-helper rename

Audit all definitions and uses before choosing final names:

```bash
rg -n --glob '*.md' --glob '*.py' 'train_ch13|train_batch_ch13' chapter_* d2l docs
```

The source definitions are saved cells in
`chapter_computer-vision/image-augmentation.md`. Rename the source definitions
and every source call site. Rebuild `d2l/*.py`; do not edit generated copies.

The inspected source call sites include:

- `chapter_computer-vision/image-augmentation.md`;
- `chapter_computer-vision/kaggle-cifar10.md`;
- `chapter_computer-vision/fine-tuning.md`;
- `chapter_computer-vision/fcn.md`;
- `chapter_natural-language-processing-pretraining/bert-pretraining.md`;
- four files under `chapter_natural-language-processing-applications/`;
- `chapter_recommender-systems/fm.md`;
- `chapter_recommender-systems/deepfm.md`.

Re-run the inventory after pulling because the list may have changed.

For the PyTorch helper:

- place the model on its device before compilation;
- compile the inner model before wrapping it for data parallelism;
- keep input movement and metric conversion eager;
- keep an explicit `compile_model=False` or equivalent default;
- enable the option only in post-Chapter-11 call sites where the run is long
  enough to amortize compilation.

The current helper uses `nn.DataParallel`. PyTorch recommends DDP for
production training. Do not silently replace the entire multi-process design in
this pass. Test compilation of the inner module with the existing teaching
wrapper. If it is incorrect or unstable, record the evidence and ask whether a
DDP rewrite belongs in scope.

Remove chapter-numbered docstrings. Use the defining section's semantic
cross-reference in prose and "Train one classifier minibatch" style docstrings
in code.

## 11. Multi-GPU material in Chapter 11

`chapter_computational-performance/multiple-gpus-concise.md` currently wraps a
model in `nn.DataParallel`. Add a PyTorch compilation experiment only after the
single-device semantics are established.

Required order:

```text
construct model
initialize parameters
move model to the primary device
compile the inner model
wrap for data parallelism
construct optimizer
train
```

Verify parameter identity and optimizer coverage. Compare one eager distributed
step with one compiled distributed step on two GPUs. Use two GPUs for the
committed example so it matches the scheduler's multi-GPU resource class; the
host's other GPUs can execute independent notebooks.

If the section discusses DDP, state the recommended order: compile the inner
module, then give it to DDP. Do not claim that the notebook's `DataParallel`
example is the production recommendation.

The manual multi-GPU section operates on tensor functions rather than an
`nn.Module`. Do not force `torch.compile` into it unless the result teaches a
separate point.

## 12. Audit and implementation sequence

Do not start with edits. The order below prevents collisions and makes each
commit reviewable.

### Phase 0: synchronize and establish ownership

```bash
git status --short
git pull --ff-only
git log -8 --oneline --decorate
gmake detect
```

Stop if the worktree contains changes you do not own. Do not stage or rewrite
them.

Check the Chapter 7 and 8 status. Until that work is complete, confine activity
to reading, compiler experiments in scratch space, and an audit of files outside
the frozen paths. Do not edit shared config while the other process owns it.

### Phase 1: write the audit

Create `torch-compile-audit.md` at the repository root. For each finding record:

- an ID;
- `file:line` after the latest pull;
- the current statement or code;
- the failure category from `docs/fable-fix.md`;
- the proposed disposition;
- whether a notebook must be re-executed.

At minimum, audit:

- all of `chapter_computational-performance/`;
- all TorchScript and `torch.compile` references repo-wide;
- `Trainer` definitions and patches;
- direct chapter-number references affected by the move;
- chapter-numbered training helpers;
- Chapter 6 compiler touchpoints;
- post-Chapter-11 training entry points;
- installation instructions;
- slide blocks in files that will change.

Do not audit Chapters 7 and 8 until their owner releases them.

### Phase 2: write the fix plan

Create `torch-compile-fix-plan.md`. It must contain:

- global terminology;
- final file list;
- labels to preserve and labels to add;
- saved symbols to add or change;
- bibliography keys;
- ordered per-file edits;
- notebook execution requirements by framework;
- an out-of-scope list;
- acceptance commands.

If the audit changes the scope materially, let the user review the plan before
editing. Otherwise continue under the decisions in this handoff.

### Phase 3: land configuration and prose that do not change code

After the Chapter 7 and 8 gate opens:

1. update chapter order in `index.md`;
2. update `_quarto.yml`;
3. update `CHAPTER_NUMBERING`;
4. update the preface;
5. update installation instructions;
6. run corpus lint and preprocess checks.

Use an explicit staged path list. Do not include Chapter 7 and 8 edits from
another process in this commit.

### Phase 4: rewrite Chapter 11 compiler content

Edit `hybridize.md`, its slide block, and any adjacent shared prose required to
avoid duplication with `async-computation.md`. Preserve the saved `Benchmark`
API. Add the saved PyTorch `Trainer` patch.

Execute the rewritten section in all framework tabs whose code changed. At a
minimum, execute all four variants of `hybridize.md` because the shared framing
and benchmark structure will change.

### Phase 5: integrate downstream PyTorch training

Start with Chapter 12. Verify the `Trainer` path there before touching older
custom helpers. Then update the semantic training helpers and selected later
workloads.

Keep each subsystem in its own commit:

- transformer integration;
- semantic helper rename and computer-vision callers;
- NLP callers;
- GAN and recommender exceptions;
- hyperparameter-optimization policy, if code changes are justified.

### Phase 6: run the delayed Chapter 7 and 8 compatibility audit

Run this only after their active rewrite is committed. Edit only false
compiler-related claims or code that fails the agreed compatibility bar. Do not
reopen their curriculum design.

### Phase 7: full execution and render

Rebuild the library, let the output audit compute the stale set, execute it
through the scheduler, capture with framework scope, then render and inspect.

## 13. Compiler correctness tests

Add a small committed diagnostic under `tools/` if the same checks would
otherwise be copied among notebooks. The diagnostic should run under
`.venv-pytorch` and exit nonzero on a semantic failure.

### 13.1 Basic module test

For a deterministic MLP on CUDA:

1. seed PyTorch;
2. create one model and inputs;
3. copy or reuse identical state for eager and compiled calls;
4. compare outputs;
5. compare scalar loss;
6. compare parameter gradients;
7. compare one optimizer update;
8. verify `state_dict` keys before and after in-place compilation;
9. save and load the checkpoint state;
10. export, save, load, and execute the exported program.

Use dtype-appropriate tolerances and print the chosen tolerance. Do not require
bitwise equality.

### 13.2 Trainer test

Run one short eager fit and one compiled fit with the same seed and fixed data.
Check:

- training completes;
- parameters change;
- the board flushes;
- the optimizer owns the original parameters;
- compile is disabled by default;
- `compile=True` compiles after device movement;
- an eager retry remains possible after a compiler failure.

Do not implement blanket `suppress_errors=True`. A compiler failure should be
visible during development.

### 13.3 Shape and mode tests

Call the compiled model with:

- the normal training batch;
- a smaller final batch;
- training mode;
- evaluation mode.

Inspect recompilation with `TORCH_LOGS=recompiles`. The test need not ban all
recompilation. It should catch recompilation on every batch.

### 13.4 Representative book models

After the Chapter 7 and 8 gate opens, test final versions of the models named in
Section 8.3. Also test:

- one scratch recurrent model from Chapter 9 or 10 as a compatibility check;
- the Chapter 12 Transformer encoder;
- Vision Transformer;
- one model through the two-GPU teaching path.

Earlier chapters remain eager even when their models pass these tests.

### 13.5 Graph completeness

Use `fullgraph=True` in tests for tensor-pure helper functions and simple
modules. Do not require it for every end-to-end notebook. A permissive compiled
model may contain a justified graph break outside the hot tensor region.

For each graph break kept in a sustained training path, record:

- source location;
- reason;
- frequency;
- measured cost or reason it is negligible;
- why moving the compile boundary is worse.

## 14. Remote execution on four RTX 4090 GPUs

### 14.1 Environment setup

Use the repository setup and lockfile. Do not install an arbitrary nightly into
the book environment.

```bash
git lfs install
git lfs pull
./bootstrap.sh --full
gmake detect
gmake venv-pytorch venv-jax venv-tensorflow venv-mxnet
gmake lib
```

Run the repository dependency checks if the bootstrap output requests them.
Confirm:

```bash
.venv-pytorch/bin/python -c \
  "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"
```

Expected device count is four. Expected public PyTorch version is 2.11.0.

### 14.2 Resource policy

The repository scheduler expects roughly three light jobs per 24 GiB GPU. A
compiler-heavy notebook can consume more GPU memory and much more CPU during
code generation.

Start by executing the rewritten compiler notebook alone. If memory use exceeds
the light-job assumption, add it to the appropriate resource classification in
`tools/runtime_env.py` rather than relying on a private shell override for the
final build.

Do not edit source while notebook execution is running. A source mtime change
can regenerate a framework's notebook set underneath the scheduler.

### 14.3 Focused execution

For one source file and one framework:

```bash
gmake notebooks-pytorch
gmake -B _notebooks/pytorch/chapter_computational-performance/hybridize.executed
python3 tools/capture_outputs.py --frameworks pytorch \
  chapter_computational-performance/hybridize.md
```

For all four changed variants, execute each generated notebook and capture only
the frameworks that succeeded:

```bash
for fw in pytorch jax tensorflow mxnet; do
  gmake -B "_notebooks/$fw/chapter_computational-performance/hybridize.executed"
done
python3 tools/capture_outputs.py \
  --frameworks pytorch,jax,tensorflow,mxnet \
  chapter_computational-performance/hybridize.md
```

Never run an unscoped capture after executing only one framework. It can replace
valid committed outputs for tabs that were not run.

### 14.4 Saved-symbol blast radius

After the Chapter 11 saved `Trainer` patch and any helper rename:

```bash
gmake lib
gmake notebooks
python3 tools/audit_outputs.py --stale
```

Feed the reported set to the repository scheduler. `gmake refresh-stale` is the
preferred route when all framework environments are ready:

```bash
gmake refresh-stale
```

Review the capture log. If only a subset of frameworks was executed, use
`tools/notebook_scheduler.py --files` and scoped `capture_outputs.py` instead
of the all-framework convenience target.

### 14.5 Timing discipline on the remote host

The four GPUs allow independent notebook execution. They do not make a timing
result immune to contention.

For committed Chapter 11 benchmark output:

- run the benchmark notebook without another job on the assigned GPU;
- record whether other GPUs were busy;
- synchronize CUDA around timed regions;
- avoid timing dataset download or first import;
- state batch size, dtype, model, device, and repeat count;
- label first-process invocation accurately;
- do not compare results gathered under different power or thermal states as
  if they differed only by compilation.

Use one 4090 for the single-device benchmark and two for the multi-GPU example.
The other cards may run non-benchmark notebooks only after the benchmark output
has been captured.

## 15. Output capture and rendering

### 15.1 Per-file gates

For each edited source file:

```bash
python3 tools/lint_source.py path/to/file.md
python3 tools/add_cell_ids.py --check
```

Check slide references when a slide block changes:

```bash
tools/audit_slides.py
```

Apply the `docs/writing-avoid.md` checks, including:

```bash
rg -n '—' path/to/file.md
rg -ni 'worth |not only|not just|not merely|delve|crucial|robust|seamless' path/to/file.md
```

Judge literal technical uses rather than deleting them blindly.

### 15.2 Repo-wide source gates

Before the final render:

```bash
rg -n --glob '*.md' --glob '*.py' \
  'torch\.jit|TorchScript|torchscript' chapter_* d2l
rg -n --glob '*.md' --glob '*.py' \
  'train_ch13|train_batch_ch13' chapter_* d2l
rg -n --glob '*.md' 'Chapter 11|Chapter 12|Chapter 13|Chapter 14' chapter_*
python3 tools/lint_source.py --corpus
python3 tools/add_cell_ids.py --check
git diff --check
```

Expected results:

- no TorchScript hit in active chapter or generated library content;
- no old chapter-numbered helper if its semantic rename is in scope;
- every direct chapter number judged against the new order;
- zero source-lint errors;
- zero missing cell IDs;
- no whitespace errors.

### 15.3 Output-store gates

```bash
gmake audit-outputs
gmake verify-outputs-fresh
```

On a four-GPU host the freshness gate is strict for CPU, GPU, and multi-GPU
notebooks. Do not call the project complete with a deferred stale set.

Review `git status outputs/` and the manifest diffs before staging. Binary plot
assets are Git LFS pointers; confirm LFS is active.

### 15.4 Render gates

```bash
gmake html
gmake -j4 slides
gmake -j4 pdfs
gmake check-all-artifacts
```

Inspect at least:

- the Chapter 11 HTML page in every framework tab;
- the compile benchmark output;
- the export round trip;
- the Chapter 11 slide deck;
- Chapter 12 training output under PyTorch;
- the preface and TOC numbering;
- all PDF pages that contain new equations or code listings.

Check for unresolved `?` references and stale chapter numbers in the rendered
text.

## 16. Commit plan

Use small commits with explicit staged paths. A suitable sequence is:

1. `docs: audit torch compilation migration`
2. `book: move computational performance before transformers`
3. `performance: replace TorchScript with PyTorch 2 compilation`
4. `torch: add opt-in compiled Trainer path`
5. `transformers: use compiled PyTorch training`
6. `book: replace chapter-numbered training helpers`
7. `book: integrate compiler notes outside Chapter 11`
8. `outputs: refresh compilation migration notebooks`

The exact split may change if the audit finds a tighter dependency boundary.
Do not combine output manifests with an unreviewed source rewrite.

Before each commit:

```bash
git diff --name-only --cached
git diff --cached --check
git diff --cached
```

Do not use `git add -A` while another process is editing the repository. Do not
force-push. Pull with `--ff-only` before starting a commit group; if the remote
moved, inspect the new commits and rebase or merge without discarding another
author's work.

## 17. Stop conditions

Stop and ask the user before proceeding if any of these occurs:

- the Chapter 7 and 8 process is still active when a required shared file must
  be edited;
- the final Chapter 7 or 8 file map conflicts with this handoff;
- enabling `model.compile()` changes checkpoint keys or optimizer parameter
  identity;
- `torch.export` cannot represent the book's proposed example without a large
  detour;
- the two-GPU compiled path is unstable under PyTorch 2.11;
- the compiler migration requires a DDP rewrite rather than a bounded change;
- a claimed performance benefit does not reproduce under a synchronized
  benchmark;
- a framework tab cannot support the proposed shared lesson without misleading
  prose;
- the output audit identifies unrelated stale work owned by another process.

Compiler fallback is not evidence that the compiled path works. If a notebook
completes because Dynamo graph-breaks around the entire model, report that as a
failed integration.

## 18. Definition of done

This project is complete when all conditions below hold.

1. Computational Performance is Chapter 11, after both RNN chapters and before
   transformers.
2. Transformers is Chapter 12 and Optimization is Chapter 13. Chapter 14 and
   later retain their existing numbers.
3. Active book and generated library content contain no TorchScript API or
   teaching prose.
4. Chapter 11 teaches eager execution, capture, compilation, graph breaks,
   recompilation, and export without merging their meanings.
5. PyTorch benchmarks separate first invocation from steady state and use
   correct CUDA synchronization.
6. `Trainer` offers an opt-in compiled path with eager as the default.
7. Published sustained PyTorch training after Chapter 11 uses compilation where
   it pays for its first-invocation cost; exceptions are documented in the
   audit.
8. Checkpointing still uses `state_dict`; program export uses `torch.export`.
9. MXNet remains fully represented and its hybridization content still runs.
10. Chapters 7 and 8 were audited only after their active rewrite finished.
11. Eager and compiled correctness tests pass on representative CNN, recurrent,
    transformer, and two-GPU paths.
12. All changed notebook outputs are captured with correct framework scope.
13. The multi-GPU host reports no stale output that it has enough hardware to
    regenerate.
14. HTML, slides, and all framework PDFs build successfully.
15. The final source passes `writing-avoid.md`, source lint, cell-ID checks, and
    cross-reference checks.

Report the final commit range, frameworks executed, benchmark hardware, output
audit result, artifact paths, and any deliberate opt-outs. Do not describe an
untested path as complete.
