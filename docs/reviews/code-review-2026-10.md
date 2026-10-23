# Code Review of Basics, Advanced, and the Mathematics Appendix

Date: 2026-10-23. Source: `main` at `732c1431`.
Standard: `docs/code-guide.md` (cited below as "code guide §n").

## Method

Eight read-only reviews covered the 23 chapters in scope, reading every
section's code in PyTorch and JAX, and in TensorFlow or MXNet where present,
together with the `d2l` library code each chapter defines or calls. Reviewers
ran small probes where a claim needed checking; those are marked *verified*
below. Line numbers refer to the source `.md` files at the commit above.

Scale: 3,510 code cells; about 17,350 non-blank lines in the PyTorch tabs and
18,300 in the JAX tabs (Basics and the appendix also carry about 7,400 each in
TensorFlow and MXNet); 46 cells over 40 lines; 355 `#@save` blocks.

## Main Findings

1. **The library stops where the Basics end.** The object-oriented API of
   `oo-design.md` (`Module`, `DataModule`, `Trainer`) is a sound split, but it
   has no training history, no `evaluate`, no scheduler hook, and no lazy
   initialization. Results can only be read back from a plotting buffer, and a
   trained model cannot be pickled (*verified*: `copy.deepcopy` and
   `torch.save` fail on the threaded progress board). So later chapters write
   their own harnesses. Per framework there are about a dozen training loops,
   a dozen evaluators, ten timers with conflicting semantics, five copies of
   `smooth`, and about twenty inline parameter counts. Eight of the nine
   Advanced chapters never use `d2l.Trainer`.
2. **Copying instead of saving.** `ResidualBlock` is defined in six sections
   of ch. 5, and the copies have drifted. ResNet-18 exists four times, `rope`
   five times, EMA in three designs and about twelve copies, the three-mode
   mixture of ch. 17 nine times. Whole sections are copies: `ddim.md:211–255`
   reproduces DDPM's schedule, model, and training loop; `conditional.md`
   copies about 95 of its 273 PyTorch lines from `dcgan.md`; the JAX hot path
   of `actor-critic.md:155–222` is pasted verbatim into `ppo.md:206–269`.
3. **Tabs that disagree change results.** `TinyLM` is initialized differently
   in PyTorch and JAX, so every later experiment needs a different learning
   rate per tab and per-tab prose (ch. 9–11). PyTorch schedules learning
   rates per epoch, JAX per step (ch. 7). JAX's Recipe A in
   `training-recipes.md:557` has no weight decay, so the results table
   compares different recipes. `val_ppl` means two different quantities in
   ch. 12. Max pooling pads with zeros in JAX and TensorFlow and with
   $-\infty$ in PyTorch (*verified*).
4. **Performance habits that should not be taught.** In JAX, `nnx.jit` is
   applied inside functions that are called repeatedly, so every call
   retraces: `train_lm` (*verified*: three calls, three traces), `gpt.md:286`
   (10 compiles), `kv-cache.md:829` (24), `moe.md:706` (12), and the training
   loops of `deltanet.md` and `hybrids.md`. Host synchronization (`.item()`,
   `float()`, `if norm > c`) sits inside hot loops in the library itself,
   including the PyTorch `clip_gradients`.
5. **Two attention APIs and five decoders.** The `MultiHeadAttention` of
   ch. 10 (heads folded into the batch, `valid_lens`) is not what any later
   model uses. `TinyLM`, `TinyCharLM`, `CharLM`, `GPT`, and `GQAAttention`
   each implement 4-D attention themselves, and the return contract differs
   between PyTorch (a tensor) and JAX (a tuple), which spreads `[0]`,
   `(Y, None)`, and `forward(self, X, *_)` through ch. 10–13.
6. **Mechanism hidden behind flags and globals.** Training functions switch
   on strings (`loss_type='rp'`, `'MMD'`, `'classic'`); functions read globals
   defined hundreds of lines earlier (`gamma`, `alpha_bar`, `net`, `T`,
   `MASK`); ch. 17 silently rebinds `T, MASK, alpha_bar`
   (`discrete-diffusion.md:733`). JAX tabs run 10–40% longer than PyTorch,
   mostly from state plumbing that a shared helper or `nnx.Sequential` would
   remove.
7. **Derivatives by hand where autodiff is the lesson** (appendix). The
   appendix computes scores with `np.gradient`, the ELBO gradient and Adam by
   hand, a dense Jacobian where the prose promises a vector–Jacobian product
   (`mdl-odes-solvers.md:1333`), and an "Adam" without bias correction
   (`mdl-score-matching-diffusion-flow.md:303–337`). The section "Jacobian
   Products Without Explicit Jacobians" has no code.

Plain NumPy sections are intended (code guide §1) and are not counted as
defects. The good code in the book follows the same pattern: a short cell that
writes one mechanism exactly as the equation states it, often with an
assertion that checks it (examples under "Keep" below).

## Defects to Fix Regardless of Restructuring

These are wrong or misleading today. *Verified* marks a claim a reviewer
checked by running code.

| Where | Defect |
|---|---|
| `pooling.md:275, :306, :365` (JAX, TF) | max pooling pads with zeros, PyTorch with $-\infty$; results differ (*verified*) |
| `blocks.md:159, :379, :672`, `resnet.md:440`, `cnn-design.md:196` | `net.apply(d2l.init_cnn)` on lazy layers does nothing (*verified*); `RegNetX32` never initializes |
| `scaling-laws.md:605` (JAX) | every swapped block gets `nnx.Rngs(0)`, so all blocks start with identical weights |
| `training-recipes.md:557` (JAX) | Recipe A without weight decay; PyTorch uses `5e-4` |
| `adam.md:384`, `:422` | `TinyLM` initialization differs between tabs, forcing per-tab learning rates |
| `autograd.md:476` (JAX) | `stop_gradient` applied outside the differentiated function: a no-op |
| `calculus.md:414` | `d2l.plot` draws only four of six series (*verified*) |
| `oo-design.md` | trained models cannot be copied or saved (*verified*) |
| `lstm.md:366`, `ssm.md:431`, `mamba.md:661`, `deltanet.md:901` | `val_ppl` computes different quantities in the two tabs |
| `saving-loading.md:117, :124` (JAX, TF) | saves with `np.save` and reloads with `allow_pickle=True`, the path the next subsection warns against |
| `dqn.md:225` | claims a compiled function cannot call `Batch.td_target`; it can (*verified*) |
| `mdl-odes-solvers.md:1333` | builds a dense Jacobian where the prose says none is materialized |
| `mdl-statistics.md:235, :297, :488` | unseeded generator; the prose's claim that the interval brackets the mean fails on about 5% of runs |
| `mdl-geometry-linear-algebraic-ops.md:693–776` | four loaders scale pixels by 1, 255, or 256 |
| MXNet Fashion-MNIST mirror | different bytes from the canonical files (*verified* by md5): MXNet accuracy 0.922 against 0.9155 elsewhere; every MXNet Fashion-MNIST number in the book may differ |
| `mdl-naive-bayes.md:275` | a 630 MB temporary, recomputed five times, for an affine score |
| `efficient-convnets.md:522` | quotes timings ("8.9 versus 13.6 seconds") against the house rule; the MAC counts in its table come from no cell |

## Library Plan

Each item is taught once, in the section listed, and called everywhere else.
The plan follows the code guide §4; the "Callers" column is what each item
replaces.

| Item | Home | Callers |
|---|---|---|
| `Trainer`: steps return values, the trainer logs; `history`, `evaluate`, `(optimizer, scheduler)` from `configure_optimizers`, lazy init; the board owned by the trainer and drawn once per epoch | `oo-design` (ch. 2), `classification` (ch. 3), `training-recipes` (ch. 7) | about 15 board scrapes; two `RecipeTrainer`s; five validation loops; eight `apply_init` lines and five dead `init_cnn` calls |
| `synchronize`, `benchmark(f, *args, warmup, repeats)`, `peak_memory(f, *args)` | `gpus-devices-memory` (ch. 5) | timers in `attention-at-scale`, `kv-cache`, `encoders-decoders`, `ssm`, `matrix-state`, `mamba`, ch. 13 |
| `num_params`, `param_bytes` | `parameters-state-memory` (ch. 5) | about twenty inline counts |
| `ResidualBlock`, `ResNet`, `ResNet18(base=64)` | `model-construction` (ch. 5), `resnet` (ch. 7) | six `ResidualBlock`s; four ResNet-18s, including ch. 13's |
| `EMA` (`AveragedModel` / `optax.ema`), `mixup`, `drop_path` | `training-recipes` (ch. 7) | copies in ch. 7, 16, 17 |
| `lm_perplexity`, vectorized `sample_next` | `rnn-scratch`, `decoding` (ch. 8) | five `val_ppl`s; two sampling implementations |
| `TinyLM`, `SmallCNN` (identical init in both tabs), `lm_loss`, `train_steps(model, data, optimizer, num_steps, loss_fn)`, `evaluate`, `lr_sweep`, `smooth` | a new section in ch. 9, before Adam | about a dozen loops and evaluators in ch. 9–17; `run_lm`, `run_cnn`, `final_loss`, `val_loss`; the `*_ch11` family |
| Scratch optimizers as `torch.optim.Optimizer` / `optax.GradientTransformation` | ch. 9 (as `muon.md:314, :333` already does) | dict-of-hyperparameters optimizers with `t` inside |
| One 4-D `MultiHeadAttention(num_hiddens, num_heads, num_kv_heads=None, rope=False)` with `mask` and `is_causal`; `causal_mask`, `padding_mask`, `rope`, `sinusoidal_encoding` | ch. 10 | five decoders' private attention; five `rope`s; the `valid_lens` shims |
| `GPT` with `num_kv_heads` and `ffn_factory`; `load_gpt2()`; `kv_cache_bytes` | ch. 11 | module surgery after construction; the copied GPT-2 loader |
| `linear_scan(a, b, h0=None, axis=0)`, `MinGRU`, `S4D.discretize`, `SeqStack` | ch. 12 | three scan signatures; duplicated stacks and discretization |
| JAX hot path (`bucket`, jitted `act`, `value_np`, `fit_value`); `policy_step(..., clip=None)`; `TabularMDP.policy_value`; `seed=` on constructors | ch. 14–15 | the pasted JAX block; four exact linear solves; nine framework branches that only seed |
| `rpgan_step`, a JAX `r1_r2_penalty` that accepts closures, `load_cifar10`, `FeatureCNN`, `fid_score`, `kid_score` | ch. 16 | the copies in `conditional.md` |
| `GaussianMixture.three_modes`, `Schedule(T, beta_max)`, `train_ddpm`, `ddim_sample`, `denoising_loss` | ch. 17 | nine mixtures; the DDPM copy in `ddim.md`; `ddim_images` |
| `load_mnist_arrays` (NumPy) | ch. 24 | four-tab loaders in ch. 24 and ch. 27 |

## Framework Idioms

Most JAX issues are the same few patterns: `jit` inside functions, Python loops
where `lax.scan` or `fori_loop` fit, hand-built optimizers and schedules where
`optax` has them, one-off module classes where `nnx.Sequential` fits, and a
global key. Most PyTorch issues are the scheduler and EMA done by hand, mixed
generations of autograd APIs (`.backward`, `autograd.functional`,
`torch.func`), and raw tensors where `nn.Parameter` belongs. Code guide §5
lists the idioms.

Places where the tabs *should* differ and do not yet:

- A first `lax.scan` in Basics: the autoregressive rollout of `sequence.md:380`.
- A JVP/VJP cell in "Jacobian Products Without Explicit Jacobians"
  (`mdl-matrix-calculus-autodiff.md:930`).
- MoE dispatch: sorting tokens and `index_add_` in PyTorch, a capacity-factor
  one-hot einsum in JAX (`moe.md`).
- Generation as one compiled `lax.fori_loop` in JAX (`gpt.md:438`).
- Fixed-horizon vectorized rollouts in JAX, which remove all padding
  (`ppo.md:541` already shows them).

## Chapter Summaries

Severity runs from 1 (polish) to 5 (restructure).

| Ch. | Sev. | Main problems | Keep |
|---|:-:|---|---|
| 1 Preliminaries | 2 | JAX `stop_gradient` no-op; raw matplotlib in probability; global key in JAX | HIV Monte Carlo (`probability.md:950`); `create_graph` against `grad(grad(f))` (`autograd.md:762`) |
| 2 Linear Regression | 4 | the `Trainer` design (finding 1); four step contracts; scratch tensors not `nn.Parameter`; JAX loader depends on TensorFlow | `fit_degree` (`generalization.md:354`); jitted step (`linear-regression-scratch.md:530`) |
| 3 Linear Classification | 3 | `show_heatmaps` ×4 for one matrix; three confusion-matrix algorithms; test loop instead of `evaluate` | stable softmax (`softmax-regression-scratch.md:90`) |
| 4 Multilayer Perceptron | 3 | fold score read from the plot buffer; `k_fold` ×3; 24 activation-plot cells | XOR by hand (`mlp.md:269`); `dropout_layer` |
| 5 Computation | 3 | `ResidualBlock` ×6; parameter counts ×20; JAX saving/loading 40% longer; pickle path | straight-through estimator pair (`custom-layers.md:864`); `AveragedModel` |
| 6 CNNs | 2 | pooling padding bug; `layer_summary` names | im2col as one matmul (`conv-layer.md:521`) |
| 7 Modern ConvNets | 4 | dead initialization; four ResNet-18s; two `RecipeTrainer`s; tabs run different recipes | RepVGG fusion with `allclose` (`efficient-convnets.md:670`) |
| 8 Sequence Models | 3 | rollout without `no_grad`; two samplers; pure-Python `sample_next`; 70-line BPE cell | `NGramLM`; Jacobian powers (`bptt.md:193`) |
| 9 Optimization | 4 | harness copied in every section; `TinyLM` tabs differ; schedules without scheduler APIs; jit inside `train_lm` | scratch Muon (`muon.md:314, :333`); `optax.chain` clipping |
| 10 Attention | 3 | the legacy attention API; `TinyCharLM` re-implements the section's own functions | chunked online softmax with `lax.scan` (`attention-at-scale.md:326`) |
| 11 Transformers | 4 | five decoders; GPT-2 loader copied; GPT forward re-implemented three times; recompiles | preallocated JAX cache (`kv-cache.md:219`); `cost_analysis` pricing |
| 12 State Space Models | 4 | three training styles; duplicated stacks; `val_ppl` disagreement; three scan signatures | doubling scan against `lax.associative_scan` (`ssm.md:222`); WY chunked form (`deltanet.md:605`) |
| 13 Computational Performance | 3 | timers bypass `d2l.Benchmark`; torchrun scripts as strings; tabs measure different workloads; JAX key reuse | `allclose` before benchmark (`compilation.md:166`) |
| 14 Reinforcement Learning | 2 | repeated NumPy idioms; `train_reinforce` in three signatures | `TabularMDP` (`mdp.md:56`); contraction asserted (`value-iter.md:215`) |
| 15 Deep RL | 3 | JAX hot path pasted between sections; five actor-critic loops; results as positional tuples | `Batch.gae` on `backward_scan` (`actor-critic.md:403`); Double DQN as one line |
| 16 GANs | 3 | evaluation kit copied; string-flag training; JAX penalty cannot take a closure | pairing losses and penalties (`convergence.md:247`) |
| 17 Diffusion Models | 4 | DDPM copied into DDIM; mixture ×9; globals rebound; JAX samplers as Python loops | `ancestral_sample` (`ddpm.md:637`); VJP against `jvp`+`vmap` (`score-matching.md:409`) |
| 24 Linear Algebra | 2 | four Fashion-MNIST loaders; `np`/`onp` aliasing | Newton–Schulz polar iteration (`mdl-svd-low-rank.md:868`) |
| 25 Calculus | 3 | `VDual` duplicates `Dual`; four PyTorch API generations; no JVP/VJP code | 36-line reverse-mode tape; HVP tabs |
| 26 Optimization | 3 | variable-suffix walls (`_vr`, `_cd`, `_bfgs`); Adam ×3; slow scalar loops; finite-difference Hessian | simplex projection checked by KKT (`mdl-constrained-optimization-duality.md:493`) |
| 27 Probability and Statistics | 3 | alias mix; unseeded generator; 630 MB broadcast; hand-derived ELBO gradient | vectorized bootstrap; EM in 20 lines |
| 28 Information Theory | 2 | three entropies and three KLs; hand-written log-sum-exp ×5 | distillation gradient by autodiff and in closed form |
| 29 Dynamics | 4 | integrators written inline ten times; mixture score five ways; reflow copies CFM four times; overlaps ch. 17 | Euler/RK4 order slopes (`mdl-odes-solvers.md:611`); JAX CFM with `lax.scan` |

The per-reviewer notes, with every finding and line number, are kept with the
review drafts and can be turned into issues chapter by chapter.

## Order of Work

Code changes cost more than prose changes: each one requires re-executing and
re-capturing the affected notebooks, and a library change can affect every
notebook that imports it (`CLAUDE.md`, "Build dependency notes").

1. **Defects** in the table above. Each is local; most need one notebook
   re-run.
2. **Measurement helpers and `num_params`** in ch. 5. These are additive, and
   later chapters can adopt them one at a time.
3. **The ch. 9 shared-model section** (`TinyLM` with identical initialization,
   `train_steps`, `evaluate`, `lr_sweep`), then adoption in ch. 9–11.
4. **Generative and RL toolkits** (ch. 15–17), which are local to their
   chapters.
5. **The `Trainer` redesign** in `oo-design`. It has the widest effect: every
   Basics notebook in all four frameworks must be re-executed.
6. **The attention API** in ch. 10, which ripples into ch. 11–13. Best done
   together with the ch. 12 rewrite.

## Decisions Needed

- **Trainer redesign.** Approve changing the step contract (steps return
  values, the trainer logs and keeps history). It touches every Basics
  notebook in four frameworks and needs a full re-execution.
- **Advanced training loop.** Use the extended `Trainer` everywhere, or add
  `train_steps` for step-based experiments (recommended: both, with
  `train_steps` taught in ch. 9).
- **Attention API.** Replace the ch. 10 `MultiHeadAttention` with the 4-D
  version that later chapters use.
- **MXNet Fashion-MNIST data.** Point MXNet at the canonical files, which
  changes MXNet's stored numbers.
