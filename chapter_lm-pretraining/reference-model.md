# The Reference Pretraining Run
:label:`sec_lm_reference_model`

<!-- Author brief, copied verbatim from the Language Models part plan
(docs/llm/language_models_part_plan.md, "4.2 The Reference Pretraining Run · T2").
Blueprint: docs/llm/blueprints/ch04_pretraining.md section 4.2; docs/llm/blueprints/ch04_pretraining.md section 4.3.
The plan is a specification, not book text; this comment is replaced
when the section is written.

**Idea:** A run is read through its logs, and the instabilities that appear in them have a mechanism, attention-logit growth, with cures that bound it.
**Opening:** Last week's logs of the reference run show a gradient-norm spike at hour six that the loss curve shows a hundred steps later. The reader knows the recipe (AdamW, WSD, clipping, z-loss) from Advanced; what is new is the run as an object: the quantities to log, the order in which they move, the branch point from which a decay can be rerun, and the ladder of smaller runs that makes the big one's loss predictable. The closing question is how to size a model that will be served, which 4.6 answers.
**Teaches:**
- What to log and why each quantity moves first: per-domain loss and bits per byte, update-to-weight RMS by module (noise-dominated at $\sqrt{(1-\beta_1)/(1+\beta_1)} = 0.229$, signal-dominated near 1), gradient norm, the largest attention logit per layer, MFU from first principles ($6N_{\textrm{mm}} + 12LdT$ FLOPs per token, within 5% of a profiler count), downstream accuracy against tokens; the branch point and checkpoint averaging; selection bias when the best of eleven checkpoints is chosen on the evaluation set.
- Instabilities as one `##` subsection: attention-logit growth from the growth of the query and key weight norms, the output-logit divergence that z-loss damps, and the cures restated with their one difference each (QK-norm bounds the logit by construction, QK-clip rescales the product after the fact, softcapping bounds by a `tanh`, z-loss pulls $\log Z$ to zero, rewind-and-skip removes a bad batch); why QK-norm and QK-clip do not combine.
- Optimizers as a stated fact with its bound: against a tuned AdamW, matrix optimizers (Muon, Soap, Kron) gain at most about 1.4x at small scale and about 1.1x at larger scale, and the gain shrinks with scale under standard weight decay; no optimizer race is run here.
- The reference model and its ladder as the chapter's committed artifact: one T3 job per release that trains the 138M model and three smaller sizes at two token budgets, with the decay branched from each plateau checkpoint so the ladder has intermediate-token points for 4.6's fit.
- Reading a fully open run's logs the same way (SmolLM3-3B's published training logs, read in 4.4 alongside its stage checkpoints).
**Derivation:** Attention-logit growth and its bound. With unit-norm inputs after RMSNorm scaled by a gain $g$, a score satisfies $|s_{ij}| \le g^2 d\, \sigma_{\max}(W_Q)\, \sigma_{\max}(W_K) / \sqrt{d_h}$; gradient flow conserves $W_Q^\top W_Q - W_K^\top W_K$ and the product $M = W_Q W_K^\top$ obeys $dM/dt = -G W_K W_K^\top - W_Q W_Q^\top G$, a positive feedback through the Gram matrices, so the largest logit grows for thousands of steps before the loss shows it. QK-clip multiplies the head's query and key projections by $\sqrt{\tau / S}$ each when the batch maximum $S$ exceeds $\tau$, which scales the head's logits by $\tau / S$ (`sec_muon`); QK-norm gives $|s| \le \sqrt{d_h}\, \|g_q\|_\infty \|g_k\|_\infty$. Smallest case: width 256, gain 20 and head dimension 8 give the bound 640. Checks: the bound holds over $10^5$ samples; the conservation law drifts by less than $10^{-2}$ relative over $2 \cdot 10^4$ gradient-descent steps; after a clip the batch maximum equals $\tau$ to relative `1e-5`, a second clip is a no-op, and heads below $\tau$ are untouched bit for bit.
**Notebook:**
1. Question: what do the reference run's logs say about when each quantity moved, and does a high learning rate spike where QK-clip does not?
2. Model and data: `d2l.r2_config('r2_short')`, the reference architecture at the T2 schedule (2,000 steps, 1.05B tokens, about 3.1 to 4.0 h on one A100 40 GB); its log is committed and read. Live at T2: two arms of the same code for 500 steps, one at ten times the peak learning rate (6e-3) with no cure, one at the same rate with `d2l.qk_clip_` ($\tau$ set after the baseline's logit scale is seen; Kimi K2 uses 100); at the `t1` preset all three arms rerun in minutes. The committed T3 job (`r2`, 6,000 steps, 3.15B tokens, plus the ladder) is launched once per release with `d2l.launch` and its eleven checkpoints are scored with the full item sets.
3. Check: the exact parameter count $P = L(2d^2 + 2dH_{kv}d_h + 3df + 2d + 2d_h) + Vd + d = 138{,}442{,}752$ equals `sum(p.numel())`; FLOPs per token $1{,}509{,}949{,}440$ at $T = 4096$ within 3% of `FlopCounterMode`; the initial loss $\ln V + d\sigma^2/2 = 10.55$ within 0.1; accumulation 1 against 4 on a two-layer toy equal to `1e-5`; resume after a checkpoint reproduces batches exactly and losses within `1e-3` in bf16; `d2l.attn_logit_stats` equals an explicit softmax-input maximum.
4. Claim that survives noise: the unclipped high-learning-rate arm's largest attention logit grows for hundreds of steps and the loss then becomes non-finite or exceeds the baseline's final loss by more than $\delta = 0.3$ (the stability criterion, with $\delta$ checked to exceed three times the two-seed difference); the clipped arm trains at that rate. On the reference run's log, the gradient norm and the largest logit move before the loss at each phase; adjacent checkpoints are compared with a paired item bootstrap and differences inside the T1 three-seed spread are marked "not resolved".
5. Reused: `d2l.EvalSuite(['hellaswag', 'arc_easy', 'piqa', 'sciq', 'winogrande'], n=..., seed=0)` and `d2l.paired_bootstrap` (1.5); `d2l.bits_per_byte` (1.4); `d2l.PackedStream` and `d2l.chunked_lm_loss` (4.1); `d2l.launch` (6.1).
**Figures:** `fig_lm_run_dashboard` (plot: loss, gradient norm, update-to-weight RMS by module, largest attention logit and MFU against steps for the reference run, with the phase boundaries marked); `fig_lm_stable_lr_range` (plot: final loss against learning rate for the baseline and the clipped arm at the `t1` preset, with the $\ell_{\textrm{ref}} + \delta$ band); `fig_lm_ladder_curves` (plot: validation loss against tokens for the four ladder sizes with the branched decays).
**Restate / forward:** `sec_practice` (z-loss adds $\lambda (\log Z)^2$ and pulls the partition function to one; a spike is handled by rewinding about 100 steps and skipping the batches); `sec_muon` (QK-clip rescales a head's projections so its largest logit returns to $\tau$); `subsec_scheduler-branching` (the decay can be branched from any plateau checkpoint, which is how the ladder gets two points per run); `sec_scaling-laws` (the parameter count and the 6ND FLOP count). Forward: 4.6 (the ladder fitted).
**Exercises:** 6 to 8. Derive: the logit bound under QK-clip and the Gram-matrix change one clip makes; why $Vd$ is counted once in parameters and $6Vd$ in FLOPs; the fraction of the learning-rate integral the WSD decay holds ($\delta/(2 - \delta)$, about 8% at $\delta = 0.15$). Measure: update-to-weight ratios by module against the noise floor 0.229; a learning-rate sweep at the `t1` preset; four ways to finish the T1 run from its plateau checkpoint (short decay, long decay, EMA, average of four). Open: the Muon variant of the `t1` preset against the stated bound.
**Writer notes:**
- One experiment per notebook: the live T2 cells run only the two 500-step instability arms; the baseline `r2_short` log and the T3 artifacts are committed and read. If the editor rules against a T3 job (blueprint ch04 §4.2 Risks), `r2_short` alone becomes the reference model (about 6.9 tokens per parameter) and the ladder shrinks to two sizes.
- The ladder: three sizes below 138M on the same depth-width recipe (the `t1` preset at 25.2M, and two intermediate presets to be fixed by the author, about 50M and 85M), each at two token budgets chosen so the ladder adds about 40% to the reference run's $4.75 \cdot 10^{18}$ FLOP (budgets near 10 and 20 tokens per parameter satisfy this; the decay is branched from the plateau at both budgets so every run yields two finished points, and the plateau checkpoints supply the intermediate-token points). Commit the configs, logs and per-item evaluation scores; 4.6 and A5 depend on them. Settle the exact sizes and budgets before writing 4.6.
- `d2l.qk_clip_(model, stats, tau, alpha=0.5)` is a `sec_muon` export that does not exist yet (muon.md:1215 states the rule in prose); carry a local copy with the contract (in place after an optimizer step; per head with batch maximum above $\tau$, scale $W_q$ rows by $(\tau/S)^\alpha$ and $W_k$ rows by $(\tau/S)^{1-\alpha}$; returns per-layer factors; no-op with a warning when `qk_norm=True`) and delete it when the book-side edit lands.
- The fused attention kernel does not return scores, so the explicit `math` backend probe of `d2l.attn_logit_stats` is mandatory for the logit monitor; gradient clipping can mask pre-spike growth, so log the norm before clipping; a spike that disappears on same-checkpoint replay is not fixed (replay from an earlier checkpoint too); a resumed run whose scheduler step restarted at zero is the classic silent failure; the skip set changes the data order and the stream state must record it.
- MFU depends on the FLOP convention by 10 to 30%: state the PaLM convention once and use `d2l.flops_per_token` everywhere (6.2 and 7.6 import it). Near-chance accuracies carry no information at 138M (ARC-Challenge, MMLU); print chance levels and plot loss only for those.
- Fallback ladder if no instability appears at the high rate: extend the rate grid, set $\beta_2 = 0.999$ with no warmup, run 800 steps, widen to 512; run once during authoring and commit the setting. Replay nondeterminism on Ada is measured and reported as the floor.
- Reference implementations: TorchTitan re-expresses the configuration and the first 200 steps are compared within the spread of three short runs; `lm-evaluation-harness` cross-checks HellaSwag on the final checkpoint.
- Do not re-teach: AdamW and what not to decay (`sec_adamw`, `subsec_adamw-what-not-to-decay`), WSD (`sec_scheduler`), clipping and EMA (`sec_practice`), z-loss and rewind (`subsec_practice-nan`, practice.md:322-340), QK-norm (`subsec_qk-norm` and its exercise, which already trains with and without QK-norm across learning rates), QK-clip (`sec_muon`), the 6ND count (`sec_scaling-laws`), muP (`sec_scaling`). New here: the logit-growth derivation, the combined stability map, the ladder, the optimizer bound as a fact.
- Blueprint: ch04 §4.2 (R2 spec and its rationale: 18 layers, GQA 3:1, embedding share 18.2%, 22.7 tokens per parameter; `tab_lm_r2_logs`; the OLMo 2 trajectory, which 4.4 replaces by SmolLM3-3B's stage checkpoints) and §4.3 (the instability testbed, variants V0 to V5, the stability criterion, the spike protocol with `tab_lm_inst_diagnosis`, D1 and D2 with their checks). This section merges them around one run, and the open-trajectory reading is in 4.4.
**Sources:**
- Papers: [Small-scale proxies for large-scale Transformer training instabilities](https://arxiv.org/abs/2309.14322) (logit growth reproduced at small scale; the learning-rate sensitivity measure); [Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534) (QK-clip in a trillion-parameter run without a spike); [Fantastic Pretraining Optimizers and Where to Find Them](https://arxiv.org/abs/2509.02046) (the 1.1x to 1.4x bound on matrix optimizers); [Fantastic Pretraining Optimizers and Where to Find Them II: Hyperball Optimization](https://arxiv.org/abs/2606.16899) (the advantage shrinking with scale under weight decay); [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) (the production Muon recipe); [PaLM](https://arxiv.org/abs/2204.02311) (z-loss, restated); [Understanding Warmup-Stable-Decay Learning Rates](https://arxiv.org/abs/2410.05192) (the river-valley picture of WSD branching); [Olmo 3](https://arxiv.org/abs/2512.13961), [2 OLMo 2 Furious](https://arxiv.org/abs/2501.00656) and [Dolma](https://arxiv.org/abs/2402.00159) (open runs with published logs and data); [OLMES](https://arxiv.org/abs/2406.08446) (evaluation conventions for small models during training); [SmolLM2](https://arxiv.org/abs/2502.02737) (a documented small-model run).
- Code and documentation: [OLMo](https://github.com/allenai/OLMo) and [OLMo-core](https://github.com/allenai/OLMo-core) (instrumentation and checkpoint state); [TorchTitan](https://github.com/pytorch/torchtitan) (the cross-check system); [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (the HellaSwag cross-check); [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) (small-model records).
- Courses, tutorials and blogs: [Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) (the monitoring checklist); [build-nanogpt](https://github.com/karpathy/build-nanogpt) and [nanochat](https://github.com/karpathy/nanochat) (the incremental loop and a current small pipeline); [CS336 assignment 1](https://github.com/stanford-cs336/assignment1-basics) (the trainer-state checkpoint contract).
-->

<!-- Placeholder (prose): opening. A run is read through its logs; the reader knows the recipe (AdamW, WSD, clipping, z-loss) from the Advanced part; what is new is the run as an object. The opening question: what do the reference run's logs say about when each quantity moved, and does a high learning rate spike where QK-clip does not? -->

```{.python .input #reference-model-imports}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #reference-model-the-reference-pretraining-run-1}
%%tab pytorch
import collections
import dataclasses
import pandas as pd
import tempfile
import time
import warnings
from torch.utils.flop_counter import FlopCounterMode
```

<!-- Placeholder (prose): the log's largest gradient event, read before anything is explained; the section returns to it in "Who Moves First". -->

```{.python .input #reference-model-opening}
%%tab pytorch
MILD, DELTA = 3, 0.3  # a mild event; a loss failure: DELTA above baseline
log = d2l.reference_run_log()
event = d2l.gradient_event(log, mild=MILD, delta=DELTA)
print(f'largest gradient event: step {event.step} (hour {event.hour:.1f}), '
      f'norm {event.ratio:.1f}× its running median'
      + (' (mild)' if event.mild else ''))
print(f'{event.source}, step {event.at}: largest attention logit '
      f'{event.max_logit:.0f} (layer {event.layer})')
change = (f'loss non-finite at step {event.nonfinite}' if event.nonfinite
          else f'loss change over the next 200 steps: {event.dloss:+.3f}')
print(f'{event.source}: {change}')
```

## The Reference Model and What It Costs

### The Configuration

<!-- Placeholder (prose): the reference configuration r2 and its presets: shapes, schedules, why 18 layers and GQA 3:1, the T1 preset readers can rerun, the ladder sizes (restate subsec_scheduler-branching in one sentence for the branches). -->

```{.python .input #reference-model-config}
%%tab pytorch
#@save
Arch = collections.namedtuple('Arch',
                              'L d H H_kv d_h f vocab T rope_base qk_norm tie')

@dataclasses.dataclass  #@save
class PretrainConfig:
    """One pretraining run: the architecture, the batch (in sequences of
    arch.T tokens), the WSD schedule, AdamW, and what the run saves."""
    preset: str
    arch: Arch
    steps: int
    global_batch: int
    micro_batch: int
    peak_lr: float
    warmup: int
    decay: int
    final_lr: float = 0.0
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    wd: float = 0.1
    clip: float = 1.0           # None: log the gradient norm, never clip
    clip_from: int = None       # clip after this many updates (None: all)
    z_loss: float = 1e-4
    dtype: str = 'bf16'
    compile: bool = True
    ddp: bool = False
    probe_every: int = 50       # update RMS per module
    stats_every: int = 50       # attention logit and entropy, log Z
    val_every: int = 250
    val_tokens: int = 2 ** 21
    ckpt_steps: tuple = ()      # bf16 weights after these steps
    state_steps: tuple = ()     # full training state after these steps
    state_every: int = 0        # and every state_every steps (last two kept)
    branches: tuple = ()        # (from_step, decay_steps) after the main run
    data: str = '100m'
    data_seed: int = 0
    seed: int = 0
    out_dir: str = None

    def accum(self, world):
        """Micro-batches per step on each of world GPUs."""
        if self.global_batch % (self.micro_batch * world):
            raise ValueError(f'{self.global_batch} sequences do not split '
                             f'into {world} x {self.micro_batch}')
        return self.global_batch // (self.micro_batch * world)
```

<!-- Placeholder (prose): the presets as two tables (shapes; schedules) and r2_config, which builds a preset's PretrainConfig. -->

```{.python .input #reference-model-preset-tables}
%%tab pytorch
#@save
R2_ARCHS = {  # L, d, H, H_kv, f, qk_norm, T (head width 64, V = 32,768)
    'r2': (18, 768, 12, 4, 2048, False, 4096),
    'r2_short': (18, 768, 12, 4, 2048, False, 4096),
    't1': (8, 384, 6, 2, 1024, False, 1024),
    'fp8': (8, 768, 12, 4, 2048, True, 1024),
    'ladder-25m': (8, 384, 6, 2, 1024, False, 4096),
    'ladder-54m': (10, 576, 9, 3, 1536, False, 4096),
    'ladder-88m': (10, 768, 12, 4, 2048, False, 4096)}

#@save
R2_SCHEDULES = {  # sequences per step, steps, warmup, decay, peak LR, data
    'r2': (128, 6000, 300, 900, 6e-4, '4b'),
    'r2_short': (128, 2000, 100, 300, 6e-4, '4b'),
    't1': (64, 1526, 76, 229, 1e-3, '100m'),
    'fp8': (16, 1221, 60, 183, 1e-3, '100m'),
    'ladder-25m': (64, 1920, 48, 288, 1e-3, '4b'),
    'ladder-54m': (64, 4141, 104, 621, 8e-4, '4b'),
    'ladder-88m': (64, 6721, 168, 1008, 7e-4, '4b')}

#@save
def r2_config(preset='r2', **overrides):
    """The PretrainConfig of a preset; keyword arguments replace fields."""
    L, d, H, H_kv, f, qk_norm, T = R2_ARCHS[preset]
    arch = Arch(L, d, H, H_kv, d // H, f, 32768, T, 1e4, qk_norm, True)
    B, steps, warmup, decay, lr, data = R2_SCHEDULES[preset]
    # the largest micro-batch that fits a 24 GB card (r2 at 8 does not)
    micro = 16 if T < 4096 else 4 if preset.startswith('r2') else 8
    cfg = PretrainConfig(preset, arch, steps, B, micro_batch=micro,
                         peak_lr=lr, warmup=warmup, decay=decay, data=data,
                         **d2l._preset_saves(preset))
    return dataclasses.replace(cfg, **overrides)
```

```{.python .input #reference-model-presets}
%%tab pytorch
print(f'{"preset":>10} {"L":>3} {"d":>4} {"heads":>5} {"f":>5} {"T":>5} '
      f'{"batch tokens":>12} {"steps":>5} {"tokens":>13}')
for preset in R2_ARCHS:
    c = r2_config(preset)
    a, B = c.arch, c.global_batch * c.arch.T
    print(f'{preset:>10} {a.L:>3} {a.d:>4} {f"{a.H}/{a.H_kv}":>5} {a.f:>5} '
          f'{a.T:>5} {B:>12,} {c.steps:>5} {B * c.steps:>13,}')
```

### Parameters

<!-- Placeholder (prose): the exact parameter count, term by term (attention projections with H d_h, SwiGLU's three matrices, RMSNorm gains, the tied embedding counted once); restate the 6ND count of sec_scaling-laws in one sentence. -->

```{.python .input #reference-model-param-count}
%%tab pytorch
cfg = r2_config('r2')
L, d, H, H_kv, d_h, f, V = (cfg.arch.L, cfg.arch.d, cfg.arch.H,
                            cfg.arch.H_kv, cfg.arch.d_h, cfg.arch.f,
                            cfg.arch.vocab)
P_formula = L * (2 * d * H * d_h + 2 * d * H_kv * d_h + 3 * d * f + 2 * d
                 + 2 * d_h * cfg.arch.qk_norm) + V * d + d
with torch.device('meta'):  # shapes only, no memory
    P = sum(p.numel() for p in d2l.PackedGPT.from_arch(cfg.arch).parameters())
assert P == P_formula
D = cfg.steps * cfg.global_batch * cfg.arch.T
print(f'parameters: {P_formula:,} (formula) = {P:,} (counted)')
print(f'embedding share: {V * d / P:.1%}')
print(f'tokens per parameter: {D / P:.1f}')
```

### FLOPs per Token and MFU

<!-- Placeholder (prose): 6 N_mm + 12 L H d_h T per token, the PaLM convention stated once (full score matrix; causal_half halves the attention term; MFU moves 10-30% with the convention); why V d counts once in parameters and 6 V d in FLOPs. -->

```{.python .input #reference-model-flops}
%%tab pytorch
#@save
def flops_per_token(arch, T=None, causal_half=False):
    """Training FLOPs per token, 6 N_mm + 12 L H d_h T, of an Arch or a
    Transformers config (divide by 3 for a forward pass). N_mm counts the
    matrix parameters (the head once more when untied); T=None leaves out
    the attention scores; causal_half halves them."""
    L, d, H, H_kv, d_h, f, V, tie = d2l._arch_fields(arch)
    N_mm = L * (2 * d * H * d_h + 2 * d * H_kv * d_h + 3 * d * f) + V * d
    N_mm += 0 if tie else V * d
    attn = 0 if T is None else 12 * L * H * d_h * T // (2 if causal_half
                                                        else 1)
    return 6 * N_mm + attn

#@save
def mfu(tokens_per_s, arch, peak_flops, n_gpus=1, *, T=None,
        causal_half=False):
    """Model FLOPs utilization: achieved training FLOP/s over the peak of
    n_gpus; T defaults to the arch's sequence length, if it has one."""
    T = T if T is not None else getattr(arch, 'T', getattr(arch, 'seq_len',
                                                           None))
    F_tok = flops_per_token(arch, T=T, causal_half=causal_half)
    return tokens_per_s * F_tok / (peak_flops * n_gpus)
```

<!-- Placeholder (prose): the formula against PyTorch's FLOP counter on one forward and backward pass. On the meta device the counter counts only matrix products, and it counts attention as the full square score matrix (the PaLM convention), so the two agree by construction (ratio 1.000): the check catches a wrong term in the formula, not a cost the formula leaves out; norms, softmax and the elementwise work are not counted by either. -->

```{.python .input #reference-model-flops-check}
%%tab pytorch
T = cfg.arch.T
with torch.device('meta'):  # shapes only: no memory, no arithmetic
    model = d2l.PackedGPT.from_arch(cfg.arch)
    X = torch.randint(V, (1, T))
with FlopCounterMode(display=False) as counter:
    logits = model(X)
    F.cross_entropy(logits.flatten(0, 1).float(), X.flatten()).backward()
F_formula = flops_per_token(cfg.arch, T=T)
F_counted = counter.get_total_flops() / T
print(f'formula: {F_formula:,} FLOPs per token')
print(f'counted: {F_counted:,.0f}')
print(f'ratio: {F_counted / F_formula:.3f}')
```

### The Loss at Initialization

<!-- Placeholder (prose): ln V + d sigma^2 / 2 from tied N(0, sigma^2) embeddings and unit-RMS final hidden states; a measured initial loss outside 0.1 of it is a bug in the initialization. -->

```{.python .input #reference-model-init-loss}
%%tab pytorch
sigma = 0.02  # the initialization's standard deviation
predicted = math.log(V) + d * sigma ** 2 / 2
torch.manual_seed(0)
model = d2l.PackedGPT.from_arch(cfg.arch)
measured = d2l.val_loss(model, cfg, tokens=4 * T)  # four sequences
assert abs(measured - predicted) < 0.1
print(f'predicted: {predicted:.2f}')
print(f'measured: {measured:.2f}')
```

## The Loop and What It Logs

### The Schedule and the Parameter Groups

<!-- Placeholder (prose): restate WSD (sec_scheduler) and AdamW's groups (subsec_adamw-what-not-to-decay) in one sentence each; the fraction of the learning-rate integral the decay holds is an exercise. -->

```{.python .input #reference-model-wsd}
%%tab pytorch
#@save
def wsd_lr(step, total, warmup, decay, peak, final=0.0):
    """Warmup-stable-decay: linear warmup, constant peak, linear decay to
    final over the last decay steps."""
    if step < warmup:
        return peak * (step + 1) / warmup
    if step < total - decay:
        return peak
    return final + (peak - final) * (total - step) / decay

#@save
def param_groups(model, wd):
    """AdamW groups: weight decay on the matrices, none on embeddings,
    position tables and gains."""
    decay, rest = [], []
    for name, p in model.named_parameters():
        no_decay = p.ndim < 2 or name.split('.')[0] in ('token_emb', 'pos_emb')
        (rest if no_decay else decay).append((name, p))
    return [dict(params=[p for _, p in g], names=[n for n, _ in g],
                 weight_decay=w) for g, w in ((decay, wd), (rest, 0.0))]
```

```{.python .input #reference-model-groups-check}
%%tab pytorch
groups = param_groups(model, cfg.wd)
n_t, n_p = ([len(g['params']) for g in groups],
            [sum(p.numel() for p in g['params']) for g in groups])
print(f'decayed tensors: {n_t[0]} ({n_p[0]:,} parameters); not decayed: '
      f'{n_t[1]} ({n_p[1]:,})')
steps = (0, cfg.warmup - 1, cfg.steps - cfg.decay,
         cfg.steps - cfg.decay // 2, cfg.steps - 1)
lrs = [wsd_lr(t, cfg.steps, cfg.warmup, cfg.decay, cfg.peak_lr)
       for t in steps]
print(f'lr at steps {", ".join(map(str, steps))}: '
      + ', '.join(f'{lr:.2e}' for lr in lrs))
```

### One Step

<!-- Placeholder (prose): token-weighted accumulation (each micro-batch's summed loss over the step's global count n_valid), z-loss (restate subsec_practice-nan in one sentence), the pre-clip norm logged because clipping hides the growth that precedes a spike; clip_from turns clipping on mid-run, counted like state_steps (clip_from=250: log steps 251 on are clipped, so a bad batch at log step 300 is; the spike arms switch it on at their branch step); log_z, the mean log-partition of the output softmax over the step's targets, comes from the logits the loss already has (what z-loss pulls to zero). -->

```{.python .input #reference-model-step}
%%tab pytorch
#@save
def pretrain_step(model, optimizer, micro_batches, cfg, n_valid, step=None):
    """One step on this rank's micro-batches; n_valid is the step's global
    count of valid targets, so the gradient is the token-weighted mean over
    every rank and micro-batch. step counts the updates before this one,
    as state_steps does, so clipping at cfg.clip starts at log step
    cfg.clip_from + 1. Returns loss, pre-clip norm, clipped and the mean
    log-partition log_z of the output softmax."""
    world, nll, lse = d2l._rank_world()[1], 0.0, 0.0
    for k, mb in enumerate(micro_batches):
        ctx = d2l.PackCtx(mb['doc_ids'], mb['pos_ids'])
        with d2l._no_sync(model, k < len(micro_batches) - 1):  # sync last
            with torch.autocast(mb['input_ids'].device.type, torch.bfloat16,
                                enabled=cfg.dtype == 'bf16'):
                H, W = model(mb['input_ids'], ctx, return_hidden=True)
                loss, _, part, part_lse = d2l.chunked_lm_loss_and_nll(
                    H, W, mb['labels'], z_loss=cfg.z_loss, return_lse=True)
            (loss * world / n_valid).backward()  # DDP averages the ranks
        nll, lse = nll + part.detach(), lse + part_lse.detach()
    on = step is None or cfg.clip_from is None or step >= cfg.clip_from
    clip = (cfg.clip if on else None) or math.inf
    params = [p for g in optimizer.param_groups for p in g['params']]
    norm = torch.nn.utils.clip_grad_norm_(params, clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    loss, log_z = d2l._all_reduce_sum(torch.stack([nll, lse])) / n_valid
    return dict(loss=loss, grad_norm=norm, clipped=norm > clip, log_z=log_z)
```

### The Loop

<!-- Placeholder (prose): what the loop logs and when (every step: loss, pre-clip norm, clipped, tokens/s, MFU; every probe_every steps: update RMS per module; every stats_every steps: the largest attention logit and the mean attention entropy per layer from an explicit-score probe, since the fused kernel returns no scores (the entropy is the mean over the queries and heads of one packed training sequence, rank 0's first row, so it carries some packing noise: a document's first query has entropy 0), and log_z; every val_every steps: validation), checkpoints, full states (state_steps: the resumable states a later replay or branch starts from), branches; the DDP and state plumbing lives in the library. -->

```{.python .input #reference-model-pretrain}
%%tab pytorch
#@save
def pretrain(model, stream, cfg, val_fn=None, hooks=(), resume=None):
    """Train model on stream under cfg, then each of cfg.branches from its
    saved state; one row per step (a DataFrame). resume: a state file, or
    'auto' for the latest one in cfg.out_dir."""
    run = d2l._PretrainRun(model, stream, cfg, val_fn, resume)
    for seg in run.segments():  # the main run, then each branch
        for step in range(seg.start, seg.total):
            lr = wsd_lr(step, seg.total, cfg.warmup, seg.decay, cfg.peak_lr,
                        cfg.final_lr)
            for group in run.optimizer.param_groups:
                group['lr'] = lr
            t0 = time.perf_counter()
            batch, micro, n_valid = run.batch(step)  # this rank's part
            t1 = time.perf_counter()
            before = run.snapshot(step)  # weights before, at probe steps
            out = pretrain_step(run.model, run.optimizer, micro, cfg, n_valid,
                                step)
            row = dict(step=step + 1, tokens=(step + 1) * run.step_tokens,
                       lr=lr, **out)
            run.record(seg, row, batch, before, t1 - t0, t0)
            for hook in hooks:
                hook(step, run.raw, run.optimizer, batch, row)
            if run.stopped(row):  # a non-finite loss ends the segment
                break
    return run.finish()
```

<!-- Placeholder (prose): two checks of the loop on a two-layer toy on the CPU: accumulation does not change the step, and a job killed after its step-20 state resumes onto the same batches and losses (a resumed run whose scheduler restarted at zero is the classic silent failure; the job asserts it). -->

```{.python .input #reference-model-loop-checks}
%%tab pytorch
train = d2l.reference_shards('100m')
toy = d2l.Arch(L=2, d=32, H=4, H_kv=2, d_h=8, f=64, vocab=V, T=32,
               rope_base=1e4, qk_norm=False, tie=True)

def toy_run(micro, out_dir=None, hooks=(), resume=None):
    c = PretrainConfig('toy', toy, steps=40, global_batch=8, micro_batch=micro,
                       peak_lr=3e-3, warmup=4, decay=8, dtype='fp32',
                       compile=False, probe_every=0, stats_every=0,
                       val_every=0, state_steps=(20,), out_dir=out_dir)
    torch.manual_seed(0)
    model = d2l.PackedGPT.from_arch(toy, backend='sdpa')
    stream = d2l.PackedStream(train, toy.T, c.global_batch)
    return pretrain(model, stream, c, hooks=hooks, resume=resume)

full, micro4 = toy_run(8), toy_run(2)  # one micro-batch of 8, four of 2
assert (full['loss'] - micro4['loss']).abs().max() < 1e-4
print(f'accumulation 1 vs 4: max |Δ loss| = '
      f'{(full["loss"] - micro4["loss"]).abs().max():.1e}')
```

```{.python .input #reference-model-loop-resume}
%%tab pytorch
seen = collections.defaultdict(dict)  # run: {step: its batch}
def keep(run):
    return lambda step, model, opt, batch, row: seen[run].__setitem__(
        row['step'], batch['input_ids'])

class Crash(Exception):
    pass

def crash_after_20(step, model, opt, batch, row):
    if row['step'] == 20:
        raise Crash  # the job dies after writing its step-20 state

full = toy_run(8, hooks=[keep('full')])
with tempfile.TemporaryDirectory() as job:
    try:
        toy_run(8, job, hooks=[crash_after_20])
    except Crash:
        pass
    resumed = toy_run(8, job, hooks=[keep('resumed')], resume='auto')
same = all(torch.equal(seen['full'][t], b) for t, b in seen['resumed'].items())
assert same and len(seen['resumed']) == 20
print(f'resume at step 20: batches {"identical" if same else "differ"}, '
      f'max |Δ loss| = {(full["loss"] - resumed["loss"]).abs().max():.1e}')
```

### Three Seeds at the T1 Preset

<!-- Placeholder (prose): the t1 preset (25M parameters, 100M tokens) as the reader's own run on one 16 GB GPU, three seeds; their spread is the noise floor every later comparison is read against. -->

```{.python .input #reference-model-t1-runs}
%%tab pytorch
t1 = d2l.launch_pretrain([r2_config('t1', seed=s) for s in (0, 1, 2)],
                         n_gpus=1)
t1_cfg = r2_config('t1')
print(f'{"seed":>4} {"val loss":>8} {"val bits per byte":>17} '
      f'{"tokens/s":>8} {"MFU":>5}')
finals = []
for seed, g in t1[t1['run'] == 't1'].groupby('seed'):
    stable = g[g['step'].between(t1_cfg.warmup + 1,
                                 t1_cfg.steps - t1_cfg.decay)]
    finals.append(g['val_loss'].dropna().iloc[-1])
    bpb = d2l.val_bpb(finals[-1], t1_cfg)
    print(f'{seed:>4} {finals[-1]:8.3f} {bpb:17.3f}'
          f' {stable["tok_s"].median() / 1e3:7.0f}k '
          f'{stable["mfu"].median():5.2f}')
t1_spread = max(finals) - min(finals)
print(f'spread (max − min): {t1_spread:.3f}')
```

### The Reference Run's Log

<!-- Placeholder (prose): the reference run on four RTX 4090s (DDP over shared memory, no peer-to-peer): what it cost, its MFU from first principles, where it ended. A sentence that needs the four-card run carries the campaign marker for lm-r2-train-log-v1 while that run is pending. -->

```{.python .input #reference-model-reference-run}
%%tab pytorch
log = d2l.launch_pretrain(r2_config('r2'), n_gpus=4)
main = log[log['run'] == log['run'].iloc[0]]
run_cfg = r2_config(main['run'].iloc[0])  # t1 while the run is pending
stable = main[main['step'].between(run_cfg.warmup + 1,
                                   run_cfg.steps - run_cfg.decay)]
q1, med, q3 = stable['mfu'].quantile([0.25, 0.5, 0.75])
hours = (main['time'].iloc[-1] - main['time'].iloc[0]
         + main['step_s'].iloc[0]) / 3600
end = main['val_loss'].dropna().iloc[-1]
last = main.filter(like='max_logit_').dropna(how='all').iloc[-1].astype(float)
gpu = main['gpu'].iloc[0].replace('NVIDIA GeForce ', '')
print(f'tokens: {main["tokens"].iloc[-1]:,} in {main["step"].iloc[-1]:,} '
      'steps')
print(f'wall clock: {hours:.1f} h on {main["world"].iloc[0]} × {gpu} '
      f'(restarts: {main["job_start"].nunique() - 1})')
print(f'MFU (stable phase, median [IQR]): {med:.2f} [{q1:.2f}, {q3:.2f}]')
print(f'final val loss: {end:.2f} (bits per byte: '
      f'{d2l.val_bpb(end, run_cfg):.3f})')
print(f'largest attention logit at the end: {last.max():.0f} '
      f'(layer {int(last.idxmax().rsplit("_", 1)[1]) + 1})')
print(f'steps clipped: {main["clipped"].mean():.0%}')
```

<!-- Placeholder (prose): the dashboard: loss, pre-clip gradient norm, update RMS per module, the largest attention logit and MFU against steps, the warmup end and the decay start marked. -->

```{.python .input #reference-model-dashboard}
%%tab pytorch
boundaries = (run_cfg.warmup, run_cfg.steps - run_cfg.decay)
d2l.plot_run_log(log, ['loss', 'grad_norm', 'rms', 'max_logit', 'mfu'],
                 marks=boundaries)
```

### Update Size by Module

<!-- Placeholder (prose): Adam's update in units of the learning rate has RMS sqrt((1 - beta_1)/(1 + beta_1)) = 0.229 when the gradient is noise; near 1 when it is signal. Stable phase (from a sixth of the run, step 1,000 of r2, past warmup's transient) against decay phase, per module type. One sentence on token_emb: its RMS averages over every row, including the rows of tokens absent from the step's batch (no gradient, an update from the moments alone), so its ratio to 0.229 is not the noise-floor reading the matrices give. -->

<!-- Placeholder (prose): claim 5 as measured (F12, from the committed log): per unit learning rate every module but W_v sits near the floor in the stable phase (ratios 0.83-0.94; W_v 0.73, named: its per-lr RMS climbs toward the floor through the stable phase, from about 0.60 of it near step 1,000 to 0.79 near 5,000, so 0.73 is the median of a rising series and part of its decay ÷ stable 1.19 is that trend); in the decay the per-unit-lr RMS rises slightly (decay ÷ stable 1.02-1.19, W_v the most) while the update itself falls with the learning rate (absolute decay ÷ stable 0.48-0.56, the decay's median lr being 0.47 of the peak). Never write that the per-lr value falls in the decay. Two units, two columns: say which one each sentence reads. -->

```{.python .input #reference-model-update-size-by-module-1}
%%tab pytorch
#@save
def update_rms_by_phase(log, cfg, stable_from):
    """Medians per module kind of the logged update RMS (columns rms_<kind>,
    in units of the learning rate) over the stable phase (steps stable_from
    to the decay's start) and the decay; abs_<phase> multiplies them by the
    phase's median learning rate: the size of the update itself."""
    run = log.set_index('step')
    rms = run.filter(like='rms_').dropna(how='all')
    decay_from = cfg.steps - cfg.decay
    out = {}
    for phase, rows in (('stable', rms.loc[stable_from:decay_from]),
                        ('decay', rms.loc[decay_from + 1:])):
        assert len(rows), f'the {phase} phase has no probe steps'
        out[phase] = rows.median()
        out[f'abs_{phase}'] = out[phase] * run.loc[rows.index, 'lr'].median()
    return pd.DataFrame(out).rename(lambda c: c.removeprefix('rms_'))
```

```{.python .input #reference-model-update-rms}
%%tab pytorch
beta1 = run_cfg.betas[0]
noise_floor = math.sqrt((1 - beta1) / (1 + beta1))
STABLE_FROM = run_cfg.steps // 6  # r2: step 1,000, past warmup's transient
MODULES = ['W_q', 'W_k', 'W_v', 'W_o', 'W_g', 'W_1', 'W_2', 'token_emb',
           'norm']
rms = update_rms_by_phase(main, run_cfg, STABLE_FROM).loc[MODULES]
per_lr = rms['decay'] / rms['stable']
absolute = rms['abs_decay'] / rms['abs_stable']
assert (absolute < 1).all(), 'an update grew in the decay'
print(f'update RMS per unit learning rate ({run_cfg.preset} run); '
      'absolute: times the median lr')
print(f'{"module":>9} {"stable":>7} {"decay":>7} '
      f'{f"stable ÷ {noise_floor:.3f}":>15} '
      f'{"decay ÷ stable: per lr":>23} {"absolute":>9}')
for m in MODULES:
    print(f'{m:>9} {rms.at[m, "stable"]:7.3f} {rms.at[m, "decay"]:7.3f} '
          f'{rms.at[m, "stable"] / noise_floor:15.2f} {per_lr[m]:23.2f} '
          f'{absolute[m]:9.2f}')
```

### Epsilon Headroom

<!-- Placeholder (prose): Adam divides by sqrt(v) + eps. eps = 1e-8 is correct at this scale and wrong at large scale without a visible error: once sqrt(v) of many coordinates falls toward eps, eps silently shrinks their updates (OLMo 2 moved from 1e-5 to 1e-8; Molybog et al. see it at 7B-546B). The readout: per parameter group (embedding, attention, MLP, gains) the median of sqrt(v), the bias-corrected second moment's root that eps is added to, and the shares of coordinates below 1e-5 and 1e-7, from two committed full states: t1 at step 1,297 (end of its plateau) and the reference run at 5,100. Then one pair at t1: the T1 grid's clip cell at lr 1e-3 (eps 1e-8, seeds 0 and 1; the same run, not re-run) against one new run, seed 0, whose eps is the t1 median of sqrt(v) over the matrices to one significant digit, fixed before it ran; the two runs differ in eps only. "update RMS ÷ lr" is the median over the constant phase (steps 76 to 1,297) of the matrices' update RMS in units of the learning rate (the measure of "Update Size by Module"). -->

```{.python .input #reference-model-epsilon-headroom}
%%tab pytorch
PAIR_LR = 1e-3  # t1's peak rate; the T1 grid's clip cell at it is eps 1e-8
headroom = d2l.adam_eps_headroom(PAIR_LR)
med = headroom.readout.set_index(['model', 'group'])['median']
if len(headroom.pair):  # the new eps: the t1 matrices' median, 1 digit
    assert headroom.eps == float(f'{headroom.matrix_median:.0e}')
    assert (min(med['t1', 'attention'], med['t1', 'MLP'])
            <= headroom.matrix_median
            <= max(med['t1', 'attention'], med['t1', 'MLP']))
    sci = lambda x: f'{x:.0e}'.replace('e-0', 'e-')
    print(f'{"model":>5} {"group":>9} {"median sqrt(v)":>14} '
          f'{"below 1e-5":>10} {"below 1e-7":>10}')
    for r in headroom.readout.itertuples():
        print(f'{r.model:>5} {r.group:>9} {r.median:14.1e} '
              f'{r.below_hi:10.2%} {r.below_lo:10.2%}')
    base = headroom.pair[headroom.pair['source'] == 'instability-t1']
    base = base.set_index('seed')
    new = headroom.pair[headroom.pair['source'] == 'eps-headroom'].iloc[0]
    print(f't1 pair at lr {sci(PAIR_LR)}: eps {sci(d2l.EPS_BASE)} → val '
          f'{base.at[0, "val_loss"]:.3f} (seed 1: '
          f'{base.at[1, "val_loss"]:.3f}), update RMS ÷ lr '
          f'{base.at[0, "update_rms"]:.3f}')
    print(f'{"":19} eps {sci(new["eps"])} → val {new["val_loss"]:.3f}, '
          f'update RMS ÷ lr {new["update_rms"]:.3f}')
```

<!-- Placeholder (prose): claim 25 as measured: whether sqrt(v) sits far above 1e-8 in every group of both models (medians, and how small the shares below 1e-5 and 1e-7 are; the embedding has the most small coordinates, rows of rare tokens), how much eps at the median shrank the update RMS (the ratio of the two lines), and the change in the val loss against the two seeds' difference at eps 1e-8 ("eps at the median changed the val loss by x, within the seed spread" is a result). Then the scale argument: the r2 medians sit below t1's, so the margin to eps shrinks with model size and training; never present the t1 pair as a measurement at scale. -->

### Who Moves First

<!-- Placeholder (prose): a change point per series: each series goes on one grid of 10-step blocks (the block median; the largest logit is probed every 10 steps, so every series meets the same detector, and a dense series is not judged by a stricter test than a sparse one), a line is fitted to the 20 blocks of the 200 steps before the event, and the series has moved at the first 20-step mean of the residual, within 200 steps, that leaves 3 standard errors of that line's prediction (the residual scatter plus the line's own extrapolation error; on white noise this fires in about 6-9% of windows, dense or sparse). Detrended, so that a curve that keeps falling at the same rate does not count as moving; and the line needs a stationary reference: an event whose 200 steps before fall inside warmup's ramp (the end of warmup itself; the reference run's gradient event at step 223) is not read, because the line extrapolates a convex, flattening curve and fires on its own misfit. The last row is the null: how often each series moves at 20 evenly spaced stable-phase steps (every reference window from step 1,000, horizons nearly disjoint) where nothing happens. Resolution: 10 steps (50 in the t1 stand-in, whose largest logit is probed every 50). -->

```{.python .input #reference-model-lead-lag}
%%tab pytorch
#@save
def moved(x, at, before=200, window=20, horizon=200, k=3, grid=10):
    """Steps after `at` until x moves (None within `horizon`): on blocks of
    `grid` steps (medians), the mean residual over `window` steps from the
    line fitted to the `before` steps before `at` leaves k standard errors
    of that line's prediction (residual scatter plus extrapolation)."""
    x = x.dropna()
    b = x.groupby((x.index - at) // grid).median()  # block j: at + grid * j
    ref, after = b.loc[-(before // grid):-1], b.loc[0:horizon // grid - 1]
    j, i = ref.index.to_numpy(float), after.index.to_numpy(float)
    line = np.polyfit(j, ref.to_numpy(), 1)
    s2 = ((ref - np.polyval(line, j)) ** 2).sum() / (len(j) - 2)
    w = max(1, window // grid)
    resid = (after - np.polyval(line, i)).rolling(w).mean()
    mid = pd.Series(i, after.index).rolling(w).mean() - j.mean()
    se = np.sqrt(s2 * (1 / w + 1 / len(j)
                       + mid ** 2 / ((j - j.mean()) ** 2).sum()))
    out = resid[resid.abs() > k * se]
    return int(grid * (out.index[0] + 1)) if len(out) else None
```

```{.python .input #reference-model-lead-lag-table}
%%tab pytorch
run = main.set_index('step')
series = {'grad norm': run['grad_norm'],
          'largest logit': run.filter(like='max_logit_').max(axis=1),
          'loss': run['loss']}
grid = max(int(np.median(np.diff(x.dropna().index)))  # the coarsest
           for x in series.values())                  # sampling, for all
decay_from = run_cfg.steps - run_cfg.decay
events = {'end of warmup': run_cfg.warmup, 'start of decay': decay_from,
          'gradient event' + (' (mild)' if event.mild else ''): event.step}
null = np.linspace(STABLE_FROM + 200, decay_from - 200, 20).round()
print(f'steps until each series moves ({run_cfg.preset} run)')
print(f'{"event":>23} ' + ' '.join(f'{k:>13}' for k in series))
for name, at in events.items():
    if at - 200 < STABLE_FROM:  # the line would extrapolate warmup's ramp
        print(f'{name:>23} {"(reference in warmup: not read)":>41}')
        continue
    lags = ['—' if m is None else f'+{m}'
            for m in (moved(x, at, grid=grid) for x in series.values())]
    print(f'{name:>23} ' + ' '.join(f'{m:>13}' for m in lags))
cells = []
for x in series.values():  # how often, and how late, each fires at random
    lags = [m for m in (moved(x, int(t), grid=grid) for t in null)
            if m is not None]
    cells.append(f'{len(lags)}/{len(null)}'
                 + (f', +{np.median(lags):.0f}' if lags else ''))
print(f'{"20 stable steps (null)":>23} '
      + ' '.join(f'{c:>13}' for c in cells))
```

<!-- Placeholder (prose): claim 6 as measured (F12 review, from the committed log): no monitor resolves a lead over the loss at this recipe. Only the start of the decay has a stable reference, and there no series moves within 200 steps (the learning rate falls gradually); the end of warmup and the reference run's gradient event (step 223, mild, inside warmup) are not read, and the section's gradient-event story is the ten-times arm's. The null row checks the detector: each series fires at 1 of 20 stable-phase steps (grad norm +60, largest logit +60, loss +80), about the rate white noise gives, so no series is more specific than another. Never write that the loss "moved first" at warmup's end or at the event, or that a series "never moves". -->

## Reading Checkpoints and the Ladder

### Checkpoints Scored

<!-- Placeholder (prose): eleven checkpoints scored on validation loss and bits per byte and on five suite tasks at their full item sets (reused: EvalSuite and paired_bootstrap of sec_lm_eval_suites, bits_per_byte of sec_lm_scoring); near-chance tasks (ARC-Challenge, MMLU) carry no information at 138M and are named, not tabled. -->

```{.python .input #reference-model-checkpoint-evals}
%%tab pytorch
STEPS = (500, 1000, 2000, 3000, 4000, 4500, 4800, 5100, 5400, 5700, 6000)
SUITE = ('hellaswag', 'arc_easy', 'piqa', 'sciq', 'winogrande')
ev = d2l.reference_evals(STEPS, tasks=SUITE)
d2l.plot(ev.table['bpb'], ev.table[list(SUITE)].mean(axis=1),
         'validation bits per byte', 'suite mean accuracy', fmts=('o',))
print(f'{"step":>6} {"val loss":>8} {"BPB":>6} '
      + ' '.join(f'{t:>10}' for t in SUITE))
for _, r in ev.table.iterrows():
    print(f'{r["step"]:>6.0f} {r["val_loss"]:8.3f} {r["bpb"]:6.3f} '
          + ' '.join(f'{r[t]:10.3f}' for t in SUITE))
print(f'{"chance":>6} {"":>8} {"":>6} '
      + ' '.join(f'{ev.chance[t]:10.3f}' for t in SUITE))
```

### Adjacent Checkpoints

<!-- Placeholder (prose): a paired bootstrap over items for three pairs of checkpoints; a difference whose interval holds zero is "not resolved". -->

```{.python .input #reference-model-adjacent}
%%tab pytorch
items = ev.items.sort_values(['task', 'item_id'])
n_items = items[items['step'] == STEPS[-1]].groupby('task').size()

def suite_scores(step):
    """Per-item scores weighted so that their mean is the suite mean."""
    it = items[items['step'] == step]
    w = len(it) / (len(SUITE) * it['task'].map(n_items))
    return (it['score'] * w).to_numpy()

print(f'{"pair":>12} {"Δ accuracy":>10} {"95% CI":>16} {"resolved?":>13}')
for a, b in ((5100, 6000), (5700, 6000), (4000, 5100)):
    r = d2l.paired_bootstrap(suite_scores(b), suite_scores(a))
    lo, hi = r.ci
    resolved = 'yes' if lo > 0 or hi < 0 else 'not resolved'
    print(f'{f"{a} → {b}":>12} {100 * r.mean:+9.1f}p {100 * lo:+7.1f} to '
          f'{100 * hi:+.1f} {resolved:>13}')
```

### Per-Domain Bits per Byte

<!-- Placeholder (prose): per-domain bits per byte against step (FineWeb-Edu file 012 by URL-host class, plus a held-out book): which domains the decay helps. -->

```{.python .input #reference-model-domains}
%%tab pytorch
domains = ev.domains.pivot(index='step', columns='domain', values='bpb')
d2l.plot(domains.index.to_numpy(), domains.to_numpy().T, 'step',
         'bits per byte', legend=list(domains), figsize=(5, 3))
```

### The Branch Point, Averaging and Selection Bias

<!-- Placeholder (prose): the branch point (the step-5,100 state from which a decay can be rerun); averaging four stable-phase checkpoints against the decay; and the bias of choosing the best of eleven checkpoints on the evaluation set itself. -->

```{.python .input #reference-model-averaging}
%%tab pytorch
AVERAGED, VAL_TOKENS = (4000, 4500, 4800, 5100), 2 ** 20
models = d2l.load_reference_models(AVERAGED + (6000,), dtype=torch.float32)
states = [models[t].state_dict() for t in AVERAGED]
average = models[AVERAGED[0]]  # its weights become the average
average.load_state_dict({k: sum(st[k] for st in states) / len(states)
                         for k in states[0]})
loss = {t: d2l.val_loss(m, run_cfg, VAL_TOKENS) for t, m in (
    ('average', average), (5100, models[5100]), (6000, models[6000]))}
print(f'average of four stable checkpoints: {loss["average"]:.3f} '
      f'(s5100: {loss[5100]:.3f}; s6000 after decay: {loss[6000]:.3f})')
```

<!-- Placeholder (prose): selection bias: choose the best of the eleven checkpoints on one half of the items, re-score it on the other half, 200 random halvings. -->

```{.python .input #reference-model-selection-bias}
%%tab pytorch
scores = items.pivot_table(index=['task', 'item_id'], columns='step',
                           values='score')
task = scores.index.get_level_values('task')
rng, gains, last = np.random.default_rng(0), [], STEPS[-1]
for _ in range(200):  # choose on one half of the items, re-score the other
    half = rng.random(len(scores)) < 0.5
    pick = scores[half].groupby(task[half]).mean().mean()
    other = scores[~half].groupby(task[~half]).mean().mean()
    best = pick.idxmax()
    gains.append((pick[best] - pick[last], other[best] - other[last]))
chosen, held = 100 * np.array(gains).T
print(f'best-of-11 on half the items: {chosen.mean():+.1f} points; on the '
      f'other half: {held.mean():+.1f} (bias {(chosen - held).mean():.1f} '
      f'± {(chosen - held).std():.1f} points)')
```

### The Ladder

<!-- Placeholder (prose): three smaller sizes on the same depth-width recipe, each at 20N with decays branched from its plateau to finish 10N and 5N (restate subsec_scheduler-branching), scored at their finished and plateau points; 21.6 fits them. -->

```{.python .input #reference-model-ladder}
%%tab pytorch
LADDER = ('ladder-25m', 'ladder-54m', 'ladder-88m')
ladder = d2l.launch_pretrain([r2_config(p) for p in LADDER], n_gpus=4)
pts = d2l.ladder_evals(ladder, log, tasks=SUITE)
d2l.plot_ladder_runs(ladder, log)
```

<!-- Placeholder (prose): the finished points as a table: validation loss at about 5, 10 and 20 tokens per parameter (r2's points are 5.7N, 11.4N and 22.7N). -->

```{.python .input #reference-model-ladder-table}
%%tab pytorch
done = pts.points[pts.points['finished'].astype(bool)]
print(f'{"size":>11} {"P":>11}   val loss (tokens per parameter)')
for size, g in done.sort_values(['P', 'tokens']).groupby('size', sort=False):
    print(f'{size:>11} {int(g["P"].iloc[0]):>11,} ' + ' '.join(
        f'{v:6.3f} ({n:4.1f})' for v, n in zip(g['val_loss'],
                                               g['tokens_per_param'])))
```

## Instabilities

### The Logit Bound and the Conservation Law

<!-- Placeholder (prose): the derivation. |s_ij| <= g^2 d sigma_max(W_Q) sigma_max(W_K) / sqrt(d_h); gradient flow conserves W_Q^T W_Q - W_K^T W_K, and dM/dt = -G W_K W_K^T - W_Q W_Q^T G is a positive feedback through the Gram matrices, so the largest logit grows for thousands of steps before the loss shows it. QK-norm gives |s| <= sqrt(d_h) ||g_q||_inf ||g_k||_inf. The smallest case below: width 256, ||W_Q||_2 ||W_K||_2 = 4 x 5 = 20, sqrt(d_h) = 8, bound 640. -->

```{.python .input #reference-model-logit-bound}
%%tab pytorch
d_x, d_k, n = 256, 64, 10 ** 5  # width, head dimension, samples
torch.manual_seed(0)

def with_norm(W, s):  # W rescaled to spectral norm s
    return W * s / torch.linalg.matrix_norm(W, ord=2)

def unit_rms(v):
    return v / v.pow(2).mean(-1, keepdim=True).sqrt()

W_Q, W_K = with_norm(torch.randn(d_k, d_x), 4), with_norm(
    torch.randn(d_k, d_x), 5)
x_i, x_j = unit_rms(torch.randn(2, n, d_x))  # RMSNorm outputs, gain 1
score = ((x_i @ W_Q.T) * (x_j @ W_K.T)).sum(-1) / math.sqrt(d_k)
bound = d_x * 4 * 5 / math.sqrt(d_k)
g_q, g_k = 1 + torch.rand(d_k), 1 + torch.rand(d_k)  # QK-norm gains
q, k = g_q * unit_rms(1e3 * x_i @ W_Q.T), g_k * unit_rms(1e3 * x_j @ W_K.T)
score_n = (q * k).sum(-1) / math.sqrt(d_k)
bound_n = math.sqrt(d_k) * g_q.max() * g_k.max()
assert score.abs().max() <= bound and score_n.abs().max() <= bound_n
print(f'largest |s| over 1e5 samples: {score.abs().max():.1f} (bound '
      f'{bound:.0f})')
print(f'with QK-norm, inputs scaled by 1e3: largest |s| '
      f'{score_n.abs().max():.1f} (bound √d_h‖g_q‖∞‖g_k‖∞ = {bound_n:.1f})')
```

<!-- Placeholder (prose): the conservation law on a small bilinear model trained by gradient descent: the difference of the Gram matrices stays put while each Gram matrix moves. -->

```{.python .input #reference-model-conservation}
%%tab pytorch
m, w, r, eta = 64, 16, 4, 1e-4  # samples, width, rank, step size
X = torch.randn(m, w, dtype=torch.float64) / math.sqrt(w)
target = torch.randn(m, m, dtype=torch.float64)
A, B = 0.3 * torch.randn(2, w, r, dtype=torch.float64)  # W_Q, W_K
C0, G0 = A.T @ A - B.T @ B, A.T @ A
for _ in range(20_000):  # gradient descent on 1/2 ||X A B^T X^T - target||^2
    grad = X.T @ (X @ A @ B.T @ X.T - target) @ X  # dL/dM, M = A B^T
    A, B = A - eta * grad @ B, B - eta * grad.T @ A
drift = torch.linalg.norm(A.T @ A - B.T @ B - C0) / torch.linalg.norm(C0)
moved_q = torch.linalg.norm(A.T @ A - G0) / torch.linalg.norm(G0)
assert drift < 1e-2
print(f'relative drift of W_Qᵀ W_Q − W_Kᵀ W_K: {drift:.1e} (W_Qᵀ W_Q '
      f'itself moved {moved_q:.0%})')
```

### Monitors and Cures

<!-- Placeholder (prose): the cures, each with its one difference: QK-norm bounds the logit by construction, QK-clip rescales the product after the fact (restate sec_muon), softcapping bounds it by a tanh and stops the gradient that would shrink it, z-loss pulls log Z to zero (restate subsec_practice-nan), rewind-and-skip removes a bad batch. The monitor reads explicit scores (the fused kernel returns none) on a one-sequence probe per rank; Kimi K2 uses the batch maximum. -->

```{.python .input #reference-model-instability}
%%tab pytorch
#@save
LogitStats = collections.namedtuple('LogitStats', ['max_logit', 'entropy'])

@torch.no_grad()  #@save
def attn_logit_stats(model, batch, max_seqs=1, chunk=512):
    """Largest attention logit (before a softcap) and mean attention entropy
    (of the softmax the block computes, after its softcap) of every head,
    per block, on the first max_seqs sequences of batch, from explicit
    scores (fused kernels return none); under DDP the maximum is over all
    ranks."""
    X, ctx = d2l._probe_inputs(batch, max_seqs, model)
    stats = {}
    for l, (attn, Y, tables) in enumerate(d2l._attention_inputs(model, X,
                                                                ctx)):
        q, k, _ = attn.qkv(Y, tables=tables)
        k = k.repeat_interleave(attn.num_heads // attn.num_kv_heads, 1)
        m, ent = d2l._chunked_logit_stats(q, k, ctx, chunk,
                                          getattr(attn, 'softcap', None))
        stats[l] = LogitStats(d2l._all_reduce_max(m), ent)
    return stats

#@save
def softcap(x, c):
    """Smoothly cap x to (-c, c)."""
    return c * torch.tanh(x / c)

#@save
def make_bad_batch(batch, kind='random', vocab=32768, seed=0, k=None,
                   period=None, pool=None):
    """A batch shaped like batch of one-document windows, targets shifted
    ids: uniform random ids >= 32 ('random'), one repeated id ('repeat'),
    or ('ngram') k of batch's rows (all if k is None; the others untouched)
    replaced by one n-gram of period of the ids >= 32 of pool (default:
    batch's own), repeated."""
    B, T = batch['input_ids'].shape
    g = torch.Generator().manual_seed(seed)
    pos = torch.arange(T, dtype=torch.int32).expand(B, T).contiguous()
    if kind == 'ngram':  # OLMo 2's spike shape
        pool = (batch['input_ids'] if pool is None else pool).flatten().cpu()
        pool = pool[pool >= 32]
        start = int(torch.randint(len(pool) - period + 1, (), generator=g))
        ids = pool[start:start + period].repeat(T // period + 1)[:T + 1]
        rows = torch.randperm(B, generator=g)[:k]
        bad = {key: v.clone() for key, v in batch.items()}
        for key, v in (('input_ids', ids[:-1]), ('labels', ids[1:]),
                       ('doc_ids', 0), ('pos_ids', pos[0])):
            bad[key][rows] = torch.as_tensor(v).to(bad[key])
        return bad
    if kind not in ('random', 'repeat'):
        raise ValueError(f'unknown bad-batch kind {kind!r}')
    ids = (torch.randint(32, vocab, (B, T + 1), generator=g) if kind ==
           'random' else torch.full((B, T + 1), 32 + seed % (vocab - 32)))
    return dict(input_ids=ids[:, :-1].contiguous(), labels=ids[:, 1:].clone(),
                doc_ids=torch.zeros_like(pos), pos_ids=pos)

#@save
def lr_sensitivity(final_losses, l0):
    """Wortsman et al.'s sensitivity: the mean over learning rates of
    min(loss, l0) - min(losses); a non-finite loss counts as l0."""
    x = torch.tensor([float(v) for v in final_losses], dtype=torch.float64)
    x = torch.where(torch.isfinite(x), x, l0).clamp(max=l0)
    return float((x - x.min()).mean())
```

```{.python .input #reference-model-softcap-check}
%%tab pytorch
logit = torch.linspace(-200, 200, 4001, requires_grad=True)
capped = softcap(logit, 50)
capped.sum().backward()
at_200 = logit.grad[(logit.detach() - 200).abs().argmin()]
assert capped.abs().max() < 50
print(f'softcap(s, 50): max |s| = {capped.abs().max():.2f} over s ∈ [−200, '
      f'200]; gradient at s = 200: {at_200:.1e}')
```

<!-- Placeholder (prose): QK-clip after an optimizer step, per head (Kimi K2), a local copy until sec_muon exports it; why it does nothing under QK-norm (the normalized query and key absorb any rescaling). -->

```{.python .input #reference-model-qk-clip}
%%tab pytorch
#@save
def qk_clip_(model, stats, tau, alpha=0.5):
    """Rescale in place the query and key rows of every head whose largest
    logit S exceeds tau: W_q by (tau/S)^alpha, W_k by (tau/S)^(1-alpha); a
    key head shared by a group of query heads uses the group's maximum.
    Heads at or below tau (a largest logit can be negative) keep factor 1.
    Returns each block's per-head logit factors; {} under QK-norm."""
    import warnings
    raw = d2l._unwrap(model)
    if raw.blks[0].attention.qk_norm:
        warnings.warn('qk_clip_ does nothing under QK-norm')
        return {}
    factors = {}
    with torch.no_grad():
        for l, s in stats.items():
            attn = raw.blks[l].attention
            S = s.max_logit.to(attn.W_q.weight.device).float()
            S_kv = S.view(attn.num_kv_heads, -1).amax(1)  # per key head
            f_q = torch.where(S > tau, tau / S, 1.0) ** alpha
            f_k = torch.where(S_kv > tau, tau / S_kv, 1.0) ** (1 - alpha)
            d_h = attn.head_dim
            attn.W_q.weight.mul_(f_q.repeat_interleave(d_h)[:, None])
            attn.W_k.weight.mul_(f_k.repeat_interleave(d_h)[:, None])
            factors[l] = f_q * f_k.repeat_interleave(S.numel() //
                                                     S_kv.numel())
    return factors
```

<!-- Placeholder (prose): the contract on a one-block toy (with two blocks, clipping the first changes the inputs of the second): clipped heads return to tau, a second clip only rounds, heads below tau are untouched, QK-norm makes it a no-op. -->

```{.python .input #reference-model-qk-clip-check}
%%tab pytorch
torch.manual_seed(0)
one_block = d2l.PackedGPT.from_arch(toy._replace(L=1), backend='sdpa').eval()
attn = one_block.blks[0].attention
with torch.no_grad():  # large logits in query heads 0-1 and key head 0
    attn.W_q.weight[:2 * toy.d_h] *= 6
    attn.W_k.weight[:toy.d_h] *= 3
batch = d2l.PackedStream(train, toy.T, 4).batch(0)
S = attn_logit_stats(one_block, batch)[0].max_logit
tau = 0.6 * float(S.max())
W_q0 = attn.W_q.weight.detach().clone()
qk_clip_(one_block, attn_logit_stats(one_block, batch), tau)
S1 = attn_logit_stats(one_block, batch)[0].max_logit
hit = S > tau
owns = hit & (S == S.view(toy.H_kv, -1).amax(1).repeat_interleave(
    toy.H // toy.H_kv))  # the head that sets its key head's maximum
err = float(((S1[owns] - tau).abs() / tau).max())
W_q1 = attn.W_q.weight.detach().clone()
below = torch.equal(W_q1.view(toy.H, toy.d_h, -1)[~hit],
                    W_q0.view(toy.H, toy.d_h, -1)[~hit])
assert err < 1e-5 and below
print(f'after clip: max logit = τ to {err:.1e} relative')
print(f'heads below τ: {"bit-identical" if below else "changed"}')
```

```{.python .input #reference-model-qk-clip-again}
%%tab pytorch
qk_clip_(one_block, attn_logit_stats(one_block, batch), tau)  # again
rounding = float(((attn.W_q.weight.detach() - W_q1).abs() / W_q1.abs()).max())
torch.manual_seed(0)
normed = d2l.PackedGPT.from_arch(toy._replace(L=1, qk_norm=True))
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    factors = qk_clip_(normed, attn_logit_stats(normed, batch), tau)
assert factors == {} and caught
print(f'second clip: weights move by at most {rounding:.0e} (rounding)')
print(f'under QK-norm: no layer clipped, {len(caught)} warning')
```

### The Stable Learning-Rate Range

<!-- Placeholder (prose): the T1 grid on the full t1 schedule (1,526 steps, warmup 76, constant to step 1,297, linear decay over the last 229; only the peak rate changes, six rates, two seeds), clipping off because it hides the growth that precedes a spike, z-loss and weight decay as in t1; tau fixed beforehand by the rule below. Six arms: the baseline, QK-norm, QK-clip, softcap (c = 50, through the math backend), and two baselines with the production recipe's fixes, gradient clipping at 1.0 and a longer warmup (20% of the run, 305 steps, about four times the presets' 76). The two seeds differ in initialization and data order; every variant at a seed sees the same batches, so the variants are paired. Statistics every 10 steps: the largest attention logit and the mean attention entropy per layer; the final validation loss on 2^20 tokens; the zero-layer reference (t1 with no blocks: embedding, final norm, tied head) at 1e-3 is Wortsman et al.'s line a model with attention must beat. Wortsman et al.'s divergence is a final loss far above the best (counted at ln V in the sensitivity), not a NaN: the collapse is silent. A rate has collapsed when some layer's mean entropy falls below 0.1 nats (pre-registered). -->

```{.python .input #reference-model-stable-lr}
%%tab pytorch
#@save
def tau_rule(log, lo, hi):
    """QK-clip's threshold, fixed before the run that uses it: ten times the
    ceiling of a tenth of the largest logit in log over steps lo to hi."""
    rows = log[log['step'].between(lo, hi)]
    return 10 * math.ceil(rows.filter(like='max_logit_').max().max() / 10)

#@save
def first_collapsed_rate(arms, variant, seed=None, floor=0.1):
    """The smallest learning rate at which an arm of variant (of seed, if
    given) had a layer whose mean attention entropy fell below floor nats
    at some statistics step; None if no rate collapsed."""
    hit = arms[(arms['variant'] == variant) & (arms['min_entropy'] < floor)]
    if seed is not None:
        hit = hit[hit['seed'] == seed]
    return float(hit['lr'].min()) if len(hit) else None

#@save
def departure(x, start, before=200, window=20, k=3):
    """The first step from start + before on at which the rolling median of
    x over window steps leaves k MAD around the median of x over the before
    steps from start (for a paired series, so no trend line); None if it
    never does."""
    x = x[x.index >= start].dropna()
    ref = x[x.index < start + before]
    center = ref.median()
    mad = (ref - center).abs().median()
    roll = x.rolling(window).median()
    out = roll[(roll.index >= start + before)
               & ((roll - center).abs() > k * mad)]
    return int(out.index[0]) if len(out) else None
```

```{.python .input #reference-model-stable-lr-arms}
%%tab pytorch
LRS = (1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2, 3.2e-2)
VARIANTS = ('none', 'qk_norm', 'qk_clip', 'softcap', 'clip', 'warmup')
TAU_T1 = tau_rule(t1[t1['seed'] == 0], t1_cfg.warmup, 400)
inst = d2l.instability_arms(r2_config('t1', clip=None), lrs=LRS,
                            variants=VARIANTS, seeds=(0, 1), tau=TAU_T1)
arms = inst.table.assign(  # diverged: ln V, as the sensitivity counts it
    loss=inst.table['val_loss'].fillna(math.inf).clip(upper=math.log(V)))
mean = arms.groupby(['variant', 'lr'])['loss'].mean()  # of the two seeds
top = LRS[-1]
print(f'{"variant":>8} {"best lr":>7} {"sensitivity":>11} '
      f'{f"{top:g} − best":>12} {"largest logit":>13} '
      f'{f"> 1,000 at {top:g}":>21}')  # first step, at the top rate
for v in VARIANTS:
    m, g = mean[v], arms[arms['variant'] == v]
    at_top = g.loc[g['lr'] == top, 'first_logit_1000']  # one per seed
    hit = at_top.dropna()
    alarm = (f'step {int(hit.min()):,} ({len(hit)} of {len(at_top)})'
             if len(hit) else '—')  # the earliest, and how many seeds
    print(f'{v:>8} {m.idxmin():7g} {lr_sensitivity(m, math.log(V)):11.2f} '
          f'{m[top] - m.min():12.2f} {g["max_logit"].max():13,.0f} '
          f'{alarm:>21}')
print(f'zero-layer reference: {inst.zero_layer:.2f}')
print(f'τ (t1) = {inst.tau}')
```

<!-- Placeholder (prose): read the table: the best rate and the sensitivity per arm; what QK-norm, QK-clip and softcap change, and whether clipping and the longer warmup change the baseline's curve (claims 15, 15b, 15c; the prediction is little), as measured. Softcap's largest logit is the value before the cap: the softmax sees at most 50 while the raw product grows (Kimi K2: it "can still grow excessively before capping"); its entropy is that of the softmax the model computes, over the capped scores. QK-clip's largest logit can sit far above tau: it is read after the optimizer step and before that step's clip, and both the clip and the statistic see one sequence of the batch. QK-norm's largest logit is the measured maximum (its bound sqrt(d_h)·max|g_q|·max|g_k| holds by construction, claim 20). The last column gives the earliest seed's step and how many of the two seeds crossed. -->

```{.python .input #reference-model-stable-lr-edge}
%%tab pytorch
by_seed = arms.pivot_table('loss', 'lr', ['variant', 'seed'])
gap = by_seed['none'] - by_seed['qk_norm']  # paired: same seed, batches
spread = (by_seed['none'][0] - by_seed['none'][1]).abs()
noise = 3 * max(spread[LRS[0]],
                abs(by_seed['qk_norm'][0] - by_seed['qk_norm'][1])[LRS[0]])
clip_gap = mean['clip'] - mean['none']
dead = arms[arms['min_entropy'] < 0.1]
print(f'Δ = baseline − QK-norm (paired); 3 × seed difference at '
      f'{LRS[0]:g}: {noise:.3f}')
print(f'{"lr":>7} {"Δ seed 0":>9} {"Δ seed 1":>9} {"resolved":>9} '
      f'{"clip − none":>11} {"seed diff":>9} {"collapsed":>9}')
for lr in LRS:
    hit = ((dead['variant'] == 'none') & (dead['lr'] == lr)).any()
    print(f'{lr:7g} {gap.loc[lr, 0]:+9.3f} {gap.loc[lr, 1]:+9.3f} '
          f'{"yes" if (gap.loc[lr] > noise).all() else "no":>9} '
          f'{clip_gap[lr]:+11.3f} {spread[lr]:9.3f} '
          f'{"yes" if hit else "no":>9}')
base, low = arms[arms['variant'] == 'none'], dead[dead['variant'] == 'none']
print(f'collapsed baseline cells: {len(low)} of {len(base)}; finite: '
      f'{int(low["first_nonfinite"].isna().sum())}; above the zero-layer '
      f'loss: {int((low["loss"] > inst.zero_layer).sum())}')
print(f'{"":>21}' + ''.join(f'{v:>9}' for v in VARIANTS))
rates = [first_collapsed_rate(arms, v) for v in VARIANTS]
print(f'{"first collapsed rate":>21}'
      + ''.join(f'{"—" if r is None else f"{r:g}":>9}' for r in rates))
print(f'{"smallest entropy":>21}' + ''.join(
    f'{arms.loc[arms["variant"] == v, "min_entropy"].min():9.3f}'
    for v in VARIANTS))
```

<!-- Placeholder (prose): the three panels on one log axis of the rate, both seeds as points and a line through their mean: the final loss with the zero-layer line; the largest logit with tau marked; the smallest per-layer entropy at the end of the peak phase (step 1,290). Panels 2 and 3 are the mechanism: the loss departs where the entropy reaches zero, for the softcap arm as well, whose capped logit never passes 50. Then the collapsed cells: how many end with a finite loss and where they sit against the zero-layer model (claim 15a), and whether the decay brings them back (claim 15d), as measured. -->

```{.python .input #reference-model-stable-lr-plot}
%%tab pytorch
end = t1_cfg.steps - t1_cfg.decay  # the last step at the peak rate
end -= end % d2l.INST_STATS_EVERY  # its last statistics step
d2l.plot_lr_grid(arms, {'loss': ('final val loss', inst.zero_layer),
                        'max_logit': ('largest logit', TAU_T1),
                        'peak_entropy': (f'entropy at step {end:,} (nats)',
                                         None)},
                 VARIANTS, logy=['max_logit'])
```

### Who Moves First at T1

<!-- Placeholder (prose): the baseline (seed 0) at the first collapsed rate against its QK-norm twin (same rate, seed and batches): when the largest logit passes 1,000, when the smallest entropy falls below 0.1 nats, when the gradient norm moves, and when the loss first sits 0.1 nats above the twin's (20-step means; a diverged run's first non-finite step counts, since the run stops there). The gradient-norm detector is paired, like the loss: the log ratio of the baseline's norm to the twin's, its 20-step rolling median against ± 3 MAD of the ratio over the first 200 stable steps after warmup (steps 76-275), searched from step 276 to the end of the run; no trend line, since the pairing removes the schedule. (Cell 16's detector cannot be reused here: from the end of warmup its reference line is the warmup ramp, where the norm falls tenfold, so it fires within 30 steps in every run, collapse or not.) If the baseline departs within those first 200 stable steps, the reference band contains the departure and the detector reads late; say so if the printed steps show it. The statistics are sampled every 10 steps, so a lead under 10 steps is no lead (claim 6b). If no rate collapsed, the cell says so and the claim is dropped. -->

```{.python .input #reference-model-who-moves-first-at-t1}
%%tab pytorch
lr_c = first_collapsed_rate(arms, 'none', seed=0)
if lr_c is None:  # pre-registered: the claim is dropped
    print('no baseline rate collapsed: every entropy stayed above 0.1 nats')
else:
    start = t1_cfg.warmup  # the grad-norm band: the 200 steps from here
    steps = d2l.collapse_crossings(inst, lr_c, start, seed=0)
    rule = 'loss 0.1 above QK-norm'
    at = steps[rule]
    print(f'baseline at lr {lr_c:g} (first collapsed rate), seed 0')
    for name, s in steps.items():
        lead = ('' if s is None or at is None or s == at else
                f', {abs(at - s)} steps {"before" if s < at else "after"} '
                'the loss')
        never = (f'never (searched from step {start + 200})'
                 if name == 'grad norm moves' else 'never')
        print(f'{name:>22}: ' + (never if s is None else f'step {s:,}')
              + lead)
    calm = arms[(arms['variant'] == 'none') & (arms['seed'] == 0)
                & (arms['min_entropy'] >= 0.1)]  # the same rule, no collapse
    null = [(lr, d2l.collapse_crossings(inst, lr, start)[rule])
            for lr in sorted(calm['lr'])]
    print(f'{"loss rule, no collapse":>22}: ' + ', '.join(
        f'{"never" if s is None else f"step {s:,}"} at {lr:g}'
        for lr, s in null))
    tr = inst.traces[(inst.traces['seed'] == 0) & (inst.traces['lr'] == lr_c)]
    pair = tr[tr['variant'].isin(['none', 'qk_norm'])].copy()
    loss = pair.pivot(index='step', columns='variant', values='loss')
    gap = (loss['none'] - loss['qk_norm']).rolling(20).mean()  # the rule's
    pair['loss − QK-norm'] = pair['step'].map(gap).where(
        pair['variant'] == 'none')  # one line: the baseline's gap
    d2l.plot_traces(pair, ['max_logit', 'min_entropy', 'grad_norm',
                           'loss − QK-norm'],
                    by='variant', logy=['max_logit', 'grad_norm'],
                    dashed=['qk_norm'], refs={'loss − QK-norm': 0.1},
                    figsize=(5, 5))
```

<!-- Placeholder (prose): claim 6b as measured (F11 review findings 1-3). The loss crossed first, but the printed control line shows the same loss rule firing at rates whose entropy never fell below 0.1 nats (QK-norm trains faster early at every rate from 2e-3 up), so the loss crossing does not warn of collapse; of the four monitors only the entropy and the logit are specific. The last panel is that rule's series, the 20-step mean of baseline − QK-norm with the 0.1 line. The grad norm cannot be read here: every collapsed cell departs before or inside the detector's reference window (steps 76-275), and the detector searches only from step 276. 4e-3 is a borderline first collapsed rate (seed 0 at 0.068, seed 1 at 0.1002); its paired gap is no larger than at the uncollapsed 2e-3, and 8e-3 is where the loss first fails clearly. -->

### The Largest Logit and Model Size

<!-- Placeholder (prose): size transfer (Wortsman et al., Figure 1 and 9): the rate at which logit growth sets in falls as the model grows, so a rate tuned on a small proxy can fail on a larger model; the method is to fit the largest logit at an early step against model size per rate on small shapes and predict a larger one before running it. Here: four shapes at T = 1,024 (t1, the ladder's 10 x 576 and 10 x 768, the reference shape 18 x 768; non-embedding parameters 12.6M, 35.4M, 62.9M, 113.3M), the T1 grid's baseline recipe (clipping off, warmup 76, then a constant rate), 300 steps of 64 x 1,024 tokens, rates 2e-3 to 1.6e-2, two seeds; the statistic is the largest logit at step 300 (Wortsman et al. read theirs at step 2,000 of 100,000, also early). A pilot came first (t1 against 10 x 768 at 4e-3 and 8e-3, both seeds; go only if the larger shape's logit is higher at both rates with both seeds agreeing; pre-registered). The fit is a least-squares line of log logit on log N per rate through the three smaller shapes' six runs; its 95% prediction interval at 113.3M is for the mean of log logit over the two seeds there (t with 4 degrees of freedom); the job stored that prediction before it trained the 113.3M shape, and the cell checks that its own fit equals the stored one. A run whose loss goes non-finite (diverged) gets no fitted line at its rate: leaving it out would drop the largest logits and flatten the slope; in the pilot, a larger shape that diverges where the smaller one does not counts as above it (fixed before the pilot ran). -->

```{.python .input #reference-model-the-largest-logit-and-model-size-1}
%%tab pytorch
#@save
def logit_fit(runs, at, level=0.95):
    """Per rate: the least-squares line of log largest logit on log N
    through runs (columns lr, seed, N, logit), its slope with a level
    interval (slope_hw, the half-width), and its prediction at N = at with
    a level prediction interval for the mean of log logit over as many
    seeds as runs has, back in logit units. A rate with a non-finite logit
    (a diverged run) gets no line: dropping the point would censor the
    largest logits."""
    rows = []
    for lr, g in runs.groupby('lr'):
        x, y = np.log(g['N'].to_numpy(float)), np.log(g['logit'].to_numpy())
        n, x0 = len(x), math.log(at)
        if n < 3 or not np.isfinite(y).all():  # no interval from two points
            continue
        b, a = np.polyfit(x, y, 1)
        s = math.sqrt(((y - a - b * x) ** 2).sum() / (n - 2))  # residual sd
        sxx, t = ((x - x.mean()) ** 2).sum(), d2l.t_quantile(n - 2, level)
        h = t * s * math.sqrt(1 / g['seed'].nunique() + 1 / n
                              + (x0 - x.mean()) ** 2 / sxx)
        y0 = a + b * x0
        rows.append(dict(lr=lr, slope=b, slope_hw=t * s / math.sqrt(sxx),
                         intercept=a, predicted=math.exp(y0),
                         lo=math.exp(y0 - h), hi=math.exp(y0 + h)))
    return pd.DataFrame(rows, columns=['lr', 'slope', 'slope_hw', 'intercept',
                                       'predicted', 'lo', 'hi'])
```

```{.python .input #reference-model-the-largest-logit-and-model-size-2}
%%tab pytorch
SHAPES = ('t1', 'ladder-54m', 'ladder-88m', 'r2')  # the last one held out
xfer = d2l.size_transfer(SHAPES, lrs=(2e-3, 4e-3, 8e-3, 1.6e-2),
                         seeds=(0, 1), steps=300)
runs, held = xfer.table, xfer.table['shape'] == SHAPES[-1]
if xfer.gate == 'no-go':  # the pilot's pre-registered rule
    print(f'pilot: no size trend from {runs["N"].min() / 1e6:.1f}M to '
          f'{runs["N"].max() / 1e6:.1f}M parameters at 300 steps '
          '(gate failed)')
    for lr, g in runs.groupby('lr'):  # a diverged pilot run, if any
        if note := d2l.diverged_note(g):
            print(f'lr {lr:g}: {note}')
elif held.any():  # the full transfer ran
    N = runs.loc[held, 'N'].iloc[0]
    fit = logit_fit(runs[~held], at=N)
    assert np.allclose(fit.to_numpy(float),  # the prediction stored before
                       xfer.fit[fit.columns].to_numpy(float))  # r2 ran
    print(f'{"lr":>6} {"slope ± 95%":>11} '
          f'{f"logit at {N / 1e6:.1f}M, predicted [95%]":>32} '
          f'{"measured":>9}')
    for lr, g in runs.groupby('lr'):
        h = g[g['shape'] == SHAPES[-1]]  # the held-out runs at lr
        if lr not in fit['lr'].values:  # a diverged run at a smaller shape
            print(f'{lr:6g} no fit: {d2l.diverged_note(g.drop(h.index))}')
            continue
        r = fit[fit['lr'] == lr].iloc[0]
        pi = f'{r.predicted:,.0f} [{r.lo:,.0f}–{r.hi:,.0f}]'
        if h['first_nonfinite'].notna().any():
            m = f'diverged (step {h["first_nonfinite"].min():.0f})'
        else:  # the geometric mean of the two seeds
            m = math.exp(np.log(h['logit']).mean())
            m = f'{m:9,.0f} ' + ('inside' if r.lo <= m <= r.hi else 'outside')
        print(f'{lr:6g} {r.slope:5.2f} ± {r.slope_hw:.2f} {pi:>32} {m}')
    d2l.plot_size_transfer(xfer, fit, held=SHAPES[-1])
```

<!-- Placeholder (prose): claim 16b as measured. Go: per rate the slope with its interval (does the largest logit grow with size at every rate, the interval clear of zero?) and whether the 113.3M shape's measured logit (the geometric mean of its two seeds) falls inside the prediction interval ("inside", or "outside (measured x, predicted y)"); four separate 95% intervals, so one "outside" alone is not a failed transfer; "no fit" or "diverged (step k)" rows, if any, are read as divergences, not as misses; the open markers are the held-out runs. No-go: one sentence ("12.6M against 62.9M, a fivefold step, showed no trend at 300 steps", as the cell prints it), Wortsman et al. for the trend over three orders of magnitude, no figure. Cell 26's ten-times arms use another recipe (clip 1.0, T = 4,096, 128 sequences); say so when the two are read together (claim 16). -->

### Ten Times the Learning Rate

<!-- Placeholder (prose): at the reference architecture and recipe (clip 1.0, z-loss), 500 steps at ten times the peak rate, with and without QK-clip at a tau fixed from the reference log's steps 300-500 before the arms ran; the reference run's own first 500 steps are the baseline. Pending T2 (C1d): campaign marker for lm-r2-inst-t2-v1. -->

```{.python .input #reference-model-ten-x}
%%tab pytorch
TAU = tau_rule(log, 300, 500)
t2 = d2l.instability_arms(r2_config('r2'), lrs=(6e-3,),
                          variants=('none', 'qk_clip'), steps=500, tau=TAU)
d2l.plot_traces(t2.traces, ['max_logit', 'loss'], by=['lr', 'variant'],
                logy=['max_logit'])
```

```{.python .input #reference-model-ten-x-table}
%%tab pytorch
names = {'baseline': 'baseline', 'none': 'no cure', 'qk_clip': '+ QK-clip'}
print(f'τ = {t2.tau}')  # TAU, or the T1 grid's τ while the arms are pending
print(f'{"arm":>18} {"final loss":>10} {"largest logit":>13} '
      f'{"first non-finite step":>21}')
for _, r in t2.table.iterrows():
    first = '—' if pd.isna(r['first_nonfinite']) else int(
        r['first_nonfinite'])
    print(f'{f"{r.lr:g} {names[r.variant]}":>18} {r["final_loss"]:10.3f} '
          f'{r["max_logit"]:13.0f} {first:>21}')
```

### One Bad Batch Under Adam

<!-- Placeholder (prose): one bad batch under Adam, paired. After one gradient far larger than its history, Adam's first step has size about lr(1 - beta1)/sqrt(1 - beta2) per coordinate (0.45 lr at beta2 = 0.95, 3.2 lr at 0.999 for a fully warmed second moment; at the experiment's bad step 300 the bias correction sqrt(1 - 0.999^300) = 0.51 makes it about 1.6 lr at 0.999, and about 0.45 lr at 0.95; quote 1.6 lr for the experiment; stated arithmetic, claim 22), and the kick decays like (beta1/sqrt(beta2))^k: second-moment underestimation in its simplest form, the reason for beta2 = 0.95 and PaLM's 1 - k^-0.8. Setup: t1 at 1e-3 with clipping off, one run per beta2 keeps full states (steps 76, the end of warmup, and 250); the late state is the t1 run's own at step 1,297 (it trained with clipping at 1.0; the arms run clipping off from there, at the same constant rate). Every arm replays 150 steps from a state with the bad batch at its step, against the skip replay from the same state, which takes the ordinary batch at that step, so both see identical data before and after (skipping the step in the stream, as PaLM does, would shift every later batch). The bad batch is built from step 300's data at every state (the same n-gram, period 7, in the same rows): `random` replaces all 64 sequences, `n-gram ×k` k of them, the shape OLMo 2 found in its spiking batches. The early and late states see it 4 steps after their state (steps 80 and 1,301), the mid state at 300, 50 steps after its own. Every replay starts at or after the end of warmup (the library refuses an earlier state) and runs at the same constant rate 1e-3, so the three states differ only in the model. Clip twins clip at 1.0 from the state on and pair against a skip replay that clips too. Statistic: the excess loss over the skip replay on the 100 steps after the bad batch, its peak, the steps until it stays below twice the floor (within the 100-step window: the offset of the last step at or above it; >100 = not back by the window's end), its sum; the pre-clip norm at the bad step over the skip replay's median norm. The replay floor (claim 23: the largest loss difference between replays of one state without a bad batch, i.e. a second skip replay over its window and every arm against its skip replay before its bad step; one floor for all rows) sets every comparison on Ada: replays are not bit-exact, and the smokes showed single pairs drifting from 3e-5 to 3e-3, so one pair alone would understate it. Read claims 22, 24 (clip ratio against min(1, 1/norm), within a factor of 2, from the table's peak columns and the stored norm; only for rows whose pre-clip norm exceeds 1 and whose no-clip peak is well above twice the floor, the others "no difference beyond twice the floor"; Adam normalizes the clipped gradient, so the peak may shrink far less than 1/norm: report the measured cut, do not widen the tolerance), 24b (beta2), 24c (k) and 24d (state) as measured. -->

```{.python .input #reference-model-spike}
%%tab pytorch
ARMS = (  # β2, state (step), bad step, batch, sequences of 64, clip twin
    (0.95, 250, 300, 'random', 64, True),
    (0.95, 250, 300, 'ngram', 1, True),
    (0.95, 250, 300, 'ngram', 8, True),
    (0.95, 250, 300, 'ngram', 64, True),
    (0.999, 250, 300, 'random', 64, True),
    (0.999, 250, 300, 'ngram', 8, True),
    (0.95, 76, 80, 'ngram', 8, False),
    (0.95, 1297, 1301, 'ngram', 8, False))
spike = d2l.spike_replay(r2_config('t1', clip=None), ARMS, period=7,
                         source=300)
d2l.plot_spike(spike)
```

```{.python .input #reference-model-spike-table}
%%tab pytorch
def show(v, fmt):  # steps: None (NaN) = not back within the window
    if fmt != 'steps':
        return f'{v:{fmt}}'
    return f'>{d2l.SPIKE_WINDOW}' if pd.isna(v) else f'{v:.0f}'

def pair(r, col, fmt):  # no clip / clip; '—' where there is no clip twin
    twin = not pd.isna(r.peak_clip)
    return (f'{show(getattr(r, col), fmt)} / '
            + (show(getattr(r, col + '_clip'), fmt) if twin else '—'))

if len(spike.table):  # nothing but the label while the result is pending
    print(f'{"β2":>5} {"state":>5} {"batch":>10} {"norm÷med":>8} '
          f'{"peak excess":>15} {"steps < 2×floor":>15} '
          f'{"summed excess":>15}')
    for r in spike.table.itertuples():
        batch = 'random' if r.batch == 'random' else f'n-gram ×{r.k}'
        print(f'{r.beta2:>5g} {r.state:>5} {batch:>10} {r.norm_ratio:8.1f} '
              f'{pair(r, "peak", ".4f"):>15} {pair(r, "recover", "steps"):>15}'
              f' {pair(r, "total", ".2f"):>15}')
    print(f'replay floor (same state, same batches, no bad batch): '
          f'{spike.floor:.1e}')
```

<!-- Placeholder (prose): tab_lm_inst_diagnosis, three causes, each row with the measured example of the table above that matches it or "none produced": bad data (a bad batch that hurts from every state: the rows whose excess exceeds the floor at all three states), unstable state (the same batch hurting from one state and not another: compare the n-gram ×8 rows at steps 76, 250 and 1,297), numerics (a difference the same-state replay shows without any bad batch: the floor itself). -->

### OLMo 2's Repeated N-gram Rule

<!-- Placeholder (prose): OLMo 2's data-side cure: drop documents with 32 or more consecutive repeats of an n-gram of 1 to 13 tokens, the shape of its spiking batches. Applied here to the training windows of the 100M-token t1 set on the CPU (claim 24e: a count, 0 would be a result too); the windows are the training sequences, not OLMo 2's documents. -->

```{.python .input #reference-model-olmo-2-s-repeated-n-gram-rule}
%%tab pytorch
flags = d2l.repeated_ngram_flags(train, T=1024, min_repeats=32, max_n=13)
k, n = len(flags.table), flags.windows
print(f"windows flagged by OLMo 2's rule: {k:,} of {n:,} ({k / n:.3%})")
if k:
    r = flags.table.iloc[0]
    print(f'example: n-gram of length {r["period"]} repeated '
          f'{r["repeats"]} times: {r["text"]}')
```

## Optimizers as a Stated Fact

<!-- Placeholder (prose): no optimizer race here. Against a tuned AdamW, matrix optimizers (Muon, Soap, Kron) gain at most about 1.4x at small scale and about 1.1x at larger scale, and the gain shrinks with scale under standard weight decay (Fantastic Pretraining Optimizers I and II); the production Muon recipe (Muon is Scalable). The Muon variant of the t1 preset is an exercise. -->

## Summary

<!-- Placeholder (prose): 3-6 sentences; hand-off: how to size a model that will be served (sec_lm_budgets). -->

## Exercises

<!-- Placeholder (prose): 6-8 exercises per the spec (derive the bound under QK-clip and the Gram change one clip makes; Vd once vs 6Vd; the WSD integral fraction delta/(2 - delta); update ratios on the t1 run; a t1 learning-rate sweep; four ways to finish the t1 run from its step-1,297 state; the Muon variant against the stated bound). -->
