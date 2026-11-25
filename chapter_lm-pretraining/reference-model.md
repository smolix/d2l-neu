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
import tempfile
import time
import warnings
import pandas as pd
from torch.utils.flop_counter import FlopCounterMode
```

<!-- Placeholder (prose): the log's largest gradient event, read before anything is explained; the section returns to it in "Who Moves First". -->

```{.python .input #reference-model-opening}
%%tab pytorch
MILD, DELTA = 3, 0.3  # a mild event; the stability margin (see below)
log = d2l.load_artifact('reference-run-log')
main = log[log['run'] == 'r2'].set_index('step')
g = main['grad_norm']
ratio = g / g.rolling(100).median().shift()  # against the 100 steps before
event = int(ratio.idxmax())
hours = (main['time'].loc[event] - main['time'].iloc[0]) / 3600
print(f'largest gradient event: step {event} (hour {hours:.1f}), norm '
      f'{ratio[event]:.1f}x its running median'
      + (' (mild)' if ratio[event] < MILD else ''))
label = ''
if ratio[event] < MILD:  # pre-registered: the 10x arm's failure instead
    t2_arms = d2l.load_artifact('instability-t2').traces.set_index('step')
    main = t2_arms[t2_arms['variant'] == 'none']
    gap = main['loss'] - t2_arms.loc[t2_arms['variant'] == 'baseline', 'loss']
    over = gap.index[~(gap <= DELTA)]  # non-finite, or above the margin
    event = int(over[0] if len(over) else gap.idxmax())
    label = '10x learning rate, 500-step arm: '
logits = main.filter(like='max_logit_').dropna(how='all')
near = logits.index[abs(logits.index - event).argmin()]
top = logits.loc[near].astype(float)
layer = int(top.idxmax().rsplit('_', 1)[1]) + 1
dloss = main['loss'].loc[event + 1:event + 200].mean() - main['loss'].loc[
    event - 199:event].mean()
print(f'{label}largest attention logit then: {top.max():.0f} (layer {layer})')
print(f'{label}loss change over the next 200 steps: {dloss:+.3f}')
```

## The Reference Model and What It Costs

### The Configuration

<!-- Placeholder (prose): the reference configuration r2 and its presets: shapes, schedules, why 18 layers and GQA 3:1, the T1 preset readers can rerun, the ladder sizes (restate subsec_scheduler-branching in one sentence for the branches). -->

```{.python .input #reference-model-config}
%%tab pytorch
#@save
Arch = collections.namedtuple('Arch',
                              'L d H H_kv d_h f vocab T rope_base qk_norm tie')

import dataclasses  #@save

@dataclasses.dataclass
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
    z_loss: float = 1e-4
    dtype: str = 'bf16'
    compile: bool = True
    ddp: bool = False
    log_every: int = 1
    probe_every: int = 50       # update RMS per module
    stats_every: int = 50       # largest attention logit per layer
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
    cfg = PretrainConfig(preset, arch, steps, B, micro_batch=16 if T < 4096
                         else 8, peak_lr=lr, warmup=warmup, decay=decay,
                         data=data, **d2l._preset_saves(preset))
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

<!-- Placeholder (prose): the formula against PyTorch's FLOP counter on one forward and backward pass; what the counter counts differently (norms and softmax are not matrix products). -->

```{.python .input #reference-model-flops-check}
%%tab pytorch
T = cfg.arch.T  # shapes only: the count runs on the meta device
model = d2l.PackedGPT.from_arch(cfg.arch).to('meta')
X = torch.randint(V, (1, T), device='meta')
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

<!-- Placeholder (prose): token-weighted accumulation (each micro-batch's summed loss over the step's global count n_valid), z-loss (restate subsec_practice-nan in one sentence), the pre-clip norm logged because clipping hides the growth that precedes a spike. -->

```{.python .input #reference-model-step}
%%tab pytorch
#@save
def pretrain_step(model, optimizer, micro_batches, cfg, n_valid):
    """One step on this rank's micro-batches; n_valid is the step's global
    count of valid targets, so the gradient is the token-weighted mean over
    every rank and micro-batch. Returns loss, pre-clip norm and clipped."""
    world, nll = d2l._rank_world()[1], 0.0
    for k, mb in enumerate(micro_batches):
        ctx = d2l.PackCtx(mb['doc_ids'], mb['pos_ids'])
        with d2l._no_sync(model, k < len(micro_batches) - 1):  # sync last
            with torch.autocast(mb['input_ids'].device.type, torch.bfloat16,
                                enabled=cfg.dtype == 'bf16'):
                H, W = model(mb['input_ids'], ctx, return_hidden=True)
                loss, _, part = d2l.chunked_lm_loss_and_nll(
                    H, W, mb['labels'], z_loss=cfg.z_loss)
            (loss * world / n_valid).backward()  # DDP averages the ranks
        nll = nll + part.detach()
    params = [p for g in optimizer.param_groups for p in g['params']]
    norm = torch.nn.utils.clip_grad_norm_(params, cfg.clip or math.inf)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return dict(loss=d2l._all_reduce_sum(nll) / n_valid, grad_norm=norm,
                clipped=norm > (cfg.clip or math.inf))
```

### The Loop

<!-- Placeholder (prose): what the loop logs and when (every step: loss, pre-clip norm, clipped, tokens/s, MFU; every probe_every steps: update RMS per module; every stats_every steps: largest attention logit per layer from an explicit-score probe, since the fused kernel returns no scores; every val_every steps: validation), checkpoints, full states, branches; the DDP and state plumbing lives in the library. -->

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
            out = pretrain_step(run.model, run.optimizer, micro, cfg, n_valid)
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
d2l._show_label(t1)
print(f'{"seed":>4} {"val loss":>8} {"val bits per byte":>17} '
      f'{"tokens/s":>8} {"MFU":>5}')
finals = []
for seed, g in t1[t1['run'] == 't1'].groupby('seed'):
    stable = g[g['step'].between(t1_cfg.warmup + 1,
                                 t1_cfg.steps - t1_cfg.decay)]
    finals.append(g['val_loss'].dropna().iloc[-1])
    print(f'{seed:>4} {finals[-1]:8.3f} {d2l.val_bpb(finals[-1], t1_cfg):17.3f}'
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
d2l._show_label(log)
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

<!-- Placeholder (prose): Adam's update in units of the learning rate has RMS sqrt((1 - beta_1)/(1 + beta_1)) = 0.229 when the gradient is noise; near 1 when it is signal. Stable phase against decay phase, per module type. -->

```{.python .input #reference-model-update-rms}
%%tab pytorch
beta1 = run_cfg.betas[0]
noise_floor = math.sqrt((1 - beta1) / (1 + beta1))
rms = main.set_index('step').filter(like='rms_').dropna(how='all')
decay_from = run_cfg.steps - run_cfg.decay
phases = {'stable': rms.loc[run_cfg.steps // 6:decay_from],  # after warmup
          'decay': rms.loc[decay_from + 1:]}
med = {k: v.median() for k, v in phases.items()}
print(f'{"module":>9} {"stable":>7} {"decay":>7} '
      f'{f"ratio to {noise_floor:.3f}":>15}')
for col in rms:
    print(f'{col[4:]:>9} {med["stable"][col]:7.3f} {med["decay"][col]:7.3f} '
          f'{med["stable"][col] / noise_floor:15.2f}')
```

### Who Moves First

<!-- Placeholder (prose): a change point per series: the first step after an event where the 50-step rolling median leaves the median ± 3 MAD of the 200 steps before it. Read at the end of warmup, at the start of the decay and at the opening's gradient event. -->

```{.python .input #reference-model-lead-lag}
%%tab pytorch
def moved(series, at, before=200, window=50, k=3):
    """Steps after `at` until the rolling median of series over `window`
    steps leaves the median ± k MAD of the `before` steps before `at`."""
    x = series.dropna()
    ref = x[(x.index >= at - before) & (x.index < at)]
    center, mad = ref.median(), (ref - ref.median()).abs().median()
    width = max(1, round(window / np.median(np.diff(x.index))))  # samples
    roll = x[x.index >= at].rolling(width, min_periods=width).median()
    out = roll[(roll - center).abs() > k * mad]
    return f'+{int(out.index[0] - at)}' if len(out) else '—'

series = {'grad norm': main.set_index('step')['grad_norm'],
          'largest logit': main.set_index('step').filter(
              like='max_logit_').max(axis=1),
          'loss': main.set_index('step')['loss']}
events = {'end of warmup': run_cfg.warmup, 'start of decay': decay_from,
          'gradient event': event}
print(f'{"event":>15} ' + ' '.join(f'{k:>13}' for k in series))
for name, at in events.items():
    print(f'{name:>15} ' + ' '.join(f'{moved(x, at):>13}'
                                   for x in series.values()))
```

## Reading Checkpoints and the Ladder

### Checkpoints Scored

<!-- Placeholder (prose): eleven checkpoints scored on validation loss and bits per byte and on five suite tasks at their full item sets (reused: EvalSuite and paired_bootstrap of sec_lm_eval_suites, bits_per_byte of sec_lm_scoring); near-chance tasks (ARC-Challenge, MMLU) carry no information at 138M and are named, not tabled. -->

```{.python .input #reference-model-checkpoint-evals}
%%tab pytorch
STEPS = (500, 1000, 2000, 3000, 4000, 4500, 4800, 5100, 5400, 5700, 6000)
SUITE = ('hellaswag', 'arc_easy', 'piqa', 'sciq', 'winogrande')
ev = d2l.reference_evals(STEPS, tasks=SUITE)
d2l._show_label(ev)
d2l.plot(ev.table['bpb'], ev.table[list(SUITE)].mean(axis=1),
         'validation bits per byte', 'suite mean accuracy', fmts=('o-',))
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
    print(f'{f"{a} → {b}":>12} {100 * r.mean:+9.1f}p {100 * lo:+7.1f} to '
          f'{100 * hi:+.1f} {"yes" if lo > 0 or hi < 0 else "not resolved":>13}')
```

### Per-Domain Bits per Byte

<!-- Placeholder (prose): per-domain bits per byte against step (FineWeb-Edu file 012 by URL-host class, plus a held-out book): which domains the decay helps. -->

```{.python .input #reference-model-domains}
%%tab pytorch
domains = ev.domains.pivot(index='step', columns='domain', values='bpb')
d2l.plot(domains.index.to_numpy(), [domains[c].to_numpy() for c in domains],
         'step', 'bits per byte', legend=list(domains), figsize=(5, 3))
```

### The Branch Point, Averaging and Selection Bias

<!-- Placeholder (prose): the branch point (the step-5,100 state from which a decay can be rerun); averaging four stable-phase checkpoints against the decay; and the bias of choosing the best of eleven checkpoints on the evaluation set itself. -->

```{.python .input #reference-model-averaging}
%%tab pytorch
AVERAGED, VAL_TOKENS = (4000, 4500, 4800, 5100), 2 ** 20
states = [d2l.load_reference_model(step=t, dtype=torch.float32).state_dict()
          for t in AVERAGED]
average = d2l.load_reference_model(step=AVERAGED[-1], dtype=torch.float32)
average.load_state_dict({k: sum(st[k] for st in states) / len(states)
                         for k in states[0]})
loss = {'average': d2l.val_loss(average, cfg, VAL_TOKENS)}
for t in (5100, 6000):
    loss[t] = d2l.val_loss(d2l.load_reference_model(step=t), cfg, VAL_TOKENS)
print(f'average of four stable checkpoints: {loss["average"]:.3f} '
      f'(s5100: {loss[5100]:.3f}; s6000 after decay: {loss[6000]:.3f})')
```

<!-- Placeholder (prose): selection bias: choose the best of the eleven checkpoints on one half of the items, re-score it on the other half, 200 random halvings. -->

```{.python .input #reference-model-selection-bias}
%%tab pytorch
scores = items.pivot_table(index=['task', 'item_id'], columns='step',
                           values='score')
task = scores.index.get_level_values('task')
rng, gains = np.random.default_rng(0), []
for _ in range(200):  # choose on one half of the items, re-score the other
    half = rng.random(len(scores)) < 0.5
    pick = scores[half].groupby(task[half]).mean().mean()
    other = scores[~half].groupby(task[~half]).mean().mean()
    best = pick.idxmax()
    gains.append((pick[best] - pick[STEPS[-1]], other[best] - other[STEPS[-1]]))
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
d2l._show_label(pts)
done = pts.points[pts.points['finished'].astype(bool)]
d2l.set_figsize((5, 3))
for i, df in enumerate([ladder, log]):
    for j, (job, g) in enumerate(df.groupby('job')):
        main_run = g[g['run'] == g['run'].iloc[0]].dropna(subset=['val_loss'])
        ends = g[g['run'] != g['run'].iloc[0]].groupby('run').last()
        color = f'C{3 * i + j}'
        d2l.plt.plot(main_run['tokens'], main_run['val_loss'], color=color,
                     label=job.rsplit('-s', 1)[0])
        d2l.plt.plot(ends['tokens'], ends['val_loss'], 'o', color=color)
d2l.plt.xscale('log')
d2l.plt.xlabel('tokens')
d2l.plt.ylabel('validation loss')
d2l.plt.legend(fontsize=7)
print(f'{"size":>11} {"P":>11} {"5N":>6} {"10N":>6} {"20N":>6}')
for size, g in done.sort_values(['P', 'tokens']).groupby('size', sort=False):
    print(f'{size:>11} {int(g["P"].iloc[0]):>11,} '
          + ' '.join(f'{v:6.3f}' for v in g['val_loss']))
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
    """Largest attention logit and mean attention entropy of every head, per
    block, on the first max_seqs sequences of batch, from explicit scores
    (fused kernels return none); under DDP the maximum is over all ranks."""
    X, ctx = d2l._probe_inputs(batch, max_seqs, model)
    stats = {}
    for l, (attn, Y, tables) in enumerate(d2l._attention_inputs(model, X,
                                                                ctx)):
        q, k, _ = attn.qkv(Y, tables=tables)
        k = k.repeat_interleave(attn.num_heads // attn.num_kv_heads, 1)
        m, ent = d2l._chunked_logit_stats(q, k, ctx, chunk)
        stats[l] = LogitStats(d2l._all_reduce_max(m), ent)
    return stats

#@save
def softcap(x, c):
    """Smoothly cap x to (-c, c)."""
    return c * torch.tanh(x / c)

#@save
def make_bad_batch(batch, kind='random', vocab=32768, seed=0):
    """A batch shaped like batch of one-document windows: uniform random ids
    >= 32 ('random') or one repeated id ('repeat'), targets shifted ids."""
    B, T = batch['input_ids'].shape
    g = torch.Generator().manual_seed(seed)
    ids = (torch.randint(32, vocab, (B, T + 1), generator=g) if kind ==
           'random' else torch.full((B, T + 1), 32 + seed % (vocab - 32)))
    pos = torch.arange(T, dtype=torch.int32).expand(B, T).contiguous()
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
            f_q = (tau / S).clamp(max=1) ** alpha
            f_k = (tau / S_kv).clamp(max=1) ** (1 - alpha)
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

<!-- Placeholder (prose): the T1 grid (six learning rates, two seeds, 400 steps, clipping off because it hides the growth that precedes a spike, z-loss and weight decay as in t1); tau fixed beforehand by the rule below; the stability criterion: finite and final loss at most l_ref + delta, l_ref the best final loss of the baseline over the grid, delta = 0.3 checked against three times the two-seed difference; the learning-rate sensitivity of Wortsman et al. -->

```{.python .input #reference-model-stable-lr}
%%tab pytorch
#@save
def tau_rule(log, lo, hi):
    """QK-clip's threshold, fixed before the run that uses it: ten times the
    ceiling of a tenth of the largest logit in log over steps lo to hi."""
    rows = log[log['step'].between(lo, hi)]
    return 10 * math.ceil(rows.filter(like='max_logit_').max().max() / 10)
```

```{.python .input #reference-model-stable-lr-arms}
%%tab pytorch
LRS = (1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2, 3.2e-2)
VARIANTS = ('none', 'qk_clip', 'qk_norm', 'softcap')
TAU_T1 = tau_rule(t1[t1['seed'] == 0], t1_cfg.warmup, 400)
inst = d2l.instability_arms(r2_config('t1', clip=None), lrs=LRS,
                            variants=VARIANTS, seeds=(0, 1), steps=400,
                            tau=TAU_T1)
d2l._show_label(inst)
arms = inst.table
ell_ref = arms.loc[arms['variant'] == 'none', 'final_loss'].min()
arms = arms.assign(stable=arms['final_loss'] <= ell_ref + DELTA)  # inf fails
pair = arms[(arms['variant'] == 'none') & (arms['lr'] == LRS[0])]
print(f'τ (t1) = {TAU_T1}')
print(f'δ = {DELTA}; 3 × two-seed difference at {LRS[0]:g}: '
      f'{3 * pair["final_loss"].diff().abs().iloc[-1]:.3f}')
for v in VARIANTS:
    g = arms[arms['variant'] == v]
    ok, agree = (g.groupby('lr')['stable'].all(),
                 g.groupby('lr')['stable'].nunique() == 1)
    good = [lr for lr in LRS if ok[lr]]
    sens = lr_sensitivity(g.groupby('lr')['final_loss'].mean().to_numpy(),
                          math.log(V))
    note = '' if agree.all() else (
        '; seeds disagree at ' + ', '.join(f'{lr:g}' for lr in agree.index[
            ~agree]))
    print(f'{v}: ' + (f'stable at lr {good[0]:g} to {good[-1]:g}' if good
                      else 'no rate stable')
          + f'; lr sensitivity {sens:.2f}{note}')
```

<!-- Placeholder (prose): final loss against the learning rate per variant (mean of the two seeds; a diverged run drawn at ln V), the band l_ref to l_ref + delta shaded. -->

```{.python .input #reference-model-stable-lr-plot}
%%tab pytorch
curves = arms.groupby(['lr', 'variant'])['final_loss'].mean().unstack()
d2l.plot(list(LRS), [curves[v].clip(upper=math.log(V)).to_numpy()
                     for v in VARIANTS], 'learning rate', 'final loss',
         legend=list(VARIANTS), xscale='log', fmts=('o-', 's-', '^-', 'd-'))
d2l.plt.axhspan(ell_ref, ell_ref + DELTA, color='0.9')
d2l.plt.show()
```

### Ten Times the Learning Rate

<!-- Placeholder (prose): at the reference architecture and recipe (clip 1.0, z-loss), 500 steps at ten times the peak rate, with and without QK-clip at a tau fixed from the reference log's steps 300-500 before the arms ran; the reference run's own first 500 steps are the baseline. Pending T2 (C1d): campaign marker for lm-r2-inst-t2-v1. -->

```{.python .input #reference-model-ten-x}
%%tab pytorch
TAU = tau_rule(log, 300, 500)
t2 = d2l.instability_arms(r2_config('r2'), lrs=(6e-3,),
                          variants=('none', 'qk_clip'), steps=500, tau=TAU)
d2l._show_label(t2)
names = {'baseline': 'baseline', 'none': 'no cure', 'qk_clip': '+ QK-clip'}
fig, axes = d2l.plt.subplots(2, 1, sharex=True, figsize=(5, 4))
for (v, lr), g in t2.traces.groupby(['variant', 'lr'], sort=False):
    probe = g.dropna(subset=['max_logit'])
    axes[0].plot(probe['step'], probe['max_logit'], label=f'{lr:g} {names[v]}')
    axes[1].plot(g['step'], g['loss'], lw=0.8)
axes[0].set_yscale('log')
axes[0].set_ylabel('largest logit')
axes[0].legend(fontsize=7)
axes[1].set_ylabel('train loss')
axes[1].set_xlabel('step')
print(f'τ = {t2.tau}')  # TAU, or the T1 grid's τ while the arms are pending
print(f'{"arm":>18} {"final loss":>10} {"largest logit":>13} '
      f'{"first non-finite step":>21}')
for _, r in t2.table.iterrows():
    first = '—' if pd.isna(r['first_nonfinite']) else int(
        r['first_nonfinite'])
    print(f'{f"{r.lr:g} {names[r.variant]}":>18} {r["final_loss"]:10.3f} '
          f'{r["max_logit"]:13.0f} {first:>21}')
```

### The Spike Protocol

<!-- Placeholder (prose): a bad batch on purpose at step 300 of the t1 preset with clipping off; PaLM's diagnosis by replay from the step-250 state with and without the batch and from the step-50 state; clipping at 1.0; the replay floor that sets every comparison on Ada (replays are not bit-exact). A table tab_lm_inst_diagnosis beside it maps outcomes to causes: bad data, unstable state, numerics. -->

```{.python .input #reference-model-spike}
%%tab pytorch
REPLAYS = (('from 250, same batches', 250, (), None),  # (label, from step,
           ('from 250, skip 300', 250, (300,), None),  # skipped steps, clip)
           ('from 50, same batches', 50, (), None),
           ('clip 1.0', 0, (), 1.0))
spike = d2l.spike_replay(r2_config('t1', clip=None), bad_step=300,
                         kind='random', replays=REPLAYS)
d2l._show_label(spike)
fig, axes = d2l.plt.subplots(2, 1, sharex=True, figsize=(5, 4))
for name, g in spike.traces.groupby('run', sort=False):
    g = g[g['step'] >= 250]
    axes[0].plot(g['step'], g['loss'], lw=0.8, label=name)
    axes[1].plot(g['step'], g['grad_norm'], lw=0.8)
axes[0].set_ylabel('train loss')
axes[0].legend(fontsize=7)
axes[1].set_ylabel('grad norm (pre-clip)')
axes[1].set_xlabel('step')
print(f'{"run":>22} {"loss at 300":>11} {"peak norm, 295-305":>18} '
      f'{"max |Δ loss|, 301-400":>21}')
for _, r in spike.table.iterrows():
    print(f'{r["run"]:>22} {r["loss_at_bad"]:11.3f} '
          f'{r["peak_grad_norm"]:18.2f} '
          + ('—' if pd.isna(r['max_dloss']) else f'{r["max_dloss"]:.1e}')
          .rjust(21))
print(f'replay floor (same state, same batches, no bad batch): '
      f'{spike.floor:.1e}')
```

## Optimizers as a Stated Fact

<!-- Placeholder (prose): no optimizer race here. Against a tuned AdamW, matrix optimizers (Muon, Soap, Kron) gain at most about 1.4x at small scale and about 1.1x at larger scale, and the gain shrinks with scale under standard weight decay (Fantastic Pretraining Optimizers I and II); the production Muon recipe (Muon is Scalable). The Muon variant of the t1 preset is an exercise. -->

## Summary

<!-- Placeholder (prose): 3-6 sentences; hand-off: how to size a model that will be served (sec_lm_budgets). -->

## Exercises

<!-- Placeholder (prose): 6-8 exercises per the spec (derive the bound under QK-clip and the Gram change one clip makes; Vd once vs 6Vd; the WSD integral fraction delta/(2 - delta); update ratios on the t1 run; a t1 learning-rate sweep; four ways to finish the t1 run from its step-1,297 state; the Muon variant against the stated bound). -->
