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

<!-- Placeholder (stage 2, infra step): the cells below are the section's
shared library, written before the section; the prose around them comes with
the section. Cell ids follow the section specification (S/spec/21.2.md). -->

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
import time
```

<!-- Placeholder (cell 3): the reference configuration and its presets. -->

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

<!-- Placeholder (cell 5): FLOPs per token (PaLM convention) and MFU. -->

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

<!-- Placeholder (cell 8): the WSD schedule and AdamW's two groups. -->

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

<!-- Placeholder (cell 9): one optimizer step from token-weighted
micro-batches. -->

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
                loss, _, part = d2l.chunked_lm_loss(
                    H, W, mb['labels'], z_loss=cfg.z_loss, return_nll=True)
            (loss * world / n_valid).backward()  # DDP averages the ranks
        nll = nll + part.detach()
    params = [p for g in optimizer.param_groups for p in g['params']]
    norm = torch.nn.utils.clip_grad_norm_(params, cfg.clip or math.inf)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return dict(loss=d2l._all_reduce_sum(nll) / n_valid, grad_norm=norm,
                clipped=norm > (cfg.clip or math.inf))
```

<!-- Placeholder (cell 10): the loop; DDP, compilation, logging, probes,
validation and checkpoint plumbing live in _PretrainRun. -->

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

<!-- Placeholder (cells 23-24): attention-logit statistics and the cures. -->

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
    x = torch.as_tensor(final_losses, dtype=torch.float64)
    x = torch.where(torch.isfinite(x), x, l0).clamp(max=l0)
    return float((x - x.min()).mean())
```

<!-- Placeholder (cell 24): QK-clip after an optimizer step (Kimi K2). -->

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
