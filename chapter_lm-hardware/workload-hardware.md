# Training, Prefill and Decode Workloads
:label:`sec_lm_workload_hardware`

<!-- Author brief, copied verbatim from the Language Models part plan
(docs/llm/language_models_part_plan.md, "5.3 Training, Prefill and Decode Workloads · T2 (T1 reduced)").
Blueprint: docs/llm/blueprints/ch05_hardware.md section 5.2.
The plan is a specification, not book text; this comment is replaced
when the section is written.

**Idea:** Training, prefill and decode want different hardware because they sit on different sides of the ridge and fill memory in different ways.
**Opening:** The reader knows from `subsec_hw-two-workloads` that prefill is compute-bound and single-stream decode reads every weight per token, and from the batched-decode measurement of `sec_perf_model` that batching moves decode past the ridge at about batch 164 for one weight matrix. One concrete example: Qwen3-8B on one A100 at batch one leaves the chip idle, and at batch 128 with 8,192-token contexts it runs out of memory before it runs out of bandwidth. The open question: for each of the three workloads, which datasheet number binds, and what one line in the throughput-interactivity plane says about cost.
**Teaches:**
- Training needs FLOPs, capacity and network bandwidth together: FLOPs per token (restating `eq_six_nd`), bytes of state per parameter under the pure-BF16 and the mixed recipe (restating `subsec_mp-anatomy`), activation memory and communication per step; a requirements card per workload.
- Prefill is compute-bound and sets time to first token; a chip used only for prefill can trade HBM for cheaper memory (the Rubin CPX row of 5.1); the quadratic attention term and the crossover prompt length where it equals the matmul term.
- Decode is bound by bandwidth and capacity: the batch at which it turns compute-bound, the context beyond which it never does, the maximum batch that fits; for a mixture of experts the total parameters set capacity and the active parameters set FLOPs, while the bytes read per step grow with the batch as more experts are touched.
- The throughput-interactivity frontier: tokens per second per GPU against tokens per second per user is a straight line in the bandwidth regime, capped by the compute roof and cut by capacity; GPU-seconds per million tokens are read along it; consequences in one sentence each: separate prefill and decode pools, SRAM parts for low-latency decode, large-memory parts for long contexts, why a training cluster is rarely the cheapest place to serve.
**Derivation:** The decode step time t(b, L) = max(2 N_act b / P, (W + b L k_KV) / beta) under W + b L k_KV <= mu C. The compute-bound batch is b* = W / (2 N_act / R - L k_KV), which exists only if L < L_crit = 2 N_act / (R k_KV); at L = 0 and bf16 weights b* = R, the ridge point itself; in the bandwidth regime per-user rate u and per-GPU throughput T satisfy T = (beta - W u) / (L k_KV), a line whose intercept falls as 1/L (blueprint ch05 5.2 D1 to D3). Smallest numeric case: Qwen3-8B on the A100 PCIe 40 GB row, k_KV = 147,456 bytes per token from its configuration, gives L_crit of a few hundred tokens (about 550 with all parameters, computed in the cell and copied into prose at that precision), so at any realistic context the step is bandwidth-bound at every batch; an MLA-style cache a tenth the size moves L_crit by ten.
**Notebook:**
1. Question: does a calibrated max(compute, memory) model predict Qwen3-8B's training step time, prefill latency and decode throughput on one A100 over batch and context within 15%, and does it rank H100, B200 and an SRAM part the way a public benchmark does?
2. Model and data: Qwen3-8B random-initialized from its configuration for training and prefill timings (throughput does not depend on weight values; one control with real weights at one point), real weights for decode at batch 1 as a cross-check; depth-truncated series for training because the full 8B training state does not fit 40 GB; Qwen3-30B-A3B on the T2 pair for the expert-coverage series.
3. Calibration from 5.1's and 5.2's logs: P_eff from the shape sweep, beta_eff from the copy bench, the launch overhead; `d2l.hw_calibrate` returns the same dictionary that A6a's `calib.json` uses.
4. Measurement under the A6a protocol (CUDA events, static cache, CUDA-graph decode, warm-up and repeats, median and MAD) through `d2l.hw_train_bench`, `d2l.hw_prefill_bench`, `d2l.hw_decode_bench`; the grid is cut to the points whose predicted peak fits 0.9 x 40 GB and the dropped points are listed.
5. Fit: at most three free parameters beyond the calibration; predicted against measured on held-out grid points; the frontier plotted at three contexts.
6. Prediction: the frontier for H100, B200 and the Groq row from datasheets, against the committed InferenceX snapshot (Apache-2.0, with attribution and upstream commit hash); the comparison is a ranking, not a number.
7. Reduced T1 configuration: Qwen3-0.6B (train, prefill, decode) and Qwen3-1.7B (decode) on the reader's GPU, a grid of 12 points within 14 GiB, the same functions behind `config.tier`.
8. Claims that survive noise: the calibrated model lands within 15% on held-out points while a datasheet-only roofline misses by more (the notebook prints both); tokens per second per GPU rise with batch while tokens per second per user fall, and the measured points lie on the predicted line in the bandwidth regime.
**Figures:** `fig_lm_hw_three_workloads` (the three requirements cards against the ridge); `fig_lm_hw_frontier` (plot: per-GPU against per-user throughput at three contexts, compute roof and capacity cut); `fig_lm_hw_moe_coverage` (plot: distinct experts touched against batch, formula against the measured router).
**Restate / forward:** `subsec_hw-two-workloads` and `eq_hw_decode_bound` (the bytes-per-token bound is the b = 1 case); `subsec_kv-cache-batching` (batching shares weight reads, not cache reads); `eq_kv-cache-bytes` (k_KV per token taken as given); `subsec_mp-anatomy` (16 bytes per parameter with Adam); `sec_moe` (total against active parameters); `eq_six_nd`; the batch-164 crossover of `sec_perf_model`. Forward: 13.8 (prefill and decode pools).
**Exercises:** 6 to 8. Derive b* and L_crit for Qwen3-30B-A3B with the expert-coverage term (derive); derive the prompt length at which attention FLOPs equal matmul FLOPs for Qwen3-8B (derive); measure KV-cache growth against context at fixed batch and compare with k_KV (measure); measure the eager decode series and attribute the gap to launch overhead (measure); check that `FlopCounterMode` halves causal attention or not, and state the convention (check); open: extend `ServingModel` with a KV cache a tenth the size and predict where the frontier moves.
**Writer notes:** The model is `d2l.ServingModel(cfg, chip, calib=None, dtype_bytes=2, kv_bytes=2, mem_util=0.9, kv_bytes_override=None)` with `.decode_step(b, L)`, `.prefill_time(n, b=1)`, `.max_batch(L)`, `.b_star(L)`, `.L_crit()`, `.frontier(...)`, `.gpu_s_per_mtok(b, L)`; it is the one decode step-time model of the part (`eq_lm_hw_decode_step`), used by Chapters 12, 13 and 14; there is no `d2l.decode_step_time`. `d2l.TrainStepModel(cfg, chip, calib=None, recipe='bf16_pure'|'mixed16')` delegates FLOPs and MFU to 4.2's `d2l.flops_per_token` and `d2l.mfu`. Pitfalls (blueprint 5.2): "decode is memory-bound" is true of weights at small batch, not of the KV cache at long context; MFU on peak against on achievable FLOP/s differ by P_eff/P, state the denominator; TTFT in a service includes queueing, so the measured quantity is "prefill latency of the model forward"; pure-BF16 training (8 bytes per parameter) is a different algorithm from the mixed recipe; eager decode of a small model measures launch overhead. Risks: the full-depth 8B training state is 65.5 GB at 8 bytes per parameter, so the depth-truncated series with its linearity check is the method, not a fallback; the Transformers static-cache plus CUDA-graph path may break for Qwen3 at the pin, in which case `d2l.load_qwen3` (1.2) with preallocated caches is the path and both are timed once at batch 1; Qwen3-30B-A3B does not fit one 40 GB card, so it is split layer-wise over the T2 pair and the grid is cut to b L of about 1e5 tokens; the InferenceX dashboard application is GPL-3.0 while the results repository is Apache-2.0, so commit the trimmed CSV only. Do not re-teach the roofline, prefill against decode intensity at one batch (`subsec_hw-two-workloads`), FlashAttention. Artifacts: `lm-hw-a100-workloads`, `lm-hw-inferencex-snapshot`. Detail: blueprint ch05 5.2 (grid, protocol, D1 to D5, risks). The MoE series stays in this section: it is a hardware statement about bytes against batch.
**Sources:**
- Papers: [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102) (the step-time model and the frontier); [Splitwise: Efficient generative LLM inference using phase splitting](https://arxiv.org/abs/2311.18677) (prefill and decode on different hardware); [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://arxiv.org/abs/2401.09670) (the pool argument; 13.8 develops it); [LLM Inference Unveiled: Survey and Roofline Model Insights](https://arxiv.org/abs/2402.16363) (roofline view of inference).
- Code and documentation: SemiAnalysis [InferenceX](https://github.com/SemiAnalysisAI/InferenceX) and its [dashboard](https://inferencex.semianalysis.com/) (the public frontier the predictions are ranked against); [MLCommons inference benchmarks](https://mlcommons.org/benchmarks/inference-datacenter/) (scenario definitions).
- Courses, tutorials and blogs: [Scaling Book: All About Transformer Inference](https://jax-ml.github.io/scaling-book/inference/) (the derivation's source); [Efficiency in LLMs (MLSS)](https://alex.smola.org/posts/45-mlss-efficiency/) (prefill and decode narrative).
-->

<!-- Placeholder (prose): opening. The reader knows from
subsec_hw-two-workloads that prefill is compute-bound and that decode at
batch one reads every weight per token, and from sec_perf_model that
batching moves decode past the ridge. Qwen3-8B on one RTX 4090: its 16.4 GB
of bf16 weights fit the 24 GB card; by the datasheet a decode step at
batch one takes at least about 1/65 of a second (measured below: about
1/50) and keeps the matrix units busy for under 1% of it; and
eight sequences of 8,192 tokens do not fit (b_max = 4 at that context), so
capacity binds before compute. The open question: which datasheet number
binds each of the three workloads, and what one line in the
throughput-interactivity plane says about cost. -->

```{.python .input #workload-hardware-imports}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #workload-hardware-extras}
%%tab pytorch
import pandas as pd
```

<!-- Placeholder (prose): the configurations come from the pinned Hub
configurations (no weights), the chips from the section-22.1 table; the
model of this section, ServingModel, is built below, and the cell uses it
ahead of its derivation. -->

```{.python .input #workload-hardware-opening}
%%tab pytorch
specs = d2l.chip_specs()
chip, a100 = specs.loc['RTX 4090'], specs.loc['A100 PCIe 40 GB']
cfg_8b, cfg_06b, cfg_moe = (d2l.hw_config(name) for name in (
    'Qwen/Qwen3-8B', 'Qwen/Qwen3-0.6B', 'Qwen/Qwen3-30B-A3B'))
m = d2l.ServingModel(cfg_8b, chip)
t_compute, t_memory, _ = m.decode_terms(b=1, L=2048)  # seconds
t_step = m.decode_step(b=1, L=2048)
print(f'Qwen3-8B weights in bf16: {m.weight_bytes() / 1e9:.1f} GB')
print(f'decode at b = 1, L = 2,048, datasheet bound: {t_step * 1e3:.1f} ms '
      f'per token ({1 / t_step:.0f} tokens/s)')
print(f'matrix units busy for {t_compute / t_step:.1%} of that step')
print(f'largest batch at L = 8,192 in 90% of {chip.mem_gb:.0f} GB: '
      f'b_max = {m.max_batch(8192)}')
```

## Three Workloads, Three Requirement Cards
:label:`subsec_lm_hw_cards`

<!-- Placeholder (prose): what each workload asks of the machine, one card
each (figure fig_lm_hw_three_workloads, generator three-workloads: the
three cards against the ridge). Training needs FLOPs, capacity and network
bandwidth together: FLOPs per token restating eq_six_nd, bytes of state per
parameter (pure bf16 8, the mixed recipe 16, restating
subsec_mp-anatomy), activations, and per step a gradient exchange of 2 N
bytes per replica. Prefill is compute-bound and sets time to first token;
a prefill-only chip can trade HBM for cheaper memory (the Rubin CPX row of
sec_lm_accelerator_design); its attention term grows with the square of
the prompt. Decode is bound by bandwidth and capacity. Everything below is
counted from the configuration: parameters from its shapes, bytes from
parameters. -->

```{.python .input #workload-hardware-serving-model}
%%tab pytorch
class ServingModel:  #@save
    """Time and memory of serving a dense Qwen3 or Qwen3-MoE configuration
    on n_chips identical chips (a chip_specs row): weights of dtype_bytes,
    KV entries of kv_bytes (or kv_bytes_override bytes per token), the
    fraction mem_util of memory usable. A calibration (hw_calibrate)
    replaces the datasheet's FLOP/s and bandwidth by measured ones."""
    def __init__(self, cfg, chip, calib=None, dtype_bytes=2, kv_bytes=2,
                 mem_util=0.9, kv_bytes_override=None, n_chips=1):
        self.cfg, self.calib = d2l._hw_cfg(cfg), dict(calib or {})
        self.n_chips = n_chips
        self.dtype_bytes, self.kv_bytes = dtype_bytes, kv_bytes
        self.kv_override = kv_bytes_override
        P, beta = d2l.hw_rates(chip, calib, dtype_bytes)  # FLOP/s, byte/s
        self.P, self.beta = n_chips * P, n_chips * beta
        self.C = n_chips * chip.mem_gb * 1e9 * mem_util  # usable bytes
```

<!-- Placeholder (prose): parameters from the shapes: attention
(query, key, value and output projections; grouped-query attention has
fewer key and value heads), the gated MLP (three d x d_ff matrices), or
for a mixture of experts E such MLPs of width d_ff_moe and a router, of
which a token uses K. The tests check the count against the model built
on the meta device. -->

```{.python .input #workload-hardware-n-params}
%%tab pytorch
@d2l.add_to_class(ServingModel)  #@save
def experts(self):
    """(E, K, d_ff of an expert, MoE layers); (1, 1, d_ff, 0) when dense."""
    c = self.cfg
    E = getattr(c, 'num_experts', 0) or 0
    if E <= 1:
        return 1, 1, c.intermediate_size, 0
    dense = set(getattr(c, 'mlp_only_layers', None) or [])
    n_moe = sum(1 for i in range(c.num_hidden_layers)
                if i not in dense and (i + 1) % c.decoder_sparse_step == 0)
    return E, c.num_experts_per_tok, c.moe_intermediate_size, n_moe

@d2l.add_to_class(ServingModel)  #@save
def n_params(self):
    """(total, active): every expert, or the K experts a token uses."""
    c, (E, K, f, n_moe) = self.cfg, self.experts()
    d, n_layers, d_h = c.hidden_size, c.num_hidden_layers, c.head_dim
    attn = 2 * d * d_h * (c.num_attention_heads + c.num_key_value_heads)
    norms = 2 * d_h + 2 * d  # query and key norms, two layer norms
    embed = c.vocab_size * d * (1 if c.tie_word_embeddings else 2) + d
    base = (embed + n_layers * (attn + norms)
            + (n_layers - n_moe) * 3 * d * c.intermediate_size)
    return (base + n_moe * (E * 3 * d * f + d * E),
            base + n_moe * (K * 3 * d * f + d * E))
```

<!-- Placeholder (prose): bytes. Capacity holds every weight; a decode step
reads every weight but the input embedding table (a lookup reads one row
per token), and of a mixture of experts only the experts its tokens touch;
the KV cache holds a key and a value per layer, KV head and token
(eq_kv-cache-bytes, taken as given). A token multiplies with the active
parameters less the embedding table: N_act below; N_body leaves out the LM
head too (prefill runs the head on one position only). -->

```{.python .input #workload-hardware-bytes}
%%tab pytorch
@d2l.add_to_class(ServingModel)  #@save
def weight_bytes(self):
    return self.n_params()[0] * self.dtype_bytes

@d2l.add_to_class(ServingModel)  #@save
def kv_bytes_per_token(self):
    if self.kv_override is not None:
        return self.kv_override
    c = self.cfg
    return (2 * c.num_hidden_layers * c.num_key_value_heads * c.head_dim
            * self.kv_bytes)

@d2l.add_to_class(ServingModel)  #@save
def n_matmul(self):
    """N_act: active parameters less a separate input embedding table."""
    c = self.cfg
    lookup = 0 if c.tie_word_embeddings else c.vocab_size * c.hidden_size
    return self.n_params()[1] - lookup

@d2l.add_to_class(ServingModel)  #@save
def n_body(self):
    """N_act less the LM head: the matrix parameters of the layers."""
    return self.n_matmul() - self.cfg.vocab_size * self.cfg.hidden_size
```

<!-- Placeholder (prose): the bytes a decode step of b tokens reads: the
weights less the embedding table, and of a mixture of experts only the
experts some token routes to, a fraction that grows with b (uniform
routing below, or a measured coverage). -->

```{.python .input #workload-hardware-three-workloads-three-requirement-cards-1}
%%tab pytorch
@d2l.add_to_class(ServingModel)  #@save
def read_bytes(self, b, coverage=None):
    """Weight bytes one step of b tokens reads; coverage: the fraction of
    experts they touch (a number, a function of b, or None for uniform
    routing)."""
    c, (E, K, f, n_moe) = self.cfg, self.experts()
    lookup = 0 if c.tie_word_embeddings else c.vocab_size * c.hidden_size
    W = self.weight_bytes() - lookup * self.dtype_bytes
    if n_moe == 0:
        return W
    frac = coverage(b) if callable(coverage) else coverage
    if frac is None:
        frac = d2l.moe_expert_coverage(E, K, b) / E
    expert_bytes = 3 * c.hidden_size * f * self.dtype_bytes
    return W - n_moe * E * expert_bytes * (1 - frac)
```

<!-- Placeholder (prose): prefill of n tokens up to the first new token:
2 N_body n FLOPs for the layers' matrix products, 2 V d for the LM head on
the last position only (what a service computes before the first token;
the benchmark below keeps one position's logits too), plus the causal
attention scores, 2 n_layers H d_h n^2 (half of the 4 n^2 d_h per head of
the two products); the weights are read once and the cache written once.
The attention term equals the layers' matrix term at
n = N_body / (n_layers H d_h), the crossover prompt length (an exercise
derives it). -->

```{.python .input #workload-hardware-prefill}
%%tab pytorch
@d2l.add_to_class(ServingModel)  #@save
def prefill_time(self, n, b=1):
    """Seconds to prefill b prompts of n tokens up to the last position's
    logits: the larger of the FLOP time and the time to read the weights
    and write the cache."""
    c = self.cfg
    head = self.n_matmul() - self.n_body()  # V d, run on the last position
    attn = 2 * c.num_hidden_layers * c.num_attention_heads * c.head_dim * n**2
    flops = b * (2 * self.n_body() * n + 2 * head + attn)
    nbytes = self.read_bytes(b * n) + b * n * self.kv_bytes_per_token()
    return max(flops / self.P, nbytes / self.beta) + self.launch()

@d2l.add_to_class(ServingModel)  #@save
def prefill_crossover(self):
    """Prompt length at which attention FLOPs equal the FLOPs of the
    layers' matrix products."""
    c = self.cfg
    return self.n_body() / (c.num_hidden_layers * c.num_attention_heads
                            * c.head_dim)
```

<!-- Placeholder (prose): training on one chip: forward and backward at the
chip's FLOP/s (6 N_mm per token plus causal attention, chapter 21's
flops_per_token with the attention term halved: the convention this
chapter states each time; two corrections for what the benchmark runs: an
untied input embedding is a lookup, not a product, so its 6 V d go, and
chunked_lm_loss recomputes the head's forward in backward, so 2 V d come;
chapter 21's count is 29-40% high for the depth-truncated 8B models, 5%
high at full depth, and 7% low for Qwen3-0.6B, whose head is tied), then
one AdamW pass over the state at memory
bandwidth: pure bf16 reads weight, gradient and both moments and writes
three of them back, 14 bytes per parameter; the mixed recipe moves 28.
One more memory term: the loss works on fp32 logits, V of them per token
(151,936 for Qwen3), and writes or reads each about ten times over the
forward pass, the backward's recompute and the softmax gradient: about
40 bytes per logit, 6 MB per token, 12 GB for 2,048 tokens, as much as the
weights. Measured on the box, the 2,048-token loss spends 34 ms beyond its
four matrix products (98 ms in all), more than 40 bytes per logit at the
copy rate: its elementwise kernels run below the copy rate; the fitted
eta_compute absorbs the rest. Without this term one compute efficiency
cannot fit training and prefill at once. Pure bf16 is a different
algorithm from the mixed recipe (pitfall). MFU
divides chapter 21's count (not the corrected one) by the datasheet peak;
against the achievable P_eff it is larger by P / P_eff: state the
denominator. -->

```{.python .input #workload-hardware-train-model}
%%tab pytorch
class TrainStepModel:  #@save
    """Seconds per training step on one chip: forward and backward at its
    FLOP/s, then one AdamW pass over the state at its bandwidth. Recipe
    'bf16_pure' keeps weights, gradients and both moments in bf16;
    'mixed16' adds fp32 master weights and fp32 moments."""
    state = {'bf16_pure': 8, 'mixed16': 16}  # bytes held per parameter
    traffic = {'bf16_pure': 14, 'mixed16': 28}  # bytes AdamW moves per one
    loss_bytes = 40  # per logit: about ten fp32 passes over the logits

    def __init__(self, cfg, chip, calib=None, recipe='bf16_pure'):
        self.cfg, self.chip, self.recipe = d2l._hw_cfg(cfg), chip, recipe
        self.serving = d2l.ServingModel(cfg, chip, calib)  # counts, rates
        self.N = self.serving.n_params()[0]

    def flops_per_token(self, s):
        """Chapter 21's count, less an untied input embedding (a lookup)
        plus the head's forward again (chunked_lm_loss recomputes it)."""
        c = self.cfg
        head = c.vocab_size * c.hidden_size
        lookup = 0 if c.tie_word_embeddings else head
        return (d2l.flops_per_token(c, T=s, causal_half=True)
                - 6 * lookup + 2 * head)
```

```{.python .input #workload-hardware-train-step}
%%tab pytorch
@d2l.add_to_class(TrainStepModel)  #@save
def step_time(self, b, s, ckpt=False):
    """b sequences of s tokens; ckpt recomputes the forward pass."""
    m = self.serving
    flops = self.flops_per_token(s) * b * s * (4 / 3 if ckpt else 1)
    logits = self.cfg.vocab_size * b * s  # each one moved in fp32 by the loss
    return (flops / m.P + (self.traffic[self.recipe] * self.N
                           + self.loss_bytes * logits) / m.beta
            + m.launch(passes=3))  # forward, and backward twice its kernels

@d2l.add_to_class(TrainStepModel)  #@save
def mfu(self, tokens_per_s, s):
    """Model FLOPs utilization against the datasheet's bf16 peak."""
    return d2l.mfu(tokens_per_s, self.cfg, self.chip.p16_tflops * 1e12,
                   T=s, causal_half=True)
```

<!-- Placeholder (prose): the three cards for Qwen3-8B on the RTX 4090 and
the A100 PCIe 40 GB: the full training state does not fit either card
(the depth-truncated series below is therefore the method, not a
fallback), a 2,048-token prefill takes a fraction of a second, and decode
is a matter of bytes. -->

```{.python .input #workload-hardware-cards}
%%tab pytorch
s, n, L = 4096, 2048, 2048  # training sequence, prompt, decode context
cards = {}
for name, c in (('RTX 4090, datasheet', chip),
                ('A100 PCIe, datasheet', a100)):
    train, serve = d2l.TrainStepModel(cfg_8b, c), d2l.ServingModel(cfg_8b, c)
    state = {r: k * train.N / 1e9 for r, k in train.state.items()}  # GB
    cards[name] = {
        'train: GFLOP per token': train.flops_per_token(s) / 1e9,
        'train: state, pure bf16 (GB)': state['bf16_pure'],
        'train: state, mixed (GB)': state['mixed16'],
        'memory of the chip (GB)': c.mem_gb,
        'prefill: crossover n': serve.prefill_crossover(),
        f'prefill: {n:,} tokens (ms)': serve.prefill_time(n) * 1e3,
        'decode: k_KV (KiB per token)': serve.kv_bytes_per_token() / 2**10,
        f'decode: b = 1, L = {L:,} (ms)': serve.decode_step(1, L) * 1e3}
d2l.show_table(pd.DataFrame(cards))
```

## The Decode Step
:label:`subsec_lm_hw_decode_step`

<!-- Placeholder (prose): derivation D1 and eq_lm_hw_decode_step (the
part's one decode step-time model, used by later chapters): with N_act
active parameters, W bytes of weights read, k_KV bytes per token of cache
and context L, a step of b sequences takes
t(b, L) = max(2 N_act b / P, (W + b L k_KV) / beta) under the capacity
condition W + b L k_KV <= mu C. Attention's own FLOPs are left out: decode
attention has intensity about 2g / bytes per element (g the GQA group),
far below the ridge R = P / beta, so its FLOPs hide behind its reads. The
b = 1 case is eq_hw_decode_bound. Batching shares weight reads, not cache
reads (subsec_kv-cache-batching). "Decode is memory-bound" is about weights
at small batch, not the KV cache at long context (pitfall). -->

```{.python .input #workload-hardware-decode-step}
%%tab pytorch
@d2l.add_to_class(ServingModel)  #@save
def decode_terms(self, b, L, coverage=None):
    """Seconds of one decode step of b sequences at context L: matrix
    products, memory reads (weights, then every sequence's cache), and the
    launch term of a calibrated chip."""
    t_compute = 2 * self.n_matmul() * b / self.P
    eta_kv = self.calib.get('eta_kv', 1.0)  # fitted cache-read efficiency
    t_memory = (self.read_bytes(b, coverage) / self.beta
                + b * L * self.kv_bytes_per_token() / (eta_kv * self.beta))
    return t_compute, t_memory, self.launch()

@d2l.add_to_class(ServingModel)  #@save
def decode_step(self, b, L, coverage=None):
    t_compute, t_memory, t_launch = self.decode_terms(b, L, coverage)
    return float(np.maximum(t_compute, t_memory)) + t_launch  # NaN stays

@d2l.add_to_class(ServingModel)  #@save
def max_batch(self, L):
    """Largest b whose weights and caches fit (derivation D2)."""
    free = self.C - self.weight_bytes()
    return max(0, int(free // (L * self.kv_bytes_per_token())))
```

<!-- Placeholder (prose): solving 2 N_act b / P = (W + b L k_KV) / beta
for b gives the compute-bound batch b* = W / (2 N_act / R - L k_KV), which
exists only while L < L_crit = 2 N_act / (R k_KV); at L = 0 and bf16
weights b* is the ridge R itself (the batch-164 crossover of sec_perf_model
on this card). Beyond L_crit the step is bandwidth-bound at every batch. An
MLA-style cache a tenth the size moves L_crit by ten. -->

```{.python .input #workload-hardware-b-star}
%%tab pytorch
@d2l.add_to_class(ServingModel)  #@save
def L_crit(self):
    """Context beyond which no batch makes decode compute-bound."""
    k = self.kv_bytes_per_token() / self.calib.get('eta_kv', 1.0)
    return 2 * self.n_matmul() / (self.P / self.beta * k)

@d2l.add_to_class(ServingModel)  #@save
def b_star(self, L):
    """Batch at which decode turns compute-bound (inf beyond L_crit)."""
    k = self.kv_bytes_per_token() / self.calib.get('eta_kv', 1.0)
    slack = 2 * self.n_matmul() / (self.P / self.beta) - L * k
    return self.read_bytes(1) / slack if slack > 0 else math.inf
```

<!-- Placeholder (prose): the numbers for Qwen3-8B on both cards and for
Qwen3-0.6B on the 4090 (k_KV = 147,456 and 114,688 bytes per token): L_crit
is a few hundred tokens for the 8B model (about 630 on the 4090, about 520
on the A100 PCIe; counting the embedding table as the plan does, about 680
and 550) and about 60 for the 0.6B model, and at every context the
capacity cut b_max comes before b*: capacity binds before compute (claim
W3). -->

```{.python .input #workload-hardware-decode-limits}
%%tab pytorch
contexts = (512, 2048, 8192)
models = {'8B, RTX 4090': (cfg_8b, chip), '8B, A100 PCIe': (cfg_8b, a100),
          '0.6B, RTX 4090': (cfg_06b, chip)}
limits = {}
for name, (cfg, c) in models.items():
    sm = d2l.ServingModel(cfg, c)
    limits[name] = {'ridge R (FLOP/byte)': sm.P / sm.beta,
                    'L_crit (tokens)': sm.L_crit(),
                    'L_crit, embedding counted':
                        sm.L_crit() * sm.n_params()[1] / sm.n_matmul(),
                    'b* at L = 0': sm.b_star(0),
                    'b* at L = 512': sm.b_star(512)}
    for L in contexts:
        limits[name][f'b_max at L = {L:,}'] = sm.max_batch(L)
    assert all(sm.max_batch(L) < sm.b_star(L) for L in contexts)  # W3
print('from the datasheets (b* infinite: none)')
d2l.show_table(pd.DataFrame(limits).replace(math.inf, 'none'), digits=0)
```

### Mixtures of Experts
:label:`subsec_lm_hw_moe_bytes`

<!-- Placeholder (prose): for a mixture of experts the total parameters set
capacity, the active ones the FLOPs (sec_moe), and the bytes of a step grow
with the batch as its tokens touch more experts. Derivation D4: under
independent uniform top-K routing of b tokens over E experts, an expert is
untouched with probability (1 - K/E)^b, so E (1 - (1 - K/E)^b) experts are
read per layer; half of Qwen3-30B-A3B's 128 experts are touched at about
b = 11. Real routers are not uniform; the measurement below records the
router's choices for real text. -->

```{.python .input #workload-hardware-moe-coverage}
%%tab pytorch
def moe_expert_coverage(E, K, b):  #@save
    """Expected number of distinct experts that b tokens touch when each
    picks K of E uniformly (and independently of the others)."""
    return E * (1 - (1 - K / E)**b)
```

<!-- Placeholder (prose): the router of Qwen3-30B-A3B (48 layers, 128
experts, top-8; Apache-2.0) on real text, the model split layer by layer
over the box's four cards (61 GB of bf16 weights): 1,536 documents of
FineWeb-Edu's validation split (ODC-By) and 64 windows of the code text
of sec_lm_scoring's held-out texts, 256 tokens each; per sequence and
layer the eight experts chosen at its last position. Coverage at batch b
is the number of distinct experts in the union of b sequences, averaged
over the layers, from 200 random draws (the band: 5-95%). The four-card
model is not run on the page: the result is the committed regeneration
(claim W5). -->

```{.python .input #workload-hardware-moe}
%%tab pytorch
cov = d2l.hw_moe_coverage('box')
d2l.plot_coverage(cov, d2l.moe_expert_coverage)
```

<!-- Placeholder (prose): bytes per step with the measured coverage against
the uniform formula (the measured columns appear once box-moe is
committed). On the box the model runs layer-split over four
cards, which work one after another, so a step reads its bytes at one
card's bandwidth: the one-card prediction is the box's (capacity is not
used here; 61 GB does not fit one card). -->

```{.python .input #workload-hardware-moe-bytes}
%%tab pytorch
moe = d2l.ServingModel(cfg_moe, chip)
L = 2048
rows = {}
for b in (1, 8, 64):
    rows[b] = dict(uniform_gb=moe.read_bytes(b) / 1e9,
                   measured_gb=moe.read_bytes(b, cov.coverage) / 1e9,
                   uniform_ms=moe.decode_step(b, L) * 1e3,
                   measured_ms=moe.decode_step(b, L, cov.coverage) * 1e3)
table = pd.DataFrame(rows).T.dropna(axis=1)  # measured: NaN while pending
d2l.show_table(table.rename_axis('b'), headers={
    'uniform_gb': 'GB, uniform', 'measured_gb': 'GB, measured',
    'uniform_ms': 'ms, uniform', 'measured_ms': 'ms, measured'})
```

## Calibrating and Measuring
:label:`subsec_lm_hw_calibrate`

<!-- Placeholder (prose): the model's inputs as measured, not as printed:
P_eff (the best bf16 product of the shape sweep of sec_lm_kernels), beta_eff
(a DRAM-sized copy, read plus write counted, of sec_lm_accelerator_design)
and alpha_k, the fixed cost of one kernel inside a CUDA graph (the
intercept of t = alpha + s / beta fitted to the copy times with 22.4's
fit_alpha_beta); the all-reduce's alpha and beta come from 22.5's fit of
this box. The dictionary is A5's calib.json. P_eff is a maximum over the
sweep and lands above the datasheet's 165 TFLOP/s: the card boosts its
clock above the one the datasheet figure assumes (say so; the fitted
eta_compute below absorbs the difference). -->

```{.python .input #workload-hardware-hw-calibrate}
%%tab pytorch
def hw_calibrate(device=None, log=None):  #@save
    """The chip as measured: P_eff_bf16 (FLOP/s), beta_eff (byte/s),
    alpha_k (seconds per kernel in a CUDA graph), the GPU, and the
    all-reduce's alpha_ar and beta_ar (None until 22.5's fit is committed).
    From log, the box-microbench tables, or measured now on device (about
    half a minute); None without a GPU or a committed log."""
    tables = d2l._calib_tables(log, device)  # copies by size, bf16 products
    if tables is None:
        return None
    copy, bf16_tflops, gpu = tables
    alpha_k, _, _ = d2l.fit_alpha_beta(2 * copy.bytes, copy.ms / 1e3)
    return dict(P_eff_bf16=bf16_tflops.max() * 1e12,  # the best product
                beta_eff=copy.gbs.iloc[-1] * 1e9,  # the largest copy
                alpha_k=alpha_k,  # the intercept: one kernel's fixed cost
                gpu=gpu, **d2l._calib_allreduce())  # None before 22.5
```

```{.python .input #workload-hardware-calibrate}
%%tab pytorch
calib = d2l.hw_calibrate(log=d2l.hw_box_log('box-microbench'))
d2l.show_calib(calib, chip, live=d2l.hw_calibrate())
```

<!-- Placeholder (prose): the measurement protocol (blueprint A5): random
weights from the configuration (timing does not depend on weight values;
the control below checks it), CUDA events, warm-up then repeats, median
and MAD. Decode: one new token per sequence attends over a static cache
whose L + 1 slots are all in use, so the step reads each cached byte once
(Transformers' own masked path copies the cache once per query head of a
KV group, four times for Qwen3-8B, which a serving kernel never does: the
harness drops the all-true mask and lets SDPA handle the groups); the step
is captured once in a CUDA graph and replayed, so no Python runs between
kernels. The eager series without the graph (d2l.hw_eager_decode, the
loop of sec_lm_scoring's hf_decode_ms) comes from the same library; eager
decode of a small model measures launch overhead (pitfall). Each bench
returns the median and MAD in milliseconds; the grid also records the
peak memory of every point. -->

```{.python .input #workload-hardware-decode-bench}
%%tab pytorch
def hw_decode_bench(model, b, L, steps=64):  #@save
    """Milliseconds per decode step of b sequences at context L: median
    and MAD over steps replays of one CUDA graph."""
    token = torch.randint(model.config.vocab_size, (b, 1), device='cuda')
    with torch.inference_mode():
        cache, refill = d2l.hw_static_cache(model, b, L)
        position = torch.tensor([L], device='cuda')

        def step():  # the cache holds L tokens again: every replay is alike
            refill()
            model(token, past_key_values=cache, cache_position=position,
                  logits_to_keep=1)
        times = d2l._hw_time(step, steps, max_inner=1)  # events, graph
    ms, mad = d2l._hw_stats(times)  # median and MAD
    return dict(ms=ms, mad=mad)
```

<!-- Placeholder (prose): prefill at batch one through the model's forward
with logits_to_keep=1 (sec_lm_scoring's hf_prefill), so the full-vocabulary
logits are never built (5.0 GB at 16,384 tokens); the interval ends when
the last position's logits exist: "prefill latency of the model forward",
TTFT without queueing or tokenization (pitfall). -->

```{.python .input #workload-hardware-prefill-bench}
%%tab pytorch
def hw_prefill_bench(model, n, b=1, repeats=5):  #@save
    """Milliseconds to prefill b random prompts of n tokens: median and MAD
    after one warm-up."""
    ids = torch.randint(model.config.vocab_size, (b, n), device='cuda')
    start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    times = []
    for _ in range(1 + repeats):
        start.record()
        out = d2l.hf_prefill(model, ids, n, keep=1)[0]
        stop.record()
        stop.synchronize()
        times.append(start.elapsed_time(stop))
        del out
    ms, mad = d2l._hw_stats(times[1:])  # median and MAD
    return dict(ms=ms, mad=mad)
```

<!-- Placeholder (prose): training: forward, backward and a fused AdamW
step in pure bf16, eager, on random tokens; the loss runs chunk by chunk
on the hidden states (chunked_lm_loss of sec_lm_data_path), because
Qwen3's 151,936-token vocabulary would make the logits alone 1.2-2.5 GB per
4,096 tokens. The full 8B state is 65.5 GB at 8 bytes per parameter, so
the 8B series keeps the full width and 1-3 of its 36 layers; step time
must be linear in the number of layers, which the grid checks. -->

```{.python .input #workload-hardware-train-bench}
%%tab pytorch
def hw_train_bench(model, b, s, repeats=10, warmup=3, chunk=2048):  #@save
    """Milliseconds per training step on b random sequences of s tokens:
    median and MAD."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    ids = torch.randint(model.config.vocab_size, (b, s + 1), device='cuda')

    def step():
        hidden = model.model(input_ids=ids[:, :-1]).last_hidden_state
        loss, n = d2l.chunked_lm_loss(hidden, model.lm_head.weight,
                                      ids[:, 1:], chunk=chunk)
        (loss / n).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    times = d2l._hw_time(step, repeats, graph=False, max_inner=1,
                         warmup=warmup)  # CUDA events around eager steps
    ms, mad = d2l._hw_stats(times)  # median and MAD
    return dict(ms=ms, mad=mad)
```

<!-- Placeholder (prose): the grid (part 3 of the spec in prose: decode of
Qwen3-8B at b = 1-64 and L = 512, 2,048, 8,192, and of Qwen3-0.6B; prefill
at batch one from 256 to 16,384 tokens; training of 1-3 layers of the 8B
model on one and two sequences of 2,048 tokens, and of the whole 0.6B model
on one sequence of 512 to 2,048 tokens (its activations at 4,096 tokens
would pass 20 GiB); the 0.6B decode points at L = 1,024 are
those of chapter 21's decode table), cut to the points whose weights and
caches fit 90% of the card (ServingModel.max_batch); the dropped points are
listed. The committed run measured every point on a quiet card; the live
check repeats four of them on the reader's GPU. -->

```{.python .input #workload-hardware-grid}
%%tab pytorch
grid = d2l.hw_workload_grid('box')
```

<!-- Placeholder (prose): the weight-value control: Qwen3-8B with random
weights, its real weights (from the Hub cache, loaded into the same class,
dtype and attention) and random weights again, back to back at b = 1,
L = 2,048; the two random arms measure the run-to-run noise (claim W4: the
real arm within the noise, the larger of 3 MADs and the random arms'
difference; same kernels in every arm). -->

```{.python .input #workload-hardware-real-weights}
%%tab pytorch
control = d2l.hw_weight_control('Qwen/Qwen3-8B', b=1, L=2048)
```

### A Ladder of Models
:label:`subsec_lm_hw_model_ladder`

<!-- Placeholder (prose): four models, each adding one term. M0: the
datasheet roofline. M1: the calibrated P_eff and beta_eff. M2: plus the
launch terms, measured, not fitted: in a CUDA graph the kernels of a decode
step (counted from its graph) times alpha_k; an eager step (prefill and
training here) cannot be faster than the CPU launches its kernels, which a
narrow twin of the model (same layers and kernels, width 256, negligible
GPU work) measures: the launch floor, which sets the time of the small
model's prefill and training (pitfall: eager small models measure the
CPU). M3: plus at most three
fitted parameters (a compute efficiency for the real kernels, a cache-read
efficiency of the attention kernel and a factor on alpha_k), fitted on the
calibration subset only (every third grid point); errors are on the other
points. Claim W1: M3 within 15% on held-out points for all three workloads
while M0 misses by more. Held-out points whose prediction is the launch
floor (Qwen3-0.6B's short prefills and its training) compare a measurement
with a measurement; the cell counts them apart. The depth-truncated
training series is a straight line in the number of layers (the cell
prints the largest distance), and the line at 36 layers is the full-depth
Qwen3-8B step that no 24 GB card can hold, beside M3's prediction there. -->

```{.python .input #workload-hardware-launch}
%%tab pytorch
@d2l.add_to_class(ServingModel)  #@save
def launch(self, passes=1):
    """Kernel cost of passes forward passes on a calibrated chip (M2):
    kernels per forward times alpha_k (times a fitted kappa in M3)."""
    k = self.calib
    return (passes * k.get('kernels_per_step', 0) * k.get('alpha_k', 0.0)
            * k.get('kappa', 1.0))
```

```{.python .input #workload-hardware-ladder}
%%tab pytorch
fits = d2l.fit_workload_models(grid, calib, chip)
```

## The Throughput-Interactivity Frontier
:label:`subsec_lm_hw_frontier`

<!-- Placeholder (prose): derivation D3. In the bandwidth regime the
per-user rate u = beta / (W + b L k_KV) and the per-GPU rate T = b u
satisfy T = (beta - W u) / (L k_KV): a line of slope -W / (L k_KV) and
intercept beta / (L k_KV), which falls as 1/L; capped by the compute roof
T <= P / (2 N_act) and cut by capacity at b_max. GPU-seconds per million
tokens are 10^6 / T, read along the line. -->

```{.python .input #workload-hardware-frontier-def}
%%tab pytorch
@d2l.add_to_class(ServingModel)  #@save
def frontier(self, L, batches):
    """Per-user and per-GPU tokens/s at each context in L and batch in
    batches and at the capacity limit max_batch(L); regime 'bw', 'compute'
    or 'infeasible' (beyond max_batch)."""
    rows = []
    for L_i in np.atleast_1d(L):
        b_max = self.max_batch(int(L_i))
        for b in sorted(set(np.atleast_1d(batches)) | {max(b_max, 1)}):
            t_c, t_m, t_l = self.decode_terms(int(b), int(L_i))
            t = max(t_c, t_m) + t_l
            regime = ('infeasible' if b > self.max_batch(L_i) else
                      'compute' if t_c > t_m else 'bw')
            rows.append(dict(b=int(b), L=int(L_i), user_tps=1 / t,
                             gpu_tps=b / t / self.n_chips, regime=regime))
    return pd.DataFrame(rows)

@d2l.add_to_class(ServingModel)  #@save
def gpu_s_per_mtok(self, b, L):
    """GPU-seconds per million generated tokens at batch b, context L."""
    return 1e6 * self.decode_step(b, L) * self.n_chips / b
```

<!-- Placeholder (prose): the frontier of Qwen3-8B on the 4090 under the
fitted model M3 at three contexts, each line ending at the capacity limit
b_max, with the measured decode points of the grid on it (claim W2:
per-GPU tokens/s rise with batch while per-user tokens/s fall, and the
held-out measured points sit within about 7% of the predicted line, with a
curvature the model lacks; the cell prints the error range on held-out
points and, per context, what an added sequence costs). The curvature,
root-caused with a kernel trace: at b = 1 cuBLAS reads the weights with a
GEMV kernel at the copy rate (16.5 ms of the 19.6 ms step at L = 512); from
b = 2 it switches to tensor-core GEMM tiles that read the same bytes 5-9%
slower (17.4 ms at b = 2, 18.1 ms at b = 16), so going from b = 1 to 2 costs
about a millisecond the model does not have. The attention kernels'
time grows as L k_KV / beta predicts. The roofline counts bytes; which
kernel reads them is a property of the library (pitfall-sized remark). -->

```{.python .input #workload-hardware-frontier}
%%tab pytorch
m_fit = d2l.ServingModel(cfg_8b, chip, calib=fits['calib']['Qwen/Qwen3-8B'])
front = m_fit.frontier(L=(512, 2048, 8192), batches=2**np.arange(8))
measured = d2l.plot_frontier(front, m_fit, measured=grid, fits=fits)
for L, p in measured.groupby('L'):  # W2: along b, per GPU up, per user down
    assert p.gpu_tps.is_monotonic_increasing, L
    assert p.user_tps.is_monotonic_decreasing, L
```

<!-- Placeholder (prose): cost along the line: GPU-seconds per million
tokens at an interactive point, at the capacity limit and at a short
context. Consequences, one sentence each: separate prefill and decode pools
(forward to Chapter 30 if its label resolves), SRAM parts for low-latency
decode, large-memory parts for long context, and why a training cluster is
rarely the cheapest place to serve. -->

```{.python .input #workload-hardware-gpu-seconds}
%%tab pytorch
L = 2048
b_cap = m_fit.max_batch(L)
points = {f'interactive, b = 1, L = {L:,}': (1, L),
          f'capacity limit, b = {b_cap}, L = {L:,}': (b_cap, L),
          'short context, b = 64, L = 512': (64, 512)}
for label, (b, context) in points.items():
    cost = m_fit.gpu_s_per_mtok(b, context)
    print(f'{label}: {round(cost, -2):,.0f} GPU-seconds per million tokens')
```

### Other Chips from Their Datasheets
:label:`subsec_lm_hw_other_chips`

<!-- Placeholder (prose): the same model predicts other chips from their
datasheet rows: H100, H200, B200, MI355X, the RTX 4090, and the Groq LPU
with the chip count that holds the weights (sec_lm_accelerator_design's
sram_chips_needed). The public reference is SemiAnalysis's InferenceX
(Apache-2.0, snapshot with attribution and upstream commit), a dense
70B-class model at its own precision and tensor-parallel degree, per GPU
at one interactivity point; the comparison is a ranking, not a number
(claim W6; chips absent from the snapshot are predicted, not ranked). The
Groq row is a bandwidth bound only: the table has no FLOP/s figure for it
and the model has no inter-chip hops, so its number stays out of the prose
(the compute term alone would cut it many times). -->

```{.python .input #workload-hardware-other-chips}
%%tab pytorch
snap = d2l.inferencex_snapshot()
ranks = d2l.rank_against(snap, specs)
```

<!-- Placeholder (prose): the A100 twin: campaign C-T2 runs the same grid on
one A100 PCIe 40 GB (8B training at 2, 4 and 6 layers); claim W7, the same
15% check there, is pending: part of the A100 campaign. -->
<!-- campaign: lm-hw-a100-workloads-v1 -->

```{.python .input #workload-hardware-a100-twin}
%%tab pytorch
fits_a100 = d2l.fit_workload_models(d2l.hw_campaign_log('a100-workloads'),
                                    chip=a100)
```

<!-- Placeholder (prose): close: where the bytes that do not fit go
(sec_lm_memory_tiers), and the serving pools of Chapter 30 by :numref: if
its labels resolve at writing time. -->

## Summary

<!-- Placeholder (prose): 3-6 sentences. -->

## Exercises

<!-- Placeholder (prose): 6-8 exercises per S/spec/22.3.md part 9: b* and
L_crit for Qwen3-30B-A3B with the expert coverage (derive; cfg_moe and cov
of the page); the prompt length where attention FLOPs equal matrix-product
FLOPs for Qwen3-8B (derive); KV-cache growth against context at fixed
batch against k_KV (measure, 3 min); the eager decode series and the
launch-overhead gap (measure, 3 min); FlopCounterMode and the causal
convention (check); open: a KV cache a tenth the size (kv_bytes_override)
and where the frontier moves. -->
