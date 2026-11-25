# Training Budgets beyond the Compute Optimum
:label:`sec_lm_budgets`

<!-- Author brief, copied verbatim from the Language Models part plan
(docs/llm/language_models_part_plan.md, "4.6 Training Budgets beyond the Compute Optimum · T0").
Blueprint: docs/llm/blueprints/ch04_pretraining.md section 4.7.
The plan is a specification, not book text; this comment is replaced
when the section is written.

**Idea:** Once the tokens a model will generate enter the objective, the compute-optimal model is smaller and trained longer, and a three-point ladder can say so with an interval.
**Opening:** The Chinchilla rule of `sec_scaling-laws` calls the reference model undertrained for its size, and a first cell confirms it: at 22.7 tokens per parameter it sits near the training optimum, while Llama 3 8B reads almost 1,900. The reader saw in `sec_scaling-laws` that a model serving trillions of tokens is cheaper when smaller. Here the budget is written down, fitted to the chapter's own ladder, and solved. The closing question is what hardware the answer assumes, which Chapter 5 takes up.
**Teaches:**
- The lifetime objective $C = N(6D + 2D_{\textrm{inf}})$ at a target loss, its first-order condition, and the direction of the answer: more demand gives a smaller model and more tokens per parameter, down to a feasibility floor set by the irreducible loss.
- Measured rather than counted inference cost: decode is memory-bound at small batch, so GPU-seconds per generated token follow a roofline $r_{\textrm{dec}} = b / \max(t_{\textrm{mem}}, t_{\textrm{cmp}})$ with the KV bytes per token in the memory term; the measured version of the optimum shifts further toward small models.
- Repeated data under a token shortage: effective tokens $D' = U + U R^*(1 - e^{-R/R^*})$ over $R$ repeats of $U$ unique tokens, so the value of an epoch decays and compute spent on the fortieth repeat buys little.
- Fitting a scaling law on real logs: the Chinchilla form fitted to the ladder's loss-against-tokens points by least squares in log space, exponents with bootstrap intervals over ladder points, a withheld configuration predicted, and what a three-size ladder can and cannot say (the level of the optimal ratio is not resolved; the direction and the ordering are). Carries A5.
**Derivation:** The optimum with inference tokens. Minimize $N(6D + 2D_{\textrm{inf}})$ subject to $E + AN^{-\alpha} + BD^{-\beta} = \ell$. With $a = AN^{-\alpha}$ and $b = BD^{-\beta}$ the first-order condition is $\alpha a = \beta b\,(1 + D_{\textrm{inf}} / (3D))$, which reduces to `eq_chinchilla_optimum` at $D_{\textrm{inf}} = 0$, has a unique root above $N_{\min} = (A/(\ell - E))^{1/\alpha}$, and sends $N \to N_{\min}$ as demand grows. Smallest case (Hoffmann constants, $\ell = 2.0$): $N^*$ falls from 15.3B at no inference to 11.5B at $10^{12}$ and 6.5B at $10^{13}$ inference tokens, with tokens per parameter rising from 79 to 143 to 600; the notebook checks the $D_{\textrm{inf}} = 0$ root against the closed form to relative `1e-8` and monotonicity in demand, and repeats under the Besiroglu constants.
**Notebook:**
1. Question: fitted to the chapter's committed ladder, does the Chinchilla form give exponents inside the published intervals, and where does the optimum move once inference tokens count?
2. Model and data: the committed ladder logs and evaluations of 4.2 (`lm-ladder-logs`, four sizes at two budgets with plateau checkpoints as intermediate points), the miniature grid of `sec_scaling-laws`, the two published fits (Hoffmann; Besiroglu), and `lm-decode-rates` (tokens per second and memory for the reference model, Qwen3-0.6B, 1.7B and 4B at batches 1, 8, 32, 128 and context 1,024 on one Ada GPU, measured with 12.1's decoder or Hugging Face `generate` with a static cache as the fallback). CPU only, minutes.
3. Check: the fit recovers synthetic exponents from simulated ladders with the ladder's noise; `lifetime_cost` at no inference equals $6ND$; the roofline at batch 1 equals $W / (Nw + ck)$; `effective_tokens` at $R = 0$ equals $U$ and its derivative matches finite differences; the decode predictions at batch 1 land within 15% of the measured rates (larger deviations at large batch are reported).
4. Claim that survives noise: the ladder's fitted exponents agree with the published fits within their intervals (or the disagreement is stated as the ladder's limit); under both fits the lifetime-optimal tokens per parameter exceed the training-optimal ratio and grow with demand; measured decode costs shift the optimum further toward small models; the two fits disagree on the level of the optimal ratio by more than a tenfold change in demand over part of the range, so no single ratio is named and a comparison that flips inside the band is "undecided".
5. Reused: `sec_scaling-laws`'s fit code and its `optimum`; `d2l.flops_per_token` (4.2); `d2l.ServingModel` (5.3) once it exists, else the inline roofline.
**Figures:** `fig_lm_ladder_fit` (plot: ladder losses against tokens with the fitted law and the withheld point); `fig_lm_lifetime_cost` (plot: lifetime cost against model size for four demand levels, with the optimum and the break-even demand marked); `fig_lm_bud_repeat` (plot: effective tokens against repeats).
**Restate / forward:** `sec_scaling-laws` (loss follows a power law in parameters and tokens, `eq_chinchilla_law`; the compute-optimal split `eq_chinchilla_optimum`; its "Training Past the Optimum" subsection and the Llama 3 8B example; scaling-laws.md:498 and :825 defer budgets beyond the optimum and the cost of running out of data here). Forward: Chapter 5 (the hardware the decode rates assume).
**Exercises:** 5 to 7. Derive: the optimum under repeated data (substitute $D'$ for $D$); the break-even demand at which overtraining pays; the Llama 3 8B exercise of `sec_scaling-laws` under the lifetime objective. Measure: exponent sensitivity to removing one ladder point; the optimum's sensitivity to the MFU and decode-rate bands. Open: a mixture-of-experts row (active against total parameters) and why the dense law does not apply.
**Writer notes:**
- Fit constants are published fits of other corpora and tokenizers: never present $D^*/N^*$ as a property of real models; give the structure and a sensitivity band. Released-model token budgets in `tab_lm_bud_released` (Llama 3 8B about 15T, OLMo 2 4T, SmolLM2 1.7B about 11T, Qwen3 about 36T, the reference model 3.15B) are re-checked at authoring and unverifiable rows dropped.
- The ladder fit is core: with four sizes and two budgets plus plateau points, fit $E + AN^{-\alpha} + BD^{-\beta}$ by least squares on log loss with a bootstrap over points; state the degrees of freedom honestly (five parameters on about a dozen points); A5's tolerance on the exponents is set from that bootstrap, not from a hoped-for agreement with Chinchilla.
- Parameters are not serving cost: the decode roofline needs the KV bytes per token (the reference model's 18,432 B per token) and the batch; `lm-decode-rates` depends on the serving stack and is re-measured when 12.1's `FastGenerator` lands.
- The repeated-data constant $R^*$ is read from the data-constrained scaling paper at authoring; if it cannot be read, use the shape only. Overtraining multiplies training cost, so it pays only above break-even; smaller models may need longer contexts or more reasoning tokens, which raises $D_{\textrm{inf}}$.
- Do not re-teach: the 6ND count, the Chinchilla fit and the miniature grid (`sec_scaling-laws`); muP (`sec_scaling`). New here: the lifetime objective and its optimum, measured decode cost, repeated data, the fit on the ladder.
- Blueprint: ch04 §4.7 (D1 to D3 with numeric checks, `solve_optimum`, `lifetime_cost`, `decode_roofline`, `effective_tokens`, the scenario table). The section is core and adds the ladder fit and A5.
**Sources:**
- Papers: [Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws](https://arxiv.org/abs/2401.00448) (the lifetime objective); [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264) (effective tokens under repetition); [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) (the fit form and the training-only optimum, restated).
- Code and documentation: [CS336 assignment 3](https://github.com/stanford-cs336/assignment3-scaling) (fitting a law under a query budget, the pattern A5 follows); [SmolLM2 model card](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B) (an overtrained small model with its budget on the card).
- Courses, tutorials and blogs: [Alex Smola's MLSS efficiency tutorial](https://alex.smola.org/posts/45-mlss-efficiency/) (deployment cost as part of the design); [Stanford CS336](https://cs336.stanford.edu/) (the two scaling-law lectures).
-->

<!-- Placeholder (prose): opening. The Chinchilla rule of sec_scaling-laws (eq_chinchilla_law, eq_chinchilla_optimum, restated in one sentence each) asks whether the reference model is undertrained for its size; the cell computes its training compute 6ND and the training optimum at that compute under the two published fits: it sits near the optimum, while Llama 3 8B reads almost 1,900 tokens per parameter. The "Training Past the Optimum" subsection of sec_scaling-laws said why; here the budget is written down, fitted to the chapter's own ladder, and solved. Opening question: fitted to the chapter's committed ladder, does the Chinchilla form give exponents inside the published intervals, and where does the optimum move once inference tokens count? -->

```{.python .input #budgets-imports}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #budgets-imports-extra}
%%tab pytorch
from scipy import optimize, special
```

```{.python .input #budgets-opening}
%%tab pytorch
FITS = {  # (E, A, B, α, β) of eq_chinchilla_law, as in sec_scaling-laws
    # Hoffmann et al. (2022); d2l.HOFFMANN_FIT (the simulated ladder) too
    'Hoffmann': (1.69, 406.4, 410.7, 0.34, 0.28),
    'Besiroglu': (1.82, 482.0, 2085.4, 0.348, 0.366)}  # their refit (2024)

def train_optimum(C, E, A, B, alpha, beta):
    """(N*, D*) of eq_chinchilla_optimum at training compute C = 6ND."""
    G = (alpha * A / (beta * B)) ** (1 / (alpha + beta))
    N = G * (C / 6) ** (beta / (alpha + beta))
    return N, C / (6 * N)

cfg = d2l.r2_config('r2')
D_r2 = cfg.steps * cfg.global_batch * cfg.arch.T  # training tokens
C_r2 = d2l.flops_per_token(cfg.arch) * D_r2  # 6ND, no attention term
N_r2 = C_r2 / (6 * D_r2)
print(f'reference model: {N_r2 / 1e6:.0f}M parameters, {D_r2 / 1e9:.2f}B '
      f'tokens, {D_r2 / N_r2:.1f} tokens per parameter')
train_opt = {k: train_optimum(C_r2, *fit) for k, fit in FITS.items()}
print('training-optimal at its compute: N* = ' + ', '.join(
    f'{N / 1e6:.0f}M ({k})' for k, (N, D) in train_opt.items())
    + '; D*/N* = ' + ', '.join(f'{D / N:.0f}' for N, D in train_opt.values()))
N_llama, D_llama = 8.0e9, 15e12  # Llama 3 8B: parameters, training tokens
print(f'Llama 3 8B: ≈ {D_llama / 1e12:.0f}T tokens / {N_llama / 1e9:.1f}B '
      f'≈ {D_llama / N_llama:,.0f} per parameter')
```

<!-- Placeholder (prose): reading of the opening: the reference model is close to the training optimum under both fits; Llama 3 8B is two orders of magnitude past it, and so are most released small models (tab_lm_bud_released). Training compute alone cannot explain that choice. Roadmap: the lifetime objective (this section's derivation), its measured version, repeated data, and the fit of the law on the chapter's own ladder. -->

<!-- Table (prose): tab_lm_bud_released, a markdown table of released models' training budgets: Llama 3 8B (about 15T), OLMo 2 (4T), SmolLM2 1.7B (about 11T), Qwen3 (about 36T), the reference model (3.15B); parameters, tokens, tokens per parameter; each row re-checked against its model card or paper at authoring, unverifiable rows dropped. -->

## The Lifetime Objective and Its Optimum

### The First-Order Condition

<!-- Placeholder (prose): the derivation. Lifetime FLOPs N(6D + 2 D_inf) at a target loss ℓ on the contour E + A N^-α + B D^-β = ℓ; with a = A N^-α and b = B D^-β the first-order condition is α a = β b (1 + D_inf/(3D)), which reduces to eq_chinchilla_optimum at D_inf = 0. The condition's left side falls and its right side rises in N, so the root is unique; it lies above N_min = (A/(ℓ − E))^(1/α), where b = 0 and the model needs infinitely many tokens, and N* approaches N_min as demand grows. The two functions below compute the cost and solve the condition by a one-dimensional root search in log N. -->

```{.python .input #budgets-solve-optimum}
%%tab pytorch
def lifetime_cost(N, D, D_inf):
    """Training plus inference FLOPs: 6N per training token, 2N per
    generated token."""
    return N * (6 * D + 2 * D_inf)

def contour_tokens(N, loss, E, A, B, alpha, beta):
    """The D at which N parameters reach the loss: L(N, D) = loss."""
    return (B / (loss - E - A * N ** -alpha)) ** (1 / beta)

def log_n_min(loss, E, A, B, alpha, beta):
    """log N_min: there a = A N^-α alone is loss - E, so b = 0 and the
    model needs infinitely many tokens."""
    return math.log(A / (loss - E)) / alpha

def solve_optimum(loss, D_inf, *fit):
    """(N*, D*) minimizing lifetime_cost on the contour L(N, D) = loss: the
    root in log N of α a = β b (1 + D_inf / 3D), a = A N^-α, b = loss-E-a."""
    E, A, B, alpha, beta = fit

    def foc(log_N):  # positive below the optimum, negative above it
        N = math.exp(log_N)
        a = A * N ** -alpha
        b = loss - E - a
        inv_D = (b / B) ** (1 / beta)  # 1/D on the contour: 0 at N_min
        return alpha * a - beta * b * (1 + D_inf * inv_D / 3)

    lo = log_n_min(loss, *fit)
    hi = lo + 1
    while foc(hi) > 0:  # foc falls in N: step up until it changes sign
        hi += 1
    N = math.exp(optimize.brentq(foc, lo + 1e-9, hi, xtol=1e-14, rtol=1e-14))
    return N, contour_tokens(N, loss, *fit)
```

<!-- Placeholder (prose): the checks: at no demand the root equals the closed form of eq_chinchilla_optimum (to rounding); N* falls with demand under both fits. The smallest case (Hoffmann's constants, ℓ = 2.0): N* falls from 15.3B to 11.5B to 6.5B at 0, 10^12 and 10^13 generated tokens while tokens per parameter rise from 79 to 143 to 600; Besiroglu's constants give the same direction at a different level. The fits' constants belong to other corpora and tokenizers: the table shows the structure, never a property of real models. -->

```{.python .input #budgets-optimum-checks}
%%tab pytorch
LOSS, DEMANDS = 2.0, (0, 1e12, 1e13)  # target loss (nats), generated tokens
print(f'{"fit":>9} {"D_inf":>6} {"N*":>7} {"D*":>7} {"D*/N*":>6} '
      f'{"lifetime FLOPs":>15}')
for name, fit in FITS.items():
    for D_inf in DEMANDS:
        N, D = solve_optimum(LOSS, D_inf, *fit)
        print(f'{name:>9} {D_inf:>6g} {N / 1e9:6.1f}B {D / 1e12:6.2f}T '
              f'{D / N:6.0f} {lifetime_cost(N, D, D_inf):15.2e}')
err = 0.0
for fit in FITS.values():  # no demand: the closed form, eq_chinchilla_optimum
    N0, D0 = solve_optimum(LOSS, 0, *fit)
    err = max(err, abs(N0 / train_optimum(6 * N0 * D0, *fit)[0] - 1))
grid = np.logspace(9, 15, 25)  # generated tokens
falls = all(np.all(np.diff([solve_optimum(LOSS, x, *fit)[0] for x in grid])
                   < 0) for fit in FITS.values())
print(f'closed form at D_inf = 0: relative error {err:.1e}; '
      f'N* decreasing in demand: {falls}')
assert err < 1e-8 and falls
```

### Cost against Model Size

<!-- Placeholder (prose): reading the plot: each curve is the lifetime cost of the models that reach ℓ = 2.0 (Hoffmann's fit), from just above N_min up; the dots are the optima of the four demand levels. The dotted curve is the break-even demand: the demand at which the 10^13-optimal model (smaller, trained longer) and the training-optimal model cost the same over their lifetimes (the two crosses); below it the extra training of the small model does not pay. -->

```{.python .input #budgets-lifetime-cost}
%%tab pytorch
hoff = FITS['Hoffmann']
N_min = math.exp(log_n_min(LOSS, *hoff))  # the smallest model reaching LOSS
Ns = N_min * np.logspace(0.25, 1.1, 200)  # 1.8 to 12.6 times N_min
Ds = contour_tokens(Ns, LOSS, *hoff)  # their tokens on the contour
levels = (0, 1e11, 1e12, 1e13)  # generated tokens
optima = [solve_optimum(LOSS, x, *hoff) for x in levels]
(N_c, D_c), (N_s, D_s) = optima[0], optima[-1]  # training-optimal, smallest
D_even = 3 * (N_s * D_s - N_c * D_c) / (N_c - N_s)  # equal lifetime cost
costs = {x: lifetime_cost(Ns, Ds, x) for x in (*levels, D_even)}
dots = [(n, lifetime_cost(n, d, x)) for (n, d), x in zip(optima, levels)]
crosses = [(n, lifetime_cost(n, d, D_even)) for n, d in (optima[0],
                                                          optima[-1])]
d2l.plot_lifetime_cost(Ns, costs, dots, crosses)
```

### How Far Apart the Two Fits Are

<!-- Placeholder (prose): the optimal tokens per parameter against demand under both fits, and the band between them. Under both the lifetime-optimal ratio exceeds the training-optimal one and grows with demand (claim 4); the level differs. Where the band is wider than what a tenfold change in demand does to either curve, the fits disagree by more than the demand question can resolve: no single ratio is named, and a comparison that flips inside the band is "undecided". The printed span starts at the grid's first point (10^9) because the disagreement holds at every lower demand too (at no demand the ratios are 79 and 19, while a tenfold change in a small demand moves neither curve): quote it as "every demand up to ...", never with a lower limit. -->

```{.python .input #budgets-ratio-band}
%%tab pytorch
demand = np.logspace(9, 14, 41)  # generated tokens, 8 points per decade
ratio = {k: np.array([D / N for N, D in (solve_optimum(LOSS, x, *fit)
                                         for x in demand)])
         for k, fit in FITS.items()}
assert all(np.all(np.diff(r) > 0) for r in ratio.values())  # claim 4
gap = np.abs(np.log(ratio['Hoffmann'] / ratio['Besiroglu']))[:-8]
tenfold = np.max([np.abs(np.log(r[8:] / r[:-8])) for r in ratio.values()], 0)
wide = demand[:-8][gap > tenfold]
if len(wide):  # below the grid a 10x change in demand moves neither curve
    start = 'up to' if wide[0] == demand[0] else (
        f'from {d2l.fmt_power(wide[0])} to')
    print('the fits differ by more than a 10× change in demand moves them:'
          f' every D_inf {start} {d2l.fmt_power(wide[-1])}')
else:
    print(f'the fits differ by at most {np.exp(gap.max()):.1f}× (D_inf = '
          f'{d2l.fmt_power(demand[gap.argmax()])}), less than a 10× change '
          'in demand moves them')
d2l.plot_ratio_band(demand, ratio)
```

## Measured Inference Cost

### A Decode Roofline

<!-- Placeholder (prose): parameters are not serving cost. A decode step at batch b reads every weight once and every sequence's key-value cache, and does 2N FLOPs per token, so its rate is the batch over the larger of the memory time (N w + b c k)/W and the compute time 2 N b / P (W the memory bandwidth, P the peak, c the context, k the cache bytes per token). At small batch the memory time wins: decode is memory-bound. The RTX 4090's numbers are NVIDIA's specification (1,008 GB/s; the dense bf16 peak with fp32 accumulation); cite the MLSS efficiency tutorial's table for the comparison with data-centre parts. The reference model's 18,432 bytes per cached token come from its configuration (2 L H_kv d_h, keys and values, 2 bytes each). -->

```{.python .input #budgets-decode-roofline}
%%tab pytorch
BW = 1008e9  # RTX 4090 memory bandwidth, bytes/s (NVIDIA's specification)
PEAK = d2l.GPU_PEAK_BF16['NVIDIA GeForce RTX 4090']  # dense bf16 FLOP/s
MODELS = ('r2', 'Qwen/Qwen3-0.6B', 'Qwen/Qwen3-1.7B', 'Qwen/Qwen3-4B')
CTX = 1024  # tokens already in each sequence's cache

def decode_roofline(N, w, kv_bytes, ctx, batch, bw=BW, peak=PEAK):
    """Tokens per second of a decode step: the larger of the memory time
    (weights once, every sequence's cache) and the compute time."""
    t_mem = (N * w + batch * ctx * kv_bytes) / bw
    t_cmp = 2 * N * batch / peak
    return batch / np.maximum(t_mem, t_cmp)

arch = cfg.arch
kv_r2 = 2 * arch.L * arch.H_kv * arch.d_h * 2  # keys and values, 2 bytes each
shapes = d2l.decode_shapes(MODELS)  # parameters and cache bytes per token
assert shapes['kv_bytes'][0] == kv_r2
print(f'KV cache per token: {kv_r2:,} bytes (reference model)')
roof = decode_roofline(shapes['params'], 2, shapes['kv_bytes'], CTX, 1)
print('roofline at batch 1, context 1,024 (tok/s): ' + ', '.join(
    f'{m.split("/")[-1]} {r:,.0f}' for m, r in zip(MODELS, roof)))
```

### Measured Decode Rates

<!-- Placeholder (prose): the measured rates on one RTX 4090 against the roofline. At batch 1 the measured rate lands within 15% of the roofline (claim 6; a larger deviation is reported with the stack used: CUDA-graph replay of eager kernels, re-measured when the serving chapters' FastGenerator lands); at large batch the cache term and the compute time take over and the deviations grow (reported, not hidden). The roofline the rates are compared with counts the bytes a step of the harness reads: the whole cache buffer of 1,024 + 128 slots, the unwritten ones masked (a static step, so that it can be replayed as a CUDA graph); the batch-1 ratio moves by under 1% against a 1,024-slot roofline, the batch-128 one by about 12%. The Hugging Face eager rate at batch 1 is recorded with the table for reference (several times slower; the harness must beat it). Peak memory decides feasibility: the 4B model at batch 128 does not fit. -->

```{.python .input #budgets-decode-rates}
%%tab pytorch
NEW_TOKENS = 128  # decode steps per timed repeat
rates = d2l.decode_rates(MODELS, batches=(1, 8, 32, 128), ctx=CTX,
                         new_tokens=NEW_TOKENS)
# A step reads the whole cache buffer of CTX + NEW_TOKENS slots
rates['roofline'] = decode_roofline(rates['params'], 2, rates['kv_bytes'],
                                    CTX + NEW_TOKENS, rates['batch'])
d2l.show_decode_table(rates)
d2l.plot_decode(rates)
```

### The Optimum in GPU-Seconds

<!-- Placeholder (prose): counted FLOPs price a decode token at a third of a training token; measured, a decode token at serving batch costs more GPU time per FLOP, because decode is memory-bound and training runs at the reference run's MFU. GPU-seconds per training token come from the reference run's measured throughput, scaled in proportion to N (a fixed MFU); GPU-seconds per generated token from the batch-32 roofline of an N-parameter model (the largest measured model's cache bytes per token) times η, the largest model's measured share of its roofline (η = 1 while the rates are pending). The optima lie beyond the largest measured model (4B), where decode at batch 32 is weight-bound; a power law fitted through the four measured rates would extrapolate the small models' cache-dominated slope and under-price decode there by about a quarter. Re-solving on the same contour moves the optimum further toward small models (claim 7), or, if the ratio stays within 5% of 1, the cell says no further shift at these demand levels. If the training log is a stand-in (the t1 run) or pending, the label says so. -->

```{.python .input #budgets-measured-optimum}
%%tab pytorch
log = d2l.reference_run_log()  # r2's training log, or its labelled stand-in
train = d2l.training_rate(log)  # GPU-seconds per token at train.N
big = rates[rates['batch'] == 32].iloc[-1]  # the largest model, batch 32
measured = not math.isnan(big['tok_s'])
eta = big['tok_s'] / big['roofline'] if measured else 1.0  # its efficiency

def decode_seconds(N):  # GPU-seconds per generated token at batch 32
    return 1 / (eta * decode_roofline(N, 2, big['kv_bytes'],
                                      CTX + NEW_TOKENS, 32))

def gpu_seconds(N, D, D_inf):  # lifetime GPU-seconds on one RTX 4090
    return D * train.seconds * N / train.N + D_inf * decode_seconds(N)

def measured_optimum(loss, D_inf, *fit):
    """N* minimizing gpu_seconds on the contour, searched in log N."""
    def log_cost(log_N):
        N = math.exp(log_N)
        return math.log(gpu_seconds(N, contour_tokens(N, loss, *fit), D_inf))
    lo = log_n_min(loss, *fit)
    return math.exp(optimize.minimize_scalar(
        log_cost, bounds=(lo + 1e-6, lo + 10), method='bounded',
        options=dict(xatol=1e-9)).x)

print(f'{"D_inf":>6} {"N* (FLOPs)":>11} {"N* (GPU-s)":>11} {"ratio":>6}')
for D_inf in (1e11, 1e12, 1e13):
    N_f = solve_optimum(LOSS, D_inf, *hoff)[0]
    N_t = measured_optimum(LOSS, D_inf, *hoff)
    print(f'{d2l.fmt_power(D_inf):>6} {N_f / 1e9:10.1f}B {N_t / 1e9:10.1f}B '
          f'{N_t / N_f:6.2f}')
print(f'training at MFU {train.mfu:.2f}; decode at batch 32: roofline × '
      + (f'{eta:.2f} ({big["model"].split("/")[-1]} measured)' if measured
         else '1 (rates pending)'))
```

## Repeated Data

<!-- Placeholder (prose): restate in one sentence the effective-tokens form of sec_lm_data_mixtures (D′ = U + U R*(1 − e^(−R/R*)) over R repeats of U unique tokens; R* = 15.4 from the data-constrained scaling paper), not re-derived. A repeated epoch is worth dD′/dE = e^(−(E−1)/R*) of a fresh one at the margin (the cell prints this derivative: 0.82 at the 4th epoch, 0.38 at the 16th, 0.08 at the 40th): the 4th still most of one, the 40th little; the effective tokens approach U(1 + R*), so compute spent on the fortieth repeat buys little. Substituting D′ for D in the lifetime objective is an exercise. -->

```{.python .input #budgets-repeated-data}
%%tab pytorch
U, R_star = 1.0, 15.4  # unique tokens (one unit); Muennighoff et al.'s R*
def eff(epochs):  # effective tokens D' of U unique tokens repeated
    return d2l.effective_tokens(U, epochs, r_star=R_star)

print(f'effective tokens at one epoch equal U: {eff(1.0) == U}')
epoch, h = np.array([4.0, 16.0, 40.0]), 1e-5  # epochs, step
fd = (eff(epoch + h) - eff(epoch - h)) / (2 * h)
marginal = U * np.exp(-(epoch - 1) / R_star)  # dD'/dE above one epoch
print(f'derivative vs finite differences: {np.abs(fd - marginal).max():.1e}')
print("marginal value dD'/dE at epoch 4 / 16 / 40: " + ' / '.join(
    f'{m:.2f}' for m in marginal) + ' of a fresh epoch')
epochs = np.linspace(1, 60, 237)
d2l.plot(epochs, [epochs * U, eff(epochs), np.full_like(epochs,
                                                       U * (1 + R_star))],
         'epochs over U unique tokens', "effective tokens D' / U",
         legend=['fresh data', 'repeated', 'limit U(1 + R*)'],
         fmts=('k:', '-', 'k--'), ylim=(0, 25), figsize=(4.5, 3))
```

## Fitting the Law on the Ladder

### The Ladder's Points

<!-- Placeholder (prose): the points: 21.2's ladder, four sizes (three ladder models and the reference model) each finished at three budgets (about 5N, 10N and 20N; the reference model's 5.7N, 11.4N and 22.7N), every point a fully decayed run, read at its validation loss on the same held-out FineWeb-Edu tokens. Twelve points against five parameters: seven degrees of freedom. The plateau points (the branches' undecayed starting states) enter only a sensitivity check below; a plateau loss is not a finished model's loss (the Kaplan-schedule pitfall of sec_scaling-laws). While the ladder campaign is pending, the cells fit a simulated ladder at the same twelve (N, D) with Hoffmann's losses plus seed noise, labelled: the procedure and its intervals are shown, no exponent is claimed. -->

<!-- campaign: lm-ladder-logs-v1 -->

```{.python .input #budgets-ladder-points}
%%tab pytorch
pts = d2l.ladder_points()  # 21.2's ladder: finished and plateau points
fin = pts[pts['finished']].reset_index(drop=True)
print(f'{"size":>11} {"N":>6} {"tokens":>7} {"D/N":>5} {"val loss":>8}')
for r in fin.itertuples():
    print(f'{r.size:>11} {r.N / 1e6:5.0f}M {r.tokens / 1e9:6.2f}B '
          f'{r.tokens_per_param:5.1f} {r.val_loss:8.3f}')
```

### Fitting the Chinchilla Form

<!-- Placeholder (prose): the Hoffmann procedure, restated: fit log L = LSE(e, a − α log N, b − β log D), with E = e^e, A = e^a, B = e^b, by minimizing a Huber loss (δ = 10^-3) of the residuals in log space with L-BFGS from a grid of starting points; the minimum over starts is kept. The Huber loss keeps one bad point from steering the fit; the grid guards against the flat directions of a five-parameter fit on twelve points. Two practical details (measured at authoring on simulated ladders): the summed loss is about 10^-5, so L-BFGS-B's default tolerances stop after two to nine iterations, close to wherever it started; on the flat ridge along which E, A and α trade off, that leaves a refit near its starting point and makes every interval built from refits far too narrow (a warm start from the true exponents "recovered" them on every noise draw, while converged fits spread α from 0.23 to 0.42). With tight tolerances, refining only the 8 grid points of lowest loss reaches the same minimum as refining 200 of them. Later refits start from the main fit and seven perturbations of it, and refine the best two. The residual RMS (in nats, over seven degrees of freedom) is the ladder's noise level, which the next cells use. While the ladder campaign is pending every fit cell prints the simulated ladder's label first. -->

```{.python .input #budgets-fitting-the-chinchilla-form-1}
%%tab pytorch
def law(N, D, E, A, B, alpha, beta):  # eq_chinchilla_law
    return E + A / N ** alpha + B / D ** beta

def huber_objective(p, logN, logD, logL, delta=1e-3):
    """Huber loss of the residuals LSE(e, a − α log N, b − β log D) − log L,
    and its gradient in p = (e, a, b, α, β)."""
    e, a, b, alpha, beta = p
    terms = np.stack([np.full_like(logN, e), a - alpha * logN,
                      b - beta * logD])
    lse = np.logaddexp.reduce(terms, axis=0)  # log of the law's loss
    r = lse - logL
    g = np.exp(terms - lse) * np.clip(r, -delta, delta)  # softmax × ψ(r)
    grad = np.r_[g.sum(1), -g[1] @ logN, -g[2] @ logD]
    return special.huber(delta, r).sum(), grad
```

```{.python .input #budgets-fit-law}
%%tab pytorch
BOUNDS = [(-2, 3), (-5, 40), (-5, 40), (0, 3), (0, 3)]  # e, a, b, α, β
HOFFMANN_STARTS = np.stack(np.meshgrid(  # e, a, b, α, β: their grid
    np.arange(-1, 1.5, 0.5), np.arange(0, 30, 5), np.arange(0, 30, 5),
    np.arange(0, 2.5, 0.5), np.arange(0, 2.5, 0.5)), -1).reshape(-1, 5)
TOL = dict(ftol=1e-15, gtol=1e-14, maxiter=5000)  # the loss is about 1e-5

def fit_law(N, D, L, init=None, seed=0):
    """(E, A, B, α, β) minimizing huber_objective within BOUNDS (which keep
    exp() finite when a ladder cannot pin an exponent), refined from the 8
    best of HOFFMANN_STARTS, or from the 2 best of init and 7 perturbations
    of it."""
    data = np.log(N), np.log(D), np.log(L)
    if init is None:
        starts, refine = HOFFMANN_STARTS, 8
    else:
        p = np.r_[np.log(init[:3]), init[3:]]
        noise = np.random.default_rng(seed).standard_normal((7, 5))
        starts = np.clip(p + np.r_[np.zeros((1, 5)), 0.1 * noise],
                         *np.array(BOUNDS).T)
        refine = 2
    best = np.argsort([huber_objective(s, *data)[0] for s in starts])
    fit = min((optimize.minimize(huber_objective, starts[i], data, jac=True,
                                 method='L-BFGS-B', bounds=BOUNDS,
                                 options=TOL) for i in best[:refine]),
              key=lambda res: res.fun)
    return (*np.exp(fit.x[:3]), *fit.x[3:])
```

```{.python .input #budgets-fitting-the-chinchilla-form-2}
%%tab pytorch
d2l.show_ladder_label(fin)  # a stand-in ladder's numbers carry its label
N, D, L = (fin[c].to_numpy(float) for c in ('N', 'tokens', 'val_loss'))
params = fit_law(N, D, L)
dof = len(L) - len(params)
rms = math.sqrt(np.sum((L - law(N, D, *params)) ** 2) / dof)
print('E = {:.2f}, A = {:.3g}, B = {:.3g}, α = {:.2f}, β = {:.2f}'.format(
    *params))
print(f'residual RMS: {rms:.4f} nats ({dof} degrees of freedom)')
```

### Does the Procedure Recover the Exponents?

<!-- Placeholder (prose): before reading the exponents, the procedure is tested at the ladder's own noise: 200 simulated ladders at the same twelve (N, D), losses from the fit plus Gaussian noise of the residual RMS, each refitted. If the 90% range of the recovered α or β spans more than 0.3, the cell says the ladder cannot pin that exponent (claim 9). -->

```{.python .input #budgets-recovery}
%%tab pytorch
d2l.show_ladder_label(fin)
rng = np.random.default_rng(0)
sims = np.array([fit_law(N, D, law(N, D, *params) + rng.normal(
    0, rms, len(L)), init=params)[3:] for _ in range(200)])
for j, name in enumerate(('α', 'β')):
    lo, med, hi = np.percentile(sims[:, j], [5, 50, 95])
    pinned = '' if hi - lo <= 0.3 else '; the ladder cannot pin this exponent'
    print(f'recovered {name}: median {med:.2f} (true {params[3 + j]:.2f}), '
          f'90% range [{lo:.2f}, {hi:.2f}]{pinned}')
```

### Intervals for the Exponents

<!-- Placeholder (prose): a bootstrap over points with the sizes kept: each resample draws, within every size, three of its budgets with replacement, and is refitted. The 95% percentile intervals of α and β are set beside the published values. Agreement means a published value inside the interval; disagreement is stated as the ladder's limit, never as "confirms Chinchilla" (claim 10). On the simulated ladder (generated at Hoffmann's constants) containment would be circular, and the cell says "no verdict". Assignment A5 (sec_lm_a5) takes its tolerance on the exponents from this interval. -->

```{.python .input #budgets-intervals}
%%tab pytorch
simulated = d2l.show_ladder_label(fin)
RESAMPLES = 1000  # 500 if cells 12-17 exceed 90 s on the committed ladder
rng = np.random.default_rng(1)
by_size = [np.flatnonzero(fin['size'] == s) for s in fin['size'].unique()]
draws = [np.concatenate([rng.choice(i, len(i)) for i in by_size])
         for _ in range(RESAMPLES)]  # budgets resampled within each size
boot = np.array([fit_law(N[i], D[i], L[i], init=params) for i in draws])
inside = {}
print(f'95% intervals from {RESAMPLES:,} resamples')
for j, name in ((3, 'α'), (4, 'β')):
    lo, hi = np.percentile(boot[:, j], [2.5, 97.5])
    inside[name] = [k for k, fit in FITS.items() if lo <= fit[j] <= hi]
    print(f'{name}: {params[j]:.2f} [{lo:.2f}, {hi:.2f}]; ' + ', '.join(
        f'{k} {fit[j]:g}' for k, fit in FITS.items()))
count = {2: 'both fits', 1: 'one fit', 0: 'neither fit'}
if simulated:  # generated from Hoffmann's constants: containment is circular
    print("simulated ladder: generated at Hoffmann's α and β, so no verdict")
else:
    print('published value inside the 95% interval: ' + '; '.join(
        f'{n} {count[len(k)]}' + (f' ({k[0]})' if len(k) == 1 else '')
        for n, k in inside.items()))
```

### A Withheld Point

<!-- Placeholder (prose): the most demanding check of a fit is a point it has not seen: the reference model at 22.7 tokens per parameter, the largest model's longest run, is withheld, the law refitted on the other eleven and its loss predicted. The refit is a full search (no start informed by the withheld point). The interval comes from the bootstrap resamples that did not draw the point, plus the residual noise. If the measured loss falls outside, the cell's numbers say by how much, and the prose says the law extrapolates poorly from twelve points (claim 11). -->

```{.python .input #budgets-withheld}
%%tab pytorch
d2l.show_ladder_label(fin)
rng = np.random.default_rng(2)
w = int(np.lexsort((D, N))[-1])  # the largest model's largest budget
keep = np.arange(len(L)) != w
p_w = fit_law(N[keep], D[keep], L[keep])  # a full search: no start saw w
pred = law(N[w], D[w], *p_w)
out = [law(N[w], D[w], *p) + rng.normal(0, rms)
       for p, i in zip(boot, draws) if w not in i]
lo, hi = np.percentile(out, [2.5, 97.5])
print(f'predicted {pred:.3f} [{lo:.3f}, {hi:.3f}], measured {L[w]:.3f} '
      f'({len(out)} resamples without it)')
d2l.plot_ladder_fit(fin, lambda n, d: law(n, d, *p_w), w, (pred, lo, hi))
```

### Plateau Points

<!-- Placeholder (prose): the sensitivity check of the schedule pitfall: the plateau points added to the fit. The shift of the exponents is read against the bootstrap spread of the finished-point fit; within the spread, the cell says so (claim 13). -->

```{.python .input #budgets-plateau}
%%tab pytorch
simulated = d2l.show_ladder_label(pts)
plateau = (~pts['finished']).sum()
if simulated:
    print('simulated ladder: no plateau points to add')
else:
    p_all = fit_law(*(pts[c].to_numpy(float) for c in (
        'N', 'tokens', 'val_loss')), init=params)
    shift, sd = np.subtract(p_all[3:], params[3:]), boot[:, 3:].std(0)
    print(f'with {plateau} plateau points: α = {p_all[3]:.2f}, '
          f'β = {p_all[4]:.2f}')
    print(f'shift vs the finished-point fit: Δα = {shift[0]:+.2f} '
          f'({shift[0] / sd[0]:+.1f} bootstrap SDs), Δβ = {shift[1]:+.2f} '
          f'({shift[1] / sd[1]:+.1f} SDs)')
```

### The Ladder's Own Lifetime Answer

<!-- Placeholder (prose): the ladder's fit, solved at the loss its largest finished point reaches, for no demand and for 10^11 and 10^12 generated tokens, with the 95% bootstrap range (the fits that reach that loss with a finite optimum: E below it, both exponents positive, N_min below e^50; the cell counts the others when there are any). What a ladder of this size can and cannot say (claim 12): the direction of the shift and the ordering of the demand levels are resolved where their bootstrap ranges exclude the alternative; the level of the optimal ratio is not. Under a pending ladder campaign this paragraph describes the simulated ladder's interval widths, labelled simulated. -->

<!-- campaign: lm-ladder-logs-v1 -->

```{.python .input #budgets-ladder-optimum}
%%tab pytorch
d2l.show_ladder_label(fin)
target = L[w]  # the loss the ladder reaches: its largest finished point
MAX_LOG = 50  # usable: N_min and D_min between 1 and e^50

def reaches(loss, E, A, B, alpha, beta):  # D_min: N_min, roles swapped
    return (E < loss and min(alpha, beta) > 0
            and 0 < log_n_min(loss, E, A, B, alpha, beta) < MAX_LOG
            and 0 < log_n_min(loss, E, B, A, beta, alpha) < MAX_LOG)

usable = [p for p in boot if reaches(target, *p)]
print(f'{"D_inf":>6} {"N*":>5} {"95% range":>12} {"D*/N*":>6} '
      f'{"95% range":>12}')
for D_inf in (0, 1e11, 1e12):
    N_o, D_o = solve_optimum(target, D_inf, *params)
    sol = np.array([solve_optimum(target, D_inf, *p) for p in usable])
    n_lo, n_hi = np.percentile(sol[:, 0], [2.5, 97.5]) / 1e6
    r_lo, r_hi = np.percentile(sol[:, 1] / sol[:, 0], [2.5, 97.5])
    n_rng = f'[{n_lo:.0f}, {n_hi:.0f}]M'
    r_rng = f'[{d2l.fmt_sig(r_lo)}, {d2l.fmt_sig(r_hi)}]'
    print(f'{d2l.fmt_power(D_inf):>6} {N_o / 1e6:4.0f}M {n_rng:>12} '
          f'{d2l.fmt_sig(D_o / N_o):>6} {r_rng:>12}')
if len(usable) < len(boot):
    print(f'({len(usable)} of {len(boot)} resamples reach the target loss)')
```

<!-- Placeholder (prose): closing: the lifetime answer assumes the decode rates of one RTX 4090; what those rates depend on (memory bandwidth, interconnect, precision) is the subject of chap_lm_hardware. Smaller models may need longer contexts or more reasoning tokens, which raises D_inf (one sentence). -->

## Summary

<!-- Placeholder (prose): summary, 3 to 6 sentences: the lifetime objective and the direction of its optimum; measured decode cost moves it further; repeated data loses value with each epoch; a twelve-point ladder resolves the direction and the ordering, not the level of the optimal ratio. -->

## Exercises

<!-- Placeholder (prose): 7 exercises (spec H): derive the optimum under repeated data (substitute D′ for D); the break-even demand at which overtraining pays; the Llama 3 8B exercise of sec_scaling-laws under the lifetime objective (CPU); exponent sensitivity to removing one ladder point, leave-one-out (CPU); the optimum's sensitivity to the MFU and the decode-rate bands (CPU); open: a mixture-of-experts row (active against total parameters) and why the dense law does not apply; fit the miniature grid printed in sec_scaling-laws with fit_law and say why its exponents are not comparable with the ladder's (tokenizer, loss unit, unfinished points, two-decimal values). -->
