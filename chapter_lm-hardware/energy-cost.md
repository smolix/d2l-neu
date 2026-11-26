# Energy, Power and Cost
:label:`sec_lm_energy_cost`

<!-- Author brief, copied verbatim from the Language Models part plan
(docs/llm/language_models_part_plan.md, "5.8 Energy, Power and Cost · T1 (T0 calculator; optional T2 clock sweep)").
Blueprint: docs/llm/blueprints/ch05_hardware.md section 5.6.
The plan is a specification, not book text; this comment is replaced
when the section is written.

**Idea:** A token's price is the chip-hour price divided by tokens per chip-hour and its energy is the chip's power divided by the same rate, so both are set by utilization before they are set by anything else.
**Opening:** The reader has the energy ladder of `subsec_hw-energy`, picojoules per operation and per byte, and the throughput measurements of 5.3. One concrete example: the reader's GPU draws a steady idle power and about three times that at a compute-bound batch, while its decode throughput between batch 1 and batch 64 rises fifty-fold (to be measured), so the joules per token at batch 1 are an order of magnitude above those at batch 64. The open question: what a million tokens cost in joules and in dollars for training and for serving, what a power cap or a lower clock does to each, and when owning a chip beats renting one.
**Teaches:**
- Power of a chip: static against dynamic, TDP against drawn, clocks against voltage; power capping and frequency scaling: a compute-bound step loses throughput with the clock while a bandwidth-bound step does not, so one setting for both workloads is the usual error.
- Energy per token: training at 6N FLOPs per token at the measured MFU, decode across the batch; rack and datacentre overheads (PUE); the power swing of synchronous training (tens of megawatts in a frontier run, from the Llama 3 report) and the remedies in one paragraph each: GPU-level smoothing, filler work, rack-level energy storage.
- Cost: price per chip-hour times chip-hours per million tokens; training cost as 6ND in dollars at an MFU and a goodput (restating `eq_six_nd` and the "GPU-hours before dollars" exercise of `sec_cloud_instances`); serving cost at an interactivity point on 5.3's frontier, prefill billed separately with a cache hit rate; owned against rented with a break-even utilization; carbon as energy times grid intensity in one paragraph without numbers in prose.
- Energy bloat: stages of a pipeline or ranks of a data-parallel group that finish early can run slower at lower power without lengthening the step (one paragraph; the critical path it relies on is 6.8's).
**Derivation:** The energy-optimal clock. If compute-bound throughput is proportional to the clock f and power is P(f) = P_s + k f^3, throughput per watt f / (P_s + k f^3) peaks at f* = (P_s / (2k))^(1/3), where P(f*) = 1.5 P_s; a bandwidth-bound step's throughput is independent of f, so its energy per token falls with the clock until the clock itself bounds the memory pipeline (blueprint ch05 5.6 D3). Smallest numeric case: P_s = 80 W and P(1.4 GHz) = 300 W give k = 80 W per GHz^3, f* = 0.79 GHz, P(f*) = 120 W, and tokens per joule at f* are 1.4 times those at the top clock, at 0.57 times the throughput. Cost per million tokens for training, C = 1e6 c_s F_tok / (MFU g P), is stated and checked against the book's exercise exactly (blueprint 5.6 D2); the owned-against-rented break-even u* = h_own / r is stated with its sensitivity (D5).
**Notebook:**
1. Question: how do joules and dollars per token vary with batch and clock for a training step and for decode on one GPU, and does the cubic power model predict the energy-optimal clock?
2. Model and data: the reference model (training step) and Qwen3-0.6B (decode at batches 1 to 128) on the reader's GPU; Qwen3-8B on the A100 as the T2 twin from 5.3's harness.
3. Power traces at 10 Hz through `d2l.BenchLog.power_trace()` (pynvml: power draw, SM and memory clocks, temperature, throttle reasons) during 5.3's timed windows; energy per token as the integral over the window divided by tokens; plot against batch for both workloads.
4. The clock sweep: `nvidia-smi -lgc` over the supported range if the host permits it (optional T2 log, produced once per release; the T1 cell tries, and on failure loads the committed A100 log and says so); fit (P_s, k) by least squares on the compute-bound series; predicted f* against the measured energy minimum.
5. The cost calculator: `d2l.cost_per_mtok_train` with MFU from 6.7's `lm-mfu-table` (fallback: 5.3's single-GPU MFU, labelled an upper bound) and prices from `lm-hw-prices` (dated quotes, `kind` quote or assumption); `d2l.cost_per_mtok_serve` at three points on the frontier; `d2l.owned_vs_rented` with a tornado over plus or minus 30% of each input and a sampled interval for u*.
6. Claims that survive noise: energy per decode token falls roughly as 1/b up to the compute-bound batch and then flattens, because power rises by less than a factor of two while throughput rises by more than an order of magnitude; the predicted f* lies within one sweep step of the measured minimum, or the text says the model extrapolates; a power cap costs the training step throughput and costs small-batch decode nothing within noise; no dollar figure is quoted in prose.
**Figures:** `fig_lm_hw_power_model` (static plus cubic dynamic power against clock, throughput per watt and f*); `fig_lm_hw_energy_per_token` (plot: joules per token against batch for training and decode, with power overlaid); `fig_lm_hw_cost_tornado` (plot: break-even utilization sensitivity).
**Restate / forward:** `subsec_hw-energy` and `fig_energy_ladder` (picojoules per operation and per byte; this section measures the whole chip); `eq_hw_decode_bound` (bandwidth-bound decode); `eq_six_nd`; `tab_cloud_prices` and the exercises of `sec_cloud_instances` (prices and rental practice stay there; the appendix is linked); `sec_hardware_buyers` (desktop and unified-memory buying, linked); 5.3's frontier. Forward: 6.7 (`lm-mfu-table`).
**Exercises:** 6 to 8. Derive the energy per token of an SRAM-only design that holds Qwen3-8B on N_c chips at their rated power (derive); derive the electricity price at which a GPU's lifetime energy equals its purchase price (derive); measure power at the compute-bound batch against idle and the ratio to TDP (measure); measure the step-level power swing of a synchronous training loop at 100 Hz and its amplitude (measure); check the units of the cost functions with a dimensional test (check); open: a filler-work smoother that bounds the swing below a threshold and the energy it costs.
**Writer notes:** This section hosts assignment A6a (`sec_lm_a6a`) at its end: the calibrated performance model of a chip, a node and a cluster plus the fused Triton kernel (see the chapter block). NVML power readings are averaged over windows that differ by generation (`power.draw` against `power.draw.instant`); state which field is read and at what rate, and sample at 10 Hz for energy and 100 Hz for the swing exercise. Locking clocks and setting power limits needs root; the notebook never fails on a permission error, it loads the committed A100 sweep and labels it. Shared hosts: record other processes and discard windows with a second GPU user. Datasheet TDP is not what a run draws; clocks fall under a cap (5.1's pitfall returns here with a measurement). The power-swing paragraph cites the Llama 3 report's statement on diurnal and step-level fluctuations and the 2025 power-stabilization paper for the remedies; quote directions, not megawatt numbers. Perseus and Zeus are the sources for energy bloat and for the measurement method; keep the critical-path argument to one paragraph (6.8 owns the trace). Every price is a dated row in `lm-hw-prices` with a source URL and a `kind`; the notebook prints the age of each quote and warns above 90 days; PUE and electricity prices are assumption rows; a conclusion that rests on one assumption is not a conclusion (blueprint 5.6 pitfalls). Carbon: one paragraph, grid intensity as an input, no numbers in prose. Goodput enters the training cost as a factor g; this section defines it in one sentence as the fraction of wall time that advanced the model and uses a stated value; 6.10 measures it and writes `lm-par-goodput`, which this section's calculator reads when present. Exports: `d2l.energy_per_token`, `d2l.dvfs_fit`, `d2l.hw_clock_sweep`, `d2l.cost_per_mtok_train`, `d2l.cost_per_mtok_serve`, `d2l.owned_vs_rented`; artifacts `lm-hw-prices`, `lm-hw-a100-clocksweep`, `lm-hw-buildbox-power`. Not re-taught: the energy ladder, the 6ND cost in GPU-hours (the exercise is recomputed, not re-explained), rental tiers and prices (`sec_cloud_instances`). Detail: blueprint ch05 5.6 (D2, D3, D5, `prices.csv` schema, risks). The swing measurement is an exercise on a T1 GPU, not a cell (a single GPU's swing is small); the AMD and TPU rows get power numbers where vendor pages give rack kW, kept as rows with sources.
**Sources:**
- Papers: [Reducing Energy Bloat in Large Model Training](https://arxiv.org/abs/2312.06902) (Perseus: slowing non-critical stages saves energy at no time cost); [Zeus: Understanding and Optimizing GPU Energy Consumption of DNN Training](https://arxiv.org/abs/2208.06102) (power limits and the energy-time trade-off, measurement method); [Power Stabilization for AI Training Datacenters](https://arxiv.org/abs/2508.14318) (the synchronous power swing and its remedies); [Carbon Emissions and Large Neural Network Training](https://arxiv.org/abs/2104.10350) (energy to carbon, the one-paragraph treatment); [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) (power fluctuations of a frontier run); [Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures](https://arxiv.org/abs/2505.09343) (cost-driven hardware choices).
- Code and documentation: [NVIDIA Management Library (NVML)](https://developer.nvidia.com/management-library-nvml) (power and clock readings); [DCGM](https://docs.nvidia.com/datacenter/dcgm/latest/index.html) (fleet-level power and clock telemetry, one sentence); [Zeus repository](https://github.com/ml-energy/zeus) (energy measurement library); [ML.ENERGY leaderboard](https://ml.energy/leaderboard) (published energy per token for released models).
- Courses, tutorials and blogs: [The Datacenter as a Computer](https://link.springer.com/book/10.1007/978-3-031-01761-2) (PUE, power provisioning, total cost of ownership); [Scaling Book: All About Transformer Inference](https://jax-ml.github.io/scaling-book/inference/) (cost per token along the frontier); [Efficiency in LLMs (MLSS)](https://alex.smola.org/posts/45-mlss-efficiency/) (lifecycle cost framing).
-->

<!-- Placeholder (prose): opening, on the box. The reader has the energy
ladder of subsec_hw-energy (picojoules per operation and per byte) and the
throughput of sec_lm_workload_hardware. One RTX 4090 of the build box draws
a steady idle power and several times that under a compute-bound matrix
product, where it sits on its 450 W limit (cell idle-and-peak, quote the
committed W at 2 digits); Qwen3-0.6B's decode throughput rises by more than
an order of magnitude from batch 1 to batch 64 while the power rises by
less than 2x (cell decode-energy), so a token at batch 1 costs an order of
magnitude more joules than at batch 64. The open question: what a million
tokens cost in joules and in dollars for training and for serving, what a
power cap or a lower clock does to each, and when owning a chip beats
renting one. Idea in one sentence: a token's price is the chip-hour price
divided by tokens per chip-hour and its energy is the chip's power divided
by the same rate, so both are set by utilization first. -->

```{.python .input #energy-cost-imports}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #energy-cost-extras}
%%tab pytorch
import collections
import pandas as pd
```

## The Power of a Chip
:label:`subsec_lm_hw_power`

<!-- Placeholder (prose): static against dynamic power (leakage, the clock
tree and memory refresh draw power at any load; switching draws C V^2 f per
transistor); TDP is a cooling budget and a limit, not what a run draws;
clocks and voltage move together on a DVFS curve, and the board's power
limiter lowers both when a load would pass the limit (5.1's pitfall returns
here with a measurement). Capping and frequency scaling: a compute-bound
step loses throughput with the clock, a bandwidth-bound step hardly does, so
one setting for both workloads is the usual error. What NVML reports:
`power.draw` is an average over the last second on Ampere-and-later
consumer and data-centre parts (except GA100), `power.draw.instant` the
instantaneous value; the library reads the averaged field at 10 Hz
(BenchLog.power_trace of sec_lm_accelerator_design) and warms each
workload up for 2 s first, so the average covers only the load it
measures. Windows with another user's process on the card are discarded
and repeated (rule of the chapter). -->

<!-- Figure: fig_lm_hw_power_model (generator power-model): static power
P_s plus cubic dynamic power k f^3 against the clock f, the limit line, the
throughput per watt f / P(f) on a second axis with its maximum f* marked. -->

<!-- Placeholder (prose): the three windows below, each 10 s of one
workload on a whole card after its warm-up: idle (the CUDA context open, no
kernels: the card then holds a high clock and draws well above its rest
power, as the table shows), decode of Qwen3-0.6B at batch 1 over a
512-token cache (CUDA graph), and a bf16 8,192^3 matrix product. Read the
table against the limit line under it: the matrix product sits at the
limit with the power-cap reason set, and the limiter has pulled its SM
clock below the clock the card holds under the light decode load (claim
E1: quote that drop in MHz, not NVML's maximum clock, which no load
reaches); decode at batch 1 draws a fraction of the power. Quote the
committed W at 2 digits and the reason names; the live line only shows
agreement. -->

```{.python .input #energy-cost-idle-and-peak}
%%tab pytorch
prof = d2l.power_profile(workloads=('idle', 'decode-b1', 'matmul-8192'),
                         seconds=10)
w = prof.set_index('workload')  # mean board power (W), median SM clock
peak, light = w.loc['matmul-8192'], w.loc['decode-b1']
assert peak.power_w > 0.9 * prof.attrs['power_limit_w']  # E1: at the limit
assert peak.thr_SW_power_cap >= 0.5  # E1: the power limiter acted
assert peak.sm_mhz < light.sm_mhz  # E1: clocked below the light load
assert light.power_w < 0.6 * peak.power_w  # E1: decode draws far less
```

### The Energy-Optimal Clock
:label:`subsec_lm_hw_dvfs`

<!-- Placeholder (prose): derivation D3 in full. If compute-bound
throughput is proportional to the clock f and power is P(f) = P_s + k f^3
(dynamic power C V^2 f with V roughly proportional to f), the work per
joule f / (P_s + k f^3) has its maximum where P_s + k f^3 = 3 k f^3, so
f* = (P_s / (2k))^(1/3) and P(f*) = 1.5 P_s. Numeric case (the plan's):
P_s = 80 W and P(1.4 GHz) = 300 W give k = 80 W/GHz^3, f* = 0.79 GHz,
P(f*) = 120 W, and 1.4x the work per joule at 0.57x the throughput of the
top clock (a worked example in the prose only; the cells below fit the
box's measured sweep). A bandwidth-bound step's
throughput does not depend on f, so its energy per token falls with the
clock until the clock itself bounds the memory pipeline. The function fits
(P_s, k) by least squares on the compute-bound series: P is linear in 1 and
f^3. -->

```{.python .input #energy-cost-dvfs-fit-def}
%%tab pytorch
#@save
DVFSFit = collections.namedtuple('DVFSFit', [
    'P_s', 'k', 'f_star', 'f_min', 'step', 'edge', 'locks', 'clocks',
    'power', 'rate', 'spread', 'source', 'a100'])

def dvfs_fit(log):  #@save
    """Fit P(f) = P_s + k f^3 (f in GHz: the clock each window ran at) by
    least squares to the locked-clock windows of the compute-bound
    workload; f* = (P_s / (2k))^(1/3) maximizes the work per joule. f_min
    is the lock of least energy per unit of work, step the sweep's largest
    step, edge whether f_min is the lowest lock, spread the largest
    relative spread of a window's power across processes (NaN: one run)."""
    s = log[(log.kind == 'clock') & log.workload.str.startswith('matmul')]
    s = s.sort_values('setting')
    f = s.sm_mhz.to_numpy() / 1e3  # the clock it ran at, GHz
    A = np.stack([np.ones_like(f), f**3], axis=1)  # P is linear in 1, f^3
    (P_s, k), *_ = np.linalg.lstsq(A, s.power_w.to_numpy(), rcond=None)
    lock = s.setting.to_numpy() / 1e3
    i = int(np.argmin(s.power_w.to_numpy() / s.rate.to_numpy()))  # J/FLOP
    spread = (log.attrs.get('spread') or {}).get('power_w')
    spread = (np.nanmax(np.asarray(spread, float)[s.index]) if spread
              else 0.0 if log.attrs.get('n_proc', 1) > 1 else math.nan)
    return DVFSFit(P_s, k, (P_s / (2 * k)) ** (1 / 3), lock[i],
                   np.diff(lock).max() if len(lock) > 1 else math.nan,
                   i == 0, lock, f, s.power_w.to_numpy(),
                   s.rate.to_numpy(), spread, log.attrs.get('source', ''),
                   log.attrs.get('a100'))
```

<!-- Placeholder (prose): how the sweep is measured, in a few sentences.
Locking a clock needs the driver's permission (sudo -n nvidia-smi -lgc f,f
inside the box-clocksweep regeneration, which holds the card's lock with no
other user's process on it and resets with -rgc on any exit), so no
notebook locks a clock: the cell reads the committed sweep (9 locks from
600 to 3,000 MHz plus the default clocks, three workloads, 10 s windows
after a warm-up, NVML at 10 Hz, three processes). The fit uses the clock
each window ran at: above about 2.5 GHz the product hits the 450 W limit
and runs below its lock (the third printed line names those locks). Read
the plot: power grows as the cube of the clock past a static floor; the
squares are the measured work per joule, the dashed line the model's, the
star the least energy. Claim E4: f* within one sweep step of the lock of
least energy per FLOP (quote both and the step from the printed line;
if the least is at the lowest lock, the job swept one step lower and the
text says "at or below" that lock). The A100 sweep (campaign C-T2) is
pending; the last line says so. -->

<!-- Placeholder (prose): the committed sweep (F4, 2026-11-26) misses E4 as
pre-registered: the cubic fit puts f* at 1.66 GHz, 1.5 steps below the
least energy per FLOP at the 2.10 GHz lock. Power rises about linearly with
the clock up to 2.1 GHz (roughly 35 W per 300 MHz: a voltage floor, where
P is close to P_0 + c f) and steeply above it, so the least energy sits at
that knee; the plot shows the cubic missing the points. Wording, and the
cell's E4 assert, await the coordinator and the author; state it as
measured, never widen the step. -->

```{.python .input #energy-cost-dvfs}
%%tab pytorch
sweep = d2l.hw_clock_sweep_log()  # committed: a notebook never locks clocks
fit = d2l.dvfs_fit(sweep)
d2l.plot_dvfs(fit)
near = abs(fit.f_star - fit.f_min) <= fit.step  # within one sweep step
assert near or (fit.edge and fit.f_star <= fit.f_min + fit.step)  # E4
```

<!-- Placeholder (prose): the sweep the regeneration runs, shown because it
is the method: lock the SM clock at each setting (minimum = maximum), run
each workload in a traced window, then restore the default clocks whatever
happens. hw_smi_card returns a guarded nvidia-smi: before every lock and at
each window's start and end it checks the card's compute processes, and a
watchdog checks them once a second while the clock is locked; another
process on the card resets the clock at once and the window is measured
again once the card is free (one sentence; the code is in the library).
Without the permission nvidia-smi refuses and the function raises one
PermissionError. Exercise: run it on your own card if your driver lets you
lock clocks. -->

```{.python .input #energy-cost-clock-sweep-def}
%%tab pytorch
def hw_clock_sweep(clocks, workloads, device=0):  #@save
    """Board power, clocks and work rate of each named workload with the SM
    clock locked at each of clocks (MHz) on visible GPU `device`; needs
    permission to lock clocks (sudo -n inside the regeneration)."""
    smi = d2l.hw_smi_card(device)  # nvidia-smi on that card, guarded
    if smi('-lgc', f'{clocks[0]},{clocks[0]}').refused:
        raise PermissionError('the driver refused to lock the SM clock')
    rows = []
    try:
        for f in clocks:
            smi('-lgc', f'{f},{f}')  # minimum = maximum: the clock is pinned
            rows += d2l._sweep_rows('clock', f, workloads, smi)
    finally:
        smi.restore('-rgc')  # back to the default clocks, or say it failed
    return pd.DataFrame(rows)
```

<!-- Placeholder (prose): what a lower clock costs (claim E5, restated on
clocks by the coordinator until the author rules; a power cap pulls the
same lever on a compute-bound kernel, and the -pl arm is off: ruling H1
covers clocks only). The table: each lock from the top down, the training
step's clock as run, its power and step time, the batch-1 decode step time,
each as a ratio x to the default-clock run with its noise (the larger of
5 x the larger median absolute deviation and the spread across the three
processes: the locks and the default clocks are not run back to back).
The training step slows about in proportion to
the clock; batch-1 decode, bound by memory and per-kernel fixed costs,
stays within its noise down to the clock the last line names (quote it and
the training step's ratio there from the output; if decode slows at every
lowered clock, the line says so and the text states it as measured; the
committed sweep (F4) does: decode at b = 1 slows about as much as the
training step, x 1.03 at 2,700 MHz and 2.33 at 600 MHz against 2.46 for
training: a lowered clock costs decode about what it costs training). One
setting for both workloads is the usual error. Zeus measures the same
energy-time trade-off per job. -->

```{.python .input #energy-cost-cap-effect}
%%tab pytorch
effect = d2l.clock_effect(sweep)  # step times against the default clocks
floor = effect.attrs['decode_floor']  # lowest lock: decode within noise
row = effect.set_index('clock MHz').loc[floor] if floor else None
assert row is None or row['train x'] - 1 > row['train noise']  # E5
```

## Energy per Token
:label:`subsec_lm_hw_energy_token`

<!-- Placeholder (prose): energy per token is power divided by tokens per
second, measured as the integral of a window's power trace over its
duration divided by the tokens produced in it (the function below; the
trace's samples are 0.1 s apart). The card also keeps a cumulative energy
counter (NVML's total energy consumption, in mJ, Volta and later); the
library reads it at both ends of every window, an independent measurement
the cells below check the trace against. -->

```{.python .input #energy-cost-energy-per-token-def}
%%tab pytorch
def energy_per_token(power_trace, tokens):  #@save
    """Joules per token of a traced window: the trapezoid integral of the
    samples' power (W) over their times (s), divided by the tokens
    produced over the trace's span."""
    t = np.array([s['t'] for s in power_trace])
    P = np.array([s['power_w'] for s in power_trace])
    return float(np.sum((P[1:] + P[:-1]) / 2 * np.diff(t))) / tokens
```

<!-- Placeholder (prose): decode across the batch. Qwen3-0.6B with random
weights (shapes and bytes are what matter) at b = 1 to 128 and contexts 512
and 2,048, each point a 5 s CUDA-graph window; points whose cache does not
fit 90% of the card are cut (b = 128 at L = 2,048: 30.1 GB). While the
weights dominate a step's bytes, b sequences share one read of them, so
J/token falls about as 1/b; once the caches dominate, every sequence pays
its own reads and the curve flattens. The two terms are equal at
b = W / (L k_KV) (sec_lm_workload_hardware's step model): about 20 at
L = 512 and 5 at L = 2,048 for Qwen3-0.6B (W = 1.19 GB, k_KV = 114,688
bytes per token). Decision Q9: the plan's "until the compute-bound batch"
does not apply, Qwen3-0.6B's L_crit is about 60 tokens, so its decode never
turns compute-bound at these contexts. Read the plot (log-log, dotted lines
the knees, dashed the power) and the lines: quote the J/token at b = 1 and
64, the power and throughput ratios, and the two slopes per context (claim
E2), at 2 digits from the committed run. -->

```{.python .input #energy-cost-decode-energy}
%%tab pytorch
specs = d2l.chip_specs()
chip = specs.loc['RTX 4090']
cfg_06b = d2l.hw_config('Qwen/Qwen3-0.6B')
e = d2l.decode_energy(cfg_06b, batches=2**np.arange(0, 8), L=(512, 2048))
sm = d2l.ServingModel(cfg_06b, chip)
for L in (512, 2048):
    knee = sm.weight_bytes() / (L * sm.kv_bytes_per_token())  # W = b L k_KV
    s = d2l.decode_slopes(e[e.L == L], knee)
    print(f'L = {L:,}: knee at b = {knee:.0f}; d log(J/token) / d log b '
          f'{s.below:.2f} below, {s.above:.2f} above')
    assert s.below < s.above  # E2: the curve flattens past the knee
g = e[e.L == 512].set_index('b')
assert g.power_w[64] / g.power_w[1] < 2  # E2: power less than doubles
assert g.tokens_per_s[64] / g.tokens_per_s[1] > 10  # E2: tokens/s x10+
assert np.allclose(e.j_per_token_counter, e.j_per_token, rtol=0.05)
```

<!-- Placeholder (prose): training at 6N FLOPs per token (plus attention,
counted at half for a causal mask) at the measured MFU: the reference
architecture of sec_lm_reference_model (138.4M parameters, random weights)
on one card, micro-batches of 2 sequences of 4,096 tokens, fp32 master
weights, bf16 autocast, fused AdamW and the chunked loss, eager; steps of
8,192 tokens (no accumulation) and 32,768 (four micro-batches). J/token is
power over tokens per second (claim E3): the card's energy counter, read
independently of the power trace, agrees with W / (tokens/s) to the
percentage the last line prints, which also checks the 1 s averaging of
power.draw; MFU sets the tokens per second. Quote the measured power
against the 450 W limit (the smoke run
drew about 355 W at MFU 0.36: this step stays below the limit). MFU here
halves the
causal attention term; chapter 21 prints MFU with the full term, so a
chapter-21 MFU quoted here is converted or labelled. Quote J/token and MFU
at 2 digits. -->

```{.python .input #energy-cost-train-energy}
%%tab pytorch
R = d2l.r2_config('r2').arch
tr = d2l.train_energy(R, tokens_per_step=(8192, 32768), micro_batch=2)
assert np.allclose(tr.j_per_token_counter, tr.power_w / tr.tokens_per_s,
                   rtol=0.05)  # E3: the counter's J/token is W / (tokens/s)
```

<!-- Placeholder (prose): from the chip to the building. A datacentre
delivers each watt to a chip at a PUE of about 1.5 (cooling, power
conversion; the price table's assumption row, sourced to Uptime's 2025
survey), so the energy bill is the chip's energy times the PUE. The power
swing of synchronous training: every rank computes, then every rank waits
for the collective, so a frontier run's draw swings by tens of megawatts in
step with the step (the Llama 3 report on diurnal and step-level
fluctuations); remedies, one sentence each, directions only: GPU-level
power smoothing (a floor on the draw), filler work during communication,
rack-level energy storage (the 2025 power-stabilization paper). Energy
bloat in one paragraph (Perseus): stages or ranks off the critical path
can run at a lower clock without lengthening the step; the critical path is
chapter 23's. Carbon in one paragraph, no numbers: emissions are energy
times the grid's carbon intensity, which varies by place and hour (Patterson
et al.). -->

## Cost per Token
:label:`subsec_lm_hw_cost`

<!-- Placeholder (prose): the price table: on-demand rents per GPU-hour from
at least two providers per data-centre chip, one marketplace rent of an
RTX 4090, purchase prices, electricity, PUE, operations and life. Each row
is a quote (with its page) or an assumption, dated; the table prints each
row's age and marks those older than 90 days. No dollar figure is quoted in
the prose; a conclusion that rests on one assumption is stated as
conditional on it. Prices and rental practice stay in sec_cloud_instances
(tab_cloud_prices), buying a desktop in sec_hardware_buyers: both linked,
not repeated. -->

```{.python .input #energy-cost-prices}
%%tab pytorch
prices = d2l.hw_prices()
```

<!-- Placeholder (prose): derivation D2. A GPU-second costs c_s = price /
3600; it trains MFU g P / F_tok tokens, with F_tok the FLOPs per token, P
the peak and g the goodput, the fraction of wall time that advanced the
model (chapter 23 measures it; a stated value here); so a million tokens
cost C = 1e6 c_s F_tok / (MFU g P). Serving: C = 1e6 c_s / (T_gpu u), T_gpu
the tokens per second per GPU at the operating point and u the fraction of
paid hours the GPU serves. -->

```{.python .input #energy-cost-cost-defs}
%%tab pytorch
#@save
def cost_per_mtok_train(price_gpu_h, flops_per_token, p_peak, mfu,
                        goodput=1.0):
    """Dollars per million training tokens: a GPU-second costs
    price_gpu_h / 3600 and trains mfu * goodput * p_peak / flops_per_token
    tokens."""
    return 1e6 * price_gpu_h / 3600 * flops_per_token / (
        mfu * goodput * p_peak)

def cost_per_mtok_serve(price_gpu_h, tps_per_gpu, utilization=1.0):  #@save
    """Dollars per million served tokens at tps_per_gpu tokens/s while the
    GPU serves a fraction utilization of the hours it is paid for."""
    return 1e6 * price_gpu_h / 3600 / (tps_per_gpu * utilization)
```

<!-- Placeholder (prose): the formula against eq_six_nd: with F_tok = 6N
and a price of one dollar per GPU-hour, the dollars are GPU-hours, and D
tokens take 6 N D / (MFU P 3600) of them: about 4,200 for N = 1e9,
D = 1e12, MFU = 0.4 on an H100 (the "GPU-hours before dollars" arithmetic,
restated here because sec_cloud_instances no longer carries that
exercise). -->

```{.python .input #energy-cost-six-nd}
%%tab pytorch
N, D, P_h100 = 1e9, 1e12, specs.loc['H100 SXM'].p16_tflops * 1e12
gpu_h = d2l.cost_per_mtok_train(1.0, 6 * N, P_h100, 0.4) * D / 1e6  # $1/h
print(f'6 N D at N = 1e9, D = 1e12, MFU 0.4, H100: {gpu_h:,.0f} GPU-hours')
```

<!-- Placeholder (prose): the table: the reference model and Qwen3-8B on
the box's card, and Qwen3-8B on three data-centre chips, each chip at the
median of its rent quotes and each MFU with its source. The reference
model's MFU is chapter 21's four-card run (or this section's one-card
measurement). Qwen3-8B on the RTX 4090 is 22.3's one-card depth series,
an upper bound because it pays no communication, and it stays on its own
chip. The data-centre rows use chapter 23's measured MFU once committed;
until then a stated MFU of 0.4, the value Llama 3 reports for its H100
training (38-43% bf16 MFU, :cite:`Grattafiori.Dubey.Jauhri.ea.2024`), the
same for A100 and B200 as an assumption. Those rows are pending T2-T4
(chapter 23), and their dollar column is conditional on the stated MFU.
The dollar column is read in the table, never quoted. -->
<!-- campaign: lm-mfu-table-v1 -->

```{.python .input #energy-cost-train-cost}
%%tab pytorch
g, g_source = d2l.training_goodput(stated=0.9)
mfu = d2l.training_mfu()
cfgs = {'reference': R, 'Qwen3-8B': d2l.hw_config('Qwen/Qwen3-8B')}
mfu['$/M tok'] = [d2l.cost_per_mtok_train(
    d2l.rent_per_hour(prices, r.chip),
    d2l.flops_per_token(cfgs[r.model], T=r.T, causal_half=True),
    specs.loc[r.chip].p16_tflops * 1e12, r.mfu, goodput=g)
    for r in mfu.itertuples()]
d2l.show_table(mfu.drop(columns='T'), digits={'mfu': 2}, width=79)
print(f'goodput g = {g:.2f} ({g_source})')
```

<!-- Placeholder (prose): serving cost at three points on the box's
frontier (sec_lm_workload_hardware's measured Qwen3-8B decode at L = 2,048:
b = 1, 4, 16; tokens/s per user against per GPU), at the rented RTX 4090's
price and a stated utilization; the more users share a step, the cheaper
each token and the slower each user's stream. Prefill is billed separately
(per prompt token, from the measured prefill time of a 2,048-token prompt);
a prefix-cache hit rate h removes the hits' prefill (loading a hit's cache
is counted as free here; sec_lm_memory_tiers weighs loading against
recomputing). -->

```{.python .input #energy-cost-serve-cost}
%%tab pytorch
grid = d2l.hw_box_log('box-workloads', 'grid-box')
price, u = d2l.rent_per_hour(prices, 'RTX 4090'), 0.5  # u: busy half the time
points = d2l.frontier_points(grid, 'Qwen/Qwen3-8B', L=2048, batches=(1, 4, 16))
points['$/M tok'] = [d2l.cost_per_mtok_serve(price, T_gpu, u)
                     for T_gpu in points.gpu_tps]
d2l.show_table(points, digits={'user_tps': 0, 'gpu_tps': 0, '$/M tok': 2},
               headers={'user_tps': 'tok/s per user',
                        'gpu_tps': 'tok/s per GPU'}, width=79)
n, h = 2048, 0.5  # prompt tokens, prefix-cache hit rate
t_prefill = d2l.prefill_seconds(grid, 'Qwen/Qwen3-8B', n)
C_prefill = d2l.cost_per_mtok_serve(price, n / t_prefill, u)
print(f'prefill of {n:,}-token prompts: ${C_prefill:.3f} per M tokens, '
      f'${(1 - h) * C_prefill:.3f} at {h:.0%} cache hits')
```

<!-- Placeholder (prose): derivation D5. Owning costs h_own = capex / life +
P_kW PUE e + ops per hour whether the chip works or not; renting costs r
per used hour; per hour of work, owning costs h_own / u at utilization u,
so owning is cheaper above u* = h_own / r. -->

```{.python .input #energy-cost-owned-def}
%%tab pytorch
#@save
def owned_vs_rented(capex, life_h, power_kw, pue, elec_per_kwh, ops_per_h,
                    rent_per_h):
    """Owning costs h_own per hour, used or not (capex over its life, power
    at the wall, operations); renting costs rent_per_h per used hour. Owning
    is cheaper above the utilization u* = h_own / rent_per_h.
    owned_per_used_h is h_own, the cost of an owned hour of work at full
    use (h_own / u at utilization u)."""
    h_own = capex / life_h + power_kw * pue * elec_per_kwh + ops_per_h
    return dict(owned_per_used_h=h_own, u_star=h_own / rent_per_h)
```

<!-- Placeholder (prose): the box's card (a quarter of the four-card
server, an assumption; its marketplace rent) against an H100 SXM (the
quoted purchase price per GPU, the midpoint of a 20-30k USD range; the
median of its rents), both at their TDP and the shared electricity, PUE and
life rows. The H100 stands in for the plan's A100: no dated A100 purchase
quote was found. Name the simplifications: power is charged at TDP for
every hour, used or not, and the host's power (CPU, memory, fans, a large
share of a server's wall power) is left out. The interval draws every input
uniformly within +-30% (10,000 draws, seed 0); the tornado moves one input
at a time, assumption inputs hatched. Claim E6: the rent comes first in
both tornadoes, but that is a property of the formula, not of the data:
u* = h_own / r has elasticity -1 in the rent and less than 1 in any owned
input, whose share of h_own is below one. The data decide the order of the
rest: read it from the plot (the smoke run had power, PUE and electricity
first for the box card and the life, then the purchase, for the H100).
State the intervals, and that a conclusion resting on one assumption row
is conditional on it. -->

```{.python .input #energy-cost-owned-rented}
%%tab pytorch
inputs = d2l.owned_inputs(prices, specs)  # the seven inputs per chip
for name, x in inputs.items():
    u_star = d2l.owned_vs_rented(**x)['u_star']
    lo, hi = d2l.u_star_interval(x, rel=0.3, draws=10_000, seed=0)
    print(f'{name}: u* = {u_star:.2f}, 90% of draws in {lo:.2f}-{hi:.2f}')
d2l.plot_tornado(inputs, rel=0.3)
```

<!-- Placeholder (prose): one line on units: the calculators run unchanged
on unit-carrying arguments and return dollars per token and per hour and a
pure number. -->

```{.python .input #energy-cost-units}
%%tab pytorch
d2l.check_cost_units()
```

## Summary

<!-- Placeholder (prose): 3-6 sentences: a chip's power is a static floor
plus a dynamic part that grows as the cube of the clock, and a compute-bound
load sits at the limit; energy per token is power over tokens per second,
so batching (decode) and MFU (training) set it; a token's price is the
chip-hour price over tokens per chip-hour, owning pays above u* = h_own / r,
and u* moves most with the rent; forward to Chapter 23 (goodput, the
measured MFU per layout) by :numref: if its label resolves. -->

## Exercises

<!-- Placeholder (prose): 6-8 exercises per S/spec/22.8.md part 9: the
energy per token of an SRAM-only design holding Qwen3-8B on N_c chips at
rated power (derive); the electricity price at which a GPU's lifetime energy
equals its purchase price (derive); power at the compute-bound batch
against idle and the ratio to TDP (measure, 1 min); the step-level power
swing of a synchronous training loop at 100 Hz and its amplitude (measure,
3 min, one or two GPUs); the units of the cost functions (check); open: a
filler-work smoother that bounds the swing below a threshold and the energy
it costs. -->
