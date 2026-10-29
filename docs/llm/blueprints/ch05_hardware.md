# Chapter 5: Hardware for Training and Inference — Blueprint

**Label:** `chap_lm_hardware` · **Arc:** III. Scaling out (what hardware training and serving need) · **Sections:** 5.1 `sec_lm_accelerator_design`, 5.2 `sec_lm_workload_hardware`, 5.3 `sec_lm_memory_tiers`, 5.4 `sec_lm_server_paths`, 5.5 `sec_lm_networks`, 5.6 `sec_lm_scale_up_out` · **Tiers:** T0 (5.6, plus the table, simulator and log-analysis cells of every section), T2 (5.1, 5.2: one A100 40 GB PCIe; the two-GPU runs of 5.2's MoE series, 5.4 and 5.5 use the T2 pair), T3 (5.3, 5.4), T4 (5.5); every T2+ section has a T1 reduction on one build-box GPU (RTX 4090, 24 GB, PCIe). The build box has no NVLink and unreliable peer-to-peer transfers, so no multi-GPU run happens there: GPU-to-GPU measurements run on the T2 pair, and the T1 versions of the network and collective cells use one GPU, host-to-device transfers or a gloo/CPU process group · **Assignment:** A5

## Chapter overview

### Draft introduction (for the chapter page)

Advanced taught how one GPU behaves: a roofline with two machine numbers, a memory hierarchy, and the prefill/decode contrast. This chapter asks what happens when the machine is a design, a server and a cluster. First, why do accelerators look as they do? A chip is a budget of die area, watts and package perimeter, split among matrix units, on-chip memory, off-chip memory and links, and every split moves the ridge point and the time to sweep memory (5.1). Training, prefill and decode load those resources very differently, so the best chip for each differs (5.2). Next, where do bytes live and how do they move? Every memory and storage tier from SRAM to disk (5.3), every link inside a server (5.4) and every link between servers (5.5) gets a bandwidth, a latency and a measurement. Finally, scale-up and scale-out domains, power and price turn those measurements into layouts and dollars per token (5.6). The chapter is quantitative throughout: each hardware number cites a datasheet, each model is fitted to a measurement, and the assignment asks for a performance model within 15% of what the machine does.

### Learning objectives (refined from v5; each is checked by a cell or an exercise)

1. Given a chip's area, power and package budget, compute how a different split between SRAM and matrix units changes the ridge point and the on-chip capacity needed to reach it (5.1, derivation D1; check cell).
2. From a datasheet row, compute the five numbers (peak FLOP/s per format, memory bandwidth, capacity, scale-up bandwidth, scale-out bandwidth) and the three ratios (ridge, sweep time `C/β`, network bytes per FLOP), and place a chip on the design plane (5.1).
3. Predict training step time, time to first token and decode throughput of a dense or mixture-of-experts model from its Hugging Face config and a calibrated chip, to within 15% on held-out configurations (5.2, A5).
4. Derive the batch size at which decode turns compute-bound, the context length beyond which it never does, the maximum batch that fits, and the throughput-versus-interactivity frontier (5.2, D1 to D4).
5. State, for each of eight memory and storage tiers, a measured bandwidth, latency and capacity, and place weights, optimizer state, activations, KV cache, datasets and checkpoints on tiers (5.3).
6. Decide whether offloading pays from the overlap condition, and compute the training time lost per checkpoint for synchronous, asynchronous and sharded writes (5.3).
7. Read `nvidia-smi topo -m`, measure the GPU-to-GPU bandwidth and latency matrix and fit α–β parameters per link class (5.4).
8. Predict flat and hierarchical all-reduce and all-to-all times across nodes, and locate the ring/tree crossover (5.5).
9. Decide which parallel axes fit inside one scale-up domain, and convert measured throughput and a price into dollars per million tokens, for owned and for rented hardware (5.6).

### Prerequisites

Book labels (each with what is used; none is re-derived here):

| Label | Used here as given |
|---|---|
| `sec_perf_model` (`eq_roofline`, `eq_matmul_intensity`, `subsec_perf-sweep`, `d2l.Benchmark`) | Roofline, ridge point `I* = P/β`, square-matmul sweep and the `d2l.Benchmark` timing discipline (warm up, synchronize, time). Every timing here uses `d2l.Benchmark` or CUDA events with the same discipline. |
| `sec_hardware` (`subsec_hw-bytes`, `subsec_hw-shoreline`, `subsec_hw-gpu`, `subsec_hw-cpu`, `subsec_hw-interconnects`, `subsec_hw-energy`, `subsec_hw-two-workloads`, `tab_gpu_specs`, `eq_hw_decode_bound`, `fig_memory_hierarchy`, `fig_bandwidth_ladder`, `fig_latency_ladder`, `fig_pcie_topology`, `fig_energy_ladder`) | The ladders, the shoreline argument (compute ∝ area, I/O ∝ perimeter), tensor-core formats, pinned memory, the build box's host-staged PCIe topology, the energy ladder, and the bytes-per-token decode bound. `tab_gpu_specs` (four chips) is extended, not replaced: `d2l.chip_specs()` must reproduce its four columns exactly (test). |
| `sec_memory_precision` (`subsec_mp-anatomy`: 16 bytes per parameter with Adam) | Training memory anatomy. |
| `sec_multi_gpu` (`eq_ring_traffic`, `eq_dp_cost`, bus-bandwidth exercise, "Latency in the allreduce" exercise) | Ring all-reduce traffic `2(k−1)/k·N`, `t_step = t_comp + 2N/β`, the algbw/busbw convention. The α–β model appears in the book only as exercises (a ring and a binary-tree formula); 5.4 states it as a model in one paragraph and fits it. |
| `sec_multi_gpu_concise` (`tab_collectives`, the `NCCL_SHM_USE_CUDA_MEMCPY` recovery on the P2P-less box) | The collective table and the build-box NCCL finding. |
| `sec_fast_transformer` | The tokens/s benchmark contract (fixed shapes, warm-up, end-to-end). |
| `sec_kv-cache` (`eq_kv-cache-bytes`, `subsec_kv-cache-prefill-decode`, GQA) | KV bytes per token, prefill versus decode intensity. |
| `sec_moe` | Total versus active parameters. |
| `sec_scaling-laws` | `6ND`. |
| `sec_read_write`, `sec_use_gpu` | safetensors, device handling. |

Earlier sections of this part: 1.1 (`d2l.load_hf`, `d2l.config_memory`, used to read Qwen configs), 1.2 (`d2l.load_qwen3` for the T1 measurements), 4.2 (`d2l.flops_per_token`, `d2l.mfu`), 4.1 (`d2l.PackedStream`: token bytes per sample and shard sizes), 4.4 (low-precision formats; FP8 on the Ada build box). Forward references only by label: 6.1 (`d2l.launch`), 6.7 (commits `lm-mfu-table`, used in 5.6), 6.8 (exports `d2l.failure_rate`; reliability, Young/Daly intervals; consumes 5.3's `ckpt_records.jsonl`), 13.4 (disaggregated serving), 12.1 (`d2l.FastGenerator`, not needed here).

### Section dependency graph

```
5.1 chip design ──► 5.2 workloads ─────────────────────────┐
   │ (chip_specs, BenchLog, calibration)                    ▼
   ├──► 5.3 memory/storage tiers ──► 5.4 inside a server ──► 5.5 between servers ──► 5.6 scale-up/out & cost
   │                                      (α–β fits)              (hierarchical models)     ▲
   └──────────────────────────────────────────────────────────────────────────────────────┘
A5 uses 5.1 (calibration protocol), 5.2 (model), 5.4–5.5 (α–β fits, simulator).
```
5.3 and 5.4 can be taught in either order (5.3 uses only host-to-device links from 5.4's list). 5.6 uses 5.2's `ServingModel` and 5.5's simulator, and is the only section that reads another chapter's measurement (6.7 MFU), by `DATA_HUB` artifact (`lm-mfu-table`, committed by 6.7) with a one-line fallback.

### Running examples and shared artifacts

Used: R1 (Qwen3 configs; `Qwen/Qwen3-8B`: 36 layers, 8 KV heads, head dim 128, hidden 4096, 8.19B parameters, from its `config.json`), `Qwen/Qwen3-0.6B` (28 layers, hidden 1024, 16 query and 8 KV heads; T1 stand-in), `Qwen/Qwen3-30B-A3B` (48 layers, 128 experts, top-8, 4 KV heads; MoE case in 5.2 and 5.6), R2 (reference model, as the unit of checkpoint size in 5.3).

Produced (all committed; `DATA_HUB` keys are registered by 5.1, which owns the registry block):

| Artifact | Producer | Format | Consumers |
|---|---|---|---|
| `d2l.chip_specs()` and `data/hw/chip_specs.csv` | 5.1 | DataFrame; one row per chip, `source_url` and `retrieved` per row | 5.2, 5.6, A5 |
| `d2l.BenchLog` and `d2l.env_fingerprint()` | 5.1 | JSONL logger with an environment fingerprint per record | 5.2–5.5, A5, 6.x |
| `lm-hw-a100-microbench` | 5.1 | tar of JSONL logs and PNGs | 5.1, 5.2, A5 |
| `lm-hw-a100-workloads` | 5.2 | JSONL: train/prefill/decode grid on one A100 | 5.2, 5.6, A5 |
| `lm-hw-inferencex-snapshot` | 5.2 | CSV subset of public InferenceX results (upstream Apache-2.0; shipped with attribution), with date, upstream commit hash, a copy of the licence and a `NOTICE.md` | 5.2 |
| `data/hw/tiers.csv` | 5.3 | tier table with source per row | 5.3, 5.6 |
| `lm-hw-node-storage`, `lm-hw-node-p2p` | 5.3, 5.4 | JSONL: nvbandwidth, fio, gdsio, p2pBandwidthLatencyTest, nccl-tests | 5.3, 5.4, A5 |
| `data/hw/links.csv` | 5.4 | link-class table (NVLink, PCIe, C2C, IB, Ethernet) with source per row | 5.4, 5.5, 5.6 |
| `lm-hw-cluster-nccl` | 5.5 | nccl-tests JSON/text from 2, 3 and 4 nodes, `ib_write_bw` and `ib_write_lat` | 5.5, A5 |
| `lm-hw-buildbox-*` | 5.1–5.5 | the T1 versions of the above, from the build box: single-GPU and host-to-device measurements, topology captures and gloo/CPU loopback runs (no GPU-to-GPU run) | all |
| `lm-hw-t2pair-p2p`, `lm-hw-t2pair-net` | 5.4, 5.5 | the two-GPU versions from the T2 pair: nccl-tests at `g=2` over the NVLink bridge and with `NCCL_P2P_DISABLE=1`; NCCL over TCP loopback for 5.5 | 5.4, 5.5, A5 |
| `lm-hw-a5-reference` | A5 | staff reference calibrations and measurements (A100 and RTX 4090) | autograder |

The `lm-hw-a100-*` keys hold logs from the A100 40 GB PCIe (the `A100 PCIe 40 GB` row of `chip_specs`); T3 logs come from the 8× A100 node, whose interconnect (NVSwitch on SXM boards, or NVLink-bridged PCIe pairs) is recorded by `nvidia-smi topo -m` in a committed `topo.txt` that decides which row of `tab_lm_hw_links` and `systems_table()` applies. Its GPUs have 40 GB each (1,555 GB/s in both the PCIe and the SXM 40 GB part).

### Compute plan

| § | Tier | Reduced T1 config | GPU-hours (full) | Peak memory | Committed artifacts |
|---|---|---|---|---|---|
| 5.1 | T0 table; T2 microbenchmarks | RTX 4090, same code, `--dtype` list without TF32 sweep, FP8 added | ~0.5 A100-h | < 20 GB (A100); ≤ 8 GiB (T1) | `lm-hw-a100-microbench`, `chip_specs.csv` |
| 5.2 | T2 | Qwen3-0.6B (train, prefill, decode) and Qwen3-1.7B (decode) on one 4090, grid cut to 12 points | ~3 A100-h, plus a short T2-pair job for the MoE series | ≤ 36 GB (A100 40 GB: 0.9 × 40; points predicted above it are dropped); ≤ 14 GiB (T1) | `lm-hw-a100-workloads`, `lm-hw-inferencex-snapshot` |
| 5.3 | T3 | 1 GPU, build-box NVMe and pinned/pageable host copies, 0.6B training state (~10 GB) instead of 8B (~131 GB) | ~2 node-h | host 256 GB for the 8B state | `lm-hw-node-storage` |
| 5.4 | T3 + T2 pair | one build-box GPU (host-to-device series, topology capture); `g=2` collectives on the T2 pair (NVLink bridge versus `NCCL_P2P_DISABLE=1`) | ~1 node-h, plus a short T2-pair job | small | `lm-hw-node-p2p`, `lm-hw-t2pair-p2p` |
| 5.5 | T4 (analysis of logs, ≤ 1 h of notebook time); T0 simulator | T2 pair with NCCL forced onto TCP loopback (2 ranks); gloo/CPU loopback with 2 and 4 ranks on the build box; T0 simulator | core: 2 nodes × ≤ 1 h; optional: 4 nodes × ≤ 1 h | small | `lm-hw-cluster-nccl`, `lm-hw-t2pair-net` |
| 5.6 | T0 | none needed | 0 | — | `data/hw/prices.csv` (dated, sourced) |
| A5 | T1 (build box) or T2–T3 | tasks 1–3 on one build-box GPU; task 4 (all-reduce) is graded in reference mode from committed logs | ~1 GPU-h of staff reference runs | — | `lm-hw-a5-reference` |

Total: about 4.5 A100-hours (5.1: 0.5, 5.2: 3, A5: 1; the short T2-pair jobs are extra), ~3 node-hours (5.3 and 5.4) and one multi-node allocation of about 2 node-hours (about 6 with the optional 4-node pass). All T3/T4 logs are produced once per release (v5 build-system rule), not per build.

### Microbenchmark contract (applies to 5.1, 5.3, 5.4, 5.5, A5)

Every microbenchmark in the chapter follows one contract so that logs are comparable and graders can check them:

1. **Command.** The exact command line is a string constant in the notebook, printed in the cell output and stored in the log record (`cmd`).
2. **Log.** Every run appends one JSON object per measurement to `logs/<section>/<tool>_<tier>.jsonl` through `d2l.BenchLog`. A record has `ts`, `tool`, `cmd`, `env` (from `d2l.env_fingerprint()`: GPU model and UUID prefix, driver, CUDA, PyTorch and NCCL versions, SM and memory clocks, power limit, `nvidia-smi topo -m` hash, host name, node count, git commit), `params`, `results` and `raw` (the unparsed tool output, gzip+base64, so that parsers can be fixed later).
3. **Statistics.** Warm-up runs are discarded; at least 5 timed repeats; report median and the median absolute deviation (MAD); a headline number is the median of per-repeat medians over 3 independent processes. A comparison between two configurations is a claim only if the gap exceeds 3× the larger MAD.
4. **Parser.** One parser function per tool, tested on a committed raw sample (`tests/data/<tool>.txt`) in a T0 cell; the committed logs are re-parsed from `raw` in the analysis cell, never trusted from `results`.
5. **Plot.** Each microbenchmark has a plot specification (axes, scale, series, annotations) and a datasheet line drawn on the plot as a dashed reference.
6. **T1 twin.** The same function runs on the build box behind `config.tier = 'T1'`; the plot overlays the T1 and T3 logs (or shows the T1 log alone when the T3 log is not committed yet).

### Notation

`P` peak FLOP/s of a format; `β` memory bandwidth; `C` memory capacity; `R = P/β` ridge point (FLOP/byte); `τ = C/β` sweep time; `N` parameters, `N_act` active parameters; `W` weight bytes; `k_KV` KV bytes per token; `b` batch (sequences); `L` context length; `n` prompt length; link α–β: `t(m) = α + m/β_link`; ranks `k` (as in `eq_ring_traffic`), nodes `n_nodes`, GPUs per node `g`. Chip subscript `eff` marks a measured (calibrated) value, as in `P_eff` and `β_eff`; no subscript marks a datasheet value. Notation matches `sec_perf_model` and `sec_multi_gpu`.

## Assignment

### A5. A performance model of a chip, a node and a cluster

**Label:** `sec_lm_a5`

**Goal.** Build and calibrate a performance model that predicts, from a model config and a hardware description, (i) the training step time, (ii) the time to first token and (iii) the decode throughput of a decoder-only transformer on one GPU, and (iv) the time of an all-reduce on a node and across nodes. Predictions must be within 15% of measurement on configurations the model was not fitted on.

**Files (starter repository `a5_perfmodel/`).**

```
calibrate.py      # TODO: runs the microbenchmarks of 5.1 and 5.4, writes calib.json
model.py          # TODO: FLOP/byte counting from a config; predict_train_step, predict_ttft, predict_decode_tps, predict_allreduce
measure.py        # given: measurement harness implementing the protocol below; writes JSONL via d2l.BenchLog
check_log.py      # given: validates a measurement log against the protocol (warm-up, repeats, CV, env fingerprint)
grade.py          # given: reference mode (T0) and own-hardware mode
configs/          # given: calibration configs C and (visible) validation configs V
```
The hidden test configurations `H` are generated at grading time from the same family as `V` with different values (see below).

**Config family.** Qwen3-style dense decoders, random-initialized from a `Qwen3Config` (no download): hidden size `d ∈ {1024, 1536, 2048}`, layers `∈ {8, 16, 28}`, query heads `d/128`, KV heads `∈ {4, 8}`, head dim 128, MLP width `3d`, vocabulary 151,936, tied embeddings. `Qwen3-0.6B` (d=1024, 28 layers) is a member. Two members are in `C`, two in `V`, three in `H`; no member of `H` is in `C` or `V`.

**Tasks.**

1. **Calibrate (calibrate.py).** Measure, on the target GPU: `P_eff` for BF16 (square matmul, sizes 4096, 8192, and the largest size that fits; take the maximum of medians), `β_eff` (streaming copy of 1 GiB, bytes counted as read plus write), kernel-launch overhead `α_k` (median time of 1,000 back-to-back empty CUDA-graph-less kernel launches, divided by 1,000) and, on a multi-GPU machine, the α and β of an intra-node NCCL all-reduce (5.4). Output `calib.json` with fields `P_eff_bf16`, `beta_eff`, `alpha_k`, `alpha_ar`, `beta_ar`, `n_gpu`, and the environment fingerprint. The datasheet values may be used only as sanity checks (printed, not graded).
2. **Count (model.py).** Implement `count_params(cfg)`, `flops_per_token(cfg, L)` (a student-local function in `model.py`, not a `d2l` export; the reference solution cross-checks it against `d2l.flops_per_token` (4.2); forward FLOPs: `2·N_matmul + 4·n_layers·n_heads·d_head·L` for attention over a context of `L`, with the causal factor stated), and `bytes_per_token_decode(cfg, L, dtype_bytes)`. Tested against `sum(p.numel())` of the instantiated model (exact), and against `torch.utils.flop_counter.FlopCounterMode` (±2%).
3. **Predict, single GPU.** Implement `predict_train_step(cfg, b, s, calib)`, `predict_ttft(cfg, n, calib)`, `predict_decode_tps(cfg, b, L, calib)` using a model of the form `t = max(t_compute, t_memory) + n_kernels·α_k` where the kernel counts and efficiencies are the student's modelling choices. At most **three free parameters beyond `calib.json`** may be fitted, and only on `C`. The student documents each in the report.
4. **Predict, node and cluster.** Implement `predict_allreduce(size_bytes, topo, calib)` for flat ring, tree and two-level hierarchical algorithms (5.5 simulator interface) and `predict_ddp_step(cfg, b, s, k, calib)` (compute plus non-overlapped communication, with overlap fraction a fitted parameter). Own-hardware mode on one GPU skips this task and receives its credit from the committed T4 logs (reference mode).
5. **Written analysis (≈ 800 words, 4 questions):** (a) Which regime (compute, bandwidth, launch overhead) binds each of the three workloads at the smallest and largest grid point, with the evidence? (b) Where is your model worst, and what physical effect is missing? (c) Predict the same three quantities for the same model on an H100 and a B200 from datasheets only and say how much of each prediction you trust and why. (d) For the chip and the model of your choice, at what price per GPU-hour does an owned cluster beat a rented one, using 5.6's formula and your measured throughput?

**Measurement protocol (normative; `measure.py` implements it; `check_log.py` enforces it).**

- *Environment.* One process per measurement series. Record `nvidia-smi -q -d CLOCK,POWER,PERFORMANCE` before and after; record GPU temperature before each series. Clocks are locked (`nvidia-smi -lgc <min>,<max>`) if permitted, else unlocked and recorded; the log carries `clocks_locked` as a boolean. No other process on the GPU (`nvidia-smi --query-compute-apps` empty).
- *Training step.* Synthetic token batch on the device, `bfloat16` parameters and activations (`torch.autocast` off; the model is cast with `.to(torch.bfloat16)`), fused AdamW with `lr=1e-4`, `zero_grad(set_to_none=True)`, scaled dot-product attention with the default backend, activation checkpointing **off** unless the step does not fit (then on, flagged `ckpt=true`, and the model must count the recompute), eager mode (`torch.compile` off). One timed unit is `forward + backward + optimizer.step()`; the data loader is excluded. Warm-up 10 steps; timed 30 steps; each step bracketed by CUDA events; the step time is the median of the 30.
- *TTFT.* Batch 1, prompt of `n` random tokens, Hugging Face `forward(input_ids, use_cache=True)` on a fresh cache; the timed interval ends when the last-position logits are available on the device (`torch.cuda.synchronize()`); tokenization, sampling and scheduling are excluded, so the quantity is "prefill latency of the model forward" and is called TTFT_model in the report. Warm-up 5, timed 20, median.
- *Decode throughput.* Prefill `L` tokens at batch `b`, then time `64` decode steps (one token per sequence per step) with a static KV cache and a CUDA-graph-captured step (`torch.cuda.CUDAGraph` around one `forward`), greedy argmax inside the graph; tokens/s = `64·b / t`. Warm-up 8 graph replays; 5 repeats of the 64-step window; median. A second, **ungraded-but-reported** eager series (no graph) exposes the launch-overhead term.
- *All-reduce.* `all_reduce_perf -b 8 -e 4G -f 2 -g <k> -w 10 -n 50 -c 0` for one process with `k` GPUs; cross-node with `mpirun -np <nodes·g> -N <g> ... -g 1`; time column (out-of-place) is the measurement; 3 repeats of the whole sweep; the median per size.
- *Acceptance.* A measurement point is *valid* if its coefficient of variation (1.4826·MAD/median over the timed steps) is ≤ 5% and the three repeats' medians agree within 5%; invalid points are re-run up to twice, then dropped from grading with a note.

**Autograder checks and tolerances.**

| Check | Pass condition | Weight |
|---|---|---|
| `count_params` exact on all of `C ∪ V ∪ H` | equal to `sum(p.numel())` | 10 |
| `flops_per_token` against `FlopCounterMode` | relative error ≤ 2% on every config and `L ∈ {512, 4096}` | 10 |
| `check_log.py` on the student's own measurement log | all protocol fields present; ≥ 80% of points valid | 10 |
| Calibration sanity | `P_eff_bf16 ≥ 0.5·P_datasheet` and `β_eff ≥ 0.5·β_datasheet` for the detected GPU (guards against broken measurement, not a performance target) | 5 |
| Training step, held-out grid | per-point error `ε = |t_pred − t_meas|/t_meas`; the point *passes* if `ε ≤ 0.15`; score = fraction passing; full marks at ≥ 0.80 of points, linear credit below, zero below 0.40 | 20 |
| TTFT, held-out grid | same rule | 15 |
| Decode (graph) tokens/s, held-out grid | same rule | 15 |
| All-reduce (reference mode) | sizes ≥ 16 MiB: `ε ≤ 0.15`; sizes < 16 MiB: `ε ≤ 0.30` (latency-dominated); full marks at ≥ 0.80 of points | 10 |
| Written analysis | rubric: four questions × 1.25 points each, plus 0 deductions for hedging | 5 |

*Reference mode (T0, deterministic).* `grade.py --reference` feeds the student's `model.py` the staff calibration `lm-hw-a5-reference/calib_{a100,rtx4090}.json` and compares predictions with the staff measurements `lm-hw-a5-reference/meas_{a100,rtx4090}.jsonl` on `H` (`a100` is the A100 PCIe 40 GB, whose datasheet row `grade.py` uses for the calibration sanity check). It needs no GPU. *Own-hardware mode* compares predictions with measurements the student made (or the staff re-ran on the student's submitted code at the submission commit on the same GPU type). The tolerance 0.15 is three times the 5% acceptance CV, so a pass is not decided by noise; the staff reference measurements carry their own CV and points with CV > 5% are removed from the denominator.

**Why 15%.** It is the v5 target, and the book's own check: the 5.2 notebook shows that a datasheet-only roofline misses by more than this, and a calibrated model with an overhead term lands inside it (to be measured; the notebook prints both).

**T1 path.** Everything except the 8-GPU and multi-node parts runs on one build-box GPU within 14 GiB: the config family is small (at most about 1.7B parameters, at `d=2048` and 28 layers). The protocol trains in pure bfloat16, so state is about 8 bytes per parameter (weights, gradients, two bfloat16 AdamW moments) plus activations; `H` training points are therefore limited on T1 to configs of at most 1.0B parameters (to be confirmed by `measure.py --dry-run`, which reports peak bytes), and larger members are TTFT and decode only. No all-reduce runs on the build box (it has no multi-GPU path): task 4 is graded in reference mode from committed logs (the T2 pair at `g=2` with the NVLink bridge and with `NCCL_P2P_DISABLE=1`, the T3 node and the T4 cluster). A student without a multi-GPU machine can still fit the α–β model on a gloo/CPU process group (optional `calibrate.py --gloo-cpu`); that fit earns no all-reduce credit.

**Estimated student time:** 14–18 hours (calibration 3, counting 2, single-GPU model 5, node/cluster model 3, report 3).

## Teaching sources

All of v5's list is kept; annotations say what each is used for.

- [How to Scale Your Model](https://jax-ml.github.io/scaling-book/), especially [How to Think About GPUs](https://jax-ml.github.io/scaling-book/gpus/) and [All About Transformer Inference](https://jax-ml.github.io/scaling-book/inference/): the rooflines for training and inference (5.2), GPU network levels (5.4–5.6). Translate its JAX examples to PyTorch.
- Stas Bekman, [Machine Learning Engineering Open Book](https://github.com/stas00/ml-engineering): the practical network, storage and benchmark chapters (5.3–5.5); `all_reduce_bench.py`-style scripts as a cross-check of nccl-tests.
- [Modal GPU Glossary](https://modal.com/gpu-glossary): vocabulary (SM, warp, tensor core, NVLink) for 5.1 and 5.4.
- [GPU MODE lectures](https://github.com/gpu-mode/lectures): microbenchmark methodology and profiling (5.1).
- [Efficiency in LLMs (MLSS)](https://alex.smola.org/posts/45-mlss-efficiency/): accelerator/model matching (5.2); carried over from Codex 4.8.
- [Berkeley Scalable AI syllabus](https://scalable-ai.eecs.berkeley.edu/assets/syllabus.pdf): course ordering of hardware, networks and parallelism.
- Barroso, Hölzle and Ranganathan, [The Datacenter as a Computer](https://link.springer.com/book/10.1007/978-3-031-01761-2) (3rd ed.): storage tiers and the latency/bandwidth/capacity hierarchy (5.3), power and cooling and TCO (5.6).
- [Latency Numbers Every Programmer Should Know](https://gist.github.com/jboner/2841832): historical ladder; used as a dated contrast to measured values in 5.3 only, never as a source for current hardware numbers.
- From the Codex plan (carried over): [MLCommons inference benchmarks](https://mlcommons.org/benchmarks/inference-datacenter/) (scenario definitions, 5.2); [Scaling Book: sharding](https://jax-ml.github.io/scaling-book/sharding/) (placement and communication models, 5.4–5.6); [NCCL user guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/) and [nccl-tests](https://github.com/NVIDIA/nccl-tests) (benchmark methodology and bandwidth conventions, 5.4–5.5). The Codex plan's private authoring decks are not cited, by v5 convention.

## Sections

### 5.1 Accelerator design: compute, memory and I/O

- **Label:** `sec_lm_accelerator_design` · **Tier:** T0 (table, ratios, derivation checks) + T2 (microbenchmarks on one A100 40 GB PCIe, the `A100 PCIe 40 GB` row of `chip_specs`) · **Reduced T1 config:** the same functions on one RTX 4090 (≤ 8 GiB), `config.tier='T1'`; FP8 series added (Ada has FP8 tensor cores, A100 does not); TF32 series kept · **Budget:** T0 < 1 min; T2 ≈ 25 min wall-clock, < 20 GB peak; T1 ≈ 10 min, ≤ 8 GiB · **Codex units:** 4.8
- **Builds on (book):** `subsec_hw-shoreline` (recapped in one paragraph, not re-derived: compute ∝ area, I/O ∝ perimeter; reticle limit; HBM as the countermeasure), `subsec_hw-gpu` (tensor cores, format halving), `tab_gpu_specs` (extended to ~15 chips; its four columns are the regression test), `subsec_hw-bytes` and the ladders (taken as given: they are the *targets* the microbenchmarks reproduce), `subsec_perf-sweep` and `d2l.Benchmark` (the square bf16 matmul sweep is **not** repeated; 5.1 sweeps shapes that language models use), `eq_roofline`, `subsec_hw-energy` (energy per operation, used in D1's power term). The `sec_hardware` exercises "Memory sweep time" (computes `C/β` for the four chips and a single-stream decode bound in the fill fraction `f`), "Ridge point of a link" and "Little's law" are acknowledged and not redone: 5.1 promotes `C/β` to a first-class design ratio, computes it for the full table, and adds the KV-cache and multi-chip consequences (D2, D3).
- **Builds on (this part):** 1.1 (`d2l.config_memory`, `AutoConfig` for Qwen3-8B), nothing else (first section of the chapter).
- **Exports (`#@save`):**
  - `d2l.chip_specs() -> pd.DataFrame` (reserved name). One row per chip. Columns: `chip, vendor, kind` (`gpu|tpu|asic|wafer|lpu`), `year, mem_type, mem_gb, mem_bw_tbs, sram_mb` (on-chip SRAM the vendor states: L2 plus shared/L1 or total SRAM; `NaN` if not stated), `p16_tflops` (dense BF16/FP16, FP32 accumulate), `p8_tflops` (dense FP8), `p4_tflops` (dense FP4), `tdp_w, scaleup_gbs` (per-chip bidirectional aggregate, as vendors state; column `scaleup_note` names the fabric), `scaleup_domain` (chips per domain), `scaleout_gbs` (NIC line rate per chip, converted to GB/s), `sparsity_note, source_url, retrieved` (ISO date), `notes`. Every numeric cell is either dense, as stated by the vendor, or `NaN` with a note; no cell is estimated. Contract: loading fails (assert) if any row lacks a `source_url` or `retrieved`, or if the four `tab_gpu_specs` chips differ from the book table. The `A100 PCIe 40 GB` row is an extra row beyond those four (the regression test does not cover it); the `A100 SXM 80 GB` row is kept for datasheet comparisons.
  - `d2l.derive_ratios(df) -> pd.DataFrame`: adds `ridge16 = p16·1e12/(mem_bw·1e12)` in FLOP/byte, `sweep_ms = mem_gb/mem_bw_tbs` (in ms: GB divided by TB/s gives ms), `net_bytes_per_flop = scaleout_gbs·1e9/(p16·1e12)`, `scaleup_bytes_per_flop`, and `hbm_fraction_sram = sram_mb/(1000·mem_gb)`.
  - `d2l.BenchLog(path, tool, tier)`: context manager; `.record(cmd, params, results, raw)` appends one JSONL record under the contract in the chapter overview; `.read(path) -> list[dict]`; `.power_trace()` is a context manager that samples `pynvml` at 10 Hz during a timed window (power draw in W, SM and memory clocks, temperature, clock-throttle reasons) and stores the samples in `results['power']`; 5.6 reads them for energy per token and the DVFS fit.
  - `d2l.env_fingerprint() -> dict`: GPU name/UUID prefix, driver, CUDA, torch, NCCL, clocks (`pynvml`), power limit, host, `nvidia-smi topo -m` hash, git commit.
  - `d2l.hw_matmul_bench(m, n, k, dtype, repeats=...) -> dict` (`tflops`, `ms`, `mad`) and `d2l.hw_copy_bench(nbytes, kind='copy'|'read') -> dict` (`gbs`, `ms`, `mad`), both on `d2l.Benchmark`'s discipline with a CUDA-graph option that amortizes launch.
  - Data: `DATA_HUB['lm-hw-a100-microbench']` (tar of JSONL logs per tool, and the figures) and `DATA_HUB['lm-hw-buildbox-microbench']`.
- **Models and data:** `Qwen/Qwen3-8B` config only (Apache-2.0, 36 layers, hidden 4096, intermediate 12288, 32 query and 8 KV heads, head dim 128, vocabulary 151,936) to generate the LLM-shaped GEMM list. No weights. No datasets.
- **Packages:** `torch==2.15.0`, `pandas`, `matplotlib`, `nvidia-ml-py`; optional CUDA toolkit with `nvcc` ≥ 12 for the two `load_inline` kernels (shared-memory bandwidth, pointer chase); if absent the notebook loads the committed log and says so. Optional Nsight Compute (`ncu`) for an L2 hit-rate cross-check; absence is not an error.

#### Specification table: rows and where each number comes from

Rows (the author transcribes numbers at authoring time from the page named, records `retrieved`, and the CI test re-fetches the page title only, not the numbers). No number is carried in this blueprint except those already in the book or confirmed in this session (marked ✓).

| Row | Source page (data source, not reading list) | Notes |
|---|---|---|
| A100 PCIe 40 GB | the same [A100 datasheet PDF](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf), 40GB PCIe column ✓ (re-read 2026-10-29; HTTP 200): 40 GB HBM2; 1,555 GB/s; 250 W (max thermal design power); BF16 and FP16 tensor 312 TFLOPS dense, 624 with sparsity; TF32 156 dense; "NVIDIA NVLink Bridge for 2 GPUs: 600GB/s" (footnote: PCIe GPUs via NVLink Bridge for up to two GPUs); PCIe Gen4 64 GB/s, all as printed | **The T2 chip: every T2 notebook uses this row.** `mem_type='HBM2'`, `mem_gb=40`, `mem_bw_tbs=1.555`, `p16_tflops=312`, `tdp_w=250`; `scaleup_gbs=600` with `scaleup_note='NVLink bridge, one partner GPU'` and `scaleup_domain=2` (all other traffic goes over PCIe Gen4 at 64 GB/s, recorded in `notes`); `scaleout_gbs=NaN` (the datasheet states no NIC). No FP8 or FP4 (`NaN`, "no FP8 tensor cores"). Same GA100 die as the 80 GB part, so L2 and shared sizes come from the Ampere whitepaper below. Derived from the row (computed): ridge `P16/β` = 200.6 FLOP/byte and sweep time `C/β` = 25.7 ms, against 153.0 FLOP/byte and 39.2 ms for the SXM 80 GB row. |
| A100 SXM 80 GB | [A100 datasheet PDF](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf) ✓ (BF16 312 TFLOPS dense, 624 with sparsity; HBM2e 2,039 GB/s; 400 W; NVLink 600 GB/s; PCIe Gen4 64 GB/s, as printed) | Kept for datasheet comparisons (the 8-GPU SXM node row of 5.6; not a T2 chip). L2 and shared sizes from the [Ampere whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf). No FP8, no FP4: cells are `NaN`, note "no FP8 tensor cores". |
| H100 SXM, H200 SXM | [H100](https://www.nvidia.com/en-us/data-center/h100/), [H200](https://www.nvidia.com/en-us/data-center/h200/); H100 row must equal `tab_gpu_specs` | Dense, not "with sparsity" figures. |
| B200 (HGX) | [HGX platform](https://www.nvidia.com/en-us/data-center/hgx/), [DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/); must equal `tab_gpu_specs` | |
| B300 / GB300 | [GB300 NVL72](https://www.nvidia.com/en-us/data-center/gb300-nvl72/), [Blackwell Ultra blog](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-for-the-era-of-ai-reasoning/) | Per-GPU memory and NVFP4 dense figure as stated. |
| Rubin CPX | [announcement](https://nvidianews.nvidia.com/news/nvidia-unveils-rubin-cpx-a-new-class-of-gpu-designed-for-massive-context-inference) (v5 source) and [Vera Rubin platform](https://www.nvidia.com/en-us/data-center/technologies/rubin/) | GDDR7 part; record only what the page states (NVFP4 figure, memory); BF16 `NaN` if absent. |
| MI300X, MI355X | [MI300X](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html), [MI350](https://www.amd.com/en/products/accelerators/instinct/mi350.html) | |
| TPU v5p, TPU v7 (Ironwood) | [v5p](https://docs.cloud.google.com/tpu/docs/v5p), [TPU7x](https://docs.cloud.google.com/tpu/docs/tpu7x), [Ironwood blog](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/ironwood-tpu-age-of-inference/) | Per-chip figures; pod size in `scaleup_domain`. |
| Trainium2 | [Neuron docs: Trainium2](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium2.html) | |
| Groq LPU, Cerebras WSE-3 | [Groq: The LPU Explained](https://groq.com/blog/the-groq-lpu-explained), [Cerebras chip page](https://www.cerebras.ai/chip) | SRAM-only: `mem_gb` = on-chip SRAM, `mem_type='SRAM'`, off-chip memory absent. If the page gives no dense 16-bit figure, `NaN` plus note. |
| RTX 4090, RTX 5090 | the book's `tab_gpu_specs` (vendor pages cited there) | The 4090 is the T1 chip. |

#### Narrative arc

1. **The question.** The opening cell prints three lines from `d2l.chip_specs()` and one computed column: for an HBM GPU, a GDDR prefill part and an SRAM-only chip, "time to read all of memory once" and "chips needed to hold Qwen3-8B in bf16" (computed from `AutoConfig` via `d2l.config_memory`). The measured symptom is the same cell run on the A100: a BF16 `8192³` matmul reaches a fraction of the datasheet 312 TFLOP/s and a streaming copy reaches a fraction of the 40GB PCIe column's 1,555 GB/s (both to be measured; the cell prints the fractions). Motto: the five numbers on the page are the *design*; what the chip delivers is the *calibration*.
2. **A chip is a budget (subsection "Area, power, perimeter").**
   - One paragraph recapping `subsec_hw-shoreline` (`:numref:`), then the new part: the three budgets and what each limits. Area: matrix units, SRAM (register file, shared memory/L1, L2), memory controller and link PHYs, control. Power: per chip (`tdp_w`) and per rack; the energy ladder of `subsec_hw-energy` as the exchange rate. Perimeter: how many HBM stacks and SerDes lanes fit; chiplets and interposers raise the usable perimeter (B200: two reticle-sized dies, per `sec_hardware`).
   - Figure `fig_lm_hw_budget`: a stylized floorplan with the three budgets annotated (drawn with matplotlib patches from a dict of fractions; fractions are *parameters of the drawing*, labelled "illustrative", not data). A second panel is the area/power breakdown the vendors publish for one die if and only if the Hopper or Ampere architecture pages give one; otherwise omit and say none is public.
   - Equation `eq_lm_hw_areabudget`: `A = A_mm + A_sram + A_phy + A_io + A_ctl` and `P_chip = P_mm + P_sram + P_phy + P_io`.
3. **The off-chip memory menu (subsection "HBM, GDDR, LPDDR, SRAM").**
   - Table `tab_lm_hw_memmenu`: rows HBM, GDDR, LPDDR (Grace-style), DDR5, SRAM-only. Columns: bandwidth per device and capacity per device (from `chip_specs` rows and the [Grace CPU page](https://www.nvidia.com/en-us/data-center/grace-cpu/) for LPDDR5X), interface width and rate (HBM3 from the [JEDEC page](https://www.jedec.org/standards-documents/docs/jesd238b01): 1,024-bit interface per stack, so bandwidth per stack = width × rate / 8; the cell computes it from the JEDEC rate and checks it against the datasheet bandwidth divided by the stack count), relative cost per GB (rank only, from the source table of 5.3; no dollar value is asserted here), and the reason each exists.
   - Reading: HBM buys bandwidth per package perimeter at high cost per GB and a stack-count ceiling; GDDR buys cheap bandwidth per pin at low capacity; LPDDR buys capacity per dollar and per watt at low bandwidth; SRAM-only buys the highest bandwidth at tiny capacity.
4. **Five numbers and three ratios (subsection).** Define the five numbers and the three ratios (ridge, `C/β`, network bytes per FLOP). `d2l.derive_ratios` cell. The new ratio is `C/β`: units ms; what it means for decode (D2).
5. **The design plane (subsection).** Scatter `fig_lm_hw_plane`: x = sweep time `C/β` (ms, log), y = ridge `P16/β` (FLOP/byte, log), marker area ∝ capacity, colour by `kind`, annotate each chip; chips without a dense 16-bit figure sit in a labelled strip below the axis. Reading guide: left = fast to sweep (SRAM, small-memory GPUs); up = compute-heavy relative to bandwidth (prefill-oriented). A second panel `fig_lm_hw_netplane`: network bytes per FLOP (scale-out and scale-up) against chip. Observations are read off the plot in text only after the plot exists; the author writes them from the data, not from this blueprint.
6. **Inside one A100 (PCIe 40 GB): the numbers as measured (subsection, T2).** Four microbenchmarks, each with command, parser and plot:
   1. *LLM-shaped matmuls.* `d2l.hw_matmul_bench` over the GEMMs of one Qwen3-8B block (`q/o` 4096×4096, `k/v` 4096×1024, `gate/up` 4096×12288, `down` 12288×4096) and the LM head 4096×151,936, for token counts `M ∈ {1, 2, 4, …, 32768}` (powers of two), dtypes BF16 and FP16 (T2) plus TF32 and FP32 (`torch.set_float32_matmul_precision('high'|'highest')`) for the dtype ladder at `M=N=K=8192`; T1 adds FP8 via `torch._scaled_mm` with E4M3 inputs and per-tensor scales. Plot `fig_lm_hw_gemm`: x = `M` (log2), y = achieved TFLOP/s (log), one line per weight shape, dashed roofline `min(P, I·β)` with datasheet values and a second dashed with `P_eff, β_eff` from this section's calibration, vertical line at `M* = R·b/2` (the batch at which the skinny matmul crosses the ridge, `eq_matmul_intensity` with `B=M`). Raw sizes that are not multiples of 8 (e.g. `K=4097`) are measured in a side table to show the alignment penalty.
   2. *Wave and tile quantization.* `M=4096, K=4096`, `N` from 1024 to 8192 in steps of 64: achieved TFLOP/s against `N` shows the sawtooth where the output-tile count crosses a multiple of the SM count (A100 has 108 SMs per the whitepaper; the notebook reads `torch.cuda.get_device_properties(0).multi_processor_count`). The notebook does *not* assume the tile shape: it reports the period of the sawtooth and compares it with `⌈M/Tm⌉⌈N/Tn⌉ / n_SM` for `(Tm,Tn) ∈ {(128,128),(128,256),(256,128)}`.
   3. *The memory ladder.* `d2l.hw_copy_bench` for `kind='copy'` and `kind='read'` over sizes 256 KiB to 8 GiB (powers of two), `read` implemented as `x.sum()` replayed 100× inside a CUDA graph so launch cost is amortized and a working set smaller than L2 is served from L2 after the first pass. Plot `fig_lm_hw_ladder`: x = bytes (log2), y = GB/s (log); dashed horizontal at the datasheet HBM bandwidth; vertical band at the L2 capacity; annotate the three regimes (launch-latency rise, L2 plateau, HBM plateau). The `sec_hardware` exercise on this sweep is acknowledged and its function is reused.
   4. *Shared memory and latency (optional, `nvcc`).* (a) A 25-line CUDA kernel in which each thread block loads a tile to shared memory and re-reads it `ITER` times with a conflict-free access pattern; bytes read from shared memory per second against `n_SM × 128 B/clk × boost clock` (the per-SM shared-memory width and clock come from the whitepaper and the dissection paper; the cell prints which source each constant has). (b) A pointer-chase kernel (single thread, random cyclic permutation) over working sets 4 KiB to 4 GiB giving latency per load for L1, L2 and HBM, with the method of Jia et al. and the measured table of the Ampere dissection paper as the reference for comparison. Plot `fig_lm_hw_latency`: x = working-set bytes (log2), y = ns per load (log), annotated plateaus and the L1/L2/HBM capacities. All of 4 is behind `--latency` and `--smem` flags.
7. **Reading the datasheet again (subsection).** A table `tab_lm_hw_datasheet_vs_measured`: datasheet versus measured for BF16 peak, TF32, FP32, HBM bandwidth, L2 read bandwidth, shared-memory bandwidth; column "fraction of datasheet". Direction of expected outcome: streaming bandwidth and large-matmul throughput below the datasheet by a margin the cell measures; the *book* states that a well-shaped streaming kernel lands within tens of percent (`subsec_hw-bytes`), which is the claim being tested, not assumed.

#### Derivations

**D1. How much on-chip memory does it take to reach the ridge, and how does the split between SRAM and matrix units move it?**

*Statement.* For a chip with off-chip bandwidth `β` (set by perimeter) and matrix peak `P`, a GEMM kernel that keeps an output tile of `Tm×Tn` accumulators on chip reaches off-chip intensity `I(Tm,Tn) = 2·Tm·Tn / (b·(Tm+Tn))` FLOP/byte; reaching the compute roof needs `I ≥ R = P/β`; the on-chip bytes this tile needs scale as `R²`. Raising `P` by a factor `f` at fixed `β` therefore multiplies the on-chip memory needed per tile by `f²`.

*Assumptions.* Operand element size `b` bytes (2 for BF16), accumulators 4 bytes (FP32); each operand slab is fetched from off-chip memory once per tile; the `K` extent is large; staging buffers have `s_d` stages of depth `Tk`; no reuse of a slab across concurrently running tiles (relaxed in step 6).

*Steps.*
1. Per output tile and per K-slab of depth `Tk`: loads `b·Tk·(Tm+Tn)` bytes (one `Tm×Tk` slab of A and one `Tk×Tn` slab of B); FLOPs `2·Tm·Tn·Tk`.
2. Intensity `I = 2·Tm·Tn / (b·(Tm+Tn))`. For `Tm=Tn=T`: `I = T/b`.
3. Footprint per resident tile: `S_tile = 4·Tm·Tn + s_d·b·Tk·(Tm+Tn)` bytes (accumulators plus staging).
4. Roof condition `I ≥ R` gives `T ≥ b·R` and `S_tile ≥ 4·b²·R² + 2·s_d·b²·Tk·R`, i.e. `S_tile ∝ R²` for the accumulator term.
5. Scaling: `R = P/β`. Fix `β` (perimeter) and raise `P` by `f`: `R → fR`, `T → fT`, accumulator bytes `→ f²·` . Equivalently, replace HBM by memory `g` times slower (GDDR or LPDDR): the same `g²` growth.
6. Cross-tile sharing: if `ρ` concurrently running tiles use the same operand slab and a shared level (L2, or TMA multicast within a thread-block cluster) serves it from on-chip, off-chip traffic falls by `ρ` and the condition becomes `ρ·I(Tm,Tn) ≥ R`; so the required `ρ ≥ R/I(Tm,Tn)` with `I` bounded by the register file: `4·Tm·Tn ≤ RF` per SM.
7. Area split (stylized): let matrix area fraction `m` give `P = m·A·ρ_P` (ρ_P FLOP/s per mm²) and SRAM area fraction `s` give capacity `S = s·A·σ`; the requirement `S ≥ c·R²` with `c = n_tile·4b²` becomes `s ≥ κ m²` with `κ = c ρ_P² A /(σ β²)`; with `m + s ≤ 1` the largest usable matrix fraction is `m* = (−1 + √(1+4κ)) / (2κ)`. Halving `β` quadruples `κ` and lowers `m*`.

*Final formulas.* `I = 2TmTn/(b(Tm+Tn))`; `T* = bR`; `S_tile* ≈ 4b²R²`; `ρ ≥ R/I`; `m* = (−1+√(1+4κ))/(2κ)`.

*Numerical check (cell, T0).* (a) A NumPy tile-traffic counter loops over output tiles for random `(M,N,K,Tm,Tn)` and counts bytes loaded; assert it equals `b·K·(Tm+Tn)·(#tiles)` and that FLOPs/bytes equals the formula (exact, integer arithmetic). (b) With `R` from `derive_ratios` for A100, H100 and B200, `b=2`, and the register-file size per SM from the whitepapers (parameters in a dict with source strings; A100 and H100 whitepapers state their register-file sizes), compute `T_max = √(RF/4)`, `I_max = T_max/b` and `ρ_required = R/I_max`; the cell prints them and asserts `ρ_required > 1` for the Hopper row, which is the quantitative reason for L2 reuse and multicast (check against the [Hopper in-depth blog](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/), which names thread-block clusters and TMA multicast; the blog does not give `ρ`, so this is a consistency argument, stated as such). (c) A brute-force grid over `m` confirms `m*` to 1e-9 for three values of `κ`.

**D2. Sweep time is a floor on per-token latency for a model that fills memory.**

*Statement.* If the weights `W` and the KV cache of `b` sequences of length `L` together fill capacity `C` (`W + b·L·k_KV = C`), every decode step reads at least `C` bytes, so the step time is at least `τ = C/β`, independent of `b`; aggregate throughput is then `b/τ`, and per-user throughput `1/τ`. With `c` chips in tensor parallelism, `C → cC` and `β → cβ`, so `τ` is unchanged: the floor is a property of the *chip*, not of the number of chips.

*Assumptions.* Memory-bound decode (`eq_hw_decode_bound`); perfect bandwidth use; no communication cost; all state is read once per step (dense model, no cache bypass).

*Steps.* (1) `t_step ≥ (W + b·L·k_KV)/β` (`eq_hw_decode_bound` generalised to a batch). (2) Fill condition makes the numerator `C`. (3) Hence `t_step ≥ C/β`. (4) Per-user rate `u ≤ β/C`; per-GPU throughput `T = b·u ≤ b·β/C`. (5) With `c` chips sharing every read evenly, both `C` and `β` scale by `c`. (6) The only chip-level levers are a smaller `C/β` (SRAM, fewer GB per TB/s) or a model that does not fill memory. (7) The `sec_hardware` exercise's result `u ≤ β/(f·C)` for a weight-only fill is the `b=0`, `f<1` case of this statement.

*Numerical check.* For each chip in `chip_specs` with `mem_bw`, compute `τ` and the implied per-user ceiling `1/τ`; for Qwen3-8B (`W = 2N`, `k_KV` from the config), compute the maximum `b` at `L=8192` filling memory and assert `W + b·L·k_KV ≤ C`, then assert `t_step(b_max) ≥ τ·(1−ε)` using `eq_hw_decode_bound`; for `c = 1, 2, 4, 8` assert that `τ` is constant.

**D3. How many SRAM chips does a model need, and what does it cost per token?**

*Statement.* An SRAM-only chip with on-chip capacity `S_c` and SRAM bandwidth `β_s` holds a model of `W + b·L·k_KV` bytes on `N_c = ⌈(W + b·L·k_KV)/(η·S_c)⌉` chips, where `η<1` reserves activation and buffer space. If the layers are pipelined across the chips and enough concurrent sequences fill the pipeline, throughput is `1/t_stage` tokens/s with `t_stage ≈ max(S_c/β_s, t_link)`, and the per-user latency of one token is `N_c·t_stage`. Cost per token, in chip-seconds, is `N_c·t_stage` for a full pipeline.

*Steps.* (1) Capacity: total bytes over usable bytes per chip. (2) A stage holds `1/N_c` of the weights; reading them from SRAM takes `S_c/β_s` at most (the sweep time of one chip). (3) A token traverses `N_c` stages: latency `N_c·t_stage`. (4) A new token enters every `t_stage` once the pipeline is full: throughput `1/t_stage`. (5) The pipeline is full only with at least `N_c` independent sequences in flight; with fewer, throughput is `n_seq/(N_c·t_stage)`. (6) Chip-seconds per token `= N_c·t_stage/min(1, n_seq/N_c)`. (7) Compare with an HBM chip at batch `b`: chip-seconds per token `= t_step(b)/b`. The ratio is the cost penalty or bonus of SRAM before prices enter; prices enter in 5.6.

*Numerical check.* For Qwen3-8B and for a 70B-parameter dense config (parameter count from a config object defined in the cell), compute `N_c` for the Groq and Cerebras rows (their `mem_gb` as `S_c`, `η=0.8` as an explicit parameter), and chip-seconds per token against the A100 PCIe 40 GB row at `b = 1, 8, 64` (at a context short enough that `b = 64` fits 40 GB; the cell asserts feasibility and prints `b_max`); assert monotonicity in `η` and that `N_c·S_c·η ≥ W`. Datasheet inputs are those of `chip_specs`; the results are stated as model outputs, never as vendor facts.

#### From-scratch implementation

Components (≤ ~150 lines total, excluding plotting):
- `load_specs(path) -> DataFrame` and `check_specs(df)`: asserts source and date per row, units, dense-only flags, reproduction of `tab_gpu_specs` (≈ 25 lines).
- `derive_ratios(df)` (≈ 15 lines).
- `gemm_tile_traffic(M, N, K, Tm, Tn, b)`: NumPy loop returning (flops, bytes) (≈ 15 lines); `rho_required(R, rf_bytes, b)` (≈ 5 lines); `m_star(kappa)` (≈ 5 lines).
- `sram_chips_needed(model_bytes, S_c, eta)` and `chip_seconds_per_token(...)` (≈ 20 lines).
- `hw_matmul_bench`, `hw_copy_bench` (≈ 40 lines; CUDA-event timing, graph option).
- `BenchLog` and `env_fingerprint` (≈ 40 lines, may live in a helper cell marked `#@save`).

Tests: tile counter equals the formula exactly (integer arithmetic; seed 0, 200 random shapes); `derive_ratios` reproduces the book's ridge row of `tab_gpu_specs` (±1 FLOP/B, the book rounds); `m_star` against grid search (1e-9); `hw_matmul_bench` against `torch.utils.flop_counter.FlopCounterMode` FLOP count for one shape (exact); `hw_copy_bench` byte accounting against `x.numel()·element_size()`.

#### In production

- `torch.cuda.get_device_properties` and `pynvml` give SM count, memory, clocks, power; `nvidia-smi --query-gpu=...,clocks.max.sm,clocks.max.mem,power.limit` is the log's authority for clocks.
- Nsight Compute (`ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct`) is the library practitioners use to answer "which budget binds a given kernel". The notebook runs it on one 8192³ matmul **if** `ncu` is installed and counters are permitted, and prints the three percentages next to the roofline estimate; otherwise it prints the committed output. The from-scratch roofline numbers are the oracle: the cell asserts that the kernel is on the compute side of the ridge and that DRAM throughput is below the compute percentage.
- cuBLAS versus CUTLASS: `torch.matmul` dispatches to cuBLAS; the notebook does not claim a tile shape and reads kernel names from `torch.profiler` output (names such as `ampere_bf16_s16816gemm_…` encode the tile shape) to compare with D1's candidate tiles.

#### Headline experiment

*Hypothesis.* On one A100, for LLM-shaped GEMMs, achieved TFLOP/s as a function of token count `M` follows the roofline of the *calibrated* chip, `min(P_eff, I(M)·β_eff)`, with the knee near `M* = R_eff·b/2`, and `P_eff, β_eff` are below the datasheet by fractions the notebook reports. A shape-dependent departure (small `N`, misaligned `K`) exists and is explained by quantization, not by bandwidth.

*Setup.* One A100 PCIe 40 GB (datasheet row `A100 PCIe 40 GB`), `torch==2.15.0`, BF16 and FP16, shapes from Qwen3-8B (above), `M` powers of two from 1 to 32,768, CUDA events, warm-up 10, 30 timed repeats each, 3 independent processes. Clocks recorded; locked if permitted. The card's 250 W limit (datasheet) is below the SXM part's 400 W, so a long matmul is more likely to run power-limited here (a hypothesis to test, not an assumption: the log records clocks and throttle reasons). One seed (inputs `torch.randn`, seed 0; values do not affect throughput except through power: also run once with zeros to check, since data-dependent power can move clocks).

*Procedure.* Run all shapes, take per-shape median of medians, fit `P_eff` and `β_eff` by non-linear least squares of `t = max(F/P_eff, B/β_eff) + t_0` (three parameters) on `M ≤ 32768` points; report residuals. Compare with the datasheet roofline.

*Metrics.* Achieved TFLOP/s and fraction of datasheet; fitted `(P_eff, β_eff, t_0)` with bootstrap 95% intervals (resample the repeats, 2,000 draws, `sec_mdl-statistics` method, cited not re-derived); position of the knee against `M*`. The datasheet-only knee `M* = R·b/2` is 201 tokens for this row (`R` = 312/1.555 = 200.6, `b` = 2 bytes; computed) against 153 for the SXM 80 GB row; the measured knee is compared with the calibrated `R_eff`, not with either.

*Plot specification.* `fig_lm_hw_gemm` as in item 6.1: x log2 `M`, y log TFLOP/s, 5 lines (weight shapes), two dashed rooflines (datasheet, calibrated), vertical `M*`; error bars are MAD over repeats; legend names shapes as `4096→12288`. Companion `fig_lm_hw_ladder`.

*Expected outcome.* A direction only: the calibrated roofline fits the large-`M` and small-`M` ends; the middle shows a transition more gradual than the sharp roofline corner (to be measured; do not quote). The Ampere dissection paper (arXiv 2208.11174) reports measured throughputs and memory behaviour of A100 against peak; the notebook compares its table with the committed log and quotes the paper's value only after reading it at authoring time.

*Noise check.* 3 processes × 30 repeats; a claim about a gap (e.g. FP16 versus BF16, or aligned versus misaligned `K`) requires a gap > 3× the larger MAD; one idle-GPU zero-input control for the data-dependence of power.

*T1 reduction.* Same code on the RTX 4090 with 4096-wide shapes only; the same plots overlay the T1 and T2 logs (log-log lines with a different dash style), and the notebook prints the 4090's measured BF16 and FP8 fractions of its datasheet values (FP8 is the new rung that A100 lacks; ratio FP8/BF16 is measured, datasheet predicts 2×).

*Artifacts committed (T2+).* `logs/5.1/matmul_a100.jsonl`, `copy_a100.jsonl`, `smem_a100.jsonl`, `latency_a100.jsonl`, `ncu_a100.txt` (if available), `chip_specs.csv`, the four figures as SVG, bundled as `lm-hw-a100-microbench`.

#### Figures and tables

- `fig_lm_hw_budget`: stylized floorplan and budget annotations (matplotlib patches; dict of illustrative fractions, labelled as such).
- `tab_lm_hw_memmenu`: HBM, GDDR, LPDDR, DDR5, SRAM compared (bandwidth, capacity, interface, rank cost, reason). Cell computes per-stack HBM bandwidth from the JEDEC rate.
- `tab_lm_hw_chips`: the spec table (rendered subset of columns) with derived ratios; the full CSV is the artifact.
- `fig_lm_hw_plane`, `fig_lm_hw_netplane`: design plane and network-bytes-per-FLOP panel.
- `fig_lm_hw_gemm`, `fig_lm_hw_ladder`, `fig_lm_hw_latency`: the three microbenchmark plots.
- `tab_lm_hw_datasheet_vs_measured`: datasheet against measured, with the fraction column.
- `eq_lm_hw_areabudget`, `eq_lm_hw_tileintensity` (`I = 2TmTn/(b(Tm+Tn))`), `eq_lm_hw_sweepfloor` (`t_step ≥ C/β`).

#### Pitfalls and misconceptions

- "Peak FLOP/s" depends on the format, on sparsity and on the accumulation type; mixing sparse and dense figures is the commonest table error (the A100 datasheet prints both side by side). `chip_specs` stores dense only.
- Vendor "memory bandwidth" for multi-die parts is aggregate over dies; one die cannot read the other die's HBM at that rate.
- A bandwidth number measured with a copy kernel counts read plus write; a read-only kernel counts read only. State which one the plot uses.
- TF32 is a compute format with 32-bit storage (`subsec_hw-gpu`): the dtype ladder shows a speedup in FLOP/s without a memory saving.
- SRAM-only chips quote enormous bandwidth per chip; the per-*model* picture divides by the number of chips needed to hold it (D3).
- Datasheet TDP is not what a run draws; clocks fall when power-limited (5.6 returns to this).
- A microbenchmark of a library GEMM measures the library's tile choices, not the hardware's limit.

#### Summary (draft bullets)

- A chip is a budget of area, power and perimeter; compute grows with area while off-chip bandwidth grows with perimeter, so the split between matrix units and SRAM and the choice of memory technology define the design.
- Five numbers (peak FLOP/s per format, memory bandwidth, capacity, scale-up and scale-out bandwidth) and three ratios (ridge, sweep time `C/β`, network bytes per FLOP) place any chip on a plane; GPUs, TPUs, GDDR parts and SRAM-only chips occupy different regions.
- Reaching the ridge with a tiled GEMM needs on-chip memory that grows like the square of the ridge point unless tiles share operands through L2 or multicast.
- Capacity over bandwidth is a per-step latency floor for any model that fills memory, and adding chips in tensor parallelism does not lower it; only SRAM-like designs do, at the price of many chips per model.
- Measured on one A100, the datasheet is a ceiling reached only by well-shaped kernels; calibrated `P_eff` and `β_eff`, not datasheet values, are the inputs to the models of 5.2.

#### Exercises

1. **(check)** Two chips have equal `P16` and equal `β`; one has 4× the memory. Which of the three ratios differ, and which workloads does each favour? *Sketch:* only `C/β` differs (4×); the larger-memory chip holds larger models or batches/caches (training state, long-context decode) but has a 4× longer sweep floor per step when full; compute-bound prefill is unaffected.
2. **(check)** Why does the `f²` growth of D1 not apply to an operation of intensity far below `R` (e.g. elementwise)? *Sketch:* such operations are bandwidth-bound whatever the tile; the tile argument is about reaching the roof, which they never attempt.
3. **(derive)** Extend D1 to rectangular tiles with `Tm ≠ Tn`: for a fixed footprint `Tm·Tn = S/4`, show the intensity is maximized by `Tm = Tn` and give `I` as a function of the aspect ratio `r = Tm/Tn`. *Sketch:* `I = 2TmTn/(b(Tm+Tn))` with `TmTn` fixed is maximal when `Tm+Tn` is minimal, i.e. `Tm=Tn`; `I(r) = (2/b)·√(TmTn)·√r/(1+r)`, falling by `2√r/(1+r)`; for `r = 2` the loss is about 6%.
4. **(derive)** Prove that with `c` chips in tensor parallelism the sweep-time floor of D2 is unchanged, and find what *does* change when the model is *not* split evenly (e.g. 40% of the weights on one chip). *Sketch:* the step time is the maximum over chips of bytes over `β`; uneven splits raise the floor to `max_i C_i/β`.
5. **(measure)** Predict, then measure, the matmul sawtooth period in `N` for `M=K=4096` on your GPU from its SM count and three candidate tile shapes. Which shape is consistent, and how does the period move when `M` is halved? Compare the committed A100 log.
6. **(measure)** On the build box, measure the FP8-to-BF16 throughput ratio for `8192³` with `torch._scaled_mm` and compare with the datasheet ratio; explain any shortfall with D1 (bytes halve, but accumulate in higher precision) and with the scale-handling overhead. State your noise check.
7. **(open)** Add a row to `chip_specs` for a chip released after this chapter was written. Fill every column from vendor pages, flag every missing dense figure, place it on the design plane, and write a paragraph on which workload (training, prefill, decode) it is designed for and which number in the row supports your claim.

#### Further reading

- Papers: [Demystifying the Ampere Architecture](https://arxiv.org/abs/2208.11174) (microbenchmark method and A100 table, the reference for narrative item 6, part 4); [Benchmarking and Dissecting the Hopper Architecture](https://arxiv.org/abs/2402.13499) (what changes with TMA and clusters; D1 step 6); [Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks](https://arxiv.org/abs/2507.10789) (new in this chapter; verified title; used for the next-generation comparison); [Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking](https://arxiv.org/abs/1804.06826) (new; the pointer-chase latency method used in narrative item 6, part 4(b)); [AI and Memory Wall](https://arxiv.org/abs/2403.14123); [Chiplet Cloud](https://arxiv.org/abs/2307.02666) (design-space view of SRAM-heavy chiplets, D3); [WaferLLM](https://arxiv.org/abs/2502.04563) (wafer-scale decode, D3); [TPU v4](https://arxiv.org/abs/2304.01433).
- Vendor and blog material: Google, [Ironwood](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/ironwood-tpu-age-of-inference/); NVIDIA, [Rubin CPX announcement](https://nvidianews.nvidia.com/news/nvidia-unveils-rubin-cpx-a-new-class-of-gpu-designed-for-massive-context-inference); [AMD Instinct MI350](https://www.amd.com/en/products/accelerators/instinct/mi350.html); [AWS Trainium](https://aws.amazon.com/ai/machine-learning/trainium/); Groq, [The LPU Explained](https://groq.com/blog/the-groq-lpu-explained); Cerebras, [Inference 3× faster](https://www.cerebras.ai/blog/cerebras-inference-3x-faster); NVIDIA, [Hopper architecture in depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) and [Ampere architecture in depth](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/) (new; both returned HTTP 200).
- Teaching: [Scaling Book: GPUs](https://jax-ml.github.io/scaling-book/gpus/); [Modal GPU Glossary](https://modal.com/gpu-glossary); [GPU MODE lectures](https://github.com/gpu-mode/lectures) (profiling and microbenchmark lectures).
- Code: [NVIDIA nvbandwidth](https://github.com/NVIDIA/nvbandwidth) (used from 5.3), [CUTLASS](https://github.com/NVIDIA/cutlass) (new; verified; the tile shapes of D1), PyTorch `torch.utils.flop_counter`.

#### Risks and open decisions

- *A vendor page changes or blocks scripted access.* Fallback: the committed CSV holds the transcribed values with `retrieved`; CI checks only the page title and a stored hash of the cited paragraph; a mismatch opens an issue, it does not fail the build.
- *SRAM-chip dense 16-bit figures are not published.* Fallback: `NaN`, plotted in the side strip; D3 needs only capacity and bandwidth.
- *`nvcc` or `ncu` unavailable on the T2 host.* Fallback: committed logs for the shared-memory, latency and `ncu` cells; the notebook runs the other three benchmarks live.
- *Clocks cannot be locked on a shared host.* Fallback: record clocks; raise the claim threshold to 5× MAD for any ratio of two series that were not run back-to-back.
- *The T2 host is not the PCIe 40 GB part.* The datasheet has separate columns for 40 GB and 80 GB, PCIe and SXM; `check_specs` compares `get_device_properties().name` and total memory with the `A100 PCIe 40 GB` row (the device-name string is read on the T2 host at authoring, not assumed here) and refuses to draw the dashed datasheet line on a mismatch; the SXM 80 GB row then applies if that is the part found.

### 5.2 Training, prefill and decode: why their hardware needs differ

- **Label:** `sec_lm_workload_hardware` · **Tier:** T2 (one A100 40 GB PCIe, the `A100 PCIe 40 GB` row of `chip_specs`; the MoE series on the T2 pair) · **Reduced T1 config:** RTX 4090 with `Qwen/Qwen3-0.6B` (training step, prefill, decode) and `Qwen/Qwen3-1.7B` (decode), grid of 12 points, same functions behind `config.tier='T1'` · **Budget:** T2 ≈ 3 h wall-clock, peak ≤ 36 GB (0.9 × 40 GB; training points predicted above it are dropped and listed) and 16.4 GB of 8B inference weights (2·8,190,735,360 bytes, computed) plus KV; T1 ≈ 15 min, ≤ 14 GiB · **Codex units:** 4.8, 14.1
- **Builds on (book):**
  - `subsec_hw-two-workloads` and `eq_hw_decode_bound`: the bytes-per-token bound `tokens/s ≲ β/(2N + KV bytes)` is *recapped in one paragraph and generalised*, not re-derived; the book's `tab_rules_of_thumb` is the starting point.
  - `sec_kv-cache` (`eq_kv-cache-bytes`, `subsec_kv-cache-prefill-decode`, GQA): KV bytes per token is taken as given; `ServingModel.kv_bytes_per_token` is tested against it.
  - `subsec_mp-anatomy` (16 bytes per parameter with Adam): training state bytes; this section adds the *requirements card* but does not re-derive memory anatomy.
  - `sec_moe`: total versus active parameters (the layer is taken as given).
  - `sec_scaling-laws` (`6ND`): FLOPs per token.
  - `sec_fast_transformer`: the tokens/s benchmark contract.
  - Not re-taught: roofline (`eq_roofline`), ridge point, prefill versus decode intensity at one batch size, FlashAttention.
- **Builds on (this part):** 5.1 (`d2l.chip_specs`, `d2l.derive_ratios`, `d2l.hw_matmul_bench`, `d2l.hw_copy_bench`, `d2l.BenchLog`, the `lm-hw-a100-microbench` logs for `P_eff`, `β_eff`), 1.1 (`d2l.load_hf`, `d2l.config_memory`; `d2l.config_memory` is the oracle for KV bytes and weight bytes), 1.2 (`d2l.load_qwen3` for the T1 decode path if the Hugging Face static-cache path fails), 4.2 (`d2l.flops_per_token` and `d2l.mfu`, which `TrainStepModel.flops_per_token` and `.mfu` delegate to). Forward: 5.6 (cost per token), 13.4 (disaggregation), A5.
- **Exports (`#@save`):**
  - `d2l.hw_calibrate(device=None, log=None) -> dict`: runs the 5.1 benchmarks and returns `{P_eff_bf16, beta_eff, alpha_k, ...}` with the environment fingerprint (the same dictionary schema as A5's `calib.json`).
  - `d2l.ServingModel(cfg, chip, calib=None, dtype_bytes=2, kv_bytes=2, mem_util=0.9, kv_bytes_override=None)`: `.n_params() -> (total, active)`, `.weight_bytes()`, `.kv_bytes_per_token()`, `.max_batch(L)`, `.decode_step(b, L)`, `.prefill_time(n, b=1)`, `.b_star(L)`, `.L_crit()`, `.frontier(L, batches) -> DataFrame` with columns `b, user_tps, gpu_tps, regime` (`'bw'|'compute'|'infeasible'`), `.gpu_s_per_mtok(b, L)`. `cfg` is a Hugging Face config for dense Qwen3-style or Qwen3-MoE; `kv_bytes_override` supports MLA or hybrid caches (fed by `d2l.config_stats`, 14.1, and used by 13.x and 14.x). `ServingModel(...).decode_step(b, L)` is the canonical decode step-time model `eq_lm_hw_decode_step`, used by Chapters 12, 13 and 14; no other chapter defines a separate decode-time function (there is no `d2l.decode_step_time`).
  - `d2l.TrainStepModel(cfg, chip, calib=None, recipe='bf16_pure'|'mixed16')`: `.state_bytes_per_param()`, `.flops_per_token(s)`, `.step_time(b, s, ckpt=False)`, `.mfu(tokens_per_s, s)`.
  - `d2l.moe_expert_coverage(E, K, b) -> float`: expected number of distinct experts selected by `b` tokens under uniform top-`K` routing.
  - `d2l.hw_train_bench`, `d2l.hw_prefill_bench`, `d2l.hw_decode_bench`: the measurement harness of the A5 protocol (CUDA events, static cache, CUDA-graph decode), returning dictionaries and logging through `d2l.BenchLog`. A5's `measure.py` imports them unchanged, so a student's log and the section's log come from the same code.
  - Data: `DATA_HUB['lm-hw-a100-workloads']` (JSONL, one record per grid point, with the fingerprint; the MoE points from the T2 pair carry `tier='T2pair'`) and `DATA_HUB['lm-hw-inferencex-snapshot']` (CSV; columns documented below).
- **Models and data:** `Qwen/Qwen3-8B` (Apache-2.0; 36 layers, hidden 4096, 32 query and 8 KV heads, head dim 128; `k_KV = 2·36·8·128·2 = 147,456` bytes per token in bf16, which the cell recomputes from the config); random initialization from `AutoConfig` for all training and TTFT timings (no weights downloaded: throughput does not depend on weight values; one control with real weights at one point); real weights for decode as a cross-check at `b=1`. `Qwen/Qwen3-30B-A3B` (Apache-2.0, 48 layers, 128 experts, top-8, 4 KV heads: `k_KV = 98,304` bytes per token, ≈ 61 GB in bf16) for the MoE series, real weights, on the T2 pair only: 61 GB does not fit one 40 GB card, so the model is split layer-wise over the two GPUs (`device_map`; the cards work in turn, so one card's bandwidth bounds the step, and the hand-off is one `b × 4096 × 2`-byte copy per step, which the notebook times once). `Qwen/Qwen3-0.6B` for T1. A snapshot of the public SemiAnalysis InferenceX results. The repository `SemiAnalysisAI/InferenceX` is Apache-2.0 (its [`LICENSE`](https://raw.githubusercontent.com/SemiAnalysisAI/InferenceX/main/LICENSE), HTTP 200, and the GitHub licence field were re-read on 2026-10-29; copyright line "2025 SemiAnalysis LLC, Advanced Micro Devices, NVIDIA CORPORATION"), so the trimmed CSV is committed with attribution, a copy of the licence, a statement of the changes (rows and columns kept) and the upstream commit hash (Apache-2.0 section 4); "In production" gives the data-location check.
- **Packages:** `transformers` (pin; needs `StaticCache`), `torch==2.15.0` (`torch.cuda.CUDAGraph`, `torch.utils.flop_counter`), `scipy` (`curve_fit`), `pandas`.

#### Narrative arc

1. **The question.** One cell prints, for Qwen3-8B on the A100, the datasheet-only roofline prediction and the measurement for three quantities: training tokens/s (depth-truncated model, below), TTFT at `n=2048` and decode tokens/s at `b=1` and `b=32`, `L=2048` (`b=64` at `L=2048` needs 35.7 GB of weights and KV, which the 36 GB budget cannot hold once the reserve is counted). The datasheet-only first line for `b=1` is `W/β` = 16.38 GB / 1,555 GB/s = 10.5 ms per token, about 95 tokens/s (computed from the row; the measurement is to be made). The symptom is the ratio measured/predicted for each (to be measured; the text says only that the three ratios differ, which is the point: three workloads, three different gaps). The book's decode bound `β/(2N+KV)` is the first line of the model; the chapter question is "what else is on the right-hand side?".
2. **What each workload asks of the machine (subsection "A requirements card").** Table `tab_lm_hw_requirements` with rows training, prefill, decode and columns FLOPs per token, bytes of state or traffic per token, capacity driver, network driver, binding resource at typical operating points. Training: `6N_mm + 12·n_layers·n_heads·d_head·s` FLOPs per token (PaLM convention, causal masking not halved; the cell states the convention and checks what `FlopCounterMode` counts), state bytes per parameter under two recipes (`mixed16`: the book's 16–20; `bf16_pure`: 8, which the measurement uses because the depth-truncated model fits one 40 GB card), activation bytes per token per layer (take from `sec_memory_precision`; not re-derived), communication per step `2N·bytes` per replica (`eq_dp_cost`). Prefill: `2N_act·n` FLOPs plus an attention term quadratic in `n`, weights read once per forward, KV written. Decode: `2N_act·b` FLOPs, `W + b·L·k_KV` bytes, capacity `W + b·L·k_KV`. Net result: training wants FLOPs, capacity and network together; prefill wants FLOPs; decode wants bandwidth and capacity.
3. **Training on one chip (subsection).** MFU defined as in the PaLM paper (cited): `MFU = (tokens/s × FLOPs per token)/P_peak`, HFU if recompute is counted. Measurement on the A100 of `forward+backward+AdamW` for depth-truncated Qwen3-8B (`n_layers ∈ {2, 4, 6}`, full width, whose `bf16_pure` state is 13.0, 16.1 and 19.2 GB (computed at 8 bytes per parameter); step time must be linear in `n_layers` — checked by a three-point fit with the residual printed; extrapolate to 36). The full 36-layer model cannot be measured on 40 GB (its 65.5 GB of state alone exceeds the card), so the extrapolation to 36 layers is reported as an extrapolation, with its fit interval. The cell prints MFU against datasheet and against `P_eff` (two different denominators; says which is which).
4. **Prefill (subsection).** TTFT versus prompt length `n ∈ {128, 512, 2048, 8192, 32768}` at batch 1: where the matmul term and the attention term cross (D5), where TTFT stops being weight-read-bound (small `n`) and becomes compute-bound, and why prefill-only parts can trade HBM capacity for cheaper memory (use D1 of 5.1 to say what that costs in SRAM).
5. **Decode (subsection).** The batch-and-context grid; the roofline decode model with its three refinements (weights amortised over batch, KV not amortised, launch overhead); regimes; the existence of `L_crit` beyond which no batch reaches compute-bound (D1); maximum batch (D2).
6. **The throughput–interactivity frontier (subsection).** `T(u)` per GPU against `u` per user (D3); how context shifts it; read `gpu_s_per_mtok` off the curve (GPU-seconds per million tokens: multiply by a price per GPU-second to get dollars, 5.6; no price here). Measured A100 points over the model curve (`fig_lm_hw_frontier`).
7. **MoE (subsection).** Total parameters set capacity, active parameters set compute, *touched* parameters set traffic: `moe_expert_coverage` (D4) versus the number of distinct experts counted from real routing of `Qwen3-30B-A3B` at `b ∈ {1, 4, 16, 64}` (on the T2 pair, layer-split; routing counts and the decode step time are the only measurements).
8. **Other chips from datasheets (subsection).** Frontier of H100, B200 and an SRAM-pipeline design (D3 of 5.1) from `chip_specs` rows, same model, same `L`; comparison with the InferenceX snapshot (procedure below) and with vendor claims for SRAM chips (labelled as claims).
9. **Consequences for system design (subsection).** Separate prefill and decode pools (pointer to 13.4), SRAM parts for latency-critical decode, large-memory parts for long context, why a training cluster is rarely the cheapest place to serve: each stated as a sentence plus the single number from the notebook that supports it (to be filled from the cells, not from this blueprint).

#### Derivations

**D1. The compute-bound batch and the context beyond which decode never is.**

*Statement.* For a dense model with `N_act` active parameters, weight bytes `W`, KV bytes per token `k` and context `L`, the decode step time is `t(b,L) = max(2·N_act·b/P, (W + b·L·k)/β)`. Let `R = P/β`. Decode becomes compute-bound at the batch `b* = W / (2N_act/R − L·k)`, which exists only if `L < L_crit = 2N_act/(R·k)`; for `L ≥ L_crit` the step is bandwidth-bound at every batch.

*Assumptions.* Attention FLOPs are neglected because attention decode has intensity `≈ 2g/bytes_per_element` FLOP/byte (`g = n_heads/n_kv_heads`, the GQA group size), far below `R`, so its FLOPs hide behind its own KV reads; weights are read once per step regardless of `b`; per-token KV reads scale with `b·L`; launch overhead and `P_eff, β_eff` are added after, not inside, this derivation.

*Steps.* (1) Compute time `t_c = 2N_act·b/P`; memory time `t_m = (W + bLk)/β`. (2) Equality: `2N_act·b/P = (W + bLk)/β`. (3) Multiply by `β`: `b·(2N_act/R) = W + bLk`. (4) Collect `b`: `b·(2N_act/R − Lk) = W`. (5) Solve: `b* = W/(2N_act/R − Lk)`. (6) The denominator is positive iff `L < 2N_act/(R·k)`; define `L_crit` as that value. (7) For `L ≥ L_crit` the memory slope per added sequence (`Lk/β`) exceeds the compute slope (`2N_act/P`) so `t_m > t_c` at every `b ≥ 1`. (8) Special case `L=0`: `b* = W·R/(2N_act) = R·W/(2N_act) = R·(bytes per weight)/2` for a dense bf16 model (`W = 2N_act`), i.e. `b* = R`: the batch equals the ridge point (the book's skinny-matmul intensity `2B/b_w = R`).

*Numerical check (cell).* For Qwen3-8B with bf16 weights, `N_act` and `k` from `AutoConfig` and `R` from the A100 datasheet row (as the first model) and then from `calib` (the second): (a) assert the closed form `b*` equals the root found by bisection on `t_c − t_m` to 1e-9 for ten values of `L < L_crit`; (b) print `L_crit` for each chip in `chip_specs` (the cell computes; expected order a few hundred tokens for an 8B dense model with 8 KV heads on A100- and H100-class chips: about 550 for the A100 PCIe 40 GB row with `N_act` = 8.19e9 and `k` = 147,456 bytes, about 510 counting matmul parameters only, computed from the row; the cell confirms or corrects, and the author writes the sentence from the printed value); (c) assert that for `L > L_crit` `t_m(b) > t_c(b)` for `b ∈ {1,…,256}`; (d) the same with an MLA-style small `k` (override `k` to a tenth) shows `L_crit` moving by 10×, linking to 12.4 and 14.3.

**D2. Maximum batch and the capacity wall.**

*Statement.* `b_max(L) = ⌊(μ·C − W − A − M)/(L·k)⌋` with `μ` the usable fraction of capacity (0.9 by convention, as serving engines default; stated as a parameter), `A` the peak activation workspace and `M` the runtime reserve (CUDA context, graph pools). Capacity binds before bandwidth when `b_max(L) < b*(L)`.

*Steps.* (1) Capacity constraint `W + A + M + b·L·k ≤ μC`. (2) Solve for `b`. (3) Compare `b_max` and `b*`: if `b_max < b*`, the chip never leaves the bandwidth regime; the only lever is a smaller `k` (GQA, MLA, quantized KV) or more capacity. (4) MoE: `W` is the *total* weight bytes (all experts resident), so capacity is set by total parameters. (5) With tensor parallelism over `c` chips `W → W/c` per chip but `k` also splits across KV heads (up to `n_kv`), so `b_max` grows by about `c` in the best case (6.3 returns).

*Numerical check.* Compare `b_max` from the formula with the largest batch that actually runs on the A100 (`torch.cuda.OutOfMemoryError` bisection, 6 steps) at `L ∈ {2048, 8192}` for Qwen3-8B with a static cache of exactly `b·L` tokens; assert agreement within the reserve `A+M` (the cell fits `A+M` as one number and prints it, which should be a small fraction of 40 GB; to be measured). The upper bounds without reserve are `b_max` = 259 at `L=512`, 64 at `L=2048` and 16 at `L=8192` (computed with `μ = 0.9`, `W` = 16.38 GB, `k` = 147,456 bytes).

**D3. The throughput–interactivity frontier is a line.**

*Statement.* In the bandwidth regime and above `L ≥ 1`, per-user rate `u = β/(W + bLk)` and per-GPU throughput `T = b·u` satisfy `T(u) = (β − W·u)/(L·k)`: a straight line of slope `−W/(Lk)` and intercept `β/(Lk)` in the `(u,T)` plane. It is capped by the compute roof `T ≤ P/(2N_act)` and by capacity (`b ≤ b_max`, i.e. `u ≥ β/(W + b_max·L·k)`). Longer context lowers the intercept as `1/L`.

*Steps.* (1) `u = 1/t`, `t = (W + bLk)/β`. (2) Solve for `b`: `b = (β/u − W)/(Lk)`. (3) `T = b·u = (β − W·u)/(Lk)`. (4) The end points: `u_max = β/W` (at `b → 0`, `T → 0`) and `T_max = β/(Lk)` as `u → 0` (`b → ∞`; KV-read-bound asymptote). (5) Compute roof: in the compute regime `t = 2N_act·b/P`, `u = P/(2N_act·b)` and `T = P/(2N_act)` constant. (6) The corner where the line meets the roof is `b*` of D1 (when it exists). (7) Capacity truncates the line at `b_max`. (8) GPU-seconds per million tokens: `1e6/T`; read along the line.

*Numerical check.* Evaluate `ServingModel.frontier` at 50 batches and assert the points lie on the line to 1e-9 relative (bandwidth regime points), and that `T` stops increasing at `b*` when it exists; assert the intercept scales as `1/L` for `L ∈ {1024, 2048, 4096}` (three-point check).

**D4. MoE: touched parameters and expert coverage.**

*Statement.* With `E` routed experts per layer, `K` selected per token, and `b` tokens in a decode step routed independently and uniformly, the expected number of distinct experts touched per layer is `E·(1 − (1 − K/E)^b)`. Weight bytes read per step are `W_dense + n_moe_layers · E_touched · w_expert` and tend to the full model as `b` grows.

*Assumptions.* Independent uniform routing (real routers are skewed, so coverage is an upper bound for a *balanced* router; load-balanced training makes it close; the measurement tests it); `K` distinct experts per token.

*Steps.* (1) For one token, an expert is selected with probability `K/E`. (2) For `b` independent tokens, an expert is untouched with probability `(1−K/E)^b`. (3) Linearity of expectation gives `E·(1−(1−K/E)^b)`. (4) Bytes per step follow by multiplying by the expert size (`3·d·d_ff_moe·bytes` for gated MLP). (5) The batch at which half of the experts are touched is `b_{1/2} = ln(1/2)/ln(1−K/E)`; for `E=128`, `K=8` this is `≈ 10.7` (the cell prints it). (6) Consequence: MoE decode at small batch reads `≈ K/E` of the expert bytes (fast), at large batch reads all of them (the weight term of D1 becomes `W_total`), so `b*` computed with `N_act` and `W(b)` moves with `b`; the cell solves `2N_act·b/P = (W(b) + bLk)/β` numerically.

*Numerical check.* Monte Carlo with `E=128, K=8` and `b ∈ {1, 2, 4, 8, 16, 32, 64}`, 20,000 trials each, seed 0: formula within 3 standard errors; for the real model (T2 pair) hook the router outputs of `Qwen3-30B-A3B` on a fixed set of 512 prompts (from R3 or a fixed text file) and count distinct experts per layer per batch of `b` consecutive tokens; plot against the formula (hypothesis: real coverage is at or below the uniform curve; to be measured, with a bootstrap interval over batches).

**D5. Prefill time and the quadratic term.**

*Statement.* `t_prefill(n) = [2N_mm·n + 2·n_layers·n_heads·d_head·n²]/(P_eff) + t_ov` for causal attention (half of the `4n²` score and value products), valid when the matmul intensity `n·(1/b_w)` exceeds `R`; attention equals matmul FLOPs at `n_× = N_mm/(n_layers·n_heads·d_head)`.

*Steps.* (1) Matmul FLOPs per token `2N_mm`; attention per layer causal `2·n²·d_head·n_heads` (QKᵀ and PV each `n²·d_head·n_heads` multiply-adds, halved by the mask, times 2 FLOPs). (2) Sum over layers. (3) Equate: `2N_mm·n = 2·n_layers·n_heads·d_head·n²` gives `n_×`. (4) For `n ≪ n_×` TTFT is linear in `n`, for `n ≫ n_×` quadratic. (5) Memory floor `W/β` bounds small `n` from below.

*Numerical check.* Compute `n_×` for Qwen3-8B (`N_mm` from the config), compare with a `FlopCounterMode` count of one forward at `n ∈ {512, 8192, 65536}` (relative error ≤ 2% if the convention for causal masking is matched; the cell first measures which convention `FlopCounterMode` applies to `scaled_dot_product_attention` with `is_causal=True` and states it).

#### From-scratch implementation

Components (≈ 150 lines): `ServingModel` (≈ 70 lines: parameter counting from config fields including the MoE branch, `weight_bytes`, `kv_bytes_per_token`, `max_batch`, `decode_step`, `prefill_time`, `b_star`, `L_crit`, `frontier`); `TrainStepModel` (≈ 35 lines); `moe_expert_coverage` (≈ 5 lines); `fit_model(measurements, model_class, free) -> params` (≈ 25 lines: `scipy.optimize.least_squares` on log-ratios so that relative error is minimized); `ladder_table(...)` (≈ 15 lines).

Model ladder (table `tab_lm_hw_model_ladder`, each row adds one term; errors computed on held-out points):
- M0: datasheet roofline (`P`, `β` from `chip_specs`).
- M1: calibrated `P_eff`, `β_eff` (5.1 microbenchmarks; no fitting on workload data).
- M2: M1 plus launch overhead `n_kernels·α_k` where `n_kernels` is counted from a `torch.profiler` trace of one forward (printed per layer), so no free parameter.
- M3: M2 plus one fitted attention-kernel efficiency `η_attn` and one fitted elementwise efficiency; at most two fitted parameters, fitted on the calibration subset only.
The A5 limit of three fitted parameters is therefore met by the reference model.

Numerical tests: `ServingModel.n_params()` against `sum(p.numel())` of the instantiated meta-device model (exact; dense and MoE); `kv_bytes_per_token()` against `eq_kv-cache-bytes` and `d2l.config_memory` (exact); `weight_bytes()` against `d2l.config_memory`; `decode_step` limiting cases (`b=1`: equals `eq_hw_decode_bound`; `b → ∞` with `L>0`: slope `L·k/β`); `frontier` line identity (D3). Seeds: Monte Carlo seed 0, routing prompts fixed by id list.

#### In production

- Serving engines (vLLM, SGLang) report throughput and latency in the same axes; their benchmark scripts (`vllm bench serve`) are cited and the *offline* decode throughput of one engine at one point is measured to compare with the model (T2 optional cell; the engine's version is pinned and the result is labelled "engine, not hardware limit").
- `torch.utils.flop_counter.FlopCounterMode` is the FLOP oracle; `torch.cuda.memory_stats` and `torch.cuda.max_memory_allocated` are the memory oracle; `torch.profiler` the kernel-count oracle. The from-scratch model is checked against each.
- SemiAnalysis InferenceX is the production-scale oracle for other chips. Procedure for the snapshot: (1) clone the repository, record the commit; (2) locate the result artifacts (files and schema are to be read at authoring time; the blueprint does not assume a schema). At 2026-10-29 the repository root lists benchmark configuration and tooling folders (`inferencex-e2e`, `collectivex`, `operatorx`, `power_model`, `ubenchX`, …) and no obvious results directory, and the dashboard application is a separate repository (`InferenceX-app`, GPL-3.0 per the GitHub licence field). Before committing anything, confirm where the points are published and that this source is covered by the Apache-2.0 repository licence; use only points from the Apache-2.0 repository or its release artifacts, and none taken from the GPL-3.0 dashboard repository's code or data files (unverified whether its data carry the same terms); (3) select one model that `ServingModel` supports (dense Llama-3.3-70B-style or a Qwen3-MoE-style config if present; otherwise the closest dense config, labelled as an approximation) and one sequence-length pair; (4) extract, for H100 and B200, points of (tokens/s/GPU, tokens/s/user) and the framework and precision; (5) commit the trimmed CSV with columns `model, hw, framework, precision, isl, osl, conc, tps_per_gpu, tps_per_user, commit, retrieved` (`commit` is the upstream commit hash of `SemiAnalysisAI/InferenceX` at extraction), next to `LICENSE` (a copy of the upstream Apache-2.0 text) and `NOTICE.md` (the copyright line, the upstream URL, the commit hash, and the statement that rows and columns were selected and trimmed); the name "InferenceX" is used for attribution only (the upstream README labels unofficial forks as unofficial); (6) plot the datasheet-only and calibrated-by-ratio model curves, where "calibrated-by-ratio" scales the A100-derived efficiency factors, never fitted on InferenceX points.

#### Headline experiment

*Hypothesis.* A roofline-plus-overhead model calibrated on the A100 with microbenchmarks only (no workload data) predicts training step time, TTFT and decode throughput of Qwen3-8B on held-out points within 15% (the A5 target), whereas the datasheet-only roofline does not.

*Setup.* One A100 PCIe 40 GB, Qwen3-8B config random-initialized. Training: depth-truncated `n_layers ∈ {2, 4, 6}`, `(b, s) ∈ {(1,2048), (2,2048), (4,2048), (2,4096)}` (at most 8,192 tokens per step: `(8,2048)` is cut because its 16,384 tokens make the unchunked logits term alone 24.9 GB, `T·V·10` bytes, computed, beside 13 to 19 GB of state), `bf16_pure` recipe, fused AdamW, eager. Prefill: `n ∈ {128, 512, 2048, 8192, 32768}`, `b=1` (the forward asks for the last position's logits only, `logits_to_keep=1` to be confirmed at the pinned `transformers`, so the 10.0 GB full-logits tensor at `n=32768`, `32768 × 151936 × 2` bytes computed, is not built on 40 GB; otherwise that point is dropped). Decode (graph): `b ∈ {1, 2, 4, 8, 16, 32, 64, 128}`, `L ∈ {512, 2048, 8192}` (infeasible points skipped and listed: by the bounds in D2, `b ≥ 128` at `L=2048` and `b ≥ 32` at `L=8192` do not fit 40 GB, and `(64, 2048)` and `(16, 8192)` sit at the limit, so the `b_max` bisection decides them). Calibration subset: `n_layers=4` training points, `n ∈ {512, 8192}`, and decode `(b,L) ∈ {(1,512), (32,2048)}`; every other point is held out. Seeds: input seed 0, parameter init seed 0, three independent processes.

*Procedure.* A5 measurement protocol (warm-up, timed repeats, CUDA events, CV ≤ 5% acceptance). Compute errors `ε = |t_pred − t_meas|/t_meas` per point for M0 to M3. Fit M3 on the calibration subset only.

*Metrics.* Median and maximum `ε` per workload and model; fraction of held-out points with `ε ≤ 0.15`; MFU of the training points against datasheet and against `P_eff`; regime label per decode point from `.frontier`.

*Plot specification.* (i) `fig_lm_hw_decode_regimes`: x = batch (log2), y = measured step time (ms, log), one panel per `L`, measured points with MAD bars, curves for M0 and M3, vertical line at `b*` (absent when `L ≥ L_crit`, stated in the panel title), vertical line at `b_max`. (ii) `fig_lm_hw_frontier`: x = tokens/s per user, y = tokens/s per GPU (both linear), one curve per `L`, measured points, dashed compute roof, shaded infeasible region (`b > b_max`); secondary axis on the right shows GPU-seconds per million tokens `1e6/T`. (iii) `fig_lm_hw_ttft`: x = `n` (log2), y = TTFT (ms, log), M0 and M3, with `n_×` marked. (iv) `fig_lm_hw_train`: step time against `n_layers` with the linear fit and its extrapolation to 36 layers (no full-depth point: it does not fit 40 GB).

*Expected outcome.* Direction only: M0 over-predicts speed for all three workloads (the datasheet is a ceiling); M1 closes most of the gap for large-batch decode and for training; M2 is needed for small-batch decode on small models where launch overhead is visible; M3's two fitted parameters matter for training and long-context decode. The size of each gap is to be measured, and the 15% claim is tested, not assumed. The book's own measurement in `sec_kv-cache` of decode far below the bandwidth ceiling for an eager Python loop is the known precedent for the launch-overhead term.

*Noise check.* Three processes × the protocol's repeats; a difference between model rows counts only if the error gap exceeds the measurement CV (5%); held-out points with CV > 5% are re-run or dropped and listed.

*T1 reduction.* RTX 4090 with Qwen3-0.6B (training `(b,s) ∈ {(1,1024), (2,1024), (4,1024)}`, full depth, `bf16_pure` ≈ 8 bytes per parameter) and decode of Qwen3-1.7B at `L ∈ {512, 2048}`, `b ∈ {1, 4, 16, 64}`; the same ladder and plots with `P_eff, β_eff` from the 4090 calibration; no MoE series (24 GB cannot hold the 30B model).

*Artifacts committed (T2+).* `lm-hw-a100-workloads` (JSONL points, calibrations, fingerprint), the four figures as SVG, `tab_lm_hw_model_ladder.csv`, `lm-hw-inferencex-snapshot`.

#### Figures and tables

- `tab_lm_hw_requirements`; `tab_lm_hw_model_ladder`; `fig_lm_hw_decode_regimes`; `fig_lm_hw_frontier`; `fig_lm_hw_ttft`; `fig_lm_hw_train`; `fig_lm_hw_moe_coverage` (x = batch, y = distinct experts per layer; uniform formula, Monte Carlo, real routing); `fig_lm_hw_otherchips` (frontiers of A100, H100, B200 and an SRAM pipeline for one model and `L`, with InferenceX points overlaid and vendor claims marked as claims); equations `eq_lm_hw_decode_step`, `eq_lm_hw_bstar`, `eq_lm_hw_lcrit`, `eq_lm_hw_frontier`, `eq_lm_hw_expert_coverage`.

#### Pitfalls and misconceptions

- "Decode is memory-bound" is true of weights at small batch, not of the KV cache at long context at any batch (D1): the KV term never amortises.
- Tokens per second per GPU rises with batch while tokens per second per user falls; quoting only the first is the commonest benchmark mistake (the frontier is the honest object).
- MFU defined on peak versus on achievable FLOP/s differ by the fraction `P_eff/P`; state the denominator.
- TTFT in a service includes queueing and scheduling; `TTFT_model` is the forward only.
- MoE "active parameters" set FLOPs, not bytes, once the batch touches most experts (D4).
- Pure-bf16 training (8 bytes per parameter) fits where the mixed recipe (16–20) does not; the two are different training algorithms with different numerics (`sec_memory_precision`, 4.4).
- Eager decode of a small model measures launch overhead, not the chip.

#### Summary (draft bullets)

- Training needs FLOPs, state capacity and network together; prefill needs FLOPs; decode needs bandwidth and capacity, and the three are limited by different numbers on the same datasheet.
- Decode step time is `max(2N_act·b/P, (W + bLk)/β)`; weights amortise over the batch and the KV cache does not, so there is a context `L_crit` beyond which no batch makes decode compute-bound.
- The throughput-versus-interactivity frontier is a straight line in the bandwidth regime, with intercept `β/(Lk)` and slope `−W/(Lk)`, capped by the compute roof and the capacity wall.
- For mixture-of-experts models total parameters set capacity, active parameters set FLOPs, and the experts touched by a batch set weight traffic.
- A model calibrated by microbenchmarks plus a launch-overhead term is the minimum to approach a 15% prediction; the datasheet-only roofline is a ceiling, not a forecast.

#### Exercises

1. **(check)** A deployment serves a dense 8B model at `L = 8192` on a chip with `R ≈ 300`. A colleague proposes doubling the batch to raise throughput. Under D1, what happens to tokens per second per GPU and per user? *Sketch:* if `L > L_crit` the step time grows almost linearly with `b` (KV reads), so per-GPU throughput saturates at `β/(Lk)` and per-user rate falls; doubling the batch gains little.
2. **(check)** Why does tensor parallelism over `c` chips not lower the per-token latency floor of 5.1 D2 but does raise `b_max`? *Sketch:* each chip reads `1/c` of every byte at bandwidth `β`, so the floor `C/β` is unchanged while the capacity for KV grows (`b_max` up to `c` times, bounded by KV-head splitting).
3. **(derive)** Extend D1 to include the attention FLOPs of decode (intensity `2g/bytes`) and find the condition under which attention is *not* hidden behind its own KV read. *Sketch:* attention time is `max(attention FLOPs/P, KV bytes/β)`; the FLOP term exceeds the byte term iff `2g/bytes_per_element > R`, i.e. `g > R·bytes/2`; for `bytes=2`, `g > R`, never for realistic GQA group sizes; for MHA with `g=1` never either.
4. **(derive)** For a top-`K` router with a learned load imbalance in which expert `i` is chosen with probability `p_i` (`Σ p_i = K`), give the expected number of distinct experts touched by `b` tokens and show it is at most the uniform value. *Sketch:* `Σ_i (1 − (1−p_i)^b)`; concave in each `p_i` so Jensen gives the bound for equal `Σ p_i`.
5. **(measure)** Predict, then measure on your GPU, the context length at which the step time of decode at batch 16 doubles relative to `L=512`. Use D1's slope `Lk/β`; compare with the measured value and name the effect responsible for any gap. State the noise check.
6. **(measure)** Using the committed A100 log, reconstruct `fig_lm_hw_frontier` for `L=2048`. Fit a line to the bandwidth-regime points and compare its intercept and slope with `β_eff/(Lk)` and `−W/(Lk)`; report the relative error and bootstrap interval over repeats.
7. **(open)** Choose a hardware design point (for example a prefill-only chip with GDDR memory and a decode chip with SRAM) and a workload mix (long prompts, short answers, or the reverse). Use `ServingModel` and the frontier to argue for or against disaggregation (13.4), stating which assumption of the model your argument depends on most.

#### Further reading

- Papers: [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102) (the analytic inference model that D1 to D3 specialise); [Splitwise](https://arxiv.org/abs/2311.18677) and [DistServe](https://arxiv.org/abs/2401.09670) (separate prefill and decode pools; goodput); [LLM Inference Unveiled](https://arxiv.org/abs/2402.16363) (roofline view of inference); [Mind the Memory Gap](https://arxiv.org/abs/2503.08311) (new; verified title; shows that large-batch decode is bound by attention and KV reads, the D1 finding at long context); [PaLM](https://arxiv.org/abs/2204.02311) (new; verified title; the definition of MFU and the FLOPs-per-token convention).
- Benchmarks and blogs: SemiAnalysis [InferenceX](https://github.com/SemiAnalysisAI/InferenceX) and its [dashboard](https://inferencex.semianalysis.com/) (throughput versus interactivity across chips); [Scaling Book: inference](https://jax-ml.github.io/scaling-book/inference/); [MLCommons inference benchmarks](https://mlcommons.org/benchmarks/inference-datacenter/) (scenario definitions, from Codex 4.8); [Efficiency in LLMs (MLSS)](https://alex.smola.org/posts/45-mlss-efficiency/) (from Codex 4.8). From Codex 14.1: [Sarathi-Serve](https://arxiv.org/abs/2403.02310) (latency–throughput analysis; cited again in 13.2) and [vLLM documentation](https://docs.vllm.ai/) (benchmark and metric definitions).
- Code: [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang) benchmark scripts (new; both verified); Hugging Face Transformers `StaticCache`; `torch.utils.flop_counter`.

#### Risks and open decisions

- *Full-depth Qwen3-8B training does not fit on 40 GB.* At 8 bytes per parameter state is 65.5 GB (computed) before activations, above the card; the depth-truncated series (`n_layers ∈ {2, 4, 6}`) is therefore the method, not a fallback, and its linearity in `n_layers` is the stated validation. Points whose predicted peak exceeds 0.9 × 40 GB are dropped and listed; if the 6-layer points do not fit with the allocator's answer, the series falls back to `{1, 2, 4}`.
- *Transformers static-cache plus CUDA-graph path breaks for Qwen3 at the pinned version.* Fallback: `d2l.GPT` loaded with `d2l.load_qwen3` (1.2) with preallocated cache tensors and a manual graph; both paths are timed once at `b=1` to prove they agree within noise.
- *InferenceX schema, data location or licence changes.* The repository licence was re-verified as Apache-2.0 on 2026-10-29, so the default is to commit the trimmed CSV with attribution and the upstream commit hash. Fallback: if the points turn out to be published only under other or unclear terms (the dashboard application repository is GPL-3.0), the notebook downloads at run time and the committed figure is a PNG; the A5 autograder does not depend on it.
- *Qwen3-30B-A3B does not fit one 40 GB card, and barely fits the pair with KV.* Weights are ≈ 61 GB; on the T2 pair `0.9 × 80 − 61 = 11` GB remain for KV and workspace (computed), about 1.1e5 tokens at 98,304 bytes per token: batch and context are limited (`b·L ≤ ~1e5`); the grid is cut accordingly and the cell prints `b_max`. If the pair is unavailable, the MoE series is replaced by its Monte Carlo and the `ServingModel` prediction, labelled as not measured.
- *`FlopCounterMode` does not halve causal attention.* The notebook measures the convention first and states the difference (D5); tolerance 2% applies after matching conventions.

### 5.3 Memory and storage tiers, from SRAM to disk

- **Label:** `sec_lm_memory_tiers` · **Tier:** T3 (one 8× A100 40 GB node, local NVMe and a shared file system) · **Reduced T1 config:** one build-box GPU, build-box NVMe, the 0.6B training state (`Qwen/Qwen3-0.6B`, ≈ 12 bytes per parameter checkpointed, about 7 GB) instead of the 8B state (≈ 98 GB); `config.tier='T1'` · **Budget:** T3 ≈ 2 h wall-clock, host RAM ≥ 256 GB, ≥ 400 GB free on the local NVMe and on the shared file system; T1 ≈ 15 min, ≤ 15 GiB GPU, ≥ 30 GB free disk · **Codex units:** 1.8 (data infrastructure: I/O side only), 7.7 (checkpoint I/O cost only)
- **Builds on (book):**
  - `subsec_hw-bytes` (`fig_memory_hierarchy`, `fig_bandwidth_ladder`, `fig_latency_ladder`): the ladders are taken as given and are *extended* downward (NVMe, parallel file system, object store, HDD) and sideways (LPDDR5X, CXL); the section's plot overlays measured points on them.
  - `subsec_hw-cpu`: pinned versus pageable host memory and the 256 MB host-to-device measurement; 5.3 adds size sweeps, both directions, per-NUMA placement and an α–β fit. The book's exercise "Latency and bandwidth, measured" (host-to-device sweep and fit) and "Starved input pipeline" are acknowledged; their algebra is reused.
  - `sec_hardware` exercise "Ridge point of a link": the principle "a link has a ridge point and communication is hidden iff FLOPs per byte moved exceed it" is recapped in one paragraph; D1 applies it to offloading.
  - `sec_read_write` (safetensors, `state_dict`): file layout is not re-taught.
  - `subsec_mp-anatomy` (16 bytes per parameter): the *checkpointed* bytes per parameter (12 or 14) are derived here and differ from the 16 held in memory.
- **Builds on (this part):** 5.1 (`d2l.BenchLog`, `d2l.chip_specs` for the HBM row, `d2l.hw_copy_bench`), 5.2 (`d2l.ServingModel` for `k_KV` and `P_eff`, `d2l.TrainStepModel` for step time), 4.1 (`d2l.PackedStream`: token storage dtype and shard format; 5.3 only measures the I/O side). Forward by label: 5.4 (links), 6.2 (FSDP, used as a black box to get sharded state), 6.8 (Young/Daly checkpoint interval; 5.3 supplies the stall term), 13.3 and 13.4 (KV tiers in serving engines).
- **Exports (`#@save`):**
  - `d2l.tier_table() -> pd.DataFrame`: loads `data/hw/tiers.csv`. Columns `tier, example_device, bw_gbs, bw_note, latency_ns, capacity_gb_node, cost_rank, source_url, retrieved, measured_bw_gbs, measured_latency_ns` (the last two filled from committed logs when available). Contract: every row has a source and a date; `cost_rank` is an ordinal 1 (cheapest per GB) to 9, never a dollar amount; dollar figures live only in `data/hw/prices.csv` (5.6).
  - `d2l.hw_h2d_bench(nbytes, pinned=True, direction='h2d', numa=None) -> dict` (GB/s, MAD).
  - `d2l.parse_nvbandwidth(text) -> dict[str, np.ndarray]`, `d2l.parse_fio(json_text) -> dict` (`bw_gbs, iops, lat_mean_us, lat_p99_us, bs, rw, qd, jobs`), `d2l.parse_gdsio(text) -> dict` (`xfer, threads, io_size, gib_s, lat_us`).
  - `d2l.offload_breakeven(flops_per_byte, p_eff, link_bw) -> float` (ratio of compute time to transfer time) and `d2l.kv_load_vs_recompute(cfg, n, link_bw, p_eff) -> dict` (D1, D2).
  - `d2l.ckpt_stall(size_bytes, mode, bw_d2h, bw_write, ranks, interval_s) -> dict` (`stall_s, durable_s, backpressure_s, overhead`; D4).
  - Data: `DATA_HUB['lm-hw-node-storage']` and `DATA_HUB['lm-hw-buildbox-storage']` (tars of JSONL logs, each record with raw tool output). `lm-hw-node-storage` also includes two summaries that other chapters consume (there are no separate `lm-tier-bandwidth` or `lm-ckpt-io` keys):
    - `tiers.json`, consumed by 13.3 and 13.4: a list of records `{"tier": str, "bw_gbs": float, "bw_ci95": [lo, hi], "latency_us": float, "latency_p99_us": float, "n_repeats": int, "tool": str, "note": str}` with `tier` in `{"hbm", "h2d_pinned", "h2d_pageable", "d2h_pinned", "d2h_pageable", "nvme_seq_read", "nvme_seq_write"}` plus `"object_store"` and `"hdd"` when measured (unmeasured tiers are omitted, not zero-filled).
    - `ckpt_records.jsonl`, consumed by 6.8: one record per checkpoint write `{"mode": "sync"|"async", "size_bytes": int, "ranks": int, "interval_s": float, "write_s": float, "stall_s": float, "durable_s": float, "backpressure_s": float, "overhead": float, "predicted": {"stall_s": float, "durable_s": float}}`, with `predicted` computed by `d2l.ckpt_stall`; 6.8 reads `stall_s` as the checkpoint-stall term of Young/Daly.
- **Models and data:** `Qwen/Qwen3-8B` config, random-initialized and sharded for the checkpoint test (no download), real weights not needed; for the T1 test `Qwen/Qwen3-0.6B`. Synthetic token shards (`uint32` random tokens; vocabulary 151,936 exceeds 65,535, so `uint16` does not suffice, which the cell asserts) generated locally for I/O tests; no external datasets, no licenses involved.
- **Packages:** `fio` ≥ 3.30 (system), `nvbandwidth` (built from the repository, needs CUDA ≥ 11 and CMake ≥ 3.20), `gdsio` (ships with GPUDirect Storage; absent on many hosts), optional `kvikio`, `numactl`, `torch==2.15.0` (`torch.distributed.checkpoint`, `async_save`), `safetensors`, `boto3` or `s3fs` (object-store probe, optional), `psutil`.

#### Narrative arc

1. **The question.** One cell prints, for Qwen3-8B, the byte count of each thing a training and serving system must keep: bf16 weights, in-memory training state (16 B/param), checkpointed state (12–14 B/param), KV cache for `b=32, L=8192` (`k·L·b`, computed from the config), and a 1-trillion-token dataset stored as `uint32`. Next to each: the tier it sits on today and the tier it would have to sit on to be cheap. The measured symptom: the wall-clock time of a *synchronous* full-state checkpoint of the 8B model on the T3 node (to be measured; the cell prints it next to the step time from 5.2 so the reader sees "one checkpoint costs k training steps").
2. **The tier table (subsection "Every tier, one table").** Table `tab_lm_hw_tiers`, rows: registers and shared memory; L2; HBM; host DDR5; Grace-style LPDDR5X (coherent, via NVLink-C2C); CXL-attached memory; local NVMe; parallel file system; object store; hard disk. Columns as in the export contract. Datasheet sources: A100 datasheet (the PCIe 40 GB column: 1,555 GB/s HBM2 for the HBM row) and 5.1 logs (first three rows); [Grace CPU](https://www.nvidia.com/en-us/data-center/grace-cpu/) and [Grace Hopper](https://www.nvidia.com/en-us/data-center/grace-hopper-superchip/) pages and the measured data-movement study of the Grace Hopper paper (new; arXiv 2408.11556) for LPDDR5X and C2C, with the paper's numbers copied only after the author reads its tables; [Kioxia CM7-R](https://americas.kioxia.com/en-us/business/ssd/enterprise-ssd/cm7-r.html) as the NVMe example (read at authoring; the notebook uses the datasheet of the drive actually reported by `nvme list`); [3FS](https://github.com/deepseek-ai/3FS) and the Fire-Flyer paper for a cluster file system's published aggregate read bandwidth; the hard-disk row cites a vendor datasheet for an enterprise drive (Seagate Exos; the product page redirected when checked, so the URL is not given here: (unverified), author to supply). Latencies for tiers below the book's ladder come from fio (NVMe) and the object-store probe; the CXL row is datasheet-only and marked "not measured here".
3. **What lives where (subsection).** Table `tab_lm_hw_placement`: object, size formula, size for Qwen3-8B and for the T1 model, tier, access pattern, bandwidth demanded. Weights (HBM; read every step or token), optimizer state (HBM, or host if offloaded), activations (HBM; recompute or offload), KV cache (HBM, then host, then NVMe or network), dataset (NVMe or shared FS; sequential reads of large shards), checkpoints (written to NVMe, copied to shared storage or object store; read on restart). The table is generated from config fields, so changing the model rewrites it.
4. **Host links (subsection).** Measurement M1. Pageable versus pinned, both directions, local versus remote NUMA node, with an α–β fit from the size sweep. The PCIe Gen4 ×16 per-direction ceiling is computed from the [PCI-SIG](https://pcisig.com/specifications) rate and encoding (16 GT/s, 128b/130b) and compared with the A100 datasheet's printed "PCIe Gen4: 64 GB/s" (which is the bidirectional aggregate); neither number is claimed as achievable.
5. **Offloading (subsection).** Statement of the overlap condition (D1), the three uses (optimizer state: ZeRO-Offload; layer weights: ZeRO-Infinity and FlexGen; KV blocks), and a cell that evaluates the break-even for each with measured link bandwidth and `P_eff`. The notebook then runs a 3-layer toy (depth-truncated Qwen3-8B, T3) with optimizer states in pinned host memory and shows step time against tokens per step: below the threshold the step stalls, above it the step time approaches the on-GPU time (to be measured; the plot is the claim).
6. **KV-cache tiers (subsection).** D2: load versus recompute. Cell: for Qwen3-8B, `n ∈ {1k, 4k, 16k, 64k}`, compute the recompute time from `ServingModel.prefill_time` and the load time from each tier's measured bandwidth (pinned host, NVMe, shared FS, object store, with their α terms); the cell prints which tier wins at which `n`. Pointers to Mooncake and LMCache (`:cite:`) and to 13.3–13.4 for engines.
7. **Storage at cluster scale (subsection).** Measurements M2 to M5; D3 on shard sizes; the book's exercise on random versus burst access is the in-GPU analogue; here the same shape appears at disk scale. The lesson stated as measured: text training is not bandwidth-bound on I/O (demand in bytes per second is tiny, computed in the cell from tokens/s × 4 bytes), but small random reads and per-object latency matter, so shards are large.
8. **Checkpoint I/O (subsection).** Measurement M6, D4. Synchronous versus asynchronous; sharded versus gathered; host-memory staging; resume time. Closing paragraph hands the interval question to 6.8 by label.

#### Microbenchmarks (exact commands, parsers, plots)

All commands run under `d2l.BenchLog` per the chapter contract; `$D` is the local NVMe directory, `$S` the shared-file-system directory, `N` a NUMA node.

**M1. Host↔device bandwidth and NUMA.**
- Tool commands (one per NUMA pair `(cpu=N, mem=M)`, with the tool's own affinity control disabled by `-d`):
  `numactl --cpunodebind=N --membind=M nvbandwidth -d -b 512 -i 5 -t host_to_device_memcpy_ce device_to_host_memcpy_ce host_to_device_bidirectional_memcpy_ce device_to_host_bidirectional_memcpy_ce`
  and, for the kernel-copy variants, `-t host_to_device_memcpy_sm device_to_host_memcpy_sm`. Options are from the tool's README (verified): `-b` buffer MiB, `-i` samples, `-t` test cases, `-d` disables automatic affinity, `-p` selects by prefix, `-F json` is available.
- Torch probe: `d2l.hw_h2d_bench` for `nbytes = 2^12 … 2^30`, `pinned ∈ {False, True}`, `direction ∈ {h2d, d2h}`; timing by CUDA events around `tensor.to(device, non_blocking=True)` for pinned and a blocking copy for pageable; 3 processes.
- Parser `parse_nvbandwidth`: scan for lines `Running <testcase>.`; the next line is a header containing `bandwidth (GB/s)` and the direction text; the next line is the column index row; then one row per source device as `<row> v v …` where `N/A` or `0.00` mean not measured; stop at a blank line or at `SUM <testcase> <value>` and `COEFFICIENT_OF_VARIATION <testcase> <value>`, which are captured as scalars. Tested on a committed raw sample (the README example is the format reference).
- Plot `fig_lm_hw_h2d`: x = bytes (log2), y = GB/s (linear); four lines (pageable/pinned × local/remote NUMA) for h2d; second panel for d2h and the bidirectional sum; dashed line at the computed PCIe per-direction ceiling; annotate fitted `α` (µs) and `β` (GB/s) per line.

**M2. NVMe with fio.**
- Sequential: `fio --name=seq --directory=$D --size=32G --rw=read --bs=1M --iodepth=32 --numjobs=4 --ioengine=io_uring --direct=1 --group_reporting --time_based --runtime=30 --ramp_time=5 --output-format=json --output=seq_bs1M_qd32.json`, after a one-off write pass to create the files (`--rw=write`) so reads hit flash, not unwritten space.
- Random: same with `--rw=randread --bs=4k --iodepth=64 --numjobs=8`.
- Sweep (loop in Python): `bs ∈ {4k, 16k, 64k, 256k, 1M, 4M}` × `rw ∈ {read, randread}` × `iodepth ∈ {1, 64}`; each run 15 s after 3 s ramp; three repeats (72 runs, about 22 minutes of the 2 h; cut: `iodepth` 4 and 16, which the plot does not use and exercise 6 asks readers to measure).
- Parser `parse_fio`: from `jobs[0].read.bw_bytes`, `jobs[0].read.iops`, `jobs[0].read.clat_ns.percentile['99.000000']` and `jobs[0].read.lat_ns.mean` (with `--group_reporting` the jobs are merged into one entry); unit conversion to GB/s (decimal) and µs; fails loudly if the key `percentile` is missing.
- Plot `fig_lm_hw_fio`: x = block size (log2), left axis GB/s (log), right axis IOPS (log); series `read`/`randread` at `iodepth=64` solid and at `iodepth=1` dashed; vertical marker at the knee `bs = β/IOPS` where the two regimes cross; dashed horizontal at the drive datasheet's sequential figure.
- `--direct=1` is mandatory: without it the page cache serves repeated reads and the plot would show DRAM; the cell asserts `direct=1` in `cmd`.

**M3. GPUDirect Storage against a bounce buffer.**
- Commands (flags per the GDS benchmarking guide; the author confirms them with `gdsio -h` on the host): `gdsio -f $D/gds.dat -d 0 -w 8 -s 8G -i 1M -x 0 -I 0` for GPU-direct, `-x 2` for the CPU-staged bounce buffer, and `-x 1` for CPU-only, each repeated for `-i ∈ {64K, 1M, 16M}` and `-w ∈ {1, 8}` (cut: `-w` 4 and 16; the claim is GPU-direct versus bounce buffer, which two thread counts show).
- Parser `parse_gdsio`: one summary line per run with the transfer type, thread count, data size, throughput (GiB/s) and average latency; test on a committed sample.
- Plot `fig_lm_hw_gds`: grouped bars, x = I/O size, y = GiB/s, bars = {GPU-direct, bounce buffer, CPU-only}, one panel per thread count; dashed NVMe sequential ceiling from M2. If `nvidia-fs` is not loaded the tool runs in compatibility mode; the log records `gds_mode` and the notebook states which mode was measured, because compatibility mode is a bounce buffer in disguise.

**M4. Shared file system and metadata.**
- Same fio runs with `--directory=$S`, launched on all 8 GPU ranks simultaneously by `mpirun -np 8 fio …` or `torchrun` calling `subprocess`, reporting aggregate and per-rank GB/s; plus a Python probe that creates, then reads, 10,000 files of 64 KiB and 100 files of 1 GiB (time per file, total bytes/s).
- Plot `fig_lm_hw_sharedfs`: bars of aggregate bandwidth per access pattern (large sequential, small random, many small files); y log; the small-file bar shows why shards are large.

**M5. Object store (optional, if an endpoint is available).**
- Python probe: range-GET of sizes `{64 KiB … 256 MiB}` (powers of 4) with concurrency `{1, 8, 64}`; fit `t = α + s/β` per concurrency from the single-stream series; log `α` (first-byte latency) and aggregate bandwidth.
- Plot `fig_lm_hw_object`: x = object size (log), y = effective MB/s (log), one line per concurrency; marker at `s_90 = 9αβ` (D3).

**M6. Checkpoints.** Details in the headline experiment.

#### Derivations

**D1. When does offloading pay?**

*Statement.* Suppose a kernel stream on the GPU does `F` FLOPs of work that needs `B` bytes of state that live on the far side of a link of bandwidth `β_ℓ` (and the GPU runs at `P_eff`). With perfect prefetch overlap there is no stall iff `F/P_eff ≥ B/β_ℓ`, i.e. iff the FLOPs per byte moved over the link, `F/B`, are at least the *link's ridge* `P_eff/β_ℓ`. Without overlap the step takes `F/P_eff + B/β_ℓ`.

*Assumptions.* One link, full duplex use when both directions are needed; transfers do not slow compute (HBM contention ignored); the link is dedicated; chunking is fine-grained enough for pipelining; bytes are counted per step.

*Steps.* (1) Times: `t_c = F/P_eff`, `t_x = B/β_ℓ`. (2) With double buffering the step is `max(t_c, t_x)` plus one chunk of fill and drain latency; without it `t_c + t_x`. (3) No stall iff `t_c ≥ t_x ⟺ F/B ≥ P_eff/β_ℓ`. *Optimizer state offload* (ZeRO-Offload): per step the GPU sends gradients (`2N` bytes, bf16) down and receives updated weights (`2N` bytes) up, in opposite directions on a full-duplex link, so `t_x = 2N/β_ℓ` per direction; `F = 6N·T` for `T` tokens per step. (4) No stall iff `6NT/P_eff ≥ 2N/β_ℓ ⟺ T ≥ P_eff/(3β_ℓ)` (independent of `N`). If the link is *shared* (half-duplex model) the bound is `T ≥ 2P_eff/(3β_ℓ)`. (5) The CPU must also run Adam: bytes read and written per parameter `≈ 28–30` (the cell counts them from the update rule: read `p, g, m, v`, write `p, m, v`, plus the bf16 copy), so `t_cpu = 30N/β_host`; no stall also needs `t_cpu ≤ t_c`, which gives `T ≥ 5P_eff/β_host` (using `6NT/P_eff ≥ 30N/β_host`), a second threshold that is usually *larger* than the link's. (6) *Weight streaming* (ZeRO-Infinity, FlexGen) for decode at batch `b`: `F = 2N_layer·b`, `B = 2N_layer` (bf16), so no stall iff `b ≥ P_eff/β_ℓ` (bf16: `2b/2`): the batch must equal the *link's* ridge, thousands of tokens, which is why FlexGen targets throughput-oriented offline batches. (7) The three conditions are the book's "ridge of a link" with the numerator and denominator chosen per use.

*Numerical check.* (a) Algebra: a symbolic check with `sympy` that the thresholds in steps 4–6 follow from the equalities (exact). (b) Simulation: a two-stage pipeline simulator (a loop over chunks with `compute_i` and `transfer_i` events) returns total time; assert it equals `max(t_c, t_x) + chunk fill` to 1% for 200 random `(F, B, chunks)` draws, seed 0. (c) Evaluation: with measured `β_ℓ` from M1 and `P_eff` from 5.1, the cell prints the three thresholds (tokens per step; host bandwidth; decode batch) for the T3 node; T1 repeats with the 4090. (d) Measurement (T3): the toy offloaded run's step time versus `T` crosses from stalled to overlapped near the predicted `T` (agreement within a factor of 1.5 is the test; the cell prints the ratio).

**D2. KV-cache reuse: load or recompute?**

*Statement.* A prefix of `n` tokens whose KV (`n·k` bytes) is stored on a tier with first-byte latency `α_t` and bandwidth `β_t` is cheaper to load than to recompute iff `α_t + n·k/β_t < t_prefill(n)`, with `t_prefill` from 5.2 D5. Ignoring attention (`n ≪ n_×`), the condition is `β_t > k·P_eff/(2N_mm)` per token, independent of `n`; with attention, recompute grows quadratically so loading wins by more at long `n`.

*Steps.* (1) Recompute: `t_r(n) = [2N_mm·n + 2n_l n_h d_h n²]/P_eff` (5.2 D5). (2) Load: `t_l(n) = α_t + n·k/β_t`. (3) Break-even bandwidth at large `n` (drop `α_t`): `β_t* (n) = n·k/t_r(n) = k·P_eff/(2N_mm + 2n_l n_h d_h n)`, decreasing in `n`. (4) At small `n` the latency term dominates: for `n ≪ n_×` solving `α_t + nk/β_t = 2N_mm·n/P_eff` gives `n_min = α_t/(2N_mm/P_eff − k/β_t)`, which exists only when `β_t > k·P_eff/(2N_mm)`; the cell solves the full equation (with the attention term) numerically for `n_min(tier)`. (5) Loading also frees GPU compute for other requests, so the true benefit exceeds `t_r − t_l`. (6) For a tier shared across servers, `β_t` is divided by concurrent readers.

*Numerical check.* For Qwen3-8B (`k = 147,456` bytes, `N_mm` from the config) assert `β_t*(n)` equals the bisection root; evaluate with the measured bandwidths of pinned host memory, NVMe, shared file system and object store and print the crossover `n_min`; the 5.2 prefill model supplies `t_r`, and one *measured* prefill at `n=8192` on the A100 validates `t_r` within the 5.2 error.

**D3. Shard size and random-read granularity.**

*Statement.* Reading `s` bytes in one request takes `t(s) = α + s/β`. The fraction of the bandwidth achieved is `η(s) = s/(αβ + s)`, so `η ≥ 1−ε` iff `s ≥ αβ(1−ε)/ε`; for `ε=0.1`, `s ≥ 9αβ`. Under a device IOPS limit `μ` the throughput is `min(μ·s, β)`, with knee at `s = β/μ`.

*Steps.* (1) `η = (s/β)/(α + s/β)`. (2) Solve `η = 1−ε`. (3) With `μ`, a single stream of requests of size `s` achieves `s/t(s)` at queue depth 1; at queue depth `q`, `min(q·s/t(s), μ s, β)`. (4) Consequence for datasets: shards of `s_shard ≥ 9αβ` amortise per-open latency; the book's burst-versus-random exercise gives the in-GPU analogue.

*Numerical check.* Fit `(α, β)` to the M2 or M5 size sweep by non-linear least squares; assert predicted `η(s)` within 10% of measured `bw/β` for all sizes; print `s_90`.

**D4. Training time lost per checkpoint.**

*Statement.* Let the training state be `S` bytes, `g` ranks, device-to-host bandwidth `β_d` per rank (pinned), write bandwidth `β_w` per rank-local NVMe, shared file system aggregate `β_fs`. For a synchronous sharded save the training stalls for `t_sync = S/(g·β_eff,w)` with `β_eff,w = min(β_w, β_fs/g)` plus fixed overhead `t_0`. For an asynchronous save the stall is the staging copy `t_stage = S/(g·β_d)` (plus `t_0`) and the write proceeds in the background taking `t_bg = S/(g·β_eff,w)`; if the next save is requested after `T_c` seconds and `t_bg > T_c`, training stalls for `t_bg − T_c` more (backpressure). The overhead fraction is `ε = t_stall/T_c`.

*Steps.* (1) Per-rank bytes `S/g` (sharded state, no replication; with data-parallel replicas only one replica writes, which DCP's deduplication provides). (2) Sync stall is the write time. (3) Async stall is the staging copy; the pinned staging buffer costs `S/g` of host memory per rank. (4) Backpressure when `t_bg > T_c`. (5) `ε_sync = t_sync/T_c`, `ε_async = max(t_stage, t_stage + t_bg − T_c)/T_c`. (6) The minimum interval for a target `ε`: `T_c ≥ t_stall/ε`. (7) The *optimal* interval under failures (Young/Daly) minimises `ε` plus expected rework and is the subject of 6.8; this section only supplies `t_stall`.

*Numerical check.* (a) The function `d2l.ckpt_stall` equals the closed form on random inputs (exact). (b) T3 measured `stall_s` and `durable_s` versus predictions from M1/M2 bandwidths: within 30% for sync (I/O stacks are noisy; the cell prints the ratio and says that tolerance), within 30% for async stall. (c) Edge cases: `t_bg > T_c` yields positive backpressure; `g → ∞` makes `ε → t_0/T_c`.

#### From-scratch implementation

Components (≈ 150 lines): `parse_nvbandwidth`, `parse_fio`, `parse_gdsio` (≈ 60 lines total, tested on committed raw samples); `hw_h2d_bench` (≈ 20); `offload_breakeven`, `kv_load_vs_recompute` (≈ 25); `ckpt_stall` (≈ 15); `pipeline_sim(chunks, compute, transfer)` (≈ 15); `fit_alpha_beta(sizes, times)` (≈ 10, reused by 5.4 and 5.5 as `d2l.fit_alpha_beta`; exported once here under that name).

Additional export: `d2l.fit_alpha_beta(sizes, times, weights=None) -> (alpha, beta, cov)` (weighted least squares on `t = α + s/β`; weights `1/t²` so relative error is minimized).

Tests: parsers against committed samples (exact values); `pipeline_sim` versus closed form (1%); `fit_alpha_beta` recovers synthetic `(α, β)` with 1% noise to within 2% (seed 0); `ckpt_stall` closed-form identity; token-bytes assertion for `uint32`.

#### In production

- `torch.distributed.checkpoint` (DCP): `dcp.save`, `dcp.async_save` (returns a future; stages to pinned host memory) and `dcp.load` with resharding; the from-scratch `ckpt_stall` is the oracle for what DCP should cost, and the measured gap is the library's overhead. `safetensors` for single-file weights. TorchTitan's checkpoint options as a pointer only.
- NVIDIA GPUDirect Storage through `gdsio` and, from Python, `kvikio`; the oracle is the `-x 2` bounce-buffer run.
- Cluster file systems (3FS, Lustre, GPFS) and object stores are named; only the measured one is analysed.
- KV tiers in engines (LMCache, Mooncake) are cited; D2 is the oracle for whether a tier is worth enabling.

#### Headline experiment

*Hypothesis.* On the T3 node, asynchronous sharded checkpointing of the Qwen3-8B training state cuts the training stall per checkpoint by a large factor relative to a synchronous gathered `torch.save` of the same state, and the stall of the asynchronous path is set by device-to-host staging bandwidth (M1), not by disk bandwidth (M2); the model of D4 predicts the stalls within 30%.

*Setup.* One 8× A100 node (planned for 40 GB per GPU; 6.2's memory model puts `16P/k` = 16.4 GB of state on each GPU plus transient terms, and the dummy step uses at most 1,024 tokens, so the logits term is negligible; the cell prints `fsdp_costs` and the plan fits if its total is at most 36 GB), FSDP2 sharding as a black box (`fully_shard` per block, 6.2 teaches it), Qwen3-8B config random-initialized in bf16, AdamW with fp32 master weights and moments created by one dummy step so the state has realistic shape (checkpoint ≈ 12 B/param = ≈ 98 GB; with a separate bf16 copy ≈ 115 GB; the cell prints the exact number from the state dict). Destinations: local NVMe (`$D`) and shared file system (`$S`). Methods: (a) gather to rank 0 and `torch.save` (synchronous, single writer); (b) DCP synchronous sharded `save`; (c) DCP `async_save` with pinned staging. Three timed saves each, after one discarded warm-up save (that first save carries the pinned-buffer allocation, so the three timed saves are the reused-buffer case, and a separate method (d) for buffer reuse is cut); seeds irrelevant (state values do not affect I/O) but fixed (0). Page cache dropped between runs only where permitted (`echo 3 > /proc/sys/vm/drop_caches` needs root; otherwise read-back uses `O_DIRECT` through fio or the file is read after writing a much larger file; the cell records which).

*Procedure.* For each method and destination: time the *blocking* interval seen by the training loop (`stall_s`: from calling `save` until it returns to the training loop), the time until the files are durable (`durable_s`: `future.result()` plus `fsync` of the directory), and the resume time (`load_s`: sharded load to the same world size, checksum compare of three tensors against the saved values). Measure the step time of 20 dummy FSDP steps in the same script to convert stall to an overhead `ε` for `T_c ∈ {10, 50, 200, 1000}` steps.

*Metrics.* `stall_s`, `durable_s`, `load_s`, aggregate GB/s, `ε(T_c)`, host-memory high-water mark (`psutil`), per-rank file sizes.

*Plot specification.* `fig_lm_hw_ckpt`: grouped stacked bars, x = method × destination (6 bars), stack = blocking `stall_s` (solid) and background `durable_s − stall_s` (hatched), y = seconds (linear), error bars = MAD over three saves, dashed horizontal lines at 1, 10 and 100 training steps. `fig_lm_hw_ckpt_interval`: x = checkpoint interval `T_c` in steps (log), y = overhead `ε` (log), one line per method, dotted horizontal at 1% and 5%, markers at the intervals measured.

*Expected outcome.* Direction: synchronous gathered save is slowest (single writer), async stall is a small multiple of the pinned device-to-host copy of `S/g` per rank, and durable time is bounded by write bandwidth. The magnitudes are to be measured; D4's predictions are printed beside them.

*Noise check.* Three saves per cell (a fourth warm-up discarded), median and MAD; a method comparison is a claim only when the gap exceeds 3× MAD; file-system caching state recorded.

*T1 reduction.* One GPU, Qwen3-0.6B (state ≈ 7 GB), local NVMe only: methods (a), (b) and (c) with `g=1`; M1 on the build box; M2 on the build-box drive; M3 skipped unless `nvidia-fs` is present (the committed T3 log is analysed instead). Same plots, T1 overlaid on T3 where both exist.

*Artifacts committed (T2+).* `lm-hw-node-storage` (M1 to M6 logs with raw outputs, `nvme list`, `lsblk`, `df`, `mount`, `numactl --hardware`, `lscpu` captured next to each series; plus the `tiers.json` and `ckpt_records.jsonl` summaries defined in the exports), the figures, `tiers.csv`.

#### Figures and tables

- `tab_lm_hw_tiers`, `tab_lm_hw_placement`; `fig_lm_hw_tiers` (the book's bandwidth ladder redrawn with measured points overlaid as markers, and a second panel for latency); `fig_lm_hw_h2d`, `fig_lm_hw_fio`, `fig_lm_hw_gds`, `fig_lm_hw_sharedfs`, `fig_lm_hw_object`, `fig_lm_hw_ckpt`, `fig_lm_hw_ckpt_interval`, `fig_lm_hw_offload` (step time versus tokens per step with the predicted threshold as a vertical line), `fig_lm_hw_kvtier` (load versus recompute time against prefix length `n` per tier); equations `eq_lm_hw_offload_overlap`, `eq_lm_hw_kv_breakeven`, `eq_lm_hw_shard_eff`, `eq_lm_hw_ckpt_stall`.

#### Pitfalls and misconceptions

- Reading a file twice measures the page cache; use `O_DIRECT` or drop caches, and say which.
- A pageable copy is staged by the driver; benchmarking only pinned memory overstates what an ordinary `.to(device)` achieves.
- `nvbandwidth` buffer size matters: small buffers under-report; use ≥ 512 MiB for peak figures (the README says so).
- "NVMe is fast" refers to large sequential reads at high queue depth; a single-threaded 4 KiB read at queue depth 1 is latency-bound.
- The page cache can make a checkpoint "finish" before it is durable; time `fsync`.
- Checkpointed bytes per parameter (12–14) are not the in-memory 16; gradients and activations are not saved.
- GPUDirect Storage in compatibility mode is not GPUDirect.

#### Summary (draft bullets)

- Memory is a ladder of at least nine tiers; each has a measured bandwidth, latency, capacity and relative cost, and language-model state should live on the cheapest tier that its access pattern can tolerate.
- Offloading pays when the FLOPs per byte moved over the link exceed the link's ridge point (`P_eff/β_ℓ`); for optimizer state this is a token count per step independent of model size, for weight streaming it is a batch size equal to the ridge.
- Reusing a stored KV prefix beats recomputing it above a surprisingly low bandwidth, and by more for long prefixes, which is why tiered KV stores exist.
- Datasets are read in large shards because per-request latency, not bandwidth, limits small reads; text training demands little I/O bandwidth.
- A synchronous checkpoint stalls training for the write time; an asynchronous sharded one stalls only for device-to-host staging, and the overhead is that stall divided by the checkpoint interval.

#### Exercises

1. **(check)** Why does the checkpoint of an 8B model weigh ≈ 12–14 bytes per parameter while training memory holds 16? *Sketch:* gradients (4 B) and activations are not saved; fp32 master weights and two Adam moments are (12 B), plus an optional bf16 copy (2 B).
2. **(check)** A drive sustains 14 GB/s of sequential reads and 2.7 million random-read IOPS at 4 KiB (hypothetical round numbers for the exercise). At what block size do the two regimes cross, and what does that say about 4 KiB reads? *Sketch:* `β/μ = 14e9/2.7e6 ≈ 5.2 KB` (with your own drive's datasheet figures the arithmetic is the same).
3. **(derive)** Redo D1 step 4 when the link is half-duplex and the optimizer step on the CPU is overlapped with the next forward pass. Which threshold on `T` results, and what is the minimum host bandwidth? *Sketch:* transfers `4N/β_ℓ` per step versus `6NT/P_eff`; CPU Adam `30N/β_host` hidden if `≤ t_c`; thresholds `T ≥ 2P_eff/(3β_ℓ)` and `T ≥ 5P_eff/β_host`.
4. **(derive)** Show that with `g` ranks writing to a shared file system of aggregate bandwidth `β_fs`, doubling `g` stops reducing the stall once `g·β_w > β_fs`, and compute the speedup of sharded versus gathered saves as a function of `g`. *Sketch:* `t_sync = S/min(gβ_w, β_fs)`; gathered `t = S/β_w` (single writer); speedup `min(g, β_fs/β_w)`.
5. **(measure)** On your machine measure pageable and pinned host-to-device bandwidth for sizes 4 KiB to 1 GiB and fit `(α, β)` per curve; predict, before measuring, the size at which pinned reaches half of its asymptote and compare. State the noise check.
6. **(measure)** Using M2, measure random-read IOPS at queue depths 1, 4, 16, 64 for 4 KiB blocks on your drive and fit Little's law (`IOPS = qd/latency` until saturation). At what queue depth does the drive saturate?
7. **(open)** Design the storage layout for a 1-trillion-token training run on 64 GPUs with a 100 GB checkpoint every 30 minutes: shard size, where shards live, checkpoint destination, retention. Justify each choice with a number from this section's measurements and state which number you would re-measure first on a different cluster.

#### Further reading

- Papers: [ZeRO-Infinity](https://arxiv.org/abs/2104.07857) (offload to CPU and NVMe; D1); [FlexGen](https://arxiv.org/abs/2303.06865) (weight streaming; the `b ≥` link ridge condition); [Mooncake](https://arxiv.org/abs/2407.00079) (KV-centric disaggregated serving with a tiered cache; D2); [Fire-Flyer AI-HPC](https://arxiv.org/abs/2408.14158) (storage and network co-design; 3FS); [Llama 3](https://arxiv.org/abs/2407.21783) (checkpoint storage at cluster scale); [Understanding Data Movement in Tightly Coupled Heterogeneous Systems: A Case Study with the Grace Hopper Superchip](https://arxiv.org/abs/2408.11556) (new; verified title; measured CPU–GPU and LPDDR5X behaviour for the Grace tier).
- Tools and documentation: [GPUDirect Storage](https://docs.nvidia.com/gpudirect-storage/) (v5 source); [GPUDirect Storage benchmarking and configuration guide](https://docs.nvidia.com/gpudirect-storage/configuration-guide/index.html) (verified; the `gdsio` flags) and [O_DIRECT requirements guide](https://docs.nvidia.com/gpudirect-storage/o-direct-guide/index.html) (new; verified); [PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html); [fio documentation](https://fio.readthedocs.io/en/latest/fio_doc.html) (new; verified; job options and JSON output); [fio](https://github.com/axboe/fio); [nvbandwidth](https://github.com/NVIDIA/nvbandwidth) (new; verified; flags and output format); [MLPerf Storage](https://mlcommons.org/benchmarks/storage/) (new; verified; standardised training-I/O benchmark, cited as a pointer).
- Code and systems: [LMCache](https://github.com/LMCache/LMCache); [3FS](https://github.com/deepseek-ai/3FS). From Codex 1.8: [Hugging Face dataset streaming](https://huggingface.co/docs/datasets/stream), [PyTorch data loading](https://docs.pytorch.org/docs/stable/data.html), [TorchData](https://github.com/pytorch/data) and [Datatrove](https://github.com/huggingface/datatrove) (shard formats and I/O overlap; the resumable-cursor material itself is owned by 4.1). From Codex 7.7: [PyTorch elastic launcher](https://docs.pytorch.org/docs/stable/elastic/run.html) and [TorchTitan](https://github.com/pytorch/torchtitan) (checkpoint practice; owned by 6.8).
- Reference: [The Datacenter as a Computer](https://link.springer.com/book/10.1007/978-3-031-01761-2) (storage hierarchy and latency numbers).

#### Risks and open decisions

- *No GDS support on the T3 node (no `nvidia-fs`, unsupported file system).* Fallback: report the compatibility-mode run, label it, and analyse the committed log of a GDS-enabled host if one exists; otherwise the bar for GPU-direct is omitted and the text says so.
- *No root, so no page-cache drop.* Fallback: `O_DIRECT` reads through fio; checkpoint read-back through `dd iflag=direct`; the cell records the method in the log.
- *No object-store endpoint or CXL hardware.* Fallback: M5 skipped with a committed log from a previous allocation; CXL remains datasheet-only.
- *The shared file system is quota-limited below 400 GB.* Fallback: checkpoint the 4B model (`Qwen/Qwen3-4B`, ≈ 50 GB) and extrapolate by `S`, labelled as such.
- *Grace-tier numbers come only from a paper.* Fallback: label them "from the paper, not measured here"; no D-section depends on them.

### 5.4 Data paths inside a server

- **Label:** `sec_lm_server_paths` · **Tier:** T3 (one 8× A100 node, NVSwitch SXM or NVLink-bridged PCIe pairs; the committed `topo.txt` decides which row of `tab_lm_hw_links` applies) plus the T2 pair for the `g=2` comparison · **Reduced T1 config:** the build box runs no GPU-to-GPU traffic (no NVLink, unreliable peer-to-peer): one GPU, host-to-device and device-to-host series by `nvbandwidth` and `hw_h2d_bench`, the box's topology capture, and the parser and fit tests on committed samples, behind `config.tier='T1'`; the two-GPU collective measurements run on the T2 pair behind `config.tier='T2pair'` (`g=2`: NVLink bridge versus `NCCL_P2P_DISABLE=1`, which sends the traffic through the host), and the node's own `NCCL_P2P_DISABLE=1` run is the T3 emulation of a host-staged path · **Budget:** T3 ≈ 1 h wall-clock, small memory (≤ 4 GiB per GPU plus the 4 GiB message buffers); T2 pair ≈ 15 min and T1 ≈ 5 min (estimates) · **Codex units:** 4.7
- **Builds on (book):**
  - `subsec_hw-interconnects` and `fig_pcie_topology`: the build box is the worked example of a host-staged path; the figure is regenerated here from `nvidia-smi topo -m` text, not redrawn by hand. The NVLink and NVL72 *numbers* of that subsection are taken as given.
  - `sec_multi_gpu`: `eq_ring_traffic` (`2(k−1)/k·N` bytes per device), `eq_dp_cost`, the bus-bandwidth convention (one sentence recap; the exercise's `busbw = (S/t)·2(k−1)/k`), the star-versus-ring contrast. The exercise "Latency in the allreduce" (ring and binary-tree time with `α + m/β`) is acknowledged: 5.4 states the α–β model as a model, fits it, and does not re-derive the ring formula.
  - `sec_multi_gpu_concise`: `tab_collectives` and the `NCCL_SHM_USE_CUDA_MEMCPY=1` recovery, cited as the book's build-box finding and re-run only on the T2 pair.
  - Not re-taught: ring all-reduce, DDP bucketing, pinned memory.
- **Builds on (this part):** 5.1 (`d2l.BenchLog`, `d2l.env_fingerprint`, `d2l.chip_specs` for NVLink column), 5.3 (`d2l.parse_nvbandwidth`, `d2l.fit_alpha_beta`, `d2l.hw_h2d_bench`). Forward by label: 5.5 (extends the collective model across nodes), 6.3 to 6.7 (placement of parallel axes), 6.1 (`d2l.launch` for multi-process runs; 5.4 uses `torchrun` and `mpirun` directly).
- **Exports (`#@save`):**
  - `d2l.links_table() -> pd.DataFrame` from `data/hw/links.csv` with columns `link, generation, lanes, rate_gts, encoding, gbs_per_dir, gbs_bidir, latency_note, source_url, retrieved`. PCIe rows compute `gbs_per_dir = lanes × rate_gts × (128/130) / 8` for Gen 3 to 5 in code (not stored by hand); NVLink, C2C, Infinity Fabric, InfiniBand and Ethernet rows are transcribed from vendor pages. Contract: every transcribed row has a source and a date.
  - `d2l.parse_topo_matrix(text) -> TopoMatrix` (fields `labels, link[i][j], cpu_affinity, numa_affinity, legend`) for `nvidia-smi topo -m`; `d2l.topo_graph(topo, links) -> networkx.Graph` (nodes GPU, NIC, CPU socket, switch-class nodes inferred from `PIX/PXB/PHB/NODE/SYS`; edge attribute `gbs`).
  - `d2l.parse_p2p(text) -> dict[str, np.ndarray]` (the `p2pBandwidthLatencyTest` matrices keyed by their titles); `d2l.parse_nccl_tests(text) -> pd.DataFrame` (columns `size, count, dtype, redop, root, oop_us, oop_algbw, oop_busbw, oop_wrong, ip_us, ip_algbw, ip_busbw, ip_wrong`; ignores `#` comment lines; the `Avg bus bandwidth` summary is captured as an attribute).
  - `d2l.nccl_busbw_factor(op, k) -> float` (`allreduce: 2(k−1)/k`; `allgather, reducescatter, alltoall: (k−1)/k`; `broadcast, reduce: 1`, per the nccl-tests performance page) and `d2l.coll_time(op, S, k, alpha, beta, algo='ring') -> float` (α–β model of ring, binary tree and direct all-to-all on a full crossbar; 5.5 extends it).
  - `d2l.tree_link_loads(k, groups, pattern) -> dict` (the link-load counter of D2: routes a traffic pattern through a tree of switches and returns bytes per link).
  - Data: `DATA_HUB['lm-hw-node-p2p']`, `DATA_HUB['lm-hw-t2pair-p2p']`, `DATA_HUB['lm-hw-buildbox-p2p']` (the box's topology capture and host-to-device series only). `lm-hw-node-p2p` includes an `alpha_beta.json`: a list of records `{"scope": "intra_node"|"inter_node", "link": str (e.g. "nvlink", "pcie", "pcie_host_staged", "ib", "roce"), "op": "p2p"|"all_reduce"|"all_gather"|"reduce_scatter"|"all_to_all", "g": int, "algo": str|null, "alpha_s": float, "beta_Bps": float, "alpha_ci95": [lo, hi], "beta_ci95": [lo, hi], "n_half_bytes": float, "resid_rel": float, "n_points": int}`, with α in seconds and β in bytes per second, fitted with `d2l.fit_alpha_beta` (weights `1/t²`); 6.x and 13.4 consume it. This file holds the intra-node (NVLink or PCIe) fits for point-to-point and for all-reduce, all-gather, reduce-scatter and all-to-all (there is no separate `lm-link-alpha-beta` key).
- **Models and data:** none beyond tensor buffers; message-size payloads are `float32` random tensors. A one-line cell reads `Qwen3-8B`'s hidden size and a token count to translate the measured α–β into the cost of one tensor-parallel all-reduce of activations.
- **Packages:** `nccl-tests` built from source (CUDA, NCCL, optionally MPI), `cuda-samples` (`p2pBandwidthLatencyTest`; its location inside the repository is confirmed at authoring), `nvbandwidth`, `numactl`, `lspci`/`pciutils`, `networkx`, `torch==2.15.0` with NCCL ≥ 2.18 (the `NCCL_*` variables used are those of the NCCL environment-variable page, verified reachable).

#### Narrative arc

1. **The question.** Two `nvidia-smi topo -m` matrices, parsed and drawn side by side (the 8-GPU node and the build box; a matrix is a capture, not a transfer), then the single number that differs most: all-reduce bus bandwidth of a 256 MiB message on all GPUs of the node and, on the T2 pair, over the NVLink bridge and with `NCCL_P2P_DISABLE=1` (to be measured). The book already states roughly 2 GB/s of NCCL busbw on its build box at default settings and about 20 GB/s for a raw copy; the cell quotes those as the book's earlier measurement and does not re-run them (no multi-GPU run happens on the build box), and uses the `NCCL_P2P_DISABLE=1` series as the live host-staged analogue.
2. **A server is a graph (subsection).** GPUs, NVSwitch chips, PCIe switches, CPU sockets, NICs and NVMe drives as nodes; links as edges with bandwidth and latency; SXM versus PCIe form factors; HGX-style 8-GPU boards (every GPU wired to every NVSwitch) versus PCIe boxes with bridges. `topo_graph` builds the graph from the parsed matrix and `links_table()`; `fig_lm_hw_server_graph` draws two graphs with edge width ∝ bandwidth.
3. **The links (subsection).** `tab_lm_hw_links` and a one-line formula for each: NVLink generations (per-GPU bandwidth from vendor pages), NVSwitch, PCIe Gen 3 to 6 (rate and encoding give per-direction bandwidth; computed, cited to PCI-SIG), CPU–GPU links (PCIe versus NVLink-C2C, from the Grace Hopper page), AMD Infinity Fabric (MI300X page), plus Ethernet and InfiniBand rows that 5.5 completes. Reading rule from `sec_hardware`: every chip boundary costs about an order of magnitude; the table gives the datasheet rungs. The A100 appears in two rows: NVLink on SXM boards behind NVSwitch (600 GB/s per GPU, as printed, bidirectional) and the NVLink bridge of the PCIe card (600 GB/s between the two cards of a pair, "for up to two GPUs" in the datasheet footnote) with PCIe Gen4 64 GB/s for every other pair. `topo.txt`, the committed `nvidia-smi topo -m` of the T3 node, decides which row labels the node's measurements; the notebook prints both.
4. **Peer-to-peer and RDMA (subsection).** What GPUDirect P2P requires (a P2P-capable path: NVLink, or a common PCIe root complex or switch with ACS and IOMMU settings that allow it) and what happens without it (the book's host-staged path); GPUDirect RDMA (a NIC reading and writing GPU memory directly; one NIC per GPU on a node built for scale-out); NUMA and CPU/NIC affinity (a process whose GPU, NIC and memory are on different sockets pays the socket link). Commands: `nvidia-smi topo -p2p r`, `nvidia-smi topo -p2p w`, `nvidia-smi topo -p2p n` (capability matrices; the cell records them), `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH` to print which transport each channel uses (`P2P`, `SHM`, `NET`), and `NCCL_TOPO_DUMP_FILE=topo.xml` to save NCCL's own view.
5. **Copy engines versus SM copies; the latency floor (subsection).** A copy is done by DMA engines (`cudaMemcpy`) or by kernels doing loads and stores (what NCCL does on NVLink); the build-box NCCL finding of `sec_multi_gpu` is exactly this distinction. Measurements: nvbandwidth `_ce` versus `_sm` test cases and the latency test; the smallest transfer time bounds `α`.
6. **α–β per link class (subsection).** D1; fits for NVLink pair, PCIe same-switch pair, cross-socket pair, host-staged pair; the map `fig_lm_hw_alphabeta`.
7. **Bisection and all-to-all (subsection).** D2; measurement of all-to-all versus all-reduce on the node and the build box; the ratio plot.
8. **Where each parallel axis goes (subsection).** Table `tab_lm_hw_axes` (axis, collective, how often per layer or per step, latency sensitivity, which link it needs) with the formulas *pointing to* 6.3 to 6.7 and not derived here. The one derived number: the time of a tensor-parallel all-reduce of the activations of 8,192 tokens (`8192 × 4096 × 2` bytes for Qwen3-8B in bf16) on NVLink and on the host-staged path, from the fitted `(α, β)`; ratio printed (to be measured). Conclusion stated from that ratio: tensor and expert parallelism need the fast domain, data and pipeline parallelism tolerate the slow one.

#### Microbenchmarks (exact commands, parsers, plots)

**P1. Topology capture.** Commands, each logged raw: `nvidia-smi topo -m`; `nvidia-smi topo -p2p r`; `nvidia-smi nvlink -s` (per-link status and speed); `lscpu`; `numactl -H`; `lspci -tv`; on nodes with RDMA NICs `ibstat` and `ibv_devinfo` (5.5). Parser `parse_topo_matrix`: the header row lists `GPU0 … GPUn`, then `NIC0 …` if present, then `CPU Affinity`, `NUMA Affinity`, `GPU NUMA ID`; each row starts with the device label; cells are one of `X` (self), `SYS`, `NODE`, `PHB`, `PXB`, `PIX`, `NV#` (number of NVLinks); the legend below the matrix is stored verbatim. Tested on two committed samples (node and build box).

**P2. p2pBandwidthLatencyTest.** Command: run the compiled sample with no arguments (it iterates all GPU pairs); output is a device list, a connectivity matrix, then blocks titled "Unidirectional P2P=Disabled Bandwidth Matrix (GB/s)", "Unidirectional P2P=Enabled Bandwidth (P2P Writes) Matrix (GB/s)", the bidirectional pair, and "P2P=Disabled Latency Matrix (us)" and "P2P=Enabled Latency (P2P Writes) Matrix (us)", each followed by a header row `D\D 0 1 …` and one row per device. Parser `parse_p2p`: for every title line ending in `Matrix (GB/s)` or `Matrix (us)` read the next header row and the following rows; return arrays keyed by title. Plot `fig_lm_hw_p2pmatrix`: two heatmaps (bandwidth GB/s, latency µs), rows = source GPU, columns = destination GPU, cell text = value, cell border colour = link class from P1 (`NV#`, `PIX`, `PHB`, `SYS`); the diagonal is masked; separate colour scales for the two panels; one figure per machine.

**P3. nvbandwidth device-to-device.** Commands: `nvbandwidth -p device_to_device -b 512 -i 5` (prefix selects every device-to-device case, README-verified option `-p`), and `nvbandwidth -t device_to_device_latency_sm` for the latency matrix (the README lists this test among those affected by `--pair`). Parser: `parse_nvbandwidth` (5.3). If P2P is unavailable the tool may skip cases; the log records the skip text.

**P4. NCCL collectives.**
- Single process, `g` GPUs: `all_reduce_perf -b 8 -e 4G -f 2 -g g -w 10 -n 50 -c 0 -J out.json` for `g ∈ {2, 4, 8}` on the node and `g = 2` on the T2 pair; also `all_gather_perf`, `reduce_scatter_perf` and `alltoall_perf` with the same flags at `g=8` on the node (and `g=2` on the T2 pair). Flags verified in the nccl-tests README: `-b/-e/-f` size sweep, `-g` GPUs per thread, `-w` warm-up, `-n` iterations, `-c` correctness check count (0 disables the slow check), `-J` JSON file. The stdout table is the parsed artifact; the JSON is committed as a cross-check.
- Algorithm controls: `NCCL_ALGO=Ring` and `NCCL_ALGO=Tree` for the `g=8` all-reduce sweep only (D3 and exercise 6 need one pair of series; other combinations are cut); `NCCL_P2P_DISABLE=1` for the host-staged emulation, at `g=8` for all-reduce and all-to-all on the node and at `g=2` on the T2 pair; on the T2 pair `NCCL_SHM_USE_CUDA_MEMCPY=1` versus unset under `NCCL_P2P_DISABLE=1` (the book's recovery, one pair of series). Each setting is a separate series with its environment string stored in `cmd`.
- Repeats: 3 independent invocations; the per-size median of the out-of-place time.
- Parser `parse_nccl_tests`: skip lines starting `#`; each data line has 13 whitespace-separated fields (size, count, type, redop, root, then out-of-place time, algbw, busbw, wrong, then the same four in place); assert `wrong == 0` when `-c` was nonzero.
- Plots: `fig_lm_hw_nccl_node`: x = message bytes (log2, 8 B to 4 GiB), y = bus bandwidth GB/s (log); series all-reduce `g=2,4,8` (link class from `topo.txt`: NVLink on an NVSwitch node; on a PCIe node only the bridged `g=2` pair is NVLink and `g=4,8` cross PCIe), all-reduce `g=8` with `NCCL_P2P_DISABLE=1`, all-to-all `g=8`; dashed horizontal at the vendor NVLink figure per direction (A100 datasheet NVLink 600 GB/s is bidirectional: the cell divides by two and says so); right panel: time in µs against size for sizes ≤ 1 MiB (log-log) with the fitted `α` as a horizontal asymptote. `fig_lm_hw_alltoall_ratio`: bars of `busbw(all-to-all)/busbw(all-reduce)` at 256 MiB for each machine and setting.

**P5. Fits.** `d2l.fit_alpha_beta` applied to (a) point-to-point times from P3-style sweeps for each link class, and (b) collective times from P4, per `(op, g)`; weights `1/t²`; 2,000-draw bootstrap over repeats for 95% intervals. Output table `tab_lm_hw_alphabeta` (link class or collective, `α` µs, `β` GB/s with interval, `n_{1/2}=αβ` bytes, fit residual). Plot `fig_lm_hw_alphabeta`: scatter of `(α, β)` on log-log axes, one marker per fit, dotted iso-lines of constant `n_{1/2}`.

#### Derivations

**D1. The α–β model of a link and its fit.**

*Statement.* A transfer of `m` bytes on a link takes `t(m) = α + m/β`. Fitting measured `(m_i, t_i)` by weighted least squares with weights `w_i = 1/t_i²` (minimizing relative error) gives closed-form estimates; the message size at which half of the asymptotic bandwidth is reached is `n_{1/2} = αβ`.

*Assumptions.* Time is affine in size over the fitted range; noise is multiplicative; sizes span at least two decades on each side of `n_{1/2}`; the link is otherwise idle.

*Steps.* (1) Write `t_i = a + c·m_i` with `a = α`, `c = 1/β`. (2) Weighted normal equations: `[Σw, Σw m; Σw m, Σw m²][a; c] = [Σw t; Σw m t]`. (3) Solve the 2×2 system; `β = 1/c`. (4) Covariance `σ²(XᵀWX)^{-1}` with `σ²` the weighted residual variance; delta method for `β`. (5) Half-bandwidth: achieved rate `m/t = β/(1 + αβ/m)` equals `β/2` at `m = αβ`. (6) Collective models inherit the form `t(S) = α_coll + S·κ(k)/β` with `κ` the algorithm's byte factor (`2(k−1)/k` for ring all-reduce per `eq_ring_traffic`), so the fitted slope of a collective is `κ/β`, not `1/β`.

*Numerical check.* Generate synthetic data with known `(α, β)` and 3% multiplicative log-normal noise (seed 0, 40 sizes from 8 B to 4 GiB); assert recovered parameters within 2%; bootstrap 95% intervals contain the truth in at least 90 of 100 replicate datasets; assert that unweighted least squares on the same data mis-estimates `α` by more than the weighted fit (the reason for the weights).

**D2. Bisection bandwidth: why all-to-all suffers more than all-reduce on a tree.**

*Statement.* On a full crossbar of `k` GPUs with per-GPU, per-direction bandwidth `β`, an all-to-all in which each GPU holds `S` bytes (`S/k` for each destination) takes `t_a2a = ((k−1)/k)·S/β` and a ring all-reduce of `S` bytes takes `t_ar = (2(k−1)/k)·S/β` (bandwidth terms). On a two-level tree with two switches of `k/2` GPUs joined by one uplink of per-direction bandwidth `β_u`, the ring all-reduce is unchanged if `β_u ≥ β` (a ring crosses the uplink once per direction per step), but the all-to-all takes at least `t_a2a^tree = (k/4)·S/β_u`, so `t_a2a^tree/t_a2a = (k²/(4(k−1)))·(β/β_u)`.

*Assumptions.* Uniform traffic; full-duplex links; ideal routing; two groups of equal size; the ring order is chosen so that it crosses the uplink exactly twice (once per direction); latency terms neglected.

*Steps.* (1) Crossbar all-to-all: each GPU sends `(k−1)S/k` bytes through its own port: `t = (k−1)S/(kβ)`. (2) Crossbar ring all-reduce: `2(k−1)` steps each moving `S/k` per link: `t = 2(k−1)S/(kβ)` (`eq_ring_traffic`). (3) Tree, all-to-all: each of the `k/2` GPUs in one group sends `S/k` to each of the `k/2` GPUs in the other, i.e. `(k/2)(S/k) = S/2` bytes across; the group total across the uplink in one direction is `(k/2)(S/2) = kS/4`. (4) The uplink carries `kS/4` bytes per direction at `β_u`: `t ≥ kS/(4β_u)`. (5) Tree, ring all-reduce: at each of the `2(k−1)` steps the uplink carries one chunk `S/k` per direction, so its time per step equals that of a leaf link when `β_u ≥ β`. (6) The ratio follows. For `k = 8`: `64/28 ≈ 2.29` times `β/β_u`; it grows linearly in `k`. (7) Reading: all-to-all traffic is bisection-limited, all-reduce traffic is not; this is why expert parallelism (all-to-all) is the first parallelism to need the fast domain (6.6).

*Numerical check.* `d2l.tree_link_loads(k, groups=2, pattern)` routes the traffic patterns through the tree and returns bytes per link; assert for `k ∈ {4, 8, 16}` that the busiest-link time equals the formulas above exactly (integers), for both patterns, and that the ring pattern with the best ring order loads the uplink exactly twice per step. Then compare with the measured `busbw(all-to-all)/busbw(all-reduce)` on the NVSwitch node (prediction near 1 after the busbw normalization, since both formulas normalize to per-GPU link bandwidth) and under `NCCL_P2P_DISABLE=1` on the node and on the T2 pair (prediction below 1; the cell prints the measured ratio next to the model's; the host-staged path is a *star through host memory*, not a tree of switches, so the model is only qualitative there, and the text says so).

**D3. What the bus-bandwidth normalization hides.**

*Statement.* For `op` with factor `φ_op(k)` in `d2l.nccl_busbw_factor`, `busbw = φ·S/t` equals the per-GPU link bandwidth if the algorithm saturates each link once; `busbw` for different `op` and `k` are comparable *only* under that assumption.

*Steps.* (1) Per-GPU bytes sent for ring all-reduce `2(k−1)S/k`, for all-gather and reduce-scatter and all-to-all `(k−1)S/k`. (2) Dividing by `t` gives the bandwidth each GPU's port sustains, i.e. `φ·S/t`. (3) If the algorithm uses fewer links than a port has (tree, `NCCL_ALGO=Tree`) or more (NVLS in-switch reduction), `busbw` can fall below or exceed the port bandwidth, so it is *not* a link measurement (5.5 returns).

*Numerical check.* With the committed sweeps assert that `φ·S/t` for `NCCL_ALGO=Ring` on the NVSwitch node does not exceed the datasheet per-direction NVLink bandwidth by more than measurement noise, and print the ratio for the default algorithm.

#### From-scratch implementation

Components (≈ 150 lines): `parse_topo_matrix` (≈ 30 lines), `parse_p2p` (≈ 20), `parse_nccl_tests` (≈ 20), `topo_graph` (≈ 20), `tree_link_loads` (≈ 25), `coll_time` and `nccl_busbw_factor` (≈ 20), `links_table` with the PCIe arithmetic (≈ 15).

Numerical tests: parsers on committed raw samples (exact arrays); `coll_time('allreduce', ..., algo='ring')` reproduces `eq_ring_traffic` for `α=0`; `tree_link_loads` exact for three `k`; busbw factor table against the nccl-tests page's formulas (four rows); PCIe rates against the PCI-SIG generation table (Gen 3: 8 GT/s, Gen 4: 16 GT/s, Gen 5: 32 GT/s with 128b/130b; Gen 6 is listed with raw `lanes × 64 GT/s / 8` and a note that flit overhead is excluded).

#### In production

- NCCL (`NCCL_DEBUG`, `NCCL_ALGO`, `NCCL_PROTO`, `NCCL_TOPO_DUMP_FILE`, `NCCL_P2P_DISABLE`, `NCCL_SHM_USE_CUDA_MEMCPY`) is the library; the from-scratch `coll_time` is its test oracle: the cell asserts the model's prediction against the measured time for every `(op, g)` within the section's tolerance and prints the algorithm NCCL chose (`NCCL_DEBUG=INFO` prints ring or tree per message size in recent versions; the author checks the format of the pinned version).
- `torch.distributed` collectives through the NCCL backend reproduce nccl-tests within noise: one cell runs a 256 MiB `all_reduce` in PyTorch and compares (≤ 10%).
- Stas Bekman's `all_reduce_bench.py` and the NCCL tests are the field tools; `nvbandwidth` for point-to-point and `p2pBandwidthLatencyTest` as the classic baseline.

#### Headline experiment

*Hypothesis.* An α–β model fitted on small sweeps predicts all-reduce time for `g ∈ {2, 4, 8}` on the NVSwitch node within 15% for messages ≥ 1 MiB, and the ratio of all-to-all to all-reduce bus bandwidth on a host-staged or tree-like path is below its value on the switched node, as D2 predicts qualitatively.

*Setup.* One 8× A100 node (either interconnect: the prediction table carries an NVSwitch row and a PCIe-pairs row, `topo.txt` decides which is reported, and on a PCIe-pairs node the NVLink clause of the hypothesis holds for `g = 2` only) and the T2 pair for the `g = 2` comparison. Sweeps as in P4 with 3 repeats. Fit set: sizes 8 B to 64 MiB (even-indexed powers of two); test set: odd-indexed sizes plus 128 MiB to 4 GiB. Fit `(α_coll, β_coll)` per `(op, g, algo)` on the fit set; predict the test set. Independent comparison: the cell converts each fitted slope `κ(k)/β_coll` to a per-link bandwidth through the busbw factor `φ(k)` and compares it with the point-to-point `β` of P3.

*Procedure.* P1 to P5; additionally rerun P4 with `NCCL_P2P_DISABLE=1` for the host-staged emulation. Record clocks and the NCCL version.

*Metrics.* Relative prediction error per test size; `α_coll, β_coll` with intervals per `(op, g)`; `busbw` at 256 MiB; ratio `busbw_a2a/busbw_ar`; `n_{1/2}`.

*Plot specification.* `fig_lm_hw_nccl_node` and `fig_lm_hw_alltoall_ratio` as above; `fig_lm_hw_alphabeta_pred`: x = message bytes (log2), y = measured time (µs, log), points = test set, line = fitted model, shaded ±15% band.

*Expected outcome.* Direction only: NVLink-path bus bandwidth is orders of magnitude above the host-staged path; the fitted `α` is larger on the host-staged path; the all-to-all ratio drops off the switched node; absolute values are to be measured. The "Demystifying NCCL" paper (arXiv 2507.04786) gives NCCL's protocol latencies and bandwidth regimes; the notebook compares the fitted `α` with it after the author reads the paper's tables (no values are quoted in this blueprint).

*Noise check.* 3 independent invocations; the per-size MAD; a claim about two series requires a gap > 3× the larger MAD at the sizes compared; cross-socket versus same-socket placement recorded by `NUMA Affinity` from P1.

*T1 reduction.* One build-box GPU: the host-to-device and device-to-host series (`nvbandwidth` host copies and `hw_h2d_bench`, fitted with `fit_alpha_beta`), the topology capture, and the parser and fit tests; no GPU-to-GPU traffic. The collective part runs on the T2 pair at `g = 2` (NVLink-bridge path versus `NCCL_P2P_DISABLE=1`, with and without `NCCL_SHM_USE_CUDA_MEMCPY=1`) with the same fit and plots; the node's `NCCL_P2P_DISABLE=1` series is analysed from the committed log. The book's finding that busbw does not improve from 2 to 4 GPUs on the staged path is cited from `sec_multi_gpu`, not re-measured (it needs four GPUs on the build box).

*Artifacts committed (T2+).* `lm-hw-node-p2p` (P1 to P5 raw outputs, JSONL results, JSON from nccl-tests, and `alpha_beta.json`), `lm-hw-t2pair-p2p`, `lm-hw-buildbox-p2p` (topology capture and host-to-device series), `links.csv`, figures as SVG.

#### Figures and tables

- `tab_lm_hw_links`, `tab_lm_hw_axes`, `tab_lm_hw_alphabeta`; `fig_lm_hw_server_graph`; `fig_lm_hw_p2pmatrix`; `fig_lm_hw_nccl_node`; `fig_lm_hw_alltoall_ratio`; `fig_lm_hw_alphabeta`; `fig_lm_hw_alphabeta_pred`; equations `eq_lm_hw_alphabeta`, `eq_lm_hw_bisect_a2a`, `eq_lm_hw_busbw`.

#### Pitfalls and misconceptions

- NVLink figures on vendor pages are bidirectional aggregates per GPU; a per-direction measurement is half.
- `busbw` is not a measurement of a link when the algorithm does not use each link once (D3).
- One process driving 8 GPUs (`-g 8`) and eight processes (`mpirun -np 8 … -g 1`) differ in CPU launch behaviour; state which, and keep it fixed.
- The first NCCL call includes communicator setup; warm-up iterations must be discarded (`-w`).
- `-c 1` (the default) checks results and slows large-message runs; use `-c 0` for timing and a separate run with `-c 1` for correctness.
- P2P disabled by the driver or by BIOS settings (ACS, IOMMU) looks like a slow fabric; read the capability matrix before reading bandwidth.
- MIG, power capping and shared hosts change the numbers; record them.

#### Summary (draft bullets)

- A server is a graph of GPUs, switches, CPUs, NICs and drives; its topology, read from `nvidia-smi topo -m`, decides which paths are fast.
- NVLink and NVSwitch give memory-class bandwidth between GPUs; PCIe and host staging give one to two orders of magnitude less, with larger fixed costs.
- Each link class has an α and a β; fitting them from message-size sweeps gives `n_{1/2}=αβ` and predicts collective times.
- All-to-all traffic is limited by bisection bandwidth while ring all-reduce is not, so all-to-all degrades far more on a tree or a host-staged path.
- The placement rule for parallel axes follows from the measurements: the most frequent and most latency-sensitive collectives belong on the fastest links.

#### Exercises

1. **(check)** A topology matrix shows `PHB` between GPU0 and GPU1 and `SYS` between GPU0 and GPU2. What does each mean for traffic between them, and which pair would you put in one tensor-parallel group? *Sketch:* `PHB` crosses a PCIe host bridge within one CPU socket, `SYS` crosses the inter-socket link; prefer the `PHB` pair.
2. **(check)** Why is the `busbw` of an all-to-all divided by `(k−1)/k`, and why does a value above the per-GPU link bandwidth indicate something other than a single link? *Sketch:* each GPU sends `(k−1)/k` of its data; busbw above port speed means more links than one are being used (e.g. in-switch reduction) or the messages stay in cache.
3. **(derive)** Extend D2 to a tree with `g` leaf groups of `k/g` GPUs joined by a root switch with uplink `β_u` per group. Give the all-to-all time and the ratio to the crossbar. *Sketch:* each group sends `(k/g)·(k−k/g)·S/k` bytes up; `t ≥ (k/g)(1−1/g)S/β_u`; ratio `= (k/g)(1−1/g)(k/(k−1))·β/β_u`.
4. **(derive)** Show that fitting `t = α + m/β` by unweighted least squares over sizes spanning eight decades estimates `β` well but `α` poorly, and that weights `1/t²` fix it. *Sketch:* large sizes dominate the unweighted sum of squares so the intercept is determined by few points with large absolute noise; relative-error weights equalize the contribution of small sizes.
5. **(measure)** Predict, then measure with nvbandwidth, the latency matrix of your machine and the bandwidth matrix for `_ce` and `_sm` copies. Which link classes differ most between the two copy methods? State the noise check.
6. **(measure)** On your machine run all-reduce with `NCCL_ALGO=Ring` and `Tree` for sizes 1 KiB to 1 GiB and find the crossover size; compare with the formula of the `sec_multi_gpu` exercise using your fitted `α` and `β`.
7. **(open)** Draw the topology graph of a machine you have access to from `topo -m`, annotate every edge with a measured bandwidth and latency, and propose the placement of tensor, data and pipeline parallel groups for a 16-GPU job on two such machines, justified by your α–β fits.

#### Further reading

- Papers: [Evaluating Modern GPU Interconnect](https://arxiv.org/abs/1903.04611) (PCIe, NVLink, NVSwitch measurements and methodology); [Demystifying NCCL](https://arxiv.org/abs/2507.04786) (protocols, algorithms, latency constants); [Collective Communication for 100k+ GPUs](https://arxiv.org/abs/2510.20171) (new; verified title; NCCLX at scale, referenced for what changes beyond a node).
- Documentation and tools: [NCCL user guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/) and its [environment variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) (new; verified); [nccl-tests](https://github.com/NVIDIA/nccl-tests) and its [performance page](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md) (new; verified; busbw formulas); [CUDA samples](https://github.com/NVIDIA/cuda-samples) (`p2pBandwidthLatencyTest`); [nvbandwidth](https://github.com/NVIDIA/nvbandwidth); [NVIDIA NVLink page](https://www.nvidia.com/en-us/data-center/nvlink/) (new; verified; link generations).
- Teaching: [Scaling Book](https://jax-ml.github.io/scaling-book/) (GPU networking chapter); [Scaling Book: sharding](https://jax-ml.github.io/scaling-book/sharding/) (from Codex 4.7; placement and communication models); Stas Bekman's [ML Engineering Open Book](https://github.com/stas00/ml-engineering) (network benchmarks).
- Code: PyTorch `torch.distributed` NCCL backend; `nccl-tests` sources as the reference implementation of the timing loop.

#### Risks and open decisions

- *The node's interconnect is unknown until `topo.txt` is read* (NVSwitch SXM, NVLink-bridged PCIe pairs, or PCIe with no bridge). The section runs unchanged in all three cases: the series labelled NVLink are whatever `topo.txt` says (a bridged pair covers `g = 2` only), the links table carries both A100 rows, and the text states which hardware was measured; a committed SXM log (from a previous allocation) supplies NVSwitch curves if available, and the T2 pair supplies the bridge-versus-host contrast in any case.
- *`p2pBandwidthLatencyTest` moved or renamed in `cuda-samples`.* Fallback: nvbandwidth device-to-device matrices carry the same information; the parser for the sample is tested on the committed raw text only.
- *NCCL version differences change algorithm selection or output format.* Fallback: pin the version in the log; the parser accepts the column layout checked at authoring and fails with a clear message otherwise.
- *No P2P and no NVSwitch on T3 either.* Then the node resembles the build box; the section still holds and the contrast with NVLink comes from the committed vendor data and from the datasheet row, labelled "not measured here".
- *Shared host noise.* Fallback: run in an exclusive allocation; record `load average` and other GPU users in the log; discard runs where another process used the GPUs.

### 5.5 Networks between servers

- **Label:** `sec_lm_networks` · **Tier:** T4 (core: 2 nodes of 8× A100 with an RDMA fabric; optional: one 4-node pass; the notebook analyzes committed logs, produced once per release) plus a T0 simulator · **Reduced T1 config:** the build box runs no GPU-to-GPU traffic, so the network cells use a gloo/CPU process group with 2 and 4 ranks over loopback TCP (each rank a "node", the transport a socket; labelled "not a datacenter network"), including the hierarchical run with gloo for the inter-group stage; the NCCL version of the same idea runs on the T2 pair with NCCL forced onto TCP loopback (`NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo`, 2 ranks, each GPU a one-GPU "node") · **Budget:** T4 production ≈ 1 h on 2 nodes (≈ 2 node-hours with the perftest runs; the optional 4-node pass ≈ 1 h, 4 node-hours); notebook analysis ≈ 5 min on CPU; T1 ≈ 10 min, small memory · **Codex units:** 4.7, 6.2 (communication cost models, network half)
- **Builds on (book):**
  - `sec_multi_gpu`: `eq_ring_traffic`, the algbw/busbw convention, and the exercises "Latency in the allreduce" (a naive binary tree sends the full `N` at every level; its crossover with the ring is a single-level, single-bandwidth statement) and "Bandwidth conventions". 5.5 replaces the naive tree by the *pipelined double binary tree* NCCL uses, adds two levels of bandwidth, and does not re-derive the ring.
  - `subsec_mgp-bridge` (`sec_multi_gpu_concise`): the promise "a hierarchy of fabrics (NVLink inside a node, a network between nodes)" and "a second, slower bandwidth term" is what this section delivers.
  - `tab_collectives` (all-to-all as the expert-parallel collective), `subsec_hw-interconnects` (NVL72's scale-up bandwidth, taken as given).
- **Builds on (this part):** 5.4 (`d2l.coll_time`, `d2l.fit_alpha_beta`, `d2l.parse_nccl_tests`, `d2l.nccl_busbw_factor`, the node's α–β fits from `lm-hw-node-p2p`, `d2l.links_table`), 5.1 (`d2l.BenchLog`). Forward by label: 6.1 (`d2l.launch`, used only to submit the multi-node job; the section's scripts are launchable with plain `srun`/`mpirun` too), 6.6 (expert parallelism), 6.7 (placement), 5.6 (scale-up versus scale-out).
- **Exports (`#@save`):**
  - `d2l.ring_allreduce_time(S, G, alpha, beta)`, `d2l.tree_allreduce_time(S, n_nodes, g, a_in, a_out, beta, pipelined=True)` (double binary tree across nodes with an intra-node chain), `d2l.hier_allreduce_time(S, n_nodes, g, a_in, b_in, a_out, b_out, rails=1)` (reduce-scatter in the node, all-reduce across nodes on the local shard, all-gather in the node), `d2l.a2a_inter_node_time(S, n_nodes, g, a_out, b_out, a_in, b_in)`, `d2l.sharp_allreduce_time(S, alpha, beta)` (the in-network lower bound of D3).
  - `d2l.sim_collective(topo, algo, S, chunks) -> float`: event-driven α–β simulator of ring and tree all-reduce on a graph of full-duplex links; the oracle for the closed forms.
  - `d2l.hier_allreduce(tensor, intra_group, inter_group)`: the hand-written PyTorch composition (reduce-scatter, all-reduce, all-gather) used by N3.
  - `d2l.fat_tree(radix, tiers, oversub=1.0, rails=1) -> dict` (endpoint, switch and link counts and a `networkx` graph).
  - `d2l.parse_perftest(text, kind)` for `ib_write_bw` and `ib_write_lat` tables.
  - Data: `DATA_HUB['lm-hw-cluster-nccl']` (tar: for 2 nodes and, if the optional pass ran, 4 nodes, nccl-tests stdout and JSON for every variant below; `ibstat`, `ibv_devinfo`, `iblinkinfo` or the Slurm node list with leaf identifiers; perftest output; the NCCL version and `NCCL_DEBUG=INFO` header of one run showing channels and transports), `DATA_HUB['lm-hw-t2pair-net']` (NCCL over TCP loopback on the T2 pair) and `DATA_HUB['lm-hw-buildbox-net']` (gloo/CPU loopback). `lm-hw-cluster-nccl` also includes an `alpha_beta.json` with the schema of 5.4's (`scope: "inter_node"`, for 2 nodes and, if run, 4 nodes: all-reduce with `algo` in `{ring, tree}`, all-gather, reduce-scatter, all-to-all, and point-to-point from `ib_write_bw` and `ib_write_lat`), fitted with `d2l.fit_alpha_beta`; 6.x and 13.4 consume it together with 5.4's file.
- **Models and data:** none; `float32` buffers. No licenses.
- **Packages:** nccl-tests built with `MPI=1`; `perftest` (`ib_write_bw`, `ib_write_lat`; flags confirmed with `-h` on the host); `rdma-core` tools (`ibstat`, `ibv_devinfo`, `iblinkinfo`); Slurm `srun` or OpenMPI `mpirun`; `torch==2.15.0` with NCCL ≥ 2.18; `networkx`.

#### Narrative arc

1. **The question.** The cell loads the committed 2-node (and, if run, 4-node) `all_reduce_perf` sweeps and plots bus bandwidth next to the intra-node one from 5.4. The measured symptom: bus bandwidth for a 1 GiB all-reduce across nodes is a fraction of the intra-node value (to be measured; the cell prints the ratio and the per-node NIC aggregate from `ibstat` so the reader sees which wall was hit).
2. **NICs, links and fabrics (subsection).** Table `tab_lm_hw_nics`: link rates of InfiniBand generations and of Ethernet (EDR/HDR/NDR/XDR and 100/200/400/800 GbE; rates from the NVIDIA InfiniBand and ConnectX pages, converted to GB/s in code); one NIC per GPU as the scale-out design; `tab_lm_hw_fabrics`: InfiniBand, RoCE and Ultra Ethernet compared on transport, flow control (credit-based versus priority flow control), congestion control (ECN-based schemes), routing (static, ECMP hashing, adaptive routing, packet spraying), loss handling; cited to the NVIDIA networking pages, the Ultra Ethernet Consortium and the two SIGCOMM 2024 production papers (Meta, Alibaba; cited by title).
3. **The latency budget of a message (subsection).** One-way latency = software and NIC + switch hops × per-hop latency + cable propagation (`5 ns/m` in fibre, physics, computed in the cell) + serialization `m/β`. N1 measures the first two terms with `ib_write_lat` between two nodes under the same leaf and under different leaves when the scheduler exposes that; the figure stacks the measured components. Contrast with intra-node `α` from 5.4.
4. **Topologies (subsection).** Rail-optimized fat trees (GPU `i` of every node on leaf switch `i`), non-blocking versus oversubscribed (`o:1`), rail-only designs that rely on NVLink for cross-rail traffic (Rail-only paper), two- and three-tier counts (D4). `d2l.fat_tree` builds the graph; `fig_lm_hw_fattree` draws a radix-8 two-tier tree with `o = 2` and its rail-optimized variant. Congestion: flow collisions with hashed routing, adaptive routing and packet spraying as remedies (text, cited; no experiment).
5. **Collectives across nodes (subsection).** Hierarchical algorithms (D2), NCCL's ring, tree and in-network (SHARP) algorithms (D1, D3), bus bandwidth versus algorithm bandwidth across nodes, all-to-all across nodes (D3). Each result is followed by the measured check.
6. **Reading NCCL's choice (subsection).** `NCCL_DEBUG=INFO` header of one run: channels, rings, trees, transports (`NET/IB`, `P2P`, `SHM`); `NCCL_ALGO`, `NCCL_PROTO`, `NCCL_MAX_NCHANNELS`, `NCCL_IB_HCA`, `NCCL_COLLNET_ENABLE` as levers (names from the NCCL environment-variable page); a diagnostic example: restrict NCCL to one NIC with `NCCL_IB_HCA` and watch the busbw fall by about the number of NICs, which is the "unintended slow path" case of Codex 4.7 (the expected factor is `g` from D2; the measurement is the claim).
7. **From model to prediction (subsection).** Feed measured `(α_in, β_in)` (5.4) and `(α_out, β_out)` (N1, N2) into `hier_allreduce_time` and compare with NCCL across 2 nodes (and 4, if run) for sizes ≥ 16 MiB: the A5 target of 15% (reference mode uses these logs).

#### Microbenchmarks (exact commands, parsers, plots)

**N1. RDMA point-to-point (`perftest`).**
- Bandwidth: on the server node `ib_write_bw -d <hca> -a --report_gbits -F`; on the client `ib_write_bw -d <hca> -a --report_gbits -F <server>`; for RoCE add `-x <gid_index>`. `-a` sweeps message sizes 2 B to 8 MiB; `-F` suppresses the CPU-frequency-governor error; `<hca>` is chosen from `ibv_devinfo` to match the GPU under test (the rail). Single message size for the headline: `-s 1048576 -n 5000 -q 1`. Repeat 3 times per node pair.
- Latency: `ib_write_lat -d <hca> -a -F [<server>]`. Pairs: (a) two nodes under one leaf, (b) two nodes under different leaves, when the allocation allows choosing them; each pair and `<hca>` recorded.
- Parser `parse_perftest`: skip lines until the header containing `#bytes`; for bandwidth rows read `bytes, iterations, BW peak [Gb/sec], BW average [Gb/sec], MsgRate [Mpps]`; for latency rows `bytes, iterations, t_min, t_max, t_typical, t_avg, t_stdev, 99% percentile, 99.9% percentile` in µs; the `-h` output of the installed version is stored in the log and the parser asserts the header text it was written for.
- Plot `fig_lm_hw_ibsweep`: x = message bytes (log2); left axis average bandwidth (Gb/s, linear; also shown as GB/s on a twin axis), right axis typical latency (µs, log); dashed horizontal at the NIC line rate from `ibstat`; annotate `n_{1/2}`.

**N2. NCCL across nodes.**
- All nodes: `srun -N <n> --ntasks-per-node=8 --gpus-per-node=8 all_reduce_perf -b 8 -e 4G -f 2 -g 1 -w 10 -n 50 -c 0 -J out.json` (or the `mpirun -np $((8n)) -N 8 --bind-to none` equivalent; the README's multi-node example uses `-g 1` with one rank per GPU). The same for `alltoall_perf`, `all_gather_perf`, `reduce_scatter_perf`.
- Variants, one series each (the core list, run at 2 nodes): `NCCL_ALGO=Ring`, `NCCL_ALGO=Tree` (default protocol); `NCCL_IB_HCA` restricted to one NIC; the `AND` and `DIV` split runs below; `NCCL_COLLNET_ENABLE=1` only where the fabric has SHARP (otherwise the series is absent and the text says so). Cut to fit a handful of T4 runs: the `NCCL_PROTO` series (`Simple`, `LL128`, `LL`; N5 uses the default protocol) and the `NCCL_MAX_NCHANNELS` series (exercise 6 asks readers to measure it); neither supports a claim of the headline experiment.
- Rail and node-only bandwidth from one binary: `NCCL_TESTS_SPLIT="AND 0x7"` runs eight simultaneous all-reduces, each with one GPU per node (purely inter-node, one per rail); `NCCL_TESTS_SPLIT="DIV 8"` runs one all-reduce per node (purely intra-node). The nccl-tests README states these semantics and that reported bandwidth is per group; the aggregate is the per-group figure times the number of groups (the cell applies this and says so).
- Parser: `parse_nccl_tests` (5.4). Plot `fig_lm_hw_nccl_multinode`: x = message bytes (log2), y = bus bandwidth (GB/s, log); series `n = 2` and, if run, `n = 4` nodes (default algorithm), intra-node (from the `DIV` run), rail-only (from the `AND` run, aggregate); right panel: time (µs) for sizes ≤ 1 MiB showing the `α` floor at each `n`.

**N3. Hierarchical all-reduce by hand.** `torchrun --nnodes <n> --nproc_per_node 8 hier_bench.py` builds `intra_group` (the 8 ranks of a node) and `inter_group` (ranks with equal local index across nodes) with `dist.new_group`, and times (CUDA events, 10 warm-up, 50 timed) for sizes `2^20 … 2^32` bytes: (a) `dist.all_reduce` on the world group (NCCL default), (b) the same with `NCCL_IB_HCA` limited to one NIC, (c) `d2l.hier_allreduce`: `reduce_scatter_tensor` in `intra_group`, `all_reduce` of the shard on `inter_group`, `all_gather_into_tensor` in `intra_group`. Correctness: the three outputs agree with a CPU sum to float32 tolerance (`rtol=1e-5`).

**N4. All-to-all across nodes.** `alltoall_perf` as in N2 at `n = 2` (and `n = 4` if run; the trend with `n` in claim (c) needs the 4-node pass, and without it the notebook checks one point and shows the D3 curve as a prediction); the measured busbw is compared with the D3 prediction `β_out·(G−1)/(G−g)` where `β_out` is the rail-only bandwidth from the `AND` run. Plot `fig_lm_hw_a2a_nodes`: grouped bars (measured, predicted) per `n`.

**N5. Ring/tree crossover.** For each `algo` (default protocol) fit `(α, β)` on sizes ≤ 64 MiB (`d2l.fit_alpha_beta`), predict the crossover size between `Ring` and `Tree` from D1, and find the measured crossing by linear interpolation of the log-time curves; plot `fig_lm_hw_crossover`: x = size (log2), y = time (µs, log), measured Ring and Tree curves, fitted models as thin lines, vertical lines at predicted and measured crossover.

#### Derivations

**D1. Ring versus pipelined double binary tree: costs and crossover.**

*Statement.* Let `G` be the number of ranks, `β` the per-rank network bandwidth per direction, and `α` the latency of one step (taken equal for all steps; the two-level refinement follows). A ring all-reduce of `S` bytes takes `T_ring = 2(G−1)α + 2((G−1)/G)·S/β`. A pipelined double binary tree takes `T_tree = 2dα + 2S/β` where `d` is the number of sequential latency-bearing hops from a leaf to the root. The tree is faster iff `S < S* = G·α·β·(G − 1 − d)`.

*Assumptions.* Full-duplex links; each rank has one link of bandwidth `β`; messages are cut into enough chunks that pipelining hides all but the fill and drain latency; the two trees of the double-tree scheme split the data in halves and every rank is interior in exactly one tree; per-step `α` is the same for both algorithms (relaxed by fitting `(α, β)` per algorithm and protocol).

*Steps.* (1) Ring: `2(G−1)` steps, each moving `S/G` per link: `T_ring` (`eq_ring_traffic` plus a latency per step). (2) One binary tree carries half of the data, `S/2`. In its reduce phase every non-root rank sends `S/2` up; in its broadcast phase every interior rank sends `S/2` to each of two children. (3) Count bytes sent by one rank over both trees: in the tree where it is interior, `S/2` (up) plus `2·S/2` (down); in the tree where it is a leaf, `S/2` (up). Total `2S`, the same as the ring's `2(G−1)S/G ≈ 2S`; hence a rank's port time is `2S/β`. (4) Latency: the reduce phase has depth `d` and the broadcast phase depth `d`: `2dα`. (5) `T_tree = 2dα + 2S/β`. (6) Difference: `T_ring − T_tree = 2α(G−1−d) − 2S/(Gβ)`. (7) Set to zero for `S*`. For `d = ⌈log2 G⌉` this is `G α β (G − 1 − ⌈log2 G⌉)`. (8) With multi-GPU nodes the depth splits: `d = (g − 1)` intra-node chain hops at `α_in` plus `⌈log2 n⌉` hops at `α_out`; the cell evaluates `T_tree` with the two latencies.

*Numerical check.* `d2l.sim_collective` simulates both algorithms event by event on a graph of `G` ranks (full-duplex links, one send and one receive resource per rank, 1,000 chunks) for `G ∈ {4, 8, 16, 32}` with random `(α, β)` drawn once per case (seed 0); assert the simulated times equal the closed forms within 1% at `S ≫ Gαβ` and within 5% near `S*`, and that the simulated crossover equals `S*` within 5%.

**D2. Hierarchical all-reduce.**

*Statement.* With `n` nodes of `g` GPUs, intra-node link `(α_in, β_in)` and one NIC per GPU `(α_out, β_out)` (a *rail*), reduce-scatter in each node, then all-reduce of each GPU's shard of size `S/g` across the `n` GPUs sharing its rail, then all-gather in the node, takes
`T_hier = 2(g−1)α_in + 2((g−1)/g)·S/β_in + 2(n−1)α_out + 2((n−1)/n)·(S/g)/β_out` (ring across nodes). A flat ring over all `G = ng` ranks in which the ring leaves each node through **one** NIC takes `T_flat ≈ 2(G−1)α_out + 2((G−1)/G)·S/β_out` per ring (every step is as slow as the slowest link), so for large `S` the ratio `T_flat/T_hier → [ (G−1)/G ] / [ ((g−1)/g)(β_out/β_in) + ((n−1)/(ng)) ]`, which tends to `≈ g` when `β_in ≫ β_out` and `n` is large.

*Steps.* (1) Intra-node reduce-scatter on `g` ranks: `(g−1)` steps of `S/g` bytes: `(g−1)α_in + ((g−1)/g)S/β_in`. (2) Each rank now holds a fully reduced-within-node shard of `S/g` bytes; the `g` ranks of one node do this in parallel on different rails. (3) Inter-node ring all-reduce of `S/g` over `n` ranks: `2(n−1)α_out + 2((n−1)/n)(S/g)/β_out`. (4) Intra-node all-gather: same as step 1. (5) Sum. (6) Flat single-rail ring: cross-node edges carry `S/G` per step on one NIC. (7) Divide, take `S → ∞`. (8) If NCCL uses several rings (channels) on different NICs, the flat ring recovers the hierarchical bandwidth, which is why default NCCL is competitive with the hand-built version; the single-NIC series (N3b) isolates the effect.

*Numerical check.* (a) The closed form equals `d2l.sim_collective` on a two-level topology (full-duplex links, two bandwidths) within 2% for `(n, g) ∈ {(2,8), (4,8), (4,4)}`, seed 0; (b) the limit ratio equals `g` as `β_in/β_out → ∞`, `n → ∞` (assert within 1% at `β_in/β_out = 1e4`, `n = 1e4`); (c) `hier_allreduce` on the build box (four gloo/CPU ranks in two groups of two, gloo for the inter group) gives the same values as the world all-reduce to `rtol=1e-5`.

**D3. In-network reduction and cross-node all-to-all.**

*Statement.* (i) With in-network reduction (SHARP in InfiniBand switches, NVLS inside an NVLink switch), each endpoint sends `S` once to the switch tree and receives `S` once, so the bandwidth term is `S/β` and `T_sharp = S/β + α_sharp`, a factor `2(G−1)/G` below the ring's bandwidth term; its busbw (computed with the ring's `2(G−1)/G` factor) can therefore reach about `2β`. (ii) For a uniform all-to-all across `n` nodes of `g` GPUs with `G = ng`, the bytes each GPU must push through its own NIC are `S(1 − g/G)`, so `t ≥ S(G−g)/(G β_out)` and the measured bus bandwidth is `busbw_a2a ≈ β_out·(G−1)/(G−g)` if the NIC is the bottleneck and NVLink is not.

*Steps.* (i-1) Each endpoint's port must carry `S` out and `S` in once; ports are full duplex, so `t ≥ S/β`. (i-2) Divide the ring's `2(G−1)S/(Gβ)` by `S/β`. (ii-1) Each GPU sends `S/G` to each of `G` destinations; `g` of them (including itself) are in the same node: `S·g/G` stays on NVLink. (ii-2) The rest, `S(1−g/G)`, crosses the network on the GPU's own NIC (rail-optimized), time `S(1−g/G)/β_out`. (ii-3) `busbw = ((G−1)/G)·S/t = β_out(G−1)/(G−g)`. (ii-4) `n=2,g=8`: `15/8·β_out`; `n=4`: `31/24·β_out`; limit `β_out` as `n → ∞`: wide expert parallelism is NIC-bound. (ii-5) NCCL's PXN optimisation moves data over NVLink to the GPU on the destination's rail first, which makes the rail assumption hold even when GPU `i` sends to a GPU on another rail (cited blog).

*Numerical check.* Closed forms against the simulator for `n ∈ {2,3,4}` (exact for the counting argument, 2% for times); against the committed `alltoall_perf` logs: measured busbw within 20% of the formula using `β_out` from the `AND 0x7` run (the cell prints the ratio; a larger gap is reported, not tuned away).

**D4. Fat-tree sizing and oversubscription.**

*Statement.* With radix-`r` switches, a non-blocking two-tier fat tree (leaf-spine) connects `r²/2` endpoints with `3r/2` switches; a three-tier tree connects `r³/4` endpoints with `5r²/4` switches. Oversubscribing the leaf by `o:1` (`d` down ports and `u` up ports with `d + u = r` and `o = d/u`) gives `d = o·r/(1+o)` endpoints per leaf and cuts the bandwidth each endpoint sees across leaves to `β/o` under uniform traffic.

*Steps.* (1) Leaf: `r/2` ports down, `r/2` up (non-blocking). (2) Spine: `r` ports, one to each leaf, so `r` leaves; spines needed `r/2`. (3) Endpoints `r·r/2`; switches `r + r/2`. (4) Three tiers: `r` pods of `r/2` edge and `r/2` aggregation switches plus `(r/2)²` cores: switches `r² + r²/4`, endpoints `r·(r/2)·(r/2)`. (5) Oversubscription: a leaf with `d` down and `u` up ports has `o = d/u`; uniform cross-leaf traffic from `d` endpoints shares `u` uplinks: per-endpoint cross-leaf bandwidth `β·u/d = β/o`; within-leaf traffic is unaffected. (6) Rail-only: delete the spine, so cross-rail traffic has no network path and must hop over NVLink to the right rail first (cost: NVLink bytes, saved: spine switches).

*Numerical check.* `d2l.fat_tree` constructs the graphs for `r ∈ {4, 8, 16}` and `tiers ∈ {2, 3}`; assert counts of endpoints, switches and links equal the formulas (exact), and that the max-flow across a leaf cut with `o` oversubscription equals `β·u` (networkx max-flow, exact).

#### From-scratch implementation

Components (≈ 150 lines): closed-form time functions (≈ 35 lines), `sim_collective` (≈ 50 lines: event queue with one send and one receive resource per rank, ring and double-tree schedules over chunks), `fat_tree` (≈ 25 lines), `hier_allreduce` (≈ 20 lines), `parse_perftest` (≈ 20 lines).

Numerical tests: simulator versus closed forms (D1, D2, D3); `fat_tree` counts; `hier_allreduce` versus `dist.all_reduce` (`rtol=1e-5`, seed 0, four gloo/CPU ranks on the build box); parsers on committed samples; `coll_time` of 5.4 equals `ring_allreduce_time` for `G` ranks and a single level.

#### In production

- NCCL, as above; the NCCL tuner chooses algorithm and protocol per message size, and the notebook shows its choice from the `NCCL_DEBUG=INFO` header and tests D1's crossover against the forced `Ring` and `Tree` series.
- `torch.distributed` with the NCCL backend over `dist.new_group` implements `hier_allreduce`; PyTorch's own hierarchical or multi-ring behaviour is NCCL's; the from-scratch composition is the oracle for NCCL's multi-NIC default.
- Slurm (`srun --ntasks-per-node`, `--gpus-per-node`) and OpenMPI launch; topology files (`NCCL_TOPO_FILE`) for virtualized clusters are named only.
- Fabric telemetry (`ibstat`, `iblinkinfo`, `perfquery`) is used to read link rates; production network papers are cited for congestion control and routing.

#### Headline experiment

*Hypothesis.* (a) The hierarchical α–β model with node and network parameters measured independently (5.4 and N1/N2) predicts the NCCL all-reduce time across 2 nodes (and 4, if run) within 15% for sizes ≥ 16 MiB. (b) Restricting NCCL to one NIC lowers large-message bus bandwidth by a factor close to the number of NICs used by default (D2). (c) The measured all-to-all bus bandwidth follows `β_out(G−1)/(G−g)` as `n` grows (the trend needs the optional 4-node pass; the core run checks the single point `n = 2`). Claims (a) and (b) are core.

*Setup.* 2 nodes of 8× A100 (core) and, optionally, 4 nodes (the 3-node runs of the earlier plan are cut: two or three node counts fit the same α–β model, and the claims need one point and, for the trend, a second), one RDMA NIC per GPU where available (the actual NIC count and rate are read from `ibstat` and recorded), nccl-tests built with MPI, NCCL version recorded; sizes 8 B to 4 GiB by powers of two; the core variants of N2 plus N1, N3 (the three conditions at `n = 2`), N4 and N5; 3 independent launches per variant, each `-w 10 -n 50`.

*Procedure.* Fit `(α_in, β_in)` from the 5.4 node logs (intra-node `DIV 8` series), `(α_out, β_out)` from N1 and from the rail-only series; predict the test sizes of every variant with the closed forms; compute errors; find ring/tree crossovers (N5); compare all-to-all (N4).

*Metrics.* Per-size relative error of `T_hier` and `T_flat` against NCCL; measured/predicted busbw ratio of all-to-all; `S*` predicted versus measured per `algo` (default protocol); NIC utilization `busbw / (NIC rate)` at 1 GiB.

*Plot specification.* `fig_lm_hw_nccl_multinode`, `fig_lm_hw_hier` (x = size, y = time in ms, log-log; NCCL default, single NIC, hierarchical by hand, and the model curves; shaded ±15% band around the model), `fig_lm_hw_a2a_nodes`, `fig_lm_hw_crossover`, `fig_lm_hw_ibsweep`, `fig_lm_hw_latbudget`.

*Expected outcome.* Direction: the single-NIC series is much slower than the default at large sizes; the hand-built hierarchical all-reduce is close to the default; all-to-all busbw decreases toward the NIC rate with `n`; the ring/tree crossover moves with `n` (visible only if the 4-node pass ran) and with protocol (not measured here: the protocol series are cut). Magnitudes are to be measured. Departures of the measured crossover from `S*` are expected because the real tuner and protocol overheads differ from the idealized model, and the notebook reports the departure rather than adjusting the model (the "Demystifying NCCL" paper discusses these effects).

*Noise check.* 3 launches; per-size MAD; claims about gaps need > 3× MAD; runs with another job on the same leaf switch (visible in the scheduler) are flagged, since fabric contention changes the result; the committed log records the Slurm job ids and node lists.

*T1 reduction.* Build box, gloo/CPU process group over loopback TCP with 2 and 4 ranks (no GPU-to-GPU traffic): N2/N5-style sweeps of `all_reduce` on CPU tensors and `(α, β)` fits for the ring and a tree-like exchange built from `isend/irecv`; the D1 crossover prediction against measurement; the T0 simulator for the hierarchical cases; hierarchical `hier_allreduce` with gloo as a check of D2's algebra; all labelled "not a datacenter network". The NCCL-over-TCP-loopback version (`NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo`, 2 ranks) runs on the T2 pair and is committed as `lm-hw-t2pair-net`.

*Artifacts committed (T2+).* `lm-hw-cluster-nccl` (all raw logs, parsed JSONL and `alpha_beta.json`), `lm-hw-t2pair-net`, `lm-hw-buildbox-net` (gloo/CPU), `links.csv` (IB and Ethernet rows), figures.

#### Figures and tables

- `tab_lm_hw_nics`, `tab_lm_hw_fabrics`, `tab_lm_hw_netcounts` (D4 counts); `fig_lm_hw_fattree`, `fig_lm_hw_ibsweep`, `fig_lm_hw_latbudget`, `fig_lm_hw_nccl_multinode`, `fig_lm_hw_hier`, `fig_lm_hw_a2a_nodes`, `fig_lm_hw_crossover`; equations `eq_lm_hw_ring_tree`, `eq_lm_hw_hier`, `eq_lm_hw_a2a_nodes`, `eq_lm_hw_fattree`.

#### Pitfalls and misconceptions

- `NCCL_TESTS_SPLIT` runs groups in parallel; the printed bandwidth is per group (README), so aggregate by multiplying.
- A single-NIC restriction is also a common accidental misconfiguration (wrong `NCCL_IB_HCA`, a missing rail, a NIC down); check the `NCCL_DEBUG=INFO` header before trusting a slow result.
- Bus bandwidth across nodes can exceed the NIC rate for all-to-all (intra-node traffic) and for in-network reduction; it is not a link measurement (5.4 D3).
- RoCE needs the right GID index and a lossless or congestion-controlled fabric configuration; `ib_write_bw` failures are often configuration, not hardware.
- Latency measured by `ib_write_lat` is not the latency of an NCCL step; NCCL adds kernel and protocol overhead (fit `α` from NCCL itself for predictions).
- Oversubscription is invisible at small scale and bites only when jobs span leaf switches; ask the scheduler where the nodes are.
- Results depend on neighbours' traffic on a shared fabric.

#### Summary (draft bullets)

- Between servers the bandwidth per GPU falls by about an order of magnitude and the latency rises to microseconds; one NIC per GPU, arranged in rails, is the standard design for scale-out.
- A message's latency is software, NIC, switch hops, cable and serialization; the hop and software terms set `α`, and `α·β` sets the message size at which a link becomes efficient.
- Hierarchical all-reduce uses the fast domain to reduce the data before it crosses the network, and gains roughly the number of rails per node over a single-NIC flat ring.
- Ring versus tree is a latency-versus-bandwidth trade that depends on the number of ranks and the protocol; in-network reduction lowers the bandwidth term to one pass.
- All-to-all across nodes is limited by the NIC rate once the job spans many nodes, which bounds wide expert parallelism.

#### Exercises

1. **(check)** Why can `busbw` of an all-to-all across 2 nodes exceed the NIC bandwidth? *Sketch:* by D3(ii) `(G−1)/(G−g) > 1` because the `g/G` fraction of each GPU's data goes over NVLink.
2. **(check)** A job spans two leaf switches of a 2:1 oversubscribed fabric. What happens to cross-leaf all-reduce and all-to-all bandwidth relative to within-leaf? *Sketch:* a ring crosses a leaf boundary only twice per step, so it needs few uplinks and is barely affected; uniform all-to-all sees `β/o = β/2` across leaves, i.e. half (D4 step 5).
3. **(derive)** Extend D1 to `m` independent rings (channels) over `m` NICs, each carrying `S/m`. Show the bandwidth term falls by `m` while the latency term is unchanged, and give the new `S*`. *Sketch:* `T_ring(m) = 2(G−1)α + 2((G−1)/G)·S/(mβ)`; `S*` for tree with the same `m` is `m·G α β (G−1−d)`.
4. **(derive)** Derive the all-reduce time of a three-level hierarchy (GPU, node, rack) with bandwidths `β_1 ≫ β_2 ≫ β_3`. Which level dominates for large `S`? *Sketch:* recursively reduce-scatter down the levels; time is `Σ_ℓ 2((n_ℓ−1)/n_ℓ)·S/(Π_{j<ℓ} n_j · β_ℓ)`; the top level moves `S/Π_{j<L} n_j` bytes on the slowest link.
5. **(measure)** Measure `ib_write_lat` for two node pairs under the same leaf and under different leaves (or, on the build box, loopback TCP latency with `qperf` or a Python echo) and report the per-hop increment. What does it imply for `α_out`?
6. **(measure)** Using nccl-tests on 2 nodes, measure the bus bandwidth of all-reduce at 256 MiB with 1, 2, 4 and 8 channels (`NCCL_MAX_NCHANNELS`) and compare with `min(m·β_ch, β_NIC·rails)`. State your noise check.
7. **(open)** Design the network for a 512-GPU cluster used for training with expert parallelism of width 64 and for serving a 700B-parameter mixture-of-experts model: choose radix, tiers, rails, oversubscription and justify with D1 to D4 and with one measured number from your own cluster.

#### Further reading

- Papers: Meta, "RDMA over Ethernet for Distributed AI Training at Meta Scale" (SIGCOMM 2024, cited by title); Alibaba, "HPN: A Data Center Network for Large Language Model Training" (SIGCOMM 2024, cited by title as in v5); [Rail-only](https://arxiv.org/abs/2307.12169); [Demystifying NCCL](https://arxiv.org/abs/2507.04786); [Collective Communication for 100k+ GPUs](https://arxiv.org/abs/2510.20171) (new; verified in 5.4); [Fire-Flyer AI-HPC](https://arxiv.org/abs/2408.14158) (network design with PCIe A100 nodes).
- Documentation and standards: [SHARP](https://docs.nvidia.com/networking/display/sharpv300) (v5 source); [Ultra Ethernet Consortium](https://ultraethernet.org/); [nccl-tests](https://github.com/NVIDIA/nccl-tests) and its README (verified; `NCCL_TESTS_SPLIT`); [NCCL environment variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) and [collective operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) (new; both verified); [NVIDIA InfiniBand](https://www.nvidia.com/en-us/networking/products/infiniband/), [Quantum-2](https://www.nvidia.com/en-us/networking/quantum2/), [Spectrum Ethernet](https://www.nvidia.com/en-us/networking/products/ethernet/) and [ConnectX adapters](https://www.nvidia.com/en-us/networking/ethernet-adapters/) (new; all four verified; link rates); [DGX SuperPOD](https://www.nvidia.com/en-us/data-center/dgx-superpod/) (new; verified; a reference rail-optimized design); NVIDIA blog [Doubling all2all performance with NCCL 2.12](https://developer.nvidia.com/blog/doubling-all2all-performance-with-nvidia-collective-communication-library-2-12/) (new; verified; PXN).
- Teaching and code: [Scaling Book](https://jax-ml.github.io/scaling-book/) (cluster networking); [Scaling Book: sharding](https://jax-ml.github.io/scaling-book/sharding/) (Codex 4.7); [NCCL user guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/); [perftest](https://github.com/linux-rdma/perftest) (new; verified; `ib_write_bw`, `ib_write_lat`).

#### Risks and open decisions

- *No multi-node allocation is available for the release.* Fallback: the notebook runs entirely from committed logs from the last allocation; the T0 simulator and the T1 gloo/CPU loopback runs keep every cell executable; the release notes state the log date.
- *The fabric is RoCE on a shared Ethernet network with other tenants.* Fallback: record the job ids and repeat three times at different times of day; widen the claim threshold to 5× MAD; label variance as fabric contention.
- *No SHARP or NVLS.* Fallback: the in-network series is omitted; D3(i) is stated as a bound and checked only in the simulator.
- *Fewer than 8 NICs per node (shared NICs).* Fallback: D2 uses `rails` equal to the actual NIC count from `ibstat`, and the single-NIC experiment becomes a comparison between the actual count and one.
- *`NCCL_NET` and loopback tricks differ across NCCL versions on the T2 pair.* Fallback: if forcing TCP between the two local GPUs fails, the T2-pair run is dropped and the gloo/CPU run (the only network-style run the build box does) carries the T1 role; the plots are labelled by transport.
- *The Alibaba paper's title is cited from v5 without re-verification (not on arXiv).* Kept as in v5.

### 5.6 Scale-up, scale-out and what it costs

- **Label:** `sec_lm_scale_up_out` · **Tier:** T0 (calculators and analysis of committed logs; one optional T2 clock-sweep log is produced once per release on the 5.1/5.2 A100 PCIe 40 GB host) · **Reduced T1 config:** not needed (the section runs on CPU); the cost cells read the build-box logs from 5.1 to 5.3 when the reader wants a local chip in the calculator · **Budget:** T0 ≈ 3 min on a laptop CPU; the clock sweep (optional, release only) ≈ 20 min on one A100 · **Codex units:** 4.8, 7.7 (cost of interruptions only: the goodput factor; reliability itself is 6.8)
- **Builds on (book):**
  - `sec_hardware` (`tab_gpu_specs`, the NVL72 and NVLink numbers of `subsec_hw-interconnects`, `subsec_hw-energy`, `eq_hw_decode_bound`): taken as given; the energy-per-token exercise of `sec_hardware` is *extended* with measured power.
  - `sec_multi_gpu` (`eq_dp_cost`, strong versus weak scaling) and `sec_scaling-laws` (`eq_six_nd`, the `6ND` training cost): taken as given.
  - `sec_cloud_instances` (`tab_cloud_prices`, a dated July 2026 rental snapshot; the exercises "GPU-hours before dollars", "NVLink or PCIe" and "Restarts on spot capacity"): its snapshot seeds `data/hw/prices.csv` and its exercises are acknowledged. This section scales the same arithmetic to clusters, adds owned hardware, serving, utilization and power, and does not repeat the single-GPU examples.
  - `sec_hardware_buyers` (retiring): v5 assigns its datacenter half to this section. The appendix section has no datacenter content (it covers desktop and unified-memory buying), so there is nothing to port; the editor should confirm that nothing else is expected (see open questions).
- **Builds on (this part):** 5.1 (`d2l.chip_specs`, `d2l.derive_ratios`, power traces in the `BenchLog` records), 5.2 (`d2l.ServingModel` and the `lm-hw-a100-workloads` log: decode `T(u)`, `gpu_s_per_mtok`; `d2l.TrainStepModel`), 5.3 (`d2l.ckpt_stall`: the checkpoint term of goodput), 5.4 and 5.5 (the fitted link parameters and `d2l.hier_allreduce_time`, `d2l.a2a_inter_node_time`). **Imported from other chapters:** (i) `DATA_HUB['lm-mfu-table']`, committed by 6.7: a JSON mapping `config_id → {model, gpus, layout, tokens_per_s_per_gpu, mfu, source_log}`; one-line fallback if the artifact is not yet available: the reader supplies an MFU, or the cell uses `TrainStepModel.mfu` from 5.2's A100 measurements (single GPU, an upper bound for multi-GPU), labelled as such. (ii) `d2l.failure_rate(gpus)`, exported by 6.8 (failures per hour; default source `'llama3'`: 419 unexpected interruptions in a 54-day snapshot on 16,384 GPUs, which is 3.79 failures per 1,000 node-days or 1.97e-5 per GPU-hour, arithmetic in 6.8; `'rsc1'` (6.50 per 1,000 node-days) and `'rsc2'` (2.34) from Meta's RSC paper are the alternatives); a one-line fallback keeps a reader-set `lambda_per_gpu_hour`. (iii) `d2l.flops_per_token` and `d2l.mfu`, exported by 4.2 (arch may be a Hugging Face config); 5.x does not define its own under these names (5.2's `TrainStepModel.flops_per_token` and `.mfu` are methods that delegate to 4.2's functions).
- **Exports (`#@save`):**
  - `d2l.systems_table() -> pd.DataFrame` from `data/hw/systems.json`; one row per published system with `system, chips, domain_chips, scaleup_gbs_per_chip, scaleup_topology, hbm_gb_per_chip, domain_mem_tb, scaleout_gbs_per_chip, tdp_w_per_chip, rack_kw, source_url, retrieved`. Rows: 8-GPU A100 node in two variants (the NVSwitch SXM node: A100 datasheet, 600 GB/s NVLink per GPU as printed, bidirectional; and the PCIe node: the NVLink bridge of 600 GB/s joins the two cards of a pair, every other pair uses PCIe Gen4 at 64 GB/s, as printed in the 40GB PCIe column; the committed `topo.txt` of the T3 node decides which variant its measurements belong to), 8-GPU H100 and B200 HGX nodes, GB200 NVL72 rack, a TPU pod slice, Trainium2 UltraServer, MI355X 8-GPU node; UALink as a spec row with no shipping system. Contract: bidirectional aggregate versus per-direction is a column (`scaleup_dir`), and torus systems carry the number of ICI links per chip and the per-link rate instead of a single figure.
  - `d2l.axis_ratio(axis, cfg, link_ridge, **par) -> float`: the communication-to-compute ratio of D1 for `axis ∈ {'tp','ep','pp','dp'}`.
  - `d2l.layout_fit(cfg, system, layout, tokens_per_step, p_eff=None, eps=0.15) -> pd.DataFrame`: for a layout `(t, e, p, d)` returns per axis the chosen link, the volume per step, the exposed-time ratio, fit/no-fit, and per-chip memory.
  - `d2l.cost_per_mtok_train(price_gpu_h, flops_per_token, p_peak, mfu, goodput=1.0)`, `d2l.cost_per_mtok_serve(price_gpu_h, tps_per_gpu, utilization=1.0)`, `d2l.owned_vs_rented(capex, life_h, power_kw, pue, elec_per_kwh, ops_per_h, rent_per_h) -> dict` (hourly owned cost per *used* hour and break-even utilization), `d2l.train_goodput(ckpt_stall_s, interval_s, lam, restart_s, interval_opt=None)` (the first-order goodput of D4; the optimal interval is deferred to 6.8), `d2l.diloco_wan_bandwidth(n_params, bytes_per_param, h_steps, t_step_s)`, `d2l.hw_clock_sweep(clocks, workloads)` (needs permission to lock clocks).
  - Data: `DATA_HUB['lm-hw-systems']` (JSON with sources), `DATA_HUB['lm-hw-prices']` (CSV `item, value, unit, kind, source_url, retrieved` where `kind ∈ {quote, assumption}`), `DATA_HUB['lm-hw-a100-clocksweep']` (JSONL).
- **Models and data:** configs only, no weights: `Qwen/Qwen3-8B` (dense), `Qwen/Qwen3-30B-A3B` (MoE, 48 layers, 128 experts, top-8, `moe_intermediate_size` 768), `Qwen/Qwen3-235B-A22B` (MoE, 94 layers, hidden 4096, 128 experts, top-8, `moe_intermediate_size` 1536, 4 KV heads; Apache-2.0 per the model card, to be re-checked at authoring), all read with `AutoConfig`. Prices: the book's `tab_cloud_prices` rows and a fresh quote file (provider pages, with date and URL, fetched by the author at release); no price is hard-coded in code cells.
- **Packages:** `pandas`, `numpy`, `matplotlib`, `transformers` (configs only), `networkx` (optional, for torus drawings), `pynvml` (clock sweep only).

#### Narrative arc

1. **The question.** The cell asks: "Can this model be tensor-parallel across 8 GPUs and expert-parallel across 64?" and answers with a table. For Qwen3-30B-A3B on an eight-GPU NVLink node versus on eight such nodes joined by a network, `layout_fit` prints, for each axis, the communication-to-compute ratio on the link it would use (from the closed forms of D1, no measurement needed), then flags the axis that does not fit. The measured symptom is imported rather than created: the 5.5 all-to-all busbw across nodes (committed log) and 5.4's intra-node busbw set the two link ridges, and the table shows that the expert-parallel ratio changes by about the ratio of the two ridges (the cell prints the factor; to be measured from the logs).
2. **Two networks (subsection).** A scale-up domain is a set of chips with memory-class bandwidth between all of them (NVLink racks such as NVL72; TPU pods with torus links and optical circuit switches; Trainium UltraServers; UALink as the open standard); a scale-out network connects domains (5.5). `tab_lm_hw_domains` lists, from `systems_table()`: chips per domain, scale-up bandwidth per chip, aggregate, HBM per chip and per domain, scale-out bandwidth per chip; `fig_lm_hw_domains`: scatter of domain size (x, log) against scale-up bandwidth per chip (y, log), marker area ∝ domain memory, a dashed band for the scale-out rate per chip. Topologies: a crossbar through switches (every pair at full rate) versus a torus (each chip has a fixed number of links, bisection falls with size, collectives run per axis; the Scaling Book formulas for torus axes are used and cited; the notebook does not derive them).
3. **What a domain bounds (subsection).** Tensor-parallel degree `t ≤ domain size` and (for attention without KV replication) `t ≤ n_kv_heads` (Qwen3-8B: 8); expert-parallel width `e ≤ domain size` for a fast all-to-all (5.5 D3: beyond the domain the all-to-all is NIC-bound); memory per domain bounds the model served without crossing the slow network: `tab_lm_hw_fit` lists, for Qwen3-8B, Qwen3-30B-A3B and Qwen3-235B-A22B in bf16 and FP8, the weight bytes (computed from configs) and the KV bytes per 100k tokens (`k_KV × 1e5`), against each system's domain memory. Statements like "fits" are computed, not asserted.
4. **Which layouts fit (subsection).** D1's four closed forms and the `layout_fit` calculator, applied to three published systems chosen so that the answer changes: an eight-GPU node (run for both A100 variants: NVSwitch, and PCIe pairs, where tensor parallelism of degree 2 can use the bridge and larger groups cross PCIe), a 72-GPU NVL72 rack, and a TPU pod slice (the slice shape and per-link rate transcribed from the Google Cloud TPU documentation page for the chosen generation; the author picks one slice, e.g. 64 chips, and records it). Output: `tab_lm_hw_layouts` (system × model × layout × axis → ratio, fit) and `fig_lm_hw_axis_ratio`.
5. **Scale-across (subsection).** Training across datacenters: latency is physics (about 5 µs per km of fibre one way, computed), bandwidth is far below the intra-datacenter fabric, so per-step synchronous all-reduce is infeasible and low-communication methods (DiLoCo, Streaming DiLoCo) become necessary; D4 gives the WAN bandwidth they need.
6. **Power and cooling (subsection).** From `systems_table()`: `tdp_w_per_chip` and `rack_kw` (vendor pages, transcribed at authoring); chips per rack × TDP × overhead versus the rack figure; liquid cooling as stated by the vendor; power capping and clocks: the DVFS model (D3) and, if `lm-hw-a100-clocksweep` is committed, its measured throughput and power against SM clock for a compute-bound GEMM and for decode (bandwidth-bound) steps; thermal throttling is read from `nvidia-smi -q -d PERFORMANCE` records in the logs.
7. **What a token costs (subsection).** D2 (training and serving cost), D4's goodput factor and the owned-versus-rented break-even (D5); `tab_lm_hw_cost` and `fig_lm_hw_cost_tornado`. The cell converts the measured MFU (6.7 artifact or fallback) and the measured decode throughput (5.2 artifact) into dollars per million tokens for rented hardware (rows from `prices.csv`) and for owned hardware under an *explicit* set of assumptions marked as such, then perturbs every input by ±30% (tornado).
8. **Closing (subsection).** A paragraph linking to Chapter 6 (layouts are chosen, not just checked) and to 13.4 (disaggregated pools use two systems).

#### Derivations

**D1. Which parallel axes fit on which link: communication-to-compute ratios.**

*Statement.* Let `R_ℓ = P_eff/β_ℓ` be the ridge of the link an axis uses (FLOP per byte; `β_ℓ` per direction), `b` bytes per activation element, `d` the hidden size, `N_layer` the matmul parameters of one transformer layer, `d_ff` the width of one expert's hidden layer, `K` the experts per token, `N_stage` the parameters of a pipeline stage, `T` the tokens a rank processes per step, `b_g` bytes per gradient element. Ignoring overlap and attention FLOPs, the ratio of communication time to compute time is, for each axis:
 - tensor parallel (degree `t`, Megatron-style, two all-reduces forward and two backward per layer): `ρ_tp = (4/3)(t−1)·(d·b/N_layer)·R_ℓ`;
 - expert parallel (width `e`, balanced routing, two all-to-alls forward and two backward per MoE layer): `ρ_ep = (2b(1−1/e)/(9·d_ff))·R_ℓ`;
 - pipeline parallel (point-to-point of boundary activations forward and backward): `ρ_pp = (d·b/(3N_stage))·R_ℓ`;
 - data parallel (gradient all-reduce): `ρ_dp = (b_g/(3T))·R_ℓ·(k−1)/k`.
An axis *fits* on a link if `ρ·(1 − overlap) ≤ ε` for a chosen `ε` (default 0.15); the calculator uses overlap 0 for TP, 0.5 for EP as a parameter, and 0.9 for DP and PP; these are parameters, not facts, and the notebook plots the dependence.

*Assumptions.* Dense layers have `N_layer` matmul parameters and `6·N_layer·T` forward-and-backward FLOPs per `T` tokens (attention score FLOPs ignored: stated, and bounded in the check); ring collectives (`2(k−1)/k` factor for all-reduce); all-to-all moves the fraction `(1 − 1/e)` of the routed bytes off the rank; compute at `P_eff` per chip; link bandwidth per direction; first-order volumes, which 6.3 to 6.7 refine (sequence parallelism, overlap, hierarchical collectives).

*Steps (tensor parallel; the others follow the same pattern).* (1) Per layer and per rank, compute time `t_c = 6·N_layer·T/(t·P_eff)`. (2) Communication: four all-reduces (two forward, two backward) of `T·d·b` bytes each; ring all-reduce moves `2(t−1)/t` of the payload per rank: bytes per layer `= 4·2((t−1)/t)·T·d·b`; time `t_x = 8((t−1)/t)·T·d·b/β_ℓ`. (3) Divide: `ρ = t_x/t_c = (8(t−1)/t)·T·d·b·t·P_eff/(6·N_layer·T·β_ℓ) = (4/3)(t−1)(d·b/N_layer)R_ℓ`; `T` cancels. (4) For a dense layer with `N_layer ≈ c·d²` the ratio scales as `1/d`: wider models tolerate slower links for tensor parallelism. *Expert parallel:* (5) expert compute per token (forward and backward) `18·K·d·d_ff` FLOPs, balanced across the `e` ranks so per-rank time `18·K·d·d_ff·T/P_eff` for `T` tokens per rank; (6) four all-to-alls move `T·K·d·b·(1−1/e)` bytes per rank each; (7) divide: `ρ_ep = 4·T·K·d·b(1−1/e)/β ÷ (18·K·d·d_ff·T/P_eff) = (2b(1−1/e)/(9·d_ff))R_ℓ` (`K` and `d` cancel): fine-grained experts (small `d_ff`) are the most link-hungry. *Pipeline:* (8) boundary bytes `2·T·d·b` per step (forward and backward) against `6·N_stage·T/P_eff`. *Data parallel:* (9) `2((k−1)/k)·N·b_g/β` against `6·N·T/P_eff`, `N` cancels (the `sec_hardware` exercise "Ridge point of a link" derives the threshold `T ∝ R_ℓ`; recapped).

*Numerical check (cell).* (a) A tracing communicator: a four-line PyTorch module implementing a Megatron-style TP MLP and attention shard on the `meta` device with a dummy `all_reduce` that records `numel × element_size`; one layer forward and backward records exactly four payloads of `T·d·b` bytes (assert), and the calculator's volume equals `2(t−1)/t` times that (exact). (b) FLOPs: `FlopCounterMode` on one layer of Qwen3-8B (`N_layer` from the config, 192.9M parameters by the cell's own count) equals `6·N_layer·T` plus the attention term to within 3% at `T=4096` for `s=2048`; the attention share is printed, bounding the "ignore attention" assumption. (c) The expert-parallel formula against the same counting for the MoE layer of Qwen3-30B-A3B. (d) Plug measured link ridges: `R_ℓ` from the 5.4 NVLink fit and the 5.5 NIC fit; print `ρ_tp, ρ_ep, ρ_pp, ρ_dp` for Qwen3-8B and Qwen3-30B-A3B on both links; the cell asserts only `ρ_pp < ρ_tp`, which the formulas imply for these configs (the test is on computed numbers, not hard-coded).

**D2. Cost per million tokens, for training and for serving.**

*Statement.* With `c` the price of a chip-hour (`c_s = c/3600` per second), training cost per million tokens is `C_train = 1e6·c_s·F_tok/(MFU·g·P)` where `F_tok` is FLOPs per token (`6N` plus attention), `P` the peak FLOP/s at the training precision and `g ≤ 1` the goodput (D4). Serving cost per million output tokens is `C_serve = 1e6·c_s/(T_gpu·u)` where `T_gpu` is tokens per second per chip at the operating point on the frontier of 5.2 D3 and `u ≤ 1` the fraction of time the chip is busy at that point.

*Steps.* (1) A chip-second of rented time costs `c_s`. (2) Tokens per chip-second in training `= MFU·P/F_tok`; with goodput `g` the productive rate is `g` times that. (3) Cost per token `= c_s/(tokens per productive chip-second)`; multiply by `1e6`. (4) Total training cost for `D` tokens `= C_train·D/1e6 = c_s·F_tok·D/(MFU·g·P)`, which with `F_tok = 6N` is the book's `6ND` in dollars (`eq_six_nd`; the exercise "GPU-hours before dollars" is the case `g = 1`, `MFU = 0.4`). (5) Serving: `T_gpu` from the frontier at the chosen per-user rate `u_user`; (6) prefill is billed separately: `C_prefill = 1e6·c_s·t_prefill(n)/(n·u)`, and a KV hit rate `h` lowers it to `(1−h)·C_prefill + h·C_load` with `C_load` from 5.3 D2. (7) Utilization `u` enters as `1/u`: an idle chip bills the same.

*Numerical check.* Units: a dimensional test (pint-style or manual) that `C_train` and `C_serve` are currency per million tokens; limit cases: `g = u = 1` equals the closed forms with `FLOP`-counted inputs; the cell recomputes the book's exercise ("GPU-hours before dollars", `N=1e9`, `D=1e12`, `MFU=0.4`, H100 dense BF16 from `tab_gpu_specs`) and asserts it equals `6·N·D/(MFU·P·3600)` GPU-hours (exact), then reports dollars with the quoted prices.

**D3. Power: DVFS, efficiency and the cost of energy.**

*Statement.* If compute-bound throughput is proportional to the SM clock `f` and chip power is `P(f) = P_s + k·f³` (static plus dynamic with voltage tracking frequency), throughput per watt `f/(P_s + k f³)` has its maximum at `f* = (P_s/(2k))^{1/3}`; a bandwidth-bound step (decode at small batch) has throughput independent of `f`, so its throughput per watt is maximized by the lowest clock that does not slow the memory pipeline.

*Steps.* (1) Dynamic power `∝ C V² f`; if `V ∝ f` over the DVFS range then `P_dyn = k f³`. (2) Efficiency `η(f) = f/(P_s + k f³)`. (3) `dη/df = 0` gives `P_s + k f³ = 3k f³`, i.e. `f*³ = P_s/(2k)`. (4) At `f*`, `P = P_s + P_s/2 = 1.5 P_s`. (5) Bandwidth-bound: time `= bytes/β` independent of `f`, power falls with `f`, so energy per token falls with `f` until the clock becomes the bottleneck of the memory pipeline. (6) Energy per token `= P/T`; cost `= (energy in kWh) × price per kWh × PUE`, with `PUE` the datacentre power overhead factor.

*Numerical check.* Fit `(P_s, k)` to the committed clock-sweep log by least squares on `(f, power)` of the GEMM series; report the fit residual; assert that the predicted `f*` lies within the measured range or print that it does not (then the DVFS model is extrapolating and the text says so); verify the algebra symbolically with `sympy` (exact). If the log is absent, the cell runs on a synthetic set with known `(P_s, k)` and says so.

**D4. Goodput and the scale-across synchronization bandwidth.**

*Statement.* (i) With a checkpoint every `T_c` seconds costing a stall `t_s` (5.3 D4), failures at rate `λ` per second for the whole job (rate per chip times chip count) and a restart cost `t_r`, the first-order goodput is `g ≈ (1 − t_s/T_c)·(1 − λ·(t_r + T_c/2))`; the interval minimizing the loss is deferred to 6.8 (Young/Daly). (ii) For a data-parallel group of `n_dc` datacentres joined by WAN links of bandwidth `β_wan`, synchronizing pseudo-gradients every `H` steps of duration `t_step` costs `N·b_c/β_wan` per synchronization (`b_c` bytes per parameter on the wire); the synchronization is hidden iff `β_wan ≥ N·b_c/(H·t_step)` (overlap of the outer step with `H` inner steps, as in Streaming DiLoCo).

*Steps.* (i-1) Fraction lost to stalls `t_s/T_c`. (i-2) A failure at a uniformly random time loses on average `T_c/2` of work plus `t_r`; expected loss per unit time `λ(t_r + T_c/2)`. (i-3) Multiply the survival fractions (first order). (ii-1) Bytes on the wire per synchronization `N·b_c`. (ii-2) Time `N·b_c/β_wan`. (ii-3) Available window `H·t_step`. (ii-4) Hidden iff ratio ≤ 1. (ii-5) One-way latency `ℓ = d_km·5 µs/km` matters only for the first byte: negligible against the window when `H·t_step ≫ ℓ`, decisive for per-step synchronization (`H=1`, `t_step` of seconds, `ℓ` of milliseconds is small but the bandwidth term is not).

*Numerical check.* Closed-form versus a Monte Carlo of failures (`λ`, `T_c`, `t_r`, `t_s` drawn once; 10^5 simulated runs, seed 0): goodput within 1% at small `λT_c`; the WAN formula is checked by a unit test and by evaluating it for the 8B model at `H ∈ {1, 100, 500}` (cell prints the required GB/s; the text compares with `β_NIC` of 5.5 and with typical WAN rates from provider pages, which the author supplies with sources).

**D5. Owned versus rented.**

*Statement.* Owning costs a fixed hourly amount `h_own = capex/L_h + opex_h` per chip (where `L_h` is the amortization life in hours and `opex_h` includes electricity `P_kW·PUE·e`, space, staff and network share), billed whether or not the chip works. Renting at price `r` per chip-hour is billed only for used hours (on-demand). The cost per *productive* hour owned is `h_own/u`, and owning is cheaper iff `u > u* = h_own/r`.

*Steps.* (1) Owned cost per hour `h_own`. (2) Productive hours per hour `u`. (3) Cost per productive hour `h_own/u`. (4) Compare with `r`. (5) For reserved rentals billed continuously the comparison is `h_own` versus `r_res` with no `u`, and the benefit of ownership is the ratio `r_res/h_own`. (6) Sensitivity: `∂u*/∂capex`, `∂u*/∂r` and the elasticities (the cell prints them).

*Numerical check.* Unit tests on the algebra; a tornado over ±30% of each input in `prices.csv` (assumption rows are flagged and shown in a different colour); the output `u*` is reported with its interval over the sampled assumptions (uniform ±30%, 10,000 draws, seed 0).

#### From-scratch implementation

Components (≈ 150 lines): `axis_ratio` and `layout_fit` (≈ 55 lines), cost functions and `owned_vs_rented` (≈ 25), `goodput` with Monte Carlo test (≈ 20), DVFS fit (≈ 15), `diloco_wan_bandwidth` (≈ 5), table builders (≈ 30), `hw_clock_sweep` (≈ 20, optional).

Numerical tests: tracing-communicator volumes (exact); FLOPs against `FlopCounterMode` (3%); the book exercise reproduced; goodput Monte Carlo (1%); tornado inputs have `kind` flags; dimensional analysis.

#### In production

- Public references for real cluster designs and costs: Llama 3's cluster description, MegaScale's, DeepSeek-V3's hardware co-design paper, NVIDIA's NVL72 pages; the from-scratch calculators are checked against their *stated* parallel layouts: the cell reproduces which axis each of them placed on which link and flags disagreements, which are reported, not tuned away.
- Cloud price pages and billing exports (instance price, committed-use discounts) for rented rates; the notebook never fetches prices at build time.
- `nvidia-smi -q -d CLOCK,POWER,PERFORMANCE`, `nvidia-smi -pl` and `-lgc` for power capping and clock locking; DCGM exporters in production clusters (named only).

#### Headline experiment

*Hypothesis.* (a) The communication-to-compute ratios of D1, evaluated with link ridges measured in 5.4 and 5.5, reproduce the layout choices of published large-scale systems (tensor parallelism inside the NVLink domain, expert parallelism as wide as the domain allows, data and pipeline parallelism across the network). (b) For serving a mixture-of-experts model, cost per million output tokens is dominated by the utilization `u` and by the frontier point chosen (5.2), more than by the rental price difference between providers of the same chip.

*Setup.* Pure calculation on committed artifacts: `systems.json`, `prices.csv`, `lm-hw-a100-workloads`, the link fits from `lm-hw-node-p2p` and `lm-hw-cluster-nccl`, and (if available) `lm-mfu-table` and `lm-hw-a100-clocksweep`. Models: Qwen3-8B, Qwen3-30B-A3B, Qwen3-235B-A22B configs. Systems: an 8-GPU A100 node (NVSwitch row and PCIe-pairs row, both computed; `topo.txt` decides which is the measured node), an NVL72 rack, a TPU pod slice. No randomness except the sensitivity draws (seed 0).

*Procedure.* (1) `layout_fit` for each model × system × a grid of layouts `(t, e, p)` with `t ∈ {1,2,4,8}`, `e ∈ {1,2,4,8,16,32,64}`, `p ∈ {1,2,4}`, data parallel filling the rest. (2) Count the layouts that fit on one domain and those that need the network for one axis. (3) Cost per million tokens for training (MFU from 6.7, or fallback) and for serving (the frontier point of 5.2 at a chosen interactivity) on rented chips (quotes) and owned chips (assumptions), with `u` swept from 0.1 to 1. (4) Tornado and a plot of cost against utilization.

*Metrics.* Per layout: `ρ` per axis, fit flags, memory per chip; per cost scenario: dollars per million tokens, break-even utilization `u*`, rank of the sensitivity of each input.

*Plot specification.* `fig_lm_hw_axis_ratio`: x = link ridge `R_ℓ` (log), y = communication-to-compute ratio `ρ` (log), one line per axis for the chosen model, horizontal dashed at `ε`, vertical markers at the measured ridges of NVLink, PCIe and the NIC from the logs. `fig_lm_hw_domains`: as above. `fig_lm_hw_cost_util`: x = utilization `u` (0.1 to 1, linear), y = dollars per million tokens (log), lines for rented (flat in `u` for on-demand) and owned (decreasing as `1/u`), the intersection marked `u*`. `fig_lm_hw_cost_tornado`: horizontal bars of the change in `u*` or cost for ±30% of each input, sorted, assumption inputs hatched. `fig_lm_hw_dvfs`: x = SM clock, left axis throughput and right axis power, measured points and the fitted model, `f*` marked.

*Expected outcome.* Direction only. The formulas predict that expert parallelism fits only on the fast link for fine-grained experts (smaller `d_ff` is more demanding), tensor parallelism fits on NVLink and not on PCIe, and pipeline and data parallelism fit on the slow link. Dollar values are to be computed from the committed quotes and assumptions and are not stated here; any statement in the chapter text about which is cheaper must come with its `u*` and its sensitivity.

*Noise check.* The inputs are measurements with their intervals; the cell propagates them by Monte Carlo (10,000 draws, seed 0) into `ρ` and into the cost per million tokens and reports 90% intervals; a claim that one layout fits and another does not requires the interval of `ρ` to lie on one side of `ε`.

*T1 reduction.* None needed (T0). A reader with only the build box can enter its measured PCIe link ridge (the host-to-device fit in 5.4's T1 log; the box has no GPU-to-GPU path to measure) and see `layout_fit` reject tensor parallelism across PCIe, which reproduces the book's lesson in the new vocabulary.

*Artifacts committed (T2+).* `lm-hw-systems`, `lm-hw-prices` (with dates and URLs), `lm-hw-a100-clocksweep` (one release-time log, optional), figures as SVG.

#### Figures and tables

- `tab_lm_hw_domains`, `tab_lm_hw_fit` (model bytes versus domain memory), `tab_lm_hw_layouts`, `tab_lm_hw_cost`, `tab_lm_hw_power` (rack kW, chip TDP, kW per rack from sources); `fig_lm_hw_domains`, `fig_lm_hw_axis_ratio`, `fig_lm_hw_cost_util`, `fig_lm_hw_cost_tornado`, `fig_lm_hw_dvfs`, `fig_lm_hw_wan` (required WAN bandwidth against `H`); equations `eq_lm_hw_rho_tp`, `eq_lm_hw_rho_ep`, `eq_lm_hw_rho_pp`, `eq_lm_hw_rho_dp`, `eq_lm_hw_cost_train`, `eq_lm_hw_cost_serve`, `eq_lm_hw_breakeven`, `eq_lm_hw_goodput`, `eq_lm_hw_dvfs`.

#### Pitfalls and misconceptions

- Vendor scale-up figures are bidirectional aggregates per chip; use per-direction numbers in `ρ`.
- A larger domain raises the bound on tensor and expert parallelism but not for free: the all-to-all latency and the radix of the switch grow, and utilization of the domain matters when the model is smaller than the domain.
- MFU depends on the denominator (peak or achievable) and on the layout; an MFU measured on one GPU is an upper bound for a cluster.
- Rented prices are dated quotes; owned costs are mostly assumptions. A conclusion that depends on one assumption is not a conclusion.
- Cost per token for serving is meaningless without the interactivity at which it was measured (5.2).
- Power capping is nearly free for bandwidth-bound decode and costs throughput for compute-bound training; applying one setting to both is the usual error.
- TCO omits what it does not model: failures, idle time, data egress, engineering time (`sec_cloud_instances`).

#### Summary (draft bullets)

- Scale-up domains give memory-class bandwidth between chips and bound tensor and expert parallelism; scale-out networks connect domains and carry data and pipeline traffic.
- For each parallel axis the ratio of communication to compute is the link's ridge point times a model-dependent constant: tensor parallelism favours wide models, expert parallelism is most demanding for fine-grained experts, pipeline and data parallelism are the most tolerant.
- A layout fits in a domain when the exposed communication ratio stays below a chosen fraction; the calculator makes the choice explicit and checks it against published systems.
- Power and cooling bound rack density; clocks trade throughput for energy differently for compute-bound and bandwidth-bound work.
- Cost per token is price over throughput, with MFU, goodput and utilization as the factors; owned hardware wins above a break-even utilization that depends on assumptions the reader must state.

#### Exercises

1. **(check)** Why does the data-parallel ratio `ρ_dp` not depend on the model size, while `ρ_tp` depends on the width? *Sketch:* gradient bytes and FLOPs both scale with `N`, so `N` cancels; TP volume scales with `d` per layer while FLOPs scale with `N_layer ∝ d²`.
2. **(check)** An NVL72 rack has 72 GPUs in one NVLink domain. What bounds the tensor-parallel degree of Qwen3-8B on it, and what bounds the expert-parallel width of Qwen3-235B-A22B? *Sketch:* TP by the 8 KV heads (or by replication cost), EP by the 128 experts per layer and by the domain size.
3. **(derive)** Derive `ρ_tp` for sequence-parallel tensor parallelism (reduce-scatter and all-gather instead of all-reduce) and show that the volume per rank is unchanged. *Sketch:* an all-reduce is a reduce-scatter plus an all-gather, each moving `(t−1)/t` of the payload; replacing it by the pair gives the same bytes.
4. **(derive)** For `H` inner steps and a compression ratio `ξ` on the wire (e.g. 4-bit versus 16-bit), find the minimum `H` such that synchronization is hidden on a WAN of bandwidth `β_wan`, and the largest model for which a given `β_wan` suffices. *Sketch:* `H ≥ N·b_c·ξ/(β_wan·t_step)`; `N ≤ β_wan·H·t_step/(b_c·ξ)`.
5. **(measure)** On any machine with an NVIDIA GPU and permission to lock clocks, sweep the SM clock and measure throughput and power for a large GEMM and for a batch-1 decode step of a small model; fit the DVFS model and report `f*` and the energy per token at `f*` versus at the maximum clock. State the noise check.
6. **(measure)** Using the committed 5.2 log, compute dollars per million output tokens at three interactivity levels (`u_user` = 20, 50, 100 tokens/s/user) for Qwen3-8B on an A100 at a price from `prices.csv`; report how many times more the highest interactivity costs than the lowest and bootstrap an interval over repeats.
7. **(open)** Pick a workload (training a 7B model on 2T tokens, or serving a 235B mixture-of-experts model at 10 million requests a day), decide between rented and owned hardware, and write a one-page decision memo that states every assumption with its source or marks it "assumption", gives `u*`, and names the single input whose uncertainty matters most.

#### Further reading

- Papers: [TPU v4](https://arxiv.org/abs/2304.01433) (torus, optical circuit switches, pod-scale design); [Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures](https://arxiv.org/abs/2505.09343) (hardware co-design, scale-up versus scale-out for expert parallelism; title as in v5, whose verification is kept); [Llama 3](https://arxiv.org/abs/2407.21783) (cluster design and MFU of its configurations); [MegaScale](https://arxiv.org/abs/2402.15627) (large-cluster training efficiency and fault handling); [The rising costs of training frontier AI models](https://arxiv.org/abs/2405.21015) (new; verified title; cost accounting: hardware amortization, energy, staff); [DiLoCo](https://arxiv.org/abs/2311.08105) and [Streaming DiLoCo](https://arxiv.org/abs/2501.18512) (new; verified titles; D4(ii)); [From Words to Watts](https://arxiv.org/abs/2310.03003) (new; verified title; measured inference energy) and [Reducing Energy Bloat in Large Model Training](https://arxiv.org/abs/2312.06902) (new; verified title; GPU clocks and training energy; D3).
- Vendor material: NVIDIA, [GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) (verified) and the blog [NVL72 for training and real-time inference](https://developer.nvidia.com/blog/nvidia-gb200-nvl72-delivers-trillion-parameter-llm-training-and-real-time-inference/); [GB300 NVL72](https://www.nvidia.com/en-us/data-center/gb300-nvl72/) and [Vera Rubin NVL72](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/) (new; both verified); [UALink Consortium](https://ualinkconsortium.org/); [Google Cloud TPU documentation: TPU7x](https://docs.cloud.google.com/tpu/docs/tpu7x) and [TPU architecture](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm) (new; both verified).
- Teaching and reference: [Scaling Book](https://jax-ml.github.io/scaling-book/) (TPU and GPU chapters: torus collectives, rooflines per axis); [The Datacenter as a Computer](https://link.springer.com/book/10.1007/978-3-031-01761-2) (power, cooling and TCO chapters); [Efficiency in LLMs (MLSS)](https://alex.smola.org/posts/45-mlss-efficiency/) and [MLCommons inference benchmarks](https://mlcommons.org/benchmarks/inference-datacenter/) (from Codex 4.8). The Codex 7.7 sources ([Llama 3 report](https://arxiv.org/abs/2407.21783), PyTorch DCP, elastic launcher, TorchTitan) are cited only for the goodput factor and are taught in 6.8.
- Code: the section's own `layout_fit` (a miniature of the planners in [TorchTitan](https://github.com/pytorch/torchtitan), named for orientation); NVIDIA DCGM documentation for production power telemetry ([DCGM user guide](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/index.html), new; verified) and the [NVML API reference](https://docs.nvidia.com/deploy/nvml-api/latest/index.html) (new; verified; the power and clock calls used by `hw_clock_sweep` and by `BenchLog` power traces).

#### Risks and open decisions

- *Vendor power and rack figures differ across pages or change with SKUs.* Fallback: the cell prints which page each number came from and its retrieval date; the tornado treats rack kW as an input with ±30%.
- *No MFU artifact from 6.7.* Fallback: the reader-supplied MFU, or the 5.2 single-GPU measurement as an upper bound, labelled; the headline experiment's claim (a) does not depend on it.
- *Rental prices move.* Fallback: prices are dated quotes in a CSV, never embedded in code; the notebook prints the age of each price and warns above 90 days.
- *No permission to lock clocks on the A100 host.* Fallback: the DVFS part runs on the synthetic set and on any committed log from a previous allocation; the text labels the missing measurement.
- *A TPU slice's per-link rate is not in the docs page for the chosen generation.* Fallback: choose another generation whose page states it; if none does, mark the row "not stated" and omit it from `layout_fit`.
- *`sec_hardware_buyers` had datacenter content that I could not find.* Open question for the editor (the appendix section covers desktop and unified-memory buying only).
