# Chapter 6: Parallelism Beyond One Node — Blueprint

**Label:** `chap_lm_parallelism` · **Arc:** III. Scaling out · **Sections:** 6.1–6.8 (`sec_lm_multinode_jobs`, `sec_lm_fsdp`, `sec_lm_tensor_parallel`, `sec_lm_pipeline_parallel`, `sec_lm_context_parallel`, `sec_lm_expert_parallel`, `sec_lm_composing_parallelism`, `sec_lm_long_runs`) · **Tiers:** T3 (6.2–6.6, 6.8), T4 (6.1, 6.7); every section has a T0/T1 configuration · **Assignment:** A6

## Chapter overview

### Draft introduction (for the chapter's index page)

Advanced stops at one node. `sec_multi_gpu` builds data parallelism by hand and derives ring all-reduce. `sec_multi_gpu_concise` runs DDP and sketches FSDP, but nothing is sharded across machines and nothing runs at a scale where memory, bandwidth, or failures decide the design. This chapter executes the rest. It starts with the unglamorous part, launching one job across machines and reading its logs when it hangs (6.1). It then adds one form of parallelism per section: sharded data parallelism (6.2), tensor and sequence parallelism (6.3), pipeline parallelism (6.4), context parallelism (6.5) and expert parallelism (6.6). Each section derives the bytes it moves per step and the memory it leaves per GPU, and then checks both numbers against a measurement. 6.7 composes the five axes on a device mesh, ranks the feasible configurations with a cost model, and compares its predictions with logged multi-node runs. 6.8 asks what keeps a month-long run alive: failure statistics, the Young–Daly checkpoint interval, distributed and asynchronous checkpoints, restart on a different number of GPUs, and straggler and silent-corruption detection. Chapter 5 supplies the hardware numbers; this chapter supplies the algorithms that use them.

### Learning objectives

After this chapter a reader can:

1. **Launch and diagnose.** Launch a script across one and two nodes with `d2l.launch`, read rank-tagged logs, and diagnose a hang caused by mismatched collective order from a flight-recorder dump (6.1).
2. **Predict memory.** Predict peak memory per GPU of an FSDP2, TP, PP, CP or EP run to within a stated band, itemized by term (parameters, gradients, optimizer, transient unsharded layers, activations, logits) (6.2–6.6).
3. **Derive and verify traffic.** Derive the bytes per step each form of parallelism moves per GPU, and verify the formula against counted collective bytes in the same run, exactly for from-scratch code and within a stated band for library code (6.2–6.6).
4. **Verify against one GPU.** Implement tensor-, pipeline-, context- and expert-parallel blocks from collectives, and verify outputs and every parameter gradient against a single-process reference (float64 on CPU, 1e-9 relative; BF16 on GPU, band stated per section).
5. **Place axes on topology.** State which mesh axis belongs on which link, from the ratio of its communication to its compute, and show the ratio on a log.
6. **Choose a configuration.** Enumerate feasible DP×TP×PP×CP×EP layouts for a 7–8B model on 2–4 nodes, rank them with a cost model, and compare predicted with measured step times (6.7).
7. **Keep a run alive.** Fit failure rates from published interruption tables, derive and numerically verify the Young–Daly interval and its higher-order correction, restart a killed run on a different world size with the loss trajectory and data cursor verified, and detect a straggler and a corrupted replica (6.8).

### Prerequisites

Book labels, and what is used from each:

| Label | Used for | Not re-taught here |
|---|---|---|
| `sec_multi_gpu` | the hand-built data-parallel step; ring all-reduce, `eq_ring_traffic` ($2(k-1)N/k$ per device); the $t_{\text{step}}$ cost model, `eq_dp_cost`; α–β exercise; two-stage pipeline exercise | star/ring derivation, α–β definition, weak vs strong scaling |
| `sec_multi_gpu_concise` | the `torchrun --standalone` sidecar-script idiom (`DDP_SCRIPT`, `ddp_throughput`); DDP bucketing and `no_sync`; the FSDP lifecycle (`fig_fsdp_lifecycle`), `tab_collectives`; ZeRO budget and "cost of sharding" exercises; TP-vs-DP MLP exercise | DDP, collectives table, ZeRO stage definitions, 16-bytes-per-parameter anatomy |
| `sec_memory_precision` | `subsec_mp-anatomy` (16P bytes), activation checkpointing, gradient accumulation, `d2l.Benchmark` | all of it |
| `sec_perf_model`, `sec_hardware` | roofline, correct GPU timing, profiler reading; bandwidth ladder; `subsec_hw-interconnects` (the build box) | |
| `sec_transformer-block`, `sec_gpt`, `sec_kv-cache` | `d2l.TransformerBlock` (`attn_factory`, `ffn_factory` seams), `d2l.FeedForward` (SwiGLU), `d2l.GPT`, `d2l.GQAAttention` (separate `W_q,W_k,W_v,W_o`) | block anatomy, GQA |
| `sec_attention-at-scale` | `eq_online-softmax`, `chunked_attention` | online softmax derivation |
| `sec_moe` | `MoELayer`, `eq_moe-layer` (raw-probability gate), capacity factor exercise, `moe_gpt` | routing, balancing |
| `sec_custom_layer` | `torch.autograd.Function` | |
| `sec_batch_size` | critical batch size, global batch bookkeeping | |
| `sec_training_systems` (retiring appendix; re-pointed here) | scaling ladder, library landscape, "Keeping a Long Run Alive" | absorbed by 6.1, 6.7, 6.8 |

Earlier sections of this part:

| Section | Used for |
|---|---|
| 1.1–1.2 | `d2l.load_hf`, `d2l.config_memory`, Qwen3 configs; `d2l.GPT` shaped like Qwen3 |
| 4.1 | `d2l.PackedStream` (resumable cursor, 6.8) |
| 4.2 | R2 for T1 reductions |
| 5.1 | `d2l.chip_specs()` (A100 peak BF16, HBM, NVLink rows) |
| 5.3 | measured checkpoint write bandwidth and stall (6.8) |
| 5.4 | measured intra-node α–β, all-reduce and all-to-all busbw (6.3, 6.5, 6.6) |
| 5.5 | measured inter-node α–β, hierarchical all-reduce (6.1, 6.3, 6.7) |
| 5.6 | scale-up domain size; the layout calculator (6.7 extends it) |

### Section dependency graph

```
6.1 launch, logs, meshes ──┬─► 6.2 FSDP2 ─────────────┐
                           ├─► 6.3 TP / SP ───────────┤
                           ├─► 6.4 PP (sim T0) ───────┼─► 6.7 composition (T4) ─► (7.8, 13.4, 14.2)
                           ├─► 6.5 CP ────────────────┤        ▲
                           └─► 6.6 EP ────────────────┘        │
6.2 ─────────────────────────────────────────────► 6.8 long runs (also uses 6.1, 5.3)
A6 = 6.3 (TP block) + 6.4 (PP runner + simulator); 6.5 and 6.6 are independent of each other.
```

Order within the chapter is fixed (6.1 first, 6.7 and 6.8 last); 6.3–6.6 can be taught in any order after 6.2.

### Running examples and shared artifacts

**Used.**
- **Workhorse shape.** `Qwen/Qwen3-8B`: d=4096, 36 layers, 32 heads, 8 KV heads, head_dim 128, FFN 12288, vocab 151936, untied head (HF `config.json`, fetched 2026-10-28). Used for derivations in 6.2–6.5 and 6.7. Random initialization; the experiments are systems measurements, so no pretrained weights are needed.
- **MoE shape.** `Qwen/Qwen3-30B-A3B`: d=2048, 48 layers, 128 experts, top-8, expert FFN width 768 (`config.json`). Used for derivations in 6.6 and to measure a realistic routing-load profile. A miniature of the same family is trained in 6.6.
- **Small blocks.** `d2l.TransformerBlock` with `d2l.GQAAttention`/`d2l.FeedForward` in float64 on CPU: the correctness oracle for 6.3–6.6 and for A6.
- **R2** (4.2) as the T1 model for FSDP and checkpoint tests.

**Produced (data artifacts via `d2l.DATA_HUB`; keys are placeholders to be registered by the build).**
- `lm-launch-<tag>`: one bundle per T3/T4 launch (see 6.1, "Committing outputs").
- No separate α–β artifact: 6.1 reads the `alpha_beta.json` summaries inside `lm-hw-node-p2p` (5.4) and `lm-hw-cluster-nccl` (5.5). If a link class is missing there, 6.1 measures and fits it locally and does not register a new key.
- `lm-qwen3-30b-a3b-expert-load` (6.6): per-layer expert token counts on a fixed 64k-token sample. 14.2 may reuse it.
- `lm-pp-calibration` (6.4): per-stage F, B_in, B_w times for the simulator.
- `lm-matmul-eff` (6.7): achieved TFLOP/s at the shapes a TP/PP layout produces.
- `lm-mfu-table` (6.7): JSON of measured MFU per configuration (nodes, dp/tp/pp/cp, sequence, convention string); consumed by Chapter 5.
- `lm-failure-tables` (6.8): the published tables transcribed to CSV (Llama 3 Table 5, OPT-175B counts, RSC-1/RSC-2 rates), each with a source line.
- `lm-ckpt-reshard` (6.8): logs of the kill-and-reshard run.

**Imports from Chapter 5 (not redefined here).** Collective models: `d2l.fit_alpha_beta` (5.3); `d2l.parse_nccl_tests`, `d2l.nccl_busbw_factor`, `d2l.coll_time` (5.4); `d2l.ring_allreduce_time`, `d2l.tree_allreduce_time`, `d2l.hier_allreduce_time`, `d2l.a2a_inter_node_time` (5.5); the first-order per-axis ratios `d2l.axis_ratio` (5.6, TP/EP/PP/DP; `:numref:sec_lm_scale_up_out`); the artifact `lm-hw-a100-microbench`. The per-step volumes derived in 6.2–6.6 reduce to 5.6's ratios in the limits stated in each section's ratio equation ($\rho_{\text{TP}}$ in 6.3 D3, $\rho_{\text{PP}}$ in 6.4 D6, $\rho_{\text{EP}}$ in 6.6 D1, $T^*$ in 6.2 D2), and each notebook asserts agreement with `d2l.axis_ratio` to rounding. Artifact keys use hyphens (`lm-...`).

**Cross-chapter commitments.** 6.7 commits `lm-mfu-table` (JSON, measured MFU per parallel configuration and node count, with convention) for Chapter 5. 6.8 exports `d2l.failure_rate(gpus)`. 6.2 exports `d2l.fsdp_wrap` and `d2l.fsdp_train_step`, imported by 7.8 (FSDP with LoRA) and 8.6. 6.6 exports `d2l.a2a_bytes`, used by Chapter 13.

**Names exported by this chapter** (all `#@save`, all prefixed `d2l.`; the owner section lists signatures):
6.1: `launch` (reserved), `LaunchResult`, `init_dist`, `rank_log`, `read_rank_logs`, `CommCounter`, `comm_events`, `parse_nccl_info`.
6.2: `ModelShape`, `fsdp_wrap`, `fsdp_train_step`, `FSDPLite`, `fsdp_costs`, `exposed_comm`, `saved_bytes`. (FLOPs per token and MFU are imported from 4.2: `d2l.flops_per_token`, `d2l.mfu`.)
6.3: `copy_to_tp`, `reduce_from_tp`, `ColumnParallelLinear`, `RowParallelLinear`, `TPBlock`, `tp_costs`.
6.4: `PipelineSim`, `pipeline_schedule`, `plot_gantt`, `PipelineRunner`, `pp_costs`, `balance_stages`.
6.5: `merge_partial_attention`, `ring_attention`, `zigzag_split`, `cp_costs`.
6.6: `EPMoE`, `dispatch_plan`, `sample_routing`, `a2a_bytes`, `ep_costs`.
6.7: `ParallelConfig`, `ParallelCostModel`, `enumerate_configs`, `make_mesh`.
6.8: `failure_rate`, `expected_waste`, `young_daly`, `simulate_run`, `CursorSampler`, `validate_checkpoint`, `StragglerDetector`, `replica_checksum`.

**Common conventions inside the chapter.**
- All `*_costs(shape, par, ...)` functions return `{'comm': {collective: bytes_per_gpu_per_step}, 'mem': {term: bytes_per_gpu}, 'flops': flops_per_gpu_per_step}`; 6.7 composes them.
- Notation: $k$ world size of a group, $t$ TP degree, $p$ PP stages, $m$ microbatches, $c$ CP degree, $e$ EP degree, $b$ microbatch sequences, $s$ sequence length, $d$ hidden size, $P$ parameters, $F$ peak FLOP/s, $\beta$ per-GPU bus bandwidth, $\eta$ achieved fraction of $F$. Bytes are per GPU per optimizer step unless stated. Sizes use decimal GB.
- Every section prints a "prediction vs measurement" table with columns `quantity`, `formula`, `predicted`, `measured`, `rel. error`.
- **Batch sizes.** The global batch of 512 sequences of 4,096 tokens (2,097,152 tokens per step) belongs to 6.7 only (2 nodes = 16 GPUs, optionally 4 nodes = 32 GPUs). The batches of 6.2–6.5, 6.6 and 6.8 are per-node settings (one 8-GPU node) chosen to exercise each section's quantity (tokens per GPU, microbatches, one long sequence): 65,536 tokens per step in 6.2, 6.3 (regime A) and 6.8, 16,384 to 65,536 in 6.4 (`m` of 4 to 16 sequences), one sequence of 16k to 128k tokens in 6.5, and 131,072 tokens (8 sequences per GPU) in 6.6. R2's step in 4.2 is 524,288 tokens (128 × 4,096, `ch04` 4.2 setup), so 6.7's batch is exactly 4× R2's; it is a design setting for a 7–8B model, in the range `sec_batch_size` gives for language models (batches of millions of tokens; Llama 3 moved from 4M to 16M), and not a claim about the critical batch size of any model run here.
- **Hardware.** T3 means one 8× A100 40 GB node. The node's interconnect is unknown until the committed `topo.txt` (`nvidia-smi topo -m`) is read: every prediction that uses an intra-node α–β takes the NVSwitch row of 5.4's `alpha_beta.json` (`link='nvlink'`) as the reference and, where the conclusion depends on it (A6 written analysis 2, 6.2's $T^*$, 6.3's TP-domain table, 6.5's overlap threshold, 6.6's all-to-all ratio, 6.7's cluster description), adds the PCIe-pair row (`link='pcie'`: the NVLink bridge joins two cards, everything else crosses PCIe Gen4), and `topo.txt` decides which row is reported as the measured node. T4 means 2 nodes of such GPUs as the core allocation; 4-node runs are optional.
- **Build box.** The build box (4× RTX 4090, no NVLink, unreliable peer-to-peer) runs no multi-GPU experiment. Every T1 reduction in this chapter uses CPU ranks with gloo (float64), a single process, or one GPU; anything that needs GPU-to-GPU traffic at reduced size runs on the T2 pair (two A100 40 GB joined by an NVLink bridge).
- **Software.** `torch==2.15.0` (the part's pin, `00_conventions.md`). The chapter's PyTorch API names (`fully_shard`, `parallelize_module`, `context_parallel`, `torch.distributed.pipelining`, `torch.distributed.checkpoint`, torchrun and flight-recorder variables) were checked against the 2.15 documentation, so documentation and pin now agree; re-check only if the pin moves.

### Compute plan (planned budgets; to be confirmed on the first release run)

| § | Tier | Reduced T1 configuration | Planned wall-clock at full tier | GPU-hours | Committed artifacts |
|---|---|---|---|---|---|
| 6.1 | T4 (2 nodes), T3 part | 2 "nodes" × 2 CPU ranks (gloo, loopback rendezvous) | ≤ 1 h on 2×8 A100 | ≤ 16 | rank logs, NCCL INFO log, flight-recorder dumps, sbatch script, `meta.json` |
| 6.2 | T3 | `FSDPLite` and `fully_shard` on 2 gloo CPU ranks, R2 block stack, float64 (the two-GPU `fully_shard` check runs on the T2 pair, not on the build box) | ≤ 1.5 h | ≤ 12 | per-config step logs, memory snapshots, profiler summaries |
| 6.3 | T3 | 2-way TP on CPU float64 | ≤ 1 h | ≤ 8 | TP sweep logs, saved-tensor byte counts |
| 6.4 | T3 (simulator T0) | 4 gloo ranks, sleep-calibrated stages; simulator needs nothing | ≤ 1 h | ≤ 8 | calibration JSON, per-rank timelines |
| 6.5 | T3 | 2–4 gloo ranks, sequence 512, float64 | ≤ 1.5 h | ≤ 12 | ring/Ulysses/all-gather sweep logs |
| 6.6 | T3 | 2–4 gloo ranks, 8 tiny experts, float64 | ≤ 1.5 h | ≤ 12 | skew sweep logs, expert-load profile |
| 6.7 | T4 (core: 2 nodes; optional: 4 nodes and one EP=16 MoE run) | cost model only (T0) plus the 2-rank gloo mesh checks | core collection ≈ 1 h on 2 nodes (6 two-node runs and 2 one-node runs of ~8 min); optional 4-node pass ≈ 0.5 h; notebook analyzes logs | core ≈ 15 (6 × 16 GPUs + 2 × 8 GPUs, ~8 min each); optional ≈ 17 (4 four-node runs) and ≈ 2 (EP=16) | per-config TorchTitan logs, `config` recipes, calibration |
| 6.8 | T3 | 2 gloo ranks killed, restart on 1 | ≤ 1.2 h | ≤ 10 | kill-and-reshard logs, straggler injection logs |

All T3/T4 notebooks are re-executed per release, not per build; the build replays committed artifacts (6.1). Memory plans are for 40 GB GPUs: where a section's own memory model predicts more than 0.9 × 40 GB, the section says which reduced shape runs instead, or records the predicted out-of-memory as a run.

## Assignment

### A6. Tensor- and pipeline-parallel blocks verified against one GPU

**Label:** `sec_lm_a6`

**Target.** Implement a tensor-parallel transformer block and a pipeline-parallel executor from collectives, match a single-process reference in outputs and gradients, and match a schedule simulator's bubble to within 10%.

**Starter code** (`assignments/a6/`, scaffolded from 6.3 and 6.4):
- `reference.py`: a `d2l.TransformerBlock` stack (GQA attention via `d2l.GQAAttention`, SwiGLU via `d2l.FeedForward`, RMSNorm, RoPE), `make_reference(seed, dtype)` returning the model, a fixed batch, and the single-process loss, parameter gradients and input gradient.
- `tp.py`: stubs `copy_to_tp`, `reduce_from_tp`, `ColumnParallelLinear`, `RowParallelLinear`, `TPBlock.from_reference(block, group, sequence_parallel=False)`.
- `pp.py`: stubs `PipelineRunner(stages, group, schedule)` with `run(batch, microbatches)` for `'gpipe'` and `'1f1b'`, and `split_layers(model, p)`.
- `sim.py`: the 6.4 `PipelineSim` with its schedule generators removed; students supply `gpipe` and `1f1b`.
- `harness.py`: launches the checks with `d2l.launch(..., mode='loopback', D2L_BACKEND='gloo')` and writes `a6_submission.json`.

**Tasks.**
1. **TP linear layers and the conjugate operators.** Implement `copy_to_tp` (identity forward, all-reduce backward) and `reduce_from_tp` (all-reduce forward, identity backward), then `ColumnParallelLinear` and `RowParallelLinear` with the bias handled once. Gradient-check both operators with `torch.autograd.gradcheck` in float64.
2. **TP block.** Shard the reference block's attention by heads (query heads split; KV heads split when `num_kv_heads % t == 0`, else replicated with the replicated gradients summed over the replica group) and its SwiGLU MLP (gate and up column-split identically, down row-split). Replicated parameters (RMSNorm weights) must end with identical gradients on every rank.
3. **Sequence parallelism** (required for the T3 grade, optional for T1): replace the all-reduces by reduce-scatter and all-gather over the sequence dimension and shard the norms.
4. **Pipeline executor.** Partition the layer stack into `p ∈ {2, 4}` stages, run GPipe and 1F1B with `batch_isend_irecv`, scale each microbatch loss so the accumulated gradient equals the full-batch gradient, and handle the last microbatch of unequal token count.
5. **Simulator.** Implement the `gpipe` and `1f1b` generators. Given per-stage F and B times, predict bubble fraction and peak live activations per stage.
6. **Bubble validation.** Run the executor with sleep-calibrated stage costs (T1) or real stages (T3), record per-rank timelines, and compare the measured bubble with the simulator.

**Autograder checks** (every check runs on 4 gloo ranks, float64, seeds 0–2; the first failing check names the quantity):

| Check | Tolerance |
|---|---|
| `gradcheck` of `copy_to_tp`, `reduce_from_tp` | PyTorch defaults (eps 1e-6, atol 1e-5) |
| TP block output vs reference, $t \in \{2, 4\}$ | $\max\lvert\Delta\rvert \le 10^{-9}\max(1,\max\lvert\text{ref}\rvert)$ |
| every TP parameter gradient (gathered to the reference layout) and input gradient | same |
| collective bytes counted by `d2l.CommCounter` for one forward+backward | equal to the formula of 6.3 exactly (integers) |
| replicated-parameter gradients identical across ranks | bitwise |
| pipeline loss and every parameter gradient vs full-batch reference, `gpipe` and `1f1b`, $p \in \{2, 4\}$, $m \in \{4, 8\}$ | $10^{-9}$ relative, as above |
| unequal last microbatch | same |
| simulator makespan and peak activation per stage vs the hidden reference traces | exact (integer unit costs) |
| measured bubble vs simulator, $p = 4$, $m \in \{4, 8, 16\}$, sleep-calibrated costs F = 20 ms, B = 40 ms | relative difference ≤ 10% (v5 target) for each $m$ |
| T3 only, BF16 on 8 GPUs | TP block output within $2\times 10^{-2}$ relative of an FP32 reference (BF16 round-off; band to be confirmed on the first run) |

**Written analysis** (about half a page each):
1. Why do the f and g operators appear in conjugate pairs, and what breaks if the MLP's second linear is column-split too? Use your measured byte counts.
2. For Qwen3-8B on an 8-GPU A100 node, evaluate the TP communication-to-compute ratio of 6.3 for $t = 2, 4, 8$ with Chapter 5's bandwidth (the NVSwitch row of `alpha_beta.json`; and the PCIe-pair row if `topo.txt` shows bridged PCIe cards) and say where TP should stop on each.
3. Your measured bubble versus the simulator: where is the largest discrepancy, and which assumption (equal stage times, zero latency, no contention) accounts for it?
4. The last pipeline stage holds the LM head. Compute the stage-time imbalance for 36 layers on 4 stages and propose a partition.
5. Choose $m$ and $p$ for global batch 512 sequences and 8 GPUs under activation memory of 40 GB per GPU; defend it with the simulator.

**T1 path.** Everything above except the T3 row runs on CPU with 2–4 gloo ranks in under 10 minutes total (float64, hidden size 64, 4 layers, sequence 32). **Estimated student time:** 12–16 hours (tasks 1–2: 4 h; 4: 4 h; 5–6: 3 h; analysis: 2 h).

## Teaching sources

All of v5's list is kept, with what to take from each.

- [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) (Hugging Face, nanotron): the measured, interactive memory and throughput heat-maps over 4,000+ configurations on up to 512 GPUs. Use its configuration sweep as the ground truth to compare 6.7's cost model against qualitatively; take the TP/PP/DP/CP/EP diagrams as the visual vocabulary.
- [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) (Austin et al.): the analytical treatment. Take its transformer-training and "How to Think About GPUs" parts for the roofline view of each collective; translate its JAX sharding notation into DTensor placements.
- [CS336](https://cs336.stanford.edu/) (systems assignment): the data-parallel and optimizer-sharding exercises, already adapted in `sec_multi_gpu_concise`. 6.2's from-scratch FSDP is the next step.
- [CMU 11-968 LLM Systems syllabus](https://www.cmu.edu/online/welcome/pdfs/11968-llm-systems-syllabus-f25.pdf): lecture order for parallelism and fault tolerance; compare with the section order here.
- [Megatron Core guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html): the reference for TP/SP/PP/CP/EP knobs and the distributed optimizer; 6.7's library table cites it.
- [TorchTitan](https://github.com/pytorch/torchtitan): the PyTorch-native composition used for every library-level run (6.2, 6.5–6.8). Its Python-recipe configuration (`MODULE=... CONFIG=...`) and its `docs/debugging.md` are read in 6.1 and 6.7.
- New: Stas Bekman's [Machine Learning Engineering Open Book](https://github.com/stas00/ml-engineering) (200 OK, checked 2026-10-28): the practical troubleshooting chapters for hangs, NCCL environment variables and SLURM, which 6.1 draws on and 5's teaching sources already list.

## Sections

### 6.1 Running jobs across machines

- **Label:** `sec_lm_multinode_jobs` · **Tier:** T4 (hang notebook and `d2l.launch` also run at T3) · reduced T1 config: `mode='loopback'`, 2 "nodes" × 2 CPU ranks with gloo; the NCCL-only cells (flight recorder, transport parsing) replay committed logs · **Budget:** T4 ≤ 1 h wall on 2×8 A100 (≈ 16 GPU-h, planned; includes the 1-node baseline and a deliberate hang bounded by a 60 s timeout); T1 ≈ 3 min on CPU; peak GPU memory ≈ 20 GB (R2-scale model, 16k tokens per GPU) · **Codex units:** 6.1 (processes and collectives; the safely bounded order mismatch)
- **Builds on (book):** `sec_multi_gpu_concise` (the `torchrun --standalone` sidecar idiom, `DDP_SCRIPT`/`ddp_throughput`, per-rank result files, stale-result clearing; keep that idiom and generalize it; do not re-teach DDP); `sec_multi_gpu` (α–β, `eq_ring_traffic`, weak vs strong scaling); `tab_collectives`; `subsec_hw-interconnects` (the build box has no P2P; read its transport lines as the contrast case); `sec_repro` (seeds per rank).
- **Builds on (this part):** 5.4 (node topology, intra-node α–β and busbw), 5.5 (inter-node α–β, IB vs RoCE, hierarchical all-reduce), `d2l.chip_specs()` (5.1). Artifacts used: `alpha_beta.json` in `lm-hw-node-p2p` (5.4) and `lm-hw-cluster-nccl` (5.5). Chapter 5 supplies the models and data: `d2l.fit_alpha_beta` (5.3), `d2l.parse_nccl_tests`, `d2l.nccl_busbw_factor`, `d2l.coll_time` (5.4), `d2l.ring_allreduce_time`, `d2l.tree_allreduce_time`, `d2l.hier_allreduce_time`, `d2l.a2a_inter_node_time` (5.5), and the measurements in `lm-hw-a100-microbench`, `lm-hw-node-p2p` and `lm-hw-cluster-nccl`. 6.1 imports them and defines no α–β model of its own. If those artifacts lack a link class, the notebook measures a 6-point all-reduce sweep (1 MB – 1 GB), fits it with `d2l.fit_alpha_beta`, keeps the fit in its own launch bundle (`lm-launch-<tag>`) and labels it "measured here".
- **Exports:**
  - `d2l.launch(script, nodes=1, gpus_per_node=8, **env) -> LaunchResult` (reserved name; contract below).
  - `d2l.LaunchResult`: dataclass with `tag, mode, ok, returncode, elapsed_s, world_size, hosts, log_dir, rank_logs, errors, meta, replayed`; methods `.logs(align=True) -> DataFrame`, `.rank0_stdout() -> str`, `.nccl_info() -> dict`, `.raise_for_status()`.
  - `d2l.init_dist(timeout_s=300, backend=None, mesh_shape=None, mesh_dim_names=None) -> SimpleNamespace(rank, local_rank, world_size, local_world_size, node_rank, hostname, device, backend, mesh)`: called inside launched scripts.
  - `d2l.rank_log(event, **fields)`: appends one JSON line (`t_wall`, `t_mono`, `rank`, `host`, `event`, fields) to `$D2L_LOG_DIR/rank{r}.jsonl`, flushed per line.
  - `d2l.read_rank_logs(source, align=True) -> pandas.DataFrame`: all ranks, time-sorted; `align=True` subtracts the per-rank clock offset estimated from `sync` events (below).
  - `d2l.CommCounter(group=None)`: context manager that wraps `torch.distributed` collectives and records `op, group_size, bytes_in, bytes_out, wire_bytes, t_ms`; `.summary() -> DataFrame`. Wire-byte factors are those of nccl-tests: all-reduce $2(k-1)/k$ of the message, all-gather and reduce-scatter $(k-1)/k$ of the full tensor, all-to-all $(k-1)/k$ of the input, broadcast and send/recv 1.
  - `d2l.comm_events(prof_or_trace) -> DataFrame`: reads the `record_param_comms` events of a `torch.profiler` trace (collective name, in/out element counts, group size, dtype) into bytes. This is the counter for library code (FSDP2, DDP, DTensor) that bypasses `dist.*` wrappers.
  - `d2l.parse_nccl_info(text) -> dict`: transports per channel (`P2P/CUMEM`, `SHM`, `NET/IB`, `NET/Socket`, GDRDMA flag), bootstrap interface, NCCL version, ring and tree summaries.
- **Models and data:** none (synthetic tokens). R2-scale `d2l.GPT` (≈ 150–400M parameters, from 4.2's config) for the DDP baseline.
- **Packages:** `torch==2.15.0` (the part's pin; the doc tree this chapter was checked against is the 2.15 one, so they agree), NCCL as shipped with it, `pandas`; SLURM ≥ 21.08 on the cluster for `--gpus-per-node`.

#### Narrative arc

1. **The question.** Opening cell: the sidecar idiom of `sec_multi_gpu_concise` on a two-node allocation. Run `torchrun --standalone --nproc-per-node=8 train.py` on each node. Both nodes train happily, print `WORLD_SIZE=8`, and the loss falls, but the global batch is half what the config says and the two "jobs" never talk. Printed symptom: world size 8 where 16 was intended, and identical logs from two hosts. The notebook calls this split-brain, and 6.1 is about making it impossible.
2. **The scaling ladder** (`tab_lm_scaling_ladder`; replaces `subsec_training_ladder`/`fig_tools_training_ladder` of `sec_training_systems`). Rungs: one GPU → one node (DDP) → one node (FSDP, TP) → few nodes → hundreds of nodes. Columns: what breaks first, the symptom, the fix, the section. Rows are justified with cited numbers only: RSC-1 reports a 1024-GPU job MTTF of 7.9 h and projects 1.8 h at 16,384 GPUs (Revisiting Reliability, 2410.21680); Llama 3 saw 419 unexpected interruptions in 54 days on a 16K-GPU job (2407.21783); MegaScale reports 55.2% MFU at 12,288 GPUs (2402.15627). The table is read, not derived; 6.8 derives its consequences.
3. **The process model.** One torchrun agent per node, one worker process per GPU. Table `tab_lm-launch-env`: `RANK, LOCAL_RANK, WORLD_SIZE, LOCAL_WORLD_SIZE, GROUP_RANK (= node rank), ROLE_RANK, MASTER_ADDR, MASTER_PORT, TORCHELASTIC_RESTART_COUNT, TORCHELASTIC_MAX_RESTARTS, TORCHELASTIC_RUN_ID` (all verified in the torchrun docs, 2.15). Global rank = `GROUP_RANK × LOCAL_WORLD_SIZE + LOCAL_RANK`.
4. **Rendezvous.** The c10d backend (`--rdzv-backend=c10d --rdzv-endpoint=HOST:PORT --rdzv-id=ID`) versus the static `--master-addr` form; `--standalone` is single-node only. Elastic ranges `--nnodes=MIN:MAX` and `--max-restarts` (used in 6.8). Two networks, two interface variables: the control path (TCPStore, NCCL bootstrap) uses `NCCL_SOCKET_IFNAME`/`GLOO_SOCKET_IFNAME`; the data path uses the IB/RoCE HCAs (`NCCL_IB_HCA`). Cell: parse an `NCCL_DEBUG=INFO` log and print which transport each channel used.
5. **SLURM.** Anatomy of the batch script (template below): `#SBATCH --nodes --ntasks-per-node=1 --gpus-per-node --cpus-per-task --exclusive --time --output`; head node via `scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1`; port derived from the job id; `srun torchrun ...` with `--rdzv-id=$SLURM_JOB_ID`. Contrast with the "one `srun` task per GPU and rank from `SLURM_PROCID`" alternative and say why this chapter uses the torchrun-per-node form (elastic restarts, per-rank log redirection). TorchTitan's `scripts/multinode_trainer.slurm` is the production counterpart (README, 2026-10).
6. **`d2l.launch` in detail** (see "From-scratch implementation").
7. **Process groups and device meshes.** `dist.new_group(ranks)` and `init_device_mesh("cuda", (nodes, gpus_per_node), mesh_dim_names=("inter", "intra"))`. Because ranks are numbered node-major and a mesh is row-major, the last mesh axis is intra-node by construction when its size divides `gpus_per_node`. Cell: gather hostnames with `all_gather_object` and assert every `mesh["intra"]` row sits on one host and every `mesh["inter"]` column spans all hosts. This is the invariant 6.7 builds on (`d2l.make_mesh`).
8. **Rank-tagged logs and clock alignment.** Use `--redirects 3 --tee 3 --log-dir DIR --log-line-prefix-template "[rank${rank}@${hostname}]: "` for raw streams; `d2l.rank_log` for structured events. After `dist.barrier()`, each rank logs a `sync` event; `read_rank_logs(align=True)` sets each rank's offset to the median difference between its sync stamp and rank 0's. Cell: print the residual spread after alignment; state that cross-node alignment is only as good as the barrier's skew (microseconds to a few hundred microseconds) and is not used below that.
9. **Hangs.** The notebook builds up in four steps (all bounded by `timeout_s=30`, so CI-safe):
   - (a) *Same shape, swapped order.* Rank 0 calls `all_reduce(A); all_reduce(B)`, rank 1 calls `all_reduce(B); all_reduce(A)` with equal shapes. The job completes with silently wrong values; the cell prints the expected and actual sums.
   - (b) *Different shapes.* Rank 1 skips one collective behind a data-dependent branch. The job hangs until the watchdog fires; show the `Watchdog caught collective operation timeout` text from the rank logs.
   - (c) `TORCH_DISTRIBUTED_DEBUG=DETAIL`: the mismatch is reported at the call with op and shapes of every rank (works on gloo too, so it runs at T1).
   - (d) The flight recorder: `TORCH_NCCL_TRACE_BUFFER_SIZE=2000`, `TORCH_NCCL_DUMP_ON_TIMEOUT=1`, dump prefix via `TORCH_FR_DUMP_TEMP_FILE` (older name `TORCH_NCCL_DEBUG_INFO_TEMP_FILE`). Run `torchfrtrace <dir>` (or `python -m torch.distributed.flight_recorder.fr_trace`) on the dumps from the T3/T4 run; the committed output names the rank and collective sequence id at which the ranks diverge. Also send `SIGUSR1` (registered by `init_dist` through `faulthandler`) to the hung ranks to collect Python stacks; show which line each rank is blocked on.
10. **Reading a job's first minutes.** A table of startup symptoms with their cause: init timeout (wrong interface or port), `NET/Socket` instead of `NET/IB` (throughput 10× lower, no error), duplicate GPU detected (two ranks on one device: `LOCAL_RANK` mapping), zombie processes from a previous run holding memory. Each has a one-line detector in `d2l.launch`'s pre-flight (see below).

#### Derivations

**D1. Matching by sequence number (why a mismatch silently corrupts or hangs).**
- *Statement.* On one communicator, the $i$-th collective issued by each rank is matched with the $i$-th issued by every other rank, regardless of call site or intended meaning.
- *Assumptions.* NCCL/gloo semantics: collectives on a communicator are ordered per rank; matching is positional.
- *Steps.* (1) Let rank $r$ issue the sequence $c^r_1, c^r_2, \dots$, each a pair (op, element count, dtype). (2) The $i$-th operation executes when all ranks have issued their $i$-th. (3) If op and count agree for all $r$, it completes and writes into the buffers each rank passed, so swapped call sites combine the wrong tensors. (4) If counts differ, the transport either errors, truncates, or blocks until timeout. (5) If one rank issues fewer operations, the others block at the first unmatched index.
- *Result.* Silent corruption iff the (op, count, dtype) sequences agree but the buffers' meanings differ; hang or error otherwise.
- *Check.* Cells 9(a) and 9(b) compare the observed outputs with the values computed by hand from this rule (e.g. with $A=1$, $B=10$ on two ranks, rank 0's "A" becomes $1+10=11$).

**D2. Job MTTF scales as $1/N$ and unhealthy nodes at launch.**
- *Statement.* A job on $N$ nodes with independent per-node failure rate $r_f$ has $\mathrm{MTTF} = 1/(N r_f)$; a launch that draws $N$ nodes each unhealthy with probability $q$ succeeds with probability $(1-q)^N$.
- *Steps.* (1) Superposition of independent Poisson processes sums the rates. (2) $\mathrm{MTTF}=1/\lambda$. (3) Independence of the draws gives the product.
- *Check.* Evaluate with RSC-1's $r_f = 6.50$ per 1000 node-days and RSC-2's $2.34$ (Fig. 7 legend of 2410.21680) at 16,384 and 131,072 GPUs; the paper's projections (1.8 h and 0.23 h) are reproduced. State that the paper's empirical points at 8 and 1024 GPUs (47.7 days, 7.9 h) lie below the line $1/(N r_f)$, so the fit is a projection. 6.8 refits properly.

**D3. Mesh contiguity.** Claim: with row-major mesh `mesh = arange(W).reshape(shape)` and node-major ranks, the members of one group along axis $j$ are $r_0 + i\cdot\text{stride}_j$ for $i<\text{shape}_j$, where $\text{stride}_j=\prod_{i>j}\text{shape}_i$. All members share a node iff $\text{stride}_j\cdot\text{shape}_j\le G$ (where $G$ = `gpus_per_node`), because then the whole span lies inside one aligned block of $G$ consecutive ranks. Proof by index arithmetic (three lines). Check: assert on the gathered hostnames for shapes $(2,8)$, $(4,4)$, $(2,2,4)$ and a counter-example $(8,2)$ where the *first* axis is intra-node.

#### From-scratch implementation

`d2l.launch` (≤ 150 lines including SLURM template; no framework beyond `subprocess`, `pathlib`, `hashlib`).

**Signature and option convention.** `launch(script, nodes=1, gpus_per_node=8, **env)`. Keyword arguments in UPPER_CASE are environment variables for the ranks (`NCCL_DEBUG='INFO'`, `D2L_BACKEND='gloo'`, `TORCH_DISTRIBUTED_DEBUG='DETAIL'`). Lower-case keywords are launcher options: `args=()` (script arguments), `tag=None`, `mode='auto'`, `timeout=600`, `partition=None`, `time_limit='00:30:00'`, `cpus_per_task=None`, `flight_recorder=True`, `max_restarts=0`, `hosts=None`, `rdzv_port=None`, `check_gpus=True`, `stream=True`. `script` is a path written by a `%%writefile` cell, the text of a script, or a module spec such as `"-m torchtitan.train"` (used in 6.7).

**Modes.**
- `local`: one node. `torchrun --standalone --nproc-per-node=G --redirects=3 --tee=3 --log-dir=LOG --log-line-prefix-template=... script args` in a new session (`start_new_session=True`); rank-0 lines stream to the notebook; on timeout send `SIGUSR1` (stacks), wait 5 s, then `SIGTERM` to the whole process group, then `SIGKILL`. This is the T3 path (the notebook's kernel is the submitter, never a rank).
- `loopback`: simulate multi-node on one host. Start `nodes` torchrun agents with `--nnodes=N --node-rank=i --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:PORT`, each with `gpus_per_node` workers. With `D2L_BACKEND=gloo` the workers use CPU (T1 simulated process group); with GPUs (the T2 pair or the T3 node, never the build box), `CUDA_VISIBLE_DEVICES` is partitioned across agents. Real rendezvous, real rank arithmetic, fake network.
- `slurm`: write `job.sbatch` (template below) into `D2L_SHARED_DIR`, `sbatch --parsable`, poll `squeue`/`sacct`, tail the rank-0 log, `scancel` on timeout. If `SLURM_JOB_ID` is already set (notebook inside an `salloc`), skip `sbatch` and call `srun --nodes N --ntasks-per-node=1` directly.
- `ssh`: for clusters without SLURM; `hosts=[...]`, one `ssh host torchrun --nnodes N --node-rank i ...` per host, same flags.
- `replay`: load the committed bundle for `tag` (below); no execution.
- `auto`: `D2L_LAUNCH_MODE` if set; else `local` when `nodes==1` and `torch.cuda.device_count() >= gpus_per_node`; else `slurm` when `nodes>1` and `sbatch` exists; else `replay`. If nothing applies, raise with the message "no GPUs and no committed run for tag X".

**SLURM template** (spec; `{...}` filled by `launch`):
```
#!/bin/bash
#SBATCH --job-name={tag} --nodes={nodes} --ntasks-per-node=1 --gpus-per-node={g}
#SBATCH --cpus-per-task={cpus} --exclusive --time={time_limit} --output={log}/slurm-%j.out
{extra_sbatch}
export HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export PORT=$((20000 + SLURM_JOB_ID % 20000))
export OMP_NUM_THREADS=$(( {cpus} / {g} )) PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export D2L_LOG_DIR={log} D2L_TAG={tag} {env_exports}
srun torchrun --nnodes={nodes} --nproc-per-node={g} --rdzv-backend=c10d \
     --rdzv-endpoint=$HEAD:$PORT --rdzv-id={tag}-$SLURM_JOB_ID --max-restarts={max_restarts} \
     --redirects=3 --tee=3 --log-dir={log}/torchelastic \
     --log-line-prefix-template='[rank${{rank}}@${{hostname}}]: ' {script} {args}
```
**Environment set by default** (overridable): `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`, `TORCH_NCCL_TRACE_BUFFER_SIZE=2000`, `TORCH_NCCL_DUMP_ON_TIMEOUT=1`, `TORCH_FR_DUMP_TEMP_FILE={log}/fr/rank_`, `NCCL_DEBUG=WARN` (`INFO` on request), `PYTHONUNBUFFERED=1`. `NCCL_SOCKET_IFNAME` is not guessed; the pre-flight prints the interface NCCL chose, and `D2L_SLURM_ARGS`, `NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME` are passed through when set by the user. Variable names verified against the PyTorch flight-recorder tutorial and torchrun docs (2.15); re-check at the pinned version.

**Pre-flight (inside `init_dist`, before the script's own code).** (1) All-reduce of `rank` equals $W(W-1)/2$. (2) `all_gather_object` of hostnames: asserts `nodes` distinct hosts and `gpus_per_node` ranks per host, stores the map in `meta['hosts']`. (3) Parses the first NCCL INFO lines when `NCCL_DEBUG=INFO`; warns if any inter-node channel is `NET/Socket` while an IB device exists. (4) `launch` itself, before starting, runs `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` (local/ssh) and refuses to start when foreign processes hold more than 2 GiB on the target GPUs unless `check_gpus=False`; the message names the PIDs. (5) The init timeout is 120 s, so a rendezvous failure surfaces in two minutes, not at the 10-minute scheduler limit.

**Log collection.** `launch` creates `LOG/{rank_logs,torchelastic,fr,nccl}`; the sbatch/`ssh` modes require `D2L_SHARED_DIR` on a shared file system (checked by writing and reading a file from every node during pre-flight). After the run it builds `LaunchResult` and parses `errors` from torchelastic's per-rank `error.json` (scripts decorate `main` with `@record`).

**Committing outputs for readers without GPUs.**
- A release script (`make lm-release NB=<section>`) re-executes the T3/T4 notebooks with `D2L_LAUNCH_MODE` set to `local` or `slurm`. Every `launch` call writes a bundle `LOG/` and a `meta.json` containing: `tag`, script text hash (sha256), `nodes`, `gpus_per_node`, `env` (UPPER_CASE kwargs only), hostnames, GPU model, driver, CUDA, PyTorch and NCCL versions, `git` commit, SLURM job id, start/end time.
- A packager compresses the bundle (rank JSONL, torchelastic logs, `fr/` dumps, `nccl/` INFO logs, sbatch script, `meta.json`; profiler traces reduced to JSON summaries, with at most one full Chrome trace per notebook, gzip, ≤ 20 MB) to `lm-launch-<tag>.tar.gz`, uploads it, and registers `d2l.DATA_HUB['lm-launch-<tag>'] = (url, sha1)`.
- In `replay` mode `launch` downloads the bundle, recomputes the script hash, and raises `StaleArtifactError` if it differs from `meta.json`, with the message "script changed since the cluster run; re-run on the cluster". Otherwise it returns a `LaunchResult` with `replayed=True` whose `rank0_stdout()` is the recorded stream. Notebook cells therefore run unchanged on both paths.
- Test: CI runs each T3/T4 notebook once with the bundle present and once with the bundle's script hash corrupted, and asserts the second run fails with `StaleArtifactError`.

**Numerical tests (T1, gloo).** (1) `launch(..., nodes=2, gpus_per_node=2, mode='loopback', D2L_BACKEND='gloo')` returns `world_size == 4`, hostnames identical (one machine), `mesh['intra']` rows `[[0,1],[2,3]]`. (2) A script printing `os.environ` entries returns the table of `tab_lm-launch-env` values with the arithmetic of item 3 holding. (3) A script exiting with code 3 on rank 1 gives `ok=False`, `errors[1]` populated, and no zombie processes (`psutil` check). (4) A 5 s hang bounded by `timeout=20` is killed with the process group.

#### In production

- `torchrun` is the launcher of record (PyTorch docs). Table `tab_lm_launchers`: `torchrun`, `accelerate launch`, `deepspeed`, `mpirun`/`srun`-per-GPU, TorchTitan's `run_train.sh`/`multinode_trainer.slurm`, Megatron-LM's launch scripts. Columns: who assigns ranks, elastic restart, log redirection, when to use. Only facts from the cited docs; the Megatron and DeepSpeed rows cite their READMEs.
- Containers (pyxis/enroot or Apptainer), `--numa-binding=node`, and NCCL environment tuning are named, not taught. The NCCL environment-variable reference and Stas Bekman's book are the pointers.
- The from-scratch `d2l.launch` is the test oracle for the structure of any production launch script: the sbatch of `scripts/multinode_trainer.slurm` must produce the same `tab_lm-launch-env` values.

#### Headline experiment

- **Hypothesis.** (i) The same DDP job runs with global world size 16 on two nodes and is slower per GPU than on one node, by an amount the α–β model of 5.5 predicts within the spread of repeated runs. (ii) Forcing the socket transport (`NCCL_IB_DISABLE=1`) slows the job by a large factor, and `parse_nccl_info` detects it from the log of the first minute.
- **Setup.** R2-scale `d2l.GPT` (BF16, DDP, `torch.compile` off), microbatch 8 × 2048 tokens per GPU, synthetic tokens, 60 steps (10 warm-up), seeds 0–2. Launches: 1 node × 8 GPUs, 2 nodes × 8 GPUs, 2 nodes × 8 GPUs with `NCCL_IB_DISABLE=1`. Tier T4 for the last two.
- **Procedure.** `d2l.launch` three times per condition; each run logs `step_time_ms` per rank per step and `time_to_first_step_s` (process start to first completed optimizer step). Predicted step time: $t_{\text{compute}}$ from the 1-node run plus exposed communication from the model with the DDP bucket overlap of `sec_multi_gpu_concise` and the inter-node busbw of 5.5.
- **Metrics.** Tokens/s/GPU (median of steps 10–59), weak-scaling efficiency, per-rank step-time spread (max − min over ranks), time to first step, transport string.
- **Plots.** `fig_lm-launch-scaling`: bars of tokens/s/GPU for the three conditions with min–max whiskers over the three launches and the predicted value as a marker. `fig_lm-launch-timeline`: one launch's per-rank step starts on the aligned clock (ranks on the y-axis, hosts colour-coded), with the barrier-skew band.
- **Expected outcome.** Direction only: 2-node < 1-node per GPU; socket < IB. Magnitudes are "to be measured"; no number is quoted from a source for this model.
- **Noise check.** Three launches per condition; a difference counts only if it exceeds the largest min–max whisker; the alignment residual is printed alongside.
- **Hang notebook (second experiment, same tier).** Cells 9(a)–(d) on 2 nodes × 8 GPUs with the 60 s timeout; the committed bundle contains the `fr/` dumps and the `torchfrtrace` output.
- **T1 reduction.** Loopback 2×2 gloo: conditions 1–2 only (no socket variant), tiny model; hang cells (a)–(c) run live; (d) replays.
- **Committed.** Three bundles (`ddp_1x8`, `ddp_2x8`, `ddp_2x8_sockets`), `hang_2x8`.

#### Figures and tables

- `fig_lm-launch-topology`: processes, agents, rendezvous store and the two networks, drawn from a real run's `meta['hosts']` (cell `launch-topology`).
- `tab_lm_scaling_ladder` (rungs, what breaks, fix); `tab_lm-launch-env` (variables and values from a live loopback run); `tab_lm_launchers`; `tab_lm_startup_symptoms`.
- `fig_lm-launch-scaling`, `fig_lm-launch-timeline` as above; `tab_lm_hang_cases` (cases a–d: observed behaviour, detector, time to detect).

#### Pitfalls and misconceptions

- `--standalone` on each node of a multi-node job (the opening symptom).
- "A hang is a slow network": a hung collective is a sequencing bug until the flight recorder says otherwise.
- Equal shapes hide order bugs (D1).
- Starting the launch from a kernel that already holds GPU memory.
- Clock skew read as straggling.
- A shared file system that is not shared: logs and the script exist on node 0 only.
- `NCCL_DEBUG=INFO` left on in production: log volume and slowdown.
- Treating `ok=True` as success without checking world size and transport.

#### Summary

- One torchrun agent per node and rendezvous at a single endpoint turn $N$ hosts into one world of $N\times G$ ranks; `--standalone` makes $N$ separate worlds.
- A launch is only as good as its first minute: interface, transport, hostnames and GPU ownership can all be checked before training starts.
- Collectives match by position, so ordering bugs corrupt silently when shapes agree and hang when they do not.
- Rank-tagged structured logs plus the flight recorder turn a hang into a named rank and sequence id.
- Committed bundles with a script hash let the notebook run unchanged without GPUs and refuse to replay stale results.

#### Exercises

1. *(check)* On two ranks, rank 0 issues `all_reduce(A); broadcast(B, src=0)` and rank 1 issues `broadcast(B, src=0); all_reduce(A)`, with `A` and `B` of equal size. Predict what each rank ends with, or that the job hangs, before running. **Sketch:** by D1 position 1 pairs an all-reduce with a broadcast; mismatched ops on one communicator are undefined; on NCCL the usual outcome is a hang or garbage, not a clean error; `TORCH_DISTRIBUTED_DEBUG=DETAIL` reports the op mismatch.
2. *(check)* A 4-node × 8-GPU job uses `init_device_mesh("cuda", (8, 4))`. Which mesh axis is intra-node? Which collectives would you place on each axis? **Sketch:** by D3 axis 1 (size 4, stride 1) spans 4 consecutive ranks: intra-node; axis 0 has stride 4 and size 8, so each column touches all nodes. Put the high-volume low-latency collectives on axis 1.
3. *(derive)* A job needs $N$ healthy nodes out of a pool; each is unhealthy with probability $q$. Derive the expected number of launch attempts and the probability of success within three attempts for $N=128$, $q=0.003$, and show how a spare-node pool of $N+s$ changes it. **Sketch:** success per attempt $(1-q)^N$; attempts are geometric with mean $(1-q)^{-N}$; with spares, success is $P(\text{Binomial}(N+s,q)\le s)$.
4. *(derive)* Prove D3 for an arbitrary mesh shape whose product is $N\cdot G$, and give a shape for which the condition fails although shape$_j\le G$. **Sketch:** ranks on an axis are $r_0 + i\,\text{stride}_j$; node id is $\lfloor r/G\rfloor$; all equal for $i<\text{shape}_j$ iff the span fits inside one aligned block of $G$.
5. *(measure)* Predict, then measure, the time-to-first-step of the 2×8 launch as a function of model size (R2-scale vs 4× larger). Which part of it depends on size? **Hint:** log `init_complete`, `first_forward_done`, `first_step_done` with `d2l.rank_log`.
6. *(measure)* Set `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` and the `timeout_s` of `init_dist` to different values and run the hang of 9(b). Predict which one fires first and the wall-clock to termination; record the stack dump. State the trade-off of short timeouts when a rank is legitimately slow (e.g., during checkpointing, 6.8).
7. *(open)* Write a launcher for a scheduler other than SLURM (Kubernetes with the PyTorchJob operator, or Ray). Which parts of `d2l.launch` survive unchanged (rank log schema, pre-flight, bundle format) and which need a new adapter?

#### Further reading

- Primary: [PyTorch Distributed Data Parallel paper](https://arxiv.org/abs/2006.15704) (Li et al.), for the process-group design; [torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html) (the elastic launcher, rendezvous and environment contract); [DeviceMesh recipe](https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html); [NCCL user guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/) and its [environment-variable reference](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html).
- New (verified 200 on 2026-10-28): [Flight Recorder tutorial](https://docs.pytorch.org/tutorials/unstable/flight_recorder_tutorial.html) (env vars, dump files, `fr_trace`); [Slurm `sbatch`](https://slurm.schedmd.com/sbatch.html) and [`srun`](https://slurm.schedmd.com/srun.html) manuals; [Revisiting Reliability in Large-Scale Machine Learning Research Clusters](https://arxiv.org/abs/2410.21680) (title re-verified 2026-10-29).
- Tutorial: Stas Bekman, [Machine Learning Engineering Open Book](https://github.com/stas00/ml-engineering), the sections on SLURM and debugging hangs; TorchTitan [`docs/debugging.md`](https://github.com/pytorch/torchtitan/blob/main/docs/debugging.md).
- Code: [torchtitan](https://github.com/pytorch/torchtitan) (`run_train.sh`, `scripts/multinode_trainer.slurm`); [nccl-tests](https://github.com/NVIDIA/nccl-tests).

#### Risks and open decisions

- **No cluster allocation or no SLURM.** Fallback: `ssh` mode, or run the T4 notebook with the committed bundle only; the T1 loopback path still demonstrates rendezvous and rank arithmetic.
- **Chapter 5's microbenchmark artifact lacks a link class.** Fallback: 6.1 measures and commits its own fit (documented above).
- **Environment-variable names drift** (`TORCH_NCCL_DEBUG_INFO_TEMP_FILE` vs `TORCH_FR_DUMP_TEMP_FILE`). Pin the PyTorch version, test in CI on the T2 pair, and keep both names in `launch`'s defaults behind a version check.
- **Cross-node clock alignment** may be too coarse for sub-millisecond Gantt charts; the notebook says so and 6.4 does not depend on it.
- **Flight recorder is NCCL-only.** T1 replays committed dumps; this is stated in the notebook.
- **Zombie ranks after a failed launch** can poison the next run on shared nodes; `launch` kills the process group and checks GPU ownership, but a cluster-level epilog is the real fix.

### 6.2 Sharded data parallelism at scale

- **Label:** `sec_lm_fsdp` · **Tier:** T3 (reduced T1 config: `FSDPLite` and, if the build image supports gloo, `fully_shard`, on 2–4 CPU ranks in float64 with a 4-block `d2l.GPT`, hidden 64) · **Budget:** T3 ≤ 1.5 h wall on 8× A100 (40 GB per GPU assumed; ≈ 12 GPU-h, planned: ~15 runs of ~4 min each plus sweeps); peak GPU memory per run is the quantity under study, predicted ≤ 36 GB (0.9 × 40) for the configurations that fit, and W2, W5–W7 and W11 are the ones that do not at full depth (see the headline experiment); T1 ≈ 5 min · **Codex units:** 6.7 (state sharding), 6.8 (prefetching and practical implementation)
- **Builds on (book):** `sec_multi_gpu_concise` (`subsec_mgp-fsdp`, `fig_fsdp_lifecycle`, `tab_collectives`; keep its lifecycle picture and recap it in one paragraph), its "ZeRO memory budget" and "Cost of sharding" exercises (6.2 starts where those stop: with precisions that differ between parameters and reduction); `sec_memory_precision` (`subsec_mp-anatomy` 16P bytes, `subsec_mp-checkpointing`, `subsec_mp-measuring`: the allocation-trace and `memory_snapshot` idiom); `sec_multi_gpu` (`eq_ring_traffic`, bf16 gradient rounding exercise); `sec_perf_model` (profiler reading). Not re-taught: ZeRO stage definitions, the 16-byte budget, collectives.
- **Builds on (this part):** 6.1 (`d2l.launch`, `init_dist`, `CommCounter`, `comm_events`, `read_rank_logs`); 5.4 (intra-node busbw and α: the NVSwitch row, with the PCIe-pair row alongside; Chapter 5's `d2l.fit_alpha_beta`/`d2l.coll_time` fits on the artifact `lm-hw-a100-microbench` (intra-node entries)), 5.1 (`d2l.chip_specs()` column `p16_tflops` for peak BF16 FLOP/s), 4.2 (`d2l.flops_per_token`, `d2l.mfu`), 5.3 (host↔device bandwidth for the offload prediction; host-device bandwidth from `lm-hw-a100-microbench`); 1.1 (`d2l.load_hf`, `d2l.config_memory`); 1.5 (`d2l.paired_bootstrap`).
- **Exports:**
  - `d2l.ModelShape` (dataclass) with `from_hf(config)`; fields `d, L, heads, kv_heads, head_dim, ffn, vocab, tied, num_experts, top_k, moe_ffn, shared_experts`; properties `params_layer, params_root, params_total, params_active`. It also exposes the HF-named attributes that 4.2's `flops_per_token` and `mfu` read (`num_hidden_layers, hidden_size, num_attention_heads, num_key_value_heads, head_dim, intermediate_size, vocab_size`) as read-only aliases of `L, d, heads, kv_heads, head_dim, ffn, vocab`. Counts every matmul and norm parameter; checked against `sum(p.numel())` of the HF model on the meta device.
  - `d2l.fsdp_costs(shape, k, *, tokens, param_bytes=2, reduce_bytes=4, reshard_after_forward=True, ac='full', group_layers=1, offload=False) -> dict` (`comm`, `mem`, `flops` as per the chapter convention).
  - `d2l.fsdp_wrap(model, mesh, *, block_cls, mp_policy='bf16-fp32', reshard_after_forward=None, forward_prefetch=1, backward_prefetch=1, offload=False) -> model`: the production FSDP2 recipe of this section as one call (per-block `fully_shard`, root, `MixedPrecisionPolicy`, prefetch lists); the trainer module imported by 7.8 (with LoRA: wrap with frozen base parameters via `ignored_params`/uniform dtype as 7.8 requires) and 8.6. `d2l.fsdp_train_step(model, optimizer, batch, loss_fn, *, accum=1, grad_clip=None) -> loss`: one optimizer step with accumulation (`set_requires_gradient_sync` on the last microbatch) and global-norm clipping.
  - `d2l.FSDPLite(model, group=None, *, is_block, reshard_after_forward=True, param_dtype=None, reduce_dtype=None, grad_divide=None)`: flat-parameter layer-wise sharding in ≤ 110 lines; `.shard_parameters()`, `.gather_full_state_dict()`.
  - `d2l.exposed_comm(trace, comm_pattern='nccl') -> dict(compute_ms, comm_ms, exposed_ms, overlap_frac)`: interval arithmetic on GPU kernels of a profiler Chrome trace (communication kernels minus the union of other kernels).
  - `d2l.saved_bytes(fn, *args) -> int`: bytes of tensors autograd saves for backward, enumerated with `torch.autograd.graph.saved_tensors_hooks` (deduplicated by storage); used for per-layer activation counts here and in 6.3 and 6.5.
  - No FLOPs or MFU function is exported here: 6.2 imports 4.2's `d2l.flops_per_token(arch, T=None, causal_half=False)` and `d2l.mfu(tokens_per_s, arch, peak_flops, n_gpus=1)` and passes a `ModelShape` as `arch`. 4.2's `mfu` is per job; 6.2 measures per GPU, so the sweeps call `d2l.mfu(tokens_per_s_job, shape, peak_flops, n_gpus=k)` (or equivalently per-GPU tokens/s with `n_gpus=1`). Convention below.
- **Models and data:** `Qwen/Qwen3-8B` config only (random initialization on the meta device, `torch.nn.Module.to_empty`, then `reset_parameters`/`_init_weights`; no weights downloaded; Apache-2.0). Synthetic token batches for the sweeps; a 200-step FineWeb-Edu slice through `d2l.PackedStream` (4.1) for one loss-decrease sanity run (loss curve only checks that sharding does not break training).
- **Packages:** `torch==2.15.0` (`torch.distributed.fsdp.fully_shard`, `MixedPrecisionPolicy`, `OffloadPolicy`/`CPUOffloadPolicy`, `FSDPModule.set_modules_to_forward_prefetch`, `set_modules_to_backward_prefetch`, `set_requires_gradient_sync`, `set_reshard_after_forward`; all names verified in the 2.15 docs), `transformers` (for `Qwen3ForCausalLM.from_config`), `torch.profiler`.

#### Narrative arc

1. **The question.** Opening cell: for Qwen3-8B, `ModelShape` gives $P = 8{,}190{,}735{,}360$ (reference value from the config; the cell recomputes it and asserts equality with the meta-device parameter count). The 16-bytes-per-parameter state is $131$ GB, more than any single A100 (40 or 80 GB). The cell prints the DDP attempt (T3 run, committed: out-of-memory at the optimizer step) and an FSDP2 run on 8 GPUs that trains. It then prints the measured peak next to the "obvious" prediction $16P/k = 16.4$ GB and shows the gap. The rest of the section explains the gap term by term.
2. **Recap in one paragraph** (`:numref:sec_multi_gpu_concise`): ZeRO stages, lifecycle, `tab_collectives`. Introduce the FSDP2 conventions: sharded parameters live in fp32 (the master copy), each layer is all-gathered in bf16 (`param_dtype`), gradients are reduced in `reduce_dtype`, and the sharded fp32 gradient meets sharded Adam moments.
3. **Where the bytes are: a term-by-term memory model** (`eq_lm_fsdp_mem`): sharded state, transient unsharded layers, reduce buffers, root group, activations under full checkpointing, logits, and memory outside the allocator (CUDA context, NCCL buffers). Cell: `fsdp_costs` for Qwen3-8B, $k = 8$, 8192 tokens per GPU, printed as a stacked-bar table (`tab_lm_fsdp_mem`).
4. **Communication per step** (`eq_lm_fsdp_comm`): forward all-gather, backward all-gather, reduce-scatter; the effect of `reshard_after_forward`, of the `int` form (reshard to a subgroup), and of `reduce_dtype`. Cell: bytes for each policy; check the book's "cost of sharding" claim of 1.5× DDP for equal precisions as a special case.
5. **When the all-gather hides: the critical tokens per GPU** (`eq_lm_fsdp_tstar`): a threshold independent of model size. Cell: evaluate with Chapter 5's intra-node busbw (the NVSwitch row, and the PCIe-pair row alongside; `topo.txt` decides which is the measured node) and `d2l.chip_specs()` peak (the `A100 PCIe 40 GB` row); plot the exposed fraction predicted by the model against tokens per GPU. $T^*$ scales as $1/\beta$, so the two rows differ by the busbw ratio of 5.4 (to be measured).
6. **Wrapping granularity.** Per block, per $g$ blocks, the whole model; embeddings and head as separate groups. Collective efficiency versus message size (`eq_lm_collective_eff`) and the memory cost of larger groups.
7. **Prefetching and resharding.** The default implicit prefetch (one ahead), explicit `set_modules_to_forward_prefetch`/`set_modules_to_backward_prefetch`, `reshard_after_forward ∈ {True, False, int}` with the root module's special default (False; verified in the 2.15 docs). Show the memory timeline of one layer under each.
8. **Mixed-precision policies.** `(param, reduce)` $\in$ {(bf16, fp32), (bf16, bf16), (fp32, fp32)}: bytes and numerical effect. Cell (CPU, T0): reduce-scatter in bf16 versus fp32 of 8 random gradient shards with the rounding model of the book's bf16-ring exercise; relative error versus $k$. Cite Llama 3: they reduce-scatter in FP32 for stability (2407.21783, section 3.3.2).
9. **CPU offload.** `CPUOffloadPolicy`: what moves (sharded params, grads and optimizer state live on CPU; the optimizer step runs on CPU) and the overlap condition from 5.3 applied to FSDP: offload pays only if per-layer transfer time fits under per-layer compute time. Predicted and measured for the 8B config; larger-model offload is 7.8's subject.
10. **Reading the trace.** Compute stream versus NCCL streams; `d2l.exposed_comm`; one annotated figure with the all-gather for layer $\ell+1$ under the compute of layer $\ell$, and the reduce-scatter lagging one layer behind in backward.
11. **Beyond one group.** HSDP (shard inside the node, replicate across nodes: a 2D mesh) and FSDP with TP or EP appear here as forward pointers to 6.3, 6.6, 6.7; one cell constructs the 2D mesh from 6.1's `make_mesh` and counts the collectives. A paragraph on optimizers that need whole matrices (Muon, `sec_muon`): sharded parameters force a gather per update.

#### Derivations

**D1. Peak memory per GPU** (`eq_lm_fsdp_mem`).
- *Statement.* For $P_L$ parameters per block, $P_R$ root parameters (embeddings, head, final norm), $L$ blocks, world size $k$, activations $A$ and logits $G$ per microbatch:
  $M_{\text{peak}} \approx \underbrace{16P/k}_{\text{state}} + \underbrace{(u_f + u_b)\,P_L}_{\text{transient layers}} + \underbrace{c_R P_R}_{\text{root}} + A + G + M_{\text{ext}}$, with $u_b = 14$ in the worst case (derivation below).
- *Assumptions.* FSDP2 with `MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)`; fp32 sharded parameters; AdamW; one block per group; implicit forward prefetch and one-ahead backward prefetch; full activation checkpointing; $T$ tokens per GPU.
- *Steps.* (1) Persistent: fp32 shard $4P/k$, fp32 gradient shard $4P/k$, two Adam moments $8P/k$: total $16P/k$. (2) Forward: current block's unsharded bf16 weights $2P_L$ plus the prefetched next block $2P_L$; $u_f = 4$. (3) Backward: current block's bf16 weights $2P_L$, the prefetched previous block $2P_L$, its unsharded bf16 gradient $2P_L$, the fp32 copy that feeds the reduce-scatter $4P_L$ (the previous block's reduce-scatter may still be running), and the output shard; bound $u_b = 2+2+2+4+4 = 14$. (4) Root group (not resharded after forward by default): bf16 weights $2P_R$ live through forward, bf16 gradient $2P_R$ and fp32 reduce buffer $4P_R$ at the end of backward: $c_R \le 8$ at the instant the root's reduce-scatter runs, and $2$ otherwise. (5) Activations with full checkpointing: $2LTd$ bytes of block inputs (bf16) plus one block's recompute working set $a_{\text{layer}}T$, with $a_{\text{layer}}$ counted by `d2l.saved_bytes`. (6) Logits: $T\cdot V$ elements times the bytes of the logits, their fp32 upcast and the loss gradient (HF's loss upcasts); for Qwen3-8B at $T = 8192$, $V = 151936$: $1.245\times10^9$ elements, the dominant activation term. (7) $M_{\text{ext}}$ = CUDA context and NCCL buffers, outside `max_memory_allocated`: measure by `torch.cuda.mem_get_info` minus `max_memory_reserved`.
- *Result.* `fsdp_costs(...)['mem']` returns each term.
- *Check.* `tab_lm_fsdp_mem`: predicted per term and total versus `max_memory_allocated` and `max_memory_reserved` for the baseline run, and the OOM prediction for the no-checkpointing run. The hypothesis "peak within ±10%" is stated before the run; the cell then decomposes any larger miss using `torch.cuda.memory._dump_snapshot` and names the term.

**D2. Communication per step and the critical token count** (`eq_lm_fsdp_comm`, `eq_lm_fsdp_tstar`).
- *Statement.* Per GPU per step (ring, group of $k$): $V_{\text{FSDP}} = \frac{k-1}{k}\,\big(b_p P\,[1 + \mathbb{1}_{\text{reshard}}] + b_r P\big)$ with $b_p$ the all-gather bytes per parameter (2 for bf16), $b_r$ the reduce-scatter bytes per parameter (4 for fp32). For (bf16, fp32) with resharding: $8P\,(k-1)/k$; without resharding: $6P\,(k-1)/k$; for (bf16, bf16): $6P(k-1)/k$ and $4P(k-1)/k$. DDP with bf16 gradients: $4P(k-1)/k$.
- *Overlap threshold.* The all-gather of block $\ell+1$ hides behind the compute of block $\ell$ iff $t_{\text{compute}}(\ell) \ge t_{\text{comm}}(\ell+1)$.
- *Steps.* (1) Forward block compute $\approx 2P_L T(1+\epsilon_a)/(\eta F)$, where $\epsilon_a = sh\,\text{head\_dim}/P_L$ accounts for attention at sequence $s$ ($\epsilon_a \approx 0.087$ for Qwen3-8B at $s=4096$). (2) Forward comm $\approx \alpha + \frac{k-1}{k}\,2P_L/\beta$. (3) Hence forward overlaps iff $T \ge T_f^* = \frac{\eta F}{(1+\epsilon_a)\beta}\frac{k-1}{k}$ (the $\alpha$ term is dropped when $n \gg 9\alpha\beta$, D3). (4) Backward compute is $4P_LT(1+\epsilon_a)/(\eta F)$ and the link carries the next block's all-gather ($2P_L$) and the previous block's reduce-scatter ($b_rP_L$): $T_b^* = \frac{(b_p + b_r)\,\eta F}{4(1+\epsilon_a)\beta}\frac{k-1}{k}$ (with $b_p, b_r$ in bytes per parameter; the forward threshold is $T_f^* = \frac{b_p\,\eta F}{2(1+\epsilon_a)\beta}\frac{k-1}{k}$, which equals the expression in step 3 for $b_p = 2$). For (bf16, fp32): $T_b^* = 1.5\,T_f^*$. (5) The step is communication-bound when $T < T_b^*$.
- *Result.* $T^*$ does not depend on $P$ or $L$: FSDP scales to larger models at the same tokens per GPU, and gets worse when $\beta$ falls (across nodes) or $F/\beta$ rises (faster chips).
- *Check.* Evaluate $T^*$ with $\eta$ measured from the compute-only step (comm kernels removed from the trace) and Chapter 5's busbw; sweep tokens per GPU at $\{\tfrac14, \tfrac12, 1, 2, 4\}\times T^*$ of the row `topo.txt` selects (rounded to multiples of 1,024 tokens, built from sequences of 1,024 to 4,096 tokens, capped at 16,384 per GPU; the earlier fixed grid $\{2\text{k},\dots,16\text{k}\}$ could lie wholly above $T^*$ on an NVSwitch node, where it would not test the threshold) and compare the predicted and measured exposed fraction from `exposed_comm`.

**D3. Message-size efficiency** (`eq_lm_collective_eff`). With $t(n) = \alpha + n/\beta$ the delivered fraction of $\beta$ is $1/(1+\alpha\beta/n)$, which is at least 0.9 iff $n \ge 9\alpha\beta$. Check: with Chapter 5's intra-node $\alpha,\beta$ (NVSwitch row; PCIe-pair row alongside) compute $9\alpha\beta$ and compare with the per-group all-gather bytes of each wrapping (one block, 2, 4, whole model); confirm on a bare all-gather sweep (T3, 1 min).

#### From-scratch implementation

`d2l.FSDPLite` (≤ 110 lines) and its helpers (≤ 40 lines). Total ≤ 150.

- `_Gather(torch.autograd.Function)`: forward casts the local flat shard to `param_dtype`, all-gathers into a full flat buffer (`dist.all_gather_into_tensor`); backward casts the incoming full gradient to `reduce_dtype`, `dist.reduce_scatter_tensor`, divides by `grad_divide` (default $k$, correct for equal token counts per rank, with the unequal-count correction left to the caller as in the book's gradient-normalization exercise), casts back to the shard dtype.
- `FSDPUnit(nn.Module)`: owns the flat shard (`nn.Parameter` of length $\lceil n/k\rceil$, zero padded), the names/shapes/offsets of the original parameters, and forward as `torch.func.functional_call(module, views, *args)` where `views` are slices of the gathered buffer. `reshard_after_forward=True` wraps the call in `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`, so that backward re-gathers; `False` lets autograd keep the gathered views. This couples resharding to activation recomputation, which FSDP2 does not; the section says so, and states that the byte counts of "reshard + checkpointing" are identical.
- `FSDPLite.__init__`: replaces every submodule selected by `is_block` and the leftover root parameters by an `FSDPUnit`; tied parameters (as in `d2l.GPT`) stay in the root unit.
- `gather_full_state_dict()`: all-gathers each unit and unflattens (test oracle).
- Numerical tests (4 gloo ranks, float64, seeds 0–2, `d2l.GPT(vocab_size=100, num_hiddens=64, num_heads=4, num_blks=4)`, AdamW with weight decay 0 because `FSDPLite` decays the flat shard and the reference must use one parameter group): (1) losses after 5 steps equal the single-process reference to $10^{-12}$ relative; (2) every full parameter after 5 steps equals the reference to $10^{-12}$; (3) `CommCounter` bytes per step equal `fsdp_costs(...)['comm']` exactly, padding included; (4) with `reshard_after_forward=False` one all-gather per unit is removed and the bytes equal the $6P$ formula; (5) peak persistent state bytes on each rank (sum of `numel × element_size` of shards, gradients and optimizer states after a step) equal $16P_{\text{padded}}/k$ in fp32.

#### In production

- `fully_shard(block, mesh=mesh["dp"], reshard_after_forward=..., mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32))` for each block, then `fully_shard(model)` for the root; explicit prefetch lists for the headline configs; `CPUOffloadPolicy(pin_memory=True)` for the offload run; initialization on the meta device then `to_empty` and re-initialization. Distributed checkpoint calls belong to 6.8.
- `Qwen3ForCausalLM.from_config(cfg, attn_implementation='sdpa', torch_dtype=torch.float32)` on `torch.device('meta')`; the block class is `Qwen3DecoderLayer`; `loss` through HF's `labels=` argument for the baseline, and a chunked cross-entropy (loop over 2k-token chunks with `torch.utils.checkpoint`) as the fix for the logits term.
- TorchTitan's FSDP2 integration (`torchtitan/distributed/FSDP.md`) applies the same calls per block; compare its defaults with the sweep winner.
- The from-scratch `FSDPLite` is the oracle for (i) loss and parameters after a step, (ii) per-step bytes of the collectives. On the T3 node, the same tiny model (hidden 512, 8 blocks) is run under `FSDPLite` and `fully_shard` on 8 GPUs and their parameters after 5 steps agree to BF16 round-off bands (to be measured).

#### Headline experiment

- **Hypothesis.** (H1) The term-by-term model predicts peak allocated memory within ±10% for every configuration. (H2) Exposed communication is below 5% of step time iff tokens per GPU exceed the $T^*$ of D2 (within a factor of 1.5). (H3) Fewer, larger groups and `reshard_after_forward=False` raise throughput only until the memory model says they no longer fit (on 40 GB parts the "no longer fit" side is the recorded out-of-memory at full depth and the throughput side is measured at the truncated depth).
- **Setup.** Qwen3-8B shape, random init, sequence 4096, 2 sequences per GPU per microbatch (8192 tokens) unless swept, full activation checkpointing except one run, AdamW (fused), learning rate $10^{-4}$ (irrelevant to the metrics), 30 steps with 10 warm-up, 8× A100 40 GB, one process per GPU via `d2l.launch`. Seeds fixed (0); data are synthetic tokens (so timing is not data-dependent). **Memory on 40 GB.** Every run uses chunked cross-entropy (2k-token chunks): unchunked, the logits term is $T V$ = 1.245e9 elements at 8,192 tokens, 12.4 GB at 2 + 4 + 4 bytes per element (D1 step 6, computed). W1's additive D1 total is then 35.7 GB (16.4 state + 2.7 transient layers + 10.0 root group + 3.6 activations + 3.1 chunked logits, each computed from the stated terms; the terms are not simultaneous, so the measured peak should be lower) against the 36 GB budget, and 45.0 GB with the unchunked loss. If W1 does not fit at 8,192 tokens per GPU, the whole table runs at 4,096.
- **Procedure.** Configs W1–W11 (table `tab_lm_fsdp_configs`): (W1) per-block, reshard, (bf16, fp32); (W2) as W1 without resharding; (W3) W1 with explicit prefetch depth 2 forward and backward; (W4) W1 with embeddings and head as separate groups; (W5) 2 blocks per group; (W6) 4 blocks per group; (W7) one group for the whole model; (W8) (bf16, bf16); (W9) (fp32, fp32); (W10) CPU offload; (W11) no activation checkpointing (predicted to run out of memory). (W12) token sweep for W1 at $\{\tfrac14, \tfrac12, 1, 2, 4\}\times T^*$ (D2 check). **Which configurations do not fit.** At full depth the additive bound predicts 50.0 GB for W2 (+14.3 GB of unsharded bf16 weights), 38.4 GB for W5, 43.8 GB for W6 and 8P = 65.5 GB for W7's whole-model group alone (bf16 weights, bf16 gradients and the fp32 reduce buffer live at once), all above 36 GB. Each runs once at full depth to record the predicted out-of-memory (a falsifiable prediction of H1), and its throughput comparison runs at the deepest truncation of the same width that `fsdp_costs` predicts to fit (18 layers, $P$ = 4.72 B, for W2, W5 and W6; fewer for W7), with W1 re-run at that depth as the paired baseline (at most two extra W1 runs). W11 is predicted to run out of memory at either depth and is run once to record it; W9 sits at the limit and joins the truncated group if its printed total exceeds 36 GB. Each run records per-step time, `max_memory_allocated/reserved`, one profiler window of 3 steps, collective bytes through `comm_events`, MFU.
- **Metrics.** Step time (median, IQR), MFU (convention below), peak memory per term, exposed fraction, bytes per step versus formula.
- **MFU convention.** $\text{FLOPs/token} = 6N_{\text{mm}} + 12\,L\,h\,d_h\,s$ with $N_{\text{mm}}$ the matmul parameters including the head (embedding lookup excluded) and the attention term following the PaLM decomposition (formula as commonly quoted; check Appendix B of 2204.02311 at authoring); the table prints the convention. It is expressed as `d2l.flops_per_token(shape, T=s, causal_half=False)` (full attention term at sequence length $s$, no causal halving; the causal-halved variant is `causal_half=True` and is reported only as a side column), and MFU as `d2l.mfu(tokens_per_s, shape, peak_flops, n_gpus=k)`. Peak: the A100 BF16 dense figure (`p16_tflops` column) in `d2l.chip_specs()` (datasheet, nvidia.com/en-us/data-center/a100).
- **Plots.** `fig_lm_fsdp_mem`: stacked bars of predicted terms beside the measured peak, one cluster per config. `fig_lm_fsdp_tstar`: exposed-communication fraction (y) against tokens per GPU (x, log), measured points with the predicted curve and a vertical line at $T^*$. `fig_lm_fsdp_trace`: one block's compute and NCCL rows from W1 and W7.
- **Expected outcome.** Direction: W2 faster than W1 and with more memory; W7 uses more transient memory; W9 sends twice the all-gather bytes of W1; W10 slower than W1; W11 runs out of memory. Magnitudes are "to be measured". Llama 3 reports FSDP with FP32 reduce-scatter as its data-parallel dimension (2407.21783, 3.3.2).
- **Noise check.** Step times of the 20 post-warm-up steps; W1 repeated in two launches; every variant-versus-baseline claim uses `d2l.paired_bootstrap` on per-step times (paired by step index) and counts only if the 95% interval excludes zero and the median differs by more than the W1 repeat difference.
- **T1 reduction.** Same code with `D2L_BACKEND=gloo`, a 4-block hidden-64 model, float64: only the tests of "From scratch" and `fsdp_costs` bytes run; MFU and traces are not produced.
- **Committed (T3).** `lm-launch-fsdp-W1..W12` bundles, memory snapshots (one pickle per config, ≤ 5 MB), profiler summaries, `exposed_comm` JSON.

#### Figures and tables

`fig_lm_fsdp_lifecycle_timeline` (the book's lifecycle figure redrawn with measured intervals), `fig_lm_fsdp_mem`, `fig_lm_fsdp_tstar`, `fig_lm_fsdp_trace`, `tab_lm_fsdp_mem` (term-by-term prediction), `tab_lm_fsdp_comm` (bytes per policy), `tab_lm_fsdp_configs` (configs W1–W12 with measured step time, memory, MFU), `tab_lm_fsdp_prediction_vs_measurement`.

#### Pitfalls and misconceptions

- "16P/k is the memory": it omits transient layers, the root group, logits and memory outside the allocator.
- Wrapping only the root `fully_shard(model)` gives one giant group (no overlap).
- Reading `max_memory_allocated` as device usage (NCCL buffers and the CUDA context are outside it).
- Calling `model.to_empty()` and forgetting to re-initialize.
- `reshard_after_forward=False` on every block defeats sharding in forward.
- Gradient accumulation with `set_requires_gradient_sync(False)` keeps full-size unsharded gradients (exercise 4).
- Comparing step times without a paired comparison.
- Weight decay applied to norm weights and biases inside a flat shard.

#### Summary

- FSDP2 stores $16P/k$ bytes of state per GPU but needs several times that at the peak: transient unsharded blocks, the root group, logits and non-allocator memory.
- Per step it moves $8P(k-1)/k$ bytes per GPU in (bf16, fp32) with resharding, $6P(k-1)/k$ without; DDP moves $4P(k-1)/k$ in bf16.
- Overlap requires about $\eta F(k-1)/(k\beta)$ tokens per GPU, independent of model size; below it, communication is exposed.
- Granularity, prefetching and resharding trade latency, overlap and transient memory; each is predicted by the model and measured by the trace.
- The from-scratch version is exact against one process in float64, which makes every later composed-parallelism claim testable.

#### Exercises

1. *(check)* For Qwen3-8B at $k=8$, how much extra memory does `reshard_after_forward=False` cost, and which bytes does it save? **Sketch:** it keeps all bf16 weights unsharded after forward: $2P(1-1/k) = 14.3$ GB extra; removes the backward all-gather, $2P(k-1)/k$ of the $8P(k-1)/k$ bytes (−25%).
2. *(check)* Why does FSDP2 default the root module to no resharding? **Sketch:** the root holds the head, needed immediately when backward starts; a reshard would add an all-gather that nothing can overlap.
3. *(derive)* Derive the communication and the extra memory of gradient accumulation over $a$ microbatches in two ways: reduce-scatter every microbatch, or `set_requires_gradient_sync(False)` until the last. **Sketch:** first: $a$ reduce-scatters and $a$ (or $2a$) all-gathers per optimizer step, sharded gradient only; second: all-gathers still $a$–$2a$, one reduce-scatter, but an unsharded gradient of $2P$ (bf16) must be kept, i.e. $2P(1-1/k)$ more bytes.
4. *(derive)* Show that with `reduce_dtype=bf16` the backward threshold equals the forward one. **Sketch:** backward link bytes $(2+2)P_L$ over compute $4P_LT$: ratio equals forward's $2P_L$ over $2P_LT$.
5. *(measure)* Before running, predict the peak memory of W5 (2 blocks per group) and W7 (one group) from D1; then measure. Which term mispredicts? **Hint:** the whole-model group keeps all of $2P$ bf16 unsharded.
6. *(measure)* At 4096 tokens per GPU, vary `set_modules_to_backward_prefetch` depth from 0 to 3. Predict the exposed fraction and the extra memory per extra layer ($2P_L$ = 386 MB per layer for Qwen3-8B); report where more prefetch stops paying.
7. *(open)* Implement HSDP in `FSDPLite` with a 2D mesh and decide, with the cost model of 6.7, for which tokens per GPU and node count it beats flat FSDP across two nodes.

#### Further reading

- Primary: [ZeRO](https://arxiv.org/abs/1910.02054) (state-sharding taxonomy; memory per stage); [PyTorch FSDP](https://arxiv.org/abs/2304.11277) (system design, prefetching, mixed precision); [TorchTitan](https://arxiv.org/abs/2410.06511) (FSDP2 within a composed stack); Llama 3 herd of models, section 3.3 (FP32 reduce-scatter; [2407.21783](https://arxiv.org/abs/2407.21783)).
- Reference and tutorials: [fully_shard API](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html); new, 200 OK on 2026-10-28: [FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html), [PyTorch CUDA memory snapshot docs](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html), TorchTitan's [`distributed/FSDP.md`](https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/FSDP.md) and [`REMAT.md`](https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/REMAT.md).
- Teaching: the Ultra-Scale Playbook's ZeRO and "memory usage" chapters; Scaling Book's training chapter.
- Code: [torchtitan](https://github.com/pytorch/torchtitan); [PyTorch `fully_shard` tests](https://github.com/pytorch/pytorch) (the `test/distributed/_composable/fsdp` directory) as executable semantics.

#### Risks and open decisions

- **`fully_shard` on CPU/gloo** may be unsupported in the build image. Fallback: T1 runs `FSDPLite` only and replays the `fully_shard` bundle.
- **HF meta-device initialization** differs across `transformers` versions (`_init_weights`, tied weights). Fallback: build the model as 1.2's `d2l.Qwen3GPT(**d2l.qwen3_gpt_kwargs(cfg))` on the meta device, which carries GQA, QK-norm, `head_dim` and untied heads.
- **Logits memory** may dominate and mask the sharding effects; the notebook reports it as its own term (predicted 12.4 GB unchunked and 3.1 GB chunked at 8,192 tokens, checked against the memory snapshot of the chunked run; no unchunked run is made on 40 GB) and uses chunked cross-entropy in every run, which on 40 GB is not optional.
- **No headroom on 40 GB.** W1's additive total is 35.7 GB against 36 GB; the fallback is 4,096 tokens per GPU for the whole table.
- **Allocator fragmentation** (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) changes `reserved` but not `allocated`; both are reported.
- **A100 caveats:** no FP8, no FlashAttention-3; SDPA's flash backend (FA2) is used. Anything FP8 in TorchTitan is off.

### 6.3 Tensor and sequence parallelism

- **Label:** `sec_lm_tensor_parallel` · **Tier:** T3 (reduced T1 config: 2- and 4-way TP on gloo CPU ranks, float64, `d2l.TransformerBlock` with hidden 64, 4 heads, 2 KV heads, sequence 32; timing cells skipped) · **Budget:** T3 ≤ 1 h wall on 8× A100 (≈ 8 GPU-h, planned: 4 TP degrees × 2 regimes × ~4 min plus the DTensor run); peak memory per GPU is predicted by `tp_costs`; T1 ≈ 3 min · **Codex units:** 6.5 (tensor parallelism), 6.6 (differentiation through distributed computation)
- **Builds on (book):** `sec_multi_gpu_concise` exercise "Tensor versus data parallelism" (the column/row split of the MLP, the sum as an all-reduce, the TP-vs-DP volume count, `parallelize_module` with `ColwiseParallel`/`RowwiseParallel`, `CommDebugMode`): 6.3 recaps its result in one paragraph and goes further, to attention, the conjugate operators as a derived rule with gradients, sequence parallelism and the whole-block cost; `sec_transformer-block` (`TransformerBlock`, `FeedForward` SwiGLU with `W_g, W_1, W_2`, `attn_factory`/`ffn_factory`); `sec_kv-cache` (`GQAAttention` with separate `W_q, W_k, W_v, W_o`; GQA head grouping); `sec_custom_layer` (`torch.autograd.Function`); `sec_memory_precision` (activation memory per layer).
- **Builds on (this part):** 6.1 (`d2l.launch`, `init_dist`, `CommCounter`, `comm_events`; the 2D mesh is built with `init_device_mesh` directly); 6.2 (`ModelShape`, `fsdp_costs`, `exposed_comm`, `saved_bytes`, FSDP2 over the data-parallel axis; 4.2's `d2l.mfu`); 5.4 (intra-node busbw and α), 5.5 (inter-node busbw and α for the cross-node prediction; `d2l.coll_time` on `lm-hw-a100-microbench`, as in 6.1); `d2l.chip_specs()`.
- **Exports:**
  - `d2l.copy_to_tp(x, group)` (the operator $f$: identity forward, all-reduce backward); `d2l.reduce_from_tp(x, group)` (the operator $g$: all-reduce forward, identity backward).
  - `d2l.ColumnParallelLinear(in_features, out_features, group, bias=False, weight=None)` and `d2l.RowParallelLinear(in_features, out_features, group, bias=False, weight=None)`: when `weight` (the full weight) is given the layer slices its shard from it, which is how the tests copy a reference layer.
  - `d2l.TPBlock.from_reference(block, group, sequence_parallel=False)`: builds a tensor-parallel copy of a `d2l.TransformerBlock` whose attention is `d2l.GQAAttention` (heads split; KV heads split when `num_kv_heads % t == 0`, replicated otherwise) and whose FFN is `d2l.FeedForward` (`W_g`, `W_1` column-split identically, `W_2` row-split); `.forward(x)` equals the reference; `.replicated_parameters()` lists parameters whose gradients need an all-reduce over the TP group.
  - `d2l.tp_costs(shape, t, *, dp=1, tokens, sp=True, ac='full', elem_bytes=2, async_tp=False) -> dict` (chapter convention; includes per-layer collective counts).
- **Models and data:** `Qwen/Qwen3-8B` shape (random init; `config.json`), as in 6.2; the DTensor run builds `Qwen3ForCausalLM` from config on the meta device. No datasets.
- **Packages:** `torch==2.15.0` (`torch.distributed.tensor`, `torch.distributed.tensor.parallel`: `parallelize_module`, `ColwiseParallel`, `RowwiseParallel`, `SequenceParallel`, `PrepareModuleInput`, `loss_parallel`; names verified in the 2.15 docs), `transformers`.

#### Narrative arc

1. **The question.** Opening cell (T3, committed): Qwen3-8B under FSDP2 with 16 sequences of 4096 tokens per step on 8 GPUs gives each GPU 8192 tokens, the 6.2 baseline. Cut the global batch to 4 sequences (`B_global` = 16384 tokens) and the exposed fraction of the step rises because tokens per GPU fall below 6.2's $T^*$; or ask for 32k-token sequences and activations no longer fit. The printed symptom is the measured exposed-communication fraction against tokens per GPU, with a vertical line at 6.2's $T^*$. The question: how can eight GPUs share one batch of tokens without each replica needing a large batch?
2. **Recap in one paragraph** of the book exercise: column-split $\mathbf W_1$, row-split $\mathbf W_2$, partial sums, one all-reduce.
3. **Linear layers split two ways.** $Y = XA$ with $A$ split by columns (no communication, outputs concatenate, or stay sharded) or by rows (inputs sharded, outputs summed). Figure `fig_lm_tp_linear` with shapes. Cell: both splits equal the full product in float64.
4. **The MLP and the attention block.** Gated MLP: `W_g` and `W_1` must be split identically (the product is elementwise); `W_2` row-split. Attention: heads are independent, so $W_q, W_k, W_v$ split by heads (column), $W_o$ by the same heads (row). GQA: each rank needs the KV heads of its query groups; $t \le h_{kv}$ splits them, $t > h_{kv}$ replicates (Qwen3-8B, $h_{kv}=8$: no replication up to $t=8$; Qwen3-30B-A3B, $h_{kv}=4$: replicated 2× at $t=8$). Figure `fig_lm_tp_block`: the block with its two all-reduces.
5. **The conjugate operators** (D1). $f$ and $g$ with their Jacobians; the rule for replicated parameters (D2). Cells: `gradcheck`, then `TPBlock` versus reference for outputs and gradients.
6. **Sequence parallelism.** Norms and residual adds are token-wise and replicated under plain TP; shard them along the sequence, replace each all-reduce by reduce-scatter plus all-gather (same bytes, D3), and divide the activations those layers store by $t$. Cell: `saved_bytes` of one block under TP and under TP+SP against the formulas of Korthikanti et al. (D4).
7. **DTensor placements.** `Shard(dim)`, `Replicate()`, `Partial()`; table `tab_lm_dtensor_transitions` (redistribution → collective): Shard→Replicate = all-gather; Partial→Replicate = all-reduce; Partial→Shard = reduce-scatter; Replicate→Shard = local slice (free); Shard(i)→Shard(j) = all-to-all. Cell: the 2-layer `Sequential` of the book exercise through `parallelize_module`, then the same for a full block with `SequenceParallel` and `PrepareModuleInput`, counting collectives with `CommDebugMode` against the formulas.
8. **The vocabulary dimension.** Vocab-parallel embedding (masked lookup, all-reduce or reduce-scatter) and vocab-parallel cross-entropy (`loss_parallel`: logit shard of width $V/t$; two $T$-sized all-reduces for the max and the normalizer; the target logit by masked sum). Why it matters: the $T \times V$ logits of 6.2's memory model shrink by $t$. Cell (T0): vocab-parallel loss equals the full loss in float64 for $V = 1000$, $t = 4$.
9. **Why TP stays inside the NVLink domain** (D3 evaluated). Evaluate the comm-to-compute ratio $\rho_{\text{TP}}$ with Chapter 5's intra-node busbw and with its inter-node per-GPU busbw. Table `tab_lm_tp_domain`: $t \in \{2, 4, 8, 16\}$, link, predicted $\rho$, predicted step-time inflation. The intra-node rows are the NVSwitch row of 5.4's fits and, alongside, the PCIe-pair row (on a node of NVLink-bridged PCIe cards only $t=2$ stays on the bridge, so "the domain" is 2 and $t \ge 4$ pays the PCIe $\rho$); `topo.txt` decides which row is the measured node. Also the latency argument: $4L$ dependent collectives per microbatch cannot be hidden behind unrelated work.
10. **Asynchronous TP** (named, measured only if the pinned PyTorch supports it): chunked matmul plus collective overlap through symmetric memory under `torch.compile`; `CUDA_DEVICE_MAX_CONNECTIONS` and its effect on overlap. Reported as a configuration of the TorchTitan run, not implemented here.
11. **TP and the data axis** (D5): TP lets a fixed GPU count run with fewer data-parallel replicas, hence a smaller global batch for the same overlap; connects to `sec_batch_size` and to 6.7.

#### Derivations

**D1. The conjugate operators $f$ and $g$** (`eq_lm_tp_fg`).
- *Statement.* For $Z = \phi(X A)B$ with $A = [A_1, \dots, A_t]$ split by columns, $B = [B_1; \dots; B_t]$ split by rows, and an elementwise $\phi$, rank $i$ computes $Z_i = \phi(XA_i)B_i$ and $Z = \sum_i Z_i$. If $X$ is replicated, then the correct forward is $Z = g\big(\sum\text{local}\big)$ with $g$ = all-reduce, and the correct backward for $X$ requires $f$ = identity forward and **sum over ranks** backward.
- *Assumptions.* $X$ identical on all ranks; scalar loss $\mathcal L$ computed identically on all ranks from the replicated $Z$.
- *Steps.* (1) Forward: $Z_i = \phi(XA_i)B_i$; $Z = \sum_i Z_i$ is an all-reduce (sum). (2) Backward of $g$: $\partial\mathcal L/\partial Z_i = \partial\mathcal L/\partial Z$ for every $i$ (derivative of a sum): identity, because $\partial\mathcal L/\partial Z$ is already replicated. (3) Local gradients: $\partial\mathcal L/\partial B_i = (\phi(XA_i))^\top G$ and $\partial\mathcal L/\partial A_i = X^\top\big[(G B_i^\top)\odot\phi'(XA_i)\big]$, with $G = \partial\mathcal L/\partial Z$; no communication. (4) Input gradient: $X$ feeds every branch, so $\partial\mathcal L/\partial X = \sum_i\big[(G B_i^\top)\odot\phi'(XA_i)\big]A_i^\top$, a sum over ranks: backward of $f$ is an all-reduce. (5) Hence $f$ and $g$ are a conjugate pair: each is the other's backward. (6) If both $A$ and $B$ were column-split, the intermediate $\phi(XA)$ would have to be gathered ($4d$ per token instead of $d$), and the sum over $i$ in the forward disappears; the pair above is the unique split with one collective.
- *Result.* One all-reduce of $b s d$ elements in forward ($g$) and one in backward ($f$) per sublayer.
- *Check.* `gradcheck` on `copy_to_tp`/`reduce_from_tp` (float64, 2 gloo ranks); `TPBlock` versus the reference in outputs, every parameter gradient and the input gradient (tolerances of the assignment); counter-example cell: delete $f$'s backward all-reduce and show the input gradient wrong by exactly the missing off-rank terms.

**D2. Gradients of replicated parameters.**
- *Statement.* A parameter replicated on all TP ranks needs a gradient all-reduce over the TP group iff each rank's loss depends on it through a different part of the computation; it does not when every rank computes the identical full function of it.
- *Steps.* (1) The true gradient is the sum of the contributions of all uses of the parameter. (2) Under plain TP, RMSNorm weights act on the replicated full-sequence activations: each rank computes the same gradient: no all-reduce (and none must be added, or the gradient is $t\times$ too large). (3) Under sequence parallelism each rank normalizes its $s/t$ tokens: it holds a partial sum over tokens: sum over the TP group. (4) The same holds for QK-norm weights (shared across heads, each rank sees its heads) and for replicated KV projections (each replica sees its own query heads). (5) Biases of row-parallel layers are added once after the reduction.
- *Check.* Cells list `TPBlock.replicated_parameters()` with the rule for each; the test compares gradients with and without the extra all-reduce.

**D3. Communication volume and the comm-to-compute ratio** (`eq_lm_tp_comm`, `eq_lm_tp_ratio`).
- *Statement.* Per layer and microbatch (no activation recomputation), each GPU moves $8\,\frac{t-1}{t}\,b\,s\,d\,e$ wire bytes ($=16\frac{t-1}{t}bsd$ for $e=2$, BF16), with $e$ the element size, the same with or without sequence parallelism. Per GPU, a layer's compute is $6\,b s\,P_L/t$ FLOPs. The ratio $\rho_{\text{TP}} = t_{\text{comm}}/t_{\text{compute}}$ with unoverlapped collectives is $\rho = \frac{4}{3}\,(t-1)\,\frac{d\,e\,\eta F}{P_L\,\beta}$ (for $e=2$: $\frac{8}{3}(t-1)\frac{d\,\eta F}{P_L\beta}$), independent of $b$ and $s$.
- *Steps.* (1) Two sublayers, each with one all-reduce forward ($g$) and one backward ($f$): 4 all-reduces of $b s d$ elements. (2) Ring all-reduce wire bytes $\frac{2(t-1)}{t}$ times the tensor bytes $b s d e$ (`eq_ring_traffic`): $4 \cdot \frac{2(t-1)}{t}\,bsde = 8\frac{t-1}{t}bsde$ — with $e$ bytes this equals the $16(t-1)/t\cdot bsd$ above for $e = 2$. (3) With sequence parallelism each all-reduce becomes a reduce-scatter and an all-gather of $(t-1)/t$ of the tensor each: the same total. (4) With full activation recomputation the forward collectives run twice: 6 per layer instead of 4. (5) Compute per GPU: $6bsP_L/t$ (forward plus backward, ignoring the attention-score term). (6) Divide: $\rho = \frac{8\frac{t-1}{t}bsde/\beta}{6bsP_L/(t\eta F)} = \frac{4(t-1)\,d\,e\,\eta F}{3 P_L\beta}$; $b$ and $s$ cancel.
- *Result.* TP efficiency is set by $(t-1)$, $d/P_L \approx 1/(12 d)$ for MHA, and the ratio $F/\beta$ of the link it runs on. Cross-node, $\beta$ falls by the ratio of Chapter 5's intra- and inter-node busbw, and $\rho$ rises by the same factor.
- *Check.* Cell A: `CommCounter` bytes of a float64 `TPBlock` forward+backward equal $8\frac{t-1}{t}\,b s d\cdot 8$ exactly. Cell B (T3): `comm_events` bytes per layer for the 8B run equal the formula within 1%, and the measured exposed fraction at two different token counts agree within the repeat spread, as the independence of $b, s$ predicts. Cell C: evaluate $\rho$ with Chapter 5's numbers for $t=2,4,8$ (intra: NVSwitch row, and the PCIe-pair row alongside) and $t=16$ (inter).

**D4. Activation memory under TP and SP** (`eq_lm_tp_act`).
- *Statement.* For the GPT-style block with the attention-score terms removed (fused attention) and no dropout, the per-layer activation bytes are $s b h\,(34/t)$ with TP+SP and $s b h\,(10 + 24/t)$ with TP only, against $34\,sbh$ without parallelism (Korthikanti et al., Eq. 1, 2, 4, at 16-bit).
- *Steps.* (1) Itemize every tensor the block saves for backward (block input, the two norm inputs, Q/K/V, attention output, output-projection input, MLP input, the two hidden activations and the down-projection input; two dropout masks in the original). (2) Mark each as sharded by $t$ under TP (anything inside a column/row-parallel matmul or an attention head) or replicated (norm inputs, residual inputs, dropout masks, the layer input and the row-parallel output). (3) Sum: the replicated part contributes the constant 10, the sharded part $24/t$; under SP the replicated tensors are also sharded along the sequence. (4) For the actual Qwen3 block (SwiGLU 12288, GQA 32/8, QK-norm, no dropout), redo the itemization and obtain a different constant; `d2l.saved_bytes` measures it.
- *Check.* Cell: the itemized prediction for Qwen3's block (written out as a table with one row per saved tensor) versus `saved_bytes` of the TP and TP+SP blocks, equal to within the dropout-mask and norm-statistics terms; the cell reports the residual by tensor.

**D5. Why TP buys a smaller global batch.**
- *Statement.* With $N$ GPUs, TP degree $t$ and FSDP over $N/t$ replicas, the FSDP overlap condition of 6.2 holds per replica with $T_{\text{replica}} \ge T^*$ tokens, so the minimum global batch for an overlapped step is $B_{\min} = (N/t)\,T^*$ tokens.
- *Steps.* (1) A TP group holds one copy of the model sharded $t$ ways; its members process the same tokens. (2) Per GPU, FSDP all-gathers $P_L/t$ per layer over the replica group and computes $2(P_L/t)T_{\text{replica}}$: the threshold of 6.2 applies with tokens $T_{\text{replica}}$ unchanged. (3) The number of replicas is $N/t$, so the global batch is $(N/t)T_{\text{replica}}$.
- *Result.* If the critical batch size of `sec_batch_size` caps the useful global batch at $B_{\text{crit}}$, then data parallelism alone supports $N \le B_{\text{crit}}/T^*$ GPUs, and TP raises that limit by $t$ at the price of $\rho_{\text{TP}}$. Evaluate with 6.2's $T^*$.
- *Check.* Table of $B_{\min}$ for $N = 8, 64, 512$ and $t = 1, 2, 4, 8$.

#### From-scratch implementation

≤ 150 lines in total.

- `_CopyToTP`, `_ReduceFromTP` (`torch.autograd.Function`, 12 lines each); `copy_to_tp`, `reduce_from_tp`.
- `_GatherSeq` and `_ScatterSeq` (sequence-parallel $f$ and $g$: forward all-gather/backward reduce-scatter and the reverse, along dim 1; 25 lines).
- `ColumnParallelLinear`, `RowParallelLinear` (20 lines each); weight slicing helper `shard(w, dim, rank, t)`.
- `TPBlock` (≤ 60 lines): `from_reference` slices the reference block's `GQAAttention` (`W_q`, `W_k`, `W_v` rows by head; `W_o` columns) and `FeedForward` (`W_g`, `W_1` rows; `W_2` columns); the RoPE helper of `GQAAttention` runs unchanged on local heads; norms are copied; `forward` inserts `copy_to_tp`/`reduce_from_tp` (or `_GatherSeq`/`_ScatterSeq` for SP).
- Tests (2 and 4 gloo ranks, float64, seeds 0–2, hidden 64, 8 heads, 4 KV heads (and 2 KV heads at $t=4$ to exercise replication), sequence 32, batch 2): output, every parameter gradient (all-gathered to the reference layout; replicated KV gradients all-reduced over the replica group) and input gradient within $10^{-9}\max(1,\max\lvert\text{ref}\rvert)$; collective bytes equal D3 exactly; the SP variant matches the same references; replicated-parameter gradients are bitwise identical across ranks after the TP all-reduce.

#### In production

- `parallelize_module(block, tp_mesh, plan)` with `ColwiseParallel()` for `q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`; `RowwiseParallel(output_layouts=Shard(1))` for `o_proj`, `down_proj` (sequence-parallel output); `SequenceParallel()` for the norms; `PrepareModuleInput(input_layouts=Shard(1), desired_input_layouts=Replicate())` at attention and MLP inputs; `RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1))` for `embed_tokens`; `ColwiseParallel(input_layouts=Shard(1), output_layouts=Shard(-1), use_local_output=False)` for `lm_head` with `loss_parallel()`. Module names are those of `Qwen3ForCausalLM`; inspect `Qwen3Config.base_model_tp_plan` at authoring and align.
- FSDP2 on the data-parallel axis of a 2D `init_device_mesh("cuda", (dp, tp), mesh_dim_names=("dp", "tp"))`; `fully_shard(block, mesh=mesh["dp"])` after `parallelize_module`.
- Library options: Megatron-Core `--tensor-model-parallel-size`, `--sequence-parallel`; TorchTitan `tensor_parallel_degree` (config attribute, verified in its config guide) with async TP; each is described, not run.
- `TPBlock` is the oracle: the DTensor-parallelized block must reproduce its outputs and gradients (BF16 band to be measured on first run) and its per-layer collective counts (`CommDebugMode`: expect two reduce-scatter and two all-gather per sublayer in forward under SP, equal to the formulas).

#### Headline experiment

- **Hypothesis.** (H1) The collective bytes per layer equal D3 within 1% for $t = 2, 4, 8$. (H2) The exposed TP communication fraction is independent of tokens per GPU (D3), unlike FSDP's (6.2). (H3) The predicted cross-node ratio at $t=16$ exceeds 1, so a cross-node TP group is predicted slower than any in-node choice; the prediction is committed now and tested by 6.7's T4 run.
- **Setup.** Qwen3-8B shape, random init, BF16, SDPA (FA2), full activation checkpointing, FSDP2 over the data axis (6.2's W1 settings, chunked loss), 8× A100 (40 GB per GPU assumed). $t \in \{1, 2, 4, 8\}$. Two regimes: (A) fixed global batch 16 sequences of 4096 tokens (65,536 tokens, a per-node setting); (B) fixed 8192 tokens per replica. At $t=1$ both regimes are 6.2's W1 (16 sequences over 8 replicas is 8,192 tokens per replica), so that point is reused from 6.2's bundle and not re-run (cut: two runs), with 6.2's memory caveat (35.7 GB additive total against 36 GB); for $t \ge 2$ the per-GPU state falls as $1/t$ and the plan has headroom. 30 steps with 10 warm-up; seed 0.
- **Procedure.** Per launch record step time, `comm_events` split by group size into TP and FSDP collectives, `exposed_comm` per group, peak memory, MFU; two launches of $t=4$ (A) for repeat spread. A separate 3-min run at $t=4$ with 4096 and 16384 tokens per replica tests H2.
- **Metrics.** Bytes per layer (measured vs formula), $\rho$ (measured vs predicted), step time, MFU, peak memory vs `tp_costs`.
- **Plots.** `fig_lm_tp_bytes`: bytes per layer per $t$, measured vs predicted. `fig_lm_tp_mfu`: MFU and stacked exposed time (TP, FSDP, compute) against $t$ for regimes A and B. `fig_lm_tp_ratio`: exposed TP fraction vs tokens per replica for $t=4$ (flat predicted) next to 6.2's FSDP curve (falling predicted). `tab_lm_tp_domain` with the cross-node prediction.
- **Expected outcome.** Direction: FSDP's exposed time falls with $t$, TP's rises with $t$; MFU in regime A has a maximum at an intermediate $t$ or at $t=1$ depending on Chapter 5's $\beta$ (to be measured). Bytes equal formula. The only quoted external reference is Llama 3 (TP=8 inside the node in every row of its Table 4 at 8,192–16,384 GPUs, 38–43% BF16 MFU, 2407.21783), used for orientation in `tab_lm_tp_domain`, not as a target.
- **Noise check.** Repeat spread from the two $t=4$ launches; step times as median/IQR over 20 steps; a trend across $t$ is claimed only if it exceeds the repeat spread; $\rho$ at two token counts compared with a bootstrap interval from `d2l.paired_bootstrap`.
- **T1 reduction.** Float64 gloo: all correctness and byte-count tests; no timing.
- **Committed (T3).** Launch bundles `tp_A_t{2,4,8}`, `tp_B_t{2,4,8}` (the $t=1$ points are 6.2's W1 bundle), `tp_ratio_t4`; `saved_bytes` table; DTensor `CommDebugMode` output.

#### Figures and tables

`fig_lm_tp_linear` (column vs row split), `fig_lm_tp_block` (block with $f$, $g$), `fig_lm_tp_sp` (activations sharded vs replicated, before/after SP), `fig_lm_tp_bytes`, `fig_lm_tp_mfu`, `fig_lm_tp_ratio`; `tab_lm_dtensor_transitions`, `tab_lm_tp_domain`, `tab_lm_tp_act` (D4 itemization), `tab_lm_tp_prediction_vs_measurement`.

#### Pitfalls and misconceptions

- Adding an all-reduce for a replicated parameter whose gradient is already complete (D2): the gradient is $t\times$ too large.
- Forgetting the all-reduce under sequence parallelism.
- Splitting `W_g` and `W_1` differently.
- Vocabulary not divisible by $t$ (pad the vocabulary).
- Initializing each shard independently with the full-matrix standard deviation: the distribution of the whole matrix is right, but a run is no longer reproducible across $t$; initialize the full matrix from one seed and slice.
- Dropout in the TP region needs different random streams per rank, outside it identical ones (not used by Qwen3 pretraining; noted).
- TP across nodes "because the memory is needed": check $\rho$ first.
- Treating TP as a memory feature only: it also shrinks logits and per-GPU weight reads.

#### Summary

- A column-split followed by a row-split needs one all-reduce per sublayer; $f$ and $g$ are conjugates and are derived from the chain rule.
- Per layer, each GPU moves $16\frac{t-1}{t}bsd$ bytes in BF16, with or without sequence parallelism; the comm-to-compute ratio does not depend on batch or sequence length.
- Sequence parallelism makes the norms and residuals sharded; activation memory falls from $34sbh$ to $34sbh/t$ for the textbook block.
- Replicated parameters need an all-reduce of gradients exactly when each rank sees only part of their uses.
- TP stays inside the fastest domain because $\rho$ scales with $F/\beta$ and its collectives are on the critical path; it also lets a fixed GPU count run at a smaller global batch.

#### Exercises

1. *(check)* Why can the MLP not be split column–column without a collective between the matrices, and how many bytes would that collective move per token compared with the all-reduce of D1? **Sketch:** the second matmul needs the full intermediate of width $4d$ (or $8d/3$ for SwiGLU); an all-gather of the intermediate moves $\frac{t-1}{t}$ of $bs\cdot 4d$ elements, four times the $bsd$ of the output reduction.
2. *(check)* For Qwen3-8B at $t=16$ on two nodes, which parameters are replicated, and which collectives involve them? **Sketch:** $h_{kv}=8 < 16$: each KV head on two ranks; their gradients need an all-reduce over the 2-rank replica group; the norms under SP need an all-reduce over the TP group.
3. *(derive)* Derive $34\,sbh/t$ for TP+SP by itemizing the saved tensors of a GPT block, and show which two terms of the 34 disappear when dropout is off. **Sketch:** each dropout mask is $sbh$ bytes (1 byte per element); the remaining constants follow from the tensor list; check against `saved_bytes`.
4. *(derive)* Re-derive $\rho_{\text{TP}}$ for SwiGLU with GQA, using the actual $P_L$ of Qwen3-8B; at which $t$ does $\rho$ cross 0.25 for a link with $\beta = 100$ GB/s and $\eta F = 150$ TFLOP/s? **Sketch:** $\rho=\frac{4(t-1)d e\,\eta F}{3P_L\beta}$; solve for $t$ (answer to be computed in the notebook; it is an exercise, so the numbers are inputs).
5. *(measure)* Predict, then measure, $\rho$ at $t=4$ for 4096 and 16384 tokens per replica; explain any difference with exposed-communication and `NCCL` launch latency. **Hint:** small messages fall below $9\alpha\beta$.
6. *(measure)* Re-run the $t=4$ sweep with `NCCL_P2P_DISABLE=1` (forcing host-staged transfers) as an emulation of a slow link; predict the new $\beta$ from a bare all-reduce sweep and the resulting $\rho$; compare.
7. *(open)* Add asynchronous TP (chunked all-gather/matmul overlap) to `TPBlock`, and measure how much of the exposed TP time it hides on the node; relate the result to the D3 ratio.

#### Further reading

- Primary: [Megatron-LM](https://arxiv.org/abs/1909.08053) (the $f$/$g$ operators; MLP and attention splits); [Reducing Activation Recomputation](https://arxiv.org/abs/2205.05198) (sequence parallelism; Eq. 1–5 of the activation accounting, which the section checks); [Efficient Large-Scale Training](https://arxiv.org/abs/2104.04473) (TP inside the node, combination with PP and DP).
- Reference and tutorial: [PyTorch TP API](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html); [DTensor](https://docs.pytorch.org/docs/stable/distributed.tensor.html); new, 200 OK on 2026-10-28: [TP tutorial](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html).
- Teaching: Scaling Book (tensor parallelism and the "which axis on which link" analysis); Ultra-Scale Playbook (TP/SP sections and heat-maps).
- Code: [Megatron-LM `tensor_parallel` package](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core) (200 OK; the reference `ColumnParallelLinear`/`RowParallelLinear`); [torchtitan](https://github.com/pytorch/torchtitan) Llama and Qwen3 `parallelize` functions.

#### Risks and open decisions

- **HF module naming and head reshaping under local shapes** may break `parallelize_module` on `Qwen3ForCausalLM` at the pinned `transformers` version. Fallback: use TorchTitan's Qwen3 model and its parallelization for the T3 runs and keep HF only for shapes.
- **Async TP** needs symmetric-memory support that may not exist on A100 builds. Fallback: omit, state the exposed fraction as an upper bound.
- **`d2l.GQAAttention` has no QK-norm**, while Qwen3 does (replicated, shared across heads); the from-scratch block omits it and D2 covers the rule; the DTensor run includes it.
- **BF16 tolerance for T3** is set after a first run; the float64 CPU tolerance is not negotiable.
- **Cross-node TP prediction** depends on Chapter 5's inter-node busbw; if only a datasheet NIC rate is available, the prediction is flagged "datasheet-based" and the T4 run in 6.7 replaces it.

### 6.4 Pipeline parallelism

- **Label:** `sec_lm_pipeline_parallel` · **Tier:** T3 for the executed pipeline; the schedule simulator is T0 · reduced T1 config: 4 gloo ranks, sleep-calibrated stage costs (F = 20 ms, B = 40 ms) for the timing check, and a 4-layer float64 `d2l.GPT` for the numerical check · **Budget:** T3 ≤ 1 h wall on 4 of the 8 A100 (≈ 8 GPU-h, planned: 4 schedules × 3 microbatch counts × ~2 min plus calibration); peak memory per GPU predicted by `pp_costs`; simulator and T1 ≈ 4 min on CPU · **Codex units:** 7.1 (pipeline parallelism), 7.2 (schedules), 7.3 (advanced scheduling)
- **Builds on (book):** `sec_multi_gpu` exercise 1 (two-stage pipeline: bytes crossing the cut, the micro-batch bubble for two stages: recap in one paragraph, then generalize to $p$ stages and the other schedules), `fig_splitting`; `sec_memory_precision` (activation memory, checkpointing); `sec_gpt` (`d2l.GPT` blocks as stages); `sec_custom_layer` is not needed (stages use `torch.autograd.backward` with explicit output gradients).
- **Builds on (this part):** 6.1 (`d2l.launch`, `init_dist`, `read_rank_logs` for timelines, `CommCounter`); 6.2 (`ModelShape`, `saved_bytes`, `exposed_comm`); 6.3 (`tp_costs` for the TP-vs-PP communication comparison); 5.5 (inter-node α–β for the cross-node boundary cost); 4.1 (loss normalization by tokens for unequal microbatches, `d2l.PackedStream` is not required).
- **Exports:**
  - `d2l.pipeline_schedule(policy, p, m, v=1) -> list[list[tuple]]`: per-device ordered action lists `(kind, chunk, microbatch)` with `kind ∈ {'F','B','Bi','W'}`. Policies: `'gpipe'`, `'1f1b'`, `'interleaved'` (Narayanan et al.'s order; requires `m % p == 0`, `v ≥ 2`), `'zb_h1'` (backward split into `Bi` and `W`, warm-up cap $p-s$). Also `'flex'`: interleaved with a free group size $N$ as in Llama 3 (Fig. 6), requires only $m \ge 1$.
  - `d2l.PipelineSim(p, m, F=1.0, B=2.0, W=0.0, v=1, comm=0.0, stage_scale=None, act=1.0, keep=1.0)`; `.run(schedule) -> SimResult(makespan, bubble_frac, idle_per_device, peak_act (per device, in units of one stage's activation of one microbatch), msgs, timeline)`. `W>0` means the full backward `B` is split into `Bi = B − W` and `W`; `stage_scale` gives per-stage multipliers; `keep` is the fraction of a microbatch's activation still held between `Bi` and `W` (default 1, to be calibrated).
  - `d2l.plot_gantt(result, ax=None)`: draws a `SimResult.timeline` or a measured timeline with the same record format.
  - `d2l.PipelineRunner(stage_module, group, p, schedule_policy, microbatch_shape, dtype, device)`: executes `'gpipe'` and `'1f1b'` with `dist.isend/irecv`; `.run(batch, labels=None, loss_fn=None) -> (loss, timeline)`; writes per-action start/end stamps via `d2l.rank_log`.
  - `d2l.balance_stages(layer_cost, p, v=1, head_cost=0.0, embed_cost=0.0, trim=(0, 1)) -> list[int]`: layers per chunk; the Llama 3 heuristic of removing `trim` layers from the first and last chunks, with the layer counts chosen to minimize the predicted maximum device load.
  - `d2l.pp_costs(shape, p, *, v=1, m, tokens_per_microbatch, policy, elem_bytes=2, ac='full') -> dict`.
- **Models and data:** `Qwen/Qwen3-8B` shape, 36 layers, random init (as 6.2); synthetic tokens. The float64 test uses a stack of `d2l.TransformerBlock`.
- **Packages:** `torch==2.15.0`; `torch.distributed.pipelining` (`PipelineStage`, `ScheduleGPipe`, `Schedule1F1B`, `ScheduleInterleaved1F1B`, `ScheduleZBVZeroBubble`, `ScheduleDualPipeV`, `ScheduleLoopedBFS`; names verified in the 2.15 docs); `matplotlib`.

#### Narrative arc

1. **The question.** Opening cell: from 6.3, TP across 16 GPUs is predicted communication-bound; from 6.2, FSDP across 16 GPUs wants more tokens per GPU than a small batch gives. A pipeline sends one activation per microbatch per boundary. The cell prints, for Qwen3-8B, the bytes crossing one boundary per step ($2m\,b\,s\,d\,e$) next to TP's and FSDP's per-step bytes (`tp_costs`, `fsdp_costs`): about two orders of magnitude fewer than TP at $t = 8$ (the cell computes the ratio). Then the symptom: a measured 4-stage GPipe run with $m=4$ whose Gantt chart is mostly idle (committed T3 timeline), with the idle fraction printed.
2. **Recap of the two-stage case** (one paragraph, `sec_multi_gpu` exercise 1), then partition by depth: stages, microbatches, the activation and gradient crossing each boundary, the optimizer step after the last backward, loss normalization (`1/m`, or token-weighted for unequal microbatches).
3. **The schedule as data** (`d2l.pipeline_schedule`): per-device lists of actions; the same object drives the simulator and the executor. Figure `fig_lm_pp_schedules`: GPipe and 1F1B for $p=4$, $m=8$, drawn by `plot_gantt` from the simulator.
4. **The bubble** (D1). GPipe and 1F1B have the same bubble fraction $(p-1)/(m+p-1)$; 1F1B changes memory, not time. Cell: the simulator's makespan equals $(m+p-1)(F+B)$ exactly for unit costs.
5. **Activation memory per schedule** (D3). GPipe stores $m$ microbatches on every stage; 1F1B stores $p-s$ on stage $s$. Cell: the peak per stage from the simulator against the formula; `d2l.saved_bytes` of one stage to convert units to bytes.
6. **Interleaving** (D2). $v$ model chunks per device: bubble $\times 1/v$, communication $\times v$, first-stage activation memory $p + (p-1)/v$ stage-units for the Megatron order. Cell: simulated bubble equals $(p-1)(F+B)/v$ for $(p, m, v) \in \{(4,8,2), (4,8,3), (8,16,2)\}$. Llama 3's flexible group size $N$ (no `m % p == 0` constraint) and its fraction $(PP-1)/(V\cdot M)$ are cited (2407.21783, 3.3.2).
7. **Splitting the backward** (D4). $B = B_i + W$: input gradients on the critical path, weight gradients deferrable; ZB-H1 as a 1F1B variant that fills the drain with $W$; zero-bubble variants (ZB-H2, ZBV) and DualPipe(V) as named extensions measured only through `torch.distributed.pipelining` (`ScheduleZBVZeroBubble`, `ScheduleDualPipeV`). The DualPipe repository's table of bubbles (1F1B: $(PP-1)(F+B)$; ZB1P: $(PP-1)(F+B-2W)$; DualPipe: $(PP/2-1)(F\&B+B-3W)$) is reproduced by the simulator for the first two rows, as a check, and cited for the third.
8. **Partitioning and endpoint effects** (D5). Equal layer counts are not equal time: the head's matmul costs $2Vd$ FLOPs per token, 3.2 layers of Qwen3-8B; the embedding costs memory not time. `balance_stages`; Llama 3 removes one layer from each of the first and last stages.
9. **Boundary communication** (D6): bytes and the ratio $\rho_{\text{PP}}$ against $\rho_{\text{TP}}$ of 6.3; why PP is the axis that goes across nodes; interaction with FSDP (each microbatch re-gathers parameters unless resharding is off: the factor $m$ in the all-gather bytes of `fsdp_costs`).
10. **Executing it.** The from-scratch runner on 4 gloo ranks (float64) against the full-batch reference; then `torch.distributed.pipelining` on 4 A100 with the four policies. Measured timelines go through `plot_gantt`, next to the simulator's.

#### Derivations

**D1. Bubble of GPipe and 1F1B** (`eq_lm_pp_bubble`).
- *Statement.* With $p$ stages, $m$ microbatches, per-stage forward time $F$ and backward time $B$ per microbatch, equal across stages and no communication time, the makespan of GPipe and of 1F1B is $(m+p-1)(F+B)$, the idle time per device is $(p-1)(F+B)$, and the idle fraction is $(p-1)/(m+p-1)$.
- *Steps.* (1) Forward of microbatch $i$ on stage $s$ starts no earlier than $(s+i)F$ (it needs stage $s-1$'s output and the stage's own previous microbatch). (2) The last forward finishes at $(p+m-1)F$ on stage $p-1$. (3) In GPipe all forwards precede all backwards; backward of microbatch $i$ on stage $s$ needs stage $s+1$'s backward and mirrors the forward: the last backward finishes at $(p+m-1)(F+B)$. (4) Each device does $m(F+B)$ of work, so it idles $(p-1)(F+B)$. (5) 1F1B reorders each device's work (a warm-up of $p-s-1$ forwards, then alternating forward/backward, then a cool-down) without changing the dependency graph's longest path: the critical path still contains $p-1$ forward hops, $m$ forwards-and-backwards on the last stage, and $p-1$ backward hops, so the makespan is the same. (6) The ratio idle/ideal is $(p-1)/m$; the fraction idle/total is $(p-1)/(m+p-1)$.
- *Result.* Both forms are quoted; this chapter reports the fraction idle/total.
- *Check.* Simulator, unit costs: $(p,m) \in \{(4,8), (4,16), (8,16)\}$, makespans 33, 57, 69 (the cell recomputes them) for both policies; the CPU prototype used to design this blueprint gave exactly these. Stage-imbalance variation in exercise 6.

**D2. Interleaving** (`eq_lm_pp_interleave`).
- *Statement.* With $v$ chunks per device (chunk $c$ of device $s$ is global stage $cp+s$), each of time $F/v$, $B/v$, the bubble per device is $(p-1)(F+B)/v$, the idle fraction $(p-1)/(vm+p-1)$, the point-to-point message count per microbatch is $v$ times that of $v=1$, and the first device's peak live activations are $p + (p-1)/v$ stage-units for Megatron's order.
- *Steps.* (1) The fill path has $p-1$ hops per chunk traversal but each hop costs $F/v$. (2) The drain has $p-1$ backward hops of $B/v$. (3) The work per device is unchanged, $m(F+B)$. (4) Messages: each microbatch crosses $vp-1$ boundaries instead of $p-1$. (5) Memory: the warm-up of device $s$ is $2(p-s-1)+(v-1)p$ chunk-forwards (Megatron's order), so device 0 holds $(v+1)p-1$ chunk activations at its peak, each $1/v$ of a stage: $p + (p-1)/v$ stage-units.
- *Check.* The simulator with Megatron's order reproduces bubble $4.5$ for $(4,8,2)$ and $3.0$ for $(4,8,3)$ with $F=1$, $B=2$, and peaks $5.5$ and $5.0$ (recomputed in the cell). The cell prints that the $v=1$ case of this order holds $2p-1$ rather than 1F1B's $p$, so "interleaved with $v=1$" is not 1F1B.

**D3. Activation memory per schedule** (`eq_lm_pp_mem`).
- *Statement.* GPipe: $m\,a$ per stage. 1F1B: stage $s$ ($0$-indexed) holds $\min(m, p-s)\,a$, the first stage $p\,a$, where $a$ is one microbatch's activation of one stage; the last stage holds $a$.
- *Steps.* (1) After warm-up, 1F1B alternates one forward and one backward, so the number of microbatches in flight on a stage is constant. (2) A stage's warm-up admits $p-s-1$ forwards before its first backward can arrive, then one more arrives with the first steady-state forward: $p-s$ in flight. (3) GPipe never frees before all forwards complete. (4) With full checkpointing, $a$ shrinks to the stage boundary input; see `fsdp_costs`.
- *Check.* Simulator peaks per stage equal $p-s$ for 1F1B ($m \ge p$) and $m$ for GPipe; `PipelineRunner` counts stored (input, output) pairs per rank and matches; T3: measured peak memory vs `pp_costs` (hypothesis: within ±10%; first-stage memory is higher than the last by the embedding and warm-up, as Llama 3 notes).

**D4. Splitting the backward** (`eq_lm_pp_zb`).
- *Statement.* With $B = B_i + W$, a schedule that runs $W$ only when a device would otherwise idle has bubble $(p-1)(F+B_i-W)$ for the unit-cost cases $(F,B_i,W) \in \{(1,1,1), (1,2,1), (2,1,1)\}$; in general the simulator is authoritative and the expression is a lower-envelope heuristic.
- *Steps.* (1) Dependencies: $B_i(s,i)$ needs $B_i(s+1,i)$ and $F(s,i)$; $W(s,i)$ needs only $B_i(s,i)$. (2) The pipeline's drain path now contains only $B_i$ hops. (3) The deferred $W$ tasks of the last $p-1-s$ microbatches fit into the idle slots of the drain, each of length up to $W$. (4) The bubble equals fill $(p-1)F$ plus drain $(p-1)B_i$ minus the absorbed $(p-1)W$. (5) This requires $W \le F+B_i$ (enough idle time) and the in-flight cap $p-s$; memory rises because activations live until $W$.
- *Result.* For $F=1$, $B_i = W = 1$: $(p-1)$ versus $3(p-1)$ for 1F1B.
- *Check.* Simulator at the three cost triples; the cell then runs $(1, 1.2, 0.8)$ and $(1, 0.9, 1.1)$, where the prototype's greedy schedule gave 3.6–3.8 and 3.0 against the expression's 4.2 and 2.4, and prints the deviation. This is the numerical record of the validity region. ZB-H2 and ZBV, which reach near-zero bubble, need the paper's schedules; the greedy `zb_h1` does not (the prototype's `zb_h2` cap gave the same bubble as H1).

**D5. Stage imbalance** (`eq_lm_pp_balance`). A device's time is $\sum_{\text{layers}} c_\ell + c_{\text{head}}\mathbb 1_{\text{last}}$; the step time is set by the slowest device; the idle fraction of the others is $1 - \bar c/c_{\max}$. For Qwen3-8B with $p=4$: the cell computes $c_{\text{head}}/c_{\text{layer}} = 2Vd/(2P_L) = 3.225$ from the config, the naive partition 9/9/9/9 gives $c_{\max}/\bar c = 12.2/9.8 = 1.25$, and the trimmed partition (first stage 10, middle 10, 10, last 6) gives the minimum the heuristic finds. The cell verifies the arithmetic and measures the stage times in calibration. Attention-score FLOPs at $s=4096$ add $\epsilon_a\approx 0.087$ to every layer (6.2) and do not change the ratio materially.

**D6. Boundary communication versus TP** (`eq_lm_pp_comm`). Per boundary, per microbatch, forward and backward send $2bsd\,e$ bytes; compute per stage per microbatch is $6bs(L/p)P_L/(\eta F)$. The ratio is $\rho_{\text{PP}} = \frac{p}{3L}\cdot\frac{d\,e\,\eta F}{P_L\beta}$, against 6.3's $\rho_{\text{TP}} = \frac43 (t-1)\frac{d\,e\,\eta F}{P_L\beta}$. The ratio is $\rho_{\text{TP}}/\rho_{\text{PP}} = 4(t-1)L/p$, which is 252 for $L=36$, $p=4$, $t=8$ (the cell evaluates it from the config). Check: bytes counted by `CommCounter` in the runner (float64) equal $2mbsde$ per interior stage exactly.

#### From-scratch implementation

≤ 150 lines: `pipeline_schedule` (generators for `gpipe`, `1f1b`, `interleaved`; ≈ 35 lines), `PipelineSim.run` (event relaxation over per-device lists with dependency lookups, memory accounting; ≈ 55 lines), `PipelineRunner` (≈ 55 lines), `balance_stages` (≈ 10 lines).

- *Runner logic.* Rank = stage. For `F(mb)`: receive the input activation from the previous rank (stage 0 reads its microbatch), run the stage, keep `(input, output)`, send the output. For `B(mb)`: receive the output gradient (the last stage computes the loss, scaled by `tokens_mb/tokens_total`, and backpropagates from it), call `torch.autograd.backward(output, grad)`, send `input.grad` to the previous stage. Non-blocking `isend` with the handles waited on before buffer reuse; receives are posted by the schedule position, so no deadlock; the schedule action list is the only control flow.
- *Tests.* (1) 4 gloo ranks, float64, a stack of four `d2l.TransformerBlock`s (one per stage) with an untied embedding on stage 0 and an untied head on stage 3, and a $b=8$, $m \in \{4, 8\}$ batch: loss and every parameter gradient equal the full-batch single-process reference to $10^{-9}$ relative, for `gpipe` and `1f1b`; a last microbatch of unequal token count (token-weighted loss) also matches. (2) Peak stored pairs per rank equal the simulator's per-stage peak. (3) Sleep-calibrated costs (F = 20 ms, B = 40 ms) at $p=4$, $m \in \{4, 8, 16\}$: measured idle fraction equals the simulator's within 10% relative (the assignment's criterion; communication is negligible at these costs). (4) Simulator unit tests: closed forms of D1–D4.

#### In production

- `torch.distributed.pipelining`: `PipelineStage(stage_module, stage_index, num_stages, device, group=pp_group)`, `Schedule1F1B(stage, n_microbatches, loss_fn=...)`, `schedule.step(x)` on the first stage and `schedule.step(target=y, losses=losses)` on the last; multi-stage-per-rank schedules take a list of stages (`ScheduleInterleaved1F1B([...])`, `ScheduleZBVZeroBubble`, `ScheduleDualPipeV`). The stage modules are built by constructing the Qwen3 model on the meta device and deleting the layers the rank does not own (embeddings on the first stage, norm and head on the last); no automatic tracer is needed.
- Timelines: forward and backward hooks with CUDA events on each stage module, written with `d2l.rank_log`; compared with the simulator after a calibration run that measures $F_s$, $B_{i,s}$, $W_s$ per stage on one GPU (`DATA_HUB['lm-pp-calibration']`).
- TorchTitan: `pipeline_parallel_degree` and its schedule options (docs: `PIPELINE_PARALLEL.md`); Megatron-Core: `--pipeline-model-parallel-size`, `--num-layers-per-virtual-pipeline-stage`; DualPipe repository for the bidirectional schedule (code only; its `DualPipeV` also exists in `torch.distributed.pipelining`).
- The from-scratch runner is the oracle for loss and gradient equality; `PipelineSim` is the oracle for the order of events: the production schedule's measured per-device action sequence must equal the simulator's input list for the same policy (compared action by action from the committed timeline).

#### Headline experiment

- **Hypothesis.** (H1) For each of four schedules and $m \in \{4, 8, 16\}$, the simulator driven by calibrated per-stage times predicts the measured idle fraction within 10% relative. (H2) At fixed $p=4$, bubble falls as $m$ grows as $(p-1)/(m+p-1)$; interleaving ($v=3$, the divisor of 36 layers/12 chunks) lowers it at the cost of more messages; ZB-H1 lowers it further with higher memory. (H3) The trimmed partition beats the naive 9/9/9/9 partition by the predicted factor.
- **Setup.** Qwen3-8B shape, 36 layers, BF16 params/activations, fp32 Adam on each stage's parameters, full checkpointing per layer, sequence 4096, microbatch 1 sequence, $p=4$ on GPUs 0–3, schedules: GPipe, 1F1B, interleaved 1F1B ($v=3$; $m \in \{4, 8, 16\}$ are all divisible by $p=4$), ZB-H1 via the pipelining library's zero-bubble schedule if available else `ScheduleZBVZeroBubble` with the V placement; 12 steps per run with 4 warm-up; seed 0. **Memory on 40 GB.** Stage state at 12 bytes per parameter (bf16 weights 2, bf16 gradients 2, fp32 Adam moments 8; an fp32 master copy would make it 16 and the first and last stages would not fit) is 28.3 GB on the first and last stages of the naive 9/9/9/9 partition (2.36 B parameters each: the embedding or the head plus nine layers) and 20.8 GB on the middle stages; the trimmed 10/10/10/6 partition puts 30.6 GB on stage 0 (all computed from the config). That leaves 7.7 GB (naive) or 5.4 GB (trimmed, stage 0) under 36 GB for boundary activations (full checkpointing keeps one 33.6 MB input per layer and microbatch: 4.8 GB for 16 microbatches on 9 layers), the recompute workspace and the last stage's chunked loss. GPipe at $m=16$ is the configuration at risk; if a stage exceeds 36 GB the sequence length halves for every schedule.
- **Procedure.** (i) Calibration run: $F_s, B_{i,s}, W_s$ per stage, and boundary transfer time (`isend/irecv` of $4096\cdot4096\cdot2$ bytes), 2 min. (ii) For each schedule and $m$: run, collect the per-rank action timeline, compute idle fraction $=1-\sum_s\text{busy}_s/(p\cdot\text{makespan})$ over steps 5–12, peak memory per stage. (iii) Feed the calibration into `PipelineSim` with the same schedule and compare. (iv) Partition comparison: naive vs trimmed at $m=16$ with 1F1B.
- **Metrics.** Idle fraction (measured and simulated), makespan per step, tokens/s, peak memory per stage (vs `pp_costs`), message count, MFU.
- **Plots.** `fig_lm_pp_gantt`: measured and simulated Gantt charts for 1F1B at $m=8$, one above the other. `fig_lm_pp_bubble`: idle fraction vs $m$ for the four schedules, measured markers and simulated lines. `fig_lm_pp_mem`: peak memory per stage, measured vs predicted. `fig_lm_pp_balance`: per-stage busy time for the two partitions.
- **Expected outcome.** Direction from D1–D5; magnitudes "to be measured". The DualPipe repository table and the Zero Bubble paper give the bubble formulas for the schedules under their own timing assumptions; the notebook states the assumption each time it compares.
- **Noise check.** Eight steady-state steps per run; per-step idle fraction varies less than the differences claimed, or the claim is withdrawn; $m=8$ 1F1B repeated in two launches; sim-vs-measured agreement is judged against that spread.
- **T1 reduction.** Four gloo ranks, sleep-calibrated stages; runner tests; simulator tests; Gantt from the sleep run.
- **Committed (T3).** `lm-pp-calibration` JSON; per-rank timelines (JSONL) for every run; memory summaries.

#### Figures and tables

`fig_lm_pp_schedules`, `fig_lm_pp_gantt`, `fig_lm_pp_bubble`, `fig_lm_pp_mem`, `fig_lm_pp_balance`; `tab_lm_pp_formulas` (bubble, memory, messages for GPipe, 1F1B, interleaved, ZB-H1, with the assumption of each); `tab_lm_pp_stage_cost` (per-stage F, $B_i$, $W$ from calibration); `tab_lm_pp_prediction_vs_measurement`.

#### Pitfalls and misconceptions

- "1F1B has a smaller bubble than GPipe": it has the same bubble and smaller memory (D1, D3).
- Equal layer counts per stage (the head stage is the bottleneck).
- Microbatch count tied to batch size: $m$ larger than needed increases per-microbatch overhead; $m$ smaller than $p$ leaves devices empty (Llama 3's flexible $N$ removes the divisibility constraint).
- FSDP re-gathering parameters per microbatch.
- Measuring the bubble including the optimizer step or data loading.
- Ignoring boundary latency when $F$ is small.
- Quoting a zero-bubble claim without the timing and memory assumptions.
- Loss normalization across unequal microbatches.

#### Summary

- A pipeline sends two orders of magnitude fewer bytes than TP and goes across nodes; its cost is idle time, $(p-1)/(m+p-1)$ of the step.
- 1F1B changes activation memory from $m$ to $p-s$ microbatches per stage, not the bubble.
- Interleaving divides the bubble by $v$, multiplies messages by $v$ and raises first-stage memory; splitting the backward lets weight-gradient work fill the drain.
- The schedule is data: one list drives both the simulator and the executor, so predicted and measured idle time can be compared action by action.
- Embeddings and the head break stage balance; partitions must be computed from costs.

#### Exercises

1. *(check)* For $p=8$ and $m=8$, what fraction of a GPipe step is idle, and what is it at $m=32$? **Sketch:** $7/15 = 46.7\%$ and $7/39 = 17.9\%$.
2. *(check)* Why does 1F1B not reduce the bubble, and what does it reduce? **Sketch:** D1 step 5; D3.
3. *(derive)* Derive the idle fraction when stage $s$ takes $F_s$ and $B_s$ and the longest stage dominates. **Sketch:** the steady state is paced by $\max_s(F_s+B_s)$; the fill and drain sum the stage times along the path; show the fraction is at least $1-\bar c/c_{\max}$.
4. *(derive)* Derive $p + (p-1)/v$ for the first device's peak in-flight activations under Megatron's interleaved order. **Sketch:** D2 step 5; count the warm-up chunk-forwards plus one.
5. *(measure)* Predict, then measure, the idle fraction of 1F1B at $m=4,8,16$ with the naive partition; fit the effective $p_{\text{eff}}-1$ from the measurements and compare with the stage imbalance of D5.
6. *(measure)* Insert a 5 ms artificial delay on one stage's backward in the sleep-calibrated runner; predict with the simulator the makespan change per step and compare. Which dependency carries the delay?
7. *(open)* Add DualPipeV to `pipeline_schedule` (using the DualPipe repository and the V-shape stage placement as references) and compare its simulated bubble and memory with `ScheduleDualPipeV` timelines.

#### Further reading

- Primary: [GPipe](https://arxiv.org/abs/1811.06965); [Efficient Large-Scale Training](https://arxiv.org/abs/2104.04473) (1F1B with interleaving, the bubble and memory analysis); [Zero Bubble Pipeline Parallelism](https://arxiv.org/abs/2401.10241); new, title verified 2026-10-28: [PipeDream](https://arxiv.org/abs/1806.03377) (origin of 1F1B scheduling); the Llama 3 paper's pipeline section (flexible $N$, trimmed first and last stages; [2407.21783](https://arxiv.org/abs/2407.21783)); [DeepSeek-V3](https://arxiv.org/abs/2412.19437) (DualPipe in a production training system; the PDF was read for the node-limited routing text used in 6.6).
- Reference and tutorials: [`torch.distributed.pipelining`](https://docs.pytorch.org/docs/stable/distributed.pipelining.html); new, 200 OK 2026-10-28: [Introduction to Distributed Pipeline Parallelism](https://docs.pytorch.org/tutorials/intermediate/pipelining_tutorial.html); TorchTitan's [`PIPELINE_PARALLEL.md`](https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/PIPELINE_PARALLEL.md).
- Code: [DualPipe](https://github.com/deepseek-ai/DualPipe) (schedules and the bubble table); [Megatron-LM `pipeline_parallel`](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core) (schedules).

#### Risks and open decisions

- **Library API churn.** `torch.distributed.pipelining` schedule names and the zero-bubble schedules change across versions; pin and re-verify. Fallback: report only GPipe, 1F1B and interleaved from the library and the simulator for the rest.
- **Stage memory on 40 GB** is tight (previous paragraph); fallback: sequence 2,048 for every schedule.
- **ZB-H1 availability.** The library offers `ScheduleZBVZeroBubble` and `ScheduleInterleavedZeroBubble` rather than a literal H1; the notebook maps names to policies in a table and says where the simulator's `zb_h1` differs.
- **Hooks distort timing.** CUDA-event hooks add microseconds per action; the calibration measures their overhead and subtracts it.
- **The 10% target** may fail at small $F$ because of boundary latency; the autograder's sleep-based costs avoid this, and the T3 notebook reports the actual agreement.
- **Meta-device stage construction** with HF models can break the tied-embedding path; Qwen3-8B is untied, so this does not arise here, but `d2l.GPT` ties its embedding and head, so the float64 test builds the untied stack from `d2l.TransformerBlock` instead (documented).

### 6.5 Context parallelism

- **Label:** `sec_lm_context_parallel` · **Tier:** T3 (reduced T1 config: 2–4 gloo ranks, float64, sequence 512, hidden 64, 8 query heads, 4 KV heads; timing cells skipped) · **Budget:** T3 ≤ 1.5 h wall on 8× A100 (≈ 12 GPU-h, planned: 3 sequence lengths × 4 variants × ~3 min); peak memory per GPU predicted by `cp_costs` and `fsdp_costs`; T1 ≈ 4 min · **Codex units:** 7.4 (sequence and context parallelism)
- **Builds on (book):** `sec_attention-at-scale` (`eq_online-softmax`, `chunked_attention`: the three running statistics $m, s, \mathbf o$ and the rescaling by $e^{m-m'}$; recap in one paragraph, do not re-derive the single-device online softmax); `sec_kv-cache` (GQA: $H$ query and $H_{kv}$ KV heads; `GQAAttention`); `sec_positional-information` (RoPE: positions must be global); `sec_custom_layer` (`torch.autograd.Function`).
- **Builds on (this part):** 6.1 (`launch`, `CommCounter`, `comm_events`); 6.2 (`ModelShape`, `fsdp_costs`, `exposed_comm`, `saved_bytes`, FSDP2 over a combined data × context mesh); 6.3 (`tp_costs`, head splitting; D5's critical-batch argument); 5.4–5.5 (busbw and α intra-/inter-node for the overlap condition and the cross-node prediction); 4.1 (document masks under packing); 4.6 (long-context quality is out of scope: here only the systems side).
- **Exports:**
  - `d2l.merge_partial_attention(o_a, lse_a, o_b, lse_b) -> (o, lse)`: the associative merge of two normalized partial results over disjoint key sets, safe for fully masked rows ($\text{lse}=-\infty$).
  - `d2l.zigzag_split(x, c, rank, dim=1)` and `d2l.zigzag_gather(x_local, c, dim=1, group=None)`: the $2c$-chunk, ranks-hold-chunks-$(i, 2c-1-i)$ partition of Llama 3 and its inverse.
  - `d2l.ring_attention(q, k, v, group, causal=True, zigzag=True, scale=None)`: forward and backward by autograd function, pure PyTorch blocks (reference quality: tests and ≤ 16k tokens on GPU); documented as the oracle, not the fast path.
  - `d2l.cp_costs(shape, c, *, seq_len, batch=1, variant='ring'|'allgather'|'ulysses', tp=1, causal=True, elem_bytes=2) -> dict` (chapter convention; includes per-step overlap margins).
- **Models and data:** `Qwen/Qwen3-8B` shape (random init; 32 query heads, 8 KV heads, head dim 128), sequences from a long-document sample of the FineWeb-Edu slice of 4.1 concatenated to the target length (loss is not the object); no quality claim at 128K tokens (4.6 owns that).
- **Packages:** `torch==2.15.0` (`torch.distributed.tensor.experimental.context_parallel`, `set_rotate_method`; verified in the 2.15 tutorial), `torch.nn.attention` (SDPA backends, `flex_attention`); optional `flash-attn` 2.x (A100: FlashAttention-2 only; no FA3); optional `ring-flash-attn`, `yunchang` (USP) for pointers only.

#### Narrative arc

1. **The question.** Opening cell: Qwen3-8B at a sequence of 131,072 tokens on one A100. `cp_costs`/`fsdp_costs` print the term that cannot fit: the logits alone are $131072 \times 151936$ elements; under full checkpointing the stored layer inputs are 1.07 GB per layer ($131072 \times 4096 \times 2$ bytes), 38.7 GB over 36 layers, and the recompute working set scales with the sequence; TP does not shard the sequence dimension of the norms unless sequence parallelism is on, and does nothing for the attention's quadratic compute. Printed symptom: the T3 run on 1 GPU fails with out-of-memory at 32k tokens (committed), the predicted peak at 128k, and the share of FLOPs that are attention at 128k: $12Lhd_h s$ against $6N_{\text{mm}}$ per token (4.4× for Qwen3-8B at $s=131072$; the cell recomputes it).
2. **Recap in one paragraph** of online softmax (`eq_online-softmax`): the statistics $(m, s, \mathbf o)$ combine across key blocks. The new idea: the key blocks live on different GPUs.
3. **Merging partial attention** (D1): the merge operator in $(\text{out}, \text{lse})$ form, its identity and associativity, safe handling of fully masked blocks. Cell: split keys into $c$ blocks, merge in every order, compare with dense attention at logit scale 900 in float32 and float64; show the NaN that an unsafe `exp(-inf - -inf)` produces.
4. **Three ways to distribute attention** (figure `fig_lm_cp_variants`): (a) *ring* (pass-KV): KV blocks rotate, queries stay; (b) *all-gather KV* (Llama 3): all-gather K and V along the sequence, attend locally; (c) *Ulysses*: all-to-all from sequence-sharded to head-sharded and back. Bytes and memory of each (D5).
5. **Ring attention in detail.** The forward loop; the backward loop (D2) which recomputes $P_{ij}=\exp(S_{ij}-\text{lse}_i)$ from the saved `out` and `lse` and circulates $(\mathbf K,\mathbf V,d\mathbf K,d\mathbf V)$; overlap condition (D3).
6. **Causal load balance** (D4): contiguous shards give rank $r$ work proportional to $r+\tfrac12$; the zigzag partition equalizes it; Llama 3's $2\times CP$ chunking. Cell: per-rank work table for $c=8$, both partitions.
7. **Ulysses and the head-count constraint.** $H \bmod c = 0$ and $H_{kv}\bmod c=0$; with GQA $H_{kv}=8$ caps Ulysses at $c=8$ unless KV heads are replicated; combination with TP (heads already split by $t$: $c_u\,t \le H_{kv}$); 2D (ring × Ulysses) as in USP; table `tab_lm_cp_feasible` of feasible $(t, c_u, c_r)$ for Qwen3-8B on 16 GPUs.
8. **Everything that must be sharded with the sequence.** Token ids, labels (shift labels *before* sharding so the boundary token pairs are right), position ids/RoPE frequencies (the PyTorch tutorial warns that a missing `freqs_cis` buffer gives wrong rotary embeddings), document masks under packing, the loss mean (equal tokens per rank under zigzag).
9. **CP as an extra data-parallel axis for parameters.** Parameters are replicated across CP ranks, so gradients reduce over the combined (data × context) mesh; FSDP2 shards over that mesh. Cell: count the gradient collective group sizes with `comm_events`.
10. **Training at long context** (up to 64K; 128K is predicted, not run). The run: Qwen3-8B, $c=8$, one sequence per step, FSDP over the 8-GPU mesh, full checkpointing, chunked cross-entropy; memory and step time per variant; the figure of per-rank busy time under contiguous vs zigzag.
11. **Where each wins** (table `tab_lm_cp_choice`): all-gather when KV is small (GQA) and masks are irregular (document masks); ring when KV is large or memory is tight; Ulysses when heads are plentiful and the all-to-all fabric is fast; hybrid beyond. Connection to 4.6 (extending context at training time).

#### Derivations

**D1. Merging partial attention** (`eq_lm_cp_merge`).
- *Statement.* Let key set $\mathcal K = \mathcal K_a \sqcup \mathcal K_b$. For a query $q$ define for a subset $\mathcal S$: $\text{lse}_{\mathcal S} = \log\sum_{j\in\mathcal S}e^{a_j}$ and $o_{\mathcal S}=\sum_{j\in\mathcal S}\text{softmax}_{\mathcal S}(a)_j v_j$. Then with $\text{lse}=\operatorname{logaddexp}(\text{lse}_a,\text{lse}_b)$: $o = e^{\text{lse}_a-\text{lse}}\,o_a + e^{\text{lse}_b-\text{lse}}\,o_b$ equals the attention output over $\mathcal K$. The operator is associative and commutative with identity $(o=0,\text{lse}=-\infty)$.
- *Steps.* (1) $o_{\mathcal S}=\sum_j e^{a_j-\text{lse}_{\mathcal S}}v_j$. (2) For the union, $o=\sum_j e^{a_j-\text{lse}}v_j = e^{\text{lse}_a-\text{lse}}\sum_{j\in\mathcal K_a}e^{a_j-\text{lse}_a}v_j+(\text{same for }b)$. (3) $\text{lse}=\log(e^{\text{lse}_a}+e^{\text{lse}_b})$. (4) Associativity: both groupings give $\sum_j e^{a_j-\text{lse}}v_j$ over the union (step 2 applied twice). (5) Relation to the book's $(m,s,\mathbf o)$ form: $\text{lse}=m+\log s$; $\mathbf o_{\text{book}}/s = o$. (6) Fully masked rows have $\text{lse}=-\infty$ and weight $e^{-\infty}=0$; the code must not compute $-\infty-(-\infty)$: use `where(lse_new == -inf, 0, exp(lse_i - lse_new))`.
- *Check.* Cell: random key partitions in random merge orders versus dense attention at logit scale 900: float64 error $\sim10^{-16}$ (the prototype measured $2\times10^{-16}$), and the unsafe version returning NaN.

**D2. Ring backward needs only $(O, \text{lse})$** (`eq_lm_cp_bwd`).
- *Statement.* For $S_{ij}=q_i\cdot k_j/\sqrt{d_h}$, $P_{ij}=\exp(S_{ij}-\text{lse}_i)$, $O_i=\sum_jP_{ij}v_j$: $dV_j=\sum_iP_{ij}\,dO_i$, $dP_{ij}=dO_i\cdot v_j$, $dS_{ij}=P_{ij}(dP_{ij}-D_i)$ with $D_i=dO_i\cdot O_i$, $dQ_i=\sum_jdS_{ij}k_j/\sqrt{d_h}$, $dK_j=\sum_idS_{ij}q_i/\sqrt{d_h}$.
- *Steps.* (1) Softmax Jacobian: $dS_{ij}=P_{ij}\big(dP_{ij}-\sum_{j'}P_{ij'}dP_{ij'}\big)$. (2) $\sum_{j'}P_{ij'}dP_{ij'}=\sum_{j'}P_{ij'}(dO_i\cdot v_{j'})=dO_i\cdot O_i=D_i$. (3) Hence each block $(i\in\text{local queries}, j\in\text{block})$ can be processed with only $q_i, dO_i, \text{lse}_i, D_i$ (local) and $k_j, v_j$ (visiting). (4) $dQ$ accumulates locally; $dK_j, dV_j$ accumulate on the *visiting* block and travel with it, returning home after $c$ steps.
- *Result.* Backward moves $(K, V, dK, dV)$ per ring step, about twice the forward bytes plus one extra hop for $dK, dV$.
- *Check.* `ring_attention` gradients (all of $dq, dk, dv$) equal dense attention's autograd to $10^{-9}$; bytes counted by `CommCounter` equal the formula.

**D3. Overlap condition** (`eq_lm_cp_overlap`).
- *Statement.* A ring step overlaps iff the local attention block takes at least as long as passing one KV block: $n/c \ge \dfrac{\eta F\,e\,d_{kv}}{2\beta\,d_q}\cdot\kappa$, with $d_q=H d_h$, $d_{kv}=H_{kv}d_h$, and $\kappa=1$ for a dense step, $2$ when only half the work of a step is unmasked (causal average).
- *Steps.* (1) Per step, a rank computes $QK^\top$ and $PV$ for $(n/c)\times(n/c)$ scores: $4(n/c)^2d_q$ FLOPs for a full block. (2) It sends one KV block: $2(n/c)\,d_{kv}\,e$ bytes. (3) Overlap iff $4(n/c)^2d_q/(\eta F)\ge 2(n/c)d_{kv}e/\beta$. (4) Under a causal mask, steps average half the work (and some are skipped), doubling the threshold. (5) The threshold depends on $H_{kv}/H$ (GQA lowers it $4\times$ for Qwen3-8B), on $F/\beta$, and not on model width or on $n$ separately: only on the local block length $n/c$.
- *Result.* With Chapter 5's $\beta$ and $\eta$ measured from a compute-only block, the cell prints the threshold for the intra-node link (the NVSwitch row, and the PCIe-pair row alongside; `topo.txt` decides) and the inter-node link, in tokens per rank, and marks the tested $(n, c)$ points that fall below.
- *Check.* Measured exposed fraction of the library ring (rotate method `allgather` and `alltoall`) versus $n/c$ against the predicted step function; hypothesis: exposed communication appears when $n/c$ falls below the threshold within a factor of 1.5.

**D4. Causal load balance** (`eq_lm_cp_balance`).
- *Statement.* With $c$ ranks and contiguous shards of $n/c$ tokens, rank $r$ computes $r+\tfrac12$ full-block equivalents; the step is paced by rank $c-1$: imbalance $\max/\text{mean}=(2c-1)/c$ (1.875 at $c=8$). With the zigzag partition (rank $r$ holds chunks $r$ and $2c-1-r$ of $2c$ equal chunks) every rank computes the same work.
- *Steps.* (1) Contiguous: rank $r$ attends to $r$ full blocks plus half of its diagonal block. (2) Mean over ranks $=c/2$; max $=c-\tfrac12$. (3) Zigzag chunk $i$ attends to $i$ full chunks plus half its diagonal: $i+\tfrac12$ chunk-units, in units of $(n/2c)^2$. (4) Rank $r$: $(r+\tfrac12)+(2c-1-r+\tfrac12)=2c$ chunk-units for every $r$, i.e. $c/2$ block-units. (5) Equal work; ring steps become equal as well.
- *Check.* Cell: the per-rank work from a counted-FLOPs mask for $c=8$ (exactly the sums above), and measured per-rank attention time on T3 for both partitions at $n=64$k.

**D5. Bytes and memory of the three variants** (`eq_lm_cp_bytes`). Per rank per attention layer, forward, in wire bytes: all-gather KV $2\,n\,d_{kv}e\frac{c-1}{c}$; ring $2\,n\,d_{kv}e\frac{c-1}{c}$ (the same bytes in $c-1$ steps); Ulysses $(2d_q+2d_{kv})\,n\,e\frac{c-1}{c^2}$. Ratio Ulysses/ring $=(1+d_q/d_{kv})/c$, which is $5/c$ for Qwen3-8B (GQA 4:1) and $2/c$ for MHA. Backward: all-gather variant adds a reduce-scatter of $dK,dV$ of the same size; ring about doubles; Ulysses repeats its four all-to-alls. Peak transient KV memory: all-gather $2nd_{kv}e$ (the full sequence), ring $\approx 2\cdot2(n/c)d_{kv}e$ (current and incoming blocks). Check: `CommCounter` bytes of the from-scratch ring and of own all-gather and Ulysses implementations (float64, $c=4$) equal these exactly; at T3 `comm_events` within 1%.

#### From-scratch implementation

≤ 150 lines:
- `_block_attn(q, k, v, qpos, kpos, scale)`: scores, causal mask by positions (so zigzag chunks and rotated blocks use the same code), returns $(o, \text{lse})$ with the fully-masked guard (≈ 14 lines).
- `merge_partial_attention` (≈ 10 lines), `zigzag_split`/`zigzag_gather` (≈ 15 lines).
- `_RingAttention(torch.autograd.Function)`: forward loop over $c$ steps with `batch_isend_irecv` of the KV block (and its positions) to the next rank, merging each block's partial result; saves `q, k, v, out, lse`. Backward: recompute $P$ from `lse`, compute $D$, accumulate $dQ$ locally and rotate $(k, v, dk, dv)$ so each $(dk,dv)$ ends on its home rank (≈ 70 lines).
- `ring_attention` wrapper, and an all-gather reference and a Ulysses implementation used only as byte-count oracles (≈ 30 lines together; not exported).
- *Tests.* 2 and 4 gloo ranks, float64, seeds 0–2, $n=64$, $H=8$, $H_{kv}=4$ (repeat-interleaved to $H$ for the block code), contiguous and zigzag: output and $dq, dk, dv$ equal single-process dense causal attention to $10^{-9}\max(1,\max|\text{ref}|)$; extreme logits (scale 900) finite and equal; a document-mask case (two documents packed, positions reset) equal; bytes equal D2/D5 exactly; per-rank block counts equal D4.

#### In production

- `from torch.distributed.tensor.experimental import context_parallel` and `set_rotate_method("allgather" | "alltoall")` (the latter from `...experimental._attention`, a private path to be re-checked at the pin): `with context_parallel(mesh["cp"], buffers=[x, labels, freqs_cis], buffer_seq_dims=[1, 1, 0]): loss = model(x, labels)`; the context replaces SDPA by ring attention and shards the listed buffers in place; use the flash or memory-efficient SDPA backend (`sdpa_kernel`).
- TorchTitan: `context_parallel_degree` with all-gather KV and Ulysses options (README feature list, 2026-10); Megatron-Core's Context Parallel package (ring with point-to-point, TransformerEngine kernels); the `yunchang` USP library and `ring-flash-attention` for standalone use.
- Own all-gather and Ulysses implementations (≈ 25 lines each, using `torch.nn.attention.bias.causal_lower_right` or FlexAttention for the offset causal mask) complete the three-way comparison where the library lacks a variant.
- The from-scratch `ring_attention` is the oracle: the library path's output and gradients on a 4k-token, 4-rank case agree within BF16 round-off (band to be measured); the three T3 variants must agree with each other on loss over 3 steps to the same band.

#### Headline experiment

- **Hypothesis.** (H1) Counted bytes per attention layer equal D5 for all variants. (H2) At $c=8$ intra-node, every variant is above the overlap threshold at $n \ge 16$k, so differences among ring, all-gather and Ulysses come from memory and kernel efficiency, not exposed communication; the notebook's *inter-node prediction* (threshold with Chapter 5's inter-node $\beta$) says which $(n, c)$ would expose it, to be tested in 6.7's T4 run. (H3) Contiguous sharding is slower than zigzag by about the imbalance of D4.
- **Setup.** Qwen3-8B shape, random init, BF16, FSDP2 over the 8-GPU mesh (6.2 W1 settings), full checkpointing, chunked cross-entropy (2k-token chunks), $c=8$, one sequence per step, $n \in \{16\text{k}, 32\text{k}, 64\text{k}\}$: 6.2's additive terms with $T=n/c$ tokens per rank and two ring KV blocks give 33.1, 34.0 and 35.8 GB at $n$ = 16k, 32k and 64k (computed), against the 36 GB budget, with 64k at the limit (128k would need 39.4 GB and is predicted, not run); variants: library ring (`allgather` rotate), library ring (`alltoall` rotate), own all-gather KV, own Ulysses; partitions contiguous vs zigzag for the ring at 64k; 10 steps with 3 warm-up; seed 0; single node.
- **Procedure.** Each launch records step time, peak memory, `comm_events` bytes per layer, `exposed_comm`, per-rank attention time (CUDA events around the attention call). A dense-attention correctness case at 4k tokens (c=4) is run for every variant against the same model on one GPU: loss and gradient norm equal within the band.
- **Metrics.** Step time and MFU (attention FLOPs counted with the PaLM convention of 6.2, causal-halved version also reported), peak memory vs `cp_costs`, bytes vs D5, exposed fraction, per-rank busy time.
- **Plots.** `fig_lm_cp_scaling`: step time and MFU vs sequence length per variant. `fig_lm_cp_mem`: peak memory vs sequence length with prediction. `fig_lm_cp_balance`: per-rank attention time for contiguous vs zigzag. `fig_lm_cp_overlap`: exposed fraction vs $n/c$ with the predicted threshold (intra-node measured, inter-node predicted as a second shaded line).
- **Expected outcome.** Direction: step time grows faster than linearly with $n$ because attention FLOPs are quadratic; contiguous is slower than zigzag; memory per GPU falls as $1/c$ for the sequence-dependent terms. Magnitudes "to be measured". For orientation only: Llama 3 reports 38% BF16 MFU for 128K-token sequences on 16,384 GPUs with CP=16 (Table 4, 2407.21783), with an all-gather CP implementation; this notebook's single-node numbers are not comparable and are not tuned to match.
- **Noise check.** Seven steady-state steps per run; a variant comparison counts only if the step-time intervals do not overlap; the 64k case is repeated in two launches; correctness comparisons use the float64 oracle, not the BF16 runs.
- **T1 reduction.** Gloo, float64: all tests and byte counts; the Gantt-like figures use the CPU timings only as illustrations flagged "not representative".
- **Committed (T3).** Bundles `cp_<variant>_<n>`, per-rank attention-time JSON, memory snapshots; `data/cp_workload.json`.

#### Figures and tables

`fig_lm_cp_variants` (ring, all-gather, Ulysses data movement for $c=4$), `fig_lm_cp_zigzag` (contiguous vs zigzag causal work), `fig_lm_cp_scaling`, `fig_lm_cp_mem`, `fig_lm_cp_balance`, `fig_lm_cp_overlap`; `tab_lm_cp_choice`, `tab_lm_cp_feasible` (feasible $(t, c_u, c_r)$), `tab_lm_cp_bytes` (D5 per variant), `tab_lm_cp_prediction_vs_measurement`.

#### Pitfalls and misconceptions

- Unsafe `exp(-inf - -inf)` on fully masked rows: NaN (D1).
- Contiguous causal shards: the last rank does $(2c-1)/c$ times the mean work (D4).
- Position ids not sharded with the tokens: wrong RoPE.
- Labels shifted after sharding: wrong targets at shard boundaries.
- Document masks under packing: masks and position resets must travel with the shards.
- Forgetting that parameters are replicated across CP ranks: their gradients need the combined-mesh reduction.
- Ring with MHA at small $n/c$: communication exposed; Ulysses without head divisibility.
- Judging CP by memory only: at moderate $n$ the extra communication can make it slower than FSDP+TP.

#### Summary

- Attention over a sequence sharded across GPUs is exact: partial outputs merge by their log-sum-exps.
- Ring passes KV blocks (bytes $2nd_{kv}e(c-1)/c$, overlapped when $n/c$ exceeds a threshold that GQA lowers); all-gather moves the same bytes but exposes them; Ulysses moves $(1+d_q/d_{kv})/c$ times as many, limited by head counts.
- Under a causal mask, contiguous shards are imbalanced by $(2c-1)/c$; the zigzag partition makes every rank's work equal.
- The backward pass needs only the output and log-sum-exp, and circulates $dK, dV$ with the blocks.
- Context parallelism shards activations along the sequence; every sequence-indexed tensor must be sharded consistently.

#### Exercises

1. *(check)* For $c=4$ and contiguous shards, what fractions of the total causal work do the four ranks do? With zigzag? **Sketch:** $(0.5, 1.5, 2.5, 3.5)/8 = 6.25\%, 18.75\%, 31.25\%, 43.75\%$; zigzag: 25% each.
2. *(check)* Why does the merge need $\text{lse}$ (or $m$ and $s$), and not only the partial outputs? **Sketch:** each partial output is normalized by its own block's sum; merging needs the blocks' relative weights $e^{\text{lse}_a-\text{lse}}$.
3. *(derive)* Show that merge is associative and that its identity is $(0,-\infty)$; then show that the merged result is independent of the ring order. **Sketch:** D1 steps 2–4.
4. *(derive)* Derive the overlap threshold for the backward pass of the ring (compute $\approx 2.5\times$ forward, bytes $\approx 2\times$) and say whether backward or forward is binding. **Sketch:** compute per step $2.5\cdot4(n/c)^2d_q$; bytes per step $4(n/c)d_{kv}e$ (K, V, dK, dV); the threshold is $0.8\times$ the forward one, so forward binds.
5. *(measure)* Predict the exposed fraction of the library ring at $n=16$k, $c=8$ from D3 with Chapter 5's $\beta$ and your measured $\eta$; measure with both rotate methods and explain any gap.
6. *(measure)* Add a sequence-length sweep for Ulysses with $c=4$ and $c=8$ and fit the time to $a + b\,n^2$ (attention) $+ c\,n$ (the rest); compare the fitted split with `d2l.flops_per_token(shape, T=n)` (4.2).
7. *(open)* Implement document-mask-aware context parallelism (packed documents with position resets) for the all-gather variant using FlexAttention's block masks, and measure the wasted work with and without skipping fully masked blocks.

#### Further reading

- Primary: [Ring Attention](https://arxiv.org/abs/2310.01889); [DeepSpeed Ulysses](https://arxiv.org/abs/2309.14509); [TorchTitan](https://arxiv.org/abs/2410.06511) (CP in a composed stack). New, titles verified 2026-10-28: [Striped Attention](https://arxiv.org/abs/2311.09431) (causal load balancing); [USP](https://arxiv.org/abs/2405.07719) (ring × Ulysses); [Context Parallelism for Scalable Million-Token Inference](https://arxiv.org/abs/2411.01783); [LoongTrain](https://arxiv.org/abs/2406.18485) (head-context 2D parallelism); the Llama 3 paper's context-parallelism paragraph (zigzag chunks, all-gather CP, [2407.21783](https://arxiv.org/abs/2407.21783)).
- Reference and tutorials: PyTorch's [Introduction to Context Parallel](https://docs.pytorch.org/tutorials/unstable/context_parallel.html) (200 OK 2026-10-28); the single-device online-softmax derivation in `sec_attention-at-scale`; Scaling Book on attention and sharding.
- Code: [ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention) and [long-context-attention](https://github.com/feifeibear/long-context-attention) (USP), both 200 OK; [torchtitan](https://github.com/pytorch/torchtitan).

#### Risks and open decisions

- **PyTorch CP API stability** (`experimental`, private `_attention` module for `set_rotate_method`). Fallback: report the all-gather and Ulysses variants (own code) plus the ring through the library only if the pinned version exposes it; the from-scratch ring still verifies correctness.
- **Offset causal masks** in SDPA backends: `causal_lower_right` support differs per backend; fallback: FlexAttention with a position-based `mask_mod`, compiled once.
- **128K-token runs** exceed 40 GB even at $c=8$ and chunked loss (39.4 GB additive total, above), so they are not run; the sequence tops out at 64k (35.8 GB, at the limit; fallback 32k). Chunked loss is mandatory and the notebook says so.
- **A100 kernels:** FlashAttention-2 only; the T3 numbers are not predictive of H100/Hopper kernels.
- **No quality claims** at long context; those belong to 4.6.

### 6.6 Expert parallelism

- **Label:** `sec_lm_expert_parallel` · **Tier:** T3 (reduced T1 config: 2–4 gloo ranks, float64, 8 experts, top-2, hidden 64, 2 layers; no timing) · **Budget:** T3 ≤ 1.5 h wall on 8× A100 (≈ 12 GPU-h, planned: the 1-GPU reference run, the EP=8 run, 5 skew levels × 3 dispatch variants × ~2 min, plus a ~10 min routing-profile job on Qwen3-30B-A3B on the T2 pair); peak memory per GPU predicted by `ep_costs`; T1 ≈ 3 min · **Codex units:** 3.2 (routing, accounting convention), 3.8 (upcycling), 7.5 (expert parallelism)
- **Builds on (book):** `sec_moe` (`MoELayer`: linear router, softmax, top-$k$ by `probs + expert_bias`, gate = raw probability `eq_moe-layer`, experts = `d2l.FeedForward`; the capacity-factor exercise $C = cNk/E$ with its normal-approximation drop fraction; the `gathered_moe` sparse-dispatch exercise; `moe_gpt`; fine-grained and shared experts `tab_moe-experts`; accounting `moe_accounting`). 6.6 must not re-derive routing or balancing, and recaps the capacity-factor result in one paragraph with `:numref:`. `sec_multi_gpu_concise` `tab_collectives` (all-to-all: "expert parallelism") and `sec_multi_gpu` (collective cost model).
- **Builds on (this part):** 6.1 (`launch`, `CommCounter`, `comm_events`, `init_dist`); 6.2 (`ModelShape` with MoE fields, `fsdp_costs`, `exposed_comm`, `saved_bytes`, FSDP2 over the expert-data-parallel mesh); 6.3 (the replicated-gradient rule, D2 of 6.3); 5.4 (all-to-all and all-reduce busbw intra-node; the bisection-bandwidth discussion for PCIe vs NVSwitch), 5.5 (inter-node all-to-all; Chapter 5's `d2l.coll_time` and `d2l.a2a_inter_node_time` on `lm-hw-a100-microbench` (all-to-all entries); fallback: 6.6 measures its own all-to-all sweep with `d2l.fit_alpha_beta`); 1.1 (`d2l.load_hf` for Qwen3-30B-A3B). 14.2 later reuses this section's expert-load artifact.
- **Exports:**
  - `d2l.dispatch_plan(expert_idx, num_experts, ep_size) -> DispatchPlan`: pure function from `(n, k)` expert indices to the permutation that groups token copies by destination rank, per-destination counts, per-expert counts, and the inverse permutation; no communication, fully testable on CPU.
  - `d2l.EPMoE(layer_or_weights, group, *, capacity_factor=None, norm_topk=False, grouped='loop'|'grouped_mm')`: expert-parallel MoE layer. `EPMoE.from_reference(layer, group)` shards a `sec_moe` `MoELayer`'s experts (rank $r$ gets experts $[rE/e, (r+1)E/e)$) and replicates the router. `capacity_factor=None` is dropless with a counts exchange (one device-to-host sync per layer); a float selects static-shape buffers (no host sync, dropped overflow). `.expert_parameters()` and `.replicated_parameters()` identify the two gradient-reduction groups (D3).
  - `d2l.sample_routing(n, num_experts, k, skew, *, kind='zipf', seed=0, rank_perm=None, profile=None) -> LongTensor[n, k]`: controlled routing for the skew sweep: `kind='zipf'` (probability $\propto j^{-\text{skew}}$ over a permuted expert order, `skew=0` uniform) or `kind='empirical'` (draws from a `profile` of measured expert frequencies).
  - `d2l.a2a_bytes(tokens, hidden, top_k, ep, *, elem_bytes=2, direction='fwd+bwd', nodes=1, node_limit=None) -> dict(intra_bytes, inter_bytes)`: per-GPU all-to-all wire bytes per MoE layer by D1 and D2 (the formula `4*tokens*top_k*hidden*elem_bytes*(ep-1)/ep` for `fwd+bwd`, balanced routing); time follows from Chapter 5's `d2l.coll_time`/`d2l.a2a_inter_node_time`.
  - `d2l.ep_costs(shape, e, *, tokens, nodes=1, node_limit=None, capacity_factor=None, elem_bytes=2, ac='full') -> dict` (chapter convention; comm split into intra- and inter-node all-to-all bytes).
- **Models and data:** `Qwen/Qwen3-30B-A3B` (30.5B total, 3.35B active parameters computed from `config.json`: 48 layers, $d=2048$, 128 experts, top-8, expert width 768, `norm_topk_prob=True`; Apache-2.0; ≈ 61 GB in BF16) for the routing-load profile and the single-layer oracle; a miniature "MoE-1.7B" of the same family (hidden 1024, 16 layers, 16 query heads with head dim 64, 4 KV heads, 64 experts, top-4, expert width 512, vocab 32,768, random init, ≈ 1.7B parameters, ≈ 0.2B active by the arithmetic of the config; the notebook recomputes both) for training. Synthetic tokens for timing; a 200-step slice of the 4.1 stream for the loss-equality check. `Qwen/Qwen3-0.6B` (dense) for the upcycling cell.
- **Packages:** `torch==2.15.0`; `transformers` (Qwen3-MoE block as reference); optional `megablocks` (grouped/dropless kernels; Ampere supported, to be verified at install) and `torch._grouped_mm` (private name; verify at the pin). DeepEP is described, not run: its README (2026-10) lists "NVIDIA Hopper or newer GPUs" as a requirement.

#### Narrative arc

1. **The question.** Opening cell (T1): `sec_moe`'s `MoELayer` computes all $E$ experts on every token; `gathered_moe` computes only the routed ones. Now put the experts on different GPUs. Two numbers print for Qwen3-30B-A3B at 8192 tokens per GPU on $e = 8$: the expert parameters per GPU (1/8 of 29 B = 3.6 B, 7.2 GB in BF16) and the bytes each GPU must send per layer and step ($4\,n\,k\,d\,e_b\,\frac{e-1}{e}$ = 0.94 GB; 48 layers: 45 GB; the cell recomputes). Then the measured symptom (T3 trace, committed): between the router and the first expert matmul the GPU idles while the CPU reads the token counts to size the all-to-all.
2. **Recap in one paragraph:** `sec_moe`'s layer and the capacity-factor exercise; what changes when experts live on $e$ ranks.
3. **Dispatch and combine** (figure `fig_lm_ep_dispatch`): router → top-$k$ → token copies sorted by destination rank (`dispatch_plan`) → all-to-all → sort by local expert → expert MLPs (loop or grouped GEMM) → reverse all-to-all → unsort → weighted sum. Cell: the permutation round trip is the identity (T0).
4. **Backward through the all-to-all.** The adjoint of an all-to-all with split sizes $(s_{ij})$ is the all-to-all with the transposed sizes; `_AllToAll(torch.autograd.Function)`. Cell: `gradcheck` (float64).
5. **Dropless versus capacity-limited execution.** Recap of $C = cNk/E$; new: what each costs on a real system (D5): dropless needs variable-size splits (host sync) and variable-size GEMMs; capacity needs padding (compute and bytes inflated by $c$) and drops tokens; grouped GEMM (MegaBlocks: block-sparse; `_grouped_mm`) avoids both padding and per-expert launches.
6. **Host synchronization.** `split_sizes` must be Python ints, so every layer does a device-to-host copy that stalls the launch queue; the symptom in the trace; the remedies: static capacity, overlap of two microbatches, device-initiated dispatch (DeepEP: Hopper-only, so named only), CUDA graphs with padded shapes.
7. **Routing skew** (D4): the all-to-all and the expert compute are paced by the busiest rank; cell: imbalance factor $\rho_L(\text{skew})$ for the Zipf family and for the measured Qwen3-30B-A3B profile.
8. **Who reduces what** (D3): router, attention and norms are replicated across EP ranks (they see different tokens) and need an all-reduce; expert parameters live on one rank per expert-data-parallel group and need reduction over that smaller group only; loss normalization by the global token count.
9. **Across nodes** (D2): hierarchical dispatch, per-node deduplication, node-limited routing (DeepSeek-V3: at most 4 nodes per token, NVLink 160 GB/s against IB 50 GB/s in their cluster, 3.2 experts per node on average), and what each does to inter-node bytes. Cell: exact expectation and Monte Carlo.
10. **Why fine-grained experts raise communication intensity** (D1 ratio): $\rho_{\text{EP}}\propto 1/f$, with $f$ the expert width; forward pointer to LatentMoE in 14.2.
11. **Upcycling a dense model** (D6): copy the FFN into $E$ experts; with renormalized top-$k$ gates the upcycled layer reproduces the dense output exactly for any router; with `sec_moe`'s raw-probability gate it does not. Cell on real `Qwen3-0.6B` hidden states (T0/T1): equality, then the parameter and EP-memory bill.
12. **Training the miniature** at EP = 8 against one GPU, then the skew sweep (headline).

#### Derivations

**D1. All-to-all volume and communication intensity** (`eq_lm_ep_bytes`, `eq_lm_ep_ratio`).
- *Statement.* With $n$ tokens per rank, top-$k$ routing, hidden size $d$, element size $e_b$ and $e$ EP ranks, uniform routing: wire bytes per rank per layer and step are $4\,n\,k\,d\,e_b\,\frac{e-1}{e}$ (dispatch and combine in forward, their adjoints in backward), plus $2\,n\,k\,d\,e_b\frac{e-1}{e}$ per extra forward under activation recomputation. The ratio of all-to-all time to expert compute time is $\rho_{\text{EP}}=\dfrac{2e_b}{9}\cdot\dfrac{e-1}{e}\cdot\dfrac{\eta F}{f\,\beta}$ (for $e_b=2$: $\tfrac49\tfrac{e-1}{e}\tfrac{\eta F}{f\beta}$), independent of $d$ and $k$.
- *Steps.* (1) Each token yields $k$ copies; each copy's expert is on rank $r'$ uniform over $e$ ranks, so a fraction $(e-1)/e$ leaves the rank. (2) Dispatch sends $d\,e_b$ bytes per remote copy; combine returns the same. (3) Backward repeats both with gradients of the same size. (4) Total $4\,nk\,d\,e_b(e-1)/e$. (5) Expert compute per token and step: $6\cdot k\cdot 3df$ FLOPs (three SwiGLU matrices, forward and backward). (6) Divide: $\rho = \frac{4kd\,e_b\frac{e-1}{e}/\beta}{18kdf/(\eta F)}$.
- *Result.* Fine-grained experts (small $f$) raise $\rho$: Qwen3-30B-A3B has $f=768$ with $d=2048$.
- *Check.* Cell A: `CommCounter` bytes of the from-scratch layer under *balanced* synthetic routing (float64) equal the formula within the sampling error of the counts (exactly when the routing is made exactly balanced by construction). Cell B (T3): `comm_events` bytes per layer for MoE-1.7B within 2%. Cell C: $\rho_{\text{EP}}$ for Qwen3-30B-A3B and MoE-1.7B with Chapter 5's intra-node all-to-all busbw (the NVSwitch row; the PCIe-pair row alongside, where the all-to-all crosses PCIe for $e > 2$; `topo.txt` decides).

**D2. Node-limited routing and per-node deduplication** (`eq_lm_ep_nodes`).
- *Statement.* With $E$ experts split evenly over $N$ nodes and a token choosing $k$ distinct experts uniformly, the expected number of distinct nodes it visits is $N\big(1-\binom{E-E/N}{k}\big/\binom{E}{k}\big)$, of which $\big(N-1\big)\big(1-\binom{E-E/N}{k}\big/\binom{E}{k}\big)$ are remote when the token is on a fixed node. If a token is sent to each remote node once (deduplicated, then fanned out over NVLink), inter-node bytes per token are $2d\,e_b$ per remote node per direction; with a limit of $M$ nodes per token they are at most $(M-1)$ remote nodes (the token's own node counts toward $M$).
- *Steps.* (1) Hypergeometric probability that a given node holds none of the $k$ chosen experts. (2) Linearity of expectation over nodes. (3) For DeepSeek-V3's setting ($E=256$, $N=8$, $k=8$): $5.29$ distinct nodes in total, $4.63$ remote, against 7.0 expected remote *copies* without deduplication ($k(N-1)/N$). (4) A limit of $M=4$ caps remote nodes at 3 (or 4 if the own node is not counted; the cell states the convention), so inter-node bytes fall by about $1-3/4.63=35\%$ against deduplicated unrestricted routing, at the price of a routing constraint the router must learn.
- *Check.* Monte Carlo (CPU, $10^5$ tokens) against the hypergeometric formula to three digits; the numbers 5.29 and 4.63 are reproduced by the prototype used to write this blueprint (Monte Carlo 5.289, 4.633). `ep_costs(..., node_limit=M)` is checked against the Monte Carlo.

**D3. Gradient groups and loss normalization** (`eq_lm_ep_groups`).
- *Statement.* Let the world have $W = e\cdot e_d$ ranks, $e$ experts-shards by $e_d$ expert-data-parallel replicas. Replicated parameters (router, attention, norms, embeddings) need a gradient reduction over all $W$ ranks; expert parameters over the $e_d$ replicas of the same shard only. With per-rank loss $\mathcal L_r=\frac1{N_{\text{glob}}}\sum_{\text{local tokens}}\ell$, the *sum* reduction gives the gradient of the global mean loss for both groups; with per-rank mean losses, the expert gradients (which already collect all ranks' tokens through the all-to-all) are $e$ times too large relative to the replicated ones.
- *Steps.* (1) The global loss is $\sum_r\mathcal L_r$. (2) A replicated parameter $\theta$ is used by every rank's tokens: $\nabla\theta=\sum_r\nabla_\theta\mathcal L_r$, an all-reduce sum. (3) Expert $j$'s parameters receive, through the adjoint all-to-all, the gradient of every rank's loss through the tokens routed to $j$: $\nabla\theta_j$ is already the full sum over sources; replicas of the same expert shard (other $e_d$ members) hold the gradients of *their* token sets, so the reduction is over $e_d$ only. (4) Per-rank mean losses would scale all contributions by $1/n$ instead of $1/N_{\text{glob}}$; this is harmless for equal $n$ but, combined with a mean all-reduce for replicated parameters and none for experts, mismatches by the factor $e$.
- *Check.* A 4-rank float64 test: per-parameter gradients of `EPMoE` plus the two reductions equal the single-process gradients of `MoELayer` on the concatenated tokens; the same test with the wrong scaling fails by exactly the factor.

**D4. Routing skew** (`eq_lm_ep_skew`).
- *Statement.* With rank loads $\ell_r$ (token copies received) and mean $\bar\ell$, the layer's expert-compute time is $\propto\max_r\ell_r$ and its all-to-all time is $\propto\max_r\max(s_r,\ell_r)$ for send size $s_r$ on a full-bisection fabric; define $\rho_L=\max_r\ell_r/\bar\ell$. For uniform multinomial routing $E[\rho_L]\approx1+\sqrt{2\ln e\,(1-1/e)/(nk)}$; for a pmf $p_j$ over experts and a fixed expert-to-rank map, $\rho_L=e\max_r\sum_{j\in r}p_j$.
- *Steps.* (1) Synchronous layer: all ranks wait for the slowest. (2) Uniform case: the load of one rank is Binomial$(ek\,n, 1/e)$; the maximum of $e$ nearly Gaussian variables with standard deviation $\sigma$ is about $\sigma\sqrt{2\ln e}$ above the mean. (3) Zipf case: sum the pmf over each rank's experts. (4) A random expert permutation (`rank_perm`) lowers $\rho_L$ for Zipf routing relative to contiguous placement.
- *Result.* Under uniform routing at $n k = 65{,}536$ and $e=8$ the imbalance is under 1%; skew, not sampling noise, is the problem.
- *Check.* Predicted $\rho_L$ from the pmf versus the counted $\max\ell_r/\bar\ell$ of `sample_routing` (exact up to sampling); versus measured expert time per layer in T3 (hypothesis: the time grows linearly in $\rho_L$ with slope given by compute+all-to-all at $\rho_L=1$, within 15%; the 15% is a stated test tolerance, not a source value).

**D5. Capacity and padding** (`eq_lm_ep_capacity`). With capacity factor $c_f$, static buffers per (source rank, expert) of $C = c_f\,nk/E$ slots make the all-to-all bytes and expert FLOPs $c_f$ times the dropless balanced values; the dropped fraction under uniform routing follows `sec_moe`'s exercise; under skew it grows with $\rho_L$. Dropless grouped GEMM pays no padding but a variable-size sync. Check: bytes and FLOPs counted in the from-scratch layer at $c_f\in\{1, 1.25, 2\}$ equal $c_f\times$ the dropless values for balanced routing; the drop fraction under $c_f=1$ matches the book's normal approximation within sampling error.

**D6. Function-preserving upcycling** (`eq_lm_ep_upcycle`).
- *Statement.* Copy a dense SwiGLU FFN $f$ into $E$ experts and use any router with top-$k$ selection and renormalized gates $w_i = p_i/\sum_{j\in\text{top-}k}p_j$: then $\sum_{i\in\text{top-}k}w_if(x)=f(x)$ for every $x$. With `sec_moe`'s raw gate $p_i$ the output is $\big(\sum_{\text{top-}k}p_i\big)f(x)$, equal to $\frac kE f(x)$ for a zero (uniform) router, so a rescaling by $E/k$ restores equality only at that router.
- *Steps.* (1) Identical experts: $f_i=f$. (2) Gates sum to one by renormalization. (3) The sum collapses to $f(x)$. (4) Qwen3-MoE configs set `norm_topk_prob=True`, which is this construction.
- *Check.* On 1024 real hidden states from `Qwen3-0.6B` layer 0: max error $\sim10^{-6}$ (float32) between the dense MLP and the upcycled MoE with $E=8$, $k=2$ and a random router; the raw-gate variant fails by the predicted factor.

#### From-scratch implementation

≤ 150 lines:
- `dispatch_plan` (≈ 20 lines): stable sort of the $nk$ copies by destination rank, counts per rank and per expert, inverse permutation.
- `_AllToAll(torch.autograd.Function)` (≈ 18 lines): forward `dist.all_to_all_single` with `input_split_sizes`/`output_split_sizes`; backward swaps them.
- `EPMoE.forward` (≈ 70 lines): router (identical to `MoELayer`'s, or renormalized with `norm_topk`), plan, counts exchange (tiny all-to-all, then `.tolist()`), dispatch of hidden states and local expert ids, local sort, expert compute by `loop` (per-expert slices through the reference `FeedForward`) or `grouped_mm`, reverse all-to-all, unsort, gate-weighted sum over $k$. Capacity mode: per-(source, expert) slot index by cumulative counts, drop overflow, fixed-size buffers, no `.tolist()`.
- `sample_routing` (≈ 15 lines), `ep_costs` helpers.
- *Tests* (2 and 4 gloo ranks, float64, seeds 0–2, $E=8$, $k=2$, $d=64$, $n=24$ tokens per rank with uneven counts; reference `MoELayer` run on the concatenated tokens, with `expert_bias` zero and non-zero): output slices, gradients of every expert and router parameter (after the reductions of D3) and input gradient equal the reference to $10^{-9}$; `gradcheck` of `_AllToAll`; counted bytes equal D1 for exactly-balanced routing; capacity mode with $c_f=E$ equals dropless; with $c_f=1$ the dropped fraction equals the plan's overflow count; renormalized mode equals `Qwen3MoeSparseMoeBlock` semantics on a toy config.

#### In production

- **Oracle on a real model (T3).** Layer 0 of `Qwen3-30B-A3B` (128 experts) on 4096 real hidden states: `EPMoE` with $e=8$ equals the Transformers `Qwen3MoeSparseMoeBlock` output within the BF16 band (to be measured; the float64 test covers exactness).
- **Library paths.** TorchTitan expert parallelism (`expert_parallel_degree`, standard all-to-all dispatch on A100; DeepEP/HybridEP dispatch on Hopper-class hardware); Megatron-Core MoE (all-to-all token dispatcher, `--expert-model-parallel-size`, documented in its "Mixture of Experts" page); Tutel and MegaBlocks as kernel libraries; DeepSpeed-MoE for reference. The MoE-1.7B training run is done with TorchTitan's EP path where a recipe can be written for it in a few lines (the README shows how to add a recipe); otherwise with `EPMoE` inside a minimal training loop; the notebook states which.
- **Grouped GEMM.** `megablocks` (block-sparse dropless MLP) or `torch._grouped_mm` against the per-expert loop; throughput versus tokens per expert (table `tab_lm_ep_gemm`).
- **Equality of training.** MoE-1.7B EP = 8 versus the same model on one A100 (it fits 40 GB: ≈ 1.7B parameters with 16 bytes per parameter of training state ≈ 27 GB, and 16,384 tokens per step with checkpointing and chunked loss add a few GB; if `ep_costs` prints more than 36 GB the single-GPU reference halves its tokens per step and EP = 8 follows): loss over 20 steps with identical data and seed, difference within the BF16 band (to be measured; computed relative to two same-config reruns on the single GPU).

#### Headline experiment

- **Hypothesis.** (H1) EP = 8 reproduces the single-GPU loss trajectory of MoE-1.7B within the BF16 reproducibility band. (H2) Counted all-to-all bytes per layer equal D1; the host-sync variant leaves a GPU gap per layer that the static-capacity variant removes; (H3) layer time grows linearly with the imbalance factor $\rho_L$ as in D4.
- **Setup.** MoE-1.7B, BF16, FSDP2 for non-expert parameters over the 8 GPUs and EP = 8 for experts ($e_d=1$), full checkpointing, sequence 2048, 8 sequences (16,384 tokens) per GPU, AdamW, 30 steps with 10 warm-up per measurement, seed 0. Dispatch variants: dropless with `.tolist()` sync (`loop` and `grouped_mm`), capacity $c_f\in\{1.25, 2\}$, library path if a recipe is available. Skew: `sample_routing` replaces the router's top-$k$ indices (the router's probabilities still supply gates, so compute is unchanged) with Zipf skew $\in\{0, 0.5, 1, 1.5, 2\}$ and contiguous vs random expert placement, plus the empirical profile of Qwen3-30B-A3B (profiled below).
- **Routing profile job (T2 pair).** Qwen3-30B-A3B on the T2 pair (BF16, 61 GB, split layer-wise over the two 40 GB cards with `device_map`; one card cannot hold it, and only router counts are recorded, so the split does not affect the result) over 64k tokens drawn from the 4.1 slice (code, math, prose mix as available there; 14.2 does the domain analysis): per-layer expert token counts saved as `lm-qwen3-30b-a3b-expert-load` (`.npz`, ≈ 48 × 128 integers per domain).
- **Procedure.** (i) Equality run: single GPU vs EP = 8, loss every step. (ii) Bytes: `comm_events` per layer. (iii) Gap: profiler trace around the counts `.tolist()`: time between the last kernel before the sync and the first kernel after it (`d2l.exposed_comm` plus a small gap finder in the notebook). (iv) Skew sweep: per-layer router→combine time, expert compute time and all-to-all time per skew level and variant.
- **Metrics.** Loss difference per step; bytes vs D1; gap per layer and fraction of step; per-layer dispatch time vs $\rho_L$; step time; tokens dropped (capacity variants).
- **Plots.** `fig_lm_ep_loss` (EP=8 vs 1 GPU); `fig_lm_ep_gap` (a trace excerpt of the sync gap); `fig_lm_ep_skew` (dispatch time and expert time vs $\rho_L$, measured points and the D4 line, one colour per variant); `fig_lm_ep_load` (the measured Qwen3-30B-A3B load heat-map: layers × experts, with the max/mean per layer).
- **Expected outcome.** Direction: dispatch time rises with skew; the capacity variants remove the host gap and drop tokens at high skew; contiguous placement is worse than random under Zipf. Magnitudes "to be measured". DeepSeek-V3 reports training without token dropping and a limit of 4 nodes per token (2412.19437, section 4.2); that is context for D2, not a target for this single-node run.
- **Noise check.** 20 steady-state steps per point; two launches at skew 1.0; trend claims need the slope's bootstrap interval (over steps) to exclude zero; loss equality is compared against the single-GPU rerun spread.
- **T1 reduction.** Gloo, float64: all unit tests and byte/permutation checks; no timing; the skew sweep is run on counts only (imbalance factors without times).
- **Committed (T3).** Bundles for each variant and skew level, `lm-qwen3-30b-a3b-expert-load`, trace excerpts.

#### Figures and tables

`fig_lm_ep_dispatch`, `fig_lm_ep_nodes` (distinct nodes visited vs $k$ and $M$), `fig_lm_ep_loss`, `fig_lm_ep_gap`, `fig_lm_ep_skew`, `fig_lm_ep_load`; `tab_lm_ep_bytes` (per-token bytes for Qwen3-30B-A3B, MoE-1.7B and DeepSeek-V3-like settings), `tab_lm_ep_variants` (dropless/capacity/grouped, what each costs), `tab_lm_ep_gemm`, `tab_lm_ep_prediction_vs_measurement`.

#### Pitfalls and misconceptions

- A host sync per layer hidden inside `.tolist()`/`.item()`.
- Gradient scaling of expert parameters (D3), and expert gradients all-reduced over the wrong group.
- Capacity dropping silently changes the function; compare dropless and capacity outputs before trusting a throughput number.
- Balanced routing in a benchmark hides the skew a trained router has.
- Counting only the all-to-all and forgetting the expert-count exchange and the sort.
- Experts per rank not divisible: $E \bmod e \ne 0$.
- Upcycling with raw-probability gating is not function-preserving.
- Assuming EP needs NVLink-class bandwidth for correctness: it needs it for speed; the bisection bandwidth of the node (5.4) sets the all-to-all.

#### Summary

- Expert parallelism sends each token copy to the rank that holds its expert and back: $4nkd\,e_b(e-1)/e$ bytes per rank per layer and step, with intensity inversely proportional to expert width.
- The adjoint of an all-to-all is the all-to-all with transposed split sizes; replicated and expert parameters have different gradient-reduction groups.
- Dropless execution costs a host sync per layer; capacity buffers remove it, inflate compute and bytes by $c_f$ and drop tokens.
- Skew, not sampling noise, sets the straggler: layer time follows the busiest rank.
- Across nodes, per-node deduplication and node-limited routing cut inter-node bytes; function-preserving upcycling needs renormalized gates.

#### Exercises

1. *(check)* For Qwen3-30B-A3B with $e=8$ and 8192 tokens per GPU, how many bytes does one GPU send per layer in the forward pass? **Sketch:** $2\,nkd\,e_b(e-1)/e = 2\cdot8192\cdot8\cdot2048\cdot2\cdot0.875=0.47$ GB.
2. *(check)* Why must the expert-data-parallel gradient reduction exclude the other EP shards? **Sketch:** D3 step 3: different shards hold different experts.
3. *(derive)* Derive the expected inter-node bytes per token for unrestricted routing without deduplication, with deduplication, and with a node limit $M$, for $E=128$, $N=4$, $k=8$. **Sketch:** copies $k(N-1)/N=6$; hypergeometric distinct remote nodes; cap $M-1$.
4. *(derive)* Show that under Zipf routing with exponent $s$ and contiguous expert placement the imbalance is $\rho_L=e\,\sum_{j\le E/e}j^{-s}/\sum_{j\le E}j^{-s}$, and evaluate its limit as $E\to\infty$ for $s>1$. **Sketch:** rank 0 holds the heaviest $E/e$ experts; the sums converge.
5. *(measure)* Predict then measure the dispatch time at skew 1 for contiguous vs random placement, using D4; report the largest discrepancy and its likely cause (all-to-all not purely max-paced; kernel efficiency at small groups).
6. *(measure)* Time `loop` versus `grouped_mm` expert compute as tokens per expert vary from 8 to 4096 at $f=768$ and $f=1536$; find the crossover and relate it to the arithmetic intensity of a $\text{tokens}\times d\times f$ matmul (`sec_perf_model`).
7. *(open)* Implement two-microbatch overlap (dispatch of microbatch 2 while computing microbatch 1) in `EPMoE` and measure how much of the all-to-all time it hides at balanced routing and at skew 1.

#### Further reading

- Primary: [Switch Transformers](https://arxiv.org/abs/2101.03961) (capacity factor, token dropping); [MegaBlocks](https://arxiv.org/abs/2211.15841) (dropless block-sparse expert computation); [Sparse Upcycling](https://arxiv.org/abs/2212.05055) (dense-to-MoE initialization); [DeepSeek-V3](https://arxiv.org/abs/2412.19437) (node-limited routing, communication kernels, no token dropping). New, titles verified 2026-10-28: [GShard](https://arxiv.org/abs/2006.16668) (expert-parallel all-to-all and capacity); [Tutel](https://arxiv.org/abs/2206.03382) (adaptive parallelism for MoE at scale).
- Reference: [DeepEP](https://github.com/deepseek-ai/DeepEP) (README read 2026-10: Hopper or newer required); Megatron-Core's MoE page; TorchTitan's expert-parallel documentation.
- Teaching: Ultra-Scale Playbook (expert parallelism section); Scaling Book (sharding MoE).
- Code: [megablocks](https://github.com/databricks/megablocks) (200 OK); [torchtitan](https://github.com/pytorch/torchtitan) (Qwen3-MoE and DeepSeek-V3 models with EP); [DeepEP](https://github.com/deepseek-ai/DeepEP).

#### Risks and open decisions

- **DeepEP cannot run on A100.** Stated in the notebook; the host-sync remedy is demonstrated with static capacity, and DeepEP is cited as the production answer on Hopper.
- **`torch._grouped_mm` and MegaBlocks availability on the build image.** Fallback: per-expert loop only; the crossover exercise then uses the loop and a padded `bmm`.
- **No multi-node MoE run in this section.** The inter-node claims of D2 are checked by Monte Carlo and by a prediction; 13.4 runs expert-parallel inference across nodes. Decided 2026-10-29: 6.7 adds one optional EP=16 two-node run of MoE-1.7B (a single configuration, run only if 6.7's two-node allocation has spare time); without it, 6.7 keeps the prediction-only treatment for the inter-node all-to-all.
- **Qwen3-30B-A3B routing profile** depends on the text mix; the artifact records the mix and token count; 14.2 may replace it with its domain-resolved profile.
- **MoE-1.7B is our own configuration**, not a released model; it is named as such everywhere and does not claim a quality.

### 6.7 Composing parallelism across nodes

- **Label:** `sec_lm_composing_parallelism` · **Tier:** T4 (core: 2 nodes of 8× A100; optional: a 4-node pass and one EP=16 MoE run; the notebook analyzes committed logs) · reduced T1 config: the mesh and cost-model cells are T0; a 2-rank gloo mesh test checks group membership; calibration of the GEMM-efficiency table runs on the build box's single Ada GPU as a method demonstration (its numbers are not used for the A100 predictions) · **Budget:** T4 collection, core ≈ 1 h wall on 2 nodes (≈ 15 GPU-h, planned: 6 two-node runs and 2 one-node runs of ~8 min each, listed under Procedure), plus optional ≈ 0.5 h on 4 nodes (≈ 17 GPU-h, 4 runs) and an optional EP=16 run (≈ 2 GPU-h); the notebook itself runs in minutes from the committed bundles; T0 cost-model cells < 1 min · **Codex units:** 7.6 (multidimensional parallelism), 7.8 (choosing a configuration)
- **Builds on (book):** `sec_multi_gpu` (weak vs strong scaling, `eq_dp_cost`), `sec_training_systems` (the scaling ladder and library landscape it retires: `tab_training_tools` is rewritten here from checked facts; `eq_global_batch`: $B_{\text{global}}=B_{\text{device}}N_{\text{ranks}}N_{\text{accum}}$ is reused, not re-derived), `sec_batch_size` (critical batch size caps useful global batch), `sec_scaling-laws` (6ND), `sec_perf_model` (roofline, achieved FLOP/s).
- **Builds on (this part):** 5.4–5.6 (intra- and inter-node α–β and busbw; the layout calculator of 5.6, which 6.7 extends with time and memory; MFU-to-dollars conversion), 6.1 (`launch` with `-m` modules, `make_mesh` checks use its gathered hostnames, bundles), 6.2–6.6 (each section's `*_costs` function, its measured one-node numbers as calibration data, and its `exposed_comm`/`comm_events` measurements), `d2l.chip_specs()` (peak FLOP/s and HBM for A100). Chapter 5's `d2l.coll_time`, `d2l.hier_allreduce_time`, `d2l.a2a_inter_node_time` on `lm-hw-a100-microbench` and the `alpha_beta.json` summaries in `lm-hw-node-p2p` and `lm-hw-cluster-nccl` (as in 6.1 and 6.6); fallback: the notebook uses 6.1's locally measured fits (in its launch bundle) and labels them so. Artifacts consumed: the 8-GPU bundles of 6.2 (W1), 6.3 (TP sweep), 6.4 (PP runs), 6.5 (CP runs) as the one-node calibration set.
- **Exports:**
  - `d2l.ParallelConfig` (dataclass): `dp_shard, dp_replicate, tp, pp, cp, ep, microbatch, accum, vstages, reshard_after_forward, ac ('none'|'selective'|'full'), seq_len`, with `.world_size` and `.validate(shape, cluster)` returning the list of violated constraints.
  - `d2l.make_mesh(config_or_dims, *, gpus_per_node=8, order=('pp','dp_replicate','dp_shard','cp','tp'), real=False)`: returns a `MeshSpec` (rank array, group membership per axis, node span per axis, link class per axis) without needing `torch.distributed`; with `real=True` builds the matching `DeviceMesh` via `init_device_mesh` and flattens `dp_shard×cp` into a named dimension. `.assert_intra_node(axis)`, `.groups(axis)`, `.render()` (a coloured grid by node).
  - `d2l.enumerate_configs(shape, nodes, gpus_per_node, *, global_tokens, seq_len, mem_limit, microbatch_options=(1, 2, 4), tp_max=None) -> (list[ParallelConfig], DataFrame of rejected configs with reasons)`.
  - `d2l.ParallelCostModel(shape, cluster, calib)`: `cluster = {'gpus_per_node', 'peak_flops', 'hbm_bytes', 'intra': {'alpha_us', 'busbw_GBs'}, 'inter': {...}, 'alltoall': {...}}` (`intra` is the NVSwitch row of 5.4's fits; for a node of NVLink-bridged PCIe cards a second description sets `intra` to the PCIe row and adds `intra_pair` for the bridged pair, so that only groups of two use the bridge; `topo.txt` decides which description the committed node gets), `calib = {'eta_table', 'overlap_kappa'}`; `.evaluate(config) -> Estimate(step_time_s, mfu, mem_peak_bytes, breakdown: dict, waterfall: list)`; `.rank(configs) -> DataFrame`. Composes `fsdp_costs`, `tp_costs`, `pp_costs`, `cp_costs`, `ep_costs`, the simulator of 6.4 for the bubble and `balance_stages` for stage imbalance.
- **Models and data:** the 7–8B dense model run by TorchTitan: `Qwen/Qwen3-8B` shape (36 layers, $d=4096$) if the pinned TorchTitan recipe set supports TP, PP and CP for it (its README lists `qwen3_14b` as the verified Qwen3 recipe, so an 8B recipe is a one-function change in a user module), otherwise the verified `llama3_8b` recipe (the notebook prints the `ModelShape` of whichever ran). Random initialization, the training data path of the recipe (a small preprocessed text set; loss is not analyzed), sequence 4096. Licenses: Qwen3 Apache-2.0; Llama 3 weights are not needed (random init) but its tokenizer is gated: prefer the Qwen3 path.
- **Packages:** `torch==2.15.0` (if the pinned TorchTitan commit needs another torch it runs in its own environment, as the serving engines do, and `d2l.launch` calls it by command line); `torchtitan` pinned by commit (the README as of 2026-10 lists Python recipes selected with `MODULE=… CONFIG=…`; the `--module/--config` options; `scripts/multinode_trainer.slurm`); `pandas`, `matplotlib`.

#### Narrative arc

1. **The question.** Opening cell: Qwen3-8B on 2 nodes (16 GPUs) at a global batch of 512 sequences of 4096 tokens. `enumerate_configs` prints how many ordered factorizations of 16 into (dp, tp, pp, cp) exist (35: D2; 56 at the optional 4-node scale of 32 GPUs), how many survive divisibility and memory, and for three hand-picked layouts (FSDP only; TP=8 × FSDP=2; TP=4 × PP=2 × FSDP=2) the committed measured MFU, which differ by a large factor (printed, to be measured). The question: can we rank them before spending cluster hours?
2. **Meshes and topology.** `make_mesh`: rank arrays, intra- versus inter-node classification, `render()` for 32 GPUs coloured by node; the invariant of 6.1 D3; the order [TP, CP, PP, DP] with the ranks of one TP group adjacent (Llama 3, Fig. 5 and 3.3.2, explicitly: "innermost parallelism requires the highest network bandwidth and lowest latency"); why EP and FSDP shard dimensions are not independent Cartesian factors (in TorchTitan the expert axis is carved out of the data-parallel(-times-context) ranks; check at the pin) and how `make_mesh` represents it. Cell: group-membership assertions for 3 layouts, overlap and ordering detection (a deliberately mis-ordered mesh is rejected with the offending axis named).
3. **What each axis moves and how much of it hides** (`tab_lm_compose_axes`): for Qwen3-8B, bytes per GPU per step and fraction exposed for TP (6.3, unoverlapped), CP (6.5, overlapped above a threshold), PP (6.4, small bytes; cost is the bubble), FSDP (6.2, overlapped above $T^*$), EP (6.6, for MoE). All entries computed by the `*_costs` functions.
4. **The placement rule** (D1) and why [TP, CP, PP, DP] falls out when $T\ge T^*$, and what changes when it does not.
5. **Enumerating configurations** (D2): constraints table (`tab_lm_compose_constraints`): divisibility of heads and KV heads by TP, layers by PP·vstages, microbatches ≥ PP, global batch divisibility, TP ≤ node size (a soft constraint: allowed but flagged), memory ≤ limit, CP only when sequence/CP ≥ a block. Cell: the funnel of counts.
6. **The cost model** (equations E1–E7 in `tab_lm_compose_model`), including every assumption: compute time from FLOPs and a GEMM-efficiency table $\eta(M,N,K)$; TP collectives exposed; FSDP collectives overlapped up to the per-layer budget; PP via the simulator plus imbalance; CP per the overlap condition; optimizer step and norm/pointwise overhead as a fixed fraction measured on one node. Two free parameters only: the $\eta$ table (from a one-GPU sweep of the TP-sharded shapes) and the overlap efficiency $\kappa\in[0,1]$ (fit on one-node runs).
7. **Calibrate on one node, predict two and four.** Fit $\eta$ and $\kappa$ on the committed 8-GPU bundles of 6.2–6.5; freeze; write `predictions.json` (ranked configs, predicted step times and memory for 2 nodes and, for the optional pass, 4 nodes) and commit it with the repository hash **before** the cluster runs. The notebook prints the hash and the file's modification order.
8. **The runs.** `d2l.launch(..., mode='slurm')` with TorchTitan; configuration recipes in a user module; core (2 nodes): the top-3 predicted layouts, one predicted-worst (TP=16 across nodes, the cross-node TP prediction of 6.3), an FSDP-only baseline and a repeat of the top-1 layout; optional (4 nodes): the top-2 predicted, the predicted-worst and the FSDP-only baseline; logs record step time, TorchTitan's own TFLOPs/MFU/memory metrics, per-rank times. All from committed bundles in the build. **Optional EP=16 run (decided 2026-10-29; closes the editor question of 6.6).** If the two-node allocation has spare time after the core runs, one MoE configuration runs: 6.6's MoE-1.7B (64 experts, top-4, expert width 512) with $e=16$ expert-parallel ranks over the 2 nodes ($e_d=1$, four experts per rank), the same `EPMoE` or TorchTitan recipe as 6.6, 30 steps, the capacity setting chosen beforehand from 6.6's result, one launch (about 2 GPU-h). It supports one claim: under balanced routing the share of all-to-all wire bytes that crosses nodes is $(e-g)/(e-1)=8/15$ ($g=8$ GPUs per node), computed from `dispatch_plan`'s per-destination counts (NCCL reports no per-link bytes), and the measured step time agrees with the all-to-all time from `ep_costs(nodes=2)` and `d2l.a2a_inter_node_time` at 5.5's $\beta_{\text{out}}$ added to the single-node step. It does not test node-limited routing or per-node deduplication, which `EPMoE` does not implement (6.6 D2 stays a Monte Carlo and a prediction). If the time is not spare, 6.7 keeps the prediction-only treatment; nothing else depends on this run.
9. **Predicted versus measured** (`tab_lm_compose_pred_meas`): step time, MFU, peak memory, ranking (Kendall's $\tau$ with a bootstrap interval, top-1 hit), and the breakdown error per axis from `comm_events`/`exposed_comm`. Discrepancies explained, not hidden.
10. **MFU waterfall** (D3): from the ideal 100% through GEMM efficiency, recompute, bubble, exposed TP, exposed FSDP, imbalance and other overhead to the measured value, for the best and a poor config.
11. **Strong scaling at scale** (D4): the best layout and FSDP-only at the fixed global batch on 1 and 2 nodes (and 4 if the optional pass ran); the weak series (tokens per GPU fixed) is cut, and D4's weak side is shown from the model only; the critical-batch limit on useful $N$; Llama 3's reported rows (8,192 GPUs: TP8 PP16 DP64, 43%; 16,384 GPUs: 41% and, at 128K tokens with CP16, 38%; Table 4 of 2407.21783) and MegaScale's 55.2% at 12,288 GPUs for 175B (2402.15627) are shown *as context*, with their MFU conventions and scales stated, never as targets.
12. **Sensitivity and recommendation.** Re-run the model over sequence length $\{4\text{k}, 16\text{k}, 32\text{k}\}$ and inter-node bandwidth $\{\tfrac14, \tfrac12, 1, 2, 4\}\times$: heat-map of the best layout and its predicted MFU; the recommendation for the 7–8B case on 2–4 nodes, with the conditions under which it flips.
13. **The library landscape** (`tab_lm_compose_libraries`; replaces `tab_training_tools`): TorchTitan, Megatron-Core, DeepSpeed, nanotron, with checked features only (below), plus where each fits by scale. Cost: convert the measured MFU of the best config to dollars per million tokens with 5.6's price inputs (T0 cell).

#### Derivations

**D1. Placement by exposed volume** (`eq_lm_compose_place`).
- *Statement.* Let axis $a$ carry $V_a$ wire bytes per GPU per step of which a fraction $x_a$ is exposed (not hidden behind compute), and let the links assigned to the axes have per-GPU bandwidth $\beta_{\pi(a)}$. The exposed communication time $\sum_aV_ax_a/\beta_{\pi(a)}$ is minimized by giving the axis with the largest exposed volume $V_ax_a$ the fastest link, and so on in order.
- *Assumptions.* Axes use disjoint links of a hierarchy with distinct bandwidths; times add (no overlap between axes); each axis's exposed fraction does not depend on its link (an approximation to be stated, since a slower link lowers the overlap).
- *Steps.* (1) Two axes with exposed volumes $U_1\ge U_2$ and links $\beta_1\ge\beta_2$. (2) Compare assignments: $U_1/\beta_1+U_2/\beta_2$ against $U_1/\beta_2+U_2/\beta_1$. (3) The difference is $(U_1-U_2)\,(1/\beta_2-1/\beta_1)\ge0$. (4) Extend to $n$ axes by adjacent exchanges (rearrangement inequality). (5) Evaluate $U_a$ for Qwen3-8B: TP and CP exposed (large), PP small, FSDP exposed only when $T<T^*$.
- *Result.* TP innermost, then CP, then PP, then data parallelism outermost when $T\ge T^*$ (the order of Llama 3); when tokens per GPU fall below $T^*$, FSDP's exposed volume rises and may outrank PP and even CP, which is what the sensitivity heat-map shows.
- *Check.* Brute force over all axis permutations for the three layouts with the cost model: the minimizer equals the sorted order in each case where the stated assumption holds; the cell prints the exceptions.

**D2. Counting configurations** (`eq_lm_compose_count`). The number of ordered factorizations of $2^a$ into four factors (dp, tp, pp, cp) is $\binom{a+3}{3}$: 35 for 16 GPUs, 56 for 32. *Steps:* distribute the exponent $a$ over four axes (stars and bars). *Check:* `enumerate_configs` without constraints returns exactly these counts; with a fifth axis (ep) $\binom{a+4}{4}$.

**D3. MFU as a product** (`eq_lm_compose_mfu`).
- *Statement.* With model FLOPs per step $\Phi$ on $N$ GPUs of peak $F$, $\text{MFU}=\Phi/(N F\,T_{\text{step}})$. If $\Phi_{\text{hw}}=\rho_r\Phi$ is the executed FLOPs ($\rho_r=4/3$ for full activation recomputation in the matmul-dominated limit, 1 without), $\bar\eta$ the mean GEMM efficiency, $b$ the pipeline idle fraction, and $T_{\text{exp}}$ the exposed communication plus other non-overlapped time, then $T_{\text{step}}=\dfrac{\rho_r\Phi}{NF\bar\eta(1-b)}+T_{\text{exp}}$ and
  $\text{MFU}=\dfrac{\bar\eta\,(1-b)}{\rho_r}\cdot\dfrac{1}{1+T_{\text{exp}}/T_{\text{compute}}}$.
- *Steps.* (1) Compute time is hardware FLOPs over achieved rate. (2) The bubble inflates compute time by $1/(1-b)$. (3) Exposed time adds. (4) Divide $\Phi/(NF)$ by $T_{\text{step}}$. (5) Each factor is a separate bar of the waterfall.
- *Result.* An upper bound $\text{MFU}\le\bar\eta(1-b)/\rho_r$ before any communication; e.g. with full recomputation MFU $\le0.75\,\bar\eta$.
- *Check.* For each measured run, measure the factors independently ($\bar\eta$ from the compute-only kernels of the trace, $b$ from the PP timeline or 0, $T_{\text{exp}}$ from `exposed_comm`) and verify that the product reproduces the run's reported MFU within 5% (a test tolerance; larger gaps are attributed to the unmodelled remainder in the waterfall).

**D4. Weak, strong scaling and the useful number of GPUs** (`eq_lm_compose_strong`).
- *Statement.* With global tokens $B$, the data-parallel degree $D=N/(tpc)$ and tokens per replica $B/D$, FSDP overlap requires $B/D\ge T^*$ (6.2), so the largest GPU count that keeps FSDP overlapped is $N_{\max}=t\,p\,c\,B/T^*$; pipeline efficiency further requires $m=B/(D\,b_\mu s)\gtrsim 4p$ for a bubble below $p/(m+p)$ small. Strong scaling at fixed $B$ therefore degrades once $N>N_{\max}$ unless $t,p,c$ grow.
- *Steps.* (1) $B=D\cdot b_\mu\cdot m\cdot s$ (`eq_global_batch` with accumulation). (2) Per replica tokens are $B/D$. (3) The overlap condition of 6.2 is a lower bound on per-replica tokens. (4) Solve for $D$, then $N=Dtpc$. (5) The critical batch $B_{\text{crit}}$ of `sec_batch_size` caps $B$ itself; weak scaling cannot raise $B$ beyond it without losing sample efficiency.
- *Check.* Table of $N_{\max}$ for $B=2^{21}$ tokens and $T^*$ from 6.2 over $(t,p,c)$; the measured strong-scaling series on 1 and 2 nodes (and 4 if run) marks where the best config's MFU falls.

**D5. Hierarchical sharding (HSDP)** (`eq_lm_compose_hsdp`). With a shard group of $g$ GPUs inside a node and $r$ replicas across nodes, the all-gathers and reduce-scatters of FSDP stay on the intra-node link and carry $\frac{g-1}{g}$ of the bytes of 6.2; across nodes only an all-reduce of each rank's gradient shard remains: $2\frac{r-1}{r}\cdot b_rP/g$ bytes per GPU. *Steps:* (1) shard over the node; (2) each GPU owns $1/g$ of the parameters; (3) replicas of the same shard all-reduce it. *Check:* bytes by `comm_events` on an HSDP run (the 2-node run if the model ranks one in its top 3, otherwise 6.8's one-node 2-replica HSDP run, where the shard and replica groups are still separate groups) equal the formula within 2%; the cost model predicts when HSDP beats flat FSDP across nodes (inter-node $\beta$ lower than intra-node by more than the factor that makes the extra all-reduce cheaper than the larger all-gathers).

#### From-scratch implementation

≤ 150 lines (T0, no GPU):
- `MeshSpec` and `make_mesh` (≈ 40 lines): rank array of shape `(pp, dp_replicate, dp_shard, cp, tp)` with `tp` fastest; group extraction by fixing the other coordinates; node span by `rank // gpus_per_node`; link class; ordering validation; flattening of axes for EP.
- `enumerate_configs` (≈ 30 lines): divisor loops, constraint checks with reasons.
- `ParallelCostModel.evaluate` (≈ 70 lines): composes the section cost functions per E1–E7; the pipeline bubble from `PipelineSim` with calibrated per-stage times (cached by $(p, m, v, \text{partition})$).
- *Tests.* (1) `make_mesh` on shapes $(\text{pp},\text{dp},\text{cp},\text{tp})\in\{(2,4,1,4), (1,2,2,8), (4,1,1,8)\}$ on 32 GPUs: every group has the right size; groups of an axis partition the ranks; intra-node classification equals the stride rule of 6.1 D3; a swapped order is rejected. (2) 2-rank gloo: `real=True` mesh groups equal the spec's (rank lists). (3) D2 counts. (4) A single-axis special case: with $t=p=c=1$ the model's FSDP prediction equals `fsdp_costs` and the 6.2 headline within the calibration error on the committed one-node runs. (5) Determinism: `rank()` returns the same order for the same inputs.

#### In production

- **TorchTitan** is the executed stack: a user module `d2l_recipes.py` defines one function per config, derived from the verified recipe (`config = llama3_8b(); config.parallelism.tensor_parallel_degree = 4; ...`, as the README's configuration guide shows; fields for pipeline/context/replicate degrees to be taken from the pinned `ParallelismConfig`), launched through `d2l.launch` with the SLURM template of 6.1 and `MODULE=d2l_recipes CONFIG=…` (or `--module/--config`); its fake-backend dry run (README, "Fake-backend dry runs of large layouts on one GPU") is used on the build box to validate that a layout builds before cluster time is spent. TorchTitan's own metrics (loss, memory, throughput, TFLOPs, MFU) are parsed from its logs and cross-checked against `d2l.mfu`.
- **Library table** (checked sources in parentheses): *TorchTitan*: FSDP2/HSDP/DDP, TP (including async TP), PP (1F1B, interleaved 1F1B, GPipe, other `torch.distributed.pipelining` schedules), CP (all-gather KV and Ulysses), EP with DeepEP/HybridEP dispatch, async distributed checkpointing, Python recipes (README 2026-10). *Megatron-Core*: parallelism strategies guide, mixture of experts, context parallel, Megatron-FSDP, distributed optimizer, optimizer CPU offload, custom pipeline layouts, fine-grained activation offloading (docs table of contents, 2026-10). *DeepSpeed*: ZeRO, ZeRO-Infinity, 3D parallelism, Ulysses sequence parallelism, DeepSpeed-MoE, AutoTP, long-sequence training (README 2026-10). *nanotron*: 3D parallelism (DP+TP+PP), expert parallelism, AFAB and 1F1B schedules, ZeRO-1; roadmap lists ZeRO-3, ring attention and interleaved 1F1B (README 2026-10). Each row gets "reach for it when" from the Ultra-Scale Playbook and the sources above; no performance claims about any library.
- The cost model and `make_mesh` are the oracle for the TorchTitan layouts: before each run the notebook builds the `MeshSpec` and compares its group lists with the `DeviceMesh` TorchTitan creates (rank lists printed from rank logs).

#### Headline experiment

- **Hypothesis (pre-registered).** (H1) The cost model, calibrated on one node only, ranks the measured 2-node configurations (and the 4-node ones if the optional pass ran) with the predicted top-1 inside the measured top-2 and a positive rank correlation; (H2) a TP group that crosses nodes is predicted and measured to be the slowest layout; (H3) the MFU of the best layout falls from 1 to 2 nodes (to 4 with the optional pass) at fixed global batch by an amount the waterfall attributes mainly to exposed FSDP communication and bubble, not GEMM efficiency. The notebook reports whichever way each comes out.
- **Setup.** 7–8B dense model (above), BF16 params/activations with fp32 reductions as in 6.2, AdamW, full activation checkpointing unless a layout needs none, sequence 4096, global batch 512 sequences (2,097,152 tokens: 6.7's batch only, since the other sections' batches are per-node settings; exactly 4× R2's 524,288 tokens per step in 4.2, a design setting in the range `sec_batch_size` gives for language models, and not a claim about this model's critical batch), synthetic or small-corpus data, 30 steps with 10 warm-up, seeds fixed, the same nodes for every layout of one scale (hostnames logged). 2 nodes = 16 GPUs (core), 4 nodes = 32 GPUs (optional).
- **Procedure.** (i) Calibration from the one-node bundles of 6.2–6.5. (ii) Predictions frozen and committed. (iii) Runs, core, 2 nodes (6 runs): the top-3 predicted layouts (an HSDP layout is one of them only if the model ranks it there), the predicted-worst (TP=16 across nodes), an FSDP-only baseline, and a repeat of the top-1 layout submitted last (drift check); plus the scaling reference on 1 node (2 runs: the best layout and FSDP-only at the same global batch, by accumulation). Optional, 4 nodes (4 runs): the top-2 predicted, the predicted-worst (TP=16 or PP over a slow axis) and FSDP-only; optional EP=16 (1 run, narrative item 8). **Cut** from the earlier plan (about 64 GPU-h): the 4-node layouts as core, the third predicted layout at 4 nodes, the dedicated HSDP run, the weak-scaling series on 1, 2 and 4 nodes, and the second interleaved pair of the noise check; each is either not needed to compare the cost model with measurement or is replaced by a model-only statement. (iv) Analysis from committed logs.
- **Metrics.** Step time (median of steps 10–29, IQR), TorchTitan MFU and `d2l.mfu(..., n_gpus=N)` (4.2; must agree, with the same `T` and `causal_half` convention), peak memory, per-axis exposed time (one profiled run per layout), predicted vs measured errors, Kendall's $\tau$ with a bootstrap interval over steps.
- **Plots.** `fig_lm_compose_rank`: predicted vs measured step time per layout (log–log, identity line, labels with the layout tuple), one panel per scale. `fig_lm_compose_waterfall`: MFU waterfall for best and worst layouts. `fig_lm_compose_scaling`: MFU vs GPUs for strong and weak scaling with Llama 3 Table 4 rows and MegaScale as annotated context. `fig_lm_compose_sensitivity`: best layout and predicted MFU over sequence length and inter-node bandwidth. `fig_lm_compose_mesh`: `render()` of the winning layout on 2 nodes (4 if the optional pass ran).
- **Expected outcome.** Direction only; magnitudes "to be measured". Reference points for orientation: 38–43% BF16 MFU for Llama 3 405B at 8,192–16,384 GPUs (2407.21783, Table 4) and 55.2% for a 175B model at 12,288 GPUs (2402.15627); an 8B model on 16–32 GPUs is a different regime and the notebook says it is not comparable.
- **Noise check.** 20 steady-state steps per layout; adjacent layouts are called different only if their median step-time intervals do not overlap; the top-1 layout is submitted first and again last (A, B, C, …, A) so node and time drift show up as a difference between repeats; any difference smaller than the repeat difference is reported as a tie; node-level slowness is checked from per-rank step times (6.8's detector).
- **T1 reduction.** Cost model, mesh tests and the calibration method on the build box; the predicted ranking on a *synthetic* cluster description labelled as such.
- **Committed (T4).** `predictions.json` (hash-stamped), recipes, bundles for every layout that ran (core: 8; optional: the 4-node and EP=16 bundles), the scaling reference, calibration tables (`lm-matmul-eff`), and `lm-mfu-table` (measured MFU per configuration, with the convention string).

#### Figures and tables

`fig_lm_compose_mesh`, `fig_lm_compose_rank`, `fig_lm_compose_waterfall`, `fig_lm_compose_scaling`, `fig_lm_compose_sensitivity`; `tab_lm_compose_axes`, `tab_lm_compose_constraints`, `tab_lm_compose_model` (E1–E7 with assumptions), `tab_lm_compose_libraries`, `tab_lm_compose_pred_meas`, `tab_lm_compose_ref_points` (Llama 3 Table 4 rows and MegaScale with their conventions).

#### Pitfalls and misconceptions

- Tuning the cost model on the layouts it is then judged on (hence the one-node calibration and pre-registration).
- Mixing MFU conventions (causal attention counted or not; recompute counted as model FLOPs or not; embedding and head included or not).
- Comparing layouts at different global batches.
- A layout that needs more microbatches than the global batch provides.
- Reading a layout's MFU from a single run on different nodes.
- Treating "TP ≤ 8" as a law rather than as an outcome of D1 with NVLink-class intra-node and slower inter-node links.
- Forgetting the optimizer step, data loading and checkpointing in timed regions.
- Believing the best layout at one scale is best at another.

#### Summary

- A parallel layout is a mesh: each axis has a communication volume, an exposed fraction and a link; the exposed volume decides which axis goes on the fastest link.
- The number of layouts is a stars-and-bars count; divisibility, microbatch and memory constraints prune it to a handful.
- MFU is a product of GEMM efficiency, bubble, recompute and exposed communication; a waterfall assigns every lost percent to a cause.
- A cost model calibrated on one node and frozen before the cluster runs is a test of understanding, not a fit.
- At a fixed global batch, the useful GPU count is capped by tokens per replica; TP, PP and CP raise the cap at a price each section has quantified.

#### Exercises

1. *(check)* For 64 GPUs, how many ordered (dp, tp, pp, cp) factorizations exist, and how many have tp ≤ 8 and pp ≤ 4? **Sketch:** $\binom{6+3}{3}=84$; count with the constraints by enumeration.
2. *(check)* Why is a mesh axis order of (dp, pp, tp) with tp *last* correct, and what goes wrong with (tp, dp, pp)? **Sketch:** row-major: the last axis has stride 1, hence adjacent ranks (one node); with tp first its stride is large and TP groups span nodes (6.1 D3).
3. *(derive)* Prove the rearrangement argument of D1 for three axes. **Sketch:** adjacent transposition changes the sum by $(U_i-U_j)(1/\beta_j-1/\beta_i)\ge0$ when out of order.
4. *(derive)* Derive the largest $N$ for strong scaling of an 8B model at $B=2^{21}$ tokens with $t=4$, $p=2$, $c=1$, using the $T^*$ of 6.2 (value taken from the notebook). **Sketch:** $N_{\max}=t\,p\,c\,B/T^*$.
5. *(measure)* Before looking at the logs, rank the three top predicted layouts at 2 nodes by your own intuition; compare with the model and with the measurement; where did the model or you err?
6. *(measure)* Perturb the model's inter-node bandwidth by ±50% and report which layouts change rank; then re-run the cross-node TP=16 layout with `NCCL_IB_DISABLE=1`-like degradation if allocation time permits (or use the committed socket run of 6.1) and compare with the prediction.
7. *(open)* Extend `ParallelCostModel` to a MoE shape (EP axis, node-limited routing from 6.6) and predict the best layout for Qwen3-30B-A3B on 4 nodes; state which measurement would falsify it.

#### Further reading

- Primary: [Efficient Large-Scale Training](https://arxiv.org/abs/2104.04473) (the combination of TP, PP, DP, interleaving, and its topology-aware guidance); [TorchTitan](https://arxiv.org/abs/2410.06511); the Llama 3 paper section 3.3 (4D parallelism order [TP, CP, PP, DP], Table 4, memory and compute estimators; [2407.21783](https://arxiv.org/abs/2407.21783)); [MegaScale](https://arxiv.org/abs/2402.15627) (MFU at 12,288 GPUs and its stability measures). New, title verified 2026-10-28: [Alpa](https://arxiv.org/abs/2201.12023) (automatic search over inter- and intra-operator parallelism; the cost-model-and-search counterpart to this section's enumeration).
- Teaching: [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) (its measured sweeps are the qualitative check of the model); [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) (the analytic treatment of mesh axes); TorchTitan's [`torchtitan/distributed/README.md`](https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/README.md) (200 OK 2026-10-28).
- Code: [torchtitan](https://github.com/pytorch/torchtitan); [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and its [Core guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html); [DeepSpeed](https://github.com/deepspeedai/DeepSpeed); [nanotron](https://github.com/huggingface/nanotron) (all 200 OK 2026-10-28).

#### Risks and open decisions

- **TorchTitan API and recipe drift** (Python recipes are recent; PP/CP support differs per model). Fallback: pin a commit, use the verified `llama3_8b` recipe if the Qwen3-8B recipe lacks PP or CP; the cost model takes whichever `ModelShape`.
- **Allocation size and time.** T4 is a handful of runs: the core needs 2 nodes for ~1 h per release (plus 1 node for two short runs); the 4-node pass and the EP=16 run are optional. Fallback: core only (the notebook still ranks and compares), and any 4-node panel is shown from a previous release's bundle with its date, or omitted.
- **Tokenizer or data access** for the TorchTitan recipe (gated Llama tokenizer; dataset download on cluster nodes). Fallback: Qwen3 tokenizer, a locally staged corpus slice.
- **Cross-node TP run** may fail for memory reasons or be unsafe for the allocation; it is run with a short step count and a timeout.
- **Overfitting the model to the cluster.** Mitigation: pre-registration; report all layouts; publish the model's residuals.
- **MFU convention** may differ from TorchTitan's built-in; the notebook computes both and prints the ratio.
- **Reference global batch (closed 2026-10-29).** The 2.1 M-token batch applies to 6.7 only and is 4× R2's step; the per-node batches of 6.2–6.6 and 6.8 are labelled as such (Common conventions). No reconciliation with the critical batch size is claimed.

### 6.8 Checkpoints, failures and long runs

- **Label:** `sec_lm_long_runs` · **Tier:** T3 (derivations and failure-rate fits are T0) · reduced T1 config: 2 gloo ranks, one killed, restart on 1 rank, tiny `d2l.TransformerBlock` model, float64 · **Budget:** T3 ≤ 1.2 h wall on 8× A100 (≈ 10 GPU-h, planned: reference run, kill-and-reshard 8→4, async-vs-sync timing, straggler and corruption injection); peak memory below the 6.2 baseline (12 layers of the 8B width: 32.0 GB additive bound for the 4-GPU resume, computed from 6.2's D1 terms against the 36 GB budget; the full 36-layer shape would need 52.1 GB at $k=4$ and does not fit 40 GB parts); T0 cells < 2 min; T1 ≈ 3 min · **Codex units:** 7.7 (operational reliability), plus 5.9's resume discipline
- **Builds on (book):** `sec_training_systems` (`subsec_training_long_run`: atomic checkpoints via temp file and `os.replace`, the resume-parity exercise, "drill the recovery"; keep, extend to sharded and asynchronous), `sec_read_write` (safetensors, state dicts), `sec_repro` (seeds, determinism limits), `sec_adamw`/`sec_practice` (clipping, EMA state belongs in the checkpoint).
- **Builds on (this part):** 5.3 (measured checkpoint write bandwidth, synchronous vs asynchronous stall of an 8B state; reads the checkpoint write-bandwidth and stall records from `lm-hw-node-storage`; fallback: 6.8 measures its own if a record for the 8B state is missing); 6.1 (`launch`, `rank_log`, kill and timeouts, flight recorder); 6.2 (`fsdp_wrap`, `FSDPLite`, `fsdp_costs`); 4.1 (`d2l.PackedStream` data cursor; its `state_dict()` stores the global consumed count and reloads under a different world size; fallback: `d2l.CursorSampler` below); 4.3 (loss-spike diagnosis is owned there; 6.8 holds only the run-level playbook).
- **Exports:**
  - `d2l.failure_rate(gpus, source='llama3') -> float`: expected failures per hour for a job of `gpus` GPUs, $N_{\text{nodes}}\,r_f/24$ with $N_{\text{nodes}}=\texttt{gpus}/8$ and $r_f$ per node-day. `source` is `'llama3'` (the default), `'rsc1'` or `'rsc2'`, or a float taken as $r_f$ in failures per 1,000 node-days. The three tabulated rates are: Llama 3, 419 unexpected interruptions in a 54-day snapshot on 16,384 GPUs (2,048 nodes of 8): 3.79 per 1,000 node-days, 1.97e-5 per GPU-hour (arithmetic in D1); `'rsc1'`, 6.50 per 1,000 node-days (3.39e-5 per GPU-hour); `'rsc2'`, 2.34 (1.22e-5 per GPU-hour). `failure_rate(16384)` with the default source is $2048\times3.79\times10^{-3}/24=0.323$ per hour, an MTBF of 3.09 h.
  - `d2l.young_daly(delta, mtbf, restart=0.0, order='daly'|'young'|'exact') -> tau`: compute time between checkpoints; `d2l.expected_waste(tau, delta, mtbf, restart)` the exact waste fraction.
  - `d2l.simulate_run(work, tau, delta, restart, mtbf, runs=10000, seed=0) -> DataFrame`: Monte Carlo of wall-clock time with exponential failures.
  - `d2l.CursorSampler(num_samples, global_batch, seed, rank, world, global_consumed=0)`: deterministic global order; rank $r$ of $W$ takes positions $r, r+W, \dots$ of each global batch; `.state_dict()` stores `global_consumed` only.
  - `d2l.validate_checkpoint(path) -> list[str]`: problems found (missing commit marker, missing or short shard files, manifest checksum mismatch); empty list means loadable.
  - `d2l.StragglerDetector(window=50, k=5.0, consecutive=5)`: per-rank robust z-score (median/MAD) of per-step compute time, `.update(times) -> list[int]` of flagged ranks.
  - `d2l.replica_checksum(tensors, group) -> (digest, all_equal)`: 64-bit hash of tensors, all-gathered across a replica group.
- **Models and data:** R2-scale `d2l.GPT` for T1; the Qwen3-8B width at 12 layers for T3 (`ModelShape` with `L=12`, $P$ = 3.56 B; random init, 6.2 FSDP2 settings; training state of fp32 parameters plus Adam moments = 12 bytes per parameter ≈ 42.7 GB total, 5.3 GB per GPU at $k=8$ and 10.7 GB at $k=4$, computed) so that the 8→4 resume fits 40 GB; the full 8B state (≈ 98 GB, 12.3 GB per GPU at $k=8$) is the one 5.3 measures, and D3 uses 5.3's bandwidths for either size. Published tables transcribed into `lm-failure-tables` (CSV with source lines): Llama 3 Table 5 (2407.21783, with the discrepancy note of D1), RSC-1/RSC-2 rates (2410.21680), OPT-175B counts (2205.01068).
- **Packages:** `torch==2.15.0` `torch.distributed.checkpoint` (`save`, `load`, `async_save` returning `AsyncSaveResponse` with `staging_completion`/`upload_completion`, `FileSystemWriter`, `DefaultStager`, `Stateful`, `state_dict_saver`; names verified in the 2.15 docs), `torch.distributed.checkpoint.state_dict` (`get_state_dict`, `set_state_dict`, `StateDictOptions`), `scipy` (chi-square quantiles; fallback normal approximation), optional `torchft`.

#### Narrative arc

1. **The question.** Opening cell: a 54-day Llama 3 405B pre-training snapshot with 466 interruptions, 419 unexpected, on a 16K-GPU job (2407.21783, 3.3.4). The cell prints the implied job MTBF (54 × 24 / 419 = 3.1 h, which is `failure_rate(16384)` under the default source) and asks: with that MTBF, how often should a run checkpoint, and what does each checkpoint cost? Then the T3 symptom: kill rank 5 of an 8-GPU run at step 60 and restart; the committed log shows the run resuming from step 50, repeating 10 steps.
2. **What fails and how often** (D1): the three sources side by side (`tab_lm_failure_sources`, Llama 3 first because it is the default of `failure_rate`): Llama 3 by component (counts as printed; the notebook recomputes percentages from counts and records the table's internal inconsistency, D1), RSC-1 and RSC-2 ($r_f = 6.50$ and $2.34$ per 1000 node-days, 2410.21680 Fig. 7 legend and the paper's text), OPT-175B (at least 35 manual restarts plus an estimated 70+ automatic ones over about two months on 992 A100 80 GB GPUs, 2205.01068); MegaScale's stragglers. Caveats: definitions of "failure", exposure, selection of the snapshot window.
3. **From rate to job MTBF** (6.1 D2 recalled): $M(N)=1/(N_{\text{nodes}}r_f)$ and the paper's projections (1.8 h at 16,384 GPUs; 0.23 h at 131,072). Cell: `failure_rate(gpus)`; fit per-cause rates with exact Poisson intervals; compare Llama 3's $3.79$ per 1000 node-days (419 / (54 × 2048 nodes), with 16,384 GPUs from Table 4 and eight GPUs per server from §3.3.1 of the paper; it lies between RSC-2's 2.34 and RSC-1's 6.50, at 0.58× RSC-1 and 1.62× RSC-2, computed) with RSC-1 and RSC-2; note RSC's empirical small-job MTBFs (47.7 days at 8 GPUs, 7.9 h at 1,024) lie below the $1/N$ line.
4. **How often to checkpoint** (D2): Young–Daly, its exact form and Daly's higher-order correction; waste curve; flatness of the optimum; Monte Carlo check; worked table for 64, 1,024, 8,192, 16,384 and 131,072 GPUs with $\delta$ from 5.3 (sync and async).
5. **What a checkpoint is** (D3): the state to save (parameters, optimizer, scheduler, step, RNG streams per rank, data cursor, EMA, loss-scaler); sizes; atomicity (the book's temp-and-rename recipe lifted to a directory of per-rank shards plus a manifest and a commit marker, written last by rank 0 after a barrier); retention (keep the last $k$ and a sparse archive).
6. **Distributed checkpoints** (`torch.distributed.checkpoint`): `.metadata` plus one file per rank; tensors saved as chunks with offsets; load plans map chunks to the new layout. Cell: save an FSDP2 state on 4 gloo ranks, inspect the metadata, load on 2 ranks and on 1.
7. **Asynchronous saves** (D3): `async_save` stages to pinned host memory (the blocking part), then persists in a thread or process; conditions for back-pressure. Cell: measured stall and total time, sync versus async, against $S_{\text{gpu}}/\beta_{h2d}$; why the process group for the background collective must be gloo.
8. **Resharding and the data cursor** (D4): the kill-and-restart experiment on a different number of GPUs; the invariant that a global step's sample multiset does not depend on $W$.
9. **Detecting and replacing failed ranks.** Watchdog timeouts and the heartbeat (`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC`), torchrun's `--max-restarts` and elastic ranges, SLURM requeue and spare nodes, pre-flight health checks; fault-tolerant data parallelism with torchft (replica groups, quorum, live healing from a peer, restart-free for the survivors) and its cost model (D6); Streaming DiLoCo and Oobleck as pointers.
10. **Stragglers** (D5) and **silent data corruption** (D7): per-rank compute-time telemetry and robust detection; replica checksums and their limits. Injection experiments.
11. **Run hygiene**: the alert list (NaN/inf, loss spike against an EMA band, gradient-norm spike, step-time regression, throughput drop, ECC/XID counters, checkpoint age), the loss-spike playbook from the literature (OPT: lower the learning rate and restart from an earlier checkpoint; PaLM: restart from a checkpoint about 100 steps before the spike and skip about 200–500 batches, 2204.02311), `run_manifest.json` (git hash, config hash, data-manifest hash from 3.x, seeds, world size and layout, library versions, hostnames, checkpoint lineage). This is the retiring appendix's "Keeping a Long Run Alive" brought to scale.

#### Derivations

**D1. Failure rates and job MTBF** (`eq_lm_fail_rate`).
- *Statement.* Observing $n$ failures over exposure $E$ (GPU-hours or node-days) of a homogeneous Poisson process, the maximum-likelihood rate is $\hat\lambda=n/E$ with exact two-sided interval $\big[\chi^2_{\alpha/2}(2n)/2E,\ \chi^2_{1-\alpha/2}(2n+2)/2E\big]$; the MTBF of an $N$-node job is $1/(N\hat\lambda_{\text{node}})$; per-cause shares are multinomial.
- *Steps.* (1) Likelihood $\lambda^ne^{-\lambda E}$. (2) Maximize: $n/E$. (3) The count's distribution gives the Garwood interval by the gamma–Poisson relation. (4) Superposition of independent node processes sums rates. (5) For Llama 3, the default source: $n=419$ unexpected interruptions in 54 days on 16,384 GPUs (Table 4 lists 16,384 GPUs; §3.3.1 gives eight GPUs per server, so 2,048 nodes: the snapshot's node count is an assumption of this blueprint). Node-days $E=54\times2{,}048=110{,}592$, so $\hat\lambda=419/110{,}592=3.79\times10^{-3}$ per node-day, i.e. 3.79 per 1,000 node-days. GPU-hours $E=54\times24\times16{,}384=21{,}233{,}664$, so $\hat\lambda=419/21{,}233{,}664=1.97\times10^{-5}$ per GPU-hour. Job MTBF at 16,384 GPUs: $54\times24/419=3.09$ h. The count's interval is about $\pm10\%$ (normal approximation $1.96/\sqrt{419}=9.6\%$). For the other two sources the same conversion gives RSC-1: $6.50/1000/(8\times24)=3.39\times10^{-5}$ per GPU-hour and an MTBF of 1.80 h at 16,384 GPUs; RSC-2: $2.34/1000/192=1.22\times10^{-5}$ and 5.01 h (all computed). Definitions differ across the sources: Llama 3 counts every unexpected interruption of one job (hardware, software and unplanned maintenance), RSC counts failures of jobs above 128 GPUs by the paper's attribution.
- *Data check (verified 2026-10-29 against the rendered page 13 of arXiv 2407.21783, not only the extracted text).* Table 5's 18 counts sum to 419 only with Faulty GPU = 148, whose share is $148/419=35.3\%$, while the table prints 30.1% for that row (which equals $126/419$). The printed percentages sum to 94.9%, not 100%; every other row's printed percentage equals count/419 to rounding. The text's "GPU issues ... 58.7%" equals the sum of the six printed GPU-row percentages ($30.1+17.2+4.5+4.1+1.4+1.4$), while the counts give $268/419=64.0\%$ ($148+72+19+17+6+6$). So either the percentage column (and the 58.7%) predates the final Faulty GPU count, or the 148 is wrong and 22 interruptions are unlisted; the text says all 419 are classified in the table, which favours the first reading, but the PDF alone cannot decide. The notebook recomputes shares from counts, prints both versions, and states this. The failure rate does not depend on the split: it uses only 419 and 54 days.
- *Check.* Fit and interval cells; the projections 1.8 h and 0.23 h of 2410.21680 are reproduced from $r_f=6.50$ per 1000 node-days (cell asserts within rounding).

**D2. Young–Daly interval** (`eq_lm_yd`).
- *Statement.* Checkpoint every $\tau$ seconds of compute at cost $\delta$, restart cost $R$, exponential failures with mean $M$. Expected wall time for $T_s$ seconds of work is $T_{\text{wall}}=\dfrac{T_s}{\tau}\,M\,e^{R/M}\big(e^{(\tau+\delta)/M}-1\big)$. Its minimizer satisfies $e^{(\tau+\delta)/M}(1-\tau/M)=1$. To first order $\tau^*\approx\sqrt{2\delta M}$ (Young 1974) and, with higher-order terms (Daly 2006), for $\delta<2M$: $\tau^*=\sqrt{2\delta M}\Big[1+\tfrac13\sqrt{\tfrac{\delta}{2M}}+\tfrac19\tfrac{\delta}{2M}\Big]-\delta$, and $\tau^*=M$ otherwise.
- *Assumptions.* Failures memoryless and independent of the checkpoint phase; failures may occur during checkpoint and restart; the checkpoint is blocking for $\delta$ seconds; work between failures is lost back to the last completed checkpoint.
- *Steps.* (1) Segment length $L=\tau+\delta$. (2) Let $T(L)$ be the expected time to complete a segment. With probability $e^{-L/M}$ it finishes; otherwise it fails at time $t<L$ (density $e^{-t/M}/M$), wastes $t$, pays an expected restart $E_R=M(e^{R/M}-1)$ (restarts can fail too), and starts over. (3) $T(L)=Le^{-L/M}+\int_0^L\frac1Me^{-t/M}(t+E_R+T(L))\,dt$. (4) Using $\int_0^L\frac tMe^{-t/M}dt=M(1-e^{-L/M})-Le^{-L/M}$, solving gives $T(L)=(e^{L/M}-1)(M+E_R)=Me^{R/M}(e^{L/M}-1)$. (5) There are $T_s/\tau$ segments. (6) Minimizing $f(\tau)=(e^{(\tau+\delta)/M}-1)/\tau$: $f'=0$ gives $e^{(\tau+\delta)/M}(1-\tau/M)=1$. (7) Expand for $\tau,\delta\ll M$: $\delta/M+(\delta^2-\tau^2)/(2M^2)\approx0$, so $\tau\approx\sqrt{2\delta M}$ at leading order; keeping the next terms gives Daly's correction. (8) Waste at the optimum, first order: $W(\tau)\approx\delta/\tau+\tau/2M+R/M$, minimized at $\sqrt{2\delta/M}+R/M$. (9) Flatness: at $\tau=1.3\tau^*$ the waste term rises by $\tfrac12(1/1.3+1.3)-1=3.5\%$ of its minimum.
- *Result.* Checkpoint rarely enough that overhead $\delta/\tau$ balances the expected loss $\tau/2M$; the optimum is flat; the optimal waste is $\sqrt{2\delta/M}$ before restarts.
- *Check.* Cell A: for $(\delta,M)\in\{(60,28800),(10,28800),(300,7200),(600,6480),(30,828),(900,1800)\}$ s the exact root (bisection), Young and Daly intervals were computed in the prototype used for this blueprint: exact 1819/752/1884/2404/203/1257 s, Young 1859/759/2079/2789/223/1800 s, Daly 1819/752/1883/2403/203/1250 s (the notebook recomputes them); Daly matches the exact root to within 0.6% in all six, Young is off by up to 16% when $\delta/M\ge0.05$. Cell B: `simulate_run` with $10^4$ runs agrees with $T_{\text{wall}}$ within 1% (standard error printed). Cell C: the worked table over 64–131,072 GPUs with $r_f$ from `failure_rate`, $\delta$ in {sync, async} from 5.3, $R=10$ min as a stated assumption; the "effective training time" $1-W$ at each size, compared against the "more than 90%" Llama 3 reports (3.3.4) and the sub-minute checkpoint overhead requirement discussed in 2410.21680 (which assumes Daly–Young with a stated restart cost).

**D3. Asynchronous checkpoint cost** (`eq_lm_async`).
- *Statement.* The blocking cost is the staging time $\delta_{\text{stage}}=S_{\text{gpu}}/\beta_{h2d}$ plus planning and a barrier; persistence $t_{\text{persist}}=S_{\text{tot}}/B_{\text{storage}}$ runs in the background and must satisfy $t_{\text{persist}}\le\tau$, otherwise the next save blocks. Young–Daly uses $\delta_{\text{eff}}=\delta_{\text{stage}}+\phi\,t_{\text{persist}}$ with $\phi$ the fractional training slowdown while persisting (measured).
- *Steps.* (1) Device-to-pinned-host copy of each GPU's shard. (2) Training resumes. (3) A thread or process writes the shard files. (4) Back-pressure if the previous persist is unfinished. (5) $\delta_{\text{eff}}$ enters D2.
- *Check.* Measured stall per save (sync and async) on the 8B state versus $S_{\text{gpu}}/\beta_{h2d}$ with 5.3's $\beta$; $\phi$ from step times during persist.

**D4. Resharding and the data cursor** (`eq_lm_reshard`).
- *Statement.* (a) A tensor saved as chunks at offsets can be loaded into any layout by reading, for each target shard, the set of overlapping chunks; for 1-D contiguous sharding of $n$ elements, going from $k$ to $k'$ shards, each new shard overlaps at most $\lceil k/k'\rceil+1$ old chunks. (b) With global batch $B$ divisible by $W$ and sample $g=sB+j$ assigned to rank $j \bmod W$, the multiset of samples of global step $s$ is $\{sB,\dots,sB+B-1\}$ for every $W$; the cursor is the single integer $sB$.
- *Steps.* (a) Interval overlap arithmetic. (b) Union over ranks of $\{j\equiv r \pmod W\}$ is all $j<B$.
- *Check.* Property test over $W\in\{1,2,4,8\}$; overlap counts from the arithmetic equal the number of read items in the `DefaultLoadPlanner` plan on CPU for $(8\to4)$, $(8\to6)$, $(8\to3)$ with `FileSystemReader`; after the T3 restart the first 10 global batches' sample ids equal the reference run's.

**D5. Stragglers** (`eq_lm_strag`).
- *Statement.* A synchronous step takes $\max_rt_r$; with i.i.d. noise of standard deviation $\sigma$ over $n$ ranks the expected maximum is about $\mu+\sigma\sqrt{2\ln n}$; one persistent straggler with extra time $s$ per step costs every rank $s$ per step. Step times are equal across ranks after the collective; the straggler is visible in the per-rank compute time (time between collectives) and in the others' wait time.
- *Steps.* (1) Extreme-value approximation. (2) Persistent excess adds to the critical path. (3) Robust $z_r=(t_r-\text{median})/(1.4826\,\text{MAD})$ per step. (4) Flag if $z_r>k$ for `consecutive` steps; for Gaussian noise the per-step false-positive probability of one rank is $1-\Phi(k)$ and of a run $(1-\Phi(k))^{\text{consecutive}}$.
- *Check.* Simulation: 64 ranks, 1,000 baseline steps with the measured per-rank jitter from the T3 reference run: false flags counted; then injected slowdowns of 5%, 10%, 20% on one rank: detection delay in steps; T3 injection on rank 3 (extra dummy matmul calibrated to a fixed fraction of compute): flagged rank equals injected rank. Reference observation: MegaScale found hosts about 10% slower in forward and reports roughly a 0.7% MFU gain from removing them (2402.15627).

**D6. Fault-tolerant replicas versus restart** (`eq_lm_ft`). Restart-based loss per failure: $\tau/2+R$ (all GPUs idle). With replica groups healed live (torchft), loss per failure is the time to transfer the state $S_{\text{rg}}$ to the replacement, $T_{\text{heal}}=S_{\text{rg}}/\beta$, during which the surviving groups continue at reduced capacity $(1-1/G)$ for $G$ groups. *Steps:* count lost GPU-time per failure in each scheme; ETTR$_{\text{restart}}=1-\delta/\tau-(\tau/2+R)/M$, ETTR$_{\text{ft}}\approx1-T_{\text{heal}}/(G\,M)$. *Check:* a T0 cell with Chapter 5's bandwidths; an optional T3 demonstration with torchft (two replica groups of 4 GPUs, kill one) whose recorded timeline shows the survivors continuing.

**D7. What replica checksums can and cannot detect** (`eq_lm_sdc`). *Statement:* with data-parallel replicas, identical updates require identical inputs; a corruption of stored state on one replica (parameters or optimizer moments after the step) changes that replica's hash at the next check; a corruption of a local gradient or activation *before* the gradient all-reduce is averaged into every replica identically and the hashes still agree. *Steps:* (1) deterministic all-reduce gives all ranks the same bits; (2) update is a deterministic function of reduced gradient and state; (3) hence state divergence implies post-reduce corruption; (4) pre-reduce corruption shows only in gradient-norm outliers of the offending rank (depending on the flipped bit's significance). *Check:* flip bit 20 of a parameter on rank 2: detected at the next check; flip the sign bit of a gradient element before the all-reduce: hashes agree, per-rank pre-reduce norm flags only for high-significance bits; overhead of the check measured (one 64-bit all-gather per $K$ steps). Llama 3 attributes 6 of 419 unexpected interruptions (1.4%) to silent data corruption; TorchTitan documents an SDC replay tool (`SILENT_DATA_CORRUPTION.md`).

#### From-scratch implementation

≤ 150 lines:
- `young_daly` and `expected_waste` (≈ 25 lines, with a bisection for the exact root); `simulate_run` (≈ 25 lines, vectorized exponential draws); `failure_rate` (≈ 10 lines).
- `CursorSampler` (≈ 20 lines); `validate_checkpoint` (≈ 25 lines: directory contains the commit marker `COMMITTED`, `.metadata`, every file named in `manifest.json` exists with matching size and CRC-32); atomic save wrapper (write to `step_N.tmp/`, barrier, rank 0 writes manifest and renames to `step_N/` then writes `COMMITTED`, updates `latest` by `os.replace`) (≈ 25 lines).
- `StragglerDetector` (≈ 20 lines); `replica_checksum` (≈ 12 lines).
- *Tests.* (1) `young_daly(order='exact')` satisfies the root equation to $10^{-9}$ relative; Daly within 1% of exact for $\delta/M\le0.2$ (six cases above); (2) `simulate_run` mean within 1% of $T_{\text{wall}}$; (3) `CursorSampler` unions equal across $W$, resume mid-epoch continues exactly; (4) `validate_checkpoint` rejects: missing `COMMITTED`, one shard file deleted, one truncated, a flipped byte (CRC); (5) float64 2-rank kill and restart on 1 rank: parameters after 10 further steps equal an uninterrupted 1-rank run started from the same checkpoint to $10^{-12}$; (6) detector false-flag rate on i.i.d. noise within the binomial prediction; (7) checksum flags the corrupted rank.

#### In production

- `torch.distributed.checkpoint`: `dcp.save({"app": AppState(model, optimizer)}, storage_writer=FileSystemWriter(path))` with a `Stateful` wrapper using `get_state_dict`/`set_state_dict`; `dcp.async_save(..., process_group=gloo_group)` returning staging and upload futures (call `.result()` on the previous response before the next save); `dcp.load` into a differently sharded model for resharding; metadata inspection with `FileSystemReader.read_metadata()`.
- TorchTitan's checkpointer (`torchtitan/components/checkpointer/README.md`): async saves, saving and loading Hugging Face safetensors directly, conversion scripts, `--resume-step`; its Grain data pipeline with checkpointable state; its Flight Recorder and SDC replay documents. torchft for restart-free replica groups. Megatron-Core `dist_checkpointing` for a different format (named only).
- Oracles: the float64 `validate_checkpoint` tests are the oracle for the production directory layout (the notebook runs it on a real DCP directory); the `CursorSampler` property is the oracle for production data cursors; the Monte Carlo is the oracle for the interval formulas.

#### Headline experiment

- **Hypothesis.** (H1) Young–Daly with Daly's correction predicts the waste-minimizing interval of a simulated run to within 1%, and the exact formula predicts simulated wall time within 1%. (H2) A run killed at step 60 and restarted from the step-50 checkpoint on 4 GPUs instead of 8 reproduces the reference loss trajectory for steps 51–70 within a band set by same-layout reruns and consumes exactly the sample ids of the reference. (H3) The straggler detector flags the injected rank and no others in the baseline.
- **Setup (T3).** The 12-layer Qwen3-8B width (Models and data), FSDP2 with 6.2's W1 settings (chunked loss), 8× A100 (40 GB per GPU assumed), global batch 16 sequences of 4096 tokens (65,536 tokens, a per-node setting; 8,192 tokens per GPU on 8 GPUs, and on 4 GPUs two accumulated microbatches of two sequences so that per-GPU memory matches), 80 steps, checkpoint every 25 steps (async) plus one sync checkpoint for timing, the `CursorSampler` over a fixed synthetic corpus of random token sequences with a seeded permutation (so sample ids are checkable), seed 0.
- **Procedure.** (i) Reference: 8 GPUs, steps 1–80, loss and per-step sample ids logged. (ii) Kill: rerun with `os.kill(os.getpid(), SIGKILL)` on rank 5 at step 60 (after the step-50 checkpoint; `d2l.launch(..., max_restarts=0)` returns `ok=False` with the failing rank in `errors`); (iii) Resume on 4 GPUs from step 50 (DCP resharding), steps 51–80; (iv) control: reference restarted from step 50 on 4 GPUs without a kill, and a second 8-GPU same-layout rerun for the noise band. (v) Timings: sync vs async checkpoint stall and total. (vi) Straggler injection on rank 3 (5%, 10%, 20% of compute); detector output. (vii) Corruption injection per D7 on a 2-replica HSDP config.
- **Metrics.** Max and mean absolute loss difference per step between runs; set equality of sample ids per step; stall seconds (sync, async); detector flags and delay; checksum detection; checkpoint size.
- **Plots.** `fig_lm_long_waste`: waste vs $\tau$ (exact, Young, Daly, Monte Carlo points) for two $(\delta, M)$. `fig_lm_long_interval`: optimal $\tau$ and waste vs job size from 64 to 131,072 GPUs for sync and async $\delta$. `fig_lm_long_loss`: reference, resumed-on-4 and control trajectories with the noise band. `fig_lm_long_timeline`: per-rank timeline around the kill and restart. `fig_lm_long_straggler`: detector $z$ for the injected rank vs steps.
- **Expected outcome.** Direction: async stall much shorter than sync (magnitudes from 5.3's measurement and this run, "to be measured"); resumed trajectory within the same-layout noise band up to non-associativity of reductions with a different world size; no false straggler flags. Published context: 5.3-style checkpoint costs and Llama 3's "higher than 90% effective training time" (2407.21783), MegaScale's "over 90%" with recovery inside 15 minutes (2402.15627).
- **Noise check.** The same-layout rerun defines the noise band $\varepsilon$; the resharded run passes if its max per-step loss difference is at most $3\varepsilon$ plus the loss-print precision (the factor 3 is a stated convention); sample-id equality is exact, no tolerance. Detector false flags counted over 1,000 baseline steps.
- **T1 reduction.** Two gloo ranks, kill rank 1, restart on one; all tests of "From scratch"; no timings.
- **Committed (T3).** `lm-ckpt-reshard` bundles (reference, killed, resumed, control, reruns), checkpoint metadata and manifest (not the tensors), straggler and corruption logs, `lm-failure-tables`.

#### Figures and tables

`fig_lm_long_waste`, `fig_lm_long_interval`, `fig_lm_long_loss`, `fig_lm_long_timeline`, `fig_lm_long_straggler`, `fig_lm_long_ckpt_layout` (files of a DCP directory and what each holds); `tab_lm_failure_sources` (all three sources: basis, rate per 1,000 node-days and per GPU-hour, MTBF at 16,384 GPUs; Llama 3 first, marked default), `tab_lm_failure_llama3` (counts, recomputed shares, the discrepancy), `tab_lm_long_yd` (interval and waste per job size), `tab_lm_long_alerts` (alert, threshold rule, action), `tab_lm_long_checks` (cheap consistency checks and what each can detect).

#### Pitfalls and misconceptions

- Checkpoint interval chosen by habit ("every 1000 steps") rather than from $\delta$ and $M$; optimal waste scales as $\sqrt{\delta/M}$, so halving $\delta$ helps by $1/\sqrt2$ only.
- Counting the async checkpoint as free: staging blocks, persistence competes for bandwidth.
- A checkpoint directory without an atomic commit: a partial save is silently picked up as "latest".
- Resuming with a different world size and a per-rank data cursor: duplicated or skipped samples.
- Forgetting optimizer, scheduler, RNG, EMA and loss-scaler state.
- Reading equal step times across ranks as "no straggler" (they are equal after the collective).
- Expecting replica hashes to catch every corruption (D7).
- Short collective timeouts that kill a run during a slow but legitimate checkpoint.
- Published failure rates transplanted without regard to hardware generation, firmware and workload.

#### Summary

- Failures scale with job size; published tables give per-node rates of a few per 1000 node-days, hence MTBFs of hours at 10K+ GPUs.
- Checkpoint every $\sqrt{2\delta M}$ (with Daly's correction when $\delta/M$ is not small); the optimum is flat and its waste is $\sqrt{2\delta/M}$.
- Distributed checkpoints store chunks with offsets, so a run restarts on a different world size; the data cursor must be a global count.
- Asynchronous saves cut the blocking part to the staging copy; commit markers and manifests make partial writes detectable.
- Stragglers show in per-rank compute time, not in step time; replica checksums catch only post-reduce corruption.

#### Exercises

1. *(check)* A job has MTBF 3 h and $\delta=30$ s. Estimate the Young interval and the waste at the optimum. **Sketch:** $\sqrt{2\cdot30\cdot10800}=805$ s; waste $\approx\sqrt{2\cdot30/10800}=7.5\%$ before restart costs.
2. *(check)* Why must `COMMITTED` be written after every shard file and the manifest? **Sketch:** a reader that sees the marker may assume all data are present; the order makes a crash leave either no marker or a complete directory.
3. *(derive)* Prove the first-order optimum $\sqrt{2\delta M}$ from $W(\tau)=\delta/\tau+\tau/2M$ and derive the effect of an unreliable restart (restart can fail) on $T(L)$. **Sketch:** $W'=0$; $E_R=M(e^{R/M}-1)$ (D2 step 2).
4. *(derive)* For the property of D4(b), show that a per-rank cursor (position within each rank's own shard) breaks when $W$ changes unless $sB$ is a multiple of both $W$ and $W'$, and give the counterexample $B=24$, $W=8\to6$ at step 1. **Sketch:** per-rank counts 3 vs 4 samples per rank shift the interleaving.
5. *(measure)* Measure async stall versus state size by saving 1, 2, 4, 8 GPUs' worth of 8B state; predict $\delta_{\text{stage}}$ from $S_{\text{gpu}}/\beta_{h2d}$ and compare. Where does the prediction fail?
6. *(measure)* Inject straggler slowdowns of 2%, 5%, 10% and find the smallest the detector flags within 20 steps; relate to $k$ and the baseline jitter; compute the step-time cost the undetected straggler imposed.
7. *(open)* Design a restart-free policy for a 4-replica-group HSDP job with torchft: what is saved, what is healed from peers, what does a failure cost compared with D2, and what failures does it not cover (a failure during healing, correlated rack failures)?

#### Further reading

- Primary: Young (CACM 1974) and Daly (FGCS 2006), cited by title; [Llama 3](https://arxiv.org/abs/2407.21783) (3.3.4: interruptions, effective training time, flight recorder, stragglers); [Revisiting Reliability in Large-Scale Machine Learning Research Clusters](https://arxiv.org/abs/2410.21680); [MegaScale](https://arxiv.org/abs/2402.15627) (checkpointing and recovery, stragglers). New, titles verified 2026-10-28: [OPT](https://arxiv.org/abs/2205.01068) (hardware failures, restarts and loss divergences; logbook); [PaLM](https://arxiv.org/abs/2204.02311) (loss-spike mitigation); [ByteCheckpoint](https://arxiv.org/abs/2407.20143) (unified checkpointing and resharding); [Silent Data Corruptions at Scale](https://arxiv.org/abs/2102.11245); [Characterization of LLM Development in the Datacenter](https://arxiv.org/abs/2403.07648); [FALCON](https://arxiv.org/abs/2410.12588) and [Understanding Stragglers in Large Model Training](https://arxiv.org/abs/2505.05713); [Oobleck](https://arxiv.org/abs/2309.08125); [Streaming DiLoCo](https://arxiv.org/abs/2501.18512).
- Reference and tutorials: [PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html); new, 200 OK 2026-10-28: [Asynchronous Saving with DCP](https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html); TorchTitan's checkpointer, debugging and SDC documents.
- Code: [torchft](https://github.com/pytorch/torchft) (redirects to meta-pytorch/torchft); [torchtitan](https://github.com/pytorch/torchtitan).

#### Risks and open decisions

- **Rate data quality.** Tables are from different hardware, eras and definitions; the notebook shows all three and the discrepancy in Llama 3's table instead of a single "true" rate.
- **`async_save` API drift** (future vs `AsyncSaveResponse`; process-based checkpointer). Pin the version; fallback: a thread-based writer of our own with the same stall measurement.
- **Kill injection on SLURM** can leave zombie ranks; `launch` cleans the process group; the restart uses a fresh launch.
- **Different-world-size equality** is non-bitwise; the band is defined empirically and the convention (factor 3) is the editor's call.
- **torchft demo** is optional; if installation fails the section keeps D6 and the reading.
- **Storage bandwidth** differs per site; 6.8 uses 5.3's committed measurement for the reference cluster and prints the formula for others.
- **Default of `failure_rate` (decided 2026-10-29).** The default is Llama 3's rate (3.79 per 1,000 node-days), the measured interruption record of one frontier-scale job; RSC-1 (6.50, the most pessimistic) and RSC-2 (2.34) remain as `source='rsc1'` and `'rsc2'`, and `tab_lm_failure_sources` shows all three.
