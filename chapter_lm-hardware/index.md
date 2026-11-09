# Hardware for Training and Inference
:label:`chap_lm_hardware`

The reader knows one GPU from :numref:`sec_hardware` and :numref:`sec_perf_model`: a roofline, a memory hierarchy, a prefill that is compute-bound and a decode that is not. A released model is trained on thousands of chips and served from racks, and the chips, the kernels that run on them, the links between them and the power they draw decide what a token costs.

<!-- Author brief, copied verbatim from the Language Models part plan
(docs/llm/language_models_part_plan.md, "Chapter 5 Hardware for Training and Inference").
Blueprint: docs/llm/blueprints/ch05_hardware.md.
The plan is a specification, not book text; this comment is replaced
when the chapter is written.

**Label:** `chap_lm_hardware` · 8 sections · T0 to T4 (every T2+ section has a T1 or T0 path).
**Hook:** The reader knows one GPU from `sec_hardware` and `sec_perf_model`: a roofline, a memory hierarchy, a prefill that is compute-bound and a decode that is not. A released model is trained on thousands of chips and served from racks, and the chips, the kernels that run on them, the links between them and the power they draw decide what a token costs.
**Route:** What does a chip's area, power and package perimeter buy, and where does each accelerator sit (5.1)? How does a kernel turn that design into achieved FLOP/s, and what limits a hand-written attention kernel (5.2)? Why do training, prefill and decode want different hardware (5.3)? On which memory or storage tier can each byte of a workload live without stalling the GPU (5.4)? Which pairs of devices inside a server can talk fast, and what does the topology cost an all-to-all (5.5)? What bounds parallelism across a rack and a cluster, and how are the two networks built (5.6)? How does a collective library choose its algorithm and protocol, and when is it wrong (5.7)? What does a token cost in joules and in dollars, and what moves that cost (5.8)?
**Out of scope:** The roofline, memory hierarchy, tensor cores, number formats, compilation and CUDA graphs, ring all-reduce and the alpha-beta cost model are owned by `sec_perf_model`, `sec_hardware`, `sec_compilation`, `sec_memory_precision` and `sec_multi_gpu`; this chapter restates them in one sentence each. Buying and renting stay in `sec_hardware_buyers` and `sec_cloud_instances` (linked). Decode kernels are 12.2 (`chap_lm_inference`); serving pools are 13.7 and 13.8; the parallel algorithms that use the measured links are `chap_lm_parallelism`; failures and goodput are 6.10; the fp8 and fp4 training recipe is 4.3.
**Writer notes:** Running models are configurations only (Qwen3-8B dense, Qwen3-30B-A3B as the mixture of experts, Qwen3-0.6B and the reference model of 4.2 for T1 twins); no section downloads weights except 5.3's decode cross-check and 5.8's energy trace. Shared artifacts the chapter introduces: `d2l.chip_specs()` and `data/hw/chip_specs.csv` (5.1), `d2l.BenchLog` and `d2l.env_fingerprint()` (5.1, the microbenchmark contract of blueprint ch05 "Microbenchmark contract" applies to every measurement in 5.1 to 5.8), `lm-hw-a100-microbench` (5.1 bandwidth entries, 5.2 GEMM and attention-kernel entries), `lm-hw-a100-workloads` and `lm-hw-inferencex-snapshot` (5.3), `lm-hw-node-storage` with `tiers.json` (5.4), `lm-hw-node-p2p` and `lm-hw-t2pair-p2p` with `alpha_beta.json` (5.5), `lm-hw-cluster-nccl` with `alpha_beta.json` and the perftest files (5.6 and 5.7 share one two-node allocation and one job script), `lm-hw-systems` (5.6), `lm-hw-prices`, `lm-hw-a100-clocksweep` and `lm-hw-buildbox-power` (5.8). Exports consumed by later chapters: `d2l.ServingModel` and `d2l.TrainStepModel` (5.3; Chapters 12 to 14), `d2l.fit_alpha_beta` (5.4), `d2l.parse_nccl_tests`, `d2l.nccl_busbw_factor`, `d2l.coll_time` (5.5), `d2l.axis_ratio`, `d2l.layout_fit`, `d2l.systems_table` (5.6), `d2l.ring_allreduce_time`, `d2l.tree_allreduce_time`, `d2l.hier_allreduce_time`, `d2l.a2a_inter_node_time`, `d2l.sim_collective` (5.7), `d2l.cost_per_mtok_train`, `d2l.cost_per_mtok_serve`, `d2l.owned_vs_rented`, `d2l.energy_per_token` (5.8), `d2l.flash_attention_triton`, `d2l.triton_rmsnorm`, `d2l.gemm_shapes` (5.2). Interface placement against `01_interfaces.md`: `ckpt_records.jsonl` and `d2l.ckpt_stall` belong to 6.9 (`lm-par-ckpt`), not to this chapter; `d2l.axis_ratio` belongs to 5.6 (`sec_lm_networks`); the cost functions belong to 5.8 (`sec_lm_energy_cost`); the label `sec_lm_scale_up_out` is not used. The chapter hosts assignment A6a (`sec_lm_a6a`, hosted at the end of 5.8; a calibrated performance model of a chip, a node and a cluster, plus a fused Triton kernel; blueprint ch05 A5 is its spec plus the kernel task; T1 on one 16 GB GPU within 14 GiB, all-reduce tasks graded in reference mode from committed logs). Compute: the target accelerator for T2 to T4 is the A100 (an H100 is not assumed to be available); the build order is the T0 and T1 sections first, then T2 and T3, and T4 (5.6 and 5.7) last; about 5 A100-hours (5.1, 5.2 twin, 5.3, 5.8 clock sweep), about 3 node-hours (5.4, 5.5), one two-node allocation of about 2 node-hours shared by 5.6 and 5.7 (optional 4-node pass), all produced once per release; the build box runs no GPU-to-GPU traffic (no NVLink, unreliable peer-to-peer), so every T1 twin uses one GPU, host-device copies, or a gloo loopback group, and two-GPU measurements run on the T2 pair. Cross-chapter consistency: the node's interconnect is unknown until the committed `topo.txt` is read (NVSwitch or NVLink-bridged PCIe pairs); every prediction that uses an intra-node alpha-beta carries both rows and `topo.txt` decides which is reported (blueprint ch06 "Common conventions"); the A100 PCIe 40 GB row of `chip_specs` is the T2 reference; 5.2 owns the Triton vocabulary that 12.2 reuses (12.2 restates it in one paragraph); MFU conventions follow 4.2's `d2l.mfu` (causal attention term halved, recomputation excluded); dollar figures live in dated CSV rows, never in prose; the Tools appendix is linked, never absorbed.
**Teaching sources:**
- [How to Scale Your Model](https://jax-ml.github.io/scaling-book/), with [How to Think About GPUs](https://jax-ml.github.io/scaling-book/gpus/), [All About Transformer Inference](https://jax-ml.github.io/scaling-book/inference/) and [sharding](https://jax-ml.github.io/scaling-book/sharding/): the rooflines for training and inference (5.3), the collectives on GPU networks (5.5 to 5.7) and the placement models (5.6); JAX examples are translated to PyTorch.
- Stas Bekman, [Machine Learning Engineering Open Book](https://github.com/stas00/ml-engineering): the practical network, storage and benchmark chapters (5.4 to 5.7); its all-reduce benchmark scripts cross-check nccl-tests.
- [Modal GPU Glossary](https://modal.com/gpu-glossary): vocabulary for SMs, warps, tensor cores, NVLink (5.1, 5.2).
- [GPU MODE lectures](https://github.com/gpu-mode/lectures): microbenchmark methodology, Triton and profiling (5.2).
- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html), in particular [fused attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html): the reference kernel 5.2 tests against.
- [Stanford CS336](https://cs336.stanford.edu/) and its [assignment 2 (systems)](https://github.com/stanford-cs336/assignment2-systems): the Triton FlashAttention-2 assignment pattern that A6a follows.
- [CMU 11-868 LLM Systems, Spring 2026](https://llmsystem.github.io/llmsystem2026spring/): two GPU-programming lectures and a GPU-acceleration lecture before distributed training; the ordering this chapter follows.
- [Berkeley Scalable AI, Spring 2026 syllabus](https://scalable-ai.eecs.berkeley.edu/assets/syllabus.pdf): hardware constraints, interconnect topology and the parallelism assignment on one 8x H100 node per group; the "one node per group" compute model matches the T3 tier.
- [Efficiency in LLMs (MLSS)](https://alex.smola.org/posts/45-mlss-efficiency/): accelerator and model matching, the prefill and decode narrative (5.3).
- [NCCL user guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html) and [nccl-tests](https://github.com/NVIDIA/nccl-tests): benchmark methodology and bandwidth conventions (5.5 to 5.7).
- Barroso, Hölzle and Ranganathan, [The Datacenter as a Computer](https://link.springer.com/book/10.1007/978-3-031-01761-2) (3rd ed.): storage tiers (5.4), networks (5.6), power, cooling and total cost (5.8).
- [MLCommons inference benchmarks](https://mlcommons.org/benchmarks/inference-datacenter/): scenario definitions for 5.3's workloads.
- [Latency Numbers Every Programmer Should Know](https://gist.github.com/jboner/2841832): a dated contrast to measured tiers in 5.4 only, never a source for current numbers.
**Further Reading:** none.
-->

```toc
:maxdepth: 2

accelerator-design
kernels
workload-hardware
memory-tiers
server-paths
networks
collectives
energy-cost
```
