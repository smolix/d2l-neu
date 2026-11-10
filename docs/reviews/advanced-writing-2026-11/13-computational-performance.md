# Chapter 13, Computational Performance: writing review

Sources: the nine `.md` files of `chapter_computational-performance/` in book order (introduction, seven sections, further reading), each read in full to the slides marker by a section reviewer; `index.md` and `further-reading.md` by me. Stored PyTorch and JAX outputs were checked against the prose by every section reviewer. About 43,000 words outside code, of which about 10,700 are exercises. I read all nine review files in full and checked the most serious cross-section claims in the sources (listed under "Checked in the sources").

Per-section reviews:
[index](13-computational-performance/index.md) ·
[performance-model](13-computational-performance/performance-model.md) ·
[hardware](13-computational-performance/hardware.md) ·
[compilation](13-computational-performance/compilation.md) ·
[memory-precision](13-computational-performance/memory-precision.md) ·
[multiple-gpus](13-computational-performance/multiple-gpus.md) ·
[multi-gpu-practice](13-computational-performance/multi-gpu-practice.md) ·
[fast-transformer](13-computational-performance/fast-transformer.md) ·
[further-reading](13-computational-performance/further-reading.md)

## Verdict

This is a chapter with a real idea, honest measurements and the best exercises in the part, and it does not yet deliver the method it is built on. Its idea is right and is stated in one line of the introduction: which of three limits holds a step back decides which change speeds it up. Where the chapter counts first and measures second it is as good as anything in the book: four elementwise operations of very different cost take the same 0.58 ms; an eight-operation chain is predicted at nine times one operation and measured at 8.9; a hand count of stored activations (20 values per token and block) is met by autograd at 18.7, and the measured peak then contradicts the budget by 77% and the text explains why. The arithmetic is right almost everywhere: the section reviewers recomputed the numbers of every section and found a handful of slips. But a prediction precedes a measurement once or twice per section, and the rest is measured and then explained. The one cost model the chapter builds for several GPUs leaves out the term that dominates its own stored PyTorch run. Shared prose is repeatedly written from one framework's output and is false for the other, three times against a stored number. And the constants of one machine (four RTX 4090 cards in a host without peer-to-peer transfers) harden from "on our card" in the first section into facts by the fourth. What matters most: make every experiment a prediction from $P$, $\beta$ and the host time per call, stated before the cell, and end the case study with the account in milliseconds that the introduction's question asks for. Almost all the predictions are two lines of arithmetic that the reviewers have already done from the stored outputs.

## Does the chapter deliver its method?

No, not as a method. It delivers the model (the roofline and its three regimes are taught clearly and correctly in the first section) and the habit of honest measurement (correct timing, checks before timings, negative results reported as such). It does not deliver "predict, then measure" as a practice the reader watches seven times. Three facts make the case.

1. The introduction itself states a weaker method: "measure a program, find the resource that limits it, relieve that resource, and measure again" (IDX-2). The first section says of the overhead limit "Only measurement reveals the third" (PM-16), although one measured constant, the host time per call, makes it as predictable as the other two. The predictive procedure is stated once, as the lead-in to a table at the end of the second section (HW-17).
2. Of the 40 experiments and quantitative claims in the table below (two rows of the case study hold several), a number precedes the measurement in 14, and three of those are weak tests: one cannot test its claim, one is contradicted by one tab's output, one repeats an earlier measurement. In 21 a prediction was available from numbers already on the page and is absent, stated after the result, or made in a form that cannot fail. One is a deliberate discovery. In 4 a prediction is made and never measured.
3. Where the chapter leaves a single GPU, the method is not applied at all: the regime of the LeNet step is asserted without a profile and the stored PyTorch output contradicts the assertion (MG-12; S2 in that review); the explanation of why overlap hides little rests on a trace no cell prints (MGP-12).

The repair is cheap because the model works. The reviewers computed the missing predictions from the stores, and most land within a few percent: 2.5 ms predicted for the MLP step against 2.55 ms profiled; 6.7 and 13.3 ms for the bf16 and tf32 GPT step against 8 to 10 and 14 to 15 ms; 31 and 47 ms for the allreduce of the case study against 31 and 48 ms measured. The chapter owns a model that predicts its own experiments and mostly does not say so.

### Every place where a prediction precedes a measurement, and every place where it should and does not

"Yes" means a number is stated before the cell and the cell could have contradicted it. IDs point to the section reviews; S-numbers are structural findings of the named review.

| Section | Experiment or claim | Prediction before measurement? | What happens, and what is available | Finding |
|---|---|---|---|---|
| performance-model | Chain of eight elementwise operations | Yes | 18 tensors against 2 predicts a factor 9; measured 8.9 (PyTorch), 8.5 (JAX) | What works |
| performance-model | Four elementwise operations | No (discovery, then explained) | Same 0.58 ms for add and sine; bandwidth 0.9 TB/s read off afterwards. Acceptable as a surprise | What works |
| performance-model | Square matrix sweep | Yes, and the test cannot test it | Roofline predicts compute-bound from $n = 500$; measured from about 2,048; gap attributed to a host time the cell does not show | PM-8, S2 |
| performance-model | Knee of the sweep | No | Formula after the output, with $h = 20$ µs that no cell prints; the printed 12.5 µs gives $n = 1010$ | PM-9 |
| performance-model | Reading a value every step | No | "one and a half times" (PyTorch), "three to four" (JAX); the formula $(h+d+\sigma)/\max(h,d)$ is on the page | PM-6 |
| performance-model | Profile of the MLP step | No | Five products at 0.42 ms plus 0.40 ms for the update predicts 2.5 ms; profile says 2.55 ms | S3 |
| hardware | Skinny product against batch size | Yes | Crossover predicted at $B = 164$; flat to 64, proportional beyond; both roofs reached. The grid (64, 256) brackets the knee without resolving it | What works |
| hardware | Pinned host-to-device copy | Yes (a bound) | At most 32 GB/s; measured about 23 | HW-13 |
| hardware | Memory bandwidth of a 1 GiB vector | Yes, and repeated | Same measurement as the first section | HW-4 |
| hardware | Single-stream decoding at 500 tokens per second | Predicted, not measured here | Measured in `sec_kv-cache`; quoted here as 500 and as 450 | HW-1, S3 |
| hardware | Random against sequential access, latency | Claimed, not measured | Five claims, no cell; the gather of exercise 9 is three lines | S4 |
| compilation | Fused against unfused chain | Yes | Factor 9, 1.15 against 0.13 ms; measured 1.25 against 0.15 ms | What works |
| compilation | Unfused activation in the billion-parameter GPT | Predicted, not measured | $9P/(\beta d) \approx 0.72$ of the product time | COMP-6 |
| compilation | A new shape needs a new compilation | Yes, and the PyTorch output contradicts it | Two compilations for three shapes (symbolic second compile) | COMP-4 |
| compilation | Compiled training step | No | Roofline gives about 0.2 ms; JAX 0.26 ms, compiled PyTorch 1.71 ms; no speedup stated, regime not named | COMP-9 |
| compilation | Kernel replay | No | $h$ over the per-kernel floor, about 18/2.5, predicts the factor of seven; division by 120 described, not printed | COMP-12 |
| memory-precision | Values stored per token and block | Yes | Tally 20; measured 18.7 (PyTorch), 29.0 (JAX, unexplained in shared prose) | MEM-5 |
| memory-precision | Peak memory against the budget | Yes, and it fails instructively | 511 KiB per token predicted, 905 measured; the gap is $8V$ bytes (PyTorch only) | MEM-6 |
| memory-precision | Activation bytes under bf16 | No | Tally predicts about 0.6; measured 0.58 and 0.60 | MEM-9 |
| memory-precision | Step time in fp32, tf32, bf16 | No | $1.1 \times 10^{12}$ FLOPs give 6.7 and 13.3 ms; measured 8 to 10 and 14 to 15 ms; prose says "close to the factor of two" for 1.55 | MEM-10 |
| memory-precision | Checkpointing, time | Yes | Four passes against three; measured 23 to 24% | What works |
| memory-precision | Checkpointing, memory | No | The section's formula gives 0.37 and 1.18 GiB against 0.42 and 1.25 measured; not stated | MEM-11 |
| memory-precision | Gradient accumulation | A check that cannot fail | Mean of means; the two real claims (peak follows the micro-batch, time unchanged) unmeasured | MEM-13 |
| memory-precision | The closing recipe | Predicted in memory, no time | About 25 s per optimizer step by the chapter's own model | MEM-14 |
| multiple-gpus | One message or ten | Yes | 598 against 575 µs and 83 against 82 µs (PyTorch); the one test of $\alpha$ and $\beta$ with power | Code and experiments |
| multiple-gpus | LeNet on two GPUs | In form only | The model's own prediction is 1.35 ms; measured 5.45 ms; the printed "prediction" substitutes the measured two-GPU local time | MG-14 |
| multiple-gpus | Regime of the LeNet step | Asserted | "overhead-bound" without a profile; PyTorch times are proportional to the batch | MG-12, S2 |
| multi-gpu-practice | Exposed communication | Yes | $2N/\beta = 18$ ms; measured 15 (PyTorch), 19 (JAX); the bucket model predicted far less and the text says so | What works |
| multi-gpu-practice | Weak and strong scaling efficiency | No | Ring traffic fitted at $k = 2$ predicts 83.5% at $k = 4$; measured 83%. It is exercise 2 | MGP-17 |
| multi-gpu-practice | Why overlap hides little | Explained from numbers no cell prints | A `no_sync` timing would measure it | MGP-12 |
| multi-gpu-practice | Sharded training | Predicted, never run | 7.1 GiB per GPU, 7 s of communication for 0.4 s of computation | S2 |
| fast-transformer | bf16 against fp32 logits; compile; bf16; batch 512; checkpointing | One of five | The logits check is predicted. Compile: bound 1.6, measured 1.13. bf16: 1.55 to 2.0, measured 1.75. Batch 512: the lead argues for a gain, the profile bounds it at 4%, measured 1.01. Checkpointing: sign predicted, size (0.75) not; measured 0.85 | S1, FT-5, FT-9, FT-10, FT-11 |
| fast-transformer | Data parallelism, four configurations | Yes | 0.84, 1.27, 1.52, 2.69 predicted; 0.91, 1.36, 1.63, 2.92 measured (PyTorch). Predicts a loss and a gain | What works |
| fast-transformer | Allreduce time; final account of the step; training check | No | Allreduce: 31 and 47 ms from $\beta$, measured 31 and 48, but the measured value is substituted. Account: 5.7 ms of arithmetic, 1.5 ms shortfall, 5.4 ms memory traffic per 8,192 tokens, never stated. Training check: a threshold any training model passes | FT-14, S2, FT-19 |

## Scores

As the section reviewers gave them; 1 to 5.

| Section | Explanation | Thought | Pedagogy | Language | Enjoyment |
|---|---|---|---|---|---|
| [index](13-computational-performance/index.md) | 4 | 3 | 3 | 4 | 4 |
| [performance-model](13-computational-performance/performance-model.md) | 4 | 3 | 4 | 4 | 4 |
| [hardware](13-computational-performance/hardware.md) | 4 | 3 | 3 | 4 | 3 |
| [compilation](13-computational-performance/compilation.md) | 4 | 3 | 4 | 4 | 3 |
| [memory-precision](13-computational-performance/memory-precision.md) | 3 | 4 | 4 | 3 | 3 |
| [multiple-gpus](13-computational-performance/multiple-gpus.md) | 4 | 3 | 3 | 4 | 3 |
| [multi-gpu-practice](13-computational-performance/multi-gpu-practice.md) | 3 | 3 | 3 | 4 | 3 |
| [fast-transformer](13-computational-performance/fast-transformer.md) | 4 | 3 | 3 | 3 | 3 |
| [further-reading](13-computational-performance/further-reading.md) | 4 | 3 | 3 | 4 | 3 |

The reviewers calibrated alike, and the table discriminates little: nothing above 4, nothing below 3, and "thought" is 3 in eight of nine rows for the same reason each time (an experiment that does not test the claim it sits under). So in words. **Strongest:** `performance-model.md`, for its opening, its two elementwise experiments and its exercises, and the first half of `memory-precision.md`, which is the method working (count, check the count, let the peak contradict it). **Weakest:** `multi-gpu-practice.md`, which is two sections in one file, has a third of its body in tabs, spends a subsection on one machine's transport, and has no experiment in its second half; and the last third of `multiple-gpus.md`, where the cost model, the experiment and the prose disagree. `fast-transformer.md` scores like its neighbours but falls furthest below its role: as the capstone it should be the section where the method pays off, and it predicts one of its five single-GPU measurements.

## Technical correctness and currency

The mathematics and the arithmetic are sound. No reviewer found a wrong formula in the roofline, the intensity of a matrix product, the memory tally, the ring allreduce, the ZeRO table or the sharding arithmetic, and the hardware numbers that could be checked from certain knowledge are right. The problems are of three other kinds: prose that a stored output contradicts, attributions that the experiments do not support, and claims about products and library internals that could not be confirmed and will date.

**Stored output contradicts the prose, or the explanation rests on numbers no cell prints (ordered by severity).**

- **MG-12** (major, sure). "On one GPU, halving the minibatch shortens the local work by far less": the PyTorch store prints 2.68 ms at 256 examples and 1.27 ms at 128. True only of JAX (0.43 and 0.43). Repeated in the Summary. Confirmed in `outputs/pytorch/.../multiple-gpus.json`.
- **MG-13** (major, sure). "about twice as long as on one GPU": the store gives 5.19 ms against 1.27 ms, a factor of 4.1. The unexplained second factor of two is the whole measured slowdown of the section. Confirmed in the store.
- **MEM-8** (major, sure). "so the term $h$ hardly changes": PyTorch prints 212 KiB in fp32 and 306 KiB under autocast, a rise of 44%. True of JAX only.
- **MGP-12** (major). The central explanation of why overlap hides little (backward kernels "about a tenth longer", allreduces "more than twice as long") comes from a trace the reader does not see; the evidence is delegated to an exercise.
- **PM-8** (major; could not be run). The size-independent time of small matrix products is attributed to the host. At $n = 512$ in PyTorch a device-side cause (too few tiles for 128 SMs) predicts the same number; the section's own profiler would settle it and is not used.
- **COMP-4** (moderate, sure). "a new shape ... therefore needs a new compilation" and the Summary: the PyTorch cell shows two compilations for three shapes.
- **MEM-6** (moderate). "therefore needs up to $8V$ bytes more than the count" holds in PyTorch; the JAX plan is not linear in tokens and its near-agreement is coincidental, as the JAX tab says.
- **MGP-10** (moderate). "For 45 MB of gradients in a few calls": three buckets in PyTorch, 76 allreduces in JAX; the fixed-cost term is dropped for both without a measurement.
- **COMP-10** (moderate, unverified). "captures the step in two graphs": nothing printed shows it; one `explain` line would.
- **PM-11** (moderate). The tile bound ("at most the share of SMs that hold a tile") predicts at most half the peak at $n = 1024$; the store shows 78%.
- **MG-10** (moderate). Per-pair link rates to two digits ("4.9 GB/s for GPUs 0 and 1, 6.2 GB/s for GPUs 2 and 3") of which the cell measures one.
- **PM-10**, **PM-7**, **FT-4** (minor). "five to ten" against stored ratios of 6.0 to 11.4 and a model that says eight; "more than 150 TFLOP/s" for 152.7; "nine tenths of the step or more" for 87%.
- **MG-15** (minor). Run-to-run nondeterminism on one GPU asserted with no evidence.

**Errors and overstatements of substance.**

- **COMP-7** (major, about 85% sure). "Compilers fuse three kinds of patterns", the third being an "epilogue" fused into a matrix product "while the tile is still in registers". Neither Inductor nor XLA does this for matrix products on a GPU; the section contradicts itself two paragraphs later, its replay experiment counts 120 kernels for 60 layers, and `further-reading.md` states the correct version ("fuse chains of elementwise operations and reductions").
- **MG-9** (moderate, high confidence). "Only the first term shrinks as devices are added; the other two stay." By the section's own ring count the bytes per device grow from $N$ toward $2N$ and the fixed cost grows linearly in $k$. The next section corrects it silently ("$3N/\beta$" at $k = 4$).
- **PM-3** (moderate). One compute roof $P$ per card is taught; elementwise kernels cannot use the tensor-core rate, so "less than a six-hundredth of the compute roof" compares against the wrong roof. Conclusion unchanged.
- **PM-4** (minor). The roofline with DRAM bandwidth is applied to operands that fit in the 72 MB on-die cache.
- **HW-5**, **HW-12**, **HW-15** (minor). "Workloads that were compute-bound a decade ago are bandwidth-bound" holds only for intensities between 140 and 290; "more than ten times" a 64-core CPU is about seven; prefill intensity "about $n$" is capped by the width.
- **COMP-3** (minor). "Autograd records this graph during the forward pass" is shared prose and false for JAX.
- **MGP-1**, **FT-15**, **FT-16**, **FT-18** (minor). The recap of LeNet repeats the misreading of MG-14; "helps only once the step on each GPU is long" is contradicted at $k = 4$; "more than ten times faster" where the previous section says hundreds; `lm_perplexity` attributed to `sec_lstm` (it is defined in `sec_adam`).
- **FR-7**, **IDX-3**, **IDX-5** (minor; mine). Citations for fp8 offered for fp4 training; "the rate in the GPU's specification"; a reference to `sec_custom_layer` for hand-written kernels, which that section does not treat.

**Unverified claims about hardware, papers and systems.** None is known to be wrong; each carries weight or will date.

- **PM-1**, **HW-9**, **HW-10**, **FT-20**: the peak $P = 165$ TFLOP/s in bf16 on the RTX 4090, on which every roofline in the chapter rests, is called a specification value. It is derived (16,384 cores times two FLOPs times 2.52 GHz gives 82.6 in fp32; the doubling for bf16 with fp32 accumulation, "GeForce cards run fp8 and fp16 tensor arithmetic at half rate", is what no reviewer could confirm). The stored measurements (157 to 172) support it. The fp8 rate of 330 and the 5090's fp4 entry are measured nowhere. The table caption says "from vendor datasheets" (HW-11).
- Hardware, without IDs: Gaudi 3, B300 and CDNA 4 format rates; the GPU cache latencies attributed to Luo et al.; no peer-to-peer on the RTX 5090. (The reviewer's query on `Kung.1988` is resolved: the key is S. Y. Kung, *VLSI Array Processors*, a fair source for systolic arrays.)
- **COMP-5**, **COMP-13**, **FR-5**: the recompile limit of eight and its name, the on-disk cache default, `torch._dynamo.explain`, XLA command buffers on by default.
- **MEM-4**: that `scaled_dot_product_attention` keeps one number per head and token holds for the flash kernel on a GPU in this configuration, not in general. **MEM-7**: the allocator trace uses four private names. **MEM-2**: the unit of GPT-3's batch size.
- **MGP-13**, **MGP-16**: the NCCL parameter, its removal in 2.31 and the bundled versions; what the Shardy partitioner chooses for split weights.
- **FR-3**, **FR-9**: accelerator throughput to four digits; speedrun ingredients.

**Currency.** The chapter is current for 2026 and names no superseded practice as current. Two omissions a reader of older material will notice: TorchScript and `torch.export` are absent from the compilation section (they are in further reading; the compilation review asks for one sentence), and `nn.DataParallel` is mentioned correctly as superseded. No TensorFlow or MXNet material appears in any of the nine files; `legacy-multigpu-lib.md` in the same directory is build-only and unlisted.

## What depends on the capture machine or on library versions

The chapter argues from one machine, which makes it concrete and credible. The hardware reviewer's observation holds for the whole chapter: constants of that machine harden into facts as the chapter goes on. Sorted by what they depend on.

| Depends on | Statements | Where | Finding |
|---|---|---|---|
| Nothing (ratios of byte or FLOP counts) | Chain factor 9; fusion factor 9; all elementwise operations equal; four passes against three; 16 bytes per parameter; $2(k-1)N/k$; ZeRO bytes per parameter | sections 1, 3, 4, 5, 6 | Keep as they are |
| The GPU model (RTX 4090) | $P = 165$ and 82.6 TFLOP/s, $\beta = 1.008$ TB/s, ridge 164 and 82; knee of the sweep; crossover batch 164; 500 or 450 tokens per second; 24 GiB and every "fits" statement; tf32 as the fp32 default; 1.15 ms and 0.13 ms; "three to four sequences" | all | PM-5, PM-17, HW-1, FT-2. Fine with "on our card" and a recipe for another card (PM-1, HW-9) |
| The host CPU and the framework's dispatch | Host time per call: 12 µs printed, 20 µs in the knee formula, "5–15 µs" six times in two sections, 8 µs in an exercise, 60 to 100 µs for a compiled JAX call; "less than two percent" of peak at $n = 256$ (1.6% PyTorch, 0.3% JAX); reads make the loop "one and a half times" or "three to four times" as slow; replay gains "seven to eight" and "twenty to thirty" times; eager against compiled step times; 8 ms for a no-op `nnx.jit` call | sections 1, 2, 3, 7 | PM-6, PM-9, COMP-2, COMP-12; the pattern section of the performance-model review. These are not flagged as host-dependent |
| The host's slots and topology | 23 GB/s pinned copy, reused as a constant in sections 2, 6 and further reading; no peer-to-peer; NCCL at about 5 GB/s (2 GB/s in a hardware exercise, 2.5 per direction elsewhere); per-pair rates 4.9 and 6.2 GB/s; $\alpha$ of 57 and 272 µs; scaling efficiencies 88% and 83%; "a few percent of the speed that its computation would allow" for sharding; every data-parallel speedup of the case study | sections 2, 5, 6, 7 | HW-13, HW-14, MG-10, MGP-17, FT-16 |
| Library versions (PyTorch 2.11, the pinned JAX, bundled NCCL) | Recompile limit of eight and its option name; `torch._dynamo.explain` fields; on-disk Inductor cache; symbolic second compile; "two graphs" with a break at `autograd.grad`; `_record_memory_history` and snapshot keys; dispatch of SDPA to a flash kernel; `saved_tensors_hooks`; `NCCL_SHM_USE_CUDA_MEMCPY`, NCCL 2.28.9, 2.30.4 and 2.31; DDP's first-bucket cap; XLA command buffers, the combiner threshold, `--xla_gpu_enable_triton_gemm=false`, `XLA_PYTHON_CLIENT_ALLOCATOR=vmm`, the CPU scheduler flag; `TEARDOWN_CUPTI=1`; `torch.cond`, `mark_dynamic`, `reshard` and selective-checkpoint names in exercises; eight API names in further reading | sections 3, 4, 6, 7, further reading | COMP-5, COMP-10, COMP-13, MEM-4, MEM-7, MEM-12, MGP-3, MGP-5, MGP-13, FT-3, FR-5 |
| The product cycle | `tab_gpu_specs` ("Representative GPUs in late 2026"); Gaudi 3, B300, CDNA 4, NVFP4; NVLink 4 and 5, PCIe generations, "GB200 NVL72"; which GeForce cards had NVLink; "Other Accelerators" in further reading | section 2, further reading | HW-8, HW-10, HW-11, PM-5, FR-3 |
| The sequence of bottlenecks | "The device is now busy for nearly the whole step" and the order of the four changes: one model, one GPU, one host | section 7 | S6 of that review |

Three decisions would fix most of this. (i) One short table near the start of the chapter, "our machine, measured", with a symbol for each constant ($P$ per format, $\beta$, the host time per call, the host link rate, the collective rate), so that later sentences are written in symbols with "on our machine" beside each number. (ii) One dated table for products, and per-product asides moved to `sec_hardware_buyers` or the Language Models hardware chapter. (iii) Version-specific names in code comments, with one sentence in the introduction naming the pinned versions.

## Arc and order

The chapter has one line of argument and the sections are in the right order: a model, where its two constants come from, the two toolkits for one GPU (time, then memory), several GPUs by hand and then with libraries, and a case study. Every reviewer confirmed the placement of their section. The hand-offs are good and should be kept: the performance-model section ends by asking where $P$ and $\beta$ come from; the compilation section cashes the `gelu_ish` chain of the first section; memory and precision motivates the multi-GPU sections with the 17 GiB that no technique reduces.

**Boundaries that leak.**

- *Performance model and hardware.* Three items sit on the wrong side. The skinny-product sweep `hardware-batched-decoding` is the clean test of the roofline that the first section lacks and is that section's own exercise 2; move it there. The predictive procedure in "Rules of Thumb" (HW-17) is the method and belongs in the first section. In return the SM and tile paragraph of the first section (PM-11) belongs with the GPU's internals in the second (HW-7). Both reviewers reached this independently.
- *Compilation.* "Compiling a Training Step" tests an overhead-bound step under a lead-in about fusion; it and "Kernel Replay" should be one argument (COMP-9, COMP-12).
- *The two multi-GPU sections.* `eq_dp_cost`, the definition of $\beta$, the ResNet-18 numbers and the list of shortcomings of the hand-written loop are told in both (S3 and S4 of the two reviews). The first owns the model and the ring; the second cites.

**Sections that are really two.**

- `multi-gpu-practice.md`: split at "Sharding the Training State" (S1 there). Recommended, and the second part needs its experiment.
- `memory-precision.md`: clean seam before "Mixed Precision" (structural finding 2 there). Optional at 5,200 words of body.
- `hardware.md`: two subjects (why $\beta$ is small; what lies outside the GPU), no split needed (structural finding 5 there).

**Proportion.** The seven sections run 5,000 to 7,200 words each and are nearly equal, so length does not follow importance. The first section, on which everything rests, is among the shortest in body (3,800 words); the hardware catalogue, the NCCL transport and the tab plumbing take space that the missing predictions and the sharding experiment need.

**No ending.** The introduction asks "Where does the rest of the time go?" The case study closes at "about half of the card's bf16 peak" and three causes without numbers. The stored profile holds the answer (FT S2): 5.7 ms of arithmetic at peak, about 1.5 ms of shortfall in the products, about 5.4 ms of memory traffic per 8,192 tokens. That paragraph is the chapter's ending and is not yet written.

## Most important findings

1. **State the method as prediction and practise it in every section.** One formula in time form in the first section, $t \gtrsim \max(h, \textrm{FLOPs}/P, \textrm{bytes}/\beta)$, then a number before each cell and a comparison after. PM-16, HW-17, IDX-2, structural findings 1 and 3 of the performance-model review, COMP-9, COMP-12, MEM-9, MEM-10, MEM-11, FT S1 with FT-5, FT-9, FT-11, FT-14.
2. **Rebuild the end of `multiple-gpus.md`.** The cost model omits the launcher that dominates the stored PyTorch run; its "prediction" has no power; two sentences contradict the store; the equation is wrong for $k > 2$. Add the case where data parallelism works, tell the truth about LeNet, and profile it. MG-12, MG-13, MG-14, MG-9, MG-16, S1 and S2 there; MGP-1 follows.
3. **Close the account in the case study and say what the changes were worth.** The final split in milliseconds (S2), bf16 as the one change that did the work, the larger minibatch as a predicted non-event (FT-10), the order-dependence thesis demonstrated by the data-parallel pair (FT-13), and the caveat that throughput is not progress (FT-19, with FR-2).
4. **One convention for the two tabs.** Shared prose states what any implementation must satisfy and names both results where the frameworks differ; tabs give numbers; mechanics go to code comments. COMP-3, COMP-4, COMP-9, MEM-5, MEM-6, MEM-8, MG-4, MG-11, MG-12, MGP-6, MGP-8, MGP-10, FT-8, PM-14; structural finding 1 of the memory review and S4 of the case-study review.
5. **Separate what lasts from the snapshot of one machine and one year.** A table of measured constants with symbols; one dated product table; per-device asides and library version numbers out of the prose. HW structural finding 1, HW-8, HW-10, HW-11, PM-1, PM-9, COMP-2, COMP-5, MGP-13, FR-3, and S3 of the multi-GPU practice review (the NCCL subsection).
6. **Fix the fusion claim.** Epilogue fusion is not something these compilers do for matrix products. COMP-7.
7. **Run sharding once.** The chapter's only treatment of a model that no GPU holds is a table and arithmetic. MGP S2, MGP-14, IDX-4.
8. **Replace the checks that cannot fail and the experiments that test the wrong thing.** MEM-13 (accumulation), FT-19 (training check), MG-7 (ring byte count), PM-17 (a quotient of two literals), HW-4 (a repeated measurement), PM-8 with structural finding 2 (the square sweep).
9. **One symbol, one meaning.** $\beta$ is the memory bandwidth in five sections and a link rate in two; $h$ is the host time per call and, in `memory-precision.md`, the bytes per token of the output layer. MG-8, MGP-11, FR-6.
10. **Give prefill and decode one owner.** Taught in `chapter_transformers/kv-cache.md` with a ridge of 80 and again here three times with a ridge of 164 and no reconciliation. PM-2, HW-1, structural finding 3 of the hardware review.

## Focus

- **A device catalogue inside a teaching section.** About a third of the body of `hardware.md`: four GPUs with a footnote on GeForce accumulation rates, Gaudi 3, B300, CDNA 4, NVFP4, which cards had an NVLink connector (HW structural finding 1, HW-8, HW-10, HW-14). The same again in "Other Accelerators" (FR-3). Meanwhile three invariants that would outlast the devices sit in exercises: the sweep time $C/\beta$ of about 20 ms on all four GPUs, power as the cap on bandwidth, and Little's law (HW structural finding 2).
- **The authors' debugging of their own machine.** "The Collective Transport": an undocumented NCCL variable, three version numbers, a deadlock, and an exercise (S3 of the multi-GPU practice review). Environment variables and allocator settings in tab prose before any idea (MGP-3, FT-3, FT-17, MEM-12, and S5 of the case-study review).
- **Framework internals at the weight of ideas.** Dynamo guard limits and `explain` fields (COMP-5); the profiler's planes, lines and events (structural finding 5 of the performance-model review); `saved_tensors_hooks` and `nnx.split` (structural finding 1 of the memory review).
- **No paper retelling.** The chapter does what it describes. The one survey passage is the formats paragraph (HW-8). Lists at equal weight: the five limits of compilation (COMP-14), the fixes table without its most useful lever, the batch size (PM-12), the 24 unranked topics of further reading (FR-1).
- **Each section has one question**, and four of them answer a neighbouring one in their main experiment: the square sweep (overhead where the roofline was to be tested), the compiled step (overhead where fusion was promised), LeNet (the launcher where the exchange was asked about), the larger minibatch (a non-event where a gain was argued).

## Experiments

The predict-or-not table above is the main record. Beyond it:

- **Cells that only confirm or cannot fail:** `memory-precision-gradient-accumulation` (MEM-13); the byte count of `multiple-gpus-ring-allreduce-2` (MG-7); `performance-model-counting-flops-bytes-and-arithmetic-intensity` (PM-17); the training check of the case study (FT-19); the two-GPU "prediction" of LeNet (MG-14). Tolerances a thousand times looser than the observed error (FT-7).
- **Experiments read for more than they show:** host time from a flat timing curve (PM-8); the tile bound (PM-11); replay credited with a gain that mixes two effects (COMP-12); "two graphs" (COMP-10); co-scheduling as the reason overlap fails (MGP-12); the JAX peak as support for the linear budget (MEM-6); "as the separately measured terms predict" (MG-14).
- **Confounded or unfair comparisons:** the JAX "+ bf16" step also swaps the attention kernel (FT-8); the PyTorch "+ compile" step applies fusion and CUDA graphs together (FT-5); eager modules against `reduce-overhead` (COMP-12); the two tabs time different allreduces under one name (MG-11); the opening gap of the case study compares a tf32 step with the bf16 peak (FT-1).
- **Stored outputs against prose:** MG-12, MG-13, MEM-8, COMP-4, PM-10, PM-7, FT-4, and the JAX halves of MEM-5, MEM-6, MGP-10.
- **Numbers no cell prints:** the host time of 20 µs (PM-9); per-kernel replay costs (COMP-12); every budget figure for the billion-parameter model, computed by hand in each tab separately (code notes of the memory review); per-pair link rates (MG-10); the trace figures (MGP-12); bucket sizes 9, 26, 9 MB (correct).
- **Experiments that work and have power:** the chain and the fused chain; the batch sweep; the two-subtraction memory measurement and the peak that disagrees; checkpointing; the one-step checks in both multi-GPU sections (the batch-normalization one teaches what DDP computes); one message or ten; exposed communication; the data-parallel prediction of the case study; checkpointing predicted as a loss.
- **Scale.** Everything runs in seconds to minutes on the capture machine. The cost is reach: sections 5 and 6 assert two GPUs in their first cells, five of eight exercises in section 6 need two or four, and a reader without a GPU can run two cells of the three multi-GPU sections. Both multi-GPU reviewers show how the semantic checks could run on two logical devices (S4 and S5 of the two reviews).
- **Scenarios with precision that will not survive re-execution:** mostly well handled; ratios are quoted as ranges. Exceptions are in the "host" row of the dependence table.

## Repetition

| What | Told where | Proposed single home |
|---|---|---|
| Prefill, decode, tokens per second $\lesssim \beta$ over bytes; arithmetic intensity and the roofline | `chapter_transformers/kv-cache.md` (`subsec_kv-cache-prefill-decode`: 83 TFLOP/s, ridge "about 80", confirmed in the source); `hardware.md` introduction (500), "Single-Stream Decoding" (450), Summary (450); `performance-model.md` with ridge 164 and no mention of the earlier meeting | Chapter 11 owns the phenomena and the formula. Chapter 13 owns the roofline; the first section acknowledges the earlier use and reconciles 80 with 164 in one sentence (PM-2). The hardware introduction keeps the bound once as motivation (HW-1, structural finding 3 there) |
| Asynchronous dispatch, synchronization points, the timing rules, `Benchmark`, a timeline figure | `chapter_builders-guide/gpus-devices-memory.md` (`subsec_async_dispatch`); "Timing GPU Code" here, about 500 words and a second figure | Basics. Three sentences and the new experiment here (structural finding 4 of the performance-model review) |
| The 18-transfer count of the GELU chain | Four times: section 1, the introduction of section 3, "Counting Memory Traffic", a caption | "Counting Memory Traffic" (COMP-1) |
| Memory bandwidth measured at 0.9 TB/s | Sections 1 and 2, same method | Section 1 (HW-4); replace the second by the gather experiment |
| Host time per eager operation | "5–15 µs" in six places in sections 2 and 3, 20 µs and 12 µs in section 1, 8 µs in an exercise | One measured symbol, defined in section 1 (PM-9, COMP-2) |
| SMs, tiles, small kernels below the roof | Section 1 (points forward), section 2 (repeats), exercise 7, further reading | Section 2, with Little's law (PM-11, HW-7, FR-8) |
| $16N$ bytes, activations, checkpointing with its four-against-three argument, mixed precision | Basics (`subsec_memory_during_training`, `subsec_activation_checkpointing`, `sec_numerics`); `memory-precision.md` restates $16N$ four times and copies the checkpointing paragraph nearly word for word | Basics for the general result; one clause here, then what is new for a transformer (MEM-1, MEM-11) |
| Number formats (E4M3, microscaling, NVFP4) | `sec_numerics`, `hardware.md`, further reading | `sec_numerics` for formats; further reading for block scaling; two consequences stay in `hardware.md` (HW-8, FR-4) |
| `eq_dp_cost`, the definition of $\beta$, ResNet-18 numbers, shortcomings of the hand-written loop | End of section 5 and start of section 6; the equation displayed twice | Section 5 for the model; section 6 for ResNet-18 and the shortcomings (S3, S4, MGP-2, MGP-4, MGP-11) |
| No peer-to-peer path between the GPUs | Three times in section 5, again in sections 6 and 7 | `subsec_hw-interconnects`; one clause elsewhere (October item, closed only in part) |
| The 1.9-billion-parameter GPT and its 29 GiB; "data parallelism makes no room for a larger model" | Twice each inside section 6, once in section 5 | Opening of the sharding part (S4 there) |
| Critical batch size and time to target | `sec_batch_size`; further reading twice; missing from the case study body | `sec_batch_size`; one sentence in the case study (FT-10, FT-19) |
| Results restated inside one file | "one to three sequences" three times, "half a million tokens" four times, "running example" 16 times (MEM-16); "about twofold or more" three times (FT-21); eleven numbers in the hardware Summary (HW-18) | Once in the body, once in the Summary |

**With the appendix.** Nothing. The chapter has no reference into the mathematics appendix (no `sec_mdl-*` or `eq_mdl-*` label occurs in any of its files) and needs none. The author's rule on duplication does not arise.

## Recurring patterns

**Shared prose written from one tab.** The chapter's two frameworks do not only differ in API here; they sit in different regimes (eager launches against one compiled program, an allocator against a compiler's memory plan, one process per GPU against one partitioned program). So "the step takes ..." sentences are often true of one store. Counts where the reviewers gave them: 31 tab blocks in `memory-precision.md`; a third of the body of `multi-gpu-practice.md` and of `fast-transformer.md` is tab prose; one subsection ("The Execution Model") has no shared sentence; nine JAX-only and five PyTorch-only prose blocks in the first section. Results that only one reader sees: the sixfold gap between the compiled steps (COMP-9), global against per-shard batch-normalization statistics (MGP-8), the derivative of replication as a sum (MG-4), "a measured $\beta$ describes a pair of GPUs" (MG-10), 29 against 19 stored values (MEM-5), which regime the baseline of the case study is in (S4 there).

**The first experiment of a section is predicted, later ones are only measured.** Noted independently by the compilation, memory and case-study reviewers; see the table.

**Plumbing in prose.** Environment variables, allocator settings, registration warnings, bundled library versions and workarounds appear before or between ideas: three in section 6, five in section 7, others in sections 1 and 4 (PM-14, MEM-7, MEM-12, MGP-3, FT-3, FT-17). The guide sends these to code comments.

**Two endings.** A recap before the Summary in five of seven sections, three beginning "We can now" (confirmed in the sources: `performance-model.md`, `multiple-gpus.md`, `multi-gpu-practice.md`): PM-15, COMP-14, MG-17, MGP-18, FT-21. The author's decision is one ending.

**Insights hidden in exercises.** The exercises are the strongest part of the chapter: every reviewer called their set the best they had seen, and the worked answers they checked are right. The other side of that is content the body needs and lacks: sweep time, power and Little's law (hardware), the cost of accumulation (memory, exercise 7), the widened model on which data parallelism pays (section 5, exercise 8), the efficiency extrapolation and the overlap trace (section 6, exercises 2 and 4), the fixed cost of a step, MFU and time to target (case study, exercises 2, 3 and 5). Exercises are also long: about 10,700 words, a quarter to nearly a half of some sections.

**Voice.** The October mannerisms are gone from every file (no dashes in prose, no draft history, no hardcoded section numbers). What replaced them in the later sections is a uniform register: quantity, cell, reading (MG-19, MGP-20, FT-22, the voice pattern of the compilation review). The sentences reviewers quote with approval are the ones where the authors judge: "The compiler made memory no faster and arithmetic no cheaper. It removed 16 of the 18 array transfers." "The fair baseline for a speedup from mixed precision is tf32, not true fp32." "On our machine, sharding lets the model train on four GPUs, at a few percent of the speed that its computation would allow." The case study, where a judgment on each change is most needed, has fewest.

**Notation drift.**
- $\beta$: memory bandwidth of the GPU (`eq_roofline`, sections 1 to 4 and 7); link rate defined through $2N/\beta$ (sections 5, 6 and 7, redefined without a word; confirmed in both sources). Four rates for one link once $b$ of exercise 3, the bus bandwidth and the algorithm bandwidth are counted (MG-8, MGP-11, FR-6).
- $h$: host time per call (sections 1 to 3); bytes per token of the output layer and loss in `eq_mp-budget` (section 4; found in the source at the budget equation). Not flagged by the section reviewers, who each saw one use.
- $b$: bytes per number (sections 1 and 7), examples per device (cells of section 5), a link rate (exercises of section 5).
- $N$: parameters, bytes of gradients, and bytes of weights within one sentence (MGP-15).
- $P$: one number per card in section 1, a number per format from section 2 on (PM-2, PM-3).
- Units: MB and GB in section 3, MiB and GiB in its neighbours; "1 GB" for a 1 GiB tensor (COMP-15, HW-4). "Compute graph" against "computation graph", "matrix multiplication" against "matrix product" (COMP-15).
- The running model: a GELU MLP for one paragraph in section 3, SwiGLU in section 4 and in `d2l.GPT` (COMP-8).

## Dependencies and connection to the main text

**From Basics.** Heavy and mostly well cited: `sec_use_gpu` with `subsec_async_dispatch`, `subsec_memory_during_training`, `subsec_count_without_alloc`, `subsec_activation_checkpointing`; `sec_numerics` with `subsec_loss_scaling`; `d2l.Benchmark`. The memory reviewer checked that these labels hold the cited content. The problem is the opposite of a missing reference: three results are re-taught at near full length (see Repetition). The chapter needs a stated rule on what it may assume from chapter 5, and the introduction should list the prerequisites (structural finding 3 of the index review).

**From earlier Advanced chapters.** `sec_gpt` (the model of the case study, with its width doubled), `sec_kv-cache` (prefill and decode), `sec_adam` (`lm_perplexity` and the corpus; FT-18 has the wrong label), `sec_batch_size`, `sec_attention-at-scale` (FlashAttention), `sec_moe` (a table row). The roofline is used in chapter 11 before chapter 13 defines it and with a different ridge; this is still open from October (PM-2).

**From the appendix.** None, and none needed.

**Into the chapter.** `chapter_optimization/sgd.md` says "`subsec_mp-accumulation` implements this technique"; the subsection checks an identity and implements no training loop (confirmed in the source; MEM-13 and structural finding 4 of the memory review). `hardware.md` promises that the multi-GPU practice section "traces where their time goes"; the trace is in an exercise (MGP-12). The introduction promises that libraries "hide communication behind computation and divide a model that no single GPU holds"; the first is disconfirmed by the section's measurement and the second is not run (IDX-4).

**Forward references to the Language Models part.** Eleven in the chapter body and further reading. None is load-bearing according to the reviewers, provided HW-8 and HW-16 are applied (the formats paragraph and a list of serving terms currently lean on it). The skeleton chapters already cite this chapter heavily (`chapter_lm-hardware`, `chapter_lm-parallelism`, `chapter_lm-inference`, `chapter_lm-compression` and others refer to its labels), so labels must survive any split: `subsec_mp-mixed`, `subsec_mp-measuring`, `subsec_mp-checkpointing`, `subsec_mgp-fsdp`, `tab_mgp-zero`, `eq_dp_cost`.

**Unnamed pointers.** "The exercises ask", "an exercise derives", "the exercise on Little's law", "an exercise studies that model" (MG-18, FR-8, and the case study's setup). Label and cite.

**Wrong targets.** `sec_custom_layer` for hand-written kernels (IDX-5); `sec_lstm` for `lm_perplexity` (FT-18).

## What to cut

Roughly 3,000 words, none of it argument.

- `performance-model.md`: the restatement of asynchronous dispatch and the second timeline figure; the profiler API notes; the cell that divides two literals; the recap before the Summary; the tile bound as an assertion.
- `hardware.md`: the repeated bandwidth cell; the formats paragraph down to its two consequences; the GeForce accumulation footnote, Gaudi 3, B300, CDNA 4, the NVLink history; "Single-Stream Decoding" down to one sentence; the eleven numbers of the Summary.
- `compilation.md`: the second byte count in the introduction; the epilogue pattern; half of each Dynamo tab note; two of the five limits; the recap sentence.
- `memory-precision.md`: the restated two-part split and $16N$; the copied checkpointing paragraph; the accumulation identity cell; framework mechanics in tabs (the XLA CPU scheduler flag among them).
- `multiple-gpus.md`: the fp32 clause of the opening; the ResNet-18 numbers; the list of shortcomings; the "We can now" paragraph; the byte comparison of the ring simulation.
- `multi-gpu-practice.md`: "The Collective Transport" down to one paragraph, with its cell and exercise 5; the redisplayed equation and definition; the allocator and registration note; the recap.
- `fast-transformer.md`: environment variables in tab prose; the checkpointing row as a step of the cumulative table; one of the three "twofold or more".
- `further-reading.md`: the second paragraph of "Other Accelerators"; "Bus Bandwidth" down to two sentences once the link rate is defined directly.

## What is missing

- The predictive formula in time form, and a prediction before each experiment (the table).
- A table of the capture machine's measured constants with symbols, and the derivation of $P$ so a reader can redo it for another card.
- A matrix product that crosses the ridge, in the section that defines the ridge (it exists one section later).
- One measured access-pattern or latency experiment in the hardware section, and the three invariants now in its exercises.
- A profile after compilation, showing that the bottleneck moved; one sentence on TorchScript and `torch.export`.
- The rule that generates the memory tally (a linear layer keeps its input, a product keeps both factors; MEM-3); an accumulated training step with its memory and time; the step time of the closing recipe (about 25 s, which is the real motivation for more GPUs).
- A case where the hand-written data-parallel loop wins, and a profile of LeNet on one and two GPUs.
- The mechanism of ZeRO stage 2 (MGP-14); one run of sharded training.
- Shared sentences for the two execution models and the two batch-normalization defaults.
- The closing account of the case study; a statement of which tools it leaves out and why; what to expect on another GPU; the caveat on time to target.
- A paragraph in the introduction on prerequisites and required hardware; a chapter ending.
- An entry on the input pipeline in further reading; a ranking there.

## What is best and should set the model

- **`performance-model.md`, the opening and the intensity of a linear layer.** One chip, one operation, a factor of sixty; then "The same layer, with the same weights, spans a factor of about a thousand in intensity between the two uses."
- **`performance-model.md` and `compilation.md`, the chain.** A count (18 transfers against 2) predicts a factor that any hardware will reproduce, the cell confirms it, and one sentence says what the compiler did and did not do. This is the form every experiment in the chapter should take.
- **`hardware.md`, area against perimeter**, and the sentence that states the programme: "They will change, but the relations between them change slowly."
- **`memory-precision.md`, count, check, contradiction.** The tally, the two-subtraction measurement, and "A count is a model of what the framework stores, and a model can leave things out."
- **`multiple-gpus.md`, the ring figure and the one-step check** introduced by the bug it catches.
- **`multi-gpu-practice.md`, the batch-normalization check** that shows what DDP computes, and the candour of "Overlap hides far less of the communication on this machine than the bucket model predicts."
- **`fast-transformer.md`, the table of measurement rules, the check before every timing, and the data-parallel prediction** that forecasts a loss in one configuration and a gain in another.
- **The exercises throughout**, in particular those that make the reader compute a consequence the text did not state. Promote the best consequences into the body and keep the sets shorter.

## Where the section reviews disagree

No reviewer contradicts another on a fact. Three tensions are worth a note.

- **The peak rate.** The performance-model and hardware reviews treat $P = 165$ TFLOP/s as derived and unverified (PM-1, HW-9); the compilation, memory and case-study reviews compute their proposed predictions from it as given. The predictions agree with the stores, which supports the value, but every proposed number inherits the caveat.
- **The host time.** The compilation review writes that the first section "uses $h = 20$ µs"; the performance-model review shows that 20 µs is nobody's measured value (PM-9). The first is a report of the text, the second a finding; they agree that one measured symbol is needed.
- **LeNet.** The multiple-GPU review praises the roadmap sentence that LeNet's steps are "so short that fixed costs dominate them" and then shows (MG-12) that the stored PyTorch times are proportional to the batch, which contradicts it for the default tab. The praise should be read as praise for the sentence's form; its content needs the profile.

The two reviewers of the seam between sections 1 and 2, and the two of the seam between sections 5 and 6, each proposed the same ownership independently.

## Checked in the sources

- `outputs/pytorch/.../multiple-gpus.json` and the JAX file: the timings behind MG-12, MG-13 and MG-14 are as the reviewer reports (2.68, 1.27, 5.19, 5.45, 5.23 ms; 0.43, 0.43, 0.83, 1.07, 1.18 ms).
- `chapter_transformers/kv-cache.md`: defines arithmetic intensity and the roofline with 83 TFLOP/s and "about 80 FLOPs per byte" and points forward to this chapter; `hardware.md` cites it at three places and re-derives the bound.
- $\beta$ is redefined as a link rate at `subsec_mg-accounting` and again, in the same words, in `multi-gpu-practice.md`. $h$ is the per-token bytes of the output layer in `memory-precision.md`.
- "5–15 µs" occurs three times in the body of `hardware.md` and three in `compilation.md`; 20 µs once in `performance-model.md`.
- `sec_custom_layer` does not treat kernels; `Kung.1988` and the two citations of FR-7 were looked up in `d2l.bib`.
- No `sec_mdl-*` reference in the chapter; no TensorFlow or MXNet output tree for it.
- Every finding ID cited here was checked against the section files.

## Could not check

- Nothing was executed: there is no GPU, PyTorch or JAX on the review machine. PM-8, MG-12 (the cause), COMP-10 and every proposed experiment are untested; each fix says what to verify.
- Claims listed as unverified above, against vendor documentation, the pinned library versions or the cited papers.
- Rendered HTML pages; only sources, stores and rasterized figures were read.
- Run-to-run variation: one stored run per framework.
