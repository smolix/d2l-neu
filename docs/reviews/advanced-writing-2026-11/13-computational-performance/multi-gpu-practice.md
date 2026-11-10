# Multi-GPU in Practice

Source: `chapter_computational-performance/multi-gpu-practice.md` (read to the `<!-- slides -->` marker at line 1376). About 3,000 words of shared prose, 890 (PyTorch) or 1,110 (JAX) words of tab prose, about 1,450 words of exercises per tab; roughly 5,400 in all. Stored outputs read: `outputs/pytorch/chapter_computational-performance/multi-gpu-practice.json`, the JAX file, and the stored scaling plot (rendered). Figures rendered and inspected: `mdl-perf-ddp-overlap`, `mdl-perf-fsdp-lifecycle`. The ZeRO table, the bucket sizes, the multiply-add count of the ResNet and every figure of the sharding arithmetic were recomputed.

## Verdict

A competent section with honest measurements, held back by three structural problems. First, it is two sections: production data parallelism with its scaling study (lines 107 to 853) and sharding of the training state (lines 855 to 998). The first half has code, a check and four measurements. The second half has a table, a figure and arithmetic, and not one cell: the section opens by asking "what can several GPUs do for a model that is too large for any one of them?" and the reader never sees such a model train. Second, the prose is split between the tabs more than anywhere else in the chapter: a third of the body is tab prose, one subsection (`### The Execution Model`) has no shared prose at all, and the two frameworks compute different things in the one-step check, which each reader learns only for their own framework. Third, about 400 words and one long exercise are spent on an undocumented NCCL environment variable that later NCCL versions removed and that deadlocks training in one of the two frameworks. That is the authors' debugging of their own machine, and it sits where the section's main result should be read. The single most important fix is the second half: run sharded training once, on the model the section itself introduces, and check the predicted memory and the predicted seconds per step against it. The arithmetic is already on the page and it is striking (7 s of communication for 0.4 s of computation); a measurement would make it the most memorable result of the two multi-GPU sections.

On the questions asked. Is this the previous section told twice? No. The division of labour is sound (mechanism and cost model there, tools, scaling and sharding here), but four items are told in both (see S4). Does a reader with one GPU or none learn from it? From the prose and the stored outputs, yes: the batch-normalization check, weak against strong scaling, the ZeRO table and the sharding arithmetic are all readable without hardware. From the code, almost nothing: every measurement is guarded by `d2l.num_gpus() >= 2`, and five of eight exercises need two or four GPUs. S5 says what could run anyway. Are the claims about the tools right? Those I can confirm from certain knowledge are right (the list is under "Technical claims checked"); the claims about bundled NCCL versions and the environment variable are unverified.

## Scores

| dimension | score | note |
|:--|:-:|:--|
| Clarity of explanation | 3 | Buckets, weak and strong scaling, and the ZeRO stages are each clear. The reader of one tab gets half of the picture in three places, and stage 2 is asserted without its mechanism. |
| Clarity of thought and focus | 3 | Two sections in one file; a subsection on one machine's transport; a headline mechanism (overlap) that the measurement then shows to do little, with the explanation delegated to an exercise. |
| Pedagogy | 3 | The one-step check and the predicted-against-measured cell are good. Half of the section has no experiment, and almost nothing runs on a reader's machine. |
| Language | 4 | Plain and exact; no mannerisms. Loses nothing on sentences; the problem is where they are placed. |
| Ease and enjoyment | 3 | The scaling study reads well. The setup (scripts in strings, environment variables before any idea) and the transport subsection drag. The sharding arithmetic is the high point and comes without a payoff. |

## What works

- **The one-step check with batch normalization** (cell `multi-gpu-practice-jax-annotate-the-layout-the-compiler-writes-the-collectives-5`). The PyTorch script compares the DDP step with two references, one GPU on all 256 examples and one GPU accumulating the two halves, and finds that it matches the second (1.2e-07) and not the first (1.1e-03 against a largest update of 6.7e-02). This is a check that teaches: the reader learns what DDP computes, not only that it runs. In JAX the same check comes out the other way (3.3e-05 against 2.4e-02: global statistics), and the cell that counts the compiled allreduces ("76 allreduces: 2 of 25.8 and 18.9 MB, the other 74 of 0.10 MB together") shows why. Keep both; see MGP-8 for how to let each reader see the contrast.
- **Weak and strong scaling measured side by side on the same model**, with the efficiency defined in one clause. Stored: weak 1.77 and 3.33, strong 1.78 and 2.82 (PyTorch); 1.68 and 3.08, 1.72 and 2.49 (JAX). The prose ("about 1.7", "a little over three", "diverge at four") holds for both stores. The explanation of why the conventions agree at two GPUs and part at four is correct and short.
- **The predicted-against-measured cell** (`multi-gpu-practice-ddp-really-run-5`): "predicted 2N/beta = 18 ms, measured exposed 15 ms" (19 ms in JAX). The section then says what this means without flinching: "Overlap hides far less of the communication on this machine than the bucket model predicts." A negative result reported as one. Keep the candour.
- **The three assumptions of the bucket model, named** (lines 780 to 782). Stating what a model assumes before saying which assumption fails is the right way to read a failed prediction.
- **`tab_mgp-zero`**. Bytes per parameter by stage, with one concrete model: 28.6, 17.9, 12.5 and 7.1 GiB (recomputed for 1.92 billion parameters at 16, 10, 7 and 4 bytes: correct). A reader can use this table the next day.
- **The sentence** "On our machine, sharding lets the model train on four GPUs, at a few percent of the speed that its computation would allow." It is exact (0.4 s of 7.4 s is 5%), it is quotable, and it carries the section's point about interconnects.
- **"The arithmetic also says when to shard"** (lines 984 to 991): the 18-block GPT at 15.1 GiB, sharded to 3.8 GiB, with the judgment that the freed 11 GiB are not worth the communication here. Authors who judge.
- **Exercises 2, 3 and 4** (see below).

## Structural findings

**S1. Two sections; split at `## Sharding the Training State`** (line 855). The first answers "how do production tools organize data parallelism, and how does a real model scale?" with ResNet-18 and measurements. The second answers "how can $k$ GPUs train a model that none of them holds?" with a 1.9-billion-parameter GPT and arithmetic. They share the ring's two phases and the cost model, both of which come from the previous section. Proposed cut:

- *Data Parallelism in Practice*: `Production Data Parallelism`, `Scaling on Four GPUs`. About 2,000 shared words after the cuts below. Exercises 1, 2, 4, 7.
- *Sharding the Training State*: ZeRO stages, FSDP, memory against communication, one experiment (S2). About 1,000 words now; 1,500 with the experiment. Exercises 3, 6, 8, and 5 if it survives.

If the author prefers one file, the second half still needs its experiment, and the opening must stop promising it otherwise.

**S2. Sharding is never run.** `fully_shard`, `ZeroRedundancyOptimizer` and `with_sharding_constraint` are named in tab prose; no cell calls any of them, here or in the case study that follows (`sec_fast_transformer` uses data parallelism only). The chapter introduction says this section "asks how production libraries ... divide a model that no single GPU holds". The reader who came for that finds a description. The arithmetic of `### Memory and Communication` is an exact prediction waiting for its measurement: 7.1 GiB of training state per GPU, about 7 s of communication and 0.4 s of computation per step for the 36-block GPT on four GPUs. Three steps of that model take half a minute. Proposed experiment, for the development machine: (i) show that the 36-block GPT with Adam raises an out-of-memory error on one GPU (or print its 29 GiB against `torch.cuda.mem_get_info`); (ii) apply `fully_shard` per block (PyTorch) or the placement with the constraint (JAX), take three steps on four GPUs with two sequences each and checkpointing, and print the peak memory per GPU and the seconds per step next to the predictions. This uses every tool of the previous section, turns the table into something the reader has seen happen, and gives the split-off section a reason to exist. If the JAX path proves too fragile (the tab itself says the partitioner may choose another scheme), run stage 1 in both tabs instead, which exercise 6 already specifies.

**S3. `### The Collective Transport` is written for the authors.** The subsection reports that NCCL's host-memory transport is slow by default, that an undocumented parameter makes it 4.6 times faster in a bare benchmark, that the parameter exists in NCCL 2.28.9 and 2.30.4 and was removed in 2.31, that DDP deadlocks with it, and that the outcome in JAX is left to an exercise. A reader on any other machine can use none of this, and a reader on the same kind of machine is told not to use it. The general lesson in its last two sentences ("Whether such a setting is safe has to be checked on the training run itself, not only on a benchmark of bare collectives") is worth one sentence, not a subsection, a cell, two tab blocks and exercise 5. Fix: keep one paragraph at the end of `### Predicted and Measured Communication`: the default transport between GPUs without peer-to-peer access reaches about a fifth of what a staged copy can (`subsec_hw-interconnects`), which is why $\beta$ is 5 and not 23 GB/s; tuning it is version-specific and belongs to `further-reading.md`. Move the measurement there or to an exercise, and delete the version numbers from the body. This also closes the October item on the NCCL story.

**S4. Repetition across the seam with `sec_multi_gpu`** (writing guide 7.7). Told there and again here: (i) `eq_dp_cost`, displayed in full a second time at line 683; (ii) the definition of $\beta$, near-verbatim (quote in MGP-11); (iii) the ResNet-18 numbers, which close the previous section and open this one; (iv) "the hand-written loop waited for the whole backward pass", which ends the previous section, appears in this section's second paragraph and again as the first three sentences of `### Overlapping Communication with the Backward Pass`. Inside this file: the 1.9-billion-parameter GPT and its 29 GiB are introduced in the opening and again, with the same numbers, at line 862; "data parallelism makes no room for a larger model" is said at lines 30 and 858 and in the previous section. Each needs one home and a clause elsewhere.

**S5. What runs without two GPUs.** Nothing that teaches. Two things could. (i) The one-step check is about semantics. With the `gloo` backend, `DistributedDataParallel` runs in two CPU processes; `CHECK_DDP` on 32×32 inputs would show the batch-normalization result on a laptop. `init_distributed` would choose the backend from `torch.cuda.device_count()`. (ii) In JAX, the count of compiled allreduces needs only two logical devices (`--xla_force_host_platform_device_count=2`); whether XLA emits the same collectives for CPU devices is something I have not verified. For the timing cells, say once at the top that they need the hardware and that the stored outputs are from four RTX 4090s without peer-to-peer access, and give the stored efficiencies as the fallback data for exercises 2 and 3 ("if you have no measurement, use 88% at $k = 2$").

**S6. The section's first idea is the one its experiment disconfirms.** The reader is taught bucketing and overlap with a figure whose arrow says "time saved" (rendered and checked), in the first subsection, as "above all" what production tools change. Three subsections later the measured exposed time is 15 to 19 ms against 18 ms for no overlap at all. The honest report is to the section's credit, but the order builds an expectation and then withdraws it. Since the opening also asks "How much better, once the exchange is organized well?", the answer the section actually has is: one process per GPU and a collective library fix the launcher problem of the previous section completely, and overlap helps only where the links are fast enough for a bucket to finish under the backward pass. Say that in the opening, present the process model first (it is what made ResNet-18 scale), and present overlap as the second mechanism with its condition.

## Passage-level findings

**MGP-1** (minor, correctness of the recap). Opening.
Quote: "GPU made the training steps of LeNet slower: each GPU had too little work"
This summarizes the previous section as "too little work to outweigh even a small exchange of gradients". The previous section's stored measurement says the exchange was under 2% of the step and the launcher was the cause (see MG-14 in the review of `multiple-gpus.md`). The recap repeats the misreading. After that section is fixed, write: "a second GPU made LeNet's steps slower, because one Python thread could not keep two GPUs fed".

**MGP-2** (minor, language: repetition). Opening.
Quote: "fifty for every byte of its 45 MB of gradients, so the work that a second"
The same three numbers ended the previous section thirty lines of rendered text earlier. Keep them here, where the model is trained, and cut them there (S4).

**MGP-3** (moderate, language: plumbing before any idea). JAX tab after the imports.
Quote: "Two more settings of our machine live in the environment of the notebook,"
The JAX reader's first 110 words after the roadmap concern an allocator setting and an NCCL registration warning. Neither is needed to read any cell. The writing guide (section 9) sends this to code comments. Move both to a comment in the setup cell, or to a short "Running this notebook" note at the end. Unverified: I do not know `vmm` as a value of `XLA_PYTHON_CLIENT_ALLOCATOR` (I know `platform` and the default BFC allocator); the author should confirm the spelling against the pinned jaxlib.

**MGP-4** (minor, language: repetition). Under `### Overlapping Communication with the Backward Pass`.
Quote: "the GPUs sat idle in the meantime. The production tools group the"
The first three sentences of the subsection retell the opening's second paragraph. Start at "The production tools group the gradients into *buckets* ...".

**MGP-5** (minor, explanation). PyTorch tab, same subsection.
Quote: "ResNet-18's 45 MB of gradients form three buckets of about 9, 26, and 9 MB."
Recomputed from the layer sizes: correct. But the text has just said that a bucket "holds up to 25 MiB", and the reader sees a first bucket of 9 MB that is closed long before the cap. The reason is that DDP gives its first bucket a much smaller cap (1 MiB, as far as I know; unverified against 2.11) so that communication starts early, and the first tensor to arrive, a 9 MB convolution weight, fills it. One clause would remove the puzzle. Also correct and worth a clause: the buckets are rebuilt after the first iteration in the order in which gradients actually became ready.

**MGP-6** (moderate, pedagogy: tab fragmentation). `### The Execution Model`.
Quote: "tells each copy its *rank*, its index from 0 to $k-1$, through environment"
The whole subsection is tab prose: 230 words for PyTorch (one process per GPU, NCCL, `torchrun`, rank) and 200 for JAX (one process, SPMD, placement decides, Shardy). There is no shared sentence. Yet the contrast is the idea: there are two ways to drive $k$ GPUs, $k$ ordinary programs that meet in a collective library, or one program that a compiler partitions, and every current system is one or the other. A PyTorch reader should finish knowing what SPMD means, and a JAX reader should know why PyTorch needs a launcher. Fix: three or four shared sentences stating both models and what each buys (processes: no shared interpreter, any Python code, communication explicit; one program: the compiler sees the whole step and inserts and schedules collectives, at the price of static shapes and of trusting its choices), then two-sentence tabs with the names of the tools.

**MGP-7** (minor, language). PyTorch tab and cell `multi-gpu-practice-one-program-per-gpu`.
The three PyTorch scripts live in raw strings and run in subprocesses. The text says why in one sentence, which is enough. One small thing would help the reader: say that each string is a complete script that can be saved and started with `torchrun` from a shell (that is how the reader will use DDP).

**MGP-8** (moderate, pedagogy: each tab sees half of the result). Under `### A One-Step Check`.
Quote: "each GPU's shard with the statistics of that shard, so it computes what"
The shared prose sets up the question well ("these statistics can cover either one GPU's shard or the whole minibatch"). The answer is in the tabs only: per shard in PyTorch, global in JAX. That the two leading frameworks answer differently by default is the most transferable fact in the subsection, and a reader of either tab never learns it. Fix: one shared sentence after the outputs: "The two frameworks differ here: DDP normalizes each shard with its own statistics unless the model is converted to `SyncBatchNorm`; a jitted JAX step computes the statistics over the global minibatch, at the price of one small allreduce per normalization layer." Then each tab keeps only the reading of its own numbers.

**MGP-9** (minor, explanation). Under `## Scaling on Four GPUs`.
Quote: "therefore the ratio of the single-GPU step time to the step time on $k$"
Correct and a useful identity. Add the number it gives so the reader can connect it to the next subsection: on two GPUs the step is 15 ms longer than on one (PyTorch store: 112 ms against 127 ms), and the next subsection asks where those milliseconds come from. As it stands the scaling subsection ends on LeNet and the next begins on the equation.

**MGP-10** (moderate, correctness: shared prose true of one tab). Under `### Predicted and Measured Communication`.
Quote: "communication that a step exposes. For 45 MB of gradients in a few calls,"
"In a few calls" holds for PyTorch (three buckets). The JAX cell above prints 76 allreduces, 74 of them small and each, as the JAX tab itself says, on the path of the forward or backward pass and incurring the fixed cost. The term $m\alpha$ is dropped for both tabs without a measurement, and the JAX exposed time (19 ms) exceeds the full $2N/\beta$ (18 ms). Fix: say "three buckets in PyTorch; in JAX two gradient buffers and 74 small allreduces for the normalization statistics, whose fixed costs we come back to in exercise `ex_mgp-syncbn`", and do not claim that the bandwidth term dominates in JAX until something shows it.

**MGP-11** (moderate, language: repetition; explanation). Same subsection.
Quote: "As in :numref:`subsec_mg-accounting`, we define $\beta$ as $2N$ divided by"
The equation is redisplayed and $\beta$ redefined with the same sentence as in the previous section ("counts the bytes that leave and enter a GPU together"). Cite `eq_dp_cost` and the definition; do not repeat them. The notation problem recorded as MG-8 in the review of `multiple-gpus.md` bites hardest in this section: `### The Collective Transport` needs "$\beta/2$ in each direction", and `### Memory and Communication` needs "$3N/\beta$" for $1.5N$ bytes. With a per-direction link rate defined once, both become plain.

**MGP-12** (major, correctness: explanation rests on numbers no cell prints). Same subsection.
Quotes: "allreduce that nothing overlaps. Overlap hides far less of the"; "tenth longer than in the same step without communication, and the"
The section's central empirical finding is that overlap hides little. Its explanation ("On this machine the third fails: the kernels of an allreduce take a share of the GPU's processors and memory bandwidth") is supported by a trace that the reader does not see: backward kernels "about a tenth longer", allreduces "more than twice as long as the bare benchmark", "together ... about 15 ms" (PyTorch tab), and an account of XLA's schedule (JAX tab). The evidence is delegated to part 4 of exercise `ex_mgp-buckets`. This is the case the writing guide names: an experiment is read for more than it shows. The measurement is cheap to add. In PyTorch, `step_time` at $k = 2$ with the backward pass inside `model.no_sync()` gives the two-process step without communication; the difference from the DDP step is the full cost of communication including the slowdown of the kernels beside it, and the difference from the one-process step is whatever two processes cost each other. One more line of output would turn three asserted numbers into measured ones. If the author prefers to keep the trace in the exercise, the body should claim only what its cell shows: the exposed time is about $2N/\beta$, so on this machine communication behaves as if nothing overlapped; why is the subject of the exercise.

**MGP-13** (minor, correctness: unverified). `### The Collective Transport`.
Quote: "NCCL parameter, `NCCL_SHM_USE_CUDA_MEMCPY=1`, makes NCCL use them. It"
Unverified in this session: that the parameter was removed in NCCL 2.31, that PyTorch 2.11 bundles NCCL 2.28.9 and jaxlib 0.12 bundles 2.30.4, and the description of the default transport as "ordinary loads and stores across PCIe". I believe the parameter exists in the shared-memory transport of recent NCCL sources. The stored output ("4.9 GB/s by default, 22.6 GB/s with the copy engines (4.6x)") matches "nearly five-fold" and "from about 18 ms to about 4 ms". Whatever is kept (S3), version numbers of bundled libraries should not be in prose; they change with the next pin.

**MGP-14** (moderate, explanation: missing step). Under `### Partitioning the Optimizer State, Gradients, and Weights`.
Quote: "updates only those parameters. Stage 2 also partitions the gradients, and"
The reader has just learned that every replica runs a full backward pass, which produces every gradient. How then can a GPU hold only $1/k$ of them? The text states the result (the table says "$1/k$") and moves on. The mechanism is short and is the same idea as bucketing: as soon as the backward pass has produced the gradients of a bucket, a reduce-scatter leaves each GPU with its own part of the sum and the rest is freed, so the full gradient never exists at one time. Without that sentence stage 2 reads as an accounting trick. Two smaller points in the same paragraphs: the table's memory is per GPU at the end of the backward pass, and the transient peak (one bucket of full gradients in stage 2, one or two gathered blocks in stage 3) deserves its clause where the table is introduced, not only for stage 3; and "Stages 1 and 2 therefore reduce the memory per GPU at no cost in communication" is the ZeRO paper's claim and is correct in bytes, but the all-gather of the updated parameters now sits between the optimizer step and the next forward pass, where nothing can overlap it. Say "at no cost in bytes".

**MGP-15** (minor, explanation). Same subsection.
Quote: "and one reduce-scatter of $N$ bytes, three phases where an allreduce has"
Correct (and the paper's 1.5). Here $N$ changes meaning in mid-sentence: the all-gathers move weights and the reduce-scatter moves gradients. They have the same size in fp32, which is why the count works; exercise 3 part 2 then separates them (bf16 weights and gradients). Say "of $N$ bytes each, since weights and gradients have the same size".

**MGP-16** (minor, correctness: unverified). JAX tab, same subsection.
Quote: "lets the partitioner choose a different scheme, such as matrix products on"
A claim about what the Shardy partitioner chooses for split weights, and that the constraint on activations yields FSDP's collectives. Plausible and consistent with how GSPMD propagates shardings; I cannot confirm it. Since no cell runs it, the reader cannot either. With S2 this becomes a cell that prints the collectives of the compiled step with the existing `allreduce_megabytes` pattern, and the claim is then shown.

**MGP-17** (minor, explanation). Under `### Memory and Communication`.
Quote: "$3N/\beta$, about 5 s, and the three phases of FSDP take $4.5N/\beta$,"
Recomputed: 4.7 s and 7.1 s at 4.9 GB/s; 0.29 s and 0.39 s of computation; about 19 ms over NVLink 5. All correct. The assumption "if every GPU keeps the bandwidth $\beta$ ... that we measured on two" can be checked against the section's own data and should be: the weak-scaling store gives an exposed time of 15 ms at $k = 2$ and 23 ms at $k = 4$ (PyTorch), a ratio of 1.54 against the predicted 1.5. One clause ("the scaling measurement above supports this: the exposed time grew by half from two GPUs to four") replaces an "if" with evidence. Exercise 2 currently asks the reader to discover this.

**MGP-18** (moderate, language: second ending). Before `## Summary`.
Quote: "We can now train a model on all GPUs of one machine with the tools that"
A "We can now" recap of three sentences plus a pointer, directly above a Summary that says the same. The second sentence holds the section's real conclusion and says it well: "when they exchange data through host memory, communication sets the pace long before the arithmetic does". Move that sentence into the Summary, where it is currently missing, and cut the paragraph.

**MGP-19** (minor, focus). `tab_collectives`.
The table's fourth row (all-to-all, for mixtures of experts) is used by nothing in the chapter. It is one row and points to `sec_moe`; keep it, but the sentence introducing the table says "the four collectives that data-parallel and sharded training use", which is true of three. Write "the collectives of this chapter, and one more that `sec_moe` needs".

**MGP-20** (minor, enjoyment: voice). `## Scaling on Four GPUs`.
The section reports 83% efficiency on four GPUs and never says whether that is good. It is good for 45 MB over a 5 GB/s path and poor against what the same GPUs would do with a direct link. An author's sentence would help: "Four GPUs give 3.3 times the throughput. A sixth of the machine is spent moving 45 MB per step through host memory."

## Technical claims checked

Confirmed from certain knowledge: DDP's default bucket cap of 25 MiB (`bucket_cap_mb`); buckets assigned in reverse order of `model.parameters()`; gradient-ready hooks; `nn.DataParallel` drives all GPUs from one process and the documentation recommends DDP; DDP normalizes per shard and `nn.SyncBatchNorm` synchronizes; `gradient_as_bucket_view`; `torch.distributed.fsdp.fully_shard` and `ZeroRedundancyOptimizer` exist as described; ZeRO's three stages and the factor of 1.5 for stage 3; XLA's allreduce combiner with a 30 MiB default and the flag named; the latency-hiding scheduler flag, off by default; GSPMD as the origin of JAX's SPMD partitioning; NVLink at 450 GB/s per direction on the H100 and 900 GB/s on the B200; 6 FLOPs per parameter and token.

Unverified, with what I believe: NCCL's allreduces running "on a CUDA stream of their own" (I believe the NCCL process group uses its own streams); the first-bucket cap of 1 MiB (I believe so); the NCCL parameter, its removal in 2.31 and the bundled versions; `XLA_PYTHON_CLIENT_ALLOCATOR=vmm`; `jax.sharding.reshard` in the exercises (I believe it exists in current JAX); the behaviour of the partitioner for split weights; the deadlock of DDP with the parameter set.

## Code and experiments

- `multi-gpu-practice-the-training-step-1`: "11.2M parameters, N = 45 MB". Matches the prose.
- `...-collectives-5` (one-step check): real test, both tabs; prose matches both stores.
- `...-collectives-3` (JAX allreduce count): matches "about 26 and 19 MB".
- `multi-gpu-practice-ddp-really-run-3` and `-4`: prose matches both stores at the quoted precision. The plot's legend covers the upper left of the panel, where the curves coincide; harmless.
- `multi-gpu-practice-ddp-really-run-5`: 18 ms predicted, 15 ms (PyTorch) and 19 ms (JAX) exposed. "Close to $2N/\beta$" holds. The weak-scaling times it subtracts each come from one window of 50 steps, and the difference of two 110 to 130 ms steps is 15 ms; a 3% error in either step moves the difference by a quarter. The prose's claim (exposed time comparable to the unhidden allreduce) survives that; the PyTorch tab's "a fifth of $2N/\beta$, about 4 ms" comparison does too. Do not read the 15 against 19 across tabs.
- Numbers no cell prints: the bucket sizes 9, 26, 9 MB (recomputed, correct); "8.7 of the 45 MB"; the trace figures of MGP-12; 983 MB for the enlarged training set (correct); the 2.2 billion multiply-adds (correct for this stem).
- `multi-gpu-practice-ddp-really-run-6`: see S3.
- No cell for anything in `## Sharding the Training State` (S2).
- `resnet18` is saved "since later chapters train it on several GPUs"; `legacy-multigpu-lib.md` says the consumers are the four-framework Image Models notebooks. Fine, but the sentence should name where.
- The JAX `train_step` contains no parallel code and the PyTorch script differs from a single-GPU script in two lines. Both facts are stated in tab prose and both are the best advertisement for the tools; they deserve the shared sentence proposed in MGP-6.

## Exercises

Eight per tab, about 1,450 words per tab, close to half of the body. Quality is high; reach is narrow.

- **1 (splitting the data)**: good. It breaks the split on purpose and asks for a cheap check. Needs two GPUs in PyTorch; with a CPU fallback (S5) it would not.
- **2 (weak-scaling extrapolation)**: the best in the section. Fit one constant at $k = 2$, predict $E(4)$, compare two models. With the stored PyTorch numbers the ring-traffic model predicts 83.5% at $k = 4$ against a measured 83%, and the constant-exposure model 88%. Give the stored efficiencies as fallback data so that a reader without GPUs can do it.
- **3 (cost of sharding)**: paper problem, good; part 3 uses "the $\beta$ that you measured": give 5 GB/s as the fallback.
- **4 (bucket size)**: parts 1 and 2 are a clean piece of modelling (formula re-derived: correct). Part 4 carries the evidence for the body's main explanation (MGP-12); either the body shows it or the exercise does, not "the body asserts and the exercise checks".
- **5 (transport configuration)**: follows the fate of S3. Part 1 (read NCCL's debug output and find the transport) is generally useful and could survive as a part of exercise 4.
- **6 (sharded optimizer state)** and **8 (tensor against data parallelism)**: adapted from CS336 with attribution; both are substantial and well specified. If S2 puts a sharding experiment in the body, 6 becomes its natural extension.
- **7 (synchronized batch normalization / batch-norm statistics)**: good, and it is where the $m\alpha$ term of the JAX step is finally measured (MGP-10).
- Count: with a split, four and four. No exercise needs to be cut for quality.

## Still open from October

"The NCCL bug told four times": now told once at length here (subsection, cell, two tab blocks, exercise 5) and in a clause in each neighbouring section. S3 closes it. Dashes, draft history and hardcoded section numbers: none found in this file.

## For the chapter review

- **Order and role.** Sixth of seven. It consumes the ring, `eq_dp_cost`, $\alpha$ and $\beta$ from `sec_multi_gpu`, the 16 bytes per parameter and both GPTs from `sec_memory_precision`, the interconnect facts from `subsec_hw-interconnects`, and `d2l.Benchmark`. It supplies `d2l.torchrun`, `d2l.init_distributed`, `d2l.resnet18` and (JAX) `d2l.allreduce_megabytes` to the case study, and `tab_mgp-zero` to the Language Models plan.
- **Split recommended** (S1), and the second part needs an experiment (S2). This is the chapter's only place for sharding; the case study does not use it.
- **`hardware.md` promises** that this section "traces where their time goes" for the collective library. The trace is in an exercise (MGP-12). Either deliver it or change the promise.
- **Pattern for the chapter:** tab prose carries ideas, not only API names. Here a third of the body. The fix is always the same: state the idea once for both, keep the names in the tabs.
- **Pattern for the chapter:** machine-specific plumbing in prose (allocator settings, registration warnings, bundled library versions). Three occurrences here, more in the case study.
- **Forward references** to the Language Models part (tensor parallelism, multi-machine training) are not load-bearing; the plan in `docs/llm/language_models_part_plan.md` assigns both.
