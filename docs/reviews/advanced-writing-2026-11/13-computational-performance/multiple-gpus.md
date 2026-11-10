# Multi-GPU from First Principles

Source: `chapter_computational-performance/multiple-gpus.md` (read to the `<!-- slides -->` marker at line 1149). About 3,500 words of shared prose, 375 (PyTorch) or 670 (JAX) words of tab prose, about 1,500 words of exercises per tab; roughly 5,500 in all. Stored outputs read: `outputs/pytorch/chapter_computational-performance/multiple-gpus.json` and the JAX file. Figures rendered and inspected: `mdl-perf-splitting`, `mdl-perf-data-parallel`, `mdl-perf-ring-allreduce`. The ring simulation was re-run in NumPy for $k = 2, 3, 4, 7$ and LeNet's multiply-add count recomputed.

## Verdict

The first two thirds of this section are good teaching: a felt question ("when does a second GPU shorten a step?"), data parallelism built from two functions, a one-step check that can fail, and a ring allreduce explained by a figure that is the best in the chapter. The last third, the cost model and its test on LeNet, does not hold together, and it is the part the section is named for. The section asks when the exchange of gradients costs more than it saves. Its experiment then shows a slowdown in which the exchange is 1.5% of the step (83 µs of 5.45 ms in the stored PyTorch output). The slowdown comes from one Python thread launching the kernels of two devices, an effect that the cost model `eq_dp_cost` does not contain. The "prediction" that the prose says is confirmed is obtained by substituting the measured two-GPU local time for the model's first term, so it tests only whether 83 µs add to 5.19 ms. With the model as written, the stored PyTorch numbers predict a speedup (1.27 + 0.08 = 1.35 ms against 2.63 ms on one GPU) and the measurement is a slowdown by a factor of two. On top of this, two sentences of shared prose are contradicted by the stored PyTorch output (halving the minibatch does halve the time; the two-GPU local work takes four times, not twice, as long). The single most important fix: make the experiment answer the section's question. Either give the hand-written loop a second model on which the cost model's three measured terms predict a gain and the measurement shows it, or state plainly that LeNet fails for a reason outside the model and show that reason with the chapter's own profiler.

On the questions asked. This section and the next are two sections, not one told twice: this one owns the mechanism (replicas, allreduce, ring, the $\alpha$ and $\beta$ of a transfer), the next the production tools and sharding. But the seam leaks in both directions (see S3). A reader with one GPU or none can run exactly two cells here (the ring simulation), because the notebook asserts two GPUs in its fourth cell; the prose and the four paper exercises still teach that reader the mechanism and the cost model, and S4 says how to let more of the code run.

## Scores

| dimension | score | note |
|:--|:-:|:--|
| Clarity of explanation | 4 | Replicas, allreduce and the ring are clear on first reading. The definition of $\beta$ is the one hard paragraph, and it clashes with the $\beta$ of the performance model. |
| Clarity of thought and focus | 3 | One line of argument until the cost model; then the experiment answers a different question than the one asked, and the prose reads it as a confirmation. |
| Pedagogy | 3 | Prediction before measurement, a check that can fail, a superb figure. But the only end-to-end result is negative for an unmodelled reason, and the section ends without one case where data parallelism works. |
| Language | 4 | Plain, exact, no mannerisms, no dashes. A few sentences with voice. Two tab blocks are long. |
| Ease and enjoyment | 3 | The ring pulls the reader along. The last third is a chain of measurements with three constants, two tabs that disagree, and a let-down. |

## What works

- **The opening** (lines 4 to 30). Three quarters of the machine idle, a link "about thirty times less than a GPU reads from its own memory" (32 GB/s against 1,008: recomputed, 31.5), and the question in one sentence. Keep. The last sentence of the roadmap is the best in the section and should be protected: LeNet's steps are "so short that fixed costs dominate them, which makes it the hardest case for data parallelism and the easiest one to take apart".
- **"Since all copies start equal and receive the same update, they stay equal without ever being compared."** (line 114). The whole invariant of data parallelism in one sentence.
- **Parameters as a list of tensors** (lines 133 to 135) so that "every byte that moves between devices appears in the code". The right decision for a from-scratch section, and it is said why.
- **The one-step check** (cell `multiple-gpus-data-parallelism-by-hand-7`) and its motivation: "A step that sums the shard gradients instead of averaging them trains with a learning rate $k$ times too large, and it may still reach a plausible accuracy." This is a check that can fail, introduced by the bug it catches. Stored: largest update 1.0e-02, difference 1.5e-08 (PyTorch); 2.6e-02 and 3.0e-08 (JAX). The prose ("less than a hundred-thousandth") holds for both.
- **`fig_ring_allreduce`**. Rendered and checked cell by cell against the text and the simulation: counts 1 to 4, the received chunk outlined, a different finished chunk on each GPU after three steps. A reader can follow the algorithm from the figure alone. This is the standard the brief asks for.
- **Byte counts that can be recomputed.** 61,706 parameters and 247 KB; 188 MB of images; $2(k-1)N/k$; 67 MB against 134 MB for ResNet-18 on four GPUs; 0.4 million multiply-adds per LeNet image (416,520), 1.7 per gradient byte. All correct.
- **"A measured $\beta$ thus describes a pair of GPUs, not the machine"** (JAX tab, line 662). A true and useful warning, currently visible to one tab only.
- **Strong and weak scaling** (lines 913 to 925): two paragraphs, correct, with the caveat that examples per second under weak scaling is not time to a given loss. The right length.
- **Exercises 2, 3, 4 and 7** (see below).

## Structural findings

**S1. The experiment does not answer the section's question, and the cost model is not tested.** The opening asks when the exchange costs more than it saves. The cost model has three terms. The LeNet measurement then finds: allreduce 83 µs (PyTorch) against a step of 5.45 ms; local work on two GPUs 5.19 ms against 1.27 ms for the same 128 examples on one. The slowdown is the launcher, a fourth effect. The PyTorch tab says so ("here they share one launcher") and the shared prose then continues as if the model had predicted the result. Three ways out, in order of preference:

1. *Add the case where it works.* After LeNet, run the same hand-written loop on one model that is large enough per kernel for the launcher not to matter (exercise 8 already designs it: LeNet widened by $w = 16$ at batch 1,024, or the ResNet-18 of the next section through `torch.func.functional_call` so that parameters stay a list). Predict the two-GPU step from $t_{\textrm{compute}}(B/2)$ measured on one GPU, $\alpha$ and $\beta$; then measure. That is a genuine test of `eq_dp_cost`, it ends the section on the answer to its own question, and it makes the next section's opening ("how much better, once the exchange is organized well?") a comparison of two measured numbers.
2. *Keep LeNet alone but tell the truth about it.* Print the three numbers side by side: what `eq_dp_cost` predicts from single-GPU measurements, what it predicts once the two-GPU local time is substituted, and the measurement. Say that the first is wrong by a factor of four in eager PyTorch and why. Then the lesson is "a cost model is only as good as its list of costs", which is a real lesson.
3. *Fix the launcher.* One thread per device in PyTorch removes most of the contention and takes ten lines; then LeNet's two-GPU step is limited by $\alpha$ as the model says.

I recommend 1 with 2's honesty about LeNet. This belongs on the list of experiments to redesign on the development machine.

**S2. The regime of the LeNet step is asserted, not measured.** The chapter's method is measure, classify, fix, re-measure, and `d2l.kernel_times` and `d2l.show_profile` exist since `subsec_perf-profiler`. Here the prose says the step "is close to the overhead-bound regime" without a profile, and the stored PyTorch numbers do not look overhead-bound (2.68 ms at 256 examples, 1.27 ms at 128: time proportional to the batch). One profile cell would settle it and would show the device-busy fraction falling when the second GPU is added, which is the actual mechanism of the slowdown.

**S3. The seam with the next section leaks.** Four things are told here and again there: the full `eq_dp_cost` display, the definition of $\beta$ (near-verbatim), the ResNet-18 numbers (2.2 billion multiply-adds, 45 MB, fifty per byte), and the three shortcomings of the hand-written loop (lines 927 to 936 here; the opening and the first subsection there). Proposed ownership: this section owns the cost model, $\alpha$, $\beta$ and the ring; the next cites them by label and equation number and never redisplays. The ResNet-18 numbers belong to the section that trains ResNet-18; here one clause is enough ("a ResNet-18 performs about fifty per byte, and the next section trains it"). The list of shortcomings belongs to the opening of the next section only.

**S4. What a reader without two GPUs can do.** The notebook stops at `assert d2l.num_gpus() >= 2`. Everything up to the timing cells is about semantics, not speed, and would run on any machine if the "devices" were allowed to coincide: in PyTorch, `broadcast`, `allreduce`, `train_step` and the one-step check run unchanged when both entries of `devices` name the same device (the one GPU, or the CPU); in JAX, `--xla_force_host_platform_device_count=2` gives two CPU devices on which `shard_map` and `psum` work. Suggest: replace the assert by a one-line notice ("with fewer than two GPUs the cells below run on two logical devices; the timings then mean nothing, the check still does"), and say in the opening which parts need the hardware. The paper exercises (1 to 4, 7) are already independent of hardware; say so.

**S5. Length is right; the weight is slightly off.** About 3,500 words is proportionate. Within it, the three-ways-to-split subsection (350 words) and strong and weak scaling (150 words) are the right size. The cost-model subsection carries too many measurements for one idea (four cells, three constants, four tab blocks); with S1 done it should be: measure $\alpha$ and $\beta$ (one cell), one message or ten (one cell), predict and measure (one cell per model).

## Passage-level findings

**MG-1** (minor, language). Opening.
Quote: "A PCIe 4.0 link carries a nominal"
The opening paragraph gives four rates in four sentences (83 trillion operations, 1 TB/s, 32 GB/s, thirty times). The 83 trillion fp32 operations per second play no part in the argument that follows. Cut that clause and the comparison of 1 TB/s with 32 GB/s stands out.

**MG-2** (minor, dependencies). Under `## Three Ways to Split`.
Quote: "parallelism matter at the scale of the models in the Language Models part,"
The forward reference is not load-bearing here, and the plan (`docs/llm/language_models_part_plan.md`, the row for multi-node training) does assign tensor and pipeline parallelism to that part. But the chapters are skeletons. Until they exist, add the two existing anchors: exercise 1 of this section (pipeline) and `ex_mgp-tensor` (tensor), so that the reader who wants to know more now has somewhere to go.

**MG-3** (minor, explanation). Under `### The Model and the Data`.
Quote: "A copy on every GPU takes the input pipeline out of the"
Good decision, well explained. One clause is missing for the reader who will later train on data that does not fit on a GPU: say that real pipelines feed each device from the host and that `sec_hardware` covers that path. One clause, not a paragraph.

**MG-4** (moderate, language: tab prose as a second explanation). JAX tab under `### The Data-Parallel Step`.
Quote: "the replicated copy directly, JAX's autodiff would insert a sum over the"
This 150-word block explains a subtle point (the transpose of replication is a sum, hence `pcast(..., to='varying')`) that is the single hardest idea in the JAX code and that the next section then relies on ("the derivative of 'copy to every device' is the sum over the devices"). It is correct and worth keeping, but it sits in a tab, so the PyTorch reader never learns the fact on which the next section's explanation of implicit collectives rests, and the JAX reader gets it as plumbing. Fix: state the fact once in shared prose in one sentence where the averaging of gradients is derived ("the gradient with respect to a parameter that is copied to $k$ devices is the sum of the $k$ per-device gradients; this is all an allreduce computes"), and reduce the tab to what is specific to JAX: why the `pcast` is needed here.

**MG-5** (minor, explanation). Under `### A One-Step Check`.
Quote: "The check runs in true fp32. Elsewhere in this section the GPU's tensor"
The paragraph explains a precaution before the reader knows what could go wrong, and "can select different convolution algorithms" arrives without preparation. Move the paragraph after the result as one sentence ("we ran the check in full fp32; in tf32 the two steps differ by rounding of about 1e-3 because batches of 256 and 128 select different convolution algorithms"), or into a code comment, which the cell already has.

**MG-6** (minor, explanation). Under `## Ring Allreduce`.
Quote: "way and then sit idle. The time of the star therefore grows linearly"
True for the hub's link. Say which time: "the hub's link carries $(k-1)N$ bytes each way, so the time of the star grows linearly in $k$". As written the reader has to infer that the hub is the bottleneck from the preceding sentence about the other links sitting idle.

**MG-7** (minor, pedagogy: check that half cannot fail). Under `## Ring Allreduce`, cells `multiple-gpus-ring-allreduce-1` and `-2`.
Quotes: "The chunk bookkeeping is easy to get wrong, so we check it in a simulation."; "Every device ends with the sum, and each sends exactly the bytes of"
The sum check is a real test of the index arithmetic (re-run for $k = 2, 3, 4, 7$: passes). The byte count is not: the loop sends $2(k-1)$ messages of $N/k$ bytes by construction, so "sent per device = 6144" cannot differ from the formula. Keep the simulation, since it is the only executable form of the ring in the body and it runs without a GPU. Drop the byte comparison from the print statement and from the sentence, or replace it with something the simulation can teach: the same function at $k = 2, 4, 8, 16$ printing bytes per device, which shows the approach to $2N$ that the text states. Also "stock noun": "chunk bookkeeping" (writing guide 10.3); write "The indices are easy to get wrong".

**MG-8** (moderate, explanation). Under `### A Cost Model for One Step`.
Quote: "defines $\beta$ by measurement: $2N/\beta$ is the time of an allreduce so"
This paragraph is where a first-time reader gets lost. Four problems. (i) $\beta$ is the memory bandwidth of the GPU in `sec_perf_model` (`eq_roofline`), two sections earlier and in every later section; here it is redefined as an interconnect rate without a word. (ii) The definition is indirect: "$\beta$ counts the bytes that leave and enter a device together", then "for our star ... $\beta$ is the bandwidth of one copy". The reader has to hold both. (iii) Exercise 3 then introduces a per-direction rate $b$ and warns that it "is not the $\beta$ of `eq_dp_cost`"; `further-reading.md` adds that $\beta$ is twice the algorithm bandwidth and $k/(k-1)$ times the bus bandwidth. Four rates for one link. (iv) $b$ is bytes per number in `sec_perf_model`, examples per device in the cells here, and a link rate in the exercises. Fix: give the link its own symbol (for example $\beta_{\textrm{link}}$, or $w$ for "wire"), define it directly as the measured rate at which one device sends during a large allreduce, and write the allreduce time from the ring count: $\alpha + \frac{2(k-1)}{k}\,N/\beta_{\textrm{link}}$. At $k = 2$ this is $\alpha + N/\beta_{\textrm{link}}$. The formula then says how the time grows with $k$ (which the current one hides, see MG-9), exercise 3 no longer needs a second symbol, and the next section's "$3N/\beta$ at $k = 4$" becomes a substitution instead of a puzzle.

**MG-9** (moderate, correctness; confidence high). Same subsection.
Quote: "cost $\alpha$. At $k = 1$ both communication terms vanish. Only the first"
"Only the first term shrinks as devices are added; the other two stay." Neither stays. By the section's own `eq_ring_traffic` the bytes per device grow from $N$ at $k = 2$ to $1.5N$ at $k = 4$ and toward $2N$, and by its own paragraph on the ring's $2(k-1)$ sequential steps the fixed cost grows linearly in $k$. `eq_dp_cost` is calibrated at $k = 2$ and written for general $k$. The next section silently corrects this ("one allreduce of data parallelism takes $3N/\beta$" at $k = 4$) and the stored measurements there confirm the growth (exposed communication 15 ms at $k = 2$, 23 ms at $k = 4$). Fix with MG-8: put the factor $2(k-1)/k$ into the equation and write "the communication terms do not shrink; they grow slowly toward a limit".

**MG-10** (moderate, correctness: numbers no cell prints; hardware-specific). JAX tab under `### Latency and Bandwidth of the Allreduce`.
Quote: "$\beta$ is 4.9 GB/s for GPUs 0 and 1, 6.2 GB/s for GPUs 2 and 3, the other"
Three rates to two digits for four pairs of GPUs of one machine; the cell measures only the first. They will not survive re-execution elsewhere and only one of them survives it here. Keep the idea, which is good, in shared prose and without digits: "the rate depends on which two GPUs take part and on the path between them (`fig_pcie_topology`), so a measured rate describes a pair, not a machine". The PyTorch tab's "about 20 GB/s" and "this is about what one PCIe 4.0 link carries in one direction after that detour" are fine as tab remarks but should say "on our machine".

**MG-11** (minor, thought). After the cell `multiple-gpus-latency-and-bandwidth-of-the-allreduce-1`.
The two tabs measure different things under one name. PyTorch's `allreduce` is the hand-written star of two `.to()` copies (18.9 GB/s, $\alpha$ = 57 µs); JAX's is NCCL's `psum` (4.9 GB/s, $\alpha$ = 272 µs, most of it program dispatch). The shared prose ("At these bandwidths") covers a factor of four. That is acceptable only if the shared text says so: one sentence, "the PyTorch tab times the star we wrote; the JAX tab times the collective library's allreduce, whose transport on this machine is slower", and the reader of either tab understands why the next section's PyTorch number (4.9 GB/s through NCCL) differs from this one's.

**MG-12** (major, correctness: stored output contradicts the prose). Under `### LeNet on One and Two GPUs`.
Quote: "On one GPU, halving the minibatch shortens the local work by far less"
Stored PyTorch output: `1 GPU(s), 256 examples each: 2.68 ms`, `1 GPU(s), 128 examples each: 1.27 ms`. Halving the minibatch shortened the local work by slightly more than half. Only the JAX output (0.43 ms and 0.43 ms) shows what the sentence says. The paragraph's explanation (overhead-bound, "halving the minibatch leaves the number of kernels unchanged") is therefore refuted for the default tab by its own numbers, and the Summary repeats the claim ("Halving LeNet's minibatch does not halve its computation either"). I cannot tell from the store why an eager LeNet step would be proportional to the batch (a convolution algorithm chosen for batch 256, the gather `X[i]`, or a loaded host are candidates). Fix: profile the step (S2), find out, and write the sentence from the measurement. If the PyTorch step turns out to be device-bound at batch 256, the section's premise that LeNet is overhead-bound holds for JAX only and the shared prose must say what is true of both.

**MG-13** (major, correctness: stored output contradicts the prose). PyTorch tab, same subsection.
Quote: "On two GPUs, the local work takes about twice as long as on one GPU with"
Stored: 5.19 ms on two GPUs against 1.27 ms on one with the same 128 examples, a factor of 4.1, not 2. One thread serving two devices explains a factor of two; the other factor of two is unexplained. This matters because the factor of four is the whole measured slowdown of the section.

**MG-14** (major, thought: experiment read for more than it shows). Same subsection.
Quotes: ":eqref:`eq_dp_cost` predicts that a step on two GPUs takes longer than a"; "The step on two GPUs is slower than the step on one, as the separately"
`eq_dp_cost` with its own first term $t_{\textrm{compute}}(128)$ predicts $1.27 + 0.057 + 0.026 = 1.35$ ms (PyTorch) and $0.43 + 0.27 + 0.10 = 0.80$ ms (JAX): a gain of 2 in PyTorch and a loss in JAX. The printed "prediction" (5.23 ms, 1.18 ms) replaces that term by the measured local time on two GPUs, which already contains the slowdown. What the cell then tests is whether an allreduce of 83 µs adds to 5.19 ms. It does, to within the 4% noise of the measurement, so the test has no power. The sentence "as the separately measured terms predict" is true only in this weak sense, and "For LeNet, a second GPU makes training slower" is true for a reason the section's model does not name. Draft for the honest version (if S1 option 2 is taken): "The cost model fails here, and it fails instructively. From the one-GPU measurement it predicts 1.35 ms; the step takes about four times that. The model assumes that two devices work side by side. In our loop one Python thread feeds both, and for kernels this short each device waits while the thread serves the other. The exchange itself is under 2% of the step. Communication is not what makes LeNet slow on two GPUs; a starved launcher is."

**MG-15** (minor, correctness: claim no cell supports). After the training cell.
Quote: "they differ only in rounding, which a thousand steps of SGD amplify. Two runs on one GPU differ about as much, because a"
The stored accuracies are 0.84 and 0.86 (PyTorch) and 0.87 and 0.87 (JAX), so "within a point or two" holds. The second sentence asserts run-to-run nondeterminism on one GPU that nothing here shows; it is true of some cuDNN algorithms and of scatter-add in general, but with fixed orders and one initialization the reader has no evidence. Either cut it or print two one-GPU runs. I would cut it; the first sentence is enough.

**MG-16** (moderate, thought). End of `### LeNet on One and Two GPUs`.
Quote: "computation it moves outweighs the exchange it adds. LeNet performs 0.4"
The paragraph draws the section's conclusion: a second device helps when the per-device batch keeps each GPU busy and when the model performs many operations per byte of gradient, and it offers "under two per byte" for LeNet against "about fifty" for ResNet-18. The numbers are right (416,520 multiply-adds over 246,824 bytes is 1.7; 2.2 billion over 45 MB is 49). But the experiment has just shown that LeNet's bytes cost 26 µs, so operations per byte is not why LeNet failed, and the criterion is incomplete in any case: compute time grows with the batch and the allreduce does not, so the ratio that matters is (operations per example) times (examples per device) over (gradient bytes), compared with the ratio of the GPU's achieved rate to the link rate. Write that inequality once: a second GPU pays when $c\,B/2 > \alpha + N/\beta_{\textrm{link}}$ with $c$ the measured time per example. Exercise 3 derives exactly this; the body should state it, since it is the answer to the opening question.

**MG-17** (moderate, language: second ending). Before `## Summary`.
Quote: "We can now predict from three measurements whether more GPUs shorten a"
A "We can now" recap followed by the list of shortcomings, a tab sentence, a pointer, and then the Summary, which restates all of it. By the author's decision 6 the body should end on its last result. Cut the paragraph; move the one new thing it holds (the loop waits for the whole backward pass before communicating) to the opening of the next section, which already says it twice.

**MG-18** (minor, language: unnamed pointers). Two places.
Quotes: "(The exercises ask what changes when the shards differ in size.)"; "trees, whose depth grows only like $\log_2 k$; an exercise derives where"
Give exercises 2 and 7 an `:exlabel:` and cite them with `:exref:`, as the next section does.

**MG-19** (minor, enjoyment: voice). `## Communication Cost and Scaling` as a whole.
From the cost model on, every paragraph has the same shape: quantity, cell, reading. The section has a surprising result (a second GPU doubles the step time of LeNet) and reports it in the tone of a parameter count. Say that it is surprising and say what the surprise is: the exchange of gradients, which the reader was led to fear, is negligible; what hurts is that two GPUs are harder to keep fed than one.

## Code and experiments

- `multiple-gpus-data-parallelism-by-hand-3` (two vectors summed to 3): fine once, three lines.
- `multiple-gpus-data-parallelism-by-hand-7`: a real test. Keep.
- `multiple-gpus-ring-allreduce-2`: the byte count cannot fail (MG-7).
- `multiple-gpus-the-accounting`: stored $\alpha$ = 57 µs, $\beta$ = 18.9 GB/s (PyTorch); 272 µs, 4.9 GB/s (JAX). Prose ("about 20 GB/s", "well under 10 GB/s") holds.
- `multiple-gpus-latency-and-bandwidth-of-the-allreduce-1`: 575 against 598 µs predicted and 82 against 83 (PyTorch); 2,821 against 2,823 and 288 against 373 (JAX). "Several times as long" holds (7 and 10). This is the one place where the $\alpha$ and $\beta$ model is tested with power, and it passes; the JAX tab explains its one miss well. Keep, and say that this is the test of the model.
- `multiple-gpus-lenet-on-one-and-two-gpus-1`: stored PyTorch output contradicts two sentences (MG-12, MG-13).
- `multiple-gpus-data-parallelism-by-hand-6`: 2.63 and 5.45 ms measured, 5.23 predicted (PyTorch); 0.40 and 1.07 measured, 1.18 predicted (JAX). "Within a few percent" holds (4%); the prediction has no power (MG-14). In JAX the measured step is below the prediction by 10%, which the tab explains.
- Numbers no cell prints: ResNet-18's 2.2 billion multiply-adds and 45 MB (recomputed by hand for the 3×3-stem network of the next section at 64×64: 2.21 billion; correct); LeNet's 0.4 million (correct); the JAX rates for other GPU pairs (MG-10).
- `train` in the PyTorch tab is 17 lines and `shard_grads` plus `train_step` 20; compact and readable. `p.grad = None` with its comment is the kind of detail that should stay.
- JAX `to_device` and `broadcast` take a sharding or a mesh where PyTorch takes a device; the shared prose calls both "device". Harmless.

## Exercises

Eight per tab. Their quality is high and four of them (1, 2, 3, 4) need no hardware.

- **1 (data versus pipeline)**: good; the micro-batch part derives GPipe's bubble. Recomputed part 1: 2.1 GB against 8.4 MB.
- **2 (gradient normalization)**: the best exercise in the section; the 3,000 against 1,000 token case is a bug every practitioner meets. Keep.
- **3 (strong-scaling speedup)**: recomputed. $S(2) = 1.78$, $S(4) = 2.93$, limit 8.2 at 20 GB/s; 1.46, 1.88, 2.67 at 5 GB/s; no gain below 1.75 GB/s; the condition $cB > 2N/b$ is indeed the same for every $k$. Excellent. With MG-8 its preamble about $b$ against $\beta$ disappears.
- **4 (parameter servers)**: good, and it connects to `further-reading.md`. Keep.
- **5 (ring by hand)**: needs four GPUs. Fine as marked. Part 1 is a paper problem and should be reachable without the hardware; split it off or say so.
- **6 (bf16 gradients)**: good numerical question; runs on one GPU or none (the shards can be computed on one device). Say so in the exercise, since the reader who skipped the two-GPU cells will assume otherwise.
- **7 (latency in the allreduce)**: recomputed part 2: the tree wins below about 376 KB. Good. Part 3 quotes DDP's 25 MB bucket, a nice link forward.
- **8 (where data parallelism helps)**: this is the experiment the body lacks (S1). Promote its core (one widened model at one batch size) into the body and keep the sweep as the exercise.
- To add: none. To fix: label 2 and 7 for the body's pointers.

## Still open from October

The October review's items for this chapter (the NCCL story told four times, draft history, hardcoded section numbers, em dashes) are closed in this file: no dashes, no section numbers, no draft history. The missing peer-to-peer path is still told three times here (opening, end of the ring section, JAX tab) and again in each of the next two sections; once per section in one clause with the reference to `subsec_hw-interconnects` is enough.

## For the chapter review

- **Order and role.** Fifth of seven; follows memory and precision, precedes the production tools. The placement is right. It assumes `d2l.Benchmark` and the three regimes from `sec_perf_model`, the interconnect facts from `subsec_hw-interconnects`, and the 16 bytes per parameter from `sec_memory_precision`. It sets up the ring's two phases (used by ZeRO and FSDP), `eq_dp_cost` (reused in the next two sections), and the strong and weak conventions.
- **The symbol $\beta$ changes meaning inside the chapter** (memory bandwidth in sections 1 to 4 and 7, link rate in sections 5 and 6). Resolve at chapter level.
- **Pattern to check in the whole chapter:** shared prose that describes an output true of one tab only (here MG-12, MG-13; the KV-cache review found the same). The two frameworks behave differently enough in this chapter (eager launch against one compiled program) that every "the step takes ..." sentence needs checking against both stores.
- **Pattern:** the chapter's method (profile, classify) is asserted rather than run when the subject is not a single GPU. Here one profile would have caught MG-12.
- **Overlap with the next section** is real but bounded (S3); fix by ownership, not by merging.
- **Hardware.** All timings are from a machine whose GPUs lack peer-to-peer access. The prose mostly says "on our machine", which is honest; the conclusions should be phrased as inequalities in measured quantities so that they hold on a reader's machine too (MG-16).
