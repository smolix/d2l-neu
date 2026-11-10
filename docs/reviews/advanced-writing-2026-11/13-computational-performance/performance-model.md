# The Performance Model

Source: `chapter_computational-performance/performance-model.md` (lines 1 to 962; about 5,000 words of prose and exercises, of which about 1,250 are exercises; 16 PyTorch cells). Stored outputs read: `outputs/pytorch/.../performance-model.json` and `outputs/jax/.../performance-model.json`. Figures rendered: `mdl-perf-roofline`, `mdl-perf-async-timeline`, `mdl-perf-regimes`.

## Verdict

This is one of the better sections in the book and close to the bar. It opens on a fact the reader can feel (one chip, one operation, a factor of sixty in speed), it defines one quantity (arithmetic intensity), and every worked number I recomputed is right. Two experiments are of MacKay quality: four elementwise operations of very different cost take the same 0.58 ms, and an eight-operation chain takes 8.9 times as long as one operation because it moves 18 tensors instead of 2. What keeps it from a 5 is that the section's central promise, predicting the regime before measuring, is only half kept. The one experiment that is set up as a test of the roofline (the square sweep) cannot test it: the roofline says square products are compute-bound from n = 500, the measurement says from n = 2048, and the gap is explained by a host time h that is asserted, not shown, and quoted with three different values. The bandwidth-bound side of a matrix product is never measured here at all; the experiment that does it cleanly sits in the next section. **Most important fix:** give the reader one predictive formula in time form, t ≥ max(h, FLOPs/P, bytes/β), test it on a skinny product whose batch crosses the ridge (move the cell `hardware-batched-decoding` here), and predict the MLP step before profiling it (the prediction is 2.5 ms; the stored profile says 2.55 ms).

## Scores

| dimension | score | reason |
|---|---|---|
| Clarity of explanation | 4 | Terms defined in order, a number after each formula. Loses a point for P treated as one number and for the unexplained jump from "time is flat" to "host time". |
| Clarity of thought and focus | 3 | The main experiment does not test the main claim; h has three values; the overhead regime is called unpredictable two pages before a formula predicts it; the timing subsection re-teaches Basics. |
| Pedagogy | 4 | Strong motivation, strong experiments, excellent exercises. No whole-step prediction; the lever the reader controls most (batch size) is missing from the fix table. |
| Language | 4 | Plain, exact, unmannered. A second ending before the Summary; nine JAX-only prose blocks fragment the flow. |
| Ease and enjoyment | 4 | The opening and the two elementwise experiments make a reader want to continue. The profiler plumbing and the tab switches drag. |

## What works

- **The opening** (lines 4 to 15). "A product of two $256 \times 256$ matrices reaches less than two percent of it, although the operation and the chip are the same" is a problem the reader feels, stated with numbers, followed by the question the section answers. The October review called this a model opening; that still holds. Keep it.
- **Intensity of a linear layer** (`eq_matmul_intensity` and the two paragraphs after it). The middle form "shows that the smallest of the three dimensions limits the intensity", and the GPT example gives 910 FLOP/byte in training and 1 FLOP/byte in generation: "The same layer, with the same weights, spans a factor of about a thousand in intensity between the two uses." I recomputed both (910.2 and 0.999). This is the sentence a reader will remember.
- **Elementwise operations.** Add, multiply, sine and sigmoid all take 0.58 ms. "A sine costs many more FLOPs than an addition, yet it runs no slower, because the bytes each kernel moves set its time." A surprise, a measurement, an explanation in one line.
- **The chain.** The count "read and write 18 tensors of the same size between them" predicts a factor of 9; the stored outputs give 8.9 (PyTorch) and 8.5 (JAX). This is a prediction from counting that survives on any hardware, the best kind of experiment for this chapter.
- **The SGD update in the profile** (PyTorch tab): "It moves 12 bytes for each of the 34 million weights, 0.4 GB, which takes 0.4 ms at the card's bandwidth", against 0.42 ms in the stored profile. A back-of-envelope number that matches a profiler line.
- **Exercises 4, 5 and 6** (diagnosis from timings, idle time, the pipeline model of dispatch). They test the method, not recall. Exercise 6's formula $R\max(h,d) + \min(h,d) + \sigma$ is correct (checked both cases).
- All of these check: 2BDF FLOPs; n/3 for a square bf16 product; ridge 165.2/1.008 = 164; $0.25\beta$ is 1/655 of P; $2 \cdot 4096^3 = 1.37 \times 10^{11}$; knee $(hP/2)^{1/3} = 1182$ for h = 20 µs; 256 MiB and $5.4 \times 10^8$ bytes; intensity 341 and ridge 82 for the tf32 MLP products; 33.6 M weights.

## Structural findings

1. **The reader cannot yet predict the regime before measuring; the section stops one step short.** The roofline is given in rate form, $\min(P, I\beta)$, and the overhead in a separate time form, $\max(h, 2n^3/P)$, for square products only. The introduction says of the third limit "Only measurement reveals the third", and the method is measure, classify, fix. Yet everything needed for prediction is on the page. State once, after the sweep: the time of one call is at least

   $$t \;\gtrsim\; \max\!\left(h,\; \frac{\textrm{FLOPs}}{P},\; \frac{\textrm{bytes}}{\beta}\right),$$

   and whichever term is largest names the regime. With h about 10 µs eager on this host, any operation with fewer than about $10^9$ FLOPs (bf16 matrix product) and fewer than about 10 MB moved is overhead-bound, whatever its intensity. That is a rule the reader can apply to a model on paper. Then the hardware section's sentence "a workload can be placed on the roofline before it runs: count its FLOPs and bytes, divide by $P$ and $\beta$, and compare the result with the host cost of its operations" becomes a citation of this section, where it belongs.

2. **The square sweep is the wrong test of the roofline.** It is introduced as one ("To see whether the GPU agrees") and then shows overhead instead; the text concedes "the sweep never shows the bandwidth-bound regime" and switches to elementwise operations. A matrix product that crosses from bandwidth-bound to compute-bound at the predicted intensity is never shown here. The cell `hardware-batched-decoding` in the next section does exactly this: a $16{,}384^2$ bf16 matrix times a batch of B rows, flat at 0.57 ms up to B = 64 and proportional to B beyond, with the crossover predicted at B = 164 and both roofs reached (0.94 TB/s at B = 1, 171 TFLOP/s at B = 1024). It is also this section's own Exercise 2. Move that cell here as the test of `eq_roofline` (it supports the sentence "the intensity grows linearly with the batch size" directly), keep the square sweep as the discovery of the third regime, and let the hardware section cite it for decoding.

3. **Predict the MLP step before profiling it.** The profile is read after the fact. The model can do better: the step has five large products (two forward, three backward: two weight gradients and one activation gradient) of $2 \cdot 1024 \cdot 4096^2 = 3.4 \times 10^{10}$ FLOPs each, 0.42 ms each at 82.6 TFLOP/s, plus 0.40 ms for the update: 2.5 ms. The stored PyTorch profile reports 2.55 ms busy, with gemm lines of 0.82, 0.79 and 0.38 ms, which are two, two and one products. Putting the prediction first turns the profile from a printout into a confirmation, and it is the single most persuasive demonstration that the model works on a whole step.

4. **"Timing GPU Code" re-teaches the Basics.** `chapter_builders-guide/gpus-devices-memory.md` (`subsec_async_dispatch`) already has the queue, the 32-product experiment, the definition of a synchronization point, a timeline figure (`fig_bg_async_queue`), the `Trainer` rule, the four timing rules, and `Benchmark`. This section restates all of them in about 500 words and a second timeline figure. New here are only the data-dependent-shape barrier, the measured cost of reading every step, and the max-versus-sum model. Cut the restatement to three sentences with the citation, drop `fig_async_timeline` (or keep one figure in the book, not two), and keep the experiment.

5. **The profiler machinery is long for what the section reads from it.** `kernel_times` and `show_profile` are about 45 lines (JAX: a trace directory, xplane protobuf, planes, lines, events) introduced for one six-line printout here; their real use is the case study. The reader of this section needs the printout and the three facts read from it. Consider defining the two functions with a one-sentence description each and moving the API notes ("one *plane* per device, one *line* per CUDA stream") out of the main line.

6. **Length and proportion.** About 3,800 words of body is right for the content. The exercises (nine, about 1,250 words, three marked ●) are a quarter of the section; they are good, but Exercise 2 duplicates the next section's cell and Exercise 7 rests on a tile model the stored output does not support (see PM-12).

## Passage-level findings

**PM-1** (moderate, correctness, unverified). Under the title, and "The Ridge Point of the RTX 4090".
> comes close to the 165 TFLOP/s that the specification

and
> so we take $P$ and $\beta$ from the specification sheet

I know of no NVIDIA specification sheet for a GeForce card that states tensor-core throughput or memory bandwidth. Both numbers are derived: 1.008 TB/s is 384 bits times 21 Gb/s per pin (certain); 82.6 TFLOP/s fp32 is 16,384 cores times two FLOPs per FMA times the 2.52 GHz boost clock (certain); 165.2 is twice that, and the factor two for bf16 with fp32 accumulation is what I cannot confirm from a public source (the hardware section states it as "GeForce cards run fp8 and fp16 tensor arithmetic at half rate when they accumulate in fp32"). The stored measurements (170 and 172 TFLOP/s at n = 8192) fit a peak of about 165 to 170, so the number is plausible. Fix: say "derived from the specification" and give the derivation once (one line here or in the hardware section), so that a reader with another card can redo it; "replace the two constants with the values from its specification" is not actionable when the datasheet does not list them.

**PM-2** (moderate, thought). "Counting FLOPs and Bytes".
> $\beta = 1.008 \times 10^{12}$ bytes/s. Which budget runs out first depends

The reader has met this already. `chapter_transformers/kv-cache.md` (`subsec_kv-cache-prefill-decode`) defines arithmetic intensity and the roofline and says of the same card "performs up to about 83 TFLOP/s in fp32 ... so a computation needs about 80 FLOPs per byte". Here the card has 165 TFLOP/s and a ridge of 164, with no word that P depends on the format, until "a ridge point of about 82 for tf32" appears 600 lines later. Add one sentence where P is introduced: P is a property of the format and of the kind of operation (165 for bf16 matrix products, 83 for fp32 and tf32), which is why the transformer chapter quoted a ridge of 80. Acknowledge the earlier meeting ("we used this once, for decoding; here is the model in full").

**PM-3** (moderate, correctness). "The Roofline Model".
> That is less than a six-hundredth of the compute roof.

The roof of 165 TFLOP/s is the tensor-core rate, which only matrix products (and convolutions) can use. An elementwise kernel runs on the scalar units, whose roof is lower (about 83 TFLOP/s fp32 on this card), and a sine is dozens of FLOPs. The comparison is safe as an upper bound but it teaches that there is one P per card, and the later sentence "additional arithmetic costs no time until the intensity approaches the ridge point" then refers to a ridge that is not the 164 of the figure. Fix: say that P is the roof for matrix products and that other kernels have a lower one; the conclusion (elementwise operations are bandwidth-bound by two orders of magnitude) is unchanged.

**PM-4** (minor, correctness). "The Ridge Point of the RTX 4090".
> so the roofline model declares every such product with $n$ above about 500

The β of the model is the bandwidth of the card's DRAM. A $256 \times 256$ bf16 matrix is 128 KB and a $512 \times 512$ one is 0.5 MB; three of them fit many times over in the 4090's L2 cache (72 MB, the "about a tenth of a gigabyte" of the next section). For operands that fit on the die the DRAM bandwidth is not the limit, so "bandwidth-bound below n = 500" is not what the roofline with a cache-aware β would say. One clause suffices: the model as used here applies to operands larger than the on-die cache.

**PM-5** (minor, currency). Same subsection.
> 120 FLOP/byte (RTX 5090) or about 300 (H100, B200). Only large matrix

The values agree with `tab_gpu_specs` (117, 295, 281). "Other current GPUs" will date; cite the table instead of repeating its numbers.

**PM-6** (moderate, thought, timing that must survive re-execution). "Synchronization Points".
> Here the reads make the loop about one and a half times as slow.

Stored: 0.064 s against 0.043 s (PyTorch), 0.371 s against 0.106 s (JAX, "three to four times"). Both ratios depend on the host CPU as much as on the GPU, since they are $(h + d + \sigma)/\max(h, d)$; on a laptop host or a faster card they will differ. The explanation also does not quite fit its own numbers: with 43 µs per step for the larger of host and device time and a device time of at least 26 µs for a $1024^3$ tf32 product, "host time plus the device time plus the latency" predicts at least 69 µs, and the cell measures 64, because a step is three calls and the device starts on the first while the host issues the others. Fix: state the ratio as a formula and a range ("between about 1.3 and 4 on our machine, depending on the framework; the smaller the device time per step, the larger the penalty"), and say "at most" before "the host time plus the device time".

**PM-7** (minor, timing). "A Benchmark Helper".
> more than 150 TFLOP/s, over 90% of its specification.

Stored: 0.90 ms, which is 152.7 TFLOP/s and 92%; the sweep below measures the same product at 154.1. "More than 150" fails on a 2% fluctuation. Write "about 150 TFLOP/s, roughly nine tenths of the peak".

**PM-8** (major, correctness and explanation). "Matrix Products across Sizes".
> per call that barely depends on the size cannot be spent on arithmetic. It is the time $h$ that the host needs to issue one call:

The first sentence is right; the second does not follow from the cell. A size-independent time can also be a device-side fixed cost: a kernel that fills only a few of the 128 SMs takes about the same time for n = 256 and n = 512, which the section's own next paragraph says. In the PyTorch output the two explanations cannot be told apart at n = 512: 12.8 µs is 21.0 TFLOP/s, and the tile model of Exercise 7 (16 tiles on 128 SMs, one eighth of the peak) predicts 20.6. In JAX the floor is 63 to 90 µs and clearly host-side. The section has the tool to settle it: `kernel_times` on the n = 256 loop gives the device time per call; if the device is busy a small fraction of the 12 µs, the time is host time. As written, the reader is asked to accept the attribution, and Exercise 4 then asks them to distinguish exactly these two causes. I could not run the check (no GPU here). Fix: either show the busy fraction (one line of output) or write "most of it is the time h" and point to the profile.

**PM-9** (moderate, thought). Same subsection.
> $n \approx (hP/2)^{1/3}$: for $h = 20\,\mu\textrm{s}$ and the peak of the

The printed output directly above gives 12.3 and 12.8 µs per call (PyTorch) and 63 to 90 µs (JAX); the next sentence says "The time per call at the smallest sizes in the output above is the value of $h$"; the hardware section says "5–15 µs"; its Exercise 8 uses 8 µs. Nobody's h is 20 µs. With the measured 12.5 µs the knee is at n = 1010, which matches the output better (n = 1024 already reaches 128 TFLOP/s, 78% of peak); with JAX's 65 µs it is at n = 1750. Use the printed h, and give both tabs' knees, since the factor of five between the eager PyTorch floor and the compiled JAX floor is one of the more surprising facts in the section and the PyTorch reader never sees it.

**PM-10** (minor, correctness against stored output). Same subsection.
> and each doubling of $n$ multiplies the throughput by five to ten

Stored ratios: 7.8 and 6.1 (PyTorch), 6.0 and 11.4 (JAX). The model two lines later says exactly eight ($2n^3/h$). Write "by about eight, as the cube of n predicts" and drop the range.

**PM-11** (moderate, correctness against stored output). Same subsection.
> The share of the peak that it can reach is then at most

"at most the share of SMs that hold a tile" is stated as a fact without a tile size, a measurement or a source. Exercise 7 supplies the tile size ($128 \times 128$, from NVIDIA's guide), and the profile's kernel names contain `128x128` and `256x128`. With that tile size the n = 1024 product has 64 tiles and could reach at most half the peak; the stored PyTorch output shows 128 TFLOP/s, 78% of it. So either the bf16 kernel uses smaller tiles or the bound does not hold as stated (unverified which). The exercise asks the reader to find where the model fails, which is fair for an exercise, but the body should not assert the bound. Fix: reduce the paragraph to "a product can also fall short for a reason inside the device: a kernel too small to occupy all 128 SMs; Exercise 7 models this", or state the tile size and show the prediction next to the measurement.

**PM-12** (moderate, pedagogy). "Diagnosis and Fixes", the table.
> | compute-bound | device busy; intensity above the ridge; time grows with the FLOPs | a format with faster tensor cores; more devices |

The lever the reader controls most easily is missing from every row: the batch size. The section's first result is that intensity grows with B, so a larger batch moves a bandwidth-bound layer toward the ridge and an overhead-bound step toward fewer, larger calls. "A narrower format" is missing from the bandwidth row although it halves the bytes (the hardware section says so). And the compute row omits the plain answer, do less arithmetic. Add them; a table of fixes that sends the reader to tensor-core formats and more GPUs before "raise the batch size" misorders the advice.

**PM-13** (minor, language). After the table.
> of the third fix for bandwidth: it computes attention in tiles held on the

"The third fix for bandwidth" means the third item of a semicolon list in one table cell; the reader has to count. Name it: "an example of an algorithm that reuses data on the chip".

**PM-14** (moderate, code). "Recording the Kernels", JAX tab, and "Reading the Profile".
> import collections  # Already imported by the d2l library

Three imports sit in the middle of the notebook, one with an apologetic comment and a stray `#@save` between them; the house rule is one imports cell. In `show_profile` the regular expression `'gemm|xmma|cutlass|matmul|sdpa'` decides the headline number "matrix products 79%" and is not explained; `xmma` and `sdpa` match nothing in this section. The shared sentence "Three quarters or more of the device time goes to a handful of matrix-product kernels" covers 79% (PyTorch) and 97% (JAX), which are not comparable: in JAX the update is fused into `gemm_fusion_dot`, so the "matrix products" share includes bandwidth-bound work. Say so in the shared prose, not only in the JAX tab, because it is the first sight of fusion changing a profile.

**PM-15** (moderate, pedagogy and language). End of "Reading the Profile".
> The four steps of the method can now be stated with their tools.

This paragraph is a second ending placed before the Summary (author decision: one ending per section). It restates the introduction's four bold verbs and the table, then adds "We can now place any operation or program in its regime", which overclaims: the section has classified three operations and one MLP step. Move the one new thing (which tool serves which step) into the Summary and keep only the bridge to the hardware section, which is good: "We cannot yet say where $P$ and $\beta$ come from, or why the ridge point has risen over the hardware generations".

**PM-16** (minor, pedagogy). Introduction.
> intensity, tells which of the first two applies. Only measurement reveals

See structural finding 1. Replace "Only measurement reveals the third" with a statement that the third needs one measured constant, the host time per call, after which it too can be predicted.

**PM-17** (minor, explanation). "The Ridge Point of the RTX 4090".
> print(f'ridge point = {peak_tflops / bandwidth_tbs:.0f} FLOP/byte')

A cell that divides two constants typed in the line above and prints 164 is not an experiment, and the heading names a device. Fold the constants into the imports or the sweep cell, give 164 in the text, and retitle the subsection "The Ridge Point".

## Pattern: timings that must survive other hardware

The section anchors every number to "the RTX 4090 that runs this book's notebooks", which is honest, and tells the reader to substitute their own P and β. About a dozen statements quote a measured time or ratio. Sorted by how well they travel:

- **Hardware-independent (keep as they are):** "eight to nine times" for the chain (a ratio of byte counts); "All four operations take the same time"; the shape of the sweep.
- **GPU-dependent but stated as a fraction of a roof (fine with "on our card"):** "within about 15% of its specification"; "about 0.9 TB/s, close to the 1.008"; "0.4 ms at the card's bandwidth".
- **Host-dependent, will not reproduce on another machine and are not flagged as such:** "less than two percent" in the introduction and the Summary (it is $2n^3/(hP)$, so it depends on the host CPU and the framework: 1.6% in PyTorch, 0.3% in JAX); "about one and a half times as slow"; "three to four times"; "between about 60 microseconds and a tenth of a millisecond"; the 20 µs of PM-9.

For the third group, say once that h is a property of the host and the framework and varies by a factor of ten between setups, and express the claims through h.

## Code and experiments

- `performance-model-counting-flops-bytes-and-arithmetic-intensity`: prints a quotient of two literals (PM-17).
- `performance-model-measuring-without-lying-2`: a good experiment; prose reads it slightly beyond the model (PM-6).
- `performance-model-the-sweep-mapping-our-gpu` with `-1` and `-2`: read for more than they show on the host-time attribution (PM-8) and on the tile bound (PM-11). The plot cell's bound has its corner at the ridge while the points sit far below it for three of six sizes; say in the text that this gap is the subject of the next paragraphs (it does, well).
- `performance-model-matrix-products-across-sizes-3` (JAX only): confirms the floor is independent of the product; useful, but the PyTorch reader has no counterpart.
- `performance-model-three-regimes-1` and `-2`: the best cells of the section. Stored outputs match the prose in both tabs.
- `performance-model-profiling-a-training-step-2`: prose matches the stored profile (99% busy, 79%, 16% for the update). No prediction precedes it (structural finding 3).
- Numbers no cell prints: the 2.7 and 0.5 TFLOP/s behind "less than two percent" are printed; the ridge of 82 for tf32 and the intensity of 340 are computed in prose only, which is fine.
- No TensorFlow or MXNet material. Tabs: nine JAX-only and five PyTorch-only prose blocks in the body. Most are API notes; two carry results (the 1.5 and 3 to 4 ratios; the fused update). Results belong in shared prose with both numbers.

## Exercises

Nine exercises, three starred. The set is strong and graded from a two-line bound (1a) to a project (9). Specifics:

- **Exercise 2** is the cell `hardware-batched-decoding` of the next section with a different width. If that cell moves here (structural finding 2), cut 2b and keep 2a and 2c.
- **Exercise 7** (tiles and waves): the model's prediction for n = 1024 (half the peak) is contradicted by the book's own stored sweep (78%). Sub-question (a) asks the reader to compare, so this may be intended; then say in the exercise that the model is deliberately crude, and do not assert its bound in the body (PM-11).
- **Exercise 4**: operation D (a $1024^2$ product in 40 µs) differs from the stored 16.8 µs for the same product. It is hypothetical ("Suppose"), but a reader who has just run the sweep will notice; change the size or the time.
- **Exercise 3** numbering: the tab blocks start at "3." and the next item at "4." while the others use "1."; harmless in the build, confusing in the source.
- Missing: an exercise that asks for a paper prediction of a whole step (FLOPs and bytes of each layer of a small network, predicted time, then measurement). Exercise 9 does this for ResNet-18 as an extended project; a ten-minute version for the MLP of the section would close the loop for every reader (or becomes unnecessary if structural finding 3 is adopted).

## Still open from October

The October review said this section "is a model opening" and listed "two matrix sizes on one GPU" as a keeper; agreed. Its note that "roofline" is used before chapter 13 defines it (`kv-cache.md`) is still open from this side: this section does not acknowledge the earlier use (PM-2).

## For the chapter review

- Order is right: this section must come first, and it sets up the hardware section with a clean question. The seam between the two is misdrawn in one place: the skinny-product sweep that tests the roofline lives in the hardware section, and the SM and tile paragraph that belongs with the GPU's internals lives here.
- Overlap: with Basics (`subsec_async_dispatch`, about 500 words re-taught); with chapter 11 (`subsec_kv-cache-prefill-decode` introduces intensity and the roofline with a ridge of 80; this section uses 164 without reconciling); with the hardware section (the bandwidth measurement is made twice, 0.9 TB/s both times).
- A chapter-wide constant needs one owner: the host time per call is 12 µs (printed), 20 µs (knee), 5 to 15 µs (hardware), 8 µs (hardware Exercise 8), 60 to 100 µs (JAX compiled call). Likewise P and β: decide whether the chapter argues from the derived peak (165, 1.008) or from the measured values (about 170 and 0.9), and say which numbers are derived.
- Suspected chapter pattern: results stated inside framework tabs, so that each reader sees half the evidence; and timings quoted from the development machine without separating what depends on the GPU from what depends on the host.
- Assumes from Basics: bf16, tf32, tensor cores (`sec_numerics`), `d2l.Benchmark`, GELU. No dependence on the appendix. No forward reference to the Language Models part.
