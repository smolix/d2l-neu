# Case Study: Optimizing a Transformer

Source: `chapter_computational-performance/fast-transformer.md` (read to the `<!-- slides -->` marker at line 1170). About 2,950 words of shared prose, 830 (PyTorch) or 970 (JAX) words of tab prose, 1,140 words of exercises; roughly 5,000 in all. Stored outputs read: `outputs/pytorch/chapter_computational-performance/fast-transformer.json` and the JAX file, including the three profiles, the data-parallel table and the cumulative table. The opening arithmetic, the intensities, and every ratio quoted in the prose were recomputed from the stores.

## Verdict

The section is disciplined and honest, and it is not yet the capstone the chapter needs. Its discipline is real: fixed rules of measurement stated in a table, a correctness check before every timing, three profiles, a change that is predicted to hurt and is measured anyway, and a data-parallel speedup predicted from a formula before it is measured and then met within a few percent. Its numbers are quoted at the precision that survives re-execution. But it opens with a gap of more than four ("the arithmetic of a step needs less than 6 ms ... takes 25 to 30 ms"), closes at "about twofold or more", and never tells the reader where the remaining factor of two went, although the stored profiles contain the answer. Of the four single-GPU changes, one (bf16) does nearly all the work; one (the larger minibatch) is introduced with an argument for why it should help and then gains 1% in PyTorch; and in the JAX tab the bf16 step changes the attention kernel at the same time, against the section's own rule of one change at a time. A prediction precedes two of the six measurements; for the other four the reader is given a reason, not a number, and for one of them the explanation arrives after the result. The single most important fix: make each step a quantitative prediction from the preceding profile (the arithmetic is two lines each and I give it below), and end with an account of the whole step in milliseconds: arithmetic at peak, the shortfall of the matrix products, the kernels between them. Then the section answers the question the chapter introduction asked ("Where does the rest of the time go?") instead of stopping halfway.

On the question asked: does the case study apply every tool of the chapter to one model, with a prediction before each measurement? It applies most tools (S3 lists what is used and what is not) and predicts before two measurements out of six (S1).

## Scores

| dimension | score | note |
|:--|:-:|:--|
| Clarity of explanation | 4 | Each step is clear about what it changes and why. The setup table and the profile readings are models of their kind. |
| Clarity of thought and focus | 3 | One model, one method, one line. But the thesis (order matters) is asserted three times and shown once without the text noticing, one step is confounded in JAX, and the opening question is left half answered. |
| Pedagogy | 3 | Checks before timings and a predicted negative result are excellent. Predictions are missing where they are cheapest, and the training check invites a comparison that comes out against the section. |
| Language | 3 | Correct and plain, and voiceless: a capstone in which nothing is called surprising, disappointing or decisive. A third of the body is tab prose, much of it environment variables. |
| Ease and enjoyment | 3 | The data-parallel prediction is satisfying. The single-GPU half is a sequence of small ratios told at equal weight, and the ending is flat. |

## What works

- **The opening arithmetic** (lines 13 to 23): $6 \times 1.9 \times 10^7 \times 8{,}192 \approx 9 \times 10^{11}$ FLOPs, under 6 ms at 165 TFLOP/s, against a measured 25 to 30 ms (stored: 25.1 and 27.2 ms). Recomputed: $9.3 \times 10^{11}$ and 5.7 ms. A gap the reader can feel, in one paragraph.
- **The table of measurement rules** under `### Throughput Measurement`, and the sentence "One timed window of 50 steps varies by a few percent from run to run, so we read ratios to one decimal place." The prose then keeps to it. This is the hedging discipline the house rules ask for, and exercise 1 lets the reader test it.
- **A check before every timing**: compiled against eager logits (7.4e-06), bf16 against fp32 logits (1.9e-03 and 1.0e-03, predicted beforehand as "two to three significant digits"), bound against unbound step (0.0e+00), checkpointed against plain (`allclose`). The bf16 check is a prediction that could fail and does not.
- **Why the width is 512** (lines 93 to 104). The model is changed for a stated reason, with the intensity computed (250 to 370 FLOPs per byte in bf16; recomputed 248 and 356) and the unchanged model left as exercise 6. Start simple, say so.
- **Checkpointing as a predicted loss**: "our method predicts that checkpointing can only slow the step down. We measure it to check the prediction". Stored: 0.85 and 0.79 in throughput, memory from 8.6 to 3.2 GiB and 7.4 to 2.6 GiB. A case study that includes the tool that should not be used, and says so, teaches more than one that only stacks gains. Keep.
- **The data-parallel prediction** (`eq_ft-dp` and the cell `fast-transformer-rungs-each-one-measured-8`). Predicted 0.84, 1.27, 1.52, 2.69; measured 0.91, 1.36, 1.63, 2.92 (PyTorch). Predicted 0.88, 1.32, 1.50, 2.62; measured 0.88, 1.31, 1.71, 3.17 (JAX). The model is called "a floor" before the measurement and the measurements sit on or above it. This is the best experiment in the three multi-GPU sections: the cost model has power here, because it predicts a loss for one configuration and a gain for the other.
- **"A ranking of techniques that measures each one alone on the same baseline can therefore mislead."** The right lesson for a capstone.
- **Exercises 1, 2, 4 and 5** (see below). Exercise 4, the causality gate, is the most valuable problem in the chapter.

## Structural findings

**S1. Predict each ratio from the profile before measuring it.** The section says of its method "find the resource that limits it, change one thing ... and measure again", and for data parallelism "we predict the gain from a cost model before we measure it". The single-GPU steps deserve the same, and the stored profiles make it a two-line calculation each time. With the PyTorch store:

| change | prediction available from the previous profile | measured |
|:--|:--|:-:|
| compile | products are 64% of 24.4 ms of device time and will not change; at most the other 8.8 ms and the 0.6 ms of idle time can go, so at most 1.6 | 1.13 |
| bf16 | products are 71% of 22.2 ms; if only they halve, 1.55; if every kernel halves its bytes as well, 2.0 | 1.75 |
| batch 512 | the device is busy for the whole step, so only fixed device work can be divided: AdamW moves about 0.5 GB per step, 0.5 ms of 12.7 ms, so at most about 4% | 1.01 |
| checkpointing | one more forward pass of the blocks, 8 FLOPs per parameter and token instead of 6: about 0.75 | 0.85 |

(JAX: idle time of 13% predicts at most 1.15 for the bound step, measured 1.14; bf16 on products alone predicts 1.48, measured 1.97, see FT-8; checkpointing 0.79.) Each line turns "we tried it and it gave 1.1" into "the profile bounded the gain at 1.6 and we got 1.13, so most of the short kernels are still there", which is what the third profile then shows. The checkpointing line turns a prediction of sign into a prediction of size, and its miss (0.85 against 0.75) is explained by the share of the step that is not block computation. I would put the prediction in one sentence before each cell and the comparison in one sentence after.

**S2. Close the account.** The section ends: "The fastest configuration still leaves about half of the card's bf16 peak unused. What remains lies in the memory-bound kernels between the matrix products ..., in the efficiency of the products themselves, and in the width of the numbers." The third profile lets the authors be exact. PyTorch store at batch 512: the step takes 100 ms; its FLOPs ($6 \times 1.9 \times 10^7 \times 65{,}536 \approx 7.5 \times 10^{12}$) need 45 ms at 165 TFLOP/s; matrix products take 57% of the device time, 57 ms, so they run at about four fifths of the peak; the remaining 43 ms belong to the kernels between them. In the unit of the opening: 12.5 ms per 8,192 tokens against the 5.7 ms of pure arithmetic, of which about 1.5 ms is the shortfall of the products and about 5.4 ms is memory traffic. (JAX: 91 ms, products 54 ms, the rest 37 ms.) A closing paragraph or a three-row table with these numbers would answer the chapter introduction's question, show the reader that the roofline's two regimes split the final step roughly half and half, and say what would have to change to go further (fewer bytes between products, not faster arithmetic). The same account for the baseline is also instructive: in tf32 the arithmetic needs 11 ms, the products take 15.6 ms and everything else 9 ms, so the opening's "three quarters or more of the card's capacity go unused" compares a tf32 step with the bf16 peak and half of that gap is the format alone. Say so there.

**S3. Which tools of the chapter are applied.** Applied: FLOP counting and intensity (to choose the width and for the last table column); correct timing with `d2l.Benchmark`; the profiler, three times; compilation and fusion (`torch.compile`, CUDA graphs) or the removal of host work (`nnx.jit_partial`); mixed precision with a fused attention kernel; memory measurement; activation checkpointing; the cost model of data parallelism, DDP and the sharded JAX step. Not applied, and not mentioned: gradient accumulation (`subsec_mp-accumulation`), which is the tool that would let the larger minibatch be tested without its memory; sharding (`subsec_mgp-fsdp`), reasonably, since the model has 19 million parameters, but then the chapter never runs it anywhere; the input pipeline and pinned memory of `sec_hardware`, which the section removes by putting the corpus on the GPU. One sentence in the setup should say which tools the case study leaves out and why ("the training state of this model is 0.3 GiB, so nothing here needs accumulation or sharding"). The roofline itself is used once, for the width; S2 would use it again at the end, where it belongs.

**S4. The two tabs take different second steps, and the JAX third step is two changes.** The shared heading `### Host Overhead and Small Kernels` covers a PyTorch step that is about fusion (the device was busy 97% of the time; the host was never the limit) and a JAX step that is about host work (XLA had already fused; the device idled 13% of the time). The shared prose is written carefully so that both are true, and the two profiles support the two diagnoses, so this is not the case of tabs drawing opposite conclusions. But a reader of one tab never sees the other regime diagnosed in the case study, and the table rows differ ("+ compile" against "+ jit_partial"). Since the difference is the best demonstration in the chapter that the same model in two frameworks sits in two regimes, say it in shared prose in two sentences and show the two baseline profile lines side by side (they are short). The JAX host overhead is specific to `nnx.jit` walking Flax objects (8 ms for a call that does nothing); tell the reader that this is a property of the convenience wrapper and not of JAX. For the confounded bf16 step, see FT-8.

**S5. Tab prose is a third of the body and much of it is plumbing.** Environment variables and workarounds in prose: `TEARDOWN_CUPTI=1`, `--xla_gpu_enable_triton_gemm=false`, `NCCL_LOCAL_REGISTER=0`, the release of CUDA-graph memory on GPU 0 before the first launch (also a code comment in the same cell), `torch._inductor.config.compile_threads`. Each is true and each is needed to reproduce the notebook; none helps the reader understand why the step got faster. Move all of them to code comments or to one "Running this notebook" note. The JAX tab on reduced precision (lines 482 to 502, 200 words) is the longest and mixes an idea the reader needs (the attention kernel changes) with API detail (`dtype`, `param_dtype`).

**S6. Hardware dependence.** The sequence of bottlenecks is a fact about one model on one RTX 4090 with one host CPU. The section says which hardware it uses and hedges its ratios well, but the shared prose states the diagnoses as facts ("The device is now busy for nearly the whole step"). On a faster GPU the eager PyTorch baseline would wait for the host and the first step would gain more; on a slower one the first step would gain nothing. Add three sentences near the end telling the reader what to expect elsewhere and that the order of changes follows from the profile, not from this section; and extend exercise 8 ("another model") with "or another GPU: replace the two constants of `sec_perf_model` and predict which step changes most before you run it". A reader without a GPU can run nothing here; that is acceptable for this section but should be said in the setup.

## Passage-level findings

**FT-1** (minor, explanation). Opening.
Quote: "step needs less than 6 ms. A plain training step, in fp32 with an AdamW"
The 6 ms are at the bf16 rate; the plain step runs in tf32, whose peak is half. "Three quarters or more of the card's capacity go unused" is true only against a format the step does not use. Not wrong, since bf16 is the card's capacity, but the reader later learns that bf16 alone recovers most of the gain and may feel the opening counted the same factor twice. Add a clause: "half of that gap is the number format, as we will see".

**FT-2** (minor, explanation). Under `### Model and Data`.
Quote: "between 250 and 370 FLOPs per byte in bf16. The ridge point of the"
The first two configurations run in tf32 with four bytes per number, where the same products have half the intensity (125 to 185) against a tf32 ridge of about 82 FLOPs per byte. The conclusion (compute-bound) holds in both formats, but the paragraph argues with bf16 numbers about a baseline that is not in bf16. One clause fixes it.

**FT-3** (minor, language: plumbing). PyTorch tab under `### Baseline`.
Quote: "`TEARDOWN_CUPTI=1`, which detaches the profiler of the next cells from the"
See S5. The reader meets a profiler environment variable before the first measurement. Code comment.

**FT-4** (minor, correctness: stored output). JAX tab after the first profile.
Quote: "The device runs kernels for nine tenths of the step or more and waits for"
Stored: "device busy 23.61 ms (87%)". Below nine tenths. The later sentence "a little more than a tenth when the device waited a tenth of the step" has the same problem (13% idle, gain 1.14). Write "for about seven eighths of the step" and "by as much as the idle eighth allowed".

**FT-5** (moderate, pedagogy: no prediction; two changes in one). PyTorch tab under `### Host Overhead and Small Kernels`.
Quote: "Our step is in the second regime: the device hardly waits, but much of its"
Two things. First, the tab applies fusion and CUDA graphs together (`mode='reduce-overhead'`) after diagnosing that the host is not the limit, so the second of the two can gain at most the 3% of idle time, and the reader cannot tell which did what. The stored profiles can: device time fell from 24.4 to 22.3 ms (fusion) and the idle 0.6 ms closed (replay). Say so in one sentence; the tab's last paragraph has the observation without the attribution. Second, no bound is stated before the cell (S1): "at most 1.6, since the products will not change" is available, and against it "about a tenth" becomes a finding: fusion removed a quarter of the non-product time and three quarters of it is still there.

**FT-6** (minor, pedagogy: order). JAX tab after the bound step.
Quote: "Throughput rises by as much as the idle time of the first profile"
This is the prediction, stated after the result. Move it before the cell: "the device idled for an eighth of the step, so removing the host's work can gain at most that".

**FT-7** (minor, thought). Shared prose after the second profile, and the tolerance in the cells.
The compiled-logits check asserts `error < 0.01` where the measured difference is 7.4e-06, and the prose says "three or more significant digits". A tolerance a thousand times looser than the observation would pass a compiled model with a real defect in one head. Tighten to 1e-4 and say "five". The same for `assert error < 0.05` on bf16 logits measured at 1e-3 to 2e-3: 0.01 matches the prose ("within one percent").

**FT-8** (major, thought: confounded step). JAX tab under `### Reduced Precision`.
Quote: "to select cuDNN's fused kernel instead, which accepts only 16-bit inputs,"
The JAX "+ bf16" configuration changes the number format and replaces XLA's attention, which materializes all query-key weights, by cuDNN's fused kernel. The measured 1.97 is then explained as the sum of both ("Faster matrix products alone could not do this"). The section's rule is "change one thing", its table row is named "+ bf16", and its Summary credits "bf16". The shared prose and the PyTorch tab attribute a factor of 1.7 to precision; the JAX reader is told about 2 and that part of it is something else. Options: (a) split the JAX step in two rows, "+ bf16 (XLA attention)" and "+ fused attention", which costs one cell and teaches that the attention kernel matters, a point the chapter makes in `sec_attention-at-scale` and the PyTorch baseline gets for free; (b) if the fused kernel cannot be isolated because it accepts only 16-bit inputs, rename the row "+ bf16 and fused attention" in the JAX tab and say in shared prose that the PyTorch baseline already uses a fused attention kernel. I prefer (a) with bf16 first.

**FT-9** (minor, explanation). Shared prose and PyTorch tab under `### Reduced Precision`.
Quote: "Throughput rises by about 1.7 times. That is less than the factor of two"
The explanation lists what did not halve. With the prediction of S1 the sentence can say more: the products were 71% of the step, so halving them alone gives 1.55; the measured 1.75 shows that the kernels between them got faster too, because they move half the bytes. That is the roofline's second regime responding to the same change, and it is the more interesting half of the result.

**FT-10** (major, thought: the argument predicts a gain and the measurement shows none). Under `### Larger Minibatches`.
Quotes: "among more tokens, and its larger matrix products keep the tensor cores"; "Throughput stays the same. With the update fused and the kernels replayed"
The subsection gives two reasons why a larger minibatch should raise throughput, raises it eightfold, and measures 1.01 (PyTorch) and 1.06 (JAX). The explanation comes afterwards: nothing was left to divide. The second profile, printed a page earlier, already said so (device busy 100%, products 71%), and the fixed device work can be bounded in one line (AdamW moves seven arrays of 76 MB, about half a millisecond of a 12.7 ms step). So the method, applied, predicts "at most a few percent", and the experiment confirms the method. As written, the subsection reads as a change that failed. Rewrite the lead: "A larger minibatch divides the fixed costs of a step among more tokens. Our profile says there is little to divide: the device is busy throughout, and the one fixed cost we can name, the optimizer update, is 4% of the step. We predict a gain of that size and measure it, because the larger minibatch is what the data-parallel runs below need." Three further consequences. (i) The Summary credits "a larger minibatch" with part of the twofold gain; in PyTorch it contributed nothing. (ii) The JAX reading "rises by less than a tenth" (1.06) is at the edge of the stated noise of "a few percent"; do not build the sentence about remaining fixed cost on it. (iii) A larger minibatch is not free in training terms (`sec_batch_size`), which the section knows (exercise 5) and the body never says. A change that gains 1% in throughput and costs sample efficiency would, by the section's own logic for checkpointing, be left out of the following configurations. It is kept because data parallelism needs long steps; say that this is the reason.

**FT-11** (minor, language). Under `### Activation Checkpointing`.
Quote: "checkpointing can only slow the step down. We measure it to check the prediction, and to see how much"
Good paragraph. Add the size of the prediction (S1): one more forward pass of the blocks, so about a quarter slower. The stored 0.85 and 0.79 then read as "a little better than predicted, because the embedding, the loss and the update are not repeated".

**FT-12** (minor, thought). `### Cumulative Comparison`.
The printed table lists "+ checkpointing" as a fifth row with a total of 1.70 (1.88 in JAX), as if it were the next step of the sequence, and the prose then has to say "Checkpointing then lowers the throughput again." It is a branch, not a step. Print it below a rule or mark it "(not kept)", and let `total` stop at batch 512.

**FT-13** (moderate, thought: thesis asserted, evidence unclaimed). `### Cumulative Comparison` and the opening.
Quote: "busy. A ranking of techniques that measures each one alone on the same"
The claim that each ratio depends on its place in the sequence is made in the opening ("removing one changes how much the others matter"), in the method paragraph ("each moves the bottleneck"), and here. No single-GPU measurement shows it: the changes are applied in one order only. The section does contain the proof and does not point at it: data parallelism gives 0.9 on the compiled fp32 step at batch 64 and 1.6 on the bf16 step at batch 512. The same technique, a loss in one place and a gain in the other. Say so here in one sentence, and the thesis is demonstrated. If a single-GPU demonstration is wanted as well, an exercise can apply bf16 to the eager baseline and compare with 1.75.

**FT-14** (minor, explanation). Under `### Predicted Speedup`.
Quote: "tens of milliseconds. That is as long as a compiled fp32 step at batch 64"
"Tens of milliseconds" is where the previous two sections' cost model should be used, and it predicts the number: with the 5 GB/s measured in `subsec_mgp-scaling`, $2N/\beta$ for 76 MB is 31 ms at $k = 2$ and $3N/\beta$ is 47 ms at $k = 4$. The cell then measures 31 and 48 ms in both frameworks. The section instead "substitute[s] the measured time of one allreduce", which discards a prediction that would have been met to the millisecond. Print both: the allreduce predicted from $\beta$, and measured. It is the cleanest confirmation of `eq_dp_cost` in the chapter and it costs one clause.

**FT-15** (minor, correctness: overstatement). End of `### Measured Speedup`.
Quote: "scaling. On this machine, data parallelism therefore"
"helps only once the step on each GPU is long": at batch 64 two GPUs lose (0.9) but four gain (1.3 predicted and measured, in both stores). The condition printed three paragraphs earlier is the precise one: the allreduce must take less time than the step, and what "long" means depends on $k$. Write "pays well only once the step on each GPU is several times longer than the allreduce".

**FT-16** (minor, correctness: inconsistent with the previous section). Same place.
Quote: "move data more than ten times faster than our path through host memory"
The previous section says "hundreds of times the rate measured here" for the same comparison (450 or 900 GB/s per direction against about 2.5), and that is the right figure. "More than ten times" is true and misleading. Use the previous section's number and its reference (`subsec_hw-interconnects`).

**FT-17** (minor, language: plumbing). PyTorch tab under `### Measured Speedup`.
Quote: "notebook's own compiled steps still hold memory, so the notebook releases"
The same fact is a comment in the cell below. Delete it from the prose. The tab paragraph is 190 words, of which the reader needs two sentences: one process per GPU, hence a script; the script reuses `windows` and `make_step` from this section.

**FT-18** (minor, correctness: wrong owner). Under `### Training Check`.
Quote: "compute its loss on the validation text with `d2l.lm_perplexity`"
The reference that follows is `sec_lstm`. `lm_perplexity` is defined and saved in `chapter_optimization/adam.md` (`sec_adam`); `lstm.md` itself cites it as "of `sec_adam`". Change the reference.

**FT-19** (major, pedagogy: a check that cannot fail, and a comparison that goes the wrong way). `### Training Check`.
Quote: "reached about 1.35 after 1,000 steps of 8,192 tokens with a tuned"
Two problems. (i) The acceptance test is "validation loss below the bigram level of 2.45". Any model that trains at all passes it, and exercise 4 says as much about one defect. A check with power is available at the same cost: train the baseline configuration and the final one from the same initialization on the same minibatches for the same number of steps and compare the two loss curves, which should agree to within the bf16 rounding of the gradients. That would catch a wrong learning rate under DDP, a broken cast, a dropped weight decay. (ii) The paragraph supplies two reference numbers "to set the scale", and against them the fast configuration looks bad: `sec_gpt` reached 1.35 after 8.2 million tokens; this run processes 19.7 million tokens and ends at 1.63 (PyTorch store) or 1.86 (JAX store). An attentive reader concludes that the optimized step trains worse on more data. The reasons are known to the authors (a constant learning rate, 300 steps where the reference took 1,000, a minibatch eight times larger, a wider model) and are the subject of exercise 5, but the body does not say a word. Either remove the 1.35 or, better, use it: "The loss is higher than the 1.35 of `sec_gpt` although this run saw more than twice the tokens. Throughput is not progress: the larger minibatch takes fewer steps per token (`sec_batch_size`), and we did not retune the schedule. Exercise 5 works out when the trade is worth it." That is the most important caveat a performance chapter can end on, and here it would be earned by the section's own numbers.

**FT-20** (minor, correctness: unverified). Last paragraph before the Summary.
Quote: "the RTX 4090 run fp8 at twice the bf16 rate, but training in fp8 needs a"
Consistent with `hardware.md` (330 TFLOP/s in fp8). I can confirm that the card's tensor cores run 8-bit integer products at twice the 16-bit floating-point rate; whether floating-point fp8 (E4M3, E5M2) products are exposed on this generation at that rate through the frameworks the book uses I cannot confirm. The owner of the claim is `hardware.md`; flag it there.

**FT-21** (moderate, language: second ending and hedge). Last paragraph before the Summary, and the Summary.
Quotes: "The fastest configuration still leaves about half of the card's bf16"; "A training step can be made about twice as fast or more by a sequence of"
The paragraph before the Summary is a closing that lists what is left and where the book goes next; with S2 it becomes the section's last result and earns its place. The Summary then says the main number twice in six sentences ("about twice as fast or more", "about twofold or more") in a phrase that hedges in both directions; the body uses the same phrase a third time. The stores say 2.0 and 2.4. Write "by a factor of two to two and a half" once.

**FT-22** (moderate, enjoyment: voice). The section as a whole.
The October review quoted this file for an extended money metaphor; the rewrite removed the mannerisms and with them every sign of an author. A capstone should judge. Things the authors know and do not say: that one change of the four did nearly all the work and the profile said which before it was tried; that the compiler, to which the chapter gives a whole section, gains a tenth here because this model's time is already in large products; that the larger minibatch is a non-event; that after everything half of the step is still memory traffic, which is the chapter's thesis from its first page. Each is one sentence, each is true, and each is what a reader would repeat to a colleague.

## Code and experiments

- `fast-transformer-the-subject-1`: "19.0M parameters, 270M training bytes". Matches.
- `fast-transformer-rung-0-baseline-profiled` and `fast-transformer-the-baseline-2`: 25.1 ms, device busy 97%, products 64% (PyTorch); 27.2 ms, 87%, 65% (JAX). Prose matches except FT-4.
- `fast-transformer-rungs-each-one-measured-1`: PyTorch 1.13 ("about a tenth": holds). JAX no-op calls 8.37, 1.73 and 0.23 ms ("about 8 ms, a third of the device time": holds; "removes most ... nearly all of the rest": holds).
- `fast-transformer-host-overhead-1` (JAX): the stored difference is exactly zero, as it should be; good check.
- `fast-transformer-rungs-each-one-measured-2`: 1.75 and 1.97. Prose matches; see FT-8 for what the JAX number measures.
- `fast-transformer-rungs-each-one-measured-3`: 1.01 and 1.06; 8.6 and 7.4 GiB. "Well under half of the card's 24 GiB" is generous for 8.6 GiB (36%); "about a third" is what the store says.
- `fast-transformer-larger-minibatches-1`: products 57% and 59% ("half or more": holds). The PyTorch reading of the largest generated kernel as the backward pass of SwiGLU matches the kernel name in the store.
- `fast-transformer-rungs-each-one-measured-4`: 0.85 and 0.79; 3.2 and 2.6 GiB. "A tenth to a quarter" and "less than half": hold.
- `fast-transformer-rungs-each-one-measured-8`: all eight predictions and measurements match the prose. The PyTorch data-parallel script compiles without CUDA graphs, so its one-GPU reference (365,575 and 663,763 tokens/s) is not the notebook's configuration; the text says so. Fine.
- `fast-transformer-the-waterfall`: totals 2.00 and 2.39 at batch 512; 74.6 and 82.2 TFLOP/s ("roughly half of the 165": holds).
- `fast-transformer-the-waterfall-1`: 5.62 to 1.68, validation 1.63 (PyTorch); 5.69 to 1.93, validation 1.86 (JAX). Prose quotes neither, correctly. The two frameworks differ by 0.2 nat after the same 300 steps, which is larger than I would expect from seeds alone and worth a look by the authors (different default initializations or precision of the embedding path are candidates). See FT-19.
- Numbers no cell prints: the intensities 250 to 370 (correct); the 76 MB of gradients (correct: four bytes times 19.0 million).
- No cell only confirms a derived result. Every measurement here can come out otherwise.

## Exercises

Eight, 1,140 words, a healthy 30% of the body. The set is the best of the three sections.

- **1 (noise floor)**: teaches blocked against alternating designs under drift. Excellent, and it audits the section's own claim about precision.
- **2 (model FLOPs utilization)**: the ratio $T/(6d)$ (4% here, a third at 1,024 tokens) and the distinction between required and executed FLOPs. Good; part 3 names what S2 asks the body to state.
- **3 (fixed cost of a step)**: this is the prediction that FT-10 wants in the body (part 2 computes the AdamW bytes). Move part 2's one line into the body and keep the fit as the exercise.
- **4 (causality gate)**: a silent bug that every check in the section passes, and the reader writes the test that catches it. Keep exactly as it is, and consider citing it in the body where the training check is discussed (FT-19).
- **5 (throughput against time to target)**: the caveat the body lacks. The threshold formula is right (re-derived: batch 512 is faster to target when $b_{\textrm{noise}} > (512 - 64\rho)/(\rho - 1)$; with the stored $\rho$ of 1.01 the threshold is about 45,000, so on this section's own PyTorch numbers batch 512 cannot pay on one GPU). The suggested values $\rho = 1.2$ and $1.7$ are far from what the section measured; add "and for the $\rho$ of the cumulative table".
- **6 (width 256)**: good use of the roofline.
- **7 (data-parallel overlap)**: good; "What does a negative value of $\phi$ indicate?" is a sharp question. Needs two GPUs for part 2; part 1 uses the printed table and runs on paper.
- **8 (another model)**: add the variant "or another GPU" (S6).
- To add: one exercise on order dependence (FT-13).

## Still open from October

The October review listed for this file: draft history ("each corrupted an early draft of this section"), a hardcoded "13.3's whole-step lesson", a sentence broken by a rename, the highest dash density in the book, and the extended metaphor about bf16. All are gone. What the rewrite left behind is the opposite fault, recorded as FT-22.

## For the chapter review

- **Role.** Last teaching section; it should be where the chapter's method pays off. It uses `d2l.Benchmark`, `d2l.kernel_times` and `d2l.show_profile` from `sec_perf_model`, `torch.compile` and CUDA graphs from `sec_compilation`, `d2l.to_bf16`, the memory tools and checkpointing from `sec_memory_precision`, `eq_dp_cost` from `sec_multi_gpu`, and `d2l.torchrun` and `d2l.init_distributed` from `sec_multi_gpu_concise`. Nothing depends on the appendix. The forward reference to the Language Models part (fp8 recipes, custom kernels, more machines) is not load-bearing.
- **The chapter introduction asks where the time goes; this section is where the answer must be given in full** (S2). At present the chapter ends with half of the gap unexplained in numbers, although the profile holds them.
- **The data-parallel experiment here is a better test of the cost model than either experiment in the two multi-GPU sections**, because it predicts a loss and a gain. The chapter review may want to say that those sections can lean on it, or borrow its design (two configurations on either side of the break-even).
- **Sharding is run nowhere in the chapter** (see the review of `multi-gpu-practice.md`, S2); the case study is not the place to add it for a 19-million-parameter model.
- **Patterns shared with the neighbouring sections:** environment plumbing in tab prose (five items here); shared prose that holds for one store (FT-4, minor here); the symbol $\beta$ for the link rate against the memory bandwidth of `sec_perf_model`, both of which this section uses.
- **Hardware.** Ratios are hedged to survive re-execution on the same machine. The sequence of diagnoses will differ on other GPUs; the section should say so once (S6).
