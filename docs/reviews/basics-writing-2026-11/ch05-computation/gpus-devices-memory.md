# GPUs, Devices, and Memory

Source: `chapter_builders-guide/gpus-devices-memory.md` (read to the slides marker, line 2039). Prose a PyTorch reader sees: about 3,300 words plus 550 words of exercises; across all four tabs about 5,800 words in 75 tab blocks, and 27 code cells per framework. It is the longest section of the chapter.

## Verdict

Two sections share this file. The second half ("Asynchronous Execution" from the queue experiment on, and "Training Memory") is good textbook writing: it predicts a number, measures it, and explains the gap. The four memory readings of one training step, in which the third reading falls, and the checkpointing experiment, in which a predicted third more time comes out as a quarter, are the best quantitative teaching in the second half of the chapter. The first third ("Devices and Placement") is an API tour inherited from the first edition: twelve cells and some thirty tab blocks to establish that an array lives on one device and that operands must share it. The section also carries four topics (placement, asynchrony, memory, the `Trainer`), defines a `Benchmark` class it never uses, and opens with a question ("Will a given model fit?") that it answers "no" and then leaves. The most important fix is to cut "Devices and Placement" to about a third of its length so that the reader reaches the queue experiment while still fresh.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought | 3 |
| Pedagogy | 3 |
| Language | 3 |
| Ease and enjoyment | 3 |

## What works

- The first sentence: "Since :numref:`sec_oo-design` our `Trainer` has accepted an argument `num_gpus` and ignored it." It starts from something the reader has typed many times and never questioned.
- The queue experiment. "We queue 32 products of $4096 \times 4096$ matrices and read the clock twice: when Python has issued them, and when the device has finished them." The stored output (0.0006 s against 0.0534 s) makes asynchrony visible in two lines, and the consequence is drawn at once: "the time a Python statement takes says nothing about the time its computation takes."
- The link back to the book's own code: "This is why the steps of the book's `Trainer` return their losses as arrays". A design decision from chapter 2 is finally explained.
- `fig_bg_async_queue` and `fig_bg_allocator` are clear and carry weight; the three nested views of memory are easier to keep apart with the figure than with the prose.
- "What Fills Memory During Training". The prose computes the activations before measuring (three $4096 \times 4096$ fp32 arrays, 64 MiB apiece, "about a third more than the weights"), and the four readings (160, 353, 314, 602 MiB) match. "The third *falls*: `backward` frees the activations as it uses them" is a small surprise that teaches how backpropagation uses memory.
- "The remaining term, the *activations*, is of a different kind." A plain sentence that separates what scales with the model from what scales with the minibatch.
- Activation checkpointing: the count $K/s + s$, the optimum at $\sqrt{K}$, the prediction of four forward passes instead of three, and a measurement (1096 to 452 MiB, 25.7 to 32.2 ms) that the prose reads honestly: "about a quarter longer, a little less than the third that the count of forward passes predicted."
- The practical diagnosis "If memory instead grows from one step to the next, something keeps the arrays of earlier steps alive".

## Structural findings

1. **Four topics, one title that lists them.** The section opening says so itself ("This section answers three questions"), and the `Trainer` is a fourth. Placement and asynchrony are about where and when computation happens; training memory continues `sec_parameters`. The memory material would be stronger next to the bytes-per-parameter accounting it completes: the reader meets $16N$ in 5.2, again in the exercise of 5.5, as $12N$ in 5.6, in this section's opening, and a fifth time at the top of "What Fills Memory During Training". A reviser should consider two sections: "Devices and Asynchronous Execution" (short), and "Training Memory" (the activation term, the allocator, checkpointing), or else move the activation accounting into `sec_parameters` and leave only the measurement here.
2. **"Devices and Placement" is three times too long.** Device objects, `cpu()`, `gpu()`, `num_gpus()`, `try_gpu()`, `try_all_gpus()`, array creation, a second array on a second GPU, the failing sum, the copy, the sum again, the no-op copy, model placement, and where parameters report their device: twelve PyTorch cells, most of which print a device string. One idea is taught (operands must share a device, and copies are explicit and slow), and it is taught by the one cell that fails (`X + Y`). Keep: the helper functions in one cell, the failing sum and its fix, model placement with the rule about creating the optimizer afterwards. Cut: the standalone device listing, the no-op copy subsection, the cell that prints where the weight lives.
3. **The opening question is abandoned.** "Will a given model fit? ... Training it with Adam in fp32 needs 128 GB ... That does not fit." The section never returns to the 8-billion-parameter model. Its last paragraph says "What the `Trainer` cannot do is tell whether a model fits before it runs out of memory." A reader who has followed the 8B model through four sections is owed two sentences: what the tools of this chapter buy (bf16 activations, checkpointing, a smaller batch), that the persistent 128 GB still exceeds one card, and that `chap_performance` spreads it over several.
4. **`Benchmark` is defined and not used.** The class is introduced as packaging the timing rules "for later chapters", and then both timing cells that follow (the pinned-memory copy and the checkpointing comparison) write their own loops. Either use `Benchmark` in the checkpointing cell, which would shorten it and show the class at work, or define it where it is first used.
5. **The last experiment shows nothing.** Five epochs of `ResMLPClassifier` on Fashion-MNIST end by printing `device(type='cuda', index=0)`. The section began by claiming that a GPU "computes much faster than a CPU" and never compares the two. Timing one epoch with `num_gpus=0` and `num_gpus=1`, on this model and on a wider one, would close the section with its own opening claim, and would show the honest result that a small model gains little.

## Passage-level findings

**GPU-1** (minor, explanation). Opening: "than a CPU: a current desktop GPU multiplies two $4096 \times 4096$".
"Much faster" is supported by one number, with nothing to compare it to. Add the CPU time for the same product (a second number in the same sentence).

**GPU-2** (minor, pedagogy). Opening: "128 GB: 16 bytes per parameter for the weights, their gradients, and the two".
The third full restatement of the $16N$ accounting in the chapter (after `sec_parameters` and the opening of `sec_read_write`), with a fourth 1,100 lines later. One clause and a reference would do: "128 GB by the count of :numref:`sec_parameters`".

**GPU-3** (moderate, pedagogy). Heading "### Device Objects" and what follows to "### Copying Between Devices".
See structural finding 2. Every tab block here is a paragraph of naming conventions ("`torch.device('cuda:0')` is the first GPU and its memory, `cuda:1` the second, and plain `'cuda'` is `cuda:0`"; the MXNet history of *context* against *device*). This is reference material and reads as such.

**GPU-4** (minor, language). Under "Device Objects": "and with the count, write two functions that fall back to the CPU when the".
One sentence is split by a four-tab code block: it begins "We can count the GPUs that are available," and resumes after the output with "and with the count, write two functions". In the rendered page the reader meets a paragraph that starts with a lower-case "and". Make two sentences.

**GPU-5** (minor, pedagogy). Under "Copying Between Devices": "What if `Z` already lives on the target device? Then `.to` returns `Z`".
A question nobody asked, answered with an identity check (`Z.to(try_gpu(1)) is Z`). At most a clause in the rule that follows ("moving data that is already there costs nothing").

**GPU-6** (minor, thought). Under "Copying Between Devices": "A transfer between devices is slow compared with arithmetic on either of".
Asserted, never measured in the body; exercise 3 measures it. One number here (the round-trip time of the $4096 \times 4096$ matrix against the 1.7 ms of its product) would make "slow" mean something, and the pinned-memory cell already has the machinery.

**GPU-7** (moderate, language). Caption of `fig_bg_async_queue`: "Python queues kernels k1 through k4 and continues while the GPU runs them in order."
"Kernel" is not defined in this section, and the chapter uses the word for three things: a GPU routine (here, and "operations whose kernels add in a varying order" in `sec_repro`), a weight matrix ("Keras calls a weight matrix a *kernel*", `saving-loading.md`), and, from the next chapter on, a convolution filter. Write "operations" in the caption and figure, as the body text does.

**GPU-8** (minor, correctness). Under "The Execution Queue": "resolution is far finer than a millisecond. The cell above follows the first".
The paragraph lists four rules (warm up; wait before starting and stopping; repeat and divide; use `perf_counter`) and says the cell follows "the first three". The cell does use `perf_counter` and does not divide by the repetitions. Say which rules it follows by name, or drop the sentence.

**GPU-9** (moderate, pedagogy). Under "The Execution Queue": "`Benchmark` packages these rules for later chapters. It calls a function".
See structural finding 4. The two tab blocks after the class explain signature details (`devices`, the JAX return-value requirement) of code the reader will not run in this chapter.

**GPU-10** (moderate, pedagogy). Under "Transfers from the Host": "An ordinary copy first moves the data".
The subsection claims an overlap benefit (Python continues while the copy runs) and then measures something else, the synchronized duration of the copy, where the stored gain is modest (0.55 ms against 0.37 ms). Three of four tabs have no code. It ends "The full treatment of streams, overlap, and several devices is in :numref:`chap_performance`", with "streams" undefined. This is a performance-tuning detail for the chapter that owns it. Cut to two sentences in the queue subsection, or move it whole to `chap_performance`.

**GPU-11** (minor, explanation). Under "Three Views of Memory", PyTorch tab: "After the deletion, `memory_allocated()` falls by a gibibyte while".
The stored readings are 1160.1 and 136.1 MiB. The reader created "an array of 1 GiB" and sees 1160. Say where the other 136 MiB come from (the matrices `A` and `B` of the queue experiment are still alive), or report differences as the next cell does.

**GPU-12** (moderate, thought). Under "What Fills Memory During Training": "common cause is appending the loss array itself, rather than its value, to a".
This contradicts advice given 600 lines earlier. The asynchrony subsection tells the reader not to convert the loss to a Python number every step, and praises the `Trainer` for returning losses "as arrays"; `numerics.md` does `losses.append(loss.detach())` and explains that "the loss values stay on the device". Now appending the array is "a common cause" of a leak. The missing word is *detach*: an array that still carries its computation graph keeps that step's activations alive; a detached array does not. Draft: "a common cause is appending the loss to a Python list without detaching it, so that the list keeps every step's computation graph alive."

**GPU-13** (minor, language). Under "Activation Checkpointing": "intermediate result only to read it once, on the way back. *Activation".
The reader has just finished a section in which "checkpoint" meant a file on disk. One sentence should separate the two uses ("The name is unrelated to the training checkpoints of :numref:`sec_read_write`: here the checkpoints are activations kept in memory").

**GPU-14** (minor, explanation). Under "Activation Checkpointing": "backward pass costs about twice a forward pass, a step then costs about four".
"A backward pass costs about twice a forward pass" is the premise of the prediction and is given without a reason. One clause: for a linear layer the backward pass computes two products (the gradient of the weight and the gradient of the input) where the forward pass computes one.

**GPU-15** (minor, language). Under "Activation Checkpointing", PyTorch tab: "plain closure would look up `start` and `stop` at the time of the call and".
Four lines of prose on Python's late-binding closures. The TensorFlow tab says the same thing in a two-line code comment; do that in every tab.

**GPU-16** (moderate, language). Under "Device-Aware Trainer", PyTorch tab: "`torch.backends.cudnn.benchmark` for the rest of the process, unless".
The paragraph is a change log of `prepare_model`: the dry pass, lazy parameters, `weight_init`, which models are exempt, and cuDNN autotuning with a forward reference to `subsec_deterministic_arithmetic`. October listed "cudnn minutiae" as a cut; it is still here. The reader needs two facts: the model moves once, each minibatch moves at the start of its step. Put the rest in code comments. Each of the four tab blocks also opens with the same sentence about `num_gpus`, which is the first sentence of the section a third time over.

**GPU-17** (moderate, pedagogy). Under "Device-Aware Trainer": "To see the `Trainer` at work, we train a classifier built from the residual".
See structural finding 5. The prose after the run talks about memory accounting, not about what the run showed.

**GPU-18** (minor, language; pattern). MXNet tab under "What Fills Memory During Training": "device-wide count cannot separate the terms, and this tab measures nothing".
Tabs that exist to say the framework cannot do the thing occur throughout the section ("MXNet has no per-segment activation checkpointing ... This tab therefore has no code in this subsection"). "Has no" occurs 8 times in this file. A reader of that tab gets a section with holes; a reader of the other tabs is not affected. Record as a cost of the four-framework policy, not something a line edit fixes.

## Exercises

Five per tab. Exercise 1 (predict peak memory at three batch sizes, then measure, "Where does the prediction break down, and what did it omit?") is the model exercise for this chapter. Exercise 2 extends the checkpointing experiment in the right direction (segment size 1, 4, all). Exercises 3, 4 and 5 are each two-part GPU measurement projects of an hour or more; three of those is one too many, and all five need a GPU. Exercise 4 (precision crossover) carries the timing that `sec_numerics` omitted; if that timing moves into the body (see `numerics.md`, structural finding 2), cut it to its second part. There is no paper exercise for a reader without a GPU; the MXNet version of exercise 2 (the checkpointing schedule worked out on paper) is a good one and should be offered in every tab.

## Still open from October

"A duplicated clause left from editing (`gpus-devices-memory.md:685`)" is fixed. "Device survey" is reduced to one paragraph ("Other accelerators, such as the GPUs of Apple-silicon Macs and Google's TPUs, have device names of their own"), which is fine. "cudnn minutiae" remains (GPU-16).
