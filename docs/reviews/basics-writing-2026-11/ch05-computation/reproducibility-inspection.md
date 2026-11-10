# Reproducibility and Inspection

Source: `chapter_builders-guide/reproducibility-inspection.md` (read to the slides marker, line 1347). Prose a PyTorch reader sees: about 2,300 words plus 580 words of exercises; across all four tabs about 4,100 words of prose in 43 tab blocks and 2,200 words of exercises, and 11 code cells per framework.

## Verdict

The weakest of the four sections, with one of the best passages in the chapter inside it. The opening poses a real question (is a gain of 0.3 points real?) and the section's wisest paragraph answers it (bitwise repetition is for debugging; conclusions must hold across seeds). But no experiment in the section measures how much two runs differ, so the opening question and the closing advice are both taken on trust. The reproducibility half demonstrates the easy things (the same seed gives the same numbers) and only describes the hard ones: neither failure of data-loader workers is shown, and nondeterministic GPU arithmetic is represented by an error message from 3-D average pooling. The inspection half is better. The activation-statistics experiment, with its prediction of a factor $\sqrt{1.5}$ per block, is excellent. The section then ends on a gradient cell whose output no sentence interprets. The two halves are joined by a title and a sentence, not by an argument. The most important fix is to add the one experiment the section is about: the same small classifier trained under several seeds, with the spread printed.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 3 |
| Clarity of thought | 3 |
| Pedagogy | 3 |
| Language | 3 |
| Ease and enjoyment | 3 |

## What works

- The opening paragraph: "Suppose we change one line of a training script and the validation accuracy of a Fashion-MNIST classifier rises by 0.3 points. Before we credit the change, we need to know how much two runs of the unchanged script differ." This is the question every practitioner should ask and few textbooks pose.
- The list of where randomness enters ("initialization draws every weight ..., dropout draws a new mask at every step, the data loader shuffles ...") is complete and brisk.
- The generator-object motivation: "Inserting one extra random call, a new layer's initialization or a stray draw while debugging, shifts every number drawn after it". A concrete failure the reader will recognize the first time it happens to them.
- "A global seed is the convenient form of the same contract: one implicit key that all code shares." One sentence that relates the JAX design to the others.
- The three-number example of non-associativity: "Near $10^8$, adjacent float32 numbers are 8 apart, so $-10^8 + 1$ rounds back to $-10^8$", with output `1.0 0.0`. The smallest possible case, worked completely.
- The paragraph beginning "Bitwise agreement holds only for one machine, one library version, and one configuration of these switches". It states the limit, then the purpose ("it lets us find the step at which two runs first differ"), then the weaker and more useful goal. This is the clearest thinking in the section.
- "Activation Statistics". The table of standard deviations (0.70 rising to 3.44) is followed by an explanation that predicts it: "the first linear layer keeps the variance, the ReLU halves the mean square, and the second linear layer keeps it again. The variance therefore grows by a factor of about 1.5 per block". The reader can check $0.7 \times 1.22^8 \approx 3.4$ against the output. This is the standard to hold the rest of the section to.
- "When a loss becomes NaN at step 40,000, the useful question is which layer produced the first non-finite value." A concrete scenario, and the hook answers it in ten lines.
- The closing paragraph says what the tools cannot do.

## Structural findings

1. **The motivating claim is never measured.** "They differ by a few tenths of a point" is stated in the opening, and "we report the mean and the spread over several runs rather than one run" in the middle. The only seeded experiment is five steps of regression on random data, which prints two losses (0.864 and 0.931) for two seeds. Train softmax regression or the MLP of chapter 4 on Fashion-MNIST for a few epochs under five seeds, print the five accuracies, and then run one seed twice. The first output answers the opening question with the reader's own numbers; the second shows what a seed buys. The cost is under a minute of CPU time.
2. **The hard cases are described, not shown.** (a) "Data Loader Workers" names two failures; the PyTorch cell runs with `num_workers=0` and its own tab block concedes that the worker initializer "repeats what the loader already does". (b) "Deterministic Arithmetic" says that two seeded GPU runs "can then differ in the last bits of their weights"; no cell shows two runs differing. The PyTorch cell shows instead that the determinism switch raises an error for `AvgPool3d`, an operation the reader has not met and will not use. In both cases exercise 3 or a guarded GPU cell could show the real thing. See REP-5 and REP-7.
3. **Two sections under one title.** The opening argues that "The two halves depend on each other". In the text they do not: the inspection half begins by discarding the seed (`torch.seed()  # a fresh random seed: the demos above left one pinned`) and never uses repeatability. The promised scenario, "repeat the run exactly up to that point and then look inside the model", is never carried out. Two repairs are possible. Either carry it out, as the chapter's capstone: a seeded run that diverges, restarted from a checkpoint of `sec_read_write`, with the NaN finder attached. That would use three sections of the chapter at once and give it the ending it lacks. Or move "Model Inspection with Hooks" next to `sec_init_param`, whose experiment it repeats ("The initialization experiments of :numref:`sec_init_param` measured the standard deviation of the activations at every depth"), and let this section be about reproducibility alone.
4. **The first half reads as a comparison of four frameworks.** Under "Seeds and Random Number Generators" nearly all the content is inside tab blocks, and the untabbed prose between them is connective ("Where the random functions get their state from is the first thing to know about a framework"). "Randomness as a Value" is an essay on the JAX design placed where three of four readers have just read their own framework's tab. The ideas are sound; the form is a feature matrix.
5. **The section, and so the chapter, stops.** After the gradient cell there is one closing paragraph for the section. Nothing closes the chapter or points to convolutional networks. The chapter introduction promises that "later chapters rely on it throughout"; one paragraph here should say what the reader now has (a `Trainer` that uses a GPU, checkpoints, mixed precision, hooks) and where each is next used.

## Passage-level findings

**REP-1** (major, pedagogy). Opening: ":numref:`chap_perceptrons`, they differ by a few tenths of a point, because".
See structural finding 1. A specific quantitative claim about the reader's own earlier experiments, with no evidence in this section or, as far as this reviewer can tell, in the chapters it cites.

**REP-2** (moderate, thought). Opening: "running models with *hooks*, functions that observe each layer of an".
See structural finding 3. The sentence that follows ("a measurement taken inside a model can be compared across two versions of the code only if the runs differ in nothing but that code") is true but is not used: no measurement in the section is compared across two versions of anything.

**REP-3** (minor, language). Under "Global Seeds": "Where the random functions get their state from is the first thing to know".
A 30-word sentence with a clause as its subject and the point at the end. Draft: "A framework's random functions either consult a global generator or take their state as an argument. Which of the two a framework does decides how a run is seeded."

**REP-4** (minor, pedagogy). Under "Global Seeds": "The two runs with seed 0 agree *bitwise*, down to the last bit of the loss".
The result is correct and expected; what the reader does not learn is how large the seed-to-seed difference is in a case that matters (see REP-1). Also note that the same sentence explains the agreement by "same operations in the same order", which is the subject of the next `##` section; a forward clause ("we return to the order of operations below") would connect them.

**REP-5** (major, pedagogy). Under "Data Loader Workers", PyTorch tab: "PyTorch's reproducibility notes. For `random` and NumPy it repeats what the".
The demonstration is a loader with zero workers and an initializer that, by the text's own account, changes nothing. Its output (the same shuffle order twice) shows that a seeded generator repeats, which the previous subsection already showed. Meanwhile the interesting failure, forked workers that "inherit the same generator state and apply the same 'random' augmentations", takes eight lines to show: a dataset whose `__getitem__` returns `np.random.rand()`, read with two workers. Either show it, or reduce the subsection to its first paragraph and leave the rest to exercise 3, which is well designed and covers exactly this.

**REP-6** (minor, language). Under "Data Loader Workers": "main process, as with the *fork* start method, Linux's default before".
"Linux's default before Python 3.14" is version trivia inside a sentence that already has three commas, and the line is visibly a late insertion (the source lines around it are ragged). Move it to a footnote or cut it.

**REP-7** (moderate, pedagogy). Under "Deterministic Arithmetic", PyTorch tab: "matrix multiplications raise as well. The backward pass of 3-D average".
See structural finding 2(b). The cell's output is `avg_pool3d_backward_cuda does not have a deterministic implementation`. It demonstrates the switch's error path on an operation chosen because it fails, and pooling is not introduced until the next chapter. A guarded GPU cell that runs a scatter-add over a million entries twice and prints whether the results are equal, then turns the switch on and repeats, would show the phenomenon the prose describes. The environment variable `CUBLAS_WORKSPACE_CONFIG=:4096:8` in the same tab is documentation.

**REP-8** (minor, explanation). Under "Deterministic Arithmetic": "arithmetic combines them. Floating-point addition is not associative".
The reference is to `sec_numerics`, which shows a sum drifting but never uses the word "associative" or states the property. The three-number example that follows is self-sufficient; drop the reference or add the sentence to `numerics.md`.

**REP-9** (minor, language). Under "Model Inspection with Hooks": "is the residual block of :numref:`sec_model_construction`, unchanged: it adds".
The sentence has two colons ("unchanged: it adds the output of a small network, its *body*, to its input: $\mathbf{h} = \ldots$") and restates a block that is then defined in code for the third time in this half of the chapter (`gpus-devices-memory.md` defines `ResidualBlock` twice, once per relevant tab group). If the block is the chapter's shared model, save it once in `d2l` and import it, as the writing guide asks for shared models (its section 7.5).

**REP-10** (minor, language). PyTorch code under "Model Inspection with Hooks": "torch.seed()  # a fresh random seed: the demos above left one pinned".
Housekeeping for the authors (so that the prose can quote a range instead of fixed numbers) shown to the reader as the first line of a cell. It also quietly undoes the first half of the section. If the cell must reseed, say why in the prose, or structure the earlier demos so that they do not leave global state behind, which would itself illustrate the "Generator Objects" subsection.

**REP-11** (minor, explanation). Under "Activation Statistics": "The standard deviation grows from block to block, from about 0.7 after the".
Two small additions would complete an already fine explanation. Say where 0.7 comes from (Xavier with 20 inputs and 64 outputs gives variance $20 \cdot 2/84 \approx 0.48$), and close the loop on the prediction: "Eight blocks give $0.7 \times 1.22^8 \approx 3.4$, which is what the table shows."

**REP-12** (moderate, thought). Under "A NaN Finder": "compute. To test this, we put a NaN into one weight of the fourth layer of".
"The fourth layer of the stack, the third residual block" is hard to parse (the fourth module of the `Sequential`, counting the input layer, is the third block). More substantively, a NaN planted in a weight is the easy case: the faulty layer is known. The scenario the subsection opened with (a loss that becomes NaN during training) is harder, because the first non-finite output usually appears downstream of the cause (an overflow in one layer, `inf - inf` in a later one, as `sec_numerics` explained). One sentence should say that the finder names the first *symptom* and that the cause is at or before it. Exercise 4 ("Divergence monitor") does the realistic version and is a good candidate for promotion into the body in place of the planted NaN.

**REP-13** (major, pedagogy). Under "Gradient Inspection", PyTorch tab: "Here are the norms of the gradients with respect to the output of each".
The cell prints `block output gradients: 1.26 1 0.834 0.666 0.502 0.395 0.297 0.236` and `weight 19.9, bias 3.52`, and the next sentence of the section is the closing paragraph ("We can now repeat a run bit for bit ..."). No sentence reads the output. There is something to read: the gradient with respect to a block's output grows by a factor of about 1.2 to 1.3 per block from the last block back to the first, the mirror image of the forward growth of $\sqrt{1.5}$ that the previous subsection explained, and for the same reason. Add: "The gradient norm grows from 0.24 at the last block to 1.26 at the first, by about a quarter per block: on the way back each block again adds the signal through its body to the signal that bypasses it." The weight and bias norms (19.9, 3.52) are given no purpose at all; either explain the ratio or leave them to exercise 5, which asks exactly that.

**REP-14** (moderate, language). Under "Gradient Inspection", PyTorch tab: "`module.register_full_backward_hook(f)` runs `f(module, grad_input,".
Ten sentences of signature description (`grad_input`, `grad_output`, tuples, return values, handles) before the reader knows what question the gradients will answer. The untabbed lead-in gives two possible purposes ("to log the size of each layer's gradient or to find where the gradients start to explode") and pursues neither. State one question first (does the backward signal grow with depth the way the forward signal did?), then give the two lines of API needed to answer it.

**REP-15** (minor, language; pattern). Tabs throughout, for example "MXNet has no generator objects, and no sampling function accepts one." and "TensorFlow has no backward hooks and needs none".
"Has no" occurs 14 times in this file, almost always opening a tab block that explains what a framework lacks and what to do instead. For the PyTorch reader these are invisible; for the others, the section is a list of workarounds for a PyTorch-shaped lesson (hooks most of all: the TensorFlow tab has to build a different model, the functional `taps` list, to take the same measurement). This is the clearest case in the chapter of prose that fits one framework.

## Exercises

Six for PyTorch and JAX, five for MXNet and TensorFlow, all [code]. Quality is high. Exercise 1 (block ablation, then "Rebuild `net` three more times ... State whether the conclusion 'block $k$ matters most' survives") teaches the section's real lesson about seeds better than the body does. Exercise 2 ("name the term this equation omits") is a neat check on the variance argument. Exercise 3 covers the data-loader failures that the body only describes. Exercises 4 and 5 are substantial and well specified. Two gaps. There is no exercise on the opening question; add: "Train the classifier of :numref:`sec_mlp-implementation` under ten seeds. Report the mean and standard deviation of the validation accuracy. How many runs per variant would you need before a difference of 0.3 points is convincing?" And nothing can be done on paper; a short problem on non-associativity (find three fp16 numbers whose sum depends on the grouping, using the spacings of `sec_numerics`) would fit. The exercise text is 2,200 words in the source because four of the five problems are repeated per tab with small changes; "Divergence monitor" is identical in three tabs and could be untabbed.

## Still open from October

October's one note on this section was a "keep": "the NaN at step 40,000". It is kept, and it is still the best opening sentence of a subsection here; see REP-12 for how to make the subsection live up to it.
