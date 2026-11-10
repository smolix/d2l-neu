# Synthetic Data

Source: `chapter_linear-regression/synthetic-regression-data.md`, about 2,600 words of prose before the exercises (all tabs; about 2,000 for the PyTorch reader).

## Verdict

The first half of this section contains one of the best ideas in the chapter: a test needs a threshold, and the threshold can be computed. From the normal equations the text derives that even an exact fit misses the true parameters by $\sigma/\sqrt{n} \approx 3 \times 10^{-4}$, checks it with `lstsq`, and hands the number to the next section. Few textbooks teach how to test learning code at all. The second half is about data loaders, and although it has a nice moment (a generator that silently yields nothing on its second pass), it is a different subject, mostly plumbing, and heavy with per-framework text. The most important fix is to separate the two: let this section be about the test, and move the loader machinery next to `DataModule`.

Stored outputs agree with the prose (noise of the first example $-0.0037$; estimation errors $-1.6$, $-4.0$ and $-5.8 \times 10^{-4}$ against a standard error of $3.2 \times 10^{-4}$; 32 minibatches, the last of 8; 0 on the second pass).

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|---|---|---|---|---|
| 4 | 3 | 4 | 4 | 3 |

## What works

- **The opening argument.** "Three explanations compete: the model may be misspecified, the optimization algorithm or its implementation may be flawed, or the data may be pathological." A real problem, stated before the remedy.
- **"A test needs a threshold."** The five-word sentence that opens "Recovering the Parameters", followed by a derivation that ends in a number and a cell that confirms it. This is the predict-then-measure habit that makes the next section work.
- **"An error of $0.1$ cannot be blamed on the noise. It points to a bug or to an optimization that has not converged; once the loss has converged, to a bug."** Dry, exact, memorable.
- **The silent failure.** The second pass over the generator prints `second pass: 0 minibatches`, and the text draws the consequence: the trainer "would train for one epoch and then do nothing, without an error". A limit shown, in the manner of MacKay, before the tool that removes it.
- **The closing** states what the test does not show ("It does not tell us how the method behaves when the model does not match the data, which is the usual case with real data").
- **Exercise 2** (planted bug: features and labels shuffled independently) asks the reader to predict what a broken loader does to the fit. Excellent.

## Structural findings

- **Two ideas in one section.** "Generating the Data" and "Recovering the Parameters" teach how to build a test with a known answer. "Reading the Data in Minibatches" and "Reusing a Loader Across Epochs" teach how a data loader works. The second pair is about 45% of the prose and nearly all of the per-framework text (an `ArrayLoader` class in two tabs, `shuffle` buffers in TensorFlow, keys in JAX). Fix: move the loader half to the section that defines `DataModule`, where `get_tensorloader` belongs in any case, and keep here only the sentence that the data are served in shuffled minibatches of 32.
- **The summary reports trivia.** See SYN-6.

## Passage-level findings

**SYN-1** (moderate, explanation). Under "Recovering the Parameters".
Quote: "$\mathbf{w}^* = (2, -3.4, 4.2)$ now has three entries and"
Within one section $\mathbf{w}^*$ means $(2, -3.4)$, then $(2, -3.4, 4.2)$, and in the next section $(2, -3.4)$ again; $\mathbf{X}$ changes with it. Fix: use a separate symbol for the extended vector and matrix, here and in `linear-regression.md` (see LR-5), for example $\tilde{\mathbf{w}}^*$ and $\tilde{\mathbf{X}}$, which the exercises of the scratch section already use for inputs.

**SYN-2** (moderate, explanation). Under "Recovering the Parameters".
Quote: "$z^2$, with mean $1$ and variance $\mathbb{E}[z^4] - 1 = 3 - 1 = 2$ for a standard"
To justify $\mathbf{X}^\top\mathbf{X} \approx n\mathbf{I}$ the paragraph brings in the fourth moment of the normal distribution and reports each entry to two digits ($1{,}000 \pm 45$, $0 \pm 32$). The argument needs only that an average of 1,000 products is close to its mean. The detail slows the one derivation in the section that matters. Fix: two sentences ("Each entry of $\mathbf{X}^\top\mathbf{X}/n$ is an average over 1,000 examples of a product of two columns; these averages are close to 1 on the diagonal and to 0 off it, to within a few percent") and, if a check is wanted, print $\mathbf{X}^\top\mathbf{X}/n$ rounded to two decimals in the cell that follows. The code would then carry the claim.

**SYN-3** (moderate, thought). Under "Recovering the Parameters".
Quote: "about $10^{-3}$ of the true parameters, and essentially always within"
The paragraph gives four numbers for one threshold ($3 \times 10^{-4}$, $10^{-3}$, "fewer than one run in a hundred", $2 \times 10^{-3}$) and supports the step from least squares to SGD with an unquantified "by a similar amount". What the reader needs is an order of magnitude and a contrast. Fix: "Least squares misses each parameter by about $3 \times 10^{-4}$, and minibatch SGD fluctuates around the least-squares solution by a similar amount, as the next section measures. We therefore expect errors below about $10^{-3}$. An error of $0.1$ cannot be blamed on the noise."

**SYN-4** (moderate, explanation). Under "Reading the Data in Minibatches" and "Reusing a Loader Across Epochs".
Quote: "error. Our loader has further limits. It cannot report how many minibatches"
The text lists the single-pass failure, then three "further limits", and the next subsection begins "A reusable data loader removes the first two limits" and later "the other two". The reader has to count back across a heading to find which are meant. Fix: number them in one sentence (one pass only; no length; all data in memory; batches assembled in the training process) and refer to them by content ("the framework loaders also handle data that do not fit in memory").

**SYN-5** (minor, pedagogy). MXNet tab under "Reusing a Loader Across Epochs".
Quote: ":numref:`sec_rnn-scratch`, a pass over the data becomes about 50 times"
A speed-up measured six chapters ahead is reported before the reader has met the experiment (writing guide, section 6.3). Not on the default tab, but it shows how much maintainer knowledge the tabs carry. Cut the number.

**SYN-6** (minor, language). Summary.
Quote: "them into minibatches, 32 per epoch with the last holding 8 examples. A"
The summary restates incidental numbers and both loader implementations. It should answer the four questions of the guide. Draft: "Real data cannot tell a bug from a wrong model; data we generate ourselves can. For labels drawn from a linear function with Gaussian noise, least squares misses each true parameter by about $\sigma/\sqrt{n}$, here $3 \times 10^{-4}$, and that sets the precision a correct implementation should reach. The test says nothing about data that a linear model does not fit."

**SYN-7** (minor, language). Under "Reusing a Loader Across Epochs".
Quote: "dataset can use it. It takes a tuple of tensors whose first axis indexes the"
A sentence that reads the function signature aloud before the reader sees the three-line function. Show the cell, then say what it is for.

## Exercises

Seven problems, three of them written four times over for the four frameworks. Exercise 2 (planted bug) and exercise 5 (measure the standard error against $\sigma/\sqrt{n}$, find where the derivation fails, then correlated features) are the two that test the section's idea, and both are very good; 5(c) quietly prepares the weight decay section. Exercise 6 (on-the-fly generation) is a sound design problem. Exercise 7(c) asks about pseudorandom permutations and Feistel networks; it is a curiosity far from the chapter and should be cut, and 7(a) and 7(b) with it unless the loader material stays in this section. Exercise 1 (seeding) and exercise 3 (`drop_last`) are framework-specific and fine but minor.

## Still open from October

No item in the October review concerned this file.
