# Chapter 2, Linear Regression: writing review (November 2026)

Sources: `chapter_linear-regression/*.md`, read in full up to the slides marker, in the order of `_quarto.yml`. Stored PyTorch outputs and all figures were checked. Per-notebook reviews:

- [index](ch02-linear-regression/index.md)
- [linear-regression](ch02-linear-regression/linear-regression.md)
- [oo-design](ch02-linear-regression/oo-design.md)
- [synthetic-regression-data](ch02-linear-regression/synthetic-regression-data.md)
- [linear-regression-scratch](ch02-linear-regression/linear-regression-scratch.md)
- [linear-regression-concise](ch02-linear-regression/linear-regression-concise.md)
- [generalization](ch02-linear-regression/generalization.md)
- [weight-decay](ch02-linear-regression/weight-decay.md)

## Verdict

The chapter is good, and three of its seven sections are close to the bar the author set. It has a method that most textbooks lack and that deserves to be the book's signature: say what a correct result must look like, with a number, then run the code and compare. The corrupted label that pulls a slope from 2 to 23, the standard error $\sigma/\sqrt{n}$ as a test threshold, the loss that falls by a predicted factor of 7 per epoch to a predicted floor, bias and variance computed in eight lines, ridge returning $(1, 1)$ where least squares returns $(-0.84, 2.85)$: these are sharp examples, economically told, and they trust the reader. The sentences are plain and correct almost everywhere; I found no error of mathematics and no number in the prose that the stored outputs contradict.

What holds the chapter back is structure, not sentences. First, the software scaffolding arrives in the second section, before anything has been trained, runs nothing, and is written for chapters 3 to 13; the first training run comes roughly 9,000 words after the algorithm is derived. Second, the main experiment of the weight decay section does not show what the section claims. Third, the same few facts (the factor of 2 between the book's loss and the framework's, the standard error, host synchronization) are retold section after section. A reader with stamina finishes the chapter well taught. A reader without it may stop in section 2.

## Scores

| Notebook | Explanation | Thought | Pedagogy | Language | Enjoyment |
|---|---|---|---|---|---|
| index | 4 | 4 | 4 | 4 | 3 |
| linear-regression | 4 | 4 | 4 | 4 | 4 |
| oo-design | 3 | 3 | 2 | 4 | 2 |
| synthetic-regression-data | 4 | 3 | 4 | 4 | 3 |
| linear-regression-scratch | 4 | 4 | 4 | 4 | 4 |
| linear-regression-concise | 4 | 3 | 3 | 4 | 3 |
| generalization | 4 | 4 | 5 | 4 | 4 |
| weight-decay | 4 | 3 | 3 | 4 | 3 |

No 5 for language or enjoyment anywhere: the prose is clean and even, but it is the same voice at the same pitch for 26,000 words, with few sentences a reader would quote (the exceptions are listed under "What is best").

## Arc and ordering

The chapter has three movements.

1. **The idea** (section 1): model, loss, least squares, SGD, maximum likelihood. Strong, a little overloaded.
2. **Building and testing** (sections 2 to 5): base classes, synthetic data, scratch, concise. Four software sections in a row, about 11,700 words, between the idea and the question of whether a fitted model can be trusted.
3. **Trust** (sections 6 and 7): generalization and weight decay. The strongest and the most ambitious sections.

The order within the second movement is the chapter's main structural problem. The reader derives minibatch SGD in section 1 and then reads, in order: an abstract class design in which six methods raise `NotImplementedError`; a dataset class and two data loaders; and only then a model, a gradient check and a training run. The first loss curve appears in the fourth section. By then the reader has been asked to hold `Module`, `DataModule`, `Trainer`, `ProgressBoard`, `HyperParameters`, `add_to_class`, `as_metrics`, `get_tensorloader`, `prepare_batch` and `Averager` in mind without having seen any of them do anything.

**Proposed order.** (1) Linear Regression. (2) Synthetic Data, cut to its first half: generate the data, derive the threshold. (3) Implementation from Scratch, with a bare training loop of eight lines (see SCR-1), the prediction of the loss curve, the run, the check. (4) Object-Oriented Design, now a refactoring of code that the reader has just seen work: the loop becomes `Trainer`, the generator becomes `DataModule` (the loader material from the present section 3 moves here), the model becomes `Module`. Every method has a body, and the section ends by re-running section 3 through `trainer.fit` and getting the same numbers. (5) Concise. (6) Generalization. (7) Weight Decay. This costs no content, removes all stubs, and roughly halves the distance between the algorithm and the first training run.

## The chapter-specific question: does the software scaffolding help or get in the way, and is it introduced at the right moment?

**It helps later and it gets in the way here. It is introduced too early and with too much.**

Where it helps, visibly, inside this chapter:

- The learning-rate sweep in the scratch section is five lines because `Trainer` and `trainer.history` exist.
- `WeightDecayScratch` changes training by overriding one three-line method. This is the design's promise ("later chapters change only the part under study") kept on the page.
- The concise section can truthfully say that the trainer "runs unchanged".

Where it gets in the way:

- **Nothing runs in the section that introduces it** (OO-1). The only outputs are from the toy classes `A` and `B`. The chapter's own method, check every claim against behaviour, is suspended exactly where the reader most needs reassurance.
- **Chapter 2 pays for chapters 3 to 13** (OO-5, OO-7, OO-8, OO-10, OO-11). Dictionary metrics, `num_gpus`, gradient clipping, `evaluate`, schedules, asynchronous devices and parameter saving are all explained or hooked here and none is used. The two sections that build the trainer defer to "later chapters" seven times.
- **"From scratch" is not from scratch** (SCR-1). The training loop shown is the production loop; the four operations of SGD are four of its fourteen lines, and one of the others calls a method that does not exist yet.
- **`add_to_class` scatters every class.** It is a reasonable device for a notebook, and the text is honest about it, but the reader never sees `Trainer` or `LinearRegressionScratch` whole. A collapsed "the complete class" listing at the end of each defining section would help.
- **The framework tabs multiply the cost.** The JAX tabs in `oo-design.md` and the scratch section carry paragraphs on optimizer caches and compilation keys. The source word counts are inflated by a quarter to a third by per-framework prose; a PyTorch reader is spared most of it, a JAX reader is not.

The decisive observation is the generalization section. It uses none of the scaffolding (NumPy and `d2l.plot` only) and it is the clearest, best-paced section of the chapter. The ideas of this chapter do not need the framework; the later chapters' experiments do. That argues for introducing it as late as the chapter allows, as a refactoring, and for giving it only the features this chapter uses.

## The most important findings

1. **The scaffolding section runs nothing and is written for later chapters.** OO-1, OO-5, OO-8, OO-10; consequences in SCR-1, SCR-2, SCR-4. Fix by the reordering above and by cutting `oo-design.md` to what this chapter uses.
2. **The weight decay experiment does not support the section.** WD-1. By the text's own analysis the penalty helps only by erasing the random initialization and a model started from zero does as well; in addition the regularized model (stored validation loss 0.0104) is indistinguishable from predicting the constant $b^*$ for every input (expected loss 0.01005). Build the section on its duplicated-features example instead.
3. **The first training loop is the production loop.** SCR-1. Show the bare loop first.
4. **One fact told seven times.** CON-1, WD-7, with its origin in `linear-regression.md` under "Minibatch Stochastic Gradient Descent". The factor $\frac{1}{2}$ in the book's loss and its absence in the frameworks' produce a learning rate of 0.015 against 0.03, a weight decay of 6 against 3, and validation losses that "correspond" to each other. The lesson (a learning rate has meaning only with the scale of the loss) is worth teaching once. Remove the mismatch at the source: let the concise and weight decay models return half of the framework loss, or drop the $\frac{1}{2}$ from the scratch loss and accept a 2 in the gradient.
5. **Section 1 is overloaded in the middle.** LR-7 (the noise ball argued by three unsupported proportionalities), LR-9 and the misplaced vectorization subsection, LR-5 (the silent change of meaning of $\mathbf{X}$ and $\mathbf{w}$).
6. **Synthetic Data holds two sections.** The test-with-a-threshold idea and the data-loader mechanics (structural finding in the synthetic data review; SYN-3, SYN-4).
7. **Validation and test are used before they are distinguished.** GEN-2.
8. **Prose understates an output by ten orders of magnitude.** GEN-4: the text says "more than a thousand times larger", the page shows a test error of $2.3 \times 10^{14}$.
9. **The principled value of $\lambda$ is derived and then ignored.** WD-2: the formula gives 0.05, the experiment uses 3, and only the exercises say so.
10. **The chapter opening is told three times.** IDX-1: part introduction, chapter introduction and section 1 each give Legendre and Gauss and the list "model, loss, algorithm".

## Recurring patterns

**Repetition across sections** (the chapter's most frequent defect):

- Factor of 2 between the losses: one general statement in section 1, six in the concise section, one more conversion in weight decay.
- The standard error $\sigma/\sqrt{n} \approx 3 \times 10^{-4}$: derived in Synthetic Data, restated with the arithmetic in the scratch section, again in the concise section, again in two summaries.
- The bias remainder $4.2 \times 0.97^{320} \approx 2.5 \times 10^{-4}$: scratch and concise, same words (CON-2).
- Host synchronization on an accelerator: twice in `oo-design.md`, once in the scratch section, once more as an exercise (OO-8, SCR-4).
- "The loss first, the other metrics in alphabetical order": `record` in `oo-design.md`, `Averager` in the scratch section (OO-9).
- The noise ball: defined in section 1, invoked in Synthetic Data, the scratch section, and recalled in `oo-design.md`.

**Symbols with two meanings:**

- $\mathbf{X}$ and $\mathbf{w}$ with and without the bias column, switched silently (LR-5), and $\mathbf{w}^*$ with two and three entries (SYN-1).
- $L$ for the full-data loss and the minibatch loss (SCR-3).
- $d$ for the number of features and for the polynomial degree (GEN-10); $p$ and $q$ appear for related counts.
- `sigma` (initialization) and $\sigma$ (noise), both 0.01 (SCR-7); $\mu$ for a mean and for eigenvalues (WD-9).

**Material written for the authors** (writing guide, section 4.5): "the book build skips the intermediate redraws" (OO-6); "The trainer of this chapter does not call `weight_init`; the trainer of ... does" (CON-3); the note on shared references in the resource list (IDX-3); the raw-versus-centered caveat (WD-6); the optimizer-cache paragraph in the JAX tab (OO-10).

**Statistics of repeated runs stated in prose.** "in fewer than one run in a hundred", "in about three runs out of four", "In about four runs out of five", "between about 0.015 and 0.025 from run to run", "0.11 to 0.15 from run to run". The intention is right (prose that survives re-execution, as `CLAUDE.md` asks), and in the generalization section it is done gracefully. In the scratch and weight decay sections it tips into reporting distributions the reader cannot check from one run (SCR-6, WD-8). Rule of thumb for the reviser: give a range when the reader will look at that number; do not give frequencies of events.

**Paragraphs that are ledgers.** Chains of six or more sentences each carrying one or two numbers: the convergence-rate paragraph (SCR-5), the $\mathbf{X}^\top\mathbf{X}$ paragraph (SYN-2), the last two paragraphs of the spectral analysis (WD-8). The fix is the same each time: let a printed table carry the numbers and let the prose state the rule.

**Roadmap by template.** "Section N asks ..." seven times in the chapter introduction (IDX-2). The form follows the writing guide (section 12.1) and I think the guide pushes too hard here: applied to every chapter it becomes a tic. Section openings also preview their content in a fixed "We first ... then ... Finally" paragraph; this is useful, but with the chapter roadmap, the section roadmap, the closing paragraph and the summary, each section is outlined four times.

**Dependence on other chapters.** Light and well handled. Section 1 leans on the calculus section of chapter 1 six times (the three-point line model, gradient descent), each time restating what it uses. Appendix pointers are for depth only, with one exception: the equivalence of the penalized and constrained forms (WD, under the $\ell_1$ penalty). Forward references are mostly promises, as the guide allows. Reference density is high (about 30 `:numref:` per section in three sections) and gives the rendered page a busy look.

**Language.** Very few of the mannerisms the October guide lists survive: no theatrical verbs, no slogans, almost no dashes. What remains is evenness: sentences of similar length and build, "therefore" and "so that" as the standard joints, hedged quantities ("about" 25 times in the weight decay section). It reads as careful rather than alive. The moments of voice are worth protecting (see below).

## What to cut, and what is missing

**Cut or move:**

- From `oo-design.md`: "Optimizers and Schedules", "Evaluation", the second reason under "Epoch Means", dictionary metrics and `as_metrics`, the Lightning comparison, the stub of `HyperParameters`. (To the sections that first use them.)
- From `linear-regression.md`: the `wait` helper and half of the vectorization subsection; the with/without replacement parenthesis; the invertibility proof; the heteroscedastic aside; either the noise-ball argument or its vagueness.
- From `synthetic-regression-data.md`: the loader half, to the `DataModule` section; exercise 7.
- From `linear-regression-scratch.md`: `prepare_batch`, `Averager` and the full `fit_epoch` listing, to the `Trainer` section.
- From `linear-regression-concise.md`: five of the six tellings of the factor of 2; the repeated error analysis.
- From `generalization.md`: the $R^2$ paragraph (or use it), the fixed-design caveat.
- From `weight-decay.md`: the 200-feature experiment as the main demonstration; the monomial formula; the $\ell_1$ geometry could shrink to a paragraph and an exercise.

A rough count suggests the chapter could lose 4,000 to 5,000 of its 26,000 words with no loss of a teachable point.

**Missing:**

- **A house.** The chapter opens twice with a house to be valued and never values one. Every experiment is synthetic. The synthetic data are the right tool for testing code, and the text says so well, but a reader who came for prediction never sees the method meet real measurements; real house prices arrive two chapters later. At the least, section 1 or the concise section should say where the house is (the house-price section of the multilayer perceptron chapter). Better, a short capstone at the end of the concise or weight decay section could fit a handful of real features and read off the price per square meter, closing the loop the first paragraph opened.
- **A bare training loop that runs** (SCR-1).
- **A baseline in the weight decay experiment** (WD-1): the loss of predicting the mean label.
- **A numeric instance of likelihood** (LR-10).
- **Feature scaling.** It is mentioned twice in passing (the learning-rate limit $2/s^2$; an exercise) and matters for everything that follows. One short paragraph with a number in the scratch section's learning-rate subsection would do.

## What is best, and should set the model for the rest

- **Predict, then measure.** `linear-regression-scratch.md`, "Checking the Implementation": the expected loss gives a floor and a rate; the run shows both; the text then explains what the loss cannot reveal. `synthetic-regression-data.md`, "Recovering the Parameters": "A test needs a threshold." If every experimental section of the book were built this way, the book would have a character of its own.
- **One example carried through.** The corrupted label in `linear-regression.md` (loss, likelihood, loss menu, and again in the scratch exercises); the degree-19 polynomial in `generalization.md` (overfitting, variance, capacity, cross-validation, and the "weights of a million" that open the next section); the near-duplicate features in `weight-decay.md`.
- **Small cases with every digit shown.** The three-point normal equations and the perpendicularity check; the 480,000 dollar house.
- **Limits shown, not mentioned.** The generator that yields nothing on its second pass; the lookup-table model with zero training error; "This reading relies on the model being linear".
- **Sentences with a voice.** "'normal' is an old word for perpendicular." "An error of $0.1$ cannot be blamed on the noise." "That was a demonstration, not a method." "Underfitting is error from bias, and overfitting is error from variance." "The model still trains, only less well."
- **Exercises that continue the section's method.** The scratch and generalization sets in particular: predict, derive, then check numerically. These are much better than the usual "try different hyperparameters".

## Status of the October findings

Resolved: AdamW before Adam; double descent; the triple statement in `generalization.md`; the unexplained difference in learning rates (now over-explained, CON-1). Partly open: `oo-design.md` still interrupts the chapter and still explains asynchronous devices (OO-8) and installs the schedule hook (OO-10). Not followed, rightly: the effective number of parameters is still in the weight decay section and is now self-contained.

## Could not check

- TensorFlow, JAX and MXNet outputs (only the PyTorch store was compared with the prose). The prose is written to hold in every tab, and I saw no tab-specific claim in the shared text, but I did not verify the other stores.
- Frequencies claimed for repeated runs ("three runs out of four", "four runs out of five", "fewer than one run in a hundred"). I checked them by calculation where possible (they are plausible) but did not rerun notebooks.
- The rendered HTML pages; I read the sources and rendered the figures and stored plots separately.
