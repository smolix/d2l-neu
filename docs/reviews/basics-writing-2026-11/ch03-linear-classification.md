# Chapter 3, Linear Classification: writing review

Sources: `chapter_linear-classification/*.md`, eight files, read in full up to the slides marker, in the order of `_quarto.yml`. Body prose about 21,000 words (framework tabs included), exercises about 5,900 words. Stored PyTorch outputs and all generated figures were checked against the prose.

Per-notebook reviews:

- [index](ch03-linear-classification/index.md)
- [softmax-regression](ch03-linear-classification/softmax-regression.md)
- [image-classification-dataset](ch03-linear-classification/image-classification-dataset.md)
- [classification](ch03-linear-classification/classification.md)
- [softmax-regression-scratch](ch03-linear-classification/softmax-regression-scratch.md)
- [softmax-regression-concise](ch03-linear-classification/softmax-regression-concise.md)
- [generalization-classification](ch03-linear-classification/generalization-classification.md)
- [environment-and-distribution-shift](ch03-linear-classification/environment-and-distribution-shift.md)

## Verdict

This is a good chapter, and a much better one than the October review describes. The tangents are gone, the notation is consistent within each section, openings start from the reader, every displayed equation is read in a plain sentence, and nearly every abstraction is followed by a number the reader can check. I verified several dozen of those numbers and found two small mismatches with stored outputs and no errors of arithmetic. Three stretches reach the bar set by the model texts: the derivation of softmax and cross-entropy on one small example in `softmax-regression.md`, the screening test in `classification.md`, and the first two parts of `generalization-classification.md` (how precise a test set is, and what reusing it does).

It is not yet an excellent chapter, for three reasons. First, it repeats itself: the numerical-stability story is told across three sections, the calibration and temperature story across three, the step-function argument three times, the column-normalized confusion matrix three times. The cause is partly the writing guide's own rule to restate facts from earlier sections, applied as "explain again". Second, the middle of the chapter is in the wrong order: the reader meets precision, recall, cost thresholds, calibration and reliability diagrams on invented numbers before any classifier has been trained, and then meets them again on real numbers. Third, the prose is clean but even. The revision removed the mannerisms and with them most of the voice; outside the examples, the sentences are correct, short and interchangeable. The chapter is easy to read and seldom a pleasure, except where an example carries it.

## Scores

Scale 1 to 5 per the brief (5 = a page of Cover and Thomas or MacKay; 4 = good textbook prose with local flaws; 3 = competent, or good with a structural problem).

| Notebook | Explanation | Thought | Pedagogy | Language | Enjoyment |
|:--|:-:|:-:|:-:|:-:|:-:|
| index | 4 | 4 | 4 | 4 | 4 |
| softmax-regression | 4 | 4 | 4 | 4 | 4 |
| image-classification-dataset | 4 | 4 | 3 | 4 | 3 |
| classification | 4 | 3 | 3 | 4 | 3 |
| softmax-regression-scratch | 4 | 4 | 4 | 4 | 4 |
| softmax-regression-concise | 4 | 3 | 3 | 4 | 3 |
| generalization-classification | 4 | 4 | 4 | 4 | 5 |
| environment-and-distribution-shift | 4 | 4 | 3 | 4 | 4 |

## Does the chapter hold together?

The assignment asks whether a chapter that mixes a model, software, learning theory and an essay on distribution shift is one chapter, and whether the theory and shift sections are pitched right for a reader at this point.

**It holds together, and better than its table of contents suggests.** The chapter introduction asks a double question: "How should a model predict a category, and when can we trust its prediction?" The first five sections answer the first half. The last two answer the second half, and they are written as its continuation: `generalization-classification.md` opens with the 83% the reader has just measured and asks what it is worth; `environment-and-distribution-shift.md` opens with the condition under which the previous section's answer holds and asks what happens when it fails. The shift section then uses this chapter's own tools (logistic regression as a domain classifier, the gradient "prediction minus label", the confusion matrix for label shift). A reader who finishes the chapter has one story: build a classifier, measure it, ask how much the measurement means on new data from the same source, ask what happens when the source changes.

**The weak joint is in the middle, not at the end.** The second half of the chapter's question has three natural steps: what to measure (beyond accuracy), how precise a measurement is (generalization), and when it stops applying (shift). The first step is currently split over three places: `softmax-regression.md` ("Accuracy and Confidence"), `classification.md` ("Beyond Accuracy", before any training), and the ends of the scratch and concise sections (confusion matrix, reliability diagram). Consolidating it after the concise implementation would give the chapter two clean halves. Proposed order:

1. Softmax Regression (model and loss; ends after "The Error Rate")
2. Image Classification Data
3. The `Classifier` class, as a short section or as the first subsection of the scratch section (loss and accuracy, 600 to 800 words)
4. Softmax Regression from Scratch (with the weight templates; confusion matrix may stay)
5. Concise Implementation (the stable loss, shown working)
6. Measuring Classifiers (class imbalance with the screening test, costs and thresholds, calibration with the real reliability diagram and temperature scaling applied once)
7. Generalization in Classification
8. Environment and Distribution Shift

This removes about 2,000 words of repetition, lets the first training run arrive after about 6,500 words where it now takes 8,600, and makes sections 6 to 8 a visible unit that the introduction can announce.

**The theory section is pitched right in its first two parts and one step high in its third.** "The Test Set" and "Test Set Reuse" need only a biased coin, the standard error (defined in the preliminaries) and the union bound; they answer questions a practitioner has and they are a pleasure to read. "Statistical Learning Theory" is compact and honest, and its bridge from the union bound to uniform convergence is elegant, but it introduces seven notions in 1,300 words and reaches a negative practical conclusion. It should stay (a book of this ambition should say why fitting differs from memorizing, and the memorizer thread does that well), be marked as skippable on first reading, lose the general-dimension proof sketch (GEN-4), state that it treats binary classifiers (GEN-1), and be tied to the observed train-test gap of the chapter's model (GEN-2).

**The shift section is pitched right.** It is no longer an essay: one idea, a table, two derivations of the same form, numbers before formulas. Its level (densities, Bayes' theorem, an integral identity) matches the generalization section. Its weakness is in the experiments, not the pitch (ENV-2, ENV-3). Length is proportionate: theory and shift together are about 8,000 of 21,000 body words, which is defensible for the part of the chapter that answers half of its question. If something must be shortened, it is the middle (software and repeated numerics), not the end.

## Most important findings

1. **Reorder the middle: measure after training** (CLS-1, CLS-9, with SMR-8 and CON-5). `classification.md` teaches a base class and evaluation methodology in one section, before the first training run, on invented numbers. Split it; move "Beyond Accuracy" after the concise implementation, where it can use the Fashion-MNIST confusion matrix and the real reliability table.

2. **Give the calibration and temperature thread one home** (SMR-8, CLS-8, CLS-9, CON-5). Calibration is defined three times, temperature scaling explained three times as a cure for overconfidence, and the one experiment finds the model slightly underconfident and applies no scaling. The paragraph that starts the thread (SMR-8) is the least readable in the chapter: it packs separability, weight growth, max-margin bias, implicit regularization, calibration and deep-network overconfidence into nine sentences, and changes the meaning of "margin" on the way.

3. **Tell the numerical-stability story once** (SMR-2, SCR-1, SCR-2, CON-1, CON-2, CON-3). The overflow threshold of 88 is stated at least five times across three sections; shift invariance is derived three times; the log-sum-exp form of the loss is derived twice; a floor is added to the scratch loss, explained, and disowned. The framework loss is never run on the logits that broke the scratch version. Let the scratch section show the failure in one cell and the concise section own the argument and show the repair.

4. **Show what the model learned** (SCR structural finding, SCR-6). The chapter never displays the weight matrix as ten templates; it is exercise 5 of the scratch section. This is the picture that explains why the model works on trousers and fails on shirts, and it prepares the argument for hidden layers in the next chapter. The confusion heatmap should carry class names.

5. **Connect the theory to the trained model** (GEN-1, GEN-2, SCR-4). The VC bound is stated for binary classifiers and applied to ten-class Fashion-MNIST without comment. Training accuracy is never reported anywhere in the chapter, so the reader cannot set the observed train-test gap (a point or two) beside the bound ($0.11c$). The scratch section explains the small gap by counting parameters, which the theory section then says is not a rule.

6. **Make each experiment say one thing, on fixed data** (ENV-2, ENV-3, GEN-3, CLS-9). The shift experiments are unseeded and the prose reports ranges over runs the reader cannot see. The clipped arm of the covariate shift experiment changes nothing. The first generalization simulation tests the normal approximation where it is known to work. The reliability table in `classification.md` is invented.

7. **Fix the notation for the predicted class** (CLS-4, CON-7). $\hat{y}$ is a probability vector in the first section and a class index in `classification.md`, `softmax-regression-scratch.md` and `environment-and-distribution-shift.md`. $\bar{o}$ is the mean logit in one place and the largest in another. The confusion matrix orientation (predictions in rows) is the opposite of the reader's likely tools and is defended by warnings, not by a reason (CLS-5).

8. **Justify the exponential where it is introduced** (SMR-1) and be candid that the error bound is loose (SMR-9). Two small changes in the best section that remove the two places where an attentive reader feels handled.

9. **Use the chapter's own classifier for label shift** (ENV structural finding). The matrix $\mathbf{C}$ computed in the scratch section, applied to a resampled validation set, would tie the last section to the first five in six lines of code.

## Recurring patterns

**Repetition across sections** (the chapter's main defect; counts are tellings in body prose):

| Fact | Tellings | Where |
|:--|:-:|:--|
| $\exp$ overflows above about 88 in single precision | 5 or more | softmax-regression; scratch (opening, "Overflow", summary); concise (opening, "Overflow and Underflow", summary) |
| Subtracting the largest logit leaves the softmax unchanged | 3, twice derived | softmax-regression; scratch; concise |
| Accuracy is a step function with zero gradient; loss over $\log 2$ bounds the errors | 3 | softmax-regression; classification (opening and "Loss and Accuracy") |
| 0.51 to 0.99 lowers the loss from 0.67 to 0.01 | 2 | softmax-regression; classification |
| Definition of calibration | 3 | softmax-regression; classification; concise |
| Temperature scaling | 3 | softmax-regression; classification; concise |
| Column-normalized confusion matrix $\mathbf{C}$ | 3, near verbatim | classification; scratch; shift |
| Fashion-MNIST has no validation split | 2 | dataset; scratch |
| Size of the last minibatch | 3 | dataset; classification; scratch |
| The ten class names | 3 | softmax-regression; dataset; scratch |

The writing guide (section 7.2) says to restate a fact that is more than one section back. Here that rule has produced re-explanation. The guide should say that a restatement is one clause with a reference, and that a fact needed by three sections should be explained in the one that uses it most.

**Formulaic section frames.** Five of seven sections close with "We can now ..." or "We have written ..." followed by "Two questions remain" or "we still need". Seven roadmap sentences in the introduction use "X asks". Each instance is fine; the series is audible. Vary the closings: some sections can end on their last result.

**"therefore".** 22 uses in body prose (five each in `softmax-regression.md` and `softmax-regression-concise.md`). It often marks a step that the sentence order already makes clear.

**"at the price of".** Three uses (dataset, scratch summary, shift), each an instance of the economic metaphor the guide asks to avoid.

**Narration of variability.** "Repeated runs stay between", "in repeated runs", "In every run", "between runs", "across runs": six instances, three in the shift section. Appropriate once per experiment at most; elsewhere fix the seed.

**Evenness of voice.** Sentences are mostly 12 to 25 words, subject first, one clause. This is what the guide asks for and it is clear. Over 20,000 words it becomes monotone, and the passages a reader will remember are those where a specific image breaks the pattern: "not 5.5 of anything", the 81-times penalty, the useless classifier with 99% accuracy, "84.37% has more digits than such a test set supports", the vote that "never saw a label", the bank and the Oxfords, the healthy students. A reviser should protect every one of these and look for two or three more in the software sections, which have none.

**Framework tabs.** The prose mostly holds for all tabs. The exceptions are in `image-classification-dataset.md` (TensorFlow and JAX readers get a dense pipeline paragraph with tracing and compilation, ICD-2), `classification.md` (the MXNet paragraph on `collect_params`) and `softmax-regression-concise.md` (the PyTorch tab needs a lazy layer and a `weight_init` hook, CON-6). The three theory-side notebooks use NumPy only and are identical in all tabs, which is the right choice.

**Boilerplate imports.** Every notebook opens with the same seven-line import cell (`math`, `numpy`, `torch`, `nn`, `F`, `d2l`), mostly unused, and the shift section places it mid-section. A reader notices by the third notebook.

**Dependence on other chapters.** Light and well handled. Backward: the `Module`, `Trainer` and `DataModule` classes and the risk and validation discipline of chapter 2; Bayes' theorem (the HIV test), the standard error and broadcasting from chapter 1. Each is restated in a clause. Appendix: seven pointers, all for depth, none carrying a step. Forward: `chap_perceptrons` (twice), `chap_cnn`, `sec_generalization_deep`, `chap_reinforcement_learning`, plus two that should go, `subsec_implicit_regularization` (SMR-8) and `sec_training_recipes` (SMR-12).

**Stored outputs versus prose.** All checked claims hold except: validation loss "near 0.49" against a stored 0.475 (SCR-5); "the few dozen effective examples" against a stored effective sample size of 13, and a weighted-boundary slope "between about 1 and 2" against a stored slope just under 1 (ENV-2, ENV-3).

## What to cut

- "Calibration and Temperature" in `softmax-regression.md`, except two sentences on separable data (about 450 words), moved to the consolidated measuring section.
- The retelling of the step argument in `classification.md` (about 150 words) and "The Default Optimizer" there (about 150 words with the tab).
- The invented reliability table and the third temperature-scaling paragraph in `classification.md` (about 250 words).
- The floor in the scratch loss and its explanation; the axis-sum demonstration cell (about 200 words).
- The second opening paragraph, the re-derivations and the log-sum-exp plot in `softmax-regression-concise.md` (about 450 words).
- The last-minibatch paragraph and the pipeline prose in the TensorFlow and JAX tabs of the dataset section (about 250 words).
- The upper-bound sketch for the VC dimension in $\mathbb{R}^d$; the generative-classifier parenthesis and the clipped arm of the shift experiment (about 300 words).
- Exercises: the extended heatmap (dataset), the five-format table (concise, reduce), one or two of the five VC problems, the open-ended WILDS problem.

## What is missing

- The learned weights shown as images (scratch).
- Training accuracy of the chapter's model, anywhere.
- The stable loss run on the logits that broke the direct one (concise).
- A sentence on why the exponential (softmax-regression).
- A look at the data that shows the hard classes side by side (dataset).
- The binary restriction of the VC results, and the observed gap beside the bound (generalization).
- A label shift experiment on Fashion-MNIST (shift).
- An easy first exercise in `softmax-regression.md`, `classification.md` and `generalization-classification.md`. The exercises are the most consistently strong element of the chapter (nearly all teach something, many are memorable: Bradley-Terry with a "none" option, the double softmax, leaking a test set, features missing at serving time), but almost every set starts at medium difficulty.

## What is best and should set the model

- **One example carried through.** `softmax-regression.md` takes the logits (1.0, 2.2, 0.3) through the softmax, the log-odds, the loss, the gradient, the scratch implementation's output and the temperature figure. Every later chapter that introduces a model should have such a thread.
- **A number that overturns an intuition.** The always-healthy classifier at 99% against the useful test at 95%; the 2% patient who should be flagged; random guessers reaching 70% by voting. Each is set up with human-scale counts and stated in a flat sentence.
- **The memorizer as a standing counterexample** in `generalization-classification.md`: introduced in the opening, used three times to delimit claims.
- **Impossibility before method.** The label-swap argument in the shift section shows that nothing can be done without an assumption, then organizes the methods by assumption.
- **Honest limits.** "The guarantee is loose", "This is a property of the trained model, not of the softmax", "no weighting of the students' samples can stand in for them". The chapter says what its results do not show, once, without a stack of hedges.
- **Closings that hand over a question.** Every section ends by naming what is still open and which section takes it up. The device is right; only its wording is too uniform.

## October findings

All four October findings for this chapter are resolved or nearly so: the softmax tangents are cut (one sentence each remains for the exponential family and the appendix's three derivations, SMR-7 and SMR-11); label encoding and the layout of $\mathbf{W}$ are explained; the shift section uses $p_S$, $p_T$ and $z \in \{0,1\}$ throughout; the VC dimension is defined before the bound. Radon's theorem survives unnamed in one sentence (GEN-4). The "keep" items are kept. October did not look at the dataset, scratch or concise sections, or at repetition across sections, which is now the chapter's main problem.

## Not checked

- Outputs of the TensorFlow, JAX and MXNet tabs (only the PyTorch store was read).
- The rendered HTML pages; figures and output plots were rendered from the SVG sources.
- Claims about variability over repeated runs (nothing was re-executed).
- The appendix sections that the chapter points to.
