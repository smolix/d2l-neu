# Environment and Distribution Shift

Source: `chapter_linear-classification/environment-and-distribution-shift.md`. About 4,350 words of body prose (near the 4,500 ceiling), 970 words of exercises, seven figures, six computed cells (NumPy only, identical in all tabs).

## Verdict

This is no longer the long essay it used to be. It is a worked section with one organizing idea (nothing transfers unless something stays fixed, and what stays fixed decides the correction), a table that sorts the three cases, two derivations of the same shape, and an experiment for each. It also uses what the chapter built: the domain classifier is the logistic regression of the first section, its gradient is "prediction minus label", and label shift is corrected with the confusion matrix. The opening examples (the footwear model, the label swap) and the startup anecdote give it more voice than any other section in the chapter. What holds it back is in the experiments. They are unseeded, so the prose describes ranges over runs the reader cannot see; one arm of the main experiment (clipping) makes no visible difference; and both experiments run on synthetic Gaussians when the chapter has a trained Fashion-MNIST classifier and its confusion matrix at hand. The most important fix is to make the experiments say one thing each, on fixed data, and to run the label shift correction on the chapter's own classifier.

## Scores

| Explanation | Thought | Pedagogy | Language | Enjoyment |
|:-:|:-:|:-:|:-:|:-:|
| 4 | 4 | 3 | 4 | 4 |

## What works

- The opening links to the previous section in its first sentence ("provided the new data come from the distribution of the test set") and gives three one-clause examples a reader recognizes.
- The footwear story, told in six plain sentences and ending on its point: "The model was accurate on the data it was tested on and is useless on the data it now sees, because using it changed the data."
- The label-swap argument under "Types of Distribution Shift". It proves an impossibility with cats and dogs: "no statistic computed from them can tell this world from one without any shift. Yet a classifier that is always right on the source is always wrong on the target." This earns the sentence that follows, on which the whole section rests.
- The two factorizations of the joint density and the table that assigns each kind of shift a factor that changes, one that stays, and a causal reading. `fig_mdl-clf-shift-types` draws exactly the table.
- "Label shift changes the labeling rule as well." A point most treatments miss, made with Bayes' theorem in one line.
- The numbers for the importance weight: $e^{-2} \approx 0.14$ at the origin, $e^2 \approx 7.4$ at the target mean, $e^4 \approx 55$ one unit further. The reader feels why the weights will be a problem before the effective sample size is defined.
- The startup anecdote ("no weighting of the students' samples can stand in for them"). It is the support condition, remembered.
- "The odds that an input came from the target are its importance weight, up to the ratio of the sample sizes". A plain sentence after an identity.
- The baseline in the experiment: the unweighted model does worse on the target (0.60 in the stored run) than always predicting the majority class (0.70). That makes the cost of ignoring shift vivid.
- The worked 2x2 label shift example, including why 23% flagged overstates a prevalence of 10%.
- "Decisions That Change the Data" stays short, names what statistics cannot settle, and stops.

## Structural findings

- **Does it belong in this chapter?** Yes, as written. It answers the last clause of the chapter introduction's question ("a test set vouches only for data that resemble it"), it needs nothing beyond densities, Bayes' theorem and this chapter's tools, and it is the first place the reader uses logistic regression and the confusion matrix for a purpose other than the one they were introduced for. See the chapter file for the longer answer.
- **A paragraph in the wrong subsection.** Subpopulation shift, spurious features and worst-group accuracy are appended to "Concept Shift" (ENV-1).
- **The experiments lean on prose about variability.** See ENV-2. This is a consequence of an unseeded generator combined with the house rule to quote only what survives re-execution. The rule is right; the remedy is to fix the seed and state the result once.
- **Missed connection to the chapter's data.** `softmax-regression-scratch.md` has already computed the column-normalized confusion matrix $\mathbf{C}$ of a trained classifier on ten classes. Resampling the Fashion-MNIST validation images with skewed class proportions (say, shirts five times as frequent), computing $\boldsymbol{\mu}$ and solving $\mathbf{C}\mathbf{p}_T = \boldsymbol{\mu}$ takes six lines, gives a ten-dimensional answer that can be compared with the truth, and would tie the last section to the first five. The scalar Gaussian simulation mostly re-confirms the worked example's $0.84$.
- **Imports** appear in the middle of the section, 280 lines in, with the sentence "The computation needs only NumPy; we import `d2l` for plotting", above a cell that imports `torch`, `nn` and `F`.

## Passage-level findings

**ENV-1** (moderate, structure). Anchor: "Data often consist of groups, such as patients from different hospitals, or", under "Concept Shift".
Subpopulation shift "resembles label shift", as the paragraph itself says, yet it closes the subsection on concept shift. In eight sentences it introduces subpopulation shift, spurious features, worst-group accuracy and group reweighting, with two citations. It is a fourth kind of shift without a row in the table or a panel in the figure. Either give it a short subsection of its own after "Label Shift" with the water-bird example told properly (what the classifier learned, what the group accuracies were), or cut it to two sentences in "Label Shift Correction" where reweighting by class is derived.

**ENV-2** (moderate, language and pedagogy). Anchor: "magnitude: the largest ranges from about 20 to several hundred between runs,", under "A Two-Dimensional Example".
The cells draw fresh random data on every execution, and the prose reports intervals across executions: "In every run the learned $h$ is close to", "occasionally more than 1,000", "sometimes below ten", "40% to 65%", "about 90% to 97%", "between about 1 and 2", "across runs". The reader sees one output and reads a description of a population of outputs. In the stored run the effective sample size is 13, which the prose covers only by "sometimes below ten ... at most about 150", and the weighted boundary has a slope just under 1, at the edge of "between about 1 and 2". Seed the generator, and write what the one run shows at stable precision ("the largest weight is in the hundreds and the 1,000 source examples count for about as much as a dozen"). If the variability is itself a lesson (it is: importance weighting is high-variance), say that once and make it an exercise to rerun.

**ENV-3** (moderate, pedagogy). Anchor: "accuracy by at most about three points either way: a linear model has only", under "A Two-Dimensional Example".
Clipping and the effective sample size are introduced with a formula and a bias-variance argument, then the experiment shows clipping changing nothing, and the prose explains why it did not matter here ("a linear model has only three parameters"). The same sentence says "the few dozen effective examples that remain", while the stored output reports 13. The guide's rule applies: an experiment must show what the prose says it shows. Options: drop the clipped arm from the experiment and keep clipping as one sentence plus the existing exercises, or choose a setting in which the unclipped fit visibly suffers (fewer source examples, or a target mean at 3) and show that clipping helps.

**ENV-4** (minor, language). Anchor: "reason, a classifier that models each class by a Gaussian with a shared", under "A Two-Dimensional Example".
A parenthesis of three sentences introduces generative classifiers and points to the appendix, in the middle of the description of an experiment. It is not needed to read the cell. Delete it, or move one sentence to the chapter's resource list.

**ENV-5** (minor, correctness). Anchor: "p_T(\mathbf{x}) \;d\mathbf{x}\,dy", under "Importance Weights".
The target risk is written with an integral over $y$, though labels in this chapter are classes; the label shift derivation two pages later correctly uses a sum over $y$. Use $\sum_y \int \ldots d\mathbf{x}$ in both.

**ENV-6** (minor, language). Anchor: "estimating one number per class rather than a density over inputs, which", under "Label Shift".
"a density over inputs, which for images have millions of dimensions" attaches the relative clause to the wrong noun on first reading. Write: "one number per class, not a density over inputs with thousands or millions of dimensions".

**ENV-7** (minor, pedagogy). Anchor: "Consider a screening test validated on a source sample with equally many", under "Estimating the Target Class Proportions".
This is a second screening test (84% detection, 16% false alarms, 10% prevalence) next to the one in `classification.md` (90%, 5%, 1%). Reusing the earlier test would let the reader carry the confusion table they already know into the new computation. The 84% is chosen so that the Gaussian simulation reproduces it; if the simulation is replaced by the Fashion-MNIST experiment proposed above, that constraint disappears.

**ENV-8** (minor, thought). Anchor: "about 30% of the runs estimate a negative fraction of sick patients. In", under "Estimating the Target Class Proportions".
A result of an experiment that is not shown ("In the simulation above with a prevalence of 1% instead of 10% and 1,000 target examples"). The point (the linear solve can return negative proportions for rare classes) is worth making; make it by argument (the estimate is $(\mu_1 - 0.16)/0.68$, and $\mu_1$ falls below 0.16 by chance when the true value is near 0.167) or turn it into an exercise.

**ENV-9** (moderate, thought). Anchor: "itself make it fair, since formal definitions of fairness can conflict.", under "Decisions That Change the Data".
Fairness enters in a subordinate clause, with a claim (definitions conflict) that has no example, citation or pointer. As it stands the sentence gestures at a large subject and the reader can do nothing with it. Either give one concrete instance in two sentences (a classifier cannot in general have equal false positive rates, equal false negative rates and equal precision across two groups with different base rates, which follows from the confusion-matrix arithmetic of `classification.md`) or end the paragraph at "whether its errors fall unequally on different groups" and add a reference to the resource list.

**ENV-10** (minor, explanation). Anchor: "boundary rises with a slope between about 1 and 2 along the right arm of the", under "A Two-Dimensional Example".
The final plot is read well, but it is the only place where the reader sees why reweighting helps, and it comes after the accuracy table. Show the plot first (the boundary moves from the bottom of the parabola to its right arm), then the accuracies as confirmation.

## Exercises

Eight problems, strong. "Missing support" makes the support condition quantitative. "Features missing at serving time" is a realistic failure that none of the section's assumptions covers, which is its lesson. "Shift assumptions from unlabeled data" has the reader compute that the same unlabeled target sample implies different optimal classifiers under the two assumptions, the section's opening impossibility argument in numbers. "Learned versus true weights" is the hardest and the most instructive (the sample effective size hides the shortage that the population formula $n e^{-s^2}$ reveals).

- "Real-world shifts" (extended, WILDS) is open-ended to the point of being a reading assignment. Narrow it to one named dataset and one question, or move it to the resource list.
- "Feedback loops" is a discussion prompt; fine as the first, easy problem.
- If the Fashion-MNIST label shift experiment is not added to the text, add it here.

## Still open from October

The notation switches ($p_S, p_T$ versus $q, p$; domain labels $\pm 1$ versus $0/1$) are fixed: the section uses $p_S, p_T$ and $z \in \{0, 1\}$ throughout. "Beyond Static Supervised Learning" and the tank anecdote are cut. The footwear example and Goodhart's law are kept, as October asked.
