# Dropout

Source: `chapter_multilayer-perceptrons/dropout.md`. About 3,400 words of prose, of which about 270 in nine tab notes; 760 words of exercises; nine code cells; one figure (`dropout2.svg`); three stored training plots.

## Verdict

The first third of this section is very good textbook writing: the operation, its mean and variance, and a two-unit example that shows in four lines why the test-time network is not the average of the training-time networks. The rest has a problem the prose cannot fix. The section's experiment trains the same MLP with and without dropout and finds no difference in accuracy ("which of them is ahead changes from run to run"). The text reports this honestly, and then spends a thousand words on a method that, on the evidence shown, did nothing for the quantity the reader cares about. The one positive finding, that a network trained with masks tolerates masks, is read as evidence of "co-adaptation", which it is not. The writing guide (§9) describes this case exactly: fix the experiment or remove it, do not keep it and explain what it fails to show. The most important fix is an experiment in which dropout visibly changes the outcome.

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|---|---|---|---|---|
| 4 | 3 | 3 | 4 | 3 |

## What works

- **Masking and Rescaling.** Motivation in one sentence ("With half of the units set to zero, a weighted sum over them is on average half as large"), the definition, the expected value, the variance, and a plain reading of each: "For $p = 0.5$ a kept unit is doubled and a dropped one vanishes, so the standard deviation of $h_j'$ equals $|h_j|$." Then the contrast that matters: "dropout perturbs a unit by as much as its own value."
- **The two-unit counterexample.** $h_1 = h_2 = 1$, $z = h_1' - h_2'$, $\mathbb{E}[z] = 0$ but $\mathbb{E}[\textrm{ReLU}(z)] = 0.5$ against $\textrm{ReLU}(0) = 0$. This is a Cover and Thomas counterexample: the smallest case that fences a claim. It is the best single passage in the section; preserve it exactly.
- **The check on a million ones.** Predicted mean 1 and variance $p/(1-p) = 0.25$; printed `mean 1.000, variance 0.250`.
- **The reading of the training curve**: the plotted training loss "is computed with the masks switched on, so it measures the thinned networks", which is why it starts above the validation loss. A point that confuses most first-time users, explained in two sentences.
- **The ensemble paragraph** is sceptical in the right way: "Every training step trains as many of them as the minibatch has examples, each on one example", then "The reading is loose", with two reasons and the exact statement of what does hold (a geometric mean, when only the last hidden layer is masked).
- **The opening** connects to the previous section's conclusion (the training procedure decides which fit is found) and places dropout among the regularizers already met.

## Structural findings

- **The experiment does not show the method working.** See DO-1. This also affects the chapter: it is the third experiment in a row (after 84% to 87% for the hidden layer, and before 0.15 to 0.135 on house prices) in which the new idea changes little or nothing on real data.
- **Two noise-is-weight-decay derivations.** "Simplicity and Robustness" derives that additive input noise on a linear model equals weight decay, then says the analysis "motivates dropout without explaining it". "Noise Injection", 600 lines later, derives that dropout noise on the inputs of a linear model equals a weighted weight decay. The algebra is the same computation twice (cross term vanishes, variance adds). The first is a detour the section then disowns; the second is the one about dropout. October noted that the section "disowns its own intuitions"; one instance remains. See DO-4.
- **Proportion.** At 3,400 words the section is as long as backpropagation, for a method the text itself says is used "sparingly" in later architectures and often not at all in large language models. With one noise derivation, no learning-rate digression and a two-sentence "Use in Practice", it would be about 2,300 words and better.
- **Three training runs of 20 epochs** (scratch, concise, baseline) before the reader sees a comparison. The concise run exists only to report "the same validation accuracy". Consider training the concise model as the baseline arm or dropping its training run.
- **"Interpretations" comes after the result it should interpret has turned out null.** If the experiment is replaced (DO-1), the three interpretations have something to explain. As it stands, "why it helps there is a different question" is the bridge, which asks the reader to care about benchmarks they have not seen.

## Passage-level findings

- **DO-1** (major, pedagogy: the experiment)
  - Anchor (Accuracy and Robustness): "On Fashion-MNIST, with this network and 20 epochs of training, dropout does"
  - The sentence continues "not make the MLP more accurate". Stored output: 0.861 with dropout, 0.845 without, and the text says the order flips between runs. The two-layer MLP at 20 epochs is not overfitting enough for a regularizer to matter (its training loss is 0.25 against a validation loss near 0.35 to 0.40).
  - Fix: use a setting in which the baseline overfits, and the chapter already has one. The 1,000-image subset with 20% label noise of `subsec_early_stopping`, trained for 200 epochs, memorizes the wrong labels and loses several points of validation accuracy. Running it with and without dropout would show whether dropout delays or prevents that (the section's own exercise 3 asks this). If it does, the section has its demonstration and a link to the previous one. If no small experiment shows a benefit, say so in two sentences, shorten the section to the mechanism, and move the interpretations to further reading. Either way, do not present three interpretations of an effect the reader has just seen to be absent.

- **DO-2** (major, thought: inference not supported)
  - Anchor (Accuracy and Robustness): "are missing: they are co-adapted. Dropout removes this dependence among the"
  - The measurement: with masks switched on at test time, the network trained with masks loses about 1 point, the one trained without loses about 10. The text concludes that the second network's units "have learned to work in the combinations present during training" and later that the measurement "confirm[s] the effect that dropout was designed to have". But any network evaluated under a perturbation it was not trained on does worse than one trained on it. Masking half of a layer and doubling the rest changes the scale and variance of every pre-activation; a network can fail under that without any unit depending on a particular partner. The experiment shows robustness to the training-time noise. It does not separate that from co-adaptation, and the opening's definition of co-adaptation (a unit useful "only together with particular other units") is never tested.
  - Fix: claim what was measured: "The network trained with dropout tolerates missing units; the other does not. Whether that tolerance reflects units that depend less on each other, or only familiarity with the noise, this experiment cannot say." If a test of co-adaptation is wanted, it needs a different measurement (for example, accuracy when single units are removed without rescaling, or the correlation between units).

- **DO-3** (moderate, scope: digression)
  - Anchor (Accuracy and Robustness): "85% to 87%. The fix is to decay the learning rate toward the end of"
  - A paragraph on why validation accuracy wobbles at a constant learning rate, followed by learning-rate decay, warmup, and a forward reference to `sec_scheduler`. This is about SGD, not dropout, and it sits between the result and its reading.
  - Fix: explain the wobble once, in `mlp-implementation.md` where it is first seen (IMPL-5). Here: "As in :numref:`sec_mlp-implementation`, single epochs vary by a point or more, so we compare averages over the last five epochs."

- **DO-4** (moderate, repetition)
  - Anchor (Simplicity and Robustness): "For a linear model this is exactly weight decay. Let the input"
  - Anchor (Noise Injection): "Gaussian input noise pushes a linear model toward small weights. For a"
  - See the structural finding. The first derivation (additive noise) ends by saying it does not explain dropout. The second (dropout noise) is the relevant one and gives the more interesting penalty, $\frac{p}{1-p}\sum_j x_j^2 w_j^2$, which scales with the feature.
  - Fix: delete the additive-noise derivation from "Simplicity and Robustness", keeping its first paragraph (robustness as a third measure of simplicity) as motivation in three sentences. Do the one derivation, with dropout noise, in "Noise Injection", and mention there in a clause that additive Gaussian noise gives plain weight decay.

- **DO-5** (moderate, scope: a list of forward references)
  - Anchor (Use in Practice): "the early 2010s, among them AlexNet (:numref:`sec_alexnet`), from"
  - The subsection names AlexNet, data augmentation, batch normalization, residual connections, stochastic depth, a training-recipes section, the dropout-before-batch-norm variance mismatch, transformers, attention weights and large language models, with five forward references. The reader of chapter 4 knows none of these. The writing guide rules out "In Practice" tails (§12.2) and name lists (§4.3).
  - Fix: two sentences that a reader can use now: "Dropout matters most where a network has many more parameters than it has examples and sees each example many times. Later chapters use it at small probabilities, around 0.1, and models trained on data they see only once often omit it." Move the rest to the chapters that introduce those architectures.

- **DO-6** (minor, experimental precision)
  - Anchor (Training): "below it, at about 0.32, while the validation loss levels off near 0.35."
  - Two-decimal loss values, repeated later ("ends at about 0.25, against 0.32"). They match the stored plots, but by the project's own rule on re-execution noise they are the kind of number that drifts. The point is the ordering of the curves.
  - Fix: "It ends a little below the validation loss" and "ends well below it, at about 0.25".

- **DO-7** (minor, correctness of framing)
  - Anchor (closing paragraph): "Dropout completes the regularizers of this chapter: weight decay limits the"
  - Weight decay was introduced in chapter 2 (`sec_weight_decay`), not in this chapter.
  - Fix: "the regularizers we have met".

- **DO-8** (minor, notation)
  - Anchor (Exercises): "In a network with $L$ hidden layers, each followed by dropout with"
  - $L$ is the loss everywhere else in the chapter, and the chapter counts layers with $K$.
  - Fix: "$K - 1$ hidden layers".

- **DO-9** (minor, explanation)
  - Anchor (Masking and Rescaling): "the same expected size during training and at test time. Inverted dropout puts"
  - The paragraph on the original formulation is clear, but "inverted dropout" as a name is introduced for a version the reader has no uninverted version to compare with until the next sentence. Minor reordering: describe the original scheme first (scale weights by $1 - p$ at test time), then say frameworks instead rescale during training, "which is called inverted dropout".

## Exercises

Seven problems, a good mix. "Dropout and He initialization" asks a question the text raised and did not answer (dropout multiplies the mean square by $1/(1-p)$ during training). "Ensemble size" turns $2^{512}$ into a probability argument. "Dropout at test time" is the geometric-mean result cited in the text.

- "Dropout and label noise" is the experiment that should be in the text (DO-1). If it moves there, replace it with a variation (different noise level or probability).
- "Monte Carlo dropout" mentions calibration, which the reader met in chapter 3; fine.
- "Dropout probabilities" part (a) is a sweep with a question; passes.

## Still open from October

October: "derives inverted dropout twice ... and then disowns its own intuitions." The duplicate derivation of the rescaling is gone. The disowning remains in one place: the additive-noise derivation that "motivates dropout without explaining it" (DO-4). A new, larger version of the same problem is the experiment that shows no gain (DO-1).
