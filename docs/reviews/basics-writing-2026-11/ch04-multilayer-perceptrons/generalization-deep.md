# Generalization in Deep Learning

Source: `chapter_multilayer-perceptrons/generalization-deep.md`. About 3,600 words of prose, 1,100 words of exercises, eleven code cells, three generated figures, three stored plots.

## Verdict

The most ambitious section of the chapter and, with `mlp.md`, the most rewarding. It takes a question the field has not settled and handles it the way MacKay would: it says so ("The answer is not completely known, so this section measures"), runs four small experiments the reader can repeat, proves one clean proposition, and ends each part with what was learned. Two of its results are memorable: a network that fits random labels to 100%, and a model whose test error goes from a third to 90% when the training set is *doubled*. Against the bar, the weaknesses are of three kinds. The opening puzzle is posed on a model that has not in fact fitted its training data, so the puzzle is weaker than it sounds. The double-descent experiment carries a visible amount of scaffolding (frozen layer, no biases, squared loss, a filter for rarely active units, a paragraph on the neural tangent regime) that slows the reader just before the best plot. And the conclusion claims slightly more for the MLP than the experiments on the MLP show. The most important fix is to simplify the double-descent setup and to tighten the frame, so that the four experiments read as one argument.

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|---|---|---|---|---|
| 4 | 4 | 3 | 4 | 4 |

## What works

- **Honesty about the state of knowledge**, stated once and without hedging: "The answer is not completely known, so this section measures." And at the end of the roadmap: "theories that bound the test error of deep networks directly are still being developed."
- **Random labels.** One plot with four curves: the same network, optimizer and images, "only the labels differ". The conclusion is drawn in one sentence: "Whatever lets the network generalize from the true labels, it is not a lack of capacity to memorize."
- **The VC bound, evaluated.** $\sqrt{203{,}530/60{,}000} \approx 1.8$, "An error never exceeds 1". A bound becomes a number and the number is useless. This is how to teach that a bound is vacuous.
- **"More data can hurt", in two printed lines.** `500 training images: test error 0.34`, `1000 training images: test error 0.88`. The text reads it plainly: "because the larger set puts the model onto its interpolation threshold."
- **Conditioning at the threshold.** The explanation of the peak (a square random matrix is nearly singular in some direction; the exact fit needs a large component along it) is given in words and then shown in numbers: smallest singular value `8.6e-04` at $h = 1000$ against `5.7e-02` at 1,100, norm of the solution 776 against 42. The monotonicity remark ("give the new unit zero weight") is a two-line proof inside a sentence.
- **Minimum-norm proposition.** Statement, proof, picture, check. The figure (`mdl-mlp-min-norm`) shows the argument in two dimensions: steps perpendicular to the line of exact fits. The numerical check (`distance to w* 0.0` from zero, `7.5` from a random start) confirms both halves.
- **Clean labels first.** The plot of accuracy on correctly labelled, mislabelled and validation images is the clearest picture in the chapter of what overfitting *is* in a network: a second phase of training in which wrong labels are memorized and validation accuracy falls.
- **Discipline about data splits.** The best epoch is chosen on held-out training images and judged on test images "which played no part in it".
- Facts from other chapters are restated where they are used (the degree-19 polynomial, the VC bound with its symbols), as the writing guide asks.

## Structural findings

- **One idea, four demonstrations.** The section's idea can be said in a sentence: the model class is large enough to fit anything, so the training procedure decides what is learned. Random labels show the first half. Double descent, minimum norm, maximum margin and clean-first ordering show the second. The idea is stated in the roadmap ("Both observations point to the training algorithm") and in the closing, but the four `##` parts are not explicitly joined to it as they go by. One sentence at the start of each part saying which half of the idea it serves would turn a sequence of phenomena into an argument. At present a reader can come away with a list: random labels, double descent, implicit bias, early stopping, grokking.
- **From random features back to the MLP.** Double descent and minimum norm are shown on a model with a frozen hidden layer and a squared loss; maximum margin on a linear model. Only random labels and clean-first use the trained MLP. The text says so ("Both results concern linear models"), but the step back to the trained network is one citation (Nakkiran et al.) and the sentence "For deep networks, statements this exact are rare". The reader should be told directly what carries over and what is conjecture.
- **Scaffolding in the double-descent experiment.** See GEN-4 and GEN-5.
- **Grokking** is a counterexample to the patience rule and delimits the claim, which is good practice. But it gets a paragraph, a schematic figure that is not data, and a two-bullet extended exercise, for a phenomenon on "small algorithmic tasks" that no later section uses. A sentence would do. See GEN-8.
- **Length.** At 3,600 words this is within budget and the length is mostly experiments. Cuts suggested below amount to about 300 words.
- The imports cell appears after the first paragraph of "Random Labels" instead of before the first `##`, unlike every other section of the chapter. The reader meets an imports block in the middle of an argument.

## Passage-level findings

- **GEN-1** (moderate, thought: the puzzle is weaker than stated)
  - Anchor (section opening): "memorize them. Yet it does not: its accuracy on test images, about"
  - The MLP of `sec_mlp-implementation` has about 89% training accuracy after 30 epochs. It has not memorized its training set because it was not trained long enough to, with a slow initialization, and its validation accuracy was still rising. So "Yet it does not" is true for a mundane reason (it is in effect early-stopped), not for the deep reason the section goes on to investigate. The 89% is also new to the reader: `mlp-implementation.md` reports no training accuracy.
  - Fix: pose the puzzle as a possibility, then make it real: "It could memorize them. After 30 epochs it has not: [numbers]. But that run was short. What happens if we train until the network fits every training label?" The 1,000-image experiment then answers exactly this question (99% training accuracy, about 80% test accuracy with true labels), and the gap it shows is the thing to be explained.

- **GEN-2** (moderate, explanation: the key step is in an exercise)
  - Anchor (Capacity Bounds): "rule out every bound that depends only on the class, $n$, and $\delta$."
  - This is the logical punchline of the random-label experiment and it is delegated to exercise 1. The argument is two sentences.
  - Fix: write it out: "Such a bound must hold for every distribution, including images with random labels. There the trained network has training error 0 and population error 90%, so the bound's second term is at least 0.9. The same term then applies to the true labels, where it guarantees nothing." Keep the exercise as a check.

- **GEN-3** (minor, correctness)
  - Anchor (Inductive Bias): "The learners we have met make different bets:"
  - The table lists 1-nearest neighbor, which the book has not introduced (I found it only in an exercise of the linear-algebra section). Its last row reads "the subject of the rest of this section", a placeholder inside a table.
  - Fix: "Different learners make different bets", introduce the nearest-neighbor rule in one sentence before the table (the paragraph after it does this, so reorder), and replace the last row's cell with a short honest entry such as "not fully known; the rest of this section".

- **GEN-4** (moderate, pedagogy: experiment friction)
  - Anchor (The Interpolation Threshold): "draw 11,000 units and keep the first 10,000 of those that are active on at"
  - To make the square feature matrix invertible, the text removes hidden units that are active on fewer than ten images. This costs a forward reference ("as described below"), a full paragraph on why nonnegative pixels produce nearly empty columns and proportional columns, an extra printed statistic, and three lines of code. None of it is about double descent. By the writing guide's own test (§9: an experiment that needs a paragraph of confounders is a research experiment), the setup should be simplified.
  - Fix: remove the cause instead of filtering the effect. Subtracting the mean training image from every image (or adding a random bias to each unit) makes each unit active on about half the images, and the filter, its paragraph and the forward reference disappear. Check that the peak at $h = n$ survives; it should, since it comes from the conditioning of a square random matrix.

- **GEN-5** (moderate, density)
  - Anchor (The Interpolation Threshold): "from the gradients at their initialization; this is the *lazy* or *neural"
  - Between the setup and the experiment sits a paragraph that names the random-features model, the lazy regime, the neural tangent regime, cites two papers, and points forward to `sec_scaling` for "width-dependent learning rates". Four names in four sentences, none used again in the section.
  - Fix: keep the first sentence (the model has a name and a citation). Replace the rest with the one fact the reader needs: "Real networks also change their hidden features during training, so this model is a simplification that we can solve."

- **GEN-6** (minor, thought: conclusion inside the noise)
  - Anchor (The Interpolation Threshold): "error falls again, and at 10,000 hidden units it reaches 18% to 23%, below"
  - "Below the best width before the threshold", whose range was just given as 21% to 25%. The ranges overlap, and in the stored run the two values are about 0.205 and 0.225. The claim that matters is that the error comes back down to the level of the best small model although the model now fits every label.
  - Fix: "it is back at about a fifth, as low as the best width before the threshold or lower".

- **GEN-7** (minor, pedagogy)
  - Anchor (Random Labels): "Two small helpers run the experiments. `ArrayData` serves arrays as"
  - The helper classes are needed, and the paragraph describing them is fine, but together with the data cell and the model cell the reader goes through four cells and four paragraphs of setup between the question ("train the network on labels drawn at random") and the experiment.
  - Fix: state the expected outcome before the setup, so the reader has something to hold: "If the network can fit random labels, its training accuracy will reach 100% while its test accuracy stays at 10%. We need three pieces of setup first."

- **GEN-8** (minor, scope)
  - Anchor (The Stopping Rule): "after a hundred to a thousand times as many steps (:numref:`fig_grokking`). This is called *grokking*. A patience"
  - A named phenomenon from one paper, a schematic figure and a double-starred exercise. It delimits the patience rule, which is its value, but the figure is a drawing of a result the reader cannot check here.
  - Fix: keep two sentences without the figure ("The rule assumes that validation accuracy does not rise again after a long plateau. On some small algorithmic tasks it does, after a hundred times as many steps, a phenomenon called grokking."). Keep the exercise for those who want it, or move the figure to it.

- **GEN-9** (moderate, thought: closing claim)
  - Anchor (closing paragraph): "Gradient descent, not the parameter count, decides which of the many exact"
  - As a summary of the two linear results this is right. As a statement about "a network" it goes beyond what the section showed for the MLP, which is the order of fitting. The honest position, which the opening took, has been dropped by the end.
  - Fix: restore it, in the manner of MacKay's "is not going to make all these problems go away": "For the linear models we could say exactly which fit gradient descent finds. For the MLP we have seen one preference, clean examples before noisy ones, and no complete account of why its fits generalize exists yet."

- **GEN-10** (minor, explanation)
  - Anchor (Maximum-Margin Classification): "smallest margins dominate the gradient, and gradient descent works mostly on"
  - The intuition is good. The term "margin" is defined as $m = y\mathbf{w}^\top\mathbf{x}$ and then "maximum-margin" is defined geometrically ("smallest distance to a training point is largest") without connecting the two (distance is $m / \|\mathbf{w}\|$).
  - Fix: one clause: "the distance of an example from the separator is $m / \|\mathbf{w}\|$".

- **GEN-11** (minor, notation)
  - $m$ is the margin here, the mask in `dropout.md`, the mean square $m^{(k)}$ in the initialization section and the number of compositions in `mlp.md`; $\mathbf{h}$ is a row of the feature matrix in the minimum-norm figure and the hidden activation elsewhere; $\tau$ is a weight standard deviation in the previous section and a tolerance in exercise 2. Each use is local and defined. A chapter-level notation pass should still reduce the reuse.

## Exercises

Eight problems. The first five are well matched to the text: "More data can hurt" extends the two-line result to a curve; "Weight decay at the threshold" asks whether tuning $\lambda$ removes the peak, which is the natural next question; "Early stopping and weight decay" makes the "nearly equivalent" claim of the text exact.

- Exercise 1 should become a check once GEN-2 moves its argument into the text.
- "Epoch-wise double descent" and "Grokking" are both extended and both need long runs (several hundred epochs; $10^5$ steps). Two such problems at the end of one section is one too many. Keep the first, which uses the section's own model, and fold grokking into a pointer.
- "Implicit bias of the logistic loss" part (b) is the support for the claim in the text that the parameter norm is not a property of the function. Good.

## Still open from October

October called this "4,000 words of qualified survey with no experiment" that "repeats double descent from ch. 2" and asked for 1,200 words around one experiment. The section has been rebuilt around experiments and is no longer a survey; double descent is now continued from chapter 2 instead of repeated ("We now continue past this point"). The length is still about 3,600 words. I think the length is now earned, with the cuts in GEN-4, GEN-5 and GEN-8.
