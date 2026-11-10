# Normalization Layers

Source: `chapter_convolutional-modern/batch-norm.md`. About 3,650 words of prose before the exercises (a good part of it in per-framework tab notes), 750 words of exercises.

## Verdict

The best-taught mechanism in this half, with one problem of intellectual honesty that is half solved. The section opens on the reader's own experience (LeNet needed a learning rate of 2.0), defines batch normalization with a four-number example, shows the scale invariance in three lines, explains the two modes from a requirement instead of from an API, and runs the one clean paired experiment of the half-chapter. The move from batch statistics to layer and group normalization is motivated by two stated failures and carried by a figure that works. Against Cover and Thomas this is a 4 in most respects.

The honesty problem: the section contains a good, short paragraph saying that the reason batch normalization works "is less settled than what it computes". But the opening, the "Scale Invariance" subsection and the summary all present one causal story as established, the experiment demonstrates a narrower thing than the story claims, and the best-supported findings of the later literature (it permits much larger learning rates; the minibatch noise regularizes) are missing. The single most important fix is to make the section say the same thing in all three places: what batch normalization provably does (rescales signal and gradient; makes the layer scale-invariant), what the experiment shows (it repairs the gradient scale in a sigmoid network), and what remains explanation after the fact.

## Scores

| Dimension | Score |
|---|:-:|
| Clarity of explanation | 4 |
| Clarity of thought | 3 |
| Pedagogy | 4 |
| Language | 4 |
| Ease and enjoyment | 4 |

## What works

- The opening is the guide's model: "A network of five layers should not need such care. Why does the right learning rate depend so strongly on the network, and can a layer inside the network remove that dependence?" The reader has felt this problem.
- The numerical example under "Standardizing over a Minibatch": "The values $(20, 40, 60, 80)$ and $(102, 104, 106, 108)$ are standardized to the same four numbers: the standardization removes the scale and the offset of the feature." Meaning in one sentence, checked on numbers. This is Cover and Thomas's rhythm.
- "A sigmoid, for instance, is nearly linear on inputs of unit variance" gives $\gamma$ and $\beta$ a reason to exist.
- "Training and Prediction" derives the two modes from a requirement: "its output for an image should not depend on which other images happen to share its minibatch". The reading of $\rho$ ("these weights halve about every seven steps ... a small fraction of the 469 minibatches") and the warning that frameworks call "$\rho$ or $1 - \rho$ the *momentum*" will save readers real confusion.
- The parameter accounting: "Each of the $6 + 16 + 120 + 84 = 226$ normalized features thus trades one bias for a scale and a shift", followed after the experiment by "it changed the scale of every layer's signal and gradient, not the capacity of the network." A precise claim, and the best sentence in the section.
- The extreme case that motivates the alternatives: "a minibatch of one example makes every standardized feature 0, since each value equals its own mean; the layer then outputs $\beta$ whatever the input."
- :numref:`fig_norm_axes` with "They differ only in the set of numbers that share one mean and one variance" reduces five methods to one idea. I rendered the figure; it is clear and the caption matches it.
- The group-normalization check that "normalizing the first example alone gives the same output as normalizing it within the minibatch" tests the property that matters, and the Wu and He numbers (34.7% against 24.1% at two images per GPU) make the stakes concrete.
- The stored outputs match the prose: running means of about 0.2 after one step, `{'CosineLeNet': 0.577, 'BNLeNet': 0.884}`, and the plotted curves (plain LeNet at chance through epoch 4).

## Structural findings

- **Three places state the mechanism as fact; one paragraph says it is unsettled.** See BN-1, BN-5, BN-10, BN-13. Decide what the section claims and say it consistently. A workable division: (a) what is provable, the forward rescaling, the $1/\hat\sigma_\mathcal{B}$ factor on the gradient, and scale invariance; (b) what is measured, that batch-normalized networks tolerate much larger learning rates and train faster, that minibatch noise regularizes, and Santurkar et al.'s smoother loss; (c) what is conjecture, any single account of why (b) follows. The NFNets paper cited in this section lists the benefits of batch normalization it had to replace (it downscales the residual branch, removes mean shift, regularizes, and permits large batches and learning rates); that list is the most useful modern summary and the section does not use it.
- **The experiment shows less than the headline.** The index promises that normalizing "makes training faster and less sensitive to the learning rate", and the section says "larger rates become safe". The experiment trains both networks at one rate, 0.1, which is twenty times *smaller* than plain LeNet needs. It shows that batch normalization repairs the gradient scale of a sigmoid network. It does not show insensitivity to the learning rate or safety at large rates. Exercise 2 (final accuracy of both models against learning rates from 0.1 to 10) is the experiment that would. Promote it: two curves of final accuracy against learning rate on a log axis is one cell and one plot, and it demonstrates the claim in the section's first paragraph.
- **The reader has just seen ReLU and He initialization solve the sigmoid problem.** AlexNet and VGG trained without normalization two sections ago. A careful reader asks why they need batch normalization if they will never use sigmoids. The section does not answer. One paragraph should: ReLU and initialization fix the scale at the start of training and in networks of ten or twenty layers; they do not hold it there as the weights move or as depth grows, which is where the next section needs normalization. Without this, the sigmoid LeNet looks like a problem chosen to suit the solution.
- **Regularization is absent.** The word does not occur in the section. Each example is normalized with statistics that depend on 127 random companions, so batch normalization injects noise much as dropout does, and networks with it usually need less dropout. The parallel with dropout is drawn for the two modes ("Dropout ... also has two modes") but not for the noise. Two sentences, and exercise 5 already measures the effect.
- **"Normalizer-Free Networks" does not pass the teachable-core test as written.** The reader will not use weight standardization, and the paragraph is the densest in the section (a formula, $n_\textrm{in}$, a gain, "a further constant that depends on the activation function", adaptive gradient clipping). It earns a place only as evidence about what normalization is for. Cut it to three sentences making that point (networks without any normalization layer can match normalized ones if the scale of the weights is controlled by other means, which supports the view that controlling scale is what matters), and move the formula to an exercise or drop it.
- **The from-scratch layer is implemented, checked, and never used.** `BNLeNet` uses the framework layers. That is a reasonable choice (speed), but say so in one sentence, or use the scratch layer in the experiment, so the reader does not wonder why they built it. Exercise 8 is its only later use.

## Passage-level findings

**BN-1** (moderate, thought). Section opening, second paragraph. Quote: "these scales drift as the weights change during training".
"The cause is the scale of the signals inside the network", and then this sentence, which is the internal covariate shift hypothesis in plain words, stated as fact. Forty lines from the end the section reports that Santurkar et al. "found that batch normalization does not reduce this shift". The opening should claim only what survives: that the scale of signals and gradients differs from layer to layer (shown for LeNet's sigmoids) and that no one learning rate suits all layers. Drop the drift sentence or mark it as the original authors' motivation.

**BN-2** (minor, correctness). Section opening, first paragraph. Quote: "stayed at chance for the first few epochs and reached only 70--73% after ten".
:numref:`sec_lenet` says "about 73% after 10". This section's own baseline, at the same peak rate of 0.1, then reaches "about 60%" (0.577 in the stored run). The difference is the warmup and decay, which lower the average learning rate of a network that already has too little, but the text never says so, and a reader comparing the opening with the result sees a 13-point discrepancy. Add a clause where the result is read: "(lower than the 73% of :numref:`sec_lenet` because the schedule spends its first and last epochs at small rates)".

**BN-3** (minor, explanation). Under "Standardizing over a Minibatch". Quote: "the layer would undo the standardization entirely, so".
$\gamma$ and $\beta$ are fixed parameters while $\hat\mu_\mathcal{B}$ and $\hat\sigma_\mathcal{B}$ change with every minibatch, so no setting undoes the standardization exactly. The intended point holds for the population statistics. Write "With $\gamma$ and $\beta$ equal to the standard deviation and the mean of the feature over the data, the layer would restore the original scale and offset (up to the fluctuation of the minibatch statistics)".

**BN-4** (moderate, explanation). Under "Scale Invariance". Quote: "The same holds for the signal itself. A feature that"
This sentence and the next carry the mechanism that actually explains the experiment (sigmoid outputs near 1/2 have small variance; dividing by a small $\hat\sigma_\mathcal{B}$ restores both the signal and its gradient), and they are appended to a paragraph about a different invariance (rescaling the weights). "The same holds" is loose: the first is an invariance of the function, the second a rescaling of a signal. Give the second its own paragraph, lead with it since it is the simpler idea, and attach a measured number: print $\hat\sigma_\mathcal{B}$ for each of LeNet's four hidden layers at initialization, so the reader can compare the factors $1/\hat\sigma_\mathcal{B}$ that batch normalization restores with the four factors of at most 1/4 that the sigmoids take away.

**BN-5** (moderate, thought). Under "Scale Invariance". Quote: "layer, so one rate can serve all layers, and larger rates become safe".
The conclusion does not follow from what precedes it without two more steps. From "the gradient at $a \mathbf{w}$ is $1/a$ times the gradient at $\mathbf{w}$" the reader can see that large weights get small updates. That "larger rates become safe" needs the consequence spelled out: a step that is too large increases the norm of the weights (the gradient is orthogonal to $\mathbf{w}$, since the loss does not change along $\mathbf{w}$), which shrinks all later gradients of that layer, so an overlarge rate corrects itself instead of diverging. That is a two-sentence argument and a pleasing one. State it, or weaken the claim to what was shown.

**BN-6** (minor, explanation). Under "Fully Connected and Convolutional Layers". Quote: "Other placements, such as after the".
A dangling remark. Either say what changes ("after the activation, where it standardizes what the next layer reads; the ResNet variants of the next section's exercises move it again") or cut it.

**BN-7** (minor, explanation). Under "Training and Prediction". Quote: "so early in training its estimates lag behind".
The lag is real and the section returns to it before the experiment, where it is the stated reason for the decaying learning rate. Here it is an aside at the end of a paragraph about naming. Move it to the paragraph "Prediction mode raises one more issue", where it is used.

**BN-8** (minor, pedagogy). Under "Implementation from Scratch". Quote: "We print the largest per-channel mean and".
The cell returns a tuple of four unlabeled quantities and the prose lists them in a 50-word sentence the reader must hold while reading the output. Print them with labels, or split into two cells (standardization and agreement with the framework; running means). The PyTorch tab note that follows ("Its running variance, which we do not compare, is updated with the unbiased variance") explains a discrepancy the reader was not shown; cut it or reduce it to a code comment.

**BN-9** (moderate, pedagogy). Under "LeNet with Batch Normalization". Quote: "accuracy of BN-LeNet drops by ten points or more in single epochs of some"
An interesting phenomenon (the running averages describe a network that has since moved), told and not shown, from runs the reader does not see, and used to justify a design decision. It is also the only place the model is called "BN-LeNet". Exercise 4 has the reader produce the effect and its cure (recomputing statistics). Either show it (one constant-rate curve added to the plot, with dips) or shorten to: "With a constant learning rate the running averages lag behind the weights, and the validation accuracy in prediction mode fluctuates from epoch to epoch (exercise 4). A rate that decays to zero lets them catch up."

**BN-10** (moderate, thought). After the experiment. Quote: "The scale invariance above is a third, simpler part of the answer."
This is the hinge of the honesty paragraph and it is asserted. Nothing in the section tests whether scale invariance accounts for the gain; "simpler" describes the argument, not the evidence. It is also listed third after two accounts of which one was refuted, so "third part of the answer" miscounts. Suggested replacement for the paragraph's second half: "What can be said with confidence is narrower. Batch normalization rescales the signal and its gradient at every layer, which is what the experiment above shows, and it makes each layer invariant to the scale of its weights. In practice normalized networks tolerate learning rates many times larger, and the noise of the minibatch statistics acts as a regularizer. Why these effects add up to faster training of very deep networks is still argued over."

**BN-11** (minor, language). Same paragraph. Quote: "The debate is a widely cited example of an".
The Lipton and Steinhardt citation is apt, but the sentence is about the literature, not about the method, and "widely cited" is filler. "The effect was established years before any explanation of it was" says it.

**BN-12** (minor, explanation). Under "Group and Instance Normalization". Quote: "normalizes each example over all channels and positions at once, the layer".
The section now has two different operations called layer normalization: over all channels and positions of an example (group normalization with $G = 1$, second panel of the figure), and over the channels at one position (transformers and ConvNeXt, last panel). The formula under "Layer Normalization" is written for a vector of $n$ features and fits both. The text handles it with "Which features are averaged together depends on the model", which is true but leaves the reader to notice the clash. Say it outright: "The name is used for both; they differ in whether positions are pooled."

**BN-13** (moderate, thought). Summary. Quote: "invariant to the scale of its weights, so its gradient adapts to their size"
The summary states the scale-invariance account as the reason for the result ("one learning rate suits every layer: LeNet with batch normalization trains well at a learning rate twenty times smaller than LeNet needs without it") and omits that the explanation is unsettled. A summary is what readers remember. Add one sentence from the honesty paragraph, and attribute the LeNet result to what the section showed (rescaled signal and gradient through the sigmoids).

**BN-14** (minor, language). Under "Group and Instance Normalization". Quote: "remedy: it computes the statistics over the global minibatch of all devices".
Synchronized batch normalization with a forward reference to chapter 13 is a deployment detail in a paragraph about why group normalization exists. Cut.

## Exercises

Eight problems, well graded, and several are better experiments than the one in the text. Exercise 1 (bias before batch norm, then "What happens to such a bias in prediction mode?") tests the definition. Exercise 2 should move into the text (second structural finding). Exercise 4 (running averages against recomputed statistics) is first-rate and long; keep it. Exercise 5 (training-mode evaluation at batch sizes 128 and 1) makes the coupling tangible. Exercise 7 (backward pass, "the gradient with respect to the inputs sums to zero over the minibatch, and explain why") is the classic and well posed. Exercise 8 (probability integral transform, whitening) is a research prompt with three parts of which only "Implement at least one of them" is checkable; keep whitening, since it connects to what standardization leaves undone, and drop the probability integral transform, or mark the whole as open-ended.

## Still open from October

"Uses $\epsilon$ before defining it" is fixed. "Explains train and test modes three times" is fixed in the prose; the per-framework tab notes still each explain the mode flag, which is acceptable. "A 500-word essay on internal covariate shift" has been cut to about 110 words, which is the right length; what remains open is that the rest of the section does not agree with it (BN-1, BN-10, BN-13).
