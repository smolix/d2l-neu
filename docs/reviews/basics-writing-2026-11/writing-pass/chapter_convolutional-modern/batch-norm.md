# Writing pass: Normalization Layers

Source: `chapter_convolutional-modern/batch-norm.md` (third section of
chapter 7, after `alexnet` and `blocks`). Prose only; code cells, labels,
figures and the slides section are unchanged (`check_prose_only.py`: "OK: only
prose changed"; `lint_source.py` clean).

## Size

Prose words (checker count, exercises included): **4272 before, 4266 after**.
The count is nearly flat because about 190 words of new argument (see
"Claims added or changed") replaced about 200 words of cuts. One unlabeled
display equation was removed (the NFNets weight standardization), so the
`$$` delimiter count went from 9 to 8.

## Before and after

1. Opening, the reader's objection (structural finding 3).
   - Before: "Deeper networks make the problem worse: every layer changes the
     scale of the signals that enter the next one, and these scales drift as
     the weights change during training."
   - After: "Depth makes matters worse, since every layer changes the scale
     of the signals that enter the next one. Have the ReLU and a careful
     initialization (:numref:`subsec_he_init`) not solved this already? Only
     in part: they set the scales right at the start of training, and nothing
     holds them there once the weights move."

2. Opening, what the method does (BN-1, BN-13).
   - Before: "A layer followed by batch normalization computes the same
     output when its weights are multiplied by a constant, so its gradient
     adapts to the size of its weights and one learning rate can serve every
     layer."
   - After: "This rescales the signal on the way forward and its gradient on
     the way back, and it makes the layer indifferent to the length of its
     weights."

3. Scale and shift (BN-3).
   - Before: "With $\gamma = \sqrt{\hat{\sigma}^2_\mathcal{B} + \epsilon}$
     and $\beta = \hat{\mu}_\mathcal{B}$ the layer would undo the
     standardization entirely, so batch normalization does not restrict what
     a network can represent."
   - After: "Nothing is lost by standardizing first: with $\gamma$ and
     $\beta$ equal to the standard deviation and the mean of the feature over
     the data, the layer would restore the original scale and offset, up to
     the fluctuation of the minibatch statistics. Batch normalization
     therefore does not restrict what a network can represent."

4. Scale invariance, the conclusion (BN-5).
   - Before: "The learning rate no longer has to match the scale of each
     layer, so one rate can serve all layers, and larger rates become safe."
   - After: "What happens when the learning rate is too large? Since the loss
     does not change along $\mathbf{w}$, the gradient is orthogonal to
     $\mathbf{w}$, and every step of gradient descent makes the weights
     longer. A step that is too large makes them much longer, which shrinks
     all later gradients of the layer. An overlarge rate thus corrects itself
     to some degree. This is the argument for why normalized networks
     tolerate large learning rates."

5. The name "momentum".
   - Before: "Framework layers call $\rho$ or $1 - \rho$ the *momentum* of
     the layer, a name unrelated to the momentum method of optimization."
   - After: "Unfortunately, some framework layers call $\rho$ the *momentum*
     of the layer and others $1 - \rho$, and the name has nothing to do with
     the momentum method of optimization."

6. Reading the experiment (structural finding 2, BN-2).
   - Before: "Its curve ends without the dips of a constant learning rate."
   - After: "Its curve is flat over the last epochs, where the learning rate
     has decayed. [...] Note, however, that both networks ran at a single
     learning rate. We have seen batch normalization repair gradients that
     were too small. How large a learning rate it tolerates is left to the
     exercises."

7. What is known (BN-10, BN-11).
   - Before: "The scale invariance above is a third, simpler part of the
     answer. The debate is a widely cited example of an effect that was
     established long before its explanation :cite:`Lipton.Steinhardt.2018`."
   - After: "What can be said with confidence is narrower. Batch
     normalization rescales the signal and its gradient at every layer, and
     it makes each layer invariant to the scale of its weights. In practice
     normalized networks also tolerate larger learning rates, and the noise
     of the minibatch statistics acts as a regularizer, much as the noise of
     dropout does :cite:`brock2021nfnet`. Why these effects add up to faster
     training of very deep networks is still argued over. The method spread
     on the strength of its results, well ahead of its explanation
     :cite:`Lipton.Steinhardt.2018`."

8. Normalizer-free networks (structural finding 5; retelling a paper).
   - Before: a 150-word paragraph with the weight-standardization formula,
     $n_\textrm{in}$, the gain $g_o$, "a further constant that depends on the
     activation function", ending "Without batch statistics, the network
     computes the same function in training and at prediction time and its
     examples are independent."
   - After: "They standardize the weights of each convolution instead of its
     outputs, which keeps throughout training the scale that a careful
     initialization (:numref:`subsec_he_init`) sets only at the start, and
     they clip a gradient whose norm is large relative to the norm of the
     weights it updates. We will not use these networks. They matter here as
     evidence: a network that trains this well once the scale of its weights
     is held fixed by other means suggests that controlling scale is the main
     thing a normalization layer does."

## Repetition removed

- **Cosine schedule with warmup** (owned by `alexnet`, the earliest section):
  the restated definition ("a linear warmup from zero to the peak rate over
  the first $t_\textrm{w}$ steps, then the half-cosine ... as computed by
  `d2l.cosine_warmup`") is now one clause with `:eqref:` and `:numref:`, plus
  the one forward reference to `chap_optimization` that the amendments ask
  for.
- **LeNet's architecture and initialization** (owned by `sec_lenet`, chapter
  6): the full layer list (kernel sizes, padding, pooling, the 10 outputs,
  Xavier weights and zero biases) is reduced to the four hidden widths that
  the count $6 + 16 + 120 + 84 = 226$ needs.
- **Dropout's two modes** (chapter 4): "it zeroes units during training and
  passes them through unchanged at prediction time" reduced to a clause; the
  summary no longer repeats the comparison.
- **Lag of the running averages**: told under "Training and Prediction" and
  again before the experiment. Kept only before the experiment, where it is
  used (BN-7).
- **Which axes are averaged**: the scratch subsection no longer repeats the
  previous subsection (and the code comments) on examples against positions.
- **State that is saved but not optimized**: stated once in the shared prose;
  the MXNet and PyTorch tab notes no longer repeat it.
- **The mechanism**: stated three ways (opening, "Scale Invariance", summary)
  with "one learning rate can serve every layer" each time. The three places
  now say the same, narrower thing.
- Tab prose trimmed (guide §9): MXNet `hybridize()` sentence (already a code
  comment, and a chapter 5 fact); PyTorch note on the unbiased running
  variance, a discrepancy the reader is never shown (BN-8); PyTorch "which
  the Trainer applies after the first minibatch has fixed the shapes"; JAX
  note shortened.

## Review findings

Applied (prose):

- **BN-1**: drift sentence dropped from the opening.
- **BN-2**: opening now says "about 73%", as `sec_lenet` does; the result
  paragraph explains the gap to this section's 60%.
- **BN-3**: population statistics, "up to the fluctuation of the minibatch
  statistics".
- **BN-4** (prose half): the signal rescaling has its own paragraph and comes
  first.
- **BN-5**: the missing two steps are stated; "larger rates become safe" is
  gone.
- **BN-6**: dangling remark on other placements cut.
- **BN-7**: lag aside removed from the paragraph on naming.
- **BN-8** (prose half): the four returned quantities are listed with their
  expected values; PyTorch variance note cut.
- **BN-9**: "BN-LeNet" is now `BNLeNet`. The sentence on drops of ten points
  under a constant rate is kept, as a record of the authors' repeated runs
  (amendments, "run-to-run ranges are deliberate").
- **BN-10, BN-11, BN-13**: honesty paragraph and summary rewritten to agree
  with the opening.
- **BN-12**: the two uses of the name "layer normalization" are named and
  tied to the two panels of the figure.
- **BN-14**: synchronized batch normalization cut.
- Structural: the ReLU and He initialization objection is answered in the
  opening; regularization by minibatch noise is mentioned; the NFNets
  paragraph is cut to its use as evidence; one clause says why `BNLeNet` uses
  the framework layers after we built our own.

Deferred:

- **BN-4** (second half): printing $\hat{\sigma}_\mathcal{B}$ for LeNet's four
  hidden layers needs a new cell.
- **BN-8** (first half): labeled output needs a code change.
- **BN-9** (alternative): showing the constant-rate curve needs a new run.
- **Structural finding 2**: promoting exercise 2 (accuracy against learning
  rate) into the text needs a new experiment. The prose now says that the
  experiment does not test this.
- **Exercise 8** (drop the probability integral transform): exercises are
  out of scope for this pass.
- **Symbol collisions** ($\gamma$, $\beta$ across the chapter): needs one
  decision for the chapter.

## Claims added or changed

Please check these; they are new statements, not rewordings.

1. "they [ReLU and careful initialization] set the scales right at the start
   of training, and nothing holds them there once the weights move." From the
   review's structural finding 3.
2. The orthogonality argument under "Scale Invariance": the gradient is
   orthogonal to $\mathbf{w}$, so every step of gradient descent lengthens
   the weights, and an overlarge rate "corrects itself to some degree". This
   is exact for plain SGD without weight decay and with $\epsilon$ ignored
   ($\|\mathbf{w} - \eta \mathbf{g}\|^2 = \|\mathbf{w}\|^2 + \eta^2
   \|\mathbf{g}\|^2$); the text does not state these conditions. Not tested
   in the section.
3. "That it falls short of the 73% of :numref:`sec_lenet` is no surprise,
   since the schedule halves the average learning rate." The factor is
   right (the warmup and half-cosine average to 0.5 of the peak); that this
   is the cause of the lower accuracy is the reviewer's inference, not a
   measurement.
4. "Its curve is flat over the last epochs, where the learning rate has
   decayed" replaces "ends without the dips of a constant learning rate"
   (the constant-rate curve is not shown). Checked on the stored plots of all
   four frameworks. Three of them show small dips in epochs 2 to 5, which the
   text does not mention.
5. "the noise of the minibatch statistics acts as a regularizer, much as the
   noise of dropout does" and "normalized networks also tolerate larger
   learning rates", both cited to `brock2021nfnet`, whose list of the
   benefits of batch normalization the review points to. I did not reread
   the paper.
6. "Later measurements found that batch normalization does not reduce this
   shift" keeps the earlier claim but drops the detail of the injected-noise
   experiment.
7. Layer normalization: "the name is used rather loosely. For a
   convolutional layer it can mean all channels and positions of an example".
8. NFNets: "suggests that controlling scale is the main thing a
   normalization layer does" is an inference the old text did not draw.
9. Removed as not true of every tab: "Our layer is told the number of
   features and the number of axes of its input; the framework layers infer
   both from the first minibatch" (the TensorFlow scratch layer infers its
   shape; `nn.BatchNorm2d(6)` is told).
10. JAX tab: "`use_fast_variance=False` subtracts the mean before squaring,
    as our implementation does."

Stored outputs, all four frameworks, for the numbers the prose quotes:
plain LeNet ends at 0.58 to 0.65 ("about 60%") after two to three epochs at
chance; `BNLeNet` ends at 0.884 to 0.887 ("about 88%") and is at 0.77 to 0.78
after the first epoch ("near 70% or above"); running means about 0.2;
agreement with the framework layer to $10^{-6}$. No number was changed except
"70--73%" to "about 73%" in the opening.

## Slides that now disagree with the body

The slides were not edited.

- "At the MLP's Learning Rate, LeNet Barely Learns in 10 Epochs": "70–73%
  after 10" (body: about 73%), and "these scales drift as the weights
  change", the sentence BN-1 removed from the body.
- "(2, 4, 6, 8), ... Normalize Alike": "$\gamma =
  \sqrt{\hat\sigma^2_\mathcal{B}+\epsilon}$, $\beta = \hat\mu_\mathcal{B}$
  would undo it, so nothing is lost" (body: population statistics, up to
  minibatch fluctuation).
- "Scaling the Weights by a ...": "**One learning rate can serve every
  layer.**" The body no longer claims this.
- "Recap": "one learning rate for all layers", the same claim.
- "A Learning Rate That Decays to Zero ...": calls the model "BN-LeNet"
  (body: `BNLeNet`), as does the title of the following slide.
- The goals slide's "explain why batch normalization makes the gradient
  adapt to the scale of the weights" still matches the body.
