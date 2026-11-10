# Writing pass: `chapter_convolutional-modern/training-recipes.md`

Section 7.6 (sixth of eight: alexnet, blocks, batch-norm, resnet,
efficient-convnets, **training-recipes**, convnext, cnn-design). Prose only;
no code cell, label, figure, exercise, or slide was changed.

## Words

Prose words (checker): 4095 before, 4073 after (-22). The checker ends
"OK: only prose changed"; `tools/lint_source.py` is clean. The history
subsection lost about 330 words; about 300 came back as the content the
review asked for (REC-3, REC-4, REC-7, REC-9, REC-10).

## Before and after

1. Opening.
   - Before: "If an architecture paper reports a gain of two points over a
     baseline trained with an older recipe, the gain may therefore come from
     the recipe, not the architecture. The same study makes this point by
     retraining the unchanged ResNet-50 to the accuracy of several of its
     successors."
   - After: "So when a paper reports two points over a baseline trained with
     an older recipe, we cannot tell whether the architecture or the recipe
     deserves the credit."

2. History.
   - Before: "The 'ResNet strikes back' study :cite:`wightman2021resnet` then
     kept the architecture fixed and trained the unmodified ResNet-50 with
     three procedures of increasing cost: A3 (100 epochs) reaches 78.1%, A2
     (300 epochs) 79.8%, and A1 (600 epochs, the LAMB optimizer, a binary
     cross-entropy loss, and heavy augmentation) 80.4%."
   - After: "Over the following years every one of these choices was
     revisited, first one technique at a time and then in studies that
     combined them :cite:`He.Zhang.Zhang.ea.2019,wightman2021resnet`. [...]
     Do not worry about the names in it: we explain each ingredient where we
     implement it below."

3. Label smoothing.
   - Before: "With $\epsilon = 0.1$ the loss of this example is 0.67 instead
     of 0.0001, so the optimizer stops pushing confident examples further and
     spends its effort on the others."
   - After: "With $\epsilon = 0.1$ the loss of this example is 0.67 instead
     of 0.0001. Is 0.67 large? With three classes the smoothed loss cannot
     fall below about 0.29, a value it reaches when the correct logit leads
     the others by about 3.3. A lead of 10 is therefore penalized as
     overconfident, and the gradient now pulls the logits back together
     instead of pushing them further apart."

4. Cosine schedule.
   - Before: "The rates are 0, 0.002, 0.001, and nearly 0, as the formula
     predicts."
   - After: "The rates are 0, 0.002, 0.001, and nearly 0, as
     :eqref:`eq_cosine_warmup` predicts: nothing, the base rate, half of it,
     and nothing again."

5. CutMix.
   - Before: "CutMix :cite:`yun2019cutmix` mixes in space instead: it cuts a
     rectangle from one image and pastes it into the other [...]"
   - After: "A mixed image is a double exposure that no camera would take.
     CutMix :cite:`yun2019cutmix` mixes in space instead: it cuts a rectangle
     from one image and pastes it into the other [...]"

6. Weight average.
   - Before: "With $\beta = 0.9$ the average reaches back about 10 steps: it
     fluctuates far less than the iterates and follows their mean. The same
     horizon is the risk of the method."
   - After: "With $\beta = 0.9$ the average reaches back about 10 steps. It
     starts at the random initial weight of the model and needs a few such
     horizons to forget it; after that it fluctuates far less than the
     iterates and follows their mean. The slow start is a small instance of
     the risk of the method."

7. Experiment set-up.
   - Before: "The two recipes differ only in their optimizers, schedules, and
     training losses."
   - After: "The network and the data are the same for both recipes. Almost
     everything else differs: the optimizer, the weight decay, the schedule,
     the training loss, and the length of training."

8. Published results.
   - Before: "The experiment changes how published comparisons should be
     read. A gain of a point on ImageNet between two architectures says
     little unless both were trained with comparable recipes, since the
     recipe alone moves a fixed ResNet-50 by four points."
   - After: "How, then, should a table of published accuracies be read? With
     some suspicion. The recipe alone moves a fixed ResNet-50 by four points,
     so a gain of a point on ImageNet between two architectures says little
     unless both were trained with comparable recipes."

## Repetition removed

- **Cosine schedule with warmup** (owned by `alexnet`, the earliest section,
  which displays :eqref:`eq_cosine_warmup` and gives the two reasons). Here it
  is now one sentence with the reference, plus the one reason AlexNet does
  not give (running statistics of batch normalization and of the optimizer
  settle). The sentence about the tail of small rates letting the iterates
  settle is gone. This is why REC-5's request to display the formula again
  here was not followed: under the earliest-section rule this section does
  not own it.
- **ResNet-18 architecture** (owned by `resnet`): the enumeration of the
  stem, the four stages and their widths, the bias conventions, and the
  variance $2/n_\textrm{in}$ of He initialization is reduced to "the
  ResNet-18 of :numref:`sec_resnet`, rebuilt from the library's `Residual`
  block, with the He initialization of :numref:`sec_numerical_stability`".
- **Momentum** (owned by `efficient-convnets`): one reference at first use,
  no explanation.
- **Within the file**: the list of techniques was given in the history
  paragraphs, in the table, and in the summary. The paragraphs are gone, the
  table stays, and the summary names the axes instead of the techniques.
  "Evaluated weights are a running average" was explained under the table and
  again under "Averaging Weights"; only the second remains. The four-point
  ResNet-50 result is still cited three times (opening, after the experiment,
  published results) because each use carries an argument; the A1/A2/A3
  accuracies and the 75.3% to 79.3% result are gone.

## Review findings

Applied:

- Structural 1, REC-1, REC-2: the history paragraphs are cut to one short
  paragraph before the table and two remarks after it. ResNet-D, A1 to A3,
  LARS, knowledge distillation and zero initialization are no longer named.
  Citations for label smoothing, the cosine schedule and RandAugment moved to
  where each is taught.
- Structural 5: one sentence in the opening says the ingredients are not
  specific to convnets.
- REC-3: one clause on what Adam does with a forward reference to
  :numref:`chap_optimization`, one on what AdamW subtracts, and a short
  paragraph in the experiment comparing the two decays per step.
- REC-4: the lower bound 0.29 and the gap 3.3 added.
- REC-5 (in part): "as the formula predicts" now cites the equation.
- REC-6: the five-line sentence is split; "no drop times to tune".
- REC-7 (prose part): the text now says what the stored plot shows at its
  start, and uses it as the instance of lag.
- REC-8 (prose part): heading renamed "Two Recipes on One Network" (it has no
  label); the false "differ only in" sentence replaced; the text says plainly
  that the experiment cannot attribute the gain.
- REC-9 (prose part): the claim "its worst run is better than recipe A's best
  run" is removed, since the table's ranges touch at 90.5%; the stored pair is
  described ("about a point"); the reader is told to rerun the two cells for
  another pair. The run-to-run ranges are kept as they were.
- REC-10: the validation loss under recipe A is now pointed out and tied to
  label smoothing.
- REC-11: the paragraph starts from the four-point result; "backbones", COCO
  and ADE20K replaced by "locating objects in an image". The ImageNet-V2 and
  relabeling paragraph is shortened to its two conclusions.
- REC-12 (in part): the MXNet and TensorFlow notes are shortened (the factors
  0.32 and 0.8 are gone). They cannot move into code comments in this pass.

Deferred:

- Structural 3 (stochastic depth and `EMA` never used in a run), REC-8 as an
  ablation, REC-9's paired-difference column and number of runs: need new
  runs.
- REC-6 (plot both schedules over the same number of epochs) and REC-7 (a
  drifting toy weight, two values of $\beta$): need changed cells and
  outputs.
- REC-7, symbol $\beta$ for the decay against $\rho$ in `batch-norm`: a
  chapter-wide notation decision, and `beta` appears in code comments.
- REC-12, moving the notes into comments: code change.
- Exercises (cost of the four hard ones): exercises were not touched.

## Factual claims added or changed

1. "None of the ingredients is specific to convnets. The same ones are used
   to train transformers." General statement about practice, not about a
   later chapter of the book.
2. Adam "divides each coordinate of the gradient by a running estimate of its
   typical size, so that every weight moves at about the pace that the
   learning rate sets"; AdamW "subtracts from every weight the fraction
   (learning rate times decay) of its value". Wording from REC-3.
3. "LAMB is a relative of Adam for very large minibatches."
4. Label smoothing with three classes and $\epsilon = 0.1$: minimum loss
   about 0.29 at a logit gap of about 3.3. Recomputed: the targets are
   (0.933, 0.033, 0.033), whose entropy is 0.291, and $\log 28 = 3.33$.
5. Decay per step at the peak rate: $0.002 \times 0.05 = 10^{-4}$ for AdamW
   and $0.05 \times 5 \times 10^{-4} = 2.5 \times 10^{-5}$ for SGD, "which
   momentum multiplies by up to 10". The last clause goes beyond the review:
   the decay is added to the gradient and therefore passes through the
   momentum buffer, whose steady-state gain is $1/(1-0.9)$. With it, the two
   recipes shrink weights at a similar rate, and SGD's is the larger.
   **Please check this one.** The MXNet tab (bias-corrected decay) deviates
   from the AdamW figure, as its tab note says.
6. EMA plot: "It starts at the random initial weight of the model and needs a
   few such horizons to forget it." Checked on the stored plots of all four
   frameworks (the start is at about 0.75, 0.1, 0.6 and -0.8); it is least
   visible in the PyTorch tab.
7. "In the pair of runs above it is about a point." Stored outputs: PyTorch
   89.8 and 90.7, JAX 89.4 and 90.9, TensorFlow 89.6 and 90.8, MXNet 89.7 and
   90.5, so 0.8 to 1.5 points.
8. Recipe A: training loss falls to zero, validation loss "ends between 0.4
   and 0.5, above the minimum it reached early in training, even though the
   accuracy kept improving". Recipe B: validation loss "ends below 0.4".
   Read off the stored plots of all four frameworks (A ends at about 0.44 to
   0.51 after minima of 0.35 to 0.39; B ends at about 0.37 to 0.38).
9. "The warmup is our one concession to later practice", with the citation of
   Goyal et al. only. The sentence that He et al. (2016) used a warmup for
   their deepest ResNet is removed.
10. Removed claim: "its worst run is better than recipe A's best run".
11. ImageNet-V2 and relabeling: same claims as before, shorter. "The models
    had not overfitted to the old test set" restates the old "not from
    overfitting to the old test set".

## Slides that now disagree with the body

The slides were not edited.

- The experiment's slides carry the title and kicker "A controlled
  comparison"; the body heading is now "Two Recipes on One Network" and the
  text says the comparison is of two bundles.
- A slide table lists "modern, A3 | 100 | 78.1%" and "modern, A1 (LAMB, heavy
  augmentation) | 600 | 80.4%"; the body no longer names A1 to A3 or gives
  78.1%.
- "Skips: RandAugment, LAMB and LARS details, distillation": the body no
  longer mentions LARS or distillation.
- The slides do not have the two additions of the body: the comparison of the
  decays per step, and the validation loss under recipe A.
