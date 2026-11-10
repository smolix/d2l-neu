# Writing pass: `chapter_convolutional-modern/alexnet.md` (7.1)

Prose only; no code cell, label, figure, or slide changed. Checker:
"OK: only prose changed". `tools/lint_source.py`: clean.

**Prose words:** 3342 before, 3334 after (checker count, exercises included).

Position in the chapter: first section after the introduction, so it owns
every fact it explains within chapter 7 (the warmup and cosine schedule, the
parameter and multiply-add ledger, the ImageNet 2012 numbers). Facts from
earlier chapters (the sigmoid's factor of 1/4, He initialization, the
counting rules, max-pooling) are cited, not re-derived.

## Before and after

1. Before: "The winners of the two previous years, both built on engineered
   features, had top-5 errors of about 28% and 26%."
   After: "The year before, the winning error had fallen by about two
   points. This time it fell by more than ten."
2. Before: "The network found these operators because they reduce the
   classification loss, ..."
   After: "Nobody told the network to look for edges. It found these
   operators because they reduce the classification loss, ..."
3. Before: "A crop can start at any of about $32 \times 32$ offsets ... so
   each image yields about $32 \cdot 32 \cdot 2 = 2{,}048$ different inputs,
   the factor the paper counts. Even with this much data the network overfit
   without dropout."
   After: "... different inputs. These are overlapping views of one
   photograph, of course, and even with them the network overfit without
   dropout."
4. Before: "Our version simplifies the network of the paper in two ways. The
   paper split every layer between two GPUs of 3 GB each, which could not
   hold the whole model, and restricted some layers to their own GPU's half
   of the channels; a single GPU now holds the model, so our version has no
   split."
   After: "Our version leaves out two features of the paper's network, and
   neither is missed. The paper split every layer between two GPUs, because
   one GPU with 3 GB of memory could not hold the model; a single GPU now
   can."
5. Before: "whose derivative is 1 for every positive input, so the gradient
   passes the active units of a layer unchanged, however many layers lie in
   between."
   After: "whose derivative is 1 for every positive input. A ReLU, unlike a
   sigmoid, therefore does not shrink the gradient on its active units. What
   remains is the scaling by the weights of each layer, which the
   initialization below is chosen to control."
6. Before: "We also change the learning rate during training, as AlexNet's
   authors did: they divided it by 10 whenever the validation error stopped
   improving. ... A common schedule handles both ends."
   After: "A constant learning rate suits neither the first steps nor the
   last. ... AlexNet's authors took care of the end by hand: they divided
   the rate by 10 whenever the validation error stopped improving. We use a
   schedule that handles both ends without supervision. It is a later
   default and not part of AlexNet; the following sections use it too, so
   that their results can be compared."
7. New, before the training cell: "What should we expect? Fashion-MNIST has
   neither the number of images nor the detail that AlexNet's capacity is
   meant for. The run checks the implementation and measures the cost. It
   cannot show what made AlexNet win on ImageNet, and the gain over LeNet
   should be small."
8. Before (Summary opening): "Before 2012, image classifiers learned a
   classifier on top of hand-designed features. AlexNet learned the features
   as well, ..."
   After: "Learned features overtook hand-designed ones when three things
   were in place at once: a dataset large enough to constrain tens of
   millions of parameters, GPUs that could compute their gradients in days,
   and ReLU activations, dropout, and data augmentation to keep such a
   network trainable."

## Paper retelling shortened

- ImageNet provenance (WordNet categories, Mechanical Turk verification,
  "images came at varying resolutions") cut; the citation `Miller.1995` is
  no longer used in this file.
- History of programmable GPUs ("In the mid-2000s, NVIDIA and ATI ...",
  `Fernando.2004`) cut; the cuda-convnet sentence reduced to one clause.
- Two-GPU split and local response normalization reduced (pair 4).
- The two earlier challenge winners reduced to one clause (pair 1).

## Repetition removed

- Sigmoid derivative $\sigma'(x) = \sigma(x)(1-\sigma(x)) \le 1/4$: was
  re-derived; now one clause with `sec_lenet` (owner: chapter 6).
- LeNet's learning rate of 2.0 and its reason: told in "ReLU Activations"
  and again in "Training"; the second is now one clause.
- "Almost all parameters sit in the fully connected layers" was stated in
  the opening roadmap, in "Dropout", in "Parameters and Multiply-Adds", in
  the closing paragraph (with 9/10 and 5% again), and in the Summary. The
  roadmap sentence is cut, the closing paragraph cites the table without
  numbers, and "the first of them alone holds 26 million" (a table row) is
  cut. Numbers remain in the ledger paragraph and the Summary.
- "LeNet was trained end to end in 1989 / share most of their design":
  the transition paragraph after the filter figure repeated the opening;
  cut.
- "Upsampled images hold no more information" was said before and after the
  run; now only before.
- Counting rules for parameters: reduced to one sentence with `sec_lenet`.
- Tab prose: "Gluon / Keras / Flax initializes biases to zero" (three tabs)
  replaced by one shared clause.

## Review findings

Applied: ALX-1, ALX-2 (definition moved to the first paragraph), ALX-3,
ALX-4, ALX-5 (three components named; one clause of reason for max-pooling
with `sec_pooling`; overlap noted), ALX-6, ALX-7 ("one reason why"), ALX-8
(sentence split), ALX-9 (prose part: new `### Learning-Rate Schedule`
heading, schedule declared a later default and not part of AlexNet), ALX-10,
ALX-11, ALX-12. Structural finding 2 (say before the run what it cannot
show): applied, under a new `### Comparison with LeNet` heading.

Deferred:
- Structural finding 1, code part: the saved `cosine_warmup` and the four
  redefinitions of `configure_optimizers` across sections stay where they
  are (moving code is out of scope).
- Structural finding 3 (a cell that displays the trained first-layer
  kernels): needs a new cell and output.
- Chapter finding 6 (one home for optimizers and schedules before first
  use): chapter-wide decision. The forward reference to `sec_scheduler` is
  kept.

## Headings added

`### Learning-Rate Schedule` and `### Comparison with LeNet` under
"Training on Fashion-MNIST" (no labels; existing headings and labels
unchanged). The training cell now sits under the second.

## Factual claims added or changed

- "The year before, the winning error had fallen by about two points. This
  time it fell by more than ten." Rests on the removed figures (about 28%
  in 2010, about 26% in 2011, 15.3% in 2012).
- "The template was more than twenty years old" (1989 to 2012).
- "Even that leaves some fifty parameters for every training image" (60
  million parameters of the paper's network over 1.2 million images).
- "These are overlapping views of one photograph, of course" (replaces "the
  factor the paper counts").
- Max-pooling "replaces LeNet's average pooling, because a maximum keeps a
  strong response that an average would dilute (`sec_pooling`), and its
  windows overlap."
- ReLU: "the gradient passes the active units unchanged, however many layers
  lie in between" weakened as in ALX-6; "This is why LeNet needed 2.0" is
  now "one reason why".
- "All biases start at zero, which is the default of every library here
  except PyTorch" (PyTorch's cell zeroes them explicitly).
- MXNet tab: dropped "which MXNet executes several times faster than one
  operator at a time" (not measured here).
- "A CPU at its peak would have needed more than a month, for every
  attempt."
- Schedule: "It is a later default and not part of AlexNet; the following
  sections use it too, so that their results can be compared."
- "We use 0.03, the largest of the peaks we tried at which every network of
  the next section trains reliably (`sec_nin`)." This leans on the
  statement in `blocks.md` that 0.03 is the largest peak tried at which NiN
  trains reliably.
- Result: added "Nor can we credit its size alone with even this gain, since
  the two runs differ in activation, initialization, and schedule as well."
  Removed "AlexNet's extra capacity helps on ImageNet ..." after the run
  (now said before it).
- Output statements unchanged: about 91% validation accuracy, training loss
  about 0.22, validation loss about 0.25, less than a point of change over
  the last epochs. Checked against the stored training curves of all four
  frameworks; all four show this.

## Slides that now differ from the body

- Slide title "A ReLU Passes the Gradient Through; a Sigmoid Multiplies It
  by at Most 1/4" keeps the stronger wording that the body dropped (ALX-6).
- The slides still list the 2010 and 2011 winners (28%, 26%); the body now
  gives only the two-point fall. Consistent, but the numbers are slide-only.
- The Fashion-MNIST slides ("AlexNet Passes LeNet's Accuracy in a Third of
  the Epochs") do not carry the body's caveats: that the run cannot show
  what AlexNet's capacity is for, and that activation, initialization, and
  schedule differ as well as size.
- The schedule slide does not say that the schedule is a later default and
  not AlexNet's.
