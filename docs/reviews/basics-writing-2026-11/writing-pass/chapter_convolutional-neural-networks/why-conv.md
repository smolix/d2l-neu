# Writing pass: `chapter_convolutional-neural-networks/why-conv.md`

Prose only. No code cell, label, figure include, or slide was changed. Checker:
"OK: only prose changed"; `tools/lint_source.py` clean.

## Size

Prose words (checker count, exercises included): **3275 before, 3270 after**.
Display-equation delimiters 11 to 10: the unlabeled continuous convolution
integral became a clause (WHY-11). Every labeled equation is unchanged.

The count is flat although about 200 words were removed, because three review
findings asked for additions (why the hidden layer has one unit per pixel,
which half of the proposition matters, a concrete failure case and the term
"inductive bias").

## Before and after

1. Opening, why the hidden layer has the size of the image (WHY-2).
   - Before: "We start from a layer that maps a $1000 \times 1000$ image to a
     hidden representation of the same size, which has $10^{12}$ weights."
   - After: "The hidden units of an MLP have no location, so we first give the
     hidden layer one unit per pixel, a map of responses the size of the
     image. For a $1000 \times 1000$ image this layer has $10^{12}$ weights,
     which looks like a step in the wrong direction."

2. One-dimensional derivation.
   - Before: "Set the biases aside (the proof below shows that they must all
     be equal) and feed the layer a signal with a single 1 at sample $k$."
   - After: "What does equivariance demand of this matrix? Set the biases
     aside (the proof below shows that they must all be equal) and feed the
     layer a signal with a single 1 at sample $k$."

3. End of the one-dimensional derivation.
   - Before: "Panel (c) has 3 weights, the same three in every row: the layer
     compares one small pattern with the signal at every position."
   - After: the same sentence, followed by "This is the whole derivation. The
     rest of the section repeats it with more indices."

4. Before the proposition (WHY-9).
   - Before: "... is equivariant exactly when its weights and its bias do not
     depend on the location $(i, j)$ of the hidden unit."
   - After: the same, followed by "That shared weights give equivariance is a
     one-line check. The converse matters more: equivariance leaves no other
     choice."

5. After the proof.
   - Before: "The image with a single bright pixel plays the part of the
     signal with a single 1, and a tensor $\mathsf{V}$ that does not depend on
     $(i, j)$ is the matrix that is constant along its diagonals."
   - After: "In words, a single bright pixel reveals the weights, as the
     single 1 did in one dimension, and a tensor $\mathsf{V}$ that does not
     depend on $(i, j)$ is the matrix that is constant along its diagonals."

6. Opening of "Locality".
   - Before: "We next impose locality: the hidden unit at $(i, j)$ should
     depend only on pixels within distance $\Delta$ of $(i, j)$ along each
     axis."
   - After: "Four million weights are still too many for one pattern, and most
     of them tie a hidden unit to distant pixels. We therefore impose
     locality: the hidden unit at $(i, j)$ should depend only on pixels within
     distance $\Delta$ of $(i, j)$ along each axis."

7. The restriction paragraph (WHY-10).
   - Before: "When they do not, for instance when the same local pattern calls
     for a different response at different locations, the model may fail even
     to fit its training data."
   - After: "When they do not, the model may fail even to fit its training
     data. In a passport photograph, a dark blob in the upper half is an eye
     and a similar one lower down is a nostril, yet a convolutional layer must
     respond to both alike."

8. Cross-correlation.
   - Before: "Because the kernel is learned, a layer that cross-correlates
     with $\mathbf{K}$ and a layer that convolves with $\mathbf{K}'$ represent
     the same functions."
   - After: "Does the difference matter? For a learned kernel it does not: a
     layer that cross-correlates with $\mathbf{K}$ and a layer that convolves
     with $\mathbf{K}'$ represent the same functions."

## Repetition removed

- **Pixel shuffling** (owned by the chapter introduction): the sentence "If we
  permuted the pixels of every image in the same way, it would learn just as
  well" is gone; the opening now has one clause, "As the chapter introduction
  noted, ...".
- **"More weights than pet photographs could determine"** was said in the
  opening and again at the $10^{12}$ count ("about a million times the number
  of training images we could plausibly collect"). Kept once, in the opening.
- **The $10^{12}$ count**: dropped from the end of the "Locality" count
  paragraph, where it appeared a third time in the body. It remains where it
  is announced, where it is derived, in the comparison with $4 \times 10^6$,
  in the restriction paragraph (protected by the review), and in the Summary.
- **Color image triples the count to three billion**: removed; the number was
  never used again.
- **Equivariance stated twice in a row** (list item 1, then the definition):
  the list item is now one line and the definition carries the term.
- **"One weight per offset instead of one per pair of locations"** after the
  proposition: removed; the one-dimensional paragraph says it.
- **Neocognitron** (told in the chapter introduction): removed here. The other
  two history sentences are gathered into one, after the layer is named
  (WHY-8).
- **Infinite grid** was explained in a paragraph and restated in the first
  sentence of the proposition. The paragraph now stands just before the
  proposition and the proposition's hypothesis is one clause (WHY-4).
- **Summary**: the 46-word new claim about a stack with an aggregating head is
  replaced by the reviewer's one-clause statement of the restriction (WHY-14).

Facts this section owns and still explains in full, as the first section of
the chapter: locality, translation equivariance and invariance, the parameter
ladder $10^{12} \to 4 \times 10^6 \to (2\Delta+1)^2$, the cross-correlation
convention, the definition of feature map, and the channel parameter count
with the 1,792 example (`channels.md` cites that number from here).

## Review findings

Applied:

- WHY-1: shuffling claim reduced to a clause.
- WHY-2: reason for one hidden unit per pixel. I kept the $10^9$ / 4 GB
  example (the slides use it) and dropped only the three-billion figure.
- WHY-3: two requirements in the list; the third is a sentence after it.
- WHY-4: infinite-grid paragraph moved to just before the proposition.
- WHY-5: already in the file from today's edit; voice polished lightly.
- WHY-7, in prose only: the motive for relabeling by offset now stands before
  the display ("Both constraints concern where a pixel lies *relative to* the
  hidden unit, so the second line relabels ..."). The display is unchanged.
- WHY-8: history gathered into one sentence after the layer is named.
- WHY-9: which half of the proposition matters; Summary says "layers of this
  form" instead of "linear layers".
- WHY-10: concrete case (passport photograph) and the term "inductive bias"
  with a reference to `sec_generalization_deep`, where it is defined.
- WHY-11: integral reduced to a clause; citation of Rudin dropped.
- WHY-12 (a): the stray $1024 \times 1024 \times 3$ removed.
- WHY-14: Summary no longer introduces a new claim.
- WHY-15: reference to `sec_sentiment_cnn` (Attic) cut.
- The Waldo figure `fig_waldo_mask` now follows the channel paragraphs that
  refer to it; the paragraph on one- and three-dimensional grids moved below
  it (the figure line itself did not move relative to other figures).

Deferred:

- WHY-6: palette and labels of the weight-sharing figure (figure generator).
- WHY-7, splitting the display `eq_mlp_fourth_order` in two: changes a
  labeled display equation.
- WHY-12 (b): channels-last against channels-first index order is a
  chapter-wide notation decision. `channels.md` reconciles the two with
  "(there called $d$)".
- WHY-12 (c): cutting the channel count to leave it to `sec_channels`. Under
  the ownership rule the earliest section keeps it, and `channels.md` cites
  the 1,792 from here. The later sections should be the ones to shorten.
- WHY-13: withdrawn by the author. The suggested extra figure of equivariance
  needs a new figure.
- Exercise suggestions (merge parts of exercise 5, add a periodic
  one-dimensional exercise): exercises were not touched.

## Factual claims added or changed

- "The hidden units of an MLP have no location" (new, the motive for the
  $10^{12}$ layer).
- "Four million weights ... most of them tie a hidden unit to distant pixels"
  (new; true since nearly all of the $1999^2$ offsets are large).
- Passport photograph example: a convolutional layer must respond alike to
  the same local pattern at two heights (new illustration of an existing
  claim).
- "Such assumptions, made before any data are seen, are the layer's inductive
  bias" (new term, referenced to chapter 4).
- History sentence: "Among the first models to share weights across positions
  in this way were time-delay neural networks for speech recognition and the
  shift-invariant networks of Zhang et al. (1988)". The Zhang citation used to
  hang on the equivariance statement; it is now attributed as weight sharing,
  following the title of the cited paper ("Shift-invariant pattern recognition
  neural network ..."). Please confirm that this is the attribution intended.
- "a few hundred weights" became "at most a few hundred weights" (for
  $\Delta < 10$ the count is at most $19^2 = 361$, and 9 for $\Delta = 1$).
- "The name comes from mathematics, and it does not quite fit" (judgment; the
  paragraph then shows the flip).
- Removed: the three-billion count for color, the "million times the number of
  training images" comparison, the Summary's claim about stacks with an
  aggregating head, the continuous integral, the Neocognitron sentence.

## Slides that now differ from the body

No slide contradicts the body. Differences of emphasis only:

- "Equivariant Layers Move with the Input; Invariant Heads Do Not" carries the
  infinite-grid remark; the body now makes it later, before the proposition.
- "The Savings Come from Restricting the Function Class" uses the nine-cell
  grid as the failure case; the body uses the passport photograph (the grid
  remains exercise 4).
- "A Grid-to-Grid Dense Layer Has a Fourth-Order Weight Tensor" comes before
  the one-dimensional figure in the deck, while the body now derives the
  one-dimensional case first (a consequence of today's earlier edit, not of
  this pass).
