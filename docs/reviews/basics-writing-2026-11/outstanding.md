# Basics writing pass: outstanding issues (2026-11-09)

State after the voice-and-repetition pass over Part 1 (chapters 1 to 8, all
sections and introductions). The pass changed prose only: no code cell, label,
figure, or slide section was touched, so the committed `outputs/` store is
still fresh. Related files in this directory:

- `overview.md`: the review, with the author's decisions at the top.
- `chNN-*.md` and `chNN-*/`: chapter and section reviews (finding IDs).
- `writing-pass/`: one log per section with before/after sentence pairs,
  findings applied and deferred, claims added, and slide disagreements.
- `needs-dev-box.md`: changes that need the GPU machine.

## 1. For the author to read and confirm

Judgments and claims the writers added. Each section log lists all of them;
these are the ones most worth a look.

**Strong judgments (voice).**

- 2.6: the IID assumption is "a bold one"; counting parameters is
  "convenient but rather crude".
- 2.7: "We should be candid about how little was gained" by the headline
  weight-decay experiment (predicting $b^*$ everywhere reaches about the same
  loss).
- 4.2: std 0.01 "is not a good scale for a network with a hidden layer".
- 4.5: the two linear results "prove nothing about our MLP".
- 5.3: "We have not trained this stack, and with initial activations in the
  tens of thousands there is little reason to try."
- 5.2: the third part (tying, freezing, EMA) is "closer to a reference" and
  "can be skimmed now".
- 6.4: a $1 \times 1$ kernel is "a convolution in name only".
- 6.5: pooling helps "less than the word 'invariance' suggests".
- 6.6: a gain of one to three points over the MLP should not be over-read
  (depth, activation, and learning rate also differ).
- 7.6: "The remedy is better understood than the disease."
- 7.8 (chapter ending): ResNet-18 at about 94% on upsampled Fashion-MNIST left
  later designs little room to show an advantage.
- 8.5: the embedding table is "little more than a fixed random projection";
  a tie with the trigram model "is less than the hidden state seemed to
  promise".

**Claims made from memory of a paper, not re-read.**

- 5.7 numerics: Micikevicius et al. (2018) report networks that match their
  fp32 accuracy only with loss scaling.
- 7.6 resnet: He et al. report healthy gradient norms in plain
  batch-normalized networks.
- 7.3 batch-norm: minibatch noise regularizes, cited to `brock2021nfnet`;
  the orthogonality argument that an overlarge learning rate "corrects itself
  to some degree" (plain SGD, conditions unstated).
- 7.7 convnext: "We know of no convincing argument" for the gain from fewer
  activations (now in the authors' voice).
- 5.4 custom layers: the stop-gradient form of the straight-through estimator
  "is the form one usually meets".

**Claims derived by arithmetic, not printed by a cell.**

- 1.4: in float32 the finite-difference error is flat for $h$ below about
  $10^{-7}$ (checked in NumPy, not against the stored plots).
- 1.1: the sum of two `uint8` tensors is again `uint8` (not verified for
  MXNet).
- 2.7: at $\lambda = 0.05$ the 120 steps keep about 94% of the weights.
- 7.2: GoogLeNet would need "about 1.5 billion" multiply-adds at
  $224 \times 224$ (scaled from the 96-pixel count).
- 7.5: per-step shrinkage $10^{-4}$ (AdamW) against $2.5 \times 10^{-5}$
  (SGD), "which momentum multiplies by up to 10".
- 8.2: the corpus has "nearly 180,000 bytes, more than five for each of its
  words".
- 8.3: the best $\alpha$ of the exercise sweep leaves trigram perplexity
  above 2,500 (the writer ran the section's counting code).
- 1.1 (new paragraph on ranges): "Most experiments in this book deliberately
  do not fix the seed" is read off the code comments, not checked cell by
  cell.

**Cuts that touch the "ranges are deliberate" decision.**

- 2.3: "fewer than one run in a hundred" and "essentially always within
  $2 \times 10^{-3}$" became "would be rare" (finding SYN-3). Left as cut.
- 7.2: the paragraph on NiN at a constant rate of 0.01 (78 to 87%) was cut as
  a run the section does not show. "At 0.1 some runs diverge" was restored.

**Skippable marks.** 1.3 marks the eigenvalue part skippable on a first
reading although 2.7 (weight decay) and 4.4 cite it; 1.5 marks higher-order
derivatives skippable (no code in chapters 2 to 8 uses them).

## 2. Length

Part 1 is about 1% shorter (roughly 2,700 of 245,000 prose words). Repetition
was removed, but most sections spent the savings on content the review asked
for. Sections that grew: 1.6 probability (+192), 4.3 backprop (+287; cheapest
cut is the K-layer paragraph, about 115 words), 2.1 (+64), 8.1 (+73), 8.4
(+43), 8.6 (+27). Seven of eight introductions are slightly over 400 words.
Real shortening needs code and experiment cuts (section 4 below).

## 3. Slides are out of step with the bodies

The writers were not allowed to edit slide sections. Almost every section log
ends with the slide statements that now disagree with the body. Re-syncing is
mechanical and should be the next step; examples: 2.5 (factor of 2 restated on
five slides), 4.4 ("decides whether a deep network trains at all"), 4.6
(co-adaptation "measured"), 5.5 (full Adam update, "0.5 to 6"), 6.6 and the
chapter 7 slides (ImageNet numbers and titles the body dropped), 7.2 ("The
Dense Head Adds Little Accuracy"), 7.6 (block "starts as the identity"), 8.7
("text that repeats"). The chapter 3 pilot list is in `needs-dev-box.md`.
The conv-layer deck still shows `field_visual`, which now lives in the
chapter 6 introduction.

## 4. Deferred: needs code, cells, or figures

No code changed in this pass (laptop, no GPUs). The deferred findings are
listed per section in the logs; the main groups:

- **Experiments that show less than the section needs:** 2.7 weight-decay
  headline run (WD-1), 4.4 no network is trained (NS-1), 4.6 dropout tie
  (DO-1, DO-2), 4.2 per-class errors (IMPL-1), 5.5 bit-for-bit resume, 7.7
  mini-roadmap (CNX-9), 7.8 sampled population (DSP-5), 7.4 equal-arithmetic
  comparison (EFF-6), 6.5 global pooling cell and figure (POOL-5, POOL-8).
- **Reordering or removing cells:** 1.6 (coin question answered 5,000 words
  after it is asked), 2.2 scaffolding before the first run, 2.4 the bare loop
  never runs (SCR-1), 3 middle sections, 4.1 activation functions, 4.3 the
  scalar-graph cells that repeat autograd, 5.1 and 5.6 placement tour, 8.3
  (LM-S1), 8.4 concatenation check.
- **Plots and figures:** 2.6 three-panel plot (GEN-5), 8.1 rollout plot,
  learned-feature pictures for chapters 3, 4, 6, stronger contrast in the
  stride figure (needs an off-palette colour).
- **Re-capture items** in `needs-dev-box.md`: 1.6 coin sample, JAX outlier
  fold in 4.7, dropout ahead in all four stored runs.

## 5. Deferred: chapter-wide decisions

- **Notation:** $\hat{y}$ and $\bar{o}$ (chapter 3); $\mathbf{w}^*$ with two
  then three entries (2.3, SYN-1); $L$ for the minibatch loss (2.4, SCR-3);
  per-example $\nabla$ against minibatch $\partial J/\partial \mathbf{Z}$
  (4.3 and 4.4, BP-11); per-side padding $p$ (6.3, PAD-3); output-channel
  index $d$ against $o$ and channels-last against channels-first (6.1, 6.4);
  $b_t$, $g$ (8.6, BPTT-2); RNN notation (8.4, RNN-1; 8.3, LM-2).
- **Terms:** "validation" against "test" accuracy in chapter 7 (CNX-12, EFF
  exercises) and 2.6; "branch" against "body" in chapter 5; "treatments"
  against "choices of gain" in 5.3 exercises.
- **Titles:** 6.6 (LEN-1, quoted in seven slide tables), 7.7 (CNX-11).
- **Lazy layers** (MC-9): 5.1 now gives a verdict scoped to that section while
  5.4, 5.6, 5.7 still use `LazyLinear`.
- **Exercises:** left untouched except for errors; some now overlap the body
  (2.7 exercises 3(a) and 5(c), 4.3 exercise 3(a), 8.1 exercise 1(b)).
- **Pixel shuffling** is now proposed in both the chapter 4 closing and the
  LeNet ending; keep one or let the second cite the first.
- **Preface** still describes the old three-part plan (deferred earlier).

## 6. Not done

- The book was not rendered after the pass. `gmake html` on staging is the
  first full check of the result.
- No multi-reader consistency pass across chapters was made after the
  parallel writers finished, beyond the cross-section checks recorded in the
  commit history (cost argument in chapter 1, factor 1/2 in chapter 2, memory
  figures in chapter 5, clipping and temperature in chapter 8).
