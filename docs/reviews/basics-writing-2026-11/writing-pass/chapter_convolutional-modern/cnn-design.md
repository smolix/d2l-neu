# Writing pass: `chapter_convolutional-modern/cnn-design.md` (7.8, last section of the chapter)

Prose only; no code cell, label, figure, heading, exercise, or slide changed.
Checker: "OK: only prose changed". `tools/lint_source.py`: clean.

## Words

Checker count (prose incl. exercises): 4240 before, 4211 after (-29).
Body prose without exercises: about 3,360 before, 3,300 after. The cut in the
lab-notebook paragraphs (about 330 words to about 190) is partly offset by a
new three-sentence chapter ending (about 75 words) and by sentences that say
plainly what the section does not do.

## Before and after

1. Before: "Trying every combination is out of the question. With only two
   values for each of the 16 numbers that describe the networks of this
   section, there are $2^{16} = 65{,}536$ variants, each needing its own
   training run."
   After: "Could we not simply try every combination? Each stage of the
   networks below is described by four numbers, and with four stages and only
   two values for each number there are already $2^{16} = 65{,}536$ variants,
   each needing its own training run."
2. Before: "We build the template, read the evidence for each rule, and train
   a small RegNet on Fashion-MNIST."
   After: "We build the template and train a small RegNet on Fashion-MNIST.
   The populations themselves are beyond a notebook, since each of the
   paper's comparisons trains 500 networks on ImageNet, so we read its
   evidence instead of reproducing it."
3. Before: "Optimizing with such cheap approximations of the objective is
   called *multi-fidelity optimization* :cite:`forrester2007multi`."
   After: "This is why 10 epochs per network will suffice below." And after
   the list: "None of the four is obvious, and the second and third are
   rather bold; the exercises test them at small scale."
4. Before: "Requiring widths to increase across stages, $c_{i+1} \geq c_i$,
   lowers the mean error of the sampled networks from 49.4% to 43.2%, while
   equal or decreasing widths raise it to 56.1% and 63.2%."
   After: "The constraint that matters is on the widths. Requiring them to
   increase across stages, $c_{i+1} \geq c_i$, lowers the mean error of the
   sampled networks by six points, to about 43%, while equal or decreasing
   widths raise it to 56% and 63%."
5. Before: "A further analysis of the resulting space finds that its best
   networks use $k = 1$: the block does not narrow its channels at all."
   After: "The paper reports a further finding in the resulting space, and it
   runs against the practice of ResNet-50 and ResNeXt-50: the best networks
   use $k = 1$, so the block does not narrow its channels at all."
6. Before: "A network assembled from a few rules and four numbers per stage
   thus matches the hand-designed one with about a third of its multiply-adds
   and less than 5% of its 11.2 million parameters. The comparison is not
   controlled in every respect: the two networks differ in depth, width, and
   block, and the rules did not come from Fashion-MNIST."
   After: "What does the tie show? It cannot show that the rules produce
   better networks: the two networks differ in depth, width, and block all at
   once, and 94% will turn out to be the ceiling of this dataset. It does
   show that ResNet-18 is far larger than the task requires. A network that
   obeys the rules reaches the same accuracy with about a third of the
   multiply-adds."
7. Before: "so widths and depths increase."
   After: "We picked these numbers by hand so that widths and depths
   increase; nothing was sampled or searched."
8. Before: "RegNetY adds one component to every block of RegNetX: a
   *squeeze-and-excitation* (SE) gate. [...] EfficientNet also places an SE
   gate in every block."
   After: "[...] None of the networks above uses it. We build it anyway,
   since it is small and common (EfficientNet also places one in every
   block), and the exercises add it to RegNetX32."

## Repetition removed

Owners were determined by chapter order (alexnet, blocks, batch-norm, resnet,
efficient-convnets, training-recipes, convnext, cnn-design); this section is
last and owns none of the shared facts.

- Stem, body, head: defined in `blocks.md` ("Stem, Body, and Head"). Now
  "the stem, body, and head of :numref:`sec_googlenet`"; the NiN/GoogLeNet
  history of the head is gone. Stages and blocks stay, since the section
  needs them.
- Architecture search and compound scaling (owner: efficient-convnets): three
  sentences with method names reduced to one sentence each of contrast; the
  three method names and "the first NAS methods trained thousands of
  candidates" are gone.
- He initialization (owner: ch. 4, restated in resnet): the variance formula
  dropped; one clause with both references.
- Momentum (owner: efficient-convnets): the "up to $1/(1-0.9) = 10$ times
  longer" explanation reduced to "whose steps follow a running sum of recent
  gradients" with references to `sec_efficient_cnns` and `chap_optimization`.
- Warmup and cosine schedule (owner: alexnet): one clause.
- ResNet-18's stem and stages (owner: resnet): the layer-by-layer restatement
  replaced by a reference.
- "Multiply-adds are not time" (owner: efficient-convnets): the 150-word
  timing paragraph with its own explanation reduced to the measurement plus
  one clause with a reference.
- The paper's EfficientNet claim was stated under "Design Principles" and
  again, qualified, in the timing paragraph. It is now stated once, with the
  qualification.
- The scope caveat ("specific to the paper's block, budget, and protocol")
  appeared three times (after the figure, in the result discussion, in the
  Summary). It is now stated once, where the evidence is read; the Summary
  keeps only the open question.
- Parameter counts: "0.52 million", "11.2 million", "less than 5%" each given
  once in the body.

## Review findings

Applied (prose): DSP-1; DSP-2; DSP-4 (term and citation cut, assumption 2
tied to the 10 epochs of the figure); DSP-7; DSP-8 (claim changed to what the
run shows; "four numbers per stage" removed from the claim; the hand choice
of RegNetX32 stated); DSP-9 (weight-decay sweep, larger models, zero
initialization, Nesterov, 30 epochs, augmentation reduced to four sentences;
the run-to-run range is kept, per the author); DSP-10 ("kernel" no longer
used for GPU routines; fused-kernel speculation and the layer timings
removed); DSP-12; DSP-13; structural finding 3; and, for the missing chapter
ending, a three-sentence closing paragraph after the Summary.

Partly applied: DSP-5 and structural finding 1 (no experiment added; the text
no longer says the code is built "to sample networks from the space and train
them", and the opening says that the populations are read from the paper).
DSP-6 (one clause says what a geometric grid is and ties it to the doubling
widths of the table; the worked instance is not added because it is the
answer to exercise 2(a), and the section should not grow). DSP-11 (the gate
stays where its code cell is; the text now says that no network here uses it
and that the exercises do).

Deferred:
- DSP-3 (symbols $g$, $k$, $c_i$, $w$, $r$, $d$): needs one decision for the
  chapter, and the figure and the code use the current symbols.
- DSP-5 proper (a sampled design-space experiment): new cells and outputs.
- DSP-8, second half (compare against a rule-breaking network): new runs.
- DSP-11 (move SE to efficient-convnets or train a RegNetY32): moves code
  between files, or new runs.
- Structural finding 6 (move the section after efficient-convnets): chapter
  order.
- Exercise remarks ("VioNet" name, the "few tenths of a point" claim, the
  48-run exercise, per-framework discussion links): exercises not touched.

## Factual claims added or changed

- "less than 5% as many" parameters: 519,658 / 11,175,370 = 4.65% (already in
  the text as "less than 5%").
- Mean errors rounded: "near 49%" (paper: 49.0, 49.2, 49.4), "by six points,
  to about 43%" (49.4 to 43.2), "56% and 63%" (56.1, 63.2), "another half
  point" (43.2 to 42.7).
- New: successive values of a geometric grid "differ by a constant factor, as
  the widths of the hand-designed networks in the table do". True by the
  definition $w_0 w_m^s$ in exercise 2; the table's factor is 2.
- New: $k = 1$ "runs against the practice of ResNet-50 and ResNeXt-50" (both
  are bottleneck designs in the section's own table).
- New: "We picked these numbers by hand ...; nothing was sampled or
  searched." Taken from the review (DSP-8); please confirm.
- New: "32 layers, hence the name."
- New: the optimizer is "the one under which [the paper] found their rules"
  (the old text said this two paragraphs later).
- Changed: grouped convolutions run "well below the arithmetic rate of dense
  ones", "like depthwise ones" (old text: "a quarter to a tenth"; the
  comparison with depthwise convolutions rests on the measurement in
  efficient-convnets).
- Changed: the larger RegNetX models reach 93% to 94% "with the
  zero-initialized block scales ..., without which some runs of these deeper
  networks start badly" (old text gave more detail; nothing new is claimed).
- New, in the chapter ending: ResNet-18 at about 94% "leaves the designs that
  followed it little room to show an advantage in accuracy". This agrees with
  the chapter review's finding 1 but speaks for sections I did not edit.
- Stored outputs checked in all four frameworks: 519,658 and 11,175,370
  parameters; accuracies 0.937 to 0.940 for both networks. "About 94%" holds.

## Slides that now differ from the body

Nothing contradicts; the slides carry detail that the body no longer has.

- "Larger RegNets Gain Nothing": multiply-adds of 72 to 293M, "no gain from
  30 epochs", "crops and flips lower every network".
- "A Third of the Multiply-Adds, Yet Slower Steps on a GPU": "1/4 to 1/10 of
  the arithmetic rate", "about 1.6× as many small kernels per step", "grouped
  kernels at the dense rate, fused with what follows"; these bullets also use
  "kernels" for GPU routines.
- "Growing Widths Cut the Mean Error from 49.4% to 43.2%": the table keeps
  the paper's tenths, which the body now rounds.
- "RegNetX32 (four numbers per stage)" and the goal "build a RegNet instance
  from four numbers per stage": the body now says the numbers were picked by
  hand and no longer presents the count as the point.
- "RegNet" slide footnote "beats EfficientNet at equal multiply-adds, up to
  5× faster on GPUs": the body says "match or beat" and states the claim only
  with its qualification.
