# ConvNeXt: A ConvNet for the 2020s

Source: `chapter_convolutional-modern/convnext.md`. About 2,900 words of prose, 860 words of exercises, 8 exercises.

## Verdict

The section has the right idea about its subject, stated in its best sentence: "ConvNeXt is a controlled experiment wearing an architecture's name." Read as a lesson in attribution (which of twelve changes produced a five-point gap?) it is worth its place, and the roadmap table is a strong central object. Three things hold it back. It is motivated by a comparison with a model the reader will not meet for four chapters, so the stakes are borrowed. Its first half is a retelling of the paper, row by row, in which the reader computes nothing. And its own experiment ends with the featured architecture two points behind a plain ResNet-18, after which the prose explains why that does not count. The most important fix: let the reader do a small roadmap. Apply three or four of the changes one at a time to the ResNet-18 they already have and tabulate the result, so that the section teaches ablation by doing it, and the Fashion-MNIST outcome becomes the finding and not an embarrassment.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 3 |
| Clarity of thought | 4 |
| Pedagogy | 3 |
| Language | 4 |
| Ease and enjoyment | 3 |

## What works

- The framing question: "Did attention cause the gap? If it did, convolutional networks were obsolete." and then "Which of these differences carried the improvement?" This is how to motivate an architecture section.
- The roadmap table with cumulative accuracies, and the accounting that follows it: "Of the 5.9 points that separate the 2015 ResNet-50 from ConvNeXt, 2.7 come from training and 3.2 from the architecture." It cashes in the previous section at once.
- The patchify stem explained as something the reader already knows: slicing into patches and multiplying by one matrix "is a convolution whose kernel size equals its stride".
- The mixing argument: a depthwise convolution "mixes positions but not channels", a $1 \times 1$ convolution "mixes channels at each position but not positions", "as a transformer block separates attention from its MLP". This is the transferable idea of the block and it is said cleanly.
- The kernel-size cost argument ("multiplies a small term by $49/9 \approx 5$") and the honest dip in the roadmap (moving the depthwise convolution "costs 0.7 points", the larger kernel "recovers them").
- "The roadmap removed more than it added, so the block is short."
- "Because a single wrongly sized layer changes it, the count is a sensitive check of any reimplementation, ours included."
- The restraint of the conclusion: "The experiment does not show that convolutions are better".
- Exercise 3, "Normalization axes", is a model exercise.

## Structural findings

1. **Dependence on chapters 10 and 11.** The opening defines vision transformers, attention and Swin in six sentences, and every later subsection justifies a change by what "a transformer block" or "Swin" does. The one-sentence definition of attention is good, but a reader of the Basics part cannot judge any of these comparisons. October flagged this; it is still open. Two remedies: (a) recast the motivation so it does not need the transformer (a 2015 ResNet at 76% and a 2022 convnet at 82%: where do six points come from?), keeping the transformer as the historical reason in one paragraph; or (b) move the section to follow the vision transformer, where every comparison can be checked by the reader.
2. **The roadmap is narrated, not experienced.** About 900 words walk through twelve rows of someone else's table. Each row is explained well, but the reader's role is to believe. See CNX-9 for the remedy.
3. **The experiment contradicts the section and is then explained away.** See CNX-8.
4. **"Limits and Open Problems" is a second survey.** Masked autoencoders, ConvNeXt V2, sparse convolutions, Global Response Normalization, $31 \times 31$ kernels and deformable convolutions arrive in two paragraphs after the lesson is over. See CNX-10.
5. **The inverted bottleneck is explained twice in the chapter and motivated in neither place.** Here: "the economics reverse ... the block can keep a thin stream between blocks and expand it inside". As in the efficient-convnets section (EFF-3), the reason to want a wide interior is missing.

## Passage-level findings

**CNX-1** (moderate, explanation). Opening paragraph.
> layer lets every patch take a weighted average of the features of other

The definition is serviceable, but the paragraph then adds Swin, local $7 \times 7$ windows, stage-wise downsampling and two accuracies before the question arrives in its last two sentences. Lead with the question (a network without convolutions overtook the best convnets; was it the new layer?) and give the description of the transformer after it, in three sentences.

**CNX-2** (minor, thought). Under "The Modernization Roadmap".
> points that separate the 2015 ResNet-50 from ConvNeXt, 2.7 come from

The split into 2.7 and 3.2 depends on the order in which the changes were applied, as the first exercise acknowledges. One clause ("in this order of application") keeps the reader from taking the split as a property of the changes themselves.

**CNX-3** (moderate, explanation). Under "Depthwise Convolutions and the Inverted Bottleneck".
> depthwise, the economics reverse, as :numref:`sec_efficient_cnns` showed for

"The economics reverse" names a conclusion without the argument. State it: once the spatial convolution is depthwise its cost grows in proportion to the width and no longer to its square, so running it on the wide tensor is cheap, and a wider interior gives the block more spatial filters. The efficient-convnets section did not show this either; it asserted it in the same words ("the trade reverses").

**CNX-4** (minor, explanation). Under "Depthwise Convolutions and the Inverted Bottleneck".
> as much as three stacked $3 \times 3$ layers (:eqref:`eq_receptive_field`).

A bare equation reference into chapter 6. The fact is already stated in the sentence, so the reference can become "(:numref:`sec_padding`)" or go.

**CNX-5** (moderate, explanation). Under "Micro Design".
> each. A transformer block has one activation, inside its MLP, and one

"One normalization per part" is unclear (the two parts were named sixty lines earlier). More important, the paragraph reports that removing two of three activations adds 0.7 points, "more than any other change in this group", and offers no reason beyond imitation of the transformer. If the authors of the paper give none, say so plainly: this is an empirical finding without an accepted explanation. The reader should not be left to suspect they missed one. The closing remark ("here fewer such layers, one per block, train better than three") is the right kind of sentence.

**CNX-6** (minor, pedagogy). Caption of the block figure.
> learnable per-channel layer scale $\boldsymbol{\gamma}$ multiplies the branch before the addition

The figure and caption show layer scale before the text defines it (it is introduced under "The ConvNeXt Block"). Move the definition up or the figure down. $\boldsymbol{\gamma}$ is also the batch-norm scale throughout the chapter and a compound-scaling constant in the efficient-convnets section; in a block that has just replaced batch normalization, a different letter would spare the reader a doubt.

**CNX-7** (minor, thought). Under "A Parameter-Matched Comparison".
> stochastic-depth rate of ConvNeXt to zero because the ResNet control has no

The previous section built `StochasticResidual` for exactly this ResNet block, so the stated reason is not true of the book's own code. Either use stochastic depth in both networks or say that it is switched off to keep the comparison to the architecture.

**CNX-8** (major, pedagogy). Under "A Parameter-Matched Comparison".
> The compact ResNet-18 is ahead by about two points in every run, and both

The reader has spent a section on why ConvNeXt is better and now watches it lose. The paragraph then states what the experiment does not show four times over ("The result is specific to this setting", "It does not match the running time", "do not transfer automatically", "it does not rank convolutional architectures in general"). The stored curves also weaken the comparison: after 20 epochs ConvNeXt's validation accuracy is still creeping up and its training loss still falling, so "level off ... as the learning rate decays to zero" describes the schedule, not convergence, for a design tuned for 300 epochs. And "It does not match the running time" is dropped without a number or a direction. A null result can teach, but only if the section is built around it. Two options: frame the experiment from the start as a test of transfer ("every row of the roadmap was chosen on ImageNet; we now check which survive on a small task"), or replace it by the mini-roadmap of CNX-9.

**CNX-9** (major, pedagogy). Whole section.
> We read the sequence of
> changes, build the network, and train a small ConvNeXt on Fashion-MNIST

"We read" is the problem. The method the section admires is the one-change-at-a-time ablation, and the reader never performs one. Proposal: start from the compact ResNet-18 under the modern recipe and apply four changes cumulatively (patchify stem; depthwise convolution with more width; one activation and one normalization per block; batch normalization to layer normalization), one row each, five training runs in all, and print the reader's own roadmap table beside the paper's. Whatever the numbers are, the reader learns how such a table is made, how small the steps are next to run-to-run noise, and which ImageNet findings survive. The existing final comparison is then the last row.

**CNX-10** (moderate, pedagogy). Under "Limits and Open Problems".
> convolutions that skip the masked regions, and in doing so finds that many

Six new named ideas in two paragraphs, none of which the reader can use, one of them defined only in an exercise. This fails the writing guide's teachable-core tests. Reduce to three sentences (pretraining without labels favours patch-based models; how far a convolution needs to see is unsettled) and move the names to Further Reading.

**CNX-11** (minor, language). Title.
> # ConvNeXt: A ConvNet for the 2020s

The subtitle is the paper's title and its slogan. A subject heading would say what the section teaches, for example "ConvNeXt: Modernizing a ResNet Step by Step".

**CNX-12** (minor, language). This section and the next say "validation accuracy"; the two before say "test accuracy" for the same quantity (`trainer.history['val_acc']`). The class name `ModernRecipe` is defined here a second time with a different structure (a mixin) from the class of the same name in the previous section.

## Exercises

Eight exercises with a good ramp: two counting exercises, the excellent "Normalization axes", then code. "Roadmap ablations" part (a) is a well-built calculation; part (b) asks the reader to design runs without running them, which is fine. "Channel redundancy" depends on the GRN exercise and on a paper the text only gestures at; cut it together with the "Limits" material, or keep one of the two. "Recipe downgrade" completes a two-by-two table and is the right hard exercise.

## Still open from October

"ConvNeXt is explained through Transformers, four chapters ahead" is still open: the opening now defines attention in a sentence, but every design choice is still justified by reference to Swin. October's keeper ("Which of these differences ... carried the improvement?") and the "wearing an architecture's name" line are intact and should stay.
