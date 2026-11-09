# Convolutional Network Design Spaces

Source: `chapter_convolutional-modern/cnn-design.md`. About 3,400 words of prose, 1,040 words of exercises, 8 exercises.

## Verdict

The weakest of the four sections, and the one that most needs a decision. Its question is good ("Where do these numbers come from, and are there rules for choosing them?") and its idea is worth teaching: judge a design rule by the error distribution of a population of networks, not by one winner. But the reader never sees a population. The section builds code "to sample networks from the space and train them", then samples nothing; the evidence is a figure copied from the paper; and the one experiment trains a single hand-picked network that ties ResNet-18 at the 94% ceiling of the dataset. Around that experiment sit about 330 words of results from runs that are not in the notebook, written as a lab report. The section closes with an unrelated component (squeeze-and-excitation) that no experiment uses, and the chapter ends there. The most important fix: run a small design-space experiment in the notebook (two populations of a dozen networks each, a few epochs, two empirical CDFs), or cut the section to half its length and say plainly that it reports a method.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 3 |
| Clarity of thought | 3 |
| Pedagogy | 2 |
| Language | 3 |
| Ease and enjoyment | 2 |

## What works

- The opening names a gap the reader has had since VGG: the depths $(2,2,2,2)$, $(3,4,6,3)$, $(3,3,9,3)$ and "ResNet's depths come with no stated rule, and ConvNeXt-T's were copied from Swin and checked once." The observation that "a single good network, however it was found, does not say which of its choices mattered" is the thesis in one sentence.
- The table of hand-designed stages with its two-line reading: "Their widths double from stage to stage in every row. Their depths follow no visible rule."
- The stem, body and head template gathers what the chapter has built and gives it names. The AnyNet figure is clear.
- The empirical CDF is defined in words before the formula and read afterwards, with the right caution: "This compares populations, not best networks: the best network of the two spaces may be the same."
- "there are many good needles in the haystack" is brief, concrete and true.
- The count $(16 \cdot 128 \cdot 3 \cdot 6)^4 \approx 2 \times 10^{18}$ makes the size of the space felt.
- Exercises 4 ("Design spaces versus architecture search") and 7 ("Proxy reliability") test the method's assumptions and are among the best in the chapter.

## Structural findings

1. **The central method is never demonstrated.** See DSP-5. This is the gap between a section that teaches a way of thinking and one that summarizes a paper.
2. **The experiment does not support the claim drawn from it.** See DSP-8.
3. **Lab-notebook prose.** Three paragraphs after the result report optimizer swaps, larger models, zero-initialized scales, Nesterov momentum, 30-epoch runs, augmentation and layer-by-layer GPU timings, none of which the reader can see. See DSP-9 and DSP-10.
4. **Squeeze-and-excitation is an orphan.** It sits as the last subsection of "RegNet", after the experiment and its discussion, is used in no network, and leaves the section (and the chapter) ending on "512 weights plus 40 biases". See DSP-11.
5. **Notation.** The section uses the paper's symbols beside the book's, and several collide with the rest of the chapter. See DSP-3.
6. **Position in the chapter.** This is a 2020 paper about ResNeXt blocks placed after the 2022 ConvNeXt section, and it ends the chapter without a closing thought on what the chapter established. The efficient-convnets section already raises its two foils (architecture search, compound scaling). It would sit more naturally directly after that section, leaving recipes and ConvNeXt, which looks ahead to transformers, to close the chapter.

## Passage-level findings

**DSP-1** (minor, explanation). Opening paragraph.
> combination is out of the question. With only two values for each of the 16

"The 16 numbers that describe the networks of this section" are not yet known to the reader; four per stage is explained 110 lines later. Say "four numbers for each of four stages".

**DSP-2** (moderate, explanation). Second opening paragraph.
> reinforcement learning, evolution, or a gradient-based relaxation

Architecture search is disposed of in two sentences with three method names the reader cannot unpack, and compound scaling in one more. The paragraph's job is the contrast (search returns one network, a design space returns a rule). Make that contrast with one sentence about search and drop the method names.

**DSP-3** (moderate, language, notation). Under "Stage Parameters of Hand-Designed Networks".
> channels per group. The paper's $g_i$ counts channels per group, whereas $g$

A sentence that has to reconcile two meanings of one symbol signals a clash. There are several here. $g$ counted groups in the ResNeXt section and counts channels per group now. $k$ is the kernel size in every other section of the chapter ("$k \times k$") and the bottleneck ratio here, while the code calls its inverse `bot_mul`. $c_i$ is the width of stage $i$ here and $c_\textrm{i}$ is the number of input channels in the efficient-convnets section; the two are nearly indistinguishable in print. Width is $c_i$ in the text, $w_0$, $w_a$, $u_j$ "in the paper's symbols" in the linear rule, and $w_i$ in the reproduced figure, which the caption has to translate. $r$ is the input resolution in the figure caption and the reduction ratio of the gate; $d$ is the stage depth here and the depth multiplier in the compound-scaling formula. Choose the book's symbols once (number of groups $g$ as before, bottleneck ratio $b$, widths $c$) and translate the paper into them, including the linear rule.

**DSP-4** (minor, pedagogy). Under "Comparing Design Spaces".
> approximations of the objective is called *multi-fidelity optimization*

A term and a citation that the section never uses again. Cut. The four assumptions are a paraphrase of the paper's list; they would land better if each were tied to what the reader will see ("assumption 2 is why 10 epochs suffice in the figure below").

**DSP-5** (major, pedagogy). Under "Comparing Design Spaces".
> train each for 10 epochs on
> ImageNet, and compare the CDFs in :numref:`fig_regnet-fig`. Each step adds

The whole evidence for the section's rules is this paragraph and a reproduced figure with seven mean errors read out in prose. The reader has just built `AnyNet` for the stated purpose of sampling networks, and the empirical CDF takes three lines to plot. Run the experiment at the reader's scale: sample, say, 12 two-stage networks with $c_1 \le c_2$ and 12 with $c_1 \ge c_2$ at a matched parameter budget, train each for two epochs on the 10,000-image subset of the previous section, and plot the two CDFs. It costs about as much compute as the current two 10-epoch runs, it is the only experiment in the chapter where the unit of evidence is a population, and it would make exercises 5 and 8 extensions of something seen and not first encounters. If Fashion-MNIST cannot separate the two populations, that is worth knowing before the section is kept. The reproduced figure can stay as the ImageNet-scale counterpart; its legends cover parts of the curves in the lower panels and it uses $w_i$ for width, so a redrawn version in the house style would also help.

**DSP-6** (moderate, explanation). Under "Design Principles".
> stage share one width, RegNet snaps each $u_j$ to a nearby value on a

Why a geometric grid, and how do stages emerge? The paragraph defers to an exercise. A five-line worked instance would do more than the formula: with $w_0 = 24$ and slope 36, blocks 0 to 12 have target widths 24, 60, 97, ..., and snapping to the grid 24, 60, 149, 371 groups them into stages of 1, 1, 4 and 7 blocks. (These are the parameters of the exercise; they reproduce the paper's RegNetX-200MF.) The reader then sees the staircase the text mentions.

**DSP-7** (minor, thought). Under "Design Principles".
> that RegNet models match or beat EfficientNet at equal multiply-adds under

A headline claim from the paper with "up to five times faster on GPUs", which the section's own timing paragraph later has to qualify at length. State it once, with the qualification, where the timing is discussed, or leave it out.

**DSP-8** (major, thought). Under "RegNetX32 on Fashion-MNIST".
> from a few rules and four numbers per stage thus matches the hand-designed

Three problems. First, `RegNetX32` is itself hand-designed: its depths $(4, 6)$ and widths $(64, 160)$ were picked by the authors to satisfy two inequalities, the linear width rule is not used, and it has two stages where the rules were derived for four. Second, "four numbers per stage" is the unconstrained AnyNet; the point of RegNet is that the count shrinks. Third, the text says two paragraphs later that "the dataset, not the architecture, limits the accuracy to about 94%", so a tie at 94% cannot show that the rules produce good networks; any reasonable network ties. What the experiment does show is that the same accuracy is reached with 5% of the parameters and a third of the arithmetic. Claim that, and compare against a network that breaks the rules (decreasing widths, the first variant of exercise 5) so that the rules themselves are tested.

**DSP-9** (major, pedagogy). Under "RegNetX32 on Fashion-MNIST".
> and 94%, provided the last batch normalization of each block starts with a

This paragraph and its neighbours report experiments the reader cannot run from the page: three larger RegNets, a zero-initialized scale, "Nesterov's variant of momentum, the default in the paper's code", runs that "start badly", ResNet-18 at 30 epochs, and augmentation "of every network we tried". Earlier, the choice of weight decay is narrated the same way ("of the decays $5 \times 10^{-5}$, $5 \times 10^{-4}$, $10^{-3}$, and $2 \times 10^{-3}$, it gives RegNetX32 its best accuracy and ResNet-18 one as good as any other"), and the result paragraph speaks of "three or more runs in each framework". This is the authors' record of how the notebook was tuned. Keep two sentences: both networks are near the ceiling of this dataset, so larger members of the family gain nothing; the result needs momentum and weight decay, as in the paper. Move the optimizer finding to exercise 6, which already covers it.

**DSP-10** (moderate, explanation and language). Under "RegNetX32 on Fashion-MNIST".
> because the GPU library runs it with generic kernels that do not use the

In a chapter where "kernel" has meant a convolution's weights on every page, this paragraph uses it for GPU routines three times ("generic kernels", "small kernels per step", "Kernels that ran grouped convolutions at the rate of dense ones"). The paragraph is also the third telling in the chapter that multiply-adds do not predict time (after the efficient-convnets measurement and the remark in the ConvNeXt comparison), with new unverifiable numbers ("a quarter to a tenth of the arithmetic rate", "about 1.6 times as many"). Replace by two sentences that cite the earlier measurement: grouped convolutions, like depthwise ones, do little arithmetic per value read, so an epoch of RegNetX32 takes one and a half to two times as long as ResNet-18's despite a third of the multiply-adds.

**DSP-11** (moderate, pedagogy). Under "Squeeze-and-Excitation Gates".
> RegNetY adds one component to every block of RegNetX: a

The gate is a useful idea, explained compactly ("lets the network reweight whole channels based on the entire image"), but it arrives after the experiment, belongs to no argument of this section, and is never trained. It is a component of efficient networks more than of design spaces. Move it to the efficient-convnets section (where EfficientNet, which uses it, is discussed) with a small figure, or train a RegNetY32 beside RegNetX32 and report the difference, or leave it to exercise 3.

**DSP-12** (minor, explanation). Under "RegNetX32 on Fashion-MNIST".
> exponentially weighted sum of recent gradients, so that its steps are up to

The second near-verbatim explanation of momentum in the chapter, inside a ten-line sentence that also covers the schedule, the paper's optimizer and weight decay. See EFF-4: explain once, restate in a clause.

**DSP-13** (minor, thought). After "Comparing Design Spaces".
> The evidence is specific to the paper's block, budget, and training

A fair caveat, but it is made again in the result discussion ("the rules did not come from Fashion-MNIST") and a third time in the Summary. Once is enough, at the end.

## Exercises

Eight exercises; the two conceptual ones (2 and 4) and "Proxy reliability" are strong. "Quantized linear widths" carries material the text should contain (DSP-6); keep it as practice once the text has a worked instance. "VioNet" needs a plainer name ("Breaking the rules") and its part (a) contains a result the reader has not been shown how to obtain ("runs of one network differ by a few tenths of a point"). "A ConvNeXt design space" asks for 48 training runs and is the experiment the section itself should model at smaller scale (DSP-5). The per-framework discussion links at the end differ from the single link used by the three sections before it.

## Still open from October

"Training runs without a readout (`cnn-design.md:427`)" is fixed: both runs now print parameters and accuracy and the prose reads them. The catalogue complaint has changed form: the list of architectures is gone, and a list of unreported experiments has taken its place.
