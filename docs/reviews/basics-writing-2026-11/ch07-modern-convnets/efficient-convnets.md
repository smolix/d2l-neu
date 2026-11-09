# Efficient Convnets

Source: `chapter_convolutional-modern/efficient-convnets.md`. About 3,400 words of prose, 940 words of exercises, 9 exercises.

## Verdict

A good section with a real question (how to make a convnet ten times cheaper, and what it costs) and one finding a reader will remember: 7.7 times fewer multiply-adds gives only 2 to 3.5 times less time on a GPU. The first third (cost ratio, MobileNet, the measured table) and the batch-norm folding algebra are close to the bar. The section falls short in two places. Its only training experiment compares the two networks at equal parameters, where the efficient network loses by two points, and leaves the comparison that would vindicate it (equal arithmetic) to an exercise. And the middle part, "Scaling and Searching", is a prose survey of four papers with no example, no code and no number the reader computes. The single most important fix: put the equal-arithmetic dense network into the table as a third row, so that the section's own experiment shows what a separable network is for.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought | 3 |
| Pedagogy | 3 |
| Language | 4 |
| Ease and enjoyment | 3 |

## What works

- The opening. The phone that classifies every camera frame, "460 billion multiply-adds per second", and 550 MB "in memory next to every other app" make the budget concrete before any mechanism appears. Keep it as written.
- The cost ratio is restated on the page, not only cited from chapter 6, and it is read in numbers: "the separable layer is 8.4 times cheaper, and no number of output channels makes it more than 9 times cheaper." This is the Cover and Thomas rhythm (formula, number, plain sentence).
- The prediction is checked against the built network: 4,663,296 dense weights against 530,528 separable ones, "8.8 times". I verified the arithmetic.
- "Multiply-adds count arithmetic, not time." followed by a measurement and a mechanism: a dense convolution "performs $9 c_\textrm{o}$ multiply-adds for every input value it reads", a depthwise one performs 9. This is the best explanatory paragraph in the section.
- "Fusing the Branches": three named steps, a displayed fold, a numerical check that prepares the test honestly (running statistics moved off 0 and 1 so "every term of :eqref:`eq_bn_fold` matters"), and a result of order $10^{-6}$. The code is the explanation made exact.
- "The network that is trained has residual paths; the network that is deployed is a plain stack." One sentence that states the idea of re-parameterization.
- The quantization caveat (75.1% to 40.2%) is a true counterexample that fences the claim of equivalence, in six sentences.

## Structural findings

1. **Three ideas, one of them a survey.** The section announces "three ideas". Factorization is taught (formula, code, experiment). Re-parameterization is taught (algebra, code, check). "Scaling and Searching" is two paragraphs that name EfficientNet, neural architecture search, EfficientNetV2 and MobileNetV4, with one displayed formula and nothing the reader does. It reads as a literature paragraph between two lessons. Either give compound scaling one worked number and cut the rest to two sentences, or move all of it to the design-spaces section, where NAS and compound scaling are discussed again.
2. **The experiment answers a different question from the one the section asks.** See EFF-6. The reader finishes the measured part having seen the efficient network lose accuracy and gain less speed than promised. The favourable comparison is described in prose and delegated to an exercise.
3. **The inverted bottleneck is told, not shown.** It has its own heading, no figure, no code and no experiment, and ConvNeXt two sections later depends on it. See EFF-3.
4. **Optimizer and schedule are introduced inside an architecture subsection.** Momentum SGD is first explained in this chapter here, in a parenthesis in "A Mini-MobileNet". See EFF-4.

## Passage-level findings

**EFF-1** (minor, explanation). Under the opening.
> frames a second the phone would have to perform 460 billion multiply-adds per

The reader has no yardstick for whether 460 billion per second is a lot. One clause fixes it: give the order of magnitude a phone processor sustains, so that "cannot ship" is felt and not asserted.

**EFF-2** (minor, explanation). Under "MobileNet".
> *width multiplier* scales every channel count, so that one network design

The width multiplier is introduced here and then used only in exercises. Either say in one clause what halving it does to cost (about a quarter of the multiply-adds) or leave the term to the exercise that uses it.

**EFF-3** (major, explanation). Under "The Inverted Bottleneck".
> is depthwise, it is cheap even on wide tensors, and the trade reverses.

The paragraph says the wide spatial convolution is now affordable but never says why one would want it wide. The missing sentence: a depthwise convolution has one spatial filter per channel, so the number of filters the block can learn equals the width at which the depthwise layer runs; expanding first gives it six times as many. The paragraph then packs the expansion factor, the linear projection, the lost-information argument for dropping the ReLU, activation memory on a phone, and a forward pointer to ConvNeXt into one 19-line block with no picture. Add a two-panel figure (bottleneck beside inverted bottleneck, widths labelled, in the style of `arch-resnet-vs-convnext-block.svg`), split the paragraph in three (shape; linear projection; memory), and give the widths of one concrete block, for example 24 to 144 to 24.

**EFF-4** (major, pedagogy). Under "A Mini-MobileNet".
> which steps along an exponentially weighted sum of recent gradients instead of

This 19-line paragraph describes the architecture, introduces momentum with a forward reference to the optimization chapter, warns that results are not comparable with the ResNet section, restates the cosine schedule, reports that a constant rate "wobbles by one to three points", and ends on "The schedule is a built-in of the framework." Six jobs in one paragraph, three of them about training in a subsection about a network. Split it: one paragraph for the network, one short paragraph headed by what changes in training and why. The momentum explanation recurs almost word for word in the design-spaces section ("so that its steps are up to $1/(1 - 0.9) = 10$ times longer than plain SGD steps"). Explain momentum once for the chapter, in a place of its own (the training-recipes section is the natural home, or the AlexNet section where the schedule is defined), and restate it in one clause afterwards.

**EFF-5** (minor, language). Under "Arithmetic, Accuracy, and Time".
> parameters about nine times faster per pair of channel counts, so the dense

"Spend parameters about nine times faster per pair of channel counts" does not parse on first reading. Write: "A dense $3 \times 3$ layer between the same two widths has about nine times as many weights, so with the same budget the dense network stops at 128 channels where the separable one reaches 512."

**EFF-6** (major, pedagogy and thought). Under "Arithmetic, Accuracy, and Time".
> Matching the arithmetic instead shows what the separable network buys. With

The paragraph asserts what the equal-arithmetic comparison would show, gives the widths (11, 23, 46, 46) and then says "the exercises measure what that buys in accuracy". This is the experiment the section exists for. As it stands, the table shows MobileNet two points worse at 7.7 times less arithmetic, and the text needs a paragraph ("Part of the gap is ... Part is arithmetic itself ...") to explain that the comparison "matches parameters, not architectures". Train the narrow dense network in the same loop and add it as a third row. The table then reads in both directions: at equal parameters the dense network wins by two points and costs 7.7 times more; at equal arithmetic the separable network wins (if it does; if it does not, the section has a different and more interesting story to tell). The exercise "Equal arithmetic" then becomes redundant and can go.

**EFF-7** (minor, thought). Under "Arithmetic, Accuracy, and Time".
> proxy for time; how good a proxy depends on the hardware, which is the theme of

"Which is the theme of the rest of this section" is true of the EfficientNetV2 paragraph and of RepVGG's motivation, but compound scaling and the fusion algebra are not about it. Drop the clause or make it specific.

**EFF-8** (moderate, language, notation). Under "Scaling and Searching".
> d = \alpha^\phi, \qquad w = \beta^\phi, \qquad r = \gamma^\phi

Within this one section $w$ is the feature-map width ("an $h \times w$ output") and the width multiplier, and $\beta$, $\gamma$ are compound-scaling constants here and the batch-norm shift and scale 130 lines later in :eqref:`eq_bn_fold`. Rename the scaling constants (for example $a_d, a_w, a_r$) or drop the formula in favour of the numbers: "each step multiplies depth by 1.2, width by 1.1 and resolution by 1.15, which about doubles the cost, since $1.2 \cdot 1.1^2 \cdot 1.15^2 = 1.92$".

**EFF-9** (moderate, pedagogy). Under "Scaling and Searching".
> inverted bottleneck, a ConvNeXt-style block, and a plain feedforward block

The MobileNetV4 sentence describes a searchable block in terms of a ConvNeXt block the reader meets two sections later, and neural architecture search is defined in one clause ("an optimization over inverted-bottleneck blocks that rewards accuracy and penalizes multiply-adds") with no hint of how such an optimization runs. This is the list of systems the writing guide's section 4.3 warns about. Keep EfficientNetV2, because it confirms the reader's own measurement ("The measurement above shows the weakness of that objective" is a fine link), and cut MobileNetV4 to Further Reading.

**EFF-10** (minor, explanation). Under "Fusing the Branches".
> \mathbf{W}_o$ and $\hat{b}_o = 0.5 - 1 = -0.5$. The convolutions of

The numerical example sets $\gamma_o = 2$ and $\sigma_o^2 + \epsilon = 4$, so the scale factor is 1 and the kernel is unchanged. The example hides the half of the formula it should illustrate. Use $\sigma_o^2 + \epsilon = 16$: the kernel is halved and $\hat{b}_o = 0.5 - 0.5 = 0$.

**EFF-11** (minor, thought). Under "The RepVGG Block".
> ResNet (:numref:`sec_resnet`) showed that networks with several paths per

The claim that the three-branch network trains better than the plain stack is the premise of the whole subsection and is never shown; the last exercise asks the reader to test it. A single sentence with the paper's number (plain against re-parameterized accuracy on ImageNet) would let the reader believe the premise before doing the algebra.

**EFF-12** (minor, language). Recurring economic verbs. I count 7 uses of spend/spends, 2 of buys, 3 of trade: "spend parameters", "spends the same arithmetic", "what the separable network buys", "spend the savings on more channels", "the trade reverses". In a section about budgets some of this is literal, but "buys" and "the trade reverses" can be plain verbs.

**EFF-13** (minor, language). The table and prose call the number "test accuracy"; the code stores `trainer.history['val_acc']`, and the first half of the chapter and the two later sections call the same quantity "validation accuracy". Use one term across the chapter.

## Exercises

Nine exercises, seven of them code, four marked hard. Quality is high: "Cost ratio for larger kernels" is a good warm-up, "Normalization before the convolution" (the fold fails at the padded border) is an excellent exercise that tests understanding, and "Arithmetic intensity" turns the section's key observation into a calculation. Problems: "Equal arithmetic" should move into the text (EFF-6). "The inverted bottleneck" part (c) and "Training with RepVGG blocks" are each a small project with three sub-experiments; together the nine exercises ask for more than a dozen training runs. Trim to six, and add one pencil exercise on compound scaling (how many multiply-adds after $\phi = 3$?) if that topic stays.

## Still open from October

October marked `efficient-convnets.md:62` (the cost ratio) as a keeper; it still is. The October complaint of a catalogue of architectures still applies to "Scaling and Searching".
