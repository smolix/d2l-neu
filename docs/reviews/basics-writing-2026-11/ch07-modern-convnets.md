# Chapter 7: Modern Convnets

Sources: `chapter_convolutional-modern/` (index and eight sections). Reviewed
2026-11-09 at `201f2cb6`. Two reviewers read the chapter in full (sections 7.1
to 7.4 and 7.5 to 7.8); this file combines their chapter-level observations.
Detail and finding IDs are in the per-notebook files:

- [index](ch07-modern-convnets/index.md) (IDX)
- [alexnet](ch07-modern-convnets/alexnet.md) (ALX)
- [blocks](ch07-modern-convnets/blocks.md) (BLK)
- [batch-norm](ch07-modern-convnets/batch-norm.md) (BN)
- [resnet](ch07-modern-convnets/resnet.md) (RES)
- [efficient-convnets](ch07-modern-convnets/efficient-convnets.md) (EFF)
- [training-recipes](ch07-modern-convnets/training-recipes.md) (REC)
- [convnext](ch07-modern-convnets/convnext.md) (CNX)
- [cnn-design](ch07-modern-convnets/cnn-design.md) (DSP)

## Verdict

The chapter has escaped the history tour. Every section opens on a problem,
has a transferable idea (blocks as the unit of design, control of signal
scale, the identity as a starting point, cost as an objective, recipe against
architecture, attribution by ablation, populations against point designs), and
closes on a bridge. The sentences are clean. What holds it below the bar is
that its experiments cannot carry its claims. Every network is trained on
upsampled Fashion-MNIST, where everything lands between 91 and 94 percent, so
the chapter repeatedly states a result from a paper, runs an experiment that
does not show it, and then spends 100 to 300 words explaining what the
experiment does not show. The most important sections suffer most: ResNet
never shows that a plain deep network trains worse, and the two sections that
praise a method (ConvNeXt's ablation, RegNet's design spaces) do not perform
it. The first half is the stronger; the last two sections are Advanced or
survey material and the chapter ends without a conclusion.

## Scores

Scale: 5 = at home in Cover and Thomas or MacKay; 4 = good textbook prose with
local flaws; 3 = competent but ordinary, or good with a structural problem;
2 = a reader struggles.

| Section | Explanation | Thought | Pedagogy | Language | Enjoyment |
|---|---|---|---|---|---|
| index | 4 | 4 | 4 | 4 | 3 |
| 7.1 alexnet | 4 | 4 | 3 | 4 | 4 |
| 7.2 blocks | 4 | 3 | 3 | 4 | 3 |
| 7.3 batch-norm | 4 | 3 | 4 | 4 | 4 |
| 7.4 resnet | 3 | 3 | 3 | 3 | 3 |
| 7.5 efficient-convnets | 4 | 3 | 3 | 4 | 3 |
| 7.6 training-recipes | 4 | 3 | 4 | 4 | 4 |
| 7.7 convnext | 3 | 4 | 3 | 4 | 3 |
| 7.8 cnn-design | 3 | 3 | 2 | 3 | 2 |

## Most important findings

1. **The dataset cannot separate the designs.** The mini-MobileNet loses by 2
   points, the modern recipe gains 0.5 to 1.5 points with 50% more epochs,
   ConvNeXt loses to ResNet-18 by 2 points, RegNet ties at a 94% ceiling
   (EFF-6, REC-8, CNX-8, DSP-8); the first four sections all land at 91 to 93
   percent. A reader who runs the notebooks sees, eight times, that the new
   idea makes little or no difference, and is told each time to trust the
   paper. Either the experiments must be redesigned so that each isolates the
   one effect its section is about (a paired run, as with LeNet with and
   without batch norm, the one clean comparison in the chapter), or the
   chapter needs a harder small dataset. This is the central decision for the
   revision.
2. **ResNet does not demonstrate its own claim.** The degradation problem is
   quoted from the paper; the only run trains ResNet-18. Plain against
   residual at equal depth is left to exercise 4 (RES structural). This is
   the most consequential idea in the chapter, reused in every transformer,
   and it deserves the chapter's best experiment. "Function Classes and
   Depth" also argues beside the point: the text grants that a plain network
   contains the shallower solution while the nested-classes figure suggests
   it does not, and the real content is one sentence about where the identity
   sits in parameter space (RES-1, RES-2). The index promises that residual
   connections "let a deep network start out as a shallow one", but
   `Residual` does not zero-initialize (IDX-1, RES-3).
3. **Methods admired but not performed.** ConvNeXt's one-change-at-a-time
   ablation is read from the paper (CNX-9). The design-space section builds
   sampling code, samples nothing, and rests on a figure copied from the
   paper (DSP-5). "Controlled Comparison" in training-recipes changes five
   things at once (REC-8). The equal-arithmetic comparison that would
   vindicate MobileNet is an exercise (EFF-6).
4. **The comparison table in `blocks` is confounded.** GoogLeNet is trained
   at 96 by 96 and the others at 224 by 224, yet the text concludes that it
   "matches AlexNet with less than a third of its arithmetic"; most of that
   saving is the resolution (BLK-9, BLK-10). NiN's accuracy ranges over 84 to
   88 percent from a simplified last block (BLK-8).
5. **Batch normalization is only half honest.** One good paragraph says the
   explanation "is less settled than what it computes". But the opening
   restates internal covariate shift as fact ("these scales drift as the
   weights change"), the summary presents scale invariance as the cause, the
   experiment runs at a single learning rate and so does not show that
   "larger rates become safe", regularization by minibatch noise is absent,
   and the section never says why a reader who already has ReLU and He
   initialization needs the layer (BN-1, BN-5, BN-10, BN-13). ResNet has the
   mirror problem: no paragraph on what is known and what is folklore.
6. **The optimizers and the schedule are used before they are taught.**
   Warmup plus cosine is defined inside the AlexNet experiment and restated
   five times, with `configure_optimizers` redefined in four sections
   (ALX-9). Momentum is first explained in a parenthesis in the MobileNet
   subsection and again, almost verbatim, in cnn-design (EFF-4, DSP-12).
   AdamW rests on a one-line mention of Adam in chapter 5 (REC-3). The part
   needs one home for these before their first use; see the overview.
7. **`cnn-design` reads as a lab notebook.** About 330 words report runs that
   are not in the notebook: larger RegNets, zero-initialized scales,
   Nesterov, weight-decay sweeps, GPU timings (DSP-9, DSP-10). The section is
   a paper summary plus one tied training run, and it closes the chapter on
   an orphaned squeeze-and-excitation subsection (DSP-11).
8. **Survey passages remain.** Full GoogLeNet with its hand-tuned channel
   counts, the NFNets paragraph, the ResNet-D aside, DenseNet (three code
   cells, never assembled or trained), "Scaling and Searching" (EFF-9), the
   2015 to 2022 recipe history (REC-1, REC-2), and ConvNeXt's "Limits and
   Open Problems" (CNX-10).

## Arc, order and level

The first four sections are in the right order and belong in Basics, as do
depthwise factorization, batch-norm folding and the four recipe ingredients
(which are general training technique and not specific to convnets). ConvNeXt
is motivated by the Swin transformer, four chapters ahead, and RegNet is a
study of a research method; both are Advanced or survey material. At about
32,000 prose words the chapter is as long as Linear Regression, and its second
half is not proportionate: `cnn-design` could halve or shrink to a subsection,
and `convnext` could lose about 800 words of roadmap narration.

The current order ends on a 2020 paper after a 2022 one, with no closing
synthesis. A better order: efficient networks, then design spaces (its foils,
architecture search and compound scaling, are introduced there), then
recipes, then ConvNeXt as the closer that points to transformers. An
alternative is to place the design-space material directly after ResNet, since
it builds on `d2l.ResNeXtBlock`.

Repetition across the chapter: ResNet-18 is rebuilt three times in the second
half, `ModernRecipe` is defined twice with different structure, He
initialization is restated three times, the cosine schedule appears in at
least four sections, "multiply-adds are not time" three times, and the
inverted bottleneck is explained twice ("the trade reverses", "the economics
reverse") without a motivation or a figure (EFF-3, CNX-3). `drop_path` and
`EMA` are built and never used in a training run. "Bottlenecks" is in the
title of `blocks` and the word does not occur in its body.

## Recurring patterns

- **Narration of experiments the reader does not see.** Ranges "over repeated
  runs" with the number of runs unstated (REC-9), and passages on NiN, batch
  norm, ResNet and RegNet runs that are not in the notebook.
- **Symbol collisions within the chapter.** $\gamma$ and $\beta$ (batch norm,
  compound scaling, layer scale, EMA); $k$ (kernel size, bottleneck ratio);
  $g$ (groups, group width); $c_i$ (stage width, input channels); the paper's
  $w$ beside the book's $c$ (EFF-8, DSP-3).
- **Inconsistent metric name.** "Test accuracy" in two sections and
  "validation accuracy" elsewhere for the same `val_acc`.
- **Number-dense paragraphs.** Eleven numbers in five sentences for VGG-11. A
  parameter and multiply-add counting function would let the reader compute
  these instead of reading them.
- **Exercises that are projects.** About 30 training runs in
  training-recipes, 48 in one cnn-design exercise. The best exercises (blocks
  1, batch-norm 2, resnet 4) are the experiments the text itself should run.
- **Language.** Clean and free of the October tics, but flat; the reviewer of
  the first half found one light moment in four sections. Residual economic
  verbs remain in efficient-convnets (EFF-12).

## What to keep as the model

`alexnet` (a real question, arithmetic the reader can check, a cost table
that motivates the next section), the opening of `training-recipes` (76.1% to
80.4% from the recipe alone) with its four ingredients each given as meaning,
formula, code and check, the batch-norm folding algebra, and the measurement
showing that multiply-adds count arithmetic and not time.

## Not checked

Only PyTorch code and stored outputs were compared with the prose. Paper
numbers for MobileNet, RepVGG, ImageNet-V2 and RegNet timings were not checked
against the sources, nor were the parameter totals for GoogLeNet and
ResNet-18; AlexNet's, VGG's and NiN's counts were recomputed and are right.
Rendered HTML pages were not inspected.
