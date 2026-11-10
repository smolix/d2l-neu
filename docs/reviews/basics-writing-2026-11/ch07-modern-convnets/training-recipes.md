# Modern Training Recipes for Convnets

Source: `chapter_convolutional-modern/training-recipes.md`. About 3,400 words of prose, 800 words of exercises, 6 exercises.

## Verdict

The best section of the chapter's second half, and the one whose idea transfers furthest: a reported accuracy is a property of a network and a training procedure together. The opening is exemplary, and the four worked ingredients (label smoothing, Mixup, stochastic depth, weight averaging) each get a meaning, a formula, a few lines of code and a check, which is the method of the model texts. Two things keep it from the bar. The history subsection in front of the ingredients names about fifteen techniques and systems before the reader has met any of them. And the closing experiment, headed "Controlled Comparison", changes optimizer, schedule, loss, augmentation and training length at once, uses two of the five ingredients the reader just built, and yields half a point to a point and a half. The most important fix: make the experiment an ablation the reader can read (start from recipe A and add ingredients in steps, at equal epochs), so that it demonstrates the method the section preaches.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought | 3 |
| Pedagogy | 4 |
| Language | 4 |
| Ease and enjoyment | 4 |

## What works

- The opening places the section in the chapter and states the question in the reader's terms: "So far in this chapter, the networks of each section were trained under one shared procedure". The ResNet-50 fact (76.1% to 80.4% with the architecture untouched) is the one sharp example the section needs. October marked it as a keeper and it still is.
- Label smoothing follows the full protocol: the defect of one-hot targets in words, the formula, a sentence that reads each term, the minimizer, and a three-class example whose two numbers (0.0001 and 0.67) I checked against the stored output.
- Mixup: "The cross-entropy is linear in the target distribution, so the loss on $\tilde{\mathbf{y}}$ equals the $\lambda$-weighted sum of the losses on the two original labels, and we implement it that way." The identity is used, not just stated. The picture of mixed Fashion-MNIST images earns its place.
- Stochastic depth is introduced as a change to something known ("applies the same idea to whole residual branches"), and the check (the fraction of examples whose output equals the ReLU of the input "should be close to $p$") is a neat, cheap test.
- Weight averaging: "the parameters of $k$ steps ago enter $\bar{\theta}$ with weight $(1-\beta)\beta^k$, so the average reaches back about $1/(1-\beta)$ steps", followed by the limit of the method ("The same horizon is the risk of the method"). Meaning, formula, number, boundary.
- Two observations after the experiment are worth more than its headline: training losses cannot be compared across recipes, and "AdamW's base rate is 25 times smaller than SGD's, so an ablation that swaps one ingredient must retune the others around it."
- "Comparing Published Results" gives the reader a habit for life (one-point gaps across papers of different years mean little) and backs it with ImageNet-V2.

## Structural findings

1. **The survey precedes the teaching.** "Training Recipes from 2015 to 2022" lists warmup, cosine decay, label smoothing, Mixup, zero initialization, knowledge distillation, ResNet-D, A1 to A3, LAMB, binary cross-entropy, AdamW, LARS, RandAugment, CutMix and stochastic depth in three paragraphs, before any of them is explained. The table that follows does the same job in six rows. Keep the table and one short paragraph that reads it, and let each ingredient be named where it is taught.
2. **The experiment is not controlled in the sense the heading claims.** See REC-8 and REC-9.
3. **Two of five ingredients are built and never used.** Stochastic depth and the `EMA` class are implemented here; recipe B omits both, and the ConvNeXt section sets its stochastic-depth rate to zero. In the whole chapter neither is ever part of a training run. The reader builds tools and is not shown them working on a network.
4. **Adam is assumed.** The section trains with AdamW and explains the "W" but not the "Adam". See REC-3.
5. **Scope of the title.** Nothing here is specific to convnets; the same ingredients train the transformers of later chapters. Saying so in one sentence would tell the reader this section is worth more than its place in the chapter suggests.

## Passage-level findings

**REC-1** (moderate, pedagogy). Under "Training Recipes from 2015 to 2022".
> tweak, *ResNet-D*, changes the downsampling blocks: their stride moves from

A three-line description of an architecture tweak inside the survey of training methods, in a section whose point is to separate the two. It is never used again. Cut it; if the 75.3% to 79.3% figure stays, say "together with small changes to the network".

**REC-2** (moderate, explanation). Under "Training Recipes from 2015 to 2022".
> of increasing cost: A3 (100 epochs) reaches 78.1%, A2 (300 epochs) 79.8%, and

Five accuracies, three procedure names, LAMB and binary cross-entropy in one sentence. The reader needs one of these numbers (80.4%), which the opening already gave. Keep the observation that matters, stated later in the same subsection: longer training and stronger regularization have to grow together.

**REC-3** (major, explanation). Under "Training Recipes from 2015 to 2022".
> implementations of Adam added weight decay to the gradient, where Adam's

A reader of chapters 1 to 7 has met Adam only as an optimizer that "keeps two numbers per parameter" (in the Computation chapter's memory accounting); its update rule comes in the optimization chapter. "Adam's per-coordinate rescaling distorts it" therefore explains nothing yet, and the experiment then rests on AdamW with a decay of 0.05 beside SGD with $5 \times 10^{-4}$, a hundredfold difference the text never comments on. Add two sentences: Adam divides each coordinate of the gradient by a running estimate of its typical size, so every weight moves at about the rate set by the learning rate; AdamW subtracts a fraction (learning rate times decay) of each weight directly. Then note that $0.002 \times 0.05 = 10^{-4}$ per step for AdamW against $0.05 \times 5 \times 10^{-4} = 2.5 \times 10^{-5}$ for SGD, so the two decays are closer than they look. The LARS and LAMB sentence can go.

**REC-4** (minor, explanation). Under "Label Smoothing".
> With $\epsilon = 0.1$ the loss of this example is 0.67 instead of 0.0001, so

The example would be sharper with its reference point: the smallest value the smoothed loss can take for three classes is about 0.29, reached at a logit gap of about 3.3, so 0.67 means the prediction is now penalized for being too confident. "Spends its effort on the others" is loose; the smoothed loss actively pulls the gap of 10 back toward 3.3.

**REC-5** (moderate, explanation). Under "Cosine Schedules with Warmup".
> $t_\textrm{w}$ steps, then the half-cosine *decay* :eqref:`eq_cosine_warmup`

The equation lives three sections back, in the AlexNet section, and the paragraph closes its numerical check with "as the formula predicts" although no formula is on the page. Display the two-piece schedule again here; this is the section that owns the topic. The schedule has by now been restated in prose in four sections of the chapter; one full statement here and a clause elsewhere would do.

**REC-6** (minor, language). Under "Cosine Schedules with Warmup".
> the 2015 recipe, each over the budget it gets in the experiment later in

The sentence runs to five lines and needs the experiment the reader has not seen ("30 epochs for the step decay, to which the experiment adds a two-epoch warmup, and 45 for the cosine, with 79 minibatches per epoch"). The function's docstring already says "Recipe A". Plot both schedules over the same number of epochs here and introduce the budgets with the experiment. "There is no time of a drop to tune" in the same paragraph should read "and it has no drop times to tune".

**REC-7** (minor, pedagogy). Under "Averaging Weights".
> With $\beta = 0.9$ the average reaches back about 10 steps: it fluctuates

The toy shows that an average of noise is smoother than noise, which the reader knew. It does not show the risk the next sentences describe (the average lagging behind weights that are still improving). Let the toy weight drift (for example from 0 to 1 over the first 50 steps, then noise around 1) and plot $\beta = 0.9$ and $0.99$: the lag becomes visible and the paragraph's claim is demonstrated. The stored plot also shows the average starting near 0.6 and taking ten steps to reach 1, which the text does not mention. Finally, $\beta$ here is the decay, while the exercises use $\boldsymbol{\beta}$ for the batch-norm shift; the batch-norm section calls its own decay $\rho$, which would serve here too.

**REC-8** (major, thought). Heading and first paragraph of "Controlled Comparison on One Network".
> The two recipes differ only in their optimizers, schedules, and training

They also differ in length (30 against 45 epochs) and in weight decay, and recipe A carries a warmup that the 2015 recipe did not have. With five things changed at once on 10,000 images, the reader cannot tell whether the gain comes from Mixup or from fifteen more epochs, and the text later concedes that such questions need the ablation it leaves to the hardest exercise. A comparison of two bundles is not a controlled comparison. Either rename the subsection ("Two Recipes on One Network") or, better, make it the ablation: recipe A; plus cosine schedule; plus label smoothing and Mixup; plus AdamW; each at the same number of epochs, in one table. Four runs of a ResNet-18 on 10,000 images is a small price for showing the reader the method the ConvNeXt section then relies on.

**REC-9** (moderate, thought). Under the results table.
> and a half, and its worst run is better than recipe A's best run. On ImageNet

The table gives the ranges as 89–90.5% and 90.5–91%, which touch, so the claim is not visible in the numbers the reader is shown; nor does the text say how many runs stand behind "every pair of runs". State the number of pairs, and report the paired difference (the quantity that carries the claim) as a column of the table. More generally the reader executes one run per recipe and is then told about a population of runs that happened elsewhere. One sentence on how to reproduce the spread (rerun the two cells with a different seed) would close the gap.

**REC-10** (moderate, pedagogy). Under the results table.
> The training curves also show why training losses cannot be compared across

A good observation, and the plots support it. The same plots show something the text misses: under recipe A the training loss falls to zero while the validation loss climbs from about 0.35 after the third epoch to about 0.6 before the first rate drop and settles near 0.45, even as accuracy improves. That is the growing overconfidence that the label-smoothing subsection described in the abstract. Point at it; it ties the experiment back to the ingredient.

**REC-11** (minor, thought). Under "Comparing Published Results".
> The experiment changes how published comparisons should be read. A gain of

The Fashion-MNIST experiment showed a gain of about a point; what changes how papers should be read is the four-point ResNet-50 result. Begin with that. In the next paragraph, "backbones", "object detection on COCO" and "segmentation on ADE20K" are undefined for a reader of the Basics part; say "differences on other tasks, such as locating objects in an image".

**REC-12** (minor, pedagogy). Framework notes. The MXNet note on bias-corrected weight decay ("by a factor of 0.32 at the first step and 0.8 at step 1,000") and the TensorFlow note on `CoupledSGD` are implementation audits. They belong in code comments.

## Exercises

Six exercises, well aimed. "Binary cross-entropy and Mixup" is a good conceptual question. "Weight averaging horizon" connects the horizon $1/(1-\beta)$ to the schedule and asks the right question about batch-norm statistics. "Weight decay and batch normalization" part (a) is the best exercise in my four sections: a scale-invariance argument with a surprising consequence. "Recipe ablation" is the experiment the text should run (REC-8); once it does, keep parts (a) and (c). The four hard exercises together ask for some thirty training runs of 45 epochs; say so, or reduce the sweeps.

## Still open from October

October's keeper (`training-recipes.md:9`) survives intact. The October complaint about lists of named methods applies to the 2015 to 2022 survey.
