# Chapter overview — chapter_convolutional-modern

Best external source by far: "Understanding Deep Learning" (Prince), ch. 10-11 — verified directly against the local `udl_book.txt`, it has genuine, previously-untapped problems on AlexNet/VGG parameter and receptive-field accounting (10.16-10.19) and batch-norm/residual derivations (11.1-11.8), the richest vein found for this chapter. Stanford CS231n (Lecture 6/9 "CNN Architectures"; Assignment 2 Q2 Batch Norm, Q5 open-ended CIFAR-10 design) and Michigan EECS 498-007 (Assignment 3) cover the same ground but mostly as implementation tasks, not distinct reasoning problems. MIT 6.5940 (Song Han) has an on-topic, verified lab for cnn-design.md (Lab 3, Neural Architecture Search) but, despite being the suggested source, no dedicated lecture/lab on classic efficient-CNN architectures (MobileNet/ShuffleNet) turned up — a genuine gap. "Bag of Tricks" and ConvNeXt are already the book's own paper citations; no course was found assigning their ablation tables as homework, so they contribute paper-provenance, not course-exercise-tradition. convnext.md, efficient-convnets.md, and cnn-design.md have essentially no external homework tradition at all (2019-2022 papers, too recent/specialized for standard syllabi) — this chapter's own exercises are already the strongest tradition here, confirming the prior style review's "excellent, fully-quantitative" rating for those three files plus resnet.md and training-recipes.md (35 of 52 current exercises need no rewrite). batch-norm.md is the chapter's weak point (5 of 7 items vague or "Can you...?" filler) and gains the most from a genuine new addition (UDL's batch-norm backward-pass derivation). Totals: 8 sections, 52 current exercises (keep 41, rewrite 8, drop 3), 53 proposed problems.

---

## chapter_convolutional-modern/alexnet.md — AlexNet and Learned Image Representations

**Topic:** Learned vs. hand-designed image representations; the AlexNet architecture, dropout, ReLU, and its computational/training properties.
**Current exercises:** 8; disposition: keep 6, rewrite 2 (ex. 7, 8) — both rewritten items are pure "Can you...?" filler-tone questions per the prior style review; everything else is already a well-formed, checkable task.

**External sources found:**
- Simon J.D. Prince, *Understanding Deep Learning* (MIT Press, 2023), Problem 10.16* — asks the reader to compute the number of parameters used in each convolutional and fully connected layer of AlexNet (fig. 10.16) and the total — direct precedent for this section's own ex. 1. Verified against local book text. — https://udlbook.github.io/udlbook/
- Prince, *Understanding Deep Learning*, Problem 10.17 — asks for the receptive-field size at each of AlexNet's first three layers — a clean, checkable problem this section does not currently ask. — https://udlbook.github.io/udlbook/
- Stanford CS231n, "CNN Architectures" lecture (Lecture 6 Part 1, 2024) — case-study walkthrough of AlexNet's design and its place in the historical arc from hand-designed features; framing only, not an assigned exercise — https://cs231n.stanford.edu/2024/slides/2024/lecture_6_part_1.pdf
- Stanford CS231n, Assignment 2, Q5 "PyTorch on CIFAR-10" (verified via https://cs231n.github.io/assignments2024/assignment2/) — open-ended: design your own network in PyTorch and train it to maximize CIFAR-10 accuracy under a stated budget — same spirit as this section's ex. 5 (design a model for native 28×28 resolution).
- Michigan EECS 498-007 (Justin Johnson), Assignment 3 (verified via page content) — implements fully-connected and convolutional nets and explicitly asks for a submitted `one_minute_deepconvnet.pth` checkpoint, i.e., a speed-vs-accuracy deliverable parallel to this section's ex. 6. — https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/assignment3.html

**Proposed problem set** (our reference format):
1. [short-code] **AlexNet's Memory and Compute Ledger.** Using the section's own AlexNet definition, compute the parameter count and FLOPs of every convolutional and fully-connected layer, and report which layer type dominates each. Then explain how read/write bandwidth and latency affect training versus inference differently.
    1. Tabulate per-layer parameters and FLOPs; state the dominant layer type for each.
    1. Discuss the memory-bandwidth/latency argument for training vs. inference.
   *Provenance:* adapted from Prince, UDL Problem 10.16* (overlap high; cite on adoption — the parameter-accounting task is essentially identical, we add the FLOPs half and the bandwidth discussion from the book's original ex. 1).
1. [conceptual] **Receptive Fields of AlexNet's Early Layers.** Using the kernel sizes, strides, and paddings of AlexNet's first three convolutional layers as defined in this section, compute the receptive field (in input pixels) of a unit in each of those three layers, and state how much of the 224×224 input each can see.
   *Provenance:* adapted from Prince, UDL Problem 10.17 (overlap high; cite on adoption).
1. [conceptual] **The Chip Designer's Trade-off.** Argue, as a hypothetical chip designer, how you would trade off compute throughput against memory bandwidth given AlexNet's layer-wise compute/memory profile from problem 1. State one concrete design choice (e.g., cache size vs. ALU count) and its expected effect on AlexNet training throughput.
   *Provenance:* original (book's own ex. 2).
1. [conceptual] **Why AlexNet Benchmarks Disappeared.** In 3-5 sentences, explain why current papers no longer report standalone AlexNet accuracy numbers as a benchmark, referencing what changed about datasets, architectures, or evaluation norms since 2012.
   *Provenance:* original (book's own ex. 3).
1. [short-code] **Training Duration vs. LeNet.** Train AlexNet (on the section's dataset) for 2x and 5x the epochs used in the text; compare the resulting accuracy trajectory to LeNet's under the same schedule, and explain the difference in one paragraph.
   *Provenance:* original (book's own ex. 4).
1. [short-code] **A Budgeted Network for 28×28 Images.** AlexNet is oversized for Fashion-MNIST's native resolution.
    1. Simplify AlexNet to train faster without a significant accuracy drop; report the accuracy delta.
    1. Design a network that operates directly on 28×28 images and reaches, within a stated epoch budget, at least the simplified model's accuracy from (a).
   *Provenance:* inspired by CS231n Assignment 2 Q5 "PyTorch on CIFAR-10" for the fixed-budget open-ended framing (overlap low); base task is original (book's own ex. 5).
1. [short-code] **Batch Size, Throughput, and Memory.** Sweep the training batch size across at least four values and plot throughput (images/s), final accuracy, and peak GPU memory against batch size.
   *Provenance:* original (book's own ex. 6).
1. [short-code] **Regularization Ablation and Deliberate Overfitting.**
    1. Run a 2×2 ablation (± dropout × ± ReLU) on LeNet-5 and report test accuracy for all four conditions in a table; add one named preprocessing step (e.g., per-channel standardization) as a fifth controlled condition.
    1. Starting from trained AlexNet, remove or change exactly one regularizing ingredient (dropout rate to 0, or remove weight decay) and report the epoch at which the train/validation accuracy gap exceeds 10 points.
   *Provenance:* original (rewrite of book's own ex. 7 and ex. 8, replacing "Does it improve? Can you improve things further...?" and "Can you make AlexNet overfit?" with named ablation conditions and a numeric success criterion).

---

## chapter_convolutional-modern/blocks.md — Blocks, Bottlenecks, and Branches: VGG, NiN, GoogLeNet

**Topic:** Repeated-block design (VGG), 1×1 channel mixing and global average pooling (NiN), multi-branch stem-body-head design (GoogLeNet).
**Current exercises:** 11; disposition: keep 7 (ex. 1, 2, 3, 5, 8, 9, 11), rewrite 2 (ex. 4, 10), drop 2 (ex. 6, 7) — the two drops are sound, quantitative questions in their own right (NiN global-pool-vs-FC swap; the 384→10 bottleneck concern) but are redundant with the accounting problem below in an already-11-exercise set; they are trimmed for space, not quality, and could be restored if the book prefers a longer set. The two rewrites are the review's flagged "Can you...?" filler items.

**External sources found:**
- Prince, *Understanding Deep Learning*, Problem 10.18 — asks how many weights and biases are used at each convolutional and fully-connected layer of the VGG architecture (fig. 10.17) — direct precedent for this section's own ex. 1. Verified against local book text. — https://udlbook.github.io/udlbook/
- Stanford CS231n, "CNN Architectures" lecture — VGG/GoogLeNet case studies (parameter counts, stem-body-head framing); context only, not an assigned exercise. — https://cs231n.stanford.edu/2024/slides/2024/lecture_6_part_1.pdf
- Michigan EECS 498-007, Assignment 3 (verified) — implements a general `convolutional_networks.py`, the same substrate this section's parameter-accounting exercises would use, but has no VGG/NiN/GoogLeNet-specific written question. — https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/assignment3.html
- No good external tradition found specifically for NiN's global-average-pooling design or GoogLeNet's stem-body-head decomposition beyond survey-lecture coverage — this is a finding, not a failure; the section's own ex. 5-9 (already citing the VGG and SE papers) remain the strongest treatment found anywhere.

**Proposed problem set:**
1. [short-code] **Cross-Architecture Parameter and FLOP Accounting.** Compare AlexNet and VGG's parameter counts and FLOPs, separating convolutional from fully-connected layers, and propose one change that would reduce the FC-layer cost. Then extend the same accounting to NiN and GoogLeNet, and explain which specific design choice in each cuts parameter count so dramatically relative to VGG.
   *Provenance:* adapted from Prince, UDL Problem 10.18 (overlap high; cite on adoption) for the VGG half; original for the rest (folds together the book's own ex. 1 and ex. 11).
1. [conceptual] **The Missing Three VGG Layers.** The dimension trace for VGG-11 shows only eight blocks (plus auxiliary transforms) even though the network is described as 11-layer. Explain where the remaining three layers appear.
   *Provenance:* original (book's own ex. 2).
1. [short-code] **Reconstructing VGG-16/19 from the Paper's Table.** Using Table 1 of :cite:`Simonyan.Zisserman.2014`, implement VGG-16 and VGG-19 as variants of this section's VGG builder, and report their parameter counts against the VGG-11 baseline.
   *Provenance:* original (book's own ex. 3).
1. [short-code] **Resolution Conversion without Accuracy Loss.** Replace the 28×224 upsampling with resizing to 56×56 and to 84×84, adjusting the network to match. Report whether each variant stays within 1 point of the 224×224 baseline's accuracy; if not, add one nonlinearity before a downsampling step (per the VGG paper's discussion) and re-report.
   *Provenance:* original (rewrite of book's own ex. 4, replacing "Can you do so without reducing accuracy?" with an explicit 1-point tolerance).
1. [short-code] **NiN's 1×1 Convolution Count.** Vary the number of 1×1 convolutions per NiN block from one to three (the default is two) and report parameter count, accuracy, and training time for each variant.
   *Provenance:* original (book's own ex. 5).
1. [short-code] **Squeeze-and-Excitation for Inception.** Add a squeeze-and-excitation gate :cite:`Hu.Shen.Sun.2018` to the Inception block (global-average-pool the output, pass through a two-layer MLP with a sigmoid, rescale channels) and report the added parameter count and the change in Fashion-MNIST accuracy.
   *Provenance:* original (book's own ex. 8).
1. [short-code] **Depthwise Replacement for Inception Branches.** Replace the Inception block's four branches with a single 7×7 depthwise convolution followed by a 1×1 convolution (:numref:`sec_depthwise_separable`); compare parameter count and FLOPs against the original block at matched input/output size.
   *Provenance:* original (book's own ex. 9).
1. [short-code] **A GoogLeNet for Fashion-MNIST's Native Resolution.** State the minimum input size GoogLeNet's stem-body-head structure requires, then design a variant that runs on 28×28 images directly, reporting the specific changes made to the stem, body, and head and the resulting accuracy versus the 224×224 baseline.
   *Provenance:* original (rewrite of book's own ex. 10, replacing "Can you design a variant...?" with an explicit deliverable of stem/body/head changes plus a reported accuracy).

---

## chapter_convolutional-modern/batch-norm.md — Normalization Layers

**Topic:** Batch normalization's derivation and training-vs-inference behavior; layer and group normalization as alternatives.
**Current exercises:** 7; disposition: keep 3 (ex. 1, 2, 4), rewrite 3 (ex. 3, 5, 6), drop 1 (ex. 7) — this is the chapter's weakest set per the prior style review (5 distinct "Can you...?"/underspecified instances); ex. 7's five-part "think of other normalization transforms" brainstorm list is replaced below by a single well-specified derivation problem rather than patched.

**External sources found:**
- Stanford CS231n, Assignment 2, `BatchNormalization.ipynb`, Inline Question 3 (content verified via a public course mirror; official notebook at https://cs231n.github.io/assignments2024/assignment2/) — asks which standard data-preprocessing step is analogous to batch normalization and which to layer normalization, reasoning about batch-wise vs. feature-wise statistics — close to a good conceptual complement for this section's own layer-norm coverage.
- Prince, *Understanding Deep Learning*, Problem 11.5* — gives the batch-norm forward pass as a seven-step computational graph and asks the reader to derive the backward pass by hand, then implement both forward and backward in Python. Verified against local book text — this is a strong, previously-missing addition; the section itself only implements the forward direction as its own code, relying on framework autodiff for gradients. — https://udlbook.github.io/udlbook/
- Prince, *Understanding Deep Learning*, Problem 11.6 — parameter-count accounting for a 10-hidden-layer fully-connected network with and without batch norm between each linear layer and ReLU — a simple, checkable warm-up not currently in this section.
- Michigan EECS 498-007, Assignment 3 (verified) — implements batch normalization as part of the same assignment as fully-connected/conv nets, but with no distinct written question found beyond the coding task.
- fast.ai, *Practical Deep Learning for Coders* Part 2, Lesson 17 "Initialization/normalization" (verified via https://course.fast.ai/) — builds normalization layers up from first principles as part of the "foundations" sequence; teaching material, not a distinct graded exercise.

**Proposed problem set:**
1. [conceptual] **Bias Before Batch Norm.** Should the bias parameter be removed from the fully connected or convolutional layer that precedes a batch normalization layer? Justify your answer from the batch-norm formula.
   *Provenance:* original (book's own ex. 1).
1. [short-code] **The Batch-Norm Learning-Rate Ceiling.** Train LeNet with and without batch normalization.
    1. Plot the validation-accuracy increase batch norm provides at a fixed learning rate.
    1. Sweep the learning rate upward for both variants and report the largest value that still converges for each.
   *Provenance:* original (book's own ex. 2).
1. [short-code] **A Lite Batch Norm.** Implement a "lite" batch norm that only subtracts the mean, and a second variant that only divides by the standard deviation. Compare training curves and final accuracy of both against full batch norm.
   *Provenance:* original (book's own ex. 4).
1. [short-code] **Leave-One-Out Batch-Norm Ablation.** Starting from the network with batch norm after every layer, remove it from exactly one layer at a time and retrain. Report, as a table, the validation-accuracy cost of removing batch norm from each layer, and identify the layer whose removal hurts most.
   *Provenance:* original (rewrite of book's own ex. 3, replacing "Experiment with it" with a leave-one-out protocol and an explicit metric).
1. [conceptual] **Frozen β and γ.** Fix β=0, γ=1 (a no-op affine transform) in every batch-norm layer and retrain; separately, fix β=1, γ=0.1 and retrain. Compare final accuracy and epochs-to-converge for both against the learned-affine baseline, and explain the direction of each effect.
   *Provenance:* original (rewrite of book's own ex. 5, replacing "Observe and analyze the results" with two named parameter settings and two named metrics).
1. [short-code] **Batch Norm as a Dropout Replacement.** In the dropout-regularized LeNet variant, replace dropout with batch normalization and retrain. Report validation accuracy and the train/validation accuracy gap at a fixed epoch budget for both variants.
   *Provenance:* original (rewrite of book's own ex. 6, replacing "How does the behavior change?" with two named metrics).
1. [short-code] **Deriving Batch Norm's Backward Pass.** Working through the section's own forward-pass computational graph (mean, centering, square, variance, add-epsilon, reciprocal, scale-and-shift), derive by hand the expression for ∂z'_i/∂z_i for every element of the batch. Implement your derivation and verify it against the framework's autodiff gradient with `allclose`.
   *Provenance:* adapted from Prince, UDL Problem 11.5* (overlap high; cite on adoption).

---

## chapter_convolutional-modern/resnet.md — Residual Networks: ResNet, ResNeXt, and DenseNet

**Topic:** The function-class-nesting argument for residual connections; ResNet's basic/bottleneck blocks and the pre-activation ordering; ResNeXt's grouped convolutions; DenseNet's concatenation-based blocks.
**Current exercises:** 7; disposition: keep 7 — this is already an excellent set per the prior style review ("no clarity problems despite technical density," the heaviest and most precise citation use in the group) and needs no rewrites; we add one new problem below rather than replace anything.

**External sources found:**
- Prince, *Understanding Deep Learning*, Problem 11.2 — for the unraveled four-residual-block network, gives the exact path-count-by-length decomposition (1, 4, 6, 4, 1 paths of length 0-4) and asks the reader to deduce the general rule for K residual blocks. Verified against local book text — this is the "ensemble of shallow networks" view of ResNets (cf. Veit et al., 2016) and is not currently in this section's exercise set at all. — https://udlbook.github.io/udlbook/
- Prince, *Understanding Deep Learning*, Problems 11.1, 11.3, 11.4*, 11.8 — companion derivations (deriving the residual composition formula; its derivative w.r.t. the first layer; why parallel residual-branch outputs are uncorrelated and variances add; parameter counts for basic vs. bottleneck blocks at named channel widths). Verified against local book text; noted here as available but not separately adopted, since this section's own ex. 1-7 already cover this ground with comparable rigor.
- Stanford CS231n, "CNN Architectures" lecture — ResNet/ResNeXt/DenseNet case studies; framing only.
- CMU 11-785, Homework 2 Part 2 ("Face Verification using Convolutional Neural Networks," verified via https://deeplearning.cs.cmu.edu/F20/document/homework/Homework_2_2.pdf) — an open-ended, Kaggle-graded project where students typically implement a ResNet-family backbone; low overlap (task is face verification, not architecture analysis), useful only as a project-format precedent.

**Proposed problem set:**
1. [conceptual] **Inception vs. Residual Blocks.** Compare the Inception block (:numref:`fig_inception`) and the residual block: how do they differ in computation, accuracy, and the function classes they can represent?
   *Provenance:* original (book's own ex. 1).
1. [short-code] **ResNet Variants from the Paper.** Using Table 1 of :cite:`He.Zhang.Ren.ea.2016`, implement at least two other ResNet depth variants beyond the one built in this section.
   *Provenance:* original (book's own ex. 2).
1. [short-code] **The Bottleneck Architecture.** For deeper networks, implement the bottleneck block ResNet uses to reduce complexity, and compare its parameter count to the basic block at matched depth.
   *Provenance:* original (book's own ex. 3).
1. [short-code] **Pre-activation Ordering.** Change the "conv, batch norm, activation" ordering to "batch norm, activation, conv" per Figure 1 of :citet:`He.Zhang.Ren.ea.2016*1`, and compare training curves against the original ordering.
   *Provenance:* original (book's own ex. 4).
1. [conceptual] **Bounding Function-Class Complexity.** Explain why increasing function-class complexity without bound can be undesirable even for nested classes.
   *Provenance:* original (book's own ex. 5).
1. [short-code] **Why DenseNet Needs Fewer Parameters.** Explain why DenseNet models have fewer parameters than comparable ResNets, in terms of what a concatenated feature saves relative to recomputing it.
   *Provenance:* original (book's own ex. 6).
1. [short-code] **Dense Block Channel and Memory Accounting.** For a dense block whose k convolutions each emit g channels on an input with c channels, derive how many channels the i-th convolution consumes; sum these to compare activation memory against k residual blocks of constant width c, and relate the result to the memory-efficient implementation of :citet:`pleiss2017memory`.
   *Provenance:* original (book's own ex. 7).
1. [conceptual] **Counting Paths through a Residual Stack.** For a stack of K residual blocks, derive the number of end-to-end paths of each length from 0 to K when the block is unraveled into an ensemble of paths (as in fig. 11.4a for K=4, which has 1, 4, 6, 4, 1 paths of length 0-4). State the general rule in terms of K and relate it to why very deep plain networks are hard to train but very deep residual networks are not.
   *Provenance:* adapted from Prince, UDL Problem 11.2 (overlap high; cite on adoption).

---

## chapter_convolutional-modern/efficient-convnets.md — Efficient ConvNets: Depthwise Separability, Mobile Architectures, and Re-parameterization

**Topic:** Depthwise-separable convolutions (MobileNet), inverted bottlenecks, structural re-parameterization (RepVGG), width multipliers.
**Current exercises:** 5; disposition: keep 5 — per the prior style review this is one of the two "excellent, fully-quantitative" sets in the chapter (explicit `allclose` checks, measured batch sizes, a named α=0.5 test point); no rewrites needed.

**External sources found:**
- MIT 6.5940 "TinyML and Efficient Deep Learning Computing" (Song Han, hanlab.mit.edu, Fall 2024, verified via direct fetch of the course page) — has **Lab 1: Pruning** (due Sep 26) as its efficient-inference lab, and lectures 3-4 "Pruning and Sparsity." Despite being the suggested source, we could not find any dedicated lecture or lab on classic efficient-CNN *architectures* (MobileNet, ShuffleNet, SqueezeNet) in the current syllabus — this is a genuine, explicitly-checked gap, not an oversight. Pruning is a related but distinct efficiency technique (removing weights vs. redesigning the block), so we use it only as a low-overlap inspiration below. — https://hanlab.mit.edu/courses/2024-fall-65940
- No good external homework tradition found for RepVGG-style re-parameterization, or for the Conv-BN-folding identity this section derives, in any of CS231n, Michigan EECS 498-007, or CMU 11-785 — a finding, not a failure; this section's own exercises (fold-by-hand with `allclose` verification) appear to be the most rigorous treatment of this specific topic available anywhere we checked.

**Proposed problem set:**
1. [conceptual] **Folding Conv-BN by Hand.** Take a convolution followed by batch normalization, apply :eqref:`eq_bn_fold` to obtain one biased convolution, and verify with `allclose` that the two agree in inference mode. Explain why running statistics, not batch statistics, must be used.
   *Provenance:* original (book's own ex. 1).
1. [conceptual] **Cost Ratio for 5×5 Kernels.** Derive the cost ratio of :eqref:`eq_depthwise_sep_ratio` for 5×5 kernels. For which output-channel counts does the pointwise term dominate, and what does this imply for architectures (like ConvNeXt) that use larger depthwise kernels?
   *Provenance:* original (book's own ex. 2).
1. [short-code] **Measuring the Arithmetic-Intensity Gap.** Measure the forward-pass time of MiniMobileNet and its VGG-style twin on a batch of 128 images; despite roughly 7.7x fewer multiply-adds, MiniMobileNet's epochs are not 7.7x faster. Explain the gap in terms of operations-per-weight and operations-per-activation for depthwise vs. dense convolutions, and relate it to EfficientNetV2's fused early stages.
   *Provenance:* original (book's own ex. 3).
1. [short-code] **Plain Stack vs. Re-parameterized RepVGG.** Build a small network from `RepVGGBlock` layers and an otherwise identical one from plain convolutions; train both on Fashion-MNIST and compare training curves, given that the fused RepVGG layer computes the same function as the plain stack at inference.
   *Provenance:* original (book's own ex. 4).
1. [short-code] **Width Multiplier Scaling.** Add a width multiplier α to `MiniMobileNet` that scales every channel count; predict how parameter count and multiply-adds scale with α, then verify the prediction at α=0.5.
   *Provenance:* original (book's own ex. 5).
1. [short-code] **Pruning vs. Width Multiplier.** Apply channel-level magnitude pruning to a fully-trained MiniMobileNet until it reaches the same parameter count as the α=0.5 width-multiplier variant from problem 5. Compare final accuracy and measured inference latency between "prune after training" and "train narrower from scratch" at that matched parameter budget.
   *Provenance:* inspired by MIT 6.5940 Lab 1 (Pruning) methodology (overlap low — the technique family differs from this section's own re-parameterization/scaling focus, but the matched-budget comparison protocol is standard practice in that lab).

---

## chapter_convolutional-modern/training-recipes.md — Modern Training Recipes for Convnets

**Topic:** Isolating the training-procedure's contribution to accuracy, separately from architecture, using the 2015-vs-2022 ResNet-50 recipe gap as the running example.
**Current exercises:** 5; disposition: keep 5 — per the prior style review this is (with convnext.md and efficient-convnets.md) one of the three most rigorous, fully-quantitative exercise sets in the chapter (named ablation ingredients, a decay sweep over 0.9/0.99/0.999, a specific dataset `FashionMNIST10k`); no rewrites needed.

**External sources found:**
- "Bag of Tricks for Image Classification with Convolutional Neural Networks" (He, Zhang, Zhang, Zhang, Xie, Li, 2019, arXiv:1812.01187, verified via direct fetch) — reports raising ResNet-50 top-1 ImageNet accuracy from 75.3% to 79.29% through a stack of training-only refinements; already the section's own citation for the ingredient list. Its Section 3.1 "no bias decay" trick (excluding bias terms and BatchNorm scale/shift from weight decay) is not currently one of this section's ablated ingredients.
- Prince, *Understanding Deep Learning*, Problem 11.7* — asks what happens as training proceeds if L2 regularization is applied to convolutional weights but not to the subsequent BatchNorm scaling parameters. Verified against local book text — a direct conceptual match for the "no bias decay" trick above, and not currently in this section's exercise set.
- fast.ai, *Practical Deep Learning for Coders* Part 2, Lesson 18 "Accelerated SGD & ResNets" (verified via https://course.fast.ai/) — reconstructs a modern training recipe (momentum variants, schedules, ResNets) from first principles; teaching material that runs in parallel to this section's own comparison rather than a distinct graded exercise.
- No verified course was found that has turned "Bag of Tricks" or "ResNet strikes back" :cite:`wightman2021resnet` (already this section's own citation) ablation tables directly into a homework assignment; both function here as paper-provenance rather than course-exercise-tradition.

**Proposed problem set:**
1. [short-code] **Ablating the Modern Recipe.** On `FashionMNIST10k`: (i) remove Mixup, (ii) remove label smoothing, (iii) replace the cosine schedule with a constant rate after warmup, (iv) replace AdamW with SGD with momentum (retune the learning rate). Which single ingredient contributes the most? Do the four contributions sum to the total gap?
   *Provenance:* original (book's own ex. 1).
1. [short-code] **Disentangling Optimizer from Schedule.** Train recipe A's SGD-with-momentum under recipe B's cosine-with-warmup schedule, and AdamW under recipe A's step schedule. How much of the gap in :numref:`tab_recipe_results` is the schedule rather than the optimizer?
   *Provenance:* original (book's own ex. 2).
1. [conceptual] **Why Binary Cross-Entropy Pairs with Mixup.** :citet:`wightman2021resnet` train with a binary cross-entropy loss, treating each class as an independent target. Explain why this pairs naturally with Mixup and CutMix by considering the "correct" target for an image that genuinely contains parts of two classes, and contrast how softmax and sigmoid outputs represent it.
   *Provenance:* original (book's own ex. 3).
1. [short-code] **Weight-Averaging Decay Sweep.** Add the `EMA` class to the modern recipe, updating after every optimizer step and evaluating the shadow weights; sweep the decay over 0.9, 0.99, and 0.999. Relate the best decay to the schedule's low-learning-rate tail length, using the horizon estimate of roughly 1/(1-β) steps.
   *Provenance:* original (book's own ex. 4).
1. [short-code] **Implementing CutMix.** Following :eqref:`eq_mixup`, sample a rectangle of area fraction 1-λ, paste that region from the shuffled batch, and mix labels by actual area. Compare against Mixup on `FashionMNIST10k`, then alternate the two.
   *Provenance:* original (book's own ex. 5).
1. [short-code] **The No-Bias-Decay Ablation.** Apply weight decay only to convolutional/linear weights, excluding all bias terms and BatchNorm scale-and-shift parameters, on the section's own `FashionMNIST10k` setup. Compare final accuracy and the norm of the excluded parameters over training against decaying every parameter uniformly.
   *Provenance:* adapted from Prince, UDL Problem 11.7* (overlap high; cite on adoption) and the "no bias decay" trick in :cite:`He.Zhang.Zhang.ea.2019` (overlap high; cite on adoption).

---

## chapter_convolutional-modern/convnext.md — ConvNeXt: A ConvNet for the 2020s

**Topic:** The "modernize a ResNet toward a Transformer, one change at a time" ablation methodology; Global Response Normalization; depthwise kernel size, layer scale, and recipe ablations.
**Current exercises:** 5; disposition: keep 5 — per the prior style review this is the single best-quality exercise set in the entire chapter ("fully quantitative ablations with clear pass/fail criteria"); no rewrites needed, and we do not force an external addition here (see below).

**External sources found:**
- ConvNeXt paper itself (Liu et al., "A ConvNet for the 2020s," 2022, arXiv:2201.03545, verified via direct fetch) — reports 87.8% ImageNet top-1 at the largest scale and the step-by-step modernization roadmap; already this section's own primary citation, so it contributes no *new* provenance.
- No good external exercise tradition found for ConvNeXt specifically: it is a 2022 paper, and none of CS231n, Michigan EECS 498-007, CMU 11-785, MIT 6.5940, or fast.ai (all checked directly) has adopted its ablation roadmap as a homework assignment — this is an explicit finding, not a search failure. Given that this section's own 5 exercises are already rated the chapter's best, this is the one place where "no external tradition, and none needed" is the correct conclusion.

**Proposed problem set:**
1. [short-code] **Implementing Global Response Normalization.** For a channels-last feature map X, compute per-channel global norms g_c = ‖X_{:,:,c}‖₂, normalize as n_c = g_c / ḡ, and return γ⊙(X·n) + β + X with learnable per-channel γ, β initialized to zero. Insert it after the GELU in `ConvNeXtBlock` (ConvNeXt V2 also removes layer scale when doing so), retrain, and compare.
   *Provenance:* original (book's own ex. 1).
1. [short-code] **Swapping the Depthwise Kernel Size.** Train the model with 3×3 and with 11×11 depthwise convolutions (adjusting padding) and compare accuracy, parameter count, and time per epoch against the 7×7 baseline. Do you see the saturation :citet:`liu2022convnet` report?
   *Provenance:* original (book's own ex. 2).
1. [conceptual] **Where the Parameters Live.** Compute the fraction of ConvNeXt's parameters in depthwise convolutions, in the 1×1 expansions/projections, and in downsampling layers; compare with the ResNet-18 of :numref:`sec_training_recipes`. Which design decision explains why ConvNeXt is three times smaller at comparable depth-times-width?
   *Provenance:* original (book's own ex. 3).
1. [short-code] **Ablating Layer Scale.** Train with γ initialized to 1e-6 (default), to 1, and with the parameter removed entirely. Relate what you observe to stochastic depth, which also shrinks the effective contribution of each residual branch early in training.
   *Provenance:* original (book's own ex. 4).
1. [short-code] **The Recipe Downgrade.** Train this ConvNeXt with the 2015 recipe of :numref:`sec_training_recipes` (SGD with momentum, step decay, no Mixup or label smoothing). How much of the network's quality survives the recipe downgrade?
   *Provenance:* original (book's own ex. 5).

---

## chapter_convolutional-modern/cnn-design.md — Convolutional Network Design Spaces

**Topic:** Moving from hand-designed architectures and single-network NAS to designing a *distribution* over architectures (RegNetX/RegNetY).
**Current exercises:** 4; disposition: keep 3 (ex. 2, 3, 4), rewrite 1 (ex. 1) — the rewritten item is the review's flagged "Can you design a deeper RegNetX that performs better?" filler question; the other three are already well-specified.

**External sources found:**
- MIT 6.5940 "TinyML and Efficient Deep Learning Computing" (Song Han, hanlab.mit.edu, Fall 2024, verified via direct fetch) — **Lab 3: Neural Architecture Search** (due Oct 22) and Lecture 7-8 "Neural Architecture Search (Parts I-II)" are directly on this section's topic: searching or sampling architectures rather than hand-designing one. We could not verify the lab's exact internal tasks (its detailed handout was not accessible to us), so we cite only the verified lecture/lab titles and topic, not specific sub-tasks. — https://hanlab.mit.edu/courses/2024-fall-65940
- No good external homework tradition found that specifically reproduces the RegNet paper's design-space/EDF (empirical distribution function) methodology :cite:`Radosavovic.Kosaraju.Girshick.ea.2020` (already this section's own citation) — a finding; MIT 6.5940's NAS lab is the closest verified analog, but it targets single/weight-shared architecture search rather than population-level design-space analysis.

**Proposed problem set:**
1. [short-code] **A Deeper RegNetX at Matched Budget.** Increase the number of stages to four, holding total parameter count roughly fixed by adjusting per-stage widths; report whether accuracy improves or degrades relative to the three-stage baseline.
   *Provenance:* original (rewrite of book's own ex. 1, replacing "Can you design a deeper RegNetX that performs better?" with a matched-budget protocol and a reported outcome either way).
1. [short-code] **De-ResNeXt-ifying RegNet.** Replace the ResNeXt block with the ResNet block throughout RegNetX; report how the resulting model's accuracy and parameter count compare to the ResNeXt-block original.
   *Provenance:* original (book's own ex. 2).
1. [short-code] **VioNet: Breaking the Design Principles.** Implement multiple instances of a "VioNet" family that violate RegNetX's design principles one at a time. Report how each violation affects accuracy, and identify which of (d_i, c_i, g_i, b_i) is the most important factor.
   *Provenance:* original (book's own ex. 3).
1. [extended] **A ConvNeXt-Block Design Space.** Apply the AnyNet methodology (originally built on the ResNeXt block) to a design space built from ConvNeXt blocks (:numref:`sec_convnext`) instead: sample configurations, compare empirical CDFs of accuracy, and check which of the RegNet design principles survive the change of block.
   *Provenance:* original (book's own ex. 4).
1. [conceptual] **Design Spaces vs. Single-Network Search.** Contrast RegNet's design-space/EDF approach with a single-network NAS search (look up one method, e.g., DARTS or an evolutionary search). In 4-6 sentences, state what each approach optimizes for, what each costs in compute, and which failure mode (overfitting to one found architecture, vs. under-exploring the space) each is more prone to.
   *Provenance:* inspired by MIT 6.5940's Neural Architecture Search lecture/lab framing (overlap low — the comparison is our own synthesis; the course was not verified to ask this specific question).
