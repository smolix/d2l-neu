# Exercise Catalog: chapter_computer-vision (group B)

Files owned: rcnn.md, semantic-segmentation-and-dataset.md, transposed-conv.md, fcn.md,
neural-style.md, kaggle-cifar10.md, kaggle-dog.md — all in
`/Users/smola/Repositories/github/d2l-neu/chapter_computer-vision`, in book order.

**Chapter overview.** Best external source overall is Michigan EECS 498-007/598-005
(Justin Johnson): Assignment 5 (single-stage YOLO-style vs. two-stage Faster R-CNN
detector) is the strongest match found for rcnn.md, and Assignment 4 (Network
Visualization + Style Transfer) matches neural-style.md almost function-for-function.
Stanford CS231n's Style Transfer notebook (present through the 2020 syllabus, since
retired in favor of Transformer captioning/self-supervised learning — verified by
diffing the 2020/2021/2023/2024 assignment3 pages) supplies the exact loss functions
and named layer/weight configs used below; its still-live Network Visualization
notebook is a strong "inspired by" source. Dumoulin & Visin's convolution-arithmetic
guide (already cited in-book) grounds transposed-conv.md's shape reasoning. The two
Kaggle sections are best served not by courses but by real competition
solutions/write-ups (nagadomi's 5th-place CIFAR-10 solution; Vallee's and Pradhan's
dog-breed transfer-learning write-ups), which supply concrete numbers no course
assignment has. Three real coverage gaps: (1) no course builds an FCN from scratch
with bilinear-kernel init — modern CV courses assign detection, not segmentation, as
their graded "build it yourself" project; (2) no course exercises paired image/label
random-cropping as a data-engineering problem; (3) transposed-conv's specific
"matmul-efficiency + value-equality" pairing appears original to this book. The
chapter-wide "Can you...?" filler-question tic and bare reflection prompts (flagged in
the prior style review) are confirmed throughout my 7 files; the exercises that were
already anchored to a citation or runnable code (transposed-conv's both items,
rcnn's SSD-comparison item, fcn's Xavier-init and skip-connection items) needed only
tone fixes, not external replacement. Totals: 18 existing exercises across 7 sections
(keep 9, rewrite 7, drop 2); 39 problems proposed below.

---

## chapter_computer-vision/rcnn.md — Region-based CNNs (R-CNNs)

**Topic:** The R-CNN family (R-CNN, Fast R-CNN, Faster R-CNN, Mask R-CNN) — two-stage
detection: region proposals, per-region CNN features, RoI pooling/align, classification
+ box regression — with a runnable RoI-pooling demo and a comparison to single-stage
detectors.
**Current exercises:** 2; disposition: keep 1, rewrite 1, drop 0 — both are already
grounded in a specific citation (YOLO, and Zhao et al. 2019's comparison figure); only
the "Can we frame detection as regression?" rhetorical opening needed a tone fix, not
replacement.

**External sources found:**
- Michigan EECS 498-007/598-005 (Justin Johnson), Assignment 5, FA2019/FA2020 — Q1
  walks through a single-stage detector "similar to YOLO" (fully-convolutional, dense
  per-cell prediction) on PASCAL VOC 2007; Q2 walks through a two-stage detector
  "similar to Faster R-CNN" combining a region proposal network with a separate
  recognition head. Direct fetch of the assignment sub-page hit a TLS error this
  session; content independently corroborated via two public solution mirrors —
  https://github.com/iMeleon/EECS-498-007-598-005-solutions and
  https://github.com/seloufian/Deep-Learning-Computer-Vision (both fetched and
  verified).
- Finding, not a source: Stanford CS231n's graded assignment sequence (checked the
  2020, 2021, 2023, and 2024 `assignment3` pages directly) never assigns object
  detection as coursework — detection appears only in lecture. EECS 498's A5 is the
  standout; formal graded two-stage-vs-single-stage exercises are otherwise rare.

**Proposed problem set:**
1. [conceptual] **Two-stage vs. single-stage framing.** Argue explicitly whether
   detection can be framed as one regression problem the way YOLO does, contrasting
   it point-by-point with this section's multi-stage propose-then-classify pipeline;
   state one runtime and one accuracy consequence of each design.
   *Provenance:* adapted from d2l's own ex1 (overlap high; tone fix only).
1. [conceptual] **SSD vs. R-CNN family comparison.** Using Figure 2 of Zhao et al.
   (2019), name two structural differences between SSD and the Faster R-CNN pipeline
   just described (anchor-generation stage count; where heads attach) and state which
   one dominates inference latency.
   *Provenance:* adapted from d2l's own ex2 (overlap high; wording tightened to a
   stated deliverable).
1. [short-code] **RoI pooling vs. RoI Align.** Extend the section's `roi_pool` demo:
   build the same 4x4 `X`/`rois` example but with one RoI boundary at a non-integer
   multiple of the bin size, then compute that region with `torchvision.ops.roi_align`
   alongside `roi_pool`. Report both 2x2 outputs, identify which entries differ, and
   tie the difference to the Mask R-CNN rationale given in this section.
   *Provenance:* inspired by this section's Mask R-CNN discussion (overlap low —
   original exercise design using a standard-library op already imported here).
1. [conceptual] **Region proposal network by hand.** Given a 10x15 feature map with 3
   anchor scales x 3 aspect ratios per location (:numref:`sec_anchor`), compute how
   many boxes the RPN's classifier must score in one pass, compare that count with
   selective search's ~2000 proposals, and state which two of the four listed RPN
   steps let it use fewer, better proposals.
   *Provenance:* adapted from this section's own Faster R-CNN step list (overlap
   high; anchor count grounded in :numref:`sec_anchor`).
1. [short-code] **Why R-CNN is slow.** Using a pretrained classifier already available
   in this chapter (e.g. the ResNet-18 used later in :numref:`sec_fcn`) and ~20 boxes
   on one image, measure wall-clock time for one forward pass per box (naive R-CNN
   style) vs. one whole-image pass plus `torchvision.ops.roi_pool` for the same boxes
   (Fast R-CNN style). Report the speedup ratio.
   *Provenance:* original (motivated by this section's own stated bottleneck claim).
1. [extended] **Minimal two-stage detector on a toy dataset.** Following the structure
   of EECS 498-007 A5 Q2, combine anchor generation (:numref:`sec_anchor`), an RPN
   binary objectness head, and this section's `roi_pool`/`roi_align` code into a
   minimal 2-class detector trained on synthetic colored rectangles on blank canvases;
   report precision/recall at IoU >= 0.5 on 50 held-out synthetic images.
   *Provenance:* inspired by EECS 498-007 A5 Q2 (overlap low — scaled to a toy
   dataset and this chapter's own primitives; cite on adoption).

---

## chapter_computer-vision/semantic-segmentation-and-dataset.md — Semantic Segmentation and the Dataset

**Topic:** Semantic vs. image vs. instance segmentation; loading/labeling the VOC2012
dataset; joint random-cropping of image+label pairs to preserve pixel correspondence.
**Current exercises:** 2; disposition: keep 1, drop 1 — the augmentation-feasibility
question is concrete and kept; the "how does this apply to autonomous
vehicles/medicine? other applications?" prompt has no artifact or success criterion
and is dropped.

**External sources found:**
- CMU 16-385 Computer Vision (Kris Kitani), Programming Assignment 7 — follows the
  course's lecture on "Segmentation and graph-based techniques" (graph cuts,
  normalized cuts, mean-shift, k-means/GMM): a classical, unsupervised segmentation
  assignment, not a supervised/CNN one. https://www.cs.cmu.edu/~16385/ (course
  schedule page fetched and verified).
- Szeliski, *Computer Vision: Algorithms and Applications* — chapter 6.4 covers the
  same classical family (graph-based segmentation, mean shift, normalized cuts) per
  the publisher's chapter listing; https://link.springer.com/book/10.1007/978-3-030-34372-9
  (existence verified via search; full chapter text not independently fetched).
- Finding: no course was found that exercises "paired random-cropping of image+label
  to preserve pixel correspondence" as a data-engineering problem — most courses hand
  students a ready-made `VOCSegmentation`-style Dataset and skip the cropping-
  consistency issue entirely. This appears to be a d2l-specific pedagogical point.

**Proposed problem set:**
1. [conceptual] **Augmentation feasibility audit.** For each augmentation introduced
   in :numref:`sec_image_augmentation` (random crop, horizontal flip, color jitter,
   ...), state whether it applies unmodified to an (image, label-mask) pair, needs a
   label-aware modification, or must be dropped, with a one-sentence justification
   each.
   *Provenance:* adapted from d2l's own ex2 (overlap high — same question, made
   checkable against a specific, already-enumerated augmentation list).
1. [short-code] **Label colormap round-trip.** Using this section's RGB-to-class-index
   colormap, convert one VOC label image to class indices and back to RGB; verify
   pixel-for-pixel that you recover the original (report the mismatched-pixel
   fraction, expected 0 outside anti-aliased borders).
   *Provenance:* original (uses the section's own colormap utility).
1. [short-code] **Cropping-consistency check.** Implement the paired random crop, run
   it 100 times on one example, and programmatically verify the image crop and label
   crop always share the identical (top, left) offset; report any mismatch as a
   failure.
   *Provenance:* original (regression-test framing of this section's own crop
   function).
1. [conceptual] **Instance vs. semantic boundary.** Using a VOC image with two
   overlapping instances of the same class, describe concretely what this section's
   semantic label already captures vs. what an instance-segmentation label would
   additionally need, and name one downstream task each representation directly
   supports.
   *Provenance:* adapted from this section's own definitions (overlap med — extends
   the existing conceptual distinction into a worked example).
1. [conceptual] **Classical baseline contrast.** Per CMU 16-385's segmentation
   assignment, run (or describe by hand for a small patch) a normalized-cut or
   mean-shift segmentation of one VOC image and compare its boundaries against the
   ground-truth semantic label: name one place they agree and one place they diverge
   because classical segmentation has no notion of class.
   *Provenance:* inspired by CMU 16-385 PA7 (overlap low — different technique
   family; cite on adoption).
1. [extended] **Class-imbalance audit of VOC2012.** Compute the pixel-count histogram
   across all 21 classes over the loaded training split, identify the two most
   under-represented classes, and propose one concrete mitigation (e.g.
   class-weighted loss, oversampling rare-class crops) that a later FCN-training
   section could use, with a one-paragraph justification.
   *Provenance:* original (data-analysis extension of this section's own
   dataset-loading code).

---

## chapter_computer-vision/transposed-conv.md — Transposed Convolution

**Topic:** Upsampling counterpart to convolution; manual `trans_conv` implementation;
equivalence to `nn.ConvTranspose2d`; convolution as matrix multiplication and its
transpose relationship.
**Current exercises:** 2; disposition: keep 2, rewrite 0, drop 0 — both are correct,
checkable pen-and-paper questions; the profile found only a minor context-dependency
issue in ex2, not a blocking one, so both are kept as-is aside from being made
self-contained.

**External sources found:**
- Dumoulin & Visin, "A guide to convolution arithmetic for deep learning"
  (arXiv:1603.07285) — already this section's own citation for "fractionally-strided
  convolution"; confirmed via the abstract that its central content is exactly "the
  relationship between convolutional and transposed convolutional layers" plus shape
  formulas relating input/kernel/stride/padding/output. A reference guide, not a
  problem set, but its worked shape-arithmetic is directly adaptable.
  https://arxiv.org/abs/1603.07285
- Stanford CS231n, Assignment 2, `ConvolutionalNetworks.ipynb` (2020 edition) — has
  students implement the forward and backward pass of a conv layer "from scratch"
  (confirmed page text: "implement several new layers that are commonly used in
  convolutional networks"); the backward pass of a conv layer is itself a
  transposed-convolution-shaped computation, though the fetched assignment page does
  not spell out that connection explicitly. https://cs231n.github.io/assignments2020/assignment2/
- Finding: no course was found posing "is matmul-convolution efficient, and does
  transposing that matrix reconstruct the exact input values" as one combined written
  question — this specific pairing (a memory-cost argument plus a value-equality
  argument) appears original to this section; external material treats the two halves
  (shape/arithmetic vs. from-scratch backward-pass implementation) separately.

**Proposed problem set:**
1. [conceptual] **Value equality under matrix transposition.** For the conv-as-matmul
   construction in :numref:`subsec-connection-to-mat-transposition`, with `Y = WX` and
   reconstruction `Z = W^T Y`, determine algebraically whether `Z = X` in general,
   state the extra condition on `W` (e.g. orthogonal rows) under which equality
   holds, and verify it on the section's own 2x2 example.
   *Provenance:* adapted from d2l's own ex1 (overlap high; added a verification
   step against the section's numeric example).
1. [conceptual] **Efficiency of matmul convolution.** State the memory cost (number
   of stored entries) of the dense matrix `W` vs. the original kernel `K` for an
   $n_h \times n_w$ input and $k_h \times k_w$ kernel, and explain in one sentence why
   frameworks use direct/im2col convolution instead of materializing `W`.
   *Provenance:* adapted from d2l's own ex2 (overlap high; made self-contained by
   supplying the shape variables instead of relying on ex1's framing).
1. [short-code] **Verify the transpose-of-conv identity.** Using this section's
   `trans_conv` and the `nn.Conv2d`/`nn.ConvTranspose2d` pair already demonstrated
   with shared weight `K`: build a conv layer `f` with random weights and input `X`,
   compute `Y = f(X)`, build transposed-conv layer `g` sharing `f`'s kernel, and check
   numerically whether `g(Y)` reproduces `X`'s shape and, separately, its values.
   *Provenance:* adapted from d2l's own Summary bullet on `f`/`g` shape-matching
   (overlap high — turns a stated claim into a runnable check) + inspired by Dumoulin
   & Visin's shape formulas (overlap low).
1. [short-code] **Output-padding for stride > 1.** Following Dumoulin & Visin's shape
   arithmetic, take a stride-2 conv layer, compute two different input heights that
   yield the same output height, apply the corresponding transposed-conv layer, and
   show only the correct `output_padding` recovers each original input's exact shape.
   *Provenance:* adapted from Dumoulin & Visin, arXiv:1603.07285 (overlap med — new
   problem, but the shape relationships come directly from the guide's formulas;
   cite on adoption).
1. [conceptual] **From-scratch backward pass as transposed conv.** In the style of
   CS231n's from-scratch conv-layer assignment, derive on paper the gradient of a 1D
   convolution's output with respect to its input, and show this has the same
   "broadcast-and-sum" structure as this section's `trans_conv` — i.e. that backprop
   through conv literally is a transposed convolution.
   *Provenance:* inspired by CS231n Assignment 2 `ConvolutionalNetworks.ipynb`
   (overlap low — CS231n has students implement forward/backward directly rather than
   derive this correspondence; cite on adoption).

---

## chapter_computer-vision/fcn.md — Fully Convolutional Networks

**Topic:** Builds an FCN (pretrained ResNet backbone + 1x1 conv + transposed-conv
upsampling with bilinear-interpolation init) for semantic segmentation; trains and
predicts on VOC.
**Current exercises:** 4; disposition: keep 3, rewrite 1, drop 0 — the Xavier-vs-
bilinear-init, full-test-prediction, and skip-connection (adapted from the FCN paper)
items are concrete and kept; the "tune hyperparameters, can you improve accuracy?"
item is filler and rewritten with a stated grid and improvement threshold.

**External sources found:**
- Long, Shelhamer & Darrell, "Fully Convolutional Networks for Semantic
  Segmentation," CVPR 2015 — already this section's own basis and its own ex4
  citation (skip connections / FCN-8s,16s,32s fusing pool3/pool4/final-layer
  predictions). Not a new source; used here to attach a concrete configuration name
  ("FCN-16s"-style) to the existing exercise's success criterion.
- Finding: no dedicated FCN-from-scratch graded assignment was found in the current
  CS231n or EECS 498-007/598 syllabi — both cover segmentation only in lecture and
  assign object detection (confirmed above, EECS 498 A5) as their graded vision
  "build it yourself" project. Building an FCN with a bilinear-init upsampling layer
  is not, on this evidence, a widely externally exercised task.
- CMU 16-385 PA7 (see semantic-segmentation-and-dataset.md above) covers segmentation
  only via classical, non-CNN methods, so it does not transfer as an FCN-specific
  source either — noted for completeness, not counted as a strong match.

**Proposed problem set:**
1. [short-code] **Xavier vs. bilinear init.** Re-run this section's training with the
   transposed-conv layer initialized via Xavier instead of bilinear-upsampling
   weights, all else fixed, and report the training-loss curve and final pixel
   accuracy for both initializations side by side.
   *Provenance:* adapted from d2l's own ex1 (overlap high; unchanged, already
   well-scoped).
1. [conceptual] **Hyperparameter search with a target.** Choose 3 values each for
   learning rate and weight decay (a 3x3 grid), train for a fixed reduced number of
   epochs on a VOC-train subset, and report the grid cell (with its mean pixel
   accuracy on VOC val) that beats this section's baseline by >= 1 percentage point,
   or report that none does.
   *Provenance:* adapted from d2l's own ex2 (overlap high; filler question replaced
   with a fixed grid and a stated improvement threshold).
1. [short-code] **Full test-set prediction.** Run this section's `predict` function
   over every image in the VOC val split (not just the single demo image), save the
   predicted label maps, and report mean pixel accuracy across the whole split.
   *Provenance:* adapted from d2l's own ex3 (overlap high; "test images" made
   concrete as the full val split with a reported metric).
1. [extended] **Multi-scale fusion (FCN-16s-style).** Following Long et al. (2015),
   add a second 1x1-conv + transposed-conv branch reading an earlier, higher-
   resolution ResNet stage, sum-fuse it with the section's stride-32 prediction
   before the final upsample, and report whether the fused model improves mean pixel
   accuracy over problem 3's plain model.
   *Provenance:* adapted from d2l's own ex4 (overlap high) and from Long, Shelhamer &
   Darrell 2015 CVPR (overlap high; cite on adoption).
1. [conceptual] **Boundary-error analysis.** For the 5 validation images where the
   trained model disagrees most with ground truth, categorize each disagreement as
   (a) interior class confusion or (b) an error within ~2 pixels of an object
   boundary; report the fraction of type (b) errors and relate it to the
   transposed-conv layer's kernel size (64) and stride (32) used in this section.
   *Provenance:* original (diagnostic extension of this section's own
   prediction/visualization code).
1. [short-code] **Bilinear kernel sanity check.** Print this section's
   `bilinear_kernel` weights for a small kernel size and verify by inspection (or an
   assertion) that they implement exact bilinear upsampling on a simple test image,
   comparing against `PIL.Image.resize(..., Image.BILINEAR)` on the same input.
   *Provenance:* original (uses the section's own `bilinear_kernel` function).

---

## chapter_computer-vision/neural-style.md — Neural Style Transfer

**Topic:** Gatys et al. style transfer via a frozen pretrained CNN's features:
content loss, Gram-matrix style loss, total-variation loss; optimizing the synthesized
image itself.
**Current exercises:** 4; disposition: keep 1, rewrite 2, drop 1 — the loss-weight
tradeoff item is directionally sound and kept; the layer-choice and
different-images items are rewritten with concrete choices/criteria; the
"style transfer for text" item is dropped as an out-of-scope reading prompt (the
section never discusses text data or a text analog of content/style losses).

**External sources found:**
- Stanford CS231n, Assignment 3, "Style Transfer" notebook — present through the 2020
  syllabus, removed from the 2021+ syllabus in favor of Transformer captioning/
  self-supervised learning (confirmed by comparing the 2020/2021/2023/2024
  assignment3 pages directly). Students implement `content_loss`, `gram_matrix`,
  `style_loss`, `tv_loss` from formulas, then run three named configs, e.g.
  "Composition VII + Tubingen" (content layer 3, content weight 5e-2; style layers
  [1,4,6,7], weights [20000,500,12,1]; tv_weight 5e-2) and "The Scream + Tubingen"
  (weights [200000,800,12,1], tv_weight 2e-2).
  https://cs231n.github.io/assignments2020/assignment3/ (notebook mirror verified at
  https://notebook.community/ALEXKIRNAS/DataScience/CS231n/assignment3/StyleTransfer-PyTorch).
- Stanford CS231n, Assignment 3, "Network Visualization" notebook — still present
  (confirmed via 2021/2023/2024 pages). Students use gradients of a frozen pretrained
  SqueezeNet w.r.t. input pixels to produce saliency maps and "fooling images" — the
  same "optimize an image via gradients through a frozen network" mechanism this
  section uses, applied to an interpretability task instead of synthesis.
  https://cs231n.github.io/assignments2021/assignment3/
- Michigan EECS 498-007/598-005, Assignment 4, Q3/Q4 — Q3 Network Visualization
  (saliency maps, fooling examples, class visualization via gradient ascent); Q4
  Style Transfer ("create images with the artistic style of one image and the
  content of another"). Verified via two independent public solution mirrors
  (github.com/iMeleon/EECS-498-007-598-005-solutions,
  github.com/seloufian/Deep-Learning-Computer-Vision), corroborating the same task
  pairing as CS231n's historical A3.

**Proposed problem set:**
1. [short-code] **Named layer-choice comparison.** Following CS231n's convention of
   naming specific layers, run style transfer three times, style image fixed,
   varying only the content layer among {an early conv block, this section's default
   middle layer, a late conv block}; report the content-loss value at convergence for
   each and one sentence on how much of the original silhouette survives.
   *Provenance:* adapted from d2l's own ex1 (overlap high) + inspired by CS231n's
   Style Transfer notebook layer-naming convention (overlap low).
1. [short-code] **Content/noise weight tradeoff.** Retrain the synthesized image at 3
   style-weight settings spanning two orders of magnitude (this section's default,
   10x lower, 10x higher), content/TV weight fixed, and report content-loss and
   TV-loss at each setting, stating which best matches the paper's "vivid style,
   preserved content" goal.
   *Provenance:* adapted from d2l's own ex2 (overlap high; now with named settings
   and a reported metric).
1. [short-code] **Style-image ablation.**
    1. Substitute a style image with a similar color palette but different
       brush-stroke scale.
    1. Substitute a style image with a similar brush-stroke scale but a very
       different color palette.
   For each, report which loss term (style, content, or TV) changes most relative to
   the section's original run, and state which image property the Gram-matrix style
   loss appears more sensitive to.
   *Provenance:* adapted from d2l's own ex3 (overlap high; "more interesting"
   replaced with two controlled, measurable comparisons).
1. [short-code] **Saliency map for the content image.** Following CS231n/EECS 498's
   Network Visualization notebooks, compute the gradient of the pretrained network's
   top-1 class score for the content image w.r.t. its pixels (reusing the network
   already loaded here for feature extraction), visualize it as a saliency map, and
   check qualitatively whether the highest-saliency region overlaps the chosen
   content layer's most-activated spatial location.
   *Provenance:* inspired by CS231n Assignment 3 Network Visualization and EECS
   498-007 A4 Q3 (overlap low — different task, interpretability rather than
   synthesis; cite on adoption).
1. [extended] **Guided/masked style transfer.** Using a thresholded or hand-drawn
   binary mask over the content image (foreground vs. background), apply two
   different style images to the two regions by restricting each style loss to its
   region's feature-map pixels, and produce one composite image; report both
   regions' final style losses.
   *Provenance:* original (spatial extension of this section's own style-loss
   machinery).
1. [conceptual] **Why optimize the image, not the network.** Explain why style
   transfer treats the synthesized image as the trainable parameter while freezing
   the CNN's weights, contrasting this with ordinary supervised training; state what
   the loss would end up optimizing instead if the network's weights were left
   trainable.
   *Provenance:* adapted from this section's own Method description (overlap high —
   turns an implicit design choice already stated in the text into an explicit
   "explain why" question).

---

## chapter_computer-vision/kaggle-cifar10.md — Image Classification (CIFAR-10) on Kaggle

**Topic:** Organizing raw CIFAR-10 image files, training a CNN from scratch with a
learning-rate-decay schedule, producing a Kaggle submission.
**Current exercises:** 2; disposition: keep 1, rewrite 1, drop 0 — the
no-augmentation accuracy question is concrete and kept; the fixed-hyperparameter run
is kept but its "can you further improve them?" filler tail is dropped in favor of a
benchmarked target.

**External sources found:**
- nagadomi/kaggle-cifar10-torch7 (GitHub), a verified 5th-place solution in the
  Kaggle CIFAR-10 competition (~2014-15) — a VGG-style 4-block CNN (channels
  64->128->256->256, 3x3 kernels, dropout 0.25/0.5) on GCN+ZCA-whitened, cropped/
  flipped/scaled 24x24 inputs; single model 93.32% test accuracy, 6-model average
  (different seeds) 94.15%; ~20h training + 2.5h prediction on a GTX760.
  https://github.com/nagadomi/kaggle-cifar10-torch7
- Kaggle, "CIFAR-10 – Object Recognition in Images" competition itself (page title
  verified live) — already the book's own reference; ground truth for evaluation
  metric (accuracy) and data format rather than a new exercise source.
  https://www.kaggle.com/c/cifar-10
- Finding: no university course exercise (CS231n, EECS 498) was found assigning this
  specific competition as coursework; the strongest external material here is
  competitor solutions, not course-authored problems — a good fit for a "practice
  Kaggle competition" section.

**Proposed problem set:**
1. [short-code] **Baseline run and report.** Train this section's model on the full
   training set with `batch_size=128, num_epochs=100, lr=0.1, lr_period=50,
   lr_decay=0.1` exactly as specified, and report validation accuracy and leaderboard
   position (or, if the competition is closed to new submissions, accuracy on this
   section's own held-out split).
   *Provenance:* adapted from d2l's own ex1 (overlap high; filler tail removed,
   deliverable unchanged).
1. [short-code] **Augmentation ablation.** Re-run with all image-augmentation
   transforms removed, all else fixed from problem 1, and report the accuracy gap.
   *Provenance:* adapted from d2l's own ex2 (overlap high; unchanged).
1. [short-code] **Close the gap to a known strong result.** Using nagadomi's
   single-model 93.32% as a target, modify this section's model with at least one of
   (a) GCN+ZCA-style whitening, (b) an added conv block, (c) horizontal-flip
   augmentation; report the change(s) made, resulting accuracy, and whether you
   closed at least half the gap between problem 1's baseline and 93.32%.
   *Provenance:* adapted from nagadomi/kaggle-cifar10-torch7 (overlap med —
   architecture/preprocessing ideas adopted, not code; cite on adoption).
1. [conceptual] **Why ensembling helps here.** nagadomi's solution gained 0.83 points
   (93.32% to 94.15%) averaging 6 identically-specified models differing only in
   random seed. Explain, in terms of bias vs. variance of the training procedure, why
   this improves test error at all, and state one reason the gain is bounded rather
   than growing indefinitely with more seeds.
   *Provenance:* adapted from nagadomi/kaggle-cifar10-torch7 (overlap med — numeric
   result reused as a worked example; cite on adoption).
1. [extended] **Leaderboard-style tracking.** Train 3 model variants (this section's
   baseline plus two of your own changes), evaluate each on the same held-out split,
   and produce a small table (model, accuracy, training time) — the kind of artifact
   this section's own submission process is built around.
   *Provenance:* original (packaging exercise around this section's own
   training/evaluation loop).

---

## chapter_computer-vision/kaggle-dog.md — Dog Breed Identification (ImageNet Dogs) on Kaggle

**Topic:** 120-way dog-breed classification using pretrained-backbone feature
extraction plus a small trainable head, contrasted with CIFAR-10's from-scratch
approach because images are larger and of varying size.
**Current exercises:** 2; disposition: rewrite 2, keep 0, drop 0 — both current items
stack multiple vague asks with no named model, metric, or range; both are rewritten
into single, well-scoped comparisons grounded in the benchmark write-ups below.

**External sources found:**
- Nicolas Vallee (dev.to), "Using Transfer Learning and TensorFlow to Identify Dog
  Breeds from Images" — a frozen MobileNetV2 (TF-Hub) backbone at 224x224 with only a
  single trainable Dense(120, softmax) head, no fine-tuning, no augmentation; ~67%
  validation accuracy after training on ~1,000 images "in a few minutes," ~99%
  training accuracy by epoch 19 on the full ~10k-image set; the author explicitly
  flags "no augmentation/fine-tuning" as future work.
  https://dev.to/nicolasvallee/using-transfer-learning-and-tensorflow-to-identify-dog-breeds-from-images-5b4b
- Debasish Pradhan (Medium), "Dog Breed Identification" — compares four pretrained
  backbones (Inception ResNet V2, NasNet Large, Xception, Inception V3) at 290-331px,
  each >80% solo validation accuracy (Inception ResNet V2 best, ~90%); an ensemble of
  Inception ResNet V2 + NasNet Large at 331x331, fine-tuning only the final FC layer,
  with shear/zoom/flip/rotation/brightness augmentation, reaches 0.188 log-loss on
  the Kaggle test set.
  https://medium.com/@pradhandebasish2046/dog-breed-identification-cd0a3c57805c
- Kaggle, "Dog Breed Identification" competition itself (page title verified live) —
  ground truth for the evaluation metric (multi-class log loss) and data format;
  already the book's own reference.
  https://www.kaggle.com/c/dog-breed-identification

**Proposed problem set:**
1. [short-code] **Epoch/batch-size sweep with a target.** Train at `num_epochs` in
   {10, 20, 40} with `batch_size=128`, `lr=0.01, lr_period=10, lr_decay=0.1` fixed,
   plot validation accuracy vs. `num_epochs`, and report the point (if any) past
   which additional epochs stop improving accuracy by more than 0.5 points.
   *Provenance:* adapted from d2l's own ex1 (overlap high; vague "increase" replaced
   with a named 3-point sweep and a stopping criterion).
1. [short-code] **Backbone comparison at fixed budget.** Following Pradhan's backbone
   comparison, swap this section's pretrained feature extractor for one deeper
   alternative already in the framework's pretrained-model zoo, retrain the same
   small head with everything else fixed, and report whether validation accuracy
   improves and by how much.
   *Provenance:* adapted from d2l's own ex2 (overlap high) + inspired by Pradhan's
   Medium write-up (overlap low — different backbones available in this book's
   framework than in that write-up; cite on adoption).
1. [short-code] **Does augmentation help transfer learning here?** Replicate Vallee's
   no-augmentation, frozen-backbone setup (train only a new head) on this section's
   pipeline and record accuracy; add this section's image-augmentation transforms
   with everything else unchanged and record accuracy again; report the delta and
   whether it is larger or smaller than problem 1's epoch-sweep delta.
   *Provenance:* adapted from Vallee's dev.to write-up (overlap med — its
   no-augmentation baseline adopted as an explicit control; cite on adoption).
1. [conceptual] **Why frozen-backbone feature extraction suffices here.** Vallee
   trains only a single Dense(120) layer on a frozen ImageNet backbone and still
   reaches high accuracy. Explain why this works reasonably well for dog-breed
   identification specifically (relate it to ImageNet's 1000 classes already
   including ~120 dog breeds), and name one other 120-class problem where the same
   frozen-backbone strategy would likely work far worse, with a reason.
   *Provenance:* inspired by Vallee's dev.to write-up (overlap low — result used as
   the jumping-off point for a general conceptual argument; cite on adoption).
1. [extended] **Two-model ensemble.** Train two independently pretrained-backbone
   models on this section's pipeline (two different backbones, or one backbone with
   two random seeds for the head), average predicted probabilities on the validation
   split, and report whether the ensemble's log-loss beats both individual models —
   the technique Pradhan used to reach 0.188 log-loss.
   *Provenance:* adapted from Pradhan's Medium write-up (overlap med — technique
   adopted at this book's smaller scale; cite on adoption).
