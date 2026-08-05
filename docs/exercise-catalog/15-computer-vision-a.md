# Exercise Catalog: chapter_computer-vision (group a — augmentation, transfer, detection pipeline)

Files: image-augmentation.md, fine-tuning.md, bounding-box.md, anchor.md,
multiscale-object-detection.md, object-detection-dataset.md, ssd.md.

**Chapter overview.** The single best external match for this group is Michigan
EECS 498-007/598-005 "Deep Learning for Computer Vision" (Justin Johnson),
**Assignment 5**, which has students build both an anchor-free single-stage
detector (FCOS-style: per-location classification/box/centerness heads, focal
+ L1 loss) and an anchor-based two-stage detector (RPN: multiscale anchors,
IoU matching with a three-way fg/bg/ignore split, NMS, RoI classification) —
this maps almost function-for-function onto anchor.md, multiscale-object-
detection.md, and ssd.md, and is used repeatedly below. torchvision's docs
(BoundingBoxFormat, tv_tensors-safe transforms, the object-detection
finetuning tutorial) are the best match for bounding-box.md and object-
detection-dataset.md. CS231n's transfer-learning notes and the fastai book's
questionnaire are the best match for fine-tuning.md. AutoAugment/RandAugment
are research papers, not course homework — good for "inspired by" framing on
image-augmentation.md but not "adapted from." CMU 16-385/16-720 were checked
directly and have **no** detection- or fine-tuning-specific homework (a real
negative finding, not an access failure). Szeliski's 2nd-edition page was
checked but its exercise text could not be verified online, so it is
deliberately not cited anywhere below. Most of this chapter's existing
exercises are already solid (bounding-box.md, multiscale-object-detection.md,
object-detection-dataset.md keep everything); the recurring "Can you...?"
filler-question tic (fine-tuning.md, image-augmentation.md, ssd.md, one item
in anchor.md) is rewritten with explicit deliverables/success criteria
throughout, per the prior style review's findings.

---

## chapter_computer-vision/bounding-box.md — Object Detection and Bounding Boxes

**Topic:** Two equivalent bounding-box parameterizations (corner vs.
center-width-height) and the plumbing (conversion functions, drawing helper)
reused by every later detection section.

**Current exercises:** 2; disposition: keep 2, rewrite 0, drop 0 — both are
already concrete with a clear deliverable (label a new image and compare
effort; explain why the box tensor's innermost dimension is 4), exactly the
kind of exercise the rubric says to keep as-is.

**External sources found:**
- torchvision docs, `torchvision.tv_tensors.BoundingBoxFormat` (current
  release) — defines six box formats: `XYXY`, `XYWH`, `CXCYWH`, plus three
  rotated variants (`XYWHR`, `CXCYWHR`, `XYXYXYXY`) — the real-world library
  analog of this section's `box_corner_to_center`/`box_center_to_corner` pair,
  extended to a format this section doesn't cover — https://docs.pytorch.org/vision/stable/generated/torchvision.tv_tensors.BoundingBoxFormat.html
- PyTorch, "TorchVision Object Detection Finetuning Tutorial" (current docs)
  — the `PennFudanDataset` example derives boxes from instance segmentation
  masks via `masks_to_boxes()` rather than hand-labeling coordinates directly
  — a concrete alternative box-construction path this section doesn't show —
  https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html
- Michigan EECS 498-007/598-005 (Justin Johnson), Assignment 5, Fall 2020 —
  both the one-stage and two-stage detector implementations treat IoU and box
  conversions as reusable scaffolding functions the student must implement
  once and reuse everywhere, the same role this section's helpers play in
  this book — https://web.eecs.umich.edu/~justincj/teaching/eecs498/FA2020/
  (assignment content cross-verified via https://github.com/iMeleon/EECS-498-007-598-005-solutions
  and https://raw.githubusercontent.com/grygry12345/DLCV-EECS498/master/A4/two_stage_detector.py)

No course found poses bare corner/center box conversion as its *own* graded
problem — it is universally treated as one-line scaffolding inside a bigger
detector assignment. That is itself the finding: this section's topic has a
thin standalone exercise tradition, so the proposals below borrow from
library design (extra formats, mask-derived boxes) rather than from any
course's problem set.

**Proposed problem set:**
1. [short-code] **Label and Compare a New Image.** Find a new image with at
   least two objects, hand-label bounding boxes using `bbox_to_rect`, and
   report which took longer: drawing boxes or assigning class names.
   *Provenance:* original (book's existing ex. 1).
1. [conceptual] **Why Boxes Need Four Numbers.** Explain in 2-3 sentences why
   the innermost dimension of `boxes` in `box_corner_to_center`/
   `box_center_to_corner` is always 4, and what would have to change for a
   *rotated* box.
   *Provenance:* original (book's existing ex. 2), sharpened by naming the
   rotated-box case that item 5 below develops.
1. [short-code] **Add a Third Box Format.** Implement `box_corner_to_xywh`
   and `box_xywh_to_corner` (matching torchvision's `XYWH`), then verify on
   the dog/cat boxes that corner → xywh → corner and corner → center → xywh
   → center both round-trip exactly.
   *Provenance:* adapted from torchvision `BoundingBoxFormat` (overlap med).
1. [short-code] **Bounding Box from a Segmentation Mask.** Given a binary
   mask array (e.g., a hand-drawn rectangle-plus-noise array, or a simple
   thresholded region), write `mask_to_bbox` that returns the tight
   `(x1,y1,x2,y2)` box, and check it against a box you drew by eye on the
   same region.
   *Provenance:* adapted from torchvision's `masks_to_boxes` utility used in
   the object-detection finetuning tutorial (overlap med).
1. [conceptual] **Rotated Bounding Boxes.** Sketch (no code) what extra piece
   of information a rotated box needs beyond `(cx, cy, w, h)`, and explain
   why center-width-height alone can no longer determine the four corners
   uniquely once rotation is allowed.
   *Provenance:* inspired by torchvision's `XYWHR`/`XYXYXYXY` formats
   (overlap low).

---

## chapter_computer-vision/anchor.md — Anchor Boxes

**Topic:** Generating multi-scale/aspect-ratio anchor boxes per pixel, IoU,
greedy ground-truth-to-anchor assignment, offset labeling, and (soft) NMS.

**Current exercises:** 5; disposition: keep 3, rewrite 2, drop 0 — ex. 1-3
(vary sizes/ratios; construct two boxes at IoU 0.5; perturb the worked
anchor example) are concrete and kept. Ex. 4 (soft-NMS) already cites Bodla
et al. correctly but never states a comparison deliverable, and ex. 5 ("can
NMS be learned?") is a bare rhetorical question per the prior style review —
both are rewritten below with an explicit artifact and success criterion.

**External sources found:**
- Michigan EECS 498-007/598-005, Assignment 5, Fall 2020, Q2 (two-stage/RPN
  detector) — implements multiscale anchor generation over FPN levels with
  configurable aspect ratios, a vectorized pairwise-IoU function, and a
  **three-way** (foreground / background / ignore) threshold assignment
  instead of this book's greedy one-to-one matching, plus anchor-delta
  transforms and NMS for proposal generation — this is the closest external
  analog to `multibox_prior`/`assign_anchor_to_bbox`/`nms` in the whole
  chapter — https://raw.githubusercontent.com/grygry12345/DLCV-EECS498/master/A4/two_stage_detector.py
  (assignment identity confirmed via https://github.com/iMeleon/EECS-498-007-598-005-solutions)
- Michigan EECS 498-007/598-005, Assignment 5, Fall 2020, Q1 (one-stage FCOS
  detector) — assigns ground truth to feature locations by a point-inside-box
  + object-scale-range rule instead of anchor-IoU matching, i.e., it solves
  the same labeling problem this section solves, without anchors at all —
  https://raw.githubusercontent.com/grygry12345/DLCV-EECS498/master/A4/one_stage_detector.py
- Bodla, Singh, Chellappa, Davis, "Soft-NMS — Improving Object Detection With
  One Line of Code," ICCV 2017 — already cited by the book's own ex. 4;
  confirmed as the standard reference for score-decay (rather than removal)
  suppression.

**Proposed problem set:**
1. [short-code] **Vary Anchor Scales and Ratios.** Change `sizes`/`ratios` in
   `multibox_prior` and report, for the same image, how the number and shape
   of anchors centered on one pixel changes.
   *Provenance:* original (book's existing ex. 1).
1. [short-code] **Two Boxes at IoU 0.5.** Construct and plot two boxes whose
   IoU is exactly 0.5 (solve for the overlap analytically first), and
   describe how they overlap.
   *Provenance:* original (book's existing ex. 2).
1. [short-code] **Perturb the Worked Anchor Example.** Modify the `anchors`
   tensor in the labeling/NMS worked example and report how the returned
   class labels and offsets change.
   *Provenance:* original (book's existing ex. 3).
1. [extended] **Soft Non-Maximum Suppression.** Implement Soft-NMS (Bodla et
   al.): replace `nms`'s hard removal with a Gaussian or linear score decay
   for boxes overlapping the current best box above a threshold, run it on
   the anchor boxes from the dog/cat example, and report which boxes survive
   under hard NMS but are merely down-weighted under soft-NMS.
   *Provenance:* original (book's existing ex. 4, tightened with a concrete
   comparison artifact); Bodla et al. 2017 already cited by the book.
1. [conceptual] **Can Suppression Be Learned?** Name one concrete learned-NMS
   design (e.g., a small network that scores box pairs instead of a fixed
   IoU threshold) and, in 3-4 sentences, state what input a learned
   suppressor would need that hard-threshold NMS ignores.
   *Provenance:* original (book's existing ex. 5, given a concrete
   deliverable in place of the bare rhetorical question).
1. [short-code] **Three-Way Anchor Labeling.** Extend `assign_anchor_to_bbox`
   with a second, lower IoU threshold: anchors above the high threshold are
   positive, below the low threshold are negative, and anchors in between are
   marked "ignore" (excluded from the offset loss). Report how many of the
   9 example anchors move from "assigned" to "ignore" versus the book's
   single-threshold version.
   *Provenance:* adapted from Michigan EECS 498-007 A5 Q2's RPN fg/bg/ignore
   split (overlap med — cite on adoption).
1. [conceptual] **Anchors vs. Anchor-Free Matching.** FCOS-style detectors
   assign ground truth to feature locations by "is this point inside the box
   and in its size range?" instead of anchor-IoU. Give one concrete ground-
   truth configuration (e.g., two objects of very different scale centered
   at the same pixel) where this book's anchor-IoU assignment could leave a
   real object with zero positive anchors, and one way anchor-free matching
   could itself fail.
   *Provenance:* inspired by Michigan EECS 498-007 A5 Q1 (FCOS) (overlap
   low).

---

## chapter_computer-vision/multiscale-object-detection.md — Multiscale Object Detection

**Topic:** Generating anchors on coarser feature maps to cut anchor count,
and using several feature-map resolutions to detect objects of different
sizes — the conceptual setup SSD implements next.

**Current exercises:** 3; disposition: keep 3, rewrite 0, drop 0 — all three
(abstraction-level question, generate overlapping anchors at 4×4, derive the
output shape of a class/offset transform) already have clear deliverables per
the prior style review, with no clarity issues found.

**External sources found:**
- Michigan EECS 498-007/598-005, Assignment 5, Fall 2020 — both detector
  variants build a feature pyramid (FPN) over 3 backbone levels and route
  ground-truth boxes to a level by an explicit size-based rule (rather than
  by IoU alone), which is exactly the "smaller anchors, smaller objects"
  intuition this section states in prose but never turns into an assignment
  rule — https://raw.githubusercontent.com/grygry12345/DLCV-EECS498/master/A4/two_stage_detector.py
- (Same source as above informs both proposed problems; no second
  independent course was found treating multiscale anchor generation as its
  own homework item — CMU 16-385/16-720 syllabi were checked directly and
  neither lists a detection-specific assignment.)

This is a case worth flagging explicitly: outside of Michigan's assignment,
no other checked course or textbook poses multiscale anchor generation as a
standalone exercise — it is always folded into a full detector build. The
two additions below are accordingly both keyed to the same source.

**Proposed problem set:**
1. [conceptual] **Do Scales Track Abstraction?** Argue, using the
   AlexNet-style hierarchical-feature discussion from `sec_alexnet`, whether
   feature maps at different scales in this section correspond to different
   levels of abstraction, or whether scale and abstraction are independent
   axes.
   *Provenance:* original (book's existing ex. 1).
1. [short-code] **Overlapping Anchors at 4×4.** At the first demo scale
   (`fmap_w=4, fmap_h=4`), generate anchor boxes that *do* overlap (rather
   than the section's non-overlapping demo choice) and visualize them.
   *Provenance:* original (book's existing ex. 2).
1. [conceptual] **From Feature Map to Predictions.** Given a feature map of
   shape $1\times c\times h\times w$, state the transform needed to produce
   per-anchor classes and offsets, and give the resulting output shape.
   *Provenance:* original (book's existing ex. 3).
1. [short-code] **Route Ground-Truth Boxes to a Scale.** Using the dog/cat
   boxes from `sec_bbox`, compute the IoU of each box against the anchors
   generated at each of this section's 3 demo scales (0.15/4×4, 0.4/2×2,
   0.8/1×1) and report which scale gives the highest max-IoU match for each
   box — i.e., which scale "owns" each object.
   *Provenance:* adapted from Michigan EECS 498-007 A5's FPN level-assignment
   rule (overlap med).
1. [conceptual] **What's Missing for a True FPN.** This section's 3 scales
   are independent downsamplings of one shared early feature map, not a
   feature pyramid with a top-down pathway. Name the two components (a
   top-down upsampling path; lateral 1×1 convolutions merging it with each
   level) that an FPN adds, and state one problem each solves.
   *Provenance:* inspired by Michigan EECS 498-007 A5's FPN backbone
   (overlap low).

---

## chapter_computer-vision/object-detection-dataset.md — The Object Detection Dataset

**Topic:** Loading a detection dataset (images + per-image bounding boxes),
padding variable object counts to a fixed `m`, and the banana toy dataset
used by the rest of the chapter.

**Current exercises:** 2; disposition: keep 2, rewrite 0, drop 0 — both are
concrete per the prior style review (visualize and compare other images;
reason about why detection-specific cropping is different, with a hint
already pointing at partially-cropped objects).

**External sources found:**
- PyTorch, "TorchVision Object Detection Finetuning Tutorial" (current docs)
  — uses the 170-image Penn-Fudan pedestrian dataset, derives boxes from
  masks via `masks_to_boxes()`, and wraps images/boxes/masks in
  `tv_tensors` so that transforms like `RandomHorizontalFlip` move the boxes
  automatically along with the pixels — a working, citable answer to this
  section's own ex. 2 hint about detection-safe augmentation —
  https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html
- torchvision `tv_tensors` design (same tutorial) — the general principle
  that a detection transform pipeline must jointly transform image and
  labels, not just the image, which this book's `sec_image_augmentation`
  pipeline (image-only) does not do.

Genuinely thin outside this one tutorial: this section's actual content
(padding a per-image object list to a fixed `m`, building the tiny banana
toy dataset) is bespoke teaching infrastructure, not something courses set
as homework — the torchvision tutorial is the only strong external anchor
found, so overlap is capped at "high" for the augmentation-safety problem
and "med" elsewhere.

**Proposed problem set:**
1. [short-code] **Visualize More Bananas.** Demonstrate 10 different images
   with ground-truth boxes and describe how box size/position/rotation vary
   across them.
   *Provenance:* original (book's existing ex. 1).
1. [conceptual] **Why Cropping Breaks Detection Labels.** Explain, using the
   hint about partially-visible objects, why a classification-style random
   crop (as in `sec_image_augmentation`) cannot be reused unmodified for
   detection, and name the two invariants a detection-safe crop must
   preserve (the box must still describe a visible region; the label must
   be dropped or clipped, not silently kept wrong).
   *Provenance:* original (book's existing ex. 2).
1. [short-code] **A Detection-Safe Random Crop.** Implement a random crop
   for `BananasDataset` that clips each box to the crop window and discards
   any box whose clipped area falls below 20% of its original area (setting
   its class to -1, matching the existing padding convention). Verify on 10
   images that no surviving box has zero area.
   *Provenance:* adapted from the torchvision object-detection finetuning
   tutorial's tv_tensors-safe transform design (overlap high — cite on
   adoption).
1. [conceptual] **Boxes from Masks vs. Boxes by Hand.** The banana dataset's
   boxes come from a labeled CSV; the Penn-Fudan tutorial instead derives
   boxes from segmentation masks via `masks_to_boxes()`. State one advantage
   and one disadvantage of deriving boxes from masks instead of hand
   annotation.
   *Provenance:* adapted from the torchvision tutorial (overlap med).
1. [short-code] **Two Bananas Per Image.** Build a small (~20-image) variant
   of the banana dataset with two bananas per image, by pasting a second
   banana crop from the existing training images onto a fresh background
   (matching the construction method described in this section's own
   introduction) and writing a matching 2-row-per-image label CSV. Adapt
   `read_data_bananas`/`BananasDataset` to a padded `m=2` and confirm the
   returned label tensor has shape `(batch, 2, 5)` with illegal rows marked
   class -1.
   *Provenance:* original.

---

## chapter_computer-vision/ssd.md — Single Shot Multibox Detection

**Topic:** The full TinySSD model — class/bbox conv predictors, downsampling
base network, 5-scale anchor generation, multi-task loss, training loop, and
NMS-based prediction.

**Current exercises:** 2 numbered items bundling 6 distinct embedded tasks
(ex. 1's two loss-function swaps: smooth L1, focal loss; ex. 2's four
sub-items: bigger input, downsample negatives, weighted loss, mAP
evaluation); disposition: keep 2, rewrite 4, drop 0 — sub-items 3-4 of ex. 2
(weighted loss, evaluate like the paper) are already concrete and kept as-is;
ex. 1's two loss swaps and ex. 2's sub-items 1-2 (resize input bigger,
downsample negatives) open with the house-style "Can you...?" filler and
lack a magnitude/ratio or success criterion per the prior style review, so
they are rewritten below into 4 separately checkable problems.

**External sources found:**
- Michigan EECS 498-007/598-005, Assignment 5, Fall 2020, Q1 (FCOS one-stage
  detector) — trains with **focal loss** for classification and an
  **L1/box-regression + centerness** loss, i.e., the same loss-function
  substitutions this section's ex. 1 already proposes, confirmed as a real,
  graded implementation task rather than a suggestion —
  https://raw.githubusercontent.com/grygry12345/DLCV-EECS498/master/A4/one_stage_detector.py
- Michigan EECS 498-007/598-005, Assignment 5, Fall 2020, Q2 (two-stage/RPN
  detector) — samples a *balanced* subset of positive/negative anchors per
  image for the RPN loss (rather than using every negative anchor), which is
  the working version of this section's ex. 2 sub-item 2 ("downsample
  negative anchor boxes") — https://raw.githubusercontent.com/grygry12345/DLCV-EECS498/master/A4/two_stage_detector.py
- Lin, Goyal, Girshick, He, Dollár, "Focal Loss for Dense Object Detection,"
  ICCV 2017, and Liu et al., "SSD: Single Shot MultiBox Detector," ECCV 2016
  — both already cited by the book's own exercises; confirmed as the correct
  primary references for the two loss functions this section discusses.

**Proposed problem set:**
1. [short-code] **Smooth L1 vs. L1 Box Loss.** Swap `calc_loss`'s `bbox_loss`
   for the smooth-L1 formula already given in this section (try
   $\sigma\in\{1, 10\}$), retrain for the same 20 epochs, and report the
   final bbox MAE against the section's L1 baseline.
   *Provenance:* original (from ex. 1's first embedded task).
1. [short-code] **Focal Loss for Class Imbalance.** Swap the cross-entropy
   class loss for focal loss (try $\gamma\in\{0, 2\}$, fixed $\alpha=0.25$),
   retrain, and report class error after 20 epochs against the baseline,
   given that background anchors outnumber foreground anchors by roughly
   1000:1 in this dataset.
   *Provenance:* original (from ex. 1's second embedded task); Lin et al.
   2017 already cited by the book.
1. [short-code] **Bigger Input, Smaller Objects.** Resize the training and
   prediction pipeline from $256\times256$ to $512\times512$ (leave anchor
   `sizes`/`ratios` unchanged), retrain, and report whether bbox MAE
   improves for the (largest) bananas whose ground-truth box area is in the
   smallest quartile of the dataset.
   *Provenance:* original (from ex. 2 sub-item 1, given an explicit
   resolution and a measurable target group).
1. [short-code] **Hard-Negative Downsampling.** Cap the negative:positive
   anchor ratio used in `calc_loss` at 3:1 by randomly subsampling negative
   anchors per image, retrain, and compare the class-error and bbox-MAE
   curves against the unmodified (all-negatives) baseline.
   *Provenance:* original (from ex. 2 sub-item 2, given an explicit ratio);
   the ratio-based balanced-sampling design is standard in RPN-style
   training as implemented in Michigan EECS 498-007 A5 Q2 (inspired by,
   overlap low).
1. [short-code] **Weighted Multi-Task Loss.** Assign separate weights to the
   class and offset terms in `calc_loss` (try weight ratios
   $\{1{:}1, 1{:}5, 5{:}1\}$), retrain each, and report which ratio gives the
   lowest combined class-error + bbox-MAE after 20 epochs.
   *Provenance:* original (book's existing ex. 2 sub-item 3).
1. [conceptual] **Evaluate Like the SSD Paper.** Describe how you would
   compute mean average precision (mAP) for TinySSD's banana predictions
   following Liu et al.'s evaluation protocol, and explain one respect in
   which this section's class-error + bbox-MAE metrics understate or
   overstate detection quality compared to mAP.
   *Provenance:* original (book's existing ex. 2 sub-item 4); Liu et al.
   2016 already cited by the book.
1. [extended] **A Two-Stage Objectness Pre-Filter.** Add a coarse
   "objectness" pre-filter to TinySSD, inspired by RPN's two-stage design:
   before computing the full class + box loss, threshold anchors by
   predicted max non-background probability and backpropagate the box loss
   only through the top-$K$ per image. Train for the same 20-epoch schedule
   and report whether bbox MAE and epoch wall-clock time change relative to
   the unmodified TinySSD baseline.
   *Provenance:* adapted from Michigan EECS 498-007 A5 Q2's two-stage RPN →
   RoI-classifier design (overlap med — cite on adoption).

---

## chapter_computer-vision/image-augmentation.md — Image Augmentation

**Topic:** Flips, random-resized crop, color jitter, composing multiple
augmentations, and comparing CIFAR-10 test accuracy with vs. without
augmentation.

**Current exercises:** 3; disposition: keep 1, rewrite 2, drop 0 — ex. 1
(train without augmentation and compare) is concrete and kept. Ex. 2
("does it improve test accuracy?") and ex. 3 (a bare "look it up in the
docs" prompt) both lack a stated deliverable per the prior style review and
are rewritten below with explicit artifacts.

**External sources found:**
- Cubuk, Zoph, Mane, Vasudevan, Le, "AutoAugment: Learning Augmentation
  Strategies From Data," CVPR 2019 (arXiv:1805.09501, 2018) — searches a
  space of augmentation sub-policies by validation reward and reports that
  policies learned on one dataset (ImageNet) transfer to others (Oxford
  Flowers, Caltech-101, Stanford Cars) — https://arxiv.org/abs/1805.09501
- Cubuk, Zoph, Shlens, Le, "RandAugment: Practical Automated Data
  Augmentation with a Reduced Search Space," NeurIPS 2020 (arXiv:1909.13719,
  2019) — replaces AutoAugment's policy search with two hyperparameters (N
  transforms applied, magnitude M), matching or beating searched policies on
  CIFAR-10/100, SVHN, and ImageNet — https://arxiv.org/abs/1909.13719
- torchvision docs, "Illustration of transforms" example gallery (current
  release) — a visual catalog of every built-in transform (flips, crops,
  color-jitter variants, `AutoAugment`, `RandAugment`, `TrivialAugmentWide`,
  `AugMix`), confirmed to be a pure example gallery with no built-in
  comparison exercise — https://docs.pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_illustrations.html

Both AutoAugment and RandAugment are research papers, not course homework —
no course was found that assigns them as a graded exercise, so every
proposal below keyed to them is "inspired by," not "adapted from."

**Proposed problem set:**
1. [short-code] **Augmentation vs. No Augmentation.** Run
   `train_with_data_aug(test_augs, test_augs)` (no augmentation) and compare
   final train/test accuracy against the section's augmented run; state
   whether the gap supports the claim that augmentation mitigates
   overfitting.
   *Provenance:* original (book's existing ex. 1).
1. [short-code] **Compose Three Augmentations, Name a Threshold.** Compose
   random-resized-crop + horizontal flip + color jitter (all already defined
   in this section), retrain for the same 10 epochs, and report whether test
   accuracy improves by at least 1 percentage point over the flip-only
   baseline.
   *Provenance:* original (book's existing ex. 2, given an explicit
   composition and a numeric threshold for "improve").
1. [short-code] **Two More Augmentations from the Docs.** Pick two
   augmentations not used in this section from your framework's transform
   docs (or the torchvision transforms gallery), add them to the `Compose`
   pipeline, and report the resulting test accuracy against the section's
   baseline.
   *Provenance:* original (book's existing ex. 3, converted from a bare
   lookup into a short-code task); torchvision transforms gallery cited as
   the concrete doc to consult (overlap low).
1. [short-code] **RandAugment-Style Composition.** Instead of hand-picking
   augmentations, apply $N{=}2$ randomly-sampled transforms from this
   section's own list (flip, crop, brightness, hue) at a fixed magnitude
   each training step, retrain, and compare test accuracy against the
   section's hand-composed pipeline from ex. 2 above.
   *Provenance:* inspired by RandAugment's $(N, M)$ parameterization (overlap
   low).
1. [conceptual] **Why Two Numbers Instead of a Search.** Explain in 3-4
   sentences why reducing augmentation choice to two numbers (RandAugment's
   $N$, $M$) trades optimality-per-dataset for practicality compared to
   AutoAugment's learned policy search, and say which of this section's own
   hyperparameters (flip probability, crop scale range, brightness/hue
   delta) would have to collapse onto a single shared magnitude knob to fit
   RandAugment's scheme.
   *Provenance:* inspired by AutoAugment and RandAugment (overlap low).
1. [conceptual] **Do Hand-Tuned Magnitudes Transfer?** This section tunes
   color-jitter strength (brightness/contrast/saturation/hue = 0.5) by eye on
   CIFAR-10. Without running code, argue whether the same magnitudes would
   still "look reasonable" applied to Fashion-MNIST (grayscale, so drop
   hue/saturation) from `sec_fashion_mnist`, given AutoAugment's finding that
   *learned* policies transfer across datasets but were never tuned by eye
   in the first place.
   *Provenance:* inspired by AutoAugment's cross-dataset policy-transfer
   finding (overlap low).

---

## chapter_computer-vision/fine-tuning.md — Fine-Tuning

**Topic:** Transfer learning via a pretrained ImageNet backbone, a freshly
initialized output layer, discriminative (10×) learning rates, and a
from-scratch baseline comparison on the hotdog dataset.

**Current exercises:** 4; disposition: keep 1, rewrite 3, drop 0 — ex. 3
(freeze the backbone, given working code) is already concrete and kept.
Ex. 1 ("keep increasing the learning rate") and ex. 2 ("adjust
hyperparameters... do they still differ?") are open-ended with no range or
target; ex. 4 ("how can we leverage this weight parameter?") has no
requested action — all three are rewritten below with explicit
ranges/techniques and a comparison target.

**External sources found:**
- Stanford CS231n, "Transfer Learning" course notes (cs231n.github.io) — a
  2×2 decision framework (new-dataset size × similarity to the source
  dataset) recommending linear-probe-only, partial fine-tune, or full
  fine-tune, plus the explicit rationale for using a smaller learning rate on
  pretrained weights: *"we don't wish to distort them too quickly and too
  much"* — https://cs231n.github.io/transfer-learning/ (no assignment
  attached to these notes; no explicit exercises on the page itself)
- fast.ai, *Deep Learning for Coders* ("fastbook"), Chapter 1 end-of-chapter
  questionnaire, 2020 — Q25-27 ask what pretrained models buy you, what a
  model's "head" is, and how early vs. late CNN layers differ in what they
  detect (generic vs. task-specific features) —
  https://butchland.github.io/butchland-machine-learning-notes/fastai/2020/06/01/fast-ai-chapter-1-intro-questionnaire-answers.html

CMU 16-385/16-720 syllabi were checked directly and list no fine-tuning- or
transfer-learning-specific homework (negative finding, not an access gap).

**Proposed problem set:**
1. [short-code] **How Far Can the Learning Rate Go?** Sweep
   `train_fine_tuning`'s learning rate over $\{5{\times}10^{-5},
   5{\times}10^{-4}, 5{\times}10^{-3}, 5{\times}10^{-2}\}$ and report the
   rate at which final test accuracy first drops more than 5 points below
   the section's baseline.
   *Provenance:* original (book's existing ex. 1, given an explicit grid and
   stopping criterion).
1. [short-code] **Matching Schedules for a Fair Comparison.** Train
   `finetune_net` and `scratch_net` with the same `num_epochs` and
   `batch_size` at $\{3, 10, 20\}$ epochs, and report the accuracy gap
   between them at each setting.
   *Provenance:* original (book's existing ex. 2, given explicit matched
   settings to sweep).
1. [short-code] **Freeze and Compare.** Freeze the backbone as shown (the
   section's own code cells) and report the resulting accuracy change versus
   full fine-tuning.
   *Provenance:* original (book's existing ex. 3).
1. [short-code] **Initialize the Head from ImageNet's "Hotdog" Class.** Use
   `pretrained_net`'s ImageNet "hotdog" row (already extracted in this
   section) to initialize the positive-class row of `finetune_net`'s 2-way
   classifier, instead of Xavier init; train for the same schedule and
   report whether final accuracy or convergence epoch changes versus the
   random-init baseline.
   *Provenance:* original (book's existing ex. 4, converted from a bare
   question into an implementable technique).
1. [conceptual] **Where Does Hotdog Recognition Sit in the Decision
   Framework?** Using CS231n's dataset-size × similarity framework, classify
   the hotdog dataset (1000 training images, food photos, ImageNet-similar
   domain) into one cell, state which of this section's three configurations
   (full fine-tune with 10× head LR, frozen backbone, from-scratch) that cell
   recommends, and check whether the section's own reported numbers are
   consistent with that recommendation.
   *Provenance:* adapted from CS231n's transfer-learning decision framework
   (overlap med — cite on adoption).
1. [short-code] **Partial Unfreezing.** Freeze all backbone layers except the
   last residual stage, fine-tune only {last stage + head}, and compare test
   accuracy and wall-clock training time against this section's full-fine-
   tune and fully-frozen (ex. 3) results.
   *Provenance:* inspired by fastbook's early-vs-late-layer framing (overlap
   low).
1. [conceptual] **Why Early Layers Transfer Better.** Without running code,
   argue why early convolutional layers (edge/color detectors) are more
   likely to transfer unchanged from ImageNet to hotdog recognition than
   late layers (object-part detectors), and name one signal in the training
   curves from problem 6 (if run) that would support or refute the claim.
   *Provenance:* inspired by fastbook Ch. 1 Q27 (early vs. late CNN features)
   (overlap low).
