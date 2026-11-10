# Vision Transformer

Source: `chapter_transformers/vision-transformer.md` (read to the `<!-- slides -->` marker at line 888). About 2,700 words of body prose and 1,900 words of exercises; about 4,600 in all. Stored outputs read: `outputs/pytorch/chapter_transformers/vision-transformer.json`, the JAX file, and the four stored plots of each. Figure `img/mdl-transformers-vit.svg` rendered and inspected.

## Verdict

A clean, honest, well-built section that is one experiment away from the bar. The construction (patches as a strided convolution, class token, learned positions, the block reused through `ffn_factory`) is taught in the right order at the right length, the ViT against CNN comparison is the fairest small comparison of its kind I have seen in a textbook (same parameters, same FLOPs, same recipe, each model tuned over the same three learning rates, confounds named), and the exercises are excellent. The weak point is the middle. The position-embedding experiment measures a quantity that cannot answer the question it is asked (what the model has learned about the layout of its patches), is read for less than it shows in one respect and for more in another, and occupies the place where the section's best experiment (permute the patches and watch the accuracy) sits unused in exercise 3. The single most important fix: replace or re-read the grid-similarity experiment so that the body shows whether the trained ViT uses position, and by how much. Second: the section contains two paper retellings (registers, and the comparison at other scales) of which the first can shrink to two sentences.

Does it belong in a chapter on the transformer? Yes, as a short section. The chapter's claim is that one block serves every modality, and this is the one place where the reader sees the block run on something that is not text. It is a teaching section, not a survey. What it should not try to be is the book's treatment of vision transformers (pretraining, DeiT recipes, Swin, registers); those belong to the Image Models part and to `further-reading.md`, which already carries them.

## Scores

| dimension | score | note |
|:--|:-:|:--|
| Clarity of explanation | 4 | Patches, class token and positions are each clear on first reading. The `ViTBlock` paragraph is API plumbing in prose. |
| Clarity of thought and focus | 3 | The grid-similarity experiment does not measure what the text concludes from it; registers are a detour the model does not use. |
| Pedagogy | 3 | Fair, well-controlled comparison and very good exercises; but the body's own experiment is close to a null result while the decisive one is an exercise. |
| Language | 4 | Plain and exact, almost no mannerisms. Flat in the retelling paragraphs; That no part of the model knows which patches are neighbors is said three times. |
| Ease and enjoyment | 3 | The opening question pulls. The middle (block configuration, registers, a curve near zero) loses momentum, and the ending is a literature summary. |

## What works

- **The opening** (lines 4 to 29). The question "Can the same blocks, told nothing about images, learn to classify them?" is the right one, and the pixel-token arithmetic makes the obstacle felt: 9,216 pixels, 85 million pairs, against 36 tokens and 1,296 pairs (recomputed: $9216^2 = 8.49 \times 10^7$; $50176^2 = 2.52 \times 10^9$). Keep.
- **The convolution as projection** (lines 67 to 74). "The convolution's weight tensor of shape $(d, c, p, p)$ is the projection matrix, reshaped." One sentence that removes the mystery from `Conv2d(kernel_size=p, stride=p)`.
- **The controlled comparison** (`subsec_vit-cnn-baseline`). Parameter counts match (recomputed: ViT 6.46M, `CompactResNet` 6.45M), forward FLOPs match (recomputed: ViT about $4.8 \times 10^8$, CNN about $5.5 \times 10^8$), the recipe is literally shared (`configure_optimizers = ViT.configure_optimizers`), and each model gets the better of three learning rates. Then the text refuses the easy conclusion: "The experiment measures the two models under one protocol; it does not say which property of the CNN produces the gap." This is the candour the guide asks for (10.2), and the gap (0.94 against 0.89 in both tabs) is ten times the noise.
- **The reason for exempting position embeddings from weight decay**: "Decaying the position embeddings would also pull them toward zero while we try to measure what they learn." A detail with its reason attached.
- **The exercises.** Parameters against compute, readout, the permutation test, shift equivariance with the receptive-field count, attention distance, the Swin reach argument, and distillation with a control. Each has a question, most ask for a prediction or a control, and they are graded. Exercise 8 part 4 ("Construct a control that separates the two explanations") is the standard the body should meet.
- **One ending.** The body stops at its last result and the Summary follows.

## Structural findings

1. **The body's position experiment and exercise 3 should trade places.** The question of `subsec_vit-pos-grid` is whether the model has learned where its patches are. The direct measurement is functional: permute the patch tokens of the trained model and report the accuracy drop; train one model with zero position embeddings and compare. That is exercise 3, parts 2 and 3. It takes ten lines, cannot fail to be interpretable (a drop of zero means position is unused, a large drop means it is used), and connects to `subsec_pos-ignores-order`. The cosine-similarity curve can stay as a second, shorter look, or move to the exercises. See VIT-7 and VIT-8.
2. **Register tokens** (lines 350 to 371, about 210 words) retell one paper about models a thousand times larger, and end "so we train it without registers". The subsection sits between the model definition and its training, where the reader wants to run the model. It also reappears in the Summary and in `further-reading.md` ("Windows, Two-Dimensional Rotations, and Registers"). Reduce to two sentences at the end of the class-token paragraph, or move whole to further reading. See VIT-5.
3. **"The Comparison at Other Scales"** (lines 639 to 664) is a retelling too, but it is needed: without it the reader leaves believing ViTs lose. Keep it, shorten the first paragraph, and state the conclusion first. See VIT-11.
4. **What a reader looks for and does not find.** (a) Any picture of what the trained model attends to; exercise 6 has the measurement. (b) One sentence on why this matters in 2026: ViTs are the image encoders of vision-language models and are almost always pretrained, which is why the from-scratch comparison here is the hard case for them. `further-reading.md` line 256 has the sentence; the section's last paragraph should carry a clause of it. (c) That the images are $28 \times 28$ upsampled to $96 \times 96$, so a $16 \times 16$ patch covers under five original pixels per side; the patch-size discussion later depends on this and never says it.
5. **Length is right** if registers shrink. Nothing else needs to go.

## Passage-level findings

**VIT-1** (minor, language). Under the title, third paragraph. Anchor: "Nothing in a ViT tells it that two
patches are neighbors."
The same statement closes the first paragraph ("nothing in it says that the vectors stand for words, or that two of them are neighbors") and returns under "The Class Token and Position Embeddings" ("has no notion that patch 7 lies directly to the left of patch 8"). Three tellings in 250 lines. Keep the first and the third (the third is concrete); delete this sentence.

**VIT-2** (minor, explanation). Under "Patches as Tokens". Anchor: "A minibatch of four $96 \times 96$ color images"
The section's images are grayscale ($c = 1$, stated twelve lines earlier and used in `ViT`), but the shape check uses three channels because `PatchEmbedding` defaults to `num_channels=3` while `ViT` defaults to 1. The reader meets two conventions before training anything. Use one channel here, or say in a clause that the stem works for any $c$.

**VIT-3** (moderate, explanation). Under "The Encoder Block". Anchor: "`ViTBlock` is therefore a `d2l.TransformerBlock` with two changes, both"
This paragraph explains the code by naming keyword arguments (`norm='layer'`, "the `ffn_factory` hook", "MLP of free width `mlp_num_hiddens`", "in its exact form"). The reader who knows the default block (RMSNorm, SwiGLU) wants to know why the ViT goes back to LayerNorm and GELU, and gets "both from the original ViT recipe". Say in one sentence that we reproduce the 2020 recipe so that the model is the published one, that neither choice matters much at this size, and leave the argument names to the code. "In its exact form" (GELU without the tanh approximation) serves only cross-framework parity and belongs in the code comment that is already there.

**VIT-4** (minor, correctness, fairly sure). Under "The Encoder Block". Anchor: "ViT adopted this arrangement after analyses had shown that"
The three citations are a pre-norm language model (Baevski and Auli), a machine translation study (Wang et al.) and an analysis (Xiong et al.), the last of which appeared in 2020, the same year as ViT. "After analyses had shown" suggests a settled finding that ViT followed; the ViT paper cites the first two as its source for the layout. Write "ViT took this arrangement from earlier language models" and keep the citations.

**VIT-5** (moderate, focus). Under "Register Tokens". Anchor: "Our two-block model is far below the size at which the
artifacts appear, so we train it without registers."
A subsection that ends by saying it does not apply. The facts are right as far as I can check from memory (artifacts at about 2% of tokens with norms about ten times larger; one register removes them; four used in most experiments; under 2% extra FLOPs), but none is used again except in the Summary. Fold into the class-token paragraph: "Large ViTs trained for long turn a few background patches into such storage on their own, which spoils their patch outputs; adding a few more input-free tokens, called registers, prevents this :cite:`Darcet.Oquab.Mairal.ea.2024`." Drop the register sentence from the Summary.

**VIT-6** (minor, pedagogy). Under "A First Run". Anchor: "The ViT trains best at $3 \times 10^{-4}$ and ends four to five points lower at"
A useful, honest sentence, and no cell prints it. Either mark it as the authors' tuning ("in our runs") or make the sweep exercise-sized. As written it reads like a result of this notebook. The same paragraph (lines 383 to 404) is the longest in the section and carries five separate facts (DeiT ablation, optimizer, learning-rate tuning, warmup length, decay exemptions, omitted regularizers); split it after "so we use $10^{-3}$ for it".

**VIT-7** (major, correctness, sure of the numbers). Under "Grid Structure in the Learned Embeddings". Anchor: "The learned
embeddings show only a trace of this structure."
and, in the same paragraph, "an exercise tests whether the small dip is more than
chance". The stored curve (both tabs) is about $+0.05$ at distance 1, $-0.06$ at distances 2 to 4, $+0.04$ at distance 5. I simulated the null the text names (36 independent Gaussian vectors in 512 dimensions, centered, 200 draws): each distance's mean is $-0.029 \pm 0.004$, and the difference between distances 1 and 3 has standard deviation 0.005. The observed difference of about 0.10 is twenty standard deviations from chance. So the structure is not "a trace" whose reality an exercise must establish: it is small in size and beyond doubt. The curve looks flat only because the sinusoidal reference sets the axis to $[-0.5, 0.75]$. State it straight: "Neighbors are more similar than distant patches by about a tenth. That is far outside what random vectors give (their curve is flat at $-1/35$ to within a hundredth), and far below the fixed code."

**VIT-8** (major, thought). Same paragraph. Anchor: "Ten epochs on 60,000 images teach the ViT little about its own
geometry."
The conclusion does not follow from the measurement, in either direction. The embeddings start as noise of norm about $0.02\sqrt{512} \approx 0.45$ per vector and are trained at a peak rate of $3 \times 10^{-4}$ for 4,700 steps without decay, so most of each vector is still its initial noise; the cosine between two positions is dominated by that noise whatever the model has learned. And the model does not need similar embeddings for neighbors: 37 distinct random vectors already identify every position, and attention can learn any relation between them through $\mathbf{W}_q^\top \mathbf{W}_k$. Cosine similarity of the table measures how the table is organized, not what the network knows. The later sentence "the position embeddings of :numref:`subsec_vit-pos-grid` show how much of the spatial layout the ViT has inferred after ten epochs" repeats the overreading. Fix in two steps. (1) Measure the learned part: keep a copy of the initial table and plot `grid_similarity(learned - init)`; if the update is grid-structured this shows it at full scale, with no change of model. (2) Measure use: move exercise 3 parts 2 and 3 into the body (accuracy of the trained model under a random patch permutation; accuracy of a model trained with zero position embeddings). Then the sentence that leads to the CNN can be a number: "shuffling the patches costs the ViT x points, so it does use the layout, but a model that never knew the layout loses only y."

**VIT-9** (minor, explanation). Same subsection. Anchor: "counted as the larger
of the row offset and the column offset, so that the eight cells around a
patch are at distance 1"
Clear. But the reader is not told why this distance and not Euclidean distance, and the choice produces the artifact that the text must then explain away (distance 5 contains only edge-to-edge pairs, 140 of 1,260). A row-offset by column-offset heat map of the mean similarity (a $6 \times 6$ array printed or shown) would show the row and column structure that Dosovitskiy et al. report and remove the need for the border explanation. Optional; the one-curve version is acceptable if VIT-7 and VIT-8 are fixed.

**VIT-10** (minor, language). Under "A Convolutional Baseline at the Same Budget". Anchor: "Plotting the validation accuracy of the two runs against the epoch shows
how the gap develops over training:"
The plot is then never read. The stored plot shows the CNN ahead from the first epoch (0.865 against 0.83) and the gap roughly constant at four to five points; that is worth one sentence because it rules out "the ViT merely needs longer". If the plot is not read, drop the cell.

**VIT-11** (moderate, correctness, unverified). Under "The Comparison at Other Scales". Anchor: "The patch size alone moves the result by two points on Fashion-MNIST and
by sixteen on CIFAR-10."
The paragraph opens with a conclusion whose source arrives a sentence later, and then gives seven accuracies to one decimal from one paper (93.2, 94.5, 95.2, 94.8, 78, 94, and parameter counts of 3.7 to 3.9 million). I cannot confirm these numbers from memory; the paper (Hassani et al., compact transformers) does report ViT-Lite models at patch sizes 16, 8 and 4 against ResNets on small datasets, and the direction is right. Check them against the paper's table before the revision. Independently, the passage is too exact for its job: the reader needs "with long training and augmentation the same small ViT reaches 93% at $p = 16$ and 95% at $p = 4$, level with a ResNet-18", and the link to our images (upsampled from 28 pixels, so $p = 16$ is a coarse cut of the original).

**VIT-12** (minor, thought). Same subsection, last sentence. Anchor: "the assumptions that the CNN builds in are no
longer needed for accuracy, and the transformer contributes its generality:
the same block serves text, images, and both together"
A good closing thought stated as a general truth after an experiment that showed the opposite at the reader's scale. Say what the reader should take away for practice: at this scale use the CNN or a pretrained ViT; from-scratch ViTs need data or a recipe. One clause that ViTs are today's standard image encoders because they are pretrained (see Structural 4b) completes it.

**VIT-13** (minor, language). Summary. Anchor: "Large ViTs also need a few register tokens as internal storage."
"Need" overstates (they benefit; many deployed ViTs have none), and the sentence is the only trace of a subsection the model does not use. Delete with VIT-5. The line also runs to 150 characters in the source, a sign it was pasted in.

## Code and experiments

- `vision-transformer-patches-as-tokens-2`, `vision-transformer-the-vision-transformer-block-2`: shape checks. They can fail and are short; fine, though the block check (`ViTBlock(24, 48, 8, 0.5)` on 100 tokens of width 24) uses a configuration unrelated to anything else in the section.
- `vision-transformer-a-first-run-2`: prints 0.89 in both tabs; prose "about 90%" matches.
- `vision-transformer-do-the-position-embeddings-discover-the-grid`: the only experiment the section designs itself, and the one that needs work (VIT-7, VIT-8). The sinusoidal reference values quoted in the prose are right (recomputed with NumPy: 0.76, 0.21, $-0.30$, $-0.46$, $-0.36$), as is the $-1/35$ baseline (simulated: $-0.029$).
- `vision-transformer-a-convolutional-baseline-at-the-same-budget-2`: 0.94 against 0.89 in both tabs; "four to five points" matches at the printed precision. Single runs, but the gap is far above seed noise.
- `...-budget-3`: plot not read by the prose (VIT-10).
- Numbers no cell prints: the learning-rate sweep (VIT-6); "about half a billion floating-point operations per image" (deferred to exercise 1, which is acceptable, and my count agrees).
- Tabs: PyTorch and JAX only, no TensorFlow or MXNet. The JAX `PatchEmbedding` carries a custom initializer to mimic PyTorch's default; it is commented in the code and does not touch the prose. Tab prose: none. Good.

## Exercises

Eight problems, about 70% of the body length; all are worth keeping. Changes: exercise 3 parts 2 and 3 move into the body (Structural 1), which leaves parts 1, 4 and 5 as a good problem. Exercise 4 (permutation test of the grid curve) loses most of its point once VIT-7 is fixed, since the answer is "one in a thousand permutations at most"; keep part 2 (30 epochs) and attach it to exercise 3, or let the exercise ask for the test on `learned - init`. Exercise 7 (Swin reach) is the best pure-thought problem in the three sections. Exercise 8 is long for an exercise but every part teaches. One addition worth making: resize to $48 \times 48$ with $p = 8$ (same 36 tokens, a quarter of the stem input) and compare accuracy and time, which exposes the upsampling.

## Still open from October

October asked to cut "the duplicate ViT block": done, `ViTBlock` now subclasses `d2l.TransformerBlock`. Nothing else from October applies to this file.

## For the chapter review

- Position: after `encoders-decoders.md`, which is right, since the ViT is "the encoder without a mask" plus a tokenizer. It assumes `sec_transformer-block`, `subsec_pos-ignores-order`, `sec_positional-information`, `sec_resnet`, `sec_training_recipes` and `subsec_adamw-what-not-to-decay`; all labels resolve. No appendix dependence. No load-bearing reference to the Language Models part.
- Overlap: registers are told here and in `further-reading.md`; the DeiT story is told here (two places) and in further reading ("Vision Transformers on Modest Data"). One home each.
- Pattern to check across the chapter: an experiment designed around a quantity that is easy to compute (a cosine curve, a slope) is read as if it measured what the model knows or why. The neighbouring sections have the same tendency (the bend in `scaling-laws.md`, the feedback loop in `moe.md`).
- Pattern to preserve: the matched-budget comparison with named confounds. It is the template for every A/B in this part.
- Figure `mdl-transformers-vit.svg` is clear; the token row and the "37 tokens" and "96 x 96 image" labels are small relative to the block labels (house rule 2).
