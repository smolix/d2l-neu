# Chapter 6, Convolutional Neural Networks: writing review

Reviewed 2026-11-09 from the `.md` sources in `chapter_convolutional-neural-networks/`, every line up to the `<!-- slides -->` marker, with the stored PyTorch outputs and the rendered figures. Per-notebook reviews:

- [index](ch06-cnn/index.md)
- [why-conv](ch06-cnn/why-conv.md) (From Fully Connected Layers to Convolutions)
- [conv-layer](ch06-cnn/conv-layer.md) (Convolutions for Images)
- [padding-and-strides](ch06-cnn/padding-and-strides.md) (Padding, Stride, and Dilation)
- [channels](ch06-cnn/channels.md) (Multiple Input and Multiple Output Channels)
- [pooling](ch06-cnn/pooling.md)
- [lenet](ch06-cnn/lenet.md)

## Verdict

This chapter has been rewritten since October and is now good textbook prose throughout: plain sentences, almost no mannerisms (no em dashes, one "exactly" per section at most, no stagecraft), every section opened by a question the reader can hold, every formula read back in a sentence, every number in the prose that I checked against the stored outputs correct. Its arguments are sound; I found no mathematical error. It is a solid 4 on language and correctness. It is not yet at the level of the model texts for two reasons. First, the conceptual heart, the derivation of convolution, is told in algebra where it could be told in a picture, and the chapter as a whole shows the reader very few *images* for a chapter about images. Second, the arc that the chapter sets up (an MLP ignores pixel order; equivariance against invariance; convolution as the layer that knows two facts about images) is not closed: the final section reports one to three points of accuracy over the MLP and stops, and the operation that turns equivariance into invariance (global pooling) is named only in a Summary. The chapter derives well and demonstrates less than it derives.

## Scores

| Notebook | Explanation | Thought | Pedagogy | Language | Enjoyment | Prose words |
|---|---|---|---|---|---|---|
| index | 4 | 4 | 4 | 4 | 4 | 400 |
| why-conv | 4 | 4 | 3 | 4 | 3 | 2,650 |
| conv-layer | 4 | 3 | 4 | 4 | 4 | 2,600 |
| padding-and-strides | 4 | 4 | 4 | 4 | 3 | 2,800 |
| channels | 4 | 4 | 3 | 4 | 3 | 2,270 |
| pooling | 4 | 3 | 3 | 4 | 3 | 2,000 |
| lenet | 4 | 4 | 3 | 4 | 3 | 1,830 |

No section earns a 5 in any column: nothing here would be out of place in a good textbook, and nothing yet has the inevitability of a page of Cover and Thomas or the pleasure of MacKay. The first half of `conv-layer.md` and the shift experiment in `pooling.md` come closest.

## The chapter question: is the derivation as clear and as visual as it could be?

**As clear: nearly. As visual: no.**

What the text does. `why-conv.md` writes a fully connected layer between two $1000\times1000$ grids with a fourth-order weight tensor, relabels weights by offset ($\mathsf{W} \to \mathsf{V}$), proves that translation equivariance holds if and only if $\mathsf{V}$ does not depend on the location of the hidden unit, and then truncates the offsets for locality. Counts fall from $10^{12}$ to $4\times10^6$ to $(2\Delta+1)^2$. The proof is short and its core is a good idea, well put: "a single bright pixel reveals the weights". The equivariance and invariance definitions are precise and kept apart. Judged as a derivation in prose, this is a 4, held back only by doing the hardest case first.

What an "Illustrated" treatment would do, and what the chapter already has the material for:

1. **Do it in one dimension, on eight samples, with the matrix in view.** The figure `fig_why_conv_weight_sharing` (rendered: three $8\times8$ weight matrices, "64 weights", "15 weights", "3 weights") is exactly the right picture. The text gives it one sentence and never returns to it; the numbers 64, 15, and 3 do not appear in the prose. In one dimension a weight has two indices, the relabeling by offset is "read the matrix along its diagonals", and the proof is visible: column $k$ of the matrix is the response to a single bright sample at $k$; equivariance says column $k+1$ is column $k$ moved down by one; so the matrix is constant along diagonals. That is the whole theorem, seen. The two-dimensional statement then follows by "positions and offsets become pairs". See [WHY-5](ch06-cnn/why-conv.md), the most important finding in the chapter, with [WHY-6](ch06-cnn/why-conv.md) (the single-hue palette makes panel (b) hard to verify) and [WHY-7](ch06-cnn/why-conv.md).
2. **Draw the definition.** There is no picture of $f(T_v\mathbf{X}) = T_v f(\mathbf{X})$: an input with one distinctive patch and its response map, and the same pair after a shift. (Author's note, 2026-11-09: the Waldo photograph is a deliberate joke and stays; the following criticism of it is withdrawn.) The Waldo photograph was meant to play this role and cannot: it shows a gathering in which every person is dressed as Waldo, and the "detector" image is the same photo washed out with two boxes left clear, one around a woman. A reader who accepts the caption's invitation ("Can you find Waldo") finds the example undone. See [WHY-13](ch06-cnn/why-conv.md).
3. **Show a kernel sliding and what comes out, on a real image.** The edge detector in `conv-layer.md` is the right example and is shown as printed tensors. No photograph is ever filtered. See [CONV-3](ch06-cnn/conv-layer.md).
4. **Say why the hidden layer has the shape of the image.** The step from "an MLP with 1000 hidden units" to "one hidden unit per pixel" is the conceptual move that makes equivariance statable, and it is made silently. See [WHY-2](ch06-cnn/why-conv.md).
5. **Close the loop with the banded matrix.** `conv-layer.md` mentions that a convolution can be unrolled into "a sparse, banded matrix" and defers it to the exercises. That matrix is panel (c) of the figure from the previous section. Saying so would tie the implementation back to the derivation in one sentence. See [CONV-7](ch06-cnn/conv-layer.md).

**Do the figures carry their share?** I rendered all fifteen SVG figures and the three raster images.

- Carrying real weight: `conv-rf-towers` (strides against dilations; the best figure in the chapter, and the text uses it fully), `lenet` (to scale, with a shirt), `conv-groups`, `arch-conv-dilation`, and the classic numeric panels `correlation`, `conv-pad`, `conv-multi-in`, `pooling`, each of which the text works through number by number.
- Good but underused: `why-conv-weight-sharing` (above); `conv-receptive-field` (fine, but one-dimensional while the text speaks of $7\times7$).
- Flawed: `conv-stride` (two windows and two outputs in one color, and strides 3 and 2, [PAD-5](ch06-cnn/padding-and-strides.md)); `arch-conv-depthwise` (draws one output channel, the one case where the separable layer costs more, [CHAN-7](ch06-cnn/channels.md)); `conv-reuse` (stretched cells unexplained, [PAD-2](ch06-cnn/padding-and-strides.md)).
- Misplaced: `field-visual`, the only figure in the chapter with real kernels and real responses, sits in a neurophysiology paragraph under "Effective Receptive Fields" ([CONV-9](ch06-cnn/conv-layer.md)).
- Missing: a picture of equivariance; a filtered photograph; a picture of the pooling shift experiment ([POOL-5](ch06-cnn/pooling.md)); learned first-layer feature maps of the trained LeNet.

So the mechanical figures (shape arithmetic) are complete and good, and the conceptual figures (what a convolution *means* and *finds*) are one underused matrix plot and a photograph that works against the text. The chapter is illustrated where it is easy and bare where it matters.

## Arc and order

The order is right and should stay: derive the layer; compute and learn it; control its shape and reach; add channels; summarize; assemble. Each section ends by naming what is still open, and the next one picks it up; these closings are consistently good ("Two limits remain: every layer shrinks the image ... and a single-channel kernel detects only one pattern").

Three problems with the arc:

- **The invariance thread is dropped.** `why-conv.md`: the invariant answer comes from "a final step that aggregates the equivariant map over all locations". `pooling.md`: local pooling gives tolerance, "It does not make it invariant". `lenet.md`: the head is flatten-and-dense, which is not invariant, and the text does not say so. Global pooling, the missing step, is defined in the Summary of `pooling.md` and used in a LeNet exercise. See [POOL-8](ch06-cnn/pooling.md).
- **The ending is weaker than the beginning.** See finding 2 below.
- **Proportion.** Dilation (550 words, 3 exercises) and grouped and depthwise convolutions (450 words, 2 figures, 2 exercises) together take about 1,000 words on material that Basics barely uses again and that chapter 7 partly re-teaches, while global pooling gets one sentence and the trained network is never opened.

Dependence on other chapters is light and well handled: the MLP result is restated with its numbers, Xavier initialization and activation memory are cited with a restated fact. Three forward references leave the Basics part and should go: `sec_sentiment_cnn` (Attic) in `why-conv.md`, and `sec_transposed_conv` (Image Models) twice.

## The most important findings

1. **Run the derivation in one dimension on the weight-matrix figure before the two-dimensional algebra** ([WHY-5](ch06-cnn/why-conv.md), with WHY-2, WHY-6, WHY-7). Highest value for the least text.
2. **Give the chapter a demonstration to end on** ([LEN-2](ch06-cnn/lenet.md), [IDX-1](ch06-cnn/index.md), [IDX-3](ch06-cnn/index.md)). The claim that an MLP is indifferent to a fixed shuffling of the pixels is made three times and never tested. Train both models on permuted Fashion-MNIST in `lenet.md`: the MLP should not notice and LeNet should lose its edge. That one experiment replaces three assertions and shows what the convolutional prior is worth, which a gap of one to three points does not.
3. ~~Replace the Waldo images with a figure of equivariance~~ ([WHY-13](ch06-cnn/why-conv.md)). **Withdrawn 2026-11-09 (author):** the Waldo photographs are a deliberate joke (a crowd in which everyone is dressed as Waldo) and stay as they are. A figure of equivariance could still be added beside them.
4. **Teach global pooling in the body of `pooling.md`** with the same single-response experiment ([POOL-8](ch06-cnn/pooling.md)), and say in `lenet.md` which parts of LeNet break equivariance and invariance (currently its exercise 5).
5. **Let $p$ mean padding per side** ([PAD-3](ch06-cnn/padding-and-strides.md)). The formula on the page and the argument the reader types differ by a factor of two.
6. **Move grouped and depthwise-separable convolutions to chapter 7** ([CHAN-6](ch06-cnn/channels.md)) and trim dilation ([PAD-9](ch06-cnn/padding-and-strides.md)).
7. **Decide what `conv-layer.md` is about** ([CONV-6](ch06-cnn/conv-layer.md), [CONV-7](ch06-cnn/conv-layer.md), [CONV-9](ch06-cnn/conv-layer.md)). The im2col paragraph of hedges, the banded-matrix paragraph that defers to exercises, and the neurophysiology under the wrong heading are the weakest 500 words in the chapter.
8. **Show images** ([CONV-3](ch06-cnn/conv-layer.md), [POOL-5](ch06-cnn/pooling.md), and the structural note in [lenet](ch06-cnn/lenet.md)): the edge map as a picture, a filtered photograph, the learned feature maps of LeNet.
9. **Tame the LeNet training story** ([LEN-4](ch06-cnn/lenet.md), [LEN-5](ch06-cnn/lenet.md)). The sigmoid configuration needs a learning rate of 2.0, an unshown comparison run, and a caveat about epochs at chance. Show both learning rates in one plot, or train the ReLU variant beside it.
10. **Keep Summaries to what was taught** ([WHY-14](ch06-cnn/why-conv.md), [POOL-8](ch06-cnn/pooling.md), [POOL-9](ch06-cnn/pooling.md), [LEN-6](ch06-cnn/lenet.md)).

## Recurring patterns

- **New material in Summaries: 4 instances in 3 sections.** `why-conv.md` ("cannot represent a classifier whose label changes when the same object moves from the left half of the image to the right"), `pooling.md` (strided convolution in place of pooling; global average pooling), `lenet.md` (the table of modern substitutions). In each case the author has thought of the right closing point and put it after the section ended. Likely book-wide after the October revision, which standardized Summaries.
- **The same fact stated in several sections.** Pixel shuffling (3: index, `why-conv.md`, `lenet.md`). "libraries compute cross-correlation and call it convolution" (2, under identical headings). The definition of feature map (3: `why-conv.md`, `conv-layer.md`, `channels.md`). Receptive field (defined in `conv-layer.md`, restated in `padding-and-strides.md` and `pooling.md`, with three slightly different wordings, [CONV-11](ch06-cnn/conv-layer.md)). Hubel and Wiesel, and the Neocognitron (3 each). Restating is the house rule and mostly helps; the shuffling claim and the cross-correlation convention are over the line.
- **Notation drift.** Channel order: channels-last with $[\mathsf{K}]_{a,b,c,d}$ in `why-conv.md`, channels-first with $\mathbf{K}_{o,c}$ elsewhere ([WHY-12](ch06-cnn/why-conv.md), [CHAN-2](ch06-cnn/channels.md)). Padding: $p$ is a total in formulas and per side in words and code ([PAD-3](ch06-cnn/padding-and-strides.md)). Kernel extent: $\Delta$ in `why-conv.md`, $k_\textrm{h}, k_\textrm{w}$ after. Image size: $1000$, then $1024$, then $224$. Pixels: $[\mathbf{X}]_{i,j}$, then $x_{i,j}$ and $f(i,j)$ in one paragraph of `conv-layer.md`. Output index: $d$, then $o$.
- **Two-axis formulas.** Every shape formula is written for height and width in full, up to eight subscripted symbols twice in one display ([PAD-9](ch06-cnn/padding-and-strides.md)). One axis plus "the same for the width" would halve the visual load of `padding-and-strides.md`.
- **Drill cells that print a shape.** Three in `padding-and-strides.md` and `pooling.md` use kernel, stride, and padding combinations nobody would choose and print `torch.Size([2, 2])` or a repeat of the previous output ([PAD-6](ch06-cnn/padding-and-strides.md), [POOL-7](ch06-cnn/pooling.md)).
- **Framework tab notes.** About 900 words across the chapter, heaviest in `pooling.md` (12 notes in one subsection) and `padding-and-strides.md`. Each is correct and short. Together they make the mechanical subsections read as API documentation. Most could be code comments. The prose outside the tabs is framework-neutral, which is good; the one place it is not is the four-dimensional input layout, introduced in a code comment in `conv-layer.md` ([CONV-5](ch06-cnn/conv-layer.md)).
- **Scattered history.** One-sentence attributions (Zhang et al., time-delay networks, Neocognitron, Riesenhuber and Poggio, Yamaguchi et al.) are dropped into derivations and definitions ([WHY-8](ch06-cnn/why-conv.md), [POOL-3](ch06-cnn/pooling.md)). The chapter introduction tells the history well once; the rest can go or be gathered.
- **Language tics: essentially none.** Em dashes 0. "exactly" 2 in 14,500 words. Semicolons 34, mostly sound. No stock nouns from the guide's list. A mild habit of the colon followed by a restatement ("it answers whether the feature is present somewhere in the window" after a colon, and the like) recurs and is harmless. The writing guide has not pushed the prose in a bad direction here, with one exception: the rule "each output is interpreted" has produced a few sentences that interpret nothing ([POOL-4](ch06-cnn/pooling.md), [CHAN-8](ch06-cnn/channels.md)).
- **A missing standard term.** "Inductive bias" does not occur in the chapter's prose, although `why-conv.md` describes it precisely.

## What to cut and what is missing

Cut or move (about 1,900 words of prose, plus cells):

- Grouped and depthwise-separable convolutions to `sec_efficient_cnns` (450 words, 2 figures, 1 cell, 2 exercises).
- Most of "Convolution as Matrix Multiplication" (keep one paragraph and the check; about 250 words saved) and the PyTorch-only `F.unfold` aside.
- The neurophysiology paragraph and the Field figure from "Effective Receptive Fields" (150 words), or move to the chapter introduction.
- The "Channels" subsection of `why-conv.md` down to its idea (250 words saved); the continuous convolution integral.
- Half of "Padding and Stride" in `pooling.md` (300 words, 2 cells); the two anisotropic drill cells in `padding-and-strides.md`.
- The dilation shape formula and its code check, to a sentence (150 words).
- The 1998-against-teaching-version detail in `lenet.md` (80 words) and the Summary table.
- Exercises: `channels.md` 4 (strip buffering) and 2 (replace); `padding-and-strides.md` 7 (merge into 4); `conv-layer.md` 4 (merge into 6).

Missing (about 900 words and four figures or cells):

- The one-dimensional derivation on the weight-matrix figure.
- A figure of equivariance.
- The edge example and one photograph as images.
- Global pooling, defined and tested.
- The permutation experiment (or a shift test) on the trained MLP and LeNet.
- Learned feature maps of LeNet for two inputs.
- The term "inductive bias", and one concrete case where the convolutional assumption is wrong.

The exchange leaves the chapter shorter and with a stronger line.

## What is best and should set the model

- **The first half of `conv-layer.md`**: opening on a nameable problem (find the vertical edges), a hand-built solution with two numbers, a negative control (the transposed image gives zeros), then learning the same two numbers by gradient descent, with the loss predicted before it is printed ("each of the 12 edge positions misses its target $\pm 1$ by about 1"). This is the guide's sequence executed without visible effort.
- **"Tolerance to Small Shifts" in `pooling.md`**: a claim that most books assert ("pooling gives translation invariance") reduced to a three-line experiment that shows it is half true, and a conclusion stated at the strength the evidence supports.
- **"Padding Values" in `padding-and-strides.md`**: a uniform image, a kernel of ones, and the frame of 4s and 6s that zero padding invents.
- **The cost table of `lenet.md`** and the two sentences that read it.
- **The restriction paragraphs**: "The reduction ... comes from restricting the class of functions" (`why-conv.md`) and "The savings come from a restriction" (`channels.md`). Each new layer is presented with what it cannot do.
- **Section closings** that name the open problem for the next section, in every notebook.
- **The exercise sets of `why-conv.md` and `lenet.md`**: conceptual problems with a point (spectrogram axes, the nine-cell task, rotation equivariance) and experiments with a question (predict, then measure).

## Status of the October findings

All five October findings for this chapter are closed: the legacy prose and grammar errors are gone; the key step of the derivation is proved and $\mathsf{V}$ is used consistently; padding and stride are no longer used before they are defined; LeNet's run is read out and compared with the MLP; the MNIST aside has been rewritten and I found no error in it. The three passages October asked to keep are kept.

## Not checked

- Only the PyTorch outputs were compared with the prose. Sentences written to cover all four tabs ("zero or at most about $10^{-6}$", "about 88--90%", the epochs at chance) were not verified against the TensorFlow, JAX, and MXNet stores.
- The unshown run at learning rate 0.1 in `lenet.md` ("about 73% after 10") cannot be checked from the store.
- I did not run the experiments I propose (permuted pixels, shifted test images, global pooling). The expected outcomes are standard, but the permutation result in particular should be confirmed in all four frameworks before prose is written around it.
- The rendered HTML pages were not opened; figures were rendered from `img/` directly, so figure sizing and placement in the page were not judged.
- Citations were not checked against the bibliography.
