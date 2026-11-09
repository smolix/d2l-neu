# Padding, Stride, and Dilation

Source: `chapter_convolutional-neural-networks/padding-and-strides.md`, about 2,800 words of prose (220 of them in framework tabs), 7 exercises (about 700 words).

## Verdict

A section on shape arithmetic that manages to be about something: how large each output is and how far each unit sees. Each of the three settings is introduced by the need it meets, each formula is followed by a sentence that reads it, and the receptive-field thread (linear growth at stride 1, geometric with stride, geometric at full resolution with dilation) gives the section a spine that the legacy version lacked. The zero-padding experiment (a uniform image comes back with a 9/6/4 frame) and the strides-against-dilations figure are the high points. The main weakness is a notation choice inherited from the first edition: $p$ counts padding in total over both sides, while every code cell, the phrase "padding 1", and all the outside literature count per side. The text has to stop twice to translate. The second weakness is proportion: dilation, which the Basics part barely uses again, takes a fifth of the prose and four of the seven exercises.

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|---|---|---|---|---|
| 4 | 4 | 4 | 4 | 3 |

## What works

- The opening states two opposed needs as questions and gives each a number: "a $240\times 240$ image comes out as $200\times 200$. The stack has removed 30% of the image's area, all of it at the border."
- The reading of the stride formula: "The fraction counts the steps the window can take and the $+1$ counts its starting position." And the corollary "stride $s$ divides the resolution by $s$, rounding up."
- Why kernels are odd: "For an even $k_\textrm{h}$ ... the output is shifted by half a pixel. This is why CNNs almost always use odd kernels".
- The subsection "Padding Values". A four-line experiment whose output the reader can predict ("Each output then counts the image pixels in its window"), followed by the consequence: "the layer sees a dark frame around an image that has none." This is the section's MacKay moment: the method shown failing, on a picture.
- "Padding does not appear, because it changes which outputs exist, not how far each one sees." It anticipates the reader's question about the receptive-field formula.
- `fig_conv_rf_towers` (rendered and inspected): clean, the two panels share scale and shading, and the claim "Both top units see the same 15 input pixels" can be verified by counting. With the sentence "the strided stack returns exactly every fourth output of the dilated stack" it is the best figure and text pairing in the chapter.
- The patchify paragraph: one concrete forward link to vision transformers, in four sentences.

## Structural findings

- **Total padding against per-side padding.** See PAD-3. This is the finding most likely to trip a reader who then writes code.
- **Dilation is over-weighted for Basics.** About 550 words, one figure, half of a second figure, and exercises 4, 5, and 7 (two of which turn on "gridding", a term the text does not use). Within chapters 1 to 8, dilation is used again only in passing. The receptive-field comparison is good enough to keep, but the output-shape formula with dilation (`eq_conv_output_shape`) and the code check could shrink to a sentence, and the exercises should be rebalanced. See PAD-9 and the exercise notes.
- **The figure for panel (a) is far from its text.** "Stride and the Receptive Field" refers to "panel (a) of :numref:`fig_conv_rf_towers` below", which appears 100 lines later, at the end of the dilation section. The reader computes $r = 1+2+4+8 = 15$ without the picture that makes it obvious. Fix: either split the figure (panel (a) with the stride subsection, both panels again with dilation), or move the worked numbers $1+2+4+8$ next to the figure.
- **Drill cells.** Two of the shape checks (the $5\times3$ kernel; the $3\times5$ kernel with stride $(3,4)$) confirm formulas the reader already believes and print a `torch.Size`. See PAD-6.
- **Framework notes fragment the stride section.** Between the first stride example and the receptive-field subsection there are six tab blocks. They are correct and each is short, but the TensorFlow reader gets a different lesson (asymmetric `same` padding, `ZeroPadding2D`) from the others. Acceptable; the per-framework defaults could be one comment in the first code cell.

## Passage-level findings

**PAD-1** (minor, thought). Opening:

> design decision, and two needs pull in opposite directions.

The next paragraph ends "A third need is for one kernel to see a wide region". Two needs, then a third. Fix: "three needs" with the questions to match, or fold dilation into the second need (seeing far) as the alternative that keeps resolution, which is how the body treats it.

**PAD-2** (minor, explanation). Figure `img_conv_reuse` (rendered and inspected) and its caption:

> Interior pixels are covered $k^2$ times, corner pixels once.

The panels draw cells of unequal size: thin border cells and a large middle cell labeled 4 or 9. The counts are those of the smallest image that shows every case ($3\times3$ for the $2\times2$ kernel, $5\times5$ for the $3\times3$ kernel), and the stretched middle cell stands for the whole interior of a large image. Neither the caption nor the text says this, so the reader has to work out what a cell is. Fix: add to the caption "The stretched cells stand for all pixels at that distance from the border", or draw an actual $7\times7$ image with per-pixel counts.

**PAD-3** (major, explanation). Under "Zero Padding":

> If we add $p_\textrm{h}$ rows of padding in total, split between top and
> bottom, and $p_\textrm{w}$ columns in total, split between left and right,

The symbol $p$ is the total over both sides. Every code cell passes the per-side amount, the prose says "padding 1 on each side" and "padding $(0, 1)$ on each side", `lenet.md` says "pads by 2 on each side", and the standard references the chapter recommends (CS231n, Dumoulin and Visin) write $n + 2p - k$. The text has to translate in a parenthesis ("$p_\textrm{h}=p_\textrm{w}=2$ in total; padding arguments in code count the rows or columns on each side, here 1"), and later says "When height and width use the same setting, we speak of padding $p$", which leaves open whether "padding $p$" is 1 or 2 for the layer just built with `padding=1`. A reader who memorizes the formula from this page and types `padding=p` gets the wrong shape. Fix: let $p$ be the padding on each side, as in code, and write the output size as $\lfloor (n + 2p - k)/s \rfloor + 1$ throughout the chapter (`pooling.md` and the exercise in `channels.md` use the total convention too). Shape-preserving padding becomes $p = (k-1)/2$, which is the number the reader types. Mention asymmetric padding for even kernels in the one paragraph that needs it.

**PAD-4** (minor, language). Under "Boundary Pixels":

> center of its input, and the next layer receives an input from which the border has been
> cut away.

The paragraph has just argued that border pixels are *underused*; this clause returns to the earlier point that they are *cut away*, and the two get blurred. Fix: end the paragraph at "barely reaches the next layer." The shrinking was the subject of the opening.

**PAD-5** (moderate, explanation). Under "Strided Convolution", and figure `img_conv_stride` (rendered and inspected):

> outputs are $0\times0+0\times1+1\times2+2\times3=8$, whose window lies two
> columns to the right of the first window, and
> $0\times0+6\times1+0\times2+0\times3=6$, whose window lies three rows below
> it.

The figure shades two windows and two outputs in the same blue, so the reader cannot tell which window gives 8 and which gives 6 without redoing both sums; the "first window" that both are measured from is not shaded at all. The strides 3 and 2 are also a strange first example: no network uses them, and the reader must hold two different numbers for two axes while learning what a stride is. Fix: redraw with stride 2 on both axes, shade one window and its output (as every other figure in the chapter does), or use two colors; then mention unequal strides in a sentence.

**PAD-6** (minor, pedagogy). Under "Strided Convolution":

> The settings can also differ between the axes. A $3\times 5$ kernel with
> padding $(0, 1)$ on each side and stride $(3, 4)$ gives

A drill with numbers nobody would choose, whose output is `torch.Size([2, 2])`. The same holds for the $5\times3$ padding example. Each costs four code tabs. Cut both cells and keep one sentence: "Height and width may use different kernel sizes, paddings, and strides; the formula applies to each axis separately."

**PAD-7** (minor, explanation). Under "Stride and the Receptive Field":

> with depth. After a layer with stride 2, neighboring positions of its output
> lie two input pixels apart, and a $3\times 3$ kernel applied to that output
> reaches $2\times 2=4$ input pixels further instead of 2.

"$2\times 2=4$" reads as a kernel size in a chapter full of "$2\times2$" windows. Write "reaches $(3-1)\cdot 2 = 4$ input pixels further".

**PAD-8** (minor, thought). Under "Stride and the Receptive Field":

> later kernels, so a downsampling network's receptive field grows
> geometrically with depth, which is how classifiers see the whole image after
> a few dozen layers.

Geometric growth holds only while the network keeps striding; a real classifier strides four or five times, after which growth is linear again with a large step. "A few dozen layers" also sits oddly beside the next section's count of six convolutions and two poolings for a $32\times32$ patch. Fix: "Each stride doubles the reach of every later kernel, so a network that halves its resolution five times sees 32 times further per layer at the top than at the bottom."

**PAD-9** (moderate, pedagogy). Under "Dilated Convolution":

> $$\left(\left\lfloor\frac{n_\textrm{h}+p_\textrm{h}-d_\textrm{h}(k_\textrm{h}-1)-1}{s_\textrm{h}}\right\rfloor+1\right)\times\left(\left\lfloor\frac{n_\textrm{w}+p_\textrm{w}-d_\textrm{w}(k_\textrm{w}-1)-1}{s_\textrm{w}}\right\rfloor+1\right).$$

The summary calls this the "one formula" for all three settings, and it is the hardest display in the section to read: eight subscripted symbols, twice. The idea is one line: a dilated kernel behaves like a dense kernel of size $k' = d(k-1)+1$. Fix: state the single-axis formula once, $\lfloor (n + 2p - k')/s \rfloor + 1$ with $k' = d(k-1)+1$ (using the per-side $p$ of PAD-3), and say "the same holds for the width". The section's formulas all shrink by half if written for one axis after the first.

**PAD-10** (minor, language). Under "Dilation and the Receptive Field":

> with nonlinearities between the layers: dilation is a strided stack that
> discards nothing.

A good sentence slightly misworded: a dilation is not a stack. "The dilated stack is the strided stack with nothing discarded."

**PAD-11** (minor, explanation). Under "Dilation and the Receptive Field":

> A network that downsamples with strides can instead restore
> the resolution at its end with learned upsampling, the transposed
> convolution of :numref:`sec_transposed_conv`.

A second forward reference to a section in the Image Models part (the first is in `conv-layer.md`). The reader of Basics cannot follow it and does not need to. Cut the clause after "learned upsampling".

## Exercises

Seven problems; the balance is off.

- Exercise 1 (mirror padding): good, and part 3 (a ramp that zero padding bends and mirror padding flattens) is a fine small discovery.
- Exercise 2.2: "give one statistical benefit of a stride larger than 1". A legacy phrase; it is not clear what "statistical" asks for (a larger receptive field means each unit pools more evidence?). Rewrite to ask what the larger receptive field lets a unit decide that the smaller one cannot.
- Exercise 3 (stride as sampling): good, but it relies on "the banded-matrix form of :numref:`subsec_conv_matmul`", which that subsection only names (see CONV-7).
- Exercises 4 and 7 (gridding; gridding-free dilations) are a pair on a topic outside the text, and 5 (strides against dilations) is a third dilation problem. Keep 5, which proves the claim the text makes about `fig_conv_rf_towers`. Merge 7 into 4 as a marked last part, or drop 7.
- Exercise 6 (fractional stride): a nice lead-in to transposed convolution, properly marked as hard.
- Missing: a plain problem on the main formula in a realistic setting (trace the shapes of a $224\times224$ image through a given five-layer stem and give the receptive field of the last unit). That is the skill the section exists to teach, and the reader will need it in the next chapter.

## Still open from October

October asked to keep the exponential receptive-field growth; it is kept and improved by the figure. No October finding for this section remains open.
