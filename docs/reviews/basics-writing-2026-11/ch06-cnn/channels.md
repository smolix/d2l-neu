# Multiple Input and Multiple Output Channels

Source: `chapter_convolutional-neural-networks/channels.md`, about 2,270 words of prose (220 in framework tabs), 6 exercises (about 700 words).

## Verdict

Clear, compact, and accurate. The first half builds the multi-channel layer from the single-channel `corr2d` in two one-line functions, with a figure and a hand-checked number for each step; the red-minus-green kernel is a sharp example of why channels must be summed. The cost paragraph gives the reader the one law that governs every later architecture decision (cost grows with $c_\textrm{i} c_\textrm{o}$ times the output area). The second half (the $1\times1$ convolution, bottlenecks, grouped, depthwise, and depthwise-separable convolutions) is well written but answers a problem the reader does not yet have: at this point they have never built or trained a convolutional network, and the largest layer they have run has one channel. The most important fix is structural: keep the $1\times1$ convolution here, and move grouped and depthwise-separable convolutions to `sec_efficient_cnns` in the next chapter, where the text explains them a second time anyway.

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|---|---|---|---|---|
| 4 | 4 | 3 | 4 | 3 |

## What works

- The opening question: "How does one layer read several input channels and produce several output channels, and what does that cost?" and the example that makes the first half necessary: "A boundary between a red and a green region of equal brightness is visible only when the channels are compared."
- The red-minus-green $1\times1$ kernel under "Multiple Input Channels": "positive in red regions, negative in green ones, and zero in gray ones, however bright." A concrete kernel the reader can compute in their head. It also plants the $1\times1$ convolution two pages before it is named.
- The test of the three-channel kernel is read, not just run: "Adding 1 to every weight adds to each output the sum of its window over both input channels, which is 20 for the first element, so the second channel starts at $56 + 20 = 76$." (Stored output: 56 and 76. Correct.)
- "it is a fully connected layer applied to each pixel separately, with weights shared across pixels." The right one-sentence meaning for the $1\times1$ convolution.
- The bottleneck arithmetic, with its cost stated: "Such a *bottleneck* can only represent maps that pass through 64 channels".
- "The savings come from a restriction ... $\mathbf{K}_{o,c} = P_{o,c}\,\mathbf{D}_c$: all output channels see the same spatial filter of each input channel, only scaled". This says what a depthwise-separable layer cannot do, in one formula. It echoes the "restricting the class of functions" paragraph of `why-conv.md` and is the kind of sentence the guide asks for (a boundary for every definition).
- `fig_conv_groups` (rendered): the channel-connectivity matrices are the right picture, and they rhyme with the weight matrices of `fig_why_conv_weight_sharing`.

## Structural findings

- **Two ideas.** "How a layer handles many channels" and "how to make channel mixing cheaper" are separate ideas. The first belongs here. The second is architecture design, the subject of chapter 7, whose `efficient-convnets.md` opens by defining depthwise and pointwise convolutions again and citing `eq_depthwise_sep_ratio` from here. A reader of this chapter meets grouped convolution, ResNeXt, depthwise convolution, Xception, MobileNet, and the bottleneck of ResNet before LeNet. See CHAN-6.
- **The cost paragraph is one block of 190 words with seven numbers.** It is the conceptual center of the section and deserves to be two or three paragraphs. See CHAN-4.
- **Channel order.** This section is channels-first; `why-conv.md` was channels-last. See WHY-12 and CHAN-2.

## Passage-level findings

**CHAN-1** (minor, pedagogy). Opening, second paragraph:

> The rest of the
> section reduces that product: the $1\times 1$ convolution mixes channels
> without looking at neighbors, and grouped and depthwise convolutions restrict
> which channels are mixed.

A roadmap that previews three terms before any is defined. Fix: "The rest of the section asks how to make that product smaller." (If CHAN-6 is adopted, the sentence shortens further.)

**CHAN-2** (moderate, explanation). Under "Multiple Output Channels":

> :numref:`sec_why-conv`, written channel by channel: $\mathbf{K}_{o,c}$ is its
> $k_\textrm{h}\times k_\textrm{w}$ slice for input channel $c$ and output
> channel $o$ (there called $d$).

The reconciliation is needed only because the two sections chose different index names and orders for the same tensor. The four tab notes that follow ("PyTorch stores this kernel as ...", "Keras stores this kernel as ..., the index order of :eqref:`eq_conv-layer-channels`") then tell the reader that both orders exist in practice, which is true and useful, but it is three versions of one fact. Fix: adopt $c_\textrm{o}\times c_\textrm{i}\times k_\textrm{h}\times k_\textrm{w}$ and the letter $o$ in `why-conv.md` as well, delete "(there called $d$)", and keep one sentence here on the two storage layouts.

**CHAN-3** (minor, pedagogy). Under "Multiple Output Channels":

> jointly, so one feature, such as an edge at some orientation, may be spread
> over several channels instead of occupying one.

A true and slightly deflating remark, right after the reader has been told that each output channel is "one detector's responses". It is not used again. Either support it with a sentence on why (any invertible mixing of the output channels can be undone by the next layer, so nothing singles out the "pure" basis), or cut it.

**CHAN-4** (moderate, explanation). Under "Parameters and Cost":

> are applied at each. The first layer from 3 to 64 channels with $3\times 3$
> kernels in :numref:`sec_why-conv` has 1,792 parameters. A layer deeper in,
> from 256 to 256 channels with $3\times 3$ kernels, has 590,080 parameters,

From here the paragraph runs through 1,792; 590,080; $56^2\cdot 589{,}824 \approx 1.8\times10^9$; $224^2\cdot 64 \approx 3.2\times10^6$; and a reference to activation memory in chapter 5. All correct (I checked each product). The reader gets three distinct lessons in one breath: parameters scale with $c_\textrm{i}c_\textrm{o}$; computation scales with that times the output area; activation memory is largest where parameters are fewest. Fix: three short paragraphs, one per lesson, or a four-row table (layer, parameters, multiply-adds, activations) for "first layer" and "deep layer". The last lesson is striking (about $3.2\times10^6$ stored values per example against 1,792 parameters) and is lost in the middle of the block. The number 590,080 switches to 589,824 (without biases) in the next line with no comment; use one of them.

**CHAN-5** (minor, explanation). Under "The $1\times 1$ Convolution":

> channels, a restriction that networks such as ResNet accept for the savings.

ResNet has not been introduced and gets no reference. Either add `:numref:` to its section or say "a restriction that many deep networks accept (:numref:`chap_modern_cnn`)". Similarly "ResNeXt" under "Grouped Convolution" and "Efficient architectures" at the end.

**CHAN-6** (major, pedagogy). Section "Grouped, Depthwise, and Depthwise-Separable Convolutions":

> The factor $c_\textrm{i} c_\textrm{o}$ arises because every output channel
> reads every input channel. A *grouped convolution* splits the input channels
> into $g$ groups

The writing is good. The placement is the problem. (a) Motivation: the reader has run no network in which the factor $c_\textrm{i}c_\textrm{o}$ hurt; LeNet, two sections on, has 6 and 16 channels and nobody would group them. (b) Use: nothing in the rest of this chapter uses grouped or depthwise layers. (c) Duplication: `efficient-convnets.md` in chapter 7 re-defines "depthwise" and "pointwise" and quotes the ratio from here. (d) Density: in about 450 words the subsection introduces grouped, depthwise, pointwise, and depthwise-separable convolutions and cites four architectures. Fix: move the section, with `fig_conv_groups`, `fig_conv_depthwise`, the ratio, the factored-kernel paragraph, and the 128-channel code check, to the start of `sec_efficient_cnns`, where the problem (a network for a phone) has been posed. Leave here one closing sentence: "The factor $c_\textrm{i} c_\textrm{o}$ arises because every output channel reads every input channel; :numref:`sec_efficient_cnns` shows layers that give this up." This section then has one idea and ends at about 1,600 words. Exercises 5 and 6 move with it.

**CHAN-7** (minor, explanation). Figure `fig_conv_depthwise` (rendered and inspected) and its caption:

> Top: a dense convolution, in which every output channel reads every input channel. Bottom: a depthwise convolution filters each channel separately, and a pointwise $1\times 1$ convolution mixes the channels.

The figure draws a single output channel in both rows. With one output channel the dense layer has 27 weights and the separable pair 30, so the picture shows a case in which the separable layer is *more* expensive, and the saving (which comes from sharing the depthwise stage among many output channels) is invisible. Fix: draw two or three output channels in each row, so that the top row needs a second full $3\times3\times3$ kernel and the bottom row only a second $1\times1$ kernel.

**CHAN-8** (minor, language). After the $1\times1$ check:

> The total absolute difference over the 18 outputs is zero or at most about
> $10^{-6}$, the size of rounding errors in 32-bit floating-point arithmetic.

A sentence written to be true in four tabs at once. The stored PyTorch output is `0.0`. "The two results agree up to rounding error." is enough.

## Exercises

Six problems of uneven fit.

- Exercise 1 (composition of convolutions): very good, especially part 3 (is every $7\times7$ kernel a composition of three $3\times3$ kernels?), which makes the reader count degrees of freedom.
- Exercise 2 (cost and memory accounting): three parts of counting with general symbols, including "How does the count change when the padding is doubled?" This is drill. Replace by one concrete accounting task that uses the section's law: for a given four-layer stack, find the layer with the most parameters, the most multiply-adds, and the most activation memory, and explain why they are three different layers.
- Exercise 3 (multi-channel im2col): fine as a coding check; depends on the im2col subsection of `conv-layer.md` (see CONV-6).
- Exercise 4 (strip-buffered convolution): a cache-blocking puzzle unrelated to anything in the section. Cut.
- Exercises 5 and 6 (separable block budget; channel permutation between groups): both good, both belong with the grouped material if it moves (CHAN-6). Exercise 6 is ShuffleNet's idea posed as a problem and is properly marked hard.

## Still open from October

No October finding was specific to this section. The legacy prose is gone.
