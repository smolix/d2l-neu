# Blocks, Bottlenecks, and Branches: VGG, NiN, GoogLeNet

Source: `chapter_convolutional-modern/blocks.md`. About 3,500 words of prose before the exercises, 710 words of exercises.

## Verdict

A competent section with a good frame and a weak evidential core. The frame is right: the opening names two problems left by AlexNet (no rule for choosing layers; a head that holds 92% of the parameters), VGG answers the first, NiN the second, and a closing table compares four networks under one training procedure. The VGG third is the best: the $18c^2$ against $25c^2$ argument is short and complete. The weakness is that the comparison table, which is the payoff of the section, does not support the three "observations" drawn from it. The networks are trained at different input sizes, NiN's result ranges over four points and comes from a simplified last block, and the third observation credits GoogLeNet's design for a saving that comes from feeding it smaller images. GoogLeNet itself is the one place in this half where the text reads as catalogue: a paragraph of layer sizes and a 9-row tuple of hand-chosen channel counts for a design the section then says is "rare in current networks". Fix the comparison first.

## Scores

| Dimension | Score |
|---|:-:|
| Clarity of explanation | 4 |
| Clarity of thought | 3 |
| Pedagogy | 3 |
| Language | 4 |
| Ease and enjoyment | 3 |

## What works

- The opening turns AlexNet's ledger into a motivation: "That works for eight layers, but a designer who wants twenty needs a rule for choosing them." The second problem is stated with the numbers the reader just computed.
- Under "Stacks of $3 \times 3$ Convolutions": the halving argument ("224, 112, 56, 28, 14, 7, 3, 1, and the network is over after seven halvings") makes the reader feel why several convolutions per resolution are needed, before the block is defined. Then "they have $2 \cdot 9c^2 = 18c^2$ weights instead of $25c^2$" settles the kernel size. Problem, smallest case, general rule: this is the guide's sequence done well.
- "VGG is therefore a *family* of networks: VGG-11, VGG-16, and VGG-19 differ only in `arch`" is the transferable idea (a network as data), and the code makes it literal.
- The honest end of the VGG experiment: "Blocks organized the body of the network; the head is untouched." It hands the reader to NiN with a reason.
- Under "Global Average Pooling": the explanation of why the last ReLU must go ("once a class's pre-activations are negative at every position of every image, its logit is 0, its gradient is zero, and training can never bring it back") is the sort of detail a practitioner learns the hard way. And the translation argument ("A convolution commutes with translations ... an average over all positions does not change when its entries are permuted") is correct, short, and properly fenced ("Up to the borders of the image and the strides of the network").
- The worked reduction under "The Inception Block" ("$25 \cdot 192 \cdot 32 = 153{,}600$ weights ... 15,872 in total: a tenth") is the bottleneck idea in three lines of arithmetic.
- The last paragraph before the summary ("None of these networks normalizes its activations, and none is much deeper than twenty layers ...") is a real bridge: it names the devices these networks needed and promises their removal.
- I recomputed every parameter and multiply-add count in the section (VGG-11, quarter-width VGG, NiN, the Inception example). All are right.

## Structural findings

- **The comparison confounds architecture with input size.** AlexNet, VGG and NiN see $224 \times 224$ images; GoogLeNet sees $96 \times 96$. The table reports multiply-adds "at the training input size", so GoogLeNet's 276M is not comparable with the other three. At $224 \times 224$ GoogLeNet would perform about 1.5 billion multiply-adds, more than AlexNet. Either train all four at one size (at $96 \times 96$ AlexNet and NiN need a smaller first stride, so $224$ for all is simpler, or drop to a size all four accept), or give GoogLeNet's count at $224$ in the table and compare cost at equal input. See BLK-9.
- **NiN is not a stable enough experiment to teach from.** The text reports "84–88%" for NiN, says that 0.03 is "the largest at which NiN trains reliably", that "at 0.05 an occasional run stays near chance for several epochs", and that at a constant rate the result lies "anywhere between about 78% and 87%". This is the authors' lab notebook, and it tells the reader that the comparison is fragile. The likely cause is named in the text and then left alone: "ours narrows the whole block to ten channels", so the last $3 \times 3$ convolution squeezes 384 channels into 10 before two further layers. The guide's rule applies (§9: fix the experiment or remove it; do not keep it and explain what it fails to show). Use the paper's last block (wide until the final $1 \times 1$ convolution) and see whether NiN then trains as reliably as the others.
- **The cleanest test of the section's main claim is an exercise.** The claim is that the dense head is unnecessary. The direct test is one body with two heads: the quarter-width VGG as trained, and the same body with global average pooling and one linear layer. That is a one-line change to `VGG`, it removes 42.5M of 43.1M parameters, and the reader sees the accuracy barely move. It would replace the indirect and confounded argument through NiN. Exercise 1 does this for NiN; the text should do it for VGG.
- **GoogLeNet: keep the two ideas, shrink the network.** The section itself says which parts lasted: "reduced channels with $1 \times 1$ convolutions before expensive operations, and divided a network into stem, body, and head". The Inception block as a teaching vehicle for the first is fine. But the full GoogLeNet costs the reader a paragraph of stem specifications, a nine-entry `googlenet_arch` of 54 hand-tuned numbers that the text can only describe as "from the paper, which chose them by hand", and four framework implementations. A reader learns nothing from the tuple. Consider training a small Inception network (three or four blocks with a regular width rule) and stating GoogLeNet's layout in two sentences, or keep the full network and cut the prose description of the stem to what :numref:`fig_inception` and the code already show.
- **The word in the title is never used.** The title promises "Bottlenecks". The body never uses the word; the idea appears as "Reducing the channels before an expensive convolution recurs throughout this chapter." Name it where it is introduced: "This reduce, compute, restore pattern is called a *bottleneck*, and it recurs throughout the chapter (:numref:`subsec_resnext`, :numref:`sec_efficient_cnns`)." The ResNet section later uses "*bottleneck* block" as if new.
- **Parameter and multiply-add counts are asserted, never computed.** "Counted as for AlexNet, VGG-11 has 128.8 million parameters and performs 7.5 billion multiply-adds" and the same for NiN and GoogLeNet. The reader cannot check them without half an hour of arithmetic, and the exercises then ask for exactly that arithmetic. A ten-line function that walks a network with one input and returns parameters and multiply-adds per layer would make the chapter's central tool executable, would replace four asserted counts here and more in later sections, and is code that teaches.

## Passage-level findings

**BLK-1** (minor, thought). Section opening. Quote: "Three networks of 2013 and 2014 answered these two problems".
Two problems, three networks. GoogLeNet answers a third question that the section poses only later ("Which kernel size should a layer use?"). Either state three questions in the opening (a rule for depth, the cost of the head, the choice of kernel size) or say that GoogLeNet combines the first two answers and adds one idea.

**BLK-2** (minor, explanation). Under "Stacks of $3 \times 3$ Convolutions". Quote: "and GPU libraries implement them with".
The Winograd remark is a fact the section never uses and the reader cannot evaluate. Cut the clause and the citation.

**BLK-3** (minor, explanation). Same subsection. Quote: "and they apply three".
"three nonlinearities instead of one" is offered as an advantage without saying why. One clause: "so the stack can compute functions of its $7 \times 7$ window that a single linear $7 \times 7$ filter followed by one ReLU cannot".

**BLK-4** (moderate, language). Under "The VGG Network". Quote: "Its eight convolutional layers hold 9.2 million of the parameters and perform 98% of the arithmetic, eight"
This paragraph has eleven numbers in five sentences, and the next section ("Training") opens with five more. The reader needs three facts: VGG-11 costs eight times AlexNet's arithmetic; its head still holds 93% of its parameters; the first dense layer alone is 103 million. Put the rest in a two-row table beside AlexNet's totals, or in the output of the counting function proposed above. Number density is a pattern of this section (also "It performs 512 million multiply-adds per image, 470 million of them in the convolutions: about half of AlexNet's 938 million").

**BLK-5** (minor, language). Under "Training" (VGG), "Training" (NiN), and before the GoogLeNet run. Quotes: "minibatches of 128, and 10 epochs of SGD whose learning rate rises linearly"; "128, and 10 epochs of SGD whose learning rate warms up to a peak of 0.03"; "We train it with the same procedure and the".
The procedure is restated in full twice and in short once, and there are three sibling headings all called "Training". After the first, "trained as AlexNet was (:numref:`sec_alexnet`)" suffices.

**BLK-6** (moderate, explanation). Under "NiN: $1 \times 1$ Convolutions and Global Average Pooling". Quote: "Inserting fully connected layers earlier, to let the network compute more per position, would destroy the spatial structure and cost even more."
This is the whole motivation for the $1 \times 1$ convolutions, and the reader has not been given a reason to want "more per position". NiN bundles two ideas with different motivations. The head problem is well motivated. For the per-pixel MLP, state the problem first: a convolution followed by a ReLU computes, at each position, a thresholded linear function of its window; to compute a more complex function of the same window without widening it, add layers that look at one position only. Then define the $1 \times 1$ convolution.

**BLK-7** (minor, language). Same paragraph. Quote: "problems with layers that the reader has met in".
The book addresses the reader as "we" or "you". "With a layer we met in :numref:`subsec_1x1`."

**BLK-8** (moderate, thought). Under "Training" (NiN). Quote: "Of the peaks we tried, 0.03 is"
and, two paragraphs on, "NiN is also the network that depends most on the learning-rate schedule." Both passages narrate experiments the reader does not see (peaks 0.05 and 0.1, a constant rate of 0.01, a range of 78% to 87%). The guide classes this as material written for the authors (§4.5). It also undermines the table: if the result depends this much on the schedule, the three-way comparison is not measuring architecture. See the second structural finding. If the passage stays, cut it to one sentence where the peak 0.03 is first chosen (ALX-10).

**BLK-9** (major, correctness). Under "Comparing the Networks". Quote: "Third, GoogLeNet matches AlexNet with less than a third of its arithmetic, although it sees $96 \times 96$ images instead of $224 \times 224$."
"Although" has the logic backwards. GoogLeNet's arithmetic is less than a third of AlexNet's *because* it sees images with $(96/224)^2 \approx 18\%$ of the pixels; at equal input size it performs more arithmetic than AlexNet. The observation the data do support is different and still interesting: GoogLeNet matches AlexNet from images with a fifth of the pixels and with an eighth of the parameters. State that, or equalize the input sizes.

**BLK-10** (moderate, thought). Same paragraph. Quote: "First, the parameters of a dense head add little accuracy here: NiN and GoogLeNet, which replace it by"
The evidence is one network that matches AlexNet (at a different resolution) and one that ends "a few points below it". The quarter-width VGG, with a dense head holding 99% of its parameters, is the most accurate network in the table. A skeptical reader would conclude the opposite. The claim is true and important; it needs the direct test (third structural finding).

**BLK-11** (minor, thought). Same paragraph. Quote: "Ten epochs on upsampled $28 \times 28$ images are a small test; on ImageNet, in"
This is the right caveat in the wrong place. It follows three confident observations. Put it first ("These differences are small and the benchmark is easy; we read them only for ...") and then make the observations the table can bear.

**BLK-12** (minor, correctness). Under "The Inception Block". Quote: "comes from the film *Inception* and its line"
The paper cites the internet meme "we need to go deeper", which derives from the film; the sentence is usually said not to occur in the film in that form. "The name alludes to the film *Inception* and the catchphrase it inspired, 'we need to go deeper'" is safe. Keep the remark: it is the only moment of levity in the section and it is in the right place.

**BLK-13** (minor, language). Under "Stem, Body, and Head". Quote: "a $7 \times 7$ convolution with stride 2 and 64"
The stem is described as a list of five layers with their sizes in one 50-word sentence, which the code cell below repeats. Keep the definitions of stem, body and head (they are used for the rest of the chapter) and replace the inventory by "five layers that reduce the resolution eightfold".

**BLK-14** (minor, language). Class name. Quote: "`GoogleNet` takes this tuple as its argument `arch`".
The prose writes GoogLeNet (correctly, after LeNet) and the code `GoogleNet`. A reader searching the page for one finds half the mentions. Rename the class or note the spelling once.

**BLK-15** (minor, explanation). Under "Comparing the Networks". Quote: "Grouped convolutions, as in ResNeXt (:numref:`subsec_resnext`), are multi-branch"
Two forward references in one sentence, to ideas (grouped convolution, re-parameterization) the reader has not met, used to argue that Inception's ideas "survive". The reader cannot evaluate the claim here. Move this sentence to the ResNeXt subsection, where "Unlike Inception, it makes all branches identical" already says it.

## Exercises

Seven problems, a good spread. Exercise 1 (global average pooling against a dense head, then a 32-pixel shift test) is the best problem in the section and half of it belongs in the main text (third structural finding); the shift test should stay as the exercise. Exercise 3 (count VGG-16 by hand, "why do several of them tie?") is a good ledger problem. Exercise 4 (depthwise Inception) depends on a layer from chapter 6 and is fine. Exercise 5 part 2 ("What do the $3 \times 3$ and $5 \times 5$ branches compute at the smallest of these resolutions?") is a nice trap. Exercise 6 partly repeats what a counting function would print; keep the second part. Exercise 7 (single-branch ablation) is the honest test of whether mixing kernel sizes matters, and given the section's own verdict on hand-tuned branches the text could quote its outcome in one sentence.

## Still open from October

"A catalogue ... with no organizing question" is fixed for VGG and NiN. It is still partly true of the GoogLeNet third (fourth structural finding).
