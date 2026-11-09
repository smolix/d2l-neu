# Convolutional Neural Networks (LeNet)

Source: `chapter_convolutional-neural-networks/lenet.md`, about 1,830 words of prose (180 in framework tabs), 6 exercises (about 750 words).

## Verdict

The section does what the October review asked: the network is assembled from the four preceding sections, its shapes are checked, its cost is counted layer by layer, and the training run is read out and compared with the MLP under the same loss, optimizer, and number of epochs. The cost table and the paragraph that reads it ("The convolutional layers hold 4% of the parameters and perform 86% of the multiply-adds") are first-rate. As the end of the chapter, though, the section is an anticlimax. The chapter opened by claiming that convolution builds in what an MLP has to learn; the evidence offered is one to three points of accuracy, reached with a learning rate of 2.0 that the text has to justify, after a start at chance of unpredictable length. The properties that the chapter derived (the same kernel everywhere, indifference of the MLP to pixel order, tolerance to shifts) are never tested on the trained network, and the reader never sees a learned kernel or a feature map. The single most important fix is to add one short experiment that shows what the convolutional prior does, not just what accuracy it gets.

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|---|---|---|---|---|
| 4 | 4 | 3 | 4 | 3 |

## What works

- The opening restates what is being compared and under what terms: "We train the convolutional classifier with the same loss, optimizer, and number of epochs as the MLP and compare their validation accuracy." A fair test, announced before it is run.
- The historical paragraph is short and ends on its significance: "It showed that a network trained from examples could replace a hand-engineered recognition pipeline."
- `img_lenet` (rendered and inspected): feature maps drawn to scale, a shirt as input, each window tied to the unit it produces. It carries the architecture by itself. `img_lenet_vert` adds the shapes.
- "A wrong padding or a missing flattening step changes the shape of every layer that follows it, so we check the shapes before training." A reason for the cell, in the reader's interest.
- `tab_lenet_cost` and its reading. Every entry checks (61,706 parameters; 416,520 multiply-adds; 4%, 86%, 78%, 12%). "The two kinds of layers divide the work in opposite ways" is the one-sentence takeaway, and the comparison with the MLP ("fewer than a third of the parameters and performs about twice the arithmetic") is exact (61,706 against 203,530; 416,520 against about 203,000).
- The candour about the teaching version: "This is the common teaching version of LeNet-5, not an exact reproduction."
- The exercises are the best set in the chapter (see below).

## Structural findings

- **The climax is small and hedged.** See LEN-2 and LEN-5.
- **The network is trained in a configuration chosen for history, and the text then spends a paragraph managing its side effects.** Sigmoids and average pooling make LeNet slow to start; the fix (learning rate 2.0) is unusual enough to need a derivation; the result still "can stay at chance" for a varying number of epochs. The modern variant (ReLU, max-pooling) is exercise 1. A reader gets the fragile version in the text and the robust one as homework. See LEN-4.
- **The Summary carries a five-row table of components the reader has not met.** See LEN-6.
- **No look inside the trained network.** The chapter is about feature detectors and never shows one that was learned. Exercise 3 asks the reader to do it. Two rows of images (six first-layer feature maps for a sneaker and for a shirt) would be the "Illustrated" payoff of the whole chapter.

## Passage-level findings

**LEN-1** (minor, language). Section title:

> # Convolutional Neural Networks (LeNet)

The section has the chapter's title with a parenthesis. In the table of contents the reader sees "6 Convolutional Neural Networks ... 6.6 Convolutional Neural Networks (LeNet)". "LeNet" is enough.

**LEN-2** (major, pedagogy). Opening, second paragraph:

> layer. Shuffling the pixels in the same way in every image would therefore
> not change what it can learn: the MLP does not know which pixels are
> neighbors, and it must learn a pattern separately at every position where the
> pattern can occur.

Third statement of the shuffling argument in the chapter (after the chapter introduction and `why-conv.md`), and still an assertion. This section has both models and the data in hand; it is the place to test it. Fix: after the main training run, apply one fixed random permutation to the 784 pixels of every image and train both models again (the `Trainer` and data class are already there; the permutation is a one-line transform). Expected reading: the MLP's accuracy is unchanged up to run-to-run noise; LeNet's falls to the MLP's level or below. One plot or one two-row table, and three sentences: "The MLP does not notice. LeNet loses its advantage: its kernels now see pixels that were never neighbors. The gain of the convolutional network was the gain from knowing which pixels are adjacent." This replaces three assertions with one demonstration, closes the question the chapter introduction opened, and gives the chapter a memorable ending. A second candidate, cheaper still: evaluate both trained models on test images shifted by one and two pixels, which tests the tolerance claimed in `sec_pooling`. Either would do; the permutation is the sharper one. (Check that the result holds in all four frameworks before writing the prose.)

**LEN-3** (minor, thought). Under "The LeNet Architecture":

> $32\times32$ images. The MNIST digits fit in a $20\times20$ box within the
> $28\times28$ images, so in the larger input even features near the border
> of a digit, such as the end of a stroke, could fall at the center of the
> receptive field of a unit in the last convolutional layer.

The paragraph lists five differences from the 1998 network (scaled tanh, trainable pooling, partial connections, radial-basis outputs, the 120-unit layer as a convolution) and then explains the $32\times32$ input at length. October flagged an earlier version of this aside as likely wrong; the present wording follows the 1998 paper and I see no error. But for a reader building LeNet on Fashion-MNIST the content they need is one sentence: the first layer pads by 2 so that its output is $28\times28$, as in the original, which took $32\times32$ inputs. Fix: keep the first sentence, one clause listing the differences, and the padding sentence; move the receptive-field reasoning to an exercise or cut it. About 80 words instead of 160.

**LEN-4** (moderate, pedagogy). Under "Training":

> therefore learns slowly: its validation accuracy stays at chance for the
> first two to four epochs and reaches only about 73% after 10. We use a
> learning rate of 2.0 instead, twenty times larger.

Three issues. (a) The run at learning rate 0.1 is reported but not shown; the reader takes "two to four epochs" and "73%" on trust, and these are per-run numbers of the kind `CLAUDE.md` asks not to quote. (b) The argument (four sigmoids, each a factor of at most $1/4$, so $4^4 = 256$) does not lead to "twenty times"; the number 2.0 is empirical and the text should say so in a clause. (c) The paragraph is the text managing a weakness of the chosen configuration. Fix: either show both curves in one plot (0.1 and 2.0), which makes the vanishing-gradient point visible and is worth the extra minute of training, or state the claim without numbers: "At the MLP's learning rate of 0.1 LeNet barely moves in the first epochs. A learning rate of 2.0, found by trial, compensates." Consider also training the ReLU and max-pooling variant in the text as a second model (it is exercise 1.3): it trains at an ordinary learning rate without the plateau, and "sigmoid LeNet against the same network with ReLU" is the lesson that the next chapter's introduction starts from ("Deeper stacks of sigmoid layers trained worse than shallow ones").

**LEN-5** (moderate, thought). After the training cell:

> Even at this learning rate the validation accuracy can stay at chance, 10%,
> before it climbs; how long it stays there varies from run to run, from no
> epoch at all to several.

The stored PyTorch curve shows two epochs at chance, then a rise to about 0.89, so the prose is consistent with the output. But the paragraph opens the reading of the chapter's main result with a caveat about instability, and the result that follows ("about 88--90%, against about 87% for the MLP") is a difference of one to three points, at the edge of what the stated run-to-run stability (about half a point) can support at its lower end. The writing-guide rule applies: an experiment that needs its confounders explained before the result is not yet a teaching experiment. Fix: lead with the result and what it means; put the plateau in one clause; and strengthen the comparison (LEN-2) so that the conclusion "A convolution kernel learns a local pattern once and detects it at every position" rests on more than a small gap in accuracy. That concluding sentence is not shown by the accuracy numbers at all; it is the chapter's premise restated.

**LEN-6** (moderate, thought). Summary, table `tab_lenet_modern`:

> | LeNet (this section) | Modern (2020s) | Effect of the change |

A table in a Summary that introduces batch and layer normalization, global average pooling, and strided-convolution downsampling as replacements, with an "effect" column of compressed claims ("stable activation scales, so much deeper stacks train"). GELU and He initialization are known from chapter 4; the others are not. The intent (a bridge to chapter 7) is good, and the next chapter's introduction already builds that bridge in prose. Fix: replace the table by two sentences in the closing paragraph: "Each of LeNet's components was later replaced: the sigmoid by the ReLU, average pooling by max-pooling or strided convolutions, the large dense head by an average over positions. The next chapter makes these changes one at a time and asks what each contributes." The Summary then restates only what the section taught.

**LEN-7** (minor, language). Summary:

> twenty times that of the MLP, because each sigmoid shrinks the backward
> gradient by a factor of at least four.

"Shrinks by a factor of at least four" is right but the body said "multiplies it by at most $1/4$"; use one phrasing. Also see LEN-4 (b): the "because" overstates how the factor of twenty follows.

**LEN-8** (minor, pedagogy). PyTorch tab under "The LeNet Architecture":

> Lazy layers have no weights until the first minibatch fixes their shapes, so
> the model cannot initialize them in its constructor.

Four tab notes of initialization plumbing (58 words in the PyTorch tab) sit between the architecture and its code. The reader who wants to see LeNet has to read how the `Trainer` calls `weight_init` first. Move these notes below the class definition, or into code comments (the PyTorch class already has one: "The Trainer calls this after the first minibatch fixed the shapes").

## Exercises

The strongest set in the chapter: every problem is an experiment with a question.

- Exercise 1 (modern components) is well staged, and part 4 (explain the plateau from the sigmoid's mean and slope) is a real piece of analysis, correctly marked hard. If the text adopts the ReLU variant (LEN-4), parts 1 to 3 shrink to a comparison the text has started.
- Exercise 2 (original MNIST; difference against ratio of error rates) teaches a point about benchmarks that is seldom made.
- Exercise 3 (activation maps): part 1 is the picture the section should show itself. Part 2 (activation ranges on out-of-distribution inputs) is good.
- Exercise 4 (overfitting sanity check): practical, and part 2 ("Name one bug that this check would catch and one that it would miss") asks for judgment.
- Exercise 5 (equivariance against augmentation): the only place where the chapter's derivation meets its final network. Keep, and consider promoting part 1 into the text as a closing paragraph: which components of LeNet break exact equivariance (the border, the stride-2 pooling) and which prevent invariance (the flatten and dense head).
- Exercise 6 (the dense head) asks the reader to predict before measuring. Good. It needs global average pooling, which the chapter defines only in the Summary of `pooling.md` (POOL-8).

## Still open from October

Closed: "LeNet's training run is never read out or compared with the MLP" (it now is, with a cost table). The MNIST aside that October called likely wrong has been rewritten; I found no error in it, only excess length (LEN-3).
