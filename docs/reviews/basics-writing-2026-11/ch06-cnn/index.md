# Convolutional Neural Networks (chapter introduction)

Source: `chapter_convolutional-neural-networks/index.md`, about 400 words of prose before the resource list.

## Verdict

A good chapter introduction, close to the bar. It starts from what the reader has done (an MLP on 784 pixel values), states a fact the reader can test in their head (shuffle the pixels and nothing changes), and ends the first paragraph on a real question. The history paragraph is three sentences and each one earns its place. The roadmap is phrased as questions, in the reader's words. The one thing to fix is that the third paragraph answers the opening question before the reader has had a chance to hold it, and in doing so uses two terms (*locality*, *translation equivariance*) and one result (far fewer parameters, far less data) that the chapter never demonstrates at the level claimed.

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|---|---|---|---|---|
| 4 | 4 | 4 | 4 | 4 |

## What works

- The opening paragraph. "We recognize a sneaker by its outline, however, and an outline is a pattern among neighboring pixels. It also means the same thing wherever in the image it appears. What would a layer look like that knew these two facts about images?" This is the Strang move (place the chapter, pose the question) in four sentences. Keep it verbatim.
- The roadmap: one sentence per section, each a question ("asks how to summarize a neighborhood so that small shifts do not matter"). No previewed terminology apart from "dilation" and "channels".
- The scope sentence at the end is one sentence and says where the rest lives.
- The resource list is annotated and each entry names the section it accompanies.

## Structural findings

- The pixel-shuffling observation is the chapter's best hook, and it is stated three times in nearly the same words: here, in the first paragraph of `why-conv.md`, and in the second paragraph of `lenet.md`. It is never tested. See IDX-1 and the chapter file (finding 2): either the repetition goes, or the claim is turned into the experiment that closes the chapter.
- The introduction promises "far fewer parameters and far less data". The chapter delivers the first (LeNet has under a third of the MLP's parameters) and never touches the second. See IDX-3.

## Passage-level findings

**IDX-1** (moderate, pedagogy). Under the chapter title:

> if we shuffled the pixels of every image in the
> same way and retrained, it would do as well as before.

Stated as fact here, again in `why-conv.md` ("If we permuted the pixels of every image in the same way, it would learn just as well") and again in `lenet.md` ("Shuffling the pixels in the same way in every image would therefore not change what it can learn"). Three statements, no demonstration. Fix: keep the statement here, shorten the one in `why-conv.md` to a back-reference of one clause, and replace the one in `lenet.md` by a ten-line experiment (train the MLP and LeNet on a fixed pixel permutation of Fashion-MNIST; see LEN-2).

**IDX-2** (moderate, explanation). Third paragraph:

> It builds in
> *locality*, since each output depends only on nearby pixels, and
> *translation equivariance*, since a shifted input gives a shifted output.

The first paragraph asked the question in plain words ("knew these two facts"). This paragraph answers it with the two technical names, two sentences before the roadmap says that `sec_why-conv` "asks which layer follows from locality and translation equivariance alone". The definitions given are short and correct, so this is not a serious violation, but the answer arrives before the reader has felt the question. Fix: say what the layer does ("applies one small set of weights to every neighborhood of the image") and let the roadmap sentence say "asks which layer follows from these two facts alone". Leave the two names to `why-conv.md`, which defines them properly.

**IDX-3** (moderate, thought). Third paragraph:

> A
> network that need not relearn a pattern at every position needs far fewer
> parameters and far less data.

"Far less data" is the stronger and more interesting half of the claim, and nothing in the chapter supports it: every experiment uses all of Fashion-MNIST, and LeNet beats the MLP by one to three points. Either drop "and far less data" or add the experiment that shows it (LeNet and the MLP on 1,000 or 5,000 training images, where the gap is much larger than on the full set). The second option would also give the chapter a stronger ending than it has.

**IDX-4** (minor, language). Third paragraph:

> They remain the default where data
> are scarce, latency matters, or every pixel needs a prediction, and vision
> transformers (:numref:`chap_transformers`) now share the field.

The same three conditions close the introduction of the next chapter ("they remain the default where data are limited, latency matters, or every pixel needs a prediction"). Here the sentence joins two unrelated claims with "and". Fix: end at "prediction." and give the transformers their own short sentence, or cut the clause here since ch. 7 says it.

**IDX-5** (minor, language). Second paragraph:

> network of local detectors repeated across the image, and LeCun and his colleagues trained networks of this kind by

An unwrapped line with two "and"s in a row; a sentence break after "image" reads better: "... repeated across the image. LeCun and his colleagues trained networks of this kind by backpropagation to read handwritten zip codes."

## Exercises

None, as expected for an index.

## Still open from October

Nothing specific to the index. The October review found unrevised legacy prose in the chapter; none remains here.
