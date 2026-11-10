# Preliminaries (chapter introduction)

Source: `chapter_preliminaries/index.md`. About 400 words of prose before the resource list.

## Verdict

A good chapter opening, close to the bar. It starts from the four components the introduction named, poses one concrete task (sorting photographs of clothing), and derives the chapter's contents from what a program for that task would have to do. A reader finishes it knowing why tensors, linear algebra, calculus, autodiff and probability are all here. The one thing to fix is the roadmap paragraph, which uses the same sentence frame seven times and reads as a filled-in template.

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|:-:|:-:|:-:|:-:|:-:|
| 4 | 4 | 4 | 4 | 3 |

## What works

- The second paragraph turns a list of topics into a sequence of needs: "It would store the photographs as numbers and compute with all of them at once. It would combine those numbers into a score for each category, find out how the loss changes when each parameter changes, and judge whether its accuracy reflects a real pattern or the luck of the examples it saw." Each clause is one section. Keep this.
- The historical sentence is short, concrete and true: "probability began in 1654 with letters between Pascal and Fermat about a game of chance". It also tells the rusty reader that nothing here is exotic.
- The scope sentence is honest about the appendix: readers "new to these subjects can turn to it at any point". The chapter then keeps that promise (see the chapter file): no section needs the appendix to be followed.

## Structural findings

- The same garment example is introduced three times in the reader's first pages: in the part introduction (`chapter_basics/index.md`, "a small photograph of a garment: it should say whether the photograph shows a shirt, a sneaker, or a bag"), here, and again in the opening of the linear algebra section ("small grayscale images of clothing ... such as shirts, sneakers, and bags"). The repetition is harmless once, but a front-to-back reader meets the same setup twice within two pages. The chapter introduction could refer to "the garment classifier of the part introduction" and spend the saved words on the minibatch numbers it actually uses.
- The introduction does not say that the chapter is long (seven sections, roughly 27,000 words of prose in one framework's view) or how a reader who already knows NumPy and calculus should use it. One sentence of reading advice would help: which sections can be skimmed by whom, and which (automatic differentiation, the standard error) should be read by everyone.

## Passage-level findings

**IDX-1** (moderate, language). Under the chapter title, roadmap paragraph.

> asks how to store such data and compute with all of it

The roadmap uses "X asks how ..." seven times in seven sentences. The writing guide asks for one question per section, and the result follows the letter of that rule so closely that the paragraph has no rhythm. The same frame is used in the part introduction and in the chapter 2 introduction, so by the third occurrence the reader sees the template and skims. Vary the construction and group sections: "The first two sections are about data: how to store and compute with it, and how to turn a table with gaps and words into tensors. The next three follow one prediction from weighted sum to gradient ..." Three sentences can carry seven sections.

**IDX-2** (minor, thought). Same paragraph.

> asks how evidence updates a probability, and when a measured difference, such as a change in accuracy from one minibatch to the next, is larger than sampling noise

This is the longest roadmap entry and the only one that needs terms ("sampling noise") the reader may not have. It is also the entry where the section really has two ideas (Bayes' theorem and the standard error; see PRB structural findings). If the probability section is tightened, shorten this to its main question.

**IDX-3** (minor, language). Second paragraph.

> with two kinds of software: array libraries, which apply one operation to millions of numbers in compiled code, and automatic differentiation, which computes the derivatives of a program

Good content, but it is the fifth clause of a 50-word sentence. Split after "out of the question."

## Still open from October

The October note called the chapter "a tour ... each pointing to the appendix". The introduction now gives the chapter a single task and the appendix pointers have dropped from 34 to about 17, all for proofs. Closed.
