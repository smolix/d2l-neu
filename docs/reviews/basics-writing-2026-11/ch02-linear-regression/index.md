# Linear Regression (chapter introduction)

Source: `chapter_linear-regression/index.md`, about 400 words of introduction plus a 400-word resource list.

## Verdict

A good chapter introduction: it starts from something the reader does (pricing a house by comparison), states the idea in one sentence, gives the history in two, and names the two dangers the chapter studies. Its weakness is that it is not the only such opening. The part introduction before it and the section opening after it tell the same three things (Legendre and Gauss, the house, the four parts of learning), so a reader going front to back meets them three times within two pages. The most important fix is to divide that material between the three openings.

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|---|---|---|---|---|
| 4 | 4 | 4 | 4 | 3 |

## What works

- The first paragraph. "Looking at a house for sale, most of us estimate its price by comparison" starts from the reader and ends in two real questions ("How should we find the rule, and how far can we trust it for a house that has not been sold?"). Those two questions are the two halves of the chapter. Keep it.
- "Each weight is one adjustment, such as the price per square meter." One sentence that makes a weight concrete before any symbol.
- The Ceres anecdote is brief and true, and it is the only wit-bearing historical detail in the chapter.
- "an implementation can be wrong without raising an error" announces the chapter's most distinctive theme (testing code against a known answer) in plain words.

## Structural findings

- **Triple opening.** `chapter_basics/index.md` tells the Legendre and Gauss story and the recipe "a model with parameters, a loss that measures its errors, and an algorithm". This file tells it again with dates. `linear-regression.md` then opens with the house again, cites Legendre and Gauss again, and lists the four parts again. See IDX-1.
- **The route hides the chapter's shape.** Seven sentences of the form "X asks how ..." give each section equal weight. The chapter really has three movements: the model and its mathematics (section 1), building and testing an implementation (sections 2 to 5), and trusting a fitted model (sections 6 and 7). Saying so would prepare the reader for the four consecutive software sections.

## Passage-level findings

**IDX-1** (moderate, thought). Under the chapter title.
Quote: "Linear regression contains every part of learning in its simplest form: a"
The sentence and its list (model, loss, algorithm, predictions) reappear almost word for word nine lines into `linear-regression.md` ("It is also the simplest model in which every part of learning appears: a model with parameters, a loss ...") and in the part introduction. Fix: keep the list here, where it frames the chapter, and delete it and the Legendre and Gauss sentence from the opening of `linear-regression.md`, which can go straight from the house to the definition of regression.

**IDX-2** (moderate, language). Route paragraph.
Quote: ":numref:`sec_generalization_basics` asks"
Seven consecutive sentences with the subject "section N" and the verb "asks". The form comes from the writing guide (section 12.1, item 3), and it reads as a filled-in template here and in the neighbouring chapter introductions (the Basics part introduction uses "asks" eight times, chapter 3 seven times). Fix: three sentences, one per movement. Draft: "The first section derives the model, its loss, and the algorithm that trains it. The next four build the software that every later chapter reuses, generate data with a known answer, and train the model twice, once with every part written by hand and once with a framework. The last two ask when a model that fits its training data can be trusted on new data, and how to limit its flexibility."

**IDX-3** (minor, language). Resources.
Quote: "The next chapter builds on the shared statistical references rather than listing them again."
A note about how the book's resource lists are organized, written for the authors. Delete; the first sentence of that paragraph is enough.

## Exercises

None, correctly.

## Still open from October

Nothing specific to this file.
