# Linear Classification (chapter introduction)

Source: `chapter_linear-classification/index.md`. About 370 words of prose before the table of contents, plus an annotated resource list of ten entries.

## Verdict

A good chapter introduction, close to the bar. It starts from what the reader did in the previous chapter, makes the problem felt in three sentences, names the model, says what carries over and what is new, and gives one question per section. It also announces the double theme (predict a category, and know when to trust the prediction) that is the only thing holding the last two sections to the first five. The most important fix is small: the introduction should say outright that the chapter has two halves, a model and its implementation, then three sections on how far a measured accuracy can be trusted, because the table of contents alone does not show it.

## Scores

| Explanation | Thought | Pedagogy | Language | Enjoyment |
|:-:|:-:|:-:|:-:|:-:|
| 4 | 4 | 4 | 4 | 4 |

## What works

- The opening image: "shows a shirt, a coat or a sandal, not 5.5 of anything". One sentence makes the reader see why regression on class codes is wrong. Keep it.
- "Nor could one number express doubt." Seven words that motivate probabilities as outputs.
- The Fisher paragraph gives the idea an origin (a weighted sum of four measurements, 1936) and then states the whole method in two sentences.
- "A probability is not yet a decision, accuracy can mislead when one class is rare, and a test set vouches only for data that resemble it." This sentence is the chapter's argument for containing evaluation, theory and shift. It is the best sentence in the introduction.
- The scope line (one class per example, unordered classes) is stated once and briefly.

## Structural findings

- The roadmap gives seven sentences of the form "X asks ...". The previous chapter's introduction uses the same frame seven times. As a house pattern it is clear, but seven identical sentence openings in a row read as a form being filled in. Group the sections: two sentences for the model and data, two for the implementation, three for trust.
- The introduction repeats the 5.5 example that opens `softmax-regression.md` two pages later (there with "halfway between a coat and a sandal", here with "as if a pullover lay halfway between a trouser and a dress"). One of the two should use a different image, or the section opening should refer back in half a sentence and move on to the 81-times argument, which is new.

## Passage-level findings

**IDX-1** (minor, pedagogy). Anchor: "classifier in this book, up to the largest language models, ends in the same", second paragraph.
The paragraph runs from Fisher to language models to evaluation in ten sentences, and the turn to the second half of the chapter ("Evaluation changes more.") is three words long. A reader skims past the hinge of the chapter. Give the turn its own short paragraph: say that the second half of the chapter asks how far the measured accuracy can be trusted, on new data from the same source and on data from a different one.

**IDX-2** (minor, language). Anchor: "asks which dataset is small enough for quick", roadmap paragraph.
Seven consecutive "asks" sentences. See the structural finding. Suggested shape: "The first two sections define the model and the data. ... The next three build and measure it. ... The last two ask what the measured accuracy means."

**IDX-3** (minor, language). Anchor: "free PDF; derives logistic and softmax regression and the cross-entropy loss", resource list.
The resource list uses spaced em dashes as separators (19 of them), the only em dashes in the chapter. If the house rule against dashes applies to lists, use a colon or a period. This is trivia unless the rule is enforced elsewhere.

## Exercises

None, as expected.

## Still open from October

Nothing specific to the introduction.
