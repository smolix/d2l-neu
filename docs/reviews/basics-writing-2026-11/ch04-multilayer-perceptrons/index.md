# Multilayer Perceptron (chapter introduction)

Source: `chapter_multilayer-perceptrons/index.md`. About 330 words of introduction, then a resources list of about 500 words.

## Verdict

A good chapter introduction, close to the house template and better than the template deserves: it starts from the Fashion-MNIST classifier the reader has just trained, names the one thing that classifier cannot do, gives the history in five sentences, and states the three problems that organize the chapter. The main thing to fix is not in this file alone: its first paragraph and the first paragraph of `mlp.md` are the same paragraph, so a reader going front to back reads the sleeve, the strap and the background twice within a page (see MLP-1). The route paragraph is correct but mechanical.

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|---|---|---|---|---|
| 4 | 4 | 4 | 4 | 3 |

## What works

- The limit of softmax regression is put in one plain sentence that the whole chapter then leans on: "Its limit is that each pixel adds the same evidence for a class whatever the pixels around it show."
- The history paragraph (Rosenblatt 1958, Minsky and Papert 1969, Rumelhart, Hinton and Williams 1986) is the only history in the chapter and it earns its place: it tells the reader that the problem is old and that the missing piece was a way to learn the features.
- "hidden layers raise three new problems" followed by three sentences, one per problem, is the best orientation device in the chapter. Each problem maps onto one later section (backprop, initialization, generalization).
- The scope paragraph is two sentences and says what is deferred.

## Structural findings

- The three problems cover `backprop`, `numerical-stability-and-init` and `generalization-deep`. Dropout and the Kaggle section are not placed by them; they appear only in the route list. One clause each would do ("dropout is a regularizer built for hidden units; the last section applies all of this to a table of house prices").
- The introduction does not say what the reader should already know. The chapter leans on `sec_softmax_scratch` (the 83% and the confusion pattern), `sec_generalization_basics` (polynomial fits, the U curve, cross-validation), `sec_weight_decay` and `sec_autograd`. One sentence of prerequisites as knowledge (writing guide §7.4) would help a reader who skipped part of chapter 2.

## Passage-level findings

- **IDX-1** (moderate, pedagogy)
  - Anchor (opening paragraph): "sleeve, a strap, or the background, and only its neighbors can tell which."
  - The same example, in nearly the same words, opens `mlp.md` fifteen lines of reading later, and a third version ("evidence for a shirt or for a bag") follows in its first subsection. The repetition makes the first page of the chapter feel like it starts twice.
  - Fix: keep the example here. Let `mlp.md` open by referring to it in a clause and go straight to XOR. See MLP-1.

- **IDX-2** (minor, explanation)
  - Anchor (opening paragraph): "How do we build a model that combines its inputs in this way, and how do we"
  - "In this way" has no clear referent: the preceding sentence says what a pixel may belong to, not a way of combining. The reader has to reconstruct "lets the evidence of one pixel depend on the others".
  - Fix: "How do we build a model in which the evidence from one input depends on the others, and how do we train it?"

- **IDX-3** (minor, language)
  - Anchor (route paragraph): ":numref:`sec_generalization_deep` asks why a network that can memorize its"
  - Seven consecutive sentences have the form "Section X asks ...". The questions are good, the rhythm is a list read aloud. This is the house template (§12.1 item 3) applied literally, and I expect it in every chapter.
  - Fix: group the route by the three problems just stated ("The first two sections build and train the model. The next three take the three problems in turn: ..."), so the route reads as an argument and the labels become parenthetical.

- **IDX-4** (minor, thought)
  - Anchor (second paragraph): "Computing new features first escapes this limit, but for years no"
  - "Escapes this limit" before the reader knows what a feature computed first would look like. It is true, but the sentence only becomes meaningful after the XOR construction in the next section.
  - Fix: add the smallest hint: "Feeding the classifier new features computed from the inputs, such as the product of the two bits, escapes this limit".

## Exercises

None, correctly.

## Still open from October

Nothing specific to this file. The resources list uses a dash-separated format with 31 em dashes; that is list formatting, not prose, and I would leave it.
