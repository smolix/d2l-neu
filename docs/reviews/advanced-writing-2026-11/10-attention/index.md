# Attention (chapter introduction)

Source: `chapter_attention/index.md`, read to the slides marker. About 330 words of introduction and a resources list of 16 annotated entries (about 500 words). No code, no figure.

## Verdict

A good short introduction with one flaw of form and one of substance. The first three paragraphs are what an introduction should be: a problem the reader feels (the name from page one of a mystery novel), the idea in four verbs (keeps, scores, turns, returns), and the three new problems the idea creates. The fourth paragraph then drops into a list of six sentences of the form "section X asks how ...", which is the pattern the writing guide names as a defect, and it gives no judgment: nothing says which sections a reader needs before chapter 11 and which can wait. The flaw of substance is that the word "self-attention" does not occur in the introduction's prose. The route advertises "heads that share keys and values", a serving optimization, and does not mention the layer that the rest of the book is built from. The most important fix is to rewrite the route as a paragraph with a spine (the three problems of paragraph three already supply it) and to name self-attention as what the first half of the chapter builds.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought and focus | 4 |
| Pedagogy | 3 |
| Language | 4 |
| Ease and enjoyment | 3 |

## What works

- The opening image: "A person turns back and looks. A recurrent network must have kept the name in its state all along, written when nothing marked it as important." Two sentences state the whole case against a fixed state. It does not repeat the example that opens `queries-keys-values.md` (the 768 numbers of a GPT-2-wide state); the two complement each other. Keep both.
- The definition in one sentence: "It keeps a representation of every position, scores the current query against a key at every position, turns the scores into weights, and returns the weighted average of the values stored there." A reader can carry this through the chapter.
- The third paragraph: "A weighted average ignores the order of its inputs. Scoring all pairs of positions costs time and memory that grow with the square of the length. And a map of attention weights shows where a head looks, not what it computes." Three sentences, three sections, and the reader knows why each exists. This is the route with a voice; the paragraph after it should be written the same way.
- The closing paragraph delimits the chapter against chapters 11 and 12 in two sentences.
- The resources list is annotated, grouped, and ties each entry to the section it goes with. That is more useful than most such lists.

## Structural findings

1. **The route is a table of contents in prose.** Six consecutive sentences have the verb "asks" (IDX-1). The slides of the same file carry this content as a table, which is the right form for it.
2. **No guidance on what matters.** The reader is not told that the first three sections are needed for everything that follows, that the positional section supplies the model and the RoPE functions that chapter 11 uses, that the cost section is needed for chapter 12 and for the cache section of chapter 11 but can be read later, or that the circuits section introduces the residual stream on which `sec_transformer-block` opens (IDX-2).
3. **Self-attention is not named** (IDX-3). See the chapter review, "Arc and order".
4. **No picture.** For a chapter measured against The Illustrated Transformer, an introduction without one figure of the lookup is a missed chance (IDX-6).

## Passage-level findings

**IDX-1** (moderate, pedagogy). Fourth paragraph.
Anchor: `:numref:`sec_queries-keys-values` asks how a lookup can be made`
Six sentences in a row begin with a section reference and "asks". The reader skips them. Rewrite as a paragraph that follows the argument, not the file order. Draft: "We start from a lookup in a table of two records and make it differentiable (:numref:`sec_queries-keys-values`), then choose the score a network can learn and the masks that hide padding and the future (:numref:`sec_attention-scoring-functions`). Letting a sequence query itself, with several heads at once, gives the self-attention layer that every later chapter uses (:numref:`sec_multihead-attention`). The remaining three sections take the three problems above in turn: order (:numref:`sec_positional-information`), cost (:numref:`sec_attention-at-scale`), and what a trained head computes (:numref:`sec_what-attention-computes`)."

**IDX-2** (moderate, pedagogy). Fifth paragraph.
Anchor: `The chapter studies the attention layer by itself.`
Add the reading guidance here, in two sentences. Draft: "A reader in a hurry to reach the transformer needs the first four sections and the residual stream of the last one. The cost section can wait until :numref:`sec_kv-cache` or :numref:`chap_modern_rnn` asks for it."

**IDX-3** (moderate, thought). Fourth paragraph.
Anchor: `express, and what several heads, and heads that share keys and values,`
The route sentence for the multi-head section names key-value sharing and omits self-attention and cross-attention, which are in the section's title and are what later chapters cite it for. Whatever is decided about where self-attention is taught, the introduction must say where it is.

**IDX-4** (minor, correctness; sure that the chapter contradicts itself). Second paragraph and resources.
Anchors: `weighted average makes the lookup differentiable. The idea came from` and `the first learned alignment, whose additive score`
The introduction says the idea came from machine translation and calls Bahdanau et al. "the first learned alignment". `further-reading.md` says "Bahdanau's attention of :numref:`sec_attention-scoring-functions` was not the first differentiable alignment" and credits Graves (2013). Both cannot stand. Write "became known through machine translation" and "the learned alignment that started the field".

**IDX-5** (minor, explanation). Last paragraph.
Anchor: `:numref:`chap_modern_rnn` develops the recurrent side of linear attention.`
"Linear attention" has not been introduced, and the route above describes it as "a recurrence of linear cost". Use one name, or write "develops the recurrent form of attention that :numref:`sec_attention-at-scale` derives".

**IDX-6** (minor, pedagogy). Second paragraph.
Anchor: `Attention does this. It keeps a representation of every position,`
The four-verb definition is exactly what `fig_qkv` of the first section draws. A small figure here (query, three keys, scores, weights, a point inside the triangle of values) would give the introduction its picture and the chapter its emblem.

**IDX-7** (minor, correctness; partly unverified). Resources list.
Anchors: `Thinking Like Transformers (Transformer Puzzles)` and `pictures of scaled dot-product and multi-head attention with the shapes of every matrix`
(a) RASP and the paper "Thinking Like Transformers" are by Weiss, Goldberg and Yahav (2021). Rush wrote a blog post of the same name about it and the puzzle repository that the link points to. The entry merges the three under one name. Give the puzzles their own title and credit RASP to its authors. I am sure of the authorship of RASP; I did not open the repository to confirm that the puzzles are written in RASP itself. (b) The Illustrated Transformer is described by its matrix shapes. What it is known for is a walk through self-attention on one sentence, step by step. Since the book names it as a bar, describe it as that. (c) The list has 16 entries at equal weight. One sentence at its head should say which two to read first (my choice: 3Blue1Brown for the picture, Elhage et al. for the last section).

## Still open from October

Not applicable to this file beyond the chapter-level findings.

## For the chapter review

- The introduction's third paragraph gives the chapter a real plan (lookup, then three problems). The chapter mostly follows it; the route paragraph hides it.
- The introduction is the place to fix the reader's expectation about self-attention, whichever section ends up teaching it.
- The resources list uses a long dash between title and annotation in every entry. If the house rule against em dashes applies to lists as well as prose, replace with a colon or a period.
