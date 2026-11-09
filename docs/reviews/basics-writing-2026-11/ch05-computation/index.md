# Computation (chapter introduction)

Source: `chapter_builders-guide/index.md`. About 380 words of prose before the table of contents, plus a 15-item resource list.

## Verdict

A good introduction, close to the house template and better than most chapter openings in the part. It starts from what the reader has done ("a class with some layers, a loss, and a call to `fit`"), asks three questions the reader can feel, gives the origin of the module idea in three sentences, and names the one idea that holds the chapter together: every operation on a model is a walk over a tree of modules. The most important thing to fix is small: the introduction never mentions the 8-billion-parameter model that five of the eight sections use as their running example, so the reader meets it in each section as if for the first time.

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|---|---|---|---|---|
| 4 | 4 | 4 | 4 | 4 |

## What works

- The first paragraph. "We never had to ask how. Once a model grows, we have to." followed by three plain questions (fit in memory, resume after a crash, a layer the library lacks). This is the Strang move of placing the chapter in the course.
- The history is short and does work: Bottou and LeCun's modules explain why "a new architecture is a new arrangement of existing modules", which is the thesis of the first section.
- The sentence "The optimizer, the initializer, the checkpoint, and the move to a GPU all walk this tree." gives the reader the one idea to hold on to. Sections 5.1 to 5.4 do keep returning to it.
- The scope paragraph is honest: "The chapter introduces no new models or datasets".

## Structural findings

- **The running example is missing.** Sections 5.1, 5.2, 5.4, and by their headings 5.5 to 5.7 all compute on "the 8B model". The introduction should introduce it once, in the first paragraph, as the model the chapter keeps asking about: "Take a language model with 8 billion parameters. Will it fit on one GPU? What must be saved to resume it?" The slide deck below the marker already calls it the running example; the prose does not.
- **The roadmap is eight sentences of identical form** (":numref: asks how ..."). It complies with the template and reads like a table. Grouping the eight into three movements would show the arc the chapter in fact has: building a model (5.1 to 5.4: modules, state, initialization, custom layers), the numbers it computes in (5.5), and running it (5.6 to 5.8: saving, devices, reproducibility).
- **Section 5.3 does not fit the stated scope.** "What happens to a model between its definition and a trained result is a matter of software" is true of seven sections. Initialization is the continuation of the variance analysis of the MLP chapter and contains the only derivation in the first half. One clause should say so, or the reader is surprised by mathematics in an engineering chapter.

## Passage-level findings

- **IDX-1** (minor, pedagogy). Under the chapter title. Quote: "model grows, we have to. Does the model fit in the memory of a GPU? Can" The questions are abstract where the chapter is concrete. Fix: attach the 8B model and one number here ("its weights alone take 32 GB").
- **IDX-2** (minor, explanation). Quote: "can lose precision without an error, and one on the wrong device runs slowly". The sentence lists four failures in one breath, and "a coarser number format" means nothing yet to this reader. Fix: keep the first two failures (initialization, incomplete checkpoint), which the reader can already picture, and let the roadmap carry the other two.
- **IDX-3** (minor, language). The roadmap paragraph beginning ":numref:`sec_model_construction` asks how a model is assembled from modules,". Eight parallel "asks" sentences. Fix: three groups as above, one sentence each for the minor sections.
- **IDX-4** (minor, language). Resource list. Quote: "free PDF from the PyTorch team (archived copy; the original pytorch.org link has gone away); Part 1 covers". Every entry is punctuated with two em dashes (fifteen in the list), in a book whose guide asks for almost none. Fix: "Title, by Authors: annotation."

## Exercises

None, correctly.

## Still open from October

The October note that the roadmap disagreed with the order in `_quarto.yml` is fixed: the roadmap and the `toc` block agree.
