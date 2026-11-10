# Optimization Algorithms (chapter introduction)

Source: `chapter_optimization/index.md` (lines 1 to 93, up to the slides marker). About 330 words of introduction and 600 words of annotated resources (13 entries). No code, no figures. Read after the ten section reviews, so the judgement below is about whether the introduction describes and leads into the chapter that follows it.

## Verdict

The first two paragraphs are good. The opening turns a habit into a question ("When a loss shoots up to `nan`, we halve the learning rate and try again. Why does a larger step ruin a run, and why does a smaller one cost so much time?"), and the second names the two problems that organize everything after it: curvature that differs between directions, and a gradient that is only an estimate. Then the introduction stops leading. The route is ten consecutive sentences of the form "X asks ...", one per section at equal weight, which is the template the Basics review found and the author's decision 6 rules out. It does not say what the chapter is built on (the valley for four sections, then a small transformer that the reader has not been taught), what matters most, what can be skipped on a first reading, or what the reader will be able to do at the end. The slides after the marker have all of this (a goals slide, a table of testbeds) and the prose has none of it. The single most important fix: replace the route paragraph by a route with a voice, in three movements, that names the running examples, warns about `TinyLM`, and tells the reader which sections are the core.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought and focus | 3 |
| Pedagogy | 3 |
| Language | 3 |
| Ease and enjoyment | 3 |

## What works

- **The opening paragraph.** It starts from what the reader has done in every earlier chapter (a recipe, a `nan`, a halved learning rate) and asks three questions the chapter answers. Keep it word for word.
- **Two problems, stated plainly.** "The curvature of the loss differs between directions, so a step that is safe along the steepest one barely moves along the flattest. And the gradient is only an estimate, so every step is slightly wrong." This is the spine of the chapter, and "Momentum, per-coordinate learning rates, schedules and batch sizes each answer one of these problems" is the right promise.
- **History in two sentences that carry content.** Cauchy in 1847 for orbits, Robbins and Monro in 1951 for a root measured with noise. Both are accurate as far as I know, and the second one prepares decaying steps.
- **The resources are annotated by section.** Each entry says which section it accompanies and what it adds, and the list is short enough to be read. The entry for the Distill article on momentum is the model: it names the critical $\beta$ and the ringing that :numref:`sec_momentum` shows.

## Structural findings

1. **The route is a list, not a route** (IDX-1). Ten sentences, ten times "asks". A reader cannot tell from it that sections 1 to 4 are one argument on one quadratic, that Adam and AdamW are the working optimizer of the rest of the book, that Muon and Scaling Up can be left for later, or that Practice is where the tools are meant to come together.
2. **The chapter has two organizing frames and the introduction owns only one.** The introduction says two problems (curvature, noise). `optimization-intro.md` closes with "Three Decisions" (direction, step size over time, treatment of noise), `lr-scheduler.md` opens by citing them, and `practice.md` closes the chapter with "The Chapter's Three Decisions". The two frames fit together (two problems, three things an optimizer can decide about them), but nobody says so, and the frame the chapter ends on is introduced at the end of section 1 instead of here. State the three decisions once, here, as the answer to the two problems; the sections then cite the introduction (see LAND-16 and PRAC-24 in the section reviews).
3. **The test models are not announced.** From :numref:`sec_adam` on the chapter runs on `TinyLM`, a two-block transformer built from material of chapter 11, and on a small CNN. Chapters 10 to 13 then reuse both. The introduction should say in two sentences that the chapter builds a small language model as an instrument, that the reader need not understand attention to use it, and where it is defined.
4. **No prerequisites, no statement of what is optional.** One sentence each: the chapter needs gradients, the Hessian and eigenvalues at the level of :numref:`sec_calculus`, and minibatch SGD from :numref:`subsec_linear-regression-minibatch-sgd`; it reads without the appendix. Nothing later in the book depends on :numref:`sec_muon` beyond two pointers from chapter 13, and :numref:`sec_scaling` is cited only from two places in chapter 11, so both can be marked as skippable on a first pass.
5. **Two lists with nearly the same name.** "Resources and Further Reading" here and the section "Further Reading" at the end of the chapter. The opening of each says what the other holds, which helps, but the headings still collide. Rename this one "Resources" (the same finding is open for the appendix chapters).

## Passage-level findings

**IDX-1** (major, pedagogy and language). Route paragraph.
Anchor: ":numref:`sec_optimization-intro` asks what makes a loss hard to minimize."
Ten "asks" in ten sentences. Direction: three movements with judgement. Draft: "The first four sections work on one function, a valley that is twenty times steeper across than along. On it every claim can be computed by hand: the largest stable step, the 47 steps gradient descent needs at its best, the cloud in which noisy steps stall, and what momentum does to all three. Sections five to seven are about the optimizers and schedules that train today's networks (Adam, AdamW, Muon, warmup and decay), tested on a small language model and a small convnet that we build for the purpose in :numref:`subsec_tinylm`; you do not need to understand attention to use it. The last three sections are about size: how large a batch is worth its cost, how a learning rate carries over to a wider model, and how to set up a run you can afford only once. Readers short of time should read sections one to six and :numref:`sec_practice`; Muon and Scaling Up can wait." Adjust the section numbers to whatever order the chapter ends up with.

**IDX-2** (moderate, thought). Second paragraph.
Anchor: "measure the size of a step. We take the questions one at a time, on"
"on problems we can solve exactly" is true of the first four sections and of the valley parts of Schedules. It is not true of the edge-of-stability network, of `TinyLM`, of the CNN or of the width family, which carry the second half of the chapter, and the first section itself says that "the edge of stability lies outside" the exact analysis. Write what the chapter does: "first on a problem we can solve exactly, then on small networks where we can only measure."

**IDX-3** (minor, thought). Second paragraph.
Anchor: "sizes each answer one of these problems. Newer methods also ask how to"
"Newer methods also ask how to measure the size of a step" is the only mention of the idea behind Muon and it is too compressed to mean anything yet. One more clause: "(as a vector of numbers, coordinate by coordinate, or as a matrix that acts on activations)".

**IDX-4** (minor, correctness of a description). Third paragraph.
Anchor: "The mathematical appendix (:numref:`chap_mdl-optimization`) proves the rates"
"proves the rates that this chapter measures" is right for the valley and for momentum. The chapter also states several small results with their proofs (the noise ball, the steepest-descent step under three norms, the first Adam step under width) and it should, by the author's rule on duplication. Say: "The chapter proves what is short and measures the rest; the longer proofs are in the appendix, which is not needed to read what follows."

**IDX-5** (moderate, correctness; confident). Resources, books.
Anchor: "state without proof are carried out in full."
Boyd and Vandenberghe analyse gradient descent, steepest descent and Newton's method. The book has no treatment of stochastic gradient methods, so the analyses that :numref:`sec_sgd` states without proof are not in it. Name :numref:`sec_gd` only, and point to Bottou, Curtis and Nocedal (already in the bibliography and cited in the Further Reading section) for the stochastic side. "projections" in the same entry is vocabulary this chapter hardly uses; drop it.

**IDX-6** (moderate, thought). Resources, courses and tutorials.
Anchors: "that :numref:`sec_practice` turns into a method" and "the budget-tiered sweeps that :numref:`sec_practice` teaches"
Both entries describe a section the chapter does not yet have. The review of `practice.md` finds that "Tuning Budgets" is 220 words, gives no grid, no rule for picking from a sweep and nothing on batch size, and that the scientific, nuisance and fixed classification arrives in the last 650 words of the chapter (PRAC-5, PRAC-21). Either the section is rebuilt as that review proposes, or these annotations are reduced to what is there ("the classification of hyperparameters that :numref:`sec_practice` uses").

**IDX-7** (minor, correctness; unverified). Resources, papers.
Anchor: "re-benchmarks ten optimizers under matched tuning"
The count of ten and the shrinking speedups agree with my memory of the abstract and with what `muon.md` and `practice.md` say; I could not check the paper. The same holds for the arXiv numbers of the four papers, which I believe are right.

**IDX-8** (minor, language). Resources list.
Anchor: "## Resources and Further Reading {.unnumbered}"
The entries are unranked. With 13 entries a reader wants to know where to start. Mark two or three ("start here": the Distill article for sections one to four, the tuning playbook for :numref:`sec_practice`, Bernstein's note for :numref:`sec_muon`). The entries use a spaced dash twice each as a separator, which is the list format of every chapter introduction and a book-wide decision, not a fault of this file; note it only because the prose rule against dashes will otherwise be applied to it by an editing agent.

## Code and experiments

None.

## Exercises

None.

## Still open from October

Nothing specific to this file.

## For the chapter review

- The introduction is accurate about the order of sections and about each section's question. If sections are split or moved (Adam and AdamW, the test models, Muon before Schedules), the route paragraph and the `toc` block change with them.
- The opening sentence of this chapter and of the appendix chapter were the same in the appendix review (item 12 of its table). The appendix introduction has since been rewritten ("A network is trained by one loop ...") and now leads from this chapter, so that item is closed.
- The phrase "weight decay must then bypass it" in the route is a good one-clause statement of AdamW and can survive any rewrite.
- The slides hold a goals list and a testbed table that the prose lacks. The testbed table is also the quickest way to see a chapter-level fact: the airfoil regression serves two sections and shows the effect in neither.
