# Exercise Review — Group: Foundations
(chapter_preface, chapter_introduction, chapter_preliminaries, chapter_linear-regression)

Repo: `/Users/smola/Repositories/github/d2l-neu`. Source of truth: `.md` files only.
16 files enumerated via `grep -rln "^## Exercises" <chapter_dir> --include="*.md"`.

---

## chapter_preface

```
file: chapter_preface/index.md
heading_line: 612
n_exercises: 3
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: none
discussions: tabbed(4 tabs: mxnet t/18, pytorch t/20, tensorflow t/186, jax t/17963)
defects:
  - none found (markup renders cleanly; blank lines present before list and before :begin_tab: blocks)
clarity:
  - ex 3: "Follow the links at the bottom of the section to the forum, where you will be able to seek out help and discuss the book..." is a navigation/reading instruction, not a task with a deliverable or success criterion — a "go look at X" prompt rather than an exercise.
notable: Only 3 items, all onboarding/setup actions (register on forum, install Python, find the forum links) rather than intellectual exercises — appropriate for a preface but a stylistic outlier vs. later chapters. Followed by a 3-slide deck running to EOF (L662), unrelated in content to the exercises.
```

## chapter_introduction

```
file: chapter_introduction/index.md
heading_line: 1581
n_exercises: 4
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: none
discussions: single-link (https://d2l.discourse.group/t/22) — plain markdown link, no :begin_tab: framework tabs
defects:
  - L1593: Discussions block is a bare single link, not wrapped in per-framework :begin_tab: tabs — inconsistent with the tabbed pattern used in chapter_preface and most of chapter_preliminaries/chapter_linear-regression.
clarity:
  - ex 1: "Which parts of code that you currently write could be 'learned'...?" — reflective "think about X" prompt with no stated deliverable.
  - ex 2: "Which problems that you encounter have many examples of successful solutions but no explicit procedure...?" — same think-about-it pattern, no concrete output requested.
notable: Shortest exercises section of the group (13 lines heading-to-EOF); file ends immediately after the single Discussions link, with no trailing slide deck (unlike every other file in this group).
```

## chapter_preliminaries

```
file: chapter_preliminaries/lookup-api.md
heading_line: 223
n_exercises: 3
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: none
discussions: tabbed(4 tabs: mxnet t/38, pytorch t/39, tensorflow t/199, jax t/17972)
defects:
  - L235: exercise 3 embeds raw HTML entities inline in prose — "discover&nbsp;&rarr;&nbsp;inspect&nbsp;&rarr;&nbsp;read&nbsp;&rarr;&nbsp;verify" — literal `&nbsp;`/`&rarr;` codes instead of plain text or unicode arrows.
clarity:
  - none found — all three exercises state a concrete action and a verifiable outcome.
notable: Followed by a 6-slide deck (§2.7 "Using an unfamiliar API") to EOF (L345), thematically matched to the exercises. Exercise 3 is the only exercise in the group's early files that directs the reader to consult a coding assistant.
```

```
file: chapter_preliminaries/autograd.md
heading_line: 832
n_exercises: 8
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: none
discussions: tabbed(4 tabs: mxnet t/34, pytorch t/35, tensorflow t/200, jax t/17970)
defects:
  - none found (math delimiters all balanced; list/blank-line structure intact)
clarity:
  - none found — ex 1-3 are direct conceptual questions; ex 4-8 form a well-specified chained derivation.
notable: Exercises 4-8 form an explicit interdependent chain (ex6 references "the aforementioned function," ex7 "the graph...constructed previously," ex8 "the graph of exercise 5") — a hardcoded plain-text cross-reference to another exercise's number rather than a :numref:-style link, which would go stale if the list were reordered. Followed by the group's longest trailing slide deck (L832→1127), with framework-conditional slide attributes (only="jax", except="pytorch", etc.) reflecting real framework divergence.
```

```
file: chapter_preliminaries/probability.md
heading_line: 1240
n_exercises: 8
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 1 (:cite:`Mangram.2013`, L1259)
crossrefs: 1 (:numref:`subsec_probability_hiv_app`, L1251)
subproblems: nested-list(3,7,8) — ex3 (L1245-1247, 3 sub-items), ex7 (L1252-1254, 3 sub-items), ex8 (L1256-1259, 4 sub-items); all indented 4 spaces, nesting renders correctly
discussions: tabbed(4 tabs: mxnet L1261-1263, pytorch L1265-1267, tensorflow L1269-1271, jax L1273-1275)
defects:
  - none found (blank line after heading at L1241; no stray markup or broken links)
clarity:
  - none flagged — all 8 items have a concrete prove/compute/derive/example-give deliverable; ex8's Markowitz sub-question explicitly scopes out solving the QP ("beyond the scope of this book"), a clarifying bound, not a defect.
notable: A large non-exercise slide block (Pandoc `::: {.slide}` divs) follows the Discussions tabs from L1277 to EOF (L1721) — not part of Exercises/Discussions proper.
```

```
file: chapter_preliminaries/ndarray.md
heading_line: 981
n_exercises: 6
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0 (informal prose references only — "the broadcasting section" L987, "the saving-memory discussion" L988 — plain text, not :numref:)
subproblems: none
discussions: tabbed(4 tabs: mxnet L990-992, pytorch L994-996, tensorflow L998-1000, jax L1002-1004)
defects:
  - none found (code spans/italics closed; blank line after heading L982)
clarity:
  - none flagged — every item is a concrete run/predict/verify task with an implicit or explicit check.
notable: Most code-centric Exercises section in the group — every item references real API calls (`X == Y`, `arange`, `axis=`, `keepdims=True`, `id()`). Same trailing slide block (L1008→EOF L1486) follows Discussions.
```

```
file: chapter_preliminaries/calculus.md
heading_line: 470
n_exercises: 11
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: none — several items (1,5-11) wrap onto indented continuation lines (3 spaces), but these are lazy paragraph continuations of the same list item, not nested sub-lists.
discussions: tabbed(4 tabs: mxnet L500-502, pytorch L504-506, tensorflow L508-510, jax L512-514)
defects:
  - none found — 3-space continuation indents keep text inside the same list item without breaking nesting.
clarity:
  - none flagged — all 11 items are standard prove/derive/compute/plot instructions with clear deliverables; ex11 is a well-specified multi-step numeric experiment (three explicit learning rates) with an explicit closing question.
notable: Largest exercise count in chapter_preliminaries (11); purely mathematical/derivation-based, no code, contrasting with ndarray.md's code-heavy style. Same trailing slide block (L518→EOF L847).
```

```
file: chapter_preliminaries/pandas.md
heading_line: 281
n_exercises: 7
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0 (4 plain markdown hyperlinks instead — UCI repository L283, pandas indexing docs L284, NumPy `load` docs L289, Pillow L289 — all well-formed, none broken)
subproblems: none
discussions: tabbed(4 tabs: mxnet L291-293, pytorch L295-297, tensorflow L299-301, jax L303-305)
defects:
  - none found — blank line after heading (L282), links well-formed.
clarity:
  - ex 6 (L288): "How would you handle a categorical column with a very large number of categories? What if every label is unique: should you include the column at all?" — discussion-style question with no requested artifact or success criterion.
  - ex 7 (L289): "What alternatives to pandas can you think of?" — "can you think of" framing with no count/format/deliverable requested; borders on house-style empty framing.
notable: Exercises reference concrete objects from the section's own worked example (NumRooms, categorical column, standardization step), so no dangling context references. Same trailing slide block (L309→EOF L513).
```

```
file: chapter_preliminaries/linear-algebra.md
heading_line: 1393
n_exercises: 12
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: none
discussions: tabbed(4 tabs: mxnet L1408, pytorch L1412, tensorflow L1416, jax L1420)
defects:
  - none found
clarity:
  - ex 6 (L1400): "Run `A / A.sum(axis=1)` and see what happens. Can you analyze the results?" — no stated expectation or comparison target, so a reader cannot tell what a correct "analysis" looks like.
notable: Largest single exercise count in the group (12). Exercises + Discussions (L1393-1422) followed by a large unrelated slide deck (~470 lines, L1424-1895). Ex 8-9 are legitimate research-flavored complexity/memory-layout questions with concrete deliverables, not clarity issues.
```

## chapter_linear-regression

```
file: chapter_linear-regression/synthetic-regression-data.md
heading_line: 395
n_exercises: 5
numbering: repeated-1
names: some(2/5)
name_style: bold-parenthetical (e.g. "**(Reproducibility.)**" L402, "**(Signal-to-noise and recovery.)**" L409) — distinct from both the italic-period and bracket-tag anchor styles; items 1-3 (L397, L398, L401) are unnamed.
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 1 (:cite:`Naor.Reingold.1999`, L400)
crossrefs: 2 (:numref:`sec_linear_scratch` L411, :numref:`sec_linear_concise` L412)
subproblems: nested-list(exercise 2, L399-400: two sub-items, correctly indented 4 spaces)
discussions: tabbed(4 tabs: mxnet L419, pytorch L423, tensorflow L427, jax L431)
defects:
  - none found
clarity:
  - ex 2a (L399): "What happens if we cannot hold all data in memory?" — no required artifact or success criterion, unlike sibling 2b which specifies a concrete efficient-algorithm design task.
notable: Same trailing slide deck (L435-639). Exercises 4-5 are unusually long/well-specified relative to 1-3, giving the section mixed granularity between terse bare-list items and detailed named ones. Only file in the group whose named exercises use bold-parenthetical naming.
```

```
file: chapter_linear-regression/linear-regression.md
heading_line: 811
n_exercises: 8
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 1 (:cite:`Black.Scholes.1973`, L834)
crossrefs: 2 (:numref:`sec_weight_decay` L825, :numref:`sec_mdl-distributions` L826)
subproblems: nested-list(exercises 1, 4, 5, 7, 8 — sub-counts 3, 6, 3, 3, 4 respectively; all indented 4 spaces, correctly nested)
discussions: tabbed(3 tabs: mxnet L842, pytorch L846, tensorflow L850) — no jax tab
defects:
  - L842-850: Discussions block has only mxnet/pytorch/tensorflow Discourse links; the jax tab is missing even though the section body itself uses `%%tab jax` blocks (L66, 266, 527, 600, 652), and sibling files (linear-algebra.md, synthetic-regression-data.md, weight-decay.md) all include a jax Discussions tab. Verified via grep — confirmed missing.
clarity:
  - none found — all 8 items (and sub-items) state a concrete prove/derive/design/implement task with a clear artifact.
notable: Heaviest sub-problem nesting in the group (up to 6 sub-items under exercise 4). "Can you...?" phrasing appears 3 times (L816, L828, L829) but functions as the genuine instruction verb (find/fix), not filler — not flagged as a tone violation. Same trailing slide deck (L850-1224 approx.).
```

```
file: chapter_linear-regression/weight-decay.md
heading_line: 751
n_exercises: 6
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 3 (:numref:`fig_mdl-bias-variance-u-curve` L753, :numref:`subsec_lin-algebra-norms` L756, :numref:`fig_wd-map-prior` L758)
subproblems: none (all 6 items are flat, single-paragraph)
discussions: tabbed(4 tabs: mxnet L760, pytorch L764, tensorflow L768, jax L772)
defects:
  - none found
clarity:
  - ex 2 (L754): "Is it really the optimal value? Does this matter?" — two open rhetorical questions with no comparison target or expected artifact.
  - ex 5 (L757): "Review the relationship between training error and generalization error... what other ways might help us deal with overfitting?" — open brainstorm with no bound on scope; "Review..." reads as think-about-it rather than a task with a deliverable.
notable: Shortest exercises section in chapter_linear-regression (6 flat items, no nesting). Discussions block is complete across all 4 frameworks, consistent with `%%tab jax` usage throughout the body (L71, 318, 387, 425, 538, 662, 711) — a useful positive contrast to linear-regression.md's missing jax tab.
```

```
file: chapter_linear-regression/generalization.md
heading_line: 549
n_exercises: 8
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: none
discussions: tabbed(4 tabs: mxnet, pytorch, tensorflow, jax)
defects:
  - none found
clarity:
  - none clearly flagged — all 8 items (L551-558) are standard "why/can you/derive" prompts with a concrete ask; ex 8 references "the polynomial-fitting experiment above" and `n_train`, both of which genuinely exist earlier in the file (code around L342-405) — a valid reference, not a defect.
notable: Exercise list itself is short (8 bare items, ~8 lines) relative to the rest of the file. After the tabbed Discussions block (L560-574), an extensive reveal-slide deck (~350 lines, L576-925) runs to EOF, unrelated to the exercises.
```

```
file: chapter_linear-regression/linear-regression-concise.md
heading_line: 438
n_exercises: 6
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 3 (:numref:`subsec_linear-regression-loss-function` L444, :numref:`fig_linreg-loss-menu` L444, :numref:`sec_linear_scratch` L450 — all valid targets)
subproblems: nested-list(exercise 5) — sub-items L448-449 use sequential numbering ("1.", "2."), correctly indented 4 spaces.
discussions: tabbed(4 tabs: mxnet, pytorch, tensorflow, jax)
defects:
  - L451-452: double blank line before the Discussions block — cosmetic only, does not change rendering, noted for completeness rather than as a real defect.
clarity:
  - ex 4 (L446): "What is the effect on the solution if you change the learning rate and the number of epochs? Does it keep on improving?" — no range of values and no metric/threshold for "improving" — underspecified vary-a-parameter prompt.
notable: Exercise 2 (L441-444) embeds a displayed Huber-loss formula and the file's only :numref: cross-references. Exercise 5's nested sub-items use sequential (1,2) numbering, differing from linear-regression-scratch.md's repeated-"1." nested sub-items — an inconsistency in sub-item numbering convention between two closely related sibling files.
```

```
file: chapter_linear-regression/oo-design.md
heading_line: 688
n_exercises: 6
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: "(Advanced)" (exercise 5, L692, verbatim prefix); "Optional:" (exercise 6, L693, verbatim inline)
citations: 0
crossrefs: 0
subproblems: inline-letters(exercise 1) — L690 packs "(a) ... (b) ..." as two sub-questions inline within a single paragraph rather than as nested list items.
discussions: tabbed(4 tabs: mxnet, pytorch, tensorflow, jax)
defects:
  - L690: Exercise 1 crams two sub-questions as inline "(a) Add a method `greet(self)`...(b) What happens if you define `greet` *without* the decorator..." lettering inside one paragraph instead of a clean nested list — per rubric, this is a defect (should be split into indented nested list items).
clarity:
  - none clearly flagged — all 6 items state a concrete task and an implicit or explicit success criterion.
notable: Only file in the group with explicit difficulty/optionality markers ("(Advanced)", "Optional:"). After the Discussions block, a reveal-slide deck (L713-1043) unrelated to the exercises runs to EOF.
```

```
file: chapter_linear-regression/linear-regression-scratch.md
heading_line: 689
n_exercises: 8
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: nested-list(exercise 7) — sub-items L708-710 all use repeated "1.", correctly indented 4 spaces; a continuation line (L711, "Hint: ...") indented 7 spaces to align under the sub-item's content column.
discussions: tabbed(4 tabs: mxnet, pytorch, tensorflow, jax)
defects:
  - none found — wrapped-text continuations (L692, L697-701, L703, L706, L711) consistently align with their list marker's content column.
clarity:
  - ex 6 (L705-706): "Experiment using different learning rates to find out how quickly the loss function value drops. Can you reduce the error by increasing the number of epochs of training?" — no specific learning-rate range or target error/metric — underspecified vary-a-parameter prompt.
notable: Exercises 2 and 3 (L693-701) are physics-themed narrative exercises (Ohm's law, Planck's law), ex 3 embedding a displayed Planck's-law formula. Ex 7's nested sub-items use repeated-"1." numbering, differing from linear-regression-concise.md's sequential (1,2) sub-item numbering — same cross-file inconsistency noted there. After Discussions, a reveal-slide deck (L730-1007) runs to EOF.
```

---

## Group-level summary

**Scope:** 16 files, 109 exercises total (`## Exercises` sections only, Discussions blocks included).

**Dominant style:** Uniformly the legacy **bare list** (repeated `1.` auto-numbering, no names, no bracket tags) across all 16 files — the "Named + tagged" anchor style (e.g. chapter_reinforcement-learning) does not appear anywhere in this group. `tag_vocab` is `n/a` everywhere; no bracketed type tags exist in this group at all.

**Names:** Only 1/16 files has any named exercises — `synthetic-regression-data.md`, some(2/5), using a **bold-parenthetical** style (`**(Reproducibility.)**`) not seen elsewhere in the rubric's anchor examples. All other 15 files: `names: none`.

**Difficulty markers:** Only 1/16 files — `oo-design.md`, verbatim `(Advanced)` and `Optional:`.

**Citations/crossrefs:** Sparse and chapter-clustered. Citations: 3 total (`probability.md`, `linear-regression.md`, `synthetic-regression-data.md`). Crossrefs (`:numref:`): 11 total, concentrated in `chapter_linear-regression` (synthetic-regression-data 2, linear-regression 2, weight-decay 3, linear-regression-concise 3) plus 1 in `probability.md`. The other 10 files use zero formal citations/crossrefs (informal prose pointers instead, e.g. ndarray.md).

**Worst formatting defects:**
1. `linear-regression.md` L842-850 — Discussions block missing the jax tab despite jax content in the section body (confirmed via grep); sibling files in the same chapter (weight-decay.md) have all 4 tabs.
2. `chapter_introduction/index.md` L1593 — Discussions is a bare single link, not tabbed per-framework, unlike every sibling chapter.
3. `oo-design.md` L690 — inline `(a)/(b)` sub-lettering crammed into one paragraph instead of a nested list.
4. `lookup-api.md` L235 — literal HTML entities (`&nbsp;`, `&rarr;`) left un-rendered in prose.
5. `linear-regression-concise.md` L451-452 — cosmetic double blank line before Discussions (minor).

**Worst clarity offenders (all "vary X, see what happens" / think-about-it with no metric or deliverable):**
- `weight-decay.md` ex2, ex5; `linear-regression-concise.md` ex4; `linear-regression-scratch.md` ex6; `linear-algebra.md` ex6; `pandas.md` ex6-7; `chapter_introduction/index.md` ex1-2; `chapter_preface/index.md` ex3 (navigation, not a task); `synthetic-regression-data.md` ex2a.

**Cross-file inconsistency:** nested sub-item numbering differs between `linear-regression-concise.md` (sequential 1,2) and `linear-regression-scratch.md` (repeated 1.) for structurally similar exercises.
