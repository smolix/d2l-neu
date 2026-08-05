# Exercise Review — chapter_recommender-systems + chapter_appendix-tools-for-deep-learning

Repo: /Users/smola/Repositories/github/d2l-neu (`.md` files only, read in full from
`## Exercises` heading to end of file). 18 files total, verified via
`grep -rln "^## Exercises" chapter_recommender-systems chapter_appendix-tools-for-deep-learning --include="*.md"`.

Chapter orderings (from each chapter's `index.md` toc, used to validate cross-references):
- recommender-systems: recsys-intro, movielens, mf, autorec, ranking, neumf, seqrec, ctr, fm, deepfm
- appendix-tools: interactive-development, hosted-notebooks, cloud-instances, hardware,
  software-ecosystem, training-systems, model-serving, developers-guide

---

## chapter_recommender-systems

```
file: chapter_recommender-systems/recsys-intro.md
heading_line: 78
n_exercises: 3
numbering: sequential
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: none
discussions: single-link (no mxnet/pytorch tabs — file is conceptual, no code)
defects: none found
clarity: none — all three exercises (construct two histories, give a false-negative
  example, design warm/cold-start splits) state a concrete deliverable
notable: shortest file in the group with no `<!-- slides -->` block afterward (file
  ends right after the Discussions link, total 88 lines). Exemplary: no tags/names but
  clean, precise, self-contained exercises.
```

```
file: chapter_recommender-systems/movielens.md
heading_line: 301
n_exercises: 2
numbering: other(unordered `* ` bullets, no numbers at all)
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: none
discussions: tabbed(2: mxnet, pytorch)
defects: none found
clarity:
  - ex 1: "What other similar recommendation datasets can you find?" — no deliverable
    (how many? compared how?); also reads as a "can you...?" filler question (house
    style, docs/style-guide.md).
  - ex 2: "Go through the https://movielens.org/ site for more information about
    MovieLens." — pure reading prompt, no task or artifact at all.
notable: both exercises are exploratory/reading-prompt style rather than
  constructive tasks — weakest pair in the chapter alongside ctr.md.
```

```
file: chapter_recommender-systems/mf.md
heading_line: 292
n_exercises: 3
numbering: other(unordered `* ` bullets, no numbers at all)
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: none
discussions: tabbed(2: mxnet, pytorch)
defects: none (double blank line at L297–298 before the begin_tab block — extra
  blank, not missing, so does not change rendering; noted but not counted as a defect)
clarity:
  - ex 1: "Vary the size of latent factors. How does the size of latent factors
    influence the model performance?" — no range for latent-factor size, no metric
    named (RMSE is defined earlier in the section but not referenced here).
  - ex 2: "Try different optimizers, learning rates, and weight decay rates." — no
    range, no comparison criterion.
  - ex 3: "Check the predicted rating scores of other users for a specific movie." —
    no success criterion; unclear what to report.
notable: ex 1 is near-verbatim identical in wording to neumf.md ex 1 ("Vary the size
  of latent factors...") — duplicated legacy boilerplate across two files.
```

```
file: chapter_recommender-systems/autorec.md
heading_line: 249
n_exercises: 3
numbering: other(unordered `* ` bullets, no numbers at all)
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: none
discussions: tabbed(2: mxnet, pytorch)
defects: none found
clarity:
  - ex 1: "Vary the hidden dimension of AutoRec to see its impact on the model
    performance." — no range, no metric named.
  - ex 3: "Can you find a better combination of decoder and encoder activation
    functions?" — "better" undefined (no metric/threshold); also a "Can you...?"
    filler-question tone violation per house style.
notable: ex 2 ("Try to add more hidden layers. Is it helpful...?") is borderline but
  at least ties to a yes/no outcome, so not flagged.
```

```
file: chapter_recommender-systems/ranking.md
heading_line: 159
n_exercises: 3
numbering: sequential
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: none
discussions: tabbed(2: mxnet, pytorch)
defects: none found
clarity: none — all three are precise derivation/construction tasks (differentiate
  and compare gradients at named limits; write and compare expectations for two
  sampling schemes; construct a concrete counterexample and explain a fix).
notable: exemplary file — sequential numbering, dense math, no filler, no defects.
  Best-specified file in the chapter.
```

```
file: chapter_recommender-systems/neumf.md
heading_line: 459
n_exercises: 4
numbering: other(unordered `* ` bullets, no numbers at all)
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: none
discussions: tabbed(2: mxnet, pytorch)
defects: none found
clarity:
  - ex 1: "Vary the size of latent factors. How does the size of latent factors
    impact the model performance?" — no range, no metric (Hit@50/AUC are defined
    earlier but not named here).
  - ex 2: "Vary the architectures ... of the MLP to check its impact on the
    performance." — no metric, no comparison protocol.
  - ex 3: "Try different optimizers, learning rate and weight decay rate." — same
    underspecified pattern.
notable: ex 4 ("Try to use hinge loss defined in the last section...") correctly
  refers to ranking.md, which does immediately precede this file in the chapter toc —
  a valid cross-reference, not a defect. 3 of 4 exercises are the classic
  legacy-d2l "vary X, see impact" pattern with zero success criteria.
```

```
file: chapter_recommender-systems/seqrec.md
heading_line: 413
n_exercises: 3
numbering: other(unordered `* ` bullets, no numbers at all)
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 1 (:cite:`Hidasi.Karatzoglou.Baltrunas.ea.2015`, standard :cite: form)
crossrefs: 0
subproblems: none
discussions: tabbed(3: mxnet, pytorch, plus a combined "tensorflow,jax" tab with
  substantive prose explaining the framework gap instead of a dead/missing pattern —
  a good practice, unique to this file in the group)
defects:
  - L415: "...which component is the more important ?" — stray space before the
    question mark; "the more important" is also an awkward comparative (should read
    "which component matters more" or similar).
  - L416: "Does longer historical interactions bring higher accuracy?" —
    subject–verb agreement error ("interactions" plural vs. "Does... bring" singular
    framing; should be "Do longer historical interactions bring..." or "Does a
    longer history bring...").
clarity:
  - ex 3: ends "Can you explain the differences between these two tasks?" — "Can
    you...?" filler-question tone violation (the citation and setup are fine; only
    the closing question is a tone issue).
notable: double blank line at L418–419 before the begin_tab block (extra, not
  missing — does not affect rendering). Ex 1's ablation task and ex 2's
  hyperparameter sweep are otherwise reasonably concrete.
```

```
file: chapter_recommender-systems/ctr.md
heading_line: 164
n_exercises: 1
numbering: other(unordered `* ` bullet, no number)
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: none
discussions: tabbed(2: mxnet, pytorch)
defects: none found
clarity: none — the single exercise ("Extend `CTRDataset` with an explicit path for
  continuous fields. How will you fit normalization or bin boundaries without
  leaking information from the test set?") is concrete and well-posed.
notable: only one exercise for the whole section — a conspicuously thin Exercises
  block compared to the rest of the chapter (next-shortest is 2). Worth a look for
  whether more exercises were intended.
```

```
file: chapter_recommender-systems/fm.md
heading_line: 165
n_exercises: 2
numbering: other(unordered `* ` bullets, no numbers at all)
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: none
discussions: tabbed(2: mxnet, pytorch)
defects:
  - L167: "Can you test FM on other dataset such as Avazu, MovieLens, and Criteo
    datasets?" — grammar: "other dataset" should be plural ("other datasets").
  - L168: "Vary the embedding size to check its impact on performance, can you
    observe a similar pattern as that of matrix factorization?" — comma splice
    joining two independent clauses into one run-on sentence, with a lowercase
    "can you" clause stitched onto the end.
clarity:
  - ex 1 and ex 2 are both "Can you...?" filler-question tone violations (L167,
    L168) per house style.
  - ex 1: no criterion for what "test on" should produce (a number? a comparison
    table?).
notable: weakest-written file in the chapter — the only one with two distinct
  grammar defects and two tone violations in just two exercises.
```

```
file: chapter_recommender-systems/deepfm.md
heading_line: 173
n_exercises: 2
numbering: other(unordered `* ` bullets, no numbers at all)
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: none
discussions: tabbed(2: mxnet, pytorch)
defects: none found
clarity:
  - ex 1: "Vary the structure of the MLP to check its impact on model performance."
    — no range, no metric.
  - ex 2: "Change the dataset to Criteo and compare it with the original FM model."
    — has a concrete comparison target (FM) but no metric named for the comparison
    (log loss was used earlier in the section but isn't referenced here); mild.
notable: none.
```

## chapter_appendix-tools-for-deep-learning

All eight files in this chapter share one consistent style: sequential-looking but
literally repeated `1.` numbering (classic Markdown auto-numbering), no names, no
tags, no difficulty markers, and — unlike every recommender-systems file — **no
Discussions block of any kind** (no per-framework tabs, no single link; each file
simply ends after its last exercise). None have a trailing `<!-- slides -->` block
either. Exercises are uniformly long-form, quantitative, and tied to a concrete
artifact or number to produce.

```
file: chapter_appendix-tools-for-deep-learning/model-serving.md
heading_line: 205
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
discussions: missing
defects: none found
clarity: none — all four exercises (extend the scheduler and compare policies on
  p95 TTFT; benchmark two serving engines at three concurrency levels; compute a
  KV-cache byte budget; design a cache-key policy) name a concrete metric or
  artifact and a success condition.
notable: file ends immediately after the last exercise (L218 = last line of file,
  no Discussions, no slides).
```

```
file: chapter_appendix-tools-for-deep-learning/cloud-instances.md
heading_line: 190
n_exercises: 4
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 2 (:numref:`tab_cloud_prices`, used in ex 1 and ex 2; label verified
  defined at chapter_appendix-tools-for-deep-learning/cloud-instances.md:58)
subproblems: inline-letters(ex 4: "(a) a marketplace host... and (b) a hyperscaler...")
discussions: missing
defects: none found
clarity: none — every exercise names a concrete deliverable (a priced comparison, a
  measured memory high-water mark matched against the table, a timed recovery
  drill, a trust-boundary diagram for two labeled cases).
notable: ex 1 explicitly asks the reader to check whether the book's own price
  table has gone stale — a nice self-aware, dated-content caveat consistent with
  the chapter's index.md note that "prices ... are dated to mid-2026."
```

```
file: chapter_appendix-tools-for-deep-learning/hardware.md
heading_line: 238
n_exercises: 4
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 2 (:numref:`tab_unified_memory` in ex 1, verified defined at
  chapter_appendix-tools-for-deep-learning/hardware.md:148; :numref:`tab_cloud_prices`
  in ex 3, a valid backward reference to cloud-instances.md, which precedes this
  file in the chapter toc)
subproblems: none
discussions: missing
defects: none found
clarity: none — all four name a specific number to compute (decode bound in tok/s,
  LoRA memory fit, break-even hours, $/GB) against a specific table or price.
notable: none.
```

```
file: chapter_appendix-tools-for-deep-learning/interactive-development.md
heading_line: 226
n_exercises: 4
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 1 (:numref:`fig_tools_kernel_state`, verified defined at
  chapter_appendix-tools-for-deep-learning/interactive-development.md:41)
subproblems: none
discussions: missing
defects: none found
clarity: none — all four exercises ask for a specific hands-on comparison
  (reproduce and detect an out-of-order dependency; compare `sys.executable` across
  three environments; time a matrix product with/without sync; map an SSH-tunnel
  session's components) with an explicit "explain" or "identify" closing.
notable: none.
```

```
file: chapter_appendix-tools-for-deep-learning/hosted-notebooks.md
heading_line: 251
n_exercises: 4
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 1 (:numref:`sec_bert-pretraining`, verified defined at
  chapter_natural-language-processing-pretraining/bert-pretraining.md:2 — a
  legitimate cross-chapter reference)
subproblems: none
discussions: missing
defects: none found
clarity: none — all four name a specific platform action and question (where the
  edited copy lives and its lifetime; extend a fingerprint without breaking CPU
  runtime; compare two saved versions; estimate a fine-tuning time budget against
  a quota).
notable: none.
```

```
file: chapter_appendix-tools-for-deep-learning/software-ecosystem.md
heading_line: 202
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
discussions: missing
defects: none found
clarity: none — ex 3 references "your shortlist from Exercise 1," a valid
  intra-section reference (not a nonexistent-context violation). All four name a
  concrete artifact (a 3-model shortlist with disagreement analysis, a file
  inventory with a code-execution-risk flag, a 25-example eval set with a
  match/mismatch verdict, a license-obligations checklist).
notable: none.
```

```
file: chapter_appendix-tools-for-deep-learning/training-systems.md
heading_line: 232
n_exercises: 4
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 1 (:numref:`sec_bert-pretraining`, same valid label as above)
subproblems: none
discussions: missing
defects: none found
clarity: none — all four name a specific quantity or artifact (a world-size
  threshold, three batch/world-size/accumulation combinations with throughput
  reasoning, a measured multi-GPU speedup vs. the ideal 2x, a checkpoint/restore
  design with per-rank vs. global state identified).
notable: none.
```

```
file: chapter_appendix-tools-for-deep-learning/developers-guide.md
heading_line: 180
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
discussions: missing
defects: none found
clarity: none — all four are concrete build/contribution tasks with a measurable
  or verifiable outcome (rebuild timing, a docstring change verified via import, an
  agent-driven micro-fix reviewed as a maintainer would, a side-by-side notebook
  diff traced to its source).
notable: ex 3 ("Use a coding agent...") is a notably contemporary/meta exercise —
  appropriate for this chapter's subject matter, not a tone violation.
```

---

## Group-level summary

See final message for the required summary (max ~30 lines).
