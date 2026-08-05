# Exercise Review — d2l-neu, all live chapters

**Scope.** Every `## Exercises` section in the chapters listed in `_quarto.yml`,
read from the `.md` sources (the source of truth; `.qmd` files are generated).
That is **201 files across 31 chapters, containing 1,186 exercises** (1,218 raw
list items; the difference is `chapter_builders-guide`, which repeats each
exercise inside per-framework `:begin_tab:` blocks — 4 real exercises can appear
as 16 items).

**Method.** Twelve parallel reviewers each read one chapter group in full and
profiled every file against a fixed rubric (naming, tags, difficulty markers,
citations, subproblem format, formatting defects, clarity). A separate
mechanical scan of all 201 files cross-checks the counts (numbering regime,
name/tag markup, citation and cross-reference density, Discussions-block
state). Headline defects were re-verified by hand against the sources. The
twelve full per-file profiles are in the appendix.

---

## 1. Executive summary

The corpus is split between two eras, and the split explains almost everything
the review found.

- **Three chapters use a complete modern convention** — `reinforcement-learning`,
  `deep-reinforcement-learning`, and `recurrent-modern`: every exercise is
  typed with a bracketed tag (`[conceptual]`, `[short-code]`, `[extended]`),
  named, and precisely specified. RL and deep-RL are near-defect-free (1
  formatting defect in 88 exercises, no clarity flags). This is the obvious
  candidate for the book-wide target format.
- **Everything else is legacy d2l style or a local variant.** 15% of exercises
  have names, in at least five different markups. Difficulty markers
  effectively do not exist (4 ad-hoc instances corpus-wide, each formatted
  differently). Eight recommender-systems files still use *unnumbered bullet*
  exercises.
- **End matter is the most inconsistent dimension**: seven different
  Discussions-block states coexist, and **78 of 201 files (39%) have a dead
  placeholder link or none at all**. The healthy links all point at the
  upstream d2l.ai forum threads, which is itself a decision to revisit for
  this fork.
- **~30 hard formatting defects**, including one live rendering bug (an
  unclosed `$` in `hardware.md` that opens a math span over the rest of the
  section) and two files whose list nesting is broken (flush-left code fences
  in `fine-tuning.md`; 3-space indents in `bptt.md`).
- **~70 clarity flags in three recurring classes** — underspecified
  "vary X and see what happens" experiments (~25), reading prompts with no
  deliverable (~12), and "Can you…?" filler questions (~30, a direct
  violation of docs/style-guide.md §17.5) — plus **4 genuine content bugs**.
  Nearly all of this debt sits in un-rewritten legacy sections; the
  correlation between "rewritten for this book" and "clean" is almost perfect.

## 2. The conventions currently in use

### 2.1 Numbering (four regimes)

| Regime | Files | Where |
|---|---|---|
| Repeated `1.` (markdown auto-numbering) | 135 | dominant everywhere |
| Literal sequential `1. 2. 3.` | 41 | transformers (6/7), GP, HPO, most of the math appendix, half of attention |
| Mixed literal numbering in one list | 17 | scattered (usually collateral of nesting or edits) |
| Unnumbered `*` bullets | 8 | recommender-systems only (20 exercises) |

The split runs *within* chapters: `chapter_attention` is 3 files repeated-`1.`
vs 3 sequential; `vision-transformer.md` is the lone repeated-`1.` file in an
otherwise sequential transformers chapter. Sub-list numbering also differs
between sibling files (`linear-regression-concise.md` sequential vs
`linear-regression-scratch.md` repeated-`1.`).

### 2.2 Names (181 exercises, ≥5 markups)

| Style | Example | Where |
|---|---|---|
| Italic-period (modern) | `*Cost of a sweep.*` | RL ×44, deep-RL ×44, recurrent-modern ×23, dropout.md, generalization-deep.md |
| Bold-period | `**Descent guarantee.**` | math appendix (mdl-optimization ×13, mdl-dynamics ×19, mdl-calculus ×3), linear-regression ×6, MLP |
| Bold-parenthetical | `**(Reproducibility.)**` | synthetic-regression-data.md (2) |
| Italic-parenthetical | one file | mdl appendix |
| Bold name + bold tag | `**[conceptual]** **Name.**` | test-time-regression.md |

### 2.3 Type tags (133 exercises in 21 files, 3 chapters)

Vocabulary is exactly three values, used consistently where present:
`[short-code]` (implement/run something small), `[conceptual]`
(derive/prove/explain), `[extended]` (larger project). Corpus totals: 61 / 39 /
16 in plain square brackets, plus 17 more in recurrent-modern hidden behind
markup variants.

The markup, however, varies file-to-file even inside `recurrent-modern` —
five micro-conventions in seven files: plain `[tag]` only (lstm, matrix-state),
**bold** `**[tag]**` (deltanet), tag-then-name (ssm, mamba), name-then-tag
(hybrids), bold-tag + bold-name (test-time-regression). No two files agree
exactly.

One outlier: `mdl-mutual-information.md` uses *parenthetical topic* tags
(`(Data-processing)`, `(Synergy)`) — a different concept (subject, not type).

### 2.4 Difficulty markers (essentially none)

Four ad-hoc instances in 1,186 exercises: `*Advanced*` twice in HPO (with
inconsistent punctuation/placement between the two), `(Advanced)` and
`Optional:` in oo-design.md, and a lone `(*)` in multilayer-perceptrons.
There is no working difficulty scheme; `[extended]` is the nearest thing.

### 2.5 Citations and cross-references (very uneven)

105 `:cite:`/`:citet:` uses and 505 `:numref:`/`:eqref:` uses inside exercise
sections. Density tracks recency: deep-RL alone has 30 citations and 55
crossrefs; GP has zero of both. `resnet.md` and `scaling-laws.md` mix `:cite:`
and `:citet:` within a single exercise set. `fast-transformer.md` ex 8
hardcodes "§13.6" instead of `:numref:`. All sampled keys resolve (the odd
`*1`-suffixed resnet bib key is real, not a typo).

### 2.6 Subproblems (clean nesting vs inline cramming)

Clean 4-space nested sub-lists are the norm in the conv chapters, HPO,
optimization, and foundations. Against that, **~15 exercises cram subproblems
inline** as `a) … b) …`, `(i) … (ii) …` in one paragraph:
`classification.md` (×4), `oo-design.md`, `mlp-implementation.md` ex 9,
`hardware.md` ex 2, `qlearning.md` ex 4, `convergence.md` (a–d),
`relativistic.md` (a–c), and seven instances across the math appendix
(convexity ×2, adaptive-stochastic-methods, fokker-planck, sdes,
score-matching-diffusion-flow, concentration-generalization).

### 2.7 End matter (seven Discussions states, plus slides)

| State | Files | Where |
|---|---|---|
| Tabbed, real per-framework thread links | 89 | the pre-rewrite mainline (74 with 4 tabs; 15 with 1–3 tabs, several *missing* tabs their section content warrants) |
| Tabbed, but all tabs share one URL | 13 | recsys ×9, HPO ×2, mdl ×2 |
| Single real link | 16 | scattered |
| **Single dead placeholder** (`d2l.discourse.group/` root) | **29** | RL ×7, deep-RL ×7, GANs ×7, recurrent-modern ×6, adamw, batch-size |
| Tabbed dead placeholder | 6 | mdl-dynamics ×4, mdl-calculus, mdl-info |
| Prose `## Discussions` section instead | 5 | all of mdl-optimization |
| **None at all** | **43** | builders-guide ×8, appendix-tools ×8, attention ×6, transformers ×7, comp-perf ×7, conv-modern ×3 (its three newest files), optimization ×3 (muon, practice, scaling), mdl-prob ×1 |

The pattern: **every chapter written or rewritten recently either dropped the
Discussions block or left a dead placeholder.** The 89 healthy links all point
at the upstream d2l.ai Discourse threads — worth an explicit decision for this
fork. 178/201 sections are followed by instructor slide decks (`<!-- slides -->`);
the gaps are builders-guide (0/8), appendix-tools (0/8), and a few NLP/recsys/GP
files.

## 3. Hard formatting defects (ranked)

**Rendering bugs (fix first):**

1. `chapter_computational-performance/hardware.md:511` — unclosed `$` in
   "At $0.30/kWh…"; the math span stays open across the rest of the exercise
   section (verified: next literal `$` is in the following section's slides).
2. `chapter_computer-vision/fine-tuning.md:799–852` — eight framework code
   fences flush-left instead of indented under their parent items (ex 3, 4),
   which breaks the list and renumbers subsequent exercises.
3. `chapter_recurrent-neural-networks/bptt.md:303–305` — sub-items indented
   3 spaces (needs 4), so they do not render as a nested list. Same class of
   defect in `mdl-geometry-linear-algebraic-ops`, `mdl-svd-low-rank`,
   `mdl-eigendecomposition`.
4. `chapter_preliminaries/lookup-api.md:235` — literal `&nbsp;`, `&rarr;`
   entities in prose; `chapter_transformers/vision-transformer.md:548` —
   HTML-entity `<cls>` where the same file's slides use a code span.

**Systematic inconsistencies:**

5. 35 dead placeholder Discussions links (§2.7), 43 missing blocks, 13
   all-tabs-identical blocks.
6. Missing framework tabs: `linear-regression.md` (no jax tab despite jax
   content), `bounding-box.md` and `transposed-conv.md` (no tensorflow tab
   despite tensorflow cells), `reproducibility-inspection.md` ex 5 (no
   pytorch variant — see also content bug #1 below).
7. `model-construction.md` — exercise tabs in a non-canonical framework
   order vs the rest of the same file; `subword-embedding.md` — unique
   combined `tensorflow,jax` tab; `rs-async.md` — 1 tab vs 3 in its chapter
   siblings.

**Local slips (complete list in appendix):** grammar/typos in `seqrec.md`
(415–416), `fm.md` (167–168), `word-embedding-dataset.md` (599); blank-line
noise (3 blank lines in `subword-embedding.md`; doubles in
`sentiment-analysis-and-dataset.md`, `linear-regression-concise.md`; missing
blank after the heading in 4 mdl files; stray mid-list blanks isolating
trailing exercises in 3 mdl-optimization files); an orphan nested `3.1` with
no `3.2` in `hyperopt-intro.md`; a `q_learning_optimization_problem` eqref key
breaking `qlearning.md`'s own `eq_` prefix convention.

## 4. Content bugs (4)

1. `chapter_builders-guide/reproducibility-inspection.md` ex 4 (shared,
   untabbed) asserts "a forward hook that returns a value *replaces* the
   module's output" — true only for PyTorch. The MXNet tab of ex 5 explicitly
   patches this ("Gluon ignores a hook's return value… edit the block's
   `forward` instead"), but the TensorFlow and JAX tabs never do, so readers
   on those frameworks get an exercise their framework cannot express
   (verified: the section's own text says TF hooks are observe-only / have no
   equivalent).
2. `chapter_multilayer-perceptrons/backprop.md` ex 6.d — six Greek-letter
   variables that are never mapped to the described scenario.
3. `chapter_optimization/gd.md` ex 2 — describes "line search" but specifies
   a binary-search procedure; the two don't match as written.
4. `chapter_convolutional-neural-networks/padding-and-strides.md` ex 2 —
   asks about audio signals in a section that never introduces audio
   (minor, but a non sequitur as written).

## 5. Clarity flags (~70, three classes)

**A. Underspecified "vary X and see what happens" (~25).** No metric, no
range, no comparison, no artifact. Concentrated in: recsys (7 instances across
mf, neumf, autorec, deepfm), foundations (weight-decay ex 2/5,
linear-regression-concise ex 4, linear-regression-scratch ex 6, linear-algebra
ex 6, pandas ex 6–7, introduction ex 1–2), optimization legacy files (adam
ex 1, momentum ex 1/4, gd ex 1, minibatch-sgd ex 1/3), batch-norm ex 3/5,
computer-vision (kaggle-dog ex 1–2, neural-style ex 1, ssd ex 2),
word-embedding-dataset ex 2, bert ex 2.

**B. Reading prompts with no deliverable (~12).** movielens ex 2 ("Go through
the site…"), hyperopt-intro ×2 (including "*Advanced*: Read [paper] for an
elegant approach" — literally just "read this"), anchor ex 5, neural-style
ex 4 (references a survey never covered), image-augmentation ex 3,
semantic-segmentation ex 1, finetuning-bert ex 2–3 ("Can we leverage BERT in
machine translation?"), queries-keys-values ex 1/3/4 (unmodified legacy
prompts: "Design a differentiable search engine…"), preface ex 3.

**C. "Can you…?" / "Can we…?" filler questions (~30).** The single most
common house-style violation (style-guide §17.5 bans false-intimacy
questions). Heaviest: computer-vision (8 files), NLP-applications (6 of 7
files), batch-norm (5 instances), numerical-stability-and-init (3), plus
alexnet, blocks, cnn-design, sequence, softmax-regression, backprop, dropout,
kaggle-house-price, optimization-intro, sgd, and 4 recsys items.

**Where clarity is already strong:** RL and deep-RL (0 flags in 88),
computational-performance (0 in 42), appendix-tools, the math appendix (2 soft
flags in 229), GAN chapter (dense but every item names an exact deliverable),
and standout single files elsewhere: seq2seq.md (explicit sweeps, thresholds),
natural-language-inference-bert.md ex 1 (numeric success criterion),
practice.md ex 1 ("Report the log, not just the winning configuration").

## 6. The quality gradient

The strongest single finding: **style and substance move together.** Groups
where exercises were rewritten around this book's own code and experiments
(RL, deep-RL, recurrent-modern, comp-perf, appendix-tools, the three newest
conv-modern files, 11 of 13 attention/transformers files, seq2seq) have
essentially zero clarity debt — and, revealingly, they are also the files with
dead or missing Discussions links, because the new content never inherited the
legacy end-matter. Conversely, nearly every clarity flag lives in untouched
legacy d2l material (recsys bullets, queries-keys-values, vision-transformer,
the Kaggle sections, NLP-applications). Two files in attention/transformers
account for every flag in that group, and both are the two not yet rewritten.

## 7. Decisions this sets up (for discussion)

1. **Target convention.** The RL format is the de-facto candidate:
   `1. [tag] *Name.* body` with 4-space nested sub-lists. To settle: the tag
   vocabulary (keep `[conceptual|short-code|extended]`? add `[math]`,
   `[experiment]`?), whether tags encode type only or also difficulty, and
   whether names are mandatory.
2. **Numbering.** Repeated-`1.` (edit-friendly) vs literal sequential
   (source-readable). Both are common; pick one.
3. **Discussions links.** 39% dead or missing, and the live ones point at
   upstream d2l.ai threads. Drop the blocks entirely, or stand up this fork's
   own forum? This decision unblocks ~78 files' end matter.
4. **Recsys bullets and thin sections.** The 8 bullet-style files need
   renumbering regardless; several sections (ctr.md: 1 exercise) are thin.
5. **Clarity rewrites.** The ~70 flags are concentrated and mechanical to
   triage: fix the 4 content bugs, then rewrite classes A–C
   (roughly 60 exercises) to the standard the new chapters already meet.
6. **Enforcement.** Once conventions are fixed, an exercise linter in
   `tools/` (numbering, tag vocabulary, name markup, indentation, end-matter
   state) plus a short section in the style guide would keep this from
   drifting again.

---

## Appendix

Per-file profiles (all 201 files) from the twelve group reviews, in
`scratchpad/exercise-review/`: 01-foundations.md, 02-classification-mlp.md,
03-convnets.md, 04-rnns.md, 05-attention-transformers.md,
06-optimization-perf.md, 07-computer-vision.md, 08-nlp.md, 09-rl.md,
10-gp-hpo-gans.md, 11-recsys-tools.md, 12-mdl.md. Mechanical scan data:
scan.tsv (names/tags/citations per file), scan2.tsv (tab-deduplicated counts,
Discussions state, slides).
