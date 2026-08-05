# Exercise Review — chapter_attention, chapter_transformers

Files enumerated via `grep -rln "^## Exercises" chapter_attention chapter_transformers --include="*.md"` (13 files).

---

## chapter_attention/attention-at-scale.md

```
file: chapter_attention/attention-at-scale.md
heading_line: 981
n_exercises: 6
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 7 (:eqref: x3 — ex1 x2, ex6 x1; :numref: x4 — ex3 x1, ex4 x1, ex6 x2)
subproblems: none (multi-part questions packed into single dense paragraphs, no lettering/nesting)
discussions: missing (file has no Discussions block at all; `<!-- slides -->` deck follows the last exercise directly, L1028)
defects: []
clarity: []
notable: Every item is a long compound paragraph (derive → implement → measure → explain), each grounded in a specific in-section function (`chunked_attention`) or equation. Dense but each sub-question is concrete and answerable; no ambiguity found. Confirmed `chunked_attention` is defined earlier in the file (L303, L335) so ex2/ex3 do not reference missing context.
```

## chapter_attention/attention-scoring.md

```
file: chapter_attention/attention-scoring.md
heading_line: 623
n_exercises: 6
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 1 (:eqref:`eq_softmax_QK_V`, ex3)
subproblems: none
discussions: missing (slides follow directly, L651)
defects: []
clarity: []
notable: All six items reference the section's own code (`DotProductAttention`, `masked_softmax`) precisely; ex5 is a legitimate "implement and time" performance-comparison exercise with a clear deliverable.
```

## chapter_attention/multihead-attention.md

```
file: chapter_attention/multihead-attention.md
heading_line: 541
n_exercises: 6
numbering: sequential
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 2 (:eqref:`eq_multihead-flops`, ex4 and ex5)
subproblems: none
discussions: missing (slides follow directly, L574)
defects: []
clarity: []
notable: First file in the chapter to switch from repeated-1 to sequential numbering (see attention-at-scale.md and attention-scoring.md above, both repeated-1) — a numbering-style change mid-chapter. Ex6 ("design an experiment to measure how much each head matters... guard against two heads that are individually prunable but not jointly") is open-ended but states a concrete deliverable (an experiment design) and a specific failure mode to guard against, so it is not flagged as underspecified.
```

## chapter_attention/positional-information.md

```
file: chapter_attention/positional-information.md
heading_line: 709
n_exercises: 6
numbering: sequential
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: none
discussions: missing (slides follow directly, L743)
defects: []
clarity: []
notable: Exercises are tightly bound to section internals (`TinyCharLM._rope`, the `'none'`/`'rope'`/`'alibi'` model variants, the four-times-training-length results table). Verified `TinyCharLM`, `_rope`, and ALiBi are all defined/discussed earlier in the same file (L340, L390, L411) — no dangling references despite the exercises sitting after a slide deck that duplicates this material.
```

## chapter_attention/queries-keys-values.md

```
file: chapter_attention/queries-keys-values.md
heading_line: 268
n_exercises: 6
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 1 (:cite:`Hu.Shen.Sun.2018`, ex4)
crossrefs: 1 (:eqref:`eq_softmax_attention`, ex2)
subproblems: none
discussions: missing (slides follow directly, L293)
defects: []
clarity:
  - ex 1: "Suppose that you wanted to reimplement approximate (key, query) matches as used in classical databases, which attention function would you pick?" — "classical databases" approximate-match schemes are never introduced in this section or chapter; no success criterion for "which... would you pick."
  - ex 3: "Design a differentiable search engine using the attention mechanism." — no deliverable, scope, or success criterion; a bare open-ended prompt.
  - ex 4: "Review the design of the Squeeze and Excitation Networks :cite:`Hu.Shen.Sun.2018` and interpret them through the lens of the attention mechanism." — a reading/interpretation prompt with no stated artifact to produce.
notable: This is the one file in the group with no code or experiment references at all — unmodified legacy d2l phrasing throughout ("Suppose that you wanted to...", "Design a..."), unlike its five siblings in the same chapter, which are all rewritten around this book's own experiments (Nadaraya–Watson demo, etc.). Matches the rubric's "bare list" anchor style exactly.
```

## chapter_attention/what-attention-computes.md

```
file: chapter_attention/what-attention-computes.md
heading_line: 755
n_exercises: 6
numbering: sequential
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 3 (:numref:`sec_mdl-geometry-linear-algebraic-ops`, ex1; :eqref:`eq_qkov` and :eqref:`eq_copy-matrix`, ex6)
subproblems: none (no lettering, but ex1, L757-771, is an unusually long single paragraph specifying ~5 sequential construction steps plus 2 questions)
discussions: missing (slides follow directly, L805)
defects: []
clarity: []
notable: The most technically dense file in the group (mechanistic-interpretability content: QK/OV circuits, induction heads, K-composition). Ex1 in particular crams a full hand-constructed-circuit procedure into one list item without nested sub-steps; not a formatting defect under the rubric's letter-crammed-subproblem definition (no inline "a) b) c)" lettering used), but flagged here as a candidate for restructuring into a nested list for readability. All referenced objects (`TinyCharLM`, `model_two`, `head_scores`) confirmed defined earlier in the file.
```

## chapter_transformers/encoders-decoders.md

```
file: chapter_transformers/encoders-decoders.md
heading_line: 1031
n_exercises: 6
numbering: sequential
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 1 (:cite:`Devlin.Chang.Lee.ea.2018`, ex1)
crossrefs: 1 (:numref:`sec_multihead-attention`, ex3)
subproblems: none
discussions: missing (slides follow directly, L1063)
defects: []
clarity: []
notable: Uses `<mask>` correctly as a code span throughout (ex1, L1033-1038) — contrast with the HTML-entity defect found in vision-transformer.md below. `sample_batch` and `PerceiverEncoder` (ex4-6) both confirmed defined earlier in file (L387/406, L795/825).
```

## chapter_transformers/gpt.md

```
file: chapter_transformers/gpt.md
heading_line: 716
n_exercises: 6
numbering: sequential
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 1 (:cite:`Holtzman.Buys.Du.ea.2020`, ex3)
crossrefs: 2 (:numref:`sec_transformer-block`, ex1; :numref:`sec_positional-information`, ex5)
subproblems: none
discussions: missing (slides follow directly, L757)
defects: []
clarity: []
notable: Clean, well-scoped set; each item names an exact quantity to compute/measure (parameter fraction, entropy at four temperatures, bits-per-character comparison against GPT-2).
```

## chapter_transformers/kv-cache.md

```
file: chapter_transformers/kv-cache.md
heading_line: 1272
n_exercises: 6
numbering: sequential
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: none
discussions: missing (slides follow directly, L1308)
defects: []
clarity: []
notable: Systems/performance-flavored exercises (bytes/token, tokens-per-second sweeps, latency before/after a fix) — every item states a measurable quantity and, where relevant, an expected before/after comparison.
```

## chapter_transformers/moe.md

```
file: chapter_transformers/moe.md
heading_line: 764
n_exercises: 6
numbering: sequential
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 1 tag / 2 works (:cite:`Lepikhin.Lee.Xu.ea.2021,fedus2022switch`, ex2)
crossrefs: 0
subproblems: none
discussions: missing (slides follow directly, L810)
defects: []
clarity: []
notable: Ex3 and ex6 refer to "the triptych" — an informal label for the three-run balancing comparison shown under the slide title "Balancing methods at fixed compute" (L881-897); the term itself never appears in the section prose, only in the exercises. Not flagged as a hard defect (the referent is unambiguous from context and figure ID `moe-three-runs-one-budget`), but it is looser terminology than the rest of the group's precise code/quantity references.
```

## chapter_transformers/scaling-laws.md

```
file: chapter_transformers/scaling-laws.md
heading_line: 669
n_exercises: 6
numbering: sequential
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 2 (:cite:`Kimi.Team.2025`, ex3; :citet:`Muennighoff.Rush.Barak.ea.2023`, ex2) — mixed :cite:/:citet: forms within the same exercise set
crossrefs: 4 (:eqref:`eq_six_nd` x2 — ex1, ex6; :numref:`tab_modern-recipe`, ex3; :numref:`sec_scaling`, ex5)
subproblems: none
discussions: missing (slides follow directly, L710)
defects: []
clarity: []
notable: Ex6 is a genuine "predict twice, then run" exercise with two explicit competing predictions and a stated criterion for adjudicating between them — a strong example of a well-specified open-ended exercise per the rubric's "do not flag" carve-out.
```

## chapter_transformers/transformer-block.md

```
file: chapter_transformers/transformer-block.md
heading_line: 747
n_exercises: 6
numbering: sequential
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 1 (:citet:`Shazeer.2020`, ex4)
crossrefs: 1 (:eqref:`eq_dot_product_attention`, ex6)
subproblems: none
discussions: missing (slides follow directly, L785)
defects: []
clarity: []
notable: Clean, uniform set; every item names the exact code change (one line in `FeedForward`, wrapping `MultiHeadAttention` in an `attn_factory`) and the exact measurement to take.
```

## chapter_transformers/vision-transformer.md

```
file: chapter_transformers/vision-transformer.md
heading_line: 542
n_exercises: 5
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 1 (:numref:`sec_attention-at-scale`, ex1)
subproblems: none
discussions: missing (slides follow directly, L565)
defects:
  - L548: exercise text writes the class token as HTML-entity-escaped “&lt;cls&gt;” (quoted, escaped angle brackets), matching the file's legacy main-prose convention (also at L52, 58, 63, 234, 238, 322, 519) but inconsistent with the same file's own slide deck, which uses a plain code span `` `<cls>` `` for the identical token (L625, 627, 682). Renders correctly either way, but the two conventions coexist within one file.
clarity:
  - ex 1: opening clause "How does the value of `img_size` affect training time?" gives no range or values to test before the sentence pivots to a concrete, well-specified sub-task (predict the effect of halving `patch_size` to 8, then measure). Minor — the file overall is not flagged as reading-prompt-style since 4 of 5 items have clear deliverables.
notable: Along with queries-keys-values.md, this is the group's other "legacy-style" file: repeated-1 numbering, only 5 exercises (vs. 6 everywhere else), and prose that predates the rest of the chapter's code-grounded rewrite (see the &lt;cls&gt; entity usage above). Exercises 2-5 are nonetheless concrete and well-specified (implement a variant, measure accuracy/drop, verify a claim).
```

---

## Group-Level Summary

**Scope:** 13 files, 77 exercises total (chapter_attention: 6 files / 36 exercises; chapter_transformers: 7 files / 41 exercises).

**Numbering:** No file mixes numbering styles internally. chapter_attention splits evenly 3-repeated-1 (attention-at-scale, attention-scoring, queries-keys-values) / 3-sequential (multihead-attention, positional-information, what-attention-computes) — no dominant style. chapter_transformers is dominantly sequential (6/7 files); the outlier is vision-transformer.md (repeated-1, 5 items).

**Names/tags:** 0/13 files use exercise names or bracketed type tags — the "Named + tagged" and "Named only" variants described in the rubric do not appear anywhere in this group. tag_vocab is empty.

**Discussions:** 0/13 files retain a Discussions block. Every file's exercises are followed immediately by a `<!-- slides -->` deck instead — the single most consistent structural pattern (and deviation from the rubric's assumed format) across the entire group.

**Citations/crossrefs:** 7 `:cite:`/`:citet:` tags across 6/13 files (8 distinct works; moe.md bundles 2 in one tag); both `:cite:` and `:citet:` forms appear, mixed even within scaling-laws.md's own exercise set. 23 `:numref:`/`:eqref:` uses total, concentrated in attention-at-scale.md (7) and scaling-laws.md (4); positional-information.md, kv-cache.md, and moe.md have none.

**Worst formatting defect:** vision-transformer.md L548 — HTML-entity “&lt;cls&gt;” where the same file's own slides use a code span, `` `<cls>` ``.

**Worst clarity offenders:** queries-keys-values.md ex1/ex3/ex4 — unmodified legacy d2l reading-prompts ("Design a differentiable search engine...", "Review... and interpret...") with no stated deliverable or success criterion. This is the only file in the 13 with no grounding in the book's own code/experiments.

**Cross-cutting pattern:** the two exercise sets not rewritten around this book's own code/experiments — queries-keys-values.md and vision-transformer.md — are exactly the two files using repeated-1 numbering and are the group's only sources of both formatting and clarity flags. All 11 fully-rewritten files are essentially defect-free and clarity-clean.
