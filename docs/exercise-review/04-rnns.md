# Exercise Review: chapter_recurrent-neural-networks + chapter_recurrent-modern

Group files (14 total, via `grep -rln "^## Exercises"`):
- chapter_recurrent-neural-networks: sequence.md, rnn.md, bptt.md, language-model.md, text-sequence.md, decoding.md, rnn-implementation.md
- chapter_recurrent-modern: lstm.md, hybrids.md, ssm.md, mamba.md, matrix-state.md, deltanet.md, test-time-regression.md

All 14 files use `## Exercises` followed by a Discussions block, followed by `<!-- slides -->` and a `::: {.slide ...}` deck that runs to end of file (not just discussion links as the rubric's "typical" pattern suggests — noted per file under `notable`).

---

## chapter_recurrent-neural-networks/sequence.md

```
file: chapter_recurrent-neural-networks/sequence.md
heading_line: 486
n_exercises: 6
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: nested-list(ex 1: four sub-items, correctly 4-space indented at L489-492)
discussions: tabbed(4 tabs: mxnet L499, pytorch L503, tensorflow L507, jax L511)
defects:
  - none found (clean nesting, balanced $, no stray markup)
clarity:
  - ex 1.4 (L492): "Change the network architecture... and retrain (possibly for more epochs). What do you observe?" — no metric/comparison specified; borderline underspecified but nested under a broader "improve the model" prompt.
  - ex 1.3 (L491): "Can you incorporate older observations while keeping the number of features fixed?" — "Can you...?" filler phrasing (rubric tone rule), though the sentence does carry real technical content (accuracy comparison follow-up).
notable: Exercises section (486-513) is short relative to the slide deck that follows (514-591), which is typical for this chapter group, not unique to this file.
```

## chapter_recurrent-neural-networks/rnn.md

```
file: chapter_recurrent-neural-networks/rnn.md
heading_line: 212
n_exercises: 5
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: none
discussions: tabbed(4 tabs: mxnet L221, pytorch L225, tensorflow L229, jax L233)
defects:
  - none found
clarity:
  - none flagged; all 5 are concrete (dimension question, conceptual "why," gradient behavior, an explicit parameter-count derivation with numbers plugged in, and an open reflection tied directly to the section's stated limitations).
notable: Shortest, cleanest file in the group — pure conceptual/derivation exercises, no code, no data.
```

## chapter_recurrent-neural-networks/bptt.md

```
file: chapter_recurrent-neural-networks/bptt.md
heading_line: 300
n_exercises: 4
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 1 (:numref:`sec_rnn-scratch` at L308)
subproblems: nested-list(ex 1: three sub-items at L303-305)
discussions: single-link (L310: "[Discussions](https://d2l.discourse.group/t/334)", not tabbed by framework)
defects:
  - L303-305: exercise 1's three sub-items are indented with only 3 spaces ("   1. Show that..."), not the 4 spaces this pipeline requires for list nesting — a nesting-breaking indentation defect (contrast with sequence.md L489-492, which uses 4 spaces correctly for the same pattern).
  - L310: Discussions block is a single bare link, not the `:begin_tab:`-wrapped per-framework set used by every other file in chapter_recurrent-neural-networks (sequence.md, rnn.md, language-model.md, text-sequence.md, rnn-implementation.md all have 4 tabs) — missing frameworks relative to the chapter's own convention.
clarity:
  - none flagged; all 4 exercises (eigen-analysis proof, nonlinear-Jacobian rerun, open "other methods" question, and an explicit gradient-norm-vs-lag measurement with a defined threshold) specify a concrete deliverable.
notable: none beyond the two defects above.
```

## chapter_recurrent-neural-networks/language-model.md

```
file: chapter_recurrent-neural-networks/language-model.md
heading_line: 589
n_exercises: 4
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 2 (:numref:`subsec_perplexity` L607, :numref:`subsec_markov-models-and-n-grams` L615)
subproblems: none
discussions: tabbed(4 tabs: mxnet L617, pytorch L621, tensorflow L625, jax L629)
defects:
  - none found
clarity:
  - none flagged; all 4 exercises specify exact parameters (temperatures, vocab size, byte/perplexity formula to verify, smoothing-constant sweep) and a concrete artifact (a plot, a derived formula, a comparison).
notable: none.
```

## chapter_recurrent-neural-networks/text-sequence.md

```
file: chapter_recurrent-neural-networks/text-sequence.md
heading_line: 818
n_exercises: 5
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 1 (:numref:`sec_rnn-scratch` L828)
subproblems: none
discussions: tabbed(4 tabs: mxnet L848, pytorch L852, tensorflow L856, jax L860)
defects:
  - none found
clarity:
  - none flagged; each exercise names a specific deliverable (compression ratio comparison, per-vocab-size perplexity comparison with a reasoning prompt about why the metric is wrong, an ablation with named longest tokens, an explanation of glitch tokens, and a digit-tokenization exercise with a concrete proposal required).
notable: none.
```

## chapter_recurrent-neural-networks/decoding.md

```
file: chapter_recurrent-neural-networks/decoding.md
heading_line: 673
n_exercises: 7
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 1 (:citet:`Nguyen.Baker.Neo.ea.2025` L694)
crossrefs: 2 (:numref:`fig_beam-search` L680, :eqref:`eq_beam-search-score` L690)
subproblems: none
discussions: single-link (L706: "[Discussions](https://d2l.discourse.group/t/338)", not tabbed by framework)
defects:
  - L706: same single-link-instead-of-tabbed pattern as bptt.md — missing the mxnet/pytorch/tensorflow/jax tab set used elsewhere in this chapter.
clarity:
  - none flagged; all exercises specify concrete parameters/thresholds (θ=1.2, α∈{0,0.75,1.5} at k=4, T=1.5, k∈{1,2,4,8,16}) and an explicit question to answer or artifact to produce.
notable: ex 6 depends on "the sequence-to-sequence translation model of the next chapter" — a genuine forward reference (explicitly flagged as such in the text itself, so not a hidden defect, just worth noting as a cross-chapter dependency).
```

## chapter_recurrent-neural-networks/rnn-implementation.md

```
file: chapter_recurrent-neural-networks/rnn-implementation.md
heading_line: 1159
n_exercises: 9
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: none
discussions: tabbed(4 tabs: mxnet L1190, pytorch L1194, tensorflow L1198, jax L1202)
defects:
  - none found
clarity:
  - none flagged; largest bare-list file in the chapter but every item names a specific hyperparameter change, dataset, or metric (validation perplexity, bits per byte, parameter count, cross-book perplexity gap).
notable: Longest exercise list in chapter_recurrent-neural-networks (9), all uniformly bare/untagged/unnamed — the chapter's "classic d2l" bare-list style is the most consistent here.
```

## chapter_recurrent-modern/lstm.md

```
file: chapter_recurrent-modern/lstm.md
heading_line: 858
n_exercises: 10
numbering: repeated-1
names: none
name_style: n/a
tags: all (10/10)
tag_vocab: [extended, short-code, conceptual]
difficulty_markers: bracketed tags above double as difficulty/type markers (no separate marker)
citations: 0
crossrefs: 4 (:eqref:`lstm_update` L868 & L884, :eqref:`lstm_c_tilde` L883, :eqref:`gru_H` L888)
subproblems: none
discussions: tabbed(2 tabs only: pytorch L901, jax L905 — no mxnet/tensorflow tabs)
defects:
  - none found (2-tab Discussions is plausibly intentional for this newer chapter's framework coverage, not a broken link, so not counted as a defect — contrast with the six other chapter_recurrent-modern files below, which have a literally empty/placeholder link)
clarity:
  - none flagged; every item specifies an exact change (ablate forget gate, bias init to 1, parameter-matched GRU width, layer sweep 1-4) and a comparison metric (perplexity, wall-clock).
notable: Tag-only style (bracket tag, no italic/bold name) — the format anchor for this chapter's "tagged, unnamed" sub-variant, later reused by matrix-state.md and deltanet.md.
```

## chapter_recurrent-modern/hybrids.md

```
file: chapter_recurrent-modern/hybrids.md
heading_line: 1051
n_exercises: 5
numbering: repeated-1
names: all (5/5)
name_style: italic-period (name precedes tag: "*Name.* [tag]")
tags: all (5/5)
tag_vocab: [extended, conceptual, short-code]
difficulty_markers: none beyond tags
citations: 3 (:cite:`Ren.Liu.Lu.ea.2024` L1071 and L1100 [same ref, twice], :cite:`Kimi.Team.2025b` L1092)
crossrefs: 3 (:eqref:`eq_kv-cache-bytes` L1073, :eqref:`eq_ms-state-bytes` L1080, :numref:`subsec_ms-capacity` L1105)
subproblems: none (ex 1 bundles ~5 sequential sub-steps into one un-lettered paragraph, see notable)
discussions: other (L1111: "[Discussions](https://d2l.discourse.group/)" — a bare root URL with no thread id, i.e. a broken/placeholder link)
defects:
  - L1111: Discussions link points to the discourse group root, not a specific thread — effectively a dead/placeholder link.
clarity:
  - ex 1 (L1053-1071): six chained instructions ("Rerun... probe length generalization... Remove the confound... verify this first... extend `make_recall`... evaluate...") packed into a single un-lettered list item. Each step is individually well-specified, but the sheer chaining makes it hard to tell where the exercise's actual deliverable ends — a strong candidate for splitting into lettered sub-parts.
notable: Introduces the "name-before-tag" ordering (*Name.* [tag]), the opposite order from ssm.md/mamba.md ([tag] *Name.*) — same building blocks, inconsistent order across files in the same chapter.
```

## chapter_recurrent-modern/ssm.md

```
file: chapter_recurrent-modern/ssm.md
heading_line: 1487
n_exercises: 7
numbering: repeated-1
names: all (7/7)
name_style: italic-period (tag precedes name: "[tag] *Name.*")
tags: all (7/7)
tag_vocab: [short-code, extended]
difficulty_markers: none beyond tags
citations: 2 (:citet:`Feng.Tung.Ahmed.ea.2024` L1520 and L1523, same ref twice)
crossrefs: 8 (:numref:`subsec_bptt-gradient-pathologies` L1496, :eqref:`eq_ssm_kernel` L1507, :eqref:`eq_hippo` L1510, :eqref:`eq_ssm_cont` L1510, :eqref:`eq_affine_recurrence` L1518, :numref:`sec_lstm` L1513, :numref:`subsec_ssm-step` L1527, :numref:`sec_bptt` L1543)
subproblems: none
discussions: other (L1545: "[Discussions](https://d2l.discourse.group/)" — same broken placeholder link as hybrids.md)
defects:
  - L1545: broken/placeholder Discussions link (root URL, no thread id).
clarity:
  - none flagged beyond the link; all 7 exercises specify exact sweeps (Δ ranges, discretization values, tolerance-based verification) and explicit questions.
notable: Densest crossref count in the group (8) — appropriately so, since several exercises explicitly build on named equations from this and prior sections.
```

## chapter_recurrent-modern/mamba.md

```
file: chapter_recurrent-modern/mamba.md
heading_line: 1161
n_exercises: 5
numbering: repeated-1
names: all (5/5)
name_style: italic-period ("[tag] *Name.*", same order as ssm.md)
tags: all (5/5)
tag_vocab: [extended, short-code]
difficulty_markers: none beyond tags
citations: 0
crossrefs: 2 (:numref:`subsec_zoh` L1169, :numref:`sec_kv-cache` L1199)
subproblems: none
discussions: other (L1201: "[Discussions](https://d2l.discourse.group/)" — same broken placeholder link)
defects:
  - L1201: broken/placeholder Discussions link.
clarity:
  - none flagged; each exercise specifies a concrete ablation and comparison metric (accuracy vs. symbols, bits-per-byte vs. perplexity, logit match verification).
notable: none beyond the shared Discussions defect.
```

## chapter_recurrent-modern/matrix-state.md

```
file: chapter_recurrent-modern/matrix-state.md
heading_line: 979
n_exercises: 6
numbering: repeated-1
names: none
name_style: n/a
tags: all (6/6)
tag_vocab: [conceptual, short-code]
difficulty_markers: none beyond tags
citations: 0
crossrefs: 4 (:eqref:`eq_ms-semiseparable` L981, :numref:`chap_attention` L997, :numref:`sec_attention-at-scale` L1000, :numref:`sec_ssm` L1016)
subproblems: none
discussions: other (L1022: "[Discussions](https://d2l.discourse.group/)" — same broken placeholder link)
defects:
  - L1022: broken/placeholder Discussions link.
clarity:
  - none flagged; each exercise specifies exact sweep values and the quantity to report.
notable: L1016 references "`:numref:`sec_ssm`'s exercises" — i.e., an exercise here explicitly points to another section's exercises (the effective-memory-horizon definition from ssm.md ex 1) as shared context. Unusual but well-specified, not flagged as a clarity defect since the referent is real and locatable.
```

## chapter_recurrent-modern/deltanet.md

```
file: chapter_recurrent-modern/deltanet.md
heading_line: 1309
n_exercises: 6
numbering: repeated-1
names: none
name_style: n/a
tags: all (6/6)
tag_vocab: [short-code, extended]
difficulty_markers: none beyond tags
citations: 0
crossrefs: 7 (:numref:`subsec_dn-trained` L1312, :numref:`subsec_dn-overwrite` L1319, :eqref:`eq_ms-retrieval-error` L1323, :numref:`subsec_ms-capacity` L1327, :eqref:`eq_dn-gated` L1339, :numref:`subsec_ms-chunked` L1342, :numref:`subsec_dn-reflection` L1349)
subproblems: none
discussions: other (L1355: "[Discussions](https://d2l.discourse.group/)" — same broken placeholder link)
defects:
  - L1311-1344: tag is wrapped in bold markup, "**[short-code]**" / "**[extended]**", unlike the plain "[tag]" used in lstm.md and matrix-state.md for the same tag-only style — an unnecessary/inconsistent extra markup layer (renders as bold-bracket text, not a rendering error per se, but a formatting inconsistency within the same tag vocabulary).
  - L1355: broken/placeholder Discussions link.
clarity:
  - none flagged; all exercises specify exact verification targets (float tolerance, FLOP counts, word-length-64 tracking test).
notable: none beyond the two defects.
```

## chapter_recurrent-modern/test-time-regression.md

```
file: chapter_recurrent-modern/test-time-regression.md
heading_line: 1130
n_exercises: 6
numbering: repeated-1
names: all (6/6)
name_style: bold-period ("**[tag]** **Name.**" — the only file in the group using bold instead of italic for the name)
tags: all (6/6)
tag_vocab: [short-code, conceptual]
difficulty_markers: none beyond tags
citations: 0
crossrefs: 9 (:eqref:`eq_ttr-least-squares` L1134, :numref:`subsec_ttr-spectrum` L1139, :numref:`tab_ttr-recipe` L1142, :eqref:`eq_ttr-longhorn-gate` L1143, :eqref:`eq_ttr-longhorn-objective` L1145, :numref:`subsec_ttr-bandwidth` L1149, :eqref:`eq_ttr-parametric-nw` L1157, :numref:`subsec_ttr-tracking` L1161, :eqref:`eq_ttr-titans` L1163)
subproblems: none
discussions: other (L1178: "[Discussions](https://d2l.discourse.group/)" — same broken placeholder link)
defects:
  - L1132-1167: both the tag AND the name are bold ("**[short-code]** **Recursive least squares joins the table.**"), a third distinct name/tag markup combination in this chapter (vs. hybrids.md's plain-tag+italic-name and ssm.md/mamba.md's plain-tag+italic-name in the other order, and lstm.md/matrix-state.md/deltanet.md's no-name variants) — no two files in chapter_recurrent-modern use the same tag+name markup convention.
  - L1178: broken/placeholder Discussions link.
clarity:
  - none flagged; every exercise specifies an exact derivation target or sweep with a numeric comparison.
notable: Highest crossref density in the group (9 in 6 exercises) — every exercise is tightly anchored to a specific numbered equation or subsection, which is good practice but makes this file the most dependent on surrounding prose to parse cold.
```

---

## Group-level summary

**Totals**: 14 files, **85 exercises** (chapter_recurrent-neural-networks: 40 across 7 files; chapter_recurrent-modern: 45 across 7 files). All 85 use `repeated-1` markdown auto-numbering; none mix literal sequential numbers.

**Dominant style by chapter**: chapter_recurrent-neural-networks (all 7 files) is uniformly **bare list** — no names, no tags, no difficulty markers anywhere. chapter_recurrent-modern (all 7 files) is uniformly **tagged**, but the *how* is inconsistent: lstm.md/matrix-state.md (tag only, plain brackets), deltanet.md (tag only, **bold** brackets), hybrids.md (italic name + tag, name-first), ssm.md/mamba.md (tag + italic name, tag-first), test-time-regression.md (bold tag + bold name). No two files share an identical tag/name markup convention, though only 3 tag strings ever appear.

**Names/tags counts**: 0/7 recurrent-neural-networks files have tags or names. 7/7 recurrent-modern files have tags (45/45 exercises); 4/7 of those also add names (hybrids, ssm, mamba, test-time-regression = 23 exercises), 3/7 are tag-only (lstm, matrix-state, deltanet = 22 exercises).

**Tag vocabulary** (chapter_recurrent-modern only): `[conceptual]`, `[short-code]`, `[extended]` — exactly these three, consistently, across all 7 tagged files.

**Worst formatting defects**: (1) **6 of 7 recurrent-modern files** (hybrids, ssm, mamba, matrix-state, deltanet, test-time-regression) end their exercises with a dead placeholder Discussions link, `[Discussions](https://d2l.discourse.group/)` — root URL, no thread id — vs. lstm.md's real (if only 2-tab) links. (2) bptt.md and decoding.md (recurrent-neural-networks) use a single un-tabbed Discussions link where every sibling file in the chapter uses 4 framework tabs. (3) bptt.md L303-305: exercise sub-items indented 3 spaces instead of the required 4, breaking list nesting.

**Worst clarity offenders**: hybrids.md ex 1 (L1053-1071) chains roughly six sequential instructions into one un-lettered list item — individually clear, collectively hard to parse as a single deliverable. No outright underspecified ("try things and see") exercises were found in either chapter; sequence.md ex 1.3 has a minor "Can you...?" filler-phrasing tic but otherwise carries real content.
