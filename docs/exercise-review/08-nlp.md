# Exercise Review — Group 08: NLP (pretraining + applications)

Chapters reviewed:
- `chapter_natural-language-processing-pretraining` (11 files)
- `chapter_natural-language-processing-applications` (7 files)

Source of truth: `.md` files only. Enumerated via
`grep -rln "^## Exercises" <chapter_dir> --include="*.md"`.

---

## chapter_natural-language-processing-pretraining

### file: chapter_natural-language-processing-pretraining/similarity-analogy.md
```
heading_line: 343
n_exercises: 2
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: none
discussions: tabbed(4 tabs: mxnet/pytorch/jax/tensorflow)
defects: none found
clarity:
  - ex 1: "Test the fastText results using `TokenEmbedding('wiki.en')`." — no metric or comparison specified for what "testing" should produce; borderline underspecified.
  - ex 2: "When the vocabulary is extremely large, how can we find similar words or complete a word analogy faster?" — pure "how can we" reading-prompt, no deliverable (no algorithm to name, no complexity target).
notable: Followed by a full revealjs slide deck (`<!-- slides -->` + `::: {.slide}` blocks) — this trailing structure is present in most files of this group and is not itself a defect.
```

### file: chapter_natural-language-processing-pretraining/word-embedding-dataset.md
```
heading_line: 597
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
discussions: tabbed(4 tabs: mxnet/pytorch/jax/tensorflow)
defects:
  - L599: "How does the running time of code in this section changes if not using subsampling?" — subject-verb agreement error ("does ... changes" should be "does ... change").
clarity:
  - ex 2: "Set `k` to other values and see how it affects the data loading speed." — classic "experiment and see" pattern: no candidate values for k, no speed metric, no comparison basis.
  - ex 3: "What other hyperparameters in the code of this section may affect the data loading speed?" — open-ended, no deliverable (list? measure? just "think about it").
notable: none
```

### file: chapter_natural-language-processing-pretraining/word2vec-pretraining.md
```
heading_line: 612
n_exercises: 2
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: none
discussions: tabbed(4 tabs: mxnet/pytorch/jax/tensorflow)
defects: none found
clarity:
  - ex 1: "Can you improve the results by tuning hyperparameters?" — "Can you...?" filler question (style-guide 17.5/generic filler); no hyperparameters or metric named.
notable: none
```

### file: chapter_natural-language-processing-pretraining/seq2seq.md
```
heading_line: 1134
n_exercises: 7
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 1 (:citet:`Sutskever.Vinyals.Le.2014`)
crossrefs: 3 (:numref:`sec_beam-search`, :eqref:`eq_beam-search-score`, :numref:`sec_lstm`)
subproblems: none
discussions: tabbed(4 tabs: mxnet=1062, pytorch=1062, tensorflow=3865, jax=18022)
defects: none found — wrapped continuation lines (e.g. L1136-1138, L1139-1142) use 3-space indent aligned under "1. ", which is correct for plain paragraph continuation (not a sub-list), so this does not break rendering.
clarity: none flagged — every exercise names a concrete manipulation (rebuild with two tokenizers; sweep beam width over an explicit set; sweep length-penalty alpha over an explicit set; swap loss/replace GRU with LSTM/change context-feeding), asks a specific comparative or causal question, and several point to a specific :numref:/:citet: for grounding.
notable: By far the best-specified exercise set in the entire NLP group — quantitative sweeps with explicit value sets ({1,2,4,8,16}, {0,0.75,1.5}), precise citations/crossrefs, no filler tone — despite still using bare repeated-1 numbering with no names or tags. Also notable: mxnet and pytorch share the same Discourse thread ID (1062) while tensorflow (3865) and jax (18022) each have distinct ones — the reverse of the usual pattern elsewhere in the corpus (where pytorch/jax/tensorflow usually share one thread and mxnet is the outlier). Not verified against the live forum; flagged as a possible cross-reference inconsistency.
```

### file: chapter_natural-language-processing-pretraining/subword-embedding.md
```
heading_line: 230
n_exercises: 4
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 1 (:cite:`Bojanowski.Grave.Joulin.ea.2017`)
crossrefs: 0
subproblems: none
discussions: other — 3 tab-blocks covering 4 frameworks (mxnet; pytorch; combined `tensorflow,jax` sharing one tab and one Discourse ID 4587), unlike the standard 4-separate-tabs pattern used elsewhere.
defects:
  - L236-238: three consecutive blank lines between the exercise list and the Discussions tabs (sibling files use a single blank line here).
clarity:
  - ex 2: "How to design a subword embedding model based on the continuous bag-of-words model?" — open design prompt, no stated deliverable or success criterion.
  - ex 4: "How to extend the idea of byte pair encoding to extract phrases?" — same pattern, no deliverable.
notable: Recurring grammatically-fragmentary "How to X?" question stem (should be "How can we X?" / "How do we X?") — a systemic tic across the legacy bare-list files in this group (see also approx-training.md, glove.md, word2vec.md, finetuning-bert.md).
```

### file: chapter_natural-language-processing-pretraining/bert-dataset.md
```
heading_line: 631
n_exercises: 2
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: none
discussions: tabbed(4 tabs: mxnet/pytorch/jax/tensorflow)
defects: none found (all inline code spans are properly closed)
clarity: none flagged — both exercises are concrete (install/apply an alternative sentence splitter and observe the effect; recompute vocabulary size with no frequency cutoff).
notable: ex 1 is an 11-sentence, single unwrapped source line (L633) that hands the reader the near-complete solution (exact `pip install`, `import`, `nltk.download`, and the precise `nltk.tokenize.sent_tokenize` call plus its expected output) — the exercise is reduced to copy-pasting supplied code rather than designing anything, and the source line is far longer than the ~80-char wrapping used in, e.g., seq2seq.md.
```

### file: chapter_natural-language-processing-pretraining/bert.md
```
heading_line: 926
n_exercises: 2
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 1 (:cite:`Hendrycks.Gimpel.2016`)
crossrefs: 0
subproblems: none
discussions: tabbed(4 tabs: mxnet/pytorch/jax/tensorflow)
defects: none found
clarity:
  - ex 2: "Research into the difference between GELU and ReLU." — a reading prompt with no stated deliverable (what should the "research" produce: a proof, a list of properties, an experiment?).
notable: none
```

### file: chapter_natural-language-processing-pretraining/approx-training.md
```
heading_line: 234
n_exercises: 3
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 1 (:eqref:`eq_hi-softmax-sum-one`)
subproblems: none
discussions: single-link (`[Discussions](url)`, no :begin_tab: framework wrapper at all — the only pattern in the file)
defects:
  - L240: Discussions rendered as one flat link with no per-framework tabs, unlike every sibling file in the chapter (which all use tabbed mxnet/pytorch/jax/tensorflow blocks) — a discussions-block inconsistency.
clarity: none flagged beyond the "How to train..." fragment noted below — ex 2 is a clean eqref-anchored derivation ("Verify that :eqref:`eq_hi-softmax-sum-one` holds").
notable: This file has no trailing slide deck at all (ends at L240, immediately after the Discussions link) — the only files in the group with no slides are this one, glove.md, word2vec.md, and (in applications) finetuning-bert.md. Also carries the "How to train..." fragmentary question stem (ex 3).
```

### file: chapter_natural-language-processing-pretraining/glove.md
```
heading_line: 284
n_exercises: 2
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 1 (:cite:`Pennington.Socher.Manning.2014`)
crossrefs: 0
subproblems: none
discussions: single-link (no framework tabs), same pattern as approx-training.md
defects:
  - L290: flat single-link Discussions, inconsistent with the tabbed-discussions convention used by the majority of the chapter.
clarity: none flagged — both exercises are concrete, hint-anchored derivations.
notable: No trailing slide deck (file ends at L290 right after the Discussions link) — same "unconverted" pattern as approx-training.md, word2vec.md.
```

### file: chapter_natural-language-processing-pretraining/bert-pretraining.md
```
heading_line: 591
n_exercises: 2
numbering: sequential (literal "1." then "2." — the only file in this chapter using sequential rather than repeated-1 numbering)
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: none
discussions: tabbed(4 tabs: mxnet/pytorch/jax/tensorflow)
defects: none found
clarity: none flagged — ex 1 is a clear conceptual "why" question, ex 2 gives concrete config values (max length 512, BERT-Large-scale config) and asks a well-posed observe-and-explain question.
notable: Sole numbering outlier in the chapter (sequential "1./2." vs. repeated "1./1." everywhere else) — itself a small illustration of the group's known style inconsistency, though not a rendering defect.
```

### file: chapter_natural-language-processing-pretraining/word2vec.md
```
heading_line: 262
n_exercises: 3
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 1 (:cite:`Mikolov.Sutskever.Chen.ea.2013`)
crossrefs: 0
subproblems: none
discussions: single-link (no framework tabs)
defects:
  - L268: flat single-link Discussions, same inconsistency as approx-training.md/glove.md.
clarity: none flagged — all three are standard derive/explain questions with clear referents.
notable: No trailing slide deck (file ends at L268). Together with approx-training.md and glove.md, forms a trio of short, math-focused files (hierarchical softmax / negative sampling / GloVe bias terms) that all lack both tabbed Discussions and the slide-deck appendix — these three read as not yet migrated to the newer per-file pipeline used by the rest of the chapter.
```

---

## chapter_natural-language-processing-applications

### file: chapter_natural-language-processing-applications/sentiment-analysis-and-dataset.md
```
heading_line: 301
n_exercises: 2
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: none
discussions: tabbed(4 tabs: mxnet/pytorch/jax/tensorflow)
defects:
  - L302-303: double blank line after the "## Exercises" heading before the first exercise (sibling files use a single blank line).
  - L306-307: double blank line again between the exercise list and the Discussions tabs.
clarity:
  - ex 1: "What hyperparameters in this section can we modify to accelerate training sentiment analysis models?" — no target metric for "accelerate," no candidate hyperparameters named.
  - ex 2: "Can you implement a function to load the dataset of [Amazon reviews]... into data iterators and labels for sentiment analysis?" — "Can you...?" filler-question framing (style-guide 17.5).
notable: none
```

### file: chapter_natural-language-processing-applications/sentiment-analysis-rnn.md
```
heading_line: 484
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
discussions: tabbed(4 tabs: mxnet/pytorch/jax/tensorflow)
defects: none found (inline code in ex 3 is fully and correctly backtick-delimited)
clarity:
  - ex 1: "Can you improve the training and testing accuracies? How about tuning other hyperparameters?" — double filler-question pattern ("Can you...?" / "How about...?"), no hyperparameters or metric named.
  - ex 3: "Can we improve the classification accuracy by using the spaCy tokenization?" — filler-question opener, though the exercise becomes concrete because it supplies exact install/import/tokenizer-function code afterward.
notable: ex 3 gives away nearly the full implementation (install command, import, model load, exact replacement function) — same "handed-solution" pattern as bert-dataset.md ex 1.
```

### file: chapter_natural-language-processing-applications/sentiment-analysis-cnn.md
```
heading_line: 633
n_exercises: 3
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 2 (:numref:`sec_sentiment_rnn` x2, in ex 1 and ex 2)
subproblems: none
discussions: tabbed(4 tabs: mxnet/pytorch/jax/tensorflow)
defects: none found
clarity:
  - ex 2: "Can you further improve the classification accuracy of the model by using the methods introduced in the exercises of :numref:`sec_sentiment_rnn`?" — filler-question opener ("Can you...?"), though the cross-reference to a specific sibling section's exercises is legitimate and resolvable.
notable: none
```

### file: chapter_natural-language-processing-applications/natural-language-inference-bert.md
```
heading_line: 884
n_exercises: 2
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: none
discussions: tabbed(4 tabs: mxnet/pytorch/jax/tensorflow)
defects: none found
clarity: none flagged — ex 1 gives exact parameter values to change (`bert.small`→`bert.base`, `num_hiddens=256→768`, `ffn_num_hiddens=512→3072`, `num_heads=4→12`, `num_blks=2→12`) and an explicit numeric success threshold ("testing accuracy higher than 0.86"); ex 2 is a standard, well-posed compare/derive task.
notable: ex 1 is one of the best-specified exercises in the whole group — concrete parameter changes plus an explicit numeric success criterion. Worth using as the group's positive contrast case against the many vague "Can you...?" / "how can we..." items elsewhere.
```

### file: chapter_natural-language-processing-applications/finetuning-bert.md
```
heading_line: 188
n_exercises: 3
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 1 (:numref:`subsec_negative-sampling`)
subproblems: none
discussions: single-link (no framework tabs) — same pattern as approx-training.md/glove.md/word2vec.md
defects:
  - L194: flat single-link Discussions, inconsistent with the tabbed convention used by the other 6 files in this chapter.
clarity:
  - ex 2: "How can we leverage BERT in training language models?" — open reading-prompt, no deliverable.
  - ex 3: "Can we leverage BERT in machine translation?" — bare yes/no question with no deliverable at all; the most purely rhetorical exercise in the group.
notable: No trailing slide deck (file ends at L194 right after the Discussions link) — the only "unconverted" file in the applications chapter, mirroring the approx-training/glove/word2vec trio in the pretraining chapter. ex 1, by contrast, is a well-formed open design scenario (concrete setup: query, labeled pool, ranked output) with a grounded :numref: pointer.
```

### file: chapter_natural-language-processing-applications/natural-language-inference-and-dataset.md
```
heading_line: 430
n_exercises: 2
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: none
discussions: tabbed(4 tabs: mxnet/pytorch/jax/tensorflow)
defects: none found
clarity:
  - ex 1: "Can you design a measure for evaluating machine translation results by using natural language inference?" — "Can you...?" filler opener, though the underlying ask ("design a measure using NLI") is a legitimate, adequately scoped open research question.
notable: none
```

### file: chapter_natural-language-processing-applications/natural-language-inference-attention.md
```
heading_line: 790
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
discussions: tabbed(4 tabs: mxnet/pytorch/jax/tensorflow)
defects: none found
clarity:
  - ex 1: "Train the model with other combinations of hyperparameters. Can you get better accuracy on the test set?" — filler "Can you...?" plus no specific hyperparameter values suggested (contrast with seq2seq.md's explicit sweep sets).
  - ex 3: second half, "Can you design a model with attention mechanisms?" — filler-question tail on an otherwise reasonably well-scoped open design exercise (defines the 0-1 continuous similarity target and asks about data collection first).
notable: none
```

---

## Cross-file tallies (for aggregation)

- Files reviewed: 18 (11 pretraining + 7 applications)
- Total exercises: 50 (32 pretraining + 18 applications)
- Files with any named exercises: 0 / 18
- Files with any tagged exercises: 0 / 18
- Tag vocabulary observed in this group: empty (no `[tag]` bracketed tags anywhere)
- Numbering: repeated-1 in 17/18 files; sequential in 1/18 (bert-pretraining.md)
- Discussions: tabbed (4-way) in 13/18 files; single-link (no tabs) in 4/18 (approx-training.md, glove.md, word2vec.md, finetuning-bert.md); combined-tab variant in 1/18 (subword-embedding.md)
- Trailing slide deck present in 14/18 files; absent in the same 4 single-link files above
- Subproblems (nested/inline lettered): none in any of the 18 files
- Citations (:cite:/:citet:) inside exercises: 5 total, across 5 files (seq2seq.md, subword-embedding.md, bert.md, glove.md, word2vec.md)
- Crossrefs (:numref:/:eqref:) inside exercises: 7 total, across 4 files (seq2seq.md ×3, approx-training.md ×1, sentiment-analysis-cnn.md ×2, finetuning-bert.md ×1)
