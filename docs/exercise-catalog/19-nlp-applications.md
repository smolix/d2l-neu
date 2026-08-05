# Chapter Overview — chapter_natural-language-processing-applications

Best external match: Stanford CS224N's Default Final Project ("minBERT and Downstream
Tasks," Spring 2024) is explicitly adapted from CMU 11-711's "Build Your Own BERT"
assignment (Zhou, Jiang, Dutt, Boldt, Veerubhotla, Neubig) — the same from-scratch-BERT
+ sentiment(SST/CFIMDB) + pair-task(paraphrase/STS) design serves as provenance for
three of our seven sections (sentiment-and-dataset, finetuning-bert, nli-bert).
The single best "new idea" import is Gururangan et al. 2018's hypothesis-only-baseline
finding (SNLI labels ~67%-predictable from the hypothesis alone) — a genuinely novel,
checkable, code-based exercise buildable entirely from this chapter's own SNLI loader,
reused as a callback in the BERT-fine-tuning section. Zhang & Wallace 2015 (CNN
sensitivity analysis) and Blitzer et al. 2007 (multi-domain sentiment) give concrete
ablation/generalization designs for the two sentiment-classifier sections. SLP3 is
nearly exercise-free here: current ch. 4 (Logistic Regression) ships an *empty*
Exercises section in the public draft, and ch. 22 (sentiment/affect lexicons) has
exactly one exercise — a PMI-derivation unrelated to neural pipelines — a genuine
coverage gap, not an oversight. No course assignment implementing the decomposable-
attention architecture itself was found (it appears in lecture slides, e.g. UIUC
CS546, but not as homework). finetuning-bert.md is the chapter's only section with
zero code; we added one short-code problem grounded in the already-introduced
BERTEncoder to give it a hands-on component. Existing sets are weak chapter-wide (bare
numbering, no tags/names, mostly "Can you...?" filler); nli-bert.md's ex1 already
matches external best practice (explicit hyperparameters + 0.86 accuracy threshold)
and is kept as-is. Totals: 7 sections, 18 current exercises (keep 6 / rewrite 9 / drop
3), 35 proposed problems.

---

## chapter_natural-language-processing-applications/sentiment-analysis-and-dataset.md — Sentiment Analysis and the Dataset

**Topic:** Downloading, tokenizing, and batching Stanford's IMDb movie-review corpus
into a reusable `(train_iter, test_iter, vocab)` pipeline for sentiment classification.
**Current exercises:** 2; disposition: keep 0, rewrite 2, drop 0 — ex1 ("what
hyperparameters... accelerate training") names no candidate hyperparameters or metric;
ex2 ("Can you implement...") is a legitimate second-dataset task buried in filler
phrasing — both are worth keeping in substance but need concrete deliverables.

**External sources found:**
- Jurafsky & Martin, *SLP3* (2025 draft), Ch. 4 "Logistic Regression" — the
  chapter's Exercises heading is present but contains **no exercises** at all before
  the References begin; a genuine gap, not a missed search. —
  https://web.stanford.edu/~jurafsky/slp3/4.pdf
- Jurafsky & Martin, *SLP3* (2025 draft), Ch. 22 "Lexicons for Sentiment, Affect, and
  Connotation," Ex. 22.1 — the chapter's *only* exercise: show that a word/category
  association score is a variant of pointwise mutual information without the log term
  — pure math, no code, not about data pipelines. —
  https://web.stanford.edu/~jurafsky/slp3/22.pdf
- Maas, Daly, Pham, Huang, Ng & Potts 2011, ACL, "Learning Word Vectors for Sentiment
  Analysis" — the paper that introduced this exact IMDb 25k/25k balanced corpus and
  its train/test protocol; reports 88.89% test accuracy for their best model on this
  same split, a useful external benchmark number. —
  https://aclanthology.org/P11-1015/
- Blitzer, Dredze & Pereira 2007, ACL, "Biographies, Bollywood, Boom-boxes and
  Blenders" — introduces the Multi-Domain Sentiment Dataset (books/DVDs/
  electronics/kitchen, 1000+1000 reviews/domain), built for exactly the cross-domain
  question this section's loader invites. —
  https://aclanthology.org/P07-1056/
- Stanford CS224N (Spring 2024), Default Final Project "minBERT and Downstream
  Tasks" — uses SST + CFIMDB for sentiment fine-tuning; treats data loading as given
  plumbing rather than an exercise target, consistent with our finding that
  loader-design exercises have no external tradition. —
  https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/project/default-final-project-handout-minbert-spr2024-updated.pdf

Data-loading/preprocessing pipelines have essentially no dedicated external exercise
tradition of their own — every source above either has no exercises here, or treats
loading as a prerequisite rather than the task itself.

**Proposed problem set** (5 problems, our reference format):
1. [conceptual] **Balanced-Split Verification.** Print the exact positive/negative
   counts in the IMDb train and test splits returned by `read_imdb`, and explain in
   2–3 sentences why the review-length histogram plotted in this section matters for
   choosing `num_steps`. Deliverable: printed counts + written justification. Success:
   counts match 12500/12500 and the `num_steps` choice is justified against the
   histogram's shape.
   *Provenance:* original.
1. [short-code] **Truncation Length Sweep.** Rerun preprocessing with `num_steps` in
   {100, 250, 500, 1000} and report, for each, the fraction of reviews truncated and
   the fraction padded by more than 50%. Deliverable: a 4-row table. Success: the
   table shows the expected monotonic trend and locates the section's default (500)
   on it.
   *Provenance:* adapted from the section's own ex1 (overlap high; rewritten from
   "what hyperparameters... accelerate training" into a concrete swept, measured
   table).
1. [short-code] **Amazon Reviews Loader.** Implement `load_data_amazon`, mirroring
   `load_data_imdb`'s contract (tokenize, `min_freq=5` vocab, pad/truncate to
   `num_steps`, wrap in `load_array`), for a sample of the Stanford SNAP Amazon
   reviews dataset, and report the resulting vocabulary size and example counts.
   Deliverable: working loader + printed stats. Success: the loader returns the same
   `(train_iter, test_iter, vocab)` shape contract used elsewhere in the chapter.
   *Provenance:* adapted from the section's own ex2 (overlap high; rewritten from a
   bare "Can you...?" into a concrete function-contract + deliverable).
1. [short-code] **Cross-Domain Vocabulary Overlap.** Build a vocabulary (same
   `min_freq=5` rule) from the books domain of Blitzer et al.'s Multi-Domain
   Sentiment Dataset, and report what fraction of IMDb's vocabulary tokens also
   appear in it. Deliverable: one overlap percentage + one comment on what it implies
   for reusing an IMDb-trained model on book reviews. Success: the comment is tied to
   the actual number reported.
   *Provenance:* adapted from Blitzer, Dredze & Pereira 2007 (overlap med — dataset
   and domain-shift framing borrowed, not their SCL algorithm).
1. [conceptual] **Why Word-Level Tokens.** IMDb reviews contain HTML line breaks and
   contractions (e.g., "didn't"). Explain in 3–4 sentences how `d2l.tokenize(...,
   token='word')` handling such artifacts could bias the vocabulary, and propose one
   concrete preprocessing fix. Deliverable: written explanation + one proposed fix.
   Success: the fix names a specific rule (e.g. a regex), not just "clean the data
   better."
   *Provenance:* original.

---

## chapter_natural-language-processing-applications/sentiment-analysis-rnn.md — Sentiment Analysis: Using Recurrent Neural Networks

**Topic:** Feeding frozen pretrained GloVe vectors into a bidirectional multilayer
LSTM to classify IMDb review sentiment.
**Current exercises:** 3; disposition: keep 1, rewrite 2, drop 0 — ex2 (300-dim GloVe
swap) is already concrete and worth keeping; ex1 (vague hyperparameter tuning) and ex3
(handed-solution spaCy swap, per prior style review) need tightened deliverables.

**External sources found:**
- Maas, Daly, Pham, Huang, Ng & Potts 2011, ACL — same IMDb 25k/25k split, reports
  88.89% test accuracy for their best (bag-of-words + learned-vector) model, verified
  directly from Table 2 of the paper — a solid external benchmark number for a
  "how close do you get" comparison exercise. —
  https://ai.stanford.edu/~ang/papers/acl11-WordVectorsSentimentAnalysis.pdf
- Stanford CS224N (Spring 2024), Default Final Project "minBERT and Downstream
  Tasks" — the direct successor comparison point (fine-tuned encoder vs. frozen
  pretrained vectors + BiLSTM) for the same sentiment task, though architecture
  comparison itself is not exercised there either. —
  https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/project/default-final-project-handout-minbert-spr2024-updated.pdf
- No dedicated course assignment comparing frozen-GloVe+BiLSTM hyperparameter
  sensitivity was found; this is a genuine gap — such comparisons appear in papers
  (see sentiment-analysis-cnn.md's sources), not graded homework.

**Proposed problem set** (5 problems, our reference format):
1. [short-code] **Epoch and Hidden-Size Sweep.** Train `BiRNN` for epochs in {2, 5,
   10} crossed with `num_hiddens` in {50, 100, 200} (9 runs), holding other
   hyperparameters fixed, and report test accuracy for each in a table. Deliverable:
   3×3 accuracy table. Success: the best cell is named and whether accuracy has
   plateaued by epoch 10 is stated.
   *Provenance:* adapted from the section's own ex1 (overlap high; rewritten from
   "can you improve...how about tuning other hyperparameters" into a concrete swept
   grid).
1. [short-code] **Embedding Dimensionality Comparison.** Retrain with 300-dimensional
   GloVe (`glove.6b.300d`) in place of 100-dimensional vectors, changing only
   `embed_size` and the loaded embedding, and report the test-accuracy delta versus
   the section's 100-dim baseline. Deliverable: one accuracy delta + one sentence on
   whether the added parameters were worth it. Success: the parameter-count tradeoff
   is named explicitly.
   *Provenance:* adapted from the section's own ex2 (overlap high; already
   well-scoped, tightened only by requiring the tradeoff be named).
1. [conceptual] **Tokenizer Mismatch.** Without running code, explain why GloVe
   joining some multi-word phrases with hyphens (e.g. "new-york") while a
   general-purpose tokenizer splits them ("new york") could silently reduce
   in-vocabulary hits after a tokenizer swap, and give one additional example phrase
   where this occurs. Deliverable: written explanation + one new example. Success:
   the example is a genuine compound/proper-noun phrase GloVe joins but a generic
   tokenizer splits.
   *Provenance:* adapted from the section's own ex3 (overlap high; the conceptual
   insight buried in the handed-solution exercise is asked directly instead of
   copy-pasted).
1. [short-code] **Benchmark Against the Original Paper.** Report your trained
   BiLSTM+GloVe test accuracy alongside Maas et al.'s verified 88.89% figure on the
   same 25k/25k split, and give one hypothesis for the gap (or lack of one) in 2–3
   sentences. Deliverable: two-number comparison + one hypothesis. Success: the
   hypothesis names a concrete modeling difference (e.g. no bag-of-words features,
   frozen vs. tunable vectors, no unlabeled-data pretraining).
   *Provenance:* adapted from Maas, Daly, Pham, Huang, Ng & Potts 2011 (overlap
   med — external accuracy figure used only as a benchmark).
1. [conceptual] **Freeze vs. Fine-Tune.** The section freezes the GloVe embedding
   (`requires_grad = False`). Explain in 2–3 sentences what changes in the
   optimization problem if it were fine-tuned end-to-end instead, and identify one
   concrete risk given IMDb's 25k training reviews relative to vocabulary size.
   Deliverable: written explanation naming the specific risk. Success: the risk is
   tied to an actual number (vocab size vs. example count), not a vague "might
   overfit."
   *Provenance:* original.

---

## chapter_natural-language-processing-applications/sentiment-analysis-cnn.md — Sentiment Analysis: Using Convolutional Neural Networks

**Topic:** textCNN (Kim, 2014) — parallel 1D convolutions over dual (frozen +
trainable) GloVe embeddings with max-over-time pooling, as an alternative
architecture for the same sentiment task.
**Current exercises:** 3; disposition: keep 1, rewrite 1, drop 1 — ex1 (RNN-vs-CNN
comparison) is well-scoped and cross-references correctly; ex3 (positional encoding)
is concrete but lacks a success criterion; ex2 just points back at the RNN section's
exercises with no new content, so it is dropped in favor of new material.

**External sources found:**
- Kim 2014, EMNLP, "Convolutional Neural Networks for Sentence Classification" —
  primary source for the textCNN model itself (already `:cite:`d in-text); its own
  ablations vary filter widths and channel counts, the template our filter-width
  exercise follows. — https://aclanthology.org/D14-1181/
- Zhang & Wallace 2015/2017, arXiv:1510.03820 (IJCNLP 2017), "A Sensitivity Analysis
  of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence
  Classification" — a widely cited ablation study of exactly the CNN hyperparameters
  (filter region size, number of filters, regularization) this section leaves
  untuned. — https://arxiv.org/abs/1510.03820
- Blitzer, Dredze & Pereira 2007 (see sentiment-analysis-and-dataset.md above) —
  reused here for a cross-architecture domain-generalization comparison. —
  https://aclanthology.org/P07-1056/
- No dedicated course assignment doing a 1D-conv-for-text ablation was found; the
  tradition here lives in papers (Kim 2014; Zhang & Wallace), not homework — a
  finding, not a failure.

**Proposed problem set** (5 problems, our reference format):
1. [conceptual] **RNN vs CNN Tradeoffs.** After training both the BiRNN (previous
   section) and textCNN (this section) with matched embedding size, report which
   architecture wins on test accuracy and which wins on wall-clock time per epoch in
   a 2×2 table, then explain in 2–3 sentences why max-over-time pooling makes textCNN
   more parallelizable than the BiLSTM's sequential recurrence. Deliverable: table +
   explanation. Success: the table has concrete numbers from both sections' actual
   runs.
   *Provenance:* adapted from the section's own ex1 (overlap high; kept largely
   as-is, made concrete by naming exactly which two metrics to report).
1. [short-code] **Filter-Width Ablation.** Retrain textCNN with `kernel_sizes` fixed
   to a single width (try 3, then 5, then 7 separately, keeping total channels equal
   to the section's default sum of 300) and report test accuracy for each against the
   default multi-width {3,4,5} run. Deliverable: 4-row accuracy table (3 single-width
   + the multi-width baseline). Success: the table lets the reader state whether
   multi-width pooling actually helps on IMDb or one width dominates.
   *Provenance:* adapted from Zhang & Wallace 2015 (overlap med — the ablation
   design is borrowed, not their code or dataset).
1. [short-code] **Positional Encoding Addition.** Add a fixed sinusoidal positional
   encoding (:numref:`sec_self-attention-and-positional-encoding`) to the concatenated
   embeddings before the convolutional layers, retrain, and report the test-accuracy
   delta versus the no-position baseline. Deliverable: one accuracy delta + one
   sentence connecting the (likely small) result to max-over-time pooling's
   position-invariance. Success: the sentence correctly ties the result to the
   pooling operation.
   *Provenance:* adapted from the section's own ex3 (overlap high; same ask, now
   pointed at a specific implementation with a required success criterion).
1. [conceptual] **Channel-Budget Reallocation.** The default model spends 100
   channels at each of 3 kernel widths (300 total). Without training anything,
   propose one alternative (width, channel-count) allocation of the same 300-channel
   budget, and argue — grounded in what different kernel widths detect
   (:numref:`fig_conv1d_textcnn`) — why it might do better or worse on movie reviews
   specifically. Deliverable: proposed allocation + one argument. Success: the
   argument names a concrete linguistic pattern (e.g. short evaluative bigrams like
   "not good" vs. longer phrases).
   *Provenance:* original.
1. [extended] **Cross-Domain CNN vs RNN.** Train both the BiRNN and textCNN models
   on IMDb, then evaluate both (no retraining) on the book-review test split of the
   Multi-Domain Sentiment Dataset (Blitzer et al. 2007), and report which
   architecture degrades less under this domain shift, with a one-paragraph
   hypothesis for why. Deliverable: 2×1 cross-domain accuracy comparison + hypothesis.
   Success: both models are evaluated under identical preprocessing, and the
   hypothesis names a specific architectural property (global n-gram detection vs.
   sequential state), not just "the CNN is different."
   *Provenance:* adapted from Blitzer, Dredze & Pereira 2007 (overlap med — dataset
   and domain-shift framing borrowed; the comparison design is ours).

---

## chapter_natural-language-processing-applications/natural-language-inference-and-dataset.md — Natural Language Inference and the Dataset

**Topic:** Defining the entailment/contradiction/neutral NLI task and loading the
570k-pair SNLI corpus into padded premise/hypothesis/label minibatches.
**Current exercises:** 2; disposition: keep 0, rewrite 2, drop 0 — ex1 (design an
MT metric using NLI) is a legitimate open question buried in "Can you...?" filler;
ex2 ("how can we change hyperparameters to reduce vocabulary size") names no specific
sweep — both are worth keeping in substance with concrete deliverables attached.

**External sources found:**
- Bowman, Angeli, Potts & Manning 2015, EMNLP, "A large annotated corpus for
  learning natural language inference" — the SNLI paper itself (already `:cite:`d);
  its image-caption-grounded annotation protocol for generating entailment/
  contradiction/neutral hypotheses is the direct provenance for a label-justification
  exercise. — https://aclanthology.org/D15-1075/
- Gururangan, Swayamdipta, Levy, Schwartz, Bowman & Smith 2018, NAACL, "Annotation
  Artifacts in Natural Language Inference Data" — shows a hypothesis-only classifier
  reaches ~67% accuracy on SNLI (vs. 33% chance), i.e. labels are partly predictable
  from the hypothesis alone — directly implementable with only this section's own
  `read_snli`/vocab/loader machinery. — https://aclanthology.org/N18-2017/
- NYU DS-GA 1011 ("NLP with Representation Learning," taught by SNLI's own author S.
  Bowman, with K. Cho) covers SNLI in lecture; specific homework text (in `hw1`/`hw2`
  of the course repo) was not accessible in a form we could verify, so no exercise is
  drawn from it. — https://github.com/nyu-mll/DS-GA-1011-Fall2017
- Stanford CS224N — no dedicated 3-way-NLI assignment was found (its default
  projects use paraphrase/STS instead of classic entailment/contradiction/neutral
  classification) — a partial coverage gap worth naming.

**Proposed problem set** (5 problems, our reference format):
1. [conceptual] **Reading the SNLI Label Scheme.** For each of the first 3
   premise/hypothesis/label triples printed by `read_snli(data_dir,
   is_train=True)`, write one sentence naming the specific word or phrase in the
   premise that justifies the printed label. Deliverable: 3 short justifications.
   Success: each justification names an actual token from the printed premise, not a
   restatement of the label.
   *Provenance:* original.
1. [short-code] **Hypothesis-Only Baseline.** Train a classifier that sees only the
   tokenized hypothesis (drop the premise entirely) — a bag-of-embeddings averaged
   and passed through one linear layer — and report its test accuracy against the
   33% random-guessing floor. Deliverable: one test-accuracy number + a one-sentence
   comparison to chance. Success: the number is reported alongside the 33% baseline
   for direct comparison.
   *Provenance:* adapted from Gururangan, Swayamdipta, Levy, Schwartz, Bowman & Smith
   2018 (overlap high — we reproduce their core finding using this section's own
   SNLIDataset machinery rather than their released features).
1. [short-code] **Vocabulary-Size Sweep.** Rebuild the SNLI vocabulary with
   `min_freq` in {1, 5, 20, 50} and report the resulting vocabulary size for each,
   then state which value the section's default (`min_freq=5`) sits closest to on
   the resulting curve. Deliverable: 4-row (cutoff, vocab size) table. Success: the
   table shows the expected monotonic decrease and the comparison cites actual
   numbers from it.
   *Provenance:* adapted from the section's own ex2 (overlap high; rewritten from an
   open "how can we change hyperparameters" into a concrete swept table over the one
   hyperparameter that controls vocabulary size).
1. [conceptual] **An NLI-Based MT Metric.** Sketch (4–6 sentences, no code) an
   algorithm that scores a candidate machine-translation output against a reference
   translation by treating one as premise and the other as hypothesis and using an
   NLI model's entailment probability as the score. Identify one concrete failure
   mode with a worked example pair. Deliverable: algorithm sketch + one worked
   failure-mode example. Success: the failure-mode example is concrete (specific
   reference/candidate sentences), not an abstract caveat.
   *Provenance:* adapted from the section's own ex1 (overlap med — framing is the
   book's own; grounded in the entailment/contradiction/neutral definitions of
   :citet:`Bowman.Angeli.Potts.ea.2015`, overlap low).
1. [short-code] **Label Balance by Premise Length.** Beyond the three-way count
   already printed for train/test, compute the label distribution separately for
   pairs where the premise has fewer than 10 tokens versus 10 or more, and report
   whether entailment/contradiction/neutral balance holds within both length
   buckets. Deliverable: two 3-count rows. Success: the reader states in one
   sentence, from the actual counts, whether balance is length-dependent.
   *Provenance:* original.

---

## chapter_natural-language-processing-applications/natural-language-inference-attention.md — Natural Language Inference: Using Attention

**Topic:** Parikh et al.'s (2016) decomposable attention model — attend, compare,
aggregate over premise/hypothesis token embeddings — for 3-way NLI without
recurrence or convolution.
**Current exercises:** 3; disposition: keep 1, rewrite 2, drop 0 — ex2 (name
drawbacks of the model) is a solid, well-posed conceptual question worth keeping;
ex1 (vague hyperparameter retraining) and ex3 (continuous-similarity design, buried
in "Can you...?" filler) need concrete deliverables.

**External sources found:**
- Parikh, Täckström, Das & Uszkoreit 2016, EMNLP, "A Decomposable Attention Model
  for Natural Language Inference" — the primary source for this entire section's
  model; its own ablation (with/without an "intra-sentence attention" extension)
  is directly reusable as an exercise design. — https://aclanthology.org/D16-1244/
- Bowman, Angeli, Potts & Manning 2015 (see above) — reused for grounding the
  drawback-analysis exercise in the dataset's own label taxonomy.
- UIUC CS 546 (Spring 2018), lecture slides presenting the decomposable attention
  model — confirms the paper is taught as lecture material, not assigned as
  homework; no exercise text is attached to it. —
  https://courses.grainger.illinois.edu/cs546/sp2018/Slides/Apr19_Parikh.pdf
- No graded course assignment implementing the decomposable-attention architecture
  itself was found in our search — a genuine finding: this architecture's external
  "exercise tradition" is the paper's own ablations, not homework sets.

**Proposed problem set** (5 problems, our reference format):
1. [short-code] **Hidden-Size and Dropout Sweep.** Retrain `DecomposableAttention`
   with `num_hiddens` in {100, 200, 400} crossed with MLP dropout in {0.1, 0.2, 0.4}
   (holding epochs at the section's default of 4), and report test accuracy for each
   of the 9 combinations, marking the best cell. Deliverable: 3×3 accuracy table.
   Success: the best cell is marked and compared explicitly to the section's own
   reported baseline run.
   *Provenance:* adapted from the section's own ex1 (overlap high; rewritten from
   "other combinations... can you get better accuracy" into a concrete swept grid).
1. [conceptual] **Two Drawbacks, Grounded.** Name two concrete drawbacks of the
   decomposable attention model for NLI, each illustrated with a specific
   premise/hypothesis pair you construct where the drawback would cause a wrong
   prediction (e.g. reliance on word order, or multi-hop reasoning the
   attend/compare/aggregate pipeline cannot represent). Deliverable: 2 drawbacks,
   each with a constructed example. Success: each example is a plausible SNLI-style
   pair and the failure is traceable to a specific pipeline step.
   *Provenance:* adapted from the section's own ex2 (overlap high; kept largely
   as-is, now requiring a constructed example per drawback) and from Parikh et al.
   2016's own discussion of the model's order-insensitivity (overlap low).
1. [short-code] **Ablate Intra-Sentence Attention.** Parikh et al. (2016) report an
   "intra-sentence attention" extension. Implement a simplified version — before the
   Attend step, replace each token's embedding with a weighted average of itself and
   the other tokens in its own sentence, weighted by dot-product similarity — and
   report the test-accuracy delta versus the section's base model. Deliverable: one
   accuracy delta + a one-sentence verdict on whether it helped. Success: the
   ablation isolates exactly one change (intra- vs. inter-sentence attention).
   *Provenance:* adapted from Parikh, Täckström, Das & Uszkoreit 2016 (overlap med
   — we reimplement a simplified version of one of their reported extensions, not
   their full model).
1. [conceptual] **Designing a Continuous-Similarity Dataset.** Sketch a
   data-collection protocol for a 0–1 continuous sentence-similarity task (instead
   of 3-way NLI): what instruction annotators receive, what scale they use, and one
   concrete inter-annotator-agreement measure. Then describe in 2–3 sentences the
   one architectural change needed to turn this section's Aggregate step from a
   3-way softmax into a bounded [0,1] regressor. Deliverable: protocol sketch +
   architecture-change description. Success: the architecture description names the
   specific layer/activation change (e.g. `Dense(3)` → `Dense(1)` + sigmoid).
   *Provenance:* adapted from the section's own ex3 (overlap high; split into its
   two genuinely distinct asks, each now with a concrete deliverable) and inspired
   by the STS task defined in :numref:`sec_finetuning-bert`
   (:cite:`Cer.Diab.Agirre.ea.2017`, overlap low).
1. [conceptual] **Complexity Accounting.** Using this section's $m+n$ (decomposable)
   vs. $mn$ (jointly-scored) complexity argument, compute the number of $f$-network
   evaluations for a premise/hypothesis pair with $m=20, n=15$ tokens under each
   scheme. Deliverable: two numbers and the ratio between them. Success: both
   numbers (35 vs. 300) are computed correctly and the ratio is stated.
   *Provenance:* original.

---

## chapter_natural-language-processing-applications/finetuning-bert.md — Fine-Tuning BERT for Sequence-Level and Token-Level Applications

**Topic:** A conceptual survey (no code) of how BERT's `<cls>`/per-token
representations are reused, via small added MLP heads, for single-text
classification, text-pair classification/regression, tagging, and question
answering.
**Current exercises:** 3; disposition: keep 1, rewrite 0, drop 2 — ex1 (search-engine
design with negative sampling) is already a well-formed, grounded design scenario per
the prior style review; ex2 ("how can we leverage BERT in training language models?")
and ex3 ("Can we leverage BERT in machine translation?") are bare reading-prompts
with no deliverable and are dropped in favor of concrete replacements.

**External sources found:**
- Warstadt, Singh & Bowman 2019, TACL, "Neural Network Acceptability Judgments" —
  the CoLA paper, already `:cite:`d in-text as the single-text-classification example.
- Rajpurkar, Zhang, Lopyrev & Liang 2016, EMNLP, "SQuAD: 100,000+ Questions for
  Machine Comprehension of Text" — already `:cite:`d in-text as the QA example.
- Devlin, Chang, Lee & Toutanova 2019, NAACL, "BERT: Pre-training of Deep
  Bidirectional Transformers for Language Understanding" — Section 4 of the original
  paper defines exactly this section's four task categories (sentence-level,
  sentence-pair, tagging via a variant, span-extraction QA); the direct origin of
  this section's own taxonomy, though not itself cited here. —
  https://arxiv.org/abs/1810.04805
- Hugging Face NLP Course, Chapter 7 ("Main NLP Tasks") — Token Classification
  (7.2) and Question Answering (7.7) walk through fine-tuning for exactly the two
  token-level tasks this section describes only in prose, with worked code — a
  natural "what this looks like in practice" companion. —
  https://huggingface.co/learn/nlp-course/chapter7/2 ,
  https://huggingface.co/learn/nlp-course/chapter7/7
- Stanford CS224N / CMU 11-711 minBERT default project (see sentiment-analysis
  entries above) — implements the sequence-level heads (sentiment, paraphrase, STS)
  this section only describes conceptually, one level more concrete.

**Proposed problem set** (5 problems, our reference format):
1. [conceptual] **Search-Engine Design with BERT.** Design a news-search ranking
   algorithm that, given a query and a pool of articles with one labeled relevant
   article per query, uses negative sampling (:numref:`subsec_negative-sampling`)
   for the training signal and BERT to score query-article relevance. Specify: the
   BERT input format (single sequence or pair), the negative-sampling ratio, and the
   training objective. Deliverable: a short numbered design spec covering those
   three points. Success: the input format is stated unambiguously as one of this
   section's own two patterns.
   *Provenance:* kept from the section's existing ex1 (original to the book; already
   well-scoped per the prior style review).
1. [short-code] **A Text-Tagging Head.** Using the `BERTEncoder` class introduced in
   :numref:`sec_bert`, add a per-token classification head (one `Dense`/`Linear`
   layer applied at every position of `encoded_X`, not just position 0) and run it
   on a toy batch of shape `(batch_size, seq_len)` to confirm the output shape is
   `(batch_size, seq_len, num_tags)`. Deliverable: the head's code + one printed
   shape check. Success: the printed shape matches for a concrete toy `num_tags`
   (e.g. 5).
   *Provenance:* adapted from this section's own :numref:`fig_bert-tagging`
   description (overlap high; turns the prose description into a concrete,
   shape-checked implementation using tools already introduced earlier in the book).
1. [conceptual] **Why `<cls>` vs. Every Token.** Explain in 2–4 sentences why single-
   text classification reads only the `<cls>` representation while text tagging
   reads every token's representation, referencing what each representation is
   pretrained (via next-sentence prediction and masked-token prediction,
   respectively) to summarize. Deliverable: written explanation naming both
   pretraining objectives from :numref:`sec_bert`. Success: both objectives are
   named and connected to the respective read-out choice.
   *Provenance:* original.
1. [conceptual] **STS as Regression, Concretely.** For the pair "A woman is eating
   something." / "A woman is eating meat." (STS-B, similarity 3.000, already quoted
   in this section per :cite:`Cer.Diab.Agirre.ea.2017`), state exactly what changes
   in :numref:`fig_bert-two-seqs`'s architecture to turn an entailment classifier
   into an STS regressor (output width, activation, loss function), and give the
   numeric range the raw output must be rescaled into to match the benchmark's 0–5
   scale. Deliverable: 3 named architecture changes + the rescaling range. Success:
   all three changes are named, not just "make it regression."
   *Provenance:* adapted from Cer, Diab, Agirre, Lopez-Gazpio & Specia 2017 (the
   STS-Benchmark paper this section itself cites; overlap med — requires tracing the
   concrete architecture edit the prose leaves implicit).
1. [short-code] **Fine-Tuning Head Parameter Count.** For a BERT-base-sized encoder
   (`num_hiddens=768`), compute the number of new parameters introduced by each of
   this section's four task heads — single-text classification into `k` classes;
   text-pair classification with the same head shape; and per-token tagging into
   `k` tags — assuming a shared hidden layer width of 768 and `k=3`. Deliverable: the
   parameter counts. Success: the tagging count is recognized as identical
   per-token to the single-text count (the same head applied at every position, not
   `seq_len`-times more parameters).
   *Provenance:* original.

---

## chapter_natural-language-processing-applications/natural-language-inference-bert.md — Natural Language Inference: Fine-Tuning BERT

**Topic:** Loading a pretrained (bert.small / bert.base) BERT checkpoint, wrapping
it in a 3-way `BERTClassifier` head on the `<cls>` token, and fine-tuning end-to-end
on SNLI.
**Current exercises:** 2; disposition: keep 2, rewrite 0, drop 0 — both are already
concrete and well-specified (ex1 names exact hyperparameter substitutions plus a
0.86 accuracy threshold; ex2 asks for a specific compare/derive task); this is the
strongest existing set in the chapter and is kept rather than replaced.

**External sources found:**
- Stanford CS224N / CMU 11-711 minBERT default project — same from-scratch-BERT +
  fine-tune-with-a-head recipe, graded against a numeric dev-accuracy threshold —
  corroborates that this section's own ex1 design (explicit hyperparameters +
  numeric threshold) already matches current field practice. —
  https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/project/default-final-project-handout-minbert-spr2024-updated.pdf
- Devlin, Chang, Lee & Toutanova 2019, NAACL, "BERT: Pre-training of Deep
  Bidirectional Transformers for Language Understanding" — Appendix A.3 reports the
  exact fine-tuning grid used across all their downstream tasks (batch size
  {16, 32}; learning rate {5e-5, 3e-5, 2e-5}; epochs {2, 3, 4}), verified directly
  — a useful contrast to this section's own fixed (`lr=1e-4`, 5 epochs, batch 512)
  recipe. — https://arxiv.org/abs/1810.04805
- Hugging Face NLP Course, Chapter 3.3, "Fine-tuning a model with the Trainer API"
  — covers dynamic (per-batch) padding, in contrast to this section's fixed
  `max_len=128` padding for every example. —
  https://huggingface.co/learn/nlp-course/chapter3/3
- Gururangan, Swayamdipta, Levy, Schwartz, Bowman & Smith 2018 (see
  natural-language-inference-and-dataset.md above) — reused here as a cross-section
  callback comparing the hypothesis-only shortcut against the fine-tuned model.

**Proposed problem set** (5 problems, our reference format):
1. [extended] **Scale to BERT-Base.** Fine-tune bert.base in place of bert.small
   (`num_hiddens` 256→768, `ffn_num_hiddens` 512→3072, `num_heads` 4→12, `num_blks`
   2→12), increasing epochs as needed, and report whether you reach a testing
   accuracy above 0.86. Deliverable: final test accuracy + hyperparameter/epoch log.
   Success: the accuracy is reported directly against the 0.86 threshold.
   *Provenance:* kept from the section's existing ex1 (original to the book;
   already excellent per the prior style review).
1. [conceptual] **Truncation Strategies Compared.** Describe a length-ratio-
   proportional truncation scheme for a premise/hypothesis pair and compare it
   against `SNLIBERTDataset`'s current rule, listing one pro and one con for each.
   Deliverable: 2 pros + 2 cons total. Success: each pro/con references a concrete
   scenario (e.g. a long premise paired with a one-word hypothesis).
   *Provenance:* kept from the section's existing ex2 (original to the book).
1. [short-code] **Dynamic vs. Fixed-Length Padding.** `SNLIBERTDataset` pads every
   example to a fixed `max_len=128`. For one training batch of size 512 sampled from
   `train_set`, compute what fraction of (batch_size × max_len) token slots are
   padding versus real tokens, then estimate the fraction if padding were instead
   applied per-batch to that batch's own longest example. Deliverable: two padding-
   fraction numbers (fixed vs. per-batch). Success: the per-batch number is computed
   from the actual longest example in that sampled batch, not assumed.
   *Provenance:* adapted from the Hugging Face NLP Course, Chapter 3.3 (dynamic
   padding) (overlap med — the padding-efficiency idea is borrowed; the measurement
   is performed on this section's own data).
1. [conceptual] **Hyperparameter Grid, Grounded.** Devlin et al. (2019, Appendix
   A.3) sweep batch size in {16, 32}, learning rate in {5e-5, 3e-5, 2e-5}, and
   epochs in {2, 3, 4}. Compare this grid against the section's own fixed choice
   (`lr=1e-4`, 5 epochs, batch 512) and state, in 2–3 sentences, one reason the
   section's recipe can use values outside the original paper's range (consider
   bert.small vs. bert-base, and SNLI's training-set size vs. GLUE tasks).
   Deliverable: written comparison + reasoned hypothesis. Success: the hypothesis
   names at least one concrete quantitative difference (model size or dataset size).
   *Provenance:* adapted from Devlin, Chang, Lee & Toutanova 2019 (overlap med —
   hyperparameter figures quoted from their Appendix A.3 as a comparison point only).
1. [short-code] **Does Fine-Tuning Erase the Hypothesis-Only Shortcut?** Using the
   hypothesis-only classifier built in :numref:`sec_natural-language-inference-and-dataset`
   (or a fresh one on the same SNLI hypotheses), compare its accuracy against the
   fine-tuned `BERTClassifier`'s accuracy on the identical test set, and report both
   side by side. State in one sentence whether the gap suggests BERT has moved past
   the hypothesis-only shortcut or is still partly exploiting it. Deliverable: two
   accuracy numbers, side by side. Success: both numbers are computed on the
   identical test split.
   *Provenance:* adapted from Gururangan, Swayamdipta, Levy, Schwartz, Bowman &
   Smith 2018 (overlap med — extends their hypothesis-only-baseline finding, already
   used in :numref:`sec_natural-language-inference-and-dataset`, as a cross-section
   callback to this section's fine-tuned model).
