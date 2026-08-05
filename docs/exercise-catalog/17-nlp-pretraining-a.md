# Problem Catalog — chapter_natural-language-processing-pretraining (Group A)

Files: seq2seq.md, word2vec.md, approx-training.md, word-embedding-dataset.md,
word2vec-pretraining.md, glove.md, subword-embedding.md (book order per
`index.md`'s toc; seq2seq is placed first in this chapter as the general
encoder-decoder primer, ahead of word2vec).

**Chapter overview.** Two sources turned out to be near-exact canonical matches:
Stanford CS224n Assignment #2 ("word2vec", Winter 2021 offering) supplies, almost
problem-for-problem, the gradient derivations this book's word2vec.md and
approx-training.md state can be "obtained in the same way" but never actually
show (∂/∂u_o, ∂/∂u_k for negative sampling) — high-overlap adoptions. Stanford
CS336 ("Language Modeling from Scratch") Assignment 1 is the canonical match for
subword-embedding.md: it assigns byte-level BPE training/encoding at production
scale (TinyStories/OpenWebText, compression-ratio reporting, longest-token
sanity checks) that our toy four-word BPE trainer can replicate at small scale.
SLP3's current (Jan 2026) draft chapter 12 ("Machine Translation") has exactly
one exercise — hand-computing chrF — and it is a direct, verified match for
seq2seq.md's chrF metric. Two clean **gaps**: SLP3's Chapter 5 ("Embeddings")
and Chapter 12 both currently ship an "Exercises" heading with **zero** exercises
underneath (verified by fetching and extracting both PDFs) — there is no live
SLP3 exercise tradition for word2vec/GloVe math or for anything but chrF in MT.
Hierarchical softmax and the word2vec minibatch/subsampling data pipeline have
**no exercise tradition anywhere checked** — courses hand students this
infrastructure pre-built rather than assigning it. A real **content gap** in
this book: nowhere in the whole pretraining chapter is there a bias/fairness
exercise, despite CS224n devoting three of Assignment 1's nine sub-questions to
exactly that; word2vec-pretraining.md is the first point with trained vectors
in hand, so that's where the addition lands. Existing-set quality is bimodal:
seq2seq.md's 7 exercises are already the best-specified set in the whole NLP
group (quantitative sweeps, explicit citations) and are kept nearly intact;
word-embedding-dataset.md's 3 and word2vec-pretraining.md's 2 are the weakest
(bare "Can you...?" / "see how it affects" filler) and needed full rewrites.

---

## chapter_natural-language-processing-pretraining/seq2seq.md — Encoder-Decoder Models for Sequence Transduction

**Topic:** The Encoder/Decoder/EncoderDecoder abstraction; a GRU seq2seq
translator on English–French Tatoeba data tokenized with one shared byte-level
BPE vocabulary; teacher forcing and masked cross-entropy; greedy vs. beam-search
decoding; chrF vs. BLEU evaluation; the fixed-vector-bottleneck experiment.

**Current exercises:** 7; disposition: keep 7, rewrite 0, drop 0 — the prior
style review calls this "by far the best-specified exercise set in the entire
NLP group" (explicit value sets, precise `:numref:`/`:citet:` grounding, no
filler tone), and a direct read confirms it: every exercise names a concrete
manipulation and a comparison target. Nothing here needs fixing; the one gap is
a hand-computation warm-up before the metric-disagreement exercise, which SLP3
supplies almost verbatim.

**External sources found:**
- Stanford CS224n, Assignment #4 "Neural Machine Translation with RNNs and
  Analyzing NMT Systems" (Winter 2021 offering, same course-archive quarter as
  Assignment 2 below) — Part 2(f): given a source sentence, two reference
  translations, and two candidate translations, compute BLEU by hand (modified
  n-gram precision, brevity penalty) for both candidates against both one- and
  two-reference sets, then judge whether BLEU's ranking is believable —
  https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1214/assignments/a4.pdf
- Stanford CS224n, Assignment #4, Part 1(g) — written question asking students
  to explain (in ~3 sentences) what effect an encoder padding mask has on an
  attention computation, and why masking is necessary — same URL as above; low
  overlap with our masked-loss exercise (different masking site: attention
  scores vs. loss terms) but the same "why mask padding" pedagogical shape.
- Jurafsky & Martin, *Speech and Language Processing* (3rd ed. draft, Jan 6
  2026), Chapter 12 "Machine Translation," Exercise 12.1 — "Compute by hand the
  chrF2,2 score for HYP2 ... (the answer should round to .62)"; the chapter body
  explicitly sets this up ("we'll leave the computation of the chrF value for
  HYP2 as an exercise for the reader") — https://web.stanford.edu/~jurafsky/slp3/12.pdf
  (verified by downloading and text-extracting the PDF).
- **Finding:** SLP3 Chapter 12's entire "Exercises" section contains only that
  one chrF item — no assigned exercise on beam width, length normalization, or
  BLEU-vs-chrF disagreement exists there, even though the chapter body discusses
  all three at length. Chapter 12 does discuss length-normalized beam search in
  its own prose, useful as grounding but not itself an assigned exercise.

**Proposed problem set** (8 problems):
1. [short-code] **Two Separate Tokenizers.** Rebuild `MTFraEng` with one
   byte-level BPE tokenizer per language instead of one shared tokenizer, retrain,
   and compare final training loss and chrF on the same three held-out
   sentences. State when separate vocabularies should help and when they should
   hurt, referencing what a shared vocabulary buys on this English/French pair.
   *Provenance:* original (book exercise, unchanged).
2. [short-code] **Beam Width Sweep.** Decode the three example sentences and a
   handful of longer held-out sentences at beam width k ∈ {1, 2, 4, 8, 16},
   scoring each with chrF. Report the k at which quality stops improving and
   relate this to the large-beam discussion in `:numref:`sec_beam-search``.
   *Provenance:* original (book exercise, unchanged).
3. [short-code] **Length-Normalization Exponent.** Decode with `d2l.beam_search`'s
   length-normalization exponent α ∈ {0, 0.75, 1.5} and relate the length of the
   winning translation at each α to `:eqref:`eq_beam-search-score``.
   *Provenance:* original (book exercise, unchanged).
4. [short-code] **Unmasked Loss Ablation.** Retrain with the masked loss replaced
   by plain cross-entropy over all positions including padding, and report what
   changes in the resulting translations and why.
   *Provenance:* original (book exercise, unchanged).
5. [short-code] **GRU-to-LSTM Swap.** Replace the GRU with an LSTM
   (`:numref:`sec_lstm``) in both encoder and decoder and report whether
   translation quality changes at this scale.
   *Provenance:* original (book exercise, unchanged).
6. [short-code] **Context Injection Timing.** Feed the context vector **c** only
   at the decoder's first step (as in :citet:`Sutskever.Vinyals.Le.2014`)
   instead of at every step, and explain why repeating the context at every step
   might help a small model.
   *Provenance:* original (book exercise, unchanged).
7. [conceptual] **chrF by Hand.** Using the chrF formula given in this section
   (order n up to 6, β = 2), compute by hand the chrF2,2 score (n = 2) for the
   candidate translation "the love can always do" against reference "love can
   always find a way," showing character-bigram precision and recall before
   combining them. Check your arithmetic against `chrf()` from this section.
   *Provenance:* adapted from Jurafsky & Martin SLP3 (2026 draft), Ch. 12,
   Exercise 12.1 (overlap: high — same metric and hand-computation format;
   our example sentence is drawn from CS224n A4's BLEU example rather than
   SLP3's own HYP2, so cite both on adoption).
8. [short-code] **chrF versus BLEU Disagreement.** Score the example
   translations with both `chrf` and `d2l.bleu`. Construct or find a
   translation on which the two metrics disagree the most, and explain the
   mechanism (e.g. word-order sensitivity, partial credit for near-miss words).
   *Provenance:* original (book exercise, unchanged).

---

## chapter_natural-language-processing-pretraining/word2vec.md — Word Embedding (word2vec)

**Topic:** One-hot vectors' failure to encode similarity; the skip-gram and
CBOW models; softmax-based conditional probabilities; the skip-gram loss
gradient with respect to the center-word vector.

**Current exercises:** 3; disposition: keep 3, rewrite 0, drop 0 — all three
are legitimate derivation/reflection questions with a clear referent (softmax
complexity; phrase vectors, hinting at the word2vec paper's §4; dot-product vs.
cosine-similarity relationship). No code exists in this section, so all
proposed additions stay conceptual, matching the existing pattern.

**External sources found:**
- Stanford CS224n, Assignment #2 "word2vec" (Winter 2021) — Part 1(b): derive
  ∂J_naive-softmax/∂v_c in vectorized form (shape convention: same shape as v_c),
  answer in terms of y, ŷ, U — https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1214/assignments/a2.pdf
  (verified by downloading and reading the PDF directly).
- Stanford CS224n, Assignment #2, Part 1(c) — derive ∂J_naive-softmax/∂u_w for
  the two cases w = o and w ≠ o, in terms of y, ŷ, v_c — same URL; this is
  exactly the derivation our section states can be "obtained in the same way"
  for u_o but never shows.
- Stanford CS224n, Assignment #2, Part 1(a) — show in one line that the
  naive-softmax loss equals the cross-entropy loss between one-hot y and
  predicted ŷ — same URL; low-priority addition since our section already
  frames the loss this way in prose without the explicit y/ŷ vector notation.
- **Finding:** Jurafsky & Martin SLP3 (2026 draft) Chapter 5 "Embeddings" covers
  skip-gram/CBOW/negative-sampling in its body text in similar depth to this
  book, but its "Exercises" heading is followed immediately by the
  bibliography — there are currently **zero** exercises there (verified by
  downloading and text-extracting https://web.stanford.edu/~jurafsky/slp3/5.pdf).

**Proposed problem set** (5 problems):
1. [conceptual] **Softmax Gradient Complexity.** What is the computational
   complexity of calculating one gradient in :eqref:`eq_skip-gram-grad`? What
   goes wrong when the dictionary is huge, and what does that motivate?
   *Provenance:* original (book exercise, unchanged).
2. [conceptual] **Multi-Word Phrase Vectors.** English has fixed multi-word
   phrases such as "New York." Propose how to train a vector for such a phrase.
   Hint: see Section 4 of :cite:`Mikolov.Sutskever.Chen.ea.2013`.
   *Provenance:* original (book exercise, unchanged).
3. [conceptual] **Dot Product versus Cosine Similarity.** Taking skip-gram as an
   example, what is the relationship between two word vectors' dot product and
   their cosine similarity? For semantically similar words, why should the
   trained cosine similarity of their vectors tend to be high?
   *Provenance:* original (book exercise, unchanged).
4. [conceptual] **Missing Context-Vector Gradients.** The section derives
   ∂ log P(w_o|w_c)/∂v_c in full (:eqref:`eq_skip-gram-grad`) but only asserts
   that "the gradients for the other word vectors can be obtained in the same
   way." Derive ∂ log P(w_o|w_c)/∂u_o and ∂ log P(w_o|w_c)/∂u_j for j ≠ o
   explicitly, and confirm each reduces to a softmax-weighted correction term
   analogous to :eqref:`eq_skip-gram-grad`.
   *Provenance:* adapted from Stanford CS224n Assignment #2 (Winter 2021),
   Part 1(c) (overlap: high; cite on adoption).
5. [conceptual] **Skip-Gram versus CBOW Update Cost.** For a context window of
   size m, count how many per-pair gradient computations of the form in
   :eqref:`eq_skip-gram-grad` one skip-gram update over a window performs,
   versus one CBOW update using :eqref:`eq_cbow-gradient`. State which model
   issues more SGD updates per pass over the corpus and which does more
   arithmetic per update, and argue the total work per corpus pass is roughly
   the same either way.
   *Provenance:* original (extends exercise 1's complexity theme to the
   two-model comparison the section sets up in prose but never asks about).

---

## chapter_natural-language-processing-pretraining/approx-training.md — Approximate Training

**Topic:** Negative sampling (binary logistic loss over K noise words) and
hierarchical softmax (binary-tree factorization of the skip-gram softmax) as
approximations to the exact softmax's O(|V|) cost.

**Current exercises:** 3; disposition: keep 1, rewrite 2, drop 0 — exercise 2
("verify :eqref:`eq_hi-softmax-sum-one` holds") is a clean, hint-free derivation
and is kept as-is. Exercises 1 and 3 use the fragmentary "How can we / How to
X?" stems the group-wide style review flags and give no deliverable, so their
content is kept but each is given an explicit output to produce.

**External sources found:**
- Stanford CS224n, Assignment #2 "word2vec" (Winter 2021), Part 1(e)+(f) —
  first derive σ'(x) = σ(x)(1 − σ(x)) for the sigmoid, then use it to derive
  ∂J_neg-sample/∂v_c, ∂J_neg-sample/∂u_o, and ∂J_neg-sample/∂u_k for the
  negative-sampling loss J_neg-sample(v_c, o, U) = −log σ(u_o^T v_c) −
  Σ_k log σ(−u_k^T v_c) — https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1214/assignments/a2.pdf
  (verified directly). This is essentially the same loss our section gives in
  the paragraph before :eqref:`eq_hi-softmax-sum-one`'s sibling equation, with
  no derivation of its gradient anywhere in the d2l chapter.
- Stanford CS224n, Assignment #2, Part 1(g) — repeat the u_k derivative
  *without* assuming the K negative samples are distinct, via splitting the sum
  into "samples equal to u_k" and "samples not equal to u_k" — same URL.
- **Finding:** No source checked (CS224n, SLP3, CS336) assigns anything on
  hierarchical softmax specifically — it has effectively dropped out of the
  modern teaching tradition in favor of negative sampling and full-softmax with
  efficient kernels. SLP3 Ch. 5's exercises section is empty in any case (see
  word2vec.md entry above), so this is a genuine, if unsurprising, gap.

**Proposed problem set** (6 problems):
1. [conceptual] **Negative-Sampling Distribution Design.** Propose a concrete
   distribution P(w) for drawing noise words (other than uniform over the
   vocabulary), argue why very frequent words should be sampled less often than
   their raw corpus frequency would suggest, and state one property you would
   check empirically to validate your choice.
   *Provenance:* rewrite of original exercise 1 (adds an explicit deliverable
   and success check to the bare "How can we sample noise words?" prompt).
2. [conceptual] **Hierarchical-Softmax Probabilities Sum to One.** Verify that
   :eqref:`eq_hi-softmax-sum-one` holds.
   *Provenance:* original (book exercise, unchanged).
3. [conceptual] **CBOW under Both Approximations.** Write out explicitly the
   negative-sampling loss and the hierarchical-softmax approximation for the
   CBOW conditional probability P(w_c | context), by substituting into the CBOW
   loss the same way :eqref:`eq-negative-sample-conditional-prob` substitutes
   into the skip-gram loss.
   *Provenance:* rewrite of original exercise 3 (turns "How to train CBOW using
   negative sampling and hierarchical softmax?" into a concrete written-output
   task).
4. [conceptual] **Sigmoid Derivative and Negative-Sampling Gradients.** Derive
   σ'(x) = σ(x)(1 − σ(x)). Then derive ∂J_neg-sample/∂v_c and
   ∂J_neg-sample/∂u_o for the single-pair negative-sampling loss given in this
   section, assuming the K negative samples w_1, ..., w_K are distinct.
   *Provenance:* adapted from Stanford CS224n Assignment #2 (Winter 2021),
   Parts 1(e)-(f) (overlap: high; cite on adoption).
5. [conceptual] **Repeated Negative Samples.** Derive ∂J_neg-sample/∂u_k for the
   same loss, first assuming the K samples are distinct, then without that
   assumption (hint: split the sum over samples equal to w_k and samples not
   equal to w_k).
   *Provenance:* adapted from Stanford CS224n Assignment #2 (Winter 2021),
   Parts 1(f)-(g) (overlap: high; cite on adoption).
6. [conceptual] **Crossover Vocabulary Size.** Negative sampling costs O(K) dot
   products per update; hierarchical softmax costs O(log₂|V|). Using K = 5 (the
   value :numref:`sec_word2vec_data` actually uses), solve for the vocabulary
   size |V| at which hierarchical softmax first requires fewer dot products,
   and check whether the PTB vocabulary (roughly 10,000 words) falls above or
   below that crossover.
   *Provenance:* original (numeric extension of the asymptotic facts the
   section states, tied to the K=5 constant used two sections later).

---

## chapter_natural-language-processing-pretraining/word-embedding-dataset.md — The Dataset for Pretraining Word Embeddings

**Topic:** Reading the PTB corpus; subsampling frequent words; extracting
(center, context) pairs with a randomized window; negative sampling of noise
words via `RandomGenerator`; padding/masking into minibatches
(`contexts_negatives`, `masks`, `labels`); the `_pad_ptb`/`load_data_ptb`
one-time-padding pipeline.

**Current exercises:** 3; disposition: rewrite 3, drop 0 — all three ask a
reasonable question (subsampling's effect on running time; the `RandomGenerator`
cache size `k`'s effect on loading speed; other speed-relevant hyperparameters)
but none names a metric or a candidate value, which is exactly the "experiment
and see" pattern the group-wide style review flags for this file. Every
original idea is kept; each gets an explicit measurement and candidate values.

**External sources found:**
- **Finding: no good external exercise tradition for this topic.** None of
  CS224n Assignment #2 (which loads Stanford Sentiment Treebank via a provided
  script with no data-pipeline exercise), SLP3 (Chapter 5's exercises are
  empty), or CS336 (a different domain — Transformer-scale tokenization, not
  word2vec minibatch construction) assigns anything resembling
  subsampling/negative-sampling/padding pipeline construction as a *problem* —
  every source we checked hands students this kind of infrastructure
  pre-built rather than asking them to build or benchmark it. This is a
  genuine, useful negative finding, not a gap in our search.

**Proposed problem set** (6 problems):
1. [conceptual] **Subsampling Discard Probabilities.** Using
   P(w_i) = max(1 − √(t/f(w_i)), 0) with t = 10⁻⁴, compute the discard
   probability for a word with relative frequency f(w) = 10⁻² and for one with
   f(w) = 10⁻⁵. State, in one sentence, the general trend this implies for how
   discard probability scales with relative frequency once f(w) > t.
   *Provenance:* original.
2. [short-code] **Subsampling Wall-Clock Effect.** Using `time.perf_counter`,
   time `get_centers_and_contexts` on the PTB `corpus` once built from
   `subsampled` sentences and once built directly from `sentences` (skipping
   `subsample`). Report the ratio of the two wall-clock times and the ratio of
   the two resulting `# center-context pairs` counts, and explain how the two
   ratios relate.
   *Provenance:* rewrite of original exercise 1 (adds the missing metric; also
   fixes the exercise's subject-verb agreement defect the style review flagged).
3. [short-code] **RandomGenerator Cache Size.** Benchmark `RandomGenerator.draw()`
   called 10⁶ times for cache sizes k ∈ {10, 100, 1000, 10000 (the default),
   100000}. Tabulate calls/second for each k and explain the shape of the curve
   in terms of how often `random.choices` is invoked internally.
   *Provenance:* rewrite of original exercise 2 (supplies the missing candidate
   values and metric for "set k to other values and see how it affects... speed").
4. [short-code] **Hyperparameter Sweep for Loading Speed.** Holding the PTB
   corpus fixed, sweep `max_window_size` ∈ {2, 5, 10} and `num_noise_words` K ∈
   {2, 5, 10}. For each combination, report `# center-context pairs`, the
   resulting `max_len` in `batchify`, and wall-clock time for
   `get_centers_and_contexts` + `get_negatives` + `batchify`. State which
   hyperparameter has the larger effect on `max_len` and why.
   *Provenance:* rewrite of original exercise 3 (names the specific
   hyperparameters and requires a table instead of open-ended reflection).
5. [conceptual] **Zero-Padding Index Safety.** `_pad_ptb` and `batchify` both
   pad with index 0. Is index 0 ever a valid word or noise-word index in this
   vocabulary (check how `d2l.Vocab` assigns index 0)? Explain why padding with
   0 is safe here regardless, referencing what `masks` guarantees about the
   loss.
   *Provenance:* original.
6. [extended] **Streaming Minibatch Construction.** `_pad_ptb` materializes the
   whole padded dataset (`contexts_negatives`, `masks`, `labels`) in memory
   before batching. Using the actual `max_len` and `len(vocab)`-independent
   int64/float32 storage from this section's run, estimate in GB the memory
   cost of this eager-padding approach for a corpus 100× the size of PTB. Then
   sketch (in a short prototype or in words) a per-batch padding scheme —
   recovering the older `batchify`-per-minibatch design this section mentions
   in "Putting It All Together" — and estimate the memory it would save.
   *Provenance:* original (extends the section's own discussion of the
   `batchify`-vs-`_pad_ptb` tradeoff).

---

## chapter_natural-language-processing-pretraining/word2vec-pretraining.md — Pretraining word2vec

**Topic:** Implementing skip-gram with embedding layers and batched matrix
multiplication; masked binary cross-entropy; the training loop; using trained
embeddings for cosine-similarity nearest-neighbor queries via
`get_similar_tokens`.

**Current exercises:** 2; disposition: rewrite 2, drop 0 — exercise 1 is a bare
"Can you improve the results by tuning hyperparameters?" filler question with
no named hyperparameter or metric; exercise 2 asks a good conceptual question
(benefits of per-epoch resampling of negatives/contexts) but "try to implement
this" has no stated success check. Both ideas are kept and given explicit
deliverables.

**External sources found:**
- Stanford CS224n, Assignment #1 "Exploring Word Vectors" (Spring 2024, due
  April 9 2024) — Questions 2.7–2.9: find profession-related words whose
  nearest neighbors (by cosine similarity) reveal gender bias, independently
  find another bias pattern, and explain likely sources plus mitigation
  strategies — https://web.stanford.edu/class/cs224n/assignments/a1_preview/exploring_word_vectors.html
  (verified directly, including the due-date string). CS224n probes pretrained
  GloVe vectors via `gensim`'s `most_similar`; this section is the first point
  in the book with trained vectors and a working nearest-neighbor query
  (`get_similar_tokens`) in hand, so the same probing method transfers directly
  even though the underlying vectors and toolchain differ.
- Stanford CS224n, Assignment #2, Part 2(c) "Show time!" — train word vectors
  on the Stanford Sentiment Treebank via a provided script, then "briefly
  explain in at most three sentences what you see" in a 2D visualization of the
  result — same URL as approx-training.md's CS224n A2 entry above; low overlap
  (different toolchain/visualization method), used only as loose inspiration.
- **Finding:** SLP3 Ch. 5's exercises are empty (see word2vec.md entry) — no
  tradition there for this section's implementation-level topics either.

**Proposed problem set** (5 problems):
1. [short-code] **Tuned Nearest Neighbors.** Using the trained model, find
   semantically similar words for three query words of your choice. Then retrain
   after changing one hyperparameter (`embed_size` ∈ {50, 100, 300}, or
   `num_epochs`, or `lr`) and report the top-3 neighbors for the same three
   query words before and after, stating whether the change visibly improved
   neighbor quality.
   *Provenance:* rewrite of original exercise 1 (names concrete hyperparameters
   and a before/after reporting requirement in place of "Can you improve...?").
2. [short-code] **Per-Epoch Resampling Effect.** Implement re-sampling context
   and noise words for each center word freshly every epoch (rather than once,
   as the current `load_data_ptb` does), and compare the resulting loss curve
   against the original. In 2-3 sentences, explain why per-epoch resampling
   acts like a form of data augmentation.
   *Provenance:* rewrite of original exercise 2 (keeps the implementation task,
   adds the loss-curve comparison as the success check).
3. [short-code] **Bias in Learned Embeddings.** Using `get_similar_tokens`,
   query at least two profession- or attribute-related words available in the
   PTB vocabulary and check whether their nearest neighbors correlate with an
   unrelated social category (e.g. gendered names or pronouns present in the
   corpus). Report the neighbor lists and give a 2-3 sentence explanation of why
   a co-occurrence-based training objective would reproduce such a correlation
   from the training corpus.
   *Provenance:* adapted from Stanford CS224n Assignment #1 (Spring 2024),
   Questions 2.7-2.9 (overlap: medium — same cosine-similarity probing method,
   different vectors/toolchain; cite on adoption).
4. [conceptual] **Center versus Context Tables.** The book text notes that
   center-word vectors are typically used as "the" word representation.
   Compare `get_similar_tokens('chip', 3, net[0])` (center table) against the
   same query run on the context-word table (`net[1]` / `embed_u`). Do the two
   neighbor lists agree? Give one hypothesis for why they might differ despite
   representing "the same" word.
   *Provenance:* original (directly invited by this section's own note that
   "the context table was trained jointly and contains similar information").
5. [extended] **Training-Length Convergence Sweep.** Retrain with `num_epochs` ∈
   {5, 15, 30}, holding `lr` fixed, and track final training loss plus the
   top-3 neighbors of three fixed query words at each checkpoint. Produce a
   small table showing how neighbor quality changes with epoch count, and state
   at what point returns visibly diminish.
   *Provenance:* original.

---

## chapter_natural-language-processing-pretraining/glove.md — Word Embedding with Global Vectors (GloVe)

**Topic:** Reinterpreting skip-gram via global co-occurrence counts x_ij; the
GloVe weighted squared-loss objective with center/context bias terms and
weighting function h(x); the ratio-of-co-occurrence-probabilities
interpretation (ice/steam/solid/gas/water/fashion example).

**Current exercises:** 2; disposition: keep 2, rewrite 0, drop 0 — the prior
style review calls both "concrete, hint-anchored derivations," and a direct
read confirms it (distance-weighted p_ij redesign hinting at GloVe paper §4.2;
bias-term equivalence question). No code exists in this section, so, as with
word2vec.md, all additions stay conceptual.

**External sources found:**
- Stanford CS224n, Assignment #1 "Exploring Word Vectors" (Spring 2024),
  Question 2.1 — compare a GloVe embedding visualization against a
  co-occurrence-matrix + truncated-SVD embedding visualization and discuss
  similarities/differences — https://web.stanford.edu/class/cs224n/assignments/a1_preview/exploring_word_vectors.html
  (verified directly); low overlap with our conceptual (no-code) treatment,
  since CS224n's version is empirical/visual and code-based.
- **Finding:** No source checked assigns a derivation exercise on GloVe's own
  weighting function h(x) or its bias terms specifically — CS224n's word2vec
  assignment (A2) does not cover GloVe at all, and SLP3 Ch. 5's exercises are
  empty (see word2vec.md entry). The GloVe paper itself
  (:cite:`Pennington.Socher.Manning.2014`) is the only primary source for this
  material, already cited by the book's own exercise 1.

**Proposed problem set** (5 problems):
1. [conceptual] **Distance-Weighted Co-occurrence.** If words w_i and w_j
   co-occur in the same context window, how could their distance in the text
   sequence be used to redesign the calculation of p_ij? Hint: see Section 4.2
   of :cite:`Pennington.Socher.Manning.2014`.
   *Provenance:* original (book exercise, unchanged).
2. [conceptual] **Bias Term Equivalence.** For any word, are its center-word
   bias b_i and context-word bias c_i mathematically equivalent in GloVe? Why?
   *Provenance:* original (book exercise, unchanged).
3. [conceptual] **Weighting Function Boundary Behavior.** The section defines
   h(x) = (x/c)^α for x < c, else 1. Show that this choice makes h(0) = 0 (in
   the x→0 limit), and explain in 2-3 sentences why this specific property —
   not merely "h is increasing" — is what lets GloVe training skip all
   zero-count (i, j) pairs, tying back to the text's claim that only non-zero
   x_ij need be sampled at each iteration.
   *Provenance:* original.
4. [conceptual] **GloVe versus Unweighted Least Squares.** Compare GloVe's
   objective (:eqref:`eq_glove-loss`) to the simpler alternative of fitting
   u_j^T v_i ≈ log x_ij by ordinary, unweighted, unbiased least squares. Name
   one concrete failure mode this simpler model would have on rare
   co-occurring pairs (x_ij = 1) that GloVe's weighting function h(x_ij) and
   bias terms are each separately designed to fix.
   *Provenance:* inspired by Stanford CS224n Assignment #1 (Spring 2024),
   Question 2.1 (overlap: low — CS224n's question is empirical/visual with
   code; this is a conceptual why-question with no code available at this
   point in the book).
5. [conceptual] **Extending the Ice/Steam Table.** :numref:`tab_glove` lists
   p1/p2 ratios for "solid," "gas," "water," "fashion" relative to "ice" and
   "steam." Pick two new probe words and reason, using the section's four-case
   classification (related to ice only / steam only / both / neither), about
   whether their p(w_k|ice)/p(w_k|steam) ratio should be large, small, or ≈1.
   *Provenance:* original (extends the section's own worked example with fresh
   probe words).

---

## chapter_natural-language-processing-pretraining/subword-embedding.md — Subword Embedding

**Topic:** The fastText model (a word as the sum of its character n-gram
subword vectors); byte pair encoding (BPE) — greedy statistical merging from a
character vocabulary, implemented here as `get_max_freq_pair`, `merge_symbols`,
and `segment_BPE` over a toy four-word frequency dictionary.

**Current exercises:** 4; disposition: keep 1, rewrite 3, drop 0 — exercise 1
(the 6-gram explosion, hinting at the fastText paper §3.2) and exercise 3
(merges needed to reach vocabulary size m) are already precise; exercise 1 is
kept, exercise 3 is kept but given an added sanity-check-against-the-code
deliverable (folded into problem 3 below). Exercises 2 and 4 use the
group's recurring fragmentary "How to X?" stem and name no deliverable, so
their content is kept but each now produces a concrete artifact.

**External sources found:**
- Stanford CS336 "Language Modeling from Scratch," Assignment 1 "Basics"
  (current main-branch version 26.0.3, Spring 2026) — Problem (train_bpe):
  train a byte-level BPE tokenizer given `input_path`, `vocab_size`, and
  `special_tokens`, returning a `vocab: dict[int, bytes]` and an ordered
  `merges: list[tuple[bytes, bytes]]` —
  https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_assignment1_basics.pdf
  (verified by downloading and text-extracting the PDF).
- Stanford CS336 Assignment 1, Problem (train_bpe_tinystories) — train on the
  real TinyStories corpus at vocabulary size 10,000, serialize the result, and
  report the training time/memory and the longest token in the learned
  vocabulary, with "does it make sense?" as the actual assigned question — same
  PDF.
- Stanford CS336 Assignment 1, Problem (tokenizer_experiments), part (a) —
  sample documents, encode them with the trained tokenizer, and report each
  tokenizer's compression ratio in bytes/token — same PDF.
- Stanford CS336 Assignment 1, Problems (unicode1)/(unicode2) — byte-level
  edge cases (e.g. `chr(0)`, malformed UTF-8 decoding, a 2-byte sequence that
  decodes to nothing) motivating *why* production BPE operates on raw bytes
  rather than characters — same PDF; our section's BPE operates on a
  hand-picked 28-symbol lowercase-letter alphabet, not raw bytes, so this is
  lower-overlap inspiration for a byte-level extension exercise rather than a
  direct match.
- **Finding:** Jurafsky & Martin SLP3 (2026 draft) Chapter 2 "Words and Tokens"
  teaches BPE in its body text in comparable depth to this section, but its
  seven assigned exercises (2.1-2.7) are all about regular expressions, edit
  distance, and an ELIZA-style program — none touches BPE or subword
  tokenization at all (verified by downloading and text-extracting
  https://web.stanford.edu/~jurafsky/slp3/2.pdf). A genuine gap between a
  chapter's expository content and its assigned problems.

**Proposed problem set** (7 problems):
1. [conceptual] **The 6-Gram Explosion.** There are about 3×10⁸ possible
   6-grams in English. What is the issue when there are too many subwords, and
   how would you address it? Hint: see the end of Section 3.2 of
   :cite:`Bojanowski.Grave.Joulin.ea.2017`.
   *Provenance:* original (book exercise, unchanged).
2. [conceptual] **CBOW-Style Subword Vectors.** fastText represents a *center*
   word as the sum of its subword vectors. Give the analogous formula for a
   CBOW-style subword embedding model, where a center word is instead predicted
   from the (summed) subword vectors of its *context* words, and state in 2-3
   sentences what changes relative to fastText's skip-gram-based objective.
   *Provenance:* rewrite of original exercise 2 (turns "How to design a subword
   embedding model based on CBOW?" into a task with a stated written
   deliverable).
3. [short-code] **Vocabulary Size from Merge Count.** Given an initial symbol
   vocabulary of size n and a target vocabulary size m, how many BPE merges are
   needed? State the formula, then confirm it against this section's own run:
   the initial `symbols` list has 28 entries (26 letters, `'_'`, `'[UNK]'`), and
   after `num_merges = 10` iterations, check that `len(symbols)` matches your
   formula's prediction.
   *Provenance:* rewrite of original exercise 3 (adds the sanity-check-against-the-code
   deliverable to an already-precise question).
4. [short-code] **Phrase-Level BPE.** Using the same three functions
   (`get_max_freq_pair`, `merge_symbols`, `segment_BPE`) unmodified, treat each
   short sentence in a small toy corpus as a "word" (join its constituent
   words with a placeholder separator instead of joining a word's characters),
   and run a few merge iterations. Report the first phrase-level merge produced
   and explain why it was the most frequent pair.
   *Provenance:* rewrite of original exercise 4 (turns "How to extend BPE to
   extract phrases?" into a runnable, checkable task using the section's own
   code unmodified).
5. [short-code] **Compression Ratio versus Merges.** Run this section's BPE
   trainer for `num_merges` ∈ {5, 10, 20, 40} on the `raw_token_freqs` toy
   corpus (optionally extended with a few more words), and for each value
   report the compression ratio = (total characters in the corpus) / (total
   number of symbols after `segment_BPE`). Tabulate ratio vs. `num_merges` and
   describe the trend.
   *Provenance:* adapted from Stanford CS336 Assignment 1, Problem
   (tokenizer_experiments), part (a) (overlap: medium — same bytes/token-style
   compression metric, applied here to the section's toy character-level
   trainer rather than CS336's production byte-level one on a real corpus;
   cite on adoption).
6. [short-code] **BPE on Real Text.** Adapt this section's `token_freqs`-building
   step to read word frequencies from a short paragraph of real English text
   (a few hundred words) instead of the four hard-coded words, run
   `num_merges = 50`, and report the longest merged symbol in the final
   `symbols` list. Does it correspond to a real morpheme or common word, the
   way CS336's TinyStories run checks for?
   *Provenance:* adapted from Stanford CS336 Assignment 1, Problem
   (train_bpe_tinystories), part (a) (overlap: medium — same "run on real text,
   inspect the longest learned token" idea, at toy scale with the section's own
   character-level trainer rather than CS336's production byte-level one with
   multiprocessing; cite on adoption).
7. [extended] **Byte-Level BPE from Scratch.** This section's BPE operates on a
   hand-picked 28-symbol alphabet of English lowercase letters. Reimplement
   `get_max_freq_pair`/`merge_symbols`/`segment_BPE` to instead start from the
   256 possible UTF-8 byte values, run it on a short text sample containing at
   least one non-ASCII character (e.g. an accented letter or emoji), and
   confirm the byte-level version produces no `[UNK]` output for input the
   character-level version could not segment.
   *Provenance:* adapted from Stanford CS336 Assignment 1, Problems
   (unicode2) and (train_bpe) (overlap: medium — CS336 assigns full
   production-scale byte-level BPE training/encoding/decoding with regex
   pre-tokenization and special-token handling; this is a scaled-down
   single-file reimplementation using only what this section has already
   shown; cite on adoption).
