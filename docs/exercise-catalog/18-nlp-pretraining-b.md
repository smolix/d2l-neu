# Chapter Overview — chapter_natural-language-processing-pretraining (group b: similarity-analogy, bert, bert-dataset, bert-pretraining)

Best sources found: (1) Stanford CS224N Assignment 1 "Exploring Word Vectors" is a near-perfect
match for similarity-analogy.md — it already poses similarity/polysemy, analogy-completion, and
gender-bias probes on the same GloVe vectors this section loads. (2) The CMU 11-711
"minbert-assignment" (Neubig et al.), reused verbatim as the CS224N Default Final Project
("minBERT and Downstream Tasks"), is the best match for bert.md/bert-pretraining.md — a genuine
from-scratch BERT implementation + [CLS]-based downstream fine-tuning, at project scale. (3) The
Hugging Face LLM Course ch.7 §3 "Fine-tuning a masked language model" is the best fit for the
*mechanics* of bert-dataset.md/bert-pretraining.md (whole-word masking, perplexity tracking).
Coverage gap: SLP3's current online draft has **no populated Exercises section at all** in either
ch.5 "Embeddings" or ch.10 "Masked Language Models" (both checked directly — just the bare
heading) — a real gap in that source, though its prose (WEAT/debiasing citations, CLS fine-tuning
diagrams) still backs several problems below. No course targets BERT's *dataset-construction*
mechanics (NSP pair sampling, 80/10/10 masking) as a standalone exercise outside HF's course and
the BERT/RoBERTa papers themselves — most treat it as a solved library call. Existing-set quality:
bert-pretraining.md's two exercises are already well-posed (both kept, matching the CS224N/CMU
question quality bar); similarity-analogy.md and bert-dataset.md have the weakest existing sets
(underspecified "how can we"/handed-solution patterns per the prior style review) and get the most
rewrites. Totals: 4 sections, 8 existing exercises (keep 4, rewrite 4, drop 0), 24 proposed
problems.

---

## chapter_natural-language-processing-pretraining/similarity-analogy.md — Word Similarity and Analogy

**Topic:** Applying pretrained static word vectors (GloVe/fastText) to cosine-similarity nearest-neighbor lookup and vector-arithmetic analogy completion.
**Current exercises:** 2; disposition: keep 0, rewrite 2 — ex 1 ("test the fastText results") names no metric or comparison target; ex 2 ("how can we find similar words... faster?") is a pure reading prompt with no algorithm or complexity target to name — both flagged in the prior style review and worth upgrading rather than dropping, since the section's underlying code (`knn`, `get_analogy`) is otherwise solid and reusable.

**External sources found:**
- Stanford CS224N, Assignment 1 "Exploring Word Vectors" (Q2.2, Q2.4–2.6, Q2.7–2.9) — asks students to find a polysemous word whose top-10 nearest neighbors mix both senses (and explain why most fail this test); to solve `man:grandfather::woman:x` via vector arithmetic and find/break an analogy of their own; and to probe gender bias by comparing `most_similar(['man','profession'],['woman'])` vs. the swapped query, then find an independent bias example and discuss one mitigation — https://web.stanford.edu/class/cs224n/assignments/a1_preview/exploring_word_vectors.html
- SLP3 (Jurafsky & Martin), ch. 5 "Embeddings", §5.8–5.9 — formalizes the analogy task as $a:b::a^*:b^*$ (citing Turney and Littman 2005) and surveys intrinsic evaluation sets (WordSim-353, SimLex-999, the TOEFL synonym test); discusses embedding bias via Caliskan, Bryson & Narayanan 2017 (*Science* 356(6334):183–186 — the WEAT paper, applied to GloVe cosines) and debiasing via Bolukbasi et al. 2016 ("Man is to Computer Programmer as Woman is to Homemaker", NeurIPS) — https://web.stanford.edu/~jurafsky/slp3/5.pdf — **note: the current draft's Exercises heading is empty (no posed problems)**, so this source contributes narrative/citation backing only, not a ready-made problem.
- Caliskan, Bryson & Narayanan 2017, "Semantics derived automatically from language corpora contain human-like biases" — the WEAT methodology itself (association-strength test between target/attribute word sets), a natural upgrade over eyeballing `most_similar` output.

**Proposed problem set** (6 problems, our reference format):
1. [short-code] **fastText vs. GloVe similarity.** Load `TokenEmbedding('wiki.en')` alongside the section's `glove_6b50d`, run `get_similar_tokens` on the same 3 query words for both, and report where the top-3 neighbor sets diverge and by how much (count of non-overlapping words). Deliverable: a small table of the two neighbor lists plus the overlap count.
   *Provenance:* adapted from CS224N Assignment 1 (overlap med; same "compare embedding sources" spirit, different concrete task).
1. [conceptual] **Polysemy stress test.** Pick a word with at least two unrelated senses (e.g., "bank", "spring") and show that GloVe's single vector for it returns neighbors dominated by only one sense; explain, in terms of how GloVe averages over contexts, why the minority sense's neighbors are pushed out of the top-10.
   *Provenance:* adapted from CS224N Assignment 1 Q2.2 (overlap high; cite on adoption).
1. [short-code] **Analogy success/failure catalog.** Using `get_analogy`, evaluate 5 analogies you construct yourself (state each as $a:b::c:d$) plus the 2 already in the section; report which succeed and, for at least one failure, inspect the runner-up candidates via `knn` with $k=5$ to show how close the correct answer came.
   *Provenance:* adapted from CS224N Assignment 1 Q2.4–2.6 (overlap high; cite on adoption).
1. [conceptual] **Gender-bias probe.** Compute `get_analogy('man', 'doctor', 'woman', glove_6b50d)` and its occupation-swapped counterpart for 3 occupation words of your choice; summarize the pattern and state, in one sentence each, why this is a property of the training corpus rather than of the algorithm.
   *Provenance:* adapted from CS224N Assignment 1 Q2.7–2.8 and SLP3 ch.5's citation of Caliskan et al. 2017 (overlap med; cite on adoption).
1. [conceptual] **Debiasing critique.** Read the summary of Bolukbasi et al.'s hard-debiasing method (project out the gender direction while preserving definitional gender pairs); explain one scenario in this section's own examples (e.g., "king"/"queen" vs. "doctor"/"nurse") where blind debiasing would remove real-world signal along with the bias.
   *Provenance:* inspired by Bolukbasi et al. 2016 as cited in SLP3 ch.5 (overlap low).
1. [extended] **Intrinsic evaluation report.** Load all three GloVe sizes offered in this section (50d, 100d, 300d) plus fastText; build a small intrinsic-evaluation harness that runs a handful of hand-picked similarity pairs and analogy quadruples across all four embeddings, and report a table of qualitative agreement/disagreement (this section has no labeled benchmark like WordSim-353 available locally, so the "evaluation set" is the reader's own curated list). Discuss which embedding size changes the most results and why.
   *Provenance:* inspired by SLP3 ch.5 §5.9's survey of intrinsic evaluation sets (WordSim-353, SimLex-999, TOEFL) (overlap low).

---

## chapter_natural-language-processing-pretraining/bert.md — Bidirectional Encoder Representations from Transformers (BERT)

**Topic:** BERT's architecture — a Transformer encoder consuming summed token/segment/position embeddings, with MLM and NSP heads bolted on top — motivated by contrast with ELMo and GPT.
**Current exercises:** 2; disposition: keep 1, rewrite 1 — ex 1 (MLM vs. left-to-right LM convergence speed) is a clean, well-posed conceptual question worth keeping as-is; ex 2 ("research the difference between GELU and ReLU") names no deliverable (proof? plot? experiment?) and is flagged in the prior style review.

**External sources found:**
- BERT paper (Devlin, Chang, Lee & Toutanova, 2019, NAACL-HLT), §5 "Ablation Studies" — compares BERT against a "No NSP" variant and an "LTR & No NSP" (GPT-like) variant, and separately ablates model size (#layers/#hidden/#heads) — https://aclanthology.org/N19-1423.pdf
- CS224N (Spring 2024) Default Final Project, "minBERT and Downstream Tasks" — Part 1 has students implement multi-head self-attention and the Transformer encoder layer inside a minimal BERT (`bert.py`), sanity-checked against reference activations, then run the pretrained weights through a `[CLS]`-based sentiment classifier on SST/CFIMDB — https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/project/default-final-project-handout-minbert-spr2024-updated.pdf — explicitly credited on p.2 as adapted from the CMU 11-711 "minbert-assignment" (Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, Graham Neubig).
- CMU 11-711 Advanced NLP, `minbert-assignment` (github.com/neubig/minbert-assignment) — the original of the above: implement BERT's attention/encoder blocks from scratch under a no-`transformers`-library constraint, then evaluate on sentence classification.
- SLP3 ch. 10 "Masked Language Models", §10.4 — diagrams the `[CLS]`-based sequence-classification and sequence-pair-classification fine-tuning recipes built directly on top of a bidirectional encoder like the one this section defines — https://web.stanford.edu/~jurafsky/slp3/10.pdf — **note: no populated Exercises section in the current draft** (prose/diagrams only).

**Proposed problem set** (6 problems, our reference format):
1. [conceptual] **MLM convergence speed.** (Unchanged from the book.) All else equal, will a masked language model need more or fewer pretraining steps to converge than a left-to-right language model? Justify your answer in terms of what fraction of tokens contributes gradient signal per forward pass.
   *Provenance:* original (kept from the book).
1. [short-code] **Attention block from scratch.** Reimplement the multi-head self-attention computation used inside `d2l.TransformerEncoderBlock` (query/key/value projections, scaled dot-product, head concatenation) in a standalone function, and unit-test it by feeding the same `tokens`/`segments` batch from this section through both your version and `BERTEncoder` and checking that output shapes and values agree.
   *Provenance:* adapted from the CS224N minBERT project / CMU 11-711 `minbert-assignment` (overlap med; cite on adoption — the "implement attention, sanity-check against reference" task pattern is theirs, but the reference implementation here is d2l's own).
1. [conceptual] **GELU vs. ReLU, with a deliverable.** Plot GELU and ReLU (and their derivatives) over $x \in [-3,3]$; identify the one qualitative property GELU has that ReLU lacks (non-monotonicity / smoothness near 0), and state in one sentence why a smooth activation might matter more for a stack of 12+ Transformer blocks than for a single hidden layer.
   *Provenance:* adapted from the existing exercise + Hendrycks & Gimpel 2016 (already cited in-text) (overlap high; cite on adoption).
1. [conceptual] **Predict the NSP ablation.** Without running anything, predict how removing the next-sentence-prediction task (train MLM-only) would change performance on a single-sentence task (e.g., sentiment analysis) versus a sentence-pair task (e.g., natural language inference); state which BERT paper ablation (§5.1, "No NSP") this corresponds to and whether your prediction matches its direction.
   *Provenance:* inspired by BERT paper §5.1 ablation studies (overlap med; cite on adoption).
1. [short-code] **Segment-embedding ablation.** Modify `BERTEncoder` to drop the segment embedding (feed only token + position embeddings) and rerun the shape-check forward pass from this section with a two-segment input; report whether the encoder still runs, and explain what information about the input the model has now lost.
   *Provenance:* inspired by the BERT paper's input-representation design (token + segment + position) (overlap low).
1. [extended] **BERT input pipeline from scratch.** Reimplement `get_tokens_and_segments`, the embedding sum, and one Transformer encoder block without calling `d2l.TransformerEncoderBlock`, wiring your own attention (problem 2) and a feed-forward sublayer together with residual connections and layer norm; verify numerically against `d2l.BERTEncoder` on the section's own example inputs (vocab_size=10000, 2 layers).
   *Provenance:* adapted from the CS224N minBERT project's Part 1 scope (implement the encoder end-to-end) (overlap med; cite on adoption).

---

## chapter_natural-language-processing-pretraining/bert-dataset.md — The Dataset for Pretraining BERT

**Topic:** Turning WikiText-2 into BERT pretraining examples — NSP sentence-pair sampling, 80/10/10 masked-token generation, and padding into the seven tensors a training loop consumes.
**Current exercises:** 2; disposition: keep 1, rewrite 1 — ex 2 (vocab size with no frequency cutoff) is concrete and worth keeping; ex 1 hands the reader the complete solution (exact `pip install`/`import`/call sequence per the prior style review) and needs a stated comparison metric to become a real exercise rather than a copy-paste step.

**External sources found:**
- Hugging Face LLM Course, ch. 7 §3 "Fine-tuning a Masked Language Model" — walks through whole-word masking as an upgrade over token-level random masking, and tracks perplexity (exp of cross-entropy) before vs. after training (21.75 → 11.32 in their DistilBERT/IMDb example) as the deliverable metric — https://huggingface.co/learn/llm-course/en/chapter7/3
- BERT paper (Devlin et al. 2019) — the original 80/10/10 masking recipe and 30K WordPiece vocabulary that this section's `_replace_mlm_tokens`/`_WikiTextDataset` reimplement at smaller scale.
- RoBERTa paper (Liu, Ott, Goyal et al. 2019, arXiv:1907.11692; cited in SLP3 ch.10's bibliography) — motivates dropping NSP and using dynamic (per-epoch) masking instead of the static masking this section performs once at dataset-construction time.
- **Finding:** no course or textbook in our source list poses an exercise about the *dataset-construction* mechanics specifically (NSP negative sampling, masking-ratio choice) — HF's course exercises masking at the fine-tuning/collator level (`DataCollatorForLanguageModeling`), not the from-scratch `_get_next_sentence`/`_replace_mlm_tokens` level this section implements. The problems below are correspondingly more original than adapted.

**Proposed problem set** (6 problems, our reference format):
1. [short-code] **Sentence-splitter comparison, with a metric.** (Rewrite of existing ex 1.) Reimplement `_read_wiki`'s sentence splitting using `nltk.tokenize.sent_tokenize` instead of the period-delimiter split, and report the number of paragraphs whose sentence count changes and the total sentence-count delta across the first 1,000 paragraphs.
   *Provenance:* adapted from the existing exercise, sharpened with a required metric; task pattern also appears in HF Course ch.7 §3's tokenization comparisons (overlap high; cite on adoption).
1. [conceptual] **Vocabulary size without frequency cutoff.** (Unchanged from the book.) What is the vocabulary size if `min_freq=5` is removed? Report the number and explain, in terms of Zipf's law, why most of the added tokens are hapax legomena unlikely to receive a useful embedding.
   *Provenance:* original (kept from the book).
1. [short-code] **Whole-word masking.** Modify `_replace_mlm_tokens`/`_get_mlm_data_from_tokens` so that when a WordPiece-style multi-token word is selected for masking, all of its subword pieces are masked together rather than independently; run it on 10 sample sentences and show one example where whole-word masking changes which tokens get masked versus the original per-token scheme.
   *Provenance:* adapted from HF Course ch.7 §3's whole-word-masking exercise (overlap high; cite on adoption).
1. [conceptual] **Masking-ratio trade-off.** BERT masks 15% of tokens. Explain what would go wrong at the extremes — too few masked tokens per sequence (slow training signal) and too many (too little unmasked context to predict from) — and state, without running code, whether you'd expect a 40%-masking run's MLM loss curve (as plotted by this section's `Animator`) to start higher or lower than the book's 15% run.
   *Provenance:* inspired by the BERT paper's 15% choice and RoBERTa's re-examination of masking design (overlap low).
1. [short-code] **NSP negative-sampling bias check.** For 200 examples generated by `_get_nsp_data_from_paragraph`, compare the true-next-sentence pairs (`is_next=True`) against the randomly-sampled pairs (`is_next=False`) on two cheap proxies — sentence-length difference and token-vocabulary overlap — and report whether the random negatives are trivially distinguishable from true pairs on either proxy.
   *Provenance:* original, motivated by the same NSP-negatives critique that led RoBERTa to drop NSP entirely (overlap low).
1. [conceptual] **Padding waste.** Using this section's own WikiText-2 examples, estimate the fraction of `<pad>` tokens in a batch at `max_len=64` versus what it would be at the original BERT's `max_len=512`; explain why the original BERT paper trained most steps at a shorter sequence length before a final phase at 512.
   *Provenance:* original, motivated by the BERT paper's two-phase (128-then-512) pretraining schedule (overlap low).

---

## chapter_natural-language-processing-pretraining/bert-pretraining.md — Pretraining BERT

**Topic:** The end-to-end pretraining loop for a small BERT on WikiText-2 — combined MLM+NSP loss, the `train_bert` training loop, and a qualitative demonstration that the resulting encoder gives context-sensitive representations (the "crane" example).
**Current exercises:** 2; disposition: keep 2, rewrite 0, drop 0 — both are already well-specified per the prior style review (ex 1 poses a precise "why" comparison, ex 2 gives concrete config values and a well-posed observe-and-explain question); this is the strongest existing set in the group, so external material is added on top rather than replacing anything.

**External sources found:**
- BERT paper (Devlin et al. 2019), §5.1 "Effect of Pre-training Tasks" — the "No NSP" ablation, directly extendable from this section's own combined-loss training loop.
- Hugging Face LLM Course, ch. 7 §3 — uses perplexity (exp of the MLM cross-entropy) as the headline before/after-training metric, which this section's `Animator`-based loss plot doesn't currently surface numerically — https://huggingface.co/learn/llm-course/en/chapter7/3
- CS224N (Spring 2024) Default Final Project, "minBERT and Downstream Tasks", §1.2/§4 — the natural next step after this section: load the pretrained encoder, attach a `[CLS]`-based classification head, and fine-tune on SST/CFIMDB sentiment — https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/project/default-final-project-handout-minbert-spr2024-updated.pdf
- SLP3 ch. 10, §10.4.1 "Sequence Classification" — the `[CLS]` + classification-head recipe diagrammed abstractly; useful as the written spec for the extended problem below (no populated exercises in the current draft, prose only).

**Proposed problem set** (6 problems, our reference format):
1. [conceptual] **MLM loss vs. NSP loss.** (Unchanged from the book.) Why is the masked-language-modeling loss significantly higher than the next-sentence-prediction loss in the training run above? Answer in terms of the size of each task's output space (vocabulary-size softmax vs. binary classification).
   *Provenance:* original (kept from the book).
1. [short-code] **Scaling to BERT-Large config.** (Unchanged from the book.) Set `max_len=512` and use BERT-Large-scale hyperparameters (24 layers, 1024 hidden, 16 heads). Do you hit an error running this section's code on your hardware? Report what fails (OOM, shape mismatch, or neither) and why.
   *Provenance:* original (kept from the book).
1. [short-code] **Perplexity tracking.** Add a perplexity metric (exponential of the per-token MLM cross-entropy, restricted to masked positions) to the `_get_batch_loss_bert`/`train_bert` loop, and report its value at step 1 and step 50 of this section's smoke-test run.
   *Provenance:* adapted from HF Course ch.7 §3's before/after perplexity metric (overlap med; cite on adoption).
1. [short-code] **Quantifying context-sensitivity.** Extend the section's qualitative "crane" example into a number: compute the cosine similarity between the two `encoded_text_crane` vectors (from "a crane is flying" vs. "a crane driver came") at pretraining step 0 (randomly initialized encoder) and step 50 (after training), and report whether the similarity moves in the direction you'd expect as the model learns to use context.
   *Provenance:* original, extending the section's own qualitative demonstration into a checkable deliverable.
1. [conceptual] **Predict the No-NSP ablation.** Without retraining, predict what would happen to this section's "crane driver came" / "he just left" sentence-pair demonstration if the model had been pretrained with the NSP loss term removed (MLM-only); name which BERT-paper ablation (§5.1, "No NSP") this corresponds to.
   *Provenance:* inspired by BERT paper §5.1 (overlap med; cite on adoption).
1. [extended] **Fine-tune a sentiment head.** Attach a linear classification head on top of this section's pretrained `net`'s `[CLS]` output (following the `get_bert_encoding` pattern already in this section), fine-tune it on a small labeled sentiment sample (e.g., a few hundred IMDb or SST-style sentences), and report accuracy against a baseline head trained on top of the *un*pretrained (step-0) encoder — showing quantitatively what the 50 pretraining steps bought you.
   *Provenance:* adapted from the CS224N Default Final Project's sentiment-classification-via-`[CLS]` task and SLP3 §10.4.1's sequence-classification recipe (overlap med; cite on adoption).
