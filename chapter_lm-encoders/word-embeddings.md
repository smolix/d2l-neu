# Word Embeddings
:label:`sec_lm_word_embeddings`

<!-- Author brief, copied verbatim from the Language Models part plan
(docs/llm/language_models_part_plan.md, "2.1 Word Embeddings · T1").
Blueprint: docs/llm/blueprints/ch02_encoders.md section 2.1.
The plan is a specification, not book text; this comment is replaced
when the section is written.

**Idea:** A word vector is a factorization of co-occurrence statistics.
**Opening:** The reader has seen the token-embedding table of `sec_gpt` trained end to end and never asked what its rows mean. Two words that share contexts ("physician" and "doctor") sit near each other in that table. The section builds the simplest model whose rows are explained by counts, verifies the explanation numerically, and closes on the one thing a static vector cannot do: separate two meanings of one word.
**Teaches:**
- Skip-gram with negative sampling: the loss over center and context scores with K noise words, the 0.75-power noise distribution, subsampling of frequent words; the labels `subsec_negative-sampling` and `sec_word2vec_data` are defined here because `sec_mdl-mi-estimation` cites them.
- The optimum as shifted pointwise mutual information, stated and checked numerically (the derivation is the exercise in `sec_mdl-mi-estimation`); truncated SVD of the positive PMI matrix as the closed-form counterpart; GloVe as weighted least squares on log co-occurrences, in one sentence with its one difference.
- Analogies as a cosine query with the inputs excluded, and their caveats; bias as a description of the corpus, and why projection hides rather than removes it; tied input and output tables.
- Static vectors against the input table of a language model: one point per word type, and what a contextual encoder adds (2.2).
- Closing paragraph `subsec_lm_categorical_embeddings`: an embedding table for any categorical variable, the target of a pointer `sec_pandas` may add.
**Derivation:** none (the shifted-PMI result is stated; its proof is `ex_mdl-mutual-information-negative-sampling-pmi`). The numerical check: on a 4-word toy corpus, compute the count matrix, the PMI shifted by log K, and the SGNS scores after convergence, and show they agree.
**Notebook:**
1. Build a word-level slice of WikiText-103 from the download `d2l.WikiText103` uses (whitespace tokens, minimum count 10), with `d2l.Vocab`; count co-occurrences in a window of 5 as a sparse matrix.
2. Train SGNS with 5 negatives for 5 epochs (3 seeds), compute SVD of positive PMI at dimension 100, and train GloVe with the same window; evaluate Spearman correlation with human similarity judgments on the covered pairs of two similarity sets (coverage printed).
3. Verify the shifted-PMI relation on the toy corpus of the derivation and, approximately, on the slice (scatter of SGNS scores against shifted PMI for frequent pairs).
4. Inspect nearest neighbours of a few words in Qwen3-0.6B's token-embedding table (loaded through `d2l.load_qwen3`) and in the SGNS table; show one word with two meanings sharing one vector.
5. Claim that survives noise: all three methods correlate with human judgments alike (within the pair bootstrap and the seed range), and the choice of noise exponent and subsampling moves the correlation by as much as the choice of method. No ordering is asserted.
6. Reused pieces: `d2l.Vocab`, `d2l.WikiText103`'s download, `d2l.load_qwen3`; new `d2l.sgns_loss`, `d2l.ppmi`, `d2l.cooccurrence_counts`, `d2l.spearman_eval`, `d2l.nearest_neighbors`.
**Figures:** `fig_lm_sgns_pmi` (the window, the center and context tables, the noise draws, and the PMI matrix that the optimum factorizes).
**Restate / forward:** `subsec_mdl-pmi` (PMI measures how far a pair's co-occurrence exceeds independence); `sec_mdl-svd-low-rank` (the truncated SVD is the best low-rank approximation in Frobenius norm, `subsec_mdl-eckart-young`). Forward: 2.2 (one vector per token in context).
**Exercises:** 5 to 7. Load-bearing: derive the effect of subsampling with threshold t on the expected count of a word of frequency f and on the PMI of pairs involving it (derive); measure analogy accuracy against dimension 25 to 300 with and without excluding the query words (measure); check numerically that the SGNS optimum with free scores on the toy corpus equals PMI minus log K (check); open: fit a 32-dimensional embedding table for a categorical column of a tabular dataset and compare with one-hot features under the same linear model.
**Writer notes:**
- The corpus is a word-level WikiText-103 slice, not Penn Treebank; the exercise owner is `sec_mdl-mi-estimation`, not `sec_mdl-mutual-information`; the closing paragraph `subsec_lm_categorical_embeddings` belongs to this section. `d2l.WikiText103` (defined in `chapter_optimization/adam.md`) yields byte-level data; 2.1 reads the same downloaded parquet as words. Cap the slice (a few million words) so SGNS trains in minutes on CPU; print the vocabulary size and the coverage of the similarity pairs, which is far better than PTB's (the blueprint measured 217 of 353 WordSim pairs covered under PTB's 6,719-type vocabulary).
- Legacy names: `d2l.DATA_HUB['ptb']` moves to `lstm.md`; the retiring `read_ptb`, `load_data_ptb`, `get_negatives`, `batchify` helpers are not kept unless an exercise needs them; `d2l.TokenEmbedding` with the `glove.6b.50d` and `glove.6b.100d` keys stays for the analogy and bias cells; state dropped names in the migration note.
- The two labels: define `subsec_negative-sampling` on the `##` subsection that states the SGNS loss with K noise words and center and context scores, and `sec_word2vec_data` on the subsection that states the count^0.75 noise distribution, so `mdl-mi-estimation.md:1230-1247` resolves unchanged.
- Pitfalls to state: evaluating the center table, the context table or their sum changes the numbers (state the choice); padded context positions must be masked out of the loss; subsampling per occurrence, not per type; SGNS "is" PMI factorization only at the optimum with free scores; pairs outside the vocabulary silently shrink the test set.
- Data: `almogtavor/WordSim353` and `Yuti/SimLex-999` (both CC BY 4.0) downloaded at run time; pretrained GloVe-6B through the book's mirror.
- Do not re-teach: PMI (`subsec_mdl-pmi`), the SVD (`sec_mdl-svd-low-rank`), the shifted-PMI proof (`sec_mdl-mi-estimation`).
- Committed artifacts: the similarity result table as JSON for the freshness check; nothing else (T0-capable).
- Optional: the preliminaries chapter may add a one-line pointer in `sec_pandas` to `subsec_lm_categorical_embeddings` (nothing breaks either way).
**Sources:**
- Papers: [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) (skip-gram and CBOW); [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) (negative sampling, subsampling, the 3/4 power); Levy and Goldberg, "Neural Word Embedding as Implicit Matrix Factorization" (NeurIPS 2014; cited by key `Levy.Goldberg.2014`: the shifted-PMI optimum); [GloVe: Global Vectors for Word Representation](https://aclanthology.org/D14-1162/) (weighted least squares on log counts); [Improving Distributional Similarity with Lessons Learned from Word Embeddings](https://aclanthology.org/Q15-1016/) (hyperparameters matter as much as the method: the notebook's thesis); [Man is to Computer Programmer as Woman is to Homemaker?](https://arxiv.org/abs/1607.06520) (bias in embeddings); [Entity Embeddings of Categorical Variables](https://arxiv.org/abs/1604.06737) (the closing paragraph).
- Code and documentation: the [WikiText dataset card](https://huggingface.co/datasets/Salesforce/wikitext) (the `wikitext-103-raw-v1` parquet files that `d2l.WikiText103` downloads and 2.1 reads as words); the Qwen3-0.6B card for the tied table ([Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)).
- Courses, tutorials and blogs: [CS224N](https://web.stanford.edu/class/cs224n/) (word-vector lectures and assignment A1); Jay Alammar, [The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/).
-->
