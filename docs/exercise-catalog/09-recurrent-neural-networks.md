# Chapter Overview — chapter_recurrent-neural-networks

> **Status (2026-11 book pass).** The book pass rewrote this chapter's exercises.
> Each section now carries 6–8 exercises; most of the problems proposed below
> were adopted, several were merged or sharpened, and new ones were added
> (8.2 encoding by rank and rank replay; 8.3 overlapping windows and seen
> versus unseen trigrams; 8.4 the input product, exposure bias, a window inside
> a recurrence, and the embedding lookup; 8.5 scratch-versus-concise timing and
> the bits-per-byte factorization; 8.6 the backward pass, split into a
> derivation and an implementation; 8.7 truncation under temperature and
> constant cost per token). 50 exercises in all.
> Each section below lists its **current exercises** (one line each, markers as
> in the source: [code], [extended], ● hard, ●● very hard); the overview
> bullets, the dispositions, and the external sources describe the pre-pass
> audit and stay as provenance. The source `.md` files are authoritative.

- 7 exercise-bearing sections, 40 existing exercises total (per prior style review); disposition below is overwhelmingly **keep** — this chapter's sets are the cleanest in the book (near-zero clarity defects chapter-wide).
- Best verified source by far: Jurafsky & Martin, *Speech and Language Processing* 3rd-ed draft, **Ch. 3 (N-gram Language Models)** — 12 exercises (3.1–3.12), directly reusable for `language-model.md`.
- Surprising, repeated finding (confirmed by full-text scan, not spot-check): SLP3's own **Ch. 13 (RNNs and LSTMs)** and **Ch. 7 (Large Language Models)** contain **zero** end-of-chapter exercises, and Goldberg's *Neural Network Methods for NLP* has **zero** formal exercises anywhere in the book. None of the three textbooks this chapter most resembles has a problem-set tradition for RNNs/LMs at all.
- Every implementation-flavored exercise we verified for the RNN sections came from a hands-on course or tutorial, never a textbook: CMU 11-785 HW3 ("derive the 14 gradients of an RNN-cell backward pass by hand") and MIT 6.S191 Lab 1 (a data→model→loss→generate TODO progression) closely parallel `bptt.md` and `rnn-implementation.md` respectively.
- SLP3 Ch. 2 covers BPE at length in its prose but its own 7 exercises are all regex-writing/edit-distance — a clean non-overlap finding for `text-sequence.md`. Karpathy's `minbpe` exercise progression is the much closer match there.
- CS224n's A4 (NMT with RNNs) supplied a verified, adoptable error-taxonomy and hand-BLEU exercise for `decoding.md` — but despite being the course most associated with beam search in NMT, its written questions never ask students to reason about beam search itself, unlike this section's own ex. 2–4.
- `sequence.md` (autoregression/Markov/rollout framing) and `rnn.md` (pure pencil-and-paper, no code) both have **no verified external exercise tradition**: every course/text we checked pairs the RNN concept with code immediately, or frames sequences purely as n-grams/LMs rather than this section's stats-flavored autoregression setup.
- Totals below: 40 existing exercises audited (39 keep, 1 rewrite, 0 drop), 47 problems proposed across the 7 sections — 6 new problems carry verified "adapted" provenance (citation owed on adoption: SLP3 ×2, CMU 11-785, MIT 6.S191, CS224n A4, Karpathy `minbpe`), the remaining 41 are original (including regroupings and one rewritten-for-tone item).

---

## chapter_recurrent-neural-networks/sequence.md — Working with Sequences

**Topic (2026-11):** Autoregressive modeling with a fixed window of $m$ lags or a latent state $\mathbf{h}_t = g(\mathbf{h}_{t-1}, x_t)$; the chain rule; Markov condition and stationarity; a linear $m = 4$ autoregressor on a noisy sine wave whose rollout averages the last four values, so its $k$-step forecast is a delayed, damped copy of the series.

**Current exercises (2026-11 book pass):** 7

1. [code] **Window length.** Increase `m` until validation error stops improving; without noise, how many past observations suffice for exact prediction, and why.
1. **Momentum strategies.** State the assumption, in Markov and stationarity terms, that picking stocks by their past returns requires, and one mechanism by which it fails.
1. [code] **Nonlinear autoregressor.** Replace the linear model by a small MLP on the same four lags; compare one-step and multistep predictions.
1. [code] **Error growth with the horizon.** Plot the mean squared $k$-step error for $k = 1, 2, 4, \ldots, 256$ on a log axis, relate its bend to the damped 16- and 64-step forecasts, and find the horizon at which it exceeds the variance of the series.
1. [code] **Dilated and averaged lags.** ● Four features that reach further back (lags 1, 2, 4, 8, or four block means) against $m = 4$ and $m = 16$: which construction helps, and why.
1. [code] **Regime switching.** ● A series alternating between $\sin(0.01t)$ and $\sin(0.05t)$ every 200 steps: per-regime training and validation error of one $m = 4$ linear model, and where the fit is worst.
1. [code] **Scheduled sampling.** ● Replace each lag by the model's own prediction with probability $\epsilon \in \{0, 0.25, 0.5, 1\}$; does any $\epsilon > 0$ lower the 64-step error, and what happens to the one-step error. Cites Bengio et al. (2015).

**Pre-pass exercises:** 6 (ex. 1 has 4 lettered sub-items); disposition: keep 5, rewrite 1, drop 0 — the prior review found this set essentially defect-free and concrete throughout ("no outright underspecified exercises... otherwise carries real content"); the one exception is ex. 1.3's "Can you incorporate older observations...?" filler phrasing, worth a straight rewrite into a direct instruction.

**External sources found:**
- No good external exercise tradition located for this section's specific framing. We checked Jurafsky & Martin SLP3 Ch. 3 (N-grams) — https://web.stanford.edu/~jurafsky/slp3/3.pdf — which shares the chain-rule/Markov-condition idea in its prose (used centrally for `language-model.md` below) but poses no exercise on multistep rollout, regime-switching, or fixed-window-vs-latent-state comparison. General NLP/ML course searches for "autoregressive rollout error accumulation" homework turned up nothing verifiable at the right level: courses either jump straight to n-gram LMs (no continuous-signal autoregression) or straight to RNNs (no fixed-window baseline). This is a real finding, not a search failure: the section's stats/forecasting framing sits between what NLP courses and what time-series/econometrics courses each typically assign.

**Pre-pass proposed problem set** (provenance; superseded by the current exercises above):
1. [short-code] **Improve the lag model.** Extend this section's τ=4 linear autoregressor along four axes and report validation MSE for each: (a) widen τ and find the point of diminishing returns; (b) for the noiseless case, derive from the sine's ODE how many past observations suffice in principle; (c) keep 4 features but let them span a wider history (e.g. strided lags) — does accuracy improve, and why; (d) replace the linear model with a small MLP and retrain. Report one table with all four variants' validation MSE.
   *Provenance:* original
1. [short-code] **k-step error curve.** For k ∈ {1, 2, 4, 8, 16}, plot mean k-step-ahead prediction error against k on a log axis. Identify where the curve visibly bends and relate the bend to the qualitative decay seen in the 1/4/16/64-step plots earlier in the section.
   *Provenance:* original
1. [conceptual] **Forecast horizon vs. series variance.** Compute the horizon at which multistep rollout error first exceeds the variance of the series itself. Explain in 2-3 sentences what forecasting past that horizon actually buys you (if anything).
   *Provenance:* original
1. [short-code] **Regime-switching stress test.** Splice two sines of different frequency into one series and refit the fixed-τ linear model. Report training and validation MSE per regime separately, and identify concretely where and why a single fixed-τ model fails.
   *Provenance:* original
1. [conceptual] **Momentum-strategy critique.** An investor picks a stock by its own past returns. Using this section's autoregressive framing, state precisely what assumption about the return series such a strategy requires, and give one concrete scenario (in this section's own vocabulary — Markov order, stationarity) under which it fails.
   *Provenance:* original (rewritten from existing ex. 1.3/5 to remove "Can you...?" filler phrasing and demand a named assumption, not an open impression)
1. [conceptual] **The case for a latent state.** Give one concrete example of a sequence where a latent autoregressive model (fixed-size $h_t$) is clearly needed rather than any fixed-window model, and justify why no finite window suffices.
   *Provenance:* original
1. [short-code] **Scheduled sampling.** The section's own text claims that "exposing a model to its own predictions during training... narrows the gap" between one-step and multistep regimes, but never tests this. Retrain the τ=4 model feeding a mix of true and model-predicted lags during training (vary the mix fraction), and measure whether the multistep rollout error at k=64 falls relative to the section's baseline. Report the mix fraction that helps most, if any.
   *Provenance:* original

---

## chapter_recurrent-neural-networks/text-sequence.md — From Text to Tokens

**Topic (2026-11):** Characters, words, and bytes; byte pair encoding traced by hand, implemented (`BPETokenizer`: merge, train, rank-order encode, decode), trained on *The Time Machine* with GPT-2 pre-tokenization; `Vocab` and special tokens above the BPE range; GPT-2's rank table loaded into our encoder and checked against `tiktoken`; fertility, digit chunking, and glitch tokens.

**Current exercises (2026-11 book pass):** 7

1. **Encoding by rank.** For merges (`b`,`c`) then (`a`,`b`), segment "abc" under the rank rule and the leftmost rule; which one agrees with the end-of-training segmentation.
1. [code] **Cross-corpus compression.** Train a 1,024-token tokenizer with the GPT-2 pattern on another text; overlap of its first and last 100 merges with `bpe`; cross-evaluated bytes per token.
1. [code] **Special-token round trip.** Add `<sep>`, encode and re-encode with and without `allow_special`, and compare the raise-an-error and plain-bytes policies. *Adapted from Karpathy, minbpe exercise step 4.*
1. [code] **Digit chunking.** Token boundaries of three addition prompts and their sums under `gpt2` and `o200k_base`; place-value alignment under the three-digit cap; propose an aligning pre-tokenization rule and one drawback.
1. **Glitch tokens.** Why the merge rule makes one token of a frequent string and what pre-tokenization fixes; propose a test that flags glitch tokens from the embedding table and the training corpus.
1. **Rank replay.** ● Argue that the rank rule reproduces the end-of-training segmentation of every chunk of the training corpus.
1. [code] **Pre-tokenization ablation.** ● 4,096-token tokenizers with and without the pattern on 90% of the text: longest tokens, in-domain compression, and compression on the held-out 10%.

**Pre-pass exercises:** 5; disposition: keep 5, rewrite 0, drop 0 — the prior review rated this the cleanest, most concrete set in the chapter (every item names an exact deliverable: a compression ratio, a longest-token list, a proposal); nothing to fix.

**External sources found:**
- Karpathy, `minbpe` repo and its `exercise.md` progression (2024) — https://github.com/karpathy/minbpe, https://raw.githubusercontent.com/karpathy/minbpe/master/exercise.md — a verified 5-step build order: (1) `BasicTokenizer` with train/encode/decode; (2) add the GPT-4 regex pre-tokenizer; (3) load real GPT-4 merges and match `tiktoken` exactly (handling rank recovery and the internal byte permutation); (4) special-token handling; (5) stretch goal, non-byte schemes like SentencePiece. Very high overlap — this is essentially the book's own approach, and step 4 (special tokens) is a natural extension the book's own set doesn't yet exercise.
- Jurafsky & Martin, SLP3 3rd-ed draft, Ch. 2 "Words and Tokens" — https://web.stanford.edu/~jurafsky/slp3/2.pdf — verified by direct read: despite covering BPE at length in its prose (and citing Kudo & Richardson's SentencePiece, Bostrom & Durrett's "BPE is suboptimal," etc. in its references), its own 7 end-of-chapter exercises (2.1–2.7) are entirely regex-writing and minimum-edit-distance computation — none touch BPE, vocabulary size, or subword tokenization at all. A clean non-overlap finding in an otherwise-canonical source.

**Pre-pass proposed problem set** (provenance; superseded by the current exercises above):
1. [short-code] **Cross-corpus compression.** Train `BPETokenizer` on a different text of your choice; inspect its first/last 20 merges; measure bytes-per-token on both *The Time Machine* and your corpus, cross-evaluated, and compare to the in-domain numbers.
   *Provenance:* original
1. [short-code] **Vocabulary-size ablation via bits-per-byte.** Retrain at vocab sizes 512/2,048/4,096; after training the language model of the next section on each, explain why per-token perplexity is the wrong cross-vocab-size metric and report the correct one instead.
   *Provenance:* original
1. [short-code] **Pre-tokenization ablation.** Retrain a 4,096-vocab tokenizer without the GPT-2 regex pattern; list the 20 longest learned tokens of both runs and identify which are compression wins in-domain but liabilities out-of-domain.
   *Provenance:* original
1. [conceptual] **Glitch tokens, explained.** Explain concretely what property a tokenizer's training corpus must have relative to a model's training corpus, and why BPE's frequency-based merge rule turns that mismatch into a single (near-randomly-embedded) token rather than several.
   *Provenance:* original
1. [short-code] **Digit chunking and place value.** Tokenize "123+456=" and "1234+5678=" with `gpt2` and `o200k_base`; show which digit positions of the summands land in the same token; propose a pre-tokenization rule that aligns tokens to place value and state one drawback.
   *Provenance:* original
1. [short-code] **Special-token round trip.** Add a reserved special token (e.g. an end-of-document marker) to your trained vocabulary, verify no sequence of merges on ordinary text can ever produce it, and confirm encode→decode preserves it exactly across a document boundary.
   *Provenance:* adapted from Karpathy `minbpe` exercise.md, step 4 (overlap medium — same feature, applied to this section's own tokenizer rather than a from-scratch GPT-4 clone)

---

## chapter_recurrent-neural-networks/language-model.md — Language Models

**Topic (2026-11):** The chain rule as next-token prediction; word-level $n$-gram tables (`NGramLM`) on *The Time Machine*, sampling and verbatim copying, count sparsity and Laplace smoothing; Zipf's law for words, pairs, and triples; perplexity and bits per byte (a trigram under character, word, and BPE tokenizations); overlapping shift-by-one windows of BPE ids (`d2l.TimeMachine`).

**Current exercises (2026-11 book pass):** 8

1. [code] **Temperature sampling.** Sample successors in proportion to $N^{1/\tau}$ at $\tau \in \{0.3, 1, 3\}$; the $\tau \to 0$ and $\tau \to \infty$ limits, and the effect on contexts seen once.
1. **Dense versus sparse storage.** A dense 5-gram table over 50,000 words, the memory and coverage of storing only a trillion-token corpus's 5-grams, compared with the measured trigram coverage.
1. [code] **Smoothing constant.** Sweep $\alpha = 10^{k/4}$, $k = -16, \ldots, 2$, for the three word models; how the best $\alpha$ moves with $n$, and the ranking at each model's best $\alpha$.
1. **Perplexity under skew.** Unigram perplexity of a 91%/1% ten-symbol corpus; read the word unigram's held-out perplexity as a number of choices and explain it with Zipf's law. *Adapted from SLP3 ex. 3.12.*
1. **Bits per byte from perplexity.** Show $\textrm{bpb} = (T/B)\log_2 \textrm{PPL}$, recover bytes per token from the printed table, and find the perplexity the BPE trigram needs for 1 bit per byte.
1. **Overlapping windows.** Corpus positions touched by the training and validation rows at `num_steps=32`, targets per token per epoch, and a comparison with non-overlapping windows.
1. [code] **Normalization across lengths.** ● Without an end token, a smoothed bigram's probabilities sum to one for each length separately; with one, all complete sequences sum to one. *Adapted from SLP3 ex. 3.5.*
1. [code] **Seen and unseen trigrams.** ● Average NLL of the bigram and the trigram on held-out positions whose trigram was or was not seen in training; what this suggests about combining orders.

**Pre-pass exercises:** 4; disposition: keep 4, rewrite 0, drop 0 — the prior review flagged no clarity defects; every exercise specifies exact parameters (temperatures, vocab size, a formula to verify, a smoothing sweep) and a concrete artifact.

**External sources found:**
- Jurafsky & Martin, SLP3 3rd-ed draft, Ch. 3 "N-gram Language Models," Exercises 3.1–3.12 — https://web.stanford.edu/~jurafsky/slp3/3.pdf — verified by direct read; the canonical exercise set for this exact topic. Particularly relevant: 3.2/3.3 (compute a sentence's probability unsmoothed vs. add-1-smoothed and explain which is higher and why), 3.5 (show that an unsmoothed bigram model *without* an end symbol assigns probability 1.0 to the sentences of every length separately — i.e., is not a valid distribution over sequences), 3.6 (derive the general add-1-smoothed trigram formula in terms of raw counts and vocabulary size V), 3.9–3.11 (build a small n-gram program, compare unigram/bigram statistics across two corpora, add random-sentence generation and perplexity), 3.12 (unigram perplexity of a test set drawn from a 91%-skewed 10-symbol training corpus).
- Georgetown University, COSC 572 (S23), Assignment 1 "N-Gram Language Models" — https://people.cs.georgetown.edu/nschneid/cosc572/s23/a1/ — verified by direct read; builds unigram/bigram/add-α-smoothed-bigram models on the Brown corpus and asks students to explain *why* smoothing shrinks high-frequency-context probabilities less than rare-context ones, and why that is desirable — a sharper framing of the "why" behind smoothing than a pure compute-the-number exercise.

**Pre-pass proposed problem set** (provenance; superseded by the current exercises above):
1. [short-code] **Temperature sampling from an n-gram.** Add temperature τ to `NGramLM.sample` (sample ∝ N^(1/τ)); generate trigram text at τ ∈ {0.3, 1, 3}; describe the τ→0 and τ→∞ limits.
   *Provenance:* original
1. [conceptual] **Dense vs. sparse 5-gram storage.** For a 5-gram model over |V|=50,000, compute the dense table size, then the memory of storing only observed 5-grams from a trillion-token corpus at 16 bytes/entry, and the resulting coverage fraction of all possible 5-grams.
   *Provenance:* original
1. [conceptual] **Bits-per-byte, derived and checked.** Show bpb = (T/B)·log₂(perplexity) for a tokenizer splitting a B-byte text into T tokens; verify it on this section's own perplexity table; find the perplexity needed to reach 1 bpb.
   *Provenance:* original
1. [short-code] **Smoothing-constant sweep.** Sweep add-α smoothing over α ∈ {1, 0.1, 0.01, 0.001, 0.0001} for unigram/bigram/trigram word models; plot held-out perplexity vs. α on a log axis; explain why the optimal α shrinks as n grows and whether any α makes the trigram beat the bigram.
   *Provenance:* original
1. [conceptual] **Smoothed n-gram distribution check.** Adapted from SLP3 ex. 3.5 (overlap high — same proof technique, applied to this section's own model): using a small enumerable toy vocabulary, sum your add-α-smoothed bigram model's probability over all sentences of a fixed length, for two different lengths. Does each sum to 1.0 individually? What would go wrong if you dropped the end-of-sequence token, as SLP3's example does?
   *Provenance:* adapted from SLP3 ex. 3.5 (overlap high; cite on adoption)
1. [short-code] **Perplexity under extreme skew.** Adapted from SLP3 ex. 3.12 (overlap high — same construction, applied to this section's own corpus/tokenizer): build a 10-symbol training set that is 91% one symbol, 1% each of nine others; compute unigram perplexity on an analogously skewed test set; compare the number to this section's own Time Machine unigram perplexity and explain the gap.
   *Provenance:* adapted from SLP3 ex. 3.12 (overlap high; cite on adoption)

---

## chapter_recurrent-neural-networks/rnn.md — Recurrent Neural Networks

**Topic (2026-11):** The neural $n$-gram and its window; the recurrence $\mathbf{H}_t = \phi(\mathbf{X}_t\mathbf{W}_\textrm{xh} + \mathbf{H}_{t-1}\mathbf{W}_\textrm{hh} + \mathbf{b}_\textrm{h})$, a scalar memory example, and the concatenated form checked numerically; constant memory per step; the RNN language model over BPE tokens, its parameter and cost table (222,915 parameters, 89% facing the vocabulary), and teacher forcing.

**Current exercises (2026-11 book pass):** 6

1. **Parameter counts.** Table size of an $(m+1)$-gram, parameters of the RNN language model and of the neural $n$-gram at $|\mathcal{V}| = 10{,}000$, $m = 20$, $h = d = 256$, and their growth in $m$.
1. **Separating the input product.** Which product can be computed for all $T$ steps in one multiplication, which must stay in the loop, and whether the advantage survives generation.
1. **Exposure bias.** Probability of 100 error-free sampled tokens at $\epsilon = 0.01$; which property of teacher forcing scheduled sampling gives up, and its cost.
1. **A window inside a recurrence.** Choose linear-RNN weights whose state of $md$ units stores the last $m$ embeddings; show $\mathbf{W}_\textrm{hh}^m = \mathbf{0}$; what a 256-unit state must do instead.
1. [code] **Embedding lookup as a product.** Indexing against a one-hot product, timed for $|\mathcal{V}| \in \{10^3, 10^4, 10^5\}$, and the sparsity of the embedding gradient.
1. **Parity beyond any window.** ● No window model computes the running parity; a two-unit threshold RNN does; how many bits the state must keep.

**Pre-pass exercises:** 5; disposition: keep 5, rewrite 0, drop 0 — the prior review called this "the shortest, cleanest file in the group," with every item concrete (a dimension question, a "why," gradient behavior, an explicit parameter-count derivation with numbers plugged in, and a reflection tied directly to the section's own stated limitations).

**External sources found:** No good external exercise tradition located for this section's specific framing — an explicit, verified finding, not a search gap:
- Jurafsky & Martin, SLP3 3rd-ed draft, Ch. 13 "RNNs and LSTMs" — https://web.stanford.edu/~jurafsky/slp3/13.pdf — full-text scan confirms **zero** occurrences of the word "exercise" anywhere in the chapter; no end-of-chapter problems exist at all, despite the chapter covering the RNN recurrence and language modeling with RNNs in its body.
- Goldberg, *Neural Network Methods for Natural Language Processing* (Morgan & Claypool / Springer) — https://link.springer.com/book/10.1007/978-3-031-02165-7 — full-text scan of the book confirms "exercise" occurs exactly twice in the entire text, both in passing prose ("as a simple exercise, the reader should..."), never as a numbered problem-set heading. The book has no formal exercises anywhere, RNN chapter included.
- CMU 11-785 HW3 (the course explicitly suggested as a source for this chapter) does cover the RNN recurrence's mechanics in real depth — but always bundled with a from-scratch coding deliverable (see `rnn-implementation.md` and `bptt.md` below), never as a standalone pencil-and-paper problem the way this section poses it.
- Conclusion: every source we checked pairs the RNN concept with an immediate implementation task; none offers this section's "just the recurrence, pencil-and-paper" framing as a discrete exercise.

**Pre-pass proposed problem set** (provenance; superseded by the current exercises above):
1. [conceptual] **Required output dimension.** If we use an RNN to predict the next token in a text sequence, what dimension must its output take?
   *Provenance:* original
1. [conceptual] **Full-history conditioning argument.** Explain why an RNN can express the conditional probability of a token given *all* previous tokens, not just a fixed window.
   *Provenance:* original
1. [conceptual] **Gradient behavior over long sequences.** What happens to the gradient when backpropagating through a long sequence? State the qualitative behavior you expect and why.
   *Provenance:* original
1. [conceptual] **n-gram table vs. RNN weights.** For a context of k previous tokens over a vocabulary of size |V|, write the parameter (or table-entry) count for a (k+1)-gram and for an RNN with h hidden units and d-dimensional embeddings; evaluate both at |V|=10,000, k=20, h=d=256. Which grows with k, and by how much?
   *Provenance:* original
1. [conceptual] **What this model still gets wrong.** Name a concrete limitation of the language model built in this section, and state which specific idea in the chapters that follow addresses it.
   *Provenance:* original
1. [conceptual] **A task no fixed n-gram can solve.** On paper, describe a synthetic sequence task (for instance: report the parity of the number of 1s seen so far in a binary string) that no k-th-order n-gram can solve exactly for *any* finite k, but that a small RNN can represent exactly regardless of sequence length. State the argument in 3-4 sentences.
   *Provenance:* original
1. [conceptual] **Weight sharing vs. one big classifier.** An RNN reuses the same $W_{xh}, W_{hh}$ at every step. Compare this to a naive alternative that concatenates all t-1 previous one-hot tokens into one fixed-size, zero-padded vector and trains a single feedforward classifier per output position. State the parameter-count difference and name the specific failure mode the RNN avoids for a test sequence longer than the training maximum.
   *Provenance:* original

---

## chapter_recurrent-neural-networks/rnn-implementation.md — Implementing RNNs

**Topic (2026-11):** An RNN language model on 65,741 BPE tokens of *The Time Machine*: `RNNScratch` cell and shape checks, embedding lookup in place of one-hot inputs, a shared output layer; gradient clipping as a projection; training (222,915 parameters, perplexity near 90, 2.3–2.4 bits per byte, about the BPE bigram); greedy and temperature generation; the concise `RNN`/`RNNLM` with matched initialization, and a timing comparison.

**Current exercises (2026-11 book pass):** 8

1. [code] **Gradient clipping ablation.** Plot the gradient norm before clipping over the first epoch; train unclipped at `lr=4`, then halve the learning rate until it trains; a ReLU cell with and without clipping.
1. [code] **Usable history.** Cross-entropy on 512 validation tokens with the state reset every 32 tokens against carried across all of them.
1. [code] **Hyperparameters and data budget.** Tune to beat the bigram's bits per byte and compare relative gains in bpb and perplexity; train on the default 10,000 windows for 100 epochs and count parameters per distinct training token.
1. **Weight tying savings.** Parameters saved by tying at $|\mathcal{V}| = 1{,}027$ and 128 hidden units, and the cost of untying GPT-2 small's $50{,}257 \times 768$ embedding.
1. [code] **Scratch versus concise timing.** The ratio `t_scratch / t_concise` over `num_hiddens` and `num_steps`, and at batch size 64: which of launch overhead, arithmetic, and compilation dominates.
1. [code] **Vocabulary size and bits per byte.** ● Vocabularies of 512 to 4,096 with an 80/20 split of the windows: the best bits per byte, and how parameter counts and corpus length explain the trend.
1. [code] **Weight tying.** ● Tie $\mathbf{E}^\top$ to the output projection at $d = h = 128$, restore a near-uniform initial prediction, and compare tied and untied validation perplexity.
1. [code] **Cross-book generalization.** ● Train on *The War of the Worlds*, evaluate on both novels, and factor the bits-per-byte ratio into a prediction factor and a compression factor.

**Pre-pass exercises:** 9; disposition: keep 9 (regrouped), rewrite 0, drop 0 — the prior review called this "the longest exercise list in the chapter... all uniformly bare/untagged/unnamed" yet flagged zero clarity defects (every item names a specific hyperparameter, dataset, or metric). Nothing here needs cutting; below we regroup two closely related pairs under nested sub-items to fit the catalog's 5–8-problem format without losing any content.

**External sources found:**
- CMU 11-785 (Introduction to Deep Learning) HW3 Part 1, Fall 2021/2022 writeups — https://deeplearning.cs.cmu.edu/F21/document/homework/HW3/HW3P1_writeup.pdf, https://deeplearning.cs.cmu.edu/F22/document/homework/HW3/HW3P1_F22.pdf — verified by direct read (via `pdftotext`, since the PDFs are not machine-summarizable as rendered): builds an RNN cell's forward *and* manual backward pass from scratch, explicitly enumerating "fourteen gradients" (∂L/∂W_hh, ∂L/∂h_{t-1}, etc.) that must be derived by hand and implemented without autodiff.
- MIT 6.S191, Lab 1 Part 2 "Music Generation with RNNs" (PyTorch notebook) — https://raw.githubusercontent.com/MITDeepLearning/introtodeeplearning/master/lab1/PT_Part2_Music_Generation.ipynb — verified by direct download and inspection of the notebook's own TODO cells: "Write a function to convert the all songs string to a vectorized... representation," "construct a list of input sequences for the training batch" (and the corresponding shifted-by-one target batch), "Add LSTM and Linear layers to define the RNN model," "define the compute_loss function," and "generate ABC format text of length 1000" via temperature-based sampling. Structurally this is the same data→model→loss→train→generate progression as this section, built on a different (music) corpus.
- Karpathy, "The Unreasonable Effectiveness of Recurrent Neural Networks" — http://karpathy.github.io/2015/05/21/rnn-effectiveness/ — verified by direct fetch: demonstrates the same char-level RNN trained separately on Paul Graham essays, Shakespeare, Wikipedia markdown, LaTeX (algebraic geometry), and Linux kernel C source, and states the temperature effect directly ("Decreasing the temperature... makes the RNN more confident, but also more conservative... higher temperatures will give more diversity but at cost of more mistakes").

**Pre-pass proposed problem set** (provenance; superseded by the current exercises above):
1. [short-code] **Effective context window.** Does the trained model ever condition on tokens further back than the start of its current window? Which hyperparameter bounds the usable history length?
   *Provenance:* original
1. [short-code] **Hyperparameter sweep for validation perplexity.**
    1. Adjust epochs, hidden units, embedding dimension, `num_steps`, and learning rate to minimize validation perplexity. Does bits-per-byte improve by the same factor?
    1. Restore the pipeline defaults (`num_train=10000`) and train for 100 epochs; compare train/validation perplexity curves and report parameters-per-training-token in this regime.
   *Provenance:* original
1. [short-code] **Gradient clipping ablation.**
    1. Run training without gradient clipping. What happens?
    1. Replace $\tanh$ with ReLU in the recurrent cell and repeat. Do you still need clipping? Why?
   *Provenance:* original
1. [short-code] **Vocabulary size vs. bits-per-byte.** Rebuild the dataset with a 2,048-token tokenizer and retrain; compare against this section's model using bits-per-byte, and explain why comparing validation perplexities directly would mislead here.
   *Provenance:* original
1. [short-code] **Weight tying.** Set the embedding dimension equal to the hidden size and tie $\mathbf{W}_\textrm{e}^\top$ to the output projection instead of a separate $\mathbf{W}_{\textrm{hq}}$. Report the parameter savings and the resulting perplexity change.
   *Provenance:* original
1. [conceptual] **Temperature sampling, derived.** Show that sampling at temperature T draws token x with probability proportional to $P(x)^{1/T}$; state what the $T\to 0$ and $T\to\infty$ limits recover.
   *Provenance:* original
1. [short-code] **Cross-book generalization.** Train on a different H. G. Wells novel (e.g. *The War of the Worlds*) and evaluate on *The Time Machine*; report perplexity and bits-per-byte, and what the gap versus in-book validation tells you.
   *Provenance:* original
1. [short-code] **Unit-test the data pipeline.** Write a small assertion-based test that checks, for several random windows drawn from `get_dataloader`, that each target sequence equals its input sequence shifted by exactly one token position. Run it against your own implementation. What kind of bug would this test catch if the shift were off by one?
   *Provenance:* adapted from MIT 6.S191 Lab 1 Part 2's input/target batch-construction step (overlap low — same underlying engineering step, reframed here as a correctness check rather than an initial implementation task, since this section already implements it)

---

## chapter_recurrent-neural-networks/bptt.md — Backpropagation Through Time

**Topic (2026-11):** The scalar recurrence $h_t = w_\textrm{h}h_{t-1} + x_t$ ($0.9^{40} \approx 0.015$, $1.1^{40} \approx 45$); backpropagation through the unrolled graph as a sum over earlier steps; the linear RNN's backward recursion $\mathbf{G}_t$ and matrix powers governed by the spectral radius, checked on random symmetric matrices; clipping for explosion, gated cells and stable linear recurrences for vanishing; truncated BPTT, detaching, and its bias.

**Current exercises (2026-11 book pass):** 6

1. **Backward pass by hand.** Derive one `RNNScratch` step's contributions to the gradients of $\mathbf{W}_\textrm{xh}$, $\mathbf{W}_\textrm{hh}$, $\mathbf{b}_\textrm{h}$, and $\mathbf{H}_{t-1}$, and reduce the recursion to the linear case. *Adapted from CMU 11-785 HW3 Part 1.*
1. **Power iteration and alignment.** For symmetric $\mathbf{M}$, show $\mathbf{M}^k$ has eigenvalues $\lambda_i^k$ and that $\mathbf{M}^k\mathbf{x}$ aligns with $\mathbf{v}_1$; what this implies for the direction of RNN gradients.
1. **Beyond clipping.** Propose one parameter-side and one optimizer-side remedy for exploding gradients, and say whether each changes the Jacobian product and helps against vanishing.
1. [code] **Backward pass in code.** ● Implement the full-window backward pass of the trained `RNNScratch` cell with tensor operations and compare with autograd. *Adapted from CMU 11-785 HW3 Part 1.*
1. [code] **Nonlinear Jacobians.** ● A nonsymmetric $\mathbf{W}_\textrm{hh}$ (transient growth at $\rho = 0.9$), the tanh RNN's Jacobian product at two input scales, and why a singular value above one is necessary but not sufficient for growth.
1. [code] **Effective memory horizon.** ● Gradient norm of the last loss with respect to the state $k$ steps back in the trained model; the 1% horizon against `num_steps`, and again after training with `num_steps=128`.

**Pre-pass exercises:** 4; disposition: keep 4, rewrite 0, drop 0 — the prior review flagged no clarity defects (eigen-analysis proof, nonlinear-Jacobian rerun, an open "other methods" question, and an explicit gradient-norm-vs-lag measurement with a defined 1%-of-baseline threshold); the two formatting defects noted elsewhere (3-space vs. 4-space sub-list indentation, an untabbed Discussions link) are file-level markup issues, not exercise-content problems, and are out of scope for this catalog.

**External sources found:**
- CMU 11-785 (Introduction to Deep Learning) HW3 Part 1, Fall 2021/2022 — https://deeplearning.cs.cmu.edu/F21/document/homework/HW3/HW3P1_writeup.pdf, https://deeplearning.cs.cmu.edu/F22/document/homework/HW3/HW3P1_F22.pdf — verified by direct read: requires deriving "fourteen gradients" for one RNN-cell step's manual backward pass, and the F22 writeup's own transition into its GRU section states the vanishing/exploding argument in prose ("a long product of matrices can cause the long-term gradients to vanish... or explode... one of the earliest methods proposed to solve this issue is LSTM") — the same phenomenon this section derives via eigenvalues, reached instead by deriving the backward pass by hand.
- Jurafsky & Martin, SLP3 3rd-ed draft, Ch. 13 "RNNs and LSTMs" — https://web.stanford.edu/~jurafsky/slp3/13.pdf — checked directly by full-text scan: zero end-of-chapter exercises of any kind, despite the chapter's body covering BPTT and the vanishing-gradient problem. No adoptable material here.
- Beyond CMU's homework, we did not find a course or textbook that poses the spectral-radius/eigenvector-alignment argument itself (this section's ex. 1) as a discrete, gradeable problem — every other source we checked states it as exposition rather than assigning a proof. This section's own set may be more rigorous on this specific point than what is available externally.

**Pre-pass proposed problem set** (provenance; superseded by the current exercises above):
1. [conceptual] **Power iteration and RNN gradients.** For symmetric $\mathbf{M}\in\mathbb{R}^{n\times n}$ with eigenvalues ordered $|\lambda_i|\geq|\lambda_{i+1}|$: show $\mathbf{M}^k$ has eigenvalues $\lambda_i^k$; prove that for random $\mathbf{x}$, $\mathbf{M}^k\mathbf{x}$ aligns with high probability with the top eigenvector $\mathbf{v}_1$; state what this implies for RNN gradients.
   *Provenance:* original
1. [short-code] **Nonlinear rerun.** Rerun the numerical demo with a general (non-symmetric) $\mathbf{W}_\textrm{hh}$, and with the true Jacobian product of a $\tanh$-nonlinear RNN. How does the nonlinearity change the growth of $\|\mathbf{J}^k\|$, and why can it never stabilize an $|\lambda|>1$ direction on its own?
   *Provenance:* original
1. [conceptual] **Beyond clipping.** Besides gradient clipping, propose one other method to cope with gradient explosion in RNNs, and state what property of the recurrence it targets.
   *Provenance:* original
1. [short-code] **Effective memory horizon.** For the RNN language model of the previous section, measure the gradient norm of the final-step loss with respect to the hidden state k steps earlier, as a function of k. Plot it, read off the lag where the norm drops below 1% of its k=0 value, and compare that horizon to the truncation length τ used in training.
   *Provenance:* original
1. [short-code] **Manual backward pass vs. autograd.** For one time step of the trained RNN cell, derive by hand the local Jacobians $\partial h_t/\partial \mathbf{W}_{hh}$, $\partial h_t/\partial h_{t-1}$, and $\partial h_t/\partial \mathbf{b}$; implement them as plain tensor operations; confirm they match autograd's `.grad` to float32 tolerance on a single training batch.
   *Provenance:* adapted from CMU 11-785 HW3 Part 1's "fourteen gradients" manual-backward-pass exercise (overlap high — same deliverable, one RNN-cell step's local Jacobians derived and verified by hand; cite on adoption)

---

## chapter_recurrent-neural-networks/decoding.md — Decoding and Generation

**Topic (2026-11):** The decoding problem ($1027^{50} > 10^{150}$ continuations); a `step_fn` interface over the concise RNN language model; greedy decoding, its myopia and loops; beam search with width $w$ and length normalization (`beam_search`); temperature, top-$k$, top-$p$, and min-$p$ in one sampler (`sample_next`, `generate`); log-probability against distinct-3 for greedy, beam, and min-$p$; the quadratic cost of re-running the prefix.

**Current exercises (2026-11 book pass):** 8

1. [code] **Repetition penalty.** Add a logit-dividing penalty $\gamma = 1.2$ to `sample_next`; distinct-3 with and without it, its effect on negative logits, and the legitimate repetitions it punishes.
1. **Beam search properties.** Construct probabilities for which width-2 beam search misses the most probable sequence; the width that guarantees exhaustive search; `step_fn` calls of beam and exhaustive search at $|\mathcal{V}| = 1{,}027$, $T = 50$, $w = 4$.
1. [code] **Length normalization.** Beam search with $w = 4$ and newline as `eos_id` at $\beta \in \{0, 0.75, 1.5\}$: lengths and log-probabilities of the winners, explained by the normalized score.
1. **Truncation under temperature.** Rewrite the min-$p$ condition in untempered probabilities, compute the kept sets of the three cutoffs at $\tau = 2$ for a five-token distribution, and find the rule whose set never changes with temperature.
1. [code] **Constrained sampling.** Mask every token containing "e"; the removed probability mass, why banning a word is harder than banning a token, and what beam search adds for constraints that span several tokens.
1. [code] **Top-$p$ versus min-$p$.** ● At $\tau = 1.5$, sweep both cutoffs and plot the log-probability of drawn tokens against the number kept; compare with Nguyen et al. (2025).
1. [code] **Failure taxonomy.** ● Classify the failures of 20 greedy and 20 min-$p$ continuations, and say which automated statistics detect each category. *Adapted from CS224n A4 Part 2(b).*
1. [code] **Constant cost per token.** ● A stateful `step_fn` that consumes only the newest token: check equal greedy output, time 50, 200, and 800 tokens, and explain why `beam_search` needs more bookkeeping.

**Pre-pass exercises:** 7; disposition: keep 7, rewrite 0, drop 0 — the prior review found all seven concrete, each with exact parameters or thresholds (θ=1.2, α∈{0,0.75,1.5} at k=4, T=1.5, k∈{1,2,4,8,16}); ex. 6's dependency on next chapter's translation model is a genuine, explicitly-flagged forward reference, not a hidden defect.

**External sources found:**
- Stanford CS224n, Assignment 4 "Neural Machine Translation with RNNs and Analyzing NMT Systems" (archived handout) — https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/assignments/a4.pdf — verified by direct read (via `pdftotext`): Part 2(b) gives four (reference, model) translation pairs and asks students to identify the specific error, hypothesize a linguistic or architectural cause, and propose one concrete fix; Part 2(c) has students derive single-example BLEU by hand (modified n-gram precision + brevity penalty) for two candidates against one vs. two reference translations, and explain why scoring against a single reference is more fragile.
- Notable non-finding, also verified: this is the course most closely associated with beam search in NMT — its own assignment decodes with beam search — yet a full-text search of its written/graded questions across two archived years turned up **zero** mentions of "beam" or "search" outside the coding spec. Beam search appears only as code to implement, never as a target of written analysis. By contrast, this section's own ex. 2–4 (does the best 3-step beam candidate contain the best 2-step one as a prefix; is exhaustive search a special case of beam search; a length-normalization sweep) demand real reasoning about beam search that we could not find replicated anywhere externally.

**Pre-pass proposed problem set** (provenance; superseded by the current exercises above):
1. [short-code] **Repetition penalty.** Implement a repetition penalty in `sample_next` (divide the probability of any already-generated token by θ>1 before truncation); decode greedily at θ=1.2; report whether it cures loops and what legitimate text it punishes.
   *Provenance:* original
1. [conceptual] **Beam search and prefix consistency.** Could the best 3-step beam candidate (by cumulative probability) fail to contain the best 2-step candidate as its prefix? Construct explicit probabilities or prove it impossible, and state what your answer implies about beam search missing the true argmax sequence.
   *Provenance:* original
1. [conceptual] **Exhaustive search as beam search.** For which beam size k does beam search coincide with exhaustive search? Justify your answer.
   *Provenance:* original
1. [short-code] **Length normalization sweep.** Add an `eos_id` to the beam-search demo; vary α ∈ {0, 0.75, 1.5} at k=4; compare the winning candidates' lengths and scores and explain the trend via the length-normalized scoring equation.
   *Provenance:* original
1. [short-code] **Top-p vs. min-p tuning.** At T=1.5, tune top-p's p and min-p's $p_{\min}$ until each just keeps continuations coherent; compare distinct-3 diversity at matched coherence; relate your finding to :citet:`Nguyen.Baker.Neo.ea.2025`.
   *Provenance:* original
1. [extended] **Beam width, quality, and speed.** Once the sequence-to-sequence translation model of the next chapter is trained, decode it with `beam_search` for k ∈ {1, 2, 4, 8, 16}; measure translation quality and decode time as functions of k; report where quality peaks and why it does not keep improving.
   *Provenance:* original
1. [conceptual] **Token-level vs. word-level constraints.** Ban every token whose text contains the letter "e" by masking logits to $-\infty$ before `sample_next`. Explain why banning a whole *word* is harder than banning a token under subword tokenization, and what can go wrong at a token boundary.
   *Provenance:* original
1. [short-code] **Decoding failure taxonomy.** Decode 20 continuations from a fixed prompt set under greedy decoding; sort the failures into a short taxonomy you define (e.g. repetition loop, premature EOS, topic drift, other); report each category's frequency with one example; state which category the repetition penalty of problem 1 targets and whether raising θ reduces that category's frequency without increasing another's.
   *Provenance:* adapted from CS224n A4 Part 2(b)'s translation error-taxonomy methodology (overlap medium — same identify/hypothesize-cause/propose-fix structure, applied to this section's own decoder rather than to translation errors; cite on adoption)
