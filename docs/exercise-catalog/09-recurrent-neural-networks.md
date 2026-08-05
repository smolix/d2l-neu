# Chapter Overview — chapter_recurrent-neural-networks

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

**Topic:** Autoregressive sequence modeling — fixed-window (n-gram-style) vs. latent-state (RNN-style) prediction, the Markov condition and stationarity, and one-step vs. multistep (rollout) forecasting with its error accumulation.
**Current exercises:** 6 (ex. 1 has 4 lettered sub-items); disposition: keep 5, rewrite 1, drop 0 — the prior review found this set essentially defect-free and concrete throughout ("no outright underspecified exercises... otherwise carries real content"); the one exception is ex. 1.3's "Can you incorporate older observations...?" filler phrasing, worth a straight rewrite into a direct instruction.

**External sources found:**
- No good external exercise tradition located for this section's specific framing. We checked Jurafsky & Martin SLP3 Ch. 3 (N-grams) — https://web.stanford.edu/~jurafsky/slp3/3.pdf — which shares the chain-rule/Markov-condition idea in its prose (used centrally for `language-model.md` below) but poses no exercise on multistep rollout, regime-switching, or fixed-window-vs-latent-state comparison. General NLP/ML course searches for "autoregressive rollout error accumulation" homework turned up nothing verifiable at the right level: courses either jump straight to n-gram LMs (no continuous-signal autoregression) or straight to RNNs (no fixed-window baseline). This is a real finding, not a search failure: the section's stats/forecasting framing sits between what NLP courses and what time-series/econometrics courses each typically assign.

**Proposed problem set** (our reference format):
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

**Topic:** Tokenization units (char/word/byte), byte pair encoding implemented from scratch and checked against `tiktoken`, vocabularies/special tokens, and real-world tokenizer quirks (glitch tokens, digit chunking).
**Current exercises:** 5; disposition: keep 5, rewrite 0, drop 0 — the prior review rated this the cleanest, most concrete set in the chapter (every item names an exact deliverable: a compression ratio, a longest-token list, a proposal); nothing to fix.

**External sources found:**
- Karpathy, `minbpe` repo and its `exercise.md` progression (2024) — https://github.com/karpathy/minbpe, https://raw.githubusercontent.com/karpathy/minbpe/master/exercise.md — a verified 5-step build order: (1) `BasicTokenizer` with train/encode/decode; (2) add the GPT-4 regex pre-tokenizer; (3) load real GPT-4 merges and match `tiktoken` exactly (handling rank recovery and the internal byte permutation); (4) special-token handling; (5) stretch goal, non-byte schemes like SentencePiece. Very high overlap — this is essentially the book's own approach, and step 4 (special tokens) is a natural extension the book's own set doesn't yet exercise.
- Jurafsky & Martin, SLP3 3rd-ed draft, Ch. 2 "Words and Tokens" — https://web.stanford.edu/~jurafsky/slp3/2.pdf — verified by direct read: despite covering BPE at length in its prose (and citing Kudo & Richardson's SentencePiece, Bostrom & Durrett's "BPE is suboptimal," etc. in its references), its own 7 end-of-chapter exercises (2.1–2.7) are entirely regex-writing and minimum-edit-distance computation — none touch BPE, vocabulary size, or subword tokenization at all. A clean non-overlap finding in an otherwise-canonical source.

**Proposed problem set** (our reference format):
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

**Topic:** n-gram language models (counting and temperature-controlled sampling), Zipf's law, perplexity and bits-per-byte as evaluation metrics, add-α smoothing.
**Current exercises:** 4; disposition: keep 4, rewrite 0, drop 0 — the prior review flagged no clarity defects; every exercise specifies exact parameters (temperatures, vocab size, a formula to verify, a smoothing sweep) and a concrete artifact.

**External sources found:**
- Jurafsky & Martin, SLP3 3rd-ed draft, Ch. 3 "N-gram Language Models," Exercises 3.1–3.12 — https://web.stanford.edu/~jurafsky/slp3/3.pdf — verified by direct read; the canonical exercise set for this exact topic. Particularly relevant: 3.2/3.3 (compute a sentence's probability unsmoothed vs. add-1-smoothed and explain which is higher and why), 3.5 (show that an unsmoothed bigram model *without* an end symbol assigns probability 1.0 to the sentences of every length separately — i.e., is not a valid distribution over sequences), 3.6 (derive the general add-1-smoothed trigram formula in terms of raw counts and vocabulary size V), 3.9–3.11 (build a small n-gram program, compare unigram/bigram statistics across two corpora, add random-sentence generation and perplexity), 3.12 (unigram perplexity of a test set drawn from a 91%-skewed 10-symbol training corpus).
- Georgetown University, COSC 572 (S23), Assignment 1 "N-Gram Language Models" — https://people.cs.georgetown.edu/nschneid/cosc572/s23/a1/ — verified by direct read; builds unigram/bigram/add-α-smoothed-bigram models on the Brown corpus and asks students to explain *why* smoothing shrinks high-frequency-context probabilities less than rare-context ones, and why that is desirable — a sharper framing of the "why" behind smoothing than a pure compute-the-number exercise.

**Proposed problem set** (our reference format):
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

**Topic:** The RNN hidden-state recurrence $h_t=\phi(W_{xh}x_t+W_{hh}h_{t-1}+b)$, why it defines a valid $P(x_t\mid x_{<t})$, and how its parameter count compares to an n-gram table's. Pure conceptual/derivation — no code, no data.
**Current exercises:** 5; disposition: keep 5, rewrite 0, drop 0 — the prior review called this "the shortest, cleanest file in the group," with every item concrete (a dimension question, a "why," gradient behavior, an explicit parameter-count derivation with numbers plugged in, and a reflection tied directly to the section's own stated limitations).

**External sources found:** No good external exercise tradition located for this section's specific framing — an explicit, verified finding, not a search gap:
- Jurafsky & Martin, SLP3 3rd-ed draft, Ch. 13 "RNNs and LSTMs" — https://web.stanford.edu/~jurafsky/slp3/13.pdf — full-text scan confirms **zero** occurrences of the word "exercise" anywhere in the chapter; no end-of-chapter problems exist at all, despite the chapter covering the RNN recurrence and language modeling with RNNs in its body.
- Goldberg, *Neural Network Methods for Natural Language Processing* (Morgan & Claypool / Springer) — https://link.springer.com/book/10.1007/978-3-031-02165-7 — full-text scan of the book confirms "exercise" occurs exactly twice in the entire text, both in passing prose ("as a simple exercise, the reader should..."), never as a numbered problem-set heading. The book has no formal exercises anywhere, RNN chapter included.
- CMU 11-785 HW3 (the course explicitly suggested as a source for this chapter) does cover the RNN recurrence's mechanics in real depth — but always bundled with a from-scratch coding deliverable (see `rnn-implementation.md` and `bptt.md` below), never as a standalone pencil-and-paper problem the way this section poses it.
- Conclusion: every source we checked pairs the RNN concept with an immediate implementation task; none offers this section's "just the recurrence, pencil-and-paper" framing as a discrete exercise.

**Proposed problem set** (our reference format):
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

## chapter_recurrent-neural-networks/rnn-implementation.md — Implementing RNN Language Models

**Topic:** An RNN language model built from raw tensor ops (embedding, recurrence, output head) on *The Time Machine* over a 1,024-token BPE vocabulary, then gradient clipping, training, text generation with temperature, and a concise framework-layer version.
**Current exercises:** 9; disposition: keep 9 (regrouped), rewrite 0, drop 0 — the prior review called this "the longest exercise list in the chapter... all uniformly bare/untagged/unnamed" yet flagged zero clarity defects (every item names a specific hyperparameter, dataset, or metric). Nothing here needs cutting; below we regroup two closely related pairs under nested sub-items to fit the catalog's 5–8-problem format without losing any content.

**External sources found:**
- CMU 11-785 (Introduction to Deep Learning) HW3 Part 1, Fall 2021/2022 writeups — https://deeplearning.cs.cmu.edu/F21/document/homework/HW3/HW3P1_writeup.pdf, https://deeplearning.cs.cmu.edu/F22/document/homework/HW3/HW3P1_F22.pdf — verified by direct read (via `pdftotext`, since the PDFs are not machine-summarizable as rendered): builds an RNN cell's forward *and* manual backward pass from scratch, explicitly enumerating "fourteen gradients" (∂L/∂W_hh, ∂L/∂h_{t-1}, etc.) that must be derived by hand and implemented without autodiff.
- MIT 6.S191, Lab 1 Part 2 "Music Generation with RNNs" (PyTorch notebook) — https://raw.githubusercontent.com/MITDeepLearning/introtodeeplearning/master/lab1/PT_Part2_Music_Generation.ipynb — verified by direct download and inspection of the notebook's own TODO cells: "Write a function to convert the all songs string to a vectorized... representation," "construct a list of input sequences for the training batch" (and the corresponding shifted-by-one target batch), "Add LSTM and Linear layers to define the RNN model," "define the compute_loss function," and "generate ABC format text of length 1000" via temperature-based sampling. Structurally this is the same data→model→loss→train→generate progression as this section, built on a different (music) corpus.
- Karpathy, "The Unreasonable Effectiveness of Recurrent Neural Networks" — http://karpathy.github.io/2015/05/21/rnn-effectiveness/ — verified by direct fetch: demonstrates the same char-level RNN trained separately on Paul Graham essays, Shakespeare, Wikipedia markdown, LaTeX (algebraic geometry), and Linux kernel C source, and states the temperature effect directly ("Decreasing the temperature... makes the RNN more confident, but also more conservative... higher temperatures will give more diversity but at cost of more mistakes").

**Proposed problem set** (our reference format):
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

**Topic:** The unrolled computation graph and full gradient of an RNN; vanishing/exploding gradients via spectral-radius analysis of repeated Jacobian products (linear case, proved; nonlinear tanh case, numerical); truncated BPTT.
**Current exercises:** 4; disposition: keep 4, rewrite 0, drop 0 — the prior review flagged no clarity defects (eigen-analysis proof, nonlinear-Jacobian rerun, an open "other methods" question, and an explicit gradient-norm-vs-lag measurement with a defined 1%-of-baseline threshold); the two formatting defects noted elsewhere (3-space vs. 4-space sub-list indentation, an untabbed Discussions link) are file-level markup issues, not exercise-content problems, and are out of scope for this catalog.

**External sources found:**
- CMU 11-785 (Introduction to Deep Learning) HW3 Part 1, Fall 2021/2022 — https://deeplearning.cs.cmu.edu/F21/document/homework/HW3/HW3P1_writeup.pdf, https://deeplearning.cs.cmu.edu/F22/document/homework/HW3/HW3P1_F22.pdf — verified by direct read: requires deriving "fourteen gradients" for one RNN-cell step's manual backward pass, and the F22 writeup's own transition into its GRU section states the vanishing/exploding argument in prose ("a long product of matrices can cause the long-term gradients to vanish... or explode... one of the earliest methods proposed to solve this issue is LSTM") — the same phenomenon this section derives via eigenvalues, reached instead by deriving the backward pass by hand.
- Jurafsky & Martin, SLP3 3rd-ed draft, Ch. 13 "RNNs and LSTMs" — https://web.stanford.edu/~jurafsky/slp3/13.pdf — checked directly by full-text scan: zero end-of-chapter exercises of any kind, despite the chapter's body covering BPTT and the vanishing-gradient problem. No adoptable material here.
- Beyond CMU's homework, we did not find a course or textbook that poses the spectral-radius/eigenvector-alignment argument itself (this section's ex. 1) as a discrete, gradeable problem — every other source we checked states it as exposition rather than assigning a proof. This section's own set may be more rigorous on this specific point than what is available externally.

**Proposed problem set** (our reference format):
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

**Topic:** Turning a trained LM's per-step conditionals into an actual sequence — greedy decoding, beam search (with length normalization), sampling and its dials (temperature, top-k/p, min-p), and evaluating decoding strategies for diversity/efficiency.
**Current exercises:** 7; disposition: keep 7, rewrite 0, drop 0 — the prior review found all seven concrete, each with exact parameters or thresholds (θ=1.2, α∈{0,0.75,1.5} at k=4, T=1.5, k∈{1,2,4,8,16}); ex. 6's dependency on next chapter's translation model is a genuine, explicitly-flagged forward reference, not a hidden defect.

**External sources found:**
- Stanford CS224n, Assignment 4 "Neural Machine Translation with RNNs and Analyzing NMT Systems" (archived handout) — https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/assignments/a4.pdf — verified by direct read (via `pdftotext`): Part 2(b) gives four (reference, model) translation pairs and asks students to identify the specific error, hypothesize a linguistic or architectural cause, and propose one concrete fix; Part 2(c) has students derive single-example BLEU by hand (modified n-gram precision + brevity penalty) for two candidates against one vs. two reference translations, and explain why scoring against a single reference is more fragile.
- Notable non-finding, also verified: this is the course most closely associated with beam search in NMT — its own assignment decodes with beam search — yet a full-text search of its written/graded questions across two archived years turned up **zero** mentions of "beam" or "search" outside the coding spec. Beam search appears only as code to implement, never as a target of written analysis. By contrast, this section's own ex. 2–4 (does the best 3-step beam candidate contain the best 2-step one as a prefix; is exhaustive search a special case of beam search; a length-normalization sweep) demand real reasoning about beam search that we could not find replicated anywhere externally.

**Proposed problem set** (our reference format):
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
