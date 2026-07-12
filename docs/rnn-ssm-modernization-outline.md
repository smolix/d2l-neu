# Chapters 9–10 Detailed Outline (for review)

*2026-07-11. Companion to `rnn-ssm-modernization-overview.md` (approved). Content
and structure only; no code is written yet (JAX/NNX refactor in flight). Code cells
are described by what they compute; "all tabs" means pytorch/jax/tf/mxnet.*

## 0. Cross-cutting decisions

**Directories and files.** Both chapter directories keep their names (renames would
churn the outputs store, labels, and scheduler config for no reader-visible gain).
File plan:

| Chapter 9 (`chapter_recurrent-neural-networks/`) | Chapter 10 (`chapter_recurrent-modern/`) |
|---|---|
| `sequence.md` — rewrite in place | `lstm.md` — rewrite in place (absorbs `gru.md`, `deep-rnn.md`, `bi-rnn.md`) |
| `text-sequence.md` — rewrite in place | `seq2seq.md` — rewrite in place (absorbs `machine-translation-and-dataset.md`, `encoder-decoder.md`) |
| `language-model.md` — rewrite in place | `ssm.md` — **new** |
| `rnn.md` — moderate revision | `mamba.md` — **new** |
| `rnn-implementation.md` — **new** (merges `rnn-scratch.md` + `rnn-concise.md`) | deleted: `gru.md`, `deep-rnn.md`, `bi-rnn.md`, `machine-translation-and-dataset.md`, `encoder-decoder.md`, `beam-search.md` |
| `bptt.md` — targeted cuts + additions | |
| `decoding.md` — **new** (absorbs `beam-search.md`) | |
| deleted: `rnn-scratch.md`, `rnn-concise.md` | |

Chapter counts: 9 has 7 sections (was 7), 10 has 4 (was 8). Deleted files require a
crossref sweep (labels like `sec_beam-search`, `sec_seq2seq`, `sec_machine_translation`
are referenced from the attention chapter and beyond); keep old `:label:` names alive
on the nearest surviving anchor wherever possible (e.g. `sec_beam-search` moves onto
the beam-search subsection of `decoding.md`) so most crossrefs fix themselves.

**Chapter titles.**
- Ch. 9: **"Sequence Models and Language Models"**
- Ch. 10: **"Gated and Linear Recurrence"**

**Framework-tab policy.**
- 9.1–9.7: all tabs. The BPE tokenizer and all decoding strategies are pure-Python
  against logits, so they are framework-light by construction.
- 10.1–10.2: all tabs (carried-over LSTM/seq2seq code exists for all four).
- 10.3–10.4 (SSM/Mamba): **pytorch + jax + tensorflow; MXNet omitted** (no scan
  primitive, per agreement). JAX is the natural showcase (`lax.associative_scan`);
  pytorch gets a short hand-rolled log-depth scan helper in `d2l` (used by both
  sections); TF uses the same algorithm via its own ops. If TF proves too costly in
  practice we drop to pytorch+jax; decision deferred to implementation.

**Datasets.** Time Machine remains the corpus for 9.2–9.7 and 10.4 (right size, zero
new infra). Fra-eng survives only inside 10.2, loaded compactly. Sequential/permuted
MNIST (already in the repo's data pipeline as FashionMNIST cousin) is used once in
10.3 as the long-range benchmark. TinyStories appears only as an exercise pointer.

**Figures.** Each chapter gets a generator (`tools/gen_rnn_figures.py`,
`tools/gen_modern_rnn_figures.py`) in the house style; existing hand SVGs that
survive (unfolded RNN, LSTM/GRU gate diagrams, encoder-decoder, beam tree) are
redrawn in that style so each chapter is single-style. New figures are listed per
section below. Data plots that teach a computed result stay inline as `d2l.plot`.

**New Python dependency:** `tiktoken` (shared, all venvs; pulls `regex`). Used in
9.2 to verify our from-scratch BPE against production tokenizers. Prebuilt wheels
on all supported platforms; encoding files download+cache at notebook-execution
time only (pre-seedable via `TIKTOKEN_CACHE_DIR` if we want hermetic runs).

**New `d2l` library additions** (via `#@save`, exact API at implementation time):
BPE tokenizer class (train/encode/decode), decoding helpers (temperature/top-k/top-p
sampling over a logits callback, beam search), log-depth associative scan (pytorch),
chrF scorer. These are deliberately small and reused across 9.2→10.4.

**Citations added** (BibTeX): Sennrich et al. 2016 (BPE), Holtzman et al. 2020,
min-p (Nguyen et al. 2025), Gu-Goel-Ré 2022 (S4), Gu et al. 2020 (HiPPO), Gu et al.
2022 (S4D), Smith et al. 2023 (S5), Orvieto et al. 2023 (LRU), Feng et al. 2024
(minGRU/minLSTM), Gu & Dao 2023 (Mamba), Dao & Gu 2024 (Mamba-2), De et al. 2024
(Griffin), Lieber et al. 2024 (Jamba), Jelassi et al. 2024 (Repeat After Me), Arora
et al. 2023 (MQAR/zoology), Popović 2015 (chrF), Blelloch 1990 (scan), Tay et al.
2021 (LRA), Beck et al. 2024 (xLSTM), Peng et al. (RWKV), Yang et al. 2024 (GLA).

---

## Chapter 9 — Sequence Models and Language Models

### 9.0 `index.md` (rewrite, ~700 words)

Framing: much of intelligence is next-thing prediction; sequences break the i.i.d.
fixed-length assumption of Parts I–II. Introduce the two ideas the chapter builds:
(1) autoregressive factorization turns generation into supervised learning; (2) a
*hidden state* summarizes an unbounded past in bounded memory. Preview: language
modeling as the running application, culminating in a small LM we train and sample
from properly. One honest paragraph on history and status: RNNs powered the 2010s
breakthroughs, transformers displaced them at scale, yet recurrence returns in
modern guise (ch. 10) because bounded-memory inference matters; this chapter's
concepts (factorization, perplexity, BPTT, decoding) are exactly the ones the LLM
chapters stand on. TOC.

### 9.1 `sequence.md` — Working with Sequences (~3,000 words; from 3,850)

*Learning goals: autoregressive factorization; Markov assumptions; stationarity;
why multi-step prediction is fundamentally harder than one-step.*

1. **Sequential data and its challenges.** Trimmed carryover (examples list cut to a
   third; anchor examples: text, audio, time series). Notation.
2. **Autoregressive models.** P(x_t | x_{t-1},…,x_1); the two practical strategies
   (fixed window = AR, latent summary = latent AR) — unchanged core, tightened.
   Explicit forward hook: the latent-AR diagram *is* the RNN of 9.4 and the SSM of
   ch. 10; the fixed window *is* the n-gram of 9.3 (and, later, the attention
   context window).
3. **Markov models and stationarity.** Carryover, compressed. First-order vs
   higher-order; when the factorization order matters (drop the lengthy
   causality digression to two sentences).
4. **Training.** Code: synthetic sine+noise series; build lag-feature dataset; fit a
   small MLP (all tabs). Carryover code, lightly modernized.
5. **Prediction.** Code: one-step-ahead predictions (good) vs k-step rollout where
   predictions are fed back (divergence plot); compute per-horizon error curve.
   New closing prose: this error accumulation is generic to *all* autoregressive
   generation — LLMs drifting off-topic, world models losing coherence — and is why
   decoding strategies (9.7) and exposure to real data matter. (One paragraph, no
   jargon-dump.)

Figures: keep the k-step rollout plots (computed); one small generated figure:
AR-window vs latent-state schematic (replaces nothing; aids 9.4 hook).
Exercises: mostly carryover (vary order k; regime-switching series) + new: compute
the horizon at which rollout error exceeds the series' own variance.

### 9.2 `text-sequence.md` — From Text to Tokens (~3,900 words; from 2,240)

*Learning goals: tokenization as lossless compression with a learned dictionary;
byte-level BPE end-to-end including pre-tokenization; vocabulary/sequence-length
trade-off; how production tokenizers (tiktoken) work and misbehave.*

1. **Reading the dataset.** Time Machine loader (carryover, condensed).
2. **Characters, words, and the trade-off.** Char tokenization: tiny vocab, long
   sequences, no OOV. Word tokenization: short sequences, huge vocab, OOV and
   morphology problems (carryover material reframed as *two ends of a spectrum*).
   Code: tokenize both ways, report vocab size vs corpus length in tokens.
3. **Text as bytes.** Short primer (half page): strings are UTF-8 byte sequences;
   256 base symbols cover every language, emoji, and typo — start from bytes and
   *no input is ever out of vocabulary*; the cost is long sequences (≈1 byte/char
   for English, more elsewhere — hook for the fertility discussion below). Code:
   inspect the UTF-8 bytes of a mixed English/accented/emoji string.
4. **Byte-pair encoding.** The algorithm as iterated merge-of-most-frequent-pair
   over byte sequences. Hand-trace 3 merge steps on a toy 5-word corpus (table).
   Code (pure Python, all tabs share it): `train_bpe(text, vocab_size)` producing
   ranked merges + vocab; `encode` (apply merges greedily by rank) / `decode`;
   train on Time Machine with vocab 1,024; show the first/last learned merges
   (readable: " th", "e ", … → " the time traveller"); compression ratio in
   bytes/token as vocab grows (computed plot, 256→4k sweep). Remark: encoding is
   defined by merge *rank order*, so the same table is applied identically at
   train and inference time.
5. **Pre-tokenization: telling BPE where words end.** The missing ingredient
   between naive BPE and production tokenizers. Unconstrained merges happily
   cross word boundaries (" of the" as one token) and glue punctuation to words,
   bloating the vocab with brittle composites. Fix: first split text into chunks
   with a regex (letters runs, digit runs, punctuation, leading-space handling —
   show GPT-2's actual pattern, lightly annotated), then run BPE *within* chunks
   only. Digit-run capping (cl100k/o200k limit numbers to ≤3-digit chunks)
   previewed here, paying off in the arithmetic pathology below. Code: add
   pre-tokenization to our trainer, retrain, diff the learned vocab against the
   unconstrained run (fewer multi-word tokens, better generalization). Mirrors
   the minbpe basic→regex progression (cite Sennrich 2016; Radford et al. 2019).
6. **Vocabulary and special tokens.** Slimmed carryover `Vocab` unified with the
   BPE vocab; `<pad>`, `<bos>`, `<eos>` introduced here once (10.2 reuses); why
   special tokens bypass BPE entirely and must be handled explicitly (tiktoken's
   `allowed_special` guard as the cautionary example — prompt-injection-adjacent
   footnote).
7. **Tokenizers in the wild.** What **tiktoken** stores: a bytes→rank table (show
   a few entries), i.e. exactly our merge structure — then the verification
   moment: load GPT-2's published ranks into *our* encoder from subsection 4–5
   and reproduce tiktoken's output token-for-token on a test paragraph. Compare
   our 1k-vocab Time-Machine tokenizer, `gpt2` (50k), and `o200k_base` (200k) on
   the same text; token counts fall across generations. Pathologies, each in 2–4
   sentences with a one-line demo where cheap: fertility across languages
   (English/German/Greek tokens-per-word table), digit chunking and arithmetic
   (ties back to subsection 5), glitch tokens (prose + citation). Frame: the
   tokenizer is part of the model; mismatches cost context budget and capability.
   (tiktoken = new shared dependency; ~2–3 MB of wheels incl. `regex`; encoding
   files fetched/cached at execution time like DATA_HUB assets.)
8. **Exploratory statistics → moved to 9.3** (Zipf belongs with n-grams).

Figures: generated: the char↔BPE↔word granularity spectrum with vocab/length axes;
merge-tree snippet for the toy corpus; pre-tokenization pipeline strip (text →
regex chunks → merges within chunks). Computed: compression-vs-vocab plot,
fertility bar chart.
Exercises: BPE on your own text; vocab-size sweep effect on 9.5's LM perplexity
(forward exercise); ablate pre-tokenization and inspect the worst learned tokens;
explain why "solidgoldmagikarp"-style tokens arise from a train-corpus mismatch;
digits: compare 2-digit vs 3-digit chunking on addition prompts.

### 9.3 `language-model.md` — Language Models (~3,000 words; from 2,700)

*Learning goals: LM = probability model over token sequences; the universal-interface
view; n-gram estimation and its sparsity wall; perplexity and bits-per-byte;
sequence partitioning for training.*

1. **What a language model buys you.** Rewritten opening: assign probabilities to
   sequences and you can *generate* (sample continuations), *score* (rank ASR/MT
   hypotheses), and — the modern realization — *do any task expressible as a
   continuation* (translation, QA as prompted continuation; two-line teaser of ch.
   LLMs). Replaces the current speech-recognition-only motivation.
2. **n-gram language models.** Markov chain on tokens; MLE from counts; the
   sparsity/zero-count wall as vocab and order grow; Laplace smoothing in one
   formula + one sentence of history (Kneser-Ney relegated to a citation). Code:
   build unigram/bigram/trigram count models on Time Machine (word tokens for
   readability); **sample text from each** — the visible quality progression is the
   teaching moment; report each model's perplexity (defined next) and its
   zero-count rate on a held-out split.
3. **Word frequency and Zipf's law.** Moved from 9.2; unigram/bigram/trigram
   log-log frequency plot (carryover code); implication: the tail is where n-grams
   die and generalizing models are needed.
4. **Perplexity and bits-per-byte.** Cross-entropy per token; perplexity as
   effective branching factor (keep current intuition passages, they're good).
   New: per-token metrics are tokenizer-dependent — define bits-per-byte and show
   the same model's ppl vs bpb under char/BPE/word tokenization (code, small
   table); note bpb is how modern LM training runs are compared.
5. **Partitioning sequences.** Carryover (random offset, minibatch layout figure),
   condensed; emit BPE token ids now.

Figures: carryover partitioning diagram (redrawn house-style); Zipf plot (computed).
Exercises: trigram sampling with temperature (teaser of 9.7); estimate memory of a
5-gram table at vocab 50k; ppl↔bpb conversion; held-out smoothing comparison.

### 9.4 `rnn.md` — Recurrent Neural Networks (~2,400 words; from 2,150)

*Learning goals: hidden state as a learned, fixed-size summary of unbounded history;
the recurrence equation; weight tying across time; what an RNN LM computes.*

1. **Neural networks without hidden state.** Carryover, one page: MLP on a fixed
   window = neural n-gram (cite Bengio 2003), inherits the window limit.
2. **Recurrent networks with hidden state.** The recurrence
   H_t = φ(X_t W_xh + H_{t-1} W_hh + b); unrolling; parameter sharing across time;
   code: the two-matmul equivalence demo (carryover). New emphasis paragraph:
   **constant memory per step** — the state never grows with sequence length; this
   is recurrence's defining trade (contrast teased with attention's grow-with-length
   memory; resolved in ch. 10/11). This paragraph is the seed of the whole ch. 10
   arc, worth its own `:label:` for later reference.
3. **RNN language models.** Update from char-level to BPE-token-level throughout
   (embedding lookup instead of one-hot, matching 9.2); output head, softmax over
   vocab; training-time inputs/targets shift-by-one figure. Teacher forcing named
   here (not in 10.2): train on gold prefixes, generate on own outputs — connect
   back to 9.1's rollout-error discussion.
4. **Summary + what could go wrong.** Perplexity recap; teaser: gradients through
   long products (9.6) and the fix (10.1).

Figures: unfolded RNN (existing, restyled); shift-by-one LM training diagram
(restyled); no new ones.
Exercises: carryover set trimmed; add: parameter count vs n-gram table size at
equal context.

### 9.5 `rnn-implementation.md` — Implementing RNN Language Models (~3,600 words; merges 4,290 + 1,270)

*Learning goals: an RNN LM end-to-end from scratch; gradient clipping; then the
framework's fused layer; sampling a first continuation.*

1. **From scratch.** Parameters; embedding lookup; the step function; the unrolled
   forward pass returning logits at every step (all tabs; NNX idioms per the
   ongoing refactor — this outline does not fix JAX API details).
2. **Gradient clipping.** Stays here (practical training machinery), carryover math
   condensed; norm-clipping code.
3. **Training.** Loop on Time Machine BPE ids; perplexity curve; a note on why we
   detach state between minibatch chunks (truncated BPTT preview → 9.6).
4. **Generation, first contact.** Greedy and temperature sampling from the trained
   model with a prefix (compact helper reused/expanded in 9.7); show a few
   continuations, note the repetition/degeneration to be fixed in 9.7.
5. **Concise implementation.** The framework RNN layer replaces the hand-rolled
   step; identical training call; brief speed comparison table (why fused kernels
   win — one paragraph, no cuDNN lore).
6. **Summary.**

Figures: none new (perplexity curves computed).
Exercises: merged/pruned from both old sections; add: swap in the 2,048-vocab BPE
from 9.2's exercise and compare bpb (not ppl) against the 1,024 default; implement
weight tying between embedding and output head.

### 9.6 `bptt.md` — Backpropagation Through Time (~2,100 words; from 2,420)

*Learning goals: the gradient of a recurrence is a product of Jacobians; when it
vanishes/explodes; truncation as the standard practical answer and its bias.*

1. **The unrolled graph and the full gradient.** Carryover derivation kept
   essentially intact (it's rigorous and good), with the three-equation summary box.
2. **Vanishing and exploding gradients.** Eigenvalue/spectral-radius analysis
   (carryover); numerical demo of the Jacobian-product norm across time (small
   code cell). Connect: clipping (9.5) handles explosion; vanishing needs
   *architecture* — gates (10.1) or a reparameterized linear recurrence with decay
   pinned inside the unit circle (10.3). This paragraph is the load-bearing bridge
   of the whole part; the SSM section refers back to it by label.
3. **Truncated BPTT.** Regular truncation only; what bias it introduces; the
   detach-state idiom from 9.5 named as the implementation. **Cut: randomized
   truncation** subsection and its estimator analysis (retain the citation in a
   one-line remark). New closing remark (3–4 sentences): the store-vs-recompute
   trade generalizes — activation checkpointing in modern large-model training is
   the same idea applied to depth; pointer to the LLM chapter.
4. **Summary.**

Figures: keep the full/truncated comparison figure, drop the randomized panel
(regenerate house-style).
Exercises: prune to gradient-analysis ones; add: measure gradient norm vs time-lag
for the 9.5 model and identify the effective memory horizon.

### 9.7 `decoding.md` — Decoding and Generation (**new**, ~3,400 words)

*Learning goals: generation = choosing from an exponential space using local
conditionals; failure modes of maximization; the modern sampling toolkit; where
beam search still earns its keep.*

1. **The decoding problem.** The trained LM gives conditionals; a "best" sequence
   is an argmax over |V|^T candidates — intractable, and (for open-ended text) not
   even desirable. Two families: *maximization* (deterministic tasks: MT, ASR,
   code) and *sampling* (open-ended generation). This taxonomy organizes the
   section.
2. **Greedy decoding.** One-liner algorithm; demo on the 9.5 model: fluent-ish but
   repetitive loops (show a degenerate continuation). Why: argmax feedback
   amplifies the mode; cite degeneration findings.
3. **Beam search.** Algorithm + worked k=2 example (carryover from old 10.8,
   restyled figure); length normalization; compute: beam-decode the same prefixes,
   show it fixes local myopia but *increases* repetition for open-ended text; the
   large-beam curse in one paragraph. Where it lives today: ASR/MT/constrained
   decoding, not chat. (`sec_beam-search` label preserved here.)
4. **Sampling and its dials.** Temperature (logit scaling; T→0 = greedy);
   top-k; nucleus/top-p with Holtzman's unreliable-tail argument; min-p as the
   2025 refinement (scale cutoff by the max prob). Code: one
   `sample(logits_fn, strategy)` helper; generate continuations under each
   strategy from the same prefix; a computed figure visualizing one actual
   next-token distribution from our model with the k/p/min-p cutoffs overlaid —
   this single plot carries the section.
5. **Evaluating generated text.** Short: ppl scores the model, not a sample;
   distinguishability/human preference as the real target; pointer to LLM-chapter
   evaluation (LLM-as-judge etc.). Three paragraphs, no benchmark tour.
6. **Efficiency preview.** One paragraph: each generated token re-runs the model;
   recurrent states make this cheap (O(1) per token), a virtue ch. 10 builds on;
   speculative decoding name-dropped with citation, deferred to the LLM chapter.
7. **Summary.**

Figures: beam tree (restyled carryover); truncation-strategies overlay (computed
from the model); generated schematic: maximization-vs-sampling task map.
Exercises: implement repetition penalty; beam width vs quality on 10.2's MT task
(forward); tune min-p vs top-p at high temperature; sampling with a banned-word
constraint.

---

## Chapter 10 — Gated and Linear Recurrence

### 10.0 `index.md` (rewrite, ~600 words)

The chapter's single question: *what should a hidden state remember?* Vanilla RNNs
answer "whatever SGD finds," and 9.6 showed the gradients fight you. Three answers,
in order: **gate it** (LSTM/GRU — learned, input-and-state-dependent read/write/
forget); **linearize it** (drop the nonlinearity in state propagation so training
parallelizes and memory is analyzable: minGRU, SSMs/S4); **select it** (make the
linear dynamics input-dependent again — Mamba — recovering content-awareness at
linear cost). Ends with the honest limit (a fixed-size state cannot recall
arbitrary detail) that hands off to attention. Note that this chapter now spans
1997–2024 and that the seq2seq section carries the historical MT thread.

### 10.1 `lstm.md` — Gated Recurrent Networks (~4,200 words; absorbs gru/deep/bi-rnn: was 3,560+2,550+1,940+1,450)

*Learning goals: multiplicative gating as learned memory control; the LSTM cell;
GRU as its streamlining; depth and bidirectionality as orthogonal axes.*

1. **Why gates.** From 9.6: additive state updates with weight ~1 preserve
   gradients; but memory must also be *written selectively* and *cleared*. The
   gate = sigmoid-controlled elementwise multiply. (Carryover intro condensed;
   Lyapunov/stability framing from the research kept to one intuitive paragraph.)
2. **The LSTM cell.** Carryover equations/figures (restyled): input/forget/output
   gates, candidate, cell state, hidden state; the constant-error-carousel
   intuition. From-scratch implementation and training on the 9.5 task; then the
   concise framework layer (same in-section scratch→concise rhythm as 9.5).
3. **The GRU.** Absorbed from old 10.2, compressed ~4:1: equations, reset/update
   roles, one figure; *no* from-scratch build — one concise-layer training run for
   comparison. Framing: fewer gates, comparable quality, cheaper — and its real
   2020s legacy is that stripping it further yields a *linear* recurrence (explicit
   pointer to 10.3's minGRU).
4. **Deep recurrent networks.** Absorbed from old 10.3, one subsection: stacking
   as function composition over the same time axis; one figure; `num_layers=2`
   demo appended to the LSTM training cell. Residual connections between layers
   mentioned (callback to ResNet chapter).
5. **Bidirectional recurrent networks.** Absorbed from old 10.4, one subsection:
   two passes, concatenated states; *offline* encoders only — the LM-generation
   misuse caveat (the old section's best content, kept); shapes demo via
   `bidirectional=True`; legacy note: ELMo's bi-LSTMs → BERT's bidirectional
   attention (forward pointer).
6. **Gating beyond RNNs.** New 3-paragraph closer: gates recur everywhere —
   GLU/SwiGLU in transformer MLPs, the forget-gate reborn in Mamba's Δ and in
   Griffin/xLSTM; table of "same idea, different clothes." Sets up 10.3.
7. **Summary.**

Figures: LSTM cell, GRU cell, deep-RNN, bi-RNN diagrams (all existing, redrawn in
one house style — this is the chapter's biggest figure batch).
Exercises: merged/pruned from the four sections; add: ablate the forget gate and
measure ppl; verify GRU≈LSTM at matched parameter count; why bi-RNNs can't
generate.

### 10.2 `seq2seq.md` — Encoder–Decoder Models for Sequence Transduction (~4,000 words; merges 1,980+1,240+5,260 → less than half)

*Learning goals: the encoder-decoder abstraction; conditional generation with
teacher forcing; masked loss over padded batches; chrF/BLEU evaluation; the
fixed-vector bottleneck.*

1. **Sequence transduction and the abstraction.** Encoder maps source → state;
   decoder is a conditional LM (9.3's machinery, conditioned). The abstraction
   outlived its first implementation: Whisper (speech→text), image captioning,
   modern multimodal front-ends all keep this shape. (Absorbs old encoder-decoder
   section including its interface figure; the abstract base-class code kept, it's
   three short cells and genuinely reused.)
2. **The machine translation dataset.** Absorbed from old 10.5, cut ~4:1: fra-eng
   download; **BPE tokenization using 9.2's tokenizer** (one shared 4k vocab over
   both languages — one sentence on why sharing is fine and standard); padding,
   truncation, `<bos>`/`<eos>`; one compact cell producing batches; the
   length-histogram plot stays (it motivates padding/masking). All
   bucketing/engineering digressions dropped.
3. **The seq2seq model.** GRU encoder; decoder consuming final encoder state
   (context repeated at each step, as now); teacher forcing referenced from 9.4
   (one reminder sentence, not a re-explanation); masked cross-entropy (kept —
   genuinely practical and nowhere else in the book).
4. **Training and inference.** Train; greedy translation via 9.7's helper; then
   beam search via 9.7's helper with a 2–3 point score table (greedy vs k=2,5) —
   the payoff of having decoding as a first-class section.
5. **Evaluation.** chrF as the taught metric (definition + ~10-line scorer, cite
   Popović 2015; it's the WMT-recommended lexical metric); BLEU demoted to a
   remark ("you will still see it; brevity penalty in one formula, no code");
   one-line pointer: neural metrics (COMET) and LLM judges live in the LLM
   chapter.
6. **The bottleneck.** New closing: everything about the source crosses one fixed
   vector. Computed demo: chrF vs source-sentence length, visibly degrading.
   Two escapes exist: let the decoder *look back at all encoder states*
   (attention, next part) — or make the fixed state *much better at remembering*
   (rest of this chapter). Both forward hooks are honest; the chapter continues
   with the second.
7. **Summary.** (Labels `sec_machine_translation`, `sec_encoder-decoder` preserved
   on subsections 2 and 1.)

Figures: encoder-decoder schematic, seq2seq unrolled diagram (restyled carryovers);
computed: length-vs-chrF bar plot.
Exercises: pruned carryover set; add: swap shared BPE for per-language vocabs;
beam-width sweep; evaluate with and without length normalization.

### 10.3 `ssm.md` — Linear Recurrence and State Space Models (**new**, ~4,500 words)

*Learning goals: linearizing the recurrence enables parallel training (scan) and
analyzable memory (eigenvalues); the continuous-time view gives a principled
parameterization (discretization, Δ-as-gate); LTI recurrence ≡ convolution; S4D as
the assembled architecture. MXNet tab omitted.*

1. **The parallelization problem.** RNN training is inherently sequential in t;
   GPUs starve. Set the goal: keep O(1)-state inference, gain parallel training.
   (No attention spoilers needed; "parallel over the sequence like the CNNs of
   ch. 7" is the reference point.)
2. **Linearizing the gates: minGRU.** Take 10.1's GRU; make gates depend on x_t
   only and drop the candidate's dependence on h_{t-1}:
   h_t = (1−z_t)⊙h_{t-1} + z_t⊙h̃_t — now *linear in h*. Cite Feng et al. 2024;
   also LRU. Two consequences, each a subsection driver: (a) parallel evaluation
   via associative scan; (b) memory you can read off the decay coefficients.
3. **Parallel scans.** The associative operator for affine recurrences; Blelloch
   log-depth idea with a small worked diagram (8 elements). Code: `d2l` scan
   helper (jax native / pytorch hand-rolled log-depth); numerically verify
   scan == sequential loop; wall-clock vs sequence length plot (the "trains like
   a CNN" moment). Then a minGRU cell + a short LM training run matching 10.1's
   GRU quality at a fraction of the time (modest claim, honest setup).
4. **State space models.** Now the principled version: continuous-time linear
   dynamics ẋ = Ax + Bu, y = Cx (+ Du); why continuous — parameterize *dynamics*,
   not step tables; ZOH discretization with closed form Ā = exp(ΔA),
   B̄ = (ΔA)^{-1}(exp(ΔA)−I)ΔB; **Δ is a learnable step size and acts exactly like
   a forget/update gate** (the single most illuminating bridge from 10.1 — box
   this). Diagonal A (complex or real-negative) as the practical parameterization;
   eigenvalues inside the unit circle after discretization ↔ 9.6's stability
   analysis, now *by construction* rather than by hope.
5. **Recurrence is convolution.** Unroll the LTI recurrence → y = K * u with
   K = (CB̄, CĀB̄, CĀ²B̄, …); code: materialize K for a trained/random SSM,
   verify conv == scan == loop to numerical precision; the three views
   (ODE/recurrence/convolution) figure. FFT training mentioned in prose (one
   paragraph); we *implement* the scan path since it also survives selectivity
   (10.4).
6. **Remembering the past: HiPPO, stated.** The memory question: what should A be
   so x(t) summarizes history? HiPPO answer stated (optimal online compression of
   the past onto orthogonal polynomials), the matrix given, derivation cited not
   derived; one generated figure: reconstructing a function from its HiPPO state
   at increasing state sizes. S4→S4D→S5 lineage compressed to one paragraph
   (structure needed for stability/efficiency; diagonal suffices).
7. **S4D in practice.** Assemble: embedding → stacked (S4D layer + gated MLP +
   norm/residual) blocks → head. Code: train on **sequential MNIST** (784-step
   pixel sequences); comparison table vs 10.1 LSTM: accuracy and time/epoch —
   the long-range-memory payoff made concrete. Brief LRA mention (cite) for
   where this lineage was proven out.
8. **Summary.** Linear recurrence: parallel training, principled memory, still
   content-*blind* — the kernel is fixed regardless of input. Cliffhanger for 10.4.

Figures (all generated, new): scan tree; ODE/recurrence/convolution triptych;
HiPPO reconstruction; SSM block schematic. Computed: scan wall-clock plot, sMNIST
training curves.
Exercises: eigenvalue-radius sweep vs effective memory; verify ZOH against Euler
discretization error; kernel visualization for random vs HiPPO A; minLSTM from
minGRU by analogy.

### 10.4 `mamba.md` — Selective State Space Models (**new**, ~4,000 words)

*Learning goals: why LTI models cannot do content-dependent computation;
selectivity as input-dependent (Δ,B,C); the Mamba block; what fixed-state models
provably trade away; the hybrid landscape. MXNet tab omitted.*

1. **The selectivity problem.** A concrete failing task: *selective copy*
   (remember tokens flagged by context, ignore filler) — an LTI SSM/S4D cannot
   solve it because its dynamics ignore content (its kernel is input-independent);
   gated RNNs can, slowly. Code: generate the task, show S4D plateaus while the
   10.1 LSTM solves it. (Also name induction heads/associative recall as the
   LM-relevant version; cite MQAR.)
2. **Selective SSMs.** Make Δ_t, B_t, C_t functions of x_t. Δ_t as input-dependent
   forgetting = the LSTM gate, rediscovered in linear-state form (callback to the
   10.3 box). The price: kernel view dies (no fixed K); the save: the recurrence
   is *still* an associative affine map, so the 10.3 scan applies unchanged.
   Hardware-awareness (fusion, recomputation — why the real kernel is fast) in
   one prose paragraph, no CUDA content.
3. **The Mamba block.** Architecture walk: linear up-projection, short causal
   conv1d, SiLU, selective SSM, multiplicative gate branch, down-projection
   (annotated block figure); where it sits vs the transformer block it competes
   with (one sentence; full comparison deferred). Code: minimal Mamba block from
   parts already built (10.3 scan + gating), a few dozen lines; assemble a small
   LM; train on Time Machine BPE tokens; capstone table: ppl/bpb, params,
   time/epoch, and generation samples via 9.7 for **LSTM (10.1) vs minGRU (10.3)
   vs Mamba (10.4)** — the chapter's three answers, measured on one task. Code:
   selective-copy rerun showing Mamba solves what S4D could not.
4. **What a fixed state cannot do.** Honest-limits subsection: exact recall/
   copying needs state that grows with what you must recall; phone-book/"Repeat
   After Me" and MQAR findings summarized (cite; no experiments — our scale is
   too small to demo this cleanly, say so). Gu's brains-vs-databases framing:
   compressed always-on memory vs lossless growing log. Neither dominates.
5. **The recurrent frontier.** One page, prose: hybrids won in practice —
   interleaved attention/SSM layers (Jamba, Griffin/RecurrentGemma, Qwen-Next-
   style GDN ratios ~3–7:1); the sibling lineages (RWKV, xLSTM, GLA/DeltaNet) as
   one paragraph with the unifying "different state-update rules on a matrix
   state" sentence; Mamba-2's exact SSM↔attention correspondence *named* and
   explicitly deferred until after attention is taught; where SSMs win outright
   (raw/byte-level modalities: audio, DNA — one sentence each). Mamba-3 one-line
   footnote.
6. **Summary + handoff.** The chapter's arc restated in four sentences ending:
   the remaining option is to keep *everything* and learn what to look at —
   attention, next chapter. (This paragraph replaces the old chapter's
   beam-search ending as the bridge into ch. 11.)

Figures (generated): selective-copy task cartoon; Mamba block diagram; the
"three answers" summary strip (gate/linearize/select). Computed: selective-copy
curves (S4D vs LSTM vs Mamba), capstone comparison table.
Exercises: make only Δ selective (keep B,C fixed) and measure what breaks; state-
size sweep vs recall length on selective copy; swap SiLU/gate branch out;
byte-level (no BPE) Time Machine run — where does Mamba's advantage go?

---

## Downstream obligations (implementation phase, listed so nothing is lost)

1. **Crossref sweep**: `grep -rn` for labels of deleted files (`sec_beam-search`,
   `sec_machine_translation`, `sec_encoder-decoder`, `sec_gru`, `sec_bi_rnn`,
   `sec_deep_rnn`, `sec_rnn-scratch`, `sec_rnn-concise`) across all chapters and
   slides; most persist on new anchors per the plan above, the rest get retargeted.
2. **`_quarto.yml` + CHAPTER_NUMBERING**: update file lists for both chapters (slot
   numbers unchanged); scheduler config check for renamed/deleted notebooks
   (per repo memory: grep `runtime_env.py` MULTI_GPU_NOTEBOOKS — none of these
   are multi-GPU, but verify).
3. **Outputs store**: deleted files' outputs pruned; new/renamed sections need
   capture (CPU-feasible for most; sMNIST S4D and Mamba LM runs are the ones worth
   checking on the GPU box; MXNet capture skipped for 10.3/10.4).
4. **Slides**: every rewritten file needs its `<!-- slides -->` section rebuilt
   against new cell IDs (`tools/add_cell_ids.py` after content lands).
5. **Figure generators**: `tools/gen_rnn_figures.py`, `tools/gen_modern_rnn_figures.py`
   + restyling of ~10 carryover SVGs; run `figure-style-audit` per chapter.
6. **Bibliography**: ~20 new entries (list in §0).
7. **Coordination**: no code lands until the JAX→NNX refactor clears chapters 9–10;
   prose-only edits could proceed earlier on files the refactor doesn't touch, but
   the merge/delete operations above touch everything — safest to sequence after.
