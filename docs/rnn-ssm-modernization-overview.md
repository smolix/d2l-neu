# Chapters 9–10 Modernization: Overview for Discussion

*Draft for discussion, 2026-07-11. Based on a four-way research sweep: (1) 2024–2026
syllabi at Stanford (CS224n W2026, CS336 S2026, CS25), CMU (11-785 S2026, 11-711
S2026, 10-423), Berkeley (EECS 182 F2025, CS288 S2026), MIT 6.S191, Princeton
(COS 484, 597R), Cornell, GaTech, UT Austin; (2) the SSM teaching canon (Annotated
S4, Gu/Dao blogs and thesis, Grootendorst, Songlin Yang's linear-attention lineage);
(3) the modern NLP-fundamentals canon (SLP3 Jan-2026 draft, Karpathy minbpe/nanoGPT/
nanochat, CS336 assignments, HF LLM course, Raschka, Xiao & Zhu); (4) evidence on
where classic RNNs still matter in production/research.*

## 1. Diagnosis

The two chapters are frozen in 2021. Sixteen sections, ~40k words, organized around
a pipeline (word-level tokenization → n-grams → RNN char-LM → LSTM/GRU → small-corpus
word-level MT → beam search for MT) that no top course teaches anymore:

- **CS224n went from 4 lectures of this material (2019) to 1 (2026)**; GRU, bi-RNN,
  stacked RNNs, and beam-search-for-MT were cut outright. SLP3's Jan-2026 draft moved
  RNNs/LSTMs to **chapter 13, after transformers, post-training, and even MT**.
- MT as the organizing application is dead everywhere — replaced by "one LM, many
  tasks via next-token prediction." BLEU is a legacy footnote (chrF/COMET/LLM-judge
  now); word-level tokenization is obsolete (byte-level BPE is the universal
  from-scratch build target); beam search survives only as one point on a general
  decoding-strategy spectrum.
- What did **not** disappear, anywhere that RNNs are taught at all: autoregressive
  factorization, BPTT + vanishing/exploding gradients, gating (LSTM), the
  hidden-state-as-compressed-memory idea, and perplexity. That's the durable core.
- Meanwhile the genuinely modern recurrence story — linear RNNs, S4/Mamba, the
  gated-recurrence layers inside production hybrid LLMs (Jamba, Qwen-Next ~3:1
  GDN:attention, MiniMax, Griffin/RecurrentGemma) — has **no textbook-scale home**.
  Courses either bolt on one late "efficient alternatives" guest lecture (CMU, CS25)
  or reach Mamba-2 via linear attention with no RNN connection at all (CS336).
  Berkeley 182 is the lone course teaching RNN → SSM → transformer as one arc.

**The opportunity:** no textbook currently teaches the unified recurrence lineage —
*nonlinear recurrence (RNN/LSTM) → linearized recurrence (minGRU/LRU) → structured
state spaces (S4) → selective state spaces (Mamba) → the fixed-state limitation that
motivates attention*. Every piece of that arc is now well-understood pedagogically;
nobody has assembled it. d2l can be first, and the arc gives basic RNNs a *modern*
reason to exist rather than a nostalgic one.

## 2. Proposed shape

Two chapters, same slots, retitled. Net: 16 sections → ~13, ~same total length but
redistributed. Every section keeps the d2l signature (runnable code that computes
something, scratch→concise where warranted).

### Chapter 9 — Sequence Models and Language Models
*"Everything you need to build, train, evaluate, and sample from a small language
model, with recurrence as the first architecture that does it well."*

| # | Section | Fate vs today |
|---|---------|---------------|
| 9.1 | Working with sequences | **Keep, tighten** (autoregressive factorization, Markov, stationarity; the AR toy example stays — one-step vs multi-step rollout error is load-bearing for LLMs and world models) |
| 9.2 | From text to tokens | **Rewrite.** Chars/words as strawmen → **byte-level BPE built from scratch** (train/encode/decode, ~Karpathy-minbpe scale), verify against tiktoken; fertility across languages, glitch tokens, why-LLMs-can't-spell as exercises |
| 9.3 | Language models | **Rewrite.** LM-as-universal-interface framing; n-grams compressed to a *computed baseline* (not a section of smoothing lore); Zipf; **perplexity and bits-per-byte** (tokenizer-agnostic comparison); data partitioning |
| 9.4 | Recurrent neural networks | **Keep** (hidden state, unrolling, char/BPE LM) |
| 9.5 | RNN implementation | **Merge** scratch + concise into one section; train an LM on the BPE tokens from 9.2 |
| 9.6 | Training: BPTT and gradient pathologies | **Keep core, cut mechanics.** Vanishing/exploding analysis, clipping; drop randomized truncation; add forward pointers (truncation→activation checkpointing, exploding gradients→why residuals) |
| 9.7 | Decoding and generation | **New.** Greedy → temperature → top-k/top-p/min-p (Holtzman) → beam search (incl. the large-beam curse) → brief speculative-decoding teaser. Absorbs and generalizes old 10.8. No current course teaches this as one coherent unit; SLP3 splits it — this is a lead-the-field section |

### Chapter 10 — Gated and Linear Recurrence *(working title; alt: "Modern Recurrent Architectures")*
*"Take the recurrence idea seriously: gate it, then linearize it, and it scales —
until you need exact recall, which is where attention comes in."*

| # | Section | Fate vs today |
|---|---------|---------------|
| 10.1 | LSTM: learning to gate | **Keep, central.** Gating as the chapter's big idea; **GRU folded in as a variant subsection**; deep/bidirectional RNNs reduced to short subsections (bi-RNN = one page + ELMo→BERT pointer). Kill standalone 10.2/10.3/10.4 |
| 10.2 | Encoder–decoder and seq2seq | **Merge 3 sections into 1** (old MT-dataset + encoder-decoder + seq2seq). Keep the abstraction (still how Whisper/multimodal systems are framed), teacher forcing = LLM pretraining's mechanism, a compact trained example with BPE tokens; BLEU demoted to a remark (chrF/COMET named). Ends on the **bottleneck**: the whole source squeezed through one fixed vector |
| 10.3 | Linear recurrence and state space models | **New.** Bridge: make LSTM/GRU gates input-only (minGRU/minLSTM, "Were RNNs All We Needed?") → the recurrence becomes linear → parallel scan trains it like a CNN. Then the principled version: continuous SSM → ZOH discretization (Δ *is* a gate) → recurrent↔convolutional duality → HiPPO stated (not derived) → S4D. Long-range benchmarks as the payoff |
| 10.4 | Selective state space models (Mamba) | **New.** Content-blindness of LTI systems → input-dependent (Δ, B, C) = selectivity → conv view dies, associative scan + hardware-awareness save it → the Mamba block; **honest limits** (fixed-state recall/copying, MQAR, "Repeat After Me"; Gu's brains-vs-databases framing) → hybrid LLMs (Jamba/Qwen-Next/Griffin) as where this landed in production. Closing handoff: *the fixed-size state is the same bottleneck as 10.2's, at scale — next chapter, attention* |

The chapter now has **one** villain (fixed-size state) introduced twice at growing
sophistication, and the attention chapter's motivation writes itself. Mamba-2/SSD
("transformers are SSMs") and the linear-attention unification (GLA/DeltaNet) are
deliberately **deferred to the attention part** — they require attention to state.

### The third piece: NLP after transformers

The pretraining→post-training→prompting→evaluation arc **cannot live in ch. 9–10**
(it needs transformers) and cannot stay a single overloaded section
(`large-pretraining-transformers.md`). Proposal: split it out as a **new chapter
right after the attention/transformers chapter — "Large Language Models"** — with
roughly: pretraining (data, objective, scaling laws at Chinchilla-punchline depth) ·
post-training (SFT; preference learning with **DPO derived**, PPO described — this is
what's actually taught now) · prompting and in-context learning · inference (KV cache,
speculative decoding, connecting back to 9.7) · evaluation (perplexity→MMLU-style→
HELM→LLM-as-judge, plus the Wei-vs-Schaeffer emergence debate as a methodology case
study) · and the SSM revisit (SSD/linear-attention duality, hybrid architectures).
Chapters 9–10 are written to feed it: tokenization, perplexity/bpb, decoding, teacher
forcing, and the fixed-state-vs-KV-cache contrast all get set up in advance.

## 3. What gets cut (and why it's safe)

| Cut | Justification |
|---|---|
| GRU as a standalone section | Not mentioned in CS224n 2026 or MIT 6.S191 at all; folded into LSTM as variant + minGRU pointer (where it returns, linearized, in 10.3) |
| Deep-RNN, bi-RNN standalone sections | Taught nowhere as standalone; one subsection each inside 10.1 |
| MT dataset engineering (fra-eng loading/bucketing) | Word-level small-corpus MT is dead pedagogy; a compact example survives inside 10.2 |
| Beam search as an MT section | Relocated into 9.7 as one decoding strategy among several |
| Randomized/truncated-BPTT mechanics, second-order details | Core intuition kept; mechanics replaced by a checkpointing forward-pointer |
| n-gram smoothing lore (Laplace etc.) | Compressed to a computed baseline; smoothing is appendix-grade in 2026 |
| BLEU as a taught metric | Demoted to a remark; chrF/COMET named as current practice |

Nothing on the durable-core list (AR factorization, BPTT, gating, encoder-decoder,
perplexity, hidden-state-as-memory) is cut — each is *re-motivated* by where it
leads (LLM pretraining, checkpointing, Mamba selectivity, Whisper, eval, SSM state).

## 4. Implementation notes / risks (flagged, not blocking)

- **Framework tabs.** S4D/Mamba teaching implementations need an associative scan.
  JAX has `lax.associative_scan` natively (and the NNX rewrite makes JAX a natural
  showcase here); PyTorch can use a short hand-rolled log-depth scan or the
  sequential form at teaching scale; TF likewise via `tf.scan`. MXNet will need the
  sequential/quadratic fallback — fine at textbook scale, worth deciding early
  whether the SSM sections are 4-tab or (like some existing sections) reduced-tab.
- **Datasets.** Time Machine stays for BPE + RNN-LM (it's the right size). Worth
  considering TinyStories as an optional "scale it up" exercise dataset. The fra-eng
  MT data survives only inside 10.2, if at all.
- **Reference implementations to adapt:** Annotated S4 (JAX), `mamba-minimal`
  (PyTorch, single-file), `BorealisAI/minRNNs` (minGRU — nearly a diff against our
  existing GRU cell), `minimal-LRU`.
- **Citations to anchor the new material:** S4 (Gu et al. 2022), S4D, HiPPO, Mamba
  (Gu & Dao 2023), Mamba-2 (Dao & Gu 2024), LRU (Orvieto et al. 2023), minGRU/minLSTM
  (Feng et al. 2024), Griffin (De et al. 2024), Jamba, "Repeat After Me" (Jelassi et
  al. 2024), MQAR (Arora et al.), Holtzman et al. 2020, min-p (ICLR 2025), Gu's
  "Tradeoffs" essay (2025). Mamba-3 (ICLR 2026) as a forward-pointing remark only.

## 5. Open questions for discussion

1. **SSM placement.** Recommended: before attention, inside ch. 10 (the arc above).
   The majority course pattern is *after* transformers as "efficient alternatives";
   Berkeley 182 and the Dao/Gu expositions show before-attention works and needs no
   attention machinery. Before-attention is the differentiating move; after would be
   safer but forfeits the unified-recurrence thesis and weakens ch. 10's ending.
2. **One SSM section or two?** Two (10.3 + 10.4) recommended: S4-lineage (linear,
   LTI, scan/conv) and Mamba (selective) are distinct ideas and each carries a real
   implementation. One mega-section is possible but would be the longest in the book.
3. **Scope of the "Large Language Models" chapter.** Green-lighting ch. 9–10 implies
   committing to it (they set up hooks for it). Is a new chapter acceptable, or
   should it be an expanded multi-section back half of the attention chapter?
   Numbering churn is nontrivial (CHAPTER_NUMBERING, labels, crossrefs).
4. **Does any MT example survive?** Options: (a) compact fra-eng seq2seq inside 10.2
   with BPE (recommended — we need *some* trained encoder–decoder, and MT is still
   the cleanest teachable seq2seq task); (b) replace with a synthetic task
   (copy/reverse — cleaner but sterile); (c) drop trained seq2seq entirely and keep
   the abstraction only (what CS224n did — but d2l's identity is runnable code).
5. **Scratch/concise merge in 9.5** breaks a long-standing d2l pattern — acceptable?
6. **Chapter titles.** "Gated and Linear Recurrence" vs "Modern Recurrent
   Architectures" vs keeping "Modern Recurrent Neural Networks".
