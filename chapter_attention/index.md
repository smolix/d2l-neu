# Attention
:label:`chap_attention`

A recurrent network reads a sequence the way a clerk with one notepad
reads a ledger: each entry is folded into the running summary, and by the
end the summary is all that remains. :numref:`chap_rnn` built models of
exactly this shape and :numref:`chap_modern_rnn` will confront their
limit directly: a fixed-size state must eventually discard something,
and the model cannot know at reading time what it will be asked later.
Attention removes the constraint by refusing to summarize. It keeps
every entry, and when a question arrives it *looks the answer up*:
compare the question against a key describing each entry, convert the
comparison scores into weights, and return the weighted average of the
entries' values. The three names — query, key, value — come from
databases, and the analogy is exact except on one point: a database
returns the single best match, while attention returns a soft blend of
all of them. That one change makes lookup differentiable, and a
differentiable lookup can be *learned*.

This chapter is about that mechanism, on its own terms. The reason it
deserves a chapter is what became of it. Attention entered deep learning
in 2014 as a patch: machine-translation models were forcing whole
sentences through a fixed-size vector, and letting the decoder look back
at the source at every step repaired the bottleneck
:cite:`Bahdanau.Cho.Bengio.2014`. Three years later the patch swallowed
the architecture — :citet:`Vaswani.Shazeer.Parmar.ea.2017` discarded the
recurrence entirely and kept only attention, and the resulting
*transformer* is the subject of :numref:`chap_transformers`. The
mechanism then outgrew its birthplace: the models that read and write
text, classify and generate images, transcribe speech, and predict
protein structures are, at their core, stacks of the same lookup. Its
weights are a quantity the field measures, engineers around, and argues
about; its quadratic cost sets the economics of long-context models; and
its learned circuits are the best-understood fragments of what large
models actually compute. A mechanism with that reach is worth
understanding precisely, before any architecture is wrapped around it.

The six sections build it up in order. :numref:`sec_queries-keys-values`
states the lookup abstraction and shows that classical kernel regression
is attention with hand-picked weights — the case for learning them.
:numref:`sec_attention-scoring-functions` turns similarity into scores:
scaled dot products, why the $1/\sqrt{d}$ factor is a matter of softmax
saturation rather than convention, and masking, the small bookkeeping
device that lets one implementation serve padded batches and causal
language models alike. :numref:`sec_multihead-attention` runs several
attention functions in parallel and proves, on the smallest task that
defeats it, that a single head must average what a query asks for
separately while two heads recover it, then separates the two wirings —
self-attention, where a sequence queries itself, and cross-attention,
where one sequence queries another. :numref:`sec_positional-information` restores what the lookup
deliberately ignores: order. Attention is permutation-equivariant, so
position must be injected, and we follow the idea from sinusoidal
encodings to the rotary embeddings used by essentially every current
model, ending with an experiment on what happens beyond the training
length — whose outcome is not what the folklore promises.
:numref:`sec_attention-at-scale` faces the price: computing every
query–key pair is quadratic in sequence length, and we measure it, then
implement the three escapes: computing exact attention without ever
materializing the score matrix, restricting it to a window, and
linearizing it, which turns attention back into a recurrence and hands
the story to :numref:`chap_modern_rnn`. :numref:`sec_what-attention-computes`
closes by asking what trained attention layers actually do, and answers
with running code: a two-layer attention-only model visibly learns an
induction circuit — find the previous occurrence of the current token,
copy what followed it — and that small mechanism is the clearest
laboratory example of how in-context learning arises.

Along the way we train one model *architecture*: a character-level,
attention-only language model a few blocks deep, introduced in
:numref:`sec_positional-information` and trained afresh on a different
task in :numref:`sec_what-attention-computes`, with no run longer than
about a minute. Everything else executes in seconds on data chosen so
that the phenomenon under study is unmistakable. A word on what this
chapter is not. It builds no transformer (no feed-forward layers, no
normalization, no training recipes); that assembly, and the architecture
zoo around it, is :numref:`chap_transformers`. It says nothing about
optimizers, which have :numref:`chap_optimization` to themselves. And it
treats the efficient-attention literature with deliberate economy: a
decade of approximations is summarized in a paragraph, because the
variants that survived are the ones we implement. Where attention
touches state-space models, :numref:`chap_modern_rnn` owns the
recurrence side of the correspondence.

```toc
:maxdepth: 2

queries-keys-values
attention-scoring
multihead-attention
positional-information
attention-at-scale
what-attention-computes
```

## Resources and Further Reading {.unnumbered}

The references follow the chapter's arc: intuition for the mechanism,
the founding papers, the cost of attention, and the circuits view. All
are freely accessible online.

**Visual introductions**

- [Attention in transformers, step-by-step — 3Blue1Brown (2024)](https://www.3blue1brown.com/lessons/attention) — the geometric reading of :numref:`sec_queries-keys-values`: queries as questions, keys as advertisements, values as contributions, animated in embedding space; deliberately code-free, which this chapter's notebooks complement.
- [The Illustrated Transformer — Jay Alammar (2018)](https://jalammar.github.io/illustrated-transformer/) — the field's shared picture vocabulary for :numref:`sec_attention-scoring-functions` and :numref:`sec_multihead-attention`; the "it"-coreference attention pattern reproduced in half the talks you will ever see started here.
- [Transformers from scratch — Peter Bloem (2019)](https://peterbloem.nl/blog/transformers) — derives self-attention as a permutation-equivariant set operation before any architecture appears, the same order of ideas as :numref:`sec_positional-information`'s opening.

**Founding and load-bearing papers**

- [Neural Machine Translation by Jointly Learning to Align and Translate — Bahdanau et al. (2014)](https://arxiv.org/abs/1409.0473) — where the mechanism began, as the alignment story :numref:`sec_attention-scoring-functions` retells; the attention-weight heatmap over a translation originates here.
- [Attention Is All You Need — Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762) — scaled dot-product attention and multi-head attention as this chapter teaches them; the architecture half of the paper belongs to :numref:`chap_transformers`.
- [RoFormer: Enhanced Transformer with Rotary Position Embedding — Su et al. (2021)](https://arxiv.org/abs/2104.09864) — rotary embeddings as derived in :numref:`sec_positional-information`, now the default positional scheme of open models.
- [Train Short, Test Long — Press et al. (2022)](https://arxiv.org/abs/2108.12409) — ALiBi, and the extrapolation question that :numref:`sec_positional-information` puts to experiment.

**The cost of attention**

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness — Dao et al. (2022)](https://arxiv.org/abs/2205.14135) — the industrial form of :numref:`sec_attention-at-scale`'s chunked online-softmax cell: exact attention, quadratic time, linear memory, organized around memory traffic rather than arithmetic.
- [Transformers are RNNs — Katharopoulos et al. (2020)](https://arxiv.org/abs/2006.16236) — the linear-attention equivalence that :numref:`sec_attention-at-scale` verifies numerically, and the cleanest bridge between this chapter and :numref:`chap_modern_rnn`.
- [The Transformer Family v2.0 — Lilian Weng (2023)](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) — a taxonomy of the attention-variant design space; the map for which :numref:`sec_attention-at-scale` implements the surviving territory.

**What attention computes**

- [A Mathematical Framework for Transformer Circuits — Elhage et al. (2021)](https://transformer-circuits.pub/2021/framework/index.html) — the QK/OV factorization and residual-stream view that :numref:`sec_what-attention-computes` teaches in executable form.
- [In-context Learning and Induction Heads — Olsson et al. (2022)](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — the evidence, at model scale, for the phase change our two-block model reproduces in minutes.
- [Thinking Like Transformers (Transformer Puzzles) — Sasha Rush](https://github.com/srush/Transformer-Puzzles) — RASP puzzles on what a fixed-depth attention machine can compute at all — a complement to this chapter's exercises that requires no training.

**Exercises with teeth**

- [Stanford CS224n, self-attention and transformers assignment](https://web.stanford.edu/class/cs224n/) — the graded counterpart of this chapter: constructions for what single heads cannot do, the permutation-equivariance proof, and a RoPE relative-position derivation, all of which appear here as section material or exercises.
