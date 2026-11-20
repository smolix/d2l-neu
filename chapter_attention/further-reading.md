# Further Reading
:label:`sec_attention-further-reading`

The preceding sections developed one representative design for each layer
component: softmax lookup (:numref:`sec_queries-keys-values`), scaled
dot-product scoring and masking (:numref:`sec_attention-scoring-functions`),
multi-head and grouped-query self-attention (:numref:`sec_multihead-attention`),
positional encodings (:numref:`sec_positional-information`), tiled online
softmax (:numref:`sec_attention-at-scale`), sliding-window and linear
attention (:numref:`sec_windowed-linear-attention`), and mechanistic circuits
(:numref:`sec_what-attention-computes`). This section surveys closely related
variants, architectural refinements in modern open-weights models, and primary
literature.

## The Lookup and Its Normalization

### Attention as Associative Memory

:citet:`Ramsauer.Schafl.Lehner.ea.2021` connect softmax attention to classical
associative memories by showing that one step of softmax attention corresponds
to a single update of a continuous *modern Hopfield network* whose stored
patterns are the key/value rows. The inverse temperature ($1/\sigma^2$ in
kernel regression, $1/\sqrt{d}$ in dot-product attention) determines whether
the fixed points are individual stored patterns (high inverse temperature) or
metastable averages over multiple patterns (low inverse temperature). Using
this lens, they observe that early layers in pretrained encoders such as BERT
predominantly operate in the broad-averaging regime, whereas deeper layers
operate in the sharp-retrieval regime. Adaptive kernel bandwidths
:cite:`Silverman86` correspond to query- or head-dependent softmax
temperatures.

### Monotone Alignment

A year before :citet:`Bahdanau.Cho.Bengio.2014` introduced unconstrained
content-based alignment for machine translation, :citet:`Graves.2013`
introduced a differentiable *monotone* location-based attention mechanism for
handwriting synthesis, parameterizing a mixture of Gaussian windows whose mean
positions advance monotonically with each pen stroke—an inductive bias suited
to speech and handwriting (:cite:`rabiner1993fundamentals`) but too
restrictive for word-order inversions in translation
(:numref:`subsec_learned-alignment`).

### Weights That Need Not Sum to One
:label:`subsec_attention-fr-sinks`

Because a softmax row must sum to one, a query with no relevant keys often
dumps its probability mass onto the initial token—the *attention sink* of
:numref:`subsec_cross-attention-alignment` and
:numref:`sec_what-attention-computes` (see also :numref:`subsec_kv-cache-sinks`).
:citet:`Sun.Chen.Kolter.ea.2024` link attention sinks to massive outlier
activations in the residual stream that act as implicit bias terms, while
:citet:`Barbero.Arroyo.Gu.ea.2025` argue that sinks prevent representation
collapse in deep long-context transformers by providing a no-op target.

Several recent designs relax the unit-sum constraint so heads can idle without
an input token sink:

* **Sigmoid attention** :cite:`Ramapuram.Danieli.Dhekane.ea.2025` replaces
  the row-wise softmax with an elementwise logistic sigmoid shifted by
  $-\log n$, initializing weights near $1/(n + 1)$.
* **Gated attention** :cite:`Qiu.Wang.Zheng.ea.2025`, used in Qwen3-Next
  :cite:`Qwen.Team.2025`, multiplies each head's softmax output by an
  input-dependent sigmoid gate.
* **Learnable sink logits**, used in `gpt-oss` :cite:`OpenAI.2025`, append a
  trainable virtual key score to the softmax denominator with no associated
  value vector.
* **Differential attention** :cite:`Ye.Dong.Xia.ea.2025` subtracts two
  softmax distributions per head to cancel common-mode background weight on
  irrelevant keys.

## Self-Attention and Heads

### Origins and Head Redundancy

Self-attention evolved from *intra-attention* recurrences
:cite:`Cheng.Dong.Lapata.2016`, recurrence-free decomposable attention for
natural language inference :cite:`Parikh.Tackstrom.Das.ea.2016`, and
multi-row sentence embeddings with diversity penalties
:cite:`Lin.Feng.Santos.ea.2017`, before :citet:`Vaswani.Shazeer.Parmar.ea.2017`
unified projected multi-head self-attention in the Transformer.

While multiple heads are essential to overcome the single-mixture bottleneck
(:numref:`subsec_one-mixture-bottleneck`), trained models often exhibit
substantial head redundancy at test time: :citet:`Michel.Levy.Neubig.2019`
and :citet:`Voita.Talbot.Moiseev.ea.2019` showed that a large fraction of
heads in BERT and NMT models can be pruned post hoc with minimal degradation,
leaving a core of positional, syntactic, and rare-word heads. In modern
architectures, parameter and cache efficiency is achieved prior to training
via grouped-query attention (:numref:`subsec_kv-head-sharing`,
:numref:`sec_kv-cache-reduction`).

## Positions

### Learned Relative Positions

Before RoPE, relative offsets were injected via trainable per-offset key
embeddings :cite:`shaw2018self`, skew-indexed relative logits in the Music
Transformer :cite:`huang2018music`, or logarithmic distance-bucket scalar
biases in T5 :cite:`raffel2020exploring`. ALiBi replaced learned buckets with
fixed linear slopes, and RoPE achieved exact offset-only score dependence via
parameter-free coordinate rotations.

### RoPE in Trained Models
:label:`subsec_attention-fr-rope`

Open-weights models have progressively increased the RoPE base frequency $b$
to support longer contexts: from $b = 10^4$ in Llama 2 to $5 \times 10^5$ in
Llama 2 Long :cite:`Xiong.Liu.Molybog.ea.2024` and Llama 3
:cite:`Grattafiori.Dubey.Jauhri.ea.2024`, and $10^6$ in the global layers of
Gemma 3 :cite:`Gemma.Team.2025`. Increasing $b$ shifts more coordinate pairs
into the slow regime whose wavelength exceeds the training window
(:numref:`subsec_pos-context-extension`).
:citet:`Barbero.Vitvitskyi.Perivolaropoulos.ea.2025` show that trained models
use high-frequency RoPE pairs for positional heads (such as previous-token
heads) and low-frequency pairs as nearly position-invariant semantic
channels—motivating *p-RoPE*, which zeroes out the lowest frequencies entirely
so semantic channels do not drift at extreme distances. For post-training
context extension beyond YaRN :cite:`Peng.Quesnelle.Fan.ea.2024`, *LongRoPE*
:cite:`Ding.Zhang.Zhang.ea.2024` uses evolutionary search over per-dimension
rescale factors to extend context windows up to 2M tokens.

### Length Generalization and Long-Context Utilization

:citet:`Chi.Fan.Rudnicky.ea.2023` analyze ALiBi's effective receptive field,
corroborating our finding in :numref:`subsec_pos-experiment` that flat
perplexity extrapolation reflects local windowing rather than long-range
retrieval. For causal models without positional encodings (NoPE,
:cite:`Haviv.Ram.Press.ea.2022`), :citet:`Wang.Ji.Wu.ea.2024` show that
attention entropy grows with sequence length and can be counteracted by
per-head temperature scaling—a mechanism used at inference time in the NoPE
global layers of Llama 4 :cite:`Meta.2025`. Finally, low aggregate perplexity
on long documents does not guarantee effective retrieval across the full
window: :citet:`Liu.Lin.Hewitt.ea.2024` demonstrate a *"lost in the middle"*
phenomenon where accuracy drops significantly when relevant facts appear in
the middle of a long context rather than near the start or end (see
:numref:`sec_lm_long_context`).

## Cost

### Early Linear-Complexity Approximations

Prior to FlashAttention, numerous architectures approximated the $n \times n$
attention matrix: *Performer* :cite:`Choromanski.Likhosherstov.Dohan.ea.2021`
used positive orthogonal random features to approximate the exponential
kernel; *Linformer* :cite:`Wang.Li.Khabsa.ea.2020` applied low-rank sequence
projections; *Reformer* :cite:`Kitaev.Kaiser.Levskaya.2020` used
locality-sensitive hashing; and *Longformer* :cite:`beltagy2020longformer` and
*Sparse Transformer* :cite:`child2019generating` combined local windows with
strided or global tokens (surveyed by :citet:`Tay.Dehghani.Bahri.ea.2020`).

### Windows and Content-Adaptive Sparsity in Released Models

While Mistral 7B :cite:`Jiang.Sablayrolles.Mensch.ea.2023` applied sliding
windows in every layer (later dropped in :cite:`Mistral.AI.2023`), subsequent
architectures interleave local windows with periodic global layers: 1:1 in
Gemma 2 :cite:`Gemma.Team.2024`, 5:1 in Gemma 3 ($w = 1{,}024$,
:cite:`Gemma.Team.2025`), 1:1 with $w = 128$ in `gpt-oss`
:cite:`OpenAI.2025`, and chunked local layers paired with NoPE global layers
in Llama 4 :cite:`Meta.2025`.

Beyond fixed positional windows, hardware-aligned *content-based sparse
attention* has emerged for long-context training and decoding
(:numref:`sec_lm_sparse_attention`), including *Natively Sparse Attention*
(NSA, :cite:`Yuan.Gao.Dai.ea.2025`), *Mixture of Block Attention* (MoBA,
:cite:`Lu.Jiang.Liu.ea.2025`), and *DeepSeek Sparse Attention*
:cite:`Liu.Mei.Lin.ea.2025`.

### FlashAttention-2, FlashAttention-3, and FlexAttention

Following FlashAttention :cite:`Dao.Fu.Ermon.ea.2022`, *FlashAttention-2*
:cite:`Dao.2024` parallelizes across the query sequence dimension and defers
normalization until the end of the key loop. On Hopper GPUs, *FlashAttention-3*
:cite:`Shah.Bikshandi.Zhang.ea.2024` overlaps asynchronous Tensor Core matrix
multiplications with warp-specialized softmax exponentiation and supports FP8
via incoherent orthogonal pre-rotations. *FlexAttention*
:cite:`Dong.Feng.Guessous.ea.2025` compiles arbitrary score-modification
closures and block masks into fused Triton kernels, while parallel KV-cache
decoding kernels (:numref:`sec_lm_decode_kernels`) split the key axis and
reduce partial $(m_B, s_B, \mathbf{o}_B)$ states via online softmax.

### Gated Recurrences and Hybrids
:label:`subsec_attention-fr-gated`

Modern linear-attention descendants address memory capacity via two upgrades
taught in :numref:`chap_modern_rnn`: (1) data-dependent forget gates (*Gated
Linear Attention*, :cite:`Yang.Wang.Shen.ea.2024`, :numref:`sec_matrix-state`)
and (2) error-correcting delta updates (*DeltaNet*,
:cite:`Schlag.Irie.Schmidhuber.2021,Yang.Wang.Zhang.ea.2024`, and *Gated
DeltaNet*, :cite:`Yang.Kautz.Hatamizadeh.2024`, :numref:`sec_deltanet`).
Production architectures combine these recurrent layers with periodic full
softmax attention layers (typically 1 in 4 to 1 in 8 layers) to preserve
exact associative recall
:cite:`Qwen.Team.2025,Kimi.Team.2025b,MiniMax.2025` (:numref:`sec_hybrids`).

## Circuits in Large Models

### Activation Patching, Superposition, and Explanation

To go beyond zero/mean ablation (:numref:`subsec_head-ablation`), *activation
patching* substitutes activations from a clean run into a corrupted run to
isolate causal pathways :cite:`Meng.Bau.Andonian.ea.2022,Wang.Variengien.Conmy.ea.2023`,
automated via ACDC :cite:`Conmy.MavorParker.Lynch.ea.2023` or first-order
*attribution patching*
:cite:`Nanda.2023,Kramar.Lieberum.Shah.ea.2024,Zhang.Nanda.2024,Heimersheim.Nanda.2024`.
When features outnumber residual-stream dimensions (*superposition*,
:cite:`Elhage.Hume.Olsson.ea.2022`), *sparse autoencoders* and attribution
graphs disentangle polysemantic directions into monosemantic features
:cite:`Bricken.Templeton.Batson.ea.2023,Cunningham.Ewart.Riggs.ea.2024,Ameisen.Lindsey.Pearce.ea.2025`
(:numref:`sec_lm_interpretability`). These interventional tools also resolve
the classical debate between :citet:`Jain.Wallace.2019` and
:citet:`Wiegreffe.Pinter.2019` over whether attention weights constitute
faithful explanations.

### In-Context Learning as an Algorithm

Alongside induction-head copying :cite:`Olsson.Elhage.Nanda.ea.2022`,
transformers trained on in-context regression families can implement gradient
descent, ridge regression, and least-squares solvers directly in their forward
pass
:cite:`Garg.Tsipras.Liang.ea.2022,Oswald.Niklasson.Randazzo.ea.2023,Akyurek.Schuurmans.Andreas.ea.2023`,
though the extent to which general text pretraining induces implicit gradient
descent remains debated :cite:`Shen.Mishra.Khashabi.2024`.

## Recommended Resources

* **Visual introductions:** 3Blue1Brown, [*Attention in Transformers,
  Step-by-Step*](https://www.3blue1brown.com/lessons/attention) (2024); Jay
  Alammar, [*The Illustrated
  Transformer*](https://jalammar.github.io/illustrated-transformer/) (2018);
  Peter Bloem, [*Transformers from
  Scratch*](https://peterbloem.nl/blog/transformers) (2019); Lilian Weng,
  [*The Transformer Family Version
  2.0*](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/)
  (2023).
* **Foundational and circuit monographs:** Elhage et al., [*A Mathematical
  Framework for Transformer
  Circuits*](https://transformer-circuits.pub/2021/framework/index.html)
  (2021); Olsson et al., [*In-Context Learning and Induction
  Heads*](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
  (2022); Weiss, Goldberg, and Yahav, [*Thinking Like
  Transformers*](https://arxiv.org/abs/2106.06981) (2021) and Sasha Rush's
  companion [*Transformer
  Puzzles*](https://github.com/srush/Transformer-Puzzles).

[Discussions](https://d2l.discourse.group/t/36852)
