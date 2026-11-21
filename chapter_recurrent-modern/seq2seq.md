# Sequence-to-Sequence Learning
:label:`sec_seq2seq`

The language models of the previous chapters continue a single sequence: they read text and predict the next token. Many tasks map one sequence to another of a different length, where positions do not align one-to-one: a translator reads an English sentence and writes a French one, or a speech recognizer transcribes an audio clip into words. Translation was the benchmark on which neural networks first rivaled statistical phrase-based systems in 2014 :cite:`Sutskever.Vinyals.Le.2014,Cho.Van-Merrienboer.Gulcehre.ea.2014`. Their architecture was simple: one recurrent network reads the source, a second writes the target, and a single fixed-width state vector passes between them.

You have already met the successors of this design—cross-attention (:numref:`chap_attention`) and the Transformer encoder--decoder (:numref:`sec_transformer`)—which were invented to bypass its single-vector bottleneck. Why revisit the 2014 design here? Because that bottleneck is the central subject of this chapter: every recurrent model in the sections that follow compresses its past into a state of fixed size. Here we set up the encoder--decoder abstraction and the standard lexical translation metrics (`chrF` and `BLEU`), and then measure directly on a string-reversal benchmark how much information a single fixed state vector can carry as the input grows.

```{.python .input #seq2seq-imports}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #seq2seq-imports}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #seq2seq-encoder-decoder-models-for-sequence-transduction-1}
%%tab pytorch, jax
import collections
```

## The Encoder--Decoder Abstraction
:label:`sec_encoder-decoder`

Mapping one sequence to another of variable length (*sequence-to-sequence learning* or *sequence transduction*) splits naturally into two modules (:numref:`fig_encoder_decoder`): an *encoder* that consumes the source sequence and summarizes it into a state, and a *decoder* that consumes that state and generates the target one token at a time. Everything the decoder learns about the source must pass through that state.

![The encoder--decoder architecture: the encoder reads the source sequence into a state, and the decoder generates the target sequence from that state.](../img/mdl-modernrnn-encoder-decoder.svg)
:label:`fig_encoder_decoder`

Given a source $x_1, \ldots, x_T$ and a target $y_1, \ldots, y_{T'}$ (where $T'$ need not equal $T$), the decoder factorizes the conditional probability autoregressively just like a language model (:numref:`sec_language-model`), except that every factor is conditioned on the encoder state $\mathbf{c}$:

$$P(y_1, \ldots, y_{T'} \mid x_1, \ldots, x_T) = \prod_{t'=1}^{T'} P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c}).$$
:eqlabel:`eq_seq2seq_conditional`

Notice that the source sequence disappears on the right-hand side, leaving only $\mathbf{c}$: the equation holds only if $\mathbf{c}$ retains everything about the source that the decoder will need. During training we minimize next-token cross-entropy under teacher forcing (:numref:`subsec_teacher-forcing`), feeding ground-truth previous target tokens; at inference time the decoder feeds back its own emitted tokens until it produces an end-of-sequence token.

Two groups introduced this recurrent design in 2014. :citet:`Cho.Van-Merrienboer.Gulcehre.ea.2014` used a GRU encoder--decoder (:numref:`sec_gru`) to score phrase pairs inside a statistical translation system, while :citet:`Sutskever.Vinyals.Le.2014` translated English to French end-to-end with deep LSTMs connected through a single vector. Notably, Sutskever et al. fed the source sentence to the encoder *backward* so that the first words of the translation sat only a few recurrent steps from the corresponding source words—shortening the gradient path without changing how much total information the state had to hold.

While recurrent encoders and decoders were later superseded by Transformers (such as Whisper for speech :cite:`radford2023whisper` or vision-language captioners), the encoder--decoder abstraction endured. What changed in :numref:`sec_transformer` is that the state passed to the decoder became a sequence of one vector per source token rather than a single vector for the entire source. Here we keep the 2014 single-vector state because its capacity limit is the exact quantity we want to measure.

## Scoring Translations
:label:`sec_translation-scoring`

Because human evaluation is too slow to run inside a training loop, machine translation models are scored by lexical $n$-gram overlap against a reference translation. The 2014 papers reported BLEU; modern evaluation benchmarks recommend character-level chrF in its place :cite:`Popovic.2015,Kocmi.Federmann.Grundkiewicz.ea.2021`. We implement both here and reuse them in :numref:`sec_lm_scoring`.

**chrF** compares character $n$-grams (with spaces stripped) for orders $n = 1, \ldots, 6$, avoiding tokenization sensitivity and granting partial credit for shared stems or near-miss inflections. For each order $n$, precision is the fraction of candidate $n$-grams present in the reference, and recall is the fraction of reference $n$-grams present in the candidate (using multiset intersection `p & r` so repeated $n$-grams are clipped at their reference count). Averaging precision and recall across the six orders to obtain $P$ and $R$, chrF computes the $F_\beta$-score:

$$F_\beta = \frac{(1 + \beta^2)\, P R}{\beta^2 P + R}.$$

With $\beta = 2$, recall weighs more heavily than precision.

```{.python .input #seq2seq-chrf}
%%tab pytorch, jax
def chrf(pred, label, n=6, beta=2):  #@save
    """chrF (Popovic, 2015): character n-gram F-score."""
    def ngrams(s, k):
        s = s.replace(' ', '')
        return collections.Counter(s[i:i + k] for i in range(len(s) - k + 1))
    prec = rec = 0.0
    for k in range(1, n + 1):
        p, r = ngrams(pred, k), ngrams(label, k)
        overlap = sum((p & r).values())
        if sum(p.values()) and sum(r.values()):
            prec += overlap / sum(p.values())
            rec += overlap / sum(r.values())
    prec, rec = prec / n, rec / n
    if prec + rec == 0:
        return 0.0
    return (1 + beta**2) * prec * rec / (beta**2 * prec + rec)
```

**BLEU** :cite:`Papineni.Roukos.Ward.ea.2002` measures clipped word $n$-gram precisions $p_n$ ($n = 1, \ldots, k$) and multiplies their weighted geometric mean by a *brevity penalty* so short, high-precision fragments cannot game the score:

$$\text{BLEU} = \exp\!\left(\min\!\left(0, 1 - \frac{\text{len}_{\text{label}}}{\text{len}_{\text{pred}}}\right)\right) \prod_{n=1}^{k} p_n^{1/2^n}.$$
:eqlabel:`eq_bleu`

Because it is a product, a single zero precision $p_n = 0$ at any order $n \le k$ drives sentence-level BLEU to $0$. (Standard corpus-level BLEU weights orders $n \le 4$ equally by $1/4$ and pools counts across an entire test set, where zero counts are rare.)

```{.python .input #seq2seq-bleu}
%%tab pytorch, jax
def bleu(pred_seq, label_seq, k):  #@save
    """Compute the BLEU."""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, min(k, len_pred) + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score
```

To see how the two metrics diverge on a single sentence, we score an exact copy, a near miss with one wrong inflection on the third of six words, and a valid paraphrase. Because every 4-gram in a six-word sentence touches the third word, $p_4 = 0$ on the near miss, driving BLEU with $k = 4$ to zero while chrF loses only the character $n$-grams overlapping the stray *s*.

```{.python .input #seq2seq-scoring-translations-1}
%%tab pytorch, jax
reference = 'we will visit our grandparents tomorrow'
candidates = {'copy': reference,
              'near miss': 'we will visits our grandparents tomorrow',
              'paraphrase': 'we are going to see our grandparents tomorrow'}
for name, candidate in candidates.items():
    print(f'{name:10s}  chrF {chrf(candidate, reference):.2f}  '
          f'BLEU {bleu(candidate, reference, k=4):.2f}')
```

The near miss scores 0.91 on chrF and 0.00 on BLEU, while the meaning-preserving paraphrase scores 0.67 on chrF and 0.00 on BLEU. Because both metrics count surface overlap rather than semantics, modern evaluation pairs chrF (or corpus-level sacreBLEU :cite:`Post.2018` for historical comparisons) with learned neural evaluators such as COMET :cite:`Rei.Stewart.Farinha.ea.2020` or LLM judges.

## The Fixed-State Bottleneck

How much of an input sequence fits through a single state vector? Natural language confounds length with syntactic rarity, so we isolate state capacity using the synthetic string-reversal benchmark of :numref:`subsec_reversal-task`: reversing a random string of $T$ independent letters drawn uniformly from an alphabet of 16 symbols. Because the letters are independent, previously decoded letters give zero information about the next letter—every letter must travel through the encoder's final state $\mathbf{c}$. Reversal also naturally gives the last encoder token first to the decoder, matching Sutskever et al.'s reversed-input ordering.

Each minibatch draws fresh random strings. `Reversal` packs the source string and the right-shifted reversed target (prefixed by a start token `V`) into a single tensor `X` for `d2l.train_steps`.

```{.python .input #seq2seq-the-fixed-state-bottleneck-1}
%%tab pytorch
class Reversal:
    """Fresh batches of random strings and their reversals."""
    def __init__(self, length, batch_size=128, num_letters=16):
        self.length, self.batch_size = length, batch_size
        self.num_letters = num_letters

    def batch(self, batch_size):
        V = self.num_letters  # Letters 0 to V - 1; V is the start token
        src = torch.randint(0, V, (batch_size, self.length),
                            device=d2l.try_gpu())
        tgt = src.flip(1)
        dec_in = torch.cat([torch.full_like(tgt[:, :1], V), tgt[:, :-1]], 1)
        return torch.cat([src, dec_in], 1), tgt

    def train_dataloader(self):
        while True:
            yield self.batch(self.batch_size)
```

```{.python .input #seq2seq-the-fixed-state-bottleneck-1}
%%tab jax
class Reversal:
    """Fresh batches of random strings and their reversals."""
    def __init__(self, length, batch_size=128, num_letters=16):
        self.length, self.batch_size = length, batch_size
        self.num_letters = num_letters

    def batch(self, batch_size, key):
        V = self.num_letters  # Letters 0 to V - 1; V is the start token
        src = jax.random.randint(key, (batch_size, self.length), 0, V)
        tgt = src[:, ::-1]
        dec_in = jnp.concatenate([jnp.full_like(tgt[:, :1], V),
                                  tgt[:, :-1]], 1)
        return jnp.concatenate([src, dec_in], 1), tgt

    def train_dataloader(self):
        key = d2l.get_key()
        while True:
            key, subkey = jax.random.split(key)
            yield self.batch(self.batch_size, subkey)
```

In raw bit count, each of the 16 equiprobable letters carries $\log_2 16 = 4$ bits, so a length-$T$ string carries $4T$ bits (32 bits at $T = 8$; 256 bits at $T = 64$). In principle, 64 single-precision floats occupy 2,048 bits—room for 512 letters. How many letters can a trained GRU actually pack and unpack through 64 or 256 floats?

Following :numref:`fig_seq2seq_details` and :citet:`Cho.Van-Merrienboer.Gulcehre.ea.2014`, the GRU encoder's final state $\mathbf{c}$ both initializes the GRU decoder ($\mathbf{s}_0 = \mathbf{c}$) and is concatenated to the input embedding at every decoding step:

$$\mathbf{s}_{t'} = g(y_{t'-1}, \mathbf{c}, \mathbf{s}_{t'-1}).$$
:eqlabel:`eq_seq2seq_s_t`

![A GRU encoder--decoder on reversal, unrolled. The encoder reads the source; its final state $\mathbf{c}$ starts the decoder and enters every decoder step. The decoder reads the reversed string shifted right behind a start token and predicts the reversed string.](../img/mdl-modernrnn-seq2seq.svg)
:label:`fig_seq2seq_details`

```{.python .input #seq2seq-the-fixed-state-bottleneck-2}
%%tab pytorch
class Reverser(nn.Module):
    """A GRU encoder whose final state starts and conditions a GRU decoder."""
    def __init__(self, num_letters, num_hiddens, num_inputs=32):
        super().__init__()  # One more embedding: the start token
        self.embedding = nn.Embedding(num_letters + 1, num_inputs)
        self.encoder = nn.GRU(num_inputs, num_hiddens, batch_first=True)
        self.decoder = nn.GRU(num_inputs + num_hiddens, num_hiddens,
                              batch_first=True)
        self.head = nn.Linear(num_hiddens, num_letters)
        d2l.init_lecun(self)
        nn.init.normal_(self.embedding.weight, std=num_inputs ** -0.5)

    def forward(self, X):  # X: the source, then the decoder input
        src, dec_in = X.chunk(2, dim=1)
        _, c = self.encoder(self.embedding(src))  # (1, batch, num_hiddens)
        context = c[0, :, None].expand(-1, dec_in.shape[1], -1)
        dec_X = torch.cat([self.embedding(dec_in), context], -1)
        return self.head(self.decoder(dec_X, c)[0])
```

```{.python .input #seq2seq-the-fixed-state-bottleneck-2}
%%tab jax
class Reverser(nnx.Module):
    """A GRU encoder whose final state starts and conditions a GRU decoder."""
    def __init__(self, num_letters, num_hiddens, num_inputs=32, *, rngs):
        lecun = {'recurrent_kernel_init': nnx.initializers.lecun_normal()}
        self.embedding = nnx.Embed(num_letters + 1, num_inputs, rngs=rngs,
            embedding_init=nnx.initializers.normal(num_inputs ** -0.5))
        self.encoder = nnx.RNN(nnx.GRUCell(
            num_inputs, num_hiddens, rngs=rngs, **lecun), return_carry=True)
        self.decoder = nnx.RNN(nnx.GRUCell(
            num_inputs + num_hiddens, num_hiddens, rngs=rngs, **lecun))
        self.head = nnx.Linear(num_hiddens, num_letters, rngs=rngs)

    def __call__(self, X):  # X: the source, then the decoder input
        src, dec_in = jnp.split(X, 2, axis=1)
        c, _ = self.encoder(self.embedding(src))  # (batch, num_hiddens)
        context = jnp.broadcast_to(c[:, None], (*dec_in.shape, c.shape[-1]))
        dec_X = jnp.concatenate([self.embedding(dec_in), context], -1)
        return self.head(self.decoder(dec_X, initial_carry=c))
```

We train each model for 1,000 steps on minibatches of 128 fresh strings using AdamW with a cosine decay from $3 \times 10^{-3}$ to zero (`d2l.LMOptimizer` and `d2l.train_steps`), matching :numref:`sec_transformer`. We evaluate per-letter accuracy on 1,000 fresh strings under teacher forcing: because the letters are independent, feeding ground-truth previous tokens leaks nothing about the next target letter while preventing early decoding errors from confounding the state-capacity measurement.

```{.python .input #seq2seq-the-fixed-state-bottleneck-3}
%%tab pytorch
def reversal_accuracy(num_hiddens, length):
    """Train a fresh model for 1,000 steps; return its per-letter accuracy."""
    data = Reversal(length)
    model = Reverser(data.num_letters, num_hiddens).to(d2l.try_gpu())
    optimizer = d2l.LMOptimizer(model, lr=3e-3, num_steps=1000, warmup=0,
                                weight_decay=0.0)
    d2l.train_steps(model, data, optimizer, 1000)
    X, Y = data.batch(1000)
    with torch.inference_mode():
        return (model(X).argmax(-1) == Y).float().mean().item()
```

```{.python .input #seq2seq-the-fixed-state-bottleneck-3}
%%tab jax
def reversal_accuracy(num_hiddens, length):
    """Train a fresh model for 1,000 steps; return its per-letter accuracy."""
    data = Reversal(length)
    model = Reverser(data.num_letters, num_hiddens,
                     rngs=nnx.Rngs(d2l.get_key()))
    optimizer = d2l.LMOptimizer(model, lr=3e-3, num_steps=1000, warmup=0,
                                weight_decay=0.0)
    d2l.train_steps(model, data, optimizer, 1000)
    X, Y = data.batch(1000, d2l.get_key())
    return float((model(X).argmax(-1) == Y).mean())
```

We run two seeds for state widths 64 and 256 across lengths $T \in \{8, 16, 32, 64\}$ (chance accuracy is $1/16 \approx 0.06$).

```{.python .input #seq2seq-the-fixed-state-bottleneck-4}
%%tab pytorch, jax
lengths, widths = (8, 16, 32, 64), (64, 256)
print('width' + ''.join(f'{f"T = {T}":>12}' for T in lengths))
for num_hiddens in widths:
    row = [[reversal_accuracy(num_hiddens, T) for _ in range(2)]
           for T in lengths]
    print(f'{num_hiddens:5d}' + ''.join(f'   {a:.2f}/{b:.2f}' for a, b in row))
```

At width 64, accuracy drops from 0.99 at $T = 8$ to 0.72 at $T = 16$ and 0.26–0.27 at $T = 64$. Quadrupling the state width to 256 holds accuracy at 0.99 through $T = 16$ and recovers roughly half the letters at $T = 64$. Even so, effective capacity is a small fraction of raw floating-point bits: at $T = 16$, the string carries just 64 bits (1 bit per state float at width 64), yet a quarter of the letters are lost. Moreover, scaling state width proportionally with sequence length still loses ground ($T = 8$ at width 64 reaches 0.99, whereas $T = 32$ at width 256 reaches ~0.80), because the network must also propagate early letters across a longer chain of nonlinear recurrent updates.

Neural machine translation hit the exact same bottleneck in 2014 :cite:`Cho.Van-Merrienboer.Bahdanau.ea.2014`, prompting :citet:`Bahdanau.Cho.Bengio.2014` to introduce cross-attention so the decoder could look back at every encoder state (:numref:`sec_attention-scoring-functions`, :numref:`sec_transformer`)—solving length-12 reversal perfectly within 300 steps at the cost of a cache that grows linearly with input length. The remainder of this chapter explores the complementary path: keeping a fixed-size state, making it parallelizable to train, expanding it from a vector into a key--value matrix memory, and finally combining it with a small fraction of attention layers.

## Summary

The encoder--decoder architecture maps a variable-length source sequence to a variable-length target sequence by summarizing the source into an encoder state $\mathbf{c}$ and conditioning an autoregressive decoder on $\mathbf{c}$. Translation quality is commonly measured by character $n$-gram overlap (chrF) or word $n$-gram precision with a brevity penalty (BLEU), supplemented by learned semantic metrics. On synthetic string reversal, a single fixed-width GRU state vector degrades rapidly as sequence length grows, storing well under one effective bit per state coordinate and motivating both attention mechanisms and the structured higher-capacity recurrent states developed in the rest of this chapter.

## Exercises

1. **BLEU against chrF.** Score the near miss and the paraphrase with
   `bleu` for $k = 1, 2, 3, 4$, and explain each value from the $n$-gram
   counts. Then
   construct a candidate on which chrF and BLEU disagree as much as
   possible, in either direction, and explain which property of each
   metric the candidate exploits.
1. **Bits through the state.** The counting argument of this section says
   that a state of $N$ numbers of $b$ bits each can take at most $2^{Nb}$
   values.
    1. Show that reversing every string of $T$ letters over 16 letters
       without error requires $Nb \ge 4T$.
    1. Let $p$ be the per-letter accuracy of the decoder, measured as in
       this section with the correct previous letters as input. Use Fano's
       inequality (:numref:`subsec_mdl-fano`) and the chain rule of mutual
       information to show that at least
       $T\,(4 - H_\textrm{b}(1 - p) - (1 - p)\log_2 15)$ bits about the
       string came through $\mathbf{c}$, where $H_\textrm{b}$ is the binary
       entropy in bits.
    1. Evaluate this bound for each entry of the table and compare it with
       $32N$. How many bits per number does the trained encoder use, and
       does that number depend on $T$?
    1. Train the width-64 model four times longer at each length. Which
       lengths gain most, and does the fall with length disappear? What
       does your answer say about whether the state or the training sets
       the limit?
1. **Sutskever's trick, undone.** Our reversal task already has the
   layout that reversing the source gave Sutskever et al.: the first letter
   to write is the last one read. Undo it by feeding the encoder the source
   backward, so that the decoder has to copy the string in the order in
   which the encoder read it.
    1. For each letter, count the steps between reading it and writing it,
       in reversal and in copying. Which task has the shorter dependencies
       on average, and which has the shortest single dependency?
    1. Train both tasks at width 64 and $T = 16$, and measure the accuracy
       overall and at each output position. Which task is easier to learn?
       Does the answer agree with the argument of
       :citet:`Sutskever.Vinyals.Le.2014` for reversing the source?
1. **Context at every step.** Pass the context $\mathbf{c}$ to the
   decoder only as its initial state, as Sutskever et al. did, and not as
   an input at every step. How does that change the accuracies at each
   length, and why might repeating the context help a small model?

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/1062)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/18022)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §12.2]{.kicker}

Sequence-to-Sequence Learning<br>
**the encoder--decoder · scoring translations · the cost of one fixed state, measured**
:::
:::

::: {.slide title="One State Must Carry the Whole Source"}
[The problem]{.kicker}

- Translation reads a whole sentence, then writes another of a different
  length; positions do not line up.
- 2014: an **encoder** RNN reads the source into one state $\mathbf{c}$,
  a **decoder** RNN writes the target from $\mathbf{c}$.
- Reverse 64 random letters through a GRU state of 64 numbers, trained
  for 1,000 steps: about **1 letter in 4** comes out right. Guessing gets
  1 in 16.

::: {.d2l-note .warn}
How much of the source fits through one state, and what are the ways out?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- write the decoder as a conditional language model;
- score a translation with chrF and BLEU, and predict when BLEU is zero;
- count the bits a string carries and compare them with a state;
- read accuracy against string length for two state widths;
- name the two ways out: look back with attention, or build a better
  state.

**Running example:** random strings over 16 letters, to be reversed.
:::

::: {.slide title="This Deck Measures the Limit the Chapter Works Against"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Gated Recurrent Networks | gates decide what to keep |
| **Sequence-to-Sequence Learning** | **what one fixed state costs** |
| Linear Recurrences and Parallel Scans | training on all positions at once |
| State Space Models | how fast a state forgets |
| Selective State Space Models | the input decides what to keep |
| Matrix-State Recurrences | a state of key-value pairs |
| The Delta Rule | correcting a pair |
| Hybrid Architectures | a little attention back |
:::

::: {.col .narrow}
- **Uses:** the GRU, teacher forcing, cross-entropy.
- **Skips:** a trained translator, subword vocabularies, beam search. The
  modern encoder--decoder with cross-attention is in the transformer
  chapter.
:::
:::
:::

::: {.slide title="The State Is the Only Channel from Encoder to Decoder"}
[The abstraction]{.kicker}

![](../img/mdl-modernrnn-encoder-decoder.svg){width=88%}

- **Encoder:** reads the whole source, of any length, into a state.
- **Decoder:** writes the target one token at a time from that state.

::: {.d2l-note}
Everything the decoder learns about the source, it learns through the
state.
:::
:::

::: {.slide title="The Decoder Is a Conditional Language Model"}
[The abstraction]{.kicker}

$$P(y_1, \ldots, y_{T'} \mid x_1, \ldots, x_T) = \prod_{t'=1}^{T'} P(y_{t'} \mid y_{<t'}, \mathbf{c})$$

- Source $x_1, \ldots, x_T$; target $y_1, \ldots, y_{T'}$, with $T' \ne T$
  allowed; $y_{<t'}$ is the target written so far; $\mathbf{c}$ is the
  encoder state.
- Trained like a language model: next-token cross-entropy, correct
  previous tokens fed in (teacher forcing). At test time it feeds back its
  own tokens.
- Both halves as transformers, audio in: Whisper. A vision encoder:
  captioning.

::: {.d2l-note .warn}
The right side no longer sees the source, only $\mathbf{c}$. The equality
holds only if $\mathbf{c}$ keeps everything the decoder needs.
:::
:::

::: {.slide title="In 2014 Two Recurrent Networks Were Joined Through One Vector"}
[History]{.kicker}

- A GRU encoder--decoder scored phrase pairs inside a statistical
  translator.
- Deep LSTMs translated English to French end to end, through one vector.
- Their trick: feed the source **backward**, so the first target words sit
  a few steps from the source words they render. Short dependencies, the
  authors argued, train more easily.
- Soon measured: BLEU falls as sentences grow longer.

::: {.d2l-note}
Reversing the source does not change what the state must hold, only how
far each word travels through the recurrence.
:::

[Cho et al., EMNLP 2014; Sutskever, Vinyals, Le, NeurIPS 2014; Cho et al., SSST 2014]{.small}
:::

::: {.slide title="chrF Counts Characters, BLEU Counts Words"}
[Scoring]{.kicker}

- **Precision:** the share of the candidate's $n$-grams found in the
  reference; **recall:** the share of the reference's found in the
  candidate. Counts are clipped: a repeated word scores only as often as
  the reference has it.
- **chrF:** character $n$-grams, $n \le 6$, spaces removed; $P$ and $R$
  averaged over $n$; score $\frac{(1+\beta^2) P R}{\beta^2 P + R}$ with
  $\beta = 2$, so a missed $n$-gram costs more than an extra one.
- **BLEU:** $\text{BP} \prod_{n \le k} p_n^{w_n}$, with $p_n$ the
  precision of word $n$-grams and brevity penalty
  $\text{BP} = \exp(\min(0, 1 - \ell_\text{ref}/\ell_\text{cand}))$.
  Here $w_n = 1/2^n$, sentence by sentence; the standard uses $w_n = 1/4$
  for $n \le 4$, pooled over a test set.

::: {.d2l-note .rule}
Both score overlap with a reference translation, not meaning.
:::
:::

::: {.slide title="Check: What Is BLEU for the Near Miss?"}
[Check]{.kicker}

Reference: `we will visit our grandparents tomorrow`<br>
Near miss: `we will visits our grandparents tomorrow`

Predict BLEU with $k = 4$ before you compute anything.

::: {.fragment .fade-up}
::: {.d2l-note}
**Zero.** A six-word sentence has three word 4-grams, and every one of
them contains word 3. So $p_4 = 0$ and the product vanishes.
:::
:::
:::

::: {.slide title="chrF Gives the Near Miss 0.91, BLEU Gives It Zero"}
[Scoring]{.kicker}

@!seq2seq-scoring-translations-1

- Paraphrase: `we are going to see our grandparents tomorrow`.
- The correct paraphrase scores **below** the ungrammatical near miss on
  chrF and zero on BLEU.

::: {.d2l-note}
Use chrF for a quick lexical score, standard corpus BLEU (sacreBLEU) to
compare with older work, and a learned metric (COMET) or an LLM judge for
meaning.
:::
:::

::: {.slide title="Reversal Forces Every Letter Through the State"}
[The experiment]{.kicker}

![](../img/mdl-modernrnn-seq2seq.svg){width=86%}

- Random letters are independent: the letters already written say nothing
  about the next one.
- The encoder's final state $\mathbf{c}$ starts the decoder and is fed to
  every decoder step.
:::

::: {.slide title="Check: How Many Letters Could 64 Floats Hold?"}
[Check]{.kicker}

A letter from 16 carries 4 bits. The state is 64 numbers of 32 bits.

::: {.fragment .fade-up}
::: {.d2l-note}
$64 \times 32 = 2{,}048$ bits, so **512 letters**, if training could use
every bit. A string of 16 letters needs only 64 bits.
:::
:::
:::

::: {.slide title="Longer Strings Overflow a Fixed State Long Before Counting Says"}
[The result]{.kicker}

1,000 steps of 128 fresh strings per model, two models per entry;
per-letter accuracy on fresh strings (guessing: 0.06):

@!seq2seq-the-fixed-state-bottleneck-4

- The two runs of each entry agree to within 0.02.
- Width 64: 0.99 at 8 letters, about 0.73 at 16 (64 bits, one per
  number), about 0.26 at 64.
- Same letters per number, four times both: 8 through 64 gives 0.99, 32
  through 256 about 0.8. Longer is harder even with a wider state.

::: {.d2l-note}
The code that 1,000 steps of training find, not the 2,048 bits, sets the
limit.
:::
:::

::: {.slide title="Two Ways Out: Look Back, or Build a Better State"}
[Where next]{.kicker}

- **Attention:** let the decoder read every encoder state. A
  cross-attention encoder--decoder reverses fresh 12-letter strings without
  a single error after 300 steps.
- Its cost: a memory that grows with the input.
- **A better state:** keep one state of fixed size, train it fast, and make
  it a better memory. That is the rest of this chapter, until a little
  attention comes back in its last section.

[Bahdanau, Cho, Bengio, ICLR 2015]{.small}
:::

::: {.slide title="Recap"}
- Encoder--decoder: source $\to$ state $\to$ target; the decoder is a
  conditional language model.
- chrF 0.91 against BLEU 0 for one wrong inflection; neither measures
  meaning.
- 64 numbers of state: 0.99 of 8 letters reversed, about 0.26 of 64, far
  below the 512 letters counting allows; 256 numbers hold out longer.

**Next:** keep the single state, and make it train on all positions at
once.
:::
