# Sequence-to-Sequence Learning
:label:`sec_seq2seq`

The language models of the previous chapters continue a sequence: they read
text and predict the next token. Many tasks have a different shape. A
translator reads an English sentence and writes a French one; a speech
recognizer listens to a clip and writes down the words. Input and output are
both sequences, but their lengths differ, and the third French word need not
render the third English one. Translation was the problem on which, in 2014,
neural networks first rivalled the statistical systems that then dominated the
field :cite:`Sutskever.Vinyals.Le.2014,Cho.Van-Merrienboer.Gulcehre.ea.2014`.
The design was simple. One recurrent network reads the source, a second writes
the target, and a single state passes between them.

You have met the successors of this design before the design itself. The
attention of :numref:`chap_attention` was invented to repair its weakness,
and the transformer encoder--decoder of :numref:`sec_transformer` replaced
it. Why, then, does it get a section in a chapter on recurrent networks?
Because its weakness is the subject of this chapter: one state of fixed
size stands between reading and writing, and every recurrent model in the
sections that follow keeps such a state. Here we can measure what that
costs. We first set up the encoder--decoder abstraction, which outlived the
networks that first filled it, and the scores by which one translation is
judged better than another. Then we train a small recurrent
encoder--decoder on a task in which every letter of the input must pass
through the state, and watch it fail as the input grows.

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

Mapping one sequence to another of a different length is called
*sequence-to-sequence learning*, or sequence transduction. The idea that
makes it tractable is to split the model in two
(:numref:`fig_encoder_decoder`). An *encoder* consumes the source sequence
and compresses it into a state. A *decoder* consumes that state and
produces the target, one token at a time. The state is the sole channel
between them: everything the decoder ever learns about the source, it
learns through that one object.

![The encoder--decoder architecture: the encoder reads the source sequence into a state, and the decoder generates the target sequence from that state.](../img/mdl-modernrnn-encoder-decoder.svg)
:label:`fig_encoder_decoder`

The decoder half should look familiar. Take a source $x_1, \ldots, x_T$
and a target $y_1, \ldots, y_{T'}$, whose length $T'$ need not equal $T$.
A language model (:numref:`sec_language-model`) factorizes the probability
of the target token by token. The decoder does the same, except that every
factor is also conditioned on the encoder's state $\mathbf{c}$:

$$P(y_1, \ldots, y_{T'} \mid x_1, \ldots, x_T) = \prod_{t'=1}^{T'} P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c}).$$
:eqlabel:`eq_seq2seq_conditional`

A decoder is thus a *conditional* language model: the same next-token
machinery, pointed at a source. Look at the right-hand side, though. The
source has disappeared from it, and only $\mathbf{c}$ is left. The
equation holds only if $\mathbf{c}$ keeps everything about the source that
the decoder will need, and the last part of this section measures how far
a state of fixed size falls short of that.

The decoder is trained like a language model, on the cross-entropy of each
next target token, with the teacher forcing of
:numref:`subsec_teacher-forcing`: it is fed the correct previous target
tokens rather than its own guesses. At test time there are no correct
tokens to feed. The decoder writes a token, reads it back as its next
input, and goes on until it writes an end-of-sequence token. Our
measurement below keeps the correct tokens, for a reason we give there.

Two groups built this design in 2014.
:citet:`Cho.Van-Merrienboer.Gulcehre.ea.2014` used an encoder--decoder made
of the gated units of :numref:`sec_gru` to score phrase pairs inside a
statistical translation system. :citet:`Sutskever.Vinyals.Le.2014` let the
design translate on its own: a deep LSTM read the English sentence, a
second deep LSTM wrote the French one, and the state between them was a
single vector. One detail of their recipe looks like a hack and is in fact
instructive. They fed the source sentence to the encoder backward. The
first words of the translation then sit a few steps away from the source
words they render, instead of a whole sentence away, and these short
dependencies, the authors argued, made the network much easier to train.
Reading backward changes nothing about how much the state must hold. It
changes how far each word travels through the recurrence before it is
needed, and that distance turned out to matter too.

The recurrent networks of that design did not last, but the abstraction
did. Make both halves transformers and put a small convolutional front-end
before the encoder, and you have Whisper transcribing speech
:cite:`radford2023whisper`; put a vision backbone in place of the encoder,
and you have an image-captioning model. What changed is the state. In the
encoder--decoder of :numref:`sec_transformer` it is one vector per source
token rather than one vector for the whole source. That section builds the
wiring as it is practiced today. This one keeps the 2014 version, because
its single vector is the object we want to measure.

## Scoring Translations
:label:`sec_translation-scoring`

We said that in 2014 neural networks rivalled the statistical systems.
Rivalled by what number? A human judge is the gold standard, and far too
slow to consult after every epoch, so the field scores a candidate by its
overlap with a reference translation. The comparisons of 2014 were made in BLEU.
The overlap score that evaluation studies now recommend in its place is
chrF :cite:`Popovic.2015,Kocmi.Federmann.Grundkiewicz.ea.2021`. We
implement both, and :numref:`sec_lm_scoring` uses them again on text that
language models write.

chrF compares *character* $n$-grams rather than words. This sidesteps
tokenization and gives a word that is almost right (a shared stem, a wrong
inflection) partial credit instead of none. Spaces are removed first. For
each order $n = 1, \ldots, 6$, the precision is the share of the
candidate's $n$-grams that also occur in the reference, and the recall is
the share of the reference's $n$-grams that occur in the candidate. chrF
averages each over the six orders, to $P$ and $R$, and combines the two
into an F-score:

$$F_\beta = \frac{(1 + \beta^2)\, P R}{\beta^2 P + R}.$$

With $\beta = 2$, a reference $n$-gram that the candidate misses costs more
than an extra $n$-gram that the candidate adds. In the code, watch the
intersection `p & r` of the two counters. It keeps the smaller of the two
counts, so a candidate that repeats a correct word ten times gets credit
for it only as often as the reference contains it.

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

BLEU :cite:`Papineni.Roukos.Ward.ea.2002`, which you will meet in most
older papers, counts words. Its $p_n$ is the precision of word $n$-grams
for $n = 1, \ldots, k$, clipped in the same way. BLEU multiplies these
precisions, each raised to a weight, and damps the product with a *brevity
penalty*, so that a model cannot raise its precision by writing only the
few words it is sure of:

$$\text{BLEU} = \exp\!\left(\min\!\left(0, 1 - \frac{\text{len}_{\text{label}}}{\text{len}_{\text{pred}}}\right)\right) \prod_{n=1}^{k} p_n^{1/2^n}.$$
:eqlabel:`eq_bleu`

Here $\text{len}_{\text{pred}}$ and $\text{len}_{\text{label}}$ count the
words of the candidate and the reference. No weight rescues a zero:
$0^{1/16}$ is still 0. So for a candidate of at least $k$ words, BLEU is 0
as soon as one order $n \le k$ has no match at all, and it is 1 only when
every $n$-gram of the candidate is matched and the candidate is not
shorter than the reference. This form weights order $n$ by $1/2^n$ and
scores a single sentence. The standard corpus-level BLEU weights the orders
up to 4 equally and pools the counts over a whole test set, where an order
without any match is rare.

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

The two metrics part ways on a single sentence. We score three candidates
against one reference: an exact copy, a near miss with one wrong
inflection, and a paraphrase that means the same thing in other words. The
near miss lets us predict one number before running anything. It changes
the third of six words, and each of the three word 4-grams of a six-word
sentence contains the third word. So $p_4 = 0$, and BLEU with $k = 4$ is
zero, however good the rest of the sentence is. chrF loses only the
character $n$-grams that touch the stray *s*.

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

The near miss scores 0.91 on chrF and zero on BLEU. The paraphrase
says the same thing as the reference, yet it scores 0.67 on chrF, below the
ungrammatical near miss, and zero on BLEU. Neither metric knows what a
sentence means; both count shared pieces of text. Learned metrics fill
that gap. COMET :cite:`Rei.Stewart.Farinha.ea.2020`, for instance, is a
network trained to predict human ratings of translations. So, increasingly,
does a large language model asked to judge, a model of the kind
:numref:`chap_transformers` builds. In practice, use chrF for a quick
lexical score. To compare with older work, compute the standard
corpus-level BLEU with a shared implementation such as sacreBLEU
:cite:`Post.2018`, not the sentence-level variant above. Check meaning with
a learned metric or a human reader.

## The Fixed-State Bottleneck

How much of a source fits through one state? Translation is a poor place
to ask, because long sentences are also rarer and harder, and the two
effects cannot be told apart. We use the reversal task of
:numref:`subsec_reversal-task` instead: reverse a random string over an
alphabet of 16 letters. The letters are drawn independently, so the
letters the decoder has already written tell it nothing about the next
one, and every letter must come through the encoder's final state. The
task also has the layout that Sutskever et al. obtained by feeding their
source backward: the first letter to write is the last one read.

Each batch is fresh, so the model never sees a string twice. The class
`Reversal` hands the model one sequence `X`: the source, followed by the
decoder input, which is the reversed string shifted right behind a start
token, as teacher forcing requires. The two travel as one sequence because
`d2l.train_steps` passes the model a single input.

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

Counting tells us what to expect, up to a point. One of 16 equally likely
letters carries $\log_2 16 = 4$ bits, so a string of $T$ letters carries $4T$
bits: 32 bits at $T = 8$, 256 bits at $T = 64$. A state of fixed size holds
only so many bits, so beyond some length the accuracy must fall, and a wider
state should hold out longer. Counting says only that the fall must begin
before the string outgrows the state. A state of 64 numbers stored as 32-bit
floats has room for 2,048 bits, enough for 512 letters. Will a trained network
reverse 500 letters through it? A hundred? Make a guess before you read the
table.

The model follows :numref:`fig_seq2seq_details`. A GRU encoder
(:numref:`sec_gru`) reads the source. Its final state $\mathbf{c}$ starts
the GRU decoder, which also reads $\mathbf{c}$ at every step, concatenated
to the embedding of its input letter:

$$\mathbf{s}_{t'} = g(y_{t'-1}, \mathbf{c}, \mathbf{s}_{t'-1}).$$
:eqlabel:`eq_seq2seq_s_t`

Here $g$ is the GRU update and $\mathbf{s}_0 = \mathbf{c}$. The source
thus enters the decoder twice: as its starting state, and again at every
step, so a decoder state that loses track of the source while writing is
handed it back. Feeding $\mathbf{c}$ at every step follows the decoder of
:citet:`Cho.Van-Merrienboer.Gulcehre.ea.2014`; Sutskever et al. used it
only as the starting state. A linear layer maps each decoder state to
scores for the 16 letters.

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

Each model trains for 1,000 steps of 128 fresh strings with AdamW and a cosine
schedule that takes the learning rate from $3 \times 10^{-3}$ to zero, without
warmup or weight decay, as for the reversal model of :numref:`sec_transformer`.
`d2l.LMOptimizer` packages this optimizer, and `d2l.train_steps`
(:numref:`sec_test_models`) runs the steps. We then score the model on 1,000
fresh strings by the fraction of letters it predicts correctly, still given the
correct previous letters. We keep the correct letters for a simple reason. The
letters are independent, so the previous letters alone reveal nothing about the
next one, and whatever the decoder knows about it came through the state. Had
the decoder fed back its own guesses, its early mistakes would mix into the
measurement.

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

We train two models for each state width, 64 and 256, and each string
length from 8 to 64, and print both accuracies. Guessing gives
$1/16 \approx 0.06$.

```{.python .input #seq2seq-the-fixed-state-bottleneck-4}
%%tab pytorch, jax
lengths, widths = (8, 16, 32, 64), (64, 256)
print('width' + ''.join(f'{f"T = {T}":>12}' for T in lengths))
for num_hiddens in widths:
    row = [[reversal_accuracy(num_hiddens, T) for _ in range(2)]
           for T in lengths]
    print(f'{num_hiddens:5d}' + ''.join(f'   {a:.2f}/{b:.2f}' for a, b in row))
```

The two runs of each entry agree to within 0.02, and the trend is ten
times larger than that. With 64 numbers of state, accuracy falls from 0.99
at 8 letters to 0.26 to 0.27 at 64 letters: still four times better than
guessing, but most of the string is lost. The state four times wider holds
out longer. It is still at 0.99 at 16 letters and gets about half the
letters right at 64. So much counting predicted.

Counting did not predict where, and here the table surprises. At 16
letters the string carries 64 bits, a thirty-second of what 64 floats could
store, and the narrow network already loses a quarter of the letters. The
string needs only one bit per number of state, and the trained network
does not deliver even that. Nor is the width of the state the whole story.
Compare entries with the same number of letters per number of state. Eight
letters through 64 numbers come out at 0.99, but 32 letters through 256
numbers only at about 0.8. Likewise, 16 letters through 64 numbers come out
at nearly three in four, and 64 letters through 256 numbers at about half.
Four times the state
for four times the letters loses about 0.2 of accuracy, so a longer string
is harder even when the state grows with it.

The limit is not the precision of the floats. It lies in the code that the
network finds: a way to write the letters into its state through a long chain
of gated updates and to read them back out in reverse order. The table cannot
tell which of three things limits that code: the GRU's architecture, the
optimizer, or the budget of 1,000 steps. It is a measurement at a fixed budget,
and an exercise asks how far longer training moves it.

Translation ran into the same wall.
:citet:`Cho.Van-Merrienboer.Bahdanau.ea.2014` found that the BLEU score of
a recurrent encoder--decoder falls quickly as sentences grow longer. The
much larger LSTMs of :citet:`Sutskever.Vinyals.Le.2014` held up much
better on long sentences, which the authors credited to reading the source
backward; their state was also far wider, which helps in our table too.
The repair came the same year, when :citet:`Bahdanau.Cho.Bengio.2014` let
the decoder look back at every encoder state
(:numref:`sec_attention-scoring-functions`).

The cross-attention model of :numref:`sec_transformer` has no single state
between its encoder and its decoder. Each letter it writes reads the
encoder's vector at every source position, so nothing has to be squeezed.
On strings of 12 letters its loss falls below 0.03 nats per letter by step
100, a tenth of our budget, and after its 300 steps of training it
reverses fresh strings without a single error. But looking back requires
keeping every encoder state, a memory that grows with the input, which is
what a recurrent network was meant to avoid. The rest of this chapter
takes the other road. It keeps one state of fixed size, first makes it
fast to train, and then asks how to make it a better memory, until the
last section mixes a little attention back in.

## Summary

Sequence-to-sequence learning maps one sequence to another of a different
length. The encoder--decoder abstraction splits the model in two: an
encoder reads the source into a state, and a decoder, a conditional
language model, writes the target from it. In the 2014 design that state
is a single vector, and it stands in for the whole source in every factor
of the decoder's probability. Translations are scored by their overlap
with a reference, chrF by character $n$-grams and BLEU by word $n$-grams,
and neither score knows when two sentences mean the same thing. Trained to
reverse random strings, a GRU encoder--decoder with 64 numbers of state
falls from almost perfect at 8 letters to about a quarter of the letters
at 64, far below what counting allows, and a wider state holds out longer.
Attention removes the single state by letting the decoder look back, at
the cost of a memory that grows with the input; the rest of this chapter
keeps the single state and makes it better.

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
1. **An LSTM encoder--decoder.** Replace the GRUs with the LSTMs of
   :numref:`sec_lstm` in both the encoder and the decoder of `Reverser`.
   How does the per-letter accuracy change at each length? An LSTM
   carries a memory cell besides its hidden state; decide which of the two
   the decoder should start from, or whether it should start from both,
   and test your choice.

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
