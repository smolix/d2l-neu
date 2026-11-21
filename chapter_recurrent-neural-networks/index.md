# Sequence Models
:label:`chap_rnn`

Having spent two chapters on images, we close this part of the book by turning
to sequential data. Every model we have built so far expects an input of fixed
shape—a tabular vector of attributes or a $28 \times 28$ grid of pixels—and
treats each example independently. A sentence, an audio recording, a video, or
a stream of sensor measurements behaves quite differently: its length varies
from one example to the next, and its entries depend strongly on one another
and on the order in which they arrive. Consider the opening sentence of H. G.
Wells' *The Time Machine*, which serves as our running dataset throughout this
chapter: "The Time Traveller (for so it will be convenient to speak of him)
was expounding a recondite matter to us." Guessing the final word requires
reading the nineteen words before it in order. How can a single model make
that prediction when the preceding text might be arbitrarily long?

Claude Shannon framed this question as a guessing game :cite:`Shannon.1951`,
estimating the predictability of printed English by asking human readers to
guess a passage one character at a time. A *language model* plays the same
game: it assigns a probability to the next *token*—whether a character, a
subword, or a whole word—given the tokens that preceded it, and can generate
new text by feeding each chosen token back in as its next input.

Because the history already read grows with every step, restricting attention
to a fixed window of recent tokens discards everything earlier. A *recurrent
neural network* (RNN) instead maintains a fixed-size *hidden state* that
summarizes the sequence read so far, updating that state with the same
parameters after every token. Even where modern large language models replace
recurrence with attention (:numref:`chap_attention`) or structured state space
models (:numref:`chap_modern_rnn`), the core machinery of this
chapter—subword tokenization, perplexity and bits-per-byte evaluation, and
autoregressive decoding—carries over unchanged.

We begin by formulating autoregressive sequence forecasting and observing why
multistep rollouts degrade much faster than one-step predictions
(:numref:`sec_sequence`). Next, we convert raw text into subword token
sequences via byte-level Byte Pair Encoding (:numref:`sec_text-sequence`) and
build count-based $n$-gram language models alongside the perplexity and
bits-per-byte metrics (:numref:`sec_language-model`). We then formulate and
implement recurrent neural network language models from scratch and with
high-level APIs (:numref:`sec_rnn`), stabilizing training with gradient
clipping and comparing the learned representations against Kneser--Ney
$n$-grams. Finally, :numref:`sec_bptt` derives backpropagation through time to
explain why vanilla RNN gradients vanish or explode, and
:numref:`sec_decoding` studies greedy search, beam search, and truncated
sampling methods for generating text.

```toc
:maxdepth: 2

sequence
text-sequence
language-model
rnn
bptt
decoding
```

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §8]{.kicker}

Predicting text one token at a time<br>
**sequences · tokens · language models · recurrent networks · implementation · backpropagation through time · decoding**
:::
:::

::: {.slide title="The Next Token Depends on a Prefix That Grows Without Bound"}
[The problem]{.kicker}

- "The Time Traveller (for so it will be convenient to speak of him) was
  expounding a recondite matter to ___": nineteen words, in order, decide
  the next one.
- Earlier models take an input of fixed shape and treat examples as
  independent.
- A count table over 20 BPE tokens of context: $1024^{20} \approx 10^{60}$
  possible contexts.
- Even for word triples, nearly nine in ten held-out triples of *The Time
  Machine* never occur in training.

::: {.d2l-note}
How can a model of fixed size read a prefix of any length, and how do we
turn its predictions back into text?
:::
:::

::: {.slide title="By the End of This Chapter You Can"}
[Goals]{.kicker}

- turn sequence modeling into next-step prediction, and measure how a
  rollout's error grows with the horizon;
- train a byte-level BPE tokenizer and check it against GPT-2's;
- estimate $n$-gram models and compare language models by perplexity and
  bits per byte;
- write down an RNN language model, count its parameters, and train it
  with gradient clipping;
- explain why gradients through time vanish or explode, measure how far
  back they reach, and say what truncation removes;
- decode text by greedy decoding, beam search, or truncated sampling.

**Running example:** *The Time Machine*, 1,024 BPE tokens plus three
special tokens.
:::

::: {.slide title="Map of the Chapter"}
[Chapter map]{.kicker}

| section | question | adds |
|---|---|---|
| Working with Sequences | how do we predict a sequence from its own past? | autoregression, latent state, rollout error |
| From Text to Tokens | how do we cut text into tokens? | characters, words, bytes, BPE |
| Language Models | how probable is a text, and how good is a model? | $n$-grams, perplexity, bits per byte |
| Recurrent Neural Networks | how does a fixed-size state replace the window? | a hidden state updated at every step |
| Implementing RNNs | how do we build and train an RNN language model? | an RNN language model, scratch and concise |
| Backpropagation Through Time | why do gradients through time vanish or explode? | gradients through time, truncation |
| Decoding and Generation | how do we turn next-token distributions into text? | greedy decoding, beam search, sampling |
:::

::: {.slide title="Every Section Ends With a Number on The Time Machine"}
[Running examples]{.kicker}

| section | key number |
|---|---|
| Working with Sequences | 64 steps ahead, the rollout error is four to six times the one-step error |
| From Text to Tokens | our encoder matches GPT-2's tokenizer token for token; 1,024 tokens give about two thirds of the compression of vocabularies 50 and 200 times larger |
| Language Models | 87% of held-out word trigrams are unseen; Kneser–Ney lowers the word trigram's perplexity from 2,481 to 300 |
| Recurrent Neural Networks | 222,915 parameters, 89% of them in the embedding table and the output layer; 512 bytes of state per sequence |
| Implementing RNNs | about 2.1 bits per byte, level with the Kneser–Ney trigram; the trained model uses about three tokens of its 32-token window |
| Backpropagation Through Time | 40 steps: $0.9^{40} \approx 0.015$, $1.1^{40} \approx 45$; in the trained model the gradient falls below 1% within about eight steps |
| Decoding and Generation | more than $10^{150}$ continuations of 50 tokens; greedy text loops, sampled text has distinct-3 near 1 |
:::
