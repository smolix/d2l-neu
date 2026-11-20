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
