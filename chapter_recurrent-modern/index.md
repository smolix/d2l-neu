# State Space Models
:label:`chap_modern_rnn`

The recurrent networks of :numref:`chap_rnn` compress an unbounded past into a
fixed-size hidden state: one vector that must carry forward everything later
steps will need. That raises a question the vanilla RNN never really answers.
What should the state remember? Its reply is "whatever stochastic gradient
descent happens to find," and :numref:`sec_bptt` showed why that reply is
unsatisfying. Learning a long-range dependency means pushing a gradient back
through many steps, where the recurrent Jacobian multiplies it again and again
until it vanishes or explodes. Clipping tames the blow-up, but a signal that
has decayed into noise cannot be recovered. To decide what a state remembers,
we have to change the recurrence itself.

This chapter gives three answers, in roughly historical and conceptual order.
The first is to *gate* it. The long short-term memory (LSTM) cell and its
streamlined cousin the gated recurrent unit (GRU, :numref:`sec_lstm`) attach
learned, input- and state-dependent gates that control what the state reads,
writes, and forgets; a value protected by a forget gate can survive hundreds
of steps. The second is to *linearize* it. Dropping the nonlinearity from the
state update, as minGRU does and as the structured state space models S4 and
S4D do from a continuous-time starting point (:numref:`sec_ssm`), turns the
recurrence into an affine map. Training then parallelizes across the sequence
with a scan instead of crawling one step at a time, and the memory becomes
analyzable, its decay read straight off the eigenvalues. The third is to
*select* it. A linear recurrence with fixed coefficients is content-blind, so
we make its dynamics input-dependent again, the move behind Mamba
(:numref:`sec_mamba`); this recovers the content-awareness of a gate while
keeping the linear-time, parallel-training cost. Gate, linearize, select:
three ways to answer one question. Remarkably, the forget gate of the first
answer reappears inside the third, since Mamba's input-dependent step size is
a gate wrapped around a linear state, and the arc closes on the idea it opened
with.

Woven through this progression is a second thread. :numref:`sec_seq2seq`
develops the encoder-decoder architecture that first carried recurrent
networks to large-scale machine translation, the application that drove the
field for years. The same abstraction still frames speech recognition,
captioning, and multimodal front-ends, so it is worth building once even
though word-level translation is behind us. It also exposes a limitation that
reframes the whole chapter: the encoder squeezes an entire source sentence
through a single fixed-size vector. That bottleneck is the memory question in
another guise, and it points to two escapes, namely building a *better* fixed
state (the rest of this chapter) or letting the model *look back* at
everything at once (attention, the next part).

The chapter therefore has one recurring adversary, the fixed-size state, met
at growing sophistication and never fully beaten. Gating, linearization, and
selection each make a bounded memory hold more of what matters, and modern
hybrid language models run these layers in production for exactly that reason.
What recurrence buys in exchange is inference at constant memory and linear
time in the sequence length, a bargain that keeps it attractive wherever
generation must run cheaply and at scale. Yet no fixed-size state can recall
arbitrary detail on demand; asked to
reproduce a phone number it read a thousand tokens ago, it fails where a
simple lookup would not. That honest limit is exactly what attention
(:numref:`chap_attention`) pays quadratic cost to remove, and this
chapter maps how far recurrence goes without paying it. The ideas gathered here span nearly
three decades, from the 1997 LSTM to the 2024 selective state space models,
and together they make recurrence a living architecture rather than a
historical one.

```toc
:maxdepth: 2

lstm
seq2seq
ssm
mamba
```
