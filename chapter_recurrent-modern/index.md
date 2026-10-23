# State Space Models
:label:`chap_modern_rnn`

A language model that generates text one token at a time has to remember
what it has read. A transformer keeps everything: it stores a key and a value
for every previous token, so the memory it needs, and the work per new token,
grow with the length of the context. A recurrent network keeps a summary of
fixed size instead, its *state*, and updates it once per token. Generating
the millionth token then costs as much as generating the tenth.

A fixed-size summary raises two problems. Classical recurrent networks update
the state with a nonlinear function of the previous state, so training has to
process a sequence one step at a time and leaves most of a GPU idle. And a
state of fixed size must forget, which raises the question of what it should
keep and how much it can hold.

:numref:`sec_lstm` shows how gates let a network learn what to keep, and why
gated networks still train one step at a time.
:numref:`sec_linear-recurrence` makes the update linear in the state, which
lets us compute the states at all positions at once. :numref:`sec_ssm`
borrows from physics a principled way to choose how fast the state forgets,
and :numref:`sec_mamba` lets each input decide how much to write and how much
to forget. :numref:`sec_matrix-state` turns the state into a matrix that
stores key–value pairs, which is linear attention, and asks how many pairs it
can hold. :numref:`sec_deltanet` lets a write correct an old association
instead of adding to it. The last section, :numref:`sec_hybrids`, shows what
no fixed-size state can do and how a few attention layers repair it.

We assume you know how a recurrent network carries a hidden state, why its
gradients vanish over long spans, and how attention reads keys and values.
The chapter teaches algorithms rather than GPU kernels; fused kernels are the
subject of :numref:`chap_performance`.

```toc
:maxdepth: 2

lstm
linear-recurrence
ssm
mamba
matrix-state
deltanet
test-time-regression
hybrids
```
