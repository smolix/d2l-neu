# Codes, Compression, and Perplexity
:label:`sec_mdl-coding-compression`

A probability model is a code, and its log-loss is the length of that code to within
two bits per message. This section builds such codes and measures how close they come
to the entropy floor.

We use the following imports throughout the section.

```{.python .input #mdl-coding-compression-codes-compression-and-perplexity}
#@tab mxnet
import math
```

```{.python .input #mdl-coding-compression-codes-compression-and-perplexity}
#@tab pytorch
import math
```

```{.python .input #mdl-coding-compression-codes-compression-and-perplexity}
#@tab tensorflow
import math
```

```{.python .input #mdl-coding-compression-codes-compression-and-perplexity}
#@tab jax
import math
```

## The Coding View and Perplexity

The coding interpretation makes entropy and KL operational. We now derive it
for binary codes. This subsection therefore uses bits; dividing by $\ln 2$
converts each result to nats.

### Prefix Codes and the Kraft Inequality

Suppose we must transmit a stream of symbols drawn i.i.d. from a distribution
$P$ over $k$ outcomes, encoding each outcome $x_i$ as a binary string
(*codeword*) of length $l_i$. So that the receiver can split the stream back
into codewords without separators, we require the code to be *prefix-free*: no
codeword is a prefix of another (like telephone numbers: once a dialed string
matches a number, no longer number starts the same way). The Kraft inequality
:cite:`Kraft.1949` characterizes the possible codeword lengths.

**Proposition (Kraft inequality).** *A prefix-free binary code with codeword
lengths $l_1, \ldots, l_k$ exists if and only if*

$$
\sum_{i=1}^k 2^{-l_i} \leq 1.
$$

**Proof.** Identify each codeword $c$ of length $l$ with the dyadic interval
$[0.c,\, 0.c + 2^{-l}) \subseteq [0, 1)$ of all real numbers whose binary
expansion starts with $c$ (e.g., the codeword $10$ owns
$[0.10_2, 0.11_2) = [\tfrac{1}{2}, \tfrac{3}{4})$, of length $2^{-2}$). One
codeword is a prefix of another exactly when the second's interval sits inside
the first's, so a prefix-free code corresponds to *disjoint* intervals, whose
total length $\sum_i 2^{-l_i}$ cannot exceed $1$. Conversely, given lengths
with $\sum_i 2^{-l_i} \leq 1$, sort them increasingly and lay down intervals
of length $2^{-l_i}$ left to right; each starts at a multiple of $2^{-l_i}$,
hence is a dyadic interval, and its binary address is a valid codeword.
$\blacksquare$

The same inequality constrains every *uniquely decodable* code (one whose
concatenations parse unambiguously even without prefix-freeness), so
restricting to prefix-free codes costs nothing in code length
:cite:`McMillan.1956`.

![A prefix-free binary code as a tree, for lengths $1, 2, 3, 3$: each codeword is a root-to-leaf path (left edge $= 0$, right edge $= 1$), and prefix-freeness says every codeword ends at a leaf: no codeword continues through another. A leaf at depth $\ell$ is the tree view of a dyadic interval of length $2^{-\ell}$ in the proof, and here the Kraft budget is spent exactly: $2^{-1} + 2^{-2} + 2^{-3} + 2^{-3} = 1$.](../img/mdl-it-kraft-tree.svg)
:label:`fig_mdl-kraft-tree`

:numref:`fig_mdl-kraft-tree` redraws the proof's intervals as a binary tree.
At depth $l_i$, a codeword occupies a fraction $2^{-l_i}$ of the available
leaves. Thus shorter codewords consume a larger fraction of the code tree.

**Proposition (entropy bounds the code length).** *Every prefix-free binary
code for $P$ has expected length $E[l] \geq H_2(P)$ (entropy in bits), and
the* Shannon code *with lengths $l_i = \lceil \log_2 (1/p_i) \rceil$ is
prefix-free and achieves $E[l] < H_2(P) + 1$.*

**Proof.** For the lower bound, let $Z = \sum_i 2^{-l_i} \leq 1$ (Kraft) and
define the distribution $q_i = 2^{-l_i}/Z$, the one *implied* by the code
lengths. Then $l_i = -\log_2 q_i - \log_2 Z$, so

$$
E[l] = \sum_i p_i l_i
= -\sum_i p_i \log_2 q_i - \log_2 Z
= H_2(P) + D_{\textrm{KL}}^{(2)}(P \| Q) - \log_2 Z
\geq H_2(P),
$$
:eqlabel:`eq_mdl-code-length-decomp`

since the KL term is non-negative by Gibbs and $-\log_2 Z \geq 0$ (the
superscript $(2)$ marks base-$2$ logs, here and in $\textrm{CE}_2$ below).
For the
upper bound, the Shannon lengths satisfy
$2^{-l_i} = 2^{-\lceil \log_2(1/p_i)\rceil} \leq p_i$, so they pass Kraft's
test ($\sum_i 2^{-l_i} \leq \sum_i p_i = 1$) and a prefix code with these
lengths exists; and $l_i < \log_2(1/p_i) + 1$ gives
$E[l] < H_2(P) + 1$. $\blacksquare$

Thus entropy determines the optimal expected length to within one bit. This is
the source-coding part of Shannon's theorem, and the
one-bit slack can be driven to zero by coding long blocks of symbols at once
:cite:`Cover.Thomas.1999`. The same proof also identifies the role of KL. If a
code is designed for a distribution $Q$, assigning approximately
$-\log_2 q_i$ bits to symbol $i$, its expected length on data from $P$ is

$$
-\sum_i p_i \log_2 q_i = \textrm{CE}_2(P, Q) = H_2(P) + D_{\textrm{KL}}^{(2)}(P\|Q).
$$

Cross-entropy is therefore the expected length under a code based on $Q$, and
KL is the excess expected length relative to a code matched to $P$. The same
statement in nats differs by the factor $\ln 2$.

The bound is tight enough to see in a four-line example. Take
$P = (\tfrac{1}{2}, \tfrac{1}{4}, \tfrac{1}{8}, \tfrac{1}{8})$, whose
probabilities are exact powers of two, so the Shannon code wastes nothing; a
codebook built for the *uniform* distribution instead spends $2$ bits on every
symbol.

```{.python .input #information-theory-coding}
p = [1/2, 1/4, 1/8, 1/8]
lengths = [math.ceil(math.log2(1 / p_i)) for p_i in p]
kraft = sum(2 ** -l for l in lengths)
avg_len = sum(p_i * l for p_i, l in zip(p, lengths))
h_bits = -sum(p_i * math.log2(p_i) for p_i in p)

q = [1/4] * 4   # the wrong codebook, built as if the data were uniform
lengths_q = [math.ceil(math.log2(1 / q_i)) for q_i in q]
avg_len_q = sum(p_i * l for p_i, l in zip(p, lengths_q))
kl_bits = sum(p_i * math.log2(p_i / q_i) for p_i, q_i in zip(p, q))

print(f'Shannon lengths {lengths}, Kraft sum = {kraft}')
print(f'E[l] = {avg_len} bits = H(P) = {h_bits} bits: no waste')
print(f'uniform codebook: {avg_len_q} bits = H + KL = {h_bits + kl_bits} bits')
```

A possible Shannon code here is $0,\, 10,\, 110,\, 111$, exactly the tree of
:numref:`fig_mdl-kraft-tree`; check that it is prefix-free and that its
lengths $1, 2, 3, 3$ are the ones computed above.

### From Symbol Codes to Arithmetic Coding
:label:`subsec_mdl-arithmetic-coding`

The Shannon code is optimal only up to that "$+1$", and the leak is real: the
ceiling in $l_i = \lceil \log_2(1/p_i) \rceil$ rounds a fraction of a bit up
to a whole bit, once per symbol. For a skewed source the rounding dominates
the expected length. A symbol of probability $0.9$ carries only
$-\log_2 0.9 \approx 0.15$ bits of surprisal, yet no binary codeword is
shorter than one bit, so a symbol-by-symbol code overpays more than sixfold
on the most common symbol it ever transmits.

The fix is to stop coding symbols and code the *whole message*. *Arithmetic
coding* :cite:`Rissanen.1976,Witten.Neal.Cleary.1987` runs the Kraft proof's
dyadic-interval picture in reverse. Maintain a
current interval, initially $[0, 1)$. To encode the next symbol, partition
the current interval into one slice per possible symbol, with lengths
proportional to the model's conditional probabilities, and shrink to the
slice of the symbol actually observed. After the message $x_1, \ldots, x_N$
the interval has width

$$
w = \prod_{i=1}^N q(x_i \mid x_{<i}),
$$

and it identifies the message: a decoder replaying the same model recovers
each symbol in turn by watching which slice the transmitted number falls
into. All that remains is to *name* a point of the final interval in binary,
and the dyadic intervals of the Kraft proof do it: any interval of width $w$
contains a dyadic interval of length $2^{-l}$ with
$l = \lceil \log_2(1/w) \rceil + 1$, because $2^{-l} \leq w/2$, so some
multiple of $2^{-l}$ falls in the interval's left half and the dyadic
interval of length $2^{-l}$ starting there fits inside; that dyadic
interval's binary address is the codeword. The total length is therefore

$$
l \;<\; \sum_{i=1}^N \log_2 \frac{1}{q(x_i \mid x_{<i})} + 2,
$$
:eqlabel:`eq_mdl-arith-bound`

the message's total surprisal under the model plus at most two bits: the
ceiling is paid once per *message*, not once per symbol. A ten-line
demonstration (interval narrowing on a skewed source, then the two bit
counts) makes the gap vivid.

```{.python .input #mdl-information-theory-from-symbol-codes-to-arithmetic-coding}
msg = 'aaaaaaaaba'                    # nine a's, one b
q = {'a': 0.9, 'b': 0.1}              # a simple i.i.d. character model

lo, hi = 0.0, 1.0
for ch in msg:                        # narrow to the observed symbol's slice
    width = hi - lo
    if ch == 'a':                     # 'a' owns the left 90% of the interval
        hi = lo + width * q['a']
    else:                             # 'b' owns the right 10%
        lo = lo + width * q['a']

prod_q = math.prod(q[ch] for ch in msg)
surprisal = -math.log2(prod_q)                     # the floor, in bits
arith_bits = math.ceil(surprisal) + 1              # one interval, one ceiling
shannon_bits = sum(math.ceil(math.log2(1 / q[ch])) for ch in msg)

print(f'final width = {hi - lo:.6e}, prod of probs = {prod_q:.6e}')
print(f'total surprisal = {surprisal:.2f} bits')
print(f'arithmetic code: {arith_bits} bits, symbol-by-symbol Shannon code: '
      f'{shannon_bits} bits')
```

The final interval width equals the product of the per-symbol probabilities,
as the construction promises, and the bookkeeping lands exactly where the
theory says: the message's surprisal is $4.69$ bits, the arithmetic code
spends $6$ bits (the floor plus the once-per-message rounding), and the
symbol-by-symbol Shannon code spends $13$ bits, because it pays the ceiling's
rounded-up whole bit on each of the nine cheap `a`'s. On long messages the
two-bit overhead vanishes into the total and the arithmetic code sits *on*
the surprisal floor.

Notice what the construction never required: the model need not be i.i.d.
Any conditional next-symbol distribution $q(\cdot \mid x_{<i})$ works,
because encoder and decoder both condition on the already-processed prefix.
An autoregressive language model therefore supplies the probability model for
a lossless arithmetic coder. If encoder and decoder share the tokenizer,
model, and finite-precision probability implementation, the ideal code length
for a document is its total surprisal plus fewer than two bits. The model's
token cross-entropy times the token count estimates that code length; a file
format also carries framing and does not include the cost of distributing the
model. :citet:`Deletang.Ruoss.Duquenne.ea.2023` drive an arithmetic coder with
an LLM's next-token probabilities and compress text better than gzip. Thus
minimizing cross-entropy improves the code length available to such a coder.

How well can any compressor do on a real source such as English text? For a
*stationary* source $X_1, X_2, \ldots$ (one whose statistics do not drift
with position), the *entropy rate*

$$
H_{\textrm{rate}} = \lim_{n \to \infty} \frac{1}{n} H(X_1, \ldots, X_n)
= \lim_{n \to \infty} H(X_n \mid X_{n-1}, \ldots, X_1)
$$
:eqlabel:`eq_mdl-entropy_rate`

is the per-symbol generalization of entropy. That the two limits exist and
agree for stationary sources is a theorem we grant :cite:`Cover.Thomas.1999`;
the second limit is the average surprisal of the next symbol given unlimited
context (conditional entropy is developed in
:numref:`sec_mdl-mutual-information`), and $H_{\textrm{rate}}$ is the floor
on the expected code length per symbol of any lossless code, as $H$ is
for i.i.d. symbols. Dependence lowers the floor (conditioning reduces
entropy), so a source with memory compresses better than its letter
frequencies suggest: Shannon estimated the entropy rate of printed English at
roughly one bit per character :cite:`Shannon.1951`, far below the
$\log_2 27 \approx 4.75$ bits of uniformly random letters and spaces.

### Learning by Compression: Minimum Description Length
:label:`subsec_mdl-mdl`

Likelihood measures the bits needed to encode data **after** a model is shared.
Model selection must also account for sharing the model. The minimum description
length (MDL) principle chooses the hypothesis that gives the shortest complete
code,

$$
\hat h_{\textrm{MDL}}
=\mathop{\mathrm{argmin}}_h
\bigl[L(h)+L(\mathcal D\mid h)\bigr].
$$
:eqlabel:`eq_mdl-mdl`

The second term is usually a negative log-likelihood; the first pays for model
complexity. A highly flexible model may compress the training residuals while
losing overall because its parameters or structure are expensive to describe.
This is Occam's razor in operational units rather than a parameter-count slogan.

The connection to Bayes is close: choosing
$L(h)=-\log P(h)$ and $L(\mathcal D\mid h)=-\log P(\mathcal D\mid h)$ gives the
MAP objective. A one-part Bayesian code instead uses the marginal likelihood
$-\log\int P(\mathcal D\mid\theta)P(\theta)\,d\theta$, averaging rather than
plugging in one parameter. For real-valued neural weights, a literal two-part
code also needs a quantization precision and a concrete coder; counting
parameters alone is not a description length. MDL is therefore a disciplined
way to compare complete predictive codes, not an automatic theorem that every
compressed network generalizes.

### Typical Sequences and the Source-Coding Theorem
:label:`subsec_mdl-aep`

The entropy bound above concerns an expected code length. A stronger statement
explains what a **typical long message** looks like. Let
$X_1,\ldots,X_n$ be i.i.d. draws from a finite-alphabet distribution $P$. The
probability of the realized block is
$P(X_{1:n})=\prod_i p(X_i)$, so its surprisal per symbol is a sample mean:

$$
-\frac{1}{n}\log P(X_{1:n})
=\frac{1}{n}\sum_{i=1}^n -\log p(X_i)
\xrightarrow{P} H(P).
$$
:eqlabel:`eq_mdl-aep`

This is the **asymptotic equipartition property** (AEP): with probability
approaching one, a long draw belongs to a **typical set** in which every block
has probability approximately $e^{-nH}$. Because the typical set carries almost
all the probability, it must contain approximately $e^{nH}$ blocks. The full
space may have $k^n$ possible strings, but a source uses only an exponentially
smaller effective subset when $H<\log k$.

This counting picture is the operational core of Shannon's source-coding
theorem :cite:`Shannon.1948,Cover.Thomas.1999`. Encode typical blocks by their
index using about $nH$ nats (or $nH_2$ bits), and handle the vanishing atypical
set separately: rates just above entropy are achievable with failure
probability tending to zero. Conversely, fewer than about $e^{nH}$ codewords
cannot name almost all typical blocks, so rates below entropy cannot be
reliable. For stationary ergodic sources the entropy rate
:eqref:`eq_mdl-entropy_rate` replaces $H$.

For a Bernoulli source the counting claim can be checked *exactly*, because
strings with the same number of ones share one probability. The cell groups
the $2^{200}$ binary strings of length $n=200$ by their count of ones,
declares a string typical when its surprisal is within $0.1$ bit per symbol
of $H$, and adds up exactly how many strings are typical and how much
probability they carry.

```{.python .input #mdl-information-theory-typical-sequences-and-the-source-coding-theorem}
import numpy as onp
n, p = 200, 0.3
k = onp.arange(n + 1)
log_fact = onp.concatenate([[0.0], onp.cumsum(onp.log(onp.arange(1, n + 1)))])
log_count = log_fact[n] - log_fact[k] - log_fact[n - k]     # log #strings, k ones
prob_k = onp.exp(log_count + k * onp.log(p) + (n - k) * onp.log(1 - p))
h2 = -(p * onp.log2(p) + (1 - p) * onp.log2(1 - p))
bits_per_symbol = -(k * onp.log2(p) + (n - k) * onp.log2(1 - p)) / n
typical = onp.abs(bits_per_symbol - h2) < 0.1

m = log_count[typical].max()
log2_typical = (m + onp.log(onp.exp(log_count[typical] - m).sum())) / onp.log(2)
print(f'entropy H = {h2:.3f} bits/symbol, so nH = {n * h2:.0f} bits')
print(f'typical strings: 2^{log2_typical:.1f} of all 2^{n}')
print(f'probability they carry: {prob_k[typical].sum():.4f}')
```

A small fraction of the strings carries almost all of the probability: the
$2^{188.8}$ typical strings are about one in $2^{11}$ of the $2^{200}$, and
they carry probability $0.989$. Indexing only those strings *is* the
compressor, with an index of $189$ bits instead of $200$; at this length and
tolerance the count still exceeds $2^{nH} \approx 2^{176}$.

### Perplexity

Language models are evaluated by exactly the quantity this section has been
studying: the per-token cross-entropy of held-out text,
$\textrm{CE} = -\tfrac{1}{N}\sum_{i=1}^N \log q(x_i \mid x_{<i})$ nats, where
$q$ is the model's predicted next-token distribution. The community
exponentiates it and calls the result *perplexity*
:cite:`Jelinek.Mercer.Bahl.ea.1977`:

$$
\textrm{PPL} = \exp(\textrm{CE})
= \exp\!\Big(-\frac{1}{N}\sum_{i=1}^N \log q(x_i \mid x_{<i})\Big)
= \Big(\prod_{i=1}^N q(x_i \mid x_{<i})\Big)^{-1/N},
$$

the inverse *geometric mean* of the probabilities the model assigned to what
occurred next. Exponentiating gives an interpretation: a model that is
uniformly undecided among $V$ tokens at every step has
$\textrm{CE} = \ln V$ and hence $\textrm{PPL} = V$, so a perplexity of,
say, $20$ means the model is, on average, as uncertain as if it were choosing
uniformly among $20$ tokens: an *effective branching factor*. A perfect model
of a deterministic sequence has perplexity $1$, and the floor is
$\textrm{PPL} \geq e^{H}$, where $H$ is the entropy rate
:eqref:`eq_mdl-entropy_rate` of the language itself, the per-token cost of
its irreducible randomness (Gibbs applied to each next-token prediction,
averaged, plus the entropy-rate limit granted above). Because $\textrm{PPL} = b^{\textrm{CE}_b}$ for *any* log base $b$,
perplexity is also the rare information-theoretic quantity with no unit
ambiguity: nats and bits give the same number (Exercise :exref:`ex_mdl-information-theory-show-that-perplexity-does`).

A five-token toy stream makes the bookkeeping concrete.

```{.python .input #information-theory-perplexity}
q_tok = [0.2, 0.1, 0.4, 0.25, 0.05]   # model prob. of each observed token
nll = [-math.log(q_i) for q_i in q_tok]
mean_nll = sum(nll) / len(nll)

print(f'per-token NLL (nats): {[round(v, 3) for v in nll]}')
print(f'mean NLL = {mean_nll:.4f} nats, perplexity = {math.exp(mean_nll):.2f}')
```

A mean of $\approx 1.84$ nats per token corresponds to a perplexity of
$\approx 6.3$: this model is as confused as a uniform choice among six or so
tokens. The geometric mean penalizes confident mistakes: the single
$0.05$ contributes $3.0$ nats, as much as several good predictions
combined. Perplexity in :numref:`sec_language-model` and the Transformer
chapters is this number.

Empirical studies have fitted held-out cross-entropy by power laws in model size,
data, and training compute over substantial observed ranges
:cite:`kaplan2020scaling,hoffmann2022training`. Such fits can support planning
within a studied regime, but their fitted irreducible term is not an identified
entropy rate of natural language. Nor does a generic power law imply that each
constant multiple of compute removes a constant fraction of the remaining loss;
that interpretation describes exponential decay. Per-token quantities also
depend on the tokenizer, so perplexities are directly comparable only under a
shared tokenization and evaluation protocol. Bits per character or byte can
improve cross-tokenizer comparability when text normalization is held fixed.

## Summary

* In the coding view, entropy is the optimal expected code length (Kraft
  inequality + Shannon code) and KL is the extra bits per symbol from coding
  with the wrong distribution.

* Arithmetic coding removes the symbol code's per-symbol rounding: with a
  shared model and coder, a whole message costs its total surprisal plus at
  most two bits. An autoregressive model supplies the probabilities for that
  lossless code; its cross-entropy times the token count is the model code
  length before file-format overhead. The entropy rate is the per-symbol floor
  for any such code. The AEP explains this floor by showing that almost all
  length-$n$ messages occupy a typical set of size about $e^{nH}$.

* **Minimum description length** selects a complete code for model plus data;
  negative log-likelihood is only the data-given-model part.

* Perplexity $\textrm{PPL} = \exp(\textrm{CE})$ is the exponentiated
  per-token cross-entropy of a language model: an effective branching factor,
  independent of the log base. Empirically it falls as a power law in
  parameters, data, and compute, and it is comparable only under a shared
  tokenizer.

## Exercises

1. **Perplexity across tokenizers.** Perplexity is computed per token, so
   it depends on how a text is split into tokens.
    1. Show that perplexity does not depend on the log base: if
       $\textrm{CE}_b$ denotes the per-token cross-entropy in base-$b$ units,
       then $b^{\textrm{CE}_b}$ is the same for every $b$.
    1. Two language models are evaluated on the same text of $10^6$
       characters. The tokenizer of model A splits it into
       $2.5 \times 10^5$ tokens, and A reaches a per-token perplexity of
       $20$. The tokenizer of model B splits it into $6.7 \times 10^5$
       tokens, and B reaches a per-token perplexity of $8$. Convert both
       results to bits per character with :eqref:`eq_bpb`, counting
       characters in place of bytes. Which model predicts the text better?
    1. Take the entropy rate of the text to be one bit per character, the
       estimate of :citet:`Shannon.1951`. Per-token perplexity is bounded
       below by $e^{H}$, where $H$ is the entropy rate per token in nats.
       What is the lowest per-token perplexity that any model can reach
       with the tokenizer of A, and with that of B? How far is each model
       from its floor?
   :exlabel:`ex_mdl-information-theory-show-that-perplexity-does`

1. [code] **Exact arithmetic coding.** ● The interval narrowing of
   :numref:`subsec_mdl-arithmetic-coding` stores `lo` and `hi` as
   floating-point numbers. Draw a message of $N = 1000$ symbols, each of
   which is `a` with probability $0.9$ and `b` otherwise.
    1. Run the floating-point narrowing on this message with the model
       $q(\texttt{a}) = 0.9$. After how many symbols does `hi - lo` first
       differ from the product of the probabilities by more than $0.1\%$,
       and after how many does it become zero? Relate both answers to the
       $53$-bit significand of a double-precision number.
    1. Repeat the narrowing with exact rationals (`fractions.Fraction`).
       Name a point of the final interval of width $w$ with
       $\lceil \log_2(1/w) \rceil + 1$ bits, as in
       :numref:`subsec_mdl-arithmetic-coding`, and write a decoder that
       replays the model to recover the message. Compare the code length
       with the surprisal of the message and with $N H_2(0.9)$.
    1. Encode the same message with the mismatched model
       $q(\texttt{a}) = 0.7$. Compare the increase in code length with
       $N D_{\textrm{KL}}^{(2)}(P \| Q)$. Why do the two numbers differ for
       a single message?
   :exlabel:`ex_mdl-information-theory-exact-arithmetic-coding`

1. [code] **Typical and high-probability sets.** ● The code of
   :numref:`subsec_mdl-aep` declares a binary string of length $n = 200$
   from a Bernoulli source with $p = 0.3$ typical when its surprisal per
   symbol lies within $\epsilon = 0.1$ bit of the entropy $H$ in bits. It
   finds $2^{188.8}$ typical strings, many more than $2^{nH} \approx 2^{176}$.
    1. Which string is the most probable one? Compute its surprisal per
       symbol and decide whether it is typical. Why does this string not
       matter for compression?
    1. Let $A$ be the typical set and let $B$ be any set of strings with
       $P(B) \geq 1 - \delta$. Show that
       $P(A \cap B) \geq 1 - \delta - P(A^c)$, and conclude that
       $|B| \geq \big(1 - \delta - P(A^c)\big)\, 2^{n(H - \epsilon)}$.
    1. The smallest set with probability at least $0.9$ collects strings in
       order of decreasing probability. Group the strings by their number of
       ones, as the code does, and compute $\tfrac{1}{n}\log_2$ of the size
       of this set for $n = 200$, $1000$, $5000$, and $20000$. Compare it
       with $H$ and with $\tfrac{1}{n}\log_2 |A|$ at the same $n$. Which of
       the two sets grows like $2^{nH}$, and why does the typical set with a
       fixed $\epsilon$ not?

   *Adapted from David Tse,
   [EE376A/Stats376A: Information Theory, Homework 2](https://web.stanford.edu/~kedart/ee376a_winter1617/hw2.pdf),
   Problem 6.*
   :exlabel:`ex_mdl-information-theory-typical-sets`

1. **Two-part code for a coin.** ● A sequence $\mathcal{D}$ of $n$ coin
   flips contains $k$ heads, and $\hat\theta = k/n$. Hypothesis $h_0$ is
   the fair coin: it has no parameter to describe, and
   $L(\mathcal{D} \mid h_0) = n \log 2$. Hypothesis $h_1$ is a coin with
   bias $\theta$. Its encoder sends the point $\theta_q$ nearest to
   $\hat\theta$ on a grid of spacing $\delta$ in $(0, 1)$, which costs
   $L(h_1) = \log(1/\delta)$, and the decoder knows $n$ and $\delta$.
    1. Use :eqref:`eq_mdl-ce_decomp` to show that
       $L(\mathcal{D} \mid h_1) = n H(\hat\theta) +
       n D_{\textrm{KL}}(\hat\theta \,\|\, \theta_q)$, where entropy and KL
       divergence are those of the Bernoulli distributions with these
       parameters. Expand the KL term to second order and show that the
       total length :eqref:`eq_mdl-mdl` of $h_1$ exceeds
       $n H(\hat\theta)$ by at most about
       $\log(1/\delta) + n\delta^2 / \big(8\hat\theta(1 - \hat\theta)\big)$.
    1. Which spacing $\delta$ minimizes this bound? The decoder must know
       $\delta$ before it decodes $\theta_q$. Why can the code therefore not
       use the minimizer? Show that the spacing $\delta = 1/\sqrt{n}$, which
       depends on $n$ alone, gives an excess of
       $\tfrac{1}{2}\log n + O(1)$ whenever $\hat\theta$ stays bounded away
       from $0$ and $1$.
    1. Take $n = 1000$ and the approximation
       $L(h_1) + L(\mathcal{D} \mid h_1) \approx n H_2(\hat\theta) +
       \tfrac{1}{2}\log_2 n$ bits. Which hypothesis does MDL select for
       $k = 600$, and which for $k = 520$? Approximately how far from $n/2$
       must $k$ lie for $h_1$ to win? Compare this distance with the
       standard deviation $\sqrt{n}/2$ of $k$ under the fair coin.
   :exlabel:`ex_mdl-information-theory-two-part-coin`

1. [extended] **Adaptive context-model compression.** ●● An adaptive model
   transmits no parameters. Before coding the $i$-th symbol, it estimates
   $q(x_i \mid x_{<i})$ from the counts of the symbols already coded, and
   the decoder repeats the same counting. The arithmetic code length
   :eqref:`eq_mdl-arith-bound` of a text is then its total surprisal under
   these predictions plus at most two bits. This total is a complete
   description length in the sense of :eqref:`eq_mdl-mdl`, with no separate
   term for the model. :citet:`Blier.Ollivier.2018` use such prequential
   codes to measure the description length of deep networks (see also
   Section 3.2 of :citet:`Deletang.Ruoss.Duquenne.ea.2023`). Work with the
   text of *The Time Machine* (:numref:`sec_text-sequence`), downloaded
   with `d2l.download(d2l.DATA_URL + 'timemachine.txt')`, lowercased, and
   reduced to the $27$ symbols of letters and space as in
   `TimeMachine._preprocess`.
    1. An order-$m$ model predicts each character from the $m$ characters
       before it with the estimate
       $q(x \mid s) = \big(n(s, x) + \tfrac{1}{2}\big) /
       \big(n(s) + \tfrac{27}{2}\big)$, where $n(s, x)$ counts how often $x$
       has followed the context $s$ so far and $n(s) = \sum_x n(s, x)$. For
       $m = 0, 1, \ldots, 8$, compute the adaptive code length in bits per
       character. Compare it with the per-character cross-entropy of the
       order-$m$ model whose counts are taken from the entire text, and
       with the size of `zlib.compress(text.encode(), 9)` in bits per
       character.
    1. Explain why the adaptive code length has a minimum over $m$ while
       the cross-entropy of the fitted model keeps falling. Which order does
       MDL select? Which term of :eqref:`eq_mdl-mdl` does the cross-entropy
       of the fitted model leave out?
    1. Design an adaptive model that combines contexts of several orders,
       for example by falling back to shorter contexts when a long context
       has few counts. Report a table of bits per character for every model
       you tried, together with zlib, and a paragraph on the design choice
       that mattered most.
   :exlabel:`ex_mdl-information-theory-adaptive-compression`

[Discussions](https://d2l.discourse.group/)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §26.0]{.kicker}

**Codes, compression, and perplexity**
:::
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[The coding view]{.dtitle}

[Kraft, Shannon, arithmetic coding: why KL is *extra bits*]{.dsub}
:::
:::

::: {.slide title="Prefix codes and the Kraft inequality"}
[Codes]{.kicker}

::: {.cols .vc}
::: {.col}
A prefix-free binary code with lengths $\ell_i$ exists **iff**
$\sum_i 2^{-\ell_i} \le 1$: each codeword claims a dyadic interval, and
prefix-free means the intervals are disjoint.

Shannon picks $\ell_i = \lceil\log_2 \tfrac{1}{p_i}\rceil$.
:::

::: {.col .fig}
@fig:mdl-it-kraft-tree
:::
:::
:::

::: {.slide title="Shannon's code: KL is the extra bits"}
[Coding interpretation]{.kicker}

The matched code spends $H_2(P) \le \mathbb{E}[\ell] < H_2(P)+1$; a code
based on $Q$ has excess expected length $D_{\mathrm{KL}}$:

@!information-theory-coding

::: {.d2l-note .rule}
$H$ is the minimum expected code length, cross-entropy is the length obtained
under $Q$, and $D_{\mathrm{KL}}$ is their difference.
:::
:::

::: {.slide title="Arithmetic Coding and Rounding Overhead"}
[Beyond symbol codes]{.kicker}

Shannon's $\lceil\log_2(1/p)\rceil$ rounds up to a whole bit on *every*
symbol: a $p=0.9$ symbol carries $0.15$ bits yet costs $1$. Arithmetic coding
runs the Kraft picture in reverse: narrow one interval per message, to width
$w = \prod_i q(x_i \mid x_{<i})$, and name it in
$\lceil\log_2(1/w)\rceil + 1$ bits:

@!mdl-information-theory-from-symbol-codes-to-arithmetic-coding

$4.69$ bits of surprisal: the arithmetic code spends $6$; the symbol-by-symbol
code, $13$.
:::

::: {.slide title="A language model is a compressor"}
[Prediction = compression]{.kicker}

Nothing required i.i.d.: *any* next-symbol model $q(\cdot\mid x_{<i})$ drives
the coder. With a shared tokenizer, model, and finite-precision coder, a
document's total surprisal costs fewer than two extra bits; file framing and
model distribution are separate costs. Delétang et al. (2023) demonstrate
this connection using language models as compressors.

. . .

The floor is the source's **entropy rate**
$H_{\mathrm{rate}} = \lim_n H(X_n \mid X_{<n})$: about $1$ bit/character for
English (Shannon, 1951), far below the $4.75$ of random letters.

::: {.d2l-note}
The scaling laws, read in this lens: held-out CE falls as a *power law* in
compute, so the approach to the entropy-rate floor is empirically
**predictable**.
:::
:::

::: {.slide title="Perplexity: an effective branching factor"}
[Language models]{.kicker}

$\mathrm{PPL} = \exp(\mathrm{CE})$: the inverse geometric-mean probability, the
number of equally-likely choices the model is as confused among:

@information-theory-perplexity

Base-free: a single bad token ($p=0.05$) costs as much as several good ones.
:::
