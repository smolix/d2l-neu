# Language Models
:label:`sec_language-model`

A smartphone keyboard suggesting the next word and a speech recognizer
deciding whether a speaker said "to recognize speech" or "to wreck a nice
beach" both depend on the same capability: assigning a numerical score that
reflects how plausible a passage of text is. Because those two phrases sound
nearly identical, acoustics alone cannot tell them apart; what resolves the
ambiguity is knowing that the first is idiomatic English while the second is
outlandish. A *language model* provides this judgment: given a sequence of
tokens $x_1, \ldots, x_T$ drawn from a vocabulary $\mathcal{V}$—such as the
BPE tokens constructed in :numref:`sec_text-sequence`—it estimates the joint
probability $P(x_1, \ldots, x_T)$.

At first glance, assigning a probability to every possible sequence appears
hopeless: even with our modest 1,024-token vocabulary, there are already
$1024^{10} \approx 10^{30}$ sequences of length ten. Fortunately, the chain
rule of probability (:eqref:`eq_chain_rule` in :numref:`sec_sequence`)
converts this intractable joint distribution—with no approximation—into a
product of next-token conditionals, each conditioned on the preceding prefix
$x_{<t} = x_1, \ldots, x_{t-1}$:

$$P(x_1, \ldots, x_T) = \prod_{t=1}^T P(x_t \mid x_{<t}).$$
:eqlabel:`eq_lm_factorization`

A language model is therefore fundamentally a next-token predictor. The same
conditional distributions let us both *score* an existing sequence (by
multiplying them, or adding their logarithms) and *generate* new text (by
sampling $x_1$, then sampling $x_2$ given $x_1$, and continuing step by step).
Remarkably, a wide range of tasks can be framed directly as text continuation:
completing the prompt "The novella The Time Machine was written by" answers a
factual question, which is precisely the paradigm behind modern large language
models (:numref:`sec_gpt`).

We begin with the simplest possible estimator for these conditional
probabilities—counting occurrences in a corpus—and investigate why it breaks
down. To evaluate improvements objectively, we introduce perplexity, compare
two classical smoothing methods, and then define bits per byte so that models
with different tokenizers can be compared fairly. Finally, we build the
minibatch data loader that feeds tokenized sequences into the neural language
models of the upcoming sections.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input #language-model-language-models}
%%tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #language-model-language-models}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #language-model-language-models}
%%tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #language-model-language-models}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #language-model-language-models-1}
import collections
import random
import re
```

## $n$-gram Language Models
:label:`subsec_markov-models-and-n-grams`

### Markov Models and Counting

Because each factor in :eqref:`eq_lm_factorization` conditions on the entire
prefix $x_{<t}$, the number of distinct contexts grows without bound as $t$
increases. Invoking the Markov condition (:numref:`subsec_markov-models`)
truncates the context to the $n-1$ most recent tokens:

$$P(x_t \mid x_{<t}) \approx P(x_t \mid x_{t-n+1}, \ldots, x_{t-1}).$$

Although this assumption is undeniably crude—a sentence hardly forgets its
subject after two words—it reduces the set of possible contexts to a finite
number. A model with no context ($n=1$) is called a *unigram* model, one that
conditions on a single preceding token ($n=2$) a *bigram* model, and one that
conditions on two preceding tokens ($n=3$) a *trigram* model. Under maximum
likelihood estimation, each conditional probability is simply a ratio of
corpus counts. Writing $N(\cdot)$ for the number of times a token tuple
appears in the training corpus, we obtain

$$\hat{P}(x_t \mid x_{t-n+1}, \ldots, x_{t-1}) = \frac{N(x_{t-n+1}, \ldots, x_{t})}{N(x_{t-n+1}, \ldots, x_{t-1})},$$
:eqlabel:`eq_ngram_mle`

the empirical fraction of times the context $(x_{t-n+1}, \ldots, x_{t-1})$ was
followed by $x_t$. An $n$-gram language model is nothing more than a lookup
table of these conditional relative frequencies.

Let us estimate unigram, bigram, and trigram tables on *The Time Machine*.
While the neural language models later in this chapter operate on BPE tokens,
here we tokenize the novella into words so that generated passages are easy to
inspect by eye. Reusing the simple lowercase-letter normalization from
:numref:`sec_text-sequence`, we split on whitespace and hold out the final 10%
of the text for evaluation. The vocabulary consists of all words occurring in
the training split; any held-out word absent from training is mapped to the
unknown token `<unk>`.

```{.python .input #language-model-markov-models-and-counting-1}
fname = d2l.download(d2l.DATA_URL + 'timemachine.txt',
                     sha1_hash='090b5e7e70c295757f55df93cb0a180b9691891a')
raw_text = open(fname).read()
text = re.sub('[^A-Za-z]+', ' ', raw_text).lower()
split = text.rindex(' ', 0, int(0.9 * len(text)))
words_train = text[:split].split()
known = set(words_train)  # A held-out word never seen in training is <unk>
words_val = [w if w in known else '<unk>' for w in text[split:].split()]
len(words_train), len(words_val)
```

The `NGramLM` class below stores, for every observed $(n-1)$-token context, a
counter of the tokens that followed it in training—everything needed to
evaluate :eqref:`eq_ngram_mle` (the `alpha` attribute will be used shortly
when we introduce smoothing). Its `sample` method extends a given prefix by
repeatedly drawing the next token in proportion to its count given the most
recent $n-1$ tokens.

```{.python .input #language-model-markov-models-and-counting-2}
class NGramLM:
    """An n-gram language model estimated from counts."""
    def __init__(self, tokens, n, vocab_size, alpha=0.01):
        self.n, self.vocab_size, self.alpha = n, vocab_size, alpha
        cont = collections.defaultdict(collections.Counter)
        for gram in zip(*(tokens[i:] for i in range(n))):
            cont[gram[:-1]][gram[-1]] += 1
        self.cont = dict(cont)
        self.total = {ctx: sum(c.values()) for ctx, c in self.cont.items()}

    def sample(self, prefix, num_tokens, rng):
        """Extend prefix by sampling from the unsmoothed count model."""
        out = list(prefix)
        for _ in range(num_tokens):
            ctx = tuple(out[len(out) - self.n + 1:])
            counts = self.cont[ctx]
            out.append(rng.choices(list(counts), counts.values())[0])
        return out

vocab_size = len(known) + 1  # The training words and <unk>
models = {n: NGramLM(words_train, n, vocab_size) for n in (1, 2, 3)}
```

To inspect what these tables actually capture, let us query the trigram model
on the context `"the time"`, which appears frequently throughout a novella
about time travel.

```{.python .input #language-model-markov-models-and-counting-3}
trigram = models[3]
total = trigram.total[('the', 'time')]
for word, count in trigram.cont[('the', 'time')].most_common(3):
    print(f'P({word} | the time) = {count}/{total} = {count / total:.2f}')
single = sum(len(c) == 1 for c in trigram.cont.values()) / len(trigram.cont)
print(f'word pairs followed by a single word in training: {single:.0%}')
```

In the training split, `"the time"` occurs 87 times and is followed by
`"traveller"` roughly half the time and by `"machine"` roughly a third of the
time—showing that the table readily picks up the central phrases of the novel.
Yet frequent contexts like this are the rare exception: for more than four out
of five observed word pairs, the training table records only a single
continuation.

### Sampling from $n$-gram Models

Generating text directly applies :eqref:`eq_lm_factorization`: draw the next
token from the conditional distribution given the prefix so far, append it,
and repeat. Because fluent-looking output can easily mask verbatim
memorization, we pair our samples with a diagnostic function, `longest_copy`,
that measures the longest contiguous span of generated words appearing
verbatim in the training set.

```{.python .input #language-model-sampling-from-n-gram-models-1}
def longest_copy(words, train_words):
    """Length of the longest run of words that also occurs in train_words."""
    corpus, best = ' ' + ' '.join(train_words) + ' ', 0
    for i in range(len(words)):
        while (i + best < len(words) and
               ' ' + ' '.join(words[i:i + best + 1]) + ' ' in corpus):
            best += 1
    return best
```

We seed each model with the prefix `"the time"` and generate 25 words, leaving
the random number generator unseeded so that each execution produces fresh
samples.

```{.python .input #language-model-sampling-from-n-gram-models}
rng = random.Random()
for n, model in models.items():
    sample = model.sample(['the', 'time'], 25, rng)
    copied = longest_copy(sample, words_train)
    d2l.print_wrapped(f'{n}-gram (longest copied run: {copied} words): '
                      + ' '.join(sample))
```

The unigram model emits words at their corpus frequencies with no grammatical
coherence; the bigram model strings together locally plausible pairs that
wander aimlessly, since each word depends only on its immediate predecessor;
and the trigram sample already sounds remarkably like H. G. Wells.

On a 30,000-word corpus, however, the trigram sounds like Wells largely
because much of it *is* Wells. The longest copied run exposes this behavior
immediately, rising from two or three words for the unigram, to a handful for
the bigram, and typically seven to a dozen consecutive words for the trigram.
Whenever the sampler lands on a word pair that occurred with only one
successor in training, it is forced onto a single deterministic track,
replaying entire clauses of the novella verbatim. Rather than generalizing,
the model is stitching together memorized fragments. (When estimated over
billions of words, $n$-gram tables do capture rich local English syntax, which
is why they powered speech recognition and statistical machine translation
for decades.)

### Count Sparsity

This memorization is the visible symptom of a deeper statistical bottleneck.
The number of *possible* $n$-grams is $|\mathcal{V}|^n$, growing exponentially
with $n$, whereas the number of $n$-grams actually present in a corpus grows
only linearly with corpus length. We can measure the resulting sparsity
directly by checking what fraction of held-out $n$-grams never appeared even
once in training.

```{.python .input #language-model-the-sparsity-wall}
for n in (1, 2, 3):
    seen = set(zip(*(words_train[i:] for i in range(n))))
    grams = list(zip(*(words_val[i:] for i in range(n))))
    rate = sum(gram not in seen for gram in grams) / len(grams)
    print(f'unseen held-out {n}-grams: {rate:.1%} '
          f'({len(seen)} distinct in training)')
```

Roughly one in twelve held-out words never appeared in training, half of all
held-out word pairs are unseen, and nearly nine out of ten word triples have
count zero. Under the maximum likelihood estimate :eqref:`eq_ngram_mle`, every
unseen $n$-gram receives probability zero—and a single zero factor in
:eqref:`eq_lm_factorization` assigns probability zero to an entire valid
English passage.

### Word Frequency and Zipf's Law
:label:`subsec_natural-lang-stat`

Why do count tables become sparse so rapidly? The underlying cause is the
strongly skewed distribution of word frequencies in natural language. Using
the `Vocab` class from :numref:`sec_text-sequence`, which ranks tokens by
descending frequency, let us inspect the ten most common words in *The Time
Machine*.

```{.python .input #language-model-word-frequency-and-zipf-s-law-1}
words = text.split()
vocab = d2l.Vocab(words)
vocab.token_freqs[:10]
```

These ten most frequent words reveal nothing about time travel: they are
articles, pronouns, and prepositions—often called *stop words*—and almost any
Victorian novel would produce a nearly identical list. What is striking is how
steeply frequencies drop off: the tenth most common word is already less than
one-fifth as frequent as the first. Plotted on log–log axes, frequency versus
rank closely follows a straight line beyond the top few ranks, which is the
signature of a power law. Specifically, the frequency $n_i$ of the
$i^\textrm{th}$ most common word obeys *Zipf's law*:

$$n_i \propto \frac{1}{i^s}, \quad \textrm{equivalently} \quad \log n_i = -s \log i + c,$$
:eqlabel:`eq_zipf_law`

for an exponent $s$ and constant $c$. The helper `zipf_fit` below estimates $s$
and $c$ by fitting a straight line in log–log space over ranks 10 through
1,000.

```{.python .input #language-model-word-frequency-and-zipf-s-law-4}
def zipf_fit(freqs, lo=10, hi=1000):
    """Fit log n_i = -s log i + c over ranks lo..hi; return (s, c)."""
    ranks = np.arange(lo, hi + 1)
    slope, c = np.polyfit(np.log(ranks), np.log(freqs[lo - 1:hi]), 1)
    return -slope, c
```

Applying `zipf_fit` to the word frequencies `freqs` from `vocab.token_freqs`,
we print the fitted exponent $s$ and overlay the power-law line on the
empirical counts.

```{.python .input #language-model-word-frequency-and-zipf-s-law-2}
freqs = [freq for token, freq in vocab.token_freqs]
s, c = zipf_fit(freqs)
print(f'Zipf exponent s = {s:.2f}')
ranks = np.arange(1, len(freqs) + 1)
d2l.plot(ranks, [freqs, np.exp(c) * ranks ** -s], xlabel='rank $i$',
         ylabel='frequency $n_i$', xscale='log', yscale='log',
         legend=['words', 'Zipf fit'])
```

The fitted exponent sits close to 1, matching Zipf's classic observation
across natural languages. Power-law distributions are heavy-tailed: although
any individual rare word appears infrequently, there are so many distinct rare
words that together they account for a substantial share of any text. To see
whether bigrams and trigrams behave similarly, we compute their frequency
curves, fit their Zipf exponents, measure the fraction of distinct $n$-grams
that appear only once, and plot all three curves together.

```{.python .input #language-model-word-frequency-and-zipf-s-law-3}
counts = [sorted(collections.Counter(zip(*(words[i:] for i in range(n))))
                 .values(), reverse=True) for n in (1, 2, 3)]
for n, f in zip((1, 2, 3), counts):
    print(f'{n}-grams: s = {zipf_fit(f)[0]:.2f}, '
          f'{np.mean(np.array(f) == 1):.0%} of distinct {n}-grams occur once')
d2l.plot([np.arange(1, len(f) + 1) for f in counts], counts,
         xlabel='rank $i$', ylabel='frequency $n_i$', xscale='log',
         yscale='log', legend=['unigram', 'bigram', 'trigram'])
```

Word pairs and triples also follow power laws, albeit with smaller exponents
(roughly 0.7 and 0.5). Crucially for count-based models, the tail grows
dramatically heavier as $n$ increases: while half of all distinct words occur
only once, more than nine in ten distinct trigrams are singletons whose
probability estimates rest on a lone observation. It is this heavy tail that
produces the deluge of unseen trigrams on held-out text: any model that
estimates each context independently requires exponentially more data as the
context lengthens.

## Perplexity and Smoothing
:label:`subsec_perplexity`

### Perplexity

Inspection of the count tables revealed severe sparsity that our sample
generations had concealed. Before attempting to repair these tables, we need a
principled quantitative metric so we can tell whether a modification genuinely
improves generalization. Evaluating the joint probability of held-out text is
an obvious starting point, but raw sequence probabilities shrink exponentially
with document length: every model assigns a vastly smaller probability to
Tolstoy's *War and Peace* than to Saint-Exupéry's *The Little Prince*,
regardless of model quality. What we need is a per-token average.

Information theory gives us exactly that. In
:numref:`subsec_info_theory_basics`, we saw that the cross-entropy loss of a
probabilistic classifier equals the *surprisal* of the true label—the negative
logarithm of the probability assigned to it. Since a language model is a
next-token classifier at every position, we evaluate it by its average
surprisal across all $T$ tokens of the sequence:

$$\frac{1}{T} \sum_{t=1}^T -\log P(x_t \mid x_{<t}),$$
:eqlabel:`eq_avg_ce_for_lm`

where $P$ is the model's predictive distribution and $x_t$ is the actual token
at step $t$. Because this average normalizes by length, documents of different
lengths become comparable. In natural language processing, it is customary to
report the exponential of :eqref:`eq_avg_ce_for_lm`, known as *perplexity*
:cite:`Jelinek.Mercer.Bahl.ea.1977`:

$$\textrm{PPL} = \exp\left(-\frac{1}{T} \sum_{t=1}^T \log P(x_t \mid x_{<t})\right).$$

Equivalently, perplexity is the geometric mean of the inverse probability
assigned to each observed token. A model that assigns probability $1/4$ to
every target token achieves a perplexity of 4, matching the uncertainty of a
uniform choice among four equally likely candidates. Perplexity is therefore
naturally interpreted as the model's *effective branching factor* per token.
Three limiting cases anchor the scale:

* A model that always predicts the true token with probability 1 achieves the
  minimum possible perplexity of 1.
* A model that assigns probability 0 to even a single observed token incurs
  infinite perplexity.
* A baseline model predicting the uniform distribution over the vocabulary
  achieves perplexity $|\mathcal{V}|$, setting an upper bound that any useful
  model must beat.

Our unsmoothed maximum likelihood $n$-grams immediately run into the second
case: because they assign zero probability to every unseen $n$-gram, a single
novel trigram in the held-out text sends perplexity to infinity. To evaluate
and compare count models meaningfully, we must first *smooth* the counts to
eliminate zeros.

### Laplace Smoothing

The classical remedy is *Laplace smoothing* (also called add-$\alpha$
smoothing), which adds a positive constant $\alpha > 0$ to every count and
$\alpha |\mathcal{V}|$ to the denominator so that the resulting distribution
remains normalized:

$$\hat{P}(x_t \mid x_{t-n+1}, \ldots, x_{t-1}) = \frac{N(x_{t-n+1}, \ldots, x_{t}) + \alpha}{N(x_{t-n+1}, \ldots, x_{t-1}) + \alpha |\mathcal{V}|}.$$
:eqlabel:`eq_laplace_smoothing`

Every unseen $n$-gram now receives a small positive probability, and any
completely unseen context defaults to the uniform distribution
$1/|\mathcal{V}|$. The idea predates language modeling by two centuries:
Laplace introduced it with $\alpha = 1$ to estimate the probability that the
sun will rise tomorrow given that it has risen every day so far.

Let us first add a Laplace-smoothed `prob` method to `NGramLM` that implements
:eqref:`eq_laplace_smoothing` using the model's stored smoothing constant
`alpha`.

```{.python .input #language-model-laplace-smoothing-1}
@d2l.add_to_class(NGramLM)
def prob(self, context, token):
    """Laplace-smoothed P(token | last n-1 tokens of context)."""
    ctx = tuple(context[len(context) - self.n + 1:])
    num = self.cont.get(ctx, {}).get(token, 0) + self.alpha
    return num / (self.total.get(ctx, 0) + self.alpha * self.vocab_size)
```

With `prob` in place, `eval_nll` sums the negative log-likelihood across a
token sequence so we can compute held-out perplexity on `words_val` for each
$n$-gram order.

```{.python .input #language-model-perplexity}
def eval_nll(model, tokens):
    """Total negative log-likelihood of tokens, in nats."""
    return -sum(math.log(model.prob(tokens[max(0, t - model.n + 1):t],
                                    tokens[t]))
                for t in range(len(tokens)))

for n, model in models.items():
    ppl = math.exp(eval_nll(model, words_val) / len(words_val))
    print(f'{n}-gram held-out perplexity, Laplace smoothing: {ppl:7.1f}')
```

Under Laplace smoothing, the trigram model—the one whose generated samples
looked most convincing—actually suffers the *worst* held-out perplexity by a
wide margin, while the bigram barely edges past the unigram. Count sparsity
explains both outcomes: when 87% of held-out trigrams are unseen, the vast
majority of trigram predictions collapse into nearly uniform smoothed guesses.
Nor can tuning $\alpha$ save it: even the optimal $\alpha$ from exercise
:exref:`ex_language-model-smoothing-constant` leaves trigram perplexity above
2,400. Fluent samples turned out to be a misleading guide to generalization.

These numbers expose the two core flaws of Laplace smoothing. First, it
distributes probability mass equally across all unseen successors, ignoring
how plausible each word is in general. Second, it fails to fall back on
shorter contexts: an unseen trigram receives the same flat estimate regardless
of whether its final two words form a frequent bigram or an implausible pair.

### Kneser--Ney Smoothing
:label:`subsec_kneser-ney`

Kneser--Ney smoothing :cite:`Kneser.Ney.1995`, in the interpolated formulation
of :citet:`Chen.Goodman.1999`, addresses both flaws and reigned as the
strongest count-based language model for two decades. Denote by $u$ a context
of $n-1$ tokens, by $w$ the next token, and by $u'$ the shorter context
obtained by dropping the oldest token of $u$. The method combines three
interlocking ideas.

The first is *absolute discounting*. Empirical counts systematically
overestimate the true frequencies of rare events: an $n$-gram observed once in
training will, on average, appear fewer than once in a fresh sample of the
same size because of selection bias among the huge pool of rare $n$-grams.
Subtracting a fixed discount $0 < D < 1$ from every nonzero count corrects
this upward bias—most strongly for small counts, where the distortion is
largest—and frees a total probability mass of $D\,S(u)/N(u)$ in context $u$,
where $S(u)$ is the number of distinct token types observed immediately after
$u$ in training.

The second idea is *interpolation*: rather than spreading the discounted mass
uniformly, we allocate it to the next-lower-order estimate $P_\textrm{KN}(w \mid u')$,
which is defined recursively in the exact same way down to the uniform floor
$1/|\mathcal{V}|$:

$$P_\textrm{KN}(w \mid u) = \frac{\max(N(u, w) - D, 0)}{N(u)} + \frac{D\,S(u)}{N(u)}\, P_\textrm{KN}(w \mid u').$$
:eqlabel:`eq_kneser_ney`

Under this recursion, a frequently seen context relies primarily on its own
empirical counts, whereas a context observed with many diverse successors
shifts more weight to the shorter context—reflecting the intuition that a
context with many past continuations is also more likely to admit novel ones.
If a context $u$ was never observed in training at all, all probability mass
passes directly down: $P_\textrm{KN}(w \mid u) = P_\textrm{KN}(w \mid u')$.

The third idea changes *what* the lower orders count. Because the lower-order
distribution is invoked primarily to score words that have not appeared in the
longer context, it should measure how readily a word enters *new* contexts
rather than its raw corpus frequency. In our training text, `"traveller"`
appears 49 times, yet follows only two distinct words (`"time"` 48 times and
`"real"` once), whereas `"said"` appears 76 times after 48 different
predecessors. Raw unigram frequency would mistakenly predict `"traveller"`
almost as eagerly as `"said"` in an unfamiliar context, whereas counting
distinct predecessors recognizes that `"traveller"` rarely appears without
`"time"`. (The classic NLP example is `"Francisco"`, which is common in news
text yet almost exclusively preceded by `"San"`.) Kneser--Ney therefore
replaces $N(u, w)$ at every order below the highest with the *continuation
count*—the number of distinct tokens that precede $(u, w)$ in training—with
$N(u)$ and $S(u)$ computed analogously from those continuation counts.

Better still, the discount parameter $D$ has a clean closed-form estimate
derived by :citet:`Ney.Essen.Kneser.1994`: $D = c_1 / (c_1 + 2 c_2)$ at each
order, where $c_1$ and $c_2$ are the numbers of distinct $n$-grams at that
order occurring exactly once and twice, respectively (using continuation
counts below the highest order).

The function `kneser_ney` below implements :eqref:`eq_kneser_ney` in full: it
tallies the highest-order $n$-grams, derives continuation counts for every
lower order from the distinct tuples one order above, computes $N(u)$, $S(u)$,
and $D$, and evaluates the recursion across a token sequence, returning the
log-probability (in nats) of each token given its available prefix.

```{.python .input #language-model-kneser-ney-smoothing}
def kneser_ney(train, tokens, vocab_size, n=3):  #@save
    """Interpolated Kneser-Ney log-probabilities (nats) of each of tokens."""
    counts = {n: collections.Counter(zip(*(train[i:] for i in range(n))))}
    for k in range(n - 1, 0, -1):  # Lower orders: continuation counts
        counts[k] = collections.Counter(g[1:] for g in counts[k + 1])
    stats = {}
    for k, c in counts.items():
        c1, c2 = (sum(x == r for x in c.values()) for r in (1, 2))
        N, S = collections.Counter(), collections.Counter()
        for g, x in c.items():
            N[g[:-1]] += x
            S[g[:-1]] += 1
        stats[k] = c, N, S, c1 / max(c1 + 2 * c2, 1)
    logp = []
    for i in range(len(tokens)):
        p = 1 / vocab_size  # Below the unigram: uniform
        for k in range(1, min(n, i + 1) + 1):
            c, N, S, D = stats[k]
            g = tuple(tokens[i - k + 1:i + 1])
            if N[g[:-1]]:  # An unseen context passes p down unchanged
                p = (max(c[g] - D, 0) + D * S[g[:-1]] * p) / N[g[:-1]]
        logp.append(math.log(p))
    return logp
```

Before scoring the full validation split, let us watch Kneser--Ney at work on
our familiar context `"the time"`. Below, we print the trigram discount $D$
and compare the probabilities that Kneser--Ney and Laplace smoothing assign to
three candidate continuations: `"traveller"`, which followed `"the time"` in
47 of its 87 training occurrences, and two words that never followed it in
training, `"was"` and `"filby"`.

```{.python .input #language-model-kneser-ney-smoothing-1}
c3 = collections.Counter(zip(words_train, words_train[1:], words_train[2:]))
c1, c2 = (sum(x == r for x in c3.values()) for r in (1, 2))
print(f'trigram discount D = {c1 / (c1 + 2 * c2):.2f}')
for w in ('traveller', 'was', 'filby'):
    lp = kneser_ney(words_train, ['the', 'time', w], vocab_size)[-1]
    print(f'P({w} | the time): Kneser-Ney {math.exp(lp):.2g}, '
          f'Laplace {models[3].prob(["the", "time"], w):.2g}')
```

The trigram discount is roughly $0.9$, subtracting nearly an entire count from
every observed trigram—a vivid reflection of how noisy single-count
observations are. For a common trigram with count 47, subtracting $0.9$ barely
moves the needle, keeping `"traveller"` close to its empirical frequency of
$47/87$, whereas Laplace smoothing squanders nearly a third of that
probability mass across the vocabulary. The contrast on unseen continuations
is even sharper: Laplace smoothing treats `"was"` and `"filby"` identically,
whereas Kneser--Ney recognizes that `"was"` follows a wide variety of contexts
while `"filby"` appears in very few, separating their probabilities by more
than an order of magnitude. Let us measure how much this gains on the held-out
text.

```{.python .input #language-model-kneser-ney-smoothing-2}
for n in (1, 2, 3):
    kn = kneser_ney(words_train, words_val, vocab_size, n)
    print(f'{n}-gram held-out perplexity, Kneser-Ney: '
          f'{math.exp(-sum(kn) / len(kn)):7.1f}')
```

Kneser--Ney completely reverses the ranking. Even its unigram beats Laplace's,
its bigram cuts perplexity roughly in half, and its trigram outperforms all
other count models because unseen trigrams back off smoothly to well-calibrated
bigram and unigram continuation estimates rather than collapsing to a uniform
guess. Notice, though, that the trigram improves only modestly over the
bigram: because most held-out trigrams remain unseen, the trigram model falls
back to the bigram for the majority of positions. Achieving low perplexity on
*unseen* text is the defining test of a language model, and perplexity will
serve as our primary benchmark for the neural models that follow.

Still, Kneser--Ney shares a fundamental limitation with every count-based
method: while smoothing redistributes probability mass away from zeros, it
cannot infer which unseen combinations are semantically plausible. Because a
count table treats each token as an atomic, unrelated symbol, it has no way of
knowing that `"cat"` and `"feline"` share syntactic and semantic roles, or
that `"I want to eat, grandma"` is benign whereas `"I want to eat grandma"` is
alarming. Sharing statistical strength across related words and contexts
demands learned continuous representations—the exact motivation for turning to
neural networks in :numref:`sec_rnn`.

### Bits per Byte
:label:`subsec_bits-per-byte`

Perplexity suffers from one crucial limitation: it is tied to the choice of
tokenization. Because a character model chooses among 27 symbols at each step
while a word model chooses among thousands, their raw perplexities are on
completely different scales and cannot be compared directly. Yet tokenization
is an internal design choice (:numref:`sec_text-sequence`), and a fair
evaluation metric must be invariant to it. Fortunately, the remedy is simple:
replace the denominator. Instead of dividing total surprisal by the number of
tokens, we divide by the number of *bytes* in the evaluated text, a quantity
determined solely by the text itself. Measuring surprisal in bits (base-2
logarithms) gives *bits per byte* (bpb):

$$\textrm{bpb} = \frac{1}{B} \sum_{t=1}^T -\log_2 P(x_t \mid x_{<t}),$$
:eqlabel:`eq_bpb`

where $B$ is the length of the evaluated text in UTF-8 bytes. Bits per byte
has an immediate operational meaning as a lossless compression rate: paired
with an arithmetic coder, a model achieving $\textrm{bpb}$ bits per byte
compresses the text to $\textrm{bpb}/8$ of its original byte size.

To compare perplexity and bits per byte directly, we fit trigram models—first
with Laplace smoothing and then with Kneser--Ney—on the same normalized
training text under three tokenizations: characters, words, and a 1,024-token
BPE vocabulary (trained on the normalized training split using the tokenizer
from :numref:`sec_text-sequence`). One subtlety is required to keep the
comparison honest: whenever the word model predicts `<unk>`, it has not yet
specified *which* out-of-vocabulary word occurred. To encode the held-out text
losslessly, the word model must still spell each unknown word out, so we add
$\log_2 27$ bits per letter plus the trailing space—the cost of coding each
character uniformly from the 27 normalized symbols.

```{.python .input #language-model-bits-per-byte}
bpe = d2l.BPETokenizer(vocab_size=1024, pattern=d2l.BPETokenizer.GPT2_PATTERN)
bpe.train(text[:split])
streams = {'character': (list(text[:split]), list(text[split:]),
                         len(set(text))),
           'BPE (1,024)': (bpe.encode(text[:split]), bpe.encode(text[split:]),
                           bpe.vocab_size),
           'word': (words_train, words_val, vocab_size)}
n_bytes = len(text[split:].encode('utf-8'))
spell = sum((len(w) + 1) * math.log(27)  # Nats to spell out the <unk> words
            for w in text[split:].split() if w not in known)
print(f'{"tokenization":>12} {"|V|":>5} {"tokens":>7} {"ppl":>7} '
      f'{"bpb":>5} {"KN bpb":>6}')
for name, (train, val, m) in streams.items():
    nll = eval_nll(NGramLM(train, 3, m), val)
    nll_kn = -sum(kneser_ney(train, val, m))
    extra = spell if name == 'word' else 0
    print(f'{name:>12} {m:>5} {len(val):>7} {math.exp(nll / len(val)):>7.1f} '
          f'{(nll + extra) / math.log(2) / n_bytes:>5.2f} '
          f'{(nll_kn + extra) / math.log(2) / n_bytes:>6.2f}')
```

Measured by raw perplexity, the three Laplace-smoothed models appear separated
by more than two orders of magnitude; measured by bits per byte, they lie
within a third of a bit of one another, and the character model—seemingly far
ahead in perplexity—no longer wins. For a fixed Markov order $n$, coarser
tokens span longer stretches of text: two words of context reach back roughly
ten characters, whereas two characters scarcely cover a syllable. The word
model would lead were it not penalized by out-of-vocabulary words, whose
spelling cost adds roughly $0.6$ bits per byte and drops it to last place
under Laplace smoothing. Byte-level BPE suffers no OOV penalty at all,
achieving the strongest compression and providing yet another reason modern
language models adopt subword tokenization.

In the final column, Kneser--Ney trims more than half a bit per byte from both
the BPE and word trigrams while leaving the character trigram virtually
unchanged, since with only 27 symbols very few character trigrams are unseen
and smoothing has little work to do. At roughly $1.9$ bits per byte, the
Kneser--Ney BPE trigram is our strongest count-based baseline.

Bits per byte is therefore the standard metric when comparing modern language
models built on different vocabularies (for instance, a 50,000-token model
against a 200,000-token model). Three benchmarks help calibrate the scale:
Shannon's classic human-prediction experiments estimated the entropy of
printed English at roughly 1 bit per character :cite:`Shannon.1951`; modern
neural language models compress diverse web text to well below 1 bit per byte;
and the count-based models of this section sit between roughly $1.9$ and $2.8$
bits per byte.

## Partitioning Sequences
:label:`subsec_partitioning-seqs`

The neural language models in the upcoming sections ingest minibatches of
fixed-length BPE token subsequences produced by the tokenizer from
:numref:`sec_text-sequence`. Let us assemble that data pipeline now.

Picture the tokenized corpus as a single contiguous stream of token ids
$x_1, x_2, x_3, \ldots$. An input window of length $T$ starting at index $t$
contains $x_t, \ldots, x_{t+T-1}$, and its paired target is the same window
shifted one token to the right, $x_{t+1}, \ldots, x_{t+T}$. As the model steps
through the input window, it predicts the next token at every position, so
each window supplies $T$ supervised next-token targets. Every starting offset
in the corpus except the final $T$ tokens defines a valid window, with
adjacent windows overlapping in all but one token
(:numref:`fig_lang_model_data`). During training, we sample minibatches of
these windows at random to provide the shuffling that stochastic gradient
descent requires.

![The first two input windows of length $T = 5$ over the opening BPE tokens of *The Time Machine* (␣ marks a leading space). Each target is its input shifted forward by one token, and consecutive windows overlap in all but one token; a minibatch gathers windows drawn at random.](../img/mdl-rnn-partitioning.svg)
:label:`fig_lang_model_data`

The `TimeMachine` constructor below puts the pipeline together: it downloads
the novella, trains the BPE tokenizer from :numref:`sec_text-sequence`,
encodes the text into integer ids, and stacks all overlapping windows of
length `num_steps` into aligned input (`X`) and target (`Y`) arrays. The
trained tokenizer is stored on both `data.tokenizer` and `data.vocab` so that
we can decode ids back to text and query `len(data.vocab)` uniformly across
tokenization modes.

```{.python .input #language-model-partitioning-sequences-1}
%%tab mxnet
@d2l.add_to_class(d2l.TimeMachine)  #@save
def __init__(self, batch_size, num_steps, num_train=50000, num_val=5000,
             tokenization='bpe', vocab_size=1024):
    super(d2l.TimeMachine, self).__init__()
    self.save_hyperparameters()
    raw_text = self._download()
    if tokenization == 'bpe':
        self.tokenizer = d2l.BPETokenizer(
            vocab_size, pattern=d2l.BPETokenizer.GPT2_PATTERN)
        self.tokenizer.train(raw_text)
        corpus, self.vocab = self.tokenizer.encode(raw_text), self.tokenizer
    else:  # tokenization='char': characters, via TimeMachine.build
        corpus, self.vocab = self.build(raw_text)
    windows = [corpus[i:i+num_steps+1] for i in range(len(corpus)-num_steps)]
    # Drop the windows that straddle the split, so that no validation token
    # is a training target
    windows = windows[:num_train] + windows[num_train+num_steps:]
    array = d2l.tensor(windows, dtype=d2l.int32)  # Integer ids, not float32
    self.X, self.Y = array[:,:-1], array[:,1:]
```

```{.python .input #language-model-partitioning-sequences-1}
%%tab pytorch, tensorflow, jax
@d2l.add_to_class(d2l.TimeMachine)  #@save
def __init__(self, batch_size, num_steps, num_train=50000, num_val=5000,
             tokenization='bpe', vocab_size=1024):
    super(d2l.TimeMachine, self).__init__()
    self.save_hyperparameters()
    raw_text = self._download()
    if tokenization == 'bpe':
        self.tokenizer = d2l.BPETokenizer(
            vocab_size, pattern=d2l.BPETokenizer.GPT2_PATTERN)
        self.tokenizer.train(raw_text)
        corpus, self.vocab = self.tokenizer.encode(raw_text), self.tokenizer
    else:  # tokenization='char': characters, via TimeMachine.build
        corpus, self.vocab = self.build(raw_text)
    windows = [corpus[i:i+num_steps+1] for i in range(len(corpus)-num_steps)]
    # Drop the windows that straddle the split, so that no validation token
    # is a training target
    windows = windows[:num_train] + windows[num_train+num_steps:]
    array = d2l.tensor(windows)
    self.X, self.Y = array[:,:-1], array[:,1:]
```

:begin_tab:`mxnet`
MXNet arrays hold float32 numbers unless told otherwise, so the MXNet
constructor asks for integer token ids explicitly.
:end_tab:

We reserve the first `num_train` windows for training and `num_val` subsequent
windows for validation. Because adjacent windows overlap heavily, the windows
immediately following the training split would contain tokens that already
appeared as training targets; dropping a buffer of `num_steps` boundary
windows prevents any data leakage between training and validation. During
validation, minibatches are iterated sequentially without shuffling.

```{.python .input #language-model-partitioning-sequences-2}
@d2l.add_to_class(d2l.TimeMachine)  #@save
def get_dataloader(self, train):
    idx = slice(0, self.num_train) if train else slice(
        self.num_train, self.num_train + self.num_val)
    return self.get_tensorloader([self.X, self.Y], train, idx)
```

Each minibatch consists of two integer tensors of shape
(`batch_size`, `num_steps`). Decoding a sample row confirms that `X` contains
Wells' prose and that `Y` is shifted forward by exactly one token.

```{.python .input #language-model-partitioning-sequences-3}
data = d2l.TimeMachine(batch_size=2, num_steps=10)
for X, Y in data.train_dataloader():
    print('X:', X, '\nY:', Y)
    break
print('X[0]:', repr(data.tokenizer.decode([int(i) for i in X[0]])))
print('Y[0]:', repr(data.tokenizer.decode([int(i) for i in Y[0]])))
```

A window of BPE tokens supplies far more context than any count table can
exploit—our word tables already ran out of data at just two words of history.
In the next section, we replace lookup tables with a neural network that
compresses an arbitrarily long prefix into a learned hidden state.

## Summary

A language model assigns probabilities to token sequences, and the chain rule
of probability reduces this task to next-token prediction, enabling the same
model to both score and generate text. Under a Markov assumption, $n$-gram
models estimate next-token probabilities directly from corpus counts; although
simple to construct and capable of locally fluent samples, they suffer from
exponentially growing context spaces and Zipfian heavy tails, leaving most
observed $n$-grams as singletons and causing the models to memorize rather
than generalize. Perplexity—the exponential of average per-token
cross-entropy—measures a model's effective branching factor and exposes why
Laplace-smoothed trigrams perform poorly on held-out text. Interpolated
Kneser--Ney smoothing discounts observed counts and redistributes the freed
mass to lower-order continuation counts, yielding the strongest count-based
models, yet it still cannot generalize across semantically related tokens.
Because perplexity depends on vocabulary granularity, bits per byte provides
a tokenization-invariant compression rate for fair cross-model comparisons.
Finally, to feed neural language models in the sections ahead, we convert the
BPE-tokenized corpus into overlapping fixed-length input–target windows with a
clean gap preventing leakage into the validation split.

## Exercises

1. **Dense versus sparse storage.** Consider a 5-gram model over a
   vocabulary of $|\mathcal{V}| = 50{,}000$ word tokens.
    1. How many entries does a dense table of the conditional
       probabilities :eqref:`eq_ngram_mle` contain?
    1. Suppose instead that only the 5-grams observed in a trillion-token
       corpus are stored, at 16 bytes per entry. How much memory is needed
       at most, and what fraction of all possible 5-grams is covered?
    1. The word vocabulary of :numref:`subsec_markov-models-and-n-grams`
       has $|\mathcal{V}| = 4{,}316$ entries, and the code there prints the
       number of distinct word trigrams in the training text. Which
       fraction of all possible trigrams occurs in training? Which fraction
       of the held-out trigrams does? Is the coverage computed in (b)
       therefore a good estimate of how often a query on new text finds
       its 5-gram in the table?

1. [code] **Smoothing constant.** Sweep $\alpha$ in
   :eqref:`eq_laplace_smoothing` over the 19 values
   $\alpha = 10^{k/4}$ for $k = -16, -15, \ldots, 2$, for the unigram,
   bigram, and trigram word models, and plot held-out perplexity against
   $\alpha$ on a logarithmic axis.
    1. How does the best $\alpha$ change with $n$, and why?
    1. Rank the three models, each at its own best $\alpha$, by held-out
       perplexity. How does this ranking differ from the one printed in
       :numref:`subsec_perplexity` with the default $\alpha = 0.01$ of
       `NGramLM`?
   :exlabel:`ex_language-model-smoothing-constant`

1. **Perplexity under skew.** A training corpus over ten symbols consists
   of one symbol 91% of the time and each of the other nine 1% of the
   time.
    1. Compute the perplexity of the unigram model estimated from this
       corpus on a long test text drawn from the same distribution. How
       does it compare with the vocabulary size of ten?
    1. The word-level unigram model of :numref:`subsec_perplexity` chooses
       among thousands of word types. Read off its held-out perplexity as
       an effective number of equally likely choices, and explain with
       Zipf's law :eqref:`eq_zipf_law` why this number is far below
       $|\mathcal{V}|$.

    *Adapted from Daniel Jurafsky and James H. Martin,
    [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/),
    3rd ed. draft, Exercise 3.12.*

1. **Overlapping windows.** `d2l.TimeMachine` builds one row of
   `num_steps + 1` consecutive tokens for every possible start position
   in the corpus. The first `num_steps` tokens of a row form a row of `X`,
   and the last `num_steps` tokens form the matching row of `Y`. The first
   `num_train` rows go to training, the next `num_steps` rows are dropped,
   and the `num_val` rows after them go to validation
   (:numref:`subsec_partitioning-seqs`). Take `num_steps=32`,
   `num_train=10000`, and `num_val=5000`.
    1. Counting input and target tokens together, which corpus positions do
       the training rows touch, and which do the validation rows touch?
       How many positions would both touch if no rows were dropped, and
       why would that inflate the validation score?
    1. How many times per epoch does a token in the interior of the
       training range serve as a target?
    1. How many windows does a partition of the entire corpus of about
       66,000 tokens into non-overlapping 32-token windows yield? Compare
       this partition with the 10,000 overlapping training rows in the
       number of windows and in the number of distinct corpus positions
       covered.

1. [code] **Normalization across lengths.** ● `NGramLM` has no
   end-of-sequence token. Train a bigram `NGramLM` with $\alpha = 1$ and
   `vocab_size=3` on a short string over a three-symbol vocabulary. Use
   `prob` and the chain rule :eqref:`eq_lm_factorization` to sum the
   probabilities of all sequences of length 2, and separately those of all
   sequences of length 3.
    1. What does each sum equal, and what do the two sums say about the
       model as a distribution over sequences whose length is not fixed in
       advance?
    1. Add an end-of-sequence token to the vocabulary, so that
       `vocab_size=4`, and append it to the training string. Treat a
       sequence as complete once it emits this token. Show that the
       probabilities of all complete sequences now sum to exactly one.
       Which property of the model rules out probability mass on
       sequences that never end?

    *Adapted from Daniel Jurafsky and James H. Martin,
    [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/),
    3rd ed. draft, Exercise 3.5.*

1. [code] **Seen and unseen trigrams.** ● Split the held-out positions
   from the third word on by whether the word trigram ending at that
   position occurred in training. Take the bigram and the trigram word
   models, each at its best $\alpha$ from the sweep of
   problem :exref:`ex_language-model-smoothing-constant`,
   and compute the average negative log-likelihood of each model
   on each group. On which group is the trigram the better model, and how
   does Kneser--Ney :eqref:`eq_kneser_ney` exploit this?

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/117)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/118)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/1049)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/18012)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §8.3]{.kicker}

Language Models, Perplexity, and Bits per Byte<br>**chain-rule factorization · n-gram MLE · Zipf's law · perplexity · Laplace vs. Kneser–Ney smoothing · bits per byte · sliding windows**
:::
:::

::: {.slide title="The Core Problem of Statistical Language Modeling"}
[Motivation & Chain-Rule Formulation]{.kicker}

::: {.cols .vc}
::: {.col}
$$P(x_1, \ldots, x_T) = \prod_{t=1}^T P(x_t \mid x_{<t})$$

- **Overcoming the $10^{30}$ sequence explosion:** Over a 1,024-token vocabulary, $1024^{10} \approx 10^{30}$ length-10 sequences exist, yet the chain rule reduces joint density estimation to next-token classification at each step $t$.
- **One distribution, two operations:** The conditionals $P(x_t \mid x_{<t})$ both **score** candidate passages (e.g., disambiguating acoustic homophones in ASR) and **generate** new text via left-to-right ancestral sampling.
:::

::: {.col .fig}
@fig:ch08-rnn-lm-scoring-generation
:::
:::
:::

::: {.slide title="Markov Truncation and n-Gram Maximum Likelihood"}
[Count-Based Language Models]{.kicker}

Because the full prefix $x_{<t}$ grows without bound, an **$n$-gram language model** applies an $(n - 1)^\textrm{th}$-order Markov approximation and estimates each conditional probability by the empirical ratio of training corpus counts $N(\cdot)$:

$$\hat{P}(x_t \mid x_{<t}) \approx \hat{P}(x_t \mid x_{t-n+1}, \ldots, x_{t-1}) = \frac{N(x_{t-n+1}, \ldots, x_t)}{N(x_{t-n+1}, \ldots, x_{t-1})}$$

@!language-model-markov-models-and-counting-3

- **Orders $n = 1, 2, 3$:** Unigram ($n = 1$, zero context), bigram ($n = 2$, 1 preceding token), and trigram ($n = 3$, 2 preceding tokens).
- **Local phrase capture vs. rigid determinism:** After `"the time"` ($N = 87$), the word trigram assigns $0.54$ to `"traveller"` and $0.34$ to `"machine"`, yet **$83\%$ of all observed word pairs in training are followed by only a single unique continuation**!
:::

::: {.slide title="Building and Sampling Count-Based n-Gram Models"}
[From-Scratch Implementation]{.kicker}

`NGramLM` maps every observed $(n - 1)$-token context to a `Counter` of its successors:

@-language-model-markov-models-and-counting-2
:::

::: {.slide title="The Memorization Illusion in Higher-Order n-Grams"}
[Qualitative Sampling vs. True Generalization]{.kicker}

To test whether fluent-looking samples reflect generalization or verbatim copying, `longest_copy` measures the longest contiguous span of generated words copied directly from the training text:

@!language-model-sampling-from-n-gram-models

::: {.d2l-note .warn}
**Why Sample Inspection Can Mislead:** The unigram emits disconnected word salad; the bigram chains locally valid pairs; and the trigram sounds remarkably like H. G. Wells—**because it copies 7 to 12 consecutive words verbatim** whenever the sampler lands on one of the $83\%$ of word pairs that had only a single training continuation.
:::
:::

::: {.slide title="The Sparsity Wall: Exponential Context Growth"}
[Statistical Bottleneck of Count Tables]{.kicker}

Why do higher-order $n$-grams collapse into memorization on training text and fail on held-out text? The number of *possible* $n$-grams grows exponentially as $|\mathcal{V}|^n$, while corpus size grows only linearly:

@!language-model-the-sparsity-wall

- **Catastrophic zero counts on held-out text:** On the final $10\%$ of *The Time Machine*, $8.1\%$ of unigrams (mapped to `<unk>`), $52.8\%$ of word bigrams, and **$87.0\%$ of word trigrams** never occurred in the training split.
- **Storage complexity:** Instead of allocating a dense $|\mathcal{V}|^n$ array ($4316^3 \approx 8 \times 10^{10}$ cells), a nested hash map `self.cont[ctx][token]` stores only the $n$-grams that actually occur with nonzero count in training.
- **Zero-probability collapse:** Under maximum likelihood estimation ($\hat{P} = N(\text{trigram}) / N(\text{bigram})$), any unseen $n$-gram receives probability $0$, causing a single novel word triple to assign $\prod_{t=1}^T \hat{P}(x_t \mid x_{<t}) = 0$ to the entire validation passage.
:::

::: {.slide title="Zipf's Power Law in Natural Language Frequencies"}
[Natural Language Statistics]{.kicker}

Count tables go sparse so rapidly because word frequencies in natural language are extremely skewed, following **Zipf's power law** across ranks $i = 1, 2, \ldots, |\mathcal{V}|$:

::: {.cols .vc}
::: {.col}
@!language-model-word-frequency-and-zipf-s-law-1

- **Stop-word dominance:** The top 10 words (*the, I, and, of, a, to, was, in, that, my*) are function words; rank 10 (`'my'`, $440$) is already $5\times$ rarer than rank 1 (`'the'`, $2{,}261$).
- **Linear log–log slope:** Fitting over ranks $10\text{--}1000$ gives Zipf exponent $s = 1.03 \approx 1$.
:::

::: {.col .fig .narrow}
$$n_i \propto \frac{1}{i^s} \iff \log n_i = -s \log i + c$$

@!language-model-word-frequency-and-zipf-s-law-2
:::
:::
:::

::: {.slide title="Heavy Singleton Tails Across Unigrams to Trigrams"}
[Zipfian Sparsity Scaling]{.kicker}

::: {.cols .vc}
::: {.col}
Plotting rank–frequency curves for word unigrams, bigrams, and trigrams on the same log–log axes reveals how power-law tails flatten and become dominated by singletons ($N = 1$) as $n$ increases:

- **Flattening Zipf exponents:** The power-law slope drops from $s = 1.03$ (unigrams) to $s = 0.68$ (bigrams) and $s = 0.49$ (trigrams).
- **Singleton explosion:** Exactly **$50\%$ of distinct words**, **$80\%$ of distinct bigrams**, and **$92\%$ of distinct trigrams** in *The Time Machine* occur only *once* in the entire corpus.
- **Implication:** Estimating each context tuple independently requires exponentially larger corpora for each additional token of context.
:::

::: {.col .fig .narrow}
@!language-model-word-frequency-and-zipf-s-law-3
:::
:::
:::

::: {.slide title="Perplexity: Effective Branching Factor per Token"}
[Quantitative Evaluation Metric]{.kicker}

::: {.cols .vc}
::: {.col}
$$\textrm{PPL} = \exp\left(-\frac{1}{T}\sum_{t=1}^T \log P(x_t \mid x_{<t})\right)$$

- **Effective branching factor:** Assigning probability $1/k$ to every target token yields $\textrm{PPL} = k$—matching the uncertainty of rolling a fair $k$-sided die at each step.
- **Calibration scale:** Ranges from $\textrm{PPL} = 1$ (deterministic oracle) to $|\mathcal{V}|$ (uniform guessing) and $\infty$ whenever any test token receives probability $0$.
:::

::: {.col .fig}
@fig:ch08-rnn-perplexity-scale
:::
:::
:::

::: {.slide title="Laplace (Add-α) Smoothing and Its Failure Modes"}
[Classical Count Smoothing]{.kicker}

**Laplace (add-$\alpha$) smoothing** eliminates zero probabilities by adding a pseudocount $\alpha > 0$ to every $n$-gram numerator and $\alpha|\mathcal{V}|$ to the context denominator:

$$\hat{P}_\alpha(x_t \mid x_{t-n+1}, \ldots, x_{t-1}) = \frac{N(x_{t-n+1}, \ldots, x_t) + \alpha}{N(x_{t-n+1}, \ldots, x_{t-1}) + \alpha |\mathcal{V}|}$$

@!language-model-perplexity

- **The fluency-versus-perplexity paradox:** Under Laplace smoothing ($\alpha = 0.01$), the trigram—whose generated samples looked most like Wells—suffers the **worst held-out perplexity ($2{,}481.2$ vs. $584.5$ for the bigram)**!
- **Why Laplace smoothing fails at $n = 3$:** Because $87\%$ of validation trigrams are unseen, Laplace smoothing predicts a flat uniform distribution $1/|\mathcal{V}|$ on $87\%$ of test positions, ignoring both general word frequency and the observed bigram context $(x_{t-1})$.
:::

::: {.slide title="Interpolated Kneser–Ney: Discounting & Backoff"}
[State-of-the-Art Count Smoothing]{.kicker}

::: {.cols .vc}
::: {.col}
$$P_{\textrm{KN}}(w \mid u) = \frac{\max(N(u, w) - D,\, 0)}{N(u)} + \frac{D\,S(u)}{N(u)}\,P_{\textrm{KN}}(w \mid u')$$

- **Absolute discounting ($D \approx 0.93$):** Subtracts $D = c_1 / (c_1 + 2c_2)$ from every nonzero count to correct singleton selection bias, freeing mass $D\,S(u)/N(u)$.
- **Recursive interpolation:** Allocates the freed mass to the backed-off suffix context $u' = u_{2:}$ down to the uniform floor $1/|\mathcal{V}|$.
- **Continuation counts at lower orders:** Lower orders count how many *distinct predecessors* precede $w$ (rewarding versatile `"said"` over context-bound `"traveller"`).
:::

::: {.col .fig}
@fig:ch08-rnn-kneser-ney-flow
:::
:::
:::

::: {.slide title="Implementing Interpolated Kneser–Ney in Python"}
[From-Scratch Implementation]{.kicker}

`kneser_ney` derives continuation counts, estimates $D_k = c_1 / (c_1 + 2c_2)$, and evaluates the recursion:

@-language-model-kneser-ney-smoothing
:::

::: {.slide title="Kneser–Ney vs. Laplace on Seen and Unseen Words"}
[Empirical Smoothing Comparison]{.kicker}

Comparing Kneser–Ney against Laplace ($\alpha = 0.01$) after context `"the time"` ($N = 87$) and across the entire validation split demonstrates why Kneser–Ney dominated statistical NLP for two decades:

@!language-model-kneser-ney-smoothing-1

@!language-model-kneser-ney-smoothing-2

- **Preserving frequent $n$-grams:** For seen `"traveller"` ($N = 47$), subtracting $D \approx 0.93$ keeps $P_{\textrm{KN}} = 0.53 \approx 47/87$, whereas Laplace wastes a third of the mass ($\hat{P}_\alpha = 0.36$) on $4{,}316 \times 0.01 = 43.16$ phantom counts.
- **Discriminating unseen continuations:** Laplace gives unseen `"was"` and `"filby"` the exact same probability ($7.7 \times 10^{-5}$), whereas Kneser–Ney scores versatile `"was"` **$16\times$ higher** than context-bound `"filby"`, cutting held-out trigram perplexity by $8\times$ ($2{,}481.4 \to 300.1$).
:::

::: {.slide title="Why Perplexity Cannot Compare Different Tokenizers"}
[Tokenizer-Invariant Evaluation]{.kicker}

::: {.cols .vc}
::: {.col}
$$\textrm{bpb} = \frac{1}{B}\sum_{t=1}^T -\log_2 P(x_t \mid x_{<t}) = \frac{T}{B}\log_2 \textrm{PPL}$$

- **The scale mismatch:** Perplexity divides by token count $T$, so a character model ($\textrm{PPL} \approx 6.4$), BPE-1024 model ($\textrm{PPL} \approx 63$), and word model ($\textrm{PPL} \approx 300$) lie on incompatible scales.
- **Bits per Byte ($\textrm{bpb}$):** Normalizing total base-2 surprisal by the invariant UTF-8 byte length $B$ measures the exact lossless compression rate ($\textrm{bpb}/8$).
:::

::: {.col .fig}
@fig:ch08-rnn-ppl-vs-bpb
:::
:::
:::

::: {.slide title="Comparing Characters, BPE, and Words in Bits/Byte"}
[Cross-Granularity Benchmark]{.kicker}

Evaluating trigram models across character, BPE-1024, and word tokenizations on the exact same held-out text (adding $\log_2 27$ bits/letter to spell out each `<unk>` token in the word model) places all three on one scale:

@!language-model-bits-per-byte

- **From $400\times$ PPL gap to $0.2$ bpb:** While Laplace perplexities span $6.4$ to $2{,}481.4$, Laplace bits per byte sit within $2.58\text{--}2.81$ bpb.
- **Why BPE-1024 wins ($1.90$ KN bpb):** Two BPE tokens span $\sim 5.6$ characters (vs. 2 characters for a character trigram) while incurring **zero `<unk>` spelling penalty** (which costs the word trigram $\approx 0.6$ bpb).
:::

::: {.slide title="Sliding Input–Target Windows for Neural Training"}
[Sequence Data Partitioning]{.kicker}

To train the neural language models of §{sec_rnn}–§{sec_decoding} with minibatch SGD, we slice the BPE-encoded corpus $x_1, x_2, \ldots$ into overlapping windows of length $T = \texttt{num\_steps}$:

::: {.cols .vc}
::: {.col}
- **Aligned input and 1-step shifted target:** A window starting at index $i$ forms input row $\mathbf{X}_{i,:} = (x_i, \ldots, x_{i+T-1})$ and target row $\mathbf{Y}_{i,:} = (x_{i+1}, \ldots, x_{i+T})$.
- **Dense supervision ($T$ targets per row):** Every step $t \in \{1, \ldots, T\}$ inside a window predicts the next token $x_{i+t}$ from prefix $(x_i, \ldots, x_{i+t-1})$.
- **Random minibatch sampling:** Shuffling window start indices into minibatches of shape `(batch_size, num_steps)` decorrelates gradients across SGD updates.
:::

::: {.col .fig .big}
![](../img/mdl-rnn-partitioning.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="Preventing Train–Validation Leakage Across Windows"}
[Data Pipeline Implementation & Leakage Guard]{.kicker}

@-language-model-partitioning-sequences-1
:::

::: {.slide title="Preventing Train–Validation Leakage Across Windows (cont.)"}
[Data Pipeline Implementation & Leakage Guard]{.kicker}

::: {.cols .vc}
::: {.col}
- **Why drop `num_steps` boundary windows:** Adjacent length-$T$ windows overlap in $T - 1$ tokens; skipping `windows[num_train : num_train + num_steps]` ensures no validation token ever appeared as a training target.
:::

::: {.col .fig}
@fig:ch08-rnn-window-leakage-guard
:::
:::
:::

::: {.slide title="Inspecting Minibatches from TimeMachine Loader"}
[Verifying Tensor Shapes & Alignment]{.kicker}

Each minibatch from `data.train_dataloader()` yields two integer ID tensors `(X, Y)` of shape `(batch_size, num_steps)` where every entry of `Y` is `X` shifted one column to the left:

@language-model-partitioning-sequences-3
:::

::: {.slide title="Concept Check: Smoothing, Perplexity & Leakage"}
[Check Your Understanding]{.kicker}

Work through these three quantitative questions on smoothing, bits per byte, and window partitioning:

1. **Laplace Pseudocount Dilution:** In *The Time Machine*, `"the time"` occurs $N(u) = 87$ times and is followed by `"traveller"` $N(u, w) = 47$ times ($|\mathcal{V}| = 4{,}316$). Compute the Laplace-smoothed probability $\hat{P}_\alpha(\textrm{traveller} \mid \textrm{the time})$ at $\alpha = 0.01$ and at $\alpha = 1.0$. Why does $\alpha = 1.0$ collapse?
2. **Perplexity to Bits per Byte:** A neural BPE language model achieves validation perplexity $\textrm{PPL} = 64.0$ on text where each BPE token averages $2.80$ UTF-8 bytes. What is its bits per byte ($\textrm{bpb}$), and to what percentage of the original UTF-8 byte size could an arithmetic coder compress the text?
3. **Continuation Counts in Kneser–Ney:** The word `"Francisco"` has high raw frequency in news corpora, yet appears almost exclusively after `"San"`. Compare its unigram MLE probability against its Kneser–Ney unigram continuation probability after an unseen bigram context like `"ate ___"`.

:::

::: {.slide title="Concept Check: Smoothing, Perplexity & Leakage (Answers)"}
[Check Your Understanding]{.kicker}

::: {.d2l-note}
**Answers:**

1. **$\alpha = 0.01$:** $(47 + 0.01) / (87 + 43.16) = 47.01 / 130.16 \approx 0.361$. **$\alpha = 1.0$:** $(47 + 1) / (87 + 4316) = 48 / 4403 \approx 0.0109$—the $4{,}316$ uniform pseudocounts swamp the $87$ real observations by $50\times$.
2. **$\textrm{bpb} = \log_2(64.0) / 2.80 = 6.00 / 2.80 \approx 2.14\text{ bits/byte}$**, corresponding to a lossless compression ratio of $2.14 / 8 \approx 26.8\%$ of the original byte size.
3. **Unigram MLE vs. Continuation:** Raw unigram MLE scores `"Francisco"` high based on total occurrences, whereas Kneser–Ney continuation counts how many *distinct predecessors* precede `"Francisco"` ($\approx 1$, `"San"`), assigning it near-zero probability after `"ate"`.
:::
:::

::: {.slide title="Summary: Language Modeling & Evaluation Metrics"}
[Section Takeaways]{.kicker}

| Model / Metric | Mathematical Definition | Core Strength | Fundamental Limitation |
|---|---|---|---|
| **$n$-Gram MLE** | $\hat{P}(x_t \mid u) = \frac{N(u, x_t)}{N(u)}$ for $\|u\| = n - 1$ | Fast single-pass counting; captures frequent local phrases | $87\%$ of held-out word trigrams have $N = 0 \implies \textrm{PPL} = \infty$ |
| **Laplace (Add-$\alpha$) Smoothing** | $\hat{P}_\alpha(x_t \mid u) = \frac{N(u, x_t) + \alpha}{N(u) + \alpha \lvert\mathcal{V}\rvert}$ | Guarantees $\hat{P}_\alpha > 0$ everywhere | Dilutes seen counts; backs off to flat uniform $1/\lvert\mathcal{V}\rvert$ on unseen contexts |
| **Interpolated Kneser–Ney** | $\frac{\max(N(u,w)-D, 0)}{N(u)} + \frac{D\,S(u)}{N(u)} P_{\textrm{KN}}(w \mid u')$ | Discounts singletons ($D \approx 0.9$); backs off to continuation counts | Treats tokens as discrete orthogonal symbols (`"cat"` $\perp$ `"feline"`) |
| **Perplexity ($\textrm{PPL}$)** | $\exp\bigl(-\frac{1}{T}\sum_{t=1}^T \log P(x_t \mid x_{<t})\bigr)$ | Length-normalized effective branching factor per token | Scale depends on tokenizer vocabulary size $\lvert\mathcal{V}\rvert$ and granularity |
| **Bits per Byte ($\textrm{bpb}$)** | $\frac{1}{B}\sum_{t=1}^T -\log_2 P(x_t \mid x_{<t}) = \frac{T}{B}\log_2 \textrm{PPL}$ | Tokenizer-invariant lossless compression rate ($\textrm{bpb}/8$) | Requires explicit OOV character-spelling penalty if tokenizer has `<unk>` |
:::

