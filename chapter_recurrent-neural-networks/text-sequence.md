# Converting Raw Text into Sequence Data
:label:`sec_text-sequence`

Throughout this book,
we will often work with text data
represented as sequences
of words, characters, or word pieces.
To get going, we will need some basic
tools for converting raw text
into sequences of the appropriate form.
Typical preprocessing pipelines
execute the following steps:

1. Load text as strings into memory.
1. Split the strings into tokens (e.g., words or characters).
1. Build a vocabulary dictionary to associate each vocabulary element with a numerical index.
1. Convert the text into sequences of numerical indices.

```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input #text-sequence-converting-raw-text-into-sequence-data  n=2}
%%tab mxnet
import collections
import re
from d2l import mxnet as d2l
from mxnet import np, npx
import random
npx.set_np()
```

```{.python .input #text-sequence-converting-raw-text-into-sequence-data  n=3}
%%tab pytorch
import collections
import re
from d2l import torch as d2l
import torch
import random
```

```{.python .input #text-sequence-converting-raw-text-into-sequence-data  n=4}
%%tab tensorflow
import collections
import re
from d2l import tensorflow as d2l
import tensorflow as tf
import random
```

```{.python .input #text-sequence-converting-raw-text-into-sequence-data}
%%tab jax
import collections
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import random
import re
```

## Reading the Dataset

Here, we will work with H. G. Wells'
[The Time Machine](http://www.gutenberg.org/ebooks/35),
a book containing just over 30,000 words.
While real applications will typically
involve significantly larger datasets,
this is sufficient to demonstrate
the preprocessing pipeline.
The following `_download` method
reads the raw text into a string.

```{.python .input #text-sequence-reading-the-dataset-1  n=5}
class TimeMachine(d2l.DataModule): #@save
    """The Time Machine dataset."""
    def _download(self):
        fname = d2l.download(d2l.DATA_URL + 'timemachine.txt', self.root,
                             '090b5e7e70c295757f55df93cb0a180b9691891a')
        with open(fname) as f:
            return f.read()

data = TimeMachine()
raw_text = data._download()
raw_text[:60]
```

For simplicity, we ignore punctuation and capitalization when preprocessing the raw text.

```{.python .input #text-sequence-reading-the-dataset-2  n=6}
@d2l.add_to_class(TimeMachine)  #@save
def _preprocess(self, text):
    return re.sub('[^A-Za-z]+', ' ', text).lower()

text = data._preprocess(raw_text)
text[:60]
```

## Tokenization

*Tokens* are the atomic (indivisible) units of text.
Each time step corresponds to 1 token,
but what precisely constitutes a token is a design choice.
For example, we could represent the sentence
"Baby needs a new pair of shoes"
as a sequence of 7 words,
where the set of all words comprise
a large vocabulary (typically tens
or hundreds of thousands of words).
Or we could represent the same sentence
as a much longer sequence of 30 characters,
using a much smaller vocabulary
(there are only 256 distinct ASCII characters).
Below, we tokenize our preprocessed text
into a sequence of characters.

```{.python .input #text-sequence-tokenization  n=7}
@d2l.add_to_class(TimeMachine)  #@save
def _tokenize(self, text):
    return list(text)

tokens = data._tokenize(text)
','.join(tokens[:30])
```

## Vocabulary

These tokens are still strings.
However, the inputs to our models
must ultimately consist
of numerical inputs.
Next, we introduce a class
for constructing *vocabularies*,
i.e., objects that associate
each distinct token value
with a unique index.
First, we determine the set of unique tokens in our training *corpus*.
We then assign a numerical index to each unique token.
Rare vocabulary elements are often dropped for convenience.
Whenever we encounter a token at training or test time
that had not been previously seen or was dropped from the vocabulary,
we represent it by a special "&lt;unk&gt;" token,
signifying that this is an *unknown* value.

```{.python .input #text-sequence-vocabulary-1  n=8}
class Vocab:  #@save
    """Vocabulary for text."""
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        # Flatten a 2D list if needed
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        # Count token frequencies
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # The list of unique tokens, ordered by descending frequency.
        # Reserve <unk> at index 0 so vocab[0] is the unknown token.
        self.idx_to_token = ['<unk>'] + reserved_tokens + [
            token for token, freq in self.token_freqs
            if freq >= min_freq and token not in reserved_tokens]
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

    @property
    def unk(self):  # Index for the unknown token
        return self.token_to_idx['<unk>']
```

We now construct a vocabulary for our dataset,
converting the sequence of strings
into a list of numerical indices.
Note that we have not lost any information
and can easily convert our dataset
back to its original (string) representation.

```{.python .input #text-sequence-vocabulary-2  n=9}
vocab = Vocab(tokens)
indices = vocab[tokens[:10]]
print('indices:', indices)
print('words:', vocab.to_tokens(indices))
```

## Putting It All Together

Using the above classes and methods,
we package everything into the following
`build` method of the `TimeMachine` class,
which returns `corpus`, a list of token indices, and `vocab`,
the vocabulary of *The Time Machine* corpus.
The modifications we did here are:
(i) we tokenize text into words, not characters,
to simplify the training in later sections;
(ii) `corpus` is a single list, not a list of token lists,
since each text line in *The Time Machine* dataset
is not necessarily a sentence or paragraph.

```{.python .input #text-sequence-putting-it-all-together  n=10}
@d2l.add_to_class(TimeMachine)  #@save
def build(self, raw_text, vocab=None):
    tokens = self._tokenize(self._preprocess(raw_text))
    if vocab is None: vocab = Vocab(tokens)
    corpus = [vocab[token] for token in tokens]
    return corpus, vocab

corpus, vocab = data.build(raw_text)
len(corpus), len(vocab)
```

## Exploratory Language Statistics
:label:`subsec_natural-lang-stat`

Using the real corpus and the `Vocab` class defined over words,
we can inspect basic statistics concerning word use in our corpus.
Below, we construct a vocabulary from words used in *The Time Machine*
and print the ten most frequently occurring of them.

```{.python .input #text-sequence-exploratory-language-statistics-1  n=11}
words = text.split()
vocab = Vocab(words)
vocab.token_freqs[:10]
```

Note that the ten most frequent words
are not all that descriptive.
You might even imagine that
we might see a very similar list
if we had chosen any book at random.
Articles like "the" and "a",
pronouns like "i" and "my",
and prepositions like "of", "to", and "in"
occur often because they serve common syntactic roles.
Such words that are common but not particularly descriptive
are often called *stop words* and,
in previous generations of text classifiers
based on so-called bag-of-words representations,
they were most often filtered out.
However, they carry meaning and
it is not necessary to filter them out
when working with modern RNN- and
Transformer-based neural models.
If you look further down the list,
you will notice
that word frequency decays quickly.
The $10^{\textrm{th}}$ most frequent word
is less than $1/5$ as common as the most popular.
Word frequency tends to follow a power law distribution
(specifically the Zipfian) as we go down the ranks.
To get a better idea, we plot the figure of the word frequency.

```{.python .input #text-sequence-exploratory-language-statistics-2  n=12}
freqs = [freq for token, freq in vocab.token_freqs]
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
         xscale='log', yscale='log')
```

After dealing with the first few words as exceptions,
all the remaining words roughly follow a straight line on a log--log plot.
This phenomenon is captured by *Zipf's law*,
which states that the frequency $n_i$
of the $i^\textrm{th}$ most frequent word is:

$$n_i \propto \frac{1}{i^\alpha},$$
:eqlabel:`eq_zipf_law`

which is equivalent to

$$\log n_i = -\alpha \log i + c,$$

where $\alpha$ is the exponent that characterizes
the distribution and $c$ is a constant.
This should already give us pause for thought if we want
to model words by counting statistics.
After all, we will significantly overestimate the frequency of the tail, also known as the infrequent words. But what about the other word combinations, such as two consecutive words (bigrams), three consecutive words (trigrams), and beyond?
Let's see whether the bigram frequency behaves in the same manner as the single word (unigram) frequency.

```{.python .input #text-sequence-exploratory-language-statistics-3  n=13}
bigram_tokens = ['--'.join(pair) for pair in zip(words[:-1], words[1:])]
bigram_vocab = Vocab(bigram_tokens)
bigram_vocab.token_freqs[:10]
```

One thing is notable here. Out of the ten most frequent word pairs, nine are composed of both stop words and only one is relevant to the actual book---"the time". Furthermore, let's see whether the trigram frequency behaves in the same manner.

```{.python .input #text-sequence-exploratory-language-statistics-4  n=14}
trigram_tokens = ['--'.join(triple) for triple in zip(
    words[:-2], words[1:-1], words[2:])]
trigram_vocab = Vocab(trigram_tokens)
trigram_vocab.token_freqs[:10]
```

Now, let's visualize the token frequency among these three models: unigrams, bigrams, and trigrams.

```{.python .input #text-sequence-exploratory-language-statistics-5  n=15}
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
```

This figure is quite exciting.
First, beyond unigram words, sequences of words
also appear to be following Zipf's law,
albeit with a smaller exponent
$\alpha$ in :eqref:`eq_zipf_law`,
depending on the sequence length.
Second, the number of distinct $n$-grams is not that large.
This gives us hope that there is quite a lot of structure in language.
Third, many $n$-grams occur very rarely.
This makes certain methods unsuitable for language modeling
and motivates the use of deep learning models.
We will discuss this in the next section.


## Summary

Text is among the most common forms of sequence data encountered in deep learning.
Common choices for what constitutes a token are characters, words, and word pieces.
To preprocess text, we usually (i) split text into tokens; (ii) build a vocabulary to map token strings to numerical indices; and (iii) convert text data into token indices for models to manipulate.
In practice, the frequency of words tends to follow Zipf's law. This is true not just for individual words (unigrams), but also for $n$-grams.


## Exercises

1. In the experiment of this section, tokenize text into words and vary the `min_freq` argument value of the `Vocab` instance. Qualitatively characterize how changes in `min_freq` impact the size of the resulting vocabulary.
1. Estimate the exponent of Zipfian distribution for unigrams, bigrams, and trigrams in this corpus.
1. Find some other sources of data (download a standard machine learning dataset, pick another public domain book,
   scrape a website, etc). For each, tokenize the data at both the word and character levels. How do the vocabulary sizes compare with *The Time Machine* corpus at equivalent values of `min_freq`. Estimate the exponent of the Zipfian distribution corresponding to the unigram and bigram distributions for these corpora. How do they compare with the values that you observed for *The Time Machine* corpus?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/117)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/118)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1049)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18011)
:end_tab:

<!-- slides -->

::: {.slide title="Text as Sequence Data"}
Text isn't tensors out of the box. The pipeline:

1. **Read** the raw string.
2. **Tokenize** — split into characters, words, or subwords.
3. **Build a vocabulary** — map each token to an integer index.
4. **Index** the corpus → a sequence of ints.

Running example: H. G. Wells's *The Time Machine* (32 k tokens) —
small enough to index in a notebook, big enough to train a
language model on. The other half of the chapter looks at the
**statistics** of natural-language text: long-tail word
distributions, stop words, bigrams.
:::

::: {.slide title="Read"}
@text-sequence-converting-raw-text-into-sequence-data

. . .

@text-sequence-reading-the-dataset-1

. . .

@text-sequence-reading-the-dataset-2
:::

::: {.slide title="Tokenize"}
Word-level splits on whitespace; character-level keeps
individual characters:

@text-sequence-tokenization
:::

::: {.slide title="Vocabulary"}
A `Vocab` class maps tokens ↔ integer indices, with reserved
slots for `<unk>` (rare/OOV tokens) and a few specials:

@text-sequence-vocabulary-1
:::

::: {.slide title="Building the vocab"}
Pass the tokenized corpus and (optionally) a min-frequency
threshold to drop very rare tokens:

@text-sequence-vocabulary-2
:::

::: {.slide title="One-stop dataloading"}
Wrap the whole pipeline in a `TimeMachine.build()` so models
just call `data.build(...)` to get tensors:

@text-sequence-putting-it-all-together
:::

::: {.slide title="Word-frequency statistics"}
Tokenize words, count occurrences, sort by decreasing count.

- The head of the distribution is mostly **function words**:
  "the", "of", "and", "to", "a", ...
- These words are common because they carry grammatical structure.
- In old bag-of-words classifiers they were often removed as
  *stop words*; neural sequence models usually keep them.
:::

::: {.slide title="Zipf law"}
After the first few words, frequency is close to a straight line
on log-log axes:

$$n_i \propto \frac{1}{i^\alpha}, \qquad
\log n_i = -\alpha \log i + c.$$

Interpretation:

- A few tokens appear extremely often.
- Most tokens are rare.
- Count tables waste probability mass in the tail.
:::

::: {.slide title="Word-frequency plot"}
@!text-sequence-exploratory-language-statistics-2
:::

::: {.slide title="Bigrams"}
Bigrams count consecutive word pairs:

$$ (w_t, w_{t+1}). $$

The most common bigrams are still dominated by function-word
phrases. One exception in this corpus is semantically meaningful:
`the--time`.

Pedagogical point: increasing context length makes the counts more
specific, but also much sparser.
:::

::: {.slide title="Trigrams"}
Trigrams count consecutive triples:

$$ (w_t, w_{t+1}, w_{t+2}). $$

The vocabulary grows quickly with $n$, while the corpus size is fixed.
Most possible triples are never observed.

This is the classic n-gram tradeoff:

- larger $n$ captures more context;
- larger $n$ creates a much longer tail.
:::

::: {.slide title="Zipf at every n"}
Plot unigram, bigram, and trigram frequencies on log-log axes —
all three follow Zipf-like power laws, with steeper slopes (and
sparser high-frequency regimes) for higher n:

@!text-sequence-exploratory-language-statistics-5

This **long-tail sparsity** is exactly why neural language
models — which embed each token into a continuous space — work so
much better than n-gram count tables.
:::

::: {.slide title="Recap"}
- Pipeline: **read → tokenize → build vocab → index** = corpus
  as `LongTensor`.
- Word vs character tokenization is a tradeoff between
  vocabulary size and sequence length; subword (BPE) is the
  modern default.
- Natural-language frequencies are **Zipfian** at every n —
  long-tail sparsity makes neural models a much better fit than
  count-based ones.
:::
