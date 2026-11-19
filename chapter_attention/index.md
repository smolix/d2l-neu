# Attention
:label:`chap_attention`

The sequence models of :numref:`chap_rnn` read a text one token at a time
and carry what they have read in a state of fixed size. Think of a reader
who reaches the last chapter of a mystery novel and needs the name of a
minor character from the first page. A person turns back and looks. A
recurrent network must have kept the name in its state all along, written
when nothing marked it as important. Can a network look back
instead, and read only what the current prediction needs?

Attention does this. It keeps a representation of every position, scores
the current query against a key at every position, turns the scores into
weights, and returns the weighted average of the values stored there. The
*query* says what a position looks for, a *key* what it offers, and a
*value* what it hands back. A weighted average, unlike a hard choice, is
differentiable, so gradient descent can learn what to look for. A layer
runs several such lookups side by side, called *heads*. The idea became
known through machine translation, where it let a network consult every
source word instead of one vector summarizing the sentence
:cite:`Bahdanau.Cho.Bengio.2014`. The Transformer then made attention the
only operation that mixes positions :cite:`Vaswani.Shazeer.Parmar.ea.2017`.

Reading every position raises problems that recurrent networks never had.
A weighted average ignores the order of its inputs. Scoring all pairs of
positions costs time and memory that grow with the square of the length.
And a map of attention weights shows where a head looks, not what it
computes.

The first three sections build the layer: a lookup in a table of two
records, a learned score with masks, and a sentence that queries itself.
This *self-attention* (:numref:`sec_multihead-attention`) is the
layer a transformer stacks. The three problems are not equally settled.
Order has a standard answer. Cost still shapes new architectures, and two
sections take it on: exact attention without stored scores, then cheaper
variants that read only recent keys or, as *linear attention*, return to a
state of fixed size, the very thing attention set out to avoid. What a
trained head computes is understood in detail only in small models, and the
last section takes one apart.

A reader heading for :numref:`chap_transformers` needs the first four
sections and the opening of the last, on the *residual stream* that every
layer writes into. The sections on cost can wait until
:numref:`sec_kv-cache` and :numref:`chap_modern_rnn` need them.

```toc
:maxdepth: 2

queries-keys-values
attention-scoring
multihead-attention
positional-information
attention-at-scale
windowed-linear-attention
what-attention-computes
further-reading
```

## Resources and Further Reading {.unnumbered}

If you read two of these, take the 3Blue1Brown lesson for the picture of a
query meeting its keys, and Elhage et al. alongside
:numref:`sec_what-attention-computes`. Each entry names the section it goes
with, and all are freely accessible online. Related topics that the chapter
does not teach are collected in :numref:`sec_attention-further-reading`.

**Visual introductions**

- [Attention in Transformers, Step-by-Step](https://www.3blue1brown.com/lessons/attention), 3Blue1Brown (2024): a geometric introduction to queries, keys, values, and attention weights, the lookup of :numref:`sec_queries-keys-values` drawn as vectors, with queries and keys in a small space of their own and values in the embedding space.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/), Jay Alammar (2018): follows a two-word input through self-attention one step at a time (query, key, and value vectors, scores, scaling, softmax, weighted sum), with a picture for each step; read it with :numref:`sec_attention-scoring-functions` and :numref:`sec_multihead-attention`.
- [Transformers from Scratch](https://peterbloem.nl/blog/transformers), Peter Bloem (2019): builds self-attention from a plain weighted sum before adding learned projections, and treats its input as a set, the permutation equivariance that :numref:`sec_positional-information` proves and then breaks.

**Foundational papers**

- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473), Bahdanau et al. (2014): the learned alignment that started the field, whose additive score :numref:`sec_attention-scoring-functions` compares with the dot product.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Vaswani et al. (2017): scaled dot-product attention, multi-head attention, and sinusoidal positions, the starting point of :numref:`sec_attention-scoring-functions` through :numref:`sec_positional-information`.

**Position**

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864), Su et al. (2021): rotary embeddings, derived as in :numref:`sec_positional-information` from the requirement that scores depend only on offsets.
- [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409), Press et al. (2022): ALiBi and the length-extrapolation experiment that :numref:`sec_positional-information` repeats on a small model.
- [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071), Peng et al. (2024): reviews position interpolation and the NTK-aware change of base that preceded it, and adds a separate treatment of fast and slow pairs and a softmax temperature, the context extension of :numref:`sec_positional-information`.

**The cost of attention**

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135), Dao et al. (2022): the tiled online softmax of :numref:`sec_attention-at-scale`, with the memory-traffic analysis that explains its speed.
- [Fast Transformer Decoding: One Write-Head Is All You Need](https://arxiv.org/abs/1911.02150), Shazeer (2019): multi-query attention, in which all heads read one shared set of keys and values. :numref:`sec_multihead-attention` shows when such sharing loses nothing, and :numref:`sec_kv-cache-reduction` why it is done and what it costs.
- [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236), Katharopoulos et al. (2020): linear attention and its recurrent form, as in :numref:`sec_windowed-linear-attention`.
- [The Transformer Family Version 2.0](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/), Lilian Weng (2023): a survey of attention variants, sparse patterns, and efficient approximations, for the windows and kernels of :numref:`sec_windowed-linear-attention` and the many relatives the section leaves out.

**What attention computes**

- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html), Elhage et al. (2021): the residual stream and the QK and OV circuits that :numref:`sec_what-attention-computes` reads from trained weights.
- [In-Context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html), Olsson et al. (2022): induction heads in small and large models, the circuit that :numref:`sec_what-attention-computes` trains and dissects.
- [Thinking Like Transformers](https://arxiv.org/abs/2106.06981), Weiss, Goldberg, and Yahav (2021): RASP, a small programming language whose operations are those of a transformer, used to write down what a fixed number of attention layers can compute; a companion to the expressivity arguments of :numref:`sec_what-attention-computes`.
- [Transformer Puzzles](https://github.com/srush/Transformer-Puzzles), Sasha Rush: short coding puzzles in Python on the internals of the transformer, based on the RASP paper above; a way to practice thinking in attention layers.

**Courses**

- [Stanford CS224n: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/): lectures and an assignment on self-attention, positions, and transformer components, the material of :numref:`sec_attention-scoring-functions` through :numref:`sec_positional-information`.

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §10]{.kicker}

Reading every position<br>
**soft lookup · scoring and masks · self-attention and heads · positions · cost · windows and linear attention · circuits**
:::
:::

::: {.slide title="A Fixed-Size State Must Guess What Will Be Asked"}
[The problem]{.kicker}

- A recurrent model as wide as GPT-2 small (768) that has read 1,024
  tokens keeps 768 numbers about all of them.
- It writes them before anyone knows which token a later prediction
  needs.
- A reader who forgot a name on page one turns back and looks.

::: {.d2l-note}
Attention lets a network look back: keep one vector per position and read
only what the current prediction needs.
:::
:::

::: {.slide title="Attention Scores Every Key and Averages the Values"}
[The idea]{.kicker}

![A query scored against three keys; the weights sum to one, and the output is the weighted average of the values.](../img/mdl-attention-soft-lookup.svg){width=70%}

- **query**: what a position looks for; **key**: what a position offers;
  **value**: what it hands back.
- A weighted average is differentiable, so the network learns what to
  look for.
- One such lookup is a **head**; a layer runs several side by side.
:::

::: {.slide title="Looking Back Creates Three Problems a Recurrent Network Never Had"}
[The problem]{.kicker}

| problem | why |
|---|---|
| **order** | a weighted average is the same for any order of its inputs |
| **cost** | all pairs of positions: time and memory grow as $n^2$ for $n$ tokens |
| **meaning** | a map of weights shows where a head looks, not what it computes |

**Check:** how many scores does one head compute over 1,024 tokens?

::: {.fragment .fade-up}
::: {.d2l-note}
One per pair of positions: $1{,}024^2 \approx 10^6$, about half of them
hidden when each token may look only backward. Doubling the context
quadruples the count.
:::
:::
:::

::: {.slide title="By the End of This Chapter You Can"}
[Goals]{.kicker}

- compute scaled dot-product attention by hand, with masks for padding and
  for the future;
- run one head of self-attention on a sentence, and prove what one set of
  weights per query cannot express;
- encode position by rotating queries and keys (RoPE) or by a penalty on
  distance (ALiBi), and extend a trained context;
- compute exact attention without storing its scores, and say what a
  window of recent keys and linear attention (a state of fixed size) give
  up;
- read from the weights where a head looks (QK) and what it moves (OV),
  and test the reading by removing heads.

**Running example:** one head of GPT-2 small, width 64 in a model of width
768, reading 1,024 tokens.
:::

::: {.slide title="Four Sections Lead to the Transformer, Two on Cost Can Wait"}
[Scope and place]{.kicker}

- **Assumes:** softmax, matrix products, the recurrent language model.
- **Core:** the first four sections, and the opening of the last: how
  every layer adds its output to one running vector per position, the
  residual stream.
- **Can wait:** the two sections on cost, until the transformer chapter's
  cache of keys and values or the state space model chapter builds on
  them.
- **Leaves out:** feed-forward layers, normalization, and the training
  recipe, which the next chapter adds to make a transformer.
:::

::: {.slide title="Build the Layer, Then Solve What It Breaks"}
[Chapter map]{.kicker}

| | section | question |
|---|---|---|
| **build** | Queries, Keys, and Values | how can a lookup be made differentiable? |
| | Attention Scoring and Masking | which score can a network learn, and which pairs may count? |
| | Self-Attention and Multiple Heads | what does a sentence reading itself compute, and what can one head not express? |
| **order** | Positional Information | how does an order-blind layer learn where a token stands? |
| **cost** | The Cost of Attention | what does exact attention cost, and how much of it is avoidable? |
| | Windowed and Linear Attention | what is given up for a cost linear in the length? |
| **meaning** | Attention Circuits | what do trained heads compute, and how can we tell? |
:::

::: {.slide title="Every Section Tests Its Claims on an Example Small Enough to Run"}
[Running examples]{.kicker}

| section | example |
|---|---|
| Queries, Keys, and Values | a table of two records; kernel regression on noisy 1-D data |
| Attention Scoring and Masking | two queries and three keys by hand; the letters of two words; a learned permutation |
| Self-Attention and Multiple Heads | "Alice gave Bob the book"; a task that needs two values per query |
| Positional Information | `TinyCharLM` on WikiText-103 bytes, five position schemes |
| The Cost of Attention | one GPT-2-small head, up to 16,384 tokens |
| Windowed and Linear Attention | GPT-2-sized heads with a window of 128 keys, up to 32,768 tokens; 1,000 queries against 8 keys |
| Attention Circuits | `TinyCharLM` on repeated random patterns |

::: {.d2l-note}
The first five sections also count their costs on the GPT-2 head.
:::
:::
