# Sequence Models
:label:`chap_rnn`

Every model of the earlier chapters reads an input of fixed shape, such as
a $28 \times 28$ image of a piece of clothing, and treats each example on
its own. A sentence, a recording, or a series of measurements is different:
it varies in length, and its entries depend on each other and on their
order. Take the opening sentence of H. G. Wells' *The Time Machine*, which
the models of this chapter read: "The Time Traveller (for so it will be
convenient to speak of him) was expounding a recondite matter to us." To
guess its last word, a reader uses the nineteen words before it, in order.
How can a model do the same when the text before it can have any length?

Claude Shannon made a game of this. In 1951 he measured how predictable
English is by asking people to guess a text one letter at a time. A
*language model* plays the same game: it predicts each token, which may be
a character or a word, from the tokens before it. A model that can predict
can also write, by feeding each token it chooses back in as its next input.

The difficulty is the text already read, which grows with every token. A
window of the last few tokens forgets everything older. A *recurrent neural
network* instead summarizes all of it in a *hidden state* of fixed size,
which the same layer updates after every token. Today's large language
models also predict one token at a time, so what this chapter teaches about
cutting text into tokens, scoring a model, and turning its predictions into
text carries over to them.

The chapter sets up the problem before any recurrent network appears. We
predict a sequence from its own past and see why forecasting many steps ahead
is harder than forecasting one (:numref:`sec_sequence`), cut text into tokens
(:numref:`sec_text-sequence`), and build language models that only count,
which also gives us the measure by which language models are compared
(:numref:`sec_language-model`). The counts run out, and the recurrent network
is our answer: :numref:`sec_rnn` replaces the window by a hidden state,
:numref:`sec_rnn-scratch` implements and trains the model and compares it
with the counts, and :numref:`sec_bptt` explains why gradients through the
recurrence vanish or explode. The last section, :numref:`sec_decoding`, turns
next-token distributions into text.

Gated cells and state space models, which let a recurrent network remember
further back, follow in :numref:`chap_modern_rnn`, and attention, which
keeps every past token, in :numref:`chap_attention`.

```toc
:maxdepth: 2

sequence
text-sequence
language-model
rnn
rnn-implementation
bptt
decoding
```

## Resources and Further Reading {.unnumbered}

The resources below accompany the sections of this chapter; each names the
section it goes with. All are freely accessible online except where noted.

**Books**

- [Speech and Language Processing, 3rd ed. — Jurafsky & Martin](https://web.stanford.edu/~jurafsky/slp3/) — free draft (complete PDF, updated January 2026); Chapter 2 (words and tokens) and Chapter 3 ($n$-gram language models) are the standard treatment of :numref:`sec_text-sequence` and :numref:`sec_language-model`, and treat at length the interpolated and discounted smoothing schemes, of which :numref:`sec_language-model` derives Kneser–Ney; Chapter 13 covers RNNs and LSTMs.
- [Deep Learning — Goodfellow, Bengio & Courville](https://www.deeplearningbook.org/) — free HTML; Chapter 10 (Sequence Modeling) develops recurrent networks, teacher forcing, and backpropagation through time with fuller gradient derivations than :numref:`sec_rnn` and :numref:`sec_bptt`, and introduces the gated architectures of :numref:`chap_modern_rnn`.
- [Natural Language Processing — Jacob Eisenstein](https://github.com/jacobeisenstein/gt-nlp-class/blob/master/notes/eisenstein-nlp-notes.pdf) — free PDF of the notes that became *Introduction to Natural Language Processing* (MIT Press, 2019); Chapter 6 treats $n$-gram and RNN language models as one story, the step this chapter takes from :numref:`sec_language-model` to :numref:`sec_rnn`.

**Courses and video lectures**

- [Stanford CS224N: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/) — free slides and notes, with complete lecture videos on YouTube; its language-modeling and RNN lectures cover the material from :numref:`sec_language-model` to :numref:`sec_bptt`, the vanishing-gradient argument included, before turning to transformers.
- [Neural Networks: Zero to Hero — Andrej Karpathy](https://karpathy.ai/zero-to-hero.html) — free video series; the *makemore* lectures build character-level language models from a counting bigram model up to neural networks, the path from :numref:`sec_language-model` to :numref:`sec_rnn-scratch` typed live, and "Let's build the GPT Tokenizer" builds the byte-level BPE of :numref:`sec_text-sequence`.
- [MIT 6.S191: Introduction to Deep Learning — Amini & Amini](https://introtodeeplearning.com/) — free, updated annually; the "Deep Sequence Modeling" lecture is a fast, visual pass over sequence tasks, RNNs, and their gradient pathologies (:numref:`sec_rnn`, :numref:`sec_bptt`).

**Foundational papers**

- [Prediction and Entropy of Printed English — Claude E. Shannon (1951), *Bell System Technical Journal*](https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf) — free scan; frames language as a game of guessing the next symbol and measures the entropy of English with human predictors, the origin of the perplexity and bits per byte of :numref:`sec_language-model`.
- [Finding Structure in Time — Jeffrey Elman (1990), *Cognitive Science*](https://doi.org/10.1207/s15516709cog1402_1) — paywalled, widely reproduced online; introduces the network of :numref:`sec_rnn`, whose hidden state, fed back on itself, discovers word boundaries and grammatical categories from raw sequences.
- [A Neural Probabilistic Language Model — Bengio, Ducharme, Vincent & Jauvin (2003), *JMLR*](https://jmlr.org/papers/v3/bengio03a.html) — free; replaces the sparse count tables of :numref:`sec_language-model` by learned embeddings and a neural network, the neural $n$-gram of :numref:`sec_rnn` and the embedding and softmax layers of :numref:`sec_rnn-scratch`.
- [Recurrent Neural Network Based Language Model — Mikolov et al. (2010), *Interspeech*](https://www.isca-archive.org/interspeech_2010/mikolov10_interspeech.html) — free; shows that RNN language models beat smoothed $n$-grams in practice, with the model of :numref:`sec_rnn-scratch` evaluated by perplexity as in :numref:`sec_language-model`.
- [On the Difficulty of Training Recurrent Neural Networks — Pascanu, Mikolov & Bengio (2013), *ICML*](https://proceedings.mlr.press/v28/pascanu13.html) — free; the analysis of vanishing and exploding gradients in :numref:`sec_bptt` and the source of the gradient-norm clipping of :numref:`sec_rnn-scratch`. The difficulty itself was first formalized by [Bengio, Simard & Frasconi (1994)](https://doi.org/10.1109/72.279181) (paywalled).
- [Neural Machine Translation of Rare Words with Subword Units — Sennrich, Haddow & Birch (2016), *ACL*](https://aclanthology.org/P16-1162/) — free; brought byte pair encoding into NLP, the merge algorithm implemented from scratch in :numref:`sec_text-sequence`.
- [Language Models are Unsupervised Multitask Learners — Radford et al. (2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — free; the GPT-2 report, source of the byte-level BPE with regex pre-tokenization that :numref:`sec_text-sequence` rebuilds and checks token for token against `tiktoken`, and of the top-$k$ sampling of :numref:`sec_decoding`.
- [The Curious Case of Neural Text Degeneration — Holtzman et al. (2020), *ICLR*](https://arxiv.org/abs/1904.09751) — free; explains why maximization loops and why the unreliable tail spoils pure sampling, and proposes nucleus (top-$p$) sampling, the argument :numref:`sec_decoding` follows.
- [Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs — Nguyen et al. (2025), *ICLR*](https://arxiv.org/abs/2407.01082) — free; the min-$p$ rule in the sampler of :numref:`sec_decoding`, which keeps tokens in proportion to the model's confidence, with evidence that it degrades less than top-$p$ at high temperatures.

**Tutorials, notes, and interactive**

- [The Unreasonable Effectiveness of Recurrent Neural Networks — Andrej Karpathy](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) — free; the 2015 essay in which character-level RNN language models write Shakespeare, Wikipedia markup, LaTeX, and C code, and single hidden units track quotes and line lengths; a larger version of what :numref:`sec_rnn-scratch` trains, sampled as in :numref:`sec_decoding`.
- [minbpe — Andrej Karpathy](https://github.com/karpathy/minbpe) — free; a minimal, commented byte-level BPE codebase (the companion to the tokenizer video above) that reproduces GPT-4's `tiktoken` tokenizer, a second implementation to compare with the one of :numref:`sec_text-sequence`.
- [Tiktokenizer](https://tiktokenizer.vercel.app/) — free, nothing to install; shows how production tokenizers segment text as it is typed, a quick way to repeat the fertility and digit experiments of :numref:`subsec_fertility-digits-glitch` on GPT-2 and current vocabularies.
- [How to Generate Text — Patrick von Platen (Hugging Face)](https://huggingface.co/blog/how-to-generate) — free; greedy search, beam search, top-$k$, and top-$p$ with runnable `transformers` code, the experiments of :numref:`sec_decoding` repeated on models large enough to show each failure clearly.
- [Hugging Face LLM Course, Chapter 6: Tokenizers](https://huggingface.co/learn/llm-course/en/chapter6/1) — free; trains and dissects production BPE, WordPiece, and Unigram tokenizers with the `tokenizers` library, a companion to the production tokenizers examined at the end of :numref:`sec_text-sequence`.

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
- explain why gradients through time vanish or explode, and what
  truncation removes;
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
| Language Models | 87% of held-out word trigrams are unseen; Kneser–Ney lowers the word trigram's perplexity from 2,618 to 302 |
| Recurrent Neural Networks | 222,915 parameters, 89% of them facing the vocabulary; 512 bytes of state per sequence |
| Implementing RNNs | about 2.1 bits per byte with a wider embedding and dropout, near the Kneser–Ney trigram; their average beats both |
| Backpropagation Through Time | 40 steps: $0.9^{40} \approx 0.015$, $1.1^{40} \approx 45$ |
| Decoding and Generation | more than $10^{150}$ continuations of 50 tokens; min-$p$ samples reach distinct-3 near 1 |
:::
