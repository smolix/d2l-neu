# Sequence Models
:label:`chap_rnn`

Many learning problems involve ordered observations: words in a sentence,
notes in a melody, or measurements in a time series. Earlier chapters mapped a
fixed-length input to an output and usually treated examples as independent.
Sequence models must instead represent dependence across positions and inputs
whose lengths may vary.

Earlier objectives usually treated examples as exchangeable, fixed-shape
records. Sequence tasks retain the order within each example: conditioning on
earlier words, notes, or measurements can change the distribution of later
ones. Sequence lengths may also vary, even though implementations often pad or
batch them into equal-sized arrays.

The chapter develops two main ideas. The first is *autoregressive
factorization*.
Rather than model the probability of a whole sequence at once,
we factor it into a product of one-step-ahead predictions:
the probability of each element given the elements that precede it.
This turns an unwieldy generative problem into an ordinary supervised one,
in which the input is a prefix and the label is the next element.
Every position in a sequence thus becomes another training signal.
Generation repeats this prediction while feeding each generated output back
as the next input.

The second idea is the *hidden state*.
A prefix grows with time, but a recurrent model carries a fixed-size state
$h_t=f(x_t,h_{t-1})$. The state is a function of the prefix, and training
determines which predictive information it retains. Its fixed size makes
incremental inference inexpensive but may discard information from the distant
past.

Language modeling provides the running application. We turn text into tokens,
factor sequence probabilities into next-token conditionals, and replace a
fixed context window with a recurrent state. The implementation then exposes
two remaining problems: gradients must propagate through repeated state
updates, and a probability distribution still needs a decoding rule to produce
text.

Recurrent networks powered the deep-learning breakthroughs of the 2010s
in speech recognition and machine translation,
and for a while they were the default model for anything sequential.
Transformers later displaced them at scale, and much of today's attention goes there.
Yet recurrence has returned in modern guise, which we take up in :numref:`chap_modern_rnn`,
precisely because a bounded-memory state makes inference cheap
when the alternative grows with the length of the sequence.
Whichever architecture wins a given task,
the later material on large language models uses the same concepts introduced
here: autoregressive factorization, perplexity,
backpropagation through time, and decoding.
They are worth learning once, and learning carefully.

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

The references below retrace this chapter's arc — text into tokens, a counting
baseline, the recurrence that replaces it, the gradients that make training
hard, and the decoding that turns predictions back into text. Because
tokenization, perplexity, and sampling are practiced today essentially as
these sources describe, the list runs unusually far back and unusually far
forward: Shannon's 1951 guessing game and a 2025 sampling rule sit here
comfortably side by side. All are freely accessible online except where noted.

**Books**

- [Speech and Language Processing, 3rd ed. — Jurafsky & Martin](https://web.stanford.edu/~jurafsky/slp3/) — free draft (complete PDF, updated January 2026); Chapter 2 (words and tokens) and Chapter 3 ($n$-gram language models) are the standard treatment of :numref:`sec_text-sequence` and :numref:`sec_language-model`, and develop the Kneser–Ney smoothing our sparsity discussion names but does not derive; Chapter 13 covers RNNs and LSTMs.
- [Deep Learning — Goodfellow, Bengio & Courville](https://www.deeplearningbook.org/) — free HTML; Chapter 10 (Sequence Modeling) develops recurrent networks, teacher forcing, and backpropagation through time with fuller gradient derivations than :numref:`sec_rnn` and :numref:`sec_bptt` carry, and previews the gated architectures of the next chapter.
- [Natural Language Processing — Jacob Eisenstein](https://github.com/jacobeisenstein/gt-nlp-class/blob/master/notes/eisenstein-nlp-notes.pdf) — free PDF of the notes that became *Introduction to Natural Language Processing* (MIT Press, 2019); Chapter 6 treats $n$-gram and RNN language models as one continuous story, exactly the handoff this chapter makes between :numref:`sec_language-model` and :numref:`sec_rnn`.

**Courses and video lectures**

- [Stanford CS224N: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/) — free slides and notes, with complete lecture videos on YouTube; its language-modeling and RNN lectures cover the span from :numref:`sec_language-model` to :numref:`sec_bptt`, vanishing-gradient argument included, before heading toward transformers.
- [Neural Networks: Zero to Hero — Andrej Karpathy](https://karpathy.ai/zero-to-hero.html) — free video series; the *makemore* lectures build character-level language models from a counting bigram model up to neural networks — the arc from :numref:`sec_language-model` to :numref:`sec_rnn-scratch`, compressed and typed live — and "Let's build the GPT Tokenizer" does the same for the byte-level BPE of :numref:`sec_text-sequence`.
- [MIT 6.S191: Introduction to Deep Learning — Amini & Amini](https://introtodeeplearning.com/) — free, updated annually; the "Deep Sequence Modeling" lecture is a fast, visual pass over sequence tasks, RNNs, and their gradient pathologies (:numref:`sec_rnn`, :numref:`sec_bptt`).

**Foundational papers**

- [Prediction and Entropy of Printed English — Claude E. Shannon (1951), *Bell System Technical Journal*](https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf) — free scan; frames language as a next-symbol guessing game and measures the entropy of English with human predictors — the direct ancestor of the perplexity and bits-per-byte of :numref:`sec_language-model`, and still a delight to read.
- [Finding Structure in Time — Jeffrey Elman (1990), *Cognitive Science*](https://doi.org/10.1207/s15516709cog1402_1) — the network of :numref:`sec_rnn` in its original habitat: a hidden state fed back on itself discovers word boundaries and grammatical structure from raw sequences (paywalled, noted; widely reproduced online).
- [A Neural Probabilistic Language Model — Bengio, Ducharme, Vincent & Jauvin (2003), *JMLR*](https://jmlr.org/papers/v3/bengio03a.html) — free; replaces the sparse count tables of :numref:`sec_language-model` with learned embeddings and a neural network, the embedding-plus-softmax design that :numref:`sec_rnn-scratch` inherits.
- [Recurrent Neural Network Based Language Model — Mikolov et al. (2010), *Interspeech*](https://www.isca-archive.org/interspeech_2010/mikolov10_interspeech.html) — free; the demonstration that RNN language models beat smoothed $n$-grams in practice — the model of :numref:`sec_rnn-scratch`, evaluated the way :numref:`sec_language-model` teaches.
- [On the Difficulty of Training Recurrent Neural Networks — Pascanu, Mikolov & Bengio (2013), *ICML*](https://proceedings.mlr.press/v28/pascanu13.html) — free; the modern account of vanishing and exploding gradients and the source of the gradient-norm clipping used in :numref:`sec_rnn-scratch`; the difficulty itself was first formalized by [Bengio, Simard & Frasconi (1994)](https://doi.org/10.1109/72.279181) (paywalled, noted).
- [Neural Machine Translation of Rare Words with Subword Units — Sennrich, Haddow & Birch (2016), *ACL*](https://aclanthology.org/P16-1162/) — free; the paper that brought byte pair encoding into NLP — the merge algorithm implemented from scratch in :numref:`sec_text-sequence`.
- [Language Models are Unsupervised Multitask Learners — Radford et al. (2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — free; the GPT-2 report, source of both the byte-level BPE with regex pre-tokenization that :numref:`sec_text-sequence` rebuilds and checks token-for-token against `tiktoken`, and the top-$k$ truncation of :numref:`sec_decoding`.
- [The Curious Case of Neural Text Degeneration — Holtzman et al. (2020), *ICLR*](https://arxiv.org/abs/1904.09751) — free; diagnoses why maximization loops and why the unreliable tail spoils pure sampling, then proposes nucleus (top-$p$) sampling — the argument on which :numref:`sec_decoding` is built.
- [Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs — Nguyen et al. (2025), *ICLR*](https://arxiv.org/abs/2407.01082) — free; the confidence-scaled truncation rule in the unified sampler of :numref:`sec_decoding`, with evidence that it degrades most gracefully at the high temperatures where creative sampling wants to run.

**Tutorials, notes, and interactive**

- [The Unreasonable Effectiveness of Recurrent Neural Networks — Andrej Karpathy](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) — free; the 2015 essay that showed character-level RNN language models writing Shakespeare, Wikipedia markup, LaTeX, and C code, and visualized single hidden units tracking quotes and line lengths — the best advertisement for what :numref:`sec_rnn-scratch` trains, sampled the way :numref:`sec_decoding` explains.
- [minbpe — Andrej Karpathy](https://github.com/karpathy/minbpe) — free; a minimal, thoroughly commented byte-level BPE codebase (the companion to the tokenizer video above) that reaches parity with GPT-4's `tiktoken` tokenizer — the natural second implementation to compare against the one built in :numref:`sec_text-sequence`.
- [Tiktokenizer](https://tiktokenizer.vercel.app/) — free, zero-install; type text and watch production tokenizers segment it live — an instant way to rerun the fertility, digit-splitting, and whitespace experiments of :numref:`sec_text-sequence` on GPT-2 versus current vocabularies.
- [How to Generate Text — Patrick von Platen (Hugging Face)](https://huggingface.co/blog/how-to-generate) — free; greedy search, beam search, top-$k$, and top-$p$ with runnable `transformers` code — the experiments of :numref:`sec_decoding` repeated on models large enough for each failure mode to be vivid.
- [Hugging Face LLM Course, Chapter 6: Tokenizers](https://huggingface.co/learn/llm-course/en/chapter6/1) — free; train and dissect production BPE, WordPiece, and Unigram tokenizers with the `tokenizers` library — the industrial-strength counterpart to the tokenizers-in-the-wild tour closing :numref:`sec_text-sequence`.
