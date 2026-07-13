# Sequence Models and Language Models
:label:`chap_rnn`

A great deal of intelligent behavior looks like predicting what comes next.
We finish each other's sentences, anticipate the next note in a melody,
and extrapolate a trend from the numbers seen so far.
Each of these tasks takes a *sequence* of observations and asks for its continuation.
The parts of this book that came before treated learning as a mapping
from a fixed-length input to a single output, one example at a time.
That framing served us well for tabular data and for images,
but it quietly assumed away two features that define sequential data,
and it is exactly those two features that this chapter is built to handle.

The first assumption was that examples are independent and identically distributed.
When we fit linear and logistic regression in :numref:`chap_regression`
and :numref:`chap_classification`, or multilayer perceptrons in :numref:`chap_perceptrons`,
we drew each example without regard to the others.
In a sentence or a time series the opposite is true:
every element depends on the ones before it,
and that dependence is precisely the signal we want to model.
The second assumption was that each input has a fixed shape.
Even the images of :numref:`chap_cnn` arrived as a fixed grid of pixels.
A document, a recording, or a price history has no such fixed length,
and two examples rarely share one.

Two ideas carry us through the whole chapter.
The first is *autoregressive factorization*.
Rather than model the probability of a whole sequence at once,
we factor it into a product of one-step-ahead predictions:
the probability of each element given the elements that precede it.
This turns an unwieldy generative problem into an ordinary supervised one,
in which the input is a prefix and the label is the next element.
Every position in a sequence thus becomes another training signal,
and generation becomes nothing more than repeating that prediction
and feeding each output back in as the next input.

The second idea is the *hidden state*.
A prefix grows without bound as a sequence unrolls,
yet we cannot afford a memory that grows along with it.
So we insist that the model carry a fixed-size summary
of everything it has read so far, revising that summary as each new element arrives.
This is what a recurrent neural network does,
and the tension at its heart runs through the rest of the part:
how much of an unbounded past can a bounded state honestly remember?
Getting a useful answer to that question is what makes the difference
between a model that forgets within a few steps and one that holds a thought.

Language modeling is the running application that ties these ideas together,
and by the end we will have trained a small language model
and sampled fresh text from it properly.
The path there is a single build, one section handing off to the next:
we tokenize a corpus, set a baseline to beat, learn a recurrence,
face down the gradients that make it hard to train, and finally sample from it.
We begin with sequences in the abstract in :numref:`sec_sequence`,
then turn text into tokens a model can consume in :numref:`sec_text-sequence`.
:numref:`sec_language-model` frames the language-modeling task
and fits a simple counting baseline to beat.
:numref:`sec_rnn` introduces the recurrent network itself,
and :numref:`sec_rnn-scratch` implements an RNN language model end to end.
:numref:`sec_bptt` confronts the gradient realities of training through time,
and :numref:`sec_decoding` closes the loop
by turning a trained model's predictions back into readable text.

A word on where this material stands.
Recurrent networks powered the deep-learning breakthroughs of the 2010s
in speech recognition and machine translation,
and for a while they were the default model for anything sequential.
Transformers later displaced them at scale, and much of today's attention goes there.
Yet recurrence has returned in modern guise, which we take up in :numref:`chap_modern_rnn`,
precisely because a bounded-memory state makes inference cheap
when the alternative grows with the length of the sequence.
Whichever architecture wins a given task,
the concepts introduced here, namely autoregressive factorization, perplexity,
backpropagation through time, and decoding,
are exactly the ones on which the later material on large language models stands.
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
- [A Neural Probabilistic Language Model — Bengio, Ducharme, Vincent & Jauvin (2003), *JMLR*](https://jmlr.org/papers/v3/bengio03a.html) — free; the answer to the sparsity wall of :numref:`sec_language-model`: replace counting with learned embeddings and a neural network, the embedding-plus-softmax design that :numref:`sec_rnn-scratch` inherits.
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
