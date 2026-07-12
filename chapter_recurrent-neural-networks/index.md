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
