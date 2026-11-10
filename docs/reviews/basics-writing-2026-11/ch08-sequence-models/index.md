# Sequence Models (chapter introduction)

Source: `chapter_recurrent-neural-networks/index.md`. About 400 words of introduction, followed by a "Resources and Further Reading" list of about 870 words.

## Verdict

A good chapter opening, close to the house template and better than most. It starts from what the reader's models cannot do, picks one concrete sentence from the running text, and brings in Shannon's guessing game as the origin of the idea. The one thing to fix is the second paragraph, which packs the whole chapter (language model, chain rule, generation by feedback, the window, the count table, the hidden state, large language models) into nine sentences and ends on three terms the reader has not met.

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|---|---|---|---|---|
| 4 | 4 | 4 | 4 | 4 |

## What works

- **The first paragraph.** "To guess its last word, a reader uses the nineteen words before it, in order." The count is right (the sentence has twenty words), the example is the corpus the chapter trains on, and the paragraph ends on a question the reader can hold: "How can a model do the same when the text before it can have any length?"
- **Shannon in one sentence.** "In 1951 Claude Shannon measured how predictable English is by asking people to guess a text one letter at a time. A *language model* plays the same game". This is the human-scale origin the guide asks for, and it pays off later when `language-model.md` quotes Shannon's one bit per character.
- **The route** is one question per section, in words the reader mostly knows.
- **The scope sentence** names where gated cells, state space models and attention live, so a reader who expected LSTMs in a chapter called "Sequence Models" is not left wondering.

## Structural findings

- The introduction never says that the chapter is the last of the Basics part or that it changes the data type from images to text after two chapters of convolutional networks. One clause ("After two chapters on images") would place it, the way Strang's "This chapter begins the second half" does.
- The Resources list is long (twice the introduction) but each entry says which section it accompanies, which is the right form. It uses the spaced dash as a separator in every entry; that is a list convention, not prose, and does no harm.

## Passage-level findings

- **IDX-1** (moderate, explanation). Second paragraph. Quote: "feeding each chosen token back in as input. The difficulty is the". The paragraph introduces, in order: language model, token, chain rule of probability, generation by feedback, the growing prefix, the window, the count table and its exponential appetite, the recurrent network, the hidden state, and large language models. That is ten ideas in nine sentences. A reader who has only seen image classifiers cannot unpack "By the chain rule of probability, these predictions multiply to the probability of the whole text". Fix: split in two. First paragraph: the guessing game, and that a model which predicts the next token can also write, by feeding its choice back in. Second paragraph: the prefix grows, a window forgets, and a recurrent network keeps a summary of fixed size. Leave the chain rule and the count table to their sections.
- **IDX-2** (minor, explanation). Quote: "predict one token at a time, so tokenization, perplexity and decoding". Three terms that the roadmap has not introduced ("perplexity" appears nowhere else in the introduction). Fix: "so what this chapter teaches about cutting text into tokens, scoring a model, and turning its predictions into text carries over to them."
- **IDX-3** (minor, thought). Quote: "how it compares with counts. :numref:`sec_bptt` asks why gradients through". The route lists implementation before backpropagation through time, while `rnn.md` ends by promising the gradient analysis "before we train it" (see RNN-5 and the chapter file on order). Whichever order is kept, the route and the hand-offs must agree.

## Exercises

None, correctly.

## Still open from October

Nothing specific to the introduction.
