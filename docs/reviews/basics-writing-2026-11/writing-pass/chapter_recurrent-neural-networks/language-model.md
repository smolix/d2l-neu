# Writing pass: `chapter_recurrent-neural-networks/language-model.md`

Section 8.3 of 7 (order: sequence, text-sequence, **language-model**, rnn,
rnn-implementation, bptt, decoding). Prose only; no code cell, label, figure,
exercise, or slide changed. Checker: "OK: only prose changed"; lint clean.

**Prose words:** 4,851 before, 4,841 after (the checker's count includes
exercises). The hand was light: the review rates the language of this section
highly, and the passages it lists under "What works" are untouched.

## Before and after

1. Opening (LM-1).
   - Before: "The chain rule of probability, [...], removes the difficulty."
   - After: "The chain rule of probability, [...], turns the problem into one
     we know."
2. Markov assumption (new judgment).
   - Before: (nothing)
   - After: "The assumption is crude, since a sentence does not forget its
     subject after two words, but it leaves a finite number of contexts to
     estimate."
3. Sampling.
   - Before: "The text a model writes shows directly how good its conditionals
     are. To check how much of a sample is new, we also measure the longest
     run [...]"
   - After: "It is tempting to judge a model by the text it writes. As a
     precaution, we also measure the longest run [...]"
4. Memorization.
   - Before: "On a corpus of 30,000 words, much of this fluency is copied, and
     the longest copied run measures how much."
   - After: "Unfortunately, on a corpus of 30,000 words it reads like Wells
     because much of it *is* Wells. The longest copied run measures how much."
5. Count sparsity.
   - Before: ":eqref:`eq_ngram_mle` gives each of them probability zero, and a
     single zero in the product :eqref:`eq_lm_factorization` declares the
     whole held-out text impossible."
   - After: "The maximum likelihood estimate :eqref:`eq_ngram_mle` gives each
     of them probability zero, and a single zero in the product
     :eqref:`eq_lm_factorization` declares impossible a text that we hold in
     our hands."
6. Perplexity result (LM-8).
   - Before: "[...] nearly uniform guesses, and no choice of $\alpha$ rescues
     it (the exercises ask you to try). The trigram's fluent samples replay
     its training text; on new text most of its table is empty."
   - After: "[...] nearly uniform guesses. No choice of $\alpha$ rescues it:
     the best value in the sweep of exercise
     :exref:`ex_language-model-smoothing-constant` still leaves its perplexity
     above 2,500. The fluent samples were a poor guide. On new text most of
     the trigram's table is empty."
7. Bits per byte.
   - Before: "Perplexity depends on what a token is. [...] We therefore change
     the denominator."
   - After: "Perplexity has one serious flaw: it depends on what a token is.
     [...] The repair is simple: we change the denominator."
8. Closing of the section.
   - Before: "We can now turn any text into minibatches of next-token
     prediction problems. The count models of this section could not use them
     well. Their tables cannot relate one context to another, and their
     context ends after $n-1$ tokens."
   - After: "A window offers far more context than a count table can use: our
     tables ran out of data at two words of context."

## Repetition removed

- **"Count tables cannot generalize"** was told at the end of Kneser--Ney
  (cat/feline), at the end of the Zipf section, in the closing paragraph, and
  in the summary. Kept at Kneser--Ney (owner) and in the summary; the Zipf
  sentence ("Models that *generalize* across related contexts avoid this") is
  cut and the closing paragraph reduced to one sentence.
- **"Most word pairs have a single successor"**: stated with the trigram table
  (owner), then again in full under sampling. The second is now "We have seen
  why."
- **87% unseen trigrams**: the Zipf section restated it ("most trigrams of new
  text are missing from the table, as we measured above", LM-9). Reduced to
  "This is the heavy tail behind the unseen trigrams of the held-out text."
- **"The trigram's samples replay its training text"** (sampling, then again
  after the perplexity table): second telling replaced by "The fluent samples
  were a poor guide."
- **Bits per byte does not depend on the tokenizer**: said at the start and
  again at the end of the subsection; the second is one clause now.
- **Facts owned by `text-sequence.md`**: "tokenization is a design choice" is
  a parenthetical reference; the gloss "(byte-level BPE with the GPT-2
  pre-tokenization pattern)" is dropped.
- **Random minibatches**: said in the window paragraph and again for the data
  loader; the second now only says that validation is read in order.
- Formulaic closing ("We can now ...") replaced; the roadmap no longer reads
  "This section first ... It then ...".

## Review findings

Applied:
- **LM-1**: "removes the difficulty" replaced as proposed.
- **LM-5** (partly): the Pitman--Yor paragraph is now one parenthetical
  sentence that keeps both citations and drops the forward reference to the
  Zipf subsection. The review wants it in the chapter's Further Reading; that
  is `index.md`, outside this file. If it is moved there, delete the
  parenthesis here.
- **LM-7**: "the corresponding sums" is now "the total and the number of
  nonzero continuation counts in context $u$" (checked against the function).
- **LM-8**: the claim about $\alpha$ now carries its result (see below).
- **LM-9**: restatement removed.
- **LM-10** (partly): "which later chapters use for character-level models"
  dropped; the option is one parenthetical sentence. Moving it to a code
  comment needs a cell change.
- **LM-11** (partly): the train/validation gap paragraph rewritten to lead
  with the reason. Marking the dropped windows in the figure is deferred.

Deferred:
- **LM-S1** (order: perplexity before smoothing, Zipf with sparsity): moves
  code cells. Two sentences now soften it: the class paragraph says why
  `alpha` is there "we see below", and the Kneser--Ney closing says its gain
  is measured in :numref:`subsec_perplexity` instead of asserting "far better
  use of the counts" before any number.
- **LM-2** (conditioning order, $x_{<t}$): chapter-wide notation decision.
- **LM-3** (`prob` smooths before smoothing is motivated): code.
- **LM-4** (word vocabulary includes held-out words): code and outputs. The
  prose now says "we must discount its win".
- **LM-6** (a worked number for the Kneser--Ney recursion): new cell.
- Exercise 2c rewording: exercises not touched.

## Factual claims added or changed

1. "the best value in the sweep of exercise [...] still leaves its perplexity
   above 2,500." Checked by running the section's own counting code (pure
   Python, deterministic) over the 19 values of the exercise: best trigram
   perplexity 2,586 at $\alpha \approx 0.0056$; the script reproduces the
   stored 840.7 / 775.1 / 2618.1 at $\alpha = 0.01$. Side observation, not
   written into the text: at its own best $\alpha$ (1.0) the unigram reaches
   607 and beats the bigram (772), which is what exercise 3b asks about.
2. "For more than four in five word pairs the table lists a single successor"
   replaces "most word pairs occur once, so their row of the table holds a
   single successor, observed once." The cell prints 84% for contexts with one
   distinct successor; it does not print how often they occur.
3. "The assumption is crude, since a sentence does not forget its subject
   after two words, but it leaves a finite number of contexts to estimate."
4. "Surprisingly many tasks can be phrased as a continuation."
5. Laplace smoothing: "It is a crude remedy, with two defects."
6. "The fluent samples were a poor guide."
7. Train/validation gap: "the windows right after the training range contain
   tokens that training has already used as targets" (checked against the
   constructor: the last training window's target reaches position
   `num_train - 1 + num_steps`).
8. "our tables ran out of data at two words of context" (87% of held-out
   trigrams unseen).

## Slides

No statement in the slides section disagrees with the revised body. The
slides never mentioned the Pitman--Yor reading.
