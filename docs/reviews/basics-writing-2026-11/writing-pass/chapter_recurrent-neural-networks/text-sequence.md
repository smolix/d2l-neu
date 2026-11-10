# Writing pass: `chapter_recurrent-neural-networks/text-sequence.md`

Section 8.2, "From Text to Tokens" (second of seven sections; only
`sequence.md` precedes it, and it explains nothing about tokenization, so
this section owns every tokenizer fact in the chapter).

## Size

Prose words (checker count, exercises included): 4,052 before, 4,052 after.
The review asked for six prose additions (TOK-4, 5, 6, 8, 9, 11 and the
Zipf pointer, about 130 words); they are paid for by removing repetition.
`check_prose_only.py`: "OK: only prose changed". `lint_source.py`: clean.

The section was already plain and had voice in the places the review
lists, so the hand is light: about one sentence in five changed.

## Before and after

1. Before: "This choice determines the size of the model's output layer, the
   length of every document it reads, and even which arithmetic problems it
   can solve."
   After: "This choice determines the size of the model's output layer, the
   length of every document it reads, and even how hard arithmetic is for
   it."

2. Before: "We keep such a step, because it simplifies the character- and
   word-level experiments below and because several later sections reuse
   it. It has two drawbacks. Normalization throws away information"
   After: "We keep such a step for the character- and word-level experiments
   below, and for the sections of the book that still work with such tokens.
   It is crude, though: normalization throws away information"

3. Before: "Second, the third merge already glued a word to the *space that
   follows it*: nothing stops BPE from merging across word boundaries. On a
   real corpus this becomes a problem, which we fix below."
   After: "Second, the third merge glued a word to the *space that follows
   it*. A trailing space is harmless in itself, but nothing stops a later
   merge from attaching the next word to it. On a real corpus this is
   exactly what happens, and we fix it below."

4. Before: "**The tokenizer is defined by its merge ranks**, and training and
   inference agree by construction."
   After: "**the tokenizer is defined by its merge ranks**. Why lowest rank
   first? Training applied each merge to text in which all earlier merges
   had already been made, so replaying them in rank order retraces training
   (the exercises make this precise)."

5. Before: "Production vocabularies of 50,000 to 200,000 tokens are therefore
   not much further along the curve: on English prose they reach four to
   five bytes per token, about one token per word, as we measure below."
   After: "It would be a mistake, however, to extend the line to production
   vocabularies of 50,000 to 200,000 tokens. The upper end of our curve is
   flattered: at 4,096 tokens the tokenizer has learned words and phrases of
   the very text it is measured on. On English prose at large, production
   vocabularies reach four to five bytes per token, about one token per
   word, as we measure below."

6. Before: "The unconstrained tokenizer has a flaw. Alongside word fragments,
   its vocabulary contains tokens that straddle word boundaries:"
   After: "The composite "Time Traveller " is not alone. Alongside word
   fragments, the vocabulary of the unconstrained tokenizer contains many
   tokens that straddle word boundaries:"

7. Before: "Our tokenizer reaches about two thirds of the compression of
   tokenizers 50 and 200 times its size, because it was trained on this
   text."
   After: "Although it is 50 to 200 times smaller, our tokenizer reaches
   about two thirds of their compression, because it was trained on this
   text."

8. Before: "The two parents of each longer token are recovered by re-running
   BPE on its own bytes with all lower-ranked merges"
   After: "The table stores the bytes of each token, not the pair that was
   merged to form it, so we recover the two parents of a longer token by
   re-running BPE on its bytes with all lower-ranked merges [...] This loop
   can be skipped; what matters is the check that follows."

## Repetition removed

- "Later sections use it", told three times (normalization, `Vocab`,
  `build`): now said once, at the normalization step. `Vocab` is introduced
  as "the character- and word-level pipeline from the start of this
  section".
- "Nothing is out of vocabulary": owned by "Text as Bytes". The restatement
  under the merge rule is reduced to the new fact (unmerged input falls back
  to bytes); the one in the `Vocab` paragraph ("the OOV problem that
  byte-level BPE eliminates") is gone. The Summary keeps it.
- The three-digit cap of later tokenizers was told under "Pre-Tokenization"
  and again under "Fertility, Digits, and Glitch Tokens"; only the second
  remains.
- Tie-breaking was explained at the hand trace and again at `train`; `train`
  now refers to the trace in one clause.
- "In-domain text flatters the larger vocabularies" stood as an aside before
  the plot; it is now the explanation after the plot (TOK-6), stated once.
- The byte lengths of "é", the Chinese character and the emoji were given
  before and after the cell; the second telling is one clause.
- Closing restatements cut: "A vocabulary entry carries only what the model
  learned about it during training" (the paragraph opens with the same
  thought); "Text that one tokenizer encodes in few tokens can therefore
  take many under another"; "frequent fragments become single tokens, rare
  ones fall back toward bytes" after the toy encoding; "`to_tokens` maps
  indices back to tokens"; "The next section models the token sequences it
  produces" in the opening (the Summary ends with it).
- The sentence saying that `from_tiktoken` imports `tiktoken` lazily is cut;
  the code comment says it.

## Review findings

Applied (prose): TOK-1, TOK-4, TOK-5 (reason given in one sentence, with a
pointer to the exercises), TOK-6 (second option: the text now says the
upper end of the curve reflects in-domain text; the non sequitur
"therefore" is gone), TOK-7, TOK-8 (second option: the loop is marked as
skippable), TOK-9, TOK-10 (the use named is `<eos>` in `sec_decoding`),
TOK-11. Structural: the Zipf pointer (one clause to
`sec_language-model`); the three "later sections" justifications reduced to
one.

Deferred:
- TOK-2: needs the plain `_merge` loop and a simpler `train` (code change).
- TOK-3: needs the class cell split across subsections (code change). The
  prose now names all three items to ignore (`pattern`, `specials`,
  `byte_ids`).
- TOK-6, first option: a held-out compression curve needs a new cell and
  output.
- Structural, classical pipeline inside the BPE story: moving `_tokenize`,
  `Vocab` and `build` reorders code cells.
- Structural, glitch tokens to the Language Models part: moves a cell
  between files.
- Structural, special tokens unused in the chapter: needs a chapter
  decision (the 1,027 count runs through four sections).

## Factual claims added or changed

- "Our novella runs to nearly 180,000 bytes, more than five for each of its
  words" (TOK-9). No cell prints the byte count; it follows from the stored
  output 65,741 tokens at 2.72 bytes per token (about 179,000 bytes; the
  same in all four frameworks) and the 32,775 words. The word count is of
  the normalized text. Please check on the next execution.
- "at 4,096 tokens the tokenizer has learned words and phrases of the very
  text it is measured on" and "It would be a mistake [...] to extend the
  line": the stored plot reaches about 4.6 bytes per token at 4,096, above
  the 4.06 measured for GPT-2 below. "Phrases" holds because `big` is
  trained without a pattern.
- "A trailing space is harmless in itself, but nothing stops a later merge
  from attaching the next word to it" (TOK-4).
- "Training applied each merge to text in which all earlier merges had
  already been made, so replaying them in rank order retraces training":
  replaces "agree by construction". This is the claim of starred exercise 6.
- "most of them rarely used (:numref:`sec_language-model`)": said of the
  word types; rests on the Zipf subsection there.
- "the decoders of :numref:`sec_decoding` accept it as a signal to stop":
  replaces "The encoder-decoder models of `sec_machine_translation` will
  use all three" (that section is in the Attic). `beam_search` and
  `generate` take `eos_id`; nothing in the chapter uses `<pad>` or `<bos>`,
  and the text no longer names a user for them.
- "*prompt injection*, in which text supplied by a user is read by the
  model as an instruction": definition added.
- "the table `byte_ids` [...] needed only later": it is the identity until
  `from_tiktoken` replaces it.
- "Although it is 50 to 200 times smaller": was "50 and 200 times its
  size"; same numbers.
- Removed claim: "English text takes about one byte per character" after
  the bytes cell (the sentence before the cell already says ASCII takes one
  byte).

## Slides that now differ from the body

None contradicts the body. Three small differences in emphasis:
- "Check: Split “I've got 1234 cats!”" mentions that later tokenizers split
  after three digits; the body now says this only under "Fertility, Digits,
  and Glitch Tokens", not under "Pre-Tokenization".
- "Encoding Applies the Lowest-Rank Merge": "training and encoding agree by
  construction"; the body now gives the reason instead of that phrase.
- "Each Doubling of the Vocabulary Adds About the Same Compression" says
  "Measured on the training text" but not that the upper end overstates
  what a general vocabulary reaches.
