# Writing pass log: Codes, Compression, and Perplexity

Source: `chapter_mdl-information-theory/mdl-coding-compression.md`. Not
split. Lightest hand, as assigned: correctness items, one ending, cell
introductions, the missing definitions, perplexity reduced against the main
text.

## Words

Prose words (checker count, body and exercises): 7277 before, 7394 after
(+117). Checker: "OK: only prose changed", no em-dash warning;
`lint_source.py` clean. The section is longer, against the rule. About 300
words are additions the review or the assignment asked for (definitions of
joint and conditional entropy with the chain rule, the questions game, what
`zlib` does, the cost of the 70-billion-parameter model, the answer to
"1.2 nats", why the typical set exceeds $2^{nH}$); about 190 were cut
(perplexity subsections, Bayesian paragraph, recap, exercise 10(a)). If the
parallel writer's definitions in the entropy section are sufficient, the
gloss in "The Entropy Rate" can shrink to the reference plus the chain rule.

Numeric checks were done with `python3` (NumPy-free arithmetic) and by hand;
no SciPy or PyTorch on this machine.

## Correctness changes

- **COD-14.** Before: "So every rate below $H$ fails on almost every
  message." Now: "So every fixed-length code of rate below $H$ fails with
  probability tending to one." The converse is also introduced as "a
  fixed-length code with only $2^{nR}$ codewords". Achievability is now the
  fixed-length scheme (index the typical set, give up on the rest); the
  flag-bit variant is one parenthesis. Verified against the displayed bound
  $P(B) \to 0$, which is what the proof shows.
- **COD-11, units of $H_{\textrm{rate}}$.** Before: bits in
  :eqref:`eq_mdl-entropy_rate`, "in nats" in the perplexity floor. Now bits
  everywhere. The floor reads
  $\mathbb{E}[\textrm{CE}_n] \geq H_{\textrm{rate}} \ln 2$,
  $\mathbb{E}[\textrm{PPL}_n] \geq 2^{H_{\textrm{rate}}}$; the proof carries
  the factor $\ln 2$; "$e^H = 1.38$" is "$2^H = 1.38$"
  ($2^{0.469} = 1.384$, recomputed); the Summary and exercise 10 (now part
  (a), "$\textrm{PPL}_{10} < 2^H$") follow. This agrees with exercise 1(b),
  which already wrote $2^H$. One sentence in "The Entropy Rate" says that
  entropies of sequences are in bits without subscript from there on.
  **This changes a labelled equation** (`eq_mdl-ppl-floor`); the statement
  is equivalent ($e^{H \ln 2} = 2^H$). The binary-entropy notation
  $H_b(\cdot)/\ln 2$ in exercises 8 and 9 is unchanged.
- **COD-S1 / COD-12.** "The Entropy Rate" now opens with the joint entropy
  (entropy of the block as one variable), the conditional entropy (kept
  definition), and the chain rule as the averaged logarithm of the product
  rule, with "defined in :numref:`sec_mdl-information_theory`" (section
  label; the parallel writer's new labels are not visible). After the
  equation: "By the chain rule, the first expression is the average of the
  losses of which the second is the last." The monotone-decrease intuition
  names its fact: "(conditioning never increases entropy)".
- **COD-5.** The "one bit is not an artifact" paragraph now says that
  Huffman's $1.4$ bits is only $0.1$ above the entropy, that no codeword is
  shorter than a bit, and adds the case $0.999$: entropy $0.011$ bits
  (recomputed: $0.0114$), cost still $1$.
- **COD-4.** Lagrange step: "Hence $2^{-\ell_i} = c\,p_i$ for one constant
  $c$ and all $i$. Summing both sides over $i$ gives $c = 1$".
- **COD-6.** The "Proof" with no proposition is a "because termwise ..."
  clause after :eqref:`eq_mdl-wrong-code-length`.
- **COD-7.** "name dyadic cells inside disjoint message intervals, so by the
  first half of the Kraft proof they form a prefix code".
- **COD-8.** Added after the four percentages: "These rates leave out the
  model. At two bytes per parameter it occupies $140$ GB, against $1$ GB for
  each data set, and the authors report that with the model counted it no
  longer competes." Checked in the paper (arXiv 2309.10668, Section 5.1 and
  Table 1 caption): "All datasets are of size 1GB"; "We encode each neural
  network parameter with 2 bytes"; "language models suffer a huge loss in
  compression rate due to their large size, which cannot be offset when
  compressing only 1GB of data". I did not extract the adjusted percentages
  from the table, so none is quoted.
- **COD-9.** $4.078$ kept, with "(the cell does not print it)". Recomputed
  from the text file (sha1 matches the cell's): 173,428 characters, 27
  symbols, entropy $4.0783$ bits. Also recomputed and kept: $226$ bits and
  $0.001$ bits per character; $0.645$ (Huffman on pairs); $971$, $29$,
  $1.2$ bits (coin); $2^{-103}$; $0.005 + 0.300$.
- **Exercise 8(a).** Cites :eqref:`eq_mdl-ce_decomp` instead of
  :eqref:`eq_mdl-wrong-code-length` (the decomposition, which is in nats in
  its home, is what the part uses; "in nats" dropped).
- **Exercise 10(a)** (prove the floor) removed; the body owns the proof.
  Former (b), (c) are now (a), (b). No reference to the parts by letter
  exists in the repository.
- No finding disagreed with. COD-S4 (do not split) followed.

## Six before/after pairs

1. Before: "How long are the Shannon codes for the two running sources, and
   what does the wrong code cost? The cell builds each code ..." After: "The
   cell builds each code from the distribution it is given and averages its
   lengths under the data distribution, for the four cases just worked by
   hand."
2. Before: "What do the coder's interval and codeword look like for
   `aaaaaaaaba`, and what does the symbol-by-symbol Shannon code spend on
   the same message?" After: "We run the coder on `aaaaaaaaba`."
3. Before: "How many bits per character does such a model need for *The Time
   Machine* ... and how does it compare with `zlib`?" After (before the
   second cell): "A longer context predicts better once its counts are
   learned, so one might expect the code length to fall with every added
   character. Does it?" and after the cell: "It does not."
4. Before: "Thus every code wastes bits in two ways: ..." After: "The
   identity :eqref:`eq_mdl-code-length-decomp` is worth more than the bound
   it proves. Every code wastes bits in two ways: ..."
5. Before: "What perplexity does a model reach on a stream of five tokens
   ..., and does the base of the logarithm matter?" After: "The cell
   computes it in nats and in bits for five tokens to which a model assigned
   the probabilities ..."
6. Before: "We can now build a prefix code from any distribution ... We can
   code ... We can compare ..." After: the paragraph starts at "Every code
   in this section needs a model whose probability for each symbol the
   encoder and the decoder can both evaluate."

"Thus" at the start of a sentence: eleven before, none after. The question
before the typical-set cell is kept (the review asks for that).

## Repetition

- Removed: the "We can now" recap (COD-17); the hand numbers repeated after
  the Shannon-code cell (the paragraph now reads the Kraft column, which the
  hand calculation did not give: $1$, $1$, $0.8125$, $0.875$); the
  marginal-likelihood half of the Bayesian paragraph (COD-10); "Dependence
  lowers this floor ..."; "The cost of learning a model is the subject of
  the next subsection."
- Against the main text (COD-S3): "Perplexity as a Branching Factor" keeps
  the definition (labelled equation), cites :numref:`subsec_perplexity` for
  the branching-factor reading in one sentence, and keeps the log-base
  remark. "Perplexity across Tokenizers" cites
  :numref:`subsec_bits-per-byte`, keeps :eqref:`eq_mdl-bpc` (cited by two
  language-model chapter briefs), and says that bits per character equal
  bits per byte when each character is one byte.
- Kept as owner: the perplexity floor (proposition, proof, the 35% example);
  Kraft, the entropy bound, arithmetic coding, MDL, entropy rate, AEP.

## Ranking, propositions, labels

- Demoted: the unnamed "Proof" under :eqref:`eq_mdl-wrong-code-length`
  (equation and label kept). The four Proposition blocks stay.
- Ranking added in the roadmap: "Two results carry the section, the entropy
  bound for symbol codes and the two-bit bound ... typical sequences can be
  skipped on a first reading."
- No labels added, removed or renamed. No split markers.
- COD-10: the Bayesian information criterion is named in the coin example;
  the precision remark closes the coin; the subsection ends on the
  compressor paragraph.

## Cells to cut or redesign on the development machine

- `information-theory-perplexity` (COD-S3): five-token toy; "the least
  interesting cell". Prose around it is now two sentences. Cut.
- `information-theory-coding` (no finding ID; same kind as INF-S5): confirms
  four hand calculations. Kept short; candidate to cut or to print the
  slack decomposition instead.
- Imports cell (COD-19): four framework tabs on a section that uses `math`,
  `numpy`, `re`, `zlib` and `d2l`.
- COD-9: print the letter entropy in
  `mdl-coding-compression-a-language-model-as-a-compressor-2`.

## Other deferred findings

- Exercises out of text order, and a new exercise that plays the questions
  game as a code (COD-S2): exercise changes beyond named errors.
- Exercise 9(d): $\sigma$ as a stay probability (notation clash with the
  standard deviation); left.
- Exercise 1 title "Perplexity across tokenizers" now carries the tokenizer
  point alone, as the review anticipated.
- Bits-back coding and ANS in Further Reading: another file.
- COD-1: done in a clause. COD-2, COD-3, COD-13, COD-15, COD-16, COD-17,
  COD-18: done.

## Claims added or changed

- `zlib` "replaces repeated strings by pointers and codes the result with a
  Huffman code" (DEFLATE: LZ77 plus Huffman; from knowledge).
- The code $0, 10, 110, 111$ as the questions "Is it the first symbol?",
  "the second?", "the third?"; the entropy bound "is the bound on the number
  of questions that the previous section promised".
- Surprisal $= \log_2(1/p_i)$, "the self-information of
  :numref:`sec_mdl-information_theory` in bits".
- Skewed source at $0.999$: entropy $0.011$ bits.
- Kraft sums left unspent: $0.1875$ and $0.125$ (from the stored output).
- The model-size sentence (COD-8), verified as above.
- "Charging $\tfrac12 \log_2 n$ bits for each parameter is the *Bayesian
  information criterion*."
- Chain rule as stated; "conditioning never increases entropy".
- "$1.2$ nats per character ... is $1.73$ bits" ($1.2/\ln 2 = 1.731$).
- Typical set: "far more than $2^{nH} = 2^{176.3}$ ... reaches out to $76$
  ones".
- Floor in bits: $H_{\textrm{rate}} \ln 2$ and $2^{H_{\textrm{rate}}}$.
- :numref:`subsec_bits-per-byte` "shows this on three tokenizers" (from the
  review's description of that section and its opening lines; I did not
  read the experiment cell).

## Slides that now disagree with the body

- The floor slide boxes
  "$\mathbb{E}[\textrm{PPL}_n] \ge e^{H_{\textrm{rate}}}$
  ($H_{\textrm{rate}}$ in nats)" and writes
  $\mathbb{E}[\textrm{CE}_n] \ge H_{\textrm{rate}}$; the body now has bits,
  $2^{H_{\textrm{rate}}}$ and the factor $\ln 2$. Equivalent, different
  unit.
- The cover kicker "§28.2" is a hardcoded number (rewritten by
  `gen_slides.py`).
- Any slide that presents the flag-bit scheme as the achievability argument
  now differs in emphasis only.
