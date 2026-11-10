# Codes, Compression, and Perplexity

Source: `chapter_mdl-information-theory/mdl-coding-compression.md`. About 6,000 words of prose and mathematics before the exercises, 7,900 with them; 9 code cells, all framework-independent apart from the imports.

## Verdict

This is the best section I read in the part and the one that comes closest to MacKay. It opens on a measurement (zlib stores *The Time Machine* in 2.84 bits per character, a three-character counting model in 2.55), promises to explain it, and does: a prefix code is a budget, the best lengths are the surprisals, rounding costs a bit per symbol, arithmetic coding rounds once per message, and so any next-symbol model is a compressor. Every step has a number the reader can check and most have a counterexample. The arithmetic coder is worked to the bit (`011001`, six bits for 4.69 bits of surprisal), the typical set of 200 biased bits is counted exactly, and the text says plainly what its bounds are worth ("up to a factor $2^{\pm n\delta}$, which is large"). I recomputed all the numbers and found them right; I found no error in the propositions or proofs. What separates the section from the models is smaller than in the first section and mostly at its edges. It uses joint entropy, conditional entropy and "surprisal" without any of them having been defined in the chapter so far; it never says that the yes/no questions of the previous section are codewords, although that section promised it would; the unit of $H$ changes twice; and the perplexity subsections re-teach the main text at the same level. The single most important fix is the first: give the entropy-rate subsection its definitions.

## Scores

| Dimension | Score |
|:--|:--|
| Clarity of explanation | 4 |
| Clarity of thought | 4 |
| Pedagogy | 5 |
| Language | 4 |
| Ease and enjoyment | 4 |

## Comparison with Cover and Thomas and with MacKay

Cover and Thomas spread this material over chapters 3 (AEP), 4 (entropy rates), 5 (data compression: Kraft, optimal lengths, McMillan, Huffman) and 13 (arithmetic and universal coding), and prove the AEP first. MacKay (chapters 4 to 6) also starts from the typical set and the bent coin, then symbol codes, then stream codes. This section reverses both: codes first, the typical set last, with the verdict "The typical set serves proofs, and it makes a poor coder." For a deep learning reader that is the right order, because the result they need (log-loss is a file size) comes from arithmetic coding and needs no asymptotics. The section also takes MacKay's central point about arithmetic coding, that it separates the model from the coder ("Nothing in the construction required independent symbols"), and puts it where MacKay puts it.

Where it matches the models: the Kraft inequality by dyadic intervals is the proof both books would recognise, and it is reused to name the arithmetic code's interval, an economy neither book has. The Huffman code for $Q$ that beats the "lower bound" $\textrm{CE}_2(P, Q)$ is a Cover and Thomas style counterexample that fences the claim. The most probable message against the typical message (all zeros at $2^{-103}$; 60 ones at $2^{-176.3}$) is MacKay's bent-coin lesson with exact numbers. One source, $P(\texttt{a}) = 0.9$, serves the symbol-code penalty, the arithmetic coder, the Markov chain and the perplexity floor.

Where it falls short: Cover and Thomas define joint and conditional entropy and the chain rule in chapter 2, before any of this; here they come two sections later. MacKay tells the reader how the competitor works (his chapter 6 explains Lempel and Ziv next to arithmetic coding); here `zlib` is a black box that loses. And MacKay's exercises on this material include the ones a reader does for fun (the weighing problem, the game of sixty-three); here the questions game of the previous section, which could play that part, is dropped.

## What works

- **The opening experiment and its return.** The numbers 2.84 and 2.55 open the section, are reproduced by a 12-line model in the middle, and come back as an upper bound on the entropy rate. The order-4 model losing to order-3 (2.78 against 2.55) is a surprise the code delivers, and it is then used as the motivating case for MDL.
- **The budget picture.** "A codeword of length $\ell$ ends at a leaf and removes everything below it, a fraction $2^{-\ell}$ of the tree"; "Four codewords of length $1$ would use $2$ trees". The figure, the telephone-number example and the three length profiles make the Kraft inequality obvious before it is stated.
- **Heuristic, then proof.** The Lagrange argument finds $\ell_i = \log_2(1/p_i)$ "for a moment" with real lengths; the proposition then proves the bound through the implied distribution $q_i = 2^{-\ell_i}/Z$. The identity $\mathbb{E}[\ell] = H_2 + D_{\textrm{KL}} - \log_2 Z$ and its reading ("every code wastes bits in two ways") is the most illuminating equation of the section. The numbers $0.305 = 0.005 + 0.300$ are correct.
- **The arithmetic coding subsection** is complete in 600 words: the interval, the width as the message probability, the dyadic cell of length $2^{-\ell}$ with $\ell = \lceil \log_2(1/w) \rceil + 1$ and why it fits, the two-bit bound, the code cell with an `assert`, and the honest limits (the decoder must know $n$; floating point fails after about a hundred symbols).
- **Candour.** "We grant these facts." "a theorem we grant". "No compressor reports a lower bound." "The floor does not bound the perplexity of a single short text", with the 35% example. These are the sentences of an author who knows what is proved and what is not.
- **Sentences to protect:** "after `q` comes `u`"; "Most messages are unlikely, and the most likely message is unrepresentative."; "MDL is Occam's razor with simplicity measured in bits."; "The typical set serves proofs, and it makes a poor coder."
- **Code that teaches.** All nine cells compute something the prose reads. The one plot (probability and fraction of strings against the number of ones) shows a computed result and the text reads its two peaks.

## Structural findings

**COD-S1 (major, explanation). Joint and conditional entropy are used before the chapter defines them.**
"The Entropy Rate" writes $H(X_1, \ldots, X_n)$ and $H(X_n \mid X_{<n})$, and argues that "each extra symbol of context can only help prediction". The chapter defines joint entropy, conditional entropy and the chain rule in `mdl-mutual-information.md`, two sections later. The previous section mentioned "conditional entropy" once, in words, as the name of the label-noise floor. A lookup reader, and a careful front-to-back reader, has no definition of either symbol here, and the fact that conditioning does not increase entropy is used for the monotone decrease without a name or a reference. The October review asked for these definitions to move forward; that is still open.
Direction: preferably define them in the first section (see INF-S4 in `mdl-information-theory.md`). Failing that, add three sentences at the start of "The Entropy Rate": the joint entropy is the entropy of the block as one variable ("nothing new", as Cover and Thomas say); the conditional entropy is the average entropy of the next symbol given the past; the chain rule $H(X_1, \ldots, X_n) = \sum_i H(X_i \mid X_{<i})$ is the logarithm of the product rule. The two limits in :eqref:`eq_mdl-entropy_rate` then read as an average and its last term.

**COD-S2 (moderate, pedagogy). The questions game is not closed.**
The previous section said: "the best questioning strategy needs between $H_2$ and $H_2 + 1$ questions on average, as :numref:`sec_mdl-coding-compression` proves." This section never mentions questions. The bridge is one sentence and the figure for it already exists: each internal node of the tree in :numref:`fig_mdl-kraft-tree` is a yes/no question, the codeword is the list of answers, and the code $0, 10, 110, 111$ is the strategy "Is it $a$? Is it $b$? Is it $c$?". Without it the promised proof is not delivered in the reader's eyes, and the best device of the first section is lost.

**COD-S3 (moderate, thought). The perplexity subsections repeat the main text at the same level.**
`chapter_recurrent-neural-networks/language-model.md` ("Perplexity", "Bits per Byte") defines perplexity, reads it as an effective branching factor, lists the reference points (1, the vocabulary size), shows that it depends on the tokenizer with a three-tokenizer experiment, converts to bits per byte, and quotes Shannon's one bit per character. "Perplexity as a Branching Factor" and "Perplexity across Tokenizers" here cover the same ground with a five-token toy cell and the same conclusions. What this section adds is "The Perplexity Floor" (the proposition, the use of Jensen, and the warning that the floor holds only in expectation) and the remark that perplexity does not depend on the log base. The appendix also says "bits per character" where the main text says "bits per byte".
Direction: reduce the first and third subsections to one paragraph each that states the definition, cites `subsec_perplexity` and `subsec_bits-per-byte`, and uses the main text's unit. Keep the floor in full. This removes about 400 words and the least interesting cell.

**COD-S4 (minor, thought). Is the section two sections?**
At 6,000 words it has four parts: symbol codes, stream codes with MDL, entropy rate with typical sequences, perplexity. It holds together because each part answers "how short can the file be", and I would not split it. If length must come down, the perplexity reduction of COD-S3 and the Bayesian paragraph of the MDL subsection are the places.

## Passage-level findings

**COD-1** (minor, explanation). Under the section opening. Anchor: "general-purpose compressor `zlib` stores it in $2.84$ bits per character. A"

The comparison is the hook of the section, and the reader is not told what `zlib` does. One clause would connect it to the section: it replaces repeated strings by pointers and codes the result with a Huffman code, the symbol code of the first part. The reader then understands why a model that predicts characters can beat it.

**COD-2** (minor, thought). Under the section opening. Anchor: "The section rests on one fact from the previous section. The expected"

Restating the decomposition in bits with the running pair is right for a reader who lands here from a reference. It should cite :eqref:`eq_mdl-ce_decomp`, which it does not, so that the reader can find the proof.

**COD-3** (moderate, language). Under "Shannon Lengths and the Entropy Bound" and after. Anchor: "So each symbol gets as many bits as its surprisal, and the expected length"

"Surprisal" is used here for the first time in the chapter and about ten times afterwards, without definition. The previous section says "self-information" and "surprise". See INF-4 in `mdl-information-theory.md`. Fix there, or define it here in a clause: "its *surprisal* $\log_2(1/p_i)$, the self-information of the previous section in bits".

**COD-4** (minor, explanation). Under "Shannon Lengths and the Entropy Bound". Anchor: "Hence $p_i = \lambda\, 2^{-\ell_i} \ln 2$ for one constant $\lambda$ and"

The derivative of the constraint was just given as $-2^{-\ell_i}\ln 2$, so the sign has been absorbed into $\lambda$ without comment, and "The constraint fixes the constant to $1$" refers to a different constant (the proportionality between $2^{-\ell_i}$ and $p_i$) from the $\lambda$ just named. Nothing is wrong, but a reader following with a pencil stumbles. Fix: "so $2^{-\ell_i} = c\,p_i$ for one constant $c$, and summing both sides over $i$ gives $c = 1$."

**COD-5** (moderate, thought). Under "Shannon Lengths and the Entropy Bound". Anchor: "The one bit is not an artifact of the proof. The best symbol code, built by"

The paragraph makes its claim and then gives as first evidence the Huffman code for the pair, which spends $1.4$ bits against an entropy of $1.295$: a gap of $0.1$, which suggests the opposite. The real evidence is the skewed source that follows (a gap of $0.53$), and even that is not one bit. The reader is left unsure what "the one bit" refers to. The claim that survives is that the gap can come arbitrarily close to one bit. Draft: "Huffman's algorithm builds the best symbol code, and for the pair it does much better than the Shannon code: lengths $1, 2, 2$ and $1.4$ bits, only $0.1$ above the entropy. But no symbol code can spend less than one bit on a symbol, and for a skewed source that is nearly the whole of the allowed slack. A source that emits `a` with probability $0.9$ has entropy $0.47$ bits and still pays $1$; at probability $0.999$ the entropy is $0.011$ bits and the cost is still $1$."

**COD-6** (minor, language). Under "Mismatched Codes and Cross-Entropy". Anchor: "**Proof.** Termwise, $\log_2(1/q_i) \leq \ell_i < \log_2(1/q_i) + 1$."

A "Proof" with no proposition before it. The one-line argument is fine as a sentence ("because termwise ...").

**COD-7** (minor, explanation). Under "Arithmetic Coding". Anchor: "length $n$ lie in disjoint message intervals, so they form a prefix code,"

Correct, and the step is fast: disjoint message intervals contain disjoint dyadic cells, and by the first half of the Kraft proof disjoint cells mean no codeword is a prefix of another. Half a sentence naming the Kraft proof would make it follow for the reader.

**COD-8** (moderate, pedagogy). Under "A Language Model as a Compressor". Anchor: "probabilities compresses ImageNet image patches to $43.4\%$ of their raw"

The four percentages are impressive and the paragraph ends on them. The sentence before says the model "must be shared once", but the reader is not told what that does to the result: a 70-billion-parameter model is far larger than the data sets it compresses, and the paper reports that with the model counted the rates are no longer competitive at that scale (I recall this from the paper and did not check the exact figures; the reviser should). That fact is the best possible motivation for the next subsection, where $L(\mathcal{M})$ enters. Fix: add one sentence stating it, and let "Minimum Description Length" open from it.

**COD-9** (minor, thought). Under "A Language Model as a Compressor". Anchor: "$4.08$ agrees to two decimals with the entropy $4.078$ of the text's letter"

No cell computes the entropy $4.078$; the reader must take it on trust in a section that otherwise shows every number. Add it to the print of the second cell (one line), or drop the third decimal.

**COD-10** (moderate, thought). Under "Minimum Description Length". Anchor: "A parameter count alone is not a description length. For a model with"

The subsection ends on its weakest paragraph. "There, a parameter was worth its cost only if ..." refers back three paragraphs to the coin, and the paragraph stops without saying what follows for a network. The rule it describes, $\tfrac12 \log_2 n$ bits per parameter, is the Bayesian information criterion, which a lookup reader will search for and not find named anywhere in the part. The paragraph on MAP and marginal likelihood before it is a list of correspondences without an example. Fix: end the subsection on the compressor paragraph ("The compressor experiment above applied the same rule ..."), which is its real conclusion; fold the precision remark into the coin example, with the name: "Charging $\tfrac12\log_2 n$ bits for each parameter is the *Bayesian information criterion*"; shorten the Bayesian paragraph to one sentence or move it to the Further Reading notebook, which already has "Description Length beyond Two-Part Codes".

**COD-11** (moderate, explanation; notation). Under "The Entropy Rate" and "The Perplexity Floor". Anchors: "$H(X_n \mid X_{<n}) = \mathbb{E}\big[-\log_2 p(X_n \mid X_{<n})\big]$. The" and "exponentiated entropy rate, and it holds only in expectation. Here we"

The opening sets a convention: the subscript $2$ marks bits. From "The Entropy Rate" on, $H$, $H(X_1)$ and $H_{\textrm{rate}}$ carry no subscript and are in bits; in "Typical Sequences" the text says "All numbers are in bits"; in "The Perplexity Floor" the same symbol $H_{\textrm{rate}}$ is "in nats". The binary entropy is written $H_2(0.9, 0.1)$ in the text, $H_b(\hat\theta)$ and $H_b(r_s)/\ln 2$ in the exercises. A reader who looks up :eqref:`eq_mdl-ppl-floor` from the main text and then :eqref:`eq_mdl-entropy_rate` gets two units for one symbol. Fix: keep the subscript throughout ($H_{2,\textrm{rate}}$ or state once that the rate is in bits), and write the perplexity floor as $2^{H_{\textrm{rate}}}$ with the rate in bits, which also matches exercise 1(b), where it already appears as $2^{H}$.

**COD-12** (moderate, explanation). Under "The Entropy Rate". Anchor: "*conditional entropy* of the next symbol given the past,"

See COD-S1.

**COD-13** (moderate, pedagogy). Under "The Entropy Rate". Anchor: "Shannon estimated the entropy rate of printed English at about one bit per"

This is where the chapter introduction's question ("a loss of 1.2 nats per character, is that good?") can be answered, and it is not. One sentence does it: 1.2 nats is 1.73 bits per character, better than the 2.55 of the counting model and still well above Shannon's estimate. See IDX-1.

**COD-14** (minor, correctness; confident). Under "Typical Sequences". Anchor: "So every rate below $H$ fails on almost every message. For stationary"

The converse just proved is for codes with $2^{nR}$ codewords, that is, fixed-length block codes that give up on the messages outside $B$. As a sentence about "every rate" it is too broad: the variable-length statement is the entropy bound of the first part, which concerns the average and says nothing about "almost every message". The paragraph also packs two achievability schemes (flag bit plus uncompressed fallback; fixed length with failure) and the converse into about 150 words. Fix: say "every fixed-length code of rate below $H$ fails ...", and present one achievability scheme, preferably the fixed-length one, since it pairs with the converse.

**COD-15** (minor, explanation). Under "Typical Sequences". Anchor: "The $2^{188.8}$ typical messages, those with $44$ to $76$ ones, are one in"

The reader has just been told that the typical set has "about $2^{nH}$ members", and $nH = 176.3$. The count is $2^{12.5}$ times larger. The text gave the warning three paragraphs earlier ("which is large") and exercise 6(c) explains it, but at this point the reader needs the connection made: "far more than $2^{176.3}$, because with $\delta = 0.1$ the set reaches out to 76 ones, and the messages there are the most numerous."

**COD-16** (moderate, language; pattern). Throughout. Anchors: "Thus codeword lengths are a budget: each codeword of length $\ell$ spends", "Thus every code wastes bits in two ways: by implying a distribution $Q$", "Thus cross-entropy is the length of a code built from the model, and KL is", "Thus bits per character is the total code length divided by the length of"

Eleven sentences in this section begin with "Thus" (one in the previous section). Each is the plain sentence after a result that the guide takes from Cover and Thomas, and each is good. Eleven with the same first word are a visible template. A second pattern: five of the seven cells that compute something are introduced by a question ("How long are the Shannon codes for the two running sources, and what does the wrong code cost?", "What do the coder's interval and codeword look like", "How many bits per character does such a model need", "How many of the $2^{200}$ messages are typical", "What perplexity does a model reach"). Fix: keep the sentences, vary the openings, and reserve the question for cells whose answer is not predictable (the order-4 model; the size of the typical set).

**COD-17** (moderate, language; repetition). Last paragraph before the summary. Anchor: "We can now build a prefix code from any distribution and bound its"

Four sentences of recap ("We can now ... We can code ... We can compare ..."), then the summary says the same again. The previous section closes with the same formula. The last three sentences of the paragraph are the real ending and are good: every code here needs a model whose probabilities both sides can evaluate, and two distributions known only through samples "give no code to measure". Fix: cut the recap and close on those sentences.

**COD-18** (minor, language). Under "Summary". Anchor: "bits, and MDL compares models by the total length of model and data. For a stationary source the"

The summary is one paragraph of about 170 words listing eight results. The sentence at the anchor joins arithmetic coding and MDL with "and". Three or four sentences that answer the opening's two questions (is there a code whose length is the log-loss; how close do real compressors come) would serve better.

**COD-19** (minor, pedagogy). Imports. Anchor: "from torch.nn import functional as F"

The four-tab imports cell loads `torch`, `nn`, `F` (and `autograd`, `gluon`, `init`, `nnx`, `optax` in the other tabs), none of which the section uses. The section needs `math`, `numpy`, `re`, `zlib` and `d2l` for one download and one plot. A reader sees four tabs and expects framework code that never comes.

## Exercises

Eleven problems, within the band, and the strongest set I have seen in the part: Huffman optimality by exchange, McMillan's theorem with a hint that makes it doable, the slack decomposition (which asks why Huffman's code for $Q$ does not contradict the bound), exact arithmetic coding with a decoder, typical against high-probability sets (credited to its source), the two-part coin that derives $\tfrac12 \log n$ and asks why the encoder cannot use the optimal spacing, and an extended compression project. I checked the stated numbers and derivations in exercises 1, 2, 4, 6, 8 and 9.

- **Exercise 10(a)** ("Prove :eqref:`eq_mdl-ppl-floor`, saying where Gibbs' inequality, the ...") asks for the proof that the text gives in full under "The Perplexity Floor", with the three ingredients named there. Cut part (a); parts (b) and (c) are good.
- **Exercise 8(a)** cites :eqref:`eq_mdl-wrong-code-length`, the ceiling inequality for Shannon codes. What it uses is the decomposition $\textrm{CE} = H + D_{\textrm{KL}}$, without rounding. Cite :eqref:`eq_mdl-ce_decomp` (correctness of a reference; confident).
- **Exercise 9(d)** uses $\sigma$ for a stay probability; every other use of $\sigma$ in the chapter is a standard deviation.
- **Exercise 1** carries the title "Perplexity across tokenizers" and is cited in the text for the claim that perplexity does not depend on the log base, which is only its first sentence. Fine, but if COD-S3 is adopted this exercise becomes the place where the tokenizer point is made, and it does that well (floors of 16 and about 2.8 for the two tokenizers).
- **Order.** As in the previous section, the exercises do not follow the text (perplexity first, Huffman second). Reorder.
- **To add:** one problem that plays the questions game as a code (design the questions for the pair $P$, compare with Huffman), closing COD-S2.

## Still open from October

The October review covered an older arrangement in which this material was part of one long first section; the split into its own section, the coding interpretation next to the definitions, and the worked dyadic example have all been done. Still open: joint and conditional entropy are defined after their first use (COD-S1). October proposed cutting MDL. I disagree for the present text: the order-3 against order-4 result needs it, and the coin example is short. Trim it as in COD-10.

## For the chapter review

- This section delivers the promise of the first ("we called the loss a code length without building the code") and is the chapter's strongest. It should be the model for the revision of the others: one measured hook, one source reused throughout, every bound followed by its number and its limits.
- It assumes from the first section: the decomposition and Gibbs (restated in its opening), the running pair, the dyadic source. It also assumes three things the first section does not supply: "surprisal", joint and conditional entropy, and the equivalence of questions and codewords.
- It sets up: the code-length reading used by the divergences section's closing contrast (codes need evaluable probabilities; samples do not give them), conditional entropy for the mutual-information section (which then defines it formally, too late), and rate for rate-distortion.
- Overlap with the main text is confined to perplexity and bits per byte (`language-model.md`), where the appendix is not deeper except for the floor. Everything else here is new to the reader and is named with a use: file size of a language model, model selection, the meaning of a perplexity.
- Not here and not in Further Reading: bits-back coding and asymmetric numeral systems, the two ideas that connect this section to latent-variable models and to the coders used in learned compression. A paragraph in the Further Reading notebook would be enough.
- Patterns shared with the first section: "We can now" closings that duplicate the summary, a question before every cell, exercises out of text order, and the binary entropy in three notations.
