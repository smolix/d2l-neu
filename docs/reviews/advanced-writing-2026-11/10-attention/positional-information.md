# Positional Information

Source: `chapter_attention/positional-information.md`. About 3,300 words of prose, 1,550 words of exercises, 13 code cells per framework (PyTorch and JAX only; no TensorFlow or MXNet material).

## Verdict

A good section, and in its first half close to the bar. It opens on a problem the reader feels ("dog bites man"), proves the symmetry, and then derives RoPE from a requirement instead of announcing it, which is the best single piece of exposition in my three sections. The second half is an experiment that does show its phenomenon (three schemes break at four times the training length, two do not, and rescaling the slow RoPE pairs repairs the break without training). The most important thing to fix is how that experiment is read. The in-range ranking is confounded (the sinusoidal arm also rescales the token embeddings by $\sqrt{d}$, which with a tied output head changes far more than position), the models sit close to what I believe is bigram level, and the caveat that decides the interpretation of the two flat curves is handed to an exercise. The section should say plainly what the table can and cannot support.

## Scores

| dimension | score |
|:--|:--|
| Clarity of explanation | 4 |
| Clarity of thought and focus | 3 |
| Pedagogy | 4 |
| Language | 4 |
| Ease and enjoyment | 4 |

## What works

- The opening (lines 4 to 11). Three tokens, two sentences, one consequence: "A classifier that pools these outputs cannot tell who bit whom." No machinery before the problem. Keep verbatim.
- The RoPE derivation under "Rotary Position Embeddings". Stating the goal $\mathbf{R}_i^\top \mathbf{R}_j = \mathbf{R}_{j-i}$, then reading off orthogonality and the group law in four lines, makes the rotation the answer to a question. This is the Illustrated-Transformer move done with algebra. The figure `mdl-attention-rope-rotation.svg` is clear and its caption carries the argument.
- "To such a table, position 500 is not "a bit beyond position 128". Its row carries no information at all." A sentence with voice that states the limitation exactly.
- The two caveats after the proposition (depth does not help; the causal mask breaks the symmetry) are placed where the reader would raise them, and the second sets up NoPE.
- The fast and slow pair analysis in "Extending a Trained Model's Context": "six of the 16 pairs are fast in this sense, with periods from 6 to 112 positions" (I recomputed: correct) followed by an experiment whose outcome is not foregone. Uniform rescaling fails, slow-pair rescaling works, and the prose says why. This is an experiment that teaches beyond the derivation.
- The ALiBi worked number: "a key 40 positions back already loses a factor $e^{-10}$ of its weight". One number makes the recency prior concrete.

## Structural findings

1. **Order of the absolute schemes.** The sequence is Sinusoidal Encodings, Learned Positions, Sinusoidal Encodings as Rotations. The reader leaves sinusoids, meets learned tables, and returns. "Sinusoidal Encodings as Rotations" is the bridge to RoPE, and its last sentence hands over to it. Move "Learned Positions" before "Sinusoidal Encodings" (learned first is also the simpler idea: one free vector per position), so that sinusoid definition, the rotation property and RoPE run without interruption.
2. **Two schemes are introduced inside the experiment.** ALiBi and NoPE appear under "Extrapolation Beyond the Training Length", after the section has already built its other schemes under their own `##` headings. That is defensible (they were designed for extrapolation) but the heading "ALiBi and NoPE" then sits as a sibling of "The Experiment". Either promote it to a `##` "Schemes Without Position Vectors" before the extrapolation heading, or keep it and say in one clause why these two come here.
3. **The reading of the experiment is split between text, an exercise and the further-reading notebook.** The sentence that matters most for a practitioner, that a flat perplexity curve does not mean the context is used, is stated and then delegated ("as Exercise ... measures"). `further-reading.md` then explains it with citations. The main text should carry the measurement itself in one short cell or one printed number (attention mass beyond offset 127 for the ALiBi heads), since the Summary's claim "ALiBi and NoPE stay flat" is otherwise read as praise.
4. **Code weight.** `TinyCharLM` is defined here because two later sections reuse it. With both tabs it is about 180 lines before the experiment starts, and the reader of this section needs only `embed` and `attend`. Consider introducing the class with one sentence that says which three methods matter here and that `stream` and `attention_weights` are for :numref:`sec_what-attention-computes`. The prose already half does this; make it explicit so the reader skims the rest.
5. **Length.** Proportionate. Nothing needs splitting. Candidates to trim: the binary-counting print cell (see POS-4) and the paragraph on `rope_tables` arguments, which documents an API before the reader needs it.

## Passage-level findings

**POS-1** (minor, language). Under the opening. Quote: "Linear biases (ALiBi) penalize distant keys, and a causal model can even". The second paragraph names five schemes and three acronyms before any of them is needed. The reader cannot use the names yet. Fix: keep the first and last two sentences of the paragraph ("There are several ways to do so." and the route), and reduce the list to "absolute position vectors, rotations of queries and keys, and a distance penalty on the scores".

**POS-2** (minor, explanation). Under "Permutation Equivariance". Quote: "Thus the layer does not use a". The proposition is about a single head with $\sqrt{d}$ scaling, and the proof's last sentence extends it to multi-head attention in one clause. That is fine. But the sentence after the proof says "each token's output is unchanged except for its row index", which is the whole content and could come before the proof as the statement in words (guide 8.1: say what the equation expresses first).

**POS-3** (minor, pedagogy). Under "Permutation Equivariance". Quote: "The two outputs differ only by float32 round-off: the shuffled input". This cell confirms the proposition just proved. One such check per section is within the author's allowance, and this one also exercises `d2l.MultiHeadAttention`. Keep it, but note that the JAX output is exactly `0.00e+00`, so "differ only by float32 round-off" is not what that tab shows. Write "agree to float32 round-off or exactly".

**POS-4** (minor, pedagogy). Under "Sinusoidal Encodings". Quote: "This is a continuous version of". The analogy with binary counting is good and the October review asked to keep it. The cell that prints 0 to 7 in binary computes nothing the reader could not read from a three-column inline table, and it interrupts between the line plot and the heatmap that show the same structure. Replace the cell by the table in prose, or by one sentence ("the lowest bit of 0, 1, 2, ... flips every step, the next every two, the next every four").

**POS-5** (moderate, explanation). Under "Sinusoidal Encodings as Rotations". Quote: "expands into four terms, and only the position--position term retains the". This is the reason additive sinusoids are weaker than RoPE, and it is asserted in one sentence. The four terms are never written, yet the shift test below ("where the four-term expansion leaves absolute position in the score") and an exercise both rely on them. Write the expansion once as a displayed line, $(\mathbf{x}_i + \mathbf{p}_i)^\top \mathbf{A} (\mathbf{x}_j + \mathbf{p}_j)$ into content-content, content-position, position-content and position-position, and say which depend on $i$ and $j$ separately. It costs three lines and makes the RoPE motivation exact.

**POS-6** (minor, explanation). Under "Rotary Position Embeddings". Quote: "\overset{!}{=}". The exclamation mark over the equals sign ("is required to equal") is a German convention that many readers have not seen. Either say "we require" in words before the equation and use a plain equals sign, or gloss the symbol.

**POS-7** (minor, correctness, sure). Under "Rotary Position Embeddings". Quote: "(position 0 is left as it is)". $\mathbf{R}_0 = \mathbf{I}$ is presented as a parenthetical convention, but the argument needs it and it follows from the requirement: $i = j = 0$ gives $\mathbf{R}_0^\top \mathbf{R}_0 = \mathbf{R}_0$, so $\mathbf{R}_0$ is a symmetric projection, and it is the identity as soon as we ask that no query direction be discarded. Say either "we take $\mathbf{R}_0 = \mathbf{I}$" as an explicit choice or give the one-line reason. Also "planar rotations are the simplest such group" could add the fact that makes it less arbitrary: every such family is $\mathbf{R}_i = \mathbf{R}_1^i$ for one orthogonal matrix, and an orthogonal matrix is, in a suitable basis, a set of planar rotations (and signs). Then RoPE is not one choice among many but the general solution with the frequencies left to choose.

**POS-8** (moderate, language and explanation). Under "Rotary Position Embeddings". Quote: "where $d$ is now the width of one attention head". The symbol $d$ changes meaning mid-section: model width in :eqref:`eq_pos-selfattn` and in the sinusoidal table, head width from here on, and again unspecified in the ALiBi score. The next two sections write $d_h$ for the head width. Use $d_h$ here and in :eqref:`eq_alibi-def`, and in the exercises that say "for the head width 32".

**POS-9** (minor, explanation). Under "Rotary Position Embeddings". Quote: "optional arguments change the angles themselves: `base` replaces the". The paragraph documents `offset`, `base`, `scale` and `tables` before any of them is used, and ends with a forward reference to cached generation in the next chapter. The reader needs only `offset` now. Move `base` and `scale` to "Extending a Trained Model's Context", where they are used, and put the `tables` remark in the docstring.

**POS-10** (minor, language). Under "ALiBi and NoPE". Quote: "encodes no positions at all". It does encode relative position, in the bias. Write "adds no position vectors and rotates nothing".

**POS-11** (moderate, correctness, fairly sure). Under "An Attention-Only Language Model". Quote: "The positional scheme is the one constructor argument that the experiment". The sinusoidal arm differs in a second way: `embed` multiplies the token embeddings by $\sqrt{d} \approx 11$. Because the output head is tied to the embedding, this changes the scale of the residual stream and of the logits by an order of magnitude relative to the other four arms, which is a large change to the optimization problem at 3,000 steps. The stored results fit that reading: sinusoidal reaches a training loss near 1.9 against 2.3 for the learned table (both tabs), a much larger gap than position schemes usually produce. The sentence "the sinusoidal table and RoPE give the lowest" therefore ranks something other than positional schemes. Fix: either apply the same embedding scale in every arm, or keep the code and remove the in-range ranking from the prose and the Summary, saying that the in-range column only establishes that every model trained.

**POS-12** (moderate, pedagogy; experiment scale). Under "The Experiment". Quote: "At the training length, every explicit scheme gives a lower perplexity than". The perplexities are 6.7 to 11.7 per byte. I believe a byte bigram model of English text reaches roughly 10 to 12 (unverified on this corpus; no cell prints a baseline). If so, the learned, ALiBi and NoPE models are at or barely below bigram level after 3,000 steps, and a model that uses one byte of context will of course be flat at any context length. The reader has no baseline to judge this. Fix: print the perplexity of a zero-block model (or a bigram count model) next to the table, and say what it implies for the flat rows. This also sharpens the claim in the Summary ("all explicit schemes beat NoPE in range"), which currently rests on single runs at a short budget, and which a reader may take as a general ranking. Published work finds causal models without positions competitive (cited in the further-reading notebook), so the Summary should not state the opposite as a result.

**POS-13** (minor, code). Under "The Experiment". Quote: "All three read the same 1.1 million bytes of the". No cell prints `num_bytes`. Print it or drop the number.

**POS-14** (moderate, thought). Under "The Experiment". Quote: "A flat curve, however,". This is the most important sentence of the experiment for anyone who will choose a scheme, and it gets three lines and an exercise reference. Meanwhile the paragraph before it explains ALiBi's flatness by a different cause ("because its bias depends only on distance"). The two explanations compete. Fix: state one account. ALiBi is defined at every distance, so nothing breaks; and its steep heads give keys beyond the training window negligible weight, so nothing new is read either. Then show the second half with a number (see structural finding 3).

**POS-15** (minor, correctness of reading). Under "Extending a Trained Model's Context". Quote: "below that of the unscaled model, and close to its level at the training". Stored: 9.8 (PyTorch) and 9.1 (JAX) at context 512 after slow-pair rescaling, against 7.2 and 7.4 unscaled at 128. That is about a third and a quarter higher, which is "much of the way back", not "close". The paragraph corrects itself five lines later ("neither rescaled model reaches the perplexity"). Say it once: the rescaling recovers most of the loss, at a cost of about one perplexity point at the training length itself (8.4 against 7.2), which the prose does not mention and which is the practical reason the published recipes fine-tune.

**POS-16** (minor, language). Under "Extending a Trained Model's Context". Quote: "pairs too, blurs the distinctions between neighboring bytes that they were". The sentence does not parse ("It compresses the fast pairs too, blurs ..."). Write "It compresses the fast pairs too, and so blurs the distinctions between neighboring bytes that they were trained to make."

**POS-17** (minor, correctness, unverified details). Under "Extending a Trained Model's Context". Claims about papers that match what I know: position interpolation extended LLaMA from 2,048 to 32,768 tokens with about 1,000 fine-tuning steps; YaRN interpolates low frequencies only, with a ramp and a temperature; the NTK-aware base $b' = b\,s^{d/(d-2)}$ (I checked the algebra: fastest pair unchanged, slowest divided by $s$); Llama 2 Long raised the base to 500,000. One wording issue: "NTK-aware scaling, which the YaRN paper also describes" reads as if YaRN introduced it; it predates YaRN as a community method that the paper formalizes. "RoPE is the positional scheme of most open-weights language models released since 2023" is true to my knowledge. The ALiBi slope rule $2^{-8h/H}$ matches the paper for $H = 8$.

## Code and experiments

- `positional-information-attention-ignores-order`: confirms the proposition (allowed once; see POS-3 for the JAX wording).
- `positional-information-sinusoidal-encodings-2`: prints binary numbers; computes nothing (POS-4).
- `positional-information-rotary-position-embeddings-2`: a check that can fail (an implementation error in `rope` would show) and that carries a contrast (additive score moves by 2.6 to 3.3). Good. The prose "several orders of magnitude below the scores themselves" matches the stored $10^{-6}$ to $2\times10^{-5}$.
- `positional-information-the-experiment-1` and `-3`: outputs match the prose in both tabs. The learned table degrades about threefold in PyTorch and twofold in JAX; "by up to several-fold" covers both. In-range ranking is confounded (POS-11) and uncalibrated (POS-12).
- `positional-information-extending-a-trained-model-s-context`: outputs match the prose in both tabs (uniform rescaling about 35 and 42 at every context; slow pairs 8.4 to 9.8 and 8.1 to 9.1). This is the strongest cell of the section.
- The JAX tab note about `d2l.Embed` is plumbing (guide section 9) and belongs in a code comment.

## Exercises

Eight problems, all with real content, several excellent (5, biases that keep the symmetry; 6, offset preference in RoPE, which explains how a head attends to "the previous token"; 8, the fine-tune that completes the recipe). At 1,550 words they are almost half the length of the prose, well above the 30% guide. Suggestions: cut part (c) of Exercise 2 (the diagonal standard-deviation statistic is busywork relative to its insight); Exercise 4(a) to (c) carries the main reading of the experiment and should lose its key measurement to the text (structural finding 3); Exercise 1 parts (c) and (d) could be one part. Exercise 3 depends on the four-term expansion that the text never writes (POS-5); once the text writes it, part (a) goes.

## Still open from October

The October review singled out the binary position bits as something to keep. The analogy is intact; only the print cell is questioned here (POS-4).

## For the chapter review

- Assumes `d2l.MultiHeadAttention`, `d2l.masked_softmax` and `d2l.causal_mask` from the two preceding sections, and `d2l.WikiText103`, `d2l.train_steps`, `d2l.lm_perplexity` from :numref:`sec_adam` in chapter 9. Those are reasonable dependencies, but the chapter 9 helpers are used without a word on what a "step" or the dataset is; one clause each would do.
- Sets up `TinyCharLM` for the circuits section and `rope` for chapter 11. The forward reference to cached generation (`subsec_kv-cache-cached-forward`) is not load-bearing.
- Overlap with `further-reading.md` ("RoPE in Trained Models", "Per-Pair Rescaling", "Length Generalization of ALiBi and NoPE") is well divided, except that the explanation of why the flat curves are flat lives there and in an exercise instead of in the section.
- Chapter-wide pattern to check: $d$ against $d_h$ (POS-8). This section and the next both let $d$ mean the head width in places; the circuits section uses $d_h$ consistently.
- Chapter-wide pattern to check: single-run comparisons at 3,000 steps read as rankings.
