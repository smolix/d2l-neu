# Muon

Source: `chapter_optimization/muon.md` (lines 1 to 1317; about 6,100 words of body prose, 1,100 of exercises, 17 code cells per framework). In October the section had about 3,800 words.

## Verdict

The first third of this section is the best explanation of Muon I have read: steepest descent under three balls, the spectral norm as the change a step can make to a layer's output, a one-paragraph proof, a table and a figure. If the section stopped after the comparison on the language model it would be a 4 to 5 throughout. It does not stop. It goes on to optimized polynomials, a second comparison, library versions, Shampoo, a per-neuron variant with two cells, distributed communication, a low-rank variant and a clipping rule, and by the end one optimizer of 2024 has more words than SGD, more than momentum and gradient descent together, and nearly as many as Adam and AdamW combined. The weight is not proportionate, and the added material is mostly retold from papers whose results the section itself calls small. There is also a flaw in the core: Newton–Schulz is demonstrated on a Gaussian matrix, the easy case, and the reader learns 700 lines later, inside the NorMuon subsection, that on a real gradient five steps leave most singular values far from 1. The most important fix is to test the iteration on a real gradient where it is introduced, say once what Muon's update actually is, and cut the variants to a page.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought and focus | 3 |
| Pedagogy | 3 |
| Language | 4 |
| Ease and enjoyment | 3 |

## What works

- **"Norms and Steepest Descent"** (lines 59 to 175). One question ("which norm defines the ball"), three answers, each tied to an optimizer the reader knows. The remark that SGD solves the penalized problem while the ball "fixes the direction" prevents a real confusion. The sentence "The problem is not posed until we say which norm defines the ball" is the one a reader will quote.
- **The spectral norm is motivated before it is used**: "the largest change it can cause in the layer's output", contrasted with a Frobenius norm that "can call an update large even when its effect on every activation is small".
- **The cost count** (lines 308 to 321) derives $5m/B$, then applies the same count to the reader's own model and reports the unflattering 12%. I recomputed both (0.73% and 12.2%).
- **The RMS-matching factor** is derived with a worked shape ($512 \times 128$) and the alternatives are named in three lines.
- **Candour about evidence**: "A model of 0.47 million parameters demonstrates the mechanism but cannot rank optimizers", "a real gain for Muon is measured in tens of percent, not in multiples", and, on the CNN, "Optimizing faster and predicting better are different claims". Protect these.
- The figure of the three balls and the figure of the polynomial maps both render cleanly and carry the argument. My recomputation of the maps agrees with the caption (quintic: inputs above 0.01 land in 0.68 to 1.13; Polar Express: 0.85 to 1.12 over the whole range; cubic: 0.94 from 0.2).
- All numerical claims about the iterations that I could recompute are right (0.01 passes 0.5 in ten cubic steps and four quintic steps; $p(1) = 0.701$; the Polar Express table matches the published coefficients divided by powers of 1.01).

## Structural findings

1. **Length and proportion.** Approximate prose words: norms 900; Newton–Schulz with the tuned quintic 850; optimized polynomials 750; update and parameter groups 800; language-model comparison 500; CNN 300; library versions 330; Shampoo 250; per-neuron normalization 1,150; scale 520. About 3,000 words are variants and relatives. The Further Reading notebook of the chapter already has a Muon part (Kovalev, Gram Newton–Schulz, Turbo-Muon, Dion3, NorMuon scaling), so there is a home for them. Proposed section of about 3,800 words: norms; Newton–Schulz tested on a real gradient; the update with weight decay shown; parameter groups; the language-model comparison read for what it shows; one paragraph on scale and evidence; Shampoo in a paragraph. Move out: the Polar Express construction (to the appendix subsection `subsec_mdl-orthogonalized-updates`, which analyses the cubic and is the natural owner of a minimax derivation, or to Further Reading), NorMuon and AdaMuon (Further Reading, with the exercise kept), the communication accounting and Dion (Further Reading or the Computational Performance chapter). If the author prefers to keep the material, the seam for a split is before "Relatives and Variants", with "Optimized Polynomials" moved behind it; but a second section on Muon variants would be a survey, and I recommend the cut.
2. **The demonstration uses the wrong matrix** (MUON-1). This is the main defect in the teaching.
3. **The idea is never challenged.** The section says a gradient is dominated by a few directions and that equalizing them is the best step under the spectral norm. A reader will object that the small directions of a minibatch gradient may be mostly noise. The section later reports that leaving small singular values unconverged "costs little". Both statements stand without being reconciled (MUON-2).
4. **Dependencies.** The section needs the anatomy of a transformer (`qkv`, `proj`, LayerNorm, residual stream, attention heads, logits) from chapters 10 and 11, and sharding vocabulary from chapter 13. The model was introduced in the Adam section with the promise that "an optimizer does not need to know what attention does". The late subsections break that promise (MUON-11, MUON-12).
5. **Duplication with the appendix** (decision 3). The odd-polynomial identity and the memoryless-Shampoo identity appear here and in the appendix. Both are three-line results the main text needs, so the duplication is allowed. The appendix analyses the cubic (basin, step count); this section treats the tuned quintic. No long derivation is written out twice, so the earlier finding does not apply to the present text. What is missing is the link: the section never cites `subsec_mdl-orthogonalized-updates`, and cites `subsec_mdl-preconditioning-ladder` twice (MUON-15).

## Passage-level findings

**MUON-1** (major, pedagogy/thought). Under "The Tuned Quintic" and "Per-Neuron Normalization".

> band. A random Gaussian matrix has no such singular values, but we will see

> lift them into the band around $1$. On real gradients, then,

The cell tests `newton_schulz` on a $96 \times 64$ Gaussian matrix, whose normalized singular values lie between about 0.03 and 0.22, and concludes "That is $\mathbf{U}\mathbf{V}^\top$ for an optimizer's purposes". The stored output of `muon-per-neuron-normalization-1` then shows the opposite for the matrices Muon is applied to: for `qkv` and `proj` the largest singular value carries 0.99 of the Frobenius norm, most others start below 0.01, and after five steps the row norms are 0.2 to 0.9 where exact orthogonalization would give 0.4 to 1.0. So the update the reader has implemented is not close to $\mathbf{U}\mathbf{V}^\top$, the RMS-matching argument is off (MUON-6), and the Summary's "Muon computes it without an SVD" is not what happens. Fix: compute one `TinyLM` gradient in the Newton–Schulz subsection and plot its spectrum before and after five steps next to the Gaussian one (the measurement cell of the later subsection can be reduced to this). Then say once what five steps of the quintic do: a singular value above about 1% of the Frobenius norm lands in the band 0.68 to 1.13; a smaller one is multiplied by up to $3.4445^5 \approx 485$ and stays below. Muon in practice compresses the spectrum by orders of magnitude and flattens its top. The gradient at initialization is an extreme case; exercise 10 asks for the spectrum of the momentum blend during training, and the text should show that measurement, not leave the central empirical premise to the reader.

**MUON-2** (moderate, explanation). Under "Matrices and the Spectral Norm" and "Optimized Polynomials".

> nonzero singular value by one, so every direction that the gradient

> one barely affect training, so leaving them unconverged costs little.

The first passage says the point of Muon is that every direction moves at the same rate. The second says the directions with small singular values barely matter. A reader who holds both will ask which is true, and why moving at full rate along directions that a minibatch barely identifies is a good idea at all. Raise the objection where the step is derived and answer it: the input to the iteration is the momentum average, not one gradient; the spectral ball bounds the worst-case change of the layer's output whatever the direction; and the published evidence is that flattening the upper part of the spectrum is what helps, while the tail is immaterial. Two or three sentences, placed after :eqref:`eq_muon-spectral-step`.

**MUON-3** (moderate, correctness, confident about the mathematics). Under "Matrices and the Spectral Norm".

> output head and the one-dimensional vectors (biases and normalization

"do not transform activations through a dense matrix product either" is false for the head: it is a dense $128 \to 256$ linear layer applied to the final hidden state, exactly the case for which the spectral norm was motivated two paragraphs earlier. Keeping the head on AdamW is an empirical choice of the Muon authors. From memory, the original post says so and notes that it is not what the norm-based theory predicts; unverified. Rewrite: "The output head is a dense matrix, and the argument above would give it to Muon. In practice it trains better with AdamW, and we follow that convention. The reason is not understood." This is a place for the candour the voice guide asks for.

**MUON-4** (moderate, thought). Under "Comparison on the Language Model".

> Both optimizers have their best rate inside the grid, so the comparison is

The final gap is 0.026 nats in the stored PyTorch run and 0.042 in JAX. The section itself puts the spread between runs at about 0.02 ("no more than the spread between runs", under "Library Implementations"). The grid has half-decade steps, and AdamW's loss changes by 0.06 to 0.1 between neighbouring points around its best one, so its true optimum may lie 0.02 below the grid's. At the shared rate $3\times10^{-3}$ the two optimizers differ by 0.009 (PyTorch) and 0.016 (JAX). The conclusion "the hybrid is a few hundredths of a nat ahead" at step 2,000 therefore rests on a difference of the size of the noise. What the stored runs do show, well outside the noise: (a) the hybrid is 0.15 to 0.25 nats ahead during the first 500 steps and reaches a given loss in about two thirds of the steps; (b) its loss stays within 0.05 of its best over a decade of learning rates, where AdamW's does not. Lead with these. The plot supports "the gap narrows"; drop "but does not close" or support it with repeated runs. Also: both arms run at a constant rate with no decay, one section after the reader learned that the final rate sets the final loss. Say that the protocol is the one of :numref:`sec_adam` and that a decayed run would lower both.

**MUON-5** (minor, repetition). The saving in optimizer state is stated three times (lines 595, 737 and 1170), and the 84% twice.

> does not close, and the hybrid needs only 58% of AdamW's optimizer state.

Keep the statement under "Parameter Groups". It is not a result of the comparison.

**MUON-6** (moderate, correctness). Under "The Update".

> n)}$ gives every matrix's update an entrywise RMS of $0.2\,\eta$, whatever

True for an exact orthogonalization of a full-rank matrix. With five quintic steps on a real gradient the stored row norms imply a smaller update: for `qkv` the rows have norm 0.24 to 0.57 where the exact value averages 0.58, so the RMS is roughly two thirds of $0.2\,\eta$ (my estimate from the printed range; the cell does not print the Frobenius norm for `qkv`) and differs from matrix to matrix. Write "an entrywise RMS of at most about $0.2\,\eta$, reached when all singular values arrive in the band". The NorMuon parenthesis at line 1044 already concedes the point.

**MUON-7** (moderate, technical detail). Under "Comparison on the Language Model".

> decay is off in both arms, so the only difference between them is the

The update :eqref:`eq_muon-update` and both implementations have no weight decay, and the comparison switches it off. The paper cited for the RMS convention reports (from memory, unverified) that Muon needs decoupled weight decay in long runs because the orthogonalized step has a fixed size and the weights otherwise keep growing; the Further Reading notebook says Muon "needs weight decay" as well. A reader who builds from this section gets an optimizer without it. Add the decay term to the displayed update with one sentence on why a fixed-size step makes it matter, and a `weight_decay` argument to `Muon`. Turning it off for the short comparison is then a stated choice.

**MUON-8** (minor, explanation). Under "Comparison on the Language Model".

> and stays slightly below the training loss: a run reads only a few percent

The reason given explains why the validation loss is not above the training loss. It does not explain why it is below. The likely reason is that the training curve is a smoothed average over earlier steps of a model that is still improving. Say that, or drop "slightly below".

**MUON-9** (moderate, experiments; decision 4). Under "Per-Neuron Normalization", cell `muon-per-neuron-normalization-2`.

> On the first step the running average is proportional to each neuron's

The cell divides each row by its own RMS and rescales to the old Frobenius norm, then prints that all rows have equal norm (0.445 to 0.445) and that the Frobenius norm is unchanged (10.07 before and after). Both follow from the two lines of code above the print. The check cannot fail and teaches nothing the formula did not. Remove the cell; if NorMuon stays, the test that teaches is the one in exercise 6.

**MUON-10** (moderate, focus; correctness unverified). Under "Per-Neuron Normalization".

> How much does this help? The NorMuon paper reports that plain Muon needs 6%

A paragraph of results from a paper and a benchmark, for a method the section does not run, ending in "its gain is small and depends on how carefully the baseline is tuned". This is a retelling, and it is the clearest candidate for the Further Reading notebook. On the numbers: what I recall from the paper's abstract is a gain of about 11% over Muon and 22% over Adam for a 1.1B-parameter model; I cannot confirm "6% to 15%" on the 124M and 350M models, nor the statement about a shared learning rate, nor the modded-NanoGPT "optimization track" result. Check all three against the sources.

**MUON-11** (moderate, focus/dependency). Under "Per-Neuron Normalization".

> every path from the loss to the residual stream passes through a LayerNorm.

Fifteen lines explain why the gradients have rank 127, why `proj` has about 120, and why the row norms of `mlp2` equal $\sqrt{127/128}$. The argument is correct and rather pretty, but it serves a printed column the subsection does not need, and it requires the reader to know the residual stream and pre-norm blocks, which chapter 11 teaches. Reduce to one sentence ("LayerNorm removes the mean, which costs every gradient one direction; we therefore drop zero singular values before forming $\mathbf{U}\mathbf{V}^\top$") or turn it into an exercise for chapter 11.

**MUON-12** (moderate, dependency). Under "Muon at Scale".

> reduce-scatter the gradient and 4 to all-gather the updated parameters),

"Sharded" is defined; "reduce-scatter", "all-gather" and "data-parallel group" are not, and the byte accounting cannot be followed without them. The QK-clip paragraph uses "attention logits", "heads", "query and key projections" before chapter 10. What the reader needs here is three sentences: Newton–Schulz couples all rows of a matrix, so a matrix split across devices must be gathered, which adds a bounded communication cost; a low-rank variant (Dion) avoids the gather; a trillion-parameter model was pretrained this way with one added stabilizer. The details I could check from memory are right (Kimi K2: one trillion parameters, 15.5 trillion tokens, threshold 100, the square-root split between query and key, no loss spike). The byte counts, the per-device repetition in Moonlight, the NorMuon assignment of matrices to devices and Dion's "a quarter to a half of the dimension" are unverified.

**MUON-13** (minor, correctness). Under "Optimized Polynomials".

> the round-off floor of single precision, near $10^{-6}$. The plot also shows

In exact arithmetic the seven-step distance on `G` is $2.5\times10^{-6}$ (recomputed in double precision), so the value near $10^{-6}$ at step seven is where the iteration is, not the round-off floor; the floor shows from step eight. Also: `newton_schulz` divides `H` by its Frobenius norm, so its singular values run from 0.44 down to $4.4\times10^{-4}$, below the design interval $[10^{-3}, 1]$, and "leads at every step" is a tie for the first two steps. The claim that more than six iterations or an exact SVD "do not lower the final loss" is from the paper and unverified.

**MUON-14** (minor, correctness). Under "The Tuned Quintic".

> One iteration lifts the whole spectrum, and three compress it to within a

The stored plot shows the curves for three and five iterations on top of each other: after three steps the Gaussian spectrum is already in 0.68 to 1.20. The sentence suggests five are needed for this matrix. They are needed for smaller singular values, which is the point of MUON-1.

**MUON-15** (minor, references). Under "Matrices and the Spectral Norm".

> an exercise. :numref:`subsec_mdl-preconditioning-ladder` places this result

The pointer is correct (the appendix section "Structured Preconditioners" holds the ladder and the Shampoo identity) but it interrupts the proof and is repeated at line 925. Keep the second. Add, after the cubic is mentioned, a pointer to `subsec_mdl-orthogonalized-updates` for the polar decomposition and the cubic's convergence count; the appendix already points back here.

**MUON-16** (minor, explanation). Under "Matrices and the Spectral Norm".

> The proof is one line of duality. For any $\mathbf{A}$ with

The step "no entry of a matrix with unit spectral norm exceeds one" is applied to $\mathbf{U}^\top\mathbf{A}\mathbf{V}$, and the reader must supply that this product also has spectral norm at most one. Say so. And name what was obtained: the best decrease is $\eta\sum_i\sigma_i$, the nuclear norm of the gradient, as it is $\eta\|\mathbf{g}\|_2$ and $\eta\|\mathbf{g}\|_1$ for the other two balls. A fourth column "decrease of the linearization" in :numref:`tab_muon_norms` would complete the parallel and give "duality" a meaning.

**MUON-17** (minor, focus; unverified). Under "Comparison on the Language Model".

> significant speedup below 0.5B parameters for Sophia, a second-order method

The Sophia sentence is an aside about another optimizer in another paper. The numbers for Muon in the same study ($1.4\times$ falling to $1.1\times$, ten optimizers, four sizes) agree with my memory of the abstract. I cannot confirm the Sophia detail. Cut it; the paragraph's conclusion stands without it.

**MUON-18** (minor, experiments). Cell `muon-the-same-race-on-a-cnn-3`.

> best_adamw_cnn = min(adamw_cnn, key=lambda lr: d2l.final_loss(adamw_cnn[lr]))

The language-model comparison selects the learning rate by validation loss; the CNN comparison selects it by training loss and then reports test accuracy. Select by the reported metric in both. The conclusion ("within a point") survives either way in both stores.

**MUON-19** (minor, language, pattern; 6 instances). Paragraphs that open with a question to be answered at once: "Which polynomial should we iterate?", "How close to orthogonal is the result?", "How expensive are fifteen multiplications next to an elementwise Adam", "Does a more accurate polar factor train better?", "How much does this help?", and the opening "What does it mean for a direction to be "steepest"?". The first and last earn their place. The others are the question-then-cell formula the author asked to remove.

**MUON-20** (minor, language). "hidden matrix" is used at line 104 and defined by implication at line 157; define it at first use ("a weight matrix between two hidden layers"). Ornament to remove: "speaks for the optimizer" (line 649), "a matrix in disguise" (line 479), "is out of the question" (line 183). Cell identifiers and exercise 6 still say "race".

## Code and experiments

- `muon-orthogonalization-by-newton-schulz-2`: wrong test matrix (MUON-1, MUON-14).
- `muon-optimized-polynomials`: correct and readable, but it illustrates a refinement whose effect on training the text calls slight. Goes with the Polar Express text wherever that goes.
- `muon-the-race-on-the-language-model-3`: prints validation loss at steps 500 and 2,000 only. The prose also quotes "1.6 after 1,000 to 1,250 steps" and "after about 1,500 of the 2,000 steps", which are read off the plot (and agree with it). Print the step at which each run first passes 1.6 if the sentence stays.
- `muon-the-same-race-on-a-cnn-*`: prose matches both stores ("several times lower": 0.028 against 0.129 and 0.025 against 0.111; accuracies within half a point).
- `muon-library-implementations`: "within about 0.02" is generous to the store (0.000 and 0.001 apart). The cell confirms that two implementations of one formula agree, which is worth one sentence and no subsection; the useful content is the list of defaults that differ (weight decay 0.1, 2-D only, the EMA form of momentum).
- `muon-per-neuron-normalization-1`: prose matches both stores. The PyTorch store prints rank 126 for `qkv`, covered by "at most 127".
- `muon-per-neuron-normalization-2`: cannot fail (MUON-9).
- JAX: the tab note on TF32 repeats a code comment; the guide puts such notes in comments only.

## Exercises

Ten problems, most of them good and several better than the text they follow (2(b) measures whether 0.2 is AdamW's update RMS; 3 asks where quality saturates; 8 derives $\sigma/\sqrt{\sigma^2+\epsilon}$ for damped Shampoo; 10 tests the section's premise).

- **Exercise 5 is flawed** (moderate, correctness).

  > 1. Let $\mathbf{M}$ be a matrix with fewer rows than columns and full

  Part (a) proves unit row norms for a wide matrix of full row rank and asks what this does to a rare token's row of `token_emb`. But `token_emb` is $256 \times 128$: it has more rows than columns, cannot have full row rank, and a byte value absent from the minibatch gives a zero row. The result of (a) does not apply to the matrix of (b). Restate (a) for the tall case (squared row norms sum to 128, and a row's share is its leverage, not its frequency) or pose it for a vocabulary smaller than the width.
- Exercise 9 (Adam with $\beta_1 = 0$) is correct (Jensen gives the bound, the largest step is $\eta/\sqrt{1-\beta_2}$) but belongs to :numref:`sec_adam`.
- Exercise 10 should become a measurement in the text (MUON-1).
- If the variants are cut, exercises 4 and 6 need the definitions moved into the exercise statements or go with the material.

## Still open from October

The chapter-level finding that `TinyLM` is built from chapter 11 material and that later sections depend on it is still open, and this section now leans on transformer internals more than it did (MUON-11, MUON-12). The section has grown from about 3,800 to about 6,100 body words since October.

## For the chapter review

- Proportion: at 7,200 words in all, this is the second-longest section of the chapter, behind only Adam and AdamW together. Muon deserves a main-text section (it is in PyTorch and Optax, and the norm view reorganizes the whole chapter), but one of about 4,000 words.
- The norm table (SGD, Adam, Muon as three balls) is a unifying idea that would serve the chapter introduction or the close of the Adam section; at present it arrives two sections after Adam.
- Order: Muon answers "which direction", as Adam does. It would follow Adam more naturally than it follows Schedules, and its closing sentence about batch size would then pass to Schedules instead.
- Appendix: the split of labour is right in principle (appendix: polar factor, odd polynomials, the cubic; main text: the optimizer). The main text should cite `subsec_mdl-orthogonalized-updates`, and that subsection is the natural home for the Polar Express construction.
- Suspected chapter pattern: comparisons at a constant learning rate on a half-decade grid, with final differences of a few hundredths of a nat reported as wins. The early-training gap and the width of the usable range are the robust observations here and probably in the Adam section too.
- Suspected part pattern: sections rebuilt in 2026 end with a tour of that year's papers (results quoted, not reproduced). These tours date fastest and belong in Further Reading.
