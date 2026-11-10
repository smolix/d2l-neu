# Numerical Stability and Conditioning

Source: `chapter_mdl-optimization/mdl-numerical-stability-conditioning.md` (read to the slides marker, line 1679). About 7,850 words of prose before the exercises (Floating-Point Arithmetic 1,490; Stable Softmax and Cross-Entropy 1,220; Catastrophic Cancellation 1,900; Conditioning 2,750; Summary 180) and 2,280 words of exercises (12 problems). Stored PyTorch outputs checked for every cell; four figures rendered; the worked numbers recomputed.

## Verdict

This is one of the sections a reader could find in a good numerical-analysis text and not be disappointed, and it is close to the bar. It opens on two failures the reader can feel, states one organizing principle ("an unstable algorithm routes a well-conditioned problem through an ill-conditioned intermediate step"), and then does what the Basics review asked for and rarely got: nearly every cell is preceded by a prediction from a short argument and followed by a number that confirms it (the 128 of the naive variance, the 0.125 increment of the float32 sum, 149 log 2 = 103.2789, the 6,908 steps). I recomputed about sixty quoted numbers and found them right, with the exceptions listed below. The most important thing to fix is the last third. The section is really two: floating point and stable formulas (4,600 words), and the conditioning of linear systems (2,750 words). The second half retells material the appendix already teaches elsewhere (the two-line system and the amplification bound of :numref:`subsec_mdl-condition-number`, the ridge filter factors of `eq_mdl-ridge-filter`, the contraction factor and the 6,908 steps of :numref:`sec_mdl-gradient-based-optimization`), and its ridge subsection drifts from "what does rounding do" to "how many steps does gradient descent take", a question another section owns. There is also one wrong explanation (finding NSC-17) and a few quoted measurements that no cell computes.

## Scores

| Dimension | Score | Note |
|---|---|---|
| Clarity of explanation | 4 | Toy format before the general one, every symbol defined, concrete numbers everywhere. Loses a point for the missing link between $\kappa_f$ and $\kappa(\mathbf A)$ and the hand-waved Welford stability argument. |
| Clarity of thought | 4 | One principle carried through three quarters of the section. The ridge tail and one wrong mechanism cost a point. |
| Pedagogy | 5 | Predict, run, compare, in almost every cell. The best use of code I have seen in this part. |
| Language | 4 | Plain, exact, almost free of the banned patterns. Mixed format names and "part" for a subsection are the recurring defects. |
| Ease and enjoyment | 4 | Sharp examples and a few quotable sentences through "Summation Order"; the Conditioning half reads as careful bookkeeping of results the reader has met. |

## What works

Preserve these in revision.

- **The opening** (lines 4 to 35). Two failures, two causes, and the distinction that organizes everything: "The first failure stops training with a `NaN` loss; the second goes unnoticed." The reader knows by line 25 that one failure is curable by rewriting and the other is not.
- **The toy format** ("What does a computer store when it stores $1.1$?"). Four values per octave, worked roundings with their relative errors, then the general format "with more bits". This is example before abstraction done properly, and the figure `mdl-opt-fp-number-line` carries it.
- **The $\log(1+x)$ diagnosis** under "The Condition Number of a Function". A question is posed ("Is $\log(1+x)$ near $x=0$ a hard function to compute, or is the formula to blame?"), a tool is built to answer it, and the answer names the guilty intermediate step with its own condition number $1/|\log y|\approx10^8$. This is the best page of the section.
- **Exact subtraction, wrong answer.** "it made no error of its own: the inputs were stored as $1.0002343655$ and $1.0002341270$, and their difference is what the cell printed." The cell prints the stored inputs so the claim can be seen. The same point returns in the log-softmax cell, where the error $1.32\times10^{-5}$ in `lse` is traced digit for digit to the sum $1.0000132$.
- **The condition number of the variance.** The text asks "How much of this does the problem itself force?", derives $2(1+|m|/s)$, and observes that the one-pass formula "*squares* the condition number, as the normal equations will". That one sentence ties the two halves of the section together better than any transition.
- **Predictions that land.** "The naive result is a whole multiple of $128$" with the reason ($2^{59-52}$); "each addition rounds to two gaps, $0.125$, 25% too much (between $2^{17}$ and $2^{19}$ it rounded to $0.09375$, too little)". I checked both against the float32 running sum; they are exact.
- **Sentences with a voice.** "a negative output is the exact variance of no data at all"; "Higher precision does not remove a cancellation; it moves the input at which the failure starts"; "A solver with a tiny backward error has solved a nearby problem exactly, and it can still be far from the answer."
- **The PyTorch tab note for cross-entropy.** One surviving significant bit turns a loss of 103 into $149\log2=103.2789$, "wrong in the first decimal place and with no warning". A memorable, checkable fact.
- **Honesty about limits**: "The last decimals vary with the LAPACK build; the trend does not"; "printed values of $\omega$ below about $u$ are noise"; the closing sentence that these tools do not cover error growing through a hundred layers.

## Structural findings

**S1. The section is two sections, and the second leans on three others.** Up to "Summation Order" the subject is rounding: what is stored, which formulas avoid unrepresentable intermediates, which subtractions expose input error. "Conditioning" then changes object (matrices), tools (SVD, norms), and at the end subject (step counts of gradient descent, shrinkage of the ridge solution). Both halves are good; together they run to 7,850 words, and a reader who arrives from :numref:`sec_numerics` for floating point is done after 4,600 words and has 2,750 more in front of the summary, while a reader sent here for backward error starts 900 lines into the file. Options, in order of preference: (a) keep one section but cut the Conditioning half to what is new here (forward and backward error, the residual proposition, the Hilbert experiment, the normal equations) and reduce the ridge subsection to a short paragraph with references, saving about 1,000 words; (b) split at "Conditioning" into "Floating-Point Arithmetic and Stable Formulas" and "Conditioning and Backward Error".

**S2. Repetition across the appendix (guide 7.7).** Three items are taught here a second time with the same numbers:
- The two-line system. :numref:`subsec_mdl-condition-number` (in `mdl-svd-low-rank.md`, lines 1085 to 1150) already gives $\mathbf K$, $\mathbf b=(2,2.0001)$, the jump to $(0,2)$, the relative change $3.5\times10^{-5}$, the singular values $2.00005$ and $5\times10^{-5}$, $\kappa\approx4\times10^4$, and the explanation that the shift has only $1/\sqrt2$ of its length along the most amplified direction. "Two Nearly Parallel Lines" repeats all of them. What is new here is the picture of two lines and the reading of $(0,2)$ as a solution with small backward error. Keep those; cite the numbers.
- The filter factors. `eq_mdl-ridge-filter` in the same linear-algebra section derives $\sigma_i^2/(\sigma_i^2+\gamma)$ from the SVD. Here it is a proposition with a proof, in different notation ($\lambda$, $\mathbf w$, an objective with factors $\tfrac12$; there $\gamma$, $\mathbf x$, no $\tfrac12$).
- The step count. :numref:`sec_mdl-gradient-based-optimization` proves the contraction $(\kappa-1)/(\kappa+1)$ and computes $6{,}908$ four times (lines 25, 405, 585, 970), and refers to "the cost law". The "Proposition (iteration bound)" here re-proves the contraction and derives the law. If the law deserves a proposition, its home is that section.

**S3. The link between the two condition numbers is never made.** The first half defines $\kappa_f(x)=|xf'(x)/f(x)|$ for a scalar function and builds the principle on it. The second half uses $\kappa(\mathbf A)=\sigma_1/\sigma_d$ and says only that it "measures" the amplification. A reader is not told that $\kappa(\mathbf A)$ is the same quantity for the map $\mathbf b\mapsto\mathbf A^{-1}\mathbf b$, taken in the worst case over $\mathbf b$ and over the direction of the perturbation. One sentence would make the section one argument instead of two (see NSC-15).

**S4. Measurements quoted in prose that no cell computes.** Three places quote measured values for which there is no cell: the finite-difference errors (NSC-11), the "exact solution of the *stored* system" (NSC-17), and "(perturbing every $x_i$ ... gives $1.6\,|m|/s$ for Gaussian data)" (this one is a calculation, $2\,\mathbb E|z|=1.596$, and is fine). The first two did not reproduce on my machine. In a section whose method is "predict, then print", unprinted measurements stand out.

**S5. Overlap with the main text: deeper, not a re-teaching.** :numref:`sec_numerics` (`chapter_builders-guide/numerics.md`) shows the same four phenomena by experiment (range against precision, a small update lost at $w=1$, log-sum-exp with the max shift, a drifting fp16 sum) and builds mixed-precision practice on them. This section derives them: the rounding model, the threshold as a proposition, the sandwich bound, the summation bounds, and everything about condition numbers, backward error, and Welford, none of which the main text has. The division is stated at the end of the opening ("here we derive why computations in them fail") and is right. Three small frictions: the main text writes `bf16`, this section `bfloat16`; the main text *adds* $10^{-3}$ to a weight of 1 (threshold $2^{-8}$ in bf16), this section *subtracts* it (threshold $2^{-9}$), so a reader comparing the two sees different thresholds with no remark; and the max shift is now explained in three places (:numref:`subsec_softmax-implementation-revisited`, :numref:`sec_numerics`, here), which is acceptable only because this one adds the bounds. :numref:`sec_numerical_stability` (`chapter_multilayer-perceptrons/numerical-stability-and-init.md`) shares only the words "numerical stability": it is about products of Jacobians and initialization. There is no overlap, and the closing sentence here points to it correctly. Consider one clause in the opening as well, since a reader following the title from chapter 4 may expect vanishing gradients.

**S6. Connection to deep learning is concrete throughout the first half** (logits, cross-entropy, weight updates in bf16, loss scaling, batch-norm variance, merging moments across devices) **and thin in the second.** The Hilbert matrix and the random designs are numerical-analysis classics. One sentence naming where a deep learning reader meets an ill-conditioned solve would help: the GPTQ Hessian (the stub in `chapter_lm-compression/gptq.md` already plans to cite `eq_mdl-opt-kappa-squared`), K-FAC damping, or kernel and GP solves.

## Passage-level findings

**NSC-1** · minor · correctness (certain) · under "Formats"
> "its upper neighbor and rounds to the even one, $1$, so the bound"

The text concludes that "$|\delta|\le u$ is attained". In the model $\mathrm{fl}(x)=x(1+\delta)$ it is not: for $x=1+u$, $\delta=-u/(1+u)$, strictly inside the bound. The toy example four paragraphs earlier already shows this ($1.125\mapsto1$ has error $-0.111$, not $-0.125$). Write "so the bound is approached to within a factor $1+u$" or "so half a gap, the worst case, does occur".

**NSC-2** · minor · explanation · under "Overflow, Underflow, and Small Updates"
> "In bfloat16 a weight equal to $1$ ignores any decrease below"

The display `eq_mdl-opt-small-update` is stated for $2^e<w<2^{e+1}$ with threshold $u\,2^e$, and "An update smaller than about $u|w|$ is lost". The worked example and the cell then use $w=1$ with a decrease, which is exactly the exceptional case of the proposition (threshold $u/2$). The numbers are right ($2^{-9}$, $2^{-12}$, $2^{-25}$), but the reader has to go back to the exception clause to see why they are half of what the display says. Either say so in a clause ("$w=1$ is the exceptional case: below a power of two the gap halves, so the threshold is $2^{-9}$, not $2^{-8}$") or use an increase, which also matches the experiment in :numref:`sec_numerics`.

**NSC-3** · minor · thought · opening
> "three times, even in float64. And `np.linalg.solve`, applied to a"

The opening promises that the softmax of $(1000,1001,1002)$ "returns `NaN` three times". No cell shows it. The softmax cell uses $(0,100)$ and prints one `nan` and one `0`; the cell on $(1000,1001,1002)$ computes log-sum-exp and prints `inf`. Either open with the example the body runs, or add one line to the softmax cell that prints the three `nan`.

**NSC-4** · minor · explanation · under "The Max Shift"
> "The shifted softmax returns $(3.8\times10^{-44}, 1)$, where the first entry"

Six lines of prose earlier the same probability is "$3.7\times10^{-44}$". Both are right (the true value is $3.72\times10^{-44}$; the float32 subnormal grid has spacing $1.4\times10^{-45}$ and the nearest point is $3.78\times10^{-44}$). Say it, since it is the first subnormal the reader sees at work: "which is $3.7\times10^{-44}$ rounded to the subnormal grid, with five significant bits left".

**NSC-5** · minor · pedagogy · tab notes under "Cross-Entropy from Logits"
> "The from-logits column reads $20$, $60$, $103$, $104$ at every gap. The"

This sentence and the next ("The probability column agrees at gaps $20$ and $60$") appear in all four tab notes. State them once in the shared prose and leave in each tab only what differs: one surviving bit in PyTorch, flush to zero elsewhere, the Keras clipping. Also, the column header and the notes call $t$ a "gap", a word the prose before the cell does not use for $t$.

**NSC-6** · minor · language · under "Bounds for Log-Sum-Exp" and elsewhere (5 instances)
> "error that was already in its input, the theme of the next part."

"Part" is used for a `##` unit of the section: "The running example of this part", "the theme of the next part", "the subject of the last part", "in the last part", and "the third subsection" for a `###` unit. In a book whose top-level divisions are called parts, and in a file that also writes "one part in $2^{p+1}$", this is ambiguous. Use the label (:numref:`subsec_mdl-catastrophic-cancellation`, :numref:`subsec_mdl-conditioning-revisited`), which also survives reordering.

**NSC-7** · moderate · language · whole section (pattern)
> "about one part in 17 million for fp32"

The same formats carry two names with no rule: "fp32" 12 times and "float32" 14 times, "fp16" 9 times, "float64" 11 times, "bfloat16" 8 times (the main text says "bf16"). Within one paragraph the softmax discussion goes from "fp32's threshold $88.72$" to "leaves float32's normal range". Pick one family for prose (the main text's fp16, bf16, fp32, fp64 would match :numref:`sec_numerics`) and keep `float32` for code identifiers only.

**NSC-8** · minor · explanation · under "The Condition Number of a Function"
> "`log1p` evaluates a series in $x$ and never forms"

Low confidence that this matters, moderate confidence that it is not how libraries do it: the common implementation (fdlibm and its descendants) does form $1+x$ and then corrects for the rounding error of that sum. The pedagogical point (the rounding of $1+x$ must not be lost) stands either way. Safer: "`log1p` is written so that the rounding error of $1+x$ never reaches the result; for tiny $x$ it returns $x$ itself."

**NSC-9** · minor · thought · the table under "The Condition Number of a Function"
> "| variance of data with mean $m$, spread $s$ | at most $2(1+\lvert m\rvert/s)$ | derived below |"

The table is introduced as "The condition numbers of a few problems met in this section", but $1-\cos x$ and $\sqrt x$ are met only in the table (the first once more in a list), and the variance has not been met yet. It works better as a small reference table if the intro says so, and better still with a column "naive formula loses" ($\log(1+x)$: everything below $x\approx u$; variance: $(m/s)^2u$), which is the comparison the section's principle is about.

**NSC-10** · moderate · thought · under "Subtraction of Nearly Equal Numbers"
> "differences, which trade one error against another. The forward difference"

The finite-difference analysis (about 150 words and a display) is hung on the last item of a list of places where cancellation occurs. It is the only example in the section where the cure is a tradeoff and not a reformulation, it is what a deep learning reader does when gradient-checking, and it gets no heading, no cell, and no figure. Give it its own short paragraph opened by its use ("A gradient check compares autograd with a finite difference; how small should $h$ be?"), and say which $h$ to use in float32 ($h^\star\approx\sqrt u\approx 5\times10^{-4}$ times the scale of $x$, three to four digits at best), which is the fact a practitioner needs.

**NSC-11** · moderate · correctness (the specific number; fairly sure) · same paragraph
> "error $1.8\times10^{-8}$. The measured error is $3\times10^{-9}$ at"

No cell measures this. On macOS arm64 with NumPy and with `math.sin` I get $1.4\times10^{-8}$ at $h=10^{-8}$ (and $4.3\times10^{-5}$ at $10^{-12}$, $4.2\times10^{-3}$ at $10^{-2}$, which match the text). At the optimum the error is dominated by rounding noise, so its value depends on the math library and on the last bit of $h$; $3\times10^{-9}$ may be right on another machine. Do not quote one digit of a noise-dominated quantity. Write "The measured error is about $10^{-8}$ at $h=10^{-8}$" or add the three-line cell and let the output speak.

**NSC-12** · moderate · explanation · under "One-Pass Variance"
> "error is absolute, a few dozen gaps at $|m|$ in a run of $10^5$ steps, and it"

The argument for why Welford's recursion is accurate is the hardest step of the section and is the least rigorous. Three assertions carry it: the error of the running mean is "a few dozen gaps" (my estimate agrees, about 50, but the text gives no reason), it "changes slowly, so it acts like a slightly wrong center", and a wrong center matters only at second order. The third is exact only for a *constant* shift $\varepsilon$; with a slowly drifting one the first-order terms $\sum_k\varepsilon_k(x_k-m)$ do not cancel, and they, not $N\varepsilon^2$, set the observed error of about $10^{-8}$. The conclusion (error within $u|m|/s$) is right and the cell confirms it. Following guide 10.2 ("candour about limits"), say that this is a heuristic and where the proof is (Chan, Golub and LeVeque 1983; Higham 2002, section 1.9), and state the real mechanism in one sentence: every quantity that is squared or accumulated is a deviation of size $s$, so the rounding errors enter relative to $s^2$, and only the error of the center, of relative size $u|m|/s$, is inherited from the scale $m$.

**NSC-13** · minor · language · under "One-Pass Variance"
> "**Proposition (Welford's recursion is exact).**"

In a section about rounding, "exact" in a title reads as "exact in floating point", which the next paragraph denies. The statement says "in exact arithmetic". Title it "Welford's recursion computes the mean and the centered sum of squares" or simply "(Welford's recursion)", and change "That asymmetry makes the recursion exact" to "makes the recursion correct".

**NSC-14** · minor · thought · end of "One-Pass Variance"
> "GPU batch-normalization kernels commonly compute the mean and variance of"

"this way" has three possible antecedents (naive, Welford, two-pass). I believe Welford with a pairwise merge is meant, which is what PyTorch's native CUDA kernel does; say "with Welford's recursion and the merge rule of Exercise ..." and, since this is a claim about software, name the library. Also under "Summation Order": "Once the sum passes $2^{19}=524{,}288$, float32 numbers are $1/16$ apart" holds only up to $2^{20}$; the sum ends at $1.09\times10^6$, where the gap is $1/8$ and $0.1$ still rounds to $0.125$. Add "(and above $2^{20}$, where the gap is $1/8$, to one gap, again $0.125$)".

**NSC-15** · moderate · explanation · under "Two Nearly Parallel Lines"
> "The amplification from backward to forward error is a property of the"

This is the point where the scalar $\kappa_f$ of the first half should meet the matrix $\kappa$. Add: "It is the condition number of the first half, for the function $\mathbf b\mapsto\mathbf A^{-1}\mathbf b$: the largest factor by which a relative change of the input can become a relative change of the output, taken over all right-hand sides and all directions of the change." Without it the table of $\kappa_f$ and the rule of thumb for $\kappa(\mathbf A)$ look like two unrelated uses of one letter. The same paragraph is also where the repeated numbers of S2 can go: replace the restated singular values and the $2.8\times10^4$ against $4\times10^4$ comparison by a reference.

**NSC-16** · minor · pedagogy · figure `fig_mdl-opt-two-lines`
> "The text's system, with $1.0001$ in place of $1.1$, has the same geometry at an angle of"

The figure draws a different system from the text's, with its own numbers (3.4%, amplification 29, $\kappa\approx42$), so the reader holds two examples and five extra numbers for one idea. The caption handles it honestly, and the text's lines could not be drawn. Still, consider making $1.1$ the text's example for the geometry paragraph and reserving $1.0001$ for the numbers taken from :numref:`subsec_mdl-condition-number`. In the rendered panel (b) the axis label $x_1$ sits on the three lines at the lower right.

**NSC-17** · moderate · correctness (sure of the logic; the numbers are platform dependent) · under "Example: The Hilbert Matrix"
> "Rounding the entries $1/3$, $1/7$, … to float64 is already a backward error"

The sentence attributes the gap between $\mathbf 1$ and "the exact solution of the *stored* system" to the rounding of the matrix entries. That cannot be the cause. The right-hand side is built as `b = H @ x_true` from the *stored* $\mathbf H$, so if $\mathbf b$ were computed without rounding, the stored system would have $\mathbf 1$ as its exact solution whatever the entries of $\mathbf H$ are. The deviation comes entirely from the rounding of the ten or so additions in each entry of `H @ x_true`, a relative perturbation of size $u$ in $\mathbf b$, amplified by $\kappa$. I checked with exact rational arithmetic: stored $\mathbf H$ with the stored $\mathbf b$ gives relative deviations $1.3\times10^{-7}$, $2.4\times10^{-4}$, $0.21$ for $n=8,10,12$ on my machine (the text says $7\times10^{-8}$, $4.5\times10^{-5}$, $0.14$; the difference is the summation order of the BLAS that computed $\mathbf b$), and stored $\mathbf H$ with $\mathbf b=\mathbf H_{\text{stored}}\mathbf 1$ evaluated exactly gives exactly $\mathbf 1$. Fix: "Computing $\mathbf b=\mathbf H\mathbf 1$ in float64 already rounds each entry of $\mathbf b$, a backward error of size $u$ before the solver starts: the exact solution of the system as stored differs from $\mathbf 1$ by about $10^{-7}$, $10^{-4}$, and $0.2$ ...". Quote orders of magnitude only, since no cell computes them and they vary with the BLAS; or move the computation into Exercise `ex_mdl-numerical-stability-conditioning-residual-certificate`, part 3, which already uses `fractions.Fraction`.

**NSC-18** · minor · thought · after the Hilbert cell
> "while the backward error stays of order $10^{-16}$ in every row: each"

The stored output prints `0.0e+00` in the row $n=4$. The earlier caveat ("printed values of $\omega$ below about $u$ are noise") covers it, but "in every row" contradicts what the reader sees. Write "stays at or below about $10^{-16}$".

**NSC-19** · minor · thought · under "Conditioning of the Normal Equations"
> "least-squares objective of a design with $\mathbf A^\top\mathbf A=\mathrm{diag}(1,1000)$, so"

The paragraph computes $\kappa(\mathbf A)=\sqrt{1000}\approx32$ for the running example and then, in the next sentence, builds designs with $\kappa(\mathbf A)=10^3$ and $10^6$ that have nothing to do with it. The sentence is true and interesting (it is where the reader learns that the chapter's $\kappa=1000$ is a $\kappa(\mathbf A)^2$), but it answers no question here. Move it to the ridge subsection, where the running example is used, or complete the thought: "so the running example is too well conditioned to show the effect ($16-3$ against $16-1.5$ digits); the cell uses harder designs."

**NSC-20** · moderate · pedagogy · under "Ridge Regularization and Conditioning"
> "**Proposition (filter factors).** *Let"

See S2. The proposition, its proof, and the worked numbers repeat `eq_mdl-ridge-filter` in other notation. Replace by one sentence: "By :eqref:`eq_mdl-ridge-filter`, ridge multiplies the component of the solution along $\mathbf v_i$ by $\sigma_i^2/(\sigma_i^2+\lambda)$ (there with $\gamma$ for $\lambda$)", then go straight to the observation that is new and good: the directions it damps most are the ones in which the solve amplifies data errors by $1/\sigma_i$. Reconcile $\lambda$ and $\gamma$ across the two chapters while there.

**NSC-21** · moderate · pedagogy · same subsection
> "**Proposition (iteration bound).** *Let $\mathbf M$ be symmetric positive"

See S2. The proof spends half its length re-deriving the contraction $\rho$ that "Proposition (optimal fixed step on a quadratic)" in :numref:`sec_mdl-gradient-based-optimization` already proves, and the result (the "cost law" that section quotes) is about gradient descent, not about rounding. Move the proposition there, where $6{,}908$ is first computed, and cite it here in a clause. What remains for this section is the short and pleasant experiment: $\lambda$ lowers $\kappa$, the steps fall in proportion, the last column shows the price. A small imprecision if the proposition stays: "falls below $\mathrm{tol}$" should be "is at most $\mathrm{tol}$", since the proof gives $\rho^k\le\mathrm{tol}$.

**NSC-22** · minor · thought · after the ridge table
> "except at $\kappa\approx2$, where $\log\frac{\kappa+1}{\kappa-1}\approx\log3$"

The stored ratios are $6.91$, $6.91$, $6.92$, $6.98$, $6.51$. A deviation from $6.9$ to $6.5$ is hardly an exception worth a sentence about which bound is sharper; the row $\lambda=100$ already drifts the other way. One clause is enough: "(at small $\kappa$ the approximation $2/\kappa$ is loose and rounding up to a whole step matters)".

**NSC-23** · minor · pedagogy · figure `fig_mdl-opt-conditioning-ellipse`
> "Level sets of the least-squares objective before and after adding the ridge term"

Two things in the rendered figure can mislead. Both panels put the star at the same place, although the text's point two lines later is that ridge moves the minimizer toward the origin. And the panels are drawn at different scales (the vertical arrow, proportional to $1/\sqrt{\sigma_1^2+\lambda}$, is longer in (b) than in (a)), so "The penalty lifts every eigenvalue" looks as if it lengthened an axis. Mark the unregularized minimizer in panel (b), or say in the caption that each panel is centered on its own minimizer and scaled separately.

**NSC-24** · minor · language · last paragraph before the Summary
> "You can now predict from $u$ and $\kappa$ how many digits a computation"

This is the "We can now ..." closing formula of guide 12.2 with the pronoun changed, and the only place where the section addresses "you" in the body. The second sentence (what these tools do not cover, with the pointer to :numref:`sec_numerical_stability`) is the real closing and is good. Drop the first sentence or fold its content into the second.

## Exercises

Twelve problems, at the top of the 8 to 12 band, graded with ● marks, 2,280 words (29% of the body). As a set they are the best I have read in this part: almost every one has a computation with a predictable answer (the loss-scale window is $S\in\{2^{16},\ldots,2^{19}\}$, the stochastic-rounding weight has mean $0.9$ and standard deviation about $0.02$, the Läuchli matrix rounds to all ones). I worked the first, second, fourth, fifth, seventh and twelfth far enough to confirm that they are answerable and that the stated facts are right.

- **Fix, Exercise 2 part 3** (anchor: "$\log(1+e^{-t})$, at $t\in\{-50,0,50,800\}$"). The exercise asks to show that the rewritten form "never overflows", but none of the test points makes the naive form overflow: at $t=800$ the naive form returns $0$, which is also what the true value $e^{-800}$ rounds to, and at $t=-50$ float32 holds $e^{50}$. The overflow case is $t=-800$ (naive: `inf`, true: 800). Replace $800$ by $-800$, or add it. Confidence: high (checked in float32).
- **Trim, Exercise 11** (residual certificate). Four parts and about 330 words, with exact rational arithmetic, to explain why the rule of thumb is pessimistic by a digit. Parts 2 and 3 are one idea (a residual computed in working precision is noise); merge them. If NSC-17 is fixed by moving the stored-system computation here, that replaces part 3.
- **Exercise 12** (iterative refinement) is a small project, correctly marked ●●, and :numref:`sec_mdl-optimization-further-reading` relies on it (see the further-reading review, FR-2). Keep.
- **Exercises 3 and 9** overlap slightly (both run Welford on small or sharded data); both earn their place, since 3 carries the backward-stability argument and 9 the merge rule the text cites.
- The text defers four explanations to exercises (Sterbenz's lemma twice, the misaligned error twice). That is acceptable for a lemma and for a refinement, but a reader who skips exercises never sees why exact subtraction is exact. One sentence of reason for Sterbenz in the text would do: "both numbers are multiples of the smaller one's gap, and their difference is too small to need more bits".
- Nothing to add. A gradient-check exercise would be the natural companion if NSC-10 is adopted.

## Still open from October

The October review (`docs/reviews/writing-review-2026-10.md`) does not discuss this section by name. Its general complaints about the appendix (openings that are topic lists, stock nouns, dash asides) do not apply here: the opening is a problem, I found none of the stock nouns, and the file has no em dashes in prose.

## For the chapter review

- **Fit and order.** Last of seven sections and the only one about arithmetic. It depends on the chapter for two things only: the running example $\tfrac12(x_1^2+1000x_2^2)$ and the contraction factor of :numref:`sec_mdl-gradient-based-optimization`; and on :numref:`subsec_mdl-weight-decay-duality` for one closing remark. It depends more on the linear-algebra chapter (:numref:`subsec_mdl-condition-number`, :numref:`sec_mdl-svd-low-rank`, :numref:`subsec_mdl-pseudoinverse`) than on its own. Placing it last is right; a case could be made that the first half belongs with :numref:`sec_numerics` in spirit and the second half with the SVD section, which is one more reason to consider the split of S1.
- **Overlap inside the appendix** is the chapter-level issue: the two-line system, the amplification bound, the filter factors, and the $6{,}908$ steps each have two homes. The assembler should check whether the other sections of this chapter also re-derive the step count; the number $6{,}908$ appears at least seven times in `mdl-gradient-based-optimization.md` alone.
- **Pattern worth spreading, not fixing:** prediction before every cell, a printed number after, and prose that reads the number. If other sections of the chapter have cells whose output the prose does not read, this section is the model.
- **Patterns I suspect hold for the part:** (1) two names for one object across chapters ($\lambda$ against $\gamma$ for ridge, bf16 against bfloat16, $\mathbf x$ against $\mathbf w$ for the unknown, $\sigma_d$, $\sigma_n$, $\sigma_r$ for the smallest singular value); (2) measured values quoted in prose with no cell behind them; (3) "part" used for a subsection.
- **Title collision** with :numref:`sec_numerical_stability` ("Numerical Stability and Initialization"): different subjects under nearly the same name. A clause in each opening that names the other would prevent a wrong landing.
- **Scope.** A deep learning reader might look here for quantization (integer and fp8 inference) and for the non-determinism of parallel sums across runs and devices; the first is mentioned in one sentence, the second not at all, though "Summation Order" explains its cause. One sentence there would be enough.
