# Estimating Mutual Information

Source: `chapter_mdl-information-theory/mdl-mi-estimation.md` (lines 1 to 1284, up to the slides marker). About 5,800 words of body prose, 2,200 words of exercises, 20 code cells (two of them in four framework tabs), 4 proposition and proof blocks, 2 generated figures and 2 computed plots.

## Verdict

This is the strongest of the two sections I read and, in its design, at the level of the model texts. It has one question a practitioner actually has (when a contrastive loss reports less than the truth, is the critic at fault or the bound?), it builds the experiment that answers it (the exact critic and a learned critic on the same batches of a pair whose information is known), and it finds one idea under four results: "an estimator cannot report more than the logarithm of the number of things it can tell apart." The mathematics is correct; I checked each proof and re-simulated every number the prose quotes, including the ones no cell prints, and all hold. What keeps it from a 5 is execution in the middle third: about ten quantitative claims are made that the stored outputs do not show (one with the admission "the cell does not print that case"), two paragraphs deliver a dozen numbers each before the explanation that would let the reader predict them, and the word "certify" is used for a population bound, for its minibatch estimate and for a high-confidence statement without the one sentence that separates the three. The most important fix is the first: either print what is claimed or predict it from the argument before the cell, so that the section's best explanation (rare pairs carry $\mathbb{E}[e^{\textrm{pmi}}]=1$) comes before the table it explains and not after.

## Scores

| Dimension | Score | Note |
|---|---|---|
| Clarity of explanation | 4 | KSG, the tilting proof of DV and the three-step InfoNCE proof are clear; NWJ paragraph and the reading of the exact-critic table are overloaded |
| Clarity of thought | 4.5 | One question, one unifying idea, an experiment that separates critic error from bound slack; "certify" blurs three meanings |
| Pedagogy | 4 | Motivation and design excellent; unprinted claims; explanation after the numbers |
| Language | 4 | Plain and exact, with judgment in places; opening leans on "the previous section" seven times; formula closing |
| Ease and enjoyment | 4 | The 128-candidate opening, the $2^{256}$ example and the ceilings make a reader want to go on |

## What works

- **The opening.** "That number can never exceed $4.85$ nats, however much the views share, because a model that always picks the right partner has shown only that it can tell $128$ candidates apart." A real practice, a number, and the whole section's thesis in one sentence. "we need to tell a measurement from a bound, and the slack of a bound from the error of a model" then states the two distinctions the section delivers. Protect this page.
- **A unifying idea, stated by the authors.** After the KSG ceiling: "This is the first ceiling of the section, and every later one has the same shape". The ceilings $\psi(n)-\psi(m)$, $2\log n$, $\log N$ and $2/\tau$ are then recognizably one phenomenon. This is the "authors judge" voice the guide asks for.
- **The distribution-free limit made concrete.** "A million samples cannot certify $177$ nats, and they cannot certify $30$." The $2^{256}$ example turns an $O(\log n)$ theorem into something a reader will remember (I checked: collision probability at most $4.3\times10^{-66}$, cap 27.6 nats).
- **Proofs that are the short ones.** The KSG ceiling by counting; Donsker-Varadhan by tilting $Q$ and reading the gap as $D_{\textrm{KL}}(P\|\tilde Q)$; InfoNCE as Barber-Agakov applied to the bag of candidates, with the candid marker "This is the step that is not routine" on the exchangeability argument. Each is led by a sentence saying what the proof will do.
- **The experiment design.** Exact critic first (so that any shortfall is the bound's), then a learned dual-encoder critic evaluated on the *same* held-out batches, with the result read correctly: at $N=8$ "the ceiling $\ln 8 = 2.08$ already exceeds the truth, so the shortfall of about $0.55$ nat there is the slack of the bound, not its cap". Slack and cap are separated, which most treatments do not manage.
- **The rare-pairs explanation.** "On such a batch NWJ reports $\mathbb{E}_P[T^*] - 0 = 1 + I$, one nat high, and DV reports the average pmi minus the logarithm of a near-zero average, many nats high." This explains the stored table (NWJ median 6.91 at $I=6$, mean 5.85; DV median 12.19) exactly.
- **Candour.** "Two questions remain open: how to estimate a large mutual information in high dimension at all ... and why loose bounds still train good representations."
- **Exercises** are deep and tied to the body (see below).

## Structural findings

1. **Claims the cells do not print.** All of the following are stated as observations and none appears in a stored output: KSG at $d=2$ ("it does as well in two (the cell does not print that case)") and at $d=8$; the mean strip counts 70, 31, 26, 26; that at $I=6$ the batch average of $e^{\textrm{pmi}}$ has a median of "at most a few hundredths" with "more than half of the batches" below 0.1; that "About one batch in four" holds a pair with $e^{\textrm{pmi}}\ge N$; and the curve readings 3.7, 4.6 and 6.3, which must be read off a plot to a precision the plot does not offer. I reproduced every one (KSG 1.67 at $d=2$ against 1.66 true, 2.90 at $d=8$; counts 71, 31, 25, 26; median 0.003, 61% below 0.1, 25% with a large pair; InfoNCE 3.68, 4.61, 6.31), so nothing is false. But the model texts never ask the reader to take a measurement on trust, and here the trust is asked at the two places where the section's argument turns. Under the author's decision that code changes wait, the writing pass can (a) derive what can be derived (the strip counts follow from the box-side argument already in the text; the "one nat high" follows from the rare-pairs argument), and (b) attribute the rest to the exercises that compute them, in the form "Exercise ... shows that". The later code pass should add $d=2, 8$ and the mean count to the KSG loop (two lines), and print a small table in place of the first plot.

2. **Explanation after the numbers.** In "Exact Critics Compared" the order is: a paragraph narrating the cell and its plot, the cell, a paragraph of twelve numbers, and only then "The rare pairs described in the NWJ subsection explain these numbers." The argument is strong enough to *predict* the table: a typical batch of 128 mismatched pairs contains none of the rare pairs, so its average of $e^{\textrm{pmi}}$ is near zero; hence NWJ should read $I+1$ on most batches and far below on a few, DV should read far above, and InfoNCE should be steady. Put that prediction before the cell, and the twelve numbers reduce to three confirmations. This is the method the Basics overview named as the book's own ("predict a number from a short argument, run the code, and let the number confirm or surprise").

3. **"Certify" carries three meanings.** The word appears 17 times. It is used (a) for the population inequality $I\ge\log N-\mathcal{L}_{\textrm{NCE}}$, (b) for the number a held-out minibatch average gives, which is unbiased for the bound but is not itself a guaranteed lower bound, and (c) for a high-confidence statement in the McAllester-Stratos sense. The section never says how (a) and (b) relate to (c), and a careful reader will ask: you proved that no distribution-free lower bound exceeds $O(\log n)$, and then you say a loss "certifies" information. The reconciliation is short and instructive, and it strengthens the unifying idea: InfoNCE reports at most $\log N$ with $N\le n$, so it respects the limit by construction; DV and NWJ can report more, and the section shows the price (minibatch DV is not a bound even in expectation; NWJ is carried by rare batches). Say this once, at the end of "Exact Critics Compared", and reserve "certify" for (a).

4. **The opening re-teaches the previous section.** Before the first heading the reader gets the Gaussian closed form again, the pmi definition again, and (in "Histograms in High Dimension") the histogram result with its numbers again; "the previous section" occurs seven times. Some restating is right for an appendix section that is reached by lookup. But the histogram is the one case where the restatement is the whole content of a subsection that has no computation of its own. Resolve it together with the companion finding in the mutual-information review: move the three histogram experiments (Gaussian sweep, $X^2$, $x^3$ with ranks) here. This section then starts its ladder with a cell (histogram fails on a thin ridge and under a change of coordinates), and Exercise 1, which calls `binned_mi`, finds the function in its own notebook.

5. **One result taught in two sections.** The "first instance" of Barber-Agakov, $I(X;Y)\ge\log k-L$, is the proposition "a loss certifies information" of the previous section (:eqref:`eq_mdl-info-certificate`), derived again here with a new example and no reference. Cite it in one clause and keep only the sentence that is new ("since $L\ge0$, no classifier certifies more than $\ln 10$. InfoNCE will have this shape").

6. **Heading structure.** "The Compression-Phase Debate" is a top-level `##` section of two paragraphs with the label `subsec_mdl-compression-debate`. It is an application of the binned estimator, not a peer of "Variational Lower Bounds". Make it a `###` under "Nonparametric Estimation" (after the histogram material, which it applies), or keep it last as a `###`. The content is right and short; the experiment is correctly left to an extended exercise.

7. **Length.** About right. The section is the shortest in the chapter and nothing in it is filler. The four-tab encoder and training cells are the only framework code in my half of the chapter; they are compact, and the shared prose around them is framework-neutral.

## Passage-level findings

**EST-1** · minor · repetition · "Histograms in High Dimension"
Anchor: "formula. At $\rho = 0.99$ that estimate was about $0.2$ nats low (roughly"
A worked number from another section is restated with its explanation (guide §7.7: "never repeats the worked number"). If the histogram experiments move here (structural finding 4) this becomes the reading of a cell. If they stay, reduce to "the histogram of :numref:`subsec_mdl-mi-gaussian` already fell short on a thin ridge".

**EST-2** · minor · explanation · "The KSG Estimator"
Anchor: "$\psi = (\log \Gamma)'$ in place of the logarithm to remove the bias of small"
Why digamma removes the bias is not said, and the reader cannot guess. One clause is enough and is true: "(for a count $j$ of points in a region of small probability $p$, $\psi(j)-\psi(n)$ is an unbiased estimate of $\log p$ where $\log(j/n)$ is biased)". If that is more than the authors want to justify, say so in the manner of the voice guide: "we do not derive this correction".

**EST-3** · minor · language · figure caption, "The KSG Estimator"
Anchor: "the ring lies on the vertical strip's boundary, so it counts in the horizontal strip only"
The figure itself is good: box, two strips and the three counts are visible at once. The caption is 110 words and spends its middle on an edge case of the drawing. Move the strict-inequality detail to the text (where "Both strip counts use the strict inequality" already says it) and cut the parenthesis.

**EST-4** · moderate · pedagogy (claims not shown) · "KSG in High Dimension"
Anchor: "as well in two (the cell does not print that case). At $d = 4$ it reports"
See structural finding 1. Also in this passage: "about what it reports at $d = 8$" and "They hold about $70$ of the $2{,}000$ points at $d = 1$, about $31$ at $d = 4$, and about $26$ at both $d = 8$ and $d = 16$". The strip counts are the evidence for the paragraph's explanation and deserve to be printed. Until they are, lead with the argument that is already there (a box holding $m/n$ of the sample has side $(m/n)^{1/(2d)}$, 0.05 against 0.83) and cite the exercise for the counts.

**EST-5** · moderate · explanation · "Limits of Distribution-Free Estimation"
Anchor: "with probability at least $1 - \delta > \tfrac{1}{2}$. The two events"
The argument is correct and is the right one to give, but its key step is compressed into "The two events overlap, so $B \leq 2 \log n$ on some all-distinct sample." A first-time reader has to reconstruct that (i) the bound is being applied to one particular distribution, uniform on $n^2$ symbols, whose entropy is $2\log n$; (ii) validity says $B\le 2\log n$ on most samples from *that* distribution; (iii) most samples from it are all-distinct; (iv) so some all-distinct sample has $B\le2\log n$; (v) $B$ sees only the type, so this holds for all all-distinct samples, from any source. Set it as those numbered steps, or at least give (i) its own sentence: "Apply the guarantee to the uniform distribution on $M=n^2$ symbols, whose entropy is $2\log n$." State that $B$ is a deterministic function of the type. Since Exercise 6 then walks through the same proof in four parts, decide where the proof lives (see Exercises). A small notational point: $\delta$ was the flip probability throughout the previous section and is the failure probability here.

**EST-6** · minor · thought · "Limits of Distribution-Free Estimation"
Anchor: "sample at once, and all it can rely on is which values repeat. Any estimator"
For continuous data no values repeat, so read literally the sentence says that nothing can be estimated, and KSG, which was accurate to 0.1 nat two pages earlier, plainly relies on more. The next sentence supplies the resolution (KSG assumes smoothness, that is, that the coordinates mean something). Merge the two so the claim is not made in the unqualified form: "an estimator that makes no assumption about the coordinates can rely only on which values repeat; one that uses distances, as KSG does, is assuming smoothness in the given coordinates."

**EST-7** · minor · language · "Variational Lower Bounds", opening paragraph
Anchor: "rests on Gibbs' inequality from the first section of this chapter: the KL"
Use the label (:numref:`sec_mdl-information_theory`), as everywhere else. In the same paragraph, "One estimation problem becomes an optimization problem, and the result carries the model's approximation error and the finite-sample error" is the paragraph's real point and is buried mid-paragraph between a definition and a citation; let it end the paragraph.

**EST-8** · moderate · repetition · "The Barber-Agakov Bound"
Anchor: "Two instances show what the bound certifies. First, exchange the roles of"
See structural finding 5. The second instance (the unit-variance decoder that "never certifies more than $\tfrac{1}{2}$ however strong the dependence") is new, sharp, and correct ($\rho^2/2=0.405$ against 0.830); keep it as is.

**EST-9** · minor · correctness (hypotheses, confident; no error in the proof) · "The Donsker-Varadhan Bound"
Anchor: "**Proposition (Donsker–Varadhan representation).** *For distributions $P$"
The proposition is titled a "representation" and states an inequality; the representation (supremum equals the divergence) is in the trailing clause. Either title it "bound" or lead with $D_{\textrm{KL}}(P\|Q)=\sup_T\{\ldots\}$. The hypothesis should also ask that $\mathbb{E}_P[T]$ be defined. It is worth one sentence after the NWJ bound that, for the same critic, DV is never smaller than NWJ (that is what $\log z\le z/e$ says), so the price of NWJ's unbiased minibatch value is a looser bound away from the optimal critic.

**EST-10** · moderate · explanation · "The NWJ Bound"
Anchor: "The logarithm in front of $\mathbb{E}_Q[e^T]$ couples all the samples of a"
One paragraph of 230 words carries seven ideas in sequence: the minibatch bias of DV, the tangent inequality, the NWJ bound, its optimal critic, its identity with the f-GAN bound, its unbiasedness, and its heavy tail. Each sentence is correct (I checked the direction of the Jensen step: the minibatch DV value is on average *above* the population value for that critic). The density is the problem. Split in three: (1) what is wrong with a minibatch DV estimate; (2) the tangent step, displayed, giving NWJ and its link to :numref:`sec_mdl-divergences-distances`; (3) what NWJ pays for unbiasedness. Paragraph (3) is the rare-pairs argument and is where the prediction asked for in structural finding 2 should start.

**EST-11** · minor · explanation · "The InfoNCE Bound", Step 3
Anchor: "Apply the ratio form of"
Step 3 divides by $p(v)$, defined in Step 2 as the law of $N$ independent draws from $P_Y$. Barber-Agakov needs the *marginal* law of the actual bag $V=(Y_1,Y_{2:N})$. The two coincide (the positive has marginal $P_Y$ and is independent of the negatives), but the proof does not say so, and it is the one place a careful reader will stop. Add: "The marginal distribution of the bag is $p(v)$, because the positive is marginally a draw from $P_Y$ and is independent of the negatives."

**EST-12** · minor · thought · "The InfoNCE Bound", after the proof
Anchor: "raises the ceiling, which is one reason CPC, SimCLR and CLIP use large"
Asserted without evidence, and in tension with the section's own later statement that "A tighter bound does not mean a better representation." Either give the reason that is independent of the bound (a larger batch contains harder negatives, so the loss keeps a useful gradient) or soften to "which is the reason usually given".

**EST-13** · minor · pedagogy (plot) · "The Ceiling with an Exact Critic"
Anchor: "$I = 4$ the $N = 128$ curve reads $3.7$, although its ceiling is $4.85$; at"
The stored plot shows the qualitative claim well ("Each curve leaves the diagonal early and smoothly, not at its ceiling" is visible and is a real finding). The three quoted values cannot be read from it, the legend covers the right end of the $N=16$ curve and of its ceiling line, and the dotted ceilings are not in the legend. For the writing pass, say "about". A later code pass should print the three or four values quoted.

**EST-14** · moderate · thought · "The Ceiling with an Exact Critic", last paragraph
Anchor: "nats. An InfoNCE bound between the features and the label can"
The paragraph applies InfoNCE to (features, label), which is not how any contrastive method is used: the loss is computed between two views, and the bound is on the information the views share. The step from view-to-view information to label information is missing, so the conclusion ("only with more than a thousand" candidates) answers a question nobody poses. The honest version is weaker and still worth saying: "Fano's inequality says that $5\%$ error on $1{,}000$ classes needs $6.36$ nats about the label. A loss with $N=128$ candidates cannot show that two views share even that much, since it reports at most $4.85$; whatever makes such representations good for classification, it is not something this number can demonstrate." That also prepares the Tschannen paragraph. ($e^{6.36}=578$, $e^{6.364}=581$; with "about" either is fine.)

**EST-15** · moderate · explanation, pedagogy · "Exact Critics Compared"
Anchor: "spread better than a standard deviation does. The plot draws the"
The lead-in paragraph describes the cell and its plot ("solid lines", "dotted lines of the same colour", "computed with `logsumexp` so that it cannot underflow") where it should say what to expect. The reading paragraph that follows ("Up to $I = 3$ the three medians are within about $0.15$ nat ...") is accurate against the stored table in every figure, but it is a list. See structural finding 2 for the reordering. The plot has eleven lines and the DV curve leaves the frame; the printed table carries the content, and the plot could go.

**EST-16** · minor · thought (overgeneralization) · "Summary", and "Training a Critic"
Anchor: "$\log N$. A learned critic reproduces the exact one to a few hundredths of a"
The result is for a scalar Gaussian pair with a two-hidden-layer MLP; there the critic has an easy job. The summary states it as a general fact and draws "so the gap between a contrastive loss and the truth belongs to the bound". The body is careful ("The gap to the true $1.96$ nats belongs to the bound" refers to this run). Add "on this pair" in the summary, and one sentence of candour in the body: in high dimension the critic's error is no longer negligible, and without a known truth the two cannot be separated, which is why the experiment was run where the truth is known.

**EST-17** · minor · thought · "Summary", first sentence
Anchor: "From samples, mutual information can be bounded from below but not measured:"
Overstated against the section's own evidence: KSG was within 0.1 nat in one and two dimensions. The accurate statement is the more interesting one: it can be measured when it is small and the dimension is low, and beyond that every method reports at most the logarithm of something it can afford.

**EST-18** · minor · language · closing paragraph before "Summary"
Anchor: "We can now name the two ceilings of a nearest-neighbor estimator, read a"
The previous section closes with "We can now compute ..." and this one with "We can now name ...". The guide (§12.2) asks that two consecutive sections not close with the same construction, and names this formula. The second half of the paragraph ("Two questions remain open ...") is a real closing; start the paragraph there.

## Figures and plots

`mdl-it-ksg-box.svg` is clear and matches the estimator as coded. `mdl-it-infonce-pos-neg.svg` is adequate; its right panel (a softmax bar chart with the bound printed above it) says little that the formula does not, and "nearer candidates score higher" in the caption introduces a geometry the critic does not have. The two computed plots are data plots of results and are legitimate under the house rule; see EST-13 and EST-15 for their legibility.

## Exercises

Ten problems, 2,200 words, about 39% of the body. That is above the guide's 30%, and I would not cut for length: these are the best exercises in my half of the chapter, and several carry real content (heavy tails of $e^{\textrm{pmi}}$, optimal critic, temperature cap, in-batch negatives).

- **1 (histogram bias).** Good ($G$-statistic prediction $(b-1)^2/2n=0.0038$ against the observed 0.003), but it calls `binned_mi`, which is defined in the other notebook. It should travel with the histogram material (structural finding 4).
- **2 (heavy tails).** Excellent and long; I checked (b) ($\det(2\boldsymbol\Sigma^{-1}-\mathbf I)=1$, $\int p^2/q=1/(1-\rho^2)$), the values in (c) (0.43, 23, 1270) and (d) (0.65, 4.8, 36). Part (e) is where the body's unprinted claims are actually computed; if the body prints them, (e) can shrink.
- **3 (KSG ceiling).** Good; (d) with noise $10^{-6}$ (true $I\approx13.8$ nats, report 6.09) is a memorable case and could be mentioned in the body.
- **6 (entropy bounds from samples).** Parts (a) to (c) repeat the body's argument step for step, numbers included ($2^{256}$, $10^6$, 27.6). The body says the exercise "works through the proof", but the body already contains the proof. Choose: either the body states the result with the $2^{256}$ illustration and leaves the proof to the exercise, or the body keeps the proof (expanded as in EST-5) and the exercise keeps only (d) plus a new part.
- **9 (negative sampling as shifted PMI).** Good mathematics, but it depends on `subsec_negative-sampling` and `sec_word2vec_data`, which live in the retired NLP Pretraining chapter now in the Attic. The exercise is self-contained once the loss is displayed, so drop the two references or repoint them when the new word-embedding section exists. Consider moving it to the previous section, where PMI is taught and where the result is quoted.
- **10 (compression phase, extended).** A fair project; (a) is easy and (b) to (d) are an afternoon. Marked ●● and [extended], which is honest.
- 4, 5, 7, 8: keep. 5(a) I checked numerically (2.0, 4.0, 8.3 nats at $\tau=1, 0.5, 0.1$). 8 quietly answers a question the body raises and does not settle (rows share candidates; the bound holds in expectation), and supports structural finding 3.

## Still open from October

The October review did not treat this section separately; nothing to carry over.

## For the chapter review

- **Fit.** The right last section: it takes the chapter's central object and asks what survives contact with data. It assumes the Gaussian closed form, pmi, the chain rule and Fano from the previous section and the f-GAN bound from `mdl-divergences-distances.md`, and restates each. It can be read from a forward reference (the main text's embedding plans already point at `sec_mdl-infonce`) with the previous section's formula for the Gaussian pair as the only real prerequisite.
- **Boundary with `mdl-mutual-information.md`.** Three items straddle it: the histogram experiments (taught there, recapped here), the loss certificate (proposition there, re-derived here as Barber-Agakov), and the shifted-PMI exercise (here, while PMI is taught there). Fix all three in one decision: known distributions there, samples here.
- **A model for the part.** The design of this section (a practitioner's question, an exact baseline that isolates one source of error, a unifying idea stated by the authors) is what the other sections should be measured against. If the chapter review needs an example of "voice" from inside the appendix, it is the paragraph ending "an estimator cannot report more than the logarithm of the number of things it can tell apart."
- **Suspected chapter-wide patterns.** (a) Quantitative claims made beside a cell that does not print them; here every one checked out, but the habit should be looked for elsewhere. (b) "We can now ..." closings in consecutive sections. (c) A summary that lists every result. (d) Section openings that restate the previous section at paragraph length.
- **Scope.** Upper bounds on mutual information (needed for the compression term of a bottleneck) are correctly left to `further-reading.md`, which also holds the noisy-network resolution of the compression debate. Nothing a deep learning reader would look for here is missing, except one sentence relating the variational bounds to the distribution-free limit (structural finding 3).
