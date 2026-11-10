# Mutual Information

Source: `chapter_mdl-information-theory/mdl-mutual-information.md` (lines 1 to 1367, up to the slides marker). About 7,000 words of body prose, 1,600 words of exercises, 18 code cells (all NumPy), 14 proposition and proof blocks, 5 generated figures.

## Verdict

This is good textbook prose, close to the bar in its first half and somewhat below it in the second. The mathematics is correct: I recomputed every worked number (the flip-channel entropies, the PMI table, the $X^2$ integral, the Fano table, the bottleneck operating points) and checked every proof, and found no error, only two loosely stated side remarks. The section has the thing the Basics part lacked, a pair of small running examples that are made to do real work: the bit flipped with probability 0.1 is the joint table, the Markov chain, the tight case of Fano, and the channel whose capacity is its own mutual information. The single most important thing to fix is structural, not local: the section declares "Throughout this section the distributions are known" and then runs three histogram-estimation experiments that belong to the next section, while the things a deep learning reader most needs said in the body (that the data-processing inequality does not contradict "depth helps", that a deterministic layer has infinite or meaningless $I(X;Z)$, that the KL term of a VAE is a rate) are left to an exercise, to a remark 350 lines later, or unsaid. Second in importance: fourteen results are all dressed as "Proposition, Proof", so the reader cannot tell the three that matter (data processing, Fano, the loss certificate) from one-line algebra.

## Scores

| Dimension | Score | Note |
|---|---|---|
| Clarity of explanation | 4 | Example before or right after every definition; two measure-theoretic asides arrive too early |
| Clarity of thought | 4 | One line of argument through DPI and Fano; estimation leaks in; one result taught again in the next section |
| Pedagogy | 4 | Running examples and predict-then-compute cells are strong; capacity and rate-distortion lack a deep learning use |
| Language | 4 | Plain, exact, mostly tic-free; flat register in the second half; a recap closing plus an inventory summary |
| Ease and enjoyment | 3.5 | The first half is a pleasure; "Communication and Compression" and the bottleneck read as a sequence of closed forms |

## What works

- **The opening example.** $Y=X^2+0.3\,\epsilon$ has zero correlation, and the section promises that mutual information "credits this pair with more shared information than a Gaussian pair with correlation $0.9$". The promise is kept 400 lines later with a number (1.141 against 0.830) and a figure (`mdl-it-mi-corr-vs-mi.svg`, which renders cleanly and carries the point in its three panel titles). This is how MacKay opens a chapter. Keep it.
- **One table, used six times.** The $2\times2$ table with 0.45 and 0.05 gives $H(X,Y)=1.018$, $H(Y\mid X)=0.325$, $I=0.368$, the pmi values $+0.588$ and $-1.609$, the tight case of Fano, and the capacity of its own channel. Each reuse costs one sentence. The second flip of 0.2 composes to 0.26 by hand ("Two flips in series are one flip") and the cell confirms $0.120$. Nothing here is decoration.
- **Sentences worth protecting.** "a copy that is wrong one time in ten carries about half a bit about the original"; "Flips destroy information only to the extent that the receiver cannot predict them"; "Mutual information is invariant; an estimator is invariant only if it is built from something invariant, such as ranks"; "Agreements occur $1.8$ times as often as independence predicts, and disagreements one fifth as often."
- **Counterexamples placed where the reader would overreach.** XOR directly after the data-processing inequality ("The inequality is a statement about Markov chains, not about conditioning in general"); the always-abstaining estimator directly after Fano's statement, showing why $\hat Y$ must live in the label set.
- **"A loss certifies information."** $I(X;Y)\ge H(Y)-L$ with the confidence sweep (only the calibrated $c=0.9$ certifies the full 0.368 nats; the overconfident model "makes the same decisions" and certifies 0.22) is the best deep learning connection in the section, and the closing remark that Fano and the certificate bound the same $H(Y\mid X)$ by two routes is exactly the kind of judgment the voice guide asks for.
- **The Gaussian bottleneck.** The frontier slope $\rho^2$ at the origin explains the collapse threshold $1/\rho^2$ in one sentence ("a nat of rate adds $1$ to the objective but removes at most $\beta\rho^2\le1$"), and the grid search confirms the closed form on both sides of the threshold (0.000 at $\beta=1.2$, 0.032 at $\beta=1.25$).
- **Code.** Every cell computes a number the prose then reads, and the stored PyTorch outputs match the prose everywhere I checked. No cell draws.
- **Exercises** are the best part of the section (see below).

## Structural findings

1. **Estimation leaks into a section about known distributions.** The introduction says "Throughout this section the distributions are known" and hands estimation to the next section. The body then bins 200,000 samples three times: the Gaussian sweep (cell `mutual-information-gaussian-binned`), the $X^2$ pair (`mutual-information-nonlinear-correlation`), and the $x^3$ and rank experiment. That is three cells and about 450 words on plug-in bias, bin resolution and rank invariance. The next section's "Histograms in High Dimension" then has no code of its own, recaps the result with its number ("roughly $1.78$ against $1.96$"), and hosts an exercise on `binned_mi`, a function that is not defined in its notebook. Move the three experiments to the start of the estimation section, where they are the first rung of its ladder (histogram, KSG, variational). Here, keep only the exact computations: the closed form, and the quadrature for $X^2$.

2. **Fourteen propositions, one rank.** In 7,000 words the reader meets: chain rule for entropy, properties of MI, Gaussian MI, reparameterization invariance, chain rule for MI, DPI, Fano, loss certificate, BSC capacity, Gaussian rate-distortion, Gaussian channel, collapse for $\beta\le1$, bottleneck as rate-distortion, operating point. Several are one line of algebra (chain rule for entropy, Gaussian channel, collapse for $\beta\le 1$). Set as propositions with a tombstone they look as weighty as Fano. The voice guide's first item is "the authors judge": say which results the section exists for (DPI, Fano, the certificate, the bottleneck identity), and fold the one-liners into running text ("Factoring $p_{X,Y}=p_X\,p(y\mid x)$ inside the expectation gives the chain rule $H(X,Y)=H(X)+H(Y\mid X)$").

3. **The deep learning reader's three questions are answered late or not at all.**
   - *Does DPI mean depth destroys what we need?* The body says "no later layer can recover what an earlier layer discarded" and stops. The answer (information is not created, but it can be made accessible to a simple reader) is Exercise 3, an excellent problem, but the point is too important to live only there.
   - *Is $I(X;Z_\ell)$ even finite for a real network?* The DPI bullet treats it as a number. The warning comes under the bottleneck ("Encoders with continuous outputs must be stochastic") and in the next section's compression-phase debate.
   - *What is the KL penalty of a VAE measuring?* The chapter introduction asks this question explicitly. This section holds the answer (the bound $I(X;Z)\le\mathbb{E}_x D_{\textrm{KL}}(p(z\mid x)\|r(z))$ is the VAE's KL term, so that term is an upper bound on the rate) and does not say so. See MI-19.

4. **Channel capacity has no named use.** "Communication and Compression" is motivated as "Shannon's two operational questions". Rate-distortion earns its place because the bottleneck is built on it. Capacity is used by nothing later in the section or, as far as I can find, by the main text. Either name a use (the Gaussian channel $\tfrac12\log(1+\textrm{SNR})$ is the natural one: it is the bottleneck encoder, the noising step of a diffusion model, and Exercise 9) or reduce the subsection to the paragraph that the symmetry "max of $I$ / min of $I$" needs. If kept, the Gaussian-channel proposition, which currently sits under "Rate-Distortion", belongs here.

5. **The certificate is the Barber-Agakov bound, and it is taught twice.** "Proposition (a loss certifies information)" here and "Proposition (Barber-Agakov bound)" in `mdl-mi-estimation.md`, whose first instance re-derives $I(X;Y)\ge\log k-L$ from scratch. Neither cites the other. One home, one clause in the other (guide §7.7). I would keep the proposition here, name it ("this is the bound of Barber and Agakov, which :numref:`sec_mdl-mi-estimation` generalizes"), and let the next section cite :eqref:`eq_mdl-info-certificate`.

6. **Closing recap plus inventory summary.** The body ends with a "We can now compute ... We can prove ... We can compute ..." paragraph (the fixed formula §12.2 names), and the summary then lists every result again with its number in one 18-line paragraph. A reader gets the 0.368, 0.120 and 6.36 for the third or fourth time. Keep the last two sentences of the closing (every computation used the joint density; images and text do not have one) and cut the rest. Rewrite the summary as answers: what the number is, why it is the right one, what it limits, what the bottleneck is.

7. **Length and split.** At 7,000 words the section is not too long, and I would not split it. Moving the histogram material out (finding 1) and trimming capacity (finding 4) would remove about 700 words; the body additions asked for in finding 3 cost about 200.

## Passage-level findings

**MI-1** · minor · language · under the title, opening
Anchor: "The tool we already have, the correlation coefficient, answers only for" and, two paragraphs later, "The tool for such a number already exists."
Two different things are "the tool" within 20 lines (correlation, then KL divergence), and the second sentence echoes the first. Rename the second: "We already have what we need to build such a number."

**MI-2** · minor · thought · opening
Anchor: "information from samples alone is the open problem that"
"Open problem" means unsolved in a mathematics text. The next section shows something more precise: it is hard, and provably limited. Write "is a different and harder problem, which :numref:`sec_mdl-mi-estimation` takes up."

**MI-3** · minor · pedagogy · imports cell
Anchor: "#@tab mxnet" followed by `import numpy as np` in four tabs
The section is pure NumPy, which is a strength: no tab ever interrupts the prose. The imports cell should then be a single untagged cell, not four identical tabs.

**MI-4** · minor · explanation · "Joint and Conditional Entropy"
Anchor: "discrete variables, $H(X,Y)$ lies between $\max\{H(X),H(Y)\}$ and"
The two-sided bound is asserted for every discrete pair before the chain rule that proves it (lower bound from $H(Y\mid X)\ge0$, upper bound from $I\ge0$, which comes a subsection later). Either move the sentence to after the properties proposition, where it is a one-line corollary, or mark it "as we show below".

**MI-5** · minor · language · "Mutual Information"
Anchor: "It measures how far the pair is from independence, in nats."
This repeats the sentence before the display ("measures with the KL divergence how far the actual joint distribution is from that product"). Delete it and start the paragraph with "In the running example".

**MI-6** · moderate · explanation, correctness (low severity, confident) · "Mutual Information"
Anchor: "The definition needs the joint distribution to have a density with respect"
Three problems in one paragraph. (a) Placement: the reader has just computed their first mutual information on a $2\times2$ table and is immediately given a measure-theoretic caveat using "density with respect to the product of the marginals" and "a continuous variable without atoms", neither of which has been introduced. Nothing so far is continuous. (b) "$Y=\textrm{sign}(X)$ has $I(X;Y)=\ln2$" holds only when $P(X>0)=\tfrac12$; no distribution for $X$ is given. (c) "For a discrete $Y$ the next proposition gives $I(X;Y)=H(Y)-H(Y\mid X)\le H(Y)$, whatever $X$ is": the next proposition is stated "when the entropies involved are finite" and proved by splitting into $H(X)$, $H(Y)$, $H(X,Y)$, which do not exist for a continuous $X$. The claim is true, but that proposition does not give it.
Fix: move the paragraph to the end of "The Gaussian Pair", where $\rho\to\pm1$ supplies the concrete case, and say it plainly: "Mutual information can be infinite. If $Y=X$ for a continuous $X$, knowing $X$ fixes $Y$ to infinitely many digits. The same happens whenever $Y$ is a function of $X$ and still takes a continuum of values. It does not happen when $Y$ is discrete: then $I(X;Y)=H(Y)-H(Y\mid X)\le H(Y)$ (apply the identity for each value of $x$ and average). For standard normal $X$, the sign of $X$ carries exactly $\ln 2$ nats." The later references to "the atomless case of the remark after the definition" (twice) then need rewording.

**MI-7** · moderate · pedagogy (proofs) · "The Gaussian Pair"
Anchor: "We compute the three differential entropies; the variances"
The proof is correct but is the long route: it needs the bivariate density, a determinant, and the trace identity for the quadratic form. The short route uses the identity the section relies on later for both the $X^2$ example and rate-distortion: $I=h(Y)-h(Y\mid X)$, and given $X=x$ the variable $Y$ is Gaussian with variance $\sigma_Y^2(1-\rho^2)$ whatever $x$ is, so $I=\tfrac12\log(2\pi e\sigma_Y^2)-\tfrac12\log(2\pi e\sigma_Y^2(1-\rho^2))$. Two lines, and it shows why the answer depends only on the fraction of variance left unexplained, $1-\rho^2$, which is the intuition the formula deserves. Also, the paragraph before the proposition re-introduces differential entropy ("the formulas above with integrals in place of sums define") although :numref:`sec_mdl-information_theory` has a subsection on it; cite that and keep only the two facts needed here (it can be negative, it depends on units).

**MI-8** · moderate · structure · "The Gaussian Pair"
Anchor: "How well does a histogram recover the closed form from samples?"
See structural finding 1. The reading of the output is accurate (stored output: 0.003, 0.815, 1.790 against 0, 0.830, 1.959; my reruns give 1.776 to 1.799 at $\rho=0.99$). It is the right experiment in the wrong section.

**MI-9** · minor · pedagogy · "Pointwise Mutual Information"
Anchor: "negative samples per pair implicitly factorizes the matrix of shifted"
The Levy-Goldberg result is stated in one clause, correctly, and the proof is Exercise 9 of the *next* section. Add the pointer (":exref:`ex_mdl-mutual-information-negative-sampling-pmi`"), or move that exercise here, where PMI is taught and where the main text's word-embedding plan (`chapter_lm-encoders/word-embeddings.md`) expects to find PMI.

**MI-10** · moderate · thought (proof stronger than stated is available) · "Mutual Information as Nonlinear Correlation"
Anchor: "them be continuously differentiable with continuously differentiable"
The smoothness hypotheses are an artifact of the change-of-variables proof, and "except possibly at finitely many points" only makes sense for scalar variables. The statement is true for any bijection that is measurable in both directions, with a two-line proof from data processing: $X\to Y\to\psi(Y)$ gives $I(X;\psi(Y))\le I(X;Y)$, and $X\to\psi(Y)\to Y$ gives the reverse. The Jacobian argument is still worth keeping because it shows why $h$ is *not* invariant (the factor does not cancel). Suggested fix: keep the proposition and proof here but say "we prove the smooth case, which shows the mechanism", and add after the DPI: "Applied in both directions, the inequality shows that any invertible map preserves mutual information; the smoothness assumed in the earlier proof was not needed." Confident.

**MI-11** · moderate · structure, pedagogy · "Mutual Information as Nonlinear Correlation"
Anchor: "changes, from about $0.81$ to roughly half of that (between about $0.3$ and"
The $x^3$ experiment demonstrates a property of the histogram estimator (its value depends on the sample extremes; I see 0.34 to 0.53 over 30 runs), not of mutual information, and its question ("Does the invertible map ... change the correlation, the histogram estimate, or the mutual information") cannot be answered for the third item by the cell, since the true $I$ is never computed after the map. The rank-binned value (0.8122) is printed and never compared with the truth 0.830. Move the experiment to the estimation section with the other histogram material. Here, the exact value $3\rho/\sqrt{15}=0.697$ for the correlation is enough to show that correlation is not invariant.

**MI-12** · minor · explanation · "Conditional Mutual Information and the Chain Rule"
Anchor: "conditionally independent given $Z$. On the chain, $I(X;Y\mid Z)=0.248$"
The number appears with no computation and is confirmed only by a cell two subsections on. One clause fixes it: "(by the chain rule below, $0.368-0.120$)", or move the number to after the chain rule. The explanation that follows ("because $Z$ already carries part of what $Y$ says about $X$") teaches that conditioning reduces shared information, which the XOR example later contradicts. Add "here" and a forward flag: "On this chain conditioning lowers the information; :numref:`subsec_mdl-dpi` shows a case where it raises it." Also $H(X\mid Y,Z)$ is used without a word that it is the conditional entropy given the pair.

**MI-13** · moderate · thought · "The Data-Processing Inequality", first bullet
Anchor: "lacks, and no later layer can recover what an earlier layer discarded."
True, and the place where a deep learning reader will draw the wrong conclusion: that later layers know less and so are worse. The correction is Exercise 3 (usable information). Put its point in the body, in two sentences: "This does not make depth harmful. Layers do not add information about the label; they rearrange it so that a simple final layer can read it, and a linear classifier can do better on $Z_2$ than on $Z_1$ although $I(Y;Z_2)\le I(Y;Z_1)$ (:exref:`ex_mdl-mutual-information-usable-information`)." Add also the finiteness caveat here and not 350 lines later: for a deterministic layer of a continuous input, $I(X;Z_\ell)$ is infinite or, after quantization, just the entropy of $Z_\ell$, so the first chain is informative only for noisy or discrete representations.

**MI-14** · minor · correctness (wording, confident) · "The Data-Processing Inequality", second bullet
Anchor: "nothing exactly when it keeps a *sufficient statistic*: a $T(Y)$ such that"
"A many-to-one or noisy map loses nothing exactly when it keeps a sufficient statistic: a $T(Y)$": a noisy map $Z$ is not of the form $T(Y)$. The condition from the proposition is that $X\to Z\to Y$ is also a Markov chain; for a deterministic $Z=T(Y)$ this is sufficiency. State the general condition first and sufficiency as its deterministic case. The last sentence treats the parameter $\theta$ as a random variable $X$; say "with a prior on $\theta$" so that $I(\theta;\cdot)$ is defined.

**MI-15** · minor · explanation · "Channel Capacity"
Anchor: "The maximization matches the input to the channel."
The sentence says nothing a reader can use. Replace it with what the maximization does in the example that follows: "A sender who transmits mostly zeros wastes the channel: the input has little entropy to convey." The cell output (0.146 nats at $P(X=1)=0.1$ against 0.368 at the uniform input) then confirms a stated prediction.

**MI-16** · minor · explanation · "Rate-Distortion"
Anchor: "**Proof of the lower bound.** We need one fact from"
The proposition claims equality; the proof block proves $\ge$ and ends with a tombstone; achievability comes afterwards, after a second proposition, in running text ("Adding Gaussian noise and taking the conditional mean attains the curve"), and only for $\sigma_X^2=1$. A reader who stops at the tombstone thinks the proposition is proved. Say at the tombstone "It remains to exhibit an encoder that attains the bound", and close the achievability paragraph with "which completes the proof (for general $\sigma_X^2$, rescale)". $\mathbb{E}[X\mid Z]=Z/(1+\sigma^2)$ is used without a reason; one clause ("the regression of $X$ on $Z$, with slope $\textrm{Cov}(X,Z)/\textrm{Var}(Z)$") suffices.

**MI-17** · minor · pedagogy (code) · "Rate-Distortion"
Anchor: "information kept equals $R(D)$ at all three noise levels. In numbers,"
The cell measures only the squared error. The two columns "I(X;Z)" and "R(D)" are two closed-form expressions that are equal by algebra, so their agreement to four digits is not evidence of anything. The numbers the prose then quotes ("$D=0.1\sigma_X^2$ costs $1.15$ nats ... $D=0.5\sigma_X^2$ costs $0.35$") are not the cell's: the cell's first row is $D=0.0909$, rate 1.199. For the writing pass, say what the cell shows ("The measured squared error agrees with $D$ to within a few thousandths; the rate is then $\tfrac12\log(1/D)$ by the identity above") and quote the cell's own rows. A later code pass could choose $\sigma^2=1/9, 1, 3$ so that $D=0.1, 0.5, 0.75$.

**MI-18** · minor · explanation · "The Bottleneck Objective"
Anchor: "and the encoder approaches a minimal sufficient statistic"
"Minimal sufficient statistic" is not defined (sufficient statistic was, 330 lines earlier). One clause: "the coarsest function of $X$ that still predicts $Y$ as well as $X$ does". The next paragraph ("Encoders with continuous outputs must be stochastic") is a single orphan sentence between two propositions; attach it to the definition of the encoder at the top of the subsection, where "stochastic encoder" is first said.

**MI-19** · moderate · thought (missed connection) · "The Bottleneck Objective"
Anchor: ":numref:`sec_mdl-latent-em-elbo` uses the same substitution: it replaces an"
"The same substitution" is imprecise. The ELBO replaces a posterior by a variational family; the bound just derived replaces the *marginal* $p(z)$ by a prior $r(z)$. The exact link is closer and more useful: $\mathbb{E}_x D_{\textrm{KL}}(p(z\mid x)\,\|\,r(z))$ is, term for term, the KL penalty of a variational autoencoder. Draft: "This is the KL term of a variational autoencoder. That term is therefore an upper bound on the rate $I(X;Z)$, loose by $D_{\textrm{KL}}(p(z)\,\|\,r(z))$, and the weight $\beta$ placed on it in a $\beta$-VAE is the multiplier of a rate-distortion problem whose distortion is the reconstruction loss." This answers the question the chapter introduction poses ("with a KL penalty in a variational autoencoder ... what exactly are we measuring?") and nothing else in the chapter does.

**MI-20** · minor · language · notation, whole section
Anchor: "I(X;Y)=0.9\ln1.8+0.1\ln0.2=0.529-0.161=0.368\textrm{ nats}"
`\ln` and `\log` both mean the natural logarithm and alternate within paragraphs ($\ln2-H_b(\delta)$ next to $\tfrac12\log(2\pi e\sigma^2)$; the Fano display has $\log(k-1)$ and the text below it $\ln1000$). The convention seems to be `\ln` for numbers and `\log` for formulas, but it is not stated and not kept ($0.1\log1$). State it once in the units sentence of the introduction or use one symbol. Similarly conditional densities are $p(y\mid x)$ while joints and marginals carry subscripts ($p_{X,Y}$, $p_X$); harmless, but say so once.

**MI-21** · moderate · language, repetition · end of "The Gaussian Information Bottleneck" and "Summary"
Anchor: "We can now compute mutual information and its conditional forms for any"
See structural finding 6. Draft closing: "Every computation in this section used the joint density. For images and text neither the joint distribution nor the marginals are available, and :numref:`sec_mdl-mi-estimation` asks what can be measured from samples alone."

**Pattern: cell lead-ins.** The body introduces cells with "The next cell ..." six times and with a question ("How well does ...?", "Do samples show the same thing?", "Does the distortion of the simulated encoder match the formula?", "Does a direct minimization ... agree with the closed form ...?") four times. The questions are good when the answer is not known in advance (the collapse threshold). When the question is "does the formula we just proved hold?", the answer is always yes and the device wears thin. Keep it for the cells that can surprise.

## Figures

All five render cleanly and each carries weight. `mdl-it-mi-overlap.svg` prints the three identities beside the disks, which duplicates the proposition; harmless. `mdl-it-capacity-rd.svg` marks the running points (0.37, 0.12) on the capacity curve, which ties it to the text. `mdl-it-ib-tradeoff.svg` is the best of the set: slope, operating points, collapse and ceiling in one panel. No changes needed.

## Exercises

Ten problems, graded, almost all good; several are better than the body they follow.

- Keep without change: 1 (Gaussian pmi; part (d) sets up the $\log N$ problem of the next section with 13.3 nats at $m=16$), 2 (rare pairs and PMI, a real pitfall), 3 (information versus usable information, the best problem in the chapter half), 5 (Fano and its converse, with a bracket [about 0.014, 0.19] that makes a point about predicting error from information), 6, 7 (the 0.255-bit quantizer gap; I checked $\tfrac12\log(2\pi e/12)=0.1765$ nats), 8 (dropout as an erasure channel is the capacity subsection's only deep learning link), 9, 10 (checked: $\chi^2=(1-2\delta)^2$, and $0.8\cdot0.6=0.48=1-2\cdot0.26$ composes, which part (d) could ask the reader to notice).
- Fix: 4(a) asks which of three Markov chains hold for the XOR triple; the body already answers ("No Markov chain holds among the three"). Drop (a) or drop the sentence from the body.
- Fix: the labels of 7 and 8 use the prefix `ex_mdl-information-theory-` while the others use `ex_mdl-mutual-information-`. Cosmetic.
- Promote: the point of 3(c) and 3(d) should also be stated in the body (MI-13).
- Consider moving here: the negative-sampling and shifted-PMI problem from the next section (MI-9).
- Missing: nothing essential. A short problem on the VAE reading of the rate bound (MI-19) would fit after the bottleneck.

## Still open from October

The October review asked for joint entropy, mutual information and data processing to move into the first section in Cover and Thomas order. They remain here, fourth, after 8,000 words of divergences; with the present chapter introduction that order is defensible, and I do not repeat the request. Its "keep" item (mutual information as the KL divergence from the independent product) survives intact. The "rebuttals to an imagined reviewer" it flagged near old line 684 are gone.

## For the chapter review

- **Fit.** The section assumes entropy, Gibbs' inequality, the Gaussian entropy $\tfrac12\log(2\pi e\sigma^2)$ and the max-entropy property from `mdl-information-theory.md`, lossless coding from `mdl-coding-compression.md`, and f-divergence data processing from `mdl-divergences-distances.md`, each by label and each restated in one clause. It stands on its own when reached by reference, with the exception of the "atomless case of the remark after the definition" back-references.
- **Boundary with the next section is the main chapter-level issue.** Histogram estimation is taught here and recapped there; the loss certificate is taught here and re-derived there as Barber-Agakov; the shifted-PMI exercise sits there while PMI is taught here. Draw the line once: this section computes with known distributions, the next one estimates.
- **Proportion.** Capacity and rate-distortion are classical material whose only customer in the chapter is the bottleneck. If the chapter needs trimming, capacity is the cheapest cut in this half.
- **Suspected chapter-wide patterns.** (a) Every result set as "Proposition, Proof" regardless of weight; (b) a "We can now ..." recap paragraph before a summary that lists every result with its number; (c) cells that confirm an identity just proved, introduced by a question whose answer is yes; (d) `\ln` and `\log` mixed. On the positive side, (e) small running examples reused across subsections and (f) correct, checkable numbers throughout; if the other sections share these, the chapter is in better shape than Basics was on the experiments-versus-claims axis.
- **Deep learning connection.** Strong for PMI, DPI, Fano, the certificate and the bottleneck; absent for capacity; and the VAE link, promised by the chapter introduction, is one sentence away from being made.
