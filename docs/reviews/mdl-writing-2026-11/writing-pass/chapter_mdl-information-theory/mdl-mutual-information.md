# Writing pass: Mutual Information

File: `chapter_mdl-information-theory/mdl-mutual-information.md`. Not split.
Checker: "OK: only prose changed", no em-dash warning; `tools/lint_source.py`
clean. Numeric checks were done by hand or with `python3` (standard library
`math` only; no SciPy, no PyTorch on this machine), and against the stored
PyTorch outputs in `outputs/pytorch/chapter_mdl-information-theory/`.

## Words

- Checker count (prose and exercises): 7,948 before, 7,957 after (+9). This
  counter counts words of inline math that is broken across lines, so it moves
  with re-wrapping.
- Counted paragraph by paragraph with inline math removed: body 6,462 to
  6,429; exercises unchanged at 1,399.
- The section is only marginally shorter. The cuts (long Gaussian proof, four
  Proposition wrappers, histogram prose, recap closing, inventory summary,
  repeated numbers) removed about 600 words; the three deep-learning points,
  the repaired hypotheses, the main-text references and the ranking sentences
  put about 570 back.
- Display equations: 49 to 45 delimiters (the two displays of the old Gaussian
  proof are gone).

## Correctness changes

- **MI-6.** Was, directly after the first $2\times2$ computation: a
  measure-theoretic paragraph ("density with respect to the product of the
  marginals"), "$Y=\textrm{sign}(X)$ has $I=\ln2$" with no law for $X$, and
  "the next proposition gives $I\le H(Y)$ whatever $X$ is". Now: the paragraph
  sits at the end of "The Gaussian Pair", after $\rho\to\pm1$, and says that
  $I$ is infinite when $Y$ is a function of $X$ whose distribution has no
  atoms, that it cannot be when $Y$ is discrete ($I=H(Y)-H(Y\mid X)\le H(Y)$,
  with $H(Y\mid X)$ the average entropy of $p(\cdot\mid x)$), and that the sign
  of a *standard normal* $X$ carries $\ln2$. Verified by hand: the graph of
  $f$ has product measure $\int P_Y(\{f(x)\})\,dP_X=0$ when $P_Y$ is atomless.
  The two later back-references ("the atomless case of the remark after the
  definition") now cite `subsec_mdl-mi-gaussian`.
- **MI-7.** The Gaussian proof is now the two-line route
  $I=h(Y)-h(Y\mid X)$ with conditional variance $\sigma_Y^2(1-\rho^2)$,
  followed by "the fraction of the variance of $Y$ that $X$ leaves
  unexplained". Differential entropy is cited
  (`subsec_mdl-differential-entropy`) and not re-introduced. Checked the
  quoted values: $-\tfrac12\ln(1-\rho^2)$ = 0.144, 0.830, 1.959 at 0.5, 0.9,
  0.99.
- **MI-10.** The invariance proposition is stated for invertible maps without
  smoothness; the proof says it proves the smooth case, and one sentence after
  the data-processing inequality gives the general case (two Markov chains).
  Not circular: the DPI proof uses only the chain rule.
- **MI-11.** The $x^3$ cell is no longer asked whether the mutual information
  changes. The text states before the cell that $I$ stays at 0.830 by the
  proposition and that the correlation becomes $3\rho/\sqrt{15}=0.697$
  (checked: $\mathbb{E}[X^3Y]=\rho\,\mathbb{E}[X^4]=3\rho$,
  $\textrm{Var}(X^3)=15$), and asks only what the histogram does.
- **MI-12.** $I(X;Y\mid Z)=0.248$ now comes after the chain rule, as
  $0.368-0.120$, with "here" and a pointer to the XOR case; $H(X\mid Y,Z)$ is
  glossed.
- **MI-13.** In the DPI bullet: depth is not harmful (layers rearrange
  information; a linear probe can improve although $I(Y;Z_\ell)$ does not;
  exercise cited), and $I(X;Z_\ell)$ is "typically infinite" for a
  deterministic layer of a continuous input, or $H(Z_\ell)$ after
  quantization. "Typically" because a layer with a constant or discrete
  output is the exception.
- **MI-14.** Equality in DPI is now stated as "$X\to Z\to Y$ is also Markov",
  with sufficiency as the deterministic case, and "$\theta$, given a prior".
- **MI-16.** The rate-distortion proof is one Proof block: lower bound, "It
  remains to exhibit an encoder", achievability, "rescale", then the proof
  mark. Added that $D=\sigma^2/(1+\sigma^2)$ takes every value in $(0,1)$,
  and the reason for $\mathbb{E}[X\mid Z]=Z/(1+\sigma^2)$.
- **MI-17.** Was: "the information kept equals $R(D)$ at all three noise
  levels. In numbers, $D=0.1\sigma_X^2$ costs 1.15 nats ... $0.5\sigma_X^2$
  costs 0.35". Now: the rate columns "are equal by algebra, not by
  measurement", and the numbers are the cell's rows ($D=0.09$: 1.20;
  0.5: 0.35; 0.75: 0.14; stored output 1.1989, 0.3466, 0.1438).
- **MI-19.** "The same substitution" as the ELBO is gone. Now: the bound
  $\mathbb{E}_xD_{\textrm{KL}}(p(z\mid x)\|r(z))$ is the KL penalty of a
  variational autoencoder, hence an upper bound on the rate, and its weight
  trades rate against distortion; a parenthesis warns that there the weight
  multiplies the rate while $\beta$ in the bottleneck multiplies the label
  term.
- **MI-4.** The two-sided bound on $H(X,Y)$ is no longer asserted before it
  can be shown: lower bound at the chain rule, upper bound after the
  properties proposition.
- **MI-2, MI-1, MI-5, MI-15, MI-18, MI-20 (first half), MI-21** applied as
  the review asks. MI-9: pointer to the shifted-PMI exercise added.
- Exercise 4(a): fixed from the body side; the sentence "No Markov chain
  holds among the three" is deleted, so the exercise has a question again.
- No finding disputed. Not done: MI-3 (imports cell, code), MI-20 second half
  (I dropped the sentence on subscripts for length), exercise label prefixes
  (labels may not be renamed).

Numbers in prose that no cell prints, kept after recomputation: 0.144 and
1.959 (Gaussian MI at $\rho=0.5$, 0.99; the second is printed by the
histogram cell); 0.697; 0.248 as a difference; 0.26; "two thirds"
(1 - 0.120/0.368 = 0.67); "92%" (6.364/6.908); "well under half"
(0.146/0.368 = 0.40); 1.235; 0.725, 0.484, 2.079, 0.798 (printed by the
bottleneck cell).

## Before and after

1. Before: "Estimating mutual information from samples alone is the open
   problem that :numref:`sec_mdl-mi-estimation` takes up."
   After: "Estimating mutual information from samples alone is a different
   and harder problem, which :numref:`sec_mdl-mi-estimation` takes up."
2. Before: "The next cell checks that the divergence form and the entropy
   identity agree on the table."
   After: "The function `mutual_information` implements the divergence form
   for any table; we reuse it throughout."
3. Before: "Does the invertible map $x\mapsto x^3$, which is smooth with a
   smooth inverse except at $0$, change the correlation, the histogram
   estimate, or the mutual information of the running Gaussian pair?"
   After: "The mutual information stays at $0.830$ nats by the proposition.
   The correlation falls to $3\rho/\sqrt{15}=0.697$ ... What will the
   histogram do?"
4. Before: "Thus no processing of $Y$, deterministic or random, can increase
   the information it carries about $X$."
   After: "No processing of $Y$, deterministic or random, can increase the
   information it carries about $X$."
5. Before: "The maximization matches the input to the channel."
   After: "A sender who transmits mostly zeros wastes the channel, since such
   an input has little entropy to convey."
6. Before: "Does a direct minimization over the noise scale agree with the
   closed form, and does the collapse happen where the proposition says? The
   next cell minimizes the Lagrangian on a grid of $4{,}001$ noise scales."
   After: "A threshold this sharp invites a test: the proposition predicts
   that the encoder keeps nothing at $\beta=1.2$ and something at
   $\beta=1.25$. We minimize the Lagrangian directly on a grid of $4{,}001$
   noise scales, without using the closed form."

"The next cell" occurred six times and now does not occur.

## Repetition and ownership

- **Loss certificate: owned here, kept.** The proposition and
  :eqref:`eq_mdl-info-certificate` stay; one sentence names it as the bound of
  Barber and Agakov (`Barber.Agakov.2003` is in `d2l.bib`) and points to
  `sec_mdl-mi-variational-bounds`.
- **Histogram estimation: owned by the estimation section.** The three cells
  stay. Prose around them went from about 330 words to about 190: plug-in
  bias at $\rho=0$ and the long bin-geometry explanation are gone; the first
  cell points to `subsec_mdl-mi-high-dim`.
- **Joint and conditional entropy.** Definitions and labels kept; the
  image-and-label motivation and "nothing new in this definition" are gone. I
  did not cite the short definitions the parallel writer is adding to the
  entropy section, since I cannot see their label.
- **Worked numbers.** 0.368 is derived once (the KL display). It is cited,
  not re-derived, at the chain rule, capacity, and the certificate. The
  re-derivations at the entropy identity and at pmi are gone. 0.120 is
  derived once (two flips make one), 6.36 once (after the Fano cell). The
  Summary quotes none of the three.
- **Closing.** The "We can now ..." paragraph is cut to its last two
  sentences. The Summary is rewritten as answers (about 130 words, from
  about 165).
- Differential entropy before the Gaussian pair: reduced to a clause with a
  reference.

## Ranking

Proposition blocks: 14 before, 10 after. Kept: properties of mutual
information, Gaussian mutual information (label cited by estimation),
reparameterization invariance, chain rule for mutual information, data
processing, Fano, the loss certificate, Gaussian rate-distortion, the
bottleneck identity, operating point and collapse.

Demoted to running text, equation labels kept: chain rule for entropy;
capacity of the binary symmetric channel (:eqref:`eq_mdl-bsc-capacity`);
Gaussian channel (:eqref:`eq_mdl-gaussian-channel`, now a step of the
rate-distortion proof); collapse for $\beta\le1$.

Judgments added: the opening names the three results that carry the section;
"Processing and Prediction" says its two guarantees are the main results; the
Fano proof "needs one idea, the error indicator"; in rate-distortion "the
lower bound is the harder half"; capacity "can be skimmed" on a first reading.

Labels added: none. Split markers: none.

## Deep learning points now in the body

- Depth and data processing; infinite $I(X;Z_\ell)$ for deterministic layers:
  first DPI bullet. The bottleneck's "encoders must be stochastic" is now a
  clause at the definition of the encoder.
- VAE KL term as a rate: variational-bottleneck paragraph.
- Main-text references added, each checked with `git grep`:
  `sec_diffusion-discrete` (the bound charges $I(X^1;X^2)$ for independent
  prediction of two masked tokens), `sec_basic_gan` (Jensen-Shannon as the
  mutual information with the source label, at most $\ln2$),
  `sec_diffusion-latent` (the VAE with weight $\beta$; the rate-distortion
  argument for an autoencoder), `chap_diffusion` (noising as a Gaussian
  channel), `sec_lm_word_embeddings` (shifted PMI; this file is still a
  skeleton). Appendix: `subsec_mdl-differential-entropy`,
  `eq_mdl-f-div-dpi`, `subsec_mdl-mi-high-dim`,
  `sec_mdl-mi-variational-bounds`, and
  `ex_mdl-mutual-information-negative-sampling-pmi`.

## Cells to cut or redesign on the development machine

- `mutual-information-gaussian-binned`, `mutual-information-nonlinear-correlation`,
  `mdl-mutual-information-mutual-information-as-nonlinear-correlation-2`
  (MI structural 1, MI-8, MI-11): move to the estimation section.
  `binned_mi` and `gauss_pair` are defined in the first; exercise 1 here uses
  `gauss_pair`. After the move, print the true 0.830 beside the rank-binned
  0.812.
- `mdl-mutual-information-rate-distortion` (MI-17): two columns equal by
  algebra; choose $\sigma^2=1/9, 1, 3$ and print one rate.
- `mutual-information-joint-conditional`, `mutual-information-discrete-joint`
  (pattern note): confirm hand arithmetic; they stay because they define
  `entropy`, `p_xy` and `mutual_information`. Prose is one sentence each.
- `mdl-mutual-information-the-data-processing-inequality-1` (XOR): confirms a
  two-line argument; one sentence.
- `mutual-information-imports` (MI-3): one untagged cell.

## Other deferred findings

- MI structural 4: capacity is kept, shortened by the Proposition wrapper,
  with two named uses (dropout exercise, Gaussian channel). Cutting it
  further is the author's call.
- MI-9, second option (move the shifted-PMI exercise here): moving material
  between files is out of scope.
- Exercise on the VAE reading of the rate bound: new material.
- Exercise labels of 7 and 8 keep the `ex_mdl-information-theory-` prefix.

## Claims added or changed

1. Mutual information is the extra coding cost per pair of a model that
   treats $X$ and $Y$ as independent (KL as excess cross-entropy).
2. $H(X,Y)\ge\max\{H(X),H(Y)\}$ for discrete variables, with equality when
   the other variable is a function of the one.
3. $I(X;Y)\le H(Y)$ for discrete $Y$ and any $X$; hence the Jensen-Shannon
   divergence is at most $\ln2$.
4. Given $X=x$, $Y$ is Gaussian with variance $\sigma_Y^2(1-\rho^2)$.
5. Invariance holds for any invertible map, by data processing in both
   directions.
6. A linear classifier can do better on $Z_2$ than on $Z_1$ although
   $I(Y;Z_2)\le I(Y;Z_1)$ (exercise 3 is the example).
7. $I(X;Z_\ell)$ for a deterministic layer: typically infinite; $H(Z_\ell)$
   after quantization.
8. The certificate is tight only for the true conditional; it is the
   Barber-Agakov bound with the roles of the variables exchanged.
9. Dropout is a memoryless channel; the noising step of a diffusion model is
   a Gaussian channel; :eqref:`eq_mdl-gaussian-channel` is that channel's
   capacity at unit power (exercise 9).
10. The VAE's KL penalty is an upper bound on $I(X;Z)$, loose by
    $D_{\textrm{KL}}(p(z)\|r(z))$.
11. A minimal sufficient statistic is "the coarsest function of $X$ that
    still predicts $Y$ as well as $X$ does".
12. An input with $P(X=1)=0.1$ gets "well under half" of the capacity through
    (0.146 of 0.368).

## Slides that now disagree with the body

- "$D=0.1\sigma_X^2$: $1.15$ nats": correct, but the body now quotes the
  cell's rows (0.09: 1.20).
- The slides do not use Proposition blocks, so the demotions change nothing
  there. No number on the slides contradicts the body.
