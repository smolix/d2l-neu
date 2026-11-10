# Chapter 42, Information Theory and Divergences: writing review

Sources: the seven `.md` files of `chapter_mdl-information-theory/`, each read in full by a section reviewer up to the slides marker. About 39,000 words of prose and mathematics and 9,900 words of exercises, 76 code cells, 52 exercises. This file is assembled from the section reviews; I opened the sources only to check the cross-section claims marked "checked" below.

Section reviews:
[index](42-information-theory/index.md) ·
[mdl-information-theory](42-information-theory/mdl-information-theory.md) ·
[mdl-coding-compression](42-information-theory/mdl-coding-compression.md) ·
[mdl-divergences-distances](42-information-theory/mdl-divergences-distances.md) ·
[mdl-mutual-information](42-information-theory/mdl-mutual-information.md) ·
[mdl-mi-estimation](42-information-theory/mdl-mi-estimation.md) ·
[further-reading](42-information-theory/further-reading.md)

Finding IDs: `IDX` (index), `INF` (entropy section), `COD` (coding), `DD` (divergences), `MI` (mutual information), `EST` (estimation), `FR` (further reading). `IDX` and `FR` recur in other chapters; here they always mean the files above. Structural findings carry the tags `INF-S1` to `INF-S5` and `COD-S1` to `COD-S4`; in the other three files they are numbered without a prefix and are cited here as "DD S4", "MI structural 1", "EST structural 1".

## Verdict

This is the strongest kind of chapter the appendix can have: correct throughout, built on a handful of small examples that are reused until they are familiar, and tied to deep learning by numbers the reader can check. Four reviewers recomputed every worked number and checked every proof and found no mathematical error, only loose hypotheses and one false remark about notation. Against the author's named models it stands as follows. Two sections can be put beside them without embarrassment. The coding section takes MacKay's central point (arithmetic coding separates the model from the coder) and reaches it by a route better suited to this reader than either book's, codes first and the typical set last; its reviewer would be happy to find its pedagogy in MacKay. The estimation section has no counterpart in either book and has what both books have and most of this appendix lacks, one idea stated by the authors under four results ("an estimator cannot report more than the logarithm of the number of things it can tell apart"). The first half of the entropy section is the Cover and Thomas method applied faithfully: meaning before formula, a convention with its reason, a number at once. The chapter falls short of the models in three ways. It does not keep Cover and Thomas's order: joint entropy, conditional entropy and the chain rule are defined in the fourth section and used in the second. It does not have MacKay's economy: Gibbs' inequality is proved in chapter 41 and again here and then applied five times as if new, the divergences section is a 15,700-word survey with ten propositions of equal weight, and the mutual-information section sets fourteen results in the same dress. And outside the two best sections the prose is clean and even, with few sentences in which an author says what is surprising or what matters. What matters most is therefore structural and cheap relative to its effect: decide the home of each shared result (chapter 41 against section 1; mutual information against estimation), move the definitions forward, split and cut the divergences section, and rank the results.

## Scores

As given by the section reviewers; 5 means the reviewer would be happy to find the page in Cover and Thomas or MacKay.

| Section | Explanation | Thought | Pedagogy | Language | Enjoyment |
|---|---|---|---|---|---|
| [index](42-information-theory/index.md) | 4 | 4 | 4 | 4 | 4 |
| [mdl-information-theory](42-information-theory/mdl-information-theory.md) | 4 | 3 | 4 | 4 | 3 |
| [mdl-coding-compression](42-information-theory/mdl-coding-compression.md) | 4 | 4 | 5 | 4 | 4 |
| [mdl-divergences-distances](42-information-theory/mdl-divergences-distances.md) | 4 | 4 | 3 | 4 | 3 |
| [mdl-mutual-information](42-information-theory/mdl-mutual-information.md) | 4 | 4 | 4 | 4 | 3.5 |
| [mdl-mi-estimation](42-information-theory/mdl-mi-estimation.md) | 4 | 4.5 | 4 | 4 | 4 |
| [further-reading](42-information-theory/further-reading.md) | 4 | 4 | 3 | 4 | 3 |

Calibration. The reviewer of the last two sections used half points; the others did not. The scores are otherwise consistent with the verdicts. One difference of standard is visible in the text and not in the numbers: the divergences reviewer counts "every cell prints a number the prose has already derived" as a strength, while the entropy reviewer counts six such cells as a weakness (INF-S5). The two agree on the rule underneath (a confirming cell earns its place only if the prediction could have failed), and the divergences reviewer applies it too in DD-29 and DD-31.

## Correctness

No finding of substance: no false theorem, no proof that fails, no stored output that contradicts the prose. What the reviewers found is loose hypotheses, over-broad sentences, and numbers the reader cannot see. In order of severity:

Statements that are false or too broad as written.

1. **DD-22** (moderate; sure). "The section uses a dozen symbols, each with one meaning" is false: $d$, $\delta$, $n$, $\mathbf{K}$ and $\kappa$ each have several.
2. **FR-6** (moderate; fairly sure). "Every estimator of :numref:`sec_mdl-mi-estimation` is a lower bound" is contradicted by the KSG estimator of that section; true only of the variational estimators.
3. **MI-6** (moderate; confident). $Y = \textrm{sign}(X)$ has $I = \ln 2$ only if $P(X > 0) = \tfrac12$, and no distribution is given; the bound $I \le H(Y)$ for continuous $X$ is attributed to a proposition whose hypotheses exclude that case.
4. **DD-20** (moderate; fairly sure it is loose, not wrong). "For such *characteristic* kernels": no property has been stated that makes a kernel characteristic, and "compares moments of every order" does not imply the conclusion.
5. **INF-13** (major, tagged thought; a false statement about the book). The text says chapter 41 "used the answer" for Gaussian maximum entropy; chapter 41 proved it. Checked in the source (`mdl-information-theory.md`, line 665).
6. **EST-17** (minor, tagged thought). The summary's "can be bounded from below but not measured" contradicts the section's own KSG result, accurate to 0.1 nat in low dimension. **EST-16** (minor, tagged thought): the summary states a result for one scalar Gaussian pair as a general fact.
7. **COD-14** (minor; confident). "So every rate below $H$ fails on almost every message" was proved only for fixed-length block codes.
8. **INF-15** (minor; confident). The floor of 0.545 nats and "best accuracy is 90%" need the clean label to be a function of the input; exercise 7 states the hypothesis, the text does not.
9. **MI-14** (minor; confident). "A many-to-one or noisy map loses nothing exactly when it keeps a sufficient statistic: a $T(Y)$": a noisy map is not of the form $T(Y)$, and $\theta$ needs a prior for $I(\theta; \cdot)$ to be defined.
10. **DD-30** (minor, tagged thought). Noise in diffusion models is given one reason as "the" reason. **DD-17** (minor, tagged thought): the attribution of Pinsker's constant omits Kullback and Kemperman, and an experiment is said to ask whether a proved inequality "is ever violated".

Hypotheses and proofs that do not cover what they are used for (the conclusions hold).

11. **DD-5** (minor; sure). $f$ is defined on $(0, \infty)$ and evaluated at $0$ a page before $f(0)$ is defined.
12. **DD-14** (minor; sure). The f-GAN optimal critic is applied to disjoint supports, where the proposition's hypothesis $P \ll Q$ fails.
13. **DD-16** (minor; sure). Pinsker's Step 1 is for $a, b \in (0, 1)$; Step 2 uses it at $0$ and $1$.
14. **DD-25** (minor; sure). The Fubini justification is attached to the step that does not need it and missing from the one that does.
15. **DD-21** (minor; sure). The dimension convention drifts (scalar MMD, $\mathbb{R}^n$ for $W_1$, "In $d$ dimensions" for the Stein kernel).
16. **EST-9** (minor; confident). The Donsker and Varadhan proposition is titled a representation and states an inequality; $\mathbb{E}_P[T]$ should be assumed defined.
17. **EST-11** (minor, tagged explanation). InfoNCE Step 3 divides by a $p(v)$ that is the marginal law of the bag only by an argument the proof does not give.
18. **MI-10** (moderate, tagged thought; confident). Reparameterization invariance is stated with smoothness hypotheses the result does not need; the two-line proof by data processing gives the general case.
19. **Exercise 8(a) of the coding section** (reference; confident; no ID, see "Exercises" in that file). It cites :eqref:`eq_mdl-wrong-code-length` where it uses :eqref:`eq_mdl-ce_decomp`.

Numbers no cell prints, and readings the output does not support. Every one was reproduced by the reviewer, so none is wrong.

20. **EST structural 1, EST-4, EST-13** (moderate). About ten quantitative claims in the estimation section have no stored output: KSG at $d = 2$ and $d = 8$ (with the admission "the cell does not print that case"), the mean strip counts 70, 31, 26, 26, the median of the batch average of $e^{\textrm{pmi}}$, "about one batch in four", and three values read off a plot to a precision it does not offer.
21. **MI-17** (minor). The rate-distortion cell's two agreeing columns are equal by algebra, and the numbers quoted after it ("$D = 0.1\sigma_X^2$ costs 1.15 nats") are not the cell's rows.
22. **MI-11** (moderate). The $x^3$ cell is asked whether the map changes the mutual information and never computes the true value after the map; the rank-binned 0.8122 is printed and not compared with 0.830.
23. **COD-9** (minor). The entropy 4.078 of the letter distribution is quoted and not computed.
24. **INF-18** (moderate). "The raw gradient decays like $1/\tau^2$" is not shown; the table prints only the scaled gradient.
25. **COD-8** (moderate). The proposed addition (compression rates with the model's size counted) rests on the reviewer's recollection of the paper and must be checked by the reviser.

## Arc and order

The chapter has one line of argument, and the introduction states it: the surprise of one outcome, its average, the cost of the wrong model, the same quantities as file sizes, what to do when KL fails, how much one variable tells about another, and how much of that can be measured. The hand-offs are real. The entropy section ends on a code it has not built and the coding section builds it; the coding section ends on distributions known only through samples, which "give no code to measure"; the mutual-information section ends on densities that images and text do not have, and the estimation section opens there.

Four things break the line.

- **Definitions come after their first use.** Joint entropy, conditional entropy and the chain rule are defined in the fourth section ("Joint and Conditional Entropy", `mdl-mutual-information.md`) and used in the second, in "The Entropy Rate" (COD-S1, COD-12, INF-S4). Checked: the coding section does give conditional entropy a one-clause definition as an expected log-loss, so COD-S1's "no definition of either symbol" is slightly too strong, but the block entropy $H(X_1, \ldots, X_n)$, the chain rule and "conditioning cannot increase entropy" are used with none. "Surprisal" is used about ten times from the second section on and never appears in the first (INF-4, COD-3; checked, zero occurrences). The entropy section promises that the coding section "proves" the bound on the number of questions, and the coding section never mentions questions (COD-S2).
- **The divergences section interrupts.** It sits between coding and mutual information, is the longest section of the part, and the mutual-information section needs from it only data processing. Its reviewer proposes three sections along the existing `##` boundaries (DD S1): f-divergences, with total variation, Pinsker, data processing and the f-GAN bound; test functions and transport; scores. The f-divergence half fits where the section is now. The transport half can stay or follow the two mutual-information sections. The score material (Fisher divergence, Stein's identity) is used only by the dynamics chapter and the diffusion chapter and fits better as the opening of the score-matching section there.
- **The boundary between mutual information and its estimation leaks in three places.** The mutual-information section declares "Throughout this section the distributions are known" and then runs three histogram experiments, which the estimation section recaps with their numbers in a subsection that has no cell of its own (MI structural 1, MI-8, MI-11, EST structural 4, EST-1). Checked: `binned_mi` is defined only in `mdl-mutual-information.md` and is called by exercise 1 of the estimation section. The loss certificate $I(X;Y) \ge H(Y) - L$ is a proposition in one section and is derived again as the first instance of the Barber and Agakov bound in the next (MI structural 5, EST structural 5, EST-8). Checked: `mdl-mi-estimation.md` does not cite :eqref:`eq_mdl-info-certificate`, and `mdl-mutual-information.md` does not name Barber and Agakov. The shifted-PMI exercise sits in the estimation section while PMI is taught in the other (MI-9). Both reviews propose the same rule: known distributions in one section, samples in the other.
- **The second half of the entropy section is not on the line.** After the decomposition it applies Gibbs' inequality five times (INF-S3) and spends 800 words on distillation, whose information-theoretic content is one sentence (INF-S4). Differential entropy interrupts the discrete argument and uses KL before defining it (INF-S2, INF-7).

Sections that are really two or three: divergences (three, DD S1). Not to be split: coding (COD-S4), mutual information (MI structural 7). A heading that is not a section: "The Compression-Phase Debate", a top-level `##` of two paragraphs (EST structural 6); "Divergences and Metrics", a top-level `##` of five paragraphs (DD-4).

Length is not proportionate to use. The divergences section takes about a third of the chapter; the subsections the main text cites take about half of it (DD S2). The estimation section, the shortest, has nothing to cut.

The section reviews disagree on one point of order. The entropy and coding reviewers ask that joint and conditional entropy move into the first section, as Cover and Thomas have it (INF-S4, COD-S1). The mutual-information reviewer notes the same October request and writes "with the present chapter introduction that order is defensible, and I do not repeat the request". These are compatible in practice: a short subsection in the first section (definition, chain rule, conditioning does not increase entropy) serves the coding section and the irreducible-loss floor, and the fourth section keeps its fuller treatment with the flip-channel table and cites the first.

## The most important findings

1. **Give entropy, Gibbs' inequality and maximum entropy one home each between chapters 41 and 42.** Chapter 41 defines entropy, KL and cross-entropy, proves Gibbs' inequality twice (a proposition in `mdl-maximum-likelihood.md`, a lemma in `mdl-exponential-family.md`; checked) and proves maximum entropy with the uniform, Gaussian and geometric cases. The entropy section proves all of it again and says chapter 41 "used" it. Recommended by the reviewer: this chapter owns entropy and Gibbs; chapter 41 owns the maximum-entropy proposition. INF-S1, INF-13, INF-1, INF-5, IDX-4, and structural finding 1 of the index file. This needs a decision that binds the revisers of both chapters.
2. **Move the missing definitions into the first section and close the questions game.** A short "Joint and Conditional Entropy" subsection, the word "surprisal" at the definition of self-information, and one sentence that a questioning strategy is a binary code. The space comes from distillation. INF-S4, INF-4, COD-S1, COD-12, COD-3, COD-S2.
3. **Split the divergences section and cut about a quarter.** Three sections as above; the kernel Stein discrepancy to one paragraph in further reading; Sinkhorn to a short derivation; the $\alpha$-divergence paragraph out; data processing in its own labelled subsection, since four chapters cite it through the wrong label or through the label of the whole section. DD S1, DD S2, DD S4, DD-32, DD-27, DD-8, DD-15.
4. **Draw the line between mutual information and estimation once.** Histogram experiments to the start of the estimation section; the certificate stated once and named as the Barber and Agakov bound; the shifted-PMI exercise next to PMI. MI structural 1, MI structural 5, MI-8, MI-9, MI-11, EST structural 4, EST structural 5, EST-1, EST-8.
5. **In the estimation section, print what is claimed and predict before the table.** The rare-pairs argument is strong enough to predict the comparison of the three bounds; it now comes after twelve numbers. EST structural 1, EST structural 2, EST-4, EST-10, EST-13, EST-15. The same habit in smaller form: MI-17, COD-9.
6. **Answer the three questions the introduction asks.** "1.2 nats per character, is that good?" is never answered with that number, though the coding section has the material (IDX-1, COD-13). "With a KL penalty in a variational autoencoder ... what exactly are we measuring?" is one sentence away from an answer in the bottleneck subsection and is not given one (MI-19, MI structural 3; checked, the mutual-information section does not contain the words "variational autoencoder"). And the data-processing inequality is left reading as if depth destroyed information, with the correction in an exercise (MI-13).
7. **Rank the results.** Ten propositions in the divergences section and fourteen in the mutual-information section arrive with the same weight; in the entropy section a proposition's proof begins by saying it is the previous proposition. Say which results the chapter exists for (Gibbs; the entropy bound for codes; the f-GAN bound and Kantorovich and Rubinstein duality; data processing, Fano and the certificate; the InfoNCE ceiling) and fold one-line algebra into the text. DD S6, MI structural 2, INF-S3, INF-17, COD-6.
8. **Point back to the main text by label.** The divergences section names GANs, WGAN, diffusion and variational inference and has no reference to any main-text label; the perplexity subsections repeat `language-model.md` at the same level; FID and KID, which are a $W_2$ and an MMD, are not mentioned. DD S3, DD-26, COD-S3, FR-7, FR-1, FR-2.
9. **Repair the places where notation or a word misleads.** The roles of $P$ and $Q$ are swapped without notice from Stein's identity on (DD-28, the one place a careful reader will be misled). $H_{\textrm{rate}}$ is in bits in one equation and in nats in another that the main text cites (COD-11). "Generator" means the convex function and the network in the same paragraph, and the GAN chapter resolved the clash the other way (DD-2). "Certify" carries three meanings in the estimation section, and the section never reconciles "a loss certifies information" with its own distribution-free limit (EST structural 3).
10. **End each section once.** All five sections close with a "We can now ..." recap followed by a summary that lists the same results with the same numbers. In every case the reviewer found a real closing in the last two or three sentences of that paragraph and asks that it be kept alone. INF-19, COD-17, DD-35, MI-21, MI structural 6, EST-18, INF-20, COD-18.

## Repetition

Inside the chapter.

| What | Where it is taught | Proposed home |
|---|---|---|
| Gibbs' inequality applied as a new result | five times in the entropy section: cross-entropy bound, irreducible loss, proper scoring, label smoothing, distillation (INF-S3, INF-17) | one proposition (the irreducible loss); the rest one clause each |
| Units of differential entropy against ratios of densities | four statements in the entropy section (INF-S2, INF-8); again before the Gaussian pair (MI-7) | "Differential Entropy", moved after "Cross-Entropy" |
| Uniform and Gaussian maximum entropy | two unrelated proofs in the entropy section (INF-5) | one paragraph after Gibbs |
| The loss certificate | proposition in mutual information; first instance of Barber and Agakov in estimation (MI structural 5, EST-8) | mutual information; estimation cites the equation |
| Histogram estimation | three cells in mutual information; recap with numbers in estimation (MI structural 1, EST-1) | estimation |
| Every f-divergence is constant on disjoint supports | four tellings in the divergences section (DD S5, DD-1, DD-23) | after the conventions; the others a clause |
| Total variation | generator table and, 450 lines on, as a supremum (DD S4) | the f-divergence part |
| Fisher information as the local quadratic | "The Local Quadratic" and "Information Geometry", with two symbols (DD-9, FR-3) | the divergences section; further reading keeps Chentsov's uniqueness |
| Rényi divergence | exercise 1(d) of divergences and a further-reading entry (FR-4) | the exercise; the entry starts from privacy |
| Sinkhorn divergences | exercise 11 of divergences and a further-reading entry (FR-5) | one of the two |
| The proof of the distribution-free limit | the body and exercise 6(a) to (c) of estimation, numbers included (EST-5) | choose one |
| The perplexity floor | the body and exercise 10(a) of coding | the body |
| Worked numbers | 0.545 five times, $\ln 10$ four times, $\ln 91$ three times (INF-16); 0.368, 0.120 and 6.36 three or four times (MI structural 6) | derive once |

With other appendix chapters.

- Chapter 41: entropy, KL, cross-entropy, Gibbs (twice there), maximum entropy with three corollaries; exercises 4 and 5 of the entropy section restate chapter 41's text (INF-S1). Home as in finding 1 above.
- Chapter 41: the Gaussian maximum-likelihood fit is redone as "forward KL fits moments" (DD-11). Home: chapter 41; two sentences here.
- Chapter 43: the Fisher divergence and Stein's identity (DD S1). Home: the score-matching section of the dynamics chapter.

With the main text.

- Perplexity as a branching factor, across tokenizers, and bits per byte: `chapter_recurrent-neural-networks/language-model.md` (COD-S3). Home: the main text; the appendix keeps the floor.
- Lipschitz critics: nearly the same paragraph in the GAN chapter's further reading (FR-2). Home: the GAN chapter.
- Reliability diagrams: the linear-classification chapter (FR-1). Home: the main text; cite by label.
- Logits against probabilities in a cross-entropy loss (INF-14, INF-S5). Home: the main text.
- The disjoint-support argument, told a fifth time in `objectives.md` with a labelled reference (DD S5). That one is correct as it is.

## Recurring patterns

**"We can now ..." closing plus inventory summary.** Five of five sections (INF-19, COD-17, DD-35, MI-21, EST-18). The entropy section also opens a subsection with the phrase and announces in its opening the two questions its closing will leave open (INF-1).

**"Thus".** Eleven sentences in the coding section, twelve in the divergences section, each the plain restatement after a result (COD-16, DD-34). Individually good; together a template.

**A question before every cell.** Five of seven computing cells in the coding section (COD-16); four questions and six "The next cell ..." in mutual information (pattern note in that file); ten "The following cell", four of them asking, in divergences (DD-34). The reviewers agree on the rule: keep the question where the answer is not known in advance.

**"The step that matters."** In all three proofs of the entropy section, in the Kraft proof, and in chapter 41 (INF-10). The judgement is right; the fixed phrase is not.

**Verification cells that cannot fail.** Six of thirteen cells in the entropy section confirm hand arithmetic (INF-S5, INF-3, INF-14). Stein's identity is checked on the distribution for which it holds (DD-31). A helper is unit-tested in the text (DD-29). Two closed forms equal by algebra are compared to four digits (MI-17). "Does the formula we just proved hold?" (pattern note in the mutual-information file). The remedy the reviewers give is the same each time: test an identity by breaking it, or drop the cell.

**Claims beside a cell that does not print them.** About ten in the estimation section (EST structural 1), one in coding (COD-9), one in mutual information (MI-17). All reproduced.

**Every result a proposition.** Ten, fourteen and four in the last three sections, plus a "Proof" with no proposition (COD-6) and a proposition proved by the previous one (INF-17).

**Section openings that restate the previous section.** "The previous section" seven times before the first heading of the estimation section (EST structural 4); the divergences section runs 1,100 words before its first heading (DD-3).

**Framework import tabs on NumPy sections.** Coding, divergences and mutual information load four frameworks they do not use (COD-19, DD-36, MI-3). Only the entropy and estimation sections have framework code.

**Exercises.** Out of text order in the entropy and coding sections. Nine references in the divergences exercises name a cell by its source identifier, which the rendered page does not show. Leftover or inconsistent exercise labels in two sections. `binned_mi` called where it is not defined. Two references to the retired NLP Pretraining chapter (estimation, exercise 9).

**Notation drift.**

- $-\log p$: "self-information" and "surprise" in the first section, "surprisal" in the second and in the main text (INF-4, COD-3).
- Binary entropy: $H_2(0.9, 0.1)$, $H_b$, and $H_b/\ln 2$ (COD-11).
- $H_{\textrm{rate}}$: bits and nats (COD-11).
- `\ln` and `\log` both natural, alternating (MI-20).
- $P$ and $Q$: data and model, then model and data (DD-28).
- $d$, $\delta$, $n$ against $N$, $\mathbf{K}$, $\kappa$ (DD-22); $\delta$ is also the flip probability in mutual information and the failure probability in estimation (EST-5); $\sigma$ is a stay probability in one coding exercise.
- Fisher information: $\mathbf{I}$ in the divergences section and chapter 41, $\mathbf{F}$ in further reading (FR-3).
- The maximized function: critic, test function, auditor, witness (DD-19). "Generator": $f$ here, the network in the GAN chapter, which calls $f$ the "divergence function" (DD-2). Density ratio: $u$ here, $\rho$ in the GAN chapter.
- "Bits per character" here, "bits per byte" in the main text (COD-S3).
- Conditional densities $p(y \mid x)$ beside subscripted joints $p_{X,Y}$ (MI-20).

**What is consistently good.** Small running examples reused across subsections (the dyadic source, the pair on three outcomes, the coin at 0.9, the flip channel, the Gaussian pair, the point masses at distance $d$). Predict, then compute. Accurate tolerances for unseeded cells (200 reruns in the divergences review). Candour about what is granted and not proved. The exercise sets, which three reviewers independently call the best they saw in the part.

## Connection to the main text

From the main text into the chapter.

- **Stale.** `chapter_linear-classification/softmax-regression.md` says Gibbs' inequality is proved in `sec_mdl-information_theory` "together with the interpretation of entropy as the length of the shortest code". The code interpretation now lives in `sec_mdl-coding-compression` (index file, "For the chapter review"; checked at line 385 of that source).
- **Too coarse.** `relativistic.md` cites data processing through `sec_mdl-tv-pinsker`; within the chapter, the mutual-information and estimation sections cite data processing and the f-GAN bound through the label of the whole divergences section, four times (DD S4; checked). Data processing needs its own label.
- **Unit mismatch.** A reader who follows :eqref:`eq_mdl-ppl-floor` and then :eqref:`eq_mdl-entropy_rate` gets nats and bits for one symbol (COD-11).
- **The part introduction** describes the chapter as "entropy, cross-entropy, and the KL divergence that most losses descend from", which covers one section of five (index file).
- **Retired targets.** `subsec_negative-sampling` and `sec_word2vec_data` in exercise 9 of the estimation section; the word-embedding plan of the new language-models part expects to find PMI in the mutual-information section (MI-9).
- Labels with no inbound reference from outside the chapter: `sec_mdl-stein-discrepancy`, `subsec_mdl-sinkhorn`, `sec_mdl-what-is-a-divergence`, `sec_mdl-divergence-objective-map` (DD S2).

From the chapter to the main text.

- The divergences section has no labelled reference to `chap_gans`, `sec_gan_objectives`, `sec_gan_convergence`, the diffusion chapter, or the distillation and regularized-RL sections that cite it (DD S3, DD-26, DD-30).
- Uses named without a label: distillation (INF-18; no main-text section teaches it), the variational autoencoder's KL term (MI-19), the bottleneck's link to the ELBO, which is called "the same substitution" and is not (MI-19), CPC, SimCLR and CLIP (EST-12), reliability diagrams (FR-1), natural gradient (DD-9, FR-3).
- No named use at all: channel capacity (MI structural 4), the kernel Stein discrepancy (DD-32), $\alpha$-divergences (DD-8), polar codes (FR-9).
- The index says the chapter assumes chapter 41 and does not say that it re-develops what chapter 41 uses, so a lookup reader cannot tell that it can be read first (IDX-4). The introduction's two "recurring examples" are not the ones the first two sections use (IDX-3).

## What to cut, what is missing

Cut or shorten (about 5,000 words in all, most of it from the divergences section).

- Entropy section, 1,500 to 2,000 words: distillation to a paragraph and its exercise (INF-S4); label smoothing down to the finite logit gap and the nonzero floor (INF-S3, INF-17); the hand-against-built-in loss cells (INF-S5, INF-14); the second maximum-entropy proof (INF-5); exercises 4 and 5; the notation table's seven rows that belong to later sections, or move the table to the introduction (INF-2).
- Coding section, about 400 words: two perplexity subsections to a paragraph each (COD-S3); the Bayesian paragraph of MDL (COD-10); exercise 10(a).
- Divergences section, about a quarter: kernel Stein discrepancy (DD-32); most of the Sinkhorn experiment (DD-27); the $\alpha$-divergence paragraph and table row (DD-8); the measure-theoretic sentence (DD-6); the fourth telling of disjoint supports (DD-23); the moment-matching proof (DD-11); the mixture-score unit test (DD-29); the units paragraph of the opening (DD-3).
- Mutual information, about 700 words: the three histogram experiments (moved, not deleted); capacity down to what the bottleneck needs, unless a use is named (MI structural 4); the recap closing (MI-21).
- Estimation: nothing for length. The plot with eleven lines could go in favour of its table (EST-15).
- Further reading: the Lipschitz-critics entry to a pointer (FR-2); the restatement in "Information Geometry" (FR-3); polar codes to a sentence (FR-9).

Missing.

- Joint and conditional entropy with the chain rule, early (INF-S4).
- The sentence that a questioning strategy is a prefix code, and an exercise that plays the game as a code (COD-S2).
- The answer to "1.2 nats per character" (IDX-1, COD-13).
- The KL term of a variational autoencoder as an upper bound on a rate, and $\beta$ as a rate-distortion multiplier (MI-19).
- That the data-processing inequality does not make depth harmful, and that $I(X; Z_\ell)$ is infinite or meaningless for a deterministic layer, said at the inequality (MI-13).
- The reconciliation of variational "certificates" with the distribution-free limit (EST structural 3).
- What `zlib` does, in a clause (COD-1); the cost of sharing a 70-billion-parameter model, as the motivation for MDL (COD-8, to be verified); the name "Bayesian information criterion" for the $\tfrac12 \log_2 n$ rule (COD-10).
- The original GAN objective derived from the f-GAN bound in three displayed steps (DD-12); the words "mode-covering" and "mode-seeking" (DD-10); a definition of $W_p$; FID and KID as a $W_2$ and an MMD (DD-26, FR-7).
- A concrete reason why total variation cannot be estimated from samples (DD-18).
- Bits-back coding and asymmetric numeral systems, at least in further reading (FR-7; also raised in the index and coding reviews).
- A ranking of the fifteen further-reading entries (structural finding in that file).

## What is best and should set the model

- **The coding section as a whole.** One measured hook (2.84 against 2.55 bits per character) that is explained and returned to; one source reused for the symbol-code penalty, the arithmetic coder, the Markov chain and the perplexity floor; every bound followed by its number and by a plain statement of what it is worth ("up to a factor $2^{\pm n\delta}$, which is large"); a counterexample that fences each claim. Its sentences have a voice: "Most messages are unlikely, and the most likely message is unrepresentative."; "The typical set serves proofs, and it makes a poor coder."
- **The design of the estimation section.** A practitioner's question, an exact critic that isolates the slack of the bound from the error of the model, and one idea named by the authors under four ceilings. The opening page ("That number can never exceed 4.85 nats ...") and "A million samples cannot certify 177 nats, and they cannot certify 30" are what the voice guide asks for.
- **The questions game and the card reports** of the entropy section: human-scale, exact, and used to give KL a meaning before its formula.
- **The flip-channel table** of the mutual-information section, used six times at one sentence each, and "A loss certifies information" with its confidence sweep.
- **The opening of the divergences section**: a table in which three of five rows do not move, then one identity, $D_f(\delta_0 \| \delta_d) = f(0) + f'(\infty)$, that explains them; and the crossing argument for $W_1$ on the line, stated before the proposition it proves.
- **Generalizing by asking what a proof used** ("Gibbs' proof used only two facts about $u \log u$").
- **The Rényi entry** of further reading: a question, three clauses of reason, a consequence.
- **The exercises** of the coding, divergences, mutual-information and estimation sections. Several are small results in their own right and should not be cut for length.

## Contradictions and tensions between the section reviews

- Home of joint and conditional entropy: two reviewers ask for the move to the first section, one declines to repeat the request (see "Arc and order"). Compatible if the first section gets a short subsection and the fourth keeps the full one.
- Confirming cells: praised as predict-then-compute in the divergences review, criticised in the entropy review (see the calibration note under "Scores").
- Two reviewers each call a different section the best they read (coding; estimation). They read different halves, so this is not a disagreement.
- COD-S1 says the coding section has no definition of conditional entropy; it has a one-clause definition (checked). The finding stands for joint entropy and the chain rule.
- No review contradicts another on a matter of mathematics.

## Still open from October

Differential entropy before KL with "defined below" (INF-S2, INF-7); joint and conditional entropy after their first use (COD-S1); the $\alpha$-divergence paragraph and the kernel Stein discrepancy, unused (DD-8, DD-32), in a section that has grown from about 8,000 to about 11,000 body words. Done since October: the dyadic source in the text, the coding reading of KL at its definition, the split of coding into its own section, the point-mass table computed and first, Rényi privacy and SVGD moved to further reading. The reviewers disagree with October on two cuts: they would keep MDL and the short scoring-rules subsection.

## Could not check

- TensorFlow, JAX and MXNet outputs; only the PyTorch store was compared. The risk is small, since three sections are NumPy only.
- The rendered pages. One reviewer inspected a July build to see whether cell identifiers are visible; the current build was not rendered.
- Bibliographic details of the further-reading citations.
- The figures the proposed sentence in COD-8 needs.
- Whether chapter 41's reviewers propose the same division of entropy, Gibbs and maximum entropy; the overview must reconcile the two chapter reviews.
