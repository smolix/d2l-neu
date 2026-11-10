# Chapter 16, Generative Adversarial Networks: writing review

Sources: the nine `.md` files of `chapter_generative-adversarial-networks/`. The seven teaching sections were read in full by three section reviewers (`gan`, `objectives`, `relativistic`; `convergence`, `dcgan`; `conditional`, `adversarial-losses`). I read `index.md` and `further-reading.md` in full and checked the cross-section claims below against the sources, the stored PyTorch and JAX outputs, the appendix (`chapter_mdl-information-theory/mdl-divergences-distances.md`) and the diffusion chapter. About 27,000 words of body prose in the seven sections, about 9,000 words of exercises, 3,000 words of further reading.

Per-file reviews:
[index](16-gans/index.md) ·
[gan](16-gans/gan.md) ·
[objectives](16-gans/objectives.md) ·
[relativistic](16-gans/relativistic.md) ·
[convergence](16-gans/convergence.md) ·
[dcgan](16-gans/dcgan.md) ·
[conditional](16-gans/conditional.md) ·
[adversarial-losses](16-gans/adversarial-losses.md) ·
[further-reading](16-gans/further-reading.md)

**Author decision (Alex Smola, 2026-11-10).** "Keep R3GAN as it's probably one of the cleaner ones (that said, GANS have overall gone a bit out of fashion)." R3GAN stays as the chapter's spine, so the recommendation below to reduce the programme to one section is withdrawn. What stands: say once what the chapter adopts and what its notebooks establish, repair CONV-10, CONV-7, DCG-10 and COND-15, say candidly on the last page where GANs stand today, and do not let the chapter grow. The merge of `objectives.md` and `relativistic.md` is still open. See `overview.md`, "Author decisions".

## Verdict

The mathematics of this chapter is correct, careful and in places original, and each section contains a passage at the bar: the noise mixture and the optimal critic in `gan.md`, the coin under three losses in `objectives.md`, the two atoms in `relativistic.md`, the Dirac-GAN in `convergence.md`, the score floor in `dcgan.md`, the projection head from Bayes' rule in `conditional.md`, the two bands in `adversarial-losses.md`. No reviewer found a wrong equation. The chapter is nevertheless not at the bar, and the reason is the same in every section: it does not separate what it has shown from what a paper claims, and it reads its own experiments in the light of that paper. It is organized around one 2024 programme (the pairing loss with two gradient penalties on a plain backbone). The chapter's own analysis qualifies the pairing loss, its own image scores do not separate it from the classic loss, and on those same scores the 2015 network beats both modern arms at a tenth of the cost; the prose says "lands close". The careful statements are in the exercises and on the last page, the confident ones in the body. What matters most: say once what the chapter adopts and what its notebooks can establish, repair the four places where body and evidence disagree (CONV-10, CONV-7, DCG-10, COND-15), and cut the chapter from seven sections to five or six by merging the two sections on objectives.

## Score table

Scores as given by the section reviewers (index and further reading are mine).

| Section | Explanation | Thought | Pedagogy | Language | Enjoyment |
|---|---|---|---|---|---|
| [index](16-gans/index.md) | 4 | 3 | 3 | 4 | 3 |
| [gan](16-gans/gan.md) | 4 | 4 | 4 | 4 | 3 |
| [objectives](16-gans/objectives.md) | 3 | 3 | 3 | 4 | 3 |
| [relativistic](16-gans/relativistic.md) | 4 | 3 | 3 | 4 | 3 |
| [convergence](16-gans/convergence.md) | 4 | 3 | 3 | 4 | 4 |
| [dcgan](16-gans/dcgan.md) | 4 | 3 | 3 | 4 | 3 |
| [conditional](16-gans/conditional.md) | 4 | 4 | 3 | 4 | 3 |
| [adversarial-losses](16-gans/adversarial-losses.md) | 4 | 3 | 3 | 4 | 4 |
| [further-reading](16-gans/further-reading.md) | 4 | 3 | 3 | 4 | 3 |

The three reviewers calibrated alike, and the table discriminates little: language is 4 everywhere, pedagogy 3 in six of seven sections. The averages hide a split inside sections that three reviewers reported independently. The Dirac-GAN subsections alone were scored 5, 5, 5, 4, 5 and the first half of `adversarial-losses.md` 5, 5, 4, 4, 5, against 3, 3, 2, 4, 3 for its second half. The strongest section is `gan.md` (one question, answered, tested). The weakest as a textbook section is `relativistic.md`: its mathematics is the most elegant of the chapter, and it is also the section a reader would skip, because it trains nothing and its main conclusion is negative and made in passing. `objectives.md` has the weakest explanation (its second half is a survey with appendix pointers).

## Technical correctness and currency

No reviewer found an error in a derivation; worked numbers were recomputed in every section and agree. The findings of substance are claims stronger than their support. Ordered by severity; confidence is the reviewer's unless marked.

Errors and contradictions:

- **CONV-10** (major; confident, and I verified it). The body of `convergence.md` says four times that the penalty restores a gradient on separated supports ("Smoothed, two distributions with disjoint supports overlap, and the divergence responds to the distance between them"; "Below the cutoff the generator therefore keeps a signal when nearby supports move apart"; table entry "smooth, near equilibrium" in the column "on disjoint supports"; "The constraint decides whether a gradient survives on disjoint supports"). Exercise 2(c) of the same section asks the reader to show that for any differentiable critic under this penalty the best-response value is $2\sup_t \ell(t)$ for every $\theta \neq 0$, that is, saturated. Both texts are in the source as quoted. The exercise is right: the penalty is weighted by $m$, which vanishes in the gap. The signal across a gap comes from the smoothness of the critic network, not from the penalty.
- **CONV-7** (major; high confidence in substance, hypotheses from memory). The local-convergence theorem is invoked in three places ("small enough gradient steps converge to it at a linear rate", "needs only one", "The penalties make the equilibrium locally attracting") with no hypotheses and no scope. It is proved here only for the two-parameter game. The cited theorem needs realizability and conditions on the loss, is local, and covers the classic loss with one penalty, not the pairing loss with two.
- **CONV-5** (moderate; confident). "The equilibrium attracts the iterates only if the eigenvalues ... have negative real parts" is false as written; the section's own exercise 9 is a counterexample.
- **REL-9** (major; confident on (a) and (b)). The conclusion that redrawn pairs face "as many collapse minima as the classical loss" is stated in general, rests on an enumeration at $n = 2, 3$, and asserts in the body the answer to a question that exercise 8(c) poses as open.
- **OBJ-9** (major). The Wasserstein passage equates a supremum over Lipschitz functions with a transport cost without naming the theorem that equates them, and says the three practical devices enforce the constraint "approximately" although they enforce three different things.
- **ADV-8** (moderate; confident). "low distortion and outputs distributed like the data cannot both be reached" overstates a bounded tradeoff (at most a factor of two for squared error); exercise 1(c) of the same section exhibits a counterexample to the strong reading.
- **ADV-13** (moderate; confident). Perceptual losses are presented as a 2026 replacement for the critic in autoencoders; they have accompanied it since VQGAN. `further-reading.md` and `chapter_diffusion-models/latent-diffusion.md` say it correctly, so the chapter contradicts itself and its neighbour.
- **ADV-1** (moderate; confident). The book did not train class probabilities by squared error.
- **GAN-9** (moderate; confident in general, not verified for this model). "at this budget the game has not reached its equilibrium" implies more epochs would bring both losses to $\log 2$; the opening of `convergence.md` says of the same run that the losses "never settled".
- Minor: OBJ-4 (a hypothesis that excludes nothing), OBJ-8 ("ceiling" for divergences that are infinite), CONV-2 (circles need $\ell' > 0$ everywhere), GAN-7 ("rarely" without evidence), ADV-9 (not all samplers add noise), ADV-12 (the teacher's score is the teacher's, not the data's), DCG-3 (DCGAN's batch size was 128), DCG-17, FR-2 and FR-3 (a pointer to the wrong section; "tightly" for a soft penalty).

Stored outputs against prose:

- **DCG-10** (major; I verified the stored table in both tabs). FD and $\mathrm{MMD}^2$: PyTorch DCGAN 3.54 / 0.181, NS + $R_1$ + $R_2$ 5.69 / 0.877, RpGAN + $R_1$ + $R_2$ 4.22 / 0.454; JAX 4.17 / 0.290, 5.05 / 0.655, 4.31 / 0.485. DCGAN is lowest on both scores in both tabs. On FD "lands close" can be defended (in JAX 4.17 against 4.31, inside the stated feature-seed factor of 1.5). On $\mathrm{MMD}^2$ it cannot: DCGAN is lower by a factor of 1.7 to 4.8, and the prose says only that "$\widehat{\mathrm{MMD}}^2$ agrees". Second, "Which of the two scores lower changes from run to run" is contradicted by every number on the page: the pairing arm is lower on both scores in both tabs. The claim may be true over seeds; no cell shows it.
- **DCG-15** (minor). DCGAN is scored from its last iterate, the three arms from their moving average; the prose does not say so.
- **OBJ-13** (major). "at about $0.005$ nats" against stored medians 0.0070 and 0.0059; the lag explanation is asserted in the body and posed as open in exercise 6(b).
- **COND-15** (major). A pooled feature variance of 0.17 against 0.21 is read as "lost some of the variety within each class". The number is as stated; the reading is not supported.
- **GAN-11** (minor). "an occasional epoch in which both losses jump" is not visible in the stored plot.
- **OBJ-11** (minor). The Gaussian decay of the MMD slope lies outside the plotted range.
- Numbers no cell prints: "about 150 runs", "sixteen runs" and the coverage fractions in `convergence.md` (CONV-14); the feature-seed factors and "overlap across seeds" in `dcgan.md`; "tens of points between two training seeds" in `conditional.md`; the Jensen-Shannon values in `gan.md` (correct on recomputation).

Unverified claims about papers (the reviewers could not confirm; a reviser should check each against the source): REL-2 (who named RpGAN), CONV-6 (attribution of the unweighted penalties), CONV-17 and FR-4 (R3GAN's configuration values), DCG-5 (which loss R3GAN's stripped baseline uses), COND-9 (R3GAN's conditional critic and compute), ADV-11 (the ADD ablation digits), ADV-14 (ViTok-v2 and RAF, both 2026), CONV-9 (the StackedMNIST table), FR-9, IDX-5, and exercise 10(c) of `dcgan.md` (the augmentation probability).

Currency: the chapter is current, and dated honestly where it is old (DCGAN under "The 2015 Recipe"). The risk is the opposite one. The second half of `adversarial-losses.md` and the last heading of `further-reading.md` name about fifteen systems from 2024 to 2026, two from 2026 preprints, with their scores. One of the chapter's four closing limits ("unstable at scale") rests on a single 2026 preprint (ADV-14), and the diffusion chapter's introduction adopts it as a premise.

## Three questions put to this review

**Is the weight on R3GAN proportionate for a textbook? No, not as it stands.** The paper is named on about twenty source lines of the teaching sections (fourteen in `convergence.md`), gives its name to the saved training loop, supplies an ablation that the text invokes five times, and is the reference point against which the further reading measures other work (FR-8). Sections 16.3 to 16.6, four of seven, follow its programme. The two halves of the programme deserve different weight. The zero-centered gradient penalty is established practice with theory behind it (it predates the paper by six years and is in the StyleGAN line), and the Dirac-GAN teaches it superbly; that weight is earned. The pairing loss is one paper's choice. The chapter's own analysis says its landscape guarantee does not apply to minibatch training (REL-9), its own image experiment says "this budget does not separate them", and the only evidence for it is the cited ablation. It nevertheless gets the chapter's longest theory section. A proportionate treatment keeps the recipe as the teaching baseline (it has few parts and each has a reason), says so once in the introduction (IDX-3), gives the pairing loss a subsection instead of a section, and marks every result of the paper as the paper's.

**Can the notebooks establish anything about a method whose evidence is at FFHQ and ImageNet scale? One thing, and it is worth having.** They show, with an effect far outside the noise and in both frameworks, that a plain unnormalized backbone collapses under the classic loss and trains when the penalties are added (FD 73 to 89 against 4 to 6). They also show the mechanism on two parameters. They cannot show that pairing helps (the sprite scores differ by less than the instrument resolves; the 25-Gaussian run confounds pairing with the penalty and needs many seeds, CONV-14), that the recipe matches a tuned older network (on the chapter's own scores it does not), or that the older devices are unnecessary (no arm isolates them, DCG-4). The chapter should say exactly this, in the introduction and again where the scores are read. The candid sentence is in DCG-10: what the penalized loss buys at this scale is not a better score, it is that a network with none of DCGAN's devices trains at all; that this lets the loss scale is the paper's claim, which the notebook cannot test.

**Is seven sections the right size in 2026, next to a diffusion chapter of eight teaching sections and a further-reading page? No. Five or six.** The introduction itself says GANs have been replaced for most image synthesis and that what survives is the adversarial term. Length should follow that judgment and at present runs against it: three sections and about 12,000 words on what losses measure, and 1,400 taught words on why a critic survives, followed by a survey. The proposal under "Arc and order" merges 16.2 and 16.3, moves one derivation to the appendix, and brings the chapter to about 19,000 words of body prose in five or six sections, with the last section the one that grows.

## Arc and order

The chapter has one line of argument and the order is right: what the game measures (16.1 to 16.3), whether training converges (16.4), images and their scores (16.5), requests (16.6), where the critic survives (16.7). The thread that runs through it, the critic as an estimate of $\log(p/q)$, is stated in the introduction and returned to in the last section's reading of distribution matching distillation. Two reviewers independently praised theory steering a measurement choice (the per-class constants in `conditional.md`, the floor in `dcgan.md`).

Problems of proportion and boundary:

- **Three sections without an image or a trained neural generator.** After the garments of 16.1 the reader waits about 12,000 words for the next image, and 16.3 trains nothing at all.
- **Merge 16.2 and 16.3.** Pairing is a third choice of objective. A merged section "What the Loss Measures" of about 4,500 words would keep the coin and the proposition, the four critic links, separated supports with an honest Wasserstein passage (OBJ-9), then the two atoms, the lifting and the landscape verdict (REL-9), and train the pairing loss as a sixth arm on the mixture testbed. What goes: the cuts listed in structural finding 1 of `relativistic.md`, the estimation-rates paragraph (OBJ-10), the squared-hinge arm if the hinge finding is demoted (OBJ-13).
- **`convergence.md` is three sections in one file.** Dynamics; what the penalized value measures; the recipe in code with an experiment and a chapter table. The middle part changes the question and raises the level (CONV-11). Move its derivation to the appendix next to :numref:`sec_mdl-ot-connection` and keep the result with the Gaussian shift worked by hand.
- **Two suggestions from different reviewers pull against each other.** The reviewer of `relativistic.md` proposes folding the pairing objective into 16.4; the reviewer of `convergence.md` finds 16.4 already overloaded. Merging 16.3 into 16.2 satisfies both.
- **`dcgan.md` is three topics** (the 2015 network, the backbone and three arms, the metrics) plus "Limited Data", which shows neither of its phenomena (DCG-12). Cut, do not split.
- **`conditional.md` is two halves of unequal quality**, and its reviewer asks for more (modulation taught, COND-7) while the chapter needs to shrink. Resolve by cutting "Same-Class Pairs" to a sentence (COND-5), the pix2pix stub and the two cannot-fail baselines, and spending those words on a six-line modulation layer, which the reader meets again in every diffusion network.
- **`adversarial-losses.md` teaches, then surveys.** It is the section that justifies the chapter in 2026 and is the shortest. Its second half should be taught on the two bands (structural finding 1 there).
- **Boundaries that leak.** The notation table for the whole chapter sits inside 16.1 (GAN-2). Code and prose written for a later section sit in 16.4 and 16.5 (DCG-13). The chapter-wide table of games sits at the end of 16.4, three sections before the end (CONV-16), and is redrawn as a figure in 16.7 (ADV-16). Pix2pix is introduced in 16.6 and explained in 16.7 (ADV-2).
- **The account of pairing is inconsistent across sections.** 16.1 promises that ranking "replaces this threshold with a comparison" (GAN-8); 16.3 shows ranking leaves the collapse minima; 16.4 credits pairing with coverage from one ablation (CONV-9); 16.7 labels RpGAN a "convergent objective" in a figure (ADV-16) and then says none of the guarantee applies (ADV-19).
- **The chapter has no verdict at its end** (ADV-18). The last prose lists four limits and omits what GANs keep (one evaluation per sample) and why they were displaced.

## The most important findings

1. **Say what the chapter adopts and what it has shown.** One paragraph in the introduction on the recipe and on what small experiments can establish; every result of the R3GAN paper marked as the paper's; the StyleGAN2 caveat moved from the last page to where the ablation is cited. IDX-3, CONV-9, FR-3, FR-8, structural finding 4 of `relativistic.md`.
2. **Remove the contradiction on disjoint supports.** Put the content of exercise 2(c) in the body and correct the table of games. CONV-10.
3. **State the convergence theorem with hypotheses and scope**, and repair "only if". CONV-7, CONV-5.
4. **Read the image scores candidly.** DCGAN first, pairing below the classic arm in both stored tabs, the averaging asymmetry, the instrument's resolution in numbers; remove the two explanations no arm supports. DCG-10, DCG-15, DCG-4, DCG-9.
5. **Decide whether the book teaches the Wasserstein critic or cites it.** At present it is presented as the answer to separated supports, gets about ninety words, and is trained nowhere, while the appendix says ":numref:`sec_gan_objectives` trains such a critic" (`mdl-divergences-distances.md`, checked). Either run the one-dimensional separated-start experiment that exercise 9 of `objectives.md` describes, or say that the chapter does not train one and why, and fix the appendix sentence. OBJ-9, OBJ-12.
6. **Halve the pairing material, lead with its verdict, and train it once.** REL-9, REL-5, REL-7, REL-10, GAN-8.
7. **Rebuild the 25-Gaussians experiment** as four arms with a few seeds and a trace over time, so that it shows convergence and attributes its result. CONV-14.
8. **Measure coverage on images once.** The chapter's announced twist in 16.6 is within-class collapse, and it is shown by a baseline whose numbers are fixed by construction and a variance that is not a coverage measure. COND-15, COND-16.
9. **Teach the second half of the last section and end the chapter with a judgment.** ADV-11, ADV-7, ADV-18, ADV-17, ADV-14.
10. **Move into the body the facts it depends on that now sit in exercises.** GAN-6 (the generator's divergence is $2 D_{\textrm{KL}}(m \,\|\, p)$), REL-6, OBJ-13, CONV-9 (exercise 3), CONV-18.

## Focus

- **Paper retellings**: the StackedMNIST table (CONV-9), "Six Principles" (DCG-5), the ADD ablation (ADV-11), the correction of a paper's equation (REL-10), the estimator comparison reported from a paper (REL-11), "Limited Data" (DCG-12).
- **Surveys inside teaching sections**: the integral-probability-metric half of `objectives.md` (structural finding 2 there, OBJ-10); "Adversarial Terms in Current Systems" with ten systems at equal weight and a subsection on one 2025 method (structural findings 2 and 3 of `adversarial-losses.md`).
- **Variant lists**: seven new terms in 400 words (OBJ-2); seven coined terms in 3,400 words (COND-17); the three conditioning mechanisms, of which the dominant one gets a sentence (COND-7).
- **Further reading** does not rank and carries a scoreboard (FR-1, FR-4).
- A rule proposed by the reviewer of `adversarial-losses.md` and worth adopting for the part: a system is named in a teaching section only if the section derives or runs the idea it stands for.

## Experiments

- **Cells that only confirm or cannot fail**: three in `relativistic.md` (REL-5); the two constructed baselines in `conditional.md` (COND-16); the squared-error student, acceptable as a control (ADV-5); the Gaussian-shift check, which would teach if the shift were varied (CONV-13); the density-ratio check, allowed once (GAN-12).
- **Read for more than they show**: the hinge residual (OBJ-13); the garment losses (GAN-9); the $\gamma = 10$ prediction, applied outside both hypotheses of its derivation (CONV-14(d)); the collapse mechanism (DCG-9); "architecture" as the cause of DCGAN's success (DCG-4); variance as variety (COND-15); critics that memorize (DCG-12, DCG-17).
- **Unfair or confounded comparisons**: the grid arms change loss and penalty together (CONV-14(b)); five losses with shared untuned settings (`objectives.md`, code notes); DCGAN against the backbone differs in seven respects (DCG-4) and in averaging (DCG-15).
- **Experiments that do not test the section's question**: every arm in `objectives.md` starts from overlapping supports in a section about separated ones (OBJ-12); the grid experiment shows coverage in a section about convergence (CONV-14(a)); nothing is trained in `relativistic.md`.
- **Scale**: the planar experiments run in seconds and show their phenomena. The sprite and CIFAR-10 runs need a GPU for about half an hour each and produce no recognizable image (DCG-11, structural finding 6 of `conditional.md`); the text should say what budget would.
- **Done well and to be kept**: the saturation experiment with its candid limit in `gan.md`; four critics, four functions of one $\lambda$ in `objectives.md`; the collision count in `relativistic.md`; the three-arm design, the floor and the renamed scores in `dcgan.md`; the two bands.

## Repetition

Inside the chapter:

- The StackedMNIST ablation: stated and then invoked again in `convergence.md` and once more in `dcgan.md` (five uses by the reviewers' count), with its caveat only in `further-reading.md`. Home: one sentence in "One Penalty or Two" with the caveat beside it.
- The verdict on pairing and fixed couplings: twice in consecutive paragraphs of `relativistic.md` (REL-9(c)), again in "Reading the Table" (CONV-16) and in "Remaining Limits" (ADV-19). Home: the end of the pairing material.
- "Convex in $q$, not in the parameters": `objectives.md` and `relativistic.md` under the same heading (REL-12). Home: `objectives.md`.
- Shift invariance: three statements in `relativistic.md` (REL-7).
- The unsettled garment losses: retold in the openings of 16.2 and 16.4. One clause each once GAN-9 is fixed.
- "Use the non-saturating form": `relativistic.md` and `convergence.md` (REL-10). Home: where the loss is coded.
- "$\gamma$ is a squared length": `convergence.md`, `dcgan.md` (DCG-8), `conditional.md` (COND-12). Home: `convergence.md`, with the pixel-count clause of CONV-17.
- The table of games and :numref:`fig_gan_exits` (CONV-16, ADV-16). Keep the table, as the chapter's last element.
- Pix2pix: `conditional.md`, `adversarial-losses.md`, `further-reading.md` (ADV-2, FR-2). Home: `adversarial-losses.md`.
- The opening image, in the introduction and in `gan.md` (GAN-1, IDX-1).
- Double endings in six of seven sections (see patterns).
- `further-reading.md` against `dcgan.md` (limited data), `conditional.md` (conditional scores, auxiliary classifiers) and `adversarial-losses.md` (tokenizers, vocoders, likelihood models as critics). Home: further reading, once the sections are cut.

With Basics: the classifier that learns a log density ratio is demonstrated in :numref:`subsec_covariate-shift-correction` and again here (GAN-12); allowed once, and the cell should say what is new. The transposed convolution is explained in `dcgan.md` without a pointer to :numref:`sec_transposed_conv`.

With other Advanced chapters: the Bradley-Terry model is cited from :numref:`sec_regularized` and not re-derived, which is right. The diffusion chapter restates the conditional-mean argument and the four limits as its opening; that is a handover and should stay, with the links made two-way.

With the appendix (author's rule): mostly observed. The f-divergence bound, MMD and the Jensen-Shannon identity are stated here with proofs there, and the tables are complementary. Exceptions are under "Dependencies".

## Recurring patterns

- **The body asserts, an exercise qualifies or reverses.** CONV-10 (exercise 2(c)), CONV-14(d) (exercise 4(b)), OBJ-13 (exercise 6(b)), REL-9 (exercise 8(c)), ADV-8 (exercise 1(c)). All three reviewers reported it. The exercises are more careful than the prose.
- **A fact the body needs is in an exercise.** At least twelve cases in six sections: GAN-6, OBJ-12, OBJ-13, exercise 5 of `objectives.md`, REL-6, REL-9, CONV-9, CONV-10, CONV-18, DCG-9, COND-4, exercise 3(b) of `adversarial-losses.md`. The exercise sets are the strongest part of the chapter (about a third of the body in every section, research-grade in places); several are where the section's best argument lives.
- **Paper results stand in for experiments**, and the chapter's experiments are then read in their light. CONV-9, DCG-5, DCG-10, ADV-11, ADV-14.
- **Double endings.** A "We can now ..." recap before the Summary in `relativistic.md` (REL-15), `dcgan.md` (DCG-16) and `conditional.md` (COND-18); a restating paragraph before the Summary in `objectives.md` (OBJ-15), `convergence.md` (CONV-16) and `adversarial-losses.md` (ADV-17), where the section ends three times.
- **Claims over runs that no cell makes.** See "numbers no cell prints" above. Ranges are the author's rule; here two of them contradict what the single stored run shows (DCG-10(b), CONV-14(c)).
- **Correct results without rank.** The reader is rarely told which result matters; the judgment that does matter is in a subordinate clause (REL-9) or the last body paragraph (`objectives.md`, structural finding 3).
- **Notation and terms.** Critic against discriminator in text, figure and code (GAN-3, DCG-2). Two normalizations of the classic critic loss, with equilibria $\log 2$ and $2\log 2$ printed in one section (CONV-15, DCG-7). Three sign conventions for one game in forty lines (OBJ-1). $t$ as noise level, time index and threshold (ADV-10); $\phi$ for two feature maps (COND-8). The notation table has wrong first-use entries and a "papers write" column readers cannot decode (GAN-2).
- **Plumbing addressed to the build or to the authors**: the initialization constant in `objectives.md`, the kernel flags and import comments in `dcgan.md` (DCG-1), unused label arguments (DCG-13).
- **Language** is clean throughout, and the reviewers counted few tics: "thus" as a hinge, 3 (COND-19); announcing sentences, 4 (ADV-20); the "Read it for" closing, 7 (FR-6); an author referred to by pronoun in two files (REL-14, FR-5). Voice is present where the section has a concrete case (the wine taster, the thumbnail, the menu of losses) and absent in the theory runs.
- **Framework tabs** do not fragment the prose in any section; there is no TensorFlow or MXNet material.

## Dependencies

- **From Basics**: logistic regression, the covariate-shift classifier, KL divergence and entropy, batch normalization, convolutions. All cited where used. Missing: a pointer to :numref:`sec_transposed_conv` (which now sits later in the book, so the stamping example in `dcgan.md` must stay self-contained, as it is).
- **From earlier Advanced chapters**: Adam and its momentum (chapter 9), used with $\beta_1 = 0$ on authority instead of the reason the section has (CONV-18, GAN-14); the Bradley-Terry model of :numref:`sec_regularized` (resolves).
- **From the appendix**: the chapter reads without it, with three exceptions. The Wasserstein distance cannot be understood from the main text (OBJ-9). The dual Sobolev derivation is a long derivation that the rest of the chapter uses only through one length scale; by the author's rule it belongs in the appendix (CONV-11). And `convergence.md` uses linearization at a fixed point, centers and the stability of Euler steps without restating them or citing the appendix sections that hold them (:numref:`subsec_mdl-ode-linearization`, :numref:`sec_mdl-linear-odes-stability`; CONV-5).
- **Appendix references that point to the wrong place or content**: :numref:`sec_mdl-ot-connection` does not contain the step it is cited for (CONV-12); the data-processing proposition is stated under hypotheses that do not cover its use (REL-6); :numref:`sec_mdl-divergences-distances` is cited for two bounds the sentence has just proved, a pointer to 2,400 lines (GAN-4). In the other direction, the appendix sentence that this chapter trains a Lipschitz critic is false.
- **Forward references**: none to the skeleton chapters `chapter_lm-*` (checked). The only forward dependence is on :numref:`chap_diffusion`, and it is load-bearing in one place: "Denoisers as Conditional Means" and "One-Step Distillation" explain diffusion distillation one chapter before the reader meets a diffusion model, with no pointer to the sections that teach it (structural finding 4 of `adversarial-losses.md`). The diffusion chapter cites this chapter in at least five files; this chapter cites it twice, by chapter label only.

## Connection to the rest of the book

- The part introduction (`chapter_advanced/index.md`) and the diffusion introduction both lean on this chapter's close. The four limits are restated in `chapter_diffusion-models/index.md` and answered in `discrete-diffusion.md`; if ADV-18 and ADV-14 change the close, those two passages must follow.
- The perceptual loss is described differently here and in `latent-diffusion.md` (ADV-13).
- Add references from `adversarial-losses.md` to `sec_diffusion-denoising`, `sec_diffusion-few-step` and `sec_diffusion-latent`.
- Uses named without a label: pix2pix's patch critic, used in 16.7 without a name (ADV-13); the bilinear game and the extragradient family, which a reader will meet elsewhere and cannot connect to the Dirac-GAN (structural finding 8 of `convergence.md`; no entry in the further reading either).

## What to cut

About 8,000 words of body prose, taking the chapter from about 27,000 to about 19,000:

- `gan.md`: the notation table (moved), the first calculation of "The Same Fixed Point" (GAN-5), the initialization sentence. About 400 words.
- `objectives.md` and `relativistic.md`, merged: part (b) of the bounds proposition, the five-atom verification, the rank-weight plot, the convexity subsection, the estimator subsection, the erratum, the estimation-rates paragraph, the template figure (OBJ-16). About 3,500 words.
- `convergence.md`: the proof of the local value (to the appendix), the StackedMNIST table, "Reading the Table". About 900 words.
- `dcgan.md`: "Limited Data" to a paragraph, the collapse mechanism, the label plumbing prose, the determinism comment. About 900 words.
- `conditional.md`: "Same-Class Pairs" to a sentence, the pix2pix stub, the vehicle description, the two baselines as a cell. About 600 words.
- `adversarial-losses.md`: seven of ten systems, the subsection on likelihood models as critics, the recap paragraphs, the figure. About 1,000 words.
- `further-reading.md`: the scoreboard numbers and the configuration list (FR-4).

## What is missing

- One paragraph on what the chapter adopts and what its notebooks can show (introduction).
- A trained Wasserstein or Lipschitz critic on separated supports, or the statement that there is none.
- The pairing loss trained once, on the mixture testbed.
- The convergence theorem's hypotheses; the words "bilinear game"; what alternating updates do, since every loop in the chapter alternates (CONV-4).
- A trace over time in a convergence experiment.
- A coverage measure on images (precision and recall are in the further reading and in an exercise, without formulas).
- Latent interpolation (structural finding 7 of `dcgan.md`).
- Modulation, taught.
- What practice uses (hinge with spectral normalization, least squares in vocoders, non-saturating loss with $R_1$), in `objectives.md` (structural finding 3 there).
- The distillation claim shown on the two bands.
- A closing verdict: what GANs keep, why they were displaced, when one would train one now.
- A ranked short list in the further reading, and an entry on the dynamics of games.

## What is best and should set the model

- **`gan.md`, the noise mixture.** "Nor is likelihood what we want from a sampler", 1% against 99%, 4.6 nats. A sharp example that makes the reader want the chapter.
- **`objectives.md`, "Two Outcomes by Hand".** One coin, three losses, three known divergences, then the proposition the reader already believes.
- **`relativistic.md`, "Two Atoms by Hand"** and the lifting stated before the lemma.
- **`convergence.md`, the Dirac-GAN.** Orthogonality gives the circle, Pythagoras the outward spiral, one matrix entry the damped spring. The best pages of the chapter and among the best of the part.
- **`dcgan.md`, the instruments.** A real-against-real floor, scores renamed because the feature network is home-made, the out-of-range column. The model for every scored experiment in the book.
- **`conditional.md`, the projection head from Bayes' rule**, with the modelling assumption flagged as one, and a measurement chosen because the theory says which constants cancel.
- **`adversarial-losses.md`, the two bands.** The smallest example that separates two losses, worked by hand before any code, followed by a paragraph that refuses to oversell the critic.
- **The exercises**, in every section.

What these have in common is a case small enough to compute by hand, placed before the general statement, and an author who says what was not shown. The revision should extend that second habit from the experiments on toys to the experiments on images and to the claims taken from papers.

## Could not check

- Rendered HTML; I read sources, stored text outputs and the review files, and did not re-render figures the section reviewers had inspected.
- The claims about papers listed as unverified above, in particular everything from 2026.
- Claims over repeated runs: the store holds one run per framework.
- The proposed experiments (separated starts, four-arm grid, pairing on the mixture, a recall measure, distillation on the two bands) are untested; each belongs on the list for the development machine.
