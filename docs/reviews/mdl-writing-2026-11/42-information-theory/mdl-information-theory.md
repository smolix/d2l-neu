# Entropy, Cross-Entropy, and KL Divergence

Source: `chapter_mdl-information-theory/mdl-information-theory.md`. About 6,950 words of prose and mathematics before the exercises, 8,400 with them; 13 code cells (two of them in four framework tabs).

## Verdict

This is good textbook prose and, in its first half, a faithful application of the Cover and Thomas method: a sentence of meaning before each formula, a convention with its reason, and a number right after. The section uses Cover and Thomas's own source $(\tfrac12, \tfrac14, \tfrac18, \tfrac18)$ with 1.75 questions, and its pair $P = (0.6, 0.3, 0.1)$, $Q = (0.2, 0.5, 0.3)$ does the job that $D(p\|q) = 0.2075$ against $0.1887$ does in their chapter 2. I recomputed every number in the text, the stored outputs and the exercise hints, and found no mathematical error. The section is not yet at the level of the models for a structural reason: it does not know what the reader already has. Chapter 41 has defined entropy, KL divergence and cross-entropy, proved Gibbs' inequality, and proved maximum entropy in general with the uniform, Gaussian and geometric cases as corollaries. This section proves all of it again without saying so, then applies Gibbs' inequality five more times (the cross-entropy bound, the irreducible loss, proper scoring, label smoothing, distillation), two of them dressed as propositions. The most important fix is to decide that this section is the home of entropy and Gibbs' inequality, prove each thing once, and let the applications be one sentence each. That would shorten the section by about a quarter and leave room for joint and conditional entropy, which the next section needs and does not have.

## Scores

| Dimension | Score |
|:--|:--|
| Clarity of explanation | 4 |
| Clarity of thought | 3 |
| Pedagogy | 4 |
| Language | 4 |
| Ease and enjoyment | 3 |

## What works

- **The opening.** "A freshly initialized classifier with ten classes reports a loss of about $2.30$" followed by "Is a final loss of $0.6$ good? Is a plateau a bug or a property of the data?" These are the reader's questions, and the section answers each with a number ($\ln 10$; $0.545$ nats for 10% label noise). Keep the paragraph as it is.
- **The card reports** ("I see a card", "I see a heart", "the three of spades", the order of the deck). Four statements, four probabilities, and "as much information as $226$ fair coin flips". This is a MacKay-grade example: human-scale, exact, and memorable.
- **The questions game.** The dyadic source is identified by "Is it $a$?", "Is it $b$?", "Is it $c$?" in 1.75 questions, and the questioner who ignores the probabilities needs 2. The same game then gives KL divergence a meaning before its formula, and the bar figure shows $2 = 1.75 + 0.25$. The game is the section's best device.
- **The choice of the second example is explained.** "For this pair the reverse divergence ... also happens to be $0.25$ bits, which is why we showed the asymmetry of KL on the other pair." An author is visibly choosing. Cover and Thomas do this; most textbooks do not.
- **Sentences with voice, to protect in revision:** "a model that declares an event impossible is infinitely surprised when it happens"; "no model beats the truth on average"; "with equality only when there is nothing to average"; "Surprise depends on the model that assigns the probabilities, so a confidently wrong model finds ordinary events surprising."
- **The proofs name their decisive step.** In the uniform bound ("each outcome in the support contributes $p_i \cdot (1/p_i) = 1$") and in Gibbs ("what remains is the $Q$-mass of that support"). Both proofs handle the support correctly, which many textbooks skip.
- **Differential entropy is taught through its defects** (negative for the uniform density on $[0, \tfrac12]$; plus $\ln 100$ when metres become centimetres). The quantization identity $H(X_\Delta) + \log \Delta \to h(X)$ with the eight-cell check is exactly right and short.
- **The irreducible-loss cell** is code that teaches: the "confident 0.99" predictor scores $0.689$, above the floor of $0.545$, because the flipped labels surprise it. The reader could not have guessed that number.
- **The distillation output is read honestly:** "the $\tau = 10$ row reads $(-0.719, 0.413, 0.306)$, and the first component overshoots its limit on the way". The stored output shows this.

## Structural findings

**INF-S1 (major, thought). The section re-proves what chapter 41 has proved, and says chapter 41 only "used" it.**
`mdl-maximum-likelihood.md` ("Cross-Entropy and the Entropy Floor") defines entropy as average surprise, defines KL and cross-entropy, derives $\textrm{CE} = H + D_{\textrm{KL}}$, and proves Gibbs' inequality from $\log u \le u - 1$. `mdl-exponential-family.md` ("Maximum Entropy") defines entropy again, proves the same inequality as a lemma, and proves that an exponential-family member has the largest entropy under moment constraints, with three corollaries: uniform on a finite set ($\log K$), Gaussian for fixed mean and variance, geometric with $r = m/(1+m)$ for a fixed mean on the counts. This section then proves the uniform bound (by Jensen), Gibbs (by Jensen), and the Gaussian bound (by the chapter 41 argument specialised), and sets the tangent-line proof and the geometric case as exercises 4 and 5. The front-to-back reader meets Gibbs' inequality proved three times and Gaussian maximum entropy twice. The only acknowledgements are "Here we take that quantity apart" and ":numref:`sec_mdl-exponential-family` used the answer", and the second is not accurate: that section proved the answer.
Direction: make this section the single home of entropy, KL, cross-entropy and Gibbs' inequality (the assembling step should ask the chapter 41 reviser to state and point forward). For maximum entropy pick one home. If chapter 41 keeps the general proposition, this section states the Gaussian bound as a two-line consequence with a reference and keeps only the value $\tfrac12\log(2\pi e\sigma^2)$; exercise 5 goes. If this section keeps it, chapter 41 cites it.

**INF-S2 (major, pedagogy). Differential entropy interrupts the discrete argument and uses KL before it is defined.**
The line of thought entropy, KL, cross-entropy is the section's subject. After "Maximum Entropy on a Finite Set" the text turns to densities, Jacobians and a quantization limit for about 550 words, and has to say "This average log-ratio is the KL divergence, defined below." The message of the subsection (differential entropy depends on units, a ratio of densities does not) is then stated three more times: at the end of the subsection, in the paragraph after the decomposition cell ("The decomposition also tells which of the three quantities survive a change of units"), and in the summary. The October review flagged the order; it is still open.
Direction: move "Differential Entropy" to after "Cross-Entropy", directly before "KL Divergence Between Gaussians", and fold the change-of-units paragraph into it. The reader then has KL in hand, the invariance is said once, and the three continuous subsections (differential entropy, Gaussian KL, Gaussian maximum entropy) sit together.

**INF-S3 (moderate, thought). The last third applies one fact five times.**
After the decomposition, the section states that cross-entropy is minimised at $Q = P$ (under "Cross-Entropy"), then as a proposition for conditional distributions ("The Irreducible Loss"), then as strict propriety ("Strict propriety of the log score is Gibbs' inequality restated"), then as a proposition for a smoothed target, whose proof begins "This is the irreducible-loss proposition with $\mathbf{p}^\epsilon$ in the role of the label distribution", then for distillation ("By Gibbs' inequality, the student's optimum is to match the softened teacher"). The name Gibbs appears 14 times before the exercises. Each use is correct. Together they read as a template: target, decomposition, Gibbs, number.
Direction: keep "The Irreducible Loss" as the one proposition, since it carries the conditional version and the three cases. Reduce label smoothing to its real content, which is the finite logit gap $\ln 91$ and the nonzero floor $H(\mathbf{p}^\epsilon)$, with the optimum stated in one clause. Keep the scoring-rules subsection (200 words, one new idea: the Brier score is bounded where the log score diverges).

**INF-S4 (moderate, pedagogy). Knowledge distillation does not belong here in this form.**
The subsection is about 800 words with a four-tab cell. Its information-theoretic content is one sentence (the optimum matches the softened teacher). The rest is a first-order expansion of the softmax that explains the $\tau^2$ factor, which is a fact about gradient scale. No main-text section teaches distillation (`training-recipes.md` lists it under what it skips), so the subsection has no reader arriving from a reference, and it names no use by label. It also displaces material this chapter needs earlier: joint entropy, conditional entropy and the chain rule are defined only in `mdl-mutual-information.md`, although the next section uses $H(X_1, \ldots, X_n)$ and $H(X_n \mid X_{<n})$.
Direction: cut distillation to a paragraph plus the exercise that already exists (exercise 10 derives the high-temperature limit), or move it to the main text where a model is distilled. Use the space for a short "Joint and Conditional Entropy" subsection after "Entropy" (definition, chain rule $H(X, Y) = H(X) + H(Y \mid X)$, conditioning does not increase entropy, each with the dyadic source or the label-noise example), in the order of Cover and Thomas chapter 2. The irreducible-loss floor is a conditional entropy and would then have its definition before its use.

**INF-S5 (moderate, pedagogy). Six of thirteen cells only confirm hand arithmetic.**
`information-theory-self-information` ("The output reproduces the four numbers of the text"), `information-theory-definition` ("the number of questions we counted by hand"), the first two lines of `information-theory-definition-2`, `information-theory-kl-categorical`, `information-theory-formal-definition-2` and the four-tab built-in loss cell ("The built-in loss reports $0.9486$, the value computed by hand") each print a number the text has just computed in one line. Predict and confirm is the house method, but it earns its place only when the prediction could have failed or the number is not obvious. The cells that do teach are the $Q'$ line (infinite KL), the irreducible-loss table, the overconfident smoothed prediction and the distillation table.
Direction: define `entropy` and `kl_divergence` once in one cell with the dyadic source, the pair and $Q'$; drop the self-information cell (or keep only the $\ln 52!$ line); drop the hand-versus-built-in pair, whose real content ("Passing probabilities to a loss that expects logits would apply the softmax twice") is API advice that the main text already gives.

## Passage-level findings

**INF-1** (moderate, thought). Under the section opening. Anchor: "model. Here we take that quantity apart. Two questions remain open at the"

Two problems in one place. "Here we take that quantity apart" tells the reader this is new; chapter 41 already took it apart (INF-S1). And the opening announces the two questions that the closing paragraph will leave open, in the same words the closing uses ("Two questions remain open"). The reader has not yet read anything that would make those questions meaningful. Fix: end the third paragraph after "taken from the empirical distribution of the data to the model", say in one clause what chapter 41 did and what this section adds (the meaning in questions, the floor for noisy labels, densities), and delete the preview of the closing.

**INF-2** (minor, pedagogy). Under the section opening. Anchor: "The chapter uses the symbols of :numref:`tab_mdl-it-notation`; each later"

An 18-row notation table stands between the opening and the first idea. Seven rows belong to later sections, and the caption says so. A reader who came for entropy has to scroll past couplings and critics. Fix: move the table to the chapter introduction, or keep here only the six rows this section uses.

**INF-3** (minor, language). Under "Self-Information". Anchor: "The output reproduces the four numbers of the text. Naming the order of a"

The first sentence is the weakest kind of output reading. The second is the memorable one. Fix: cut the first sentence; if the cell goes (INF-S5), move "as much information as 226 fair coin flips" into the paragraph that lists the four values.

**INF-4** (moderate, language). Under "Self-Information" and throughout. Anchor: ":cite:`Shannon.1948` measured this *surprise* by the *self-information* of an"

The section calls $-\log p$ "self-information" in definitions and "surprise" everywhere else. The main text (`softmax-regression.md`, `language-model.md`) and the next section call it "surprisal", a word this section never uses. A reader arriving from the main text looks for "surprisal" and does not find it; a reader going on to the next section meets it undefined. Fix: introduce all three at the definition ("the *self-information* or *surprisal* of the outcome; we also say surprise") and prefer "surprisal" for the quantity, as the rest of the book does.

**INF-5** (moderate, pedagogy; proofs). Under "Maximum Entropy on a Finite Set". Anchor: "**Proof.** Write the entropy as the expectation of $\log(1/p(X))$, a concave"

The proof is correct and handles the support well. It is not the short one for this section. Once Gibbs' inequality is available the bound is one line, $\log k - H(P) = D_{\textrm{KL}}(P \| U) \ge 0$, and it is the same argument the section uses later for the Gaussian ("the cross-entropy to the Gaussian is the same for every competitor $P$") and in exercise 5 for the geometric distribution. As written, the reader sees two unrelated proofs of two maximum-entropy facts and is not told they are one fact. Fix: at this point state the bound, show the coin and the figure, and say the proof comes with Gibbs. After Gibbs, give one paragraph: if $-\log r(x)$ is constant on the constraint set, $\textrm{CE}(P, r)$ is the same for every admissible $P$, so $H(P) = \textrm{CE}(P, r) - D_{\textrm{KL}}(P\|r) \le H(r)$. Uniform, Gaussian and geometric are then three lines each. Jensen is still needed once, for Gibbs.

**INF-6** (minor, thought). Under "Maximum Entropy on a Finite Set". Anchor: "The uniform distribution hedges the most, so it is the hardest to guess. For"

The paragraph ends with two forward references in two sentences ("as the next subsection shows", "we prove this once cross-entropy is available"). With the reordering of INF-S2 and INF-5 both disappear.

**INF-7** (moderate, explanation). Under "Differential Entropy". Anchor: "factor. This average log-ratio is the KL divergence, defined below."

A term used before its definition, in a section whose purpose is to define it. See INF-S2.

**INF-8** (minor, language). Under "Differential Entropy". Anchor: "built from a ratio of two densities does not."

This is the second statement of the sentence that opened the third paragraph of the subsection ("A quantity built from the ratio of two densities behaves better"). Say it once, after the example.

**INF-9** (minor, explanation). Under "The Kullback--Leibler Divergence". Anchor: "that a code built for $Q$ wastes $0.57$ bits per symbol on long sequences"

"Code" appears here for the first time; until now the section has spoken of questions. The claim is right (it is the arithmetic-coding statement of the next section), but the reader has no code yet. Fix: stay in the questions game ("a questioner who believes $Q$ asks 0.57 more questions per outcome, on average over long runs") or add "(:numref:`sec_mdl-coding-compression` builds the code)".

**INF-10** (minor, language). Under "Gibbs' Inequality" and two other proofs. Anchor: "The step that matters is the second line. The expectation runs over the"

Naming the decisive step is good practice (guide section 10.2: "which step is the hard one"). The same phrase is used in all three proofs of this section ("The step that matters is the sum", "The step that matters is the second line", "The first step is the one that matters"), in the Kraft proof of the next section ("The key step is"), and in chapter 41's maximum-entropy proof ("The step that matters is the second"). A fixed phrase becomes a template. Fix: keep the judgement, vary the words, and drop it where the proof has only one step.

**INF-11** (minor, pedagogy). Under "KL Divergence Between Gaussians". Anchor: "Does the Monte Carlo estimate agree with the formula for the unit mean shift?"

The cell is worth keeping, but the text does not say why a reader should care that a sample average reproduces a closed form. The reason is that this is how KL terms are estimated whenever no closed form exists (the reconstruction term of a variational autoencoder, every estimator in :numref:`sec_mdl-mi-estimation`). One sentence saying so turns a check into a method. The standard-error reading ("standard error $0.0022$") is good and should stay.

**INF-12** (minor, explanation). Under "KL Divergence Between Gaussians". Anchor: "The two directions differ by a factor of $2.5$. When the narrow model has to"

The cell prints `(kl_pq, kl_qp)` and the figure has panels (a) and (b) in that order; the paragraph explains the second number first and the first second, then returns to (a) and (b). The last sentence ("The two directions contrast a model that spreads its mass to cover the data with one that concentrates it") is too compressed to mean anything to a reader who does not already know the mode-covering and mode-seeking story. Fix: explain in the order (a), (b), and replace the last sentence by a pointer: "Which of the two a fitted model prefers is the subject of :numref:`sec_mdl-fwd-vs-rev-kl`."

**INF-13** (major, thought). Under "Maximum Entropy and the Gaussian". Anchor: "density is then the hardest to predict? :numref:`sec_mdl-exponential-family`"

The sentence continues "used the answer". That section did not use the answer; it proved it, as a corollary of its maximum-entropy proposition, by the argument given here. See INF-S1. The proof below the anchor is correct (I checked the case $h(P) = -\infty$, which the inequality covers trivially) and is elegant; the question is only whether the book needs it twice.

**INF-14** (minor, language). Under "Cross-Entropy Loss from Logits". Anchor: "loss that expects logits would apply the softmax twice. Does each framework's"

A four-tab cell follows whose only output is the number $0.9486$ that the previous cell printed. The paragraph before it is API guidance (logits against probabilities, numerical stability with a reference). In the tabs the prose fragments into four near-identical blocks for one number. See INF-S5.

**INF-15** (minor, correctness; confident). Under "The Irreducible Loss". Anchor: "annotation errors. With ten classes and $10\%$ of the labels flipped"

The floor $0.545$ nats and the "best accuracy is $1 - \rho = 90\%$" hold only if the clean label is a function of the input, and the accuracy claim also needs $1 - \rho > \rho/(k-1)$. Exercise 7 states both hypotheses; the text states neither, although the same paragraph has just said "the same image can carry different labels, from ambiguity". Fix: add "if the clean label is determined by the input" to the sentence and to the third bullet.

**INF-16** (moderate, thought; repetition). Under "The Irreducible Loss". Anchor: "label surprises it. Only the Bayes predictor reaches the floor of $0.545$. A"

The value $0.545$ is given in the subsection opening ("the answer will turn out to be $0.545$ nats"), derived in the third bullet, printed by the cell, restated after the cell, and restated in the closing paragraph; exercise 7 quotes it again. Likewise $\ln 91 \approx 4.51$ three times and $\ln 10 = 2.30$ four times. Guide section 7.7: a number is derived once. Fix: do not announce $0.545$ before deriving it (state the question only), let the cell confirm it without the sentence after, and drop it from the closing paragraph.

**INF-17** (minor, thought). Under "Label Smoothing". Anchor: "**Proof.** This is the irreducible-loss proposition with"

A proposition whose proof says it is the previous proposition is not a proposition. See INF-S3. Draft replacement for the proposition and proof: "By :eqref:`eq_mdl-ce_decomp` and Gibbs' inequality the loss is smallest at $\mathbf{q} = \mathbf{p}^\epsilon$, where it equals $H(\mathbf{p}^\epsilon)$."

**INF-18** (moderate, pedagogy). Under "Knowledge Distillation". Anchor: "*Knowledge distillation* :cite:`Hinton.Vinyals.Dean.2015` trains a small"

See INF-S4. If the subsection stays, it needs a sentence saying where the reader will use it (by label), and the claim "The raw gradient decays like $1/\tau^2$" should either be shown (print the raw gradient next to the scaled one) or removed, since the table prints only the scaled gradient.

**INF-19** (moderate, language; repetition). Last paragraph before the summary. Anchor: "We can now read any log-loss as an average surprise with a floor, and compute"

The closing paragraph restates four results with their numbers, then the summary restates them a third time. Guide section 12.2 asks that the body end on its last result or the question it leaves open, and names "We can now ..." as the formula to avoid; the next section closes with the same words ("We can now build a prefix code"), and "We can now answer the last question of the opening" opens "The Irreducible Loss". Fix: keep only the two open questions (the code that was promised but not built; the infinite KL of a model with thin support), which are a good bridge, and let the summary carry the results.

**INF-20** (minor, language). Under "Summary". Anchor: "maximum likelihood. For continuous data,"

The summary is one paragraph of nine sentences that lists every result in order of appearance. It would serve the lookup reader better as the answers to the opening questions: what the loss measures, what the floor is, when the floor is reached, and what changes for densities and softened targets.

## Exercises

Ten problems, inside the band for the appendix. The best are 6 (quantized against differential entropy; the hint value $1.4292$ at $\Delta = 0.5$ is correct, and part (d) connects to binned regression), 7 (the label-noise floor, with a part that asks which explanations the floor rules out and a Fashion-MNIST part that makes the reader find the violated assumption), 8 (the initial-loss formula $\ln k + \tfrac{s^2}{2}(1 - \tfrac1k)$, which I checked by simulation) and 9 (label smoothing on wrong-class logits, leading to the Müller, Kornblith and Hinton observation). These teach something the text does not.

- **Order.** The problems are not in the order of the text: Gaussian KL comes first, the uniqueness of self-information third. Reorder to follow the section.
- **Exercise 4 (Gibbs from the tangent line)** presents as "a second proof" the proof that chapter 41 gives in its text. Either cut it or cite `subsec_mdl-nll-crossentropy`; part (c), why the two proofs are the same, is the only part with content and could become a remark.
- **Exercise 5 (geometric maximum entropy)** is a corollary stated in chapter 41 with the same $r = \mu/(1+\mu)$. Cut, or keep only part (c) on why no maximum exists on $\mathbb{R}$, which the text already answers in its last sentence before "The Classification Loss".
- **Exercise 8(c)** asks what the second-order formula says about a loss of $4.1$. The formula gives $s = 2$, where it is already off by about $0.34$ nats (simulation: $3.76$ at $s = 2$), as part (b) will have shown. Add "Is the formula still trustworthy at that scale?" so the reader does not report $s = 2$ as the answer.
- **Missing.** No problem uses joint or conditional entropy, because the section does not define them. If INF-S4 is adopted, add one on the chain rule with the label-noise example (entropy of the noisy label as entropy of the clean label plus conditional entropy of the noise).

## Still open from October

The October review (then chapter 28) asked for the dyadic source in the text and the coding reading of KL at its definition; both are done, and the binary-entropy notation is now $H_b$ here. Still open: differential entropy before KL with "defined below" (INF-S2, INF-7), and joint and conditional entropy missing from the first section (INF-S4). October also proposed cutting the Brier score; I would keep the short scoring-rules subsection.

## For the chapter review

- This section is the foundation the other four build on, and its first half (self-information through cross-entropy) is the best-taught part I read. Its second half is applications of one inequality and could lose 1,500 to 2,000 words.
- It overlaps heavily with chapter 41 (entropy, KL, Gibbs, maximum entropy). The assembling step should assign one home for each and tell both revisers. I recommend this section for Gibbs and chapter 41's proposition for maximum entropy.
- It should hand the next section three things it currently does not: the word "surprisal", joint and conditional entropy with the chain rule, and one sentence that a questioning strategy is a binary code.
- Compared with the main text (`softmax-regression.md`, "Entropy and Cross-Entropy", about 400 words), this section is clearly deeper: proofs, densities, the noise floor. The level is consistent. The main text states Gibbs and points here, as guide section 7.6 asks.
- Patterns likely to hold across the chapter: a question sentence before every code cell; "The step that matters"; closing paragraphs that begin "We can now" and repeat the summary; numbers restated three to five times.
