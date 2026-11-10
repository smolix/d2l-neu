# Relativistic Objectives

Source: `chapter_generative-adversarial-networks/relativistic.md` (lines 1 to 562, slides not read). About 4,500 words of body prose, 1,200 words of exercises, 9 code cells (NumPy, shared by both tabs). Stored outputs checked (`relativistic.json`, identical in both tabs), the rank-weight plot and :numref:`fig_gan_pairing` rendered. The two-atom numbers, the expansion in part (b), the weights and the two-point landscape recomputed; joint convexity tested numerically.

## Verdict

The mathematics here is the most elegant in the chapter and, as far as I can check, correct: the stationarity argument, the lifting of the pairing game to a log-loss game between $p \otimes q$ and $q \otimes p$, the bounds, the local factor of two, and the cycle argument for the paired loss all hold. The two-atom example is as good as anything in the book. But the section is out of proportion and does not know what it wants the reader to take away. It is the longest of the three opening sections, it spends a lemma, three propositions and a theorem on one loss, it trains nothing, and nearly everything it proves about pairing is negative: the value still saturates at $\log 2$, training still does not converge, the rank weight still permits collapse, and the one positive result (no bad minima) holds for a fixed coupling that minibatch training does not use. That last judgment is the most valuable thing in the section and it is made in passing. The next section then credits pairing with mode coverage on the strength of one ablation table from one paper, and the reader is left to reconcile the two. The most important fix: cut the section to about half, lead with the honest verdict, and say plainly that the case for pairing in practice is empirical (REL-9, structural findings 1 to 3).

## Scores

| Dimension | Score | Note |
|---|---|---|
| Clarity of explanation | 4 | Two atoms, then the general statement; the lifting is clear; the cycle argument needs a picture |
| Clarity of thought and focus | 3 | Correct results, no ranking of them; the main judgment is buried; three checks of what was just proved |
| Pedagogy | 3 | Good ladder from two atoms to the theorem; no training, no payoff for the builder; the section's best derivations sit in exercises |
| Language | 4 | Clean; a few sentences with voice in the opening |
| Ease and enjoyment | 3 | The opening and the two atoms delight; from "Bounds" on it reads as a paper |

## What works

- **The opening.** "A taster who cannot say whether one glass holds the expensive wine can often say which of two glasses does, and a chess rating is built from who beat whom, not from an absolute score per player." Concrete, true, and it carries the idea. The link to the Bradley-Terry reward model of :numref:`sec_regularized` resolves to the right passage.
- **"Two Atoms by Hand".** A four-column table, the observation that "A pair of equal atoms is equally likely in both orders and says nothing", the posterior $16/17$ against $0.8$, and "Read this as information." The general theorem is then something the reader has already seen happen. All numbers check (0.319, 0.193, ratio 1.66).
- **The lifting idea**, stated before the lemma: "The two-atom table treated the ordered pair as a single observation." The proof labels its three steps in words ("The product ratio separates", "The swap symmetry doubles the objective", "Differences suffice").
- **The remark after the lemma** that part (iii) "depends on the logistic payoff". It tells the reader which step is special.
- **The two-point landscape**: "Put both generated points on $x_1$. Half of the all-pairs comparisons are then ties". Small, exact, and it shows the mechanism.
- **The finding that the all-pairs loss has the same collision minima as the classical loss** (21 at $n = 3$). This is the one cell that tells the reader something the cited theorem does not.
- **Shift invariance** stated with its consequence in exercise 1c: the fraction of generated samples with $D > 0$ is meaningless for this critic. Practical and not obvious.

## Structural findings

1. **Proportion.** About 4,500 words, more than the section that founds the chapter, for a loss whose practical content is four facts: the critic's loss is $\mathrm{softplus}(-(D(x) - D(x')))$ on aligned pairs; the critic's level is arbitrary; the optimal critic is still $\lambda$; the generator uses the non-saturating form. A reader who wants to build needs about 600 words. A reader who wants to understand needs the two atoms, the lifting and the landscape. What can go with no loss:
   - Part (b) of the bounds proposition and its proof sketch (exercise 2 already holds it; keep "twice $\mathrm{JS}$ near $q = p$" as a stated fact).
   - "Verification on Five Atoms" (REL-5).
   - "Convexity in the Generator's Distribution" down to one sentence (REL-12).
   - The rank-weight plot (REL-7).
   - "Diagonal and All-Pairs Estimates" merged into the landscape subsection as one sentence (REL-11).
   - The second equality of the theorem (the entropy form), which is used only by the verification cell and exercise 6.
   A target of 2,500 words is realistic.
2. **Nothing is trained.** Every cell is NumPy on two, three or five atoms. The reader has not seen the pairing loss move a generator when the section ends. The mixture testbed of :numref:`sec_gan_objectives` (`train_toy`, exact $\mathrm{JS}$ after every step) would take the pairing loss as a sixth arm for the price of two lambdas. That experiment could fail to show a difference, which would itself be the honest result this section argues for. This is a redesign for the development machine.
3. **The verdict is buried, and the next section contradicts its drift.** The section shows that the landscape guarantee "belongs to a fixed coupling" and that redrawn pairs average back to the all-pairs loss with as many collapse minima as the classical loss. :numref:`sec_gan_convergence` then says "the pairing contributed the coverage measured in the StackedMNIST ablation" (1,000 against 693 modes, from the R3GAN paper; the figures match my recollection of that paper's table, unverified). Both can be true, but the reader is owed the sentence that connects them: the theory available does not explain why pairing helps under minibatch training; the evidence that it does is one ablation; here is what we think. Put that sentence here, where the theory is.
4. **Weight given to R3GAN.** In this section the paper appears as "the modern recipe" and "the modern baseline of the next two sections". It is one paper from 2024 whose own title makes that claim. Whether it has become the baseline by the time the book is read I cannot confirm. Across the chapter it is named on about 50 source lines and gives its name to a saved training function. That is a bet. It is a reasonable one for a teaching recipe (few parts, each with a stated reason), but the text should say what is the paper's claim and what the chapter has itself shown, and this section should not present a property (coverage through pairing) that its own analysis qualifies as settled.
5. **Whose theorem?** The identity $d_{\mathrm{Rp}}(p, q) = \mathrm{JS}(p \otimes q, q \otimes p)$ is given as "Theorem" with no source, next to a sentence saying an earlier paper proved the value is a divergence "without computing it". If the closed form is new to this book, say so in a clause ("the following closed form, which we have not found in the literature"); if not, cite it. A textbook reader assumes an uncited theorem is standard.
6. **Placement.** Under the author's rule the lifting theorem is specific to this chapter and stays in the main text. But the section could be the third part of :numref:`sec_gan_objectives` (a third choice: score pairs) or the first part of :numref:`sec_gan_convergence` (which trains the loss). See the chapter note.

## Passage-level findings

**REL-1** (minor, pedagogy). Under "The Bradley--Terry Critic".
Anchor: "Comparing one generated sample with several real ones gives the InfoNCE loss of contrastive learning"
An aside to a loss the reader has not met, in the paragraph that defines the objective, pointing to an exercise that points to the appendix. Move it to the exercise's own first line.

**REL-2** (minor, correctness, unverified). Same paragraph.
Anchor: "The paired form used here is abbreviated RpGAN, for relativistic *paired*"
As I recall, the 2019 paper called its paired loss the relativistic standard GAN, and "RpGAN" was introduced in the 2020 paper. The sentence reads as if the 2019 paper used the name. Check and attribute.

**REL-3** (moderate, thought). Under "Shift Invariance and Bilinearity".
Anchor: "which let the critic's problem decouple pointwise and made each value an $f$-divergence"
Not true of half of the previous section. The integral probability metrics are also suprema of functionals affine in $(p, q)$; their critic does not decouple pointwise and their value is not an f-divergence. Decoupling came from a free critic scoring one sample at a time. Say that: "With a free critic and one sample per score, the critic's problem decoupled pointwise. Here each value $D(t)$ meets every other through the differences."

**REL-4** (minor, explanation). Proof of the first proposition.
Anchor: "Perturbing $D$ at a single point $t$, that point enters $\Phi$ once as the real member of a pair and once, with opposite sign, as the generated member"
This is the only functional derivative in the three sections and the reader from Basics has not met $\delta \Phi / \delta D(t)$. On a finite space it is an ordinary partial derivative, and the five-atom code computes exactly that. One clause ("for finitely many outcomes this is the partial derivative with respect to the entry $D(t)$") makes the step safe.

**REL-5** (moderate, pedagogy: cells that confirm). Three places.
Anchors: "The cell below checks the three numbers"; "Because the theorem is an identity, a finite example can verify it without sampling error."; "as computed by hand".
Three cells confirm what the lines above them derived: the two-atom values, the theorem on five atoms "to nine decimal places", and the two-point landscape. A fourth is announced with "at $n = 3$ it checks the theorem". The author's decision allows one or two such cells where they teach something more. The count at $n = 2, 3$ does (it adds the all-pairs result). The two-atom cell repeats arithmetic done in the text. The five-atom subsection proves nothing after a proof and occupies a heading. Cut the five-atom verification and the two-atom cell; keep the five atoms only if the rank weight still needs them (REL-7), with `p`, `q`, `lam` and `sigma` defined there.

**REL-6** (moderate, explanation; appendix reference). Proof of part (a).
Anchor: "the data-processing inequality of :numref:`subsec_mdl-f-div-data-processing` gives"
The label resolves to the right proposition, but the appendix states it "on a finite or countable set" with $P \ll Q$, and the use here is on $\mathcal{X} \times \mathcal{X}$ for general densities. The section already has a proof that needs no appendix: by :eqref:`eq_gan_rp_mi` and the reading of $\mathrm{JS}$ from :numref:`sec_basic_gan`, part (a) says $I(a_1; y) \leq I\big((a_1, a_2); y\big)$, that a pair tells at least as much as one of its members. Exercise 7 holds this argument and the exact gap $d_{\mathrm{Rp}} = 2\,\mathrm{JS} - I(a_1; a_2)$, which also gives the factor of two. Move that into the body in three lines and drop the appendix pointer (or keep it as "also follows from").

**REL-7** (moderate, pedagogy). Under "The Rank Weight".
Anchor: "so the rank weight separates samples that the threshold weight treats as nearly equal"
The plot compares the rank weight with the classical threshold weight applied to a pairing critic shifted by 2. No algorithm does that, so the dashed curve is a strawman, and what the plot shows (the rank weight ignores the shift) is the third statement of shift invariance after :numref:`fig_gan_pairing` and "Formally, replacing $D$ by $D + b$". The quoted conclusion is not connected to any consequence for training, and the next paragraph begins "Neither weight by itself forbids collapse." The subsection ends without a payoff. Keep the definition and the sentence that the weight "is the probability that $x'$ outranks a random real sample"; cut the cell; the output values (0.95, 0.99, 0.75) match the prose.

**REL-8** (minor, explanation). Under "Paired Landscapes".
Anchor: "For the paired loss, draw an arrow from each real point to the position of its generated partner."
The cycle argument is correct and compact, and it is the hardest paragraph of the section to follow in words. It asks for a picture: three data points, arrows, one cycle and one chain, the scores that result. A small pre-generated figure here would do what :numref:`fig_gan_pairing` (two dots and a threshold) does not need to do.

**REL-9** (major, thought and correctness). Same subsection.
Anchor: "A critic that responds to that average faces as many collapse minima as the classical loss, so the one-to-one guarantee belongs to a fixed coupling"
This is the section's main judgment: it qualifies a published landscape theorem and the motivation the R3GAN paper gives for its loss (as I recall, that paper cites the landscape result for robustness to mode dropping; unverified). Three problems. (a) It is stated in general but supported by an enumeration at $n = 2$ and $3$, for a critic at its best response; say so, or prove that every collision is a strict local minimum of the all-pairs loss (it looks provable from the tie argument). (b) Whether a trained critic "responds to that average" or to each pairing is the open modelling question, and exercise 8c asks the reader to decide it; the body asserts one answer. (c) The same conclusion is stated twice in consecutive paragraphs (also "The guarantee belongs to the one-to-one coupling, not to the rank weight"). Rewrite as the section's conclusion, with the three levels kept apart: what is proved (fixed coupling, best response), what the count shows (all pairs, $n \leq 3$), what is conjectured (minibatch training behaves like all pairs), and what evidence exists on the other side (structural finding 3).

**REL-10** (moderate, focus). Under "Saturating and Non-Saturating Generators".
Anchor: "writes its objective as the zero-sum minimax over $\Phi$, and implementing that equation literally puts the generator on the saturating row"
A correction to a paper's equation, followed by an instruction in the imperative ("Train the generator on the non-saturating form instead"). The October review listed "paper errata" among the things that bury this chapter's ideas, and this is still here. The reader needs one fact: use the non-saturating form. :numref:`sec_gan_convergence` states it again where the loss is coded. Reduce to a clause there. I could not check the paper's equation; the description of the reference implementation agrees with my memory of it.

**REL-11** (minor, thought). Under "Diagonal and All-Pairs Estimates".
Anchor: "The all-pairs average has the smaller variance, yet in the experiments of :citet:`Jolicoeur-Martineau.2020` it did not improve the generator."
Reported from a paper, unverified by me. More to the point, the previous subsection has just given a reason to prefer the diagonal (its landscape, for a fixed coupling) and a reason to doubt that the difference survives re-pairing. This subsection mentions neither. Merge it into the landscape discussion as one sentence.

**REL-12** (minor, repetition). Under "Convexity in the Generator's Distribution".
Anchor: "Convexity in $q$ therefore does not exclude local minima in the positions of the generated samples."
The same caveat was made in :numref:`sec_gan_objectives` under an identical heading. The new fact is that joint convexity fails (I confirmed a violation numerically; the text gives no example) while convexity in $q$ survives. One sentence, citing the earlier caveat.

**REL-13** (minor, language). Jensen bound.
Anchor: "At $D^\star = \lambda$ the function $u \mapsto \log(1 + e^u)$ is convex"
The function is convex whatever the critic. "At $D^\star = \lambda$ the loss is the expectation of the convex function ..., so Jensen's inequality bounds it from below".

**REL-14** (minor, language). After the theorem and in further reading.
Anchor: "her normalization is twice ours, so her divergence is $2\, d_{\mathrm{Rp}}$"
Refer to the paper, not to the author by a pronoun: "the paper's normalization is twice ours". What that paper proves is reported from memory of it and is unverified.

**REL-15** (moderate, pedagogy: two endings). Last body paragraph.
Anchor: "We can now compute the value of a pairing game and read a critic through both weights."
This is the recap formula the author asked to remove, followed by a Summary that says the same things. Two facts in it are missing from the Summary and should move there: pairing "does not make gradient training converge", and :numref:`sec_gan_convergence` takes up both open problems.

**REL-16** (minor, pedagogy). Heading "Ranking and Mode Coverage".
Anchor: "Neither weight by itself forbids collapse."
The heading covers five subsections on five topics (convexity, the weight, landscapes, saturation, estimators). After the cuts above, three remain and the heading fits: what ranking does to the generator's push, whether it removes the collapse minima, which generator loss to use.

## Code and experiments

- `relativistic-two-atoms-by-hand`: 0.193, 0.319, 0.319. Confirms the text's arithmetic (REL-5).
- `relativistic-verifying-the-closed-form-1`: three values agree at 0.376889913; $\mathrm{JS} = 0.234$; the prose ("about 0.23", "about 0.38", "about 1.6 times") is correct. A check that cannot teach more than the proof (REL-5).
- `relativistic-verifying-the-closed-form-2`: see REL-7.
- `relativistic-paired-landscapes-1`: prints `[-0.69 -0.5 -0.5]` in units of $\log 2$. The classical entry $-0.69$ is $(\tfrac34 \log \tfrac43 - \log 2)/\log 2$ and looks like $-\log 2$ printed in the wrong unit; the prose does not read that column. Print in nats, or say what the first number is.
- `relativistic-paired-landscapes-3`: 2 and 21 collision minima for the classical and all-pairs losses, none for the paired loss. Matches the prose. This is the cell to keep.
- Numbers no cell prints: the $q$-averaged weights "about 0.24, against 0.76" on the two atoms (recomputed: 0.235 and 0.765, correct).
- The function `losses` returns best-response values in a fixed order that the caller must remember; a dict would read better. `on_cycle` tests `j not in range(len(gen))` to detect a fresh position, which works for the fractional positions used and deserves its comment.
- No training, no framework code, no tab prose.

## Exercises

Eight problems, about a quarter of the body, and the best set of the three sections. Exercise 3 (two Gaussians: the pairing value equals $\mathrm{JS}$ at a separation scaled by a constant) is the kind of problem a reader remembers. Exercise 7 contains the clean proof of part (a) and the factor of two and should supply the body (REL-6). Exercise 8 tests the section's main caveat and its outcome should be reported in the body (REL-9). Exercise 2 duplicates the proof sketch of part (b); if (b) leaves the body, the exercise stands alone. Exercise 6 depends on the appendix's InfoNCE definition; both labels resolve. Nothing to cut.

## Still open from October

Both items named for this section are open: "a lifting lemma" and "paper errata" (REL-10). The lemma is good mathematics and I would keep it, shortened; the erratum should go.

## For the chapter review

- Three sections, about 12,000 words, lie between the garments of :numref:`sec_basic_gan` and the next image, and this one trains nothing at all. Of the three it is the one a reader would skip. The chapter should either fold it into its neighbours (the objective, shift invariance and the non-saturating form into :numref:`sec_gan_convergence`, where the loss is coded and trained; the lifted game and the landscape as a shorter section or a starred one) or halve it in place.
- The chapter's account of what pairing contributes must be made consistent across 16.1 (which promises that ranking replaces the threshold), 16.3 (which shows that ranking alone leaves the minima) and 16.4 (which credits pairing with coverage from one ablation).
- R3GAN: say once, in the chapter introduction or in 16.4, that the chapter adopts one 2024 recipe as its teaching baseline, what that paper claims, and which of those claims the chapter's own experiments reproduce at small scale.
- Appendix: the only load-bearing reference is the data-processing inequality, whose stated hypotheses do not cover the use; the section can do without it. No derivation here duplicates the appendix.
- Pattern across the three sections: correct, careful mathematics with too few statements of rank. The reader is rarely told which result matters, and here the one that does is in a subordinate clause.
