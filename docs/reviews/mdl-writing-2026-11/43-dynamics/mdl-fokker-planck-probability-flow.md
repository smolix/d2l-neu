# The Fokker--Planck Equation and Probability Flow

Source: `chapter_mdl-dynamics/mdl-fokker-planck-probability-flow.md`. About 7,550 words of body prose and 2,350 words of exercises (12 problems), 9 code cells, 6 figures. All code is NumPy; outputs checked against `outputs/pytorch/chapter_mdl-dynamics/mdl-fokker-planck-probability-flow.json`, and two claims were re-run.

## Verdict

This is the best section I read in the part and close to the bar. It has one line of argument (the density of a noisy cloud obeys a PDE; the PDE's smoothing term is transport in disguise; so the same density can be moved by an ODE, and run backward by a family of processes that differ only in how much of the score they use), and every step is first done on a case small enough to check by hand: the heat equation at one point, the outward flow $x\mapsto x\sqrt{v_t/v_0}$, stationary OU as its own reverse. The derivation of the Fokker-Planck equation is a real derivation with its assumptions stated; the deterministic-twin proposition is proved; time reversal is proved for marginals and Anderson's stronger theorem is cited, and the text says which is which. I recomputed every derivation and number and found no mathematical error. The most important thing to fix is a gap in exactly this honesty: three statements identify "a solution of the Fokker-Planck equation" with "the marginal of the SDE" without the uniqueness that would license it (the mixture marginal, the hypothesis of the twin proposition, and the forward direction of the family proposition). Each is repaired by a sentence, and for the mixture by a one-line probabilistic argument that makes the PDE check a confirmation instead of the foundation. After that: the opening experiment repeats one from the SDE section, and the committed output of the reverse sampler is an unlucky draw that contradicts the prose.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought | 4 |
| Pedagogy | 5 |
| Language | 4 |
| Ease and enjoyment | 4 |

## What works

- **The opening question**: "The randomness of the paths averages out, and the density of the cloud moves deterministically. Which equation moves it?" The October review already marked this to keep.
- **Vector calculus taught for use.** Divergence as outflow from a box with two fields that "show the range" (expansion with divergence 2, rotation with divergence 0, and "A zero divergence therefore does not mean a zero field"); the Laplacian as the gap to the neighbourhood average with the one-dimensional check ($1.09$ at $x=1$, $r=0.3$). Each definition has the five parts guide section 8.3 asks for.
- **The weak derivation** states its standing assumptions up front, explains test functions as bumps that measure the probability of a box, and justifies the last step in a parenthesis a student can follow: "a bump placed where it is positive would give a positive integral".
- **"Before any general statement, we check it at one point."** The heat equation verified at $x=0$, $t=1$ ($-0.1995$ on both sides). This is the method of the best sections of the book.
- **Diffusion as transport**, the centre of the section. The heat-equation flow is solved in closed form, "every particle doubles its distance from the center", against a Brownian particle that wanders; the figure shows smooth non-crossing curves next to jagged ones. Then the limit: a point mass needs a velocity $x/(2t)$ that blows up, "because a Lipschitz flow maps one point to one point". A reader finishes this subsection understanding the probability-flow ODE.
- **The KS read-out**: "A strictly increasing map that carries $p_0$ to $p_t$ carries the sample's empirical CDF along with it, so the ODE cloud keeps the distance its starting sample had". The stored output shows $0.0029$ three times. This is an observation an author made by looking, and it teaches what a coupling is.
- **The factor of two.** Stationary OU as its own reverse, the wrong factor shown to give Brownian spreading, then the $\lambda$ family with "a reverse process with noise $\lambda g$ smooths its own density, so its drift must undo two smoothings". Clear, memorable, and correct.
- **The bias recursion** (`em_reverse_width`): the Monte Carlo error hides the first-order bias, so the text computes it exactly. Excesses $0.0096, 0.0047, 0.0024, 0.0012$.

## Structural findings

1. **The opening experiment is the previous section's.** `#fokker-planck-cloud-marginals` starts a cloud at $x_0=2$, runs Euler-Maruyama on OU and compares mean and standard deviation with the kernel. `#sdes-ou-cloud` did the same thing with the same start. The kernel is also restated under a second label (`eq_mdl-dyn-ou-marginal` next to `eq_mdl-sde-ou-kernel`). For front-to-back readers this is a repeat; for readers who land here it is useful. Keep the histogram (the SDE section printed a table and drew nothing), cut "Pathwise and Ensemble Views" to its one new idea (the time-marginal averages over start and noise), and cite the kernel by its original label.
2. **Solution of the PDE against marginal of the SDE** (findings FPK-4, FPK-7, FPK-11). The section proves "marginal implies solution" and silently uses "solution implies marginal" in three places.
3. **Scope.** The vector-calculus preliminaries (about 650 words) are not in the calculus chapter, which has no divergence or Laplacian. They are well done and used at once, so they can stay, but a reader who looks up "divergence theorem" in the appendix will look in Calculus. Either move them there with a pointer, or have the calculus chapter point here.
4. **Length is proportionate.** Unlike its neighbour, this section has little that could be cut: the opening duplicate (finding 1), the boundary-condition sentence if exercise 8a is reworded, and the likelihood paragraph could be shortened since nothing in the body uses it. It is not two sections; time reversal depends on every earlier part.
5. **Pointers to the main text are missing.** `chapter_diffusion-models/samplers.md` and `gaussian-paths.md` point to `sec_mdl-time-reversal` and `sec_mdl-probability-flow-ode`. Here, "diffusion-model samplers use this freedom" and "how likelihoods of diffusion models are reported" name uses without a label.

## Passage-level findings

**FPK-1** (minor, thought). Introduction.
Anchor: `A single noisy path cannot be reversed,`
"because it has forgotten where it started, but its density can" is a figure of speech that is not literally true once removed: a path is a record and forgets nothing; what cannot be inverted is the map from start to end, because many starts reach the same end. And "its density can" invites the objection of FPK-12. Rewrite: "No map takes the endpoint of a noisy path back to its start, since many starts lead to the same endpoint. But we know how the density of the whole cloud changed, and we can build a process whose density changes the other way."

**FPK-2** (moderate, pedagogy; repetition). Under "Simulated Marginals".
Anchor: `The opening claimed that a large cloud has the same histogram on every run.`
See structural finding 1. Also, the cell does not test the claim as stated: it runs once. "The paths differ from run to run, but the density does not" is supported by agreement with the formula, not by a second run. Say that, or overlay two runs.

**FPK-3** (minor, explanation; notation, 4 instances). Under "Divergence, Laplacian, and Integration by Parts".
Anchor: `a smooth scalar field.`
$h$ was the step size throughout the two previous sections and in the chapter's notation table; here it is a scalar field, and $\Delta h$ now means its Laplacian where $\Delta t$ and $\Delta W$ meant increments a page ago. Rename the scalar field ($\psi$, or use $\phi$ at once since it becomes the test function) and say in one clause that $\Delta$ without an argument of time is the Laplacian. Related clashes in this section: $q$ is the ODE's density, then the time-reversed density $q_{\bar t}$, then the reference law $q_T$ in exercise 12; $\lambda$ is an eigenvalue in the chapter's notation table and the sampler's noise scale here.

**FPK-4** (moderate, correctness; confident that the argument has a gap, the result is right). Under "Linearity and Mixtures".
Anchor: `No proposition about a single Gaussian covers the mixture; linearity does.`
Superposition shows that the mixture of evolved components solves the PDE with the right initial condition. It does not show that this solution is the marginal of the SDE unless the solution is unique, which the section never states for the forward equation. The read-out of the residual cell repeats the gap: "the simulation above showed that the SDE's particles follow such a solution" refers to the point-start cloud, not the mixture. There is a one-line argument that needs no PDE: given the component, $X_0$ is Gaussian and independent of the noise, so $X_t$ is Gaussian by the OU solution; averaging over the component gives the mixture. Lead with that, then present superposition as the same fact seen in the PDE and the residual cell as its check. The sentence "No proposition ... linearity does" then goes.

**FPK-5** (minor, pedagogy). Under "Linearity and Mixtures", after `#fokker-planck-ou-residual`.
Anchor: `This is the truncation`
"This is the truncation error of the finite differences" is asserted. One clause would show it: the residual should fall by four when the grid spacing is halved (second-order stencils). As it stands the reader must trust that $4\times10^{-6}$ is error and not a small genuine mismatch.

**FPK-6** (minor, language). Under "Diffusion as Transport".
Anchor: `A Brownian particle started at $x=1$ instead lands anywhere`
"lands anywhere in $1\pm1.73$ (one standard deviation, $\sqrt3$)" says two different things; a third of the particles land outside. Write "lands at $1$ plus a draw of standard deviation $\sqrt3\approx1.73$".

**FPK-7** (minor, correctness; confident). Under "The Probability-Flow ODE".
Anchor: `Let $p_t$ be a smooth positive`
"a smooth positive solution of the Fokker--Planck equation for $d\mathbf X=\ldots$" leaves open whether $p_t$ is the SDE's marginal or any solution, and the conclusion "the ODE and the SDE share all time-marginals" needs the former. Write "Let $p_t$ be the marginal density of the SDE, assumed smooth and positive, so that it solves the Fokker--Planck equation." The hypothesis "with the decay of the previous subsection" is not used in the proof and can go. The proof itself (both densities obey the same log-density rule along characteristics that cover the space) is correct and is the short one.

**FPK-8** (minor, explanation). Under "Bayes' Rule on an Infinitesimal Step".
Anchor: `Adding the linear term to the Gaussian exponent`
The exponent is written with $\mathbf f$ without an argument. In the forward kernel it is $\mathbf f(\mathbf x_t)$, a function of the variable being solved for; the display that follows has $\mathbf f(\mathbf y,t)$; the justification ("changes the mean only at order $\tau^{3/2}$") comes two sentences after the display. Move that sentence before the completion of the square: "To this order we may evaluate $\mathbf f$ at $\mathbf y$, since $\mathbf f(\mathbf x)-\mathbf f(\mathbf y)$ is of size $\sqrt\tau$ and is multiplied by $\tau$."

**FPK-9** (moderate, explanation; level). Under "Anderson's Theorem".
Anchor: `The one-step result extends to the whole process`
This is the one place where the status of the argument is not stated when it is needed. The Bayes step is a heuristic (first-order expansions, approximate kernels), the extension to the process is Anderson's theorem and is not proved, and what the section does prove, six paragraphs later, is the weaker statement about marginals. The reader learns this only at the end of the next subsection ("Anderson's theorem itself is stronger than the proposition"). Say it here: "The one-step calculation is a heuristic. Anderson proved that it gives the right answer for the whole process; we do not prove that. Below we prove the part that sampling needs, that the reversed process has the marginals $p_t$." Also, a subsection titled "Anderson's Theorem" should contain a statement set off as a theorem with its hypotheses, or be retitled "The Reverse-Time SDE".

**FPK-10** (minor, explanation). Proof of the family proposition.
Anchor: `(Other drifts with the same marginals differ from it by a field`
A three-line parenthesis about non-uniqueness of the drift interrupts the proof between finding $\mathbf b'$ and converting it to forward time. Move it after the proof, where "The proof also explains the factor of two" begins, as its own sentence.

**FPK-11** (minor, correctness; confident). Statement of the family proposition.
Anchor: `*Let $p_t$ solve the`
The statement has no regularity hypotheses, while the proof invokes "our standing assumptions" and, for $\lambda=0$, the Lipschitz condition of the twin proposition. Add "under the hypotheses of the previous proposition" to the statement. Same wording issue as FPK-7: $p_t$ should be the forward marginal.

**FPK-12** (minor, pedagogy; a missing objection). Under "Time Reversal".
Anchor: `Generation needs the process played`
A reader who knows any PDE theory has been told that the backward heat equation is ill-posed, and this section has just called the Fokker-Planck equation a heat equation with drift. The text never says why running it backward is legitimate here. One or two sentences in the manner of guide section 8.5: "The heat equation cannot be solved backward from an arbitrary final density: smoothing destroys detail that nothing can restore. We do not solve it backward. We know the whole forward family $p_t$, and ask only for a process that visits it in reverse order; the price is that its drift contains $\nabla\log p_t$, which carries the destroyed detail." This also prepares the stiffness near $t=0$ that the last cell finds.

**FPK-13** (moderate, thought; claim against stored output). Under "Reverse-Time Sampling".
Anchor: `splits the mass within`
The prose says the sampler "splits the mass within about $0.01$ of one half (the binomial standard deviation is $0.0035$)" and that "the KS distance to the exact $p_0$ is usually below $0.01$". The committed output prints `mass on the left: 0.487` and `KS to the exact p_0: 0.0148`. The first is $3.7$ binomial standard deviations from one half; in 40 reruns of the cell I obtained fractions between $0.492$ and $0.507$ (standard deviation $0.0038$). So the prose describes typical runs correctly and the stored run is roughly a one-in-several-thousand draw. Ranges are the house policy and are fine, but a reader who sees $0.487$ next to "standard deviation $0.0035$" will conclude that the sampler is biased toward the right mode. Re-capture this cell when outputs are next refreshed; no prose change is needed.

**FPK-14** (minor, thought; claim without evidence). Under "Reverse-Time Sampling".
Anchor: `Making the steps ten times finer on`
"removes nearly nine tenths of the bias" is not computed by any cell. I checked it with the recursion on a non-uniform grid: the excess falls from $0.0047$ to $0.0006$, so $87\%$; the claim is true. Either let `em_reverse_width` take a grid and print the number, or turn the sentence into an exercise ("where should the steps go?"), which would be a good one.

**FPK-15** (minor, language). End of body and under "The Heat Equation and the Two Terms".
Anchor: `We can now write the density equation of any SDE with scalar noise`
The closing paragraph recaps the section in "We can now ... We can replace ... We can also ..." and the Summary recaps it again. Same formula as the end of the SDE section. Keep the last two sentences of that paragraph and the two that follow ("For images we have samples of $p_0$ and nothing else").

**FPK-16** (minor, language; pattern, 4 instances). Throughout.
Anchor: `The smallest case shows each term at work.`
"The smallest case ..." opens four worked examples here (Laplacian, heat equation, Gaussian score, stationary OU) and two in the SDE section. The device is the section's strength; the phrase is becoming its signature. Vary two of them ("Switch the drift off.", which is already the next sentence, can open the paragraph alone).

## Exercises

Twelve problems, within the band, graded, and almost all extend the text. Best: stationary density from zero current, leading to $p_\infty\propto e^{-2U/g^2}$ and the double well (3); same marginals with different couplings (5), which makes the reader measure what the KS read-out explained; state-dependent noise on a circle, where the naive PDE predicts the wrong stationary law (7); the exact Gaussian reverse step that recovers the full $g^2$ (10); the mismatched pairs of drift and noise (11). I checked the stated formulas in 1b, 3, 5 (the hint's slope $-61$ and the variance $4.25$), 6c ($t^*=\tfrac12\ln 4.75=0.78$), 7, 9d and 10a.

- Exercise 12 (prior mismatch, five parts, about 400 words) is a project and the only use of the data-processing inequality; good, but the first candidate if the list must shrink.
- Exercise 8d (count $100^d$ grid cells) makes the single most important practical point about the equation, that nobody solves it on a grid, and that point is absent from the body. One sentence in the body would help a reader who wonders why the section never solves the PDE numerically.
- Exercises 8 and 11 have no `:exlabel:`; harmless unless something should refer to them.
- A problem I would add, from FPK-14: place the steps of the reverse sampler to minimize the width bias at a fixed budget.

## Still open from October

October asked to cut matrix diffusion, boundary conditions and JKO. JKO is gone; matrix noise is two sentences and an exercise; boundary conditions are one sentence that an exercise uses. Closed. The remark that the score is defined more than once is half open: this section defines it with a pointer to `sec_mdl-fisher-divergence`, which is acceptable under guide section 7.7 only if the geometry paragraph is not also in the information-theory chapter.

## For the chapter review

- **Fit and order.** The section is the hinge of the chapter: it consumes Itô's lemma, the zero mean of the Itô integral, the OU kernel and Euler-Maruyama from the SDE section, the uniqueness theorem and Heun from the ODE section, and the instantaneous change of variables from the neural ODE section, each by label. It hands the next section exactly one unknown, the score. The order is right.
- **Overlap.** One duplicated experiment and one duplicated equation label with the SDE section (structural finding 1). The log-density rule is derived here from the PDE and in the neural ODE section along trajectories; the text says so, and the two derivations are different enough to keep.
- **Level.** Honest, with the exceptions named in FPK-4, FPK-7, FPK-9 and FPK-11, all of the same kind: uniqueness for the forward equation is used and not stated. A single sentence under "From Itô's Lemma to the PDE" ("under our assumptions the equation has one solution with a given start; we use this without proof") would cover all of them.
- **Patterns for the chapter.** "The smallest case" as an opener; body closings that begin "We can now"; framework import tabs on NumPy-only code (here too, plus a second import cell for `erf`); symbols reused across sections ($h$, $\Delta$, $q$, $\lambda$).
- **What a deep learning reader would look for and not find.** Why the reverse process is well defined although the backward heat equation is not (FPK-12), and that the equation is never solved on a grid in practice (exercise 8d only).
