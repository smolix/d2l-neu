# Flow Matching and Optimal Transport

Source: `chapter_mdl-dynamics/mdl-flow-matching-transport.md` (lines 1 to 1684; about 7,600 words of body prose and mathematics, about 2,200 words of exercises, 14 cells, of which 5 are repeated in four framework tabs).

Context the brief did not have: chapter 17 (`chapter_diffusion-models/`) is written (about 83,000 words) and cites this section about ten times (`gaussian-paths.md`, `few-step.md`, `samplers.md`); `chapter_generative-adversarial-networks/convergence.md` cites its Benamou-Brenier theorem.

## Verdict

This is the better of the two sections and, up to the end of the optimal-transport part, close to the bar. It has one problem (curved paths cost network calls), one mechanism (the marginal velocity is an average, and averages bend where segments cross), one closed-form example that shows the failure (two points), one experiment that measures it and repairs it (two moons, reflow: one step after re-pairing equals thirty-two before), and one theorem that says what the repair converges toward (Benamou-Brenier). The proofs are short and each is announced by what it is for. I found no error in a proposition or proof; two hypotheses do not cover the section's own examples (FMT-5). The most important thing to fix is the last part, "Sampling with a Learned Field": it is a different topic (how to spend a solver budget), it restates sampler equations for the third time in the chapter, its conclusion is drawn from one one-dimensional example and explained by two paragraphs of quantities no cell computes, and it ends in a table that chapter 17 also has. Without it the section is one idea of about 6,000 words.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought | 4 |
| Pedagogy | 4 |
| Language | 4 |
| Ease and enjoyment | 4 |

## What works

- **The opening.** "Could we choose the path from noise to data ourselves, and choose one along which a solver can take long steps?" Then the two-point picture in four sentences, ending "the field there can point only one way". The reader has the whole section's difficulty before any symbol appears.
- **The "One clock" callout and figure.** The conversion recipe (substitute $1 - t$, swap the names, flip the sign of every velocity) is exactly what a reader with a flow-matching paper open needs. Keep it verbatim.
- **Two points in one dimension.** The best worked example in my half of the chapter: a closed form, two values to check by hand ($-2.86$, $0.97$; both correct), three properties read off the formula, a figure that draws them, and a cell that confirms them. "One Euler step ... sends *every* sample to $0$" and "the flow chooses a mode late" are the sentences a reader keeps. The generalization in one line ($\mathbf{u}_1(\mathbf{x}) = \mathbf{x} - \mathbb{E}[\mathbf{x}_0]$ for any data) is where it belongs.
- **The CFM theorem as a corollary** of the regression lemma, with the remark that it is the proposition, not the theorem, that guarantees transport, and that "the CFM loss of a perfect field equals $C$, not zero". The floor is then computed ($1.38$ per coordinate; I reproduced $1.385$) and the trained loss ($1.44$) is read against it.
- **The reflow experiment demonstrates its claim.** Loss $1.44 \to 0.0016$, one-step energy distance $0.656 \to 0.002$, chord gap $0.245 \to 0.009$, pair cost $4.00 \to 0.26$. Every number in the prose is in the stored output, and the text is honest about what reflow cannot do ("It inherits the first model's error", "it does not make it optimal").
- **The reflow proposition's proof**: two applications of Jensen, each named, and the remark that with squared cost the last step loses exactly the floor $C$, which ties the theorem to the number the reader has just seen.
- **Kinetic energy as the bridge** from pairings to flows, with the three Gaussian flows ($1.000$, $1.858$, $1.234$) and the observation that the independent straight flow shrinks and regrows: "Straight conditional segments under the wrong pairing do not make a straight flow."
- **Exercises.** Ten, almost all good, several excellent (late decisions; reflow on two points; reflow fixed points with the quarter-turn; a stochastic sampler for a flow-matching model).

## Structural findings

1. **The section is one and a half sections.** `## Sampling with a Learned Field` (lines 1281 to 1438) leaves the argument about paths and pairings and takes up solver order and step grids on the VP probability-flow ODE of a one-dimensional mixture, which is the previous section's object, not this one's. See FMT-11 and FMT-12. Its three subsections each have a better home or none.

2. **The dictionary subsection interrupts the arc.** "Score, Noise, and Velocity" (lines 356 to 500) sits between the two-point example, which ends "The rest of the section asks how to pair noise with data so that the flow comes out straight", and "Rectified Flow", which takes that question up. The reader is diverted for 145 lines to a different question (is a noise predictor a velocity?). See FMT-4.

3. **Mathematics background or the topic itself?**
   - *Mathematics, with its home here*: the marginal-velocity proposition, the CFM theorem, the score-velocity identity and the conversion table (chapter 17 cites `tab_mdl-score-noise-velocity` for the conversions), the reflow proposition, couplings and $W_2$, Benamou-Brenier. The two-point example is cited from chapter 17 twice and is the right kind of appendix example.
   - *The topic itself*: the two-moons experiment trains a network in four frameworks and is the only training run in sections 5 and 6. It earns its place because the reflow proposition needs a measured counterpart and chapter 17 points here for it ("on two-moons data one Euler step of the reflowed model scores as well as thirty-two steps of the original"). Keep it, but note the framework question below.
   - *Belongs to chapter 17 or nowhere*: "Deterministic and Stochastic Samplers", the EDM paragraph on $\rho$, and "Comparison of Model Families" (chapter 17 has its own `tab_diffusion_models`).

4. **Framework tabs.** The house policy gives chapters 9 to 17 PyTorch and JAX only, and the appendix all four. This section is the appendix companion of a PyTorch-and-JAX chapter and spends about 330 source lines on four-tab copies of `train_rf`, the pairing samplers, `euler_path` and the reflow cell. A reader sees one tab, so the cost is to maintainers, but the prose pays too: it describes initializers and scheduler behaviour in framework-neutral words that are needed only because four implementations must agree (FMT-7).

5. **Lookup use.** Good. Each `###` restates what it needs; the callout fixes the clock at the top; the two-point subsection is self-contained.

## Passage-level findings

**FMT-1** | moderate | thought
Anchor (opening): "optimal transport. Along the way, a noise predictor trained in the previous"
Anchor (Score, Noise, and Velocity): "The previous section trained a network to predict the noise. Do we need a"
The previous section trains no noise predictor. Its only fit is a 17-feature least-squares score at one noise level, and all its samplers use exact scores; a noise-prediction network appears only in its last exercise. The same slip closes the subsection: "A trained noise predictor is therefore also a velocity field, and the samplers of this section can run on it unchanged", which no cell runs.
Fix: "a noise predictor of the kind the previous section analysed" and "Suppose a network has been trained to predict the noise. Do we need a second training run ...". Last sentence: "A noise predictor is therefore also a velocity field: the samplers of this section apply to it after the conversion of :numref:`tab_mdl-score-noise-velocity`."

**FMT-2** | minor | language
Anchor (opening): "The cost of a sample is its number of network calls, so straight paths are"
The opening runs to five paragraphs, a callout, a figure and a notation paragraph (about 520 words before the first heading; the guide's budget is 150 to 300). The fifth paragraph repeats the first ("so many steps are needed because the paths ... curve") before giving the route. The notation paragraph introduces two names for the noise end ($\boldsymbol\epsilon$, and $\mathbf{x}_1$ "for statements about couplings") without saying why two are needed.
Fix: merge paragraph five into one route sentence. In the notation paragraph, say the reason: "When both ends are arbitrary laws, as in statements about couplings, we write the pair as $(\mathbf{x}_0, \mathbf{x}_1)$."

**FMT-3** | minor | language (repetition, guide §7.7)
Anchor (The Conditional Flow Matching Theorem): "This is the situation of the regression lemma of"
The lemma is displayed again in full, with its one-sentence reading, 500 lines after its statement and proof in the previous section, where the text already announced "the next section uses it again with a velocity in place of the score". The following subsection's last sentence ("Denoising score matching has the same structure ...") tells the parallel a third time.
Fix: "By the regression lemma :eqref:`eq_mdl-regression-lemma`, least squares against a per-sample target fits its conditional mean." Keep the closing remark about DSM; it is the payoff of having one lemma.

**FMT-4** | moderate | pedagogy (order)
Anchor (end of the two-point example): "curved, and one step collapses every sample onto the mean. The rest of the section asks how to pair"
The next heading does not ask how to pair noise with data; it asks whether a noise predictor is a velocity. The score-velocity identity is a statement about Gaussian paths in general and belongs before the section narrows to straight paths and pairings, or after the argument is finished.
Fix: move "Score, Noise, and Velocity" to follow "The Conditional Flow Matching Theorem" directly, introduced as the first family of paths ("The Gaussian paths of the previous section are probability paths of this kind, and on them the velocity is not a new object"). Then "Example: Two Points" leads straight into "Rectified Flow". The identity's proof uses nothing from the two-point example, so nothing else changes. The check on the cosine path at $t = 0.4$ is sound (stored output $3.1 \times 10^{-15}$) but verifies an identity just proved in six lines; it could become an exercise.

**FMT-5** | minor | correctness (confidence medium-high)
Anchor (reflow proposition): "independent of $t$, so that this flow exists. Then"
The reflow proposition and the Benamou-Brenier theorem assume a field Lipschitz in $\mathbf{x}$ with a constant independent of $t$. The marginal field of the section's own examples is not: for the two points, $\partial_x u_t(0) = (1 - (1-t)a^2/t^2)/t$ is $-14$ at $t = 0.5$ and unbounded as $t \to 0$, and the same holds for any data with atoms, including an empirical data set. The text nevertheless applies the proposition to the two points in the next paragraph ("The two points show the effect in closed form") and reads the two-moons numbers against its bound; exercise 6 concedes the point in passing ("the velocity grows like $1/t$, so stop short of $0$"). The conclusions are true, but the stated hypotheses do not deliver them.
Fix: state the hypothesis on $[\delta, 1]$ for every $\delta > 0$ and add: "For data with atoms the field is Lipschitz only away from $t = 0$. The conclusions then hold on $[\delta, 1]$ and extend to $t = 0$ by letting $\delta \to 0$, a step we do not prove." This is the candour the guide's voice section asks for and the section already shows in the Brenier paragraph ("The data need no density, and the two points and the two moons have none").

**FMT-6** | minor | correctness (confidence high that it is a sketch presented as a proof)
Anchor (proof of the reflow proposition): "trajectory both log-densities change at the rate $-\nabla \cdot \mathbf{u}_t$. This is the"
Uniqueness for the continuity equation is dispatched in one clause. It is the step on which part (i), and through it the Benamou-Brenier lower bound, rests. The reference to `sec_mdl-probability-flow-ode` is right, but the sentence as written is an indication, not an argument (it presupposes that the law of $\mathbf{X}_t$ has a density that can be followed along trajectories, which for point-mass conditionals is what exercise 9 has to establish weakly).
Fix: say so: "... agree at every time; :numref:`sec_mdl-probability-flow-ode` gave the argument for densities, and exercise ... extends it to the weak form."

**FMT-7** | minor | language (plumbing)
Anchor (Example: Two Moons): "a $3 \to 64 \to 64 \to 2$ tanh network with Glorot-uniform weights and zero"
The two paragraphs before the training cell describe the initializer, the bias value, the centering of time, the learning-rate schedule and its purpose, the tail average and its purpose, and a factor of two between the printed loss and $\mathcal{L}_{\mathrm{RF}}$ (about 150 words). Only two of these matter to the argument: the loss is reported per coordinate, and it is averaged over the last 500 batches because single batches scatter. The rest is in the code and its comments.
Fix: "The model is a small tanh network that takes $(\mathbf{x}, t)$ and returns a velocity. `train_rf` takes a sampler of pairs, so that we can change the pairing later, and returns the network with its loss averaged over the last 500 batches and over the two coordinates, so the printed value is $\mathcal{L}_{\mathrm{RF}}/2$."

**FMT-8** | minor | language (naming)
Anchor (Example: Two Moons): "To score samples we use the"
The quantity is defined as the "squared energy distance" $\mathcal{E}^2$, plotted with the axis label "squared energy distance", printed as "energy distance" in the reflow cell, and called "the energy distance" in the prose from then on ("One step scores about $0.7$"). A reader comparing $0.656$ with a distance takes a square root that is not wanted.
Fix: one name. Since every number reported is $\mathcal{E}^2$, call it "the energy score" or keep "squared energy distance" and change the print labels' wording in prose ("scores $0.66$").

**FMT-9** | minor | thought
Anchor (after the reflow comparison cell): "and its $32$ Euler steps only approximate that flow, and the measured drop is"
The sentence joins two unrelated facts with "and", so that the approximation appears to explain why the drop ($4 \to 0.26$) is far larger than the bound guarantees ($\le 1.24$). The real reason comes in the next sentence (Jensen for the time average is loose on curved trajectories).
Fix: "The measured drop is far larger than the bound requires, because the first step of the proof, Jensen's inequality for the time average, is far from tight on the curved trajectories of the first flow. (The bound is stated for the exact marginal flow, which the trained network and its 32 Euler steps only approximate.)"

**FMT-10** | moderate | pedagogy
Anchor (Couplings and the Wasserstein Distance): "In two dimensions, freedom from crossings is not enough. Rotate a standard"
The sentence before this subsection promised a flow that is "crossing-free and still move[s] mass farther than it must". The body's example is a rotating flow whose particles travel on arcs. It shows that a flow can waste motion "by turning", which is a point about curved paths, not about pairings; a reader may conclude that straight and crossing-free is enough. The decisive example is the straight interpolation from $\mathbf{x}_0$ to its quarter turn $\mathbf{R}\mathbf{x}_0$: straight, constant speed, crossing-free, a fixed point of reflow, cost $4$ against $W_2^2 = 0$. It appears in one sentence after the Benamou-Brenier proof and is worked only in exercise 9, yet the Further Reading notebook cites this section for it.
Fix (writing pass): put the quarter-turn pairing in the body in place of, or after, the rotating flow: "Pair each $\mathbf{x}_0$ with $\mathbf{R}\mathbf{x}_0$. The segments $((1-t)\mathbf{I} + t\mathbf{R})\mathbf{x}_0$ never cross, because that matrix is invertible for every $t$; each is traversed at constant speed; and the cost is $\mathbb{E}\|\mathbf{R}\mathbf{x}_0 - \mathbf{x}_0\|^2 = 4$, although the two clouds are the same and $W_2^2 = 0$. Reflow leaves this pairing unchanged." Deferred (figure): panel (c) of `mdl-dyn-couplings-energy` then shows chords instead of arcs.

**FMT-11** | major | thought, scope
Anchor (Sampling with a Learned Field): "A trained network is a score $\mathbf{s}_{\boldsymbol{\theta}}$ or a velocity"
"Deterministic and Stochastic Samplers" displays the probability-flow ODE and the reverse SDE again. The same pair is derived in the Fokker-Planck section and displayed as :eqref:`eq_mdl-score-samplers` in the previous one; this is the third display, with the third explanation of why the SDE "doubles the score term". The second paragraph (the trace integral gives the model's own log-density; the SDE's step count "is tuned separately") states two facts without using either.
"Comparison of Model Families" is a five-row table of DDPM, score SDE, probability-flow ODE, DDIM and flow matching. Chapter 17 has the same kind of table (`tab_diffusion_models`), and a table of named models is the topic, not its mathematics.
Fix: cut "Deterministic and Stochastic Samplers" to one sentence at the head of the solver experiment, with the two labels. Cut the table here and let chapter 17's stand; keep its one-sentence reading ("Every row regresses onto a conditional mean of a closed-form per-sample quantity") in the Summary, where it is the chapter's thesis.

**FMT-12** | moderate | thought (claims no cell shows)
Anchor (Solver Order at Equal Cost): "The shape of the paths explains this. Until about $t = 0.5$ the noised law"
Anchor: "cancel to about an eighth of their total size at every budget. Heun's"
The table of errors is clear and the prose reads it correctly (I recomputed the ratios $2.8$, $3.3$, $3.6$ and the crossover at 40 evaluations). The two paragraphs that follow explain the result through about ten quantitative statements that nothing in the section computes: the paths "cover less than a tenth of their length" by $t = 0.5$; they are "fastest near $t = 0.25$"; Heun overshoots "above about $t = 0.4$" and falls short "between about $t = 0.1$ and $0.4$"; Euler's signed errors "cancel to about an eighth of their total size"; the squared grid changes the total "by about a tenth for Euler"; the slope "reaches about $-33$ at $t = 0.1$" (this one I checked: $-32.7$). The reader must take 27 lines of mechanism on trust. This is the pattern the Basics review put first: a run, then an explanation of what the run did not show.
The conclusion, "Where the steps go matters more than the order of the solver", repeated in the Summary, rests on one one-dimensional mixture under one schedule, and the section's next paragraph says "the best grid depends on the schedule and on the data".
The experiment is also on the wrong object for this section: the VP probability-flow ODE with an exact score, not a flow-matching field. Its natural place is after "DDIM is Euler's method" in the previous section, whose exercise 6 already asks for Heun against DDIM on the same mixture, or in chapter 17's `samplers.md`, which quotes this result in a sentence and then runs its own.
Fix (writing pass): keep the cell and the paragraph that reads the table. Replace the two mechanism paragraphs by three sentences that the table and one stated fact support: "Until about $t = 0.5$ the noised law is nearly $\mathcal{N}(0,1)$ and the field nearly zero, so equal steps in $t$ spend half the budget where nothing moves. The endpoint error is a sum of signed step errors, and the grid changes how they cancel as much as how large they are: on equal steps Euler's cancel and Heun's do not, and on the crowded grid the reverse holds. Exercise ... measures the signed errors." Limit the conclusion to the example: "On this field, the placement of the steps matters more than the order." Then decide the home (structural finding 1). Deferred: an exercise or a short cell that prints the signed per-step contributions, if the mechanism is to stay.

**FMT-13** | minor | thought
Anchor (Approximate Optimal Couplings): "fix the number of solver steps, which also depends on the solver and on how"
The subsection opens with two sentences about curvature and solver steps that qualify a claim nobody has made at this point, and then reports a result (minibatch pairing lowers the plateau "from $1.4$ to about $0.06$" and the one-step score "to a few thousandths") that only an exercise produces. I ran the exercise: plateau $0.056$, one-step score $0.004$. The numbers are right, but the body should say that they come from the exercise and not from a cell above.
Fix: delete the first two sentences. "Exercise ... trains with this pairing: the loss plateau falls from $1.4$ to about $0.06$ per coordinate and one Euler step already scores a few thousandths."

**FMT-14** | minor | correctness (confidence medium; statement, not proof)
Anchor (Benamou-Brenier theorem): "has kinetic energy at least $W_2^2(p_0, p_1)$. If the straight interpolation"
The theorem is stated honestly but in a shape that is hard to read: a definition of admissibility, an unconditional lower bound, a conditional attainment ("If the straight interpolation of an optimal coupling has an admissible marginal velocity (for example, ...)"), and then "so" the equality with a $\min$, which typographically looks unconditional. Under the stated definition of admissibility the minimum need not be attained (it is not for the two points or the moons), and the classical theorem takes an infimum over weak solutions.
Fix: state two parts. "(i) Every admissible pair has kinetic energy at least $W_2^2(p_0, p_1)$. (ii) If the marginal velocity of the straight interpolation of an optimal coupling is admissible, it attains $W_2^2$, and then [display with min]. In general the equality holds with an infimum over a wider class of flows, which we do not prove." The proof needs no change.

**FMT-15** | minor | language (guide §12.2)
Anchor (end of body): "We can now learn a score or a velocity from samples by one regression and"
Same closing formula as the previous section and every other section of the chapter. The paragraph's second half is good and candid ("No known method finds the optimal pairing cheaply in high dimension, and the number of steps a learned field needs is still found by trial"); that is the open question the guide asks a section to end on.
Fix: delete the first two sentences and start at "What remains open is the pairing of noise with data."

**FMT-16** | minor | explanation
Anchor (Straight Paths and Their Crossings): "density, so the proposition does not apply as stated. Its conclusion still"
The straight path, the object of the rest of the section, is exactly the case the marginal-velocity proposition does not cover, and the repair is delegated to an exercise in a parenthesis. The two-point example avoided the issue by conditioning on $x_0$ alone. A reader may wonder why $\mathbf{z} = (\mathbf{x}_0, \boldsymbol\epsilon)$ is used at all.
Fix: one sentence of orientation: "With Gaussian noise we could condition on $\mathbf{x}_0$ alone, as in the two-point example, and the proposition would apply. Conditioning on the pair is what lets us change the pairing later, so we take the pair and pay for it with the weak form."

## Checks made

Recomputed, all in agreement with the text: the two-point velocity, its values at $(0.5, 0.5)$ and $(0.7, -1)$, the slopes $0.56$, $-0.31$, $-14$ and $t^* = 0.83$; the Gaussian cosine-path example ($s_t^2 = 2.5$, rate $-0.94$); the score-velocity identity, its VP and straight-path forms, and all five rows of the table including the $2/\pi$ in the caption; the two-point reflow cost $1.81$ and floor $C \approx 2.03$ (Monte Carlo: $2.04$); the two-moons floor $1.38$ (reproduced: $1.385$); the bound $4 - 2.76 = 1.24$; $W_2^2$ for one-dimensional Gaussians; the rotation energies $\pi^2/2$ and $4$; the three kinetic energies and the minimum of $s_t$ at $t = 0.8$; the value $0.43$ ($= 2 - \pi/2$) in exercise 5; the probability-flow slope $-33$ at $t = 0.1$; the stochastic-sampler SDE of exercise 10 against the Fokker-Planck equation; the EDM facts ($\rho = 3$ equalizes local error, $\rho = 7$ used, 35 evaluations at $32 \times 32$); the minibatch-OT numbers quoted in the body (by running exercise 7 in PyTorch). All stored PyTorch outputs agree with the prose. Figure `mdl-dyn-two-point-flow` was rendered and matches its caption.

Not checked: the TensorFlow, JAX and MXNet outputs of the training cells; the remaining figures beyond a look at their captions (`mdl-dyn-fm-paths`, `mdl-dyn-couplings-energy`, `mdl-dyn-time-conventions` were rendered but not studied in detail); the claims in the "Solver Order" mechanism paragraphs other than the slope.

## Exercises

Ten problems, inside the band. This is the strongest exercise set I read in the chapter: nearly every problem extends the text and none is a drill.

- **Keep as they are**: 1 (the floor; part 3 is the computation the body cites), 3 (late decisions; the limits $a \to \infty$ and $a \to 0$ are a real question), 4, 5, 6, 9, 10.
- **2 (stochastic interpolants)**: good, but its label carries the previous section's prefix (`ex_mdl-score-matching-diffusion-flow-stochastic-interpolants`), a leftover of the split. Harmless to the reader; worth noting for anyone grepping.
- **7 (minibatch pairing)**: its result is quoted in the body (FMT-13). Either promote it to a cell, since it is the second of the two re-pairing methods the section names and costs one call to `train_rf`, or keep it and have the body say that the numbers are the exercise's.
- **8 (conditioning of prediction targets)** is the only place where the question "which row is easiest to learn" is answered. Fine as an exercise; chapter 17's "Parameterization at Finite Capacity" covers the practice.
- **Add** (if the solver experiment stays): a part that prints the signed per-step error contributions for Euler and Heun on both grids, so that the mechanism of FMT-12 is something the reader measures.

## Still open from October

This section did not exist in October; it is the second half of the section that review asked to split. The requested cuts (EDM, log-SNR, $v$-prediction, stochastic interpolants, interacting particles) were mostly made: stochastic interpolants became an exercise and a Further Reading entry, $v$-prediction a table row. EDM survives as a paragraph in "Solver Order at Equal Cost". "Coordinate it with ch. 17" is still open (FMT-11, FMT-12).

## For the chapter review

- **Fit.** The section answers the question the previous one ends on and closes the chapter's arc: integrate a field, train through a solver, add noise, follow the density, learn the score, choose the path. It assumes the continuity equation and probability-flow ODE (Fokker-Planck section), the regression lemma and Tweedie (previous section), Lipschitz uniqueness and Euler/Heun (ODE section), continuous normalizing flows (neural-ODE section), and MMD and couplings from chapter 42. All by label, all resolving.
- **Appendix or main text.** Up to the end of "Optimal Transport and Straightness" this is mathematics that chapter 17 uses and cites: keep it here. The two-moons run is the one training experiment worth keeping in the appendix, because the reflow proposition needs a measurement. The last `##` part is topic material: fold the solver experiment into the previous section next to DDIM or leave it to `samplers.md`, and drop the sampler restatement and the model-families table.
- **Overlap inside the chapter.** The reverse SDE and probability-flow ODE are displayed in three sections. Euler against Heun at equal cost is run here and set as exercise 6 of the previous section, while Heun's method and its order are taught in the ODE section (I did not check whether that section also runs an equal-cost comparison). One home each.
- **Overlap with chapter 17.** `few-step.md` ("Straighter Paths Help but Do Not Finish") retells crossings, reflow and the two-moons result in three paragraphs with pointers here, which is the right amount. `gaussian-paths.md` re-derives the marginal velocity, the CFM loss identity and the score-velocity identity at about the same level as this section (statement, one-line reason, display), so on those three the main text is not lighter than the appendix; only the proofs differ.
- **Pattern.** As in the previous section, the body closes with "We can now ...". And as there, a sentence speaks of a network "trained in the previous section" that does not exist; the chapter's experiments run on exact fields except for two moons, and the prose should say so once, in the chapter introduction.
