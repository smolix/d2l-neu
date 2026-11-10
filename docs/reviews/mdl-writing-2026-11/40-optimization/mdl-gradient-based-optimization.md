# Gradient Descent and Curvature

Source: `chapter_mdl-optimization/mdl-gradient-based-optimization.md`, about 9,300 words of prose before the exercises and 2,250 words of exercises (code excluded; 15 PyTorch cells besides the imports).

## Verdict

The first two thirds of this section (smoothness, the quadratic model, momentum) are the best mathematical writing I have read in this book and would not look out of place in a good optimization text. The section opens on three numbers the reader can check by hand ("With $\eta = 0.01$ the value drops only to $6.01$. With $\eta = 0.1$ it drops to $1.62$. With $\eta = 0.25$ it *rises* to $12.4$"), promises three counts (69, 6,908, 315), derives each, and then measures each. The proofs are short and are the short ones: the optimal step by a weighted average in which "The terms in $\eta$ cancel", the $\sqrt\kappa$ law from the single remark that the product of the two roots is $\beta$. Every number I recomputed is right, and the stored outputs agree with the prose. Against the bar, three things hold it back. It is two sections in one: after Nesterov the subject changes from "how far can a gradient step go" to curvature models (Newton, secant, BFGS, trust regions, 2,800 words), whose use in deep learning the text itself concedes is small. The edge-of-stability subsection repeats an experiment that chapter 9 already runs on the same network, and explains it less carefully than chapter 9 does. And one worked claim about the quartic is wrong as stated (GD-4). The most important fix is structural: split or prune the second-order half, and cut the duplicated experiment.

I recomputed every worked number (the opening values, the bounds 6.012, 1.8, 13.5, the sweep table, the counts 69, 691, 6,908, 27, 315, 1,056, the heavy-ball predictions 8.6, 9.2, 9.8, the Nesterov rates, the secant errors, the trust-region step) and checked each proof line by line. Apart from the findings marked `correctness` below, all are correct.

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|---|---|---|---|---|
| 4 | 4 | 4 | 4 | 4 |

The first three parts alone would score 5, 4, 5, 4, 4. The last two parts (Newton to trust regions) are a 3 on pedagogy and enjoyment: correct, compact, and weakly motivated for this reader.

## What works

- **The opening.** A two-variable bowl, one gradient, three step sizes, three values, and the sentence "The direction was downhill every time; only the distance differed." The reader has the whole problem of the section in 120 words and can verify it with a pencil.
- **Predict, derive, measure.** The opening states that the best fixed step needs 69 iterations for six digits at $\kappa = 10$ and 6,908 at $\kappa = 1000$, and that momentum brings the latter to 315. "The Optimal Step" derives 69 and 6,908; the momentum cell prints 69, 6908, 315. This is the method the Basics review asked for, done properly. Protect it.
- **The descent lemma is explained before it is proved.** "The first-order decrease $\eta\,\|\mathbf{g}\|^2$ grows linearly in $\eta$, while the curvature error $\tfrac{L}{2}\eta^2\|\mathbf{g}\|^2$ grows quadratically. For short steps the decrease dominates, and at $\eta = 2/L$ the two terms are equal." After this the proof is a formality. The follow-up, with the two parabolas $7 - 104\,\eta + 502\,\eta^2$ and $7 - 104\,\eta + 520\,\eta^2$ and the figure, shows exactly how loose the bound is.
- **The proof of the optimal step.** "a maximum of two numbers is at least any weighted average of them", with weights chosen so that $\eta$ drops out. Four lines, no case analysis, and :numref:`fig_mdl-opt-eta-tent` is the proof drawn. This is the house style at its best.
- **The heavy-ball proposition.** One fact ("The product of the two roots is $\beta$") gives the modulus $\sqrt\beta$ "whatever $\lambda$ is". The text is then honest about the factor $k$ at the double roots, predicts its size ($8.6$, $9.2$, $9.8$) and measures it ($8.5$, $9.3$, $10.0$). A lesser text would have stopped at the asymptotic law and left the reader puzzled by the table.
- **One saddle, three visits.** $f(x, y) = \tfrac12 x^2 + \cos y$ stops gradient descent at $(0, 0)$, attracts Newton's method to the same point from $(0.5, 0.3)$ (with the pleasing aside "here even cubic, since $y - \tan y \approx -y^3/3$"), and is escaped by the trust region, whose first step is computed by hand and then reproduced by the first row of the output. One example reused, each time explaining more.
- **Candour.** "Momentum in deep learning ... rests on the local quadratic model and on experience, not on a global theorem." "ReLU networks are not smooth, so for them it describes an idealized model rather than a guarantee." "Neither replaces stochastic gradients for large noisy losses".
- **Units.** "provided the comparison is made in the same units", followed by the count of where the factor four comes from. Few texts bother.
- **The exercises** (see below) are among the best in the book.

## Structural findings

- **Two sections in one.** Parts 1 to 3 (smoothness, quadratics, momentum; about 6,300 words) answer the question the opening poses. Parts 4 and 5 (Newton, secant, BFGS, trust regions; about 2,800 words, 7 cells, 2 exercises) answer a different one: what can be done with curvature. The seam shows in the title ("Gradient Descent and Curvature"), in the opening, which mentions Newton in half a sentence, and in the summary, which has to cover both. Recommended: end this section after "Nesterov's Look-Ahead" and make "Curvature Models: Newton, Quasi-Newton, Trust Regions" its own short section. The labels `subsec_mdl-why-not-newton`, `subsec_mdl-quasi-newton` and `subsec_mdl-trust-region` are cited from chapter 9 (`gd.md`, `further-reading.md`), so the material should stay in the chapter. If a split is unwelcome, prune instead: keep Newton with its quadratic-convergence proof and the saddle failure, reduce BFGS to the secant equation, the two-step run on $\mathrm{diag}(1, 10)$ and the L-BFGS paragraph (GD-16), and keep the trust region as it is, since it completes the saddle thread.
- **The edge-of-stability experiment is run twice in the book.** See GD-9. Chapter 9 owns it.
- **Four quadratics.** The opening bowl ($\kappa = 10$), "the chapter's running example" ($\kappa = 1000$, quoted three times, run once as a table row), the valley of :numref:`fig_mdl-opt-gd-bowl-vs-valley` ($\kappa = 20$ and $1.3$, shown only in the panel titles), and $\mathrm{diag}(1, 100)$ in the Newton cell. Chapter 9 adds a fifth, $0.1x_1^2 + 2x_2^2$. Each choice is locally reasonable, and together they blur the thread. Use $\kappa = 10$ for hand numbers and $\kappa = 1000$ wherever a count is quoted; redraw the valley figure at one of them and run Newton on $\mathrm{diag}(1, 1000)$ (it still takes one step, against 6,908).
- **Rates are stated before convexity exists.** "From Quadratics to Convex Functions" defines strong convexity in a clause and tabulates three rates, two of them proved in the next section. See GD-10.
- **Overlap with the main text.** Chapter 1 and `linear-regression.md` introduce gradient descent and the noise ball informally; this section is consistently deeper and does not re-teach them. Against chapter 9 the division (chapter 9 measures, the appendix proves) holds for the stability ceiling, the $\kappa$ law and the heavy-ball tuning: `momentum.md` states :eqref:`eq_momentum-optimal` and points here for the proof, which is the right arrangement. It fails for the edge of stability (same experiment) and partly for Newton's method: `gd.md` derives the Newton step from the Taylor expansion, runs it, and states the $O(d^2)$ and $O(d^3)$ costs ("For $d = 10^9$ the Hessian has $10^{18}$ entries"); this section derives the step again and states the cost again ("a model with $10^8$ parameters would need $10^{16}$ numbers"). The appendix adds the convergence proof, affine invariance and the saddle; it could cite the step and the cost from chapter 9 in one clause each.
- **Stand-alone use.** A reader who lands on `subsec_mdl-quadratic-model`, `subsec_mdl-momentum-acceleration` or `subsec_mdl-why-not-newton` from chapter 9 can read each without the parts before it: the propositions restate their recurrences and symbols. This works well.

## Passage-level findings

**GD-1** (minor, pedagogy). Opening, third paragraph.
Quote: "single step if we can afford its Hessian. Most worked numbers below use the"
The reader is told about a running example and, in the next sentence, that it will mostly not be used. Say it the other way round: "We work by hand on the opening bowl, whose Hessian $\mathrm{diag}(1, 10)$ keeps every step checkable, and quote counts for the chapter's stiffer example $\tfrac12(x_1^2 + 1000\,x_2^2)$."

**GD-2** (minor, explanation). Under "Smoothness and the Descent Lemma".
Quote: "logistic loss of the Newton subsection below,"
The paragraph that defines $L$ gives as its second example a formula, $L = \lambda_{\max}(\mathbf{X}^\top\mathbf{X})/(4n) + \nu$, for a loss that appears 900 lines later, with $\mathbf{X}$, $n$ and $\nu$ undefined. A first-time reader cannot use it. Move the formula to the Newton subsection, where the text needs it ("which is safe since $L$ is about $0.4$ here") and where exercise 2 derives it. Here the quadratic and the quartic suffice.

**GD-3** (minor, language). Under "Newton's Method".
Quote: "\qquad \nu = 0.1,"
The ridge coefficient is $\nu$ here, $\lambda$ in the next section ("Adding the ridge penalty $\tfrac{\lambda}{2}\|\mathbf{w}\|^2$") and in the chapter's notation table, where $\nu$ is reserved for equality multipliers. Presumably $\nu$ was chosen to avoid the eigenvalues $\lambda_i$. Use $\lambda$ with a clause ("the weight-decay coefficient $\lambda$, not an eigenvalue"), or change the table; the two sections must agree.

**GD-4** (moderate, correctness; sure, verified numerically). Under "Backtracking Line Search".
Quote: "ceiling is $2/27 \approx 0.074$; six steps later the iterate sits at"
The text explains the divergence of the fixed step $\eta = 0.3$ on $\tfrac14x^4$ from $x_0 = 3$ by "the local stability ceiling is $2/27 \approx 0.074$". That is not the threshold. The update is $x \mapsto x(1 - \eta x^2)$, which contracts whenever $\eta x^2 < 2$, so from $x_0 = 3$ every fixed step below $2/9 \approx 0.222$ converges; $\eta = 0.1$ and $\eta = 0.2$, both above $0.074$, converge (I ran them). The section's own exercise 4(b) asks the reader to prove exactly this ("conclude that every $\eta < 2/9$ converges"). The rule $2/f''$ belongs to quadratics; on the quartic the curvature that matters for a step toward the origin is the average of $f''$ between $0$ and $x$, which is $x^2 = 9$, not the endpoint value $27$. A reader who trusts the sentence will be confused by the exercise. Fix: "At $x_0 = 3$ the descent lemma certifies only steps below $2/27 \approx 0.074$, and the true threshold, $2/x_0^2 = 2/9$, is not much larger (exercise ...); $\eta = 0.3$ is above both, and six steps later ...". The later sentence "The accepted step roughly tracks $2(1-c)/f''(x)$" should then say what it tracks more carefully: with $c = \tfrac12$ the Armijo threshold on this function is about $1.37/f''(x)$ ($0.0507$ at $x_0$), and halving lands within a factor two below it.

**GD-5** (minor, thought). Under "Backtracking Line Search", last sentences.
Quote: "rarely backtracks, because each probe needs another forward pass. Learning-rate"
Two loose claims close a careful subsection. The larger obstacle to a line search in training is that a minibatch loss is noisy, so comparing two values does not test the Armijo condition for the loss being minimized; cost is secondary. And a schedule does not substitute for a line search: it is fixed in advance and measures nothing. Fix: "Training rarely backtracks: a minibatch loss is too noisy to compare two steps, and each probe costs a forward pass. Practice fixes the step in advance by a schedule (:numref:`sec_scheduler`) and guards it by clipping."

**GD-6** (minor, explanation). Under "Ill-Conditioned Valleys".
Quote: "valley; a step near the stability ceiling makes the steep mode overshoot and alternate (factor $-0.94$)"
The figure's panels are titled $\kappa = 1.3$ and $\kappa = 20$, a pair of quadratics that appears nowhere else, and the caption's factors $-0.94$ and $0.90$ cannot be reproduced from anything in the text. Either name the matrix in the caption or redraw the valley panel on $\mathrm{diag}(1, 10)$ at $\eta = 0.19$, whose factors $-0.9$ and $0.81$ the sweep cell then prints.

**GD-7** (minor, correctness; sure). Under "The Optimal Step".
Quote: "Raising $\eta$ lowers the left endpoint's value and raises the right one's."
True only for $1/\lambda_{\max} \le \eta \le 1/\lambda_{\min}$; below that range both values fall, above it both rise. Add "Once $\eta$ exceeds $1/\lambda_{\max}$, raising it further lowers ...". The proof that follows does not depend on the sentence.

**GD-8** (moderate, correctness; fairly sure). Under "The Edge of Stability".
Quote: "The per-mode recursion explains why the sharpness stops at the ceiling. In"
The recursion explains why the iterate oscillates and grows along the top eigendirection once $\lambda_{\max} > 2/\eta$. It does not explain why that motion lowers the sharpness; the paragraph asserts it ("the motion lowers $\lambda_{\max}$ back below $2/\eta$") as if it followed. Chapter 9 is careful on exactly this point: "The quadratic model predicts the oscillation but not what follows. The third-order term of the Taylor expansion, which the quadratic model drops, turns the oscillation into a drift toward lower sharpness" (`optimization-intro.md`, citing Damian et al. 2023). The appendix, which is supposed to be the more rigorous of the two, claims more and cites less. If the subsection stays, replace the sentence by chapter 9's statement and its citation.

**GD-9** (major, pedagogy). Under "The Edge of Stability", both cells.
Quote: "The cell trains a two-layer $\tanh$ network with $25$ parameters on $16$"
This is the experiment of `chapter_optimization/optimization-intro.md` ("one input, eight $\tanh$ hidden units, and one output, 25 parameters in all, and it fits 16 points of $\sin(3x)$ ... at $\eta = 0.4$ and at $\eta = 0.25$"), with hand-written backprop in place of `torch.func`. The same network, data, step sizes and conclusion appear in both chapters; the October review already listed this subsection under "Cut". It also interrupts the argument: the reader is between the optimal step and the rates, on quadratics, and is taken to a network for 500 words and 45 lines of code whose output the prose must narrate with seven hedges ("usually already near its ceiling $5$, while the $\eta = 0.25$ run is usually still climbing") because the cell is unseeded. Fix: replace the subsection by one paragraph. "On a network the Hessian changes along the run, and the ceiling is then reached from below: the sharpness rises until it meets $2/\eta$ and stays there (:numref:`sec_optimization-intro` runs the experiment). The per-mode recursion says what happens at the ceiling, an oscillation along the top eigendirection, and exercise ... extends the ceiling to momentum." Keep the exercise on momentum, pointed at chapter 9's cell. This removes two cells and about 450 words.

**GD-10** (moderate, thought). Under "From Quadratics to Convex Functions".
Quote: "The quadratic does better than this bound for the whole class, provided the"
Two problems. First, the subsection states rates for classes of functions that are defined only in the next section, and its strong-convexity definition is a different one from the definition given there; the three-line table is repeated in the summary of `mdl-convexity.md`. Second, it attributes the factor four to the quadratic ("a factor two from the larger step ... and a factor two from the square"), while the next section says the square is slack in the proof: "gradient descent with $\eta = 1/L$ contracts $f - f^\star$ by $(1 - \mu/L)^2$ per step on every $L$-smooth $\mu$-strongly convex function". Read together, the two passages disagree about whether quadratics are easier than the class. Fix: shrink the subsection to a bridge of three sentences (what survives beyond quadratics is the linear dependence on $\kappa$; the proofs need convexity and are in :numref:`subsec_mdl-convergence-rates`), move the "same units" comparison to the rate-check passage of the convexity section, where the measured $0.81$ makes it concrete, and say there once which factor is the proof's and which is the step's.

**GD-11** (moderate, pedagogy). Under "Heavy-Ball Momentum".
Quote: "$\beta = (1 - \sqrt{2/11})^2 \approx 0.33$ the slow mode is critically damped"
This paragraph reads :numref:`fig_mdl-opt-momentum-damping` with seven numbers (0.05, 0.81, $-0.77$, 0.33, 0.57, 0.85, 0.92, then 0.52) and the terms "critically damped", "negative real root" and "rate", all before the characteristic polynomial that defines them. It is the one place in the first three parts where the reader has to take the numbers on trust. Fix: here describe the three regimes in words only (too little momentum zigzags, the right amount goes straight, too much loops), and move the numbers into a short paragraph after the proposition of the next subsection, where each follows from $(1 - \sqrt\beta)^2 \le \eta\lambda \le (1 + \sqrt\beta)^2$ in one line.

**GD-12** (minor, correctness; fairly sure). Under "Nesterov's Look-Ahead".
Quote: "with the same constant $\beta$ converges at the rate $1 - 1/\sqrt\kappa$ on"
The proposition just above measures the distance $|u_k|$; Nesterov's theorem for the class bounds the gap $f - f^\star$ by a constant times $(1 - 1/\sqrt\kappa)^k$, which for the distance is the factor $\sqrt{1 - 1/\sqrt\kappa} \approx 1 - 1/(2\sqrt\kappa)$. Using "the rate $1 - 1/\sqrt\kappa$" for both silently changes units, a few hundred lines after the text insisted on comparing "in the same units". Add "for $f - f^\star$" and a clause that this is half the exponent the quadratic achieves.

**GD-13** (minor, correctness; sure). Same paragraph.
Quote: "$O(1/k^2)$ or the $\sqrt\kappa$ dependence in the worst case"
The lower bound holds while the iteration count is below about half the dimension; in fixed dimension conjugate gradients finish in $d$ steps. Exercise 9 says so ("the bound governs about the first $d/2$ iterations"), the text does not. Add "in dimension large compared with the number of steps".

**GD-14** (minor, correctness; sure). Under "Newton's Method".
Quote: "contrast, change under every $\mathbf{T}$ that is not a multiple of an"
With the step $\eta$ held fixed, gradient descent on $f(\mathbf{T}\mathbf{y})$ maps back to $\mathbf{x} - \eta\,\mathbf{T}\mathbf{T}^\top\nabla f$, so the iterates change unless $\mathbf{T}\mathbf{T}^\top = \mathbf{I}$; a multiple $c\,\mathbf{Q}$ rescales the step by $c^2$. Exercise 5(b) derives exactly this $\mathbf{P}$. Write "change under every $\mathbf{T}$ that is not orthogonal (a multiple of an orthogonal matrix changes only the effective step)".

**GD-15** (minor, explanation; pattern, 4 instances). References by description instead of by label.
Quotes: "logistic loss of the Newton subsection below,"; "diagonal or structured preconditioners, which the adaptive-methods section"; "stochastic and adaptive sections take up."; "the last part of this section, repairs this."
In a part that is read by lookup these should be `:numref:` (`subsec_mdl-why-not-newton`, `sec_mdl-adaptive-stochastic-methods`, `sec_mdl-stochastic-gradient-methods`, `subsec_mdl-trust-region`). The same holds for "a Lagrange multiplier for the constraint" under "The Subproblem" (GD-18).

**GD-16** (moderate, pedagogy). Under "BFGS".
Quote: "$\mathbf{M}_k$ in a weighted Frobenius norm, but we do not use that property."
The update :eqref:`eq_mdl-opt-bfgs` arrives with "**BFGS** ... chooses the update" and no reason for its form. The text then verifies two properties and names, in order to dismiss it, the one property that would explain the formula. For a house style that puts intuition before proof this is the weakest moment of the section, and it sits in the part whose relevance the closing concedes ("less natural when fresh minibatch noise makes gradient differences unreliable"). Two acceptable fixes. Cut: keep the secant pair, the secant equation, one sentence that BFGS is the standard rank-two update that satisfies it and stays positive definite, the two-step cell, and the L-BFGS paragraph; drop the displayed update, the two bullet proofs, the Wolfe paragraph and the six-dimensional cell (about 450 words and one cell), and let exercise 11 carry the rest. Or motivate, in two sentences before the formula: "The factor $\mathbf{I} - \gamma\mathbf{y}\mathbf{s}^\top$ removes from a vector its component along the new pair, so the sandwich keeps what $\mathbf{M}_k$ knew about every other direction; the term $\gamma\mathbf{s}\mathbf{s}^\top$ then writes in the new pair."

**GD-17** (minor, pedagogy). Under "BFGS", second cell.
Quote: "legend=['gradient descent', 'BFGS']"
The cell plots six points per method and prints two lines; the prose reads only the printed lines. The plot adds nothing the print does not say. Drop the `d2l.plot` call (deferred with other cell changes), or drop the cell with GD-16.

**GD-18** (minor, explanation). Under "The Subproblem".
Quote: "shift $\tau$ is a Lagrange multiplier for the constraint"
A forward reference to a concept two sections away, with no label, as the last sentence of a dense paragraph. Either add (:numref:`sec_mdl-constrained-optimization-duality`) or move the remark to that section, which can cite the trust region as an example. In the same paragraph the proposition is stated as "if and only if" while only sufficiency is proved (in an exercise); the text says so honestly, which is fine, but the word "simple" in "The solution has a simple form" promises more than a three-condition system with a "hard case".

**GD-19** (moderate, language). Last paragraph of the body and the Summary.
Quote: "You can now predict from $L$ and $\mu$ how many steps gradient descent needs,"
The body ends with a recap paragraph in the "You can now ..." formula that the guide excludes (§12.2), the Summary recaps again, and both end on the same sentence: "asks when that point is the global minimum, and ... what noisy gradients cost" (body) and "Two questions remain open: when a stationary point is the global minimum, and what noisy gradients cost" (Summary), twenty lines apart. The next section's body ends the same way ("We can now certify ..."). Fix: delete the "You can now" paragraph; let the body end on the trust-region paragraph's last real result, and keep the two open questions once, in the Summary.

**GD-20** (minor, language; pattern, 10 instances). Lead-ins to code.
Quotes: "The cell compares the first-order slope with the actual"; "The cell sweeps $\eta$ across the ceiling on $\mathbf{A} = \mathrm{diag}(1, 10)$,"; "The cell counts the steps each method needs for a"; "The cell implements the step by bisection on $\tau$ and runs six iterations of"
Every cell is introduced by a sentence that begins "The cell" and lists what is printed. It is clear, and by the tenth time it is a drone. The best lead-ins in the section instead pose what is to be found ("How much curvature do the secant pairs recover?"). Vary: state the prediction the output will be compared with, and let the code say what it prints.

## Exercises

Twelve problems, at the top of the 8 to 12 band, graded from one-line proofs to two starred projects. This is a strong set: nearly every problem extends the text instead of repeating it, and several connect to the main text by label (Muon as steepest descent in the spectral norm, the Lessard, Recht and Packard cycle, the Nemirovski and Yudin lower bound with its dimension caveat, the edge of stability with momentum). Exercise 3 ("A saddle that attracts") turns a remark into a measurement and a prediction ($y_{k+1} \approx 2y_k$) and is a model problem.

- Exercise 4(b) contradicts the text until GD-4 is fixed. Keep the exercise; fix the text.
- Exercise 12(c) repeats the hand computation of "The Trust-Region Step at a Saddle" with $-1$ in place of $-0.955$ and gives the answer in the statement. It is the one drill in the set; parts (b) and (d) are the valuable ones. Drop (c) or fold it into (d).
- Exercise 10 (edge of stability with momentum) should point to chapter 9's cell if GD-9 is adopted.
- Exercise 2(c) asks the reader to explain why $L \approx 0.4$ "for every draw"; good use of an unseeded cell.

## Still open from October

The October review asked to cut "BFGS and trust regions, edge of stability"; all three are still here (GD-9, GD-16, and the structural finding). Its other points for this section are resolved: the opening now poses a problem instead of listing questions, and the three definitions of $\kappa$ are reconciled in a sentence ("For a symmetric positive definite $\mathbf{A}$ it equals the singular-value ratio").

## For the chapter review

- This section is the foundation of the chapter and is cited by every later section, by two other appendix chapters (linear algebra, dynamics) and by chapter 9; its first three parts are the model for the part (predict a number, derive it, measure it; short proofs with the figure drawn from the proof).
- It hands the next section two things: the descent lemma, used as the first line of both rate proofs, and a forward-stated table of rates (GD-10) that the next section then proves. The handover is duplicated, and the two sections tell the "factor of the square" differently. The count 6,908 of this section (six digits of distance at $\eta^\star$) and the count 6,900 of the next (three digits of gap by the bound at $\eta = 1/L$) are different quantities that look like one number (CV-12).
- Suspected chapter-wide patterns: body closings in the "You can now / We can now" formula followed by a summary that repeats them; lead-ins "The cell ..."; unseeded cells narrated with "usually"; references to other sections by description instead of label; the ridge coefficient written $\nu$ here and $\lambda$ elsewhere.
- Length: 9,300 words is above anything a reader takes in one sitting. Splitting off the curvature models leaves about 6,300, and cutting the edge-of-stability subsection about 5,900.
