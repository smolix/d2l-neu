# Gradient Descent

Source: `chapter_optimization/gd.md` (lines 1 to 558, up to the slides marker). About 2,300 words of body prose plus 800 words of exercises; 14 code cells, no pre-generated figures.

## Verdict

Competent and correct, but ordinary, and in the wrong place relative to its neighbour. The previous section has already run gradient descent on the valley, derived the stability ceiling, the best learning rate and the dependence on the condition number. This section then announces the same question, restarts from $f(x) = x^2$, and spends its first third on three runs whose outcome the reader can compute by multiplying 0.6 by itself ten times. The level drops where it should rise. The second half is better: Newton's method on the valley in four lines, the failure on $x\cos(cx)$ with the numbers read off the run, the millimetre-and-metre example of bad conditioning, and preconditioning solving the valley in seven steps instead of 47. The single most important fix is to rebuild the first half around what the section alone contributes: the descent guarantee with its three-line proof, the general quadratic, and (already there, but misexplained) the fact that a learning rate decides which minima are reachable at all.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought and focus | 3 |
| Pedagogy | 3 |
| Language | 4 |
| Ease and enjoyment | 3 |

## What works

- **The units example** ("Preconditioning"): two length features, one in metres and one in millimetres, give a condition number of at least $10^6$, and "Dividing by the diagonal curvature removes the factor $10^6$ without our ever noticing the units." This is the concrete, slightly funny case that the voice model asks for, and the bound is correct (for a positive semidefinite matrix $\lambda_{\max}$ is at least the largest diagonal entry and $\lambda_{\min}$ at most the smallest).
- **Newton on the valley**: "$\mathbf{H}^{-1} \nabla f(\mathbf{x}) = \mathbf{x}$, so one step from any point reaches the origin, whatever the condition number." The whole point of second-order information in one line.
- **The Newton failure is read from the run**: "The first step overshoots to $x = 5$. The curvature there is positive but small, so the next step divides a large slope by a small number and jumps to $x \approx 25$". I reproduced the trajectory (10, 5.0, 25.0, 27.8, 26.80, 26.83) and the damped one (7.5, then halving the distance to 7.27).
- **The two limitations of diagonal preconditioning** set up Adam and Muon in six sentences and hand the rotated valley to an exercise with a clean answer ($\kappa = 20$ again at $45^\circ$).
- "four exabytes in single precision, before the first step" is a number a reader keeps.
- Steepest descent is derived from Cauchy–Schwarz in three sentences, which is the right size.

## Structural findings

1. **The section's question was answered in the previous section.** See GD-1. If `optimization-intro.md` keeps the valley analysis (recommended in that review), this section should open from it: "On the valley we saw the ceiling and the pace. Three things remain: why a small enough step always descends, what the ideal step would be, and how to approximate it."

2. **The one-dimensional part is below the reader and confirms arithmetic.** "One-Dimensional Gradient Descent" and "Learning Rate" run $x \leftarrow (1-2\eta)x$ at $\eta = 0.2$, $0.05$ and $1.1$, each preceded by the predicted value and followed by a trace plot on a parabola. Five cells and four plots establish that $0.6^{10} \approx 0.006$. After the valley, which did the same in two dimensions with a sweep, this teaches nothing new, and the checks cannot fail (decision 4). Keep the Taylor argument, state and prove the $2/L$ bound, and keep one cell at most. The nonconvex run of "Local Minima" is the one-dimensional experiment worth its space.

3. **The central claim is handed to the appendix although its proof is three lines.** "every learning rate with $0 < \eta < 2/L$ decreases $f$" is the answer to "how large may the step be", and under the duplication rule a small result is duplicated (GD-4).

4. **The general quadratic is missing.** Neither this section nor the previous one says that gradient descent on $\frac12\mathbf{x}^\top\mathbf{H}\mathbf{x}$ decouples along the eigenvectors of $\mathbf{H}$. Without it the valley is one example, the diagonal preconditioner's failure under rotation (Exercise 2) has no explanation in the body, and "these are the eigenvalues ... of its Hessian" is a remark, not a result (GD-7).

5. **"Local Minima" misses its own best lesson** (GD-6): at $\eta = 2$ the deep minimum is unstable and the shallow one barely stable. That is the edge of stability of the previous section in one variable.

6. **Newton's method gets more space than its role in deep learning supports.** About 550 words and three experiments, two of them on the failure and its damping, for a method the section then rules out. The valley computation, one failure run and the cost paragraph would carry the argument; the damped run and the paragraph "Damping is no guarantee" can go to an exercise (GD-10, GD-11).

7. **Length after revision.** Cutting the three parabola runs, the damped Newton run and the `train_2d` narration, and adding the descent lemma and the eigenbasis paragraph, the section would run about 1,900 words and have one line of argument: guarantee, quadratic, ideal step, cheap approximation.

## Passage-level findings

**GD-1** (major, thought). Section opening.
Anchor: "$\eta = 0.45$ the iterate needs 49 steps to come within 1% of the minimum,"
The second paragraph re-tells the result of the previous section (49 steps, never at 0.5, floor of 47) and the first paragraph poses as open the question that result answered. The reader then restarts at $x^2$. One of the two sections must own gradient descent on a quadratic. If the previous section does, replace the first two paragraphs by a two-sentence link and state what is still unexplained: why any small step descends on a function that is not quadratic, and what could remove the dependence on $\kappa$.

**GD-2** (moderate, pedagogy). Under "One-Dimensional Gradient Descent".
Anchor: "should leave $10 \cdot 0.6^{10} \approx 0.06$. We check this numerically."
A check that cannot fail: the loop multiplies by 0.6 ten times. The same holds for the runs at $\eta = 0.05$ (prediction 3.5) and $\eta = 1.1$ (prediction 62). Under the author's decision on confirming cells, keep at most one, and prefer a case the formula does not cover, for instance a non-quadratic function where the factor changes along the way.

**GD-3** (moderate, pedagogy). Under "One-Dimensional Gradient Descent" and "Learning Rate".
Anchor: "The iterates approach the minimum geometrically, each step keeping 60% of"
`show_trace` is plotting code, and four of its seven uses draw dots on a parabola or a cosh. The house rule is that code teaches and does not draw. The plots that carry information are the nonconvex ones (`gd-local-minima`, `gd-newton-s-method-2`); the parabola plots could be one pre-generated three-panel figure (too small, too large, just right) or be dropped.

**GD-4** (moderate, explanation). Under "One-Dimensional Gradient Descent".
Anchor: "$0 < \eta < 2/L$ decreases $f$; :numref:`chap_mdl-optimization` proves this."
The reference points to a whole chapter, and the result is small. State it with its reason. Draft: "If the slope changes by at most $L$ per unit of $x$, then $f$ lies below the parabola $f(x) + f'(x)\epsilon + \frac{L}{2}\epsilon^2$. Putting $\epsilon = -\eta f'(x)$ gives $f(x - \eta f'(x)) \le f(x) - \eta\,(1 - \tfrac{\eta L}{2})\, f'(x)^2$, which is a decrease for every $0 < \eta < 2/L$ and is largest at $\eta = 1/L$ (:numref:`subsec_mdl-gd-smoothness`)." This also replaces the vague $O(\eta^2 f'(x)^2)$ remainder by a usable inequality and explains in advance why $1/\text{curvature}$ is the natural step.

**GD-5** (minor, correctness). Under "One-Dimensional Gradient Descent".
Anchor: "$f: \mathbb{R} \rightarrow \mathbb{R}$ be continuously differentiable. For"
A continuously differentiable function has a remainder $o(\epsilon)$, not $O(\epsilon^2)$; the stated expansion needs a Lipschitz derivative or two derivatives. Say "twice differentiable", or use the inequality of GD-4, which makes the assumption explicit.

**GD-6** (moderate, explanation and correctness). Under "Local Minima".
Anchor: "$7.27$. With $\eta = 2$ the first step jumps over that basin to"
The run is explained as an accident of the first jump. The real reason is sharper and ties to the previous section. The curvature at the deep minimum is $f''(7.27) \approx 1.81$, so that minimum is unstable for every $\eta > 2/1.81 \approx 1.1$: at $\eta = 2$ gradient descent cannot converge to it from any start. At the shallow minimum $f''(-1.83) \approx 0.98$, just under the ceiling $2/\eta = 1$, which is why the iterate is still "bouncing" after ten steps (the stored output is $-1.528$, the amplitude shrinks by only a few percent per step). So the learning rate filters minima by curvature. Rewrite the reading accordingly and connect it to the edge of stability: "A large learning rate does not only jump; it rules out every minimum sharper than $2/\eta$." The closing remark about deep networks then follows from the example instead of being asserted.

**GD-7** (moderate, explanation). Under "Multivariate Gradient Descent".
Anchor: "eigenvalues $\lambda_{\min}$ and $\lambda_{\max}$ of its Hessian in"
Add the general quadratic here (or in the previous section, not both): in the eigenbasis of $\mathbf{H}$ each coordinate is multiplied by $1 - \eta\lambda_i$. Three sentences. It makes the valley general, explains why steepest descent zigzags on any ill-conditioned quadratic, and prepares the rotated-valley exercise.

**GD-8** (minor, language). Under "Multivariate Gradient Descent".
Anchor: "state variables, which the methods with memory in later sections"
A paragraph of interface narration (what `train_2d` receives and returns, where the plotting utility lives) before the reader has a use for it. One sentence and a code comment suffice; the docstring already says the rest.

**GD-9** (minor, language). Under "Multivariate Gradient Descent".
Anchor: "is still $-0.3$. A larger $\eta$ would speed up progress along the floor"
Third telling of the 30-step run and its $-0.3$ (twice in the previous section). The new information is the picture in the plane and the gradient $(-1, -8)$. Keep those, drop the recap. The last sentence of the paragraph ("What the valley needs is a different learning rate in each direction.") is a good hinge; note that the source line carrying it is not reflowed.

**GD-10** (moderate, thought). Under "Newton's Method".
Anchor: "here, and the appendix proves both. First, near a minimum with positive"
"the appendix proves both" covers a theorem (quadratic convergence, which :numref:`subsec_mdl-why-not-newton` does prove) and a cost statement ("the method does not scale to deep learning"), which is not something one proves. The paragraph also arrives after two experiments as "Two further facts", an announcement. State the convergence order where the cosh run shows it, and let the cost paragraph stand without the appendix.

**GD-11** (minor, explanation). Under "Newton's Method".
Anchor: "The iterate reaches the minimum to machine precision within ten steps, with"
The run has two phases that the prose does not mention: four steps that each move by almost exactly $1/c = 2$ (10, 8.0, 6.0, 4.0, because $\tanh(cx) \approx 1$ far out), then four that collapse the error (2.1, 0.53, 0.012, $10^{-7}$). That is the textbook picture of Newton's method, slow far away and very fast near the minimum, and it is visible in the plot. One sentence would turn a demonstration into an observation. Exercise 5 asks for it; the body should at least point at it.

**GD-12** (moderate, correctness and currency). Under "Newton's Method".
Anchor: "four exabytes in single precision, before the first step. And a deep"
Storage of the Hessian is the weakest of the reasons, because Newton-type methods do not need the matrix: a Hessian-vector product costs about one extra backward pass, and conjugate gradients solve the system from such products. The reasons that hold against these matrix-free methods are that each step needs many such products, that minibatch noise corrupts curvature estimates, and that negative curvature misdirects the step. The text gives the third and omits the second. Add one sentence so that a reader who knows about Hessian-vector products is not left thinking the argument fails. (The appendix's opening of "Newton and Quasi-Newton Methods" says exactly this about noise.)

**GD-13** (minor, thought). Under "Newton's Method".
Anchor: "Damping is no guarantee: it only makes a jump into a region of negative"
The damped run is introduced as a remedy, shown to work, charged with losing the fast rate, and then withdrawn as "no guarantee". Four moves for a side issue. Either cut the damped run (Exercise 3 covers the absolute-value remedy) or keep it and end on what it shows.

**GD-14** (minor, correctness). Under "Preconditioning".
Anchor: "later methods precondition whole weight matrices rather than single"
"For this reason" gives one motive to methods that the chapter itself derives differently (:numref:`sec_muon` presents Muon as steepest descent under a matrix norm). Write "This is one motive for methods that treat a whole weight matrix at once."

**GD-15** (moderate, thought). End of "Preconditioning".
Anchor: "everything we have seen. But an exact gradient of a training loss requires"
"the step size and the curvature explain everything we have seen" contradicts the previous section, which showed gradient descent training at the edge of stability and said the quadratic analysis "misses this". Limit the claim to quadratics, or drop the sentence; the transition to noise needs only the second sentence.

**GD-16** (minor, correctness). Under "Summary".
Anchor: "enough for the first-order Taylor expansion to hold. On a quadratic each"
The Taylor expansion always holds; what the section showed is that the decrease is guaranteed for $\eta < 2/L$. Say that.

**GD-17** (minor, correctness of a reading). Under "Preconditioning".
Anchor: "so the path runs straight to the minimum. Seven steps bring both within 1%,"
The cell runs ten steps and prints the tenth iterate; no output shows seven. The number is right ($0.5^7 = 0.008$) and derivable, so write "seven steps would bring", or run seven.

## Code and experiments

- `gd-one-dimensional-gradient-descent-3`, `gd-learning-rate-1`, `gd-learning-rate-2`: confirm arithmetic that the preceding sentence states (GD-2). Stored outputs 0.060466, 3.486784, 61.917364 match.
- `gd-one-dimensional-gradient-descent-4` and the other `show_trace` calls: drawing code (GD-3).
- `gd-local-minima`: outputs 7.269388 and $-1.528$ match; the reading should change (GD-6).
- `gd-multivariate-gradient-descent-3`: output $x_1 = -0.295$ matches "$-0.3$"; third run of the same experiment.
- `gd-newton-s-method-1`, `-2`, `-3`: outputs 0.000000, 26.834133, 7.269860 match the prose. The first deserves a reading of its two phases (GD-11).
- `gd-preconditioning`: output ($-0.004883$, $-0.001953$) is $0.5^{10}$ times the start, as predicted; the "seven steps" is not printed (GD-17).
- The imports cells load `nn`, `F`, `math` and (JAX) `nnx`, `optax`, none of which the section uses.
- PyTorch and JAX share every cell after the imports, so the tabs do not fragment the prose here.

## Exercises

Six problems. Exercise 2 (rotated valley) is excellent and its numbers are right. Exercise 5 (order of convergence on cosh, which turns out cubic because the function is even) is a good surprise. Exercise 6 (stale factorizations, from Boyd and Vandenberghe) is sound but about a regime the section has just ruled out for deep learning; keep it only if space allows.

- **Fix Exercise 1 (defect).** `gd` starts at $x = 10$. On $f(x) = x^4$ the update is $x(1 - 4\eta x^2)$, stable only for $\eta < 1/(2x^2) = 0.005$ at the start, so all five runs overflow within three to five steps (I ran them). Part 3 then asks the reader to analyse the regime "once the iterates are small", which no run reaches. On $|x|^{1.5}$ no run meets the stopping test either: four settle into a two-cycle of amplitude $(0.75\eta)^2 > 10^{-3}$ and the fifth is still at $x \approx 5.8$ after 100 steps. So of fifteen runs only three, all on $x^2$, ever stop, and the table the exercise asks for is mostly "diverged". Start from $x = 1$, or add a smaller learning rate such as $0.001$, so that $x^4$ shows its $1/\sqrt{t}$ decay and $|x|^{1.5}$ shows its cycle.
- **Fix or cut Exercise 4** ("Certifying the lowest minimum"). "Modify the objective ... so that its local minima in $(-20, 20)$ differ in value by at most $10^{-3}$ except for one" does not say what modification is allowed or what is to be produced. The idea (first-order information cannot certify a global minimum) fits in one sentence of the body.
- **Add:** an exercise on GD-6: for $f(x) = x\cos(cx)$, compute $f''$ at each local minimum in $(-20, 30)$ and predict for which $\eta$ each one can be a limit of gradient descent; check with `gd`.

## Still open from October

"The test valley three times" is open and has its centre here: this section re-introduces the valley, its curvatures, its gradient and the 30-step run. Nothing else from October concerns this file.

## For the chapter review

- Order and overlap: the previous section owns the valley, the ceiling and $\kappa$ in fact, and this one claims them in its opening. Assign them once. The edge of stability (previous section) and the stability filter on minima (GD-6) are the same idea and should refer to each other.
- Against the appendix (:numref:`sec_mdl-gradient-based-optimization`): Newton's step and its derivation appear in both at similar length (about ten lines each), which the duplication rule allows. Quadratic convergence, BFGS and trust regions live only in the appendix, correctly. The descent lemma lives only in the appendix and is small, so it should be duplicated here (GD-4). All four references into the appendix resolve to passages that hold the cited content, but ":numref:`chap_mdl-optimization` proves this" should name :numref:`subsec_mdl-gd-smoothness`.
- Assumes: gradients and the Hessian from Basics (:numref:`subsec_calculus-grad`); nothing from the appendix once GD-4 is done.
- Sets up: `d2l.train_2d`, `f_valley`, `valley_grad` for `sgd.md`, `momentum.md` and `adam.md`; diagonal preconditioning for Adam; whole-matrix preconditioning for Muon.
- Suspected pattern: legacy d2l cells (parabola traces, `show_trace`) kept at full length next to rebuilt material, and section endings that announce the next section ("The next section takes up this problem") in addition to the Summary.
