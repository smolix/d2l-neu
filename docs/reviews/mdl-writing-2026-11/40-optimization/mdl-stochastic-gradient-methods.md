# Stochastic Gradient Methods

Source: `chapter_mdl-optimization/mdl-stochastic-gradient-methods.md` (read to the slides marker, line 1316). About 7,000 words of prose and mathematics in the body, 2,000 more in ten exercises; ten code cells, all NumPy.

## Verdict

This is a good section, close to the bar in its first half. It has one line of argument (noise leaves a floor; the floor is proportional to the step; every remedy trades the floor against speed) and it keeps to it for 7,000 words. The scalar recursion $e_{k+1} = \rho^2 e_k + \eta^2\sigma^2$ is solved once and then reused for the floor, the time to reach it, the $1/k$ constant, and the exact comparison of schedules, which is the economy the model texts have. I recomputed every quoted number that a short script can reach (floors, step counts, the $1/k$ constants and slopes, the schedule table, the two-coordinate valley values, the numbers given in exercises 2, 5, 7 and 8) and found no error, and the stored PyTorch outputs agree with every range the prose quotes. I found no mathematical error in the four proofs. The most important thing to fix is not in the section alone: its first movement (the noise-ball recursion, the fixed point, the Robbins-Monro conditions) is derived at the same level, in nearly the same words, in :numref:`sec_sgd` of chapter 9, and its schedule comparison repeats `lr-scheduler.md` "Four Schedules on the Valley" on a different valley. One of the two must become the home and the other a one-clause citation (guide 7.6, 7.7). Inside the section, the second half is weaker than the first: the single-neuron experiment confirms only that a measurement is consistent with an upper bound, and the variance-reduction part spends 900 words on methods the text itself says do not help deep networks while missing the place in the book where control variates do matter (policy-gradient baselines, :numref:`sec_baselines`).

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought | 4 |
| Pedagogy | 4 |
| Language | 4 |
| Ease and enjoyment | 3 |

## What works

- **The opening** (lines 4 to 29). Three paragraphs: the question, a number the reader can check in the head ($0.1/1.9 \approx 0.053$, so the iterate "stays about $0.23$ away from the minimizer however many steps we take"), and the route. "Three numbers set the loss at which a run ends: the size of the noise, the step, and the number of steps we can afford" is a sentence a reader keeps. Preserve this opening as written.
- **The floor and the transient as two parts of one formula** ("The Noise Floor"). Solving the recursion through its fixed point and then reading :eqref:`eq_mdl-opt-noise-floor-trajectory` as transient plus floor is the right order, and the worked numbers ("A third of the step gives about a third of the floor and takes four times as many steps to reach it") turn the formula into the trade-off that organizes the rest.
- **Sentences with a voice.** "a flat direction lets the iterate wander farther, by exactly the factor that its flatness takes off the loss"; "a hundred times more examples per step make the gradient ten times quieter"; "Ill-conditioning costs steps on the slow coordinate; noise costs loss on both"; "a ten times smaller gradient norm needs $10^4$ times the budget". Each is true with the figure of speech removed. Protect them.
- **The valley computation** (paragraph beginning "On the running example the two coordinates"). The result that the stiff coordinate, which gradient descent kills in one step, holds two thirds of the loss floor is a surprise, it is explained in one clause, and it is exactly the kind of sharp example the bar asks for.
- **The $1/k$ constant.** The derivation says which step is the hard one and what would go wrong without it ("Without it we would find $A = c\sigma^2/(2\mu)$ for every $c$"). The check ($k\,e_k = 1.0001$ at $k = 10^6$; slopes $-0.50$ and $-0.20$ below the threshold) is correct, and the consequence for the valley (the condition number "reappears in $k_0$") ties the abstract threshold to the chapter's example.
- **The exact schedule comparison.** "the expected loss of any schedule can be computed exactly, with no runs and no random numbers" is a better experiment than chapter 9's simulated one, and the text is candid about its scope ("The quadratic model ranks the final losses of the shapes, not such operational merits").
- **Candour about hypotheses.** The text says where $\mathbb{E}\|\mathbf{g}\|^2 \le G^2$ fails, that $\sigma^2$ is infinite for least squares on $\mathbb{R}^d$, that the convex bound is loose by more than an order of magnitude on its own example, and that SVRG runs at more than twice the step its theorem allows.
- **"Why does the guarantee speak of a *random* iterate?"** A reader's question asked and answered in four lines.
- **The exercises** are the best part of the section for a strong reader (see below).

## Structural findings

**S1. The first movement is taught twice in the book.** `chapter_optimization/sgd.md` "The Noise Ball" (lines 146 to 225) writes the same step $x_{t+1} = (1-\eta\lambda)x_t - \eta\xi_t$, squares it, drops the cross term for the same reason, solves the same fixed point to $\eta\sigma^2/(\lambda(2-\eta\lambda))$, says "it shrinks as the curvature grows" and "Far from the minimum the first term dominates, and SGD makes progress like gradient descent", and then states the two Robbins-Monro conditions with the same glosses ("travel arbitrarily far"; noise "stays bounded"). `lr-scheduler.md` "Progress and Noise" then explains that the square-summable condition is not needed on the quadratic, which this section also explains under "The Robbins--Monro Conditions". The appendix adds the trajectory, the time to the floor, the loss floor, the $\eta/b$ coupling and the proofs; those are worth having. But the shared core is at the same level in both places, so a reader who follows the pointer from chapter 9 rereads two pages before anything new arrives. Direction: decide the home. Under guide 7.6 the derivation belongs here and chapter 9 should state the formula with its intuition and point here. Whichever way it goes, the two texts also use different valleys (curvatures $0.2$ and $4$ in chapter 9; $1$ and $1000$ here) and different step indices ($t$ there, $k$ here), which makes cross-reading harder than it needs to be.

**S2. Five test problems in 7,000 words.** The scalar bowl, the $1000{:}1$ valley, a logistic regression, a $\tanh$ neuron, and a ridge regression each arrive with their own data cell and notation (`A`, `A_nc`, `A_r`). The sentence that introduces the logistic problem says it "serves all the measurements of this section", which is not true. Direction: run SVRG on the logistic problem already defined (its $f_i$ are convex and smooth, and the regularizer gives strong convexity, so the hypotheses of the cited theorem hold), which removes one data cell and lets SGD's measured floor from the schedule cell be the comparison. Keep the $\tanh$ neuron only if the experiment is strengthened (S3).

**S3. The $\tanh$-neuron experiment proves little.** The cell measures a slope of about $-0.6$ and the prose concludes that "a slightly steeper measured slope is consistent with" an upper bound of slope $-1/2$. Any slope at or below $-1/2$ would be consistent. The section's other experiments confirm a predicted number; this one cannot fail. Under the author's decision that experiments wait, the prose fix is to say plainly what the run shows: the bound is not violated, the best iterate falls much faster (slope near $-1$) and cannot be found without exact gradients. That second observation is the real content and deserves the lead. A later redesign could plot the two terms of :eqref:`eq_mdl-opt-ghadimi-lan` against $\eta$ at fixed $K$, the nonconvex twin of :numref:`fig_mdl-opt-budget-tradeoff`.

**S4. Variance reduction: proportion and the missing connection.** About 900 words and two cells go to SVRG and SAGA, and the part ends "On deep networks these estimators have not given consistent gains". A deep learning reader is entitled to ask why it is here. Two things would earn it its place. First, name the use the book does make of the idea: the baseline in policy gradients is a control variate (:numref:`sec_baselines`), and the definition given here ("subtract ... a quantity that moves with it and whose mean we know, and then add the mean back") is exactly what that section needs. Second, shorten the method description: the SAGA paragraph and the Johnson-Zhang hypotheses can each lose a third. The proposition and the SVRG cell (floor predicted by the formula of part 1, then removed) should stay; the predicted-against-measured SGD floor is one of the best checks in the section.

**S5. An orphaned paragraph.** "For the running example we model this noise directly..." (under "Unbiasedness and the $1/b$ Law") defines the noise model of the valley 160 lines before the valley is used, between the gradient-noise scale and the logistic check. A reader landing here cannot tell what it is for. Move it to the paragraph "On the running example the two coordinates of" under "The Noise Floor".

**S6. The closing is stated three times.** The last body paragraph ("We can now measure ..., predict ..., choose and compare ..., tune ..., and cancel ...") lists the section's results; the Summary lists them again with formulas; the opening had already quoted the $0.053$ and $0.23$ that the body recomputes. Guide 12.2 names "We can now ..." as the formula not to use, and the same construction closes the next section and `mdl-convexity.md`. Direction: delete the "We can now" sentence, keep the two sentences about the shared step size (they are the bridge to the next section), and let the Summary be the only restatement.

**S7. Length.** The section is not two sections; the four movements depend on one recursion. It could lose about 800 words without loss: the duplicate core of S1 if chapter 9 becomes its home, a third of S4, the "We can now" paragraph, and the second statement of the $0.053/0.23$ example.

## Passage-level findings

**SGM-1** (minor, language). Under "The Cost of a Full Gradient". Anchor: "With $n = 10^9$".
The sentence runs from the cost of one pass, through a colon, into the valley's 7,000 steps and its curvature ratio: three facts and a forward dependence in one sentence, in the first paragraph of the body. Split after "thousands of steps." and give the valley its own sentence with a reference to :numref:`sec_mdl-gradient-based-optimization`, where the 7,000 comes from.

**SGM-2** (minor, explanation). Under "Unbiasedness and the $1/b$ Law". Anchor: "which serves all the measurements of this section".
False as written (see S2); the $\tanh$ and ridge problems follow. Either make it true or write "which the next two measurements use".

**SGM-3** (moderate, pedagogy). Same subsection. Anchor: "For the running example we model this noise directly".
See S5. Also, "The variance $\sigma^2$ stands for a diagonal entry of $\boldsymbol{\Sigma}/b$" is the only place where the scalar $\sigma^2$ of the next 400 lines is tied to the proposition just proved; it is too important to sit in an aside. State it where the scalar model is introduced: "the noise variance $\sigma^2$ plays the role of an entry of $\boldsymbol{\Sigma}/b$, so everything below that is proportional to $\sigma^2$ falls like $1/b$".

**SGM-4** (minor, explanation). Figure caption under "The Noise Floor". Anchor: "three times the per-coordinate root-mean-square distance".
The caption's last sentence cannot be parsed on first reading: radii "proportional to $\sqrt{\eta}$, at $\eta$ and at $\eta/2$: three times the per-coordinate root-mean-square distance ..., enlarged so that the scatter is visible". Rewrite: "The dashed circles mark the noise ball at this step and at half of it; their radii, drawn at three times the root-mean-square distance $\sqrt{\eta\sigma^2/(2\lambda)}$, differ by $\sqrt2$." The rendered figure is clear and its labels do not collide.

**SGM-5** (minor, thought). Under "The Noise Floor". Anchor: "Take $\lambda = \sigma = 1$. At $\eta = 0.1$ the floor is".
The $0.053$ and $0.23$ were given in the opening. Here they are derived again. Keep the opening's version as the teaser and begin this paragraph with the new content, the comparison of the two steps: "At $\eta = 0.3$ the floor is $0.176$, three times that of $\eta = 0.1$, and it is reached in 9 steps instead of 36."

**SGM-6** (minor, explanation). Same subsection. Anchor: "because its step sits at $\eta\lambda_2 = 1$".
Correct, and a nice point, but "its step" reads as if the coordinate had its own step. Write "because for this coordinate $\eta\lambda_2 = 1$, where the loss floor is $\eta\sigma^2/2$, twice the small-step value".

**SGM-7** (minor, correctness; confident). Under "The Constant in a $1/k$ Schedule". Anchor: "*If $2c\mu < 1$, then $e_k$ decays like $k^{-2c\mu}$".
The proposition covers $2c\mu > 1$ and $2c\mu < 1$ and is silent on $2c\mu = 1$, where $e_k = \Theta(\log k/k)$ (my iteration at $c = 0.5$ gives $k\,e_k \approx 12$ at $10^6$ and a slope of $-0.98$, as that rate predicts). Exercise 2(d) asks the reader to run $c = 0.5$ without warning. Add the boundary case to the statement in one clause.

**SGM-8** (moderate, correctness of presentation; confident). Same subsection. Anchor: "**Derivation of the constant.** We find $A$ to leading order".
The text is honest that this is a leading-order matching and not a proof, and sends the induction to an exercise. But it then closes the following paragraph with $\blacksquare$, and the proposition also asserts "For every $c > 0$, $e_k \to 0$" and the $k^{-2c\mu}$ rate, for which the paragraph offers an approximation ("$\approx (k_0/t)^{2c\mu}$") and no argument for the injected noise. A reader cannot tell what has been proved. Direction: either relabel the statement as a claim with "Derivation" and drop the $\blacksquare$, or keep "Proposition", say "we derive the constant and leave the proof to exercise ...", and drop the $\blacksquare$. The exercise (2a to 2c) is a complete proof and is correct; I checked the induction and the hint.

**SGM-9** (minor, language; pattern, 3 instances). Anchors: "that is not routine is the shift on the left"; "This is the step that is not routine: the inner product"; and the third under "Random Iterates on Nonconvex Losses" ("Take the expectation given $\mathbf{x}_k$. This is the step that is not routine").
Saying which step is the hard one is what guide 10.2 asks for, and the first instance does it well. Three uses of the same words in one section make it a formula. Vary two of them: "Everything rests on the next line", or simply lead the sentence with the content ("Conditioning on the iterate does two things at once: ...").

**SGM-10** (minor, explanation). Under "The Constant in a $1/k$ Schedule". Anchor: "Intuitively, with $c < 1/(2\mu)$ the steps shrink faster than the iterate can".
Good intuition, placed after the numbers. It would help more before the proposition, as the picture that precedes the statement (the house style for proofs). "Intuitively," appears three times in the section; one can go.

**SGM-11** (moderate, explanation). Under "Schedule Shapes under the Exact Recursion". Anchor: "each preceded by a short linear warmup".
The figure draws a warmup ramp on every schedule; the cell runs "(without the warmup)"; the text defines WSD as holding $\eta_0$ "for the first $80\%$ of the budget" and never says what the "warmup" in its name is. Warmup is first defined in the next section. A reader who lands here sees a figure that disagrees with the formula beside it. Direction: one clause where WSD is named ("the warmup, a short initial ramp that :numref:`subsec_mdl-schedules-warmup` discusses, is omitted here because the quadratic has no unstable start"), and say the same in the caption.

**SGM-12** (minor, thought). Same subsection. Anchor: "where WSD's plateau is valued for the operational reason above".
The operational reason is given twice within 50 lines (first at "WSD is popular for an operational reason"), and a third time in exercise 8(c). State it once.

**SGM-13** (minor, explanation). Same subsection. The cell labels the schedule $\eta_0/(1+k/k_0)$ as `'c/k'`, the figure as "$c/k$", and the prose two paragraphs later explains that it "is $c/(k + k_0)$ with $c = 15$ and $k_0 = 50$". The earlier cell under "The Robbins--Monro Conditions" does the same conversion for $c = 80$, $k_0 = 100$. Introduce the schedule once in the form $c/(k+k_0)$ and use that parametrization in both cells' prose.

**SGM-14** (minor, explanation). Under "Averaged Iterates on Convex Losses". Anchor: "It holds for Lipschitz".
"It holds for Lipschitz losses with bounded features, such as the logistic loss" sits oddly beside the section's own logistic problem, whose $\ell_2$ term makes the gradient unbounded; exercise 10 has to restrict to a ball to apply the bound. Add "(without a quadratic penalty)" or point at the exercise's device.

**SGM-15** (minor, thought). Under "Random Iterates on Nonconvex Losses". Anchor: "the noise term is $140$ times the".
"Transient" has meant the term $2\Delta/(\eta K)$ since the start of this part. At the balanced step that term equals the noise term; the $2L\Delta/K$ that is 140 times smaller is the deterministic rate, which enters the stated bound only to cover the case $\eta = 1/L$. Write "the noise term is 140 times the deterministic rate $2L\Delta/K$".

**SGM-16** (minor, language). Under "The Rate on a Single $\tanh$ Neuron". Anchor: "Each term is bounded and smooth, but the sum is not convex".
The "but" suggests the terms are convex and only the sum is not. No single term is convex either. Write "Each term is bounded and smooth, and none is convex".

**SGM-17** (moderate, pedagogy). Same subsection. Anchor: "The theorem gives an upper".
See S3. Rewrite direction: "The average falls about twelvefold for a 64 times larger budget, a slope near $-0.6$. The bound has slope $-1/2$ and is respected, with room: a bound says how slowly the quantity may fall, not how slowly it does. The best iterate of each run falls much faster, with a slope near $-1$. No algorithm that sees only noisy gradients can return that iterate, which is why the theorem speaks of a random one."

**SGM-18** (moderate, pedagogy). Under "Control Variates". Anchor: "The idea is to subtract from the noisy".
The definition is clear but the reader meets no reason to care beyond finite sums. Add one sentence naming :numref:`sec_baselines` (see S4). The sentence before it ("When a model fits every example exactly, they vanish, and ... a fixed step then converges linearly with no correction") is the more important fact for deep learning, since large networks interpolate; it is buried in a subordinate position and handed to an exercise. Give it its own two sentences and say that this is one reason constant-step SGD works better on overparametrized networks than the floor formula suggests.

**SGM-19** (minor, language). Under "Control Variates". Anchor: "**Proposition (the variance of the control variate vanishes at the".
The title says "at the optimum"; the statement bounds the variance by the distance to the snapshot, which vanishes when iterate and snapshot coincide, wherever that is. Title it "the variance is bounded by the distance to the snapshot" and let the following paragraph draw the consequence at $\mathbf{x}^\star$, as it already does.

**SGM-20** (minor, explanation). Under "SVRG and SAGA". Anchor: "the theorem asks for $\eta$ below about $0.008$".
The stored output prints `L_max = 38, 1 / (4 L_max) = 0.0066`, and the prose gives "about $30$" and "about $0.008$". Both are inside the quoted range, but the reader sees $0.0066$ on the page directly above "$0.008$". Write "below $1/(4L_{\max})$, the second number printed (between about $0.006$ and $0.01$ across draws)".

**SGM-21** (minor, thought). Same subsection. Anchor: "so the theorem's constants are loose".
One run at one step size shows that the hypothesis is not necessary on this problem. Say that; "the constants are loose" is a claim about the theorem in general. The stored output also shows something the prose omits and a reader will notice in the plot: after its first epoch SVRG's gap is $2$, about a hundred times SGD's. SVRG loses early and wins late. One sentence.

**SGM-22** (minor, explanation). Same subsection. Anchor: "(The theorem takes a random inner iterate as the next".
A parenthesis the reader cannot skip, since the code does the opposite of the theorem. Promote it to a plain sentence.

**SGM-23** (minor, language). Closing paragraph. Anchor: "We can now measure the noise of a minibatch gradient".
See S6.

**SGM-24** (minor, explanation; notation). The italic $A$ of :eqref:`eq_mdl-opt-one-over-k-constant` shares the section with the data matrix `A` of four cells and with $\mathbf{a}_i$; $\lambda$ is a curvature here and the weight-decay coefficient in the next section; $\sigma^2$ is a noise variance here and a "gradient scale" there; step index $k$ here, $t$ there. The chapter's notation table admits the last two. For the first, rename the constant (for instance $C_c$ or $a_c$).

**SGM-25** (minor, enjoyment; pattern). Capitalization of exercise pointers: this section writes "exercise :exref:" in lower case, the next writes "Exercise :exref:". The body hands ten results to exercises, among them three that carry numbers quoted in the text with no cell behind them ($2.7 \times 10^{-3}$ for full linear decay, the $\Theta(\log K/K)$ rate, "more than an order of magnitude" for the convex bound). I verified the first two; the third I did not run. This is acceptable in an appendix, but a reader cannot check these from the page, so each should say "the exercise computes" and not read as a result shown.

## Code

All ten cells compute; four also plot a computed curve with `d2l.plot`, which the house rule allows. Every printed number is read by the prose, and the ranges quoted for the unseeded runs contain the stored values (ratio $2.29$ in "1.7 to 2.8"; $71$ in "20 to 90"; slope $-0.58$; SGD floor $3.1 \times 10^{-2}$ against a predicted $2.7 \times 10^{-2}$; SVRG $8 \times 10^{-10}$). Two remarks. The imports cell loads `torch`, `nn`, `F` (and `jax`, `flax`, `optax`, `tensorflow`, `mxnet` in the other tabs) and no cell uses any of them: the section is framework-free and the four tabs differ only in an import nobody needs. And cell IDs carry the names of other notebooks (`gradient-based-optimization-sgd-variance`, `adaptive-stochastic-methods-schedules`), which is invisible to the reader but will confuse an editing agent.

## Exercises

Ten problems, all substantial, and the strongest set I would expect in the chapter. Exercises 2 (the induction behind the $1/k$ constant), 5 (best constant step, $\Theta(\log K/K)$), 6 (iterate averaging reaches $\sigma^2/(\lambda^2 K)$ at any step), 7 (which Robbins-Monro condition the quadratic needs) and 9 (interpolation) each teach a result the body only states, give the numbers to check against, and are graded in parts. I verified the stated numbers in 2(d), 2(e), 5(b), 5(c), 7(c) and 8(a), and the algebra of 1(a), 2(a), 2(b), 4(a), 6(b) and 9(b); all are correct.

- Difficulty marks are inconsistent: exercises 3, 8 and 10 carry none.
- Exercise 3 is labelled `ex_mdl-adaptive-stochastic-methods-svrg-inner-loop` and exercises 1 and 2 `ex_mdl-gradient-based-optimization-...`. Harmless to the reader, a trap for an editor.
- Exercise 1 (noise scale, after McCandlish et al.) is good but long, and overlaps what :numref:`sec_batch_size` measures on networks. Keep parts (a) and (b); part (c) could point to that section instead of asking for the trajectory of $B_{\textrm{noise}}$.
- Exercise 4(c) asks about random reshuffling, clipping and stale gradients in one question; each is a real topic and the question has no stated answer to check. Split or give the expected classification.
- Missing: nothing asks the reader to derive the without-replacement factor $(n-b)/(n-1)$ quoted in the text (it is credited to :numref:`sec_sgd`), which is a two-line exercise that fits here better than there.

## Still open from October

`docs/reviews/writing-review-2026-10.md` does not cover this section.

## For the chapter review

- **Fit.** Third of seven; it assumes the stability ceiling and the $\kappa$ step count from `mdl-gradient-based-optimization`, and subgradients, strong convexity and Jensen from `mdl-convexity`, and cites both by label. It sets up the next section explicitly (one step size for all coordinates). The order is right.
- **Overlap with chapter 9.** The division is sensible in design (chapter 9 measures on networks, the appendix proves) and is kept for the rates, the $1/k$ constant, the exact schedule recursion and SVRG, none of which chapter 9 teaches. It breaks down for the noise ball and the Robbins-Monro conditions, which both chapters derive in full at the same level (S1), and partly for the schedule comparison on a quadratic. Chapter 2 (`linear-regression.md:495`) and chapter 9 both point here for the noise cloud, so this section is the natural home.
- **Chapter-wide patterns I suspect.** The closing formula "We can now ..." (here, in the next section, and in `mdl-convexity.md`); results and numbers delegated to exercises; a framework imports cell in sections whose code is pure NumPy; the running valley used well but its numbers repeated (the 3,450 steps appear twice here, the 0.053 floor twice); the phrase "the step that is not routine".
- **Level.** Consistent, and above chapter 9 throughout the last three movements. A reader arriving by lookup can use each `##` part on its own, because each restates the scalar recursion it needs.
