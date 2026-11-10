# Stochastic and Minibatch Gradient Descent

Source: `chapter_optimization/sgd.md` (lines 1 to 1074, up to the slides marker). About 3,900 words of body prose plus 1,400 words of exercises; 19 code cells (about 330 lines of code per tab), no pre-generated figures.

## Verdict

The first third of this section is at the bar. The noise ball is derived in six lines from one recursion, the formula predicts the shape of the cloud the reader has just seen (standard deviation 0.5 along the flat direction, 0.125 along the steep one), and a measurement confirms it and shows where the small-step approximation stops holding. The remaining two thirds do not hold that level. The section turns into a hardware tutorial and a utilities file: an arithmetic-intensity argument whose experiment, by the text's own admission, measures something else; a `Timer` class, `record_run`, `train_with_rule` and `train_with_optimizer` with tab notes about warm-up passes; and a closing race on a six-parameter linear regression in which full-batch gradient descent, the method the section argues against, wins on the clock and the prose explains why that does not count. The single most important fix is the second half: either make the two hardware experiments show what the prose claims, or cut them to what they do show and send the roofline argument to chapter 13, which owns it.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought and focus | 3 |
| Pedagogy | 3 |
| Language | 4 |
| Ease and enjoyment | 3 |

(The part up to "Sampling without Replacement" would score 4 or 5 on every line; the part after it, 2 or 3.)

## What works

- **The opening arithmetic**: 1.28 million images, 90 epochs, batches of 256, "which comes to 450,000 parameter updates. The same number of updates with exact gradients would take 450,000 passes over the data, 5,000 times the work." Both numbers check. A reader feels the gain before any formula.
- **"The Noise Ball"**. The recursion, "we square both sides and take expectations", the fixed point, and then the reading: "The flat directions, where gradient descent is slowest, are also where the noise carries the iterate farthest, because the pull back toward the minimum is weakest there." This sentence joins the two difficulties of the chapter and should be protected. I recomputed 0.2525 and 0.0156; the stored outputs (0.2489, 0.0159; JAX 0.2495, 0.0153) agree.
- **The measurement teaches beyond the formula**: "Along $x_2$ it shrinks by a little more than half, because there $\eta\lambda = 0.4$ is not small and the exact formula applies." This is the kind of confirming cell the author's decision allows.
- **The trade-off is stated as a trade-off**: "Halving the learning rate and doubling the batch size shrink the ball by the same factor, at different costs: the first doubles the number of steps needed to reach the ball, the second doubles the work per step."
- **Robbins and Monro in four sentences with the reason for each condition** ("so that the iterate can still travel arbitrarily far"; "so that the noise injected over the whole run ... stays bounded"), and a candid limit: "these conditions are a guide rather than a recipe".
- **$1/\sqrt{b}$**: "a batch 100 times larger, at 100 times the computation per step, makes the gradient error only 10 times smaller."
- The learning rates of the race are chosen by a stated rule ($\eta/b$ between 0.004 and 0.005, "so by the noise ball their iterates fluctuate about equally"), which is how a fair comparison should be set up and said.
- Exercises 1, 5 and 8 (see below).

## Structural findings

1. **The section is two sections, and the second is the weaker.** Part A (to the end of "Sampling without Replacement", about 1,900 words) answers "what does gradient noise do, and what do decay and averaging do about it". Part B ("Batched Computation" and "Minibatch SGD on the Airfoil Data", about 2,000 words and 250 lines of code) is about hardware and about helpers for later sections. If Part B stays at its present size, split at "### Batched Computation" into "Stochastic Gradient Descent" and "Minibatches in Practice". My recommendation is different: shrink Part B to about 700 words by moving the arithmetic-intensity argument to :numref:`chap_performance` (the October review already asked for the cache tutorial to go there), dropping the micro-batch paragraph, moving `Timer` out, and reducing the helper narration to code comments. Then no split is needed.

2. **The opening re-teaches Basics instead of building on it.** :numref:`subsec_linear-regression-minibatch-sgd` already gave minibatch SGD, unbiasedness, the $1/|\mathcal{B}|$ variance, and the noise ball with its constant ($v \approx \eta s^2/(2|\mathcal{B}|)$, by the same fixed-point argument). The first paragraph here says so ("Every model since ... has been trained on minibatches"), and "Sampled Gradients" then derives unbiasedness from the definition as if new. What is new here is the curvature: the $\lambda$ in the denominator, the exact factor $2 - \eta\lambda$, and the anisotropic cloud. Open on that. See SGD-2 and SGD-7.

3. **Two experiments are explained away** (guide section 9: "Do not keep it and explain what it fails to show"). The batched matrix product is introduced by a memory argument and then described as showing "the launch overhead and the bandwidth of the cache more than the speed of main memory" (SGD-10). The race is introduced to show which batch size "lowers the loss fastest" and is won on the clock by gradient descent, "but only because the dataset is tiny" (SGD-13). Both need a redesign or a narrower claim.

4. **Proportion inside "Minibatches".** The finite-population factor $(n-b)/(n-1)$, which the text says "is close to 1 while the batch is a small part of the dataset", gets a 25-line derivation. Reshuffling across an epoch, which is what every training loop does and where the independence assumption of the whole section fails, gets four sentences and an exercise (SGD-8).

5. **The noise model's limit is not stated.** The section models gradient noise as additive with constant variance and concludes that constant-step SGD never converges. For a network that fits its training data, every per-example gradient vanishes at the minimum, the noise vanishes with it, and constant-step SGD does converge. The reader of a 2026 book trains such networks. One candid paragraph is needed (SGD-4).

6. **Two endings** (SGD-15), and a "Concise Implementation" subsection whose content for the reader is one sentence (SGD-14).

7. **Looked for and not found:** what noise costs in rate. The section says a constant rate stalls and a $1/t$ rate converges, but not how fast: the error of SGD with decaying steps falls like $1/t$, against the geometric decay of gradient descent on the same valley. One sentence with a pointer to :numref:`subsec_mdl-step-size-decay` would complete the answer to "what is lost".

## Passage-level findings

**SGD-1** (major, thought). Whole section; seam under "Batched Computation".
Anchor: "Statistics alone would not make minibatches the default: $b$ examples in"
From this sentence on the section changes subject from optimization to hardware and utilities. See structural finding 1 for the two options. The sentence itself is a good hinge and can stay as the opening of whatever remains.

**SGD-2** (moderate, thought). Under "Gradient Variance and Batch Size".
Anchor: "proportionality of the noise ball to $1 / |\mathcal{B}|$; here we derive the"
"here we derive the exact constant" overstates the difference: the Basics passage already has the constant $\eta s^2/(2|\mathcal{B}|)$. And the back-reference comes 240 lines into the section, after unbiasedness and the noise ball have been derived afresh. Move the reference to the section opening ("we saw that ... ; this section adds the curvature and makes the statement exact"), cut the unbiasedness display in "Sampled Gradients" to a clause, and say here only what trace and covariance add over the scalar $s^2$.

**SGD-3** (minor, language). Under "SGD on the Valley".
Anchor: "The update rule `sgd_2d` has the form that `d2l.train_2d` of"
The paragraph is interface narration and its first sentence is hard to parse ("has the form that `d2l.train_2d` of :numref:`sec_gd` iterates from the starting point"). Replace by one sentence: "`sgd_2d` is the `gd_2d` of :numref:`sec_gd` with noise added to the gradient." The four-line cell says the rest. The valley is also re-introduced with its formula, both second derivatives and its condition number; a reference is enough by now.

**SGD-4** (moderate, correctness by omission). Under "SGD on the Valley".
Anchor: "does not shrink the cloud. Near the minimum the true gradient vanishes but"
True of the additive-noise model, and of least squares with noisy labels. Not true in general: the noise is the spread of the per-example gradients at the current point, and where the model fits every example (an overparameterized network trained to zero loss) that spread is zero at the minimum. The same applies to "A constant learning rate therefore forces a choice." Add after the noise-ball formula: "We assumed that the noise has the same variance everywhere. In a network the variance changes along the run, and for a model that fits every training example it vanishes at the minimum, so the ball shrinks to a point. The formula describes the common case in which the examples still disagree at the end of training, and $\sigma^2$ should be read as the variance near where the run ends." Exercise 4 (variance after training) then has a reason to exist. The appendix has an exercise on interpolation that can be cited.

**SGD-5** (minor, explanation). Under "The Noise Ball".
Anchor: "that the noise injected over the whole run, whose variances add up to"
"whose variances add up to $\sigma^2 \sum_t \eta_t^2$" ignores the contraction between steps, so it is an upper bound on what remains, not the sum. "is at most" would be exact. In the same paragraph, add the rate (structural finding 7).

**SGD-6** (minor, pedagogy). Under "Gradient Variance and Batch Size".
Anchor: "airfoils that NASA measured in a wind tunnel. Each example has five"
The dataset, its five features, its standardization and its loader (two cells and two tab notes) are introduced in the middle of the argument about variance, for a measurement that uses the data only as a source of 1,500 gradient vectors. The names of the features are never used again. If the variance cell goes (SGD-7), introduce the data where the race needs it; in either case one sentence on the data is enough ("1,500 wind-tunnel measurements of airfoil noise, five standardized features, one target").

**SGD-7** (moderate, pedagogy; experiment). Under "Gradient Variance and Batch Size".
Anchor: "The measured errors lie on the predicted line over a factor of 500 in"
This check cannot fail. The cell draws minibatches with replacement from the 1,500 gradients whose covariance it has just computed, and compares the sample mean of the squared error with its expectation. It verifies the law of large numbers, not anything about networks, and the 60 lines that build the per-example gradients with `vmap` serve only this. The Summary then cites it as a measurement ("which we measured over a factor of 500"). Under the author's decision 4 the noise-ball cell is the section's one allowed confirmation. Options: (a) cut the two cells and the plot, keep the derivation and the $1/\sqrt{b}$ remark; (b) make the cell teach something the derivation does not: draw the batches without replacement as well, so that the plot shows the bend at $b = 512$ of 1,500 where the factor $(n-b)/(n-1)$ is 0.66, which replaces the derivation of SGD-8 by a picture; or measure $\operatorname{tr}\boldsymbol{\Sigma}$ before and after training (Exercise 4), which shows that the "constant" changes by a large factor during a run.

**SGD-8** (moderate, thought). Under "Sampling without Replacement".
Anchor: "have errors that point, on average, slightly against each other:"
A neat derivation of a factor the text itself calls close to 1, at the length of the noise-ball derivation. The appendix cites this section for it (:numref:`sec_mdl-stochastic-gradient-methods`: ":numref:`sec_sgd` derives it"), which inverts the author's rule that derivations live in the appendix and results in the main text. State the result with its one-line reason (the $n$ errors sum to zero, so two distinct examples are slightly anticorrelated) and its two limits ($b \ll n$, $b = n$); move the derivation to the appendix. Use the space for reshuffling: say plainly that the analysis of the section assumed independent draws, that practice does not make them, and what is known (within one epoch the errors of the minibatches cancel exactly, since each example is used once). The citation of Mishchenko, Khaled and Richtárik (2020) is, as far as I can confirm from memory, accurate.

**SGD-9** (moderate, correctness). Under "Batched Computation".
Anchor: "A current GPU performs on the order of a hundred floating-point operations"
The figure agrees with the ridge points that :numref:`chap_performance` tabulates (117 to 295 FLOP per byte), but those are for bfloat16 on tensor cores. The experiment that follows forces full float32 (`precision='highest'`; PyTorch's default), where peak throughput is several times lower. The stored PyTorch plot levels off near 0.15 microseconds per example, about 14 TFLOP/s for $2 \cdot 1024^2$ operations, which against a bandwidth of about 1 TB/s is a ratio near 10 to 20, not 100. State the precision with the number, or give the float32 figure, since that is what the reader is about to measure.

**SGD-10** (major, pedagogy; experiment). Under "Batched Computation".
Anchor: "so the measurement shows the launch"
The paragraph before the cell argues from arithmetic intensity: with one example per call "the arithmetic units therefore wait for memory most of the time". The paragraph after the cell concedes that the 4 MB weight matrix sits in cache and that the curve shows call overhead. So the experiment does not test the argument it follows, and the reader is told so only afterwards. The stored curve bears this out: it follows the $1/b$ line up to $b \approx 16$, which is a fixed cost per call of about 18 microseconds (PyTorch) or 76 (JAX), the effect :numref:`sec_linear_regression` already measured. Fix by choosing one: (a) keep the experiment, introduce it as a measurement of the fixed cost per call, cut the arithmetic-intensity paragraph to two sentences that point to :numref:`chap_performance`; or (b) keep the argument and use a matrix that does not fit in cache (for example $8192 \times 8192$, 256 MB), on the development machine. Option (a) costs nothing and matches where the roofline is taught.

**SGD-11** (moderate, correctness of a reading). Under "Batched Computation".
Anchor: "levels off at the time of the arithmetic itself. Between one example per"
In the stored JAX output the curve does not level off: it is still falling at $b = 4096$ (about 0.055 microseconds per example, below the PyTorch plateau) and stays within a factor of three of the $1/b$ line throughout. The shared prose describes a "level part" that only the PyTorch tab shows, and "A minibatch should therefore be at least large enough to reach the level part" has no referent in the JAX tab. Under guide section 9 the shared prose must be true of every tab. Until the experiment is redesigned, describe what both show: the time per example falls by two orders of magnitude or more from $b = 1$, almost in proportion to $1/b$ at first.

**SGD-12** (moderate, thought). Under "Batched Computation".
Anchor: "fit, the gradients of several smaller *micro-batches* can be accumulated"
Gradient accumulation, its loss scaling, clipping and batch normalization are a second topic at the end of a subsection, implemented elsewhere (:numref:`subsec_mp-accumulation`). Keep one sentence ("when a batch does not fit in memory, its gradient can be accumulated over smaller pieces, :numref:`subsec_mp-accumulation`") and move Exercise 8, which is good, with the rest.

**SGD-13** (major, pedagogy; experiment). Under "Batch Size and Wall-Clock Time".
Anchor: "100 is among the fastest in both views. Gradient descent even pulls ahead"
The race does not show its conclusion. In the stored plots (both tabs) gradient descent reaches the lowest loss of the four runs (0.242) in about 0.07 seconds; the batch of 100 needs about twice as long to reach 0.243 to 0.245. On the clock, the winner is the method the section opened by calling 5,000 times too expensive, and the prose then argues from a dataset the experiment does not have ("With millions of examples ..."). Three further weaknesses: the model has six parameters and a well-conditioned quadratic loss, so ten gradient steps solve it; the first evaluation comes after 200 examples, by which time the loss has fallen from about 0.5 (half the variance of the standardized target) to 0.30, so most of the descent is off the plot; and the vertical range 0.24 to 0.30 shows mainly the noise ball. What the experiment does show well is the left panel (per example, the three stochastic runs coincide and gradient descent is ten times behind) and the hundredfold cost of single-example steps. Prose fix now: claim those two things, and state the wall-clock result for gradient descent as it is, without "even" and "only because". Deferred fix: a dataset large enough that a full pass costs many minibatch steps (all 1,503 airfoil rows are too few; a few hundred thousand synthetic rows of the same regression would do and still run in seconds), or drop gradient descent from the clock panel.

**SGD-14** (minor, correctness of a reading). Under "Batch Size and Wall-Clock Time".
Anchor: "loss. On these data $L \approx 2.1$, so the limit is about $0.95$, and we"
No cell computes $L$; Exercise 9 asks the reader to "confirm the value quoted for the race". Two lines in the cell (`np.linalg.eigvalsh` of the $6 \times 6$ Hessian) would print it. I could not verify 2.1: the data file is not on this machine. The symbol $L$ is the fourth name for the largest curvature in three sections (see LAND-18 in the `optimization-intro.md` review).

**SGD-15** (moderate, language). End of "Concise Implementation".
Anchor: "We can now predict where constant-step SGD stalls and how a minibatch"
The "We can now" recap that the author has asked to remove, followed by a three-question roadmap, followed by the Summary. The third question ("how can SGD move faster along the flat directions ... where gradient descent is slow and the noise ball is widest?") is a good last sentence for the body. Keep that one, in the place where the race ends, and delete the rest.

**SGD-16** (moderate, thought). Under "The Training Function" and "Concise Implementation".
Anchor: "To record training time we use a small timer that accumulates intervals."
About 190 lines of code per tab (`Timer`, `record_run`, `train_with_rule`, `train_with_optimizer`) and two pairs of tab notes exist to serve later sections. The reader has used optimizer objects since chapter 2, so "Every framework ships these update rules as optimizer objects" teaches nothing here, and the one fact in "Concise Implementation" worth keeping is the remark about the factor $\frac12$ and the meaning of a learning rate, which Basics also made. `Timer` duplicates `d2l.Benchmark`, which the same section uses 100 lines earlier; two timing utilities in one section need a reason or a merge. Suggested shape: keep `sgd` and one training function in the text, with comments in place of the narration; move `Timer` and `train_with_optimizer` to the section that first needs them or to :numref:`sec_utils`; delete the "Concise Implementation" heading and its confirming run (stored loss 0.244 against 0.244, a check of the framework).

**SGD-17** (minor, language). Tab notes under "Gradient Variance and Batch Size" and "The Training Function".
Anchor: "which absorbs one-time costs such as loading the GPU's math libraries."
Three pairs of tab notes describe plumbing (how `DataLoader` gathers a batch, warm-up passes, asynchronous returns, immutability of JAX arrays). The guide (section 9) puts these in code comments; most are already there as comments (`# Warm-up, then clear the gradients`, `# Wait for the queued steps`). Delete the notes.

**SGD-18** (minor, explanation). Notation.
Anchor: "$n$ training examples, a parameter vector $\mathbf{x}$, and the loss"
In this section $\mathbf{x}$ is the parameter vector, `X` in the code is the data, and Exercise 9 writes the features as $\tilde{\mathbf{x}}_i$. The chapter's convention (parameters are $\mathbf{x}$) was announced two sections ago; a clause here, and another symbol for the features in Exercise 9, would prevent the collision.

**SGD-19** (minor, pedagogy). Imports.
Anchor: "A race of full-batch, single-example, and minibatch updates against the"
The imports cells follow this roadmap paragraph. The JAX tab imports `random`, which no cell uses, in a second imports cell (`time` could join the first cell in both tabs).

## Code and experiments

- `sgd-stochastic-gradient-updates-3`: shows the cloud; the stored PyTorch plot matches the description (longer along $x_1$). Good.
- `sgd-the-noise-ball`: a confirming cell that also shows the limit of the approximation. Keep. Outputs match in both tabs.
- `sgd-gradient-variance-and-batch-size-1` and `sgd-gradient-variance-and-batch-size`: a check that cannot fail (SGD-7). In the stored PyTorch plot the measured point at $b = 1$ lies visibly below the line, the sampling error of 200 draws from a heavy-tailed population; the prose's "within the sampling error" covers it.
- `sgd-batched-computation-2`: does not test the argument that precedes it (SGD-10); the JAX output contradicts the shared prose about a level part (SGD-11).
- `minibatch-sgd-implementation-from-scratch-3` and `-7`: stored losses 0.242, 0.248, 0.244, 0.243 (PyTorch) and 0.242, 0.244, 0.244, 0.245 (JAX) support "about the same loss". The differences are within the noise ball, and the prose rightly builds nothing on them. The wall-clock panel is won by gradient descent (SGD-13). $L \approx 2.1$ is printed nowhere (SGD-14).
- `minibatch-sgd-concise-implementation-2`: confirms that `torch.optim.SGD` is SGD. Cut (SGD-16).
- `get_airfoil` (`minibatch-sgd-reading-the-dataset`): needed by later sections; fine, but its PyTorch tab note belongs in a comment.
- The PyTorch and JAX tabs differ in eleven cells here, and three tab notes per tab interrupt the prose. This is the only one of the three sections where the two tabs fragment the reading.

## Exercises

Nine problems, most of them good, and better aligned with Part A than the body's second half is.

- Keep: 1 (noise as a random dataset; I get variances 25 and 1/16 for $\mathbf{w}$, a nice surprise), 2 (reshuffling), 3 (per-coordinate rates, which sets up Adam), 4 (variance after training; becomes central if SGD-4 is adopted), 5 (duplicated data), 9 (batch size and learning rate together; the $\eta \propto b$ rule at $b = 1000$ gives $\eta = 5$ and diverges against the limit 0.95, a clean answer).
- Move with their topics to :numref:`chap_performance` if Part B shrinks: 6 (where the time per example levels off), 7 (memory traffic of matrix multiplication, which is not about SGD at all), 8 (unequal micro-batches; I get an equivalent batch of about 47 instead of 100, a good exercise wherever accumulation is taught).
- Exercise 6 part 2 gives the intensity $b/(2 + 4b/d)$, which I confirm.
- Add: one problem on the rate of SGD with $\eta_t = c/t$ on one coordinate of the valley (for which $c$ does the error fall like $1/t$?), to fill the gap of structural finding 7.

## Still open from October

"The $1/b$ variance law (four times in ch. 9)": still stated in `optimization-intro.md`, derived here, measured here, and used again in the noise-ball paragraph; with the Basics passage it is taught twice in full. "Cut: cache tutorial (to ch. 13)": mostly done, but the arithmetic-intensity paragraph, the cache remark and Exercise 7 are what is left of it.

## For the chapter review

- Order: correct, after `gd.md` and before `momentum.md`. The closing question about flat directions hands over to momentum well once the recap around it is removed.
- Overlap with Basics: unbiasedness, $1/b$ and the noise ball with its constant are in :numref:`subsec_linear-regression-minibatch-sgd`. This section should cite and extend, not restart.
- Overlap with the appendix (:numref:`sec_mdl-stochastic-gradient-methods`): the noise-ball recursion and formula appear in both at about the same length (the appendix adds the transient and the loss floor). That is a small result and may be duplicated under the author's rule. The finite-population factor is derived only here and cited from the appendix, which is the wrong way round (SGD-8). Robbins and Monro, rates and variance reduction are in the appendix only, correctly. The main text reads without the appendix.
- Overlap with chapter 13: arithmetic intensity, cache, `d2l.Benchmark`, gradient accumulation. All four are owned there.
- Sets up: `get_airfoil`, `train_with_rule`, `train_with_optimizer`, `Timer` for `momentum.md` and later; the noise ball for `lr-scheduler.md` and `batch-size.md` (which cite it often and correctly).
- Suspected chapter pattern: helper code for later sections is narrated in prose and tab notes in the section that happens to come first; small-data races (airfoil, six parameters) cannot show wall-clock claims and are then read with "only because".
