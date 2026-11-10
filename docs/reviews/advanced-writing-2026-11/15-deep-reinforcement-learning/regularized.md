# KL-Regularized Policy Optimization

Source: `chapter_deep-reinforcement-learning/regularized.md` (lines 1 to 589; about 4,800 words of prose, 1,800 words of exercises, 14 cells). Stored outputs read: PyTorch and JAX (identical; every cell is NumPy). Figures rendered: `mdl-rl-reward-gridworld.svg`; the tilting figure was checked against the printed optima.

## Verdict

This is a well-built section with one memorable experiment at its centre: fit a reward to comparisons on a gridworld, plan against it, and watch the plan walk through the hazard cells; then sweep the KL penalty and see the fitted return rise forever while the true return peaks and falls. The mathematics is correct throughout (I recomputed the reference probabilities, the return of the exploiting route, the closed form and its value, the best-of-N divergence, the forward-KL optimum and the monotonicity argument). The one-decision proof by completing the divergence is elegant, and the two-step example that shows why the tilt must use the soft value (1.44, not the best reward 3) is exactly the small concrete case an abstraction needs. Its faults are of proportion and connection. Four cells do nothing but confirm what was just proved, one of them prints `True`. The middle of the section lists consequences at equal weight. The section never names the two ideas a reader will meet under other names everywhere else: the iteration against a moving reference is mirror descent (the natural policy gradient), and the same exponentiated-gradient update appears separately as a "numerical check" without the two being connected outside an exercise. And for a section titled policy optimization in a chapter on deep reinforcement learning, nothing is ever optimized from samples. The single most important fix: replace the confirming cells by the connection they are hiding (the iterative solver is the trust-region iteration with a fixed-reference term), and say in one paragraph how the penalty is optimized with samples.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought and focus | 4 |
| Pedagogy | 4 |
| Language | 4 |
| Ease and enjoyment | 4 |

Four across the board is an honest reading, not a default: there is no passage where a reader struggles and none (apart from the true-return curve) that I would hold up as a model page. Fixing the confirming cells and the missing connections would move pedagogy and enjoyment to 5.

## What works

- **The opening** gives a problem the reader feels: "Nobody can write a formula that scores a helpful answer, a safe lane change or a readable proof." Then the consequence in one sentence: the model "guesses elsewhere, while an optimizer searches everywhere". Keep.
- **The exploit is concrete.** "The plan on $\hat r$ walks $0 \to 1 \to 5 \to 6 \to 7 \to 11 \to 15$, straight through both hazard cells" with returns of minus 2.11 against plus 0.59 (correct). And the judgment that follows is exact: "Value iteration optimized the reward it was given; the failure lies in the reward outside the reference's data."
- **The proof of the proposition**: substitute the reward in terms of the claimed optimum, and the objective becomes the log-partition value minus a divergence. Three lines, and it gives uniqueness and the value at once.
- **"One Step before the End"**: the soft value 1.44 against the best reward 3, the two resulting policies (0.43 and 0.62 probability of entering) and the loss of 0.15. This is the section's best teaching: it answers the question "which reward should the decision be tilted by" before the general equations appear.
- **"True Return against the KL Budget"**: every point an exact optimum, the monotonicity of the fitted return proved in four lines, the true return rising from minus 0.12 to plus 0.53 at 3.7 nats and falling to minus 1.83. Then the consequence a practitioner needs: "The peak of the true return is invisible to the optimizer". This is the page to preserve at all costs.
- **Trust region against penalty**: "A trust region shapes the path but not the destination" with the closed-form iterate that proves it. It closes the loop opened at the end of the PPO section.
- **Shaping invariance of comparisons** is a real observation, connected back to a proposition the reader has, and it explains a number (the rms error of 0.19) that would otherwise look like a bad fit.

## Structural findings

1. **Confirming cells.** Four cells only confirm: `regularized-the-proposition-2` (the iterative solver agrees with the closed form "to floating-point resolution"), `regularized-four-consequences-1` (the closed form at two extreme coefficients), `regularized-four-consequences-2` (prints `True` for an identity that is the same formula with a uniform reference), and the residual line of the sweep ("largest soft Bellman residual: 0e+00", a check that cannot fail after 400 contraction sweeps). The author's rule allows this once or twice. Direction: keep `regularized-the-proposition-1` (the ladder of coefficients with reward and divergence, which teaches the trade), and make the solver earn its place by what it is, see finding 2. Delete `four-consequences-2`; state the two limits of `four-consequences-1` in a sentence; drop the residual line or keep it as a comment in code.
2. **Mirror descent is in the section twice and named in neither place.** `solve_kl` is introduced as "an iterative method that knows nothing of the closed form" and called exponentiated gradient ascent. Three subsections later the idealized trust-region iteration multiplies the policy by the exponentiated reward at every step. These are the same algorithm (exercise 8b says so), and it has a name the reader will meet constantly: mirror descent with the KL divergence as its distance, which for policies is the natural policy gradient that TRPO and PPO approximate. Direction: introduce `solve_kl` in "Trust Region versus Penalty" as the iteration that takes a KL-limited step from the previous iterate, show that with the fixed-reference term it converges to the closed form and without it to the argmax (two short runs replace the confirming cell), and name it. The optimization chapter's further reading and the appendix (`chapter_mdl-optimization/further-reading.md`, "Mirror Descent") can be cited for more; the three sentences needed here are specific to this chapter and should live here.
3. **No sample-based method.** Everything is exact and tabular, which is why the central experiment is so clean. But the section title promises policy optimization, and the reader of the PPO section will ask how the penalty enters a real update. The answer is one line: since the divergence is an expectation under the policy, the penalty is a per-step reward term, the reward minus the coefficient times the log ratio of the policy to the reference at the sampled action, and any policy-gradient method of the last two sections can then be used unchanged. The last section of the chapter says this ("can be folded into the reward"); it should be said here, where the objective is defined, in the box on the two uses of KL. Consider retitling the section to what it teaches, for example "Learned Rewards and KL Regularization".
4. **"Four Consequences of the Closed Form" is a list at equal weight.** The four bold run-in items differ greatly in importance. The maximum-entropy case and the interpolation matter for the rest of the chapter; the point mass is the limit of the interpolation and needs no item of its own; "The optimum is a posterior" is a pointer to a literature. The first item also drifts into the entropy of the PPO runs. Direction: two short paragraphs. One: the coefficient interpolates between the reference and the argmax (with the point mass as its limit). Two: a uniform reference gives the maximum-entropy objective and the Boltzmann policy. The posterior remark becomes a sentence.
5. **"Invariance to Shaping" is half a detour.** It is needed to explain why the fit is good although its rms error is five times the step cost, and it is interesting. But it takes about 450 words, a least-squares cell and a paragraph on language models and GRPO to reach "up to shaping, the fit is good where the data are". Direction: keep the argument and the cell; cut the last sentence of the language-model paragraph (the factor of (K-1)/K is "as that section showed", so it is repetition); move the parenthesis about truncated trajectories to exercise 10b, which asks exactly that.
6. **The closing paragraph does four jobs.** After the practical lesson (choose the coefficient by an evaluation the optimizer does not see) it points to soft actor-critic, to the last section, and to DQN, three forward references in five sentences, one of which ("by the identity in the proof of the proposition, the two agree when the policy is the tilted one") is a technical remark about another section's critic. Keep the practical lesson and one bridge sentence; move the soft actor-critic remark to that section.

## Passage-level findings

**REG-1** (moderate, correctness and currency, fairly confident). Heading: section opening.
Quote: "This is how the language models in daily use were taught to be helpful: a reward model was fitted to human comparisons, and a policy was then trained against it"
Accurate for the 2022 generation that the second citation describes. For a book of 2026 to 2027, "the language models in daily use" claims more than can be supported: published post-training recipes since then include methods that skip the reward model (the chapter's last section teaches one) and rewards from checkers and model graders. Suggested: "This is how the first widely used chat models were taught to be helpful".

**REG-2** (minor, explanation; notation). Heading: "Preferences and the Bradley-Terry Model".
Quote: "It assigns each candidate a scalar score, here the return $r(\tau)$ of a trajectory under the reward $r$"
Chapter 14 and the PPO section write the return of a trajectory as $R(\tau)$ (it is in the shared notation table); here it becomes $r(\tau)$, the same letter as the reward function. Use $R(\tau)$, or $R_r(\tau)$ if the dependence on the reward must show.

**REG-3** (minor, explanation). Heading: "Preferences and the Bradley-Terry Model".
Quote: "with a small weight decay, $0.02$, so that the fit is well defined on pairs with little data"
The weight decay does more than that, as the next subsection shows: it selects, among rewards the comparisons cannot tell apart, the one with the smallest weights, and it is why the hazard moves sit near zero. Say so here in a clause ("and, as we will see, it decides what the fit says where the data say nothing"), since that choice is the cause of the exploit.

**REG-4** (minor, pedagogy). Heading: "Preferences and the Bradley-Terry Model".
Quote: "Their fitted rewards stay at the initial value of zero."
The section's exploit depends on an artefact of this fit: unseen pairs get reward zero, which is better than the true cost of any ordinary move after shaping. The text is honest about it ("A neural reward model would instead extrapolate according to its inductive bias, for better or worse") but the reader may conclude that the failure is a property of zero initialization. One sentence would settle it: any rule for the unseen pairs would do, as long as it does not happen to make them worse than the seen ones, and the optimizer will find whichever unseen pair the rule prices too high.

**REG-5** (minor, explanation). Heading: "Invariance to Shaping".
Quote: "Such a potential subtracts about $0.9$ from the move into the goal and about $(1 - \gamma) \times 0.8 = 0.04$ from every other move"
The arithmetic is for a constant potential, while the potential was just said to rise from 0.8 to 0.9 along the route, which adds about 0.02 back on each of the five ordinary moves of the shortest path. The printed example (an ordinary move at minus 0.08) agrees with the constant-potential figure only roughly. "About 0.04, less along the route where the potential rises" would be exact enough.

**REG-6** (minor, explanation). Heading: "Optimizing a Fitted Reward".
Quote: "Over $24$ preference datasets drawn with different seeds, the plan on $\hat r$ crosses the hazard cells in $19$ and never reaches the goal in $4$ more; one dataset yields the true optimum."
A good robustness statement of the kind the author wants kept. "Never reaches the goal in 4 more" needs three words of explanation (the plan loops, presumably on a pair whose fitted reward is positive), because a reader will wonder how an optimal plan for a reward that pays at the goal fails to get there.

**REG-7** (moderate, explanation; dependency). Heading: "The KL-Regularized Objective".
Quote: "KL divergence is the natural measure here: it is an expectation under the policy itself, so it can be estimated from the policy's own samples"
The divergence is used here, as in the PPO section, without a definition in the main text. Write it out for this objective (the sum over actions of the policy's probability times the log ratio to the reference): the proof two paragraphs later manipulates exactly that sum, and the property claimed in this sentence can only be seen from the formula. The same subsection cites Gibbs' inequality as "proved in :numref:`chap_mdl-information-theory`", a whole appendix chapter. The inequality is small (one line from $\log u \leq u - 1$); under the author's rule it should be duplicated here or at least cited by section.

**REG-8** (moderate, pedagogy). Heading: "The Closed-Form Optimum for One Decision".
Quote: "We check the proposition numerically by solving :eqref:`eq_kl_objective` with an iterative method that knows nothing of the closed form."
The proposition has a three-line proof; the check adds nothing to the reader's belief in it. See structural findings 1 and 2: keep the solver, change its job.

**REG-9** (minor, language). Heading: "The Closed-Form Optimum for One Decision".
Quote: "The iterates agree with the closed form to floating-point resolution at every $\beta$, and the objective matches $\beta \log Z$ to six decimal places."
A sentence that reports that a theorem is true. Goes with REG-8.

**REG-10** (minor, thought). Heading: "Four Consequences of the Closed Form".
Quote: "The entropy of the PPO policy in :numref:`sec_ppo` fell during training, which is consistent with movement toward the near-deterministic optimum of PPO's lightly entropy-regularized objective."
"Consistent with" marks a remark that the argument does not need. The PPO section already made the point in its closing paragraph. Cut.

**REG-11** (minor, correctness, confident). Heading: "Four Consequences of the Closed Form".
Quote: "The Boltzmann exploration rule of :numref:`sec_exploration`, with $\beta$ as its temperature, is therefore the exact optimum of a one-step objective with an entropy bonus."
Correct, and a nice closing of a loop. Note for the notation table: the symbol is listed there as "weight of the KL penalty, and soft actor-critic's temperature", with no mention that the exploration section already used it for the softmax temperature. The usage is consistent; the table should say so.

**REG-12** (minor, thought). Heading: "Four Consequences of the Closed Form".
Quote: "Control-as-inference methods build on this identity, reading rewards as log-likelihoods of an event called optimality"
A pointer to a literature under a bold heading of the same weight as the maximum-entropy result. Reduce to one sentence without a heading (structural finding 4).

**REG-13** (minor, correctness, unverified). Heading: "The One-Decision Frontier".
Quote: "its divergence from the reference is at most $\log N - (N-1)/N$, about $1.8$ nats for $N = 16$"
The formula and the value (1.835) are right, and "at most" is right for a discrete reward with ties. I believe the cited paper uses this expression for best-of-N; I recall it crediting the derivation to others, so check whether the citation should be to the derivation or to the use.

**REG-14** (moderate, explanation). Heading: "Trust Region versus Penalty".
Quote: "By the proposition each step multiplies the policy by $e^{r/\tilde{\beta}}$, so after $k$ steps"
This is the place to name the iteration (structural finding 2). Suggested addition after the display: "This iteration, a step that maximizes the reward minus a divergence from the previous iterate, is mirror descent with the KL divergence as its measure of distance; applied to policies it is the natural policy gradient, and TRPO and PPO are approximations of it for networks. `solve_kl` above is the same iteration with the fixed-reference term added, which is why it converges to the tilted reference and not to the argmax."

**REG-15** (minor, explanation). Heading: "The Direction of the Divergence".
Quote: "The penalty $D_{\textrm{KL}}(\pi \Vert \pi_{\textrm{ref}})$ is the *reverse* divergence in the taxonomy of :numref:`sec_mdl-fwd-vs-rev-kl`"
The appendix reference is correct (that subsection holds the forward and reverse cases). But "forward" and "reverse" there are defined for fitting a model to a target; here there is no target distribution, so which direction counts as reverse is a convention the reader cannot reconstruct without the appendix. Define it in place: "reverse, meaning that the expectation is taken under the distribution being optimized". Note also that the PPO section's trust region uses the other order of arguments (old policy first); a clause would prevent confusion.

**REG-16** (moderate, explanation). Heading: "Soft Value Iteration".
Quote: "Applying the proposition at every state, with $Q_\beta(s, \cdot)$ in the role of the reward,"
The two-step example makes this plausible; it is still asserted for the general case. What is missing is one sentence of argument: the objective is a sum over steps of reward minus penalty, so the choice at a state affects only that state's term and the worth of the successor, and the best choice at each state given optimal continuation is the one-decision problem with the action's worth as the reward (backward induction for a finite horizon; the contraction argument of the next paragraph for the discounted case). As written, the contraction is proved but the claim that the fixed point is the optimum of the penalized objective is not stated as needing proof.

**REG-17** (minor, correctness, confident). Heading: "Soft Value Iteration".
Quote: "With a uniform reference this is the soft value iteration of maximum-entropy reinforcement learning."
Up to a constant: with a uniform reference over $|\mathcal{A}|$ actions the value here is the usual soft value minus $\beta \log |\mathcal{A}|$ per step (discounted), because the divergence from uniform is $\log |\mathcal{A}|$ minus the entropy. The policies agree. Add "up to a constant shift of the values", since the soft actor-critic section will write the entropy form and a reader comparing the two will find they differ.

**REG-18** (minor, thought). Heading: "Soft Value Iteration".
Quote: "the log-sum-exp, being convex, is biased upward too, only less sharply"
A side remark about estimation noise in a subsection where every quantity is exact. True (Jensen) but unsupported as to "less sharply" and unused afterwards. Cut, or move to the DQN section where maximization bias is the subject.

**REG-19** (minor, correctness of statement). Heading: "True Return against the KL Budget".
Quote: "the largest Bellman residual is below $10^{-10}$"
The cell prints "largest soft Bellman residual: 0e+00". Not a contradiction, but the prose should read what is printed, or better, the line should go (structural finding 1).

**REG-20** (minor, correctness, unverified). Heading: "True Return against the KL Budget".
Quote: "measured the same rise and fall for language models optimized against a learned reward model and graded by a held-out one"
As I recall that study, the grader is a larger "gold" reward model that also supplies the labels on which the proxy reward models are trained, so "held-out" is not quite the relationship; "graded by the larger model that had labelled the proxy's training data" is closer. Check against the paper.

**REG-21** (minor, language). Heading: "The KL-Regularized Objective".
Quote: "The coefficient $\beta$ is the weight of the penalty; the reference stays fixed."
Repeats the sentence before it. Cut.

## Code and experiments

- Confirm-only cells: `regularized-the-proposition-2`, `regularized-four-consequences-1`, `regularized-four-consequences-2` (prints `True`), and the residual print in `regularized-the-measurement-true-return-against-the-kl-budget` (structural finding 1).
- All printed numbers quoted in the prose match the stored outputs (0.193 and 0.032 rms; fitted and true returns of the two plans; the ladder of coefficients; 1.44, 0.43, 0.62, 1.740 against 1.592; peak plus 0.53 at 3.7 nats; minus 1.83 at 14.2 nats). The one mismatch is REG-19.
- Numbers no cell prints: "The one trajectory in ten that is cut off after forty steps"; the two statements over 24 preference datasets. The second kind is the deliberate robustness statement; the first should be printed or dropped.
- The central experiment is at the right scale and shows the phenomenon without ambiguity; because it is exact there is no seed noise in the sweep itself, and the dependence on the sampled comparisons is reported over 24 datasets. This is the model for how the chapter should support a claim.
- The imports cell loads `torch`, `nn`, `F` (and the JAX stack) although every cell is NumPy and `d2l`. The two tabs therefore differ only in unused imports. Either say so in one line or trim the imports, so that a reader does not look for the network.
- Cell IDs (`regularized-four-consequences-*`, `regularized-preferences-and-bradley-terry-1` for the gridworld construction) no longer match their headings; harmless.
- No TensorFlow or MXNet material.

## Exercises

Ten exercises, about 1,800 words, 38 percent of the prose. A strong and well-graded set. Exercise 1 (identifiability, annotator consistency and the effective coefficient) teaches something the text does not and matters in practice. Exercise 7 (slope of the frontier, the constrained form, best-of-N) and exercise 9 (the forward penalty) complete claims the text states without proof. Exercise 8 holds the mirror-descent connection that belongs in the text (structural finding 2); if it moves, part b can ask for the convergence rate instead. Exercise 3 (tilting by the unregularized value) repeats the lesson of "One Step before the End" on the gridworld and could be cut if space is needed. Exercises 3 and 6 have no labels; harmless. Missing: an exercise that estimates the penalized objective and its gradient from samples of the policy (see structural finding 3), which would connect the section to the two before it.

## Still open from October

The October review marked `regularized.md:137` as a passage to keep; the planning experiment at that location is intact. Nothing else from October applies.

## For the chapter review

- Position: follows PPO naturally (path against destination) and is the only section of the chapter with no learning from samples and no network. That is a strength for the experiment and a question for the chapter's arc, since the next section (DQN) returns to CartPole and the thread of this one is picked up only in the last section and in soft actor-critic. Consider whether it should sit directly before the last section, with which it forms a pair (Bradley-Terry and the closed form here; direct preference optimization there). If it stays, its closing paragraph needs only one bridge.
- Appendix use: one reference to a chapter (Gibbs) and one to a subsection (direction of the divergence); both targets exist and hold the cited content. Neither the definition of the divergence nor Gibbs' inequality is given in the main text; both are small and should be.
- Notation against chapter 14: $r(\tau)$ here against $R(\tau)$ there (REG-2); the coefficient's earlier use as the softmax temperature is consistent but missing from the notation table; `F` is the shaping matrix in a cell, the shaping function in exercise 10 and `torch.nn.functional` in the imports.
- Forward references to the Language Models part: none that carry weight. The claims about language-model practice (REG-1, the box on the two uses of KL) are the ones that will date fastest.
- Pattern for the chapter: this section supports its central claim by exact computation plus a count over 24 resampled datasets. The two sections before it support theirs by a handful of seeded training runs read generously. The contrast is worth pointing out in the chapter review as the standard to aim for.
