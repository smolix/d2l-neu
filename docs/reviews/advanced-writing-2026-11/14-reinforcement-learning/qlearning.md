# Temporal-Difference Learning

Source: `chapter_reinforcement-learning/qlearning.md` (lines 1 to 593, up to the slides marker). About 5,100 words of body prose plus 1,700 words of exercises; 12 code cells (about 150 lines, plain NumPy, identical in both tabs); no figure of its own (it reuses panel (d) of `fig_rl_backups`).

## Verdict

This is good textbook writing, and its first third is at the bar. The path from the Bellman operator to one sampled transition, the reason control learns $Q$ and not $V$, and the hand-computed transitions at cells 14 and 13 form an explanation I would be glad to find in MacKay: one idea, one small case, every number checkable. The mathematics is right throughout. I recomputed every lake number the prose quotes (0.180, 73%, 54%, the tie at cell 6, the one-point gain of the undiscounted policy) and re-ran the step-size experiment in NumPy; nothing disagrees. The section then loses some of that quality in three ways. It puts its hardest passage (least squares, the variance decomposition and double sampling) before the reader has run anything. Its results paragraphs lean on frequencies and anecdotes that no cell prints, and in one place on a printed output that contradicts the general claim. And the on-policy and off-policy distinction, which the whole of the next chapter is organized around, gets two hundred words and no number. The single most important fix is the second: make the results paragraphs report what the cells show, by printing more seeds where the prose needs a frequency and cutting the anecdotes where it does not.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought and focus | 4 |
| Pedagogy | 4 |
| Language | 4 |
| Ease and enjoyment | 3 |

("The Sampled Backup" up to "One Transition by Hand" is a 5 on the first three lines. The results paragraphs of "Q-Learning on the Slippery Lake" are a 3 for enjoyment: dense with numbers, several of them unsupported.)

## What works

- **The opening question.** "Each step shows it one successor, chosen by the environment, never the list of possible successors with their probabilities. Can the agent still learn how good each command is, and from that the best policy, from such single draws?" The reader knows what was taken away and what is asked.
- **"The model enters the backup in one place, the average over the successor $s'$."** The whole method follows from this sentence, and the text says so. Protect it.
- **Why $Q$ and not $V$**: "One transition shows what one action did and says nothing about the brackets of the other actions, so it cannot estimate their maximum." Sutton and Barto never state this as plainly.
- **"One Transition by Hand."** Three updates with numbers (0.5, 0.475, back to 0.25), ending in the contrast "Value thus flows backward from the goal one cell per visit ... Value iteration moved it one cell per sweep, along all transitions at once." It also plants the question of step sizes ("The entry jumps between targets of $1$ and $0$").
- **The honest proposition.** The convergence result is stated with its hypotheses (finite sets, bounded rewards, $\gamma < 1$, every pair updated infinitely often, Robbins-Monro steps per pair), cited to the three papers that prove it, and followed by a candid account of what the simple argument does not give: "Averaging alone does not prove this, because the targets are computed from the table being learned and have no fixed mean." Cited, with the idea of the proof named. That is the right level.
- **Finite budgets.** "The conditions guarantee the limit; they do not choose a schedule for a finite budget", backed by an experiment in which a schedule that satisfies the conditions ends at a sixth of the true value. The Even-Dar and Mansour result is stated correctly (exponential in $1/(1-\gamma)$ for $1/n$, polynomial for $n^{-\omega}$). The weight $k^9$ checks: for $\alpha_n = 1/(1 + cn)$ the $k$-th target has weight proportional to $k^{1/c - 1}$.
- **Grading against the exact solution.** Because `Q_star` is known, the section can say "in no cell does a learned greedy action lose value under $Q^*$" and separate the training curve (the behavior policy) from the table. The 54% line ("The plateau of the training curve near $0.5$ is this cost of exploration") is an insight most treatments miss.
- **Maximization bias from an experiment already run.** The traces of the step-size cell are reread for the overshoot, with the argument that noise-free backups from zero can only approach $Q^*$ from below. I replicated this with a NumPy re-implementation of the lake (16 seeds): the mean trace of $Q(s_0, \textrm{left})$ is about 0.26 at episode 750 against 0.180, the peaks run from 0.22 to 0.43, and a control that bootstraps from a fixed action instead of the maximum never overshoots. The reading is sound.
- **Termination against truncation** carried into the update, with the one-factor code form.
- **The semi-gradient objection** is asked in the reader's voice ("A reader who has fit regression models will object that the natural way to fit a table to targets is least squares") and the three-state example gives the direction of the distortion without algebra. Good material in the wrong place (TD-2).

## Structural findings

1. **Four topics, one line of argument, acceptable length.** Sampled backup and convergence; the lake experiment; off-policy and maximization bias; planning in a counted model. At 5,100 words this is the longest of the tabular sections, but each part answers the opening question and the fourth earns its place by the contrast it draws ("Q-learning uses each transition once"). I would not split. If length must come down, the candidates are in TD-2 (shorten double sampling by a third), TD-9 and TD-10 (two asides), about 400 words together.

2. **The difficulty ramp has a bump.** The order is: update, hand example, least squares and double sampling, step sizes, convergence, terminal mask, then code. "Least Squares and Double Sampling" is the hardest passage in the section (a population objective, a conditional variance, a counterexample, residual gradients) and it arrives before the reader has seen the method work. Its payoff, the target network, is a chapter away. See TD-2.

3. **The title promises more than the body gives, and the comparison with Monte Carlo is asserted.** The section is called "Temporal-Difference Learning", but TD(0) for $V^\pi$ appears only inside exercise 1, and the bias and variance claim against Monte Carlo ("usually far less variable") is one sentence with no number. Sutton and Barto spend a chapter on exactly this before they reach control. For this book going straight to Q-learning is the right call, but the state-value update should be displayed once (later sections and the next chapter use it), and the text should say that it is skipping prediction. See TD-1.

4. **On-policy against off-policy is too thin for what depends on it.** The definition is correct and clear. But the next chapter divides its methods by this property (PPO against DQN, SAC and offline learning), and here it gets one paragraph, a claim about SARSA that is not computed, and the experiment in an exercise. The exact number is available in one line of code. See TD-7.

5. **Results paragraphs report more than the cells show** (TD-4, TD-5, TD-6). This is the pattern to fix; it recurs in the other sections of the chapter in milder form.

6. **Two endings.** A recap paragraph precedes the Summary (TD-11).

7. **Looked for and not found.** (a) A sentence on what replaces the table when states are many, beyond "the deep Q-learning of the next chapter": fine as is. (b) Expected SARSA and $n$-step targets: rightly left out. (c) The rate of convergence in samples for a sensible schedule: the index promises that Agarwal et al. "supplies the sample-complexity rates this chapter states qualitatively", and the section does state them qualitatively. Nothing essential is missing.

## Passage-level findings

**TD-1** (moderate, pedagogy). Under "From the Bellman Operator to One Sample".
Anchor: "Bootstrapping is a choice. *Monte Carlo* estimation waits until the episode ends"
This paragraph is the section's whole treatment of temporal-difference prediction. The claim "It is biased while the table is inaccurate and usually far less variable" is asserted on two citations; nothing in the section shows it, and the state-value update the title refers to is never displayed (exercise 1 has to define it). Add the display $V(s) \leftarrow V(s) + \alpha\,(r + \gamma V(s') - V(s))$ here as "the same step for the value of a fixed policy, called TD(0)", say in a clause that the section goes straight to control, and give the variance claim one concrete anchor from the lake: a Monte Carlo target for $Q(s_0, \cdot)$ is 0 or about $\gamma^k$ with success probability under 0.2, a one-step target is $\gamma$ times a table entry. No new cell is needed.

**TD-2** (moderate, pedagogy). Under "Least Squares and Double Sampling".
Anchor: "A reader who has fit regression models will object that the natural way"
The first paragraph of this subsection (the expected update is a damped value-iteration step) belongs exactly here and should stay; it answers "Why should such noisy steps lead anywhere?" The four paragraphs that follow answer a different question (why not least squares) and are harder than anything before the code. A first-time reader meets equation :eqref:`eq_double_sampling`, residual gradients and Baird before knowing that the method works. Move them to their own subsection after "Step Sizes and Convergence" or, better, after the lake experiment, next to "Off-Policy Learning", where "holding the target fixed" can lead straight into the pointer to the target network. The proposition's commentary already refers back to it ("Residual-gradient descent also has an expected step that vanishes at a fixed point, the wrong one"); that sentence then needs a forward pointer or moves with the rest. While moving, cut the sentence on residual-gradient methods to one clause; the reader needs the name of the problem, not the taxonomy.

**TD-3** (minor, correctness). Under "Step Sizes and Convergence".
Anchor: "which applied to all entries contracts with modulus"
The algorithm does not apply the damped backup to all entries; it applies it to the visited one. The conclusion (a constant step is safe when targets are noise-free) is true, but by the asynchronous version of the argument: an entry updated with a noise-free target moves to within $(1-\alpha) + \alpha\gamma$ of the current largest error, so the largest error cannot grow and shrinks once every pair has been updated. Say "applied to one entry at a time it never increases the largest error, and it shrinks it by this factor once every pair has been visited". High confidence.

**TD-4** (moderate, correctness of reading). Under "The Learned Table and the Exact Solution".
Anchor: "In about one run in eight some greedy action is still suboptimal after"
The cell prints five seeds, all optimal. The paragraph then reports a failure rate ("about one run in eight"), a range for those runs ("success $71\%$ to $74\%$") and a worst case ("In the rare worst case the agent still prefers *right* at the start and succeeds less than half the time"), and explains the worst case by a mechanism. None of this is on the page, and two later passages build on it ("as the run above showed", "the low run above", see TD-8). The author's rule allows ranges that record repeated runs; a narrated failure the reader cannot see is something else. Either raise `num_seeds` for the grading (the tables are cheap; 20 seeds would show two or three imperfect ones and the cell could print how many rows of the third line are nonzero), or cut to one sentence: "Five runs do not make a guarantee: over more seeds, about one table in eight still has a suboptimal greedy action at this budget, usually at a cost of a point or two." The mechanism (a low estimate is corrected slowly) is taught again, properly, in "Feedback between Estimation and Data Collection" and can be dropped here.

**TD-5** (moderate, language). Under "The Learned Table and the Exact Solution".
Anchor: "The cell prints each table's value error and action-value error, the largest value"
One sentence lists five quantities and the cell computes them with nine one-line definitions, among them `taxed` and a variable called `regret` that means "what a greedy action gives up under $Q^*$". The next section defines regret as something else. The reader has to hold `lake1`, `success`, `live`, `greedy`, `optimal`, `taxed`, `v_err`, `q_err`, `regret` and `wins` before seeing one number. Split the cell at the question each half answers: first "is the greedy policy optimal?" (the third and fourth printed lines, with $\pi^*$ and $\pi^*$ at $\epsilon = 0.05$), then "how accurate are the values?" (the first two lines). Rename `regret` to `shortfall` and `taxed` to `pi_star_eps`; "tax" is one of the economic metaphors the guide removes (the cell ID in the next section still carries "the-linear-tax").

**TD-6** (moderate, correctness of reading). Under "Planning in a Learned Model".
Anchor: "but not always, and the five runs printed here happen to favor Q-learning"
The printed output at 500 episodes shows Q-learning with the smaller value error (0.084 against 0.096), and the prose says the opposite is usual and that the print is unlucky. The guide (section 9) is explicit: do not keep an output and explain what it fails to show. The same paragraph adds three unprinted frequencies ("in about nine runs out of ten", "better in nearly every seed", "about as accurate as Q-learning's tables after twice as many episodes"). Fix on the development machine: run this cell over 20 seeds and print medians with the number of seeds on which the model wins, so that the sentence can read the output. If the value error at 500 episodes then still does not favor the model, drop that claim; the two results that matter (the planned policy is better with little data, the planned values are about twice as accurate with more) are both in the print already.

**TD-7** (moderate, pedagogy). Under "The Off-Policy Target and SARSA".
Anchor: "would value the cells next to holes below what $\pi^*$ achieves there, because its own"
Three things. (a) The claim can be made exact at no cost: the cell above already builds $\pi^*$ at $\epsilon = 0.05$, and its exact values at $\gamma = 0.95$ are 0.149 at the start against $V^*(s_0) = 0.180$, and 0.700 against 0.724 at cell 14 (my computation; lower in every cell, not only next to holes). Quote two of these numbers, or print them with `mdp.policy_value`. That is the on-policy and off-policy difference as two columns of numbers on the lake the reader knows. (b) "For a fixed behavior policy" hides that SARSA's behavior policy is $\epsilon$-greedy on its own table. Add the sentence the reader needs: with $\epsilon$ held fixed SARSA can at best find the best policy among those that must explore at rate $\epsilon$, and it reaches $Q^*$ only if $\epsilon$ goes to zero; exercise 8(c) can then cite this. (c) Say why the distinction will matter: the methods of :numref:`chap_deep_rl` are sorted by it, since only off-policy targets can reuse old data. One sentence, with the reference.

**TD-8** (minor, explanation). Under "Feedback between Estimation and Data Collection".
Anchor: "which is why the low run above took so long to change its first action"
There is no low run above. The reference is to the anecdote of TD-4, which no cell shows. After TD-4 is fixed, either point to a printed seed or state the asymmetry without the reference. The subsection itself is good and short; its second paragraph ("A table can satisfy the equations on the visited pairs and still be wrong") is the best justification of the coverage hypothesis in the proposition and could be cited from there.

**TD-9** (minor, thought). Under "The Learned Table and the Exact Solution".
Anchor: "so a policy that ignores the discount succeeds about one point more often"
True (the best stationary undiscounted policy reaches the goal 74.0% of the time within 100 steps, the best time-dependent one 74.4%), and beside the point. The reader has just been asked "Is a return near $0.5$ good?" and gets a remark about a third policy before the answer. Cut the sentence or move it to :numref:`sec_valueiter`.

**TD-10** (minor, thought). Under "The Learned Table and the Exact Solution".
Anchor: "but the two counts measure different things: a backup reads a whole row of the"
A comparison is raised (96,000 steps against 10,500 backups) and withdrawn in the same sentence. Exercise 6 does it properly. Keep "A run typically uses about $96{,}000$ environment steps" and the pointer to the exercise; drop the backups count here.

**TD-11** (minor, thought). End of "Planning in a Learned Model".
Anchor: "The agent can now learn optimal action values from samples alone and act on them"
A recap before the Summary, against the author's decision on endings. The first sentence repeats the section, and the 54% against 73% is stated for the third time (it appears twice in "The Learned Table and the Exact Solution" and again in the opening of :numref:`sec_exploration`). Keep only the forward step: "Both rest on how the agent collects its data, and acting at random one step in twenty is a crude rule. :numref:`sec_exploration` asks whether a different rule can learn as much while losing less." The clause "a greedy table needs a finite set of actions to maximize over" is a second, unrelated limitation dropped into the closing; it belongs in the next chapter's opening.

**TD-12** (minor, language). Pattern, 4 instances of a correction the reader did not need.
Anchors: "This is the typical run, not a guarantee at this budget"; "is this cost of exploration, not a defect of the table"; "The conditions concern the limit, not a finite budget"; "Transition noise, not the learning rule, sets most of the cost"
Each is individually fine; four "X, not Y" closers in one section is a mannerism (guide 10.3, reflexive "not X; it is Y"). The third also says what the best sentence of the section says 180 lines later ("The conditions guarantee the limit; they do not choose a schedule for a finite budget"). Keep the later one and rewrite the earlier as a plain lead-in: "At a finite budget the choice among such schedules matters a great deal."

**TD-13** (minor, explanation). Under "Step Sizes and Convergence".
Anchor: "Below we test $\alpha_n = 1/(1 + n)$, the sample mean with the zero initial value counted as one more target"
The paragraph introduces three schedules, two rates and a weight $k^9$ in six sentences, before the reader has a run to attach them to. The same content is retold with results under "Step-Size Schedules Compared". State here only the default and why ("it weights later targets more heavily, so the estimate follows the improving targets"), and move the comparison of $1/(1+n)$ and the Even-Dar and Mansour rates to the later subsection, where the output shows them.

**TD-14** (minor, explanation). Under "The Learning Loop".
Anchor: "Every cell in this section is plain NumPy, because a table needs only index arithmetic"
Both tabs import `torch` or `jax`, `nn`, `F`, `optax` and `math` in the first cell and use none of them (this section and :numref:`sec_exploration` never touch a tensor). The house rule is one imports cell, but an imports cell of unused libraries followed by "every cell is plain NumPy" reads oddly. Reduce the imports to `numpy`, `gymnasium` and `d2l`, in one cell for both tabs.

## Code and experiments

- `qlearning-reading-the-curve-honestly-1`: all stored numbers agree with the prose (value error 0.006 to 0.021, "within about $0.03$"; action-value error 0.075 to 0.131, "about $0.05$ to $0.17$" is wider than the print and records other runs; 0.730 and 0.537; 95,569 steps). The cell is overloaded (TD-5) and the paragraph after it reports unprinted runs (TD-4).
- `qlearning-reading-the-curve-honestly-3` (calm ice): prints 100% on 512 episodes and 3,658 steps, as stated. Note that :numref:`sec_exploration` describes this run as "$256$ episodes"; one of the two numbers is wrong (reported in the exploration file as EXP-2).
- `qlearning-reading-the-curve-honestly-4` (step sizes): stored finals 0.056 to 0.265 for the constant step, 0.180 to 0.195 for the default, 0.002 to 0.024 for $1/(1+n)$; peaks 0.205 to 0.324. All as stated. The phrase "over twelve seeds most runs overshoot in this way" is unprinted; my 16-seed replication supports it (every run peaked above 0.21).
- `qlearning-planning-in-a-learned-model-2`: the stored output at 500 episodes contradicts the general claim about value error (TD-6). The policy numbers (0.486 against 0.639) and the 2,000-episode numbers (0.023 against 0.011) support the prose.
- No cell only confirms a derived result. The two "checks" in this section (grading against `Q_star`) can fail and do carry information.
- Cell IDs `qlearning-reading-the-curve-honestly-*` carry a title the subsection no longer has; harmless, IDs are stable by rule.
- Tabs: all cells are shared between PyTorch and JAX. No TensorFlow or MXNet material.

## Exercises

Eight problems, the strongest set of the three sections. Exercise 4 (the double-sampling counterexample, with the two-successor fix in part (e)) and exercise 5 (step sizes, with the question "Which assumption of the analysis fails for this entry") are model problems: each derives something the text only stated and then tests it. Exercises 1, 3, 7 and 8 are adapted from Sutton and Barto with credit, and are the right four to borrow (batch TD against Monte Carlo, greedy SARSA, maximization bias, the cliff). Exercise 2 (truncation as termination) has a clean closed-form answer. Nothing to cut. Two notes: exercise 1 defines TD(0) because the body does not (TD-1); exercise 8(c) asks "which values does each method learn in the limit", which the body should prepare with the sentence proposed in TD-7(b).

## Still open from October

The October review said this file "carries TD learning, Robbins-Monro, bandits, regret, UCB, Thompson sampling, and maximization bias" and "defines an objective and then disowns it". The first is resolved: the bandit material is now :numref:`sec_exploration`. The second is half resolved: the least-squares objective is now introduced as an objection and answered, which is legitimate, but it still sits in the middle of the derivation (TD-2).

## For the chapter review

- The section is the hinge of the tabular half: it takes $\mathcal{T}$, $Q^*$, `TabularMDP`, `value_iteration`, `policy_evaluation` and the terminated and truncated distinction from the two sections before it, and hands `epsilon_greedy`, `linear_schedule` and `q_learning` to :numref:`sec_exploration`, and the semi-gradient and maximization-bias ideas to :numref:`sec_dqn`. All cross-references are by label.
- Against Sutton and Barto (chapter 6): this section is tighter and in three places better (why $Q$, finite-budget step sizes, exact grading and the termination mask). It is weaker on prediction, which Sutton and Barto build first and with experiments, and on on-policy control, which they run (SARSA on the windy grid and the cliff) and this section only defines. A reader of this section alone would be able to implement Q-learning correctly and would know less than a reader of Sutton and Barto about why bootstrapping helps.
- Suspected chapter pattern 1: results paragraphs that quote frequencies over runs no cell prints, and in one case explain away the printed output. Check the policy-gradient and baselines sections for the same.
- Suspected chapter pattern 2: every section closes with a recap-and-bridge paragraph before the Summary (here, in `mdp.md` "We can now write a decision problem down", and in the two sections after this one).
- Suspected chapter pattern 3: the first cell of every section imports the full framework stack whether or not the section uses it.
- No appendix dependence: the section does not cite the mathematics appendix, and does not need to. The Robbins-Monro conditions were already stated for SGD in :numref:`sec_sgd`; a clause pointing there would connect the two, but the duplication is small and allowed.
