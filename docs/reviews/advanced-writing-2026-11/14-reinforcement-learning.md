# Chapter 14, Reinforcement Learning: writing review

Sources: `chapter_reinforcement-learning/index.md` and its eight sections in book order, read by the section reviewers in full up to the slides marker; stored PyTorch and JAX outputs and the generated figures were checked against the prose. About 42,000 words of body prose and 14,500 words of exercises (71 problems). I read the eight section reviews in full, reviewed the introduction myself, and opened the sources to check the cross-section claims marked "checked" below.

Per-file reviews:
[index](14-reinforcement-learning/index.md) ·
[mdp](14-reinforcement-learning/mdp.md) ·
[value-iter](14-reinforcement-learning/value-iter.md) ·
[qlearning](14-reinforcement-learning/qlearning.md) ·
[exploration](14-reinforcement-learning/exploration.md) ·
[imitation](14-reinforcement-learning/imitation.md) ·
[policy-gradient](14-reinforcement-learning/policy-gradient.md) ·
[baselines](14-reinforcement-learning/baselines.md) ·
[deep-rl](14-reinforcement-learning/deep-rl.md)

A note on the brief: the section reviewers were told the chapter is PyTorch-only. It is not. Every section carries PyTorch and JAX tabs, `outputs/jax/chapter_reinforcement-learning/` exists, and the introduction says "The code is in PyTorch and JAX". All three reviewers noticed and reviewed what the sources contain, so no finding below depends on the wrong premise. The project's `CLAUDE.md` still says the chapter "ships PyTorch-only"; that note is stale.

## Verdict

This is a strong chapter, and in its best pages it is at the bar. It has one method that almost no other treatment of the subject has: nearly everything is done on a sixteen-cell lake and a four-prompt verifier task that can be solved exactly, so every learned quantity is graded against the truth. Out of that come the pages the reviewers would put in MacKay: the equation of cell 14 with the unknown on both sides, the stopping rule that stops eleven tolerances early, the anatomy of an optimal policy that walks away from the goal, the greedy bandit as a lottery won one run in five, the two-action "smallest case" that predicts the flat start of the training curve, the weight of $-90$ that a time limit puts on the actions that kept the pole up. The mathematics is right nearly everywhere; the reviewers recomputed several hundred numbers and found them correct. The chapter is also current: truncation against termination, group baselines, on-policy distillation, the dropped $\gamma^t$.

It is not yet at the bar as a whole, for five reasons. (1) The second half does not keep to what its cells print: results paragraphs quote frequencies from seeds no cell runs, a 200-seed study "outside the notebook", ablations "in our runs", and in two places a sentence that the stored output contradicts. (2) The chapter does not decide who owns what: the return from step $t$ is used from the first section and defined in the seventh; reward shaping is told in two sections; the zero-mean property of the score and the reward-to-go form are derived in `policy-gradient.md` and again in `baselines.md`; the hand-over to the next chapter is made three times. (3) Three sections carry material for someone else: `mdp.md` builds for later sections, `baselines.md` and `deep-rl.md` for the next chapter. (4) Four statements are wrong as written (EXP-1, BL-3, MDP-13, and the claim about what the shaping potential signals, MDP-12 and VI-15). (5) Two experiments, the lake in `imitation.md` and the coverage run in `exploration.md`, show their section's phenomenon as a minor effect or not at all, and the text says so. What matters most is (1) and (2): both are cheap, and both decide whether a careful reader trusts the rest.

Against Sutton and Barto the honest summary is this. The chapter is more exact and more rigorous than their corresponding chapters on everything an implementer needs, it reads its solutions more closely than they do, and it covers ground they do not (truncation, regret bounds, imitation, variance accounting for policy gradients, the language-model case). It is thinner than they are where a first-time reader forms concepts: one MDP against their half dozen, no definition of the return $G_t$ with its recursion, no temporal-difference prediction before control, on-policy control defined but never run, and policy iteration shown on a case where it does not iterate. A reader of this chapter alone can implement the methods correctly and can check them; a reader of Sutton and Barto knows better why bootstrapping helps and has seen more problems. The gap is closable in perhaps a thousand words (see "What is missing"), and the chapter should say in its resources list what it leaves to them.

## Scores

As given by the section reviewers; the row for the introduction is mine. 1 to 5; 5 means the page would be at home in MacKay or as an Illustrated-Transformer-quality explanation.

| File | Explanation | Thought and focus | Pedagogy | Language | Enjoyment |
|---|---|---|---|---|---|
| [index](14-reinforcement-learning/index.md) | 4 | 4 | 3 | 3 | 4 |
| [mdp](14-reinforcement-learning/mdp.md) | 4 | 3 | 4 | 4 | 3 |
| [value-iter](14-reinforcement-learning/value-iter.md) | 5 | 4 | 4 | 4 | 4 |
| [qlearning](14-reinforcement-learning/qlearning.md) | 4 | 4 | 4 | 4 | 3 |
| [exploration](14-reinforcement-learning/exploration.md) | 4 | 3 | 4 | 4 | 4 |
| [imitation](14-reinforcement-learning/imitation.md) | 4 | 3 | 3 | 4 | 3 |
| [policy-gradient](14-reinforcement-learning/policy-gradient.md) | 4 | 3 | 4 | 4 | 4 |
| [baselines](14-reinforcement-learning/baselines.md) | 4 | 3 | 3 | 4 | 3 |
| [deep-rl](14-reinforcement-learning/deep-rl.md) | 4 | 3 | 3 | 4 | 3 |

The three reviewers calibrated alike, and the table does not discriminate on two columns: explanation is 4 in seven of eight sections and language is 4 in all eight. The information is in "thought and focus" (3 in six of eight) and in the remarks under the tables, where every reviewer says the same thing: the first third or half of the section is a 4 to 5 and a later part is a 3. Reading the reviews together, the ranking is:

- **Strongest:** `value-iter.md`, without qualification; then the first third of `qlearning.md` (the sampled backup to "One Transition by Hand"), the bandit half of `exploration.md`, and the first third of `policy-gradient.md` (the opening to "The Smallest Case").
- **Weakest:** `baselines.md`, which is three sections in one and whose training study cannot be read from its own output; `imitation.md`, whose theory is the best-proved in the chapter and whose experiment shows the effect as a fifth of the loss, behind 190 lines of container code; `deep-rl.md`, which has six subjects and then turns into a hand-over.

The uniform 4 for language is fair: the prose is clean and nearly free of the guide's banned patterns everywhere. It is also even in register, which the enjoyment column records.

## Technical correctness and currency

There are findings of substance. Ordered by severity; confidence as the section reviewer gave it. Items I checked against the sources are marked.

**Errors in statements**

1. **EXP-1** (major; high confidence). The Lai and Robbins bound is asymptotic and is applied at $t = 2000$ to say an arm "must be pulled about" 24 times; the section's own figure shows that arm pulled 11 times, and the stored regret of UCB and Thompson sampling is below the bound's leading term. A reader who checks will conclude that one of the two is wrong.
2. **BL-3** (moderate; the reviewer states it without qualification and I agree). "We can therefore multiply any score in the REINFORCE estimator by such a quantity" is false as written; the identity allows adding or removing a term $c$ times a score, not multiplying the weighted score by $c$.
3. **MDP-13** (moderate; certain). The collapse of the token MDP to one step needs $\gamma = 1$; the section fixed $\gamma = 0.95$ "throughout this chapter" two pages earlier.
4. **MDP-12 and VI-15** (moderate; arithmetic certain, judgment the reviewer's). The same defect found in both sections: with $\Phi = -0.3\,\ell$ set to zero at the holes, the shaped reward for stepping into a hole is $+0.6$ to $+1.5$ against $1.3$ for the goal, so the potential the text says "gives a learner a signal long before it first reaches the goal" pays most for falling in. No cell in the chapter trains a learner with a potential, so the claim of usefulness is asserted in two sections and tested in none.
5. **VI-1** (moderate; certain). $V^\pi$ is defined through $\sum_t \gamma^t r(s_t, a_t)$ and called "the discounted return of `sec_mdp`"; the return defined there uses the realized rewards $r_t$. Same expectation, different random variable. Checked: `value-iter.md` line 42 and `mdp.md` line 124 read as the reviewer says.
6. **VI-13** (moderate; certain; filed under explanation). The "optimal" policy is judged by its probability of reaching the goal within 100 steps, for which it is not optimal (0.730 against 0.744 for the best time-dependent policy); the body does not say so, although exercise 10b asks for the second number.
7. **PG-9** (moderate; correctness of presentation). The policy gradient theorem is correct but its Proposition lists no hypotheses (finite MDP, $\gamma < 1$ or proper episodes, positive differentiable policy, dynamics independent of $\theta$).
8. **EXP-2** (moderate). `exploration.md` says the calm-ice run of the previous section used 256 episodes; it used 512. Checked: `qlearning.md` line 268 passes 512 and prints "calm ice, 512 episodes"; `exploration.md` line 522 says "$256$ episodes".
9. Minor: **MDP-3** ("nothing in this chapter loses generality" holds for expected returns only), **VI-11** (the prose warns of an `argmax` flip that the next cell assumes cannot happen; mechanism checked on a reconstruction only), **VI-16** (the Summary's "because" is too strong), **TD-3** (the damped backup is applied to one entry, not all), **EXP-4** (the variance explanation for $\kappa = 0.5$ stated as established), **PG-13** (the cited convergence theorem assumes a start distribution with full support; the lake starts at one cell), **DR-12** (curve dips attributed to interference without evidence), **DR-13** (off-policy divergence with linear features is Baird's counterexample, not the paper cited).

**Stored outputs that contradict the prose, and readings that outrun the output**

10. **BL-11** (moderate). "Centering is no faster than the plain return" has the stored direction reversed (58.5 against 63.0 updates), and an explanation is built on a gap the section's own noise paragraph calls unresolvable at twenty seeds.
11. **TD-6** (moderate). At 500 episodes the printed value error favours Q-learning (0.084 against 0.096); the prose says the opposite is usual and that "the five runs printed here happen to favor Q-learning".
12. **BL-12** (moderate). Two paragraphs quote a 200-seed study "outside the notebook" ("about 30", "one run in eight") beside a printed 100-seed table that says 35.0 and 15 of 100.
13. **DR-9** (moderate). "Every run learns to swing the pendulum up and balance it" from three seeds whose averages touch $-200$ and fall back by a hundred or more.
14. **TD-4** (moderate). The cell prints five seeds, all optimal; the paragraph reports a failure rate ("about one run in eight"), a range and a worst case with a mechanism, and two later passages refer to "the low run above" (TD-8), which is not on the page.

**Numbers no cell prints**

15. **DR-4** (the explained variance of 0.4 to 0.8 against 0.99 that justifies forty regression steps), **DR-6** (both time-limit ablations, "about half of the seeds in our runs"), **DR-10** ($\sigma$ halving), **DR-3** (a 50-seed DQN initialization comparison reported in a JAX tab note), and in `deep-rl.md` also "$\gamma = 0.9$ trained about as well in our runs". The reviewer counts six such claims in a section whose experiments have three seeds each.
16. **BL-7** ("for each of the ten seeds we tried"), **BL-13** (resampling statistics from the off-notebook seeds), and two exercise results quoted as facts in `baselines.md`.
17. `exploration.md`: "roughly one run in ten" (only the maximum is printed). `qlearning.md`: "over twelve seeds most runs overshoot". `imitation.md`: "Fourteen Adam steps" (IL-11), "within 3% at $T = 10$". `policy-gradient.md`: "for a few dozen updates" (PG-8), and two claims about slippery ice deferred to exercise 7d. All of these were recomputed or replicated by the reviewers and are correct; the point is that the reader cannot see them.

A grep of the sources confirms the phrases: "in our runs" three times in `deep-rl.md`, "seeds we tried" and "outside the notebook" once each in `baselines.md`.

**Unverified claims about papers** (each reviewer says what they believe; none could be checked from certain knowledge): the constants and form of DAgger's guarantee (IL-5, to be checked against Ross, Gordon and Bagnell before the statement goes into the body); that shaping by $\Phi$ acts like initializing the value table at $\Phi$, recalled as Wiewiora 2003 (MDP-12, VI-15); the attribution of the length divisors to GRPO, DAPO and Dr. GRPO, with DAPO held at moderate confidence (BL-8); that the convergence results of Agarwal et al. extend to the reachable cells (PG-13); whether UCB with $\kappa = 0.5$ has logarithmic expected regret on these arms (EXP-4). The claims about Nota and Thomas (PG-10), Even-Dar and Mansour (`qlearning.md`), Auer et al. and Agrawal and Goyal (`exploration.md`), and Pomerleau (`imitation.md`) match the reviewers' knowledge of the papers.

**Currency.** No superseded practice is presented as current. Two small gaps: the language-model literature's names for what `imitation.md` describes, exposure bias and teacher forcing, are absent (IL-8); and `mdp.md` tells the reader "The discount is part of the task, chosen by its designer" while `deep-rl.md` says "The discount is our choice here, not part of the task" without reconciling the two (DR-8; checked, both sentences are in the sources). The second is how practitioners treat $\gamma$, and one sentence should say that the earlier statement is being revised.

## Arc and order

The chapter has one line of argument and the order is right. Sections one to three answer the question of credit: write the problem down, compute values when the rules are known, learn them from single transitions. Section four takes up search. Sections five to eight change what is learned from a table of values to a policy: copy an expert, replace the expert's label by the reward, take the noise out of that estimate, replace the table by a network. Each section inherits code and a question from the one before, and every reviewer found the hand-overs clean (the cell 9 question from `mdp.md` to `value-iter.md` is the best). No section should move.

Three things blur the line.

**Boundaries that leak.**

- `mdp.md` works for later sections before it works for its reader: a 44-row symbol table of which 30 rows are forward references (MDP-1), a claim about optimal policies before "optimal" has a meaning (MDP-4), bootstrapped targets in the termination figure (MDP-8), the terminal condition of a telescoping sum not yet shown (MDP-11), and a 230-word API description of `VerifierTask` (MDP-14).
- Reward shaping is split between `mdp.md` and `value-iter.md` with the equation, the loop and the terminal condition in both (MDP-11, VI-14).
- The zero-mean property of the score, the reward-to-go form, the dropped $\gamma^t$ and the probe policy are each introduced in `policy-gradient.md` and again in `baselines.md` (PG-1, PG-10, BL-2, and structural finding 2 of the baselines review).
- `imitation.md` carries the `Policy` container for three later sections (IL-2); `policy-gradient.md` carries `Batch` fields that only `deep-rl.md` reads (PG-6).
- `baselines.md` and `deep-rl.md` carry material whose use is in the next chapter: the loss-divisor debate of GRPO, DAPO and Dr. GRPO (BL-8), the pathwise gradient (DR-11), and the method map (DR-15).

**Sections that are really two.** `baselines.md` has three subjects (which weights keep the mean; bookkeeping choices of implementations; a measurement study) and its reviewer gives the seam at "## Centering, Scaling, and Normalization". `deep-rl.md` has six subjects and a seam at "## Continuous Actions". `value-iter.md` has a possible seam before "The Optimal Policy on Slippery Ice". In all three cases the reviewers prefer shortening to splitting, and I agree: with the cuts listed under "What to cut", `baselines.md` and `deep-rl.md` come down to about 4,000 words each and each answers one question. `imitation.md` reads as two sections interleaved, one on imitation and one on library code; that is solved by slimming the container, not by a split.

**The chapter ends three times and the hand-over is told again by the next chapter.** I checked this against the sources, and it needs to be stated more exactly than the section review's summary. `deep-rl.md` closes with "### Limits of the Monte Carlo Agent" (a "We can now" paragraph that names three limits), then "## A Map of the Methods" (a figure and a 23-row table covering both chapters), then a Summary whose last sentence repeats the three limits. The next chapter's introduction, `chapter_deep-reinforcement-learning/index.md`, does not restate the map: it cites it in one sentence by label (":numref:`fig_rl_roadmap` and :numref:`tab_rl_map` place every method of both chapters"). What it restates is the hand-over, almost word for word: "The agent waited for every episode to end, used each batch once and never limited how far an update moved its policy", together with the "few hundred thousand steps". So the three limits appear three times within forty lines of prose across the chapter boundary, and the map sits at a point where the reader knows 11 of its 23 rows; the other 12 hold sixteen method names not yet met, five of them "named in" another section only, and two rows point to "the Language Models part", whose chapters are skeletons. The recommendation of DR-15 stands: keep the figure with this chapter's solid boxes as the chapter's closing picture, move the full figure and the table to the end of the next chapter (its `further-reading.md` is the natural home, and both labels survive the move), end the body of `deep-rl.md` on the Pendulum cost and one sentence naming the three limits, and take the limits out of the Summary. The next chapter's introduction keeps its statement, since it must open on its own. The chapter introduction's clause "and it ends with a map of the methods of this chapter and the next" then goes (RLIDX-5).

**Length against importance.** `value-iter.md` is the longest section (7,000 words) and earns it. `exploration.md` spends half of its MDP part on an experiment that shows nothing (EXP-5). The on-policy and off-policy distinction, by which the whole next chapter is sorted, gets two hundred words and no number (TD-7). Temporal-difference prediction, which gives `qlearning.md` its title, gets one paragraph (TD-1).

**A decision the section reviews leave open: the verifier task.** The four-prompt task is introduced in `mdp.md`, used for a sampling check in `exploration.md`, trained for the first time in `policy-gradient.md`, and carries the group baseline in `baselines.md`; the introduction calls `baselines.md` "the question behind the group-based updates that train language models". The baselines reviewer would move the group material to `rl-sequences.md` in the next chapter; the exploration reviewer builds on it staying. See "Where the section reviews disagree". My recommendation is a middle course: keep the thread, shorten it. `baselines.md` keeps the leave-one-out identity and the fact that a group of one gives no signal (they are baselines); the divisor debate and the three method names go to `rl-sequences.md`; `mdp.md` keeps the task and loses the API paragraph.

## The return from step $t$: checked

The reviewers reported that the return from step $t$ is never defined although later sections use it. That is correct, with one qualification: it is defined, in the seventh section of eight, under another name. In detail:

- `mdp.md` defines only the return of a whole trajectory, $R(\tau) = \sum_{t \geq 0} \gamma^t r_t$ (line 148), and bounds its tail (line 155). Its symbol table already lists $\hat{G}_t$ ("reward-to-go: the discounted sum of the rewards from step $t$ on", owner `sec_baselines`) and $y_t$ ("an estimate of the return from step $t$"). The caption of the termination figure uses the concept without a symbol or a definition: "the target for the last step, $y_7$, the return counted from that step" (line 227), and the text speaks of "the expected return from $s$ on" (line 225).
- `value-iter.md` defines $V^\pi$ by a sum from $t = 0$ and derives the expectation equation by "Split a trajectory after its first transition" in prose (line 60), with no symbol for the rest.
- `qlearning.md` describes the Monte Carlo target as "the realized return as the target for $Q(s_t, a_t)$" (line 65): a return from step $t$, unnamed.
- `policy-gradient.md` proves its theorem by "split the return into the rewards before $t$ and those from $t$ on" (line 366), again in words.
- `baselines.md` finally writes $\hat{G}_t = \sum_{t'=t}^{T-1} \gamma^{t'-t} r_{t'}$ (line 99) and, eleven lines later, the recursion $\hat{G}_t = r_t + \gamma \hat{G}_{t+1}$, as the "reward-to-go". The code's docstring calls it `G_t`. The hat suggests an estimate, but the quantity is the realized return of the rest of the episode.
- `deep-rl.md` builds its target on it: $y_t = \hat{G}_t + (1 - \mathrm{term})\,\gamma^{T-t}\,\hat{V}(s_T)$.

So five sections use an idea that the sixth names. The fix proposed in the review of `mdp.md` (structural finding 2) is right and is the most valuable single notational change in the chapter: define $G_t = \sum_{k \geq 0} \gamma^k r_{t+k}$ with $G_t = r_t + \gamma G_{t+1}$ and $G_0 = R(\tau)$ in "Discounting and the Effective Horizon", work it on one episode (MDP-6), and then $V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s]$ (VI-1), the Bellman expectation equation is the expectation of the recursion, the Monte Carlo target is $G_t$, the theorem's proof splits $R(\tau)$ into the rewards before $t$ and $\gamma^t G_t$, and `baselines.md` has nothing to introduce but the name "reward-to-go". Drop the hat, or keep it only for a truncated sum.

## Most important findings

1. **Make the results prose read what the cells print.** Where a sentence needs a frequency, print more seeds; where it does not, cut the anecdote; never explain away a printed number. TD-4, TD-6, TD-8, BL-11, BL-12, BL-13, DR-4, DR-6, DR-9, DR-10. `policy-gradient.md` is the model: twelve seeds, a band, ranges that match the store.
2. **Define the return from step $t$ once, in `mdp.md`, and build on it.** Structural finding 2 of the `mdp.md` review, MDP-6, VI-1; see the section above.
3. **Correct the four statements that are wrong as written.** EXP-1 (Lai and Robbins; the reviewer drafts a rewrite that turns the contradiction into a finding), BL-3, MDP-13, and the shaping potential (MDP-12, VI-15), which needs a decision: choose a potential for which the sentence is true, or keep this one and say candidly that it is a poor guide.
4. **Give each shared result one owner.** The zero-mean lemma and the policy gradient theorem in `policy-gradient.md`, stated once and before the experiments (PG-1, structural findings 1 and 2 of that review); `baselines.md` cites them and opens on the conditional identity (BL-2); the discounting convention stated once (PG-10, BL-4). Shaping: bonus, loop and definition in `mdp.md`, proof and terminal condition in `value-iter.md` (MDP-11, VI-14). The Bellman operator on action-value tables defined once (VI-5).
5. **Redesign the imitation experiment so that the proposition's two regimes come out of fitting.** IL-1, with IL-7 and IL-3. As it stands the quadratic curve is drawn from two formulas, and on the lake "distribution shift accounts for about a fifth of the clone's loss".
6. **Stop `mdp.md` from working for later sections.** MDP-1, MDP-4, MDP-8, MDP-11, MDP-14, MDP-15. The section becomes a clean 4,000 words: model, return, state, reward.
7. **Cut `baselines.md` and `deep-rl.md` to one question each and move what belongs to the next chapter.** BL-7, BL-8, BL-13, BL-14; DR-11, DR-14, DR-15; structural findings 1 and 4 of the baselines review and 1 to 5 of the deep-rl review. Bring "A Probe of State Coupling" forward to where its equation is introduced.
8. **Close the one gap in the foundations and make the demonstrations show their subject.** Prove existence of an optimal stationary policy in five lines (VI-3); start policy iteration where it visibly iterates (VI-10); say that $\pi^*$ does not maximize the success rate (VI-13).
9. **Give the on-policy and off-policy distinction a number, and the title of `qlearning.md` its update.** TD-7 (the exact values of $\pi^*$ at $\epsilon = 0.05$ are one line of code: 0.149 against 0.180 at the start), TD-1 (display TD(0) and say the section goes straight to control), and PG-11 (the comparison that is meant to show the cost of on-policy data does not show it).
10. **Reduce the confirming cells to the one or two the author allows.** VI-9, EXP-7, PG-4, PG-12, BL-6, IL-3, and the four cells whose whole output is `True` (PG-2).
11. **Rewrite the route of the introduction and say what can be skipped.** RLIDX-1, RLIDX-2, RLIDX-6.

## Focus

Surveys and variant lists are rare in this chapter, which is to its credit; the reviewers looked for expected SARSA, $n$-step targets, the average-reward criterion and similar, and found them rightly absent. What there is:

- **The method map** at the end of `deep-rl.md` (DR-15): a survey at the end of a teaching section, flagged in October when it stood in the introduction, now grown from 21 rows to 23.
- **Three methods named in one sentence** inside a FrozenLake measurement, none of which the reader has met: GRPO, DAPO, Dr. GRPO (BL-8). Still open from October.
- **Residual-gradient methods and Baird** before the reader has run Q-learning once (TD-2): good material in the wrong place.
- **Boltzmann exploration** defined and drawn in the middle of the $\epsilon$-greedy subsection, never run or judged (EXP-9).
- **The pathwise gradient** in `deep-rl.md`, used by nothing in the chapter (DR-11).
- **Apparatus narrated as content**: 190 lines of policy container with JAX notes on bucket padding and XLA determinism (IL-2); six helpers described before the reader knows what will be measured (BL-9); constructor options that the section does not use (DR-16); three sentences on why Adam lands on 0.735 "exactly, not approximately" (IL-7); a design note on matching step lengths and a paragraph of resampling statistics (BL-13). By the guide's own test, "If an experiment needs a paragraph of confounders before its result can be read, it is a research experiment", the DAgger run and the training study of `baselines.md` both need simplifying.
- **Paper retellings**: none of substance. Citations carry statements, not narratives.

## Experiments

The chapter's experiments are its strength and its recurring weakness, depending on the half.

**What is exemplary.** Grading against an exact solution, which makes most results free of sampling noise (`value-iter.md`, `qlearning.md`, the `visitation` evaluations of `imitation.md`, the frozen-policy table of `baselines.md`). Prediction before measurement (the bandit arithmetic of `exploration.md`: 83 of about 100, 46 against 46; the smallest case of `policy-gradient.md`; $\sigma \leq 1/2$ predicting "at least twice as long" in `baselines.md`). Experiments that could have come out otherwise and change what the reader will code (the naive stopping rule; the theory's UCB coefficient losing to an untheoretical one; four loss divisors on one batch).

**Cells that only confirm, and checks that cannot fail.**

| Finding | Section | What |
|---|---|---|
| VI-9 | value-iter | three asserts on bounds proved a page earlier; the cell-14 check on a converged iterate |
| EXP-7 | exploration | 1,000 sampled groups per prompt to confirm $(1-p)^K + p^K$ |
| IL-3 | imitation | "fresh" expert episodes that are the training pairs repeated; clones of 1, 10, 100 identical episodes |
| IL-6 | imitation | "It never rises above the reference, as part (i) requires" |
| PG-4, PG-12 | policy-gradient | two sampling tests of a proved unbiasedness, one with a non-monotone sequence the prose has to excuse, one with almost no power |
| BL-6 | baselines | one line of algebra enumerated twice, four `True`s |
| BL-7 | baselines | a sweep in which $K = 1$ cannot learn and every other arm saturates |
| PG-2 | policy-gradient, baselines, deep-rl | four cells whose whole output is `True` |

The reviewers agree on what to keep: the Monte Carlo cross-check of the model against the simulator in `value-iter.md` (the "once" the author allows), the assert that policy iteration and value iteration agree, the score check at first use in `policy-gradient.md`, and the Gaussian `log_prob` check as a test of code.

**Experiments that do not show their subject.** The lake in `imitation.md` (IL-1). The coverage run on calm ice in `exploration.md`, which "ends by saying the lake cannot show the effect" (EXP-5); the identical "19/20" under every schedule is the same seed each time, which the text should say. Policy iteration that terminates after one greedy step (VI-10). The entropy plot of `baselines.md` (BL-14) and the surrogate-loss plot of `deep-rl.md`, which draws a quantity the text says to ignore (structural finding 5 of that review).

**Experiments read for more than they show, and stored outputs against prose.** Listed under "Technical correctness and currency", items 10 to 17.

**Unfair or uninformative comparisons.** PG-11: REINFORCE at a batch mean of 0.7 against Q-learning's whole run, different criteria and the same order of steps, offered as the cost of on-policy data. DR-11: one quadratic at one $\sigma$ turned into "the lower variance" of the pathwise estimator; the appendix is more careful in its main text. The comparison with soft actor-critic in `deep-rl.md` is fair only if DR-9 is applied, since the next chapter counts steps to the same threshold.

**Scale.** Everything runs on a reader's machine; the most expensive cell is the Pendulum run (three seeds of 480,000 steps), minutes on a CPU. Three seeds is thin for the claims made about it.

**Proposed for the development machine** (untested): the corridor MDP for imitation (IL-1), policy iteration from `shortest` or always-up (VI-10), the three schedules on slippery ice (EXP-5, option b), 20 seeds for the two `qlearning.md` cells (TD-4, TD-6), a time-limit ablation arm and an explained-variance print in `deep-rl.md` (DR-4, DR-6), a learner trained with a potential if the claim of MDP-12 is kept.

## Repetition

**Inside the chapter**

| What | Where it is told | Proposed single home |
|---|---|---|
| Reward shaping: the potential equation, the loop, the purpose of a potential, the terminal condition | `mdp.md` "Reward Design"; `value-iter.md` `subsec_vi_shaping` (MDP-11, VI-14) | `mdp.md`: bonus, loop arithmetic, definition. `value-iter.md`: proof, terminal condition, slippery-ice computation, citing the equation by label |
| Existence of an optimal deterministic stationary policy | asserted in `mdp.md` (MDP-4); proposition in `value-iter.md` | `value-iter.md`, proved (VI-3); one clause in `mdp.md` |
| Zero-mean property of the score | inline in the proof of the theorem in `policy-gradient.md`; Lemma with the same proof in `baselines.md` (PG-1, BL-2) | `policy-gradient.md`, "The Score Function", as the Lemma |
| Reward-to-go form | first consequence of the theorem; re-derived under "Reward-to-Go and Causality" | `policy-gradient.md`; `baselines.md` cites `eq_pg_theorem` in two lines |
| The dropped $\gamma^t$ | third consequence in `policy-gradient.md`; a paragraph with the same two citations in `baselines.md` (PG-10, BL-4) | `policy-gradient.md`, with the sentence saying what this book's code does |
| The probe policy and its linear system | built and explained in both policy sections | `policy-gradient.md` |
| The surrogate loss | one sentence in `policy-gradient.md`; a subsection, plot and exercise in `deep-rl.md` | `policy-gradient.md`; `deep-rl.md` keeps the new identity in two sentences |
| Termination against truncation | `mdp.md` with targets (MDP-8); `qlearning.md` in the update; `policy-gradient.md` in `Batch` (PG-6); `deep-rl.md` "Time Limits" | definition in `mdp.md`; the target rule in `qlearning.md`; the bootstrap in `deep-rl.md`; nothing in `policy-gradient.md` |
| The definition of a policy | `mdp.md` and again in `value-iter.md` (VI-2) | `mdp.md` |
| The Bellman operator | on $V$ tables in `value-iter.md`, redefined on $Q$ tables in `qlearning.md` (VI-5) | `value-iter.md`, one added sentence |
| Step-size schedules | introduced with rates, then retold with results (TD-13) | "Step-Size Schedules Compared" |
| 54% against 73% | twice in `qlearning.md`, its closing, the opening of `exploration.md` (TD-11, EXP-12) | once where measured, once as the next section's hook |
| "More expert data cannot help" | five times in `imitation.md` (IL-4) | once explained, once as the motive for DAgger, once in the Summary |
| An identical group gives no signal | `exploration.md` and `baselines.md` | allowed size; let `baselines.md` own the formula if EXP-7 is applied |
| The three limits and "a few hundred thousand steps" | "Limits" paragraph and Summary of `deep-rl.md`; see next table | one sentence at the end of the body |

**With other Advanced chapters.** The only neighbour that matters is chapter 15. Its introduction restates the three limits and the step count (kept, since it must open on its own); `actor-critic.md` restates the first limit; `sac.md` and `rl-sequences.md` restate the step comparison (structural finding 1 of the deep-rl review). `rl-sequences.md` assembles GRPO and cites the same two variants as `baselines.md` (BL-8); the cell IDs `rl-sequences-the-group-is-the-baseline-2` and `-3` inside `baselines.md` show the cells came from there. The pathwise gradient is taught in `deep-rl.md` and first used in `sac.md` (DR-11). The method map is announced in this chapter's introduction, printed in `deep-rl.md` and cited by the next chapter's introduction. No overlap with chapters 9 to 13 was reported; the Robbins-Monro conditions appear in `qlearning.md` and in the optimization chapter's `sec_sgd`, a small duplication that needs only a clause.

**With Basics.** Softmax regression is cited by label where the tabular policy and behavior cloning reuse it (`imitation.md`, `policy-gradient.md`, `exploration.md`); nothing is re-taught. Teacher forcing from the RNN language model is the same thing as training on the expert's prefixes and is not named (IL-8).

**With the appendix.** The chapter never cites the mathematics appendix (a grep for `sec_mdl`, `subsec_mdl` and `eq_mdl` in the nine files finds nothing), and by the author's rule that is acceptable: it reads without it. Three overlaps exist. (a) The score-function identity, the constant baseline and an optimal constant baseline for a Gaussian are in `subsec_mdl-stochastic-gradient-estimators` (in `chapter_mdl-calculus/mdl-integral-calculus.md`; the label exists). These are small results and may stand in both; `policy-gradient.md` and `baselines.md` each owe a one-clause pointer. (b) The comparison of score-function and pathwise estimators in `deep-rl.md` repeats the appendix's comparison with a different quadratic, uncited, and is less careful than the appendix about when the pathwise estimator loses (DR-11). Under the rule this is the one case of a derivation written out in both: move the subsection to `sac.md` and cite the appendix there. (c) Hoeffding's inequality is quoted with its statement in `exploration.md`, the right amount; `sec_mdl-concentration-generalization` treats it at length and deserves a clause. No long derivation in this chapter belongs in the appendix instead; the contraction and policy-improvement proofs are specific to the chapter and short.

## Recurring patterns

**Every section ends twice.** Five sections open their last body paragraph with "We can now" (`mdp.md`, `value-iter.md`, `policy-gradient.md`, `baselines.md`, `deep-rl.md`; counted in the sources), and the other three have a recap paragraph without the words: MDP-15, VI-18, TD-11, EXP-10, IL-9, PG-14, BL-15, DR-14. In every case the reviewer finds one or two sentences of real hinge inside the recap ("An expert who shows good actions would spare the agent all of this"; "A learning agent is rarely given the table. It sees one transition at a time") and recommends keeping those and nothing else.

**Results prose that outruns the output.** Twelve findings in four sections; see above. Phrases to search for in both reinforcement-learning chapters: "in our runs", "seeds we tried", "outside the notebook", "about one run in".

**Code and definitions built two or three sections ahead of their use.** `mdp.md` (symbol table, `VerifierTask`, targets), `imitation.md` (`seeded`, `act_greedy`, `log_prob_np`), `policy-gradient.md` (`next_obs`, `term`), `deep-rl.md` (`trunk`, `state_std`, the pathwise estimator).

**Helper narration.** "`estimate` sums the weighted scores ...", "`backward_scan` runs this recursion ...", "`train` is REINFORCE by plain SGD ...", "`to90` finds the first update ..." (BL-9, five instances in one section); the `VerifierTask` paragraph (MDP-14); the nine one-line definitions before the first number in `qlearning.md` (TD-5).

**Framework plumbing and tab notes.** The first cell of every section imports the full framework stack; the first four sections then use only NumPy (reported for `mdp.md`, `value-iter.md`, TD-14, and `exploration.md`), and `qlearning.md` says "Every cell in this section is plain NumPy" under a cell that imports `torch`, `nn`, `F`, `optax`. Long JAX notes on compilation and determinism in `imitation.md` (IL-2, three times the length of the PyTorch notes) and a JAX tab note that reports an experiment (DR-3). Optimizer asides in the argument (PG-8, DR-2). The two tabs fragment the prose only in "The Policy Container". No TensorFlow or MXNet material anywhere.

**Mannerisms.** Few. "X, not Y" closers: four in `qlearning.md` (TD-12); a grep finds one or two "not a ..." constructions in most other sections. "exactly": about 37 occurrences across the chapter, ten of them in `imitation.md`; many are literal (exact evaluation), but the count is worth a pass. Economic metaphors in identifiers and comments: `taxed`, `regret` used for a shortfall (TD-5), "pricing" (EXP-14), a cell ID with "the-linear-tax". The double hyphen "state--action" in `imitation.md` only (IL-12). A hardcoded "For seven sections" (DR-1). Stale cell IDs and exercise labels that carry old headings or the wrong file prefix are noted in five reviews; IDs are permanent by house rule and no action is asked.

**Voice.** Clean, plain and mostly even. Where the authors judge, the reviewers quote it as the best prose in the chapter: "nothing the rule can see reveals this"; "The conditions guarantee the limit; they do not choose a schedule for a finite budget"; "Our constant was chosen for these arms, and the guarantee does not cover it"; "the estimate is right on average and useless in almost every single draw"; "Lower variance does not make a small step better ... it makes a longer step usable"; "the evidence that the loop works comes from runs, not from a convergence proof". What is asked for is more of this ranking of ideas (MDP-5, VI-8, and the pattern notes of the first two reviews), not ornament. One stock example should go: the restaurant in the opening of `exploration.md`, when Thompson's two treatments stand in the next paragraph (EXP-13).

**The introduction's route.** Eight consecutive "asks" sentences (RLIDX-1), the template the Basics review found in chapter 4.

### Notation: one table, with the drift

The chapter's symbol table is `tab_rl_notation` at the head of `mdp.md` (44 rows, both chapters). Its caption already apologizes for three collisions ($\epsilon$ and $\varepsilon$, $y$ and $y_t$, $\mu$ and $\mu_0$), which MDP-2 reads correctly as a symptom. The table below collects what the eight sections actually write. "Table" means `tab_rl_notation`. Entries I verified by search of the sources; line numbers are of the `.md` files.

| Symbol | Meaning and owner | Where it drifts |
|---|---|---|
| $r(s,a)$, $r_t$ | expected reward; realized reward at step $t$ (`mdp.md`) | `value-iter.md` defines values with $r(s_t, a_t)$ and calls the sum the return of `sec_mdp` (VI-1). $r(x,y)$ is the verifier's score; $r_\pi$ a policy-averaged reward vector (`value-iter.md`, `policy-gradient.md`, `baselines.md`, consistent); $r_0$ is also a radius in the caption of `fig_rl_contraction`, called $e_0$ in the text below it (VI-7); an exercise of `mdp.md` writes a transition reward as $\rho(s,a,s')$ |
| $R(\tau)$ | return of a trajectory (`mdp.md`) | `R` is the verifier's reward table in code (MDP-2); $R(a)$ is the reward of one action in the Gaussian example of `baselines.md`; $\bar{R}$ there is the batch mean of the reward-to-go values, and also the mean of a group's $K$ rewards |
| return from step $t$ | not defined until `baselines.md` | $\hat{G}_t$ "reward-to-go" (`baselines.md`, `deep-rl.md`), `G_t` in a docstring, "the return counted from that step" (`mdp.md`), "the realized return" (`qlearning.md`), "the rewards from $t$ on" (`policy-gradient.md`). See the section above |
| $y$, $y_t$ | response; target of a value update (table) | $y_7$, $y_{99}$ in a figure of `mdp.md` before targets exist (MDP-8); $y_n$ is the $n$-th target received by one entry in `qlearning.md`; $y$ the regression target in the kernel equation of `deep-rl.md`; `y_t` a generic sequence in the docstring of `backward_scan` |
| $V^\pi$, $Q^\pi$, $V^*$, $Q^*$ | values (`value-iter.md`) | $V(s)$ appears in `mdp.md` before its definition (MDP-8). Learned tables are $Q$ in `qlearning.md` but $\hat{Q}$ in its maximization-bias passage and in `exploration.md`; $\hat{V}$ is a table in `baselines.md` and a network $\hat{V}_w$ in `deep-rl.md` |
| $\mathcal{T}$ | Bellman optimality operator on tables $V$ (`value-iter.md`, 29 uses) | redefined on tables $Q$ in `qlearning.md` (VI-5). $T$ is the episode length |
| $T$, $t$ | number of steps; step index from 0 | `imitation.md` indexes steps $1, \ldots, T$ in its proposition and proof (IL-13) |
| $\pi$, $\pi^*$, $\pi_\theta$, $\hat\pi$, $\pi_e$ | policy; optimal policy; parametric policy; the clone; behavior policy | $\pi^*$ is also "the expert" in `imitation.md` (the same object on the lake, but say so). $\pi_e$ appears in `qlearning.md` only (4 times); the SARSA discussion needs it and does not use it |
| $P(s' \mid s, a)$, $P_\pi$ | kernel; kernel under $\pi$ | $P_\pi$ is used consistently in five sections (done well). $P(\tau; \theta)$ is a trajectory probability in `policy-gradient.md` |
| $d^\pi_t$, $d^\pi_\gamma$ | distribution of $s_t$; its discounted average (table: `imitation.md`, `policy-gradient.md`) | first used in exercise 2b of `mdp.md`. Exercise 4 of `policy-gradient.md` defines the unnormalized $\rho_\gamma(s) = \sum_t \gamma^t d_t(s)$ beside the normalized $d_\gamma$; an exercise of `baselines.md` uses $\rho(s)$ for an expected visit count; $\rho$ is a spectral radius in `value-iter.md` and the importance ratio $\rho_t$ in the table. `qlearning.md` writes a bare $d$ for a distribution over pairs |
| $\gamma$ | discount, 0.95 "throughout this chapter" (`mdp.md`) | 0.99 on CartPole; "our choice here, not part of the task" on Pendulum (DR-8) |
| $\epsilon$, $\varepsilon$ | exploration rate; per-step disagreement | two glyphs of one letter (MDP-2). Also $\epsilon_0$ (Adam) and $\epsilon_{\textrm{clip}}$ (table) |
| $\alpha$, $\eta$, $\eta_V$ | step size of a TD update; of a policy update; of a baseline update | three letters for a step size is defensible and the table says so; $\eta_V$ in the kernel equation of `deep-rl.md` stands for a plain gradient step while the section trains with Adam |
| $\beta$ | KL weight and soft actor-critic's temperature (table, next chapter) | Boltzmann temperature in `exploration.md`, not in the table (EXP-9); belief $\beta_t(s)$ in exercise 7 of `mdp.md` (MDP-2); mixing probability $\beta_i$ in exercise 8 of `imitation.md`; Adam's $\beta_1, \beta_2$ in an exercise of `deep-rl.md` |
| $\mu_0$, $\mu_a$, $\mu^*$, $\mu$, $\mu_\theta(s)$ | start distribution; arm means; Gaussian mean | the caption's rule ("A $\mu$ with a subscript or argument other than 0 is a mean") is needed because of this |
| $\sigma$ | standard deviation of a batch's weights (`baselines.md`); of the Gaussian policy (`deep-rl.md`) | both within two sections of each other; also the sigmoid in exercise 1 of `policy-gradient.md` |
| $n$ | (table) depth of an $n$-step target | visit count in $\alpha_n$ (`qlearning.md`); pulls $n(a)$, $n_t(a)$ (`exploration.md`); number of expert episodes (`imitation.md`); episodes per batch (`policy-gradient.md`, `baselines.md`); number of responses in the leave-one-out paragraph of `baselines.md`, where the group size is $K$ everywhere else |
| $N$ | none | corridor cells (`mdp.md`), recorded transitions (`qlearning.md`), chain length (`exploration.md`), DAgger rounds (`imitation.md`), steps in a batch (`policy-gradient.md`, `baselines.md`, `deep-rl.md`) |
| $K$, $k$, $M$, $m$ | responses per prompt; sweep or tail index; number of averaged estimates; number of arms | $k(s, s_0)$ is the tangent kernel in `deep-rl.md`; $m$ is also a mean and a sweep count in exercises |
| $\Delta$ | gap $\Delta_a$ of an arm (`exploration.md`) | action gap (exercise of `value-iter.md`); the change $\Delta\hat{V}$ (`deep-rl.md`); $\Delta_{\textrm{KL}}$ (table) |
| $\delta$ | TD error | average regret $\delta_N$ in exercise 8 of `imitation.md` |
| $\hat{u}$, $\hat{g}$ | the gradient estimate | $\hat{u}$ in `policy-gradient.md` and `baselines.md`, $\hat{g}$ in `deep-rl.md` for the same object. $u$ is the cost of one mistake in `imitation.md` and a round index in `exploration.md` |
| $w$ | parameters of a value network (table, `deep-rl.md`) | per-sample weights $w_i$, $w_t$, $\tilde{w}_t$ (`imitation.md`, `policy-gradient.md`, `deep-rl.md`, in the same section as the network's $w$); wins of an arm (`exploration.md`) |
| $c$ | a prefix-determined coefficient (`baselines.md`) | count of correct responses (`policy-gradient.md`); constants in exercises |
| $\ell$, $L$ | Manhattan distance (`mdp.md`, `value-iter.md`) | empirical loss $\ell(Q)$ and population loss $L(Q)$ (`qlearning.md`); surrogate loss $L(\theta)$ (`deep-rl.md`); response length $L$ and $\ell = \log\sigma$ in exercises |
| $p$ | probability of the rewarded action (`policy-gradient.md`) | Bernoulli mean (`exploration.md`); consistent enough |
| expectations | none fixed | subscript listing the sampling rules (`value-iter.md`), $\tau \sim P(\cdot; \theta)$ (`policy-gradient.md`, `baselines.md`), bare $\mathbb{E}$ with conditioning in prose elsewhere; VI-1 proposes $\mathbb{E}_\pi$ |
| "unbiased", "same mean" | equal in mean to $\nabla_\theta J$ | from "Reward-to-Go and Causality" on it means "same mean as `eq_rtg`", with $\gamma^t$ dropped; the Summary of `baselines.md` says "without changing the expected gradient" (BL-4) |

Done well and to be kept: $V^\pi$, $Q^\pi$, $P$, $r(s,a)$ in the research literature's convention, not Sutton and Barto's $v_\pi$, $p(s', r \mid s, a)$; $\theta_{s,a}$ and "preference" throughout; $J(\theta)$; $\kappa$; $P_\pi$ and $r_\pi$; the reward $r_t$ indexed with the action that earned it. Cheap fixes proposed by the reviewers: one symbol $G_t$ for the return from step $t$; a letter other than $y$ for value targets; two different letters for $\epsilon$ and $\varepsilon$; $e_0$ in the contraction figure; a letter for the Boltzmann temperature that is not $\beta$, or a row in the table; one name for the gradient estimate; $K$ in the leave-one-out paragraph; and the table itself moved out of `mdp.md` to the end of the introduction or a notation page (MDP-1; the plan in `chapter_lm-rl/index.md` cites `tab_rl_notation` with "owner `sec_mdp`", so update that pointer if it moves).

## Connection to the main text, and dependencies

**What the chapter needs.** From Basics: softmax regression and cross-entropy (`sec_softmax`), gradient ascent, expectations and variances, a linear solve, `einsum`, Adam as a black box, and for `deep-rl.md` a two-layer network. Cited by label where used. From earlier Advanced chapters: nothing; the chapter can be read directly after Basics, and the introduction should say so (RLIDX-2). From the appendix: nothing; there is no reference into it and none is needed to follow the text. Because there are no references into the appendix, there are none that point to the wrong place; the pointers owed are listed under "Repetition".

**Inside the chapter** the dependence is a chain with one shortcut: `imitation.md` needs only the first two sections; `policy-gradient.md` needs `imitation.md` for the container and `value-iter.md` for the linear system; `baselines.md` and `deep-rl.md` need everything policy-side before them. A reader heading for language models can skip `qlearning.md` and `exploration.md` at first reading without losing the thread, apart from the definition of on-policy and off-policy.

**Forward references.** To the next chapter: frequent and by label (`sec_dqn`, `sec_actorcritic`, `sec_ppo`, `sec_sac`, `sec_offline`, `sec_rl_sequences`); two are more than pointers: the group update in `exploration.md` leans on `sec_baselines` (not load-bearing, per the reviewer), and the JAX note DR-3 reports an experiment from `sec_dqn`. To the Language Models part: three mentions (the introduction; two rows of the method table in `deep-rl.md`, DR-15), all pointers to "more depth", none load-bearing; `imitation.md` and `mdp.md` discuss language models in self-contained paragraphs. The chapters of that part are skeletons, so the table rows promise text that does not exist yet (RLIDX-8).

**Inbound references that constrain the revision.** Skeleton chapters `chapter_lm-rl`, `chapter_lm-finetuning`, `chapter_lm-reasoning` and `chapter_lm-agents` cite labels of this chapter (among them `subsec_il_sft`, `subsec_il_dagger`, `subsec_il_error_bounds`, `tab_rl_notation`), as do `chapter_advanced/index.md`, the appendix (`chapter_mdl-calculus/index.md`, `mdl-stochastic-gradient-methods.md`), `chapter_preliminaries/autograd.md` and `chapter_linear-classification/environment-and-distribution-shift.md`. The labels must survive any move of material.

**Missing or weak links.** No clause connects the Robbins-Monro conditions to `sec_sgd`. Teacher forcing (`sec_rnn`) is not named in `imitation.md` (IL-8). Exercise labels with the prefix of another file (`ex_mdp-reward-design-its-failure` lives in `value-iter.md`; two `ex_qlearning-*` labels live in `exploration.md`) resolve correctly and mislead only editors. One hardcoded count, "For seven sections" (DR-1). References otherwise use labels.

**Further reading.** The chapter has no `further-reading.md` and does not need one; the reasons are in the review of the introduction. The next chapter's page closes both chapters in its "Beyond the Chapter" paragraph. The resources list of the introduction should be ranked and should say how the chapter relates to Sutton and Barto (RLIDX-6).

## What to cut

About 6,000 of 42,000 words, almost none of it argument:

- `mdp.md` (about 1,200): the symbol table moved out (MDP-1); the three sentences on optimal policies (MDP-4); the target formulas in the termination paragraph and caption (MDP-8); the terminal-condition paragraph (MDP-11); the API paragraph of `VerifierTask` (MDP-14); the recap (MDP-15).
- `value-iter.md` (about 400): the second telling of shaping's setup (VI-14); the three assertion sentences (VI-9); panel (d) of the backup figure, moved to `qlearning.md` (VI-6); the middle sentence of the closing (VI-18).
- `qlearning.md` (about 400): a third of the double-sampling passage (TD-2); two asides (TD-9, TD-10); the narrated failure (TD-4); the recap (TD-11).
- `exploration.md` (about 500): the lake coverage cell and plot, or their replacement by a run where the schedule matters (EXP-5); the sampling in the verifier cell (EXP-7); the recap (EXP-10).
- `imitation.md` (about 600 words and 130 lines of code): `seeded`, the acting-method narration and the JAX notes (IL-2); the checks that cannot fail (IL-3); two of the four statements that more data cannot help (IL-4); the Adam explanation (IL-7).
- `policy-gradient.md` (about 500): the `M` sweep (PG-4); the `Batch` termination sentences (PG-6); the Q-learning comparison (PG-11); the recap (PG-14).
- `baselines.md` (about 1,300): the group sweep, the divisor debate and its three method names, moved to `rl-sequences.md` with exercises 6 and 10 (BL-7, BL-8); the second enumeration (BL-6); the resampling and design-note paragraphs (BL-13); the entropy plot (BL-14); the re-derivations (BL-2).
- `deep-rl.md` (about 1,500): the pathwise subsection with its figure and exercises 3 and 6, moved to `sac.md` (DR-11); the map table, moved to the next chapter (DR-15); the surrogate-loss subsection reduced to two sentences; the "limits" paragraph reduced to one sentence (DR-14); exercise 8 on evolution strategies moved to the next chapter's further reading.

## What is missing

- The return from step $t$, $G_t$, with its recursion, worked on one episode (`mdp.md`; MDP-6).
- One trajectory that is actually scored: the only episode `mdp.md` shows has return zero (MDP-6).
- A proof, or a plain admission, at the one result the dynamic-programming section takes on trust (VI-3).
- Policy iteration that iterates (VI-10), and the cost of a sweep in operations (structural finding 4 of the `value-iter.md` review).
- The TD(0) update displayed, and a sentence that the section skips prediction (TD-1).
- On-policy against off-policy as two columns of numbers on the lake, and why the next chapter is sorted by it (TD-7).
- The one-line reason UCB has logarithmic regret, in the body where the rule is introduced (EXP-3); the stationarity assumption behind every bandit result (structural finding 2 of the exploration review); two sentences on why none of the section's rules is what a language model uses (structural finding 3 there).
- What DAgger guarantees, as a statement (IL-5); the terms exposure bias and teacher forcing (IL-8); a judgment on when plain behavior cloning is the right choice (structural finding 7c of the imitation review).
- The hypotheses of the policy gradient theorem and its tabular form $\rho_\gamma(s)\,\pi_\theta(a \mid s)\,(Q - V)$ in the text (PG-9); what this book's code does about $\gamma^t$ (PG-10).
- The variance-optimal baseline in the text, and the sentence that $V^\pi$ is not it (BL-5).
- A sentence on an entropy bonus, which the stall at $\eta = 16$ in `baselines.md` almost asks for, and one saying that limiting the step is deferred to PPO (the deep-rl review's closing note).
- A time-limit ablation the reader can run (DR-6).
- An ending for the chapter that is not a map: three or four sentences that return to the two questions of the introduction.
- In the introduction: prerequisites, a short route, the running examples by name, the relation to Sutton and Barto (RLIDX-2, RLIDX-6).

## How the chapter compares with Sutton and Barto

Assembled from the "Against Sutton and Barto" notes of the section reviews, which agree with each other.

| Material | This chapter is better at | Sutton and Barto are better at |
|---|---|---|
| MDPs (their ch. 3) | termination against truncation, absorbing states, horizon arithmetic, a reward that can be gamed, the language-model case, reading a real simulator | forming the concept: several small MDPs, the agent and environment boundary, $G_t$ with its recursion |
| Dynamic programming (ch. 4) | rigor (contraction, certificate, shaping invariance proved); a stopping rule shown to fail; the anatomy of an optimal policy | asynchronous and in-place sweeps in the text; policy iteration on examples where it iterates |
| TD learning (ch. 6) | why $Q$ and not $V$; step sizes at a finite budget; exact grading; the termination mask | prediction before control, with experiments; on-policy control actually run (windy grid, cliff); why bootstrapping helps |
| Bandits (ch. 2) | the cost of exploring: regret defined, predicted, measured, bounded below; theory's constant against a tuned one | nonstationary arms and constant step sizes; gradient bandits as the bridge to policy gradients |
| Policy gradient (ch. 13) | the smallest case; variance accounting; baselines, step sizes and divisors kept apart; the dropped $\gamma^t$ said openly | brevity; the average-reward formulation |
| Imitation | proved bounds, an exact decomposition | not treated by them; the fair comparison is the Berkeley and Stanford course material, whose driving and locomotion demonstrations show the drift far more vividly than the lake |

In one sentence: this chapter is the better reference for someone who will implement and check these methods, and Sutton and Barto remain the better first teacher of the concepts in the three places listed (the return, prediction, on-policy control); the first costs two lines to fix and the other two a paragraph and a number each.

## What is best and should set the model

- **`value-iter.md`, the equation of cell 14.** "The unknown $V^\pi(14)$ appears on both sides, because a slip can leave the agent where it was." One line shows why the equations are solved together. The section reviewer would show this page as the book's standard.
- **`value-iter.md`, stopping rules and the anatomy of $\pi^*$.** An experiment that changes what the reader codes (128 against 164, "nothing the rule can see reveals this"), then a solution read cell by cell until it is understood.
- **`qlearning.md`, from the operator to one sample.** "The model enters the backup in one place, the average over the successor $s'$", then three hand-computed updates.
- **`exploration.md`, the greedy lottery and the UCB coefficient.** Exact prediction of what a bad rule does, and a tested, derived account of the gap between a theorem's constant and a good one.
- **`imitation.md`, the opening and the proposition.** Pomerleau's van used for its content; a theorem, a ten-line proof, and a sentence that says what it means.
- **`policy-gradient.md`, the opening and "The Smallest Case".** The algorithm in the reader's hands before the derivation; one state and two actions that predict the lake's flat start. Also the model for seed handling.
- **`baselines.md`, control variates and the three-way distinction.** A baseline, a step size and a divisor are different things, each shown with a number.
- **`deep-rl.md`, the cart, the table as a linear network, and "Time Limits".** A problem felt before the machinery; a failure shown with numbers the reader can redo.
- **`mdp.md`, the cell 9 question and the loop arithmetic.** A question only values can answer, left open for the next section; four numbers that make reward hacking concrete.
- **`index.md`, the first paragraph.**
- **The exercises**, which every section review singles out (the reviewer of `value-iter.md` calls its set the best seen in the main text). They run above the band in two sections (twelve in `value-iter.md`, eleven in `baselines.md`), and several hold results the text should state (the DAgger guarantee, the optimal baseline, TD(0), the success-optimal policy). Moving those results into the body makes the exercises proofs and brings the count down.

The common thread is the chapter's method: a problem small enough to solve exactly, a prediction made before the cell runs, and a candid sentence about what the result does and does not establish. The revision's task is to hold the second half of the chapter to the standard the first half sets.

## Where the section reviews disagree

No review contradicts another on a fact. Four differences of recommendation need a decision.

1. **The group material in `baselines.md`.** The baselines reviewer moves "The Group as the Batch" and the divisor debate to `rl-sequences.md` (structural finding 1 there, BL-8). The exploration reviewer keeps the verifier thread in this chapter and suggests that `baselines.md` own the identical-group formula (EXP-7, closing notes). The introduction also promises the group update here. My recommendation is under "Arc and order": keep the identity and the $K = 1$ fact, move the debate.
2. **Where `VerifierTask` belongs.** MDP-14 says to move the class to "the section that first trains on it" and names `exploration.md`. The cell there does not train (its text says "it is not a trained policy") and EXP-7 would remove its sampling; the first training is in `policy-gradient.md`. Either keep the class in `mdp.md` with two sentences, as MDP-14 also allows, or move it to `policy-gradient.md`.
3. **Where the `Policy` container lives.** IL-2 slims it in place or moves `seeded` to `sec_utils`; the policy-gradient and deep-rl reviews rely on it being in `imitation.md`. All three positions are compatible with a slim container in `imitation.md`; the imitation reviewer asks for a chapter-level decision, and that is mine.
4. **Where truncation's target rule goes.** MDP-8 moves the bootstrapped target to `qlearning.md`; PG-6 moves the `Batch` termination sentences to `deep-rl.md`. These are different sentences and both moves are right.

The same defect was found independently in two reviews twice (MDP-12 and VI-15; MDP-11 and VI-14), and TD-9 proposes to move to `value-iter.md` the very fact that VI-13 says is missing there. EXP-2 and the code-and-experiments note of the `qlearning.md` review report the 256 against 512 mismatch from both sides.

## Could not check

- Rendered HTML and PDF pages; the reviews rest on sources, stored outputs and rasterized figures.
- Anything that needs PyTorch, JAX or Gymnasium on this machine. The reviewers reproduced the tabular results on independent NumPy reconstructions; the Monte Carlo columns, the network runs on CartPole and Pendulum, and the $8 \times 8$ lake claims in exercises were not re-run.
- Claims about run-to-run variation beyond the stored seeds, and every off-notebook number listed above.
- The proposed experiments (IL-1, VI-10, EXP-5 option b, the extra seeds, the ablation arms) are untested.
- The paper claims listed as unverified, the page count of Szepesvári's book, and the course URLs in the introduction.
- The next chapter's section reviews were not read for this file; statements about `actor-critic.md`, `sac.md` and `rl-sequences.md` are taken from this chapter's section reviews, except the next chapter's introduction and further-reading page, which I read.
