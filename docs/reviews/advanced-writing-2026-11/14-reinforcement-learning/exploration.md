# Exploration

Source: `chapter_reinforcement-learning/exploration.md` (lines 1 to 823, up to the slides marker). About 4,700 words of body prose plus 1,300 words of exercises; 9 code cells (about 150 lines, plain NumPy, shared by both tabs); two pre-generated figures (`mdl-rl-exploration.svg`, `mdl-rl-ucb.svg`), both rendered and inspected.

## Verdict

The bandit half of this section is the best exposition of the three I reviewed, and parts of it are at the bar: the greedy rule is not waved away but worked out exactly (it settles on arm $a$ with probability $\mu_a/\sum_b \mu_b$, so the best arm wins "in about one run in five" and the expected regret is 707), the cost of a fixed $\epsilon$ is predicted from the average gap before it is measured, and the theory's UCB coefficient is shown to lose to an untheoretical one by a factor the text then derives. Every one of these numbers checks. The second half is weaker. One statement is wrong as written: the Lai and Robbins bound is applied at $t = 2000$ to say an arm "must be pulled about" 24 times, while the section's own figure shows that arm pulled 11 times and its own experiments show total regret below what the bound's leading term would give. The lake coverage experiment ends by saying the lake cannot show the effect. The verifier cell confirms a one-line probability formula by sampling. And the section's two halves do not meet: the reader learns UCB and Thompson sampling and is then told that language models explore "differently from the rules above", without a word on why none of those rules is used. The single most important fix is the Lai and Robbins passage; the most important structural one is to cut or redesign the lake coverage experiment.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought and focus | 3 |
| Pedagogy | 4 |
| Language | 4 |
| Ease and enjoyment | 4 |

("The Multi-Armed Bandit" and "Optimism and Posterior Sampling" up to "The UCB Coefficient": 4 to 5 throughout. "Exploration in MDPs" from "Count-Based Bonuses" on, and the verifier section: 3.)

## What works

- **Regret is motivated before it is defined**: "A success rate says how good a policy is; for a learner we want to know how much the learning cost." Then the definition, the regrouping by arm, and a number: "loses $0.40$ per pull, $800$ after $2000$ pulls".
- **The greedy lottery.** "Greedy therefore pulls random arms until one returns a reward and then never pulls anything else." The argument that follows predicts the distribution of outcomes over runs, and the median of the stored output (800) is the value $2000\,\Delta_a$ for the arms with mean 0.50, as the text says. This is the specific, checkable, slightly funny example the writing guide asks for. Protect the whole paragraph.
- **Prediction before measurement, twice.** Fixed $\epsilon$: $0.1 \times 0.41 \approx 0.04$ per pull, "about $83$ of its median regret of about $100$" (stored: 106). Annealed: 112 random pulls, cost 46, against a stored median of 46. A reader learns to do the arithmetic before running the code.
- **Candour about constants**: "Our constant was chosen for these arms, and the guarantee does not cover it", with the guarantee's own constant computed ("nearly thirty times our $20/t$"; I get 27.8). The same candour for UCB: "The value $\kappa = 0.5$ lies below the range $\kappa > 1$ that the theory covers."
- **"The UCB Coefficient".** The theory's coefficient is tested, loses by "five to six times" (stored: 204 against 38), the too-small coefficient is shown to fail in the tail ("the median hides these runs, and the mean does not"), and the ratio in long runs is derived as $(\sqrt{2}/0.5)^2 = 8$. I simulated $10^5$ pulls over 100 runs: 443 against 58, a ratio of 7.6, and 14% of the $\kappa = 0.15$ runs above 100 at 2,000 pulls. The prose is right on each point. This subsection also contains the best judgment of the section: "What the theory guarantees is something else."
- **Thompson sampling in one paragraph with one computation** (an untried arm beats $\mathrm{Beta}(4, 2)$ with probability $1/3$), then exercise 3 to prove the general statements.
- **The chain.** Exponential against roughly linear, with exact numbers for the random policy (23; 3,449; 45,590, all of which I reproduced) and a three-sentence reason why optimism works: "a tried pair is marked as tried".
- **Log axes read honestly**: "$2000$ pulls cannot establish an asymptotic rate, but the bend is what logarithmic growth looks like."
- Figure `fig_rl_ucb` is good: panel (b) shows every index pressed against the dashed line, which is the mechanism.

## Structural findings

1. **Arc.** Bandit, then directed exploration, then MDPs, then the verifier task: the right order, and length is roughly proportional to importance for the first two parts (about 2,400 words). The MDP part (about 1,300 words) spends half its length on an experiment that shows nothing (EXP-5). The verifier part (about 600 words) is a different topic attached at the end (finding 3).

2. **Where each bound stands** (the caller asked for this).

   | Statement | Hypotheses given | Status |
   |---|---|---|
   | fixed $\epsilon$: linear regret | implicit (stationary arms) | derived by a two-line count; correct |
   | $\epsilon_t \propto 1/t$: logarithmic regret | $m$ arms, smallest gap $\Delta_{\min}$, $c > 5$ | counted for this instance, cited (Auer et al.) for the general case; hypotheses as in the paper; "rewards in $[0,1]$" omitted |
   | UCB1: logarithmic regret | rewards in $[0, 1]$, $\kappa = \sqrt{2}$; $\kappa > 1$ by Bubeck and Cesa-Bianchi | cited in the body; the bound itself appears only in exercise 2, as a guided proof; a heuristic for the count is given 100 lines later (EXP-3) |
   | Thompson sampling: logarithmic regret | none stated (the cited result is for Bernoulli rewards, which is the case here) | cited |
   | Lai and Robbins lower bound | "any rule that does well on every bandit of a family" | cited; hypothesis vague; applied at finite $t$, which is wrong (EXP-1) |
   | count-based Q-learning: $\sqrt{\textrm{steps}}$ regret | none (the result is for episodic finite-horizon MDPs) | cited, one clause |

   Nowhere does the body say that all of this assumes each arm's rewards are independent draws from a fixed distribution. One sentence in "Arms and Regret" would do, with a clause that a drifting arm needs a constant step size or a window, which connects to the constant $\alpha$ of the previous section.

3. **The two halves do not meet.** After 3,500 words on $\epsilon$-greedy, UCB, Thompson sampling, optimistic tables and count bonuses, the verifier part opens with "A language model explores such a task differently from the rules above." It does not say why. The reader deserves the judgment: a response is one of astronomically many, so there are no counts to build a bonus from and no posterior to sample; the policy's own sampling distribution is the only exploration the method has, which is why its temperature and the group size $K$ are the levers. Two sentences of that kind would turn the last part from an appendix into the section's conclusion. It also depends on a description of an update that :numref:`sec_baselines` derives two sections later; the three-sentence description here is enough to follow, so the forward reference is not load-bearing, but see EXP-7 for what the cell adds.

4. **The UCB argument is scattered.** The rule and its citation are under "Upper Confidence Bounds"; the reason regret is logarithmic (an arm is pulled until its bonus falls to about its gap) is under "The UCB Coefficient", introduced to explain a ratio; the theorem is in exercise 2. See EXP-3.

5. **Two endings** (EXP-10).

6. **Looked for and not found.** (a) The dependence of regret on the gaps: the bound is $\sum_a \log t/\Delta_a$, so near-best arms cost the most. The text says this only for the lower bound. (b) What a practitioner uses when rewards are not Bernoulli or arms are many: one clause on contextual bandits with a model is there; nothing on why Thompson sampling is the usual default in practice (no coefficient, handles delayed feedback). Not essential. (c) Boltzmann exploration is defined and drawn but never run or judged (EXP-9).

## Passage-level findings

**EXP-1** (major, correctness). Under "A Lower Bound on Regret".
Anchor: "about $\ln 2000 / 0.31 \approx 24$ times by round $2000$. With"
The Lai and Robbins theorem is asymptotic: for a rule whose regret is $o(t^p)$ for every $p > 0$ on every bandit of the family, $\liminf_t \mathbb{E}[n_t(a)]/\ln t \geq 1/D(\mu_a \,\|\, \mu^*)$. It gives the slope of the pull count against $\ln t$. It does not say how many pulls an arm must have received by a finite round, and the section's own results show that it cannot be read that way: panel (b) of :numref:`fig_rl_ucb` shows the arm with mean 0.60 pulled 11 times after 2,000 pulls; the leading term summed over arms, $\sum_a \Delta_a \ln 2000 / D(\mu_a \,\|\, 0.9)$, is 50 (my computation), while the stored median regret is 30 for Thompson sampling and 38 for UCB with $\kappa = 0.5$. A reader who multiplies 24 by 0.3 and compares with the plot will conclude that one of the two is wrong. High confidence. Rewrite as a statement about growth: "Lai and Robbins proved that a rule cannot have small regret on every Bernoulli bandit unless, in the long run, the number of pulls of each worse arm grows at least like $\ln t / D(\mu_a \,\|\, \mu^*)$. For the arm with mean $0.60$, $D = 0.31$: every tenfold increase in $t$ must bring at least $\ln 10/0.31 \approx 7$ more pulls of it. Summed over the nine worse arms, regret must grow by at least about 6.6 per unit of $\ln t$." Then the comparison becomes a finding and not a contradiction: in my $10^5$-pull runs UCB with $\kappa = 0.5$ grows by about 6 per unit of $\ln t$ between $10^4$ and $10^5$ pulls, at the Lai and Robbins slope, so on these arms there is almost nothing left to gain, and a smaller coefficient ($\kappa = 0.15$) gets below it only by giving up the guarantee. If the authors want that sentence, exercise 4 can supply the measurement. Also define "does well on every bandit of a family" (regret growing more slowly than every power of $t$, on every bandit with Bernoulli arms).

**EXP-2** (moderate, correctness). Under "Coverage on the Lake".
Anchor: "$\alpha = 0.9$ for $256$ episodes, and its greedy table solved the lake. We"
:numref:`sec_qlearning` ran that experiment for 512 episodes ("calm ice, 512 episodes (3,658 steps)" in its stored output, and "One eighth of the episode budget" in its prose). Either say "half as many episodes as" or align the two budgets. The schedule `linear_schedule(1.0, 0.05, 128)` is consistent with 256 within this section.

**EXP-3** (moderate, pedagogy). Under "Upper Confidence Bounds".
Anchor: "whose regret grows logarithmically in"
The reader is told that UCB1 has logarithmic regret and sent to an exercise for the bound. The one-line reason is in the section already, but under "The UCB Coefficient": "UCB keeps pulling a worse arm $a$ until its bonus $\kappa\sqrt{\log t / n(a)}$ has shrunk to about $\Delta_a$, that is for about $\kappa^2 \log t / \Delta_a^2$ pulls". Move that sentence here, directly after the Hoeffding paragraph, and complete it: each such pull costs $\Delta_a$, so the regret is about $\kappa^2 \log t \sum_a 1/\Delta_a$, logarithmic in $t$ and largest for the arms closest to the best. Then state the theorem's form in the body ("Auer et al. prove $\sum_a 8 \ln t/\Delta_a$ plus a constant for $\kappa = \sqrt{2}$") and keep the proof as exercise 2. The heuristic is better than the text lets on: for these arms at $t = 2000$ it gives 39 for $\kappa = 0.5$ (stored median: 38) and 314 for $\kappa = \sqrt{2}$ (stored: 204), while the theorem's bound is about 1,275. Saying so would give the reader a formula that predicts the experiment, in the spirit of the $\epsilon$-greedy arithmetic above it. "The UCB Coefficient" can then cite the count and go straight to the ratio 8.

**EXP-4** (minor, correctness). Under "The UCB Coefficient".
Anchor: "so its empirical mean is more accurate than the"
The explanation of why $\kappa = 0.5$ suffices (the best arm has variance 0.09, not the worst-case 0.25) is plausible and I believe it is the main effect, but it is stated as established. Scaling the theory's threshold $\kappa > 1$ by the ratio of standard deviations, $0.3/0.5$, gives $\kappa > 0.6$, still above 0.5; and the lower tail of a Bernoulli(0.9) mean, which is the tail that matters for the best arm, is its heavier one. Write "which is one reason a smaller bonus is enough on this problem", and keep "$\kappa = 0.5$ has no such guarantee". Unverified beyond the simulation: whether $\kappa = 0.5$ has logarithmic expected regret on these arms; the medians and means of 100 runs to $10^5$ pulls agree (58), so no failure is visible at that scale.

**EXP-5** (moderate, pedagogy). Under "Coverage on the Lake".
Anchor: "The lake is"
(The sentence continues on the next source line: "too small to show the difficulty of the chain.") The subsection asks "Does the lake need any of this?", runs three schedules over twenty seeds with a 30-line cell and a plot, finds that all three solve the lake in 19 runs of 20, and concludes that the lake cannot show the difficulty. The guide (section 9) asks that such an experiment be fixed or removed. What it does teach is one sentence: with $\epsilon = 1$ the agent learns a table that solves the lake while winning 2% of its own episodes, so learning and earning are different goals. Options: (a) cut the cell and the plot, keep that sentence with the two numbers, and let exercise 1 carry the rest; (b) run the same three schedules on slippery ice, where the section itself says "trying every pair once is not enough" and exercise 1(b) asks the reader to predict that the greedy rule fails. Option (b) would give a result in which the schedule matters, which is what the subsection title promises. Either way, two details need attention. The "19/20" under every schedule looks like a coincidence and is not: it is the same seed each time, the run whose random walk never reaches the goal in 256 episodes (a random episode on calm ice succeeds with probability 0.014, so that happens in about 3% of runs). Say so; it is the best evidence for "What limits all three schedules is the first success." And "random walks find it within a few dozen episodes" understates: at 1.4% per episode the mean wait is about 70 episodes.

**EXP-6** (minor, explanation). Under "A Chain of States".
Anchor: "The number of episodes grows"
(Continues: "polynomially in $N$ instead of exponentially.") The print shows 2, 3, 4, 6, 8 episodes for $N$ = 4 to 20, which the paragraph has already described as "roughly in proportion to $N$". The closing sentence restates it more weakly. Cut it, or make it the claim that is proved: exercise 5(b) argues the polynomial bound, so "exercise ... shows that this is no accident of these five sizes". Also note for the reader that this works because the chain is deterministic; the next subsection says it ("In a stochastic MDP one try says little"), and a clause here would prepare it.

**EXP-7** (moderate, pedagogy). Under "Exploration on the Verifier Task".
Anchor: "The sampled fractions agree with the formula. Half of all groups are"
The cell draws 1,000 groups per prompt to confirm $(1-p)^K + p^K$, a formula that needs no confirmation, and the first sentence of the reading is that the sample agrees. This is the confirming cell the author asked to limit. Print the formula's values directly (three lines of code, no sampling, no `rng`), per prompt and at both policies; the table is the content. The saved space can go to the point of finding 3. Also "Half of all groups are identical at every $K$" reads as "exactly half"; the printed means are 0.89, 0.67 and 0.51. Write "At least half of all groups are identical whatever $K$ is".

**EXP-8** (minor, thought). Under "Exploration on the Verifier Task".
Anchor: "is a *contextual bandit*: the agent faces one bandit per prompt and, with a"
The term is needed (:numref:`sec_rl_sequences` refers back to it), but the paragraph stops at the definition. One more sentence would make it useful: a table per prompt is ten separate bandits; a model that shares parameters across prompts is what makes the problem learnable when prompts never repeat, which is the case for a language model.

**EXP-9** (minor, focus). Under "Greedy and Epsilon-Greedy Rules".
Anchor: "*Boltzmann exploration*, makes the probability of each action proportional to"
Boltzmann exploration enters in the middle of the $\epsilon$-greedy subsection, written with $\hat Q(s, a)$ and illustrated by a figure about "the four moves of the lake", inside a section whose running example at that point is ten arms. It is not run on the bandit and not judged. Its real use is later (the softmax policies of :numref:`sec_policygradient` and the verifier's sampling). Either move the paragraph and :numref:`fig_rl_exploration` to the verifier part, where "sampling at a higher temperature" is the lever under discussion, or give it one line of judgment here (it has no logarithmic-regret guarantee for a fixed temperature, and its temperature must be tuned to the scale of the rewards). The symbol $\beta$ for the temperature is not in :numref:`tab_rl_notation` for this section, and the figure legend says "temperature 0.3" without the symbol.

**EXP-10** (minor, thought). End of "Exploration on the Verifier Task".
Anchor: "Every rule in this section learns which actions are good by trying them, and"
A recap paragraph before the Summary. Its last sentence is a good hinge to :numref:`sec_imitation` ("An expert who shows good actions would spare the agent all of this"); keep that and drop the two recap sentences before it.

**EXP-11** (minor, language). Under "Thompson Sampling".
Anchor: "regret :cite:`Agrawal.Goyal.2012`, and it has no constant to choose."
It has a prior, which exercise 3(d) shows acts like a constant (a $\mathrm{Beta}(9, 1)$ prior is optimistic initialization). "and its only choice is the prior, which the uniform one settles here" is accurate. The same phrase recurs at the end of "The UCB Coefficient" ("without a constant to tune").

**EXP-12** (minor, repetition). Opening.
Anchor: "one step in twenty, the optimal policy on the slippery lake succeeds about"
Third statement of 54% against 73% in two sections (twice in :numref:`sec_qlearning`). Fine here as the hook, if the closing of the previous section stops repeating it.

**EXP-13** (minor, language). Opening.
Anchor: "Returning to a favorite restaurant gives a good meal; a new one may give a"
The restaurant is the stock example of every bandit tutorial, and the section drops it after two sentences. Thompson's own problem in the next paragraph (which of two treatments the next patient receives) is sharper, has stakes, and is true. Open with it.

**EXP-14** (minor, language). Code under "UCB and Thompson Sampling Compared".
Anchor: "# play each arm once before pricing any"
"pricing" is an economic metaphor for "computing the bonus". Write "each arm once, since the bonus is undefined at count 0".

## Code and experiments

- `qlearning-regret-and-the-linear-tax`: stored medians 800, 106, 46. Prose: "around $800$", "about $100$", "about $50$". Consistent. The expected value 707 and the settle probability $0.90/4.87$ check.
- `qlearning-optimism-ucb-and-thompson-sampling`: stored 38 (UCB) and 30 (Thompson). Prose: "roughly $30$ to $40$" and "the difference between them is smaller than the spread over runs" (quartiles 29 to 46 and 26 to 36). Correct and properly cautious.
- `exploration-optimism-ucb-and-thompson-sampling-1`: stored 8, 38, 204 with maxima 709, 59, 241. Prose agrees. "This happens in roughly one run in ten" is not printed (only the maximum is); print the number of runs above 100 so the sentence reads the output.
- `exploration-a-chain-of-states-1`: stored values as quoted. The cell teaches (EXP-6 is about one sentence only).
- `exploration-exploration-in-mdps-3`: shows no difference between schedules, by the text's own account (EXP-5).
- `exploration-exploration-on-the-verifier-task`: confirms a formula by sampling; cannot fail except by sampling noise (EXP-7).
- The claims about $10^5$ pulls ("six to eight times", "both grow like $\log t$") are delegated to exercise 4 and are correct by my simulation.
- All cells are NumPy and shared by both tabs; the imports cell loads `torch`/`jax`, `nn`, `F`, `optax` and `math`, none of which is used. No TensorFlow or MXNet material.
- Cell IDs and two exercise labels still begin with `qlearning-` (`ex_qlearning-exploration-schedule`, `ex_qlearning-ucb-logarithmic-regret`), a trace of the split from the previous section. Invisible to readers; IDs are stable by rule. No action.

## Exercises

Six problems, well graded. Exercise 2 (derive the UCB1 bound in four guided parts, then ask "Which step of the proof loses the most?" and what happens for general $\kappa$) is excellent and should stay even after EXP-3 moves the heuristic into the body. Exercise 3 (Thompson's posterior, with the prior as optimism) and exercise 6 (identical groups, with the uniform-$p$ integral and the question of what the unanswerable prompts would need) are good. Exercise 1 is long but every part has a question. Exercise 4(a) (vectorize `bandit_run`) is programming work whose only purpose is to make part (b) affordable; acceptable. Exercise 5(a)'s bound $2^{N-2}/N$ is consistent with the printed exact values (8 against 23 at $N = 8$). One addition worth making: a problem that computes the Lai and Robbins slope for these arms and compares it with the measured growth of UCB at $\kappa = 0.5$ and $\sqrt{2}$ and of Thompson sampling, which is what EXP-1 turns the misapplied bound into.

## Still open from October

The October review's point about this material was that it sat inside `qlearning.md`. It is now its own section. Nothing else carries over.

## For the chapter review

- Fits between :numref:`sec_qlearning` (which hands it $\epsilon$-greedy and the 54% cost) and :numref:`sec_imitation` (which it motivates with "An expert who shows good actions would spare the agent all of this"). It takes `TabularMDP`, `q_learning`, `epsilon_greedy`, `linear_schedule` and `VerifierTask` from earlier sections. It feeds the pessimism penalty of :numref:`sec_offline` (which cites the UCB bonus) and the group update of :numref:`sec_baselines`.
- Against Sutton and Barto (chapter 2): this section is better on the question of cost. Sutton and Barto show reward curves on the ten-armed testbed and state no regret result; here regret is defined, predicted and measured, the lower bound is given, and the gap between the theory's constant and a tuned one is examined. Sutton and Barto are better on what a practitioner of value methods needs next: nonstationary arms and constant step sizes, and gradient bandits as the bridge to policy gradients. The first deserves a sentence here (structural finding 2); the second is covered by this chapter's later sections.
- Overlap: the fact that an identical group changes nothing is stated here and again in :numref:`sec_baselines` (one sentence there). That is the allowed size. If EXP-7 is taken, consider letting :numref:`sec_baselines` own the formula and this section keep the pass@$K$ paragraph.
- No dependence on the mathematics appendix. Hoeffding's inequality is quoted with its statement, which is the right amount; relative entropy is defined inline for Bernoulli variables. The appendix proves Hoeffding's inequality in :numref:`sec_mdl-concentration-generalization`; a clause could point there for the reader who wants the proof.
- Pattern shared with the neighbouring sections: unprinted frequencies in results prose (one instance here), a recap paragraph before the Summary, a full framework imports cell above pure NumPy code.
