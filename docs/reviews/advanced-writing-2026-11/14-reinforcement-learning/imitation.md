# Learning from Demonstrations

Source: `chapter_reinforcement-learning/imitation.md` (lines 1 to 719, up to the slides marker). About 4,100 words of body prose plus 1,700 words of exercises; 13 code cells, of which 5 exist in separate PyTorch and JAX versions (about 230 lines per tab); one pre-generated figure (`mdl-rl-compounding-error.svg`, rendered and inspected).

## Verdict

The ideas are right, the proposition is correctly stated and correctly proved in ten lines, and the section is unusually honest about its own experiment. That honesty is also its problem. The section's subject is compounding error: a per-step error on the expert's states can cost order $\varepsilon T^2$, on the learner's own states order $\varepsilon T$, and DAgger moves a learner from the first regime to the second. The reader never sees that happen to a learned policy. The quadratic curve is drawn from two formulas on an assumed chain. On the lake, the text itself computes that "distribution shift accounts for about a fifth of the clone's loss", DAgger lifts success from 0.66 to 0.735 against an expert at 1.0, and three paragraphs are spent explaining artifacts of the setup (an imposed 18% error, Adam's normalization, a seed-dependent median). The single most important fix is the stage: replace or supplement the six-move lake with a task on which a fitted clone's loss visibly grows faster than linearly in the horizon and DAgger visibly straightens it. The second is the 190 lines of policy-container code and tab notes that stand between the reader and the first result.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought and focus | 3 |
| Pedagogy | 3 |
| Language | 4 |
| Ease and enjoyment | 3 |

(The opening, "Compounding Error" up to the end of the proof, and "Supervised Fine-Tuning as Behavior Cloning" are 4 to 5. "The Policy Container" and "Fitting the Clone" are a 2 for enjoyment. The experiments are a 3 for pedagogy.)

## What works

- **The opening.** Pomerleau's van is the right example and it is used for its content, not as decoration: "The driver kept the van in the middle of the lane, so the recordings contained no view of the roadside and no example of steering back." The section's whole argument is in that sentence, and the next one is quotable: "A classifier is tested on inputs like those it was trained on, but a policy chooses the inputs it sees next." Protect both. The historical claims (single hidden layer, the 1991 version trained in minutes by watching a driver, synthesized shifted views) match what I know of the two papers.
- **The proposition and its proof.** Part (i) by coupling to the first disagreement, part (ii) by hybrids that switch to the expert after $k$ steps. Both arguments are complete, short, and say where each hypothesis is used. The paragraph after the proof reads the result as a person would: "Part (i) charges a first mistake with everything that follows, because afterwards the policy is in states where nothing controls its error", and "If one mistake is fatal, $u$ is as large as the remaining horizon and part (ii) gives order $\varepsilon T^2$ again." This is the level of Cover and Thomas: a theorem, a proof one can check, and a sentence that says what it means.
- **Two failures kept apart.** "Missing coverage disappears as expert data accumulate. Distribution shift does not, because the states the clone reaches through its own mistakes are states the expert never visits." Most treatments blur these.
- **The decomposition.** Making the off-path cells absorbing to split the 34 points into "24 are falls from the path itself ($0.828 \times 0.285$) and 10 are failures after leaving it ($0.172 \times 0.606$)", and then asking what labels could recover (7.5 points) is careful, exact analysis. I reproduced every number in it with an independent implementation (0.660, 0.172, 0.715, 0.394, 0.735, 0.216, and the 14 Adam steps). The candour of "The rest is the 18% error itself, on the path and off it, which no change of data removes" is admirable; see IL-1 for what it implies.
- **Exact evaluation.** `visitation` propagates the state distribution, so no result in the section has sampling noise. The one Monte Carlo line (0.656 against 0.660) is a legitimate cross-check of the method, used once.
- **Supervised fine-tuning as behavior cloning**, with $u$ given a meaning for text: "little for an unusual but harmless word, a lot for a wrong digit early in a calculation that every later step builds on." And on-policy distillation as "DAgger with the teacher as the expert on call". Current and correct.
- **The hinge to the next section**: "Without an expert, the cloning update still applies if the weight $w_i = 1$ is replaced by a number that says how well the action turned out." This justifies the weighted form of `policy_step` and should survive any restructuring.
- Figure `fig_rl_compounding_error` is clear and its caption's numbers match the computation (holes 5 and 12 take 0.17 and 0.12 of the 0.34).

## Structural findings

1. **The running example shows the section's phenomenon as the minor effect.** Three properties of the calm $4 \times 4$ lake work against the lesson. The path is six moves, so $\varepsilon T^2$ and $\varepsilon T$ are not far apart. Leaving the path is not costly enough: a clone acting uniformly at random off the path still reaches the goal 39% of the time. And the reward is a single terminal one, so the proposition's bound ($\varepsilon T(T+1)/2 = 3.8$ at $\varepsilon = 0.18$, $T = 6$) is vacuous there and the text never applies it to the lake. On top of that, the stopping rule of `clone` holds every labeled row at 18% error, on the aggregated data as well, so DAgger's ceiling is fixed by construction at "the clone's error at every cell" (0.735), far from the expert. I checked whether the longer path of exercise 6 rescues the picture. On the standard $8 \times 8$ calm lake with the same clone (my computation, 14-move path) the clone succeeds 34% of the time and leaves the path in 57% of its episodes, so the share of the loss incurred after leaving the path rises to about two thirds; but a learner with the same error at every cell reaches only 48%, so labels can again recover only about a fifth of the loss. The lake family cannot show DAgger turning a quadratic loss into a linear one. See IL-1 for the redesign.

2. **Infrastructure before the idea.** From "### The Policy Container" to the end of the `policy_step` cells the source has about 190 lines of which perhaps 30 concern imitation. The reader of a section on demonstrations is taken through a context manager for PyTorch's generator, `inference_mode`, an embedding table, three NumPy conversion methods, and, in the JAX tab, bucket padding, a shared Adam and XLA kernel determinism. The clone that all this fits is a $16 \times 4$ table with six trained rows. See IL-2.

3. **Checks that cannot fail, and one result told five times.** The clone is evaluated on "100 fresh expert episodes" of a deterministic expert in a deterministic environment, and clones of 1, 10 and 100 identical episodes are fitted to print 0.660 three times, after the text has already explained why they must be equal (IL-3, IL-4).

4. **DAgger's guarantee is described by its hypotheses and not stated.** The body lists what the guarantee assumes and whom it concerns ("the best learner of the sequence"), and leaves what is guaranteed to exercise 8 (IL-5). The caller asked what DAgger guarantees and whether it is proved: in this section it is cited, its conclusion is missing from the body, and exercise 8 walks through the key inequality of the paper's proof.

5. **Where each bound stands.**

   | Statement | Hypotheses given | Status |
   |---|---|---|
   | Proposition (i), upper bound $\varepsilon T(T+1)/2$ | $T$ steps, rewards in $[0, 1]$, disagreement at most $\varepsilon$ under $d^{\pi^*}_t$ for every $t$ | proved in full; correct |
   | Proposition (i), tightness "whenever $\varepsilon T \leq 1$" | as stated | asserted; the chain that follows is such a task but the text does not say so (IL-6) |
   | Proposition (ii), $u \varepsilon T$ | disagreement at most $\varepsilon$ under $d^{\hat\pi}_t$; one wrong action then the expert costs at most $u$ | proved in full; correct |
   | Averaged $\varepsilon$ still gives $\varepsilon T^2$ | stated | one line, correct |
   | DAgger | "many rounds", no-regret learner, mixture policy, expert on call | cited; conclusion not stated (IL-5) |

6. **Order and length.** The order (clone, shift, missing coverage, proposition, decomposition, DAgger, language models) is right. With the container moved or slimmed, the section would be about 3,500 words, which is proportionate.

7. **Looked for and not found.** (a) The name the language-model literature uses for this phenomenon, exposure bias, and its companion, teacher forcing, which the reader met in :numref:`sec_rnn` (IL-8). (b) Demonstrations that disagree with each other (two valid ways round an obstacle): the most common practical failure of behavior cloning after shift, present only as exercise 4. One sentence in the body pointing at the exercise would be enough. (c) Any statement of when plain behavior cloning is the right choice. It is the default in robotics and in language-model training because experts on call are rare; the text says it "is the most common way to learn from an expert today" and never says that it usually works well enough when data are plentiful and $u$ is small. A judgment of that kind would fit the voice the guide asks for.

## Passage-level findings

**IL-1** (major, pedagogy). Under "Decomposition of the Clone's Loss" and "Dataset Aggregation".
Anchor: "On this lake distribution shift accounts for about a fifth of the clone's loss."
The sentence is true and it tells the reader that the experiment is about something else. What the reader should take from the section (loss growing like $T^2$ for a clone and like $T$ for a learner trained on its own states) is shown only by the chain cell, where a "cloned imitator" is defined to earn nothing after its first mistake and a "recovering imitator" to lose one reward; nothing is learned there. Direction for the development machine: build the chain as a real `TabularMDP` and let the section's own `clone` and `dagger` produce the two curves. For instance: a corridor of $T$ on-path states that pay one per step under the expert's action; a wrong action moves the agent to an off-path strip from which one action leads back and the others lead further away, with no reward off the path. The expert, asked, returns in one step, so $u$ is small. A clone has uniform rows off the path and drifts away, so its first mistake costs most of what remains. Evaluate both exactly with `visitation` for $T$ = 8, 16, 32, 64 at a fixed fit error (say $\varepsilon = 0.05$), and plot the lost return of behavior cloning, of DAgger after a few rounds, and the two references. That is one experiment showing the proposition's two regimes emerge from fitting, and it replaces the formula-only cell `imitation-the-proposition`. The lake can then keep what it does well in a shorter form: the 0.66, the figure, and the split of the loss, presented as "on a short task the shift is the smaller part". If the redesign is not wanted, the minimum is to say plainly in "Dataset Aggregation" that the gap DAgger closes here is small because the task is short and the fit error is imposed everywhere, and to stop there, without the explanations of IL-7.

**IL-2** (moderate, pedagogy). Under "The Policy Container".
Anchor: "`Policy` holds a network `policy` that maps a state to one score per action"
What can go, with reasons:
- `seeded` and the `seed` argument of `tabular`. The table is zero-initialized, so the seed has no effect on anything in this section, and the PyTorch tab note says as much ("the networks that this and later constructors build"). Move the class to :numref:`sec_utils` or to the first section that initializes a network at random.
- `act_greedy` and `log_prob_np` as separately narrated methods, and the sentence "It follows the acting protocol of `d2l.evaluate`" ("acting protocol" is not a term the reader has). The greedy clone in this section is computed by `pi_bc.argmax`, not by `act_greedy`.
- The JAX tab note of about 90 words ("`act` and `act_greedy` run the network through `d2l.forward_np`; `log_prob_np`, and `policy_step` below, pad their batch to a bucket length ...") and the note on `d2l.Embed` and atomic additions. The guide says framework plumbing belongs in code comments or in the Computation chapters, and tab prose is "at most a sentence or two".
- The first JAX cell (`imitation-setup`), which opens a section on demonstrations with a six-line comment on XLA kernel selection. If determinism flags are needed they belong in the library's import path or in :numref:`sec_utils`.

What must stay: the `Policy` class with `log_prob`, `act` and `tabular`; `policy_step` with its weights; and the sentence "The weights arrive as a NumPy array, that is, as data, and no gradient flows into them. For behavior cloning they are all one", because the next section's derivation lands on this function. Aim: one `Policy` cell and one `policy_step` cell per tab, introduced by a paragraph each, about 60 lines in place of 190. Consider also moving "The Reduction to Classification" paragraph that states the log-likelihood to directly before `policy_step`, so that the objective and the code that ascends it are adjacent.

**IL-3** (moderate, pedagogy). Under "Training and Deployment".
Anchor: "On fresh expert episodes the clone errs as often as on its training pairs"
On calm ice the expert walks the same six states in every episode, so the "fresh" episodes are the training pairs repeated and the two printed numbers (0.182 and 0.182) are equal by construction. The conclusion "so there is no generalization gap: the demonstrations cover the expert's distribution" is a fact about the setup, not a measurement. Likewise the loop that clones 1, 10 and 100 episodes prints 0.660 three times, which the text predicted fifty lines earlier ("a clone of one expert episode and a clone of a hundred are the same table"). Both are checks that cannot fail. Remove the fresh-episode line and the loop from cell `imitation-what-the-reduction-quietly-assumes-2`; keep the one comparison that carries information (0.182 on the expert's states, 0.216 on the clone's own) and say in a clause that on calm ice every expert episode is the same, so there is nothing to generalize over. The cell shrinks to five lines.

**IL-4** (minor, thought). Pattern, 5 statements of one fact.
Anchors: "so a clone of one expert episode and a clone of a hundred are the same table"; "Here more expert data cannot help: it adds no new state"; "More demonstrations cannot fix the shift, because the expert never visits"; "Behavior cloning with any number of expert labels stays at 0.66"; "more expert data cannot change the clone"
The last is in the Summary and stays. Of the four in the body keep the third, which is the one that motivates DAgger, and let the first be the place where it is explained. Cut the other two.

**IL-5** (moderate, thought). Under "Dataset Aggregation".
Anchor: "concerns the best learner of the sequence, or one drawn from it at random, not the last one"
The reader is told what the guarantee is about and what it assumes, and not what it says. Add it, in the terms of the proposition: "If the learners have no regret on the aggregated losses, then after enough rounds (a number of order $uT$, up to logarithms) some learner of the sequence disagrees with the expert on its own states about as rarely as the best policy in the class does on the aggregated data, at a rate $\varepsilon_N$. Part (ii) of the proposition then applies to it, and its loss is about $u T \varepsilon_N$: linear in the horizon." This is my recollection of Theorem 3.2 of the paper (existence of a policy in the sequence with loss $u T \varepsilon_N + O(1)$ for $N$ of order $uT$ up to logarithms); verify the constants against the paper before printing. Then the caveats that follow in the text (best of the sequence, mixture, expert on call) qualify a statement the reader has seen, and exercise 8 proves its core inequality. Also "Keeping the data of all earlier rounds prevents the fitted policy from depending only on its most recent trajectories" explains aggregation after the guarantee has been discussed; it belongs where the algorithm is introduced, as the reason for the "A" in DAgger (training on the latest round alone can oscillate).

**IL-6** (minor, explanation). Proposition and the chain.
Anchor: "It never rises above the reference, as part (i) requires."
Two small gaps. The proposition asserts tightness ("there are tasks in which the shortfall is of this order whenever $\varepsilon T \leq 1$") and the proof covers only the upper bound. The chain of the next paragraph is such a task, so say so: replace the quoted sentence, which reports a check that cannot fail, by "The chain is therefore a task that attains the order of the bound while $\varepsilon T$ is small, which is the second claim of part (i)." And the proposition speaks of a policy that "disagrees with the expert with probability at most $\varepsilon$" without saying over what: for a sampling policy and a deterministic expert it is the probability, over the state and the policy's own draw, that the action differs from the expert's. One clause, since the section's clone is a sampling policy and the reader may think of an argmax.

**IL-7** (moderate, explanation). Under "Dataset Aggregation".
Anchor: "A learner lands on the 0.735 exactly, not approximately: Adam divides each"
Three sentences explain why a tabular learner under Adam and this stopping rule ends at the same 0.818 in every labeled row. It is correct (I reproduced the 0.818 and the 14 steps with a hand-written Adam, on both the 6-state and a 14-state path), and it is an explanation of the apparatus, not of DAgger. The guide's test applies: "If an experiment needs a paragraph of confounders before its result can be read, it is a research experiment." The same paragraph has "the median over ten seeds reaches 0.70 to 0.735, depending on which ten", which is hard to parse (ten seeds were run; which other tens?) and contradicts the print for round 4 (0.735, range 0.717 to 0.735) unless it describes earlier rounds. Rewrite the reading as: "After four rounds and a few hundred labels the median success is 0.735, and every seed is above behavior cloning's 0.66. That is the success of a learner with the clone's 18% error at every cell: DAgger has removed the part of the loss that came from unlabeled cells and, by construction of our stopping rule, nothing else." Move the Adam remark to a code comment in `clone` or to exercise 6(c), which already asks about labels per state.

**IL-8** (minor, currency). Under "Supervised Fine-Tuning as Behavior Cloning".
Anchor: "so its contexts drift from those of the curated responses as a response grows"
The language-model literature calls this exposure bias, and calls training on the expert's prefixes teacher forcing, a term the reader met with the RNN language model. Name both in a clause so that the reader can connect this section to what they will read elsewhere. The subsection would also be stronger for one sentence of judgment on how much the worst case matters for text: the $\varepsilon T^2$ bound describes tasks without recovery, and whether long generations behave that way depends on $u$, which the text has just said is small for most tokens and large for a few. As written the paragraph states the mapping and leaves the reader to guess whether the bound is a real concern. The forward reference to the Language Models part is not load-bearing here; the paragraph stands alone.

**IL-9** (minor, thought). End of "Supervised Fine-Tuning as Behavior Cloning".
Anchor: "Behavior cloning turns an expert's recorded choices into a policy with nothing more than"
A recap paragraph before the Summary. Its first two sentences repeat the section; its last two are the hinge praised above. Keep "Both need an expert ..." onward. The sentence before it, on offline reinforcement learning ("Behavior cloning is also the standard first baseline in offline reinforcement learning"), is an unrelated pointer appended to the distillation paragraph; move it to the first subsection or drop it, since :numref:`sec_offline` introduces its own baseline.

**IL-10** (minor, explanation). Under "Missing Coverage" and "Dataset Aggregation".
Anchor: "at or near the 0.735 of a learner with the clone's error at every cell"
The section carries two unrelated numbers that look alike: 0.730, the expert's success on slippery ice (printed in the coverage cell as "below half the expert (0.730)"), and 0.735, the success on calm ice of a learner with 18% error everywhere. A reader skimming back will take them for the same quantity. In the coverage cell, print the threshold as a rate ("below 36.5%") or name the lake in the output line.

**IL-11** (minor, explanation). Under "Fitting the Clone".
Anchor: "Fourteen Adam steps bring the disagreement to 0.182."
The cell does not print the number of steps. It is right; either have `clone` report it or drop "Fourteen". In the same paragraph, the reader's natural question is answered a subsection later and only by analogy: if the greedy clone is perfect ("The expert reaches the goal every time, and so does the greedy clone"), why sample? Say here, where the 18% is introduced, that sampling stands in for the errors a real function approximator makes at some states whatever the decoding rule, and that it is also how a language model is deployed.

**IL-12** (minor, language). Pattern, 2 instances.
Anchors: "one preference $\theta_{s,a}$ per state--action pair"; "to an expert's state--action pairs by cross-entropy"
The double hyphen appears here and nowhere else in the chapter, which writes "state-action". Make it uniform.

**IL-13** (minor, explanation). Proposition.
Anchor: "Consider an episodic task of $T$ steps with rewards in $[0, 1]$ at every step"
The steps are indexed $t = 1, \ldots, T$ in the proposition and proof and $t = 0, 1, \ldots$ in :numref:`sec_mdp` and in the recursion for $d^\pi_t$ a page earlier ("It starts with all mass on the start state"). Harmless for the result; state the convention in the proposition ("steps $t = 1, \ldots, T$") so that $T - t + 1$ is not a puzzle.

## Code and experiments

- Stored outputs agree with every number the prose quotes: 18 pairs and states [0 4 8 9 13 14]; rows 0.818 and 0.061; 1.000, 1.000, 0.660; Monte Carlo 0.656; disagreement 0.182, 0.182, 0.216; coverage 32.0%, 1.5%, 0.0%; chain 17.03 against 20.80 and 0.64; decomposition 0.172, 0.715, 0.394, 0.735; DAgger rounds 0.660, 0.710, 0.717, 0.735, 0.735 with minimum 0.649 in round 1.
- `imitation-what-the-reduction-quietly-assumes-2`: two of its three parts cannot fail (IL-3).
- `imitation-the-proposition`: evaluates two closed forms and plots them against the bound. It does not test the proposition (it cannot violate it); its value is showing tightness and the saturation as $\varepsilon T$ grows. "within 3% at $T = 10$ and 10% at $T = 30$" is correct (I get 2.9% and 9.0%) and unprinted. To be replaced by a fitted experiment under IL-1.
- `imitation-dataset-aggregation-1`: the result is read correctly but needs three sentences about Adam to be read at all (IL-7), and the effect is small by design (IL-1).
- `imitation-distribution-shift-between-training-and-deployment-1` (missing coverage): a good small experiment; the shortcut of writing the greedy table down "instead of fitting 600 clones" is said openly.
- The cell ID `imitation-what-the-reduction-quietly-assumes-*` preserves a heading with "quietly", one of the words the guide removes; the heading itself is gone. No action.
- Tabs: the source carries PyTorch and JAX tabs for `Policy`, the three acting methods and `policy_step`, and `outputs/jax/` holds a store for all three sections. (The assignment described the chapter as PyTorch-only; the chapter introduction says "The code is in PyTorch and JAX", and that is what the sources contain.) The two tabs fragment the prose only in "The Policy Container", where the JAX notes are three times as long as the PyTorch ones (IL-2). No TensorFlow or MXNet material.

## Exercises

Nine problems, strong and varied. Exercise 1 (the chain in closed form, with the second-order term and the question of when the quadratic regime ends) is exactly the companion the chain cell needs. Exercise 2's claim that the total variation "exceeds one half" around $t = 6$ while the off-path mass is about 5% is correct (I get 0.69 and 0.050), and its part (b), on why total variation at $t = 6$ is a poor measure of shift, is a subtle and worthwhile point. Exercise 5 (coverage; the state with $q_s = 0$ is state 3) and exercise 6 (the $8 \times 8$ lake) are good; exercise 6(a) asks "Why does the first grow with the length of the path while the second does not?", which is the question structural finding 1 raises, so the authors know the limitation. Exercise 8 (DAgger's guarantee) is the hardest and holds the statement the body lacks (IL-5); after that fix it becomes a proof exercise, as it should be. Exercise 9 (inverse reinforcement learning and shaping) links back to :numref:`subsec_vi_shaping` neatly. Exercise 7 (noisy expert) is the weakest: part (a) is quick and parts (b) and (c) are a rerun; keep it but it could lose part (c). Exercise 4 (two valid paths) carries a topic the body should at least name (structural finding 7b). Nothing to cut outright.

## Still open from October

October's only note on this file was a "keep": "the first deviation from the expert (`imitation.md:306`)". The coupling argument is still there, now as part (i) of the proof, and is still the best paragraph of the section's theory.

## For the chapter review

- The section is the bridge from value methods to policy methods: it introduces the parametric policy $\pi_\theta$, the `Policy` container and `policy_step`, and the state distributions $d^\pi_t$, all of which :numref:`sec_policygradient` and :numref:`sec_deeprl` reuse by reference. That is why the container is here; it is also why the section reads as two sections interleaved, one on imitation and one on library code. A chapter-level decision is needed on where the container lives (here in slim form, at the start of the policy-gradient section, or in :numref:`sec_utils`).
- It takes `TabularMDP`, `value_iteration` and `evaluate` from the first two sections, and the token MDP from :numref:`sec_mdp`. It does not depend on :numref:`sec_qlearning` or :numref:`sec_exploration` except for the opening sentence. No appendix dependence.
- Sutton and Barto do not treat imitation learning at all, so there is no direct comparison; the fair ones are the Berkeley and Stanford course notes the chapter introduction lists, against which the theory here is as good and more carefully proved, and the experiment weaker (those courses show the drift on a driving or locomotion task where it is dramatic).
- Skeleton chapters of the Language Models part plan to cite `subsec_il_sft`, `subsec_il_dagger` and `subsec_il_error_bounds`; the labels should survive the revision.
- Patterns shared with the neighbouring sections: a recap paragraph before the Summary; one fact restated several times in results prose (here "more data cannot help", there the 54%); experiments whose prose must explain the apparatus. Pattern specific to this section and probably to :numref:`sec_policygradient` and :numref:`sec_deeprl`: long tab notes about JAX compilation and determinism.
