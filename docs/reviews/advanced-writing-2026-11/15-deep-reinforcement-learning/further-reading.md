# Further Reading

Source: `chapter_deep-reinforcement-learning/further-reading.md` (86 lines, the whole file; about 2,250 words, two tables, a five-step project, no code and no slides). Read in full. Claims about papers were checked against what I know with certainty; those I could not confirm are marked unverified.

## Verdict

This is a good further-reading page, better than most in the part, because it has a principle and keeps to it: "most of them change one thing". Each method is given with the problem it addresses and the one thing it changes, in the order of the chapter, and several paragraphs end on a judgment the reader can use ("The common thread with PQN is normalization inside the critic"; "Replay and a frozen copy are therefore one repair of correlated data and moving targets, not the only one"). The two tables (PPO's implementation details, the deep offline methods by what each constrains) are the right form. It is current to 2025. Its weaknesses: it does not rank, so about forty-five works in seven topics arrive at equal weight and the reader is not told which three to read; it holds material that the sections need and lack (the PPO details that decide results, the KL-penalty variant of PPO, in-sample offline learning); a handful of statements about papers are looser than the papers; and the project at the end, a good one, sets pass marks that rest on the two experimental readings the section reviews dispute. On the question put to this review: the page does not serve the previous chapter. Every topic follows a section of this chapter, and the previous chapter's subjects with a large literature of their own (exploration with networks, imitation beyond DAgger, the natural gradient) have no entry.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought and focus | 4 |
| Pedagogy | 3 |
| Language | 4 |
| Ease and enjoyment | 4 |

## What works

- **The framing sentence**: "A reader who opens a recent paper meets variants of these methods, and most of them change one thing: a detail of the update, the data the update may use, or what it constrains." It tells the reader how to read the page and how to read the papers.
- **The offline table.** Four methods, one column "what it constrains" (the actions, the values, which actions the backup queries, the distance from the data). A reader who has this table can place a new offline method in a minute.
- **Reported measurements, not reputations.** Rainbow's ablation "ranked prioritized replay and multi-step targets as the most important parts, and the double estimator and the dueling head as the least" (correct, and it puts the Double DQN of `sec_dqn` in proportion, which the section itself does not; see the DQN review, structural finding 7). The deadly-triad study: "Unbounded divergence was rare, but estimates that rose for a while above any achievable return were common" (correct, and it is the qualification that DQN-8 asks the section to make).
- **The REDQ remark on MBPO**: "which suggests that much of MBPO's gain came from more updates per real step". A judgment, with its evidence.
- **V-trace in two sentences**, including what the truncation costs: "the targets estimate the value of a policy between the actors' and the learner's". Correct.
- **Dr. GRPO by its mechanism**: dividing by the response length "shrinks the per-token penalty of long wrong answers, so training lengthens them". This is the one-thing-it-changes form that `rl-sequences.md` lacks for the same methods.
- **The project** extends the chapter's last update to two tokens and checks it against quantities the chapter can compute exactly. The first check ("In the first epoch every ratio equals 1 and the clip fraction is zero") is a real unit test with a stated reason.

## Structural findings

1. **No ranking.** The page is organized by topic and inside each topic by genealogy. Nothing says what to read first, and the seven topics are of very unequal use to the two kinds of reader the chapter has (control, post-training). Two cheap repairs. First, open each topic with the one work to read if the reader reads one, and why: the catalog of implementation details for PPO, the Rainbow ablation for value learning, REDQ or CrossQ for the update ratio, Dreamer or TD-MPC2 for models, IQL for offline. Second, say at the top which topics serve which reader. The sentences that already rank ("Those that change the update most are these"; Rainbow's ordering) show the tone to use.
2. **Material that belongs in the sections.** Three items here are needed where the method is taught, and the section reviews ask for each independently:
   - The table of PPO details and the finding of the two controlled studies. The PPO review (structural finding 2, PPO-10) wants one paragraph of this in `sec_ppo`, with a statement of which details the section's loop has and which it lacks. The table can stay here; the finding must be in the section.
   - The KL-penalty variant of PPO. PPO-18 points out that the section quotes a result about "its variant with neither a clip nor a penalty" without having said that a penalty variant exists. One clause there, the description here.
   - IQL. The table of `sec_offline` has two families and this table has four rows; the offline review (structural finding 3) asks for the in-sample row there. The two tables must agree on the families.
3. **Variants of the post-training update have two owners.** `rl-sequences.md` names DAPO, Dr. GRPO and GSPO, gives each a clause or a subsection, and this page then adds to each ("DAPO ... also averages the loss over all tokens"). The sequence-generation review (SEQ-5, SEQ-10, structural finding 2) asks for the variants to leave the teaching section. This page should own them completely, each in the form it already uses for Dr. GRPO, and the section should keep one sentence. The Language Models plan (`docs/llm/language_models_part_plan.md`, the cross-reference table) lists `sec_rl_sequences` as the owner of "DAPO clip bands, GSPO ratio, Dr. GRPO divisor"; if they move here that table needs the new label.
4. **Does the page serve the previous chapter? No.** The evidence: the six topic headings map one to one onto `sec_ppo`, `sec_dqn`, `sec_sac`, `sec_model_based`, `sec_offline` and `sec_rl_sequences`; the previous chapter is cited only as the origin of a part (`sec_baselines` for standardization and the group weight, `sec_qlearning` for the counted model, `sec_mdp` for the Markov property). Only "Beyond the Chapter" speaks of "these two chapters". The previous chapter's introduction has a list of textbooks, courses and implementations and no account of the literature, and its map of methods (`tab_rl_map` in `deep-rl.md`) already sends readers here for Rainbow, MuZero and Dreamer. What a reader of that chapter would look for and not find: exploration with networks (count-based and prediction-error bonuses, posterior sampling with ensembles), which `sec_exploration` stops short of; imitation beyond cloning and DAgger (inverse reinforcement learning, adversarial imitation, large behavior-cloned policies), which `sec_imitation` stops short of; and the natural policy gradient, which this chapter's third section rediscovers without naming (REG-14). Recommendation: add one topic, "Exploration and Imitation with Networks", of two paragraphs in the page's own form, change the first sentence to speak of both chapters, and add the page to the previous chapter's introduction as its further reading. A separate page for the previous chapter is not needed.
5. **The page could carry the chapter's ending.** The chapter has no closing (see the introduction review, structural finding 6). The sentence that the sequence-generation review wants moved out of its section ("Each saving rests on an estimate that can be wrong: a critic, a stale batch, a model, a log or a checker") would make the first paragraph here say what the methods below have in common, which is the same thing.

## Passage-level findings

**DFR-1** (minor, correctness, confident). Heading: "Value Learning after DQN".
Quote: "Rainbow :cite:`Hessel.Modayil.vanHasselt.ea.2018` combined six of these changes, the double estimator of :numref:`sec_dqn` among them."
Rainbow's six are the double estimator, prioritized replay, the dueling network, multi-step targets, the distributional target and noisy networks for exploration. The page has described five; noisy networks are not mentioned, so "six of these changes" counts one the reader has not met. Add a clause ("and a sixth, noise in the network's weights in place of $\epsilon$-greedy exploration").

**DFR-2** (minor, correctness, unverified). Heading: "Policy Optimization in Practice".
Quote: "found that without them PPO's ratios leave the clip band freely, and that TRPO given the same details performs about as well as PPO"
The second half matches my knowledge of the study. For the first, my recollection agrees with the PPO reviewer's (PPO-10): the study's trust-region measurements are of the mean KL divergence and the largest ratio with and without the code-level optimizations, and its conclusion is that the optimizations, not the clip, decide how well the trust region is held. "Freely" is stronger than I can confirm. Check the paper and word this sentence and the one in `sec_ppo` the same way.

**DFR-3** (minor, correctness, unverified). Same heading.
Quote: "and neither study found value-loss clipping to help"
I am confident of this for the large-scale study (it recommends against value-loss clipping). I cannot confirm that the other study isolates value-loss clipping; I recall it as one item in the bundle of optimizations it switches on and off together. If so, write "the larger study found that value-loss clipping did not help".

**DFR-4** (minor, language). Same heading.
Quote: "Reading a maintained single-file implementation beside :numref:`sec_ppo` turns this table into code."
Good advice without an address. The previous chapter's introduction names the implementation (CleanRL); name it here too.

**DFR-5** (minor, thought). Heading: "Continuous Control since 2018".
Quote: "Most progress since has come from taking more gradient steps per environment step"
The paragraph opens with this thesis and closes on another ("The common thread with PQN is normalization inside the critic"), and its own examples support the second at least as well as the first: CrossQ is introduced as "the exception" that works at one update per step, and BRO's change is a larger normalized network. State one thesis that covers all four: sample efficiency was gained by training the critic harder (more updates, or a larger network), and what made that possible each time was a way to keep its estimates from running away (an ensemble, dropout, normalization).

**DFR-6** (minor, correctness, fairly confident). Heading: "Models, Imagination and Search".
Quote: "MOPO :cite:`Yu.Thomas.Yu.ea.2020` and MOReL :cite:`Kidambi.Rajeswaran.Netrapalli.ea.2020` penalize imagined rewards where an ensemble of models disagrees"
This fits neither exactly. MOPO subtracts from the reward an uncertainty estimate, in practice the largest predicted standard deviation among the ensemble's members. MOReL does use disagreement, but as a threshold: state-action pairs where the models disagree by more than a set amount lead to an absorbing state with a large negative reward. "MOPO lowers imagined rewards by the model's estimated uncertainty, and MOReL ends an imagined rollout with a large penalty where an ensemble of models disagrees" is accurate at the same length.

**DFR-7** (minor, correctness, confident). Heading: "Beyond the Chapter".
Quote: "self-play combined with search reached superhuman play in heads-up no-limit poker :cite:`Brown.Sandholm.2017`"
Correct. For a book of 2026 the one example of multi-agent learning could be a later and broader one (six-player poker, or a game that mixes language and strategy); as it stands the paragraph's newest multi-agent result is nine years old. A matter of currency, not of error.

**DFR-8** (moderate, correctness of interpretation, unverified by a run). Heading: "A Project", step 4.
Quote: "four epochs per group should reach a success rate of 0.9 with about half the sampled responses that one epoch needs, on every seed you try"
A pass mark stated as a ratio and "on every seed", with no cell in the chapter that produces it. It is also the comparison two section reviewers found confounded: four epochs at step size 1 are four gradient steps where one epoch is one, so "half the samples" may measure the step size and not the reuse (PPO-16 for CartPole, SEQ-13 for this very update). The "about half" echoes the PPO section's reading of its own reuse experiment, which the stored plots do not support (PPO-14). Fix: give the bar as what must hold by construction (the clip fraction is zero in the first epoch and rises; four epochs move the policy further per group than one), and pose the samples-to-0.9 comparison as a question with the control included (one epoch at step size 4), reporting whatever range the authors measured.

**DFR-9** (minor, pedagogy). "A Project", step 5.
Quote: "Under the $k_3$ loss the hedge should keep about half of the probability on `8*8`"
The two targets (about one half; under two percent) are the 0.50 and 0.017 of `rl-sequences.md`, which the reviewer recomputed and found correct but which no cell prints (SEQ-12). The reviewer's point there is that training the two penalties side by side on the hedge is "the payoff of the whole section and it is missing". It is here, as the last step of a project on a further-reading page. Move the experiment into the section and let the project keep the two-token extension.

**DFR-10** (minor, explanation). "A Project", step 5.
Quote: "Explain why the leave-one-out weight is needed for these two numbers, and what group standardization would change."
A good question whose answer the section half gives away and then withdraws: `rl-sequences.md` attributes a gap of 0.02 from the optimum to standardization in a sentence the reviewer asks to cut (SEQ-8). If that sentence goes, this question becomes the place where the reader finds the effect, which is the better arrangement. Keep.

## Code and experiments

No cells. The project's numerical bars are discussed in DFR-8 and DFR-9.

## Exercises

None. The project is the page's exercise. It is well specified in its task, update and first check, and belongs at the end of `rl-sequences.md` or in that section's exercises, where a reader looking for practice will find it. A project under "Further Reading" is easy to miss.

## Still open from October

Not covered in October.

## For the chapter review

- The page is the natural owner of three things now spread over sections: the variants of the post-training update (from `rl-sequences.md`), the successors of DQN (which `dqn.md` should point to in one sentence), and TD3's target smoothing (which `sac.md` alludes to without naming, SAC-10).
- It states twice what sections state from unseen runs: REDQ against MBPO here supports the model-based section's judgment, and is sounder evidence than that section's own offstage Pendulum run (MB-12).
- It does not serve the previous chapter (structural finding 4), and a small addition would make it do so.
- Claims I checked and believe correct: the catalog of 37 details; the adaptive KL penalty of the PPO paper (doubling and halving around a target); V-trace's truncation at one; prioritized replay, dueling and distributional methods; the expected-value equivalence result for distributional methods in tabular and linear settings; PQN and the streaming methods; TD3's three changes; REDQ, DroQ, CrossQ and BRO; PETS, MBPO, Dreamer, AlphaZero, MuZero and TD-MPC2; BCQ, CQL, IQL and TD3+BC; Cal-QL and RLPD; DAPO, Dr. GRPO and GSPO; the process-reward study. Not checked: publication years in the bibliography keys, and the two points marked unverified above.
