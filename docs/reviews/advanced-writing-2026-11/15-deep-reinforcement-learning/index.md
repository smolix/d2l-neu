# Deep Reinforcement Learning (chapter introduction)

Source: `chapter_deep-reinforcement-learning/index.md` (lines 1 to 36, to the slides marker; about 440 words of introduction and 250 words of resources, no code). Read in full, together with the introduction of `chapter_reinforcement-learning/index.md` and all eight section reviews of this chapter.

## Verdict

The first two paragraphs are among the best chapter openings in the part. One number from the previous chapter (a pendulum that needed a few hundred thousand steps), two readers who cannot afford it (a robot arm, a language model), three things the last agent did not do, and a question in fourteen words: "How much can one sample teach, and what goes wrong when we make it teach more?" The second paragraph then gives four ideas, each with the error it brings. That is an author leading. The introduction falls short of the bar in its second half. The route is eight consecutive sentences of the form "X asks ...", the template the Basics review already flagged. The frame of four ideas and a count of environment steps fits five of the eight sections and leaves three (the KL penalty, offline learning, sequence generation) without a place. And the reader is not told what can be skipped, which sections depend on which, what the chapter needs from outside (the KL divergence, used from the second section on and defined nowhere in the main text), or what the two small control tasks can and cannot show. The single most important fix: replace the "asks" paragraph by a route with two lines of argument and a statement of what each reader needs.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought and focus | 3 |
| Pedagogy | 3 |
| Language | 3 |
| Ease and enjoyment | 4 |

## What works

- **The opening paragraph.** "A simulator makes that cheap. A robot arm would wear out first". The reader arrives with the result of `sec_deeprl` and is told why it is not good enough. Keep every sentence.
- **Each idea with its error.** "It can bound each update by the change in the policy's behavior ...; the bound limits each step, not where the steps lead." Four sentences of the same build, each with a clause that names the price. The chapter's best single line ("Each saving rests on an estimate that can be wrong", at the end of `rl-sequences.md`) is this paragraph in one sentence and belongs beside it (SEQ-14).
- **The citations carry history without a history paragraph**: Samuel 1959 and Sutton 1988 for the learned target, Lin 1992 for replay, Kakade and Langford 2002 for the bounded step, Sutton 1990 for the model. No "the idea is older than" sentence is needed, and none is written.
- **Candor in the resources.** rliable is recommended as "the tools to compare results over a few seeds more carefully than the counts and medians this chapter quotes". The section reviews show that this caution is needed in the text as well.
- **The resources are attached to sections** and each says what it adds. The division from the Further Reading page is stated in one sentence.

## Structural findings

1. **The frame covers five sections of eight.** The question is sample efficiency and the four ideas are a learned target, reused data, a bounded step and a learned model. `sec_actorcritic`, `sec_ppo`, `sec_dqn`, `sec_sac` and `sec_model_based` answer it and can be placed on one axis of environment steps. `sec_regularized` (how far to trust a learned reward), `sec_offline` (no new steps at all) and `sec_rl_sequences` (the update that post-trains language models) are about a different question: how far an estimate can be trusted away from the data behind it. That second question is in fact the one all eight sections share (a critic, a stale batch, a learned reward, a model, a log, a checker), and the model-based and offline reviews both name it as their section's organizing idea. The introduction should state both: the first paragraph's question for the control line, and one sentence that every method here replaces fresh experience by an estimate and that the chapter's recurring test is where that estimate was fitted.
2. **Two lines of argument are interleaved and the reader is not told.** `sec_actorcritic`, `sec_ppo`, `sec_regularized` and `sec_rl_sequences` are the policy line and end at language models. `sec_dqn`, `sec_sac`, `sec_model_based` and `sec_offline` are the value and data line and end at control from logs. `sec_regularized` sits between PPO and DQN because soft actor-critic needs its tilted optimum. A reader who came for post-training can read sections 1, 2, 3 and 8; one who came for control can read 1, 4, 5 and 6. Say so. This is what "what matters and what can be skipped" means for this chapter.
3. **Nothing on prerequisites.** One sentence: the chapter builds on the policy gradient with a baseline, Q-learning and the Bellman backup of the previous chapter, and it uses the KL divergence of two distributions from `sec_ppo` on. That divergence should then be defined where first used (PPO-6, REG-7); the introduction only needs to announce it.
4. **Nothing on what the experiments can show.** The paragraph on tasks says why CartPole and Pendulum were chosen ("so that their sample counts share one axis") and that the code "runs in CPU minutes". It does not say what the choice costs, and the section reviews show the cost in every control section: on CartPole bootstrapping does not help, on Pendulum neither the twin critics nor the learned temperature changes the return. One candid sentence here would replace the per-experiment apologies in the sections (see the soft actor-critic review, "For the chapter review"): these tasks are small enough to run many seeds and to know the best achievable value, and too easy for several of the mechanisms to change the return; where that happens the text measures the estimate the mechanism repairs, not the return.
5. **"Share one axis" promises more than the sections deliver.** The three CartPole sections count steps to different events: the batch mean return of a sampling policy reaching 475 or 500 (`sec_ppo`), a greedy policy first scoring 500 at a check every 5,000 steps (`sec_dqn`), and the batch mean reaching 450 (`sec_actorcritic`). See DQN-12 and PPO-14. Either make the measure the same in the sections or say "comparable" and not more.
6. **No chapter ending.** The chapter's closing paragraph currently sits at the end of `rl-sequences.md` (SEQ-14), where it recaps step counts that one section reviewer shows to be misread (PPO-14). A chapter of eight sections should close somewhere; the natural place is the first paragraph of the Further Reading page or a short closing in the introduction's own terms.

## Passage-level findings

**DIDX-1** (moderate, language). Third paragraph.
Quote: ":numref:`sec_ppo` asks whether one batch can drive several updates, and how far."
Eight consecutive sentences, each a section label followed by "asks". The questions themselves are well chosen (most are the section's real question, and the section reviews confirm that the sections answer them). The form is a list set as prose, with no ordering, no dependence and no judgment. Draft of a route with a voice: "The first two sections stay with the policy gradient: `sec_actorcritic` lets it learn before the episode ends, `sec_ppo` lets it use a batch more than once. `sec_regularized` steps aside to a question both raise, how far a policy may move from the data behind a learned reward, and gives the closed form that the rest of the chapter reuses. Then the data change hands: `sec_dqn` and `sec_sac` learn from stored transitions, `sec_model_based` from a model of them, `sec_offline` from a log that cannot grow. `sec_rl_sequences` returns to the policy line with tokens for actions and a checker for the reward."

**DIDX-2** (moderate, thought). Second paragraph.
Quote: "Together, a learned target, a network shared across states and another policy's data can make values diverge, which a table never does."
The deadly triad is attached to the four ideas as a fifth sentence with no connection to them, and the reader has no way to know that it is the subject of half of one section. Two smaller points. "which a table never does" is true of the tabular methods of the previous chapter under their step-size conditions; "which a table does not" is safer. And the triad's third ingredient is the first idea again (the learned target), which the sentence could say: "The first two ideas combined with a network can make values diverge".

**DIDX-3** (minor, explanation). First paragraph.
Quote: "a language model trained on its own answers cannot discard each batch of answers after one update"
The robot's reason is given (wear). The language model's is not: generation is the expensive part of a training step, so a batch of responses that took minutes to sample must be used for more than one gradient step. Half a clause.

**DIDX-4** (moderate, thought). Fourth paragraph.
Quote: "CartPole carries the three methods with discrete actions, so that their sample counts share one axis"
See structural finding 5. Also "the three methods" leaves the reader to work out which three (actor-critic, PPO, DQN).

**DIDX-5** (minor, correctness, confident). Fourth paragraph.
Quote: "the Language Models part treats post-training at scale, including learning from human preferences, in more depth"
An unlabelled forward reference to chapters that are skeletons today. It carries no weight in the argument, which is correct. But two sections of this chapter (`sec_offline`, `sec_rl_sequences`) and the Language Models plan each assume that the other teaches Direct Preference Optimization, and at present neither does (SEQ-11, OFF-10). Until that is settled this sentence promises "more depth" on something the chapter gives in four sentences.

**DIDX-6** (minor, language). Fourth paragraph.
Quote: "Several agents, hidden state, and meta- and hierarchical learning are left out"
Good that the omissions are named; the Further Reading page has a paragraph on exactly these three, so point to it here ("left out; `sec_deep_rl-further-reading` says where to read on").

**DIDX-7** (minor, thought). Fourth paragraph and the section title of `sec_rl_sequences`.
Quote: ":numref:`sec_rl_sequences` asks what survives when the environment appends tokens and a checker gives the reward."
"Reinforcement learning for sequences" has two meanings in this chapter: a policy that generates a sequence (`sec_rl_sequences`) and a sequence model that imitates logged trajectories conditioned on return (`sec_offline`, "Return Conditioning"). The introduction is where a reader looking for Decision Transformer should be told which section has it.

**DIDX-8** (minor, correctness, confident that the section omits it). Resources.
Quote: "at matched code-level choices PPO and TRPO perform nearly alike, so much of PPO's practical edge comes from details its objective does not mention"
The statement matches my knowledge of the study. The problem is placement: this is the most consequential published finding about PPO, and it appears in a resources bullet and in an exercise but not in `sec_ppo`, which cites the same paper for a smaller point (PPO-10, and structural finding 2 of that review). The Further Reading page states it a third time. One owner: the PPO section.

## Code and experiments

None in the body. The slides after the marker were not reviewed; a reviser should know that their table of environment steps repeats the PPO figure that PPO-14 disputes.

## Exercises

None, correctly.

## Still open from October

The October remark on cross-reference density applies to the route paragraph (eight labels in 110 words) and is resolved by DIDX-1.

## For the chapter review

- The introduction is the right place for three things the section reviews ask for independently: one statement about what CartPole and Pendulum can show (soft actor-critic review), the definition or at least the announcement of the KL divergence (PPO and KL-regularization reviews), and the disentangling of the two meanings of "sequence" (sequence-generation review).
- The opening paragraph and the "each with an error of its own" paragraph should be kept as they are and used as the model for the other chapter introductions of the part.
- Resources: Sutton and Barto's chapters 8 and 11 to 13 are cited by number, which is fine for another book. The Engstrom study and the catalog of 37 details each appear in three places across the two chapters' introductions and the Further Reading page.
