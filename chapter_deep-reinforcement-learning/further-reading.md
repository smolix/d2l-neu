# Further Reading
:label:`sec_deep_rl-further-reading`

Each section of this chapter taught one method and measured it on a small task. A reader who opens a recent paper meets variants of these methods, and most of them change one thing: a detail of the update, the data the update may use, or what it constrains. This page names the variants a reader is most likely to meet, in the order of the chapter, with the problem each addresses and what it changes. It ends with a project that builds the post-training loop of :numref:`sec_rl_sequences` in full.

## Policy Optimization in Practice

The PPO of :numref:`sec_ppo` reaches CartPole's ceiling, yet an implementation that reproduces published results on harder tasks differs from it in many small choices that papers rarely state. :citet:`Huang.Dossa.Raffin.ea.2022` traced 37 of them to lines of the reference code. Those that change the update most are these:

| detail | what it changes |
|:--|:--|
| observation and reward normalization | running statistics keep the network's inputs and the critic's targets near unit scale |
| advantage normalization per minibatch | the per-batch standardization of :numref:`sec_baselines`, applied to each minibatch |
| a learning rate annealed to zero | smaller steps as training ends |
| orthogonal initialization, small policy head | the first policy is nearly uniform, so the first ratios stay near 1 |
| a clip on the global gradient norm | a bound on every step in parameter space as well |
| one loss for actor and critic | policy, value and entropy terms share each minibatch pass, where :numref:`sec_ppo` fits the critic first |
| value-loss clipping | a clip band on the critic's change, shipped by some implementations and not by others |
| early stopping on the approximate KL | the epochs of a batch end once the measured KL passes a threshold |

Two studies measured which of these matter. :citet:`Engstrom.Ilyas.Santurkar.ea.2020` found that without them PPO's ratios leave the clip band freely, and that TRPO given the same details performs about as well as PPO, so much of PPO's advantage comes from the code around its objective. :citet:`Andrychowicz.Raichuk.Stanczyk.ea.2021` trained hundreds of thousands of agents over a grid of such choices and turned the results into recommendations: normalizing observations helped on most tasks, and neither study found value-loss clipping to help. Reading a maintained single-file implementation beside :numref:`sec_ppo` turns this table into code.

The PPO paper offers a second way to bound the step :cite:`Schulman.Wolski.Dhariwal.ea.2017`: a KL penalty on the change from the old policy, whose weight is doubled after a batch that overshot a target KL and halved after one that fell short. It did worse than the clip in their experiments, but it is a close relative of the penalty to a fixed reference in :numref:`sec_regularized`.

In distributed training, many actors collect data with copies of the policy that lag the learner by several updates. IMPALA's V-trace target :cite:`Espeholt.Soyer.Munos.ea.2018` corrects the $n$-step targets of :numref:`sec_actorcritic` with importance ratios truncated at one. The truncation keeps the variance bounded, and in exchange the targets estimate the value of a policy between the actors' and the learner's.

## Value Learning after DQN

The DQN of :numref:`sec_dqn` needs tens of thousands of CartPole steps, and its maximum over noisy estimates overestimates. Its successors each change one part. A multi-step target puts the $n$-step return of :numref:`sec_actorcritic` inside the Q-learning target, so a reward reaches earlier states in fewer updates. Prioritized replay :cite:`Schaul.Quan.Antonoglou.ea.2016` samples transitions in proportion to their TD error and reweights them to undo the bias this introduces. A dueling network :cite:`Wang.Schaul.Hessel.ea.2016` splits the output into a state value and per-action advantages, so the shared value learns from the transitions of every action.

Distributional methods learn the whole distribution of the return rather than its mean, with a Bellman equation for distributions :cite:`Bellemare.Dabney.Munos.2017`. :citet:`Lyle.Castro.Bellemare.2019` showed that in tabular and linear settings the expected values they learn are those of the ordinary update, so their gains come from how the richer target trains a deep network. Rainbow :cite:`Hessel.Modayil.vanHasselt.ea.2018` combined six of these changes, the double estimator of :numref:`sec_dqn` among them. Its ablation on Atari ranked prioritized replay and multi-step targets as the most important parts, and the double estimator and the dueling head as the least.

:citet:`vanHasselt.Doron.Strub.ea.2018` measured how often the deadly triad of :numref:`sec_dqn` causes trouble in deep Q-networks trained under many settings. Unbounded divergence was rare, but estimates that rose for a while above any achievable return were common; multi-step targets and target networks made such episodes rarer. Two recent designs drop the repairs of :numref:`sec_dqn` altogether. PQN :cite:`Gallici.Fellows.Ellis.ea.2025` trains on batches from many parallel environments with layer normalization in the network, and uses neither a replay buffer nor a target network. Streaming methods :cite:`Elsayed.Vasan.Mahmood.2024` learn from each transition once, as it arrives, after normalizing observations and rewards and bounding the size of each update. Replay and a frozen copy are therefore one repair of correlated data and moving targets, not the only one.

## Continuous Control since 2018

TD3 :cite:`Fujimoto.vanHoof.Meger.2018` made three changes to DDPG :cite:`Lillicrap.Hunt.Pritzel.ea.2016`, whose deterministic actor climbs its critic's errors: the twin critics that :numref:`sec_sac` adopted, an actor updated less often than the critics, and noise on the target's action, so that the actor cannot exploit a narrow peak of the critic.

Most progress since has come from taking more gradient steps per environment step, the update ratio of :numref:`sec_sac`, without the critic's estimates running away. REDQ :cite:`Chen.Wang.Zhou.ea.2021` keeps a large ensemble of critics and takes the minimum over a random pair of them, which sustains many updates per step. DroQ :cite:`Hiraoka.Imagawa.Hashimoto.ea.2022` replaces the ensemble by dropout and layer normalization in two critics. CrossQ :cite:`Bhatt.Palenicek.Belousov.ea.2024` is the exception: it matches REDQ at one update per step and has no target networks; its critic uses batch normalization and passes the current and the next state-action pairs through the network in one batch, so that both see the same statistics. BRO :cite:`Nauman.Ostaszewski.Jankowski.ea.2024` scales the critic up to a wide residual network with layer normalization and adds optimistic exploration. The common thread with PQN is normalization inside the critic.

## Models, Imagination and Search

The planner of :numref:`sec_model_based` spends thousands of model evaluations per real step and trusts its model only near the data; the methods below use a learned model differently. Dyna :cite:`Sutton.1990` uses it to imagine extra transitions for a model-free learner, the counted model of :numref:`sec_qlearning` driving Q-learning updates between real steps. MBPO :cite:`Janner.Fu.Zhang.ea.2019` keeps the imagined rollouts short, a few steps branching from states in the replay buffer, and mixes them into soft actor-critic's batches, so the model's compounding error stays small. REDQ, from the previous topic, matched MBPO's sample efficiency without a model, which suggests that much of MBPO's gain came from more updates per real step.

PETS :cite:`Chua.Calandra.McAllister.ea.2018` fits an ensemble of probabilistic networks and plans through samples from it with the cross-entropy method. The members agree where the data is dense and disagree elsewhere, so their disagreement marks where a plan should not be trusted. Dreamer :cite:`Hafner.Pasukonis.Ba.ea.2025` learns a recurrent model of the dynamics in a learned latent space, from images, and trains its actor and critic entirely on imagined latent trajectories, with one setting of its hyperparameters across many domains.

Search with a learned value closes the loop of :numref:`sec_model_based`. AlphaZero :cite:`Silver.Schrittwieser.Simonyan.ea.2017,Silver.Hubert.Schrittwieser.ea.2018` searches with the rules of the game as its model and a value network at the leaves, and distills the search's choices into the policy. MuZero :cite:`Schrittwieser.Antonoglou.Hubert.ea.2020` learns a model that predicts only rewards, values and policies, not observations, and searches in it. TD-MPC2 :cite:`Hansen.Su.Wang.2024` plans over short horizons in a latent model and closes each plan with a learned value.

A learned model also helps when no new data may be collected, provided the offline learner distrusts it where the data is thin. MOPO :cite:`Yu.Thomas.Yu.ea.2020` and MOReL :cite:`Kidambi.Rajeswaran.Netrapalli.ea.2020` penalize imagined rewards where an ensemble of models disagrees: the pessimism of :numref:`sec_offline`, applied to a model.

## Offline Methods at Scale

The penalty of :numref:`sec_offline` shrinks with the square root of the count of a state-action pair in the log. With continuous actions almost every action a learned policy proposes has a count of zero, so the deep offline methods replace the count by something a network can estimate. They differ in what they constrain:

| method | what it constrains | how |
|:--|:--|:--|
| BCQ :cite:`Fujimoto.Meger.Precup.2019` | the actions the policy may take | a generative model of the logged actions proposes candidates, and the policy picks among them |
| CQL :cite:`Kumar.Zhou.Tucker.ea.2020` | the values | a loss term lowers $Q$ on the policy's actions and raises it on the logged ones |
| IQL :cite:`Kostrikov.Nair.Levine.2022` | which actions the backup queries | $V$ is fitted to an upper expectile of $Q$ over logged actions only, and the policy is extracted by advantage-weighted regression |
| TD3+BC :cite:`Fujimoto.Gu.2021` | the policy's distance from the data | a cloning term is added to TD3's actor loss |

Return-conditioned sequence models, the other family, are discussed in :numref:`sec_offline` together with their limits.

Many applications have a log and can also collect a little new data. Cal-QL :cite:`Nakamoto.Zhai.Singh.ea.2023` calibrates the pessimistic values so that they never fall below those of a reference policy such as the logging policy, which removes the drop in return that overly low values otherwise cause when online fine-tuning begins. RLPD :cite:`Ball.Smith.Kostrikov.ea.2023` skips offline training and runs an off-policy learner online from the start, with half of every batch drawn from the log and with the normalized critics of the previous topics.

## Post-Training Variants

:numref:`sec_rl_sequences` assembles the post-training update from the group weight of :numref:`sec_baselines`, PPO's clipped ratio and the KL penalty of :numref:`sec_regularized`, and names DAPO, Dr. GRPO and GSPO. Trained at the scale of reasoning models :cite:`Shao.Wang.Zhu.ea.2024,DeepSeekAI.2025`, the update met problems that the one-token task cannot show, and the variants answer them. DAPO :cite:`Yu.Zhang.Zhu.ea.2025` also averages the loss over all tokens of a batch rather than first within each response, so that a long response counts in proportion to its length, and it shapes the reward of responses cut off at the length limit. Dr. GRPO :cite:`Liu.Chen.Li.ea.2025` traces two biases to the divisions it removes: dividing by the response length shrinks the per-token penalty of long wrong answers, so training lengthens them, and dividing by the group's standard deviation gives the most weight to prompts whose groups are nearly all right or all wrong. GSPO :cite:`Zheng.Liu.Li.ea.2025` was introduced for the stability of large mixture-of-experts models, whose per-token ratios are especially noisy.

Process rewards change where the reward comes from. :citet:`Lightman.Kosaraju.Burda.ea.2024` trained a reward model on human labels of individual reasoning steps, and it judged solutions better than a model trained on final answers alone. A learned process reward gives each step its own credit, as the prefix values of :numref:`sec_rl_sequences` do, and brings with it the exploitable errors of a learned reward that :numref:`sec_regularized` measured.

## Beyond the Chapter

Three assumptions of these two chapters can fail. When several agents learn at once, each one's environment contains the others, and nothing is stationary; self-play combined with search reached superhuman play in heads-up no-limit poker :cite:`Brown.Sandholm.2017`. When the agent sees observations rather than the state, the Markov property of :numref:`sec_mdp` fails, and the policy needs memory, a recurrent network or a belief over states :cite:`Kaelbling.Littman.Cassandra.1998`. When tasks change, an agent can learn to adapt to a new task from a few episodes (meta-learning), or learn sub-policies that act over many steps together with a policy that chooses among them (hierarchical learning). Murphy's overview :cite:`Murphy.2025` surveys all three.

## A Project

:numref:`sec_rl_sequences` assembles the post-training update and applies it to a single group. This project builds the training loop around it and checks it against what the chapter can compute exactly.

1. **The task.** Use the four prompts of the verifier task, answered with two digit tokens as in :numref:`sec_rl_sequences`, so that `04`, `12`, `64` and `81` are the correct responses. The policy is a table of preferences for the tens digit per prompt and for the units digit per prompt and tens digit; the reference is uniform.
2. **The update.** Extend `grpo_step` to two-token responses. For each prompt, sample a group of 16 responses, store their token log-probabilities, and standardize the rewards within the group. Then run $E$ epochs of gradient ascent with step size 1 on the clipped per-token surrogate with $\epsilon_{\textrm{clip}} = 0.2$, averaged over the tokens of the group, minus $\beta$ times the $k_3$ term of :numref:`sec_rl_sequences`; record the clip fraction of each epoch.
3. **The checks.** In the first epoch every ratio equals 1 and the clip fraction is zero, so with $E = 1$ the update is the REINFORCE step with group-standardized weights.
4. **The bar for reuse.** With $\beta = 0$ and the same step size, four epochs per group should reach a success rate of 0.9 with about half the sampled responses that one epoch needs, on every seed you try, while the clip fraction rises from the first epoch to the fourth.
5. **The bar for the penalty.** On the one-token verifier task with the lenient checker of :numref:`sec_rl_sequences`, use the leave-one-out weight of :numref:`sec_baselines` and $\beta = 1/2$, and train until the policy settles. Under the $k_3$ loss the hedge should keep about half of the probability on `8*8`, the maximizer of the objective with the KL in the other direction (exercise :exref:`ex_rl-sequences-kl-loss`); with the penalty folded into the reward it should stay under two percent, the tilted optimum. Explain why the leave-one-out weight is needed for these two numbers, and what group standardization would change.
