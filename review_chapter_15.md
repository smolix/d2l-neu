# Style review: Chapter 15, Deep Reinforcement Learning

## Scope

Reviewed every tracked Markdown source in `chapter_deep-reinforcement-learning`:
`index.md`, `actor-critic.md`, `ppo.md`, `dqn.md`, `offline-rl.md`,
`regularized.md`, `rl-sequences.md`, and `sac.md`. The review covers prose,
headings, mathematics, captions, code and experiments, summaries, exercises, and
all slide blocks. Line references are to the current sources.

## Executive assessment

The chapter has strong technical ambitions and an unusually valuable emphasis on
auditable diagnostics: target leakage, clipping, calibration, replay, termination
semantics, and change-of-variables checks are treated as first-class topics. Yet
the writing is much less controlled than the underlying material. Very long
paragraphs routinely combine a derivation, implementation choices, experimental
results, literature history, and rhetorical interpretation. Captions sometimes
become essays. Metaphors such as *price*, *buy*, *collapse*, *trap*, *survivor*,
and *heart* create drama where exact statistical language would be clearer. The
sequence-model and regularization sections are especially compressed. The
revision should preserve the chapter's empirical skepticism while giving each
formal result, algorithm, and experiment its own explanatory unit.

## Scores

| Dimension | Score | Assessment |
|---|---:|---|
| Writing | 5/10 | Dense, often vivid, but overloaded paragraphs, essay-length captions, slogans, and recurring metaphors impede comprehension. |
| Explanation | 7/10 | Excellent local insights and diagnostics; too many are packed together without hierarchy or intermediate summaries. |
| Technical | 8/10 | Broad and thoughtful coverage, with several assumptions and off-policy claims needing tighter qualification. |

## Architecture and order

The value/policy sequence is reasonable, but the chapter currently reads as four
different books: online actor-critic/PPO, DQN/SAC implementation case studies,
offline/regularized objectives, and sequence-model formulations. Add a roadmap
that declares these axes. Keep algorithm derivations before implementation
diagnostics. Move general offline-policy-evaluation and coverage language ahead
of offline algorithms. Either expand `rl-sequences.md` into a proper synthesis or
reduce it to a carefully scoped map; its current density does not support its
breadth. The regularization material should distinguish entropy bonuses, KL
constraints, control-as-inference, and preference objectives before relating them.

## Issue inventory

| ID | Severity | Evidence | Excerpt or description | Violated rule | Diagnosis and concrete revision |
|---|---|---|---|---|---|
| C15-01 | Medium | `chapter_deep-reinforcement-learning/index.md:1-80` | The short overview lists algorithms but does not establish the chapter's distinct problem axes. | Begin with a concrete problem and provide architectural signposting. | Add a roadmap separating action space, data source, update type, policy constraints, and sequence formulation; map every section to it. |
| C15-02 | Medium | `chapter_deep-reinforcement-learning/actor-critic.md:89-103` | One caption is exceptionally long; another says “nothing waits.” | Captions should be concise and prose should avoid absolutes. | Keep plot setup, axes, and observed lag in the caption. Move the training-loop explanation into the body and state which update is synchronous. |
| C15-03 | High | `chapter_deep-reinforcement-learning/actor-critic.md:305-363` | Moving targets, update order, and an observed “collapse” are discussed together. | Separate mechanism, evidence, and diagnosis. | Define the moving-target mechanism, state the tested intervention, report run variability, and name the measured failure rather than “collapse.” |
| C15-04 | High | `chapter_deep-reinforcement-learning/actor-critic.md:499-512` | A dense interpretation ends with the “noisy road” still reaching the same “arrival.” | One paragraph should carry one claim; avoid journey metaphors. | Split return behavior, critic error, policy entropy, and seed variability into separate observations, then give a bounded synthesis. |
| C15-05 | High | `chapter_deep-reinforcement-learning/ppo.md:127` | A caption calls the surrogate a “liar far away.” | Avoid anthropomorphism and theatrical captions. | State that the first-order surrogate becomes inaccurate after a large policy change; identify the plotted approximation error and policy distance. |
| C15-06 | High | `chapter_deep-reinforcement-learning/ppo.md:139-156` | A long passage combines asymmetric clipping, advantage sign, implementation, and says “nothing left to derive.” | Separate algebra, interpretation, and code; avoid totalizing claims. | Use a four-case table for ratio/advantage signs, derive the clipped objective, then explain remaining optimization and estimation choices. |
| C15-07 | High | `chapter_deep-reinforcement-learning/ppo.md:244-311` | The clipped objective is called the “heart”; a later paragraph combines estimator details and training guidance. | Use descriptive terms and one main explanatory job per paragraph. | Call it the policy surrogate, then separate GAE, normalization, minibatch reuse, stopping, and evaluation. |
| C15-08 | High | `chapter_deep-reinforcement-learning/ppo.md:388-456` | “Runaway” updates and “folklore” frame empirical safeguards. | Experiments and heuristics need explicit status and evidence. | Label clipping, value clipping, entropy bonuses, and KL stopping as heuristics; cite sources and state which local ablation supports each. |
| C15-09 | High | `chapter_deep-reinforcement-learning/dqn.md:67-94` | The prose anthropomorphizes audits and describes instability as a “cliff.” | Avoid anthropomorphism and manufactured drama. | Name target leakage or extrapolation error directly and show the condition under which the update becomes unstable. |
| C15-10 | High | `chapter_deep-reinforcement-learning/dqn.md:238-391` | Several very long paragraphs combine replay design, target networks, ablations, and causal conclusions. | Separate algorithm specification, experimental protocol, result, and limitation. | Factor the material into a reference update, implementation invariants, controlled ablations, and a results table with seeds and uncertainty. |
| C15-11 | Medium | `chapter_deep-reinforcement-learning/dqn.md:410-422` | A “one unit” bias appears “from nothing” and a fix “costs nothing.” | Avoid absolutes and economic idioms. | Derive maximization bias from noisy estimates and state the additional estimate/evaluation required by Double Q-learning. |
| C15-12 | High | `chapter_deep-reinforcement-learning/offline-rl.md:30` | The opening caption says ratios are “buying” something. | Captions should identify variables and evidence without metaphor. | State how importance ratios correct distribution mismatch, what variance they induce, and the assumptions in the pictured example. |
| C15-13 | High | `chapter_deep-reinforcement-learning/offline-rl.md:131` | A very long caption includes setup, fitted envelope, interpretation, and caveats. | Keep captions concise and move analysis to body prose. | Retain dataset, axes, and key comparison in the caption; create a body subsection for extrapolation error and estimator limitations. |
| C15-14 | High | `chapter_deep-reinforcement-learning/offline-rl.md:1-200` | Off-policy Q-learning is sometimes described as needing no policy correction without equally prominent coverage/function-approximation conditions. | Claims require scope and limitations. | Distinguish the Bellman target's lack of an importance ratio from convergence under adequate coverage and stable approximation. |
| C15-15 | Medium | `chapter_deep-reinforcement-learning/offline-rl.md:400-495` | A slide titled “The Rule” is vague. | Titles should describe the content. | Retitle with the actual support or conservatism rule and state the failure it prevents. |
| C15-16 | High | `chapter_deep-reinforcement-learning/regularized.md:54-105` | “As one always should,” “priced trade,” “buys,” and a one-sentence Goodhart reference surround a dense derivation. | Avoid prescriptions, economic metaphors, and compressed allusions. | State the regularized objective and reference-policy role neutrally; explain reward misspecification separately with a scoped example. |
| C15-17 | High | `chapter_deep-reinforcement-learning/regularized.md:263-300` | A posterior interpretation says “everything transfers wholesale”; a callout says “never”; reverse-KL and literature lineage are compressed into large paragraphs. | Avoid universal claims; distinguish identity, modeling interpretation, and historical relation. | State the exact variational identity and assumptions, qualify which inference tools transfer, then separate reverse-KL geometry and literature context. |
| C15-18 | High | `chapter_deep-reinforcement-learning/rl-sequences.md:39-72` | Several enormous paragraphs move from return-to-go conditioning to contextual bandits “wearing RL clothes,” survival claims, and a long caption. | Introduce one abstraction at a time and avoid anthropomorphism/evolutionary drama. | Split sequence representation, causal masking, conditioning variables, and distribution-shift limits; make the caption describe the tensor mapping only. |
| C15-19 | High | `chapter_deep-reinforcement-learning/rl-sequences.md:106-190` | Decision Transformers, DPO, and GRPO are connected in a highly compressed chain. | Do not imply equivalence from shared notation; distinguish formal identity from analogy. | Give each objective separately, state its data-generating assumptions, then add a comparison table for conditioning, normalization, reference model, and policy interpretation. |
| C15-20 | High | `chapter_deep-reinforcement-learning/sac.md:102` | The KL/maximum-entropy bridge “costs nothing to cross.” | Prefer literal mathematical interpretation to economic metaphor. | State that the objectives differ by a policy-independent log-partition term and explain what changes under a restricted parametric family. |
| C15-21 | High | `chapter_deep-reinforcement-learning/sac.md:129-140` | Two long paragraphs combine pathwise gradients, boundary atoms, change of variables, implementation conventions, and an empirical failure claim. | Separate derivation, numerical implementation, and observation. | Use separate subsections for why clipping is invalid, tanh density, stable log determinant, and log-standard-deviation bounds; qualify the “within a thousand updates” claim with its setup. |
| C15-22 | High | `chapter_deep-reinforcement-learning/sac.md:285` | “Everything else” is said to transfer unchanged from the replay buffer. | Avoid universal shorthand in implementation descriptions. | Enumerate inherited invariants and changed fields, including action dtype/shape and termination handling. |
| C15-23 | High | `chapter_deep-reinforcement-learning/sac.md:403-458` | A long loop explanation, “the quantity the objective buys,” and a three-seed ablation support broad interpretation. | Protocol and evidence must be explicit and conclusions proportional. | Split termination semantics from entropy diagnostics; report all run settings and uncertainty, and present the single-critic comparison as a small diagnostic. |
| C15-24 | High | `chapter_deep-reinforcement-learning/sac.md:507-519` | Entropy “pays,” “spends,” and “buys”; a very long paragraph interprets trace, convention, alpha tuning, and reward scaling. | Avoid extended economic metaphor; one paragraph should not perform four analyses. | Separate trace observation, differential-entropy convention, fixed-alpha limitation, and reward-scale dependence. Use equations/units for alpha. |
| C15-25 | Medium | `chapter_deep-reinforcement-learning/sac.md:647-841` | Exercise and slide language asks why “nothing” needs a ratio and ends with “self-confirming collapse.” | Slides and exercises require precise scope. | Ask why this particular Bellman target is formed without trajectory importance ratios, then state replay-coverage and approximation caveats; name the comparison failure. |

## Mathematics and notation

- Put policy, behavior-policy, data-policy, and reference-policy notation in one
  table and preserve the distinctions across PPO, offline RL, SAC, and DPO.
- State the assumptions behind performance-difference/TRPO-style bounds before
  motivating PPO; clearly label clipping as a surrogate heuristic, not a bound.
- Separate exact variational identities from approximate parametric projections
  and from empirical optimization procedures.
- For squashed distributions, keep action scale, Jacobian convention, and units
  of entropy/temperature explicit. The existing density-integration check is a
  strong model for equation validation.
- Treat termination and truncation notation consistently in every bootstrap.

## Figures, captions, and slides

Many figures are excellent diagnostics, but captions repeatedly absorb the
entire analysis. Restrict captions to setup, axes, comparison, and observed
result; move mechanisms, conventions, and limitations into nearby paragraphs.
Avoid caption personification (“liar”) and economic language. Slides should not
inherit dramatic phrases from prose and should display assumptions alongside the
objective or update they qualify.

## Code and experiment pedagogy

The chapter's assertions, density normalization checks, target audits, and
calibration plots are exemplary. Preserve them. Standardize all stochastic runs
with environment/version, preprocessing, horizon, termination convention, seeds,
number of runs, evaluation protocol, dispersion, and compute budget. Three-seed
local experiments are diagnostics, not general rankings. Every ablation should
change one mechanism and retain a shared correctness check.

## Recurring artifacts

- Essay-length paragraphs and captions joining derivation, code, result, history,
  and advice.
- “Price/pay/buy/costs nothing” used for entropy, variance, compute, and bias.
- “Collapse/trap/cliff/runaway/survivor” used instead of named failure modes.
- “Nothing/everything/never/always” used beyond the stated assumptions.
- Formal equivalence, suggestive analogy, and empirical effectiveness presented
  too close together.

## Positive patterns to preserve

- Algorithms are repeatedly audited against silent implementation errors.
- Termination versus truncation and density conventions receive concrete checks.
- Calibration and ablation figures try to test mechanisms, not merely report
  returns.
- The chapter connects policy objectives to executable updates and measured
  diagnostics.

## Prioritized revision plan

1. Split every oversized paragraph/caption, beginning with PPO, DQN,
   regularization, sequence RL, and SAC entropy analysis.
2. Add one notation/evidence map separating data policy, behavior policy, target
   policy, reference policy, and current policy.
3. Qualify off-policy and variational claims; distinguish identities, bounds,
   heuristics, and experiments.
4. Standardize stochastic protocols and bound three-seed diagnostic conclusions.
5. Expand or sharply narrow `rl-sequences.md`; do not leave multiple objectives
   connected mainly by analogy.
6. Rewrite slides and perform a chapter-wide restraint pass for economic,
   catastrophic, and universal language.
