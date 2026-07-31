# Style review: Chapter 14, Reinforcement Learning

## Scope

Reviewed every tracked Markdown source in `chapter_reinforcement-learning`:
`index.md`, `mdp.md`, `value-iter.md`, `qlearning.md`, `policy-gradient.md`,
`baselines.md`, `imitation.md`, and `deep-rl.md`. The review includes prose,
headings, mathematical exposition, captions, code and experiments, summaries,
exercises, and all slide blocks. Line references are to the current sources.

## Executive assessment

The chapter is substantially clearer than its earlier generated form. It now
develops MDPs, dynamic programming, temporal-difference learning, policy
gradients, variance reduction, imitation, and deep function approximation in a
recognizable technical sequence. Many derivations are concrete and several
figures diagnose failure rather than decorate the text. Remaining problems are
concentrated: a few legacy claims overstate the universality of value iteration,
some compressed paragraphs combine theorem, algorithm, and empirical advice,
and slide blocks retain slogan-like fragments. The revision should protect the
current structure while tightening claims, expanding the thin off-policy bridge,
and standardizing experimental evidence.

## Scores

| Dimension | Score | Assessment |
|---|---:|---|
| Writing | 7/10 | Mostly direct and coherent; isolated slogans, absolutes, and overloaded paragraphs remain. |
| Explanation | 8/10 | Good conceptual progression and examples, with several transitions compressed too aggressively. |
| Technical | 8/10 | Core mathematics is sound in presentation, but scope conditions and empirical protocols need a more systematic audit. |

## Architecture and order

The chapter order is effective. The main architectural gap is between tabular
Q-learning and deep RL: off-policy sampling, target bias, coverage, and the
deadly-triad interaction receive less development than their later importance
requires. Expand that bridge before deep function approximation. Keep imitation
learning as a clearly marked change of information source rather than another
step in the value/policy sequence. A short roadmap should distinguish planning
with a known model, learning values from interaction, direct policy optimization,
and learning from demonstrations.

## Issue inventory

| ID | Severity | Evidence | Excerpt or description | Violated rule | Diagnosis and concrete revision |
|---|---|---|---|---|---|
| C14-01 | High | `chapter_reinforcement-learning/value-iter.md:70` | Value iteration is called “very powerful,” foundational, and effectively the basis of all RL. | Avoid hype and universal lineage claims. | State its exact role: dynamic programming for finite MDPs with known transitions/rewards, and a source of Bellman ideas used by later methods. |
| C14-02 | Medium | `chapter_reinforcement-learning/value-iter.md:93-102` | “Remainder optimal” is introduced as a mnemonic, followed by a formula that “hides a fact” and drives the “whole algorithm.” | Prefer explicit conditions and direct interpretation. | State the induction invariant and Bellman optimality step plainly; remove concealment and totalizing language. |
| C14-03 | High | `chapter_reinforcement-learning/value-iter.md:178` | The text calls value iteration “the only algorithm in the book” with a stated property. | Universal comparisons require an explicit scope. | Name the property and compare only with the algorithms introduced so far, or remove the claim. |
| C14-04 | Medium | `chapter_reinforcement-learning/value-iter.md:189` | “Notice” asks the reader to infer the important distinction after the equation. | Interpret equations explicitly. | Replace with a declarative sentence naming what changes between policy evaluation and value iteration and why the maximization matters. |
| C14-05 | High | `chapter_reinforcement-learning/mdp.md:189` | Partial observability is dispatched in a short paragraph as an approximation issue. | Limitations deserve explanation proportional to their importance. | State that observations need not be Markov, introduce belief state or history as remedies, and explain that treating observations as states changes the problem. |
| C14-06 | Medium | `chapter_reinforcement-learning/mdp.md:79` | “In one line” advertises compactness instead of explaining the Markov property. | Avoid self-conscious narration about brevity. | State the conditional-independence relation, define its variables, and give the interpretation directly. |
| C14-07 | High | `chapter_reinforcement-learning/qlearning.md:191-193` | A dense paragraph combines ties, state-action coverage, learning rates, exploration, and convergence before saying behavior “collapses.” | One paragraph should carry one explanatory job; theorem conditions must be explicit. | Split into update definition, convergence assumptions, and practical exploration. Replace “collapses” with the measured or theoretical failure. |
| C14-08 | Medium | `chapter_reinforcement-learning/qlearning.md:242-261` | Bandit material refers to a “promised” case and a contextual bandit as a “rung” or problem “wearing” different clothes. | Avoid callback metaphors and anthropomorphism. | Define bandit and contextual-bandit assumptions directly and say which transition terms disappear. |
| C14-09 | High | `chapter_reinforcement-learning/qlearning.md:330-340` | Double Q-learning and off-policy limitations receive only a brief bridge. | Explanatory depth should match downstream importance. | Add the maximization-bias calculation, distinguish behavior and target policies, and state coverage plus function-approximation limitations before Chapter 15. |
| C14-10 | High | `chapter_reinforcement-learning/policy-gradient.md:1-120` | The policy-gradient result is developed without an early, compact list of regularity and on-policy assumptions. | Theorems and identities require scope conditions. | State differentiability, support, trajectory-distribution, interchange-of-gradient/expectation, and finite/discounted-return conditions next to the result. |
| C14-11 | High | `chapter_reinforcement-learning/baselines.md:64` | A choice “adds variance for free.” | Avoid economic idioms when a statistical explanation is available. | Say it increases estimator variance without changing the expectation, then show the covariance/variance term responsible. |
| C14-12 | High | `chapter_reinforcement-learning/baselines.md:240` | The leave-one-out baseline argument, estimator definition, dependence condition, and implementation advice occupy one very long paragraph. | Separate formal result, interpretation, and implementation. | Break into the unbiasedness condition, estimator, proof sketch, and batching caveat. Define which samples are conditionally independent. |
| C14-13 | Medium | `chapter_reinforcement-learning/baselines.md:538` | The synthesis opens with “Everything…” | Avoid universal summary openings. | Enumerate the three estimator-control mechanisms and state which affect bias, variance, or both. |
| C14-14 | High | `chapter_reinforcement-learning/imitation.md:127` | One long paragraph introduces actor, critic, occupancy mismatch, collection, and optimization together. | Need-driven exposition and one main claim per paragraph. | Separate information source, supervised objective, distribution-shift failure, and interactive-data remedy. |
| C14-15 | Medium | `chapter_reinforcement-learning/imitation.md:432-461` | Slides use “No Kernel…Just Expert,” “Fit is perfect. Trap,” and “Zero mistakes…” | Slides must use descriptive titles and complete, non-theatrical claims. | Retitle around supervised imitation, covariate shift, and compounding error; show the assumption/result pair. |
| C14-16 | Medium | `chapter_reinforcement-learning/deep-rl.md:123` | “Three lines…Nothing else” frames a minimal implementation. | Avoid claims that erase required context. | State what the three lines implement and separately list omitted replay, target, exploration, normalization, and evaluation machinery. |
| C14-17 | Medium | `chapter_reinforcement-learning/deep-rl.md:314-356` | Two figure captions are long enough to become mini-essays. | Captions should be self-contained but concise. | Keep setup, axes, and conclusion in the caption; move mechanism, caveats, and cross-references into body prose. |
| C14-18 | High | `chapter_reinforcement-learning/deep-rl.md:575` | Slide slogan: “loss value means nothing.” | Avoid categorical slogans; distinguish optimization diagnostics. | Say that losses are not comparable across changing target distributions and name the additional evaluation metrics required. |
| C14-19 | High | `chapter_reinforcement-learning/baselines.md:383` | A control-variate transition relies on “in one line” to carry a nontrivial identity. | Equations need a before/after explanation and assumption check. | Introduce the zero-expectation term, derive the identity, and state when the chosen baseline may depend on state or other samples. |
| C14-20 | High | `chapter_reinforcement-learning/deep-rl.md:468` | “Nothing else” follows a code result and implies sufficiency. | Do not generalize from a compact demonstration. | Label the example as a diagnostic implementation and state which stability and evaluation components a full agent still needs. |

## Mathematics and notation

- Declare whether returns are finite-horizon, discounted continuing, or episodic
  before each theorem; do not rely on context to supply boundary terms.
- Distinguish Bellman expectation identities, Bellman optimality fixed points,
  stochastic-approximation updates, and loss functions typographically and in
  prose.
- Policy-gradient and baseline results need support and dependence conditions
  next to the displayed equations.
- Use separate notation for behavior and target policies throughout the
  off-policy discussion, not only when divergence becomes relevant.

## Figures, captions, and slides

Several figures do exemplary work: they show distribution shift, estimator
variance, or instability with a controlled comparison. Preserve their diagnostic
role. Shorten captions that contain derivations or implementation advice; captions
need setup, axes, and conclusion, while the body carries mechanisms and caveats.
Slides are the least polished layer and require a full independent rewrite of
fragmentary or slogan titles.

## Code and experiment pedagogy

The chapter benefits from small environments where exact values or failure modes
are visible. Each stochastic experiment should report seeds, number of runs,
dispersion, evaluation policy, and horizon. Explicitly separate correctness checks
against analytic/tabular values from empirical performance claims. When a compact
implementation omits standard machinery, list the omission before interpreting
the output.

## Recurring artifacts

- “In one line,” “nothing else,” and “everything” used to advertise compression.
- “Collapse” and “trap” used where a named statistical failure is clearer.
- Long paragraphs combining definition, derivation, caveat, and implementation.
- Slides with clipped imperatives or slogan titles.
- Assumptions stated after rather than alongside the result they qualify.

## Positive patterns to preserve

- The progression from exact dynamic programming to sampled updates is clear.
- Many equations receive operational interpretations in terms of data and
  estimators.
- Controlled examples expose covariate shift, variance, and moving-target
  behavior.
- Summaries usually distinguish value-based, policy-based, and imitation methods.

## Prioritized revision plan

1. Correct universal claims in value iteration and explicitly scope all theorem
   statements.
2. Expand the off-policy/Double-Q bridge before deep function approximation.
3. Split the overloaded baseline and imitation paragraphs into formal result,
   interpretation, and implementation.
4. Standardize stochastic experiment reporting and bound conclusions to the
   tested setup.
5. Shorten essay-like captions and rewrite every slide title as a descriptive
   claim.
6. Finish with a restraint pass for “nothing/everything/collapse/trap.”
