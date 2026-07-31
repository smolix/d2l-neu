# Style Guide for Technical Chapters

This document defines the house style for technical chapters, slides, examples,
captions, and supporting explanations. It is written as an operational guide for
authors and agents. Following it should produce prose that is enjoyable to read,
effective at teaching, and technically precise.

The guide draws on three references:

- David MacKay, [*Information Theory, Inference, and Learning
  Algorithms*](https://www.inference.org.uk/itprnn/book.pdf), especially Chapter 1.
- Bernhard Schölkopf and Alexander Smola, [*Learning with
  Kernels*](https://mcube.lab.nycu.edu.tw/~cfung/docs/books/scholkopf2002learning_with_kernels.pdf),
  especially Chapters 1 and 2.
- Marc Finzi, Shikai Qiu, Yiding Jiang, Pavel Izmailov, J. Zico Kolter, and
  Andrew Gordon Wilson, ["From Entropy to Epiplexity: Rethinking Information
  for Computationally Bounded
  Intelligence"](https://arxiv.org/html/2601.03220v2).

The goal is not to imitate any author's surface mannerisms. It is to combine
their strongest habits:

| Reference | Principal lesson |
|---|---|
| MacKay | Begin with a concrete problem. Make abstractions answer questions the reader already understands. |
| Schölkopf and Smola | Introduce concepts in a controlled sequence. State assumptions and notation precisely, then interpret formal results in ordinary language. |
| Finzi et al. | Organize difficult material around a real conceptual tension. Distinguish definitions, heuristics, rigorous claims, experiments, and limitations. |

The resulting prose should be precise, calm, economical, and intellectually
substantial. It may be lively, but it must not sound theatrical, promotional,
portentous, or mechanically "engaging."

## 1. The three requirements

Every chapter must meet three independent standards.

1. **Writing quality:** the prose is fluent, enjoyable, and recognizably authored.
2. **Explanation quality:** the exposition teaches well and manages the reader's
   cognitive load.
3. **Technical quality:** the content is correct, precise, and ordered by logical
   dependency.

A chapter that succeeds on only two dimensions is not finished. Polished and
correct prose that does not teach well must be rewritten. So must technically
sound instruction that is tedious or awkward to read.

Technical correctness is non-negotiable. Pedagogy determines the order in which
the material is presented. Writing quality determines how naturally the reader
can follow and enjoy that order.

## 2. The target voice

Write as a knowledgeable researcher explaining a subject to an intelligent reader
who is new to this particular material.

The voice should be:

- technically exact;
- direct without being abrupt;
- confident without pretending that every claim is universal;
- explanatory rather than performative;
- interested in the problem rather than impressed with its own prose;
- concise where the argument is simple and patient where it is difficult;
- distinctive enough to sound written, but never conspicuously clever.

The reader should feel guided through a sequence of ideas. The reader should not
feel that the author is manufacturing excitement, delivering slogans, or
commenting on the act of exposition.

A good passage usually performs the following work:

1. It identifies a specific problem.
2. It gives the minimum setup needed to state that problem precisely.
3. It introduces notation because the notation helps solve the problem.
4. It derives or states a result.
5. It explains what the result says.
6. It identifies assumptions, limitations, or the next unresolved question.

The prose should make the logical structure visible without repeatedly announcing
that structure.

## 3. Where enjoyment should come from

Restraint must not produce sterile prose. Remove artificial drama, not personality,
curiosity, wit, or intellectual pleasure.

The references are enjoyable because the reasoning creates momentum. MacKay uses
concrete situations, genuine questions, numerical consequences, and progressively
better solutions. Schölkopf and Smola let a simple construction grow naturally
into a general method. Finzi et al. develop a real conflict between established
theory and observed phenomena, then resolve it through a new distinction.

The pleasure should come from understanding something. It should not depend on the
author repeatedly declaring that an idea is exciting.

### 3.1 Genuine questions

A good question creates an intellectual objective:

> A replay buffer improves sample efficiency, but its trajectories were generated
> by older policies. When can those trajectories still provide a valid update?

Avoid questions whose only function is conversational decoration:

> So what is really happening here?

> Why should we care?

> Is this not surprising?

### 3.2 Concrete situations

Prefer a specific instance that makes the abstraction necessary:

> Suppose a medical classifier is trained at one hospital and deployed at another.
> Even if the labeling rule is unchanged, differences in patient demographics and
> measurement equipment can change the input distribution.

The example gives the formal distinction a reason to exist.

### 3.3 Earned surprise

Non-obvious conclusions are welcome when the derivation earns them:

> The baseline may depend arbitrarily on the state. It does not bias the gradient
> because, conditional on that state, the policy score has expectation zero.

Do not announce surprise before supplying it:

> Remarkably, a powerful and unexpected fact emerges.

### 3.4 Progressive discovery

Let a simple method expose its own limitation. For example:

```text
nearest class mean
-> linear decision boundary
-> feature representation
-> kernel evaluation
-> sparse support-vector solution
```

The reader should see why the next method is needed.

### 3.5 Precise contrasts

A concise contrast can make a difficult distinction memorable:

> Entropy measures the uncertainty that remains after the best available
> prediction. Epiplexity measures the structure absorbed by the predictor itself.

Use contrast only when the distinction is real. Do not reach automatically for
"not merely X, but Y" or similar templates.

### 3.6 Concrete consequences

After a derivation, show what changes in practice:

> The cost is quadratic in sequence length. Doubling the context therefore
> multiplies the attention matrix by four, even before accounting for the wider
> activations stored during training.

Specific consequences give equations weight.

### 3.7 Occasional wit

Brief wit is acceptable when it arises naturally from the subject and does not
carry the explanation. MacKay uses this effectively.

- A light remark should not become an extended metaphor.
- Humor must not obscure a definition.
- The reader should not have to decode a joke before understanding the mathematics.
- Humor must not dismiss a subtlety or a competing method.

Personality may accompany the reasoning, but it may not replace it.

## 4. Explanation must follow necessity

Every concept should enter because the preceding discussion has created a need for
it.

MacKay does not begin error-correcting codes with a taxonomy of coding methods. He
first asks how reliable communication can occur over a noisy channel. The need for
redundancy and decoding follows from that problem.

Schölkopf and Smola do not begin with a catalogue of kernel constructions. They
begin with classification, ask how similarity can be represented, start with dot
products, and then explain why a more general feature representation is needed.

Finzi et al. do not introduce a new information measure in isolation. They first
identify cases in which existing notions conflict with observed practice, then
introduce computational constraints as a way to resolve the conflict.

Apply the same discipline everywhere:

- Do not introduce a definition merely because it belongs to the topic.
- Do not present an equation merely because it is standard.
- Do not add a section merely because other textbooks have one.
- Do not mention an algorithm before the reader knows what problem it solves.
- Do not introduce notation before the underlying objects and distinctions are
  clear.

Ask whether removing a paragraph would leave the reader unable to understand why
the next mathematical object is needed. If so, the paragraph is doing necessary
work. If not, it may be generic preamble or repetition.

## 5. Chapter architecture

A chapter should have a visible intellectual progression.

### 5.1 Opening sequence

Use the following order when applicable:

1. **Problem.** What task or discrepancy motivates the chapter?
2. **Concrete instance.** Give an example, experiment, diagram, or small case.
3. **Required abstraction.** Identify the concept needed to handle the example.
4. **Scope.** State what the chapter will and will not develop.
5. **Roadmap.** Explain the order of topics in terms of intellectual dependency.

The roadmap should explain why the order is useful, not merely list section names.

Weak:

> Section 3 discusses gradients. Section 4 discusses optimization algorithms.
> Section 5 presents experiments.

Better:

> We first derive the gradient of the objective. This exposes the quantities an
> optimizer must estimate. We then compare algorithms according to how they
> approximate those quantities, before testing the resulting trade-offs
> experimentally.

### 5.2 Suitable openings

A chapter opening usually takes one of three forms.

#### A concrete task

> A classifier is given labeled examples and must assign a label to a new input.
> The central difficulty is that the new input was not part of the training set.

#### A conflict between a useful idea and an observed difficulty

> Reusing old experience makes reinforcement learning more data-efficient. It also
> changes the distribution from which the update was derived. The methods in this
> section control the resulting bias or variance.

#### A discrepancy between theory and practice

> A deterministic transformation cannot increase Shannon information.
> Nevertheless, computation can expose structure that a bounded learner could not
> previously use. Resolving this discrepancy requires the observer's computational
> limits to enter the definition.

Use the third form only when there is a genuine discrepancy. Do not manufacture a
"paradox" to make ordinary material sound important.

### 5.3 Unsuitable openings

Avoid openings that:

- announce that the subject is fundamental or fascinating without explaining why;
- begin with broad claims about intelligence, science, or human history;
- tell the reader that everything changes after one observation;
- personify the chapter, equation, algorithm, or field;
- call attention to the elegance of the exposition;
- use a metaphor in place of a technical problem;
- present a roadmap before establishing why the chapter exists.

Rejected:

> Everything distinctive about the subject is a consequence of that sentence.

Preferred:

> This feedback creates three difficulties: actions determine which data are
> collected, errors affect later states, and policy updates change the distribution
> used for subsequent learning.

The revision replaces a grand assertion with specific, verifiable consequences.

### 5.4 Dependency order

Prefer this progression:

```text
problem
-> simple case
-> formal representation
-> derivation or algorithm
-> interpretation
-> limitation
-> more general method
```

Present material in the order needed to understand it, not in the order in which it
was discovered or implemented. Prefer conceptual dependency to chronology.

### 5.5 Scope and prerequisites

State prerequisites when they matter:

> This section assumes familiarity with conditional probability but not with
> measure theory.

> We use finite state and action spaces until Section 6, where function
> approximation replaces the tables.

This lets readers allocate attention correctly and prevents later caveats from
looking like omissions.

### 5.6 Summaries

A chapter summary should reconstruct the argument, not repeat the table of contents.
It should answer:

- What problem was solved?
- Which mathematical objects were introduced?
- Which relationship among them matters?
- Under which assumptions does the conclusion hold?
- What remains unresolved?

Avoid bullet lists consisting only of terms introduced in the chapter.

## 6. Section architecture

Each section should answer one main question. The question may appear in the title,
but it need not. The prose should nevertheless make it recoverable.

Examples include:

- How does a feature map induce a similarity measure?
- Why does the margin determine the scale of a linear classifier?
- How can a policy gradient be estimated from sampled trajectories?
- Which part of a coding length represents learned structure?
- Why does off-policy reuse require a correction?

### 6.1 Opening paragraph

The first paragraph should connect the section to the preceding argument. It should
normally state the unresolved issue and identify the approach taken here.

> Value iteration assumes that the transition kernel is known. In most
> applications, the agent observes transitions but not their probabilities.
> Q-learning replaces each exact Bellman backup with a stochastic backup computed
> from one observed transition.

This transition carries technical content. It does not merely announce that the
discussion is moving on.

### 6.2 Internal progression

Within a section, use small inferential steps:

1. Define the setting.
2. State the objective or question.
3. Work through the simplest case.
4. Introduce the general expression.
5. Interpret its terms.
6. Examine a consequence or boundary case.
7. State the practical implications.
8. Identify the next limitation.

Do not alternate unpredictably among intuition, implementation, historical remarks,
and formal results.

### 6.3 Closing paragraph

End a section when its question has been answered. A useful closing paragraph may:

- summarize the result in one or two sentences;
- identify the assumption that the next section removes;
- compare the method with the preceding one;
- state what an experiment established.

Avoid an artificial cliffhanger:

> We are now ready to uncover the surprising method that changes this picture
> entirely.

Prefer a technical dependency:

> The estimator is unbiased, but its variance grows with trajectory length. The
> next section introduces baselines that reduce this variance without changing its
> expectation.

## 7. Paragraph construction

A paragraph should perform one coherent piece of reasoning. Common functions include:

- posing a problem;
- defining an object;
- deriving a result;
- interpreting an equation;
- explaining an example;
- comparing two methods;
- qualifying a claim;
- connecting two sections.

Do not combine several functions merely to avoid a paragraph break.

### 7.1 Default paragraph pattern

A reliable explanatory paragraph has this shape:

1. Claim or purpose.
2. Reason or mechanism.
3. Evidence, equation, or example.
4. Consequence.

For example:

> A baseline may depend on the state but not on the sampled action. Conditional on
> the state, the policy score has expectation zero. Subtracting the baseline
> therefore adds a term whose expectation vanishes. The estimator remains unbiased,
> although its variance can change.

Each sentence advances the argument.

### 7.2 Paragraph length

Most prose paragraphs should contain roughly three to seven sentences. This is a
tendency, not a mechanical requirement.

Avoid repeated one-sentence paragraphs. They create artificial drama and fragment
the reasoning. Use a one-sentence paragraph only when the sentence genuinely changes
the mode of discussion, such as immediately before a formal theorem or a new
experimental question.

### 7.3 Topic sentences

The first sentence should identify the paragraph's subject, not advertise its
importance.

Weak:

> There is a crucial and often overlooked point here.

Better:

> The correction applies to the sampling distribution, not to the reward function.

### 7.4 Demonstrative references

Do not use bare "this," "that," or "it" when the referent might be ambiguous.

Weak:

> This causes instability.

Better:

> Reusing a stale target network causes the bootstrap target to drift during
> optimization.

When possible, follow "this" with a noun: this approximation, dependence,
estimator, discrepancy, or bound.

## 8. Sentence-level style

### 8.1 Use concrete grammatical subjects

Prefer subjects that name the mathematical or computational object performing the
action:

- The policy assigns a probability to each action.
- The Bellman operator maps one value function to another.
- The second term penalizes large weights.
- The experiment measures validation loss after each data intervention.

Avoid empty framing subjects:

- It is important to note that ...
- There are several things worth mentioning ...
- What is happening here is that ...
- The key idea is that ...

Usually the content can become the sentence directly.

### 8.2 One principal logical move per sentence

A sentence may contain qualifications, but it should not perform an entire
derivation.

Overloaded:

> Because the samples come from the behavior policy, which may differ from the
> current policy, and because the ratio can become large in regions where the
> behavior probability is small, importance sampling is unbiased but often
> unstable, which motivates clipping even though clipping introduces bias.

Improved:

> The samples come from the behavior policy rather than the current policy.
> Importance weighting corrects this mismatch and gives an unbiased estimator. The
> ratio can become large when the behavior policy assigns little probability to an
> action, however, so the estimator may have high variance. Clipping controls the
> ratio at the cost of introducing bias.

### 8.3 Sentence length and rhythm

Use varied but controlled sentence lengths.

- Short sentences are useful for definitions, conclusions, and genuine contrasts.
- Medium sentences should carry most of the exposition.
- Long sentences are appropriate when several conditions belong to one claim, but
  their grammatical structure must remain visible.

Do not use short sentences for theatrical emphasis:

> This changes everything.

> The consequence is profound.

> And that is the problem.

Replace them with the actual consequence.

### 8.4 Prefer ordinary verbs

Prefer verbs such as gives, requires, implies, measures, estimates, bounds,
increases, decreases, and depends on.

Avoid ornamental verbs unless they are literal: unlocks, unveils, reshapes,
whispers, hides, smuggles, collapses, or breathes life into. An equation does not
reveal its secret. It implies a relationship.

### 8.5 Active and passive voice

Use active voice when the actor matters:

> The optimizer updates the parameters after each minibatch.

Use passive voice when the process or result matters more:

> The parameters are initialized independently from a zero-mean Gaussian
> distribution.

Do not force active voice at the expense of natural technical prose.

### 8.6 First-person plural

Use "we" for an operation that the author and reader are genuinely performing
together:

- We now substitute the definition of \(V^\pi\).
- We first consider the finite case.
- We compare the two estimators at equal sample size.

Do not use "we" for unsupported opinion:

- We can clearly see ...
- We all know ...
- We should be amazed that ...

Do not repeat "we now" at every transition.

### 8.7 Second person

Reserve "you" for exercises, instructions, or direct descriptions of an interface.
In conceptual exposition, prefer a reader-independent statement.

Weak:

> If you increase the batch size, you will see the variance decrease.

Better:

> Increasing the batch size reduces the estimator's variance.

### 8.8 Rhetorical questions

MacKay uses questions effectively because they pose genuine technical problems and
are answered by the ensuing analysis. Use a question only if:

- the reader can understand it immediately;
- it defines the next unit of reasoning;
- the answer is not already obvious;
- the text answers it promptly.

Do not use several rhetorical questions in succession or use a question as a
substitute for a transition.

### 8.9 Punctuation

Use punctuation according to grammatical function.

- A colon should introduce an explanation, list, equation, or consequence that
  completes the preceding clause.
- An em dash should mark a genuine interruption or compressed qualification. It
  should be rare.
- Parentheses should contain secondary material, not essential reasoning.
- Semicolons should connect closely related independent clauses. They should not
  hold together an overloaded sentence.

Avoid prose built from repeated colons and em dashes. That construction often
produces breathless, self-conscious writing.

## 9. Mathematical exposition

Mathematics should enter the prose as part of the argument, not as a separate
symbolic layer.

### 9.1 The equation protocol

For every important equation, provide the following in a natural order:

1. **Purpose:** what question the equation answers.
2. **Objects:** what the variables represent.
3. **Assumptions:** what conditions are in force.
4. **Equation:** the formal statement.
5. **Reading:** what its main terms mean.
6. **Consequence:** what follows from it.
7. **Boundary or limitation:** when the conclusion changes or fails.

Not every equation requires seven sentences, but every major equation should be
understandable along these dimensions.

### 9.2 Introduce objects before symbols

Weak:

> Let \(J(\theta)=\mathbb{E}_{\tau\sim p_\theta}[R(\tau)]\).

Better:

> Let \(p_\theta(\tau)\) denote the distribution over trajectories induced by
> policy \(\pi_\theta\), and let \(R(\tau)\) be the trajectory return. The objective
> is the expected return
> \[
> J(\theta)=\mathbb{E}_{\tau\sim p_\theta}[R(\tau)].
> \]

The notation now names concepts already established in prose.

### 9.3 Equations must be grammatical

An equation should complete or support a sentence:

> Averaging over the sampled action gives
> \[
> \mathbb{E}_{a\sim\pi_\theta(\cdot\mid s)}
> [\nabla_\theta\log\pi_\theta(a\mid s)]=0.
> \]
> A state-dependent baseline therefore does not change the expected gradient.

Avoid placing an equation between unrelated fragments.

### 9.4 Interpret after deriving

Never assume that a derivation is self-explanatory. After a result, explain:

- which quantity is being optimized;
- which term controls fit;
- which term controls complexity or variance;
- which variables are held fixed;
- what happens in a simple limiting case;
- why the result matters for the algorithm.

Schölkopf and Smola repeatedly pause after a bound or constrained problem to explain
why the terms are needed. Follow that practice.

### 9.5 Explain constraints

For each constraint, say what it prevents or enforces.

Instead of writing only

> subject to \(y_i(w^\top x_i+b)\geq 1\),

add:

> The constraint requires every training example to lie on the correct side of the
> separating hyperplane. The constant \(1\) also fixes the otherwise arbitrary scale
> of \(w\) and \(b\).

### 9.6 Separate exact statements from intuition

Use explicit markers where needed:

- Formally, ...
- Equivalently, ...
- A useful interpretation is ...
- This approximation is accurate when ...
- The theorem guarantees ...
- The experiment suggests ...
- The heuristic does not establish ...

Finzi et al. provide a useful model: practical estimators are presented as useful,
and their failures to constitute rigorous bounds are stated directly. Never allow
an intuitive analogy to masquerade as a theorem.

### 9.7 Definitions

A definition should contain:

1. the object being defined;
2. its domain and codomain where relevant;
3. all assumptions;
4. the formal expression;
5. one sentence of interpretation;
6. an example or non-example when confusion is likely.

Do not introduce several definitions in a row without showing their relationship.

### 9.8 Theorems

Before a theorem, tell the reader why it is needed. After it, explain:

- what is new;
- which assumption carries the burden;
- how the result applies to the running problem;
- whether the bound is qualitative or numerically useful.

Do not call a theorem powerful, remarkable, or deep without identifying the
specific consequence that earns the description.

### 9.9 Proofs

A substantial proof should expose its plan:

1. State the proof strategy.
2. Divide the argument into meaningful lemmas or cases.
3. Explain why each step advances the strategy.
4. Identify where each assumption is used.
5. Conclude by matching the established statements to the theorem.

Use descriptive proof subheadings such as "Bound the unconditional terms,"
"Construct the forward sampler," "Control the remaining probability," and
"Combine the inequalities." Avoid headings such as "The key step" or "The
trick."

### 9.10 Notation discipline

- Reuse standard notation when it does not conflict with the chapter.
- Do not rename an object without need.
- Do not use two symbols for the same concept.
- Do not use the same symbol for unrelated concepts in nearby sections.
- State vector, matrix, and tensor shapes when they matter.
- Distinguish random variables from realizations consistently.
- Distinguish a model, its parameters, its output distribution, and a sample.
- Avoid defining symbols used only once.
- Keep subscripts semantically meaningful.
- Use notation tables only when the notation exceeds what prose can manage.

## 10. The four levels of explanation

Teaching well requires more than stating correct facts. For every important
concept, connect four levels:

1. **Concrete:** a specific problem or example.
2. **Conceptual:** the distinction or mechanism.
3. **Formal:** notation, derivation, or theorem.
4. **Operational:** what the result changes in calculation or implementation.

For example:

> Consider a policy that chooses between two actions. Increasing the probability
> of the better action raises the expected reward, but the reward itself is not
> differentiable with respect to the policy parameters. The score-function identity
> moves the derivative onto the log probability:
> \[
> \nabla_\theta \mathbb{E}_{a\sim\pi_\theta}[r(a)]
> =
> \mathbb{E}_{a\sim\pi_\theta}
> [r(a)\nabla_\theta\log\pi_\theta(a)].
> \]
> We can therefore estimate the gradient from sampled actions and their rewards,
> without differentiating through the reward function.

This passage motivates the identity, states it, and explains what it permits.

For each concept, an author or agent should be able to answer:

- What problem requires it?
- What simpler idea does it extend?
- What does the notation represent?
- What does the equation say in ordinary language?
- What is the smallest useful example?
- What assumption makes the result possible?
- What fails when that assumption is removed?
- How does the concept connect to the next method?

## 11. Examples and counterexamples

Examples are part of the reasoning, not decoration.

### 11.1 Start with the smallest informative case

MacKay's progression from a noisy bit channel to repetition codes and then to
Hamming codes works because each example exposes a limitation that motivates the
next construction.

Follow the same pattern:

- begin with a two-state MDP before a continuous control problem;
- begin with a scalar derivative before a Jacobian;
- begin with a two-dimensional feature map before a general kernel;
- begin with one bootstrap target before a full actor--critic algorithm.

### 11.2 Preserve the connection to the general case

After the example, say which features generalize:

> The two-state calculation is special only in that the Bellman equations can be
> solved by hand. The contraction argument used below does not depend on the number
> of states.

### 11.3 Use counterexamples to delimit claims

If a plausible statement is false, give the smallest counterexample that exposes
the failure. Do not merely warn that the intuition can break. Show how.

### 11.4 Numerical examples

A numerical example should answer a quantitative question. Include:

- parameter values;
- units where applicable;
- the computation;
- the conclusion;
- whether the values are representative or illustrative.

Avoid arbitrary numbers that add arithmetic without insight.

## 12. Figures, tables, and captions

### 12.1 Introduce the purpose before the figure

Weak:

> Figure 4.2 shows the process.

Better:

> The importance ratio corrects the action probability at each sampled state.
> Figure 4.2 separates this local correction from the trajectory-level product that
> causes the variance problem.

### 12.2 Write self-contained captions

A caption should identify:

- what is shown;
- what varies;
- what is held fixed;
- which comparison matters;
- the main conclusion, if it is visible in the figure.

Do not make the caption a second essay. Details that require argument belong in the
main text.

### 12.3 Interpret the figure in the main prose

Do not rely on the reader to infer the intended conclusion. State the relevant
curve or region, the observed trend, whether the result supports a prediction, and
any uncertainty or exception.

### 12.4 Avoid theatrical figure language

Rejected:

> The loop itself needs only one picture, Figure 14.1, and one identification worth
> fixing early: unroll the loop in time and its emissions are the data.

Preferred:

> Figure 14.1 shows the agent--environment interaction. Unrolling this interaction
> over time yields the trajectory
> \[
> \tau=(s_0,a_0,r_0,s_1,\ldots),
> \]
> which provides the data used by reinforcement-learning algorithms.

The preferred version states what the figure contains and defines the resulting
object. It does not comment on how many pictures the explanation needs.

### 12.5 Tables

Use a table when the reader must compare the same fields across several methods.
Useful fields include:

- learned quantity;
- required data;
- bias;
- variance;
- computational cost;
- assumptions;
- section introduced.

Do not use a table as a dumping ground for isolated facts.

## 13. Code and experiments

### 13.1 Explain the computational purpose first

Before a code block, state:

- what is being computed;
- what inputs are assumed;
- what output should result;
- which implementation detail is conceptually important.

Do not narrate obvious syntax line by line.

### 13.2 Keep implementation and concept aligned

The notation in the prose, equations, code, and figures should refer to the same
objects wherever possible. If code uses a different convention for practical
reasons, explain the mapping once.

### 13.3 Interpret output

After a code block or experiment:

- state the observed result;
- compare it with the theoretical prediction;
- discuss uncertainty or variability;
- identify failures or deviations;
- avoid declaring success merely because the code ran.

### 13.4 Match claims to evidence

Use wording proportional to the evidence.

- A controlled identity check confirms the implementation.
- A finite experiment is consistent with an asymptotic claim.
- A benchmark on several datasets suggests broader behavior.
- An experiment does not prove a general empirical claim.

State seeds, error bars, sample counts, or confidence intervals when variability
affects the conclusion.

## 14. Titles and headings

Titles should tell the reader what intellectual work occurs in the section.

### 14.1 Preferred title forms

- Data Representation and Similarity
- The Role of the Margin
- Estimating the Policy Gradient
- Correcting Distribution Mismatch
- Approximating Model Description Length
- When the Bootstrap Target Becomes Unstable

### 14.2 Questions as titles

A question is appropriate when it defines a genuine problem:

> How Much Redundancy Is Required?

Do not use vague questions:

- What Is Really Happening?
- Why Does This Matter?
- Where Do We Go from Here?

### 14.3 Avoid literary or promotional titles

Avoid titles such as:

- One Picture
- The Hidden Engine
- The Great Swap
- Where the Magic Happens
- The Waterfall
- A Beautiful Surprise
- Everything Changes
- The Secret Life of Gradients

A standard technical term that happens to be metaphorical, such as "deadly triad,"
may be used if it is established terminology and is defined.

### 14.4 Keep the hierarchy parallel

Sibling headings should be parallel in granularity and grammatical form.

Poor:

- Policy Evaluation
- Why Control Is Hard
- A Small Example That Changes Everything
- Implementation

Better:

- Policy Evaluation
- Policy Improvement
- Off-Policy Evaluation
- Implementation and Diagnostics

## 15. Transitions and conversational flow

The conversation should advance through technical dependencies.

### 15.1 Useful transitions

**Assumption removal**

> The preceding derivation assumes access to the transition kernel. We now replace
> exact expectations with sampled transitions.

**Limitation**

> The estimator is unbiased, but its variance increases with the horizon.

**Generalization**

> The scalar result extends to vectors by replacing the derivative with the
> gradient.

**Comparison**

> Both methods estimate the same fixed point. They differ in whether the
> expectation is evaluated exactly or from samples.

**Consequence**

> Because the contraction factor is \(\gamma\), repeated application converges
> geometrically when \(\gamma<1\).

### 15.2 Empty transitions

Remove or rewrite:

- With this foundation in place ...
- Armed with this insight ...
- Having established the above ...
- We now turn our attention to ...
- The stage is now set for ...
- This brings us naturally to ...
- Let us embark on ...

These phrases often conceal the connection. State that connection instead.

### 15.3 Do not over-signpost

A chapter needs a roadmap. Every paragraph does not. Repeated uses of "first,"
"next," "now," "finally," and "in this section" make prose sound generated. Use
them only when the sequence itself matters.

## 16. Claims, qualifications, and intellectual honesty

### 16.1 State the strength of the claim

Distinguish among:

- identity;
- theorem;
- bound;
- approximation;
- heuristic;
- empirical regularity;
- implementation convention;
- analogy;
- conjecture.

Never blur these categories for smoother prose.

### 16.2 Put qualifications near the claim

Weak:

> PPO provides stable policy updates.

with a caveat much later that this is not guaranteed.

Better:

> PPO's clipped objective often makes policy updates less sensitive to a small
> number of extreme importance ratios, although clipping does not guarantee
> monotonic improvement.

### 16.3 Avoid universal claims from local mechanisms

Rejected:

> Everything distinctive about reinforcement learning follows from feedback.

Preferred:

> Feedback distinguishes reinforcement learning from supervised prediction in
> several ways: the policy affects the data distribution, present actions change
> future observations, and rewards may be delayed.

### 16.4 Acknowledge unresolved points directly

Good technical prose can say:

- This approximation is convenient but not a bound.
- The analysis does not cover function approximation.
- The experiment cannot distinguish these two explanations.
- The constant is too large to make the bound numerically informative.
- The method requires samples from the current policy.
- Whether the effect persists at larger scale remains unclear.

Do not hide limitations in footnotes merely to preserve narrative momentum.

## 17. Generated-writing artifacts to remove

The following patterns require revision.

### 17.1 Grand summary slogans

Avoid:

- Everything follows from this.
- This changes the entire picture.
- This is the heart of the matter.
- The whole subject lives inside this equation.
- This single observation explains everything.

Replace the slogan with the actual consequences.

### 17.2 Self-conscious narration

Avoid:

- The loop itself needs only one picture.
- One identification is worth fixing early.
- The story begins with ...
- The chapter now earns its notation.
- We have arrived at the key insight.
- The proof has one job.
- The equation is doing more work than it appears.

Discuss the subject, not the author's staging of the subject.

### 17.3 Manufactured contrasts

Avoid habitual constructions such as:

- not merely X, but Y;
- not a defect, but a feature;
- not just an algorithm; it is a way of thinking;
- less about X and more about Y.

Use contrast only when two interpretations genuinely need to be distinguished.

### 17.4 Promotional adjectives

Treat words such as powerful, elegant, profound, remarkable, revolutionary,
dramatic, striking, fascinating, surprising, crucial, and essential as claims that
require evidence. Often the sentence becomes stronger when the adjective is
replaced by the concrete reason.

Weak:

> This elegant identity is crucial.

Better:

> This identity removes the unknown normalization constant from the gradient.

### 17.5 False intimacy

Avoid habitual uses of:

- Notice how ...
- You might be wondering ...
- At first glance ...
- It is tempting to think ...
- The reader may be surprised ...
- Let that sink in.

"Notice" is acceptable when directing attention to a specific mathematical
property, but it should not become a default sentence opener.

### 17.6 Vague importance markers

Delete or rewrite words such as importantly, interestingly, notably,
fundamentally, critically, in essence, at its core, and in a deep sense. The clause
that follows should communicate the importance by itself.

### 17.7 Anthropomorphism

Avoid saying that:

- an equation wants something;
- a model believes something unless belief is formally defined;
- a gradient searches;
- a loss punishes, except as occasional shorthand;
- a theorem tells a story;
- data refuse to cooperate;
- an algorithm knows a fact unavailable in its inputs.

Prefer the literal mechanism.

### 17.8 Unnecessary drama

Avoid:

- sentence fragments used as punchlines;
- repeated one-line paragraphs;
- ominous warnings;
- metaphors of battle, collapse, betrayal, or rescue;
- phrases such as "the catch," "the twist," "the punchline," and "the culprit."

State the limitation or causal mechanism directly.

### 17.9 Generic filler

Delete sentences that could appear unchanged in almost any chapter:

- This topic has attracted considerable attention.
- There are many ways to approach this problem.
- The method has a wide range of applications.
- Understanding this concept is important.
- This section provides an overview.
- Modern machine learning has seen rapid progress.

If context is needed, make it specific.

### 17.10 Repetition disguised as emphasis

Do not state a claim, paraphrase it, and summarize it again without adding
information. Each restatement must serve a purpose: formalization, intuition,
example, limitation, or comparison.

## 18. Preferred vocabulary and phrasing

Prefer language that expresses logical relationships precisely.

### For derivations

- follows from;
- substituting;
- rearranging;
- conditioning on;
- taking expectations;
- applying the definition;
- using the independence assumption;
- combining the two bounds.

### For interpretation

- measures;
- represents;
- depends on;
- is invariant to;
- increases with;
- vanishes when;
- is bounded by;
- differs only in;
- can be viewed as.

### For evidence

- demonstrates in this setting;
- is consistent with;
- supports the hypothesis that;
- suggests;
- does not distinguish;
- remains valid under;
- fails when.

### For transitions

- this assumption excludes;
- the remaining difficulty is;
- the preceding argument requires;
- the same reasoning extends to;
- the two methods differ in;
- removing this assumption gives.

Do not use a thesaurus to vary technical terms. Consistency is more important than
lexical variety.

## 19. Slides

Slides should follow the same intellectual order as the chapter but use less text.

### 19.1 One slide, one claim or operation

A slide should normally contain:

- a descriptive title;
- one equation, figure, comparison, or short argument;
- enough text to interpret it;
- no competing secondary narrative.

### 19.2 Slide titles

The title should identify the content:

- Importance Weighting Corrects the Sampling Distribution
- Clipping Trades Bias for Variance Control
- The Bellman Operator Is a Contraction

Do not use teaser titles such as "The Catch," "A Surprise," "The One Equation,"
"Why Everything Breaks," or "The Fix."

### 19.3 Bullets

Bullets should be parallel and compact. They should not be fragments of a paragraph
split artificially. Use staged reveals only when each stage depends on the previous
one. Do not hide bullets merely to create motion.

### 19.4 Equations on slides

A slide equation needs:

- definitions of nonstandard symbols;
- a visual indication of the relevant term;
- one plain-language consequence.

Do not place several unconnected equations on one slide.

### 19.5 Figures on slides

State what the audience should compare. A figure without an interpretive sentence
is not self-explanatory.

## 20. Revision procedure for an author or agent

Do not produce final prose in one pass.

### Pass 1: Determine the intellectual structure

Write down:

- the chapter's main question;
- the reader's prerequisites;
- the sequence of dependencies;
- the principal claims;
- the examples needed;
- the assumptions and limitations.

If the main question cannot be stated in one or two precise sentences, the chapter
structure is not ready.

### Pass 2: Build the section outline

For each section, record:

- the question answered;
- the input from the previous section;
- the new concept introduced;
- the mathematical result;
- its interpretation;
- the limitation motivating the next section.

Remove sections whose purpose is merely "background" unless that background is
used.

### Pass 3: Draft around examples and derivations

Draft the concrete problem before the general formalism. For each equation, follow:

```text
purpose -> objects -> assumptions -> equation -> interpretation -> consequence
```

### Pass 4: Check conversational flow

For every paragraph, ask:

- Why does this paragraph appear here?
- What does it add?
- What question does it answer?
- Does its final sentence create a real connection to the next paragraph?

Replace empty transitions with technical dependencies.

### Pass 5: Remove generated-writing artifacts

Search manually for:

- key insight;
- heart of;
- at its core;
- fundamentally;
- not merely;
- not just;
- the story;
- the punchline;
- the catch;
- this changes;
- everything;
- powerful;
- elegant;
- remarkable;
- surprisingly;
- we now turn;
- with this foundation;
- armed with;
- it is important to note;
- notice that.

Do not delete these mechanically. Inspect each occurrence and retain it only if it
performs necessary and specific work.

### Pass 6: Audit every claim

Classify each important claim as exact, proved, cited, approximate, heuristic,
empirical, or conjectural. Revise wording that overstates its category.

### Pass 7: Audit notation

Check that:

- every symbol is defined before use;
- symbols retain one meaning;
- dimensions are clear;
- assumptions are local;
- equations are referenced only when necessary;
- the prose explains major equations.

### Pass 8: Inspect paragraph boundaries

Read only the first and last sentence of each paragraph. They should expose the
argument's progression. If they read like generic signposts, revise them.

### Pass 9: Read the prose aloud

Revise:

- overloaded sentences;
- repeated sentence openings;
- clusters of colons or em dashes;
- artificial punchlines;
- abrupt changes in level of abstraction;
- paragraphs whose rhythm sounds promotional.

### Pass 10: Verify consistency across formats

Ensure that the chapter, slides, captions, code comments, and summary use the same:

- terminology;
- notation;
- claims;
- assumptions;
- conclusions.

## 21. Quality rubric

Score each dimension independently out of ten. A publishable chapter should score
at least eight in every dimension, not merely achieve a high average.

### 21.1 Writing quality: 10 points

- **2:** Sentences are fluent and varied.
- **2:** The prose has a recognizable but restrained voice.
- **2:** Examples and contrasts sustain interest.
- **2:** Transitions create genuine momentum.
- **2:** The chapter contains no formulaic, theatrical, or promotional artifacts.

### 21.2 Explanation quality: 10 points

- **2:** The reader understands why each concept is introduced.
- **2:** Concrete, conceptual, formal, and operational explanations are connected.
- **2:** Major equations receive interpretation.
- **2:** Examples expose both the mechanism and its limitations.
- **2:** Summaries reconstruct the argument and consolidate understanding.

### 21.3 Technical quality: 10 points

- **2:** Claims and derivations are correct.
- **2:** Assumptions and qualifications are explicit.
- **2:** Notation is defined, consistent, and economical.
- **2:** Topics follow their logical dependencies.
- **2:** Theory, approximation, empirical evidence, and conjecture are clearly
  distinguished.

A score such as `10/6/10` is a failure. So is `5/10/10`.

For a finer review, score each of the following from zero to two:

| Category | 0 | 1 | 2 |
|---|---|---|---|
| Motivation | Generic or absent | Problem stated | Concrete problem creates a need for the material |
| Structure | Topics merely listed | Mostly logical | Each section follows from a specific unresolved issue |
| Precision | Vague or overstated | Mostly accurate | Claims are explicit, scoped, and qualified |
| Mathematical exposition | Equations dropped into text | Some explanation | Purpose, assumptions, derivation, and interpretation are clear |
| Examples | Decorative or absent | Relevant | Examples drive abstraction and expose limitations |
| Paragraph flow | Fragmented | Understandable | Every paragraph advances one coherent argument |
| Sentence quality | Formulaic or theatrical | Mostly clean | Direct, varied, and technically specific |
| Titles | Vague or promotional | Descriptive | Descriptive and aligned with the conceptual hierarchy |
| Evidence | Claims exceed support | Occasional overreach | Theory, heuristic, and empirical evidence are separated |
| Restraint | Frequent slogans and hype | A few artifacts | Calm prose with no manufactured drama |

The finer rubric requires at least 18 out of 20. Precision, mathematical
exposition, and restraint must each receive a two.

## 22. Definition of done

A chapter is ready only if all of the following are true:

- The opening identifies a concrete problem or genuine conceptual tension.
- Each section answers a recoverable question.
- Concepts appear because the preceding discussion requires them.
- Notation is introduced after the underlying objects.
- Every major equation is interpreted.
- Formal results, heuristics, and experiments are distinguished.
- Assumptions and limitations appear near the claims they qualify.
- Examples lead to general conclusions rather than merely illustrating terms.
- Figures and tables have explicit analytical purposes.
- Titles describe content rather than advertise drama.
- Transitions state logical dependencies.
- The prose contains genuine intellectual tension, concrete examples, and earned
  conclusions without manufacturing excitement.
- No paragraph exists solely to create excitement.
- No sentence makes a vague claim about the whole subject.
- No metaphor carries an essential technical inference.
- Repeated one-line paragraphs do not manufacture emphasis.
- Summaries reconstruct the argument rather than repeat headings.
- Slides preserve the same terminology, notation, and evidential strength.
- A reader can explain what each method does, why it was introduced, and which
  assumptions it requires.

The final objective is prose in which the reader experiences the argument as a
sequence of worthwhile discoveries. It should be pleasant because it is concrete,
well paced, and intelligently phrased; instructive because each abstraction answers
a visible need; and trustworthy because every conclusion follows from clearly
stated premises. The writing should make difficult reasoning easier to follow
without making the author's performance visible.
