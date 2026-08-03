# Language Guide for Technical Chapters

This guide is a companion to [the structural style guide](style-guide.md).
The structural guide explains how to order a chapter, introduce a concept, build
an example, and connect theory with code. This guide addresses a narrower
question: **how should the prose itself be written?** It governs diction,
sentence construction, cohesion, cadence, qualification, and the relation
between words and mathematics.

The intended result is prose that is enjoyable because it is intelligent and
clear, not because it continually tries to sound lively. The reader should feel
guided by an author who knows where the argument is going, understands where it
is difficult, and respects both the subject and the reader's attention.

The principal language models are David MacKay's *Information Theory,
Inference, and Learning Algorithms* and Thomas Cover and Joy Thomas's *Elements
of Information Theory*. MacKay is especially useful for problem-driven,
conversational explanation. Cover and Thomas are especially useful for concise
definitions, logical progression, and economical mathematical prose. Neither
book should be imitated mechanically. Their occasional dated conventions,
extended asides, and declarations that a step is simple or obvious are not part
of the target style. We want their clarity, directness, and intellectual energy.

## 1. The language contract

Every passage should satisfy five conditions.

1. **The grammatical structure exposes the logical structure.** The main clause
   states the main claim. Conditions, causes, contrasts, and qualifications are
   attached where their roles are apparent.
2. **Concrete mathematical or computational objects do the work.** Policies
   assign probabilities, estimators have variance, buffers store transitions,
   and experiments compare methods. Abstract stage directions do not act in
   their place.
3. **Each sentence advances the argument by a recognizable step.** It defines,
   derives, explains, contrasts, qualifies, exemplifies, or concludes. It does
   not merely announce that something is interesting.
4. **The prose maintains continuity.** A reader can identify what each pronoun,
   contrast, and connective refers to, and can see why the next sentence follows
   from the previous one.
5. **The tone is confident without being promotional.** Importance is conveyed
   by consequences. Difficulty is conveyed by the reasoning required. Neither
   needs a theatrical adjective.

A useful summary is:

> Put the object before the commentary, the claim before its ornament, the
> reason beside the claim, and the qualification beside the reason.

## 2. What the reference books do well

### 2.1 MacKay: inquiry followed by resolution

MacKay often begins with a problem that can be stated in ordinary language. He
then makes the rules of the problem explicit, considers the most natural first
attempt, and examines what that attempt gains and loses. The conversation feels
active because the question is real and the answer changes the reader's model of
the problem.

Several linguistic practices make this work:

- Questions specify the next piece of reasoning rather than supplying generic
  excitement.
- Examples use physical or computational nouns: bits, disks, channels, noise,
  encoders, and decoders.
- Assumptions are stated before they are used.
- A derivation is followed by a plain-language conclusion that names the
  assumptions under which it holds.
- Technical terms remain stable. A channel does not become a conduit merely to
  avoid repetition.
- Informality appears locally, often in an example or exercise, while the
  mathematical claim remains exact.
- A tradeoff is allowed to create interest. The prose does not announce a twist;
  it shows what was gained, then quantifies what was lost.

The transferable lesson is not to write more rhetorical questions or jokes. It
is to let an honest problem organize the language.

### 2.2 Cover and Thomas: propositions in a visible chain

Cover and Thomas tend to state the scope of a discussion, introduce the needed
object, give its definition, derive immediate consequences, and then test the
definition on a small example. Their paragraphs often repeat the principal noun
instead of replacing it with a vague pronoun. This repetition makes the logical
thread unusually easy to follow.

Their most useful practices are:

- Definitions begin with the object being defined.
- The sentence before an equation tells the reader what the equation will do.
- The sentence after an equation states its meaning or consequence.
- Logical connectives have precise roles: *thus* marks a consequence, *however*
  a contrast, *conversely* a reversal, and *for example* an instance.
- Proofs are economical because each line has one purpose.
- Remarks isolate a convention, interpretation, or boundary case without
  interrupting the main statement.
- Examples first instantiate the definition and then explain what the numerical
  result means.
- Qualifications are local. A claim is not left broad for several paragraphs
  and weakened only later.

The transferable lesson is that formal prose need not be stiff. A sequence of
well-chosen propositions has its own pace and elegance.

### 2.3 The combined target

Use MacKay's problem sense and Cover--Thomas's propositional discipline.

- Begin from a question, conflict, or task when one genuinely organizes the
  material.
- State definitions and results with the economy of a reference work.
- Explain the reason for each mathematical move in ordinary language.
- Keep the reader oriented through repeated concrete nouns and exact
  connectives.
- Permit warmth, surprise, and wit only where the subject earns them.

The target voice is neither a lecture transcript nor a theorem catalogue. It is
a deliberate conversation whose turns are mathematical propositions.

## 3. Diagnosis of Chapters 10, 14, 15, and 16

This diagnosis records why a separate language guide is needed. It is not a
request to make all four chapters sound identical.

### 3.1 Chapter 10: attention

Chapter 10 is the closest of the four to the desired register. Its opening names
the limitation of a fixed-dimensional recurrent state, introduces queries,
keys, and values, and explains the operation in concrete terms. Most sentences
have visible subjects and literal verbs.

The remaining risks are density and catalogue prose. Some paragraphs enumerate
several architectural variants, costs, or cross-references without pausing to
state the relation that matters. Words such as *just*, *simply*, and
*importantly* occasionally comment on a claim instead of sharpening it. The
language pass should preserve the chapter's restraint while making the center
of each dense paragraph more visible.

### 3.2 Chapter 14: reinforcement learning

Chapter 14 usually proceeds from precise contrasts and definitions. Its account
of the agent--environment interaction is stronger when it names the policy,
state distribution, trajectory, and reward directly.

Its weaker passages sometimes narrate the presentation: an object “speaks” an
interface, an experiment becomes a “payoff,” or the prose tells the reader that
an idea is the chapter's opening move. Figure captions and resource annotations
are especially prone to this compressed, knowing tone. Such language makes the
author's staging more prominent than the mechanism. Replace it with the
operation, comparison, or dependency itself.

### 3.3 Chapter 15: deep reinforcement learning

Chapter 15 has the largest gap between technical quality and linguistic ease.
Many passages contain valuable distinctions, caveats, and experimental details,
but compress them into long sentences joined by colons, semicolons, dashes, or
participial phrases. A reader must unpack several claims before learning which
one governs the paragraph.

The chapter also uses metaphor and personification too often. Promises are
“cashed,” maximization “hunts,” an attacker sits inside an algorithm, a detail
“earns its sentence,” and a baseline “promises” a return. A single literal
metaphor may help, but a chain of metaphors obscures causal structure. The
technical content is usually clearer when the policy selects an overestimated
action, the maximum introduces upward bias, or the dataset contains no sample
that could correct the estimate.

Other recurring problems are telegraphic experiment introductions, excessive
cross-references inside already dense sentences, sentence fragments used as
emphasis, and comments on the pedagogy instead of the concept. Chapter 15 needs
more syntactic breathing room, not less technical detail.

### 3.4 Chapter 16: generative adversarial networks

Chapter 16 is generally precise about claims and evidence. It distinguishes
critic overfitting from generator memorization, separates experimental findings
from theoretical conclusions, and places important limitations close to the
reported results. These are strong models for the rest of the book.

Its main language risk is cumulative qualification. A paragraph may explain a
metric, list four limitations, compare several methods, and delimit the
experiment at once. Each sentence can be correct while the paragraph remains
hard to absorb. Long captions have the same problem. Segment such material by
logical role: result, interpretation, alternative explanation, and scope.

### 3.5 General conclusion from the comparison

The problem is not lack of intelligence or detail. It is often the opposite:
too many good observations are packed into language that tries to deliver them
simultaneously. The reference books make dense subjects readable by controlling
the order and grammatical load of each sentence. This guide makes that control
explicit.

## 4. Write propositions, not commentary about propositions

The default sentence should say something verifiable about a named object.

Weak:

> The important point here is that the replay buffer changes everything.

Better:

> The replay buffer lets the learner reuse transitions collected by older
> policies.

The first sentence comments on importance. The second states the fact from which
the important consequences follow.

Weak:

> What the equation is really telling us is that the estimator is biased.

Better:

> The expectation of the estimator differs from the target by the second term,
> so the estimator is biased.

Useful sentence roles include:

- **Definition:** “The replay buffer stores transitions for later sampling.”
- **Identity:** “Expanding the return separates the immediate reward from the
  remaining discounted return.”
- **Cause:** “The maximum is convex, so zero-mean estimation noise produces
  upward bias.”
- **Contrast:** “DQN samples actions from a finite set; SAC optimizes a
  continuous action through the critic.”
- **Condition:** “If the behavior policy assigns zero probability to an action,
  importance weighting cannot recover its value.”
- **Evidence:** “Across the five seeds, clipping reduced the variance of the
  update but did not improve the median return.”
- **Qualification:** “This argument applies to exact policy evaluation; it does
  not establish convergence with nonlinear function approximation.”

If a sentence has no such role, ask whether it is needed.

## 5. Put the center of gravity in the main clause

The main clause receives the reader's attention. It should carry the principal
claim, not a stage direction or a weak conclusion.

Weak:

> It is useful to observe that, because the target network is updated less often,
> the bootstrap target changes more slowly.

Better:

> Updating the target network less often makes the bootstrap target change more
> slowly.

Weak:

> While there are many details involved in the implementation, the main thing to
> notice is that the old log-probabilities remain fixed.

Better:

> The old log-probabilities remain fixed throughout all reuse epochs.

### 5.1 Prefer an early subject and verb

Readers should not cross a long introductory phrase before discovering what the
sentence asserts.

Weak:

> In the case where a replayed transition was generated by a policy that differs
> substantially from the current actor, the target may be evaluated at actions
> that the buffer rarely contains.

Better:

> The target may evaluate actions that are rare in the buffer when the current
> actor differs substantially from the behavior policy.

### 5.2 Keep the subject and verb close

Weak:

> The estimate of the value at the next state, after clipping the action and
> taking the smaller of two independently initialized critics, enters the
> target.

Better:

> The target uses the smaller of the two critic estimates at the next state. The
> action is clipped before either critic evaluates it.

### 5.3 Put necessary conditions before the conclusion

If a condition changes the truth of the claim, state it before or immediately
after the claim.

Good:

> When rewards are bounded and \(\gamma<1\), the Bellman expectation operator is a
> contraction in the sup norm.

Also good when the conclusion deserves initial emphasis:

> The Bellman expectation operator is a contraction in the sup norm, provided
> that rewards are bounded and \(\gamma<1\).

Do not state the unrestricted claim and add the conditions several sentences
later.

## 6. Control sentence load

Sentence length is not the primary variable. **Logical load** is. A long
sentence can be clear when all of its parts modify one claim. A medium sentence
can be difficult when it contains several independent claims.

### 6.1 One load-bearing claim

This sentence is long but coherent:

> Because the target policy assigns zero probability to the action, no finite
> importance weight can reconstruct its contribution from behavior-policy
> samples.

The reason and conclusion form one claim.

This sentence is overloaded:

> The target network changes slowly, which stabilizes training, and replay
> reduces temporal correlation, while Double DQN changes the maximizing critic,
> so the implementation combines three ideas introduced in different sections.

It contains three mechanisms and a structural comment. Split it:

> DQN uses three distinct devices. A target network slows the motion of the
> bootstrap target. Replay reduces temporal correlation and reuses data. Double
> DQN separates action selection from action evaluation to reduce maximization
> bias.

### 6.2 Count conceptual pivots

A sentence usually needs revision if it pivots more than once among:

- theory and implementation;
- cause and effect;
- one method and another;
- result and caveat;
- present section and cross-reference;
- mathematical claim and pedagogical commentary.

Words such as *while*, *whereas*, *but*, *although*, *except*, *because*, and
*so* reveal pivots. They are not defects. Several of them in one sentence often
signal competing centers of gravity.

### 6.3 Split at a change of question

If the first half answers “what does the method compute?” and the second answers
“why does the experiment use this baseline?”, they belong in separate sentences
or paragraphs.

### 6.4 Do not compress for cleverness

Telegraphic prose can resemble notes rather than a textbook.

Weak:

> Two methods, three seeds each, initialization shared, all else fixed.

Better:

> We compare two methods over three seeds. Each pair of runs uses the same
> initialization, and all other settings remain fixed.

Fragments are appropriate in tables, diagrams, and occasional labels. Complete
sentences should carry the main exposition.

## 7. Order information from known to new

Good technical prose continually answers two questions: “What are we talking
about?” and “What is being added?” Put familiar material early and new or
important material late.

Suppose the previous sentence introduced a target network.

Less cohesive:

> Slow parameter changes are produced by copying the online weights every \(K\)
> steps.

More cohesive:

> The target network changes only when the online weights are copied every \(K\)
> steps.

The second version begins with the current topic and ends with the new mechanism.

### 7.1 Use the stress position

The end of a sentence naturally receives emphasis. Put the quantity or
consequence that the next sentence will develop there.

> Clipping changes the estimator by setting large probability ratios to a fixed
> boundary. This modification introduces bias.

The first sentence ends with the mechanism; the second advances to its
consequence.

### 7.2 Introduce before contrasting

Do not contrast a new object with an object that the reader has not yet met.

Weak:

> Unlike generalized advantage estimation, the one-step temporal-difference
> error uses a single bootstrap.

if GAE has not been introduced.

Better:

> The one-step temporal-difference error uses a single bootstrap. Generalized
> advantage estimation will combine such errors across several horizons.

### 7.3 Give labels after concepts when possible

> Averaging the return over all other responses removes the current response
> from its own baseline. This estimator is called leave-one-out, or RLOO.

The reader first understands the operation and then receives the name. Reverse
the order when the name is already familiar or is needed to parse the statement.

## 8. Choose concrete subjects and literal verbs

The best subject is usually the object whose behavior the sentence explains.

| Prefer | Avoid as the grammatical center |
|---|---|
| the policy, critic, estimator, matrix, sample, bound | this, it, there, the fact that |
| assigns, estimates, stores, bounds, increases | serves to, allows us to see, plays a role in |
| the measured return, the exact gradient | the story, the picture, the takeaway |

### 8.1 Use mechanism verbs

Prefer verbs that describe an operation or relation:

- computes, samples, stores, updates, reuses;
- equals, contains, depends on, converges to;
- increases, decreases, vanishes, remains;
- estimates, bounds, approximates, preserves;
- excludes, requires, assumes, implies;
- supports, contradicts, distinguishes, leaves unresolved.

### 8.2 Avoid authorial stagecraft

Avoid making the presentation itself the actor:

- “The chapter now turns to ...”
- “The next equation earns its place ...”
- “The proof has one job ...”
- “The experiment catches the bias in the act ...”
- “The table tells the story ...”

Name the dependency:

> The preceding estimator uses complete returns. Replacing them with
> bootstrapped targets permits an update after each transition.

### 8.3 Restrict personification

Some conventional shorthand is harmless: a penalty *discourages* large weights,
or a loss *penalizes* an error. Personification becomes distracting when it
attributes intention, knowledge, or aggression to a mathematical object.

Weak:

> Maximization hunts for the critic's soft spots.

Better:

> Maximization preferentially selects positive estimation errors.

Weak:

> The replay buffer owns the environment's part of the data.

Better:

> The replay buffer supplies sampled transitions, whereas the current policy
> supplies the action used in the target.

### 8.4 Repeat technical nouns without embarrassment

Technical repetition supports reference. If three successive sentences concern
the critic, write *the critic* again when a pronoun could refer to the generator,
the loss, or the optimizer. Do not substitute *judge*, *adversary*, and
*classifier* merely for variety unless those words denote distinct roles.

## 9. Use precise words, not elevated substitutes

Prefer the shortest ordinary word that preserves the technical meaning.

| Prefer | Usually avoid |
|---|---|
| use | leverage, utilize |
| show | showcase, unveil |
| give | deliver, furnish |
| change | reshape, transform the landscape of |
| begin | embark on |
| need | call for, necessitate (unless exact) |
| combine | weave together |
| remain | continue to persist |
| explain | shed light on, tell the story of |

Technical vocabulary is not jargon when it is the exact vocabulary of the
subject. Do not replace *contraction*, *support*, *bias*, or *equivariance* with a
looser phrase. Define the term and use it consistently.

### 9.1 Remove vague evaluative adverbs

Words such as *clearly*, *obviously*, *simply*, *merely*, *importantly*, and
*interestingly* require scrutiny.

- If a step is clear, the prose need not say so.
- If it is simple, state the one or two operations that make it simple.
- If a result is important, state the consequence.
- If a result is interesting, identify the expectation it violates or the
  decision it changes.

Weak:

> Importantly, the ratio is computed under the old policy.

Better:

> The denominator uses the old policy, so it remains fixed during all reuse
> epochs.

### 9.2 Keep scale words quantitative

Replace *large*, *small*, *fast*, *stable*, and *significant* when the relevant
scale is known.

> The median return rises from 0.42 to 0.68 over 20 updates.

Do not use *significant* to mean visually noticeable when no statistical test was
performed.

### 9.3 Use modal verbs deliberately

- *must* expresses necessity;
- *cannot* expresses impossibility under stated conditions;
- *will* states a consequence or a declared operation;
- *can* expresses possibility or capability;
- *may* expresses uncertainty or permission;
- *often* and *typically* require a population or well-understood pattern;
- *tends to* requires an indicated regime or empirical basis.

Do not use *can* to avoid choosing between identity, theorem, and observation.

### 9.4 Turn nominalizations back into verbs

Technical prose becomes heavy when actions are expressed as abstract nouns.

Weak:

> We perform an evaluation of the policy after the completion of each epoch.

Better:

> We evaluate the policy after each epoch.

Common repairs include:

- *make a comparison of* → *compare*;
- *carry out the computation of* → *compute*;
- *provide an explanation for* → *explain*;
- *has dependence on* → *depends on*;
- *results in an increase in* → *increases*;
- *is an approximation to* → *approximates*.

Keep a nominalization when the noun is itself the object of study. *The
distribution of the estimate* and *policy evaluation* may name established
concepts rather than hidden actions.

### 9.5 Unpack noun stacks

Several nouns placed before another noun force the reader to guess their
relations.

Weak:

> the offline policy value overestimation correction coefficient

Better:

> the coefficient that corrects value overestimation in the offline policy

or, if that is the exact meaning:

> the pessimism coefficient for offline value estimation

Hyphens help only when the compound is conventional. They cannot make an
unclear chain of concepts clear.

### 9.6 Prefer precise detail to bureaucratic abstraction

Plain language need not be vague. Compare:

> Performance degradation may occur under distributional change.

with:

> When the test images contain a new background, accuracy falls from 91% to
> 63%.

The second sentence is livelier because the nouns and measurement are specific,
not because its tone is more animated.

## 10. Make reference explicit

Ambiguous reference is a major source of technically correct but tiring prose.

### 10.1 Attach a noun to demonstratives

Weak:

> This reduces variance.

Better:

> This subtraction reduces variance.

Weak:

> This is unavailable offline.

Better:

> This corrective interaction is unavailable offline.

The noun tells the reader which preceding proposition is being carried forward.

### 10.2 Audit every pronoun with two possible antecedents

Weak:

> The actor updates the critic with its samples.

Whose samples? Better:

> The actor supplies fresh actions for states sampled from the critic's replay
> buffer.

### 10.3 Prefer lexical continuity to synonym chains

A paragraph about distribution shift should keep the phrase *distribution
shift* or name its two distributions. Alternating among *mismatch*, *gap*,
*departure*, and *drift* may falsely imply different concepts.

### 10.4 Use equation references as nouns, not navigation commands

Weak:

> Looking back at Equation 3, we can see that ...

Better:

> Equation 3 averages over actions drawn from the current policy.

The equation is the subject; the reader is not instructed to perform page
navigation before learning why.

## 11. Connect sentences by the actual logical relation

A connective is a promise about logic. Use it only when the relation holds.

### 11.1 Consequence

Use *therefore*, *thus*, *hence*, or *so* when the second statement follows from
the first.

> The behavior policy never chooses the fifth action. Therefore the dataset
> contains no direct evidence about its value.

*So* is slightly more conversational; *therefore* and *thus* are more formal.
Vary them by register, not to avoid repetition.

### 11.2 Contrast or qualification

Use *but*, *however*, *although*, or *whereas* when two claims pull in different
directions.

> The estimator is unbiased, but its variance may be infinite.

Place *however* where its scope is clear. Avoid inserting it deep inside a long
sentence.

### 11.3 Parallel fact

Do not use *moreover* or *furthermore* merely because another sentence follows.
The new fact should genuinely reinforce the same conclusion.

### 11.4 Example

Use *for example* for an instance of the preceding class. Use *in particular*
for a special case that receives extra attention. Use *specifically* to make a
general statement more exact.

### 11.5 Reversal

Use *conversely* only when the logical direction is reversed. It does not mean
“on the other hand.”

### 11.6 Omitted connectives

Not every relation needs an adverb. Often the nouns provide enough continuity:

> The critic estimates the value of each action. The actor selects actions with
> high estimated value.

Explicit connectives are most useful when a reader might otherwise infer the
wrong relation.

## 12. Build coherent paragraphs

A paragraph should be readable as a small proof: it establishes one local
conclusion through an ordered sequence of sentences.

### 12.1 Default paragraph movement

1. **Anchor:** name the object or question.
2. **Develop:** give the definition, mechanism, derivation, or evidence.
3. **Resolve:** state the consequence, limitation, or transition that follows.

Example:

> A replay buffer changes the distribution of training data. Consecutive
> transitions enter the buffer in temporal order, but a minibatch samples their
> indices independently. The update therefore mixes transitions from different
> episodes and different stages of training. This sampling reduces temporal
> correlation, although it does not make the transitions identically
> distributed.

Every sentence concerns the sampling distribution. The final sentence states
both the consequence and its limit.

### 12.2 Maintain one paragraph subject

Paragraphs become difficult when their subject changes silently from an
estimator, to an implementation, to a historical paper, to an experiment. Start
a new paragraph when the governing question changes.

### 12.3 Give enumerations a governing sentence

Before listing limitations, state what they limit.

> The reported FID values do not support a model-independent ranking for four
> reasons.

Then give the four reasons in a list or in four visibly parallel sentences.

### 12.4 Do not hide the conclusion in the middle

If the paragraph establishes that a result is an ablation of a recipe rather
than a test of a single component, say so at the end. The last sentence should
leave the reader with the paragraph's durable result.

### 12.5 Separate result, interpretation, and limitation

These three moves often deserve three sentences:

> The clipped agent has a lower update variance in all five seeds. Clipping
> removes the largest importance ratios, which is consistent with this reduction.
> The experiment does not show that clipping improves final return.

Combining them into one sentence makes the caveat sound subordinate even when it
is essential.

## 13. Create cadence without theatrics

Readable prose varies sentence length according to function.

- Use short sentences for definitions, boundary cases, and conclusions.
- Use medium sentences for most causal and comparative explanation.
- Use long sentences when several conditions jointly support one claim.
- Follow a dense equation or sentence with a shorter interpretation.

A useful cadence is **state, explain, conclude**:

> The critic target uses an action drawn from the current policy. The replay
> buffer supplies the state and transition, but it need not contain that action.
> No importance ratio is required for this target.

The final short sentence works because the preceding sentences establish it. A
short sentence without that support sounds like a slogan.

### 13.1 Avoid serial punchlines

Do not build rhythm from fragments such as:

> No new data. No correction. The result: collapse.

Write the causal chain:

> Because the offline learner receives no new data, an overestimated action can
> remain uncorrected and dominate the learned policy.

### 13.2 Read for breath and memory

When reading aloud, mark every place where a sentence asks the reader to retain
one unresolved clause while processing another. Two such places may be
manageable. Three usually require division.

### 13.3 Use parallel syntax for parallel ideas

Parallel construction gives technical prose both clarity and rhythm.

Weak:

> The critic estimates values, action selection is done by the actor, and the
> buffer is where transitions are stored.

Better:

> The critic estimates values, the actor selects actions, and the buffer stores
> transitions.

Parallelism is especially useful in comparisons, theorem assumptions, figure
descriptions, and lists of limitations. It lets the reader spend attention on
the differences in content rather than differences in grammar.

### 13.4 Vary openings by changing the thought

A paragraph in which every sentence begins with *the method* sounds monotonous.
Do not fix it by cycling through synonyms. Change the legitimate point of
departure:

> The method reuses each batch for ten epochs. During these epochs, the old
> probabilities remain fixed. Large policy changes therefore appear as large
> ratios.

The openings move from object, to interval, to consequence. Each change follows
the argument.

### 13.5 Use controlled emphasis

Three devices provide emphasis without drama:

- place the important new information at the end of a sentence;
- give a conclusion its own short sentence after establishing it;
- repeat the key noun in a parallel contrast.

Typography, fragments, and words such as *crucially* are weaker substitutes for
these grammatical devices.

## 14. Use questions as instruments of reasoning

MacKay's questions work because they define a problem with an answer. Follow
that standard.

Good:

> How can a decoder recover the message when the channel flips some bits?

The next paragraphs define the channel, propose redundancy, and derive a
decoder.

Good:

> Why does the maximum of unbiased estimates overestimate the largest true
> value?

The next sentence or equation answers the question through convexity or a small
example.

Weak:

> So what does all this mean?

Weak:

> Isn't this remarkable?

Weak:

> But can we do better?

unless *better* has already been defined by a metric or constraint.

### 14.1 State the rules before inviting an answer

A question is useful only if the reader knows the variables, objective, and
constraints. If retransmission is forbidden, say so before asking for an
error-correcting scheme.

### 14.2 Answer promptly

Do not pose a question at the start of a section and defer its answer through
several unrelated subsections. Restate the precise question where the answer
begins if necessary.

### 14.3 Do not simulate the reader's thoughts

Avoid “you may wonder,” “one might ask,” and “the natural question is” when the
question can simply be asked or stated as a problem.

## 15. Use first and second person with purpose

### 15.1 “We” marks a shared operation

Use *we* when author and reader perform a local mathematical action:

- “We condition on the next state.”
- “We compare the two methods at equal sample size.”
- “We first hold the critic fixed.”

Avoid using *we* to manufacture agreement:

- “We clearly see ...”
- “We know this must be better ...”
- “We should be impressed ...”

### 15.2 The object is often a better subject

> Substituting the optimal critic gives the Jensen--Shannon divergence.

is usually tighter than

> We can now see that if we substitute the optimal critic, we get the
> Jensen--Shannon divergence.

### 15.3 “You” belongs mainly to action

Use *you* in exercises, code instructions, or a concrete thought experiment.
Prefer impersonal statements for general mechanisms.

> Increasing the batch size reduces the Monte Carlo variance.

rather than

> If you increase the batch size, you will see less variance.

MacKay sometimes addresses the reader playfully. Such moments work because they
are rare. They should not become the book's default relationship with the
reader.

## 16. Integrate mathematics into sentences

An equation is part of the grammar. The surrounding prose should tell the
reader why it appears and what follows from it.

### 16.1 Before the equation: state its task

> Conditioning on the first action separates its immediate reward from the
> remaining return:

Then display the Bellman identity.

Avoid:

> We have the following important equation:

### 16.2 After the equation: interpret a visible feature

> The factor \(\gamma\) multiplies every difference between two input value
> functions. The operator is therefore a contraction when \(\gamma<1\).

Do not paraphrase every symbol. Explain the feature that advances the argument.

### 16.3 Keep equations grammatical

If prose introduces a displayed equation with a colon, the preceding clause
must be complete. If the equation completes the sentence, punctuate it as part
of that sentence.

### 16.4 Name the operation before performing it

> Taking expectations over the next state gives

is better than displaying a new expectation and explaining afterward that a
conditional expectation was taken.

### 16.5 Distinguish equality, approximation, and implementation

Use exact verbs:

- “equals” for an identity;
- “converges to” for a limit;
- “approximates” for an approximation;
- “estimates” for a random estimate;
- “implements” for code realizing an operation;
- “is motivated by” when no derivation establishes equivalence.

Smooth prose must never erase these distinctions.

### 16.6 Give assumptions their own sentence when they matter twice

> Assume that the two source messages have equal prior probability and that the
> channel flips each bit independently. Under these assumptions, maximizing the
> posterior is equivalent to majority voting.

This pattern, common in strong mathematical exposition, prevents assumptions
from disappearing inside a derivation.

## 17. Write definitions for use

A definition should leave the reader able to recognize and use the object.

### 17.1 Name, formula, interpretation, boundary

A full definition usually needs four components, not necessarily four sentences:

1. the name and domain of the object;
2. its mathematical definition;
3. the operational or geometric interpretation needed here;
4. conventions or boundary cases.

Example:

> The entropy of a discrete random variable \(X\) with mass function \(p\) is
> \[
> H(X)=-\sum_x p(x)\log_2 p(x).
> \]
> It is the expected value of \(-\log_2 p(X)\) and is measured in bits. We use
> the convention \(0\log 0=0\).

### 17.2 Do not define through praise

Weak:

> A powerful and elegant way to handle uncertainty is entropy.

Better:

> Entropy measures the average information content of a draw from a probability
> distribution.

### 17.3 Repeat the defined term

After defining mutual information, use *mutual information* in the first
interpretive sentence. A bare *it* weakens the connection between name and
meaning.

## 18. Write derivations as justified moves

A derivation should alternate symbolic progress with only the prose needed to
identify its logic.

### 18.1 State the purpose

> We derive the gradient in a form that can be estimated from sampled
> trajectories.

This tells the reader which transformations matter.

### 18.2 Name nontrivial moves

> The second equality applies the log-derivative identity. The third removes the
> transition terms because their probabilities do not depend on \(\theta\).

Do not say only “after some algebra.” If the algebra is routine, omit it or show
it compactly; if it contains the idea, explain the idea.

### 18.3 Close the derivation at the promised destination

> The final expression is an expectation under the current policy, so sampled
> trajectories provide a Monte Carlo estimator of the gradient.

The conclusion should answer the purpose stated at the start.

### 18.4 Avoid retrospectively changing the question

If a derivation begins by promising an unbiased estimator, do not end by saying
only that the expression is computationally convenient. State whether the
promise was met and under which assumptions.

## 19. Explain examples in complete causal steps

Both reference books use examples to make a definition operational. A good
example has language at four points.

1. **Setup:** specify the object and its parameters.
2. **Prediction:** say what the current theory predicts.
3. **Calculation or observation:** carry out the test.
4. **Interpretation:** connect the outcome back to the general claim.

Weak:

> Consider eight outcomes. The entropy is two bits. This shows how entropy works.

Better:

> Suppose eight outcomes have unequal probabilities. A fixed-length code uses
> three bits for every outcome. Assigning shorter codewords to the more probable
> outcomes reduces the average length to two bits, which equals the entropy of
> this distribution.

### 19.1 Preserve the nouns of the setup

If an example begins with horses and codewords, do not switch midway to generic
items and representations unless the abstraction is the point.

### 19.2 Explain why the example is representative

> The two-state example isolates maximization bias because both actions have the
> same true value; any positive gap therefore comes from estimation noise.

This sentence earns the example's place without saying that it is illuminating.

## 20. Report experiments as evidence

Experimental prose should separate design, observation, inference, and scope.

### 20.1 Design

State what varies and what is controlled.

> We compare the twin-critic and single-critic variants over three seeds. Each
> pair begins from the same actor initialization and uses the same replay data.

### 20.2 Observation

Report what the run measured.

> The twin-critic variant reduces the median value overestimate from 0.31 to
> 0.08. Median return is similar in the two variants.

### 20.3 Inference

Match the inference to the design.

> In this task, the second critic improves value calibration without a measurable
> return advantage.

### 20.4 Scope

State what the experiment cannot establish.

> Because the comparison changes only the number of critics, it does not isolate
> the effects of entropy regularization or target smoothing.

### 20.5 Avoid verdict language

Do not say that an experiment “proves,” “settles,” “vindicates,” or “catches” a
mechanism unless the design warrants that claim. Prefer:

- measures;
- demonstrates in this setting;
- is consistent with;
- supports;
- does not distinguish;
- leaves open.

### 20.6 Make captions readable independently

Captions may be denser than body prose, but they still need sentence boundaries.
A caption that describes panels, reports measurements, gives causal
interpretation, and lists caveats should be divided into four sentences in that
order.

## 21. Express contrasts symmetrically

Contrasts are easiest to understand when both sides use parallel grammar.

Weak:

> DQN has a replay buffer, whereas on-policy data must be current.

Better:

> DQN updates from replayed transitions, whereas PPO updates from a recently
> collected on-policy batch.

### 21.1 Name the axis of comparison

> The methods differ in data reuse. DQN may reuse a transition many times; a
> basic policy-gradient update uses its on-policy trajectory once.

### 21.2 Avoid reflexive “not X but Y” constructions

Use this form only when correcting a likely misconception.

Justified:

> The held-out split tests critic overfitting, not generator memorization.

The distinction changes the interpretation of the measurement.

Unjustified:

> This is not merely an optimization method but a new way to understand
> learning.

The second half is broader but not more precise.

### 21.3 Do not make every contrast absolute

Prefer *differs in*, *places more weight on*, *uses under these conditions*, and
*does not require* when the methods overlap in other respects.

## 22. Qualify claims without burying them

Qualifications are part of the result, not apologies appended to it.

### 22.1 Put the strongest safe claim first

> On this finite MDP, value iteration converges geometrically to the unique fixed
> point.

This is stronger and clearer than making an unrestricted statement followed by
a list of exceptions.

### 22.2 Use one qualification per logical dimension

Separate assumptions about theory, scale, and evidence:

> The contraction proof assumes a finite discounted MDP. The experiment uses a
> tabular lake with 16 states. Neither result establishes convergence with a
> nonlinear critic.

### 22.3 Avoid caveat avalanches

Four qualifications in one sentence become unreadable and may conceal which
claim each one limits. Use a short list when several limitations are peers.

### 22.4 Prefer exact limits

Weak:

> The result should be interpreted with caution.

Better:

> The result compares two losses at one dataset size and does not determine how
> their ordering changes with model capacity.

## 23. Use punctuation to reveal structure

Punctuation cannot rescue a sentence whose logic has not been ordered.

### 23.1 Commas

Use commas for short introductory phrases, nonrestrictive modifiers, and items
in a list. Do not join two independent claims with a comma.

### 23.2 Colons

A colon introduces material promised by a complete clause:

> The update has two sources of randomness: trajectory sampling and action
> sampling.

Avoid repeated colons as a device for dramatic reveal.

### 23.3 Semicolons

A semicolon joins two independent clauses whose relation is already clear.

> The target network is fixed during an update; the online network is not.

If either clause contains another contrast or qualification, use two sentences.

### 23.4 Dashes

Use an em dash for a genuine interruption or a compact appositive. One pair in a
paragraph is usually enough. Repeated dashes produce a breathless, editorial
voice and blur the hierarchy of clauses.

### 23.5 Parentheses

Put optional examples, units, or short conventions in parentheses. Assumptions,
causal steps, and limitations belong in the main syntax.

### 23.6 The colon-fragment pattern

Avoid:

> The consequence: unstable learning.

Prefer:

> The moving target can make learning unstable.

The verb states the relation that the colon only implies.

## 24. Use metaphor and wit under strict conditions

Good textbooks are enjoyable, but enjoyment comes mainly from understanding.
Metaphor is useful when it maps a difficult relation onto a familiar structure
and the mapping remains accurate.

Useful:

> A bottleneck restricts how much information can pass through the hidden state.

Here *bottleneck* names a standard structural constraint.

Risky:

> The optimizer hunts for soft spots and cashes the critic's promissory notes.

The reader must decode two unrelated metaphors before recovering the mechanism.

Use a metaphor only if all four tests pass:

1. It makes a specific relation easier to picture.
2. Its relevant correspondence can be stated exactly.
3. It does not attribute unavailable knowledge or intention.
4. It is not followed by a second metaphor for the same mechanism.

Wit should be brief, local, and expendable. If removing a joke damages the
argument, the argument was relying on the joke to supply a logical step.

### 24.1 Plain language is not flat language

Clarity does not require a uniform administrative tone. The reference books
derive much of their character from choices that remain fully literal:

- a concrete problem appears before its abstraction;
- an honest question creates forward motion;
- a small example produces a result the reader can check;
- a precise contrast replaces a vague generalization;
- a quantified cost follows an apparent gain;
- an exception arrives exactly where the reader might overgeneralize;
- a concise final sentence records what has been learned.

These moves make prose enjoyable because they create discovery. They are more
durable than decorative language.

### 24.2 Permit the author's judgment when it is informative

An author need not pretend to have no judgment. The prose may say that an
assumption is restrictive, an implementation is fragile, a proof is economical,
or an example is representative when the following words explain why.

Good:

> This bound is conservative: at the tested discount factor it exceeds the
> largest possible return by a factor of twelve.

Good:

> The example is useful because its two actions have equal true value, so any
> measured preference comes from estimation error.

The judgment is earned by the clause that follows it.

### 24.3 Let occasional elegance remain visible

Some mathematical developments are genuinely elegant. Show the source of that
elegance instead of merely naming it.

> The same chain rule applies to entropy, conditional entropy, and mutual
> information. One algebraic pattern therefore organizes all three quantities.

This sentence communicates why the relationship is satisfying. “This elegant
identity is remarkable” does not.

## 25. Avoid generated-prose mannerisms

The structural guide lists common artifacts. The language pass should also look
for the grammatical habits that create them.

### 25.1 Announced importance

Pattern:

> One fact deserves emphasis: ...

Revision:

State the fact, then state its consequence.

### 25.2 Exposition as performance

Pattern:

> Now the experiment, fifteen times over.

Revision:

> We repeat the experiment on 15 independently collected datasets because the
> result varies substantially with the dataset.

### 25.3 Compressed knowingness

Pattern:

> Same critic, new target, everything else untouched.

Revision:

> The ablation changes only the target; it uses the same critic architecture and
> training schedule.

### 25.4 Slogan followed by explanation

Pattern:

> The max is not neutral. It hunts positive errors.

Revision:

> Taking a maximum preferentially selects estimates with positive errors and
> therefore introduces upward bias.

### 25.5 Explanation followed by slogan

Pattern:

> The policy changes the state distribution and therefore the data used for its
> next update. Feedback changes everything.

Revision:

Delete the final sentence. The specific claim is stronger.

### 25.6 Excessive “now”

*Now* is useful when time or derivation order matters. It is not a universal
transition. Replace it with the dependency:

> The exact expectation requires the transition kernel. With sampled transitions
> only, we use a temporal-difference update.

### 25.7 Decorative specificity

Exact counts and vivid nouns are useful only when they matter. “Three seeds,
caller-owned agents, one ledger” sounds concrete but may hide the experimental
logic. State which factor varies, why the seeds are paired, and what is measured.

## 26. Reliable sentence patterns

These patterns are scaffolds, not phrases to repeat mechanically.

### Definition

> A/An \(X\) is ...

> We define \(X\) as ...

> The quantity \(X\) measures ...

### Dependency

> Computing \(X\) requires \(Y\).

> The preceding argument assumes \(X\); removing this assumption requires \(Y\).

### Cause

> Because \(X\), \(Y\).

> \(X\) causes \(Y\) by ...

### Consequence

> Therefore \(X\).

> This bound implies \(X\) when ...

### Contrast

> Both methods \(X\). They differ in \(Y\).

> Method A uses \(X\), whereas method B uses \(Y\).

### Qualification

> This conclusion holds when ...

> The argument does not cover ...

> The experiment distinguishes \(X\) from \(Y\), but not \(Y\) from \(Z\).

### Equation introduction

> Conditioning on \(X\) gives ...

> Substituting \(X\) into \(Y\) yields ...

### Equation interpretation

> The first term ...; the second term ...

> The expression vanishes exactly when ...

### Experiment

> We vary \(X\) while holding \(Y\) fixed.

> Across \(n\) seeds, the measured \(X\) ...

> This result supports \(X\) in the tested setting; it does not establish \(Y\).

### Transition

> The remaining difficulty is ...

> Exact evaluation requires ...; sampled evaluation replaces it with ...

Use these forms to expose logic during revision. Vary the final prose only when
variation improves meaning or rhythm.

## 27. Paragraph rewrites modeled on recurring chapter problems

The following examples are constructed from recurring patterns in the reviewed
chapters. They illustrate the language change without prescribing chapter
content.

### 27.1 From metaphor to mechanism

Before:

> Offline maximization hunts the errors that point upward, and nothing in the
> frozen dataset comes to the rescue.

After:

> The maximum preferentially selects overestimated actions. Because the dataset
> is fixed, the learner cannot test those actions and correct their estimates
> with new transitions.

### 27.2 From stagecraft to dependency

Before:

> One bookkeeping detail earns its own sentence before the algorithm arrives.

After:

> The log-density sums both the Gaussian and Jacobian terms over the action
> dimension before averaging over the batch. Applying different reductions to
> the two terms gives an incorrect density when the action has more than one
> dimension.

### 27.3 From compressed experiment prose to design

Before:

> Two arms, three seeds each, everything else identical.

After:

> We compare the twin-critic and single-critic variants over three paired seeds.
> Within each pair, the runs share their initialization, data budget, and
> optimization settings.

### 27.4 From caveat pile to layered qualification

Before:

> The metric ranks A above B, although it depends on the feature network and the
> sample size and preprocessing can move it and the experiment uses only one
> dataset, so the values are not general.

After:

> With this feature network, method A receives a lower score than method B. The
> ordering persists across the reported seeds. The numerical values depend on
> the feature network, sample size, and preprocessing, so they should not be
> compared directly with scores from a different evaluation pipeline.

### 27.5 From vague importance to consequence

Before:

> Importantly, attention is permutation equivariant.

After:

> Permuting the input positions permutes the outputs in the same way. Attention
> alone therefore cannot distinguish two sequences that differ only in order.

### 27.6 From encyclopedic list to governed list

Before:

> Alternatives include sparse attention, kernel attention, recurrent attention,
> chunking, and memory-efficient exact attention.

After:

> Attention methods reduce cost in three different ways. Sparse methods compute
> only selected query--key pairs. Kernel and recurrent methods factor the
> interaction. Memory-efficient exact methods compute every pair but avoid
> storing the full score matrix.

## 28. The language revision workflow

Perform this pass after the chapter's technical order is sound. Revising language
cannot repair a missing premise or a misplaced section.

### Pass 1: Extract the propositions

For each paragraph, write a private one-line statement of what it establishes.
If no such statement exists, the paragraph may be filler. If there are three,
split or reorganize it.

### Pass 2: Underline subjects and verbs

Check whether the principal subjects are the mathematical and computational
objects. Replace empty subjects and authorial stage directions. Shorten the
distance from subject to verb.

### Pass 3: Mark logical relations

Label each sentence boundary as continuation, cause, consequence, contrast,
example, or qualification. Add a connective only where the relation is not
otherwise clear. Replace any connective that promises the wrong relation.

### Pass 4: Check known-to-new order

Each sentence should begin from the current topic and end with the information
that advances the argument. Move newly introduced terms out of subordinate
clauses when they deserve emphasis.

### Pass 5: Reduce sentence load

Circle every coordinating conjunction, subordinate clause, parenthesis, dash,
and semicolon. Split sentences with more than one conceptual pivot. Preserve a
long sentence only when all its parts support one claim.

### Pass 6: Restore explicit reference

Replace ambiguous *this*, *that*, *it*, *they*, and *which* with a noun or a
clearer clause. Repeat technical terms where synonyms would weaken continuity.

### Pass 7: Make the language literal

Replace performance metaphors, personification, promotional verbs, and vague
importance markers with the mechanism or consequence. Keep a metaphor only if
it passes the four tests in Section 24.

### Pass 8: Audit claim strength

Mark every *must*, *will*, *can*, *often*, *typically*, and *significant*. Check
the source of necessity, possibility, frequency, or evidence. Move each
qualification beside the claim it limits.

### Pass 9: Repair the math--prose interface

For every displayed equation, read the preceding and following sentence without
the display. The first should state the operation or purpose; the second should
interpret the feature needed next.

### Pass 10: Read aloud

Listen for:

- a delayed main verb;
- several unresolved clauses;
- repeated sentence openings;
- serial fragments;
- accidental rhyme or alliteration;
- too many dashes or colons;
- a paragraph with no place to pause;
- a conclusion that sounds advertised rather than established.

### Pass 11: Perform the deletion test

Delete every sentence that comments only on importance, elegance, surprise, or
the author's presentation. Restore it only if it contains a necessary technical
claim that cannot be stated directly.

## 29. Review rubric for language

Score each category from 0 to 2.

### 29.1 Grammatical transparency

- **0:** Main claims are routinely buried in long or fragmentary syntax.
- **1:** Most sentences are parseable, but several have competing main ideas.
- **2:** Main clauses consistently carry main claims, and qualifications have
  clear scope.

### 29.2 Diction

- **0:** Frequent stagecraft, personification, promotional language, or vague
  evaluation.
- **1:** Mostly literal language with noticeable residues.
- **2:** Concrete nouns and exact verbs dominate; figurative language is rare and
  useful.

### 29.3 Cohesion

- **0:** Pronouns, synonyms, and transitions obscure the chain of reference.
- **1:** The topic is usually recoverable with occasional backtracking.
- **2:** Each sentence begins from a clear topic and prepares the next one.

### 29.4 Cadence

- **0:** Prose is uniformly dense, uniformly choppy, or built from punchlines.
- **1:** Sentence lengths vary but some paragraphs remain breathless.
- **2:** Length follows function, and dense reasoning is followed by concise
  interpretation.

### 29.5 Mathematical integration

- **0:** Equations appear without purpose or interpretation.
- **1:** Most equations are introduced, but some symbolic moves remain
  unexplained.
- **2:** Prose names each important operation and states the consequence of each
  displayed result.

### 29.6 Intellectual precision

- **0:** The prose blurs theorem, heuristic, evidence, and speculation.
- **1:** Claims are mostly calibrated, with delayed or vague caveats.
- **2:** Modal verbs, evidence verbs, and qualifications match the strength and
  scope of each result.

### 29.7 Reader relationship

- **0:** The prose lectures, flatters, dramatizes, or simulates the reader's
  reactions.
- **1:** The tone is professional but sometimes impersonal or self-conscious.
- **2:** The prose is direct, respectful, and conversational only where a shared
  question or operation warrants it.

A chapter is ready for a final copyedit only when every category scores 2. A
total score can hide a serious local defect, so do not average away a zero.

## 30. Compact checklist for authors and agents

Before accepting a paragraph, ask:

- What single conclusion does this paragraph establish?
- Does its first sentence identify the topic?
- Does each sentence have a concrete subject and a visible main verb?
- Does each sentence make one load-bearing claim?
- Are conditions and qualifications adjacent to the claims they limit?
- Does each pronoun have one unmistakable antecedent?
- Do repeated technical nouns preserve continuity?
- Does every connective name the actual logical relation?
- Does the paragraph move from known information to new information?
- Are result, interpretation, and limitation distinguishable?
- Does an equation have a stated purpose and a useful interpretation?
- Can any metaphor, importance marker, or stage direction be replaced by the
  mechanism?
- Does sentence length vary by function rather than for dramatic effect?
- Would deleting the final sentence remove information, or only emphasis?
- Can the paragraph be read once without backtracking?

Before accepting a chapter, ask:

- Are genuine questions answered promptly?
- Are examples concrete enough to operate the definitions?
- Are comparisons made on explicit, parallel axes?
- Are experiments described as designs and evidence rather than performances?
- Are dense qualifications divided by logical role?
- Does the tone remain stable across theory, code, captions, resources, and
  exercises?
- Is the prose enjoyable because the argument becomes visible?

## 31. Definition of done

The language is ready when a technically prepared reader can move through the
chapter without stopping to decode the author's phrasing. The reader should know
at every point:

- which object is under discussion;
- what claim is being made about it;
- why that claim follows;
- under which conditions it holds;
- how the next sentence advances the argument.

The ideal prose does not call attention to its own polish. Its quality appears
as trust: definitions arrive when needed, equations have reasons, examples answer
real questions, qualifications are candid, and the language never competes with
the ideas.

## References used for the language model

- David J. C. MacKay, [*Information Theory, Inference, and Learning
  Algorithms*](https://www.inference.org.uk/itprnn/book.pdf), especially the
  opening communication problem and the introductory treatment of probability
  and inference.
- Thomas M. Cover and Joy A. Thomas, [*Elements of Information
  Theory*](https://cs-114.org/wp-content/uploads/2015/01/Elements_of_Information_Theory_Elements.pdf),
  especially the preview and the opening development of entropy, relative
  entropy, and mutual information.
